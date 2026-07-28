from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import Any

from pocketinfer.adapters.base import ModelAdapter
from pocketinfer.adapters.common import candidate_values, dtype_bytes
from pocketinfer.models import Candidate, Estimate, FidelityPolicy


class KimiK3Adapter(ModelAdapter):
    name = "kimi-k3"
    _FUSED_KDA_HEADS = (12, 24, 48, 96)

    def supports(self, config: dict[str, Any]) -> bool:
        text = config.get("text_config", {})
        return config.get("model_type") == "kimi_k3" and (
            text.get("model_type") == "kimi_linear"
        )

    def candidates(
        self,
        config: dict[str, Any],
        policy: FidelityPolicy,
    ) -> Iterable[Candidate]:
        text = config["text_config"]
        source_layers = int(text["num_hidden_layers"])
        source_heads = int(text["num_attention_heads"])
        source_experts = int(text["num_experts"])
        top_k = int(text["num_experts_per_token"])

        layers = {4}
        layers.update(range(5, source_layers + 1, 4))
        layers.add(source_layers)
        head_seeds = tuple(
            head for head in self._FUSED_KDA_HEADS if head <= source_heads
        )
        heads = candidate_values(
            source_heads,
            min(head_seeds),
            head_seeds,
        )
        local_experts = (
            source_experts // policy.reference_ep
            if source_experts % policy.reference_ep == 0
            else top_k
        )
        minimum_experts = top_k * 2 if policy.profile == "balanced" else top_k
        experts = candidate_values(
            source_experts,
            minimum_experts,
            (top_k, top_k * 2, 32, 48, 56, 64, 112, local_experts),
            multiple=8,
        )

        for layer_count in sorted(layers):
            for head_count in heads:
                for expert_count in experts:
                    generated = deepcopy(config)
                    generated_text = generated["text_config"]
                    self._apply_dimensions(
                        generated_text,
                        layer_count,
                        head_count,
                        expert_count,
                    )
                    estimate = self._estimate(generated)
                    yield Candidate(
                        config=generated,
                        dimensions={
                            "layers": layer_count,
                            "attention_heads": head_count,
                            "experts": expert_count,
                            "top_k": top_k,
                        },
                        estimate=estimate,
                        score=self._score(
                            text,
                            layer_count,
                            head_count,
                            expert_count,
                            policy,
                        ),
                        preserved=self._preserved(generated_text),
                        warnings=self._warnings(
                            text,
                            layer_count,
                            head_count,
                            expert_count,
                            policy,
                        ),
                    )

    @staticmethod
    def _apply_dimensions(
        text: dict[str, Any],
        layers: int,
        heads: int,
        experts: int,
    ) -> None:
        full_layers = [layer for layer in range(1, layers + 1) if layer % 4 == 0]
        if layers not in full_layers:
            full_layers.append(layers)
        kda_layers = [
            layer for layer in range(1, layers + 1) if layer not in full_layers
        ]
        text["num_hidden_layers"] = layers
        text["num_attention_heads"] = heads
        text["num_key_value_heads"] = heads
        text["num_experts"] = experts
        text["linear_attn_config"]["num_heads"] = heads
        text["linear_attn_config"]["full_attn_layers"] = full_layers
        text["linear_attn_config"]["kda_layers"] = kda_layers

    @staticmethod
    def _routed_weight_bytes(text: dict[str, Any]) -> float:
        quant = text.get("quantization_config", {})
        format_name = str(quant.get("format", "")).lower()
        if "mxfp4" not in format_name and "nvfp4" not in format_name:
            return float(dtype_bytes(text))
        groups = quant.get("config_groups", {})
        first_group = next(iter(groups.values()), {})
        default_group_size = 16 if "nvfp4" in format_name else 32
        group_size = first_group.get("weights", {}).get(
            "group_size",
            default_group_size,
        )
        return 0.5 + (1 / int(group_size))

    @classmethod
    def _estimate(cls, config: dict[str, Any]) -> Estimate:
        text = config["text_config"]
        h = int(text["hidden_size"])
        vocab = int(text["vocab_size"])
        layers = int(text["num_hidden_layers"])
        heads = int(text["num_attention_heads"])
        experts = int(text["num_experts"])
        top = int(text["num_experts_per_token"])
        dense_i = int(text["intermediate_size"])
        moe_i = int(text["moe_intermediate_size"])
        routed_h = int(text["routed_expert_hidden_size"])
        shared = int(text["num_shared_experts"])
        q_rank = int(text["q_lora_rank"])
        kv_rank = int(text["kv_lora_rank"])
        nope = int(text["qk_nope_head_dim"])
        rope = int(text["qk_rope_head_dim"])
        value = int(text["v_head_dim"])
        kda = text["linear_attn_config"]
        head_dim = int(kda["head_dim"])
        conv = int(kda["short_conv_kernel_size"])
        projection = heads * head_dim
        kda_layers = len(kda["kda_layers"])
        mla_layers = layers - kda_layers

        embedding = 2 * vocab * h
        kda_params = (
            h * (4 * projection + head_dim + heads)
            + head_dim * projection
            + 3 * projection * conv
            + projection * h
        )
        mla_params = (
            h * (q_rank + kv_rank + rope)
            + q_rank * heads * (nope + rope)
            + kv_rank * heads * (nope + value)
            + 2 * h * heads * value
        )
        dense_params = 3 * h * dense_i
        moe_layers = max(layers - int(text["first_k_dense_replace"]), 0)
        shared_params = 3 * h * (moe_i * shared)
        latent_params = 2 * h * routed_h + routed_h
        router_params = h * experts + experts
        routed_per_layer = 3 * routed_h * moe_i * experts
        routed_params = moe_layers * routed_per_layer
        non_routed = (
            embedding
            + kda_layers * kda_params
            + mla_layers * mla_params
            + dense_params
            + moe_layers * (shared_params + latent_params + router_params)
        )
        vision_params = cls._vision_params(config)
        non_routed += vision_params
        total = non_routed + routed_params
        weight_bytes = int(
            non_routed * dtype_bytes(text)
            + routed_params * cls._routed_weight_bytes(text)
        )
        mla_kv = mla_layers * (kv_rank + rope) * dtype_bytes(text)
        kda_state = kda_layers * (
            3 * projection * max(conv - 1, 0) * dtype_bytes(text)
            + heads * head_dim * head_dim * 4
        )
        return Estimate(
            parameter_count=total,
            weight_bytes=weight_bytes,
            kv_bytes_per_token=mla_kv,
            fixed_cache_bytes=kda_state,
            notes=(
                f"top-{top} routed experts retained",
                "minimum cache estimate covers one max-length request",
                "runtime repacking and CUDA workspaces are excluded",
            ),
        )

    @staticmethod
    def _vision_params(config: dict[str, Any]) -> int:
        vision = config.get("vision_config")
        if not isinstance(vision, dict):
            return 0
        hidden = int(vision.get("vt_hidden_size", 1024))
        intermediate = int(vision.get("vt_intermediate_size", 4096))
        qkv = int(vision.get("qkv_hidden_size", hidden))
        layers = int(vision.get("vt_num_hidden_layers", 0))
        text_hidden = int(
            vision.get("text_hidden_size", config["text_config"]["hidden_size"])
        )
        merge = vision.get("merge_kernel_size", [2, 2])
        merged_hidden = hidden * int(merge[0]) * int(merge[1])
        block = 3 * hidden * qkv + qkv * hidden + 2 * hidden * intermediate
        patch = 3 * hidden * int(vision.get("patch_size", 14)) ** 2
        position = (
            int(vision.get("init_pos_emb_height", 64))
            * int(vision.get("init_pos_emb_width", 64))
            * hidden
        )
        projector = merged_hidden**2 + merged_hidden * text_hidden
        return layers * block + patch + position + projector

    @staticmethod
    def _score(
        source: dict[str, Any],
        layers: int,
        heads: int,
        experts: int,
        policy: FidelityPolicy,
    ) -> int:
        source_heads = int(source["num_attention_heads"])
        source_experts = int(source["num_experts"])
        local_heads = (
            source_heads // policy.reference_tp
            if source_heads % policy.reference_tp == 0
            else source_heads
        )
        local_experts = (
            source_experts // policy.reference_ep
            if source_experts % policy.reference_ep == 0
            else source_experts
        )
        crosses_attn_res = layers > int(source.get("attn_res_block_size", layers))
        if policy.profile == "kernel":
            return (
                (100_000 if heads == local_heads else 0)
                + (100_000 if experts == local_experts else 0)
                + layers * 100
                + experts
            )
        return (
            (100_000 if crosses_attn_res else 0)
            + (20_000 if heads == local_heads else 0)
            + (2_000 if experts == local_experts else 0)
            + layers * 100
            + experts
        )

    @staticmethod
    def _preserved(text: dict[str, Any]) -> list[str]:
        return [
            "hidden_size",
            "KDA head_dim and convolution width",
            "Q/KV LoRA ranks and MLA dimensions",
            "LatentMoE hidden and intermediate sizes",
            "top-k routing",
            "SiTU activation",
            "MXFP4/NVFP4 group format when present",
            "vision tower",
        ]

    @staticmethod
    def _warnings(
        source: dict[str, Any],
        layers: int,
        heads: int,
        experts: int,
        policy: FidelityPolicy,
    ) -> list[str]:
        warnings = [
            "Static estimate only; validate allocator peak and selected kernels.",
            "Single-device execution does not reproduce distributed collectives.",
        ]
        if layers <= int(source.get("attn_res_block_size", layers)):
            warnings.append(
                "The scaled depth does not cross an AttnRes block boundary."
            )
        if source["num_experts"] // policy.reference_ep != experts:
            warnings.append("Local expert count differs from the reference EP plan.")
        if source["num_attention_heads"] // policy.reference_tp != heads:
            warnings.append(
                "Local attention head count differs from the reference TP plan."
            )
        return warnings
