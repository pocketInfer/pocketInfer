from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import Any

from pocketinfer.adapters.base import ModelAdapter
from pocketinfer.adapters.common import candidate_values, dtype_bytes
from pocketinfer.models import Candidate, Estimate, FidelityPolicy


class Glm52Adapter(ModelAdapter):
    name = "glm-5.2"

    def supports(self, config: dict[str, Any]) -> bool:
        return config.get("model_type") == "glm_moe_dsa" and (
            config.get("index_topk_freq") == 4
            or config.get("index_share_for_mtp_iteration") is True
        )

    def candidates(
        self,
        config: dict[str, Any],
        policy: FidelityPolicy,
    ) -> Iterable[Candidate]:
        source_layers = int(config["num_hidden_layers"])
        source_heads = int(config["num_attention_heads"])
        source_experts = int(config["n_routed_experts"])
        top_k = int(config["num_experts_per_tok"])
        dense_layers = int(config["first_k_dense_replace"])

        layers = set(range(dense_layers + 4, source_layers + 1, 4))
        layers.add(source_layers)
        local_heads = (
            source_heads // policy.reference_tp
            if source_heads % policy.reference_tp == 0
            else source_heads
        )
        heads = candidate_values(
            source_heads,
            min(local_heads, source_heads),
            (local_heads, 16, 32, source_heads),
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
            (top_k, 16, 32, 64, local_experts),
            multiple=8,
        )

        for layer_count in sorted(layers):
            for head_count in heads:
                for expert_count in experts:
                    generated = deepcopy(config)
                    self._apply_dimensions(
                        generated,
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
                            config,
                            layer_count,
                            head_count,
                            expert_count,
                            policy,
                        ),
                        preserved=self._preserved(),
                        warnings=self._warnings(
                            config,
                            head_count,
                            expert_count,
                            policy,
                        ),
                    )

    @classmethod
    def _apply_dimensions(
        cls,
        config: dict[str, Any],
        layers: int,
        heads: int,
        experts: int,
    ) -> None:
        dense_layers = min(int(config["first_k_dense_replace"]), layers)
        config["num_hidden_layers"] = layers
        config["num_attention_heads"] = heads
        config["num_key_value_heads"] = heads
        config["n_routed_experts"] = experts
        config["mlp_layer_types"] = [
            "dense" if layer < dense_layers else "sparse" for layer in range(layers)
        ]
        config["indexer_types"] = [
            cls._indexer_type(config, layer) for layer in range(layers)
        ]

    @staticmethod
    def _indexer_type(config: dict[str, Any], layer: int) -> str:
        pattern = config.get("index_topk_pattern")
        if isinstance(pattern, list) and layer < len(pattern):
            return "shared" if pattern[layer] == "S" else "full"
        frequency = int(config.get("index_topk_freq", 1))
        offset = int(config.get("index_skip_topk_offset", 2))
        skip = max(layer - offset + 1, 0) % frequency != 0
        return "shared" if skip else "full"

    @staticmethod
    def _estimate(config: dict[str, Any]) -> Estimate:
        h = int(config["hidden_size"])
        vocab = int(config["vocab_size"])
        heads = int(config["num_attention_heads"])
        layers = int(config["num_hidden_layers"])
        experts = int(config["n_routed_experts"])
        dense_i = int(config["intermediate_size"])
        moe_i = int(config["moe_intermediate_size"])
        shared = int(config["n_shared_experts"])
        q_rank = int(config["q_lora_rank"])
        kv_rank = int(config["kv_lora_rank"])
        qk = int(config["qk_head_dim"])
        nope = int(config["qk_nope_head_dim"])
        rope = int(config["qk_rope_head_dim"])
        value = int(config["v_head_dim"])
        index_heads = int(config["index_n_heads"])
        index_dim = int(config["index_head_dim"])
        dbytes = dtype_bytes(config)

        embedding = 2 * vocab * h
        attention = (
            h * (q_rank + kv_rank + rope)
            + q_rank * heads * qk
            + kv_rank * heads * (nope + value)
            + h * heads * value
        )
        indexer = (
            q_rank * index_heads * index_dim + h * (index_dim + index_heads) + index_dim
        )
        dense_mlp = 3 * h * dense_i
        routed_moe = experts * 3 * h * moe_i
        shared_moe = shared * 3 * h * moe_i
        router = h * experts + experts

        mlp_types = config["mlp_layer_types"]
        indexer_types = config["indexer_types"]
        params = embedding
        for layer in range(layers):
            params += attention
            if indexer_types[layer] == "full":
                params += indexer
            if mlp_types[layer] == "dense":
                params += dense_mlp
            else:
                params += routed_moe + shared_moe + router

        mtp_layers = int(config.get("num_nextn_predict_layers", 0))
        if mtp_layers:
            params += mtp_layers * (
                attention + indexer + routed_moe + shared_moe + router
            )

        kv_per_token = layers * (kv_rank + rope) * dbytes
        index_cache_per_token = sum(kind == "full" for kind in indexer_types) * (
            index_dim + 4
        )
        return Estimate(
            parameter_count=params,
            weight_bytes=params * dbytes,
            kv_bytes_per_token=kv_per_token + index_cache_per_token,
            notes=(
                "BF16/FP8 storage is inferred from config dtype only",
                "minimum cache estimate covers one max-length request",
                "MTP is conservatively estimated as one sparse layer",
                "runtime workspaces and distributed replication are excluded",
            ),
        )

    @staticmethod
    def _score(
        source: dict[str, Any],
        layers: int,
        heads: int,
        experts: int,
        policy: FidelityPolicy,
    ) -> int:
        source_heads = int(source["num_attention_heads"])
        source_experts = int(source["n_routed_experts"])
        local_heads = source_heads // policy.reference_tp
        local_experts = source_experts // policy.reference_ep
        if policy.profile == "kernel":
            return (
                (100_000 if heads == local_heads else 0)
                + (100_000 if experts == local_experts else 0)
                + layers * 100
                + experts
            )
        complete_share_cycle = layers >= int(source["first_k_dense_replace"]) + 4
        return (
            (100_000 if complete_share_cycle else 0)
            + (20_000 if heads == local_heads else 0)
            + (2_000 if experts == local_experts else 0)
            + layers * 100
            + experts
        )

    @staticmethod
    def _preserved() -> list[str]:
        return [
            "hidden size and MLA ranks",
            "DSA index head dimensions and top-k",
            "IndexShare frequency and phase",
            "three dense warm-up layers",
            "MoE intermediate size and top-k routing",
            "MTP layer",
            "rope configuration",
        ]

    @staticmethod
    def _warnings(
        source: dict[str, Any],
        heads: int,
        experts: int,
        policy: FidelityPolicy,
    ) -> list[str]:
        warnings = [
            "Static estimate only; DSA cache and backend must be validated in vLLM.",
            "Single-device execution does not reproduce distributed collectives.",
        ]
        if source["num_attention_heads"] // policy.reference_tp != heads:
            warnings.append("Local attention head count differs from reference TP.")
        if source["n_routed_experts"] // policy.reference_ep != experts:
            warnings.append("Local expert count differs from reference EP.")
        return warnings
