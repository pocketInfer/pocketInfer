from __future__ import annotations

from typing import Any


def kimi_k3_config() -> dict[str, Any]:
    return {
        "architectures": ["KimiK3ForConditionalGeneration"],
        "dtype": "bfloat16",
        "model_type": "kimi_k3",
        "text_config": {
            "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0,
            "attn_res_block_size": 12,
            "dtype": "bfloat16",
            "first_k_dense_replace": 1,
            "hidden_act": "situ",
            "hidden_size": 7168,
            "intermediate_size": 33792,
            "kv_lora_rank": 512,
            "latent_moe_use_norm": True,
            "linear_attn_config": {
                "full_attn_layers": [
                    *range(4, 93, 4),
                    93,
                ],
                "gate_lower_bound": -5.0,
                "head_dim": 128,
                "kda_layers": [
                    layer for layer in range(1, 94) if layer % 4 != 0 and layer != 93
                ],
                "num_heads": 96,
                "short_conv_kernel_size": 4,
                "use_full_rank_gate": True,
            },
            "max_position_embeddings": 1048576,
            "mla_use_nope": True,
            "mla_use_output_gate": True,
            "model_type": "kimi_linear",
            "moe_intermediate_size": 3072,
            "moe_layer_freq": 1,
            "num_attention_heads": 96,
            "num_experts": 896,
            "num_experts_per_token": 16,
            "num_hidden_layers": 93,
            "num_key_value_heads": 96,
            "num_shared_experts": 2,
            "q_lora_rank": 1536,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "quantization_config": {
                "config_groups": {
                    "group_0": {
                        "weights": {
                            "group_size": 32,
                            "num_bits": 4,
                            "type": "float",
                        }
                    }
                },
                "format": "mxfp4-pack-quantized",
            },
            "routed_expert_hidden_size": 3584,
            "v_head_dim": 128,
            "vocab_size": 163840,
        },
        "vision_config": {
            "init_pos_emb_height": 64,
            "init_pos_emb_width": 64,
            "merge_kernel_size": [2, 2],
            "patch_size": 14,
            "qkv_hidden_size": 1536,
            "text_hidden_size": 7168,
            "vt_hidden_size": 1024,
            "vt_intermediate_size": 4096,
            "vt_num_hidden_layers": 27,
        },
    }


def glm52_config() -> dict[str, Any]:
    layers = 78
    return {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "dtype": "bfloat16",
        "first_k_dense_replace": 3,
        "head_dim": 192,
        "hidden_act": "silu",
        "hidden_size": 6144,
        "index_head_dim": 128,
        "index_n_heads": 32,
        "index_share_for_mtp_iteration": True,
        "index_skip_topk_offset": 3,
        "index_topk": 2048,
        "index_topk_freq": 4,
        "index_topk_pattern": None,
        "indexer_types": [
            "shared" if max(layer - 3 + 1, 0) % 4 else "full" for layer in range(layers)
        ],
        "intermediate_size": 12288,
        "kv_lora_rank": 512,
        "max_position_embeddings": 1048576,
        "mlp_layer_types": [
            "dense" if layer < 3 else "sparse" for layer in range(layers)
        ],
        "model_type": "glm_moe_dsa",
        "moe_intermediate_size": 2048,
        "moe_layer_freq": 1,
        "n_group": 1,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_attention_heads": 64,
        "num_experts_per_tok": 8,
        "num_hidden_layers": layers,
        "num_key_value_heads": 64,
        "num_nextn_predict_layers": 1,
        "q_lora_rank": 2048,
        "qk_head_dim": 256,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "rope_parameters": {
            "rope_theta": 8000000,
            "rope_type": "default",
        },
        "v_head_dim": 256,
        "vocab_size": 154880,
    }
