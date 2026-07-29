from __future__ import annotations

import pytest

from pocketinfer import FidelityPolicy, ResourceBudget, ScaleError, scale_config
from pocketinfer.sizes import parse_size
from tests.config_samples import glm52_config, kimi_k3_config


def budget(total: str = "32GiB") -> ResourceBudget:
    return ResourceBudget(
        total_memory_bytes=parse_size(total),
        kv_cache_bytes=parse_size("6GiB"),
        runtime_reserve_bytes=parse_size("6GiB"),
    )


def test_kimi_balanced_preserves_kernel_widths_and_attnres() -> None:
    source = kimi_k3_config()
    result = scale_config(source, budget(), FidelityPolicy(profile="balanced"))
    text = result.config["text_config"]

    assert result.adapter == "kimi-k3"
    assert text["hidden_size"] == 7168
    assert text["q_lora_rank"] == 1536
    assert text["kv_lora_rank"] == 512
    assert text["routed_expert_hidden_size"] == 3584
    assert text["moe_intermediate_size"] == 3072
    assert text["num_hidden_layers"] > text["attn_res_block_size"]
    assert text["linear_attn_config"]["num_heads"] == 12
    assert text["num_experts_per_token"] == 16
    assert text["num_experts"] >= 2 * text["num_experts_per_token"]
    assert result.manifest["estimate"]["weight_bytes"] <= budget().weight_budget_bytes
    assert result.manifest["estimate"]["minimum_cache_bytes"] <= budget().kv_cache_bytes


def test_kimi_layer_lists_cover_every_layer_once() -> None:
    result = scale_config(kimi_k3_config(), budget())
    text = result.config["text_config"]
    linear = text["linear_attn_config"]
    all_layers = sorted(linear["full_attn_layers"] + linear["kda_layers"])

    assert all_layers == list(range(1, text["num_hidden_layers"] + 1))
    assert set(linear["full_attn_layers"]).isdisjoint(linear["kda_layers"])
    assert linear["full_attn_layers"][-1] == text["num_hidden_layers"]


def test_glm_balanced_keeps_indexshare_and_mlp_lists_synchronized() -> None:
    result = scale_config(glm52_config(), budget())
    config = result.config

    assert result.adapter == "glm-5.2"
    assert len(config["indexer_types"]) == config["num_hidden_layers"]
    assert len(config["mlp_layer_types"]) == config["num_hidden_layers"]
    assert config["mlp_layer_types"][:3] == ["dense", "dense", "dense"]
    assert set(config["mlp_layer_types"][3:]) == {"sparse"}
    assert config["indexer_types"][:7] == [
        "full",
        "full",
        "full",
        "shared",
        "shared",
        "shared",
        "full",
    ]
    assert config["num_nextn_predict_layers"] == 1
    assert config["n_routed_experts"] >= 2 * config["num_experts_per_tok"]
    assert result.manifest["estimate"]["weight_bytes"] <= budget().weight_budget_bytes
    assert result.manifest["schema_version"] == 2
    assert result.manifest["changes"]["num_hidden_layers"] == {
        "before": 78,
        "after": 11,
    }
    indexer_change = result.manifest["topology_changes"]["indexer_types"]
    assert indexer_change["before"]["length"] == 78
    assert indexer_change["after"]["sequence"] == config["indexer_types"]


def test_kernel_profile_preserves_reference_local_shapes_when_they_fit() -> None:
    result = scale_config(
        glm52_config(),
        budget(),
        FidelityPolicy(profile="kernel", reference_tp=8, reference_ep=16),
    )

    assert result.config["num_attention_heads"] == 8
    assert result.config["n_routed_experts"] == 16


def test_unknown_model_fails_closed() -> None:
    with pytest.raises(ScaleError, match="no adapter supports"):
        scale_config({"model_type": "unknown"}, budget())


def test_impossible_budget_fails() -> None:
    tiny_budget = ResourceBudget(
        total_memory_bytes=parse_size("10GiB"),
        kv_cache_bytes=parse_size("4GiB"),
        runtime_reserve_bytes=parse_size("4GiB"),
    )
    with pytest.raises(ScaleError, match="no kimi-k3 candidate fits"):
        scale_config(kimi_k3_config(), tiny_budget)


def test_cache_budget_is_enforced_for_max_model_len() -> None:
    cache_starved = ResourceBudget(
        total_memory_bytes=parse_size("32GiB"),
        kv_cache_bytes=parse_size("1MiB"),
        runtime_reserve_bytes=parse_size("6GiB"),
        max_model_len=1_000_000,
    )
    with pytest.raises(ScaleError, match="weight/cache budgets"):
        scale_config(glm52_config(), cache_starved)
