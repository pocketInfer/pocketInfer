from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MODELS = Path(__file__).parents[1] / "models"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_estimate_fits(manifest: dict[str, Any]) -> None:
    assert (
        manifest["estimate"]["weight_bytes"]
        <= manifest["budget"]["weight_budget_bytes"]
    )
    assert (
        manifest["estimate"]["minimum_cache_bytes"]
        <= manifest["budget"]["kv_cache_bytes"]
    )
    assert manifest["budget"]["max_model_len"] == 4096


def test_bundled_glm52_shape_matches_manifest() -> None:
    model_dir = MODELS / "GLM-5.2-dummy"
    config = _read_json(model_dir / "config.json")
    manifest = _read_json(model_dir / "pocketinfer-manifest.json")

    assert manifest["selected_dimensions"] == {
        "layers": config["num_hidden_layers"],
        "attention_heads": config["num_attention_heads"],
        "experts": config["n_routed_experts"],
        "top_k": config["num_experts_per_tok"],
    }
    _assert_estimate_fits(manifest)


def test_bundled_kimi_k3_shape_matches_manifest() -> None:
    model_dir = MODELS / "Kimi-K3-dummy"
    config = _read_json(model_dir / "config.json")["text_config"]
    manifest = _read_json(model_dir / "pocketinfer-manifest.json")

    assert manifest["selected_dimensions"] == {
        "layers": config["num_hidden_layers"],
        "attention_heads": config["num_attention_heads"],
        "experts": config["num_experts"],
        "top_k": config["num_experts_per_token"],
    }
    assert config["num_hidden_layers"] > config["attn_res_block_size"]
    assert config["quantization_config"]["format"] == "mxfp4-pack-quantized"
    _assert_estimate_fits(manifest)
