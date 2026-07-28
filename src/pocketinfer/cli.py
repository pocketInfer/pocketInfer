from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pocketinfer.engine import ScaleError, scale_config
from pocketinfer.models import FidelityPolicy, ResourceBudget
from pocketinfer.sizes import parse_size


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pocketinfer",
        description="Scale supported LLM configs into a declared memory envelope.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    scale = subparsers.add_parser("scale", help="generate a scaled config")
    scale.add_argument("config", type=Path, help="source config.json")
    scale.add_argument("--output-dir", type=Path, required=True)
    scale.add_argument("--memory-budget", type=parse_size, required=True)
    scale.add_argument("--kv-cache-budget", type=parse_size, default=parse_size("4GiB"))
    scale.add_argument("--runtime-reserve", type=parse_size, default=parse_size("4GiB"))
    scale.add_argument("--max-model-len", type=int, default=4096)
    scale.add_argument("--profile", choices=("balanced", "kernel"), default="balanced")
    scale.add_argument("--reference-tp", type=int, default=8)
    scale.add_argument("--reference-ep", type=int, default=16)
    scale.add_argument("--force", action="store_true")
    return parser


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_scale(args: argparse.Namespace) -> int:
    source = json.loads(args.config.read_text(encoding="utf-8"))
    budget = ResourceBudget(
        total_memory_bytes=args.memory_budget,
        kv_cache_bytes=args.kv_cache_budget,
        runtime_reserve_bytes=args.runtime_reserve,
        max_model_len=args.max_model_len,
    )
    policy = FidelityPolicy(
        profile=args.profile,
        reference_tp=args.reference_tp,
        reference_ep=args.reference_ep,
    )
    result = scale_config(source, budget, policy)
    output_dir: Path = args.output_dir
    config_path = output_dir / "config.json"
    manifest_path = output_dir / "pocketinfer-manifest.json"
    if not args.force and (config_path.exists() or manifest_path.exists()):
        raise ScaleError(
            f"{output_dir} already contains generated files; pass --force to replace"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config_path, result.config)
    _write_json(manifest_path, result.manifest)
    selected = result.manifest["selected_dimensions"]
    weight_gib = result.manifest["estimate"]["weight_gib"]
    print(f"{result.adapter}: {selected}; estimated weights={weight_gib:.2f} GiB")
    print(f"wrote {config_path}")
    print(f"wrote {manifest_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "scale":
            return run_scale(args)
    except (OSError, ValueError, json.JSONDecodeError, ScaleError) as error:
        parser.exit(2, f"error: {error}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
