from __future__ import annotations

from typing import Any

from pocketinfer.sizes import format_gib


def render_fidelity_report(manifest: dict[str, Any]) -> str:
    budget = manifest["budget"]
    estimate = manifest["estimate"]
    dimensions = manifest["selected_dimensions"]
    lines = [
        "# PocketInfer fidelity report",
        "",
        f"- Adapter: `{manifest['adapter']}`",
        f"- Profile: `{manifest['profile']}`",
        f"- Selected shape: {_format_dimensions(dimensions)}",
        (
            f"- Estimate: {estimate['parameter_count'] / 10**9:.2f}B parameters, "
            f"{estimate['weight_gib']:.2f} GiB weights"
        ),
        (
            f"- Budget: {format_gib(budget['total_memory_bytes'])} total, "
            f"{format_gib(budget['kv_cache_bytes'])} KV cache, "
            f"{format_gib(budget['runtime_reserve_bytes'])} runtime reserve"
        ),
        f"- Max model length: {budget['max_model_len']}",
        "",
        "## Shape changes",
        "",
    ]
    changes = manifest["changes"]
    lines.extend(
        f"- `{path}`: `{change['before']}` → `{change['after']}`"
        for path, change in changes.items()
    )
    lines.extend(["", "## Topology changes", ""])
    lines.extend(
        f"- `{path}`: {_format_sequence(change['before'])} → "
        f"{_format_sequence(change['after'])}"
        for path, change in manifest["topology_changes"].items()
    )
    lines.extend(["", "## Preserved", ""])
    lines.extend(f"- {item}" for item in manifest["preserved"])
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {item}" for item in manifest["warnings"])
    return "\n".join(lines) + "\n"


def _format_dimensions(dimensions: dict[str, int]) -> str:
    return ", ".join(f"{name}={value}" for name, value in dimensions.items())


def _format_sequence(summary: dict[str, Any]) -> str:
    sequence = summary.get("sequence")
    counts = summary.get("counts")
    if sequence is not None and (not counts or len(counts) == len(sequence)):
        return f"{summary['length']} entries {sequence}"
    if counts:
        formatted = ", ".join(f"{value}×{count}" for value, count in counts.items())
        return f"{summary['length']} entries ({formatted})"
    return f"{summary['length']} entries ({summary['prefix']} … {summary['suffix']})"
