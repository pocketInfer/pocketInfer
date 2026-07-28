from __future__ import annotations

from typing import Any


def dtype_bytes(config: dict[str, Any]) -> int:
    dtype = str(config.get("dtype", config.get("torch_dtype", "bfloat16"))).lower()
    if "float32" in dtype:
        return 4
    if "float8" in dtype or dtype == "fp8":
        return 1
    return 2


def candidate_values(
    source: int,
    minimum: int,
    seeds: tuple[int, ...],
    *,
    multiple: int = 1,
) -> list[int]:
    values = {source, minimum}
    values.update(value for value in seeds if minimum <= value <= source)
    return sorted(
        value
        for value in values
        if value <= source and value >= minimum and value % multiple == 0
    )


def nested_diff(
    source: Any,
    generated: Any,
    prefix: str = "",
) -> dict[str, dict[str, Any]]:
    if isinstance(source, dict) and isinstance(generated, dict):
        changes: dict[str, dict[str, Any]] = {}
        for key in sorted(source.keys() | generated.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in source:
                changes[path] = {"before": None, "after": generated[key]}
            elif key not in generated:
                changes[path] = {"before": source[key], "after": None}
            else:
                changes.update(nested_diff(source[key], generated[key], path))
        return changes
    if source != generated:
        return {prefix: {"before": source, "after": generated}}
    return {}
