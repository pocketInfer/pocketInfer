from __future__ import annotations

from typing import Any

from pocketinfer.adapters import DEFAULT_ADAPTERS
from pocketinfer.adapters.base import ModelAdapter
from pocketinfer.adapters.common import compact_diff
from pocketinfer.models import Estimate, FidelityPolicy, ResourceBudget, ScaleResult
from pocketinfer.sizes import format_gib


class ScaleError(ValueError):
    pass


def select_adapter(
    config: dict[str, Any],
    adapters: tuple[ModelAdapter, ...] = DEFAULT_ADAPTERS,
) -> ModelAdapter:
    matches = [adapter for adapter in adapters if adapter.supports(config)]
    if not matches:
        model_type = config.get("model_type", "<missing>")
        raise ScaleError(f"no adapter supports model_type={model_type!r}")
    if len(matches) > 1:
        names = ", ".join(adapter.name for adapter in matches)
        raise ScaleError(f"multiple adapters match the config: {names}")
    return matches[0]


def scale_config(
    config: dict[str, Any],
    budget: ResourceBudget,
    policy: FidelityPolicy | None = None,
) -> ScaleResult:
    policy = policy or FidelityPolicy()
    adapter = select_adapter(config)
    fitting = [
        candidate
        for candidate in adapter.candidates(config, policy)
        if _fits(candidate.estimate, budget)
    ]
    if not fitting:
        raise ScaleError(
            f"no {adapter.name} candidate fits weight/cache budgets "
            f"{format_gib(budget.weight_budget_bytes)}/"
            f"{format_gib(budget.kv_cache_bytes)}"
        )
    selected = max(
        fitting,
        key=lambda candidate: (
            candidate.score,
            candidate.estimate.parameter_count,
            -candidate.estimate.weight_bytes,
        ),
    )
    estimate = selected.estimate
    minimum_cache_bytes = estimate.minimum_cache_bytes(budget.max_model_len)
    changes, topology_changes = compact_diff(config, selected.config)
    manifest = {
        "schema_version": 2,
        "adapter": adapter.name,
        "profile": policy.profile,
        "budget": {
            "total_memory_bytes": budget.total_memory_bytes,
            "kv_cache_bytes": budget.kv_cache_bytes,
            "runtime_reserve_bytes": budget.runtime_reserve_bytes,
            "weight_budget_bytes": budget.weight_budget_bytes,
            "world_size": budget.world_size,
            "max_model_len": budget.max_model_len,
        },
        "reference_parallelism": {
            "tensor_parallel": policy.reference_tp,
            "expert_parallel": policy.reference_ep,
        },
        "selected_dimensions": selected.dimensions,
        "estimate": {
            "parameter_count": estimate.parameter_count,
            "weight_bytes": estimate.weight_bytes,
            "weight_gib": round(estimate.weight_bytes / 2**30, 3),
            "kv_bytes_per_token": estimate.kv_bytes_per_token,
            "fixed_cache_bytes": estimate.fixed_cache_bytes,
            "minimum_cache_bytes": minimum_cache_bytes,
            "minimum_cache_gib": (
                round(minimum_cache_bytes / 2**30, 3)
                if minimum_cache_bytes is not None
                else None
            ),
            "notes": list(estimate.notes),
        },
        "preserved": selected.preserved,
        "warnings": selected.warnings,
        "changes": changes,
        "topology_changes": topology_changes,
    }
    return ScaleResult(
        adapter=adapter.name,
        config=selected.config,
        manifest=manifest,
    )


def _fits(estimate: Estimate, budget: ResourceBudget) -> bool:
    minimum_cache_bytes = estimate.minimum_cache_bytes(budget.max_model_len)
    cache_fits = (
        minimum_cache_bytes is not None and minimum_cache_bytes <= budget.kv_cache_bytes
    )
    return estimate.weight_bytes <= budget.weight_budget_bytes and cache_fits
