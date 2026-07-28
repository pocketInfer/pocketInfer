from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Profile = Literal["balanced", "kernel"]


@dataclass(frozen=True)
class ResourceBudget:
    total_memory_bytes: int
    kv_cache_bytes: int
    runtime_reserve_bytes: int
    world_size: int = 1
    max_model_len: int = 4096

    def __post_init__(self) -> None:
        if self.total_memory_bytes <= 0:
            raise ValueError("total_memory_bytes must be positive")
        if self.kv_cache_bytes < 0 or self.runtime_reserve_bytes < 0:
            raise ValueError("cache and runtime reserves cannot be negative")
        if self.weight_budget_bytes <= 0:
            raise ValueError("cache and runtime reserves exhaust the memory budget")
        if self.world_size != 1:
            raise ValueError("the MVP currently supports world_size=1 only")
        if self.max_model_len <= 0:
            raise ValueError("max_model_len must be positive")

    @property
    def weight_budget_bytes(self) -> int:
        return (
            self.total_memory_bytes - self.kv_cache_bytes - self.runtime_reserve_bytes
        )


@dataclass(frozen=True)
class FidelityPolicy:
    profile: Profile = "balanced"
    reference_tp: int = 8
    reference_ep: int = 16

    def __post_init__(self) -> None:
        if self.profile not in ("balanced", "kernel"):
            raise ValueError(f"unsupported profile: {self.profile}")
        if self.reference_tp <= 0 or self.reference_ep <= 0:
            raise ValueError("reference parallel sizes must be positive")


@dataclass(frozen=True)
class Estimate:
    parameter_count: int
    weight_bytes: int
    kv_bytes_per_token: int | None = None
    fixed_cache_bytes: int = 0
    notes: tuple[str, ...] = ()

    def minimum_cache_bytes(self, max_model_len: int) -> int | None:
        if self.kv_bytes_per_token is None:
            return None
        return self.fixed_cache_bytes + self.kv_bytes_per_token * max_model_len


@dataclass
class Candidate:
    config: dict[str, Any]
    dimensions: dict[str, int]
    estimate: Estimate
    score: int
    preserved: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ScaleResult:
    adapter: str
    config: dict[str, Any]
    manifest: dict[str, Any]
