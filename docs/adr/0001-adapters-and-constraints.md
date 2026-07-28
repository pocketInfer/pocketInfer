# ADR 0001: Use adapters around a generic constraint engine

Status: accepted

## Context

Model configs expose similar capacity axes, but fields such as KDA layer lists,
AttnRes boundaries, GLM IndexShare phases, and MTP layers have family-specific
semantics. A universal ratio-based JSON patcher can create syntactically valid
but architecturally invalid models.

## Decision

Use a small generic engine for budget filtering and ranking. Each model-family
adapter declares candidate dimensions, derived fields, estimates, invariants,
and fidelity scoring. Unknown models fail closed.

## Alternatives

- Per-model scripts are initially simpler but duplicate budget and reporting
  behavior.
- A universal field-name rule engine cannot safely infer architecture semantics.
- SMT/CP-SAT is unnecessary for the MVP because each adapter has a small,
  enumerable discrete search space.

## Consequences

Adding a model requires code and tests, but serving engines receive ordinary
configs and need no special "mini model" runtime branch. A solver can replace
enumeration later without changing adapter contracts.

