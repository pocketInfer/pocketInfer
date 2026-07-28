# PocketInfer MVP design

[简体中文](design.zh-CN.md)

## Decision

PocketInfer is a deterministic config compiler for inference development. It
does not train or distill a model, and it does not claim output quality. It
preserves selected architecture and rank-local kernel shapes while shrinking
discrete capacity axes until a declared memory envelope is met.

The technique is broadly suitable for config-driven open-weight Transformer,
MoE, MLA, DSA, KDA, and hybrid models. It is not universal: closed models,
architectures whose semantics are absent from config, and hard-coded proprietary
kernels require new adapters or cannot be supported.

## Architecture

```text
config.json + memory envelope + fidelity policy
                    |
                    v
          model-family adapter
      candidates + invariants + estimator
                    |
                    v
        generic filter and rank engine
                    |
                    v
       config.json + explainable manifest
```

The generic engine contains no model-name branches. Adapters own model semantics,
candidate axes, derived lists, estimates, and fidelity scoring. Unsupported
models fail closed.

## MVP scope

- Local JSON input only; no remote code or checkpoint loading.
- Single-device memory budgets.
- `balanced` and `kernel` fidelity profiles.
- Kimi K3 and GLM-5.2 adapters.
- Static parameter, weight, and cache estimates.
- Deterministic JSON output and a machine-readable change manifest.

## Failure modes

Static estimates exclude allocator peaks, kernel repacking, CUDA graphs, and
distributed communication. Generated configs must be validated by constructing
the model with the target inference engine and then by an accelerator smoke test.
The manifest states these limitations rather than presenting estimates as
measured deployment results.
