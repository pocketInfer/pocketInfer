# PocketInfer

[简体中文](README.zh-CN.md)

PocketInfer turns a very large model configuration into a memory-bounded,
architecture-faithful configuration for inference-engine development.

It is not model distillation and does not preserve output quality. Its purpose is
to exercise the same model implementation, cache machinery, routing logic, and
kernel families without loading the original trillion-parameter checkpoint.

## Why

Naively reducing every integer in `config.json` changes kernel shapes and can
silently remove the architecture feature being tested. PocketInfer uses explicit
model adapters to preserve invariants and fails closed for unsupported models.

The MVP supports:

- Kimi K3: KDA/MLA ratio, AttnRes, LatentMoE, SiTU, top-k, and MXFP4 layout.
- GLM-5.2: DSA dimensions, IndexShare phase, dense-to-MoE transition, and MTP.

## Install

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

## Generate a config

Download the official `config.json`, then declare resources rather than a GPU SKU:

```bash
pocketinfer scale ./Kimi-K3/config.json \
  --output-dir ./out/kimi-k3 \
  --memory-budget 32GiB \
  --kv-cache-budget 6GiB \
  --runtime-reserve 6GiB \
  --max-model-len 4096 \
  --profile balanced \
  --reference-tp 8 \
  --reference-ep 16
```

Outputs:

- `config.json`: generated model config.
- `pocketinfer-manifest.json`: budget, estimates, changed fields, preserved
  invariants, and known fidelity losses.

The solver enforces both the weight envelope and a minimum cache estimate for
one `--max-model-len` request. Remaining KV capacity determines concurrency.

`balanced` prioritizes meaningful topology boundaries before local expert count.
`kernel` prioritizes rank-local attention-head and expert-count shapes from the
declared reference parallel plan.

For a 32 GiB envelope with 6 GiB each reserved for KV cache and runtime, the
representative official shapes select:

| Adapter | Profile | Layers / heads / experts | Parameters | Weight estimate |
| --- | --- | --- | --- | --- |
| Kimi K3 | balanced | 13 / 12 / 32 | 19.1B | 18.20 GiB |
| GLM-5.2 | balanced | 11 / 8 / 16 | 8.8B | 16.45 GiB |

These are compiler estimates, not measured peak memory.

## Using with vLLM

Copy tokenizer and model-side config files into the generated directory, then:

```bash
vllm serve ./out/kimi-k3 \
  --load-format dummy \
  --max-model-len 4096 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --kv-cache-memory-bytes 6G \
  --moe-backend auto \
  --enforce-eager
```

Remove `--enforce-eager` for CUDA graph and performance profiling.

## What the estimate does not prove

The estimate is not a hardware benchmark. It excludes allocator peaks, backend
repacking, CUDA graph pools, temporary activations, and communication. A generated
config must pass model-construction tests and a real accelerator smoke test before
being called runnable.

Single-device execution cannot reproduce TP/EP collectives. Accurate distributed
bottleneck work requires a future trace-replay layer in addition to config scaling.

## Development

```bash
./scripts/ci.sh
```

This is CPU-only validation: lint, formatting, unit and golden tests, package
build, and CLI smoke tests on Python 3.11 and 3.12. It does not prove model
construction or kernel execution on a GPU. See [CI and validation](docs/ci.md).

See [the design](docs/design.md), [model support](docs/model-support.md),
[contributing](CONTRIBUTING.md), [release handoff](docs/release.md), and
[ADR 0001](docs/adr/0001-adapters-and-constraints.md).
