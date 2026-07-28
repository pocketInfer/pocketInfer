# PocketInfer

[简体中文](README.zh-CN.md)

> Changing one kernel should not require booking a GPU cluster.

New models are too large for an ordinary development loop. Even with
`--load-format dummy`, vLLM still builds the shapes described by the original
config. A small kernel change can turn into hours of waiting for scarce GPUs.

Hand-editing the config is worse than it looks. Shrink the wrong head, expert,
or layer dimension and the model may silently leave the fused kernel, skip the
cache path, or stop exercising the feature you meant to test. The server starts;
the test is still meaningless.

What developers need is not a generic tiny LLM. They need a **small test model
that still behaves like the architecture under development**.

PocketInfer takes the official config and a memory budget, then generates the
most useful scaled-down version that fits—keeping KDA/DSA, MoE routing, cache
topology, and important kernel shapes where the selected profile allows. Every
compromise is written to a manifest instead of being hidden.

**Status:** alpha. Kimi K3 and GLM-5.2 are supported; GPU runtime is not yet
validated.

## Quick start

```bash
uv sync --extra dev

pocketinfer scale ./Kimi-K3/config.json \
  --output-dir ./out/kimi-k3 \
  --memory-budget 32GiB \
  --kv-cache-budget 6GiB \
  --runtime-reserve 6GiB \
  --max-model-len 4096 \
  --profile balanced
```

The command writes:

- `config.json`: generated Hugging Face config.
- `pocketinfer-manifest.json`: dimensions, memory estimate, changed fields,
  preserved invariants, and warnings.

Representative 32 GiB result, with 6 GiB each reserved for KV cache and runtime:

| Model | Layers / heads / experts | Parameters | Estimated weights |
| --- | --- | --- | --- |
| Kimi K3 | 13 / 12 / 32 | 19.1B | 18.20 GiB |
| GLM-5.2 | 11 / 8 / 16 | 8.8B | 16.45 GiB |

These are static estimates, not measured GPU peaks.

## Run with vLLM

Copy the original tokenizer files into the output directory, then:

```bash
vllm serve ./out/kimi-k3 \
  --load-format dummy \
  --max-model-len 4096 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --kv-cache-memory-bytes 6G \
  --enforce-eager
```

The generated config takes the native K3/GLM model path. Actual backend and
kernel selection still depends on the vLLM version, device, dtype, and runtime
flags.

## Development

```bash
./scripts/ci.sh
```

CPU CI covers Python 3.11/3.12, lint, tests, package build, and CLI smoke. It
does not cover vLLM model construction or GPU execution.

[Design](docs/design.md) · [Model support](docs/model-support.md) ·
[CI scope](docs/ci.md) · [Contributing](CONTRIBUTING.md) ·
[Release checklist](docs/release.md)
