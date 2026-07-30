# PocketInfer

[简体中文](README.zh-CN.md)

> Exercise giant-model inference paths without giant-model hardware.

PocketInfer compiles an official Hugging Face config and a memory budget into a
smaller, native config that preserves selected architecture constraints. Run it
with vLLM `--load-format dummy` to exercise model construction, cache topology,
MoE routing, and kernel paths without downloading the checkpoint.

This is not model compression or distillation. PocketInfer generates test
shapes, not smaller trained weights.

## Proven result

A bundled GLM-5.2 example starts on one 32 GB GPU with vLLM 0.26.0:

```text
Resolved architecture: GlmMoeDsaForCausalLM
Model loading took 15.36 GiB memory
Available KV cache memory: 12.29 GiB
GPU KV cache size: 990,144 tokens
Starting vLLM server on http://0.0.0.0:8000
```

This proves model construction, DSA/MLA and MoE backend selection, cache
allocation, and server startup. It does not prove real-weight correctness or
performance fidelity.

## Try it

Run from the repository root. The validated GLM engine-only command needs
neither weights nor tokenizer files:

```bash
vllm serve ./models/GLM-5.2-dummy \
  --load-format dummy --skip-tokenizer-init \
  --max-model-len 4096 --enforce-eager
```

The repository also includes the candidate Kimi K3 smoke command:

```bash
vllm serve ./models/Kimi-K3-dummy \
  --load-format dummy --skip-tokenizer-init \
  --max-model-len 4096 --enforce-eager
```

Kimi K3 is compiler-tested but not yet GPU-runtime-validated.

## Bundled examples

| Example | Profile | Source → generated shape | Format | Estimated weights | Evidence |
| --- | --- | --- | --- | --- | --- |
| `./models/GLM-5.2-dummy` | kernel | 78 → 11 layers, 64 → 8 heads, 256 → 16 experts | BF16 | 16.45 GiB | vLLM 0.26.0, single 32 GB GPU |
| `./models/Kimi-K3-dummy` | balanced | 93 → 13 layers, 96 → 12 heads, 896 → 32 experts | MXFP4 | 18.20 GiB | compiler-tested |

Each directory contains the source config, generated config, a short fidelity
report, and a machine-readable manifest. They are examples, not hardware tiers
or memory limits.

## What stays faithful

Adapters encode model-specific constraints instead of scaling every integer:

- Kimi K3 retains KDA/MLA schedules, Q/KV latent ranks, LatentMoE dimensions,
  top-16 routing, SiTU, MXFP4 metadata, and an AttnRes boundary.
- GLM-5.2 retains DSA/IndexShare cadence, latent ranks, the dense-to-MoE
  transition, top-k routing, MTP, and RoPE settings.

The generated model does not reproduce checkpoint values, distributed
collectives, exact kernel dispatch, quality, or production performance.

## Rebuild or customize

The planning envelope is:

```text
weight budget = memory budget - KV cache budget - runtime reserve
```

`balanced` favors architecture coverage. `kernel` favors reference-local
head/expert shapes for kernel work.

These commands reproduce the checked-in examples:

```bash
uv sync --extra dev

uv run pocketinfer scale ./models/GLM-5.2-dummy/config.json.ori \
  --output-dir ./models/GLM-5.2-dummy \
  --memory-budget 28GiB --kv-cache-budget 4GiB --runtime-reserve 5GiB \
  --max-model-len 4096 --profile kernel --force

uv run pocketinfer scale ./models/Kimi-K3-dummy/config.json.ori \
  --output-dir ./models/Kimi-K3-dummy \
  --memory-budget 28GiB --kv-cache-budget 4GiB --runtime-reserve 5GiB \
  --max-model-len 4096 --profile balanced --force
```

Change the source, output, budget, or profile for another target. Each run
writes `config.json`, `fidelity-report.md`, and
`pocketinfer-manifest.json`. Estimates are static, not measured peaks.

## Add text input

`--skip-tokenizer-init` is for engine testing. To use text through the OpenAI
API, download tokenizer metadata into the same directory and remove that flag:

```bash
uvx hf download zai-org/GLM-5.2 \
  tokenizer.json tokenizer_config.json chat_template.jinja \
  --local-dir ./models/GLM-5.2-dummy

uvx hf download moonshotai/Kimi-K3 \
  --include "*.py" --include "*.json" --include "*.model" \
  --exclude "config.json" --exclude "*.safetensors*" --exclude "assets/*" \
  --local-dir ./models/Kimi-K3-dummy
```

## Development

```bash
./scripts/ci.sh
```

CPU CI covers lint, tests, package build, and installed CLI smoke on Python
3.11/3.12. Runtime claims require separate vLLM and GPU evidence.

[Design](docs/design.md) · [Model support](docs/model-support.md) ·
[CI scope](docs/ci.md) · [Contributing](CONTRIBUTING.md) ·
[Release checklist](docs/release.md)
