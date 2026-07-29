# PocketInfer

[简体中文](README.zh-CN.md)

> Debugging a giant model should not begin with waiting for a GPU cluster.

Model integration, prefix caching, scheduling, MoE routing, quantization, and
kernel work are all ordinary engine development. With a trillion-scale config,
they become memory and cluster problems before the real work even begins. Even
`--load-format dummy` still builds the shapes described by that config.

Hand-editing the config can create false confidence. Change the wrong head
count, expert count, or layer schedule and the model may silently take a
fallback, bypass a cache path, or lose the topology under test. A server that
starts is not proof that you tested what you intended.

What developers need is not a generic tiny LLM. They need a **small test model
that still behaves like the architecture under development**.

PocketInfer takes the official config and a memory budget, then generates the
most useful scaled-down version that fits—keeping KDA/DSA, MoE routing, cache
topology, and important kernel shapes where the selected profile allows. Every
compromise is written to a manifest instead of being hidden.

**Status:** alpha. Kimi K3 and GLM-5.2 are supported. GLM-5.2 dummy-load
startup is validated on a single 32 GB GPU with vLLM 0.26.0; real weights
and output correctness are not validated.

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
- `fidelity-report.md`: short human-readable scaling report.
- `pocketinfer-manifest.json`: machine-readable audit data.

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

GLM-5.2 also starts on a single 32 GB GPU:

```bash
vllm serve models/GLM-5.2-dummy/ \
  --load-format=dummy \
  --trust-remote-code \
  --enforce-eager \
  --max-model-len 4096
```

```text
Resolved architecture: GlmMoeDsaForCausalLM
Using max model len 4096
Using FLASH_ATTN_MLA_SPARSE attention backend
Using TritonExperts MoE backend
Model loading took 15.36 GiB memory
Available KV cache memory: 12.29 GiB
GPU KV cache size: 990,144 tokens
Starting vLLM server on http://0.0.0.0:8000
NVIDIA H800 NVL: 31435 MiB / 32000 MiB
```

This validates model construction, DSA/MLA and MoE backend selection, and
server startup—not real-weight inference quality.

## Development

```bash
./scripts/ci.sh
```

CPU CI covers Python 3.11/3.12, lint, tests, package build, and CLI smoke. It
does not cover vLLM model construction or GPU execution.

[Design](docs/design.md) · [Model support](docs/model-support.md) ·
[CI scope](docs/ci.md) · [Contributing](CONTRIBUTING.md) ·
[Release checklist](docs/release.md)
