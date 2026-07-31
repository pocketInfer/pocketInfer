# PocketInfer

[简体中文](README.zh-CN.md)

**Kimi K3 development should not require waiting for a B300.**<br>
PocketInfer helps you shrink a **2.8T-parameter model** to a **single 32 GB GPU** for inference development.

Everyone knows that `--load-format dummy` skips the real checkpoint, but vLLM still constructs the full tensor shapes from the original config. For a roughly 2.8T-parameter Kimi K3, fake weights do not make the memory requirement disappear—the original model still does not fit on one GPU.

Manually removing layers, attention heads, or experts is unreliable. The model may silently bypass the cache, MoE, or kernel path you wanted to test. A server that starts is not necessarily exercising the target architecture.

PocketInfer reads an official Hugging Face config and a memory budget, then produces a **pocket model**: smaller, while still satisfying key architecture constraints. It continues to use the native vLLM model implementation for model integration, scheduler, prefix cache, MoE routing, quantization, and kernel development.

> [!IMPORTANT]
> A pocket model is a mini model that preserves key architecture constraints. It does not place the full 2.8T parameters into 32 GB. PocketInfer is not quantization, distillation, or model compression, and it does not generate weights for quality evaluation.

## Proven: GLM-5.2 on one 32 GB GPU

Run from the repository root:

```bash
vllm serve ./models/GLM-5.2-dummy \
  --load-format dummy \
  --trust-remote-code \
  --skip-tokenizer-init \
  --max-model-len 4096
```

This bundled config was started with vLLM 0.26.0 on one 32 GB GPU:

| Evidence | Result |
| --- | --- |
| Model and server | `GlmMoeDsaForCausalLM` constructed; API server started |
| Inference backends | `FLASH_ATTN_MLA_SPARSE`, FlashAttention MLA prefill, Triton MoE |
| Memory usage | 15.36 GiB model, 12.29 GiB KV cache; 31,435 / 32,000 MiB total GPU usage |

<table>
<tr>
<td width="50%" valign="top">
<a href="docs/assets/runtime-evidence/glm52-single-32gb-memory.png"><img src="docs/assets/runtime-evidence/glm52-single-32gb-memory.png" alt="Single 32 GB GPU memory usage" /></a>
<br><sub>Single-GPU memory usage: 31,435 / 32,000 MiB.</sub>
</td>
<td width="50%" valign="top">
<a href="docs/assets/runtime-evidence/glm52-vllm-bench-serve.png"><img src="docs/assets/runtime-evidence/glm52-vllm-bench-serve.png" alt="vLLM bench serve result" /></a>
<br><sub>vLLM bench serve: 10/10 requests succeeded, 0 failed. Click for the full image.</sub>
</td>
</tr>
</table>

<details>
<summary><strong>Full startup log (two screenshots)</strong></summary>

![GLM-5.2 model construction, backend selection, and KV cache allocation](docs/assets/runtime-evidence/glm52-startup-model-init.png)

![GLM-5.2 API server startup and request execution](docs/assets/runtime-evidence/glm52-startup-api-requests.png)

</details>

<details>
<summary><strong>View profiling (one screenshot)</strong></summary>

![GLM-5.2 Perfetto profiling](docs/assets/runtime-evidence/glm52-profiling-perfetto.png)

</details>

This evidence covers more than construction and server startup: repeated prefill/decode requests were executed. The benchmark environment also contained upstream tokenizer metadata solely to construct requests; PocketInfer itself still reads only `config.json`.

> [!NOTE]
> This run used dummy weights. Throughput and latency describe this engineering check only; they are not real GLM-5.2 quality or production-performance results.

## Two ready-to-run pocket models

| Example | Parameters (source → generated) | Estimated generated weights | Validation |
| --- | --- | --- | --- |
| [GLM-5.2](models/GLM-5.2-dummy) (`kernel`) | 753B → 8.83B | 16.45 GiB, BF16 | vLLM 0.26.0, started on one 32 GB GPU |
| [Kimi K3](models/Kimi-K3-dummy) (`balanced`) | 2.78T → 19.1B | 18.20 GiB, MXFP4 | Config generation and consistency tests passed; GPU runtime pending |

Parameter counts and weight sizes are static estimates from the configs. These examples demonstrate two scaling strategies; they are not fixed hardware tiers. You can change the memory budget, generated size, and strategy.

---

## How PocketInfer works

### Preserve architecture constraints

PocketInfer does not scale every number by the same ratio. Each model adapter maintains constraints specific to that architecture:

- GLM-5.2 retains DSA/IndexShare cadence, latent dimensions, the dense-to-MoE transition, top-k, MTP, and RoPE settings.
- Kimi K3 retains KDA/MLA schedules, Q/KV latent dimensions, LatentMoE, top-16, SiTU, MXFP4 metadata, and an AttnRes boundary.

The `balanced` strategy favors broader architecture coverage. The `kernel` strategy favors local head/expert shapes closer to the source model.

### Plan a memory budget

PocketInfer divides memory among weights, KV cache, and runtime reserve:

```text
weight budget = memory budget - KV cache budget - runtime reserve
```

The compiler searches for the highest-fidelity candidate that fits the weight budget. These numbers are static planning estimates, not measured GPU peaks.

## Generate your pocket model

PocketInfer needs only the source `config.json`—not the weights or tokenizer. Run all commands from the repository root.

### 1. Download the source config

```bash
uvx hf download zai-org/GLM-5.2 config.json \
  --local-dir ./models/GLM-5.2-source
```

### 2. Generate a pocket model

This example assigns 28 GiB total memory, reserves 4 GiB for KV cache and 5 GiB for runtime, and uses the remainder for weights.

```bash
uv sync --extra dev

uv run pocketinfer scale ./models/GLM-5.2-source/config.json \
  --output-dir ./models/GLM-5.2-local \
  --memory-budget 28GiB \
  --kv-cache-budget 4GiB \
  --runtime-reserve 5GiB \
  --max-model-len 4096 \
  --profile kernel
```

<details>
<summary><strong>CLI parameters</strong></summary>

| Argument | Purpose | Default |
| --- | --- | --- |
| `config` | Path to the source Hugging Face `config.json` | required |
| `--output-dir` | Output directory for the pocket model | required |
| `--memory-budget` | Total memory envelope | required |
| `--kv-cache-budget` | Memory reserved from the envelope for KV cache | `4GiB` |
| `--runtime-reserve` | Memory reserved for activations, workspaces, and other runtime costs | `4GiB` |
| `--max-model-len` | Sequence length used for KV cache planning; does not rewrite the model context declaration | `4096` |
| `--profile` | `balanced` favors architecture coverage; `kernel` favors local head/expert shapes | `balanced` |
| `--reference-tp` | Reference TP size used to derive per-GPU head shapes | `8` |
| `--reference-ep` | Reference EP size used to derive per-GPU expert shapes | `16` |
| `--force` | Replace generated files already present in the output directory | off |

Budget arguments affect PocketInfer's static plan; they do not automatically change vLLM runtime arguments.

</details>

<details>
<summary><strong>Generated files</strong></summary>

- `config.json`: scaled config loaded by the native model implementation
- `fidelity-report.md`: preserved fields, scaled fields, and risk notes
- `pocketinfer-manifest.json`: machine-readable budget and generation record

</details>

Change the source config, output directory, budget, and profile to produce another pocket model.

### 3. Start it with vLLM

```bash
vllm serve ./models/GLM-5.2-local \
  --load-format dummy \
  --trust-remote-code \
  --skip-tokenizer-init \
  --max-model-len 4096
```

## Validation boundaries

PocketInfer is useful for model integration, config constraints, runtime initialization, and reaching attention, MoE, quantization, and cache paths. It does not reproduce checkpoint values, distributed communication scale, exact kernel dispatch on every device, model quality, or production performance.

Repository CI runs lint, unit tests, package builds, and an installed CLI smoke test on Python 3.11/3.12:

```bash
./scripts/ci.sh
```

CI evidence and GPU runtime evidence are reported separately so that “the config can be generated” is never presented as “the model ran on a GPU.”

[Design](docs/design.md) · [Model support](docs/model-support.md) · [CI scope](docs/ci.md) · [Contributing](CONTRIBUTING.md) · [Release checklist](docs/release.md)
