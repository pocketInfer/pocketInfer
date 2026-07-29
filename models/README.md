# Bundled dummy model configs

The repository tracks two generated examples with their source configs,
fidelity reports, and manifests—not weights or tokenizer files. Their budgets
and profiles are examples, not hardware requirements.

From a clean clone, start either engine without tokenization:

```bash
vllm serve ./models/GLM-5.2-dummy \
  --load-format dummy --skip-tokenizer-init \
  --max-model-len 4096 --enforce-eager

vllm serve ./models/Kimi-K3-dummy \
  --load-format dummy --skip-tokenizer-init \
  --max-model-len 4096 --enforce-eager
```

The GLM kernel-profile example is GPU-startup validated on one 32 GB GPU. The
Kimi balanced-profile example uses the same 28 GiB config budget but still
needs GPU validation. Remove `--enforce-eager` when profiling CUDA Graph
behavior.

To enable text input, download the upstream tokenizer metadata into the model
directory and omit `--skip-tokenizer-init`:

```bash
uvx hf download zai-org/GLM-5.2 \
  tokenizer.json tokenizer_config.json chat_template.jinja \
  --local-dir ./models/GLM-5.2-dummy

uvx hf download moonshotai/Kimi-K3 \
  tokenizer_config.json tiktoken.model tokenization_kimi.py encoding_k3.py \
  preprocessor_config.json kimi_k3_vision_processing.py media_utils.py \
  --local-dir ./models/Kimi-K3-dummy
```

Regenerate either bundled config with:

```bash
uv run pocketinfer scale ./models/GLM-5.2-dummy/config.json.ori \
  --output-dir ./models/GLM-5.2-dummy \
  --memory-budget 28GiB --kv-cache-budget 4GiB --runtime-reserve 5GiB \
  --max-model-len 4096 --profile kernel --force

uv run pocketinfer scale ./models/Kimi-K3-dummy/config.json.ori \
  --output-dir ./models/Kimi-K3-dummy \
  --memory-budget 28GiB --kv-cache-budget 4GiB --runtime-reserve 5GiB \
  --max-model-len 4096 --profile balanced --force
```

The bundled GLM shape is 11 layers / 8 heads / 16 experts (16.45 GiB
estimated). Kimi is 13 / 12 / 32 in native MXFP4 (18.20 GiB estimated) and
crosses its 12-layer AttnRes boundary.
