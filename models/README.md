# Dummy model metadata

Run from the repository root. These commands intentionally omit model weights.

## GLM-5.2 on a single 32 GB GPU

```bash
uvx hf download zai-org/GLM-5.2 \
  config.json \
  tokenizer.json \
  tokenizer_config.json \
  chat_template.jinja \
  --local-dir ./models/GLM-5.2-dummy

mv ./models/GLM-5.2-dummy/config.json \
  ./models/GLM-5.2-dummy/config.json.ori

uv run pocketinfer scale \
  ./models/GLM-5.2-dummy/config.json.ori \
  --output-dir ./models/GLM-5.2-dummy \
  --memory-budget 28GiB \
  --kv-cache-budget 4GiB \
  --runtime-reserve 5GiB \
  --max-model-len 4096 \
  --profile kernel \
  --force

vllm serve ./models/GLM-5.2-dummy \
  --load-format dummy \
  --trust-remote-code \
  --max-model-len 4096 \
  --enforce-eager
```

This produces an 11-layer, 16-expert BF16 test model with an estimated
16.45 GiB of weights. vLLM 0.26.0 loaded it in 15.36 GiB, reserved 12.29 GiB
for KV cache, and started on one 32 GB GPU. Remove `--enforce-eager` when
profiling CUDA Graph behavior.
Read `fidelity-report.md` for the human-facing scaling summary; the JSON
manifest is the machine-readable record.

## Kimi K3 metadata

```bash
uvx hf download moonshotai/Kimi-K3 \
  config.json \
  tokenizer_config.json \
  tiktoken.model \
  tokenization_kimi.py \
  encoding_k3.py \
  preprocessor_config.json \
  kimi_k3_vision_processing.py \
  media_utils.py \
  --local-dir ./models/Kimi-K3-dummy
```
