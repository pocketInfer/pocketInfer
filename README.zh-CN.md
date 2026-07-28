# PocketInfer

[English](README.md)

把万亿参数模型的 Hugging Face 配置编译成受显存预算约束的小型 fixture，用于
推理引擎开发。它保留选定的架构和代码路径，不保留模型能力。

**状态：** alpha。已支持 Kimi K3 和 GLM-5.2；尚未完成 GPU runtime 验证。

## 快速开始

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

输出：

- `config.json`：生成的 Hugging Face 配置。
- `pocketinfer-manifest.json`：尺寸、显存估算、字段变化、保留项和警告。

32 GiB 示例，KV cache 和 runtime 各预留 6 GiB：

| 模型 | 层数 / heads / experts | 参数量 | 权重估算 |
| --- | --- | --- | --- |
| Kimi K3 | 13 / 12 / 32 | 19.1B | 18.20 GiB |
| GLM-5.2 | 11 / 8 / 16 | 8.8B | 16.45 GiB |

这是静态估算，不是 GPU 峰值显存实测。

## 配合 vLLM

把原模型 tokenizer 文件复制到输出目录，然后执行：

```bash
vllm serve ./out/kimi-k3 \
  --load-format dummy \
  --max-model-len 4096 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --kv-cache-memory-bytes 6G \
  --enforce-eager
```

生成配置仍走原生 K3/GLM model path。实际 backend 和 kernel 取决于 vLLM
版本、设备、dtype 和运行参数。

## 开发

```bash
./scripts/ci.sh
```

CPU CI 覆盖 Python 3.11/3.12、lint、测试、构建和 CLI smoke，不覆盖 vLLM
模型构造或 GPU 执行。

[设计](docs/design.zh-CN.md) · [模型支持](docs/model-support.zh-CN.md) ·
[CI 范围](docs/ci.zh-CN.md) · [贡献指南](CONTRIBUTING.zh-CN.md) ·
[发布清单](docs/release.zh-CN.md)
