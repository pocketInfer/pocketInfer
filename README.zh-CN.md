# PocketInfer

[English](README.md)

> 不下载巨型权重，先把超大模型的关键推理路径跑起来。

PocketInfer 读取官方 Hugging Face config 和显存预算，生成一份更小、仍遵循关键
架构约束的原生 config。配合 vLLM `--load-format dummy`，无需下载 checkpoint，
即可测试模型构造、cache 拓扑、MoE routing 和 kernel path。

它不是模型压缩或蒸馏：PocketInfer 生成的是测试 shape，不是训练好的小权重。

## 已验证结果

仓库内置的 GLM-5.2 范例已在单张 32 GB GPU、vLLM 0.26.0 上启动：

```text
Resolved architecture: GlmMoeDsaForCausalLM
Model loading took 15.36 GiB memory
Available KV cache memory: 12.29 GiB
GPU KV cache size: 990,144 tokens
Starting vLLM server on http://0.0.0.0:8000
```

这证明模型构造、DSA/MLA 与 MoE backend 选择、cache 分配和服务启动可行；
不代表真实权重正确性或性能保真。

## 直接运行

所有命令都从仓库根目录执行。已验证的 GLM engine-only 命令不需要权重和
tokenizer：

```bash
vllm serve ./models/GLM-5.2-dummy \
  --load-format dummy --skip-tokenizer-init \
  --max-model-len 4096 --enforce-eager
```

仓库也提供了待验证的 Kimi K3 smoke 命令：

```bash
vllm serve ./models/Kimi-K3-dummy \
  --load-format dummy --skip-tokenizer-init \
  --max-model-len 4096 --enforce-eager
```

Kimi K3 已通过 compiler 测试，尚未完成 GPU runtime 验证。

## 内置范例

| 范例 | Profile | 原始 → 生成 shape | 格式 | 权重估算 | 证据 |
| --- | --- | --- | --- | --- | --- |
| `./models/GLM-5.2-dummy` | kernel | 78 → 11 层，64 → 8 heads，256 → 16 experts | BF16 | 16.45 GiB | vLLM 0.26.0，单卡 32 GB |
| `./models/Kimi-K3-dummy` | balanced | 93 → 13 层，96 → 12 heads，896 → 32 experts | MXFP4 | 18.20 GiB | compiler-tested |

每个目录都包含原始 config、生成 config、简短的 fidelity report 和机器可读
manifest。它们只是范例，不代表固定硬件档位或显存下限。

## 保留了什么

adapter 编码模型特定约束，不会机械缩小每个整数：

- Kimi K3 保留 KDA/MLA 调度、Q/KV latent rank、LatentMoE 维度、top-16
  routing、SiTU、MXFP4 元数据，并跨过一个 AttnRes 边界。
- GLM-5.2 保留 DSA/IndexShare 周期、latent rank、dense-to-MoE 转换、top-k
  routing、MTP 和 RoPE 设置。

生成模型不复现 checkpoint 数值、分布式通信、精确 kernel dispatch、模型质量或
生产性能。

## 重新生成或自定义

预算关系：

```text
weight budget = memory budget - KV cache budget - runtime reserve
```

`balanced` 优先覆盖关键架构；`kernel` 优先保留参考部署中的本地 head/expert
shape，适合 kernel 开发。

下面的命令可精确重建仓库范例：

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

替换源 config、输出目录、预算或 profile，即可生成其他目标。每次输出
`config.json`、`fidelity-report.md` 和 `pocketinfer-manifest.json`。
所有估算都是静态值，不是 GPU 峰值实测。

## 启用文本输入

`--skip-tokenizer-init` 只适合 engine 测试。要通过 OpenAI API 发送文本，
把 tokenizer 元数据下载到同一目录，再去掉该参数：

```bash
uvx hf download zai-org/GLM-5.2 \
  tokenizer.json tokenizer_config.json chat_template.jinja \
  --local-dir ./models/GLM-5.2-dummy

uvx hf download moonshotai/Kimi-K3 \
  --include "*.py" --include "*.json" --include "*.model" \
  --exclude "config.json" --exclude "*.safetensors*" --exclude "assets/*" \
  --local-dir ./models/Kimi-K3-dummy
```

## 开发

```bash
./scripts/ci.sh
```

CPU CI 覆盖 Python 3.11/3.12 的 lint、测试、构建和安装后 CLI smoke；
runtime 结论需要单独的 vLLM 和 GPU 证据。

[设计](docs/design.zh-CN.md) · [模型支持](docs/model-support.zh-CN.md) ·
[CI 范围](docs/ci.zh-CN.md) · [贡献指南](CONTRIBUTING.zh-CN.md) ·
[发布清单](docs/release.zh-CN.md)
