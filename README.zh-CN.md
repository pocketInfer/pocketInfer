# PocketInfer

[English](README.md)

> 调试超大模型，不该从排队等B300开始。 是时候,把2.8T的模型放到一张小卡里跑起来了~

模型接入、prefix cache、scheduler、MoE routing、量化和 kernel，本来都是
普通的推理引擎开发；碰上万亿参数 config，却会先变成显存和集群问题。
每个Kimi-K3/GLM5.2的vLLM开发同学,都需要等一整台B300或者2台H200么?

即使用 `--load-format dummy`，vLLM 仍会按原始 config 构造 shape。
手改 config 又容易制造“假测试”：改错 head 数、expert 数或 layer schedule，模型可能悄悄走 fallback、绕过 cache path，或者丢掉要验证的拓扑。服务能启动，不代表你真的测到了目标功能。

开发者需要的不是一个普通“小模型”，而是一个**缩小后仍像原架构的测试模型**。

PocketInfer 接收官方 config 和显存预算，在预算内寻找最有调试价值的缩小配置：
能保留的 KDA/DSA、MoE routing、cache 拓扑和关键 kernel shape 尽量保留；
不得不失真的地方，全部写进 manifest，不悄悄掩盖。

**状态：** alpha。已支持 Kimi K3 和 GLM-5.2。GLM-5.2 已在单张 32 GB GPU、
vLLM 0.26.0 上完成 dummy-load 启动验证；尚未验证真实权重和输出正确性。

## 内置范例

仓库直接带了两份生成范例，它们不是固定硬件档位或显存下限：

| 路径 | Profile | 层数 / heads / experts | 格式 | 权重估算 | 状态 |
| --- | --- | --- | --- | --- | --- |
| `./models/GLM-5.2-dummy` | kernel | 11 / 8 / 16 | BF16 | 16.45 GiB | GPU 启动已验证 |
| `./models/Kimi-K3-dummy` | balanced | 13 / 12 / 32 | MXFP4 | 18.20 GiB | 待 GPU 验证 |

## 运行范例

所有命令都从仓库根目录执行。只测试 engine 时，不需要权重和 tokenizer：

```bash
vllm serve ./models/GLM-5.2-dummy \
  --load-format dummy --skip-tokenizer-init \
  --max-model-len 4096 --enforce-eager

vllm serve ./models/Kimi-K3-dummy \
  --load-format dummy --skip-tokenizer-init \
  --max-model-len 4096 --enforce-eager
```

要通过 OpenAI API 发送文本，把 tokenizer 元数据下载到同一目录，再去掉
`--skip-tokenizer-init`：

```bash
uvx hf download zai-org/GLM-5.2 \
  tokenizer.json tokenizer_config.json chat_template.jinja \
  --local-dir ./models/GLM-5.2-dummy

uvx hf download moonshotai/Kimi-K3 \
  --include "*.py" --include "*.json" --include "*.model" \
  --exclude "config.json" --exclude "*.safetensors*" --exclude "assets/*" \
  --local-dir ./models/Kimi-K3-dummy
```

## 重新生成或自定义

下面是仓库内两份范例的准确生成命令：

```bash
uv sync --extra dev

uv run pocketinfer scale ./models/GLM-5.2-dummy/config.json.ori \
  --output-dir ./models/GLM-5.2-dummy \
  --memory-budget 28GiB --kv-cache-budget 4GiB --runtime-reserve 5GiB \
  --max-model-len 4096 --profile kernel --force

uv run pocketinfer scale ./models/Kimi-K3-dummy/config.json.ori \
  --output-dir ./models/Kimi-K3-dummy \
  --memory-budget 28GiB --kv-cache-budget 4GiB --runtime-reserve 5GiB \
  --max-model-len 4096 \
  --profile balanced --force
```

换掉源 config、输出目录、预算或 profile，就可以生成其他目标。每次输出
`config.json`、`fidelity-report.md` 和 `pocketinfer-manifest.json`。
所有估算都是静态值，不是 GPU 峰值实测。

## GLM-5.2 单卡实测

```bash
vllm serve ./models/GLM-5.2-dummy \
  --load-format dummy \
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

这证明模型构造、DSA/MLA 与 MoE backend 选择、服务启动可行，不代表真实权重推理质量。

## 开发

```bash
./scripts/ci.sh
```

CPU CI 覆盖 Python 3.11/3.12、lint、测试、构建和 CLI smoke，不覆盖 vLLM
模型构造或 GPU 执行。

[设计](docs/design.zh-CN.md) · [模型支持](docs/model-support.zh-CN.md) ·
[CI 范围](docs/ci.zh-CN.md) · [贡献指南](CONTRIBUTING.zh-CN.md) ·
[发布清单](docs/release.zh-CN.md)
