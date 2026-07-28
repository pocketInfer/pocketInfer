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
