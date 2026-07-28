# PocketInfer

[English](README.md)

> 改一个 kernel，不该先申请一组 GPU。

新模型已经大到装不进日常开发环境。即使用 `--load-format dummy`，vLLM
仍会按原始 config 构造完整 shape。一次普通的 kernel 修改，可能先花几个
小时排队等卡。

手改 config 看似简单，风险却更大：少改一个 head、expert 或 layer，就可能
悄悄掉出 fused kernel、绕过 cache path，甚至让你真正想测的功能直接消失。
服务启动了，测试却没有意义。

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
