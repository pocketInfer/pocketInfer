# PocketInfer

[English](README.md)

**把 2.8T 参数模型，缩到单卡 32 GB 里做推理开发。**

大家都知道 `--load-format dummy` 可以跳过真实权重，但 vLLM 仍会按照原始配置构造完整的张量形状。对于约 2.8T 参数的 Kimi K3，权重可以是假的，显存占用却不会凭空消失——单卡依然装不下。

手工删层、减 attention heads 或 experts 也不可靠：模型可能悄悄绕开目标 cache、MoE 或 kernel path，服务虽然启动了，测到的却不是你想验证的架构。

PocketInfer 解决的是这个问题。它读取官方 Hugging Face 配置和显存预算，生成一份更小、但仍满足关键架构约束的原生配置。生成结果继续走 vLLM 原生模型实现，用于模型接入、调度、cache、MoE、量化和 kernel 开发。

> [!IMPORTANT]
> PocketInfer 缩小的是测试模型，不是把完整的 2.8T 参数原封不动塞进 32 GB。它不是量化、蒸馏或模型压缩，也不会生成可用于效果评测的权重。

## 单卡 32 GB 实测：GLM-5.2

仓库已经内置 GLM-5.2 和 Kimi K3 两个测试模型。以下命令均在仓库根目录执行。

```bash
vllm serve ./models/GLM-5.2-dummy \
  --load-format dummy \
  --skip-tokenizer-init \
  --max-model-len 4096 \
  --enforce-eager
```

这个 GLM-5.2 配置已在 vLLM 0.26.0、单卡 32 GB 环境中实际启动：

```text
Resolved architecture: GlmMoeDsaForCausalLM
Model loading took 15.36 GiB memory
Available KV cache memory: 12.29 GiB
GPU KV cache size: 990,144 tokens
Starting vLLM server on http://0.0.0.0:8000
```

这说明 vLLM 已完成模型构造、DSA/MLA 与 MoE 后端选择、KV Cache 分配和 API 服务启动。它不代表真实权重的正确性，也不能用于推导生产性能。

## 从巨型配置到单卡测试模型

| 范例 | 参数规模（原始 → 生成） | 生成权重估算 | 验证状态 |
| --- | --- | --- | --- |
| [GLM-5.2](models/GLM-5.2-dummy)（`kernel`） | 753B → 8.83B | 16.45 GiB，BF16 | vLLM 0.26.0，单卡 32 GB 已启动 |
| [Kimi K3](models/Kimi-K3-dummy)（`balanced`） | 2.78T → 19.1B | 18.20 GiB，MXFP4 | 配置生成与一致性测试通过，GPU 运行验证待完成 |

参数量和权重占用由配置静态估算。这两个目录只是可直接使用的范例，不对应固定的硬件档位；显存预算、模型规模和缩放策略都可以重新指定。

PocketInfer 也不是简单地把所有数字按比例减小。模型适配器会维护架构内部的约束：GLM-5.2 范例保留 DSA/IndexShare 周期、低秩维度、dense-to-MoE 转换、top-k、MTP 和 RoPE 设置；Kimi K3 范例保留 KDA/MLA 调度、Q/KV 低秩维度、LatentMoE、top-16、SiTU、MXFP4 元数据和 AttnRes 边界。

每个生成目录都包含：

- `config.json`：可由原生模型实现加载的缩放配置
- `fidelity-report.md`：保留、缩放和警告项
- `pocketinfer-manifest.json`：预算与生成过程的机器可读记录

<details>
<summary><strong>尝试启动 Kimi K3</strong></summary>

```bash
vllm serve ./models/Kimi-K3-dummy \
  --load-format dummy \
  --skip-tokenizer-init \
  --max-model-len 4096 \
  --enforce-eager
```

这条命令尚未完成 GPU 运行验证，因此不作为已跑通结果展示。

</details>

## 生成自己的测试模型

PocketInfer 将总预算拆成三部分：

```text
可用于权重的预算 = 显存预算 - KV Cache 预算 - 运行时预留
```

`balanced` 会优先覆盖更多架构特征；`kernel` 会优先保留接近原模型的局部注意力头和专家形状。下面以 GLM-5.2 为例：

```bash
uv sync --extra dev

uv run pocketinfer scale ./models/GLM-5.2-dummy/config.json.ori \
  --output-dir ./models/GLM-5.2-dummy \
  --memory-budget 28GiB \
  --kv-cache-budget 4GiB \
  --runtime-reserve 5GiB \
  --max-model-len 4096 \
  --profile kernel \
  --force
```

替换源配置、输出目录和预算即可生成其他测试模型。`--max-model-len` 只控制运行规划，不会篡改源模型声明的最大上下文长度。所有显存估算都是静态规划值，不是 GPU 峰值实测。

<details>
<summary><strong>复现 Kimi K3 内置范例</strong></summary>

```bash
uv run pocketinfer scale ./models/Kimi-K3-dummy/config.json.ori \
  --output-dir ./models/Kimi-K3-dummy \
  --memory-budget 28GiB \
  --kv-cache-budget 4GiB \
  --runtime-reserve 5GiB \
  --max-model-len 4096 \
  --profile balanced \
  --force
```

</details>

<details>
<summary><strong>启用 tokenizer 和文本 API</strong></summary>

`--skip-tokenizer-init` 只适合引擎启动测试。要通过 OpenAI API 发送文本，需要把 tokenizer 元数据下载到模型目录，并在启动时去掉该参数。

```bash
uvx hf download zai-org/GLM-5.2 \
  tokenizer.json tokenizer_config.json chat_template.jinja \
  --local-dir ./models/GLM-5.2-dummy

uvx hf download moonshotai/Kimi-K3 \
  --include "*.py" --include "*.json" --include "*.model" \
  --exclude "config.json" --exclude "*.safetensors*" --exclude "assets/*" \
  --local-dir ./models/Kimi-K3-dummy
```

</details>

## 验证边界

PocketInfer 适合验证模型代码接入、配置约束、运行时初始化，以及注意力、MoE、量化、缓存等路径能否被框架触达。它不会复现权重内容、分布式通信规模、所有设备上的精确内核选择、模型质量或生产性能。

仓库 CI 在 Python 3.11/3.12 上执行代码检查、单元测试、构建和安装后的命令行冒烟测试：

```bash
./scripts/ci.sh
```

CI 结果与 GPU 运行证据分开记录，避免把“配置可生成”误写成“模型已在 GPU 上跑通”。

[设计说明](docs/design.zh-CN.md) · [模型支持](docs/model-support.zh-CN.md) · [CI 范围](docs/ci.zh-CN.md) · [贡献指南](CONTRIBUTING.zh-CN.md) · [发布清单](docs/release.zh-CN.md)
