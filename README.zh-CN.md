# PocketInfer

[English](README.md)

**让超大模型的推理开发，不再从下载巨型权重开始。**

把 Kimi K3、GLM-5.2 这样的超大模型接入 vLLM 时，你往往不得不先下载数百 GB 甚至数 TB 的权重，只为确认几个工程问题：模型能否构造，后端能否正确选择，KV Cache 能否分配，MoE、量化和注意力路径能否工作。

PocketInfer 把这一步变轻：输入官方 Hugging Face 配置和显存预算，输出一份保留关键架构约束的精简配置。它仍然走 vLLM 的原生模型实现；配合占位权重，即可在有限资源上测试模型接入和运行时初始化。

> [!IMPORTANT]
> 它缩小的是工程测试规模，不是模型能力。PocketInfer 不是量化、蒸馏或模型压缩，也不会生成可用于效果评测的权重。

## 一分钟验证 GLM-5.2

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

## 内置模型

| 范例 | 生成规模 | 验证状态 |
| --- | --- | --- |
| [GLM-5.2](models/GLM-5.2-dummy)（`kernel`） | 11 层、8 个注意力头、16 个专家、BF16 | vLLM 0.26.0，单卡 32 GB 已启动 |
| [Kimi K3](models/Kimi-K3-dummy)（`balanced`） | 13 层、12 个注意力头、32 个专家、top-16、MXFP4 | 配置生成与一致性测试通过，GPU 运行验证待完成 |

这两个目录只是可直接使用的范例，不对应固定的硬件档位。显存预算、模型规模和缩放策略都可以重新指定。

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
