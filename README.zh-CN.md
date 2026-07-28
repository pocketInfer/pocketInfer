# PocketInfer

[English](README.md)

PocketInfer 把超大模型的配置编译成受显存预算约束、尽量保持架构特征的小型
配置，用于推理引擎开发、功能调试和 CI fixture。

它不是蒸馏工具，也不保留模型输出质量。目标是在不加载原始万亿参数权重的
情况下，继续覆盖同一个模型实现、缓存机制、路由逻辑和 kernel 家族。

## 为什么需要它

机械地缩小 `config.json` 中的所有整数，很容易改变 kernel shape，甚至悄悄
绕开真正需要测试的代码路径。PocketInfer 使用显式的模型 adapter 保存架构
不变量；遇到未知模型时直接拒绝，不猜测配置。

当前 MVP 支持：

- Kimi K3：KDA/MLA 比例、AttnRes、LatentMoE、SiTU、top-k 和 MXFP4/NVFP4
  布局。
- GLM-5.2：DSA 维度、IndexShare 相位、dense-to-MoE 过渡和 MTP。

## 安装

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

## 生成配置

先下载模型官方 `config.json`，然后声明资源预算，而不是指定某种 GPU：

```bash
pocketinfer scale ./Kimi-K3/config.json \
  --output-dir ./out/kimi-k3 \
  --memory-budget 32GiB \
  --kv-cache-budget 6GiB \
  --runtime-reserve 6GiB \
  --max-model-len 4096 \
  --profile balanced \
  --reference-tp 8 \
  --reference-ep 16
```

输出：

- `config.json`：生成的标准 Hugging Face 模型配置。
- `pocketinfer-manifest.json`：预算、静态估算、字段变化、保留的不变量和
  已知失真。

求解器同时约束权重空间，以及至少一条 `--max-model-len` 请求所需的最低
缓存空间。剩余 KV 空间决定并发能力。

`balanced` 优先保留有意义的拓扑边界；`kernel` 优先保留参考 TP/EP 方案中
每个 rank 的 attention head 和 expert shape。

在总显存 32 GiB、KV cache 和 runtime 各预留 6 GiB 的示例中：

| Adapter | Profile | 层数 / heads / experts | 参数量 | 权重估算 |
| --- | --- | --- | --- | --- |
| Kimi K3 | balanced | 13 / 12 / 32 | 19.1B | 18.20 GiB |
| GLM-5.2 | balanced | 11 / 8 / 16 | 8.8B | 16.45 GiB |

这些是编译器静态估算，不是 GPU 峰值显存实测。

## 配合 vLLM 使用

把 tokenizer 和模型侧配置文件复制到生成目录，然后执行：

```bash
vllm serve ./out/kimi-k3 \
  --load-format dummy \
  --max-model-len 4096 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --kv-cache-memory-bytes 6G \
  --moe-backend auto \
  --enforce-eager
```

需要 CUDA Graph 或性能 profiling 时去掉 `--enforce-eager`。

## 静态估算不能证明什么

估算不包含 allocator 峰值、backend repacking、CUDA Graph 内存池、临时
activation 和分布式通信。生成配置只有在完成目标推理引擎的模型构造和真实
加速卡 smoke test 后，才能称为“已验证可运行”。

单卡运行也不能还原 TP/EP collective。精确研究分布式瓶颈，需要未来增加
trace replay，而不只是缩小 config。

## 开发

```bash
./scripts/ci.sh
```

这是 CPU-only 验证：在 Python 3.11/3.12 上执行 lint、format、单元和 golden
测试、构建发行包及 CLI smoke。它不证明 GPU 模型构造或 kernel 执行。
详见[CI 与验证](docs/ci.zh-CN.md)。

更多资料：[设计](docs/design.zh-CN.md)、
[模型支持边界](docs/model-support.zh-CN.md)、
[贡献指南](CONTRIBUTING.zh-CN.md)、
[开源发布交接](docs/release.zh-CN.md)及
[ADR 0001](docs/adr/0001-adapters-and-constraints.md)。
