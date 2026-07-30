# 模型支持

[English](model-support.md)

## 已实现

| 模型 | Adapter | 内置范例 | 证据 |
| --- | --- | --- | --- |
| Kimi K3 | `kimi-k3` | 13 层 / 12 heads / 32 experts，MXFP4 | compiler-tested；待 GPU runtime 验证 |
| GLM-5.2 | `glm-5.2` | 11 层 / 8 heads / 16 experts，BF16 | 已用 vLLM 0.26.0 验证 GPU 启动 |

未知架构直接拒绝。下表只是扩展适配性判断，不代表已经支持。

## 扩展适配性

| 模型族 | 预期适用程度 | adapter 必须保留 |
| --- | --- | --- |
| Dense Transformer | 高 | head 整除、RoPE、权重绑定 |
| 常规 MoE | 高 | top-k、共享专家、TP/EP 整除 |
| MLA/KDA/DSA 混合 | 中高 | latent rank、层调度、缓存状态 |
| SSM/recurrent 混合 | 中 | state 宽度、卷积、层调度 |
| 多模态 | 中 | 语言塔、模态塔、projector |
| 闭源或写死 shape | 低 | 通常无法安全获得约束 |

字段具有模型特定语义或逐层派生状态时，必须写 adapter，不按字段名猜配置。

MVP 锚点：

- [Kimi K3 配置](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)
- [GLM-5.2 配置](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json)

扩展模型族应增加 adapter、显存估算、保真警告和聚焦测试，不应向 solver
增加模型名分支。
