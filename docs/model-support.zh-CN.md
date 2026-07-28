# 模型支持边界

[English](model-support.md)

config 缩放方法可以复用，但不同模型的语义无法完全通用。核心求解器可以被
所有模型族共享；特殊架构需要小型 adapter，声明合法维度、派生字段、不变量
和估算方式。

| 模型族 | 适用程度 | 必须保留的属性 |
| --- | --- | --- |
| Dense Transformer | 高 | head 整除关系、位置编码、权重绑定 |
| 常规 MoE | 高 | top-k、共享专家、expert/TP/EP 整除关系 |
| MLA、KDA 或 DSA 混合模型 | 中高 | latent rank、attention 调度、缓存状态 |
| Recurrent 或 SSM 混合模型 | 中 | state 宽度、卷积 shape、层调度 |
| 多模态模型 | 中 | 语言模型、模态塔和 projector 契约 |
| 写死 shape 或闭源模型 | 低或不支持 | 编译器无法获得语义和合法 shape |

如果模型包含逐层派生列表、自定义缓存状态、非标准 projection、对拓扑敏感
的 expert routing，或者无法从字段名安全推断的 kernel 约束，就必须编写
adapter。未知架构直接拒绝，不做启发式 JSON 修改。

MVP 锚点：

- [Kimi K3 官方配置](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)：
  KDA/MLA 调度、AttnRes、LatentMoE、SiTU 和 routed-expert 量化。
- [GLM-5.2 官方配置](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json)：
  DSA、IndexShare 相位、dense-to-MoE 过渡和 MTP。

增加一个模型族通常只需实现一个 adapter，不应向 CLI 或核心求解器增加模型名
分支。参见[贡献指南](../CONTRIBUTING.zh-CN.md)。
