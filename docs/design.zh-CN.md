# 设计

[English](design.md)

```text
config + 资源预算 + profile
            |
        模型族 adapter
   合法候选 / 派生字段
            |
       过滤、评分、解释
            |
       config + manifest
```

核心只负责预算过滤和确定性排序。adapter 负责模型语义：合法尺寸、层调度、
显存估算和保真警告。未知模型直接拒绝。

MVP 决策：

- 枚举少量离散候选，不引入 Z3 或通用约束 DSL。
- 输出原生 Hugging Face config，不在 vLLM 内增加 mini-model 分支。
- vLLM 是可选集成；静态编译不依赖推理引擎安装。
- GPU dispatch 和 profiling 属于运行证据，不是编译器结论。

决策记录见 [ADR 0001](adr/0001-adapters-and-constraints.md)。
