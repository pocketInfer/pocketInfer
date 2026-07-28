# PocketInfer MVP 设计

[English](design.md)

## 决策

PocketInfer 是面向推理开发的确定性 config 编译器。它不训练或蒸馏模型，也
不承诺输出质量。它在缩小离散容量轴的同时，尽量保留指定架构属性和每个
rank 的局部 kernel shape，直到满足声明的显存预算。

该方法普遍适用于由 config 描述的开源 Transformer、MoE、MLA、DSA、KDA
和混合模型，但并非万能。闭源模型、config 中没有表达关键语义的架构，以及
写死 shape 的专有 kernel，都需要新 adapter 或无法支持。

## 架构

```text
config.json + 资源预算 + 保真策略
                    |
                    v
              模型族 adapter
       候选配置 + 不变量 + 静态估算
                    |
                    v
           通用过滤与排序引擎
                    |
                    v
        config.json + 可解释 manifest
```

通用引擎不包含模型名分支。adapter 负责模型语义、候选轴、派生列表、估算和
保真评分。未知模型 fail closed。

## MVP 范围

- 只接受本地 JSON，不运行 remote code，也不下载 checkpoint。
- 单设备资源预算。
- `balanced` 和 `kernel` 两种保真 profile。
- Kimi K3 和 GLM-5.2 adapter。
- 静态参数量、权重和缓存估算。
- 确定性的 JSON 输出和机器可读变更 manifest。

## 失败模式

静态估算不包含 allocator 峰值、kernel repacking、CUDA Graph 和分布式
通信。生成配置必须先通过目标推理引擎模型构造，再通过真实加速卡 smoke
test。manifest 应明确这些限制，不能把估算包装成部署实测。
