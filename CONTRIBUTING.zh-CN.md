# 贡献指南

[English](CONTRIBUTING.md)

PocketInfer adapter 是可执行的架构契约。新增 adapter 必须：

1. 匹配明确的 `model_type` 或架构签名，并且 fail closed。
2. 枚举合法候选，而不是机械缩小每一个整数。
3. 重建所有派生字段，特别是逐层调度列表。
4. 在 manifest 中声明保留的不变量和无法避免的失真。
5. 不下载 checkpoint 即可估算参数量和权重字节数。
6. 为模型识别、合法 shape、预算约束和代表性官方配置增加聚焦测试。

模型族逻辑放在 `src/pocketinfer/adapters/`。通用引擎和 CLI 不应出现不断增长的
模型名 `if`/`elif` 链。

提交前执行：

```bash
./scripts/ci.sh
```

不要用静态估算声称性能保真。只有在提供模型构造和加速卡 smoke test 结果
后，才能声称配置已在对应硬件上验证可运行。
