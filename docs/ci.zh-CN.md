# CI 与验证

[English](ci.md)

## 基础 CI

每个 pull request 和向 `main` 的 push，都会在普通 GitHub hosted CPU runner
上使用 Python 3.11、3.12 执行 `scripts/ci.sh`。

| 检查项 | 是否覆盖 |
| --- | --- |
| Lint 和格式检查 | 是 |
| 单元测试和 golden config 测试 | 是 |
| 确定性的预算与不变量检查 | 是 |
| sdist 和 wheel 构建 | 是 |
| CLI 启动 | 是 |
| vLLM 模型构造 | 否 |
| GPU kernel 分派与执行 | 否 |
| 性能保真 | 否 |

本地运行完全相同的检查：

```bash
./scripts/ci.sh
```

设置 `UV_PYTHON=3.11` 可以选择另一个受支持的 Python 版本。

## 证据等级

报告和 release note 应明确使用以下等级：

- `compiler-tested`：静态编译器测试通过。
- `vllm-config-validated`：生成配置被指定版本或 commit 的 vLLM 接受。
- `gpu-runtime-validated`：模型在注明的加速卡上完成构造和执行。
- `measured`：profiling 或显存数字来自附带运行产物的实测。

不能用较弱的证据推导较强结论。在可选 GPU 验证实现前，release 应将
Kimi K3 和 GLM-5.2 的 runtime 支持标记为 experimental。

## 未来的可选 GPU 验证

GPU 验证不会成为 pull request 的强制门禁。未来可以提供本地命令，执行
dummy 模型构造、prefix-cache 行为检查、backend 选择断言和 profiler 采集。
社区提交的验证结果必须记录 engine commit、设备、驱动、命令、生成的
manifest 和日志。
