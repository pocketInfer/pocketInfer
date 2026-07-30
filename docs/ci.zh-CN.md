# CI 范围

[English](ci.md)

本地执行：

```bash
./scripts/ci.sh
```

GitHub Actions 在 Python 3.11 和 3.12 上运行同一命令。

| 证据 | CI 状态 |
| --- | --- |
| Lint、format、单元/golden 测试 | 已覆盖 |
| sdist、wheel、安装后 CLI smoke | 已覆盖 |
| vLLM config/model 构造 | 未运行 |
| GPU kernel 执行和 profiling | 未运行 |

此表只描述自动化 CI。根 README 的 GLM-5.2 结果属于独立的人工 runtime 证据。

结论必须精确：

- `compiler-tested`：本 CI 通过。
- `vllm-config-validated`：指定 vLLM 版本接受配置。
- `gpu-runtime-validated`：指定加速卡成功执行。
- `measured`：附带日志或 profile 的实测。

没有证据，不得升级结论等级。
