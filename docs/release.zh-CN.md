# 发布清单

[English](release.md)

公开仓库前：

- 确定最终仓库名、包名和 CLI 名。
- 填写真实项目 URL、维护者和私密安全联系方式。
- 确认 Apache-2.0，并人工检查每条 adapter 规则。
- 推送 `main`，启用 Actions，把两个 CI matrix job 设为门禁。
- 在干净 clone 中执行 `./scripts/ci.sh`。

首个 tag 前：

- 检查最终包名是否可在 PyPI 使用。
- GPU smoke test 记录 vLLM commit、设备、驱动、命令、manifest 和日志前，
  K3/GLM runtime 保持 `experimental`。
- 从 tag 对应 commit 执行 `uv build`，只发布该次产物。
- 保留 AI assistance attribution，并完成人工审查。
