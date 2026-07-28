# 开源发布交接

[English](release.md)

代码已经可以创建公开仓库，但当前项目名和维护者 metadata 仍被有意保留为
待确认状态。

## 创建公开仓库之前

- 确定最终的仓库名、Python 包名和 CLI 名。
- 如果改名，同时更新 `project.name`、console script、包 import 和文档。
- 在 `pyproject.toml` 和 `SECURITY.md` 中填写真实的 `project.urls`、作者、
  维护者和安全联系方式。
- 确认 Apache-2.0 是最终采用的许可证。
- 人工检查每一条 adapter 规则和静态显存假设。

## 发布仓库

1. 创建空的公开仓库，不要让平台自动生成 README 或 LICENSE。
2. 添加 Git remote，并推送 `main`。
3. 启用 GitHub Actions，把 `ci` 矩阵设为合并门禁。
4. 等维护流程明确后再增加 issue/PR 模板，避免无效模板制造噪声。
5. 在干净 clone 中执行 `./scripts/ci.sh`。

## 第一个 release

- 在 GPU smoke test 记录 vLLM commit、加速卡、驱动、命令、manifest 和日志
  之前，Kimi K3 和 GLM-5.2 的 runtime 状态保持 `experimental`。
- 不得把静态估算描述成峰值显存或性能实测。
- 打 tag 前检查最终包名是否可在 PyPI 使用。
- 从 release commit 执行 `uv build`，只发布该次构建产物。
- hosted CI 通过后再创建 release tag。

当前 Git 历史已包含 AI assistance attribution。请保留该信息，并由人类维护者
说明自己完成的代码审查和验证。
