# Release checklist

[简体中文](release.zh-CN.md)

Before publishing:

- Choose the final repository, package, and CLI names.
- Add real project URLs, maintainers, and a private security contact.
- Confirm Apache-2.0 and review every adapter rule.
- Push `main`, enable Actions, and require the two CI matrix jobs.
- Run `./scripts/ci.sh` from a clean clone.

Before the first tag:

- Check the final package name on PyPI.
- Keep K3/GLM runtime status `experimental` until a GPU smoke test records the
  vLLM commit, device, driver, command, manifest, and logs.
- Build from the tagged commit with `uv build`; publish only those artifacts.
- Preserve AI-assistance attribution and complete human review.
