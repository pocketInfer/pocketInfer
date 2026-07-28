# Open-source release handoff

[简体中文](release.zh-CN.md)

The implementation is ready for repository publication, but the working name
and maintainer metadata are intentionally not treated as final.

## Before creating the public repository

- Choose the final repository, Python package, and CLI names.
- Update `project.name`, the console script, package imports, and documentation
  together if the name changes.
- Add real `project.urls`, authors, maintainers, and a security contact to
  `pyproject.toml` and `SECURITY.md`.
- Confirm that Apache-2.0 is the intended license.
- Review every generated adapter rule and static-memory assumption.

## Publish the repository

1. Create an empty public repository without generated README or license files.
2. Add it as the Git remote and push `main`.
3. Enable GitHub Actions and require the `ci` matrix before merging.
4. Add issue and pull-request templates only after the maintainer workflow is
   known; generic templates add noise.
5. Run `./scripts/ci.sh` from a clean clone.

## First release

- Keep Kimi K3 and GLM-5.2 runtime status `experimental` until a GPU smoke test
  records the vLLM commit, accelerator, driver, command, manifest, and logs.
- Do not describe static estimates as measured peak memory or performance.
- Check the final package name on PyPI before tagging.
- Build from the release commit with `uv build`; publish only those artifacts.
- Tag the release after hosted CI passes.

The current Git history contains AI-assistance attribution. Preserve it and
state the review and validation performed by the human maintainer.
