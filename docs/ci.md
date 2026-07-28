# CI and validation

[简体中文](ci.zh-CN.md)

## Baseline CI

Every pull request and push to `main` runs `scripts/ci.sh` on Python 3.11 and
3.12 using ordinary GitHub-hosted CPU runners.

| Check | Covered |
| --- | --- |
| Lint and formatting | Yes |
| Unit and golden config tests | Yes |
| Deterministic budget and invariant checks | Yes |
| Source distribution and wheel build | Yes |
| CLI startup | Yes |
| vLLM model construction | No |
| GPU kernel dispatch and execution | No |
| Performance fidelity | No |

Run exactly the same checks locally:

```bash
./scripts/ci.sh
```

Set `UV_PYTHON=3.11` to select the other supported interpreter.

## Evidence levels

Reports and release notes should use explicit evidence:

- `compiler-tested`: static compiler tests passed.
- `vllm-config-validated`: the generated config was accepted by a named vLLM
  version or commit.
- `gpu-runtime-validated`: the model was constructed and executed on a named
  accelerator.
- `measured`: profiling or memory numbers came from an attached run artifact.

Do not infer a stronger level from a weaker one. Until optional GPU validation
exists, releases should mark Kimi K3 and GLM-5.2 runtime support as experimental.

## Future optional GPU validation

GPU validation is intentionally not a pull-request gate. A future local command
may run dummy model construction, prefix-cache behavior checks, backend
selection assertions, and profiler capture. Community-provided results must
record the engine commit, device, driver, command, generated manifest, and logs.
