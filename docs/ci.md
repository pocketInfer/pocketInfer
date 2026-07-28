# CI scope

[简体中文](ci.zh-CN.md)

Run locally:

```bash
./scripts/ci.sh
```

GitHub Actions runs the same command on Python 3.11 and 3.12.

| Evidence | CI status |
| --- | --- |
| Lint, formatting, unit/golden tests | Covered |
| sdist, wheel, installed CLI smoke | Covered |
| vLLM config/model construction | Not run |
| GPU kernel execution and profiling | Not run |

Use precise claims:

- `compiler-tested`: this CI passed.
- `vllm-config-validated`: a named vLLM version accepted the config.
- `gpu-runtime-validated`: a named accelerator executed it.
- `measured`: logs or profiles are attached.

Never promote one level to the next without evidence.
