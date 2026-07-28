# Contributing

[简体中文](CONTRIBUTING.zh-CN.md)

PocketInfer adapters are executable architecture contracts. A new adapter must:

1. Match a precise `model_type` or architecture signature and fail closed.
2. Enumerate legal candidates instead of blindly scaling every integer.
3. Rebuild all derived fields, especially per-layer schedules.
4. State preserved invariants and unavoidable fidelity losses in the manifest.
5. Estimate parameters and weight bytes without downloading a checkpoint.
6. Add focused tests for detection, legal shapes, budget compliance, and one
   representative official configuration.

Keep model-family logic in `src/pocketinfer/adapters/`. The generic engine and
CLI must not grow model-name `if`/`elif` chains.

Before submitting:

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run pytest -v
uv build
```

Do not claim performance fidelity from static estimates. Include model
construction and accelerator smoke results when making a runnable-hardware
claim.
