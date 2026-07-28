#!/usr/bin/env bash
set -euo pipefail

python_version="${UV_PYTHON:-3.12}"

uv sync --python "$python_version" --extra dev --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest -v
uv build
uv run \
  --isolated \
  --no-project \
  --python "$python_version" \
  --with dist/*.whl \
  pocketinfer --help >/dev/null
