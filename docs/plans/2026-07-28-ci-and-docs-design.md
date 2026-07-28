# CPU CI and bilingual documentation design

## Scope

The baseline CI validates the config compiler on ordinary GitHub-hosted CPU
runners. It does not install vLLM, construct GPU models, or claim kernel
execution coverage.

## Design

`scripts/ci.sh` is the single local and hosted entry point. It synchronizes the
locked development environment, checks lint and formatting, runs the test
suite, builds both package artifacts, and smoke-tests the CLI. GitHub Actions
runs the same script on Python 3.11 and 3.12 with read-only permissions and
concurrency cancellation.

User-facing documentation has equivalent English and Simplified Chinese entry
points. Both explain supported models, resource-based compilation, evidence
levels, CI coverage, local validation, and the boundary between static estimates
and GPU execution.

## Acceptance

- `scripts/ci.sh` succeeds locally.
- The workflow contains Python 3.11 and 3.12 matrix entries.
- English and Chinese README files link to one another.
- CI documentation explicitly marks GPU runtime as not run.
