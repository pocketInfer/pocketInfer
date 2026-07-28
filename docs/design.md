# Design

[简体中文](design.zh-CN.md)

```text
config + resource budget + profile
              |
         family adapter
  legal candidates / derived fields
              |
      filter, score, explain
              |
       config + manifest
```

The core handles budget filtering and deterministic ranking. Each adapter owns
model semantics: legal dimensions, layer schedules, memory estimates, and
fidelity warnings. Unknown models fail closed.

MVP choices:

- Enumerate small discrete candidate sets; no Z3 or generic constraint DSL.
- Preserve native Hugging Face configs; no vLLM-side mini-model branch.
- Keep vLLM optional; static compilation must work without an engine install.
- Treat GPU dispatch and profiling as runtime evidence, not compiler claims.

See [ADR 0001](adr/0001-adapters-and-constraints.md) for the decision record.
