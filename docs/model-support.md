# Model support

[简体中文](model-support.zh-CN.md)

## Implemented

| Model | Adapter | Bundled example | Evidence |
| --- | --- | --- | --- |
| Kimi K3 | `kimi-k3` | 13 layers / 12 heads / 32 experts, MXFP4 | compiler-tested; GPU runtime pending |
| GLM-5.2 | `glm-5.2` | 11 layers / 8 heads / 16 experts, BF16 | GPU startup validated with vLLM 0.26.0 |

Unknown architectures are rejected. The table below is an expansion
assessment, not a claim that those families are already supported.

## Expansion fit

| Family | Expected fit | Adapter must preserve |
| --- | --- | --- |
| Dense Transformer | High | head divisibility, RoPE, tied weights |
| Conventional MoE | High | top-k, shared experts, TP/EP divisibility |
| MLA/KDA/DSA hybrid | Medium-high | latent ranks, schedules, cache state |
| SSM/recurrent hybrid | Medium | state width, convolution, layer schedule |
| Multimodal | Medium | language tower, modality tower, projector |
| Closed or hard-coded model | Low | often unavailable |

An adapter is required when config fields have model-specific semantics or
derived per-layer state. Field-name heuristics are not used.

MVP anchors:

- [Kimi K3 config](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)
- [GLM-5.2 config](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json)

Adding a family means adding one adapter, memory estimates, fidelity warnings,
and focused tests—not model-name branches in the solver.
