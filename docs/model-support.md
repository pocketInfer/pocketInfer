# Model support

[简体中文](model-support.zh-CN.md)

| Family | Fit | Adapter must preserve |
| --- | --- | --- |
| Dense Transformer | High | head divisibility, RoPE, tied weights |
| Conventional MoE | High | top-k, shared experts, TP/EP divisibility |
| MLA/KDA/DSA hybrid | Medium-high | latent ranks, schedules, cache state |
| SSM/recurrent hybrid | Medium | state width, convolution, layer schedule |
| Multimodal | Medium | language tower, modality tower, projector |
| Closed or hard-coded model | Low | often unavailable |

An adapter is required when config fields have model-specific semantics or
derived per-layer state. Unknown architectures are rejected; field-name
heuristics are not used.

MVP anchors:

- [Kimi K3 config](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)
- [GLM-5.2 config](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json)

Adding a family means adding one adapter and focused tests—not model-name
branches in the solver.
