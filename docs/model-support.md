# Model support boundary

[简体中文](model-support.zh-CN.md)

Config scaling is reusable, but model semantics are not fully generic. The core
solver can be shared across model families; each unusual architecture needs a
small adapter that declares legal dimensions, derived fields, invariants, and an
estimator.

| Model family | Suitability | What must be preserved |
| --- | --- | --- |
| Dense Transformer | High | head divisibility, positional encoding, tied weights |
| Conventional MoE | High | top-k, shared experts, expert/TP/EP divisibility |
| MLA, KDA, or DSA hybrids | Medium-high | latent ranks, attention schedule, cache state |
| Recurrent or SSM hybrids | Medium | state width, convolution shape, layer schedule |
| Multimodal models | Medium | language model plus modality tower and projector contracts |
| Hard-coded or closed models | Low or unsupported | semantics and legal shapes unavailable to the compiler |

An adapter is required when the model has derived per-layer lists, custom cache
state, nonstandard projections, topology-sensitive expert routing, or kernel
constraints that cannot be inferred safely from field names. Unknown
architectures fail closed rather than receiving heuristic JSON edits.

The MVP anchors are:

- [Kimi K3 official config](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json):
  KDA/MLA scheduling, AttnRes, LatentMoE, SiTU, and routed-expert quantization.
- [GLM-5.2 official config](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json):
  DSA, IndexShare phase, dense-to-MoE transition, and MTP.

Adding a family normally means implementing one adapter, not adding branches to
the CLI or solver. See [CONTRIBUTING.md](../CONTRIBUTING.md).
