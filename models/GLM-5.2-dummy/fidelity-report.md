# PocketInfer fidelity report

- Adapter: `glm-5.2`
- Profile: `kernel`
- Selected shape: layers=11, attention_heads=8, experts=16, top_k=8
- Estimate: 8.83B parameters, 16.45 GiB weights
- Budget: 28.00 GiB total, 4.00 GiB KV cache, 5.00 GiB runtime reserve
- Max model length: 4096

## Shape changes

- `n_routed_experts`: `256` → `16`
- `num_attention_heads`: `64` → `8`
- `num_hidden_layers`: `78` → `11`
- `num_key_value_heads`: `64` → `8`

## Topology changes

- `indexer_types`: 78 entries (full×21, shared×57) → 11 entries (full×5, shared×6)
- `mlp_layer_types`: 78 entries (dense×3, sparse×75) → 11 entries (dense×3, sparse×8)

## Preserved

- hidden size and MLA ranks
- DSA index head dimensions and top-k
- IndexShare frequency and phase
- three dense warm-up layers
- MoE intermediate size and top-k routing
- MTP layer
- rope configuration

## Warnings

- Static estimate only; DSA cache and backend must be validated in vLLM.
- Single-device execution does not reproduce distributed collectives.
