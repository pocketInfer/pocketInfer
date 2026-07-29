# PocketInfer fidelity report

- Adapter: `kimi-k3`
- Profile: `balanced`
- Selected shape: layers=13, attention_heads=12, experts=32, top_k=16
- Estimate: 19.09B parameters, 18.20 GiB weights
- Budget: 28.00 GiB total, 4.00 GiB KV cache, 5.00 GiB runtime reserve
- Max model length: 4096

## Shape changes

- `text_config.linear_attn_config.num_heads`: `96` → `12`
- `text_config.num_attention_heads`: `96` → `12`
- `text_config.num_experts`: `896` → `32`
- `text_config.num_hidden_layers`: `93` → `13`
- `text_config.num_key_value_heads`: `96` → `12`

## Topology changes

- `text_config.linear_attn_config.full_attn_layers`: 24 entries ([4, 8, 12, 16, 20, 24, 28, 32] … [84, 88, 92, 93]) → 4 entries [4, 8, 12, 13]
- `text_config.linear_attn_config.kda_layers`: 69 entries ([1, 2, 3, 5, 6, 7, 9, 10] … [87, 89, 90, 91]) → 9 entries [1, 2, 3, 5, 6, 7, 9, 10, 11]

## Preserved

- hidden_size
- KDA head_dim and convolution width
- Q/KV LoRA ranks and MLA dimensions
- LatentMoE hidden and intermediate sizes
- top-k routing
- SiTU activation
- MXFP4/NVFP4 group format when present
- vision tower

## Warnings

- Static estimate only; validate allocator peak and selected kernels.
- Single-device execution does not reproduce distributed collectives.
- Local expert count differs from the reference EP plan.
