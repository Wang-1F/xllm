# MTGR Bundled Artifacts

This directory bundles a small set of MTGR experiment assets directly into the
repository so another checkout can rerun the same workflows without hunting
through `/tmp` paths from earlier benchmark windows.

## Datasets

- `datasets/requests_access24_growth_5000_user10x_short_heavy_mtfm3seg_seed20260605.jsonl`
  - 5k access24 short-heavy 3-segment MTFM dataset used in the historical
    `1B 3seg access24` xLLM / vLLM / torch comparisons.
  - Original source:
    `/tmp/mtgr_e2e_full_sizes_mtfm_vllm_torch_xllm_worker019e8314_20260608/datasets/requests_access24_growth_5000_user10x_short_heavy_mtfm3seg_seed20260605.jsonl`

## Model directories

- `models/model_kv16_60layer_xllm`
  - Original source:
    `/tmp/mtgr_e2e_full_sizes_mtfm_vllm_torch_xllm_worker019e8314_20260608/models/model_kv16_60layer_xllm`
- `models/model_kv16_36layer_xllm`
  - Original source:
    `/tmp/mtgr_e2e_full_sizes_mtfm_vllm_torch_xllm_worker019e8314_20260608/models/model_kv16_36layer_xllm`
- `models/model_kv16_90layer_xllm`
  - Original source:
    `/tmp/mtgr_e2e_full_sizes_mtfm_vllm_torch_xllm_worker019e8314_20260608/models/model_kv16_90layer_xllm`
- `models/model_kv16_legacy`
  - Original source:
    `/tmp/xllm_full_cache_50000_hit_checkpoints_kv16_worker019e82b4_20260601/model_kv16`

These bundled model directories are lightweight metadata/tokenizer snapshots,
not full multi-GB model weights.
