# MTGR Attention Test Harness Prototype

This directory is the compact harness layout for MTGR attention operator
development tests. It is wired into CMake through the CUDA tests target list.

## Scope

- Base backends:
  - full logical `mask_build + one_stage FlashInfer`
  - FlashInfer block-sparse custom-mask prefill using MTGR BSR block pattern
  - Python FlexAttention block-mask baseline
- Candidate backend: Hopper unified ragged segment attention through the same
  production-facing `AttentionMetadata + query/key/value + KVCache` semantics.
- Supported data mode in this prototype: BF16, no GQA, no_match and
  partial_real_time_match.
- Host performance standard: Nsight Systems NVTX ranges only.

## Files

- `mtgr_attention_contract.h`
  - Public test contract.
  - Harness metadata, case data, backend interface, product-style test helpers,
    benchmark label schema.
  - Backend `forward(...)` intentionally mirrors the production
    `MTGRAttentionImpl::forward(...)` signature.
- `mtgr_attention_backends.cu`
  - Backend interface implementations.
  - `FullFlashinferBaseBackend`: full logical `mask_build + one_stage FlashInfer`.
  - `BlockSparseFlashinferBaseBackend`: block-level MTGR BSR mask plus
    FlashInfer paged prefill `paged_run`.
  - `HopperUnifiedBackend`: production-style Hopper unified ragged segment attention.
  - Contains the harness-local packed mask CUDA kernels used by the FlashInfer
    bases.
- `mtgr_attention_test_utils.cpp`
  - Case generation, tensor/cache/metadata construction.
  - Torch full-mask precision reference, precision diff, CSV label helpers.
- `mtgr_attention_e2e_test.cpp`
  - Stable harness gtest entrypoints:
    - `Precision`
    - `PerfNvtxCsv`
    - Product-facing `MTGRAttentionImpl::forward(...)` precision and match-mode
      regression tests.
- `mtgr_flex_attention_baseline.py`
  - Python-only PyTorch FlexAttention baseline.
  - Emits the same label schema as the C++ harness, with backend
    `flex_attention_base`.
  - Supports `--input-labels` to reuse shape metadata from a C++ harness labels
    CSV and keep request shapes aligned across baselines.

The MTGR attention NVTX report helper lives outside this harness at
`../profiling/mtgr_attention_nvtx_report.py`.

## Design Rules

- Keep the gtest entrypoints stable; add new implementations as backend classes.
- Keep one deterministic case generator shared by precision and performance.
- Keep benchmark labels separate from NVTX data; Python joins them after Nsight
  export.
- Keep `chrono` and `cudaEvent` out of host performance conclusions.
- Keep benchmark-only full logical input and production-style live input as two
  `MTGRAttentionInput` values inside `MTGRAttentionCaseData`.
- Use production `MTGRAttentionImpl` directly in product-facing tests instead
  of adding a second test-only adapter around the same public interface.
- Keep this directory at four code files: `contract.h`, `backends.cu`,
  `test_utils.cpp`, and `e2e_test.cpp`. Python profiling baselines may live
  alongside them when they are intentionally not part of the C++ build.

## FlexAttention Baseline

Example smoke run:

```bash
python xllm/core/kernels/cuda/tests/mtgr_attention_harness/mtgr_flex_attention_baseline.py \
  --pairs 1 \
  --warmup 0 \
  --labels /tmp/mtgr_flex_attention_smoke_labels.csv
```

Example using the exact shape metadata from a C++ harness labels CSV:

```bash
python xllm/core/kernels/cuda/tests/mtgr_attention_harness/mtgr_flex_attention_baseline.py \
  --input-labels /tmp/mtgr_attention_harness_block_sparse_odd_random_1000_labels.csv \
  --labels /tmp/mtgr_flex_attention_odd_random_1000_labels.csv
```
