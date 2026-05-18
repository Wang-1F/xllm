# MTGR Attention Test Harness Prototype

This directory is the compact harness layout for MTGR attention operator
development tests. It is wired into CMake through the CUDA tests target list,
while older historical benchmark entrypoints remain outside this directory
until they are explicitly retired.

## Scope

- Base backend: full logical `mask_build + one_stage FlashInfer`.
- Candidate backend: Hopper unified ragged segment attention through the same
  production-facing `AttentionMetadata + query/key/value + KVCache` semantics.
- Supported data mode in this prototype: BF16, no GQA, no_match and
  partial_real_time_match.
- Host performance standard: Nsight Systems NVTX ranges only.

## Files

- `mtgr_attention_contract.h`
  - Public test contract.
  - Harness metadata, case data, backend interface, benchmark label schema.
  - Backend `forward(...)` intentionally mirrors the production
    `MTGRAttentionImpl::forward(...)` signature.
- `mtgr_attention_test_utils.cpp`
  - Case generation, tensor/cache/metadata construction.
  - Precision diff and CSV label helpers.
- `mtgr_attention_backends.cpp`
  - Backend interface implementations only.
  - `FullFlashinferBaseBackend`: full logical `mask_build + one_stage FlashInfer`.
  - `HopperUnifiedBackend`: production-style Hopper unified ragged segment attention.
- `mtgr_attention_e2e_test.cpp`
  - Two stable gtest entrypoints:
    - `Precision`
    - `PerfNvtxCsv`
- `mtgr_attention_product_e2e_test.cpp`
  - Production-facing `MTGRAttentionImpl::forward(...)` precision and match-mode
    regression tests.
- `mtgr_attention_torch_reference.h`
  - Torch full-mask attention reference for product-facing precision checks.
- `mtgr_attention_nvtx_report.py`
  - Converts Nsight sqlite + labels into per-case CSV and summary markdown.

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
