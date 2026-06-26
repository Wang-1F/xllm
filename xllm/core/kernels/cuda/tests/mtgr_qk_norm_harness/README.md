# MTGR QK Norm Test Harness

This directory keeps the MTGR Q/K normalization operator tests in one compact
single-side harness. The harness mirrors the production operator contract while
allowing project-current and custom implementations to be compared with the same
inputs.

## Scope

- Data type: BF16.
- Primary fast path: `head_dim == 128`.
- Inputs: Q and K tensors shaped `[tokens, heads, head_dim]`, including sliced
  QKV views with non-trivial strides.
- Host performance standard: NVTX/Nsight ranges only.

## Files

- `mtgr_qk_norm_contract.h`
  - Public harness contract and backend factories.
- `mtgr_qk_norm_backends.cpp`
  - Backend implementations only.
  - `project_baseline`: project-current RMSNorm semantics.
  - `cuda_rms_norm`: generic CUDA RMSNorm path.
  - `strided_bf16_hd128_kernel`: production MTGR fast path.
- `mtgr_qk_norm_test_utils.cpp`
  - Test input generation, project-module reference, diff helpers.
- `mtgr_qk_norm_e2e_test.cpp`
  - Stable gtest entrypoints for precision and NVTX replay smoke.

## Design Rules

- Keep all implementations behind one contract.
- Do not keep duplicate test-only CUDA kernels when the production kernel exists.
- Keep test utilities separate from backend implementations.
- Keep performance conclusions tied to NVTX/Nsight, not host timers.
