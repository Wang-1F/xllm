# MTGR Ragged Segment Attention TODO

## Latest Handoff - 2026-05-09

Status: the ragged segment attention path is implemented as a CUDA test/perf
experiment and should now be considered "performance converged enough"; do not
keep chasing head_dim=128 perf unless explicitly asked. The current priority is
precision correctness and long-run stability.

Container/workspace:

- Repo: `/export/home/wangyifan/Code/xllm`
- Container for build/test: `wangyifan-cuda-x86`
- Use container commands for builds/tests, e.g. `docker exec wangyifan-cuda-x86 bash -lc 'cd /export/home/wangyifan/Code/xllm && ...'`
- The workspace is dirty and has unrelated generated artifacts; do not revert
  unrelated files.

Current code changes to preserve:

- `xllm/core/kernels/cuda/tests/mtgr_fused_attention_kernel.cu`
  - Ragged segment attention CUDA kernels/wrappers.
  - q64/reg-frag fast path for head_dim 64/128 on SM80+.
  - Static segment-count dispatch now includes the four-segment case used by
    the four-segment comparison.
  - head_dim=128 now defaults to the split-diag path via
    `use_mtgr_ragged_hd128_q64_regfrag_split_diag_experimental()`.
    `XLLM_MTGR_CUDA_RAGGED_HD128_Q64_REGFRAG_SPLIT_DIAG=0` can be used to
    disable it and return to inline diagonal merge.
  - Split-diag writes prefix output/LSE in the main q64 kernel and runs
    `mtgr_ragged_segment_diag_merge_kernel` for diagonal rows. This was kept
    because it reduced head_dim=128 register pressure enough to narrow the
    perf gap.
- `xllm/core/kernels/cuda/tests/mtgr_attenion_test.cpp`
  - Ragged batch-layout/test wrapper plumbing.
  - The no-match/partial batch-layout caches now include value hashes for CPU
    length tensors in addition to pointer keys. This avoids stale cache hits
    when CPU tensor addresses are reused across random benchmark cases.
    Without this, long split-diag runs can abort with
    `query.size(0) == layout.total_q` mismatches in
    `run_fused_no_match_batched`.
- `xllm/core/kernels/cuda/tests/mtgr_attention_one_vs_multi_stage_perf_test.cpp`
  - `RaggedVsFourSegmentOddLengthRandomByBatchSizeCsv`, comparing the previous
    four-segment fused no-match path (`segment_rules=0:1:0:2`) against the new
    ragged segment path on the same packed input.
  - `RaggedVsOneStageOddLengthRandomByBatchSizeCsv`, comparing the new ragged
    packed path against the request-by-request `mask_build + one-stage
    FlashInfer` baseline for batch sizes 1, 2, 3, and 4.

Validation already completed after final changes:

- Build:
  - Command:
    `docker exec wangyifan-cuda-x86 bash -lc 'cd /export/home/wangyifan/Code/xllm && cmake --build build/cmake.linux-x86_64-cpython-312 --target mtgr_attention_one_vs_multi_stage_perf_test -j 16'`
  - Result: passed. Only existing unused-variable/function CUDA warnings.
- Default precision smoke:
  - Command:
    `CUDA_VISIBLE_DEVICES=0 XLLM_MTGR_ATTENTION_WARMUP=1 XLLM_MTGR_ATTENTION_REPEAT=1 XLLM_MTGR_ATTENTION_RAGGED_VS_FOUR_SEGMENT_GROUPS=100 XLLM_MTGR_ATTENTION_RAGGED_VS_FOUR_SEGMENT_BATCH_SIZES=1,2,3,4 XLLM_MTGR_ATTENTION_RAGGED_VS_FOUR_SEGMENT_DIFF=1 XLLM_MTGR_ATTENTION_RAGGED_VS_FOUR_SEGMENT_CSV=/export/home/wangyifan/Code/xllm/xllm/core/kernels/cuda/tests/mtgr_attention_ragged_vs_four_segment_default_precision_100_by_batch_20260509.csv build/lib.linux-x86_64-cpython-312/xllm/mtgr_attention_one_vs_multi_stage_perf_test --gtest_filter=MTGRAttentionOneVsMultiStagePerfTest.RaggedVsFourSegmentOddLengthRandomByBatchSizeCsv`
  - Result: 400 cases passed, `max_abs=0.000015`, average mean abs effectively
    zero, bad cases = 0.
  - CSV/log:
    - `xllm/core/kernels/cuda/tests/mtgr_attention_ragged_vs_four_segment_default_precision_100_by_batch_20260509.csv`
    - `xllm/core/kernels/cuda/tests/mtgr_attention_ragged_vs_four_segment_default_precision_100_by_batch_20260509.log`
- Default long stability/perf run:
  - Command:
    `CUDA_VISIBLE_DEVICES=0 XLLM_MTGR_ATTENTION_WARMUP=2 XLLM_MTGR_ATTENTION_REPEAT=5 XLLM_MTGR_ATTENTION_RAGGED_VS_FOUR_SEGMENT_GROUPS=1000 XLLM_MTGR_ATTENTION_RAGGED_VS_FOUR_SEGMENT_BATCH_SIZES=1,2,3,4 XLLM_MTGR_ATTENTION_RAGGED_VS_FOUR_SEGMENT_DIFF=0 XLLM_MTGR_ATTENTION_RAGGED_VS_FOUR_SEGMENT_CSV=/export/home/wangyifan/Code/xllm/xllm/core/kernels/cuda/tests/mtgr_attention_ragged_vs_four_segment_default_odd_random_1000_by_batch_20260509.csv build/lib.linux-x86_64-cpython-312/xllm/mtgr_attention_one_vs_multi_stage_perf_test --gtest_filter=MTGRAttentionOneVsMultiStagePerfTest.RaggedVsFourSegmentOddLengthRandomByBatchSizeCsv`
  - Result: 4000 cases passed.
  - Overall: four-segment `0.564619 ms`, ragged `0.556034 ms`,
    ratio `1.015439`, wall ratio `1.045617`.
  - By head_dim:
    - hd64: `n=1992`, ratio `1.063879`, four `0.436443`, ragged `0.410237`.
    - hd128: `n=2008`, ratio `0.987304`, four `0.691773`, ragged `0.700669`.
  - By batch/head_dim:
    - b1 hd64 ratio `1.039721`; b1 hd128 ratio `0.998238`.
    - b2 hd64 ratio `1.068134`; b2 hd128 ratio `0.988390`.
    - b3 hd64 ratio `1.068301`; b3 hd128 ratio `0.985649`.
    - b4 hd64 ratio `1.066163`; b4 hd128 ratio `0.984518`.
  - CSV/log:
    - `xllm/core/kernels/cuda/tests/mtgr_attention_ragged_vs_four_segment_default_odd_random_1000_by_batch_20260509.csv`
    - `xllm/core/kernels/cuda/tests/mtgr_attention_ragged_vs_four_segment_default_odd_random_1000_by_batch_20260509.log`
- Mask-build + one-stage FlashInfer comparison:
  - Build command:
    `docker exec wangyifan-cuda-x86 bash -lc 'cd /export/home/wangyifan/Code/xllm && cmake --build build/cmake.linux-x86_64-cpython-312 --target mtgr_attention_one_vs_multi_stage_perf_test -j 16'`
  - Precision smoke command:
    `CUDA_VISIBLE_DEVICES=0 XLLM_MTGR_ATTENTION_WARMUP=1 XLLM_MTGR_ATTENTION_REPEAT=1 XLLM_MTGR_ATTENTION_RAGGED_VS_ONE_STAGE_GROUPS=20 XLLM_MTGR_ATTENTION_RAGGED_VS_ONE_STAGE_BATCH_SIZES=1,2,3,4 XLLM_MTGR_ATTENTION_RAGGED_VS_ONE_STAGE_DIFF=1 XLLM_MTGR_ATTENTION_RAGGED_VS_ONE_STAGE_CSV=/export/home/wangyifan/Code/xllm/xllm/core/kernels/cuda/tests/mtgr_attention_ragged_vs_one_stage_precision_20_by_batch_20260509.csv build/lib.linux-x86_64-cpython-312/xllm/mtgr_attention_one_vs_multi_stage_perf_test --gtest_filter=MTGRAttentionOneVsMultiStagePerfTest.RaggedVsOneStageOddLengthRandomByBatchSizeCsv`
  - Precision smoke result: 80 cases passed, `max_abs=0.000061`,
    `bad_gt_1e-3=0`.
  - 1000-group perf command:
    `CUDA_VISIBLE_DEVICES=0 XLLM_MTGR_ATTENTION_WARMUP=2 XLLM_MTGR_ATTENTION_REPEAT=5 XLLM_MTGR_ATTENTION_RAGGED_VS_ONE_STAGE_GROUPS=1000 XLLM_MTGR_ATTENTION_RAGGED_VS_ONE_STAGE_BATCH_SIZES=1,2,3,4 XLLM_MTGR_ATTENTION_RAGGED_VS_ONE_STAGE_DIFF=0 XLLM_MTGR_ATTENTION_RAGGED_VS_ONE_STAGE_CSV=/export/home/wangyifan/Code/xllm/xllm/core/kernels/cuda/tests/mtgr_attention_ragged_vs_one_stage_odd_random_1000_by_batch_20260509.csv build/lib.linux-x86_64-cpython-312/xllm/mtgr_attention_one_vs_multi_stage_perf_test --gtest_filter=MTGRAttentionOneVsMultiStagePerfTest.RaggedVsOneStageOddLengthRandomByBatchSizeCsv`
  - 1000-group result: 4000 cases passed, CSV has 4000 data rows plus header,
    no `mask_build + one_stage_device` regressions versus ragged.
  - Overall averages across batch sizes 1-4: mask build `0.181462 ms`,
    one-stage device `3.490801 ms`, mask+device `3.672264 ms`, ragged device
    `0.558115 ms`; mask+device/ragged ratio `6.579768`.
  - By batch size, average `mask_build + one_stage_device` / `ragged_device`:
    - b1: `1.480470 / 0.275949 = 5.365021`.
    - b2: `2.951387 / 0.466617 = 6.325070`.
    - b3: `4.464930 / 0.654164 = 6.825398`.
    - b4: `5.792270 / 0.835728 = 6.930804`.
  - By batch/head_dim, average `mask_build + one_stage_device` /
    `ragged_device`:
    - b1 hd64 `6.764368`, b1 hd128 `4.508689`.
    - b2 hd64 `7.980675`, b2 hd128 `5.345353`.
    - b3 hd64 `8.562460`, b3 hd128 `5.753743`.
    - b4 hd64 `8.976644`, b4 hd128 `5.823801`.
  - CSV/log:
    - `xllm/core/kernels/cuda/tests/mtgr_attention_ragged_vs_one_stage_precision_20_by_batch_20260509.csv`
    - `xllm/core/kernels/cuda/tests/mtgr_attention_ragged_vs_one_stage_precision_20_by_batch_20260509.log`
    - `xllm/core/kernels/cuda/tests/mtgr_attention_ragged_vs_one_stage_perf_100_by_batch_20260509.csv`
    - `xllm/core/kernels/cuda/tests/mtgr_attention_ragged_vs_one_stage_perf_100_by_batch_20260509.log`
    - `xllm/core/kernels/cuda/tests/mtgr_attention_ragged_vs_one_stage_odd_random_1000_by_batch_20260509.csv`
    - `xllm/core/kernels/cuda/tests/mtgr_attention_ragged_vs_one_stage_odd_random_1000_by_batch_20260509.log`

Important interpretation:

- The ragged segment path now essentially matches or beats the old four-segment
  path overall on the 1000 odd-length random by-batch sweep.
- Compared with the mask-build + one-stage FlashInfer baseline, ragged is
  clearly faster on this odd-length by-batch sweep. Including mask build, the
  baseline is about 5.37x, 6.33x, 6.83x, and 6.93x slower for batch sizes 1, 2,
  3, and 4 respectively. Device-only one-stage is also much slower; the mask
  build contributes roughly 0.07 ms per request.
- head_dim=128 is still slightly slower than four-segment for batch sizes 2-4,
  but the split-diag default narrowed the previous roughly 2.3%-2.6% gap to
  roughly 1.2%-1.6%; batch size 1 is effectively tied.
- Do not spend more time optimizing head_dim=128 unless the user asks. The user
  explicitly accepted this performance level and asked to keep precision and
  stability as the priority.

Generated artifacts from this round that are useful to keep:

- `xllm/core/kernels/cuda/tests/mtgr_attention_ragged_vs_four_segment_default_precision_100_by_batch_20260509.csv`
- `xllm/core/kernels/cuda/tests/mtgr_attention_ragged_vs_four_segment_default_precision_100_by_batch_20260509.log`
- `xllm/core/kernels/cuda/tests/mtgr_attention_ragged_vs_four_segment_default_odd_random_1000_by_batch_20260509.csv`
- `xllm/core/kernels/cuda/tests/mtgr_attention_ragged_vs_four_segment_default_odd_random_1000_by_batch_20260509.log`

Cleanup already done:

- Removed the two core dumps produced during the transient cache-key abort
  experiments: `core.431245` and `core.431536`.
- Older unrelated core files were left untouched.

Recommended next step if this is resumed:

- Start by reading this `Latest Handoff` section.
- If asked for one more guardrail run, rerun the default precision smoke above.
- If asked to prepare for commit/PR, include the three tracked source files
  listed above and decide whether the generated CSV/log artifacts should remain
  untracked or be archived elsewhere.

## Current Goal

Finish the new ragged segment attention kernel path as a single-side CUDA
experiment. Do not wire it into the main MTGR flow yet; first validate precision
and performance in the test/perf binaries and keep a short handoff trail so the
next window can continue without rereading the whole thread.

## Done

- Traced the requirement to a more generic ragged segment attention abstraction.
- Added prototype CUDA entrypoints and kernels in `xllm/core/kernels/cuda/tests/mtgr_fused_attention_kernel.cu`.
- Added batch-layout plumbing and test wrappers in `xllm/core/kernels/cuda/tests/mtgr_attenion_test.cpp` and `.h`.
- Generalized the E2E precision reference in `xllm/core/kernels/cuda/tests/mtgr_attention_e2e_precision_test.cpp`.
- Added a ragged-segment precision test case.
- Reconfigured the project in container `wangyifan-cuda-x86`.
- Built `mtgr_attention_e2e_precision_test` successfully in the container.
- Ran `MTGRAttentionE2EPrecisionTest.RaggedSegmentAttentionFiveSegmentsAlignsWithReference` successfully.
- Ran `MTGRAttentionE2EPrecisionTest.FusedSegmentedNoMatchFiveSegmentsAlignsWithReference` successfully.
- Removed the CUDA tensor `.max().item()` validation from the ragged public wrapper so timing does not include a device reduction/sync.
- Added static segment-count dispatch for the ragged kernel for 2..8 segments.
- Added `MTGRAttentionOneVsMultiStagePerfTest.RaggedSegmentOddLengthRandomCsv` for single-side ragged perf CSV output.
- Rebuilt `mtgr_attention_e2e_precision_test` and `mtgr_attention_one_vs_multi_stage_perf_test` in container `wangyifan-cuda-x86`.
- Ran precision again:
  - `FusedSegmentedNoMatchFiveSegmentsAlignsWithReference`: `max_abs=3.016740e-05`, `mean_abs=5.949286e-07`, `device_ms=3.285472`.
  - `RaggedSegmentAttentionFiveSegmentsAlignsWithReference`: `max_abs=3.016740e-05`, `mean_abs=5.653445e-07`, `device_ms=0.285472`.
- Ran ragged perf smoke:
  - `RaggedSegmentOddLengthRandomCsv`, `cases=3`, `batch_size=2`, `warmup=1`, `repeat=3`.
  - Completed 3/3, `avg_device_total_ms=18.921034`, `avg_wall_total_ms=18.936530`, `avg_exec_ms=18.921034`.
  - CSV: `xllm/core/kernels/cuda/tests/mtgr_attention_ragged_segment_smoke.csv`.
- Added an experimental ragged q64/reg-frag fast path for head_dim 64/128 on
  SM80+:
  - Reuses the four-segment q64/reg-frag tiled QK/PV structure for the ragged
    prefix window.
  - Keeps head_dim 32 and pre-SM80 on the original one-warp-per-row fallback.
  - Merges diagonal self tokens inside the q64 writeback path.
  - Uses KV128 as the ragged default for head_dim 64 after a 20-case sweep.
- Rebuilt `mtgr_attention_e2e_precision_test` and
  `mtgr_attention_one_vs_multi_stage_perf_test` in container
  `wangyifan-cuda-x86`.
- Ran precision after the q64 fast path:
  - `FusedSegmentedNoMatchFiveSegmentsAlignsWithReference`: `max_abs=3.016740e-05`,
    `mean_abs=5.949286e-07`, `device_ms=3.428800`.
  - `RaggedSegmentAttentionFiveSegmentsAlignsWithReference`: `max_abs=3.613532e-05`,
    `mean_abs=7.508282e-07`, `device_ms=0.088192`.
- Ran ragged q64 perf smoke:
  - `RaggedSegmentOddLengthRandomCsv`, `cases=3`, `batch_size=2`, `warmup=1`,
    `repeat=3`.
  - Completed 3/3, `avg_device_total_ms=0.501977`,
    `avg_wall_total_ms=0.517044`, `avg_exec_ms=0.501977`.
  - CSV: `xllm/core/kernels/cuda/tests/mtgr_attention_ragged_segment_smoke_q64_default_kv128.csv`.
- Ran ragged q64 1000-case odd-length sweep:
  - `cases=1000`, `batch_size=2`, `warmup=2`, `repeat=5`,
    `seed=20260509`.
  - Completed 1000/1000, `avg_device_total_ms=0.488535`,
    `avg_wall_total_ms=0.503076`, `avg_exec_ms=0.488535`.
  - p50/p90/p95/p99 device: `0.445606 / 0.821402 / 0.941549 / 1.215213`.
  - Average total tokens: `9452.60`; average max request tokens: `5273.98`.
  - CSV: `xllm/core/kernels/cuda/tests/mtgr_attention_ragged_segment_odd_random_1000_q64_20260509.csv`.

## Next

- Keep the ragged path as a single-side experiment until precision/perf evidence is sufficient.
- Optionally tune head_dim 128 tile selection for ragged; KV96 did not improve
  the first three smoke cases, so the current path keeps the four-segment
  heuristic for head_dim 128.
- If needed, clean up the remaining warning noise in `mtgr_fused_attention_kernel.cu`.

## Notes

- All non-git work should be done in the container `wangyifan-cuda-x86`.
- The workspace is dirty with many unrelated user-generated files; do not revert them.
- The repo already has existing MTGR experiment files and logs; leave unrelated artifacts alone.

## Useful Files

- `xllm/core/kernels/cuda/tests/mtgr_fused_attention_kernel.cu`
- `xllm/core/kernels/cuda/tests/mtgr_attenion_test.cpp`
- `xllm/core/kernels/cuda/tests/mtgr_attenion_test.h`
- `xllm/core/kernels/cuda/tests/mtgr_attention_e2e_precision_test.cpp`
- `xllm/core/kernels/cuda/tests/mtgr_attention_one_vs_multi_stage_perf_test.cpp`
- `xllm/core/kernels/npu/tests/SEGMENTED_PREFILL_AND_GENREC_CONTEXT.md`
- `xllm/core/kernels/npu/tests/skills.md`
