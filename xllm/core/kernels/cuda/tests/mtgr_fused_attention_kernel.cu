/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMacros.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <mma.h>
#include <torch/torch.h>

#include <cstdlib>

namespace xllm::kernel::cuda::test {
namespace {

constexpr float kLog2E = 1.4426950408889634f;
constexpr int kWarpSize = 32;
constexpr float kNegInf = -INFINITY;
using b128_t = uint4;

enum class MtgrMmaMode {
  kInit = 0U,
  kInplaceUpdate = 1U,
};

template <typename T>
__device__ __forceinline__ void mtgr_ldmatrix_m8n8x4(uint32_t* R, T* smem_ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  uint32_t smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
      : "r"(smem_int_ptr));
#else
  (void)R;
  (void)smem_ptr;
#endif
}

template <typename T>
__device__ __forceinline__ void mtgr_ldmatrix_m8n8x4_trans(uint32_t* R,
                                                           T* smem_ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  uint32_t smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.trans.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
      : "r"(smem_int_ptr));
#else
  (void)R;
  (void)smem_ptr;
#endif
}

template <typename T>
__device__ __forceinline__ void mtgr_stmatrix_m8n8x4(uint32_t* R, T* smem_ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
      :
      : "r"(smem_int_ptr), "r"(R[0]), "r"(R[1]), "r"(R[2]), "r"(R[3]));
#else
  const uint32_t tx = threadIdx.x;
  uint4 word;
#pragma unroll
  for (uint32_t reg_id = 0; reg_id < 4; ++reg_id) {
    word.x = __shfl_sync(0xffffffff, R[reg_id], (tx % 8) * 4);
    word.y = __shfl_sync(0xffffffff, R[reg_id], (tx % 8) * 4 + 1);
    word.z = __shfl_sync(0xffffffff, R[reg_id], (tx % 8) * 4 + 2);
    word.w = __shfl_sync(0xffffffff, R[reg_id], (tx % 8) * 4 + 3);
    if (tx / 8 == reg_id) {
      *reinterpret_cast<uint4*>(smem_ptr) = word;
    }
  }
#endif
}

template <int N>
__device__ __forceinline__ void mtgr_cp_async_wait_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

__device__ __forceinline__ void mtgr_cp_async_commit_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <bool kPrefetchL2 = true>
__device__ __forceinline__ void mtgr_cp_async_load_128b(void* smem_ptr,
                                                        const void* gmem_ptr,
                                                        bool pred_guard) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  uint32_t smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  int src_size = pred_guard ? 16 : 0;
  if constexpr (kPrefetchL2) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n"
                 :
                 : "r"(smem_int_ptr), "l"(gmem_ptr), "r"(src_size));
  } else {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :
                 : "r"(smem_int_ptr), "l"(gmem_ptr), "r"(src_size));
  }
#else
  if (pred_guard) {
    *reinterpret_cast<b128_t*>(smem_ptr) =
        *reinterpret_cast<const b128_t*>(gmem_ptr);
  } else {
    *reinterpret_cast<b128_t*>(smem_ptr) = make_uint4(0U, 0U, 0U, 0U);
  }
#endif
}

template <MtgrMmaMode mma_mode = MtgrMmaMode::kInplaceUpdate>
__device__ __forceinline__ void
mtgr_mma_sync_m16n16k16_row_col_f16f16f32(float* C, uint32_t* A, uint32_t* B) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  if constexpr (mma_mode == MtgrMmaMode::kInit) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
        : "r"(A[0]),
          "r"(A[1]),
          "r"(A[2]),
          "r"(A[3]),
          "r"(B[0]),
          "r"(B[1]),
          "f"(0.f),
          "f"(0.f),
          "f"(0.f),
          "f"(0.f));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
        : "r"(A[0]),
          "r"(A[1]),
          "r"(A[2]),
          "r"(A[3]),
          "r"(B[2]),
          "r"(B[3]),
          "f"(0.f),
          "f"(0.f),
          "f"(0.f),
          "f"(0.f));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
        : "r"(A[0]),
          "r"(A[1]),
          "r"(A[2]),
          "r"(A[3]),
          "r"(B[0]),
          "r"(B[1]),
          "f"(C[0]),
          "f"(C[1]),
          "f"(C[2]),
          "f"(C[3]));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
        : "r"(A[0]),
          "r"(A[1]),
          "r"(A[2]),
          "r"(A[3]),
          "r"(B[2]),
          "r"(B[3]),
          "f"(C[4]),
          "f"(C[5]),
          "f"(C[6]),
          "f"(C[7]));
  }
#else
  (void)C;
  (void)A;
  (void)B;
#endif
}

__device__ __forceinline__ void mtgr_m16k16_rowsum_f16f16f32(float* d,
                                                             half* s) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  uint32_t* s_u32 = reinterpret_cast<uint32_t*>(s);
  asm volatile(
      "{\n"
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,  _,  %1,  _},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  0.,  %9,  0.};\n"
      "}\n"
      : "=f"(d[0]), "=f"(d[1])
      : "r"(s_u32[0]),
        "r"(s_u32[1]),
        "r"(s_u32[2]),
        "r"(s_u32[3]),
        "r"(1006648320),
        "r"(1006648320),
        "f"(d[0]),
        "f"(d[1]));
#else
  (void)d;
  (void)s;
#endif
}

template <typename dst_t, typename src_t>
__device__ __forceinline__ void mtgr_vec_cast_8(dst_t* dst, const src_t* src) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    dst[i] = static_cast<dst_t>(src[i]);
  }
}

template <uint32_t stride>
__device__ __forceinline__ uint32_t mtgr_perm128_offset(uint32_t i,
                                                        uint32_t j) {
  return i * stride + (j ^ (i % 8));
}

template <uint32_t step_size>
__device__ __forceinline__ uint32_t
mtgr_perm128_advance_col(uint32_t offset, uint32_t step_idx) {
  static_assert(step_size == 2 || step_size == 4 || step_size % 8 == 0);
  if constexpr (step_size == 2) {
    return (offset ^ (0x2 + (0x4 * (step_idx % 2 == 1)))) +
           (step_idx % 4 == 3) * 8;
  } else if constexpr (step_size == 4) {
    return (offset ^ 0x4) + (step_idx % 2 == 1) * 8;
  } else {
    return offset + step_size;
  }
}

template <uint32_t stride>
struct MtgrPermutedSmem128 {
  b128_t* base;

  template <typename T>
  __device__ __forceinline__ explicit MtgrPermutedSmem128(T* ptr)
      : base(reinterpret_cast<b128_t*>(ptr)) {}

  __device__ __forceinline__ static uint32_t get_permuted_offset(uint32_t i,
                                                                 uint32_t j) {
    return mtgr_perm128_offset<stride>(i, j);
  }

  template <uint32_t step_size>
  __device__ __forceinline__ static uint32_t advance_offset_by_column(
      uint32_t offset,
      uint32_t step_idx) {
    return mtgr_perm128_advance_col<step_size>(offset, step_idx);
  }

  template <uint32_t step_size, uint32_t row_stride>
  __device__ __forceinline__ static uint32_t advance_offset_by_row(
      uint32_t offset) {
    static_assert(step_size == 4 || step_size % 8 == 0);
    if constexpr (step_size == 4) {
      return (offset ^ 0x4) + step_size * row_stride;
    } else {
      return offset + step_size * row_stride;
    }
  }

  __device__ __forceinline__ void ldmatrix_m8n8x4(uint32_t offset,
                                                  uint32_t* R) {
    mtgr_ldmatrix_m8n8x4(R, base + offset);
  }

  __device__ __forceinline__ void ldmatrix_m8n8x4_trans(uint32_t offset,
                                                        uint32_t* R) {
    mtgr_ldmatrix_m8n8x4_trans(R, base + offset);
  }

  __device__ __forceinline__ void stmatrix_m8n8x4(uint32_t offset,
                                                  uint32_t* R) {
    mtgr_stmatrix_m8n8x4(R, base + offset);
  }
};

inline bool use_mtgr_sm90_q16_regpv_experimental() {
  const char* env = std::getenv("XLLM_MTGR_CUDA_FUSED_Q16_REGPV");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

inline bool use_mtgr_sm90_q32_regpv_experimental() {
  const char* env = std::getenv("XLLM_MTGR_CUDA_FUSED_Q32_REGPV");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

inline bool use_mtgr_sm90_q64_regfrag_experimental() {
  const char* env = std::getenv("XLLM_MTGR_CUDA_FUSED_Q64_REGFRAG");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

inline bool use_mtgr_sm90_q64_regfrag_kv64_experimental() {
  const char* env = std::getenv("XLLM_MTGR_CUDA_FUSED_Q64_REGFRAG_KV64");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

inline bool use_mtgr_sm90_q64_regfrag_kv80_experimental() {
  const char* env = std::getenv("XLLM_MTGR_CUDA_FUSED_Q64_REGFRAG_KV80");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

inline bool use_mtgr_sm90_q64_regfrag_kv96_experimental() {
  const char* env = std::getenv("XLLM_MTGR_CUDA_FUSED_Q64_REGFRAG_KV96");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ float warp_allreduce_sum(float value) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, mask);
  }
  return value;
}

__device__ __forceinline__ float warp_allreduce_max(float value) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, mask));
  }
  return value;
}

__device__ __forceinline__ int mtgr_visible_end_for_no_match(int q_idx,
                                                             int history_len,
                                                             int context_len,
                                                             int realtime_len) {
  const int history_end = history_len;
  const int context_end = history_len + context_len;
  const int realtime_end = context_end + realtime_len;
  if (q_idx < history_end) {
    return q_idx + 1;
  }
  if (q_idx < context_end) {
    return context_end;
  }
  if (q_idx < realtime_end) {
    return q_idx + 1;
  }
  return realtime_end;
}

template <int HeadDim, int WarpsPerBlock>
__global__
__launch_bounds__(WarpsPerBlock * 32) void mtgr_fused_no_match_attention_fallback_kernel(
    const half* __restrict__ query,
    const half* __restrict__ key,
    const half* __restrict__ value,
    half* __restrict__ output,
    float* __restrict__ target_hcr_lse,
    int total_len,
    int num_heads,
    int history_len,
    int context_len,
    int realtime_len,
    int target_len,
    float sm_scale_log2e) {
  constexpr int kLaneElems = HeadDim / 32;
  static_assert(HeadDim % 32 == 0);

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int row = blockIdx.x * WarpsPerBlock + warp_id;
  const int num_rows = total_len * num_heads;
  if (row >= num_rows) {
    return;
  }

  const int q_idx = row / num_heads;
  const int head_idx = row - q_idx * num_heads;
  const int realtime_end = history_len + context_len + realtime_len;
  const int visible_end = mtgr_visible_end_for_no_match(
      q_idx, history_len, context_len, realtime_len);
  const bool write_target_lse = q_idx >= realtime_end;

  float q_frag[kLaneElems];
  float o_frag[kLaneElems];
  const int dim_base = lane * kLaneElems;
  const int64_t q_base =
      (static_cast<int64_t>(q_idx) * num_heads + head_idx) * HeadDim + dim_base;
#pragma unroll
  for (int i = 0; i < kLaneElems; ++i) {
    q_frag[i] = __half2float(query[q_base + i]);
    o_frag[i] = 0.0f;
  }

  float row_m = -INFINITY;
  float row_d = 0.0f;
  for (int kv_idx = 0; kv_idx < visible_end; ++kv_idx) {
    const int64_t kv_base =
        (static_cast<int64_t>(kv_idx) * num_heads + head_idx) * HeadDim +
        dim_base;
    float dot_lane = 0.0f;
#pragma unroll
    for (int i = 0; i + 1 < kLaneElems; i += 2) {
      const half2 k2 = *reinterpret_cast<const half2*>(key + kv_base + i);
      const float2 kf = __half22float2(k2);
      dot_lane += q_frag[i] * kf.x + q_frag[i + 1] * kf.y;
    }
    if constexpr ((kLaneElems & 1) != 0) {
      dot_lane +=
          q_frag[kLaneElems - 1] * __half2float(key[kv_base + kLaneElems - 1]);
    }
    dot_lane = warp_reduce_sum(dot_lane);

    float alpha = 0.0f;
    float beta = 0.0f;
    if (lane == 0) {
      const float score = dot_lane * sm_scale_log2e;
      const float new_m = fmaxf(row_m, score);
      alpha = row_m == -INFINITY ? 0.0f : exp2f(row_m - new_m);
      beta = exp2f(score - new_m);
      row_d = row_d * alpha + beta;
      row_m = new_m;
    }
    alpha = __shfl_sync(0xffffffff, alpha, 0);
    beta = __shfl_sync(0xffffffff, beta, 0);
    row_d = __shfl_sync(0xffffffff, row_d, 0);
    row_m = __shfl_sync(0xffffffff, row_m, 0);

#pragma unroll
    for (int i = 0; i < kLaneElems; ++i) {
      const float v = __half2float(value[kv_base + i]);
      o_frag[i] = o_frag[i] * alpha + beta * v;
    }
  }

  const float inv_d = row_d > 0.0f ? 1.0f / row_d : 0.0f;
  const int64_t o_base =
      (static_cast<int64_t>(q_idx) * num_heads + head_idx) * HeadDim + dim_base;
#pragma unroll
  for (int i = 0; i < kLaneElems; ++i) {
    output[o_base + i] = __float2half_rn(o_frag[i] * inv_d);
  }

  if (write_target_lse && lane == 0) {
    const int target_row = q_idx - realtime_end;
    if (target_row < target_len) {
      target_hcr_lse[(static_cast<int64_t>(target_row) * num_heads +
                      head_idx)] = row_m + log2f(row_d);
    }
  }
}

template <int QueriesPerBlock>
__global__
__launch_bounds__(QueriesPerBlock* kWarpSize) void mtgr_fused_no_match_attention_wmma128_kernel(
    const half* __restrict__ query,
    const half* __restrict__ key,
    const half* __restrict__ value,
    half* __restrict__ output,
    float* __restrict__ target_hcr_lse,
    int total_len,
    int num_heads,
    int history_len,
    int context_len,
    int realtime_len,
    int target_len,
    float sm_scale_log2e) {
  constexpr int kHeadDim = 128;
  constexpr int kWmmaTile = 16;
  constexpr int kKvTile = 32;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  __shared__ half q_shared[kWmmaTile * kHeadDim];
  __shared__ half k_shared[kKvTile * kHeadDim];
  __shared__ half v_shared[kKvTile * kHeadDim];
  __shared__ float score_shared[kWmmaTile * kKvTile];
  __shared__ half prob_shared[kWmmaTile * kKvTile];
  __shared__ float pv_shared[kWmmaTile * kHeadDim];
  __shared__ int s_visible_end[QueriesPerBlock];
  __shared__ int s_block_max_visible_end;

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x >> 5;
  const int q_block_start = static_cast<int>(blockIdx.x) * QueriesPerBlock;
  const int q_idx = q_block_start + warp_id;
  const int head_idx = static_cast<int>(blockIdx.y);
  const int realtime_end = history_len + context_len + realtime_len;
  const bool row_valid = warp_id < QueriesPerBlock && q_idx < total_len;
  const int visible_end =
      row_valid ? mtgr_visible_end_for_no_match(
                      q_idx, history_len, context_len, realtime_len)
                : 0;

  if (warp_id < QueriesPerBlock && lane == 0) {
    s_visible_end[warp_id] = visible_end;
  }
  for (int flat = threadIdx.x; flat < kWmmaTile * kHeadDim;
       flat += blockDim.x) {
    const int row = flat / kHeadDim;
    const int dim = flat % kHeadDim;
    const int global_q = q_block_start + row;
    q_shared[flat] =
        row < QueriesPerBlock && global_q < total_len
            ? query[(static_cast<int64_t>(global_q) * num_heads + head_idx) *
                        kHeadDim +
                    dim]
            : __float2half(0.0f);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int block_max = 0;
#pragma unroll
    for (int i = 0; i < QueriesPerBlock; ++i) {
      block_max = max(block_max, s_visible_end[i]);
    }
    s_block_max_visible_end = block_max;
  }
  __syncthreads();

  float row_max = kNegInf;
  float row_sum = 0.0f;
  float row_accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int kv_tile_start = 0; kv_tile_start < s_block_max_visible_end;
       kv_tile_start += kKvTile) {
    const int valid_rows =
        min(kKvTile, s_block_max_visible_end - kv_tile_start);

    for (int flat = threadIdx.x; flat < kKvTile * kHeadDim;
         flat += blockDim.x) {
      const int row = flat / kHeadDim;
      const int dim = flat % kHeadDim;
      if (row < valid_rows) {
        const int kv_idx = kv_tile_start + row;
        const int64_t offset =
            (static_cast<int64_t>(kv_idx) * num_heads + head_idx) * kHeadDim +
            dim;
        k_shared[flat] = key[offset];
        v_shared[flat] = value[offset];
      } else {
        k_shared[flat] = __float2half(0.0f);
        v_shared[flat] = __float2half(0.0f);
      }
    }
    for (int flat = threadIdx.x; flat < kWmmaTile * kKvTile;
         flat += blockDim.x) {
      prob_shared[flat] = __float2half(0.0f);
    }
    __syncthreads();

    if (warp_id < 2) {
      using namespace nvcuda;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
      wmma::fill_fragment(c_frag, 0.0f);
      const int kv_subtile = warp_id * kWmmaTile;
#pragma unroll
      for (int kk = 0; kk < kHeadDim; kk += kWmmaTile) {
        wmma::load_matrix_sync(a_frag, q_shared + kk, kHeadDim);
        wmma::load_matrix_sync(
            b_frag, k_shared + kv_subtile * kHeadDim + kk, kHeadDim);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }
      wmma::store_matrix_sync(
          score_shared + kv_subtile, c_frag, kKvTile, wmma::mem_row_major);
    }
    __syncthreads();

    float tile_max = kNegInf;
    float tile_sum = 0.0f;
    if (row_valid) {
      int row_tile_tokens = visible_end - kv_tile_start;
      if (row_tile_tokens > valid_rows) {
        row_tile_tokens = valid_rows;
      }
      if (row_tile_tokens < 0) {
        row_tile_tokens = 0;
      }

      const float scaled_score =
          lane < row_tile_tokens
              ? score_shared[warp_id * kKvTile + lane] * sm_scale_log2e
              : kNegInf;
      tile_max = warp_allreduce_max(scaled_score);
      const float lane_prob =
          lane < row_tile_tokens ? exp2f(scaled_score - tile_max) : 0.0f;
      tile_sum = warp_allreduce_sum(lane_prob);
      if (lane < kKvTile) {
        prob_shared[warp_id * kKvTile + lane] =
            __float2half_rn(lane < row_tile_tokens ? lane_prob : 0.0f);
      }
    }
    __syncthreads();

    if (warp_id < 8) {
      using namespace nvcuda;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
      wmma::fill_fragment(c_frag, 0.0f);
      wmma::load_matrix_sync(a_frag, prob_shared, kKvTile);
      wmma::load_matrix_sync(b_frag, v_shared + warp_id * kWmmaTile, kHeadDim);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      wmma::load_matrix_sync(a_frag, prob_shared + kWmmaTile, kKvTile);
      wmma::load_matrix_sync(
          b_frag,
          v_shared + kWmmaTile * kHeadDim + warp_id * kWmmaTile,
          kHeadDim);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      wmma::store_matrix_sync(pv_shared + warp_id * kWmmaTile,
                              c_frag,
                              kHeadDim,
                              wmma::mem_row_major);
    }
    __syncthreads();

    if (row_valid) {
      const int dim_base = lane * 4;
      float tile_accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int d = 0; d < 4; ++d) {
        tile_accum[d] = pv_shared[warp_id * kHeadDim + dim_base + d];
      }

      const float new_max = fmaxf(row_max, tile_max);
      const float old_scale =
          row_max == kNegInf ? 0.0f : exp2f(row_max - new_max);
      const float tile_scale = exp2f(tile_max - new_max);
      row_sum = row_sum * old_scale + tile_sum * tile_scale;
#pragma unroll
      for (int d = 0; d < 4; ++d) {
        row_accum[d] = row_accum[d] * old_scale + tile_accum[d] * tile_scale;
      }
      row_max = new_max;
    }
    __syncthreads();
  }

  if (!row_valid) {
    return;
  }

  const float inv_denom = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;
  const int64_t output_index =
      (static_cast<int64_t>(q_idx) * num_heads + head_idx) * kHeadDim +
      lane * 4;
#pragma unroll
  for (int d = 0; d < 4; ++d) {
    output[output_index + d] = __float2half_rn(row_accum[d] * inv_denom);
  }

  if (q_idx >= realtime_end && lane == 0) {
    const int target_row = q_idx - realtime_end;
    if (target_row < target_len) {
      target_hcr_lse[(static_cast<int64_t>(target_row) * num_heads +
                      head_idx)] = row_max + log2f(row_sum);
    }
  }
#else
  (void)query;
  (void)key;
  (void)value;
  (void)output;
  (void)target_hcr_lse;
  (void)total_len;
  (void)num_heads;
  (void)history_len;
  (void)context_len;
  (void)realtime_len;
  (void)target_len;
  (void)sm_scale_log2e;
#endif
}

__global__
__launch_bounds__(8 * kWarpSize) void mtgr_fused_no_match_attention_wmma128_q16_regpv_kernel(
    const half* __restrict__ query,
    const half* __restrict__ key,
    const half* __restrict__ value,
    half* __restrict__ output,
    float* __restrict__ target_hcr_lse,
    int total_len,
    int num_heads,
    int history_len,
    int context_len,
    int realtime_len,
    int target_len,
    float sm_scale_log2e) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  constexpr int kHeadDim = 128;
  constexpr int kQueriesPerBlock = 16;
  constexpr int kWarpsPerBlock = 8;
  constexpr int kKvTile = 32;
  constexpr int kWmmaTile = 16;
  constexpr int kUpcastStrideV = kHeadDim / 8;

  __shared__ half q_shared[kQueriesPerBlock * kHeadDim];
  __shared__ half k_shared[kKvTile * kHeadDim];
  __shared__ b128_t v_shared_perm[kKvTile * kUpcastStrideV];
  __shared__ float score_shared[kQueriesPerBlock * kKvTile];
  __shared__ uint16_t s_visible_end[kQueriesPerBlock];
  __shared__ uint16_t s_block_max_visible_end;

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x >> 5;
  const int q_block_start = static_cast<int>(blockIdx.x) * kQueriesPerBlock;
  const int head_idx = static_cast<int>(blockIdx.y);
  const int realtime_end = history_len + context_len + realtime_len;

  for (int row = threadIdx.x; row < kQueriesPerBlock; row += blockDim.x) {
    const int q_idx = q_block_start + row;
    s_visible_end[row] = static_cast<uint16_t>(
        q_idx < total_len ? mtgr_visible_end_for_no_match(
                                q_idx, history_len, context_len, realtime_len)
                          : 0);
  }
  for (int flat = threadIdx.x; flat < kQueriesPerBlock * kHeadDim;
       flat += blockDim.x) {
    const int row = flat / kHeadDim;
    const int dim = flat % kHeadDim;
    const int q_idx = q_block_start + row;
    q_shared[flat] =
        q_idx < total_len
            ? query[(static_cast<int64_t>(q_idx) * num_heads + head_idx) *
                        kHeadDim +
                    dim]
            : __float2half(0.0f);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int block_max = 0;
#pragma unroll
    for (int i = 0; i < kQueriesPerBlock; ++i) {
      block_max = max(block_max, static_cast<int>(s_visible_end[i]));
    }
    s_block_max_visible_end = static_cast<uint16_t>(block_max);
  }
  __syncthreads();

  float m[2] = {kNegInf, kNegInf};
  float d[2] = {0.0f, 0.0f};
  float o_frag[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  MtgrPermutedSmem128<kUpcastStrideV> v_smem(v_shared_perm);

  const int row0_local = lane >> 2;
  const int row1_local = row0_local + 8;
  const int row0_q_idx = q_block_start + row0_local;
  const int row1_q_idx = q_block_start + row1_local;
  const bool row0_valid = row0_q_idx < total_len;
  const bool row1_valid = row1_q_idx < total_len;

  for (int kv_tile_start = 0;
       kv_tile_start < static_cast<int>(s_block_max_visible_end);
       kv_tile_start += kKvTile) {
    const int valid_rows =
        min(kKvTile, static_cast<int>(s_block_max_visible_end) - kv_tile_start);

    for (int flat = threadIdx.x; flat < kKvTile * kHeadDim;
         flat += blockDim.x) {
      const int row = flat / kHeadDim;
      const int dim = flat % kHeadDim;
      if (row < valid_rows) {
        const int kv_idx = kv_tile_start + row;
        const int64_t offset =
            (static_cast<int64_t>(kv_idx) * num_heads + head_idx) * kHeadDim +
            dim;
        k_shared[flat] = key[offset];
      } else {
        k_shared[flat] = __float2half(0.0f);
      }
    }
    for (int flat = threadIdx.x; flat < kKvTile * kUpcastStrideV;
         flat += blockDim.x) {
      const int row = flat / kUpcastStrideV;
      const int chunk = flat % kUpcastStrideV;
      const uint32_t perm_offset =
          mtgr_perm128_offset<kUpcastStrideV>(row, chunk);
      if (row < valid_rows) {
        const int kv_idx = kv_tile_start + row;
        const int64_t offset =
            (static_cast<int64_t>(kv_idx) * num_heads + head_idx) * kHeadDim;
        const b128_t* src = reinterpret_cast<const b128_t*>(value + offset);
        v_shared_perm[perm_offset] = src[chunk];
      } else {
        v_shared_perm[perm_offset] = make_uint4(0U, 0U, 0U, 0U);
      }
    }
    __syncthreads();

    if (warp_id < 2) {
      using namespace nvcuda;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
      wmma::fill_fragment(c_frag, 0.0f);
      const int kv_subtile = warp_id * kWmmaTile;
#pragma unroll
      for (int kk = 0; kk < kHeadDim; kk += kWmmaTile) {
        wmma::load_matrix_sync(a_frag, q_shared + kk, kHeadDim);
        wmma::load_matrix_sync(
            b_frag, k_shared + kv_subtile * kHeadDim + kk, kHeadDim);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }
      wmma::store_matrix_sync(
          score_shared + kv_subtile, c_frag, kKvTile, wmma::mem_row_major);
    }
    __syncthreads();

    float s_frag[2][8];
#pragma unroll
    for (int mma_kv = 0; mma_kv < 2; ++mma_kv) {
#pragma unroll
      for (int reg_id = 0; reg_id < 8; ++reg_id) {
        const int row_sel = (reg_id % 4) / 2;
        const int row_local = row_sel == 0 ? row0_local : row1_local;
        const bool row_valid = row_sel == 0 ? row0_valid : row1_valid;
        const int visible_end = row_sel == 0 ? s_visible_end[row0_local]
                                             : s_visible_end[row1_local];
        const int col_local = 2 * (lane % 4) + 8 * (reg_id / 4) + (reg_id & 1);
        const int tile_col = mma_kv * 16 + col_local;
        const bool visible = row_valid && tile_col < valid_rows &&
                             kv_tile_start + tile_col < visible_end;
        s_frag[mma_kv][reg_id] =
            visible ? score_shared[row_local * kKvTile + tile_col] : kNegInf;
      }
    }

#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const float m_prev = m[j];
      float m_new = m_prev;
#pragma unroll
      for (int mma_kv = 0; mma_kv < 2; ++mma_kv) {
        const float m_local =
            fmaxf(fmaxf(s_frag[mma_kv][j * 2 + 0], s_frag[mma_kv][j * 2 + 1]),
                  fmaxf(s_frag[mma_kv][j * 2 + 4], s_frag[mma_kv][j * 2 + 5]));
        m_new = fmaxf(m_new, m_local);
      }
      m_new = fmaxf(m_new, __shfl_xor_sync(0xffffffff, m_new, 0x2));
      m_new = fmaxf(m_new, __shfl_xor_sync(0xffffffff, m_new, 0x1));

      const float o_scale =
          m_prev == kNegInf
              ? 0.0f
              : exp2f(m_prev * sm_scale_log2e - m_new * sm_scale_log2e);
      d[j] *= o_scale;
      o_frag[j * 2 + 0] *= o_scale;
      o_frag[j * 2 + 1] *= o_scale;
      o_frag[j * 2 + 4] *= o_scale;
      o_frag[j * 2 + 5] *= o_scale;

#pragma unroll
      for (int mma_kv = 0; mma_kv < 2; ++mma_kv) {
        s_frag[mma_kv][j * 2 + 0] =
            exp2f(s_frag[mma_kv][j * 2 + 0] * sm_scale_log2e -
                  m_new * sm_scale_log2e);
        s_frag[mma_kv][j * 2 + 1] =
            exp2f(s_frag[mma_kv][j * 2 + 1] * sm_scale_log2e -
                  m_new * sm_scale_log2e);
        s_frag[mma_kv][j * 2 + 4] =
            exp2f(s_frag[mma_kv][j * 2 + 4] * sm_scale_log2e -
                  m_new * sm_scale_log2e);
        s_frag[mma_kv][j * 2 + 5] =
            exp2f(s_frag[mma_kv][j * 2 + 5] * sm_scale_log2e -
                  m_new * sm_scale_log2e);
      }
      m[j] = m_new;
    }

    half s_frag_half[2][8];
#pragma unroll
    for (int mma_kv = 0; mma_kv < 2; ++mma_kv) {
      mtgr_vec_cast_8(s_frag_half[mma_kv], s_frag[mma_kv]);
      mtgr_m16k16_rowsum_f16f16f32(d, s_frag_half[mma_kv]);

      uint32_t v_frag[4];
      uint32_t v_offset = mtgr_perm128_offset<kUpcastStrideV>(
          mma_kv * 16 + (lane % 16), lane / 16);
      v_offset = mtgr_perm128_advance_col<2>(v_offset, warp_id);
      v_smem.ldmatrix_m8n8x4_trans(v_offset, v_frag);
      mtgr_mma_sync_m16n16k16_row_col_f16f16f32(
          o_frag, reinterpret_cast<uint32_t*>(s_frag_half[mma_kv]), v_frag);
    }
    __syncthreads();
  }

  const int col_base = warp_id * 16;
  const int pair_base = (lane % 4) * 2;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int q_idx = q_block_start + row0_local + j * 8;
    if (q_idx < total_len) {
      const float inv_d = d[j] > 0.0f ? 1.0f / d[j] : 0.0f;
      const int64_t out_base =
          (static_cast<int64_t>(q_idx) * num_heads + head_idx) * kHeadDim +
          col_base;
      output[out_base + pair_base + 0] =
          __float2half_rn(o_frag[j * 2 + 0] * inv_d);
      output[out_base + pair_base + 1] =
          __float2half_rn(o_frag[j * 2 + 1] * inv_d);
      output[out_base + 8 + pair_base + 0] =
          __float2half_rn(o_frag[4 + j * 2 + 0] * inv_d);
      output[out_base + 8 + pair_base + 1] =
          __float2half_rn(o_frag[4 + j * 2 + 1] * inv_d);

      if (warp_id == 0 && (lane % 4) == 0 && q_idx >= realtime_end) {
        const int target_row = q_idx - realtime_end;
        if (target_row < target_len) {
          target_hcr_lse[(static_cast<int64_t>(target_row) * num_heads +
                          head_idx)] = m[j] + log2f(d[j]);
        }
      }
    }
  }
#else
  (void)query;
  (void)key;
  (void)value;
  (void)output;
  (void)target_hcr_lse;
  (void)total_len;
  (void)num_heads;
  (void)history_len;
  (void)context_len;
  (void)realtime_len;
  (void)target_len;
  (void)sm_scale_log2e;
#endif
}

__global__
__launch_bounds__(16 * kWarpSize) void mtgr_fused_no_match_attention_wmma128_q32_kernel(
    const half* __restrict__ query,
    const half* __restrict__ key,
    const half* __restrict__ value,
    half* __restrict__ output,
    float* __restrict__ target_hcr_lse,
    int total_len,
    int num_heads,
    int history_len,
    int context_len,
    int realtime_len,
    int target_len,
    float sm_scale_log2e) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  constexpr int kHeadDim = 128;
  constexpr int kRowsPerPhase = 16;
  constexpr int kQueriesPerBlock = 32;
  constexpr int kKvTile = 32;

  __shared__ half q_shared[kQueriesPerBlock * kHeadDim];
  __shared__ half k_shared[kKvTile * kHeadDim];
  __shared__ half v_shared[kKvTile * kHeadDim];
  __shared__ float score_shared[kRowsPerPhase * kKvTile];
  __shared__ half prob_shared[kRowsPerPhase * kKvTile];
  __shared__ float pv_shared[kRowsPerPhase * kHeadDim];
  __shared__ int s_visible_end[kQueriesPerBlock];
  __shared__ int s_block_max_visible_end;

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x >> 5;
  const int q_block_start = static_cast<int>(blockIdx.x) * kQueriesPerBlock;
  const int head_idx = static_cast<int>(blockIdx.y);
  const int realtime_end = history_len + context_len + realtime_len;

  const int q_idx0 = q_block_start + warp_id;
  const int q_idx1 = q_block_start + kRowsPerPhase + warp_id;
  const bool row0_valid = q_idx0 < total_len;
  const bool row1_valid = q_idx1 < total_len;
  const int visible_end0 =
      row0_valid ? mtgr_visible_end_for_no_match(
                       q_idx0, history_len, context_len, realtime_len)
                 : 0;
  const int visible_end1 =
      row1_valid ? mtgr_visible_end_for_no_match(
                       q_idx1, history_len, context_len, realtime_len)
                 : 0;

  if (lane == 0) {
    s_visible_end[warp_id] = visible_end0;
    s_visible_end[kRowsPerPhase + warp_id] = visible_end1;
  }
  for (int flat = threadIdx.x; flat < kQueriesPerBlock * kHeadDim;
       flat += blockDim.x) {
    const int row = flat / kHeadDim;
    const int dim = flat % kHeadDim;
    const int global_q = q_block_start + row;
    q_shared[flat] =
        global_q < total_len
            ? query[(static_cast<int64_t>(global_q) * num_heads + head_idx) *
                        kHeadDim +
                    dim]
            : __float2half(0.0f);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int block_max = 0;
#pragma unroll
    for (int i = 0; i < kQueriesPerBlock; ++i) {
      block_max = max(block_max, s_visible_end[i]);
    }
    s_block_max_visible_end = block_max;
  }
  __syncthreads();

  float row0_max = kNegInf;
  float row0_sum = 0.0f;
  float row0_accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float row1_max = kNegInf;
  float row1_sum = 0.0f;
  float row1_accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int kv_tile_start = 0; kv_tile_start < s_block_max_visible_end;
       kv_tile_start += kKvTile) {
    const int valid_rows =
        min(kKvTile, s_block_max_visible_end - kv_tile_start);

    for (int flat = threadIdx.x; flat < kKvTile * kHeadDim;
         flat += blockDim.x) {
      const int row = flat / kHeadDim;
      const int dim = flat % kHeadDim;
      if (row < valid_rows) {
        const int kv_idx = kv_tile_start + row;
        const int64_t offset =
            (static_cast<int64_t>(kv_idx) * num_heads + head_idx) * kHeadDim +
            dim;
        k_shared[flat] = key[offset];
        v_shared[flat] = value[offset];
      } else {
        k_shared[flat] = __float2half(0.0f);
        v_shared[flat] = __float2half(0.0f);
      }
    }
    for (int flat = threadIdx.x; flat < kRowsPerPhase * kKvTile;
         flat += blockDim.x) {
      prob_shared[flat] = __float2half(0.0f);
    }
    __syncthreads();

    // Phase 0: rows [0, 15]
    if (warp_id < 2) {
      using namespace nvcuda;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
      wmma::fill_fragment(c_frag, 0.0f);
      const int kv_subtile = warp_id * kRowsPerPhase;
#pragma unroll
      for (int kk = 0; kk < kHeadDim; kk += kRowsPerPhase) {
        wmma::load_matrix_sync(a_frag, q_shared + kk, kHeadDim);
        wmma::load_matrix_sync(
            b_frag, k_shared + kv_subtile * kHeadDim + kk, kHeadDim);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }
      wmma::store_matrix_sync(
          score_shared + kv_subtile, c_frag, kKvTile, wmma::mem_row_major);
    }
    __syncthreads();

    float tile0_max = kNegInf;
    float tile0_sum = 0.0f;
    if (row0_valid) {
      int row_tile_tokens = visible_end0 - kv_tile_start;
      if (row_tile_tokens > valid_rows) {
        row_tile_tokens = valid_rows;
      }
      if (row_tile_tokens < 0) {
        row_tile_tokens = 0;
      }
      const float scaled_score =
          lane < row_tile_tokens
              ? score_shared[warp_id * kKvTile + lane] * sm_scale_log2e
              : kNegInf;
      tile0_max = warp_allreduce_max(scaled_score);
      const float lane_prob =
          lane < row_tile_tokens ? exp2f(scaled_score - tile0_max) : 0.0f;
      tile0_sum = warp_allreduce_sum(lane_prob);
      prob_shared[warp_id * kKvTile + lane] =
          __float2half_rn(lane < row_tile_tokens ? lane_prob : 0.0f);
    }
    __syncthreads();

    if (warp_id < 8) {
      using namespace nvcuda;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
      wmma::fill_fragment(c_frag, 0.0f);
      wmma::load_matrix_sync(a_frag, prob_shared, kKvTile);
      wmma::load_matrix_sync(
          b_frag, v_shared + warp_id * kRowsPerPhase, kHeadDim);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      wmma::load_matrix_sync(a_frag, prob_shared + kRowsPerPhase, kKvTile);
      wmma::load_matrix_sync(
          b_frag,
          v_shared + kRowsPerPhase * kHeadDim + warp_id * kRowsPerPhase,
          kHeadDim);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      wmma::store_matrix_sync(pv_shared + warp_id * kRowsPerPhase,
                              c_frag,
                              kHeadDim,
                              wmma::mem_row_major);
    }
    __syncthreads();

    if (row0_valid) {
      const int dim_base = lane * 4;
      float tile_accum[4];
#pragma unroll
      for (int d = 0; d < 4; ++d) {
        tile_accum[d] = pv_shared[warp_id * kHeadDim + dim_base + d];
      }
      const float new_max = fmaxf(row0_max, tile0_max);
      const float old_scale =
          row0_max == kNegInf ? 0.0f : exp2f(row0_max - new_max);
      const float tile_scale = exp2f(tile0_max - new_max);
      row0_sum = row0_sum * old_scale + tile0_sum * tile_scale;
#pragma unroll
      for (int d = 0; d < 4; ++d) {
        row0_accum[d] = row0_accum[d] * old_scale + tile_accum[d] * tile_scale;
      }
      row0_max = new_max;
    }
    __syncthreads();

    // Phase 1: rows [16, 31]
    if (warp_id < 2) {
      using namespace nvcuda;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
      wmma::fill_fragment(c_frag, 0.0f);
      const int kv_subtile = warp_id * kRowsPerPhase;
#pragma unroll
      for (int kk = 0; kk < kHeadDim; kk += kRowsPerPhase) {
        wmma::load_matrix_sync(
            a_frag, q_shared + kRowsPerPhase * kHeadDim + kk, kHeadDim);
        wmma::load_matrix_sync(
            b_frag, k_shared + kv_subtile * kHeadDim + kk, kHeadDim);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }
      wmma::store_matrix_sync(
          score_shared + kv_subtile, c_frag, kKvTile, wmma::mem_row_major);
    }
    __syncthreads();

    float tile1_max = kNegInf;
    float tile1_sum = 0.0f;
    if (row1_valid) {
      int row_tile_tokens = visible_end1 - kv_tile_start;
      if (row_tile_tokens > valid_rows) {
        row_tile_tokens = valid_rows;
      }
      if (row_tile_tokens < 0) {
        row_tile_tokens = 0;
      }
      const float scaled_score =
          lane < row_tile_tokens
              ? score_shared[warp_id * kKvTile + lane] * sm_scale_log2e
              : kNegInf;
      tile1_max = warp_allreduce_max(scaled_score);
      const float lane_prob =
          lane < row_tile_tokens ? exp2f(scaled_score - tile1_max) : 0.0f;
      tile1_sum = warp_allreduce_sum(lane_prob);
      prob_shared[warp_id * kKvTile + lane] =
          __float2half_rn(lane < row_tile_tokens ? lane_prob : 0.0f);
    }
    __syncthreads();

    if (warp_id < 8) {
      using namespace nvcuda;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
      wmma::fill_fragment(c_frag, 0.0f);
      wmma::load_matrix_sync(a_frag, prob_shared, kKvTile);
      wmma::load_matrix_sync(
          b_frag, v_shared + warp_id * kRowsPerPhase, kHeadDim);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      wmma::load_matrix_sync(a_frag, prob_shared + kRowsPerPhase, kKvTile);
      wmma::load_matrix_sync(
          b_frag,
          v_shared + kRowsPerPhase * kHeadDim + warp_id * kRowsPerPhase,
          kHeadDim);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      wmma::store_matrix_sync(pv_shared + warp_id * kRowsPerPhase,
                              c_frag,
                              kHeadDim,
                              wmma::mem_row_major);
    }
    __syncthreads();

    if (row1_valid) {
      const int dim_base = lane * 4;
      float tile_accum[4];
#pragma unroll
      for (int d = 0; d < 4; ++d) {
        tile_accum[d] = pv_shared[warp_id * kHeadDim + dim_base + d];
      }
      const float new_max = fmaxf(row1_max, tile1_max);
      const float old_scale =
          row1_max == kNegInf ? 0.0f : exp2f(row1_max - new_max);
      const float tile_scale = exp2f(tile1_max - new_max);
      row1_sum = row1_sum * old_scale + tile1_sum * tile_scale;
#pragma unroll
      for (int d = 0; d < 4; ++d) {
        row1_accum[d] = row1_accum[d] * old_scale + tile_accum[d] * tile_scale;
      }
      row1_max = new_max;
    }
    __syncthreads();
  }

  const int dim_base = lane * 4;
  if (row0_valid) {
    const float inv_denom = row0_sum > 0.0f ? 1.0f / row0_sum : 0.0f;
    const int64_t output_index =
        (static_cast<int64_t>(q_idx0) * num_heads + head_idx) * kHeadDim +
        dim_base;
#pragma unroll
    for (int d = 0; d < 4; ++d) {
      output[output_index + d] = __float2half_rn(row0_accum[d] * inv_denom);
    }
    if (q_idx0 >= realtime_end && lane == 0) {
      const int target_row = q_idx0 - realtime_end;
      if (target_row < target_len) {
        target_hcr_lse[(static_cast<int64_t>(target_row) * num_heads +
                        head_idx)] = row0_max + log2f(row0_sum);
      }
    }
  }
  if (row1_valid) {
    const float inv_denom = row1_sum > 0.0f ? 1.0f / row1_sum : 0.0f;
    const int64_t output_index =
        (static_cast<int64_t>(q_idx1) * num_heads + head_idx) * kHeadDim +
        dim_base;
#pragma unroll
    for (int d = 0; d < 4; ++d) {
      output[output_index + d] = __float2half_rn(row1_accum[d] * inv_denom);
    }
    if (q_idx1 >= realtime_end && lane == 0) {
      const int target_row = q_idx1 - realtime_end;
      if (target_row < target_len) {
        target_hcr_lse[(static_cast<int64_t>(target_row) * num_heads +
                        head_idx)] = row1_max + log2f(row1_sum);
      }
    }
  }
#else
  (void)query;
  (void)key;
  (void)value;
  (void)output;
  (void)target_hcr_lse;
  (void)total_len;
  (void)num_heads;
  (void)history_len;
  (void)context_len;
  (void)realtime_len;
  (void)target_len;
  (void)sm_scale_log2e;
#endif
}

__global__
__launch_bounds__(8 * kWarpSize) void mtgr_fused_no_match_attention_wmma128_q32_regpv_kernel(
    const half* __restrict__ query,
    const half* __restrict__ key,
    const half* __restrict__ value,
    half* __restrict__ output,
    float* __restrict__ target_hcr_lse,
    int total_len,
    int num_heads,
    int history_len,
    int context_len,
    int realtime_len,
    int target_len,
    float sm_scale_log2e) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  constexpr int kHeadDim = 128;
  constexpr int kRowsPerPhase = 16;
  constexpr int kQueriesPerBlock = 32;
  constexpr int kWarpsPerBlock = 8;
  constexpr int kKvTile = 32;
  constexpr int kWmmaTile = 16;
  constexpr int kUpcastStrideV = kHeadDim / 8;

  __shared__ half q_shared[kQueriesPerBlock * kHeadDim];
  __shared__ half k_shared[kKvTile * kHeadDim];
  __shared__ b128_t v_shared_perm[kKvTile * kUpcastStrideV];
  __shared__ float score_shared[kRowsPerPhase * kKvTile];
  __shared__ int s_visible_end[kQueriesPerBlock];
  __shared__ int s_block_max_visible_end;

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x >> 5;
  const int q_block_start = static_cast<int>(blockIdx.x) * kQueriesPerBlock;
  const int head_idx = static_cast<int>(blockIdx.y);
  const int realtime_end = history_len + context_len + realtime_len;

  for (int row = threadIdx.x; row < kQueriesPerBlock; row += blockDim.x) {
    const int q_idx = q_block_start + row;
    s_visible_end[row] =
        q_idx < total_len ? mtgr_visible_end_for_no_match(
                                q_idx, history_len, context_len, realtime_len)
                          : 0;
  }
  for (int flat = threadIdx.x; flat < kQueriesPerBlock * kHeadDim;
       flat += blockDim.x) {
    const int row = flat / kHeadDim;
    const int dim = flat % kHeadDim;
    const int q_idx = q_block_start + row;
    q_shared[flat] =
        q_idx < total_len
            ? query[(static_cast<int64_t>(q_idx) * num_heads + head_idx) *
                        kHeadDim +
                    dim]
            : __float2half(0.0f);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int block_max = 0;
#pragma unroll
    for (int i = 0; i < kQueriesPerBlock; ++i) {
      block_max = max(block_max, s_visible_end[i]);
    }
    s_block_max_visible_end = block_max;
  }
  __syncthreads();

  float m[4] = {kNegInf, kNegInf, kNegInf, kNegInf};
  float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float o_frag[2][8] = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
  MtgrPermutedSmem128<kUpcastStrideV> v_smem(v_shared_perm);

  const int row0_local = lane >> 2;
  const int row1_local = row0_local + 8;

  for (int kv_tile_start = 0; kv_tile_start < s_block_max_visible_end;
       kv_tile_start += kKvTile) {
    const int valid_rows =
        min(kKvTile, s_block_max_visible_end - kv_tile_start);

    for (int flat = threadIdx.x; flat < kKvTile * kHeadDim;
         flat += blockDim.x) {
      const int row = flat / kHeadDim;
      const int dim = flat % kHeadDim;
      if (row < valid_rows) {
        const int kv_idx = kv_tile_start + row;
        const int64_t offset =
            (static_cast<int64_t>(kv_idx) * num_heads + head_idx) * kHeadDim +
            dim;
        k_shared[flat] = key[offset];
      } else {
        k_shared[flat] = __float2half(0.0f);
      }
    }
    for (int flat = threadIdx.x; flat < kKvTile * kUpcastStrideV;
         flat += blockDim.x) {
      const int row = flat / kUpcastStrideV;
      const int chunk = flat % kUpcastStrideV;
      const uint32_t perm_offset =
          mtgr_perm128_offset<kUpcastStrideV>(row, chunk);
      if (row < valid_rows) {
        const int kv_idx = kv_tile_start + row;
        const int64_t offset =
            (static_cast<int64_t>(kv_idx) * num_heads + head_idx) * kHeadDim;
        const b128_t* src = reinterpret_cast<const b128_t*>(value + offset);
        v_shared_perm[perm_offset] = src[chunk];
      } else {
        v_shared_perm[perm_offset] = make_uint4(0U, 0U, 0U, 0U);
      }
    }
    __syncthreads();

#pragma unroll
    for (int phase = 0; phase < 2; ++phase) {
      const int phase_row_offset = phase * kRowsPerPhase;
      const int q_phase_base = q_block_start + phase_row_offset;
      const int row0_q_idx = q_phase_base + row0_local;
      const int row1_q_idx = q_phase_base + row1_local;
      const bool row0_valid = row0_q_idx < total_len;
      const bool row1_valid = row1_q_idx < total_len;
      const int visible_end0 = s_visible_end[phase_row_offset + row0_local];
      const int visible_end1 = s_visible_end[phase_row_offset + row1_local];

      if (warp_id < 2) {
        using namespace nvcuda;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
            a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>
            b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        const int kv_subtile = warp_id * kWmmaTile;
#pragma unroll
        for (int kk = 0; kk < kHeadDim; kk += kWmmaTile) {
          wmma::load_matrix_sync(
              a_frag, q_shared + phase_row_offset * kHeadDim + kk, kHeadDim);
          wmma::load_matrix_sync(
              b_frag, k_shared + kv_subtile * kHeadDim + kk, kHeadDim);
          wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(
            score_shared + kv_subtile, c_frag, kKvTile, wmma::mem_row_major);
      }
      __syncthreads();

      float s_frag[2][8];
#pragma unroll
      for (int mma_kv = 0; mma_kv < 2; ++mma_kv) {
#pragma unroll
        for (int reg_id = 0; reg_id < 8; ++reg_id) {
          const int row_sel = (reg_id % 4) / 2;
          const int row_local = row_sel == 0 ? row0_local : row1_local;
          const bool row_valid = row_sel == 0 ? row0_valid : row1_valid;
          const int visible_end = row_sel == 0 ? visible_end0 : visible_end1;
          const int col_local =
              2 * (lane % 4) + 8 * (reg_id / 4) + (reg_id & 1);
          const int tile_col = mma_kv * 16 + col_local;
          const bool visible = row_valid && tile_col < valid_rows &&
                               kv_tile_start + tile_col < visible_end;
          s_frag[mma_kv][reg_id] =
              visible ? score_shared[row_local * kKvTile + tile_col] : kNegInf;
        }
      }

      const int md_offset = phase * 2;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const float m_prev = m[md_offset + j];
        float m_new = m_prev;
#pragma unroll
        for (int mma_kv = 0; mma_kv < 2; ++mma_kv) {
          const float m_local = fmaxf(
              fmaxf(s_frag[mma_kv][j * 2 + 0], s_frag[mma_kv][j * 2 + 1]),
              fmaxf(s_frag[mma_kv][j * 2 + 4], s_frag[mma_kv][j * 2 + 5]));
          m_new = fmaxf(m_new, m_local);
        }
        m_new = fmaxf(m_new, __shfl_xor_sync(0xffffffff, m_new, 0x2));
        m_new = fmaxf(m_new, __shfl_xor_sync(0xffffffff, m_new, 0x1));

        const float o_scale =
            m_prev == kNegInf
                ? 0.0f
                : exp2f(m_prev * sm_scale_log2e - m_new * sm_scale_log2e);
        d[md_offset + j] *= o_scale;
        o_frag[phase][j * 2 + 0] *= o_scale;
        o_frag[phase][j * 2 + 1] *= o_scale;
        o_frag[phase][j * 2 + 4] *= o_scale;
        o_frag[phase][j * 2 + 5] *= o_scale;

#pragma unroll
        for (int mma_kv = 0; mma_kv < 2; ++mma_kv) {
          s_frag[mma_kv][j * 2 + 0] =
              exp2f(s_frag[mma_kv][j * 2 + 0] * sm_scale_log2e -
                    m_new * sm_scale_log2e);
          s_frag[mma_kv][j * 2 + 1] =
              exp2f(s_frag[mma_kv][j * 2 + 1] * sm_scale_log2e -
                    m_new * sm_scale_log2e);
          s_frag[mma_kv][j * 2 + 4] =
              exp2f(s_frag[mma_kv][j * 2 + 4] * sm_scale_log2e -
                    m_new * sm_scale_log2e);
          s_frag[mma_kv][j * 2 + 5] =
              exp2f(s_frag[mma_kv][j * 2 + 5] * sm_scale_log2e -
                    m_new * sm_scale_log2e);
        }
        m[md_offset + j] = m_new;
      }

      half s_frag_half[2][8];
#pragma unroll
      for (int mma_kv = 0; mma_kv < 2; ++mma_kv) {
        mtgr_vec_cast_8(s_frag_half[mma_kv], s_frag[mma_kv]);
        mtgr_m16k16_rowsum_f16f16f32(d + md_offset, s_frag_half[mma_kv]);

        uint32_t v_frag[4];
        uint32_t v_offset = mtgr_perm128_offset<kUpcastStrideV>(
            mma_kv * 16 + (lane % 16), lane / 16);
        v_offset = mtgr_perm128_advance_col<2>(v_offset, warp_id);
        v_smem.ldmatrix_m8n8x4_trans(v_offset, v_frag);
        mtgr_mma_sync_m16n16k16_row_col_f16f16f32(
            o_frag[phase],
            reinterpret_cast<uint32_t*>(s_frag_half[mma_kv]),
            v_frag);
      }
      __syncthreads();
    }
  }

  const int col_base = warp_id * 16;
  const int pair_base = (lane % 4) * 2;
#pragma unroll
  for (int phase = 0; phase < 2; ++phase) {
    const int phase_row_offset = phase * kRowsPerPhase;
    const int md_offset = phase * 2;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int q_idx = q_block_start + phase_row_offset + row0_local + j * 8;
      if (q_idx < total_len) {
        const float inv_d =
            d[md_offset + j] > 0.0f ? 1.0f / d[md_offset + j] : 0.0f;
        const int64_t out_base =
            (static_cast<int64_t>(q_idx) * num_heads + head_idx) * kHeadDim +
            col_base;
        output[out_base + pair_base + 0] =
            __float2half_rn(o_frag[phase][j * 2 + 0] * inv_d);
        output[out_base + pair_base + 1] =
            __float2half_rn(o_frag[phase][j * 2 + 1] * inv_d);
        output[out_base + 8 + pair_base + 0] =
            __float2half_rn(o_frag[phase][4 + j * 2 + 0] * inv_d);
        output[out_base + 8 + pair_base + 1] =
            __float2half_rn(o_frag[phase][4 + j * 2 + 1] * inv_d);

        if (warp_id == 0 && (lane % 4) == 0 && q_idx >= realtime_end) {
          const int target_row = q_idx - realtime_end;
          if (target_row < target_len) {
            target_hcr_lse[(static_cast<int64_t>(target_row) * num_heads +
                            head_idx)] =
                m[md_offset + j] + log2f(d[md_offset + j]);
          }
        }
      }
    }
  }
#else
  (void)query;
  (void)key;
  (void)value;
  (void)output;
  (void)target_hcr_lse;
  (void)total_len;
  (void)num_heads;
  (void)history_len;
  (void)context_len;
  (void)realtime_len;
  (void)target_len;
  (void)sm_scale_log2e;
#endif
}

template <int NumMmaKV>
__global__
__launch_bounds__(4 * kWarpSize) void mtgr_fused_no_match_attention_wmma128_q64_regfrag_kernel(
    const half* __restrict__ query,
    const half* __restrict__ key,
    const half* __restrict__ value,
    half* __restrict__ output,
    float* __restrict__ target_hcr_lse,
    int total_len,
    int num_heads,
    int history_len,
    int context_len,
    int realtime_len,
    int target_len,
    float sm_scale_log2e) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  constexpr int kHeadDim = 128;
  constexpr int kQueriesPerWarp = 16;
  constexpr int kQueriesPerBlock = 64;
  constexpr int kWarpsPerBlock = 4;
  constexpr int kNumMmaKV = NumMmaKV;
  constexpr int kKvTile = kNumMmaKV * 16;
  constexpr int kNumMmaD = 8;
  constexpr int kUpcastStride = kHeadDim / 8;

  __shared__ b128_t q_shared_perm[kQueriesPerBlock * kUpcastStride];
  extern __shared__ __align__(16) uint8_t mtgr_dyn_smem[];
  auto* k_shared_perm = reinterpret_cast<b128_t*>(mtgr_dyn_smem);
  auto* v_shared_perm = k_shared_perm + kKvTile * kUpcastStride;
  auto* s_visible_end =
      reinterpret_cast<uint16_t*>(v_shared_perm + kKvTile * kUpcastStride);
  auto* s_block_max_visible_end = s_visible_end + kQueriesPerBlock;

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x >> 5;
  const int q_block_start = static_cast<int>(blockIdx.x) * kQueriesPerBlock;
  const int head_idx = static_cast<int>(blockIdx.y);
  const int realtime_end = history_len + context_len + realtime_len;

  for (int row = threadIdx.x; row < kQueriesPerBlock; row += blockDim.x) {
    const int q_idx = q_block_start + row;
    s_visible_end[row] = static_cast<uint16_t>(
        q_idx < total_len ? mtgr_visible_end_for_no_match(
                                q_idx, history_len, context_len, realtime_len)
                          : 0);
  }
  for (int flat = threadIdx.x; flat < kQueriesPerBlock * kUpcastStride;
       flat += blockDim.x) {
    const int row = flat / kUpcastStride;
    const int chunk = flat % kUpcastStride;
    const uint32_t perm_offset = mtgr_perm128_offset<kUpcastStride>(row, chunk);
    const int q_idx = q_block_start + row;
    const bool pred_guard = q_idx < total_len;
    const int64_t base =
        pred_guard
            ? (static_cast<int64_t>(q_idx) * num_heads + head_idx) * kHeadDim
            : 0;
    const b128_t* src = reinterpret_cast<const b128_t*>(query + base);
    mtgr_cp_async_load_128b(
        q_shared_perm + perm_offset, src + chunk, pred_guard);
  }
  mtgr_cp_async_commit_group();
  mtgr_cp_async_wait_group<0>();
  __syncthreads();

  if (threadIdx.x == 0) {
    int block_max = 0;
#pragma unroll
    for (int i = 0; i < kQueriesPerBlock; ++i) {
      block_max = max(block_max, static_cast<int>(s_visible_end[i]));
    }
    s_block_max_visible_end[0] = static_cast<uint16_t>(block_max);
  }
  __syncthreads();

  MtgrPermutedSmem128<kUpcastStride> q_smem(q_shared_perm);
  MtgrPermutedSmem128<kUpcastStride> k_smem(k_shared_perm);
  MtgrPermutedSmem128<kUpcastStride> v_smem(v_shared_perm);

  const int warp_q_base = q_block_start + warp_id * kQueriesPerWarp;
  const int row0_local = lane >> 2;
  const int row1_local = row0_local + 8;
  const bool row_valid[2] = {warp_q_base + row0_local < total_len,
                             warp_q_base + row1_local < total_len};

  float m[2] = {kNegInf, kNegInf};
  float d[2] = {0.0f, 0.0f};
  float o_frag[kNumMmaD][8];
#pragma unroll
  for (int mma_d = 0; mma_d < kNumMmaD; ++mma_d) {
#pragma unroll
    for (int reg_id = 0; reg_id < 8; ++reg_id) {
      o_frag[mma_d][reg_id] = 0.0f;
    }
  }

  for (int kv_tile_start = 0;
       kv_tile_start < static_cast<int>(s_block_max_visible_end[0]);
       kv_tile_start += kKvTile) {
    const int valid_rows = min(
        kKvTile, static_cast<int>(s_block_max_visible_end[0]) - kv_tile_start);

    for (int flat = threadIdx.x; flat < kKvTile * kUpcastStride;
         flat += blockDim.x) {
      const int row = flat / kUpcastStride;
      const int chunk = flat % kUpcastStride;
      const uint32_t perm_offset =
          mtgr_perm128_offset<kUpcastStride>(row, chunk);
      const int kv_idx = kv_tile_start + row;
      const bool pred_guard = row < valid_rows;
      const int64_t base =
          pred_guard
              ? (static_cast<int64_t>(kv_idx) * num_heads + head_idx) * kHeadDim
              : 0;
      const b128_t* k_src = reinterpret_cast<const b128_t*>(key + base);
      const b128_t* v_src = reinterpret_cast<const b128_t*>(value + base);
      mtgr_cp_async_load_128b(
          k_shared_perm + perm_offset, k_src + chunk, pred_guard);
      mtgr_cp_async_load_128b(
          v_shared_perm + perm_offset, v_src + chunk, pred_guard);
    }
    mtgr_cp_async_commit_group();
    mtgr_cp_async_wait_group<0>();
    __syncthreads();

    float s_frag[kNumMmaKV][8];
    uint32_t q_frag[4];
    uint32_t k_frag[4];
    uint32_t q_offset = MtgrPermutedSmem128<kUpcastStride>::get_permuted_offset(
        warp_id * kQueriesPerWarp + (lane % 16), lane / 16);
    uint32_t k_offset = MtgrPermutedSmem128<kUpcastStride>::get_permuted_offset(
        8 * (lane / 16) + (lane % 8), (lane % 16) / 8);

#pragma unroll
    for (int mma_d = 0; mma_d < kNumMmaD; ++mma_d) {
      q_smem.ldmatrix_m8n8x4(q_offset, q_frag);
      q_offset = MtgrPermutedSmem128<
          kUpcastStride>::advance_offset_by_row<16, kUpcastStride>(q_offset);

#pragma unroll
      for (int mma_kv = 0; mma_kv < kNumMmaKV; ++mma_kv) {
        k_smem.ldmatrix_m8n8x4(k_offset, k_frag);
        k_offset = MtgrPermutedSmem128<
            kUpcastStride>::advance_offset_by_row<16, kUpcastStride>(k_offset);
        if (mma_d == 0) {
          mtgr_mma_sync_m16n16k16_row_col_f16f16f32<MtgrMmaMode::kInit>(
              s_frag[mma_kv], q_frag, k_frag);
        } else {
          mtgr_mma_sync_m16n16k16_row_col_f16f16f32(
              s_frag[mma_kv], q_frag, k_frag);
        }
      }

      k_offset =
          MtgrPermutedSmem128<kUpcastStride>::advance_offset_by_column<2>(
              k_offset, mma_d) -
          kNumMmaKV * kQueriesPerWarp * kUpcastStride;
      q_offset =
          MtgrPermutedSmem128<kUpcastStride>::advance_offset_by_column<2>(
              q_offset, mma_d) -
          kQueriesPerWarp * kUpcastStride;
    }

#pragma unroll
    for (int mma_kv = 0; mma_kv < kNumMmaKV; ++mma_kv) {
#pragma unroll
      for (int reg_id = 0; reg_id < 8; ++reg_id) {
        const int row_sel = (reg_id % 4) / 2;
        const int row_local = row_sel == 0 ? row0_local : row1_local;
        const int q_idx = warp_q_base + row_local;
        const int visible_end =
            row_sel == 0
                ? static_cast<int>(
                      s_visible_end[warp_id * kQueriesPerWarp + row0_local])
                : static_cast<int>(
                      s_visible_end[warp_id * kQueriesPerWarp + row1_local]);
        const int kv_idx = kv_tile_start + mma_kv * 16 + 2 * (lane % 4) +
                           8 * (reg_id / 4) + (reg_id & 1);
        const bool visible = q_idx < total_len &&
                             kv_idx < kv_tile_start + valid_rows &&
                             kv_idx < visible_end;
        s_frag[mma_kv][reg_id] = visible ? s_frag[mma_kv][reg_id] : kNegInf;
      }
    }

#pragma unroll
    for (int j = 0; j < 2; ++j) {
      if (!row_valid[j]) {
#pragma unroll
        for (int mma_kv = 0; mma_kv < kNumMmaKV; ++mma_kv) {
          s_frag[mma_kv][j * 2 + 0] = 0.0f;
          s_frag[mma_kv][j * 2 + 1] = 0.0f;
          s_frag[mma_kv][j * 2 + 4] = 0.0f;
          s_frag[mma_kv][j * 2 + 5] = 0.0f;
        }
        continue;
      }

      const float m_prev = m[j];
      float m_new = m_prev;
#pragma unroll
      for (int mma_kv = 0; mma_kv < kNumMmaKV; ++mma_kv) {
        const float m_local =
            fmaxf(fmaxf(s_frag[mma_kv][j * 2 + 0], s_frag[mma_kv][j * 2 + 1]),
                  fmaxf(s_frag[mma_kv][j * 2 + 4], s_frag[mma_kv][j * 2 + 5]));
        m_new = fmaxf(m_new, m_local);
      }
      m_new = fmaxf(m_new, __shfl_xor_sync(0xffffffff, m_new, 0x2));
      m_new = fmaxf(m_new, __shfl_xor_sync(0xffffffff, m_new, 0x1));

      const float o_scale =
          m_prev == kNegInf
              ? 0.0f
              : exp2f(m_prev * sm_scale_log2e - m_new * sm_scale_log2e);
      d[j] *= o_scale;
#pragma unroll
      for (int mma_d = 0; mma_d < kNumMmaD; ++mma_d) {
        o_frag[mma_d][j * 2 + 0] *= o_scale;
        o_frag[mma_d][j * 2 + 1] *= o_scale;
        o_frag[mma_d][j * 2 + 4] *= o_scale;
        o_frag[mma_d][j * 2 + 5] *= o_scale;
      }
#pragma unroll
      for (int mma_kv = 0; mma_kv < kNumMmaKV; ++mma_kv) {
        s_frag[mma_kv][j * 2 + 0] =
            exp2f(s_frag[mma_kv][j * 2 + 0] * sm_scale_log2e -
                  m_new * sm_scale_log2e);
        s_frag[mma_kv][j * 2 + 1] =
            exp2f(s_frag[mma_kv][j * 2 + 1] * sm_scale_log2e -
                  m_new * sm_scale_log2e);
        s_frag[mma_kv][j * 2 + 4] =
            exp2f(s_frag[mma_kv][j * 2 + 4] * sm_scale_log2e -
                  m_new * sm_scale_log2e);
        s_frag[mma_kv][j * 2 + 5] =
            exp2f(s_frag[mma_kv][j * 2 + 5] * sm_scale_log2e -
                  m_new * sm_scale_log2e);
      }
      m[j] = m_new;
    }

    half s_frag_half[kNumMmaKV][8];
#pragma unroll
    for (int mma_kv = 0; mma_kv < kNumMmaKV; ++mma_kv) {
      mtgr_vec_cast_8(s_frag_half[mma_kv], s_frag[mma_kv]);
      mtgr_m16k16_rowsum_f16f16f32(d, s_frag_half[mma_kv]);

      uint32_t v_frag[4];
      uint32_t v_offset =
          MtgrPermutedSmem128<kUpcastStride>::get_permuted_offset(
              mma_kv * 16 + (lane % 16), lane / 16);
#pragma unroll
      for (int mma_d = 0; mma_d < kNumMmaD; ++mma_d) {
        v_smem.ldmatrix_m8n8x4_trans(v_offset, v_frag);
        mtgr_mma_sync_m16n16k16_row_col_f16f16f32(
            o_frag[mma_d],
            reinterpret_cast<uint32_t*>(s_frag_half[mma_kv]),
            v_frag);
        v_offset =
            MtgrPermutedSmem128<kUpcastStride>::advance_offset_by_column<2>(
                v_offset, mma_d);
      }
    }
    __syncthreads();
  }

  const int pair_base = (lane % 4) * 2;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int q_idx = warp_q_base + (lane / 4) + j * 8;
    if (q_idx < total_len) {
      const float inv_d = d[j] > 0.0f ? 1.0f / d[j] : 0.0f;
      const int64_t out_base =
          (static_cast<int64_t>(q_idx) * num_heads + head_idx) * kHeadDim;
#pragma unroll
      for (int mma_d = 0; mma_d < kNumMmaD; ++mma_d) {
        const int col_base = mma_d * 16;
        output[out_base + col_base + pair_base + 0] =
            __float2half_rn(o_frag[mma_d][j * 2 + 0] * inv_d);
        output[out_base + col_base + pair_base + 1] =
            __float2half_rn(o_frag[mma_d][j * 2 + 1] * inv_d);
        output[out_base + col_base + 8 + pair_base + 0] =
            __float2half_rn(o_frag[mma_d][4 + j * 2 + 0] * inv_d);
        output[out_base + col_base + 8 + pair_base + 1] =
            __float2half_rn(o_frag[mma_d][4 + j * 2 + 1] * inv_d);
      }

      if ((lane % 4) == 0 && q_idx >= realtime_end) {
        const int target_row = q_idx - realtime_end;
        if (target_row < target_len) {
          target_hcr_lse[(static_cast<int64_t>(target_row) * num_heads +
                          head_idx)] = m[j] * sm_scale_log2e + log2f(d[j]);
        }
      }
    }
  }
#else
  (void)query;
  (void)key;
  (void)value;
  (void)output;
  (void)target_hcr_lse;
  (void)total_len;
  (void)num_heads;
  (void)history_len;
  (void)context_len;
  (void)realtime_len;
  (void)target_len;
  (void)sm_scale_log2e;
#endif
}

template <int HeadDim>
void launch_mtgr_fused_no_match_attention_fallback_kernel(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    int64_t history_len,
    int64_t context_len,
    int64_t realtime_len,
    int64_t target_len,
    double sm_scale,
    torch::Tensor output_snd,
    torch::Tensor target_hcr_lse_sh1) {
  constexpr int kWarpsPerBlock = 4;
  const int total_len = static_cast<int>(query_snd.size(0));
  const int num_heads = static_cast<int>(query_snd.size(1));
  const int num_rows = total_len * num_heads;
  const dim3 block(kWarpsPerBlock * 32);
  const dim3 grid((num_rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  mtgr_fused_no_match_attention_fallback_kernel<HeadDim, kWarpsPerBlock>
      <<<grid, block, 0, stream>>>(
          reinterpret_cast<const half*>(query_snd.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(key_snd.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(value_snd.data_ptr<at::Half>()),
          reinterpret_cast<half*>(output_snd.data_ptr<at::Half>()),
          target_hcr_lse_sh1.data_ptr<float>(),
          total_len,
          num_heads,
          static_cast<int>(history_len),
          static_cast<int>(context_len),
          static_cast<int>(realtime_len),
          static_cast<int>(target_len),
          static_cast<float>(sm_scale) * kLog2E);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_mtgr_fused_no_match_attention_wmma128_q16(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    int64_t history_len,
    int64_t context_len,
    int64_t realtime_len,
    int64_t target_len,
    double sm_scale,
    torch::Tensor output_snd,
    torch::Tensor target_hcr_lse_sh1) {
  constexpr int kQueriesPerBlock = 16;
  const int total_len = static_cast<int>(query_snd.size(0));
  const int num_heads = static_cast<int>(query_snd.size(1));
  const dim3 block(kQueriesPerBlock * kWarpSize);
  const dim3 grid((total_len + kQueriesPerBlock - 1) / kQueriesPerBlock,
                  num_heads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  mtgr_fused_no_match_attention_wmma128_kernel<kQueriesPerBlock>
      <<<grid, block, 0, stream>>>(
          reinterpret_cast<const half*>(query_snd.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(key_snd.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(value_snd.data_ptr<at::Half>()),
          reinterpret_cast<half*>(output_snd.data_ptr<at::Half>()),
          target_hcr_lse_sh1.data_ptr<float>(),
          total_len,
          num_heads,
          static_cast<int>(history_len),
          static_cast<int>(context_len),
          static_cast<int>(realtime_len),
          static_cast<int>(target_len),
          static_cast<float>(sm_scale) * kLog2E);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_mtgr_fused_no_match_attention_wmma128_q16_regpv(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    int64_t history_len,
    int64_t context_len,
    int64_t realtime_len,
    int64_t target_len,
    double sm_scale,
    torch::Tensor output_snd,
    torch::Tensor target_hcr_lse_sh1) {
  constexpr int kQueriesPerBlock = 16;
  constexpr int kWarpsPerBlock = 8;
  const int total_len = static_cast<int>(query_snd.size(0));
  const int num_heads = static_cast<int>(query_snd.size(1));
  const dim3 block(kWarpsPerBlock * kWarpSize);
  const dim3 grid((total_len + kQueriesPerBlock - 1) / kQueriesPerBlock,
                  num_heads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  mtgr_fused_no_match_attention_wmma128_q16_regpv_kernel<<<grid,
                                                           block,
                                                           0,
                                                           stream>>>(
      reinterpret_cast<const half*>(query_snd.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(key_snd.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(value_snd.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output_snd.data_ptr<at::Half>()),
      target_hcr_lse_sh1.data_ptr<float>(),
      total_len,
      num_heads,
      static_cast<int>(history_len),
      static_cast<int>(context_len),
      static_cast<int>(realtime_len),
      static_cast<int>(target_len),
      static_cast<float>(sm_scale) * kLog2E);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_mtgr_fused_no_match_attention_wmma128_q32(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    int64_t history_len,
    int64_t context_len,
    int64_t realtime_len,
    int64_t target_len,
    double sm_scale,
    torch::Tensor output_snd,
    torch::Tensor target_hcr_lse_sh1) {
  constexpr int kQueriesPerBlock = 32;
  constexpr int kWarpsPerBlock = 16;
  const int total_len = static_cast<int>(query_snd.size(0));
  const int num_heads = static_cast<int>(query_snd.size(1));
  const dim3 block(kWarpsPerBlock * kWarpSize);
  const dim3 grid((total_len + kQueriesPerBlock - 1) / kQueriesPerBlock,
                  num_heads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  mtgr_fused_no_match_attention_wmma128_q32_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const half*>(query_snd.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(key_snd.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(value_snd.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output_snd.data_ptr<at::Half>()),
      target_hcr_lse_sh1.data_ptr<float>(),
      total_len,
      num_heads,
      static_cast<int>(history_len),
      static_cast<int>(context_len),
      static_cast<int>(realtime_len),
      static_cast<int>(target_len),
      static_cast<float>(sm_scale) * kLog2E);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_mtgr_fused_no_match_attention_wmma128_q32_regpv(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    int64_t history_len,
    int64_t context_len,
    int64_t realtime_len,
    int64_t target_len,
    double sm_scale,
    torch::Tensor output_snd,
    torch::Tensor target_hcr_lse_sh1) {
  constexpr int kQueriesPerBlock = 32;
  constexpr int kWarpsPerBlock = 8;
  const int total_len = static_cast<int>(query_snd.size(0));
  const int num_heads = static_cast<int>(query_snd.size(1));
  const dim3 block(kWarpsPerBlock * kWarpSize);
  const dim3 grid((total_len + kQueriesPerBlock - 1) / kQueriesPerBlock,
                  num_heads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  mtgr_fused_no_match_attention_wmma128_q32_regpv_kernel<<<grid,
                                                           block,
                                                           0,
                                                           stream>>>(
      reinterpret_cast<const half*>(query_snd.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(key_snd.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(value_snd.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output_snd.data_ptr<at::Half>()),
      target_hcr_lse_sh1.data_ptr<float>(),
      total_len,
      num_heads,
      static_cast<int>(history_len),
      static_cast<int>(context_len),
      static_cast<int>(realtime_len),
      static_cast<int>(target_len),
      static_cast<float>(sm_scale) * kLog2E);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int NumMmaKV>
void launch_mtgr_fused_no_match_attention_wmma128_q64_regfrag_impl(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    int64_t history_len,
    int64_t context_len,
    int64_t realtime_len,
    int64_t target_len,
    double sm_scale,
    torch::Tensor output_snd,
    torch::Tensor target_hcr_lse_sh1) {
  constexpr int kQueriesPerBlock = 64;
  constexpr int kWarpsPerBlock = 4;
  constexpr int kKvTile = NumMmaKV * 16;
  constexpr int kUpcastStride = 128 / 8;
  constexpr size_t kDynSharedBytes =
      2 * kKvTile * kUpcastStride * sizeof(b128_t) +
      (kQueriesPerBlock + 1) * sizeof(uint16_t);
  const int total_len = static_cast<int>(query_snd.size(0));
  const int num_heads = static_cast<int>(query_snd.size(1));
  const dim3 block(kWarpsPerBlock * kWarpSize);
  const dim3 grid((total_len + kQueriesPerBlock - 1) / kQueriesPerBlock,
                  num_heads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cudaFuncSetAttribute(
      mtgr_fused_no_match_attention_wmma128_q64_regfrag_kernel<NumMmaKV>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kDynSharedBytes));
  mtgr_fused_no_match_attention_wmma128_q64_regfrag_kernel<NumMmaKV>
      <<<grid, block, kDynSharedBytes, stream>>>(
          reinterpret_cast<const half*>(query_snd.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(key_snd.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(value_snd.data_ptr<at::Half>()),
          reinterpret_cast<half*>(output_snd.data_ptr<at::Half>()),
          target_hcr_lse_sh1.data_ptr<float>(),
          total_len,
          num_heads,
          static_cast<int>(history_len),
          static_cast<int>(context_len),
          static_cast<int>(realtime_len),
          static_cast<int>(target_len),
          static_cast<float>(sm_scale) * kLog2E);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_mtgr_fused_no_match_attention_wmma128_q64_regfrag(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    int64_t history_len,
    int64_t context_len,
    int64_t realtime_len,
    int64_t target_len,
    double sm_scale,
    torch::Tensor output_snd,
    torch::Tensor target_hcr_lse_sh1) {
  launch_mtgr_fused_no_match_attention_wmma128_q64_regfrag_impl<2>(
      query_snd,
      key_snd,
      value_snd,
      history_len,
      context_len,
      realtime_len,
      target_len,
      sm_scale,
      output_snd,
      target_hcr_lse_sh1);
}

void launch_mtgr_fused_no_match_attention_wmma128_q64_regfrag_kv64(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    int64_t history_len,
    int64_t context_len,
    int64_t realtime_len,
    int64_t target_len,
    double sm_scale,
    torch::Tensor output_snd,
    torch::Tensor target_hcr_lse_sh1) {
  launch_mtgr_fused_no_match_attention_wmma128_q64_regfrag_impl<4>(
      query_snd,
      key_snd,
      value_snd,
      history_len,
      context_len,
      realtime_len,
      target_len,
      sm_scale,
      output_snd,
      target_hcr_lse_sh1);
}

void launch_mtgr_fused_no_match_attention_wmma128_q64_regfrag_kv80(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    int64_t history_len,
    int64_t context_len,
    int64_t realtime_len,
    int64_t target_len,
    double sm_scale,
    torch::Tensor output_snd,
    torch::Tensor target_hcr_lse_sh1) {
  launch_mtgr_fused_no_match_attention_wmma128_q64_regfrag_impl<5>(
      query_snd,
      key_snd,
      value_snd,
      history_len,
      context_len,
      realtime_len,
      target_len,
      sm_scale,
      output_snd,
      target_hcr_lse_sh1);
}

void launch_mtgr_fused_no_match_attention_wmma128_q64_regfrag_kv96(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    int64_t history_len,
    int64_t context_len,
    int64_t realtime_len,
    int64_t target_len,
    double sm_scale,
    torch::Tensor output_snd,
    torch::Tensor target_hcr_lse_sh1) {
  launch_mtgr_fused_no_match_attention_wmma128_q64_regfrag_impl<6>(
      query_snd,
      key_snd,
      value_snd,
      history_len,
      context_len,
      realtime_len,
      target_len,
      sm_scale,
      output_snd,
      target_hcr_lse_sh1);
}

}  // namespace

void mtgr_fused_no_match_attention_cuda(const torch::Tensor& query_snd,
                                        const torch::Tensor& key_snd,
                                        const torch::Tensor& value_snd,
                                        int64_t history_len,
                                        int64_t context_len,
                                        int64_t realtime_len,
                                        int64_t target_len,
                                        double sm_scale,
                                        torch::Tensor output_snd,
                                        torch::Tensor target_hcr_lse_sh1) {
  CHECK(query_snd.defined());
  CHECK(key_snd.defined());
  CHECK(value_snd.defined());
  CHECK(output_snd.defined());
  CHECK(target_hcr_lse_sh1.defined());
  CHECK(query_snd.is_cuda());
  CHECK(key_snd.is_cuda());
  CHECK(value_snd.is_cuda());
  CHECK(output_snd.is_cuda());
  CHECK(target_hcr_lse_sh1.is_cuda());
  CHECK(query_snd.is_contiguous());
  CHECK(key_snd.is_contiguous());
  CHECK(value_snd.is_contiguous());
  CHECK(output_snd.is_contiguous());
  CHECK(target_hcr_lse_sh1.is_contiguous());
  CHECK_EQ(query_snd.scalar_type(), torch::kFloat16);
  CHECK_EQ(key_snd.scalar_type(), torch::kFloat16);
  CHECK_EQ(value_snd.scalar_type(), torch::kFloat16);
  CHECK_EQ(output_snd.scalar_type(), torch::kFloat16);
  CHECK_EQ(target_hcr_lse_sh1.scalar_type(), torch::kFloat32);
  CHECK_EQ(query_snd.dim(), 3);
  CHECK_EQ(key_snd.dim(), 3);
  CHECK_EQ(value_snd.dim(), 3);
  CHECK_EQ(output_snd.dim(), 3);
  CHECK_EQ(target_hcr_lse_sh1.dim(), 3);
  CHECK_EQ(query_snd.sizes(), key_snd.sizes());
  CHECK_EQ(query_snd.sizes(), value_snd.sizes());
  CHECK_EQ(query_snd.sizes(), output_snd.sizes());
  CHECK_EQ(target_hcr_lse_sh1.size(0), target_len);
  CHECK_EQ(target_hcr_lse_sh1.size(1), query_snd.size(1));
  CHECK_EQ(target_hcr_lse_sh1.size(2), 1);
  CHECK_EQ(query_snd.size(0),
           history_len + context_len + realtime_len + target_len);
  CHECK_GT(history_len, 0);
  CHECK_GT(context_len, 0);
  CHECK_GT(realtime_len, 0);
  CHECK_GT(target_len, 0);
  CHECK_GT(sm_scale, 0.0);
  CHECK_EQ(query_snd.size(1), key_snd.size(1));

  c10::cuda::CUDAGuard guard(query_snd.device());
  switch (query_snd.size(2)) {
    case 32:
      launch_mtgr_fused_no_match_attention_fallback_kernel<32>(
          query_snd,
          key_snd,
          value_snd,
          history_len,
          context_len,
          realtime_len,
          target_len,
          sm_scale,
          output_snd,
          target_hcr_lse_sh1);
      return;
    case 64:
      launch_mtgr_fused_no_match_attention_fallback_kernel<64>(
          query_snd,
          key_snd,
          value_snd,
          history_len,
          context_len,
          realtime_len,
          target_len,
          sm_scale,
          output_snd,
          target_hcr_lse_sh1);
      return;
    case 128:
      if (at::cuda::getCurrentDeviceProperties()->major >= 7) {
        if (use_mtgr_sm90_q64_regfrag_kv96_experimental()) {
          launch_mtgr_fused_no_match_attention_wmma128_q64_regfrag_kv96(
              query_snd,
              key_snd,
              value_snd,
              history_len,
              context_len,
              realtime_len,
              target_len,
              sm_scale,
              output_snd,
              target_hcr_lse_sh1);
        } else if (use_mtgr_sm90_q64_regfrag_kv80_experimental()) {
          launch_mtgr_fused_no_match_attention_wmma128_q64_regfrag_kv80(
              query_snd,
              key_snd,
              value_snd,
              history_len,
              context_len,
              realtime_len,
              target_len,
              sm_scale,
              output_snd,
              target_hcr_lse_sh1);
        } else if (use_mtgr_sm90_q64_regfrag_kv64_experimental()) {
          launch_mtgr_fused_no_match_attention_wmma128_q64_regfrag_kv64(
              query_snd,
              key_snd,
              value_snd,
              history_len,
              context_len,
              realtime_len,
              target_len,
              sm_scale,
              output_snd,
              target_hcr_lse_sh1);
        } else if (use_mtgr_sm90_q64_regfrag_experimental()) {
          launch_mtgr_fused_no_match_attention_wmma128_q64_regfrag_kv96(
              query_snd,
              key_snd,
              value_snd,
              history_len,
              context_len,
              realtime_len,
              target_len,
              sm_scale,
              output_snd,
              target_hcr_lse_sh1);
        } else if (use_mtgr_sm90_q32_regpv_experimental()) {
          launch_mtgr_fused_no_match_attention_wmma128_q32_regpv(
              query_snd,
              key_snd,
              value_snd,
              history_len,
              context_len,
              realtime_len,
              target_len,
              sm_scale,
              output_snd,
              target_hcr_lse_sh1);
        } else if (use_mtgr_sm90_q16_regpv_experimental()) {
          launch_mtgr_fused_no_match_attention_wmma128_q16_regpv(
              query_snd,
              key_snd,
              value_snd,
              history_len,
              context_len,
              realtime_len,
              target_len,
              sm_scale,
              output_snd,
              target_hcr_lse_sh1);
        } else {
          launch_mtgr_fused_no_match_attention_wmma128_q32(query_snd,
                                                           key_snd,
                                                           value_snd,
                                                           history_len,
                                                           context_len,
                                                           realtime_len,
                                                           target_len,
                                                           sm_scale,
                                                           output_snd,
                                                           target_hcr_lse_sh1);
        }
      } else {
        launch_mtgr_fused_no_match_attention_fallback_kernel<128>(
            query_snd,
            key_snd,
            value_snd,
            history_len,
            context_len,
            realtime_len,
            target_len,
            sm_scale,
            output_snd,
            target_hcr_lse_sh1);
      }
      return;
    default:
      CHECK(false) << "Unsupported head_dim for mtgr fused no-match kernel: "
                   << query_snd.size(2);
  }
}

}  // namespace xllm::kernel::cuda::test
