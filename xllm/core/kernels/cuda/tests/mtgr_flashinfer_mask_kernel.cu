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

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <torch/torch.h>

namespace {

__device__ __forceinline__ bool one_stage_visible(int64_t q,
                                                  int64_t k,
                                                  int64_t history_len,
                                                  int64_t context_len,
                                                  int64_t realtime_len) {
  const int64_t context_end = history_len + context_len;
  const int64_t realtime_end = context_end + realtime_len;

  if (q < history_len) {
    return k < history_len && k <= q;
  }
  if (q < context_end) {
    return k < context_end;
  }
  if (q < realtime_end) {
    return k < context_end || (k >= context_end && k <= q);
  }
  return k < realtime_end || k == q;
}

__device__ __forceinline__ bool rt_tgt_hcr_trapezoid_visible(
    int64_t q,
    int64_t k,
    int64_t history_len,
    int64_t context_len,
    int64_t realtime_len) {
  const int64_t context_end = history_len + context_len;
  if (q < realtime_len) {
    return k < context_end + q + 1;
  }
  return k < context_end + realtime_len;
}

__device__ __forceinline__ bool partial_rt_one_stage_visible(
    int64_t q,
    int64_t k,
    int64_t matched_prefix_len,
    int64_t realtime_unmatched_len) {
  if (q < realtime_unmatched_len) {
    return k < matched_prefix_len + q + 1;
  }
  return k < matched_prefix_len + realtime_unmatched_len ||
         k == matched_prefix_len + q;
}

__global__ void build_mtgr_packed_mask_kernel(uint8_t* __restrict__ packed_mask,
                                              int64_t q_len,
                                              int64_t kv_len,
                                              int64_t history_len,
                                              int64_t context_len,
                                              int64_t realtime_len,
                                              int64_t mask_kind,
                                              int64_t num_bytes) {
  const int64_t byte_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (byte_idx >= num_bytes) {
    return;
  }

  uint8_t packed = 0;
  const int64_t bit_base = byte_idx * 8;
#pragma unroll
  for (int bit = 0; bit < 8; ++bit) {
    const int64_t linear = bit_base + bit;
    const int64_t q = linear / kv_len;
    const int64_t k = linear - q * kv_len;
    if (q >= q_len) {
      continue;
    }

    bool visible = false;
    if (mask_kind == 0) {
      visible = one_stage_visible(q, k, history_len, context_len, realtime_len);
    } else if (mask_kind == 1) {
      visible = rt_tgt_hcr_trapezoid_visible(
          q, k, history_len, context_len, realtime_len);
    } else {
      visible = partial_rt_one_stage_visible(q, k, history_len, realtime_len);
    }
    if (visible) {
      packed |= static_cast<uint8_t>(1u << bit);
    }
  }
  packed_mask[byte_idx] = packed;
}

template <typename scalar_t>
__global__ void merge_two_attention_results_kernel(
    const scalar_t* __restrict__ out_a,
    const float* __restrict__ lse_a,
    const scalar_t* __restrict__ out_b,
    const float* __restrict__ lse_b,
    scalar_t* __restrict__ merged_out,
    int64_t target_len,
    int64_t num_heads,
    int64_t head_dim) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t num_rows = target_len * num_heads;
  if (row >= num_rows) {
    return;
  }

  __shared__ float shared_wa;
  __shared__ float shared_wb;
  __shared__ float shared_inv_denom;

  if (threadIdx.x == 0) {
    const float la = lse_a[row];
    const float lb = lse_b[row];
    const float m = fmaxf(la, lb);
    const float wa = exp2f(la - m);
    const float wb = exp2f(lb - m);
    shared_wa = wa;
    shared_wb = wb;
    shared_inv_denom = 1.0f / (wa + wb);
  }
  __syncthreads();

  const int64_t base = row * head_dim;
  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    const float va = static_cast<float>(out_a[base + d]);
    const float vb = static_cast<float>(out_b[base + d]);
    const float merged = (shared_wa * va + shared_wb * vb) * shared_inv_denom;
    merged_out[base + d] = static_cast<scalar_t>(merged);
  }
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

template <typename scalar_t, int kWarpsPerBlock>
__global__ void merge_target_diag_attention_warp_kernel(
    const scalar_t* __restrict__ hcr_out,
    const float* __restrict__ hcr_lse,
    const scalar_t* __restrict__ target_query,
    const scalar_t* __restrict__ target_key,
    const scalar_t* __restrict__ target_value,
    scalar_t* __restrict__ merged_out,
    int64_t target_len,
    int64_t num_heads,
    int64_t head_dim,
    float sm_scale_log2e) {
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int64_t row =
      (static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock) + warp_id;
  const int64_t num_rows = target_len * num_heads;
  if (row >= num_rows) {
    return;
  }

  const int64_t base = row * head_dim;
  float dot = 0.0f;
  for (int64_t d = lane; d < head_dim; d += 32) {
    dot += static_cast<float>(target_query[base + d]) *
           static_cast<float>(target_key[base + d]);
  }
  dot = warp_reduce_sum(dot);

  float hcr_weight = 0.0f;
  float diag_weight = 0.0f;
  if (lane == 0) {
    const float diag_lse = dot * sm_scale_log2e;
    const float hcr_lse_v = hcr_lse[row];
    const float m = fmaxf(hcr_lse_v, diag_lse);
    const float wh = exp2f(hcr_lse_v - m);
    const float wd = exp2f(diag_lse - m);
    const float inv_denom = 1.0f / (wh + wd);
    hcr_weight = wh * inv_denom;
    diag_weight = wd * inv_denom;
  }
  hcr_weight = __shfl_sync(0xffffffff, hcr_weight, 0);
  diag_weight = __shfl_sync(0xffffffff, diag_weight, 0);

  for (int64_t d = lane; d < head_dim; d += 32) {
    const float hcr_v = static_cast<float>(hcr_out[base + d]);
    const float diag_v = static_cast<float>(target_value[base + d]);
    merged_out[base + d] =
        static_cast<scalar_t>(hcr_weight * hcr_v + diag_weight * diag_v);
  }
}

}  // namespace

namespace xllm::kernel::cuda::test {

void build_mtgr_packed_mask(torch::Tensor packed_mask,
                            int64_t q_len,
                            int64_t kv_len,
                            int64_t history_len,
                            int64_t context_len,
                            int64_t realtime_len,
                            int64_t mask_kind) {
  CHECK(packed_mask.defined());
  CHECK(packed_mask.is_cuda());
  CHECK_EQ(packed_mask.scalar_type(), torch::kUInt8);
  CHECK_EQ(packed_mask.dim(), 1);
  CHECK_GT(q_len, 0);
  CHECK_GT(kv_len, 0);
  CHECK(mask_kind == 0 || mask_kind == 1 || mask_kind == 2);

  const int64_t num_bytes = (q_len * kv_len + 7) / 8;
  CHECK_GE(packed_mask.numel(), num_bytes);

  c10::cuda::CUDAGuard guard(packed_mask.device());
  const int threads = 256;
  const int blocks = static_cast<int>((num_bytes + threads - 1) / threads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  build_mtgr_packed_mask_kernel<<<blocks, threads, 0, stream>>>(
      packed_mask.data_ptr<uint8_t>(),
      q_len,
      kv_len,
      history_len,
      context_len,
      realtime_len,
      mask_kind,
      num_bytes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void merge_two_attention_results_cuda(const torch::Tensor& out_a_snd,
                                      const torch::Tensor& lse_a_sh1,
                                      const torch::Tensor& out_b_snd,
                                      const torch::Tensor& lse_b_sh1,
                                      torch::Tensor merged_out_snd) {
  CHECK(out_a_snd.defined());
  CHECK(lse_a_sh1.defined());
  CHECK(out_b_snd.defined());
  CHECK(lse_b_sh1.defined());
  CHECK(merged_out_snd.defined());
  CHECK(out_a_snd.is_cuda());
  CHECK(lse_a_sh1.is_cuda());
  CHECK(out_b_snd.is_cuda());
  CHECK(lse_b_sh1.is_cuda());
  CHECK(merged_out_snd.is_cuda());
  CHECK_EQ(out_a_snd.dim(), 3);
  CHECK_EQ(out_b_snd.dim(), 3);
  CHECK_EQ(merged_out_snd.dim(), 3);
  CHECK_EQ(lse_a_sh1.dim(), 3);
  CHECK_EQ(lse_b_sh1.dim(), 3);
  CHECK_EQ(out_a_snd.sizes(), out_b_snd.sizes());
  CHECK_EQ(out_a_snd.sizes(), merged_out_snd.sizes());
  CHECK_EQ(lse_a_sh1.sizes(), lse_b_sh1.sizes());
  CHECK_EQ(lse_a_sh1.size(0), out_a_snd.size(0));
  CHECK_EQ(lse_a_sh1.size(1), out_a_snd.size(1));
  CHECK_EQ(lse_a_sh1.size(2), 1);
  CHECK_EQ(out_a_snd.scalar_type(), out_b_snd.scalar_type());
  CHECK_EQ(out_a_snd.scalar_type(), merged_out_snd.scalar_type());
  CHECK_EQ(lse_a_sh1.scalar_type(), torch::kFloat32);
  CHECK_EQ(lse_b_sh1.scalar_type(), torch::kFloat32);
  CHECK(out_a_snd.is_contiguous());
  CHECK(out_b_snd.is_contiguous());
  CHECK(merged_out_snd.is_contiguous());
  CHECK(lse_a_sh1.is_contiguous());
  CHECK(lse_b_sh1.is_contiguous());

  const int64_t target_len = out_a_snd.size(0);
  const int64_t num_heads = out_a_snd.size(1);
  const int64_t head_dim = out_a_snd.size(2);
  CHECK_GT(target_len, 0);
  CHECK_GT(num_heads, 0);
  CHECK_GT(head_dim, 0);

  c10::cuda::CUDAGuard guard(out_a_snd.device());
  const int threads = head_dim <= 64 ? 64 : 128;
  const int blocks = static_cast<int>(target_len * num_heads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      out_a_snd.scalar_type(), "merge_two_attention_results_cuda", [&] {
        merge_two_attention_results_kernel<scalar_t>
            <<<blocks, threads, 0, stream>>>(
                out_a_snd.data_ptr<scalar_t>(),
                lse_a_sh1.data_ptr<float>(),
                out_b_snd.data_ptr<scalar_t>(),
                lse_b_sh1.data_ptr<float>(),
                merged_out_snd.data_ptr<scalar_t>(),
                target_len,
                num_heads,
                head_dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void merge_target_diag_attention_cuda(const torch::Tensor& hcr_out_snd,
                                      const torch::Tensor& hcr_lse_sh1,
                                      const torch::Tensor& target_query_snd,
                                      const torch::Tensor& target_key_snd,
                                      const torch::Tensor& target_value_snd,
                                      double sm_scale,
                                      torch::Tensor merged_out_snd) {
  CHECK(hcr_out_snd.defined());
  CHECK(hcr_lse_sh1.defined());
  CHECK(target_query_snd.defined());
  CHECK(target_key_snd.defined());
  CHECK(target_value_snd.defined());
  CHECK(merged_out_snd.defined());
  CHECK(hcr_out_snd.is_cuda());
  CHECK(hcr_lse_sh1.is_cuda());
  CHECK(target_query_snd.is_cuda());
  CHECK(target_key_snd.is_cuda());
  CHECK(target_value_snd.is_cuda());
  CHECK(merged_out_snd.is_cuda());
  CHECK_EQ(hcr_out_snd.dim(), 3);
  CHECK_EQ(target_query_snd.dim(), 3);
  CHECK_EQ(target_key_snd.dim(), 3);
  CHECK_EQ(target_value_snd.dim(), 3);
  CHECK_EQ(merged_out_snd.dim(), 3);
  CHECK_EQ(hcr_lse_sh1.dim(), 3);
  CHECK_EQ(hcr_out_snd.sizes(), target_query_snd.sizes());
  CHECK_EQ(hcr_out_snd.sizes(), target_key_snd.sizes());
  CHECK_EQ(hcr_out_snd.sizes(), target_value_snd.sizes());
  CHECK_EQ(hcr_out_snd.sizes(), merged_out_snd.sizes());
  CHECK_EQ(hcr_lse_sh1.size(0), hcr_out_snd.size(0));
  CHECK_EQ(hcr_lse_sh1.size(1), hcr_out_snd.size(1));
  CHECK_EQ(hcr_lse_sh1.size(2), 1);
  CHECK_EQ(hcr_out_snd.scalar_type(), target_query_snd.scalar_type());
  CHECK_EQ(hcr_out_snd.scalar_type(), target_key_snd.scalar_type());
  CHECK_EQ(hcr_out_snd.scalar_type(), target_value_snd.scalar_type());
  CHECK_EQ(hcr_out_snd.scalar_type(), merged_out_snd.scalar_type());
  CHECK_EQ(hcr_lse_sh1.scalar_type(), torch::kFloat32);
  CHECK(hcr_out_snd.is_contiguous());
  CHECK(hcr_lse_sh1.is_contiguous());
  CHECK(target_query_snd.is_contiguous());
  CHECK(target_key_snd.is_contiguous());
  CHECK(target_value_snd.is_contiguous());
  CHECK(merged_out_snd.is_contiguous());

  const int64_t target_len = hcr_out_snd.size(0);
  const int64_t num_heads = hcr_out_snd.size(1);
  const int64_t head_dim = hcr_out_snd.size(2);
  CHECK_GT(target_len, 0);
  CHECK_GT(num_heads, 0);
  CHECK_GT(head_dim, 0);

  c10::cuda::CUDAGuard guard(hcr_out_snd.device());
  constexpr int kWarpsPerBlock = 4;
  const int threads = kWarpsPerBlock * 32;
  const int blocks = static_cast<int>(
      (target_len * num_heads + kWarpsPerBlock - 1) / kWarpsPerBlock);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  constexpr double kLog2E = 1.4426950408889634;
  const float sm_scale_log2e = static_cast<float>(sm_scale * kLog2E);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      hcr_out_snd.scalar_type(), "merge_target_diag_attention_cuda", [&] {
        merge_target_diag_attention_warp_kernel<scalar_t, kWarpsPerBlock>
            <<<blocks, threads, 0, stream>>>(
                hcr_out_snd.data_ptr<scalar_t>(),
                hcr_lse_sh1.data_ptr<float>(),
                target_query_snd.data_ptr<scalar_t>(),
                target_key_snd.data_ptr<scalar_t>(),
                target_value_snd.data_ptr<scalar_t>(),
                merged_out_snd.data_ptr<scalar_t>(),
                target_len,
                num_heads,
                head_dim,
                sm_scale_log2e);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace xllm::kernel::cuda::test
