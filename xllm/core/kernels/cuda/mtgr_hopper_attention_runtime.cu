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

#include "mtgr_hopper_attention_runtime.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

#include "core/util/mtgr_nvtx.h"

// Reuse the physically split Hopper implementation while exposing a
// production-facing wrapper symbol.
#include "mtgr_ragged_hopper_attention_kernel.cu"

namespace xllm::kernel::cuda {
namespace {

__global__ void mtgr_kv_cache_writeback_bf16_kernel(
    const __nv_bfloat16* __restrict__ key_snd,
    const __nv_bfloat16* __restrict__ value_snd,
    __nv_bfloat16* __restrict__ key_cache,
    __nv_bfloat16* __restrict__ value_cache,
    const int32_t* __restrict__ segment_offsets,
    const int32_t* __restrict__ q_seq_starts,
    const int32_t* __restrict__ matched_prefix_lens,
    const int32_t* __restrict__ block_table,
    int32_t num_segments,
    int32_t segment_offsets_stride,
    int32_t block_table_stride,
    int32_t block_size,
    int32_t num_kv_heads,
    int32_t head_dim) {
  const int32_t row = static_cast<int32_t>(blockIdx.x);
  const int64_t elems_per_token =
      static_cast<int64_t>(num_kv_heads) * head_dim;
  const int64_t linear =
      static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;

  const int32_t matched = matched_prefix_lens[row];
  const int32_t cacheable_end =
      segment_offsets[row * segment_offsets_stride + num_segments - 1];
  const int64_t write_elems =
      static_cast<int64_t>(cacheable_end - matched) * elems_per_token;
  if (linear >= write_elems) {
    return;
  }

  const int32_t token_delta = static_cast<int32_t>(linear / elems_per_token);
  const int32_t elem =
      static_cast<int32_t>(linear - token_delta * elems_per_token);
  const int32_t kv_head = elem / head_dim;
  const int32_t dim = elem - kv_head * head_dim;
  const int32_t logical_token = matched + token_delta;
  const int32_t logical_block = logical_token / block_size;
  const int32_t block_offset = logical_token - logical_block * block_size;
  const int32_t physical_block =
      block_table[row * block_table_stride + logical_block];
  const int64_t src_token = q_seq_starts[row] + token_delta;

  const int64_t src_idx =
      (src_token * num_kv_heads + kv_head) * head_dim + dim;
  const int64_t dst_idx =
      (((static_cast<int64_t>(physical_block) * block_size + block_offset) *
            num_kv_heads +
        kv_head) *
           head_dim) +
      dim;
  key_cache[dst_idx] = key_snd[src_idx];
  value_cache[dst_idx] = value_snd[src_idx];
}

}  // namespace

void mtgr_ragged_segment_attention_hopper_unified_cuda(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    const torch::Tensor& segment_offsets_i32,
    const torch::Tensor& segment_rules_i32,
    const torch::Tensor& q_seq_starts_i32,
    const torch::Tensor& matched_prefix_lens_i32,
    int64_t match_mode,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& block_table_i32,
    int64_t block_size,
    int64_t max_request_len,
    double sm_scale,
    torch::Tensor output_snd) {
  MTGR_NVTX_RANGE(1, "MTGR/kernel/runtime_wrapper");
  MTGR_TRACE(2) << "[KERNEL] runtime_wrapper begin match_mode=" << match_mode
                << " query=" << query_snd.sizes()
                << " key_cache=" << key_cache.sizes()
                << " block_table=" << block_table_i32.sizes();
  mtgr_ragged_segment_attention_hopper_unified_research_cuda(query_snd,
                                                             key_snd,
                                                             value_snd,
                                                             segment_offsets_i32,
                                                             segment_rules_i32,
                                                             q_seq_starts_i32,
                                                             matched_prefix_lens_i32,
                                                             match_mode,
                                                             key_cache,
                                                             value_cache,
                                                             block_table_i32,
                                                             block_size,
                                                             max_request_len,
                                                             sm_scale,
                                                             output_snd);
}

void mtgr_kv_cache_writeback_cuda(
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    const torch::Tensor& segment_offsets_i32,
    const torch::Tensor& q_seq_starts_i32,
    const torch::Tensor& matched_prefix_lens_i32,
    const torch::Tensor& block_table_i32,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    int64_t max_request_len) {
  MTGR_NVTX_RANGE(1, "MTGR/kernel/kv_writeback");
  if (key_snd.numel() == 0 || block_table_i32.numel() == 0 ||
      max_request_len <= 0) {
    return;
  }
  CHECK_EQ(key_snd.dim(), 3);
  CHECK_EQ(value_snd.sizes(), key_snd.sizes());
  CHECK_EQ(key_snd.scalar_type(), torch::kBFloat16);
  CHECK_EQ(value_snd.scalar_type(), torch::kBFloat16);
  CHECK_EQ(key_cache.scalar_type(), torch::kBFloat16);
  CHECK_EQ(value_cache.scalar_type(), torch::kBFloat16);
  CHECK_EQ(segment_offsets_i32.scalar_type(), torch::kInt32);
  CHECK_EQ(q_seq_starts_i32.scalar_type(), torch::kInt32);
  CHECK_EQ(matched_prefix_lens_i32.scalar_type(), torch::kInt32);
  CHECK_EQ(block_table_i32.scalar_type(), torch::kInt32);

  c10::cuda::CUDAGuard guard(key_snd.device());
  const int64_t batch_size = segment_offsets_i32.size(0);
  const int64_t num_segments = segment_offsets_i32.size(1) - 1;
  const int64_t num_kv_heads = key_snd.size(1);
  const int64_t head_dim = key_snd.size(2);
  CHECK_EQ(key_cache.size(2), num_kv_heads);
  CHECK_EQ(value_cache.size(2), num_kv_heads);
  CHECK_EQ(key_cache.size(3), head_dim);
  CHECK_EQ(value_cache.size(3), head_dim);

  const int64_t elems_per_request = max_request_len * num_kv_heads * head_dim;
  if (batch_size == 0 || elems_per_request == 0) {
    return;
  }

  constexpr int threads = 256;
  dim3 grid(static_cast<unsigned int>(batch_size),
            static_cast<unsigned int>((elems_per_request + threads - 1) /
                                      threads));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  mtgr_kv_cache_writeback_bf16_kernel<<<grid, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(
          key_snd.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(
          value_snd.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(key_cache.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(value_cache.data_ptr<at::BFloat16>()),
      segment_offsets_i32.data_ptr<int32_t>(),
      q_seq_starts_i32.data_ptr<int32_t>(),
      matched_prefix_lens_i32.data_ptr<int32_t>(),
      block_table_i32.data_ptr<int32_t>(),
      static_cast<int32_t>(num_segments),
      static_cast<int32_t>(segment_offsets_i32.stride(0)),
      static_cast<int32_t>(block_table_i32.stride(0)),
      static_cast<int32_t>(key_cache.size(1)),
      static_cast<int32_t>(num_kv_heads),
      static_cast<int32_t>(head_dim));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace xllm::kernel::cuda
