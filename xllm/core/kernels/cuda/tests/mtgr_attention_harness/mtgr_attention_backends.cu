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

#include "mtgr_attention_contract.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/common/global_flags.h"
#include "core/platform/device.h"
#include "core/util/mtgr_nvtx.h"
#include "kernels/cuda/utils.h"
#include "layers/cuda/mtgr_attention.h"

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
    if (mask_kind == 0 &&
        one_stage_visible(q, k, history_len, context_len, realtime_len)) {
      packed |= static_cast<uint8_t>(1u << bit);
    }
  }
  packed_mask[byte_idx] = packed;
}

__global__ void build_mtgr_bsr_packed_mask_kernel(
    uint8_t* __restrict__ packed_mask,
    const int32_t* __restrict__ packed_mask_indptr,
    const int32_t* __restrict__ bsr_indptr,
    const int32_t* __restrict__ bsr_indices,
    int32_t block_rows,
    int32_t row_block_size,
    int32_t col_block_size,
    int64_t total_len,
    int64_t history_len,
    int64_t context_len,
    int64_t realtime_len,
    int64_t total_packed_bytes) {
  const int64_t byte_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (byte_idx >= total_packed_bytes) {
    return;
  }

  int32_t lo = 0;
  int32_t hi = block_rows;
  while (lo + 1 < hi) {
    const int32_t mid = lo + (hi - lo) / 2;
    if (static_cast<int64_t>(packed_mask_indptr[mid]) <= byte_idx) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  const int32_t row_block = lo;
  const int32_t local_byte =
      static_cast<int32_t>(byte_idx - packed_mask_indptr[row_block]);
  const int32_t q_rows = min(
      row_block_size,
      static_cast<int32_t>(total_len -
                           static_cast<int64_t>(row_block) * row_block_size));
  const int32_t nz_begin = bsr_indptr[row_block];
  const int32_t nz_end = bsr_indptr[row_block + 1];
  const int32_t nz_count = nz_end - nz_begin;
  const int64_t row_bits =
      static_cast<int64_t>(q_rows) * nz_count * col_block_size;

  uint8_t packed = 0;
  const int64_t bit_base = static_cast<int64_t>(local_byte) * 8;
#pragma unroll
  for (int bit = 0; bit < 8; ++bit) {
    const int64_t linear = bit_base + bit;
    if (linear >= row_bits) {
      continue;
    }
    const int64_t q_row = linear / (nz_count * col_block_size);
    const int64_t rem = linear - q_row * nz_count * col_block_size;
    const int64_t nz = rem / col_block_size;
    const int64_t col = rem - nz * col_block_size;
    const int64_t q =
        static_cast<int64_t>(row_block) * row_block_size + q_row;
    const int64_t k =
        static_cast<int64_t>(bsr_indices[nz_begin + nz]) * col_block_size + col;
    if (q < total_len && k < total_len &&
        one_stage_visible(q, k, history_len, context_len, realtime_len)) {
      packed |= static_cast<uint8_t>(1u << bit);
    }
  }
  packed_mask[byte_idx] = packed;
}

__global__ void mtgr_kv_writeback_bf16_kernel(
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
      (static_cast<int64_t>(blockIdx.y) * blockDim.x) + threadIdx.x;

  const int32_t matched = matched_prefix_lens[row];
  const int32_t cacheable_end =
      segment_offsets[row * segment_offsets_stride + num_segments];
  const int64_t write_elems =
      static_cast<int64_t>(cacheable_end - matched) * elems_per_token;
  if (linear >= write_elems) {
    return;
  }

  const int32_t token_delta = static_cast<int32_t>(linear / elems_per_token);
  const int32_t elem = static_cast<int32_t>(linear - token_delta * elems_per_token);
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

namespace xllm::kernel::cuda::test::mtgr_attention_harness {
namespace {

using xllm::layer::MTGRAttentionImpl;

struct FlashinferWorkspaceBuffers {
  torch::Tensor float_workspace;
  torch::Tensor int_workspace;
  torch::Tensor page_locked_int_workspace;
};

FlashinferWorkspaceBuffers& get_workspace_buffers(const torch::Device& device) {
  static thread_local FlashinferWorkspaceBuffers ws;
  const bool need_init =
      !ws.float_workspace.defined() || ws.float_workspace.device() != device;
  if (need_init) {
    ws.float_workspace =
        torch::empty({FLAGS_flashinfer_workspace_buffer_size},
                     torch::TensorOptions().dtype(torch::kUInt8).device(device));
    ws.int_workspace =
        torch::empty({8 * 1024 * 1024},
                     torch::TensorOptions().dtype(torch::kUInt8).device(device));
    ws.page_locked_int_workspace = torch::empty(
        {ws.int_workspace.size(0)},
        torch::TensorOptions()
            .dtype(torch::kUInt8)
            .device(torch::kCPU)
            .pinned_memory(true));
  }
  return ws;
}

ffi::Array<int64_t> deep_copy_plan_info(const ffi::Array<int64_t>& src) {
  if (!src.defined()) {
    return ffi::Array<int64_t>();
  }
  std::vector<int64_t> copied;
  copied.reserve(src.size());
  for (const auto& v : src) {
    copied.push_back(v);
  }
  return ffi::Array<int64_t>(copied.begin(), copied.end());
}

struct FlashinferPlan {
  std::string uri;
  ffi::Array<int64_t> plan_info;
};

torch::Tensor make_seq_indptr_host(int64_t seq_len) {
  return torch::tensor(
      {0, static_cast<int32_t>(seq_len)},
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
}

int64_t ceil_div(int64_t x, int64_t y) {
  CHECK_GT(y, 0);
  return (x + y - 1) / y;
}

bool ranges_overlap(int64_t a_begin,
                    int64_t a_end,
                    int64_t b_begin,
                    int64_t b_end) {
  return std::max(a_begin, b_begin) < std::min(a_end, b_end);
}

bool mtgr_block_has_visible_token(int64_t q_begin,
                                  int64_t q_end,
                                  int64_t k_begin,
                                  int64_t k_end,
                                  int64_t history_len,
                                  int64_t context_len,
                                  int64_t realtime_len,
                                  int64_t total_len) {
  q_end = std::min(q_end, total_len);
  k_end = std::min(k_end, total_len);
  if (q_begin >= q_end || k_begin >= k_end) {
    return false;
  }

  const int64_t context_end = history_len + context_len;
  const int64_t realtime_end = context_end + realtime_len;

  const int64_t hist_q_begin = std::max(q_begin, int64_t{0});
  const int64_t hist_q_end = std::min(q_end, history_len);
  if (hist_q_begin < hist_q_end && k_begin < history_len &&
      k_begin <= hist_q_end - 1) {
    return true;
  }

  const int64_t ctx_q_begin = std::max(q_begin, history_len);
  const int64_t ctx_q_end = std::min(q_end, context_end);
  if (ctx_q_begin < ctx_q_end && k_begin < context_end) {
    return true;
  }

  const int64_t rt_q_begin = std::max(q_begin, context_end);
  const int64_t rt_q_end = std::min(q_end, realtime_end);
  if (rt_q_begin < rt_q_end && k_begin <= rt_q_end - 1) {
    return true;
  }

  const int64_t tgt_q_begin = std::max(q_begin, realtime_end);
  const int64_t tgt_q_end = std::min(q_end, total_len);
  if (tgt_q_begin < tgt_q_end) {
    if (k_begin < realtime_end) {
      return true;
    }
    if (ranges_overlap(tgt_q_begin, tgt_q_end, k_begin, k_end)) {
      return true;
    }
  }

  return false;
}

struct BlockSparseInputs {
  int64_t row_block_size = 0;
  int64_t col_block_size = 0;
  int64_t block_rows = 0;
  int64_t block_cols = 0;
  int64_t padded_kv_len = 0;
  int64_t max_packed_bytes_per_row = 0;
  torch::Tensor qo_indptr_host;
  torch::Tensor bsr_indptr_host;
  torch::Tensor bsr_indices_host;
  torch::Tensor kv_lens_host;
  torch::Tensor qo_indptr;
  torch::Tensor bsr_indptr;
  torch::Tensor bsr_indices;
  torch::Tensor paged_kv_last_page_len;
  torch::Tensor packed_mask_indptr;
  torch::Tensor packed_mask;
};

FlashinferPlan build_custom_mask_prefill_plan(
    const torch::Device& device,
    torch::ScalarType query_dtype,
    torch::ScalarType key_dtype,
    torch::ScalarType output_dtype,
    int64_t head_dim_qk,
    int64_t head_dim_vo,
    int64_t num_qo_heads,
    int64_t num_kv_heads,
    const torch::Tensor& q_cu_seq_lens_host,
    const torch::Tensor& kv_cu_seq_lens_host) {
  auto& ws = get_workspace_buffers(device);
  xllm::kernel::cuda::bind_tvmffi_stream_to_current_torch_stream(device);

  const std::string backend =
      xllm::kernel::cuda::determine_attention_backend(
          /*pos_encoding_mode=*/0,
          /*use_fp16_qk_reduction=*/false,
          /*use_custom_mask=*/true);
  FlashinferPlan plan;
  plan.uri = xllm::kernel::cuda::get_batch_prefill_uri(
      backend,
      query_dtype,
      key_dtype,
      output_dtype,
      q_cu_seq_lens_host.scalar_type(),
      head_dim_qk,
      head_dim_vo,
      /*pos_encoding_mode=*/0,
      /*use_sliding_window=*/false,
      /*use_logits_soft_cap=*/false,
      /*use_fp16_qk_reduction=*/false);

  auto kv_len_arr_host =
      kv_cu_seq_lens_host.slice(0, 1) - kv_cu_seq_lens_host.slice(0, 0, -1);
  const int64_t total_num_rows = q_cu_seq_lens_host[-1].item<int64_t>();
  const int64_t batch_size = q_cu_seq_lens_host.size(0) - 1;
  auto plan_func = xllm::kernel::cuda::get_function(plan.uri, "plan");
  ffi::Array<int64_t> plan_result =
      xllm::Device::is_support_sm90a()
          ? plan_func(xllm::kernel::cuda::to_ffi_tensor(ws.float_workspace),
                      xllm::kernel::cuda::to_ffi_tensor(ws.int_workspace),
                      xllm::kernel::cuda::to_ffi_tensor(
                          ws.page_locked_int_workspace),
                      xllm::kernel::cuda::to_ffi_tensor(q_cu_seq_lens_host),
                      xllm::kernel::cuda::to_ffi_tensor(kv_cu_seq_lens_host),
                      xllm::kernel::cuda::to_ffi_tensor(kv_len_arr_host),
                      total_num_rows,
                      batch_size,
                      num_qo_heads,
                      num_kv_heads,
                      /*page_size=*/1,
                      /*enable_cuda_graph=*/false,
                      head_dim_qk,
                      head_dim_vo,
                      /*causal=*/false,
                      /*window_size_left=*/-1)
                .cast<ffi::Array<int64_t>>()
          : plan_func(xllm::kernel::cuda::to_ffi_tensor(ws.float_workspace),
                      xllm::kernel::cuda::to_ffi_tensor(ws.int_workspace),
                      xllm::kernel::cuda::to_ffi_tensor(
                          ws.page_locked_int_workspace),
                      xllm::kernel::cuda::to_ffi_tensor(q_cu_seq_lens_host),
                      xllm::kernel::cuda::to_ffi_tensor(kv_cu_seq_lens_host),
                      xllm::kernel::cuda::to_ffi_tensor(kv_len_arr_host),
                      total_num_rows,
                      batch_size,
                      num_qo_heads,
                      num_kv_heads,
                      /*page_size=*/1,
                      /*enable_cuda_graph=*/false,
                      head_dim_qk,
                      head_dim_vo,
                      /*causal=*/false,
                      /*window_size_left=*/-1,
                      /*fixed_split_size=*/-1,
                      /*disable_split_kv=*/false,
                      /*num_colocated_ctas=*/0)
                .cast<ffi::Array<int64_t>>();
  plan.plan_info = deep_copy_plan_info(plan_result);
  return plan;
}

FlashinferPlan build_block_sparse_prefill_plan(
    const torch::Device& device,
    torch::ScalarType query_dtype,
    torch::ScalarType key_dtype,
    torch::ScalarType output_dtype,
    int64_t head_dim_qk,
    int64_t head_dim_vo,
    int64_t num_qo_heads,
    int64_t num_kv_heads,
    int64_t page_size,
    const torch::Tensor& qo_indptr_host,
    const torch::Tensor& paged_kv_indptr_host,
    const torch::Tensor& kv_lens_host,
    bool causal) {
  CHECK_EQ(qo_indptr_host.device().type(), torch::kCPU);
  CHECK_EQ(paged_kv_indptr_host.device().type(), torch::kCPU);
  CHECK_EQ(kv_lens_host.device().type(), torch::kCPU);
  CHECK_EQ(qo_indptr_host.scalar_type(), torch::kInt32);
  CHECK_EQ(paged_kv_indptr_host.scalar_type(), torch::kInt32);
  CHECK_EQ(kv_lens_host.scalar_type(), torch::kInt32);

  auto& ws = get_workspace_buffers(device);
  xllm::kernel::cuda::bind_tvmffi_stream_to_current_torch_stream(device);

  const std::string backend =
      xllm::kernel::cuda::determine_attention_backend(
          /*pos_encoding_mode=*/0,
          /*use_fp16_qk_reduction=*/false,
          /*use_custom_mask=*/true);
  FlashinferPlan plan;
  plan.uri = xllm::kernel::cuda::get_batch_prefill_uri(
      backend,
      query_dtype,
      key_dtype,
      output_dtype,
      paged_kv_indptr_host.scalar_type(),
      head_dim_qk,
      head_dim_vo,
      /*pos_encoding_mode=*/0,
      /*use_sliding_window=*/false,
      /*use_logits_soft_cap=*/false,
      /*use_fp16_qk_reduction=*/false);

  const int64_t total_num_rows = qo_indptr_host[-1].item<int64_t>();
  const int64_t batch_size = qo_indptr_host.size(0) - 1;
  auto plan_func = xllm::kernel::cuda::get_function(plan.uri, "plan");
  ffi::Array<int64_t> plan_result =
      xllm::Device::is_support_sm90a()
          ? plan_func(xllm::kernel::cuda::to_ffi_tensor(ws.float_workspace),
                      xllm::kernel::cuda::to_ffi_tensor(ws.int_workspace),
                      xllm::kernel::cuda::to_ffi_tensor(
                          ws.page_locked_int_workspace),
                      xllm::kernel::cuda::to_ffi_tensor(qo_indptr_host),
                      xllm::kernel::cuda::to_ffi_tensor(paged_kv_indptr_host),
                      xllm::kernel::cuda::to_ffi_tensor(kv_lens_host),
                      total_num_rows,
                      batch_size,
                      num_qo_heads,
                      num_kv_heads,
                      page_size,
                      /*enable_cuda_graph=*/false,
                      head_dim_qk,
                      head_dim_vo,
                      causal,
                      /*window_size_left=*/-1)
                .cast<ffi::Array<int64_t>>()
          : plan_func(xllm::kernel::cuda::to_ffi_tensor(ws.float_workspace),
                      xllm::kernel::cuda::to_ffi_tensor(ws.int_workspace),
                      xllm::kernel::cuda::to_ffi_tensor(
                          ws.page_locked_int_workspace),
                      xllm::kernel::cuda::to_ffi_tensor(qo_indptr_host),
                      xllm::kernel::cuda::to_ffi_tensor(paged_kv_indptr_host),
                      xllm::kernel::cuda::to_ffi_tensor(kv_lens_host),
                      total_num_rows,
                      batch_size,
                      num_qo_heads,
                      num_kv_heads,
                      page_size,
                      /*enable_cuda_graph=*/false,
                      head_dim_qk,
                      head_dim_vo,
                      causal,
                      /*window_size_left=*/-1,
                      /*fixed_split_size=*/-1,
                      /*disable_split_kv=*/false,
                      /*num_colocated_ctas=*/0)
                .cast<ffi::Array<int64_t>>();
  plan.plan_info = deep_copy_plan_info(plan_result);
  return plan;
}

torch::Tensor get_cached_mask_indptr(const torch::Device& device,
                                     int64_t num_bytes) {
  static thread_local std::unordered_map<int64_t, torch::Tensor> cached_indptr;
  const int64_t device_idx = static_cast<int64_t>(device.index());
  const int64_t key = (device_idx << 32) ^ num_bytes;
  auto it = cached_indptr.find(key);
  if (it != cached_indptr.end()) {
    return it->second;
  }
  auto indptr =
      torch::tensor({0, static_cast<int32_t>(num_bytes)},
                    torch::TensorOptions().dtype(torch::kInt32).device(device))
          .contiguous();
  it = cached_indptr.emplace(key, indptr).first;
  return it->second;
}

int64_t packed_mask_num_bytes(int64_t q_len, int64_t kv_len) {
  return (q_len * kv_len + 7) / 8;
}

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
  CHECK_EQ(mask_kind, 0) << "harness base only builds full MTGR one-stage mask";

  const int64_t num_bytes = packed_mask_num_bytes(q_len, kv_len);
  CHECK_GE(packed_mask.numel(), num_bytes);

  c10::cuda::CUDAGuard guard(packed_mask.device());
  constexpr int threads = 256;
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

torch::Tensor reserve_reusable_packed_mask_buffer(const torch::Device& device,
                                                  int64_t num_bytes) {
  static thread_local std::unordered_map<int64_t, torch::Tensor> buffers;
  const int64_t device_idx = static_cast<int64_t>(device.index());
  auto& buffer = buffers[device_idx];
  if (!buffer.defined() || buffer.numel() < num_bytes ||
      buffer.device() != device) {
    buffer = torch::empty({num_bytes},
                          torch::TensorOptions()
                              .dtype(torch::kUInt8)
                              .device(device))
                 .contiguous();
  }
  return buffer.narrow(0, 0, num_bytes);
}

torch::Tensor build_one_stage_packed_mask_reusing_buffer(
    const torch::Device& device,
    int64_t history_len,
    int64_t context_len,
    int64_t realtime_len,
    int64_t target_len) {
  const int64_t seq_len =
      history_len + context_len + realtime_len + target_len;
  auto packed_mask = reserve_reusable_packed_mask_buffer(
      device, packed_mask_num_bytes(seq_len, seq_len));
  build_mtgr_packed_mask(packed_mask,
                         seq_len,
                         seq_len,
                         history_len,
                         context_len,
                         realtime_len,
                         /*mask_kind=*/0);
  return packed_mask;
}

BlockSparseInputs build_mtgr_block_sparse_inputs(const torch::Device& device,
                                                 int64_t history_len,
                                                 int64_t context_len,
                                                 int64_t realtime_len,
                                                 int64_t target_len,
                                                 int64_t row_block_size,
                                                 int64_t col_block_size) {
  const int64_t total_len =
      history_len + context_len + realtime_len + target_len;
  const int64_t block_rows = ceil_div(total_len, row_block_size);
  const int64_t block_cols = ceil_div(total_len, col_block_size);
  const int64_t padded_kv_len = block_cols * col_block_size;

  std::vector<int32_t> qo_indptr_host;
  std::vector<int32_t> bsr_indptr_host;
  std::vector<int32_t> bsr_indices_host;
  std::vector<int32_t> kv_lens_host;
  std::vector<int32_t> packed_mask_indptr_host;
  qo_indptr_host.reserve(static_cast<size_t>(block_rows + 1));
  bsr_indptr_host.reserve(static_cast<size_t>(block_rows + 1));
  kv_lens_host.reserve(static_cast<size_t>(block_rows));
  packed_mask_indptr_host.reserve(static_cast<size_t>(block_rows + 1));

  qo_indptr_host.push_back(0);
  bsr_indptr_host.push_back(0);
  packed_mask_indptr_host.push_back(0);

  int64_t max_packed_bytes_per_row = 0;
  for (int64_t qb = 0; qb < block_rows; ++qb) {
    const int64_t q_begin = qb * row_block_size;
    const int64_t q_end = std::min(q_begin + row_block_size, total_len);
    for (int64_t kb = 0; kb < block_cols; ++kb) {
      const int64_t k_begin = kb * col_block_size;
      const int64_t k_end = std::min(k_begin + col_block_size, padded_kv_len);
      if (mtgr_block_has_visible_token(q_begin,
                                       q_end,
                                       k_begin,
                                       k_end,
                                       history_len,
                                       context_len,
                                       realtime_len,
                                       total_len)) {
        bsr_indices_host.push_back(static_cast<int32_t>(kb));
      }
    }

    const int64_t nnz_for_row =
        static_cast<int64_t>(bsr_indices_host.size()) - bsr_indptr_host.back();
    CHECK_GT(nnz_for_row, 0) << "each MTGR query block must see at least one KV block";
    const int64_t q_rows = q_end - q_begin;
    const int64_t row_bits = q_rows * nnz_for_row * col_block_size;
    const int64_t row_packed_bytes = ceil_div(row_bits, 8);
    max_packed_bytes_per_row =
        std::max(max_packed_bytes_per_row, row_packed_bytes);

    qo_indptr_host.push_back(static_cast<int32_t>(q_end));
    bsr_indptr_host.push_back(
        static_cast<int32_t>(bsr_indices_host.size()));
    kv_lens_host.push_back(static_cast<int32_t>(nnz_for_row * col_block_size));
    packed_mask_indptr_host.push_back(
        static_cast<int32_t>(packed_mask_indptr_host.back() +
                             row_packed_bytes));
  }

  auto i32_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  auto i32_dev = torch::TensorOptions().dtype(torch::kInt32).device(device);

  BlockSparseInputs inputs;
  inputs.row_block_size = row_block_size;
  inputs.col_block_size = col_block_size;
  inputs.block_rows = block_rows;
  inputs.block_cols = block_cols;
  inputs.padded_kv_len = padded_kv_len;
  inputs.max_packed_bytes_per_row = max_packed_bytes_per_row;
  inputs.qo_indptr_host = torch::tensor(qo_indptr_host, i32_cpu).contiguous();
  inputs.bsr_indptr_host = torch::tensor(bsr_indptr_host, i32_cpu).contiguous();
  inputs.bsr_indices_host =
      torch::tensor(bsr_indices_host, i32_cpu).contiguous();
  inputs.kv_lens_host = torch::tensor(kv_lens_host, i32_cpu).contiguous();
  inputs.qo_indptr = inputs.qo_indptr_host.to(device).contiguous();
  inputs.bsr_indptr = inputs.bsr_indptr_host.to(device).contiguous();
  inputs.bsr_indices = inputs.bsr_indices_host.to(device).contiguous();
  inputs.paged_kv_last_page_len =
      torch::full({block_rows},
                  static_cast<int32_t>(col_block_size),
                  i32_dev)
          .contiguous();
  inputs.packed_mask_indptr =
      torch::tensor(packed_mask_indptr_host, i32_dev).contiguous();
  inputs.packed_mask =
      torch::empty({packed_mask_indptr_host.back()},
                   torch::TensorOptions().dtype(torch::kUInt8).device(device))
          .contiguous();
  return inputs;
}

void build_mtgr_block_sparse_packed_mask(BlockSparseInputs* inputs,
                                         int64_t history_len,
                                         int64_t context_len,
                                         int64_t realtime_len,
                                         int64_t target_len) {
  CHECK(inputs != nullptr);
  const int64_t total_len =
      history_len + context_len + realtime_len + target_len;
  if (inputs->packed_mask.numel() == 0) {
    return;
  }

  constexpr int threads = 256;
  const int blocks = static_cast<int>(
      (inputs->packed_mask.numel() + threads - 1) / threads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  build_mtgr_bsr_packed_mask_kernel<<<blocks, threads, 0, stream>>>(
      inputs->packed_mask.data_ptr<uint8_t>(),
      inputs->packed_mask_indptr.data_ptr<int32_t>(),
      inputs->bsr_indptr.data_ptr<int32_t>(),
      inputs->bsr_indices.data_ptr<int32_t>(),
      static_cast<int32_t>(inputs->block_rows),
      static_cast<int32_t>(inputs->row_block_size),
      static_cast<int32_t>(inputs->col_block_size),
      total_len,
      history_len,
      context_len,
      realtime_len,
      inputs->packed_mask.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::pair<torch::Tensor, torch::Tensor> pad_kv_to_block_sparse_pages(
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    int64_t padded_kv_len,
    int64_t col_block_size) {
  CHECK_EQ(key_snd.dim(), 3);
  CHECK_EQ(value_snd.sizes(), key_snd.sizes());
  CHECK_GE(padded_kv_len, key_snd.size(0));
  torch::Tensor padded_key;
  torch::Tensor padded_value;
  if (padded_kv_len == key_snd.size(0)) {
    padded_key = key_snd;
    padded_value = value_snd;
  } else {
    padded_key =
        torch::empty({padded_kv_len, key_snd.size(1), key_snd.size(2)},
                     key_snd.options());
    padded_value =
        torch::empty({padded_kv_len, value_snd.size(1), value_snd.size(2)},
                     value_snd.options());
    padded_key.narrow(0, 0, key_snd.size(0)).copy_(key_snd);
    padded_value.narrow(0, 0, value_snd.size(0)).copy_(value_snd);
    const int64_t tail = padded_kv_len - key_snd.size(0);
    if (tail > 0) {
      padded_key.narrow(0, key_snd.size(0), tail).zero_();
      padded_value.narrow(0, value_snd.size(0), tail).zero_();
    }
  }
  return {padded_key.view({padded_kv_len / col_block_size,
                           col_block_size,
                           key_snd.size(1),
                           key_snd.size(2)}),
          padded_value.view({padded_kv_len / col_block_size,
                             col_block_size,
                             value_snd.size(1),
                             value_snd.size(2)})};
}

torch::Tensor run_custom_mask_prefill(const torch::Tensor& query_snd,
                                      const torch::Tensor& key_snd,
                                      const torch::Tensor& value_snd,
                                      const torch::Tensor& packed_mask,
                                      double sm_scale) {
  const auto device = query_snd.device();
  const int64_t q_len = query_snd.size(0);
  const int64_t kv_len = key_snd.size(0);
  const int64_t num_heads = query_snd.size(1);
  const int64_t num_kv_heads = key_snd.size(1);
  const int64_t head_dim = query_snd.size(2);
  auto q_cu_host = make_seq_indptr_host(q_len);
  auto kv_cu_host = make_seq_indptr_host(kv_len);
  auto q_cu_dev = q_cu_host.to(device);
  auto kv_cu_dev = kv_cu_host.to(device);
  auto output_snd = torch::empty_like(query_snd);
  auto plan = build_custom_mask_prefill_plan(device,
                                             query_snd.scalar_type(),
                                             key_snd.scalar_type(),
                                             query_snd.scalar_type(),
                                             head_dim,
                                             head_dim,
                                             num_heads,
                                             num_kv_heads,
                                             q_cu_host,
                                             kv_cu_host);
  auto& ws = get_workspace_buffers(device);
  auto mask_indptr = get_cached_mask_indptr(device, packed_mask.numel());
  xllm::kernel::cuda::get_function(plan.uri, "ragged_run")(
      xllm::kernel::cuda::to_ffi_tensor(ws.float_workspace),
      xllm::kernel::cuda::to_ffi_tensor(ws.int_workspace),
      plan.plan_info,
      xllm::kernel::cuda::to_ffi_tensor(query_snd),
      xllm::kernel::cuda::to_ffi_tensor(key_snd),
      xllm::kernel::cuda::to_ffi_tensor(value_snd),
      xllm::kernel::cuda::to_ffi_tensor(q_cu_dev),
      xllm::kernel::cuda::to_ffi_tensor(kv_cu_dev),
      xllm::kernel::cuda::to_ffi_tensor(output_snd),
      ffi::Optional<ffi::Tensor>(),
      /*mask_mode_code=*/2,
      /*kv_layout_code=*/0,
      /*window_left=*/-1,
      xllm::kernel::cuda::support_pdl(),
      xllm::kernel::cuda::to_ffi_tensor(packed_mask),
      xllm::kernel::cuda::to_ffi_tensor(mask_indptr),
      /*maybe_alibi_slopes=*/ffi::Optional<ffi::Tensor>(),
      /*maybe_prefix_len_ptr=*/ffi::Optional<ffi::Tensor>(),
      /*maybe_token_pos_in_items_ptr=*/ffi::Optional<ffi::Tensor>(),
      /*maybe_max_item_len_ptr=*/ffi::Optional<ffi::Tensor>(),
      /*logits_soft_cap=*/0.0,
      sm_scale,
      /*rope_rcp_scale=*/1.0,
      /*rope_rcp_theta=*/1.0 / 10000.0,
      /*token_pos_in_items_len=*/0);
  return output_snd;
}

torch::Tensor run_block_sparse_custom_mask_prefill(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_pages,
    const torch::Tensor& value_pages,
    const BlockSparseInputs& inputs,
    double sm_scale) {
  CHECK_EQ(query_snd.dim(), 3);
  CHECK_EQ(key_pages.dim(), 4);
  CHECK_EQ(value_pages.sizes(), key_pages.sizes());
  const auto device = query_snd.device();
  const int64_t q_len = query_snd.size(0);
  const int64_t num_heads = query_snd.size(1);
  const int64_t num_kv_heads = key_pages.size(2);
  const int64_t head_dim = query_snd.size(2);
  auto output_snd = torch::empty_like(query_snd);

  auto plan = build_block_sparse_prefill_plan(device,
                                              query_snd.scalar_type(),
                                              key_pages.scalar_type(),
                                              query_snd.scalar_type(),
                                              head_dim,
                                              head_dim,
                                              num_heads,
                                              num_kv_heads,
                                              inputs.col_block_size,
                                              inputs.qo_indptr_host,
                                              inputs.bsr_indptr_host,
                                              inputs.kv_lens_host,
                                              /*causal=*/false);
  auto& ws = get_workspace_buffers(device);
  xllm::kernel::cuda::get_function(plan.uri, "paged_run")(
      xllm::kernel::cuda::to_ffi_tensor(ws.float_workspace),
      xllm::kernel::cuda::to_ffi_tensor(ws.int_workspace),
      plan.plan_info,
      xllm::kernel::cuda::to_ffi_tensor(query_snd),
      xllm::kernel::cuda::to_ffi_tensor(key_pages),
      xllm::kernel::cuda::to_ffi_tensor(value_pages),
      xllm::kernel::cuda::to_ffi_tensor(inputs.qo_indptr),
      xllm::kernel::cuda::to_ffi_tensor(inputs.bsr_indptr),
      xllm::kernel::cuda::to_ffi_tensor(inputs.bsr_indices),
      xllm::kernel::cuda::to_ffi_tensor(inputs.paged_kv_last_page_len),
      xllm::kernel::cuda::to_ffi_tensor(output_snd),
      ffi::Optional<ffi::Tensor>(),
      /*mask_mode_code=*/2,
      /*kv_layout_code=*/0,
      /*window_left=*/-1,
      xllm::kernel::cuda::support_pdl(),
      xllm::kernel::cuda::to_ffi_tensor(inputs.packed_mask),
      xllm::kernel::cuda::to_ffi_tensor(inputs.packed_mask_indptr),
      /*maybe_alibi_slopes=*/ffi::Optional<ffi::Tensor>(),
      /*maybe_prefix_len_ptr=*/ffi::Optional<ffi::Tensor>(),
      /*maybe_token_pos_in_items_ptr=*/ffi::Optional<ffi::Tensor>(),
      /*maybe_max_item_len_ptr=*/ffi::Optional<ffi::Tensor>(),
      /*logits_soft_cap=*/0.0,
      sm_scale,
      /*rope_rcp_scale=*/1.0,
      /*rope_rcp_theta=*/1.0 / 10000.0,
      /*token_pos_in_items_len=*/0);
  CHECK_EQ(output_snd.size(0), q_len);
  return output_snd;
}

torch::Tensor run_mtgr_one_stage_full_base_nvtx_only(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    int64_t history_len,
    int64_t context_len,
    int64_t realtime_len,
    int64_t target_len,
    double sm_scale,
    bool emit_nvtx) {
  const auto device = query_snd.device();
  torch::Tensor packed_mask;
  if (emit_nvtx) {
    xllm::MtgrNvtxRange range(2, "MTGR/harness/mtgr_attention/base/mask_build");
    packed_mask = build_one_stage_packed_mask_reusing_buffer(
        device, history_len, context_len, realtime_len, target_len);
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  } else {
    packed_mask = build_one_stage_packed_mask_reusing_buffer(
        device, history_len, context_len, realtime_len, target_len);
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }

  torch::Tensor output;
  if (emit_nvtx) {
    xllm::MtgrNvtxRange range(
        2, "MTGR/harness/mtgr_attention/base/one_stage_flashinfer");
    output = run_custom_mask_prefill(
        query_snd, key_snd, value_snd, packed_mask, sm_scale);
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  } else {
    output = run_custom_mask_prefill(
        query_snd, key_snd, value_snd, packed_mask, sm_scale);
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }
  return output;
}

torch::Tensor run_mtgr_block_sparse_full_base_nvtx_only(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    int64_t history_len,
    int64_t context_len,
    int64_t realtime_len,
    int64_t target_len,
    int64_t block_size,
    double sm_scale,
    bool emit_nvtx) {
  const auto device = query_snd.device();
  BlockSparseInputs inputs;
  if (emit_nvtx) {
    xllm::MtgrNvtxRange range(
        2, "MTGR/harness/mtgr_attention/base_block_sparse/bsr_build");
    inputs = build_mtgr_block_sparse_inputs(device,
                                            history_len,
                                            context_len,
                                            realtime_len,
                                            target_len,
                                            block_size,
                                            block_size);
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  } else {
    inputs = build_mtgr_block_sparse_inputs(device,
                                            history_len,
                                            context_len,
                                            realtime_len,
                                            target_len,
                                            block_size,
                                            block_size);
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }

  if (emit_nvtx) {
    xllm::MtgrNvtxRange range(
        2, "MTGR/harness/mtgr_attention/base_block_sparse/mask_build");
    build_mtgr_block_sparse_packed_mask(
        &inputs, history_len, context_len, realtime_len, target_len);
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  } else {
    build_mtgr_block_sparse_packed_mask(
        &inputs, history_len, context_len, realtime_len, target_len);
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }

  torch::Tensor key_pages;
  torch::Tensor value_pages;
  if (emit_nvtx) {
    xllm::MtgrNvtxRange range(
        2, "MTGR/harness/mtgr_attention/base_block_sparse/kv_pad");
    std::tie(key_pages, value_pages) = pad_kv_to_block_sparse_pages(
        key_snd, value_snd, inputs.padded_kv_len, inputs.col_block_size);
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  } else {
    std::tie(key_pages, value_pages) = pad_kv_to_block_sparse_pages(
        key_snd, value_snd, inputs.padded_kv_len, inputs.col_block_size);
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }

  torch::Tensor output;
  if (emit_nvtx) {
    xllm::MtgrNvtxRange range(
        2, "MTGR/harness/mtgr_attention/base_block_sparse/block_sparse_flashinfer");
    output = run_block_sparse_custom_mask_prefill(
        query_snd, key_pages, value_pages, inputs, sm_scale);
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  } else {
    output = run_block_sparse_custom_mask_prefill(
        query_snd, key_pages, value_pages, inputs, sm_scale);
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }
  return output;
}

class FullFlashinferBaseBackend final : public IMTGRAttentionBackend {
 public:
  explicit FullFlashinferBaseBackend(MTGRAttentionHarnessMetadata metadata)
      : metadata_(std::move(metadata)) {}

  std::string name() const override { return "full_flashinfer_base"; }

  std::string nvtx_root_name(
      const MTGRAttentionHarnessMetadata& metadata) const override {
    return std::string("MTGR/harness/mtgr_attention/base/") +
           metadata.mode_name();
  }

  std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward(
      const xllm::layer::AttentionMetadata& attn_metadata,
      torch::Tensor& query,
      torch::Tensor& key,
      torch::Tensor& value,
      xllm::KVCache& kv_cache) override {
    (void)kv_cache;
    (void)attn_metadata;
    const double scale =
        1.0 / std::sqrt(static_cast<double>(metadata_.head_dim));
    auto full_out = run_mtgr_one_stage_full_base_nvtx_only(
        query.view(
            {metadata_.total_len(), metadata_.num_heads, metadata_.head_dim}),
        key.view({metadata_.total_len(),
                  metadata_.num_kv_heads,
                  metadata_.head_dim}),
        value.view({metadata_.total_len(),
                    metadata_.num_kv_heads,
                    metadata_.head_dim}),
        metadata_.history,
        metadata_.context,
        metadata_.realtime,
        metadata_.target,
        scale,
        FLAGS_mtgr_nvtx_level >= 2);
    return {full_out
                .view({metadata_.total_len(),
                       metadata_.num_heads * metadata_.head_dim})
                .contiguous(),
            std::nullopt};
  }

 private:
  MTGRAttentionHarnessMetadata metadata_;
};

class BlockSparseFlashinferBaseBackend final : public IMTGRAttentionBackend {
 public:
  explicit BlockSparseFlashinferBaseBackend(
      MTGRAttentionHarnessMetadata metadata)
      : metadata_(std::move(metadata)) {}

  std::string name() const override { return "block_sparse_flashinfer_base"; }

  std::string nvtx_root_name(
      const MTGRAttentionHarnessMetadata& metadata) const override {
    return std::string("MTGR/harness/mtgr_attention/base_block_sparse/") +
           metadata.mode_name();
  }

  std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward(
      const xllm::layer::AttentionMetadata& attn_metadata,
      torch::Tensor& query,
      torch::Tensor& key,
      torch::Tensor& value,
      xllm::KVCache& kv_cache) override {
    (void)kv_cache;
    (void)attn_metadata;
    const double scale =
        1.0 / std::sqrt(static_cast<double>(metadata_.head_dim));
    auto full_out = run_mtgr_block_sparse_full_base_nvtx_only(
        query.view(
            {metadata_.total_len(), metadata_.num_heads, metadata_.head_dim}),
        key.view({metadata_.total_len(),
                  metadata_.num_kv_heads,
                  metadata_.head_dim}),
        value.view({metadata_.total_len(),
                    metadata_.num_kv_heads,
                    metadata_.head_dim}),
        metadata_.history,
        metadata_.context,
        metadata_.realtime,
        metadata_.target,
        metadata_.block_size,
        scale,
        FLAGS_mtgr_nvtx_level >= 2);
    return {full_out
                .view({metadata_.total_len(),
                       metadata_.num_heads * metadata_.head_dim})
                .contiguous(),
            std::nullopt};
  }

 private:
  MTGRAttentionHarnessMetadata metadata_;
};

class HopperUnifiedBackend final : public IMTGRAttentionBackend {
 public:
  explicit HopperUnifiedBackend(MTGRAttentionHarnessMetadata metadata)
      : metadata_(std::move(metadata)) {}

  std::string name() const override { return "hopper_unified"; }

  std::string nvtx_root_name(
      const MTGRAttentionHarnessMetadata& metadata) const override {
    return std::string("MTGR/harness/mtgr_attention/hopper/") +
           metadata.mode_name();
  }

  std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward(
      const xllm::layer::AttentionMetadata& attn_metadata,
      torch::Tensor& query,
      torch::Tensor& key,
      torch::Tensor& value,
      xllm::KVCache& kv_cache) override {
    const float scale =
        1.0f / std::sqrt(static_cast<float>(metadata_.head_dim));
    MTGRAttentionImpl attention(metadata_.num_heads,
                                metadata_.head_dim,
                                scale,
                                metadata_.num_kv_heads);
    xllm::MtgrNvtxRange range(2, "MTGR/harness/mtgr_attention/hopper_forward");
    auto result = attention.forward(attn_metadata, query, key, value, kv_cache);
    {
      xllm::MtgrNvtxRange sync_range(
          2, "MTGR/harness/mtgr_attention/device_sync");
      CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }
    return result;
  }

 private:
  MTGRAttentionHarnessMetadata metadata_;
};

}  // namespace

std::unique_ptr<IMTGRAttentionBackend> make_full_flashinfer_base_backend(
    const MTGRAttentionHarnessMetadata& metadata) {
  return std::make_unique<FullFlashinferBaseBackend>(metadata);
}

std::unique_ptr<IMTGRAttentionBackend> make_block_sparse_flashinfer_base_backend(
    const MTGRAttentionHarnessMetadata& metadata) {
  return std::make_unique<BlockSparseFlashinferBaseBackend>(metadata);
}

std::unique_ptr<IMTGRAttentionBackend> make_hopper_unified_backend(
    const MTGRAttentionHarnessMetadata& metadata) {
  return std::make_unique<HopperUnifiedBackend>(metadata);
}

void run_mtgr_kv_writeback_cuda(
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    const xllm::layer::AttentionMetadata& attn_metadata,
    xllm::KVCache& kv_cache) {
  MTGR_NVTX_RANGE(2, "MTGR/harness/mtgr_attention/writeback/cuda_launch");
  CHECK_EQ(key_snd.dim(), 3);
  CHECK_EQ(value_snd.sizes(), key_snd.sizes());
  CHECK_EQ(key_snd.scalar_type(), torch::kBFloat16);
  CHECK_EQ(value_snd.scalar_type(), torch::kBFloat16);

  auto key_cache = kv_cache.get_k_cache();
  auto value_cache = kv_cache.get_v_cache();
  CHECK_EQ(key_cache.scalar_type(), torch::kBFloat16);
  CHECK_EQ(value_cache.scalar_type(), torch::kBFloat16);

  c10::cuda::CUDAGuard guard(key_snd.device());
  const int64_t batch_size = attn_metadata.mtgr_segment_offsets_i32.size(0);
  const int64_t num_segments =
      attn_metadata.mtgr_segment_offsets_i32.size(1) - 1;
  const int64_t max_query_len = attn_metadata.max_query_len;
  const int64_t num_kv_heads = key_snd.size(1);
  const int64_t head_dim = key_snd.size(2);
  const int64_t elems_per_request = max_query_len * num_kv_heads * head_dim;
  if (batch_size == 0 || elems_per_request == 0) {
    return;
  }

  constexpr int threads = 256;
  dim3 grid(static_cast<unsigned int>(batch_size),
            static_cast<unsigned int>((elems_per_request + threads - 1) /
                                      threads));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  mtgr_kv_writeback_bf16_kernel<<<grid, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(
          key_snd.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(
          value_snd.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(
          key_cache.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(
          value_cache.data_ptr<at::BFloat16>()),
      attn_metadata.mtgr_segment_offsets_i32.data_ptr<int32_t>(),
      attn_metadata.mtgr_q_seq_starts_i32.data_ptr<int32_t>(),
      attn_metadata.mtgr_matched_prefix_lens_i32.data_ptr<int32_t>(),
      attn_metadata.block_table.data_ptr<int32_t>(),
      static_cast<int32_t>(num_segments),
      static_cast<int32_t>(attn_metadata.mtgr_segment_offsets_i32.stride(0)),
      static_cast<int32_t>(attn_metadata.block_table.stride(0)),
      static_cast<int32_t>(key_cache.size(1)),
      static_cast<int32_t>(num_kv_heads),
      static_cast<int32_t>(head_dim));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace xllm::kernel::cuda::test::mtgr_attention_harness
