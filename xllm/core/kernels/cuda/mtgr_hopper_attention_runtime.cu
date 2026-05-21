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
#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "core/common/global_flags.h"
#include "core/platform/device.h"
#include "core/util/mtgr_nvtx.h"
#include "utils.h"

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
    int32_t cacheable_end_offset_index,
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
      segment_offsets[row * segment_offsets_stride + cacheable_end_offset_index];
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

__device__ __forceinline__ int mtgr_token_mask_find_segment(
    int32_t q_local,
    const int32_t* __restrict__ offsets_row,
    int32_t num_segments) {
  const int32_t request_start = offsets_row[0];
#pragma unroll 1
  for (int32_t seg = 0; seg < num_segments; ++seg) {
    const int32_t seg_end = offsets_row[seg + 1] - request_start;
    if (q_local < seg_end) {
      return seg;
    }
  }
  return num_segments - 1;
}

__device__ __forceinline__ bool mtgr_token_mask_visible(
    int32_t q_local,
    int32_t k_local,
    const int32_t* __restrict__ offsets_row,
    const int32_t* __restrict__ segment_rules,
    int32_t num_segments) {
  constexpr int32_t kRuleCausal = 0;
  constexpr int32_t kRuleFull = 1;
  constexpr int32_t kRuleDiagonal = 2;

  const int32_t request_start = offsets_row[0];
  const int32_t seg = mtgr_token_mask_find_segment(
      q_local, offsets_row, num_segments);
  const int32_t seg_start = offsets_row[seg] - request_start;
  const int32_t seg_end = offsets_row[seg + 1] - request_start;
  const int32_t rule = segment_rules[seg];

  int32_t visible_end = seg_end;
  if (rule == kRuleCausal) {
    visible_end = q_local + 1;
  } else if (rule == kRuleDiagonal) {
    visible_end = seg_start;
  } else if (rule == kRuleFull) {
    visible_end = seg_end;
  }

  return k_local < visible_end ||
         (rule == kRuleDiagonal && k_local == q_local);
}

__global__ void mtgr_build_flashinfer_token_mask_kernel(
    uint8_t* __restrict__ packed_mask,
    const int32_t* __restrict__ mask_indptr,
    const int32_t* __restrict__ segment_offsets,
    const int32_t* __restrict__ segment_rules,
    const int32_t* __restrict__ matched_prefix_lens,
    int32_t num_segments,
    int32_t segment_offsets_stride,
    int32_t max_mask_bytes_per_request) {
  const int32_t row = static_cast<int32_t>(blockIdx.x);
  const int64_t byte_delta =
      static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  const int32_t mask_begin = mask_indptr[row];
  const int32_t mask_end = mask_indptr[row + 1];
  const int32_t request_bytes = mask_end - mask_begin;
  if (byte_delta >= max_mask_bytes_per_request ||
      byte_delta >= request_bytes) {
    return;
  }

  const int32_t* offsets_row =
      segment_offsets + static_cast<int64_t>(row) * segment_offsets_stride;
  const int32_t request_start = offsets_row[0];
  const int32_t total_len = offsets_row[num_segments] - request_start;
  const int32_t matched = matched_prefix_lens[row];
  const int32_t q_len = total_len - matched;
  const int64_t bit_base = byte_delta * 8;
  uint8_t packed = 0;
#pragma unroll
  for (int bit = 0; bit < 8; ++bit) {
    const int64_t linear = bit_base + bit;
    const int32_t q_delta = static_cast<int32_t>(linear / total_len);
    if (q_delta >= q_len) {
      continue;
    }
    const int32_t k_local =
        static_cast<int32_t>(linear - static_cast<int64_t>(q_delta) *
                                          total_len);
    const int32_t q_local = matched + q_delta;
    if (mtgr_token_mask_visible(
            q_local, k_local, offsets_row, segment_rules, num_segments)) {
      packed |= static_cast<uint8_t>(1u << bit);
    }
  }
  packed_mask[mask_begin + byte_delta] = packed;
}

__global__ void mtgr_gather_full_kv_cache_bf16_kernel(
    const __nv_bfloat16* __restrict__ key_cache,
    const __nv_bfloat16* __restrict__ value_cache,
    __nv_bfloat16* __restrict__ full_key_snd,
    __nv_bfloat16* __restrict__ full_value_snd,
    const int32_t* __restrict__ segment_offsets,
    const int32_t* __restrict__ kv_seq_starts,
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

  const int32_t request_start = segment_offsets[row * segment_offsets_stride];
  const int32_t total_len =
      segment_offsets[row * segment_offsets_stride + num_segments] -
      request_start;
  const int64_t read_elems =
      static_cast<int64_t>(total_len) * elems_per_token;
  if (linear >= read_elems) {
    return;
  }

  const int32_t token = static_cast<int32_t>(linear / elems_per_token);
  const int32_t elem = static_cast<int32_t>(linear - token * elems_per_token);
  const int32_t kv_head = elem / head_dim;
  const int32_t dim = elem - kv_head * head_dim;
  const int32_t logical_block = token / block_size;
  const int32_t block_offset = token - logical_block * block_size;
  const int32_t physical_block =
      block_table[row * block_table_stride + logical_block];
  const int64_t src_idx =
      (((static_cast<int64_t>(physical_block) * block_size + block_offset) *
            num_kv_heads +
        kv_head) *
           head_dim) +
      dim;
  const int64_t dst_token = kv_seq_starts[row] + token;
  const int64_t dst_idx =
      (dst_token * num_kv_heads + kv_head) * head_dim + dim;
  full_key_snd[dst_idx] = key_cache[src_idx];
  full_value_snd[dst_idx] = value_cache[src_idx];
}

struct MtgrFlashinferWorkspaceBuffers {
  torch::Tensor float_workspace;
  torch::Tensor int_workspace;
  torch::Tensor page_locked_int_workspace;
};

MtgrFlashinferWorkspaceBuffers& mtgr_flashinfer_workspace(
    const torch::Device& device) {
  static thread_local MtgrFlashinferWorkspaceBuffers ws;
  const bool need_init = !ws.float_workspace.defined() ||
                         ws.float_workspace.device() != device;
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

ffi::Array<int64_t> mtgr_deep_copy_plan_info(const ffi::Array<int64_t>& src) {
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

struct MtgrFlashinferPlan {
  std::string uri;
  ffi::Array<int64_t> plan_info;
};

MtgrFlashinferPlan mtgr_build_token_mask_prefill_plan(
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
  auto& ws = mtgr_flashinfer_workspace(device);
  bind_tvmffi_stream_to_current_torch_stream(device);

  const std::string backend =
      determine_attention_backend(/*pos_encoding_mode=*/0,
                                  /*use_fp16_qk_reduction=*/false,
                                  /*use_custom_mask=*/true);
  MtgrFlashinferPlan plan;
  plan.uri = get_batch_prefill_uri(backend,
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
  auto plan_func = get_function(plan.uri, "plan");
  ffi::Array<int64_t> plan_result =
      plan_func(to_ffi_tensor(ws.float_workspace),
                to_ffi_tensor(ws.int_workspace),
                to_ffi_tensor(ws.page_locked_int_workspace),
                to_ffi_tensor(q_cu_seq_lens_host),
                to_ffi_tensor(kv_cu_seq_lens_host),
                to_ffi_tensor(kv_len_arr_host),
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
  plan.plan_info = mtgr_deep_copy_plan_info(plan_result);
  return plan;
}

int64_t mtgr_packed_mask_bytes(int64_t q_len, int64_t kv_len) {
  return (q_len * kv_len + 7) / 8;
}

torch::Tensor mtgr_make_i32_cpu_tensor(const std::vector<int32_t>& values) {
  auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  return torch::from_blob(
             const_cast<int32_t*>(values.data()),
             {static_cast<int64_t>(values.size())},
             options)
      .clone();
}

struct MtgrTokenMaskHostMetadata {
  int64_t batch_size = 0;
  int64_t num_segments = 0;
  int64_t total_kv_len = 0;
  int64_t total_mask_bytes = 0;
  int64_t max_kv_len = 0;
  int64_t max_mask_bytes_per_request = 0;
  std::vector<int32_t> q_cu_seq_lens;
  std::vector<int32_t> kv_cu_seq_lens;
  std::vector<int32_t> mask_indptr;
};

MtgrTokenMaskHostMetadata mtgr_build_token_mask_host_metadata(
    const torch::Tensor& segment_offsets_i32,
    const torch::Tensor& q_seq_starts_i32,
    const torch::Tensor& matched_prefix_lens_i32,
    int64_t total_q) {
  auto segment_offsets_cpu = segment_offsets_i32.to(torch::kCPU).contiguous();
  auto q_seq_starts_cpu = q_seq_starts_i32.to(torch::kCPU).contiguous();
  auto matched_cpu = matched_prefix_lens_i32.to(torch::kCPU).contiguous();

  MtgrTokenMaskHostMetadata metadata;
  metadata.batch_size = segment_offsets_cpu.size(0);
  metadata.num_segments = segment_offsets_cpu.size(1) - 1;
  metadata.q_cu_seq_lens.resize(metadata.batch_size + 1, 0);
  metadata.kv_cu_seq_lens.resize(metadata.batch_size + 1, 0);
  metadata.mask_indptr.resize(metadata.batch_size + 1, 0);

  const auto* offsets = segment_offsets_cpu.data_ptr<int32_t>();
  const auto* q_starts = q_seq_starts_cpu.data_ptr<int32_t>();
  const auto* matched = matched_cpu.data_ptr<int32_t>();
  const int64_t offsets_stride = segment_offsets_cpu.stride(0);
  for (int64_t row = 0; row < metadata.batch_size; ++row) {
    const int32_t request_start = offsets[row * offsets_stride];
    const int32_t total_len =
        offsets[row * offsets_stride + metadata.num_segments] - request_start;
    const int32_t q_start = q_starts[row];
    const int32_t q_end =
        row + 1 < metadata.batch_size ? q_starts[row + 1]
                                      : static_cast<int32_t>(total_q);
    const int32_t q_len = q_end - q_start;
    CHECK_EQ(q_len, total_len - matched[row])
        << "MTGR token-mask base expects production live-Q layout";
    const int64_t mask_bytes = mtgr_packed_mask_bytes(q_len, total_len);

    metadata.q_cu_seq_lens[row + 1] =
        metadata.q_cu_seq_lens[row] + q_len;
    metadata.kv_cu_seq_lens[row + 1] =
        metadata.kv_cu_seq_lens[row] + total_len;
    metadata.mask_indptr[row + 1] =
        metadata.mask_indptr[row] + static_cast<int32_t>(mask_bytes);
    metadata.total_kv_len += total_len;
    metadata.total_mask_bytes += mask_bytes;
    metadata.max_kv_len = std::max<int64_t>(metadata.max_kv_len, total_len);
    metadata.max_mask_bytes_per_request =
        std::max<int64_t>(metadata.max_mask_bytes_per_request, mask_bytes);
  }
  return metadata;
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

void mtgr_flashinfer_token_mask_attention_cuda(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    const torch::Tensor& segment_offsets_i32,
    const torch::Tensor& segment_rules_i32,
    const torch::Tensor& q_seq_starts_i32,
    const torch::Tensor& matched_prefix_lens_i32,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& block_table_i32,
    int64_t block_size,
    double sm_scale,
    torch::Tensor output_snd) {
  MTGR_NVTX_RANGE(1, "MTGR/kernel/flashinfer_token_mask_base");
  if (query_snd.numel() == 0) {
    return;
  }

  c10::cuda::CUDAGuard guard(query_snd.device());
  const auto device = query_snd.device();
  const int64_t total_q = query_snd.size(0);
  const int64_t num_q_heads = query_snd.size(1);
  const int64_t num_kv_heads = key_snd.size(1);
  const int64_t head_dim = query_snd.size(2);
  CHECK_EQ(query_snd.dim(), 3);
  CHECK_EQ(key_snd.dim(), 3);
  CHECK_EQ(value_snd.sizes(), key_snd.sizes());
  CHECK_EQ(key_snd.size(0), total_q)
      << "MTGR FlashInfer token-mask base expects live K input";
  CHECK_EQ(key_snd.size(2), head_dim);
  CHECK_EQ(key_cache.scalar_type(), torch::kBFloat16);
  CHECK_EQ(value_cache.scalar_type(), torch::kBFloat16);
  CHECK_EQ(key_cache.size(2), num_kv_heads);
  CHECK_EQ(value_cache.size(2), num_kv_heads);
  CHECK_EQ(key_cache.size(3), head_dim);
  CHECK_EQ(value_cache.size(3), head_dim);
  CHECK_EQ(block_table_i32.scalar_type(), torch::kInt32);

  MtgrTokenMaskHostMetadata host_meta;
  {
    MTGR_NVTX_RANGE(2,
                    "MTGR/kernel/flashinfer_token_mask_base/host_metadata");
    host_meta = mtgr_build_token_mask_host_metadata(segment_offsets_i32,
                                                    q_seq_starts_i32,
                                                    matched_prefix_lens_i32,
                                                    total_q);
  }

  auto q_cu_host = mtgr_make_i32_cpu_tensor(host_meta.q_cu_seq_lens);
  auto kv_cu_host = mtgr_make_i32_cpu_tensor(host_meta.kv_cu_seq_lens);
  auto mask_indptr_host = mtgr_make_i32_cpu_tensor(host_meta.mask_indptr);
  auto q_cu_dev = q_cu_host.to(device).contiguous();
  auto kv_cu_dev = kv_cu_host.to(device).contiguous();
  auto mask_indptr_dev = mask_indptr_host.to(device).contiguous();
  auto full_key_snd =
      torch::empty({host_meta.total_kv_len, num_kv_heads, head_dim},
                   key_snd.options());
  auto full_value_snd =
      torch::empty({host_meta.total_kv_len, num_kv_heads, head_dim},
                   value_snd.options());
  {
    MTGR_NVTX_RANGE(2,
                    "MTGR/kernel/flashinfer_token_mask_base/gather_full_kv");
    constexpr int threads = 256;
    const int64_t elems_per_request =
        host_meta.max_kv_len * num_kv_heads * head_dim;
    dim3 grid(static_cast<unsigned int>(host_meta.batch_size),
              static_cast<unsigned int>((elems_per_request + threads - 1) /
                                        threads));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    mtgr_gather_full_kv_cache_bf16_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            key_cache.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            value_cache.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(
            full_key_snd.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(
            full_value_snd.data_ptr<at::BFloat16>()),
        segment_offsets_i32.data_ptr<int32_t>(),
        kv_cu_dev.data_ptr<int32_t>(),
        block_table_i32.data_ptr<int32_t>(),
        static_cast<int32_t>(host_meta.num_segments),
        static_cast<int32_t>(segment_offsets_i32.stride(0)),
        static_cast<int32_t>(block_table_i32.stride(0)),
        static_cast<int32_t>(block_size),
        static_cast<int32_t>(num_kv_heads),
        static_cast<int32_t>(head_dim));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  auto packed_mask =
      torch::empty({host_meta.total_mask_bytes},
                   torch::TensorOptions().dtype(torch::kUInt8).device(device));
  {
    MTGR_NVTX_RANGE(2,
                    "MTGR/kernel/flashinfer_token_mask_base/mask_build");
    constexpr int threads = 256;
    dim3 grid(static_cast<unsigned int>(host_meta.batch_size),
              static_cast<unsigned int>(
                  (host_meta.max_mask_bytes_per_request + threads - 1) /
                  threads));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    mtgr_build_flashinfer_token_mask_kernel<<<grid, threads, 0, stream>>>(
        packed_mask.data_ptr<uint8_t>(),
        mask_indptr_dev.data_ptr<int32_t>(),
        segment_offsets_i32.data_ptr<int32_t>(),
        segment_rules_i32.data_ptr<int32_t>(),
        matched_prefix_lens_i32.data_ptr<int32_t>(),
        static_cast<int32_t>(host_meta.num_segments),
        static_cast<int32_t>(segment_offsets_i32.stride(0)),
        static_cast<int32_t>(host_meta.max_mask_bytes_per_request));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  {
    MTGR_NVTX_RANGE(2,
                    "MTGR/kernel/flashinfer_token_mask_base/one_stage_flashinfer");
    auto plan = mtgr_build_token_mask_prefill_plan(device,
                                                   query_snd.scalar_type(),
                                                   key_snd.scalar_type(),
                                                   output_snd.scalar_type(),
                                                   head_dim,
                                                   head_dim,
                                                   num_q_heads,
                                                   num_kv_heads,
                                                   q_cu_host,
                                                   kv_cu_host);
    auto& ws = mtgr_flashinfer_workspace(device);
    get_function(plan.uri, "ragged_run")(
        to_ffi_tensor(ws.float_workspace),
        to_ffi_tensor(ws.int_workspace),
        plan.plan_info,
        to_ffi_tensor(query_snd),
        to_ffi_tensor(full_key_snd),
        to_ffi_tensor(full_value_snd),
        to_ffi_tensor(q_cu_dev),
        to_ffi_tensor(kv_cu_dev),
        to_ffi_tensor(output_snd),
        ffi::Optional<ffi::Tensor>(),
        /*mask_mode_code=*/2,
        /*kv_layout_code=*/0,
        /*window_left=*/-1,
        support_pdl(),
        to_ffi_tensor(packed_mask),
        to_ffi_tensor(mask_indptr_dev),
        /*maybe_alibi_slopes=*/ffi::Optional<ffi::Tensor>(),
        /*maybe_prefix_len_ptr=*/ffi::Optional<ffi::Tensor>(),
        /*maybe_token_pos_in_items_ptr=*/ffi::Optional<ffi::Tensor>(),
        /*maybe_max_item_len_ptr=*/ffi::Optional<ffi::Tensor>(),
        /*logits_soft_cap=*/0.0,
        sm_scale,
        /*rope_rcp_scale=*/1.0,
        /*rope_rcp_theta=*/1.0 / 10000.0,
        /*token_pos_in_items_len=*/0);
  }
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

void mtgr_kv_cache_prefix_writeback_cuda(
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    const torch::Tensor& segment_offsets_i32,
    const torch::Tensor& q_seq_starts_i32,
    const torch::Tensor& matched_prefix_lens_i32,
    const torch::Tensor& block_table_i32,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    int64_t max_request_len) {
  MTGR_NVTX_RANGE(1, "MTGR/kernel/kv_prefix_writeback");
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
  CHECK_GE(num_segments, 1);
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
      static_cast<int32_t>(num_segments - 1),
      static_cast<int32_t>(segment_offsets_i32.stride(0)),
      static_cast<int32_t>(block_table_i32.stride(0)),
      static_cast<int32_t>(key_cache.size(1)),
      static_cast<int32_t>(num_kv_heads),
      static_cast<int32_t>(head_dim));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace xllm::kernel::cuda
