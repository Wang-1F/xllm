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

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <unordered_map>

namespace xllm::kernel::cuda::test {
namespace {

constexpr int kHeadDim = 128;
constexpr int kChunkedQueriesPerBlock = 8;
constexpr int kFusedQueriesPerBlock = 4;
constexpr int kTileKv = 32;
constexpr int kMaxChunks = 128;
constexpr int kChunkedThreadsPerBlock = kChunkedQueriesPerBlock * 32;
constexpr int kFusedThreadsPerBlock = kFusedQueriesPerBlock * 32;
constexpr int kWmmaWarpsPerBlock = 8;
constexpr int kWmmaThreadsPerBlock = kWmmaWarpsPerBlock * 32;
constexpr float kLog2E = 1.4426950408889634f;
constexpr float kNegInf = -INFINITY;

enum class FullAttentionKernelMode {
  kAuto,
  kScalarChunked,
  kFusedSplitKv,
  kWmmaChunked,
  kWmmaFused,
};

struct SmallQAttentionWorkspace {
  torch::Tensor partial_out;
  torch::Tensor partial_max;
  torch::Tensor partial_sum;
};

struct SmallQAttentionHalfWorkspace {
  torch::Tensor partial_out;
  torch::Tensor partial_max;
  torch::Tensor partial_sum;
};

bool iequals(const std::string& lhs, const char* rhs) {
  size_t i = 0;
  for (; i < lhs.size() && rhs[i] != '\0'; ++i) {
    if (std::tolower(static_cast<unsigned char>(lhs[i])) !=
        std::tolower(static_cast<unsigned char>(rhs[i]))) {
      return false;
    }
  }
  return i == lhs.size() && rhs[i] == '\0';
}

FullAttentionKernelMode parse_full_attention_mode() {
  const char* env = std::getenv("XLLM_MTGR_CUDA_FULL_ATTN_MODE");
  if (env == nullptr || env[0] == '\0') {
    return FullAttentionKernelMode::kAuto;
  }
  const std::string value(env);
  if (iequals(value, "auto")) {
    return FullAttentionKernelMode::kAuto;
  }
  if (iequals(value, "scalar_chunked")) {
    return FullAttentionKernelMode::kScalarChunked;
  }
  if (iequals(value, "fused_splitkv") || iequals(value, "fused")) {
    return FullAttentionKernelMode::kFusedSplitKv;
  }
  if (iequals(value, "wmma_chunked") || iequals(value, "wmma")) {
    return FullAttentionKernelMode::kWmmaChunked;
  }
  if (iequals(value, "wmma_fused") || iequals(value, "wmma_full")) {
    return FullAttentionKernelMode::kWmmaFused;
  }
  LOG(WARNING) << "Unknown XLLM_MTGR_CUDA_FULL_ATTN_MODE=" << value
               << ", fallback to auto";
  return FullAttentionKernelMode::kAuto;
}

int64_t ceil_div_int64(int64_t x, int64_t y) { return (x + y - 1) / y; }

int parse_env_int_or_default(const char* name, int default_value) {
  const char* env = std::getenv(name);
  if (env == nullptr || env[0] == '\0') {
    return default_value;
  }
  char* end = nullptr;
  const long value = std::strtol(env, &end, 10);
  if (end == env || *end != '\0') {
    LOG(WARNING) << "Invalid " << name << "=" << env << ", fallback to "
                 << default_value;
    return default_value;
  }
  return static_cast<int>(value);
}

int sanitize_chunk_kv(int chunk_kv, int default_value) {
  if (chunk_kv < 64 || chunk_kv > 512 || (chunk_kv % 64) != 0) {
    LOG(WARNING) << "Unsupported chunk_kv=" << chunk_kv
                 << ", expect one of {64, 128, 256, 512}, fallback to "
                 << default_value;
    return default_value;
  }
  return chunk_kv;
}

int resolve_chunk_kv(FullAttentionKernelMode mode) {
  const int generic_override =
      parse_env_int_or_default("XLLM_MTGR_CUDA_FULL_ATTN_CHUNK_KV", -1);
  if (generic_override > 0) {
    return sanitize_chunk_kv(generic_override, /*default_value=*/64);
  }
  switch (mode) {
    case FullAttentionKernelMode::kScalarChunked:
      return sanitize_chunk_kv(
          parse_env_int_or_default("XLLM_MTGR_CUDA_FULL_ATTN_SCALAR_CHUNK_KV",
                                   64),
          /*default_value=*/64);
    case FullAttentionKernelMode::kWmmaChunked:
      return sanitize_chunk_kv(
          parse_env_int_or_default("XLLM_MTGR_CUDA_FULL_ATTN_WMMA_CHUNK_KV",
                                   64),
          /*default_value=*/64);
    case FullAttentionKernelMode::kFusedSplitKv:
    case FullAttentionKernelMode::kWmmaFused:
    case FullAttentionKernelMode::kAuto:
      return 0;
  }
  return 0;
}

int64_t make_workspace_key(const c10::Device& device,
                           int64_t num_chunks,
                           int64_t q_len,
                           int64_t num_heads) {
  return (static_cast<int64_t>(device.index()) << 48) ^ (num_chunks << 24) ^
         (q_len << 12) ^ num_heads;
}

SmallQAttentionWorkspace& get_small_q_attention_workspace(
    const c10::Device& device,
    int64_t num_chunks,
    int64_t q_len,
    int64_t num_heads) {
  static thread_local std::unordered_map<int64_t, SmallQAttentionWorkspace>
      workspace_map;
  const int64_t key = make_workspace_key(device, num_chunks, q_len, num_heads);
  auto& ws = workspace_map[key];
  const auto opts = torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(device)
                        .requires_grad(false);
  if (!ws.partial_out.defined() || ws.partial_out.size(0) != num_chunks ||
      ws.partial_out.size(1) != q_len || ws.partial_out.size(2) != num_heads ||
      ws.partial_out.size(3) != kHeadDim) {
    ws.partial_out =
        torch::empty({num_chunks, q_len, num_heads, kHeadDim}, opts);
    ws.partial_max = torch::empty({num_chunks, q_len, num_heads}, opts);
    ws.partial_sum = torch::empty({num_chunks, q_len, num_heads}, opts);
  }
  return ws;
}

SmallQAttentionHalfWorkspace& get_small_q_attention_half_workspace(
    const c10::Device& device,
    int64_t num_chunks,
    int64_t q_len,
    int64_t num_heads) {
  static thread_local std::unordered_map<int64_t, SmallQAttentionHalfWorkspace>
      workspace_map;
  const int64_t key = make_workspace_key(device, num_chunks, q_len, num_heads);
  auto& ws = workspace_map[key];
  const auto out_opts = torch::TensorOptions()
                            .dtype(torch::kFloat16)
                            .device(device)
                            .requires_grad(false);
  const auto stat_opts = torch::TensorOptions()
                             .dtype(torch::kFloat32)
                             .device(device)
                             .requires_grad(false);
  if (!ws.partial_out.defined() || ws.partial_out.size(0) != num_chunks ||
      ws.partial_out.size(1) != q_len || ws.partial_out.size(2) != num_heads ||
      ws.partial_out.size(3) != kHeadDim) {
    ws.partial_out =
        torch::empty({num_chunks, q_len, num_heads, kHeadDim}, out_opts);
    ws.partial_max = torch::empty({num_chunks, q_len, num_heads}, stat_opts);
    ws.partial_sum = torch::empty({num_chunks, q_len, num_heads}, stat_opts);
  }
  return ws;
}

__device__ __forceinline__ float warp_reduce_max(float value) {
  for (int mask = 16; mask > 0; mask >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, mask));
  }
  return value;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int mask = 16; mask > 0; mask >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, mask);
  }
  return value;
}

__device__ __forceinline__ float dot_128_half(const half* __restrict__ a,
                                              const half* __restrict__ b) {
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kHeadDim; i += 2) {
    const half2 a2 = *reinterpret_cast<const half2*>(a + i);
    const half2 b2 = *reinterpret_cast<const half2*>(b + i);
    const float2 af = __half22float2(a2);
    const float2 bf = __half22float2(b2);
    sum += af.x * bf.x + af.y * bf.y;
  }
  return sum;
}

template <typename T>
__device__ __forceinline__ float load_partial_as_float(const T* ptr);

template <>
__device__ __forceinline__ float load_partial_as_float<float>(
    const float* ptr) {
  return *ptr;
}

template <>
__device__ __forceinline__ float load_partial_as_float<half>(const half* ptr) {
  return __half2float(*ptr);
}

template <int QueriesPerBlock>
__device__ __forceinline__ void load_query_block_shared(
    half* __restrict__ q_shared,
    const half* __restrict__ query,
    int q_len,
    int num_heads,
    int head_idx,
    int q_block_start,
    int thread_idx,
    int threads_per_block) {
  for (int flat = thread_idx; flat < QueriesPerBlock * kHeadDim;
       flat += threads_per_block) {
    const int q_local = flat / kHeadDim;
    const int dim = flat % kHeadDim;
    const int q_idx = q_block_start + q_local;
    q_shared[flat] =
        q_idx < q_len
            ? query[(static_cast<int64_t>(q_idx) * num_heads + head_idx) *
                        kHeadDim +
                    dim]
            : __float2half(0.0f);
  }
}

template <int TileKv>
__device__ __forceinline__ void load_kv_tile_shared(
    half* __restrict__ k_shared,
    half* __restrict__ v_shared,
    const half* __restrict__ key,
    const half* __restrict__ value,
    int valid_rows,
    int tile_begin,
    int num_heads,
    int head_idx,
    int thread_idx,
    int threads_per_block) {
  for (int flat = thread_idx; flat < TileKv * kHeadDim;
       flat += threads_per_block) {
    const int row = flat / kHeadDim;
    const int dim = flat % kHeadDim;
    if (row < valid_rows) {
      const int kv_idx = tile_begin + row;
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
}

template <int QueriesPerBlock, int TileKv>
__global__ void full_attention_scalar_chunked_kernel(
    const half* __restrict__ query,
    const half* __restrict__ key,
    const half* __restrict__ value,
    float* __restrict__ partial_out,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    int q_len,
    int num_heads,
    int kv_len,
    int chunk_kv,
    float sm_scale_log2e) {
  // Chunked baseline:
  // one CTA handles one (head, q-block, kv-chunk), writes chunk-local
  // numerator/max/sum, then a second kernel merges all chunk partials.
  __shared__ half q_shared[QueriesPerBlock * kHeadDim];
  __shared__ half k_shared[TileKv * kHeadDim];
  __shared__ half v_shared[TileKv * kHeadDim];

  const int head_idx = blockIdx.x;
  const int q_block_idx = blockIdx.y;
  const int chunk_idx = blockIdx.z;
  const int q_block_start = q_block_idx * QueriesPerBlock;
  const int warp_id = threadIdx.x >> 5;
  const int lane_id = threadIdx.x & 31;
  const int q_idx = q_block_start + warp_id;
  const int chunk_begin = chunk_idx * chunk_kv;
  const int chunk_end = min(chunk_begin + chunk_kv, kv_len);

  load_query_block_shared<QueriesPerBlock>(q_shared,
                                           query,
                                           q_len,
                                           num_heads,
                                           head_idx,
                                           q_block_start,
                                           threadIdx.x,
                                           blockDim.x);
  __syncthreads();

  float row_max = kNegInf;
  float row_sum = 0.0f;
  float row_accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int tile_begin = chunk_begin; tile_begin < chunk_end;
       tile_begin += TileKv) {
    const int valid_rows = min(TileKv, chunk_end - tile_begin);
    load_kv_tile_shared<TileKv>(k_shared,
                                v_shared,
                                key,
                                value,
                                valid_rows,
                                tile_begin,
                                num_heads,
                                head_idx,
                                threadIdx.x,
                                blockDim.x);
    __syncthreads();

    if (q_idx < q_len) {
      const half* q_row = q_shared + warp_id * kHeadDim;
      float scaled_score = kNegInf;
      if (lane_id < valid_rows) {
        const half* k_row = k_shared + lane_id * kHeadDim;
        scaled_score = dot_128_half(q_row, k_row) * sm_scale_log2e;
      }
      const float tile_max = warp_reduce_max(scaled_score);
      float lane_prob =
          lane_id < valid_rows ? exp2f(scaled_score - tile_max) : 0.0f;
      const float tile_sum = warp_reduce_sum(lane_prob);

      float tile_accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      const int dim_base = lane_id * 4;
#pragma unroll
      for (int kk = 0; kk < TileKv; ++kk) {
        if (kk >= valid_rows) {
          break;
        }
        const float prob = __shfl_sync(0xffffffff, lane_prob, kk);
        const half* v_row = v_shared + kk * kHeadDim;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
          tile_accum[d] += prob * __half2float(v_row[dim_base + d]);
        }
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

  if (q_idx < q_len) {
    const int64_t stats_index =
        (static_cast<int64_t>(chunk_idx) * q_len + q_idx) * num_heads +
        head_idx;
    partial_max[stats_index] = row_max;
    partial_sum[stats_index] = row_sum;
    const int64_t out_index = stats_index * kHeadDim + lane_id * 4;
#pragma unroll
    for (int d = 0; d < 4; ++d) {
      partial_out[out_index + d] = row_accum[d];
    }
  }
}

template <int QueriesPerBlock, typename PartialT>
__global__ void full_attention_small_q_combine_kernel(
    const PartialT* __restrict__ partial_out,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    half* __restrict__ output,
    int q_len,
    int num_heads,
    int num_chunks) {
  const int head_idx = blockIdx.x;
  const int q_block_idx = blockIdx.y;
  const int warp_id = threadIdx.x >> 5;
  const int lane_id = threadIdx.x & 31;
  const int q_idx = q_block_idx * QueriesPerBlock + warp_id;
  if (q_idx >= q_len) {
    return;
  }

  float global_max = kNegInf;
  for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    const int64_t stats_index =
        (static_cast<int64_t>(chunk_idx) * q_len + q_idx) * num_heads +
        head_idx;
    global_max = fmaxf(global_max, partial_max[stats_index]);
  }

  float denom = 0.0f;
  float numer[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    const int64_t stats_index =
        (static_cast<int64_t>(chunk_idx) * q_len + q_idx) * num_heads +
        head_idx;
    const float local_max = partial_max[stats_index];
    const float local_sum = partial_sum[stats_index];
    const float scale = exp2f(local_max - global_max);
    denom += local_sum * scale;
    const int64_t out_index = stats_index * kHeadDim + lane_id * 4;
#pragma unroll
    for (int d = 0; d < 4; ++d) {
      numer[d] += load_partial_as_float(partial_out + out_index + d) * scale;
    }
  }

  const float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;
  const int64_t output_index =
      (static_cast<int64_t>(q_idx) * num_heads + head_idx) * kHeadDim +
      lane_id * 4;
#pragma unroll
  for (int d = 0; d < 4; ++d) {
    output[output_index + d] = __float2half_rn(numer[d] * inv_denom);
  }
}

template <int QueriesPerBlock, int TileKv>
__global__ void full_attention_fused_splitkv_kernel(
    const half* __restrict__ query,
    const half* __restrict__ key,
    const half* __restrict__ value,
    half* __restrict__ output,
    int q_len,
    int num_heads,
    int kv_len,
    float sm_scale_log2e) {
  // Fused split-KV attempt:
  // one CTA handles one (head, q-block) and walks the full KV range directly,
  // so partial_out / partial_stats / merge disappear completely.
  __shared__ half q_shared[QueriesPerBlock * kHeadDim];
  __shared__ half k_shared[TileKv * kHeadDim];
  __shared__ half v_shared[TileKv * kHeadDim];

  const int head_idx = blockIdx.x;
  const int q_block_idx = blockIdx.y;
  const int q_block_start = q_block_idx * QueriesPerBlock;
  const int warp_id = threadIdx.x >> 5;
  const int lane_id = threadIdx.x & 31;
  const int q_idx = q_block_start + warp_id;

  load_query_block_shared<QueriesPerBlock>(q_shared,
                                           query,
                                           q_len,
                                           num_heads,
                                           head_idx,
                                           q_block_start,
                                           threadIdx.x,
                                           blockDim.x);
  __syncthreads();

  float row_max = kNegInf;
  float row_sum = 0.0f;
  float row_accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int tile_begin = 0; tile_begin < kv_len; tile_begin += TileKv) {
    const int valid_rows = min(TileKv, kv_len - tile_begin);
    load_kv_tile_shared<TileKv>(k_shared,
                                v_shared,
                                key,
                                value,
                                valid_rows,
                                tile_begin,
                                num_heads,
                                head_idx,
                                threadIdx.x,
                                blockDim.x);
    __syncthreads();

    if (q_idx < q_len) {
      const half* q_row = q_shared + warp_id * kHeadDim;
      float scaled_score = kNegInf;
      if (lane_id < valid_rows) {
        const half* k_row = k_shared + lane_id * kHeadDim;
        scaled_score = dot_128_half(q_row, k_row) * sm_scale_log2e;
      }
      const float tile_max = warp_reduce_max(scaled_score);
      float lane_prob =
          lane_id < valid_rows ? exp2f(scaled_score - tile_max) : 0.0f;
      const float tile_sum = warp_reduce_sum(lane_prob);

      float tile_accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      const int dim_base = lane_id * 4;
#pragma unroll
      for (int kk = 0; kk < TileKv; ++kk) {
        if (kk >= valid_rows) {
          break;
        }
        const float prob = __shfl_sync(0xffffffff, lane_prob, kk);
        const half* v_row = v_shared + kk * kHeadDim;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
          tile_accum[d] += prob * __half2float(v_row[dim_base + d]);
        }
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

  if (q_idx < q_len) {
    const float inv_denom = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;
    const int64_t output_index =
        (static_cast<int64_t>(q_idx) * num_heads + head_idx) * kHeadDim +
        lane_id * 4;
#pragma unroll
    for (int d = 0; d < 4; ++d) {
      output[output_index + d] = __float2half_rn(row_accum[d] * inv_denom);
    }
  }
}

template <int QueriesPerBlock>
__global__ void full_attention_wmma_fused_kernel(const half* __restrict__ query,
                                                 const half* __restrict__ key,
                                                 const half* __restrict__ value,
                                                 half* __restrict__ output,
                                                 int q_len,
                                                 int num_heads,
                                                 int kv_len,
                                                 float sm_scale_log2e) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  __shared__ half q_shared[16 * kHeadDim];
  __shared__ half k_shared[16 * kHeadDim];
  __shared__ half v_shared[16 * kHeadDim];
  __shared__ float score_shared[16 * 16];
  __shared__ half prob_shared[16 * 16];
  __shared__ float pv_shared[16 * kHeadDim];

  const int head_idx = blockIdx.x;
  const int q_block_idx = blockIdx.y;
  const int q_block_start = q_block_idx * QueriesPerBlock;
  const int warp_id = threadIdx.x >> 5;
  const int lane_id = threadIdx.x & 31;
  const int q_idx = q_block_start + warp_id;

  for (int flat = threadIdx.x; flat < 16 * kHeadDim; flat += blockDim.x) {
    const int row = flat / kHeadDim;
    const int dim = flat % kHeadDim;
    const int global_q = q_block_start + row;
    q_shared[flat] =
        row < QueriesPerBlock && global_q < q_len
            ? query[(static_cast<int64_t>(global_q) * num_heads + head_idx) *
                        kHeadDim +
                    dim]
            : __float2half(0.0f);
  }
  for (int flat = threadIdx.x; flat < 16 * 16; flat += blockDim.x) {
    prob_shared[flat] = __float2half(0.0f);
  }
  __syncthreads();

  float row_max = kNegInf;
  float row_sum = 0.0f;
  float row_accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int tile_begin = 0; tile_begin < kv_len; tile_begin += 16) {
    const int valid_rows = min(16, kv_len - tile_begin);

    for (int flat = threadIdx.x; flat < 16 * kHeadDim; flat += blockDim.x) {
      const int row = flat / kHeadDim;
      const int dim = flat % kHeadDim;
      if (row < valid_rows) {
        const int kv_idx = tile_begin + row;
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
    __syncthreads();

    if (warp_id == 0) {
      using namespace nvcuda;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
      wmma::fill_fragment(c_frag, 0.0f);
#pragma unroll
      for (int kk = 0; kk < kHeadDim; kk += 16) {
        wmma::load_matrix_sync(a_frag, q_shared + kk, kHeadDim);
        wmma::load_matrix_sync(b_frag, k_shared + kk, kHeadDim);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }
      wmma::store_matrix_sync(score_shared, c_frag, 16, wmma::mem_row_major);
    }
    __syncthreads();

    float tile_max = kNegInf;
    float tile_sum = 0.0f;
    if (warp_id < QueriesPerBlock && q_idx < q_len) {
      const float scaled_score =
          lane_id < 16 && lane_id < valid_rows
              ? score_shared[warp_id * 16 + lane_id] * sm_scale_log2e
              : kNegInf;
      tile_max = warp_reduce_max(scaled_score);
      const float lane_prob = lane_id < 16 && lane_id < valid_rows
                                  ? exp2f(scaled_score - tile_max)
                                  : 0.0f;
      tile_sum = warp_reduce_sum(lane_prob);
      if (lane_id < 16) {
        prob_shared[warp_id * 16 + lane_id] =
            __float2half_rn(lane_id < valid_rows ? lane_prob : 0.0f);
      }
    }
    __syncthreads();

    if (warp_id < 8) {
      using namespace nvcuda;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
      wmma::fill_fragment(c_frag, 0.0f);
      wmma::load_matrix_sync(a_frag, prob_shared, 16);
      wmma::load_matrix_sync(b_frag, v_shared + warp_id * 16, kHeadDim);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      wmma::store_matrix_sync(
          pv_shared + warp_id * 16, c_frag, kHeadDim, wmma::mem_row_major);
    }
    __syncthreads();

    if (warp_id < QueriesPerBlock && q_idx < q_len) {
      const int dim_base = lane_id * 4;
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

  if (warp_id < QueriesPerBlock && q_idx < q_len) {
    const float inv_denom = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;
    const int64_t output_index =
        (static_cast<int64_t>(q_idx) * num_heads + head_idx) * kHeadDim +
        lane_id * 4;
#pragma unroll
    for (int d = 0; d < 4; ++d) {
      output[output_index + d] = __float2half_rn(row_accum[d] * inv_denom);
    }
  }
#else
  (void)query;
  (void)key;
  (void)value;
  (void)output;
  (void)q_len;
  (void)num_heads;
  (void)kv_len;
  (void)sm_scale_log2e;
#endif
}

template <int QueriesPerBlock>
__global__ void full_attention_wmma_chunked_kernel(
    const half* __restrict__ query,
    const half* __restrict__ key,
    const half* __restrict__ value,
    half* __restrict__ partial_out,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    int q_len,
    int num_heads,
    int kv_len,
    int chunk_kv,
    float sm_scale_log2e) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  // Tensor-core path:
  // per chunk, use WMMA for Q*K^T and P*V, keep chunk-level online-softmax
  // stats in registers, then reuse the generic chunk-merge kernel.
  __shared__ half q_shared[16 * kHeadDim];
  __shared__ half k_shared[16 * kHeadDim];
  __shared__ half v_shared[16 * kHeadDim];
  __shared__ float score_shared[16 * 16];
  __shared__ half prob_shared[16 * 16];
  __shared__ float pv_shared[16 * kHeadDim];

  const int head_idx = blockIdx.x;
  const int q_block_idx = blockIdx.y;
  const int chunk_idx = blockIdx.z;
  const int q_block_start = q_block_idx * QueriesPerBlock;
  const int warp_id = threadIdx.x >> 5;
  const int lane_id = threadIdx.x & 31;
  const int q_idx = q_block_start + warp_id;
  const int chunk_begin = chunk_idx * chunk_kv;
  const int chunk_end = min(chunk_begin + chunk_kv, kv_len);

  for (int flat = threadIdx.x; flat < 16 * kHeadDim; flat += blockDim.x) {
    const int row = flat / kHeadDim;
    const int dim = flat % kHeadDim;
    const int global_q = q_block_start + row;
    q_shared[flat] =
        row < QueriesPerBlock && global_q < q_len
            ? query[(static_cast<int64_t>(global_q) * num_heads + head_idx) *
                        kHeadDim +
                    dim]
            : __float2half(0.0f);
  }
  for (int flat = threadIdx.x; flat < 16 * 16; flat += blockDim.x) {
    prob_shared[flat] = __float2half(0.0f);
  }
  __syncthreads();

  float row_max = kNegInf;
  float row_sum = 0.0f;
  float row_accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int tile_begin = chunk_begin; tile_begin < chunk_end; tile_begin += 16) {
    const int valid_rows = min(16, chunk_end - tile_begin);

    for (int flat = threadIdx.x; flat < 16 * kHeadDim; flat += blockDim.x) {
      const int row = flat / kHeadDim;
      const int dim = flat % kHeadDim;
      if (row < valid_rows) {
        const int kv_idx = tile_begin + row;
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
    __syncthreads();

    if (warp_id == 0) {
      using namespace nvcuda;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
      wmma::fill_fragment(c_frag, 0.0f);
#pragma unroll
      for (int kk = 0; kk < kHeadDim; kk += 16) {
        wmma::load_matrix_sync(a_frag, q_shared + kk, kHeadDim);
        wmma::load_matrix_sync(b_frag, k_shared + kk, kHeadDim);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }
      wmma::store_matrix_sync(score_shared, c_frag, 16, wmma::mem_row_major);
    }
    __syncthreads();

    float tile_max = kNegInf;
    float tile_sum = 0.0f;
    if (warp_id < QueriesPerBlock && q_idx < q_len) {
      const float scaled_score =
          lane_id < 16 && lane_id < valid_rows
              ? score_shared[warp_id * 16 + lane_id] * sm_scale_log2e
              : kNegInf;
      tile_max = warp_reduce_max(scaled_score);
      const float lane_prob = lane_id < 16 && lane_id < valid_rows
                                  ? exp2f(scaled_score - tile_max)
                                  : 0.0f;
      tile_sum = warp_reduce_sum(lane_prob);
      if (lane_id < 16) {
        prob_shared[warp_id * 16 + lane_id] =
            __float2half_rn(lane_id < valid_rows ? lane_prob : 0.0f);
      }
    }
    __syncthreads();

    if (warp_id < 8) {
      using namespace nvcuda;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
      wmma::fill_fragment(c_frag, 0.0f);
      wmma::load_matrix_sync(a_frag, prob_shared, 16);
      wmma::load_matrix_sync(b_frag, v_shared + warp_id * 16, kHeadDim);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      wmma::store_matrix_sync(
          pv_shared + warp_id * 16, c_frag, kHeadDim, wmma::mem_row_major);
    }
    __syncthreads();

    if (warp_id < QueriesPerBlock && q_idx < q_len) {
      const int dim_base = lane_id * 4;
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

  if (warp_id < QueriesPerBlock && q_idx < q_len) {
    const int64_t stats_index =
        (static_cast<int64_t>(chunk_idx) * q_len + q_idx) * num_heads +
        head_idx;
    partial_max[stats_index] = row_max;
    partial_sum[stats_index] = row_sum;
    const int64_t out_index = stats_index * kHeadDim + lane_id * 4;
#pragma unroll
    for (int d = 0; d < 4; ++d) {
      partial_out[out_index + d] = __float2half_rn(row_accum[d]);
    }
  }
#else
  (void)query;
  (void)key;
  (void)value;
  (void)partial_out;
  (void)partial_max;
  (void)partial_sum;
  (void)q_len;
  (void)num_heads;
  (void)kv_len;
  (void)sm_scale_log2e;
#endif
}

void launch_scalar_chunked_attention(const torch::Tensor& query_snd,
                                     const torch::Tensor& key_snd,
                                     const torch::Tensor& value_snd,
                                     int chunk_kv,
                                     float sm_scale_log2e,
                                     torch::Tensor output_snd) {
  const int q_len = static_cast<int>(query_snd.size(0));
  const int num_heads = static_cast<int>(query_snd.size(1));
  const int kv_len = static_cast<int>(key_snd.size(0));
  const int num_chunks = static_cast<int>(ceil_div_int64(kv_len, chunk_kv));
  auto& ws = get_small_q_attention_workspace(
      query_snd.device(), num_chunks, q_len, num_heads);
  const int q_blocks =
      static_cast<int>(ceil_div_int64(q_len, kChunkedQueriesPerBlock));

  const half* query_ptr =
      reinterpret_cast<const half*>(query_snd.data_ptr<at::Half>());
  const half* key_ptr =
      reinterpret_cast<const half*>(key_snd.data_ptr<at::Half>());
  const half* value_ptr =
      reinterpret_cast<const half*>(value_snd.data_ptr<at::Half>());
  half* output_ptr = reinterpret_cast<half*>(output_snd.data_ptr<at::Half>());
  float* partial_out_ptr = ws.partial_out.data_ptr<float>();
  float* partial_max_ptr = ws.partial_max.data_ptr<float>();
  float* partial_sum_ptr = ws.partial_sum.data_ptr<float>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const dim3 partial_grid(num_heads, q_blocks, num_chunks);
  full_attention_scalar_chunked_kernel<kChunkedQueriesPerBlock, kTileKv>
      <<<partial_grid, kChunkedThreadsPerBlock, 0, stream>>>(query_ptr,
                                                             key_ptr,
                                                             value_ptr,
                                                             partial_out_ptr,
                                                             partial_max_ptr,
                                                             partial_sum_ptr,
                                                             q_len,
                                                             num_heads,
                                                             kv_len,
                                                             chunk_kv,
                                                             sm_scale_log2e);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const dim3 combine_grid(num_heads, q_blocks);
  full_attention_small_q_combine_kernel<kChunkedQueriesPerBlock, float>
      <<<combine_grid, kChunkedThreadsPerBlock, 0, stream>>>(partial_out_ptr,
                                                             partial_max_ptr,
                                                             partial_sum_ptr,
                                                             output_ptr,
                                                             q_len,
                                                             num_heads,
                                                             num_chunks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_fused_splitkv_attention(const torch::Tensor& query_snd,
                                    const torch::Tensor& key_snd,
                                    const torch::Tensor& value_snd,
                                    float sm_scale_log2e,
                                    torch::Tensor output_snd) {
  const int q_len = static_cast<int>(query_snd.size(0));
  const int num_heads = static_cast<int>(query_snd.size(1));
  const int kv_len = static_cast<int>(key_snd.size(0));
  const int q_blocks =
      static_cast<int>(ceil_div_int64(q_len, kFusedQueriesPerBlock));
  const half* query_ptr =
      reinterpret_cast<const half*>(query_snd.data_ptr<at::Half>());
  const half* key_ptr =
      reinterpret_cast<const half*>(key_snd.data_ptr<at::Half>());
  const half* value_ptr =
      reinterpret_cast<const half*>(value_snd.data_ptr<at::Half>());
  half* output_ptr = reinterpret_cast<half*>(output_snd.data_ptr<at::Half>());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const dim3 grid(num_heads, q_blocks);
  full_attention_fused_splitkv_kernel<kFusedQueriesPerBlock, kTileKv>
      <<<grid, kFusedThreadsPerBlock, 0, stream>>>(query_ptr,
                                                   key_ptr,
                                                   value_ptr,
                                                   output_ptr,
                                                   q_len,
                                                   num_heads,
                                                   kv_len,
                                                   sm_scale_log2e);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_wmma_chunked_attention(const torch::Tensor& query_snd,
                                   const torch::Tensor& key_snd,
                                   const torch::Tensor& value_snd,
                                   int chunk_kv,
                                   float sm_scale_log2e,
                                   torch::Tensor output_snd) {
  const int q_len = static_cast<int>(query_snd.size(0));
  const int num_heads = static_cast<int>(query_snd.size(1));
  const int kv_len = static_cast<int>(key_snd.size(0));
  const int num_chunks = static_cast<int>(ceil_div_int64(kv_len, chunk_kv));
  auto& ws = get_small_q_attention_half_workspace(
      query_snd.device(), num_chunks, q_len, num_heads);
  const int q_blocks =
      static_cast<int>(ceil_div_int64(q_len, kChunkedQueriesPerBlock));

  const half* query_ptr =
      reinterpret_cast<const half*>(query_snd.data_ptr<at::Half>());
  const half* key_ptr =
      reinterpret_cast<const half*>(key_snd.data_ptr<at::Half>());
  const half* value_ptr =
      reinterpret_cast<const half*>(value_snd.data_ptr<at::Half>());
  half* output_ptr = reinterpret_cast<half*>(output_snd.data_ptr<at::Half>());
  half* partial_out_ptr =
      reinterpret_cast<half*>(ws.partial_out.data_ptr<at::Half>());
  float* partial_max_ptr = ws.partial_max.data_ptr<float>();
  float* partial_sum_ptr = ws.partial_sum.data_ptr<float>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const dim3 partial_grid(num_heads, q_blocks, num_chunks);
  full_attention_wmma_chunked_kernel<kChunkedQueriesPerBlock>
      <<<partial_grid, kWmmaThreadsPerBlock, 0, stream>>>(query_ptr,
                                                          key_ptr,
                                                          value_ptr,
                                                          partial_out_ptr,
                                                          partial_max_ptr,
                                                          partial_sum_ptr,
                                                          q_len,
                                                          num_heads,
                                                          kv_len,
                                                          chunk_kv,
                                                          sm_scale_log2e);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const dim3 combine_grid(num_heads, q_blocks);
  full_attention_small_q_combine_kernel<kChunkedQueriesPerBlock, half>
      <<<combine_grid, kChunkedThreadsPerBlock, 0, stream>>>(partial_out_ptr,
                                                             partial_max_ptr,
                                                             partial_sum_ptr,
                                                             output_ptr,
                                                             q_len,
                                                             num_heads,
                                                             num_chunks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_wmma_fused_attention(const torch::Tensor& query_snd,
                                 const torch::Tensor& key_snd,
                                 const torch::Tensor& value_snd,
                                 float sm_scale_log2e,
                                 torch::Tensor output_snd) {
  const int q_len = static_cast<int>(query_snd.size(0));
  const int num_heads = static_cast<int>(query_snd.size(1));
  const int kv_len = static_cast<int>(key_snd.size(0));
  const int q_blocks =
      static_cast<int>(ceil_div_int64(q_len, kChunkedQueriesPerBlock));

  const half* query_ptr =
      reinterpret_cast<const half*>(query_snd.data_ptr<at::Half>());
  const half* key_ptr =
      reinterpret_cast<const half*>(key_snd.data_ptr<at::Half>());
  const half* value_ptr =
      reinterpret_cast<const half*>(value_snd.data_ptr<at::Half>());
  half* output_ptr = reinterpret_cast<half*>(output_snd.data_ptr<at::Half>());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const dim3 grid(num_heads, q_blocks);
  full_attention_wmma_fused_kernel<kChunkedQueriesPerBlock>
      <<<grid, kWmmaThreadsPerBlock, 0, stream>>>(query_ptr,
                                                  key_ptr,
                                                  value_ptr,
                                                  output_ptr,
                                                  q_len,
                                                  num_heads,
                                                  kv_len,
                                                  sm_scale_log2e);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

FullAttentionKernelMode select_auto_mode(int64_t kv_len) {
  (void)kv_len;
  // On current H800 measurements, WMMA chunked is the fastest stable mode for
  // the supported MTGR small-Q / head_dim=128 full-attention shape.
  return FullAttentionKernelMode::kWmmaChunked;
}

}  // namespace

void full_attention_cuda(const torch::Tensor& query_snd,
                         const torch::Tensor& key_snd,
                         const torch::Tensor& value_snd,
                         double sm_scale,
                         torch::Tensor output_snd) {
  CHECK(query_snd.defined());
  CHECK(key_snd.defined());
  CHECK(value_snd.defined());
  CHECK(output_snd.defined());
  CHECK(query_snd.is_cuda());
  CHECK(key_snd.is_cuda());
  CHECK(value_snd.is_cuda());
  CHECK(output_snd.is_cuda());
  CHECK(query_snd.is_contiguous());
  CHECK(key_snd.is_contiguous());
  CHECK(value_snd.is_contiguous());
  CHECK(output_snd.is_contiguous());
  CHECK_EQ(query_snd.scalar_type(), torch::kFloat16);
  CHECK_EQ(key_snd.scalar_type(), torch::kFloat16);
  CHECK_EQ(value_snd.scalar_type(), torch::kFloat16);
  CHECK_EQ(output_snd.scalar_type(), torch::kFloat16);
  CHECK_EQ(query_snd.dim(), 3);
  CHECK_EQ(key_snd.dim(), 3);
  CHECK_EQ(value_snd.dim(), 3);
  CHECK_EQ(output_snd.dim(), 3);
  CHECK_EQ(key_snd.sizes(), value_snd.sizes());
  CHECK_EQ(query_snd.sizes(), output_snd.sizes());
  CHECK_EQ(query_snd.size(2), key_snd.size(2));
  CHECK_EQ(query_snd.size(2), value_snd.size(2));

  const int64_t q_len = query_snd.size(0);
  const int64_t num_heads = query_snd.size(1);
  const int64_t num_kv_heads = key_snd.size(1);
  const int64_t head_dim = query_snd.size(2);
  const int64_t kv_len = key_snd.size(0);

  CHECK_EQ(num_heads, num_kv_heads)
      << "full_attention_cuda expects num_heads == num_kv_heads";
  CHECK_EQ(head_dim, kHeadDim)
      << "full_attention_cuda is specialized for head_dim == 128";
  CHECK_GT(q_len, 0);
  CHECK_LE(q_len, 8) << "full_attention_cuda is specialized for q_len <= 8";

  c10::cuda::CUDAGuard guard(query_snd.device());
  const float sm_scale_log2e = static_cast<float>(sm_scale) * kLog2E;
  FullAttentionKernelMode mode = parse_full_attention_mode();
  if (mode == FullAttentionKernelMode::kAuto) {
    mode = select_auto_mode(kv_len);
  }

  switch (mode) {
    case FullAttentionKernelMode::kScalarChunked: {
      const int chunk_kv = resolve_chunk_kv(mode);
      const int64_t num_chunks = ceil_div_int64(kv_len, chunk_kv);
      CHECK_LE(num_chunks, kMaxChunks) << "num_chunks <= kMaxChunks";
      launch_scalar_chunked_attention(
          query_snd, key_snd, value_snd, chunk_kv, sm_scale_log2e, output_snd);
    }
      return;
    case FullAttentionKernelMode::kFusedSplitKv:
      launch_fused_splitkv_attention(
          query_snd, key_snd, value_snd, sm_scale_log2e, output_snd);
      return;
    case FullAttentionKernelMode::kWmmaChunked: {
      const int chunk_kv = resolve_chunk_kv(mode);
      const int64_t num_chunks = ceil_div_int64(kv_len, chunk_kv);
      CHECK_LE(num_chunks, kMaxChunks) << "num_chunks <= kMaxChunks";
      launch_wmma_chunked_attention(
          query_snd, key_snd, value_snd, chunk_kv, sm_scale_log2e, output_snd);
    }
      return;
    case FullAttentionKernelMode::kWmmaFused:
      launch_wmma_fused_attention(
          query_snd, key_snd, value_snd, sm_scale_log2e, output_snd);
      return;
    case FullAttentionKernelMode::kAuto:
      break;
  }
  CHECK(false) << "unreachable full attention mode";
}

}  // namespace xllm::kernel::cuda::test
