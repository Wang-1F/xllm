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

#include "mtgr_qk_norm_test.h"

#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/cuda.h>

#include <glog/logging.h>

#include <memory>
#include <tuple>

namespace {

constexpr int kHeadDim = 128;
constexpr int kWarpSize = 32;
constexpr int kElemsPerLane = kHeadDim / kWarpSize;
constexpr uint32_t kFinalMask = 0xffffffffu;

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_xor_sync(kFinalMask, value, offset);
  }
  return value;
}

__global__ void mtgr_qk_norm_strided_bf16_hd128_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ weight,
    int64_t token_stride,
    int64_t head_stride,
    int64_t num_heads,
    int64_t num_rows,
    float eps) {
  const int warps_per_block = blockDim.x / kWarpSize;
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * warps_per_block +
                      static_cast<int64_t>(warp_id);
  if (row >= num_rows) {
    return;
  }

  const int64_t token_idx = row / num_heads;
  const int64_t head_idx = row % num_heads;
  const int64_t input_base = token_idx * token_stride + head_idx * head_stride;
  const int64_t output_base = row * kHeadDim;

  float values[kElemsPerLane];
  float sum_squares = 0.0f;
#pragma unroll
  for (int i = 0; i < kElemsPerLane; ++i) {
    const int dim = lane_id * kElemsPerLane + i;
    const float x = __bfloat162float(input[input_base + dim]);
    values[i] = x;
    sum_squares += x * x;
  }

  sum_squares = warp_reduce_sum(sum_squares);
  const float inv_rms = rsqrtf(sum_squares / static_cast<float>(kHeadDim) + eps);

#pragma unroll
  for (int i = 0; i < kElemsPerLane; ++i) {
    const int dim = lane_id * kElemsPerLane + i;
    const float scale = 1.0f + __bfloat162float(weight[dim]);
    output[output_base + dim] =
        __float2bfloat16_rn(values[i] * inv_rms * scale);
  }
}

void launch_mtgr_qk_norm_strided_bf16_hd128(torch::Tensor output,
                                            const torch::Tensor& input,
                                            const torch::Tensor& weight,
                                            double eps) {
  const int64_t rows = input.size(0) * input.size(1);
  constexpr int kWarpsPerBlock = 8;
  dim3 block(kWarpsPerBlock * kWarpSize);
  dim3 grid((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
  const at::cuda::OptionalCUDAGuard device_guard(input.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  mtgr_qk_norm_strided_bf16_hd128_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<c10::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<c10::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<c10::BFloat16>()),
      input.stride(0),
      input.stride(1),
      input.size(1),
      rows,
      static_cast<float>(eps));
  CHECK_EQ(cudaGetLastError(), cudaSuccess);
}

class MTGRQKNormStridedBf16Hd128Impl final
    : public xllm::kernel::cuda::test::MTGRQKNormTestImpl {
 public:
  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& q,
      const torch::Tensor& k,
      const torch::Tensor& q_weight,
      const torch::Tensor& k_weight,
      double eps) override {
    CHECK(q.is_cuda());
    CHECK(k.is_cuda());
    CHECK_EQ(q.scalar_type(), torch::kBFloat16);
    CHECK_EQ(k.scalar_type(), torch::kBFloat16);
    CHECK_EQ(q.dim(), 3);
    CHECK_EQ(k.dim(), 3);
    CHECK_EQ(q.size(-1), kHeadDim);
    CHECK_EQ(k.size(-1), kHeadDim);
    CHECK_EQ(q.stride(-1), 1);
    CHECK_EQ(k.stride(-1), 1);
    CHECK_EQ(q_weight.dim(), 1);
    CHECK_EQ(k_weight.dim(), 1);
    CHECK_EQ(q_weight.scalar_type(), torch::kBFloat16);
    CHECK_EQ(k_weight.scalar_type(), torch::kBFloat16);
    CHECK_EQ(q_weight.size(0), kHeadDim);
    CHECK_EQ(k_weight.size(0), kHeadDim);

    auto q_out = torch::empty(q.sizes(), q.options());
    auto k_out = torch::empty(k.sizes(), k.options());
    launch_mtgr_qk_norm_strided_bf16_hd128(q_out, q, q_weight, eps);
    launch_mtgr_qk_norm_strided_bf16_hd128(k_out, k, k_weight, eps);
    return {q_out, k_out};
  }

  const char* name() const override { return "strided_bf16_hd128_kernel"; }
};

}  // namespace

namespace xllm::kernel::cuda::test {

std::unique_ptr<MTGRQKNormTestImpl> make_mtgr_qk_norm_strided_bf16_hd128() {
  return std::make_unique<MTGRQKNormStridedBf16Hd128Impl>();
}

}  // namespace xllm::kernel::cuda::test
