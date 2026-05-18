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

#include <glog/logging.h>

#include <memory>
#include <tuple>

#include "core/kernels/cuda/cuda_ops_api.h"

namespace xllm::kernel::cuda::test {
namespace {

torch::Tensor run_project_baseline_qwen3_next_rms_norm(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    double eps) {
  CHECK_EQ(input.dim(), 3);
  CHECK_EQ(weight.dim(), 1);
  CHECK_EQ(input.size(-1), weight.size(0));

  auto input_dtype = input.dtype();
  auto input_fp32 = input.to(torch::kFloat32);
  auto variance = torch::mean(torch::pow(input_fp32, 2), -1, true);
  auto normalized = input_fp32 * torch::rsqrt(variance + eps);
  return (normalized * (1.0f + weight.to(torch::kFloat32))).to(input_dtype);
}

class MTGRQKNormProjectBaselineImpl final : public MTGRQKNormTestImpl {
 public:
  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& q,
      const torch::Tensor& k,
      const torch::Tensor& q_weight,
      const torch::Tensor& k_weight,
      double eps) override {
    return {run_project_baseline_qwen3_next_rms_norm(q, q_weight, eps),
            run_project_baseline_qwen3_next_rms_norm(k, k_weight, eps)};
  }

  const char* name() const override { return "project_baseline"; }
};

class MTGRQKNormCudaRmsNormImpl final : public MTGRQKNormTestImpl {
 public:
  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& q,
      const torch::Tensor& k,
      const torch::Tensor& q_weight,
      const torch::Tensor& k_weight,
      double eps) override {
    auto q_2d = q.reshape({-1, q.size(-1)});
    auto k_2d = k.reshape({-1, k.size(-1)});
    auto q_out_2d = torch::empty_like(q_2d);
    auto k_out_2d = torch::empty_like(k_2d);
    auto q_weight_plus_one = q_weight + 1;
    auto k_weight_plus_one = k_weight + 1;
    xllm::kernel::cuda::rms_norm(q_out_2d, q_2d, q_weight_plus_one, eps);
    xllm::kernel::cuda::rms_norm(k_out_2d, k_2d, k_weight_plus_one, eps);
    return {q_out_2d.view_as(q), k_out_2d.view_as(k)};
  }

  const char* name() const override { return "cuda_rms_norm"; }
};

}  // namespace

std::unique_ptr<MTGRQKNormTestImpl> make_mtgr_qk_norm_project_baseline() {
  return std::make_unique<MTGRQKNormProjectBaselineImpl>();
}

std::unique_ptr<MTGRQKNormTestImpl> make_mtgr_qk_norm_cuda_rms_norm() {
  return std::make_unique<MTGRQKNormCudaRmsNormImpl>();
}

}  // namespace xllm::kernel::cuda::test
