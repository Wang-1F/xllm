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

#pragma once

#include <torch/torch.h>

#include <memory>
#include <tuple>

namespace xllm::kernel::cuda::test::mtgr_qk_norm_harness {

class IMTGRQKNormBackend {
 public:
  virtual ~IMTGRQKNormBackend() = default;

  virtual std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& q,
      const torch::Tensor& k,
      const torch::Tensor& q_weight,
      const torch::Tensor& k_weight,
      double eps) = 0;

  virtual const char* name() const = 0;
};

std::unique_ptr<IMTGRQKNormBackend> make_project_baseline_backend();
std::unique_ptr<IMTGRQKNormBackend> make_cuda_rms_norm_backend();
std::unique_ptr<IMTGRQKNormBackend> make_strided_bf16_hd128_backend();

torch::Tensor run_project_module_qwen3_next_rms_norm(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    double eps);

std::tuple<torch::Tensor, torch::Tensor> make_sliced_qk_from_qkv(
    int64_t tokens,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    const torch::TensorOptions& opts);

void expect_close(const char* tag,
                  const torch::Tensor& got,
                  const torch::Tensor& expected,
                  double atol,
                  double rtol);

}  // namespace xllm::kernel::cuda::test::mtgr_qk_norm_harness
