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

namespace xllm::kernel::cuda::test {

class MTGRQKNormTestImpl {
 public:
  virtual ~MTGRQKNormTestImpl() = default;

  virtual std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& q,
      const torch::Tensor& k,
      const torch::Tensor& q_weight,
      const torch::Tensor& k_weight,
      double eps) = 0;

  virtual const char* name() const = 0;
};

std::unique_ptr<MTGRQKNormTestImpl> make_mtgr_qk_norm_project_baseline();
std::unique_ptr<MTGRQKNormTestImpl> make_mtgr_qk_norm_cuda_rms_norm();
std::unique_ptr<MTGRQKNormTestImpl> make_mtgr_qk_norm_strided_bf16_hd128();

}  // namespace xllm::kernel::cuda::test
