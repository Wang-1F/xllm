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

#include "mtgr_qk_norm_contract.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdio>
#include <tuple>

#include "layers/common/qwen3_next_rms_norm.h"

namespace xllm::kernel::cuda::test::mtgr_qk_norm_harness {

torch::Tensor run_project_module_qwen3_next_rms_norm(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    double eps) {
  auto norm = xllm::layer::Qwen3NextRMSNorm(
      input.size(-1), eps, input.options().requires_grad(false));
  norm->load_state_dict(xllm::StateDict({{"weight", weight}}));
  auto input_copy = input.clone();
  return norm->forward(input_copy);
}

std::tuple<torch::Tensor, torch::Tensor> make_sliced_qk_from_qkv(
    int64_t tokens,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    const torch::TensorOptions& opts) {
  const int64_t q_size = num_q_heads * head_dim;
  const int64_t kv_size = num_kv_heads * head_dim;
  auto qkv = torch::randn({tokens, q_size + 2 * kv_size}, opts) * 0.1;
  auto q = qkv.slice(-1, 0, q_size).reshape({tokens, num_q_heads, head_dim});
  auto k = qkv.slice(-1, q_size, q_size + kv_size)
               .reshape({tokens, num_kv_heads, head_dim});
  return {q, k};
}

void expect_close(const char* tag,
                  const torch::Tensor& got,
                  const torch::Tensor& expected,
                  double atol,
                  double rtol) {
  auto got_f32 = got.to(torch::kFloat32);
  auto expected_f32 = expected.to(torch::kFloat32);
  auto diff = (got_f32 - expected_f32).abs();
  const double max_abs = diff.max().item<double>();
  const double mean_abs = diff.mean().item<double>();
  std::fprintf(stderr,
               "[MTGR][QKNorm][%s] max_abs=%.6e mean_abs=%.6e\n",
               tag,
               max_abs,
               mean_abs);
  std::fflush(stderr);
  EXPECT_TRUE(torch::allclose(got, expected, rtol, atol));
}

}  // namespace xllm::kernel::cuda::test::mtgr_qk_norm_harness
