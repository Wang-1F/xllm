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

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <nvtx3/nvToolsExt.h>
#include <torch/cuda.h>
#include <torch/torch.h>

#include <cstdio>
#include <memory>
#include <string>
#include <tuple>

#include "layers/common/qwen3_next_rms_norm.h"
#include "mtgr_qk_norm_test.h"

namespace xllm::kernel::cuda::test {
namespace {

torch::Tensor run_project_module_qwen3_next_rms_norm(const torch::Tensor& input,
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

class ScopedNvtxRange {
 public:
  explicit ScopedNvtxRange(const std::string& name) { nvtxRangePushA(name.c_str()); }
  ~ScopedNvtxRange() { nvtxRangePop(); }
};

void run_nvtx_replay(MTGRQKNormTestImpl* impl,
                     const torch::Tensor& q,
                     const torch::Tensor& k,
                     const torch::Tensor& q_weight,
                     const torch::Tensor& k_weight,
                     double eps,
                     int warmup,
                     int iters) {
  CHECK(impl != nullptr);
  for (int i = 0; i < warmup; ++i) {
    auto outputs = impl->forward(q, k, q_weight, k_weight, eps);
    (void)outputs;
  }
  CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const std::string range_name =
      std::string("MTGR/QKNorm/") + impl->name() + "/replay";
  {
    ScopedNvtxRange range(range_name);
    for (int i = 0; i < iters; ++i) {
      auto outputs = impl->forward(q, k, q_weight, k_weight, eps);
      (void)outputs;
    }
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }
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

class MTGRQKNormE2ETest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA not available";
    }
    torch::manual_seed(20260518);
    device_ = torch::Device(torch::kCUDA, 0);
  }

  torch::Device device_ = torch::Device(torch::kCPU);
};

TEST_F(MTGRQKNormE2ETest, ProjectBaselineMatchesProjectModuleForContiguousBF16) {
  torch::NoGradGuard no_grad_guard;
  constexpr int64_t kTokens = 257;
  constexpr int64_t kNumQHeads = 16;
  constexpr int64_t kNumKVHeads = 8;
  constexpr int64_t kHeadDim = 128;
  constexpr double kEps = 1e-6;
  auto opts =
      torch::TensorOptions().device(device_).dtype(torch::kBFloat16);

  auto q = torch::randn({kTokens, kNumQHeads, kHeadDim}, opts) * 0.1;
  auto k = torch::randn({kTokens, kNumKVHeads, kHeadDim}, opts) * 0.1;
  auto q_weight = torch::randn({kHeadDim}, opts) * 0.05;
  auto k_weight = torch::randn({kHeadDim}, opts) * 0.05;

  auto impl = make_mtgr_qk_norm_project_baseline();
  auto [q_out, k_out] = impl->forward(q, k, q_weight, k_weight, kEps);

  auto q_expected = run_project_module_qwen3_next_rms_norm(q, q_weight, kEps);
  auto k_expected = run_project_module_qwen3_next_rms_norm(k, k_weight, kEps);

  EXPECT_EQ(q_out.sizes(), q.sizes());
  EXPECT_EQ(k_out.sizes(), k.sizes());
  EXPECT_EQ(q_out.scalar_type(), q.scalar_type());
  EXPECT_EQ(k_out.scalar_type(), k.scalar_type());
  expect_close("baseline_contiguous_q", q_out, q_expected, 1e-5, 1e-5);
  expect_close("baseline_contiguous_k", k_out, k_expected, 1e-5, 1e-5);
}

TEST_F(MTGRQKNormE2ETest,
       ProjectBaselineMatchesProjectModuleForSlicedQKVViews) {
  torch::NoGradGuard no_grad_guard;
  constexpr int64_t kTokens = 319;
  constexpr int64_t kNumQHeads = 16;
  constexpr int64_t kNumKVHeads = 8;
  constexpr int64_t kHeadDim = 128;
  constexpr double kEps = 1e-6;
  auto opts =
      torch::TensorOptions().device(device_).dtype(torch::kBFloat16);

  auto [q, k] =
      make_sliced_qk_from_qkv(kTokens, kNumQHeads, kNumKVHeads, kHeadDim, opts);
  auto q_weight = torch::randn({kHeadDim}, opts) * 0.05;
  auto k_weight = torch::randn({kHeadDim}, opts) * 0.05;

  auto impl = make_mtgr_qk_norm_project_baseline();
  auto [q_out, k_out] = impl->forward(q, k, q_weight, k_weight, kEps);

  auto q_expected = run_project_module_qwen3_next_rms_norm(q, q_weight, kEps);
  auto k_expected = run_project_module_qwen3_next_rms_norm(k, k_weight, kEps);

  expect_close("baseline_sliced_q", q_out, q_expected, 1e-5, 1e-5);
  expect_close("baseline_sliced_k", k_out, k_expected, 1e-5, 1e-5);
}

TEST_F(MTGRQKNormE2ETest, CudaRmsNormMatchesProjectBaselineForSlicedQKVViews) {
  torch::NoGradGuard no_grad_guard;
  constexpr int64_t kTokens = 319;
  constexpr int64_t kNumQHeads = 16;
  constexpr int64_t kNumKVHeads = 8;
  constexpr int64_t kHeadDim = 128;
  constexpr double kEps = 1e-6;
  auto opts =
      torch::TensorOptions().device(device_).dtype(torch::kBFloat16);

  auto [q, k] =
      make_sliced_qk_from_qkv(kTokens, kNumQHeads, kNumKVHeads, kHeadDim, opts);
  auto q_weight = torch::randn({kHeadDim}, opts) * 0.05;
  auto k_weight = torch::randn({kHeadDim}, opts) * 0.05;

  auto baseline = make_mtgr_qk_norm_project_baseline();
  auto custom = make_mtgr_qk_norm_cuda_rms_norm();
  auto [q_expected, k_expected] =
      baseline->forward(q, k, q_weight, k_weight, kEps);
  auto [q_out, k_out] = custom->forward(q, k, q_weight, k_weight, kEps);

  expect_close("custom_sliced_q", q_out, q_expected, 2e-2, 2e-2);
  expect_close("custom_sliced_k", k_out, k_expected, 2e-2, 2e-2);
}

TEST_F(MTGRQKNormE2ETest,
       StridedBf16Hd128KernelMatchesProjectBaselineForSlicedQKVViews) {
  torch::NoGradGuard no_grad_guard;
  constexpr int64_t kTokens = 319;
  constexpr int64_t kNumQHeads = 16;
  constexpr int64_t kNumKVHeads = 8;
  constexpr int64_t kHeadDim = 128;
  constexpr double kEps = 1e-6;
  auto opts =
      torch::TensorOptions().device(device_).dtype(torch::kBFloat16);

  auto [q, k] =
      make_sliced_qk_from_qkv(kTokens, kNumQHeads, kNumKVHeads, kHeadDim, opts);
  auto q_weight = torch::randn({kHeadDim}, opts) * 0.05;
  auto k_weight = torch::randn({kHeadDim}, opts) * 0.05;

  auto baseline = make_mtgr_qk_norm_project_baseline();
  auto custom = make_mtgr_qk_norm_strided_bf16_hd128();
  auto [q_expected, k_expected] =
      baseline->forward(q, k, q_weight, k_weight, kEps);
  auto [q_out, k_out] = custom->forward(q, k, q_weight, k_weight, kEps);

  expect_close("strided_kernel_q", q_out, q_expected, 2e-2, 2e-2);
  expect_close("strided_kernel_k", k_out, k_expected, 2e-2, 2e-2);
}

TEST_F(MTGRQKNormE2ETest, NvtxReplaySmokeForThreeImplementationsBF16) {
  torch::NoGradGuard no_grad_guard;
  constexpr int64_t kTokens = 2558;
  constexpr int64_t kNumQHeads = 16;
  constexpr int64_t kNumKVHeads = 8;
  constexpr int64_t kHeadDim = 128;
  constexpr double kEps = 1e-6;
  auto opts =
      torch::TensorOptions().device(device_).dtype(torch::kBFloat16);

  auto [q, k] =
      make_sliced_qk_from_qkv(kTokens, kNumQHeads, kNumKVHeads, kHeadDim, opts);
  auto q_weight = torch::randn({kHeadDim}, opts) * 0.05;
  auto k_weight = torch::randn({kHeadDim}, opts) * 0.05;

  auto baseline = make_mtgr_qk_norm_project_baseline();
  auto reshape_plus_kernel = make_mtgr_qk_norm_cuda_rms_norm();
  auto strided_kernel = make_mtgr_qk_norm_strided_bf16_hd128();

  std::fprintf(stderr,
               "[MTGR][QKNorm][nvtx_replay] use NVTX ranges under "
               "MTGR/QKNorm/<impl>/replay for host-side profiling\n");
  std::fflush(stderr);
  run_nvtx_replay(baseline.get(), q, k, q_weight, k_weight, kEps, 20, 1000);
  run_nvtx_replay(
      reshape_plus_kernel.get(), q, k, q_weight, k_weight, kEps, 20, 1000);
  run_nvtx_replay(
      strided_kernel.get(), q, k, q_weight, k_weight, kEps, 20, 1000);
}

}  // namespace
}  // namespace xllm::kernel::cuda::test
