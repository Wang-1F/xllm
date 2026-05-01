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
#include <torch/cuda.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdio>

#include "core/common/global_flags.h"
#include "mtgr_attenion_test.h"

namespace xllm::kernel::cuda::test {
namespace {

constexpr double kMaxAbs = 2.0e-1;
constexpr double kMeanAbs = 5.0e-3;
constexpr int64_t kBlockSize = 128;

class MTGRAttentionE2EPrecisionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA not available, skipping test.";
    }
    FLAGS_flashinfer_workspace_buffer_size = std::max<int64_t>(
        FLAGS_flashinfer_workspace_buffer_size, 64 * 1024 * 1024);
    FLAGS_block_size = kBlockSize;
    torch::manual_seed(20260429);
    torch::cuda::manual_seed_all(20260429);
    device_ = torch::Device(torch::kCUDA, 0);
  }

  void run_case(const char* name, MTGRAttentionTestShape shape) {
    run_case_vs_backend(name, shape, MTGRAttentionTestBackend::kMultiStage);
  }

  void run_case_vs_backend(const char* name,
                           MTGRAttentionTestShape shape,
                           MTGRAttentionTestBackend candidate_backend) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat16).device(device_);
    const int64_t total = shape.total_len();
    const int64_t local = shape.local_len();
    const float scale = 1.0f / std::sqrt(static_cast<float>(shape.head_dim));

    auto full_query =
        torch::randn({1, total, shape.heads, shape.head_dim}, opts) * 0.05;
    auto full_key =
        torch::randn({1, total, shape.kv_heads, shape.head_dim}, opts) * 0.05;
    auto full_value =
        torch::randn({1, total, shape.kv_heads, shape.head_dim}, opts) * 0.05;

    auto query = full_query.select(0, 0)
                     .narrow(0, shape.matched_prefix, local)
                     .contiguous()
                     .view({local, shape.heads * shape.head_dim});
    auto key = full_key.select(0, 0)
                   .narrow(0, shape.matched_prefix, local)
                   .contiguous()
                   .view({local, shape.kv_heads * shape.head_dim});
    auto value = full_value.select(0, 0)
                     .narrow(0, shape.matched_prefix, local)
                     .contiguous()
                     .view({local, shape.kv_heads * shape.head_dim});

    auto metadata = make_mtgr_attention_metadata(shape, device_, kBlockSize);
    auto one_cache =
        make_mtgr_kv_cache(shape, device_, torch::kFloat16, kBlockSize);
    auto multi_cache =
        make_mtgr_kv_cache(shape, device_, torch::kFloat16, kBlockSize);
    prefill_mtgr_matched_prefix_cache(
        full_key, full_value, shape, kBlockSize, one_cache);
    prefill_mtgr_matched_prefix_cache(
        full_key, full_value, shape, kBlockSize, multi_cache);

    MTGRAttentionImplTest one_stage(shape.heads,
                                    shape.head_dim,
                                    scale,
                                    shape.kv_heads,
                                    MTGRAttentionTestBackend::kOneStage);
    MTGRAttentionImplTest candidate(
        shape.heads, shape.head_dim, scale, shape.kv_heads, candidate_backend);

    auto one_output =
        std::get<0>(one_stage.forward(metadata, query, key, value, one_cache));
    auto candidate_output = std::get<0>(
        candidate.forward(metadata, query, key, value, multi_cache));
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

    auto one_f32 = one_output.to(torch::kFloat32);
    auto candidate_f32 = candidate_output.to(torch::kFloat32);
    auto diff = (one_f32 - candidate_f32).abs();
    const double max_abs = diff.max().item<double>();
    const double mean_abs = diff.mean().item<double>();
    const bool threshold_ok = max_abs < kMaxAbs && mean_abs < kMeanAbs;

    std::fprintf(stderr,
                 "[MTGR][CUDA][Precision][%s] matched=%lld local=%lld "
                 "max_abs=%.6e mean_abs=%.6e threshold_ok=%s\n",
                 name,
                 static_cast<long long>(shape.matched_prefix),
                 static_cast<long long>(local),
                 max_abs,
                 mean_abs,
                 threshold_ok ? "true" : "false");
    std::fflush(stderr);

    ASSERT_TRUE(torch::isfinite(one_output).all().item<bool>());
    ASSERT_TRUE(torch::isfinite(candidate_output).all().item<bool>());
    EXPECT_LT(max_abs, kMaxAbs);
    EXPECT_LT(mean_abs, kMeanAbs);
    EXPECT_TRUE(threshold_ok);
  }

  torch::Device device_ = torch::Device(torch::kCPU);
};

}  // namespace

TEST_F(MTGRAttentionE2EPrecisionTest, NoMatchAlignsWithOneStage) {
  MTGRAttentionTestShape shape;
  shape.heads = 8;
  shape.kv_heads = 8;
  shape.head_dim = 128;
  shape.history = 512;
  shape.context = 8;
  shape.realtime = 160;
  shape.target = 320;
  shape.matched_prefix = 0;
  run_case("no_match", shape);
}

TEST_F(MTGRAttentionE2EPrecisionTest, PartialRealTimeMatchAlignsWithOneStage) {
  MTGRAttentionTestShape shape;
  shape.heads = 8;
  shape.kv_heads = 8;
  shape.head_dim = 128;
  shape.history = 512;
  shape.context = 8;
  shape.realtime = 160;
  shape.target = 320;
  shape.matched_prefix = shape.history + shape.context + 96;
  run_case("partial_real_time_match", shape);
}

TEST_F(MTGRAttentionE2EPrecisionTest, FusedNoMatchHotShapeAlignsWithOneStage) {
  MTGRAttentionTestShape shape;
  shape.heads = 8;
  shape.kv_heads = 8;
  shape.head_dim = 128;
  shape.history = 2048;
  shape.context = 8;
  shape.realtime = 512;
  shape.target = 1600;
  shape.matched_prefix = 0;
  run_case_vs_backend("fused_no_match_hot_shape",
                      shape,
                      MTGRAttentionTestBackend::kFusedNoMatch);
}

TEST_F(MTGRAttentionE2EPrecisionTest,
       FusedPartialRealTimeMatchHotShapeAlignsWithOneStage) {
  MTGRAttentionTestShape shape;
  shape.heads = 8;
  shape.kv_heads = 8;
  shape.head_dim = 128;
  shape.history = 2048;
  shape.context = 8;
  shape.realtime = 512;
  shape.target = 1600;
  shape.matched_prefix = shape.history + shape.context + 409;
  run_case_vs_backend("fused_partial_real_time_match_hot_shape",
                      shape,
                      MTGRAttentionTestBackend::kFusedNoMatch);
}

}  // namespace xllm::kernel::cuda::test
