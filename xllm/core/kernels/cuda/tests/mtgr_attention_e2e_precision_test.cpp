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
#include "layers/cuda/mtgr_attention.h"
#include "mtgr_attenion_test.h"

namespace xllm::kernel::cuda::test {
namespace {

constexpr double kMaxAbs = 2.0e-1;
constexpr double kMeanAbs = 5.0e-3;
constexpr int64_t kBlockSize = 128;

class MTGRAttentionE2EPrecisionTest : public ::testing::Test {
 protected:
  void maybe_print_history_prefix_reference(const char* name,
                                            const MTGRAttentionTestShape& shape,
                                            const torch::Tensor& query_f32,
                                            const torch::Tensor& key_f32,
                                            const torch::Tensor& value_f32,
                                            const torch::Tensor& one_f32,
                                            const torch::Tensor& candidate_f32,
                                            float scale) {
    if (shape.matched_prefix != 0 || shape.kv_heads != shape.heads ||
        shape.history <= 0) {
      return;
    }
    auto q_view =
        query_f32.view({shape.local_len(), shape.heads, shape.head_dim});
    auto k_view =
        key_f32.view({shape.local_len(), shape.kv_heads, shape.head_dim});
    auto v_view =
        value_f32.view({shape.local_len(), shape.kv_heads, shape.head_dim});
    const int64_t rows = std::min<int64_t>(5, shape.history);
    for (int64_t row = 0; row < rows; ++row) {
      auto q_row = q_view[row];
      auto k_prefix = k_view.narrow(0, 0, row + 1);
      auto v_prefix = v_view.narrow(0, 0, row + 1);
      auto score =
          (k_prefix * q_row.unsqueeze(0)).sum(-1) * static_cast<double>(scale);
      auto prob = torch::softmax(score, 0);
      auto ref_row =
          (prob.unsqueeze(-1) * v_prefix).sum(0).contiguous().view({-1});
      const double ref_vs_one =
          (ref_row - one_f32[row]).abs().max().item<double>();
      const double ref_vs_cand =
          (ref_row - candidate_f32[row]).abs().max().item<double>();
      std::fprintf(stderr,
                   "[MTGR][CUDA][Precision][%s][history_ref][row=%lld] "
                   "ref_vs_one=%.6e ref_vs_cand=%.6e\n",
                   name,
                   static_cast<long long>(row),
                   ref_vs_one,
                   ref_vs_cand);
    }
  }

  void maybe_print_segment_diff(const char* name,
                                const MTGRAttentionTestShape& shape,
                                const torch::Tensor& one_f32,
                                const torch::Tensor& candidate_f32,
                                const torch::Tensor& value_f32) {
    if (shape.matched_prefix != 0) {
      return;
    }
    const struct SegmentDesc {
      const char* name;
      int64_t len;
    } segments[] = {{"history", shape.history},
                    {"context", shape.context},
                    {"realtime", shape.realtime},
                    {"target", shape.target}};
    int64_t offset = 0;
    for (const auto& segment : segments) {
      if (segment.len <= 0) {
        continue;
      }
      auto segment_diff = (one_f32.narrow(0, offset, segment.len) -
                           candidate_f32.narrow(0, offset, segment.len))
                              .abs();
      const double segment_max_abs = segment_diff.max().item<double>();
      const double segment_mean_abs = segment_diff.mean().item<double>();
      std::fprintf(stderr,
                   "[MTGR][CUDA][Precision][%s][segment=%s] offset=%lld "
                   "len=%lld max_abs=%.6e mean_abs=%.6e\n",
                   name,
                   segment.name,
                   static_cast<long long>(offset),
                   static_cast<long long>(segment.len),
                   segment_max_abs,
                   segment_mean_abs);
      if (std::strcmp(segment.name, "history") == 0) {
        auto row_max = std::get<0>(segment_diff.max(1));
        const int64_t topk = std::min<int64_t>(5, row_max.size(0));
        auto topk_result =
            row_max.topk(topk, 0, /*largest=*/true, /*sorted=*/true);
        auto topk_values = std::get<0>(topk_result).cpu();
        auto topk_indices = std::get<1>(topk_result).cpu();
        for (int64_t i = 0; i < topk; ++i) {
          std::fprintf(stderr,
                       "[MTGR][CUDA][Precision][%s][segment=%s][top_row=%lld] "
                       "row_max_abs=%.6e\n",
                       name,
                       segment.name,
                       static_cast<long long>(topk_indices[i].item<int64_t>()),
                       topk_values[i].item<double>());
        }
        if (shape.kv_heads == shape.heads && segment.len > 0) {
          auto one_row0 = one_f32[0];
          auto cand_row0 = candidate_f32[0];
          auto value_row0 = value_f32[0];
          const double one_vs_value =
              (one_row0 - value_row0).abs().max().item<double>();
          const double cand_vs_value =
              (cand_row0 - value_row0).abs().max().item<double>();
          const double cand_vs_one =
              (cand_row0 - one_row0).abs().max().item<double>();
          std::fprintf(stderr,
                       "[MTGR][CUDA][Precision][%s][segment=%s][row0_check] "
                       "one_vs_value=%.6e cand_vs_value=%.6e "
                       "cand_vs_one=%.6e\n",
                       name,
                       segment.name,
                       one_vs_value,
                       cand_vs_value,
                       cand_vs_one);
        }
      }
      offset += segment.len;
    }
  }

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
    run_case_vs_backend(
        name, shape, xllm::layer::MTGRAttentionBackend::kMultiStage);
  }

  void run_case_vs_backend(
      const char* name,
      MTGRAttentionTestShape shape,
      xllm::layer::MTGRAttentionBackend candidate_backend) {
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

    xllm::layer::MTGRAttentionImpl one_stage(
        shape.heads,
        shape.head_dim,
        scale,
        shape.kv_heads,
        xllm::layer::MTGRAttentionBackend::kOneStage);
    xllm::layer::MTGRAttentionImpl candidate(
        shape.heads, shape.head_dim, scale, shape.kv_heads, candidate_backend);

    auto one_output =
        std::get<0>(one_stage.forward(metadata, query, key, value, one_cache));
    auto candidate_output = std::get<0>(
        candidate.forward(metadata, query, key, value, multi_cache));
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

    auto query_f32 = query.to(torch::kFloat32);
    auto key_f32 = key.to(torch::kFloat32);
    auto value_f32 = value.to(torch::kFloat32);
    auto one_f32 = one_output.to(torch::kFloat32);
    auto candidate_f32 = candidate_output.to(torch::kFloat32);
    auto diff = (one_f32 - candidate_f32).abs();
    const double max_abs = diff.max().item<double>();
    const double mean_abs = diff.mean().item<double>();
    const bool threshold_ok = max_abs < kMaxAbs && mean_abs < kMeanAbs;
    maybe_print_history_prefix_reference(name,
                                         shape,
                                         query_f32,
                                         key_f32,
                                         value_f32,
                                         one_f32,
                                         candidate_f32,
                                         scale);
    maybe_print_segment_diff(name, shape, one_f32, candidate_f32, value_f32);

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
                      xllm::layer::MTGRAttentionBackend::kFused);
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
                      xllm::layer::MTGRAttentionBackend::kFused);
}

}  // namespace xllm::kernel::cuda::test
