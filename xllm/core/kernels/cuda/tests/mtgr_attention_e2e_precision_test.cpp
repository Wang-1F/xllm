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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/common/global_flags.h"
#include "layers/cuda/mtgr_attention.h"
#include "mtgr_attenion_test.h"

namespace xllm::kernel::cuda::test {
namespace {

constexpr double kMaxAbs = 2.0e-1;
constexpr double kMeanAbs = 5.0e-3;
constexpr int64_t kBlockSize = 128;

struct MultiBatchPrecisionMetrics {
  double max_abs = 0.0;
  double mean_abs = 0.0;
  bool finite = true;
};

int64_t sample_non_aligned_len(std::mt19937_64* rng, int64_t lo, int64_t hi) {
  CHECK(rng != nullptr);
  CHECK_LE(lo, hi);
  std::uniform_int_distribution<int64_t> dist(lo, hi);
  for (int attempt = 0; attempt < 1024; ++attempt) {
    const int64_t v = dist(*rng);
    if ((v % 32) != 0 && (v % 64) != 0) {
      return v;
    }
  }
  int64_t v = dist(*rng);
  if ((v % 32) != 0 && (v % 64) != 0) {
    return v;
  }
  if (v < hi) {
    ++v;
  } else if (v > lo) {
    --v;
  }
  return v;
}

std::string describe_multi_batch_shapes(
    const std::vector<MTGRAttentionTestShape>& shapes) {
  std::ostringstream oss;
  for (size_t i = 0; i < shapes.size(); ++i) {
    if (i != 0) {
      oss << " | ";
    }
    const auto& shape = shapes[i];
    oss << "#" << i << "(h=" << shape.history << ",r=" << shape.realtime
        << ",t=" << shape.target << ")";
  }
  return oss.str();
}

MTGRAttentionTestShape sample_odd_no_match_shape(std::mt19937_64* rng,
                                                 int64_t heads,
                                                 int64_t head_dim) {
  CHECK(rng != nullptr);
  MTGRAttentionTestShape shape;
  shape.heads = heads;
  shape.kv_heads = heads;
  shape.head_dim = head_dim;
  shape.history = sample_non_aligned_len(rng, 1350, 4096);
  shape.context = 8;
  shape.realtime = sample_non_aligned_len(rng, 100, 600);
  shape.target = sample_non_aligned_len(rng, 800, 2400);
  shape.matched_prefix = 0;
  return shape;
}

int64_t segmented_visible_end(int64_t q_local,
                              const std::vector<int64_t>& offsets,
                              const std::vector<int64_t>& rules) {
  CHECK_EQ(offsets.size(), rules.size() + 1);
  int64_t seg_id = 0;
  for (; seg_id < static_cast<int64_t>(rules.size()); ++seg_id) {
    if (q_local < offsets[seg_id + 1]) {
      break;
    }
  }
  CHECK_LT(seg_id, static_cast<int64_t>(rules.size()));
  const int64_t rule = rules[seg_id];
  if (rule == 1) {
    return offsets[seg_id + 1];
  }
  if (rule == 0) {
    return q_local + 1;
  }
  CHECK_EQ(rule, 2);
  return offsets[seg_id];
}

torch::Tensor run_ragged_segment_reference(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    const std::vector<std::vector<int64_t>>& segment_lens,
    const std::vector<int64_t>& segment_rules,
    double sm_scale) {
  CHECK_EQ(query_snd.dim(), 3);
  CHECK_EQ(query_snd.sizes(), key_snd.sizes());
  CHECK_EQ(query_snd.sizes(), value_snd.sizes());
  CHECK(!segment_lens.empty());
  CHECK(!segment_rules.empty());
  for (const int64_t rule : segment_rules) {
    CHECK(rule == 0 || rule == 1 || rule == 2);
  }
  const int64_t num_segments = static_cast<int64_t>(segment_rules.size());
  auto query = query_snd.to(torch::kFloat32);
  auto key = key_snd.to(torch::kFloat32);
  auto value = value_snd.to(torch::kFloat32);
  auto output = torch::empty_like(query);
  int64_t request_start = 0;
  for (const auto& lens : segment_lens) {
    CHECK_EQ(static_cast<int64_t>(lens.size()), num_segments);
    std::vector<int64_t> offsets;
    offsets.reserve(static_cast<size_t>(num_segments + 1));
    offsets.push_back(0);
    for (const int64_t len : lens) {
      CHECK_GT(len, 0);
      offsets.push_back(offsets.back() + len);
    }
    for (int64_t q_local = 0; q_local < offsets.back(); ++q_local) {
      int64_t seg_id = 0;
      for (; seg_id < num_segments; ++seg_id) {
        if (q_local < offsets[seg_id + 1]) {
          break;
        }
      }
      CHECK_LT(seg_id, num_segments);
      const int64_t q_idx = request_start + q_local;
      const int64_t visible_end =
          segmented_visible_end(q_local, offsets, segment_rules);
      auto k_attn = key.narrow(0, request_start, visible_end);
      auto v_attn = value.narrow(0, request_start, visible_end);
      if (segment_rules[seg_id] == 2) {
        k_attn = torch::cat({k_attn, key.narrow(0, q_idx, 1)}, 0);
        v_attn = torch::cat({v_attn, value.narrow(0, q_idx, 1)}, 0);
      }
      auto q_row = query.select(0, q_idx);
      auto scores =
          (k_attn * q_row.unsqueeze(0)).sum(-1) * static_cast<float>(sm_scale);
      auto probs = torch::softmax(scores, 0);
      output.select(0, q_idx).copy_((probs.unsqueeze(-1) * v_attn).sum(0));
    }
    request_start += offsets.back();
  }
  CHECK_EQ(request_start, query_snd.size(0));
  return output;
}

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

  MultiBatchPrecisionMetrics measure_multi_batch_fused_case(
      const char* name,
      const std::vector<MTGRAttentionTestShape>& shapes) {
    CHECK(!shapes.empty());
    const auto& ref = shapes.front();
    const float scale = 1.0f / std::sqrt(static_cast<float>(ref.head_dim));
    auto opts = torch::TensorOptions().dtype(torch::kFloat16).device(device_);
    const bool all_no_match =
        std::all_of(shapes.begin(), shapes.end(), [](const auto& shape) {
          return shape.matched_prefix == 0;
        });
    const bool all_partial =
        std::all_of(shapes.begin(), shapes.end(), [](const auto& shape) {
          return shape.matched_prefix > 0;
        });
    CHECK(all_no_match || all_partial)
        << "Mixed no_match and partial_match batches are not supported";

    std::vector<torch::Tensor> packed_queries;
    std::vector<torch::Tensor> packed_keys;
    std::vector<torch::Tensor> packed_values;
    std::vector<torch::Tensor> full_keys;
    std::vector<torch::Tensor> full_values;
    std::vector<torch::Tensor> baseline_outputs;
    packed_queries.reserve(shapes.size());
    packed_keys.reserve(shapes.size());
    packed_values.reserve(shapes.size());
    full_keys.reserve(shapes.size());
    full_values.reserve(shapes.size());
    baseline_outputs.reserve(shapes.size());

    xllm::layer::MTGRAttentionImpl one_stage(
        ref.heads,
        ref.head_dim,
        scale,
        ref.kv_heads,
        xllm::layer::MTGRAttentionBackend::kOneStage);
    for (const auto& shape : shapes) {
      CHECK_EQ(shape.heads, ref.heads);
      CHECK_EQ(shape.kv_heads, ref.kv_heads);
      CHECK_EQ(shape.head_dim, ref.head_dim);

      const int64_t total = shape.total_len();
      const int64_t local = shape.local_len();
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
      prefill_mtgr_matched_prefix_cache(
          full_key, full_value, shape, kBlockSize, one_cache);
      baseline_outputs.push_back(std::get<0>(
          one_stage.forward(metadata, query, key, value, one_cache)));
      packed_queries.push_back(query);
      packed_keys.push_back(key);
      packed_values.push_back(value);
      full_keys.push_back(full_key.contiguous());
      full_values.push_back(full_value.contiguous());
    }

    auto packed_query = torch::cat(packed_queries, 0).contiguous();
    auto packed_key = torch::cat(packed_keys, 0).contiguous();
    auto packed_value = torch::cat(packed_values, 0).contiguous();
    auto baseline_output = torch::cat(baseline_outputs, 0).contiguous();

    xllm::layer::MTGRAttentionImpl fused(
        ref.heads,
        ref.head_dim,
        scale,
        ref.kv_heads,
        xllm::layer::MTGRAttentionBackend::kFused);
    torch::Tensor fused_output;
    if (all_no_match) {
      auto batch_metadata =
          make_mtgr_attention_metadata(shapes, device_, kBlockSize);
      xllm::KVCache empty_cache;
      fused_output = std::get<0>(fused.forward(
          batch_metadata, packed_query, packed_key, packed_value, empty_cache));
    } else {
      auto batch_setup = make_mtgr_partial_batch_setup(
          shapes, full_keys, full_values, device_, torch::kFloat16, kBlockSize);
      fused_output = std::get<0>(fused.forward(batch_setup.metadata,
                                               packed_query,
                                               packed_key,
                                               packed_value,
                                               batch_setup.kv_cache));
    }
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

    auto baseline_f32 = baseline_output.to(torch::kFloat32);
    auto fused_f32 = fused_output.to(torch::kFloat32);
    auto diff = (baseline_f32 - fused_f32).abs();
    const double max_abs = diff.max().item<double>();
    const double mean_abs = diff.mean().item<double>();
    std::fprintf(stderr,
                 "[MTGR][CUDA][Precision][%s] batch=%zu rows=%lld "
                 "max_abs=%.6e mean_abs=%.6e\n",
                 name,
                 shapes.size(),
                 static_cast<long long>(packed_query.size(0)),
                 max_abs,
                 mean_abs);
    std::fflush(stderr);

    MultiBatchPrecisionMetrics metrics;
    metrics.max_abs = max_abs;
    metrics.mean_abs = mean_abs;
    metrics.finite = torch::isfinite(fused_output).all().item<bool>();
    return metrics;
  }

  void run_multi_batch_fused_case(
      const char* name,
      const std::vector<MTGRAttentionTestShape>& shapes) {
    const auto metrics = measure_multi_batch_fused_case(name, shapes);
    ASSERT_TRUE(metrics.finite);
    EXPECT_LT(metrics.max_abs, kMaxAbs);
    EXPECT_LT(metrics.mean_abs, kMeanAbs);
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

TEST_F(MTGRAttentionE2EPrecisionTest,
       FusedNoMatchMultiBatchAlignsWithOneStage) {
  std::vector<MTGRAttentionTestShape> shapes(4);
  for (auto& shape : shapes) {
    shape.heads = 8;
    shape.kv_heads = 8;
    shape.head_dim = 128;
    shape.context = 8;
    shape.matched_prefix = 0;
  }
  shapes[0].history = 513;
  shapes[0].realtime = 161;
  shapes[0].target = 321;
  shapes[1].history = 1027;
  shapes[1].realtime = 193;
  shapes[1].target = 287;
  shapes[2].history = 769;
  shapes[2].realtime = 257;
  shapes[2].target = 415;
  shapes[3].history = 1537;
  shapes[3].realtime = 129;
  shapes[3].target = 511;
  run_multi_batch_fused_case("fused_no_match_multi_batch", shapes);
}

TEST_F(MTGRAttentionE2EPrecisionTest,
       FusedSegmentedNoMatchFiveSegmentsAlignsWithReference) {
  const int64_t heads = 4;
  const int64_t head_dim = 64;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
  const std::vector<std::vector<int64_t>> segment_lens = {
      {257, 7, 109, 17, 193},
      {129, 11, 211, 19, 151},
  };
  const std::vector<int64_t> segment_rules = {
      0,  // causal
      1,  // full
      1,  // full
      0,  // causal
      2,  // target diagonal
  };
  const int64_t total_tokens = std::accumulate(
      segment_lens.begin(),
      segment_lens.end(),
      0LL,
      [](int64_t acc, const auto& lens) {
        return acc + std::accumulate(lens.begin(), lens.end(), 0LL);
      });
  auto opts = torch::TensorOptions().dtype(torch::kFloat16).device(device_);
  auto query = torch::randn({total_tokens, heads, head_dim}, opts) * 0.05;
  auto key = torch::randn({total_tokens, heads, head_dim}, opts) * 0.05;
  auto value = torch::randn({total_tokens, heads, head_dim}, opts) * 0.05;

  MTGRAttentionTestMetrics metrics;
  auto fused = run_fused_segmented_no_match_batched_for_test(
      query, key, value, segment_lens, segment_rules, scale, &metrics);
  auto reference = run_ragged_segment_reference(
      query, key, value, segment_lens, segment_rules, scale);
  CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

  auto fused_f32 = fused.to(torch::kFloat32);
  auto diff = (reference - fused_f32).abs();
  const double max_abs = diff.max().item<double>();
  const double mean_abs = diff.mean().item<double>();
  std::fprintf(stderr,
               "[MTGR][CUDA][Precision][FusedSegmentedNoMatchFiveSegments] "
               "rows=%lld max_abs=%.6e mean_abs=%.6e device_ms=%.6f\n",
               static_cast<long long>(total_tokens),
               max_abs,
               mean_abs,
               metrics.device_total_ms);
  std::fflush(stderr);

  ASSERT_TRUE(torch::isfinite(fused).all().item<bool>());
  EXPECT_LT(max_abs, kMaxAbs);
  EXPECT_LT(mean_abs, kMeanAbs);
}

TEST_F(MTGRAttentionE2EPrecisionTest,
       RaggedSegmentAttentionFiveSegmentsAlignsWithReference) {
  const int64_t heads = 4;
  const int64_t head_dim = 64;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
  const std::vector<std::vector<int64_t>> segment_lens = {
      {193, 17, 109, 7, 257},
      {151, 19, 211, 11, 129},
  };
  const std::vector<int64_t> segment_rules = {
      0,  // causal
      2,  // diagonal
      1,  // full
      0,  // causal
      2,  // diagonal
  };
  const int64_t total_tokens = std::accumulate(
      segment_lens.begin(),
      segment_lens.end(),
      0LL,
      [](int64_t acc, const auto& lens) {
        return acc + std::accumulate(lens.begin(), lens.end(), 0LL);
      });
  auto opts = torch::TensorOptions().dtype(torch::kFloat16).device(device_);
  auto query = torch::randn({total_tokens, heads, head_dim}, opts) * 0.05;
  auto key = torch::randn({total_tokens, heads, head_dim}, opts) * 0.05;
  auto value = torch::randn({total_tokens, heads, head_dim}, opts) * 0.05;

  MTGRAttentionTestMetrics metrics;
  auto fused = run_ragged_segment_attention_batched_for_test(
      query, key, value, segment_lens, segment_rules, scale, &metrics);
  auto reference = run_ragged_segment_reference(
      query, key, value, segment_lens, segment_rules, scale);
  CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

  auto fused_f32 = fused.to(torch::kFloat32);
  auto diff = (reference - fused_f32).abs();
  const double max_abs = diff.max().item<double>();
  const double mean_abs = diff.mean().item<double>();
  std::fprintf(stderr,
               "[MTGR][CUDA][Precision][RaggedSegmentAttentionFiveSegments] "
               "rows=%lld max_abs=%.6e mean_abs=%.6e device_ms=%.6f\n",
               static_cast<long long>(total_tokens),
               max_abs,
               mean_abs,
               metrics.device_total_ms);
  std::fflush(stderr);

  ASSERT_TRUE(torch::isfinite(fused).all().item<bool>());
  EXPECT_LT(max_abs, kMaxAbs);
  EXPECT_LT(mean_abs, kMeanAbs);
}

TEST_F(MTGRAttentionE2EPrecisionTest,
       FusedPartialMultiBatchAlignsWithOneStage) {
  std::vector<MTGRAttentionTestShape> shapes(4);
  for (auto& shape : shapes) {
    shape.heads = 8;
    shape.kv_heads = 8;
    shape.head_dim = 128;
    shape.context = 8;
  }
  shapes[0].history = 513;
  shapes[0].realtime = 161;
  shapes[0].target = 321;
  shapes[0].matched_prefix = shapes[0].history + shapes[0].context + 129;
  shapes[1].history = 1027;
  shapes[1].realtime = 193;
  shapes[1].target = 287;
  shapes[1].matched_prefix = shapes[1].history + shapes[1].context + 151;
  shapes[2].history = 769;
  shapes[2].realtime = 257;
  shapes[2].target = 415;
  shapes[2].matched_prefix = shapes[2].history + shapes[2].context + 201;
  shapes[3].history = 1537;
  shapes[3].realtime = 129;
  shapes[3].target = 511;
  shapes[3].matched_prefix = shapes[3].history + shapes[3].context + 97;
  run_multi_batch_fused_case("fused_partial_multi_batch", shapes);
}

TEST_F(MTGRAttentionE2EPrecisionTest, FusedNoMatchMultiBatchOddLengthRandom) {
  const int groups_per_batch_size = std::max(
      1, env_int("XLLM_MTGR_ATTENTION_MULTI_BATCH_ODD_RANDOM_GROUPS", 8));
  const int seed =
      env_int("XLLM_MTGR_ATTENTION_MULTI_BATCH_ODD_RANDOM_SEED", 20260503);
  std::mt19937_64 rng(static_cast<uint64_t>(seed));
  const std::vector<int64_t> heads_all = {4, 8, 12};
  const std::vector<int64_t> head_dims_all = {64, 128};
  std::uniform_int_distribution<int> heads_dist(
      0, static_cast<int>(heads_all.size() - 1));
  std::uniform_int_distribution<int> head_dim_dist(
      0, static_cast<int>(head_dims_all.size() - 1));

  std::fprintf(stderr,
               "[MTGR][CUDA][Precision][FusedNoMatchMultiBatchOddRandom] "
               "groups_per_batch_size=%d seed=%d\n",
               groups_per_batch_size,
               seed);

  int completed = 0;
  double total_mean_abs = 0.0;
  double worst_max_abs = 0.0;
  double worst_mean_abs = 0.0;
  for (int batch_size : {2, 3, 4}) {
    std::unordered_set<std::string> seen;
    for (int group_id = 1; group_id <= groups_per_batch_size; ++group_id) {
      std::vector<MTGRAttentionTestShape> shapes;
      bool accepted = false;
      for (int attempt = 0; attempt < 256 && !accepted; ++attempt) {
        shapes.clear();
        const int64_t heads = heads_all[heads_dist(rng)];
        const int64_t head_dim = head_dims_all[head_dim_dist(rng)];
        std::unordered_set<std::string> local_seen;
        for (int i = 0; i < batch_size; ++i) {
          MTGRAttentionTestShape shape =
              sample_odd_no_match_shape(&rng, heads, head_dim);
          const std::string local_key = std::to_string(shape.history) + "_" +
                                        std::to_string(shape.realtime) + "_" +
                                        std::to_string(shape.target);
          if (!local_seen.insert(local_key).second) {
            shapes.clear();
            break;
          }
          shapes.push_back(shape);
        }
        if (static_cast<int>(shapes.size()) != batch_size) {
          continue;
        }
        const std::string group_key = std::to_string(heads) + "_" +
                                      std::to_string(head_dim) + "_" +
                                      describe_multi_batch_shapes(shapes);
        accepted = seen.insert(group_key).second;
      }
      ASSERT_EQ(static_cast<int>(shapes.size()), batch_size)
          << "failed to build unique odd-length multi-batch group";

      const auto metrics = measure_multi_batch_fused_case(
          "fused_no_match_multi_batch_odd_random", shapes);
      const auto shape_desc = describe_multi_batch_shapes(shapes);
      std::fprintf(stderr,
                   "[MTGR][CUDA][Precision][FusedNoMatchMultiBatchOddRandom] "
                   "batch=%d group=%d/%d heads=%lld head_dim=%lld shapes=%s "
                   "max_abs=%.6e mean_abs=%.6e finite=%s\n",
                   batch_size,
                   group_id,
                   groups_per_batch_size,
                   static_cast<long long>(shapes.front().heads),
                   static_cast<long long>(shapes.front().head_dim),
                   shape_desc.c_str(),
                   metrics.max_abs,
                   metrics.mean_abs,
                   metrics.finite ? "true" : "false");
      std::fflush(stderr);

      ASSERT_TRUE(metrics.finite) << shape_desc;
      EXPECT_LT(metrics.max_abs, kMaxAbs) << shape_desc;
      EXPECT_LT(metrics.mean_abs, kMeanAbs) << shape_desc;
      ++completed;
      total_mean_abs += metrics.mean_abs;
      worst_max_abs = std::max(worst_max_abs, metrics.max_abs);
      worst_mean_abs = std::max(worst_mean_abs, metrics.mean_abs);
    }
  }

  const double avg_mean_abs =
      completed > 0 ? total_mean_abs / static_cast<double>(completed) : 0.0;
  std::fprintf(stderr,
               "[MTGR][CUDA][Precision][FusedNoMatchMultiBatchOddRandom] "
               "completed=%d worst_max_abs=%.6e worst_mean_abs=%.6e "
               "avg_mean_abs=%.6e\n",
               completed,
               worst_max_abs,
               worst_mean_abs,
               avg_mean_abs);
  std::fflush(stderr);
}

TEST_F(MTGRAttentionE2EPrecisionTest, FusedPartialMultiBatchOddLengthRandom) {
  const int groups_per_batch_size = std::max(
      1,
      env_int("XLLM_MTGR_ATTENTION_MULTI_BATCH_PARTIAL_ODD_RANDOM_GROUPS", 8));
  const int seed = env_int(
      "XLLM_MTGR_ATTENTION_MULTI_BATCH_PARTIAL_ODD_RANDOM_SEED", 20260504);
  std::mt19937_64 rng(static_cast<uint64_t>(seed));
  const std::vector<int64_t> heads_all = {4, 8, 12};
  const std::vector<int64_t> head_dims_all = {64, 128};
  std::uniform_int_distribution<int> heads_dist(
      0, static_cast<int>(heads_all.size() - 1));
  std::uniform_int_distribution<int> head_dim_dist(
      0, static_cast<int>(head_dims_all.size() - 1));

  std::fprintf(stderr,
               "[MTGR][CUDA][Precision][FusedPartialMultiBatchOddRandom] "
               "groups_per_batch_size=%d seed=%d\n",
               groups_per_batch_size,
               seed);

  int completed = 0;
  double total_mean_abs = 0.0;
  double worst_max_abs = 0.0;
  double worst_mean_abs = 0.0;
  for (int batch_size : {2, 3, 4}) {
    std::unordered_set<std::string> seen;
    for (int group_id = 1; group_id <= groups_per_batch_size; ++group_id) {
      std::vector<MTGRAttentionTestShape> shapes;
      bool accepted = false;
      for (int attempt = 0; attempt < 256 && !accepted; ++attempt) {
        shapes.clear();
        const int64_t heads = heads_all[heads_dist(rng)];
        const int64_t head_dim = head_dims_all[head_dim_dist(rng)];
        std::unordered_set<std::string> local_seen;
        for (int i = 0; i < batch_size; ++i) {
          MTGRAttentionTestShape shape =
              sample_odd_no_match_shape(&rng, heads, head_dim);
          shape.matched_prefix =
              shape.history + shape.context + (shape.realtime * 4) / 5;
          const std::string local_key = std::to_string(shape.history) + "_" +
                                        std::to_string(shape.realtime) + "_" +
                                        std::to_string(shape.target) + "_" +
                                        std::to_string(shape.matched_prefix);
          if (!local_seen.insert(local_key).second) {
            shapes.clear();
            break;
          }
          shapes.push_back(shape);
        }
        if (static_cast<int>(shapes.size()) != batch_size) {
          continue;
        }
        const std::string group_key = std::to_string(heads) + "_" +
                                      std::to_string(head_dim) + "_" +
                                      describe_multi_batch_shapes(shapes);
        accepted = seen.insert(group_key).second;
      }
      ASSERT_EQ(static_cast<int>(shapes.size()), batch_size)
          << "failed to build unique odd-length partial multi-batch group";

      const auto metrics = measure_multi_batch_fused_case(
          "fused_partial_multi_batch_odd_random", shapes);
      const auto shape_desc = describe_multi_batch_shapes(shapes);
      std::fprintf(stderr,
                   "[MTGR][CUDA][Precision][FusedPartialMultiBatchOddRandom] "
                   "batch=%d group=%d/%d heads=%lld head_dim=%lld shapes=%s "
                   "max_abs=%.6e mean_abs=%.6e finite=%s\n",
                   batch_size,
                   group_id,
                   groups_per_batch_size,
                   static_cast<long long>(shapes.front().heads),
                   static_cast<long long>(shapes.front().head_dim),
                   shape_desc.c_str(),
                   metrics.max_abs,
                   metrics.mean_abs,
                   metrics.finite ? "true" : "false");
      std::fflush(stderr);

      ASSERT_TRUE(metrics.finite) << shape_desc;
      EXPECT_LT(metrics.max_abs, kMaxAbs) << shape_desc;
      EXPECT_LT(metrics.mean_abs, kMeanAbs) << shape_desc;
      ++completed;
      total_mean_abs += metrics.mean_abs;
      worst_max_abs = std::max(worst_max_abs, metrics.max_abs);
      worst_mean_abs = std::max(worst_mean_abs, metrics.mean_abs);
    }
  }

  const double avg_mean_abs =
      completed > 0 ? total_mean_abs / static_cast<double>(completed) : 0.0;
  std::fprintf(stderr,
               "[MTGR][CUDA][Precision][FusedPartialMultiBatchOddRandom] "
               "completed=%d worst_max_abs=%.6e worst_mean_abs=%.6e "
               "avg_mean_abs=%.6e\n",
               completed,
               worst_max_abs,
               worst_mean_abs,
               avg_mean_abs);
  std::fflush(stderr);
}

}  // namespace xllm::kernel::cuda::test
