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

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <torch/cuda.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "common/global_flags.h"
#include "mtgr_attenion_test.h"
#include "mtgr_attention_product_test.h"
#include "mtgr_attention_torch_reference_test.h"

namespace xllm::kernel::cuda::test {
namespace {

constexpr int64_t kBlockSize = 128;
constexpr double kMaxAbs = 2.0e-1;
constexpr double kMeanAbs = 6.0e-3;

void expect_close(const char* tag,
                  const torch::Tensor& got_flat,
                  const torch::Tensor& expected_snd,
                  int64_t heads,
                  int64_t head_dim) {
  auto got = got_flat.view({expected_snd.size(0), heads, head_dim})
                 .to(torch::kCPU)
                 .to(torch::kFloat32);
  auto expected = expected_snd.to(torch::kCPU).to(torch::kFloat32);
  auto diff = (got - expected).abs();
  const double max_abs = diff.max().item<double>();
  const double mean_abs = diff.mean().item<double>();
  const bool finite = torch::isfinite(got).all().item<bool>();

  std::fprintf(stderr,
               "[MTGR][CUDA][ProductAPI][%s] max_abs=%.6e mean_abs=%.6e "
               "finite=%s\n",
               tag,
               max_abs,
               mean_abs,
               finite ? "true" : "false");
  std::fflush(stderr);

  EXPECT_TRUE(finite);
  EXPECT_LT(max_abs, kMaxAbs);
  EXPECT_LT(mean_abs, kMeanAbs);
}

MTGRAttentionTestShape make_shape(bool partial_match) {
  MTGRAttentionTestShape shape;
  shape.heads = 8;
  shape.kv_heads = 8;
  shape.head_dim = 128;
  shape.history = 257;
  shape.context = 9;
  shape.realtime = 73;
  shape.target = 129;
  shape.matched_prefix = partial_match ? shape.history + shape.context + 41 : 0;
  return shape;
}

int64_t odd_in_range(std::mt19937_64* rng, int64_t lo, int64_t hi) {
  if ((lo & 1) == 0) {
    ++lo;
  }
  if ((hi & 1) == 0) {
    --hi;
  }
  std::uniform_int_distribution<int64_t> dist(0, (hi - lo) / 2);
  return lo + 2 * dist(*rng);
}

double time_product_forward_ms(MTGRAttentionTestImpl* impl,
                               const xllm::layer::AttentionMetadata& metadata,
                               torch::Tensor& query_flat,
                               torch::Tensor& key_flat,
                               torch::Tensor& value_flat,
                               xllm::KVCache& kv_cache,
                               torch::Tensor* output) {
  cudaEvent_t start = nullptr;
  cudaEvent_t end = nullptr;
  CHECK_EQ(cudaEventCreate(&start), cudaSuccess);
  CHECK_EQ(cudaEventCreate(&end), cudaSuccess);
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  CHECK_EQ(cudaEventRecord(start, stream), cudaSuccess);
  auto [got, lse] =
      impl->forward(metadata, query_flat, key_flat, value_flat, kv_cache);
  (void)lse;
  CHECK_EQ(cudaEventRecord(end, stream), cudaSuccess);
  CHECK_EQ(cudaEventSynchronize(end), cudaSuccess);
  float ms = 0.0f;
  CHECK_EQ(cudaEventElapsedTime(&ms, start, end), cudaSuccess);
  CHECK_EQ(cudaEventDestroy(start), cudaSuccess);
  CHECK_EQ(cudaEventDestroy(end), cudaSuccess);
  if (output != nullptr) {
    *output = got;
  }
  return static_cast<double>(ms);
}

class MTGRAttentionProductE2EPrecisionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA is not available";
    }
    cudaDeviceProp prop{};
    ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
    if (prop.major < 9) {
      GTEST_SKIP() << "Hopper unified research kernel requires SM90+";
    }
    FLAGS_block_size = kBlockSize;
  }
};

TEST_F(MTGRAttentionProductE2EPrecisionTest, NoMatchForwardUsesNpuStyleApi) {
  const auto shape = make_shape(/*partial_match=*/false);
  const double scale = 1.0 / std::sqrt(static_cast<double>(shape.head_dim));
  auto opts =
      torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
  torch::manual_seed(20260513);
  auto full_query =
      torch::randn({1, shape.total_len(), shape.heads, shape.head_dim}, opts) *
      0.05;
  auto full_key =
      torch::randn({1, shape.total_len(), shape.kv_heads, shape.head_dim},
                   opts) *
      0.05;
  auto full_value =
      torch::randn({1, shape.total_len(), shape.kv_heads, shape.head_dim},
                   opts) *
      0.05;

  auto metadata = make_mtgr_attention_metadata(shape, torch::kCUDA, kBlockSize);
  auto kv_cache =
      make_mtgr_kv_cache(shape, torch::kCUDA, torch::kBFloat16, kBlockSize);
  auto query_flat =
      full_query.select(0, 0).reshape({shape.local_len(), -1}).contiguous();
  auto key_flat =
      full_key.select(0, 0).reshape({shape.local_len(), -1}).contiguous();
  auto value_flat =
      full_value.select(0, 0).reshape({shape.local_len(), -1}).contiguous();

  MTGRAttentionTestImpl impl(
      shape.heads, shape.head_dim, static_cast<float>(scale), shape.kv_heads);
  auto [got, lse] =
      impl.forward(metadata, query_flat, key_flat, value_flat, kv_cache);

  EXPECT_FALSE(lse.has_value());
  auto expected = run_mtgr_torch_mask_attention_reference(
      full_query, full_key, full_value, shape, scale);
  expect_close("no_match", got, expected, shape.heads, shape.head_dim);
}

TEST_F(MTGRAttentionProductE2EPrecisionTest,
       PartialRealTimeForwardUsesNpuStyleApi) {
  const auto shape = make_shape(/*partial_match=*/true);
  const double scale = 1.0 / std::sqrt(static_cast<double>(shape.head_dim));
  auto opts =
      torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
  torch::manual_seed(20260514);
  auto full_query =
      torch::randn({1, shape.total_len(), shape.heads, shape.head_dim}, opts) *
      0.05;
  auto full_key =
      torch::randn({1, shape.total_len(), shape.kv_heads, shape.head_dim},
                   opts) *
      0.05;
  auto full_value =
      torch::randn({1, shape.total_len(), shape.kv_heads, shape.head_dim},
                   opts) *
      0.05;

  auto metadata = make_mtgr_attention_metadata(shape, torch::kCUDA, kBlockSize);
  auto kv_cache =
      make_mtgr_kv_cache(shape, torch::kCUDA, torch::kBFloat16, kBlockSize);
  prefill_mtgr_matched_prefix_cache(
      full_key, full_value, shape, kBlockSize, kv_cache);

  auto live_query = full_query.select(0, 0)
                        .narrow(0, shape.matched_prefix, shape.local_len())
                        .contiguous();
  auto live_key = full_key.select(0, 0)
                      .narrow(0, shape.matched_prefix, shape.local_len())
                      .contiguous();
  auto live_value = full_value.select(0, 0)
                        .narrow(0, shape.matched_prefix, shape.local_len())
                        .contiguous();
  auto query_flat = live_query.reshape({shape.local_len(), -1}).contiguous();
  auto key_flat = live_key.reshape({shape.local_len(), -1}).contiguous();
  auto value_flat = live_value.reshape({shape.local_len(), -1}).contiguous();

  MTGRAttentionTestImpl impl(
      shape.heads, shape.head_dim, static_cast<float>(scale), shape.kv_heads);
  auto [got, lse] =
      impl.forward(metadata, query_flat, key_flat, value_flat, kv_cache);

  EXPECT_FALSE(lse.has_value());
  auto expected = run_mtgr_torch_mask_attention_reference(
      full_query, full_key, full_value, shape, scale);
  expect_close("partial_real_time", got, expected, shape.heads, shape.head_dim);
}

TEST_F(MTGRAttentionProductE2EPrecisionTest,
       MixedBatchForwardUsesRequestLevelMatchMode) {
  torch::NoGradGuard no_grad_guard;
  constexpr int64_t kBatchSize = 20;
  constexpr int64_t kHeads = 8;
  constexpr int64_t kHeadDim = 128;
  const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
  const auto device = torch::Device(torch::kCUDA, 0);
  auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(device);

  std::mt19937_64 rng(20260515);
  std::vector<MTGRAttentionTestShape> shapes;
  std::vector<torch::Tensor> full_keys;
  std::vector<torch::Tensor> full_values;
  std::vector<torch::Tensor> packed_queries;
  std::vector<torch::Tensor> packed_keys;
  std::vector<torch::Tensor> packed_values;
  std::vector<torch::Tensor> expected_chunks;
  shapes.reserve(kBatchSize);
  full_keys.reserve(kBatchSize);
  full_values.reserve(kBatchSize);
  packed_queries.reserve(kBatchSize);
  packed_keys.reserve(kBatchSize);
  packed_values.reserve(kBatchSize);
  expected_chunks.reserve(kBatchSize);

  torch::manual_seed(20260515);
  for (int64_t i = 0; i < kBatchSize; ++i) {
    MTGRAttentionTestShape shape;
    shape.heads = kHeads;
    shape.kv_heads = kHeads;
    shape.head_dim = kHeadDim;
    shape.history = odd_in_range(&rng, 129, 513);
    shape.context = odd_in_range(&rng, 5, 15);
    shape.realtime = odd_in_range(&rng, 65, 161);
    shape.target = odd_in_range(&rng, 65, 193);
    shape.matched_prefix = (i & 1) == 0
                               ? 0
                               : shape.history + shape.context +
                                     odd_in_range(&rng, 1, shape.realtime - 2);

    auto full_query =
        torch::randn({1, shape.total_len(), shape.heads, shape.head_dim},
                     opts) *
        0.05;
    auto full_key =
        torch::randn({1, shape.total_len(), shape.kv_heads, shape.head_dim},
                     opts) *
        0.05;
    auto full_value =
        torch::randn({1, shape.total_len(), shape.kv_heads, shape.head_dim},
                     opts) *
        0.05;

    packed_queries.push_back(
        full_query.select(0, 0)
            .narrow(0, shape.matched_prefix, shape.local_len())
            .contiguous());
    packed_keys.push_back(
        full_key.select(0, 0)
            .narrow(0, shape.matched_prefix, shape.local_len())
            .contiguous());
    packed_values.push_back(
        full_value.select(0, 0)
            .narrow(0, shape.matched_prefix, shape.local_len())
            .contiguous());
    expected_chunks.push_back(run_mtgr_torch_mask_attention_reference(
        full_query, full_key, full_value, shape, scale));
    full_keys.push_back(full_key);
    full_values.push_back(full_value);
    shapes.push_back(shape);
  }

  auto mixed_setup = make_mtgr_partial_batch_setup(
      shapes, full_keys, full_values, device, torch::kBFloat16, kBlockSize);
  auto metadata = mixed_setup.metadata;
  metadata.mtgr_match_mode = xllm::layer::MTGRMatchMode::kMixed;

  auto query_snd = torch::cat(packed_queries, 0).contiguous();
  auto key_snd = torch::cat(packed_keys, 0).contiguous();
  auto value_snd = torch::cat(packed_values, 0).contiguous();
  auto query_flat = query_snd.reshape({query_snd.size(0), -1}).contiguous();
  auto key_flat = key_snd.reshape({key_snd.size(0), -1}).contiguous();
  auto value_flat = value_snd.reshape({value_snd.size(0), -1}).contiguous();

  MTGRAttentionTestImpl impl(
      kHeads, kHeadDim, static_cast<float>(scale), kHeads);
  auto [got, lse] = impl.forward(
      metadata, query_flat, key_flat, value_flat, mixed_setup.kv_cache);

  EXPECT_FALSE(lse.has_value());
  auto expected = torch::cat(expected_chunks, 0).contiguous();
  int64_t out_start = 0;
  for (int64_t i = 0; i < kBatchSize; ++i) {
    auto chunk =
        got.narrow(0, out_start, shapes[static_cast<size_t>(i)].local_len())
            .to(torch::kFloat32);
    if (!torch::isfinite(chunk).all().item<bool>()) {
      auto finite_by_row =
          torch::isfinite(chunk.view({chunk.size(0), kHeads, kHeadDim}))
              .all({1, 2})
              .to(torch::kCPU);
      int64_t bad_row = -1;
      for (int64_t row = 0; row < finite_by_row.size(0); ++row) {
        if (!finite_by_row[row].item<bool>()) {
          bad_row = row;
          break;
        }
      }
      std::fprintf(stderr,
                   "[MTGR][CUDA][ProductAPI][mixed_request_level][bad] "
                   "request=%ld matched_prefix=%ld local_len=%ld total_len=%ld "
                   "bad_row=%ld\n",
                   i,
                   shapes[static_cast<size_t>(i)].matched_prefix,
                   shapes[static_cast<size_t>(i)].local_len(),
                   shapes[static_cast<size_t>(i)].total_len(),
                   bad_row);
      std::fflush(stderr);
      break;
    }
    out_start += shapes[static_cast<size_t>(i)].local_len();
  }
  expect_close("mixed_request_level", got, expected, kHeads, kHeadDim);
}

TEST_F(MTGRAttentionProductE2EPrecisionTest,
       PurePartialBatchPartialModeVsMixedModePerf) {
  torch::NoGradGuard no_grad_guard;
  constexpr int64_t kBatchSize = 100;
  constexpr int64_t kHeads = 8;
  constexpr int64_t kHeadDim = 128;
  const int warmup = std::max(0, env_int("XLLM_MTGR_MIX_PROBE_WARMUP", 5));
  const int repeat = std::max(1, env_int("XLLM_MTGR_MIX_PROBE_REPEAT", 20));
  const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
  const auto device = torch::Device(torch::kCUDA, 0);
  auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(device);

  std::mt19937_64 rng(20260514);
  std::vector<MTGRAttentionTestShape> shapes;
  std::vector<torch::Tensor> full_keys;
  std::vector<torch::Tensor> full_values;
  std::vector<torch::Tensor> packed_queries;
  std::vector<torch::Tensor> packed_keys;
  std::vector<torch::Tensor> packed_values;
  shapes.reserve(kBatchSize);
  full_keys.reserve(kBatchSize);
  full_values.reserve(kBatchSize);
  packed_queries.reserve(kBatchSize);
  packed_keys.reserve(kBatchSize);
  packed_values.reserve(kBatchSize);

  torch::manual_seed(20260514);
  for (int64_t i = 0; i < kBatchSize; ++i) {
    MTGRAttentionTestShape shape;
    shape.heads = kHeads;
    shape.kv_heads = kHeads;
    shape.head_dim = kHeadDim;
    shape.history = odd_in_range(&rng, 129, 513);
    shape.context = odd_in_range(&rng, 5, 15);
    shape.realtime = odd_in_range(&rng, 65, 161);
    shape.target = odd_in_range(&rng, 65, 193);
    shape.matched_prefix = shape.history + shape.context +
                           odd_in_range(&rng, 1, shape.realtime - 2);

    auto full_query =
        torch::randn({1, shape.total_len(), shape.heads, shape.head_dim},
                     opts) *
        0.05;
    auto full_key =
        torch::randn({1, shape.total_len(), shape.kv_heads, shape.head_dim},
                     opts) *
        0.05;
    auto full_value =
        torch::randn({1, shape.total_len(), shape.kv_heads, shape.head_dim},
                     opts) *
        0.05;

    packed_queries.push_back(
        full_query.select(0, 0)
            .narrow(0, shape.matched_prefix, shape.local_len())
            .contiguous());
    packed_keys.push_back(
        full_key.select(0, 0)
            .narrow(0, shape.matched_prefix, shape.local_len())
            .contiguous());
    packed_values.push_back(
        full_value.select(0, 0)
            .narrow(0, shape.matched_prefix, shape.local_len())
            .contiguous());
    full_keys.push_back(full_key);
    full_values.push_back(full_value);
    shapes.push_back(shape);
  }

  auto partial_setup = make_mtgr_partial_batch_setup(
      shapes, full_keys, full_values, device, torch::kBFloat16, kBlockSize);
  auto partial_metadata = partial_setup.metadata;
  auto mixed_metadata = partial_setup.metadata;
  partial_metadata.mtgr_match_mode = xllm::layer::MTGRMatchMode::kPartialOnly;
  mixed_metadata.mtgr_match_mode = xllm::layer::MTGRMatchMode::kMixed;

  auto query_snd = torch::cat(packed_queries, 0).contiguous();
  auto key_snd = torch::cat(packed_keys, 0).contiguous();
  auto value_snd = torch::cat(packed_values, 0).contiguous();
  auto query_flat = query_snd.reshape({query_snd.size(0), -1}).contiguous();
  auto key_flat = key_snd.reshape({key_snd.size(0), -1}).contiguous();
  auto value_flat = value_snd.reshape({value_snd.size(0), -1}).contiguous();

  MTGRAttentionTestImpl impl(
      kHeads, kHeadDim, static_cast<float>(scale), kHeads);
  torch::Tensor partial_out;
  torch::Tensor mixed_out;
  for (int i = 0; i < warmup; ++i) {
    (void)time_product_forward_ms(&impl,
                                  partial_metadata,
                                  query_flat,
                                  key_flat,
                                  value_flat,
                                  partial_setup.kv_cache,
                                  nullptr);
    (void)time_product_forward_ms(&impl,
                                  mixed_metadata,
                                  query_flat,
                                  key_flat,
                                  value_flat,
                                  partial_setup.kv_cache,
                                  nullptr);
  }

  double partial_total_ms = 0.0;
  double mixed_total_ms = 0.0;
  for (int i = 0; i < repeat; ++i) {
    partial_total_ms += time_product_forward_ms(&impl,
                                                partial_metadata,
                                                query_flat,
                                                key_flat,
                                                value_flat,
                                                partial_setup.kv_cache,
                                                &partial_out);
    mixed_total_ms += time_product_forward_ms(&impl,
                                              mixed_metadata,
                                              query_flat,
                                              key_flat,
                                              value_flat,
                                              partial_setup.kv_cache,
                                              &mixed_out);
  }
  const double partial_ms = partial_total_ms / static_cast<double>(repeat);
  const double mixed_ms = mixed_total_ms / static_cast<double>(repeat);
  auto diff =
      (partial_out.to(torch::kFloat32) - mixed_out.to(torch::kFloat32)).abs();
  const double max_abs = diff.max().item<double>();
  const double mean_abs = diff.mean().item<double>();
  std::fprintf(stderr,
               "[MTGR][CUDA][ProductAPI][PartialVsMixedModePerf] "
               "requests=%ld warmup=%d repeat=%d partial_ms=%.6f "
               "mixed_ms=%.6f mixed_over_partial=%.6f diff_max_abs=%.6e "
               "diff_mean_abs=%.6e\n",
               kBatchSize,
               warmup,
               repeat,
               partial_ms,
               mixed_ms,
               mixed_ms / partial_ms,
               max_abs,
               mean_abs);
  std::fflush(stderr);

  EXPECT_LT(max_abs, 1.0e-4);
  EXPECT_LT(mean_abs, 1.0e-6);
}

}  // namespace
}  // namespace xllm::kernel::cuda::test
