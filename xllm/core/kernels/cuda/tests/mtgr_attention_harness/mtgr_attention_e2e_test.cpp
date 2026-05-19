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

#include "mtgr_attention_contract.h"

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/cuda.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "core/common/global_flags.h"
#include "core/util/mtgr_nvtx.h"
#include "layers/cuda/mtgr_attention.h"

namespace xllm::kernel::cuda::test::mtgr_attention_harness {
namespace {

constexpr int64_t kDefaultBlockSize = 128;

int env_int(const char* name, int default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return default_value;
  }
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0') {
    return default_value;
  }
  return static_cast<int>(parsed);
}

std::string env_path(const char* name, const char* default_name) {
  const char* value = std::getenv(name);
  if (value != nullptr && *value != '\0') {
    return std::string(value);
  }
  return std::string(
             "/export/home/wangyifan/Code/xllm/xllm/core/kernels/cuda/tests/"
             "mtgr_attention_harness/") +
         default_name;
}

torch::Tensor run_base(IMTGRAttentionBackend* backend,
                       MTGRAttentionCaseData* data) {
  CHECK(backend != nullptr);
  CHECK(data != nullptr);
  auto& input = data->full_input;
  return std::get<0>(backend->forward(input.metadata,
                                      input.query,
                                      input.key,
                                      input.value,
                                      input.kv_cache));
}

torch::Tensor run_hopper(IMTGRAttentionBackend* backend,
                         MTGRAttentionCaseData* data) {
  CHECK(backend != nullptr);
  CHECK(data != nullptr);
  auto& input = data->live_input;
  return std::get<0>(backend->forward(input.metadata,
                                      input.query,
                                      input.key,
                                      input.value,
                                      input.kv_cache));
}

torch::Tensor live_slice_from_base(const torch::Tensor& base_output,
                                   const MTGRAttentionHarnessMetadata& metadata) {
  return base_output.narrow(0, metadata.matched_prefix, metadata.live_len())
      .contiguous();
}

void expect_product_close(const char* tag,
                          const torch::Tensor& got_flat,
                          const torch::Tensor& expected_snd,
                          int64_t heads,
                          int64_t head_dim) {
  constexpr double kMaxAbs = 2.0e-1;
  constexpr double kMeanAbs = 6.0e-3;
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

MTGRAttentionTestShape make_product_shape(bool partial_match) {
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

xllm::layer::MTGRAttentionImpl make_attention(int64_t heads,
                                              int64_t head_dim,
                                              double scale,
                                              int64_t kv_heads) {
  return xllm::layer::MTGRAttentionImpl(heads,
                                        head_dim,
                                        static_cast<float>(scale),
                                        kv_heads);
}

void run_warmup(const std::vector<MTGRAttentionHarnessMetadata>& metadata_cases,
                const torch::Device& device,
                int warmup) {
  for (int i = 0; i < warmup; ++i) {
    for (const auto& metadata : metadata_cases) {
      auto data = build_case_data(metadata, device);
      auto base = make_full_flashinfer_base_backend(metadata);
      auto hopper = make_hopper_unified_backend(metadata);
      CHECK(run_base(base.get(), &data).defined());
      CHECK(run_hopper(hopper.get(), &data).defined());
    }
  }
  CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

}  // namespace

class MTGRAttentionHarnessTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA not available";
    }
    cudaDeviceProp prop{};
    ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
    if (prop.major < 9) {
      GTEST_SKIP() << "Hopper unified kernel requires SM90+";
    }
    FLAGS_block_size = kDefaultBlockSize;
    FLAGS_flashinfer_workspace_buffer_size =
        std::max<int64_t>(FLAGS_flashinfer_workspace_buffer_size,
                          64 * 1024 * 1024);
    torch::manual_seed(20260518);
    torch::cuda::manual_seed_all(20260518);
    device_ = torch::Device(torch::kCUDA, 0);
  }

  torch::Device device_ = torch::Device(torch::kCPU);
};

TEST_F(MTGRAttentionHarnessTest, Precision) {
  torch::NoGradGuard no_grad;
  const int pair_count =
      std::max(1, env_int("XLLM_MTGR_HARNESS_PRECISION_PAIRS", 8));
  const int seed = env_int("XLLM_MTGR_HARNESS_SEED", 20260501);
  const double max_abs_threshold =
      static_cast<double>(env_int("XLLM_MTGR_HARNESS_MAX_ABS_MILLI", 30)) /
      1000.0;
  auto metadata_cases =
      generate_odd_length_metadata_pairs(pair_count, static_cast<uint64_t>(seed));
  for (auto& metadata : metadata_cases) {
    metadata.block_size = kDefaultBlockSize;
  }

  double worst_max_abs = 0.0;
  double worst_mean_abs = 0.0;
  for (const auto& metadata : metadata_cases) {
    auto data = build_case_data(metadata, device_);
    auto base = make_full_flashinfer_base_backend(metadata);
    auto hopper = make_hopper_unified_backend(metadata);
    auto base_out = live_slice_from_base(run_base(base.get(), &data), metadata);
    auto hopper_out = run_hopper(hopper.get(), &data);
    auto diff = compare_outputs(base_out, hopper_out);
    EXPECT_TRUE(diff.all_finite) << "mode=" << metadata.mode_name()
                                 << " pair_id=" << metadata.pair_id;
    EXPECT_LE(diff.max_abs, max_abs_threshold)
        << "mode=" << metadata.mode_name() << " pair_id=" << metadata.pair_id;
    worst_max_abs = std::max(worst_max_abs, diff.max_abs);
    worst_mean_abs = std::max(worst_mean_abs, diff.mean_abs);
  }
  std::fprintf(stderr,
               "[MTGR][Harness][Precision] pairs=%d cases=%zu "
               "worst_max_abs=%.6e worst_mean_abs=%.6e\n",
               pair_count,
               metadata_cases.size(),
               worst_max_abs,
               worst_mean_abs);
}

TEST_F(MTGRAttentionHarnessTest, PerfNvtxCsv) {
  torch::NoGradGuard no_grad;
  FLAGS_mtgr_nvtx_level = std::max(FLAGS_mtgr_nvtx_level, 2);
  const int pair_count =
      std::max(1, env_int("XLLM_MTGR_HARNESS_PERF_PAIRS", 1000));
  const int warmup = std::max(0, env_int("XLLM_MTGR_HARNESS_WARMUP", 1));
  const int repeat = std::max(1, env_int("XLLM_MTGR_HARNESS_REPEAT", 1));
  const int seed = env_int("XLLM_MTGR_HARNESS_SEED", 20260501);
  const std::string labels_path =
      env_path("XLLM_MTGR_HARNESS_LABELS", "mtgr_attention_harness_labels.csv");

  std::ofstream labels(labels_path, std::ios::out | std::ios::trunc);
  CHECK(labels.is_open()) << "failed to open labels path: " << labels_path;
  write_perf_label_header(labels);

  auto metadata_cases =
      generate_odd_length_metadata_pairs(pair_count, static_cast<uint64_t>(seed));
  for (auto& metadata : metadata_cases) {
    metadata.block_size = kDefaultBlockSize;
  }

  run_warmup(metadata_cases, device_, warmup);

  int64_t idx = 0;
  for (const auto& metadata : metadata_cases) {
    for (int r = 0; r < repeat; ++r) {
      auto data = build_case_data(metadata, device_);
      auto base = make_full_flashinfer_base_backend(metadata);
      auto hopper = make_hopper_unified_backend(metadata);

      CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
      ++idx;
      write_perf_label_row(labels, idx, base->name(), metadata, r + 1);
      labels.flush();
      {
        const auto root = base->nvtx_root_name(metadata);
        xllm::MtgrNvtxRange range(1, root.c_str());
        CHECK(run_base(base.get(), &data).defined());
      }
      CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

      CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
      ++idx;
      write_perf_label_row(labels, idx, hopper->name(), metadata, r + 1);
      labels.flush();
      {
        const auto root = hopper->nvtx_root_name(metadata);
        xllm::MtgrNvtxRange range(1, root.c_str());
        CHECK(run_hopper(hopper.get(), &data).defined());
      }
      CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }

    if ((metadata.pair_id % 50) == 0 && metadata.is_partial_match()) {
      std::fprintf(stderr,
                   "[MTGR][Harness][PerfNvtxCsv] completed_pairs=%lld/%d "
                   "labels=%lld\n",
                   static_cast<long long>(metadata.pair_id),
                   pair_count,
                   static_cast<long long>(idx));
      std::fflush(stderr);
    }
  }
  std::fprintf(stderr,
               "[MTGR][Harness][PerfNvtxCsv] done pairs=%d cases=%zu "
               "labels=%lld labels_path=%s\n",
               pair_count,
               metadata_cases.size(),
               static_cast<long long>(idx),
               labels_path.c_str());
}

TEST_F(MTGRAttentionHarnessTest, ProductNoMatchForwardUsesSegmentedApi) {
  torch::NoGradGuard no_grad_guard;
  const auto shape = make_product_shape(/*partial_match=*/false);
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

  auto metadata = make_mtgr_attention_metadata(shape, torch::kCUDA, kDefaultBlockSize);
  auto kv_cache =
      make_mtgr_kv_cache(shape, torch::kCUDA, torch::kBFloat16, kDefaultBlockSize);
  auto query_flat =
      full_query.select(0, 0).reshape({shape.local_len(), -1}).contiguous();
  auto key_flat =
      full_key.select(0, 0).reshape({shape.local_len(), -1}).contiguous();
  auto value_flat =
      full_value.select(0, 0).reshape({shape.local_len(), -1}).contiguous();

  auto impl = make_attention(shape.heads, shape.head_dim, scale, shape.kv_heads);
  auto [got, lse] =
      impl.forward(metadata, query_flat, key_flat, value_flat, kv_cache);

  EXPECT_FALSE(lse.has_value());
  auto expected = run_mtgr_torch_mask_attention_reference(
      full_query, full_key, full_value, shape, scale);
  expect_product_close("no_match", got, expected, shape.heads, shape.head_dim);
}

TEST_F(MTGRAttentionHarnessTest, ProductPartialRealTimeForwardUsesSegmentedApi) {
  torch::NoGradGuard no_grad_guard;
  const auto shape = make_product_shape(/*partial_match=*/true);
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

  auto metadata = make_mtgr_attention_metadata(shape, torch::kCUDA, kDefaultBlockSize);
  auto kv_cache =
      make_mtgr_kv_cache(shape, torch::kCUDA, torch::kBFloat16, kDefaultBlockSize);
  prefill_mtgr_matched_prefix_cache(
      full_key, full_value, shape, kDefaultBlockSize, kv_cache);

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

  auto impl = make_attention(shape.heads, shape.head_dim, scale, shape.kv_heads);
  auto [got, lse] =
      impl.forward(metadata, query_flat, key_flat, value_flat, kv_cache);

  EXPECT_FALSE(lse.has_value());
  auto expected = run_mtgr_torch_mask_attention_reference(
      full_query, full_key, full_value, shape, scale);
  expect_product_close(
      "partial_real_time", got, expected, shape.heads, shape.head_dim);
}

TEST_F(MTGRAttentionHarnessTest, ProductMixedBatchForwardUsesRequestLevelMode) {
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
      shapes, full_keys, full_values, device, torch::kBFloat16, kDefaultBlockSize);
  auto metadata = mixed_setup.metadata;
  metadata.mtgr_match_mode = xllm::layer::MTGRMatchMode::kMixed;

  auto query_snd = torch::cat(packed_queries, 0).contiguous();
  auto key_snd = torch::cat(packed_keys, 0).contiguous();
  auto value_snd = torch::cat(packed_values, 0).contiguous();
  auto query_flat = query_snd.reshape({query_snd.size(0), -1}).contiguous();
  auto key_flat = key_snd.reshape({key_snd.size(0), -1}).contiguous();
  auto value_flat = value_snd.reshape({value_snd.size(0), -1}).contiguous();

  auto impl = make_attention(kHeads, kHeadDim, scale, kHeads);
  auto [got, lse] = impl.forward(
      metadata, query_flat, key_flat, value_flat, mixed_setup.kv_cache);

  EXPECT_FALSE(lse.has_value());
  auto expected = torch::cat(expected_chunks, 0).contiguous();
  expect_product_close("mixed_request_level", got, expected, kHeads, kHeadDim);
}

TEST_F(MTGRAttentionHarnessTest, ProductPurePartialMatchesMixedMode) {
  torch::NoGradGuard no_grad_guard;
  constexpr int64_t kBatchSize = 100;
  constexpr int64_t kHeads = 8;
  constexpr int64_t kHeadDim = 128;
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
      shapes, full_keys, full_values, device, torch::kBFloat16, kDefaultBlockSize);
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

  auto impl = make_attention(kHeads, kHeadDim, scale, kHeads);
  auto [partial_out, partial_lse] = impl.forward(partial_metadata,
                                                query_flat,
                                                key_flat,
                                                value_flat,
                                                partial_setup.kv_cache);
  auto [mixed_out, mixed_lse] = impl.forward(mixed_metadata,
                                            query_flat,
                                            key_flat,
                                            value_flat,
                                            partial_setup.kv_cache);
  EXPECT_FALSE(partial_lse.has_value());
  EXPECT_FALSE(mixed_lse.has_value());

  auto diff =
      (partial_out.to(torch::kFloat32) - mixed_out.to(torch::kFloat32)).abs();
  const double max_abs = diff.max().item<double>();
  const double mean_abs = diff.mean().item<double>();
  std::fprintf(stderr,
               "[MTGR][CUDA][ProductAPI][PartialVsMixedMode] "
               "requests=%lld diff_max_abs=%.6e diff_mean_abs=%.6e\n",
               static_cast<long long>(kBatchSize),
               max_abs,
               mean_abs);
  std::fflush(stderr);

  EXPECT_LT(max_abs, 1.0e-4);
  EXPECT_LT(mean_abs, 1.0e-6);
}

}  // namespace xllm::kernel::cuda::test::mtgr_attention_harness
