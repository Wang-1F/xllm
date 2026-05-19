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

int64_t cacheable_end(const MTGRAttentionTestShape& shape) {
  return shape.history + shape.context + shape.realtime;
}

xllm::KVCache clone_kv_cache(const xllm::KVCache& kv_cache) {
  return xllm::KVCache(kv_cache.get_k_cache().clone(),
                       kv_cache.get_v_cache().clone());
}

xllm::KVCache make_sentinel_kv_cache(const MTGRAttentionTestShape& shape,
                                     const torch::Device& device,
                                     int64_t block_size,
                                     double sentinel) {
  auto kv_cache =
      make_mtgr_kv_cache(shape, device, torch::kBFloat16, block_size);
  kv_cache.get_k_cache().fill_(sentinel);
  kv_cache.get_v_cache().fill_(sentinel);
  return kv_cache;
}

torch::Tensor gather_cache_tokens(
    const torch::Tensor& cache,
    const xllm::layer::AttentionMetadata& metadata,
    int64_t row,
    int64_t begin,
    int64_t end) {
  if (begin == end) {
    return torch::empty({0, cache.size(2), cache.size(3)}, cache.options());
  }
  const int64_t block_size = cache.size(1);
  auto block_table = metadata.block_table.to(torch::kCPU).contiguous();
  auto block_table_acc = block_table.accessor<int32_t, 2>();
  std::vector<int64_t> slots;
  slots.reserve(static_cast<size_t>(end - begin));
  for (int64_t logical_token = begin; logical_token < end; ++logical_token) {
    const int64_t logical_block = logical_token / block_size;
    const int64_t block_offset = logical_token % block_size;
    const int64_t physical_block = block_table_acc[row][logical_block];
    slots.push_back(physical_block * block_size + block_offset);
  }
  auto slots_dev =
      torch::tensor(slots,
                    torch::TensorOptions()
                        .dtype(torch::kInt64)
                        .device(cache.device()))
          .contiguous();
  return cache
      .view({cache.size(0) * cache.size(1), cache.size(2), cache.size(3)})
      .index_select(0, slots_dev)
      .contiguous();
}

void expect_exact_tensor(const char* tag,
                         const torch::Tensor& got,
                         const torch::Tensor& expected) {
  ASSERT_EQ(got.sizes(), expected.sizes()) << tag;
  if (got.numel() == 0) {
    return;
  }
  auto diff = (got.to(torch::kFloat32) - expected.to(torch::kFloat32)).abs();
  const double max_abs = diff.max().item<double>();
  EXPECT_EQ(max_abs, 0.0) << tag;
}

void run_writeback_and_expect_cuda_matches_reference(
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    const xllm::layer::AttentionMetadata& metadata,
    const xllm::KVCache& initial_cache,
    xllm::KVCache* cuda_cache) {
  auto reference_cache = clone_kv_cache(initial_cache);
  *cuda_cache = clone_kv_cache(initial_cache);
  run_mtgr_kv_writeback_reference(
      key_snd, value_snd, metadata, reference_cache);
  run_mtgr_kv_writeback_cuda(key_snd, value_snd, metadata, *cuda_cache);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  expect_exact_tensor("key_cache_cuda_vs_reference",
                      cuda_cache->get_k_cache(),
                      reference_cache.get_k_cache());
  expect_exact_tensor("value_cache_cuda_vs_reference",
                      cuda_cache->get_v_cache(),
                      reference_cache.get_v_cache());
}

struct WritebackPerfCase {
  std::string mode;
  int64_t batch_size = 1;
  int64_t heads = 8;
  int64_t kv_heads = 8;
  int64_t head_dim = 128;
  int64_t total_q = 0;
  int64_t live_q = 0;
  int64_t write_tokens = 0;
  xllm::layer::AttentionMetadata metadata;
  torch::Tensor key_snd;
  torch::Tensor value_snd;
  xllm::KVCache kv_cache;
};

void write_writeback_perf_label_header(std::ostream& out) {
  out << "idx,mode,batch_size,heads,kv_heads,head_dim,total_q,live_q,"
         "write_tokens,repeat_id\n";
}

void write_writeback_perf_label_row(std::ostream& out,
                                    int64_t idx,
                                    const WritebackPerfCase& perf_case,
                                    int64_t repeat_id) {
  out << idx << "," << perf_case.mode << "," << perf_case.batch_size << ","
      << perf_case.heads << "," << perf_case.kv_heads << ","
      << perf_case.head_dim << "," << perf_case.total_q << ","
      << perf_case.live_q << "," << perf_case.write_tokens << ","
      << repeat_id << "\n";
}

MTGRAttentionTestShape make_writeback_perf_shape(bool partial_match) {
  MTGRAttentionTestShape shape;
  shape.heads = 8;
  shape.kv_heads = 8;
  shape.head_dim = 128;
  shape.history = 1351;
  shape.context = 8;
  shape.realtime = 401;
  shape.target = 801;
  shape.matched_prefix =
      partial_match ? shape.history + shape.context + (shape.realtime * 4) / 5
                    : 0;
  return shape;
}

WritebackPerfCase make_single_request_writeback_perf_case(
    const std::string& mode,
    const MTGRAttentionTestShape& shape,
    const torch::Device& device,
    int64_t block_size) {
  auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  auto full_key =
      torch::randn({shape.total_len(), shape.kv_heads, shape.head_dim}, opts);
  auto full_value =
      torch::randn({shape.total_len(), shape.kv_heads, shape.head_dim}, opts);
  auto key_snd =
      full_key.narrow(0, shape.matched_prefix, shape.local_len()).contiguous();
  auto value_snd =
      full_value.narrow(0, shape.matched_prefix, shape.local_len()).contiguous();

  WritebackPerfCase perf_case;
  perf_case.mode = mode;
  perf_case.batch_size = 1;
  perf_case.heads = shape.heads;
  perf_case.kv_heads = shape.kv_heads;
  perf_case.head_dim = shape.head_dim;
  perf_case.total_q = shape.total_len();
  perf_case.live_q = shape.local_len();
  perf_case.write_tokens = cacheable_end(shape) - shape.matched_prefix;
  perf_case.metadata = make_mtgr_attention_metadata(shape, device, block_size);
  perf_case.key_snd = std::move(key_snd);
  perf_case.value_snd = std::move(value_snd);
  perf_case.kv_cache =
      make_mtgr_kv_cache(shape, device, torch::kBFloat16, block_size);
  return perf_case;
}

WritebackPerfCase make_mixed_writeback_perf_case(const torch::Device& device,
                                                 int64_t block_size,
                                                 int64_t batch_size) {
  auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  std::vector<MTGRAttentionTestShape> shapes;
  std::vector<torch::Tensor> full_keys;
  std::vector<torch::Tensor> full_values;
  std::vector<torch::Tensor> live_keys;
  std::vector<torch::Tensor> live_values;
  shapes.reserve(static_cast<size_t>(batch_size));
  full_keys.reserve(static_cast<size_t>(batch_size));
  full_values.reserve(static_cast<size_t>(batch_size));
  live_keys.reserve(static_cast<size_t>(batch_size));
  live_values.reserve(static_cast<size_t>(batch_size));

  int64_t total_q = 0;
  int64_t live_q = 0;
  int64_t write_tokens = 0;
  for (int64_t row = 0; row < batch_size; ++row) {
    auto shape = make_writeback_perf_shape((row & 1) != 0);
    shape.history += (row % 7) * 2;
    shape.realtime += (row % 5) * 2;
    shape.target += (row % 3) * 2;
    if (shape.matched_prefix > 0) {
      shape.matched_prefix =
          shape.history + shape.context + (shape.realtime * 4) / 5;
    }
    auto full_key =
        torch::randn({1, shape.total_len(), shape.kv_heads, shape.head_dim},
                     opts);
    auto full_value =
        torch::randn({1, shape.total_len(), shape.kv_heads, shape.head_dim},
                     opts);
    live_keys.push_back(full_key.select(0, 0)
                            .narrow(0, shape.matched_prefix, shape.local_len())
                            .contiguous());
    live_values.push_back(full_value.select(0, 0)
                              .narrow(0, shape.matched_prefix, shape.local_len())
                              .contiguous());
    total_q += shape.total_len();
    live_q += shape.local_len();
    write_tokens += cacheable_end(shape) - shape.matched_prefix;
    full_keys.push_back(full_key);
    full_values.push_back(full_value);
    shapes.push_back(shape);
  }

  auto setup = make_mtgr_partial_batch_setup(
      shapes, full_keys, full_values, device, torch::kBFloat16, block_size);
  setup.metadata.mtgr_match_mode = xllm::layer::MTGRMatchMode::kMixed;

  WritebackPerfCase perf_case;
  perf_case.mode = "mixed";
  perf_case.batch_size = batch_size;
  perf_case.heads = shapes.front().heads;
  perf_case.kv_heads = shapes.front().kv_heads;
  perf_case.head_dim = shapes.front().head_dim;
  perf_case.total_q = total_q;
  perf_case.live_q = live_q;
  perf_case.write_tokens = write_tokens;
  perf_case.metadata = std::move(setup.metadata);
  perf_case.key_snd = torch::cat(live_keys, 0).contiguous();
  perf_case.value_snd = torch::cat(live_values, 0).contiguous();
  perf_case.kv_cache = std::move(setup.kv_cache);
  return perf_case;
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

TEST_F(MTGRAttentionHarnessTest, KVWritebackNoMatchSkipsTarget) {
  torch::NoGradGuard no_grad_guard;
  const auto shape = make_product_shape(/*partial_match=*/false);
  auto opts =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device_);
  torch::manual_seed(20260519);
  auto key_snd =
      torch::randn({shape.total_len(), shape.kv_heads, shape.head_dim}, opts);
  auto value_snd =
      torch::randn({shape.total_len(), shape.kv_heads, shape.head_dim}, opts);
  auto metadata = make_mtgr_attention_metadata(shape, device_, kDefaultBlockSize);
  auto initial_cache =
      make_sentinel_kv_cache(shape, device_, kDefaultBlockSize, -7.0);

  xllm::KVCache cuda_cache;
  run_writeback_and_expect_cuda_matches_reference(
      key_snd, value_snd, metadata, initial_cache, &cuda_cache);

  const int64_t write_end = cacheable_end(shape);
  expect_exact_tensor("no_match_key_written",
                      gather_cache_tokens(
                          cuda_cache.get_k_cache(), metadata, 0, 0, write_end),
                      key_snd.narrow(0, 0, write_end));
  expect_exact_tensor("no_match_value_written",
                      gather_cache_tokens(
                          cuda_cache.get_v_cache(), metadata, 0, 0, write_end),
                      value_snd.narrow(0, 0, write_end));
  auto target_key = gather_cache_tokens(cuda_cache.get_k_cache(),
                                        metadata,
                                        0,
                                        write_end,
                                        shape.total_len());
  auto target_value = gather_cache_tokens(cuda_cache.get_v_cache(),
                                          metadata,
                                          0,
                                          write_end,
                                          shape.total_len());
  expect_exact_tensor("no_match_target_key_unchanged",
                      target_key,
                      torch::full_like(target_key, -7.0));
  expect_exact_tensor("no_match_target_value_unchanged",
                      target_value,
                      torch::full_like(target_value, -7.0));
}

TEST_F(MTGRAttentionHarnessTest, KVWritebackPartialWritesUnmatchedRealtimeOnly) {
  torch::NoGradGuard no_grad_guard;
  const auto shape = make_product_shape(/*partial_match=*/true);
  auto opts =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device_);
  torch::manual_seed(20260520);
  auto full_key =
      torch::randn({shape.total_len(), shape.kv_heads, shape.head_dim}, opts);
  auto full_value =
      torch::randn({shape.total_len(), shape.kv_heads, shape.head_dim}, opts);
  auto key_snd =
      full_key.narrow(0, shape.matched_prefix, shape.local_len()).contiguous();
  auto value_snd =
      full_value.narrow(0, shape.matched_prefix, shape.local_len()).contiguous();
  auto metadata = make_mtgr_attention_metadata(shape, device_, kDefaultBlockSize);
  auto initial_cache =
      make_sentinel_kv_cache(shape, device_, kDefaultBlockSize, -11.0);

  xllm::KVCache cuda_cache;
  run_writeback_and_expect_cuda_matches_reference(
      key_snd, value_snd, metadata, initial_cache, &cuda_cache);

  const int64_t write_begin = shape.matched_prefix;
  const int64_t write_end = cacheable_end(shape);
  expect_exact_tensor(
      "partial_key_written",
      gather_cache_tokens(
          cuda_cache.get_k_cache(), metadata, 0, write_begin, write_end),
      full_key.narrow(0, write_begin, write_end - write_begin));
  expect_exact_tensor(
      "partial_value_written",
      gather_cache_tokens(
          cuda_cache.get_v_cache(), metadata, 0, write_begin, write_end),
      full_value.narrow(0, write_begin, write_end - write_begin));

  auto prefix_key = gather_cache_tokens(
      cuda_cache.get_k_cache(), metadata, 0, 0, write_begin);
  auto prefix_value = gather_cache_tokens(
      cuda_cache.get_v_cache(), metadata, 0, 0, write_begin);
  auto target_key = gather_cache_tokens(cuda_cache.get_k_cache(),
                                        metadata,
                                        0,
                                        write_end,
                                        shape.total_len());
  auto target_value = gather_cache_tokens(cuda_cache.get_v_cache(),
                                          metadata,
                                          0,
                                          write_end,
                                          shape.total_len());
  expect_exact_tensor("partial_prefix_key_unchanged",
                      prefix_key,
                      torch::full_like(prefix_key, -11.0));
  expect_exact_tensor("partial_prefix_value_unchanged",
                      prefix_value,
                      torch::full_like(prefix_value, -11.0));
  expect_exact_tensor("partial_target_key_unchanged",
                      target_key,
                      torch::full_like(target_key, -11.0));
  expect_exact_tensor("partial_target_value_unchanged",
                      target_value,
                      torch::full_like(target_value, -11.0));
}

TEST_F(MTGRAttentionHarnessTest, KVWritebackMixedBatchUsesRequestRanges) {
  torch::NoGradGuard no_grad_guard;
  auto opts =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device_);
  std::vector<MTGRAttentionTestShape> shapes(3);
  for (auto& shape : shapes) {
    shape.heads = 4;
    shape.kv_heads = 4;
    shape.head_dim = 64;
    shape.history = 129;
    shape.context = 7;
    shape.realtime = 65;
    shape.target = 67;
  }
  shapes[0].matched_prefix = 0;
  shapes[1].matched_prefix = shapes[1].history + shapes[1].context + 31;
  shapes[2].history = 257;
  shapes[2].realtime = 97;
  shapes[2].target = 101;
  shapes[2].matched_prefix = shapes[2].history + shapes[2].context + 55;

  torch::manual_seed(20260521);
  std::vector<torch::Tensor> full_keys;
  std::vector<torch::Tensor> full_values;
  std::vector<torch::Tensor> live_keys;
  std::vector<torch::Tensor> live_values;
  full_keys.reserve(shapes.size());
  full_values.reserve(shapes.size());
  live_keys.reserve(shapes.size());
  live_values.reserve(shapes.size());
  for (const auto& shape : shapes) {
    auto full_key =
        torch::randn({1, shape.total_len(), shape.kv_heads, shape.head_dim},
                     opts);
    auto full_value =
        torch::randn({1, shape.total_len(), shape.kv_heads, shape.head_dim},
                     opts);
    live_keys.push_back(full_key.select(0, 0)
                            .narrow(0, shape.matched_prefix, shape.local_len())
                            .contiguous());
    live_values.push_back(full_value.select(0, 0)
                              .narrow(0, shape.matched_prefix, shape.local_len())
                              .contiguous());
    full_keys.push_back(full_key);
    full_values.push_back(full_value);
  }

  auto setup = make_mtgr_partial_batch_setup(
      shapes, full_keys, full_values, device_, torch::kBFloat16, kDefaultBlockSize);
  setup.metadata.mtgr_match_mode = xllm::layer::MTGRMatchMode::kMixed;
  auto key_snd = torch::cat(live_keys, 0).contiguous();
  auto value_snd = torch::cat(live_values, 0).contiguous();

  xllm::KVCache cuda_cache;
  run_writeback_and_expect_cuda_matches_reference(
      key_snd, value_snd, setup.metadata, setup.kv_cache, &cuda_cache);

  for (int64_t row = 0; row < static_cast<int64_t>(shapes.size()); ++row) {
    const auto& shape = shapes[static_cast<size_t>(row)];
    const int64_t write_begin = shape.matched_prefix;
    const int64_t write_end = cacheable_end(shape);
    expect_exact_tensor(
        "mixed_key_written",
        gather_cache_tokens(
            cuda_cache.get_k_cache(), setup.metadata, row, write_begin, write_end),
        full_keys[static_cast<size_t>(row)].select(0, 0).narrow(
            0, write_begin, write_end - write_begin));
    expect_exact_tensor(
        "mixed_value_written",
        gather_cache_tokens(cuda_cache.get_v_cache(),
                            setup.metadata,
                            row,
                            write_begin,
                            write_end),
        full_values[static_cast<size_t>(row)].select(0, 0).narrow(
            0, write_begin, write_end - write_begin));
    expect_exact_tensor(
        "mixed_prefix_key_unchanged",
        gather_cache_tokens(
            cuda_cache.get_k_cache(), setup.metadata, row, 0, write_begin),
        gather_cache_tokens(
            setup.kv_cache.get_k_cache(), setup.metadata, row, 0, write_begin));
    expect_exact_tensor(
        "mixed_target_key_unchanged",
        gather_cache_tokens(cuda_cache.get_k_cache(),
                            setup.metadata,
                            row,
                            write_end,
                            shape.total_len()),
        gather_cache_tokens(setup.kv_cache.get_k_cache(),
                            setup.metadata,
                            row,
                            write_end,
                            shape.total_len()));
  }
}

TEST_F(MTGRAttentionHarnessTest, KVWritebackPerfNvtxCsv) {
  torch::NoGradGuard no_grad_guard;
  FLAGS_mtgr_nvtx_level = std::max(FLAGS_mtgr_nvtx_level, 2);
  const int repeat = std::max(1, env_int("XLLM_MTGR_WRITEBACK_PERF_REPEAT", 20));
  const int warmup = std::max(0, env_int("XLLM_MTGR_WRITEBACK_PERF_WARMUP", 3));
  const int mixed_batch =
      std::max(1, env_int("XLLM_MTGR_WRITEBACK_MIXED_BATCH", 32));
  const std::string labels_path =
      env_path("XLLM_MTGR_WRITEBACK_LABELS", "mtgr_writeback_labels.csv");

  std::ofstream labels(labels_path, std::ios::out | std::ios::trunc);
  CHECK(labels.is_open()) << "failed to open labels path: " << labels_path;
  write_writeback_perf_label_header(labels);

  torch::manual_seed(20260522);
  std::vector<WritebackPerfCase> cases;
  cases.push_back(make_single_request_writeback_perf_case(
      "no_match",
      make_writeback_perf_shape(/*partial_match=*/false),
      device_,
      kDefaultBlockSize));
  cases.push_back(make_single_request_writeback_perf_case(
      "partial_match",
      make_writeback_perf_shape(/*partial_match=*/true),
      device_,
      kDefaultBlockSize));
  cases.push_back(
      make_mixed_writeback_perf_case(device_, kDefaultBlockSize, mixed_batch));

  for (int i = 0; i < warmup; ++i) {
    for (auto& perf_case : cases) {
      run_mtgr_kv_writeback_cuda(perf_case.key_snd,
                                 perf_case.value_snd,
                                 perf_case.metadata,
                                 perf_case.kv_cache);
    }
  }
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  int64_t idx = 0;
  for (int r = 0; r < repeat; ++r) {
    for (auto& perf_case : cases) {
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      ++idx;
      write_writeback_perf_label_row(labels, idx, perf_case, r + 1);
      labels.flush();
      const std::string root_name =
          std::string("MTGR/harness/mtgr_attention/writeback/") +
          perf_case.mode;
      {
        xllm::MtgrNvtxRange root_range(1, root_name.c_str());
        run_mtgr_kv_writeback_cuda(perf_case.key_snd,
                                   perf_case.value_snd,
                                   perf_case.metadata,
                                   perf_case.kv_cache);
        {
          xllm::MtgrNvtxRange sync_range(
              2, "MTGR/harness/mtgr_attention/writeback/device_sync");
          ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        }
      }
    }
  }
  std::fprintf(stderr,
               "[MTGR][Harness][KVWritebackPerfNvtxCsv] repeat=%d "
               "warmup=%d cases=%zu labels=%lld labels_path=%s\n",
               repeat,
               warmup,
               cases.size(),
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
