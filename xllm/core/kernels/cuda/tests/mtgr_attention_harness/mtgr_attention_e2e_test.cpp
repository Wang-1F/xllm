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
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>

#include "core/common/global_flags.h"
#include "core/util/mtgr_nvtx.h"

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

}  // namespace xllm::kernel::cuda::test::mtgr_attention_harness
