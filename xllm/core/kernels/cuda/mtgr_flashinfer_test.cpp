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

#include "mtgr_flashinfer.h"

#include <gtest/gtest.h>
#include <torch/cuda.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "core/common/global_flags.h"

namespace xllm::kernel::cuda {
namespace test {
namespace {

constexpr int64_t kHistoryLen = 2;
constexpr int64_t kContextLen = 1;
constexpr int64_t kRealtimeLen = 1;
constexpr int64_t kTargetLen = 1;
constexpr int64_t kTotalLen =
    kHistoryLen + kContextLen + kRealtimeLen + kTargetLen;

constexpr int64_t kNumHeads = 4;
constexpr int64_t kNumKvHeads = 4;
constexpr int64_t kHeadDim = 64;

constexpr int64_t kBlockSize = 4;
constexpr int64_t kNumBlocks = 2;

int64_t slot_of(const std::vector<int32_t>& block_table_host,
                int64_t token_idx,
                int64_t block_size) {
  const int64_t logical_block = token_idx / block_size;
  const int64_t offset = token_idx % block_size;
  CHECK_GE(logical_block, 0);
  CHECK_LT(logical_block, static_cast<int64_t>(block_table_host.size()));
  const int64_t physical_block =
      block_table_host[static_cast<size_t>(logical_block)];
  return physical_block * block_size + offset;
}

torch::Tensor build_slot_tensor_i32(
    const std::vector<int32_t>& block_table_host,
    int64_t start_token_idx,
    int64_t token_count,
    int64_t block_size,
    const torch::Device& device) {
  std::vector<int32_t> slots;
  slots.reserve(static_cast<size_t>(token_count));
  for (int64_t i = 0; i < token_count; ++i) {
    const int64_t logical_idx = start_token_idx + i;
    slots.push_back(static_cast<int32_t>(
        slot_of(block_table_host, logical_idx, block_size)));
  }
  return torch::tensor(
      slots, torch::TensorOptions().dtype(torch::kInt32).device(device));
}

torch::Tensor build_slot_tensor_i64(
    const std::vector<int32_t>& block_table_host,
    int64_t start_token_idx,
    int64_t token_count,
    int64_t block_size,
    const torch::Device& device) {
  std::vector<int64_t> slots;
  slots.reserve(static_cast<size_t>(token_count));
  for (int64_t i = 0; i < token_count; ++i) {
    const int64_t logical_idx = start_token_idx + i;
    slots.push_back(slot_of(block_table_host, logical_idx, block_size));
  }
  return torch::tensor(
      slots, torch::TensorOptions().dtype(torch::kInt64).device(device));
}

void preload_matched_prefix_to_cache(
    const torch::Tensor& full_key_snd,
    const torch::Tensor& full_value_snd,
    int64_t matched_prefix_len,
    const std::vector<int32_t>& block_table_host,
    int64_t block_size,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache) {
  if (matched_prefix_len <= 0) {
    return;
  }
  auto slots = build_slot_tensor_i64(block_table_host,
                                     /*start_token_idx=*/0,
                                     /*token_count=*/matched_prefix_len,
                                     block_size,
                                     key_cache.device());
  auto key_flat = key_cache.view({key_cache.size(0) * key_cache.size(1),
                                  key_cache.size(2),
                                  key_cache.size(3)});
  auto value_flat = value_cache.view({value_cache.size(0) * value_cache.size(1),
                                      value_cache.size(2),
                                      value_cache.size(3)});
  key_flat.index_copy_(0, slots, full_key_snd.narrow(0, 0, matched_prefix_len));
  value_flat.index_copy_(
      0, slots, full_value_snd.narrow(0, 0, matched_prefix_len));
}

int64_t prefix_unmatched_len_for_case(int64_t matched_prefix_len) {
  const int64_t prefix_len = kHistoryLen + kContextLen + kRealtimeLen;
  if (matched_prefix_len == 0) {
    return prefix_len;
  }
  if (matched_prefix_len < kHistoryLen) {
    return (kHistoryLen - matched_prefix_len) + kContextLen;
  }
  if (matched_prefix_len < kHistoryLen + kContextLen) {
    return (kHistoryLen + kContextLen - matched_prefix_len);
  }
  return (prefix_len - matched_prefix_len);
}

torch::Tensor run_mtgr_flashinfer_case(
    const torch::Tensor& full_query_bsnd,
    const torch::Tensor& full_key_bsnd,
    const torch::Tensor& full_value_bsnd,
    int64_t matched_prefix_len,
    const std::vector<int32_t>& block_table_host) {
  CHECK_EQ(full_query_bsnd.dim(), 4);
  CHECK_EQ(full_query_bsnd.size(0), 1);
  CHECK_EQ(full_key_bsnd.dim(), 4);
  CHECK_EQ(full_key_bsnd.size(0), 1);
  CHECK_EQ(full_value_bsnd.dim(), 4);
  CHECK_EQ(full_value_bsnd.size(0), 1);

  const auto device = full_query_bsnd.device();
  const int64_t local_start = matched_prefix_len;
  const int64_t local_len = kTotalLen - matched_prefix_len;
  auto query = full_query_bsnd.narrow(1, local_start, local_len).contiguous();
  auto key = full_key_bsnd.narrow(1, local_start, local_len).contiguous();
  auto value = full_value_bsnd.narrow(1, local_start, local_len).contiguous();

  auto cache_opts =
      torch::TensorOptions().dtype(full_key_bsnd.scalar_type()).device(device);
  auto key_cache =
      torch::zeros({kNumBlocks, kBlockSize, kNumKvHeads, kHeadDim}, cache_opts);
  auto value_cache =
      torch::zeros({kNumBlocks, kBlockSize, kNumKvHeads, kHeadDim}, cache_opts);

  auto full_key_snd = full_key_bsnd.select(0, 0).contiguous();
  auto full_value_snd = full_value_bsnd.select(0, 0).contiguous();
  preload_matched_prefix_to_cache(full_key_snd,
                                  full_value_snd,
                                  matched_prefix_len,
                                  block_table_host,
                                  kBlockSize,
                                  key_cache,
                                  value_cache);

  const int64_t prefix_unmatched_len =
      prefix_unmatched_len_for_case(matched_prefix_len);
  auto slot_mapping =
      build_slot_tensor_i32(block_table_host,
                            /*start_token_idx=*/matched_prefix_len,
                            /*token_count=*/prefix_unmatched_len,
                            kBlockSize,
                            device);

  MtgrFlashinferMetadata metadata;
  metadata.history_len = kHistoryLen;
  metadata.context_len = kContextLen;
  metadata.real_time_len = kRealtimeLen;
  metadata.target_len = kTargetLen;
  metadata.matched_prefix_len = matched_prefix_len;
  metadata.block_size = kBlockSize;
  metadata.block_table =
      torch::tensor(block_table_host,
                    torch::TensorOptions().dtype(torch::kInt32).device(device));
  metadata.slot_mapping = slot_mapping;

  torch::Tensor output;
  mtgr_flashinfer_attention_forward(query,
                                    key,
                                    value,
                                    key_cache,
                                    value_cache,
                                    metadata,
                                    /*sm_scale=*/1.0,
                                    output);
  torch::cuda::synchronize();
  return output;
}

class MtgrFlashinferAttentionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA not available, skipping test.";
    }
    FLAGS_flashinfer_workspace_buffer_size = std::max<int64_t>(
        FLAGS_flashinfer_workspace_buffer_size, 64 * 1024 * 1024);
    torch::manual_seed(20260417);
    torch::cuda::manual_seed_all(20260417);
    device_ = torch::Device(torch::kCUDA, 0);
  }

  torch::Device device_ = torch::Device(torch::kCPU);
};

TEST_F(MtgrFlashinferAttentionTest, MatchedBranchesAlignWithNoMatchBaseline) {
  auto opts = torch::TensorOptions().dtype(torch::kFloat16).device(device_);
  auto full_query =
      torch::randn({1, kTotalLen, kNumHeads, kHeadDim}, opts) * 0.05;
  auto full_key =
      torch::randn({1, kTotalLen, kNumKvHeads, kHeadDim}, opts) * 0.05;
  auto full_value =
      torch::randn({1, kTotalLen, kNumKvHeads, kHeadDim}, opts) * 0.05;

  const std::vector<int32_t> block_table_host = {0, 1};

  auto baseline = run_mtgr_flashinfer_case(full_query,
                                           full_key,
                                           full_value,
                                           /*matched_prefix_len=*/0,
                                           block_table_host);
  ASSERT_EQ(baseline.sizes(),
            torch::IntArrayRef({1, kTotalLen, kNumHeads, kHeadDim}));
  ASSERT_TRUE(torch::isfinite(baseline).all().item<bool>());

  const std::vector<int64_t> matched_cases = {1, 2, 3};
  for (int64_t matched : matched_cases) {
    auto case_output = run_mtgr_flashinfer_case(
        full_query, full_key, full_value, matched, block_table_host);
    auto baseline_slice =
        baseline.narrow(1, matched, kTotalLen - matched).contiguous();

    ASSERT_EQ(case_output.sizes(), baseline_slice.sizes());
    ASSERT_TRUE(torch::isfinite(case_output).all().item<bool>())
        << "Found non-finite values in case matched=" << matched;

    EXPECT_TRUE(torch::allclose(
        case_output, baseline_slice, /*rtol=*/5e-2, /*atol=*/5e-2))
        << "Output mismatch for matched_prefix_len=" << matched;
  }
}

}  // namespace
}  // namespace test
}  // namespace xllm::kernel::cuda
