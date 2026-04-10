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

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/torch_npu.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "ops_npu/npu_ops.h"

namespace xllm::kernel::npu::test {

namespace {

class AtbPagedAttentionBSNDTest : public ::testing::Test {
 protected:
  static constexpr int32_t kDeviceId = 5;

  static void SetUpTestSuite() {
    torch_npu::init_npu("npu:" + std::to_string(kDeviceId));
  }

  static void TearDownTestSuite() { torch_npu::finalize_npu(); }

  void SetUp() override {
    device_ = torch::Device(torch::kPrivateUse1, kDeviceId);
    fp16_opts_ = torch::TensorOptions().dtype(torch::kFloat16).device(device_);
    i32_dev_opts_ = torch::TensorOptions().dtype(torch::kInt32).device(device_);
    i64_dev_opts_ = torch::TensorOptions().dtype(torch::kInt64).device(device_);
    i32_cpu_opts_ = torch::TensorOptions().dtype(torch::kInt32);
    i64_cpu_opts_ = torch::TensorOptions().dtype(torch::kInt64);
    stream_ = c10_npu::getCurrentNPUStream(kDeviceId).stream();
    ASSERT_NE(stream_, nullptr);
  }

  torch::Device device_{torch::kCPU};
  torch::TensorOptions fp16_opts_;
  torch::TensorOptions i32_dev_opts_;
  torch::TensorOptions i64_dev_opts_;
  torch::TensorOptions i32_cpu_opts_;
  torch::TensorOptions i64_cpu_opts_;
  aclrtStream stream_{nullptr};
};

std::vector<int32_t> build_slot_mapping(const std::vector<int32_t>& block_table,
                                        int64_t block_size,
                                        int64_t start_token_idx,
                                        int64_t token_count) {
  std::vector<int32_t> slots;
  slots.reserve(static_cast<size_t>(token_count));
  for (int64_t token_idx = start_token_idx;
       token_idx < start_token_idx + token_count;
       ++token_idx) {
    const int64_t logical_block = token_idx / block_size;
    const int64_t offset = token_idx % block_size;
    const int64_t physical_block = block_table.at(static_cast<size_t>(logical_block));
    slots.push_back(static_cast<int32_t>(physical_block * block_size + offset));
  }
  return slots;
}

torch::Tensor reference_single_token_decode_bsnd(
    const torch::Tensor& query_bhd_cpu_f32,
    const torch::Tensor& key_cache_cpu_f32,
    const torch::Tensor& value_cache_cpu_f32,
    const std::vector<int32_t>& block_table,
    int64_t block_size,
    int64_t seq_len,
    double scale,
    const torch::TensorOptions& i64_cpu_opts) {
  auto all_slots = build_slot_mapping(block_table, block_size, 0, seq_len);
  auto all_slots_i64 = torch::tensor(all_slots, i64_cpu_opts);

  auto k_seq = key_cache_cpu_f32
                   .view({-1, key_cache_cpu_f32.size(2), key_cache_cpu_f32.size(3)})
                   .index_select(0, all_slots_i64);  // [L, H, D]
  auto v_seq = value_cache_cpu_f32
                   .view({-1, value_cache_cpu_f32.size(2), value_cache_cpu_f32.size(3)})
                   .index_select(0, all_slots_i64);  // [L, H, D]

  auto q_hd = query_bhd_cpu_f32.squeeze(0);  // [H, D]
  auto scores_hl = torch::einsum("hd,lhd->hl", {q_hd, k_seq}) * scale;
  auto probs_hl = torch::softmax(scores_hl, -1);
  return torch::einsum("hl,lhd->hd", {probs_hl, v_seq});  // [H, D]
}

TEST_F(AtbPagedAttentionBSNDTest, ScatterThenPagedAttentionDecodeBSND) {
  torch::manual_seed(20260408);

  constexpr int64_t kBatchSize = 1;
  constexpr int64_t kNumHeads = 8;
  constexpr int64_t kNumKvHeads = 8;
  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kTotalSeqLen = 2000;
  constexpr int64_t kPrefixLen = 1500;
  constexpr int64_t kNewKvLen = kTotalSeqLen - kPrefixLen;
  constexpr int64_t kSeqBlockCount = (kTotalSeqLen + kBlockSize - 1) / kBlockSize;  // 16
  constexpr int64_t kCacheBlockCount = 24;

  const std::vector<int32_t> block_table_host = {
      3,  6,  1, 8, 2, 11, 12, 15, 13, 14, 4, 5, 7, 20, 18, 10};
  ASSERT_EQ(static_cast<int64_t>(block_table_host.size()), kSeqBlockCount);

  auto key_cache = torch::zeros(
      {kCacheBlockCount, kBlockSize, kNumKvHeads, kHeadDim}, fp16_opts_);
  
  auto value_cache = torch::zeros_like(key_cache);

  auto prefix_key = torch::randn({kPrefixLen, kNumKvHeads, kHeadDim}, fp16_opts_);
  auto prefix_value = torch::randn({kPrefixLen, kNumKvHeads, kHeadDim}, fp16_opts_);
  auto new_key = torch::randn({kNewKvLen, kNumKvHeads, kHeadDim}, fp16_opts_);
  auto new_value = torch::randn({kNewKvLen, kNumKvHeads, kHeadDim}, fp16_opts_);

  auto prefix_slots_host =
      build_slot_mapping(block_table_host, kBlockSize, /*start_token_idx=*/0, kPrefixLen);
  auto new_slots_host = build_slot_mapping(
      block_table_host, kBlockSize, /*start_token_idx=*/kPrefixLen, kNewKvLen);

  auto prefix_slots_dev_i64 = torch::tensor(prefix_slots_host, i64_dev_opts_);
  auto new_slots_dev_i32 = torch::tensor(new_slots_host, i32_dev_opts_);
  auto new_slots_dev_i64 = new_slots_dev_i32.to(torch::kLong);

  auto key_cache_flat =
      key_cache.view({kCacheBlockCount * kBlockSize, kNumKvHeads, kHeadDim});
  auto value_cache_flat =
      value_cache.view({kCacheBlockCount * kBlockSize, kNumKvHeads, kHeadDim});
  key_cache_flat.index_copy_(0, prefix_slots_dev_i64, prefix_key);
  value_cache_flat.index_copy_(0, prefix_slots_dev_i64, prefix_value);

  atb::npu_reshape_and_cache(
      new_key, new_value, key_cache, value_cache, new_slots_dev_i32);

  auto scattered_key =
      key_cache_flat.index_select(0, new_slots_dev_i64).to(torch::kFloat32);
  auto scattered_value =
      value_cache_flat.index_select(0, new_slots_dev_i64).to(torch::kFloat32);
  auto new_key_f32 = new_key.to(torch::kFloat32);
  auto new_value_f32 = new_value.to(torch::kFloat32);
  EXPECT_TRUE(torch::allclose(scattered_key, new_key_f32, 1e-3, 1e-3));
  EXPECT_TRUE(torch::allclose(scattered_value, new_value_f32, 1e-3, 1e-3));

  auto query = torch::randn({kBatchSize, kNumHeads, kHeadDim}, fp16_opts_);
  auto output = torch::zeros_like(query);
  auto block_table =
      torch::tensor(block_table_host, i32_dev_opts_).view({kBatchSize, kSeqBlockCount});
  auto kv_seq_lens = torch::tensor(
      {static_cast<int32_t>(kTotalSeqLen)}, i32_cpu_opts_);
  const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

  atb::npu_paged_attention(query,
                           key_cache,
                           value_cache,
                           kNumKvHeads,
                           kNumHeads,
                           scale,
                           block_table,
                           kv_seq_lens,
                           output);

  auto out_hd = output.to(torch::kCPU).to(torch::kFloat32).squeeze(0);
  auto ref_hd = reference_single_token_decode_bsnd(query.to(torch::kCPU).to(torch::kFloat32),
                                                   key_cache.to(torch::kCPU).to(torch::kFloat32),
                                                   value_cache.to(torch::kCPU).to(torch::kFloat32),
                                                   block_table_host,
                                                   kBlockSize,
                                                   kTotalSeqLen,
                                                   scale,
                                                   i64_cpu_opts_);

  auto diff = (out_hd - ref_hd).abs();
  const double max_abs = diff.max().item<double>();
  const double mean_abs = diff.mean().item<double>();
  std::fprintf(stderr,
               "[ATB-PagedAttention][BSND] max_abs=%.6e mean_abs=%.6e\n",
               max_abs,
               mean_abs);
  std::fflush(stderr);

  EXPECT_LT(max_abs, 6e-2);
  EXPECT_TRUE(torch::allclose(out_hd, ref_hd, 6e-2, 6e-2));
}

}  // namespace

}  // namespace xllm::kernel::npu::test
