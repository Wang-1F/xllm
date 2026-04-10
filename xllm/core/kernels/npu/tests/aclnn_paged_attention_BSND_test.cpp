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
#include <limits>
#include <vector>

#if __has_include("acl/acl.h")
#include "acl/acl.h"
#elif __has_include("third_party/acl/inc/acl/acl.h")
#include "third_party/acl/inc/acl/acl.h"
#elif __has_include("/usr/local/Ascend/ascend-toolkit/latest/include/acl/acl.h")
#include "/usr/local/Ascend/ascend-toolkit/latest/include/acl/acl.h"
#elif __has_include("/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/acl/acl.h")
#include "/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/acl/acl.h"
#else
#error "Cannot find acl/acl.h"
#endif

#if __has_include("aclnnop/aclnn_scatter_pa_kv_cache.h")
#include "aclnnop/aclnn_scatter_pa_kv_cache.h"
#elif __has_include("/usr/local/Ascend/ascend-toolkit/latest/include/aclnnop/aclnn_scatter_pa_kv_cache.h")
#include "/usr/local/Ascend/ascend-toolkit/latest/include/aclnnop/aclnn_scatter_pa_kv_cache.h"
#elif __has_include("/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/aclnnop/aclnn_scatter_pa_kv_cache.h")
#include "/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/aclnnop/aclnn_scatter_pa_kv_cache.h"
#else
#error "Cannot find aclnnop/aclnn_scatter_pa_kv_cache.h"
#endif

#if __has_include("aclnnop/aclnn_fused_infer_attention_score_v3.h")
#include "aclnnop/aclnn_fused_infer_attention_score_v3.h"
#elif __has_include("/usr/local/Ascend/ascend-toolkit/latest/include/aclnnop/aclnn_fused_infer_attention_score_v3.h")
#include "/usr/local/Ascend/ascend-toolkit/latest/include/aclnnop/aclnn_fused_infer_attention_score_v3.h"
#elif __has_include("/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/aclnnop/aclnn_fused_infer_attention_score_v3.h")
#include "/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/aclnnop/aclnn_fused_infer_attention_score_v3.h"
#else
#error "Cannot find aclnnop/aclnn_fused_infer_attention_score_v3.h"
#endif

namespace xllm::kernel::npu::test {

namespace {

class AclnnPagedAttentionBSNDTest : public ::testing::Test {
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

aclDataType to_acl_dtype(torch::ScalarType dtype) {
  switch (dtype) {
    case torch::kFloat16:
      return ACL_FLOAT16;
    case torch::kFloat32:
      return ACL_FLOAT;
    case torch::kBFloat16:
      return ACL_BF16;
    case torch::kBool:
      return ACL_BOOL;
    case torch::kInt32:
      return ACL_INT32;
    case torch::kInt64:
      return ACL_INT64;
    default:
      return ACL_DT_UNDEFINED;
  }
}

aclTensor* torch_to_acl_tensor(const torch::Tensor& t) {
  if (!t.is_contiguous()) {
    return nullptr;
  }
  const auto dtype = to_acl_dtype(t.scalar_type());
  if (dtype == ACL_DT_UNDEFINED) {
    return nullptr;
  }
  return aclCreateTensor(t.sizes().data(),
                         t.sizes().size(),
                         dtype,
                         t.strides().data(),
                         /*offset=*/0,
                         ACL_FORMAT_ND,
                         t.sizes().data(),
                         t.sizes().size(),
                         t.data_ptr());
}

bool run_scatter_pa_kv_cache(const torch::Tensor& key,
                             const torch::Tensor& value,
                             torch::Tensor& key_cache,
                             torch::Tensor& value_cache,
                             const torch::Tensor& slot_mapping,
                             aclrtStream stream) {
  aclTensor* key_acl = nullptr;
  aclTensor* value_acl = nullptr;
  aclTensor* key_cache_acl = nullptr;
  aclTensor* value_cache_acl = nullptr;
  aclTensor* slot_mapping_acl = nullptr;
  void* workspace_ptr = nullptr;
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  bool ok = false;

  auto cleanup = [&]() {
    if (key_acl != nullptr) {
      aclDestroyTensor(key_acl);
    }
    if (value_acl != nullptr) {
      aclDestroyTensor(value_acl);
    }
    if (key_cache_acl != nullptr) {
      aclDestroyTensor(key_cache_acl);
    }
    if (value_cache_acl != nullptr) {
      aclDestroyTensor(value_cache_acl);
    }
    if (slot_mapping_acl != nullptr) {
      aclDestroyTensor(slot_mapping_acl);
    }
    if (workspace_ptr != nullptr) {
      aclrtFree(workspace_ptr);
    }
  };

  key_acl = torch_to_acl_tensor(key);
  value_acl = torch_to_acl_tensor(value);
  key_cache_acl = torch_to_acl_tensor(key_cache);
  value_cache_acl = torch_to_acl_tensor(value_cache);
  slot_mapping_acl = torch_to_acl_tensor(slot_mapping);
  if (key_acl == nullptr || value_acl == nullptr || key_cache_acl == nullptr ||
      value_cache_acl == nullptr || slot_mapping_acl == nullptr) {
    cleanup();
    return false;
  }

  char cache_mode[] = "Norm";
  char scatter_mode[] = "None";
  auto ret = aclnnScatterPaKvCacheGetWorkspaceSize(
      key_acl,
      key_cache_acl,
      slot_mapping_acl,
      value_acl,
      value_cache_acl,
      /*compressLensOptional=*/nullptr,
      /*compressSeqOffsetOptional=*/nullptr,
      /*seqLensOptional=*/nullptr,
      cache_mode,
      scatter_mode,
      /*stridesOptional=*/nullptr,
      /*offsetsOptional=*/nullptr,
      &workspace_size,
      &executor);
  if (ret != ACL_SUCCESS) {
    std::fprintf(stderr,
                 "aclnnScatterPaKvCacheGetWorkspaceSize failed: %d\n",
                 static_cast<int>(ret));
    cleanup();
    return false;
  }

  if (workspace_size > 0) {
    ret = aclrtMalloc(&workspace_ptr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      std::fprintf(stderr, "aclrtMalloc scatter workspace failed: %d\n", static_cast<int>(ret));
      cleanup();
      return false;
    }
  }

  ret = aclnnScatterPaKvCache(workspace_ptr, workspace_size, executor, stream);
  if (ret != ACL_SUCCESS) {
    std::fprintf(stderr, "aclnnScatterPaKvCache failed: %d\n", static_cast<int>(ret));
    cleanup();
    return false;
  }

  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    std::fprintf(stderr, "aclrtSynchronizeStream after scatter failed: %d\n", static_cast<int>(ret));
    cleanup();
    return false;
  }

  ok = true;
  cleanup();
  return ok;
}

bool run_fused_infer_paged_attention_v3(const torch::Tensor& query_bsnd,
                                        const torch::Tensor& key_cache,
                                        const torch::Tensor& value_cache,
                                        const torch::Tensor& block_table,
                                        int64_t total_kv_len,
                                        int64_t block_size,
                                        int64_t num_heads,
                                        int64_t num_kv_heads,
                                        double scale,
                                        torch::Tensor& output_bsnd,
                                        aclrtStream stream) {
  aclTensor* query_acl = nullptr;
  aclTensor* key_cache_acl = nullptr;
  aclTensor* value_cache_acl = nullptr;
  aclTensor* block_table_acl = nullptr;
  aclTensor* output_acl = nullptr;
  aclTensorList* key_list = nullptr;
  aclTensorList* value_list = nullptr;
  aclIntArray* actual_seq_lengths = nullptr;
  aclIntArray* actual_seq_lengths_kv = nullptr;
  void* workspace_ptr = nullptr;
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  bool ok = false;

  auto cleanup = [&]() {
    if (actual_seq_lengths != nullptr) {
      aclDestroyIntArray(actual_seq_lengths);
    }
    if (actual_seq_lengths_kv != nullptr) {
      aclDestroyIntArray(actual_seq_lengths_kv);
    }
    if (key_list != nullptr) {
      aclDestroyTensorList(key_list);
      key_cache_acl = nullptr;
    }
    if (value_list != nullptr) {
      aclDestroyTensorList(value_list);
      value_cache_acl = nullptr;
    }
    if (query_acl != nullptr) {
      aclDestroyTensor(query_acl);
    }
    if (key_cache_acl != nullptr) {
      aclDestroyTensor(key_cache_acl);
    }
    if (value_cache_acl != nullptr) {
      aclDestroyTensor(value_cache_acl);
    }
    if (block_table_acl != nullptr) {
      aclDestroyTensor(block_table_acl);
    }
    if (output_acl != nullptr) {
      aclDestroyTensor(output_acl);
    }
    if (workspace_ptr != nullptr) {
      aclrtFree(workspace_ptr);
    }
  };

  query_acl = torch_to_acl_tensor(query_bsnd);
  key_cache_acl = torch_to_acl_tensor(key_cache);
  value_cache_acl = torch_to_acl_tensor(value_cache);
  block_table_acl = torch_to_acl_tensor(block_table);
  output_acl = torch_to_acl_tensor(output_bsnd);
  if (query_acl == nullptr || key_cache_acl == nullptr || value_cache_acl == nullptr ||
      block_table_acl == nullptr || output_acl == nullptr) {
    cleanup();
    return false;
  }

  aclTensor* k_arr[1] = {key_cache_acl};
  aclTensor* v_arr[1] = {value_cache_acl};
  key_list = aclCreateTensorList(k_arr, 1);
  value_list = aclCreateTensorList(v_arr, 1);
  if (key_list == nullptr || value_list == nullptr) {
    cleanup();
    return false;
  }

  int64_t q_seqlen[1] = {query_bsnd.size(1)};
  int64_t kv_seqlen[1] = {total_kv_len};
  actual_seq_lengths = aclCreateIntArray(q_seqlen, 1);
  actual_seq_lengths_kv = aclCreateIntArray(kv_seqlen, 1);
  if (actual_seq_lengths == nullptr || actual_seq_lengths_kv == nullptr) {
    cleanup();
    return false;
  }

  char layout[] = "BSND";
  constexpr int64_t kPreTokens = 2147483647;
  constexpr int64_t kNextTokens = 2147483647;
  auto ret = aclnnFusedInferAttentionScoreV3GetWorkspaceSize(
      query_acl,
      key_list,
      value_list,
      /*pseShift=*/nullptr,
      /*attenMask=*/nullptr,
      /*actualSeqLengths=*/actual_seq_lengths,
      /*actualSeqLengthsKv=*/actual_seq_lengths_kv,
      /*deqScale1=*/nullptr,
      /*quantScale1=*/nullptr,
      /*deqScale2=*/nullptr,
      /*quantScale2=*/nullptr,
      /*quantOffset2=*/nullptr,
      /*antiquantScale=*/nullptr,
      /*antiquantOffset=*/nullptr,
      /*blockTable=*/block_table_acl,
      /*queryPaddingSize=*/nullptr,
      /*kvPaddingSize=*/nullptr,
      /*keyAntiquantScale=*/nullptr,
      /*keyAntiquantOffset=*/nullptr,
      /*valueAntiquantScale=*/nullptr,
      /*valueAntiquantOffset=*/nullptr,
      /*keySharedPrefix=*/nullptr,
      /*valueSharedPrefix=*/nullptr,
      /*actualSharedPrefixLen=*/nullptr,
      /*queryRope=*/nullptr,
      /*keyRope=*/nullptr,
      /*keyRopeAntiquantScale=*/nullptr,
      num_heads,
      scale,
      kPreTokens,
      kNextTokens,
      layout,
      num_kv_heads,
      /*sparseMode=*/0,
      /*innerPrecise=*/1,
      /*blockSize=*/block_size,
      /*antiquantMode=*/0,
      /*softmaxLseFlag=*/false,
      /*keyAntiquantMode=*/0,
      /*valueAntiquantMode=*/0,
      output_acl,
      /*softmaxLse=*/nullptr,
      &workspace_size,
      &executor);
  if (ret != ACL_SUCCESS) {
    std::fprintf(stderr,
                 "aclnnFusedInferAttentionScoreV3GetWorkspaceSize failed: %d\n",
                 static_cast<int>(ret));
    cleanup();
    return false;
  }

  if (workspace_size > 0) {
    ret = aclrtMalloc(&workspace_ptr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      std::fprintf(stderr,
                   "aclrtMalloc paged-attention workspace failed: %d\n",
                   static_cast<int>(ret));
      cleanup();
      return false;
    }
  }

  ret = aclnnFusedInferAttentionScoreV3(workspace_ptr, workspace_size, executor, stream);
  if (ret != ACL_SUCCESS) {
    std::fprintf(stderr, "aclnnFusedInferAttentionScoreV3 failed: %d\n", static_cast<int>(ret));
    cleanup();
    return false;
  }

  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    std::fprintf(stderr,
                 "aclrtSynchronizeStream after fused infer attention failed: %d\n",
                 static_cast<int>(ret));
    cleanup();
    return false;
  }

  ok = true;
  cleanup();
  return ok;
}

torch::Tensor reference_single_token_decode_bsnd(
    const torch::Tensor& query_bsnd_cpu_f32,
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

  auto q_hd = query_bsnd_cpu_f32.squeeze(0).squeeze(0);  // [H, D]
  auto scores_hl = torch::einsum("hd,lhd->hl", {q_hd, k_seq}) * scale;
  auto probs_hl = torch::softmax(scores_hl, -1);
  return torch::einsum("hl,lhd->hd", {probs_hl, v_seq});  // [H, D]
}

TEST_F(AclnnPagedAttentionBSNDTest, ScatterThenPagedAttentionDecodeBSND) {
  torch::manual_seed(20260408);

  constexpr int64_t kBatchSize = 1;
  constexpr int64_t kQSeqLen = 1;
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

  auto key_cache =
      torch::zeros({kCacheBlockCount, kBlockSize, kNumKvHeads, kHeadDim}, fp16_opts_);
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

  ASSERT_TRUE(run_scatter_pa_kv_cache(
      new_key, new_value, key_cache, value_cache, new_slots_dev_i32, stream_));

  auto scattered_key =
      key_cache_flat.index_select(0, new_slots_dev_i64).to(torch::kFloat32);
  auto scattered_value =
      value_cache_flat.index_select(0, new_slots_dev_i64).to(torch::kFloat32);
  EXPECT_TRUE(torch::allclose(scattered_key, new_key.to(torch::kFloat32), 1e-3, 1e-3));
  EXPECT_TRUE(torch::allclose(scattered_value, new_value.to(torch::kFloat32), 1e-3, 1e-3));

  auto query = torch::randn({kBatchSize, kQSeqLen, kNumHeads, kHeadDim}, fp16_opts_);
  auto output = torch::zeros_like(query);
  auto block_table =
      torch::tensor(block_table_host, i32_dev_opts_).view({kBatchSize, kSeqBlockCount});
  const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));

  auto key_cache_bnbsh =
      key_cache.view({kCacheBlockCount, kBlockSize, kNumKvHeads * kHeadDim});
  auto value_cache_bnbsh =
      value_cache.view({kCacheBlockCount, kBlockSize, kNumKvHeads * kHeadDim});

  ASSERT_TRUE(run_fused_infer_paged_attention_v3(query,
                                                 key_cache_bnbsh,
                                                 value_cache_bnbsh,
                                                 block_table,
                                                 kTotalSeqLen,
                                                 kBlockSize,
                                                 kNumHeads,
                                                 kNumKvHeads,
                                                 scale,
                                                 output,
                                                 stream_));

  auto out_hd = output.to(torch::kCPU).to(torch::kFloat32).squeeze(0).squeeze(0);
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
               "[ACLNN-PagedAttention][BSND] max_abs=%.6e mean_abs=%.6e\n",
               max_abs,
               mean_abs);
  std::fflush(stderr);

  EXPECT_LT(max_abs, 6e-2);
  EXPECT_TRUE(torch::allclose(out_hd, ref_hd, 6e-2, 6e-2));
}

}  // namespace

}  // namespace xllm::kernel::npu::test
