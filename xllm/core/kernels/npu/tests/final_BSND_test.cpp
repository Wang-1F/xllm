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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
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

#if __has_include("aclnnop/aclnn_attention_update.h")
#include "aclnnop/aclnn_attention_update.h"
#elif __has_include("/usr/local/Ascend/ascend-toolkit/latest/include/aclnnop/aclnn_attention_update.h")
#include "/usr/local/Ascend/ascend-toolkit/latest/include/aclnnop/aclnn_attention_update.h"
#elif __has_include("/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/aclnnop/aclnn_attention_update.h")
#include "/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/aclnnop/aclnn_attention_update.h"
#else
#error "Cannot find aclnnop/aclnn_attention_update.h"
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

#if __has_include("aclnnop/aclnn_scatter_pa_kv_cache.h")
#include "aclnnop/aclnn_scatter_pa_kv_cache.h"
#elif __has_include("/usr/local/Ascend/ascend-toolkit/latest/include/aclnnop/aclnn_scatter_pa_kv_cache.h")
#include "/usr/local/Ascend/ascend-toolkit/latest/include/aclnnop/aclnn_scatter_pa_kv_cache.h"
#elif __has_include("/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/aclnnop/aclnn_scatter_pa_kv_cache.h")
#include "/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/aclnnop/aclnn_scatter_pa_kv_cache.h"
#else
#error "Cannot find aclnnop/aclnn_scatter_pa_kv_cache.h"
#endif

namespace xllm::kernel::npu::test {

namespace {

struct AttentionMetadata {
  torch::Tensor q_seq_lens;
  torch::Tensor kv_seq_lens;
  torch::Tensor q_cu_seq_lens;
  torch::Tensor kv_cu_seq_lens;
  torch::Tensor history_lens;
  torch::Tensor context_lens;
  torch::Tensor real_time_lens;
  torch::Tensor target_lens;
  torch::Tensor matched_prefix_lens;
  int64_t block_size = 128;
  int64_t cache_block_count = 0;
  std::vector<std::vector<int32_t>> block_tables_host;
  torch::Tensor compressed_causal_mask;
  torch::Tensor diagonal_mask;
  void* shared_workspace = nullptr;
  uint64_t shared_workspace_size = 0;
};

struct SparseParam {
  int64_t sparse_mode = 0;
  int64_t pre_tokens = 0;
  int64_t next_tokens = 0;
};

struct SegmentAttentionMetadata {
  torch::Tensor attn_mask;
  SparseParam sparse_param;
};

struct SampleSegmentMetadata {
  int64_t history = 0;
  int64_t context = 0;
  int64_t real_time = 0;
  int64_t target = 0;
  int64_t matched_prefix = 0;
  const std::vector<int32_t>* block_table_host = nullptr;
};

struct PagedAttentionMetadata {
  SegmentAttentionMetadata attn;
  torch::Tensor block_table;
  int64_t total_kv_len = 0;
  int64_t block_size = 0;
  int64_t num_kv_heads = 0;
};

struct KVCache {
  torch::Tensor key;
  torch::Tensor value;
};

class FinalBSNDTest : public ::testing::Test {
 protected:
  static constexpr int32_t kDeviceId = 5;

  static void SetUpTestSuite() {
    torch_npu::init_npu("npu:" + std::to_string(kDeviceId));
  }

  static void TearDownTestSuite() { torch_npu::finalize_npu(); }

  void SetUp() override {
    device_ = torch::Device(torch::kPrivateUse1, kDeviceId);
    fp16_opts_ = torch::TensorOptions().dtype(torch::kBFloat16).device(device_);
    stream_ = c10_npu::getCurrentNPUStream(kDeviceId).stream();
    ASSERT_NE(stream_, nullptr);
  }

  torch::Device device_{torch::kCPU};
  torch::TensorOptions fp16_opts_;
  aclrtStream stream_{nullptr};
};

bool valid_tensor(const torch::Tensor& t) { return t.defined() && t.numel() > 0; }

constexpr double kFinalRtol = 0.12;
constexpr double kFinalAtol = 0.12;
constexpr float kFinalMaxAbs = 1.5e-1f;
constexpr int64_t kDefaultWindow = 2147483647LL;

torch::Tensor create_diagonal_mask(int64_t seq_len, torch::Device device) {
  auto mask =
      ~torch::eye(seq_len, torch::TensorOptions().dtype(torch::kBool).device(device));
  return mask.view({1, 1, seq_len, seq_len}).contiguous();
}

torch::Tensor create_compressed_causal_mask_2048(torch::Device device) {
  return torch::ones({1, 1, 2048, 2048},
                     torch::TensorOptions().dtype(torch::kBool).device(device))
      .triu(1)
      .contiguous();
}

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
    const int64_t physical_block =
        block_table.at(static_cast<size_t>(logical_block));
    slots.push_back(static_cast<int32_t>(physical_block * block_size + offset));
  }
  return slots;
}

std::vector<int32_t> make_block_table(int64_t block_count) {
  static const std::vector<int32_t> kPool = {
      3,  6,  1,  8,  2,  11, 12, 15, 13, 14, 4,  5,  7,  20, 18, 10,
      21, 9,  16, 19, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33};
  CHECK_LE(block_count, static_cast<int64_t>(kPool.size()));
  return std::vector<int32_t>(kPool.begin(), kPool.begin() + block_count);
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
    case torch::kInt8:
      return ACL_INT8;
    default:
      CHECK(false) << "Unsupported dtype: " << dtype;
      return ACL_DT_UNDEFINED;
  }
}

aclTensor* torch_to_acl_tensor(const torch::Tensor& t) {
  CHECK(t.is_contiguous()) << "torch_to_acl_tensor requires contiguous tensor";
  aclDataType acl_tensor_type = to_acl_dtype(t.scalar_type());
  aclTensor* acl_t =
      aclCreateTensor(t.sizes().data(),
                      t.sizes().size(),
                      acl_tensor_type,
                      t.strides().data(),
                      /*offset=*/0,
                      ACL_FORMAT_ND,
                      t.sizes().data(),
                      t.sizes().size(),
                      t.data_ptr());
  CHECK_NE(acl_t, nullptr);
  return acl_t;
}

aclTensor* torch_to_acl_tensor_with_strides(const torch::Tensor& t) {
  aclDataType acl_tensor_type = to_acl_dtype(t.scalar_type());
  aclTensor* acl_t =
      aclCreateTensor(t.sizes().data(),
                      t.sizes().size(),
                      acl_tensor_type,
                      t.strides().data(),
                      /*offset=*/0,
                      ACL_FORMAT_ND,
                      t.sizes().data(),
                      t.sizes().size(),
                      t.data_ptr());
  CHECK_NE(acl_t, nullptr);
  return acl_t;
}

struct AclPlan {
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclTensor* q_acl = nullptr;
  aclTensor* k_acl = nullptr;
  aclTensor* v_acl = nullptr;
  aclTensor* out_acl = nullptr;
  aclTensor* mask_acl = nullptr;
  aclTensor* lse_acl = nullptr;
  aclTensorList* k_list = nullptr;
  aclTensorList* v_list = nullptr;
};

AclPlan plan_segment_attention(const torch::Tensor& query,
                               const torch::Tensor& key,
                               const torch::Tensor& value,
                               const SegmentAttentionMetadata& attn_metadata,
                               torch::Tensor& output,
                               const std::optional<torch::Tensor>& lse) {
  int64_t num_heads = query.size(2);
  int64_t num_kv_heads = key.size(2);
  float scale = 1.0f / std::sqrt(static_cast<float>(query.size(3)));
  AclPlan plan;

  plan.q_acl = torch_to_acl_tensor(query);
  plan.k_acl = torch_to_acl_tensor(key);
  plan.v_acl = torch_to_acl_tensor(value);
  plan.out_acl = torch_to_acl_tensor(output);
  if (valid_tensor(attn_metadata.attn_mask)) {
    plan.mask_acl = torch_to_acl_tensor_with_strides(attn_metadata.attn_mask);
  }
  if (lse.has_value() && valid_tensor(lse.value())) {
    plan.lse_acl = torch_to_acl_tensor(lse.value());
  }

  aclTensor* k_arr[] = {plan.k_acl};
  aclTensor* v_arr[] = {plan.v_acl};
  plan.k_list = aclCreateTensorList(k_arr, 1);
  plan.v_list = aclCreateTensorList(v_arr, 1);

  char layout[] = "BSND";
  auto ret = aclnnFusedInferAttentionScoreV3GetWorkspaceSize(
      plan.q_acl,
      plan.k_list,
      plan.v_list,
      /*pseShift=*/nullptr,
      /*attenMask=*/plan.mask_acl,
      /*actualSeqLengths=*/nullptr,
      /*actualSeqLengthsKv=*/nullptr,
      /*deqScale1=*/nullptr,
      /*quantScale1=*/nullptr,
      /*deqScale2=*/nullptr,
      /*quantScale2=*/nullptr,
      /*quantOffset2=*/nullptr,
      /*antiquantScale=*/nullptr,
      /*antiquantOffset=*/nullptr,
      /*blockTable=*/nullptr,
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
      attn_metadata.sparse_param.pre_tokens,
      attn_metadata.sparse_param.next_tokens,
      layout,
      num_kv_heads,
      attn_metadata.sparse_param.sparse_mode,
      /*innerPrecise=*/1,
      /*blockSize=*/0,
      /*antiquantMode=*/0,
      /*softmaxLseFlag=*/(lse.has_value() && valid_tensor(lse.value())),
      /*keyAntiquantMode=*/0,
      /*valueAntiquantMode=*/0,
      plan.out_acl,
      /*softmaxLse=*/plan.lse_acl,
      &plan.workspace_size,
      &plan.executor);
  CHECK_EQ(ret, ACL_SUCCESS)
      << "aclnnFusedInferAttentionScoreV3GetWorkspaceSize failed: " << ret;
  return plan;
}

void execute_planned_attention(const AclPlan& plan,
                               void* shared_workspace,
                               uint64_t shared_workspace_size,
                               aclrtStream stream) {
  void* ws_ptr = nullptr;
  if (plan.workspace_size > 0) {
    CHECK_NE(shared_workspace, nullptr)
        << "shared_workspace is required when workspace_size > 0";
    CHECK_GE(shared_workspace_size, plan.workspace_size)
        << "shared_workspace bytes(" << shared_workspace_size
        << ") is smaller than required(" << plan.workspace_size << ")";
    ws_ptr = shared_workspace;
    CHECK_EQ(aclrtMemset(ws_ptr, plan.workspace_size, 0, plan.workspace_size),
             ACL_SUCCESS)
        << "aclrtMemset attention workspace";
  }
  auto ret =
      aclnnFusedInferAttentionScoreV3(ws_ptr, plan.workspace_size, plan.executor, stream);
  CHECK_EQ(ret, ACL_SUCCESS) << "aclnnFusedInferAttentionScoreV3 failed: " << ret;
}

void destroy_planned_attention(AclPlan& plan) {
  if (plan.k_list != nullptr) {
    aclDestroyTensorList(plan.k_list);
    plan.k_list = nullptr;
    plan.k_acl = nullptr;
  }
  if (plan.v_list != nullptr) {
    aclDestroyTensorList(plan.v_list);
    plan.v_list = nullptr;
    plan.v_acl = nullptr;
  }
  if (plan.q_acl != nullptr) {
    aclDestroyTensor(plan.q_acl);
    plan.q_acl = nullptr;
  }
  if (plan.out_acl != nullptr) {
    aclDestroyTensor(plan.out_acl);
    plan.out_acl = nullptr;
  }
  if (plan.mask_acl != nullptr) {
    aclDestroyTensor(plan.mask_acl);
    plan.mask_acl = nullptr;
  }
  if (plan.lse_acl != nullptr) {
    aclDestroyTensor(plan.lse_acl);
    plan.lse_acl = nullptr;
  }
}

struct PagedAttentionPlan {
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;

  aclTensor* q_acl = nullptr;
  aclTensor* key_cache_acl = nullptr;
  aclTensor* value_cache_acl = nullptr;
  aclTensor* block_table_acl = nullptr;
  aclTensor* out_acl = nullptr;
  aclTensor* mask_acl = nullptr;
  aclTensorList* k_list = nullptr;
  aclTensorList* v_list = nullptr;
  aclIntArray* actual_q = nullptr;
  aclIntArray* actual_kv = nullptr;
};

PagedAttentionPlan plan_paged_attention_v3(const torch::Tensor& query_bsnd,
                                           const torch::Tensor& key_cache_bnbsh,
                                           const torch::Tensor& value_cache_bnbsh,
                                           const PagedAttentionMetadata& paged_metadata,
                                           torch::Tensor& output_bsnd) {
  PagedAttentionPlan plan;

  plan.q_acl = torch_to_acl_tensor(query_bsnd);
  plan.key_cache_acl = torch_to_acl_tensor(key_cache_bnbsh);
  plan.value_cache_acl = torch_to_acl_tensor(value_cache_bnbsh);
  plan.block_table_acl = torch_to_acl_tensor(paged_metadata.block_table);
  plan.out_acl = torch_to_acl_tensor(output_bsnd);
  if (valid_tensor(paged_metadata.attn.attn_mask)) {
    plan.mask_acl = torch_to_acl_tensor_with_strides(paged_metadata.attn.attn_mask);
  }

  aclTensor* k_arr[] = {plan.key_cache_acl};
  aclTensor* v_arr[] = {plan.value_cache_acl};
  plan.k_list = aclCreateTensorList(k_arr, 1);
  plan.v_list = aclCreateTensorList(v_arr, 1);
  CHECK_NE(plan.k_list, nullptr);
  CHECK_NE(plan.v_list, nullptr);

  int64_t q_seqlen[1] = {query_bsnd.size(1)};
  int64_t kv_seqlen[1] = {paged_metadata.total_kv_len};
  plan.actual_q = aclCreateIntArray(q_seqlen, 1);
  plan.actual_kv = aclCreateIntArray(kv_seqlen, 1);
  CHECK_NE(plan.actual_q, nullptr);
  CHECK_NE(plan.actual_kv, nullptr);

  const double scale = 1.0 / std::sqrt(static_cast<double>(query_bsnd.size(3)));
  const int64_t num_heads = query_bsnd.size(2);
  char layout[] = "BSND";

  auto ret = aclnnFusedInferAttentionScoreV3GetWorkspaceSize(
      plan.q_acl,
      plan.k_list,
      plan.v_list,
      /*pseShift=*/nullptr,
      /*attenMask=*/plan.mask_acl,
      /*actualSeqLengths=*/plan.actual_q,
      /*actualSeqLengthsKv=*/plan.actual_kv,
      /*deqScale1=*/nullptr,
      /*quantScale1=*/nullptr,
      /*deqScale2=*/nullptr,
      /*quantScale2=*/nullptr,
      /*quantOffset2=*/nullptr,
      /*antiquantScale=*/nullptr,
      /*antiquantOffset=*/nullptr,
      /*blockTable=*/plan.block_table_acl,
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
      paged_metadata.attn.sparse_param.pre_tokens,
      paged_metadata.attn.sparse_param.next_tokens,
      layout,
      paged_metadata.num_kv_heads,
      paged_metadata.attn.sparse_param.sparse_mode,
      /*innerPrecise=*/1,
      /*blockSize=*/paged_metadata.block_size,
      /*antiquantMode=*/0,
      /*softmaxLseFlag=*/false,
      /*keyAntiquantMode=*/0,
      /*valueAntiquantMode=*/0,
      plan.out_acl,
      /*softmaxLse=*/nullptr,
      &plan.workspace_size,
      &plan.executor);
  CHECK_EQ(ret, ACL_SUCCESS)
      << "aclnnFusedInferAttentionScoreV3GetWorkspaceSize (paged) failed: " << ret;
  return plan;
}

void execute_planned_paged_attention(const PagedAttentionPlan& plan,
                                     void* shared_workspace,
                                     uint64_t shared_workspace_size,
                                     aclrtStream stream) {
  void* ws_ptr = nullptr;
  if (plan.workspace_size > 0) {
    CHECK_NE(shared_workspace, nullptr)
        << "shared_workspace is required when workspace_size > 0";
    CHECK_GE(shared_workspace_size, plan.workspace_size)
        << "shared_workspace bytes(" << shared_workspace_size
        << ") is smaller than required(" << plan.workspace_size << ")";
    ws_ptr = shared_workspace;
    CHECK_EQ(aclrtMemset(ws_ptr, plan.workspace_size, 0, plan.workspace_size),
             ACL_SUCCESS)
        << "aclrtMemset paged-attention workspace";
  }
  auto ret = aclnnFusedInferAttentionScoreV3(
      ws_ptr, plan.workspace_size, plan.executor, stream);
  CHECK_EQ(ret, ACL_SUCCESS) << "aclnnFusedInferAttentionScoreV3 (paged) failed: " << ret;
}

void destroy_planned_paged_attention(PagedAttentionPlan& plan) {
  if (plan.actual_q != nullptr) {
    aclDestroyIntArray(plan.actual_q);
    plan.actual_q = nullptr;
  }
  if (plan.actual_kv != nullptr) {
    aclDestroyIntArray(plan.actual_kv);
    plan.actual_kv = nullptr;
  }
  if (plan.k_list != nullptr) {
    aclDestroyTensorList(plan.k_list);
    plan.k_list = nullptr;
    plan.key_cache_acl = nullptr;
  }
  if (plan.v_list != nullptr) {
    aclDestroyTensorList(plan.v_list);
    plan.v_list = nullptr;
    plan.value_cache_acl = nullptr;
  }
  if (plan.q_acl != nullptr) {
    aclDestroyTensor(plan.q_acl);
    plan.q_acl = nullptr;
  }
  if (plan.key_cache_acl != nullptr) {
    aclDestroyTensor(plan.key_cache_acl);
    plan.key_cache_acl = nullptr;
  }
  if (plan.value_cache_acl != nullptr) {
    aclDestroyTensor(plan.value_cache_acl);
    plan.value_cache_acl = nullptr;
  }
  if (plan.block_table_acl != nullptr) {
    aclDestroyTensor(plan.block_table_acl);
    plan.block_table_acl = nullptr;
  }
  if (plan.out_acl != nullptr) {
    aclDestroyTensor(plan.out_acl);
    plan.out_acl = nullptr;
  }
  if (plan.mask_acl != nullptr) {
    aclDestroyTensor(plan.mask_acl);
    plan.mask_acl = nullptr;
  }
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
      std::fprintf(stderr,
                   "aclrtMalloc scatter workspace failed: %d\n",
                   static_cast<int>(ret));
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

  ok = true;
  cleanup();
  return ok;
}

struct AttentionUpdatePlan {
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;

  int64_t b = 0;
  int64_t s = 0;
  int64_t n = 0;
  int64_t d = 0;
  torch::ScalarType out_scalar_type = torch::kFloat16;

  std::vector<torch::Tensor> local_out_flats;
  std::vector<torch::Tensor> lse_flats;
  std::vector<aclTensor*> local_out_acls;
  std::vector<aclTensor*> lse_acls;
  torch::Tensor out_flat;
  aclTensor* out_acl = nullptr;
  aclTensorList* local_out_list = nullptr;
  aclTensorList* lse_list = nullptr;
};

AttentionUpdatePlan plan_attention_update_like_benchmark(
    const std::vector<torch::Tensor>& local_outs,
    const std::vector<torch::Tensor>& lses) {
  CHECK_EQ(local_outs.size(), lses.size());
  CHECK_GE(local_outs.size(), 2UL);

  const auto& out0 = local_outs[0];
  const auto& lse0 = lses[0];
  CHECK_EQ(out0.dim(), 4);
  CHECK_EQ(lse0.dim(), 4);

  AttentionUpdatePlan plan;
  plan.b = out0.size(0);
  plan.s = out0.size(1);
  plan.n = out0.size(2);
  plan.d = out0.size(3);
  plan.out_scalar_type = out0.scalar_type();

  const int64_t b = plan.b;
  const int64_t s = plan.s;
  const int64_t n = plan.n;
  const int64_t d = plan.d;
  const int64_t bsn = b * s * n;

  for (size_t i = 1; i < local_outs.size(); ++i) {
    CHECK_EQ(local_outs[i].sizes(), out0.sizes());
    CHECK_EQ(lses[i].sizes(), lse0.sizes());
    CHECK_EQ(local_outs[i].scalar_type(), out0.scalar_type());
  }

  plan.local_out_flats.reserve(local_outs.size());
  plan.lse_flats.reserve(lses.size());
  plan.local_out_acls.assign(local_outs.size(), nullptr);
  plan.lse_acls.assign(lses.size(), nullptr);

  for (size_t i = 0; i < local_outs.size(); ++i) {
    CHECK(local_outs[i].is_contiguous())
        << "plan_attention_update_like_benchmark expects contiguous local_out";
    plan.local_out_flats.push_back(local_outs[i].view({bsn, d}).to(torch::kFloat32));
    plan.lse_flats.push_back(
        lses[i].permute({0, 2, 1, 3}).contiguous().view({bsn}));
    plan.local_out_acls[i] = torch_to_acl_tensor(plan.local_out_flats.back());
    plan.lse_acls[i] = torch_to_acl_tensor(plan.lse_flats.back());
  }

  plan.out_flat =
      torch::empty({bsn, d},
                   torch::TensorOptions().dtype(torch::kFloat32).device(out0.device()));
  plan.out_acl = torch_to_acl_tensor(plan.out_flat);

  plan.local_out_list = aclCreateTensorList(
      plan.local_out_acls.data(),
      static_cast<int32_t>(plan.local_out_acls.size()));
  CHECK_NE(plan.local_out_list, nullptr);
  plan.lse_list = aclCreateTensorList(
      plan.lse_acls.data(),
      static_cast<int32_t>(plan.lse_acls.size()));
  CHECK_NE(plan.lse_list, nullptr);

  auto ret = aclnnAttentionUpdateGetWorkspaceSize(
      plan.lse_list,
      plan.local_out_list,
      /*updateType=*/0,
      plan.out_acl,
      /*lseOut=*/nullptr,
      &plan.workspace_size,
      &plan.executor);
  CHECK_EQ(ret, ACL_SUCCESS)
      << "aclnnAttentionUpdateGetWorkspaceSize failed: " << ret;
  return plan;
}

void execute_planned_attention_update(const AttentionUpdatePlan& plan,
                                      void* shared_workspace,
                                      uint64_t shared_workspace_size,
                                      aclrtStream stream) {
  void* ws_ptr = nullptr;
  if (plan.workspace_size > 0) {
    CHECK_NE(shared_workspace, nullptr)
        << "shared_workspace is required when workspace_size > 0";
    CHECK_GE(shared_workspace_size, plan.workspace_size)
        << "shared_workspace bytes(" << shared_workspace_size
        << ") is smaller than required(" << plan.workspace_size << ")";
    ws_ptr = shared_workspace;
    CHECK_EQ(aclrtMemset(ws_ptr, plan.workspace_size, 0, plan.workspace_size),
             ACL_SUCCESS)
        << "aclrtMemset attention-update workspace";
  }

  auto ret = aclnnAttentionUpdate(ws_ptr, plan.workspace_size, plan.executor, stream);
  CHECK_EQ(ret, ACL_SUCCESS) << "aclnnAttentionUpdate failed: " << ret;
}

void destroy_planned_attention_update(AttentionUpdatePlan& plan) {
  if (plan.local_out_list != nullptr) {
    aclDestroyTensorList(plan.local_out_list);
    plan.local_out_list = nullptr;
  }
  if (plan.lse_list != nullptr) {
    aclDestroyTensorList(plan.lse_list);
    plan.lse_list = nullptr;
  }
  if (plan.out_acl != nullptr) {
    aclDestroyTensor(plan.out_acl);
    plan.out_acl = nullptr;
  }
}

torch::Tensor reference_attention_bnsd(const torch::Tensor& q_bnsd,
                                       const torch::Tensor& k_bnsd,
                                       const torch::Tensor& v_bnsd,
                                       const torch::Tensor* mask_sq_sk,
                                       double scale) {
  CHECK_EQ(q_bnsd.dim(), 4);
  auto qf = q_bnsd;
  auto kf = k_bnsd;
  auto vf = v_bnsd;
  const int64_t num_heads = qf.size(1);
  const int64_t num_kv = kf.size(1);
  if (num_kv < num_heads) {
    const int64_t group = num_heads / num_kv;
    kf = kf.repeat_interleave(group, 1);
    vf = vf.repeat_interleave(group, 1);
  }
  auto scores = torch::matmul(qf, kf.transpose(-2, -1)) * scale;
  if (mask_sq_sk != nullptr) {
    scores = scores.masked_fill(*mask_sq_sk, -1e9f);
  }
  auto weights = torch::softmax(scores, -1);
  return torch::matmul(weights, vf);
}

torch::Tensor build_reference_output_bnsd_cpu_f32(const torch::Tensor& query_bsnd,
                                                   const torch::Tensor& key_bsnd,
                                                   const torch::Tensor& value_bsnd,
                                                   int64_t h,
                                                   int64_t c,
                                                   int64_t r,
                                                   int64_t t) {
  CHECK_EQ(query_bsnd.size(0), 1);
  const double scale = 1.0 / std::sqrt(static_cast<double>(query_bsnd.size(3)));

  auto q_seq = query_bsnd.select(0, 0).to(torch::kCPU).to(torch::kFloat32);
  auto k_seq = key_bsnd.select(0, 0).to(torch::kCPU).to(torch::kFloat32);
  auto v_seq = value_bsnd.select(0, 0).to(torch::kCPU).to(torch::kFloat32);

  auto seq_to_bnsd = [](const torch::Tensor& seq_s_n_d) {
    return seq_s_n_d.unsqueeze(0).permute({0, 2, 1, 3}).contiguous();
  };

  auto q = seq_to_bnsd(q_seq);
  auto k = seq_to_bnsd(k_seq);
  auto v = seq_to_bnsd(v_seq);
  const int64_t nheads = q.size(1);
  const int64_t d = q.size(3);

  auto qh = q.slice(2, 0, h);
  auto kh = k.slice(2, 0, h);
  auto vh = v.slice(2, 0, h);
  auto mask_h = torch::triu(torch::ones({h, h}, torch::dtype(torch::kBool)), 1);
  auto history = reference_attention_bnsd(qh, kh, vh, &mask_h, scale);

  auto q_ctx = q.slice(2, h, h + c);
  auto k_hc = k.slice(2, 0, h + c);
  auto v_hc = v.slice(2, 0, h + c);
  auto context = reference_attention_bnsd(q_ctx, k_hc, v_hc, nullptr, scale);

  auto real_time = torch::empty({1, nheads, r, d}, q.options());
  for (int64_t i = 0; i < r; ++i) {
    const int64_t nkv = h + c + i + 1;
    auto qi = q.slice(2, h + c + i, h + c + i + 1);
    auto kk = k.slice(2, 0, nkv);
    auto vv = v.slice(2, 0, nkv);
    auto oi = reference_attention_bnsd(qi, kk, vv, nullptr, scale);
    real_time.narrow(2, i, 1).copy_(oi);
  }

  auto target = torch::empty({1, nheads, t, d}, q.options());
  const int64_t prefix = h + c + r;
  for (int64_t j = 0; j < t; ++j) {
    auto qj = q.slice(2, prefix + j, prefix + j + 1);
    auto k_prefix = k.slice(2, 0, prefix);
    auto k_self = k.slice(2, prefix + j, prefix + j + 1);
    auto k_cat = torch::cat({k_prefix, k_self}, 2);
    auto v_prefix = v.slice(2, 0, prefix);
    auto v_self = v.slice(2, prefix + j, prefix + j + 1);
    auto v_cat = torch::cat({v_prefix, v_self}, 2);
    auto oj = reference_attention_bnsd(qj, k_cat, v_cat, nullptr, scale);
    target.narrow(2, j, 1).copy_(oj);
  }

  auto ref_bnsd = torch::empty_like(q);
  ref_bnsd.slice(2, 0, h).copy_(history);
  ref_bnsd.slice(2, h, h + c).copy_(context);
  ref_bnsd.slice(2, h + c, h + c + r).copy_(real_time);
  ref_bnsd.slice(2, h + c + r, h + c + r + t).copy_(target);
  return ref_bnsd;
}

torch::Tensor build_reference_output_bsnd_cpu_f32(const torch::Tensor& query_bsnd,
                                                   const torch::Tensor& key_bsnd,
                                                   const torch::Tensor& value_bsnd,
                                                   int64_t h,
                                                   int64_t c,
                                                   int64_t r,
                                                   int64_t t) {
  auto ref_bnsd =
      build_reference_output_bnsd_cpu_f32(query_bsnd, key_bsnd, value_bsnd, h, c, r, t);
  return ref_bnsd.permute({0, 2, 1, 3}).contiguous();
}

void expect_output_matches_reference(const char* tag,
                                     const torch::Tensor& output_bsnd,
                                     const torch::Tensor& query_bsnd,
                                     const torch::Tensor& key_bsnd,
                                     const torch::Tensor& value_bsnd,
                                     int64_t h,
                                     int64_t c,
                                     int64_t r,
                                     int64_t t) {
  auto got_bnsd =
      output_bsnd.permute({0, 2, 1, 3}).to(torch::kCPU).to(torch::kFloat32).contiguous();
  auto ref_bnsd =
      build_reference_output_bnsd_cpu_f32(query_bsnd, key_bsnd, value_bsnd, h, c, r, t);

  auto diff = (got_bnsd - ref_bnsd).abs();
  const double max_abs = diff.max().item<double>();
  const double mean_abs = diff.mean().item<double>();
  const double rmse = diff.pow(2).mean().sqrt().item<double>();
  const bool allclose = torch::allclose(got_bnsd, ref_bnsd, kFinalRtol, kFinalAtol);

  std::fprintf(stderr,
               "[GenRecV2][PA][%s] max_abs=%.6e mean_abs=%.6e rmse=%.6e "
               "allclose(rtol=%.3f,atol=%.3f)=%s\n",
               tag,
               max_abs,
               mean_abs,
               rmse,
               kFinalRtol,
               kFinalAtol,
               allclose ? "PASS" : "FAIL");
  std::fflush(stderr);

  EXPECT_LT(static_cast<float>(max_abs), kFinalMaxAbs);
  EXPECT_TRUE(allclose);
}

void run_fa_non_history_segments(const torch::Tensor& query,
                                 const torch::Tensor& key,
                                 const torch::Tensor& value,
                                 int64_t h,
                                 int64_t c,
                                 int64_t r,
                                 int64_t t,
                                 const AttentionMetadata& attn_metadata,
                                 torch::Tensor& output) {
  CHECK_EQ(query.size(0), 1);
  const int64_t num_heads = query.size(2);
  const int64_t head_dim = query.size(3);
  auto stream = c10_npu::getCurrentNPUStream(query.device().index()).stream();

  auto out_opts = torch::TensorOptions().dtype(query.scalar_type()).device(query.device());
  auto lse_opts = torch::TensorOptions().dtype(torch::kFloat32).device(query.device());
  constexpr float kSentinel = -31415.0f;

  auto make_out = [&](int64_t seq_len) {
    return torch::full({1, seq_len, num_heads, head_dim}, kSentinel, out_opts);
  };
  auto make_lse = [&](int64_t seq_len) {
    return torch::full({1, num_heads, seq_len, 1}, kSentinel, lse_opts);
  };

  auto crt_out = make_out(c + r + t);
  auto crt_lse = make_lse(c + r + t);
  auto crt_query = query.slice(1, h, h + c + r + t).contiguous();
  auto crt_key = key.slice(1, 0, h + c).contiguous();
  auto crt_value = value.slice(1, 0, h + c).contiguous();

  SegmentAttentionMetadata crt_meta;
  crt_meta.sparse_param.sparse_mode = 0;
  crt_meta.sparse_param.pre_tokens = kDefaultWindow;
  crt_meta.sparse_param.next_tokens = kDefaultWindow;

  AclPlan crt_plan = plan_segment_attention(
      crt_query, crt_key, crt_value, crt_meta, crt_out, crt_lse);
  execute_planned_attention(
      crt_plan,
      attn_metadata.shared_workspace,
      attn_metadata.shared_workspace_size,
      stream);
  destroy_planned_attention(crt_plan);
  output.slice(1, h, h + c).copy_(crt_out.slice(1, 0, c));

  auto rt_out = make_out(r + t);
  auto rt_lse = make_lse(r + t);
  auto rt_query = query.slice(1, h + c, h + c + r + t).contiguous();
  auto rt_key = key.slice(1, h + c, h + c + r).contiguous();
  auto rt_value = value.slice(1, h + c, h + c + r).contiguous();

  SegmentAttentionMetadata rt_meta;
  rt_meta.attn_mask = attn_metadata.compressed_causal_mask;
  rt_meta.sparse_param.sparse_mode = 2;
  rt_meta.sparse_param.pre_tokens = kDefaultWindow;
  rt_meta.sparse_param.next_tokens = kDefaultWindow;

  AclPlan rt_plan =
      plan_segment_attention(rt_query, rt_key, rt_value, rt_meta, rt_out, rt_lse);
  execute_planned_attention(
      rt_plan,
      attn_metadata.shared_workspace,
      attn_metadata.shared_workspace_size,
      stream);
  destroy_planned_attention(rt_plan);

  auto target_out = make_out(t);
  auto target_lse = make_lse(t);
  auto target_query = query.slice(1, h + c + r, h + c + r + t).contiguous();
  auto target_key = key.slice(1, h + c + r, h + c + r + t).contiguous();
  auto target_value = value.slice(1, h + c + r, h + c + r + t).contiguous();

  SegmentAttentionMetadata target_meta;
  target_meta.attn_mask = attn_metadata.diagonal_mask;
  target_meta.sparse_param.sparse_mode = 0;
  target_meta.sparse_param.pre_tokens = 0;
  target_meta.sparse_param.next_tokens = 0;

  AclPlan target_plan = plan_segment_attention(
      target_query, target_key, target_value, target_meta, target_out, target_lse);
  execute_planned_attention(
      target_plan,
      attn_metadata.shared_workspace,
      attn_metadata.shared_workspace_size,
      stream);
  destroy_planned_attention(target_plan);

  auto rt_update_plan = plan_attention_update_like_benchmark(
      {crt_out.slice(1, c, c + r).contiguous(), rt_out.slice(1, 0, r).contiguous()},
      {crt_lse.slice(2, c, c + r).contiguous(), rt_lse.slice(2, 0, r).contiguous()});
  execute_planned_attention_update(rt_update_plan,
                                   attn_metadata.shared_workspace,
                                   attn_metadata.shared_workspace_size,
                                   stream);
  auto rt_merged = rt_update_plan.out_flat.view({rt_update_plan.b,
                                                 rt_update_plan.s,
                                                 rt_update_plan.n,
                                                 rt_update_plan.d})
                       .to(rt_update_plan.out_scalar_type);
  destroy_planned_attention_update(rt_update_plan);
  output.slice(1, h + c, h + c + r).copy_(rt_merged);

  auto target_update_plan = plan_attention_update_like_benchmark(
      {crt_out.slice(1, c + r, c + r + t).contiguous(),
       rt_out.slice(1, r, r + t).contiguous(),
       target_out.contiguous()},
      {crt_lse.slice(2, c + r, c + r + t).contiguous(),
       rt_lse.slice(2, r, r + t).contiguous(),
       target_lse.contiguous()});
  execute_planned_attention_update(target_update_plan,
                                   attn_metadata.shared_workspace,
                                   attn_metadata.shared_workspace_size,
                                   stream);
  auto target_merged = target_update_plan.out_flat.view({target_update_plan.b,
                                                         target_update_plan.s,
                                                         target_update_plan.n,
                                                         target_update_plan.d})
                           .to(target_update_plan.out_scalar_type);
  destroy_planned_attention_update(target_update_plan);
  output.slice(1, h + c + r, h + c + r + t).copy_(target_merged);
}

void run_fa_all_segments(const torch::Tensor& query,
                         const torch::Tensor& key,
                         const torch::Tensor& value,
                         const AttentionMetadata& attn_metadata,
                         const SampleSegmentMetadata& sample_metadata,
                         torch::Tensor& output) {
  CHECK_EQ(query.size(0), 1);
  const int64_t h = sample_metadata.history;
  const int64_t c = sample_metadata.context;
  const int64_t r = sample_metadata.real_time;
  const int64_t t = sample_metadata.target;
  const int64_t num_heads = query.size(2);
  const int64_t head_dim = query.size(3);
  auto stream = c10_npu::getCurrentNPUStream(query.device().index()).stream();

  auto out_opts = torch::TensorOptions().dtype(query.scalar_type()).device(query.device());
  constexpr float kSentinel = -31415.0f;
  auto hist_out = torch::full({1, h, num_heads, head_dim}, kSentinel, out_opts);

  SegmentAttentionMetadata hist_meta;
  hist_meta.attn_mask = attn_metadata.compressed_causal_mask;
  hist_meta.sparse_param.sparse_mode = 2;
  hist_meta.sparse_param.pre_tokens = kDefaultWindow;
  hist_meta.sparse_param.next_tokens = kDefaultWindow;

  auto hist_query = query.slice(1, 0, h);
  auto hist_key = key.slice(1, 0, h);
  auto hist_value = value.slice(1, 0, h);
  AclPlan hist_plan = plan_segment_attention(
      hist_query, hist_key, hist_value, hist_meta, hist_out, std::nullopt);
  execute_planned_attention(
      hist_plan,
      attn_metadata.shared_workspace,
      attn_metadata.shared_workspace_size,
      stream);
  destroy_planned_attention(hist_plan);
  output.slice(1, 0, h).copy_(hist_out);

  run_fa_non_history_segments(query, key, value, h, c, r, t, attn_metadata, output);
}

void run_case1_history_partial_match_pa_history_fa_tail(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    KVCache& kv_cache,
    const AttentionMetadata& attn_metadata,
    const SampleSegmentMetadata& sample_metadata,
    torch::Tensor& output) {
  CHECK_EQ(query.size(0), 1);
  CHECK_NE(sample_metadata.block_table_host, nullptr);

  const int64_t h = sample_metadata.history;
  const int64_t c = sample_metadata.context;
  const int64_t r = sample_metadata.real_time;
  const int64_t t = sample_metadata.target;
  const int64_t history_matched = sample_metadata.matched_prefix;
  const int64_t block_size = attn_metadata.block_size;
  const int64_t cache_block_count = attn_metadata.cache_block_count;
  const auto& block_table_host = *sample_metadata.block_table_host;
  const int64_t num_kv_heads = key.size(2);
  const int64_t head_dim = key.size(3);
  const int64_t history_unmatched = h - history_matched;
  CHECK_GT(history_unmatched, 0);
  auto& key_cache = kv_cache.key;
  auto& value_cache = kv_cache.value;

  auto stream = c10_npu::getCurrentNPUStream(query.device().index()).stream();
  auto i32_dev_opts = torch::TensorOptions().dtype(torch::kInt32).device(query.device());
  auto i64_dev_opts = torch::TensorOptions().dtype(torch::kInt64).device(query.device());

  auto reference_bsnd = build_reference_output_bsnd_cpu_f32(query, key, value, h, c, r, t)
                            .to(query.device())
                            .to(query.scalar_type());

  CHECK_EQ(key_cache.dim(), 4);
  CHECK_EQ(value_cache.dim(), 4);
  CHECK_EQ(key_cache.size(0), cache_block_count);
  CHECK_EQ(key_cache.size(1), block_size);
  CHECK_EQ(key_cache.size(2), num_kv_heads);
  CHECK_EQ(key_cache.size(3), head_dim);
  CHECK_EQ(value_cache.size(0), cache_block_count);
  CHECK_EQ(value_cache.size(1), block_size);
  CHECK_EQ(value_cache.size(2), num_kv_heads);
  CHECK_EQ(value_cache.size(3), head_dim);
  CHECK_EQ(key_cache.scalar_type(), key.scalar_type());
  CHECK_EQ(value_cache.scalar_type(), value.scalar_type());
  CHECK_EQ(key_cache.device(), key.device());
  CHECK_EQ(value_cache.device(), value.device());

  key_cache.zero_();
  value_cache.zero_();
  auto key_cache_flat =
      key_cache.view({cache_block_count * block_size, num_kv_heads, head_dim});
  auto value_cache_flat =
      value_cache.view({cache_block_count * block_size, num_kv_heads, head_dim});

  auto key_seq = key.select(0, 0).contiguous();
  auto value_seq = value.select(0, 0).contiguous();
  auto prefix_slots_host = build_slot_mapping(
      block_table_host, block_size, /*start_token_idx=*/0, history_matched);
  auto prefix_slots_dev_i64 = torch::tensor(prefix_slots_host, i64_dev_opts);
  key_cache_flat.index_copy_(0, prefix_slots_dev_i64, key_seq.slice(0, 0, history_matched));
  value_cache_flat.index_copy_(
      0, prefix_slots_dev_i64, value_seq.slice(0, 0, history_matched));

  auto new_slots_host = build_slot_mapping(block_table_host,
                                           block_size,
                                           /*start_token_idx=*/history_matched,
                                           history_unmatched);
  auto new_slots_dev_i32 = torch::tensor(new_slots_host, i32_dev_opts);
  auto new_key =
      key_seq.slice(0, history_matched, history_matched + history_unmatched).contiguous();
  auto new_value =
      value_seq.slice(0, history_matched, history_matched + history_unmatched).contiguous();

  CHECK(run_scatter_pa_kv_cache(
      new_key, new_value, key_cache, value_cache, new_slots_dev_i32, stream))
      << "run_scatter_pa_kv_cache failed";

  auto key_cache_bnbsh =
      key_cache.view({cache_block_count, block_size, num_kv_heads * head_dim});
  auto value_cache_bnbsh =
      value_cache.view({cache_block_count, block_size, num_kv_heads * head_dim});
  auto block_table = torch::tensor(block_table_host, i32_dev_opts)
                         .view({1, static_cast<int64_t>(block_table_host.size())});

  auto history_query_unmatched = query.slice(1, history_matched, h).contiguous();
  auto history_out_unmatched = torch::empty_like(history_query_unmatched);
  PagedAttentionMetadata pa_hist_meta;
  pa_hist_meta.attn.attn_mask = attn_metadata.compressed_causal_mask;
  pa_hist_meta.attn.sparse_param.sparse_mode = 3;
  pa_hist_meta.attn.sparse_param.pre_tokens = kDefaultWindow;
  pa_hist_meta.attn.sparse_param.next_tokens = kDefaultWindow;
  pa_hist_meta.block_table = block_table;
  pa_hist_meta.total_kv_len = h;
  pa_hist_meta.block_size = block_size;
  pa_hist_meta.num_kv_heads = num_kv_heads;
  auto hist_pa_plan = plan_paged_attention_v3(
      history_query_unmatched, key_cache_bnbsh, value_cache_bnbsh, pa_hist_meta, history_out_unmatched);
  execute_planned_paged_attention(
      hist_pa_plan, attn_metadata.shared_workspace, attn_metadata.shared_workspace_size, stream);
  destroy_planned_paged_attention(hist_pa_plan);

  output.copy_(torch::full_like(output, -12345.0f));
  output.slice(1, 0, history_matched)
      .copy_(reference_bsnd.slice(1, 0, history_matched));
  output.slice(1, history_matched, h).copy_(history_out_unmatched);

  run_fa_non_history_segments(query, key, value, h, c, r, t, attn_metadata, output);
}

void run_case2_hc_full_rt_partial_match_pa_rt_fa_target_two_way(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    KVCache& kv_cache,
    const AttentionMetadata& attn_metadata,
    const SampleSegmentMetadata& sample_metadata,
    torch::Tensor& output) {
  CHECK_EQ(query.size(0), 1);
  CHECK_NE(sample_metadata.block_table_host, nullptr);

  const int64_t h = sample_metadata.history;
  const int64_t c = sample_metadata.context;
  const int64_t r = sample_metadata.real_time;
  const int64_t t = sample_metadata.target;
  const int64_t block_size = attn_metadata.block_size;
  const int64_t cache_block_count = attn_metadata.cache_block_count;
  const auto& block_table_host = *sample_metadata.block_table_host;
  CHECK_GE(sample_metadata.matched_prefix, h + c);
  const int64_t realtime_matched = sample_metadata.matched_prefix - (h + c);
  CHECK_LT(realtime_matched, r);

  const int64_t num_kv_heads = key.size(2);
  const int64_t head_dim = key.size(3);
  const int64_t rt_unmatched = r - realtime_matched;
  const int64_t matched_prefix = h + c + realtime_matched;
  const int64_t rt_unmatched_start = h + c + realtime_matched;
  auto& key_cache = kv_cache.key;
  auto& value_cache = kv_cache.value;

  auto stream = c10_npu::getCurrentNPUStream(query.device().index()).stream();
  auto i32_dev_opts = torch::TensorOptions().dtype(torch::kInt32).device(query.device());
  auto i64_dev_opts = torch::TensorOptions().dtype(torch::kInt64).device(query.device());

  auto reference_bsnd = build_reference_output_bsnd_cpu_f32(query, key, value, h, c, r, t)
                            .to(query.device())
                            .to(query.scalar_type());

  CHECK_EQ(key_cache.dim(), 4);
  CHECK_EQ(value_cache.dim(), 4);
  CHECK_EQ(key_cache.size(0), cache_block_count);
  CHECK_EQ(key_cache.size(1), block_size);
  CHECK_EQ(key_cache.size(2), num_kv_heads);
  CHECK_EQ(key_cache.size(3), head_dim);
  CHECK_EQ(value_cache.size(0), cache_block_count);
  CHECK_EQ(value_cache.size(1), block_size);
  CHECK_EQ(value_cache.size(2), num_kv_heads);
  CHECK_EQ(value_cache.size(3), head_dim);
  CHECK_EQ(key_cache.scalar_type(), key.scalar_type());
  CHECK_EQ(value_cache.scalar_type(), value.scalar_type());
  CHECK_EQ(key_cache.device(), key.device());
  CHECK_EQ(value_cache.device(), value.device());

  key_cache.zero_();
  value_cache.zero_();
  auto key_cache_flat =
      key_cache.view({cache_block_count * block_size, num_kv_heads, head_dim});
  auto value_cache_flat =
      value_cache.view({cache_block_count * block_size, num_kv_heads, head_dim});

  auto key_seq = key.select(0, 0).contiguous();
  auto value_seq = value.select(0, 0).contiguous();
  auto prefix_slots_host = build_slot_mapping(
      block_table_host, block_size, /*start_token_idx=*/0, matched_prefix);
  auto prefix_slots_dev_i64 = torch::tensor(prefix_slots_host, i64_dev_opts);
  key_cache_flat.index_copy_(0, prefix_slots_dev_i64, key_seq.slice(0, 0, matched_prefix));
  value_cache_flat.index_copy_(
      0, prefix_slots_dev_i64, value_seq.slice(0, 0, matched_prefix));

  auto new_slots_host = build_slot_mapping(block_table_host,
                                           block_size,
                                           /*start_token_idx=*/rt_unmatched_start,
                                           rt_unmatched);
  auto new_slots_dev_i32 = torch::tensor(new_slots_host, i32_dev_opts);
  auto new_key = key_seq.slice(0, rt_unmatched_start, rt_unmatched_start + rt_unmatched)
                     .contiguous();
  auto new_value = value_seq.slice(0, rt_unmatched_start, rt_unmatched_start + rt_unmatched)
                       .contiguous();

  CHECK(run_scatter_pa_kv_cache(
      new_key, new_value, key_cache, value_cache, new_slots_dev_i32, stream))
      << "run_scatter_pa_kv_cache failed";

  auto key_cache_bnbsh =
      key_cache.view({cache_block_count, block_size, num_kv_heads * head_dim});
  auto value_cache_bnbsh =
      value_cache.view({cache_block_count, block_size, num_kv_heads * head_dim});
  auto block_table = torch::tensor(block_table_host, i32_dev_opts)
                         .view({1, static_cast<int64_t>(block_table_host.size())});

  auto rt_query_unmatched = query.slice(1, rt_unmatched_start, h + c + r).contiguous();
  auto rt_out_unmatched = torch::empty_like(rt_query_unmatched);
  PagedAttentionMetadata pa_rt_meta;
  pa_rt_meta.attn.attn_mask = attn_metadata.compressed_causal_mask;
  pa_rt_meta.attn.sparse_param.sparse_mode = 3;
  pa_rt_meta.attn.sparse_param.pre_tokens = kDefaultWindow;
  pa_rt_meta.attn.sparse_param.next_tokens = kDefaultWindow;
  pa_rt_meta.block_table = block_table;
  pa_rt_meta.total_kv_len = h + c + r;
  pa_rt_meta.block_size = block_size;
  pa_rt_meta.num_kv_heads = num_kv_heads;
  auto rt_pa_plan = plan_paged_attention_v3(
      rt_query_unmatched, key_cache_bnbsh, value_cache_bnbsh, pa_rt_meta, rt_out_unmatched);
  execute_planned_paged_attention(
      rt_pa_plan, attn_metadata.shared_workspace, attn_metadata.shared_workspace_size, stream);
  destroy_planned_paged_attention(rt_pa_plan);

  output.copy_(torch::full_like(output, -12345.0f));
  output.slice(1, 0, matched_prefix).copy_(reference_bsnd.slice(1, 0, matched_prefix));
  output.slice(1, rt_unmatched_start, h + c + r).copy_(rt_out_unmatched);

  const int64_t num_heads = query.size(2);
  auto out_opts = torch::TensorOptions().dtype(query.scalar_type()).device(query.device());
  auto lse_opts = torch::TensorOptions().dtype(torch::kFloat32).device(query.device());
  constexpr float kSentinel = -31415.0f;
  auto make_out = [&](int64_t seq_len) {
    return torch::full({1, seq_len, num_heads, head_dim}, kSentinel, out_opts);
  };
  auto make_lse = [&](int64_t seq_len) {
    return torch::full({1, num_heads, seq_len, 1}, kSentinel, lse_opts);
  };

  auto target_query = query.slice(1, h + c + r, h + c + r + t);
  auto prefix_key = key.slice(1, 0, h + c + r);
  auto prefix_value = value.slice(1, 0, h + c + r);
  auto target_key = key.slice(1, h + c + r, h + c + r + t);
  auto target_value = value.slice(1, h + c + r, h + c + r + t);

  auto out_prefix = make_out(t);
  auto lse_prefix = make_lse(t);
  SegmentAttentionMetadata prefix_meta;
  prefix_meta.sparse_param.sparse_mode = 0;
  prefix_meta.sparse_param.pre_tokens = kDefaultWindow;
  prefix_meta.sparse_param.next_tokens = kDefaultWindow;
  auto prefix_plan = plan_segment_attention(
      target_query, prefix_key, prefix_value, prefix_meta, out_prefix, lse_prefix);
  execute_planned_attention(
      prefix_plan,
      attn_metadata.shared_workspace,
      attn_metadata.shared_workspace_size,
      stream);
  destroy_planned_attention(prefix_plan);

  auto out_self = make_out(t);
  auto lse_self = make_lse(t);
  SegmentAttentionMetadata self_meta;
  self_meta.attn_mask = attn_metadata.diagonal_mask;
  self_meta.sparse_param.sparse_mode = 0;
  self_meta.sparse_param.pre_tokens = 0;
  self_meta.sparse_param.next_tokens = 0;
  auto self_plan = plan_segment_attention(
      target_query, target_key, target_value, self_meta, out_self, lse_self);
  execute_planned_attention(
      self_plan,
      attn_metadata.shared_workspace,
      attn_metadata.shared_workspace_size,
      stream);
  destroy_planned_attention(self_plan);

  auto update_plan = plan_attention_update_like_benchmark(
      {out_prefix.contiguous(), out_self.contiguous()},
      {lse_prefix.contiguous(), lse_self.contiguous()});
  execute_planned_attention_update(update_plan,
                                   attn_metadata.shared_workspace,
                                   attn_metadata.shared_workspace_size,
                                   stream);
  auto target_merged =
      update_plan.out_flat
          .view({update_plan.b, update_plan.s, update_plan.n, update_plan.d})
          .to(update_plan.out_scalar_type);
  destroy_planned_attention_update(update_plan);
  output.slice(1, h + c + r, h + c + r + t).copy_(target_merged);
}

class MTGRAttentionTestImpl {
 public:
  MTGRAttentionTestImpl(int64_t num_heads,
                        int64_t head_size,
                        float scale,
                        int64_t num_kv_heads)
      : num_heads_(num_heads),
        head_size_(head_size),
        scale_(scale),
        num_kv_heads_(num_kv_heads) {}

  std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward(
      const AttentionMetadata& attn_metadata,
      torch::Tensor& query,
      torch::Tensor& key,
      torch::Tensor& value,
      KVCache& kv_cache) {
    CHECK_EQ(query.dim(), 2);
    CHECK_EQ(key.dim(), 2);
    CHECK_EQ(value.dim(), 2);
    CHECK_EQ(query.size(0), key.size(0));
    CHECK_EQ(query.size(0), value.size(0));
    CHECK_EQ(query.size(1), num_heads_ * head_size_);
    CHECK_EQ(key.size(1), num_kv_heads_ * head_size_);
    CHECK_EQ(value.size(1), num_kv_heads_ * head_size_);
    CHECK_GT(scale_, 0.0f);

    torch::Tensor output = torch::empty_like(query);
    std::optional<torch::Tensor> output_lse = std::nullopt;

    const int64_t q_tokens = query.size(0);
    const int64_t kv_tokens = key.size(0);

    const auto& q_seq_lens_host = attn_metadata.q_seq_lens;
    const auto& kv_seq_lens_host = attn_metadata.kv_seq_lens;
    const auto& history_lens_host = attn_metadata.history_lens;
    const auto& context_lens_host = attn_metadata.context_lens;
    const auto& real_time_lens_host = attn_metadata.real_time_lens;
    const auto& target_lens_host = attn_metadata.target_lens;
    const auto& matched_prefix_lens_host = attn_metadata.matched_prefix_lens;

    const int64_t batch_size = q_seq_lens_host.size(0);

    int64_t q_offset = 0;
    int64_t kv_offset = 0;
    for (int64_t i = 0; i < batch_size; ++i) {
      const int64_t q_len = q_seq_lens_host[i].item<int64_t>();
      const int64_t kv_len = kv_seq_lens_host[i].item<int64_t>();
      CHECK_GE(q_len, 0);
      CHECK_GE(kv_len, 0);
      CHECK_LE(q_offset + q_len, q_tokens);
      CHECK_LE(kv_offset + kv_len, kv_tokens);
      if (q_len == 0) {
        kv_offset += kv_len;
        continue;
      }

      auto query_i =
          query.narrow(0, q_offset, q_len).view({1, q_len, num_heads_, head_size_});
      auto key_i = key.narrow(0, kv_offset, kv_len)
                       .view({1, kv_len, num_kv_heads_, head_size_});
      auto value_i = value.narrow(0, kv_offset, kv_len)
                         .view({1, kv_len, num_kv_heads_, head_size_});
      auto output_i = torch::empty_like(query_i);

      SampleSegmentMetadata sample_metadata;
      sample_metadata.history = history_lens_host[i].item<int64_t>();
      sample_metadata.context = context_lens_host[i].item<int64_t>();
      sample_metadata.real_time = real_time_lens_host[i].item<int64_t>();
      sample_metadata.target = target_lens_host[i].item<int64_t>();
      sample_metadata.matched_prefix = matched_prefix_lens_host[i].item<int64_t>();
      sample_metadata.block_table_host =
          &attn_metadata.block_tables_host[static_cast<size_t>(i)];

      const int64_t h = sample_metadata.history;
      const int64_t c = sample_metadata.context;
      const int64_t r = sample_metadata.real_time;
      const int64_t matched = sample_metadata.matched_prefix;
      CHECK_GE(matched, 0);
      CHECK_LT(matched, h + c + r)
          << "matched prefix must be in [0, history+context+realtime)";

      if (matched == 0) {
        run_fa_all_segments(query_i, key_i, value_i, attn_metadata, sample_metadata, output_i);
      } else if (matched < h) {
        run_case1_history_partial_match_pa_history_fa_tail(
            query_i, key_i, value_i, kv_cache, attn_metadata, sample_metadata, output_i);
      } else if (matched >= h + c && matched < h + c + r) {
        run_case2_hc_full_rt_partial_match_pa_rt_fa_target_two_way(
            query_i, key_i, value_i, kv_cache, attn_metadata, sample_metadata, output_i);
      } else {
        CHECK(false) << "Unsupported matched_prefix: matched=" << matched
                     << ", history=" << h << ", context=" << c
                     << ", realtime=" << r
                     << " (context partial-match is not supported)";
      }

      output.narrow(0, q_offset, q_len)
          .copy_(output_i.view({q_len, num_heads_ * head_size_}));
      q_offset += q_len;
      kv_offset += kv_len;
    }

    CHECK_EQ(q_offset, q_tokens);
    CHECK_EQ(kv_offset, kv_tokens);
    return {output, output_lse};
  }

 private:
  int64_t num_heads_ = 0;
  int64_t head_size_ = 0;
  float scale_ = 1.0f;
  int64_t num_kv_heads_ = 0;
};

class CustomMaskAttentionTestImpl {
 public:
  CustomMaskAttentionTestImpl(int64_t num_heads,
                              int64_t num_kv_heads,
                              int64_t head_dim,
                              int64_t hidden_size)
      : num_heads_(num_heads),
        num_kv_heads_(num_kv_heads),
        head_dim_(head_dim),
        hidden_size_(hidden_size),
        q_size_(num_heads * head_dim),
        kv_size_(num_kv_heads * head_dim),
        scale_(1.0f / std::sqrt(static_cast<float>(head_dim))),
        attn_(num_heads, head_dim, scale_, num_kv_heads) {}

  torch::Tensor forward(const torch::Tensor& positions,
                        const torch::Tensor& hidden_states,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache) {
    CHECK_EQ(hidden_states.dim(), 2);
    const int64_t tokens = hidden_states.size(0);
    CHECK_EQ(positions.numel(), tokens);
    CHECK_EQ(hidden_states.size(1), hidden_size_)
        << "fake qkv projection expects hidden_states last dim == hidden_size";

    init_fake_projection(hidden_states);
    auto q = fake_linear(hidden_states, q_proj_weight_, q_proj_bias_);
    auto k = fake_linear(hidden_states, k_proj_weight_, k_proj_bias_);
    auto v = fake_linear(hidden_states, v_proj_weight_, v_proj_bias_);

    auto q_normed =
        fake_rms_norm(q.view({tokens, num_heads_, head_dim_}))
            .reshape({tokens, q_size_});
    auto k_normed =
        fake_rms_norm(k.view({tokens, num_kv_heads_, head_dim_}))
            .reshape({tokens, kv_size_});

    auto pos = positions.to(hidden_states.device())
                   .to(hidden_states.scalar_type())
                   .view({tokens, 1});
    auto rotary = torch::sin(pos) * 1e-3;
    q = q_normed + rotary.expand({tokens, q_size_});
    k = k_normed + rotary.expand({tokens, kv_size_});

    last_query_ = q;
    last_key_ = k;
    last_value_ = v;

    auto out = std::get<0>(attn_.forward(attn_metadata, q, k, v, kv_cache));
    last_output_ = out;
    return last_output_;
  }

  const torch::Tensor& last_query() const { return last_query_; }
  const torch::Tensor& last_key() const { return last_key_; }
  const torch::Tensor& last_value() const { return last_value_; }

 private:
  static torch::Tensor fake_linear(const torch::Tensor& input,
                                   const torch::Tensor& weight,
                                   const torch::Tensor& bias) {
    return torch::matmul(input, weight) + bias;
  }

  static torch::Tensor fake_rms_norm(const torch::Tensor& x) {
    auto x_f32 = x.to(torch::kFloat32);
    auto variance = x_f32.pow(2).mean(-1, true);
    auto normalized = x_f32 * torch::rsqrt(variance + 1e-6f);
    return normalized.to(x.scalar_type());
  }

  void init_fake_projection(const torch::Tensor& hidden_states) {
    const bool ready = q_proj_weight_.defined() &&
                       q_proj_weight_.device() == hidden_states.device() &&
                       q_proj_weight_.scalar_type() == hidden_states.scalar_type();
    if (ready) {
      return;
    }

    auto opts = torch::TensorOptions()
                    .dtype(hidden_states.scalar_type())
                    .device(hidden_states.device());
    const float in_scale = 1.0f / std::sqrt(static_cast<float>(hidden_size_));
    q_proj_weight_ = torch::randn({hidden_size_, q_size_}, opts) * in_scale;
    k_proj_weight_ = torch::randn({hidden_size_, kv_size_}, opts) * in_scale;
    v_proj_weight_ = torch::randn({hidden_size_, kv_size_}, opts) * in_scale;
    q_proj_bias_ = torch::zeros({q_size_}, opts);
    k_proj_bias_ = torch::zeros({kv_size_}, opts);
    v_proj_bias_ = torch::zeros({kv_size_}, opts);
  }

  int64_t num_heads_ = 0;
  int64_t num_kv_heads_ = 0;
  int64_t head_dim_ = 0;
  int64_t hidden_size_ = 0;
  int64_t q_size_ = 0;
  int64_t kv_size_ = 0;
  float scale_ = 1.0f;
  MTGRAttentionTestImpl attn_;

  torch::Tensor q_proj_weight_;
  torch::Tensor k_proj_weight_;
  torch::Tensor v_proj_weight_;
  torch::Tensor q_proj_bias_;
  torch::Tensor k_proj_bias_;
  torch::Tensor v_proj_bias_;

  torch::Tensor last_query_;
  torch::Tensor last_key_;
  torch::Tensor last_value_;
  torch::Tensor last_output_;
};

torch::Tensor packed_to_bsnd(const torch::Tensor& packed,
                             int64_t num_heads,
                             int64_t head_dim) {
  CHECK_EQ(packed.dim(), 2);
  CHECK_EQ(packed.size(1), num_heads * head_dim);
  return packed.view({1, packed.size(0), num_heads, head_dim}).contiguous();
}

}  // namespace

TEST_F(FinalBSNDTest, ForwardNoPrefixMatch_FullFAPath) {
  torch::manual_seed(20260408);

  constexpr int64_t kBatchSize = 1;
  constexpr int64_t kNumHeads = 8;
  constexpr int64_t kNumKvHeads = 8;
  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kHiddenSize = kNumHeads * kHeadDim;
  constexpr int64_t kHistory = 1300;
  constexpr int64_t kContext = 8;
  constexpr int64_t kRealtime = 400;
  constexpr int64_t kTarget = 800;
  constexpr int64_t kMatchedPrefix = 0;
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kCacheBlockCount = 40;
  constexpr uint64_t kFixedSharedWorkspaceBytes = 256ULL * 1024ULL * 1024ULL;
  const int64_t total = kHistory + kContext + kRealtime + kTarget;
  const int64_t seq_block_count = (total + kBlockSize - 1) / kBlockSize;

  auto hidden_states = torch::randn({kBatchSize * total, kHiddenSize}, fp16_opts_);
  auto positions = torch::arange(
      total,
      torch::TensorOptions().dtype(torch::kInt64).device(hidden_states.device()));

  AttentionMetadata attn_metadata;
  auto len_opts = torch::TensorOptions().dtype(torch::kInt64);
  attn_metadata.q_seq_lens = torch::tensor({total}, len_opts);
  attn_metadata.kv_seq_lens = torch::tensor({total}, len_opts);
  attn_metadata.history_lens = torch::tensor({kHistory}, len_opts);
  attn_metadata.context_lens = torch::tensor({kContext}, len_opts);
  attn_metadata.real_time_lens = torch::tensor({kRealtime}, len_opts);
  attn_metadata.target_lens = torch::tensor({kTarget}, len_opts);
  attn_metadata.matched_prefix_lens = torch::tensor({kMatchedPrefix}, len_opts);
  attn_metadata.block_size = kBlockSize;
  attn_metadata.cache_block_count = kCacheBlockCount;
  attn_metadata.block_tables_host = {make_block_table(seq_block_count)};
  attn_metadata.compressed_causal_mask =
      create_compressed_causal_mask_2048(hidden_states.device());
  attn_metadata.diagonal_mask = create_diagonal_mask(kTarget, hidden_states.device());
  ASSERT_EQ(aclrtMalloc(&attn_metadata.shared_workspace,
                        kFixedSharedWorkspaceBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_SUCCESS);
  attn_metadata.shared_workspace_size = kFixedSharedWorkspaceBytes;

  KVCache kv_cache{
      torch::zeros({kCacheBlockCount, kBlockSize, kNumKvHeads, kHeadDim}, fp16_opts_),
      torch::zeros({kCacheBlockCount, kBlockSize, kNumKvHeads, kHeadDim}, fp16_opts_)};

  CustomMaskAttentionTestImpl custom_attn(
      kNumHeads, kNumKvHeads, kHeadDim, kHiddenSize);
  auto warmup_hidden_states =
      torch::randn({kBatchSize * total, kHiddenSize}, fp16_opts_);
  auto warmup_positions = positions.clone();
  auto warmup_out = custom_attn.forward(
      warmup_positions, warmup_hidden_states, attn_metadata, kv_cache);
  EXPECT_EQ(warmup_out.dim(), 2);
  EXPECT_EQ(warmup_out.size(0), kBatchSize * total);
  EXPECT_EQ(warmup_out.size(1), kNumHeads * kHeadDim);

  auto output = custom_attn.forward(positions, hidden_states, attn_metadata, kv_cache);
  auto output_bsnd = packed_to_bsnd(output, kNumHeads, kHeadDim);
  auto query_bsnd = packed_to_bsnd(custom_attn.last_query(), kNumHeads, kHeadDim);
  auto key_bsnd = packed_to_bsnd(custom_attn.last_key(), kNumKvHeads, kHeadDim);
  auto value_bsnd = packed_to_bsnd(custom_attn.last_value(), kNumKvHeads, kHeadDim);

  expect_output_matches_reference("ForwardNoPrefixMatch_FullFAPath",
                                  output_bsnd,
                                  query_bsnd,
                                  key_bsnd,
                                  value_bsnd,
                                  kHistory,
                                  kContext,
                                  kRealtime,
                                  kTarget);

  ASSERT_EQ(aclrtFree(attn_metadata.shared_workspace), ACL_SUCCESS);
}

TEST_F(FinalBSNDTest, ForwardHistoryPartialMatch_PaHistoryFaTail) {
  torch::manual_seed(20260408 + 17);

  constexpr int64_t kBatchSize = 1;
  constexpr int64_t kNumHeads = 8;
  constexpr int64_t kNumKvHeads = 8;
  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kHiddenSize = kNumHeads * kHeadDim;
  constexpr int64_t kHistory = 1300;
  constexpr int64_t kContext = 8;
  constexpr int64_t kRealtime = 400;
  constexpr int64_t kTarget = 800;
  constexpr int64_t kHistoryMatched = 900;
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kCacheBlockCount = 40;
  constexpr uint64_t kFixedSharedWorkspaceBytes = 256ULL * 1024ULL * 1024ULL;
  const int64_t total = kHistory + kContext + kRealtime + kTarget;
  const int64_t seq_block_count = (total + kBlockSize - 1) / kBlockSize;

  auto hidden_states = torch::randn({kBatchSize * total, kHiddenSize}, fp16_opts_);
  auto positions = torch::arange(
      total,
      torch::TensorOptions().dtype(torch::kInt64).device(hidden_states.device()));

  AttentionMetadata attn_metadata;
  auto len_opts = torch::TensorOptions().dtype(torch::kInt64);
  attn_metadata.q_seq_lens = torch::tensor({total}, len_opts);
  attn_metadata.kv_seq_lens = torch::tensor({total}, len_opts);
  attn_metadata.history_lens = torch::tensor({kHistory}, len_opts);
  attn_metadata.context_lens = torch::tensor({kContext}, len_opts);
  attn_metadata.real_time_lens = torch::tensor({kRealtime}, len_opts);
  attn_metadata.target_lens = torch::tensor({kTarget}, len_opts);
  attn_metadata.matched_prefix_lens = torch::tensor({kHistoryMatched}, len_opts);
  attn_metadata.block_size = kBlockSize;
  attn_metadata.cache_block_count = kCacheBlockCount;
  attn_metadata.block_tables_host = {make_block_table(seq_block_count)};
  attn_metadata.compressed_causal_mask =
      create_compressed_causal_mask_2048(hidden_states.device());
  attn_metadata.diagonal_mask = create_diagonal_mask(kTarget, hidden_states.device());
  ASSERT_EQ(aclrtMalloc(&attn_metadata.shared_workspace,
                        kFixedSharedWorkspaceBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_SUCCESS);
  attn_metadata.shared_workspace_size = kFixedSharedWorkspaceBytes;

  KVCache kv_cache{
      torch::zeros({kCacheBlockCount, kBlockSize, kNumKvHeads, kHeadDim}, fp16_opts_),
      torch::zeros({kCacheBlockCount, kBlockSize, kNumKvHeads, kHeadDim}, fp16_opts_)};

  CustomMaskAttentionTestImpl custom_attn(
      kNumHeads, kNumKvHeads, kHeadDim, kHiddenSize);
  auto warmup_hidden_states =
      torch::randn({kBatchSize * total, kHiddenSize}, fp16_opts_);
  auto warmup_positions = positions.clone();
  auto warmup_out = custom_attn.forward(
      warmup_positions, warmup_hidden_states, attn_metadata, kv_cache);
  EXPECT_EQ(warmup_out.dim(), 2);
  EXPECT_EQ(warmup_out.size(0), kBatchSize * total);
  EXPECT_EQ(warmup_out.size(1), kNumHeads * kHeadDim);

  auto output = custom_attn.forward(positions, hidden_states, attn_metadata, kv_cache);
  auto output_bsnd = packed_to_bsnd(output, kNumHeads, kHeadDim);
  auto query_bsnd = packed_to_bsnd(custom_attn.last_query(), kNumHeads, kHeadDim);
  auto key_bsnd = packed_to_bsnd(custom_attn.last_key(), kNumKvHeads, kHeadDim);
  auto value_bsnd = packed_to_bsnd(custom_attn.last_value(), kNumKvHeads, kHeadDim);

  expect_output_matches_reference("ForwardHistoryPartialMatch_PaHistoryFaTail",
                                  output_bsnd,
                                  query_bsnd,
                                  key_bsnd,
                                  value_bsnd,
                                  kHistory,
                                  kContext,
                                  kRealtime,
                                  kTarget);

  ASSERT_EQ(aclrtFree(attn_metadata.shared_workspace), ACL_SUCCESS);
}

TEST_F(FinalBSNDTest, ForwardHCFullRtPartialMatch_PaRtFaTargetTwoWay) {
  torch::manual_seed(20260408 + 29);

  constexpr int64_t kBatchSize = 1;
  constexpr int64_t kNumHeads = 8;
  constexpr int64_t kNumKvHeads = 8;
  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kHiddenSize = kNumHeads * kHeadDim;
  constexpr int64_t kHistory = 1300;
  constexpr int64_t kContext = 8;
  constexpr int64_t kRealtime = 400;
  constexpr int64_t kTarget = 800;
  constexpr int64_t kRealtimeMatched = 240;
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kCacheBlockCount = 40;
  constexpr uint64_t kFixedSharedWorkspaceBytes = 256ULL * 1024ULL * 1024ULL;
  const int64_t total = kHistory + kContext + kRealtime + kTarget;
  const int64_t seq_block_count = (total + kBlockSize - 1) / kBlockSize;

  auto hidden_states = torch::randn({kBatchSize * total, kHiddenSize}, fp16_opts_);
  auto positions = torch::arange(
      total,
      torch::TensorOptions().dtype(torch::kInt64).device(hidden_states.device()));

  AttentionMetadata attn_metadata;
  auto len_opts = torch::TensorOptions().dtype(torch::kInt64);
  attn_metadata.q_seq_lens = torch::tensor({total}, len_opts);
  attn_metadata.kv_seq_lens = torch::tensor({total}, len_opts);
  attn_metadata.history_lens = torch::tensor({kHistory}, len_opts);
  attn_metadata.context_lens = torch::tensor({kContext}, len_opts);
  attn_metadata.real_time_lens = torch::tensor({kRealtime}, len_opts);
  attn_metadata.target_lens = torch::tensor({kTarget}, len_opts);
  attn_metadata.matched_prefix_lens =
      torch::tensor({kHistory + kContext + kRealtimeMatched}, len_opts);
  attn_metadata.block_size = kBlockSize;
  attn_metadata.cache_block_count = kCacheBlockCount;
  attn_metadata.block_tables_host = {make_block_table(seq_block_count)};
  attn_metadata.compressed_causal_mask =
      create_compressed_causal_mask_2048(hidden_states.device());
  attn_metadata.diagonal_mask = create_diagonal_mask(kTarget, hidden_states.device());
  ASSERT_EQ(aclrtMalloc(&attn_metadata.shared_workspace,
                        kFixedSharedWorkspaceBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_SUCCESS);
  attn_metadata.shared_workspace_size = kFixedSharedWorkspaceBytes;

  KVCache kv_cache{
      torch::zeros({kCacheBlockCount, kBlockSize, kNumKvHeads, kHeadDim}, fp16_opts_),
      torch::zeros({kCacheBlockCount, kBlockSize, kNumKvHeads, kHeadDim}, fp16_opts_)};

  CustomMaskAttentionTestImpl custom_attn(
      kNumHeads, kNumKvHeads, kHeadDim, kHiddenSize);
  auto warmup_hidden_states =
      torch::randn({kBatchSize * total, kHiddenSize}, fp16_opts_);
  auto warmup_positions = positions.clone();
  auto warmup_out = custom_attn.forward(
      warmup_positions, warmup_hidden_states, attn_metadata, kv_cache);
  EXPECT_EQ(warmup_out.dim(), 2);
  EXPECT_EQ(warmup_out.size(0), kBatchSize * total);
  EXPECT_EQ(warmup_out.size(1), kNumHeads * kHeadDim);

  auto output = custom_attn.forward(positions, hidden_states, attn_metadata, kv_cache);
  auto output_bsnd = packed_to_bsnd(output, kNumHeads, kHeadDim);
  auto query_bsnd = packed_to_bsnd(custom_attn.last_query(), kNumHeads, kHeadDim);
  auto key_bsnd = packed_to_bsnd(custom_attn.last_key(), kNumKvHeads, kHeadDim);
  auto value_bsnd = packed_to_bsnd(custom_attn.last_value(), kNumKvHeads, kHeadDim);

  expect_output_matches_reference("ForwardHCFullRtPartialMatch_PaRtFaTargetTwoWay",
                                  output_bsnd,
                                  query_bsnd,
                                  key_bsnd,
                                  value_bsnd,
                                  kHistory,
                                  kContext,
                                  kRealtime,
                                  kTarget);

  ASSERT_EQ(aclrtFree(attn_metadata.shared_workspace), ACL_SUCCESS);
}

}  // namespace xllm::kernel::npu::test
