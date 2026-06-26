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

#include "mtgr_attention.h"

#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <tuple>
#include <vector>

#include "aclnnop/aclnn_attention_update.h"
#include "aclnnop/aclnn_fused_infer_attention_score_v3.h"
#include "aclnnop/aclnn_scatter_pa_kv_cache.h"
#include "core/kernels/npu/utils.h"
#include "glog/logging.h"

namespace xllm {
namespace layer {

namespace {

constexpr int64_t kDefaultWindow = 2147483647LL;

bool valid_tensor(const torch::Tensor& t) { return t.defined() && t.numel() > 0; }

torch::Tensor& get_workspace_cache(const torch::Device& device,
                                   uint64_t workspace_size) {
  static thread_local torch::Tensor workspace;
  if (workspace_size == 0) {
    return workspace;
  }
  if (!workspace.defined() || workspace.device() != device ||
      static_cast<uint64_t>(workspace.numel()) < workspace_size) {
    workspace = torch::empty(
        {static_cast<int64_t>(workspace_size)},
        torch::TensorOptions().dtype(torch::kUInt8).device(device));
  }
  return workspace;
}

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
  int64_t block_size = 128;
  torch::Tensor block_table;
  torch::Tensor slot_mapping;
  torch::Tensor compressed_causal_mask;
};

struct LoweredMtgrBatchMetadata {
  torch::Tensor history_lens;
  torch::Tensor context_lens;
  torch::Tensor real_time_lens;
  torch::Tensor target_lens;
  torch::Tensor matched_prefix_lens;
};

bool has_unified_mtgr_metadata(const AttentionMetadata& attn_metadata) {
  return attn_metadata.mtgr_segment_offsets_i32.defined() ||
         attn_metadata.mtgr_segment_rules_i32.defined() ||
         attn_metadata.mtgr_q_seq_starts_i32.defined() ||
         attn_metadata.mtgr_matched_prefix_lens_i32.defined();
}

void validate_mtgr_match_mode(const torch::Tensor& matched_prefix_lens,
                              MTGRMatchMode match_mode) {
  CHECK_EQ(matched_prefix_lens.dim(), 1)
      << "mtgr_matched_prefix_lens_i32 must be 1-D";
  const int64_t batch_size = matched_prefix_lens.size(0);
  bool has_no_match_request = false;
  bool has_partial_match_request = false;
  for (int64_t i = 0; i < batch_size; ++i) {
    const int64_t matched = matched_prefix_lens[i].item<int64_t>();
    has_no_match_request = has_no_match_request || (matched == 0);
    has_partial_match_request = has_partial_match_request || (matched > 0);
  }
  const MTGRMatchMode expected_match_mode =
      has_partial_match_request
          ? (has_no_match_request ? MTGRMatchMode::kMixed
                                  : MTGRMatchMode::kPartialOnly)
          : MTGRMatchMode::kNoMatchOnly;
  CHECK_EQ(static_cast<int32_t>(match_mode),
           static_cast<int32_t>(expected_match_mode))
      << "mtgr_match_mode does not match matched_prefix_lens batch composition";
}

LoweredMtgrBatchMetadata lower_unified_mtgr_to_npu_four_segment(
    const AttentionMetadata& attn_metadata) {
  const auto& q_seq_lens = attn_metadata.q_seq_lens;
  const auto& kv_seq_lens = attn_metadata.kv_seq_lens;
  const auto& segment_offsets = attn_metadata.mtgr_segment_offsets_i32;
  const auto& segment_rules = attn_metadata.mtgr_segment_rules_i32;
  const auto& q_seq_starts = attn_metadata.mtgr_q_seq_starts_i32;
  const auto& matched_prefix_lens = attn_metadata.mtgr_matched_prefix_lens_i32;

  CHECK(q_seq_lens.defined()) << "q_seq_lens must be defined";
  CHECK(kv_seq_lens.defined()) << "kv_seq_lens must be defined";
  CHECK(segment_offsets.defined()) << "mtgr_segment_offsets_i32 must be defined";
  CHECK(segment_rules.defined()) << "mtgr_segment_rules_i32 must be defined";
  CHECK(q_seq_starts.defined()) << "mtgr_q_seq_starts_i32 must be defined";
  CHECK(matched_prefix_lens.defined())
      << "mtgr_matched_prefix_lens_i32 must be defined";
  CHECK_EQ(segment_offsets.scalar_type(), torch::kInt32)
      << "mtgr_segment_offsets_i32 must be int32";
  CHECK_EQ(segment_rules.scalar_type(), torch::kInt32)
      << "mtgr_segment_rules_i32 must be int32";
  CHECK_EQ(q_seq_starts.scalar_type(), torch::kInt32)
      << "mtgr_q_seq_starts_i32 must be int32";
  CHECK_EQ(matched_prefix_lens.scalar_type(), torch::kInt32)
      << "mtgr_matched_prefix_lens_i32 must be int32";
  CHECK_EQ(segment_offsets.dim(), 2)
      << "mtgr_segment_offsets_i32 must be [batch_size, num_segments + 1]";
  CHECK_EQ(segment_offsets.size(1), 5)
      << "NPU MTGR adapter only supports 4 segments [history|context|real_time|target]";
  CHECK_EQ(segment_rules.dim(), 1)
      << "mtgr_segment_rules_i32 must be 1-D";
  CHECK_EQ(segment_rules.numel(), 4)
      << "NPU MTGR adapter only supports 4 segment rules";
  CHECK_EQ(q_seq_starts.dim(), 1) << "mtgr_q_seq_starts_i32 must be 1-D";
  CHECK_EQ(matched_prefix_lens.dim(), 1)
      << "mtgr_matched_prefix_lens_i32 must be 1-D";

  const int64_t batch_size = q_seq_lens.size(0);
  CHECK_EQ(kv_seq_lens.size(0), batch_size);
  CHECK_EQ(segment_offsets.size(0), batch_size);
  CHECK_EQ(q_seq_starts.size(0), batch_size);
  CHECK_EQ(matched_prefix_lens.size(0), batch_size);

  CHECK_EQ(segment_rules[0].item<int64_t>(), 0)
      << "NPU MTGR adapter expects segment_rules=[0,1,0,2]";
  CHECK_EQ(segment_rules[1].item<int64_t>(), 1)
      << "NPU MTGR adapter expects segment_rules=[0,1,0,2]";
  CHECK_EQ(segment_rules[2].item<int64_t>(), 0)
      << "NPU MTGR adapter expects segment_rules=[0,1,0,2]";
  CHECK_EQ(segment_rules[3].item<int64_t>(), 2)
      << "NPU MTGR adapter expects segment_rules=[0,1,0,2]";
  validate_mtgr_match_mode(matched_prefix_lens, attn_metadata.mtgr_match_mode);

  auto segment_lens =
      (segment_offsets.slice(/*dim=*/1, /*start=*/1, /*end=*/5) -
       segment_offsets.slice(/*dim=*/1, /*start=*/0, /*end=*/4))
          .contiguous();
  auto history_lens = segment_lens.select(/*dim=*/1, /*index=*/0).contiguous();
  auto context_lens = segment_lens.select(/*dim=*/1, /*index=*/1).contiguous();
  auto real_time_lens =
      segment_lens.select(/*dim=*/1, /*index=*/2).contiguous();
  auto target_lens = segment_lens.select(/*dim=*/1, /*index=*/3).contiguous();

  int64_t packed_q_start = 0;
  for (int64_t i = 0; i < batch_size; ++i) {
    const int64_t offset0 = segment_offsets[i][0].item<int64_t>();
    const int64_t offset1 = segment_offsets[i][1].item<int64_t>();
    const int64_t offset2 = segment_offsets[i][2].item<int64_t>();
    const int64_t offset3 = segment_offsets[i][3].item<int64_t>();
    const int64_t offset4 = segment_offsets[i][4].item<int64_t>();
    CHECK_EQ(offset0, 0) << "mtgr_segment_offsets_i32 row must start at 0";
    CHECK_LE(offset0, offset1);
    CHECK_LE(offset1, offset2);
    CHECK_LE(offset2, offset3);
    CHECK_LE(offset3, offset4);

    const int64_t history = offset1 - offset0;
    const int64_t context = offset2 - offset1;
    const int64_t real_time = offset3 - offset2;
    const int64_t target = offset4 - offset3;
    const int64_t matched = matched_prefix_lens[i].item<int64_t>();
    const int64_t q_len = q_seq_lens[i].item<int64_t>();
    const int64_t kv_len = kv_seq_lens[i].item<int64_t>();
    CHECK_GT(history, 0)
        << "NPU MTGR adapter requires history segment length > 0";
    CHECK_GE(context, 0)
        << "NPU MTGR adapter requires context segment length >= 0";
    CHECK_GT(real_time, 0)
        << "NPU MTGR adapter requires real_time segment length > 0";
    CHECK_GT(target, 0)
        << "NPU MTGR adapter requires target segment length > 0";
    CHECK_GE(matched, 0);
    CHECK_LE(matched, history + context + real_time)
        << "matched_prefix must stay within [history|context|real_time]";
    CHECK_EQ(q_len, offset4 - matched)
        << "trimmed MTGR q_seq_lens must equal total_len - matched_prefix";
    CHECK_EQ(kv_len, offset4 - matched)
        << "trimmed MTGR kv_seq_lens must equal total_len - matched_prefix";
    CHECK_EQ(q_seq_starts[i].item<int64_t>(), packed_q_start)
        << "mtgr_q_seq_starts_i32 must match packed trimmed-query layout";
    if (matched > 0) {
      CHECK_GE(matched, history + context)
          << "Current NPU MTGR kernel only supports no-match or realtime partial match";
      CHECK_LT(matched, history + context + real_time)
          << "Current NPU MTGR kernel requires at least one unmatched realtime token";
    }
    packed_q_start += q_len;
  }

  return LoweredMtgrBatchMetadata{
      .history_lens = history_lens,
      .context_lens = context_lens,
      .real_time_lens = real_time_lens,
      .target_lens = target_lens,
      .matched_prefix_lens = matched_prefix_lens.contiguous(),
  };
}

LoweredMtgrBatchMetadata lower_mtgr_to_npu_four_segment(
    const AttentionMetadata& attn_metadata) {
  CHECK(has_unified_mtgr_metadata(attn_metadata))
      << "MTGR attention requires unified mtgr_segment_* metadata";
  return lower_unified_mtgr_to_npu_four_segment(attn_metadata);
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
  const int64_t num_heads = query.size(2);
  const int64_t num_kv_heads = key.size(2);
  const float scale = 1.0f / std::sqrt(static_cast<float>(query.size(3)));
  AclPlan plan;

  kernel::npu::create_acltensor(&plan.q_acl, query);
  kernel::npu::create_acltensor(&plan.k_acl, key);
  kernel::npu::create_acltensor(&plan.v_acl, value);
  kernel::npu::create_acltensor(&plan.out_acl, output);
  if (valid_tensor(attn_metadata.attn_mask)) {
    kernel::npu::create_acltensor(&plan.mask_acl, attn_metadata.attn_mask);
  }
  if (lse.has_value() && valid_tensor(lse.value())) {
    kernel::npu::create_acltensor(&plan.lse_acl, lse.value());
  }

  aclTensor* k_arr[] = {plan.k_acl};
  aclTensor* v_arr[] = {plan.v_acl};
  plan.k_list = aclCreateTensorList(k_arr, 1);
  plan.v_list = aclCreateTensorList(v_arr, 1);
  CHECK_NE(plan.k_list, nullptr);
  CHECK_NE(plan.v_list, nullptr);

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
                               const torch::Device& device,
                               aclrtStream stream) {
  void* ws_ptr = nullptr;
  if (plan.workspace_size > 0) {
    auto& workspace = get_workspace_cache(device, plan.workspace_size);
    ws_ptr = workspace.data_ptr();
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
  aclTensor* lse_acl = nullptr;
  aclTensorList* k_list = nullptr;
  aclTensorList* v_list = nullptr;
  aclIntArray* actual_q = nullptr;
  aclIntArray* actual_kv = nullptr;
};

PagedAttentionPlan plan_paged_attention_v3(const torch::Tensor& query_bsnd,
                                           const torch::Tensor& key_cache_bnbsh,
                                           const torch::Tensor& value_cache_bnbsh,
                                           const torch::Tensor& block_table,
                                           const torch::Tensor& attn_mask,
                                           int64_t sparse_mode,
                                           int64_t pre_tokens,
                                           int64_t next_tokens,
                                           int64_t total_kv_len,
                                           int64_t block_size,
                                           int64_t num_kv_heads,
                                           torch::Tensor& output_bsnd,
                                           const std::optional<torch::Tensor>& lse =
                                               std::nullopt) {
  PagedAttentionPlan plan;

  kernel::npu::create_acltensor(&plan.q_acl, query_bsnd);
  kernel::npu::create_acltensor(&plan.key_cache_acl, key_cache_bnbsh);
  kernel::npu::create_acltensor(&plan.value_cache_acl, value_cache_bnbsh);
  kernel::npu::create_acltensor(&plan.block_table_acl, block_table);
  kernel::npu::create_acltensor(&plan.out_acl, output_bsnd);
  if (valid_tensor(attn_mask)) {
    kernel::npu::create_acltensor(&plan.mask_acl, attn_mask);
  }
  if (lse.has_value() && valid_tensor(lse.value())) {
    kernel::npu::create_acltensor(&plan.lse_acl, lse.value());
  }

  aclTensor* k_arr[] = {plan.key_cache_acl};
  aclTensor* v_arr[] = {plan.value_cache_acl};
  plan.k_list = aclCreateTensorList(k_arr, 1);
  plan.v_list = aclCreateTensorList(v_arr, 1);
  CHECK_NE(plan.k_list, nullptr);
  CHECK_NE(plan.v_list, nullptr);

  int64_t q_seqlen[1] = {query_bsnd.size(1)};
  int64_t kv_seqlen[1] = {total_kv_len};
  plan.actual_q = aclCreateIntArray(q_seqlen, 1);
  plan.actual_kv = aclCreateIntArray(kv_seqlen, 1);
  CHECK_NE(plan.actual_q, nullptr);
  CHECK_NE(plan.actual_kv, nullptr);

  const double scale = 1.0 / std::sqrt(static_cast<double>(query_bsnd.size(3)));
  const int64_t num_heads = query_bsnd.size(2);
  const bool softmax_lse_flag = (lse.has_value() && valid_tensor(lse.value()));
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
      pre_tokens,
      next_tokens,
      layout,
      num_kv_heads,
      sparse_mode,
      /*innerPrecise=*/1,
      /*blockSize=*/block_size,
      /*antiquantMode=*/0,
      /*softmaxLseFlag=*/softmax_lse_flag,
      /*keyAntiquantMode=*/0,
      /*valueAntiquantMode=*/0,
      plan.out_acl,
      /*softmaxLse=*/plan.lse_acl,
      &plan.workspace_size,
      &plan.executor);
  CHECK_EQ(ret, ACL_SUCCESS)
      << "aclnnFusedInferAttentionScoreV3GetWorkspaceSize (paged) failed: " << ret;
  return plan;
}

void execute_planned_paged_attention(const PagedAttentionPlan& plan,
                                     const torch::Device& device,
                                     aclrtStream stream) {
  void* ws_ptr = nullptr;
  if (plan.workspace_size > 0) {
    auto& workspace = get_workspace_cache(device, plan.workspace_size);
    ws_ptr = workspace.data_ptr();
    CHECK_EQ(aclrtMemset(ws_ptr, plan.workspace_size, 0, plan.workspace_size),
             ACL_SUCCESS)
        << "aclrtMemset paged-attention workspace";
  }
  auto ret =
      aclnnFusedInferAttentionScoreV3(ws_ptr, plan.workspace_size, plan.executor, stream);
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
  if (plan.lse_acl != nullptr) {
    aclDestroyTensor(plan.lse_acl);
    plan.lse_acl = nullptr;
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

  kernel::npu::create_acltensor(&key_acl, key);
  kernel::npu::create_acltensor(&value_acl, value);
  kernel::npu::create_acltensor(&key_cache_acl, key_cache);
  kernel::npu::create_acltensor(&value_cache_acl, value_cache);
  kernel::npu::create_acltensor(&slot_mapping_acl, slot_mapping);

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

  cleanup();
  return true;
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

  const int64_t bsn = plan.b * plan.s * plan.n;
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
    plan.local_out_flats.push_back(local_outs[i].view({bsn, plan.d}).to(torch::kFloat32));
    plan.lse_flats.push_back(
        lses[i].permute({0, 2, 1, 3}).contiguous().view({bsn}));
    kernel::npu::create_acltensor(&plan.local_out_acls[i], plan.local_out_flats.back());
    kernel::npu::create_acltensor(&plan.lse_acls[i], plan.lse_flats.back());
  }

  plan.out_flat =
      torch::empty({bsn, plan.d},
                   torch::TensorOptions().dtype(torch::kFloat32).device(out0.device()));
  kernel::npu::create_acltensor(&plan.out_acl, plan.out_flat);

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

AttentionUpdatePlan plan_attention_update_from_prepared_flats(
    const std::vector<torch::Tensor>& local_out_flats,
    const std::vector<torch::Tensor>& lse_flats,
    int64_t b,
    int64_t s,
    int64_t n,
    int64_t d,
    torch::ScalarType out_scalar_type) {
  CHECK_EQ(local_out_flats.size(), lse_flats.size());
  CHECK_GE(local_out_flats.size(), 2UL);

  const int64_t bsn = b * s * n;
  AttentionUpdatePlan plan;
  plan.b = b;
  plan.s = s;
  plan.n = n;
  plan.d = d;
  plan.out_scalar_type = out_scalar_type;
  plan.local_out_flats = local_out_flats;
  plan.lse_flats = lse_flats;
  plan.local_out_acls.assign(local_out_flats.size(), nullptr);
  plan.lse_acls.assign(lse_flats.size(), nullptr);

  for (size_t i = 0; i < local_out_flats.size(); ++i) {
    CHECK_EQ(local_out_flats[i].dim(), 2);
    CHECK_EQ(local_out_flats[i].size(0), bsn);
    CHECK_EQ(local_out_flats[i].size(1), d);
    CHECK(local_out_flats[i].is_contiguous())
        << "local_out_flat must be contiguous";
    CHECK_EQ(local_out_flats[i].scalar_type(), torch::kFloat32);

    CHECK_EQ(lse_flats[i].dim(), 1);
    CHECK_EQ(lse_flats[i].size(0), bsn);
    CHECK(lse_flats[i].is_contiguous()) << "lse_flat must be contiguous";
    CHECK_EQ(lse_flats[i].scalar_type(), torch::kFloat32);

    kernel::npu::create_acltensor(&plan.local_out_acls[i], plan.local_out_flats[i]);
    kernel::npu::create_acltensor(&plan.lse_acls[i], plan.lse_flats[i]);
  }

  plan.out_flat =
      torch::empty({bsn, d},
                   torch::TensorOptions().dtype(torch::kFloat32).device(
                       local_out_flats[0].device()));
  kernel::npu::create_acltensor(&plan.out_acl, plan.out_flat);

  plan.local_out_list = aclCreateTensorList(
      plan.local_out_acls.data(),
      static_cast<int32_t>(plan.local_out_acls.size()));
  CHECK_NE(plan.local_out_list, nullptr);
  plan.lse_list = aclCreateTensorList(
      plan.lse_acls.data(), static_cast<int32_t>(plan.lse_acls.size()));
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
                                      const torch::Device& device,
                                      aclrtStream stream) {
  void* ws_ptr = nullptr;
  if (plan.workspace_size > 0) {
    auto& workspace = get_workspace_cache(device, plan.workspace_size);
    ws_ptr = workspace.data_ptr();
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

std::pair<torch::Tensor, torch::Tensor> build_target_diagonal_attn_analytic(
    const torch::Tensor& target_query,
    const torch::Tensor& target_key,
    const torch::Tensor& target_value) {
  CHECK_EQ(target_query.dim(), 4);
  CHECK_EQ(target_key.dim(), 4);
  CHECK_EQ(target_value.dim(), 4);
  CHECK_EQ(target_query.sizes(), target_key.sizes());
  CHECK_EQ(target_query.sizes(), target_value.sizes());

  const double scale = 1.0 / std::sqrt(static_cast<double>(target_query.size(3)));
  auto output = target_value.contiguous();
  auto q_f32 = target_query.to(torch::kFloat32);
  auto k_f32 = target_key.to(torch::kFloat32);
  auto lse =
      (q_f32 * k_f32).sum(-1, true).mul(scale).permute({0, 2, 1, 3}).contiguous();
  return {output, lse};
}

// Keep the legacy four-segment helpers in-file. The current unified MTGR
// entry only dispatches no-match and partial-rt-match, but these paths remain
// useful for parity, debugging, and future rollback.
[[maybe_unused]] void fa_for_crt(const torch::Tensor& query,
                                 const torch::Tensor& key,
                                 const torch::Tensor& value,
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

  auto out_opts =
      torch::TensorOptions().dtype(query.scalar_type()).device(query.device());
  auto lse_opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(query.device());
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

  AclPlan crt_plan =
      plan_segment_attention(crt_query, crt_key, crt_value, crt_meta, crt_out, crt_lse);
  execute_planned_attention(crt_plan, query.device(), stream);
  destroy_planned_attention(crt_plan);
  output.slice(1, h, h + c).copy_(crt_out.slice(1, 0, c));

  auto rt_out = make_out(r + t);
  auto rt_lse = make_lse(r + t);
  auto rt_query = query.slice(1, h + c, h + c + r + t).contiguous();
  auto rt_key = key.slice(1, h + c, h + c + r).contiguous();
  auto rt_value = value.slice(1, h + c, h + c + r).contiguous();

  SegmentAttentionMetadata rt_meta;
  rt_meta.attn_mask = sample_metadata.compressed_causal_mask;
  rt_meta.sparse_param.sparse_mode = 2;
  rt_meta.sparse_param.pre_tokens = kDefaultWindow;
  rt_meta.sparse_param.next_tokens = kDefaultWindow;

  AclPlan rt_plan =
      plan_segment_attention(rt_query, rt_key, rt_value, rt_meta, rt_out, rt_lse);
  execute_planned_attention(rt_plan, query.device(), stream);
  destroy_planned_attention(rt_plan);

  auto target_out = make_out(t);
  auto target_lse = make_lse(t);
  auto target_query = query.slice(1, h + c + r, h + c + r + t).contiguous();
  auto target_key = key.slice(1, h + c + r, h + c + r + t).contiguous();
  auto target_value = value.slice(1, h + c + r, h + c + r + t).contiguous();

  SegmentAttentionMetadata target_meta;
  target_meta.attn_mask = create_diagonal_mask(t, query.device());
  target_meta.sparse_param.sparse_mode = 0;
  target_meta.sparse_param.pre_tokens = 0;
  target_meta.sparse_param.next_tokens = 0;

  AclPlan target_plan = plan_segment_attention(
      target_query, target_key, target_value, target_meta, target_out, target_lse);
  execute_planned_attention(target_plan, query.device(), stream);
  destroy_planned_attention(target_plan);

  auto rt_update_plan = plan_attention_update_like_benchmark(
      {crt_out.slice(1, c, c + r).contiguous(), rt_out.slice(1, 0, r).contiguous()},
      {crt_lse.slice(2, c, c + r).contiguous(), rt_lse.slice(2, 0, r).contiguous()});
  execute_planned_attention_update(rt_update_plan, query.device(), stream);
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
  execute_planned_attention_update(target_update_plan, query.device(), stream);
  auto target_merged = target_update_plan.out_flat.view({target_update_plan.b,
                                                         target_update_plan.s,
                                                         target_update_plan.n,
                                                         target_update_plan.d})
                           .to(target_update_plan.out_scalar_type);
  destroy_planned_attention_update(target_update_plan);
  output.slice(1, h + c + r, h + c + r + t).copy_(target_merged);
}

void no_matched(const torch::Tensor& query,
                const torch::Tensor& key,
                const torch::Tensor& value,
                const SampleSegmentMetadata& sample_metadata,
                torch::Tensor& output) {
  CHECK_EQ(query.size(0), 1);
  const int64_t h = sample_metadata.history;
  const int64_t c = sample_metadata.context;
  const int64_t r = sample_metadata.real_time;
  const int64_t t = sample_metadata.target;
  const int64_t num_heads = query.size(2);
  const int64_t head_dim = query.size(3);
  CHECK_EQ(query.size(1), h + c + r + t);
  CHECK_EQ(key.size(1), h + c + r + t);
  CHECK_EQ(value.size(1), h + c + r + t);
  CHECK_GT(r, 0);
  CHECK_GT(t, 0);
  auto stream = c10_npu::getCurrentNPUStream(query.device().index()).stream();

  auto out_opts = torch::TensorOptions().dtype(query.scalar_type()).device(query.device());
  auto lse_opts = torch::TensorOptions().dtype(torch::kFloat32).device(query.device());
  auto make_out = [&](int64_t seq_len) {
    return torch::empty({1, seq_len, num_heads, head_dim}, out_opts);
  };
  auto make_lse = [&](int64_t seq_len) {
    return torch::empty({1, num_heads, seq_len, 1}, lse_opts);
  };

  // no-match best path:
  // 1) Q=R+T, KV=H+C+R, sparse_mode=4 trapezoid;
  // 2) Q=H, KV=H, causal;
  // 3) Q=C, KV=H+C, full;
  // 4) Q=T, KV=T, diagonal analytic;
  // 5) update target prefix branch and diagonal branch.
  auto rt_tgt_out = make_out(r + t);
  auto rt_tgt_lse = make_lse(r + t);
  SegmentAttentionMetadata rt_tgt_meta;
  rt_tgt_meta.attn_mask = sample_metadata.compressed_causal_mask;
  rt_tgt_meta.sparse_param.sparse_mode = 4;
  rt_tgt_meta.sparse_param.pre_tokens = h + c + r;
  rt_tgt_meta.sparse_param.next_tokens = t;
  auto rt_tgt_query = query.slice(1, h + c, h + c + r + t).contiguous();
  auto hcr_key = key.slice(1, 0, h + c + r).contiguous();
  auto hcr_value = value.slice(1, 0, h + c + r).contiguous();
  auto rt_tgt_plan = plan_segment_attention(
      rt_tgt_query, hcr_key, hcr_value, rt_tgt_meta, rt_tgt_out, rt_tgt_lse);
  execute_planned_attention(rt_tgt_plan, query.device(), stream);
  destroy_planned_attention(rt_tgt_plan);

  const int64_t update_b = 1;
  const int64_t update_s = t;
  const int64_t update_n = num_heads;
  const int64_t update_d = head_dim;
  const int64_t update_bsn = update_b * update_s * update_n;
  auto f32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(query.device());
  auto target_prefix_out_flat_placeholder =
      torch::empty({update_bsn, update_d}, f32_opts);
  auto target_prefix_lse_flat_placeholder = torch::empty({update_bsn}, f32_opts);
  auto target_diag_out_flat_placeholder =
      torch::empty({update_bsn, update_d}, f32_opts);
  auto target_diag_lse_flat_placeholder = torch::empty({update_bsn}, f32_opts);

  auto target_update_plan = plan_attention_update_from_prepared_flats(
      {target_prefix_out_flat_placeholder, target_diag_out_flat_placeholder},
      {target_prefix_lse_flat_placeholder, target_diag_lse_flat_placeholder},
      update_b,
      update_s,
      update_n,
      update_d,
      query.scalar_type());

  if (h > 0) {
    auto hist_out = make_out(h);
    SegmentAttentionMetadata hist_meta;
    hist_meta.attn_mask = sample_metadata.compressed_causal_mask;
    hist_meta.sparse_param.sparse_mode = 2;
    hist_meta.sparse_param.pre_tokens = kDefaultWindow;
    hist_meta.sparse_param.next_tokens = kDefaultWindow;
    auto hist_query = query.slice(1, 0, h).contiguous();
    auto hist_key = key.slice(1, 0, h).contiguous();
    auto hist_value = value.slice(1, 0, h).contiguous();
    auto hist_plan = plan_segment_attention(
        hist_query, hist_key, hist_value, hist_meta, hist_out, std::nullopt);
    execute_planned_attention(hist_plan, query.device(), stream);
    destroy_planned_attention(hist_plan);
    output.slice(1, 0, h).copy_(hist_out);
  }

  if (c > 0) {
    auto ctx_out = make_out(c);
    SegmentAttentionMetadata ctx_meta;
    ctx_meta.sparse_param.sparse_mode = 0;
    ctx_meta.sparse_param.pre_tokens = kDefaultWindow;
    ctx_meta.sparse_param.next_tokens = kDefaultWindow;
    auto ctx_query = query.slice(1, h, h + c).contiguous();
    auto hc_key = key.slice(1, 0, h + c).contiguous();
    auto hc_value = value.slice(1, 0, h + c).contiguous();
    auto ctx_plan = plan_segment_attention(
        ctx_query, hc_key, hc_value, ctx_meta, ctx_out, std::nullopt);
    execute_planned_attention(ctx_plan, query.device(), stream);
    destroy_planned_attention(ctx_plan);
    output.slice(1, h, h + c).copy_(ctx_out);
  }

  target_prefix_out_flat_placeholder.view({update_b, update_s, update_n, update_d})
      .copy_(rt_tgt_out.narrow(1, r, t));
  target_prefix_lse_flat_placeholder.view({update_b, update_s, update_n})
      .copy_(rt_tgt_lse.narrow(2, r, t).permute({0, 2, 1, 3}).squeeze(-1));

  auto target_query = query.slice(1, h + c + r, h + c + r + t).contiguous();
  auto target_key = key.slice(1, h + c + r, h + c + r + t).contiguous();
  auto target_value = value.slice(1, h + c + r, h + c + r + t).contiguous();
  auto [target_diag_out, target_diag_lse] =
      build_target_diagonal_attn_analytic(target_query, target_key, target_value);

  target_diag_out_flat_placeholder.view({update_b, update_s, update_n, update_d})
      .copy_(target_diag_out);
  target_diag_lse_flat_placeholder.view({update_b, update_s, update_n})
      .copy_(target_diag_lse.permute({0, 2, 1, 3}).squeeze(-1));

  execute_planned_attention_update(target_update_plan, query.device(), stream);
  auto target_merged =
      target_update_plan.out_flat
          .view({target_update_plan.b,
                 target_update_plan.s,
                 target_update_plan.n,
                 target_update_plan.d})
          .to(target_update_plan.out_scalar_type);

  destroy_planned_attention_update(target_update_plan);
  output.slice(1, h + c, h + c + r).copy_(rt_tgt_out.slice(1, 0, r));
  output.slice(1, h + c + r, h + c + r + t).copy_(target_merged);
}

[[maybe_unused]] void partial_hist_matched(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    const SampleSegmentMetadata& sample_metadata,
    torch::Tensor& output) {
  CHECK_EQ(query.size(0), 1);
  CHECK(sample_metadata.block_table.defined())
      << "block_table is required for prefix-cache path";
  CHECK(sample_metadata.slot_mapping.defined())
      << "slot_mapping is required for prefix-cache path";

  const int64_t h = sample_metadata.history;
  const int64_t c = sample_metadata.context;
  const int64_t r = sample_metadata.real_time;
  const int64_t t = sample_metadata.target;
  const int64_t history_matched = sample_metadata.matched_prefix;
  const int64_t history_unmatched = h - history_matched;
  CHECK_GT(history_unmatched, 0);
  CHECK_GT(c, 0);
  CHECK_GT(r, 0);
  CHECK_GT(t, 0);
  const int64_t local_q_len = query.size(1);
  CHECK_EQ(local_q_len, history_unmatched + c + r + t)
      << "partial_hist_matched expects unmatched layout [h_unmatched|c|r|t]";

  const int64_t block_size = sample_metadata.block_size;
  const int64_t cache_block_count = key_cache.size(0);
  const int64_t num_kv_heads = key.size(2);
  const int64_t head_dim = key.size(3);
  auto stream = c10_npu::getCurrentNPUStream(query.device().index()).stream();
  auto i32_dev_opts =
      torch::TensorOptions().dtype(torch::kInt32).device(query.device());

  CHECK_EQ(key_cache.dim(), 4);
  CHECK_EQ(value_cache.dim(), 4);
  CHECK_EQ(key_cache.size(1), block_size);
  CHECK_EQ(key_cache.size(2), num_kv_heads);
  CHECK_EQ(key_cache.size(3), head_dim);
  CHECK_EQ(value_cache.size(0), cache_block_count);
  CHECK_EQ(value_cache.size(1), block_size);
  CHECK_EQ(value_cache.size(2), num_kv_heads);
  CHECK_EQ(value_cache.size(3), head_dim);

  const int64_t local_context_start = history_unmatched;
  const int64_t local_rt_start = local_context_start + c;
  const int64_t local_target_start = local_rt_start + r;
  const int64_t prefix_unmatched_len = history_unmatched + c;
  CHECK_GE(sample_metadata.slot_mapping.size(0), prefix_unmatched_len)
      << "slot_mapping is shorter than unmatched prefix length";

  auto key_seq = key.select(0, 0).contiguous();
  auto value_seq = value.select(0, 0).contiguous();
  auto new_slots_dev_i32 =
      sample_metadata.slot_mapping
          .slice(/*dim=*/0, /*start=*/0, /*end=*/prefix_unmatched_len)
          .to(i32_dev_opts.dtype())
          .contiguous();
  auto new_key = key_seq.slice(0, 0, prefix_unmatched_len).contiguous();
  auto new_value = value_seq.slice(0, 0, prefix_unmatched_len).contiguous();

  CHECK(run_scatter_pa_kv_cache(
      new_key, new_value, key_cache, value_cache, new_slots_dev_i32, stream))
      << "run_scatter_pa_kv_cache failed";

  auto key_cache_bnbsh =
      key_cache.view({cache_block_count, block_size, num_kv_heads * head_dim});
  auto value_cache_bnbsh =
      value_cache.view({cache_block_count, block_size, num_kv_heads * head_dim});
  auto block_table = sample_metadata.block_table.to(i32_dev_opts.dtype()).contiguous();

  auto history_query_unmatched = query.slice(1, 0, history_unmatched).contiguous();
  auto history_out_unmatched = torch::empty_like(history_query_unmatched);
  auto hist_pa_plan = plan_paged_attention_v3(
      history_query_unmatched,
      key_cache_bnbsh,
      value_cache_bnbsh,
      block_table,
      sample_metadata.compressed_causal_mask,
      /*sparse_mode=*/3,
      /*pre_tokens=*/kDefaultWindow,
      /*next_tokens=*/kDefaultWindow,
      /*total_kv_len=*/h,
      /*block_size=*/block_size,
      /*num_kv_heads=*/num_kv_heads,
      history_out_unmatched);
  execute_planned_paged_attention(hist_pa_plan, query.device(), stream);
  destroy_planned_paged_attention(hist_pa_plan);
  output.slice(1, 0, history_unmatched).copy_(history_out_unmatched);

  const int64_t num_heads = query.size(2);
  auto out_opts =
      torch::TensorOptions().dtype(query.scalar_type()).device(query.device());
  auto lse_opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(query.device());
  constexpr float kSentinel = -31415.0f;
  auto make_out = [&](int64_t seq_len) {
    return torch::full({1, seq_len, num_heads, head_dim}, kSentinel, out_opts);
  };
  auto make_lse = [&](int64_t seq_len) {
    return torch::full({1, num_heads, seq_len, 1}, kSentinel, lse_opts);
  };

  auto crt_query =
      query.slice(1, local_context_start, local_context_start + c + r + t).contiguous();
  auto crt_out = make_out(c + r + t);
  auto crt_lse = make_lse(c + r + t);
  auto crt_pa_plan = plan_paged_attention_v3(
      crt_query,
      key_cache_bnbsh,
      value_cache_bnbsh,
      block_table,
      /*attn_mask=*/torch::Tensor(),
      /*sparse_mode=*/0,
      /*pre_tokens=*/kDefaultWindow,
      /*next_tokens=*/kDefaultWindow,
      /*total_kv_len=*/h + c,
      /*block_size=*/block_size,
      /*num_kv_heads=*/num_kv_heads,
      crt_out,
      crt_lse);
  execute_planned_paged_attention(crt_pa_plan, query.device(), stream);
  destroy_planned_paged_attention(crt_pa_plan);

  output.slice(1, local_context_start, local_context_start + c)
      .copy_(crt_out.slice(1, 0, c));

  auto rt_tgt_query = query.slice(1, local_rt_start, local_rt_start + r + t).contiguous();
  auto rt_key = key.slice(1, local_rt_start, local_rt_start + r).contiguous();
  auto rt_value = value.slice(1, local_rt_start, local_rt_start + r).contiguous();
  auto rt_tgt_out = make_out(r + t);
  auto rt_tgt_lse = make_lse(r + t);
  SegmentAttentionMetadata rt_meta;
  rt_meta.attn_mask = sample_metadata.compressed_causal_mask;
  rt_meta.sparse_param.sparse_mode = 2;
  rt_meta.sparse_param.pre_tokens = kDefaultWindow;
  rt_meta.sparse_param.next_tokens = kDefaultWindow;
  auto rt_plan =
      plan_segment_attention(rt_tgt_query, rt_key, rt_value, rt_meta, rt_tgt_out, rt_tgt_lse);
  execute_planned_attention(rt_plan, query.device(), stream);
  destroy_planned_attention(rt_plan);

  auto rt_update_plan = plan_attention_update_like_benchmark(
      {crt_out.slice(1, c, c + r).contiguous(), rt_tgt_out.slice(1, 0, r).contiguous()},
      {crt_lse.slice(2, c, c + r).contiguous(), rt_tgt_lse.slice(2, 0, r).contiguous()});
  execute_planned_attention_update(rt_update_plan, query.device(), stream);
  auto rt_merged = rt_update_plan.out_flat.view({rt_update_plan.b,
                                                 rt_update_plan.s,
                                                 rt_update_plan.n,
                                                 rt_update_plan.d})
                       .to(rt_update_plan.out_scalar_type);
  destroy_planned_attention_update(rt_update_plan);
  output.slice(1, local_rt_start, local_rt_start + r).copy_(rt_merged);

  auto target_query =
      query.slice(1, local_target_start, local_target_start + t).contiguous();
  auto target_key = key.slice(1, local_target_start, local_target_start + t).contiguous();
  auto target_value =
      value.slice(1, local_target_start, local_target_start + t).contiguous();
  auto target_self_out = make_out(t);
  auto target_self_lse = make_lse(t);
  SegmentAttentionMetadata self_meta;
  self_meta.attn_mask = create_diagonal_mask(t, query.device());
  self_meta.sparse_param.sparse_mode = 0;
  self_meta.sparse_param.pre_tokens = 0;
  self_meta.sparse_param.next_tokens = 0;
  auto self_plan = plan_segment_attention(target_query,
                                          target_key,
                                          target_value,
                                          self_meta,
                                          target_self_out,
                                          target_self_lse);
  execute_planned_attention(self_plan, query.device(), stream);
  destroy_planned_attention(self_plan);

  auto target_update_plan = plan_attention_update_like_benchmark(
      {crt_out.slice(1, c + r, c + r + t).contiguous(),
       rt_tgt_out.slice(1, r, r + t).contiguous(),
       target_self_out.contiguous()},
      {crt_lse.slice(2, c + r, c + r + t).contiguous(),
       rt_tgt_lse.slice(2, r, r + t).contiguous(),
       target_self_lse.contiguous()});
  execute_planned_attention_update(target_update_plan, query.device(), stream);
  auto target_merged =
      target_update_plan.out_flat
          .view({target_update_plan.b,
                 target_update_plan.s,
                 target_update_plan.n,
                 target_update_plan.d})
          .to(target_update_plan.out_scalar_type);
  destroy_planned_attention_update(target_update_plan);
  output.slice(1, local_target_start, local_target_start + t).copy_(target_merged);
}

[[maybe_unused]] void partial_ctx_matched(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    const SampleSegmentMetadata& sample_metadata,
    torch::Tensor& output) {
  CHECK_EQ(query.size(0), 1);
  CHECK(sample_metadata.block_table.defined())
      << "block_table is required for prefix-cache path";
  CHECK(sample_metadata.slot_mapping.defined())
      << "slot_mapping is required for prefix-cache path";

  const int64_t h = sample_metadata.history;
  const int64_t c = sample_metadata.context;
  const int64_t r = sample_metadata.real_time;
  const int64_t t = sample_metadata.target;
  CHECK_GE(sample_metadata.matched_prefix, h);
  const int64_t context_matched = sample_metadata.matched_prefix - h;
  CHECK_LT(context_matched, c);
  const int64_t context_unmatched = c - context_matched;
  CHECK_GT(context_unmatched, 0);
  CHECK_GT(r, 0);
  CHECK_GT(t, 0);
  CHECK_EQ(query.size(1), context_unmatched + r + t)
      << "partial_ctx_matched expects unmatched layout [c_unmatched|r|t]";

  const int64_t block_size = sample_metadata.block_size;
  const int64_t cache_block_count = key_cache.size(0);
  const int64_t num_kv_heads = key.size(2);
  const int64_t num_heads = query.size(2);
  const int64_t head_dim = key.size(3);
  const int64_t local_rt_start = context_unmatched;
  const int64_t local_target_start = local_rt_start + r;
  CHECK_GE(sample_metadata.slot_mapping.size(0), context_unmatched)
      << "slot_mapping is shorter than unmatched context length";

  auto stream = c10_npu::getCurrentNPUStream(query.device().index()).stream();
  auto i32_dev_opts =
      torch::TensorOptions().dtype(torch::kInt32).device(query.device());

  CHECK_EQ(key_cache.dim(), 4);
  CHECK_EQ(value_cache.dim(), 4);
  CHECK_EQ(key_cache.size(1), block_size);
  CHECK_EQ(key_cache.size(2), num_kv_heads);
  CHECK_EQ(key_cache.size(3), head_dim);
  CHECK_EQ(value_cache.size(0), cache_block_count);
  CHECK_EQ(value_cache.size(1), block_size);
  CHECK_EQ(value_cache.size(2), num_kv_heads);
  CHECK_EQ(value_cache.size(3), head_dim);

  auto key_seq = key.select(0, 0).contiguous();
  auto value_seq = value.select(0, 0).contiguous();
  auto new_slots_dev_i32 =
      sample_metadata.slot_mapping
          .slice(/*dim=*/0, /*start=*/0, /*end=*/context_unmatched)
          .to(i32_dev_opts.dtype())
          .contiguous();
  auto new_key = key_seq.slice(0, 0, context_unmatched).contiguous();
  auto new_value = value_seq.slice(0, 0, context_unmatched).contiguous();

  CHECK(run_scatter_pa_kv_cache(
      new_key, new_value, key_cache, value_cache, new_slots_dev_i32, stream))
      << "run_scatter_pa_kv_cache failed";

  auto key_cache_bnbsh =
      key_cache.view({cache_block_count, block_size, num_kv_heads * head_dim});
  auto value_cache_bnbsh =
      value_cache.view({cache_block_count, block_size, num_kv_heads * head_dim});
  auto block_table = sample_metadata.block_table.to(i32_dev_opts.dtype()).contiguous();

  auto out_opts =
      torch::TensorOptions().dtype(query.scalar_type()).device(query.device());
  auto lse_opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(query.device());
  constexpr float kSentinel = -31415.0f;
  auto make_out = [&](int64_t seq_len) {
    return torch::full({1, seq_len, num_heads, head_dim}, kSentinel, out_opts);
  };
  auto make_lse = [&](int64_t seq_len) {
    return torch::full({1, num_heads, seq_len, 1}, kSentinel, lse_opts);
  };

  auto crt_query = query.contiguous();
  auto crt_out = make_out(context_unmatched + r + t);
  auto crt_lse = make_lse(context_unmatched + r + t);
  auto crt_pa_plan = plan_paged_attention_v3(
      crt_query,
      key_cache_bnbsh,
      value_cache_bnbsh,
      block_table,
      /*attn_mask=*/torch::Tensor(),
      /*sparse_mode=*/0,
      /*pre_tokens=*/kDefaultWindow,
      /*next_tokens=*/kDefaultWindow,
      /*total_kv_len=*/h + c,
      /*block_size=*/block_size,
      /*num_kv_heads=*/num_kv_heads,
      crt_out,
      crt_lse);
  execute_planned_paged_attention(crt_pa_plan, query.device(), stream);
  destroy_planned_paged_attention(crt_pa_plan);
  output.slice(1, 0, context_unmatched).copy_(crt_out.slice(1, 0, context_unmatched));

  auto rt_tgt_query =
      query.slice(1, local_rt_start, local_rt_start + r + t).contiguous();
  auto rt_key = key.slice(1, local_rt_start, local_rt_start + r).contiguous();
  auto rt_value = value.slice(1, local_rt_start, local_rt_start + r).contiguous();
  auto rt_tgt_out = make_out(r + t);
  auto rt_tgt_lse = make_lse(r + t);
  SegmentAttentionMetadata rt_meta;
  rt_meta.attn_mask = sample_metadata.compressed_causal_mask;
  rt_meta.sparse_param.sparse_mode = 2;
  rt_meta.sparse_param.pre_tokens = kDefaultWindow;
  rt_meta.sparse_param.next_tokens = kDefaultWindow;
  auto rt_plan =
      plan_segment_attention(rt_tgt_query, rt_key, rt_value, rt_meta, rt_tgt_out, rt_tgt_lse);
  execute_planned_attention(rt_plan, query.device(), stream);
  destroy_planned_attention(rt_plan);

  auto rt_update_plan = plan_attention_update_like_benchmark(
      {crt_out.slice(1, context_unmatched, context_unmatched + r).contiguous(),
       rt_tgt_out.slice(1, 0, r).contiguous()},
      {crt_lse.slice(2, context_unmatched, context_unmatched + r).contiguous(),
       rt_tgt_lse.slice(2, 0, r).contiguous()});
  execute_planned_attention_update(rt_update_plan, query.device(), stream);
  auto rt_merged = rt_update_plan.out_flat.view({rt_update_plan.b,
                                                 rt_update_plan.s,
                                                 rt_update_plan.n,
                                                 rt_update_plan.d})
                       .to(rt_update_plan.out_scalar_type);
  destroy_planned_attention_update(rt_update_plan);
  output.slice(1, local_rt_start, local_rt_start + r).copy_(rt_merged);

  auto target_query =
      query.slice(1, local_target_start, local_target_start + t).contiguous();
  auto target_key = key.slice(1, local_target_start, local_target_start + t).contiguous();
  auto target_value =
      value.slice(1, local_target_start, local_target_start + t).contiguous();
  auto target_self_out = make_out(t);
  auto target_self_lse = make_lse(t);
  SegmentAttentionMetadata self_meta;
  self_meta.attn_mask = create_diagonal_mask(t, query.device());
  self_meta.sparse_param.sparse_mode = 0;
  self_meta.sparse_param.pre_tokens = 0;
  self_meta.sparse_param.next_tokens = 0;
  auto self_plan = plan_segment_attention(target_query,
                                          target_key,
                                          target_value,
                                          self_meta,
                                          target_self_out,
                                          target_self_lse);
  execute_planned_attention(self_plan, query.device(), stream);
  destroy_planned_attention(self_plan);

  auto target_update_plan = plan_attention_update_like_benchmark(
      {crt_out.slice(1, context_unmatched + r, context_unmatched + r + t).contiguous(),
       rt_tgt_out.slice(1, r, r + t).contiguous(),
       target_self_out.contiguous()},
      {crt_lse.slice(2, context_unmatched + r, context_unmatched + r + t).contiguous(),
       rt_tgt_lse.slice(2, r, r + t).contiguous(),
       target_self_lse.contiguous()});
  execute_planned_attention_update(target_update_plan, query.device(), stream);
  auto target_merged =
      target_update_plan.out_flat
          .view({target_update_plan.b,
                 target_update_plan.s,
                 target_update_plan.n,
                 target_update_plan.d})
          .to(target_update_plan.out_scalar_type);
  destroy_planned_attention_update(target_update_plan);
  output.slice(1, local_target_start, local_target_start + t).copy_(target_merged);
}

void partial_rt_matched(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    const SampleSegmentMetadata& sample_metadata,
    torch::Tensor& output) {
  CHECK_EQ(query.size(0), 1);
  CHECK(sample_metadata.block_table.defined())
      << "block_table is required for prefix-cache path";
  CHECK(sample_metadata.slot_mapping.defined())
      << "slot_mapping is required for prefix-cache path";

  const int64_t h = sample_metadata.history;
  const int64_t c = sample_metadata.context;
  const int64_t r = sample_metadata.real_time;
  const int64_t t = sample_metadata.target;
  CHECK_GE(sample_metadata.matched_prefix, h + c);
  const int64_t realtime_matched = sample_metadata.matched_prefix - (h + c);
  CHECK_LT(realtime_matched, r);

  const int64_t block_size = sample_metadata.block_size;
  const int64_t cache_block_count = key_cache.size(0);
  const int64_t num_kv_heads = key.size(2);
  const int64_t num_heads = query.size(2);
  const int64_t head_dim = key.size(3);
  const int64_t rt_unmatched = r - realtime_matched;
  CHECK_GT(rt_unmatched, 0);
  CHECK_GT(t, 0);
  CHECK_EQ(query.size(1), rt_unmatched + t)
      << "partial_rt_matched expects unmatched layout [rt_unmatched|t]";
  CHECK_GE(sample_metadata.slot_mapping.size(0), rt_unmatched)
      << "slot_mapping is shorter than realtime unmatched length";

  auto stream = c10_npu::getCurrentNPUStream(query.device().index()).stream();
  auto i32_dev_opts = torch::TensorOptions().dtype(torch::kInt32).device(query.device());

  CHECK_EQ(key_cache.dim(), 4);
  CHECK_EQ(value_cache.dim(), 4);
  CHECK_EQ(key_cache.size(1), block_size);
  CHECK_EQ(key_cache.size(2), num_kv_heads);
  CHECK_EQ(key_cache.size(3), head_dim);
  CHECK_EQ(value_cache.size(0), cache_block_count);
  CHECK_EQ(value_cache.size(1), block_size);
  CHECK_EQ(value_cache.size(2), num_kv_heads);
  CHECK_EQ(value_cache.size(3), head_dim);

  auto key_seq = key.select(0, 0).contiguous();
  auto value_seq = value.select(0, 0).contiguous();
  auto new_slots_dev_i32 =
      sample_metadata.slot_mapping
          .slice(/*dim=*/0, /*start=*/0, /*end=*/rt_unmatched)
          .to(i32_dev_opts.dtype())
          .contiguous();
  auto new_key = key_seq.slice(0, 0, rt_unmatched).contiguous();
  auto new_value = value_seq.slice(0, 0, rt_unmatched).contiguous();

  CHECK(run_scatter_pa_kv_cache(
      new_key, new_value, key_cache, value_cache, new_slots_dev_i32, stream))
      << "run_scatter_pa_kv_cache failed";

  auto key_cache_bnbsh =
      key_cache.view({cache_block_count, block_size, num_kv_heads * head_dim});
  auto value_cache_bnbsh =
      value_cache.view({cache_block_count, block_size, num_kv_heads * head_dim});
  auto block_table = sample_metadata.block_table.to(i32_dev_opts.dtype()).contiguous();

  const int64_t local_target_start = rt_unmatched;
  auto target_query = query.slice(1, local_target_start, local_target_start + t).contiguous();
  auto target_key = key.slice(1, local_target_start, local_target_start + t).contiguous();
  auto target_value = value.slice(1, local_target_start, local_target_start + t).contiguous();

  auto out_opts =
      torch::TensorOptions().dtype(target_query.scalar_type()).device(target_query.device());
  auto lse_opts = torch::TensorOptions().dtype(torch::kFloat32).device(target_query.device());
  constexpr float kSentinel = -31415.0f;
  auto make_out = [&](int64_t seq_len) {
    return torch::full({1, seq_len, num_heads, head_dim}, kSentinel, out_opts);
  };
  auto make_lse = [&](int64_t seq_len) {
    return torch::full({1, num_heads, seq_len, 1}, kSentinel, lse_opts);
  };

  auto [target_diag_out, target_diag_lse] =
      build_target_diagonal_attn_analytic(target_query, target_key, target_value);

  // Combine the old rt_unmatched sparse PA and target-prefix full PA:
  // Q=R_unmatched+T, KV=H+C+R, sparse_mode=4.
  auto combined_query = query.contiguous();
  auto combined_out = make_out(rt_unmatched + t);
  auto combined_lse = make_lse(rt_unmatched + t);
  auto combined_pa_plan = plan_paged_attention_v3(
      combined_query,
      key_cache_bnbsh,
      value_cache_bnbsh,
      block_table,
      sample_metadata.compressed_causal_mask,
      /*sparse_mode=*/4,
      /*pre_tokens=*/h + c + r,
      /*next_tokens=*/t,
      /*total_kv_len=*/h + c + r,
      /*block_size=*/block_size,
      /*num_kv_heads=*/num_kv_heads,
      combined_out,
      combined_lse);
  execute_planned_paged_attention(combined_pa_plan, query.device(), stream);

  auto target_prefix_out =
      combined_out.slice(1, local_target_start, local_target_start + t).contiguous();
  auto target_prefix_lse =
      combined_lse.slice(2, local_target_start, local_target_start + t).contiguous();
  auto target_update_plan = plan_attention_update_like_benchmark(
      {target_prefix_out, target_diag_out.contiguous()},
      {target_prefix_lse, target_diag_lse.contiguous()});
  execute_planned_attention_update(target_update_plan, query.device(), stream);
  auto target_merged =
      target_update_plan.out_flat
          .view({target_update_plan.b,
                 target_update_plan.s,
                 target_update_plan.n,
                 target_update_plan.d})
          .to(target_update_plan.out_scalar_type);
  destroy_planned_paged_attention(combined_pa_plan);
  destroy_planned_attention_update(target_update_plan);
  output.slice(1, 0, rt_unmatched).copy_(combined_out.slice(1, 0, rt_unmatched));
  output.slice(1, local_target_start, local_target_start + t).copy_(target_merged);
}

}  // namespace

MTGRAttentionImpl::MTGRAttentionImpl(int64_t num_heads,
                                     int64_t head_size,
                                     float scale,
                                     int64_t num_kv_heads)
    : num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads) {}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> MTGRAttentionImpl::forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    KVCache& kv_cache) {
  CHECK_EQ(query.dim(), 2);
  CHECK_EQ(key.dim(), 2);
  CHECK_EQ(value.dim(), 2);
  CHECK_EQ(query.size(1), num_heads_ * head_size_);
  CHECK_EQ(key.size(1), num_kv_heads_ * head_size_);
  CHECK_EQ(value.size(1), num_kv_heads_ * head_size_);
  CHECK_GT(scale_, 0.0f);

  torch::Tensor output = torch::empty_like(query);
  std::optional<torch::Tensor> output_lse = std::nullopt;
  if (attn_metadata.is_dummy) {
    return std::make_tuple(output, output_lse);
  }

  const int64_t q_tokens = query.size(0);
  const int64_t kv_tokens = key.size(0);
  const auto& q_seq_lens_host = attn_metadata.q_seq_lens;
  const auto& kv_seq_lens_host = attn_metadata.kv_seq_lens;
  const auto lowered_mtgr = lower_mtgr_to_npu_four_segment(attn_metadata);
  const auto& history_lens_host = lowered_mtgr.history_lens;
  const auto& context_lens_host = lowered_mtgr.context_lens;
  const auto& real_time_lens_host = lowered_mtgr.real_time_lens;
  const auto& target_lens_host = lowered_mtgr.target_lens;
  const auto& matched_prefix_lens_host = lowered_mtgr.matched_prefix_lens;

  CHECK(q_seq_lens_host.defined()) << "q_seq_lens must be defined";
  CHECK(kv_seq_lens_host.defined()) << "kv_seq_lens must be defined";
  CHECK(history_lens_host.defined()) << "history_lens must be defined";
  CHECK(context_lens_host.defined()) << "context_lens must be defined";
  CHECK(real_time_lens_host.defined()) << "real_time_lens must be defined";
  CHECK(target_lens_host.defined()) << "target_lens must be defined";
  CHECK(matched_prefix_lens_host.defined())
      << "matched_prefix_lens must be defined";

  const int64_t batch_size = q_seq_lens_host.size(0);
  CHECK_EQ(kv_seq_lens_host.size(0), batch_size);
  CHECK_EQ(history_lens_host.size(0), batch_size);
  CHECK_EQ(context_lens_host.size(0), batch_size);
  CHECK_EQ(real_time_lens_host.size(0), batch_size);
  CHECK_EQ(target_lens_host.size(0), batch_size);
  CHECK_EQ(matched_prefix_lens_host.size(0), batch_size);

  auto compressed_causal_mask = create_compressed_causal_mask_2048(query.device());
  CHECK(attn_metadata.block_table.defined()) << "block_table must be defined";
  CHECK_EQ(attn_metadata.block_table.dim(), 2)
      << "block_table must be a 2D tensor [batch_size, max_blocks]";
  CHECK_EQ(attn_metadata.block_table.size(0), batch_size)
      << "block_table first dim must equal batch_size";
  CHECK(attn_metadata.slot_mapping.defined()) << "slot_mapping must be defined";
  CHECK_EQ(attn_metadata.slot_mapping.dim(), 1)
      << "slot_mapping must be a 1D tensor";
  CHECK_GE(attn_metadata.slot_mapping.size(0), kv_tokens)
      << "slot_mapping size must cover all kv tokens";

  auto key_cache = kv_cache.get_k_cache();
  auto value_cache = kv_cache.get_v_cache();
  const int64_t block_size = static_cast<int64_t>(FLAGS_block_size);
  CHECK_GT(block_size, 0) << "FLAGS_block_size must be positive";

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
    sample_metadata.block_size = block_size;
    sample_metadata.compressed_causal_mask = compressed_causal_mask;
    sample_metadata.block_table =
        attn_metadata.block_table.select(0, i).unsqueeze(0).contiguous();
    sample_metadata.slot_mapping =
        attn_metadata.slot_mapping.narrow(0, kv_offset, kv_len).contiguous();

    const int64_t h = sample_metadata.history;
    const int64_t c = sample_metadata.context;
    const int64_t r = sample_metadata.real_time;
    const int64_t matched = sample_metadata.matched_prefix;
    CHECK_GE(matched, 0);
    CHECK_LE(matched, h + c + r)
        << "matched prefix must stay within [0, history+context+realtime]";

    auto write_prefix_cache = [&](const SampleSegmentMetadata& metadata,
                                  int64_t prefix_cache_len,
                                  const char* tag) {
      if (!key_cache.defined() || !value_cache.defined() || prefix_cache_len <= 0) {
        return;
      }
      CHECK_LE(prefix_cache_len, key_i.size(1))
          << "prefix_cache_len must be <= local kv length, got prefix_cache_len="
          << prefix_cache_len << ", kv_len=" << key_i.size(1);
      CHECK_GE(metadata.slot_mapping.size(0), prefix_cache_len)
          << "slot_mapping is shorter than prefix cache length";

      auto stream = c10_npu::getCurrentNPUStream(query.device().index()).stream();
      auto i32_dev_opts =
          torch::TensorOptions().dtype(torch::kInt32).device(query.device());
      auto key_seq = key_i.select(0, 0).contiguous();
      auto value_seq = value_i.select(0, 0).contiguous();
      auto slots_dev_i32 =
          metadata.slot_mapping
              .slice(/*dim=*/0, /*start=*/0, /*end=*/prefix_cache_len)
              .to(i32_dev_opts.dtype())
              .contiguous();
      auto prefix_key = key_seq.slice(0, 0, prefix_cache_len).contiguous();
      auto prefix_value = value_seq.slice(0, 0, prefix_cache_len).contiguous();
      CHECK(run_scatter_pa_kv_cache(prefix_key,
                                    prefix_value,
                                    key_cache,
                                    value_cache,
                                    slots_dev_i32,
                                    stream))
          << "run_scatter_pa_kv_cache failed in " << tag;
    };

    if (matched == 0) {
      no_matched(query_i, key_i, value_i, sample_metadata, output_i);
      write_prefix_cache(sample_metadata,
                         h + c + r,
                         "no_matched prefix writeback");
    } else {
      CHECK(key_cache.defined() && value_cache.defined())
          << "KV cache is required for prefix-cache path";
      partial_rt_matched(query_i,
                         key_i,
                         value_i,
                         key_cache,
                         value_cache,
                         sample_metadata,
                         output_i);
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

}  // namespace layer
}  // namespace xllm
