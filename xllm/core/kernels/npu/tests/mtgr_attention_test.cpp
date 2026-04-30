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

#include "mtgr_attention_test.h"

#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/torch_npu.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <glog/logging.h>

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

#if __has_include("aclnn_mtgr_target_update.h")
#include "aclnn_mtgr_target_update.h"
#elif __has_include("/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/xllm/op_api/include/aclnn_mtgr_target_update.h")
#include "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/xllm/op_api/include/aclnn_mtgr_target_update.h"
#elif __has_include("/usr/local/Ascend/cann-8.5.0/opp/vendors/xllm/op_api/include/aclnn_mtgr_target_update.h")
#include "/usr/local/Ascend/cann-8.5.0/opp/vendors/xllm/op_api/include/aclnn_mtgr_target_update.h"
#else
#error "Cannot find aclnn_mtgr_target_update.h"
#endif

#if __has_include("aclnn_mtgr_target_update_v4.h")
#include "aclnn_mtgr_target_update_v4.h"
#elif __has_include("/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/xllm/op_api/include/aclnn_mtgr_target_update_v4.h")
#include "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/xllm/op_api/include/aclnn_mtgr_target_update_v4.h"
#elif __has_include("/usr/local/Ascend/cann-8.5.0/opp/vendors/xllm/op_api/include/aclnn_mtgr_target_update_v4.h")
#include "/usr/local/Ascend/cann-8.5.0/opp/vendors/xllm/op_api/include/aclnn_mtgr_target_update_v4.h"
#else
#error "Cannot find aclnn_mtgr_target_update_v4.h"
#endif

#include "common/global_flags.h"

namespace xllm::kernel::npu::test {
namespace {

constexpr int64_t kDefaultWindow = 2147483647LL;
constexpr uint64_t kPreallocWorkspaceBytes = 1024ULL * 1024ULL * 1024ULL;
constexpr uint64_t kPreallocScatterWorkspaceBytes = 128ULL * 1024ULL * 1024ULL;
constexpr int64_t kPartialRtMatchedNumerator = 4;
constexpr int64_t kPartialRtMatchedDenominator = 5;

struct SparseParam {
  int64_t sparse_mode = 0;
  int64_t pre_tokens = 0;
  int64_t next_tokens = 0;
};

struct SegmentAttentionMetadata {
  torch::Tensor attn_mask;
  SparseParam sparse_param;
};

struct RuntimeAttentionMetadata {
  torch::Tensor compressed_causal_mask;
  void* shared_workspace = nullptr;
  uint64_t shared_workspace_size = 0;
  void* scatter_workspace = nullptr;
  uint64_t scatter_workspace_size = 0;
};

struct SampleSegmentMetadata {
  int64_t history = 0;
  int64_t context = 0;
  int64_t real_time = 0;
  int64_t target = 0;
  int64_t matched_prefix = 0;
  int64_t block_size = 128;
  torch::Tensor compressed_causal_mask;
  torch::Tensor block_table;
  torch::Tensor slot_mapping;
};

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

struct ScatterPaKvCachePlan {
  uint64_t workspace_size = 0;
  void* workspace_ptr = nullptr;
  bool owns_workspace = false;
  aclOpExecutor* executor = nullptr;
  aclTensor* key_acl = nullptr;
  aclTensor* value_acl = nullptr;
  aclTensor* key_cache_acl = nullptr;
  aclTensor* value_cache_acl = nullptr;
  aclTensor* slot_mapping_acl = nullptr;
};

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

struct MtgrTargetUpdatePlan {
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclTensor* q_acl = nullptr;
  aclTensor* k_acl = nullptr;
  aclTensor* v_acl = nullptr;
  aclTensor* prefix_out_acl = nullptr;
  aclTensor* prefix_lse_acl = nullptr;
  aclTensor* out_acl = nullptr;
};

struct RawStagePerf {
  const char* name = nullptr;
  double workspace_ms = 0.0;
  double exec_ms = 0.0;
  aclrtEvent ev_start = nullptr;
  aclrtEvent ev_end = nullptr;
};

bool valid_tensor(const torch::Tensor& t) {
  return t.defined() && t.numel() > 0;
}

bool debug_finite_enabled() {
  const char* value = std::getenv("XLLM_MTGR_DEBUG_FINITE");
  return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

void print_finite_summary(const char* name, const torch::Tensor& tensor) {
  if (!debug_finite_enabled() || !valid_tensor(tensor)) {
    return;
  }
  const auto finite = torch::isfinite(tensor);
  const int64_t finite_count = finite.sum().item<int64_t>();
  const int64_t total_count = tensor.numel();
  std::fprintf(stderr,
               "[MTGRDebug][finite] %s finite=%ld/%ld dtype=%s dim=%ld\n",
               name,
               static_cast<long>(finite_count),
               static_cast<long>(total_count),
               c10::toString(tensor.scalar_type()),
               static_cast<long>(tensor.dim()));
  if (finite_count == total_count) {
    const auto tensor_f32 = tensor.to(torch::kFloat32);
    std::fprintf(stderr,
                 "[MTGRDebug][finite] %s min=%.9f max=%.9f\n",
                 name,
                 tensor_f32.min().item<float>(),
                 tensor_f32.max().item<float>());
  }
}

torch::Tensor create_compressed_causal_mask_2048(const torch::Device& device) {
  return torch::ones({1, 1, 2048, 2048},
                     torch::TensorOptions().dtype(torch::kBool).device(device))
      .triu(1)
      .contiguous();
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
  auto* acl_t =
      aclCreateTensor(t.sizes().data(),
                      t.sizes().size(),
                      to_acl_dtype(t.scalar_type()),
                      t.strides().data(),
                      0,
                      ACL_FORMAT_ND,
                      t.sizes().data(),
                      t.sizes().size(),
                      t.data_ptr());
  CHECK_NE(acl_t, nullptr);
  return acl_t;
}

aclTensor* torch_to_acl_tensor_with_strides(const torch::Tensor& t) {
  auto* acl_t =
      aclCreateTensor(t.sizes().data(),
                      t.sizes().size(),
                      to_acl_dtype(t.scalar_type()),
                      t.strides().data(),
                      0,
                      ACL_FORMAT_ND,
                      t.sizes().data(),
                      t.sizes().size(),
                      t.data_ptr());
  CHECK_NE(acl_t, nullptr);
  return acl_t;
}

template <size_t N>
void create_stage_events(std::array<RawStagePerf, N>* stages) {
  if (stages == nullptr) {
    return;
  }
  for (auto& stage : *stages) {
    CHECK_EQ(aclrtCreateEventWithFlag(&stage.ev_start, ACL_EVENT_TIME_LINE), ACL_SUCCESS);
    CHECK_EQ(aclrtCreateEventWithFlag(&stage.ev_end, ACL_EVENT_TIME_LINE), ACL_SUCCESS);
  }
}

template <size_t N>
void destroy_stage_events(std::array<RawStagePerf, N>* stages) {
  if (stages == nullptr) {
    return;
  }
  for (auto& stage : *stages) {
    if (stage.ev_start != nullptr) {
      CHECK_EQ(aclrtDestroyEvent(stage.ev_start), ACL_SUCCESS);
      stage.ev_start = nullptr;
    }
    if (stage.ev_end != nullptr) {
      CHECK_EQ(aclrtDestroyEvent(stage.ev_end), ACL_SUCCESS);
      stage.ev_end = nullptr;
    }
  }
}

void record_stage_begin(RawStagePerf* stage, aclrtStream stream) {
  if (stage == nullptr) {
    return;
  }
  CHECK_EQ(aclrtRecordEvent(stage->ev_start, stream), ACL_SUCCESS);
}

void record_stage_end(RawStagePerf* stage, aclrtStream stream) {
  if (stage == nullptr) {
    return;
  }
  CHECK_EQ(aclrtRecordEvent(stage->ev_end, stream), ACL_SUCCESS);
}

template <size_t N>
double sum_stage_workspace_ms(const std::array<RawStagePerf, N>& stages) {
  double sum = 0.0;
  for (const auto& stage : stages) {
    sum += stage.workspace_ms;
  }
  return sum;
}

template <size_t N>
void finalize_forward_perf(const char* path_name,
                           std::array<RawStagePerf, N>* stages,
                           double total_dev_ms,
                           double total_wall_ms,
                           MTGRForwardPerf* perf_sink) {
  if (perf_sink == nullptr || stages == nullptr) {
    return;
  }
  perf_sink->path_name = path_name;
  perf_sink->stages.clear();
  perf_sink->stages.reserve(N);
  for (const auto& stage : *stages) {
    MTGRStagePerf public_stage;
    public_stage.name = stage.name == nullptr ? "" : stage.name;
    public_stage.workspace_ms = stage.workspace_ms;
    public_stage.exec_ms = stage.exec_ms;
    perf_sink->stages.push_back(std::move(public_stage));
  }
  perf_sink->device_total_ms = total_dev_ms;
  perf_sink->wall_total_ms = std::max(0.0, total_wall_ms - sum_stage_workspace_ms(*stages));
}

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
  CHECK_NE(plan.k_list, nullptr);
  CHECK_NE(plan.v_list, nullptr);

  char layout[] = "BSND";
  const auto ret = aclnnFusedInferAttentionScoreV3GetWorkspaceSize(
      plan.q_acl,
      plan.k_list,
      plan.v_list,
      nullptr,
      plan.mask_acl,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      num_heads,
      scale,
      attn_metadata.sparse_param.pre_tokens,
      attn_metadata.sparse_param.next_tokens,
      layout,
      num_kv_heads,
      attn_metadata.sparse_param.sparse_mode,
      1,
      0,
      0,
      (lse.has_value() && valid_tensor(lse.value())),
      0,
      0,
      plan.out_acl,
      plan.lse_acl,
      &plan.workspace_size,
      &plan.executor);
  CHECK_EQ(ret, ACL_SUCCESS)
      << "aclnnFusedInferAttentionScoreV3GetWorkspaceSize failed: " << ret;
  return plan;
}

void destroy_planned_attention(AclPlan& plan) {
  if (plan.k_list != nullptr) {
    CHECK_EQ(aclDestroyTensorList(plan.k_list), ACL_SUCCESS);
    plan.k_list = nullptr;
    plan.k_acl = nullptr;
  }
  if (plan.v_list != nullptr) {
    CHECK_EQ(aclDestroyTensorList(plan.v_list), ACL_SUCCESS);
    plan.v_list = nullptr;
    plan.v_acl = nullptr;
  }
  if (plan.q_acl != nullptr) {
    CHECK_EQ(aclDestroyTensor(plan.q_acl), ACL_SUCCESS);
    plan.q_acl = nullptr;
  }
  if (plan.out_acl != nullptr) {
    CHECK_EQ(aclDestroyTensor(plan.out_acl), ACL_SUCCESS);
    plan.out_acl = nullptr;
  }
  if (plan.mask_acl != nullptr) {
    CHECK_EQ(aclDestroyTensor(plan.mask_acl), ACL_SUCCESS);
    plan.mask_acl = nullptr;
  }
  if (plan.lse_acl != nullptr) {
    CHECK_EQ(aclDestroyTensor(plan.lse_acl), ACL_SUCCESS);
    plan.lse_acl = nullptr;
  }
}

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
  plan.q_acl = torch_to_acl_tensor(query_bsnd);
  plan.key_cache_acl = torch_to_acl_tensor(key_cache_bnbsh);
  plan.value_cache_acl = torch_to_acl_tensor(value_cache_bnbsh);
  plan.block_table_acl = torch_to_acl_tensor(block_table);
  plan.out_acl = torch_to_acl_tensor(output_bsnd);
  if (valid_tensor(attn_mask)) {
    plan.mask_acl = torch_to_acl_tensor_with_strides(attn_mask);
  }
  if (lse.has_value() && valid_tensor(lse.value())) {
    plan.lse_acl = torch_to_acl_tensor(lse.value());
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

  const float scale = 1.0f / std::sqrt(static_cast<float>(query_bsnd.size(3)));
  const int64_t num_heads = query_bsnd.size(2);
  char layout[] = "BSND";
  const auto ret = aclnnFusedInferAttentionScoreV3GetWorkspaceSize(
      plan.q_acl,
      plan.k_list,
      plan.v_list,
      nullptr,
      plan.mask_acl,
      plan.actual_q,
      plan.actual_kv,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      plan.block_table_acl,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      num_heads,
      scale,
      pre_tokens,
      next_tokens,
      layout,
      num_kv_heads,
      sparse_mode,
      1,
      block_size,
      0,
      (lse.has_value() && valid_tensor(lse.value())),
      0,
      0,
      plan.out_acl,
      plan.lse_acl,
      &plan.workspace_size,
      &plan.executor);
  CHECK_EQ(ret, ACL_SUCCESS)
      << "aclnnFusedInferAttentionScoreV3GetWorkspaceSize (paged) failed: " << ret;
  return plan;
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
    CHECK_EQ(aclDestroyTensorList(plan.k_list), ACL_SUCCESS);
    plan.k_list = nullptr;
    plan.key_cache_acl = nullptr;
  }
  if (plan.v_list != nullptr) {
    CHECK_EQ(aclDestroyTensorList(plan.v_list), ACL_SUCCESS);
    plan.v_list = nullptr;
    plan.value_cache_acl = nullptr;
  }
  if (plan.q_acl != nullptr) {
    CHECK_EQ(aclDestroyTensor(plan.q_acl), ACL_SUCCESS);
    plan.q_acl = nullptr;
  }
  if (plan.key_cache_acl != nullptr) {
    CHECK_EQ(aclDestroyTensor(plan.key_cache_acl), ACL_SUCCESS);
    plan.key_cache_acl = nullptr;
  }
  if (plan.value_cache_acl != nullptr) {
    CHECK_EQ(aclDestroyTensor(plan.value_cache_acl), ACL_SUCCESS);
    plan.value_cache_acl = nullptr;
  }
  if (plan.block_table_acl != nullptr) {
    CHECK_EQ(aclDestroyTensor(plan.block_table_acl), ACL_SUCCESS);
    plan.block_table_acl = nullptr;
  }
  if (plan.out_acl != nullptr) {
    CHECK_EQ(aclDestroyTensor(plan.out_acl), ACL_SUCCESS);
    plan.out_acl = nullptr;
  }
  if (plan.mask_acl != nullptr) {
    CHECK_EQ(aclDestroyTensor(plan.mask_acl), ACL_SUCCESS);
    plan.mask_acl = nullptr;
  }
  if (plan.lse_acl != nullptr) {
    CHECK_EQ(aclDestroyTensor(plan.lse_acl), ACL_SUCCESS);
    plan.lse_acl = nullptr;
  }
}

void launch_attention_without_memset(const AclPlan& plan,
                                     const RuntimeAttentionMetadata& attn_metadata,
                                     aclrtStream stream) {
  void* ws_ptr = nullptr;
  if (plan.workspace_size > 0) {
    CHECK_NE(attn_metadata.shared_workspace, nullptr);
    CHECK_GE(attn_metadata.shared_workspace_size, plan.workspace_size);
    ws_ptr = attn_metadata.shared_workspace;
  }
  const auto ret =
      aclnnFusedInferAttentionScoreV3(ws_ptr, plan.workspace_size, plan.executor, stream);
  CHECK_EQ(ret, ACL_SUCCESS) << "aclnnFusedInferAttentionScoreV3 failed: " << ret;
}

void launch_paged_attention_without_memset(const PagedAttentionPlan& plan,
                                           const RuntimeAttentionMetadata& attn_metadata,
                                           aclrtStream stream) {
  void* ws_ptr = nullptr;
  if (plan.workspace_size > 0) {
    CHECK_NE(attn_metadata.shared_workspace, nullptr);
    CHECK_GE(attn_metadata.shared_workspace_size, plan.workspace_size);
    ws_ptr = attn_metadata.shared_workspace;
  }
  const auto ret =
      aclnnFusedInferAttentionScoreV3(ws_ptr, plan.workspace_size, plan.executor, stream);
  CHECK_EQ(ret, ACL_SUCCESS) << "aclnnFusedInferAttentionScoreV3 (paged) failed: " << ret;
}

void destroy_planned_scatter_pa_kv_cache(ScatterPaKvCachePlan& plan) {
  if (plan.key_acl != nullptr) {
    CHECK_EQ(aclDestroyTensor(plan.key_acl), ACL_SUCCESS);
    plan.key_acl = nullptr;
  }
  if (plan.value_acl != nullptr) {
    CHECK_EQ(aclDestroyTensor(plan.value_acl), ACL_SUCCESS);
    plan.value_acl = nullptr;
  }
  if (plan.key_cache_acl != nullptr) {
    CHECK_EQ(aclDestroyTensor(plan.key_cache_acl), ACL_SUCCESS);
    plan.key_cache_acl = nullptr;
  }
  if (plan.value_cache_acl != nullptr) {
    CHECK_EQ(aclDestroyTensor(plan.value_cache_acl), ACL_SUCCESS);
    plan.value_cache_acl = nullptr;
  }
  if (plan.slot_mapping_acl != nullptr) {
    CHECK_EQ(aclDestroyTensor(plan.slot_mapping_acl), ACL_SUCCESS);
    plan.slot_mapping_acl = nullptr;
  }
  if (plan.workspace_ptr != nullptr && plan.owns_workspace) {
    CHECK_EQ(aclrtFree(plan.workspace_ptr), ACL_SUCCESS);
  }
  plan.workspace_ptr = nullptr;
  plan.owns_workspace = false;
  plan.workspace_size = 0;
  plan.executor = nullptr;
}

bool plan_scatter_pa_kv_cache(const torch::Tensor& key,
                              const torch::Tensor& value,
                              torch::Tensor& key_cache,
                              torch::Tensor& value_cache,
                              const torch::Tensor& slot_mapping,
                              const RuntimeAttentionMetadata* runtime_meta,
                              ScatterPaKvCachePlan* plan) {
  CHECK_NE(plan, nullptr);
  *plan = ScatterPaKvCachePlan();
  plan->key_acl = torch_to_acl_tensor(key);
  plan->value_acl = torch_to_acl_tensor(value);
  plan->key_cache_acl = torch_to_acl_tensor(key_cache);
  plan->value_cache_acl = torch_to_acl_tensor(value_cache);
  plan->slot_mapping_acl = torch_to_acl_tensor(slot_mapping);

  char cache_mode[] = "Norm";
  char scatter_mode[] = "None";
  auto ret = aclnnScatterPaKvCacheGetWorkspaceSize(plan->key_acl,
                                                   plan->key_cache_acl,
                                                   plan->slot_mapping_acl,
                                                   plan->value_acl,
                                                   plan->value_cache_acl,
                                                   nullptr,
                                                   nullptr,
                                                   nullptr,
                                                   cache_mode,
                                                   scatter_mode,
                                                   nullptr,
                                                   nullptr,
                                                   &plan->workspace_size,
                                                   &plan->executor);
  if (ret != ACL_SUCCESS) {
    std::fprintf(stderr,
                 "aclnnScatterPaKvCacheGetWorkspaceSize failed: ret=%d key_shape=[%ld,%ld,%ld] "
                 "cache_shape=[%ld,%ld,%ld,%ld] slots=%ld\n",
                 static_cast<int>(ret),
                 key.size(0),
                 key.size(1),
                 key.size(2),
                 key_cache.size(0),
                 key_cache.size(1),
                 key_cache.size(2),
                 key_cache.size(3),
                 slot_mapping.size(0));
    destroy_planned_scatter_pa_kv_cache(*plan);
    return false;
  }

  if (plan->workspace_size > 0) {
    if (runtime_meta != nullptr && runtime_meta->scatter_workspace != nullptr &&
        runtime_meta->scatter_workspace_size >= plan->workspace_size) {
      plan->workspace_ptr = runtime_meta->scatter_workspace;
      plan->owns_workspace = false;
    } else {
      ret = aclrtMalloc(&plan->workspace_ptr, plan->workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
      if (ret != ACL_SUCCESS) {
        std::fprintf(stderr,
                     "aclrtMalloc scatter workspace failed: ret=%d workspace_size=%lu\n",
                     static_cast<int>(ret),
                     static_cast<unsigned long>(plan->workspace_size));
        destroy_planned_scatter_pa_kv_cache(*plan);
        return false;
      }
      plan->owns_workspace = true;
    }
  }
  return true;
}

bool launch_planned_scatter_pa_kv_cache(const ScatterPaKvCachePlan& plan,
                                        aclrtStream stream) {
  const auto ret =
      aclnnScatterPaKvCache(plan.workspace_ptr, plan.workspace_size, plan.executor, stream);
  if (ret != ACL_SUCCESS) {
    std::fprintf(stderr,
                 "aclnnScatterPaKvCache failed: ret=%d workspace_size=%lu\n",
                 static_cast<int>(ret),
                 static_cast<unsigned long>(plan.workspace_size));
    return false;
  }
  return true;
}

bool run_scatter_pa_kv_cache(const torch::Tensor& key,
                             const torch::Tensor& value,
                             torch::Tensor& key_cache,
                             torch::Tensor& value_cache,
                             const torch::Tensor& slot_mapping,
                             const RuntimeAttentionMetadata* runtime_meta,
                             aclrtStream stream) {
  ScatterPaKvCachePlan plan;
  if (!plan_scatter_pa_kv_cache(
          key, value, key_cache, value_cache, slot_mapping, runtime_meta, &plan)) {
    return false;
  }
  if (!launch_planned_scatter_pa_kv_cache(plan, stream)) {
    destroy_planned_scatter_pa_kv_cache(plan);
    return false;
  }
  const auto ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    std::fprintf(stderr,
                 "aclrtSynchronizeStream after scatter failed: ret=%d workspace_size=%lu\n",
                 static_cast<int>(ret),
                 static_cast<unsigned long>(plan.workspace_size));
    destroy_planned_scatter_pa_kv_cache(plan);
    return false;
  }
  destroy_planned_scatter_pa_kv_cache(plan);
  return true;
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
    CHECK(local_out_flats[i].is_contiguous());
    CHECK_EQ(local_out_flats[i].scalar_type(), torch::kFloat32);
    CHECK_EQ(lse_flats[i].dim(), 1);
    CHECK_EQ(lse_flats[i].size(0), bsn);
    CHECK(lse_flats[i].is_contiguous());
    CHECK_EQ(lse_flats[i].scalar_type(), torch::kFloat32);
    plan.local_out_acls[i] = torch_to_acl_tensor(plan.local_out_flats[i]);
    plan.lse_acls[i] = torch_to_acl_tensor(plan.lse_flats[i]);
  }

  plan.out_flat = torch::empty(
      {bsn, d},
      torch::TensorOptions().dtype(torch::kFloat32).device(local_out_flats[0].device()));
  plan.out_acl = torch_to_acl_tensor(plan.out_flat);

  plan.local_out_list =
      aclCreateTensorList(plan.local_out_acls.data(), static_cast<int32_t>(plan.local_out_acls.size()));
  CHECK_NE(plan.local_out_list, nullptr);
  plan.lse_list =
      aclCreateTensorList(plan.lse_acls.data(), static_cast<int32_t>(plan.lse_acls.size()));
  CHECK_NE(plan.lse_list, nullptr);

  const auto ret = aclnnAttentionUpdateGetWorkspaceSize(plan.lse_list,
                                                        plan.local_out_list,
                                                        0,
                                                        plan.out_acl,
                                                        nullptr,
                                                        &plan.workspace_size,
                                                        &plan.executor);
  CHECK_EQ(ret, ACL_SUCCESS) << "aclnnAttentionUpdateGetWorkspaceSize failed: " << ret;
  return plan;
}

void destroy_planned_attention_update(AttentionUpdatePlan& plan) {
  if (plan.local_out_list != nullptr) {
    CHECK_EQ(aclDestroyTensorList(plan.local_out_list), ACL_SUCCESS);
    plan.local_out_list = nullptr;
  }
  if (plan.lse_list != nullptr) {
    CHECK_EQ(aclDestroyTensorList(plan.lse_list), ACL_SUCCESS);
    plan.lse_list = nullptr;
  }
  if (plan.out_acl != nullptr) {
    CHECK_EQ(aclDestroyTensor(plan.out_acl), ACL_SUCCESS);
    plan.out_acl = nullptr;
  }
}

void launch_attention_update_without_memset(const AttentionUpdatePlan& plan,
                                            const RuntimeAttentionMetadata& attn_metadata,
                                            aclrtStream stream) {
  void* ws_ptr = nullptr;
  if (plan.workspace_size > 0) {
    CHECK_NE(attn_metadata.shared_workspace, nullptr);
    CHECK_GE(attn_metadata.shared_workspace_size, plan.workspace_size);
    ws_ptr = attn_metadata.shared_workspace;
  }
  const auto ret = aclnnAttentionUpdate(ws_ptr, plan.workspace_size, plan.executor, stream);
  CHECK_EQ(ret, ACL_SUCCESS) << "aclnnAttentionUpdate failed: " << ret;
}

MtgrTargetUpdatePlan plan_mtgr_target_update_impl(const torch::Tensor& target_query,
                                                  const torch::Tensor& target_key,
                                                  const torch::Tensor& target_value,
                                                  const torch::Tensor& prefix_out,
                                                  const torch::Tensor& prefix_lse,
                                                  torch::Tensor& out,
                                                  int variant) {
  CHECK_EQ(target_query.dim(), 4);
  CHECK_EQ(prefix_lse.dim(), 4);
  CHECK_EQ(target_query.sizes(), target_key.sizes());
  CHECK_EQ(target_query.sizes(), target_value.sizes());
  CHECK_EQ(target_query.sizes(), out.sizes());
  CHECK_EQ(prefix_out.dim(), 4);
  CHECK_EQ(prefix_out.size(0), target_query.size(0));
  CHECK_GE(prefix_out.size(1), target_query.size(1));
  CHECK_EQ(prefix_out.size(2), target_query.size(2));
  CHECK_EQ(prefix_out.size(3), target_query.size(3));
  CHECK_EQ(prefix_lse.size(0), target_query.size(0));
  CHECK_EQ(prefix_lse.size(1), target_query.size(2));
  CHECK_EQ(prefix_lse.size(2), prefix_out.size(1));
  CHECK_EQ(prefix_lse.size(3), 1);
  CHECK(target_query.is_contiguous());
  CHECK(target_key.is_contiguous());
  CHECK(target_value.is_contiguous());
  CHECK(prefix_out.is_contiguous());
  CHECK(prefix_lse.is_contiguous());
  CHECK(out.is_contiguous());
  CHECK_EQ(prefix_lse.scalar_type(), torch::kFloat32);

  MtgrTargetUpdatePlan plan;
  plan.q_acl = torch_to_acl_tensor(target_query);
  plan.k_acl = torch_to_acl_tensor(target_key);
  plan.v_acl = torch_to_acl_tensor(target_value);
  plan.prefix_out_acl = torch_to_acl_tensor(prefix_out);
  plan.prefix_lse_acl = torch_to_acl_tensor(prefix_lse);
  plan.out_acl = torch_to_acl_tensor(out);

  aclError ret = ACL_SUCCESS;
  if (variant == 4) {
    ret = aclnnMtgrTargetUpdateV4GetWorkspaceSize(plan.q_acl,
                                                  plan.k_acl,
                                                  plan.v_acl,
                                                  plan.prefix_out_acl,
                                                  plan.prefix_lse_acl,
                                                  plan.out_acl,
                                                  &plan.workspace_size,
                                                  &plan.executor);
    CHECK_EQ(ret, ACL_SUCCESS) << "aclnnMtgrTargetUpdateV4GetWorkspaceSize failed: " << ret;
  } else {
    ret = aclnnMtgrTargetUpdateGetWorkspaceSize(plan.q_acl,
                                                plan.k_acl,
                                                plan.v_acl,
                                                plan.prefix_out_acl,
                                                plan.prefix_lse_acl,
                                                plan.out_acl,
                                                &plan.workspace_size,
                                                &plan.executor);
    CHECK_EQ(ret, ACL_SUCCESS) << "aclnnMtgrTargetUpdateGetWorkspaceSize failed: " << ret;
  }
  return plan;
}

MtgrTargetUpdatePlan plan_mtgr_target_update(const torch::Tensor& target_query,
                                             const torch::Tensor& target_key,
                                             const torch::Tensor& target_value,
                                             const torch::Tensor& prefix_out,
                                             const torch::Tensor& prefix_lse,
                                             torch::Tensor& out) {
  return plan_mtgr_target_update_impl(
      target_query, target_key, target_value, prefix_out, prefix_lse, out, 1);
}

MtgrTargetUpdatePlan plan_mtgr_target_update_v4(const torch::Tensor& target_query,
                                                const torch::Tensor& target_key,
                                                const torch::Tensor& target_value,
                                                const torch::Tensor& prefix_out,
                                                const torch::Tensor& prefix_lse,
                                                torch::Tensor& out) {
  return plan_mtgr_target_update_impl(
      target_query, target_key, target_value, prefix_out, prefix_lse, out, 4);
}

void destroy_planned_mtgr_target_update(MtgrTargetUpdatePlan& plan) {
  if (plan.q_acl != nullptr) {
    CHECK_EQ(aclDestroyTensor(plan.q_acl), ACL_SUCCESS);
    plan.q_acl = nullptr;
  }
  if (plan.k_acl != nullptr) {
    CHECK_EQ(aclDestroyTensor(plan.k_acl), ACL_SUCCESS);
    plan.k_acl = nullptr;
  }
  if (plan.v_acl != nullptr) {
    CHECK_EQ(aclDestroyTensor(plan.v_acl), ACL_SUCCESS);
    plan.v_acl = nullptr;
  }
  if (plan.prefix_out_acl != nullptr) {
    CHECK_EQ(aclDestroyTensor(plan.prefix_out_acl), ACL_SUCCESS);
    plan.prefix_out_acl = nullptr;
  }
  if (plan.prefix_lse_acl != nullptr) {
    CHECK_EQ(aclDestroyTensor(plan.prefix_lse_acl), ACL_SUCCESS);
    plan.prefix_lse_acl = nullptr;
  }
  if (plan.out_acl != nullptr) {
    CHECK_EQ(aclDestroyTensor(plan.out_acl), ACL_SUCCESS);
    plan.out_acl = nullptr;
  }
}

void launch_mtgr_target_update_without_memset_impl(const MtgrTargetUpdatePlan& plan,
                                                   const RuntimeAttentionMetadata& attn_metadata,
                                                   aclrtStream stream,
                                                   int variant) {
  void* ws_ptr = nullptr;
  if (plan.workspace_size > 0) {
    CHECK_NE(attn_metadata.shared_workspace, nullptr);
    CHECK_GE(attn_metadata.shared_workspace_size, plan.workspace_size);
    ws_ptr = attn_metadata.shared_workspace;
  }
  aclError ret = ACL_SUCCESS;
  if (variant == 4) {
    ret = aclnnMtgrTargetUpdateV4(ws_ptr, plan.workspace_size, plan.executor, stream);
    CHECK_EQ(ret, ACL_SUCCESS) << "aclnnMtgrTargetUpdateV4 failed: " << ret;
  } else {
    ret = aclnnMtgrTargetUpdate(ws_ptr, plan.workspace_size, plan.executor, stream);
    CHECK_EQ(ret, ACL_SUCCESS) << "aclnnMtgrTargetUpdate failed: " << ret;
  }
}

void launch_mtgr_target_update_without_memset(const MtgrTargetUpdatePlan& plan,
                                              const RuntimeAttentionMetadata& attn_metadata,
                                              aclrtStream stream) {
  launch_mtgr_target_update_without_memset_impl(plan, attn_metadata, stream, 1);
}

void launch_mtgr_target_update_v4_without_memset(const MtgrTargetUpdatePlan& plan,
                                                 const RuntimeAttentionMetadata& attn_metadata,
                                                 aclrtStream stream) {
  launch_mtgr_target_update_without_memset_impl(plan, attn_metadata, stream, 4);
}

std::pair<torch::Tensor, torch::Tensor> build_target_diagonal_attn_analytic(
    const torch::Tensor& target_query,
    const torch::Tensor& target_key,
    const torch::Tensor& target_value) {
  CHECK_EQ(target_query.dim(), 4);
  CHECK_EQ(target_query.sizes(), target_key.sizes());
  CHECK_EQ(target_query.sizes(), target_value.sizes());
  const double scale = 1.0 / std::sqrt(static_cast<double>(target_query.size(3)));
  auto output = target_value.contiguous();
  auto q_f32 = target_query.to(torch::kFloat32);
  auto k_f32 = target_key.to(torch::kFloat32);
  auto lse = (q_f32 * k_f32).sum(-1, true).mul(scale).permute({0, 2, 1, 3}).contiguous();
  return {output, lse};
}

void prefill_matched_prefix_cache(const torch::Tensor& full_key_bsnd,
                                  const torch::Tensor& full_value_bsnd,
                                  int64_t matched_prefix,
                                  const std::vector<int32_t>& block_table,
                                  int64_t block_size,
                                  torch::Tensor& key_cache,
                                  torch::Tensor& value_cache) {
  if (matched_prefix <= 0) {
    return;
  }
  auto key_seq = full_key_bsnd.select(0, 0).contiguous();
  auto value_seq = full_value_bsnd.select(0, 0).contiguous();
  auto key_cache_flat =
      key_cache.view({key_cache.size(0) * block_size, key_cache.size(2), key_cache.size(3)});
  auto value_cache_flat =
      value_cache.view({value_cache.size(0) * block_size, value_cache.size(2), value_cache.size(3)});
  auto prefix_slots = build_slot_mapping(block_table, block_size, 0, matched_prefix);
  auto prefix_slots_dev_i64 =
      torch::tensor(prefix_slots,
                    torch::TensorOptions().dtype(torch::kInt64).device(key_cache.device()));
  key_cache_flat.index_copy_(0, prefix_slots_dev_i64, key_seq.slice(0, 0, matched_prefix));
  value_cache_flat.index_copy_(0, prefix_slots_dev_i64, value_seq.slice(0, 0, matched_prefix));
}

torch::Tensor build_irregular_genrec_mask_on_prealloc(int64_t history,
                                                      int64_t context,
                                                      int64_t realtime,
                                                      int64_t target,
                                                      torch::Tensor& mask_storage) {
  const int64_t total = history + context + realtime + target;
  CHECK_GT(total, 0);
  auto mask = mask_storage.slice(0, 0, total).slice(1, 0, total);
  mask.fill_(true);

  const int64_t hc_end = history + context;
  const int64_t rt_begin = hc_end;
  const int64_t rt_end = hc_end + realtime;
  const int64_t tgt_begin = rt_end;

  if (history > 0) {
    auto hist_blk = mask.slice(0, 0, history).slice(1, 0, history);
    hist_blk.tril_(0);
    hist_blk.logical_not_();
  }
  if (context > 0 && hc_end > 0) {
    mask.slice(0, history, hc_end).slice(1, 0, hc_end).fill_(false);
  }
  if (realtime > 0) {
    auto rt_rows = mask.slice(0, rt_begin, rt_end);
    if (rt_begin > 0) {
      rt_rows.slice(1, 0, rt_begin).fill_(false);
    }
    auto rt_blk = rt_rows.slice(1, rt_begin, rt_end);
    rt_blk.tril_(0);
    rt_blk.logical_not_();
  }
  if (target > 0) {
    auto tgt_rows = mask.slice(0, tgt_begin, total);
    if (tgt_begin > 0) {
      tgt_rows.slice(1, 0, tgt_begin).fill_(false);
    }
    auto tgt_blk = tgt_rows.slice(1, tgt_begin, total);
    tgt_blk.fill_(true);
    tgt_blk.diagonal(0, 0, 1).fill_(false);
  }
  return mask.unsqueeze(0).unsqueeze(0);
}

OneStagePerf run_one_stage_fia_with_device_mask_prealloc(const torch::Tensor& query,
                                                         const torch::Tensor& key,
                                                         const torch::Tensor& value,
                                                         const MTGRCaseConfig& cfg,
                                                         RuntimeAttentionMetadata& runtime_meta,
                                                         torch::Tensor& mask_storage,
                                                         aclrtStream stream) {
  OneStagePerf perf;

  auto out = torch::zeros_like(query);
  SegmentAttentionMetadata attn_meta;
  attn_meta.sparse_param.sparse_mode = 0;
  attn_meta.sparse_param.pre_tokens = kDefaultWindow;
  attn_meta.sparse_param.next_tokens = kDefaultWindow;

  const auto mask_t0 = std::chrono::steady_clock::now();
  attn_meta.attn_mask = build_irregular_genrec_mask_on_prealloc(
      cfg.history, cfg.context, cfg.real_time, cfg.target, mask_storage);
  CHECK_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
  perf.mask_build_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - mask_t0)
          .count();

  const auto getws_t0 = std::chrono::steady_clock::now();
  auto plan = plan_segment_attention(query, key, value, attn_meta, out, std::nullopt);
  perf.workspace_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - getws_t0)
          .count();

  aclrtEvent ev_start = nullptr;
  aclrtEvent ev_end = nullptr;
  CHECK_EQ(aclrtCreateEvent(&ev_start), ACL_SUCCESS);
  CHECK_EQ(aclrtCreateEvent(&ev_end), ACL_SUCCESS);
  const auto fia_wall_t0 = std::chrono::steady_clock::now();
  CHECK_EQ(aclrtRecordEvent(ev_start, stream), ACL_SUCCESS);
  launch_attention_without_memset(plan, runtime_meta, stream);
  CHECK_EQ(aclrtRecordEvent(ev_end, stream), ACL_SUCCESS);
  CHECK_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
  const auto fia_wall_t1 = std::chrono::steady_clock::now();

  float fia_ms = 0.0f;
  CHECK_EQ(aclrtEventElapsedTime(&fia_ms, ev_start, ev_end), ACL_SUCCESS);
  perf.fia_ms = static_cast<double>(fia_ms);
  perf.h2d_ms = 0.0;
  perf.device_total_ms = perf.mask_build_ms + perf.h2d_ms + perf.fia_ms;
  perf.wall_total_ms =
      perf.mask_build_ms + perf.h2d_ms +
      std::chrono::duration<double, std::milli>(fia_wall_t1 - fia_wall_t0).count();

  CHECK_EQ(aclrtDestroyEvent(ev_start), ACL_SUCCESS);
  CHECK_EQ(aclrtDestroyEvent(ev_end), ACL_SUCCESS);
  destroy_planned_attention(plan);
  CHECK(torch::isfinite(out).all().item<bool>());
  return perf;
}

void run_no_matched(const torch::Tensor& query,
                    const torch::Tensor& key,
                    const torch::Tensor& value,
                    const SampleSegmentMetadata& sample_metadata,
                    const RuntimeAttentionMetadata& runtime_meta,
                    torch::Tensor& output,
                    MTGRForwardPerf* perf_sink) {
  CHECK_EQ(query.size(0), 1);
  const int64_t h = sample_metadata.history;
  const int64_t c = sample_metadata.context;
  const int64_t r = sample_metadata.real_time;
  const int64_t t = sample_metadata.target;
  const int64_t num_heads = query.size(2);
  const int64_t head_dim = query.size(3);
  auto stream = c10_npu::getCurrentNPUStream(query.device().index()).stream();

  std::array<RawStagePerf, 5> stages{{
      {"rt_tgt_on_hcr_trapezoid"},
      {"hist_fa"},
      {"ctx_on_hc"},
      {"tgt_diag_bmm"},
      {"update_tgt_2way"},
  }};
  if (perf_sink != nullptr) {
    create_stage_events(&stages);
  }
  auto wall_t0 = std::chrono::steady_clock::now();

  auto out_opts = torch::TensorOptions().dtype(query.scalar_type()).device(query.device());
  auto lse_opts = torch::TensorOptions().dtype(torch::kFloat32).device(query.device());
  auto make_out = [&](int64_t seq_len) { return torch::empty({1, seq_len, num_heads, head_dim}, out_opts); };
  auto make_lse = [&](int64_t seq_len) { return torch::empty({1, num_heads, seq_len, 1}, lse_opts); };

  auto rt_tgt_out = make_out(r + t);
  auto rt_tgt_lse = make_lse(r + t);
  torch::Tensor hist_out;
  torch::Tensor ctx_out;
  AclPlan hist_plan;
  AclPlan ctx_plan;
  bool has_hist_plan = false;
  bool has_ctx_plan = false;
  SegmentAttentionMetadata rt_tgt_meta;
  rt_tgt_meta.attn_mask = sample_metadata.compressed_causal_mask;
  rt_tgt_meta.sparse_param.sparse_mode = 4;
  rt_tgt_meta.sparse_param.pre_tokens = h + c + r;
  rt_tgt_meta.sparse_param.next_tokens = t;
  auto rt_tgt_query = query.slice(1, h + c, h + c + r + t).contiguous();
  auto hcr_key = key.slice(1, 0, h + c + r).contiguous();
  auto hcr_value = value.slice(1, 0, h + c + r).contiguous();
  const auto rt_tgt_ws_t0 = std::chrono::steady_clock::now();
  auto rt_tgt_plan =
      plan_segment_attention(rt_tgt_query, hcr_key, hcr_value, rt_tgt_meta, rt_tgt_out, rt_tgt_lse);
  stages[0].workspace_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - rt_tgt_ws_t0)
          .count();
  record_stage_begin(perf_sink == nullptr ? nullptr : &stages[0], stream);
  launch_attention_without_memset(rt_tgt_plan, runtime_meta, stream);
  record_stage_end(perf_sink == nullptr ? nullptr : &stages[0], stream);
  print_finite_summary("no_match_fused.rt_tgt_out", rt_tgt_out);
  print_finite_summary("no_match_fused.rt_tgt_lse", rt_tgt_lse);

  const int64_t update_b = 1;
  const int64_t update_s = t;
  const int64_t update_n = num_heads;
  const int64_t update_d = head_dim;
  const int64_t update_bsn = update_b * update_s * update_n;
  auto f32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(query.device());
  auto rt_tgt_out_flat_placeholder = torch::empty({update_bsn, update_d}, f32_opts);
  auto rt_tgt_lse_flat_placeholder = torch::empty({update_bsn}, f32_opts);
  auto target_out_flat_placeholder = torch::empty({update_bsn, update_d}, f32_opts);
  auto target_lse_flat_placeholder = torch::empty({update_bsn}, f32_opts);

  const auto update_ws_t0 = std::chrono::steady_clock::now();
  auto target_update_plan = plan_attention_update_from_prepared_flats(
      {rt_tgt_out_flat_placeholder, target_out_flat_placeholder},
      {rt_tgt_lse_flat_placeholder, target_lse_flat_placeholder},
      update_b,
      update_s,
      update_n,
      update_d,
      query.scalar_type());
  stages[4].workspace_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - update_ws_t0)
          .count();

  if (h > 0) {
    hist_out = make_out(h);
    SegmentAttentionMetadata hist_meta;
    hist_meta.attn_mask = sample_metadata.compressed_causal_mask;
    hist_meta.sparse_param.sparse_mode = 2;
    hist_meta.sparse_param.pre_tokens = kDefaultWindow;
    hist_meta.sparse_param.next_tokens = kDefaultWindow;
    auto hist_query = query.slice(1, 0, h).contiguous();
    auto hist_key = key.slice(1, 0, h).contiguous();
    auto hist_value = value.slice(1, 0, h).contiguous();
    const auto hist_ws_t0 = std::chrono::steady_clock::now();
    hist_plan =
        plan_segment_attention(hist_query, hist_key, hist_value, hist_meta, hist_out, std::nullopt);
    has_hist_plan = true;
    stages[1].workspace_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - hist_ws_t0)
            .count();
    record_stage_begin(perf_sink == nullptr ? nullptr : &stages[1], stream);
    launch_attention_without_memset(hist_plan, runtime_meta, stream);
    record_stage_end(perf_sink == nullptr ? nullptr : &stages[1], stream);
  }

  if (c > 0) {
    ctx_out = make_out(c);
    SegmentAttentionMetadata ctx_meta;
    ctx_meta.sparse_param.sparse_mode = 0;
    ctx_meta.sparse_param.pre_tokens = kDefaultWindow;
    ctx_meta.sparse_param.next_tokens = kDefaultWindow;
    auto ctx_query = query.slice(1, h, h + c).contiguous();
    auto hc_key = key.slice(1, 0, h + c).contiguous();
    auto hc_value = value.slice(1, 0, h + c).contiguous();
    const auto ctx_ws_t0 = std::chrono::steady_clock::now();
    ctx_plan =
        plan_segment_attention(ctx_query, hc_key, hc_value, ctx_meta, ctx_out, std::nullopt);
    has_ctx_plan = true;
    stages[2].workspace_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ctx_ws_t0)
            .count();
    record_stage_begin(perf_sink == nullptr ? nullptr : &stages[2], stream);
    launch_attention_without_memset(ctx_plan, runtime_meta, stream);
    record_stage_end(perf_sink == nullptr ? nullptr : &stages[2], stream);
  }

  rt_tgt_out_flat_placeholder.view({update_b, update_s, update_n, update_d})
      .copy_(rt_tgt_out.narrow(1, r, t));
  rt_tgt_lse_flat_placeholder.view({update_b, update_s, update_n})
      .copy_(rt_tgt_lse.narrow(2, r, t).permute({0, 2, 1, 3}).squeeze(-1));

  auto target_query = query.slice(1, h + c + r, h + c + r + t).contiguous();
  auto target_key = key.slice(1, h + c + r, h + c + r + t).contiguous();
  auto target_value = value.slice(1, h + c + r, h + c + r + t).contiguous();
  stages[3].workspace_ms = 0.0;
  record_stage_begin(perf_sink == nullptr ? nullptr : &stages[3], stream);
  auto [target_out, target_lse] =
      build_target_diagonal_attn_analytic(target_query, target_key, target_value);
  target_out_flat_placeholder.view({update_b, update_s, update_n, update_d}).copy_(target_out);
  target_lse_flat_placeholder.view({update_b, update_s, update_n})
      .copy_(target_lse.permute({0, 2, 1, 3}).squeeze(-1));
  record_stage_end(perf_sink == nullptr ? nullptr : &stages[3], stream);

  record_stage_begin(perf_sink == nullptr ? nullptr : &stages[4], stream);
  launch_attention_update_without_memset(target_update_plan, runtime_meta, stream);
  record_stage_end(perf_sink == nullptr ? nullptr : &stages[4], stream);

  auto target_merged =
      target_update_plan.out_flat
          .view({target_update_plan.b, target_update_plan.s, target_update_plan.n, target_update_plan.d})
          .to(target_update_plan.out_scalar_type);

  if (h > 0) {
    output.slice(1, 0, h).copy_(hist_out);
  }
  if (c > 0) {
    output.slice(1, h, h + c).copy_(ctx_out);
  }
  output.slice(1, h + c, h + c + r).copy_(rt_tgt_out.slice(1, 0, r));
  output.slice(1, h + c + r, h + c + r + t).copy_(target_merged);
  destroy_planned_attention(rt_tgt_plan);
  if (has_hist_plan) {
    destroy_planned_attention(hist_plan);
  }
  if (has_ctx_plan) {
    destroy_planned_attention(ctx_plan);
  }
  destroy_planned_attention_update(target_update_plan);

  if (perf_sink != nullptr) {
    CHECK_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    const auto wall_t1 = std::chrono::steady_clock::now();
    for (auto& stage : stages) {
      float ms = 0.0f;
      CHECK_EQ(aclrtEventElapsedTime(&ms, stage.ev_start, stage.ev_end), ACL_SUCCESS);
      stage.exec_ms = static_cast<double>(ms);
    }
    float total_dev_ms_f = 0.0f;
    CHECK_EQ(aclrtEventElapsedTime(&total_dev_ms_f, stages.front().ev_start, stages.back().ev_end),
             ACL_SUCCESS);
    finalize_forward_perf("no_matched",
                          &stages,
                          static_cast<double>(total_dev_ms_f),
                          std::chrono::duration<double, std::milli>(wall_t1 - wall_t0).count(),
                          perf_sink);
    destroy_stage_events(&stages);
  }
}

void run_no_matched_fused_target_update(const torch::Tensor& query,
                                        const torch::Tensor& key,
                                        const torch::Tensor& value,
                                        const SampleSegmentMetadata& sample_metadata,
                                        const RuntimeAttentionMetadata& runtime_meta,
                                        torch::Tensor& output,
                                        MTGRForwardPerf* perf_sink,
                                        int variant) {
  CHECK_EQ(query.size(0), 1);
  const int64_t h = sample_metadata.history;
  const int64_t c = sample_metadata.context;
  const int64_t r = sample_metadata.real_time;
  const int64_t t = sample_metadata.target;
  const int64_t num_heads = query.size(2);
  const int64_t head_dim = query.size(3);
  auto stream = c10_npu::getCurrentNPUStream(query.device().index()).stream();

  std::array<RawStagePerf, 4> stages{{
      {"rt_tgt_on_hcr_trapezoid"},
      {"hist_fa"},
      {"ctx_on_hc"},
      {variant == 4 ? "target_fused_update_v4" : "target_fused_update"},
  }};
  if (perf_sink != nullptr) {
    create_stage_events(&stages);
  }
  auto wall_t0 = std::chrono::steady_clock::now();

  auto out_opts = torch::TensorOptions().dtype(query.scalar_type()).device(query.device());
  auto lse_opts = torch::TensorOptions().dtype(torch::kFloat32).device(query.device());
  auto make_out = [&](int64_t seq_len) {
    return torch::empty({1, seq_len, num_heads, head_dim}, out_opts);
  };
  auto make_lse = [&](int64_t seq_len) {
    return torch::empty({1, num_heads, seq_len, 1}, lse_opts);
  };

  auto rt_tgt_out = make_out(r + t);
  auto rt_tgt_lse = make_lse(r + t);
  torch::Tensor hist_out;
  torch::Tensor ctx_out;
  AclPlan hist_plan;
  AclPlan ctx_plan;
  bool has_hist_plan = false;
  bool has_ctx_plan = false;

  SegmentAttentionMetadata rt_tgt_meta;
  rt_tgt_meta.attn_mask = sample_metadata.compressed_causal_mask;
  rt_tgt_meta.sparse_param.sparse_mode = 4;
  rt_tgt_meta.sparse_param.pre_tokens = h + c + r;
  rt_tgt_meta.sparse_param.next_tokens = t;
  auto rt_tgt_query = query.slice(1, h + c, h + c + r + t).contiguous();
  auto hcr_key = key.slice(1, 0, h + c + r).contiguous();
  auto hcr_value = value.slice(1, 0, h + c + r).contiguous();
  const auto rt_tgt_ws_t0 = std::chrono::steady_clock::now();
  auto rt_tgt_plan =
      plan_segment_attention(rt_tgt_query, hcr_key, hcr_value, rt_tgt_meta, rt_tgt_out, rt_tgt_lse);
  stages[0].workspace_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - rt_tgt_ws_t0)
          .count();
  record_stage_begin(perf_sink == nullptr ? nullptr : &stages[0], stream);
  launch_attention_without_memset(rt_tgt_plan, runtime_meta, stream);
  record_stage_end(perf_sink == nullptr ? nullptr : &stages[0], stream);
  print_finite_summary("no_match_fused.rt_tgt_out", rt_tgt_out);
  print_finite_summary("no_match_fused.rt_tgt_lse", rt_tgt_lse);

  if (h > 0) {
    hist_out = make_out(h);
    SegmentAttentionMetadata hist_meta;
    hist_meta.attn_mask = sample_metadata.compressed_causal_mask;
    hist_meta.sparse_param.sparse_mode = 2;
    hist_meta.sparse_param.pre_tokens = kDefaultWindow;
    hist_meta.sparse_param.next_tokens = kDefaultWindow;
    auto hist_query = query.slice(1, 0, h).contiguous();
    auto hist_key = key.slice(1, 0, h).contiguous();
    auto hist_value = value.slice(1, 0, h).contiguous();
    const auto hist_ws_t0 = std::chrono::steady_clock::now();
    hist_plan =
        plan_segment_attention(hist_query, hist_key, hist_value, hist_meta, hist_out, std::nullopt);
    has_hist_plan = true;
    stages[1].workspace_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - hist_ws_t0)
            .count();
    record_stage_begin(perf_sink == nullptr ? nullptr : &stages[1], stream);
    launch_attention_without_memset(hist_plan, runtime_meta, stream);
    record_stage_end(perf_sink == nullptr ? nullptr : &stages[1], stream);
  }

  if (c > 0) {
    ctx_out = make_out(c);
    SegmentAttentionMetadata ctx_meta;
    ctx_meta.sparse_param.sparse_mode = 0;
    ctx_meta.sparse_param.pre_tokens = kDefaultWindow;
    ctx_meta.sparse_param.next_tokens = kDefaultWindow;
    auto ctx_query = query.slice(1, h, h + c).contiguous();
    auto hc_key = key.slice(1, 0, h + c).contiguous();
    auto hc_value = value.slice(1, 0, h + c).contiguous();
    const auto ctx_ws_t0 = std::chrono::steady_clock::now();
    ctx_plan =
        plan_segment_attention(ctx_query, hc_key, hc_value, ctx_meta, ctx_out, std::nullopt);
    has_ctx_plan = true;
    stages[2].workspace_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ctx_ws_t0)
            .count();
    record_stage_begin(perf_sink == nullptr ? nullptr : &stages[2], stream);
    launch_attention_without_memset(ctx_plan, runtime_meta, stream);
    record_stage_end(perf_sink == nullptr ? nullptr : &stages[2], stream);
  }

  auto target_query = query.slice(1, h + c + r, h + c + r + t).contiguous();
  auto target_key = key.slice(1, h + c + r, h + c + r + t).contiguous();
  auto target_value = value.slice(1, h + c + r, h + c + r + t).contiguous();
  auto target_prefix_out = rt_tgt_out;
  auto target_prefix_lse = rt_tgt_lse;
  auto target_out = output.slice(1, h + c + r, h + c + r + t);
  CHECK(target_out.is_contiguous());
  const auto fused_ws_t0 = std::chrono::steady_clock::now();
  auto target_fused_plan = variant == 4 ? plan_mtgr_target_update_v4(target_query,
                                                                     target_key,
                                                                     target_value,
                                                                     target_prefix_out,
                                                                     target_prefix_lse,
                                                                     target_out)
                                          : plan_mtgr_target_update(target_query,
                                                                    target_key,
                                                                    target_value,
                                                                    target_prefix_out,
                                                                    target_prefix_lse,
                                                                    target_out);
  stages[3].workspace_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - fused_ws_t0)
          .count();
  record_stage_begin(perf_sink == nullptr ? nullptr : &stages[3], stream);
  if (variant == 4) {
    launch_mtgr_target_update_v4_without_memset(target_fused_plan, runtime_meta, stream);
  } else {
    launch_mtgr_target_update_without_memset(target_fused_plan, runtime_meta, stream);
  }
  record_stage_end(perf_sink == nullptr ? nullptr : &stages[3], stream);
  print_finite_summary("no_match_fused.target_out", target_out);

  if (h > 0) {
    output.slice(1, 0, h).copy_(hist_out);
  }
  if (c > 0) {
    output.slice(1, h, h + c).copy_(ctx_out);
  }
  output.slice(1, h + c, h + c + r).copy_(rt_tgt_out.slice(1, 0, r));

  if (perf_sink != nullptr) {
    CHECK_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    const auto wall_t1 = std::chrono::steady_clock::now();
    for (auto& stage : stages) {
      float ms = 0.0f;
      CHECK_EQ(aclrtEventElapsedTime(&ms, stage.ev_start, stage.ev_end), ACL_SUCCESS);
      stage.exec_ms = static_cast<double>(ms);
    }
    float total_dev_ms_f = 0.0f;
    CHECK_EQ(aclrtEventElapsedTime(&total_dev_ms_f, stages.front().ev_start, stages.back().ev_end),
             ACL_SUCCESS);
    finalize_forward_perf(variant == 4 ? "no_matched_fused_target_update_v4"
                                       : "no_matched_fused_target_update",
                          &stages,
                          static_cast<double>(total_dev_ms_f),
                          std::chrono::duration<double, std::milli>(wall_t1 - wall_t0).count(),
                          perf_sink);
    destroy_stage_events(&stages);
  } else {
    CHECK_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
  }

  destroy_planned_attention(rt_tgt_plan);
  if (has_hist_plan) {
    destroy_planned_attention(hist_plan);
  }
  if (has_ctx_plan) {
    destroy_planned_attention(ctx_plan);
  }
  destroy_planned_mtgr_target_update(target_fused_plan);
}

void run_partial_rt_matched(const torch::Tensor& query,
                            const torch::Tensor& key,
                            const torch::Tensor& value,
                            torch::Tensor& key_cache,
                            torch::Tensor& value_cache,
                            const SampleSegmentMetadata& sample_metadata,
                            const RuntimeAttentionMetadata& runtime_meta,
                            torch::Tensor& output,
                            MTGRForwardPerf* perf_sink) {
  CHECK_EQ(query.size(0), 1);
  const int64_t h = sample_metadata.history;
  const int64_t c = sample_metadata.context;
  const int64_t r = sample_metadata.real_time;
  const int64_t t = sample_metadata.target;
  const int64_t realtime_matched = sample_metadata.matched_prefix - (h + c);
  const int64_t rt_unmatched = r - realtime_matched;
  CHECK_GT(rt_unmatched, 0);

  auto stream = c10_npu::getCurrentNPUStream(query.device().index()).stream();
  auto i32_dev_opts = torch::TensorOptions().dtype(torch::kInt32).device(query.device());
  const int64_t cache_block_count = key_cache.size(0);
  const int64_t num_kv_heads = key.size(2);
  const int64_t num_heads = query.size(2);
  const int64_t head_dim = key.size(3);

  std::array<RawStagePerf, 5> stages{{
      {"scatter_pa_kv_cache"},
      {"tgt_diag_bmm"},
      {"rt_unmatched_pa_s3"},
      {"target_prefix_pa_full"},
      {"update_tgt_2way"},
  }};
  if (perf_sink != nullptr) {
    create_stage_events(&stages);
  }
  auto wall_t0 = std::chrono::steady_clock::now();

  auto key_seq = key.select(0, 0).contiguous();
  auto value_seq = value.select(0, 0).contiguous();
  auto new_slots_dev_i32 =
      sample_metadata.slot_mapping.slice(0, 0, rt_unmatched).to(i32_dev_opts.dtype()).contiguous();
  auto new_key = key_seq.slice(0, 0, rt_unmatched).contiguous();
  auto new_value = value_seq.slice(0, 0, rt_unmatched).contiguous();
  ScatterPaKvCachePlan scatter_plan;
  const auto scatter_ws_t0 = std::chrono::steady_clock::now();
  CHECK(plan_scatter_pa_kv_cache(
      new_key, new_value, key_cache, value_cache, new_slots_dev_i32, &runtime_meta, &scatter_plan));
  stages[0].workspace_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - scatter_ws_t0)
          .count();
  record_stage_begin(perf_sink == nullptr ? nullptr : &stages[0], stream);
  CHECK(launch_planned_scatter_pa_kv_cache(scatter_plan, stream));
  record_stage_end(perf_sink == nullptr ? nullptr : &stages[0], stream);

  auto key_cache_bnbsh = key_cache.view({cache_block_count, sample_metadata.block_size, num_kv_heads * head_dim});
  auto value_cache_bnbsh =
      value_cache.view({cache_block_count, sample_metadata.block_size, num_kv_heads * head_dim});
  auto block_table = sample_metadata.block_table.to(i32_dev_opts.dtype()).contiguous();

  auto out_opts = torch::TensorOptions().dtype(query.scalar_type()).device(query.device());
  auto lse_opts = torch::TensorOptions().dtype(torch::kFloat32).device(query.device());
  constexpr float kSentinel = -31415.0f;
  auto make_out = [&](int64_t seq_len) {
    return torch::full({1, seq_len, num_heads, head_dim}, kSentinel, out_opts);
  };
  auto make_lse = [&](int64_t seq_len) {
    return torch::full({1, num_heads, seq_len, 1}, kSentinel, lse_opts);
  };

  const int64_t local_target_start = rt_unmatched;
  auto target_query = query.slice(1, local_target_start, local_target_start + t).contiguous();
  auto target_key = key.slice(1, local_target_start, local_target_start + t).contiguous();
  auto target_value = value.slice(1, local_target_start, local_target_start + t).contiguous();
  stages[1].workspace_ms = 0.0;
  record_stage_begin(perf_sink == nullptr ? nullptr : &stages[1], stream);
  auto [target_diag_out, target_diag_lse] =
      build_target_diagonal_attn_analytic(target_query, target_key, target_value);
  record_stage_end(perf_sink == nullptr ? nullptr : &stages[1], stream);

  auto rt_query = query.slice(1, 0, rt_unmatched).contiguous();
  auto rt_out = make_out(rt_unmatched);
  const auto rt_ws_t0 = std::chrono::steady_clock::now();
  auto rt_pa_plan = plan_paged_attention_v3(rt_query,
                                            key_cache_bnbsh,
                                            value_cache_bnbsh,
                                            block_table,
                                            sample_metadata.compressed_causal_mask,
                                            3,
                                            kDefaultWindow,
                                            kDefaultWindow,
                                            h + c + r,
                                            sample_metadata.block_size,
                                            num_kv_heads,
                                            rt_out,
                                            std::nullopt);
  stages[2].workspace_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - rt_ws_t0)
          .count();
  record_stage_begin(perf_sink == nullptr ? nullptr : &stages[2], stream);
  launch_paged_attention_without_memset(rt_pa_plan, runtime_meta, stream);
  record_stage_end(perf_sink == nullptr ? nullptr : &stages[2], stream);

  auto target_prefix_out = make_out(t);
  auto target_prefix_lse = make_lse(t);
  const auto target_prefix_ws_t0 = std::chrono::steady_clock::now();
  auto target_prefix_pa_plan = plan_paged_attention_v3(target_query,
                                                       key_cache_bnbsh,
                                                       value_cache_bnbsh,
                                                       block_table,
                                                       torch::Tensor(),
                                                       0,
                                                       kDefaultWindow,
                                                       kDefaultWindow,
                                                       h + c + r,
                                                       sample_metadata.block_size,
                                                       num_kv_heads,
                                                       target_prefix_out,
                                                       target_prefix_lse);
  stages[3].workspace_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - target_prefix_ws_t0)
          .count();
  record_stage_begin(perf_sink == nullptr ? nullptr : &stages[3], stream);
  launch_paged_attention_without_memset(target_prefix_pa_plan, runtime_meta, stream);
  record_stage_end(perf_sink == nullptr ? nullptr : &stages[3], stream);

  const int64_t b = 1;
  const int64_t s = t;
  const int64_t n = num_heads;
  const int64_t d = head_dim;
  const int64_t bsn = b * s * n;
  const auto update_ws_t0 = std::chrono::steady_clock::now();
  auto target_update_plan = plan_attention_update_from_prepared_flats(
      {target_prefix_out.view({bsn, d}).to(torch::kFloat32),
       target_diag_out.view({bsn, d}).to(torch::kFloat32)},
      {target_prefix_lse.permute({0, 2, 1, 3}).contiguous().view({bsn}),
       target_diag_lse.permute({0, 2, 1, 3}).contiguous().view({bsn})},
      b,
      s,
      n,
      d,
      query.scalar_type());
  stages[4].workspace_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - update_ws_t0)
          .count();
  record_stage_begin(perf_sink == nullptr ? nullptr : &stages[4], stream);
  launch_attention_update_without_memset(target_update_plan, runtime_meta, stream);
  record_stage_end(perf_sink == nullptr ? nullptr : &stages[4], stream);

  auto target_merged =
      target_update_plan.out_flat
          .view({target_update_plan.b, target_update_plan.s, target_update_plan.n, target_update_plan.d})
          .to(target_update_plan.out_scalar_type);

  output.slice(1, 0, rt_unmatched).copy_(rt_out);
  output.slice(1, local_target_start, local_target_start + t).copy_(target_merged);

  if (perf_sink != nullptr) {
    CHECK_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    const auto wall_t1 = std::chrono::steady_clock::now();
    for (auto& stage : stages) {
      float ms = 0.0f;
      CHECK_EQ(aclrtEventElapsedTime(&ms, stage.ev_start, stage.ev_end), ACL_SUCCESS);
      stage.exec_ms = static_cast<double>(ms);
    }
    float total_dev_ms_f = 0.0f;
    CHECK_EQ(aclrtEventElapsedTime(&total_dev_ms_f, stages.front().ev_start, stages.back().ev_end),
             ACL_SUCCESS);
    finalize_forward_perf("partial_rt_matched",
                          &stages,
                          static_cast<double>(total_dev_ms_f),
                          std::chrono::duration<double, std::milli>(wall_t1 - wall_t0).count(),
                          perf_sink);
    destroy_stage_events(&stages);
  } else {
    CHECK_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
  }
  destroy_planned_paged_attention(rt_pa_plan);
  destroy_planned_paged_attention(target_prefix_pa_plan);
  destroy_planned_attention_update(target_update_plan);
  destroy_planned_scatter_pa_kv_cache(scatter_plan);
}

void run_partial_rt_matched_fused_target_update(const torch::Tensor& query,
                                                const torch::Tensor& key,
                                                const torch::Tensor& value,
                                                torch::Tensor& key_cache,
                                                torch::Tensor& value_cache,
                                                const SampleSegmentMetadata& sample_metadata,
                                                const RuntimeAttentionMetadata& runtime_meta,
                                                torch::Tensor& output,
                                                MTGRForwardPerf* perf_sink,
                                                int variant) {
  CHECK_EQ(query.size(0), 1);
  const int64_t h = sample_metadata.history;
  const int64_t c = sample_metadata.context;
  const int64_t r = sample_metadata.real_time;
  const int64_t t = sample_metadata.target;
  const int64_t realtime_matched = sample_metadata.matched_prefix - (h + c);
  const int64_t rt_unmatched = r - realtime_matched;
  CHECK_GT(rt_unmatched, 0);

  auto stream = c10_npu::getCurrentNPUStream(query.device().index()).stream();
  auto i32_dev_opts = torch::TensorOptions().dtype(torch::kInt32).device(query.device());
  const int64_t cache_block_count = key_cache.size(0);
  const int64_t num_kv_heads = key.size(2);
  const int64_t num_heads = query.size(2);
  const int64_t head_dim = key.size(3);

  std::array<RawStagePerf, 4> stages{{
      {"scatter_pa_kv_cache"},
      {"rt_unmatched_pa_s3"},
      {"target_prefix_pa_full"},
      {variant == 4 ? "target_fused_update_v4" : "target_fused_update"},
  }};
  if (perf_sink != nullptr) {
    create_stage_events(&stages);
  }
  auto wall_t0 = std::chrono::steady_clock::now();

  auto key_seq = key.select(0, 0).contiguous();
  auto value_seq = value.select(0, 0).contiguous();
  auto new_slots_dev_i32 =
      sample_metadata.slot_mapping.slice(0, 0, rt_unmatched).to(i32_dev_opts.dtype()).contiguous();
  auto new_key = key_seq.slice(0, 0, rt_unmatched).contiguous();
  auto new_value = value_seq.slice(0, 0, rt_unmatched).contiguous();
  ScatterPaKvCachePlan scatter_plan;
  const auto scatter_ws_t0 = std::chrono::steady_clock::now();
  CHECK(plan_scatter_pa_kv_cache(
      new_key, new_value, key_cache, value_cache, new_slots_dev_i32, &runtime_meta, &scatter_plan));
  stages[0].workspace_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - scatter_ws_t0)
          .count();
  record_stage_begin(perf_sink == nullptr ? nullptr : &stages[0], stream);
  CHECK(launch_planned_scatter_pa_kv_cache(scatter_plan, stream));
  record_stage_end(perf_sink == nullptr ? nullptr : &stages[0], stream);

  auto key_cache_bnbsh = key_cache.view({cache_block_count, sample_metadata.block_size, num_kv_heads * head_dim});
  auto value_cache_bnbsh =
      value_cache.view({cache_block_count, sample_metadata.block_size, num_kv_heads * head_dim});
  auto block_table = sample_metadata.block_table.to(i32_dev_opts.dtype()).contiguous();

  auto out_opts = torch::TensorOptions().dtype(query.scalar_type()).device(query.device());
  auto lse_opts = torch::TensorOptions().dtype(torch::kFloat32).device(query.device());
  constexpr float kSentinel = -31415.0f;
  auto make_out = [&](int64_t seq_len) {
    return torch::full({1, seq_len, num_heads, head_dim}, kSentinel, out_opts);
  };
  auto make_lse = [&](int64_t seq_len) {
    return torch::full({1, num_heads, seq_len, 1}, kSentinel, lse_opts);
  };

  auto rt_query = query.slice(1, 0, rt_unmatched).contiguous();
  auto rt_out = make_out(rt_unmatched);
  const auto rt_ws_t0 = std::chrono::steady_clock::now();
  auto rt_pa_plan = plan_paged_attention_v3(rt_query,
                                            key_cache_bnbsh,
                                            value_cache_bnbsh,
                                            block_table,
                                            sample_metadata.compressed_causal_mask,
                                            3,
                                            kDefaultWindow,
                                            kDefaultWindow,
                                            h + c + r,
                                            sample_metadata.block_size,
                                            num_kv_heads,
                                            rt_out,
                                            std::nullopt);
  stages[1].workspace_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - rt_ws_t0)
          .count();
  record_stage_begin(perf_sink == nullptr ? nullptr : &stages[1], stream);
  launch_paged_attention_without_memset(rt_pa_plan, runtime_meta, stream);
  record_stage_end(perf_sink == nullptr ? nullptr : &stages[1], stream);

  const int64_t local_target_start = rt_unmatched;
  auto target_query = query.slice(1, local_target_start, local_target_start + t).contiguous();
  auto target_key = key.slice(1, local_target_start, local_target_start + t).contiguous();
  auto target_value = value.slice(1, local_target_start, local_target_start + t).contiguous();
  auto target_prefix_out = make_out(t);
  auto target_prefix_lse = make_lse(t);
  const auto target_prefix_ws_t0 = std::chrono::steady_clock::now();
  auto target_prefix_pa_plan = plan_paged_attention_v3(target_query,
                                                       key_cache_bnbsh,
                                                       value_cache_bnbsh,
                                                       block_table,
                                                       torch::Tensor(),
                                                       0,
                                                       kDefaultWindow,
                                                       kDefaultWindow,
                                                       h + c + r,
                                                       sample_metadata.block_size,
                                                       num_kv_heads,
                                                       target_prefix_out,
                                                       target_prefix_lse);
  stages[2].workspace_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - target_prefix_ws_t0)
          .count();
  record_stage_begin(perf_sink == nullptr ? nullptr : &stages[2], stream);
  launch_paged_attention_without_memset(target_prefix_pa_plan, runtime_meta, stream);
  record_stage_end(perf_sink == nullptr ? nullptr : &stages[2], stream);

  auto target_out = output.slice(1, local_target_start, local_target_start + t);
  CHECK(target_out.is_contiguous());
  const auto fused_ws_t0 = std::chrono::steady_clock::now();
  auto target_fused_plan = variant == 4 ? plan_mtgr_target_update_v4(target_query,
                                                                     target_key,
                                                                     target_value,
                                                                     target_prefix_out,
                                                                     target_prefix_lse,
                                                                     target_out)
                                          : plan_mtgr_target_update(target_query,
                                                                    target_key,
                                                                    target_value,
                                                                    target_prefix_out,
                                                                    target_prefix_lse,
                                                                    target_out);
  stages[3].workspace_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - fused_ws_t0)
          .count();
  record_stage_begin(perf_sink == nullptr ? nullptr : &stages[3], stream);
  if (variant == 4) {
    launch_mtgr_target_update_v4_without_memset(target_fused_plan, runtime_meta, stream);
  } else {
    launch_mtgr_target_update_without_memset(target_fused_plan, runtime_meta, stream);
  }
  record_stage_end(perf_sink == nullptr ? nullptr : &stages[3], stream);

  output.slice(1, 0, rt_unmatched).copy_(rt_out);

  if (perf_sink != nullptr) {
    CHECK_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    const auto wall_t1 = std::chrono::steady_clock::now();
    for (auto& stage : stages) {
      float ms = 0.0f;
      CHECK_EQ(aclrtEventElapsedTime(&ms, stage.ev_start, stage.ev_end), ACL_SUCCESS);
      stage.exec_ms = static_cast<double>(ms);
    }
    float total_dev_ms_f = 0.0f;
    CHECK_EQ(aclrtEventElapsedTime(&total_dev_ms_f, stages.front().ev_start, stages.back().ev_end),
             ACL_SUCCESS);
    finalize_forward_perf(variant == 4 ? "partial_rt_matched_fused_target_update_v4"
                                       : "partial_rt_matched_fused_target_update",
                          &stages,
                          static_cast<double>(total_dev_ms_f),
                          std::chrono::duration<double, std::milli>(wall_t1 - wall_t0).count(),
                          perf_sink);
    destroy_stage_events(&stages);
  } else {
    CHECK_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
  }
  destroy_planned_paged_attention(rt_pa_plan);
  destroy_planned_paged_attention(target_prefix_pa_plan);
  destroy_planned_mtgr_target_update(target_fused_plan);
  destroy_planned_scatter_pa_kv_cache(scatter_plan);
}

}  // namespace

std::vector<int32_t> make_block_table(int64_t block_count) {
  std::vector<int32_t> table(static_cast<size_t>(block_count));
  for (int64_t i = 0; i < block_count; ++i) {
    table[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  }
  return table;
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
    const int64_t physical_block = block_table.at(static_cast<size_t>(logical_block));
    slots.push_back(static_cast<int32_t>(physical_block * block_size + offset));
  }
  return slots;
}

int64_t partial_rt_matched_tokens(int64_t realtime_tokens) {
  const int64_t matched =
      realtime_tokens * kPartialRtMatchedNumerator / kPartialRtMatchedDenominator;
  return std::clamp<int64_t>(matched, 1, realtime_tokens - 1);
}

MTGRPreparedCase prepare_mtgr_case(const MTGRCaseConfig& cfg,
                                   const torch::TensorOptions& fp_opts,
                                   const torch::Device& device) {
  CHECK_EQ(cfg.batch_size, 1);
  CHECK_GT(cfg.history, 0);
  CHECK_GT(cfg.context, 0);
  CHECK_GT(cfg.real_time, 0);
  CHECK_GT(cfg.target, 0);
  CHECK_GE(cfg.matched_prefix, 0);
  CHECK_LT(cfg.matched_prefix, cfg.history + cfg.context + cfg.real_time);
  FLAGS_block_size = cfg.block_size;

  const int64_t total = cfg.total_len();
  const int64_t unmatched_start = cfg.matched_prefix;
  const int64_t unmatched_len = total - unmatched_start;

  auto query_full_flat =
      torch::randn({total, cfg.num_heads * cfg.head_dim}, fp_opts);
  auto key_full_flat =
      torch::randn({total, cfg.num_kv_heads * cfg.head_dim}, fp_opts);
  auto value_full_flat =
      torch::randn({total, cfg.num_kv_heads * cfg.head_dim}, fp_opts);

  MTGRPreparedCase prepared;
  prepared.cfg = cfg;
  prepared.query_case_flat = query_full_flat.narrow(0, unmatched_start, unmatched_len).contiguous();
  prepared.key_case_flat = key_full_flat.narrow(0, unmatched_start, unmatched_len).contiguous();
  prepared.value_case_flat = value_full_flat.narrow(0, unmatched_start, unmatched_len).contiguous();
  prepared.query_full_bsnd =
      query_full_flat.view({1, total, cfg.num_heads, cfg.head_dim}).contiguous();
  prepared.key_full_bsnd =
      key_full_flat.view({1, total, cfg.num_kv_heads, cfg.head_dim}).contiguous();
  prepared.value_full_bsnd =
      value_full_flat.view({1, total, cfg.num_kv_heads, cfg.head_dim}).contiguous();

  const int64_t block_count = (total + cfg.block_size - 1) / cfg.block_size;
  const auto block_table_host = make_block_table(block_count);
  auto block_table =
      torch::tensor(block_table_host,
                    torch::TensorOptions().dtype(torch::kInt32).device(device))
          .view({1, block_count})
          .contiguous();
  auto slot_mapping_host =
      build_slot_mapping(block_table_host, cfg.block_size, unmatched_start, unmatched_len);
  auto slot_mapping =
      torch::tensor(slot_mapping_host,
                    torch::TensorOptions().dtype(torch::kInt64).device(device))
          .contiguous();

  const int64_t cache_block_count = block_count + 4;
  prepared.key_cache =
      torch::zeros({cache_block_count, cfg.block_size, cfg.num_kv_heads, cfg.head_dim}, fp_opts);
  prepared.value_cache = torch::zeros_like(prepared.key_cache);
  prefill_matched_prefix_cache(prepared.key_full_bsnd,
                               prepared.value_full_bsnd,
                               cfg.matched_prefix,
                               block_table_host,
                               cfg.block_size,
                               prepared.key_cache,
                               prepared.value_cache);

  auto len_opts = torch::TensorOptions().dtype(torch::kInt64);
  prepared.attn_metadata.is_dummy = false;
  prepared.attn_metadata.q_seq_lens = torch::tensor({unmatched_len}, len_opts);
  prepared.attn_metadata.kv_seq_lens = torch::tensor({unmatched_len}, len_opts);
  prepared.attn_metadata.genrec_history_lens = torch::tensor({cfg.history}, len_opts);
  prepared.attn_metadata.genrec_context_lens = torch::tensor({cfg.context}, len_opts);
  prepared.attn_metadata.genrec_real_time_lens = torch::tensor({cfg.real_time}, len_opts);
  prepared.attn_metadata.genrec_target_lens = torch::tensor({cfg.target}, len_opts);
  prepared.attn_metadata.genrec_matched_prefix_lens =
      torch::tensor({cfg.matched_prefix}, len_opts);
  prepared.attn_metadata.block_table = block_table;
  prepared.attn_metadata.slot_mapping = slot_mapping;
  return prepared;
}

OneStagePerf profile_one_stage_maskopt(const MTGRPreparedCase& prepared_case,
                                       int warmup_iters,
                                       int repeat_iters) {
  CHECK_GT(warmup_iters, 0);
  CHECK_GT(repeat_iters, 0);
  RuntimeAttentionMetadata runtime_meta;
  runtime_meta.compressed_causal_mask =
      create_compressed_causal_mask_2048(prepared_case.query_full_bsnd.device());
  CHECK_EQ(aclrtMalloc(&runtime_meta.shared_workspace,
                       kPreallocWorkspaceBytes,
                       ACL_MEM_MALLOC_HUGE_FIRST),
           ACL_SUCCESS);
  runtime_meta.shared_workspace_size = kPreallocWorkspaceBytes;
  auto stream =
      c10_npu::getCurrentNPUStream(prepared_case.query_full_bsnd.device().index()).stream();

  const int64_t total = prepared_case.cfg.total_len();
  auto mask_storage =
      torch::empty({total, total},
                   torch::TensorOptions()
                       .dtype(torch::kBool)
                       .device(prepared_case.query_full_bsnd.device()));
  CHECK_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);

  for (int i = 0; i < warmup_iters; ++i) {
    (void)run_one_stage_fia_with_device_mask_prealloc(prepared_case.query_full_bsnd,
                                                      prepared_case.key_full_bsnd,
                                                      prepared_case.value_full_bsnd,
                                                      prepared_case.cfg,
                                                      runtime_meta,
                                                      mask_storage,
                                                      stream);
  }
  CHECK_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);

  OneStagePerf agg;
  for (int rep = 0; rep < repeat_iters; ++rep) {
    auto cur = run_one_stage_fia_with_device_mask_prealloc(prepared_case.query_full_bsnd,
                                                           prepared_case.key_full_bsnd,
                                                           prepared_case.value_full_bsnd,
                                                           prepared_case.cfg,
                                                           runtime_meta,
                                                           mask_storage,
                                                           stream);
    agg.mask_build_ms += cur.mask_build_ms;
    agg.h2d_ms += cur.h2d_ms;
    agg.workspace_ms += cur.workspace_ms;
    agg.fia_ms += cur.fia_ms;
    agg.device_total_ms += cur.device_total_ms;
    agg.wall_total_ms += cur.wall_total_ms;
  }

  const double repeat = static_cast<double>(repeat_iters);
  agg.mask_build_ms /= repeat;
  agg.h2d_ms /= repeat;
  agg.workspace_ms /= repeat;
  agg.fia_ms /= repeat;
  agg.device_total_ms /= repeat;
  agg.wall_total_ms /= repeat;

  CHECK_EQ(aclrtFree(runtime_meta.shared_workspace), ACL_SUCCESS);
  return agg;
}

MTGRAttentionTestImpl::MTGRAttentionTestImpl(int64_t num_heads,
                                             int64_t head_size,
                                             float scale,
                                             int64_t num_kv_heads)
    : num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads) {}

MTGRAttentionTestImpl::~MTGRAttentionTestImpl() {
  if (shared_workspace_ != nullptr) {
    CHECK_EQ(aclrtFree(shared_workspace_), ACL_SUCCESS);
    shared_workspace_ = nullptr;
  }
  if (scatter_workspace_ != nullptr) {
    CHECK_EQ(aclrtFree(scatter_workspace_), ACL_SUCCESS);
    scatter_workspace_ = nullptr;
  }
}

void MTGRAttentionTestImpl::set_forward_perf_sink(MTGRForwardPerf* perf_sink) {
  perf_sink_ = perf_sink;
}

void MTGRAttentionTestImpl::set_use_fused_target_update(bool use_fused_target_update) {
  use_fused_target_update_ = use_fused_target_update;
}

void MTGRAttentionTestImpl::set_use_fused_target_update_auto(bool use_fused_target_update_auto) {
  use_fused_target_update_auto_ = use_fused_target_update_auto;
}

void MTGRAttentionTestImpl::set_use_fused_target_update_v4(bool use_fused_target_update_v4) {
  use_fused_target_update_v4_ = use_fused_target_update_v4;
}

void MTGRAttentionTestImpl::set_use_current_best_target_update(
    bool use_current_best_target_update) {
  use_current_best_target_update_ = use_current_best_target_update;
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> MTGRAttentionTestImpl::forward(
    const xllm::layer::AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    xllm::KVCache& kv_cache) {
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
    return {output, output_lse};
  }

  const int32_t device_index = query.device().index();
  if (!compressed_causal_mask_.defined() || workspace_device_index_ != device_index) {
    compressed_causal_mask_ = create_compressed_causal_mask_2048(query.device());
  }
  if (shared_workspace_ == nullptr || workspace_device_index_ != device_index) {
    if (shared_workspace_ != nullptr) {
      CHECK_EQ(aclrtFree(shared_workspace_), ACL_SUCCESS);
      shared_workspace_ = nullptr;
    }
    if (scatter_workspace_ != nullptr) {
      CHECK_EQ(aclrtFree(scatter_workspace_), ACL_SUCCESS);
      scatter_workspace_ = nullptr;
      scatter_workspace_size_ = 0;
    }
    CHECK_EQ(aclrtMalloc(&shared_workspace_, kPreallocWorkspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             ACL_SUCCESS);
    shared_workspace_size_ = kPreallocWorkspaceBytes;
    workspace_device_index_ = device_index;
  }
  RuntimeAttentionMetadata runtime_meta;
  runtime_meta.compressed_causal_mask = compressed_causal_mask_;
  runtime_meta.shared_workspace = shared_workspace_;
  runtime_meta.shared_workspace_size = shared_workspace_size_;
  runtime_meta.scatter_workspace = scatter_workspace_;
  runtime_meta.scatter_workspace_size = scatter_workspace_size_;
  auto ensure_scatter_workspace = [&]() {
    if (scatter_workspace_ == nullptr) {
      CHECK_EQ(aclrtMalloc(&scatter_workspace_,
                           kPreallocScatterWorkspaceBytes,
                           ACL_MEM_MALLOC_HUGE_FIRST),
               ACL_SUCCESS);
      scatter_workspace_size_ = kPreallocScatterWorkspaceBytes;
    }
    runtime_meta.scatter_workspace = scatter_workspace_;
    runtime_meta.scatter_workspace_size = scatter_workspace_size_;
  };

  if (perf_sink_ != nullptr) {
    CHECK_EQ(attn_metadata.q_seq_lens.size(0), 1)
        << "profiled MTGR forward only supports batch_size=1";
    *perf_sink_ = MTGRForwardPerf();
  }

  const int64_t q_tokens = query.size(0);
  const int64_t kv_tokens = key.size(0);
  const auto& q_seq_lens_host = attn_metadata.q_seq_lens;
  const auto& kv_seq_lens_host = attn_metadata.kv_seq_lens;
  const auto& history_lens_host = attn_metadata.genrec_history_lens;
  const auto& context_lens_host = attn_metadata.genrec_context_lens;
  const auto& real_time_lens_host = attn_metadata.genrec_real_time_lens;
  const auto& target_lens_host = attn_metadata.genrec_target_lens;
  const auto& matched_prefix_lens_host = attn_metadata.genrec_matched_prefix_lens;

  CHECK(q_seq_lens_host.defined());
  CHECK(kv_seq_lens_host.defined());
  CHECK(history_lens_host.defined());
  CHECK(context_lens_host.defined());
  CHECK(real_time_lens_host.defined());
  CHECK(target_lens_host.defined());
  CHECK(matched_prefix_lens_host.defined());

  const int64_t batch_size = q_seq_lens_host.size(0);
  CHECK_EQ(kv_seq_lens_host.size(0), batch_size);
  CHECK_EQ(history_lens_host.size(0), batch_size);
  CHECK_EQ(context_lens_host.size(0), batch_size);
  CHECK_EQ(real_time_lens_host.size(0), batch_size);
  CHECK_EQ(target_lens_host.size(0), batch_size);
  CHECK_EQ(matched_prefix_lens_host.size(0), batch_size);
  CHECK(attn_metadata.block_table.defined());
  CHECK(attn_metadata.slot_mapping.defined());

  auto key_cache = kv_cache.get_k_cache();
  auto value_cache = kv_cache.get_v_cache();
  const int64_t block_size = static_cast<int64_t>(FLAGS_block_size);
  CHECK_GT(block_size, 0);

  int64_t q_offset = 0;
  int64_t kv_offset = 0;
  for (int64_t i = 0; i < batch_size; ++i) {
    const int64_t q_len = q_seq_lens_host[i].item<int64_t>();
    const int64_t kv_len = kv_seq_lens_host[i].item<int64_t>();
    if (q_len == 0) {
      kv_offset += kv_len;
      continue;
    }

    auto query_i =
        query.narrow(0, q_offset, q_len).view({1, q_len, num_heads_, head_size_});
    auto key_i =
        key.narrow(0, kv_offset, kv_len).view({1, kv_len, num_kv_heads_, head_size_});
    auto value_i =
        value.narrow(0, kv_offset, kv_len).view({1, kv_len, num_kv_heads_, head_size_});
    auto output_i = torch::empty_like(query_i);

    SampleSegmentMetadata sample_metadata;
    sample_metadata.history = history_lens_host[i].item<int64_t>();
    sample_metadata.context = context_lens_host[i].item<int64_t>();
    sample_metadata.real_time = real_time_lens_host[i].item<int64_t>();
    sample_metadata.target = target_lens_host[i].item<int64_t>();
    sample_metadata.matched_prefix = matched_prefix_lens_host[i].item<int64_t>();
    sample_metadata.block_size = block_size;
    sample_metadata.compressed_causal_mask = compressed_causal_mask_;
    sample_metadata.block_table = attn_metadata.block_table.select(0, i).unsqueeze(0).contiguous();
    sample_metadata.slot_mapping = attn_metadata.slot_mapping.narrow(0, kv_offset, kv_len).contiguous();

    const int64_t h = sample_metadata.history;
    const int64_t c = sample_metadata.context;
    const int64_t r = sample_metadata.real_time;
    const int64_t matched = sample_metadata.matched_prefix;
    CHECK_GE(matched, 0);
    CHECK_LT(matched, h + c + r);
    const bool use_selected_fused_target_update =
        use_current_best_target_update_ || use_fused_target_update_ ||
        use_fused_target_update_auto_ || use_fused_target_update_v4_;
    const auto resolve_fused_variant = [&](bool partial_rt_matched) -> int {
      (void)partial_rt_matched;
      if (use_fused_target_update_auto_) {
        return 4;
      }
      if (use_fused_target_update_v4_) {
        return 4;
      }
      if (use_current_best_target_update_) {
        return 4;
      }
      return 1;
    };

    auto write_prefix_cache = [&](const SampleSegmentMetadata& metadata,
                                  int64_t prefix_cache_len,
                                  const char* tag) {
      if (!key_cache.defined() || !value_cache.defined() || prefix_cache_len <= 0) {
        return;
      }
      auto stream = c10_npu::getCurrentNPUStream(query.device().index()).stream();
      auto i32_dev_opts = torch::TensorOptions().dtype(torch::kInt32).device(query.device());
      auto key_seq = key_i.select(0, 0).contiguous();
      auto value_seq = value_i.select(0, 0).contiguous();
      auto slots_dev_i32 =
          metadata.slot_mapping.slice(0, 0, prefix_cache_len).to(i32_dev_opts.dtype()).contiguous();
      auto prefix_key = key_seq.slice(0, 0, prefix_cache_len).contiguous();
      auto prefix_value = value_seq.slice(0, 0, prefix_cache_len).contiguous();
      ensure_scatter_workspace();
      CHECK(run_scatter_pa_kv_cache(
          prefix_key, prefix_value, key_cache, value_cache, slots_dev_i32, &runtime_meta, stream))
          << "run_scatter_pa_kv_cache failed in " << tag;
    };

    if (matched == 0) {
      if (use_selected_fused_target_update) {
        const int fused_variant = resolve_fused_variant(false);
        run_no_matched_fused_target_update(
            query_i,
            key_i,
            value_i,
            sample_metadata,
            runtime_meta,
            output_i,
            perf_sink_,
            fused_variant);
      } else {
        run_no_matched(query_i, key_i, value_i, sample_metadata, runtime_meta, output_i, perf_sink_);
      }
    } else if (matched < h + c) {
      SampleSegmentMetadata fallback_metadata = sample_metadata;
      fallback_metadata.matched_prefix = 0;
      CHECK_EQ(q_len, h + c + r + fallback_metadata.target);
      if (use_selected_fused_target_update) {
        const int fused_variant = resolve_fused_variant(false);
        run_no_matched_fused_target_update(
            query_i,
            key_i,
            value_i,
            fallback_metadata,
            runtime_meta,
            output_i,
            perf_sink_,
            fused_variant);
      } else {
        run_no_matched(query_i, key_i, value_i, fallback_metadata, runtime_meta, output_i, perf_sink_);
      }
    } else if (matched < h + c + r) {
      CHECK(key_cache.defined() && value_cache.defined());
      ensure_scatter_workspace();
      if (use_selected_fused_target_update) {
        const int fused_variant = resolve_fused_variant(true);
        run_partial_rt_matched_fused_target_update(query_i,
                                                   key_i,
                                                   value_i,
                                                   key_cache,
                                                   value_cache,
                                                   sample_metadata,
                                                   runtime_meta,
                                                   output_i,
                                                   perf_sink_,
                                                   fused_variant);
      } else {
        run_partial_rt_matched(query_i,
                               key_i,
                               value_i,
                               key_cache,
                               value_cache,
                               sample_metadata,
                               runtime_meta,
                               output_i,
                               perf_sink_);
      }
    } else {
      if (use_fused_target_update_) {
        const int fused_variant = resolve_fused_variant(false);
        run_no_matched_fused_target_update(
            query_i,
            key_i,
            value_i,
            sample_metadata,
            runtime_meta,
            output_i,
            perf_sink_,
            fused_variant);
      } else {
        run_no_matched(query_i, key_i, value_i, sample_metadata, runtime_meta, output_i, perf_sink_);
      }
    }

    output.narrow(0, q_offset, q_len).copy_(output_i.view({q_len, num_heads_ * head_size_}));
    q_offset += q_len;
    kv_offset += kv_len;
  }

  CHECK_EQ(q_offset, q_tokens);
  CHECK_EQ(kv_offset, kv_tokens);
  return {output, output_lse};
}

MTGRForwardPerf profile_mtgr_forward(MTGRPreparedCase& prepared_case,
                                     int warmup_iters,
                                     int repeat_iters,
                                     bool use_fused_target_update,
                                     bool use_fused_target_update_auto,
                                     bool use_fused_target_update_v4) {
  CHECK_GT(warmup_iters, 0);
  CHECK_GT(repeat_iters, 0);

  MTGRAttentionTestImpl mtgr(prepared_case.cfg.num_heads,
                             prepared_case.cfg.head_dim,
                             1.0f / std::sqrt(static_cast<float>(prepared_case.cfg.head_dim)),
                             prepared_case.cfg.num_kv_heads);
  mtgr.set_use_fused_target_update(use_fused_target_update);
  mtgr.set_use_fused_target_update_auto(use_fused_target_update_auto);
  mtgr.set_use_fused_target_update_v4(use_fused_target_update_v4);
  mtgr.set_use_current_best_target_update(false);
  xllm::KVCache kv_cache(prepared_case.key_cache, prepared_case.value_cache);

  auto check_output_finite = [&](const torch::Tensor& output, const char* phase) {
    const bool finite = torch::isfinite(output).all().item<bool>();
    if (!finite) {
      const auto& cfg = prepared_case.cfg;
      const int64_t total_seq = cfg.history + cfg.context + cfg.real_time + cfg.target;
      auto output_bsnd = output.view({1, total_seq, cfg.num_heads, cfg.head_dim});
      auto print_segment = [&](const char* name, int64_t begin, int64_t end) {
        if (end <= begin) {
          return;
        }
        const auto segment = output_bsnd.slice(1, begin, end);
        const auto finite_mask = torch::isfinite(segment);
        const int64_t finite_count = finite_mask.sum().item<int64_t>();
        std::fprintf(stderr,
                     "[MTGRDebug][nonfinite][%s] %s finite=%ld/%ld range=[%ld,%ld)\n",
                     phase,
                     name,
                     static_cast<long>(finite_count),
                     static_cast<long>(segment.numel()),
                     static_cast<long>(begin),
                     static_cast<long>(end));
        if (finite_count != segment.numel()) {
          auto bad_idx = torch::nonzero(~finite_mask).to(torch::kCPU);
          if (bad_idx.numel() > 0) {
            const int64_t b = bad_idx[0][0].item<int64_t>();
            const int64_t s = bad_idx[0][1].item<int64_t>();
            const int64_t n = bad_idx[0][2].item<int64_t>();
            const int64_t d = bad_idx[0][3].item<int64_t>();
            std::fprintf(stderr,
                         "[MTGRDebug][nonfinite][%s] %s first_bad=[b=%ld,s=%ld,n=%ld,d=%ld]\n",
                         phase,
                         name,
                         static_cast<long>(b),
                         static_cast<long>(s),
                         static_cast<long>(n),
                         static_cast<long>(d));
          }
        }
      };
      print_segment("history", 0, cfg.history);
      print_segment("context", cfg.history, cfg.history + cfg.context);
      print_segment("real_time", cfg.history + cfg.context, cfg.history + cfg.context + cfg.real_time);
      print_segment("target",
                    cfg.history + cfg.context + cfg.real_time,
                    cfg.history + cfg.context + cfg.real_time + cfg.target);
    }
    CHECK(finite);
  };

  for (int i = 0; i < warmup_iters; ++i) {
    mtgr.set_forward_perf_sink(nullptr);
    auto output =
        std::get<0>(mtgr.forward(prepared_case.attn_metadata,
                                 prepared_case.query_case_flat,
                                 prepared_case.key_case_flat,
                                 prepared_case.value_case_flat,
                                 kv_cache));
    check_output_finite(output, "warmup");
  }

  MTGRForwardPerf agg;
  for (int rep = 0; rep < repeat_iters; ++rep) {
    MTGRForwardPerf cur;
    mtgr.set_forward_perf_sink(&cur);
    auto output =
        std::get<0>(mtgr.forward(prepared_case.attn_metadata,
                                 prepared_case.query_case_flat,
                                 prepared_case.key_case_flat,
                                 prepared_case.value_case_flat,
                                 kv_cache));
    check_output_finite(output, "repeat");
    if (rep == 0) {
      agg.path_name = cur.path_name;
      agg.stages = cur.stages;
      for (auto& stage : agg.stages) {
        stage.workspace_ms = 0.0;
        stage.exec_ms = 0.0;
      }
    } else {
      CHECK_EQ(cur.path_name, agg.path_name);
      CHECK_EQ(cur.stages.size(), agg.stages.size());
    }
    agg.device_total_ms += cur.device_total_ms;
    agg.wall_total_ms += cur.wall_total_ms;
    for (size_t i = 0; i < cur.stages.size(); ++i) {
      agg.stages[i].workspace_ms += cur.stages[i].workspace_ms;
      agg.stages[i].exec_ms += cur.stages[i].exec_ms;
    }
  }

  const double repeat = static_cast<double>(repeat_iters);
  agg.device_total_ms /= repeat;
  agg.wall_total_ms /= repeat;
  for (auto& stage : agg.stages) {
    stage.workspace_ms /= repeat;
    stage.exec_ms /= repeat;
  }
  return agg;
}

}  // namespace xllm::kernel::npu::test
