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
#include <optional>
#include <string>
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

namespace xllm::kernel::npu::test {

namespace {

struct AttentionMetadata {
  torch::Tensor history_lens;
  torch::Tensor context_lens;
  torch::Tensor real_time_lens;
  torch::Tensor target_lens;
  torch::Tensor compressed_causal_mask;
  torch::Tensor diagonal_mask;
  // 预先申请的共享 workspace（单流复用，aclrtMalloc 风格）。
  void* shared_workspace = nullptr;
  uint64_t shared_workspace_size = 0;
};

struct SegmentAttentionMetadata {
  torch::Tensor attn_mask;
  int64_t sparse_mode = 0;
  int64_t pre_tokens = 0;
  int64_t next_tokens = 0;
};

class GenRecDirectRunBSNDCleanTest : public ::testing::Test {
 protected:
  static constexpr int32_t kDeviceId = 5;

  static void SetUpTestSuite() {
    torch_npu::init_npu("npu:" + std::to_string(kDeviceId));
  }

  static void TearDownTestSuite() { torch_npu::finalize_npu(); }

  void SetUp() override {
    device_ = torch::Device(torch::kPrivateUse1, kDeviceId);
    opts_ = torch::TensorOptions().dtype(torch::kFloat16).device(device_);
    stream_ = c10_npu::getCurrentNPUStream(kDeviceId).stream();
    ASSERT_NE(stream_, nullptr);
  }

  torch::Device device_{torch::kCPU};
  torch::TensorOptions opts_;
  aclrtStream stream_{nullptr};
};

bool valid_tensor(const torch::Tensor& t) { return t.defined() && t.numel() > 0; }

constexpr double kFinalRtol = 0.12;
constexpr double kFinalAtol = 0.12;
constexpr float kFinalMaxAbs = 1.5e-1f;

void expect_final_output_matches_reference_genrec_gold(
    const char* tag,
    const torch::Tensor& output_bsnd,
    const torch::Tensor& query_bsnd,
    const torch::Tensor& key_bsnd,
    const torch::Tensor& value_bsnd,
    int64_t h,
    int64_t c,
    int64_t r,
    int64_t t) {
  struct LocalSeqPartLengths {
    int64_t history;
    int64_t context;
    int64_t real_time;
    int64_t target;
    int64_t total() const { return history + context + real_time + target; }
  };
  struct LocalGenRecReferenceOutputs {
    torch::Tensor history;
    torch::Tensor context;
    torch::Tensor real_time;
    torch::Tensor target;
  };

  auto seq_to_bnsd = [](const torch::Tensor& seq_s_n_d) {
    return seq_s_n_d.unsqueeze(0).permute({0, 2, 1, 3}).contiguous();
  };
  auto bsnd_to_bnsd = [](const torch::Tensor& bsnd) {
    CHECK_EQ(bsnd.dim(), 4);
    return bsnd.permute({0, 2, 1, 3}).contiguous();
  };
  auto reference_attention_bnsd =
      [](const torch::Tensor& q,
         const torch::Tensor& k,
         const torch::Tensor& v,
         const torch::Tensor* mask_sq_sk,
         double scale) {
        CHECK_EQ(q.dim(), 4);
        int64_t num_heads = q.size(1);
        int64_t num_kv = k.size(1);
        auto qf = q;
        auto kf = k;
        auto vf = v;
        if (num_kv < num_heads) {
          int64_t group = num_heads / num_kv;
          kf = kf.repeat_interleave(group, 1);
          vf = vf.repeat_interleave(group, 1);
        }
        auto scores = torch::matmul(qf, kf.transpose(-2, -1)) * scale;
        if (mask_sq_sk != nullptr) {
          scores = scores.masked_fill(*mask_sq_sk, -1e9f);
        }
        auto weights = torch::softmax(scores, -1);
        return torch::matmul(weights, vf);
      };
  auto reference_genrec_gold =
      [&](const torch::Tensor& q_seq,
          const torch::Tensor& k_seq,
          const torch::Tensor& v_seq,
          const LocalSeqPartLengths& p,
          double scale) -> LocalGenRecReferenceOutputs {
        const int64_t hh = p.history;
        const int64_t cc = p.context;
        const int64_t rr = p.real_time;
        const int64_t tt = p.target;

        auto q = seq_to_bnsd(q_seq);
        auto k = seq_to_bnsd(k_seq);
        auto v = seq_to_bnsd(v_seq);
        const int64_t nheads = q.size(1);
        const int64_t d = q.size(3);

        auto qh = q.slice(2, 0, hh);
        auto kh = k.slice(2, 0, hh);
        auto vh = v.slice(2, 0, hh);
        auto mask_h = torch::triu(torch::ones({hh, hh}, torch::dtype(torch::kBool)), 1);
        auto history = reference_attention_bnsd(qh, kh, vh, &mask_h, scale);

        auto q_ctx = q.slice(2, hh, hh + cc);
        auto k_hc = k.slice(2, 0, hh + cc);
        auto v_hc = v.slice(2, 0, hh + cc);
        auto context = reference_attention_bnsd(q_ctx, k_hc, v_hc, nullptr, scale);

        auto real_time = torch::empty({1, nheads, rr, d}, q.options());
        for (int64_t i = 0; i < rr; ++i) {
          int64_t nkv = hh + cc + i + 1;
          auto qi = q.slice(2, hh + cc + i, hh + cc + i + 1);
          auto kk = k.slice(2, 0, nkv);
          auto vv = v.slice(2, 0, nkv);
          auto oi = reference_attention_bnsd(qi, kk, vv, nullptr, scale);
          real_time.narrow(2, i, 1).copy_(oi);
        }

        const int64_t prefix = hh + cc + rr;
        auto target = torch::empty({1, nheads, tt, d}, q.options());
        for (int64_t j = 0; j < tt; ++j) {
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

        return LocalGenRecReferenceOutputs{history, context, real_time, target};
      };

  const double scale = static_cast<double>(
      1.0f / std::sqrt(static_cast<float>(query_bsnd.size(3))));
  LocalSeqPartLengths parts{h, c, r, t};

  auto q_seq = query_bsnd.select(0, 0).to(torch::kCPU).to(torch::kFloat32);
  auto k_seq = key_bsnd.select(0, 0).to(torch::kCPU).to(torch::kFloat32);
  auto v_seq = value_bsnd.select(0, 0).to(torch::kCPU).to(torch::kFloat32);
  CHECK_EQ(q_seq.size(0), parts.total());

  auto ref = reference_genrec_gold(q_seq, k_seq, v_seq, parts, scale);
  auto got_bnsd = bsnd_to_bnsd(output_bsnd).to(torch::kCPU).to(torch::kFloat32);

  auto ref_bnsd = torch::empty_like(got_bnsd);
  ref_bnsd.slice(2, 0, h).copy_(ref.history);
  ref_bnsd.slice(2, h, h + c).copy_(ref.context);
  ref_bnsd.slice(2, h + c, h + c + r).copy_(ref.real_time);
  ref_bnsd.slice(2, h + c + r, h + c + r + t).copy_(ref.target);

  auto diff = (got_bnsd - ref_bnsd).abs();
  double max_abs = diff.max().item<double>();
  double mean_abs = diff.mean().item<double>();
  double rmse = diff.pow(2).mean().sqrt().item<double>();
  bool allclose = torch::allclose(got_bnsd, ref_bnsd, kFinalRtol, kFinalAtol);

  std::fprintf(stderr,
               "[GenRecV2][Precision][%s] max_abs=%.6e mean_abs=%.6e rmse=%.6e "
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

torch::Tensor create_diagonal_mask(int64_t seq_len, torch::Device device) {
  auto mask = ~torch::eye(seq_len, torch::TensorOptions().dtype(torch::kBool).device(device));
  return mask.view({1, 1, seq_len, seq_len}).contiguous();
}

torch::Tensor create_compressed_causal_mask_2048(torch::Device device) {
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
  aclDataType acl_tensor_type = to_acl_dtype(t.scalar_type());
  aclTensor* acl_t = aclCreateTensor(t.sizes().data(),
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
  aclTensor* acl_t = aclCreateTensor(t.sizes().data(),
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
      plan.q_acl, plan.k_list, plan.v_list,
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
      attn_metadata.pre_tokens,
      attn_metadata.next_tokens,
      layout,
      num_kv_heads,
      attn_metadata.sparse_mode,
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
        << "aclrtMemset FusedInferAttention workspace";
  }
  auto ret = aclnnFusedInferAttentionScoreV3(
      ws_ptr, plan.workspace_size, plan.executor, stream);
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
    plan.local_out_flats.push_back(
        local_outs[i].view({bsn, d}).to(torch::kFloat32));
    plan.lse_flats.push_back(
        lses[i].permute({0, 2, 1, 3}).contiguous().view({bsn}));
    plan.local_out_acls[i] = torch_to_acl_tensor(plan.local_out_flats.back());
    plan.lse_acls[i] = torch_to_acl_tensor(plan.lse_flats.back());
  }

  plan.out_flat = torch::empty(
      {bsn, d},
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
  CHECK_EQ(ret, ACL_SUCCESS) << "aclnnAttentionUpdateGetWorkspaceSize failed: " << ret;
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
        << "aclrtMemset AttentionUpdate workspace";
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

void run_genrec_v2_single_batch(const torch::Tensor& query,
                                const torch::Tensor& key,
                                const torch::Tensor& value,
                                const AttentionMetadata& attn_metadata,
                                torch::Tensor& output) {
  int32_t device_id = query.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();

  int64_t batch_size = query.size(0);
  int64_t num_heads = query.size(2);

  constexpr int64_t kSparseModeFullMask = 0;
  constexpr int64_t kSparseModeLeftUpCausal = 2;
  constexpr int64_t kDefaultWindow = 2147483647LL;

  auto out_opts = torch::TensorOptions().dtype(query.scalar_type()).device(query.device());
  auto lse_opts = torch::TensorOptions().dtype(torch::kFloat32).device(query.device());
  constexpr float kSentinel = -31415.0f;

  auto make_out = [&](int64_t seq_len) {
    return torch::full({1, seq_len, num_heads, query.size(3)}, kSentinel, out_opts);
  };
  auto make_lse = [&](int64_t seq_len) {
    return torch::full({1, num_heads, seq_len, 1}, kSentinel, lse_opts);
  };

  for (int64_t i = 0; i < batch_size; ++i) {
    const int64_t h = attn_metadata.history_lens.select(0, i).item<int64_t>();
    const int64_t c = attn_metadata.context_lens.select(0, i).item<int64_t>();
    const int64_t r = attn_metadata.real_time_lens.select(0, i).item<int64_t>();
    const int64_t t = attn_metadata.target_lens.select(0, i).item<int64_t>();

    auto hist_query = query.slice(0, i, i + 1).slice(1, 0, h);
    auto hist_key = key.slice(0, i, i + 1).slice(1, 0, h);
    auto hist_value = value.slice(0, i, i + 1).slice(1, 0, h);
    auto hist_out = make_out(h);

    SegmentAttentionMetadata hist_meta;
    hist_meta.attn_mask = attn_metadata.compressed_causal_mask;
    hist_meta.sparse_mode = kSparseModeLeftUpCausal;
    hist_meta.pre_tokens = kDefaultWindow;
    hist_meta.next_tokens = kDefaultWindow;

    AclPlan hist_plan = plan_segment_attention(
        hist_query, hist_key, hist_value, hist_meta, hist_out, std::nullopt);
    execute_planned_attention(
        hist_plan, attn_metadata.shared_workspace, attn_metadata.shared_workspace_size, stream);
    destroy_planned_attention(hist_plan);
    output.slice(0, i, i + 1).slice(1, 0, h).copy_(hist_out);

    auto crt_out = make_out(c + r + t);
    auto crt_lse = make_lse(c + r + t);
    auto crt_query = query.slice(0, i, i + 1).slice(1, h, h + c + r + t);
    auto crt_key = key.slice(0, i, i + 1).slice(1, 0, h + c);
    auto crt_value = value.slice(0, i, i + 1).slice(1, 0, h + c);

    SegmentAttentionMetadata crt_meta;
    crt_meta.sparse_mode = kSparseModeFullMask;
    crt_meta.pre_tokens = kDefaultWindow;
    crt_meta.next_tokens = kDefaultWindow;

    AclPlan crt_plan = plan_segment_attention(
        crt_query, crt_key, crt_value, crt_meta, crt_out, crt_lse);
    execute_planned_attention(
        crt_plan, attn_metadata.shared_workspace, attn_metadata.shared_workspace_size, stream);
    destroy_planned_attention(crt_plan);
    output.slice(0, i, i + 1).slice(1, h, h + c).copy_(crt_out.slice(1, 0, c));

    auto rt_out = make_out(r + t);
    auto rt_lse = make_lse(r + t);
    auto rt_query = query.slice(0, i, i + 1).slice(1, h + c, h + c + r + t);
    auto rt_key = key.slice(0, i, i + 1).slice(1, h + c, h + c + r);
    auto rt_value = value.slice(0, i, i + 1).slice(1, h + c, h + c + r);

    SegmentAttentionMetadata rt_meta;
    rt_meta.attn_mask = attn_metadata.compressed_causal_mask;
    rt_meta.sparse_mode = kSparseModeLeftUpCausal;
    rt_meta.pre_tokens = kDefaultWindow;
    rt_meta.next_tokens = kDefaultWindow;

    AclPlan rt_plan = plan_segment_attention(
        rt_query, rt_key, rt_value, rt_meta, rt_out, rt_lse);
    execute_planned_attention(
        rt_plan, attn_metadata.shared_workspace, attn_metadata.shared_workspace_size, stream);
    destroy_planned_attention(rt_plan);

    auto target_out = make_out(t);
    auto target_lse = make_lse(t);
    auto target_query =
        query.slice(0, i, i + 1).slice(1, h + c + r, h + c + r + t);
    auto target_key =
        key.slice(0, i, i + 1).slice(1, h + c + r, h + c + r + t);
    auto target_value =
        value.slice(0, i, i + 1).slice(1, h + c + r, h + c + r + t);

    SegmentAttentionMetadata target_meta;
    target_meta.attn_mask = attn_metadata.diagonal_mask;
    target_meta.sparse_mode = kSparseModeFullMask;
    target_meta.pre_tokens = 0;
    target_meta.next_tokens = 0;

    AclPlan target_plan = plan_segment_attention(
        target_query, target_key, target_value, target_meta, target_out, target_lse);
    execute_planned_attention(
        target_plan, attn_metadata.shared_workspace, attn_metadata.shared_workspace_size, stream);
    destroy_planned_attention(target_plan);

    auto rt_update_plan = plan_attention_update_like_benchmark(
        {crt_out.slice(1, c, c + r), rt_out.slice(1, 0, r)},
        {crt_lse.slice(2, c, c + r), rt_lse.slice(2, 0, r)});
    execute_planned_attention_update(
        rt_update_plan,
        attn_metadata.shared_workspace,
        attn_metadata.shared_workspace_size,
        stream);
    auto rt_merged = rt_update_plan.out_flat.view(
                         {rt_update_plan.b,
                          rt_update_plan.s,
                          rt_update_plan.n,
                          rt_update_plan.d})
                         .to(rt_update_plan.out_scalar_type);
    destroy_planned_attention_update(rt_update_plan);
    output.slice(0, i, i + 1).slice(1, h + c, h + c + r).copy_(rt_merged);

    auto target_update_plan = plan_attention_update_like_benchmark(
        {crt_out.slice(1, c + r, c + r + t), rt_out.slice(1, r, r + t), target_out},
        {crt_lse.slice(2, c + r, c + r + t), rt_lse.slice(2, r, r + t), target_lse});
    execute_planned_attention_update(
        target_update_plan,
        attn_metadata.shared_workspace,
        attn_metadata.shared_workspace_size,
        stream);
    auto target_merged = target_update_plan.out_flat.view(
                             {target_update_plan.b,
                              target_update_plan.s,
                              target_update_plan.n,
                              target_update_plan.d})
                             .to(target_update_plan.out_scalar_type);
    destroy_planned_attention_update(target_update_plan);
    output.slice(0, i, i + 1).slice(1, h + c + r, h + c + r + t).copy_(target_merged);
  }
}

}  // namespace

TEST_F(GenRecDirectRunBSNDCleanTest, DirectRunGenRecV2SingleBatchBSNDClean) {
  constexpr int64_t kBatchSize = 1;
  constexpr int64_t kNumHeads = 8;
  constexpr int64_t kNumKvHeads = 8;
  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kHistory = 1300;
  constexpr int64_t kContext = 8;
  constexpr int64_t kRealtime = 400;
  constexpr int64_t kTarget = 800;
  const int64_t total = kHistory + kContext + kRealtime + kTarget;

  auto query = torch::randn({kBatchSize, total, kNumHeads, kHeadDim}, opts_);
  auto key = torch::randn({kBatchSize, total, kNumKvHeads, kHeadDim}, opts_);
  auto value = torch::randn({kBatchSize, total, kNumKvHeads, kHeadDim}, opts_);
  auto output = torch::empty_like(query);

  AttentionMetadata attn_metadata;
  auto len_opts = torch::TensorOptions().dtype(torch::kInt64);
  attn_metadata.history_lens = torch::tensor({kHistory}, len_opts);
  attn_metadata.context_lens = torch::tensor({kContext}, len_opts);
  attn_metadata.real_time_lens = torch::tensor({kRealtime}, len_opts);
  attn_metadata.target_lens = torch::tensor({kTarget}, len_opts);
  attn_metadata.compressed_causal_mask = create_compressed_causal_mask_2048(query.device());
  attn_metadata.diagonal_mask = create_diagonal_mask(kTarget, query.device());
  constexpr uint64_t kFixedSharedWorkspaceBytes = 256ULL * 1024ULL * 1024ULL;  // 256MB
  const uint64_t max_workspace_size = kFixedSharedWorkspaceBytes;
  attn_metadata.shared_workspace_size = max_workspace_size;
  if (max_workspace_size > 0) {
    CHECK_EQ(aclrtMalloc(&attn_metadata.shared_workspace,
                         max_workspace_size,
                         ACL_MEM_MALLOC_HUGE_FIRST),
             ACL_SUCCESS)
        << "aclrtMalloc shared workspace";
  }

  auto warmup_query = torch::randn({kBatchSize, total, kNumHeads, kHeadDim}, opts_);
  auto warmup_key = torch::randn({kBatchSize, total, kNumKvHeads, kHeadDim}, opts_);
  auto warmup_value = torch::randn({kBatchSize, total, kNumKvHeads, kHeadDim}, opts_);
  auto warmup_output = torch::empty_like(warmup_query);
  run_genrec_v2_single_batch(
      warmup_query, warmup_key, warmup_value, attn_metadata, warmup_output);

  run_genrec_v2_single_batch(query, key, value, attn_metadata, output);

  expect_final_output_matches_reference_genrec_gold(
      "DirectRunGenRecV2SingleBatchBSNDClean",
      output,
      query,
      key,
      value,
      kHistory,
      kContext,
      kRealtime,
      kTarget);

  ASSERT_EQ(output.dim(), 4);
  ASSERT_EQ(output.size(0), 1);
  ASSERT_EQ(output.size(1), total);
  ASSERT_EQ(output.size(2), kNumHeads);
  ASSERT_EQ(output.size(3), kHeadDim);

  auto out_cpu = output.to(torch::kCPU).to(torch::kFloat32);
  EXPECT_TRUE(torch::isfinite(out_cpu).all().item<bool>());
  EXPECT_GT(out_cpu.abs().sum().item<float>(), 0.0f);

  if (attn_metadata.shared_workspace != nullptr) {
    EXPECT_EQ(aclrtFree(attn_metadata.shared_workspace), ACL_SUCCESS)
        << "aclrtFree shared workspace";
    attn_metadata.shared_workspace = nullptr;
    attn_metadata.shared_workspace_size = 0;
  }
}

}  // namespace xllm::kernel::npu::test
