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

#include "genrec_attention.h"

#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "acl/acl.h"
#include "aclnnop/aclnn_fused_infer_attention_score_v3.h"
#include "core/kernels/npu/utils.h"

namespace xllm::kernel::npu {

namespace {

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

struct PlannedAttentionCall {
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;

  aclTensor* q_acl = nullptr;
  aclTensor* k_acl = nullptr;
  aclTensor* v_acl = nullptr;
  aclTensor* out_acl = nullptr;
  aclTensor* mask_acl = nullptr;
  aclTensor* softmax_lse_acl = nullptr;
  aclTensor* block_table_acl = nullptr;
  aclTensor* query_padding_size_acl = nullptr;
  aclTensor* kv_padding_size_acl = nullptr;
  aclTensorList* k_list = nullptr;
  aclTensorList* v_list = nullptr;
};

PlannedAttentionCall plan_segment_attention(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const GenRecAttentionKernelData& attn_data,
    int64_t num_heads,
    int64_t num_kv_heads,
    float scale,
    torch::Tensor& output,
    const std::optional<torch::Tensor>& softmax_lse,
    int64_t sparse_mode = 0,
    int64_t pre_tokens = 2147483647LL,
    int64_t next_tokens = 2147483647LL) {
  PlannedAttentionCall plan;

  create_acltensor(&plan.q_acl, query);
  create_acltensor(&plan.k_acl, key);
  create_acltensor(&plan.v_acl, value);
  create_acltensor(&plan.out_acl, output);

  if (valid_tensor(attn_data.attention_mask)) {
    create_acltensor(&plan.mask_acl, attn_data.attention_mask);
  }
  if (softmax_lse.has_value() && valid_tensor(softmax_lse.value())) {
    create_acltensor(&plan.softmax_lse_acl, softmax_lse.value());
  }
  if (valid_tensor(attn_data.block_table)) {
    create_acltensor(&plan.block_table_acl, attn_data.block_table);
  }
  if (valid_tensor(attn_data.query_padding_size)) {
    create_acltensor(&plan.query_padding_size_acl, attn_data.query_padding_size);
  }
  if (valid_tensor(attn_data.kv_padding_size)) {
    create_acltensor(&plan.kv_padding_size_acl, attn_data.kv_padding_size);
  }

  aclTensor* k_arr[] = {plan.k_acl};
  aclTensor* v_arr[] = {plan.v_acl};
  plan.k_list = aclCreateTensorList(k_arr, 1);
  plan.v_list = aclCreateTensorList(v_arr, 1);

  char layout[] = "BNSD";
  aclnnFusedInferAttentionScoreV3GetWorkspaceSize(
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
      /*blockTable=*/plan.block_table_acl,
      /*queryPaddingSize=*/plan.query_padding_size_acl,
      /*kvPaddingSize=*/plan.kv_padding_size_acl,
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
      /*blockSize=*/0,
      /*antiquantMode=*/0,
      /*softmaxLseFlag=*/(plan.softmax_lse_acl != nullptr),
      /*keyAntiquantMode=*/0,
      /*valueAntiquantMode=*/0,
      plan.out_acl,
      /*softmaxLse=*/plan.softmax_lse_acl,
      &plan.workspace_size,
      &plan.executor);
  return plan;
}

void execute_planned_attention(const PlannedAttentionCall& plan,
                               void* workspace_ptr,
                               uint64_t workspace_size,
                               aclrtStream stream) {
  aclnnFusedInferAttentionScoreV3(
      workspace_ptr, workspace_size, plan.executor, stream);
}

void destroy_planned_attention(PlannedAttentionCall& plan) {
  if (plan.k_list != nullptr) {
    aclDestroyTensorList(plan.k_list);
  }
  if (plan.v_list != nullptr) {
    aclDestroyTensorList(plan.v_list);
  }
  if (plan.q_acl != nullptr) {
    aclDestroyTensor(plan.q_acl);
  }
  if (plan.k_acl != nullptr) {
    aclDestroyTensor(plan.k_acl);
  }
  if (plan.v_acl != nullptr) {
    aclDestroyTensor(plan.v_acl);
  }
  if (plan.out_acl != nullptr) {
    aclDestroyTensor(plan.out_acl);
  }
  if (plan.mask_acl != nullptr) {
    aclDestroyTensor(plan.mask_acl);
  }
  if (plan.softmax_lse_acl != nullptr) {
    aclDestroyTensor(plan.softmax_lse_acl);
  }
  if (plan.block_table_acl != nullptr) {
    aclDestroyTensor(plan.block_table_acl);
  }
  if (plan.query_padding_size_acl != nullptr) {
    aclDestroyTensor(plan.query_padding_size_acl);
  }
  if (plan.kv_padding_size_acl != nullptr) {
    aclDestroyTensor(plan.kv_padding_size_acl);
  }
}

torch::Tensor create_full_attention_mask(int64_t q_len,
                                         int64_t kv_len,
                                         torch::Device device) {
  return torch::zeros({1, 1, q_len, kv_len},
                      torch::TensorOptions().dtype(torch::kBool).device(device));
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

torch::Tensor combine_flash_outputs(const torch::Tensor& o1,
                                    const torch::Tensor& lse1,
                                    const torch::Tensor& o2,
                                    const torch::Tensor& lse2) {
  auto max_l = torch::maximum(lse1, lse2);
  auto e1 = torch::exp(lse1 - max_l);
  auto e2 = torch::exp(lse2 - max_l);
  auto o1_f = o1.to(torch::kFloat32);
  auto o2_f = o2.to(torch::kFloat32);
  return ((o1_f * e1 + o2_f * e2) / (e1 + e2)).to(o1.scalar_type());
}

torch::Tensor combine_flash_outputs_three(const torch::Tensor& o1,
                                          const torch::Tensor& lse1,
                                          const torch::Tensor& o2,
                                          const torch::Tensor& lse2,
                                          const torch::Tensor& o3,
                                          const torch::Tensor& lse3) {
  auto m = torch::maximum(torch::maximum(lse1, lse2), lse3);
  auto e1 = torch::exp(lse1 - m);
  auto e2 = torch::exp(lse2 - m);
  auto e3 = torch::exp(lse3 - m);
  auto o1_f = o1.to(torch::kFloat32);
  auto o2_f = o2.to(torch::kFloat32);
  auto o3_f = o3.to(torch::kFloat32);
  return ((o1_f * e1 + o2_f * e2 + o3_f * e3) / (e1 + e2 + e3))
      .to(o1.scalar_type());
}

torch::Tensor combine_lse_two(const torch::Tensor& lse1,
                              const torch::Tensor& lse2) {
  auto max_l = torch::maximum(lse1, lse2);
  return max_l + torch::log(torch::exp(lse1 - max_l) + torch::exp(lse2 - max_l));
}

torch::Tensor combine_lse_three(const torch::Tensor& lse1,
                                const torch::Tensor& lse2,
                                const torch::Tensor& lse3) {
  auto max_l = torch::maximum(torch::maximum(lse1, lse2), lse3);
  return max_l +
         torch::log(torch::exp(lse1 - max_l) + torch::exp(lse2 - max_l) +
                    torch::exp(lse3 - max_l));
}

void run_genrec_v2_single_batch(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const GenRecAttentionKernelData& attn_data,
    int64_t num_heads,
    int64_t num_kv_heads,
    float scale,
    torch::Tensor& output,
    const std::optional<torch::Tensor>& softmax_lse) {
  const int64_t h = attn_data.history_len;
  const int64_t c = attn_data.context_len;
  const int64_t r = attn_data.real_time_len;
  const int64_t t = attn_data.target_len;
  const int64_t total = h + c + r + t;

  if (h <= 0 || r <= 0 || t <= 0 || total <= 0) {
    output.zero_();
    if (softmax_lse.has_value() && valid_tensor(softmax_lse.value())) {
      softmax_lse.value().zero_();
    }
    return;
  }
  if (total > query.size(2) || total > key.size(2) || total > value.size(2) ||
      total > output.size(2)) {
    output.zero_();
    if (softmax_lse.has_value() && valid_tensor(softmax_lse.value())) {
      softmax_lse.value().zero_();
    }
    return;
  }
  output.zero_();

  auto out_opts = torch::TensorOptions()
                      .dtype(query.scalar_type())
                      .device(query.device());
  auto lse_opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(query.device());
  auto make_out = [&](int64_t seq_len) {
    return torch::empty({1, num_heads, seq_len, query.size(3)}, out_opts);
  };
  auto make_lse = [&](int64_t seq_len) {
    return torch::empty({1, num_heads, seq_len, 1}, lse_opts);
  };

  constexpr int64_t kSparseModeFullMask = 0;
  constexpr int64_t kSparseModeLeftUpCausal = 2;
  constexpr int64_t kDefaultWindow = 2147483647LL;
  constexpr uint64_t kPipelineWorkspaceBytes = 512ULL * 1024ULL * 1024ULL;
  auto causal_mask_2048 = create_compressed_causal_mask_2048(query.device());
  int32_t device_id = query.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  auto& workspace = get_workspace_cache(query.device(), kPipelineWorkspaceBytes);
  void* workspace_ptr = workspace.data_ptr();
  uint64_t workspace_size = static_cast<uint64_t>(workspace.numel());

  // Step 0: history self-attention (write back directly to output slice).
  auto out_hist = output.slice(2, 0, h);
  std::optional<torch::Tensor> lse_hist = std::nullopt;
  if (softmax_lse.has_value() && valid_tensor(softmax_lse.value())) {
    lse_hist = make_lse(h);
  }
  GenRecAttentionKernelData step_data_hist = attn_data;
  step_data_hist.attention_mask = causal_mask_2048;
  auto plan_hist = plan_segment_attention(query.slice(2, 0, h).contiguous(),
                                          key.slice(2, 0, h).contiguous(),
                                          value.slice(2, 0, h).contiguous(),
                                          step_data_hist,
                                          num_heads,
                                          num_kv_heads,
                                          scale,
                                          out_hist,
                                          lse_hist,
                                          kSparseModeLeftUpCausal,
                                          kDefaultWindow,
                                          kDefaultWindow);
  execute_planned_attention(plan_hist, workspace_ptr, workspace_size, stream);
  destroy_planned_attention(plan_hist);

  // Step 2a: Q=ctx+rt+tgt, KV=hist+ctx, full attention.
  // Context output is final result; rt/tgt part will be combined later.
  auto out_2a = output.slice(2, h, h + c + r + t);
  auto lse_2a = make_lse(c + r + t);
  GenRecAttentionKernelData step_data_2a = attn_data;
  step_data_2a.attention_mask =
      create_full_attention_mask(c + r + t, h + c, query.device());
  auto plan_2a = plan_segment_attention(query.slice(2, h, h + c + r + t).contiguous(),
                                        key.slice(2, 0, h + c).contiguous(),
                                        value.slice(2, 0, h + c).contiguous(),
                                        step_data_2a,
                                        num_heads,
                                        num_kv_heads,
                                        scale,
                                        out_2a,
                                        lse_2a,
                                        kSparseModeFullMask,
                                        kDefaultWindow,
                                        kDefaultWindow);
  execute_planned_attention(plan_2a, workspace_ptr, workspace_size, stream);
  destroy_planned_attention(plan_2a);

  // Step 2b: Q=rt+tgt, KV=rt, leftUpCausal.
  auto out_2b = make_out(r + t);
  auto lse_2b = make_lse(r + t);
  GenRecAttentionKernelData step_data_2b = attn_data;
  step_data_2b.attention_mask = causal_mask_2048;
  auto plan_2b = plan_segment_attention(query.slice(2, h + c, h + c + r + t).contiguous(),
                                        key.slice(2, h + c, h + c + r).contiguous(),
                                        value.slice(2, h + c, h + c + r).contiguous(),
                                        step_data_2b,
                                        num_heads,
                                        num_kv_heads,
                                        scale,
                                        out_2b,
                                        lse_2b,
                                        kSparseModeLeftUpCausal,
                                        kDefaultWindow,
                                        kDefaultWindow);
  execute_planned_attention(plan_2b, workspace_ptr, workspace_size, stream);
  destroy_planned_attention(plan_2b);

  // Step 3: target diagonal self-attention.
  auto out_3 = make_out(t);
  auto lse_3 = make_lse(t);
  GenRecAttentionKernelData step_data_3 = attn_data;
  step_data_3.attention_mask = create_diagonal_mask(t, query.device());
  auto plan_3 = plan_segment_attention(
      query.slice(2, h + c + r, h + c + r + t).contiguous(),
      key.slice(2, h + c + r, h + c + r + t).contiguous(),
      value.slice(2, h + c + r, h + c + r + t).contiguous(),
      step_data_3,
      num_heads,
      num_kv_heads,
      scale,
      out_3,
      lse_3,
      /*sparse_mode=*/kSparseModeFullMask,
      /*pre_tokens=*/0,
      /*next_tokens=*/0);
  execute_planned_attention(plan_3, workspace_ptr, workspace_size, stream);
  destroy_planned_attention(plan_3);

  auto rt_out_2a = out_2a.slice(2, c, c + r);
  auto rt_lse_2a = lse_2a.slice(2, c, c + r);
  auto rt_out_2b = out_2b.slice(2, 0, r);
  auto rt_lse_2b = lse_2b.slice(2, 0, r);
  auto rt_final =
      combine_flash_outputs(rt_out_2a, rt_lse_2a, rt_out_2b, rt_lse_2b);
  auto rt_lse_final = combine_lse_two(rt_lse_2a, rt_lse_2b);

  auto tgt_out_2a = out_2a.slice(2, c + r, c + r + t);
  auto tgt_lse_2a = lse_2a.slice(2, c + r, c + r + t);
  auto tgt_out_2b = out_2b.slice(2, r, r + t);
  auto tgt_lse_2b = lse_2b.slice(2, r, r + t);
  auto tgt_final = combine_flash_outputs_three(
      tgt_out_2a, tgt_lse_2a, tgt_out_2b, tgt_lse_2b, out_3, lse_3);
  auto tgt_lse_final = combine_lse_three(tgt_lse_2a, tgt_lse_2b, lse_3);

  int64_t offset = h + c;
  output.slice(2, offset, offset + r).copy_(rt_final);
  offset += r;
  output.slice(2, offset, offset + t).copy_(tgt_final);

  if (softmax_lse.has_value() && valid_tensor(softmax_lse.value()) &&
      softmax_lse.value().dim() == 4 && softmax_lse.value().size(2) >= total) {
    auto lse_out = softmax_lse.value();
    lse_out.zero_();
    int64_t lse_offset = 0;
    if (lse_hist.has_value()) {
      lse_out.slice(2, lse_offset, lse_offset + h).copy_(lse_hist.value());
    } else {
      lse_out.slice(2, lse_offset, lse_offset + h).zero_();
    }
    lse_offset += h;
    lse_out.slice(2, lse_offset, lse_offset + c).copy_(lse_2a.slice(2, 0, c));
    lse_offset += c;
    lse_out.slice(2, lse_offset, lse_offset + r).copy_(rt_lse_final);
    lse_offset += r;
    lse_out.slice(2, lse_offset, lse_offset + t).copy_(tgt_lse_final);
  }
}

}  // namespace

void genrec_fused_infer_attention_score_v3(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const GenRecAttentionKernelData& attn_data,
    int64_t num_heads,
    int64_t num_kv_heads,
    float scale,
    torch::Tensor& output,
    const std::optional<torch::Tensor>& softmax_lse,
    int64_t sparse_mode,
    int64_t pre_tokens,
    int64_t next_tokens) {
  (void)sparse_mode;
  (void)pre_tokens;
  (void)next_tokens;
  run_genrec_v2_single_batch(query,
                             key,
                             value,
                             attn_data,
                             num_heads,
                             num_kv_heads,
                             scale,
                             output,
                             softmax_lse);
}

}  // namespace xllm::kernel::npu
