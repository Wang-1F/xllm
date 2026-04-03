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

#pragma once

#include <torch/torch.h>

#include <optional>

namespace xllm::kernel::npu {

// Extra optional tensors for aclnnFusedInferAttentionScoreV3GetWorkspaceSize.
struct GenRecAttentionKernelData {
  torch::Tensor attention_mask;
  torch::Tensor actual_seq_lengths;
  torch::Tensor actual_seq_lengths_kv;
  torch::Tensor block_table;
  torch::Tensor query_padding_size;
  torch::Tensor kv_padding_size;
  // Sequence layout:
  // [history | context | real_time | target].
  int64_t history_len = 0;
  int64_t context_len = 0;
  int64_t real_time_len = 0;
  int64_t target_len = 0;
};

// query/key/value/output layout: BNSD.
void genrec_fused_infer_attention_score_v3(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const GenRecAttentionKernelData& attn_data,
    int64_t num_heads,
    int64_t num_kv_heads,
    float scale,
    torch::Tensor& output,
    const std::optional<torch::Tensor>& softmax_lse = std::nullopt,
    int64_t sparse_mode = 0,
    int64_t pre_tokens = 2147483647LL,
    int64_t next_tokens = 2147483647LL);

}  // namespace xllm::kernel::npu
