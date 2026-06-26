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

namespace xllm::kernel::cuda {

// CUDA FlashInfer MTGR attention metadata for one sequence.
// Sequence semantic:
//   [history | context | real_time | target]
// matched_prefix_len is relative to the full logical sequence before trimming.
struct MtgrFlashinferMetadata {
  int64_t history_len = 0;
  int64_t context_len = 0;
  int64_t real_time_len = 0;
  int64_t target_len = 0;
  int64_t matched_prefix_len = 0;
  int64_t block_size = 128;
  // [1, max_blocks] or [max_blocks], int32.
  torch::Tensor block_table;
  // Slot mapping for current local key/value tokens, int32.
  // For matched branches this tensor corresponds to "unmatched prefix" tokens.
  torch::Tensor slot_mapping;
};

// query/key/value: [1, local_seq_len, num_heads/num_kv_heads, head_dim] (BSND).
// key_cache/value_cache: [n_blocks, block_size, num_kv_heads, head_dim].
// output: same shape as query.
//
// This function dispatches FA/PA path according to matched_prefix_len:
// - matched == 0: full FA pipeline
// - matched < history: partial-history branch (PA + FA)
// - history <= matched < history + context: partial-context branch (PA + FA)
// - history + context <= matched < history + context + real_time:
//     partial-realtime branch (PA + FA)
//
// It intentionally keeps merge in torch math (LSE-aware) and delegates attention
// kernels to FlashInfer wrappers.
void mtgr_flashinfer_attention_forward(const torch::Tensor& query,
                                       const torch::Tensor& key,
                                       const torch::Tensor& value,
                                       torch::Tensor& key_cache,
                                       torch::Tensor& value_cache,
                                       const MtgrFlashinferMetadata& metadata,
                                       double sm_scale,
                                       torch::Tensor& output);

}  // namespace xllm::kernel::cuda
