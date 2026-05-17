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

#include "mtgr_attention_product_test.h"

#include <cstdint>

#include "common/global_flags.h"
#include "mtgr_hopper_attention_runtime.h"

namespace xllm::kernel::cuda::test {

MTGRAttentionTestImpl::MTGRAttentionTestImpl(int64_t num_heads,
                                             int64_t head_size,
                                             float scale,
                                             int64_t num_kv_heads)
    : num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads) {}

std::tuple<torch::Tensor, std::optional<torch::Tensor>>
MTGRAttentionTestImpl::forward(
    const xllm::layer::AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    xllm::KVCache& kv_cache) {
  std::optional<torch::Tensor> output_lse = std::nullopt;
  if (attn_metadata.is_dummy) {
    return {torch::empty_like(query), output_lse};
  }

  const int64_t total_q = query.size(0);

  auto query_snd = query.view({total_q, num_heads_, head_size_}).contiguous();
  auto key_snd = key.view({total_q, num_kv_heads_, head_size_}).contiguous();
  auto value_snd =
      value.view({total_q, num_kv_heads_, head_size_}).contiguous();
  auto output_snd = torch::empty_like(query_snd);
  auto key_cache = kv_cache.get_k_cache();
  auto value_cache = kv_cache.get_v_cache();

  xllm::kernel::cuda::mtgr_ragged_segment_attention_hopper_unified_cuda(
      query_snd,
      key_snd,
      value_snd,
      attn_metadata.mtgr_segment_offsets_i32,
      attn_metadata.mtgr_segment_rules_i32,
      attn_metadata.mtgr_q_seq_starts_i32,
      attn_metadata.mtgr_matched_prefix_lens_i32,
      static_cast<int64_t>(attn_metadata.mtgr_match_mode),
      key_cache,
      value_cache,
      attn_metadata.block_table,
      key_cache.size(1),
      attn_metadata.max_query_len,
      scale_,
      output_snd);

  return {output_snd.view({total_q, num_heads_ * head_size_}).contiguous(),
          output_lse};
}

}  // namespace xllm::kernel::cuda::test
