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

#include <utility>

#include "kernels/cuda/mtgr_hopper_attention_runtime.h"
#include "util/mtgr_nvtx.h"
#include "util/mtgr_trace.h"

namespace xllm {
namespace layer {

MTGRAttentionImpl::MTGRAttentionImpl(int64_t num_heads,
                                     int64_t head_size,
                                     float scale,
                                     int64_t num_kv_heads)
    : num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads) {}

MTGRAttentionImpl::~MTGRAttentionImpl() = default;

std::tuple<torch::Tensor, std::optional<torch::Tensor>>
MTGRAttentionImpl::forward(const AttentionMetadata& attn_metadata,
                           torch::Tensor& query,
                           torch::Tensor& key,
                           torch::Tensor& value,
                           KVCache& kv_cache) {
  MTGR_NVTX_RANGE(1, "MTGR/attention/fused");
  std::optional<torch::Tensor> output_lse = std::nullopt;
  if (attn_metadata.is_dummy) {
    return {torch::empty_like(query), output_lse};
  }

  const int64_t total_q = query.size(0);
  MTGR_TRACE(1) << "[ATTN] mtgr_attention fused begin total_q=" << total_q
                << " num_heads=" << num_heads_
                << " num_kv_heads=" << num_kv_heads_
                << " head_size=" << head_size_
                << " match_mode="
                << static_cast<int32_t>(attn_metadata.mtgr_match_mode)
                << " max_seq_len=" << attn_metadata.max_seq_len;
  torch::Tensor query_snd;
  torch::Tensor key_snd;
  torch::Tensor value_snd;
  torch::Tensor raw_key_snd;
  torch::Tensor raw_value_snd;
  {
    MTGR_NVTX_RANGE(2, "MTGR/attention/prepare_qkv_snd");
    query_snd = query.view({total_q, num_heads_, head_size_}).contiguous();
    raw_key_snd = key.view({total_q, num_kv_heads_, head_size_}).contiguous();
    raw_value_snd =
        value.view({total_q, num_kv_heads_, head_size_}).contiguous();
    key_snd = raw_key_snd;
    value_snd = raw_value_snd;
    if (num_kv_heads_ != num_heads_) {
      const int64_t repeat_factor = num_heads_ / num_kv_heads_;
      key_snd = key_snd.repeat_interleave(repeat_factor, 1);
      value_snd = value_snd.repeat_interleave(repeat_factor, 1);
    }
    key_snd = key_snd.contiguous();
    value_snd = value_snd.contiguous();
  }
  auto output_snd = torch::empty_like(query_snd);
  auto key_cache = kv_cache.get_k_cache();
  auto value_cache = kv_cache.get_v_cache();
  const int64_t block_size = key_cache.size(1);
  MTGR_TRACE(2) << "[ATTN] mtgr_attention launch_shapes query_snd="
                << query_snd.sizes() << " key_snd=" << key_snd.sizes()
                << " value_snd=" << value_snd.sizes()
                << " key_cache=" << key_cache.sizes()
                << " value_cache=" << value_cache.sizes()
                << " block_table=" << attn_metadata.block_table.sizes()
                << " block_size=" << block_size
                << " segment_offsets="
                << attn_metadata.mtgr_segment_offsets_i32.sizes()
                << " q_seq_starts="
                << attn_metadata.mtgr_q_seq_starts_i32.sizes()
                << " matched_prefix_lens="
                << attn_metadata.mtgr_matched_prefix_lens_i32.sizes();

  {
    MTGR_NVTX_RANGE(1, "MTGR/kernel/hopper_unified_call");
    kernel::cuda::mtgr_ragged_segment_attention_hopper_unified_cuda(
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
        block_size,
        attn_metadata.max_seq_len,
        scale_,
        output_snd);
  }

  {
    MTGR_NVTX_RANGE(1, "MTGR/kernel/kv_writeback_call");
    kernel::cuda::mtgr_kv_cache_writeback_cuda(
        raw_key_snd,
        raw_value_snd,
        attn_metadata.mtgr_segment_offsets_i32,
        attn_metadata.mtgr_q_seq_starts_i32,
        attn_metadata.mtgr_matched_prefix_lens_i32,
        attn_metadata.block_table,
        key_cache,
        value_cache,
        attn_metadata.max_seq_len);
  }

  MTGR_TRACE(1) << "[ATTN] mtgr_attention fused end output_shape="
                << output_snd.sizes();
  return {output_snd.view({total_q, num_heads_ * head_size_}).contiguous(),
          output_lse};
}

}  // namespace layer
}  // namespace xllm
