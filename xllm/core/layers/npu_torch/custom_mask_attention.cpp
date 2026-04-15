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

#include "custom_mask_attention.h"

#include <cmath>
#include <glog/logging.h>
#include <tuple>

namespace xllm {
namespace layer {

CustomMaskAttentionImpl::CustomMaskAttentionImpl(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options,
    int32_t layer_id) {
  const int64_t total_num_heads = args.n_heads();
  const int64_t total_num_kv_heads = args.n_kv_heads().value_or(args.n_heads());
  const int64_t num_kv_head_replicas = 1;
  layer_id_ = layer_id;
  num_heads_ = total_num_heads;
  num_kv_heads_ = total_num_kv_heads;

  head_dim_ = args.head_dim();
  q_size_ = num_heads_ * head_dim_;
  kv_size_ = num_kv_heads_ * head_dim_;
  scaling_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));

  qkv_proj_ = register_module(
      "qkv_proj",
      QKVParallelLinear(args.hidden_size(),
                        num_heads_,
                        num_kv_heads_,
                        args.head_dim(),
                        num_kv_head_replicas,
                        /*bias=*/args.attention_bias(),
                        /*gather_output=*/false,
                        parallel_args,
                        options));

  o_proj_ = register_module("o_proj",
                            RowParallelLinear(total_num_heads * head_dim_,
                                              args.hidden_size(),
                                              /*bias=*/false,
                                              /*input_is_parallelized=*/true,
                                              /*if_reduce_results=*/true,
                                              quant_args,
                                              parallel_args.tp_group_,
                                              options));

  q_norm_ = register_module(
      "q_norm", Qwen3NextRMSNorm(head_dim_, args.rms_norm_eps(), options));
  k_norm_ = register_module(
      "k_norm", Qwen3NextRMSNorm(head_dim_, args.rms_norm_eps(), options));

  const int rotary_dim =
      static_cast<int>(head_dim_ * args.partial_rotary_factor());
  rotary_emb_ =
      register_module("rotary_emb",
                      PartialRotaryEmbedding(rotary_dim,
                                             args.max_position_embeddings(),
                                             args.rope_theta(),
                                             head_dim_,
                                             true,
                                             false,
                                             options));

  attn_ = register_module("attn",
                          MTGRAttention(
                              num_heads_, head_dim_, scaling_, num_kv_heads_));
}

torch::Tensor CustomMaskAttentionImpl::forward(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache) {
  LOG(INFO) << "[MTGR_TRACE][ATTN] forward begin hidden_shape="
            << hidden_states.sizes() << " positions_shape="
            << positions.sizes();

  auto qkv = qkv_proj_->forward(hidden_states);
  auto q = qkv.slice(/*dim=*/-1, 0, q_size_);
  auto k = qkv.slice(/*dim=*/-1, q_size_, q_size_ + kv_size_);
  auto v = qkv.slice(/*dim=*/-1, q_size_ + kv_size_, q_size_ + 2 * kv_size_);
  LOG(INFO) << "[MTGR_TRACE][ATTN] qkv done q_shape=" << q.sizes()
            << " k_shape=" << k.sizes() << " v_shape=" << v.sizes();

  const int64_t tokens = q.size(0);
  auto q_reshaped = q.reshape({tokens, num_heads_, head_dim_});
  auto q_normed = q_norm_->forward(q_reshaped);
  auto k_reshaped = k.reshape({tokens, num_kv_heads_, head_dim_});
  auto k_normed = k_norm_->forward(k_reshaped);

  q = q_normed.view({tokens, q_size_});
  k = k_normed.view({tokens, kv_size_});

  // TEMP: Disable rotary for MTGR bring-up to unblock end-to-end pipeline
  // validation. Re-enable after fixing RopeOperation shape/setup on NPU.
  // rotary_emb_->forward(positions, q, k);
  LOG(INFO) << "[MTGR_TRACE][ATTN] rotary skipped q_shape=" << q.sizes()
            << " k_shape=" << k.sizes();
  auto out = std::get<0>(attn_->forward(attn_metadata, q, k, v, kv_cache));
  LOG(INFO) << "[MTGR_TRACE][ATTN] core attention done out_shape="
            << out.sizes();

  out = o_proj_->forward(out);
  LOG(INFO) << "[MTGR_TRACE][ATTN] o_proj done out_shape=" << out.sizes();
  return out;
}

void CustomMaskAttentionImpl::load_state_dict(const StateDict& state_dict) {
  qkv_proj_->load_state_dict(state_dict, {"q_proj.", "k_proj.", "v_proj."});
  o_proj_->load_state_dict(state_dict.get_dict_with_prefix("o_proj."));
  if (auto w = state_dict.get_tensor("q_norm.weight"); w.defined()) {
    q_norm_->load_state_dict(StateDict({{"weight", w}}));
  }
  if (auto w = state_dict.get_tensor("k_norm.weight"); w.defined()) {
    k_norm_->load_state_dict(StateDict({{"weight", w}}));
  }
}

}  // namespace layer
}  // namespace xllm
