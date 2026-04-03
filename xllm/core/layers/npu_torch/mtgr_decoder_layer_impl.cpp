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

#include "mtgr_decoder_layer_impl.h"

namespace xllm {
namespace layer {

MTGRDecoderLayerImpl::MTGRDecoderLayerImpl(const ModelContext& context,
                                           int32_t layer_id) {
  const auto& model_args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& parallel_args = context.get_parallel_args();
  const auto& options = context.get_tensor_options();

  attention_ = register_module(
      "self_attn",
      CustomMaskAttention(
          model_args, quant_args, parallel_args, options, layer_id));

  input_norm_ = register_module(
      "input_layernorm",
      Qwen3NextRMSNorm(
          model_args.hidden_size(), model_args.rms_norm_eps(), options));

  post_norm_ = register_module(
      "post_attention_layernorm",
      Qwen3NextRMSNorm(
          model_args.hidden_size(), model_args.rms_norm_eps(), options));

  mlp_ = register_module("mlp",
                         DenseMLP(model_args.hidden_size(),
                                  model_args.intermediate_size(),
                                  true,
                                  false,
                                  model_args.hidden_act(),
                                  /*enable_result_reduction=*/true,
                                  quant_args,
                                  parallel_args.tp_group_,
                                  options));
}

void MTGRDecoderLayerImpl::load_state_dict(const StateDict& state_dict) {
  attention_->load_state_dict(state_dict.get_dict_with_prefix("self_attn."));
  input_norm_->load_state_dict(
      state_dict.get_dict_with_prefix("input_layernorm."));
  post_norm_->load_state_dict(
      state_dict.get_dict_with_prefix("post_attention_layernorm."));
  mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
}

torch::Tensor MTGRDecoderLayerImpl::forward(
    torch::Tensor& x,
    torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params) {
  (void)input_params;

  torch::Tensor residual = x;
  x = input_norm_(x);
  x = attention_->forward(positions, x, attn_metadata, kv_cache);

  auto orig_dtype = x.dtype();
  if (orig_dtype == torch::kBFloat16) {
    x = x.to(torch::kFloat32);
    residual = residual.to(torch::kFloat32);
  }
  x = x + residual;

  residual = x;
  x = x.to(orig_dtype);
  x = post_norm_(x);
  x = mlp_(x);

  orig_dtype = x.dtype();
  if (orig_dtype == torch::kBFloat16) {
    x = x.to(torch::kFloat32);
    residual = residual.to(torch::kFloat32);
  }
  x = x + residual;
  x = x.to(orig_dtype);
  return x;
}

}  // namespace layer
}  // namespace xllm
