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

#include <algorithm>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/common/global_flags.h"
#include "core/framework/model/causal_lm.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context.h"
#include "core/layers/common/attention_mask.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/common/dense_mlp.h"
#include "core/layers/common/qwen3_next_rms_norm.h"
#include "core/layers/mtgr_decoder_layer.h"
#include "models/model_registry.h"

namespace xllm {

class MTGRModelImpl : public torch::nn::Module {
 public:
  explicit MTGRModelImpl(const ModelContext& context)
      : model_args_(context.get_model_args()),
        device_(context.get_tensor_options().device()),
        dtype_(context.get_tensor_options().dtype().toScalarType()) {
    const auto& quant_args = context.get_quant_args();
    const auto& parallel_args = context.get_parallel_args();
    const auto& options = context.get_tensor_options();

    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(model_args_.n_layers());

    norm_ = register_module(
        "norm",
        layer::Qwen3NextRMSNorm(
            model_args_.hidden_size(), model_args_.rms_norm_eps(), options));
    post_mlp_ = register_module("post_mlp",
                                layer::DenseMLP(model_args_.hidden_size(),
                                                model_args_.intermediate_size(),
                                                true,
                                                false,
                                                model_args_.hidden_act(),
                                                /*enable_result_reduction=*/true,
                                                quant_args,
                                                parallel_args.tp_group_,
                                                options));

    const int32_t mask_value = FLAGS_enable_chunked_prefill ? -9984 : 1;
    attn_mask_ = layer::AttentionMask(
        options.device(), options.dtype().toScalarType(), mask_value);
    dp_size_ = parallel_args.dp_size();

    for (int32_t layer_id = 0; layer_id < model_args_.n_layers(); ++layer_id) {
      auto layer = layer::MTGRDecoderLayer(context, layer_id);
      layers_.push_back(layer);
      blocks_->push_back(layer);
    }
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    torch::NoGradGuard no_grad;
    (void)tokens;

    auto local_positions = positions;
    CHECK(input_params.input_embedding.defined())
        << "MTGR requires input_params.input_embedding as hidden_states input.";

    torch::Tensor h = input_params.input_embedding;
    if (h.device() != device_) {
      h = h.to(device_);
    }
    if (h.dtype() != dtype_) {
      h = h.to(dtype_);
    }

    auto attn_metadata = layer::AttentionMetadataBuilder::build(
        input_params, model_args_, build_attention_mask(input_params));
    for (size_t i = 0; i < layers_.size(); ++i) {
      h = layers_[i]->forward(
          h, local_positions, attn_metadata, kv_caches[i], input_params);
    }

    h = norm_(h);
    h = post_mlp_(h);
    return ModelOutput(h);
  }

  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));

    auto post_mlp_dict = state_dict.get_dict_with_prefix("post_mlp.");
    if (post_mlp_dict.size() > 0) {
      post_mlp_->load_state_dict(post_mlp_dict);
    }
  }

 private:
  torch::Tensor build_attention_mask(const ModelInputParams& input_params) {
    max_seq_len_ = std::max(input_params.kv_max_seq_len, max_seq_len_);
    if (!FLAGS_enable_chunked_prefill) {
      return attn_mask_.get_attn_mask(max_seq_len_, dtype_, device_);
    }

    const int32_t num_sequences = input_params.num_sequences;
    if (num_sequences <= 0) {
      return attn_mask_.get_attn_mask(max_seq_len_, dtype_, device_);
    }

    std::vector<torch::Tensor> req_mask_vec;
    req_mask_vec.reserve(num_sequences);
    for (int32_t j = 0; j < num_sequences; ++j) {
      req_mask_vec.emplace_back(
          attn_mask_.gen_append_mask(input_params.q_seq_lens_vec[j],
                                     input_params.kv_seq_lens_vec[j],
                                     max_seq_len_,
                                     dtype_,
                                     device_));
    }
    return torch::cat(req_mask_vec, 0);
  }

  ModelArgs model_args_;
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<layer::MTGRDecoderLayer> layers_;
  int32_t max_seq_len_ = 0;
  int32_t dp_size_ = 1;
  torch::Device device_;
  torch::ScalarType dtype_ = torch::kFloat;
  layer::AttentionMask attn_mask_;
  layer::Qwen3NextRMSNorm norm_{nullptr};
  layer::DenseMLP post_mlp_{nullptr};
};
TORCH_MODULE(MTGRModel);

class MTGRForConditionalGenerationImpl : public torch::nn::Module {
 public:
  explicit MTGRForConditionalGenerationImpl(const ModelContext& context)
      : model_(register_module("model", MTGRModel(context))) {}

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    return model_->forward(tokens, positions, kv_caches, input_params);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes) {
    if (selected_idxes.defined()) {
      return hidden_states.index_select(/*dim=*/0, selected_idxes);
    }
    return hidden_states;
  }

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes) {
    if (selected_idxes.defined()) {
      return hidden_states.index_select(/*dim=*/0, selected_idxes);
    }
    return hidden_states;
  }

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model.") {
    for (const auto& state_dict : loader->get_state_dicts()) {
      StateDict model_state_dict = state_dict->get_dict_with_prefix(prefix);
      if (model_state_dict.size() == 0) {
        model_state_dict = *state_dict;
      }
      model_->load_state_dict(model_state_dict);
    }
  }

  void prepare_expert_weight(int32_t layer_id,
                             const std::vector<int32_t>& expert_ids) {
    (void)layer_id;
    (void)expert_ids;
  }

  void update_expert_weight(int32_t layer_id) { (void)layer_id; }

 private:
  MTGRModel model_{nullptr};
};
TORCH_MODULE(MTGRForConditionalGeneration);

using MTGRCausalLM = CausalLMImpl<MTGRForConditionalGeneration>;
static_assert(std::is_base_of_v<CausalLM, MTGRCausalLM>,
              "MTGR must satisfy CausalLM contract.");

REGISTER_REC_MODEL(mtgr, MTGRForConditionalGeneration);

REGISTER_MODEL_ARGS(mtgr, [&] {
  LOAD_ARG_OR(model_type, "model_type", "mtgr");
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");
  LOAD_ARG_OR(attention_bias, "attention_bias", false);
  LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0f);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 151643);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 151645);
  LOAD_ARG_OR(head_dim, "head_dim", 128);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "hidden_size", 2048);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 5120);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 262144);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 16);
  LOAD_ARG_OR_FUNC(
      n_kv_heads, "num_key_value_heads", [&] { return args->n_heads(); });
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 24);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(rope_theta, "rope_theta", 10000000.0f);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(vocab_size, "vocab_size", 151936);
  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});

}  // namespace xllm
