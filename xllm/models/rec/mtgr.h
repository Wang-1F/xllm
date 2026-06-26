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

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
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
#include "core/util/mtgr_nvtx.h"
#include "core/util/mtgr_trace.h"
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
    MTGR_NVTX_RANGE(1, "MTGR/model/forward");
    torch::NoGradGuard no_grad;
    (void)tokens;

    auto local_positions = positions;
    const auto* mtgr_params = input_params.mtgr_params();
    const bool has_input_embedding = input_params.input_embedding.defined();
    MTGR_TRACE(1) << "[MODEL] forward begin input_mode="
                  << (has_input_embedding ? "input_embedding" : "token_ids")
                  << " positions_shape=" << positions.sizes()
                  << " kv_layers=" << kv_caches.size();

    torch::Tensor h;
    if (has_input_embedding) {
      MTGR_NVTX_RANGE(2, "MTGR/model/use_input_embedding");
      h = input_params.input_embedding;
      if (h.device() != device_) {
        h = h.to(device_);
      }
      if (h.dtype() != dtype_) {
        h = h.to(dtype_);
      }
    } else {
      MTGR_NVTX_RANGE(2, "MTGR/model/fake_embedding_lookup");
      h = fake_input_embedding_lookup(mtgr_params->mtgr_input_token_ids_i64);
    }
    MTGR_TRACE(1) << "[MODEL] hidden_states_ready shape=" << h.sizes()
                  << " dtype=" << h.scalar_type()
                  << " device=" << h.device();

    auto attn_metadata = [&]() {
      MTGR_NVTX_RANGE(2, "MTGR/model/build_attention_metadata");
      return layer::AttentionMetadataBuilder::build(
          input_params, model_args_, build_attention_mask(input_params));
    }();
    MTGR_TRACE(2) << "[MODEL] attention_metadata mtgr_match_mode="
                  << static_cast<int32_t>(attn_metadata.mtgr_match_mode)
                  << " segment_offsets="
                  << attn_metadata.mtgr_segment_offsets_i32.sizes()
                  << " q_seq_starts="
                  << attn_metadata.mtgr_q_seq_starts_i32.sizes()
                  << " matched_prefix_lens="
                  << attn_metadata.mtgr_matched_prefix_lens_i32.sizes()
                  << " block_table=" << attn_metadata.block_table.sizes();
    for (size_t i = 0; i < layers_.size(); ++i) {
      MTGR_NVTX_RANGE(2, "MTGR/model/layer");
      MTGR_TRACE(2) << "[MODEL] layer_begin index=" << i
                    << " hidden_shape=" << h.sizes();
      h = layers_[i]->forward(
          h, local_positions, attn_metadata, kv_caches[i], input_params);
      MTGR_TRACE(2) << "[MODEL] layer_end index=" << i
                    << " hidden_shape=" << h.sizes();
    }

    {
      MTGR_NVTX_RANGE(2, "MTGR/model/final_norm_mlp");
      h = norm_(h);
      h = post_mlp_(h);
    }
    MTGR_TRACE(1) << "[MODEL] forward end output_shape=" << h.sizes()
                  << " dtype=" << h.scalar_type();
    return ModelOutput(h);
  }

  void load_state_dict(const StateDict& state_dict) {
    validate_and_track_config_parameter_shapes(state_dict);

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

  void begin_load_state_dict() {
    fake_initialize_post_mlp_weights_from_config();
    loaded_config_parameters_.clear();
  }

  void fake_initialize_all_weights_from_config() {
    torch::NoGradGuard no_grad;
    for (auto& param : parameters()) {
      param.zero_();
    }
  }

  void fake_initialize_post_mlp_weights_from_config() {
    torch::NoGradGuard no_grad;
    for (auto& param : post_mlp_->parameters()) {
      param.zero_();
    }
  }

  void finalize_load_state_dict() {
    const auto specs = mtgr_config_parameter_specs();
    std::vector<std::string> missing_required;
    std::vector<std::string> missing_post_mlp;

    for (const auto& spec : specs) {
      if (loaded_config_parameters_.contains(spec.name)) {
        continue;
      }
      if (is_post_mlp_parameter(spec.name)) {
        missing_post_mlp.push_back(spec.name);
      } else {
        missing_required.push_back(spec.name);
      }
    }

    CHECK(missing_required.empty())
        << "Missing MTGR checkpoint weights required by config.json mapping: "
        << join_strings(missing_required);

    if (missing_post_mlp.empty()) {
      return;
    }

    std::vector<std::string> loaded_post_mlp;
    for (const auto& spec : mtgr_post_mlp_parameter_specs()) {
      if (loaded_config_parameters_.contains(spec.name)) {
        loaded_post_mlp.push_back(spec.name);
      }
    }
    LOG(WARNING) << "Incomplete MTGR post_mlp checkpoint weights according to "
                    "config.json mapping; fake-initialized missing weights to "
                    "zero: missing "
                 << join_strings(missing_post_mlp) << ", loaded "
                 << join_strings(loaded_post_mlp);
  }

 private:
  struct MTGRParameterSpec {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<std::string> config_fields;
  };

  static bool is_post_mlp_parameter(const std::string& name) {
    return name.rfind("post_mlp.", 0) == 0;
  }

  static std::string join_strings(const std::vector<std::string>& values) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
      if (i > 0) {
        oss << ", ";
      }
      oss << values[i];
    }
    oss << "]";
    return oss.str();
  }

  static std::string shape_to_string(const std::vector<int64_t>& shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i > 0) {
        oss << ", ";
      }
      oss << shape[i];
    }
    oss << "]";
    return oss.str();
  }

  std::vector<MTGRParameterSpec> mtgr_config_parameter_specs() const {
    // xLLM sees checkpoint names after load_model strips the top-level
    // "model." prefix; vLLM uses the same mapping with that prefix attached.
    const int64_t hidden_size = model_args_.hidden_size();
    const int64_t intermediate_size = model_args_.intermediate_size();
    const int64_t num_layers = model_args_.n_layers();
    const int64_t num_heads = model_args_.n_heads();
    const int64_t num_kv_heads =
        model_args_.n_kv_heads().value_or(model_args_.n_heads());
    const int64_t head_dim = model_args_.head_dim();
    const bool attention_bias = model_args_.attention_bias();

    std::vector<MTGRParameterSpec> specs;
    specs.reserve(1 + num_layers * (attention_bias ? 15 : 12) + 4);

    auto add = [&](std::string name,
                   std::vector<int64_t> shape,
                   std::vector<std::string> config_fields) {
      specs.push_back(MTGRParameterSpec{
          std::move(name), std::move(shape), std::move(config_fields)});
    };

    const std::vector<std::string> hidden_fields = {"hidden_size"};
    const std::vector<std::string> mlp_fields = {
        "hidden_size", "intermediate_size", "hidden_act"};
    const std::vector<std::string> q_fields = {
        "hidden_size", "num_attention_heads", "head_dim"};
    const std::vector<std::string> kv_fields = {
        "hidden_size", "num_key_value_heads", "head_dim"};

    for (int64_t layer_id = 0; layer_id < num_layers; ++layer_id) {
      const std::string prefix = "layers." + std::to_string(layer_id);
      add(prefix + ".input_layernorm.weight", {hidden_size}, hidden_fields);
      add(prefix + ".self_attn.q_proj.weight",
          {num_heads * head_dim, hidden_size},
          q_fields);
      add(prefix + ".self_attn.k_proj.weight",
          {num_kv_heads * head_dim, hidden_size},
          kv_fields);
      add(prefix + ".self_attn.v_proj.weight",
          {num_kv_heads * head_dim, hidden_size},
          kv_fields);
      if (attention_bias) {
        add(prefix + ".self_attn.q_proj.bias",
            {num_heads * head_dim},
            q_fields);
        add(prefix + ".self_attn.k_proj.bias",
            {num_kv_heads * head_dim},
            kv_fields);
        add(prefix + ".self_attn.v_proj.bias",
            {num_kv_heads * head_dim},
            kv_fields);
      }
      add(prefix + ".self_attn.o_proj.weight",
          {hidden_size, num_heads * head_dim},
          q_fields);
      add(prefix + ".self_attn.q_norm.weight", {head_dim}, {"head_dim"});
      add(prefix + ".self_attn.k_norm.weight", {head_dim}, {"head_dim"});
      add(prefix + ".post_attention_layernorm.weight",
          {hidden_size},
          hidden_fields);
      add(prefix + ".mlp.gate_proj.weight",
          {intermediate_size, hidden_size},
          mlp_fields);
      add(prefix + ".mlp.up_proj.weight",
          {intermediate_size, hidden_size},
          mlp_fields);
      add(prefix + ".mlp.down_proj.weight",
          {hidden_size, intermediate_size},
          mlp_fields);
    }

    add("norm.weight", {hidden_size}, hidden_fields);
    for (auto& spec : mtgr_post_mlp_parameter_specs()) {
      specs.push_back(std::move(spec));
    }
    return specs;
  }

  std::vector<MTGRParameterSpec> mtgr_post_mlp_parameter_specs() const {
    const int64_t hidden_size = model_args_.hidden_size();
    const int64_t intermediate_size = model_args_.intermediate_size();
    const std::vector<std::string> mlp_fields = {
        "hidden_size", "intermediate_size", "hidden_act"};
    return {
        {"post_mlp.gate_proj.weight",
         {intermediate_size, hidden_size},
         mlp_fields},
        {"post_mlp.up_proj.weight",
         {intermediate_size, hidden_size},
         mlp_fields},
        {"post_mlp.down_proj.weight",
         {hidden_size, intermediate_size},
         mlp_fields},
    };
  }

  void validate_and_track_config_parameter_shapes(const StateDict& state_dict) {
    for (const auto& spec : mtgr_config_parameter_specs()) {
      auto tensor = state_dict.get_tensor(spec.name);
      if (!tensor.defined()) {
        continue;
      }
      const auto actual_shape = tensor.sizes().vec();
      CHECK(actual_shape == spec.shape)
          << "MTGR checkpoint parameter shape does not match config.json "
             "mapping for "
          << state_dict.prefix() << spec.name << ": got "
          << shape_to_string(actual_shape) << ", expected "
          << shape_to_string(spec.shape) << " from "
          << join_strings(spec.config_fields);
      loaded_config_parameters_.insert(spec.name);
    }
  }

  torch::Tensor fake_input_embedding_lookup(const torch::Tensor& token_ids) {
    auto ids = token_ids.to(device_).to(torch::kFloat32).reshape({-1, 1});
    auto dims = torch::arange(
                    model_args_.hidden_size(),
                    torch::TensorOptions().dtype(torch::kFloat32).device(device_))
                    .reshape({1, -1});
    const float scale =
        1.0f / std::sqrt(static_cast<float>(model_args_.hidden_size()));
    return (torch::sin(ids * 0.017f + dims * 0.013f) * scale).to(dtype_);
  }

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
  std::unordered_set<std::string> loaded_config_parameters_;
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
    if (!loader->has_model_weights()) {
      model_->fake_initialize_all_weights_from_config();
      return;
    }

    model_->begin_load_state_dict();
    for (const auto& state_dict : loader->get_state_dicts()) {
      StateDict model_state_dict = state_dict->get_dict_with_prefix(prefix);
      if (model_state_dict.size() == 0) {
        model_state_dict = *state_dict;
      }
      model_->load_state_dict(model_state_dict);
    }
    model_->finalize_load_state_dict();
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
  LOAD_ARG_OR(partial_rotary_factor, "partial_rotary_factor", 1.0f);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(rope_theta, "rope_theta", 10000000.0f);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(vocab_size, "vocab_size", 151936);
  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});

}  // namespace xllm
