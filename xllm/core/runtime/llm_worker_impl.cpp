/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "llm_worker_impl.h"

#include <c10/core/DeviceGuard.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <optional>
#include <sstream>
#include <utility>

#include "common/device_monitor.h"
#include "common/metrics.h"
#include "common/types.h"
#include "core/common/global_flags.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#if defined(USE_CUDA)
#include "layers/common/flashinfer_workspace.h"
#endif
#include "models/model_registry.h"
#include "util/threadpool.h"
#include "util/timer.h"
#include "util/utils.h"
#if defined(USE_NPU)
#include <tuple>

#include "kernels/npu/xllm_ops/beam_search_group.h"
#include "kernels/npu/xllm_ops/cache_select.h"
#endif

namespace xllm {

LLMWorkerImpl::LLMWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options)
    : WorkerImpl(parallel_args, device, options) {
  device_.set_device();
#if defined(USE_CUDA)
  // initialize flashinfer workspace
  layer::FlashinferWorkspace::get_instance().initialize(device_);
#endif
}

bool LLMWorkerImpl::init_model(ModelContext& context) {
  CHECK(model_ == nullptr) << "Model is already initialized.";

  // Try to create a causal LM model
  model_ = create_llm_model(context);

  // Dont find model in causal models
  CHECK(model_ != nullptr) << "Failed to create model.";
  model_executor_ = std::make_unique<Executor>(
      model_.get(), context.get_model_args(), device_, options_);

  if (FLAGS_enable_eplb) {
    eplb_executor_ = std::make_unique<EplbExecutor>(model_.get(), device_);
  }

  if (FLAGS_enable_beam_search_kernel) {
    beam_searcher_ = std::make_unique<BeamSearcher>();
  }
  return true;
}

namespace {

void printModelInputParams(const ModelInputParams& params) {
  LOG(INFO) << "=== ModelInputParams Debug Info ===";
  
  // Basic boolean and integer fields
  LOG(INFO) << "empty_kv_cache: " << params.empty_kv_cache;
  LOG(INFO) << "is_prefill: " << params.is_prefill;
  LOG(INFO) << "global_empty_kv_cache: " << params.global_empty_kv_cache;
  LOG(INFO) << "num_sequences: " << params.num_sequences;
  LOG(INFO) << "kv_max_seq_len: " << params.kv_max_seq_len;
  LOG(INFO) << "q_max_seq_len: " << params.q_max_seq_len;
  LOG(INFO) << "prefill_seq_len: " << params.prefill_seq_len;
  LOG(INFO) << "beam_width: " << params.beam_width;
  LOG(INFO) << "current_round: " << params.current_round;
  LOG(INFO) << "total_round: " << params.total_round;
  
  // Vector fields
  LOG(INFO) << "kv_seq_lens_vec size: " << params.kv_seq_lens_vec.size();
  if (!params.kv_seq_lens_vec.empty()) {
      std::ostringstream oss;
      for (size_t i = 0; i < params.kv_seq_lens_vec.size() && i < 10; ++i) {
          oss << params.kv_seq_lens_vec[i] << " ";
      }
      if (params.kv_seq_lens_vec.size() > 10) oss << "...";
      LOG(INFO) << "kv_seq_lens_vec: [" << oss.str() << "]";
  }
  
  LOG(INFO) << "q_seq_lens_vec size: " << params.q_seq_lens_vec.size();
  if (!params.q_seq_lens_vec.empty()) {
      std::ostringstream oss;
      for (size_t i = 0; i < params.q_seq_lens_vec.size() && i < 10; ++i) {
          oss << params.q_seq_lens_vec[i] << " ";
      }
      if (params.q_seq_lens_vec.size() > 10) oss << "...";
      LOG(INFO) << "q_seq_lens_vec: [" << oss.str() << "]";
  }
  
  LOG(INFO) << "decode_kv_seq_lens_vec size: " << params.decode_kv_seq_lens_vec.size();
  LOG(INFO) << "decode_q_seq_lens_vec size: " << params.decode_q_seq_lens_vec.size();
  
  LOG(INFO) << "dp_global_token_nums size: " << params.dp_global_token_nums.size();
  if (!params.dp_global_token_nums.empty()) {
      std::ostringstream oss;
      for (size_t i = 0; i < params.dp_global_token_nums.size() && i < 10; ++i) {
          oss << params.dp_global_token_nums[i] << " ";
      }
      if (params.dp_global_token_nums.size() > 10) oss << "...";
      LOG(INFO) << "dp_global_token_nums: [" << oss.str() << "]";
  }
  
  LOG(INFO) << "embedding_ids size: " << params.embedding_ids.size();
  LOG(INFO) << "extra_token_ids size: " << params.extra_token_ids.size();
  
  // Decode sequence range
  LOG(INFO) << "decode_seq_range: [" << params.decode_seq_range.first 
            << ", " << params.decode_seq_range.second << "]";
  
  // Tensor fields - check if defined and print basic info
  auto printTensorInfo = [](const torch::Tensor& tensor, const std::string& name) {
      if (tensor.defined()) {
          LOG(INFO) << name << " - shape: " << tensor.sizes() 
                    << ", dtype: " << tensor.dtype() 
                    << ", device: " << tensor.device();
      } else {
          LOG(INFO) << name << " - undefined";
      }
  };
  
  printTensorInfo(params.q_seq_lens, "q_seq_lens");
  printTensorInfo(params.kv_seq_lens, "kv_seq_lens");
  printTensorInfo(params.decode_q_seq_lens, "decode_q_seq_lens");
  printTensorInfo(params.decode_kv_seq_lens, "decode_kv_seq_lens");
  printTensorInfo(params.new_cache_slots, "new_cache_slots");
  printTensorInfo(params.block_tables, "block_tables");
  printTensorInfo(params.input_embedding, "input_embedding");
  printTensorInfo(params.visual_pos_masks, "visual_pos_masks");
  printTensorInfo(params.src_block_indices, "src_block_indices");
  printTensorInfo(params.dst_block_indices, "dst_block_indices");
  printTensorInfo(params.cum_sum, "cum_sum");
  printTensorInfo(params.expert_load_data, "expert_load_data");
  printTensorInfo(params.new_cache_slot_offsets, "new_cache_slot_offsets");
  printTensorInfo(params.kv_cache_start_offsets, "kv_cache_start_offsets");
  printTensorInfo(params.graph_buffer, "graph_buffer");
  printTensorInfo(params.paged_kv_indptr, "paged_kv_indptr");
  printTensorInfo(params.paged_kv_indices, "paged_kv_indices");
  printTensorInfo(params.paged_kv_last_page_len, "paged_kv_last_page_len");
  printTensorInfo(params.beam_width_tensor, "beam_width_tensor");
  printTensorInfo(params.current_round_tensor, "current_round_tensor");
  
  // Vector of tensors
  LOG(INFO) << "deep_stacks size: " << params.deep_stacks.size();
  for (size_t i = 0; i < params.deep_stacks.size(); ++i) {
      printTensorInfo(params.deep_stacks[i], "deep_stacks[" + std::to_string(i) + "]");
  }
  
  LOG(INFO) << "shared_k_caches size: " << params.shared_k_caches.size();
  LOG(INFO) << "shared_v_caches size: " << params.shared_v_caches.size();
  LOG(INFO) << "current_round_tensor_list size: " << params.current_round_tensor_list.size();
  LOG(INFO) << "decode_positions_tensor_list size: " << params.decode_positions_tensor_list.size();
  
  // Cache block info vectors
  LOG(INFO) << "async_copy_out_blocks size: " << params.async_copy_out_blocks.size();
  LOG(INFO) << "copy_out_blocks size: " << params.copy_out_blocks.size();
  LOG(INFO) << "copy_in_blocks size: " << params.copy_in_blocks.size();
  LOG(INFO) << "swap_blocks size: " << params.swap_blocks.size();
  
#if defined(USE_NPU)
  LOG(INFO) << "layer_synchronizer: " << (params.layer_synchronizer ? "defined" : "nullptr");
#endif
  
  LOG(INFO) << "=== End ModelInputParams Debug Info ===";
}


} // namespace

std::optional<ForwardOutput> LLMWorkerImpl::step(
    const BatchedForwardInputs& inputs) {
  LOG(INFO) << "inner LLMWorkerImpl::step.";
  Timer timer;
  // Only enter multi-round decode when explicitly enabled via global flag.
  LOG(INFO) << "inputs.micro_inputs.empty(): " << inputs.micro_inputs.empty();
  LOG(INFO ) << "inputs.micro_inputs[0].total_round: " << inputs.micro_inputs[0].total_round;
  if (FLAGS_max_decode_rounds > 0 && !inputs.micro_inputs.empty() &&
      inputs.micro_inputs[0].total_round > 0) {
    return step_multi_round(inputs);
  }
  int mb_num = inputs.micro_inputs.size();
  int32_t total_round = mb_num > 0 ? inputs.micro_inputs[0].total_round : 0;
  int32_t beam_width = mb_num > 0 ? inputs.micro_inputs[0].beam_width : 0;
  VLOG(1) << "[WORKER_STEP] micro_batches=" << mb_num
          << ", total_round=" << total_round << ", beam_width=" << beam_width
          << ", FLAGS_max_decode_rounds=" << FLAGS_max_decode_rounds;
  std::vector<torch::Tensor> flatten_tokens_micro_batches;
  std::vector<torch::Tensor> flatten_positions_micro_batches;
  std::vector<ModelInputParams> input_params_micro_batches;
  auto& concated_sampling_params = inputs.concated_sampling_params;

  std::vector<folly::SemiFuture<bool>> futures;

  for (auto i = 0; i < inputs.micro_inputs.size(); ++i) {
    flatten_tokens_micro_batches.push_back(
        std::move(inputs.micro_inputs[i].token_ids));
    flatten_positions_micro_batches.push_back(
        std::move(inputs.micro_inputs[i].positions));
    input_params_micro_batches.push_back(
        std::move(inputs.micro_inputs[i].input_params));

    if (options_.kv_cache_transfer_mode() == "PUSH" &&
        !inputs.micro_inputs[i].transfer_kv_infos.empty()) {
#if defined(USE_NPU)
      std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer =
          std::make_shared<NPULayerSynchronizerImpl>(
              context_.get_model_args().n_layers());
      const_cast<ModelInputParams*>(&(input_params_micro_batches[i]))
          ->layer_synchronizer = layer_synchronizer;

      futures.emplace_back(kv_cache_transfer_->push_kv_blocks_async(
          inputs.micro_inputs[i].transfer_kv_infos,
          context_.get_parallel_args(),
          layer_synchronizer,
          is_spec_draft_));
#endif
    }
  }
  if (FLAGS_enable_eplb) {
    eplb_executor_->eplb_execute(inputs.micro_inputs[0].eplb_info);
  }

  // temporarily use [0], will be adapted in next pr
  // call model executor forward to get hidden states
  LOG(INFO) << "input_params_micro_batches.size(): " << 
    input_params_micro_batches.size();
  for (const auto& input_param : input_params_micro_batches) {
    printModelInputParams(input_param);
  }
  LOG(INFO) << "flatten_tokens_micro_batches.size: " << flatten_tokens_micro_batches.size();
  for (const auto& flatten_token : flatten_tokens_micro_batches) {
    LOG(INFO) << flatten_token.sizes();
  }
  LOG(INFO) << "flatten_positions_micro_batches.size: " << flatten_positions_micro_batches.size();
  for (const auto& flatten_position : flatten_positions_micro_batches) {
    LOG(INFO) << flatten_position.sizes();
  }
  LOG(INFO) << "kv_caches_.size(): " << kv_caches_.size();
  for (const auto& kv_cache : kv_caches_) {
    LOG(INFO) << kv_cache.get_k_cache().sizes();
    LOG(INFO) << kv_cache.get_v_cache().sizes();
  }
  auto hidden_states = model_executor_->forward(flatten_tokens_micro_batches,
                                                flatten_positions_micro_batches,
                                                kv_caches_,
                                                input_params_micro_batches);
  if (!hidden_states.defined()) {
    return std::nullopt;
  }

  torch::Tensor logits;
  if (concated_sampling_params.selected_token_idxes.defined()) {
    logits = model_->logits(hidden_states,
                            concated_sampling_params.selected_token_idxes);
  }

  ForwardOutput output;
  if (FLAGS_enable_eplb) {
    output.expert_load_data = expert_load_data_;
    output.prepared_layer_id = eplb_executor_->get_ready_layer_id();
    if (output.prepared_layer_id != -1) {
      eplb_executor_->reset_ready_layer_id();
    }
  }

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_ &&
      !options_.enable_speculative_decode()) {
    auto ret = device_.synchronize_default_stream();
    // in p-d disaggregation scene, all micro batches should be in same
    // prefill/decode stage, so, to judge transfer_kv_infos.empty,
    // just use micro inputs.micro_inputs[0] here
    if (options_.kv_cache_transfer_mode() == "PUSH" &&
        !inputs.micro_inputs[0].transfer_kv_infos.empty()) {
      auto results =
          folly::collectAll(futures).within(std::chrono::seconds(60)).get();
      for (const auto& result : results) {
        if (!result.value()) {
          LOG(ERROR) << "kv_cache_transfer_ failed";
          return std::nullopt;
        }
      }
    }
    if (FLAGS_enable_eplb) {
      return output;
    }
    return std::nullopt;
  }

  // driver prepare model output
  SampleOutput sample_output;
  if (concated_sampling_params.selected_token_idxes.defined()) {
    sample_output = sampler_->forward(logits, concated_sampling_params);
    output.logits = logits;

    // beam search kernel
    BeamSearchOutput beam_search_output;
    if (concated_sampling_params.use_beam_search &&
        inputs.acc_logprob.defined() && inputs.acc_logprob.numel() > 0) {
      beam_search_output = beam_searcher_->forward(inputs.acc_logprob,
                                                   sample_output.top_tokens,
                                                   sample_output.top_logprobs);
    }

    // set sample output to output
    output.sample_output = sample_output;
    // carry over the sampling params
    output.do_sample = concated_sampling_params.do_sample;
    output.logprobs = concated_sampling_params.logprobs;
    output.max_top_logprobs = concated_sampling_params.max_top_logprobs;
    // set beam search output to output
    output.beam_search_output = beam_search_output;

    LOG(INFO) << "out_tokens.sizes(): " << output.beam_search_output.out_tokens.sizes();
  }

  // if running in multi_stream_parallel step, all micro batches
  // should be in same prefill stage, so, to judge empty_kv_cache,
  // just use micro batch 0 here
  if (options_.enable_speculative_decode() && !is_spec_draft_) {
    if (input_params_micro_batches[0].q_seq_lens_vec[0] > 1) {
      output.sample_output.embeddings = hidden_states;
    } else if (concated_sampling_params.sample_idxes.defined()) {
      // auto sample_idxes =
      //     concated_sampling_params.selected_token_idxes.index_select(
      //         /*dim=*/0, concated_sampling_params.sample_idxes);
      auto embeddings = hidden_states.index_select(
          /*dim=*/0, concated_sampling_params.sample_idxes);
      output.sample_output.embeddings = embeddings;
    }
  }

  // if running in multi_stream_parallel step, all micro batches
  // should be in same prefill stage, so, to judge empty_kv_cache,
  // just use micro batch 0 here
  if (options_.enable_speculative_decode() && !is_spec_draft_) {
    if (input_params_micro_batches[0].q_seq_lens_vec[0] > 1) {
      output.sample_output.embeddings = hidden_states;
    } else if (concated_sampling_params.sample_idxes.defined()) {
      // auto sample_idxes =
      //     concated_sampling_params.selected_token_idxes.index_select(
      //         /*dim=*/0, concated_sampling_params.sample_idxes);
      auto embeddings = hidden_states.index_select(
          /*dim=*/0, concated_sampling_params.sample_idxes);
      output.sample_output.embeddings = embeddings;
    }
  }

  auto ret = device_.synchronize_default_stream();

  if (options_.kv_cache_transfer_mode() == "PUSH" &&
      !inputs.micro_inputs[0].transfer_kv_infos.empty()) {
    auto results =
        folly::collectAll(futures).within(std::chrono::seconds(60)).get();
    for (const auto& result : results) {
      if (!result.value()) {
        LOG(ERROR) << "kv_cache_transfer_ failed";
        return std::nullopt;
      }
    }
  }

  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  DeviceMonitor::get_instance().update_active_activation_memory(
      device_.index());

  return output;
}
// shared是从inputs里面拿的
// unshared是从Kv_cache拿的
std::optional<ForwardOutput> LLMWorkerImpl::step_multi_round(
    const BatchedForwardInputs& inputs) {
  LOG(INFO) << "inner LLMWorkerImpl::step_multi_round.";
  device_.set_device();
  Timer timer;
  std::vector<torch::Tensor> flatten_tokens_micro_batches;
  std::vector<torch::Tensor> flatten_positions_micro_batches;
  std::vector<ModelInputParams> input_params_micro_batches;
  std::vector<folly::SemiFuture<bool>> futures;

  for (auto i = 0; i < inputs.micro_inputs.size(); ++i) {
    flatten_tokens_micro_batches.push_back(
        std::move(inputs.micro_inputs[i].token_ids));
    flatten_positions_micro_batches.push_back(
        std::move(inputs.micro_inputs[i].positions));
    input_params_micro_batches.push_back(
        std::move(inputs.micro_inputs[i].input_params));
  }

  int32_t total_rounds = inputs.micro_inputs[0].total_round;
  ForwardOutput output;

  std::vector<torch::Tensor> unshared_k_cache;
  std::vector<torch::Tensor> unshared_v_cache;
  auto args = context_.get_model_args();
  int32_t layer_num = static_cast<int32_t>(args.n_layers());
  for (auto i = 0; i < layer_num; ++i) {
    unshared_k_cache.push_back(kv_caches_[i].get_k_cache());
    unshared_v_cache.push_back(kv_caches_[i].get_v_cache());
  }
  int32_t batch = input_params_micro_batches.empty()
                      ? 0
                      : input_params_micro_batches[0].num_sequences;
  int32_t beam_width_init = inputs.micro_inputs[0].beam_width;
  auto int_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device_);
  auto fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device_);
  torch::Tensor sequence_group =
      torch::zeros({batch, beam_width_init, total_rounds}, int_options);
  // preallocate outputs and cached inputs
  int64_t num_seq = batch * beam_width_init;
  torch::Tensor acc_logprob = torch::empty({num_seq, 1}, fp32_options);
  torch::Tensor out_log_probs = torch::empty({num_seq, 1}, fp32_options);
  torch::Tensor out_token_ids = torch::empty({num_seq, 1}, int_options);
  torch::Tensor out_token_index = torch::empty({num_seq, 1}, int_options);
  torch::Tensor out_beam_count_prefix_sums =
      torch::empty({num_seq, 1}, int_options);
  auto out_seqgroup = sequence_group.clone();
  for (int32_t round = 0; round < total_rounds; ++round) {
    LOG(INFO) << "round: " << round;
    const auto& concated_sampling_params =
        round > 0 ? inputs.concated_decoder_sampling_params
                  : inputs.concated_sampling_params;
    for (auto i = 0; i < input_params_micro_batches.size(); ++i) {
      auto& mip = input_params_micro_batches[i];
      mip.is_prefill = round == 0;
      if (!mip.current_round_tensor_list.empty() && round >= 0 &&
          round < static_cast<int32_t>(mip.current_round_tensor_list.size())) {
        mip.current_round_tensor = mip.current_round_tensor_list[round];
        mip.current_round = round;
      }
      mip.beam_width = inputs.micro_inputs[0].beam_width;
    }
    LOG(INFO) << "before model_executor_->forward.";
    LOG(INFO) << "input_params_micro_batches.size(): " << 
    input_params_micro_batches.size();
    for (const auto& input_param : input_params_micro_batches) {
      printModelInputParams(input_param);
    }
    LOG(INFO) << "flatten_tokens_micro_batches.size: " << flatten_tokens_micro_batches.size();
    for (const auto& flatten_token : flatten_tokens_micro_batches) {
      LOG(INFO) << flatten_token.sizes();
    }
    LOG(INFO) << "flatten_positions_micro_batches.size: " << flatten_positions_micro_batches.size();
    for (const auto& flatten_position : flatten_positions_micro_batches) {
      LOG(INFO) << flatten_position.sizes();
    }
    LOG(INFO) << "kv_caches_.size(): " << kv_caches_.size();
    for (const auto& kv_cache : kv_caches_) {
      LOG(INFO) << kv_cache.get_k_cache().sizes();
      LOG(INFO) << kv_cache.get_v_cache().sizes();
    }
    auto hidden_states =
        model_executor_->forward(flatten_tokens_micro_batches,
                                 flatten_positions_micro_batches,
                                 kv_caches_,
                                 input_params_micro_batches);
    LOG(INFO) << "after model_executor_->forward.";
    if (!hidden_states.defined()) {
      return std::nullopt;
    }


    // 验证triton kernel调用逻辑
    // -----------------------
    // 初始化tensor
    {
      auto device_options = 
        torch::TensorOptions().device(device_);
      auto bf16_options =
        torch::TensorOptions().dtype(torch::kBFloat16).device(device_);
      auto int32_options =
        torch::TensorOptions().dtype(torch::kInt32).device(device_);
      
      uint32_t batch_size = 1;
      uint32_t beam_width = 64;
      uint32_t prompt_len = 1024;
      uint32_t num_heads = 8;
      uint32_t kv_heads = 8;
      uint32_t head_dim = 128;

      uint32_t total_beams = batch_size * beam_width;
      auto q = torch::randn({total_beams, num_heads, head_dim}, bf16_options);

      uint32_t shared_len = prompt_len;
      auto shared_k_cache = torch::randn({batch_size, kv_heads, shared_len, head_dim}, bf16_options);  
      auto shared_v_cache = torch::randn({batch_size, kv_heads, shared_len, head_dim}, bf16_options); 
      
      uint32_t max_decode_step = 2;
      auto unshared_k_cache = torch::randn({total_beams, kv_heads, max_decode_step, head_dim}, bf16_options); 
      auto unshared_v_cache = torch::randn({total_beams, kv_heads, max_decode_step, head_dim}, bf16_options);  
      
      int decode_step = 1;  // current decode step
      // int beam_size = 16;   // beam size

      float sm_scale = 0.08838834764831843;  // 通常是 1/sqrt(head_dim)
      bool warp_specialize = false;

      // LOG(INFO) << "q.sizes(): " << q.sizes();
      // LOG(INFO) << "shared_k_cache.sizes(): " << shared_k_cache.sizes();
      // LOG(INFO) << "shared_v_cache.sizes(): " << shared_v_cache.sizes();
      // LOG(INFO) << "unshared_k_cache.sizes(): " << unshared_k_cache.sizes();
      // LOG(INFO) << "unshared_v_cache.sizes(): " << unshared_v_cache.sizes();
      // LOG(INFO) << "shared_len: " << shared_len;
      // LOG(INFO) << "decode_step: " << decode_step;
      // LOG(INFO) << "beam_size: " << beam_size;
      rec_triton_kernel_.xattention(q, 
                                    shared_k_cache,
                                    shared_v_cache, 
                                    unshared_k_cache, 
                                    unshared_v_cache, 
                                    shared_len,
                                    decode_step, 
                                    beam_width, 
                                    sm_scale, 
                                    warp_specialize, 
                                    prompt_len);
    }
    

    // -----------------------


    torch::Tensor logits;
    LOG(INFO) << "before model_->logits.";
    if (concated_sampling_params.selected_token_idxes.defined()) {
      logits = model_->logits(hidden_states,
                              concated_sampling_params.selected_token_idxes);
    }
    LOG(INFO) << "after model_->logits.";
    LOG(INFO) << "before sampler_->forward.";
    if (concated_sampling_params.selected_token_idxes.defined()) {
      auto sample_output = sampler_->forward(logits, concated_sampling_params);
      LOG(INFO) << "after sampler_->forward.";
      torch::Tensor top_tokens;
      torch::Tensor top_logprobs;
      int32_t beam_width = inputs.micro_inputs[0].beam_width;

      if (round == 0) {
        top_tokens =
            sample_output.top_tokens.to(torch::kInt32).reshape({-1, 1});
        top_logprobs = sample_output.top_logprobs.reshape({-1, 1});
      } else {
        top_tokens = sample_output.top_tokens.to(torch::kInt32)
                         .reshape({-1, beam_width});
        top_logprobs = sample_output.top_logprobs.reshape({-1, beam_width});
      }

      // xllm_ops::beam_search(acc_logprob,
      //                       top_tokens,
      //                       top_logprobs,
      //                       sequence_group,
      //                       round,
      //                       out_token_ids,
      //                       out_token_index,
      //                       out_log_probs,
      //                       out_beam_count_prefix_sums,
      //                       out_seqgroup);
      #if defined(USE_CUDA)
      LOG(INFO) << "before rec_triton_kernel_.beam_search.";
      acc_logprob = acc_logprob.to(torch::kBFloat16);
      rec_triton_kernel_.beam_search(acc_logprob, 
                                     sequence_group, 
                                     top_tokens, 
                                     top_logprobs, 
                                     out_log_probs, 
                                     out_token_ids, 
                                     out_token_index, 
                                     out_beam_count_prefix_sums, 
                                     out_seqgroup, 
                                     total_rounds, 
                                     round);
      #endif
      sequence_group.copy_(out_seqgroup);
      acc_logprob.copy_(out_log_probs);
      // keep group offset contiguous across rounds (already in out_* tensors)
      // update next round tokens.
      if (round == 0) {
        flatten_tokens_micro_batches[0] =
            sample_output.top_tokens.to(torch::kInt32).reshape({-1});
      } else {
        flatten_tokens_micro_batches[0] = out_token_ids.clone().reshape({-1});
      }

      // update next round positions.
      flatten_positions_micro_batches.clear();
      for (auto i = 0; i < inputs.micro_inputs.size(); ++i) {
        auto& mip = input_params_micro_batches[i];
        if (!mip.decode_positions_tensor_list.empty() && round >= 0 &&
            round <
                static_cast<int32_t>(mip.decode_positions_tensor_list.size())) {
          flatten_positions_micro_batches.push_back(
              mip.decode_positions_tensor_list[round]);
        }
      }
      // update output at the last round.
      if (round == total_rounds - 1) {
        output.logits = logits;
        output.sample_output = sample_output;
        output.do_sample = concated_sampling_params.do_sample;
        output.logprobs = concated_sampling_params.logprobs;
        output.max_top_logprobs = concated_sampling_params.max_top_logprobs;
        output.beam_search_output.src_seq_idxes = out_token_index.reshape({-1});
        output.beam_search_output.out_tokens = out_token_ids.reshape({-1});
        output.beam_search_output.out_logprobs = out_log_probs.reshape({-1});
        output.beam_search_output.group_offset =
            out_beam_count_prefix_sums.reshape({-1});
        output.beam_sequence_group = sequence_group;
      }

#if defined(USE_CUDA)
      // if (beam_width > 1 && round > 0) {
      //   rec_triton_kernel_.grouped_cache_select(out_token_index,
      //                                           unshared_k_cache,
      //                                           unshared_v_cache,
      //                                           inputs.concated_block_tables,
      //                                           round);
      // }
#endif

#if defined(USE_NPU)
      if (beam_width > 1 && round > 0) {
        xllm_ops::cache_select(out_token_index,
                               unshared_k_cache,
                               unshared_v_cache,
                               inputs.concated_block_tables,
                               out_beam_count_prefix_sums,
                               round,
                               beam_width,
                               layer_num);
      }
#endif
    }
  }

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_ &&
      !options_.enable_speculative_decode()) {
    auto ret = device_.synchronize_default_stream();
    if (options_.instance_role() == InstanceRole::PREFILL &&
        options_.kv_cache_transfer_mode() == "PUSH" &&
        !inputs.micro_inputs[0].transfer_kv_infos.empty()) {
      auto results =
          folly::collectAll(futures).within(std::chrono::seconds(60)).get();
      for (const auto& result : results) {
        if (!result.value()) {
          return std::nullopt;
        }
      }
    }
    if (FLAGS_enable_eplb) {
      return output;
    }
    return std::nullopt;
  }

  auto ret = device_.synchronize_default_stream();
  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  DeviceMonitor::get_instance().update_active_activation_memory(
      device_.index());
  return output;
}

}  // namespace xllm
