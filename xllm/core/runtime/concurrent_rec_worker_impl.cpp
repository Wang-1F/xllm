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

#include "concurrent_rec_worker_impl.h"

#include <c10/core/DeviceGuard.h>
#include <c10/core/StreamGuard.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <thread>
#include <unordered_map>
#include <utility>

#include "common/device_monitor.h"
#include "common/metrics.h"
#include "common/rec_model_utils.h"
#include "common/types.h"
#include "core/common/global_flags.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/model_loader.h"
#include "framework/state_dict/state_dict.h"
#if defined(USE_CUDA) || defined(USE_ILU)
#include "kernels/cuda/cuda_ops_api.h"
#include "layers/cuda/flashinfer_workspace.h"
#endif
#if defined(USE_NPU)
#include "kernels/npu/npu_ops_api.h"
#endif
#include "models/model_registry.h"
#include "util/threadpool.h"
#include "util/timer.h"

namespace xllm {

ConcurrentRecWorkerImpl::ConcurrentRecWorkerImpl(
    const ParallelArgs& parallel_args,
    const torch::Device& device,
    const runtime::Options& options)
    : RecWorkerImpl(parallel_args, device, options),
      max_concurrency_(FLAGS_rec_worker_max_concurrency) {
  CHECK_GT(max_concurrency_, 0)
      << "rec_worker_max_concurrency must be greater than 0";
  // Create independent step_threadpool_ dedicated to parallel execution of
  // step() Use schedule() to assign tasks, letting ThreadPool automatically
  // select idle threads
  step_threadpool_ = std::make_unique<ThreadPool>(
      max_concurrency_, [this]() mutable { device_.set_device(); });

  LOG(INFO) << "ConcurrentRecWorkerImpl: Created step_threadpool_ with "
            << max_concurrency_ << " threads for parallel step execution";
}

bool ConcurrentRecWorkerImpl::init_model(ModelContext& context) {
  CHECK(model_ == nullptr) << "Model is already initialized.";

  // Determine rec model kind and pipeline type
  const auto& model_type = context.get_model_args().model_type();
  rec_model_kind_ = get_rec_model_kind(model_type);
  CHECK(rec_model_kind_ != RecModelKind::kNone)
      << "Unsupported rec model_type: " << model_type;

  // Create concurrent pipeline (not base class pipeline)
  auto pipeline_type = get_rec_pipeline_type(rec_model_kind_);
  work_pipeline_ = create_concurrent_pipeline(pipeline_type, *this);

  // Reserve space for model instances
  model_instances_.reserve(max_concurrency_);
  executor_instances_.reserve(max_concurrency_);
  execute_streams_.reserve(max_concurrency_);
  context_instances_.reserve(max_concurrency_);

  // Create additional model instances
  for (int32_t i = 0; i < max_concurrency_; ++i) {
    // Create corresponding execute stream
    auto stream = device_.get_stream_from_pool();
    execute_streams_.push_back(std::move(stream));

    auto stream_guard = execute_streams_[i]->set_stream_guard();
    // Create independent ModelContext for each model instance
    ModelContext instance_context(context.get_parallel_args(),
                                  context.get_model_args(),
                                  context.get_quant_args(),
                                  context.get_tensor_options());
    context_instances_.push_back(std::move(instance_context));

    // Create model instance using the corresponding context
    auto model_instance = create_llm_model(context_instances_[i]);
    CHECK(model_instance != nullptr) << "Failed to create model instance " << i;
    model_instances_.push_back(std::move(model_instance));

    // Create corresponding executor using the corresponding context
    auto executor =
        std::make_unique<Executor>(model_instances_[i].get(),
                                   context_instances_[i].get_model_args(),
                                   device_,
                                   options_);
    executor_instances_.push_back(std::move(executor));

    LOG(INFO) << "Created model instance " << i
              << " with executor, execute stream and context";
  }

  // For compatibility with base class interface, set base class's model_ and
  // model_executor_ to point to the first instance Note: Need to access base
  // class's protected members model_ and model_executor_ Use reset() to set
  // pointers, but note: ownership of model_ actually belongs to
  // model_instances_[0]. In destructor, need to release model_ and
  // model_executor_ first to avoid double deletion
  model_.reset(model_instances_[0].get());
  model_executor_.reset(executor_instances_[0].get());

  // Complete other initialization (EPLB, BeamSearcher, etc.)
  if (FLAGS_enable_eplb) {
    eplb_executor_ = std::make_unique<EplbExecutor>(model_.get(), device_);
  }

  if (FLAGS_enable_beam_search_kernel) {
    beam_searcher_ = std::make_unique<BeamSearcher>();
  }

  LOG(INFO) << "Created " << model_instances_.size()
            << " model instances for concurrent execution";
  return true;
}

void ConcurrentRecWorkerImpl::load_model(std::unique_ptr<ModelLoader> loader) {
  CHECK(!model_instances_.empty())
      << "Model instances are not initialized. Call init_model() first.";

  // Save model weights path to create new loaders for other instances
  std::string model_weights_path = loader->model_weights_path();

  // Load weights for the first model instance (using the original loader)
  model_instances_[0]->load_model(std::move(loader));
  LOG(INFO) << "Loaded weights for model instance 0";

  // Create new loaders and load weights for other model instances
  for (size_t i = 1; i < model_instances_.size(); ++i) {
    auto model_loader = ModelLoader::create(model_weights_path);
    CHECK(model_loader != nullptr)
        << "Failed to create ModelLoader for model instance " << i;
    model_instances_[i]->load_model(std::move(model_loader));
    LOG(INFO) << "Loaded weights for model instance " << i;
  }

  LOG(INFO) << "Loaded weights for all " << model_instances_.size()
            << " model instances";
}

void ConcurrentRecWorkerImpl::allocate_instance_id_for_current_thread() {
  std::thread::id current_thread_id = std::this_thread::get_id();

  // Lock to protect the allocation process
  std::lock_guard<std::mutex> lock(allocation_mutex_);

  // Check if current thread is already in the map (may have been allocated by
  // other tasks)
  auto it = thread_id_to_instance_id_.find(current_thread_id);
  if (it != thread_id_to_instance_id_.end()) {
    return;
  }

  // Select the smallest unallocated instance id
  size_t instance_id = SIZE_MAX;
  size_t stream_num = static_cast<size_t>(max_concurrency_);
  for (size_t i = 0; i < stream_num; ++i) {
    if (allocated_instance_ids_.find(i) == allocated_instance_ids_.end()) {
      instance_id = i;
      break;
    }
  }

  CHECK_NE(instance_id, SIZE_MAX)
      << "No available instance id, all " << max_concurrency_
      << " instance ids are allocated";

  // Establish mapping relationship
  thread_id_to_instance_id_[current_thread_id] = instance_id;
  allocated_instance_ids_.insert(instance_id);

  LOG(INFO) << "Allocated instance_id " << instance_id << " for thread "
            << current_thread_id;
}

void ConcurrentRecWorkerImpl::get_thread_model_instance(
    CausalLM*& model,
    Executor*& executor,
    Stream*& execute_stream,
    ModelContext*& context) {
  std::thread::id current_thread_id = std::this_thread::get_id();

  // If current thread hasn't been allocated an instance id yet, allocate it
  // first
  auto it = thread_id_to_instance_id_.find(current_thread_id);
  if (it == thread_id_to_instance_id_.end()) {
    allocate_instance_id_for_current_thread();
    it = thread_id_to_instance_id_.find(current_thread_id);
  }

  CHECK(it != thread_id_to_instance_id_.end())
      << "Failed to find instance id for thread " << current_thread_id;
  size_t instance_id = it->second;
  // LOG(INFO) << "get_thread_model_instance: thread " << current_thread_id
  // << " allocated instance_id " << instance_id;

  CHECK_LT(instance_id, model_instances_.size())
      << "Thread model index " << instance_id
      << " exceeds model instances size " << model_instances_.size();

  model = model_instances_[instance_id].get();
  executor = executor_instances_[instance_id].get();
  execute_stream = execute_streams_[instance_id].get();
  context = &context_instances_[instance_id];
}

folly::SemiFuture<std::optional<ForwardOutput>>
ConcurrentRecWorkerImpl::step_async(const ForwardInput& input) {
  ForwardInput input_on_device;
  prepare_work_before_execute(input, input_on_device);

  folly::Promise<std::optional<ForwardOutput>> promise;
  auto future = promise.getSemiFuture();

  // Use schedule() to assign tasks, letting ThreadPool automatically select
  // idle threads The logic for allocating instance_id happens when the task
  // executes (see lambda below)
  step_threadpool_->schedule([this,
                              input = std::move(input_on_device),
                              promise = std::move(promise)]() mutable {
    // When the task executes, if the current thread hasn't been allocated an
    // instance id yet, allocate it The allocation logic will lock, select the
    // smallest unallocated instance id, establish a mapping from thread id to
    // instance id Once allocation is complete, the mapping relationship is
    // saved in thread_id_to_instance_id_. This way, multiple threads complete
    // allocation after executing once

    // Handle hierarchy_kv_cache_transfer if needed (from base class logic)
    if (hierarchy_kv_cache_transfer_ != nullptr) {
      hierarchy_kv_cache_transfer_->set_layer_synchronizer(input.input_params);
    }

    // Call step() using the model instance corresponding to the current thread
    const auto output = this->step(input);

    // Handle enable_schedule_overlap logic (if needed)
    if (!enable_schedule_overlap()) {
      promise.setValue(output);
    } else {
      if (last_step_output_valid_ && !input.input_params.empty_kv_cache) {
        // replace step i model input with true output of step i-1
        input = update_input_by_last_step_output(input);
      }

      const auto output_overlap = this->step(input);
      if (output_overlap.has_value()) {
        if (is_driver() || FLAGS_enable_eplb) {
          std::unique_lock<std::mutex> lock(mtx_);
          cv_.wait(lock, [this] { return !is_recorded_; });
          update_last_step_output(output_overlap);
          is_recorded_ = true;
          cv_.notify_one();
        } else {
          update_last_step_output(output_overlap);
        }
      } else {
        if (is_driver() || FLAGS_enable_eplb) {
          std::unique_lock<std::mutex> lock(mtx_);
          cv_.wait(lock, [this] { return !is_recorded_; });
          last_step_output_valid_ = false;
          is_recorded_ = true;
          cv_.notify_one();
        } else {
          last_step_output_valid_ = false;
        }
      }
      promise.setValue(output_overlap);
    }
  });
  return future;
}

std::optional<ForwardOutput> ConcurrentRecWorkerImpl::step(
    const ForwardInput& input) {
  CHECK(work_pipeline_ != nullptr)
      << "ConcurrentRecWorkerImpl is not initialized.";
  // Pipeline's step method will use thread-specific instances
  return work_pipeline_->step(input);
}

// ============================================================
// Concurrent Pipeline Implementation
// ============================================================

std::optional<ForwardOutput>
ConcurrentRecWorkerImpl::ConcurrentLlmRecPureDevicePipeline::step(
    const ForwardInput& input) {
  return step_multi_round(const_cast<ForwardInput&>(input));
}

std::optional<ForwardOutput>
ConcurrentRecWorkerImpl::ConcurrentLlmRecPureDevicePipeline::step_multi_round(
    ForwardInput& input) {
  // Get thread-specific instances
  CausalLM* model = nullptr;
  Executor* executor = nullptr;
  Stream* execute_stream = nullptr;
  ModelContext* context = nullptr;
  concurrent_worker_.get_thread_model_instance(
      model, executor, execute_stream, context);

  // Set stream guard
  c10::StreamGuard stream_guard = execute_stream->set_stream_guard();
  auto dtype = concurrent_worker_.dtype();
  auto device = concurrent_worker_.device_;
  device.set_device();

  Timer timer;

  int32_t total_rounds = input.total_round;
  int32_t max_decode_step = total_rounds - 1;

  int32_t batch_size = input.input_params.paged_kv_last_page_len.numel();
  int32_t beam_width = input.beam_width;

  auto args = context->get_model_args();
  int32_t layer_num = static_cast<int32_t>(args.n_layers());
  int64_t num_qo_heads = context->get_model_args().n_heads();
  int64_t head_dim = context->get_model_args().head_dim();

  input.input_params.num_heads = num_qo_heads;
  input.input_params.head_dim = head_dim;
  input.input_params.beam_width = beam_width;

  auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(device);
  auto fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);

  torch::Tensor sequence_group =
      torch::zeros({batch_size, beam_width, total_rounds}, int_options);

  int64_t num_seq = batch_size * beam_width;
  torch::Tensor acc_logprob = torch::zeros({num_seq, 1}, fp32_options);
  torch::Tensor out_log_probs = torch::zeros({num_seq, 1}, fp32_options);
  torch::Tensor out_token_ids = torch::zeros({num_seq, 1}, int_options);
  torch::Tensor out_token_index = torch::zeros({num_seq, 1}, int_options);
  torch::Tensor out_beam_count_prefix_sums =
      torch::zeros({num_seq, 1}, int_options);
  auto out_seqgroup = sequence_group.clone();

  ForwardOutput output;

  for (int32_t round = 0; round < total_rounds; ++round) {
    const auto& sampling_params =
        round > 0 ? input.decoder_sampling_params : input.sampling_params;
    input.input_params.is_prefill = round == 0;

    if (!input.input_params.current_round_tensor_list.empty() && round >= 0 &&
        round < static_cast<int32_t>(
                    input.input_params.current_round_tensor_list.size())) {
      input.input_params.current_round_tensor =
          input.input_params.current_round_tensor_list[round];

      input.input_params.current_round = round - 1;
    }

    auto hidden_states = executor->forward(input.token_ids,
                                           input.positions,
                                           concurrent_worker_.kv_caches_,
                                           input.input_params);
    if (!hidden_states.defined()) {
      return std::nullopt;
    }

    torch::Tensor logits;
    if (sampling_params.selected_token_idxes.defined()) {
      logits =
          model->logits(hidden_states, sampling_params.selected_token_idxes);
    }

    if (sampling_params.selected_token_idxes.defined()) {
      auto sample_output =
          concurrent_worker_.sampler_->forward(logits, sampling_params);
      torch::Tensor top_tokens =
          sample_output.top_tokens.to(torch::kInt32).reshape({-1, beam_width});
      torch::Tensor top_logprobs =
          sample_output.top_logprobs.reshape({-1, beam_width});

#if defined(USE_NPU)
      xllm_ops::beam_search(acc_logprob,
                            top_tokens,
                            top_logprobs,
                            sequence_group,
                            round,
                            out_token_ids,
                            out_token_index,
                            out_log_probs,
                            out_beam_count_prefix_sums,
                            out_seqgroup);
#elif defined(USE_CUDA)
      xllm::kernel::cuda::beam_search(acc_logprob,
                                      sequence_group,
                                      top_tokens,
                                      top_logprobs,
                                      out_log_probs,
                                      out_token_ids,
                                      out_token_index,
                                      out_beam_count_prefix_sums,
                                      out_seqgroup,
                                      batch_size,
                                      round);

#endif
      sequence_group.copy_(out_seqgroup);
      acc_logprob.copy_(out_log_probs);

      if (round < total_rounds - 1) {
        update_input_for_next_round(input,
                                    round,
                                    sample_output,
                                    out_token_ids,
                                    batch_size,
                                    beam_width,
                                    max_decode_step);
        if (round > 0) {
#if defined(USE_NPU)
          xllm_ops::cache_select(out_token_index,
                                 input.input_params.unshared_k_caches,
                                 input.input_params.unshared_v_caches,
                                 input.input_params.block_tables,
                                 out_beam_count_prefix_sums,
                                 round,
                                 beam_width,
                                 layer_num);
#elif defined(USE_CUDA)
          xllm::kernel::cuda::cache_select(out_token_index,
                                           input.input_params.unshared_k_caches,
                                           input.input_params.unshared_v_caches,
                                           input.input_params.naive_block_table,
                                           out_beam_count_prefix_sums,
                                           round - 1,  // 对应第0步decode
                                           beam_width,
                                           layer_num);
#endif
        }
      }

      // update output at the last round.
      if (round == total_rounds - 1) {
        output.logits = logits;
        output.sample_output = sample_output;
        output.do_sample = sampling_params.do_sample;
        output.logprobs = sampling_params.logprobs;
        output.max_top_logprobs = sampling_params.max_top_logprobs;
        output.beam_search_output.src_seq_idxes = out_token_index.reshape({-1});
        output.beam_search_output.out_tokens = out_token_ids.reshape({-1});
        output.beam_search_output.out_logprobs = out_log_probs.reshape({-1});
        output.beam_sequence_group = sequence_group;
      }
    }
  }

  if (execute_stream != nullptr) {
    execute_stream->synchronize();
  } else {
    device.synchronize_default_stream();
  }
  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  DeviceMonitor::get_instance().update_active_activation_memory(device.index());
  return output;
}

// Factory method to create concurrent pipeline
std::unique_ptr<RecWorkerImpl::RecWorkPipeline>
ConcurrentRecWorkerImpl::create_concurrent_pipeline(
    RecPipelineType type,
    ConcurrentRecWorkerImpl& worker) {
  // Only kLlmRecPureDevicePipeline uses Concurrent pipeline
  // Other pipeline types use base class pipelines
  if (type == RecPipelineType::kLlmRecPureDevicePipeline) {
    return std::make_unique<ConcurrentLlmRecPureDevicePipeline>(worker);
  }
  // For other pipeline types, use base class pipeline factory
  return RecWorkerImpl::create_pipeline(type, worker);
}

void ConcurrentRecWorkerImpl::update_last_step_output(
    const std::optional<ForwardOutput>& output) {
  // Implement the same logic as the base class because the base class's method
  // is private
  if (output.has_value()) {
    if (output.value().sample_output.next_tokens.defined()) {
      last_step_output_ = std::move(output.value());
      last_step_output_valid_ = true;
    } else {
      if (FLAGS_enable_eplb) {
        last_step_output_ = std::move(output.value());
      }
      last_step_output_valid_ = false;
    }
  } else {
    last_step_output_valid_ = false;
  }
}

}  // namespace xllm
