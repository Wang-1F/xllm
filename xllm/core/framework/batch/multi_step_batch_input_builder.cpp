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

#include "multi_step_batch_input_builder.h"

#include <c10/core/DeviceType.h>
#include <torch/torch.h>

#include <thread>
#include <unordered_set>
#include <vector>

#include "common/global_flags.h"
#include "common/metrics.h"
#include "framework/batch/mposition.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/request/sequence.h"
#include "framework/sampling/sampling_params.h"
#include "runtime/params_utils.h"
#include "util/blocking_counter.h"
#include "util/slice.h"
#include "util/tensor_helper.h"
#include "util/threadpool.h"
#include "util/utils.h"

namespace xllm {

MultiStepBatchInputBuilder::MultiStepBatchInputBuilder(
    const std::vector<Sequence*>& sequences,
    const std::vector<uint32_t>& allowed_max_tokens,
    const std::vector<torch::Tensor>& input_embeddings_vec,
    const std::vector<MMData>& mm_data_vec,
    const std::vector<CacheBlockInfo>* copy_in_cache_block_infos,
    const std::vector<CacheBlockInfo>* copy_out_cache_block_infos,
    std::vector<CacheBlockInfo>* swap_cache_block_infos,
    const ModelArgs* args,
    ThreadPool* thread_pool)
    : BatchInputBuilder(sequences,
                        allowed_max_tokens,
                        input_embeddings_vec,
                        mm_data_vec,
                        copy_in_cache_block_infos,
                        copy_out_cache_block_infos,
                        swap_cache_block_infos,
                        args,
                        thread_pool) {
  // Initialize MultiStep specific state
  multi_step_state_.total_steps = FLAGS_max_decode_rounds;
  // multi_step_state_.step_tokens_vec.reserve(1000);
  // multi_step_state_.step_positions_vec.reserve(1000);
  // TODO: Add multi-step specific initialization
}

void MultiStepBatchInputBuilder::process_single_sequence(
    int32_t seq_index,
    BatchInputBuilder::BuilderState* state_ptr,
    std::unordered_set<int32_t>* write_block_ids_ptr) {
  MultiStepBuilderState& state = multi_step_state_;
  BatchInputBuilder::BuilderState& base_state = state.base_state;

  auto* sequence = sequences_[seq_index];
  const auto token_ids = sequence->tokens();
  const uint32_t n_tokens = token_ids.size();
  const uint32_t n_kv_cache_tokens = sequence->kv_state().kv_cache_tokens_num();

  // Validate and calculate sequence lengths
  CHECK(allowed_max_tokens_[seq_index] > 0);
  uint32_t q_seq_len =
      std::min(n_tokens - n_kv_cache_tokens, allowed_max_tokens_[seq_index]);
  uint32_t seq_len = q_seq_len + n_kv_cache_tokens;

  // add decode data;
  uint32_t decode_q_seq_len = 1;
  uint32_t decode_seq_len = n_kv_cache_tokens + 1;

  // Validation
  // CHECK_GE(sequence->kv_state().current_max_tokens_capacity(), seq_len);
  CHECK_GT(q_seq_len, 0) << "at least one token should be processed. "
                         << "n_tokens: " << n_tokens
                         << ", n_kv_cache_tokens: " << n_kv_cache_tokens
                         << ", current_max_tokens_capacity: "
                         << sequence->kv_state().current_max_tokens_capacity()
                         << ", allowed_max_tokens: "
                         << allowed_max_tokens_[seq_index];

  // Update state
  base_state.empty_kv_cache = true;
  base_state.max_seq_len = std::max(base_state.max_seq_len, seq_len);
  base_state.q_max_seq_len = std::max(base_state.q_max_seq_len, q_seq_len);
#if defined(USE_NPU)
  base_state.seq_lens.push_back(seq_len);
  base_state.q_seq_lens.push_back(q_seq_len);
  state.decode_seq_lens.push_back(decode_seq_len);
  state.decode_q_seq_lens.push_back(decode_q_seq_len);
#elif defined(USE_MLU)
  base_state.seq_lens.push_back(base_state.seq_lens.back() + seq_len);
  base_state.q_seq_lens.push_back(base_state.q_seq_lens.back() + q_seq_len);
  state.decode_seq_lens.push_back(state.decode_seq_lens.back() +
                                  decode_seq_len);
  state.decode_q_seq_lens.push_back(state.decode_q_seq_lens.back() +
                                    decode_q_seq_len);
#endif

  // Call our enhanced method to process tokens and positions
  // This handles both regular decode and step-level decode cases
  extract_tokens_and_positions(sequence, n_kv_cache_tokens, seq_len, &state);

  // not Setup KV cache for prefill

  if (!FLAGS_enable_continuous_kvcache) {
    setup_kv_cache_info(sequence,
                        n_kv_cache_tokens,
                        decode_seq_len,
                        decode_q_seq_len,
                        &base_state,
                        write_block_ids_ptr);
  } else {
    setup_continuous_kv_cache_info(sequence,
                                   n_kv_cache_tokens,
                                   decode_seq_len,
                                   decode_q_seq_len,
                                   &base_state);
  }

  // Track prefill sequences
  if (sequence->is_prefill_stage()) {
    base_state.prefill_seq_len++;
  }

  // Input for beam search kernel
  // if (FLAGS_enable_beam_search_kernel && sequence->check_beam_search()) {
  //   int32_t bw = std::max(1, FLAGS_beam_width);
  //   base_state.acc_logprob_vec.insert(base_state.acc_logprob_vec.end(), bw,
  //   0.0f);
  // }

  // Multi-step specific processing
}

void MultiStepBatchInputBuilder::extract_tokens_and_positions(
    Sequence* sequence,
    uint32_t n_kv_cache_tokens,
    uint32_t seq_len,
    MultiStepBuilderState* state_ptr) {
  // First call the base class implementation with in_step_decode=false
  BatchInputBuilder::extract_tokens_and_positions(
      sequence, n_kv_cache_tokens, seq_len, &state_ptr->base_state, false);
  // begin process decode data
  seq_len = n_kv_cache_tokens + 1;
  // std::unordered_map<int32_t, int32_t> adjusted_token_to_count_map;
  // 从第一个 decode 到 max_decode round 都生成采样参数，
  // 都采用prompt最后一个token做占位，实际上这些需要执行 beam_width 次
  // 应当在 worker 上再broadcast
  // uint32_t pos_idx = n_kv_cache_tokens;
  // adjusted_token_to_count_map[token_ids[pos_idx - 1]] = 1;
  // TODO_1111 不考虑 use_mrope_ 为true 的情况
  // 是对 sequence 循环调的这个，所以每个sequence 内容理论上不一样，但是prefill
  // 阶段只有一个 sequnece 所以在worker 侧需要替换或broadcast.
  // state_ptr->flatten_tokens_vec.push_back(token_ids[pos_idx - 1]);
  // state_ptr->flatten_positions_vec.push_back(static_cast<int32_t>(pos_idx));
  // 不需要执行 handle_sampling_parameters 直接填充
  // handle_sampling_parameters(
  //     sequence, pos_idx, seq_len, adjusted_token_to_count_map, state_ptr);
  // state_ptr->extra_token_ids.push_back(-1);
  // state_ptr->embedding_ids.push_back(sequence->get_embedding_id());
  // 用于sample，每次都一样
  // 以下需要配套构造 sample_params ，在param_utils.cpp 中
  // proto_to_forward_input 中使用 因为每个 request 的 每个 sequence
  // 都一样，所以decode 可以只传一份，在 worker 端进行broadcast.
  uint32_t prompt_len = sequence->num_prompt_tokens();
  state_ptr->decode_positions_vec.push_back(static_cast<int32_t>(prompt_len));

  int32_t bw = std::max(1, FLAGS_beam_width);
  const int32_t sel_start =
      static_cast<int32_t>(state_ptr->decode_selected_token_idxes.size());
  state_ptr->decode_selected_token_idxes.reserve(sel_start + bw);
  state_ptr->decode_sample_idxes.reserve(state_ptr->decode_sample_idxes.size() +
                                         bw);
  state_ptr->decode_unique_token_ids_vec.resize(
      state_ptr->decode_unique_token_ids_vec.size() + bw);
  state_ptr->decode_unique_token_counts_vec.resize(
      state_ptr->decode_unique_token_counts_vec.size() + bw);
  state_ptr->decode_unique_token_lens_vec.insert(
      state_ptr->decode_unique_token_lens_vec.end(), bw, 0);
  state_ptr->decode_sampling_params.reserve(
      state_ptr->decode_sampling_params.size() + bw);
  for (int32_t i = 0; i < bw; ++i) {
    const int32_t idx = sel_start + i;
    state_ptr->decode_selected_token_idxes.push_back(idx);
    state_ptr->decode_sample_idxes.push_back(idx);
    state_ptr->decode_sampling_params.push_back(sequence->sampling_param());
  }
}

void MultiStepBatchInputBuilder::setup_kv_cache_info(
    Sequence* sequence,
    uint32_t n_kv_cache_tokens,
    uint32_t seq_len,
    uint32_t q_seq_len,
    BatchInputBuilder::BuilderState* state_ptr,
    std::unordered_set<int32_t>* write_block_ids_ptr) {
  BatchInputBuilder::BuilderState& state =
      state_ptr ? *state_ptr : this->state_;
  const auto blocks = sequence->kv_state().kv_blocks();
  std::vector<int32_t> block_ids;
  block_ids.reserve(blocks.size());
  for (const auto& block : blocks) {
    block_ids.push_back(block.id());
  }
  state.block_tables_vec.emplace_back(std::move(block_ids));
}

void MultiStepBatchInputBuilder::setup_continuous_kv_cache_info(
    Sequence* sequence,
    uint32_t n_kv_cache_tokens,
    uint32_t seq_len,
    uint32_t q_seq_len,
    BatchInputBuilder::BuilderState* state_ptr) {
  (void)sequence;
  (void)n_kv_cache_tokens;
  (void)seq_len;
  (void)q_seq_len;
  (void)state_ptr;
}

ForwardInput MultiStepBatchInputBuilder::state_to_forward_input() {
  // First call the base class implementation to get the basic ForwardInput
  ForwardInput forward_input = BatchInputBuilder::state_to_forward_input();

  // Add multi-step specific data using existing ForwardInput fields
  auto& multi_step_state = multi_step_state_;
  multi_step_state.total_steps = FLAGS_max_decode_rounds;

  // Set step-level decode metadata for multi-step processing
  // These fields are already present in ForwardInput for step-level decode
  forward_input.beam_width = FLAGS_beam_width;
  forward_input.total_round = multi_step_state.total_steps;

  // Set shared_kv_shape if we have multi-step decode data
  if (!multi_step_state.decode_seq_lens.empty() && !sequences_.empty()) {
    // Set decode kv cache shape for step-level decode
    // Format: [batch_size * beam_width, n_kv_heads, step_rounds, head_dim]
    int64_t batch_size = static_cast<int64_t>(sequences_.size());
    int64_t step_rounds = static_cast<int64_t>(multi_step_state.total_steps);

    int64_t n_kv_heads =
        args_ ? args_->n_kv_heads().value_or(args_->n_heads()) : 0;
    int64_t head_dim = args_ ? args_->head_dim() : 0;

    forward_input.shared_kv_shape = {
        batch_size * FLAGS_max_token_per_req, n_kv_heads, head_dim};
  }

  if (!multi_step_state.decode_seq_lens.empty()) {
    auto tensor_options = torch::TensorOptions()
                              .dtype(torch::kInt)
                              .device(torch::kCPU)
                              .pinned_memory(true);
    forward_input.input_params.decode_kv_seq_lens =
        torch::tensor(multi_step_state.decode_seq_lens, tensor_options);
    forward_input.input_params.decode_q_seq_lens =
        torch::tensor(multi_step_state.decode_q_seq_lens, tensor_options);
    forward_input.input_params.decode_kv_seq_lens_vec =
        multi_step_state.decode_seq_lens;
    forward_input.input_params.decode_q_seq_lens_vec =
        multi_step_state.decode_q_seq_lens;
  }

  if (!multi_step_state.decode_positions_vec.empty()) {
    forward_input.decode_positions_vec = multi_step_state.decode_positions_vec;
  }

  return forward_input;
}

namespace {
void PrintRawForwardInput(const RawForwardInput& input) {
  LOG(INFO) << "=== RawForwardInput Debug Info ===";

  // Basic vectors
  LOG(INFO) << "flatten_tokens_vec size: " << input.flatten_tokens_vec.size();
  if (!input.flatten_tokens_vec.empty()) {
    LOG(INFO) << "flatten_tokens_vec: [";
    for (size_t i = 0;
         i < std::min(input.flatten_tokens_vec.size(), size_t(10));
         ++i) {
      LOG(INFO) << "  " << input.flatten_tokens_vec[i];
    }
    if (input.flatten_tokens_vec.size() > 10) {
      LOG(INFO) << "  ... (showing first 10 of "
                << input.flatten_tokens_vec.size() << ")";
    }
    LOG(INFO) << "]";
  }

  LOG(INFO) << "flatten_positions_vec size: "
            << input.flatten_positions_vec.size();
  if (!input.flatten_positions_vec.empty()) {
    LOG(INFO) << "flatten_positions_vec: [";
    for (size_t i = 0;
         i < std::min(input.flatten_positions_vec.size(), size_t(10));
         ++i) {
      LOG(INFO) << "  " << input.flatten_positions_vec[i];
    }
    if (input.flatten_positions_vec.size() > 10) {
      LOG(INFO) << "  ... (showing first 10 of "
                << input.flatten_positions_vec.size() << ")";
    }
    LOG(INFO) << "]";
  }

  // Sampling params
  LOG(INFO) << "sampling_params size: " << input.sampling_params.size();
  LOG(INFO) << "selected_token_idxes size: "
            << input.selected_token_idxes.size();
  LOG(INFO) << "sample_idxes size: " << input.sample_idxes.size();

  // Unique token info
  LOG(INFO) << "unique_token_ids_vec size: "
            << input.unique_token_ids_vec.size();
  LOG(INFO) << "unique_token_counts_vec size: "
            << input.unique_token_counts_vec.size();
  LOG(INFO) << "unique_token_lens_vec size: "
            << input.unique_token_lens_vec.size();
  if (!input.unique_token_lens_vec.empty()) {
    LOG(INFO) << "unique_token_lens_vec: [";
    for (size_t i = 0; i < input.unique_token_lens_vec.size(); ++i) {
      LOG(INFO) << "  " << input.unique_token_lens_vec[i];
    }
    LOG(INFO) << "]";
  }

  // Decode sampling params
  LOG(INFO) << "decode_sampling_params size: "
            << input.decode_sampling_params.size();
  LOG(INFO) << "decode_selected_token_idxes size: "
            << input.decode_selected_token_idxes.size();
  LOG(INFO) << "decode_sample_idxes size: " << input.decode_sample_idxes.size();
  LOG(INFO) << "decode_unique_token_ids_vec size: "
            << input.decode_unique_token_ids_vec.size();
  LOG(INFO) << "decode_unique_token_counts_vec size: "
            << input.decode_unique_token_counts_vec.size();
  LOG(INFO) << "decode_unique_token_lens_vec size: "
            << input.decode_unique_token_lens_vec.size();

  // Boolean flags
  LOG(INFO) << "empty_kv_cache: " << (input.empty_kv_cache ? "true" : "false");
  LOG(INFO) << "global_empty_kv_cache: "
            << (input.global_empty_kv_cache ? "true" : "false");

  // Sequence info
  LOG(INFO) << "max_seq_len: " << input.max_seq_len;
  LOG(INFO) << "q_max_seq_len: " << input.q_max_seq_len;
  LOG(INFO) << "num_sequences: " << input.num_sequences;
  LOG(INFO) << "prefill_seq_len: " << input.prefill_seq_len;

  // Sequence lengths
  LOG(INFO) << "seq_lens size: " << input.seq_lens.size();
  if (!input.seq_lens.empty()) {
    LOG(INFO) << "seq_lens: [";
    for (size_t i = 0; i < input.seq_lens.size(); ++i) {
      LOG(INFO) << "  " << input.seq_lens[i];
    }
    LOG(INFO) << "]";
  }

  LOG(INFO) << "q_seq_lens size: " << input.q_seq_lens.size();
  LOG(INFO) << "decode_seq_lens size: " << input.decode_seq_lens.size();
  LOG(INFO) << "decode_q_seq_lens size: " << input.decode_q_seq_lens.size();

  // Token slot info
  LOG(INFO) << "new_token_slot_ids size: " << input.new_token_slot_ids.size();
  if (!input.new_token_slot_ids.empty()) {
    LOG(INFO) << "new_token_slot_ids: [";
    for (size_t i = 0;
         i < std::min(input.new_token_slot_ids.size(), size_t(10));
         ++i) {
      LOG(INFO) << "  " << input.new_token_slot_ids[i];
    }
    if (input.new_token_slot_ids.size() > 10) {
      LOG(INFO) << "  ... (showing first 10 of "
                << input.new_token_slot_ids.size() << ")";
    }
    LOG(INFO) << "]";
  }

  // Block tables
  LOG(INFO) << "block_tables_vec size: " << input.block_tables_vec.size();
  for (size_t i = 0; i < input.block_tables_vec.size(); ++i) {
    LOG(INFO) << "block_tables_vec[" << i
              << "] size: " << input.block_tables_vec[i].size();
  }

  // DP info
  LOG(INFO) << "dp_global_token_nums size: "
            << input.dp_global_token_nums.size();
  if (!input.dp_global_token_nums.empty()) {
    LOG(INFO) << "dp_global_token_nums: [";
    for (size_t i = 0; i < input.dp_global_token_nums.size(); ++i) {
      LOG(INFO) << "  " << input.dp_global_token_nums[i];
    }
    LOG(INFO) << "]";
  }

  // Transfer and embedding info
  LOG(INFO) << "transfer_kv_infos size: " << input.transfer_kv_infos.size();
  LOG(INFO) << "embeddings size: " << input.embeddings.size();
  LOG(INFO) << "embedding_ids size: " << input.embedding_ids.size();

  // Extra tokens
  LOG(INFO) << "extra_token_ids size: " << input.extra_token_ids.size();
  if (!input.extra_token_ids.empty()) {
    LOG(INFO) << "extra_token_ids: [";
    for (size_t i = 0; i < input.extra_token_ids.size(); ++i) {
      LOG(INFO) << "  " << input.extra_token_ids[i];
    }
    LOG(INFO) << "]";
  }

  // Cache block info
  LOG(INFO) << "async_copy_out_blocks size: "
            << input.async_copy_out_blocks.size();
  LOG(INFO) << "copy_out_blocks size: " << input.copy_out_blocks.size();
  LOG(INFO) << "copy_in_blocks size: " << input.copy_in_blocks.size();
  LOG(INFO) << "swap_blocks size: " << input.swap_blocks.size();

  // Block indices
  LOG(INFO) << "src_block_indices size: " << input.src_block_indices.size();
  LOG(INFO) << "dst_block_indices size: " << input.dst_block_indices.size();
  LOG(INFO) << "cum_sum size: " << input.cum_sum.size();

  // Cache offsets
  LOG(INFO) << "new_cache_slot_offsets size: "
            << input.new_cache_slot_offsets.size();
  LOG(INFO) << "kv_cache_start_offsets size: "
            << input.kv_cache_start_offsets.size();

  // Beam search info
  LOG(INFO) << "acc_logprob_vec size: " << input.acc_logprob_vec.size();
  LOG(INFO) << "decode_positions_vec size: "
            << input.decode_positions_vec.size();
  LOG(INFO) << "beam_width: " << input.beam_width;
  LOG(INFO) << "current_round: " << input.current_round;
  LOG(INFO) << "total_round: " << input.total_round;

  // Shared KV shape
  LOG(INFO) << "shared_kv_shape size: " << input.shared_kv_shape.size();
  if (!input.shared_kv_shape.empty()) {
    LOG(INFO) << "shared_kv_shape: [";
    for (size_t i = 0; i < input.shared_kv_shape.size(); ++i) {
      LOG(INFO) << "  " << input.shared_kv_shape[i];
    }
    LOG(INFO) << "]";
  }

  LOG(INFO) << "=== End RawForwardInput Debug Info ===";
}

}  // namespace

RawForwardInput MultiStepBatchInputBuilder::state_to_raw_forward_input(
    BatchInputBuilder::BuilderState* state_ptr) {
  // First call the base class implementation to get the basic RawForwardInput
  RawForwardInput raw_forward_input =
      BatchInputBuilder::state_to_raw_forward_input(
          state_ptr ? state_ptr : &multi_step_state_.base_state);

  // Add multi-step specific data using existing RawForwardInput fields
  auto& multi_step_state = multi_step_state_;
  multi_step_state.total_steps = FLAGS_max_decode_rounds;

  // Set step-level decode metadata for multi-step processing
  raw_forward_input.beam_width = FLAGS_beam_width;
  raw_forward_input.total_round = multi_step_state.total_steps;

  // Set shared_kv_shape if we have multi-step decode data
  if (!multi_step_state.decode_seq_lens.empty() && !sequences_.empty()) {
    // Set decode kv cache shape for step-level decode
    // Format: [batch_size * beam_width, n_kv_heads, step_rounds, head_dim]
    int64_t batch_size = static_cast<int64_t>(sequences_.size());
    int64_t step_rounds = static_cast<int64_t>(multi_step_state.total_steps);

    int64_t n_kv_heads =
        args_ ? args_->n_kv_heads().value_or(args_->n_heads()) : 0;
    int64_t head_dim = args_ ? args_->head_dim() : 0;

    raw_forward_input.shared_kv_shape = {
        batch_size * FLAGS_max_token_per_req, n_kv_heads, head_dim};
  }

  if (!multi_step_state.decode_seq_lens.empty()) {
    raw_forward_input.decode_seq_lens.insert(
        raw_forward_input.decode_seq_lens.end(),
        multi_step_state.decode_seq_lens.begin(),
        multi_step_state.decode_seq_lens.end());
    raw_forward_input.decode_q_seq_lens.insert(
        raw_forward_input.decode_q_seq_lens.end(),
        multi_step_state.decode_q_seq_lens.begin(),
        multi_step_state.decode_q_seq_lens.end());
  }

  if (!multi_step_state.decode_positions_vec.empty()) {
    raw_forward_input.decode_positions_vec.insert(
        raw_forward_input.decode_positions_vec.end(),
        multi_step_state.decode_positions_vec.begin(),
        multi_step_state.decode_positions_vec.end());
  }

  // append multi-step decode state into raw_forward_input
  if (!multi_step_state.decode_selected_token_idxes.empty()) {
    raw_forward_input.decode_selected_token_idxes.insert(
        raw_forward_input.decode_selected_token_idxes.end(),
        multi_step_state.decode_selected_token_idxes.begin(),
        multi_step_state.decode_selected_token_idxes.end());
    raw_forward_input.decode_sample_idxes.insert(
        raw_forward_input.decode_sample_idxes.end(),
        multi_step_state.decode_sample_idxes.begin(),
        multi_step_state.decode_sample_idxes.end());
    raw_forward_input.decode_unique_token_ids_vec.insert(
        raw_forward_input.decode_unique_token_ids_vec.end(),
        multi_step_state.decode_unique_token_ids_vec.begin(),
        multi_step_state.decode_unique_token_ids_vec.end());
    raw_forward_input.decode_unique_token_counts_vec.insert(
        raw_forward_input.decode_unique_token_counts_vec.end(),
        multi_step_state.decode_unique_token_counts_vec.begin(),
        multi_step_state.decode_unique_token_counts_vec.end());
    raw_forward_input.decode_unique_token_lens_vec.insert(
        raw_forward_input.decode_unique_token_lens_vec.end(),
        multi_step_state.decode_unique_token_lens_vec.begin(),
        multi_step_state.decode_unique_token_lens_vec.end());
    raw_forward_input.decode_sampling_params.insert(
        raw_forward_input.decode_sampling_params.end(),
        multi_step_state.decode_sampling_params.begin(),
        multi_step_state.decode_sampling_params.end());
  }
  // PrintRawForwardInput(raw_forward_input);
  return raw_forward_input;
}

}  // namespace xllm
