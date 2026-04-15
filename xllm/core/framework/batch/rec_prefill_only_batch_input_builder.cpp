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

#include "rec_prefill_only_batch_input_builder.h"

#include <glog/logging.h>

#include <cstdint>
#include <sstream>

#include "framework/request/sequence.h"

namespace xllm {

namespace {

constexpr const char* kMtgrHistoryLenName = "history_len";
constexpr const char* kMtgrContextLenName = "context_len";
constexpr const char* kMtgrRealTimeLenName = "real_time_len";
constexpr const char* kMtgrTargetLenName = "target_len";

std::string format_torch_tensor(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return "undefined";
  }
  std::ostringstream oss;
  oss << "shape=[";
  for (int64_t i = 0; i < tensor.dim(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << tensor.size(i);
  }
  oss << "], dtype=" << tensor.scalar_type() << ", device=" << tensor.device();
  return oss.str();
}

int32_t get_required_length(const MMData& mm_data, const char* key) {
  auto tensor = mm_data.get<torch::Tensor>(key);
  CHECK(tensor.has_value()) << "Missing MTGR metadata tensor: " << key;
  const auto& value = tensor.value();
  CHECK(value.defined()) << "MTGR metadata tensor is undefined: " << key;
  CHECK_EQ(value.numel(), 1) << "MTGR metadata tensor must contain one value: "
                             << key;
  if (value.scalar_type() == torch::kInt32) {
    return value.reshape({-1})[0].item<int32_t>();
  }
  if (value.scalar_type() == torch::kInt64) {
    return static_cast<int32_t>(value.reshape({-1})[0].item<int64_t>());
  }
  LOG(FATAL) << "Unsupported MTGR metadata tensor dtype for " << key << ": "
             << value.scalar_type();
  return 0;
}

}  // namespace

RecPrefillOnlyBatchInputBuilder::RecPrefillOnlyBatchInputBuilder(
    const std::vector<SequencesGroup*>& sequence_groups,
    const std::vector<uint32_t>& allowed_max_tokens,
    const std::vector<torch::Tensor>& /*input_embeddings_vec*/,
    const std::vector<MMData>& /*mm_data_vec*/,
    std::vector<BlockTransferInfo>* swap_block_transfer_infos,
    uint64_t batch_id,
    const ModelArgs* args,
    BatchForwardType batch_forward_type,
    ThreadPool* thread_pool)
    : sequence_groups_(sequence_groups),
      allowed_max_tokens_(allowed_max_tokens),
      swap_block_transfer_infos_(swap_block_transfer_infos),
      batch_id_(batch_id),
      args_(args),
      batch_forward_type_(batch_forward_type),
      thread_pool_(thread_pool) {}

ForwardInput RecPrefillOnlyBatchInputBuilder::build_rec_forward_input(
    uint32_t num_decoding_tokens,
    uint32_t min_decoding_batch_size) {
  std::vector<Sequence*> sequences;
  std::vector<uint32_t> effective_allowed_max_tokens;
  std::vector<torch::Tensor> trimmed_embeddings;
  std::vector<MMData> mm_data_vec;
  std::vector<int32_t> history_lens;
  std::vector<int32_t> context_lens;
  std::vector<int32_t> real_time_lens;
  std::vector<int32_t> target_lens;
  std::vector<int32_t> matched_prefix_lens;

  for (auto* sequence_group : sequence_groups_) {
    CHECK(sequence_group != nullptr);
    for (const auto& seq_ptr : sequence_group->sequences()) {
      auto* sequence = seq_ptr.get();
      CHECK(sequence != nullptr);
      sequences.push_back(sequence);

      const uint32_t fallback_budget = sequence->num_need_compute_tokens();
      effective_allowed_max_tokens.push_back(
          effective_allowed_max_tokens.size() < allowed_max_tokens_.size()
              ? allowed_max_tokens_[effective_allowed_max_tokens.size()]
              : fallback_budget);

      const int32_t matched_prefix =
          static_cast<int32_t>(sequence->kv_state().kv_cache_tokens_num());
      matched_prefix_lens.push_back(matched_prefix);

      const auto& input_embedding = sequence->get_input_embedding();
      CHECK(input_embedding.defined())
          << "MTGR requires per-sequence input_embedding";
      CHECK_EQ(input_embedding.dim(), 2)
          << "MTGR input_embedding must be 2-D [seq_len, hidden]";
      CHECK_LE(matched_prefix, input_embedding.size(0))
          << "matched_prefix exceeds input_embedding length";
      CHECK_LT(matched_prefix, input_embedding.size(0))
          << "matched_prefix must leave at least one token to compute";
      trimmed_embeddings.emplace_back(input_embedding.narrow(
          /*dim=*/0,
          /*start=*/matched_prefix,
          /*length=*/input_embedding.size(0) - matched_prefix));

      const auto& mm_data = sequence->get_mm_data();
      mm_data_vec.emplace_back(mm_data);

      const int32_t history = get_required_length(mm_data, kMtgrHistoryLenName);
      const int32_t context = get_required_length(mm_data, kMtgrContextLenName);
      const int32_t real_time =
          get_required_length(mm_data, kMtgrRealTimeLenName);
      const int32_t target = get_required_length(mm_data, kMtgrTargetLenName);
      CHECK_EQ(history + context + real_time + target, input_embedding.size(0))
          << "MTGR segment lengths must sum to input_embedding length";
      CHECK_LE(matched_prefix, history + context + real_time)
          << "matched_prefix must stay within [history|context|real_time]";
      history_lens.push_back(history);
      context_lens.push_back(context);
      real_time_lens.push_back(real_time);
      target_lens.push_back(target);
    }
  }

  if (sequences.empty()) {
    return ForwardInput{};
  }

  BatchInputBuilder builder(sequences,
                            effective_allowed_max_tokens,
                            trimmed_embeddings,
                            mm_data_vec,
                            swap_block_transfer_infos_,
                            batch_id_,
                            args_,
                            batch_forward_type_,
                            /*cp_size=*/1,
                            thread_pool_);
  ForwardInput forward_input = builder.build_forward_input(num_decoding_tokens,
                                                           min_decoding_batch_size);

  auto& mtgr_params = forward_input.input_params.mutable_mtgr_params();
  mtgr_params.history_lens = torch::tensor(history_lens, torch::kInt32);
  mtgr_params.context_lens = torch::tensor(context_lens, torch::kInt32);
  mtgr_params.real_time_lens = torch::tensor(real_time_lens, torch::kInt32);
  mtgr_params.target_lens = torch::tensor(target_lens, torch::kInt32);
  mtgr_params.matched_prefix_lens =
      torch::tensor(matched_prefix_lens, torch::kInt32);

  LOG(INFO) << "[MTGR_TRACE][BATCH] batch_id=" << batch_id_
            << " sequences=" << sequences.size()
            << " token_ids=" << format_torch_tensor(forward_input.token_ids)
            << " positions=" << format_torch_tensor(forward_input.positions)
            << " input_embedding="
            << format_torch_tensor(forward_input.input_params.input_embedding)
            << " history_lens=" << format_torch_tensor(mtgr_params.history_lens)
            << " context_lens=" << format_torch_tensor(mtgr_params.context_lens)
            << " real_time_lens="
            << format_torch_tensor(mtgr_params.real_time_lens)
            << " target_lens=" << format_torch_tensor(mtgr_params.target_lens)
            << " matched_prefix_lens="
            << format_torch_tensor(mtgr_params.matched_prefix_lens);

  return forward_input;
}

}  // namespace xllm
