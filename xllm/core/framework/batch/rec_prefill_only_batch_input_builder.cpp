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

#include <algorithm>
#include <cstdint>
#include <optional>
#include <sstream>

#include "layers/common/attention_metadata.h"
#include "framework/request/sequence.h"

namespace xllm {

namespace {

constexpr const char* kMtgrSegmentOffsetsName = "segment_offsets";
constexpr const char* kMtgrSegmentRulesName = "segment_rules";

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

std::optional<torch::Tensor> get_optional_i32_tensor(const MMData& mm_data,
                                                     const char* key) {
  auto tensor = mm_data.get<torch::Tensor>(key);
  if (!tensor.has_value() || !tensor.value().defined()) {
    return std::nullopt;
  }
  return tensor.value().to(torch::kInt32).cpu().contiguous();
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
  std::vector<torch::Tensor> segment_offset_rows_i32;
  std::vector<torch::Tensor> matched_prefix_lens_i32;
  std::vector<torch::Tensor> q_seq_starts_i32;
  torch::Tensor batch_segment_rules_i32;
  bool has_no_match_request = false;
  bool has_partial_match_request = false;
  int32_t packed_q_start = 0;
  const auto i32_options = torch::TensorOptions().dtype(torch::kInt32);

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
      has_no_match_request = has_no_match_request || (matched_prefix == 0);
      has_partial_match_request =
          has_partial_match_request || (matched_prefix > 0);

      const auto& input_embedding = sequence->get_input_embedding();
      CHECK(input_embedding.defined())
          << "MTGR requires per-sequence input_embedding";
      CHECK_EQ(input_embedding.dim(), 2)
          << "MTGR input_embedding must be 2-D [seq_len, hidden]";
      CHECK_LE(matched_prefix, input_embedding.size(0))
          << "matched_prefix exceeds input_embedding length";
      CHECK_LT(matched_prefix, input_embedding.size(0))
          << "matched_prefix must leave at least one token to compute";
      const int32_t local_q_len =
          static_cast<int32_t>(input_embedding.size(0) - matched_prefix);
      q_seq_starts_i32.emplace_back(torch::tensor({packed_q_start}, i32_options));
      packed_q_start += local_q_len;
      trimmed_embeddings.emplace_back(input_embedding.narrow(
          /*dim=*/0,
          /*start=*/matched_prefix,
          /*length=*/local_q_len));

      const auto& mm_data = sequence->get_mm_data();
      mm_data_vec.emplace_back(mm_data);

      auto segment_offsets =
          get_optional_i32_tensor(mm_data, kMtgrSegmentOffsetsName);
      auto segment_rules =
          get_optional_i32_tensor(mm_data, kMtgrSegmentRulesName);
      CHECK(segment_offsets.has_value())
          << "MTGR segmented protocol requires segment_offsets";
      CHECK(segment_rules.has_value())
          << "MTGR segmented protocol requires segment_rules";
      auto offsets = segment_offsets.value();
      if (offsets.dim() == 2 && offsets.size(0) == 1) {
        offsets = offsets.view({offsets.size(1)}).contiguous();
      }
      auto rules = segment_rules.value();
      if (rules.dim() == 2 && rules.size(0) == 1) {
        rules = rules.view({rules.size(1)}).contiguous();
      }
      CHECK_EQ(offsets.dim(), 1);
      CHECK_EQ(rules.dim(), 1);
      CHECK_EQ(offsets.size(0), rules.size(0) + 1);
      CHECK_EQ(offsets[offsets.size(0) - 1].item<int32_t>(),
               input_embedding.size(0))
          << "MTGR segment_offsets last value must equal input length";
      CHECK_LE(matched_prefix, offsets[offsets.size(0) - 2].item<int32_t>())
          << "matched_prefix must stay before the final target segment";
      segment_offset_rows_i32.emplace_back(offsets);
      matched_prefix_lens_i32.emplace_back(
          torch::tensor({matched_prefix}, i32_options));
      if (!batch_segment_rules_i32.defined()) {
        batch_segment_rules_i32 = rules;
      } else {
        CHECK(torch::equal(batch_segment_rules_i32, rules))
            << "MTGR batch currently requires identical segment_rules";
      }
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
  mtgr_params.mtgr_segment_offsets_i32 =
      torch::stack(segment_offset_rows_i32).contiguous();
  mtgr_params.mtgr_segment_rules_i32 = batch_segment_rules_i32.contiguous();
  mtgr_params.mtgr_q_seq_starts_i32 = torch::cat(q_seq_starts_i32).contiguous();
  mtgr_params.mtgr_matched_prefix_lens_i32 =
      torch::cat(matched_prefix_lens_i32).contiguous();
  mtgr_params.mtgr_match_mode =
      has_partial_match_request
          ? (has_no_match_request ? layer::MTGRMatchMode::kMixed
                                  : layer::MTGRMatchMode::kPartialOnly)
          : layer::MTGRMatchMode::kNoMatchOnly;

  LOG(INFO) << "[MTGR_TRACE][BATCH] batch_id=" << batch_id_
            << " sequences=" << sequences.size()
            << " token_ids=" << format_torch_tensor(forward_input.token_ids)
            << " positions=" << format_torch_tensor(forward_input.positions)
            << " input_embedding="
            << format_torch_tensor(forward_input.input_params.input_embedding)
            << " mtgr_segment_offsets_i32="
            << format_torch_tensor(mtgr_params.mtgr_segment_offsets_i32)
            << " mtgr_segment_rules_i32="
            << format_torch_tensor(mtgr_params.mtgr_segment_rules_i32)
            << " mtgr_q_seq_starts_i32="
            << format_torch_tensor(mtgr_params.mtgr_q_seq_starts_i32)
            << " mtgr_matched_prefix_lens_i32="
            << format_torch_tensor(mtgr_params.mtgr_matched_prefix_lens_i32)
            << " mtgr_match_mode="
            << static_cast<int32_t>(mtgr_params.mtgr_match_mode);

  return forward_input;
}

}  // namespace xllm
