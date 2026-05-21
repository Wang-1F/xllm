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

#include "core/common/global_flags.h"
#include "core/common/rec_model_utils.h"
#include "layers/common/attention_metadata.h"
#include "framework/request/sequence.h"
#include "util/mtgr_nvtx.h"
#include "util/mtgr_trace.h"

namespace xllm {

namespace {

constexpr const char* kMtgrSegmentOffsetsName = "segment_offsets";
constexpr const char* kMtgrSegmentRulesName = "segment_rules";
constexpr const char* kMtgrTokenIdsName = "token_ids";

torch::Tensor get_i32_tensor(const MMData& mm_data, const char* key) {
  return mm_data.get<torch::Tensor>(key)
      .value()
      .to(torch::kInt32)
      .cpu()
      .contiguous();
}

std::optional<torch::Tensor> get_optional_i64_tensor(const MMData& mm_data,
                                                     const char* key) {
  auto tensor = mm_data.get<torch::Tensor>(key);
  if (!tensor.has_value()) {
    return std::nullopt;
  }
  return tensor.value().to(torch::kInt64).cpu().contiguous();
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
  MTGR_NVTX_RANGE(1, "MTGR/batch/build_rec_forward_input");
  std::vector<Sequence*> sequences;
  std::vector<uint32_t> effective_allowed_max_tokens;
  std::vector<torch::Tensor> trimmed_embeddings;
  std::vector<torch::Tensor> trimmed_token_ids_i64;
  std::vector<MMData> mm_data_vec;
  std::vector<torch::Tensor> segment_offset_rows_i32;
  std::vector<torch::Tensor> matched_prefix_lens_i32;
  std::vector<torch::Tensor> q_seq_starts_i32;
  std::vector<uint32_t> cacheable_seq_lens;
  torch::Tensor batch_segment_rules_i32;
  bool has_no_match_request = false;
  bool has_partial_match_request = false;
  int32_t packed_q_start = 0;
  const auto i32_options = torch::TensorOptions().dtype(torch::kInt32);

  for (auto* sequence_group : sequence_groups_) {
    for (const auto& seq_ptr : sequence_group->sequences()) {
      auto* sequence = seq_ptr.get();
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

      const auto& mm_data = sequence->get_mm_data();
      mm_data_vec.emplace_back(mm_data);
      auto input_token_ids_i64 =
          get_optional_i64_tensor(mm_data, kMtgrTokenIdsName);
      const auto& input_embedding = sequence->get_input_embedding();
      const bool uses_embedding = input_embedding.defined();

      int64_t total_input_len = 0;
      if (uses_embedding) {
        total_input_len = input_embedding.size(0);
      } else {
        auto token_ids = input_token_ids_i64.value();
        if (token_ids.dim() == 2 && token_ids.size(0) == 1) {
          token_ids = token_ids.view({token_ids.size(1)}).contiguous();
        }
        total_input_len = token_ids.size(0);
        input_token_ids_i64 = token_ids;
      }

      auto offsets = get_i32_tensor(mm_data, kMtgrSegmentOffsetsName);
      if (offsets.dim() == 2 && offsets.size(0) == 1) {
        offsets = offsets.view({offsets.size(1)}).contiguous();
      }
      auto rules = get_i32_tensor(mm_data, kMtgrSegmentRulesName);
      if (rules.dim() == 2 && rules.size(0) == 1) {
        rules = rules.view({rules.size(1)}).contiguous();
      }
      CHECK_GE(offsets.size(0), 2)
          << "MTGR segment_offsets must include at least begin and end";
      const auto* offsets_ptr = offsets.data_ptr<int32_t>();
      const int32_t prefix_match_limit =
          offsets_ptr[static_cast<int64_t>(offsets.size(0)) - 2];
      const int32_t logical_total_len =
          offsets_ptr[static_cast<int64_t>(offsets.size(0)) - 1];
      const int32_t cacheable_len =
          mtgr_cacheable_len_for_policy(prefix_match_limit, logical_total_len);
      CHECK_EQ(logical_total_len, total_input_len)
          << "MTGR segment_offsets last element must match input length";
      CHECK_LE(prefix_match_limit, logical_total_len)
          << "MTGR prefix match limit cannot exceed total length";
      CHECK_GE(prefix_match_limit, 0)
          << "MTGR prefix match limit must be non-negative";
      CHECK_LE(matched_prefix, cacheable_len)
          << "MTGR matched prefix cannot exceed backend cacheable length";
      if (matched_prefix > 0) {
        const auto blocks = sequence->kv_state().kv_blocks();
        CHECK(!blocks.empty())
            << "MTGR matched prefix requires allocated/shared KV blocks";
        CHECK_EQ(matched_prefix % static_cast<int32_t>(blocks[0].size()), 0)
            << "MTGR matched prefix must be block aligned";
      }
      cacheable_seq_lens.push_back(static_cast<uint32_t>(cacheable_len));

      const int32_t local_q_len =
          static_cast<int32_t>(total_input_len - matched_prefix);
      q_seq_starts_i32.emplace_back(
          torch::tensor({packed_q_start}, i32_options));
      MTGR_TRACE(1) << "[BATCH] seq_idx=" << sequences.size() - 1
                    << " input_mode="
                    << (uses_embedding ? "input_embedding" : "token_ids")
                    << " total_input_len=" << total_input_len
                    << " cacheable_len=" << cacheable_len
                    << " cache_policy="
                    << current_mtgr_cache_policy_name()
                    << " prefix_match_limit=" << prefix_match_limit
                    << " matched_prefix=" << matched_prefix
                    << " local_q_len=" << local_q_len
                    << " q_start=" << packed_q_start;
      packed_q_start += local_q_len;
      if (uses_embedding) {
        trimmed_embeddings.emplace_back(input_embedding.narrow(
            /*dim=*/0,
            /*start=*/matched_prefix,
            /*length=*/local_q_len));
      } else {
        trimmed_token_ids_i64.emplace_back(input_token_ids_i64->narrow(
            /*dim=*/0,
            /*start=*/matched_prefix,
            /*length=*/local_q_len));
      }

      MTGR_TRACE(2) << "[BATCH] seq_idx=" << sequences.size() - 1
                    << " segment_offsets_shape=" << offsets.sizes()
                    << " segment_rules_shape=" << rules.sizes();
      segment_offset_rows_i32.emplace_back(offsets);
      matched_prefix_lens_i32.emplace_back(
          torch::tensor({matched_prefix}, i32_options));
      if (!batch_segment_rules_i32.defined()) {
        batch_segment_rules_i32 = rules;
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
                            thread_pool_,
                            &cacheable_seq_lens);
  ForwardInput forward_input;
  {
    MTGR_NVTX_RANGE(2, "MTGR/batch/base_build_forward_input");
    forward_input =
        builder.build_forward_input(num_decoding_tokens, min_decoding_batch_size);
  }

  auto& mtgr_params = forward_input.input_params.mutable_mtgr_params();
  mtgr_params.mtgr_segment_offsets_i32 =
      torch::stack(segment_offset_rows_i32).contiguous();
  mtgr_params.mtgr_segment_rules_i32 = batch_segment_rules_i32.contiguous();
  mtgr_params.mtgr_q_seq_starts_i32 = torch::cat(q_seq_starts_i32).contiguous();
  mtgr_params.mtgr_matched_prefix_lens_i32 =
      torch::cat(matched_prefix_lens_i32).contiguous();
  if (!trimmed_token_ids_i64.empty()) {
    mtgr_params.mtgr_input_token_ids_i64 =
        torch::cat(trimmed_token_ids_i64).contiguous();
  }
  mtgr_params.mtgr_match_mode =
      has_partial_match_request
          ? (has_no_match_request ? layer::MTGRMatchMode::kMixed
                                  : layer::MTGRMatchMode::kPartialOnly)
          : layer::MTGRMatchMode::kNoMatchOnly;
  MTGR_TRACE(1) << "[BATCH] build_rec_forward_input end batch_size="
                << sequences.size() << " total_live_q=" << packed_q_start
                << " match_mode="
                << static_cast<int32_t>(mtgr_params.mtgr_match_mode)
                << " has_no_match=" << has_no_match_request
                << " has_partial=" << has_partial_match_request;
  MTGR_TRACE(2) << "[BATCH] mtgr_params segment_offsets="
                << mtgr_params.mtgr_segment_offsets_i32.sizes()
                << " segment_rules="
                << mtgr_params.mtgr_segment_rules_i32.sizes()
                << " q_seq_starts="
                << mtgr_params.mtgr_q_seq_starts_i32.sizes()
                << " matched_prefix_lens="
                << mtgr_params.mtgr_matched_prefix_lens_i32.sizes();

  return forward_input;
}

}  // namespace xllm
