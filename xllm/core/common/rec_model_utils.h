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

#pragma once

#include <cstdint>
#include <string_view>

#include <glog/logging.h>

#include "core/common/global_flags.h"

namespace xllm {

enum class RecModelKind : int8_t {
  kNone = 0,
  kOneRec = 1,
  kLlmRec = 2,
  kMtgr = 3,
};

// Pipeline strategy types (extensible for future strategies)
enum class RecPipelineType : uint8_t {
  kLlmRecDefault = 0,             // LlmRec without mm_data (pure qwen)
  kLlmRecWithMmData = 1,          // LlmRec with mm_data (qwen + embedding)
  kOneRecDefault = 2,             // OneRec
  kLlmRecMultiRoundPipeline = 3,  // LlmRec multi-round pipeline (device loop)
  kRecPrefillOnly = 4,            // Rec prefill-only pipeline
};

enum class MTGRCachePolicy : uint8_t {
  // Conservative baseline: cache and write back the full logical sequence,
  // including the target segment.
  kFullSequence = 0,
  // Optimized Hopper experiment: cache only the reusable prefix and keep the
  // target segment live for the current request.
  kPrefixOnly = 1,
};

// Check if Rec multi-round mode is enabled.
// Rec multi-round mode: multi-round decode loop runs on device (worker layer),
// while the engine issues a single step.
inline bool is_rec_multi_round_mode() { return FLAGS_max_decode_rounds > 0; }

// Get the number of decode rounds for Rec multi-round mode.
// Returns 0 if Rec multi-round mode is disabled.
inline int32_t get_rec_multi_round_decode_rounds() {
  return is_rec_multi_round_mode() ? FLAGS_max_decode_rounds : 0;
}

// Pipeline strategy selector: choose strategy based on RecModelKind
inline RecPipelineType get_rec_pipeline_type(RecModelKind kind) {
  switch (kind) {
    case RecModelKind::kLlmRec:
      if (is_rec_multi_round_mode()) {
        return RecPipelineType::kLlmRecMultiRoundPipeline;
      } else {
        return RecPipelineType::kLlmRecDefault;
      }
    case RecModelKind::kOneRec:
      return RecPipelineType::kOneRecDefault;
    case RecModelKind::kMtgr:
      return RecPipelineType::kRecPrefillOnly;
    default:
      return RecPipelineType::kLlmRecDefault;
  }
}

inline constexpr bool is_onerec_model_type(std::string_view model_type) {
  return model_type == "onerec";
}

inline constexpr bool is_mtgr_model_type(std::string_view model_type) {
  return model_type == "mtgr";
}

inline constexpr bool is_llmrec_model_type(std::string_view model_type) {
  return model_type == "qwen2" || model_type == "qwen3" ||
         model_type == "qwen3_moe";
}

inline constexpr RecModelKind get_rec_model_kind(std::string_view model_type) {
  if (is_onerec_model_type(model_type)) {
    return RecModelKind::kOneRec;
  }
  if (is_mtgr_model_type(model_type)) {
    return RecModelKind::kMtgr;
  }
  if (is_llmrec_model_type(model_type)) {
    return RecModelKind::kLlmRec;
  }
  return RecModelKind::kNone;
}

inline bool is_mtgr_flashinfer_token_mask_backend() {
  return FLAGS_mtgr_attention_backend == "flashinfer_token_mask";
}

inline bool is_mtgr_hopper_backend() {
  return FLAGS_mtgr_attention_backend == "hopper";
}

inline MTGRCachePolicy get_mtgr_cache_policy() {
  if (is_mtgr_flashinfer_token_mask_backend()) {
    return MTGRCachePolicy::kFullSequence;
  }
  if (is_mtgr_hopper_backend()) {
    return MTGRCachePolicy::kPrefixOnly;
  }
  CHECK(false) << "Unsupported MTGR attention backend: "
               << FLAGS_mtgr_attention_backend
               << ". Expected hopper or flashinfer_token_mask.";
  return MTGRCachePolicy::kPrefixOnly;
}

inline const char* mtgr_cache_policy_name(MTGRCachePolicy policy) {
  switch (policy) {
    case MTGRCachePolicy::kFullSequence:
      return "full_sequence";
    case MTGRCachePolicy::kPrefixOnly:
      return "prefix_only";
  }
  CHECK(false) << "Unknown MTGR cache policy";
  return "unknown";
}

inline const char* current_mtgr_cache_policy_name() {
  return mtgr_cache_policy_name(get_mtgr_cache_policy());
}

inline int32_t mtgr_cacheable_len_for_policy(int32_t prefix_match_limit,
                                             int32_t logical_total_len) {
  CHECK_GE(prefix_match_limit, 0)
      << "MTGR prefix match limit must be non-negative";
  CHECK_GE(logical_total_len, prefix_match_limit)
      << "MTGR logical total length must cover the prefix match limit";
  return get_mtgr_cache_policy() == MTGRCachePolicy::kFullSequence
             ? logical_total_len
             : prefix_match_limit;
}

}  // namespace xllm
