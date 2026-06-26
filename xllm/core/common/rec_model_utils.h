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

#include <cstddef>
#include <cstdint>
#include <string>
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
  // Prefix-only plus admission control. Low-value requests still run the full
  // forward with live K/V, but they do not allocate/write persistent KV cache.
  kPrefixOnlyValueGated = 2,
  // Full-cache matching/compute with runtime write admission. Persistent
  // prefix-cache insertion is decided from live free/total KV blocks.
  kDynamicLengthAdmission = 3,
  // Prefix-only matching/compute with runtime write admission based on observed
  // prefix-family reuse under cache pressure.
  kPrefixOnlyReuseHorizon = 4,
};

struct MTGRCachePolicyDecision {
  MTGRCachePolicy policy = MTGRCachePolicy::kPrefixOnly;
  bool cache_admitted = true;
  bool runtime_admission = false;
  bool high_value = true;
  int32_t prefix_len = 0;
  int32_t logical_total_len = 0;
  int32_t cacheable_len = 0;
  int32_t writeback_len = 0;
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
  const std::string& policy = FLAGS_mtgr_kv_cache_policy;
  if (policy == "full_cache" || policy == "full_sequence") {
    return MTGRCachePolicy::kFullSequence;
  }
  if (policy == "prefix_only") {
    return MTGRCachePolicy::kPrefixOnly;
  }
  if (policy == "prefix_only_value_gated" || policy == "value_gated") {
    return MTGRCachePolicy::kPrefixOnlyValueGated;
  }
  if (policy == "dynamic_length_admission" ||
      policy == "free_block_admission" ||
      policy == "full_cache_dynamic_length_admission") {
    return MTGRCachePolicy::kDynamicLengthAdmission;
  }
  if (policy == "prefix_only_reuse_horizon" || policy == "reuse_horizon" ||
      policy == "rha_writeback") {
    return MTGRCachePolicy::kPrefixOnlyReuseHorizon;
  }
  CHECK(policy.empty() || policy == "auto")
      << "Unsupported MTGR KV-cache policy: " << policy
      << ". Expected auto, full_cache, full_sequence, prefix_only, "
         "prefix_only_value_gated, value_gated, dynamic_length_admission, "
         "free_block_admission, full_cache_dynamic_length_admission, "
         "prefix_only_reuse_horizon, reuse_horizon, or rha_writeback.";

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
      return "full_cache";
    case MTGRCachePolicy::kPrefixOnly:
      return "prefix_only";
    case MTGRCachePolicy::kPrefixOnlyValueGated:
      return "prefix_only_value_gated";
    case MTGRCachePolicy::kDynamicLengthAdmission:
      return "dynamic_length_admission";
    case MTGRCachePolicy::kPrefixOnlyReuseHorizon:
      return "prefix_only_reuse_horizon";
  }
  CHECK(false) << "Unknown MTGR cache policy";
  return "unknown";
}

inline const char* current_mtgr_cache_policy_name() {
  return mtgr_cache_policy_name(get_mtgr_cache_policy());
}

inline bool mtgr_is_high_value_prefix(int32_t prefix_len) {
  CHECK_GE(prefix_len, 0) << "MTGR prefix length must be non-negative";
  CHECK(FLAGS_mtgr_kv_cache_value_fn == "prefix_len")
      << "Unsupported MTGR KV-cache value function: "
      << FLAGS_mtgr_kv_cache_value_fn
      << ". Currently only prefix_len is supported.";
  CHECK_GE(FLAGS_mtgr_kv_cache_prefix_len_threshold, 0)
      << "MTGR prefix_len threshold must be non-negative";
  return prefix_len >= FLAGS_mtgr_kv_cache_prefix_len_threshold;
}

inline bool mtgr_uses_runtime_admission(MTGRCachePolicy policy) {
  return policy == MTGRCachePolicy::kDynamicLengthAdmission ||
         policy == MTGRCachePolicy::kPrefixOnlyReuseHorizon;
}

inline bool mtgr_dynamic_length_admission_high_free_blocks(
    size_t free_blocks,
    size_t total_blocks) {
  if (total_blocks == 0) {
    return false;
  }
  // Strictly greater than 70%. The ==70% boundary intentionally falls through
  // to the conservative low-free side.
  return free_blocks * 10 > total_blocks * 7;
}

inline int32_t mtgr_dynamic_length_admission_threshold(size_t free_blocks,
                                                       size_t total_blocks) {
  return mtgr_dynamic_length_admission_high_free_blocks(free_blocks,
                                                        total_blocks)
             ? 1500
             : 2000;
}

inline bool mtgr_dynamic_length_admission_is_admitted(int32_t request_len,
                                                      size_t free_blocks,
                                                      size_t total_blocks) {
  CHECK_GE(request_len, 0) << "MTGR request length must be non-negative";
  return request_len >
         mtgr_dynamic_length_admission_threshold(free_blocks, total_blocks);
}

inline MTGRCachePolicyDecision mtgr_cache_policy_decision(
    int32_t prefix_match_limit,
    int32_t logical_total_len) {
  CHECK_GE(prefix_match_limit, 0)
      << "MTGR prefix match limit must be non-negative";
  CHECK_GE(logical_total_len, prefix_match_limit)
      << "MTGR logical total length must cover the prefix match limit";
  MTGRCachePolicyDecision decision;
  decision.policy = get_mtgr_cache_policy();
  decision.prefix_len = prefix_match_limit;
  decision.logical_total_len = logical_total_len;
  decision.high_value = mtgr_is_high_value_prefix(prefix_match_limit);

  switch (decision.policy) {
    case MTGRCachePolicy::kFullSequence:
      decision.cache_admitted = true;
      decision.cacheable_len = logical_total_len;
      decision.writeback_len = logical_total_len;
      return decision;
    case MTGRCachePolicy::kPrefixOnly:
      decision.cache_admitted = true;
      decision.cacheable_len = prefix_match_limit;
      decision.writeback_len = prefix_match_limit;
      return decision;
    case MTGRCachePolicy::kPrefixOnlyValueGated:
      decision.cache_admitted = decision.high_value;
      decision.cacheable_len =
          decision.cache_admitted ? prefix_match_limit : 0;
      decision.writeback_len =
          decision.cache_admitted ? prefix_match_limit : 0;
      return decision;
    case MTGRCachePolicy::kDynamicLengthAdmission:
      decision.cache_admitted = true;
      decision.runtime_admission = true;
      decision.cacheable_len = logical_total_len;
      decision.writeback_len = logical_total_len;
      return decision;
    case MTGRCachePolicy::kPrefixOnlyReuseHorizon:
      decision.cache_admitted = true;
      decision.runtime_admission = true;
      decision.cacheable_len = prefix_match_limit;
      decision.writeback_len = prefix_match_limit;
      return decision;
  }
  CHECK(false) << "Unknown MTGR cache policy";
  return decision;
}

inline int32_t mtgr_cacheable_len_for_policy(int32_t prefix_match_limit,
                                             int32_t logical_total_len) {
  return mtgr_cache_policy_decision(prefix_match_limit, logical_total_len)
      .cacheable_len;
}

}  // namespace xllm
