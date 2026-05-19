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

#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

#include "framework/kv_cache/kv_cache.h"
#include "layers/common/attention_metadata.h"

namespace xllm::kernel::cuda::test::mtgr_attention_harness {

struct MTGRAttentionTestShape {
  int64_t heads = 8;
  int64_t kv_heads = 8;
  int64_t head_dim = 128;
  int64_t history = 1300;
  int64_t context = 8;
  int64_t realtime = 400;
  int64_t target = 800;
  int64_t matched_prefix = 0;

  int64_t total_len() const { return history + context + realtime + target; }
  int64_t local_len() const { return total_len() - matched_prefix; }
  int64_t realtime_matched() const {
    return matched_prefix > history + context
               ? matched_prefix - history - context
               : 0;
  }
};

struct MTGRPartialBatchSetup {
  xllm::layer::AttentionMetadata metadata;
  xllm::KVCache kv_cache;
};

struct MTGRAttentionHarnessMetadata {
  int64_t pair_id = 0;
  int64_t num_heads = 8;
  int64_t num_kv_heads = 8;
  int64_t head_dim = 128;
  int64_t history = 1351;
  int64_t context = 8;
  int64_t realtime = 401;
  int64_t target = 801;
  int64_t matched_prefix = 0;
  int64_t block_size = 128;

  int64_t total_len() const { return history + context + realtime + target; }
  int64_t live_len() const { return total_len() - matched_prefix; }
  int64_t realtime_matched() const {
    return matched_prefix > history + context
               ? matched_prefix - history - context
               : 0;
  }
  bool is_partial_match() const { return matched_prefix > 0; }
  const char* mode_name() const {
    return is_partial_match() ? "partial_match" : "no_match";
  }
};

struct MTGRAttentionInput {
  xllm::layer::AttentionMetadata metadata;
  torch::Tensor query;
  torch::Tensor key;
  torch::Tensor value;
  xllm::KVCache kv_cache;
};

struct MTGRAttentionCaseData {
  MTGRAttentionHarnessMetadata harness_metadata;

  // Full logical input is benchmark-only. It keeps the performance base at the
  // full no-prefix compute amount even when the Hopper path receives live
  // tokens after prefix match.
  MTGRAttentionInput full_input;

  // Live input mirrors production prefix-match semantics.
  MTGRAttentionInput live_input;
};

struct MTGRAttentionDiff {
  bool all_finite = false;
  double max_abs = 0.0;
  double mean_abs = 0.0;
};

class IMTGRAttentionBackend {
 public:
  virtual ~IMTGRAttentionBackend() = default;

  virtual std::string name() const = 0;
  virtual std::string nvtx_root_name(
      const MTGRAttentionHarnessMetadata& metadata) const = 0;

  virtual std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward(
      const xllm::layer::AttentionMetadata& attn_metadata,
      torch::Tensor& query,
      torch::Tensor& key,
      torch::Tensor& value,
      xllm::KVCache& kv_cache) = 0;
};

std::vector<MTGRAttentionHarnessMetadata> generate_odd_length_metadata_pairs(
    int64_t pair_count,
    uint64_t seed);

MTGRAttentionCaseData build_case_data(
    const MTGRAttentionHarnessMetadata& metadata,
    const torch::Device& device);

std::unique_ptr<IMTGRAttentionBackend> make_full_flashinfer_base_backend(
    const MTGRAttentionHarnessMetadata& metadata);
std::unique_ptr<IMTGRAttentionBackend> make_hopper_unified_backend(
    const MTGRAttentionHarnessMetadata& metadata);

xllm::layer::AttentionMetadata make_mtgr_attention_metadata(
    const MTGRAttentionTestShape& shape,
    const torch::Device& device,
    int64_t block_size = 128);

xllm::layer::AttentionMetadata make_mtgr_attention_metadata(
    const std::vector<MTGRAttentionTestShape>& shapes,
    const torch::Device& device,
    int64_t block_size = 128);

xllm::KVCache make_mtgr_kv_cache(const MTGRAttentionTestShape& shape,
                                 const torch::Device& device,
                                 torch::ScalarType dtype,
                                 int64_t block_size = 128);

void prefill_mtgr_matched_prefix_cache(const torch::Tensor& full_key_bsnd,
                                       const torch::Tensor& full_value_bsnd,
                                       const MTGRAttentionTestShape& shape,
                                       int64_t block_size,
                                       xllm::KVCache& kv_cache);

MTGRPartialBatchSetup make_mtgr_partial_batch_setup(
    const std::vector<MTGRAttentionTestShape>& shapes,
    const std::vector<torch::Tensor>& full_key_bsnd,
    const std::vector<torch::Tensor>& full_value_bsnd,
    const torch::Device& device,
    torch::ScalarType dtype,
    int64_t block_size = 128);

torch::Tensor build_mtgr_torch_full_visible_mask(
    const MTGRAttentionTestShape& shape,
    const torch::Device& device);

torch::Tensor run_mtgr_torch_mask_attention_reference(
    const torch::Tensor& full_query_bshd,
    const torch::Tensor& full_key_bshd,
    const torch::Tensor& full_value_bshd,
    const MTGRAttentionTestShape& shape,
    double sm_scale);

MTGRAttentionDiff compare_outputs(const torch::Tensor& reference,
                                  const torch::Tensor& candidate);

void write_perf_label_header(std::ostream& out);
void write_perf_label_row(std::ostream& out,
                          int64_t idx,
                          const std::string& backend,
                          const MTGRAttentionHarnessMetadata& metadata,
                          int64_t repeat_id);

}  // namespace xllm::kernel::cuda::test::mtgr_attention_harness
