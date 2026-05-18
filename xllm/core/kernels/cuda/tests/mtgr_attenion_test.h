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

#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "framework/kv_cache/kv_cache.h"
#include "layers/common/attention_metadata.h"

namespace xllm::kernel::cuda::test {

enum class MTGRAttentionTestBackend {
  kOneStage,
  // Historical stable four-segment wrapper path.
  kMultiStage,
  // Historical fused no-match specialization.
  kFusedNoMatch,
};

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

struct MTGRStageMetric {
  std::string name;
  double workspace_ms = 0.0;
  double exec_ms = 0.0;
  double host_submit_ms = 0.0;
};

struct MTGRAttentionTestMetrics {
  double mask_build_ms = 0.0;
  double h2d_ms = 0.0;
  double workspace_ms = 0.0;
  double fia_ms = 0.0;
  double device_total_ms = 0.0;
  double wall_total_ms = 0.0;
  std::vector<MTGRStageMetric> stages;
};

struct MTGRPartialBatchSetup {
  xllm::layer::AttentionMetadata metadata;
  xllm::KVCache kv_cache;
};

class MTGRAttentionImplTest {
 public:
  MTGRAttentionImplTest() = default;
  MTGRAttentionImplTest(int64_t num_heads,
                        int64_t head_size,
                        float scale,
                        int64_t num_kv_heads,
                        MTGRAttentionTestBackend backend);

  std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward(
      const xllm::layer::AttentionMetadata& attn_metadata,
      torch::Tensor& query,
      torch::Tensor& key,
      torch::Tensor& value,
      xllm::KVCache& kv_cache);

  const MTGRAttentionTestMetrics& last_metrics() const { return last_metrics_; }

 private:
  int64_t num_heads_ = 0;
  int64_t head_size_ = 0;
  float scale_ = 1.0f;
  int64_t num_kv_heads_ = 0;
  MTGRAttentionTestBackend backend_ = MTGRAttentionTestBackend::kMultiStage;
  MTGRAttentionTestMetrics last_metrics_;
};

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

std::vector<MTGRAttentionTestShape> build_mtgr_sweep_shapes(bool full_sweep,
                                                            bool partial_match);

int env_int(const char* name, int default_value);
bool env_flag_enabled(const char* name, bool default_value);

torch::Tensor run_four_segment_no_match_batched_for_test(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    const std::vector<std::vector<int64_t>>& segment_lens,
    const std::vector<int64_t>& segment_rules,
    double sm_scale,
    MTGRAttentionTestMetrics* metrics = nullptr);

torch::Tensor run_stable_ragged_segment_attention_batched_for_test(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    const std::vector<std::vector<int64_t>>& segment_lens,
    const std::vector<int64_t>& segment_rules,
    double sm_scale,
    MTGRAttentionTestMetrics* metrics = nullptr);

// Harness bridge for the performance base: full logical mask build plus one
// FlashInfer call, with NVTX scopes but without cudaEvent/chrono metrics.
torch::Tensor run_mtgr_one_stage_full_base_nvtx_only(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    int64_t history_len,
    int64_t context_len,
    int64_t realtime_len,
    int64_t target_len,
    double sm_scale,
    bool emit_nvtx);

}  // namespace xllm::kernel::cuda::test
