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

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "framework/kv_cache/kv_cache.h"
#include "layers/common/attention_metadata.h"

namespace xllm::kernel::npu::test {

struct MTGRCaseConfig {
  int64_t batch_size = 1;
  int64_t num_heads = 8;
  int64_t num_kv_heads = 8;
  int64_t head_dim = 128;
  int64_t history = 1300;
  int64_t context = 8;
  int64_t real_time = 400;
  int64_t target = 800;
  int64_t matched_prefix = 0;
  int64_t block_size = 128;

  int64_t total_len() const {
    return history + context + real_time + target;
  }
};

struct MTGRPreparedCase {
  MTGRCaseConfig cfg;
  torch::Tensor query_case_flat;
  torch::Tensor key_case_flat;
  torch::Tensor value_case_flat;
  torch::Tensor query_full_bsnd;
  torch::Tensor key_full_bsnd;
  torch::Tensor value_full_bsnd;
  torch::Tensor key_cache;
  torch::Tensor value_cache;
  xllm::layer::AttentionMetadata attn_metadata;
};

struct OneStagePerf {
  double mask_build_ms = 0.0;
  double h2d_ms = 0.0;
  double workspace_ms = 0.0;
  double fia_ms = 0.0;
  double device_total_ms = 0.0;
  double wall_total_ms = 0.0;
};

struct MTGRStagePerf {
  std::string name;
  double workspace_ms = 0.0;
  double exec_ms = 0.0;
};

struct MTGRForwardPerf {
  std::string path_name;
  std::vector<MTGRStagePerf> stages;
  double device_total_ms = 0.0;
  double wall_total_ms = 0.0;
};

std::vector<int32_t> make_block_table(int64_t block_count);
std::vector<int32_t> build_slot_mapping(const std::vector<int32_t>& block_table,
                                        int64_t block_size,
                                        int64_t start_token_idx,
                                        int64_t token_count);
int64_t partial_rt_matched_tokens(int64_t realtime_tokens);

MTGRPreparedCase prepare_mtgr_case(const MTGRCaseConfig& cfg,
                                   const torch::TensorOptions& fp_opts,
                                   const torch::Device& device);

OneStagePerf profile_one_stage_maskopt(const MTGRPreparedCase& prepared_case,
                                       int warmup_iters,
                                       int repeat_iters);

class MTGRAttentionTestImpl {
 public:
  MTGRAttentionTestImpl(int64_t num_heads,
                        int64_t head_size,
                        float scale,
                        int64_t num_kv_heads);
  ~MTGRAttentionTestImpl();

  void set_forward_perf_sink(MTGRForwardPerf* perf_sink);
  void set_use_fused_target_update(bool use_fused_target_update);
  void set_use_fused_target_update_auto(bool use_fused_target_update_auto);
  void set_use_fused_target_update_v4(bool use_fused_target_update_v4);
  void set_use_current_best_target_update(bool use_current_best_target_update);

  std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward(
      const xllm::layer::AttentionMetadata& attn_metadata,
      torch::Tensor& query,
      torch::Tensor& key,
      torch::Tensor& value,
      xllm::KVCache& kv_cache);

 private:
  int64_t num_heads_ = 0;
  int64_t head_size_ = 0;
  float scale_ = 0.0f;
  int64_t num_kv_heads_ = 0;
  MTGRForwardPerf* perf_sink_ = nullptr;
  torch::Tensor compressed_causal_mask_;
  bool use_fused_target_update_ = false;
  bool use_fused_target_update_auto_ = false;
  bool use_fused_target_update_v4_ = false;
  bool use_current_best_target_update_ = true;
  void* shared_workspace_ = nullptr;
  uint64_t shared_workspace_size_ = 0;
  void* scatter_workspace_ = nullptr;
  uint64_t scatter_workspace_size_ = 0;
  int32_t workspace_device_index_ = -1;
};

MTGRForwardPerf profile_mtgr_forward(MTGRPreparedCase& prepared_case,
                                     int warmup_iters,
                                     int repeat_iters,
                                     bool use_fused_target_update = false,
                                     bool use_fused_target_update_auto = false,
                                     bool use_fused_target_update_v4 = false);

}  // namespace xllm::kernel::npu::test
