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
#include <string>
#include <tuple>
#include <vector>

#include "framework/kv_cache/kv_cache.h"
#include "layers/common/attention_metadata.h"

namespace xllm {
namespace layer {

enum class MTGRAttentionBackend {
  kOneStage,
  kMultiStage,
  kFused,
};

struct MTGRStageMetric {
  std::string name;
  double workspace_ms = 0.0;
  double exec_ms = 0.0;
  double host_submit_ms = 0.0;
};

struct MTGRAttentionMetrics {
  double mask_build_ms = 0.0;
  double h2d_ms = 0.0;
  double workspace_ms = 0.0;
  double fia_ms = 0.0;
  double device_total_ms = 0.0;
  double wall_total_ms = 0.0;
  std::vector<MTGRStageMetric> stages;
};

class MTGRAttentionImpl : public torch::nn::Module {
 public:
  MTGRAttentionImpl() = default;
  MTGRAttentionImpl(
      int64_t num_heads,
      int64_t head_size,
      float scale,
      int64_t num_kv_heads,
      MTGRAttentionBackend backend = MTGRAttentionBackend::kFused);
  ~MTGRAttentionImpl() override;

  std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward(
      const AttentionMetadata& attn_metadata,
      torch::Tensor& query,
      torch::Tensor& key,
      torch::Tensor& value,
      KVCache& kv_cache);

  void set_backend(MTGRAttentionBackend backend);

  const MTGRAttentionMetrics& last_metrics() const { return last_metrics_; }

  int64_t num_heads_ = 0;
  int64_t head_size_ = 0;
  float scale_ = 1.0f;
  int64_t num_kv_heads_ = 0;

 private:
  struct ForwardImpl;

  void rebuild_impl();

  MTGRAttentionBackend backend_ = MTGRAttentionBackend::kFused;
  MTGRAttentionMetrics last_metrics_;
  std::unique_ptr<ForwardImpl> impl_;
};
TORCH_MODULE(MTGRAttention);

}  // namespace layer
}  // namespace xllm
