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
#include <tuple>

#include "framework/kv_cache/kv_cache.h"
#include "layers/common/attention_metadata.h"

namespace xllm::kernel::cuda::test {

// Test-side product-facing MTGR attention adapter.
//
// This intentionally mirrors the NPU-facing contract:
//   forward(attn_metadata, query, key, value, kv_cache)
//
// The implementation is still test/research scoped. It adapts production-like
// AttentionMetadata into the current Hopper unified research kernel launch
// metadata so the project-facing API can be validated before wiring the real
// CUDA layer.
class MTGRAttentionTestImpl {
 public:
  MTGRAttentionTestImpl(int64_t num_heads,
                        int64_t head_size,
                        float scale,
                        int64_t num_kv_heads);

  std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward(
      const xllm::layer::AttentionMetadata& attn_metadata,
      torch::Tensor& query,
      torch::Tensor& key,
      torch::Tensor& value,
      xllm::KVCache& kv_cache);

 private:
  int64_t num_heads_ = 0;
  int64_t head_size_ = 0;
  float scale_ = 1.0f;
  int64_t num_kv_heads_ = 0;
};

}  // namespace xllm::kernel::cuda::test
