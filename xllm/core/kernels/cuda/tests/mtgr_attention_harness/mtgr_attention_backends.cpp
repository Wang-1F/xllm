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

#include "mtgr_attention_contract.h"

#include <cuda_runtime.h>
#include <glog/logging.h>

#include <cmath>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>

#include "core/common/global_flags.h"
#include "core/util/mtgr_nvtx.h"
#include "layers/cuda/mtgr_attention.h"
#include "../mtgr_attenion_test.h"

namespace xllm::kernel::cuda::test::mtgr_attention_harness {
namespace {

using xllm::layer::MTGRAttentionBackend;
using xllm::layer::MTGRAttentionImpl;

class FullFlashinferBaseBackend final : public IMTGRAttentionBackend {
 public:
  explicit FullFlashinferBaseBackend(MTGRAttentionHarnessMetadata metadata)
      : metadata_(std::move(metadata)) {}

  std::string name() const override { return "full_flashinfer_base"; }

  std::string nvtx_root_name(
      const MTGRAttentionHarnessMetadata& metadata) const override {
    return std::string("MTGR/harness/mtgr_attention/base/") +
           metadata.mode_name();
  }

  std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward(
      const xllm::layer::AttentionMetadata& attn_metadata,
      torch::Tensor& query,
      torch::Tensor& key,
      torch::Tensor& value,
      xllm::KVCache& kv_cache) override {
    (void)kv_cache;
    (void)attn_metadata;
    const double scale =
        1.0 / std::sqrt(static_cast<double>(metadata_.head_dim));
    auto full_out = run_mtgr_one_stage_full_base_nvtx_only(
        query.view(
            {metadata_.total_len(), metadata_.num_heads, metadata_.head_dim}),
        key.view({metadata_.total_len(),
                  metadata_.num_kv_heads,
                  metadata_.head_dim}),
        value.view({metadata_.total_len(),
                    metadata_.num_kv_heads,
                    metadata_.head_dim}),
        metadata_.history,
        metadata_.context,
        metadata_.realtime,
        metadata_.target,
        scale,
        FLAGS_mtgr_nvtx_level >= 2);
    return {full_out
                .view({metadata_.total_len(),
                       metadata_.num_heads * metadata_.head_dim})
                .contiguous(),
            std::nullopt};
  }

 private:
  MTGRAttentionHarnessMetadata metadata_;
};

class HopperUnifiedBackend final : public IMTGRAttentionBackend {
 public:
  explicit HopperUnifiedBackend(MTGRAttentionHarnessMetadata metadata)
      : metadata_(std::move(metadata)) {}

  std::string name() const override { return "hopper_unified"; }

  std::string nvtx_root_name(
      const MTGRAttentionHarnessMetadata& metadata) const override {
    return std::string("MTGR/harness/mtgr_attention/hopper/") +
           metadata.mode_name();
  }

  std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward(
      const xllm::layer::AttentionMetadata& attn_metadata,
      torch::Tensor& query,
      torch::Tensor& key,
      torch::Tensor& value,
      xllm::KVCache& kv_cache) override {
    const float scale =
        1.0f / std::sqrt(static_cast<float>(metadata_.head_dim));
    MTGRAttentionImpl attention(metadata_.num_heads,
                                metadata_.head_dim,
                                scale,
                                metadata_.num_kv_heads,
                                MTGRAttentionBackend::kFused);
    xllm::MtgrNvtxRange range(2, "MTGR/harness/mtgr_attention/hopper_forward");
    auto result = attention.forward(attn_metadata, query, key, value, kv_cache);
    {
      xllm::MtgrNvtxRange sync_range(
          2, "MTGR/harness/mtgr_attention/device_sync");
      CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }
    return result;
  }

 private:
  MTGRAttentionHarnessMetadata metadata_;
};

}  // namespace

std::unique_ptr<IMTGRAttentionBackend> make_full_flashinfer_base_backend(
    const MTGRAttentionHarnessMetadata& metadata) {
  return std::make_unique<FullFlashinferBaseBackend>(metadata);
}

std::unique_ptr<IMTGRAttentionBackend> make_hopper_unified_backend(
    const MTGRAttentionHarnessMetadata& metadata) {
  return std::make_unique<HopperUnifiedBackend>(metadata);
}

}  // namespace xllm::kernel::cuda::test::mtgr_attention_harness
