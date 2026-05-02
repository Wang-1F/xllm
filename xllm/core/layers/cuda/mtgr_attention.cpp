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

#include "mtgr_attention.h"

#include <glog/logging.h>

#include <memory>
#include <utility>

#include "kernels/cuda/tests/mtgr_attenion_test.h"

namespace xllm {
namespace layer {
namespace {

kernel::cuda::test::MTGRAttentionTestBackend to_kernel_backend(
    MTGRAttentionBackend backend) {
  switch (backend) {
    case MTGRAttentionBackend::kOneStage:
      return kernel::cuda::test::MTGRAttentionTestBackend::kOneStage;
    case MTGRAttentionBackend::kMultiStage:
      return kernel::cuda::test::MTGRAttentionTestBackend::kMultiStage;
    case MTGRAttentionBackend::kFused:
      return kernel::cuda::test::MTGRAttentionTestBackend::kFusedNoMatch;
  }
  CHECK(false) << "Unsupported MTGR backend";
  return kernel::cuda::test::MTGRAttentionTestBackend::kFusedNoMatch;
}

MTGRAttentionMetrics to_layer_metrics(
    const kernel::cuda::test::MTGRAttentionTestMetrics& metrics) {
  MTGRAttentionMetrics converted;
  converted.mask_build_ms = metrics.mask_build_ms;
  converted.h2d_ms = metrics.h2d_ms;
  converted.workspace_ms = metrics.workspace_ms;
  converted.fia_ms = metrics.fia_ms;
  converted.device_total_ms = metrics.device_total_ms;
  converted.wall_total_ms = metrics.wall_total_ms;
  converted.stages.reserve(metrics.stages.size());
  for (const auto& stage : metrics.stages) {
    converted.stages.push_back(MTGRStageMetric{
        .name = stage.name,
        .workspace_ms = stage.workspace_ms,
        .exec_ms = stage.exec_ms,
        .host_submit_ms = stage.host_submit_ms,
    });
  }
  return converted;
}

}  // namespace

struct MTGRAttentionImpl::ForwardImpl {
  ForwardImpl(int64_t num_heads,
              int64_t head_size,
              float scale,
              int64_t num_kv_heads,
              MTGRAttentionBackend backend)
      : kernel_impl(num_heads,
                    head_size,
                    scale,
                    num_kv_heads,
                    to_kernel_backend(backend)) {}

  kernel::cuda::test::MTGRAttentionImplTest kernel_impl;
};

MTGRAttentionImpl::MTGRAttentionImpl(int64_t num_heads,
                                     int64_t head_size,
                                     float scale,
                                     int64_t num_kv_heads,
                                     MTGRAttentionBackend backend)
    : num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads),
      backend_(backend) {
  rebuild_impl();
}

MTGRAttentionImpl::~MTGRAttentionImpl() = default;

void MTGRAttentionImpl::rebuild_impl() {
  if (num_heads_ <= 0 || head_size_ <= 0 || num_kv_heads_ <= 0 ||
      scale_ <= 0.0f) {
    impl_.reset();
    return;
  }
  impl_ = std::make_unique<ForwardImpl>(
      num_heads_, head_size_, scale_, num_kv_heads_, backend_);
}

void MTGRAttentionImpl::set_backend(MTGRAttentionBackend backend) {
  if (backend_ == backend) {
    return;
  }
  backend_ = backend;
  rebuild_impl();
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>>
MTGRAttentionImpl::forward(const AttentionMetadata& attn_metadata,
                           torch::Tensor& query,
                           torch::Tensor& key,
                           torch::Tensor& value,
                           KVCache& kv_cache) {
  CHECK(impl_ != nullptr)
      << "MTGRAttentionImpl is not initialized with valid attention dims";
  auto result =
      impl_->kernel_impl.forward(attn_metadata, query, key, value, kv_cache);
  last_metrics_ = to_layer_metrics(impl_->kernel_impl.last_metrics());
  return result;
}

}  // namespace layer
}  // namespace xllm
