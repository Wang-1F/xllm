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

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <torch/cuda.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdio>
#include <memory>
#include <unordered_map>

#include "common/global_flags.h"
#include "framework/state_dict/state_dict.h"
#include "kernels/cuda/cuda_ops_api.h"
#include "kernels/cuda/tests/mtgr_attenion_test.h"
#include "layers/common/linear.h"
#include "layers/common/partial_rotary_embedding.h"
#include "layers/common/qwen3_next_rms_norm.h"
#include "layers/common/tests/tests_utils.h"
#include "layers/cuda/custom_mask_attention.h"
#include "layers/cuda/mtgr_attention.h"

namespace xllm::layer::test {
namespace {

constexpr int64_t kBlockSize = 128;
constexpr int64_t kHeads = 8;
constexpr int64_t kKVHeads = 8;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kHiddenSize = kHeads * kHeadDim;
constexpr double kRmsNormEps = 1.0e-6;
constexpr double kMaxAbs = 1.0e-3;
constexpr double kMeanAbs = 1.0e-5;

class CustomMaskAttentionBaselineImpl final : public torch::nn::Module {
 public:
  CustomMaskAttentionBaselineImpl(const ModelArgs& args,
                                  const QuantArgs& quant_args,
                                  const ParallelArgs& parallel_args,
                                  const torch::TensorOptions& options,
                                  int32_t layer_id) {
    const int64_t total_num_heads = args.n_heads();
    const int64_t total_num_kv_heads =
        args.n_kv_heads().value_or(args.n_heads());
    const int64_t num_kv_head_replicas = 1;
    layer_id_ = layer_id;
    num_heads_ = total_num_heads;
    num_kv_heads_ = total_num_kv_heads;

    head_dim_ = args.head_dim();
    q_size_ = num_heads_ * head_dim_;
    kv_size_ = num_kv_heads_ * head_dim_;
    scaling_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    qkv_proj_ = register_module(
        "qkv_proj",
        QKVParallelLinear(args.hidden_size(),
                          num_heads_,
                          num_kv_heads_,
                          args.head_dim(),
                          num_kv_head_replicas,
                          /*bias=*/args.attention_bias(),
                          /*gather_output=*/false,
                          parallel_args,
                          options));

    o_proj_ = register_module("o_proj",
                              RowParallelLinear(total_num_heads * head_dim_,
                                                args.hidden_size(),
                                                /*bias=*/false,
                                                /*input_is_parallelized=*/true,
                                                /*enable_result_reduction=*/true,
                                                quant_args,
                                                parallel_args.tp_group_,
                                                options));

    q_norm_ = register_module(
        "q_norm", Qwen3NextRMSNorm(head_dim_, args.rms_norm_eps(), options));
    k_norm_ = register_module(
        "k_norm", Qwen3NextRMSNorm(head_dim_, args.rms_norm_eps(), options));

    const int rotary_dim =
        static_cast<int>(head_dim_ * args.partial_rotary_factor());
    rotary_emb_ =
        register_module("rotary_emb",
                        PartialRotaryEmbedding(rotary_dim,
                                               args.max_position_embeddings(),
                                               args.rope_theta(),
                                               head_dim_,
                                               true,
                                               false,
                                               options));

    attn_ = register_module("attn",
                            MTGRAttention(
                                num_heads_, head_dim_, scaling_, num_kv_heads_));
  }

  torch::Tensor forward(const torch::Tensor& positions,
                        const torch::Tensor& hidden_states,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache) {
    auto qkv = qkv_proj_->forward(hidden_states);
    auto q = qkv.slice(/*dim=*/-1, 0, q_size_);
    auto k = qkv.slice(/*dim=*/-1, q_size_, q_size_ + kv_size_);
    auto v = qkv.slice(/*dim=*/-1, q_size_ + kv_size_, q_size_ + 2 * kv_size_);

    const int64_t tokens = q.size(0);
    auto q_reshaped = q.reshape({tokens, num_heads_, head_dim_});
    auto q_normed = q_norm_->forward(q_reshaped);
    auto k_reshaped = k.reshape({tokens, num_kv_heads_, head_dim_});
    auto k_normed = k_norm_->forward(k_reshaped);
    q = q_normed.view({tokens, q_size_});
    k = k_normed.view({tokens, kv_size_});

    (void)positions;
    auto out = std::get<0>(attn_->forward(attn_metadata, q, k, v, kv_cache));
    return o_proj_->forward(out);
  }

  void load_state_dict(const StateDict& state_dict) {
    qkv_proj_->load_state_dict(state_dict, {"q_proj.", "k_proj.", "v_proj."});
    o_proj_->load_state_dict(state_dict.get_dict_with_prefix("o_proj."));
    if (auto w = state_dict.get_tensor("q_norm.weight"); w.defined()) {
      q_norm_->load_state_dict(StateDict({{"weight", w}}));
    }
    if (auto w = state_dict.get_tensor("k_norm.weight"); w.defined()) {
      k_norm_->load_state_dict(StateDict({{"weight", w}}));
    }
  }

 private:
  int64_t num_heads_ = 0;
  int64_t num_kv_heads_ = 0;
  int64_t head_dim_ = 0;
  int64_t q_size_ = 0;
  int64_t kv_size_ = 0;
  float scaling_ = 1.0f;
  int32_t layer_id_ = 0;

  QKVParallelLinear qkv_proj_{nullptr};
  RowParallelLinear o_proj_{nullptr};
  Qwen3NextRMSNorm q_norm_{nullptr};
  Qwen3NextRMSNorm k_norm_{nullptr};
  MTGRAttention attn_{nullptr};
  PartialRotaryEmbedding rotary_emb_{nullptr};
};
TORCH_MODULE(CustomMaskAttentionBaseline);

ModelArgs make_model_args() {
  ModelArgs args;
  args.hidden_size() = kHiddenSize;
  args.intermediate_size() = kHiddenSize * 4;
  args.hidden_act() = "silu";
  args.head_dim() = kHeadDim;
  args.n_heads() = kHeads;
  args.n_kv_heads() = kKVHeads;
  args.rms_norm_eps() = static_cast<float>(kRmsNormEps);
  args.partial_rotary_factor() = 1.0f;
  args.max_position_embeddings() = 4096;
  args.rope_theta() = 10000.0f;
  args.attention_bias() = false;
  return args;
}

ParallelArgs make_parallel_args(std::unique_ptr<xllm::ProcessGroup>* pg) {
  *pg = std::make_unique<MockProcessGroup>(torch::Device(torch::kCUDA, 0));
  ParallelArgs parallel_args(0, 1, pg->get());
  parallel_args.tp_group_ = pg->get();
  parallel_args.single_rank_group_ = pg->get();
  parallel_args.sp_group_ = pg->get();
  return parallel_args;
}

StateDict make_state_dict(const torch::TensorOptions& options) {
  std::unordered_map<std::string, torch::Tensor> weights;
  weights.emplace("q_proj.weight",
                  torch::randn({kHeads * kHeadDim, kHiddenSize}, options) *
                      0.02);
  weights.emplace("k_proj.weight",
                  torch::randn({kKVHeads * kHeadDim, kHiddenSize}, options) *
                      0.02);
  weights.emplace("v_proj.weight",
                  torch::randn({kKVHeads * kHeadDim, kHiddenSize}, options) *
                      0.02);
  weights.emplace("o_proj.weight",
                  torch::randn({kHiddenSize, kHeads * kHeadDim}, options) *
                      0.02);
  weights.emplace("q_norm.weight", torch::randn({kHeadDim}, options) * 0.02);
  weights.emplace("k_norm.weight", torch::randn({kHeadDim}, options) * 0.02);
  return StateDict(std::move(weights));
}

torch::Tensor linear_no_bias(const torch::Tensor& input,
                             const torch::Tensor& weight) {
  return torch::matmul(input, weight.transpose(0, 1));
}

torch::Tensor qwen3_next_rms_norm_reference(const torch::Tensor& input,
                                            const torch::Tensor& weight,
                                            double eps) {
  auto input_dtype = input.dtype();
  auto input_fp32 = input.to(torch::kFloat32);
  auto variance = torch::mean(torch::pow(input_fp32, 2), -1, true);
  auto normalized = input_fp32 * torch::rsqrt(variance + eps);
  return (normalized * (1.0f + weight.to(torch::kFloat32))).to(input_dtype);
}

torch::Tensor qwen3_next_rms_norm_fast(const torch::Tensor& input,
                                       const torch::Tensor& weight,
                                       double eps) {
  auto output = torch::empty(input.sizes(), input.options());
  kernel::cuda::mtgr_qk_norm_strided_bf16_hd128(output,
                                                input.contiguous(),
                                                weight.contiguous(),
                                                eps);
  return output;
}

void build_prefix_kv_from_hidden(const torch::Tensor& full_hidden_states,
                                 const StateDict& state_dict,
                                 bool use_fast_qk_norm,
                                 torch::Tensor* full_key_bsnd,
                                 torch::Tensor* full_value_bsnd) {
  auto k_proj = linear_no_bias(full_hidden_states,
                               state_dict.get_tensor("k_proj.weight"));
  auto v_proj = linear_no_bias(full_hidden_states,
                               state_dict.get_tensor("v_proj.weight"));
  auto key_snd = k_proj.view({full_hidden_states.size(0), kKVHeads, kHeadDim});
  auto value_snd =
      v_proj.view({full_hidden_states.size(0), kKVHeads, kHeadDim});
  auto norm_weight = state_dict.get_tensor("k_norm.weight");
  auto normalized_key = use_fast_qk_norm
                            ? qwen3_next_rms_norm_fast(
                                  key_snd, norm_weight, kRmsNormEps)
                            : qwen3_next_rms_norm_reference(
                                  key_snd, norm_weight, kRmsNormEps);
  *full_key_bsnd = normalized_key.unsqueeze(0).contiguous();
  *full_value_bsnd = value_snd.unsqueeze(0).contiguous();
}

void expect_close(const char* tag,
                  const torch::Tensor& baseline,
                  const torch::Tensor& integrated) {
  auto baseline_fp32 = baseline.to(torch::kCPU).to(torch::kFloat32);
  auto integrated_fp32 = integrated.to(torch::kCPU).to(torch::kFloat32);
  auto diff = (baseline_fp32 - integrated_fp32).abs();
  const double max_abs = diff.max().item<double>();
  const double mean_abs = diff.mean().item<double>();
  const bool baseline_finite = torch::isfinite(baseline_fp32).all().item<bool>();
  const bool integrated_finite =
      torch::isfinite(integrated_fp32).all().item<bool>();

  std::fprintf(stderr,
               "[MTGR][CUDA][CustomMaskAttentionIntegration][%s] "
               "max_abs=%.6e mean_abs=%.6e baseline_finite=%s "
               "integrated_finite=%s\n",
               tag,
               max_abs,
               mean_abs,
               baseline_finite ? "true" : "false",
               integrated_finite ? "true" : "false");
  std::fflush(stderr);

  EXPECT_TRUE(baseline_finite);
  EXPECT_TRUE(integrated_finite);
  EXPECT_LT(max_abs, kMaxAbs);
  EXPECT_LT(mean_abs, kMeanAbs);
}

class CustomMaskAttentionIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA is not available";
    }
    cudaDeviceProp prop{};
    ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
    if (prop.major < 9) {
      GTEST_SKIP() << "MTGR Hopper fused path requires SM90+";
    }
    FLAGS_block_size = kBlockSize;
  }
};

TEST_F(CustomMaskAttentionIntegrationTest,
       NoMatchProductionIntegrationMatchesBaselineLayer) {
  torch::NoGradGuard no_grad_guard;
  torch::manual_seed(20260518);

  const auto device = torch::Device(torch::kCUDA, 0);
  auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  auto model_args = make_model_args();
  QuantArgs quant_args;
  std::unique_ptr<xllm::ProcessGroup> process_group;
  auto parallel_args = make_parallel_args(&process_group);
  auto state_dict = make_state_dict(options);

  kernel::cuda::test::MTGRAttentionTestShape shape;
  shape.heads = kHeads;
  shape.kv_heads = kKVHeads;
  shape.head_dim = kHeadDim;
  shape.history = 257;
  shape.context = 9;
  shape.realtime = 73;
  shape.target = 129;
  shape.matched_prefix = 0;

  auto metadata =
      kernel::cuda::test::make_mtgr_attention_metadata(shape, device, kBlockSize);
  auto kv_cache = kernel::cuda::test::make_mtgr_kv_cache(
      shape, device, torch::kBFloat16, kBlockSize);
  auto positions = torch::arange(shape.local_len(),
                                 torch::TensorOptions()
                                     .dtype(torch::kInt64)
                                     .device(device));
  auto hidden_states =
      torch::randn({shape.local_len(), kHiddenSize}, options) * 0.05;

  CustomMaskAttentionBaseline baseline(
      model_args, quant_args, parallel_args, options, /*layer_id=*/0);
  baseline->load_state_dict(state_dict);
  CustomMaskAttention integrated(
      model_args, quant_args, parallel_args, options, /*layer_id=*/0);
  integrated->load_state_dict(state_dict);

  auto baseline_out =
      baseline->forward(positions, hidden_states, metadata, kv_cache);
  auto integrated_out =
      integrated->forward(positions, hidden_states, metadata, kv_cache);

  expect_close("no_match", baseline_out, integrated_out);
}

TEST_F(CustomMaskAttentionIntegrationTest,
       PartialMatchProductionIntegrationMatchesBaselineLayer) {
  torch::NoGradGuard no_grad_guard;
  torch::manual_seed(20260519);

  const auto device = torch::Device(torch::kCUDA, 0);
  auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  auto model_args = make_model_args();
  QuantArgs quant_args;
  std::unique_ptr<xllm::ProcessGroup> process_group;
  auto parallel_args = make_parallel_args(&process_group);
  auto state_dict = make_state_dict(options);

  kernel::cuda::test::MTGRAttentionTestShape shape;
  shape.heads = kHeads;
  shape.kv_heads = kKVHeads;
  shape.head_dim = kHeadDim;
  shape.history = 257;
  shape.context = 9;
  shape.realtime = 73;
  shape.target = 129;
  shape.matched_prefix = shape.history + shape.context + 41;

  auto metadata =
      kernel::cuda::test::make_mtgr_attention_metadata(shape, device, kBlockSize);
  auto baseline_kv_cache = kernel::cuda::test::make_mtgr_kv_cache(
      shape, device, torch::kBFloat16, kBlockSize);
  auto integrated_kv_cache = kernel::cuda::test::make_mtgr_kv_cache(
      shape, device, torch::kBFloat16, kBlockSize);

  auto full_hidden_states =
      torch::randn({shape.total_len(), kHiddenSize}, options) * 0.05;
  torch::Tensor baseline_full_key;
  torch::Tensor baseline_full_value;
  build_prefix_kv_from_hidden(
      full_hidden_states, state_dict, /*use_fast_qk_norm=*/false,
      &baseline_full_key, &baseline_full_value);
  torch::Tensor integrated_full_key;
  torch::Tensor integrated_full_value;
  build_prefix_kv_from_hidden(
      full_hidden_states, state_dict, /*use_fast_qk_norm=*/true,
      &integrated_full_key, &integrated_full_value);

  kernel::cuda::test::prefill_mtgr_matched_prefix_cache(
      baseline_full_key,
      baseline_full_value,
      shape,
      kBlockSize,
      baseline_kv_cache);
  kernel::cuda::test::prefill_mtgr_matched_prefix_cache(
      integrated_full_key,
      integrated_full_value,
      shape,
      kBlockSize,
      integrated_kv_cache);

  auto live_hidden_states =
      full_hidden_states.narrow(0, shape.matched_prefix, shape.local_len())
          .contiguous();
  auto positions = torch::arange(shape.local_len(),
                                 torch::TensorOptions()
                                     .dtype(torch::kInt64)
                                     .device(device));

  CustomMaskAttentionBaseline baseline(
      model_args, quant_args, parallel_args, options, /*layer_id=*/0);
  baseline->load_state_dict(state_dict);
  CustomMaskAttention integrated(
      model_args, quant_args, parallel_args, options, /*layer_id=*/0);
  integrated->load_state_dict(state_dict);

  auto baseline_out = baseline->forward(
      positions, live_hidden_states, metadata, baseline_kv_cache);
  auto integrated_out = integrated->forward(
      positions, live_hidden_states, metadata, integrated_kv_cache);

  expect_close("partial_match", baseline_out, integrated_out);
}

}  // namespace
}  // namespace xllm::layer::test
