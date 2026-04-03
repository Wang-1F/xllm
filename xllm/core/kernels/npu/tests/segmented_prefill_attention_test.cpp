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

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/torch_npu.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_fused_infer_attention_score_v3.h"

namespace xllm::kernel::npu::test {
namespace util {

// ---------------------------------------------------------------------------
// ACL dtype mapping
// ---------------------------------------------------------------------------
aclDataType ToAclDtype(torch::ScalarType dtype) {
  switch (dtype) {
    case torch::kFloat16:
      return ACL_FLOAT16;
    case torch::kFloat32:
      return ACL_FLOAT;
    case torch::kBFloat16:
      return ACL_BF16;
    case torch::kBool:
      return ACL_BOOL;
    case torch::kInt32:
      return ACL_INT32;
    case torch::kInt64:
      return ACL_INT64;
    case torch::kInt8:
      return ACL_INT8;
    default:
      CHECK(false) << "Unsupported dtype: " << dtype;
      return ACL_DT_UNDEFINED;
  }
}

// ---------------------------------------------------------------------------
// Create aclTensor from torch::Tensor (borrows the device data pointer).
// The torch::Tensor MUST be contiguous and on NPU, and must outlive the
// returned aclTensor.
// ---------------------------------------------------------------------------
aclTensor* TorchToAclTensor(const torch::Tensor& t) {
  CHECK(t.is_contiguous()) << "TorchToAclTensor requires contiguous tensor";
  auto shape = t.sizes().vec();
  auto stride = t.strides().vec();
  aclTensor* acl_t = aclCreateTensor(
      shape.data(), shape.size(), ToAclDtype(t.scalar_type()), stride.data(),
      0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), t.data_ptr());
  CHECK_NE(acl_t, nullptr)
      << "aclCreateTensor failed for tensor with shape [" << t.sizes()
      << "], dtype=" << t.scalar_type();
  return acl_t;
}

// ---------------------------------------------------------------------------
// Wall-clock timing with device synchronisation.
// ---------------------------------------------------------------------------
double MeasureWallClockMs(const std::function<void()>& fn,
                          int32_t device_id,
                          int warmup_iters = 5,
                          int measure_iters = 50) {
  CHECK_GT(measure_iters, 0);
  CHECK_GE(warmup_iters, 0);

  const aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  for (int i = 0; i < warmup_iters; ++i) {
    fn();
  }
  CHECK_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);

  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < measure_iters; ++i) {
    fn();
  }
  CHECK_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
  auto t1 = std::chrono::steady_clock::now();

  double elapsed_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  return elapsed_ms / static_cast<double>(measure_iters);
}

// ===========================================================================
// Test fixture (reused by genrec_attention_test.cpp)
// ===========================================================================
class SegmentedPrefillAttentionTest : public ::testing::Test {
 protected:
  static constexpr int32_t kDeviceId = 5;

  static void SetUpTestSuite() {
    torch_npu::init_npu("npu:" + std::to_string(kDeviceId));
  }

  static void TearDownTestSuite() { torch_npu::finalize_npu(); }

  void SetUp() override {
    device_ = torch::Device(torch::kPrivateUse1, kDeviceId);
    opts_ = torch::TensorOptions().dtype(torch::kFloat16).device(device_);
    stream_ = c10_npu::getCurrentNPUStream(kDeviceId).stream();
    CHECK_NE(stream_, nullptr) << "NPU stream is null for device " << kDeviceId;
  }

  torch::Device device_{torch::kCPU};
  torch::TensorOptions opts_;
  aclrtStream stream_{nullptr};
};

// ===========================================================================
// Correctness: verify output is finite and non-zero
// ===========================================================================
// TEST_F(SegmentedPrefillAttentionTest, CorrectnessBasic) {
//   constexpr int64_t kNumHeads = 8;
//   constexpr int64_t kNumKvHeads = 8;
//   constexpr int64_t kHeadDim = 64;
//   const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));

//   std::vector<Segment> segments = {{0, 10}, {30, 20}, {70, 15}};
//   auto ctxs =
//       BuildSegmentCtxs(kNumHeads, kNumKvHeads, kHeadDim, segments, opts_);
//   RunSegmentedPrefill(ctxs, kNumHeads, kNumKvHeads, scale, stream_);
//   CHECK_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);

//   for (size_t i = 0; i < segments.size(); ++i) {
//     const auto& seg = segments[i];
//     const auto& out = ctxs[i].out;

//     ASSERT_EQ(out.size(0), 1);
//     ASSERT_EQ(out.size(1), kNumHeads);
//     ASSERT_EQ(out.size(2), seg.length);
//     ASSERT_EQ(out.size(3), kHeadDim);

//     auto out_f32 = out.to(torch::kCPU).to(torch::kFloat32);
//     EXPECT_TRUE(torch::isfinite(out_f32).all().item<bool>())
//         << "Segment " << i << " (start=" << seg.start
//         << ", len=" << seg.length << ") has non-finite values";
//     EXPECT_GT(out_f32.abs().sum().item<float>(), 0.0f)
//         << "Segment " << i << " output is all zeros";
//   }
// }

// // ===========================================================================
// // Correctness: compare aclnn output with a PyTorch FP32 CPU reference
// // ===========================================================================
// TEST_F(SegmentedPrefillAttentionTest, CorrectnessVsReference) {
//   constexpr int64_t kSeqLen = 64;
//   constexpr int64_t kNumHeads = 8;
//   constexpr int64_t kNumKvHeads = 8;
//   constexpr int64_t kHeadDim = 64;
//   const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));

//   auto q = torch::randn({1, kNumHeads, kSeqLen, kHeadDim}, opts_);
//   auto k = torch::randn({1, kNumKvHeads, kSeqLen, kHeadDim}, opts_);
//   auto v = torch::randn({1, kNumKvHeads, kSeqLen, kHeadDim}, opts_);
//   auto mask = CreateCausalMask(kSeqLen, device_);
//   auto out = torch::empty_like(q);

//   SegmentCtx ctx{q, k, v, mask, out};
//   RunAclnnAttention(ctx, kNumHeads, kNumKvHeads, scale, stream_);
//   CHECK_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);

//   auto ref = ReferenceAttention(q, k, v, scale);
//   auto out_f32 = out.to(torch::kCPU).to(torch::kFloat32);
//   float max_diff = (out_f32 - ref).abs().max().item<float>();
//   EXPECT_LT(max_diff, 5e-2f)
//       << "aclnn output diverges from FP32 reference, max_diff=" << max_diff;
// }

// // ===========================================================================
// // Benchmarks – pipelined: pre-allocated workspace + reused ACL descriptors
// // ===========================================================================
// TEST_F(SegmentedPrefillAttentionTest, BenchFewLargeSegments) {
//   constexpr int64_t kNumHeads = 32;
//   constexpr int64_t kNumKvHeads = 8;
//   constexpr int64_t kHeadDim = 128;
//   const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));

//   std::vector<Segment> segments = {{0, 256}, {512, 512}, {1280, 256}};
//   auto ctxs =
//       BuildSegmentCtxs(kNumHeads, kNumKvHeads, kHeadDim, segments, opts_);
//   double ms = BenchmarkSegmentedPrefill(ctxs, kNumHeads, kNumKvHeads, scale,
//                                         kDeviceId, stream_, 5, 50);
//   LogBenchResult("FewLarge(3seg,1024tok)", segments, ms);
// }

// TEST_F(SegmentedPrefillAttentionTest, BenchMediumSegments) {
//   constexpr int64_t kNumHeads = 32;
//   constexpr int64_t kNumKvHeads = 8;
//   constexpr int64_t kHeadDim = 128;
//   const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));

//   std::vector<Segment> segments = {
//       {0, 128},    {256, 256},  {768, 128},
//       {1024, 256}, {1536, 128}, {1792, 256},
//   };
//   auto ctxs =
//       BuildSegmentCtxs(kNumHeads, kNumKvHeads, kHeadDim, segments, opts_);
//   double ms = BenchmarkSegmentedPrefill(ctxs, kNumHeads, kNumKvHeads, scale,
//                                         kDeviceId, stream_, 5, 50);
//   LogBenchResult("Medium(6seg,1152tok)", segments, ms);
// }

// TEST_F(SegmentedPrefillAttentionTest, BenchManySmallSegments) {
//   constexpr int64_t kNumHeads = 32;
//   constexpr int64_t kNumKvHeads = 8;
//   constexpr int64_t kHeadDim = 128;
//   const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));

//   std::vector<Segment> segments;
//   for (int i = 0; i < 32; ++i) {
//     segments.push_back({i * 256, 64});
//   }
//   auto ctxs =
//       BuildSegmentCtxs(kNumHeads, kNumKvHeads, kHeadDim, segments, opts_);
//   double ms = BenchmarkSegmentedPrefill(ctxs, kNumHeads, kNumKvHeads, scale,
//                                         kDeviceId, stream_, 5, 50);
//   LogBenchResult("ManySmall(32seg,2048tok)", segments, ms);
// }

// TEST_F(SegmentedPrefillAttentionTest, BenchLargeSegments) {
//   constexpr int64_t kNumHeads = 32;
//   constexpr int64_t kNumKvHeads = 8;
//   constexpr int64_t kHeadDim = 128;
//   const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));

//   std::vector<Segment> segments = {
//       {0, 1024},
//       {2048, 2048},
//       {6144, 1024},
//       {8192, 2048},
//   };
//   auto ctxs =
//       BuildSegmentCtxs(kNumHeads, kNumKvHeads, kHeadDim, segments, opts_);
//   double ms = BenchmarkSegmentedPrefill(ctxs, kNumHeads, kNumKvHeads, scale,
//                                         kDeviceId, stream_, 3, 20);
//   LogBenchResult("Large(4seg,6144tok)", segments, ms);
// }

// TEST_F(SegmentedPrefillAttentionTest, BenchGQAConfig) {
//   constexpr int64_t kNumHeads = 64;
//   constexpr int64_t kNumKvHeads = 8;
//   constexpr int64_t kHeadDim = 128;
//   const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));

//   std::vector<Segment> segments = {
//       {0, 256},
//       {512, 512},
//       {1280, 256},
//       {1792, 512},
//   };
//   auto ctxs =
//       BuildSegmentCtxs(kNumHeads, kNumKvHeads, kHeadDim, segments, opts_);
//   double ms = BenchmarkSegmentedPrefill(ctxs, kNumHeads, kNumKvHeads, scale,
//                                         kDeviceId, stream_, 5, 50);
//   LogBenchResult("GQA(4seg,1536tok,H64/KV8)", segments, ms);
// }

// ===========================================================================
// Generative Recommendation: 4-part heterogeneous attention
//
// Sequence: [history | context | real_time | target]
// Mask rules:
//   history:   causal (only attend to previous history tokens)
//   context:   full attention on all history + context
//   real_time: full on history+context, causal within real_time
//   target:    full on h+c+rt, each target token only sees itself
//
// Computation plan per batch (5 attn launches + 2 combines):
//   Step 1:  q=history, kv=history, causal
//   Step 2a: q=ctx+rt, kv=h+ctx, full  (output + lse)
//   Step 2b: q=rt, kv=rt, causal        (output + lse)
//   Combine: 2a[rt portion] ⊕ 2b → real_time final output
//   Step 3a: q=target, kv=h+ctx+rt, full (output + lse)
//   Step 3b: q=target, kv=target, diag   (output + lse)
//   Combine: 3a ⊕ 3b → target final output
// ===========================================================================

struct SeqPartLengths {
  int64_t history;
  int64_t context;
  int64_t real_time;
  int64_t target;
  int64_t total() const { return history + context + real_time + target; }
};

// Full-attention mask: all False (nothing masked).
// Used instead of nullptr because some CANN versions require a valid mask.
torch::Tensor CreateFullAttentionMask(int64_t q_len,
                                       int64_t kv_len,
                                       torch::Device device) {
  auto mask =
      torch::zeros({1, 1, q_len, kv_len}, torch::dtype(torch::kBool));
  return mask.to(device);
}

// Diagonal mask: position i can only attend to itself.
// True = masked, so everything except the diagonal is True.
torch::Tensor CreateDiagonalMask(int64_t seq_len, torch::Device device) {
  auto mask = ~torch::eye(seq_len, torch::dtype(torch::kBool));
  return mask.view({1, 1, seq_len, seq_len}).to(device);
}

// Compressed 2048×2048 lower-triangular mask for sparseMode 2 (leftUpCausal) /
// 3 (rightDownCausal). Required by CANN: shape must be [1,1,2048,2048] with
// upper triangle = True (masked). Reusable across any Sq/Skv <= 2048.
torch::Tensor CreateCompressedCausalMask2048(torch::Device device) {
  auto mask =
      torch::ones({1, 1, 2048, 2048}, torch::dtype(torch::kBool)).triu(1);
  return mask.to(device);
}

// Combine two flash-attention partial outputs using their log-sum-exp.
// o1, o2: [1, N, S, D],  lse1, lse2: [1, N, S, 1] (FP32).
// LSE is already 4D so it broadcasts directly with [1, N, S, D].
torch::Tensor CombineFlashOutputs(const torch::Tensor& o1,
                                   const torch::Tensor& lse1,
                                   const torch::Tensor& o2,
                                   const torch::Tensor& lse2) {
  auto max_l = torch::maximum(lse1, lse2);
  auto e1 = torch::exp(lse1 - max_l);
  auto e2 = torch::exp(lse2 - max_l);
  auto o1_f = o1.to(torch::kFloat32);
  auto o2_f = o2.to(torch::kFloat32);
  return ((o1_f * e1 + o2_f * e2) / (e1 + e2)).to(o1.scalar_type());
}

// Three-way merge (disjoint key partitions), same numerics as pairwise merge.
torch::Tensor CombineFlashOutputsThree(const torch::Tensor& o1,
                                        const torch::Tensor& lse1,
                                        const torch::Tensor& o2,
                                        const torch::Tensor& lse2,
                                        const torch::Tensor& o3,
                                        const torch::Tensor& lse3) {
  auto m = torch::maximum(torch::maximum(lse1, lse2), lse3);
  auto e1 = torch::exp(lse1 - m);
  auto e2 = torch::exp(lse2 - m);
  auto e3 = torch::exp(lse3 - m);
  auto o1_f = o1.to(torch::kFloat32);
  auto o2_f = o2.to(torch::kFloat32);
  auto o3_f = o3.to(torch::kFloat32);
  return ((o1_f * e1 + o2_f * e2 + o3_f * e3) / (e1 + e2 + e3)).to(o1.scalar_type());
}

// ---------------------------------------------------------------------------
// CPU FP32 reference: BNSD cross/self attention. mask [Sq, Skv] bool, true =
// masked position (add -inf before softmax). Empty optional = no mask.
// ---------------------------------------------------------------------------
torch::Tensor ReferenceAttentionBNSD(
    const torch::Tensor& q,   // [1, N, Sq, D] FP32 CPU
    const torch::Tensor& k,   // [1, Nkv, Skv, D]
    const torch::Tensor& v,   // [1, Nkv, Skv, D]
    const torch::Tensor* mask_sq_sk,  // [Sq, Skv] bool; nullptr = no mask
    double scale) {
  CHECK_EQ(q.dim(), 4);
  int64_t num_heads = q.size(1);
  int64_t num_kv = k.size(1);
  auto qf = q;
  auto kf = k;
  auto vf = v;
  if (num_kv < num_heads) {
    int64_t group = num_heads / num_kv;
    kf = kf.repeat_interleave(group, 1);
    vf = vf.repeat_interleave(group, 1);
  }
  auto scores = torch::matmul(qf, kf.transpose(-2, -1)) * scale;
  if (mask_sq_sk != nullptr) {
    scores = scores.masked_fill(*mask_sq_sk, -1e9f);
  }
  auto weights = torch::softmax(scores, -1);
  return torch::matmul(weights, vf);
}

// [S, N, D] -> [1, N, S, D] on same device/dtype (for reference path: CPU FP32).
torch::Tensor SeqToBNSD(const torch::Tensor& seq_s_n_d) {
  return seq_s_n_d.unsqueeze(0).permute({0, 2, 1, 3}).contiguous();
}

// Gold standard for GenRec: full attention under the same mask semantics as
// the 5-launch pipeline, computed directly on CPU FP32 (no merge formula).
struct GenRecReferenceOutputs {
  torch::Tensor history;   // [1, N, h, D]
  torch::Tensor context;   // [1, N, c, D]
  torch::Tensor real_time; // [1, N, r, D]
  torch::Tensor target;    // [1, N, t, D]
};

GenRecReferenceOutputs ReferenceGenRecGold(
    const torch::Tensor& q_seq,  // [S, N, D] FP32 CPU
    const torch::Tensor& k_seq,  // [S, Nkv, D]
    const torch::Tensor& v_seq,
    const SeqPartLengths& p,
    double scale) {
  const int64_t h = p.history, c = p.context, r = p.real_time, t = p.target;
  auto q = SeqToBNSD(q_seq);
  auto k = SeqToBNSD(k_seq);
  auto v = SeqToBNSD(v_seq);
  const int64_t nheads = q.size(1);
  const int64_t d = q.size(3);

  // Step-1-equivalent: history causal self-attention
  auto qh = q.slice(2, 0, h);
  auto kh = k.slice(2, 0, h);
  auto vh = v.slice(2, 0, h);
  auto mask_h =
      torch::triu(torch::ones({h, h}, torch::dtype(torch::kBool)), 1);
  auto history = ReferenceAttentionBNSD(qh, kh, vh, &mask_h, scale);

  // Context: queries h..h+c-1, keys 0..h+c-1, full
  auto q_ctx = q.slice(2, h, h + c);
  auto k_hc = k.slice(2, 0, h + c);
  auto v_hc = v.slice(2, 0, h + c);
  auto context = ReferenceAttentionBNSD(q_ctx, k_hc, v_hc, nullptr, scale);

  // Real_time: query at h+c+i attends to keys 0..h+c+i (prefix + causal rt)
  auto real_time = torch::empty({1, nheads, r, d}, q.options());
  for (int64_t i = 0; i < r; ++i) {
    int64_t nkv = h + c + i + 1;
    auto qi = q.slice(2, h + c + i, h + c + i + 1);
    auto kk = k.slice(2, 0, nkv);
    auto vv = v.slice(2, 0, nkv);
    auto oi = ReferenceAttentionBNSD(qi, kk, vv, nullptr, scale);
    real_time.narrow(2, i, 1).copy_(oi);
  }

  // Target: query at prefix+j attends to keys [0 : h+c+r) and [prefix+j]
  const int64_t prefix = h + c + r;
  auto target = torch::empty({1, nheads, t, d}, q.options());
  for (int64_t j = 0; j < t; ++j) {
    auto qj = q.slice(2, prefix + j, prefix + j + 1);
    auto k_prefix = k.slice(2, 0, prefix);
    auto k_self = k.slice(2, prefix + j, prefix + j + 1);
    auto k_cat = torch::cat({k_prefix, k_self}, 2);
    auto v_prefix = v.slice(2, 0, prefix);
    auto v_self = v.slice(2, prefix + j, prefix + j + 1);
    auto v_cat = torch::cat({v_prefix, v_self}, 2);
    auto oj = ReferenceAttentionBNSD(qj, k_cat, v_cat, nullptr, scale);
    target.narrow(2, j, 1).copy_(oj);
  }

  return GenRecReferenceOutputs{history, context, real_time, target};
}

// ---------------------------------------------------------------------------
// Flexible attention call descriptor: supports cross-attention (q_len !=
// kv_len), optional mask (nullptr → full attention), and optional LSE output.
// ---------------------------------------------------------------------------
struct AttnCallDesc {
  torch::Tensor q, k, v, mask, out, lse;
  int64_t sparse_mode = 0;
  int64_t pre_tokens = 2147483647LL;
  int64_t next_tokens = 2147483647LL;

  aclTensor* q_acl = nullptr;
  aclTensor* k_acl = nullptr;
  aclTensor* v_acl = nullptr;
  aclTensor* mask_acl = nullptr;
  aclTensor* out_acl = nullptr;
  aclTensor* lse_acl = nullptr;
  aclTensorList* k_list = nullptr;
  aclTensorList* v_list = nullptr;

  void CreateAcl() {
    q_acl = TorchToAclTensor(q);
    k_acl = TorchToAclTensor(k);
    v_acl = TorchToAclTensor(v);
    if (mask.defined()) mask_acl = TorchToAclTensor(mask);
    out_acl = TorchToAclTensor(out);
    if (lse.defined()) lse_acl = TorchToAclTensor(lse);
    aclTensor* k_arr[] = {k_acl};
    k_list = aclCreateTensorList(k_arr, 1);
    CHECK_NE(k_list, nullptr);
    aclTensor* v_arr[] = {v_acl};
    v_list = aclCreateTensorList(v_arr, 1);
    CHECK_NE(v_list, nullptr);
  }

  void DestroyAcl() {
    aclDestroyTensor(q_acl);
    aclDestroyTensor(k_acl);
    aclDestroyTensor(v_acl);
    if (mask_acl) aclDestroyTensor(mask_acl);
    aclDestroyTensor(out_acl);
    if (lse_acl) aclDestroyTensor(lse_acl);
  }
};

// Phase 1 (CPU) for a flexible attention call.
// sparse_mode / pre_tokens / next_tokens: see CANN FusedInferAttentionScoreV3
// docs; with a full explicit BOOL attenMask (sparseMode 0 or 1), pre/next are
// often ignored — the mask defines visibility.
std::pair<uint64_t, aclOpExecutor*> PlanFlexAttention(
    const AttnCallDesc& d,
    int64_t num_heads,
    int64_t num_kv_heads,
    double scale,
    const char* tag = "unknown",
    int64_t sparse_mode = 0,
    int64_t pre_tokens = 2147483647LL,
    int64_t next_tokens = 2147483647LL) {
  char layout[] = "BNSD";
  uint64_t ws_size = 0;
  aclOpExecutor* executor = nullptr;

  auto ret = aclnnFusedInferAttentionScoreV3GetWorkspaceSize(
      d.q_acl, d.k_list, d.v_list,
      /*pseShift=*/nullptr,
      /*attenMask=*/d.mask_acl,
      /*actualSeqLengths=*/nullptr,
      /*actualSeqLengthsKv=*/nullptr,
      /*deqScale1=*/nullptr,
      /*quantScale1=*/nullptr,
      /*deqScale2=*/nullptr,
      /*quantScale2=*/nullptr,
      /*quantOffset2=*/nullptr,
      /*antiquantScale=*/nullptr,
      /*antiquantOffset=*/nullptr,
      /*blockTable=*/nullptr,
      /*queryPaddingSize=*/nullptr,
      /*kvPaddingSize=*/nullptr,
      /*keyAntiquantScale=*/nullptr,
      /*keyAntiquantOffset=*/nullptr,
      /*valueAntiquantScale=*/nullptr,
      /*valueAntiquantOffset=*/nullptr,
      /*keySharedPrefix=*/nullptr,
      /*valueSharedPrefix=*/nullptr,
      /*actualSharedPrefixLen=*/nullptr,
      /*queryRope=*/nullptr,
      /*keyRope=*/nullptr,
      /*keyRopeAntiquantScale=*/nullptr,
      num_heads, scale,
      /*preTokens=*/pre_tokens,
      /*nextTokens=*/next_tokens,
      layout, num_kv_heads,
      /*sparseMode=*/sparse_mode,
      /*innerPrecise=*/1,
      /*blockSize=*/0,
      /*antiquantMode=*/0,
      /*softmaxLseFlag=*/(d.lse_acl != nullptr),
      /*keyAntiquantMode=*/0,
      /*valueAntiquantMode=*/0,
      d.out_acl,
      /*softmaxLse=*/d.lse_acl,
      &ws_size, &executor);
  CHECK_EQ(ret, 0) << "PlanFlexAttention failed: " << ret
                    << " (tag=" << tag << ")";
  return {ws_size, executor};
}

// [S, N, D] → [1, N, S, D] (contiguous BNSD).
torch::Tensor ToBNSD(const torch::Tensor& t) {
  return t.unsqueeze(0).permute({0, 2, 1, 3}).contiguous();
}

// Build all 5 attention calls for one batch from sliced q/k/v sequences.
// q_seq: [S, N, D],  k_seq/v_seq: [S, Nkv, D]  (contiguous, on NPU).
std::vector<AttnCallDesc> BuildGenRecCalls(
    const torch::Tensor& q_seq,
    const torch::Tensor& k_seq,
    const torch::Tensor& v_seq,
    const SeqPartLengths& p,
    int64_t num_heads,
    int64_t num_kv_heads,
    torch::Device device) {
  const int64_t h = p.history, c = p.context;
  const int64_t r = p.real_time, t = p.target;
  const int64_t head_dim = q_seq.size(2);
  auto causal2048 = CreateCompressedCausalMask2048(device);

  auto out_opts =
      torch::TensorOptions().dtype(q_seq.scalar_type()).device(device);
  auto lse_opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto make_out = [&](int64_t n, int64_t s) {
    return torch::empty({1, n, s, head_dim}, out_opts);
  };
  auto make_lse = [&](int64_t n, int64_t s) {
    return torch::empty({1, n, s, 1}, lse_opts);
  };

  std::vector<AttnCallDesc> calls(5);

  // Step 1: history causal self-attention
  calls[0].q = ToBNSD(q_seq.slice(0, 0, h));
  calls[0].k = ToBNSD(k_seq.slice(0, 0, h));
  calls[0].v = ToBNSD(v_seq.slice(0, 0, h));
  calls[0].mask = causal2048;
  calls[0].sparse_mode = 2;
  calls[0].out = make_out(num_heads, h);

  // Step 2a: q=context+real_time, kv=history+context, full attn (+ LSE)
  calls[1].q = ToBNSD(q_seq.slice(0, h, h + c + r));
  calls[1].k = ToBNSD(k_seq.slice(0, 0, h + c));
  calls[1].v = ToBNSD(v_seq.slice(0, 0, h + c));
  calls[1].mask = CreateFullAttentionMask(c + r, h + c, device);
  calls[1].out = make_out(num_heads, c + r);
  calls[1].lse = make_lse(num_heads, c + r);

  // Step 2b: real_time causal self-attention (+ LSE)
  calls[2].q = ToBNSD(q_seq.slice(0, h + c, h + c + r));
  calls[2].k = ToBNSD(k_seq.slice(0, h + c, h + c + r));
  calls[2].v = ToBNSD(v_seq.slice(0, h + c, h + c + r));
  calls[2].mask = causal2048;
  calls[2].sparse_mode = 2;
  calls[2].out = make_out(num_heads, r);
  calls[2].lse = make_lse(num_heads, r);

  // Step 3a: q=target, kv=h+ctx+rt, full attn (+ LSE)
  calls[3].q = ToBNSD(q_seq.slice(0, h + c + r, h + c + r + t));
  calls[3].k = ToBNSD(k_seq.slice(0, 0, h + c + r));
  calls[3].v = ToBNSD(v_seq.slice(0, 0, h + c + r));
  calls[3].mask = CreateFullAttentionMask(t, h + c + r, device);
  calls[3].out = make_out(num_heads, t);
  calls[3].lse = make_lse(num_heads, t);

  // Step 3b: target diagonal self-attention (+ LSE)
  calls[4].q = ToBNSD(q_seq.slice(0, h + c + r, h + c + r + t));
  calls[4].k = ToBNSD(k_seq.slice(0, h + c + r, h + c + r + t));
  calls[4].v = ToBNSD(v_seq.slice(0, h + c + r, h + c + r + t));
  calls[4].mask = CreateDiagonalMask(t, device);
  // Reduce compute window to only the diagonal token index.
  calls[4].sparse_mode = 0;
  calls[4].pre_tokens = 0;
  calls[4].next_tokens = 0;
  calls[4].out = make_out(num_heads, t);
  calls[4].lse = make_lse(num_heads, t);

  return calls;
}

// ---------------------------------------------------------------------------
// GenRec variant V2 (4 launches + rt 2-way combine + target 3-way combine):
//   1) history: causal self (unchanged)
//   2a) Q = context+real_time+target, KV = history+context, full
//   2b) Q = real_time+target, KV = real_time, mixed mask (RT causal / tgt full)
//   combine RT: 2a[rt rows] ⊕ 2b[rt rows]
//   3) Q = target, KV = target, diagonal (self only)
//   combine target: 2a[tgt] ⊕ 2b[tgt] ⊕ 3
// Requires real_time length r > 0.
// ---------------------------------------------------------------------------
std::vector<AttnCallDesc> BuildGenRecV2Calls(
    const torch::Tensor& q_seq,
    const torch::Tensor& k_seq,
    const torch::Tensor& v_seq,
    const SeqPartLengths& p,
    int64_t num_heads,
    int64_t num_kv_heads,
    torch::Device device) {
  const int64_t h = p.history, c = p.context, r = p.real_time, t = p.target;
  const int64_t head_dim = q_seq.size(2); // BNSD

  auto out_opts =
      torch::TensorOptions().dtype(q_seq.scalar_type()).device(device);
  auto lse_opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto make_out = [&](int64_t n, int64_t s) {
    return torch::empty({1, n, s, head_dim}, out_opts);
  };
  auto make_lse = [&](int64_t n, int64_t s) {
    return torch::empty({1, n, s, 1}, lse_opts);
  };

  std::vector<AttnCallDesc> calls(4);

  auto causal2048 = CreateCompressedCausalMask2048(device);

  // Step 0: history self-attention — leftUpCausal via sparseMode=2.
  calls[0].q = ToBNSD(q_seq.slice(0, 0, h));
  calls[0].k = ToBNSD(k_seq.slice(0, 0, h));
  calls[0].v = ToBNSD(v_seq.slice(0, 0, h));
  calls[0].mask = causal2048;
  calls[0].sparse_mode = 2;
  calls[0].out = make_out(num_heads, h);

  // Step 2a: Q = ctx+rt+tgt, KV = hist+ctx — full attention (no masking).
  calls[1].q = ToBNSD(q_seq.slice(0, h, h + c + r + t));
  calls[1].k = ToBNSD(k_seq.slice(0, 0, h + c));
  calls[1].v = ToBNSD(v_seq.slice(0, 0, h + c));
  calls[1].mask = CreateFullAttentionMask(c + r + t, h + c, device);
  calls[1].out = make_out(num_heads, c + r + t);
  calls[1].lse = make_lse(num_heads, c + r + t);

  // Step 2b: Q = rt+tgt, KV = rt — leftUpCausal (sparseMode=2).
  //   Top r rows: causal (rt on rt).
  //   Bottom t rows: full (tgt sees all rt, since Sq > Skv, all cols visible).
  calls[2].q = ToBNSD(q_seq.slice(0, h + c, h + c + r + t));
  calls[2].k = ToBNSD(k_seq.slice(0, h + c, h + c + r));
  calls[2].v = ToBNSD(v_seq.slice(0, h + c, h + c + r));
  calls[2].mask = causal2048;
  calls[2].sparse_mode = 2;
  calls[2].out = make_out(num_heads, r + t);
  calls[2].lse = make_lse(num_heads, r + t);

  // Step 3: target self-attention — diagonal (each token sees only itself).
  calls[3].q = ToBNSD(q_seq.slice(0, h + c + r, h + c + r + t));
  calls[3].k = ToBNSD(k_seq.slice(0, h + c + r, h + c + r + t));
  calls[3].v = ToBNSD(v_seq.slice(0, h + c + r, h + c + r + t));
  calls[3].mask = CreateDiagonalMask(t, device);
  // Use sparse token window to avoid full QK^T redundancy:
  // pre_tokens/next_tokens=0 means only the current key index is visible.
  // (We keep the diagonal BOOL mask for explicit semantics.)
  calls[3].sparse_mode = 0;
  calls[3].pre_tokens = 0;
  calls[3].next_tokens = 0;
  calls[3].out = make_out(num_heads, t);
  calls[3].lse = make_lse(num_heads, t);

  return calls;
}

}  // namespace util
}  // namespace xllm::kernel::npu::test
