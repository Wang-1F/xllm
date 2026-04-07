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

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/torch_npu.h>

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <string>
#include <cstring>
#include <tuple>
#include <vector>

#if __has_include("acl/acl.h")
#include "acl/acl.h"
#elif __has_include("third_party/acl/inc/acl/acl.h")
#include "third_party/acl/inc/acl/acl.h"
#elif __has_include("/usr/local/Ascend/ascend-toolkit/latest/include/acl/acl.h")
#include "/usr/local/Ascend/ascend-toolkit/latest/include/acl/acl.h"
#elif __has_include("/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/acl/acl.h")
#include "/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/acl/acl.h"
#else
#error "Cannot find acl/acl.h"
#endif

#if __has_include("aclnnop/aclnn_attention_update.h")
#include "aclnnop/aclnn_attention_update.h"
#elif __has_include("/usr/local/Ascend/ascend-toolkit/latest/include/aclnnop/aclnn_attention_update.h")
#include "/usr/local/Ascend/ascend-toolkit/latest/include/aclnnop/aclnn_attention_update.h"
#elif __has_include("/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/aclnnop/aclnn_attention_update.h")
#include "/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/aclnnop/aclnn_attention_update.h"
#else
#error "Cannot find aclnnop/aclnn_attention_update.h"
#endif

#if __has_include("aclnnop/aclnn_fused_infer_attention_score_v3.h")
#include "aclnnop/aclnn_fused_infer_attention_score_v3.h"
#elif __has_include("/usr/local/Ascend/ascend-toolkit/latest/include/aclnnop/aclnn_fused_infer_attention_score_v3.h")
#include "/usr/local/Ascend/ascend-toolkit/latest/include/aclnnop/aclnn_fused_infer_attention_score_v3.h"
#elif __has_include("/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/aclnnop/aclnn_fused_infer_attention_score_v3.h")
#include "/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/aclnnop/aclnn_fused_infer_attention_score_v3.h"
#else
#error "Cannot find aclnnop/aclnn_fused_infer_attention_score_v3.h"
#endif

namespace xllm::kernel::npu::test {

namespace {

struct AttentionMetadata {
  torch::Tensor history_lens;
  torch::Tensor context_lens;
  torch::Tensor real_time_lens;
  torch::Tensor target_lens;

  // for device 2048 compressed causal mask
  torch::Tensor compressed_causal_mask;
  // for device diagonal mask
  torch::Tensor diagonal_mask;
};

struct SegmentAttentionMetadata {
  torch::Tensor attn_mask;
  int64_t sparse_mode;
  int64_t pre_tokens;
  int64_t next_tokens;
};

class GenRecDirectRunTest : public ::testing::Test {
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
    ASSERT_NE(stream_, nullptr);
  }

  torch::Device device_{torch::kCPU};
  torch::TensorOptions opts_;
  aclrtStream stream_{nullptr};
};

bool valid_tensor(const torch::Tensor& t) { return t.defined() && t.numel() > 0; }

// CPU FP32 baseline: softmax(scale * QK^T + mask_bias) @ V，GQA 时对 K/V 做 head repeat。
// 与 SegmentAttentionMetadata + BNSD 张量形状对齐，便于与 aclnnFusedInferAttentionScoreV3 输出对比。
constexpr bool kEnableReferenceAttentionCheck = true;
constexpr bool kEnableReferenceLseMergeCheck = true;
constexpr bool kPrintReferenceDiffToStderr = true;
constexpr double kRefAttentionRtol = 0.12;
constexpr double kRefAttentionAtol = 0.12;
// 各段 fused 输出的 softmax LSE：CPU 基线为 logsumexp(scale*QK^T+mask)，与 output 参考同源 scores。
constexpr bool kEnableReferenceLseCheck = true;
constexpr double kRefLseRtol = 0.15;
constexpr double kRefLseAtol = 0.15;
// LSE merge 参考实现与 aclnnAttentionUpdate 若 log 底或归一化略有差异，阈值可略放宽
constexpr double kRefLseMergeRtol = 0.15;
constexpr double kRefLseMergeAtol = 0.15;

// 小 shape 调试：将中间结果 dump 到工作目录下子目录（每行一个 float，首行为 # 注释含 shape）。
constexpr bool kDumpGenRecStructuredArtifacts = true;
constexpr const char* kGenRecStructuredDumpDir = "genrec_structured_dump";

void dump_cpu_fp32_tensor_text(const char* relative_path,
                               const torch::Tensor& cpu_fp32_contiguous) {
  if (!kDumpGenRecStructuredArtifacts) {
    return;
  }
  const auto c = cpu_fp32_contiguous.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
  std::error_code ec;
  std::filesystem::create_directories(kGenRecStructuredDumpDir, ec);
  const auto full = std::filesystem::path(kGenRecStructuredDumpDir) / relative_path;
  std::ofstream ofs(full.string());
  CHECK(ofs.good()) << "open dump file failed: " << full.string();
  ofs << std::scientific << std::setprecision(9);
  ofs << "# shape=";
  for (int64_t d = 0; d < c.dim(); ++d) {
    ofs << c.size(d) << (d + 1 < c.dim() ? "x" : "");
  }
  ofs << " numel=" << c.numel() << "\n";
  const float* p = c.data_ptr<float>();
  for (int64_t i = 0; i < c.numel(); ++i) {
    ofs << p[i] << "\n";
  }
}

void dump_tensor_text(const char* relative_path,
                      aclrtStream stream,
                      const torch::Tensor& t) {
  if (!kDumpGenRecStructuredArtifacts) {
    return;
  }
  if (!t.is_cpu()) {
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS)
        << "sync before dump failed: " << relative_path;
  }
  dump_cpu_fp32_tensor_text(relative_path, t);
}

// 与 expect_* 对齐：同一 stage 下 dump NPU(attn) / CPU baseline(ref) / |npu-ref| 便于对精度。
void dump_npu_ref_diff_triplet(const char* file_basename,
                               const torch::Tensor& got_fp32_cpu,
                               const torch::Tensor& ref_fp32_cpu) {
  if (!kDumpGenRecStructuredArtifacts) {
    return;
  }
  const std::string base(file_basename);
  dump_cpu_fp32_tensor_text((base + "_npu.txt").c_str(), got_fp32_cpu);
  dump_cpu_fp32_tensor_text((base + "_ref.txt").c_str(), ref_fp32_cpu);
  dump_cpu_fp32_tensor_text((base + "_diff_abs.txt").c_str(),
                            (got_fp32_cpu - ref_fp32_cpu).abs());
}

const char* attention_stage_dump_basename(const char* stage_name) {
  if (std::strcmp(stage_name, "history") == 0) {
    return "batch0_01_hist_attn";
  }
  if (std::strcmp(stage_name, "crt_context_rt_t") == 0) {
    return "batch0_02_crt_attn";
  }
  if (std::strcmp(stage_name, "rt_realtime_t") == 0) {
    return "batch0_03_rt_attn";
  }
  if (std::strcmp(stage_name, "target") == 0) {
    return "batch0_04_target_attn";
  }
  return nullptr;
}

const char* lse_merge_stage_dump_basename(const char* stage_name) {
  if (std::strcmp(stage_name, "lse_merge_real_time") == 0) {
    return "batch0_05_lse_merge_real_time";
  }
  if (std::strcmp(stage_name, "lse_merge_target") == 0) {
    return "batch0_06_lse_merge_target";
  }
  return nullptr;
}

// 与 reference_scaled_dot_product_attention / expect_npu_attention_matches_reference 使用同一套 scores。
torch::Tensor reference_attention_scores_fp32(
    const torch::Tensor& query_bnsd,
    const torch::Tensor& key_bnsd,
    const std::optional<torch::Tensor>& attn_mask_b11ss,
    float scale) {
  const int64_t num_q_heads = query_bnsd.size(1);
  const int64_t num_kv_heads = key_bnsd.size(1);
  CHECK_EQ(num_q_heads % num_kv_heads, 0)
      << "num_heads must be divisible by num_kv_heads";
  const int64_t group = num_q_heads / num_kv_heads;

  auto q = query_bnsd.to(torch::kCPU).to(torch::kFloat32).contiguous();
  auto k = key_bnsd.to(torch::kCPU).to(torch::kFloat32).contiguous();
  if (group > 1) {
    k = k.repeat_interleave(group, /*dim=*/1);
  }

  auto scores = torch::matmul(q, k.transpose(-2, -1)) * scale;

  if (attn_mask_b11ss.has_value() && valid_tensor(attn_mask_b11ss.value())) {
    auto m = attn_mask_b11ss.value().to(torch::kCPU);
    while (m.dim() > 2 && m.size(0) == 1) {
      m = m.squeeze(0);
    }
    const int64_t sq = scores.size(2);
    const int64_t sk = scores.size(3);
    m = m.slice(0, 0, sq).slice(1, 0, sk);
    if (m.scalar_type() == torch::kBool) {
      scores = scores.masked_fill(
          m.unsqueeze(0).unsqueeze(0),
          -std::numeric_limits<float>::infinity());
    } else {
      scores = scores + m.to(torch::kFloat32).unsqueeze(0).unsqueeze(0);
    }
  }
  return scores;
}

torch::Tensor reference_scaled_dot_product_attention(
    const torch::Tensor& query_bnsd,
    const torch::Tensor& key_bnsd,
    const torch::Tensor& value_bnsd,
    const std::optional<torch::Tensor>& attn_mask_b11ss,
    float scale) {
  const int64_t num_q_heads = query_bnsd.size(1);
  const int64_t num_kv_heads = key_bnsd.size(1);
  const int64_t group = num_q_heads / num_kv_heads;

  auto scores = reference_attention_scores_fp32(
      query_bnsd, key_bnsd, attn_mask_b11ss, scale);
  auto attn = torch::softmax(scores, -1);

  auto v = value_bnsd.to(torch::kCPU).to(torch::kFloat32).contiguous();
  if (group > 1) {
    v = v.repeat_interleave(group, /*dim=*/1);
  }
  return torch::matmul(attn, v);
}

// FusedInfer softmax_lse 的 CPU 对照：对 logits 最后一维做 log-sum-exp，形状 [B,N,S,1]。
torch::Tensor reference_scaled_dot_product_attention_lse(
    const torch::Tensor& query_bnsd,
    const torch::Tensor& key_bnsd,
    const std::optional<torch::Tensor>& attn_mask_b11ss,
    float scale) {
  auto scores = reference_attention_scores_fp32(
      query_bnsd, key_bnsd, attn_mask_b11ss, scale);
  return torch::logsumexp(scores, -1, true);
}

// ---------------------------------------------------------------------------
// GenRec V2 end-to-end gold（与 segmented_prefill_attention_test /
// genrec_attention_test.cpp BenchGenRecV2Attention 一致）：CPU FP32 按段语义
// 直接算 attention，不经 LSE merge。
// ---------------------------------------------------------------------------
struct SeqPartLengths {
  int64_t history;
  int64_t context;
  int64_t real_time;
  int64_t target;
  int64_t total() const { return history + context + real_time + target; }
};

torch::Tensor ReferenceAttentionBNSD(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor* mask_sq_sk,
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

torch::Tensor SeqToBNSD(const torch::Tensor& seq_s_n_d) {
  return seq_s_n_d.unsqueeze(0).permute({0, 2, 1, 3}).contiguous();
}

struct GenRecReferenceOutputs {
  torch::Tensor history;
  torch::Tensor context;
  torch::Tensor real_time;
  torch::Tensor target;
};

GenRecReferenceOutputs ReferenceGenRecGold(const torch::Tensor& q_seq,
                                           const torch::Tensor& k_seq,
                                           const torch::Tensor& v_seq,
                                           const SeqPartLengths& p,
                                           double scale) {
  const int64_t h = p.history, c = p.context, r = p.real_time, t = p.target;
  auto q = SeqToBNSD(q_seq);
  auto k = SeqToBNSD(k_seq);
  auto v = SeqToBNSD(v_seq);
  const int64_t nheads = q.size(1);
  const int64_t d = q.size(3);

  auto qh = q.slice(2, 0, h);
  auto kh = k.slice(2, 0, h);
  auto vh = v.slice(2, 0, h);
  auto mask_h =
      torch::triu(torch::ones({h, h}, torch::dtype(torch::kBool)), 1);
  auto history = ReferenceAttentionBNSD(qh, kh, vh, &mask_h, scale);

  auto q_ctx = q.slice(2, h, h + c);
  auto k_hc = k.slice(2, 0, h + c);
  auto v_hc = v.slice(2, 0, h + c);
  auto context = ReferenceAttentionBNSD(q_ctx, k_hc, v_hc, nullptr, scale);

  auto real_time = torch::empty({1, nheads, r, d}, q.options());
  for (int64_t i = 0; i < r; ++i) {
    int64_t nkv = h + c + i + 1;
    auto qi = q.slice(2, h + c + i, h + c + i + 1);
    auto kk = k.slice(2, 0, nkv);
    auto vv = v.slice(2, 0, nkv);
    auto oi = ReferenceAttentionBNSD(qi, kk, vv, nullptr, scale);
    real_time.narrow(2, i, 1).copy_(oi);
  }

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

// 与 genrec_attention_test.cpp BenchGenRecV2Attention 中 Precision 段相同：
// max_abs(NPU 组装后的 real_time/target, ReferenceGenRecGold)。
constexpr float kGenRecV2FinalGoldMaxAbs = 1.5e-1f;

// 在 CPU FP32 上对 got/ref 逐元素 diff；log_tag 区分 FusedAttn vs LSE merge 等场景。
void print_fp32_pair_diff(const char* log_tag,
                          const char* stage_name,
                          const torch::Tensor& got_fp32_cpu,
                          const torch::Tensor& ref_fp32_cpu,
                          double rtol,
                          double atol) {
  CHECK_EQ(got_fp32_cpu.sizes(), ref_fp32_cpu.sizes());
  CHECK_EQ(got_fp32_cpu.scalar_type(), torch::kFloat32);
  CHECK_EQ(ref_fp32_cpu.scalar_type(), torch::kFloat32);
  const auto diff = (got_fp32_cpu - ref_fp32_cpu).abs();
  const double max_abs = diff.max().item<double>();
  const double mean_abs = diff.mean().item<double>();
  const double rmse = diff.pow(2).mean().sqrt().item<double>();
  const auto rel = diff / (ref_fp32_cpu.abs() + 1e-4f);
  const double max_rel = rel.max().item<double>();
  const double sum_abs = diff.sum().item<double>();
  const int64_t numel = got_fp32_cpu.numel();
  const bool all_close = torch::allclose(got_fp32_cpu, ref_fp32_cpu, rtol, atol);
  if (kPrintReferenceDiffToStderr) {
    std::fprintf(stderr,
                 "[%s] stage=%s numel=%lld max_abs=%.6e mean_abs=%.6e rmse=%.6e "
                 "max_rel=%.6e sum_abs=%.6e allclose(rtol=%.4g,atol=%.4g)=%s\n",
                 log_tag,
                 stage_name,
                 static_cast<long long>(numel),
                 max_abs,
                 mean_abs,
                 rmse,
                 max_rel,
                 sum_abs,
                 rtol,
                 atol,
                 all_close ? "PASS" : "FAIL");
    std::fflush(stderr);
  }
}

void print_attention_output_diff_vs_reference(const char* stage_name,
                                                const torch::Tensor& got_fp32_cpu,
                                                const torch::Tensor& ref_fp32_cpu) {
  print_fp32_pair_diff("RefAttnDiff",
                         stage_name,
                         got_fp32_cpu,
                         ref_fp32_cpu,
                         kRefAttentionRtol,
                         kRefAttentionAtol);
}

void print_lse_diff_vs_reference(const char* stage_name,
                                 const torch::Tensor& got_fp32_cpu,
                                 const torch::Tensor& ref_fp32_cpu) {
  print_fp32_pair_diff("RefLseDiff",
                       stage_name,
                       got_fp32_cpu,
                       ref_fp32_cpu,
                       kRefLseRtol,
                       kRefLseAtol);
}

// 与 genrec_attention_test BenchGenRecV2Attention 的 Precision 段同语义：
// max_abs(组装后的 real_time/target, ReferenceGenRecGold)。
// 分段 RefAttnDiff 通过仅说明各段 fused 与逐段 CPU 参考一致；最终与 gold 的差异通常来自
// LSE merge（aclnnAttentionUpdate）。merge 侧须与 benchmark 一致使用 row-major ND + FP32 展平。
void expect_genrec_v2_output_vs_reference_genrec_gold(
    const char* tag,
    const torch::Tensor& output_bnsd,
    const torch::Tensor& query_bnsd,
    const torch::Tensor& key_bnsd,
    const torch::Tensor& value_bnsd,
    int64_t h,
    int64_t c,
    int64_t r,
    int64_t t,
    double scale,
    float max_abs_eps = kGenRecV2FinalGoldMaxAbs) {
  auto q_seq = query_bnsd.select(0, 0)
                   .permute({1, 0, 2})
                   .contiguous()
                   .to(torch::kCPU)
                   .to(torch::kFloat32);
  auto k_seq = key_bnsd.select(0, 0)
                   .permute({1, 0, 2})
                   .contiguous()
                   .to(torch::kCPU)
                   .to(torch::kFloat32);
  auto v_seq = value_bnsd.select(0, 0)
                   .permute({1, 0, 2})
                   .contiguous()
                   .to(torch::kCPU)
                   .to(torch::kFloat32);
  SeqPartLengths parts{h, c, r, t};
  CHECK_EQ(q_seq.size(0), parts.total());
  auto ref = ReferenceGenRecGold(q_seq, k_seq, v_seq, parts, scale);

  auto max_abs = [](const torch::Tensor& a, const torch::Tensor& b) -> float {
    return (a.to(torch::kCPU).to(torch::kFloat32) -
            b.to(torch::kCPU).to(torch::kFloat32))
        .abs()
        .max()
        .item<float>();
  };

  auto rt_final = output_bnsd.slice(0, 0, 1)
                      .slice(2, h + c, h + c + r)
                      .detach()
                      .contiguous();
  auto tgt_final = output_bnsd.slice(0, 0, 1)
                       .slice(2, h + c + r, h + c + r + t)
                       .detach()
                       .contiguous();

  float d_rt = max_abs(rt_final, ref.real_time);
  float d_tgt = max_abs(tgt_final, ref.target);

  std::fprintf(stderr,
               "[GenRecV2][Precision][%s] max_abs real_time=%.6e target=%.6e "
               "(threshold=%.6e)\n",
               tag,
               static_cast<double>(d_rt),
               static_cast<double>(d_tgt),
               static_cast<double>(max_abs_eps));
  std::fflush(stderr);

  print_fp32_pair_diff("RefFinalGoldDiff",
                       "final_real_time",
                       rt_final.to(torch::kCPU).to(torch::kFloat32).contiguous(),
                       ref.real_time.to(torch::kCPU).to(torch::kFloat32).contiguous(),
                       kRefAttentionRtol,
                       kRefAttentionAtol);
  print_fp32_pair_diff("RefFinalGoldDiff",
                       "final_target",
                       tgt_final.to(torch::kCPU).to(torch::kFloat32).contiguous(),
                       ref.target.to(torch::kCPU).to(torch::kFloat32).contiguous(),
                       kRefAttentionRtol,
                       kRefAttentionAtol);

  EXPECT_LT(d_rt, max_abs_eps);
  EXPECT_LT(d_tgt, max_abs_eps);
}

// Combine two flash-attention partial outputs using their log-sum-exp.
// o1, o2: [1, N, S, D],  lse1, lse2: [1, N, S, 1] (FP32).
// LSE is already 4D so it broadcasts directly with [1, N, S, D].
// (Same as segmented_prefill_attention_test.cpp — used as CPU reference for LSE merge.)
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

torch::Tensor reference_lse_merge_flash_cpu(const std::vector<torch::Tensor>& outs,
                                              const std::vector<torch::Tensor>& lses) {
  CHECK_EQ(outs.size(), lses.size());
  const size_t k = outs.size();
  CHECK(k == 2 || k == 3);
  auto o1 = outs[0].to(torch::kCPU).to(torch::kFloat32);
  auto o2 = outs[1].to(torch::kCPU).to(torch::kFloat32);
  auto l1 = lses[0].to(torch::kCPU).to(torch::kFloat32);
  auto l2 = lses[1].to(torch::kCPU).to(torch::kFloat32);
  if (k == 2) {
    return CombineFlashOutputs(o1, l1, o2, l2).to(torch::kFloat32).contiguous();
  }
  auto o3 = outs[2].to(torch::kCPU).to(torch::kFloat32);
  auto l3 = lses[2].to(torch::kCPU).to(torch::kFloat32);
  return CombineFlashOutputsThree(o1, l1, o2, l2, o3, l3).to(torch::kFloat32).contiguous();
}

// 分解「最终 gold 对不上」：RowMajor/stride 若已对齐仍无改善，根因通常在 (1) 或 (2)：
// (1) max_abs(CombineFlashOutputs(NPU 的 out+LSE), ReferenceGenRecGold) 大
//     → CANN softmax_lse 与 PyTorch 合并公式假设的 log 域不一致，或分段与 gold 语义不完全等价；
// (2) 上式小但 max_abs(NPU aclnnAttentionUpdate 输出, CombineFlashOutputs) 大
//     → 仅 aclnn merge 与 PyTorch 公式不一致。
void print_genrec_merge_pipeline_diagnostics(
    int64_t batch_idx,
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& output,
    const torch::Tensor& crt_out,
    const torch::Tensor& crt_lse,
    const torch::Tensor& rt_out,
    const torch::Tensor& rt_lse,
    const torch::Tensor& target_out,
    const torch::Tensor& target_lse,
    int64_t h,
    int64_t c,
    int64_t r,
    int64_t t,
    float scale) {
  auto q_seq = query.select(0, batch_idx)
                   .permute({1, 0, 2})
                   .contiguous()
                   .cpu()
                   .to(torch::kFloat32);
  auto k_seq = key.select(0, batch_idx)
                   .permute({1, 0, 2})
                   .contiguous()
                   .cpu()
                   .to(torch::kFloat32);
  auto v_seq = value.select(0, batch_idx)
                   .permute({1, 0, 2})
                   .contiguous()
                   .cpu()
                   .to(torch::kFloat32);
  SeqPartLengths parts{h, c, r, t};
  auto ref = ReferenceGenRecGold(q_seq, k_seq, v_seq, parts, static_cast<double>(scale));

  auto o_rt1 = crt_out.slice(2, c, c + r).detach().cpu().to(torch::kFloat32);
  auto l_rt1 = crt_lse.slice(2, c, c + r).detach().cpu().to(torch::kFloat32);
  auto o_rt2 = rt_out.slice(2, 0, r).detach().cpu().to(torch::kFloat32);
  auto l_rt2 = rt_lse.slice(2, 0, r).detach().cpu().to(torch::kFloat32);
  auto rt_merge_cpu = CombineFlashOutputs(o_rt1, l_rt1, o_rt2, l_rt2).to(torch::kFloat32);
  auto npu_rt = output.slice(0, batch_idx, batch_idx + 1)
                    .slice(2, h + c, h + c + r)
                    .detach()
                    .cpu()
                    .to(torch::kFloat32);

  float d_cpu_merge_vs_gold =
      (rt_merge_cpu - ref.real_time.cpu()).abs().max().item<float>();
  float d_npu_vs_gold =
      (npu_rt - ref.real_time.cpu()).abs().max().item<float>();
  float d_npu_vs_cpu_merge =
      (npu_rt - rt_merge_cpu).abs().max().item<float>();

  std::fprintf(stderr,
               "[MergeDiag] real_time: |CombineFlash-Gold|=%.6e |NPU-Gold|=%.6e "
               "|NPU-CombineFlash|=%.6e\n",
               static_cast<double>(d_cpu_merge_vs_gold),
               static_cast<double>(d_npu_vs_gold),
               static_cast<double>(d_npu_vs_cpu_merge));

  auto o_t1 = crt_out.slice(2, c + r, c + r + t).detach().cpu().to(torch::kFloat32);
  auto m_t1 = crt_lse.slice(2, c + r, c + r + t).detach().cpu().to(torch::kFloat32);
  auto o_t2 = rt_out.slice(2, r, r + t).detach().cpu().to(torch::kFloat32);
  auto m_t2 = rt_lse.slice(2, r, r + t).detach().cpu().to(torch::kFloat32);
  auto o_t3 = target_out.detach().cpu().to(torch::kFloat32);
  auto m_t3 = target_lse.detach().cpu().to(torch::kFloat32);
  auto tgt_merge_cpu =
      CombineFlashOutputsThree(o_t1, m_t1, o_t2, m_t2, o_t3, m_t3).to(torch::kFloat32);
  auto npu_tgt = output.slice(0, batch_idx, batch_idx + 1)
                     .slice(2, h + c + r, h + c + r + t)
                     .detach()
                     .cpu()
                     .to(torch::kFloat32);

  float e1 =
      (tgt_merge_cpu - ref.target.cpu()).abs().max().item<float>();
  float e2 = (npu_tgt - ref.target.cpu()).abs().max().item<float>();
  float e3 = (npu_tgt - tgt_merge_cpu).abs().max().item<float>();

  std::fprintf(stderr,
               " | target: |CombineFlash-Gold|=%.6e |NPU-Gold|=%.6e "
               "|NPU-CombineFlash|=%.6e\n",
               static_cast<double>(e1),
               static_cast<double>(e2),
               static_cast<double>(e3));
  std::fflush(stderr);
}

void expect_npu_lse_merge_matches_reference(const char* stage_name,
                                            aclrtStream stream,
                                            const torch::Tensor& npu_output_fp32,
                                            const std::vector<torch::Tensor>& local_outs,
                                            const std::vector<torch::Tensor>& lses) {
  if (!kEnableReferenceLseMergeCheck) {
    return;
  }
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS)
      << "sync before LSE merge reference compare failed stage=" << stage_name;
  auto ref = reference_lse_merge_flash_cpu(local_outs, lses);
  auto got = npu_output_fp32.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
  ASSERT_EQ(ref.sizes(), got.sizes())
      << "LSE merge shape mismatch stage=" << stage_name;
  print_fp32_pair_diff("RefLseMergeDiff",
                         stage_name,
                         got,
                         ref,
                         kRefLseMergeRtol,
                         kRefLseMergeAtol);
  if (kDumpGenRecStructuredArtifacts) {
    const char* bn = lse_merge_stage_dump_basename(stage_name);
    if (bn != nullptr) {
      dump_npu_ref_diff_triplet(bn, got, ref);
    }
  }
  EXPECT_TRUE(torch::allclose(got, ref, kRefLseMergeRtol, kRefLseMergeAtol))
      << "NPU LSE merge vs CPU reference mismatch, stage=" << stage_name
      << " (see [RefLseMergeDiff] stderr)";
}

void expect_npu_attention_matches_reference(const char* stage_name,
                                            aclrtStream stream,
                                            const torch::Tensor& npu_output_bnsd,
                                            const torch::Tensor& query_bnsd,
                                            const torch::Tensor& key_bnsd,
                                            const torch::Tensor& value_bnsd,
                                            const std::optional<torch::Tensor>& attn_mask_b11ss,
                                            float scale) {
  if (!kEnableReferenceAttentionCheck) {
    return;
  }
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS)
      << "sync before reference compare failed stage=" << stage_name;

  auto ref = reference_scaled_dot_product_attention(
      query_bnsd, key_bnsd, value_bnsd, attn_mask_b11ss, scale);
  auto got = npu_output_bnsd.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
  ref = ref.to(torch::kFloat32).contiguous();
  ASSERT_EQ(ref.sizes(), got.sizes())
      << "shape mismatch stage=" << stage_name;

  print_attention_output_diff_vs_reference(stage_name, got, ref);

  if (kDumpGenRecStructuredArtifacts) {
    const char* bn = attention_stage_dump_basename(stage_name);
    if (bn != nullptr) {
      dump_npu_ref_diff_triplet(bn, got, ref);
    }
  }

  EXPECT_TRUE(torch::allclose(got, ref, kRefAttentionRtol, kRefAttentionAtol))
      << "NPU output vs CPU reference mismatch, stage=" << stage_name
      << " (see [RefAttnDiff] stderr line for max_abs/mean_abs/rmse/max_rel)";
}

void expect_npu_lse_matches_reference(const char* stage_name,
                                      aclrtStream stream,
                                      const torch::Tensor& npu_lse_bnsd,
                                      const torch::Tensor& query_bnsd,
                                      const torch::Tensor& key_bnsd,
                                      const std::optional<torch::Tensor>& attn_mask_b11ss,
                                      float scale,
                                      const char* dump_basename) {
  if (!kEnableReferenceLseCheck) {
    return;
  }
  ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS)
      << "sync before LSE reference compare failed stage=" << stage_name;

  auto ref = reference_scaled_dot_product_attention_lse(
      query_bnsd, key_bnsd, attn_mask_b11ss, scale);
  auto got = npu_lse_bnsd.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
  ref = ref.to(torch::kFloat32).contiguous();
  ASSERT_EQ(ref.sizes(), got.sizes())
      << "LSE shape mismatch stage=" << stage_name;

  print_lse_diff_vs_reference(stage_name, got, ref);

  if (kDumpGenRecStructuredArtifacts && dump_basename != nullptr) {
    dump_npu_ref_diff_triplet(dump_basename, got, ref);
  }

  EXPECT_TRUE(torch::allclose(got, ref, kRefLseRtol, kRefLseAtol))
      << "NPU softmax LSE vs CPU logsumexp(scores) mismatch, stage=" << stage_name
      << " (see [RefLseDiff] stderr)";
}

std::optional<torch::Tensor> optional_attn_mask(const SegmentAttentionMetadata& meta) {
  if (valid_tensor(meta.attn_mask)) {
    return meta.attn_mask;
  }
  return std::nullopt;
}

aclDataType to_acl_dtype(torch::ScalarType dtype) {
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

aclTensor* torch_to_acl_tensor(const torch::Tensor& t) {
  // For this standalone structured test, keep ACL descriptors simple and strict:
  // require contiguous tensors and pass logical shape as storage dims.
  CHECK(t.is_contiguous()) << "torch_to_acl_tensor requires contiguous tensor";
  aclDataType acl_tensor_type = to_acl_dtype(t.scalar_type());
  aclTensor* acl_t = aclCreateTensor(t.sizes().data(),
                                     t.sizes().size(),
                                     acl_tensor_type,
                                     t.strides().data(),
                                     /*offset=*/0,
                                     ACL_FORMAT_ND,
                                     t.sizes().data(),
                                     t.sizes().size(),
                                     t.data_ptr());
  CHECK_NE(acl_t, nullptr);
  return acl_t;
}

// 与 segmented_prefill_attention_test 中 TorchToAclTensorRowMajorNd 一致：
// aclnnAttentionUpdate 部分 CANN 版本对 ND stride 与 PyTorch 原生 strides 组合敏感，
// genrec_attention_test 的 combine_* 全部使用本函数包装展平后的张量。
aclTensor* torch_to_acl_tensor_row_major_nd(const torch::Tensor& t) {
  CHECK(t.is_contiguous())
      << "torch_to_acl_tensor_row_major_nd requires contiguous tensor";
  auto shape = t.sizes().vec();
  std::vector<int64_t> stride(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    stride[static_cast<size_t>(i)] =
        shape[static_cast<size_t>(i + 1)] * stride[static_cast<size_t>(i + 1)];
  }
  aclDataType acl_tensor_type = to_acl_dtype(t.scalar_type());
  aclTensor* acl_t = aclCreateTensor(shape.data(),
                                     shape.size(),
                                     acl_tensor_type,
                                     stride.data(),
                                     /*offset=*/0,
                                     ACL_FORMAT_ND,
                                     shape.data(),
                                     shape.size(),
                                     t.data_ptr());
  CHECK_NE(acl_t, nullptr);
  return acl_t;
}

torch::Tensor& get_workspace_cache(const torch::Device& device,
                                   uint64_t workspace_size) {
  static thread_local torch::Tensor workspace;
  if (workspace_size == 0) {
    return workspace;
  }
  if (!workspace.defined() || workspace.device() != device ||
      static_cast<uint64_t>(workspace.numel()) < workspace_size) {
    workspace = torch::empty(
        {static_cast<int64_t>(workspace_size)},
        torch::TensorOptions().dtype(torch::kUInt8).device(device));
  }
  return workspace;
}

struct AclPlan {
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;

  aclTensor* q_acl = nullptr;
  aclTensor* k_acl = nullptr;
  aclTensor* v_acl = nullptr;
  aclTensor* out_acl = nullptr;
  aclTensor* mask_acl = nullptr;
  aclTensor* lse_acl = nullptr;
  aclTensorList* k_list = nullptr;
  aclTensorList* v_list = nullptr;

  std::vector<aclTensor*> extra_acl_tensors;
};

AclPlan plan_segment_attention(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const SegmentAttentionMetadata& attn_metadata,
    torch::Tensor& output,
    const std::optional<torch::Tensor>& lse) {
      
  int64_t num_heads = query.size(1);
  int64_t num_kv_heads = key.size(1);
  float scale = 1.0f / std::sqrt(static_cast<float>(query.size(3)));
  AclPlan plan;

  plan.q_acl = torch_to_acl_tensor(query);
  plan.k_acl = torch_to_acl_tensor(key);
  plan.v_acl = torch_to_acl_tensor(value);
  plan.out_acl = torch_to_acl_tensor(output);
  if (valid_tensor(attn_metadata.attn_mask)) {
    plan.mask_acl = torch_to_acl_tensor(attn_metadata.attn_mask);
  }
  if (lse.has_value() && valid_tensor(lse.value())) {
    plan.lse_acl = torch_to_acl_tensor(lse.value());
  }

  aclTensor* k_arr[] = {plan.k_acl};
  aclTensor* v_arr[] = {plan.v_acl};
  plan.k_list = aclCreateTensorList(k_arr, 1);
  plan.v_list = aclCreateTensorList(v_arr, 1);

  char layout[] = "BNSD";
  auto ret = aclnnFusedInferAttentionScoreV3GetWorkspaceSize(
      plan.q_acl, plan.k_list, plan.v_list,
      /*pseShift=*/nullptr,
      /*attenMask=*/plan.mask_acl,
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
      num_heads,
      scale,
      attn_metadata.pre_tokens,
      attn_metadata.next_tokens,
      layout,
      num_kv_heads,
      attn_metadata.sparse_mode,
      /*innerPrecise=*/1,
      /*blockSize=*/0,
      /*antiquantMode=*/0,
      /*softmaxLseFlag=*/(lse.has_value() && valid_tensor(lse.value())),
      /*keyAntiquantMode=*/0,
      /*valueAntiquantMode=*/0,
      plan.out_acl,
      /*softmaxLse=*/plan.lse_acl,
      &plan.workspace_size,
      &plan.executor);
  CHECK_EQ(ret, ACL_SUCCESS)
      << "aclnnFusedInferAttentionScoreV3GetWorkspaceSize failed: " << ret;
  return plan;
}

void execute_planned_attention(const AclPlan& plan,
                               aclrtStream stream) {
  void* ws_ptr = nullptr;
  if (plan.workspace_size > 0) {
    CHECK_EQ(aclrtMalloc(&ws_ptr, plan.workspace_size, ACL_MEM_MALLOC_HUGE_FIRST),
             ACL_SUCCESS)
        << "aclrtMalloc attention workspace";
  }
  auto ret = aclnnFusedInferAttentionScoreV3(
      ws_ptr, plan.workspace_size, plan.executor, stream);
  CHECK_EQ(ret, ACL_SUCCESS) << "aclnnFusedInferAttentionScoreV3 failed: " << ret;
  CHECK_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS)
      << "aclrtSynchronizeStream after FusedInferAttentionScoreV3 failed";
  if (ws_ptr != nullptr) {
    CHECK_EQ(aclrtFree(ws_ptr), ACL_SUCCESS) << "aclrtFree workspace";
  }
}

void destroy_planned_attention(AclPlan& plan) {
  // aclDestroyTensorList releases the tensors it owns — do NOT double-free
  // k_acl/v_acl after destroying the list (same pattern as the working test).
  if (plan.k_list != nullptr) {
    aclDestroyTensorList(plan.k_list);
    plan.k_list = nullptr;
    plan.k_acl = nullptr;
  }
  if (plan.v_list != nullptr) {
    aclDestroyTensorList(plan.v_list);
    plan.v_list = nullptr;
    plan.v_acl = nullptr;
  }
  if (plan.q_acl != nullptr) {
    aclDestroyTensor(plan.q_acl);
    plan.q_acl = nullptr;
  }
  if (plan.out_acl != nullptr) {
    aclDestroyTensor(plan.out_acl);
    plan.out_acl = nullptr;
  }
  if (plan.mask_acl != nullptr) {
    aclDestroyTensor(plan.mask_acl);
    plan.mask_acl = nullptr;
  }
  if (plan.lse_acl != nullptr) {
    aclDestroyTensor(plan.lse_acl);
    plan.lse_acl = nullptr;
  }
}

// 与 tm/xllm genrec_attention_test.cpp 里 BenchGenRecV2Attention 的
// combine_two_way_attention_update / combine_three_way_attention_update lambda
// 同逻辑（去掉 Log/AuTrace 调试），用于结构化测试的 LSE merge。
torch::Tensor combine_two_way_attention_update_like_benchmark(
    const torch::Tensor& out1,
    const torch::Tensor& lse1,
    const torch::Tensor& out2,
    const torch::Tensor& lse2,
    aclrtStream stream) {
  CHECK_EQ(out1.sizes(), out2.sizes());
  CHECK_EQ(lse1.sizes(), lse2.sizes());
  CHECK_EQ(out1.dim(), 4);
  CHECK_EQ(lse1.dim(), 4);

  const int64_t b = out1.size(0);
  const int64_t n = out1.size(1);
  const int64_t s = out1.size(2);
  const int64_t d = out1.size(3);
  const int64_t bsh = b * n * s;

  auto local_out1_flat =
      out1.contiguous().reshape({bsh, d}).to(torch::kFloat32);
  auto local_out2_flat =
      out2.contiguous().reshape({bsh, d}).to(torch::kFloat32);
  auto lse1_flat =
      lse1.squeeze(-1).to(torch::kFloat32).contiguous().reshape({bsh});
  auto lse2_flat =
      lse2.squeeze(-1).to(torch::kFloat32).contiguous().reshape({bsh});
  auto out_flat = torch::empty(
      {bsh, d},
      torch::TensorOptions().dtype(torch::kFloat32).device(out1.device()));

  aclTensor* local_out_acls[2] = {torch_to_acl_tensor_row_major_nd(local_out1_flat),
                                  torch_to_acl_tensor_row_major_nd(local_out2_flat)};
  aclTensor* lse_acls[2] = {torch_to_acl_tensor_row_major_nd(lse1_flat),
                            torch_to_acl_tensor_row_major_nd(lse2_flat)};
  aclTensor* out_acl = torch_to_acl_tensor_row_major_nd(out_flat);

  aclTensorList* local_out_list = aclCreateTensorList(local_out_acls, 2);
  CHECK_NE(local_out_list, nullptr);
  aclTensorList* lse_list = aclCreateTensorList(lse_acls, 2);
  CHECK_NE(lse_list, nullptr);

  uint64_t ws_size = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnAttentionUpdateGetWorkspaceSize(
      lse_list,
      local_out_list,
      /*updateType=*/0,
      out_acl,
      /*lseOut=*/nullptr,
      &ws_size,
      &executor);
  CHECK_EQ(ret, ACL_SUCCESS) << "aclnnAttentionUpdateGetWorkspaceSize failed: " << ret;

  void* ws_ptr = nullptr;
  if (ws_size > 0) {
    CHECK_EQ(aclrtMalloc(&ws_ptr, ws_size, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS)
        << "aclrtMalloc AttentionUpdate workspace";
  }

  ret = aclnnAttentionUpdate(ws_ptr, ws_size, executor, stream);
  CHECK_EQ(ret, ACL_SUCCESS) << "aclnnAttentionUpdate failed: " << ret;
  CHECK_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS)
      << "aclrtSynchronizeStream after aclnnAttentionUpdate";
  if (ws_ptr != nullptr) {
    CHECK_EQ(aclrtFree(ws_ptr), ACL_SUCCESS) << "aclrtFree workspace";
  }

  torch::Tensor out_bnsd = out_flat.clone().to(out1.scalar_type());
  torch::Tensor out_ret = out_bnsd.view({b, n, s, d});

  aclDestroyTensorList(local_out_list);
  aclDestroyTensorList(lse_list);
  aclDestroyTensor(out_acl);
  return out_ret;
}

torch::Tensor combine_three_way_attention_update_like_benchmark(
    const torch::Tensor& out1,
    const torch::Tensor& lse1,
    const torch::Tensor& out2,
    const torch::Tensor& lse2,
    const torch::Tensor& out3,
    const torch::Tensor& lse3,
    aclrtStream stream) {
  CHECK_EQ(out1.sizes(), out2.sizes());
  CHECK_EQ(out1.sizes(), out3.sizes());
  CHECK_EQ(lse1.sizes(), lse2.sizes());
  CHECK_EQ(lse1.sizes(), lse3.sizes());
  CHECK_EQ(out1.dim(), 4);
  CHECK_EQ(lse1.dim(), 4);

  const int64_t b = out1.size(0);
  const int64_t n = out1.size(1);
  const int64_t s = out1.size(2);
  const int64_t d = out1.size(3);
  const int64_t bsh = b * n * s;

  auto local_out1_flat =
      out1.contiguous().reshape({bsh, d}).to(torch::kFloat32);
  auto local_out2_flat =
      out2.contiguous().reshape({bsh, d}).to(torch::kFloat32);
  auto local_out3_flat =
      out3.contiguous().reshape({bsh, d}).to(torch::kFloat32);
  auto lse1_flat =
      lse1.squeeze(-1).to(torch::kFloat32).contiguous().reshape({bsh});
  auto lse2_flat =
      lse2.squeeze(-1).to(torch::kFloat32).contiguous().reshape({bsh});
  auto lse3_flat =
      lse3.squeeze(-1).to(torch::kFloat32).contiguous().reshape({bsh});
  auto out_flat = torch::empty(
      {bsh, d},
      torch::TensorOptions().dtype(torch::kFloat32).device(out1.device()));

  aclTensor* local_out_acls[3] = {torch_to_acl_tensor_row_major_nd(local_out1_flat),
                                  torch_to_acl_tensor_row_major_nd(local_out2_flat),
                                  torch_to_acl_tensor_row_major_nd(local_out3_flat)};
  aclTensor* lse_acls[3] = {torch_to_acl_tensor_row_major_nd(lse1_flat),
                            torch_to_acl_tensor_row_major_nd(lse2_flat),
                            torch_to_acl_tensor_row_major_nd(lse3_flat)};
  aclTensor* out_acl = torch_to_acl_tensor_row_major_nd(out_flat);

  aclTensorList* local_out_list = aclCreateTensorList(local_out_acls, 3);
  CHECK_NE(local_out_list, nullptr);
  aclTensorList* lse_list = aclCreateTensorList(lse_acls, 3);
  CHECK_NE(lse_list, nullptr);

  uint64_t ws_size = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnAttentionUpdateGetWorkspaceSize(
      lse_list,
      local_out_list,
      /*updateType=*/0,
      out_acl,
      /*lseOut=*/nullptr,
      &ws_size,
      &executor);
  CHECK_EQ(ret, ACL_SUCCESS) << "aclnnAttentionUpdateGetWorkspaceSize failed: " << ret;

  void* ws_ptr = nullptr;
  if (ws_size > 0) {
    CHECK_EQ(aclrtMalloc(&ws_ptr, ws_size, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS)
        << "aclrtMalloc AttentionUpdate workspace";
  }

  ret = aclnnAttentionUpdate(ws_ptr, ws_size, executor, stream);
  CHECK_EQ(ret, ACL_SUCCESS) << "aclnnAttentionUpdate failed: " << ret;
  CHECK_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS)
      << "aclrtSynchronizeStream after aclnnAttentionUpdate";
  if (ws_ptr != nullptr) {
    CHECK_EQ(aclrtFree(ws_ptr), ACL_SUCCESS) << "aclrtFree workspace";
  }

  torch::Tensor out_bnsd = out_flat.clone().to(out1.scalar_type());
  torch::Tensor out_ret = out_bnsd.view({b, n, s, d});

  aclDestroyTensorList(local_out_list);
  aclDestroyTensorList(lse_list);
  aclDestroyTensor(out_acl);
  return out_ret;
}

torch::Tensor create_full_attention_mask(int64_t q_len,
                                         int64_t kv_len,
                                         torch::Device device) {
  return torch::zeros({1, 1, q_len, kv_len},
                      torch::TensorOptions().dtype(torch::kBool).device(device));
}

torch::Tensor create_diagonal_mask(int64_t seq_len, torch::Device device) {
  auto mask =
      ~torch::eye(seq_len, torch::TensorOptions().dtype(torch::kBool).device(device));
  return mask.view({1, 1, seq_len, seq_len}).contiguous();
}

torch::Tensor create_compressed_causal_mask_2048(torch::Device device) {
  return torch::ones({1, 1, 2048, 2048},
                     torch::TensorOptions().dtype(torch::kBool).device(device))
      .triu(1)
      .contiguous();
}

void run_genrec_v2_single_batch(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const AttentionMetadata& attn_metadata,
    torch::Tensor& output) {

  int32_t device_id = query.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  
  int64_t batch_size = query.size(0);
  int64_t num_heads = query.size(1);
  int64_t num_kv_heads = key.size(1);
  float scale = 1.0f / std::sqrt(static_cast<float>(query.size(3)));

  constexpr int64_t kSparseModeFullMask = 0;
  constexpr int64_t kSparseModeLeftUpCausal = 2;
  constexpr int64_t kDefaultWindow = 2147483647LL;

  auto out_opts = torch::TensorOptions()
                        .dtype(query.scalar_type())
                        .device(query.device());
  auto lse_opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(query.device());
    
  // 用哨兵值填充，便于区分"kernel 没写"和"kernel 写出了 0"。
  // 若 kernel 执行后仍含 kSentinel，则说明 kernel 压根没写该位置。
  constexpr float kSentinel = -31415.0f;
  auto make_out = [&](int64_t seq_len) {
    return torch::full({1, num_heads, seq_len, query.size(3)}, kSentinel, out_opts);
  };
  auto make_lse = [&](int64_t seq_len) {
    return torch::full({1, num_heads, seq_len, 1}, kSentinel, lse_opts);
  };


  for (int64_t i = 0; i < batch_size; ++i) {
    const int64_t h = attn_metadata.history_lens.select(0, i).item<int64_t>();
    const int64_t c = attn_metadata.context_lens.select(0, i).item<int64_t>();
    const int64_t r = attn_metadata.real_time_lens.select(0, i).item<int64_t>();
    const int64_t t = attn_metadata.target_lens.select(0, i).item<int64_t>();
    const int64_t total = h + c + r + t;

    auto lse = torch::empty(
      {1, num_heads, total, 1},
      torch::TensorOptions().dtype(torch::kFloat32).device(query.device()));
    
    // ---history 部分---
    auto hist_query = query.slice(0, i, i + 1).slice(2, 0, h).contiguous();
    auto hist_key = key.slice(0, i, i + 1).slice(2, 0, h).contiguous();
    auto hist_value = value.slice(0, i, i + 1).slice(2, 0, h).contiguous();
    auto hist_out = make_out(h);

    SegmentAttentionMetadata hist_attn_metadata;
    hist_attn_metadata.attn_mask = attn_metadata.compressed_causal_mask;
    hist_attn_metadata.sparse_mode = kSparseModeLeftUpCausal;
    hist_attn_metadata.pre_tokens = kDefaultWindow;
    hist_attn_metadata.next_tokens = kDefaultWindow;

    AclPlan hist_plan = plan_segment_attention(hist_query,
                                              hist_key,
                                              hist_value,
                                              hist_attn_metadata,
                                              hist_out, 
                                              std::nullopt);
    execute_planned_attention(hist_plan, stream);
    destroy_planned_attention(hist_plan);
    output.slice(0, i, i + 1).slice(2, 0, h).copy_(hist_out);
    expect_npu_attention_matches_reference("history",
                                           stream,
                                           output.slice(0, i, i + 1).slice(2, 0, h),
                                           hist_query,
                                           hist_key,
                                           hist_value,
                                           optional_attn_mask(hist_attn_metadata),
                                           scale);
    

    // ---context + real_time + target 部分---
    auto crt_out = make_out(c + r + t);
    auto crt_lse = make_lse(c + r + t);

    auto crt_query = query.slice(0, i, i + 1).slice(2, h, h + c + r + t).contiguous();
    auto crt_key = key.slice(0, i, i + 1).slice(2, 0, h + c).contiguous();
    auto crt_value = value.slice(0, i, i + 1).slice(2, 0, h + c).contiguous();

    SegmentAttentionMetadata crt_attn_metadata;
    crt_attn_metadata.attn_mask = create_full_attention_mask(c + r + t, h + c, query.device());
    crt_attn_metadata.sparse_mode = kSparseModeFullMask;
    crt_attn_metadata.pre_tokens = kDefaultWindow;
    crt_attn_metadata.next_tokens = kDefaultWindow;

    AclPlan crt_plan = plan_segment_attention(crt_query, 
                                              crt_key, 
                                              crt_value, 
                                              crt_attn_metadata, 
                                              crt_out, 
                                              crt_lse);
    execute_planned_attention(crt_plan, stream);
    destroy_planned_attention(crt_plan);
    expect_npu_lse_matches_reference("crt_context_rt_t",
                                     stream,
                                     crt_lse,
                                     crt_query,
                                     crt_key,
                                     optional_attn_mask(crt_attn_metadata),
                                     scale,
                                     "batch0_02_crt_lse");
    expect_npu_attention_matches_reference("crt_context_rt_t",
                                           stream,
                                           crt_out,
                                           crt_query,
                                           crt_key,
                                           crt_value,
                                           optional_attn_mask(crt_attn_metadata),
                                           scale);

    // context 部分来自 step2a，直接回写最终输出。
    output.slice(0, i, i + 1).slice(2, h, h + c).copy_(crt_out.slice(2, 0, c));

    // ---real_time + target 部分---
    auto rt_out = make_out(r + t);
    auto rt_lse = make_lse(r + t);

    auto rt_query = query.slice(0, i, i + 1).slice(2, h + c, h + c + r + t).contiguous();
    auto rt_key = key.slice(0, i, i + 1).slice(2, h + c, h + c + r).contiguous();
    auto rt_value = value.slice(0, i, i + 1).slice(2, h + c, h + c + r).contiguous();

    SegmentAttentionMetadata rt_attn_metadata;
    rt_attn_metadata.attn_mask = attn_metadata.compressed_causal_mask;
    rt_attn_metadata.sparse_mode = kSparseModeLeftUpCausal;
    rt_attn_metadata.pre_tokens = kDefaultWindow;
    rt_attn_metadata.next_tokens = kDefaultWindow;

    AclPlan rt_plan = plan_segment_attention(rt_query, 
                                            rt_key, 
                                            rt_value, 
                                            rt_attn_metadata, 
                                            rt_out, 
                                            rt_lse);
    execute_planned_attention(rt_plan, stream);
    destroy_planned_attention(rt_plan);
    expect_npu_lse_matches_reference("rt_realtime_t",
                                     stream,
                                     rt_lse,
                                     rt_query,
                                     rt_key,
                                     optional_attn_mask(rt_attn_metadata),
                                     scale,
                                     "batch0_03_rt_lse");
    expect_npu_attention_matches_reference("rt_realtime_t",
                                           stream,
                                           rt_out,
                                           rt_query,
                                           rt_key,
                                           rt_value,
                                           optional_attn_mask(rt_attn_metadata),
                                           scale);

    // ---target 部分---
    auto target_out = make_out(t);
    auto target_lse = make_lse(t);

    auto target_query = query.slice(0, i, i + 1).slice(2, h + c + r, h + c + r + t).contiguous();
    auto target_key = key.slice(0, i, i + 1).slice(2, h + c + r, h + c + r + t).contiguous();
    auto target_value = value.slice(0, i, i + 1).slice(2, h + c + r, h + c + r + t).contiguous();

    SegmentAttentionMetadata target_attn_metadata;
    target_attn_metadata.attn_mask = attn_metadata.diagonal_mask;
    target_attn_metadata.sparse_mode = kSparseModeFullMask;
    target_attn_metadata.pre_tokens = 0;
    target_attn_metadata.next_tokens = 0;

    AclPlan target_plan = plan_segment_attention(target_query, 
                                                target_key, 
                                                target_value, 
                                                target_attn_metadata, 
                                                target_out, 
                                                target_lse);
    execute_planned_attention(target_plan, stream);
    destroy_planned_attention(target_plan);
    expect_npu_lse_matches_reference("target",
                                     stream,
                                     target_lse,
                                     target_query,
                                     target_key,
                                     optional_attn_mask(target_attn_metadata),
                                     scale,
                                     "batch0_04_target_lse");
    expect_npu_attention_matches_reference("target",
                                           stream,
                                           target_out,
                                           target_query,
                                           target_key,
                                           target_value,
                                           optional_attn_mask(target_attn_metadata),
                                           scale);

    // lse combine：与 genrec_attention_test（tm 风格 FP32 展平 + row-major ND）同路径，aclnnAttentionUpdate。
    auto combined_real_time_output = output.slice(0, i, i + 1).slice(2, h + c, h + c + r);
    {
      auto rt_merged = combine_two_way_attention_update_like_benchmark(
          crt_out.slice(2, c, c + r),
          crt_lse.slice(2, c, c + r),
          rt_out.slice(2, 0, r),
          rt_lse.slice(2, 0, r),
          stream);
      combined_real_time_output.copy_(rt_merged);
    }

    auto combined_target_output = output.slice(0, i, i + 1).slice(2, h + c + r, h + c + r + t);
    {
      auto tgt_merged = combine_three_way_attention_update_like_benchmark(
          crt_out.slice(2, c + r, c + r + t),
          crt_lse.slice(2, c + r, c + r + t),
          rt_out.slice(2, r, r + t),
          rt_lse.slice(2, r, r + t),
          target_out,
          target_lse,
          stream);
      combined_target_output.copy_(tgt_merged);
    }

    print_genrec_merge_pipeline_diagnostics(i,
                                            query,
                                            key,
                                            value,
                                            output,
                                            crt_out,
                                            crt_lse,
                                            rt_out,
                                            rt_lse,
                                            target_out,
                                            target_lse,
                                            h,
                                            c,
                                            r,
                                            t,
                                            scale);
  }
}

}  // namespace

TEST_F(GenRecDirectRunTest, DirectRunGenRecV2SingleBatch) {
  // 各段 seq / head_dim 取中等规模，避免过小 shape 下 aclnn FusedInfer 出现「只清零不写 LSE」等边界行为；
  // dump 仍见 kGenRecStructuredDumpDir。
  constexpr int64_t kBatchSize = 1;
  constexpr int64_t kNumHeads = 8;
  constexpr int64_t kNumKvHeads = 8;
  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kHistory = 1300;
  constexpr int64_t kContext = 8;
  constexpr int64_t kRealtime = 400;
  constexpr int64_t kTarget = 800;
  const int64_t total = kHistory + kContext + kRealtime + kTarget;

  // 固定 RNG：torch::manual_seed 只控制 CPU 默认生成器；torch::randn(..., device=NPU) 会走 NPU
  // Generator，不单独设种子时每次仍可能不同。这里在 CPU 上 randn 再 .to(NPU)，输入可完全复现。
  // constexpr uint64_t kGenRecStructuredTestRngSeed = 20260404ULL;
  // torch::manual_seed(kGenRecStructuredTestRngSeed);

  // BNSD
  auto query =
      torch::randn({kBatchSize, kNumHeads, total, kHeadDim}, opts_);
  auto key =
      torch::randn({kBatchSize, kNumKvHeads, total, kHeadDim}, opts_);
  auto value =
      torch::randn({kBatchSize, kNumKvHeads, total, kHeadDim}, opts_);
  auto output = torch::empty_like(query);

  dump_tensor_text("batch0_00_in_query.txt", stream_, query);
  dump_tensor_text("batch0_00_in_key.txt", stream_, key);
  dump_tensor_text("batch0_00_in_value.txt", stream_, value);

  AttentionMetadata attn_metadata;
  // 每段长度为 int64 一维张量 [batch_size]，与 run_genrec 中 select(0, i) 对齐。

  auto len_opts =
      torch::TensorOptions().dtype(torch::kInt64);
  attn_metadata.history_lens = torch::tensor({kHistory}, len_opts);
  attn_metadata.context_lens = torch::tensor({kContext}, len_opts);
  attn_metadata.real_time_lens = torch::tensor({kRealtime}, len_opts);
  attn_metadata.target_lens = torch::tensor({kTarget}, len_opts);

  attn_metadata.compressed_causal_mask = create_compressed_causal_mask_2048(query.device());
  attn_metadata.diagonal_mask = create_diagonal_mask(kTarget, query.device());

  run_genrec_v2_single_batch(query,
                             key,
                             value,
                             attn_metadata,
                             output);

  ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
  // 与 run_genrec_v2_single_batch 内 plan_segment_attention 使用同一 float scale，
  // 再转 double 喂 ReferenceGenRecGold，避免 sqrt 精度不一致。
  const double scale = static_cast<double>(
      1.0f / std::sqrt(static_cast<float>(query.size(3))));
  expect_genrec_v2_output_vs_reference_genrec_gold(
      "DirectRunGenRecV2SingleBatch",
      output,
      query,
      key,
      value,
      kHistory,
      kContext,
      kRealtime,
      kTarget,
      scale);
  dump_tensor_text("batch0_07_out_full.txt", stream_, output);
  if (kDumpGenRecStructuredArtifacts) {
    std::error_code ec;
    std::filesystem::create_directories(kGenRecStructuredDumpDir, ec);
    const auto idx_path =
        std::filesystem::path(kGenRecStructuredDumpDir) / "INDEX.txt";
    std::ofstream idx(idx_path.string());
    if (idx.good()) {
      idx << "GenRec structured test artifacts (FP32 text; first line # shape=...)\n";
      idx << "batch0_00_in_{query,key,value}.txt  inputs (NPU tensor dumped as FP32)\n";
      idx << "Per fused-attention stage: batch0_0{1..4}_*_attn_{npu,ref,diff_abs}.txt\n";
      idx << "  CPU ref = reference_scaled_dot_product_attention (same Q/K/V/mask/scale)\n";
      idx << "  npu = aclnnFusedInferAttentionScoreV3 output (FP16->FP32)\n";
      idx << "Per-segment softmax LSE: batch0_0{2,3,4}_*_lse_{npu,ref,diff_abs}.txt\n";
      idx << "  ref = logsumexp(scores) on CPU FP32, same scores as attn ref (scale*QK^T+mask)\n";
      idx << "  npu = aclnnFusedInferAttentionScoreV3 softmaxLse output\n";
      idx << "LSE merge: batch0_0{5,6}_lse_merge_*_{npu,ref,diff_abs}.txt\n";
      idx << "  ref = CombineFlashOutputs / CombineFlashOutputsThree on CPU FP32\n";
      idx << "batch0_07_out_full.txt  assembled full output (NPU only)\n";
      idx << "stderr: [RefAttnDiff] attn / [RefLseDiff] per-segment LSE / "
             "[GenRecV2][Precision] + [RefFinalGoldDiff] vs ReferenceGenRecGold\n";
    }
    std::fprintf(stderr,
                 "[GenRecStructuredDump] FP32 text under ./%s/ (see INDEX.txt)\n",
                 kGenRecStructuredDumpDir);
    std::fflush(stderr);
  }

  ASSERT_EQ(output.dim(), 4);
  ASSERT_EQ(output.size(0), 1);
  ASSERT_EQ(output.size(1), kBatchSize * kNumHeads);
  ASSERT_EQ(output.size(2), total);
  ASSERT_EQ(output.size(3), kBatchSize * kHeadDim);

  auto out_cpu = output.to(torch::kCPU).to(torch::kFloat32);
  EXPECT_TRUE(torch::isfinite(out_cpu).all().item<bool>());
  EXPECT_GT(out_cpu.abs().sum().item<float>(), 0.0f);
}

}  // namespace xllm::kernel::npu::test
