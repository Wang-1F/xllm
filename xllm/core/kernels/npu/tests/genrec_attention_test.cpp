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

// Pull in the shared fixture + utilities from segmented_prefill_attention_test.cpp.
// We compile only this TU (see CMakeLists) to avoid duplicate definitions.
#include "segmented_prefill_attention_test.cpp"
#include "aclnnop/aclnn_attention_update.h"

#include <cstdio>
#include <sstream>
#include <string>

namespace {

std::string TorchTensorBrief(const torch::Tensor& t) {
  std::ostringstream oss;
  oss << "sizes=[";
  for (int i = 0; i < t.dim(); ++i) {
    if (i) oss << ",";
    oss << t.size(i);
  }
  oss << "] strides=[";
  for (int i = 0; i < t.dim(); ++i) {
    if (i) oss << ",";
    oss << t.stride(i);
  }
  oss << "] dtype=" << static_cast<int>(t.scalar_type())
      << " contig=" << (t.is_contiguous() ? 1 : 0) << " numel=" << t.numel();
  return oss.str();
}

// Debug aclnnAttentionUpdate inputs (stderr — works without glog init).
// For performance runs keep this disabled to avoid host-side overhead.
constexpr bool kDebugAttentionUpdate = false;
void LogAttentionUpdateBeforeGetWs(const char* where,
                                   int64_t b,
                                   int64_t n,
                                   int64_t seq,
                                   int64_t d,
                                   int64_t bsh,
                                   int sp,
                                   int64_t update_type,
                                   const void* lse_out,
                                   const torch::Tensor& lo1,
                                   const torch::Tensor& lo2,
                                   const torch::Tensor& lse_a,
                                   const torch::Tensor& lse_b,
                                   const torch::Tensor& out,
                                   const torch::Tensor* lo3,
                                   const torch::Tensor* lse_c) {
  if (!kDebugAttentionUpdate) return;
  fprintf(stderr,
          "[AttentionUpdate][%s] b=%lld n=%lld s=%lld d=%lld bsh=%lld sp=%d "
          "updateType=%lld lseOut=%p\n",
          where, static_cast<long long>(b), static_cast<long long>(n),
          static_cast<long long>(seq), static_cast<long long>(d),
          static_cast<long long>(bsh), sp,
          static_cast<long long>(update_type), lse_out);
  fprintf(stderr, "  device=%s\n", out.device().str().c_str());
  fprintf(stderr, "  localOut[0]: %s\n", TorchTensorBrief(lo1).c_str());
  fprintf(stderr, "  localOut[1]: %s\n", TorchTensorBrief(lo2).c_str());
  if (lo3 != nullptr) {
    fprintf(stderr, "  localOut[2]: %s\n", TorchTensorBrief(*lo3).c_str());
  }
  fprintf(stderr, "  lse[0]: %s\n", TorchTensorBrief(lse_a).c_str());
  fprintf(stderr, "  lse[1]: %s\n", TorchTensorBrief(lse_b).c_str());
  if (lse_c != nullptr) {
    fprintf(stderr, "  lse[2]: %s\n", TorchTensorBrief(*lse_c).c_str());
  }
  fprintf(stderr, "  out: %s\n", TorchTensorBrief(out).c_str());
  fprintf(stderr,
          "  check: lse[0].size(0)=%lld localOut[0].size(0)=%lld "
          "out.size(0)=%lld (should be equal)\n",
          static_cast<long long>(lse_a.size(0)),
          static_cast<long long>(lo1.size(0)),
          static_cast<long long>(out.size(0)));
  fflush(stderr);
}

// Step-by-step stderr trace (last line before crash = faulting stage).
inline void AuTrace(const char* tag, const char* step) {
  if (!kDebugAttentionUpdate) return;
  fprintf(stderr, "[AU][%s] %s\n", tag, step);
  fflush(stderr);
}

}  // namespace

namespace xllm::kernel::npu::test {

using namespace util;

// ===========================================================================
// GenRec benchmark: multi-batch pipelined attention + combine (V1, 5 launches)
// ===========================================================================
TEST_F(SegmentedPrefillAttentionTest, BenchGenRecAttention) {
  constexpr int64_t kNumHeads = 8;
  constexpr int64_t kNumKvHeads = 8;
  constexpr int64_t kHeadDim = 128;
  const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
  constexpr int kBatchSize = 3;

  const SeqPartLengths parts{1300, 8, 400, 800};
  const int64_t seq_len = parts.total();  // 2508

  const int64_t total_S = kBatchSize * seq_len;
  // [bs, n, d]
  auto q_all = torch::randn({total_S, kNumHeads, kHeadDim}, opts_);
  auto k_all = torch::randn({total_S, kNumKvHeads, kHeadDim}, opts_);
  auto v_all = torch::randn({total_S, kNumKvHeads, kHeadDim}, opts_);

  std::vector<int64_t> seq_lens(kBatchSize + 1, 0);
  for (int b = 0; b < kBatchSize; ++b) seq_lens[b + 1] = seq_lens[b] + seq_len;

  std::vector<std::vector<AttnCallDesc>> all_batches;
  all_batches.reserve(kBatchSize);
  for (int b = 0; b < kBatchSize; ++b) {
    int64_t s = seq_lens[b];
    all_batches.push_back(BuildGenRecCalls(q_all.slice(0, s, s + seq_len),
                                           k_all.slice(0, s, s + seq_len),
                                           v_all.slice(0, s, s + seq_len), parts,
                                           kNumHeads, kNumKvHeads, device_));
  }

  for (auto& batch : all_batches)
    for (auto& call : batch) call.CreateAcl();

  auto combine_two_way_attention_update =
      [&](const torch::Tensor& out1,
          const torch::Tensor& lse1,
          const torch::Tensor& out2,
          const torch::Tensor& lse2) -> torch::Tensor {
    CHECK_EQ(out1.sizes(), out2.sizes());
    CHECK_EQ(lse1.sizes(), lse2.sizes());
    CHECK_EQ(out1.dim(), 4);
    CHECK_EQ(lse1.dim(), 4);

    const int64_t b = out1.size(0);
    const int64_t n = out1.size(1);
    const int64_t s = out1.size(2);
    const int64_t d = out1.size(3);
    const int64_t bsh = b * n * s;

    // See V2 combine: FP32 localOut/out for aclnnAttentionUpdate on this stack.
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

    aclTensor* local_out_acls[2] = {TorchToAclTensorRowMajorNd(local_out1_flat),
                                    TorchToAclTensorRowMajorNd(local_out2_flat)};
    aclTensor* lse_acls[2] = {TorchToAclTensorRowMajorNd(lse1_flat),
                              TorchToAclTensorRowMajorNd(lse2_flat)};
    aclTensor* out_acl = TorchToAclTensorRowMajorNd(out_flat);

    aclTensorList* local_out_list = aclCreateTensorList(local_out_acls, 2);
    CHECK_NE(local_out_list, nullptr);
    aclTensorList* lse_list = aclCreateTensorList(lse_acls, 2);
    CHECK_NE(lse_list, nullptr);

    LogAttentionUpdateBeforeGetWs("BenchGenRecAttention_V1_2way", b, n, s, d,
                                    bsh, 2, 0, nullptr, local_out1_flat,
                                    local_out2_flat, lse1_flat, lse2_flat,
                                    out_flat, nullptr, nullptr);

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
    if (kDebugAttentionUpdate) {
      fprintf(stderr,
              "[AttentionUpdate][BenchGenRecAttention_V1_2way] "
              "aclnnAttentionUpdateGetWorkspaceSize -> ret=%d ws_size=%llu "
              "executor=%p\n",
              static_cast<int>(ret),
              static_cast<unsigned long long>(ws_size),
              static_cast<void*>(executor));
      fflush(stderr);
    }
    AuTrace("V1_2way", "01 after GetWs log");
    CHECK_EQ(ret, ACL_SUCCESS) << "aclnnAttentionUpdateGetWorkspaceSize failed: "
                               << ret;
    AuTrace("V1_2way", "02 CHECK GetWs ok");

    void* ws_ptr = nullptr;
    if (ws_size > 0) {
      AuTrace("V1_2way", "03 aclrtMalloc workspace (avoid torch cross-call reuse)");
      CHECK_EQ(aclrtMalloc(&ws_ptr, ws_size, ACL_MEM_MALLOC_HUGE_FIRST),
               ACL_SUCCESS)
          << "aclrtMalloc AttentionUpdate workspace";
      if (kDebugAttentionUpdate) {
        fprintf(stderr, "[AU][V1_2way] 04 ws_ptr=%p ws_size=%llu\n", ws_ptr,
                static_cast<unsigned long long>(ws_size));
        fflush(stderr);
      }
    } else {
      AuTrace("V1_2way", "04 ws_size==0 skip device workspace");
    }

    AuTrace("V1_2way", "05 before aclnnAttentionUpdate");
    ret = aclnnAttentionUpdate(ws_ptr, ws_size, executor, stream_);
    AuTrace("V1_2way", "06 after aclnnAttentionUpdate (sync still pending)");
    CHECK_EQ(ret, ACL_SUCCESS) << "aclnnAttentionUpdate failed: " << ret;
    // aclnnAttentionUpdate is asynchronous; destroying aclTensor / lists while
    // the device may still read them causes segfault on the next launch.
    AuTrace("V1_2way", "07 before aclrtSynchronizeStream");
    CHECK_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS)
        << "aclrtSynchronizeStream after aclnnAttentionUpdate";
    AuTrace("V1_2way", "08 after aclrtSynchronizeStream");
    if (ws_ptr != nullptr) {
      CHECK_EQ(aclrtFree(ws_ptr), ACL_SUCCESS) << "aclrtFree workspace";
      ws_ptr = nullptr;
    }
    // Must release executor before the next GetWorkspaceSize; otherwise the
    // runtime can segfault on the following aclnn op (see aclDestroyAclOpExecutor).
    

    // out_acl wraps out_flat.data_ptr(); aclDestroyTensor(out_acl) can
    // invalidate that device memory for PyTorch — clone BEFORE any aclDestroy.
    AuTrace("V1_2way", "10a clone+cast output (before aclDestroyTensor)");
    torch::Tensor output_flag_copy = out_flat.clone();
    output_flag_copy = output_flag_copy.to(out1.scalar_type());
    output_flag_copy = output_flag_copy.contiguous();
    AuTrace("V1_2way", "10b view BNSD on detached copy");
    torch::Tensor out_ret = output_flag_copy.view({b, n, s, d});

    // aclDestroyTensorList already releases tensors in the list; do NOT
    // aclDestroyTensor those same pointers again (double-free → next GetWs crash).

    AuTrace("V1_2way", "11 aclDestroyTensorList(local_out), aclDestroyTensorList(lse)");
    if (local_out_list != nullptr) {
      aclDestroyTensorList(local_out_list);
    }
    if (lse_list != nullptr) {
      aclDestroyTensorList(lse_list);
    }
    // aclDestroyTensor(local_out_acls[0]);
    // aclDestroyTensor(local_out_acls[1]);
    // aclDestroyTensor(lse_acls[0]);
    // aclDestroyTensor(lse_acls[1]);
    aclDestroyTensor(out_acl);
    AuTrace("V1_2way", "16 acl cleanup done, return");
    return out_ret;
  };

  const char* step_tags[] = {"hist_causal", "ctx_rt_full", "rt_causal",
                             "tgt_full", "tgt_diag"};

  uint64_t max_ws = 0;
  for (size_t b = 0; b < all_batches.size(); ++b)
    for (size_t i = 0; i < all_batches[b].size(); ++i) {
      std::string tag =
          std::string("b") + std::to_string(b) + "_" + step_tags[i];
      const auto& call = all_batches[b][i];
      auto [ws, _] = PlanFlexAttention(call, kNumHeads, kNumKvHeads, scale,
                                       tag.c_str(), call.sparse_mode,
                                       call.pre_tokens, call.next_tokens);
      max_ws = std::max(max_ws, ws);
    }

  void* workspace = nullptr;
  if (max_ws > 0) {
    CHECK_EQ(aclrtMalloc(&workspace, max_ws, ACL_MEM_MALLOC_HUGE_FIRST),
             ACL_SUCCESS);
  }

  // Attention-only event timeline profiling (no per-kernel synchronize).
  {
    static bool kPrintedTimeline = false;
    if (!kPrintedTimeline && !all_batches.empty() && max_ws > 0) {
      auto& batch0 = all_batches.at(0);

      // Warm up once to exclude first-run kernel initialization/compilation.
      constexpr int kTimelineWarmupIters = 2;
      for (int wi = 0; wi < kTimelineWarmupIters; ++wi) {
        for (size_t ci = 0; ci < batch0.size(); ++ci) {
          auto& call = batch0[ci];
          auto [ws_size, executor] = PlanFlexAttention(
              call, kNumHeads, kNumKvHeads, scale, step_tags[ci],
              call.sparse_mode, call.pre_tokens, call.next_tokens);
          CHECK_LE(ws_size, max_ws);
          CHECK_EQ(aclnnFusedInferAttentionScoreV3(workspace, ws_size,
                                                     executor, stream_),
                   0);
        }

        // Keep combine behavior consistent with the measured round.
        auto rt_out_2a = batch0[1].out.slice(2, parts.context);
        auto rt_lse_2a = batch0[1].lse.slice(2, parts.context);

        (void)combine_two_way_attention_update(
          batch0[3].out, batch0[3].lse, batch0[4].out, batch0[4].lse);
        CHECK_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
        (void)combine_two_way_attention_update(
            rt_out_2a, rt_lse_2a, batch0[2].out, batch0[2].lse);
        
        CHECK_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
      }

      aclrtEvent ev_start[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
      aclrtEvent ev_end[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
      std::chrono::steady_clock::time_point host_submit[5];

      const uint32_t flags = ACL_EVENT_TIME_LINE;
      for (int i = 0; i < 5; ++i) {
        CHECK_EQ(aclrtCreateEventWithFlag(&ev_start[i], flags), ACL_SUCCESS);
        CHECK_EQ(aclrtCreateEventWithFlag(&ev_end[i], flags), ACL_SUCCESS);
      }

      for (size_t ci = 0; ci < batch0.size(); ++ci) {
        auto& call = batch0[ci];
        auto [ws_size, executor] = PlanFlexAttention(
            call, kNumHeads, kNumKvHeads, scale, step_tags[ci], call.sparse_mode,
            call.pre_tokens, call.next_tokens);
        CHECK_LE(ws_size, max_ws);

        CHECK_EQ(aclrtRecordEvent(ev_start[ci], stream_), ACL_SUCCESS);
        host_submit[ci] = std::chrono::steady_clock::now();

        CHECK_EQ(aclnnFusedInferAttentionScoreV3(workspace, ws_size, executor,
                                                stream_),
                 0);
        CHECK_EQ(aclrtRecordEvent(ev_end[ci], stream_), ACL_SUCCESS);
      }

      // Combine (not included in total_attn).
      auto rt_out_2a = batch0[1].out.slice(2, parts.context);
      auto rt_lse_2a = batch0[1].lse.slice(2, parts.context);
      (void)combine_two_way_attention_update(
          rt_out_2a, rt_lse_2a, batch0[2].out, batch0[2].lse);
      (void)combine_two_way_attention_update(
          batch0[3].out, batch0[3].lse, batch0[4].out, batch0[4].lse);

      CHECK_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);

      double attn_ms[5] = {0, 0, 0, 0, 0};
      for (int i = 0; i < 5; ++i) {
        float ms = 0.0f;
        CHECK_EQ(aclrtEventElapsedTime(&ms, ev_start[i], ev_end[i]),
                 ACL_SUCCESS);
        attn_ms[i] = static_cast<double>(ms);
      }

      float total_attn_ms_f = 0.0f;
      CHECK_EQ(aclrtEventElapsedTime(&total_attn_ms_f, ev_start[0], ev_end[4]),
               ACL_SUCCESS);
      double total_attn_ms = static_cast<double>(total_attn_ms_f);

      double device_gap_ms[4] = {0, 0, 0, 0};
      for (int i = 0; i < 4; ++i) {
        float gap_ms = 0.0f;
        CHECK_EQ(aclrtEventElapsedTime(&gap_ms, ev_end[i], ev_start[i + 1]),
                 ACL_SUCCESS);
        device_gap_ms[i] = static_cast<double>(gap_ms);
      }

      double host_interval_ms[4] = {0, 0, 0, 0};
      for (int i = 0; i < 4; ++i) {
        host_interval_ms[i] =
            std::chrono::duration<double, std::milli>(host_submit[i + 1] -
                                                         host_submit[i])
                .count();
      }

      const double kIdleEpsMs = 0.05;
      bool host_likely_overlap = true;
      for (int i = 0; i < 4; ++i) {
        if (host_interval_ms[i] > attn_ms[i]) host_likely_overlap = false;
      }
      bool device_has_idle = false;
      for (int i = 0; i < 4; ++i) {
        if (device_gap_ms[i] > kIdleEpsMs) device_has_idle = true;
      }

      LOG(INFO) << "[GenRec][TimelineAttn] "
                << "attn(hist)=" << attn_ms[0] << "ms, "
                << "attn(ctx_rt_full)=" << attn_ms[1] << "ms, "
                << "attn(rt_causal)=" << attn_ms[2] << "ms, "
                << "attn(tgt_full)=" << attn_ms[3] << "ms, "
                << "attn(tgt_diag)=" << attn_ms[4] << "ms, "
                << "total_attn=" << total_attn_ms << "ms, "
                << "device_gap(ms)=" << device_gap_ms[0] << ","
                << device_gap_ms[1] << "," << device_gap_ms[2] << ","
                << device_gap_ms[3] << ", host_interval(ms)="
                << host_interval_ms[0] << "," << host_interval_ms[1] << ","
                << host_interval_ms[2] << "," << host_interval_ms[3]
                << ", overlap=" << (host_likely_overlap && !device_has_idle);

      for (int i = 0; i < 5; ++i) {
        CHECK_EQ(aclrtDestroyEvent(ev_start[i]), ACL_SUCCESS);
        CHECK_EQ(aclrtDestroyEvent(ev_end[i]), ACL_SUCCESS);
      }

      kPrintedTimeline = true;
    }
  }

  double ms = MeasureWallClockMs(
      [&]() {
        for (auto& batch : all_batches) {
          for (size_t ci = 0; ci < batch.size(); ++ci) {
            auto& call = batch[ci];
            auto [ws_size, executor] = PlanFlexAttention(
                call, kNumHeads, kNumKvHeads, scale, step_tags[ci],
                call.sparse_mode, call.pre_tokens, call.next_tokens);
            CHECK_EQ(aclnnFusedInferAttentionScoreV3(workspace, ws_size, executor,
                                                    stream_),
                     0);
          }
          auto rt_out_2a = batch[1].out.slice(2, parts.context);
          auto rt_lse_2a = batch[1].lse.slice(2, parts.context);
          (void)combine_two_way_attention_update(
              rt_out_2a, rt_lse_2a, batch[2].out, batch[2].lse);
          (void)combine_two_way_attention_update(
              batch[3].out, batch[3].lse, batch[4].out, batch[4].lse);
        }
      },
      kDeviceId, 3, 10);

  if (max_ws > 0) {
    CHECK_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    aclrtFree(workspace);
  }
  for (auto& batch : all_batches)
    for (auto& call : batch) call.DestroyAcl();

  LOG(INFO) << "[GenRec] " << kBatchSize << " batches x " << parts.total()
            << " tok/seq (h=" << parts.history << " c=" << parts.context
            << " r=" << parts.real_time << " t=" << parts.target << ") | " << ms
            << " ms/iter, " << (ms / kBatchSize) << " ms/batch, "
            << (ms / kBatchSize / 5) << " ms/attn_call";
}

// ===========================================================================
// GenRec V2 benchmark: shared-workspace pipeline + 4 launches + precision check
// ===========================================================================
TEST_F(SegmentedPrefillAttentionTest, BenchGenRecV2Attention) {
  constexpr int64_t kNumHeads = 8;
  constexpr int64_t kNumKvHeads = 8;
  constexpr int64_t kHeadDim = 128;
  const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
  constexpr int kBatchSize = 3;

  const SeqPartLengths parts{1300, 8, 400, 800};
  const int64_t seq_len = parts.total();

  const int64_t total_S = kBatchSize * seq_len;
  auto q_all = torch::randn({total_S, kNumHeads, kHeadDim}, opts_);
  auto k_all = torch::randn({total_S, kNumKvHeads, kHeadDim}, opts_);
  auto v_all = torch::randn({total_S, kNumKvHeads, kHeadDim}, opts_);

  std::vector<int64_t> seq_lens(kBatchSize + 1, 0);
  for (int b = 0; b < kBatchSize; ++b) seq_lens[b + 1] = seq_lens[b] + seq_len;

  std::vector<std::vector<AttnCallDesc>> all_batches;
  all_batches.reserve(kBatchSize);
  for (int b = 0; b < kBatchSize; ++b) {
    int64_t s = seq_lens[b];
    all_batches.push_back(BuildGenRecV2Calls(q_all.slice(0, s, s + seq_len),
                                             k_all.slice(0, s, s + seq_len),
                                             v_all.slice(0, s, s + seq_len),
                                             parts, kNumHeads, kNumKvHeads,
                                             device_));
  }

  for (auto& batch : all_batches)
    for (auto& call : batch) call.CreateAcl();

  auto combine_two_way_attention_update =
      [&](const torch::Tensor& out1,
          const torch::Tensor& lse1,
          const torch::Tensor& out2,
          const torch::Tensor& lse2) -> torch::Tensor {
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

    aclTensor* local_out_acls[2] = {TorchToAclTensorRowMajorNd(local_out1_flat),
                                    TorchToAclTensorRowMajorNd(local_out2_flat)};
    aclTensor* lse_acls[2] = {TorchToAclTensorRowMajorNd(lse1_flat),
                              TorchToAclTensorRowMajorNd(lse2_flat)};
    aclTensor* out_acl = TorchToAclTensorRowMajorNd(out_flat);

    aclTensorList* local_out_list = aclCreateTensorList(local_out_acls, 2);
    CHECK_NE(local_out_list, nullptr);
    aclTensorList* lse_list = aclCreateTensorList(lse_acls, 2);
    CHECK_NE(lse_list, nullptr);

    LogAttentionUpdateBeforeGetWs("BenchGenRecV2_2way", b, n, s, d, bsh, 2, 0,
                                    nullptr, local_out1_flat, local_out2_flat,
                                    lse1_flat, lse2_flat, out_flat, nullptr,
                                    nullptr);

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
    if (kDebugAttentionUpdate) {
      fprintf(stderr,
              "[AttentionUpdate][BenchGenRecV2_2way] "
              "aclnnAttentionUpdateGetWorkspaceSize -> ret=%d ws_size=%llu "
              "executor=%p\n",
              static_cast<int>(ret),
              static_cast<unsigned long long>(ws_size),
              static_cast<void*>(executor));
      fflush(stderr);
    }
    AuTrace("V2_2way", "01 after GetWs log");
    CHECK_EQ(ret, ACL_SUCCESS) << "aclnnAttentionUpdateGetWorkspaceSize failed: "
                               << ret;

    void* ws_ptr = nullptr;
    if (ws_size > 0) {
      CHECK_EQ(aclrtMalloc(&ws_ptr, ws_size, ACL_MEM_MALLOC_HUGE_FIRST),
               ACL_SUCCESS)
          << "aclrtMalloc AttentionUpdate workspace";
      if (kDebugAttentionUpdate) {
        fprintf(stderr, "[AU][V2_2way] ws_ptr=%p size=%llu\n", ws_ptr,
                static_cast<unsigned long long>(ws_size));
        fflush(stderr);
      }
    }

    AuTrace("V2_2way", "04 before aclnnAttentionUpdate");
    ret = aclnnAttentionUpdate(ws_ptr, ws_size, executor, stream_);
    AuTrace("V2_2way", "05 after aclnnAttentionUpdate");
    CHECK_EQ(ret, ACL_SUCCESS) << "aclnnAttentionUpdate failed: " << ret;
    AuTrace("V2_2way", "06 before aclrtSynchronizeStream");
    CHECK_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS)
        << "aclrtSynchronizeStream after aclnnAttentionUpdate";
    AuTrace("V2_2way", "07 after aclrtSynchronizeStream");
    if (ws_ptr != nullptr) {
      CHECK_EQ(aclrtFree(ws_ptr), ACL_SUCCESS) << "aclrtFree workspace";
      ws_ptr = nullptr;
    }

    AuTrace("V2_2way", "08a clone+view before aclDestroy (match V1 order)");
    torch::Tensor out_bnsd = out_flat.clone().to(out1.scalar_type());
    torch::Tensor out_ret = out_bnsd.view({b, n, s, d});

    // if (executor != nullptr) {
    //   AuTrace("V2_2way", "08 before aclDestroyAclOpExecutor");
    //   CHECK_EQ(static_cast<int>(aclDestroyAclOpExecutor(executor)), 0)
    //       << "aclDestroyAclOpExecutor failed";
    //   AuTrace("V2_2way", "09 after aclDestroyAclOpExecutor");
    // }

    // Lists own their aclTensor handles; do not aclDestroyTensor those pointers.
    AuTrace("V2_2way", "10 aclDestroyTensorList x2, aclDestroyTensor(out)");
    aclDestroyTensorList(local_out_list);
    aclDestroyTensorList(lse_list);
    aclDestroyTensor(out_acl);
    AuTrace("V2_2way", "11 acl cleanup done, return");
    return out_ret;
  };
  auto combine_three_way_attention_update =
      [&](const torch::Tensor& out1,
          const torch::Tensor& lse1,
          const torch::Tensor& out2,
          const torch::Tensor& lse2,
          const torch::Tensor& out3,
          const torch::Tensor& lse3) -> torch::Tensor {
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

    aclTensor* local_out_acls[3] = {TorchToAclTensorRowMajorNd(local_out1_flat),
                                    TorchToAclTensorRowMajorNd(local_out2_flat),
                                    TorchToAclTensorRowMajorNd(local_out3_flat)};
    aclTensor* lse_acls[3] = {TorchToAclTensorRowMajorNd(lse1_flat),
                              TorchToAclTensorRowMajorNd(lse2_flat),
                              TorchToAclTensorRowMajorNd(lse3_flat)};
    aclTensor* out_acl = TorchToAclTensorRowMajorNd(out_flat);

    aclTensorList* local_out_list = aclCreateTensorList(local_out_acls, 3);
    CHECK_NE(local_out_list, nullptr);
    aclTensorList* lse_list = aclCreateTensorList(lse_acls, 3);
    CHECK_NE(lse_list, nullptr);

    LogAttentionUpdateBeforeGetWs("BenchGenRecV2_3way", b, n, s, d, bsh, 3, 0,
                                    nullptr, local_out1_flat, local_out2_flat,
                                    lse1_flat, lse2_flat, out_flat,
                                    &local_out3_flat, &lse3_flat);

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
    if (kDebugAttentionUpdate) {
      fprintf(stderr,
              "[AttentionUpdate][BenchGenRecV2_3way] "
              "aclnnAttentionUpdateGetWorkspaceSize -> ret=%d ws_size=%llu "
              "executor=%p\n",
              static_cast<int>(ret),
              static_cast<unsigned long long>(ws_size),
              static_cast<void*>(executor));
      fflush(stderr);
    }
    AuTrace("V2_3way", "01 after GetWs log");
    CHECK_EQ(ret, ACL_SUCCESS) << "aclnnAttentionUpdateGetWorkspaceSize failed: "
                               << ret;

    void* ws_ptr = nullptr;
    if (ws_size > 0) {
      CHECK_EQ(aclrtMalloc(&ws_ptr, ws_size, ACL_MEM_MALLOC_HUGE_FIRST),
               ACL_SUCCESS)
          << "aclrtMalloc AttentionUpdate workspace";
      if (kDebugAttentionUpdate) {
        fprintf(stderr, "[AU][V2_3way] ws_ptr=%p size=%llu\n", ws_ptr,
                static_cast<unsigned long long>(ws_size));
        fflush(stderr);
      }
    }

    AuTrace("V2_3way", "04 before aclnnAttentionUpdate");
    ret = aclnnAttentionUpdate(ws_ptr, ws_size, executor, stream_);
    AuTrace("V2_3way", "05 after aclnnAttentionUpdate");
    CHECK_EQ(ret, ACL_SUCCESS) << "aclnnAttentionUpdate failed: " << ret;
    AuTrace("V2_3way", "06 before aclrtSynchronizeStream");
    CHECK_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS)
        << "aclrtSynchronizeStream after aclnnAttentionUpdate";
    AuTrace("V2_3way", "07 after aclrtSynchronizeStream");
    if (ws_ptr != nullptr) {
      CHECK_EQ(aclrtFree(ws_ptr), ACL_SUCCESS) << "aclrtFree workspace";
      ws_ptr = nullptr;
    }

    AuTrace("V2_3way", "08a clone+view before aclDestroy (match V1 order)");
    torch::Tensor out_bnsd = out_flat.clone().to(out1.scalar_type());
    torch::Tensor out_ret = out_bnsd.view({b, n, s, d});

    // if (executor != nullptr) {
    //   AuTrace("V2_3way", "08 before aclDestroyAclOpExecutor");
    //   CHECK_EQ(static_cast<int>(aclDestroyAclOpExecutor(executor)), 0)
    //       << "aclDestroyAclOpExecutor failed";
    //   AuTrace("V2_3way", "09 after aclDestroyAclOpExecutor");
    // }

    AuTrace("V2_3way", "10 aclDestroyTensorList x2, aclDestroyTensor(out)");
    aclDestroyTensorList(local_out_list);
    aclDestroyTensorList(lse_list);
    aclDestroyTensor(out_acl);
    AuTrace("V2_3way", "11 acl cleanup done, return");
    return out_ret;
  };

  static const char* step_tags[4] = {"v2_hist", "v2_ctx_rt_tgt_full",
                                     "v2_rt_tgt_on_rt", "v2_tgt_diag"};

  uint64_t max_ws = 0;
  for (size_t b = 0; b < all_batches.size(); ++b)
    for (size_t i = 0; i < all_batches[b].size(); ++i) {
      std::string tag =
          std::string("b") + std::to_string(b) + "_" + step_tags[i];
      const auto& call = all_batches[b][i];
      auto [ws, _] = PlanFlexAttention(call, kNumHeads, kNumKvHeads, scale,
                                       tag.c_str(), call.sparse_mode,
                                       call.pre_tokens, call.next_tokens);
      max_ws = std::max(max_ws, ws);
    }

  void* workspace = nullptr;
  if (max_ws > 0) {
    CHECK_EQ(aclrtMalloc(&workspace, max_ws, ACL_MEM_MALLOC_HUGE_FIRST),
             ACL_SUCCESS);
  }

  auto combine_rt = [&](std::vector<AttnCallDesc>& batch) {
    auto rt_out_2a =
        batch[1].out.slice(2, parts.context, parts.context + parts.real_time);
    auto rt_lse_2a =
        batch[1].lse.slice(2, parts.context, parts.context + parts.real_time);
    auto rt_out_2b = batch[2].out.slice(2, 0, parts.real_time);
    auto rt_lse_2b = batch[2].lse.slice(2, 0, parts.real_time);
    return combine_two_way_attention_update(rt_out_2a,
                                            rt_lse_2a,
                                            rt_out_2b,
                                            rt_lse_2b);
  };
  auto combine_tgt = [&](std::vector<AttnCallDesc>& batch) {
    auto tgt_out_2a = batch[1].out.slice(
        2, parts.context + parts.real_time,
        parts.context + parts.real_time + parts.target);
    auto tgt_lse_2a = batch[1].lse.slice(
        2, parts.context + parts.real_time,
        parts.context + parts.real_time + parts.target);
    auto tgt_out_2b = batch[2].out.slice(
        2, parts.real_time, parts.real_time + parts.target);
    auto tgt_lse_2b = batch[2].lse.slice(
        2, parts.real_time, parts.real_time + parts.target);
    return combine_three_way_attention_update(tgt_out_2a,
                                              tgt_lse_2a,
                                              tgt_out_2b,
                                              tgt_lse_2b,
                                              batch[3].out,
                                              batch[3].lse);
  };

  // Attention-only event timeline profiling (no per-kernel synchronize).
  {
    static bool kPrintedTimeline = false;
    if (!kPrintedTimeline && !all_batches.empty() && max_ws > 0) {
      auto& batch0 = all_batches.at(0);

      // Warm up once to exclude first-run kernel initialization/compilation.
      constexpr int kTimelineWarmupIters = 2;
      for (int wi = 0; wi < kTimelineWarmupIters; ++wi) {
        for (size_t ci = 0; ci < batch0.size(); ++ci) {
          auto& call = batch0[ci];
          auto [ws_size, executor] = PlanFlexAttention(
              call, kNumHeads, kNumKvHeads, scale, step_tags[ci],
              call.sparse_mode, call.pre_tokens, call.next_tokens);
          CHECK_LE(ws_size, max_ws);
          CHECK_EQ(aclnnFusedInferAttentionScoreV3(workspace, ws_size,
                                                    executor, stream_),
                   0);
        }
        (void)combine_rt(batch0);
        (void)combine_tgt(batch0);
        CHECK_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
      }

      aclrtEvent ev_start[4] = {nullptr, nullptr, nullptr, nullptr};
      aclrtEvent ev_end[4] = {nullptr, nullptr, nullptr, nullptr};
      std::chrono::steady_clock::time_point host_submit[4];

      const uint32_t flags = ACL_EVENT_TIME_LINE;
      for (int i = 0; i < 4; ++i) {
        CHECK_EQ(aclrtCreateEventWithFlag(&ev_start[i], flags), ACL_SUCCESS);
        CHECK_EQ(aclrtCreateEventWithFlag(&ev_end[i], flags), ACL_SUCCESS);
      }

      for (size_t ci = 0; ci < batch0.size(); ++ci) {
        auto& call = batch0[ci];
        auto [ws_size, executor] =
            PlanFlexAttention(call, kNumHeads, kNumKvHeads, scale,
                              step_tags[ci], call.sparse_mode,
                              call.pre_tokens, call.next_tokens);
        CHECK_LE(ws_size, max_ws);

        CHECK_EQ(aclrtRecordEvent(ev_start[ci], stream_), ACL_SUCCESS);
        host_submit[ci] = std::chrono::steady_clock::now();

        CHECK_EQ(aclnnFusedInferAttentionScoreV3(workspace, ws_size, executor,
                                                stream_),
                 0);
        CHECK_EQ(aclrtRecordEvent(ev_end[ci], stream_), ACL_SUCCESS);
      }

      // Combine (not included in total_attn).
      (void)combine_rt(batch0);
      (void)combine_tgt(batch0);

      CHECK_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);

      double attn_ms[4] = {0, 0, 0, 0};
      for (int i = 0; i < 4; ++i) {
        float ms = 0.0f;
        CHECK_EQ(aclrtEventElapsedTime(&ms, ev_start[i], ev_end[i]),
                 ACL_SUCCESS);
        attn_ms[i] = static_cast<double>(ms);
      }

      float total_attn_ms_f = 0.0f;
      CHECK_EQ(aclrtEventElapsedTime(&total_attn_ms_f, ev_start[0], ev_end[3]),
               ACL_SUCCESS);
      double total_attn_ms = static_cast<double>(total_attn_ms_f);

      double device_gap_ms[3] = {0, 0, 0};
      for (int i = 0; i < 3; ++i) {
        float gap_ms = 0.0f;
        CHECK_EQ(aclrtEventElapsedTime(&gap_ms, ev_end[i], ev_start[i + 1]),
                 ACL_SUCCESS);
        device_gap_ms[i] = static_cast<double>(gap_ms);
      }

      double host_interval_ms[3] = {0, 0, 0};
      for (int i = 0; i < 3; ++i) {
        host_interval_ms[i] =
            std::chrono::duration<double, std::milli>(host_submit[i + 1] -
                                                         host_submit[i])
                .count();
      }

      const double kIdleEpsMs = 0.05;
      bool host_likely_overlap = true;
      for (int i = 0; i < 3; ++i) {
        if (host_interval_ms[i] > attn_ms[i]) host_likely_overlap = false;
      }
      bool device_has_idle = false;
      for (int i = 0; i < 3; ++i) {
        if (device_gap_ms[i] > kIdleEpsMs) device_has_idle = true;
      }

      LOG(INFO) << "[GenRecV2][TimelineAttn] "
                << "attn(hist)=" << attn_ms[0] << "ms, "
                << "attn(ctx_rt_tgt_full)=" << attn_ms[1] << "ms, "
                << "attn(rt_tgt_on_rt)=" << attn_ms[2] << "ms, "
                << "attn(tgt_diag)=" << attn_ms[3] << "ms, "
                << "total_attn=" << total_attn_ms << "ms, "
                << "device_gap(ms)=" << device_gap_ms[0] << "," << device_gap_ms[1]
                << "," << device_gap_ms[2]
                << ", host_interval(ms)=" << host_interval_ms[0] << ","
                << host_interval_ms[1] << "," << host_interval_ms[2]
                << ", overlap=" << (host_likely_overlap && !device_has_idle);

      for (int i = 0; i < 4; ++i) {
        CHECK_EQ(aclrtDestroyEvent(ev_start[i]), ACL_SUCCESS);
        CHECK_EQ(aclrtDestroyEvent(ev_end[i]), ACL_SUCCESS);
      }

      kPrintedTimeline = true;
    }
  }

  double ms = MeasureWallClockMs(
      [&]() {
        for (auto& batch : all_batches) {
          for (size_t ci = 0; ci < batch.size(); ++ci) {
            auto& call = batch[ci];
            auto [ws_size, executor] =
                PlanFlexAttention(call, kNumHeads, kNumKvHeads, scale,
                                  step_tags[ci], call.sparse_mode,
                                  call.pre_tokens, call.next_tokens);
            CHECK_EQ(aclnnFusedInferAttentionScoreV3(workspace, ws_size, executor,
                                                    stream_),
                     0);
          }
          (void)combine_rt(batch);
          (void)combine_tgt(batch);
        }
      },
      kDeviceId, 3, 10);

  // Precision check (inside this benchmark test only).
  {
    auto max_abs = [](const torch::Tensor& a, const torch::Tensor& b) {
      return (a.to(torch::kCPU).to(torch::kFloat32) -
              b.to(torch::kCPU).to(torch::kFloat32))
          .abs()
          .max()
          .item<float>();
    };

    const auto& b0 = all_batches.at(0);

    auto rt_out_2a =
        b0.at(1).out.slice(2, parts.context, parts.context + parts.real_time);
    auto rt_lse_2a =
        b0.at(1).lse.slice(2, parts.context, parts.context + parts.real_time);
    auto rt_out_2b = b0.at(2).out.slice(2, 0, parts.real_time);
    auto rt_lse_2b = b0.at(2).lse.slice(2, 0, parts.real_time);
    auto rt_final = combine_two_way_attention_update(
        rt_out_2a, rt_lse_2a, rt_out_2b, rt_lse_2b);

    auto tgt_out_2a =
        b0.at(1).out.slice(2, parts.context + parts.real_time,
                           parts.context + parts.real_time + parts.target);
    auto tgt_lse_2a =
        b0.at(1).lse.slice(2, parts.context + parts.real_time,
                           parts.context + parts.real_time + parts.target);
    auto tgt_out_2b =
        b0.at(2).out.slice(2, parts.real_time, parts.real_time + parts.target);
    auto tgt_lse_2b =
        b0.at(2).lse.slice(2, parts.real_time, parts.real_time + parts.target);
    auto tgt_final = combine_three_way_attention_update(tgt_out_2a,
                                                        tgt_lse_2a,
                                                        tgt_out_2b,
                                                        tgt_lse_2b,
                                                        b0.at(3).out,
                                                        b0.at(3).lse);

    const int64_t s0 = seq_lens.at(0);
    auto q_ref =
        q_all.slice(0, s0, s0 + seq_len).to(torch::kCPU).to(torch::kFloat32);
    auto k_ref =
        k_all.slice(0, s0, s0 + seq_len).to(torch::kCPU).to(torch::kFloat32);
    auto v_ref =
        v_all.slice(0, s0, s0 + seq_len).to(torch::kCPU).to(torch::kFloat32);
    auto ref = ReferenceGenRecGold(q_ref, k_ref, v_ref, parts, scale);

    float d_rt = max_abs(rt_final, ref.real_time);
    float d_tgt = max_abs(tgt_final, ref.target);
    LOG(INFO) << "[GenRecV2][Precision] max_abs real_time=" << d_rt
              << ", target=" << d_tgt;
    EXPECT_LT(d_rt, 1.5e-1f);
    EXPECT_LT(d_tgt, 1.5e-1f);
  }

  if (max_ws > 0) {
    CHECK_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    aclrtFree(workspace);
  }
  for (auto& batch : all_batches)
    for (auto& call : batch) call.DestroyAcl();

  LOG(INFO) << "[GenRecV2] " << kBatchSize << " batches x " << parts.total()
            << " tok/seq (h=" << parts.history << " c=" << parts.context
            << " r=" << parts.real_time << " t=" << parts.target << ") | " << ms
            << " ms/iter, " << (ms / static_cast<double>(kBatchSize))
            << " ms/batch, "
            << (ms / static_cast<double>(kBatchSize) / 4.0) << " ms/attn_call";
}

}  // namespace xllm::kernel::npu::test
