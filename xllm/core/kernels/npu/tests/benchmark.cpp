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

#include <chrono>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsubobject-linkage"
#endif
#include "final_BSND_test.cpp"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace xllm::kernel::npu::test {
namespace {

struct StageTimeline {
  const char* name = nullptr;
  double get_ws_ms = 0.0;
  double device_ms = 0.0;
  std::chrono::steady_clock::time_point host_submit{};
  aclrtEvent ev_start = nullptr;
  aclrtEvent ev_end = nullptr;
};

double measure_wall_clock_ms(const std::function<void()>& fn,
                             aclrtStream stream,
                             int warmup_iters = 5,
                             int measure_iters = 20) {
  CHECK_GT(measure_iters, 0);
  CHECK_GE(warmup_iters, 0);

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
  return std::chrono::duration<double, std::milli>(t1 - t0).count() /
         static_cast<double>(measure_iters);
}

void create_stage_events_if_needed(StageTimeline* stage) {
  if (stage == nullptr) {
    return;
  }
  const uint32_t flags = ACL_EVENT_TIME_LINE;
  CHECK_EQ(aclrtCreateEventWithFlag(&stage->ev_start, flags), ACL_SUCCESS);
  CHECK_EQ(aclrtCreateEventWithFlag(&stage->ev_end, flags), ACL_SUCCESS);
}

void destroy_stage_events_if_needed(StageTimeline* stage) {
  if (stage == nullptr) {
    return;
  }
  if (stage->ev_start != nullptr) {
    CHECK_EQ(aclrtDestroyEvent(stage->ev_start), ACL_SUCCESS);
    stage->ev_start = nullptr;
  }
  if (stage->ev_end != nullptr) {
    CHECK_EQ(aclrtDestroyEvent(stage->ev_end), ACL_SUCCESS);
    stage->ev_end = nullptr;
  }
}

void record_stage_begin_if_needed(StageTimeline* stage, aclrtStream stream) {
  if (stage == nullptr) {
    return;
  }
  CHECK_EQ(aclrtRecordEvent(stage->ev_start, stream), ACL_SUCCESS);
  stage->host_submit = std::chrono::steady_clock::now();
}

void record_stage_end_if_needed(StageTimeline* stage, aclrtStream stream) {
  if (stage == nullptr) {
    return;
  }
  CHECK_EQ(aclrtRecordEvent(stage->ev_end, stream), ACL_SUCCESS);
}

void launch_attention_without_memset(const AclPlan& plan,
                                     const AttentionMetadata& attn_metadata,
                                     aclrtStream stream) {
  void* ws_ptr = nullptr;
  if (plan.workspace_size > 0) {
    CHECK_NE(attn_metadata.shared_workspace, nullptr)
        << "shared_workspace is required when workspace_size > 0";
    CHECK_GE(attn_metadata.shared_workspace_size, plan.workspace_size)
        << "shared_workspace bytes(" << attn_metadata.shared_workspace_size
        << ") is smaller than required(" << plan.workspace_size << ")";
    ws_ptr = attn_metadata.shared_workspace;
  }
  auto ret = aclnnFusedInferAttentionScoreV3(
      ws_ptr, plan.workspace_size, plan.executor, stream);
  CHECK_EQ(ret, ACL_SUCCESS) << "aclnnFusedInferAttentionScoreV3 failed: " << ret;
}

void launch_attention_update_without_memset(const AttentionUpdatePlan& plan,
                                            const AttentionMetadata& attn_metadata,
                                            aclrtStream stream) {
  void* ws_ptr = nullptr;
  if (plan.workspace_size > 0) {
    CHECK_NE(attn_metadata.shared_workspace, nullptr)
        << "shared_workspace is required when workspace_size > 0";
    CHECK_GE(attn_metadata.shared_workspace_size, plan.workspace_size)
        << "shared_workspace bytes(" << attn_metadata.shared_workspace_size
        << ") is smaller than required(" << plan.workspace_size << ")";
    ws_ptr = attn_metadata.shared_workspace;
  }
  auto ret = aclnnAttentionUpdate(ws_ptr, plan.workspace_size, plan.executor, stream);
  CHECK_EQ(ret, ACL_SUCCESS) << "aclnnAttentionUpdate failed: " << ret;
}

torch::Tensor run_fa_all_segments_bench_no_memset(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const AttentionMetadata& attn_metadata,
    const SampleSegmentMetadata& sample_metadata,
    std::vector<StageTimeline>* timeline) {
  CHECK_EQ(query.size(0), 1);
  CHECK_EQ(key.size(0), 1);
  CHECK_EQ(value.size(0), 1);

  const int64_t h = sample_metadata.history;
  const int64_t c = sample_metadata.context;
  const int64_t r = sample_metadata.real_time;
  const int64_t t = sample_metadata.target;
  const int64_t num_heads = query.size(2);
  const int64_t head_dim = query.size(3);
  auto stream = c10_npu::getCurrentNPUStream(query.device().index()).stream();

  if (timeline != nullptr) {
    CHECK_EQ(timeline->size(), 6UL);
    (*timeline)[0].name = "hist_fa";
    (*timeline)[1].name = "ctx_rt_tgt_full";
    (*timeline)[2].name = "rt_tgt_on_rt";
    (*timeline)[3].name = "tgt_diag";
    (*timeline)[4].name = "update_rt_2way";
    (*timeline)[5].name = "update_tgt_3way";
    for (auto& stage : *timeline) {
      create_stage_events_if_needed(&stage);
    }
  }

  auto out_opts =
      torch::TensorOptions().dtype(query.scalar_type()).device(query.device());
  auto lse_opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(query.device());
  constexpr float kSentinel = -31415.0f;
  auto make_out = [&](int64_t seq_len) {
    return torch::full({1, seq_len, num_heads, head_dim}, kSentinel, out_opts);
  };
  auto make_lse = [&](int64_t seq_len) {
    return torch::full({1, num_heads, seq_len, 1}, kSentinel, lse_opts);
  };

  torch::Tensor output = torch::empty_like(query);

  auto hist_out = make_out(h);
  SegmentAttentionMetadata hist_meta;
  hist_meta.attn_mask = attn_metadata.compressed_causal_mask;
  hist_meta.sparse_param.sparse_mode = 2;
  hist_meta.sparse_param.pre_tokens = kDefaultWindow;
  hist_meta.sparse_param.next_tokens = kDefaultWindow;
  auto hist_query = query.slice(1, 0, h);
  auto hist_key = key.slice(1, 0, h);
  auto hist_value = value.slice(1, 0, h);
  auto t0 = std::chrono::steady_clock::now();
  AclPlan hist_plan =
      plan_segment_attention(hist_query, hist_key, hist_value, hist_meta, hist_out, std::nullopt);
  if (timeline != nullptr) {
    (*timeline)[0].get_ws_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
            .count();
    record_stage_begin_if_needed(&(*timeline)[0], stream);
  }
  launch_attention_without_memset(hist_plan, attn_metadata, stream);
  if (timeline != nullptr) {
    record_stage_end_if_needed(&(*timeline)[0], stream);
  }
  destroy_planned_attention(hist_plan);
  output.slice(1, 0, h).copy_(hist_out);

  auto crt_out = make_out(c + r + t);
  auto crt_lse = make_lse(c + r + t);
  auto crt_query = query.slice(1, h, h + c + r + t).contiguous();
  auto crt_key = key.slice(1, 0, h + c).contiguous();
  auto crt_value = value.slice(1, 0, h + c).contiguous();
  SegmentAttentionMetadata crt_meta;
  crt_meta.sparse_param.sparse_mode = 0;
  crt_meta.sparse_param.pre_tokens = kDefaultWindow;
  crt_meta.sparse_param.next_tokens = kDefaultWindow;
  t0 = std::chrono::steady_clock::now();
  AclPlan crt_plan = plan_segment_attention(
      crt_query, crt_key, crt_value, crt_meta, crt_out, crt_lse);
  if (timeline != nullptr) {
    (*timeline)[1].get_ws_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
            .count();
    record_stage_begin_if_needed(&(*timeline)[1], stream);
  }
  launch_attention_without_memset(crt_plan, attn_metadata, stream);
  if (timeline != nullptr) {
    record_stage_end_if_needed(&(*timeline)[1], stream);
  }
  destroy_planned_attention(crt_plan);
  output.slice(1, h, h + c).copy_(crt_out.slice(1, 0, c));

  auto rt_out = make_out(r + t);
  auto rt_lse = make_lse(r + t);
  auto rt_query = query.slice(1, h + c, h + c + r + t).contiguous();
  auto rt_key = key.slice(1, h + c, h + c + r).contiguous();
  auto rt_value = value.slice(1, h + c, h + c + r).contiguous();
  SegmentAttentionMetadata rt_meta;
  rt_meta.attn_mask = attn_metadata.compressed_causal_mask;
  rt_meta.sparse_param.sparse_mode = 2;
  rt_meta.sparse_param.pre_tokens = kDefaultWindow;
  rt_meta.sparse_param.next_tokens = kDefaultWindow;
  t0 = std::chrono::steady_clock::now();
  AclPlan rt_plan =
      plan_segment_attention(rt_query, rt_key, rt_value, rt_meta, rt_out, rt_lse);
  if (timeline != nullptr) {
    (*timeline)[2].get_ws_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
            .count();
    record_stage_begin_if_needed(&(*timeline)[2], stream);
  }
  launch_attention_without_memset(rt_plan, attn_metadata, stream);
  if (timeline != nullptr) {
    record_stage_end_if_needed(&(*timeline)[2], stream);
  }
  destroy_planned_attention(rt_plan);

  auto target_out = make_out(t);
  auto target_lse = make_lse(t);
  auto target_query = query.slice(1, h + c + r, h + c + r + t).contiguous();
  auto target_key = key.slice(1, h + c + r, h + c + r + t).contiguous();
  auto target_value = value.slice(1, h + c + r, h + c + r + t).contiguous();
  SegmentAttentionMetadata target_meta;
  target_meta.attn_mask = attn_metadata.diagonal_mask;
  target_meta.sparse_param.sparse_mode = 0;
  target_meta.sparse_param.pre_tokens = 0;
  target_meta.sparse_param.next_tokens = 0;
  t0 = std::chrono::steady_clock::now();
  AclPlan target_plan = plan_segment_attention(
      target_query, target_key, target_value, target_meta, target_out, target_lse);
  if (timeline != nullptr) {
    (*timeline)[3].get_ws_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
            .count();
    record_stage_begin_if_needed(&(*timeline)[3], stream);
  }
  launch_attention_without_memset(target_plan, attn_metadata, stream);
  if (timeline != nullptr) {
    record_stage_end_if_needed(&(*timeline)[3], stream);
  }
  destroy_planned_attention(target_plan);

  t0 = std::chrono::steady_clock::now();
  auto rt_update_plan = plan_attention_update_like_benchmark(
      {crt_out.slice(1, c, c + r).contiguous(), rt_out.slice(1, 0, r).contiguous()},
      {crt_lse.slice(2, c, c + r).contiguous(), rt_lse.slice(2, 0, r).contiguous()});
  if (timeline != nullptr) {
    (*timeline)[4].get_ws_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
            .count();
    record_stage_begin_if_needed(&(*timeline)[4], stream);
  }
  launch_attention_update_without_memset(rt_update_plan, attn_metadata, stream);
  if (timeline != nullptr) {
    record_stage_end_if_needed(&(*timeline)[4], stream);
  }
  auto rt_merged = rt_update_plan.out_flat.view({rt_update_plan.b,
                                                 rt_update_plan.s,
                                                 rt_update_plan.n,
                                                 rt_update_plan.d})
                       .to(rt_update_plan.out_scalar_type);
  destroy_planned_attention_update(rt_update_plan);
  output.slice(1, h + c, h + c + r).copy_(rt_merged);

  t0 = std::chrono::steady_clock::now();
  auto target_update_plan = plan_attention_update_like_benchmark(
      {crt_out.slice(1, c + r, c + r + t).contiguous(),
       rt_out.slice(1, r, r + t).contiguous(),
       target_out.contiguous()},
      {crt_lse.slice(2, c + r, c + r + t).contiguous(),
       rt_lse.slice(2, r, r + t).contiguous(),
       target_lse.contiguous()});
  if (timeline != nullptr) {
    (*timeline)[5].get_ws_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
            .count();
    record_stage_begin_if_needed(&(*timeline)[5], stream);
  }
  launch_attention_update_without_memset(target_update_plan, attn_metadata, stream);
  if (timeline != nullptr) {
    record_stage_end_if_needed(&(*timeline)[5], stream);
  }
  auto target_merged = target_update_plan.out_flat.view({target_update_plan.b,
                                                         target_update_plan.s,
                                                         target_update_plan.n,
                                                         target_update_plan.d})
                           .to(target_update_plan.out_scalar_type);
  destroy_planned_attention_update(target_update_plan);
  output.slice(1, h + c + r, h + c + r + t).copy_(target_merged);

  return output;
}

TEST_F(FinalBSNDTest, BenchFinalBSNDTimelineAlignedWithGenRec) {
  torch::manual_seed(20260409);

  constexpr int64_t kBatchSize = 1;
  constexpr int64_t kNumHeads = 8;
  constexpr int64_t kNumKvHeads = 8;
  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kHiddenSize = kNumHeads * kHeadDim;
  constexpr int64_t kHistory = 1300;
  constexpr int64_t kContext = 8;
  constexpr int64_t kRealtime = 400;
  constexpr int64_t kTarget = 800;
  constexpr int64_t kMatchedPrefix = 0;
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kCacheBlockCount = 40;
  constexpr uint64_t kFixedSharedWorkspaceBytes = 256ULL * 1024ULL * 1024ULL;
  constexpr int kTimelineWarmupIters = 2;
  constexpr int kThroughputWarmupIters = 3;
  constexpr int kThroughputMeasureIters = 10;

  const int64_t total = kHistory + kContext + kRealtime + kTarget;
  const int64_t seq_block_count = (total + kBlockSize - 1) / kBlockSize;

  auto hidden_states = torch::randn({kBatchSize * total, kHiddenSize}, fp16_opts_);
  auto positions = torch::arange(
      total,
      torch::TensorOptions().dtype(torch::kInt64).device(hidden_states.device()));

  AttentionMetadata attn_metadata;
  auto len_opts = torch::TensorOptions().dtype(torch::kInt64);
  attn_metadata.q_seq_lens = torch::tensor({total}, len_opts);
  attn_metadata.kv_seq_lens = torch::tensor({total}, len_opts);
  attn_metadata.history_lens = torch::tensor({kHistory}, len_opts);
  attn_metadata.context_lens = torch::tensor({kContext}, len_opts);
  attn_metadata.real_time_lens = torch::tensor({kRealtime}, len_opts);
  attn_metadata.target_lens = torch::tensor({kTarget}, len_opts);
  attn_metadata.matched_prefix_lens = torch::tensor({kMatchedPrefix}, len_opts);
  attn_metadata.block_size = kBlockSize;
  attn_metadata.cache_block_count = kCacheBlockCount;
  attn_metadata.block_tables_host = {make_block_table(seq_block_count)};
  attn_metadata.compressed_causal_mask =
      create_compressed_causal_mask_2048(hidden_states.device());
  attn_metadata.diagonal_mask = create_diagonal_mask(kTarget, hidden_states.device());
  ASSERT_EQ(aclrtMalloc(&attn_metadata.shared_workspace,
                        kFixedSharedWorkspaceBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_SUCCESS);
  attn_metadata.shared_workspace_size = kFixedSharedWorkspaceBytes;

  KVCache kv_cache{
      torch::zeros({kCacheBlockCount, kBlockSize, kNumKvHeads, kHeadDim}, fp16_opts_),
      torch::zeros({kCacheBlockCount, kBlockSize, kNumKvHeads, kHeadDim}, fp16_opts_)};

  CustomMaskAttentionTestImpl custom_attn(
      kNumHeads, kNumKvHeads, kHeadDim, kHiddenSize);

  for (int i = 0; i < kTimelineWarmupIters; ++i) {
    auto warmup_hidden_states = torch::randn({kBatchSize * total, kHiddenSize}, fp16_opts_);
    auto warmup_positions = positions.clone();
    (void)custom_attn.forward(
        warmup_positions, warmup_hidden_states, attn_metadata, kv_cache);
  }
  (void)custom_attn.forward(positions, hidden_states, attn_metadata, kv_cache);

  auto query = packed_to_bsnd(custom_attn.last_query(), kNumHeads, kHeadDim);
  auto key = packed_to_bsnd(custom_attn.last_key(), kNumKvHeads, kHeadDim);
  auto value = packed_to_bsnd(custom_attn.last_value(), kNumKvHeads, kHeadDim);

  SampleSegmentMetadata sample_metadata;
  sample_metadata.history = kHistory;
  sample_metadata.context = kContext;
  sample_metadata.real_time = kRealtime;
  sample_metadata.target = kTarget;
  sample_metadata.matched_prefix = kMatchedPrefix;
  sample_metadata.block_table_host = nullptr;

  for (int i = 0; i < kTimelineWarmupIters; ++i) {
    (void)run_fa_all_segments_bench_no_memset(
        query, key, value, attn_metadata, sample_metadata, nullptr);
  }
  CHECK_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);

  std::vector<StageTimeline> timeline(6);
  auto output = run_fa_all_segments_bench_no_memset(
      query, key, value, attn_metadata, sample_metadata, &timeline);
  CHECK_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);

  for (auto& stage : timeline) {
    float ms = 0.0f;
    CHECK_EQ(aclrtEventElapsedTime(&ms, stage.ev_start, stage.ev_end), ACL_SUCCESS);
    stage.device_ms = static_cast<double>(ms);
  }

  float total_ms_f = 0.0f;
  CHECK_EQ(aclrtEventElapsedTime(&total_ms_f,
                                 timeline.front().ev_start,
                                 timeline.back().ev_end),
           ACL_SUCCESS);
  const double total_ms = static_cast<double>(total_ms_f);

  std::vector<double> device_gap_ms(timeline.size() - 1, 0.0);
  std::vector<double> host_interval_ms(timeline.size() - 1, 0.0);
  for (size_t i = 0; i + 1 < timeline.size(); ++i) {
    float gap_ms_f = 0.0f;
    CHECK_EQ(aclrtEventElapsedTime(
                 &gap_ms_f, timeline[i].ev_end, timeline[i + 1].ev_start),
             ACL_SUCCESS);
    device_gap_ms[i] = static_cast<double>(gap_ms_f);
    host_interval_ms[i] =
        std::chrono::duration<double, std::milli>(timeline[i + 1].host_submit -
                                                  timeline[i].host_submit)
            .count();
  }

  const double kIdleEpsMs = 0.05;
  bool host_likely_overlap = true;
  for (size_t i = 0; i + 1 < timeline.size(); ++i) {
    if (host_interval_ms[i] > timeline[i].device_ms) {
      host_likely_overlap = false;
      break;
    }
  }
  bool device_has_idle = false;
  for (double gap : device_gap_ms) {
    if (gap > kIdleEpsMs) {
      device_has_idle = true;
      break;
    }
  }
  const bool overlap = host_likely_overlap && !device_has_idle;

  std::fprintf(stderr,
               "[FinalBSND][TimelineAttn] "
               "attn(hist)=%.6fms, "
               "attn(ctx_rt_tgt_full)=%.6fms, "
               "attn(rt_tgt_on_rt)=%.6fms, "
               "attn(tgt_diag)=%.6fms, "
               "update(rt_2way)=%.6fms, "
               "update(tgt_3way)=%.6fms, "
               "total_attn=%.6fms, "
               "device_gap(ms)=%.6f,%.6f,%.6f,%.6f,%.6f, "
               "host_interval(ms)=%.6f,%.6f,%.6f,%.6f,%.6f, "
               "overlap=%d\n",
               timeline[0].device_ms,
               timeline[1].device_ms,
               timeline[2].device_ms,
               timeline[3].device_ms,
               timeline[4].device_ms,
               timeline[5].device_ms,
               total_ms,
               device_gap_ms[0],
               device_gap_ms[1],
               device_gap_ms[2],
               device_gap_ms[3],
               device_gap_ms[4],
               host_interval_ms[0],
               host_interval_ms[1],
               host_interval_ms[2],
               host_interval_ms[3],
               host_interval_ms[4],
               overlap ? 1 : 0);

  std::fprintf(stderr,
               "[FinalBSND][TimelineGetWs] "
               "hist=%.6fms, ctx_rt_tgt_full=%.6fms, rt_tgt_on_rt=%.6fms, "
               "tgt_diag=%.6fms, update_rt_2way=%.6fms, update_tgt_3way=%.6fms\n",
               timeline[0].get_ws_ms,
               timeline[1].get_ws_ms,
               timeline[2].get_ws_ms,
               timeline[3].get_ws_ms,
               timeline[4].get_ws_ms,
               timeline[5].get_ws_ms);
  std::fflush(stderr);

  expect_output_matches_reference("BenchFinalBSNDTimelineAlignedWithGenRec",
                                  output,
                                  query,
                                  key,
                                  value,
                                  kHistory,
                                  kContext,
                                  kRealtime,
                                  kTarget);

  double avg_ms = measure_wall_clock_ms(
      [&]() {
        (void)run_fa_all_segments_bench_no_memset(
            query, key, value, attn_metadata, sample_metadata, nullptr);
      },
      stream_,
      kThroughputWarmupIters,
      kThroughputMeasureIters);
  std::fprintf(stderr,
               "[FinalBSND][Bench] %lld batches x %lld tok/seq "
               "(h=%lld c=%lld r=%lld t=%lld) | %.6f ms/iter, %.6f ms/batch, %.6f ms/op\n",
               static_cast<long long>(kBatchSize),
               static_cast<long long>(total),
               static_cast<long long>(kHistory),
               static_cast<long long>(kContext),
               static_cast<long long>(kRealtime),
               static_cast<long long>(kTarget),
               avg_ms,
               avg_ms / static_cast<double>(kBatchSize),
               avg_ms / 6.0);
  std::fflush(stderr);

  for (auto& stage : timeline) {
    destroy_stage_events_if_needed(&stage);
  }
  ASSERT_EQ(aclrtFree(attn_metadata.shared_workspace), ACL_SUCCESS);
}

}  // namespace
}  // namespace xllm::kernel::npu::test
