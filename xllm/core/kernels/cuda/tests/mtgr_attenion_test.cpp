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

#include "mtgr_attenion_test.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <torch/cuda.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/common/global_flags.h"
#include "core/platform/device.h"
#include "cuda_ops_api.h"
#include "mtgr_flashinfer.h"
#include "utils.h"

namespace xllm::kernel::cuda::test {

// Generation 1: historical four-segment CUDA paths.

void build_mtgr_packed_mask(torch::Tensor packed_mask,
                            int64_t q_len,
                            int64_t kv_len,
                            int64_t history_len,
                            int64_t context_len,
                            int64_t realtime_len,
                            int64_t mask_kind);

void merge_target_diag_attention_cuda(const torch::Tensor& hcr_out_snd,
                                      const torch::Tensor& hcr_lse_sh1,
                                      const torch::Tensor& target_query_snd,
                                      const torch::Tensor& target_key_snd,
                                      const torch::Tensor& target_value_snd,
                                      double sm_scale,
                                      torch::Tensor merged_out_snd);

void merge_target_diag_attention_batched_cuda(
    const torch::Tensor& hcr_out_snd,
    const torch::Tensor& hcr_lse_sh1,
    const torch::Tensor& packed_query_snd,
    const torch::Tensor& packed_key_snd,
    const torch::Tensor& packed_value_snd,
    const torch::Tensor& q_seq_starts_i32,
    const torch::Tensor& history_lens_i32,
    const torch::Tensor& context_lens_i32,
    const torch::Tensor& realtime_lens_i32,
    const torch::Tensor& target_lens_i32,
    const torch::Tensor& target_seq_starts_i32,
    int64_t batch_size,
    int64_t max_target_len,
    double sm_scale,
    torch::Tensor merged_out_snd);

void merge_target_diag_attention_partial_batched_cuda(
    const torch::Tensor& hcr_out_snd,
    const torch::Tensor& hcr_lse_sh1,
    const torch::Tensor& packed_query_snd,
    const torch::Tensor& packed_key_snd,
    const torch::Tensor& packed_value_snd,
    const torch::Tensor& q_seq_starts_i32,
    const torch::Tensor& realtime_unmatched_lens_i32,
    const torch::Tensor& target_lens_i32,
    const torch::Tensor& target_seq_starts_i32,
    int64_t batch_size,
    int64_t max_target_len,
    double sm_scale,
    torch::Tensor merged_out_snd);

void mtgr_fused_no_match_attention_cuda(const torch::Tensor& query_snd,
                                        const torch::Tensor& key_snd,
                                        const torch::Tensor& value_snd,
                                        int64_t history_len,
                                        int64_t context_len,
                                        int64_t realtime_len,
                                        int64_t target_len,
                                        double sm_scale,
                                        torch::Tensor output_snd,
                                        torch::Tensor target_hcr_lse_sh1);

void mtgr_fused_no_match_attention_batched_cuda(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    const torch::Tensor& q_seq_starts_i32,
    const torch::Tensor& history_lens_i32,
    const torch::Tensor& context_lens_i32,
    const torch::Tensor& realtime_lens_i32,
    const torch::Tensor& target_lens_i32,
    const torch::Tensor& target_seq_starts_i32,
    const torch::Tensor& row_to_batch_i32,
    double sm_scale,
    torch::Tensor output_snd,
    torch::Tensor target_hcr_lse_sh1);

void mtgr_fused_segmented_no_match_attention_batched_cuda(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    const torch::Tensor& segment_offsets_i32,
    const torch::Tensor& segment_rules_i32,
    const torch::Tensor& row_to_batch_i32,
    const torch::Tensor& target_seq_starts_i32,
    const torch::Tensor& target_lens_i32,
    double sm_scale,
    torch::Tensor output_snd,
    torch::Tensor target_hcr_lse_sh1);

void mtgr_ragged_segment_attention_batched_cuda(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    const torch::Tensor& segment_offsets_i32,
    const torch::Tensor& segment_rules_i32,
    const torch::Tensor& row_to_batch_i32,
    int64_t max_request_len,
    double sm_scale,
    torch::Tensor output_snd);

void mtgr_fused_no_match_attention_batched_batch4_cuda(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    const torch::Tensor& q_seq_starts_i32,
    const torch::Tensor& history_lens_i32,
    const torch::Tensor& context_lens_i32,
    const torch::Tensor& realtime_lens_i32,
    const torch::Tensor& target_lens_i32,
    const torch::Tensor& target_seq_starts_i32,
    int64_t batch_size,
    int64_t max_q_len,
    int64_t max_history_len,
    int64_t max_target_len,
    double sm_scale,
    torch::Tensor output_snd,
    torch::Tensor target_hcr_lse_sh1);

void mtgr_fused_partial_rt_attention_cuda(const torch::Tensor& query_snd,
                                          const torch::Tensor& rt_key_snd,
                                          const torch::Tensor& rt_value_snd,
                                          const torch::Tensor& key_cache,
                                          const torch::Tensor& value_cache,
                                          const torch::Tensor& block_table_row,
                                          int64_t block_size,
                                          int64_t matched_prefix_len,
                                          int64_t realtime_unmatched_len,
                                          int64_t target_len,
                                          double sm_scale,
                                          torch::Tensor output_snd,
                                          torch::Tensor target_hcr_lse_sh1);

void mtgr_fused_partial_rt_attention_batched_batch4_cuda(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& q_seq_starts_i32,
    const torch::Tensor& matched_prefix_lens_i32,
    const torch::Tensor& realtime_unmatched_lens_i32,
    const torch::Tensor& target_lens_i32,
    const torch::Tensor& target_seq_starts_i32,
    const torch::Tensor& block_table_i32,
    int64_t batch_size,
    int64_t max_q_len,
    int64_t max_matched_prefix_len,
    int64_t max_realtime_unmatched_len,
    int64_t max_target_len,
    int64_t block_size,
    double sm_scale,
    torch::Tensor output_snd,
    torch::Tensor target_hcr_lse_sh1);

// Generation 2: stable ragged-segment CUDA path.

void full_attention_cuda(const torch::Tensor& query_snd,
                         const torch::Tensor& key_snd,
                         const torch::Tensor& value_snd,
                         double sm_scale,
                         torch::Tensor output_snd);

// Generation 3: Hopper unified research path.
// This stays opt-in and separate from the stable wrappers.
void mtgr_ragged_segment_attention_hopper_unified_research_cuda(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    const torch::Tensor& segment_offsets_i32,
    const torch::Tensor& segment_rules_i32,
    const torch::Tensor& q_seq_starts_i32,
    const torch::Tensor& matched_prefix_lens_i32,
    int64_t match_mode,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& block_table_i32,
    int64_t block_size,
    int64_t max_request_len,
    double sm_scale,
    torch::Tensor output_snd);

namespace {

struct StageTimeline {
  const char* name = nullptr;
  double get_ws_ms = 0.0;
  double device_ms = 0.0;
  double host_ms = 0.0;
  bool has_getws = false;
  std::chrono::steady_clock::time_point getws_submit{};
  std::chrono::steady_clock::time_point host_submit{};
  std::chrono::steady_clock::time_point host_end{};
  cudaEvent_t ev_start = nullptr;
  cudaEvent_t ev_end = nullptr;
};

struct TimelineSummary {
  double total_ms = 0.0;
  std::vector<double> stage_device_ms;
  std::vector<double> stage_host_ms;
  std::vector<double> stage_getws_ms;
};

std::vector<StageTimeline> build_no_match_multi_timeline() {
  return {{"rt_tgt_on_hcr_trapezoid"},
          {"tgt_diag_update_fused"},
          {"hist_fa"},
          {"ctx_on_hc"}};
}

std::vector<StageTimeline> build_partial_rt_multi_timeline() {
  return {{"rt_tgt_on_prefix_rt_trapezoid"}, {"tgt_diag_update_fused"}};
}

std::vector<StageTimeline> build_fused_no_match_timeline() {
  return {{"mtgr_fused_no_match_attention"}, {"tgt_diag_update_fused"}};
}

std::vector<StageTimeline> build_ragged_segment_timeline() {
  return {{"mtgr_ragged_segment_attention"}};
}

std::vector<StageTimeline> build_fused_partial_rt_timeline() {
  return {{"mtgr_fused_partial_rt_attention"}, {"tgt_diag_update_fused"}};
}

void create_stage_events_if_needed(StageTimeline* stage) {
  if (stage == nullptr) {
    return;
  }
  CHECK_EQ(cudaEventCreate(&stage->ev_start), cudaSuccess);
  CHECK_EQ(cudaEventCreate(&stage->ev_end), cudaSuccess);
}

void destroy_stage_events_if_needed(StageTimeline* stage) {
  if (stage == nullptr) {
    return;
  }
  if (stage->ev_start != nullptr) {
    CHECK_EQ(cudaEventDestroy(stage->ev_start), cudaSuccess);
    stage->ev_start = nullptr;
  }
  if (stage->ev_end != nullptr) {
    CHECK_EQ(cudaEventDestroy(stage->ev_end), cudaSuccess);
    stage->ev_end = nullptr;
  }
}

void prepare_timeline_events(std::vector<StageTimeline>* timeline) {
  if (timeline == nullptr) {
    return;
  }
  for (auto& stage : *timeline) {
    stage.get_ws_ms = 0.0;
    stage.device_ms = 0.0;
    stage.host_ms = 0.0;
    stage.has_getws = false;
    stage.getws_submit = std::chrono::steady_clock::time_point{};
    stage.host_submit = std::chrono::steady_clock::time_point{};
    stage.host_end = std::chrono::steady_clock::time_point{};
    create_stage_events_if_needed(&stage);
  }
}

void release_timeline_events(std::vector<StageTimeline>* timeline) {
  if (timeline == nullptr) {
    return;
  }
  for (auto& stage : *timeline) {
    destroy_stage_events_if_needed(&stage);
  }
}

void record_stage_begin_if_needed(StageTimeline* stage,
                                  const torch::Device& device) {
  if (stage == nullptr) {
    return;
  }
  const auto stream = c10::cuda::getCurrentCUDAStream(device.index()).stream();
  CHECK_EQ(cudaEventRecord(stage->ev_start, stream), cudaSuccess);
  stage->host_submit = std::chrono::steady_clock::now();
  stage->host_end = std::chrono::steady_clock::time_point{};
}

void record_stage_end_if_needed(StageTimeline* stage,
                                const torch::Device& device) {
  if (stage == nullptr) {
    return;
  }
  const auto stream = c10::cuda::getCurrentCUDAStream(device.index()).stream();
  CHECK_EQ(cudaEventRecord(stage->ev_end, stream), cudaSuccess);
  stage->host_end = std::chrono::steady_clock::now();
  stage->host_ms = std::chrono::duration<double, std::milli>(stage->host_end -
                                                             stage->host_submit)
                       .count();
}

TimelineSummary collect_timeline_summary(std::vector<StageTimeline>* timeline) {
  CHECK(timeline != nullptr);
  CHECK(!timeline->empty());
  for (const auto& stage : *timeline) {
    CHECK(stage.ev_end != nullptr);
    CHECK_EQ(cudaEventSynchronize(stage.ev_end), cudaSuccess);
  }

  TimelineSummary summary;
  summary.stage_device_ms.resize(timeline->size(), 0.0);
  summary.stage_host_ms.resize(timeline->size(), 0.0);
  summary.stage_getws_ms.resize(timeline->size(), 0.0);
  for (size_t i = 0; i < timeline->size(); ++i) {
    float ms = 0.0f;
    CHECK_EQ(cudaEventElapsedTime(
                 &ms, (*timeline)[i].ev_start, (*timeline)[i].ev_end),
             cudaSuccess);
    (*timeline)[i].device_ms = static_cast<double>(ms);
    summary.stage_device_ms[i] = (*timeline)[i].device_ms;
    summary.stage_host_ms[i] = (*timeline)[i].host_ms;
    summary.stage_getws_ms[i] = (*timeline)[i].get_ws_ms;
  }

  float total_ms = 0.0f;
  CHECK_EQ(cudaEventElapsedTime(
               &total_ms, timeline->front().ev_start, timeline->back().ev_end),
           cudaSuccess);
  summary.total_ms = static_cast<double>(total_ms);
  return summary;
}

void fill_stage_metrics(const std::vector<StageTimeline>& timeline,
                        const TimelineSummary& summary,
                        MTGRAttentionTestMetrics* metrics) {
  CHECK(metrics != nullptr);
  metrics->stages.clear();
  metrics->stages.reserve(timeline.size());
  for (size_t i = 0; i < timeline.size(); ++i) {
    metrics->stages.push_back(MTGRStageMetric{
        .name = timeline[i].name != nullptr ? timeline[i].name : "stage",
        .workspace_ms = summary.stage_getws_ms[i],
        .exec_ms = summary.stage_device_ms[i],
        .host_submit_ms = summary.stage_host_ms[i],
    });
  }
}

struct FlashinferWorkspaceBuffers {
  torch::Tensor float_workspace;
  torch::Tensor int_workspace;
  torch::Tensor page_locked_int_workspace;
};

void ensure_workspace_buffers(FlashinferWorkspaceBuffers* ws,
                              const torch::Device& device) {
  CHECK(ws != nullptr);
  const bool need_init =
      !ws->float_workspace.defined() || ws->float_workspace.device() != device;
  if (!need_init) {
    return;
  }
  ws->float_workspace =
      torch::empty({FLAGS_flashinfer_workspace_buffer_size},
                   torch::TensorOptions().dtype(torch::kUInt8).device(device));
  ws->int_workspace =
      torch::empty({8 * 1024 * 1024},
                   torch::TensorOptions().dtype(torch::kUInt8).device(device));
  ws->page_locked_int_workspace = torch::empty({ws->int_workspace.size(0)},
                                               torch::TensorOptions()
                                                   .dtype(torch::kUInt8)
                                                   .device(torch::kCPU)
                                                   .pinned_memory(true));
}

FlashinferWorkspaceBuffers& get_workspace_buffers(const torch::Device& device) {
  static thread_local FlashinferWorkspaceBuffers ws;
  ensure_workspace_buffers(&ws, device);
  return ws;
}

FlashinferWorkspaceBuffers& get_workspace_buffers_for_slot(
    const torch::Device& device,
    int64_t slot) {
  CHECK_GE(slot, 0);
  static thread_local std::unordered_map<int64_t, FlashinferWorkspaceBuffers>
      workspace_slots;
  const int64_t device_idx = static_cast<int64_t>(device.index());
  const int64_t key = (device_idx << 16) ^ slot;
  auto& ws = workspace_slots[key];
  ensure_workspace_buffers(&ws, device);
  return ws;
}

FlashinferWorkspaceBuffers& select_workspace_buffers(
    const torch::Device& device,
    FlashinferWorkspaceBuffers* workspace_override) {
  if (workspace_override != nullptr) {
    ensure_workspace_buffers(workspace_override, device);
    return *workspace_override;
  }
  return get_workspace_buffers(device);
}

ffi::Array<int64_t> deep_copy_plan_info(const ffi::Array<int64_t>& src) {
  if (!src.defined()) {
    return ffi::Array<int64_t>();
  }
  std::vector<int64_t> copied;
  copied.reserve(src.size());
  for (const auto& v : src) {
    copied.push_back(v);
  }
  return ffi::Array<int64_t>(copied.begin(), copied.end());
}

struct FlashinferPlan {
  std::string uri;
  std::string backend;
  ffi::Array<int64_t> plan_info;
};

torch::Tensor make_seq_indptr_host(int64_t seq_len) {
  CHECK_GE(seq_len, 0);
  return torch::tensor(
      {0, static_cast<int32_t>(seq_len)},
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
}

FlashinferPlan build_prefill_plan(
    const torch::Device& device,
    torch::ScalarType query_dtype,
    torch::ScalarType key_dtype,
    torch::ScalarType output_dtype,
    int64_t head_dim_qk,
    int64_t head_dim_vo,
    int64_t num_qo_heads,
    int64_t num_kv_heads,
    const torch::Tensor& q_cu_seq_lens_host,
    const torch::Tensor& kv_cu_seq_lens_host,
    bool causal,
    int64_t window_size_left,
    bool use_custom_mask,
    FlashinferWorkspaceBuffers* workspace_override = nullptr) {
  CHECK_EQ(q_cu_seq_lens_host.device().type(), torch::kCPU);
  CHECK_EQ(kv_cu_seq_lens_host.device().type(), torch::kCPU);
  CHECK_EQ(q_cu_seq_lens_host.scalar_type(), torch::kInt32);
  CHECK_EQ(kv_cu_seq_lens_host.scalar_type(), torch::kInt32);

  auto& ws = select_workspace_buffers(device, workspace_override);
  bind_tvmffi_stream_to_current_torch_stream(device);

  FlashinferPlan plan;
  plan.backend = determine_attention_backend(/*pos_encoding_mode=*/0,
                                             /*use_fp16_qk_reduction=*/false,
                                             use_custom_mask);
  plan.uri = get_batch_prefill_uri(plan.backend,
                                   query_dtype,
                                   key_dtype,
                                   output_dtype,
                                   q_cu_seq_lens_host.scalar_type(),
                                   head_dim_qk,
                                   head_dim_vo,
                                   /*pos_encoding_mode=*/0,
                                   /*use_sliding_window=*/false,
                                   /*use_logits_soft_cap=*/false,
                                   /*use_fp16_qk_reduction=*/false);

  torch::Tensor kv_len_arr_host =
      kv_cu_seq_lens_host.slice(0, 1) - kv_cu_seq_lens_host.slice(0, 0, -1);
  const int64_t total_num_rows = q_cu_seq_lens_host[-1].item<int64_t>();
  const int64_t batch_size = q_cu_seq_lens_host.size(0) - 1;

  auto plan_func = get_function(plan.uri, "plan");
  ffi::Array<int64_t> plan_result =
      Device::is_support_sm90a()
          ? plan_func(to_ffi_tensor(ws.float_workspace),
                      to_ffi_tensor(ws.int_workspace),
                      to_ffi_tensor(ws.page_locked_int_workspace),
                      to_ffi_tensor(q_cu_seq_lens_host),
                      to_ffi_tensor(kv_cu_seq_lens_host),
                      to_ffi_tensor(kv_len_arr_host),
                      total_num_rows,
                      batch_size,
                      num_qo_heads,
                      num_kv_heads,
                      /*page_size=*/1,
                      /*enable_cuda_graph=*/false,
                      head_dim_qk,
                      head_dim_vo,
                      causal,
                      window_size_left)
                .cast<ffi::Array<int64_t>>()
          : plan_func(to_ffi_tensor(ws.float_workspace),
                      to_ffi_tensor(ws.int_workspace),
                      to_ffi_tensor(ws.page_locked_int_workspace),
                      to_ffi_tensor(q_cu_seq_lens_host),
                      to_ffi_tensor(kv_cu_seq_lens_host),
                      to_ffi_tensor(kv_len_arr_host),
                      total_num_rows,
                      batch_size,
                      num_qo_heads,
                      num_kv_heads,
                      /*page_size=*/1,
                      /*enable_cuda_graph=*/false,
                      head_dim_qk,
                      head_dim_vo,
                      causal,
                      window_size_left,
                      /*fixed_split_size=*/-1,
                      /*disable_split_kv=*/false,
                      /*num_colocated_ctas=*/0)
                .cast<ffi::Array<int64_t>>();
  plan.plan_info = deep_copy_plan_info(plan_result);
  return plan;
}

torch::Tensor normalize_lse_shape(const torch::Tensor& raw_lse,
                                  int64_t seq_len,
                                  int64_t num_heads) {
  CHECK(raw_lse.defined());
  auto lse = raw_lse.contiguous();
  CHECK_EQ(lse.scalar_type(), torch::kFloat32);
  if (lse.dim() == 3 && lse.size(0) == seq_len && lse.size(1) == num_heads &&
      lse.size(2) == 1) {
    return lse;
  }
  if (lse.dim() == 2 && lse.size(0) == num_heads && lse.size(1) == seq_len) {
    return lse.transpose(0, 1).unsqueeze(-1).contiguous();
  }
  if (lse.dim() == 2 && lse.size(0) == seq_len && lse.size(1) == num_heads) {
    return lse.unsqueeze(-1).contiguous();
  }
  if (lse.dim() == 4 && lse.size(0) == 1 && lse.size(3) == 1 &&
      lse.size(1) == seq_len && lse.size(2) == num_heads) {
    return lse.squeeze(0).contiguous();
  }
  if (lse.dim() == 4 && lse.size(0) == 1 && lse.size(3) == 1 &&
      lse.size(1) == num_heads && lse.size(2) == seq_len) {
    return lse.squeeze(0).transpose(0, 1).contiguous();
  }
  CHECK(false) << "Unsupported output_lse shape: " << lse.sizes();
  return torch::Tensor();
}

torch::Tensor get_cached_mask_indptr(const torch::Device& device,
                                     int64_t num_bytes) {
  static thread_local std::unordered_map<int64_t, torch::Tensor> cached_indptr;
  const int64_t device_idx = static_cast<int64_t>(device.index());
  const int64_t key = (device_idx << 32) ^ num_bytes;
  auto it = cached_indptr.find(key);
  if (it != cached_indptr.end()) {
    return it->second;
  }
  auto indptr =
      torch::tensor({0, static_cast<int32_t>(num_bytes)},
                    torch::TensorOptions().dtype(torch::kInt32).device(device))
          .contiguous();
  it = cached_indptr.emplace(key, indptr).first;
  return it->second;
}

void launch_prefill_with_optional_packed_custom_mask(
    const FlashinferPlan& plan,
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    const torch::Tensor& q_cu_seq_lens_dev,
    const torch::Tensor& kv_cu_seq_lens_dev,
    bool causal,
    double sm_scale,
    torch::Tensor& output_snd,
    std::optional<torch::Tensor>& output_lse,
    const std::optional<torch::Tensor>& packed_custom_mask,
    FlashinferWorkspaceBuffers* workspace_override = nullptr) {
  const auto device = query_snd.device();
  auto& ws = select_workspace_buffers(device, workspace_override);

  if (plan.backend == "fa2") {
    std::optional<torch::Tensor> mask_indptr = std::nullopt;
    if (packed_custom_mask.has_value()) {
      mask_indptr =
          get_cached_mask_indptr(device, packed_custom_mask.value().numel());
    }
    get_function(plan.uri, "ragged_run")(
        to_ffi_tensor(ws.float_workspace),
        to_ffi_tensor(ws.int_workspace),
        plan.plan_info,
        to_ffi_tensor(query_snd),
        to_ffi_tensor(key_snd),
        to_ffi_tensor(value_snd),
        to_ffi_tensor(q_cu_seq_lens_dev),
        to_ffi_tensor(kv_cu_seq_lens_dev),
        to_ffi_tensor(output_snd),
        output_lse.has_value() ? to_ffi_tensor(output_lse.value())
                               : ffi::Optional<ffi::Tensor>(),
        /*mask_mode_code=*/packed_custom_mask.has_value() ? 2
                                                          : (causal ? 1 : 0),
        /*kv_layout_code=*/0,
        /*window_left=*/-1,
        support_pdl(),
        packed_custom_mask.has_value()
            ? to_ffi_tensor(packed_custom_mask.value())
            : ffi::Optional<ffi::Tensor>(),
        mask_indptr.has_value() ? to_ffi_tensor(mask_indptr.value())
                                : ffi::Optional<ffi::Tensor>(),
        /*maybe_alibi_slopes=*/ffi::Optional<ffi::Tensor>(),
        /*maybe_prefix_len_ptr=*/ffi::Optional<ffi::Tensor>(),
        /*maybe_token_pos_in_items_ptr=*/ffi::Optional<ffi::Tensor>(),
        /*maybe_max_item_len_ptr=*/ffi::Optional<ffi::Tensor>(),
        /*logits_soft_cap=*/0.0,
        sm_scale,
        /*rope_rcp_scale=*/1.0,
        /*rope_rcp_theta=*/1.0 / 10000.0,
        /*token_pos_in_items_len=*/0);
    return;
  }

  CHECK(!packed_custom_mask.has_value())
      << "Custom mask path currently requires fa2 backend";
  torch::Tensor v_scale = torch::Tensor();
  auto [scale_v_tensor, scale_v_scalar] = split_scale_param(v_scale);
  get_function(plan.uri, "ragged_run")(
      to_ffi_tensor(ws.float_workspace),
      to_ffi_tensor(ws.int_workspace),
      plan.plan_info,
      to_ffi_tensor(query_snd),
      to_ffi_tensor(key_snd),
      to_ffi_tensor(value_snd),
      to_ffi_tensor(q_cu_seq_lens_dev),
      to_ffi_tensor(kv_cu_seq_lens_dev),
      to_ffi_tensor(output_snd),
      output_lse.has_value() ? to_ffi_tensor(output_lse.value())
                             : ffi::Optional<ffi::Tensor>(),
      /*mask_mode_code=*/causal ? 1 : 0,
      /*kv_layout_code=*/0,
      /*window_left=*/-1,
      support_pdl(),
      /*maybe_prefix_len_ptr=*/ffi::Optional<ffi::Tensor>(),
      /*maybe_token_pos_in_items_ptr=*/ffi::Optional<ffi::Tensor>(),
      /*maybe_max_item_len_ptr=*/ffi::Optional<ffi::Tensor>(),
      scale_v_tensor.defined() ? to_ffi_tensor(scale_v_tensor)
                               : ffi::Optional<ffi::Tensor>(),
      /*logits_soft_cap=*/0.0,
      sm_scale,
      scale_v_scalar,
      /*token_pos_in_items_len=*/0);
}

struct AttentionRunResult {
  torch::Tensor out_snd;
  torch::Tensor lse_sh1;
  double get_ws_ms = 0.0;
};

enum class PreparedSegmentBackend {
  kFlashinfer,
  kCudaFullAttention,
};

struct PreparedFaSegment {
  PreparedSegmentBackend backend = PreparedSegmentBackend::kFlashinfer;
  FlashinferPlan plan;
  FlashinferWorkspaceBuffers* workspace = nullptr;
  torch::Tensor query_snd;
  torch::Tensor key_snd;
  torch::Tensor value_snd;
  torch::Tensor q_cu_dev;
  torch::Tensor kv_cu_dev;
  torch::Tensor out_snd;
  std::optional<torch::Tensor> output_lse = std::nullopt;
  bool causal = false;
  int64_t q_len = 0;
  int64_t num_heads = 0;
  double get_ws_ms = 0.0;
  std::optional<torch::Tensor> packed_custom_mask = std::nullopt;
};

bool should_use_cuda_full_attention(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    bool causal,
    bool need_lse,
    const std::optional<torch::Tensor>& packed_custom_mask) {
  if (!env_flag_enabled("XLLM_MTGR_USE_CUDA_FULL_ATTN", false)) {
    return false;
  }
  if (causal || need_lse || packed_custom_mask.has_value()) {
    return false;
  }
  if (!query_snd.is_cuda() || !key_snd.is_cuda()) {
    return false;
  }
  if (query_snd.scalar_type() != torch::kFloat16) {
    return false;
  }
  if (query_snd.size(1) != key_snd.size(1)) {
    return false;
  }
  constexpr int64_t kSmallQHeadDim = 128;
  constexpr int64_t kSmallQChunkSize = 64;
  constexpr int64_t kSmallQMaxChunks = 128;
  return query_snd.size(0) > 0 && query_snd.size(0) <= 8 &&
         query_snd.size(2) == kSmallQHeadDim &&
         key_snd.size(0) <= kSmallQChunkSize * kSmallQMaxChunks;
}

PreparedFaSegment prepare_fa_segment(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    bool causal,
    bool need_lse,
    StageTimeline* stage_timeline = nullptr,
    const std::optional<torch::Tensor>& packed_custom_mask = std::nullopt,
    const std::optional<torch::Tensor>& output_snd_override = std::nullopt,
    FlashinferWorkspaceBuffers* workspace_override = nullptr) {
  CHECK_EQ(query_snd.dim(), 3);
  CHECK_EQ(key_snd.dim(), 3);
  CHECK_EQ(value_snd.dim(), 3);
  CHECK_EQ(query_snd.size(2), key_snd.size(2));
  CHECK_EQ(key_snd.sizes(), value_snd.sizes());
  CHECK_GT(query_snd.size(0), 0);
  CHECK_GT(key_snd.size(0), 0);

  const auto device = query_snd.device();
  const int64_t q_len = query_snd.size(0);
  const int64_t kv_len = key_snd.size(0);
  const int64_t num_heads = query_snd.size(1);
  const int64_t num_kv_heads = key_snd.size(1);
  const int64_t head_dim = query_snd.size(2);

  auto q_cu_host = make_seq_indptr_host(q_len);
  auto kv_cu_host = make_seq_indptr_host(kv_len);
  auto q_cu_dev = q_cu_host.to(device);
  auto kv_cu_dev = kv_cu_host.to(device);

  PreparedFaSegment prepared;
  prepared.workspace = workspace_override;
  prepared.query_snd = query_snd;
  prepared.key_snd = key_snd;
  prepared.value_snd = value_snd;
  prepared.q_cu_dev = q_cu_dev;
  prepared.kv_cu_dev = kv_cu_dev;
  prepared.causal = causal;
  prepared.q_len = q_len;
  prepared.num_heads = num_heads;
  prepared.packed_custom_mask = packed_custom_mask;
  prepared.backend =
      should_use_cuda_full_attention(
          query_snd, key_snd, causal, need_lse, packed_custom_mask)
          ? PreparedSegmentBackend::kCudaFullAttention
          : PreparedSegmentBackend::kFlashinfer;

  if (prepared.backend == PreparedSegmentBackend::kFlashinfer) {
    auto t0 = std::chrono::steady_clock::now();
    if (stage_timeline != nullptr) {
      stage_timeline->has_getws = true;
      stage_timeline->getws_submit = t0;
    }
    prepared.plan =
        build_prefill_plan(device,
                           query_snd.scalar_type(),
                           key_snd.scalar_type(),
                           query_snd.scalar_type(),
                           head_dim,
                           head_dim,
                           num_heads,
                           num_kv_heads,
                           q_cu_host,
                           kv_cu_host,
                           causal,
                           /*window_size_left=*/-1,
                           /*use_custom_mask=*/packed_custom_mask.has_value(),
                           workspace_override);
    prepared.get_ws_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - t0)
                             .count();
    if (stage_timeline != nullptr) {
      stage_timeline->get_ws_ms = prepared.get_ws_ms;
    }
  }

  if (output_snd_override.has_value()) {
    prepared.out_snd = output_snd_override.value();
    CHECK_EQ(prepared.out_snd.sizes(), query_snd.sizes());
  } else {
    prepared.out_snd =
        torch::empty({q_len, num_heads, head_dim}, query_snd.options());
  }
  if (need_lse) {
    prepared.output_lse = torch::empty(
        {q_len, num_heads, 1},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
  }
  return prepared;
}

AttentionRunResult launch_prepared_fa_segment(
    PreparedFaSegment* prepared,
    double sm_scale,
    StageTimeline* stage_timeline = nullptr) {
  CHECK(prepared != nullptr);
  const auto device = prepared->query_snd.device();
  if (stage_timeline != nullptr) {
    record_stage_begin_if_needed(stage_timeline, device);
  }

  if (prepared->backend == PreparedSegmentBackend::kCudaFullAttention) {
    full_attention_cuda(prepared->query_snd,
                        prepared->key_snd,
                        prepared->value_snd,
                        sm_scale,
                        prepared->out_snd);
  } else if (prepared->packed_custom_mask.has_value()) {
    launch_prefill_with_optional_packed_custom_mask(
        prepared->plan,
        prepared->query_snd,
        prepared->key_snd,
        prepared->value_snd,
        prepared->q_cu_dev,
        prepared->kv_cu_dev,
        prepared->causal,
        sm_scale,
        prepared->out_snd,
        prepared->output_lse,
        prepared->packed_custom_mask,
        prepared->workspace);
  } else if (prepared->causal) {
    auto& ws = select_workspace_buffers(device, prepared->workspace);
    batch_prefill(prepared->plan.uri,
                  prepared->plan.plan_info,
                  ws.float_workspace,
                  ws.int_workspace,
                  ws.page_locked_int_workspace,
                  prepared->query_snd,
                  prepared->key_snd,
                  prepared->value_snd,
                  prepared->q_cu_dev,
                  prepared->kv_cu_dev,
                  /*window_left=*/-1,
                  sm_scale,
                  prepared->out_snd,
                  prepared->output_lse);
  } else {
    auto& ws = select_workspace_buffers(device, prepared->workspace);
    batch_prefill_non_causal(prepared->plan.uri,
                             prepared->plan.plan_info,
                             ws.float_workspace,
                             ws.int_workspace,
                             ws.page_locked_int_workspace,
                             prepared->query_snd,
                             prepared->key_snd,
                             prepared->value_snd,
                             prepared->q_cu_dev,
                             prepared->kv_cu_dev,
                             /*window_left=*/-1,
                             sm_scale,
                             prepared->out_snd,
                             prepared->output_lse);
  }

  if (stage_timeline != nullptr) {
    record_stage_end_if_needed(stage_timeline, device);
  }

  AttentionRunResult result;
  result.out_snd = prepared->out_snd;
  result.get_ws_ms = prepared->get_ws_ms;
  if (prepared->output_lse.has_value()) {
    result.lse_sh1 = normalize_lse_shape(
        prepared->output_lse.value(), prepared->q_len, prepared->num_heads);
  }
  return result;
}

AttentionRunResult run_fa_segment(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    double sm_scale,
    bool causal,
    bool need_lse,
    StageTimeline* stage_timeline = nullptr,
    const std::optional<torch::Tensor>& packed_custom_mask = std::nullopt,
    const std::optional<torch::Tensor>& output_snd_override = std::nullopt,
    FlashinferWorkspaceBuffers* workspace_override = nullptr) {
  auto prepared = prepare_fa_segment(query_snd,
                                     key_snd,
                                     value_snd,
                                     causal,
                                     need_lse,
                                     stage_timeline,
                                     packed_custom_mask,
                                     output_snd_override,
                                     workspace_override);
  return launch_prepared_fa_segment(&prepared, sm_scale, stage_timeline);
}

int64_t packed_mask_num_bytes(int64_t q_len, int64_t kv_len) {
  CHECK_GT(q_len, 0);
  CHECK_GT(kv_len, 0);
  return (q_len * kv_len + 7) / 8;
}

torch::Tensor reserve_reusable_packed_mask_buffer(const torch::Device& device,
                                                  int64_t mask_kind,
                                                  int64_t num_bytes) {
  static thread_local std::unordered_map<int64_t, torch::Tensor>
      reusable_buffers;
  CHECK_GE(mask_kind, 0);
  CHECK_LE(mask_kind, 2);
  CHECK_GT(num_bytes, 0);
  const int64_t device_idx = static_cast<int64_t>(device.index());
  const int64_t key = (device_idx << 8) ^ mask_kind;
  auto it = reusable_buffers.find(key);
  if (it == reusable_buffers.end() || !it->second.defined() ||
      it->second.numel() < num_bytes || it->second.device() != device) {
    auto buffer = torch::empty(
        {num_bytes},
        torch::TensorOptions().dtype(torch::kUInt8).device(device));
    it = reusable_buffers.insert_or_assign(key, buffer).first;
  }
  return it->second.slice(0, 0, num_bytes);
}

torch::Tensor build_one_stage_packed_mask_reusing_buffer(
    const torch::Device& device,
    int64_t history_len,
    int64_t context_len,
    int64_t realtime_len,
    int64_t target_len) {
  const int64_t total_len =
      history_len + context_len + realtime_len + target_len;
  auto packed = reserve_reusable_packed_mask_buffer(
      device, /*mask_kind=*/0, packed_mask_num_bytes(total_len, total_len));
  build_mtgr_packed_mask(packed,
                         total_len,
                         total_len,
                         history_len,
                         context_len,
                         realtime_len,
                         /*mask_kind=*/0);
  return packed;
}

torch::Tensor build_rt_tgt_trapezoid_packed_mask_reusing_buffer(
    const torch::Device& device,
    int64_t prefix_len,
    int64_t realtime_len,
    int64_t target_len) {
  const int64_t q_len = realtime_len + target_len;
  const int64_t kv_len = prefix_len + realtime_len;
  auto packed = reserve_reusable_packed_mask_buffer(
      device, /*mask_kind=*/1, packed_mask_num_bytes(q_len, kv_len));
  build_mtgr_packed_mask(packed,
                         q_len,
                         kv_len,
                         prefix_len,
                         /*context_len=*/0,
                         realtime_len,
                         /*mask_kind=*/1);
  return packed;
}

torch::Tensor build_partial_rt_one_stage_packed_mask_reusing_buffer(
    const torch::Device& device,
    int64_t matched_prefix_len,
    int64_t realtime_unmatched_len,
    int64_t target_len) {
  const int64_t q_len = realtime_unmatched_len + target_len;
  const int64_t kv_len =
      matched_prefix_len + realtime_unmatched_len + target_len;
  auto packed = reserve_reusable_packed_mask_buffer(
      device, /*mask_kind=*/2, packed_mask_num_bytes(q_len, kv_len));
  build_mtgr_packed_mask(packed,
                         q_len,
                         kv_len,
                         matched_prefix_len,
                         /*context_len=*/0,
                         realtime_unmatched_len,
                         /*mask_kind=*/2);
  return packed;
}

void merge_target_diag_attention_into(const torch::Tensor& hcr_out_snd,
                                      const torch::Tensor& hcr_lse_sh1,
                                      const torch::Tensor& target_query_snd,
                                      const torch::Tensor& target_key_snd,
                                      const torch::Tensor& target_value_snd,
                                      double sm_scale,
                                      torch::Tensor merged_out_snd) {
  merge_target_diag_attention_cuda(hcr_out_snd,
                                   hcr_lse_sh1,
                                   target_query_snd,
                                   target_key_snd,
                                   target_value_snd,
                                   sm_scale,
                                   merged_out_snd);
}

struct MTGRNoMatchBatchLayout {
  std::vector<int64_t> q_seq_lens;
  std::vector<int64_t> q_seq_starts;
  std::vector<int64_t> segment_offsets;
  std::vector<int64_t> segment_rules;
  std::vector<int64_t> history_lens;
  std::vector<int64_t> context_lens;
  std::vector<int64_t> realtime_lens;
  std::vector<int64_t> target_lens;
  std::vector<int64_t> target_seq_starts;
  std::vector<int64_t> row_to_batch;
  std::vector<int64_t> target_row_indices;
  torch::Tensor q_seq_starts_i32;
  torch::Tensor history_lens_i32;
  torch::Tensor context_lens_i32;
  torch::Tensor realtime_lens_i32;
  torch::Tensor target_lens_i32;
  torch::Tensor target_seq_starts_i32;
  torch::Tensor row_to_batch_i32;
  torch::Tensor target_row_indices_i64;
  torch::Tensor segment_offsets_i32;
  torch::Tensor segment_rules_i32;
  int64_t batch_size = 0;
  int64_t total_q = 0;
  int64_t total_target = 0;
  int64_t max_q = 0;
  int64_t max_target = 0;
  int64_t max_history = 0;
};

struct MTGRSegmentedNoMatchBatchLayout {
  std::vector<int64_t> q_seq_lens;
  std::vector<int64_t> segment_offsets;
  std::vector<int64_t> segment_rules;
  std::vector<int64_t> row_to_batch;
  std::vector<int64_t> target_seq_starts;
  std::vector<int64_t> target_lens;
  std::vector<int64_t> target_row_indices;
  std::vector<int64_t> target_offsets;
  torch::Tensor segment_offsets_i32;
  torch::Tensor segment_rules_i32;
  torch::Tensor row_to_batch_i32;
  torch::Tensor target_seq_starts_i32;
  torch::Tensor target_lens_i32;
  torch::Tensor target_row_indices_i64;
  torch::Tensor q_seq_starts_i32;
  torch::Tensor target_offsets_i32;
  int64_t batch_size = 0;
  int64_t num_segments = 0;
  int64_t total_q = 0;
  int64_t total_target = 0;
  int64_t max_q = 0;
  int64_t max_target = 0;
};

struct MTGRRaggedSegmentBatchLayout {
  std::vector<int64_t> q_seq_lens;
  std::vector<int64_t> segment_offsets;
  std::vector<int64_t> segment_rules;
  std::vector<int64_t> row_to_batch;
  torch::Tensor segment_offsets_i32;
  torch::Tensor segment_rules_i32;
  torch::Tensor row_to_batch_i32;
  int64_t batch_size = 0;
  int64_t num_segments = 0;
  int64_t total_q = 0;
  int64_t max_q = 0;
};

// Historical four-segment partial-match packed layout.
struct MTGRFourSegmentPartialBatchLayout {
  std::vector<int64_t> q_seq_lens;
  std::vector<int64_t> q_seq_starts;
  std::vector<int64_t> segment_offsets;
  std::vector<int64_t> segment_rules;
  std::vector<int64_t> matched_prefix_lens;
  std::vector<int64_t> realtime_unmatched_lens;
  std::vector<int64_t> target_lens;
  std::vector<int64_t> target_seq_starts;
  torch::Tensor q_seq_starts_i32;
  torch::Tensor matched_prefix_lens_i32;
  torch::Tensor realtime_unmatched_lens_i32;
  torch::Tensor target_lens_i32;
  torch::Tensor target_seq_starts_i32;
  torch::Tensor segment_offsets_i32;
  torch::Tensor segment_rules_i32;
  torch::Tensor block_table_i32;
  int64_t block_table_stride = 0;
  int64_t batch_size = 0;
  int64_t total_q = 0;
  int64_t total_target = 0;
  int64_t max_q = 0;
  int64_t max_target = 0;
  int64_t max_matched_prefix = 0;
  int64_t max_realtime_unmatched = 0;
};

size_t hash_cpu_int64_tensor_values(const torch::Tensor& tensor) {
  CHECK(tensor.defined());
  CHECK_EQ(tensor.dim(), 1);
  CHECK_EQ(tensor.device().type(), torch::kCPU);
  CHECK_EQ(tensor.scalar_type(), torch::kInt64);
  size_t h = std::hash<int64_t>{}(tensor.size(0));
  const auto mix = [&h](size_t v) {
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  };
  auto accessor = tensor.accessor<int64_t, 1>();
  for (int64_t i = 0; i < tensor.size(0); ++i) {
    mix(std::hash<int64_t>{}(accessor[i]));
  }
  return h;
}

struct MTGRNoMatchBatchLayoutCacheKey {
  const void* q_seq_lens_ptr = nullptr;
  const void* kv_seq_lens_ptr = nullptr;
  const void* history_lens_ptr = nullptr;
  const void* context_lens_ptr = nullptr;
  const void* realtime_lens_ptr = nullptr;
  const void* target_lens_ptr = nullptr;
  const void* matched_prefix_lens_ptr = nullptr;
  size_t q_seq_lens_hash = 0;
  size_t kv_seq_lens_hash = 0;
  size_t history_lens_hash = 0;
  size_t context_lens_hash = 0;
  size_t realtime_lens_hash = 0;
  size_t target_lens_hash = 0;
  size_t matched_prefix_lens_hash = 0;
  int device_idx = -1;

  bool operator==(const MTGRNoMatchBatchLayoutCacheKey& other) const {
    return q_seq_lens_ptr == other.q_seq_lens_ptr &&
           kv_seq_lens_ptr == other.kv_seq_lens_ptr &&
           history_lens_ptr == other.history_lens_ptr &&
           context_lens_ptr == other.context_lens_ptr &&
           realtime_lens_ptr == other.realtime_lens_ptr &&
           target_lens_ptr == other.target_lens_ptr &&
           matched_prefix_lens_ptr == other.matched_prefix_lens_ptr &&
           q_seq_lens_hash == other.q_seq_lens_hash &&
           kv_seq_lens_hash == other.kv_seq_lens_hash &&
           history_lens_hash == other.history_lens_hash &&
           context_lens_hash == other.context_lens_hash &&
           realtime_lens_hash == other.realtime_lens_hash &&
           target_lens_hash == other.target_lens_hash &&
           matched_prefix_lens_hash == other.matched_prefix_lens_hash &&
           device_idx == other.device_idx;
  }
};

struct MTGRFourSegmentPartialBatchLayoutCacheKey {
  const void* q_seq_lens_ptr = nullptr;
  const void* kv_seq_lens_ptr = nullptr;
  const void* history_lens_ptr = nullptr;
  const void* context_lens_ptr = nullptr;
  const void* realtime_lens_ptr = nullptr;
  const void* target_lens_ptr = nullptr;
  const void* matched_prefix_lens_ptr = nullptr;
  const void* block_table_ptr = nullptr;
  size_t q_seq_lens_hash = 0;
  size_t kv_seq_lens_hash = 0;
  size_t history_lens_hash = 0;
  size_t context_lens_hash = 0;
  size_t realtime_lens_hash = 0;
  size_t target_lens_hash = 0;
  size_t matched_prefix_lens_hash = 0;
  int device_idx = -1;

  bool operator==(
      const MTGRFourSegmentPartialBatchLayoutCacheKey& other) const {
    return q_seq_lens_ptr == other.q_seq_lens_ptr &&
           kv_seq_lens_ptr == other.kv_seq_lens_ptr &&
           history_lens_ptr == other.history_lens_ptr &&
           context_lens_ptr == other.context_lens_ptr &&
           realtime_lens_ptr == other.realtime_lens_ptr &&
           target_lens_ptr == other.target_lens_ptr &&
           matched_prefix_lens_ptr == other.matched_prefix_lens_ptr &&
           block_table_ptr == other.block_table_ptr &&
           q_seq_lens_hash == other.q_seq_lens_hash &&
           kv_seq_lens_hash == other.kv_seq_lens_hash &&
           history_lens_hash == other.history_lens_hash &&
           context_lens_hash == other.context_lens_hash &&
           realtime_lens_hash == other.realtime_lens_hash &&
           target_lens_hash == other.target_lens_hash &&
           matched_prefix_lens_hash == other.matched_prefix_lens_hash &&
           device_idx == other.device_idx;
  }
};

struct MTGRNoMatchBatchLayoutCacheKeyHash {
  size_t operator()(const MTGRNoMatchBatchLayoutCacheKey& key) const {
    size_t h = 0;
    const auto mix = [&h](size_t v) {
      h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    };
    mix(std::hash<const void*>{}(key.q_seq_lens_ptr));
    mix(std::hash<const void*>{}(key.kv_seq_lens_ptr));
    mix(std::hash<const void*>{}(key.history_lens_ptr));
    mix(std::hash<const void*>{}(key.context_lens_ptr));
    mix(std::hash<const void*>{}(key.realtime_lens_ptr));
    mix(std::hash<const void*>{}(key.target_lens_ptr));
    mix(std::hash<const void*>{}(key.matched_prefix_lens_ptr));
    mix(key.q_seq_lens_hash);
    mix(key.kv_seq_lens_hash);
    mix(key.history_lens_hash);
    mix(key.context_lens_hash);
    mix(key.realtime_lens_hash);
    mix(key.target_lens_hash);
    mix(key.matched_prefix_lens_hash);
    mix(std::hash<int>{}(key.device_idx));
    return h;
  }
};

struct MTGRFourSegmentPartialBatchLayoutCacheKeyHash {
  size_t operator()(
      const MTGRFourSegmentPartialBatchLayoutCacheKey& key) const {
    size_t h = 0;
    const auto mix = [&h](size_t v) {
      h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    };
    mix(std::hash<const void*>{}(key.q_seq_lens_ptr));
    mix(std::hash<const void*>{}(key.kv_seq_lens_ptr));
    mix(std::hash<const void*>{}(key.history_lens_ptr));
    mix(std::hash<const void*>{}(key.context_lens_ptr));
    mix(std::hash<const void*>{}(key.realtime_lens_ptr));
    mix(std::hash<const void*>{}(key.target_lens_ptr));
    mix(std::hash<const void*>{}(key.matched_prefix_lens_ptr));
    mix(std::hash<const void*>{}(key.block_table_ptr));
    mix(key.q_seq_lens_hash);
    mix(key.kv_seq_lens_hash);
    mix(key.history_lens_hash);
    mix(key.context_lens_hash);
    mix(key.realtime_lens_hash);
    mix(key.target_lens_hash);
    mix(key.matched_prefix_lens_hash);
    mix(std::hash<int>{}(key.device_idx));
    return h;
  }
};

std::vector<int64_t> tensor_to_int64_vec_cpu(const torch::Tensor& tensor) {
  CHECK(tensor.defined());
  CHECK_EQ(tensor.dim(), 1);
  CHECK_EQ(tensor.device().type(), torch::kCPU);
  CHECK_EQ(tensor.scalar_type(), torch::kInt64);
  std::vector<int64_t> values;
  values.reserve(static_cast<size_t>(tensor.size(0)));
  auto accessor = tensor.accessor<int64_t, 1>();
  for (int64_t i = 0; i < tensor.size(0); ++i) {
    values.push_back(accessor[i]);
  }
  return values;
}

torch::Tensor make_int32_tensor_on_device(const std::vector<int64_t>& values,
                                          const torch::Device& device) {
  return torch::tensor(
             values, torch::TensorOptions().dtype(torch::kInt32).device(device))
      .contiguous();
}

torch::Tensor make_int64_tensor_on_device(const std::vector<int64_t>& values,
                                          const torch::Device& device) {
  return torch::tensor(
             values, torch::TensorOptions().dtype(torch::kInt64).device(device))
      .contiguous();
}

MTGRSegmentedNoMatchBatchLayout build_segmented_no_match_batch_layout(
    const std::vector<std::vector<int64_t>>& segment_lens,
    const std::vector<int64_t>& segment_rules,
    const torch::Device& device) {
  CHECK(!segment_lens.empty());
  CHECK(!segment_rules.empty());
  const int64_t batch_size = static_cast<int64_t>(segment_lens.size());
  const int64_t num_segments = static_cast<int64_t>(segment_rules.size());
  CHECK_GE(num_segments, 2);
  CHECK_EQ(segment_rules.back(), 2)
      << "The current segmented no_match path reserves the last segment for "
         "target diagonal merge";
  for (int64_t i = 0; i + 1 < num_segments; ++i) {
    CHECK(segment_rules[i] == 0 || segment_rules[i] == 1)
        << "Non-target segmented rules must be 0(causal) or 1(full)";
  }

  MTGRSegmentedNoMatchBatchLayout layout;
  layout.batch_size = batch_size;
  layout.num_segments = num_segments;
  layout.segment_rules = segment_rules;
  layout.segment_offsets.reserve(
      static_cast<size_t>(batch_size * (num_segments + 1)));
  layout.q_seq_lens.reserve(static_cast<size_t>(batch_size));
  layout.target_seq_starts.reserve(static_cast<size_t>(batch_size));
  layout.target_lens.reserve(static_cast<size_t>(batch_size));
  layout.target_offsets.reserve(static_cast<size_t>(batch_size));

  for (int64_t b = 0; b < batch_size; ++b) {
    CHECK_EQ(static_cast<int64_t>(segment_lens[b].size()), num_segments)
        << "All requests in a segmented batch must have the same segment count";
    const int64_t request_start = layout.total_q;
    int64_t cursor = request_start;
    layout.segment_offsets.push_back(cursor);
    for (int64_t s = 0; s < num_segments; ++s) {
      const int64_t len = segment_lens[b][s];
      CHECK_GT(len, 0);
      cursor += len;
      layout.segment_offsets.push_back(cursor);
    }

    const int64_t q_len = cursor - request_start;
    const int64_t target_len = segment_lens[b].back();
    const int64_t target_begin = cursor - target_len;
    const int64_t target_offset = target_begin - request_start;
    layout.q_seq_lens.push_back(q_len);
    layout.row_to_batch.insert(layout.row_to_batch.end(), q_len, b);
    layout.target_seq_starts.push_back(layout.total_target);
    layout.target_lens.push_back(target_len);
    layout.target_offsets.push_back(target_offset);
    for (int64_t j = 0; j < target_len; ++j) {
      layout.target_row_indices.push_back(target_begin + j);
    }

    layout.total_q += q_len;
    layout.total_target += target_len;
    layout.max_q = std::max(layout.max_q, q_len);
    layout.max_target = std::max(layout.max_target, target_len);
  }

  CHECK_EQ(static_cast<int64_t>(layout.row_to_batch.size()), layout.total_q);
  CHECK_EQ(static_cast<int64_t>(layout.target_row_indices.size()),
           layout.total_target);
  layout.segment_offsets_i32 =
      make_int32_tensor_on_device(layout.segment_offsets, device)
          .view({batch_size, num_segments + 1})
          .contiguous();
  layout.segment_rules_i32 =
      make_int32_tensor_on_device(layout.segment_rules, device);
  layout.row_to_batch_i32 =
      make_int32_tensor_on_device(layout.row_to_batch, device);
  layout.target_seq_starts_i32 =
      make_int32_tensor_on_device(layout.target_seq_starts, device);
  layout.target_lens_i32 =
      make_int32_tensor_on_device(layout.target_lens, device);
  layout.target_row_indices_i64 =
      make_int64_tensor_on_device(layout.target_row_indices, device);
  layout.q_seq_starts_i32 = torch::empty(
      {batch_size}, torch::TensorOptions().dtype(torch::kInt32).device(device));
  layout.target_offsets_i32 =
      make_int32_tensor_on_device(layout.target_offsets, device);
  layout.q_seq_starts_i32.copy_(layout.segment_offsets_i32.select(1, 0));
  return layout;
}

MTGRRaggedSegmentBatchLayout build_ragged_segment_batch_layout(
    const std::vector<std::vector<int64_t>>& segment_lens,
    const std::vector<int64_t>& segment_rules,
    const torch::Device& device) {
  CHECK(!segment_lens.empty());
  CHECK(!segment_rules.empty());
  const int64_t batch_size = static_cast<int64_t>(segment_lens.size());
  const int64_t num_segments = static_cast<int64_t>(segment_rules.size());
  CHECK_GE(num_segments, 2);
  for (const int64_t rule : segment_rules) {
    CHECK(rule == 0 || rule == 1 || rule == 2)
        << "Segment rules must be causal(0), full(1), or diagonal(2)";
  }

  MTGRRaggedSegmentBatchLayout layout;
  layout.batch_size = batch_size;
  layout.num_segments = num_segments;
  layout.segment_rules = segment_rules;
  layout.segment_offsets.reserve(
      static_cast<size_t>(batch_size * (num_segments + 1)));
  layout.q_seq_lens.reserve(static_cast<size_t>(batch_size));
  layout.row_to_batch.reserve(static_cast<size_t>(batch_size));

  for (int64_t b = 0; b < batch_size; ++b) {
    CHECK_EQ(static_cast<int64_t>(segment_lens[b].size()), num_segments)
        << "All requests in a ragged batch must have the same segment count";
    const int64_t request_start = layout.total_q;
    int64_t cursor = request_start;
    layout.segment_offsets.push_back(cursor);
    for (int64_t s = 0; s < num_segments; ++s) {
      const int64_t len = segment_lens[b][s];
      CHECK_GT(len, 0);
      cursor += len;
      layout.segment_offsets.push_back(cursor);
    }

    const int64_t q_len = cursor - request_start;
    layout.q_seq_lens.push_back(q_len);
    layout.row_to_batch.insert(layout.row_to_batch.end(), q_len, b);
    layout.total_q += q_len;
    layout.max_q = std::max(layout.max_q, q_len);
  }

  CHECK_EQ(static_cast<int64_t>(layout.row_to_batch.size()), layout.total_q);
  layout.segment_offsets_i32 =
      make_int32_tensor_on_device(layout.segment_offsets, device)
          .view({batch_size, num_segments + 1})
          .contiguous();
  layout.segment_rules_i32 =
      make_int32_tensor_on_device(layout.segment_rules, device);
  layout.row_to_batch_i32 =
      make_int32_tensor_on_device(layout.row_to_batch, device);
  return layout;
}

MTGRNoMatchBatchLayout build_no_match_batch_layout_uncached(
    const torch::Tensor& q_seq_lens,
    const torch::Tensor& kv_seq_lens,
    const torch::Tensor& history_lens,
    const torch::Tensor& context_lens,
    const torch::Tensor& realtime_lens,
    const torch::Tensor& target_lens,
    const torch::Tensor& matched_prefix_lens,
    const torch::Device& device) {
  MTGRNoMatchBatchLayout layout;
  layout.q_seq_lens = tensor_to_int64_vec_cpu(q_seq_lens);
  const auto kv_seq_lens_host = tensor_to_int64_vec_cpu(kv_seq_lens);
  layout.history_lens = tensor_to_int64_vec_cpu(history_lens);
  layout.context_lens = tensor_to_int64_vec_cpu(context_lens);
  layout.realtime_lens = tensor_to_int64_vec_cpu(realtime_lens);
  layout.target_lens = tensor_to_int64_vec_cpu(target_lens);
  const auto matched_prefix_lens_host =
      tensor_to_int64_vec_cpu(matched_prefix_lens);

  const int64_t batch_size = static_cast<int64_t>(layout.q_seq_lens.size());
  layout.batch_size = batch_size;
  layout.segment_rules = {0, 1, 0, 2};
  CHECK_EQ(kv_seq_lens_host.size(), layout.q_seq_lens.size());
  CHECK_EQ(layout.history_lens.size(), layout.q_seq_lens.size());
  CHECK_EQ(layout.context_lens.size(), layout.q_seq_lens.size());
  CHECK_EQ(layout.realtime_lens.size(), layout.q_seq_lens.size());
  CHECK_EQ(layout.target_lens.size(), layout.q_seq_lens.size());
  CHECK_EQ(matched_prefix_lens_host.size(), layout.q_seq_lens.size());

  layout.q_seq_starts.reserve(static_cast<size_t>(batch_size));
  layout.target_seq_starts.reserve(static_cast<size_t>(batch_size));
  for (int64_t i = 0; i < batch_size; ++i) {
    const int64_t q_len = layout.q_seq_lens[i];
    const int64_t kv_len = kv_seq_lens_host[i];
    const int64_t h = layout.history_lens[i];
    const int64_t c = layout.context_lens[i];
    const int64_t r = layout.realtime_lens[i];
    const int64_t t = layout.target_lens[i];
    const int64_t matched = matched_prefix_lens_host[i];
    CHECK_EQ(matched, 0)
        << "Batched fused CUDA test path currently supports no_match only";
    CHECK_EQ(q_len, kv_len);
    CHECK_EQ(q_len, h + c + r + t);
    CHECK_GT(h, 0);
    CHECK_GT(c, 0);
    CHECK_GT(r, 0);
    CHECK_GT(t, 0);

    layout.q_seq_starts.push_back(layout.total_q);
    const int64_t request_start = layout.total_q;
    layout.segment_offsets.push_back(request_start);
    layout.segment_offsets.push_back(request_start + h);
    layout.segment_offsets.push_back(request_start + h + c);
    layout.segment_offsets.push_back(request_start + h + c + r);
    layout.segment_offsets.push_back(request_start + h + c + r + t);
    layout.target_seq_starts.push_back(layout.total_target);
    layout.row_to_batch.insert(layout.row_to_batch.end(), q_len, i);
    layout.max_q = std::max(layout.max_q, q_len);
    layout.max_target = std::max(layout.max_target, t);
    layout.max_history = std::max(layout.max_history, h);

    const int64_t target_begin = layout.total_q + h + c + r;
    for (int64_t j = 0; j < t; ++j) {
      layout.target_row_indices.push_back(target_begin + j);
    }

    layout.total_q += q_len;
    layout.total_target += t;
  }

  CHECK_EQ(static_cast<int64_t>(layout.row_to_batch.size()), layout.total_q);
  CHECK_EQ(static_cast<int64_t>(layout.target_row_indices.size()),
           layout.total_target);
  layout.q_seq_starts_i32 =
      make_int32_tensor_on_device(layout.q_seq_starts, device);
  layout.history_lens_i32 =
      make_int32_tensor_on_device(layout.history_lens, device);
  layout.context_lens_i32 =
      make_int32_tensor_on_device(layout.context_lens, device);
  layout.realtime_lens_i32 =
      make_int32_tensor_on_device(layout.realtime_lens, device);
  layout.target_lens_i32 =
      make_int32_tensor_on_device(layout.target_lens, device);
  layout.target_seq_starts_i32 =
      make_int32_tensor_on_device(layout.target_seq_starts, device);
  layout.row_to_batch_i32 =
      make_int32_tensor_on_device(layout.row_to_batch, device);
  layout.target_row_indices_i64 =
      make_int64_tensor_on_device(layout.target_row_indices, device);
  layout.segment_offsets_i32 =
      make_int32_tensor_on_device(layout.segment_offsets, device)
          .view({batch_size,
                 static_cast<int64_t>(layout.segment_rules.size()) + 1})
          .contiguous();
  layout.segment_rules_i32 =
      make_int32_tensor_on_device(layout.segment_rules, device);
  return layout;
}

const MTGRNoMatchBatchLayout& get_cached_no_match_batch_layout(
    const torch::Tensor& q_seq_lens,
    const torch::Tensor& kv_seq_lens,
    const torch::Tensor& history_lens,
    const torch::Tensor& context_lens,
    const torch::Tensor& realtime_lens,
    const torch::Tensor& target_lens,
    const torch::Tensor& matched_prefix_lens,
    const torch::Device& device) {
  static thread_local std::unordered_map<MTGRNoMatchBatchLayoutCacheKey,
                                         MTGRNoMatchBatchLayout,
                                         MTGRNoMatchBatchLayoutCacheKeyHash>
      cache;
  const MTGRNoMatchBatchLayoutCacheKey key{
      .q_seq_lens_ptr = q_seq_lens.data_ptr(),
      .kv_seq_lens_ptr = kv_seq_lens.data_ptr(),
      .history_lens_ptr = history_lens.data_ptr(),
      .context_lens_ptr = context_lens.data_ptr(),
      .realtime_lens_ptr = realtime_lens.data_ptr(),
      .target_lens_ptr = target_lens.data_ptr(),
      .matched_prefix_lens_ptr = matched_prefix_lens.data_ptr(),
      .q_seq_lens_hash = hash_cpu_int64_tensor_values(q_seq_lens),
      .kv_seq_lens_hash = hash_cpu_int64_tensor_values(kv_seq_lens),
      .history_lens_hash = hash_cpu_int64_tensor_values(history_lens),
      .context_lens_hash = hash_cpu_int64_tensor_values(context_lens),
      .realtime_lens_hash = hash_cpu_int64_tensor_values(realtime_lens),
      .target_lens_hash = hash_cpu_int64_tensor_values(target_lens),
      .matched_prefix_lens_hash =
          hash_cpu_int64_tensor_values(matched_prefix_lens),
      .device_idx = device.index(),
  };
  auto it = cache.find(key);
  if (it == cache.end()) {
    it = cache
             .emplace(key,
                      build_no_match_batch_layout_uncached(q_seq_lens,
                                                           kv_seq_lens,
                                                           history_lens,
                                                           context_lens,
                                                           realtime_lens,
                                                           target_lens,
                                                           matched_prefix_lens,
                                                           device))
             .first;
  }
  return it->second;
}

MTGRFourSegmentPartialBatchLayout
build_four_segment_partial_batch_layout_uncached(
    const torch::Tensor& q_seq_lens,
    const torch::Tensor& kv_seq_lens,
    const torch::Tensor& history_lens,
    const torch::Tensor& context_lens,
    const torch::Tensor& realtime_lens,
    const torch::Tensor& target_lens,
    const torch::Tensor& matched_prefix_lens,
    const torch::Tensor& block_table,
    const torch::Device& device) {
  MTGRFourSegmentPartialBatchLayout layout;
  layout.q_seq_lens = tensor_to_int64_vec_cpu(q_seq_lens);
  const auto kv_seq_lens_host = tensor_to_int64_vec_cpu(kv_seq_lens);
  const auto history_lens_host = tensor_to_int64_vec_cpu(history_lens);
  const auto context_lens_host = tensor_to_int64_vec_cpu(context_lens);
  const auto realtime_lens_host = tensor_to_int64_vec_cpu(realtime_lens);
  layout.target_lens = tensor_to_int64_vec_cpu(target_lens);
  layout.matched_prefix_lens = tensor_to_int64_vec_cpu(matched_prefix_lens);

  const int64_t batch_size = static_cast<int64_t>(layout.q_seq_lens.size());
  layout.batch_size = batch_size;
  layout.segment_rules = {0, 1, 0, 2};
  CHECK_EQ(kv_seq_lens_host.size(), layout.q_seq_lens.size());
  CHECK_EQ(history_lens_host.size(), layout.q_seq_lens.size());
  CHECK_EQ(context_lens_host.size(), layout.q_seq_lens.size());
  CHECK_EQ(realtime_lens_host.size(), layout.q_seq_lens.size());
  CHECK_EQ(layout.target_lens.size(), layout.q_seq_lens.size());
  CHECK_EQ(layout.matched_prefix_lens.size(), layout.q_seq_lens.size());
  CHECK(block_table.defined());
  CHECK_EQ(block_table.dim(), 2);
  CHECK_EQ(block_table.size(0), batch_size);
  CHECK_EQ(block_table.scalar_type(), torch::kInt32);
  CHECK_EQ(block_table.device(), device);

  layout.block_table_i32 = block_table.contiguous();
  layout.block_table_stride = layout.block_table_i32.size(1);
  layout.q_seq_starts.reserve(static_cast<size_t>(batch_size));
  layout.realtime_unmatched_lens.reserve(static_cast<size_t>(batch_size));
  layout.target_seq_starts.reserve(static_cast<size_t>(batch_size));

  for (int64_t i = 0; i < batch_size; ++i) {
    const int64_t q_len = layout.q_seq_lens[i];
    const int64_t kv_len = kv_seq_lens_host[i];
    const int64_t h = history_lens_host[i];
    const int64_t c = context_lens_host[i];
    const int64_t r = realtime_lens_host[i];
    const int64_t t = layout.target_lens[i];
    const int64_t matched = layout.matched_prefix_lens[i];
    CHECK_GT(matched, 0)
        << "Partial batched fused path expects matched_prefix > 0";
    CHECK_GE(matched, h + c) << "Batched fused partial path currently keeps "
                                "only partial_real_time_match";
    CHECK_LT(matched, h + c + r);
    CHECK_EQ(q_len, kv_len);
    const int64_t realtime_matched = matched - h - c;
    const int64_t realtime_unmatched = r - realtime_matched;
    CHECK_GT(realtime_unmatched, 0);
    CHECK_EQ(q_len, realtime_unmatched + t);

    layout.q_seq_starts.push_back(layout.total_q);
    layout.segment_offsets.push_back(0);
    layout.segment_offsets.push_back(h);
    layout.segment_offsets.push_back(h + c);
    layout.segment_offsets.push_back(h + c + r);
    layout.segment_offsets.push_back(h + c + r + t);
    layout.realtime_unmatched_lens.push_back(realtime_unmatched);
    layout.target_seq_starts.push_back(layout.total_target);
    layout.total_q += q_len;
    layout.total_target += t;
    layout.max_q = std::max(layout.max_q, q_len);
    layout.max_target = std::max(layout.max_target, t);
    layout.max_matched_prefix = std::max(layout.max_matched_prefix, matched);
    layout.max_realtime_unmatched =
        std::max(layout.max_realtime_unmatched, realtime_unmatched);
  }

  layout.q_seq_starts_i32 =
      make_int32_tensor_on_device(layout.q_seq_starts, device);
  layout.matched_prefix_lens_i32 =
      make_int32_tensor_on_device(layout.matched_prefix_lens, device);
  layout.realtime_unmatched_lens_i32 =
      make_int32_tensor_on_device(layout.realtime_unmatched_lens, device);
  layout.target_lens_i32 =
      make_int32_tensor_on_device(layout.target_lens, device);
  layout.target_seq_starts_i32 =
      make_int32_tensor_on_device(layout.target_seq_starts, device);
  layout.segment_offsets_i32 =
      make_int32_tensor_on_device(layout.segment_offsets, device)
          .view({batch_size,
                 static_cast<int64_t>(layout.segment_rules.size()) + 1})
          .contiguous();
  layout.segment_rules_i32 =
      make_int32_tensor_on_device(layout.segment_rules, device);
  return layout;
}

const MTGRFourSegmentPartialBatchLayout&
get_cached_four_segment_partial_batch_layout(
    const torch::Tensor& q_seq_lens,
    const torch::Tensor& kv_seq_lens,
    const torch::Tensor& history_lens,
    const torch::Tensor& context_lens,
    const torch::Tensor& realtime_lens,
    const torch::Tensor& target_lens,
    const torch::Tensor& matched_prefix_lens,
    const torch::Tensor& block_table,
    const torch::Device& device) {
  static thread_local std::unordered_map<
      MTGRFourSegmentPartialBatchLayoutCacheKey,
      MTGRFourSegmentPartialBatchLayout,
      MTGRFourSegmentPartialBatchLayoutCacheKeyHash>
      cache;
  const MTGRFourSegmentPartialBatchLayoutCacheKey key{
      .q_seq_lens_ptr = q_seq_lens.data_ptr(),
      .kv_seq_lens_ptr = kv_seq_lens.data_ptr(),
      .history_lens_ptr = history_lens.data_ptr(),
      .context_lens_ptr = context_lens.data_ptr(),
      .realtime_lens_ptr = realtime_lens.data_ptr(),
      .target_lens_ptr = target_lens.data_ptr(),
      .matched_prefix_lens_ptr = matched_prefix_lens.data_ptr(),
      .block_table_ptr = block_table.data_ptr(),
      .q_seq_lens_hash = hash_cpu_int64_tensor_values(q_seq_lens),
      .kv_seq_lens_hash = hash_cpu_int64_tensor_values(kv_seq_lens),
      .history_lens_hash = hash_cpu_int64_tensor_values(history_lens),
      .context_lens_hash = hash_cpu_int64_tensor_values(context_lens),
      .realtime_lens_hash = hash_cpu_int64_tensor_values(realtime_lens),
      .target_lens_hash = hash_cpu_int64_tensor_values(target_lens),
      .matched_prefix_lens_hash =
          hash_cpu_int64_tensor_values(matched_prefix_lens),
      .device_idx = device.index(),
  };
  auto it = cache.find(key);
  if (it == cache.end()) {
    it = cache
             .emplace(key,
                      build_four_segment_partial_batch_layout_uncached(
                          q_seq_lens,
                          kv_seq_lens,
                          history_lens,
                          context_lens,
                          realtime_lens,
                          target_lens,
                          matched_prefix_lens,
                          block_table,
                          device))
             .first;
  }
  return it->second;
}

torch::Tensor gather_cached_prefix(const torch::Tensor& cache,
                                   const torch::Tensor& block_table_row,
                                   int64_t block_size,
                                   int64_t prefix_len) {
  CHECK_GT(prefix_len, 0);
  CHECK_EQ(cache.dim(), 4);
  const auto device = cache.device();
  auto token_idx = torch::arange(
      prefix_len, torch::TensorOptions().dtype(torch::kInt64).device(device));
  auto logical_blocks = torch::div(token_idx, block_size, "trunc");
  auto offsets = token_idx - logical_blocks * block_size;
  auto physical_blocks = block_table_row.to(torch::kInt64)
                             .contiguous()
                             .index_select(0, logical_blocks);
  auto slots = physical_blocks * block_size + offsets;
  auto flat =
      cache.view({cache.size(0) * block_size, cache.size(2), cache.size(3)});
  return flat.index_select(0, slots).contiguous();
}

void scatter_to_cache(const torch::Tensor& key_snd,
                      const torch::Tensor& value_snd,
                      torch::Tensor& key_cache,
                      torch::Tensor& value_cache,
                      const torch::Tensor& slot_mapping_i32) {
  if (key_snd.numel() == 0) {
    return;
  }
  CHECK_EQ(key_snd.dim(), 3);
  CHECK_EQ(value_snd.dim(), 3);
  CHECK_EQ(key_snd.sizes(), value_snd.sizes());
  CHECK_EQ(slot_mapping_i32.scalar_type(), torch::kInt32);
  CHECK_EQ(slot_mapping_i32.dim(), 1);
  CHECK_EQ(slot_mapping_i32.size(0), key_snd.size(0));
  reshape_paged_cache(slot_mapping_i32.contiguous(),
                      key_snd.contiguous(),
                      value_snd.contiguous(),
                      key_cache,
                      value_cache);
}

std::vector<int32_t> make_block_table_host(int64_t block_count) {
  std::vector<int32_t> table(static_cast<size_t>(block_count));
  for (int64_t i = 0; i < block_count; ++i) {
    table[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  }
  return table;
}

std::vector<int64_t> build_slot_mapping_host(
    const std::vector<int32_t>& block_table,
    int64_t block_size,
    int64_t start_token_idx,
    int64_t token_count) {
  std::vector<int64_t> slots;
  slots.reserve(static_cast<size_t>(token_count));
  for (int64_t token_idx = start_token_idx;
       token_idx < start_token_idx + token_count;
       ++token_idx) {
    const int64_t logical_block = token_idx / block_size;
    const int64_t offset = token_idx % block_size;
    const int64_t physical_block =
        block_table.at(static_cast<size_t>(logical_block));
    slots.push_back(physical_block * block_size + offset);
  }
  return slots;
}

torch::Tensor run_one_stage_custom_mask_snd(const torch::Tensor& query_snd,
                                            const torch::Tensor& key_snd,
                                            const torch::Tensor& value_snd,
                                            const torch::Tensor& packed_mask,
                                            double sm_scale,
                                            MTGRAttentionTestMetrics* metrics) {
  StageTimeline stage;
  stage.name = "one_stage";
  create_stage_events_if_needed(&stage);
  auto result = run_fa_segment(query_snd,
                               key_snd,
                               value_snd,
                               sm_scale,
                               /*causal=*/false,
                               /*need_lse=*/false,
                               &stage,
                               packed_mask);
  CHECK_EQ(cudaEventSynchronize(stage.ev_end), cudaSuccess);
  float fia_ms = 0.0f;
  CHECK_EQ(cudaEventElapsedTime(&fia_ms, stage.ev_start, stage.ev_end),
           cudaSuccess);
  if (metrics != nullptr) {
    metrics->workspace_ms = result.get_ws_ms;
    metrics->fia_ms = static_cast<double>(fia_ms);
    metrics->device_total_ms = metrics->mask_build_ms + metrics->fia_ms;
    metrics->stages = {MTGRStageMetric{
        .name = "one_stage",
        .workspace_ms = result.get_ws_ms,
        .exec_ms = metrics->fia_ms,
        .host_submit_ms = stage.host_ms,
    }};
  }
  destroy_stage_events_if_needed(&stage);
  return result.out_snd;
}

torch::Tensor run_one_stage_no_match(const torch::Tensor& query_snd,
                                     const torch::Tensor& key_snd,
                                     const torch::Tensor& value_snd,
                                     int64_t history_len,
                                     int64_t context_len,
                                     int64_t realtime_len,
                                     int64_t target_len,
                                     double sm_scale,
                                     MTGRAttentionTestMetrics* metrics) {
  const auto device = query_snd.device();
  CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  auto t0 = std::chrono::steady_clock::now();
  auto packed_mask = build_one_stage_packed_mask_reusing_buffer(
      device, history_len, context_len, realtime_len, target_len);
  CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  if (metrics != nullptr) {
    metrics->mask_build_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - t0)
                                 .count();
    metrics->h2d_ms = 0.0;
  }
  return run_one_stage_custom_mask_snd(
      query_snd, key_snd, value_snd, packed_mask, sm_scale, metrics);
}

torch::Tensor run_one_stage_partial_rt(const torch::Tensor& query_snd,
                                       const torch::Tensor& key_snd,
                                       const torch::Tensor& value_snd,
                                       const torch::Tensor& key_cache,
                                       const torch::Tensor& value_cache,
                                       const torch::Tensor& block_table_row,
                                       int64_t block_size,
                                       int64_t matched_prefix_len,
                                       int64_t realtime_unmatched_len,
                                       int64_t target_len,
                                       double sm_scale,
                                       MTGRAttentionTestMetrics* metrics) {
  CHECK_EQ(query_snd.size(0), realtime_unmatched_len + target_len);
  auto prefix_key = gather_cached_prefix(
      key_cache, block_table_row, block_size, matched_prefix_len);
  auto prefix_value = gather_cached_prefix(
      value_cache, block_table_row, block_size, matched_prefix_len);
  auto full_key = torch::cat({prefix_key, key_snd}, 0).contiguous();
  auto full_value = torch::cat({prefix_value, value_snd}, 0).contiguous();

  const auto device = query_snd.device();
  CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  auto t0 = std::chrono::steady_clock::now();
  auto packed_mask = build_partial_rt_one_stage_packed_mask_reusing_buffer(
      device, matched_prefix_len, realtime_unmatched_len, target_len);
  CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  if (metrics != nullptr) {
    metrics->mask_build_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - t0)
                                 .count();
    metrics->h2d_ms = 0.0;
  }
  return run_one_stage_custom_mask_snd(
      query_snd, full_key, full_value, packed_mask, sm_scale, metrics);
}

torch::Tensor run_multi_no_match_preplanned(const torch::Tensor& query,
                                            const torch::Tensor& key,
                                            const torch::Tensor& value,
                                            int64_t history_len,
                                            int64_t context_len,
                                            int64_t realtime_len,
                                            int64_t target_len,
                                            double sm_scale,
                                            MTGRAttentionTestMetrics* metrics) {
  auto timeline = build_no_match_multi_timeline();
  prepare_timeline_events(&timeline);
  auto wall_start = std::chrono::steady_clock::now();

  constexpr size_t kRtTgtIdx = 0;
  constexpr size_t kFusedUpdateIdx = 1;
  constexpr size_t kHistIdx = 2;
  constexpr size_t kCtxIdx = 3;
  constexpr int64_t kRtTgtWorkspaceSlot = 1;
  constexpr int64_t kHistWorkspaceSlot = 2;
  constexpr int64_t kCtxWorkspaceSlot = 3;

  auto output = torch::empty_like(query);
  auto& rt_tgt_ws =
      get_workspace_buffers_for_slot(query.device(), kRtTgtWorkspaceSlot);
  auto& hist_ws =
      get_workspace_buffers_for_slot(query.device(), kHistWorkspaceSlot);
  auto& ctx_ws =
      get_workspace_buffers_for_slot(query.device(), kCtxWorkspaceSlot);
  auto rt_tgt_mask = build_rt_tgt_trapezoid_packed_mask_reusing_buffer(
      query.device(), history_len + context_len, realtime_len, target_len);

  auto rt_tgt_output =
      output.narrow(0, history_len + context_len, realtime_len + target_len);
  auto rt_tgt_prepared = prepare_fa_segment(
      query.narrow(0, history_len + context_len, realtime_len + target_len),
      key.narrow(0, 0, history_len + context_len + realtime_len),
      value.narrow(0, 0, history_len + context_len + realtime_len),
      /*causal=*/false,
      /*need_lse=*/true,
      &timeline[kRtTgtIdx],
      rt_tgt_mask,
      rt_tgt_output,
      &rt_tgt_ws);

  auto hist_prepared = prepare_fa_segment(query.narrow(0, 0, history_len),
                                          key.narrow(0, 0, history_len),
                                          value.narrow(0, 0, history_len),
                                          /*causal=*/true,
                                          /*need_lse=*/false,
                                          &timeline[kHistIdx],
                                          std::nullopt,
                                          output.narrow(0, 0, history_len),
                                          &hist_ws);

  auto ctx_prepared =
      prepare_fa_segment(query.narrow(0, history_len, context_len),
                         key.narrow(0, 0, history_len + context_len),
                         value.narrow(0, 0, history_len + context_len),
                         /*causal=*/false,
                         /*need_lse=*/false,
                         &timeline[kCtxIdx],
                         std::nullopt,
                         output.narrow(0, history_len, context_len),
                         &ctx_ws);

  auto rt_tgt = launch_prepared_fa_segment(
      &rt_tgt_prepared, sm_scale, &timeline[kRtTgtIdx]);

  record_stage_begin_if_needed(&timeline[kFusedUpdateIdx], query.device());
  auto target_query =
      query.narrow(0, history_len + context_len + realtime_len, target_len);
  auto target_key =
      key.narrow(0, history_len + context_len + realtime_len, target_len);
  auto target_value =
      value.narrow(0, history_len + context_len + realtime_len, target_len);
  auto target_output =
      output.narrow(0, history_len + context_len + realtime_len, target_len);
  merge_target_diag_attention_into(
      rt_tgt.out_snd.narrow(0, realtime_len, target_len),
      rt_tgt.lse_sh1.narrow(0, realtime_len, target_len),
      target_query,
      target_key,
      target_value,
      sm_scale,
      target_output);
  record_stage_end_if_needed(&timeline[kFusedUpdateIdx], query.device());

  (void)launch_prepared_fa_segment(
      &hist_prepared, sm_scale, &timeline[kHistIdx]);
  (void)launch_prepared_fa_segment(&ctx_prepared, sm_scale, &timeline[kCtxIdx]);

  auto summary = collect_timeline_summary(&timeline);
  auto wall_end = std::chrono::steady_clock::now();
  if (metrics != nullptr) {
    metrics->device_total_ms = summary.total_ms;
    metrics->wall_total_ms =
        std::chrono::duration<double, std::milli>(wall_end - wall_start)
            .count();
    metrics->workspace_ms = 0.0;
    for (double v : summary.stage_getws_ms) {
      metrics->workspace_ms += v;
    }
    fill_stage_metrics(timeline, summary, metrics);
  }
  release_timeline_events(&timeline);
  return output;
}

torch::Tensor run_multi_partial_rt(const torch::Tensor& query,
                                   const torch::Tensor& key,
                                   const torch::Tensor& value,
                                   const torch::Tensor& key_cache,
                                   const torch::Tensor& value_cache,
                                   const torch::Tensor& block_table_row,
                                   int64_t block_size,
                                   int64_t matched_prefix_len,
                                   int64_t realtime_unmatched_len,
                                   int64_t target_len,
                                   double sm_scale,
                                   MTGRAttentionTestMetrics* metrics) {
  CHECK_EQ(query.size(0), realtime_unmatched_len + target_len);
  auto timeline = build_partial_rt_multi_timeline();
  prepare_timeline_events(&timeline);
  auto wall_start = std::chrono::steady_clock::now();

  constexpr size_t kRtTgtIdx = 0;
  constexpr size_t kFusedUpdateIdx = 1;
  constexpr int64_t kRtTgtWorkspaceSlot = 4;
  auto& rt_tgt_ws =
      get_workspace_buffers_for_slot(query.device(), kRtTgtWorkspaceSlot);

  auto prefix_key = gather_cached_prefix(
      key_cache, block_table_row, block_size, matched_prefix_len);
  auto prefix_value = gather_cached_prefix(
      value_cache, block_table_row, block_size, matched_prefix_len);
  auto rt_key = key.narrow(0, 0, realtime_unmatched_len);
  auto rt_value = value.narrow(0, 0, realtime_unmatched_len);
  auto hcr_key = torch::cat({prefix_key, rt_key}, 0).contiguous();
  auto hcr_value = torch::cat({prefix_value, rt_value}, 0).contiguous();
  auto rt_tgt_mask = build_rt_tgt_trapezoid_packed_mask_reusing_buffer(
      query.device(), matched_prefix_len, realtime_unmatched_len, target_len);

  auto output = torch::empty_like(query);
  auto rt_tgt_prepared = prepare_fa_segment(query,
                                            hcr_key,
                                            hcr_value,
                                            /*causal=*/false,
                                            /*need_lse=*/true,
                                            &timeline[kRtTgtIdx],
                                            rt_tgt_mask,
                                            output,
                                            &rt_tgt_ws);
  auto rt_tgt = launch_prepared_fa_segment(
      &rt_tgt_prepared, sm_scale, &timeline[kRtTgtIdx]);

  record_stage_begin_if_needed(&timeline[kFusedUpdateIdx], query.device());
  auto target_query = query.narrow(0, realtime_unmatched_len, target_len);
  auto target_key = key.narrow(0, realtime_unmatched_len, target_len);
  auto target_value = value.narrow(0, realtime_unmatched_len, target_len);
  auto target_output = output.narrow(0, realtime_unmatched_len, target_len);
  merge_target_diag_attention_into(
      rt_tgt.out_snd.narrow(0, realtime_unmatched_len, target_len),
      rt_tgt.lse_sh1.narrow(0, realtime_unmatched_len, target_len),
      target_query,
      target_key,
      target_value,
      sm_scale,
      target_output);
  record_stage_end_if_needed(&timeline[kFusedUpdateIdx], query.device());

  auto summary = collect_timeline_summary(&timeline);
  auto wall_end = std::chrono::steady_clock::now();
  if (metrics != nullptr) {
    metrics->device_total_ms = summary.total_ms;
    metrics->wall_total_ms =
        std::chrono::duration<double, std::milli>(wall_end - wall_start)
            .count();
    metrics->workspace_ms = 0.0;
    for (double v : summary.stage_getws_ms) {
      metrics->workspace_ms += v;
    }
    fill_stage_metrics(timeline, summary, metrics);
  }
  release_timeline_events(&timeline);
  return output;
}

torch::Tensor run_fused_no_match(const torch::Tensor& query,
                                 const torch::Tensor& key,
                                 const torch::Tensor& value,
                                 int64_t history_len,
                                 int64_t context_len,
                                 int64_t realtime_len,
                                 int64_t target_len,
                                 double sm_scale,
                                 MTGRAttentionTestMetrics* metrics) {
  auto timeline = build_fused_no_match_timeline();
  prepare_timeline_events(&timeline);
  auto wall_start = std::chrono::steady_clock::now();

  constexpr size_t kFusedIdx = 0;
  constexpr size_t kFusedUpdateIdx = 1;
  auto output = torch::empty_like(query);
  auto target_hcr_lse = torch::empty(
      {target_len, query.size(1), 1},
      torch::TensorOptions().dtype(torch::kFloat32).device(query.device()));

  record_stage_begin_if_needed(&timeline[kFusedIdx], query.device());
  mtgr_fused_no_match_attention_cuda(query,
                                     key,
                                     value,
                                     history_len,
                                     context_len,
                                     realtime_len,
                                     target_len,
                                     sm_scale,
                                     output,
                                     target_hcr_lse);
  record_stage_end_if_needed(&timeline[kFusedIdx], query.device());

  record_stage_begin_if_needed(&timeline[kFusedUpdateIdx], query.device());
  auto target_begin = history_len + context_len + realtime_len;
  auto target_query = query.narrow(0, target_begin, target_len);
  auto target_key = key.narrow(0, target_begin, target_len);
  auto target_value = value.narrow(0, target_begin, target_len);
  auto target_output = output.narrow(0, target_begin, target_len);
  merge_target_diag_attention_into(output.narrow(0, target_begin, target_len),
                                   target_hcr_lse,
                                   target_query,
                                   target_key,
                                   target_value,
                                   sm_scale,
                                   target_output);
  record_stage_end_if_needed(&timeline[kFusedUpdateIdx], query.device());

  auto summary = collect_timeline_summary(&timeline);
  auto wall_end = std::chrono::steady_clock::now();
  if (metrics != nullptr) {
    metrics->device_total_ms = summary.total_ms;
    metrics->wall_total_ms =
        std::chrono::duration<double, std::milli>(wall_end - wall_start)
            .count();
    metrics->workspace_ms = 0.0;
    fill_stage_metrics(timeline, summary, metrics);
  }
  release_timeline_events(&timeline);
  return output;
}

torch::Tensor run_fused_no_match_batched(const torch::Tensor& query,
                                         const torch::Tensor& key,
                                         const torch::Tensor& value,
                                         const MTGRNoMatchBatchLayout& layout,
                                         double sm_scale,
                                         MTGRAttentionTestMetrics* metrics) {
  CHECK_EQ(query.dim(), 3);
  CHECK_EQ(key.dim(), 3);
  CHECK_EQ(value.dim(), 3);
  CHECK_EQ(query.sizes(), key.sizes());
  CHECK_EQ(query.sizes(), value.sizes());
  CHECK_EQ(query.size(0), layout.total_q);

  auto timeline = build_ragged_segment_timeline();
  prepare_timeline_events(&timeline);
  auto wall_start = std::chrono::steady_clock::now();

  constexpr size_t kUnifiedIdx = 0;
  auto output = torch::empty_like(query);
  auto matched_prefix_lens_i32 = torch::zeros(
      {layout.batch_size},
      torch::TensorOptions().dtype(torch::kInt32).device(query.device()));

  record_stage_begin_if_needed(&timeline[kUnifiedIdx], query.device());
  mtgr_ragged_segment_attention_hopper_unified_research_cuda(
      query,
      key,
      value,
      layout.segment_offsets_i32,
      layout.segment_rules_i32,
      layout.q_seq_starts_i32,
      matched_prefix_lens_i32,
      /*match_mode=*/0,
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor(),
      /*block_size=*/1,
      layout.max_q,
      sm_scale,
      output);
  record_stage_end_if_needed(&timeline[kUnifiedIdx], query.device());

  auto summary = collect_timeline_summary(&timeline);
  auto wall_end = std::chrono::steady_clock::now();
  if (metrics != nullptr) {
    metrics->device_total_ms = summary.total_ms;
    metrics->wall_total_ms =
        std::chrono::duration<double, std::milli>(wall_end - wall_start)
            .count();
    metrics->workspace_ms = 0.0;
    fill_stage_metrics(timeline, summary, metrics);
  }
  release_timeline_events(&timeline);
  return output;
}

torch::Tensor run_fused_segmented_no_match_batched(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const MTGRSegmentedNoMatchBatchLayout& layout,
    double sm_scale,
    MTGRAttentionTestMetrics* metrics) {
  CHECK_EQ(query.dim(), 3);
  CHECK_EQ(key.dim(), 3);
  CHECK_EQ(value.dim(), 3);
  CHECK_EQ(query.sizes(), key.sizes());
  CHECK_EQ(query.sizes(), value.sizes());
  CHECK_EQ(query.size(0), layout.total_q);
  CHECK_EQ(layout.segment_offsets_i32.size(0), layout.batch_size);
  CHECK_EQ(layout.segment_offsets_i32.size(1), layout.num_segments + 1);

  auto timeline = build_fused_no_match_timeline();
  prepare_timeline_events(&timeline);
  auto wall_start = std::chrono::steady_clock::now();

  constexpr size_t kFusedIdx = 0;
  constexpr size_t kFusedUpdateIdx = 1;
  auto output = torch::empty_like(query);
  auto target_hcr_lse = torch::empty(
      {layout.total_target, query.size(1), 1},
      torch::TensorOptions().dtype(torch::kFloat32).device(query.device()));
  const auto* props = at::cuda::getCurrentDeviceProperties();
  const bool use_batch4_fast_path =
      layout.batch_size > 1 && layout.batch_size <= 4 && props->major >= 8 &&
      (query.size(2) == 64 || query.size(2) == 128);

  record_stage_begin_if_needed(&timeline[kFusedIdx], query.device());
  mtgr_fused_segmented_no_match_attention_batched_cuda(
      query,
      key,
      value,
      layout.segment_offsets_i32,
      layout.segment_rules_i32,
      layout.row_to_batch_i32,
      layout.target_seq_starts_i32,
      layout.target_lens_i32,
      sm_scale,
      output,
      target_hcr_lse);
  record_stage_end_if_needed(&timeline[kFusedIdx], query.device());

  record_stage_begin_if_needed(&timeline[kFusedUpdateIdx], query.device());
  if (layout.batch_size == 1) {
    const int64_t target_len = layout.target_lens[0];
    const int64_t target_begin =
        layout.segment_offsets[layout.num_segments] - target_len;
    auto target_query = query.narrow(0, target_begin, target_len);
    auto target_key = key.narrow(0, target_begin, target_len);
    auto target_value = value.narrow(0, target_begin, target_len);
    auto target_output = output.narrow(0, target_begin, target_len);
    merge_target_diag_attention_into(target_output,
                                     target_hcr_lse,
                                     target_query,
                                     target_key,
                                     target_value,
                                     sm_scale,
                                     target_output);
  } else if (use_batch4_fast_path) {
    merge_target_diag_attention_partial_batched_cuda(
        output,
        target_hcr_lse,
        query,
        key,
        value,
        layout.q_seq_starts_i32,
        layout.target_offsets_i32,
        layout.target_lens_i32,
        layout.target_seq_starts_i32,
        layout.batch_size,
        layout.max_target,
        sm_scale,
        output);
  } else {
    auto target_query =
        query.index_select(0, layout.target_row_indices_i64).contiguous();
    auto target_key =
        key.index_select(0, layout.target_row_indices_i64).contiguous();
    auto target_value =
        value.index_select(0, layout.target_row_indices_i64).contiguous();
    auto target_hcr_out =
        output.index_select(0, layout.target_row_indices_i64).contiguous();
    auto merged_target = torch::empty_like(target_query);
    merge_target_diag_attention_into(target_hcr_out,
                                     target_hcr_lse,
                                     target_query,
                                     target_key,
                                     target_value,
                                     sm_scale,
                                     merged_target);
    output.index_copy_(0, layout.target_row_indices_i64, merged_target);
  }
  record_stage_end_if_needed(&timeline[kFusedUpdateIdx], query.device());

  auto summary = collect_timeline_summary(&timeline);
  auto wall_end = std::chrono::steady_clock::now();
  if (metrics != nullptr) {
    metrics->device_total_ms = summary.total_ms;
    metrics->wall_total_ms =
        std::chrono::duration<double, std::milli>(wall_end - wall_start)
            .count();
    metrics->workspace_ms = 0.0;
    fill_stage_metrics(timeline, summary, metrics);
  }
  release_timeline_events(&timeline);
  return output;
}

torch::Tensor run_ragged_segment_attention_batched(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const MTGRRaggedSegmentBatchLayout& layout,
    double sm_scale,
    MTGRAttentionTestMetrics* metrics) {
  CHECK_EQ(query.dim(), 3);
  CHECK_EQ(key.dim(), 3);
  CHECK_EQ(value.dim(), 3);
  CHECK_EQ(query.sizes(), key.sizes());
  CHECK_EQ(query.sizes(), value.sizes());
  CHECK_EQ(query.size(0), layout.total_q);
  CHECK_EQ(layout.segment_offsets_i32.size(0), layout.batch_size);
  CHECK_EQ(layout.segment_offsets_i32.size(1), layout.num_segments + 1);

  auto timeline = build_ragged_segment_timeline();
  prepare_timeline_events(&timeline);
  auto wall_start = std::chrono::steady_clock::now();

  constexpr size_t kRaggedIdx = 0;
  auto output = torch::empty_like(query);
  record_stage_begin_if_needed(&timeline[kRaggedIdx], query.device());
  mtgr_ragged_segment_attention_batched_cuda(query,
                                             key,
                                             value,
                                             layout.segment_offsets_i32,
                                             layout.segment_rules_i32,
                                             layout.row_to_batch_i32,
                                             layout.max_q,
                                             sm_scale,
                                             output);
  record_stage_end_if_needed(&timeline[kRaggedIdx], query.device());

  auto summary = collect_timeline_summary(&timeline);
  auto wall_end = std::chrono::steady_clock::now();
  if (metrics != nullptr) {
    metrics->fia_ms = summary.total_ms;
    metrics->device_total_ms = summary.total_ms;
    metrics->wall_total_ms =
        std::chrono::duration<double, std::milli>(wall_end - wall_start)
            .count();
    metrics->workspace_ms = 0.0;
    fill_stage_metrics(timeline, summary, metrics);
  }
  release_timeline_events(&timeline);
  return output;
}

torch::Tensor run_four_segment_partial_rt_batched(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    const MTGRFourSegmentPartialBatchLayout& layout,
    int64_t block_size,
    double sm_scale,
    MTGRAttentionTestMetrics* metrics) {
  CHECK_EQ(query.dim(), 3);
  CHECK_EQ(key.dim(), 3);
  CHECK_EQ(value.dim(), 3);
  CHECK_EQ(query.sizes(), key.sizes());
  CHECK_EQ(query.sizes(), value.sizes());
  CHECK_EQ(query.size(0), layout.total_q);
  CHECK_GT(block_size, 0);

  auto timeline = build_ragged_segment_timeline();
  prepare_timeline_events(&timeline);
  auto wall_start = std::chrono::steady_clock::now();

  constexpr size_t kUnifiedIdx = 0;
  auto output = torch::empty_like(query);

  record_stage_begin_if_needed(&timeline[kUnifiedIdx], query.device());
  mtgr_ragged_segment_attention_hopper_unified_research_cuda(
      query,
      key,
      value,
      layout.segment_offsets_i32,
      layout.segment_rules_i32,
      layout.q_seq_starts_i32,
      layout.matched_prefix_lens_i32,
      /*match_mode=*/1,
      key_cache,
      value_cache,
      layout.block_table_i32,
      block_size,
      layout.max_q,
      sm_scale,
      output);
  record_stage_end_if_needed(&timeline[kUnifiedIdx], query.device());

  auto summary = collect_timeline_summary(&timeline);
  auto wall_end = std::chrono::steady_clock::now();
  if (metrics != nullptr) {
    metrics->device_total_ms = summary.total_ms;
    metrics->wall_total_ms =
        std::chrono::duration<double, std::milli>(wall_end - wall_start)
            .count();
    metrics->workspace_ms = 0.0;
    fill_stage_metrics(timeline, summary, metrics);
  }
  release_timeline_events(&timeline);
  return output;
}

torch::Tensor run_fused_partial_rt(const torch::Tensor& query,
                                   const torch::Tensor& key,
                                   const torch::Tensor& value,
                                   torch::Tensor& key_cache,
                                   torch::Tensor& value_cache,
                                   const torch::Tensor& block_table_row,
                                   const torch::Tensor& slot_mapping,
                                   int64_t block_size,
                                   int64_t matched_prefix_len,
                                   int64_t realtime_unmatched_len,
                                   int64_t target_len,
                                   double sm_scale,
                                   MTGRAttentionTestMetrics* metrics) {
  CHECK_EQ(query.dim(), 3);
  CHECK_EQ(key.dim(), 3);
  CHECK_EQ(value.dim(), 3);
  CHECK_EQ(query.size(0), realtime_unmatched_len + target_len);
  CHECK_EQ(key.sizes(), value.sizes());
  CHECK_EQ(query.sizes(), key.sizes());
  CHECK_EQ(block_table_row.dim(), 1);
  CHECK_EQ(slot_mapping.dim(), 1);
  CHECK_GE(slot_mapping.size(0), query.size(0));

  auto rt_key = key.narrow(0, 0, realtime_unmatched_len).contiguous();
  auto rt_value = value.narrow(0, 0, realtime_unmatched_len).contiguous();

  auto timeline = build_fused_partial_rt_timeline();
  prepare_timeline_events(&timeline);
  auto wall_start = std::chrono::steady_clock::now();

  constexpr size_t kFusedIdx = 0;
  constexpr size_t kFusedUpdateIdx = 1;

  auto output = torch::empty_like(query);
  auto target_hcr_lse = torch::empty(
      {target_len, query.size(1), 1},
      torch::TensorOptions().dtype(torch::kFloat32).device(query.device()));
  (void)slot_mapping;

  record_stage_begin_if_needed(&timeline[kFusedIdx], query.device());
  mtgr_fused_partial_rt_attention_cuda(query,
                                       rt_key,
                                       rt_value,
                                       key_cache,
                                       value_cache,
                                       block_table_row,
                                       block_size,
                                       matched_prefix_len,
                                       realtime_unmatched_len,
                                       target_len,
                                       sm_scale,
                                       output,
                                       target_hcr_lse);
  record_stage_end_if_needed(&timeline[kFusedIdx], query.device());

  record_stage_begin_if_needed(&timeline[kFusedUpdateIdx], query.device());
  auto target_query = query.narrow(0, realtime_unmatched_len, target_len);
  auto target_key = key.narrow(0, realtime_unmatched_len, target_len);
  auto target_value = value.narrow(0, realtime_unmatched_len, target_len);
  auto target_output = output.narrow(0, realtime_unmatched_len, target_len);
  merge_target_diag_attention_into(
      output.narrow(0, realtime_unmatched_len, target_len),
      target_hcr_lse,
      target_query,
      target_key,
      target_value,
      sm_scale,
      target_output);
  record_stage_end_if_needed(&timeline[kFusedUpdateIdx], query.device());

  auto summary = collect_timeline_summary(&timeline);
  auto wall_end = std::chrono::steady_clock::now();
  if (metrics != nullptr) {
    metrics->device_total_ms = summary.total_ms;
    metrics->wall_total_ms =
        std::chrono::duration<double, std::milli>(wall_end - wall_start)
            .count();
    metrics->workspace_ms = 0.0;
    fill_stage_metrics(timeline, summary, metrics);
  }
  release_timeline_events(&timeline);
  return output;
}

}  // namespace

torch::Tensor run_four_segment_no_match_batched_for_test(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    const std::vector<std::vector<int64_t>>& segment_lens,
    const std::vector<int64_t>& segment_rules,
    double sm_scale,
    MTGRAttentionTestMetrics* metrics) {
  CHECK(query_snd.defined());
  CHECK(key_snd.defined());
  CHECK(value_snd.defined());
  CHECK(query_snd.is_cuda());
  CHECK(key_snd.is_cuda());
  CHECK(value_snd.is_cuda());
  CHECK_EQ(query_snd.dim(), 3);
  CHECK_EQ(query_snd.sizes(), key_snd.sizes());
  CHECK_EQ(query_snd.sizes(), value_snd.sizes());
  CHECK_EQ(query_snd.scalar_type(), torch::kFloat16);
  CHECK_EQ(key_snd.scalar_type(), torch::kFloat16);
  CHECK_EQ(value_snd.scalar_type(), torch::kFloat16);
  auto query = query_snd.contiguous();
  auto key = key_snd.contiguous();
  auto value = value_snd.contiguous();
  auto layout = build_segmented_no_match_batch_layout(
      segment_lens, segment_rules, query.device());
  CHECK_EQ(query.size(0), layout.total_q);
  return run_fused_segmented_no_match_batched(
      query, key, value, layout, sm_scale, metrics);
}

torch::Tensor run_stable_ragged_segment_attention_batched_for_test(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    const std::vector<std::vector<int64_t>>& segment_lens,
    const std::vector<int64_t>& segment_rules,
    double sm_scale,
    MTGRAttentionTestMetrics* metrics) {
  CHECK(query_snd.defined());
  CHECK(key_snd.defined());
  CHECK(value_snd.defined());
  CHECK(query_snd.is_cuda());
  CHECK(key_snd.is_cuda());
  CHECK(value_snd.is_cuda());
  CHECK_EQ(query_snd.dim(), 3);
  CHECK_EQ(query_snd.sizes(), key_snd.sizes());
  CHECK_EQ(query_snd.sizes(), value_snd.sizes());
  CHECK_EQ(query_snd.scalar_type(), torch::kFloat16);
  CHECK_EQ(key_snd.scalar_type(), torch::kFloat16);
  CHECK_EQ(value_snd.scalar_type(), torch::kFloat16);
  auto query = query_snd.contiguous();
  auto key = key_snd.contiguous();
  auto value = value_snd.contiguous();
  auto layout = build_ragged_segment_batch_layout(
      segment_lens, segment_rules, query.device());
  CHECK_EQ(query.size(0), layout.total_q);
  return run_ragged_segment_attention_batched(
      query, key, value, layout, sm_scale, metrics);
}

MTGRAttentionImplTest::MTGRAttentionImplTest(int64_t num_heads,
                                             int64_t head_size,
                                             float scale,
                                             int64_t num_kv_heads,
                                             MTGRAttentionTestBackend backend)
    : num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads),
      backend_(backend) {}

std::tuple<torch::Tensor, std::optional<torch::Tensor>>
MTGRAttentionImplTest::forward(
    const xllm::layer::AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    xllm::KVCache& kv_cache) {
  CHECK_EQ(query.dim(), 2);
  CHECK_EQ(key.dim(), 2);
  CHECK_EQ(value.dim(), 2);
  CHECK_EQ(query.size(1), num_heads_ * head_size_);
  CHECK_EQ(key.size(1), num_kv_heads_ * head_size_);
  CHECK_EQ(value.size(1), num_kv_heads_ * head_size_);
  CHECK_GT(scale_, 0.0f);

  last_metrics_ = MTGRAttentionTestMetrics{};
  std::optional<torch::Tensor> output_lse = std::nullopt;
  if (attn_metadata.is_dummy) {
    return {torch::empty_like(query), output_lse};
  }

  const auto& q_seq_lens = attn_metadata.q_seq_lens;
  const auto& kv_seq_lens = attn_metadata.kv_seq_lens;
  const auto& history_lens = attn_metadata.genrec_history_lens;
  const auto& context_lens = attn_metadata.genrec_context_lens;
  const auto& realtime_lens = attn_metadata.genrec_real_time_lens;
  const auto& target_lens = attn_metadata.genrec_target_lens;
  const auto& matched_prefix_lens = attn_metadata.genrec_matched_prefix_lens;
  CHECK(q_seq_lens.defined());
  CHECK(kv_seq_lens.defined());
  CHECK(history_lens.defined());
  CHECK(context_lens.defined());
  CHECK(realtime_lens.defined());
  CHECK(target_lens.defined());
  CHECK(matched_prefix_lens.defined());
  const int64_t batch_size = q_seq_lens.size(0);
  CHECK_GT(batch_size, 0);
  CHECK_EQ(kv_seq_lens.size(0), batch_size);
  CHECK_EQ(history_lens.size(0), batch_size);
  CHECK_EQ(context_lens.size(0), batch_size);
  CHECK_EQ(realtime_lens.size(0), batch_size);
  CHECK_EQ(target_lens.size(0), batch_size);
  CHECK_EQ(matched_prefix_lens.size(0), batch_size);

  const auto q_seq_lens_host = tensor_to_int64_vec_cpu(q_seq_lens);
  const auto kv_seq_lens_host = tensor_to_int64_vec_cpu(kv_seq_lens);
  const int64_t total_q =
      std::accumulate(q_seq_lens_host.begin(), q_seq_lens_host.end(), 0LL);
  const int64_t total_kv =
      std::accumulate(kv_seq_lens_host.begin(), kv_seq_lens_host.end(), 0LL);
  CHECK_EQ(query.size(0), total_q);
  CHECK_EQ(key.size(0), total_kv);
  CHECK_EQ(value.size(0), total_kv);

  auto query_snd = query.view({total_q, num_heads_, head_size_}).contiguous();
  auto key_snd = key.view({total_kv, num_kv_heads_, head_size_}).contiguous();
  auto value_snd =
      value.view({total_kv, num_kv_heads_, head_size_}).contiguous();
  torch::Tensor output_snd;

  auto wall_start = std::chrono::steady_clock::now();
  if (batch_size > 1) {
    CHECK(backend_ == MTGRAttentionTestBackend::kFusedNoMatch)
        << "Multi-batch CUDA test path is currently enabled only for fused "
           "kernel exploration";
    const auto matched_prefix_lens_host =
        tensor_to_int64_vec_cpu(matched_prefix_lens);
    const bool all_no_match = std::all_of(matched_prefix_lens_host.begin(),
                                          matched_prefix_lens_host.end(),
                                          [](int64_t v) { return v == 0; });
    const bool all_partial_match = std::all_of(matched_prefix_lens_host.begin(),
                                               matched_prefix_lens_host.end(),
                                               [](int64_t v) { return v > 0; });
    CHECK(all_no_match || all_partial_match)
        << "Mixed no_match and partial_match batches are not enabled yet";
    if (all_no_match) {
      const auto& layout = get_cached_no_match_batch_layout(q_seq_lens,
                                                            kv_seq_lens,
                                                            history_lens,
                                                            context_lens,
                                                            realtime_lens,
                                                            target_lens,
                                                            matched_prefix_lens,
                                                            query.device());
      output_snd = run_fused_no_match_batched(
          query_snd, key_snd, value_snd, layout, scale_, &last_metrics_);
    } else {
      CHECK(attn_metadata.block_table.defined());
      CHECK(!kv_cache.empty());
      const int64_t block_size = static_cast<int64_t>(FLAGS_block_size);
      auto key_cache = kv_cache.get_k_cache();
      auto value_cache = kv_cache.get_v_cache();
      const auto& layout = get_cached_four_segment_partial_batch_layout(
          q_seq_lens,
          kv_seq_lens,
          history_lens,
          context_lens,
          realtime_lens,
          target_lens,
          matched_prefix_lens,
          attn_metadata.block_table,
          query.device());
      output_snd = run_four_segment_partial_rt_batched(query_snd,
                                                       key_snd,
                                                       value_snd,
                                                       key_cache,
                                                       value_cache,
                                                       layout,
                                                       block_size,
                                                       scale_,
                                                       &last_metrics_);
    }
  } else {
    const int64_t q_len = q_seq_lens_host[0];
    const int64_t kv_len = kv_seq_lens_host[0];
    const int64_t h = history_lens[0].item<int64_t>();
    const int64_t c = context_lens[0].item<int64_t>();
    const int64_t r = realtime_lens[0].item<int64_t>();
    const int64_t t = target_lens[0].item<int64_t>();
    const int64_t matched = matched_prefix_lens[0].item<int64_t>();
    CHECK_GE(matched, 0);
    CHECK_LT(matched, h + c + r);
    CHECK_EQ(total_q, q_len);
    CHECK_EQ(total_kv, kv_len);

    if (matched == 0) {
      CHECK_EQ(q_len, h + c + r + t);
      if (backend_ == MTGRAttentionTestBackend::kOneStage) {
        output_snd = run_one_stage_no_match(
            query_snd, key_snd, value_snd, h, c, r, t, scale_, &last_metrics_);
      } else if (backend_ == MTGRAttentionTestBackend::kFusedNoMatch) {
        const auto& layout =
            get_cached_no_match_batch_layout(q_seq_lens,
                                             kv_seq_lens,
                                             history_lens,
                                             context_lens,
                                             realtime_lens,
                                             target_lens,
                                             matched_prefix_lens,
                                             query.device());
        output_snd = run_fused_no_match_batched(
            query_snd, key_snd, value_snd, layout, scale_, &last_metrics_);
      } else {
        output_snd = run_multi_no_match_preplanned(
            query_snd, key_snd, value_snd, h, c, r, t, scale_, &last_metrics_);
      }
    } else {
      CHECK_GE(matched, h + c)
          << "GPU test infra keeps only no_match and partial_real_time_match";
      const int64_t realtime_matched = matched - h - c;
      const int64_t realtime_unmatched = r - realtime_matched;
      CHECK_GT(realtime_unmatched, 0);
      CHECK_EQ(q_len, realtime_unmatched + t);
      CHECK(attn_metadata.block_table.defined());
      CHECK(attn_metadata.slot_mapping.defined());
      CHECK_EQ(attn_metadata.block_table.dim(), 2);
      CHECK_GE(attn_metadata.block_table.size(0), 1);
      CHECK(!kv_cache.empty());
      const int64_t block_size = static_cast<int64_t>(FLAGS_block_size);
      auto key_cache = kv_cache.get_k_cache();
      auto value_cache = kv_cache.get_v_cache();
      auto block_table_row =
          attn_metadata.block_table.select(0, 0).contiguous();
      if (backend_ == MTGRAttentionTestBackend::kOneStage) {
        output_snd = run_one_stage_partial_rt(query_snd,
                                              key_snd,
                                              value_snd,
                                              key_cache,
                                              value_cache,
                                              block_table_row,
                                              block_size,
                                              matched,
                                              realtime_unmatched,
                                              t,
                                              scale_,
                                              &last_metrics_);
      } else if (backend_ == MTGRAttentionTestBackend::kFusedNoMatch) {
        const auto& layout = get_cached_four_segment_partial_batch_layout(
            q_seq_lens,
            kv_seq_lens,
            history_lens,
            context_lens,
            realtime_lens,
            target_lens,
            matched_prefix_lens,
            attn_metadata.block_table,
            query.device());
        output_snd = run_four_segment_partial_rt_batched(query_snd,
                                                         key_snd,
                                                         value_snd,
                                                         key_cache,
                                                         value_cache,
                                                         layout,
                                                         block_size,
                                                         scale_,
                                                         &last_metrics_);
      } else {
        output_snd = run_multi_partial_rt(query_snd,
                                          key_snd,
                                          value_snd,
                                          key_cache,
                                          value_cache,
                                          block_table_row,
                                          block_size,
                                          matched,
                                          realtime_unmatched,
                                          t,
                                          scale_,
                                          &last_metrics_);
      }
    }
  }
  auto wall_end = std::chrono::steady_clock::now();
  if (last_metrics_.wall_total_ms == 0.0) {
    last_metrics_.wall_total_ms =
        std::chrono::duration<double, std::milli>(wall_end - wall_start)
            .count();
  }
  return {output_snd.view({total_q, num_heads_ * head_size_}).contiguous(),
          output_lse};
}

xllm::layer::AttentionMetadata make_mtgr_attention_metadata(
    const MTGRAttentionTestShape& shape,
    const torch::Device& device,
    int64_t block_size) {
  CHECK_GT(block_size, 0);
  CHECK_GE(shape.matched_prefix, 0);
  CHECK_LT(shape.matched_prefix,
           shape.history + shape.context + shape.realtime);
  const int64_t block_count = (shape.total_len() + block_size - 1) / block_size;
  const auto block_table_host = make_block_table_host(block_count);
  const auto slot_mapping_host = build_slot_mapping_host(
      block_table_host, block_size, shape.matched_prefix, shape.local_len());
  const std::vector<int32_t> segment_offsets = {
      0,
      static_cast<int32_t>(shape.history),
      static_cast<int32_t>(shape.history + shape.context),
      static_cast<int32_t>(shape.history + shape.context + shape.realtime),
      static_cast<int32_t>(shape.total_len())};
  const std::vector<int32_t> segment_rules = {0, 1, 0, 2};

  xllm::layer::AttentionMetadata metadata;
  auto len_opts =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  auto i32_dev_opts =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  metadata.is_dummy = false;
  metadata.is_prefill = true;
  metadata.is_chunked_prefill = false;
  metadata.mtgr_match_mode = shape.matched_prefix == 0
                                 ? xllm::layer::MTGRMatchMode::kNoMatchOnly
                                 : xllm::layer::MTGRMatchMode::kPartialOnly;
  metadata.max_query_len = shape.local_len();
  metadata.max_seq_len = shape.local_len();
  metadata.q_seq_lens = torch::tensor({shape.local_len()}, len_opts);
  metadata.kv_seq_lens = torch::tensor({shape.local_len()}, len_opts);
  metadata.genrec_history_lens = torch::tensor({shape.history}, len_opts);
  metadata.genrec_context_lens = torch::tensor({shape.context}, len_opts);
  metadata.genrec_real_time_lens = torch::tensor({shape.realtime}, len_opts);
  metadata.genrec_target_lens = torch::tensor({shape.target}, len_opts);
  metadata.genrec_matched_prefix_lens =
      torch::tensor({shape.matched_prefix}, len_opts);
  metadata.block_table =
      torch::tensor(block_table_host,
                    torch::TensorOptions().dtype(torch::kInt32).device(device))
          .view({1, block_count})
          .contiguous();
  metadata.slot_mapping =
      torch::tensor(slot_mapping_host,
                    torch::TensorOptions().dtype(torch::kInt64).device(device))
          .contiguous();
  metadata.mtgr_segment_offsets_i32 =
      torch::tensor(segment_offsets, i32_dev_opts).view({1, 5}).contiguous();
  metadata.mtgr_segment_rules_i32 =
      torch::tensor(segment_rules, i32_dev_opts).contiguous();
  metadata.mtgr_q_seq_starts_i32 =
      torch::tensor({0}, i32_dev_opts).contiguous();
  metadata.mtgr_matched_prefix_lens_i32 =
      torch::tensor({shape.matched_prefix}, i32_dev_opts).contiguous();
  return metadata;
}

xllm::layer::AttentionMetadata make_mtgr_attention_metadata(
    const std::vector<MTGRAttentionTestShape>& shapes,
    const torch::Device& device,
    int64_t block_size) {
  CHECK_GT(block_size, 0);
  CHECK(!shapes.empty());
  (void)device;

  std::vector<int64_t> q_seq_lens;
  std::vector<int64_t> kv_seq_lens;
  std::vector<int64_t> history_lens;
  std::vector<int64_t> context_lens;
  std::vector<int64_t> realtime_lens;
  std::vector<int64_t> target_lens;
  std::vector<int64_t> matched_prefix_lens;
  std::vector<int32_t> segment_offsets;
  std::vector<int32_t> q_seq_starts;
  std::vector<int32_t> matched_prefix_lens_i32;
  q_seq_lens.reserve(shapes.size());
  kv_seq_lens.reserve(shapes.size());
  history_lens.reserve(shapes.size());
  context_lens.reserve(shapes.size());
  realtime_lens.reserve(shapes.size());
  target_lens.reserve(shapes.size());
  matched_prefix_lens.reserve(shapes.size());
  segment_offsets.reserve(shapes.size() * 5);
  q_seq_starts.reserve(shapes.size());
  matched_prefix_lens_i32.reserve(shapes.size());

  int64_t total_q = 0;
  int64_t max_query_len = 0;
  for (const auto& shape : shapes) {
    CHECK_GE(shape.matched_prefix, 0);
    CHECK_LT(shape.matched_prefix,
             shape.history + shape.context + shape.realtime);
    q_seq_lens.push_back(shape.local_len());
    kv_seq_lens.push_back(shape.local_len());
    history_lens.push_back(shape.history);
    context_lens.push_back(shape.context);
    realtime_lens.push_back(shape.realtime);
    target_lens.push_back(shape.target);
    matched_prefix_lens.push_back(shape.matched_prefix);
    q_seq_starts.push_back(static_cast<int32_t>(total_q));
    matched_prefix_lens_i32.push_back(
        static_cast<int32_t>(shape.matched_prefix));
    segment_offsets.push_back(0);
    segment_offsets.push_back(static_cast<int32_t>(shape.history));
    segment_offsets.push_back(
        static_cast<int32_t>(shape.history + shape.context));
    segment_offsets.push_back(
        static_cast<int32_t>(shape.history + shape.context + shape.realtime));
    segment_offsets.push_back(static_cast<int32_t>(shape.total_len()));
    total_q += shape.local_len();
    max_query_len = std::max(max_query_len, shape.local_len());
  }

  xllm::layer::AttentionMetadata metadata;
  auto len_opts =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  auto i32_dev_opts =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  metadata.is_dummy = false;
  metadata.is_prefill = true;
  metadata.is_chunked_prefill = false;
  const bool all_no_match =
      std::all_of(matched_prefix_lens.begin(),
                  matched_prefix_lens.end(),
                  [](int64_t matched) { return matched == 0; });
  const bool all_partial =
      std::all_of(matched_prefix_lens.begin(),
                  matched_prefix_lens.end(),
                  [](int64_t matched) { return matched > 0; });
  metadata.mtgr_match_mode =
      all_no_match ? xllm::layer::MTGRMatchMode::kNoMatchOnly
                   : (all_partial ? xllm::layer::MTGRMatchMode::kPartialOnly
                                  : xllm::layer::MTGRMatchMode::kMixed);
  metadata.max_query_len = max_query_len;
  metadata.max_seq_len = max_query_len;
  metadata.q_seq_lens = torch::tensor(q_seq_lens, len_opts).contiguous();
  metadata.kv_seq_lens = torch::tensor(kv_seq_lens, len_opts).contiguous();
  metadata.genrec_history_lens =
      torch::tensor(history_lens, len_opts).contiguous();
  metadata.genrec_context_lens =
      torch::tensor(context_lens, len_opts).contiguous();
  metadata.genrec_real_time_lens =
      torch::tensor(realtime_lens, len_opts).contiguous();
  metadata.genrec_target_lens =
      torch::tensor(target_lens, len_opts).contiguous();
  metadata.genrec_matched_prefix_lens =
      torch::tensor(matched_prefix_lens, len_opts).contiguous();
  metadata.mtgr_segment_offsets_i32 =
      torch::tensor(segment_offsets, i32_dev_opts)
          .view({static_cast<int64_t>(shapes.size()), 5})
          .contiguous();
  metadata.mtgr_segment_rules_i32 =
      torch::tensor({0, 1, 0, 2}, i32_dev_opts).contiguous();
  metadata.mtgr_q_seq_starts_i32 =
      torch::tensor(q_seq_starts, i32_dev_opts).contiguous();
  metadata.mtgr_matched_prefix_lens_i32 =
      torch::tensor(matched_prefix_lens_i32, i32_dev_opts).contiguous();
  return metadata;
}

xllm::KVCache make_mtgr_kv_cache(const MTGRAttentionTestShape& shape,
                                 const torch::Device& device,
                                 torch::ScalarType dtype,
                                 int64_t block_size) {
  const int64_t block_count =
      (shape.total_len() + block_size - 1) / block_size + 4;
  auto opts = torch::TensorOptions().dtype(dtype).device(device);
  auto key_cache = torch::zeros(
      {block_count, block_size, shape.kv_heads, shape.head_dim}, opts);
  auto value_cache = torch::zeros(
      {block_count, block_size, shape.kv_heads, shape.head_dim}, opts);
  return xllm::KVCache(key_cache, value_cache);
}

void prefill_mtgr_matched_prefix_cache(const torch::Tensor& full_key_bsnd,
                                       const torch::Tensor& full_value_bsnd,
                                       const MTGRAttentionTestShape& shape,
                                       int64_t block_size,
                                       xllm::KVCache& kv_cache) {
  if (shape.matched_prefix <= 0) {
    return;
  }
  CHECK_EQ(full_key_bsnd.dim(), 4);
  CHECK_EQ(full_value_bsnd.dim(), 4);
  CHECK_EQ(full_key_bsnd.size(0), 1);
  CHECK_EQ(full_value_bsnd.sizes(), full_key_bsnd.sizes());
  auto key_cache = kv_cache.get_k_cache();
  auto value_cache = kv_cache.get_v_cache();
  const auto block_table =
      make_block_table_host((shape.total_len() + block_size - 1) / block_size);
  const auto slots_host =
      build_slot_mapping_host(block_table, block_size, 0, shape.matched_prefix);
  auto slots = torch::tensor(slots_host,
                             torch::TensorOptions()
                                 .dtype(torch::kInt64)
                                 .device(full_key_bsnd.device()));
  auto key_flat = key_cache.view(
      {key_cache.size(0) * block_size, key_cache.size(2), key_cache.size(3)});
  auto value_flat = value_cache.view({value_cache.size(0) * block_size,
                                      value_cache.size(2),
                                      value_cache.size(3)});
  key_flat.index_copy_(
      0, slots, full_key_bsnd.select(0, 0).narrow(0, 0, shape.matched_prefix));
  value_flat.index_copy_(
      0,
      slots,
      full_value_bsnd.select(0, 0).narrow(0, 0, shape.matched_prefix));
}

MTGRPartialBatchSetup make_mtgr_partial_batch_setup(
    const std::vector<MTGRAttentionTestShape>& shapes,
    const std::vector<torch::Tensor>& full_key_bsnd,
    const std::vector<torch::Tensor>& full_value_bsnd,
    const torch::Device& device,
    torch::ScalarType dtype,
    int64_t block_size) {
  CHECK_GT(block_size, 0);
  CHECK(!shapes.empty());
  CHECK_EQ(full_key_bsnd.size(), shapes.size());
  CHECK_EQ(full_value_bsnd.size(), shapes.size());

  const auto& ref = shapes.front();
  int64_t total_slot_count = 0;
  int64_t total_block_count = 0;
  int64_t max_block_count = 0;
  std::vector<std::vector<int32_t>> block_rows;
  block_rows.reserve(shapes.size());

  for (size_t i = 0; i < shapes.size(); ++i) {
    const auto& shape = shapes[i];
    CHECK_EQ(shape.heads, ref.heads);
    CHECK_EQ(shape.kv_heads, ref.kv_heads);
    CHECK_EQ(shape.head_dim, ref.head_dim);
    CHECK_GE(shape.matched_prefix, 0);
    if (shape.matched_prefix > 0) {
      CHECK_GE(shape.matched_prefix, shape.history + shape.context);
      CHECK_LT(shape.matched_prefix,
               shape.history + shape.context + shape.realtime);
    }
    CHECK(full_key_bsnd[i].defined());
    CHECK(full_value_bsnd[i].defined());
    CHECK_EQ(full_key_bsnd[i].dim(), 4);
    CHECK_EQ(full_value_bsnd[i].sizes(), full_key_bsnd[i].sizes());
    CHECK_EQ(full_key_bsnd[i].size(0), 1);
    CHECK_EQ(full_key_bsnd[i].size(1), shape.total_len());
    CHECK_EQ(full_key_bsnd[i].size(2), shape.kv_heads);
    CHECK_EQ(full_key_bsnd[i].size(3), shape.head_dim);
    CHECK_EQ(full_key_bsnd[i].device(), device);
    CHECK_EQ(full_value_bsnd[i].device(), device);
    const int64_t block_count =
        (shape.total_len() + block_size - 1) / block_size;
    std::vector<int32_t> row(static_cast<size_t>(block_count));
    for (int64_t block_idx = 0; block_idx < block_count; ++block_idx) {
      row[static_cast<size_t>(block_idx)] =
          static_cast<int32_t>(total_block_count + block_idx);
    }
    total_block_count += block_count;
    max_block_count = std::max(max_block_count, block_count);
    total_slot_count += shape.local_len();
    block_rows.push_back(std::move(row));
  }

  auto metadata = make_mtgr_attention_metadata(shapes, device, block_size);
  std::vector<int32_t> block_table_host(
      static_cast<size_t>(shapes.size() * max_block_count), 0);
  std::vector<int64_t> slot_mapping_host;
  slot_mapping_host.reserve(static_cast<size_t>(total_slot_count));

  for (size_t i = 0; i < shapes.size(); ++i) {
    const auto& row = block_rows[i];
    for (size_t block_idx = 0; block_idx < row.size(); ++block_idx) {
      block_table_host[i * static_cast<size_t>(max_block_count) + block_idx] =
          row[block_idx];
    }
    const auto seq_slot_mapping = build_slot_mapping_host(
        row, block_size, shapes[i].matched_prefix, shapes[i].local_len());
    slot_mapping_host.insert(slot_mapping_host.end(),
                             seq_slot_mapping.begin(),
                             seq_slot_mapping.end());
  }

  metadata.block_table =
      torch::tensor(block_table_host,
                    torch::TensorOptions().dtype(torch::kInt32).device(device))
          .view({static_cast<int64_t>(shapes.size()), max_block_count})
          .contiguous();
  metadata.slot_mapping =
      torch::tensor(slot_mapping_host,
                    torch::TensorOptions().dtype(torch::kInt64).device(device))
          .contiguous();

  const int64_t cache_block_count = std::max<int64_t>(total_block_count + 4, 1);
  auto opts = torch::TensorOptions().dtype(dtype).device(device);
  auto key_cache = torch::zeros(
      {cache_block_count, block_size, ref.kv_heads, ref.head_dim}, opts);
  auto value_cache = torch::zeros(
      {cache_block_count, block_size, ref.kv_heads, ref.head_dim}, opts);

  for (size_t i = 0; i < shapes.size(); ++i) {
    const auto& shape = shapes[i];
    if (shape.matched_prefix == 0) {
      continue;
    }
    const auto prefix_slots_host = build_slot_mapping_host(
        block_rows[i], block_size, 0, shape.matched_prefix);
    auto prefix_slots =
        torch::tensor(
            prefix_slots_host,
            torch::TensorOptions().dtype(torch::kInt32).device(device))
            .contiguous();
    auto prefix_key = full_key_bsnd[i]
                          .select(0, 0)
                          .narrow(0, 0, shape.matched_prefix)
                          .contiguous();
    auto prefix_value = full_value_bsnd[i]
                            .select(0, 0)
                            .narrow(0, 0, shape.matched_prefix)
                            .contiguous();
    scatter_to_cache(
        prefix_key, prefix_value, key_cache, value_cache, prefix_slots);
  }

  MTGRPartialBatchSetup setup;
  setup.metadata = std::move(metadata);
  setup.kv_cache = xllm::KVCache(key_cache, value_cache);
  return setup;
}

std::vector<MTGRAttentionTestShape> build_mtgr_sweep_shapes(
    bool full_sweep,
    bool partial_match) {
  const std::vector<int64_t> default_heads_all =
      full_sweep ? std::vector<int64_t>{4, 8, 12} : std::vector<int64_t>{8};
  const std::vector<int64_t> default_head_dims_all =
      full_sweep ? std::vector<int64_t>{32, 64, 128}
                 : std::vector<int64_t>{128};
  const std::vector<int64_t> default_histories_all =
      full_sweep ? std::vector<int64_t>{1024, 2048, 4096}
                 : std::vector<int64_t>{2048};
  const std::vector<int64_t> default_realtime_all =
      full_sweep ? std::vector<int64_t>{128, 256, 512}
                 : std::vector<int64_t>{512};
  const std::vector<int64_t> default_targets_all =
      full_sweep ? std::vector<int64_t>{800, 1600, 2400}
                 : std::vector<int64_t>{1600};

  const auto parse_env_int_list =
      [](const char* name) -> std::optional<std::vector<int64_t>> {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
      return std::nullopt;
    }
    std::vector<int64_t> parsed;
    std::string token;
    std::stringstream ss(value);
    while (std::getline(ss, token, ',')) {
      token.erase(
          std::remove_if(token.begin(),
                         token.end(),
                         [](unsigned char ch) { return std::isspace(ch); }),
          token.end());
      if (token.empty()) {
        continue;
      }
      char* end = nullptr;
      const long long v = std::strtoll(token.c_str(), &end, 10);
      if (end == token.c_str() || *end != '\0') {
        return std::nullopt;
      }
      parsed.push_back(static_cast<int64_t>(v));
    }
    if (parsed.empty()) {
      return std::nullopt;
    }
    return parsed;
  };

  const std::vector<int64_t> heads_all =
      parse_env_int_list("XLLM_MTGR_ATTENTION_SWEEP_HEADS")
          .value_or(default_heads_all);
  const std::vector<int64_t> head_dims_all =
      parse_env_int_list("XLLM_MTGR_ATTENTION_SWEEP_HEAD_DIMS")
          .value_or(default_head_dims_all);
  const std::vector<int64_t> histories_all =
      parse_env_int_list("XLLM_MTGR_ATTENTION_SWEEP_HISTORIES")
          .value_or(default_histories_all);
  const std::vector<int64_t> realtime_all =
      parse_env_int_list("XLLM_MTGR_ATTENTION_SWEEP_REALTIME")
          .value_or(default_realtime_all);
  const std::vector<int64_t> targets_all =
      parse_env_int_list("XLLM_MTGR_ATTENTION_SWEEP_TARGETS")
          .value_or(default_targets_all);

  std::vector<MTGRAttentionTestShape> cases;
  for (int64_t heads : heads_all) {
    for (int64_t head_dim : head_dims_all) {
      for (int64_t history : histories_all) {
        for (int64_t realtime : realtime_all) {
          for (int64_t target : targets_all) {
            MTGRAttentionTestShape shape;
            shape.heads = heads;
            shape.kv_heads = heads;
            shape.head_dim = head_dim;
            shape.history = history;
            shape.context = 8;
            shape.realtime = realtime;
            shape.target = target;
            shape.matched_prefix =
                partial_match ? history + shape.context + (realtime * 4) / 5
                              : 0;
            cases.push_back(shape);
          }
        }
      }
    }
  }
  return cases;
}

int env_int(const char* name, int default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return default_value;
  }
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0') {
    return default_value;
  }
  return static_cast<int>(parsed);
}

bool env_flag_enabled(const char* name, bool default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return default_value;
  }
  std::string v(value);
  std::transform(v.begin(), v.end(), v.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return (v == "1" || v == "true" || v == "yes" || v == "on");
}

}  // namespace xllm::kernel::cuda::test
