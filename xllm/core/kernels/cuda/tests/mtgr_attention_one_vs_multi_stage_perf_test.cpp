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
#include <cstring>
#include <fstream>
#include <string>

#include "core/common/global_flags.h"
#include "mtgr_attenion_test.h"

namespace xllm::kernel::cuda::test {
namespace {

constexpr int64_t kBlockSize = 128;

struct AvgMetrics {
  MTGRAttentionTestMetrics one;
  MTGRAttentionTestMetrics multi;
  double diff_max_abs = 0.0;
  double diff_mean_abs = 0.0;
};

struct CompareMetrics {
  MTGRAttentionTestMetrics base;
  MTGRAttentionTestMetrics candidate;
  double diff_max_abs = 0.0;
  double diff_mean_abs = 0.0;
};

class MTGRAttentionOneVsMultiStagePerfTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA not available, skipping test.";
    }
    FLAGS_flashinfer_workspace_buffer_size = std::max<int64_t>(
        FLAGS_flashinfer_workspace_buffer_size, 64 * 1024 * 1024);
    FLAGS_block_size = kBlockSize;
    torch::manual_seed(20260429);
    torch::cuda::manual_seed_all(20260429);
    device_ = torch::Device(torch::kCUDA, 0);
  }

  AvgMetrics run_shape(const MTGRAttentionTestShape& shape,
                       int warmup,
                       int repeat) {
    const auto cmp = run_shape_vs_backend(
        shape, warmup, repeat, MTGRAttentionTestBackend::kMultiStage);
    AvgMetrics avg;
    avg.one = cmp.base;
    avg.multi = cmp.candidate;
    avg.diff_max_abs = cmp.diff_max_abs;
    avg.diff_mean_abs = cmp.diff_mean_abs;
    return avg;
  }

  CompareMetrics run_shape_vs_backend(
      const MTGRAttentionTestShape& shape,
      int warmup,
      int repeat,
      MTGRAttentionTestBackend candidate_backend) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat16).device(device_);
    const int64_t total = shape.total_len();
    const int64_t local = shape.local_len();
    const float scale = 1.0f / std::sqrt(static_cast<float>(shape.head_dim));

    auto full_query =
        torch::randn({1, total, shape.heads, shape.head_dim}, opts) * 0.05;
    auto full_key =
        torch::randn({1, total, shape.kv_heads, shape.head_dim}, opts) * 0.05;
    auto full_value =
        torch::randn({1, total, shape.kv_heads, shape.head_dim}, opts) * 0.05;

    auto query = full_query.select(0, 0)
                     .narrow(0, shape.matched_prefix, local)
                     .contiguous()
                     .view({local, shape.heads * shape.head_dim});
    auto key = full_key.select(0, 0)
                   .narrow(0, shape.matched_prefix, local)
                   .contiguous()
                   .view({local, shape.kv_heads * shape.head_dim});
    auto value = full_value.select(0, 0)
                     .narrow(0, shape.matched_prefix, local)
                     .contiguous()
                     .view({local, shape.kv_heads * shape.head_dim});

    auto metadata = make_mtgr_attention_metadata(shape, device_, kBlockSize);
    auto one_cache =
        make_mtgr_kv_cache(shape, device_, torch::kFloat16, kBlockSize);
    auto multi_cache =
        make_mtgr_kv_cache(shape, device_, torch::kFloat16, kBlockSize);
    prefill_mtgr_matched_prefix_cache(
        full_key, full_value, shape, kBlockSize, one_cache);
    prefill_mtgr_matched_prefix_cache(
        full_key, full_value, shape, kBlockSize, multi_cache);

    MTGRAttentionImplTest one_stage(shape.heads,
                                    shape.head_dim,
                                    scale,
                                    shape.kv_heads,
                                    MTGRAttentionTestBackend::kOneStage);
    MTGRAttentionImplTest multi_stage(
        shape.heads, shape.head_dim, scale, shape.kv_heads, candidate_backend);

    for (int i = 0; i < warmup; ++i) {
      (void)std::get<0>(
          one_stage.forward(metadata, query, key, value, one_cache));
      (void)std::get<0>(
          multi_stage.forward(metadata, query, key, value, multi_cache));
    }
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

    CompareMetrics avg;
    torch::Tensor one_ref;
    torch::Tensor multi_ref;
    for (int i = 0; i < repeat; ++i) {
      auto one_output = std::get<0>(
          one_stage.forward(metadata, query, key, value, one_cache));
      const auto one_metrics = one_stage.last_metrics();
      auto multi_output = std::get<0>(
          multi_stage.forward(metadata, query, key, value, multi_cache));
      const auto multi_metrics = multi_stage.last_metrics();

      avg.base.mask_build_ms += one_metrics.mask_build_ms;
      avg.base.h2d_ms += one_metrics.h2d_ms;
      avg.base.workspace_ms += one_metrics.workspace_ms;
      avg.base.fia_ms += one_metrics.fia_ms;
      avg.base.device_total_ms += one_metrics.device_total_ms;
      avg.base.wall_total_ms += one_metrics.wall_total_ms;
      avg.candidate.workspace_ms += multi_metrics.workspace_ms;
      avg.candidate.device_total_ms += multi_metrics.device_total_ms;
      avg.candidate.wall_total_ms += multi_metrics.wall_total_ms;
      if (avg.candidate.stages.empty()) {
        avg.candidate.stages = multi_metrics.stages;
      } else {
        for (size_t s = 0; s < avg.candidate.stages.size(); ++s) {
          avg.candidate.stages[s].workspace_ms +=
              multi_metrics.stages[s].workspace_ms;
          avg.candidate.stages[s].exec_ms += multi_metrics.stages[s].exec_ms;
          avg.candidate.stages[s].host_submit_ms +=
              multi_metrics.stages[s].host_submit_ms;
        }
      }

      if (i == 0) {
        one_ref = one_output;
        multi_ref = multi_output;
      }
    }

    const double inv = 1.0 / static_cast<double>(repeat);
    avg.base.mask_build_ms *= inv;
    avg.base.h2d_ms *= inv;
    avg.base.workspace_ms *= inv;
    avg.base.fia_ms *= inv;
    avg.base.device_total_ms *= inv;
    avg.base.wall_total_ms *= inv;
    avg.candidate.workspace_ms *= inv;
    avg.candidate.device_total_ms *= inv;
    avg.candidate.wall_total_ms *= inv;
    for (auto& stage : avg.candidate.stages) {
      stage.workspace_ms *= inv;
      stage.exec_ms *= inv;
      stage.host_submit_ms *= inv;
    }

    auto diff =
        (one_ref.to(torch::kFloat32) - multi_ref.to(torch::kFloat32)).abs();
    avg.diff_max_abs = diff.max().item<double>();
    avg.diff_mean_abs = diff.mean().item<double>();
    return avg;
  }

  torch::Device device_ = torch::Device(torch::kCPU);
};

const MTGRStageMetric* find_stage(const MTGRAttentionTestMetrics& metrics,
                                  const char* name) {
  for (const auto& stage : metrics.stages) {
    if (stage.name == name) {
      return &stage;
    }
  }
  return nullptr;
}

double stage_workspace(const MTGRAttentionTestMetrics& metrics,
                       const char* name) {
  const auto* stage = find_stage(metrics, name);
  return stage != nullptr ? stage->workspace_ms : 0.0;
}

double stage_exec(const MTGRAttentionTestMetrics& metrics, const char* name) {
  const auto* stage = find_stage(metrics, name);
  return stage != nullptr ? stage->exec_ms : 0.0;
}

std::string csv_path_from_env(const char* env_name, const char* default_name) {
  const char* env = std::getenv(env_name);
  if (env != nullptr && std::strlen(env) > 0) {
    return std::string(env);
  }
  return std::string(
             "/export/home/wangyifan/Code/xllm/xllm/core/kernels/cuda/tests/") +
         default_name;
}

}  // namespace

TEST_F(MTGRAttentionOneVsMultiStagePerfTest, NoMatchSweepCsv) {
  torch::NoGradGuard no_grad_guard;
  const bool full_sweep =
      env_flag_enabled("XLLM_MTGR_ATTENTION_FULL_SWEEP", true);
  const int warmup = std::max(0, env_int("XLLM_MTGR_ATTENTION_WARMUP", 5));
  const int repeat = std::max(1, env_int("XLLM_MTGR_ATTENTION_REPEAT", 20));
  const std::string csv_path = csv_path_from_env(
      "XLLM_MTGR_ATTENTION_NO_MATCH_CSV", "mtgr_attention_no_match.csv");
  std::ofstream csv(csv_path, std::ios::out | std::ios::trunc);
  CHECK(csv.is_open()) << "failed to open csv path: " << csv_path;

  csv << "common,,,,,,one_stage,,,,,,multi_stage,,,,,,,,,,speedup,,delta_ms,"
         "precision,\n";
  csv << "mask_build_mode,heads,head_dim,history,real_time,target,"
         "mask_build_ms,h2d_ms,workspace_ms,fia_ms,device_total_ms,wall_total_"
         "ms,"
         "rt_tgt_on_hcr_trapezoid.workspace_ms,rt_tgt_on_hcr_trapezoid.exec_ms,"
         "tgt_diag_update_fused.workspace_ms,tgt_diag_update_fused.exec_ms,"
         "hist_fa.workspace_ms,hist_fa.exec_ms,ctx_on_hc.workspace_ms,"
         "ctx_on_hc.exec_ms,device_total_ms,wall_total_ms,"
         "one_device_total_ms/multi_device_total_ms,"
         "one_wall_total_ms/multi_wall_total_ms,"
         "one_device_total_ms-multi_device_total_ms,"
         "one_wall_total_ms-multi_wall_total_ms,diff_max_abs,diff_mean_abs\n";

  const auto cases =
      build_mtgr_sweep_shapes(full_sweep, /*partial_match=*/false);
  std::fprintf(
      stderr,
      "[MTGR][CUDA][Perf][NoMatch] cases=%zu warmup=%d repeat=%d csv=%s\n",
      cases.size(),
      warmup,
      repeat,
      csv_path.c_str());
  int64_t idx = 0;
  for (const auto& shape : cases) {
    ++idx;
    const auto avg = run_shape(shape, warmup, repeat);
    const double speedup_dev =
        avg.one.device_total_ms / avg.multi.device_total_ms;
    const double speedup_wall = avg.one.wall_total_ms / avg.multi.wall_total_ms;
    const double delta_dev =
        avg.one.device_total_ms - avg.multi.device_total_ms;
    const double delta_wall = avg.one.wall_total_ms - avg.multi.wall_total_ms;

    char row[2048];
    std::snprintf(
        row,
        sizeof(row),
        "device,%lld,%lld,%lld,%lld,%lld,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
        "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
        "%.6f,%.6f,%.6e,%.6e",
        static_cast<long long>(shape.heads),
        static_cast<long long>(shape.head_dim),
        static_cast<long long>(shape.history),
        static_cast<long long>(shape.realtime),
        static_cast<long long>(shape.target),
        avg.one.mask_build_ms,
        avg.one.h2d_ms,
        avg.one.workspace_ms,
        avg.one.fia_ms,
        avg.one.device_total_ms,
        avg.one.wall_total_ms,
        stage_workspace(avg.multi, "rt_tgt_on_hcr_trapezoid"),
        stage_exec(avg.multi, "rt_tgt_on_hcr_trapezoid"),
        stage_workspace(avg.multi, "tgt_diag_update_fused"),
        stage_exec(avg.multi, "tgt_diag_update_fused"),
        stage_workspace(avg.multi, "hist_fa"),
        stage_exec(avg.multi, "hist_fa"),
        stage_workspace(avg.multi, "ctx_on_hc"),
        stage_exec(avg.multi, "ctx_on_hc"),
        avg.multi.device_total_ms,
        avg.multi.wall_total_ms,
        speedup_dev,
        speedup_wall,
        delta_dev,
        delta_wall,
        avg.diff_max_abs,
        avg.diff_mean_abs);
    csv << row << "\n";
    csv.flush();
    std::fprintf(stderr,
                 "[MTGR][CUDA][Perf][NoMatch %lld/%zu] %s\n",
                 static_cast<long long>(idx),
                 cases.size(),
                 row);
  }
  std::fflush(stderr);
}

TEST_F(MTGRAttentionOneVsMultiStagePerfTest, PartialRealTimeMatchSweepCsv) {
  torch::NoGradGuard no_grad_guard;
  const bool full_sweep =
      env_flag_enabled("XLLM_MTGR_ATTENTION_FULL_SWEEP", true);
  const int warmup = std::max(0, env_int("XLLM_MTGR_ATTENTION_WARMUP", 5));
  const int repeat = std::max(1, env_int("XLLM_MTGR_ATTENTION_REPEAT", 20));
  const std::string csv_path =
      csv_path_from_env("XLLM_MTGR_ATTENTION_PARTIAL_MATCH_CSV",
                        "mtgr_attention_partial_match.csv");
  std::ofstream csv(csv_path, std::ios::out | std::ios::trunc);
  CHECK(csv.is_open()) << "failed to open csv path: " << csv_path;

  csv << "common,,,,,,,one_stage,,,,,,multi_stage,,,,,,speedup,,delta_ms,"
         "precision,\n";
  csv << "mask_build_mode,heads,head_dim,history,real_time,realtime_matched,"
         "target,mask_build_ms,h2d_ms,workspace_ms,fia_ms,device_total_ms,"
         "wall_total_ms,rt_tgt_on_prefix_rt_trapezoid.workspace_ms,"
         "rt_tgt_on_prefix_rt_trapezoid.exec_ms,tgt_diag_update_fused."
         "workspace_ms,"
         "tgt_diag_update_fused.exec_ms,device_total_ms,wall_total_ms,"
         "one_device_total_ms/multi_device_total_ms,"
         "one_wall_total_ms/multi_wall_total_ms,"
         "one_device_total_ms-multi_device_total_ms,"
         "one_wall_total_ms-multi_wall_total_ms,diff_max_abs,diff_mean_abs\n";

  const auto cases =
      build_mtgr_sweep_shapes(full_sweep, /*partial_match=*/true);
  std::fprintf(
      stderr,
      "[MTGR][CUDA][Perf][PartialRT] cases=%zu warmup=%d repeat=%d csv=%s\n",
      cases.size(),
      warmup,
      repeat,
      csv_path.c_str());
  int64_t idx = 0;
  for (const auto& shape : cases) {
    ++idx;
    const auto avg = run_shape(shape, warmup, repeat);
    const double speedup_dev =
        avg.one.device_total_ms / avg.multi.device_total_ms;
    const double speedup_wall = avg.one.wall_total_ms / avg.multi.wall_total_ms;
    const double delta_dev =
        avg.one.device_total_ms - avg.multi.device_total_ms;
    const double delta_wall = avg.one.wall_total_ms - avg.multi.wall_total_ms;

    char row[2048];
    std::snprintf(
        row,
        sizeof(row),
        "device,%lld,%lld,%lld,%lld,%lld,%lld,%.6f,%.6f,%.6f,%.6f,%.6f,"
        "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6e,%.6e",
        static_cast<long long>(shape.heads),
        static_cast<long long>(shape.head_dim),
        static_cast<long long>(shape.history),
        static_cast<long long>(shape.realtime),
        static_cast<long long>(shape.realtime_matched()),
        static_cast<long long>(shape.target),
        avg.one.mask_build_ms,
        avg.one.h2d_ms,
        avg.one.workspace_ms,
        avg.one.fia_ms,
        avg.one.device_total_ms,
        avg.one.wall_total_ms,
        stage_workspace(avg.multi, "rt_tgt_on_prefix_rt_trapezoid"),
        stage_exec(avg.multi, "rt_tgt_on_prefix_rt_trapezoid"),
        stage_workspace(avg.multi, "tgt_diag_update_fused"),
        stage_exec(avg.multi, "tgt_diag_update_fused"),
        avg.multi.device_total_ms,
        avg.multi.wall_total_ms,
        speedup_dev,
        speedup_wall,
        delta_dev,
        delta_wall,
        avg.diff_max_abs,
        avg.diff_mean_abs);
    csv << row << "\n";
    csv.flush();
    std::fprintf(stderr,
                 "[MTGR][CUDA][Perf][PartialRT %lld/%zu] %s\n",
                 static_cast<long long>(idx),
                 cases.size(),
                 row);
  }
  std::fflush(stderr);
}

TEST_F(MTGRAttentionOneVsMultiStagePerfTest, OneVsFusedNoMatchHotShape) {
  torch::NoGradGuard no_grad_guard;
  const int warmup = std::max(0, env_int("XLLM_MTGR_ATTENTION_WARMUP", 5));
  const int repeat = std::max(1, env_int("XLLM_MTGR_ATTENTION_REPEAT", 20));

  MTGRAttentionTestShape shape;
  shape.heads = 8;
  shape.kv_heads = 8;
  shape.head_dim = 128;
  shape.history = 2048;
  shape.context = 8;
  shape.realtime = 512;
  shape.target = 1600;
  shape.matched_prefix = 0;

  const auto avg = run_shape_vs_backend(
      shape, warmup, repeat, MTGRAttentionTestBackend::kFusedNoMatch);
  const double speedup_dev =
      avg.base.device_total_ms / avg.candidate.device_total_ms;
  const double delta_dev =
      avg.base.device_total_ms - avg.candidate.device_total_ms;

  std::fprintf(
      stderr,
      "[MTGR][CUDA][Perf][OneVsFusedHotShape] "
      "heads=%lld head_dim=%lld history=%lld realtime=%lld target=%lld "
      "base_mask_build_ms=%.6f base_device_total_ms=%.6f "
      "fused_device_total_ms=%.6f fused_wall_total_ms=%.6f "
      "speedup=%.6f delta_ms=%.6f diff_max_abs=%.6e diff_mean_abs=%.6e\n",
      static_cast<long long>(shape.heads),
      static_cast<long long>(shape.head_dim),
      static_cast<long long>(shape.history),
      static_cast<long long>(shape.realtime),
      static_cast<long long>(shape.target),
      avg.base.mask_build_ms,
      avg.base.device_total_ms,
      avg.candidate.device_total_ms,
      avg.candidate.wall_total_ms,
      speedup_dev,
      delta_dev,
      avg.diff_max_abs,
      avg.diff_mean_abs);
  for (const auto& stage : avg.candidate.stages) {
    std::fprintf(stderr,
                 "[MTGR][CUDA][Perf][OneVsFusedHotShape][Stage] %s "
                 "workspace_ms=%.6f exec_ms=%.6f host_ms=%.6f\n",
                 stage.name.c_str(),
                 stage.workspace_ms,
                 stage.exec_ms,
                 stage.host_submit_ms);
  }
  std::fflush(stderr);
}

}  // namespace xllm::kernel::cuda::test
