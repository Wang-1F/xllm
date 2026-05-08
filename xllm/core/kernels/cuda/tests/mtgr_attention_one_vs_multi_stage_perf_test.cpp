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

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <torch/cuda.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/common/global_flags.h"
#include "layers/cuda/mtgr_attention.h"
#include "mtgr_attenion_test.h"

namespace xllm::kernel::cuda::test {

using LayerMTGRAttentionBackend = xllm::layer::MTGRAttentionBackend;
using LayerMTGRAttentionImpl = xllm::layer::MTGRAttentionImpl;
using LayerMTGRAttentionMetrics = xllm::layer::MTGRAttentionMetrics;

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

namespace {

constexpr int64_t kBlockSize = 128;

struct AvgMetrics {
  LayerMTGRAttentionMetrics one;
  LayerMTGRAttentionMetrics multi;
  double diff_max_abs = 0.0;
  double diff_mean_abs = 0.0;
};

struct CompareMetrics {
  LayerMTGRAttentionMetrics base;
  LayerMTGRAttentionMetrics candidate;
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
        shape, warmup, repeat, LayerMTGRAttentionBackend::kMultiStage);
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
      LayerMTGRAttentionBackend candidate_backend) {
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

    LayerMTGRAttentionImpl one_stage(shape.heads,
                                     shape.head_dim,
                                     scale,
                                     shape.kv_heads,
                                     LayerMTGRAttentionBackend::kOneStage);
    LayerMTGRAttentionImpl multi_stage(
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

  LayerMTGRAttentionMetrics run_shape_fused_only(
      const MTGRAttentionTestShape& shape,
      int warmup,
      int repeat) {
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
    auto kv_cache =
        make_mtgr_kv_cache(shape, device_, torch::kFloat16, kBlockSize);

    LayerMTGRAttentionImpl fused(shape.heads,
                                 shape.head_dim,
                                 scale,
                                 shape.kv_heads,
                                 LayerMTGRAttentionBackend::kFused);

    for (int i = 0; i < warmup; ++i) {
      (void)std::get<0>(fused.forward(metadata, query, key, value, kv_cache));
    }
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

    LayerMTGRAttentionMetrics avg;
    for (int i = 0; i < repeat; ++i) {
      auto output =
          std::get<0>(fused.forward(metadata, query, key, value, kv_cache));
      CHECK(torch::isfinite(output).all().item<bool>());
      const auto metrics = fused.last_metrics();

      avg.workspace_ms += metrics.workspace_ms;
      avg.device_total_ms += metrics.device_total_ms;
      avg.wall_total_ms += metrics.wall_total_ms;
      if (avg.stages.empty()) {
        avg.stages = metrics.stages;
      } else {
        for (size_t s = 0; s < avg.stages.size(); ++s) {
          avg.stages[s].workspace_ms += metrics.stages[s].workspace_ms;
          avg.stages[s].exec_ms += metrics.stages[s].exec_ms;
          avg.stages[s].host_submit_ms += metrics.stages[s].host_submit_ms;
        }
      }
    }

    const double inv = 1.0 / static_cast<double>(repeat);
    avg.workspace_ms *= inv;
    avg.device_total_ms *= inv;
    avg.wall_total_ms *= inv;
    for (auto& stage : avg.stages) {
      stage.workspace_ms *= inv;
      stage.exec_ms *= inv;
      stage.host_submit_ms *= inv;
    }
    return avg;
  }

  LayerMTGRAttentionMetrics run_shape_fused_main_kernel_only(
      const MTGRAttentionTestShape& shape,
      int warmup,
      int repeat) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat16).device(device_);
    const int64_t total = shape.total_len();
    const float scale = 1.0f / std::sqrt(static_cast<float>(shape.head_dim));

    auto query_snd =
        (torch::randn({total, shape.heads, shape.head_dim}, opts) * 0.05)
            .contiguous();
    auto key_snd =
        (torch::randn({total, shape.kv_heads, shape.head_dim}, opts) * 0.05)
            .contiguous();
    auto value_snd =
        (torch::randn({total, shape.kv_heads, shape.head_dim}, opts) * 0.05)
            .contiguous();
    auto output_snd = torch::empty_like(query_snd);
    auto target_hcr_lse = torch::empty(
        {shape.target, shape.heads, 1},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_));

    c10::cuda::CUDAGuard guard(device_);
    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_end = nullptr;
    CHECK_EQ(cudaEventCreate(&ev_start), cudaSuccess);
    CHECK_EQ(cudaEventCreate(&ev_end), cudaSuccess);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();

    for (int i = 0; i < warmup; ++i) {
      mtgr_fused_no_match_attention_cuda(query_snd,
                                         key_snd,
                                         value_snd,
                                         shape.history,
                                         shape.context,
                                         shape.realtime,
                                         shape.target,
                                         scale,
                                         output_snd,
                                         target_hcr_lse);
    }
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

    LayerMTGRAttentionMetrics avg;
    avg.stages = {
        xllm::layer::MTGRStageMetric{.name = "mtgr_fused_no_match_attention"}};
    for (int i = 0; i < repeat; ++i) {
      const auto wall_start = std::chrono::steady_clock::now();
      CHECK_EQ(cudaEventRecord(ev_start, stream), cudaSuccess);
      mtgr_fused_no_match_attention_cuda(query_snd,
                                         key_snd,
                                         value_snd,
                                         shape.history,
                                         shape.context,
                                         shape.realtime,
                                         shape.target,
                                         scale,
                                         output_snd,
                                         target_hcr_lse);
      CHECK_EQ(cudaEventRecord(ev_end, stream), cudaSuccess);
      CHECK_EQ(cudaEventSynchronize(ev_end), cudaSuccess);
      float ms = 0.0f;
      CHECK_EQ(cudaEventElapsedTime(&ms, ev_start, ev_end), cudaSuccess);
      avg.device_total_ms += static_cast<double>(ms);
      avg.stages[0].exec_ms += static_cast<double>(ms);
      avg.wall_total_ms += std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - wall_start)
                               .count();
      CHECK(torch::isfinite(output_snd).all().item<bool>());
      CHECK(torch::isfinite(target_hcr_lse).all().item<bool>());
    }

    const double inv = 1.0 / static_cast<double>(repeat);
    avg.device_total_ms *= inv;
    avg.wall_total_ms *= inv;
    avg.stages[0].exec_ms *= inv;
    CHECK_EQ(cudaEventDestroy(ev_start), cudaSuccess);
    CHECK_EQ(cudaEventDestroy(ev_end), cudaSuccess);
    return avg;
  }

  torch::Device device_ = torch::Device(torch::kCPU);
};

const xllm::layer::MTGRStageMetric* find_stage(
    const LayerMTGRAttentionMetrics& metrics,
    const char* name) {
  for (const auto& stage : metrics.stages) {
    if (stage.name == name) {
      return &stage;
    }
  }
  return nullptr;
}

double stage_workspace(const LayerMTGRAttentionMetrics& metrics,
                       const char* name) {
  const auto* stage = find_stage(metrics, name);
  return stage != nullptr ? stage->workspace_ms : 0.0;
}

double stage_exec(const LayerMTGRAttentionMetrics& metrics, const char* name) {
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

int64_t sample_non_aligned_len(std::mt19937_64* rng, int64_t lo, int64_t hi) {
  CHECK(rng != nullptr);
  CHECK_LE(lo, hi);
  std::uniform_int_distribution<int64_t> dist(lo, hi);
  for (int attempt = 0; attempt < 1024; ++attempt) {
    const int64_t v = dist(*rng);
    if ((v % 32) != 0 && (v % 64) != 0) {
      return v;
    }
  }
  int64_t v = dist(*rng);
  if ((v % 32) != 0 && (v % 64) != 0) {
    return v;
  }
  if (v < hi) {
    ++v;
  } else if (v > lo) {
    --v;
  }
  return v;
}

MTGRAttentionTestShape make_no_match_counterpart(
    const MTGRAttentionTestShape& shape) {
  auto no_match = shape;
  no_match.matched_prefix = 0;
  return no_match;
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
  int64_t skipped = 0;
  for (const auto& shape : cases) {
    ++idx;
    AvgMetrics avg;
    try {
      avg = run_shape(shape, warmup, repeat);
    } catch (const std::exception& e) {
      ++skipped;
      std::fprintf(stderr,
                   "[MTGR][CUDA][Perf][NoMatch %lld/%zu] skipped: heads=%lld "
                   "head_dim=%lld history=%lld realtime=%lld target=%lld "
                   "reason=%s\n",
                   static_cast<long long>(idx),
                   cases.size(),
                   static_cast<long long>(shape.heads),
                   static_cast<long long>(shape.head_dim),
                   static_cast<long long>(shape.history),
                   static_cast<long long>(shape.realtime),
                   static_cast<long long>(shape.target),
                   e.what());
      continue;
    }
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
  std::fprintf(stderr,
               "[MTGR][CUDA][Perf][NoMatch] completed=%zu skipped=%lld\n",
               cases.size() - static_cast<size_t>(skipped),
               static_cast<long long>(skipped));
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
  int64_t skipped = 0;
  for (const auto& shape : cases) {
    ++idx;
    AvgMetrics avg;
    try {
      avg = run_shape(shape, warmup, repeat);
    } catch (const std::exception& e) {
      ++skipped;
      std::fprintf(
          stderr,
          "[MTGR][CUDA][Perf][PartialRT %lld/%zu] skipped: heads=%lld "
          "head_dim=%lld history=%lld realtime=%lld target=%lld reason=%s\n",
          static_cast<long long>(idx),
          cases.size(),
          static_cast<long long>(shape.heads),
          static_cast<long long>(shape.head_dim),
          static_cast<long long>(shape.history),
          static_cast<long long>(shape.realtime),
          static_cast<long long>(shape.target),
          e.what());
      continue;
    }
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
  std::fprintf(stderr,
               "[MTGR][CUDA][Perf][PartialRT] completed=%zu skipped=%lld\n",
               cases.size() - static_cast<size_t>(skipped),
               static_cast<long long>(skipped));
  std::fflush(stderr);
}

TEST_F(MTGRAttentionOneVsMultiStagePerfTest, FusedNoMatchSweepCsv) {
  torch::NoGradGuard no_grad_guard;
  const bool full_sweep =
      env_flag_enabled("XLLM_MTGR_ATTENTION_FULL_SWEEP", true);
  const int warmup = std::max(0, env_int("XLLM_MTGR_ATTENTION_WARMUP", 5));
  const int repeat = std::max(1, env_int("XLLM_MTGR_ATTENTION_REPEAT", 20));
  const std::string csv_path =
      csv_path_from_env("XLLM_MTGR_ATTENTION_FUSED_NO_MATCH_CSV",
                        "mtgr_attention_fused_no_match.csv");
  std::ofstream csv(csv_path, std::ios::out | std::ios::trunc);
  CHECK(csv.is_open()) << "failed to open csv path: " << csv_path;

  csv << "common,,,,,,,one_stage,,,,,,fused,,,,,,speedup,,delta_ms,precision,"
         "\n";
  csv << "mask_build_mode,heads,head_dim,history,real_time,target,"
         "mask_build_ms,mask_build_plus_device_ms,h2d_ms,workspace_ms,fia_ms,"
         "device_total_ms,wall_total_ms,"
         "mtgr_fused_no_match_attention.workspace_ms,"
         "mtgr_fused_no_match_attention.exec_ms,"
         "tgt_diag_update_fused.workspace_ms,tgt_diag_update_fused.exec_ms,"
         "device_total_ms,wall_total_ms,"
         "one_mask_build_plus_device_ms/fused_device_total_ms,"
         "one_wall_total_ms/fused_wall_total_ms,"
         "one_mask_build_plus_device_ms-fused_device_total_ms,"
         "one_wall_total_ms-fused_wall_total_ms,diff_max_abs,diff_mean_abs\n";

  const auto cases =
      build_mtgr_sweep_shapes(full_sweep, /*partial_match=*/false);
  std::fprintf(
      stderr,
      "[MTGR][CUDA][Perf][FusedNoMatch] cases=%zu warmup=%d repeat=%d csv=%s\n",
      cases.size(),
      warmup,
      repeat,
      csv_path.c_str());
  int64_t idx = 0;
  int64_t skipped = 0;
  for (const auto& shape : cases) {
    ++idx;
    CompareMetrics avg;
    try {
      avg = run_shape_vs_backend(
          shape, warmup, repeat, LayerMTGRAttentionBackend::kFused);
    } catch (const std::exception& e) {
      ++skipped;
      std::fprintf(
          stderr,
          "[MTGR][CUDA][Perf][FusedNoMatch %lld/%zu] skipped: heads=%lld "
          "head_dim=%lld history=%lld realtime=%lld target=%lld reason=%s\n",
          static_cast<long long>(idx),
          cases.size(),
          static_cast<long long>(shape.heads),
          static_cast<long long>(shape.head_dim),
          static_cast<long long>(shape.history),
          static_cast<long long>(shape.realtime),
          static_cast<long long>(shape.target),
          e.what());
      continue;
    }
    const double base_mask_build_plus_device_ms =
        avg.base.mask_build_ms + avg.base.device_total_ms;
    const double speedup_dev =
        base_mask_build_plus_device_ms / avg.candidate.device_total_ms;
    const double speedup_wall =
        avg.base.wall_total_ms / avg.candidate.wall_total_ms;
    const double delta_dev =
        base_mask_build_plus_device_ms - avg.candidate.device_total_ms;
    const double delta_wall =
        avg.base.wall_total_ms - avg.candidate.wall_total_ms;

    char row[2048];
    std::snprintf(
        row,
        sizeof(row),
        "device,%lld,%lld,%lld,%lld,%lld,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
        "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6e,%.6e",
        static_cast<long long>(shape.heads),
        static_cast<long long>(shape.head_dim),
        static_cast<long long>(shape.history),
        static_cast<long long>(shape.realtime),
        static_cast<long long>(shape.target),
        avg.base.mask_build_ms,
        base_mask_build_plus_device_ms,
        avg.base.h2d_ms,
        avg.base.workspace_ms,
        avg.base.fia_ms,
        avg.base.device_total_ms,
        avg.base.wall_total_ms,
        stage_workspace(avg.candidate, "mtgr_fused_no_match_attention"),
        stage_exec(avg.candidate, "mtgr_fused_no_match_attention"),
        stage_workspace(avg.candidate, "tgt_diag_update_fused"),
        stage_exec(avg.candidate, "tgt_diag_update_fused"),
        avg.candidate.device_total_ms,
        avg.candidate.wall_total_ms,
        speedup_dev,
        speedup_wall,
        delta_dev,
        delta_wall,
        avg.diff_max_abs,
        avg.diff_mean_abs);
    csv << row << "\n";
    csv.flush();
    std::fprintf(stderr,
                 "[MTGR][CUDA][Perf][FusedNoMatch %lld/%zu] %s\n",
                 static_cast<long long>(idx),
                 cases.size(),
                 row);
  }
  std::fprintf(stderr,
               "[MTGR][CUDA][Perf][FusedNoMatch] completed=%zu skipped=%lld\n",
               cases.size() - static_cast<size_t>(skipped),
               static_cast<long long>(skipped));
  std::fflush(stderr);
}

TEST_F(MTGRAttentionOneVsMultiStagePerfTest,
       FusedPartialRealTimeMatchSweepCsv) {
  torch::NoGradGuard no_grad_guard;
  const bool full_sweep =
      env_flag_enabled("XLLM_MTGR_ATTENTION_FULL_SWEEP", true);
  const int warmup = std::max(0, env_int("XLLM_MTGR_ATTENTION_WARMUP", 5));
  const int repeat = std::max(1, env_int("XLLM_MTGR_ATTENTION_REPEAT", 20));
  const std::string csv_path =
      csv_path_from_env("XLLM_MTGR_ATTENTION_FUSED_PARTIAL_MATCH_CSV",
                        "mtgr_attention_fused_partial_match.csv");
  std::ofstream csv(csv_path, std::ios::out | std::ios::trunc);
  CHECK(csv.is_open()) << "failed to open csv path: " << csv_path;

  csv << "common,,,,,,,,one_stage,,,,,,,partial_fused,,,,,,,no_match_fused,,"
         "speedup,,,,delta_ms,,,,precision,\n";
  csv << "mask_build_mode,heads,head_dim,history,real_time,realtime_matched,"
         "target,mask_build_ms,mask_build_plus_device_ms,h2d_ms,workspace_ms,"
         "fia_ms,device_total_ms,wall_total_ms,scatter_kv_cache.workspace_ms,"
         "scatter_kv_cache.exec_ms,mtgr_fused_partial_rt_attention."
         "workspace_ms,mtgr_fused_partial_rt_attention.exec_ms,"
         "tgt_diag_update_fused.workspace_ms,tgt_diag_update_fused.exec_ms,"
         "device_total_ms,wall_total_ms,no_match_device_total_ms,"
         "no_match_wall_total_ms,"
         "one_mask_build_plus_device_ms/partial_fused_device_total_ms,"
         "one_wall_total_ms/partial_fused_wall_total_ms,"
         "no_match_fused_device_total_ms/partial_fused_device_total_ms,"
         "no_match_fused_wall_total_ms/partial_fused_wall_total_ms,"
         "one_mask_build_plus_device_ms-partial_fused_device_total_ms,"
         "one_wall_total_ms-partial_fused_wall_total_ms,"
         "no_match_fused_device_total_ms-partial_fused_device_total_ms,"
         "no_match_fused_wall_total_ms-partial_fused_wall_total_ms,"
         "diff_max_abs,diff_mean_abs\n";

  const auto cases =
      build_mtgr_sweep_shapes(full_sweep, /*partial_match=*/true);
  std::fprintf(stderr,
               "[MTGR][CUDA][Perf][FusedPartialRT] cases=%zu warmup=%d "
               "repeat=%d csv=%s\n",
               cases.size(),
               warmup,
               repeat,
               csv_path.c_str());
  int64_t idx = 0;
  int64_t skipped = 0;
  for (const auto& shape : cases) {
    ++idx;
    CompareMetrics avg;
    LayerMTGRAttentionMetrics no_match_fused;
    try {
      avg = run_shape_vs_backend(
          shape, warmup, repeat, LayerMTGRAttentionBackend::kFused);
      no_match_fused = run_shape_fused_only(
          make_no_match_counterpart(shape), warmup, repeat);
    } catch (const std::exception& e) {
      ++skipped;
      std::fprintf(
          stderr,
          "[MTGR][CUDA][Perf][FusedPartialRT %lld/%zu] skipped: heads=%lld "
          "head_dim=%lld history=%lld realtime=%lld target=%lld reason=%s\n",
          static_cast<long long>(idx),
          cases.size(),
          static_cast<long long>(shape.heads),
          static_cast<long long>(shape.head_dim),
          static_cast<long long>(shape.history),
          static_cast<long long>(shape.realtime),
          static_cast<long long>(shape.target),
          e.what());
      continue;
    }
    const double base_mask_build_plus_device_ms =
        avg.base.mask_build_ms + avg.base.device_total_ms;
    const double speedup_dev =
        base_mask_build_plus_device_ms / avg.candidate.device_total_ms;
    const double speedup_wall =
        avg.base.wall_total_ms / avg.candidate.wall_total_ms;
    const double partial_vs_no_match_speedup_dev =
        no_match_fused.device_total_ms / avg.candidate.device_total_ms;
    const double partial_vs_no_match_speedup_wall =
        no_match_fused.wall_total_ms / avg.candidate.wall_total_ms;
    const double delta_dev =
        base_mask_build_plus_device_ms - avg.candidate.device_total_ms;
    const double delta_wall =
        avg.base.wall_total_ms - avg.candidate.wall_total_ms;
    const double partial_vs_no_match_delta_dev =
        no_match_fused.device_total_ms - avg.candidate.device_total_ms;
    const double partial_vs_no_match_delta_wall =
        no_match_fused.wall_total_ms - avg.candidate.wall_total_ms;

    std::ostringstream row;
    row << std::fixed << std::setprecision(6) << "device," << shape.heads << ","
        << shape.head_dim << "," << shape.history << "," << shape.realtime
        << "," << shape.realtime_matched() << "," << shape.target << ","
        << avg.base.mask_build_ms << "," << base_mask_build_plus_device_ms
        << "," << avg.base.h2d_ms << "," << avg.base.workspace_ms << ","
        << avg.base.fia_ms << "," << avg.base.device_total_ms << ","
        << avg.base.wall_total_ms << ","
        << stage_workspace(avg.candidate, "scatter_kv_cache") << ","
        << stage_exec(avg.candidate, "scatter_kv_cache") << ","
        << stage_workspace(avg.candidate, "mtgr_fused_partial_rt_attention")
        << "," << stage_exec(avg.candidate, "mtgr_fused_partial_rt_attention")
        << "," << stage_workspace(avg.candidate, "tgt_diag_update_fused") << ","
        << stage_exec(avg.candidate, "tgt_diag_update_fused") << ","
        << avg.candidate.device_total_ms << "," << avg.candidate.wall_total_ms
        << "," << no_match_fused.device_total_ms << ","
        << no_match_fused.wall_total_ms << "," << speedup_dev << ","
        << speedup_wall << "," << partial_vs_no_match_speedup_dev << ","
        << partial_vs_no_match_speedup_wall << "," << delta_dev << ","
        << delta_wall << "," << partial_vs_no_match_delta_dev << ","
        << partial_vs_no_match_delta_wall << "," << avg.diff_max_abs << ","
        << avg.diff_mean_abs;
    csv << row.str() << "\n";
    csv.flush();
    std::fprintf(stderr,
                 "[MTGR][CUDA][Perf][FusedPartialRT %lld/%zu] %s\n",
                 static_cast<long long>(idx),
                 cases.size(),
                 row.str().c_str());
  }
  std::fprintf(
      stderr,
      "[MTGR][CUDA][Perf][FusedPartialRT] completed=%zu skipped=%lld\n",
      cases.size() - static_cast<size_t>(skipped),
      static_cast<long long>(skipped));
  std::fflush(stderr);
}

TEST_F(MTGRAttentionOneVsMultiStagePerfTest, FusedNoMatchOddLengthRandomCsv) {
  torch::NoGradGuard no_grad_guard;
  const int warmup = std::max(0, env_int("XLLM_MTGR_ATTENTION_WARMUP", 2));
  const int repeat = std::max(1, env_int("XLLM_MTGR_ATTENTION_REPEAT", 5));
  const int case_count =
      std::max(1, env_int("XLLM_MTGR_ATTENTION_ODD_RANDOM_CASES", 50));
  const int seed = env_int("XLLM_MTGR_ATTENTION_ODD_RANDOM_SEED", 20260430);
  const std::string csv_path =
      csv_path_from_env("XLLM_MTGR_ATTENTION_FUSED_ODD_RANDOM_CSV",
                        "mtgr_attention_fused_odd_random.csv");
  std::ofstream csv(csv_path, std::ios::out | std::ios::trunc);
  CHECK(csv.is_open()) << "failed to open csv path: " << csv_path;

  csv << "common,,,,,,fused,,,,,,,,,\n";
  csv << "shape_id,heads,head_dim,history,real_time,target,"
         "device_total_ms,wall_total_ms,"
         "mtgr_fused_no_match_attention.workspace_ms,"
         "mtgr_fused_no_match_attention.exec_ms,"
         "mtgr_fused_no_match_attention.host_ms,"
         "tgt_diag_update_fused.workspace_ms,"
         "tgt_diag_update_fused.exec_ms,"
         "tgt_diag_update_fused.host_ms\n";

  std::mt19937_64 rng(static_cast<uint64_t>(seed));
  const std::vector<int64_t> heads_all = {4, 8, 12};
  const std::vector<int64_t> head_dims_all = {64, 128};
  std::uniform_int_distribution<int> heads_dist(
      0, static_cast<int>(heads_all.size() - 1));
  std::uniform_int_distribution<int> head_dim_dist(
      0, static_cast<int>(head_dims_all.size() - 1));

  std::vector<MTGRAttentionTestShape> cases;
  cases.reserve(static_cast<size_t>(case_count));
  std::unordered_set<std::string> seen;
  for (int attempt = 0;
       attempt < case_count * 32 && static_cast<int>(cases.size()) < case_count;
       ++attempt) {
    MTGRAttentionTestShape shape;
    shape.heads = heads_all[heads_dist(rng)];
    shape.kv_heads = shape.heads;
    shape.head_dim = head_dims_all[head_dim_dist(rng)];
    shape.history = sample_non_aligned_len(&rng, 1350, 4096);
    shape.context = 8;
    shape.realtime = sample_non_aligned_len(&rng, 100, 600);
    shape.target = sample_non_aligned_len(&rng, 800, 2400);
    shape.matched_prefix = 0;
    const std::string key =
        std::to_string(shape.heads) + "_" + std::to_string(shape.head_dim) +
        "_" + std::to_string(shape.history) + "_" +
        std::to_string(shape.realtime) + "_" + std::to_string(shape.target);
    if (seen.insert(key).second) {
      cases.push_back(shape);
    }
  }
  CHECK_EQ(static_cast<int>(cases.size()), case_count)
      << "failed to generate enough unique odd-length random cases";

  std::fprintf(stderr,
               "[MTGR][CUDA][Perf][FusedOddRandom] cases=%zu warmup=%d "
               "repeat=%d seed=%d csv=%s\n",
               cases.size(),
               warmup,
               repeat,
               seed,
               csv_path.c_str());

  int64_t idx = 0;
  int64_t skipped = 0;
  double total_device_ms = 0.0;
  double total_wall_ms = 0.0;
  double total_main_exec_ms = 0.0;
  double total_update_exec_ms = 0.0;
  double max_device_ms = 0.0;
  double min_device_ms = std::numeric_limits<double>::infinity();
  for (const auto& shape : cases) {
    ++idx;
    try {
      const auto metrics = run_shape_fused_only(shape, warmup, repeat);
      const double main_ws =
          stage_workspace(metrics, "mtgr_fused_no_match_attention");
      const double main_exec =
          stage_exec(metrics, "mtgr_fused_no_match_attention");
      const double update_ws =
          stage_workspace(metrics, "tgt_diag_update_fused");
      const double update_exec = stage_exec(metrics, "tgt_diag_update_fused");
      const auto* main_stage =
          find_stage(metrics, "mtgr_fused_no_match_attention");
      const auto* update_stage = find_stage(metrics, "tgt_diag_update_fused");
      const double main_host =
          main_stage != nullptr ? main_stage->host_submit_ms : 0.0;
      const double update_host =
          update_stage != nullptr ? update_stage->host_submit_ms : 0.0;

      total_device_ms += metrics.device_total_ms;
      total_wall_ms += metrics.wall_total_ms;
      total_main_exec_ms += main_exec;
      total_update_exec_ms += update_exec;
      max_device_ms = std::max(max_device_ms, metrics.device_total_ms);
      min_device_ms = std::min(min_device_ms, metrics.device_total_ms);

      char row[1024];
      std::snprintf(row,
                    sizeof(row),
                    "%lld,%lld,%lld,%lld,%lld,%lld,%.6f,%.6f,%.6f,%.6f,%.6f,"
                    "%.6f,%.6f,%.6f",
                    static_cast<long long>(idx),
                    static_cast<long long>(shape.heads),
                    static_cast<long long>(shape.head_dim),
                    static_cast<long long>(shape.history),
                    static_cast<long long>(shape.realtime),
                    static_cast<long long>(shape.target),
                    metrics.device_total_ms,
                    metrics.wall_total_ms,
                    main_ws,
                    main_exec,
                    main_host,
                    update_ws,
                    update_exec,
                    update_host);
      csv << row << "\n";
      csv.flush();
      std::fprintf(stderr,
                   "[MTGR][CUDA][Perf][FusedOddRandom %lld/%zu] %s\n",
                   static_cast<long long>(idx),
                   cases.size(),
                   row);
    } catch (const std::exception& e) {
      ++skipped;
      std::fprintf(stderr,
                   "[MTGR][CUDA][Perf][FusedOddRandom %lld/%zu] skipped: "
                   "heads=%lld head_dim=%lld history=%lld realtime=%lld "
                   "target=%lld reason=%s\n",
                   static_cast<long long>(idx),
                   cases.size(),
                   static_cast<long long>(shape.heads),
                   static_cast<long long>(shape.head_dim),
                   static_cast<long long>(shape.history),
                   static_cast<long long>(shape.realtime),
                   static_cast<long long>(shape.target),
                   e.what());
    }
  }

  const int64_t completed = static_cast<int64_t>(cases.size()) - skipped;
  if (completed > 0) {
    const double inv = 1.0 / static_cast<double>(completed);
    std::fprintf(
        stderr,
        "[MTGR][CUDA][Perf][FusedOddRandom] completed=%lld skipped=%lld "
        "avg_device_total_ms=%.6f avg_wall_total_ms=%.6f "
        "avg_main_exec_ms=%.6f avg_update_exec_ms=%.6f "
        "min_device_total_ms=%.6f max_device_total_ms=%.6f\n",
        static_cast<long long>(completed),
        static_cast<long long>(skipped),
        total_device_ms * inv,
        total_wall_ms * inv,
        total_main_exec_ms * inv,
        total_update_exec_ms * inv,
        min_device_ms,
        max_device_ms);
  } else {
    std::fprintf(
        stderr,
        "[MTGR][CUDA][Perf][FusedOddRandom] completed=0 skipped=%lld\n",
        static_cast<long long>(skipped));
  }
  std::fflush(stderr);
}

TEST_F(MTGRAttentionOneVsMultiStagePerfTest,
       FusedMainKernelOddLengthRandomCsv) {
  torch::NoGradGuard no_grad_guard;
  const int warmup = std::max(0, env_int("XLLM_MTGR_ATTENTION_WARMUP", 2));
  const int repeat = std::max(1, env_int("XLLM_MTGR_ATTENTION_REPEAT", 5));
  const int case_count =
      std::max(1, env_int("XLLM_MTGR_ATTENTION_ODD_RANDOM_CASES", 50));
  const int seed = env_int("XLLM_MTGR_ATTENTION_ODD_RANDOM_SEED", 20260430);
  const std::string csv_path =
      csv_path_from_env("XLLM_MTGR_ATTENTION_FUSED_MAIN_ODD_RANDOM_CSV",
                        "mtgr_attention_fused_main_odd_random.csv");
  std::ofstream csv(csv_path, std::ios::out | std::ios::trunc);
  CHECK(csv.is_open()) << "failed to open csv path: " << csv_path;

  csv << "shape_id,heads,head_dim,history,real_time,target,"
         "device_total_ms,wall_total_ms,mtgr_fused_no_match_attention.exec_"
         "ms\n";

  std::mt19937_64 rng(static_cast<uint64_t>(seed));
  const std::vector<int64_t> heads_all = {4, 8, 12};
  const std::vector<int64_t> head_dims_all = {64, 128};
  std::uniform_int_distribution<int> heads_dist(
      0, static_cast<int>(heads_all.size() - 1));
  std::uniform_int_distribution<int> head_dim_dist(
      0, static_cast<int>(head_dims_all.size() - 1));

  std::vector<MTGRAttentionTestShape> cases;
  cases.reserve(static_cast<size_t>(case_count));
  std::unordered_set<std::string> seen;
  for (int attempt = 0;
       attempt < case_count * 32 && static_cast<int>(cases.size()) < case_count;
       ++attempt) {
    MTGRAttentionTestShape shape;
    shape.heads = heads_all[heads_dist(rng)];
    shape.kv_heads = shape.heads;
    shape.head_dim = head_dims_all[head_dim_dist(rng)];
    shape.history = sample_non_aligned_len(&rng, 1350, 4096);
    shape.context = 8;
    shape.realtime = sample_non_aligned_len(&rng, 100, 600);
    shape.target = sample_non_aligned_len(&rng, 800, 2400);
    shape.matched_prefix = 0;
    const std::string key =
        std::to_string(shape.heads) + "_" + std::to_string(shape.head_dim) +
        "_" + std::to_string(shape.history) + "_" +
        std::to_string(shape.realtime) + "_" + std::to_string(shape.target);
    if (seen.insert(key).second) {
      cases.push_back(shape);
    }
  }
  CHECK_EQ(static_cast<int>(cases.size()), case_count)
      << "failed to generate enough unique odd-length random cases";

  std::fprintf(stderr,
               "[MTGR][CUDA][Perf][FusedMainOddRandom] cases=%zu warmup=%d "
               "repeat=%d seed=%d csv=%s\n",
               cases.size(),
               warmup,
               repeat,
               seed,
               csv_path.c_str());

  int64_t idx = 0;
  int64_t skipped = 0;
  double total_device_ms = 0.0;
  double total_wall_ms = 0.0;
  double max_device_ms = 0.0;
  double min_device_ms = std::numeric_limits<double>::infinity();
  for (const auto& shape : cases) {
    ++idx;
    try {
      const auto metrics =
          run_shape_fused_main_kernel_only(shape, warmup, repeat);
      total_device_ms += metrics.device_total_ms;
      total_wall_ms += metrics.wall_total_ms;
      max_device_ms = std::max(max_device_ms, metrics.device_total_ms);
      min_device_ms = std::min(min_device_ms, metrics.device_total_ms);

      char row[512];
      std::snprintf(row,
                    sizeof(row),
                    "%lld,%lld,%lld,%lld,%lld,%lld,%.6f,%.6f,%.6f",
                    static_cast<long long>(idx),
                    static_cast<long long>(shape.heads),
                    static_cast<long long>(shape.head_dim),
                    static_cast<long long>(shape.history),
                    static_cast<long long>(shape.realtime),
                    static_cast<long long>(shape.target),
                    metrics.device_total_ms,
                    metrics.wall_total_ms,
                    stage_exec(metrics, "mtgr_fused_no_match_attention"));
      csv << row << "\n";
      csv.flush();
      std::fprintf(stderr,
                   "[MTGR][CUDA][Perf][FusedMainOddRandom %lld/%zu] %s\n",
                   static_cast<long long>(idx),
                   cases.size(),
                   row);
    } catch (const std::exception& e) {
      ++skipped;
      std::fprintf(stderr,
                   "[MTGR][CUDA][Perf][FusedMainOddRandom %lld/%zu] skipped: "
                   "heads=%lld head_dim=%lld history=%lld realtime=%lld "
                   "target=%lld reason=%s\n",
                   static_cast<long long>(idx),
                   cases.size(),
                   static_cast<long long>(shape.heads),
                   static_cast<long long>(shape.head_dim),
                   static_cast<long long>(shape.history),
                   static_cast<long long>(shape.realtime),
                   static_cast<long long>(shape.target),
                   e.what());
    }
  }

  const int64_t completed = static_cast<int64_t>(cases.size()) - skipped;
  if (completed > 0) {
    const double inv = 1.0 / static_cast<double>(completed);
    std::fprintf(stderr,
                 "[MTGR][CUDA][Perf][FusedMainOddRandom] completed=%lld "
                 "skipped=%lld avg_device_total_ms=%.6f avg_wall_total_ms=%.6f "
                 "min_device_total_ms=%.6f max_device_total_ms=%.6f\n",
                 static_cast<long long>(completed),
                 static_cast<long long>(skipped),
                 total_device_ms * inv,
                 total_wall_ms * inv,
                 min_device_ms,
                 max_device_ms);
  } else {
    std::fprintf(stderr,
                 "[MTGR][CUDA][Perf][FusedMainOddRandom] completed=0 "
                 "skipped=%lld\n",
                 static_cast<long long>(skipped));
  }
  std::fflush(stderr);
}

TEST_F(MTGRAttentionOneVsMultiStagePerfTest,
       FusedPartialRealTimeMatchOddLengthRandomCsv) {
  torch::NoGradGuard no_grad_guard;
  const int warmup = std::max(0, env_int("XLLM_MTGR_ATTENTION_WARMUP", 2));
  const int repeat = std::max(1, env_int("XLLM_MTGR_ATTENTION_REPEAT", 5));
  const int case_count =
      std::max(1, env_int("XLLM_MTGR_ATTENTION_ODD_RANDOM_CASES", 50));
  const int seed = env_int("XLLM_MTGR_ATTENTION_ODD_RANDOM_SEED", 20260501);
  const std::string csv_path =
      csv_path_from_env("XLLM_MTGR_ATTENTION_FUSED_PARTIAL_ODD_RANDOM_CSV",
                        "mtgr_attention_fused_partial_odd_random.csv");
  std::ofstream csv(csv_path, std::ios::out | std::ios::trunc);
  CHECK(csv.is_open()) << "failed to open csv path: " << csv_path;

  csv << "common,,,,,,,,one_stage,,,,,,,partial_fused,,,,,,,no_match_fused,,"
         "speedup,,,,delta_ms,,,,precision,\n";
  csv << "shape_id,heads,head_dim,history,real_time,realtime_matched,target,"
         "mask_build_ms,mask_build_plus_device_ms,h2d_ms,workspace_ms,"
         "fia_ms,device_total_ms,wall_total_ms,scatter_kv_cache.workspace_ms,"
         "scatter_kv_cache.exec_ms,mtgr_fused_partial_rt_attention."
         "workspace_ms,mtgr_fused_partial_rt_attention.exec_ms,"
         "tgt_diag_update_fused.workspace_ms,tgt_diag_update_fused.exec_ms,"
         "device_total_ms,wall_total_ms,no_match_device_total_ms,"
         "no_match_wall_total_ms,"
         "one_mask_build_plus_device_ms/partial_fused_device_total_ms,"
         "one_wall_total_ms/partial_fused_wall_total_ms,"
         "no_match_fused_device_total_ms/partial_fused_device_total_ms,"
         "no_match_fused_wall_total_ms/partial_fused_wall_total_ms,"
         "one_mask_build_plus_device_ms-partial_fused_device_total_ms,"
         "one_wall_total_ms-partial_fused_wall_total_ms,"
         "no_match_fused_device_total_ms-partial_fused_device_total_ms,"
         "no_match_fused_wall_total_ms-partial_fused_wall_total_ms,"
         "diff_max_abs,diff_mean_abs\n";

  std::mt19937_64 rng(static_cast<uint64_t>(seed));
  const std::vector<int64_t> heads_all = {4, 8, 12};
  const std::vector<int64_t> head_dims_all = {64, 128};
  std::uniform_int_distribution<int> heads_dist(
      0, static_cast<int>(heads_all.size() - 1));
  std::uniform_int_distribution<int> head_dim_dist(
      0, static_cast<int>(head_dims_all.size() - 1));

  std::vector<MTGRAttentionTestShape> cases;
  cases.reserve(static_cast<size_t>(case_count));
  std::unordered_set<std::string> seen;
  for (int attempt = 0;
       attempt < case_count * 32 && static_cast<int>(cases.size()) < case_count;
       ++attempt) {
    MTGRAttentionTestShape shape;
    shape.heads = heads_all[heads_dist(rng)];
    shape.kv_heads = shape.heads;
    shape.head_dim = head_dims_all[head_dim_dist(rng)];
    shape.history = sample_non_aligned_len(&rng, 1350, 4096);
    shape.context = 8;
    shape.realtime = sample_non_aligned_len(&rng, 100, 600);
    shape.target = sample_non_aligned_len(&rng, 800, 2400);
    shape.matched_prefix =
        shape.history + shape.context + (shape.realtime * 4) / 5;
    const std::string key =
        std::to_string(shape.heads) + "_" + std::to_string(shape.head_dim) +
        "_" + std::to_string(shape.history) + "_" +
        std::to_string(shape.realtime) + "_" + std::to_string(shape.target);
    if (seen.insert(key).second) {
      cases.push_back(shape);
    }
  }
  CHECK_EQ(static_cast<int>(cases.size()), case_count)
      << "failed to generate enough unique odd-length random cases";

  std::fprintf(
      stderr,
      "[MTGR][CUDA][Perf][FusedPartialRTOddRandom] cases=%zu warmup=%d "
      "repeat=%d seed=%d csv=%s\n",
      cases.size(),
      warmup,
      repeat,
      seed,
      csv_path.c_str());

  int64_t idx = 0;
  int64_t skipped = 0;
  double total_base_mask_plus_device_ms = 0.0;
  double total_fused_device_ms = 0.0;
  double total_speedup_dev = 0.0;
  double min_speedup_dev = std::numeric_limits<double>::infinity();
  double max_speedup_dev = 0.0;
  int64_t regressions = 0;
  double total_no_match_fused_device_ms = 0.0;
  double total_partial_vs_no_match_speedup_dev = 0.0;
  double min_partial_vs_no_match_speedup_dev =
      std::numeric_limits<double>::infinity();
  double max_partial_vs_no_match_speedup_dev = 0.0;
  int64_t partial_slower_than_no_match = 0;
  for (const auto& shape : cases) {
    ++idx;
    try {
      const auto avg = run_shape_vs_backend(
          shape, warmup, repeat, LayerMTGRAttentionBackend::kFused);
      const auto no_match_fused = run_shape_fused_only(
          make_no_match_counterpart(shape), warmup, repeat);
      const double base_mask_build_plus_device_ms =
          avg.base.mask_build_ms + avg.base.device_total_ms;
      const double speedup_dev =
          base_mask_build_plus_device_ms / avg.candidate.device_total_ms;
      const double speedup_wall =
          avg.base.wall_total_ms / avg.candidate.wall_total_ms;
      const double partial_vs_no_match_speedup_dev =
          no_match_fused.device_total_ms / avg.candidate.device_total_ms;
      const double partial_vs_no_match_speedup_wall =
          no_match_fused.wall_total_ms / avg.candidate.wall_total_ms;
      const double delta_dev =
          base_mask_build_plus_device_ms - avg.candidate.device_total_ms;
      const double delta_wall =
          avg.base.wall_total_ms - avg.candidate.wall_total_ms;
      const double partial_vs_no_match_delta_dev =
          no_match_fused.device_total_ms - avg.candidate.device_total_ms;
      const double partial_vs_no_match_delta_wall =
          no_match_fused.wall_total_ms - avg.candidate.wall_total_ms;
      if (speedup_dev < 1.0) {
        ++regressions;
      }
      if (partial_vs_no_match_speedup_dev < 1.0) {
        ++partial_slower_than_no_match;
      }
      total_base_mask_plus_device_ms += base_mask_build_plus_device_ms;
      total_fused_device_ms += avg.candidate.device_total_ms;
      total_speedup_dev += speedup_dev;
      min_speedup_dev = std::min(min_speedup_dev, speedup_dev);
      max_speedup_dev = std::max(max_speedup_dev, speedup_dev);
      total_no_match_fused_device_ms += no_match_fused.device_total_ms;
      total_partial_vs_no_match_speedup_dev += partial_vs_no_match_speedup_dev;
      min_partial_vs_no_match_speedup_dev = std::min(
          min_partial_vs_no_match_speedup_dev, partial_vs_no_match_speedup_dev);
      max_partial_vs_no_match_speedup_dev = std::max(
          max_partial_vs_no_match_speedup_dev, partial_vs_no_match_speedup_dev);

      std::ostringstream row;
      row << std::fixed << std::setprecision(6) << idx << "," << shape.heads
          << "," << shape.head_dim << "," << shape.history << ","
          << shape.realtime << "," << shape.realtime_matched() << ","
          << shape.target << "," << avg.base.mask_build_ms << ","
          << base_mask_build_plus_device_ms << "," << avg.base.h2d_ms << ","
          << avg.base.workspace_ms << "," << avg.base.fia_ms << ","
          << avg.base.device_total_ms << "," << avg.base.wall_total_ms << ","
          << stage_workspace(avg.candidate, "scatter_kv_cache") << ","
          << stage_exec(avg.candidate, "scatter_kv_cache") << ","
          << stage_workspace(avg.candidate, "mtgr_fused_partial_rt_attention")
          << "," << stage_exec(avg.candidate, "mtgr_fused_partial_rt_attention")
          << "," << stage_workspace(avg.candidate, "tgt_diag_update_fused")
          << "," << stage_exec(avg.candidate, "tgt_diag_update_fused") << ","
          << avg.candidate.device_total_ms << "," << avg.candidate.wall_total_ms
          << "," << no_match_fused.device_total_ms << ","
          << no_match_fused.wall_total_ms << "," << speedup_dev << ","
          << speedup_wall << "," << partial_vs_no_match_speedup_dev << ","
          << partial_vs_no_match_speedup_wall << "," << delta_dev << ","
          << delta_wall << "," << partial_vs_no_match_delta_dev << ","
          << partial_vs_no_match_delta_wall << "," << avg.diff_max_abs << ","
          << avg.diff_mean_abs;
      csv << row.str() << "\n";
      csv.flush();
      std::fprintf(stderr,
                   "[MTGR][CUDA][Perf][FusedPartialRTOddRandom %lld/%zu] %s\n",
                   static_cast<long long>(idx),
                   cases.size(),
                   row.str().c_str());
    } catch (const std::exception& e) {
      ++skipped;
      std::fprintf(
          stderr,
          "[MTGR][CUDA][Perf][FusedPartialRTOddRandom %lld/%zu] skipped: "
          "heads=%lld head_dim=%lld history=%lld realtime=%lld target=%lld "
          "reason=%s\n",
          static_cast<long long>(idx),
          cases.size(),
          static_cast<long long>(shape.heads),
          static_cast<long long>(shape.head_dim),
          static_cast<long long>(shape.history),
          static_cast<long long>(shape.realtime),
          static_cast<long long>(shape.target),
          e.what());
    }
  }

  const int64_t completed = static_cast<int64_t>(cases.size()) - skipped;
  if (completed > 0) {
    const double inv = 1.0 / static_cast<double>(completed);
    std::fprintf(
        stderr,
        "[MTGR][CUDA][Perf][FusedPartialRTOddRandom] completed=%lld "
        "skipped=%lld regressions=%lld partial_slower_than_no_match=%lld "
        "avg_base_mask_plus_device_ms=%.6f avg_fused_device_ms=%.6f "
        "avg_no_match_fused_device_ms=%.6f avg_speedup_dev=%.6f "
        "min_speedup_dev=%.6f max_speedup_dev=%.6f "
        "avg_partial_vs_no_match_speedup_dev=%.6f "
        "min_partial_vs_no_match_speedup_dev=%.6f "
        "max_partial_vs_no_match_speedup_dev=%.6f\n",
        static_cast<long long>(completed),
        static_cast<long long>(skipped),
        static_cast<long long>(regressions),
        static_cast<long long>(partial_slower_than_no_match),
        total_base_mask_plus_device_ms * inv,
        total_fused_device_ms * inv,
        total_no_match_fused_device_ms * inv,
        total_speedup_dev * inv,
        min_speedup_dev,
        max_speedup_dev,
        total_partial_vs_no_match_speedup_dev * inv,
        min_partial_vs_no_match_speedup_dev,
        max_partial_vs_no_match_speedup_dev);
  } else {
    std::fprintf(stderr,
                 "[MTGR][CUDA][Perf][FusedPartialRTOddRandom] completed=0 "
                 "skipped=%lld\n",
                 static_cast<long long>(skipped));
  }
  std::fflush(stderr);
}

TEST_F(MTGRAttentionOneVsMultiStagePerfTest,
       NoMatchAndPartialRealTimeMatchOddLengthRandomCsv) {
  torch::NoGradGuard no_grad_guard;
  const int warmup = std::max(0, env_int("XLLM_MTGR_ATTENTION_WARMUP", 2));
  const int repeat = std::max(1, env_int("XLLM_MTGR_ATTENTION_REPEAT", 5));
  const int case_count =
      std::max(1, env_int("XLLM_MTGR_ATTENTION_ODD_RANDOM_CASES", 1000));
  const int seed = env_int("XLLM_MTGR_ATTENTION_ODD_RANDOM_SEED", 20260501);
  const std::string csv_path =
      csv_path_from_env("XLLM_MTGR_ATTENTION_NO_MATCH_PARTIAL_ODD_RANDOM_CSV",
                        "mtgr_attention_no_match_partial_odd_random.csv");
  std::ofstream csv(csv_path, std::ios::out | std::ios::trunc);
  CHECK(csv.is_open()) << "failed to open csv path: " << csv_path;

  csv << "common,,,,,,,full_base,,,no_match_fused,,,,partial_fused,,,,"
         "partial_vs_no_match,,,precision,,\n";
  csv << "shape_id,heads,head_dim,history,real_time,realtime_matched,target,"
         "base_mask_build_ms,base_fia_ms,base_total_ms,"
         "no_match_fused_device_total_ms,no_match_speedup_dev,"
         "no_match_delta_dev,partial_fused_device_total_ms,"
         "partial_speedup_dev,partial_delta_dev,"
         "partial_vs_no_match_speedup_dev,partial_vs_no_match_delta_dev,"
         "no_match_diff_max_abs,no_match_diff_mean_abs,"
         "partial_diff_max_abs,partial_diff_mean_abs\n";

  std::mt19937_64 rng(static_cast<uint64_t>(seed));
  const std::vector<int64_t> heads_all = {4, 8, 12};
  const std::vector<int64_t> head_dims_all = {64, 128};
  std::uniform_int_distribution<int> heads_dist(
      0, static_cast<int>(heads_all.size() - 1));
  std::uniform_int_distribution<int> head_dim_dist(
      0, static_cast<int>(head_dims_all.size() - 1));

  std::vector<MTGRAttentionTestShape> partial_cases;
  partial_cases.reserve(static_cast<size_t>(case_count));
  std::unordered_set<std::string> seen;
  for (int attempt = 0; attempt < case_count * 32 &&
                        static_cast<int>(partial_cases.size()) < case_count;
       ++attempt) {
    MTGRAttentionTestShape shape;
    shape.heads = heads_all[heads_dist(rng)];
    shape.kv_heads = shape.heads;
    shape.head_dim = head_dims_all[head_dim_dist(rng)];
    shape.history = sample_non_aligned_len(&rng, 1350, 4096);
    shape.context = 8;
    shape.realtime = sample_non_aligned_len(&rng, 100, 600);
    shape.target = sample_non_aligned_len(&rng, 800, 2400);
    shape.matched_prefix =
        shape.history + shape.context + (shape.realtime * 4) / 5;
    const std::string key =
        std::to_string(shape.heads) + "_" + std::to_string(shape.head_dim) +
        "_" + std::to_string(shape.history) + "_" +
        std::to_string(shape.realtime) + "_" + std::to_string(shape.target);
    if (seen.insert(key).second) {
      partial_cases.push_back(shape);
    }
  }
  CHECK_EQ(static_cast<int>(partial_cases.size()), case_count)
      << "failed to generate enough unique odd-length random cases";

  std::fprintf(
      stderr,
      "[MTGR][CUDA][Perf][NoMatchPartialOddRandom] cases=%zu warmup=%d "
      "repeat=%d seed=%d csv=%s\n",
      partial_cases.size(),
      warmup,
      repeat,
      seed,
      csv_path.c_str());

  int64_t idx = 0;
  int64_t skipped = 0;
  int64_t no_match_regressions = 0;
  int64_t partial_regressions = 0;
  int64_t partial_slower_than_no_match = 0;
  double total_no_match_speedup_dev = 0.0;
  double total_partial_speedup_dev = 0.0;
  double total_partial_vs_no_match_speedup_dev = 0.0;
  double min_no_match_speedup_dev = std::numeric_limits<double>::infinity();
  double min_partial_speedup_dev = std::numeric_limits<double>::infinity();
  double min_partial_vs_no_match_speedup_dev =
      std::numeric_limits<double>::infinity();
  double max_no_match_speedup_dev = 0.0;
  double max_partial_speedup_dev = 0.0;
  double max_partial_vs_no_match_speedup_dev = 0.0;
  for (const auto& partial_shape : partial_cases) {
    ++idx;
    try {
      const auto no_match_shape = make_no_match_counterpart(partial_shape);
      const auto no_match = run_shape_vs_backend(
          no_match_shape, warmup, repeat, LayerMTGRAttentionBackend::kFused);
      const auto partial = run_shape_vs_backend(
          partial_shape, warmup, repeat, LayerMTGRAttentionBackend::kFused);

      const double full_base_total_ms = no_match.base.device_total_ms;
      const double no_match_speedup_dev =
          full_base_total_ms / no_match.candidate.device_total_ms;
      const double no_match_delta_dev =
          full_base_total_ms - no_match.candidate.device_total_ms;
      const double partial_speedup_dev =
          full_base_total_ms / partial.candidate.device_total_ms;
      const double partial_delta_dev =
          full_base_total_ms - partial.candidate.device_total_ms;

      const double partial_vs_no_match_speedup_dev =
          no_match.candidate.device_total_ms /
          partial.candidate.device_total_ms;
      const double partial_vs_no_match_delta_dev =
          no_match.candidate.device_total_ms -
          partial.candidate.device_total_ms;

      if (no_match_speedup_dev < 1.0) {
        ++no_match_regressions;
      }
      if (partial_speedup_dev < 1.0) {
        ++partial_regressions;
      }
      if (partial_vs_no_match_speedup_dev < 1.0) {
        ++partial_slower_than_no_match;
      }

      total_no_match_speedup_dev += no_match_speedup_dev;
      total_partial_speedup_dev += partial_speedup_dev;
      total_partial_vs_no_match_speedup_dev += partial_vs_no_match_speedup_dev;
      min_no_match_speedup_dev =
          std::min(min_no_match_speedup_dev, no_match_speedup_dev);
      min_partial_speedup_dev =
          std::min(min_partial_speedup_dev, partial_speedup_dev);
      min_partial_vs_no_match_speedup_dev = std::min(
          min_partial_vs_no_match_speedup_dev, partial_vs_no_match_speedup_dev);
      max_no_match_speedup_dev =
          std::max(max_no_match_speedup_dev, no_match_speedup_dev);
      max_partial_speedup_dev =
          std::max(max_partial_speedup_dev, partial_speedup_dev);
      max_partial_vs_no_match_speedup_dev = std::max(
          max_partial_vs_no_match_speedup_dev, partial_vs_no_match_speedup_dev);

      std::ostringstream row;
      row << std::fixed << std::setprecision(6) << idx << ","
          << partial_shape.heads << "," << partial_shape.head_dim << ","
          << partial_shape.history << "," << partial_shape.realtime << ","
          << partial_shape.realtime_matched() << "," << partial_shape.target
          << "," << no_match.base.mask_build_ms << "," << no_match.base.fia_ms
          << "," << full_base_total_ms << ","
          << no_match.candidate.device_total_ms << "," << no_match_speedup_dev
          << "," << no_match_delta_dev << ","
          << partial.candidate.device_total_ms << "," << partial_speedup_dev
          << "," << partial_delta_dev << "," << partial_vs_no_match_speedup_dev
          << "," << partial_vs_no_match_delta_dev << ","
          << no_match.diff_max_abs << "," << no_match.diff_mean_abs << ","
          << partial.diff_max_abs << "," << partial.diff_mean_abs;
      csv << row.str() << "\n";
      csv.flush();
      std::fprintf(stderr,
                   "[MTGR][CUDA][Perf][NoMatchPartialOddRandom %lld/%zu] %s\n",
                   static_cast<long long>(idx),
                   partial_cases.size(),
                   row.str().c_str());
    } catch (const std::exception& e) {
      ++skipped;
      std::fprintf(
          stderr,
          "[MTGR][CUDA][Perf][NoMatchPartialOddRandom %lld/%zu] skipped: "
          "heads=%lld head_dim=%lld history=%lld realtime=%lld target=%lld "
          "reason=%s\n",
          static_cast<long long>(idx),
          partial_cases.size(),
          static_cast<long long>(partial_shape.heads),
          static_cast<long long>(partial_shape.head_dim),
          static_cast<long long>(partial_shape.history),
          static_cast<long long>(partial_shape.realtime),
          static_cast<long long>(partial_shape.target),
          e.what());
    }
  }

  const int64_t completed =
      static_cast<int64_t>(partial_cases.size()) - skipped;
  if (completed > 0) {
    const double inv = 1.0 / static_cast<double>(completed);
    std::fprintf(
        stderr,
        "[MTGR][CUDA][Perf][NoMatchPartialOddRandom] completed=%lld "
        "skipped=%lld no_match_regressions=%lld partial_regressions=%lld "
        "partial_slower_than_no_match=%lld avg_no_match_speedup_dev=%.6f "
        "min_no_match_speedup_dev=%.6f max_no_match_speedup_dev=%.6f "
        "avg_partial_speedup_dev=%.6f min_partial_speedup_dev=%.6f "
        "max_partial_speedup_dev=%.6f "
        "avg_partial_vs_no_match_speedup_dev=%.6f "
        "min_partial_vs_no_match_speedup_dev=%.6f "
        "max_partial_vs_no_match_speedup_dev=%.6f\n",
        static_cast<long long>(completed),
        static_cast<long long>(skipped),
        static_cast<long long>(no_match_regressions),
        static_cast<long long>(partial_regressions),
        static_cast<long long>(partial_slower_than_no_match),
        total_no_match_speedup_dev * inv,
        min_no_match_speedup_dev,
        max_no_match_speedup_dev,
        total_partial_speedup_dev * inv,
        min_partial_speedup_dev,
        max_partial_speedup_dev,
        total_partial_vs_no_match_speedup_dev * inv,
        min_partial_vs_no_match_speedup_dev,
        max_partial_vs_no_match_speedup_dev);
  } else {
    std::fprintf(stderr,
                 "[MTGR][CUDA][Perf][NoMatchPartialOddRandom] completed=0 "
                 "skipped=%lld\n",
                 static_cast<long long>(skipped));
  }
  std::fflush(stderr);
}

TEST_F(MTGRAttentionOneVsMultiStagePerfTest, FusedMainKernelExactShapeOnce) {
  torch::NoGradGuard no_grad_guard;
  const int warmup = std::max(0, env_int("XLLM_MTGR_ATTENTION_WARMUP", 0));
  const int repeat = std::max(1, env_int("XLLM_MTGR_ATTENTION_REPEAT", 1));

  MTGRAttentionTestShape shape;
  shape.heads = env_int("XLLM_MTGR_ATTENTION_HEADS", 4);
  shape.kv_heads = shape.heads;
  shape.head_dim = env_int("XLLM_MTGR_ATTENTION_HEAD_DIM", 128);
  shape.history = env_int("XLLM_MTGR_ATTENTION_HISTORY", 1377);
  shape.context = env_int("XLLM_MTGR_ATTENTION_CONTEXT", 8);
  shape.realtime = env_int("XLLM_MTGR_ATTENTION_REALTIME", 134);
  shape.target = env_int("XLLM_MTGR_ATTENTION_TARGET", 1093);
  shape.matched_prefix = 0;

  const auto metrics = run_shape_fused_main_kernel_only(shape, warmup, repeat);
  std::fprintf(stderr,
               "[MTGR][CUDA][Perf][FusedMainExactShape] heads=%lld "
               "head_dim=%lld history=%lld context=%lld realtime=%lld "
               "target=%lld device_total_ms=%.6f wall_total_ms=%.6f\n",
               static_cast<long long>(shape.heads),
               static_cast<long long>(shape.head_dim),
               static_cast<long long>(shape.history),
               static_cast<long long>(shape.context),
               static_cast<long long>(shape.realtime),
               static_cast<long long>(shape.target),
               metrics.device_total_ms,
               metrics.wall_total_ms);
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
      shape, warmup, repeat, LayerMTGRAttentionBackend::kFused);
  const double base_mask_build_plus_device_ms =
      avg.base.mask_build_ms + avg.base.device_total_ms;
  const double speedup_dev =
      base_mask_build_plus_device_ms / avg.candidate.device_total_ms;
  const double delta_dev =
      base_mask_build_plus_device_ms - avg.candidate.device_total_ms;
  EXPECT_GT(speedup_dev, 1.0);

  std::fprintf(
      stderr,
      "[MTGR][CUDA][Perf][OneVsFusedHotShape] "
      "heads=%lld head_dim=%lld history=%lld realtime=%lld target=%lld "
      "base_mask_build_ms=%.6f base_mask_build_plus_device_ms=%.6f "
      "base_device_total_ms=%.6f "
      "fused_device_total_ms=%.6f fused_wall_total_ms=%.6f "
      "speedup=%.6f delta_ms=%.6f diff_max_abs=%.6e diff_mean_abs=%.6e\n",
      static_cast<long long>(shape.heads),
      static_cast<long long>(shape.head_dim),
      static_cast<long long>(shape.history),
      static_cast<long long>(shape.realtime),
      static_cast<long long>(shape.target),
      avg.base.mask_build_ms,
      base_mask_build_plus_device_ms,
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

TEST_F(MTGRAttentionOneVsMultiStagePerfTest, OneVsFusedNoMatchHd64HotShape) {
  torch::NoGradGuard no_grad_guard;
  const int warmup = std::max(0, env_int("XLLM_MTGR_ATTENTION_WARMUP", 5));
  const int repeat = std::max(1, env_int("XLLM_MTGR_ATTENTION_REPEAT", 20));

  MTGRAttentionTestShape shape;
  shape.heads = 8;
  shape.kv_heads = 8;
  shape.head_dim = 64;
  shape.history = 2048;
  shape.context = 8;
  shape.realtime = 512;
  shape.target = 1600;
  shape.matched_prefix = 0;

  const auto avg = run_shape_vs_backend(
      shape, warmup, repeat, LayerMTGRAttentionBackend::kFused);
  const double base_mask_build_plus_device_ms =
      avg.base.mask_build_ms + avg.base.device_total_ms;
  const double speedup_dev =
      base_mask_build_plus_device_ms / avg.candidate.device_total_ms;
  const double delta_dev =
      base_mask_build_plus_device_ms - avg.candidate.device_total_ms;
  EXPECT_GT(speedup_dev, 1.0);

  std::fprintf(
      stderr,
      "[MTGR][CUDA][Perf][OneVsFusedHd64HotShape] "
      "heads=%lld head_dim=%lld history=%lld realtime=%lld target=%lld "
      "base_mask_build_ms=%.6f base_mask_build_plus_device_ms=%.6f "
      "base_device_total_ms=%.6f "
      "fused_device_total_ms=%.6f fused_wall_total_ms=%.6f "
      "speedup=%.6f delta_ms=%.6f diff_max_abs=%.6e diff_mean_abs=%.6e\n",
      static_cast<long long>(shape.heads),
      static_cast<long long>(shape.head_dim),
      static_cast<long long>(shape.history),
      static_cast<long long>(shape.realtime),
      static_cast<long long>(shape.target),
      avg.base.mask_build_ms,
      base_mask_build_plus_device_ms,
      avg.base.device_total_ms,
      avg.candidate.device_total_ms,
      avg.candidate.wall_total_ms,
      speedup_dev,
      delta_dev,
      avg.diff_max_abs,
      avg.diff_mean_abs);
  for (const auto& stage : avg.candidate.stages) {
    std::fprintf(stderr,
                 "[MTGR][CUDA][Perf][OneVsFusedHd64HotShape][Stage] %s "
                 "workspace_ms=%.6f exec_ms=%.6f host_ms=%.6f\n",
                 stage.name.c_str(),
                 stage.workspace_ms,
                 stage.exec_ms,
                 stage.host_submit_ms);
  }
  std::fflush(stderr);
}

TEST_F(MTGRAttentionOneVsMultiStagePerfTest,
       OneVsFusedPartialRealTimeHotShape) {
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
  shape.matched_prefix = shape.history + shape.context + 409;

  const auto avg = run_shape_vs_backend(
      shape, warmup, repeat, LayerMTGRAttentionBackend::kFused);
  const auto no_match_fused =
      run_shape_fused_only(make_no_match_counterpart(shape), warmup, repeat);
  const double base_mask_build_plus_device_ms =
      avg.base.mask_build_ms + avg.base.device_total_ms;
  const double speedup_dev =
      base_mask_build_plus_device_ms / avg.candidate.device_total_ms;
  const double partial_vs_no_match_speedup_dev =
      no_match_fused.device_total_ms / avg.candidate.device_total_ms;
  const double delta_dev =
      base_mask_build_plus_device_ms - avg.candidate.device_total_ms;
  const double partial_vs_no_match_delta_dev =
      no_match_fused.device_total_ms - avg.candidate.device_total_ms;

  std::fprintf(stderr,
               "[MTGR][CUDA][Perf][OneVsFusedPartialRTHotShape] "
               "heads=%lld head_dim=%lld history=%lld realtime=%lld "
               "realtime_matched=%lld target=%lld "
               "base_mask_build_ms=%.6f base_mask_build_plus_device_ms=%.6f "
               "base_device_total_ms=%.6f fused_device_total_ms=%.6f "
               "fused_wall_total_ms=%.6f speedup=%.6f delta_ms=%.6f "
               "no_match_fused_device_total_ms=%.6f "
               "partial_vs_no_match_speedup=%.6f "
               "partial_vs_no_match_delta_ms=%.6f "
               "diff_max_abs=%.6e diff_mean_abs=%.6e\n",
               static_cast<long long>(shape.heads),
               static_cast<long long>(shape.head_dim),
               static_cast<long long>(shape.history),
               static_cast<long long>(shape.realtime),
               static_cast<long long>(shape.realtime_matched()),
               static_cast<long long>(shape.target),
               avg.base.mask_build_ms,
               base_mask_build_plus_device_ms,
               avg.base.device_total_ms,
               avg.candidate.device_total_ms,
               avg.candidate.wall_total_ms,
               speedup_dev,
               delta_dev,
               no_match_fused.device_total_ms,
               partial_vs_no_match_speedup_dev,
               partial_vs_no_match_delta_dev,
               avg.diff_max_abs,
               avg.diff_mean_abs);
  for (const auto& stage : avg.candidate.stages) {
    std::fprintf(stderr,
                 "[MTGR][CUDA][Perf][OneVsFusedPartialRTHotShape][Stage] %s "
                 "workspace_ms=%.6f exec_ms=%.6f host_ms=%.6f\n",
                 stage.name.c_str(),
                 stage.workspace_ms,
                 stage.exec_ms,
                 stage.host_submit_ms);
  }
  std::fflush(stderr);
}

namespace {

struct MultiBatchProbeSequence {
  MTGRAttentionTestShape shape;
  torch::Tensor full_key_bsnd;
  torch::Tensor full_value_bsnd;
  torch::Tensor query;
  torch::Tensor key;
  torch::Tensor value;
  xllm::layer::AttentionMetadata metadata;
  xllm::KVCache one_cache;
  xllm::KVCache fused_cache;
  std::unique_ptr<LayerMTGRAttentionImpl> one_stage;
  std::unique_ptr<LayerMTGRAttentionImpl> fused;
};

struct MultiBatchProbeMetrics {
  double base_mask_build_ms = 0.0;
  double base_device_total_ms = 0.0;
  double base_wall_total_ms = 0.0;
  double fused_device_total_ms = 0.0;
  double fused_wall_total_ms = 0.0;
  double diff_max_abs = 0.0;
  double diff_mean_abs = 0.0;
};

std::vector<int> parse_env_int_list_or_default(const char* name,
                                               std::vector<int> defaults) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return defaults;
  }
  std::vector<int> parsed;
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
    const long v = std::strtol(token.c_str(), &end, 10);
    if (end == token.c_str() || *end != '\0' || v <= 0) {
      return defaults;
    }
    parsed.push_back(static_cast<int>(v));
  }
  return parsed.empty() ? defaults : parsed;
}

MTGRAttentionTestShape sample_probe_shape(std::mt19937_64* rng,
                                          bool partial_match) {
  CHECK(rng != nullptr);
  const std::vector<int64_t> heads_all = {4, 8, 12};
  const std::vector<int64_t> head_dims_all = {64, 128};
  std::uniform_int_distribution<int> heads_dist(
      0, static_cast<int>(heads_all.size() - 1));
  std::uniform_int_distribution<int> head_dim_dist(
      0, static_cast<int>(head_dims_all.size() - 1));

  MTGRAttentionTestShape shape;
  shape.heads = heads_all[heads_dist(*rng)];
  shape.kv_heads = shape.heads;
  shape.head_dim = head_dims_all[head_dim_dist(*rng)];
  shape.history = sample_non_aligned_len(rng, 1350, 4096);
  shape.context = 8;
  shape.realtime = sample_non_aligned_len(rng, 100, 600);
  shape.target = sample_non_aligned_len(rng, 800, 2400);
  shape.matched_prefix =
      partial_match ? shape.history + shape.context + (shape.realtime * 4) / 5
                    : 0;
  return shape;
}

MTGRAttentionTestShape sample_odd_no_match_probe_shape(std::mt19937_64* rng,
                                                       int64_t heads,
                                                       int64_t head_dim) {
  CHECK(rng != nullptr);
  MTGRAttentionTestShape shape;
  shape.heads = heads;
  shape.kv_heads = heads;
  shape.head_dim = head_dim;
  shape.history = sample_non_aligned_len(rng, 1350, 4096);
  shape.context = 8;
  shape.realtime = sample_non_aligned_len(rng, 100, 600);
  shape.target = sample_non_aligned_len(rng, 800, 2400);
  shape.matched_prefix = 0;
  return shape;
}

std::string encode_true_multi_batch_group_key(
    const std::vector<MTGRAttentionTestShape>& shapes) {
  std::ostringstream oss;
  for (size_t i = 0; i < shapes.size(); ++i) {
    if (i != 0) {
      oss << "|";
    }
    const auto& shape = shapes[i];
    oss << shape.heads << ":" << shape.head_dim << ":" << shape.history << ":"
        << shape.realtime << ":" << shape.target;
  }
  return oss.str();
}

void append_true_multi_batch_shape_cells(
    std::ostream& os,
    const std::vector<MTGRAttentionTestShape>& shapes) {
  for (size_t i = 0; i < 4; ++i) {
    if (i < shapes.size()) {
      const auto& shape = shapes[i];
      os << "," << shape.history << "," << shape.realtime << ","
         << shape.target;
    } else {
      os << ",,,";
    }
  }
}

struct TrueFusedBatchSetup {
  xllm::layer::AttentionMetadata metadata;
  xllm::KVCache kv_cache;
  torch::Tensor packed_query;
  torch::Tensor packed_key;
  torch::Tensor packed_value;
};

enum class TrueBatchBaseBackend {
  kWrapperFused,
  kOneStage,
};

std::string encode_probe_shape_local_key(const MTGRAttentionTestShape& shape) {
  return std::to_string(shape.history) + "_" + std::to_string(shape.realtime) +
         "_" + std::to_string(shape.target);
}

MTGRAttentionTestShape sample_partial_probe_shape_with_fixed_layout(
    std::mt19937_64* rng,
    int64_t heads,
    int64_t head_dim) {
  auto shape = sample_probe_shape(rng, /*partial_match=*/true);
  shape.heads = heads;
  shape.kv_heads = heads;
  shape.head_dim = head_dim;
  shape.matched_prefix =
      shape.history + shape.context + (shape.realtime * 4) / 5;
  return shape;
}

MultiBatchProbeSequence make_probe_sequence(const MTGRAttentionTestShape& shape,
                                            const torch::Device& device);

void build_probe_batch_from_shapes(
    const std::vector<MTGRAttentionTestShape>& shapes,
    const torch::Device& device,
    std::vector<MultiBatchProbeSequence>* batch) {
  CHECK(batch != nullptr);
  batch->clear();
  batch->reserve(shapes.size());
  for (const auto& shape : shapes) {
    batch->push_back(make_probe_sequence(shape, device));
  }
}

bool build_unique_true_multi_batch_no_match_group(
    std::mt19937_64* rng,
    int batch_size,
    const torch::Device& device,
    std::unordered_set<std::string>* seen_groups,
    std::vector<MTGRAttentionTestShape>* shapes,
    std::vector<MultiBatchProbeSequence>* batch) {
  CHECK(rng != nullptr);
  CHECK(seen_groups != nullptr);
  CHECK(shapes != nullptr);
  CHECK(batch != nullptr);

  const std::vector<int64_t> heads_all = {4, 8, 12};
  const std::vector<int64_t> head_dims_all = {64, 128};
  std::uniform_int_distribution<int> heads_dist(
      0, static_cast<int>(heads_all.size() - 1));
  std::uniform_int_distribution<int> head_dim_dist(
      0, static_cast<int>(head_dims_all.size() - 1));

  shapes->clear();
  batch->clear();
  for (int attempt = 0; attempt < 256; ++attempt) {
    shapes->clear();
    const int64_t heads = heads_all[heads_dist(*rng)];
    const int64_t head_dim = head_dims_all[head_dim_dist(*rng)];
    std::unordered_set<std::string> local_seen;
    for (int i = 0; i < batch_size; ++i) {
      MTGRAttentionTestShape shape =
          sample_odd_no_match_probe_shape(rng, heads, head_dim);
      if (!local_seen.insert(encode_probe_shape_local_key(shape)).second) {
        shapes->clear();
        break;
      }
      shapes->push_back(shape);
    }
    if (static_cast<int>(shapes->size()) != batch_size) {
      continue;
    }
    if (!seen_groups->insert(encode_true_multi_batch_group_key(*shapes))
             .second) {
      continue;
    }
    build_probe_batch_from_shapes(*shapes, device, batch);
    return true;
  }

  shapes->clear();
  batch->clear();
  return false;
}

bool build_unique_true_multi_batch_partial_group(
    std::mt19937_64* rng,
    int batch_size,
    const torch::Device& device,
    std::unordered_set<std::string>* seen_groups,
    std::vector<MTGRAttentionTestShape>* shapes,
    std::vector<MultiBatchProbeSequence>* batch) {
  CHECK(rng != nullptr);
  CHECK(seen_groups != nullptr);
  CHECK(shapes != nullptr);
  CHECK(batch != nullptr);

  shapes->clear();
  batch->clear();
  for (int attempt = 0; attempt < 256; ++attempt) {
    shapes->clear();
    const MTGRAttentionTestShape ref_shape =
        sample_probe_shape(rng, /*partial_match=*/true);
    const int64_t heads = ref_shape.heads;
    const int64_t head_dim = ref_shape.head_dim;
    std::unordered_set<std::string> local_seen;
    for (int i = 0; i < batch_size; ++i) {
      MTGRAttentionTestShape shape =
          sample_partial_probe_shape_with_fixed_layout(rng, heads, head_dim);
      if (!local_seen.insert(encode_probe_shape_local_key(shape)).second) {
        shapes->clear();
        break;
      }
      shapes->push_back(shape);
    }
    if (static_cast<int>(shapes->size()) != batch_size) {
      continue;
    }
    if (!seen_groups->insert(encode_true_multi_batch_group_key(*shapes))
             .second) {
      continue;
    }
    build_probe_batch_from_shapes(*shapes, device, batch);
    return true;
  }

  shapes->clear();
  batch->clear();
  return false;
}

TrueFusedBatchSetup make_true_fused_batch_setup(
    const std::vector<MultiBatchProbeSequence>& batch) {
  CHECK(!batch.empty());
  const auto& ref = batch.front().shape;
  const bool partial_match = ref.matched_prefix > 0;

  std::vector<MTGRAttentionTestShape> shapes;
  std::vector<torch::Tensor> full_keys;
  std::vector<torch::Tensor> full_values;
  std::vector<torch::Tensor> packed_queries;
  std::vector<torch::Tensor> packed_keys;
  std::vector<torch::Tensor> packed_values;
  shapes.reserve(batch.size());
  packed_queries.reserve(batch.size());
  packed_keys.reserve(batch.size());
  packed_values.reserve(batch.size());
  if (partial_match) {
    full_keys.reserve(batch.size());
    full_values.reserve(batch.size());
  }

  for (const auto& seq : batch) {
    CHECK_EQ(seq.shape.heads, ref.heads);
    CHECK_EQ(seq.shape.kv_heads, ref.kv_heads);
    CHECK_EQ(seq.shape.head_dim, ref.head_dim);
    CHECK_EQ(seq.shape.matched_prefix > 0, partial_match);
    if (partial_match) {
      CHECK_GT(seq.shape.matched_prefix, 0);
      full_keys.push_back(seq.full_key_bsnd);
      full_values.push_back(seq.full_value_bsnd);
    } else {
      CHECK_EQ(seq.shape.matched_prefix, 0);
    }
    shapes.push_back(seq.shape);
    packed_queries.push_back(seq.query);
    packed_keys.push_back(seq.key);
    packed_values.push_back(seq.value);
  }

  TrueFusedBatchSetup setup;
  setup.packed_query = torch::cat(packed_queries, 0).contiguous();
  setup.packed_key = torch::cat(packed_keys, 0).contiguous();
  setup.packed_value = torch::cat(packed_values, 0).contiguous();

  if (partial_match) {
    auto partial_setup =
        make_mtgr_partial_batch_setup(shapes,
                                      full_keys,
                                      full_values,
                                      setup.packed_query.device(),
                                      torch::kFloat16,
                                      kBlockSize);
    setup.metadata = std::move(partial_setup.metadata);
    setup.kv_cache = std::move(partial_setup.kv_cache);
  } else {
    setup.metadata = make_mtgr_attention_metadata(
        shapes, setup.packed_query.device(), kBlockSize);
  }
  return setup;
}

torch::Tensor run_true_batch_base_once(MultiBatchProbeSequence* seq,
                                       TrueBatchBaseBackend base_backend,
                                       double* base_mask_build_ms,
                                       double* base_device_ms) {
  CHECK(seq != nullptr);
  CHECK(base_mask_build_ms != nullptr);
  CHECK(base_device_ms != nullptr);

  if (base_backend == TrueBatchBaseBackend::kOneStage) {
    auto output = std::get<0>(seq->one_stage->forward(
        seq->metadata, seq->query, seq->key, seq->value, seq->one_cache));
    const auto metrics = seq->one_stage->last_metrics();
    *base_mask_build_ms += metrics.mask_build_ms;
    *base_device_ms += metrics.device_total_ms;
    return output;
  }

  auto output = std::get<0>(seq->fused->forward(
      seq->metadata, seq->query, seq->key, seq->value, seq->fused_cache));
  *base_device_ms += seq->fused->last_metrics().device_total_ms;
  return output;
}

MultiBatchProbeSequence make_probe_sequence(const MTGRAttentionTestShape& shape,
                                            const torch::Device& device) {
  auto opts = torch::TensorOptions().dtype(torch::kFloat16).device(device);
  const int64_t total = shape.total_len();
  const int64_t local = shape.local_len();
  const float scale = 1.0f / std::sqrt(static_cast<float>(shape.head_dim));

  auto full_query =
      torch::randn({1, total, shape.heads, shape.head_dim}, opts) * 0.05;
  auto full_key =
      torch::randn({1, total, shape.kv_heads, shape.head_dim}, opts) * 0.05;
  auto full_value =
      torch::randn({1, total, shape.kv_heads, shape.head_dim}, opts) * 0.05;

  MultiBatchProbeSequence seq;
  seq.shape = shape;
  seq.full_key_bsnd = full_key.contiguous();
  seq.full_value_bsnd = full_value.contiguous();
  seq.query = full_query.select(0, 0)
                  .narrow(0, shape.matched_prefix, local)
                  .contiguous()
                  .view({local, shape.heads * shape.head_dim});
  seq.key = full_key.select(0, 0)
                .narrow(0, shape.matched_prefix, local)
                .contiguous()
                .view({local, shape.kv_heads * shape.head_dim});
  seq.value = full_value.select(0, 0)
                  .narrow(0, shape.matched_prefix, local)
                  .contiguous()
                  .view({local, shape.kv_heads * shape.head_dim});
  seq.metadata = make_mtgr_attention_metadata(shape, device, kBlockSize);
  seq.one_cache =
      make_mtgr_kv_cache(shape, device, torch::kFloat16, kBlockSize);
  seq.fused_cache =
      make_mtgr_kv_cache(shape, device, torch::kFloat16, kBlockSize);
  prefill_mtgr_matched_prefix_cache(
      full_key, full_value, shape, kBlockSize, seq.one_cache);
  prefill_mtgr_matched_prefix_cache(
      full_key, full_value, shape, kBlockSize, seq.fused_cache);
  seq.one_stage = std::make_unique<LayerMTGRAttentionImpl>(
      shape.heads,
      shape.head_dim,
      scale,
      shape.kv_heads,
      LayerMTGRAttentionBackend::kOneStage);
  seq.fused = std::make_unique<LayerMTGRAttentionImpl>(
      shape.heads,
      shape.head_dim,
      scale,
      shape.kv_heads,
      LayerMTGRAttentionBackend::kFused);
  return seq;
}

MultiBatchProbeMetrics run_wrapper_loop_batch(
    std::vector<MultiBatchProbeSequence>* batch,
    int warmup,
    int repeat) {
  CHECK(batch != nullptr);
  CHECK(!batch->empty());
  for (int i = 0; i < warmup; ++i) {
    for (auto& seq : *batch) {
      (void)std::get<0>(seq.one_stage->forward(
          seq.metadata, seq.query, seq.key, seq.value, seq.one_cache));
    }
    for (auto& seq : *batch) {
      (void)std::get<0>(seq.fused->forward(
          seq.metadata, seq.query, seq.key, seq.value, seq.fused_cache));
    }
  }
  CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

  MultiBatchProbeMetrics avg;
  for (int i = 0; i < repeat; ++i) {
    std::vector<torch::Tensor> base_outputs;
    std::vector<torch::Tensor> fused_outputs;
    base_outputs.reserve(batch->size());
    fused_outputs.reserve(batch->size());

    auto base_wall_start = std::chrono::steady_clock::now();
    double base_device_ms = 0.0;
    for (auto& seq : *batch) {
      auto output = std::get<0>(seq.one_stage->forward(
          seq.metadata, seq.query, seq.key, seq.value, seq.one_cache));
      const auto metrics = seq.one_stage->last_metrics();
      base_device_ms += metrics.device_total_ms;
      base_outputs.push_back(output);
    }
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const double base_wall_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - base_wall_start)
            .count();

    auto fused_wall_start = std::chrono::steady_clock::now();
    double fused_device_ms = 0.0;
    for (auto& seq : *batch) {
      auto output = std::get<0>(seq.fused->forward(
          seq.metadata, seq.query, seq.key, seq.value, seq.fused_cache));
      const auto metrics = seq.fused->last_metrics();
      fused_device_ms += metrics.device_total_ms;
      fused_outputs.push_back(output);
    }
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const double fused_wall_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - fused_wall_start)
            .count();

    double diff_max = 0.0;
    double diff_sum = 0.0;
    int64_t diff_count = 0;
    for (size_t s = 0; s < batch->size(); ++s) {
      auto diff = (base_outputs[s].to(torch::kFloat32) -
                   fused_outputs[s].to(torch::kFloat32))
                      .abs();
      diff_max = std::max(diff_max, diff.max().item<double>());
      diff_sum += diff.sum().item<double>();
      diff_count += diff.numel();
    }

    avg.base_device_total_ms += base_device_ms;
    avg.base_wall_total_ms += base_wall_ms;
    avg.fused_device_total_ms += fused_device_ms;
    avg.fused_wall_total_ms += fused_wall_ms;
    avg.diff_max_abs = std::max(avg.diff_max_abs, diff_max);
    avg.diff_mean_abs +=
        diff_count > 0 ? diff_sum / static_cast<double>(diff_count) : 0.0;
  }

  const double inv = 1.0 / static_cast<double>(repeat);
  avg.base_device_total_ms *= inv;
  avg.base_wall_total_ms *= inv;
  avg.fused_device_total_ms *= inv;
  avg.fused_wall_total_ms *= inv;
  avg.diff_mean_abs *= inv;
  return avg;
}

MultiBatchProbeMetrics run_true_fused_batch_impl(
    std::vector<MultiBatchProbeSequence>* batch,
    int warmup,
    int repeat,
    TrueBatchBaseBackend base_backend) {
  CHECK(batch != nullptr);
  CHECK(!batch->empty());
  const auto& ref = batch->front().shape;
  const float scale = 1.0f / std::sqrt(static_cast<float>(ref.head_dim));
  auto packed_setup = make_true_fused_batch_setup(*batch);

  LayerMTGRAttentionImpl batched_fused(ref.heads,
                                       ref.head_dim,
                                       scale,
                                       ref.kv_heads,
                                       LayerMTGRAttentionBackend::kFused);

  for (int i = 0; i < warmup; ++i) {
    for (auto& seq : *batch) {
      double ignored_mask_build_ms = 0.0;
      double ignored_device_ms = 0.0;
      (void)run_true_batch_base_once(
          &seq, base_backend, &ignored_mask_build_ms, &ignored_device_ms);
    }
    (void)std::get<0>(batched_fused.forward(packed_setup.metadata,
                                            packed_setup.packed_query,
                                            packed_setup.packed_key,
                                            packed_setup.packed_value,
                                            packed_setup.kv_cache));
  }
  CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

  MultiBatchProbeMetrics avg;
  for (int i = 0; i < repeat; ++i) {
    std::vector<torch::Tensor> base_outputs;
    base_outputs.reserve(batch->size());

    auto base_wall_start = std::chrono::steady_clock::now();
    double base_mask_build_ms = 0.0;
    double base_device_ms = 0.0;
    for (auto& seq : *batch) {
      base_outputs.push_back(run_true_batch_base_once(
          &seq, base_backend, &base_mask_build_ms, &base_device_ms));
    }
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const double base_wall_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - base_wall_start)
            .count();

    auto fused_wall_start = std::chrono::steady_clock::now();
    auto fused_output =
        std::get<0>(batched_fused.forward(packed_setup.metadata,
                                          packed_setup.packed_query,
                                          packed_setup.packed_key,
                                          packed_setup.packed_value,
                                          packed_setup.kv_cache));
    const auto fused_metrics = batched_fused.last_metrics();
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const double fused_wall_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - fused_wall_start)
            .count();

    auto base_output = torch::cat(base_outputs, 0).contiguous();
    auto diff =
        (base_output.to(torch::kFloat32) - fused_output.to(torch::kFloat32))
            .abs();

    avg.base_mask_build_ms += base_mask_build_ms;
    avg.base_device_total_ms += base_device_ms;
    avg.base_wall_total_ms += base_wall_ms;
    avg.fused_device_total_ms += fused_metrics.device_total_ms;
    avg.fused_wall_total_ms += fused_wall_ms;
    avg.diff_max_abs = std::max(avg.diff_max_abs, diff.max().item<double>());
    avg.diff_mean_abs += diff.mean().item<double>();
  }

  const double inv = 1.0 / static_cast<double>(repeat);
  avg.base_mask_build_ms *= inv;
  avg.base_device_total_ms *= inv;
  avg.base_wall_total_ms *= inv;
  avg.fused_device_total_ms *= inv;
  avg.fused_wall_total_ms *= inv;
  avg.diff_mean_abs *= inv;
  return avg;
}

MultiBatchProbeMetrics run_true_fused_no_match_batch(
    std::vector<MultiBatchProbeSequence>* batch,
    int warmup,
    int repeat) {
  CHECK(batch != nullptr);
  CHECK(!batch->empty());
  CHECK_EQ(batch->front().shape.matched_prefix, 0);
  return run_true_fused_batch_impl(
      batch, warmup, repeat, TrueBatchBaseBackend::kWrapperFused);
}

MultiBatchProbeMetrics run_true_fused_no_match_batch_vs_one_stage(
    std::vector<MultiBatchProbeSequence>* batch,
    int warmup,
    int repeat) {
  CHECK(batch != nullptr);
  CHECK(!batch->empty());
  CHECK_EQ(batch->front().shape.matched_prefix, 0);
  return run_true_fused_batch_impl(
      batch, warmup, repeat, TrueBatchBaseBackend::kOneStage);
}

MultiBatchProbeMetrics run_true_fused_partial_batch(
    std::vector<MultiBatchProbeSequence>* batch,
    int warmup,
    int repeat) {
  CHECK(batch != nullptr);
  CHECK(!batch->empty());
  CHECK_GT(batch->front().shape.matched_prefix, 0);
  return run_true_fused_batch_impl(
      batch, warmup, repeat, TrueBatchBaseBackend::kWrapperFused);
}

MultiBatchProbeMetrics run_true_fused_partial_batch_vs_one_stage(
    std::vector<MultiBatchProbeSequence>* batch,
    int warmup,
    int repeat) {
  CHECK(batch != nullptr);
  CHECK(!batch->empty());
  CHECK_GT(batch->front().shape.matched_prefix, 0);
  return run_true_fused_batch_impl(
      batch, warmup, repeat, TrueBatchBaseBackend::kOneStage);
}

}  // namespace

TEST_F(MTGRAttentionOneVsMultiStagePerfTest, MultiBatchWrapperLoopProbeCsv) {
  torch::NoGradGuard no_grad_guard;
  const int warmup = std::max(0, env_int("XLLM_MTGR_ATTENTION_WARMUP", 2));
  const int repeat = std::max(1, env_int("XLLM_MTGR_ATTENTION_REPEAT", 5));
  const int group_count =
      std::max(1, env_int("XLLM_MTGR_ATTENTION_MULTI_BATCH_GROUPS", 20));
  const int seed = env_int("XLLM_MTGR_ATTENTION_MULTI_BATCH_SEED", 20260502);
  const auto batch_sizes = parse_env_int_list_or_default(
      "XLLM_MTGR_ATTENTION_MULTI_BATCH_SIZES", {1, 2, 4, 8});
  const std::string csv_path =
      csv_path_from_env("XLLM_MTGR_ATTENTION_MULTI_BATCH_WRAPPER_CSV",
                        "mtgr_attention_multi_batch_wrapper_probe.csv");
  std::ofstream csv(csv_path, std::ios::out | std::ios::trunc);
  CHECK(csv.is_open()) << "failed to open csv path: " << csv_path;
  csv << "branch,batch_size,group_id,base_device_total_ms,base_wall_total_ms,"
         "fused_device_total_ms,fused_wall_total_ms,speedup_dev,speedup_wall,"
         "delta_dev_ms,delta_wall_ms,diff_max_abs,diff_mean_abs\n";

  std::fprintf(stderr,
               "[MTGR][CUDA][Perf][MultiBatchWrapperLoop] groups=%d warmup=%d "
               "repeat=%d seed=%d csv=%s\n",
               group_count,
               warmup,
               repeat,
               seed,
               csv_path.c_str());

  for (bool partial_match : {false, true}) {
    const char* branch = partial_match ? "partial_rt" : "no_match";
    for (int batch_size : batch_sizes) {
      std::mt19937_64 rng(static_cast<uint64_t>(seed) ^
                          (static_cast<uint64_t>(batch_size) << 16) ^
                          (partial_match ? 0x9e3779b97f4a7c15ULL : 0ULL));
      double total_base_device = 0.0;
      double total_fused_device = 0.0;
      double total_speedup_dev = 0.0;
      double min_speedup_dev = std::numeric_limits<double>::infinity();
      double max_speedup_dev = 0.0;
      int regressions = 0;

      for (int group_id = 1; group_id <= group_count; ++group_id) {
        std::vector<MultiBatchProbeSequence> batch;
        batch.reserve(static_cast<size_t>(batch_size));
        for (int i = 0; i < batch_size; ++i) {
          batch.push_back(make_probe_sequence(
              sample_probe_shape(&rng, partial_match), device_));
        }

        const auto avg = run_wrapper_loop_batch(&batch, warmup, repeat);
        const double speedup_dev =
            avg.base_device_total_ms / avg.fused_device_total_ms;
        const double speedup_wall =
            avg.base_wall_total_ms / avg.fused_wall_total_ms;
        const double delta_dev =
            avg.base_device_total_ms - avg.fused_device_total_ms;
        const double delta_wall =
            avg.base_wall_total_ms - avg.fused_wall_total_ms;
        if (speedup_dev < 1.0) {
          ++regressions;
        }
        total_base_device += avg.base_device_total_ms;
        total_fused_device += avg.fused_device_total_ms;
        total_speedup_dev += speedup_dev;
        min_speedup_dev = std::min(min_speedup_dev, speedup_dev);
        max_speedup_dev = std::max(max_speedup_dev, speedup_dev);

        csv << branch << "," << batch_size << "," << group_id << ","
            << std::fixed << std::setprecision(6) << avg.base_device_total_ms
            << "," << avg.base_wall_total_ms << "," << avg.fused_device_total_ms
            << "," << avg.fused_wall_total_ms << "," << speedup_dev << ","
            << speedup_wall << "," << delta_dev << "," << delta_wall << ","
            << std::scientific << std::setprecision(6) << avg.diff_max_abs
            << "," << avg.diff_mean_abs << "\n";
        csv.flush();

        std::fprintf(stderr,
                     "[MTGR][CUDA][Perf][MultiBatchWrapperLoop %s B=%d "
                     "%d/%d] base_device=%.6f fused_device=%.6f "
                     "speedup=%.6f diff_max=%.6e\n",
                     branch,
                     batch_size,
                     group_id,
                     group_count,
                     avg.base_device_total_ms,
                     avg.fused_device_total_ms,
                     speedup_dev,
                     avg.diff_max_abs);
      }

      const double inv = 1.0 / static_cast<double>(group_count);
      std::fprintf(stderr,
                   "[MTGR][CUDA][Perf][MultiBatchWrapperLoop][Summary] "
                   "branch=%s batch_size=%d groups=%d regressions=%d "
                   "avg_base_device_ms=%.6f avg_fused_device_ms=%.6f "
                   "avg_speedup_dev=%.6f min_speedup_dev=%.6f "
                   "max_speedup_dev=%.6f\n",
                   branch,
                   batch_size,
                   group_count,
                   regressions,
                   total_base_device * inv,
                   total_fused_device * inv,
                   total_speedup_dev * inv,
                   min_speedup_dev,
                   max_speedup_dev);
    }
  }
  std::fflush(stderr);
}

TEST_F(MTGRAttentionOneVsMultiStagePerfTest,
       TrueMultiBatchFusedNoMatchOddLengthRandomCsv) {
  torch::NoGradGuard no_grad_guard;
  const int warmup = std::max(0, env_int("XLLM_MTGR_ATTENTION_WARMUP", 2));
  const int repeat = std::max(1, env_int("XLLM_MTGR_ATTENTION_REPEAT", 5));
  const int group_count = std::max(
      1, env_int("XLLM_MTGR_ATTENTION_TRUE_MULTI_BATCH_ODD_GROUPS", 20));
  const int seed =
      env_int("XLLM_MTGR_ATTENTION_TRUE_MULTI_BATCH_ODD_SEED", 20260503);
  const auto batch_sizes = parse_env_int_list_or_default(
      "XLLM_MTGR_ATTENTION_TRUE_MULTI_BATCH_ODD_BATCH_SIZES", {2, 3, 4});
  const std::string csv_path =
      csv_path_from_env("XLLM_MTGR_ATTENTION_TRUE_MULTI_BATCH_ODD_CSV",
                        "mtgr_attention_true_multi_batch_odd_random.csv");
  std::ofstream csv(csv_path, std::ios::out | std::ios::trunc);
  CHECK(csv.is_open()) << "failed to open csv path: " << csv_path;
  csv << "batch_size,group_id,heads,head_dim,"
         "s0_history,s0_realtime,s0_target,"
         "s1_history,s1_realtime,s1_target,"
         "s2_history,s2_realtime,s2_target,"
         "s3_history,s3_realtime,s3_target,"
         "wrapper_fused_device_ms,wrapper_fused_wall_ms,"
         "batched_fused_device_ms,batched_fused_wall_ms,"
         "speedup_dev,speedup_wall,delta_dev_ms,delta_wall_ms,"
         "diff_max_abs,diff_mean_abs\n";

  std::mt19937_64 rng(static_cast<uint64_t>(seed));
  std::fprintf(stderr,
               "[MTGR][CUDA][Perf][TrueMultiBatchFusedOddRandom] groups=%d "
               "warmup=%d repeat=%d seed=%d csv=%s\n",
               group_count,
               warmup,
               repeat,
               seed,
               csv_path.c_str());

  for (int batch_size : batch_sizes) {
    CHECK_GT(batch_size, 1);
    CHECK_LE(batch_size, 4);
    std::unordered_set<std::string> seen;
    double total_wrapper_device = 0.0;
    double total_wrapper_wall = 0.0;
    double total_batched_device = 0.0;
    double total_batched_wall = 0.0;
    double total_speedup_dev = 0.0;
    double total_speedup_wall = 0.0;
    double max_speedup_dev = 0.0;
    double min_speedup_dev = std::numeric_limits<double>::infinity();
    double worst_diff_max = 0.0;
    double worst_diff_mean = 0.0;
    int regressions = 0;
    int precision_regressions = 0;

    for (int group_id = 1; group_id <= group_count; ++group_id) {
      std::vector<MTGRAttentionTestShape> shapes;
      std::vector<MultiBatchProbeSequence> batch;
      const bool accepted = build_unique_true_multi_batch_no_match_group(
          &rng, batch_size, device_, &seen, &shapes, &batch);

      ASSERT_TRUE(accepted) << "failed to build unique odd-length group";
      const auto avg = run_true_fused_no_match_batch(&batch, warmup, repeat);
      const double speedup_dev =
          avg.base_device_total_ms / avg.fused_device_total_ms;
      const double speedup_wall =
          avg.base_wall_total_ms / avg.fused_wall_total_ms;
      const double delta_dev =
          avg.base_device_total_ms - avg.fused_device_total_ms;
      const double delta_wall =
          avg.base_wall_total_ms - avg.fused_wall_total_ms;

      if (speedup_dev < 1.0) {
        ++regressions;
      }
      if (!(avg.diff_max_abs < 1.0e-3) || !(avg.diff_mean_abs < 1.0e-5)) {
        ++precision_regressions;
      }
      total_wrapper_device += avg.base_device_total_ms;
      total_wrapper_wall += avg.base_wall_total_ms;
      total_batched_device += avg.fused_device_total_ms;
      total_batched_wall += avg.fused_wall_total_ms;
      total_speedup_dev += speedup_dev;
      total_speedup_wall += speedup_wall;
      max_speedup_dev = std::max(max_speedup_dev, speedup_dev);
      min_speedup_dev = std::min(min_speedup_dev, speedup_dev);
      worst_diff_max = std::max(worst_diff_max, avg.diff_max_abs);
      worst_diff_mean = std::max(worst_diff_mean, avg.diff_mean_abs);

      std::ostringstream row;
      row << std::fixed << std::setprecision(6) << batch_size << "," << group_id
          << "," << shapes.front().heads << "," << shapes.front().head_dim;
      append_true_multi_batch_shape_cells(row, shapes);
      row << "," << avg.base_device_total_ms << "," << avg.base_wall_total_ms
          << "," << avg.fused_device_total_ms << "," << avg.fused_wall_total_ms
          << "," << speedup_dev << "," << speedup_wall << "," << delta_dev
          << "," << delta_wall << "," << avg.diff_max_abs << ","
          << avg.diff_mean_abs;
      csv << row.str() << "\n";
      csv.flush();
      std::fprintf(stderr,
                   "[MTGR][CUDA][Perf][TrueMultiBatchFusedOddRandom "
                   "batch=%d group=%d/%d] %s\n",
                   batch_size,
                   group_id,
                   group_count,
                   row.str().c_str());
      std::fflush(stderr);

      EXPECT_LT(avg.diff_max_abs, 1.0e-3);
      EXPECT_LT(avg.diff_mean_abs, 1.0e-5);
    }

    const double inv = 1.0 / static_cast<double>(group_count);
    std::fprintf(stderr,
                 "[MTGR][CUDA][Perf][TrueMultiBatchFusedOddRandom][Summary] "
                 "batch_size=%d groups=%d regressions=%d "
                 "precision_regressions=%d avg_wrapper_device_ms=%.6f "
                 "avg_wrapper_wall_ms=%.6f avg_batched_device_ms=%.6f "
                 "avg_batched_wall_ms=%.6f avg_speedup_dev=%.6f "
                 "avg_speedup_wall=%.6f min_speedup_dev=%.6f "
                 "max_speedup_dev=%.6f worst_diff_max=%.6e "
                 "worst_diff_mean=%.6e\n",
                 batch_size,
                 group_count,
                 regressions,
                 precision_regressions,
                 total_wrapper_device * inv,
                 total_wrapper_wall * inv,
                 total_batched_device * inv,
                 total_batched_wall * inv,
                 total_speedup_dev * inv,
                 total_speedup_wall * inv,
                 min_speedup_dev,
                 max_speedup_dev,
                 worst_diff_max,
                 worst_diff_mean);
    std::fflush(stderr);
  }
}

TEST_F(MTGRAttentionOneVsMultiStagePerfTest,
       TrueMultiBatchFusedPartialOddLengthRandomCsv) {
  torch::NoGradGuard no_grad_guard;
  const int warmup = std::max(0, env_int("XLLM_MTGR_ATTENTION_WARMUP", 2));
  const int repeat = std::max(1, env_int("XLLM_MTGR_ATTENTION_REPEAT", 5));
  const int group_count = std::max(
      1,
      env_int("XLLM_MTGR_ATTENTION_TRUE_MULTI_BATCH_PARTIAL_ODD_GROUPS", 20));
  const int seed = env_int(
      "XLLM_MTGR_ATTENTION_TRUE_MULTI_BATCH_PARTIAL_ODD_SEED", 20260505);
  const auto batch_sizes = parse_env_int_list_or_default(
      "XLLM_MTGR_ATTENTION_TRUE_MULTI_BATCH_PARTIAL_ODD_BATCH_SIZES",
      {2, 3, 4});
  const std::string csv_path = csv_path_from_env(
      "XLLM_MTGR_ATTENTION_TRUE_MULTI_BATCH_PARTIAL_ODD_CSV",
      "mtgr_attention_true_multi_batch_partial_odd_random.csv");
  std::ofstream csv(csv_path, std::ios::out | std::ios::trunc);
  CHECK(csv.is_open()) << "failed to open csv path: " << csv_path;
  csv << "batch_size,group_id,heads,head_dim,"
         "s0_history,s0_realtime,s0_target,"
         "s1_history,s1_realtime,s1_target,"
         "s2_history,s2_realtime,s2_target,"
         "s3_history,s3_realtime,s3_target,"
         "wrapper_fused_device_ms,wrapper_fused_wall_ms,"
         "batched_fused_device_ms,batched_fused_wall_ms,"
         "speedup_dev,speedup_wall,delta_dev_ms,delta_wall_ms,"
         "diff_max_abs,diff_mean_abs\n";

  std::mt19937_64 rng(static_cast<uint64_t>(seed));
  std::fprintf(stderr,
               "[MTGR][CUDA][Perf][TrueMultiBatchFusedPartialOddRandom] "
               "groups=%d warmup=%d repeat=%d seed=%d csv=%s\n",
               group_count,
               warmup,
               repeat,
               seed,
               csv_path.c_str());

  for (int batch_size : batch_sizes) {
    CHECK_GT(batch_size, 1);
    CHECK_LE(batch_size, 4);
    std::unordered_set<std::string> seen;
    double total_wrapper_device = 0.0;
    double total_wrapper_wall = 0.0;
    double total_batched_device = 0.0;
    double total_batched_wall = 0.0;
    double total_speedup_dev = 0.0;
    double total_speedup_wall = 0.0;
    double max_speedup_dev = 0.0;
    double min_speedup_dev = std::numeric_limits<double>::infinity();
    double worst_diff_max = 0.0;
    double worst_diff_mean = 0.0;
    int regressions = 0;
    int precision_regressions = 0;

    for (int group_id = 1; group_id <= group_count; ++group_id) {
      std::vector<MTGRAttentionTestShape> shapes;
      std::vector<MultiBatchProbeSequence> batch;
      const bool accepted = build_unique_true_multi_batch_partial_group(
          &rng, batch_size, device_, &seen, &shapes, &batch);

      ASSERT_TRUE(accepted)
          << "failed to build unique odd-length partial group";
      const auto avg = run_true_fused_partial_batch(&batch, warmup, repeat);
      const double speedup_dev =
          avg.base_device_total_ms / avg.fused_device_total_ms;
      const double speedup_wall =
          avg.base_wall_total_ms / avg.fused_wall_total_ms;
      const double delta_dev =
          avg.base_device_total_ms - avg.fused_device_total_ms;
      const double delta_wall =
          avg.base_wall_total_ms - avg.fused_wall_total_ms;

      if (speedup_dev < 1.0) {
        ++regressions;
      }
      if (!(avg.diff_max_abs < 1.0e-3) || !(avg.diff_mean_abs < 1.0e-5)) {
        ++precision_regressions;
      }
      total_wrapper_device += avg.base_device_total_ms;
      total_wrapper_wall += avg.base_wall_total_ms;
      total_batched_device += avg.fused_device_total_ms;
      total_batched_wall += avg.fused_wall_total_ms;
      total_speedup_dev += speedup_dev;
      total_speedup_wall += speedup_wall;
      max_speedup_dev = std::max(max_speedup_dev, speedup_dev);
      min_speedup_dev = std::min(min_speedup_dev, speedup_dev);
      worst_diff_max = std::max(worst_diff_max, avg.diff_max_abs);
      worst_diff_mean = std::max(worst_diff_mean, avg.diff_mean_abs);

      std::ostringstream row;
      row << std::fixed << std::setprecision(6) << batch_size << "," << group_id
          << "," << shapes.front().heads << "," << shapes.front().head_dim;
      append_true_multi_batch_shape_cells(row, shapes);
      row << "," << avg.base_device_total_ms << "," << avg.base_wall_total_ms
          << "," << avg.fused_device_total_ms << "," << avg.fused_wall_total_ms
          << "," << speedup_dev << "," << speedup_wall << "," << delta_dev
          << "," << delta_wall << "," << avg.diff_max_abs << ","
          << avg.diff_mean_abs;
      csv << row.str() << "\n";
      csv.flush();
      std::fprintf(stderr,
                   "[MTGR][CUDA][Perf][TrueMultiBatchFusedPartialOddRandom "
                   "batch=%d group=%d/%d] %s\n",
                   batch_size,
                   group_id,
                   group_count,
                   row.str().c_str());
      std::fflush(stderr);

      EXPECT_LT(avg.diff_max_abs, 1.0e-3);
      EXPECT_LT(avg.diff_mean_abs, 1.0e-5);
    }

    const double inv = 1.0 / static_cast<double>(group_count);
    std::fprintf(stderr,
                 "[MTGR][CUDA][Perf][TrueMultiBatchFusedPartialOddRandom]"
                 "[Summary] batch_size=%d groups=%d regressions=%d "
                 "precision_regressions=%d avg_wrapper_device_ms=%.6f "
                 "avg_wrapper_wall_ms=%.6f avg_batched_device_ms=%.6f "
                 "avg_batched_wall_ms=%.6f avg_speedup_dev=%.6f "
                 "avg_speedup_wall=%.6f min_speedup_dev=%.6f "
                 "max_speedup_dev=%.6f worst_diff_max=%.6e "
                 "worst_diff_mean=%.6e\n",
                 batch_size,
                 group_count,
                 regressions,
                 precision_regressions,
                 total_wrapper_device * inv,
                 total_wrapper_wall * inv,
                 total_batched_device * inv,
                 total_batched_wall * inv,
                 total_speedup_dev * inv,
                 total_speedup_wall * inv,
                 min_speedup_dev,
                 max_speedup_dev,
                 worst_diff_max,
                 worst_diff_mean);
    std::fflush(stderr);
  }
}

TEST_F(MTGRAttentionOneVsMultiStagePerfTest,
       TrueMultiBatchFusedVsOneStageOddLengthRandomCsv) {
  torch::NoGradGuard no_grad_guard;
  const int warmup = std::max(0, env_int("XLLM_MTGR_ATTENTION_WARMUP", 1));
  const int repeat = std::max(1, env_int("XLLM_MTGR_ATTENTION_REPEAT", 2));
  const int groups_per_batch = std::max(
      1, env_int("XLLM_MTGR_ATTENTION_TRUE_MULTI_BATCH_ONE_STAGE_GROUPS", 250));
  const int seed =
      env_int("XLLM_MTGR_ATTENTION_TRUE_MULTI_BATCH_ONE_STAGE_SEED", 20260504);
  const auto batch_sizes = parse_env_int_list_or_default(
      "XLLM_MTGR_ATTENTION_TRUE_MULTI_BATCH_ONE_STAGE_BATCH_SIZES",
      {1, 2, 3, 4});
  const std::string csv_path = csv_path_from_env(
      "XLLM_MTGR_ATTENTION_TRUE_MULTI_BATCH_ONE_STAGE_CSV",
      "mtgr_attention_true_multi_batch_vs_one_stage_odd_random.csv");
  std::ofstream csv(csv_path, std::ios::out | std::ios::trunc);
  CHECK(csv.is_open()) << "failed to open csv path: " << csv_path;
  csv << "batch_size,group_id,heads,head_dim,"
         "s0_history,s0_realtime,s0_target,"
         "s1_history,s1_realtime,s1_target,"
         "s2_history,s2_realtime,s2_target,"
         "s3_history,s3_realtime,s3_target,"
         "one_stage_mask_build_ms,one_stage_device_ms,"
         "one_stage_mask_build_plus_device_ms,one_stage_wall_ms,"
         "batched_fused_device_ms,batched_fused_wall_ms,"
         "speedup_dev,speedup_wall,delta_dev_ms,delta_wall_ms,"
         "diff_max_abs,diff_mean_abs\n";

  std::mt19937_64 rng(static_cast<uint64_t>(seed));
  std::fprintf(stderr,
               "[MTGR][CUDA][Perf][TrueMultiBatchVsOneStageOddRandom] "
               "groups_per_batch=%d warmup=%d repeat=%d seed=%d csv=%s\n",
               groups_per_batch,
               warmup,
               repeat,
               seed,
               csv_path.c_str());

  for (int batch_size : batch_sizes) {
    CHECK_GT(batch_size, 0);
    CHECK_LE(batch_size, 4);
    std::unordered_set<std::string> seen;
    double total_base_mask_build = 0.0;
    double total_base_device = 0.0;
    double total_base_mask_plus_device = 0.0;
    double total_base_wall = 0.0;
    double total_fused_device = 0.0;
    double total_fused_wall = 0.0;
    double total_speedup_dev = 0.0;
    double total_speedup_wall = 0.0;
    double min_speedup_dev = std::numeric_limits<double>::infinity();
    double max_speedup_dev = 0.0;
    double worst_diff_max = 0.0;
    double worst_diff_mean = 0.0;
    int regressions = 0;
    int precision_regressions = 0;

    for (int group_id = 1; group_id <= groups_per_batch; ++group_id) {
      std::vector<MTGRAttentionTestShape> shapes;
      std::vector<MultiBatchProbeSequence> batch;
      const bool accepted = build_unique_true_multi_batch_no_match_group(
          &rng, batch_size, device_, &seen, &shapes, &batch);

      ASSERT_TRUE(accepted) << "failed to build unique odd-length group";
      const auto avg =
          run_true_fused_no_match_batch_vs_one_stage(&batch, warmup, repeat);
      const double base_mask_build_plus_device_ms =
          avg.base_mask_build_ms + avg.base_device_total_ms;
      const double speedup_dev =
          base_mask_build_plus_device_ms / avg.fused_device_total_ms;
      const double speedup_wall =
          avg.base_wall_total_ms / avg.fused_wall_total_ms;
      const double delta_dev =
          base_mask_build_plus_device_ms - avg.fused_device_total_ms;
      const double delta_wall =
          avg.base_wall_total_ms - avg.fused_wall_total_ms;

      if (speedup_dev < 1.0) {
        ++regressions;
      }
      if (!(avg.diff_max_abs < 1.0e-3) || !(avg.diff_mean_abs < 1.0e-5)) {
        ++precision_regressions;
      }

      total_base_mask_build += avg.base_mask_build_ms;
      total_base_device += avg.base_device_total_ms;
      total_base_mask_plus_device += base_mask_build_plus_device_ms;
      total_base_wall += avg.base_wall_total_ms;
      total_fused_device += avg.fused_device_total_ms;
      total_fused_wall += avg.fused_wall_total_ms;
      total_speedup_dev += speedup_dev;
      total_speedup_wall += speedup_wall;
      min_speedup_dev = std::min(min_speedup_dev, speedup_dev);
      max_speedup_dev = std::max(max_speedup_dev, speedup_dev);
      worst_diff_max = std::max(worst_diff_max, avg.diff_max_abs);
      worst_diff_mean = std::max(worst_diff_mean, avg.diff_mean_abs);

      std::ostringstream row;
      row << std::fixed << std::setprecision(6) << batch_size << "," << group_id
          << "," << shapes.front().heads << "," << shapes.front().head_dim;
      append_true_multi_batch_shape_cells(row, shapes);
      row << "," << avg.base_mask_build_ms << "," << avg.base_device_total_ms
          << "," << base_mask_build_plus_device_ms << ","
          << avg.base_wall_total_ms << "," << avg.fused_device_total_ms << ","
          << avg.fused_wall_total_ms << "," << speedup_dev << ","
          << speedup_wall << "," << delta_dev << "," << delta_wall << ","
          << avg.diff_max_abs << "," << avg.diff_mean_abs;
      csv << row.str() << "\n";
      csv.flush();
      std::fprintf(stderr,
                   "[MTGR][CUDA][Perf][TrueMultiBatchVsOneStageOddRandom "
                   "batch=%d group=%d/%d] %s\n",
                   batch_size,
                   group_id,
                   groups_per_batch,
                   row.str().c_str());
      std::fflush(stderr);

      EXPECT_LT(avg.diff_max_abs, 1.0e-3);
      EXPECT_LT(avg.diff_mean_abs, 1.0e-5);
    }

    const double inv = 1.0 / static_cast<double>(groups_per_batch);
    std::fprintf(stderr,
                 "[MTGR][CUDA][Perf][TrueMultiBatchVsOneStageOddRandom]"
                 "[Summary] batch_size=%d groups=%d regressions=%d "
                 "precision_regressions=%d avg_base_mask_build_ms=%.6f "
                 "avg_base_device_ms=%.6f avg_base_mask_plus_device_ms=%.6f "
                 "avg_base_wall_ms=%.6f avg_fused_device_ms=%.6f "
                 "avg_fused_wall_ms=%.6f avg_speedup_dev=%.6f "
                 "avg_speedup_wall=%.6f min_speedup_dev=%.6f "
                 "max_speedup_dev=%.6f worst_diff_max=%.6e "
                 "worst_diff_mean=%.6e\n",
                 batch_size,
                 groups_per_batch,
                 regressions,
                 precision_regressions,
                 total_base_mask_build * inv,
                 total_base_device * inv,
                 total_base_mask_plus_device * inv,
                 total_base_wall * inv,
                 total_fused_device * inv,
                 total_fused_wall * inv,
                 total_speedup_dev * inv,
                 total_speedup_wall * inv,
                 min_speedup_dev,
                 max_speedup_dev,
                 worst_diff_max,
                 worst_diff_mean);
    std::fflush(stderr);
  }
}

TEST_F(MTGRAttentionOneVsMultiStagePerfTest,
       TrueMultiBatchFusedPartialVsOneStageOddLengthRandomCsv) {
  torch::NoGradGuard no_grad_guard;
  const int warmup = std::max(0, env_int("XLLM_MTGR_ATTENTION_WARMUP", 1));
  const int repeat = std::max(1, env_int("XLLM_MTGR_ATTENTION_REPEAT", 2));
  const int groups_per_batch = std::max(
      1,
      env_int("XLLM_MTGR_ATTENTION_TRUE_MULTI_BATCH_PARTIAL_ONE_STAGE_GROUPS",
              250));
  const int seed = env_int(
      "XLLM_MTGR_ATTENTION_TRUE_MULTI_BATCH_PARTIAL_ONE_STAGE_SEED", 20260506);
  const auto batch_sizes = parse_env_int_list_or_default(
      "XLLM_MTGR_ATTENTION_TRUE_MULTI_BATCH_PARTIAL_ONE_STAGE_BATCH_SIZES",
      {1, 2, 3, 4});
  const std::string csv_path = csv_path_from_env(
      "XLLM_MTGR_ATTENTION_TRUE_MULTI_BATCH_PARTIAL_ONE_STAGE_CSV",
      "mtgr_attention_true_multi_batch_partial_vs_one_stage_odd_random.csv");
  std::ofstream csv(csv_path, std::ios::out | std::ios::trunc);
  CHECK(csv.is_open()) << "failed to open csv path: " << csv_path;
  csv << "batch_size,group_id,heads,head_dim,"
         "s0_history,s0_realtime,s0_target,"
         "s1_history,s1_realtime,s1_target,"
         "s2_history,s2_realtime,s2_target,"
         "s3_history,s3_realtime,s3_target,"
         "one_stage_mask_build_ms,one_stage_device_ms,"
         "one_stage_mask_build_plus_device_ms,one_stage_wall_ms,"
         "batched_fused_device_ms,batched_fused_wall_ms,"
         "speedup_dev,speedup_wall,delta_dev_ms,delta_wall_ms,"
         "diff_max_abs,diff_mean_abs\n";

  std::mt19937_64 rng(static_cast<uint64_t>(seed));
  std::fprintf(stderr,
               "[MTGR][CUDA][Perf][TrueMultiBatchPartialVsOneStageOddRandom] "
               "groups_per_batch=%d warmup=%d repeat=%d seed=%d csv=%s\n",
               groups_per_batch,
               warmup,
               repeat,
               seed,
               csv_path.c_str());

  for (int batch_size : batch_sizes) {
    CHECK_GT(batch_size, 0);
    CHECK_LE(batch_size, 4);
    std::unordered_set<std::string> seen;
    double total_base_mask_build = 0.0;
    double total_base_device = 0.0;
    double total_base_mask_plus_device = 0.0;
    double total_base_wall = 0.0;
    double total_fused_device = 0.0;
    double total_fused_wall = 0.0;
    double total_speedup_dev = 0.0;
    double total_speedup_wall = 0.0;
    double min_speedup_dev = std::numeric_limits<double>::infinity();
    double max_speedup_dev = 0.0;
    double worst_diff_max = 0.0;
    double worst_diff_mean = 0.0;
    int regressions = 0;
    int precision_regressions = 0;

    for (int group_id = 1; group_id <= groups_per_batch; ++group_id) {
      std::vector<MTGRAttentionTestShape> shapes;
      std::vector<MultiBatchProbeSequence> batch;
      const bool accepted = build_unique_true_multi_batch_partial_group(
          &rng, batch_size, device_, &seen, &shapes, &batch);

      ASSERT_TRUE(accepted)
          << "failed to build unique odd-length partial group";
      const auto avg =
          run_true_fused_partial_batch_vs_one_stage(&batch, warmup, repeat);
      const double base_mask_build_plus_device_ms =
          avg.base_mask_build_ms + avg.base_device_total_ms;
      const double speedup_dev =
          base_mask_build_plus_device_ms / avg.fused_device_total_ms;
      const double speedup_wall =
          avg.base_wall_total_ms / avg.fused_wall_total_ms;
      const double delta_dev =
          base_mask_build_plus_device_ms - avg.fused_device_total_ms;
      const double delta_wall =
          avg.base_wall_total_ms - avg.fused_wall_total_ms;

      if (speedup_dev < 1.0) {
        ++regressions;
      }
      if (!(avg.diff_max_abs < 1.0e-3) || !(avg.diff_mean_abs < 1.0e-5)) {
        ++precision_regressions;
      }

      total_base_mask_build += avg.base_mask_build_ms;
      total_base_device += avg.base_device_total_ms;
      total_base_mask_plus_device += base_mask_build_plus_device_ms;
      total_base_wall += avg.base_wall_total_ms;
      total_fused_device += avg.fused_device_total_ms;
      total_fused_wall += avg.fused_wall_total_ms;
      total_speedup_dev += speedup_dev;
      total_speedup_wall += speedup_wall;
      min_speedup_dev = std::min(min_speedup_dev, speedup_dev);
      max_speedup_dev = std::max(max_speedup_dev, speedup_dev);
      worst_diff_max = std::max(worst_diff_max, avg.diff_max_abs);
      worst_diff_mean = std::max(worst_diff_mean, avg.diff_mean_abs);

      std::ostringstream row;
      row << std::fixed << std::setprecision(6) << batch_size << "," << group_id
          << "," << shapes.front().heads << "," << shapes.front().head_dim;
      append_true_multi_batch_shape_cells(row, shapes);
      row << "," << avg.base_mask_build_ms << "," << avg.base_device_total_ms
          << "," << base_mask_build_plus_device_ms << ","
          << avg.base_wall_total_ms << "," << avg.fused_device_total_ms << ","
          << avg.fused_wall_total_ms << "," << speedup_dev << ","
          << speedup_wall << "," << delta_dev << "," << delta_wall << ","
          << avg.diff_max_abs << "," << avg.diff_mean_abs;
      csv << row.str() << "\n";
      csv.flush();
      std::fprintf(
          stderr,
          "[MTGR][CUDA][Perf][TrueMultiBatchPartialVsOneStageOddRandom "
          "batch=%d group=%d/%d] %s\n",
          batch_size,
          group_id,
          groups_per_batch,
          row.str().c_str());
      std::fflush(stderr);

      EXPECT_LT(avg.diff_max_abs, 1.0e-3);
      EXPECT_LT(avg.diff_mean_abs, 1.0e-5);
    }

    const double inv = 1.0 / static_cast<double>(groups_per_batch);
    std::fprintf(stderr,
                 "[MTGR][CUDA][Perf][TrueMultiBatchPartialVsOneStageOddRandom]"
                 "[Summary] batch_size=%d groups=%d regressions=%d "
                 "precision_regressions=%d avg_base_mask_build_ms=%.6f "
                 "avg_base_device_ms=%.6f avg_base_mask_plus_device_ms=%.6f "
                 "avg_base_wall_ms=%.6f avg_fused_device_ms=%.6f "
                 "avg_fused_wall_ms=%.6f avg_speedup_dev=%.6f "
                 "avg_speedup_wall=%.6f min_speedup_dev=%.6f "
                 "max_speedup_dev=%.6f worst_diff_max=%.6e "
                 "worst_diff_mean=%.6e\n",
                 batch_size,
                 groups_per_batch,
                 regressions,
                 precision_regressions,
                 total_base_mask_build * inv,
                 total_base_device * inv,
                 total_base_mask_plus_device * inv,
                 total_base_wall * inv,
                 total_fused_device * inv,
                 total_fused_wall * inv,
                 total_speedup_dev * inv,
                 total_speedup_wall * inv,
                 min_speedup_dev,
                 max_speedup_dev,
                 worst_diff_max,
                 worst_diff_mean);
    std::fflush(stderr);
  }
}

TEST_F(MTGRAttentionOneVsMultiStagePerfTest, TrueMultiBatchFusedNoMatchSmoke) {
  std::vector<MTGRAttentionTestShape> shapes(4);
  for (auto& shape : shapes) {
    shape.heads = 8;
    shape.kv_heads = 8;
    shape.head_dim = 128;
    shape.context = 8;
    shape.matched_prefix = 0;
  }
  shapes[0].history = 2048;
  shapes[0].realtime = 512;
  shapes[0].target = 1600;
  shapes[1].history = 1985;
  shapes[1].realtime = 447;
  shapes[1].target = 1537;
  shapes[2].history = 2113;
  shapes[2].realtime = 385;
  shapes[2].target = 1409;
  shapes[3].history = 1793;
  shapes[3].realtime = 577;
  shapes[3].target = 1665;

  std::vector<MultiBatchProbeSequence> batch;
  batch.reserve(shapes.size());
  for (const auto& shape : shapes) {
    batch.push_back(make_probe_sequence(shape, device_));
  }

  const int warmup = env_int("MTGR_BATCHED_FUSED_WARMUP", 1);
  const int repeat = env_int("MTGR_BATCHED_FUSED_REPEAT", 3);
  const auto avg = run_true_fused_no_match_batch(&batch, warmup, repeat);
  const double device_speedup =
      avg.base_device_total_ms / avg.fused_device_total_ms;
  const double wall_speedup = avg.base_wall_total_ms / avg.fused_wall_total_ms;

  std::fprintf(stderr,
               "[MTGR][CUDA][Perf][TrueMultiBatchFusedNoMatch] "
               "wrapper_fused_device=%.6f batched_fused_device=%.6f "
               "device_speedup=%.6f wrapper_fused_wall=%.6f "
               "batched_fused_wall=%.6f wall_speedup=%.6f "
               "diff_max=%.6e diff_mean=%.6e\n",
               avg.base_device_total_ms,
               avg.fused_device_total_ms,
               device_speedup,
               avg.base_wall_total_ms,
               avg.fused_wall_total_ms,
               wall_speedup,
               avg.diff_max_abs,
               avg.diff_mean_abs);
  std::fflush(stderr);

  EXPECT_LT(avg.diff_max_abs, 1.0e-3);
  EXPECT_LT(avg.diff_mean_abs, 1.0e-5);
}

TEST_F(MTGRAttentionOneVsMultiStagePerfTest, TrueMultiBatchFusedPartialSmoke) {
  std::vector<MTGRAttentionTestShape> shapes(4);
  for (auto& shape : shapes) {
    shape.heads = 8;
    shape.kv_heads = 8;
    shape.head_dim = 128;
    shape.context = 8;
  }
  shapes[0].history = 2048;
  shapes[0].realtime = 512;
  shapes[0].target = 1600;
  shapes[0].matched_prefix = shapes[0].history + shapes[0].context + 409;
  shapes[1].history = 1985;
  shapes[1].realtime = 447;
  shapes[1].target = 1537;
  shapes[1].matched_prefix = shapes[1].history + shapes[1].context + 347;
  shapes[2].history = 2113;
  shapes[2].realtime = 385;
  shapes[2].target = 1409;
  shapes[2].matched_prefix = shapes[2].history + shapes[2].context + 289;
  shapes[3].history = 1793;
  shapes[3].realtime = 577;
  shapes[3].target = 1665;
  shapes[3].matched_prefix = shapes[3].history + shapes[3].context + 449;

  std::vector<MultiBatchProbeSequence> batch;
  batch.reserve(shapes.size());
  for (const auto& shape : shapes) {
    batch.push_back(make_probe_sequence(shape, device_));
  }

  const int warmup = env_int("MTGR_BATCHED_FUSED_WARMUP", 1);
  const int repeat = env_int("MTGR_BATCHED_FUSED_REPEAT", 3);
  const auto avg = run_true_fused_partial_batch(&batch, warmup, repeat);
  const double device_speedup =
      avg.base_device_total_ms / avg.fused_device_total_ms;
  const double wall_speedup = avg.base_wall_total_ms / avg.fused_wall_total_ms;

  std::fprintf(stderr,
               "[MTGR][CUDA][Perf][TrueMultiBatchFusedPartial] "
               "wrapper_fused_device=%.6f batched_fused_device=%.6f "
               "device_speedup=%.6f wrapper_fused_wall=%.6f "
               "batched_fused_wall=%.6f wall_speedup=%.6f "
               "diff_max=%.6e diff_mean=%.6e\n",
               avg.base_device_total_ms,
               avg.fused_device_total_ms,
               device_speedup,
               avg.base_wall_total_ms,
               avg.fused_wall_total_ms,
               wall_speedup,
               avg.diff_max_abs,
               avg.diff_mean_abs);
  std::fflush(stderr);

  EXPECT_LT(avg.diff_max_abs, 1.0e-3);
  EXPECT_LT(avg.diff_mean_abs, 1.0e-5);
}

TEST_F(MTGRAttentionOneVsMultiStagePerfTest,
       TrueMultiBatchFusedPartialVsOneStageSmoke) {
  std::vector<MTGRAttentionTestShape> shapes(4);
  for (auto& shape : shapes) {
    shape.heads = 8;
    shape.kv_heads = 8;
    shape.head_dim = 128;
    shape.context = 8;
  }
  shapes[0].history = 2048;
  shapes[0].realtime = 512;
  shapes[0].target = 1600;
  shapes[0].matched_prefix = shapes[0].history + shapes[0].context + 409;
  shapes[1].history = 1985;
  shapes[1].realtime = 447;
  shapes[1].target = 1537;
  shapes[1].matched_prefix = shapes[1].history + shapes[1].context + 347;
  shapes[2].history = 2113;
  shapes[2].realtime = 385;
  shapes[2].target = 1409;
  shapes[2].matched_prefix = shapes[2].history + shapes[2].context + 289;
  shapes[3].history = 1793;
  shapes[3].realtime = 577;
  shapes[3].target = 1665;
  shapes[3].matched_prefix = shapes[3].history + shapes[3].context + 449;

  std::vector<MultiBatchProbeSequence> batch;
  batch.reserve(shapes.size());
  for (const auto& shape : shapes) {
    batch.push_back(make_probe_sequence(shape, device_));
  }

  const int warmup = env_int("MTGR_BATCHED_FUSED_WARMUP", 1);
  const int repeat = env_int("MTGR_BATCHED_FUSED_REPEAT", 3);
  const auto avg =
      run_true_fused_partial_batch_vs_one_stage(&batch, warmup, repeat);
  const double base_mask_build_plus_device_ms =
      avg.base_mask_build_ms + avg.base_device_total_ms;
  const double device_speedup =
      base_mask_build_plus_device_ms / avg.fused_device_total_ms;
  const double wall_speedup = avg.base_wall_total_ms / avg.fused_wall_total_ms;

  std::fprintf(stderr,
               "[MTGR][CUDA][Perf][TrueMultiBatchFusedPartialVsOneStage] "
               "one_stage_mask_build=%.6f one_stage_device=%.6f "
               "one_stage_mask_plus_device=%.6f batched_fused_device=%.6f "
               "device_speedup=%.6f one_stage_wall=%.6f "
               "batched_fused_wall=%.6f wall_speedup=%.6f "
               "diff_max=%.6e diff_mean=%.6e\n",
               avg.base_mask_build_ms,
               avg.base_device_total_ms,
               base_mask_build_plus_device_ms,
               avg.fused_device_total_ms,
               device_speedup,
               avg.base_wall_total_ms,
               avg.fused_wall_total_ms,
               wall_speedup,
               avg.diff_max_abs,
               avg.diff_mean_abs);
  std::fflush(stderr);

  EXPECT_LT(avg.diff_max_abs, 1.0e-3);
  EXPECT_LT(avg.diff_mean_abs, 1.0e-5);
}

}  // namespace xllm::kernel::cuda::test
