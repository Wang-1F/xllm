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
#include <torch_npu/torch_npu.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "mtgr_attention_test.h"

namespace xllm::kernel::npu::test {
namespace {

struct CasePerfResult {
  OneStagePerf one_stage;
  MTGRForwardPerf multi_stage;
};

int read_env_int(const char* name, int default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }
  return std::max(1, std::atoi(value));
}

std::vector<int64_t> read_env_i64_list(const char* name,
                                       const std::vector<int64_t>& default_values) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_values;
  }
  std::vector<int64_t> parsed;
  std::stringstream ss(value);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token.empty()) {
      continue;
    }
    parsed.push_back(std::stoll(token));
  }
  return parsed.empty() ? default_values : parsed;
}

bool env_enabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

void write_csv_row(std::ofstream& csv, const std::vector<std::string>& cols) {
  for (size_t i = 0; i < cols.size(); ++i) {
    if (i > 0) {
      csv << ',';
    }
    csv << cols[i];
  }
  csv << '\n';
}

void write_grouped_csv_header(std::ofstream& csv,
                              const std::vector<std::string>& common_cols,
                              const std::vector<std::string>& one_cols,
                              const std::vector<std::string>& multi_cols,
                              const std::vector<std::string>& speedup_cols,
                              const std::vector<std::string>& delta_cols) {
  std::vector<std::string> level1;
  auto append_group = [&](const std::string& group_name, size_t count) {
    if (count == 0) {
      return;
    }
    level1.push_back(group_name);
    for (size_t i = 1; i < count; ++i) {
      level1.emplace_back("");
    }
  };

  append_group("common", common_cols.size());
  append_group("one_stage", one_cols.size());
  append_group("multi_stage", multi_cols.size());
  append_group("speedup", speedup_cols.size());
  append_group("delta_ms", delta_cols.size());

  std::vector<std::string> level2;
  level2.insert(level2.end(), common_cols.begin(), common_cols.end());
  level2.insert(level2.end(), one_cols.begin(), one_cols.end());
  level2.insert(level2.end(), multi_cols.begin(), multi_cols.end());
  level2.insert(level2.end(), speedup_cols.begin(), speedup_cols.end());
  level2.insert(level2.end(), delta_cols.begin(), delta_cols.end());

  write_csv_row(csv, level1);
  write_csv_row(csv, level2);
}

std::filesystem::path resolve_repo_root() {
  namespace fs = std::filesystem;
  fs::path cur = fs::current_path();
  while (true) {
    if (fs::exists(cur / "xllm/core/kernels/npu/tests/CMakeLists.txt") &&
        fs::exists(cur / "xllm/core/layers/npu_torch/mtgr_attention.cpp")) {
      return cur;
    }
    if (cur == cur.root_path() || cur == cur.parent_path()) {
      break;
    }
    cur = cur.parent_path();
  }
  return fs::current_path();
}

std::filesystem::path resolve_compact_csv_path() {
  if (const char* custom_path = std::getenv("XLLM_MTGR_SWEEP_CSV_PATH");
      custom_path != nullptr && custom_path[0] != '\0') {
    return custom_path;
  }
  return resolve_repo_root() /
         "xllm/core/kernels/npu/tests/"
         "one_vs_preplanned_single_stream_variant_compact_excl_h2d_maskopt.csv";
}

std::filesystem::path resolve_partial_rt_compact_csv_path() {
  if (const char* custom_path = std::getenv("XLLM_MTGR_SWEEP_PARTIAL_RT_CSV_PATH");
      custom_path != nullptr && custom_path[0] != '\0') {
    return custom_path;
  }
  return resolve_repo_root() /
         "xllm/core/kernels/npu/tests/"
         "one_vs_preplanned_single_stream_variant_partial_rt80_compact_excl_h2d_maskopt.csv";
}

std::vector<std::string> build_stage_csv_cols(const std::vector<std::string>& stage_names) {
  std::vector<std::string> cols;
  cols.reserve(stage_names.size() * 2 + 2);
  for (const auto& name : stage_names) {
    cols.push_back(name + ".workspace_ms");
    cols.push_back(name + ".exec_ms");
  }
  cols.push_back("device_total_ms");
  cols.push_back("wall_total_ms");
  return cols;
}

const MTGRStagePerf& require_stage(const MTGRForwardPerf& perf, const std::string& stage_name) {
  for (const auto& stage : perf.stages) {
    if (stage.name == stage_name) {
      return stage;
    }
  }
  CHECK(false) << "Stage not found in path=" << perf.path_name
               << ", stage_name=" << stage_name;
  return perf.stages.front();
}

class MTGRAttentionPerfTest : public ::testing::Test {
 protected:
  static constexpr int32_t kDeviceId = 5;

  static void SetUpTestSuite() {
    torch_npu::init_npu("npu:" + std::to_string(kDeviceId));
  }

  static void TearDownTestSuite() { torch_npu::finalize_npu(); }

  void SetUp() override {
    device_ = torch::Device(torch::kPrivateUse1, kDeviceId);
    fp_opts_ = torch::TensorOptions().dtype(torch::kBFloat16).device(device_);
  }

  CasePerfResult run_case(const MTGRCaseConfig& cfg,
                          int warmup_iters,
                          int repeat_iters,
                          const char* expected_path_name,
                          bool use_fused_target_update = false,
                          bool use_fused_target_update_auto = false,
                          bool use_fused_target_update_v4 = false) {
    auto prepared_case = prepare_mtgr_case(cfg, fp_opts_, device_);
    auto one_stage = profile_one_stage_maskopt(prepared_case, warmup_iters, repeat_iters);
    auto multi_stage = profile_mtgr_forward(
        prepared_case,
        warmup_iters,
        repeat_iters,
        use_fused_target_update,
        use_fused_target_update_auto,
        use_fused_target_update_v4);

    EXPECT_EQ(multi_stage.path_name, expected_path_name);
    return CasePerfResult{
        .one_stage = std::move(one_stage),
        .multi_stage = std::move(multi_stage),
    };
  }

  torch::Device device_{torch::kCPU};
  torch::TensorOptions fp_opts_;
};

TEST_F(MTGRAttentionPerfTest,
       SweepOneStageVsPreplannedSingleStreamVariantCompactExclH2DMaskopt) {
  const bool quick = env_enabled("XLLM_MTGR_SWEEP_QUICK");
  const bool use_fused_target_update_auto =
      env_enabled("XLLM_MTGR_USE_FUSED_TARGET_UPDATE_AUTO");
  const bool use_fused_target_update_v4 = env_enabled("XLLM_MTGR_USE_FUSED_TARGET_UPDATE_V4");
  const bool use_fused_target_update =
      env_enabled("XLLM_MTGR_USE_FUSED_TARGET_UPDATE") || use_fused_target_update_auto ||
      use_fused_target_update_v4;
  const int warmup_iters = read_env_int("XLLM_MTGR_SWEEP_WARMUP", quick ? 1 : 5);
  const int repeat_iters = read_env_int("XLLM_MTGR_SWEEP_REPEAT", quick ? 1 : 20);

  const std::vector<int64_t> heads_list = read_env_i64_list(
      "XLLM_MTGR_SWEEP_HEADS", quick ? std::vector<int64_t>{4} : std::vector<int64_t>{4, 8, 16});
  const std::vector<int64_t> head_dim_list =
      read_env_i64_list("XLLM_MTGR_SWEEP_HEAD_DIMS",
                        quick ? std::vector<int64_t>{64} : std::vector<int64_t>{32, 64, 128});
  const std::vector<int64_t> history_list =
      read_env_i64_list("XLLM_MTGR_SWEEP_HISTORY",
                        quick ? std::vector<int64_t>{1024} : std::vector<int64_t>{1024, 2048, 4096});
  const std::vector<int64_t> realtime_list =
      read_env_i64_list("XLLM_MTGR_SWEEP_REALTIME",
                        quick ? std::vector<int64_t>{128} : std::vector<int64_t>{128, 256, 512});
  const std::vector<int64_t> target_list =
      read_env_i64_list("XLLM_MTGR_SWEEP_TARGET",
                        quick ? std::vector<int64_t>{800} : std::vector<int64_t>{800, 1600, 2400});
  const std::vector<std::string> stage_names =
      use_fused_target_update
          ? std::vector<std::string>{"rt_tgt_on_hcr_trapezoid",
                                     "hist_fa",
                                     "ctx_on_hc",
                                     use_fused_target_update_auto ? "target_fused_update_v4"
                                     : use_fused_target_update_v4 ? "target_fused_update_v4"
                                                                  : "target_fused_update"}
          : std::vector<std::string>{"rt_tgt_on_hcr_trapezoid",
                                     "hist_fa",
                                     "ctx_on_hc",
                                     "tgt_diag_bmm",
                                     "update_tgt_2way"};

  const size_t total_rows = heads_list.size() * head_dim_list.size() * history_list.size() *
                            realtime_list.size() * target_list.size();
  const auto csv_path = resolve_compact_csv_path();
  std::ofstream csv(csv_path);
  ASSERT_TRUE(csv.is_open()) << "failed to open csv: " << csv_path;

  write_grouped_csv_header(csv,
                           {"mask_build_mode",
                            "heads",
                            "head_dim",
                            "history",
                            "real_time",
                            "target"},
                           {"mask_build_ms",
                            "h2d_ms",
                            "workspace_ms",
                            "fia_ms",
                            "device_total_ms",
                            "wall_total_ms"},
                           build_stage_csv_cols(stage_names),
                           {"one_device_total_ms/multi_device_total_ms",
                            "one_wall_total_ms/multi_wall_total_ms"},
                           {"one_device_total_ms-multi_device_total_ms",
                            "one_wall_total_ms-multi_wall_total_ms"});
  csv << std::fixed << std::setprecision(6);

  size_t row_idx = 0;
  for (const auto heads : heads_list) {
    for (const auto head_dim : head_dim_list) {
      for (const auto history : history_list) {
        for (const auto realtime : realtime_list) {
          for (const auto target : target_list) {
            ++row_idx;

            MTGRCaseConfig cfg;
            cfg.num_heads = heads;
            cfg.num_kv_heads = heads;
            cfg.head_dim = head_dim;
            cfg.history = history;
            cfg.context = 8;
            cfg.real_time = realtime;
            cfg.target = target;
            cfg.matched_prefix = 0;

            torch::manual_seed(20260429 + static_cast<int64_t>(row_idx));
            auto result = run_case(cfg,
                                   warmup_iters,
                                   repeat_iters,
                                   use_fused_target_update
                                       ? (use_fused_target_update_auto
                                              ? "no_matched_fused_target_update_v4"
                                          : use_fused_target_update_v4
                                              ? "no_matched_fused_target_update_v4"
                                              : "no_matched_fused_target_update")
                                       : "no_matched",
                                   use_fused_target_update,
                                   use_fused_target_update_auto,
                                   use_fused_target_update_v4);

            const double speedup_device =
                result.one_stage.device_total_ms / result.multi_stage.device_total_ms;
            const double speedup_wall =
                result.one_stage.wall_total_ms / result.multi_stage.wall_total_ms;
            const double delta_device =
                result.one_stage.device_total_ms - result.multi_stage.device_total_ms;
            const double delta_wall =
                result.one_stage.wall_total_ms - result.multi_stage.wall_total_ms;

            csv << "device," << heads << ',' << head_dim << ',' << history << ','
                << realtime << ',' << target << ',' << result.one_stage.mask_build_ms << ','
                << result.one_stage.h2d_ms << ',' << result.one_stage.workspace_ms << ','
                << result.one_stage.fia_ms << ',' << result.one_stage.device_total_ms << ','
                << result.one_stage.wall_total_ms;
            for (const auto& stage_name : stage_names) {
              const auto& stage = require_stage(result.multi_stage, stage_name);
              csv << ',' << stage.workspace_ms << ',' << stage.exec_ms;
            }
            csv << ',' << result.multi_stage.device_total_ms << ','
                << result.multi_stage.wall_total_ms << ',' << speedup_device << ','
                << speedup_wall << ',' << delta_device << ',' << delta_wall << '\n';

            std::fprintf(stderr,
                         "[MTGRPerf][no_match][%zu/%zu] heads=%lld head_dim=%lld "
                         "h=%lld r=%lld t=%lld one(mask=%.6f,ws=%.6f,fia=%.6f,"
                         "dev=%.6f,wall=%.6f) multi(dev=%.6f,wall=%.6f) "
                         "target_impl=%s speedup_wall=%.6f\n",
                         row_idx,
                         total_rows,
                         static_cast<long long>(heads),
                         static_cast<long long>(head_dim),
                         static_cast<long long>(history),
                         static_cast<long long>(realtime),
                         static_cast<long long>(target),
                         result.one_stage.mask_build_ms,
                         result.one_stage.workspace_ms,
                         result.one_stage.fia_ms,
                         result.one_stage.device_total_ms,
                         result.one_stage.wall_total_ms,
                         result.multi_stage.device_total_ms,
                         result.multi_stage.wall_total_ms,
                         use_fused_target_update
                             ? (use_fused_target_update_auto ? "auto(v4@no_match)"
                                : use_fused_target_update_v4 ? "fused_v4"
                                                             : "fused_v1")
                             : "attention_update",
                         speedup_wall);
          }
        }
      }
    }
  }

  csv.flush();
  std::fprintf(stderr,
               "[MTGRPerf][no_match] rows=%zu warmup=%d repeat=%d mode=%s csv_path=%s\n",
               row_idx,
               warmup_iters,
               repeat_iters,
               quick ? "quick" : "full",
               csv_path.c_str());
  std::fflush(stderr);

  EXPECT_EQ(row_idx, total_rows);
}

TEST_F(MTGRAttentionPerfTest,
       SweepOneStageVsPreplannedSingleStreamVariantPartialRt80CompactExclH2DMaskopt) {
  const bool quick = env_enabled("XLLM_MTGR_SWEEP_QUICK");
  const bool use_fused_target_update_auto =
      env_enabled("XLLM_MTGR_USE_FUSED_TARGET_UPDATE_AUTO");
  const bool use_fused_target_update_v4 = env_enabled("XLLM_MTGR_USE_FUSED_TARGET_UPDATE_V4");
  const bool use_fused_target_update =
      env_enabled("XLLM_MTGR_USE_FUSED_TARGET_UPDATE") || use_fused_target_update_auto ||
      use_fused_target_update_v4;
  const int warmup_iters = read_env_int("XLLM_MTGR_SWEEP_WARMUP", quick ? 1 : 5);
  const int repeat_iters = read_env_int("XLLM_MTGR_SWEEP_REPEAT", quick ? 1 : 20);

  const std::vector<int64_t> heads_list = read_env_i64_list(
      "XLLM_MTGR_SWEEP_HEADS", quick ? std::vector<int64_t>{4} : std::vector<int64_t>{4, 8, 16});
  const std::vector<int64_t> head_dim_list =
      read_env_i64_list("XLLM_MTGR_SWEEP_HEAD_DIMS",
                        quick ? std::vector<int64_t>{64} : std::vector<int64_t>{32, 64, 128});
  const std::vector<int64_t> history_list =
      read_env_i64_list("XLLM_MTGR_SWEEP_HISTORY",
                        quick ? std::vector<int64_t>{1024} : std::vector<int64_t>{1024, 2048, 4096});
  const std::vector<int64_t> realtime_list =
      read_env_i64_list("XLLM_MTGR_SWEEP_REALTIME",
                        quick ? std::vector<int64_t>{128} : std::vector<int64_t>{128, 256, 512});
  const std::vector<int64_t> target_list =
      read_env_i64_list("XLLM_MTGR_SWEEP_TARGET",
                        quick ? std::vector<int64_t>{800} : std::vector<int64_t>{800, 1600, 2400});
  const std::vector<std::string> stage_names =
      use_fused_target_update
          ? std::vector<std::string>{"scatter_pa_kv_cache",
                                     "rt_unmatched_pa_s3",
                                     "target_prefix_pa_full",
                                     use_fused_target_update_auto ? "target_fused_update_v4"
                                     : use_fused_target_update_v4 ? "target_fused_update_v4"
                                                                  : "target_fused_update"}
          : std::vector<std::string>{"scatter_pa_kv_cache",
                                     "tgt_diag_bmm",
                                     "rt_unmatched_pa_s3",
                                     "target_prefix_pa_full",
                                     "update_tgt_2way"};

  const size_t total_rows = heads_list.size() * head_dim_list.size() * history_list.size() *
                            realtime_list.size() * target_list.size();
  const auto csv_path = resolve_partial_rt_compact_csv_path();
  std::ofstream csv(csv_path);
  ASSERT_TRUE(csv.is_open()) << "failed to open csv: " << csv_path;

  write_grouped_csv_header(csv,
                           {"mask_build_mode",
                            "heads",
                            "head_dim",
                            "history",
                            "real_time",
                            "realtime_matched",
                            "target"},
                           {"mask_build_ms",
                            "h2d_ms",
                            "workspace_ms",
                            "fia_ms",
                            "device_total_ms",
                            "wall_total_ms"},
                           build_stage_csv_cols(stage_names),
                           {"one_device_total_ms/multi_device_total_ms",
                            "one_wall_total_ms/multi_wall_total_ms"},
                           {"one_device_total_ms-multi_device_total_ms",
                            "one_wall_total_ms-multi_wall_total_ms"});
  csv << std::fixed << std::setprecision(6);

  size_t row_idx = 0;
  for (const auto heads : heads_list) {
    for (const auto head_dim : head_dim_list) {
      for (const auto history : history_list) {
        for (const auto realtime : realtime_list) {
          for (const auto target : target_list) {
            ++row_idx;

            MTGRCaseConfig cfg;
            cfg.num_heads = heads;
            cfg.num_kv_heads = heads;
            cfg.head_dim = head_dim;
            cfg.history = history;
            cfg.context = 8;
            cfg.real_time = realtime;
            cfg.target = target;

            const int64_t realtime_matched = partial_rt_matched_tokens(realtime);
            cfg.matched_prefix = cfg.history + cfg.context + realtime_matched;

            torch::manual_seed(20270429 + static_cast<int64_t>(row_idx));
            auto result = run_case(cfg,
                                   warmup_iters,
                                   repeat_iters,
                                   use_fused_target_update
                                       ? (use_fused_target_update_auto
                                              ? "partial_rt_matched_fused_target_update_v4"
                                          : use_fused_target_update_v4
                                              ? "partial_rt_matched_fused_target_update_v4"
                                              : "partial_rt_matched_fused_target_update")
                                       : "partial_rt_matched",
                                   use_fused_target_update,
                                   use_fused_target_update_auto,
                                   use_fused_target_update_v4);

            const double speedup_device =
                result.one_stage.device_total_ms / result.multi_stage.device_total_ms;
            const double speedup_wall =
                result.one_stage.wall_total_ms / result.multi_stage.wall_total_ms;
            const double delta_device =
                result.one_stage.device_total_ms - result.multi_stage.device_total_ms;
            const double delta_wall =
                result.one_stage.wall_total_ms - result.multi_stage.wall_total_ms;

            csv << "device," << heads << ',' << head_dim << ',' << history << ','
                << realtime << ',' << realtime_matched << ',' << target << ','
                << result.one_stage.mask_build_ms << ',' << result.one_stage.h2d_ms << ','
                << result.one_stage.workspace_ms << ',' << result.one_stage.fia_ms << ','
                << result.one_stage.device_total_ms << ',' << result.one_stage.wall_total_ms;
            for (const auto& stage_name : stage_names) {
              const auto& stage = require_stage(result.multi_stage, stage_name);
              csv << ',' << stage.workspace_ms << ',' << stage.exec_ms;
            }
            csv << ',' << result.multi_stage.device_total_ms << ','
                << result.multi_stage.wall_total_ms << ',' << speedup_device << ','
                << speedup_wall << ',' << delta_device << ',' << delta_wall << '\n';

            std::fprintf(stderr,
                         "[MTGRPerf][partial_rt80][%zu/%zu] heads=%lld head_dim=%lld "
                         "h=%lld r=%lld matched=%lld t=%lld "
                         "one(mask=%.6f,ws=%.6f,fia=%.6f,dev=%.6f,wall=%.6f) "
                         "multi(dev=%.6f,wall=%.6f) target_impl=%s speedup_wall=%.6f\n",
                         row_idx,
                         total_rows,
                         static_cast<long long>(heads),
                         static_cast<long long>(head_dim),
                         static_cast<long long>(history),
                         static_cast<long long>(realtime),
                         static_cast<long long>(realtime_matched),
                         static_cast<long long>(target),
                         result.one_stage.mask_build_ms,
                         result.one_stage.workspace_ms,
                         result.one_stage.fia_ms,
                         result.one_stage.device_total_ms,
                         result.one_stage.wall_total_ms,
                         result.multi_stage.device_total_ms,
                         result.multi_stage.wall_total_ms,
                         use_fused_target_update
                             ? (use_fused_target_update_auto ? "auto(v4@partial_rt)"
                                : use_fused_target_update_v4 ? "fused_v4"
                                                             : "fused_v1")
                             : "attention_update",
                         speedup_wall);
          }
        }
      }
    }
  }

  csv.flush();
  std::fprintf(stderr,
               "[MTGRPerf][partial_rt80] rows=%zu warmup=%d repeat=%d mode=%s csv_path=%s\n",
               row_idx,
               warmup_iters,
               repeat_iters,
               quick ? "quick" : "full",
               csv_path.c_str());
  std::fflush(stderr);

  EXPECT_EQ(row_idx, total_rows);
}

}  // namespace
}  // namespace xllm::kernel::npu::test
