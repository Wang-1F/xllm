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

#include "mtgr_attention_contract.h"

#include <glog/logging.h>

#include <cstdint>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "../mtgr_attenion_test.h"

namespace xllm::kernel::cuda::test::mtgr_attention_harness {
namespace {

int64_t sample_odd_len(std::mt19937_64* rng, int64_t lo, int64_t hi) {
  CHECK(rng != nullptr);
  std::uniform_int_distribution<int64_t> dist(lo, hi);
  for (int attempt = 0; attempt < 1024; ++attempt) {
    const int64_t value = dist(*rng);
    if ((value % 32) != 0 && (value % 64) != 0) {
      return value;
    }
  }
  const int64_t value = dist(*rng);
  return value == hi ? value - 1 : value + 1;
}

std::string shape_key(const MTGRAttentionHarnessMetadata& metadata) {
  return std::to_string(metadata.num_heads) + "_" +
         std::to_string(metadata.head_dim) + "_" +
         std::to_string(metadata.history) + "_" +
         std::to_string(metadata.realtime) + "_" +
         std::to_string(metadata.target);
}

MTGRAttentionTestShape to_legacy_shape(
    const MTGRAttentionHarnessMetadata& metadata) {
  MTGRAttentionTestShape shape;
  shape.heads = metadata.num_heads;
  shape.kv_heads = metadata.num_kv_heads;
  shape.head_dim = metadata.head_dim;
  shape.history = metadata.history;
  shape.context = metadata.context;
  shape.realtime = metadata.realtime;
  shape.target = metadata.target;
  shape.matched_prefix = metadata.matched_prefix;
  return shape;
}

}  // namespace

std::vector<MTGRAttentionHarnessMetadata> generate_odd_length_metadata_pairs(
    int64_t pair_count,
    uint64_t seed) {
  CHECK_GT(pair_count, 0);
  std::mt19937_64 rng(seed);
  const std::vector<int64_t> heads_all = {4, 8, 12};
  const std::vector<int64_t> head_dims_all = {64, 128};
  std::uniform_int_distribution<int> heads_dist(
      0, static_cast<int>(heads_all.size() - 1));
  std::uniform_int_distribution<int> head_dim_dist(
      0, static_cast<int>(head_dims_all.size() - 1));

  std::vector<MTGRAttentionHarnessMetadata> metadata_cases;
  metadata_cases.reserve(static_cast<size_t>(pair_count * 2));
  std::unordered_set<std::string> seen;
  for (int64_t attempt = 0;
       attempt < pair_count * 64 &&
       static_cast<int64_t>(metadata_cases.size()) < pair_count * 2;
       ++attempt) {
    MTGRAttentionHarnessMetadata partial;
    partial.pair_id = static_cast<int64_t>(metadata_cases.size() / 2) + 1;
    partial.num_heads = heads_all[heads_dist(rng)];
    partial.num_kv_heads = partial.num_heads;
    partial.head_dim = head_dims_all[head_dim_dist(rng)];
    partial.history = sample_odd_len(&rng, 1350, 4096);
    partial.context = 8;
    partial.realtime = sample_odd_len(&rng, 100, 600);
    partial.target = sample_odd_len(&rng, 800, 2400);
    partial.matched_prefix =
        partial.history + partial.context + (partial.realtime * 4) / 5;
    if (!seen.insert(shape_key(partial)).second) {
      continue;
    }

    auto no_match = partial;
    no_match.matched_prefix = 0;
    metadata_cases.push_back(no_match);
    metadata_cases.push_back(partial);
  }
  CHECK_EQ(static_cast<int64_t>(metadata_cases.size()), pair_count * 2)
      << "failed to generate enough unique odd-length MTGR attention cases";
  return metadata_cases;
}

MTGRAttentionCaseData build_case_data(
    const MTGRAttentionHarnessMetadata& metadata,
    const torch::Device& device) {
  CHECK_EQ(metadata.num_heads, metadata.num_kv_heads)
      << "prototype keeps GQA out of the benchmark contract";
  CHECK_GE(metadata.matched_prefix, 0);
  CHECK_LT(metadata.matched_prefix,
           metadata.history + metadata.context + metadata.realtime);

  auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const int64_t total = metadata.total_len();
  const int64_t live = metadata.live_len();
  auto full_query_bsnd = torch::randn(
                             {1, total, metadata.num_heads, metadata.head_dim},
                             options) *
                         0.05;
  auto full_key_bsnd =
      torch::randn({1, total, metadata.num_kv_heads, metadata.head_dim},
                   options) *
      0.05;
  auto full_value_bsnd =
      torch::randn({1, total, metadata.num_kv_heads, metadata.head_dim},
                   options) *
      0.05;

  MTGRAttentionCaseData data;
  data.harness_metadata = metadata;
  auto full_query_snd = full_query_bsnd.select(0, 0).contiguous();
  auto full_key_snd = full_key_bsnd.select(0, 0).contiguous();
  auto full_value_snd = full_value_bsnd.select(0, 0).contiguous();
  data.full_input.query =
      full_query_snd.view({total, metadata.num_heads * metadata.head_dim})
          .contiguous();
  data.full_input.key =
      full_key_snd.view({total, metadata.num_kv_heads * metadata.head_dim})
          .contiguous();
  data.full_input.value =
      full_value_snd.view({total, metadata.num_kv_heads * metadata.head_dim})
          .contiguous();
  data.live_input.query = full_query_snd.narrow(0, metadata.matched_prefix, live)
                              .contiguous()
                              .view({live, metadata.num_heads * metadata.head_dim});
  data.live_input.key = full_key_snd.narrow(0, metadata.matched_prefix, live)
                            .contiguous()
                            .view({live, metadata.num_kv_heads * metadata.head_dim});
  data.live_input.value = full_value_snd.narrow(0, metadata.matched_prefix, live)
                              .contiguous()
                              .view({live, metadata.num_kv_heads * metadata.head_dim});

  const auto legacy_shape = to_legacy_shape(metadata);
  data.live_input.metadata =
      make_mtgr_attention_metadata(legacy_shape, device, metadata.block_size);
  auto full_legacy_shape = legacy_shape;
  full_legacy_shape.matched_prefix = 0;
  data.full_input.metadata =
      make_mtgr_attention_metadata(
          full_legacy_shape, device, metadata.block_size);
  data.live_input.kv_cache = make_mtgr_kv_cache(
      legacy_shape, device, torch::kBFloat16, metadata.block_size);
  data.full_input.kv_cache = make_mtgr_kv_cache(
      full_legacy_shape, device, torch::kBFloat16, metadata.block_size);
  prefill_mtgr_matched_prefix_cache(
      full_key_bsnd,
      full_value_bsnd,
      legacy_shape,
      metadata.block_size,
      data.live_input.kv_cache);
  return data;
}

MTGRAttentionDiff compare_outputs(const torch::Tensor& reference,
                                  const torch::Tensor& candidate) {
  CHECK(reference.defined());
  CHECK(candidate.defined());
  CHECK_EQ(reference.sizes(), candidate.sizes());
  auto ref_f32 = reference.to(torch::kFloat32);
  auto cand_f32 = candidate.to(torch::kFloat32);
  auto diff = (ref_f32 - cand_f32).abs();
  return MTGRAttentionDiff{
      .all_finite = candidate.isfinite().all().item<bool>(),
      .max_abs = diff.max().item<double>(),
      .mean_abs = diff.mean().item<double>(),
  };
}

void write_perf_label_header(std::ostream& out) {
  out << "idx,backend,pair_id,mode,heads,kv_heads,head_dim,history,context,"
         "realtime,realtime_matched,target,total_q,matched_prefix,live_q,"
         "repeat_id\n";
}

void write_perf_label_row(std::ostream& out,
                          int64_t idx,
                          const std::string& backend,
                          const MTGRAttentionHarnessMetadata& metadata,
                          int64_t repeat_id) {
  out << idx << "," << backend << "," << metadata.pair_id << ","
      << metadata.mode_name() << "," << metadata.num_heads << ","
      << metadata.num_kv_heads << "," << metadata.head_dim << ","
      << metadata.history << "," << metadata.context << ","
      << metadata.realtime << "," << metadata.realtime_matched() << ","
      << metadata.target << "," << metadata.total_len() << ","
      << metadata.matched_prefix << "," << metadata.live_len() << ","
      << repeat_id << "\n";
}

}  // namespace xllm::kernel::cuda::test::mtgr_attention_harness
