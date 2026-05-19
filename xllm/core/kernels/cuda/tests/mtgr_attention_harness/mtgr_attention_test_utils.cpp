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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

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

std::vector<int32_t> make_block_table_host(int64_t block_count,
                                           int64_t block_base = 0) {
  std::vector<int32_t> table(static_cast<size_t>(block_count));
  for (int64_t i = 0; i < block_count; ++i) {
    table[static_cast<size_t>(i)] = static_cast<int32_t>(block_base + i);
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

std::vector<int32_t> four_segment_offsets(
    const MTGRAttentionTestShape& shape) {
  return {0,
          static_cast<int32_t>(shape.history),
          static_cast<int32_t>(shape.history + shape.context),
          static_cast<int32_t>(shape.history + shape.context + shape.realtime),
          static_cast<int32_t>(shape.total_len())};
}

void scatter_prefix_to_cache(const torch::Tensor& full_key_bsnd,
                             const torch::Tensor& full_value_bsnd,
                             const std::vector<int64_t>& slots_host,
                             int64_t matched_prefix,
                             torch::Tensor key_cache,
                             torch::Tensor value_cache) {
  if (matched_prefix <= 0) {
    return;
  }
  auto slots = torch::tensor(slots_host,
                             torch::TensorOptions()
                                 .dtype(torch::kInt64)
                                 .device(full_key_bsnd.device()))
                   .contiguous();
  auto key_flat = key_cache.view(
      {key_cache.size(0) * key_cache.size(1), key_cache.size(2), key_cache.size(3)});
  auto value_flat = value_cache.view({value_cache.size(0) * value_cache.size(1),
                                      value_cache.size(2),
                                      value_cache.size(3)});
  key_flat.index_copy_(
      0, slots, full_key_bsnd.select(0, 0).narrow(0, 0, matched_prefix));
  value_flat.index_copy_(
      0, slots, full_value_bsnd.select(0, 0).narrow(0, 0, matched_prefix));
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
      torch::tensor(four_segment_offsets(shape), i32_dev_opts)
          .view({1, 5})
          .contiguous();
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
    const auto offsets = four_segment_offsets(shape);
    segment_offsets.insert(segment_offsets.end(), offsets.begin(), offsets.end());
    total_q += shape.local_len();
    max_query_len = std::max(max_query_len, shape.local_len());
  }

  const bool all_no_match =
      std::all_of(matched_prefix_lens.begin(),
                  matched_prefix_lens.end(),
                  [](int64_t matched) { return matched == 0; });
  const bool all_partial =
      std::all_of(matched_prefix_lens.begin(),
                  matched_prefix_lens.end(),
                  [](int64_t matched) { return matched > 0; });

  xllm::layer::AttentionMetadata metadata;
  auto len_opts =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  auto i32_dev_opts =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  metadata.is_dummy = false;
  metadata.is_prefill = true;
  metadata.is_chunked_prefill = false;
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
  CHECK_EQ(full_key_bsnd.dim(), 4);
  CHECK_EQ(full_value_bsnd.dim(), 4);
  CHECK_EQ(full_key_bsnd.size(0), 1);
  CHECK_EQ(full_value_bsnd.sizes(), full_key_bsnd.sizes());
  const auto block_table_host =
      make_block_table_host((shape.total_len() + block_size - 1) / block_size);
  const auto slots_host =
      build_slot_mapping_host(block_table_host, block_size, 0, shape.matched_prefix);
  scatter_prefix_to_cache(full_key_bsnd,
                          full_value_bsnd,
                          slots_host,
                          shape.matched_prefix,
                          kv_cache.get_k_cache(),
                          kv_cache.get_v_cache());
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
  int64_t total_block_count = 0;
  int64_t max_block_count = 0;
  std::vector<std::vector<int32_t>> block_rows;
  block_rows.reserve(shapes.size());

  for (size_t i = 0; i < shapes.size(); ++i) {
    const auto& shape = shapes[i];
    CHECK_EQ(shape.heads, ref.heads);
    CHECK_EQ(shape.kv_heads, ref.kv_heads);
    CHECK_EQ(shape.head_dim, ref.head_dim);
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
    block_rows.push_back(make_block_table_host(block_count, total_block_count));
    total_block_count += block_count;
    max_block_count = std::max(max_block_count, block_count);
  }

  auto metadata = make_mtgr_attention_metadata(shapes, device, block_size);
  std::vector<int32_t> block_table_host(
      static_cast<size_t>(shapes.size() * max_block_count), 0);
  std::vector<int64_t> slot_mapping_host;

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
    const auto prefix_slots_host =
        build_slot_mapping_host(block_rows[i], block_size, 0, shape.matched_prefix);
    scatter_prefix_to_cache(full_key_bsnd[i],
                            full_value_bsnd[i],
                            prefix_slots_host,
                            shape.matched_prefix,
                            key_cache,
                            value_cache);
  }

  return MTGRPartialBatchSetup{
      .metadata = std::move(metadata),
      .kv_cache = xllm::KVCache(key_cache, value_cache),
  };
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

torch::Tensor build_mtgr_torch_full_visible_mask(
    const MTGRAttentionTestShape& shape,
    const torch::Device& device) {
  const int64_t history = shape.history;
  const int64_t context = shape.context;
  const int64_t realtime = shape.realtime;
  const int64_t target = shape.target;
  const int64_t matched = shape.matched_prefix;
  const int64_t target_begin = history + context + realtime;
  const int64_t total = target_begin + target;

  CHECK_EQ(total, shape.total_len());
  CHECK_EQ(total - matched, shape.local_len());
  CHECK_GT(history, 0);
  CHECK_GE(context, 0);
  CHECK_GT(realtime, 0);
  CHECK_GT(target, 0);
  CHECK_GE(matched, 0);
  CHECK_LT(matched, target_begin);
  if (matched > 0) {
    CHECK_GE(matched, history + context)
        << "partial precision reference currently covers "
           "partial_real_time_match";
  }

  auto mask =
      torch::zeros({total, total},
                   torch::TensorOptions().dtype(torch::kBool).device(device));
  for (int64_t q_abs = 0; q_abs < total; ++q_abs) {
    auto row = mask.select(0, q_abs);
    if (q_abs < history) {
      row.narrow(0, 0, q_abs + 1).fill_(true);
    } else if (q_abs < history + context) {
      row.narrow(0, 0, history + context).fill_(true);
    } else if (q_abs < target_begin) {
      row.narrow(0, 0, q_abs + 1).fill_(true);
    } else {
      row.narrow(0, 0, target_begin).fill_(true);
      row.narrow(0, q_abs, 1).fill_(true);
    }
  }
  return mask;
}

torch::Tensor run_mtgr_torch_mask_attention_reference(
    const torch::Tensor& full_query_bshd,
    const torch::Tensor& full_key_bshd,
    const torch::Tensor& full_value_bshd,
    const MTGRAttentionTestShape& shape,
    double sm_scale) {
  CHECK_EQ(full_query_bshd.dim(), 4);
  CHECK_EQ(full_key_bshd.dim(), 4);
  CHECK_EQ(full_value_bshd.dim(), 4);
  CHECK_EQ(full_query_bshd.size(0), 1);
  CHECK_EQ(full_key_bshd.size(0), 1);
  CHECK_EQ(full_value_bshd.size(0), 1);
  CHECK_EQ(full_query_bshd.size(1), shape.total_len());
  CHECK_EQ(full_key_bshd.size(1), shape.total_len());
  CHECK_EQ(full_value_bshd.size(1), shape.total_len());
  CHECK_EQ(full_query_bshd.size(2), shape.heads);
  CHECK_EQ(full_key_bshd.size(2), shape.kv_heads);
  CHECK_EQ(full_value_bshd.size(2), shape.kv_heads);
  CHECK_EQ(full_query_bshd.size(3), shape.head_dim);
  CHECK_EQ(full_key_bshd.size(3), shape.head_dim);
  CHECK_EQ(full_value_bshd.size(3), shape.head_dim);
  CHECK_EQ(shape.heads, shape.kv_heads)
      << "torch mask precision reference currently assumes MHA";

  auto query = full_query_bshd.select(0, 0).to(torch::kFloat32).contiguous();
  auto key = full_key_bshd.select(0, 0).to(torch::kFloat32).contiguous();
  auto value = full_value_bshd.select(0, 0).to(torch::kFloat32).contiguous();
  auto visible_mask =
      build_mtgr_torch_full_visible_mask(shape, query.device()).contiguous();

  auto scores = torch::einsum("qhd,khd->qhk", {query, key}) *
                static_cast<float>(sm_scale);
  auto masked_scores =
      scores.masked_fill(visible_mask.logical_not().unsqueeze(1),
                         -std::numeric_limits<float>::infinity());
  auto probs = torch::softmax(masked_scores, /*dim=*/-1);
  auto full_output = torch::einsum("qhk,khd->qhd", {probs, value}).contiguous();
  return full_output.narrow(0, shape.matched_prefix, shape.local_len())
      .contiguous();
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
