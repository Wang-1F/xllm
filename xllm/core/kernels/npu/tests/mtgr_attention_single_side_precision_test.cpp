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

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "common/global_flags.h"
#include "framework/kv_cache/kv_cache.h"
#include "layers/common/attention_metadata.h"
#include "layers/npu_torch/mtgr_attention.h"

namespace xllm::kernel::npu::test {
namespace {

constexpr double kRtol = 0.12;
constexpr double kAtol = 0.12;
constexpr float kMaxAbs = 1.5e-1f;

struct CaseSpec {
  const char* name = "";
  int64_t history = 0;
  int64_t context = 0;
  int64_t real_time = 0;
  int64_t target = 0;
  int64_t matched_prefix = 0;
};

std::vector<int32_t> make_block_table(int64_t block_count) {
  std::vector<int32_t> table(static_cast<size_t>(block_count));
  for (int64_t i = 0; i < block_count; ++i) {
    table[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  }
  return table;
}

std::vector<int32_t> build_slot_mapping(const std::vector<int32_t>& block_table,
                                        int64_t block_size,
                                        int64_t start_token_idx,
                                        int64_t token_count) {
  std::vector<int32_t> slots;
  slots.reserve(static_cast<size_t>(token_count));
  for (int64_t token_idx = start_token_idx;
       token_idx < start_token_idx + token_count;
       ++token_idx) {
    const int64_t logical_block = token_idx / block_size;
    const int64_t offset = token_idx % block_size;
    const int64_t physical_block =
        block_table.at(static_cast<size_t>(logical_block));
    slots.push_back(static_cast<int32_t>(physical_block * block_size + offset));
  }
  return slots;
}

torch::Tensor reference_attention_bnsd(const torch::Tensor& q_bnsd,
                                       const torch::Tensor& k_bnsd,
                                       const torch::Tensor& v_bnsd,
                                       const torch::Tensor* mask_sq_sk,
                                       double scale) {
  auto qf = q_bnsd;
  auto kf = k_bnsd;
  auto vf = v_bnsd;
  const int64_t num_heads = qf.size(1);
  const int64_t num_kv = kf.size(1);
  if (num_kv < num_heads) {
    const int64_t group = num_heads / num_kv;
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

torch::Tensor build_reference_output_bnsd_cpu_f32(const torch::Tensor& query_bsnd,
                                                   const torch::Tensor& key_bsnd,
                                                   const torch::Tensor& value_bsnd,
                                                   int64_t h,
                                                   int64_t c,
                                                   int64_t r,
                                                   int64_t t) {
  const double scale = 1.0 / std::sqrt(static_cast<double>(query_bsnd.size(3)));
  auto q_seq = query_bsnd.select(0, 0).to(torch::kCPU).to(torch::kFloat32);
  auto k_seq = key_bsnd.select(0, 0).to(torch::kCPU).to(torch::kFloat32);
  auto v_seq = value_bsnd.select(0, 0).to(torch::kCPU).to(torch::kFloat32);

  auto seq_to_bnsd = [](const torch::Tensor& seq_s_n_d) {
    return seq_s_n_d.unsqueeze(0).permute({0, 2, 1, 3}).contiguous();
  };

  auto q = seq_to_bnsd(q_seq);
  auto k = seq_to_bnsd(k_seq);
  auto v = seq_to_bnsd(v_seq);
  const int64_t nheads = q.size(1);
  const int64_t d = q.size(3);

  auto qh = q.slice(2, 0, h);
  auto kh = k.slice(2, 0, h);
  auto vh = v.slice(2, 0, h);
  auto mask_h = torch::triu(torch::ones({h, h}, torch::dtype(torch::kBool)), 1);
  auto history = reference_attention_bnsd(qh, kh, vh, &mask_h, scale);

  auto q_ctx = q.slice(2, h, h + c);
  auto k_hc = k.slice(2, 0, h + c);
  auto v_hc = v.slice(2, 0, h + c);
  auto context = reference_attention_bnsd(q_ctx, k_hc, v_hc, nullptr, scale);

  auto real_time = torch::empty({1, nheads, r, d}, q.options());
  for (int64_t i = 0; i < r; ++i) {
    const int64_t nkv = h + c + i + 1;
    auto qi = q.slice(2, h + c + i, h + c + i + 1);
    auto kk = k.slice(2, 0, nkv);
    auto vv = v.slice(2, 0, nkv);
    real_time.narrow(2, i, 1).copy_(reference_attention_bnsd(qi, kk, vv, nullptr, scale));
  }

  auto target = torch::empty({1, nheads, t, d}, q.options());
  const int64_t prefix = h + c + r;
  for (int64_t j = 0; j < t; ++j) {
    auto qj = q.slice(2, prefix + j, prefix + j + 1);
    auto k_prefix = k.slice(2, 0, prefix);
    auto k_self = k.slice(2, prefix + j, prefix + j + 1);
    auto v_prefix = v.slice(2, 0, prefix);
    auto v_self = v.slice(2, prefix + j, prefix + j + 1);
    auto k_cat = torch::cat({k_prefix, k_self}, 2);
    auto v_cat = torch::cat({v_prefix, v_self}, 2);
    target.narrow(2, j, 1).copy_(reference_attention_bnsd(qj, k_cat, v_cat, nullptr, scale));
  }

  auto ref_bnsd = torch::empty_like(q);
  ref_bnsd.slice(2, 0, h).copy_(history);
  ref_bnsd.slice(2, h, h + c).copy_(context);
  ref_bnsd.slice(2, h + c, h + c + r).copy_(real_time);
  ref_bnsd.slice(2, h + c + r, h + c + r + t).copy_(target);
  return ref_bnsd;
}

void prefill_matched_prefix_cache(const torch::Tensor& full_key_bsnd,
                                  const torch::Tensor& full_value_bsnd,
                                  int64_t matched_prefix,
                                  const std::vector<int32_t>& block_table,
                                  int64_t block_size,
                                  torch::Tensor& key_cache,
                                  torch::Tensor& value_cache) {
  if (matched_prefix <= 0) {
    return;
  }
  auto key_seq = full_key_bsnd.select(0, 0).contiguous();
  auto value_seq = full_value_bsnd.select(0, 0).contiguous();
  auto key_cache_flat =
      key_cache.view({key_cache.size(0) * block_size, key_cache.size(2), key_cache.size(3)});
  auto value_cache_flat = value_cache.view(
      {value_cache.size(0) * block_size, value_cache.size(2), value_cache.size(3)});

  auto prefix_slots = build_slot_mapping(block_table, block_size, 0, matched_prefix);
  auto prefix_slots_dev_i64 = torch::tensor(
      prefix_slots,
      torch::TensorOptions().dtype(torch::kInt64).device(key_cache.device()));
  key_cache_flat.index_copy_(0, prefix_slots_dev_i64, key_seq.slice(0, 0, matched_prefix));
  value_cache_flat.index_copy_(0, prefix_slots_dev_i64, value_seq.slice(0, 0, matched_prefix));
}

void expect_case_close(const char* tag,
                       const torch::Tensor& got_bsnd,
                       const torch::Tensor& expected_bsnd) {
  auto got_bnsd = got_bsnd.permute({0, 2, 1, 3}).to(torch::kCPU).to(torch::kFloat32);
  auto expected_bnsd =
      expected_bsnd.permute({0, 2, 1, 3}).to(torch::kCPU).to(torch::kFloat32);
  auto diff = (got_bnsd - expected_bnsd).abs();
  const double max_abs = diff.max().item<double>();
  const double mean_abs = diff.mean().item<double>();
  const bool allclose = torch::allclose(got_bnsd, expected_bnsd, kRtol, kAtol);

  std::fprintf(stderr,
               "[MTGR][SingleSide][%s] max_abs=%.6e mean_abs=%.6e "
               "allclose(rtol=%.3f,atol=%.3f)=%s\n",
               tag,
               max_abs,
               mean_abs,
               kRtol,
               kAtol,
               allclose ? "PASS" : "FAIL");
  std::fflush(stderr);

  EXPECT_LT(static_cast<float>(max_abs), kMaxAbs);
  EXPECT_TRUE(allclose);
}

class MTGRAttentionSingleSidePrecisionTest : public ::testing::Test {
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

  void run_case(const CaseSpec& spec, bool verify = true) {
    constexpr int64_t kNumHeads = 8;
    constexpr int64_t kNumKvHeads = 8;
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kBlockSize = 128;

    const int64_t h = spec.history;
    const int64_t c = spec.context;
    const int64_t r = spec.real_time;
    const int64_t t = spec.target;
    const int64_t total = h + c + r + t;
    const int64_t matched = spec.matched_prefix;

    ASSERT_GT(h, 0);
    ASSERT_GT(c, 0);
    ASSERT_GT(r, 0);
    ASSERT_GT(t, 0);
    ASSERT_GE(matched, 0);
    ASSERT_LT(matched, h + c + r);

    const int64_t unmatched_start = matched;
    const int64_t unmatched_len = total - unmatched_start;
    ASSERT_GT(unmatched_len, 0);

    FLAGS_block_size = kBlockSize;

    auto q_full = torch::randn({total, kNumHeads * kHeadDim}, fp_opts_);
    auto k_full = torch::randn({total, kNumKvHeads * kHeadDim}, fp_opts_);
    auto v_full = torch::randn({total, kNumKvHeads * kHeadDim}, fp_opts_);

    auto q_case = q_full.narrow(0, unmatched_start, unmatched_len).contiguous();
    auto k_case = k_full.narrow(0, unmatched_start, unmatched_len).contiguous();
    auto v_case = v_full.narrow(0, unmatched_start, unmatched_len).contiguous();

    auto q_full_bsnd = q_full.view({1, total, kNumHeads, kHeadDim}).contiguous();
    auto k_full_bsnd = k_full.view({1, total, kNumKvHeads, kHeadDim}).contiguous();
    auto v_full_bsnd = v_full.view({1, total, kNumKvHeads, kHeadDim}).contiguous();

    const int64_t block_count = (total + kBlockSize - 1) / kBlockSize;
    const auto block_table_host = make_block_table(block_count);
    auto block_table = torch::tensor(
                           block_table_host,
                           torch::TensorOptions().dtype(torch::kInt32).device(device_))
                           .view({1, block_count})
                           .contiguous();
    auto slot_mapping_host =
        build_slot_mapping(block_table_host, kBlockSize, unmatched_start, unmatched_len);
    auto slot_mapping = torch::tensor(
                            slot_mapping_host,
                            torch::TensorOptions().dtype(torch::kInt64).device(device_))
                            .contiguous();

    const int64_t cache_block_count = block_count + 4;
    auto key_cache =
        torch::zeros({cache_block_count, kBlockSize, kNumKvHeads, kHeadDim}, fp_opts_);
    auto value_cache =
        torch::zeros({cache_block_count, kBlockSize, kNumKvHeads, kHeadDim}, fp_opts_);
    prefill_matched_prefix_cache(
        k_full_bsnd, v_full_bsnd, matched, block_table_host, kBlockSize, key_cache, value_cache);

    xllm::KVCache kv_cache(key_cache, value_cache);
    xllm::layer::AttentionMetadata metadata;
    auto len_opts = torch::TensorOptions().dtype(torch::kInt64);
    metadata.is_dummy = false;
    metadata.q_seq_lens = torch::tensor({unmatched_len}, len_opts);
    metadata.kv_seq_lens = torch::tensor({unmatched_len}, len_opts);
    metadata.genrec_history_lens = torch::tensor({h}, len_opts);
    metadata.genrec_context_lens = torch::tensor({c}, len_opts);
    metadata.genrec_real_time_lens = torch::tensor({r}, len_opts);
    metadata.genrec_target_lens = torch::tensor({t}, len_opts);
    metadata.genrec_matched_prefix_lens = torch::tensor({matched}, len_opts);
    metadata.block_table = block_table;
    metadata.slot_mapping = slot_mapping;

    xllm::layer::MTGRAttentionImpl mtgr(
        kNumHeads, kHeadDim, 1.0f / std::sqrt(static_cast<float>(kHeadDim)), kNumKvHeads);
    auto out = std::get<0>(mtgr.forward(metadata, q_case, k_case, v_case, kv_cache));
    auto out_bsnd = out.view({1, unmatched_len, kNumHeads, kHeadDim}).contiguous();

    auto reference_full_bnsd =
        build_reference_output_bnsd_cpu_f32(q_full_bsnd, k_full_bsnd, v_full_bsnd, h, c, r, t);
    auto reference_full_bsnd = reference_full_bnsd.permute({0, 2, 1, 3}).contiguous();
    auto reference_case_bsnd =
        reference_full_bsnd.slice(1, unmatched_start, unmatched_start + unmatched_len)
            .to(device_)
            .to(out.scalar_type())
            .contiguous();

    if (verify) {
      expect_case_close(spec.name, out_bsnd, reference_case_bsnd);
    }
  }

  torch::Device device_{torch::kCPU};
  torch::TensorOptions fp_opts_;
};

}  // namespace

TEST_F(MTGRAttentionSingleSidePrecisionTest, CompareMatchedModesAgainstBase) {
  torch::manual_seed(20260410);

  run_case(CaseSpec{
               .name = "warmup_no_matched",
               .history = 1300,
               .context = 8,
               .real_time = 400,
               .target = 800,
               .matched_prefix = 0,
           },
           /*verify=*/false);

  run_case(CaseSpec{
      .name = "no_matched",
      .history = 1300,
      .context = 8,
      .real_time = 400,
      .target = 800,
      .matched_prefix = 0,
  });

  run_case(CaseSpec{
      .name = "partial_hist_matched",
      .history = 1300,
      .context = 8,
      .real_time = 400,
      .target = 800,
      .matched_prefix = 900,
  });

  run_case(CaseSpec{
      .name = "partial_ctx_matched",
      .history = 1300,
      .context = 8,
      .real_time = 400,
      .target = 800,
      .matched_prefix = 1300 + 3,
  });

  run_case(CaseSpec{
      .name = "partial_rt_matched",
      .history = 1300,
      .context = 8,
      .real_time = 400,
      .target = 800,
      .matched_prefix = 1300 + 8 + 240,
  });
}

}  // namespace xllm::kernel::npu::test
