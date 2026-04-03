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

#include "mtgr_attention.h"

#include <algorithm>
#include <glog/logging.h>
#include <numeric>
#include <vector>

#include "kernels/npu/genrec_attention.h"

namespace xllm {
namespace layer {

namespace {

std::vector<int64_t> tensor_to_i64_vector(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return {};
  }
  auto cpu_flat = tensor.to(torch::kCPU).to(torch::kLong).view({-1});
  std::vector<int64_t> out;
  out.reserve(cpu_flat.numel());
  for (int64_t i = 0; i < cpu_flat.numel(); ++i) {
    out.push_back(cpu_flat[i].item<int64_t>());
  }
  return out;
}

bool is_cu_seq_lens(const std::vector<int64_t>& values, int64_t total_tokens) {
  if (values.size() < 2 || values.front() != 0 || values.back() != total_tokens) {
    return false;
  }
  for (size_t i = 1; i < values.size(); ++i) {
    if (values[i] < values[i - 1]) {
      return false;
    }
  }
  return true;
}

std::vector<int64_t> normalize_seq_lens(const torch::Tensor& seq_lens,
                                        const torch::Tensor& cu_seq_lens,
                                        int64_t total_tokens) {
  auto values = tensor_to_i64_vector(seq_lens);
  if (values.empty()) {
    values = tensor_to_i64_vector(cu_seq_lens);
  }
  if (values.empty()) {
    return {total_tokens};
  }
  if (is_cu_seq_lens(values, total_tokens)) {
    std::vector<int64_t> out;
    out.reserve(values.size() - 1);
    for (size_t i = 1; i < values.size(); ++i) {
      out.push_back(values[i] - values[i - 1]);
    }
    return out;
  }
  return values;
}

bool valid_seq_lens(const std::vector<int64_t>& seq_lens, int64_t total_tokens) {
  if (seq_lens.empty()) {
    return false;
  }
  int64_t sum = 0;
  for (int64_t len : seq_lens) {
    if (len < 0) {
      return false;
    }
    sum += len;
  }
  return sum == total_tokens;
}

}  // namespace

MTGRAttentionImpl::MTGRAttentionImpl(int64_t num_heads,
                                     int64_t head_size,
                                     float scale,
                                     int64_t num_kv_heads)
    : num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads) {}

torch::Tensor MTGRAttentionImpl::build_bool_attention_mask(
    const AttentionMetadata& attn_metadata,
    int64_t batch_idx,
    int64_t q_start,
    int64_t q_len,
    int64_t kv_start,
    int64_t kv_len,
    const torch::Device& device) {
  if (attn_metadata.attn_mask.defined()) {
    auto mask = attn_metadata.attn_mask;
    if (mask.scalar_type() != torch::kBool) {
      // MTGR may carry additive masks with masked positions encoded as either
      // negative large values (e.g. -9984) or positive 1 for bf16 path.
      // Normalize both conventions to bool mask: true means masked.
      mask = mask.ne(0);
    }
    if (mask.dim() == 2) {
      mask = mask.view({1, 1, mask.size(0), mask.size(1)});
    } else if (mask.dim() == 3) {
      mask = mask.unsqueeze(1);
    } else if (mask.dim() != 4) {
      LOG(FATAL) << "Unsupported attention mask dim: " << mask.dim();
    }
    if (mask.size(0) > 1) {
      CHECK(batch_idx >= 0 && batch_idx < mask.size(0))
          << "batch_idx out of range for attention mask";
      mask = mask.slice(0, batch_idx, batch_idx + 1);
    }

    const int64_t mask_q = mask.size(-2);
    const int64_t mask_kv = mask.size(-1);
    int64_t q_slice_start = 0;
    int64_t kv_slice_start = 0;

    if (mask_q >= q_start + q_len) {
      q_slice_start = q_start;
    } else {
      CHECK(mask_q >= q_len)
          << "Attention mask q dimension is smaller than required length";
    }
    if (mask_kv >= kv_start + kv_len) {
      kv_slice_start = kv_start;
    } else {
      CHECK(mask_kv >= kv_len)
          << "Attention mask kv dimension is smaller than required length";
    }

    mask = mask.slice(-2, q_slice_start, q_slice_start + q_len)
               .slice(-1, kv_slice_start, kv_slice_start + kv_len);
    return mask.to(device).to(torch::kBool).contiguous();
  }

  const int64_t diagonal = kv_len - q_len;
  auto lower = torch::tril(torch::ones({q_len, kv_len},
                                       torch::TensorOptions()
                                           .dtype(torch::kBool)
                                           .device(device)),
                           diagonal);
  auto causal = (~lower).view({1, 1, q_len, kv_len}).contiguous();
  return causal;
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> MTGRAttentionImpl::forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    KVCache& kv_cache) {
  (void)kv_cache;

  torch::Tensor output = torch::empty_like(query);
  std::optional<torch::Tensor> output_lse = std::nullopt;
  if (attn_metadata.is_dummy) {
    return std::make_tuple(output, output_lse);
  }

  const int64_t q_tokens = query.size(0);
  const int64_t kv_tokens = key.size(0);

  auto q_seq_lens = normalize_seq_lens(
      attn_metadata.q_seq_lens, attn_metadata.q_cu_seq_lens, q_tokens);
  auto kv_seq_lens = normalize_seq_lens(
      attn_metadata.kv_seq_lens, attn_metadata.kv_cu_seq_lens, kv_tokens);

  if (!valid_seq_lens(q_seq_lens, q_tokens) ||
      !valid_seq_lens(kv_seq_lens, kv_tokens) ||
      q_seq_lens.size() != kv_seq_lens.size()) {
    LOG(WARNING) << "MTGR attention received incompatible q/kv seq_lens, "
                 << "fallback to single-batch path. q_tokens=" << q_tokens
                 << ", kv_tokens=" << kv_tokens
                 << ", q_seq_lens_size=" << q_seq_lens.size()
                 << ", kv_seq_lens_size=" << kv_seq_lens.size();
    q_seq_lens = {q_tokens};
    kv_seq_lens = {kv_tokens};
  }

  const auto history_lens = tensor_to_i64_vector(attn_metadata.genrec_history_lens);
  const auto context_lens = tensor_to_i64_vector(attn_metadata.genrec_context_lens);
  const auto real_time_lens =
      tensor_to_i64_vector(attn_metadata.genrec_real_time_lens);
  const auto target_lens = tensor_to_i64_vector(attn_metadata.genrec_target_lens);
  auto normalize_to_batch = [&](const std::vector<int64_t>& values,
                                int64_t default_value) {
    std::vector<int64_t> out(q_seq_lens.size(), default_value);
    if (values.empty()) {
      return out;
    }
    if (values.size() == 1) {
      std::fill(out.begin(), out.end(), values[0]);
      return out;
    }
    if (values.size() >= q_seq_lens.size()) {
      for (size_t i = 0; i < q_seq_lens.size(); ++i) {
        out[i] = values[i];
      }
      return out;
    }
    for (size_t i = 0; i < values.size(); ++i) {
      out[i] = values[i];
    }
    return out;
  };
  auto history_lens_batched =
      normalize_to_batch(history_lens, attn_metadata.genrec_history_len);
  auto context_lens_batched =
      normalize_to_batch(context_lens, attn_metadata.genrec_context_len);
  auto real_time_lens_batched =
      normalize_to_batch(real_time_lens, attn_metadata.genrec_real_time_len);
  auto target_lens_batched =
      normalize_to_batch(target_lens, attn_metadata.genrec_target_len);

  int64_t q_offset = 0;
  int64_t kv_offset = 0;
  for (size_t b = 0; b < q_seq_lens.size(); ++b) {
    const int64_t q_len = q_seq_lens[b];
    const int64_t kv_len = kv_seq_lens[b];

    CHECK(q_len >= 0 && kv_len >= 0) << "seq_lens must be non-negative";
    CHECK_LE(q_offset + q_len, q_tokens) << "q slice out of range";
    CHECK_LE(kv_offset + kv_len, kv_tokens) << "kv slice out of range";
    if (q_len == 0) {
      kv_offset += kv_len;
      continue;
    }
    CHECK_GT(kv_len, 0) << "kv_len must be positive when q_len > 0";

    auto query_slice = query.narrow(0, q_offset, q_len);
    auto key_slice = key.narrow(0, kv_offset, kv_len);
    auto value_slice = value.narrow(0, kv_offset, kv_len);

    auto query_bnsd = query_slice.view({q_len, num_heads_, head_size_})
                          .unsqueeze(0)
                          .permute({0, 2, 1, 3})
                          .contiguous();
    auto key_bnsd = key_slice.view({kv_len, num_kv_heads_, head_size_})
                        .unsqueeze(0)
                        .permute({0, 2, 1, 3})
                        .contiguous();
    auto value_bnsd = value_slice.view({kv_len, num_kv_heads_, head_size_})
                          .unsqueeze(0)
                          .permute({0, 2, 1, 3})
                          .contiguous();
    auto output_bnsd = output.narrow(0, q_offset, q_len)
                           .view({q_len, num_heads_, head_size_})
                           .unsqueeze(0)
                           .permute({0, 2, 1, 3});

    auto mask = build_bool_attention_mask(attn_metadata,
                                          static_cast<int64_t>(b),
                                          q_offset,
                                          q_len,
                                          kv_offset,
                                          kv_len,
                                          query.device());

    kernel::npu::GenRecAttentionKernelData kernel_data;
    kernel_data.attention_mask = mask;
    kernel_data.history_len = history_lens_batched[b];
    kernel_data.context_len = context_lens_batched[b];
    kernel_data.real_time_len = real_time_lens_batched[b];
    kernel_data.target_len = target_lens_batched[b];

    kernel::npu::genrec_fused_infer_attention_score_v3(query_bnsd,
                                                        key_bnsd,
                                                        value_bnsd,
                                                        kernel_data,
                                                        num_heads_,
                                                        num_kv_heads_,
                                                        scale_,
                                                        output_bnsd);

    q_offset += q_len;
    kv_offset += kv_len;
  }

  CHECK_EQ(q_offset, q_tokens) << "q offsets do not match total q tokens";
  CHECK_EQ(kv_offset, kv_tokens) << "kv offsets do not match total kv tokens";
  return std::make_tuple(output, output_lse);
}

}  // namespace layer
}  // namespace xllm
