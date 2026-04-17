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

#include "mtgr_flashinfer.h"

#include <glog/logging.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "core/common/global_flags.h"
#include "core/platform/device.h"
#include "cuda_ops_api.h"
#include "utils.h"

namespace xllm::kernel::cuda {
namespace {

struct FlashinferWorkspaceBuffers {
  torch::Tensor float_workspace;
  torch::Tensor int_workspace;
  torch::Tensor page_locked_int_workspace;
};

FlashinferWorkspaceBuffers& get_workspace_buffers(const torch::Device& device) {
  static thread_local FlashinferWorkspaceBuffers ws;
  const bool need_init = !ws.float_workspace.defined() ||
                         ws.float_workspace.device() != device;
  if (need_init) {
    ws.float_workspace =
        torch::empty({FLAGS_flashinfer_workspace_buffer_size},
                     torch::TensorOptions().dtype(torch::kUInt8).device(device));
    ws.int_workspace =
        torch::empty({8 * 1024 * 1024},
                     torch::TensorOptions().dtype(torch::kUInt8).device(device));
    ws.page_locked_int_workspace = torch::empty(
        {ws.int_workspace.size(0)},
        torch::TensorOptions()
            .dtype(torch::kUInt8)
            .device(torch::kCPU)
            .pinned_memory(true));
  }
  return ws;
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
  ffi::Array<int64_t> plan_info;
};

torch::Tensor make_seq_indptr_host(int64_t seq_len) {
  CHECK_GE(seq_len, 0);
  return torch::tensor({0, static_cast<int32_t>(seq_len)},
                       torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
}

FlashinferPlan build_prefill_plan(const torch::Device& device,
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
                                  int64_t window_size_left) {
  CHECK_EQ(q_cu_seq_lens_host.device().type(), torch::kCPU);
  CHECK_EQ(kv_cu_seq_lens_host.device().type(), torch::kCPU);
  CHECK_EQ(q_cu_seq_lens_host.scalar_type(), torch::kInt32);
  CHECK_EQ(kv_cu_seq_lens_host.scalar_type(), torch::kInt32);
  CHECK_GE(q_cu_seq_lens_host.numel(), 2);
  CHECK_GE(kv_cu_seq_lens_host.numel(), 2);

  auto& ws = get_workspace_buffers(device);
  bind_tvmffi_stream_to_current_torch_stream(device);

  const std::string backend =
      determine_attention_backend(/*pos_encoding_mode=*/0,
                                  /*use_fp16_qk_reduction=*/false,
                                  /*use_custom_mask=*/false);
  FlashinferPlan plan;
  plan.uri = get_batch_prefill_uri(backend,
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
  const int64_t total_num_rows =
      q_cu_seq_lens_host[-1].item<int64_t>();
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

FlashinferPlan build_chunked_prefill_plan(const torch::Device& device,
                                          torch::ScalarType query_dtype,
                                          torch::ScalarType key_dtype,
                                          torch::ScalarType output_dtype,
                                          int64_t head_dim_qk,
                                          int64_t head_dim_vo,
                                          int64_t num_qo_heads,
                                          int64_t num_kv_heads,
                                          int64_t block_size,
                                          int64_t window_size_left,
                                          const torch::Tensor& qo_indptr_host,
                                          const torch::Tensor& paged_kv_indptr_host,
                                          const torch::Tensor& kv_len_arr_host,
                                          bool causal) {
  CHECK_EQ(qo_indptr_host.device().type(), torch::kCPU);
  CHECK_EQ(paged_kv_indptr_host.device().type(), torch::kCPU);
  CHECK_EQ(kv_len_arr_host.device().type(), torch::kCPU);
  CHECK_EQ(qo_indptr_host.scalar_type(), torch::kInt32);
  CHECK_EQ(paged_kv_indptr_host.scalar_type(), torch::kInt32);
  CHECK_EQ(kv_len_arr_host.scalar_type(), torch::kInt32);

  auto& ws = get_workspace_buffers(device);
  bind_tvmffi_stream_to_current_torch_stream(device);

  FlashinferPlan plan;
  // Keep aligned with current project choice for paged prefill path.
  const std::string backend = "fa2";
  plan.uri = get_batch_prefill_uri(backend,
                                   query_dtype,
                                   key_dtype,
                                   output_dtype,
                                   paged_kv_indptr_host.scalar_type(),
                                   head_dim_qk,
                                   head_dim_vo,
                                   /*pos_encoding_mode=*/0,
                                   /*use_sliding_window=*/false,
                                   /*use_logits_soft_cap=*/false,
                                   /*use_fp16_qk_reduction=*/false);

  const int64_t batch_size = paged_kv_indptr_host.size(0) - 1;
  const int64_t total_num_rows = qo_indptr_host[-1].item<int64_t>();

  auto plan_func = get_function(plan.uri, "plan");
  auto plan_result =
      plan_func(to_ffi_tensor(ws.float_workspace),
                to_ffi_tensor(ws.int_workspace),
                to_ffi_tensor(ws.page_locked_int_workspace),
                to_ffi_tensor(qo_indptr_host),
                to_ffi_tensor(paged_kv_indptr_host),
                to_ffi_tensor(kv_len_arr_host),
                causal ? total_num_rows : batch_size,
                batch_size,
                num_qo_heads,
                num_kv_heads,
                block_size,
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

std::vector<int32_t> to_block_table_host(const torch::Tensor& block_table) {
  CHECK(block_table.defined()) << "block_table must be defined";
  torch::Tensor bt = block_table;
  if (bt.dim() == 2) {
    CHECK_GE(bt.size(0), 1);
    bt = bt.select(0, 0);
  }
  CHECK_EQ(bt.dim(), 1) << "block_table must be [1, max_blocks] or [max_blocks]";
  bt = bt.to(torch::kCPU).to(torch::kInt32).contiguous();
  std::vector<int32_t> host(bt.numel());
  auto* ptr = bt.data_ptr<int32_t>();
  std::copy(ptr, ptr + bt.numel(), host.begin());
  return host;
}

std::vector<int32_t> build_slot_mapping(const std::vector<int32_t>& block_table,
                                        int64_t block_size,
                                        int64_t start_token_idx,
                                        int64_t token_count) {
  CHECK_GE(block_size, 1);
  CHECK_GE(start_token_idx, 0);
  CHECK_GE(token_count, 0);
  std::vector<int32_t> slots;
  slots.reserve(static_cast<size_t>(token_count));
  for (int64_t token_idx = start_token_idx;
       token_idx < start_token_idx + token_count;
       ++token_idx) {
    const int64_t logical_block = token_idx / block_size;
    const int64_t offset = token_idx % block_size;
    CHECK_LT(logical_block, static_cast<int64_t>(block_table.size()))
        << "block_table too short for token index " << token_idx;
    const int64_t physical_block =
        block_table[static_cast<size_t>(logical_block)];
    slots.push_back(static_cast<int32_t>(physical_block * block_size + offset));
  }
  return slots;
}

void scatter_to_cache(const torch::Tensor& key_snd,
                      const torch::Tensor& value_snd,
                      torch::Tensor& key_cache,
                      torch::Tensor& value_cache,
                      const torch::Tensor& slot_mapping_i32) {
  if (key_snd.numel() == 0) {
    return;
  }
  CHECK_EQ(key_snd.dim(), 3) << "key_snd must be [tokens, kv_heads, head_dim]";
  CHECK_EQ(value_snd.dim(), 3)
      << "value_snd must be [tokens, kv_heads, head_dim]";
  CHECK_EQ(slot_mapping_i32.scalar_type(), torch::kInt32);
  CHECK_EQ(slot_mapping_i32.dim(), 1);
  CHECK_EQ(slot_mapping_i32.size(0), key_snd.size(0));
  reshape_paged_cache(slot_mapping_i32.contiguous(),
                      key_snd.contiguous(),
                      value_snd.contiguous(),
                      key_cache,
                      value_cache);
}

std::pair<torch::Tensor, torch::Tensor> gather_prefix_from_cache(
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const std::vector<int32_t>& block_table,
    int64_t block_size,
    int64_t total_kv_len) {
  CHECK_GE(total_kv_len, 0);
  CHECK_EQ(key_cache.dim(), 4);
  CHECK_EQ(value_cache.dim(), 4);
  CHECK_EQ(key_cache.sizes(), value_cache.sizes());
  CHECK_EQ(key_cache.size(1), block_size);

  if (total_kv_len == 0) {
    auto empty_k = torch::empty({0, key_cache.size(2), key_cache.size(3)},
                                key_cache.options());
    auto empty_v = torch::empty({0, value_cache.size(2), value_cache.size(3)},
                                value_cache.options());
    return {empty_k, empty_v};
  }

  auto slots_host =
      build_slot_mapping(block_table, block_size, /*start_token_idx=*/0, total_kv_len);
  auto slots = torch::tensor(slots_host,
                             torch::TensorOptions()
                                 .dtype(torch::kInt64)
                                 .device(key_cache.device()))
                   .contiguous();
  auto key_flat =
      key_cache.view({key_cache.size(0) * key_cache.size(1),
                      key_cache.size(2),
                      key_cache.size(3)});
  auto value_flat =
      value_cache.view({value_cache.size(0) * value_cache.size(1),
                        value_cache.size(2),
                        value_cache.size(3)});
  auto gathered_k = key_flat.index_select(0, slots);
  auto gathered_v = value_flat.index_select(0, slots);
  return {gathered_k, gathered_v};
}

torch::Tensor normalize_lse_shape(const torch::Tensor& raw_lse,
                                  int64_t seq_len,
                                  int64_t num_heads) {
  CHECK(raw_lse.defined());
  torch::Tensor lse = raw_lse.contiguous();
  CHECK_EQ(lse.scalar_type(), torch::kFloat32)
      << "output_lse must be float32";

  if (lse.dim() == 4 && lse.size(0) == 1 && lse.size(3) == 1) {
    if (lse.size(1) == seq_len && lse.size(2) == num_heads) {
      return lse.squeeze(0).contiguous();
    }
    if (lse.size(1) == num_heads && lse.size(2) == seq_len) {
      return lse.squeeze(0).transpose(0, 1).contiguous();
    }
  }

  if (lse.dim() == 3) {
    if (lse.size(0) == seq_len && lse.size(1) == num_heads && lse.size(2) == 1) {
      return lse;
    }
    if (lse.size(0) == num_heads && lse.size(1) == seq_len && lse.size(2) == 1) {
      return lse.transpose(0, 1).contiguous();
    }
    if (lse.size(0) == 1 && lse.size(1) == seq_len && lse.size(2) == num_heads) {
      return lse.squeeze(0).unsqueeze(-1).contiguous();
    }
  }

  if (lse.dim() == 2) {
    if (lse.size(0) == seq_len && lse.size(1) == num_heads) {
      return lse.unsqueeze(-1).contiguous();
    }
    if (lse.size(0) == num_heads && lse.size(1) == seq_len) {
      return lse.transpose(0, 1).unsqueeze(-1).contiguous();
    }
  }

  CHECK(false) << "Unsupported output_lse shape. raw shape=" << lse.sizes()
               << ", expected seq_len=" << seq_len
               << ", num_heads=" << num_heads;
  return torch::Tensor();
}

struct AttentionRunResult {
  torch::Tensor out_snd;  // [seq, heads, head_dim]
  torch::Tensor lse_sh1;  // [seq, heads, 1], fp32 (optional)
};

AttentionRunResult run_fa_segment(const torch::Tensor& query_snd,
                                  const torch::Tensor& key_snd,
                                  const torch::Tensor& value_snd,
                                  double sm_scale,
                                  bool causal,
                                  bool need_lse) {
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
  auto q_cu_dev = q_cu_host.to(device, /*non_blocking=*/false, /*copy=*/true);
  auto kv_cu_dev = kv_cu_host.to(device, /*non_blocking=*/false, /*copy=*/true);

  auto plan = build_prefill_plan(device,
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
                                 /*window_size_left=*/-1);

  auto& ws = get_workspace_buffers(device);
  AttentionRunResult result;
  result.out_snd = torch::empty({q_len, num_heads, head_dim}, query_snd.options());
  std::optional<torch::Tensor> output_lse = std::nullopt;
  if (need_lse) {
    output_lse = torch::empty(
        {q_len, num_heads, 1},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
  }

  if (causal) {
    batch_prefill(plan.uri,
                  plan.plan_info,
                  ws.float_workspace,
                  ws.int_workspace,
                  ws.page_locked_int_workspace,
                  query_snd,
                  key_snd,
                  value_snd,
                  q_cu_dev,
                  kv_cu_dev,
                  /*window_left=*/-1,
                  sm_scale,
                  result.out_snd,
                  output_lse);
  } else {
    batch_prefill_non_causal(plan.uri,
                             plan.plan_info,
                             ws.float_workspace,
                             ws.int_workspace,
                             ws.page_locked_int_workspace,
                             query_snd,
                             key_snd,
                             value_snd,
                             q_cu_dev,
                             kv_cu_dev,
                             /*window_left=*/-1,
                             sm_scale,
                             result.out_snd,
                             output_lse);
  }

  if (need_lse) {
    result.lse_sh1 = normalize_lse_shape(output_lse.value(), q_len, num_heads);
  }
  return result;
}

AttentionRunResult run_pa_causal_segment(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const std::vector<int32_t>& block_table_host,
    int64_t block_size,
    int64_t total_kv_len,
    double sm_scale,
    bool need_lse) {
  CHECK_EQ(query_snd.dim(), 3);
  CHECK_EQ(key_cache.dim(), 4);
  CHECK_EQ(value_cache.dim(), 4);
  CHECK_EQ(key_cache.sizes(), value_cache.sizes());
  CHECK_EQ(key_cache.size(1), block_size);
  CHECK_GT(total_kv_len, 0);

  const auto device = query_snd.device();
  const int64_t q_len = query_snd.size(0);
  const int64_t num_heads = query_snd.size(1);
  const int64_t head_dim = query_snd.size(2);
  const int64_t num_kv_heads = key_cache.size(2);
  const int64_t page_cnt = (total_kv_len + block_size - 1) / block_size;
  CHECK_LE(page_cnt, static_cast<int64_t>(block_table_host.size()));

  std::vector<int32_t> paged_indices_host(
      block_table_host.begin(), block_table_host.begin() + page_cnt);
  auto i32_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  auto i32_dev = torch::TensorOptions().dtype(torch::kInt32).device(device);
  auto paged_kv_indices =
      torch::tensor(paged_indices_host, i32_dev).contiguous();
  auto paged_kv_indptr =
      torch::tensor({0, static_cast<int32_t>(page_cnt)}, i32_dev).contiguous();
  const int32_t last_page_len =
      static_cast<int32_t>((total_kv_len - 1) % block_size + 1);
  auto paged_kv_last_page_len =
      torch::tensor({last_page_len}, i32_dev).contiguous();
  auto qo_indptr = torch::tensor({0, static_cast<int32_t>(q_len)}, i32_dev)
                       .contiguous();

  auto plan = build_chunked_prefill_plan(
      device,
      query_snd.scalar_type(),
      key_cache.scalar_type(),
      query_snd.scalar_type(),
      head_dim,
      head_dim,
      num_heads,
      num_kv_heads,
      block_size,
      /*window_size_left=*/-1,
      qo_indptr.to(torch::kCPU),
      paged_kv_indptr.to(torch::kCPU),
      torch::tensor({static_cast<int32_t>(total_kv_len)}, i32_cpu),
      /*causal=*/true);

  auto& ws = get_workspace_buffers(device);
  AttentionRunResult result;
  result.out_snd = torch::empty({q_len, num_heads, head_dim}, query_snd.options());
  std::optional<torch::Tensor> output_lse = std::nullopt;
  if (need_lse) {
    output_lse = torch::empty(
        {q_len, num_heads, 1},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
  }

  std::optional<torch::Tensor> qo_indptr_opt = qo_indptr;
  batch_chunked_prefill(plan.uri,
                        plan.plan_info,
                        ws.float_workspace,
                        ws.int_workspace,
                        ws.page_locked_int_workspace,
                        query_snd,
                        key_cache,
                        value_cache,
                        paged_kv_indptr,
                        paged_kv_indices,
                        paged_kv_last_page_len,
                        /*window_left=*/-1,
                        sm_scale,
                        result.out_snd,
                        output_lse,
                        qo_indptr_opt,
                        /*causal=*/true);
  if (need_lse) {
    result.lse_sh1 = normalize_lse_shape(output_lse.value(), q_len, num_heads);
  }
  return result;
}

struct MergeResult {
  torch::Tensor out_bsnd;  // [1, seq, heads, head_dim]
  torch::Tensor lse_sh1;   // [seq, heads, 1], fp32
};

MergeResult merge_two_attention_results(const torch::Tensor& out_a_bsnd,
                                        const torch::Tensor& lse_a_sh1,
                                        const torch::Tensor& out_b_bsnd,
                                        const torch::Tensor& lse_b_sh1) {
  CHECK_EQ(out_a_bsnd.dim(), 4);
  CHECK_EQ(out_b_bsnd.dim(), 4);
  CHECK_EQ(out_a_bsnd.sizes(), out_b_bsnd.sizes());
  CHECK_EQ(out_a_bsnd.size(0), 1);
  CHECK_EQ(lse_a_sh1.dim(), 3);
  CHECK_EQ(lse_b_sh1.dim(), 3);
  CHECK_EQ(lse_a_sh1.sizes(), lse_b_sh1.sizes());
  CHECK_EQ(lse_a_sh1.size(0), out_a_bsnd.size(1));
  CHECK_EQ(lse_a_sh1.size(1), out_a_bsnd.size(2));
  CHECK_EQ(lse_a_sh1.size(2), 1);
  CHECK_EQ(lse_a_sh1.scalar_type(), torch::kFloat32);
  CHECK_EQ(lse_b_sh1.scalar_type(), torch::kFloat32);

  auto out_a = out_a_bsnd.select(0, 0).to(torch::kFloat32);
  auto out_b = out_b_bsnd.select(0, 0).to(torch::kFloat32);
  auto lse_a = lse_a_sh1;
  auto lse_b = lse_b_sh1;

  auto lse_max = torch::maximum(lse_a, lse_b);
  auto exp_a = torch::exp2(lse_a - lse_max);
  auto exp_b = torch::exp2(lse_b - lse_max);
  auto denom = exp_a + exp_b;
  auto lse_new = lse_max + torch::log2(denom);

  auto merged = (exp_a * out_a + exp_b * out_b) / denom;

  MergeResult result;
  result.out_bsnd = merged.to(out_a_bsnd.scalar_type()).unsqueeze(0);
  result.lse_sh1 = lse_new;
  return result;
}

AttentionRunResult run_target_self_diagonal(const torch::Tensor& target_query_snd,
                                            const torch::Tensor& target_key_snd,
                                            const torch::Tensor& target_value_snd,
                                            double sm_scale) {
  CHECK_EQ(target_query_snd.dim(), 3);
  CHECK_EQ(target_key_snd.dim(), 3);
  CHECK_EQ(target_value_snd.dim(), 3);
  CHECK_EQ(target_query_snd.size(0), target_key_snd.size(0));
  CHECK_EQ(target_query_snd.size(0), target_value_snd.size(0));
  const int64_t t = target_query_snd.size(0);
  CHECK_GT(t, 0);

  std::vector<torch::Tensor> outs;
  std::vector<torch::Tensor> lses;
  outs.reserve(static_cast<size_t>(t));
  lses.reserve(static_cast<size_t>(t));
  for (int64_t j = 0; j < t; ++j) {
    auto qj = target_query_snd.narrow(0, j, 1).contiguous();
    auto kj = target_key_snd.narrow(0, j, 1).contiguous();
    auto vj = target_value_snd.narrow(0, j, 1).contiguous();
    auto self = run_fa_segment(qj, kj, vj, sm_scale, /*causal=*/true, /*need_lse=*/true);
    outs.push_back(self.out_snd);
    lses.push_back(self.lse_sh1);
  }

  AttentionRunResult result;
  result.out_snd = torch::cat(outs, 0).contiguous();
  result.lse_sh1 = torch::cat(lses, 0).contiguous();
  return result;
}

void write_output_slice(torch::Tensor& out_bsnd,
                        int64_t start,
                        const torch::Tensor& snd_slice) {
  CHECK_EQ(out_bsnd.dim(), 4);
  CHECK_EQ(out_bsnd.size(0), 1);
  CHECK_EQ(snd_slice.dim(), 3);
  CHECK_EQ(out_bsnd.size(2), snd_slice.size(1));
  CHECK_EQ(out_bsnd.size(3), snd_slice.size(2));
  out_bsnd.slice(1, start, start + snd_slice.size(0))
      .copy_(snd_slice.unsqueeze(0));
}

void run_no_match_branch(const torch::Tensor& query_bsnd,
                         const torch::Tensor& key_bsnd,
                         const torch::Tensor& value_bsnd,
                         torch::Tensor& key_cache,
                         torch::Tensor& value_cache,
                         const MtgrFlashinferMetadata& meta,
                         double sm_scale,
                         torch::Tensor& output_bsnd) {
  const int64_t h = meta.history_len;
  const int64_t c = meta.context_len;
  const int64_t r = meta.real_time_len;
  const int64_t t = meta.target_len;
  auto query = query_bsnd.select(0, 0).contiguous();
  auto key = key_bsnd.select(0, 0).contiguous();
  auto value = value_bsnd.select(0, 0).contiguous();

  auto hist = run_fa_segment(query.narrow(0, 0, h),
                             key.narrow(0, 0, h),
                             value.narrow(0, 0, h),
                             sm_scale,
                             /*causal=*/true,
                             /*need_lse=*/false);
  write_output_slice(output_bsnd, /*start=*/0, hist.out_snd);

  auto crt = run_fa_segment(query.narrow(0, h, c + r + t),
                            key.narrow(0, 0, h + c),
                            value.narrow(0, 0, h + c),
                            sm_scale,
                            /*causal=*/false,
                            /*need_lse=*/true);
  write_output_slice(output_bsnd, /*start=*/h, crt.out_snd.narrow(0, 0, c));

  auto rt = run_fa_segment(query.narrow(0, h + c, r + t),
                           key.narrow(0, h + c, r),
                           value.narrow(0, h + c, r),
                           sm_scale,
                           /*causal=*/true,
                           /*need_lse=*/true);

  auto target_self = run_target_self_diagonal(query.narrow(0, h + c + r, t),
                                               key.narrow(0, h + c + r, t),
                                               value.narrow(0, h + c + r, t),
                                               sm_scale);

  auto rt_merged = merge_two_attention_results(
      crt.out_snd.narrow(0, c, r).unsqueeze(0),
      crt.lse_sh1.narrow(0, c, r),
      rt.out_snd.narrow(0, 0, r).unsqueeze(0),
      rt.lse_sh1.narrow(0, 0, r));
  write_output_slice(output_bsnd, /*start=*/h + c, rt_merged.out_bsnd.select(0, 0));

  auto tgt_prefix_rt = merge_two_attention_results(
      crt.out_snd.narrow(0, c + r, t).unsqueeze(0),
      crt.lse_sh1.narrow(0, c + r, t),
      rt.out_snd.narrow(0, r, t).unsqueeze(0),
      rt.lse_sh1.narrow(0, r, t));
  auto tgt_merged = merge_two_attention_results(
      tgt_prefix_rt.out_bsnd,
      tgt_prefix_rt.lse_sh1,
      target_self.out_snd.unsqueeze(0),
      target_self.lse_sh1);
  write_output_slice(output_bsnd,
                     /*start=*/h + c + r,
                     tgt_merged.out_bsnd.select(0, 0));

  if (key_cache.defined() && value_cache.defined() && meta.slot_mapping.defined()) {
    const int64_t prefix_cache_len = h + c + r;
    if (prefix_cache_len > 0) {
      CHECK_GE(meta.slot_mapping.size(0), prefix_cache_len)
          << "slot_mapping is shorter than history+context+real_time";
      auto slots = meta.slot_mapping.slice(0, 0, prefix_cache_len)
                       .to(torch::kInt32)
                       .contiguous();
      scatter_to_cache(key.narrow(0, 0, prefix_cache_len),
                       value.narrow(0, 0, prefix_cache_len),
                       key_cache,
                       value_cache,
                       slots);
    }
  }
}

void run_partial_hist_branch(const torch::Tensor& query_bsnd,
                             const torch::Tensor& key_bsnd,
                             const torch::Tensor& value_bsnd,
                             torch::Tensor& key_cache,
                             torch::Tensor& value_cache,
                             const MtgrFlashinferMetadata& meta,
                             const std::vector<int32_t>& block_table_host,
                             double sm_scale,
                             torch::Tensor& output_bsnd) {
  CHECK(key_cache.defined() && value_cache.defined())
      << "partial-history branch requires defined key/value cache";
  const int64_t h = meta.history_len;
  const int64_t c = meta.context_len;
  const int64_t r = meta.real_time_len;
  const int64_t t = meta.target_len;
  const int64_t history_matched = meta.matched_prefix_len;
  const int64_t history_unmatched = h - history_matched;
  const int64_t prefix_unmatched_len = history_unmatched + c;
  const int64_t local_context_start = history_unmatched;
  const int64_t local_rt_start = local_context_start + c;
  const int64_t local_target_start = local_rt_start + r;

  auto query = query_bsnd.select(0, 0).contiguous();
  auto key = key_bsnd.select(0, 0).contiguous();
  auto value = value_bsnd.select(0, 0).contiguous();

  CHECK_GE(meta.slot_mapping.size(0), prefix_unmatched_len);
  auto scatter_slots = meta.slot_mapping.slice(0, 0, prefix_unmatched_len)
                           .to(torch::kInt32)
                           .contiguous();
  scatter_to_cache(key.narrow(0, 0, prefix_unmatched_len),
                   value.narrow(0, 0, prefix_unmatched_len),
                   key_cache,
                   value_cache,
                   scatter_slots);

  auto hist_pa = run_pa_causal_segment(query.narrow(0, 0, history_unmatched),
                                       key_cache,
                                       value_cache,
                                       block_table_host,
                                       meta.block_size,
                                       /*total_kv_len=*/h,
                                       sm_scale,
                                       /*need_lse=*/false);
  write_output_slice(output_bsnd, /*start=*/0, hist_pa.out_snd);

  auto [prefix_k, prefix_v] = gather_prefix_from_cache(
      key_cache, value_cache, block_table_host, meta.block_size, h + c);
  auto crt = run_fa_segment(
      query.narrow(0, local_context_start, c + r + t),
      prefix_k,
      prefix_v,
      sm_scale,
      /*causal=*/false,
      /*need_lse=*/true);
  write_output_slice(output_bsnd,
                     /*start=*/local_context_start,
                     crt.out_snd.narrow(0, 0, c));

  auto rt = run_fa_segment(query.narrow(0, local_rt_start, r + t),
                           key.narrow(0, local_rt_start, r),
                           value.narrow(0, local_rt_start, r),
                           sm_scale,
                           /*causal=*/true,
                           /*need_lse=*/true);

  auto target_self =
      run_target_self_diagonal(query.narrow(0, local_target_start, t),
                               key.narrow(0, local_target_start, t),
                               value.narrow(0, local_target_start, t),
                               sm_scale);

  auto rt_merged = merge_two_attention_results(
      crt.out_snd.narrow(0, c, r).unsqueeze(0),
      crt.lse_sh1.narrow(0, c, r),
      rt.out_snd.narrow(0, 0, r).unsqueeze(0),
      rt.lse_sh1.narrow(0, 0, r));
  write_output_slice(output_bsnd,
                     /*start=*/local_rt_start,
                     rt_merged.out_bsnd.select(0, 0));

  auto tgt_prefix_rt = merge_two_attention_results(
      crt.out_snd.narrow(0, c + r, t).unsqueeze(0),
      crt.lse_sh1.narrow(0, c + r, t),
      rt.out_snd.narrow(0, r, t).unsqueeze(0),
      rt.lse_sh1.narrow(0, r, t));
  auto tgt_merged = merge_two_attention_results(
      tgt_prefix_rt.out_bsnd,
      tgt_prefix_rt.lse_sh1,
      target_self.out_snd.unsqueeze(0),
      target_self.lse_sh1);
  write_output_slice(output_bsnd,
                     /*start=*/local_target_start,
                     tgt_merged.out_bsnd.select(0, 0));
}

void run_partial_context_branch(const torch::Tensor& query_bsnd,
                                const torch::Tensor& key_bsnd,
                                const torch::Tensor& value_bsnd,
                                torch::Tensor& key_cache,
                                torch::Tensor& value_cache,
                                const MtgrFlashinferMetadata& meta,
                                const std::vector<int32_t>& block_table_host,
                                double sm_scale,
                                torch::Tensor& output_bsnd) {
  CHECK(key_cache.defined() && value_cache.defined())
      << "partial-context branch requires defined key/value cache";
  const int64_t h = meta.history_len;
  const int64_t c = meta.context_len;
  const int64_t r = meta.real_time_len;
  const int64_t t = meta.target_len;
  const int64_t context_matched = meta.matched_prefix_len - h;
  const int64_t context_unmatched = c - context_matched;
  const int64_t local_rt_start = context_unmatched;
  const int64_t local_target_start = local_rt_start + r;

  auto query = query_bsnd.select(0, 0).contiguous();
  auto key = key_bsnd.select(0, 0).contiguous();
  auto value = value_bsnd.select(0, 0).contiguous();

  CHECK_GE(meta.slot_mapping.size(0), context_unmatched);
  auto scatter_slots = meta.slot_mapping.slice(0, 0, context_unmatched)
                           .to(torch::kInt32)
                           .contiguous();
  scatter_to_cache(key.narrow(0, 0, context_unmatched),
                   value.narrow(0, 0, context_unmatched),
                   key_cache,
                   value_cache,
                   scatter_slots);

  auto [prefix_k, prefix_v] = gather_prefix_from_cache(
      key_cache, value_cache, block_table_host, meta.block_size, h + c);
  auto crt = run_fa_segment(query,
                            prefix_k,
                            prefix_v,
                            sm_scale,
                            /*causal=*/false,
                            /*need_lse=*/true);
  write_output_slice(output_bsnd, /*start=*/0, crt.out_snd.narrow(0, 0, context_unmatched));

  auto rt = run_fa_segment(query.narrow(0, local_rt_start, r + t),
                           key.narrow(0, local_rt_start, r),
                           value.narrow(0, local_rt_start, r),
                           sm_scale,
                           /*causal=*/true,
                           /*need_lse=*/true);
  auto target_self =
      run_target_self_diagonal(query.narrow(0, local_target_start, t),
                               key.narrow(0, local_target_start, t),
                               value.narrow(0, local_target_start, t),
                               sm_scale);

  auto rt_merged = merge_two_attention_results(
      crt.out_snd.narrow(0, context_unmatched, r).unsqueeze(0),
      crt.lse_sh1.narrow(0, context_unmatched, r),
      rt.out_snd.narrow(0, 0, r).unsqueeze(0),
      rt.lse_sh1.narrow(0, 0, r));
  write_output_slice(output_bsnd,
                     /*start=*/local_rt_start,
                     rt_merged.out_bsnd.select(0, 0));

  auto tgt_prefix_rt = merge_two_attention_results(
      crt.out_snd.narrow(0, context_unmatched + r, t).unsqueeze(0),
      crt.lse_sh1.narrow(0, context_unmatched + r, t),
      rt.out_snd.narrow(0, r, t).unsqueeze(0),
      rt.lse_sh1.narrow(0, r, t));
  auto tgt_merged = merge_two_attention_results(
      tgt_prefix_rt.out_bsnd,
      tgt_prefix_rt.lse_sh1,
      target_self.out_snd.unsqueeze(0),
      target_self.lse_sh1);
  write_output_slice(output_bsnd,
                     /*start=*/local_target_start,
                     tgt_merged.out_bsnd.select(0, 0));
}

void run_partial_realtime_branch(const torch::Tensor& query_bsnd,
                                 const torch::Tensor& key_bsnd,
                                 const torch::Tensor& value_bsnd,
                                 torch::Tensor& key_cache,
                                 torch::Tensor& value_cache,
                                 const MtgrFlashinferMetadata& meta,
                                 const std::vector<int32_t>& block_table_host,
                                 double sm_scale,
                                 torch::Tensor& output_bsnd) {
  CHECK(key_cache.defined() && value_cache.defined())
      << "partial-realtime branch requires defined key/value cache";
  const int64_t h = meta.history_len;
  const int64_t c = meta.context_len;
  const int64_t r = meta.real_time_len;
  const int64_t t = meta.target_len;
  const int64_t realtime_matched = meta.matched_prefix_len - (h + c);
  const int64_t rt_unmatched = r - realtime_matched;
  const int64_t local_target_start = rt_unmatched;

  auto query = query_bsnd.select(0, 0).contiguous();
  auto key = key_bsnd.select(0, 0).contiguous();
  auto value = value_bsnd.select(0, 0).contiguous();

  CHECK_GE(meta.slot_mapping.size(0), rt_unmatched);
  auto scatter_slots = meta.slot_mapping.slice(0, 0, rt_unmatched)
                           .to(torch::kInt32)
                           .contiguous();
  scatter_to_cache(key.narrow(0, 0, rt_unmatched),
                   value.narrow(0, 0, rt_unmatched),
                   key_cache,
                   value_cache,
                   scatter_slots);

  auto rt_pa = run_pa_causal_segment(query.narrow(0, 0, rt_unmatched),
                                     key_cache,
                                     value_cache,
                                     block_table_host,
                                     meta.block_size,
                                     /*total_kv_len=*/h + c + r,
                                     sm_scale,
                                     /*need_lse=*/false);
  write_output_slice(output_bsnd, /*start=*/0, rt_pa.out_snd);

  auto [prefix_k, prefix_v] = gather_prefix_from_cache(
      key_cache, value_cache, block_table_host, meta.block_size, h + c + r);
  auto target_prefix = run_fa_segment(query.narrow(0, local_target_start, t),
                                      prefix_k,
                                      prefix_v,
                                      sm_scale,
                                      /*causal=*/false,
                                      /*need_lse=*/true);
  auto target_self =
      run_target_self_diagonal(query.narrow(0, local_target_start, t),
                               key.narrow(0, local_target_start, t),
                               value.narrow(0, local_target_start, t),
                               sm_scale);
  auto tgt_merged = merge_two_attention_results(
      target_prefix.out_snd.unsqueeze(0),
      target_prefix.lse_sh1,
      target_self.out_snd.unsqueeze(0),
      target_self.lse_sh1);
  write_output_slice(output_bsnd,
                     /*start=*/local_target_start,
                     tgt_merged.out_bsnd.select(0, 0));
}

}  // namespace

void mtgr_flashinfer_attention_forward(const torch::Tensor& query,
                                       const torch::Tensor& key,
                                       const torch::Tensor& value,
                                       torch::Tensor& key_cache,
                                       torch::Tensor& value_cache,
                                       const MtgrFlashinferMetadata& metadata,
                                       double sm_scale,
                                       torch::Tensor& output) {
  CHECK_EQ(query.dim(), 4) << "query must be [1, S, N, D]";
  CHECK_EQ(key.dim(), 4) << "key must be [1, S, KvN, D]";
  CHECK_EQ(value.dim(), 4) << "value must be [1, S, KvN, D]";
  CHECK_EQ(query.size(0), 1);
  CHECK_EQ(key.size(0), 1);
  CHECK_EQ(value.size(0), 1);
  CHECK_EQ(key.sizes(), value.sizes());
  CHECK_EQ(query.device(), key.device());
  CHECK_EQ(query.device(), value.device());
  CHECK_GT(sm_scale, 0.0);

  const int64_t h = metadata.history_len;
  const int64_t c = metadata.context_len;
  const int64_t r = metadata.real_time_len;
  const int64_t t = metadata.target_len;
  const int64_t matched = metadata.matched_prefix_len;
  CHECK_GT(h, 0);
  CHECK_GT(r, 0);
  CHECK_GT(t, 0);
  CHECK_GE(c, 0);
  CHECK_GE(matched, 0);
  CHECK_LT(matched, h + c + r)
      << "matched prefix must be in [0, history+context+real_time)";

  output = torch::empty_like(query);
  auto block_table_host = to_block_table_host(metadata.block_table);

  if (matched == 0) {
    CHECK_EQ(query.size(1), h + c + r + t)
        << "no-match expects local layout [h|c|r|t]";
    run_no_match_branch(query,
                        key,
                        value,
                        key_cache,
                        value_cache,
                        metadata,
                        sm_scale,
                        output);
    return;
  }

  CHECK(key_cache.defined() && value_cache.defined())
      << "matched-prefix branch requires defined key/value cache";

  if (matched < h) {
    const int64_t history_unmatched = h - matched;
    CHECK_EQ(query.size(1), history_unmatched + c + r + t)
        << "partial-history expects local layout [h_unmatched|c|r|t]";
    run_partial_hist_branch(query,
                            key,
                            value,
                            key_cache,
                            value_cache,
                            metadata,
                            block_table_host,
                            sm_scale,
                            output);
    return;
  }

  if (matched < h + c) {
    const int64_t context_unmatched = h + c - matched;
    CHECK_EQ(query.size(1), context_unmatched + r + t)
        << "partial-context expects local layout [c_unmatched|r|t]";
    run_partial_context_branch(query,
                               key,
                               value,
                               key_cache,
                               value_cache,
                               metadata,
                               block_table_host,
                               sm_scale,
                               output);
    return;
  }

  const int64_t rt_unmatched = h + c + r - matched;
  CHECK_EQ(query.size(1), rt_unmatched + t)
      << "partial-realtime expects local layout [rt_unmatched|t]";
  run_partial_realtime_branch(query,
                              key,
                              value,
                              key_cache,
                              value_cache,
                              metadata,
                              block_table_host,
                              sm_scale,
                              output);
}

}  // namespace xllm::kernel::cuda
