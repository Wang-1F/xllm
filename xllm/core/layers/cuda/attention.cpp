/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "attention.h"

#include "common/nvtx_helper.h"
#include "flashinfer_workspace.h"
#include "kernels/cuda/cuda_ops_api.h"
#include "kernels/ops_api.h"

DECLARE_bool(enable_chunked_prefill);
DECLARE_bool(enable_fa_decode);
DECLARE_bool(enable_improved_fa);

namespace {

void lse_combine(torch::Tensor shared_o,
                 torch::Tensor shared_lse,
                 torch::Tensor unshared_o,
                 torch::Tensor unshared_lse,
                 torch::Tensor output) {
  xllm::kernel::cuda::lse_combine(
      output, shared_o, shared_lse, unshared_o, unshared_lse);
}
}  // namespace

namespace xllm {
namespace layer {
AttentionImpl::AttentionImpl(int num_heads,
                             int head_size,
                             float scale,
                             int num_kv_heads,
                             int sliding_window)
    : num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads),
      sliding_window_(sliding_window - 1) {
  rec_kernel_ = std::make_unique<kernel::cuda::triton::RecTorchKernel>();
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> AttentionImpl::forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    KVCache& kv_cache) {
  LLM_NVTX_RANGE("AttentionImpl_forward");

  // LOG(INFO) << "inner AttentionImpl::forward.";
  auto output = torch::empty_like(query);
  auto output_lse = std::nullopt;
  if (attn_metadata.max_seq_len == 0) {
    output = output.view({-1, num_heads_ * head_size_});
    return std::make_tuple(output, output_lse);
  }
  // LOG(INFO) << "before kv view.";
  // LOG(INFO) << "key.is_contiguous(): " << key.is_contiguous();
  // LOG(INFO) << "value.is_contiguous(): " << value.is_contiguous();
  {
    LLM_NVTX_RANGE_COLOR("attention_reshape_inputs", 0xFF808080);  // Gray
    query = query.view({-1, num_heads_, head_size_});
    key = key.view({-1, num_kv_heads_, head_size_});
    value = value.view({-1, num_kv_heads_, head_size_});
    output = output.view({-1, num_heads_, head_size_});
  }

  torch::Tensor k_cache = kv_cache.get_k_cache();
  torch::Tensor v_cache = kv_cache.get_v_cache();

  if (FLAGS_max_decode_rounds == 0) {
    {
      LLM_NVTX_RANGE_COLOR("reshape_paged_cache", 0xFF800080);  // Purple
      xllm::kernel::ReshapePagedCacheParams reshape_paged_cache_params;
      reshape_paged_cache_params.key = key;
      reshape_paged_cache_params.value = value;
      reshape_paged_cache_params.k_cache = k_cache;
      reshape_paged_cache_params.v_cache = v_cache;
      reshape_paged_cache_params.slot_mapping = attn_metadata.slot_mapping;
      xllm::kernel::reshape_paged_cache(reshape_paged_cache_params);
    }
  }

  if (attn_metadata.is_prefill) {
    LLM_NVTX_RANGE("attention_prefill");

    CHECK(!attn_metadata.is_chunked_prefill)
        << "chunked prefill is not supported";
    if (FLAGS_max_decode_rounds > 0) {
      {
        LLM_NVTX_RANGE_COLOR("prefill_reshape_and_cache", 0xFF008080);  // Teal
        rec_kernel_->prefill_reshape_and_cache(key,
                                               value,
                                               attn_metadata.shared_k_cache,
                                               attn_metadata.shared_v_cache);
      }
    }

    {
      LLM_NVTX_RANGE_COLOR("batch_prefill", 0xFF00FF00);  // Green
      xllm::kernel::AttentionParams attention_params;
      attention_params.query = query;
      attention_params.output = output;
      attention_params.output_lse = output_lse;
      // attention_params.max_seq_len = attn_metadata.max_seq_len;
      attention_params.window_size_left = sliding_window_;
      attention_params.scale = scale_;
      attention_params.compute_dtype = attn_metadata.compute_dtype;
      // for flashinfer
      attention_params.float_workspace_buffer =
          FlashinferWorkspace::get_instance().get_float_workspace_buffer();
      attention_params.int_workspace_buffer =
          FlashinferWorkspace::get_instance().get_int_workspace_buffer();
      attention_params.page_locked_int_workspace_buffer =
          FlashinferWorkspace::get_instance()
              .get_page_locked_int_workspace_buffer();
      attention_params.kv_cu_seq_lens = attn_metadata.kv_cu_seq_lens;
      attention_params.q_cu_seq_lens = attn_metadata.q_cu_seq_lens;
      // LOG(INFO) << "attn_metadata.is_prefill: " << attn_metadata.is_prefill;
      // TODO: support chunked prefill
      attention_params.plan_info = attn_metadata.prefill_plan_info;
      attention_params.key = key;
      attention_params.value = value;
      // LOG(INFO) << "key.shape: " << key.sizes();
      // LOG(INFO) << "value.shape: " << value.sizes();
      xllm::kernel::batch_prefill(attention_params);
    }
  } else {
    LLM_NVTX_RANGE("attention_decode");

    if (FLAGS_max_decode_rounds > 0) {
      auto float_options =
          torch::TensorOptions().dtype(torch::kFloat32).device(query.device());
      // LOG(INFO) << "attention_decode_with_shared";
      LLM_NVTX_RANGE("attention_decode_with_shared");

      uint32_t batch_size = attn_metadata.kv_cu_seq_lens.size(0) - 1;
      uint32_t total_beam = query.size(0);
      uint32_t beam_size = total_beam / batch_size;

      // [max_shared_kv_len, num_kv_heads_, head_size_]
      torch::Tensor shared_k_cache = attn_metadata.shared_k_cache;
      torch::Tensor shared_v_cache = attn_metadata.shared_v_cache;
      uint32_t max_shared_kv_len = shared_k_cache.size(0);
      shared_k_cache = shared_k_cache.unsqueeze(1);
      shared_v_cache = shared_v_cache.unsqueeze(1);
      // LOG(INFO) << "shared_k_cache.shape: " << shared_k_cache.sizes();
      // LOG(INFO) << "shared_v_cache.shape: " << shared_v_cache.sizes();
      // [batch_size * beam_size * max_decode_step, num_kv_heads_, head_size_]
      key = key.view({batch_size, beam_size, num_kv_heads_, head_size_})
                .contiguous();
      value = value.view({batch_size, beam_size, num_kv_heads_, head_size_})
                  .contiguous();

      // `step` is a plain uint32_t (not a Tensor), so just print its value.
      xllm::kernel::cuda::decoder_reshape_and_cache(key,
                                                    value,
                                                    k_cache,
                                                    v_cache,
                                                    attn_metadata.block_table,
                                                    attn_metadata.step);
      torch::Tensor unshared_k_cache =
          k_cache.view({-1, 1, num_kv_heads_, head_size_});
      torch::Tensor unshared_v_cache =
          v_cache.view({-1, 1, num_kv_heads_, head_size_});

      torch::Tensor shared_o = torch::zeros_like(query);
      torch::Tensor unshared_o = torch::zeros_like(query);

      LOG(INFO) << "shared_o.shape: " << shared_o.sizes();
      LOG(INFO) << "unshared_o.shape: " << unshared_o.sizes();
      // LOG(INFO) << "shared_o.shape: " << shared_o.sizes();
      // LOG(INFO) << "unshared_o.shape: " << unshared_o.sizes();
      torch::Tensor shared_lse =
          torch::zeros({query.size(0), query.size(1), 1}, float_options);
      torch::Tensor unshared_lse =
          torch::zeros({query.size(0), query.size(1), 1}, float_options);
      // LOG(INFO) << "shared_lse.shape: " << shared_lse.sizes();
      // LOG(INFO) << "unshared_lse.shape: " << unshared_lse.sizes();
      {
        if (FLAGS_enable_improved_fa) {
          LOG(INFO) << "enable_improved_fa";
          // batch_prefill 路径（优化版本，移除 expand 操作）
          LLM_NVTX_RANGE_COLOR("batch_prefill_shared_improved",
                               0xFFFF0000);  // Red

          // 1. reshape query: [batch_size, beam_size, num_heads, head_dim] ->
          // [batch_size * beam_size, num_heads, head_dim]
          torch::Tensor query_reshaped =
              query.view({batch_size * beam_size, num_heads_, head_size_})
                  .contiguous();
          LOG(INFO) << "query_reshaped.shape: " << query_reshaped.sizes();
          // 2. 计算每个 batch 的真实 shared_kv_len
          auto int_options = torch::TensorOptions()
                                 .dtype(torch::kInt32)
                                 .device(query.device());
          torch::Tensor batch_shared_kv_lens =
              torch::diff(attn_metadata.kv_cu_seq_lens).to(int_options);
          LOG(INFO) << "batch_shared_kv_lens: " << batch_shared_kv_lens;
          // 3. 为每个 batch slice 真实的 shared_kv，直接 concat（不 expand）
          // 注意：shared_k_cache 的 shape 是 [max_shared_kv_len, num_kv_heads_,
          // head_size_] 这是所有 batch 共享的同一个 cache，每个 batch
          // 的真实长度是 batch_shared_kv_lens[batch_idx]
          std::vector<torch::Tensor> shared_k_cache_list;
          std::vector<torch::Tensor> shared_v_cache_list;
          std::vector<int32_t> kv_cu_seq_lens_values;
          kv_cu_seq_lens_values.push_back(0);
          int32_t current_offset = 0;

          for (int32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            int32_t real_shared_kv_len =
                batch_shared_kv_lens[batch_idx].item<int32_t>();
            // slice 到真实长度（不 expand）
            torch::Tensor shared_k_sliced =
                attn_metadata.shared_k_cache.slice(0, 0, real_shared_kv_len);
            torch::Tensor shared_v_sliced =
                attn_metadata.shared_v_cache.slice(0, 0, real_shared_kv_len);
            shared_k_cache_list.push_back(shared_k_sliced);
            shared_v_cache_list.push_back(shared_v_sliced);

            // 累积 kv_cu_seq_lens（每个 batch 只记录一次）
            current_offset += real_shared_kv_len;
            kv_cu_seq_lens_values.push_back(current_offset);
          }

          // concat 所有 batch 的结果
          torch::Tensor shared_k_cache_packed =
              torch::cat(shared_k_cache_list, 0);
          torch::Tensor shared_v_cache_packed =
              torch::cat(shared_v_cache_list, 0);
          LOG(INFO) << "shared_k_cache_packed.shape: "
                    << shared_k_cache_packed.sizes();
          LOG(INFO) << "shared_v_cache_packed.shape: "
                    << shared_v_cache_packed.sizes();
          // 4. 生成 q_cu_seq_lens: [0, beam_size, 2*beam_size, ..., batch_size
          // * beam_size]
          torch::Tensor q_cu_seq_lens =
              (torch::arange(batch_size + 1, int_options) * beam_size)
                  .contiguous();
          LOG(INFO) << "q_cu_seq_lens: " << q_cu_seq_lens;
          // 5. 生成 kv_cu_seq_lens: [0, shared_kv_len_0, shared_kv_len_0 +
          // shared_kv_len_1, ...]
          torch::Tensor kv_cu_seq_lens =
              torch::tensor(kv_cu_seq_lens_values, int_options).contiguous();
          LOG(INFO) << "kv_cu_seq_lens: " << kv_cu_seq_lens;
          // 6. 配置 AttentionParams
          xllm::kernel::AttentionParams shared_attention_params;
          shared_attention_params.query = query_reshaped;
          LOG(INFO) << "query_reshaped.shape: " << query_reshaped.sizes();
          shared_attention_params.key = shared_k_cache_packed;
          LOG(INFO) << "shared_k_cache_packed.shape: "
                    << shared_k_cache_packed.sizes();
          shared_attention_params.value = shared_v_cache_packed;
          LOG(INFO) << "shared_v_cache_packed.shape: "
                    << shared_v_cache_packed.sizes();
          shared_attention_params.q_cu_seq_lens = q_cu_seq_lens;
          LOG(INFO) << "q_cu_seq_lens.shape: " << q_cu_seq_lens.sizes();
          shared_attention_params.kv_cu_seq_lens = kv_cu_seq_lens;
          LOG(INFO) << "kv_cu_seq_lens.shape: " << kv_cu_seq_lens.sizes();
          shared_attention_params.output =
              shared_o.view({batch_size * beam_size, num_heads_, head_size_})
                  .contiguous();
          LOG(INFO) << "shared_attention_params.output.shape: "
                    << shared_attention_params.output.sizes();
          shared_attention_params.output_lse =
              shared_lse.view({batch_size * beam_size, num_heads_, 1})
                  .contiguous();
          LOG(INFO) << "shared_attention_params.output_lse.shape: "
                    << shared_attention_params.output_lse.value().sizes();
          shared_attention_params.return_lse = true;
          shared_attention_params.window_size_left = sliding_window_;
          LOG(INFO) << "shared_attention_params.window_size_left: "
                    << sliding_window_;
          shared_attention_params.scale = scale_;
          LOG(INFO) << "shared_attention_params.scale: " << scale_;
          shared_attention_params.compute_dtype = attn_metadata.compute_dtype;
          LOG(INFO) << "shared_attention_params.compute_dtype: "
                    << attn_metadata.compute_dtype;
          // for flashinfer
          shared_attention_params.float_workspace_buffer =
              FlashinferWorkspace::get_instance().get_float_workspace_buffer();
          LOG(INFO) << "shared_attention_params.float_workspace_buffer.shape: "
                    << FlashinferWorkspace::get_instance()
                           .get_float_workspace_buffer()
                           .sizes();
          shared_attention_params.int_workspace_buffer =
              FlashinferWorkspace::get_instance().get_int_workspace_buffer();
          LOG(INFO) << "shared_attention_params.int_workspace_buffer.shape: "
                    << FlashinferWorkspace::get_instance()
                           .get_int_workspace_buffer()
                           .sizes();
          shared_attention_params.page_locked_int_workspace_buffer =
              FlashinferWorkspace::get_instance()
                  .get_page_locked_int_workspace_buffer();
          LOG(INFO) << "shared_attention_params.page_locked_int_workspace_"
                       "buffer.shape: "
                    << FlashinferWorkspace::get_instance()
                           .get_page_locked_int_workspace_buffer()
                           .sizes();
          // 7. 调用 batch_prefill
          xllm::kernel::batch_prefill(shared_attention_params);
          // LOG(INFO) << "improved_fa_decode shared_o: " << shared_o;
          // LOG(INFO) << "improved_fa_decode shared_lse: " << shared_lse;
        } else if (FLAGS_enable_fa_decode) {
          LOG(INFO) << "enable_fa_decode";
          // batch_prefill 路径
          LLM_NVTX_RANGE_COLOR("batch_prefill_shared", 0xFFFF0000);  // Red

          // 1. reshape query: [batch_size, beam_size, num_heads, head_dim] ->
          // [batch_size * beam_size, num_heads, head_dim]
          torch::Tensor query_reshaped =
              query.view({batch_size * beam_size, num_heads_, head_size_})
                  .contiguous();

          // 2. 计算每个 batch 的真实 shared_kv_len
          auto int_options = torch::TensorOptions()
                                 .dtype(torch::kInt32)
                                 .device(query.device());
          torch::Tensor batch_shared_kv_lens =
              torch::diff(attn_metadata.kv_cu_seq_lens).to(int_options);

          // 3. 为每个 batch slice 真实的 shared_kv，然后 expand 到 beam_size
          // 并 concat 为 packed 格式
          // 注意：shared_k_cache 的 shape 是 [max_shared_kv_len, num_kv_heads_,
          // head_size_] 这是所有 batch 共享的同一个 cache，每个 batch
          // 的真实长度是 batch_shared_kv_lens[batch_idx]
          std::vector<torch::Tensor> shared_k_cache_list;
          std::vector<torch::Tensor> shared_v_cache_list;
          std::vector<int32_t> kv_cu_seq_lens_values;
          kv_cu_seq_lens_values.push_back(0);
          int32_t current_offset = 0;

          for (int32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            int32_t real_shared_kv_len =
                batch_shared_kv_lens[batch_idx].item<int32_t>();
            // slice 真实的 shared_kv: 从 shared_k_cache 中 slice 前
            // real_shared_kv_len 个元素 因为 shared_k_cache 是所有 batch
            // 共享的，所以直接 slice 到真实长度
            torch::Tensor shared_k_sliced =
                attn_metadata.shared_k_cache.slice(0, 0, real_shared_kv_len);
            torch::Tensor shared_v_sliced =
                attn_metadata.shared_v_cache.slice(0, 0, real_shared_kv_len);
            // expand 到 beam_size: [real_shared_kv_len, num_kv_heads_,
            // head_size_]
            // -> [beam_size, real_shared_kv_len, num_kv_heads_, head_size_]
            shared_k_sliced = shared_k_sliced.unsqueeze(0).expand(
                {beam_size, real_shared_kv_len, num_kv_heads_, head_size_});
            shared_v_sliced = shared_v_sliced.unsqueeze(0).expand(
                {beam_size, real_shared_kv_len, num_kv_heads_, head_size_});
            // reshape 为 packed 格式: [beam_size * real_shared_kv_len,
            // num_kv_heads_, head_size_]
            shared_k_sliced = shared_k_sliced.contiguous().view(
                {beam_size * real_shared_kv_len, num_kv_heads_, head_size_});
            shared_v_sliced = shared_v_sliced.contiguous().view(
                {beam_size * real_shared_kv_len, num_kv_heads_, head_size_});
            shared_k_cache_list.push_back(shared_k_sliced);
            shared_v_cache_list.push_back(shared_v_sliced);

            // 为每个 beam 生成 kv_cu_seq_lens
            for (int32_t beam_idx = 0; beam_idx < beam_size; ++beam_idx) {
              current_offset += real_shared_kv_len;
              kv_cu_seq_lens_values.push_back(current_offset);
            }
          }

          // concat 所有 batch 的结果
          torch::Tensor shared_k_cache_expanded =
              torch::cat(shared_k_cache_list, 0);
          torch::Tensor shared_v_cache_expanded =
              torch::cat(shared_v_cache_list, 0);

          // 4. 生成 q_cu_seq_lens: [0, 1, 2, 3, ..., batch_size * beam_size]
          torch::Tensor q_cu_seq_lens =
              torch::arange(batch_size * beam_size + 1, int_options)
                  .contiguous();

          // 5. 生成 kv_cu_seq_lens: 使用真实的 shared_kv_len
          torch::Tensor kv_cu_seq_lens =
              torch::tensor(kv_cu_seq_lens_values, int_options).contiguous();

          // 5. 配置 AttentionParams
          xllm::kernel::AttentionParams shared_attention_params;
          shared_attention_params.query = query_reshaped;
          shared_attention_params.key = shared_k_cache_expanded;
          shared_attention_params.value = shared_v_cache_expanded;
          shared_attention_params.q_cu_seq_lens = q_cu_seq_lens;
          shared_attention_params.kv_cu_seq_lens = kv_cu_seq_lens;
          shared_attention_params.output =
              shared_o.view({batch_size * beam_size, num_heads_, head_size_})
                  .contiguous();
          shared_attention_params.output_lse =
              shared_lse.view({batch_size * beam_size, num_heads_, 1})
                  .contiguous();
          shared_attention_params.return_lse = true;
          shared_attention_params.window_size_left = sliding_window_;
          shared_attention_params.scale = scale_;
          shared_attention_params.compute_dtype = attn_metadata.compute_dtype;
          // for flashinfer
          shared_attention_params.float_workspace_buffer =
              FlashinferWorkspace::get_instance().get_float_workspace_buffer();
          shared_attention_params.int_workspace_buffer =
              FlashinferWorkspace::get_instance().get_int_workspace_buffer();
          shared_attention_params.page_locked_int_workspace_buffer =
              FlashinferWorkspace::get_instance()
                  .get_page_locked_int_workspace_buffer();

          // 6. 调用 batch_prefill
          xllm::kernel::batch_prefill(shared_attention_params);
          // LOG(INFO) << "fa_decode shared_o: " << shared_o;
          // LOG(INFO) << "fa_decode shared_lse: " << shared_lse;
        } else {
          LOG(INFO) << "enable_pa_decode";
          // 原有的 batch_decode 路径
          LLM_NVTX_RANGE_COLOR("batch_decode_shared", 0xFFFF0000);  // Red
          xllm::kernel::AttentionParams shared_attention_params;

          shared_attention_params.return_lse = true;
          shared_attention_params.output_lse = shared_lse;

          shared_attention_params.window_size_left = sliding_window_;
          shared_attention_params.scale = scale_;
          shared_attention_params.compute_dtype = attn_metadata.compute_dtype;
          // for flashinfer
          shared_attention_params.float_workspace_buffer =
              FlashinferWorkspace::get_instance().get_float_workspace_buffer();
          shared_attention_params.int_workspace_buffer =
              FlashinferWorkspace::get_instance().get_int_workspace_buffer();
          shared_attention_params.page_locked_int_workspace_buffer =
              FlashinferWorkspace::get_instance()
                  .get_page_locked_int_workspace_buffer();

          // TODO: support chunked prefill
          CHECK(!attn_metadata.is_chunked_prefill)
              << "chunked prefill is not supported";
          // LOG(INFO) << "query.shape: " << query.sizes();
          // LOG(INFO) << "shared_o.shape: " << shared_o.sizes();
          shared_attention_params.query = query;
          shared_attention_params.output = shared_o;
          // LOG(INFO) << "shared_k_cache.shape: " << shared_k_cache.sizes();
          // LOG(INFO) << "shared_v_cache.shape: " << shared_v_cache.sizes();
          shared_attention_params.k_cache = shared_k_cache;
          shared_attention_params.v_cache = shared_v_cache;
          // LOG(INFO) << "attn_metadata.shared_paged_kv_indices: " <<
          // attn_metadata.shared_paged_kv_indices; LOG(INFO) <<
          // "attn_metadata.shared_paged_kv_indptr: " <<
          // attn_metadata.shared_paged_kv_indptr; LOG(INFO) <<
          // "attn_metadata.shared_paged_kv_last_page_len: " <<
          // attn_metadata.shared_paged_kv_last_page_len;
          shared_attention_params.paged_kv_indices =
              attn_metadata.shared_paged_kv_indices;
          shared_attention_params.paged_kv_indptr =
              attn_metadata.shared_paged_kv_indptr;
          shared_attention_params.paged_kv_last_page_len =
              attn_metadata.shared_paged_kv_last_page_len;

          // shared_attention_params.plan_info = attn_metadata.plan_info;

          xllm::kernel::batch_decode(shared_attention_params);
          // LOG(INFO) << "pa standard shared_o: " << shared_o;
          // LOG(INFO) << "pa standard shared_lse: " << shared_lse;
          // LOG(FATAL) << "after batch_decode.";
        }
      }

      // LOG(INFO) << "after shared batch_decode.";

      {
        LLM_NVTX_RANGE_COLOR("batch_decode_unshared", 0xFFFF0000);  // Red
        xllm::kernel::AttentionParams unshared_attention_params;

        unshared_attention_params.output_lse = unshared_lse;
        unshared_attention_params.return_lse = true;

        unshared_attention_params.window_size_left = sliding_window_;
        unshared_attention_params.scale = scale_;
        unshared_attention_params.compute_dtype = attn_metadata.compute_dtype;
        // for flashinfer
        unshared_attention_params.float_workspace_buffer =
            FlashinferWorkspace::get_instance().get_float_workspace_buffer();
        unshared_attention_params.int_workspace_buffer =
            FlashinferWorkspace::get_instance().get_int_workspace_buffer();
        unshared_attention_params.page_locked_int_workspace_buffer =
            FlashinferWorkspace::get_instance()
                .get_page_locked_int_workspace_buffer();

        // TODO: support chunked prefill
        CHECK(!attn_metadata.is_chunked_prefill)
            << "chunked prefill is not supported";

        unshared_attention_params.query = query;
        unshared_attention_params.output = unshared_o;
        // LOG(INFO) << "unshared_k_cache.shape: " << unshared_k_cache.sizes();
        // LOG(INFO) << "unshared_v_cache.shape: " << unshared_v_cache.sizes();
        unshared_attention_params.k_cache = unshared_k_cache;
        unshared_attention_params.v_cache = unshared_v_cache;
        // LOG(INFO) << "attn_metadata.unshared_paged_kv_indices: " <<
        // attn_metadata.unshared_paged_kv_indices; LOG(INFO) <<
        // "attn_metadata.unshared_paged_kv_indptr: " <<
        // attn_metadata.unshared_paged_kv_indptr; LOG(INFO) <<
        // "attn_metadata.unshared_paged_kv_last_page_len: " <<
        // attn_metadata.unshared_paged_kv_last_page_len;
        unshared_attention_params.paged_kv_indices =
            attn_metadata.unshared_paged_kv_indices;
        unshared_attention_params.paged_kv_indptr =
            attn_metadata.unshared_paged_kv_indptr;
        unshared_attention_params.paged_kv_last_page_len =
            attn_metadata.unshared_paged_kv_last_page_len;

        // unshared_attention_params.plan_info = attn_metadata.plan_info;

        xllm::kernel::batch_decode(unshared_attention_params);
        // LOG(INFO) << "unshared_o: " << unshared_o;
        // LOG(INFO) << "unshared_lse: " << unshared_lse;
      }
      xllm::kernel::cuda::lse_combine(
          output, shared_o, shared_lse, unshared_o, unshared_lse);
      // LOG(INFO) << "output: " << output;
      // LOG(INFO) << "output_lse.shape: " << output_lse.sizes();
      // LOG(INFO) << "shared_o.shape: " << shared_o.sizes();
      // LOG(INFO) << "shared_lse.shape: " << shared_lse.sizes();
      // LOG(INFO) << "unshared_o.shape: " << unshared_o.sizes();
      // LOG(INFO) << "unshared_lse.shape: " << unshared_lse.sizes();
      // LOG(FATAL) << "after lse_combine.";

      // LOG(INFO) << "output: " << output;
      // LOG(FATAL) << "after batch_decode.";
    } else {
      LLM_NVTX_RANGE("attention_decode_standard");

      {
        LLM_NVTX_RANGE_COLOR("batch_decode_standard", 0xFF0000FF);  // Blue
        query = query.view({-1, 1, num_heads_, head_size_});
        output = output.view({-1, 1, num_heads_, head_size_});
        xllm::kernel::AttentionParams decode_attention_params;
        decode_attention_params.window_size_left = sliding_window_;
        decode_attention_params.scale = scale_;
        decode_attention_params.compute_dtype = attn_metadata.compute_dtype;

        decode_attention_params.query = query;
        decode_attention_params.output = output;
        decode_attention_params.k_cache = k_cache;
        decode_attention_params.v_cache = v_cache;

        // for flashinfer
        decode_attention_params.float_workspace_buffer =
            FlashinferWorkspace::get_instance().get_float_workspace_buffer();
        decode_attention_params.int_workspace_buffer =
            FlashinferWorkspace::get_instance().get_int_workspace_buffer();
        decode_attention_params.page_locked_int_workspace_buffer =
            FlashinferWorkspace::get_instance()
                .get_page_locked_int_workspace_buffer();
        decode_attention_params.paged_kv_indptr = attn_metadata.paged_kv_indptr;
        decode_attention_params.paged_kv_indices =
            attn_metadata.paged_kv_indices;
        decode_attention_params.paged_kv_last_page_len =
            attn_metadata.paged_kv_last_page_len;

        xllm::kernel::batch_decode(decode_attention_params);
      }

      // LOG(INFO) << "output: " << output;
      // LOG(FATAL) << "after batch_decode.";
    }
  }
  // LOG(INFO) << "output: " << output;
  output = output.view({-1, num_heads_ * head_size_});
  return {output, output_lse};
}

}  // namespace layer
}  // namespace xllm