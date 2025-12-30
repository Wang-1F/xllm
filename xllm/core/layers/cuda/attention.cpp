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

#include "flashinfer_workspace.h"
#include "kernels/ops_api.h"
#include "kernels/cuda/cuda_ops_api.h"

DECLARE_bool(enable_chunked_prefill);

namespace {

void lse_combine(torch::Tensor shared_o, 
                 torch::Tensor shared_lse, 
                 torch::Tensor unshared_o, 
                 torch::Tensor unshared_lse, 
                 torch::Tensor output) {
  xllm::kernel::cuda::lse_combine(output, shared_o, shared_lse, unshared_o, unshared_lse);
}
} // namespace

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
  // LOG(INFO) << "inner AttentionImpl::forward.";
  auto output = torch::empty_like(query);
  auto output_lse = std::nullopt;
  if (attn_metadata.max_seq_len == 0) {
    output = output.view({-1, num_heads_ * head_size_});
    return std::make_tuple(output, output_lse);
  }

  query = query.view({-1, num_heads_, head_size_});
  key = key.view({-1, num_kv_heads_, head_size_});
  value = value.view({-1, num_kv_heads_, head_size_});
  output = output.view({-1, num_heads_, head_size_});

  torch::Tensor k_cache = kv_cache.get_k_cache();
  torch::Tensor v_cache = kv_cache.get_v_cache();

  if (FLAGS_max_decode_rounds == 0) {
    xllm::kernel::ReshapePagedCacheParams reshape_paged_cache_params;
    reshape_paged_cache_params.key = key;
    reshape_paged_cache_params.value = value;
    reshape_paged_cache_params.k_cache = k_cache;
    reshape_paged_cache_params.v_cache = v_cache;
    reshape_paged_cache_params.slot_mapping = attn_metadata.slot_mapping;
    xllm::kernel::reshape_paged_cache(reshape_paged_cache_params);
  }
  
  if (attn_metadata.is_prefill) {
    CHECK(!attn_metadata.is_chunked_prefill)
        << "chunked prefill is not supported";
    if (FLAGS_max_decode_rounds > 0) {
      // LOG(INFO) << "kv_seq_lens: " << attn_metadata.kv_seq_lens;
      // LOG(INFO) << "kv_cu_seq_lens: " << attn_metadata.kv_cu_seq_lens;
      // LOG(INFO) << "shared_k_cache.shape: " << attn_metadata.shared_k_cache.sizes();
      // LOG(INFO) << "shared_v_cache.shape: " << attn_metadata.shared_v_cache.sizes();
      // 改成了batch_size, num_shared_kv_seq_len, kv_heads, head_dim
      rec_kernel_->prefill_reshape_and_cache(key, 
                                             value, 
                                             attn_metadata.kv_cu_seq_lens,
                                             attn_metadata.shared_k_cache, 
                                             attn_metadata.shared_v_cache);
    }
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
    
    attention_params.key = key;
    attention_params.value = value;
    // LOG(INFO) << "key.shape: " << key.sizes();
    // LOG(INFO) << "value.shape: " << value.sizes();
    xllm::kernel::batch_prefill(attention_params);
  } else {
    if (FLAGS_max_decode_rounds > 0) {
      auto fp32_options =
        torch::TensorOptions().dtype(torch::kFloat32).device(query.device());
      
      uint32_t batch_size = attn_metadata.kv_cu_seq_lens.size(0) - 1;
      uint32_t total_beam = query.size(0);
      uint32_t beam_size = total_beam / batch_size;

      query = query.view({-1, 1, num_heads_, head_size_});

      torch::Tensor shared_lse = attn_metadata.shared_lse;
      torch::Tensor shared_o = attn_metadata.shared_o;
      shared_o = shared_o.view({-1, 1, num_heads_, head_size_});
      
      //问题在于kv_cache少了beam_size维度

      xllm::kernel::AttentionParams shared_attention_params;
      shared_attention_params.return_lse = true;
      shared_attention_params.query = query;
      shared_attention_params.output = shared_o;
      shared_attention_params.output_lse = shared_lse;

      // shared_attention_params.max_seq_len = attn_metadata.max_seq_len;
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
      // [batch_size, num_shared_kv_seq_len, kv_heads, head_dim]
      // [batch_size, beam_size, num_heads, head_dim]
      // auto shared_k_cache = attn_metadata.shared_k_cache.unsqueeze(1).expand({-1, beam_size, -1, -1, -1});
      // auto shared_v_cache = attn_metadata.shared_v_cache.unsqueeze(1).expand({-1, beam_size, -1, -1, -1});
      // shared_v_cache = shared_v_cache.view({-1, attn_metadata.shared_k_cache.size(1), num_kv_heads_, head_size_});
      // shared_k_cache = shared_k_cache.view({-1, attn_metadata.shared_k_cache.size(1), num_kv_heads_, head_size_});
      // LOG(INFO) << "shared_k_cache.shape: " << shared_k_cache.sizes();
      // LOG(INFO) << "shared_k_cache.shape: " << shared_v_cache.sizes();
      auto shared_k_cache = attn_metadata.shared_k_cache;
      auto shared_v_cache = attn_metadata.shared_v_cache;
      shared_attention_params.k_cache = shared_k_cache;
      shared_attention_params.v_cache = shared_v_cache;

      auto batch_offsets = torch::arange(0, batch_size, attn_metadata.paged_kv_indptr.options());
      batch_offsets = batch_offsets.unsqueeze(1).expand({-1, beam_size});

      auto beam_offsets = torch::zeros({beam_size}, attn_metadata.paged_kv_indptr.options());
      // auto beam_offsets = torch::arange(0, beam_size, attn_metadata.paged_kv_indptr.options());
      beam_offsets = beam_offsets.unsqueeze(0).expand({batch_size, -1});
      // auto batch_beam_offsets = batch_offsets * beam_size + beam_offsets;
      auto batch_beam_offsets = batch_offsets + beam_offsets;

      shared_attention_params.paged_kv_indptr = 
        torch::arange(batch_size * beam_size + 1, attn_metadata.paged_kv_indptr.options());
      shared_attention_params.paged_kv_indices = 
        batch_beam_offsets.flatten();
      
      
      auto batch_kv_last_page_len = torch::diff(attn_metadata.kv_cu_seq_lens);
      batch_kv_last_page_len = batch_kv_last_page_len.unsqueeze(1).expand({-1, beam_size});
      
      shared_attention_params.paged_kv_last_page_len = batch_kv_last_page_len.flatten();
      // shared_attention_params.paged_kv_indices = attn_metadata.paged_kv_indices;
      // shared_attention_params.paged_kv_indptr = attn_metadata.paged_kv_indptr;
      // shared_attention_params.paged_kv_last_page_len = attn_metadata.paged_kv_last_page_len;
      LOG(INFO) << "shared_attention_params.paged_kv_indices: " << shared_attention_params.paged_kv_indices;
      LOG(INFO) << "shared_attention_params.paged_kv_last_page_len: " << shared_attention_params.paged_kv_last_page_len;
      LOG(INFO) << "shared_attention_params.paged_kv_indptr: " << shared_attention_params.paged_kv_indptr;
      // TODO: support chunked prefill
      CHECK(!attn_metadata.is_chunked_prefill)
          << "chunked prefill is not supported";

      xllm::kernel::batch_decode(shared_attention_params);
      
      shared_o = shared_o.view({-1, num_heads_, head_size_});

      // unshared
      key = key.view({batch_size, beam_size, num_kv_heads_, head_size_});
      value = value.view({batch_size, beam_size, num_kv_heads_, head_size_});
      xllm::kernel::cuda::decoder_reshape_and_cache(key, 
                                                    value, 
                                                    k_cache, 
                                                    v_cache, 
                                                    attn_metadata.block_table, 
                                                    attn_metadata.step);
      torch::Tensor unshared_lse = attn_metadata.unshared_lse;
      torch::Tensor unshared_o = attn_metadata.unshared_o;
      
      xllm::kernel::AttentionParams unshared_attention_params;
      // auto unshared_lse = std::nullopt;
      
      unshared_attention_params.return_lse = true;
      unshared_attention_params.output_lse = unshared_lse;
      LOG(INFO) << "unshared_lse.shape: " << unshared_lse.sizes();
      LOG(INFO) << "unshared_lse.dtype: " << unshared_lse.dtype();
      LOG(INFO) << "unshared_o.shape: " << unshared_o.sizes();
      LOG(INFO) << "unshared_o.dtype: " << unshared_o.dtype();
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
      
      // total_beams = batch_size * beam_size
      query = query.view({-1, 1, num_heads_, head_size_});
      unshared_o = unshared_o.view({-1, 1, num_heads_, head_size_});
      // LOG(INFO) << "query.shape: " << query.sizes();
      unshared_attention_params.query = query;
      unshared_attention_params.output = unshared_o;
      LOG(INFO) << "query.dtype: " << query.dtype();
      LOG(INFO) << "unshared_o.dtype: " << unshared_o.dtype();
      LOG(INFO) << "query.shape: " << query.sizes();
      LOG(INFO) << "unshared_o.shape: " << unshared_o.sizes();
      int64_t max_decode_step = k_cache.size(2);

      k_cache = k_cache.view({-1, max_decode_step, num_kv_heads_, head_size_});
      v_cache = v_cache.view({-1, max_decode_step, num_kv_heads_, head_size_});

      unshared_attention_params.k_cache = k_cache;
      unshared_attention_params.v_cache = v_cache;
      LOG(INFO) << "k_cache.shape: " << k_cache.sizes();
      LOG(INFO) << "v_cache.shape: " << v_cache.sizes();
      LOG(INFO) << "k_cache.dtype: " << k_cache.dtype();
      LOG(INFO) << "v_cache.dtype: " << v_cache.dtype();
      unshared_attention_params.paged_kv_indices = attn_metadata.paged_kv_indices_unshared;
      unshared_attention_params.paged_kv_indptr = attn_metadata.paged_kv_indptr_unshared;
      unshared_attention_params.paged_kv_last_page_len = attn_metadata.paged_kv_last_page_len_unshared;
      LOG(INFO) << "unshared_attention_params.paged_kv_indices: " << unshared_attention_params.paged_kv_indices;
      LOG(INFO) << "unshared_attention_params.paged_kv_indptr: " << unshared_attention_params.paged_kv_indptr;
      LOG(INFO) << "unshared_attention_params.paged_kv_last_page_len: " << unshared_attention_params.paged_kv_last_page_len;
      xllm::kernel::batch_decode(unshared_attention_params);
      // LOG(INFO) << "after kernel::batch_decode.";
      // combine
      unshared_o = unshared_o.view({-1, num_heads_, head_size_});
      // LOG(INFO) << "unshared_o.shape: " << unshared_o.sizes();
      xllm::kernel::cuda::lse_combine(output, shared_o, shared_lse, unshared_o, unshared_lse);
      LOG(INFO) << "output: " << output;
      LOG(FATAL) << "after batch_decode.";
    } else {
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
      decode_attention_params.paged_kv_indices = attn_metadata.paged_kv_indices;
      decode_attention_params.paged_kv_last_page_len =
          attn_metadata.paged_kv_last_page_len;
      
      xllm::kernel::batch_decode(decode_attention_params);
      
      LOG(INFO) << "output: " << output;
      LOG(FATAL) << "after batch_decode.";
    }

  }
  // LOG(INFO) << "output: " << output;
  output = output.view({-1, num_heads_ * head_size_});
  return {output, output_lse};
}

}  // namespace layer
}  // namespace xllm