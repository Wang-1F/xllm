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

#include "mtgr_hopper_attention_runtime.h"

#include "core/util/mtgr_nvtx.h"

// Reuse the physically split Hopper research implementation while exposing a
// production-facing wrapper symbol from a non-test TU.
#include "tests/mtgr_ragged_hopper_attention_kernel.cu"

namespace xllm::kernel::cuda {

void mtgr_ragged_segment_attention_hopper_unified_cuda(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    const torch::Tensor& segment_offsets_i32,
    const torch::Tensor& segment_rules_i32,
    const torch::Tensor& q_seq_starts_i32,
    const torch::Tensor& matched_prefix_lens_i32,
    int64_t match_mode,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& block_table_i32,
    int64_t block_size,
    int64_t max_request_len,
    double sm_scale,
    torch::Tensor output_snd) {
  MTGR_NVTX_RANGE(1, "MTGR/kernel/runtime_wrapper");
  MTGR_TRACE(2) << "[KERNEL] runtime_wrapper begin match_mode=" << match_mode
                << " query=" << query_snd.sizes()
                << " key_cache=" << key_cache.sizes()
                << " block_table=" << block_table_i32.sizes();
  mtgr_ragged_segment_attention_hopper_unified_research_cuda(query_snd,
                                                             key_snd,
                                                             value_snd,
                                                             segment_offsets_i32,
                                                             segment_rules_i32,
                                                             q_seq_starts_i32,
                                                             matched_prefix_lens_i32,
                                                             match_mode,
                                                             key_cache,
                                                             value_cache,
                                                             block_table_i32,
                                                             block_size,
                                                             max_request_len,
                                                             sm_scale,
                                                             output_snd);
}

}  // namespace xllm::kernel::cuda
