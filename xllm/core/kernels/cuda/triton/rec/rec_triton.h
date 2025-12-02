#pragma once

#include <torch/torch.h>

#include "xllm/core/kernels/cuda/triton/ptx_kernel_loader.h"

namespace xllm::kernel::cuda::triton {

class RecTritonKernel {
 public:
  RecTritonKernel();

  void beam_search(torch::Tensor log_probs, 
                   torch::Tensor in_sequence, 
                   torch::Tensor top_tokens, 
                   torch::Tensor top_probs, 
                   torch::Tensor out_log_probs, 
                   torch::Tensor out_token_ids, 
                   torch::Tensor out_token_index, 
                   torch::Tensor out_beam_count_prefix_sums, 
                   torch::Tensor out_sequence, 
                   uint32_t max_decode_step, 
                   uint32_t current_step);
 private:
  using BeamSearchKernelConfig = GenericKernelConfig<OneDimBlockSize>;

  GenericKernelConfigs<BeamSearchKernelConfig> bf16_beam_search_configs_;
  std::vector<std::vector<int>> bf16_beam_search_input_dim_array_;
};

} // namespace xllm::kernel::cuda::triton