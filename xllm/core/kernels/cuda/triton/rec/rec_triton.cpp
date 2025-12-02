#include "rec_triton.h"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "ptx_kernels/beam_search_sm_90.ptx.h"

namespace xllm::kernel::cuda::triton {

RecTritonKernel::RecTritonKernel() {
  auto arch = get_cuda_arch();
  if (arch == ARCH::SM_90) {
    PROCESS_KERNEL(sm_90,
                   bf16,
                   BeamSearchKernelConfig,
                   bf16_beam_search_configs_,
                   bf16_beam_search_input_dim_array_,
                   beam_search,
                   BLOCK_SINGLE(beam_search_sm_90_bf16));
  } else {
    LOG(FATAL) << "do not support arch but SM_90.";
  }
}

void RecTritonKernel::beam_search(torch::Tensor log_probs, 
                                  torch::Tensor in_sequence, 
                                  torch::Tensor top_tokens, 
                                  torch::Tensor top_probs, 
                                  torch::Tensor out_log_probs, 
                                  torch::Tensor out_token_ids, 
                                  torch::Tensor out_token_index, 
                                  torch::Tensor out_beam_count_prefix_sums, 
                                  torch::Tensor out_sequence, 
                                  uint32_t max_decode_step, 
                                  uint32_t current_step) {
  LOG(INFO) << "log_probs.shape: " << log_probs.sizes();
  LOG(INFO) << "in_sequence.shape: " << in_sequence.sizes();
  LOG(INFO) << "top_tokens.shape: " << top_tokens.sizes();
  LOG(INFO) << "top_probs.shape: " << top_probs.sizes();
  
  CUfunction beam_search_kernel;
  uint32_t beam_search_shared_mem_bytes = 0;

  switch (log_probs.scalar_type()) {
    case torch::kBFloat16: {
      LOG(INFO) << "inner log_probs is bf16.";
      const auto input_dim =
          std::array<int, 3>{static_cast<int>(in_sequence.size(0)),
                             static_cast<int>(in_sequence.size(1)), 
                             current_step};
      const auto& beam_search_config =
          bf16_beam_search_configs_.find_closest_index(
              bf16_beam_search_input_dim_array_, input_dim);
      beam_search_kernel = beam_search_config.kernel;

      beam_search_shared_mem_bytes =
          beam_search_config.shared_mem_bytes;
      break;
    }
    default: {
      LOG(FATAL) << "RecTritonKernel: beam_search_kernel get upsupported dtype "
                 << log_probs.scalar_type();
      break;
    }
  }

  CUdeviceptr log_probs_ptr =
      reinterpret_cast<CUdeviceptr>(log_probs.data_ptr());
  CUdeviceptr in_sequence_ptr =
      reinterpret_cast<CUdeviceptr>(in_sequence.data_ptr());
  CUdeviceptr top_tokens_ptr =
      reinterpret_cast<CUdeviceptr>(top_tokens.data_ptr());
  CUdeviceptr top_probs_ptr =
      reinterpret_cast<CUdeviceptr>(top_probs.data_ptr());
  CUdeviceptr out_log_probs_ptr =
      reinterpret_cast<CUdeviceptr>(out_log_probs.data_ptr());
  CUdeviceptr out_token_ids_ptr =
      reinterpret_cast<CUdeviceptr>(out_token_ids.data_ptr());
  CUdeviceptr out_token_index_ptr =
      reinterpret_cast<CUdeviceptr>(out_token_index.data_ptr());
  CUdeviceptr out_beam_count_prefix_sums_ptr =
      reinterpret_cast<CUdeviceptr>(out_beam_count_prefix_sums.data_ptr());
  CUdeviceptr out_sequence_ptr =
      reinterpret_cast<CUdeviceptr>(out_sequence.data_ptr());
  LOG(INFO) << "before at::cuda::getCurrentCUDAStream().";
  // TODO
  auto stream = at::cuda::getCurrentCUDAStream();
  uint32_t batch_size = in_sequence.size(0);
  LOG(INFO) << "batch_size: " << batch_size;
  LOG(INFO) << "current_step: " << current_step;
  uint32_t beam_size = in_sequence.size(1);
  LOG(INFO) << "beam_size: " << beam_size;
  uint32_t top_k = top_tokens.size(1);
  LOG(INFO) << "top_k: " << top_k;
  void* args[] = {(void*)&log_probs_ptr,
                  (void*)&in_sequence_ptr,
                  (void*)&top_tokens_ptr,
                  (void*)&top_probs_ptr,
                  (void*)&out_log_probs_ptr,
                  (void*)&out_token_ids_ptr,
                  (void*)&out_token_index_ptr,
                  (void*)&out_beam_count_prefix_sums_ptr,
                  (void*)&out_sequence_ptr, 
                  (void*)&batch_size, 
                  (void*)&max_decode_step, 
                  (void*)&current_step, 
                  (void*)&current_step};
  uint32_t thread_nums = get_default_thread_nums();
  LOG(INFO) << "thread_nums: " << thread_nums;
  LOG(INFO) << "before cuLaunchKernel.";
  CUDA_CHECK(cuLaunchKernel(beam_search_kernel,
                            batch_size,
                            1,
                            1,
                            thread_nums,
                            1,
                            1,
                            beam_search_shared_mem_bytes,
                            stream,
                            args,
                            NULL));
  LOG(INFO) << "after cuLaunchKernel.";
  cudaStreamSynchronize(stream);
}

} // namespace xllm::kernel::cuda::triton