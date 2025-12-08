#pragma once

#include <torch/torch.h>

#include "xllm/core/kernels/cuda/triton/ptx_kernel_loader.h"

namespace xllm::kernel::cuda::triton {

class RecTritonKernel {
 public:
  RecTritonKernel();

  ~RecTritonKernel();

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
  
  void grouped_cache_select(torch::Tensor beam_index_ptr, 
                            std::vector<torch::Tensor> unshared_k_cache, 
                            std::vector<torch::Tensor> unshared_v_cache, 
                            torch::Tensor block_table, 
                            uint32_t current_step);
    
  torch::Tensor xattention(torch::Tensor q, 
                  torch::Tensor shared_k_cache, 
                  torch::Tensor shared_v_cache, 
                  torch::Tensor unshared_k_cache, 
                  torch::Tensor unshared_v_cache, 
                  uint32_t decode_step, 
                  uint32_t beam_size, 
                  float sm_scale, 
                  uint32_t prompt_len);

 private:

  struct TensorDescriptorMetaData {
    explicit TensorDescriptorMetaData(const uint64_t* shape_ptr,
                                      const uint64_t* strides_ptr,
                                      const uint32_t* block_shape_ptr,
                                      const uint32_t* element_strides_ptr, 
                                      uint32_t input_tensor_rank)
                                       : tensor_rank(input_tensor_rank) {
      // 分配内存
      shape = new uint64_t[input_tensor_rank];
      strides = new uint64_t[input_tensor_rank];
      block_shape = new uint32_t[input_tensor_rank];
      element_strides = new uint32_t[input_tensor_rank];
      
      memcpy(shape, shape_ptr, sizeof(uint64_t) * input_tensor_rank);
      memcpy(strides, strides_ptr, sizeof(uint64_t) * (input_tensor_rank));
      memcpy(block_shape, block_shape_ptr, sizeof(uint32_t) * input_tensor_rank);
      memcpy(element_strides, element_strides_ptr, sizeof(uint32_t) * input_tensor_rank);

// LOG(INFO) << "shape: " << shape[0] << ", " << shape[1] << ", " << shape[2] << ", " << shape[3];
// LOG(INFO) << "strides: " << strides[0] << ", " << strides[1] << ", " << strides[2];
// LOG(INFO) << "block_shape: " << block_shape[0] << ", " << block_shape[1] << ", " << block_shape[2] << ", " << block_shape[3];
// LOG(INFO) << "element_strides: " << element_strides[0] << ", " << element_strides[1] << ", " << element_strides[2] << ", " << element_strides[3];
    }

    ~TensorDescriptorMetaData() {
      delete[] shape;
      delete[] strides;
      delete[] block_shape;
      delete[] element_strides;
    }
      
    uint64_t* shape; // tensor_rank
    uint64_t* strides; // tensor_rank - 1
    uint32_t* block_shape; // tensor_rank
    uint32_t* element_strides; // tensor_rank

    uint32_t tensor_rank;
  };

  struct TensorDescriptor {
    TensorDescriptor(torch::Tensor tensor, 
                     const uint64_t* shape_ptr,
                     const uint64_t* strides_ptr,
                     const uint32_t* block_shape_ptr,
                     const uint32_t* element_strides_ptr, 
                     uint32_t tensor_rank)
      : base(tensor), unified_metadata_(shape_ptr, strides_ptr, block_shape_ptr, element_strides_ptr, tensor_rank) {}

    torch::Tensor base;

    TensorDescriptorMetaData unified_metadata_;
  };

  std::tuple<torch::Tensor, torch::Tensor> attention_shared_wrapper_forward(torch::Tensor q, 
                                                                            torch::Tensor shared_k_cache, 
                                                                            torch::Tensor shared_v_cache, 
                                                                            float sm_scale, 
                                                                            uint32_t batch_size, 
                                                                            uint32_t beam_width, 
                                                                            uint32_t prompt_len);

  std::tuple<torch::Tensor, torch::Tensor> attn_fwd_shared(float sm_scale, 
                       torch::Tensor M_shared,
                       uint32_t Z, 
                       uint32_t H, 
                       const TensorDescriptor& desc_q, 
                       const TensorDescriptor& desc_k, 
                       const TensorDescriptor& desc_v, 
                       const TensorDescriptor& desc_o, 
                       uint32_t N_CTX, 
                       uint32_t N_KTX, 
                       uint32_t HEAD_DIM, 
                       uint32_t batch_size, 
                       uint32_t beam_width, 
                       uint32_t prompt_len);
  
  void attention_unshared_wrapper_forward(torch::Tensor q_unshared, 
                                          torch::Tensor unshared_k_cache, 
                                          torch::Tensor unshared_v_cache, 
                                          torch::Tensor o_unshared, 
                                          torch::Tensor M_unshared, 
                                          torch::Tensor L_unshared, 
                                          uint32_t total_beams, 
                                          uint32_t num_heads, 
                                          uint32_t head_dim, 
                                          uint32_t kv_total_size, 
                                          float sm_scale, 
                                          uint32_t kv_heads, 
                                          uint32_t beam_size, 
                                          uint32_t max_decode_step, 
                                          uint32_t decode_step, 
                                          uint32_t N_KTX,
                                          uint32_t batch_size, 
                                          uint32_t prompt_length);

  void attn_fwd_unshared(const TensorDescriptor& desc_q,
                         const TensorDescriptor& desc_o, 
                         const TensorDescriptor& desc_unshared_k, 
                         const TensorDescriptor& desc_unshared_v, 
                         torch::Tensor M_unshared,
                         torch::Tensor L_unshared,
                         uint32_t kv_heads, 
                         float sm_scale, 
                         uint32_t batch_size, 
                         uint32_t total_beams, 
                         uint32_t beam_size, 
                         uint32_t max_decode_step, 
                         uint32_t current_step, 
                         uint32_t num_heads, 
                         uint32_t prompt_length);
  
  void combine_attention_kernel(torch::Tensor shared_out_2d, 
                                torch::Tensor unshared_out_2d, 
                                torch::Tensor M_shared, 
                                torch::Tensor L_shared, 
                                torch::Tensor M_unshared, 
                                torch::Tensor L_unshared, 
                                torch::Tensor final_out_2d, 
                                uint32_t total_beams, 
                                uint32_t num_heads, 
                                uint32_t head_dim, 
                                uint32_t batch_size, 
                                uint32_t prompt_length);
  
  

  CUresult create_tensor_map(const TensorDescriptor& tensor_descriptor, 
                             CUtensorMap* tensorMap, 
                             int32_t swizzle_type);

  using BeamSearchKernelConfig = GenericKernelConfig<OneDimBlockSize>;
  using GroupedCacheSelectKernelConfig = GenericKernelConfig<ThreeDimBlockSize>;
  using AttnFwdSharedKernelConfig = GenericKernelConfig<ThreeDimBlockSize>;
  using AttnFwdUnsharedKernelConfig = GenericKernelConfig<ThreeDimBlockSize>;
  using CombineAttentionKernelConfig = GenericKernelConfig<ThreeDimBlockSize>;

  GenericKernelConfigs<BeamSearchKernelConfig> bf16_beam_search_configs_;
  std::vector<std::vector<int>> bf16_beam_search_input_dim_array_;

  GenericKernelConfigs<GroupedCacheSelectKernelConfig> bf16_grouped_cache_select_configs_;
  std::vector<std::vector<int>> bf16_grouped_cache_select_input_dim_array_;

  GenericKernelConfigs<AttnFwdSharedKernelConfig> bf16_attn_fwd_shared_configs_;
  std::vector<std::vector<int>> bf16_attn_fwd_shared_input_dim_array_;
  GenericKernelConfigs<AttnFwdUnsharedKernelConfig> bf16_attn_fwd_unshared_configs_;
  std::vector<std::vector<int>> bf16_attn_fwd_unshared_input_dim_array_;
  GenericKernelConfigs<CombineAttentionKernelConfig> bf16_combine_attention_configs_;
  std::vector<std::vector<int>> bf16_combine_attention_input_dim_array_;

  using TestTMAKernelConfig = GenericKernelConfig<ThreeDimBlockSize>;
  GenericKernelConfigs<TestTMAKernelConfig> bf16_test_tma_configs_;
  std::vector<std::vector<int>> bf16_test_tma_input_dim_array_;
};

} // namespace xllm::kernel::cuda::triton