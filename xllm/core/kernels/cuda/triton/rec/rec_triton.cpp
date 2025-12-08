#include "rec_triton.h"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include "cuda.h"

#include "ptx_kernels/beam_search_sm_90.ptx.h"
#include "ptx_kernels/group_cache_select_sm_90.ptx.h"
#include "ptx_kernels/fake_xattention_sm_90.ptx.h"

#include "ptx_kernels/test_tma_sm_90.ptx.h"

namespace {

CUtensorMapDataType
get_target_tensor_map_dtype(const c10::ScalarType& dtype) {
  switch (dtype) {
    case torch::kInt32:
      return CUtensorMapDataType_enum::CU_TENSOR_MAP_DATA_TYPE_UINT32;
      break;
    case torch::kBFloat16:
      return CUtensorMapDataType_enum::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
      break;
    case torch::kFloat16:
      return CUtensorMapDataType_enum::CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
      break;
    case torch::kFloat32:
      return CUtensorMapDataType_enum::CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
      break;
    default:
      LOG(FATAL) << "unsupported dtype: " << dtype;
  }
}

constexpr bool warp_specialize{false};

}

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
    PROCESS_KERNEL(sm_90,
                   bf16,
                   GroupedCacheSelectKernelConfig,
                   bf16_grouped_cache_select_configs_,
                   bf16_grouped_cache_select_input_dim_array_,
                   group_cache_select,
                   BLOCK_TRIPLE(group_cache_select_sm_90_bf16));  

    PROCESS_KERNEL(sm_90,
                   bf16,
                   AttnFwdSharedKernelConfig,
                   bf16_attn_fwd_shared_configs_,
                   bf16_attn_fwd_shared_input_dim_array_,
                   xattention_shared_new,
                   BLOCK_TRIPLE(xattention_shared_new_sm_90_bf16));    
    PROCESS_KERNEL(sm_90,
                   bf16,
                   AttnFwdUnsharedKernelConfig,
                   bf16_attn_fwd_unshared_configs_,
                   bf16_attn_fwd_unshared_input_dim_array_,
                   xattention_unshared,
                   BLOCK_TRIPLE(xattention_unshared_sm_90_bf16));   
    PROCESS_KERNEL(sm_90,
                   bf16,
                   CombineAttentionKernelConfig,
                   bf16_combine_attention_configs_,
                   bf16_combine_attention_input_dim_array_,
                   xattention_combine,
                   BLOCK_TRIPLE(xattention_combine_sm_90_bf16));     

    PROCESS_KERNEL(sm_90,
                   bf16,
                   TestTMAKernelConfig,
                   bf16_test_tma_configs_,
                   bf16_test_tma_input_dim_array_,
                   simple_tma_copy_kernel,
                   BLOCK_TRIPLE(simple_tma_copy_kernel_sm_90_bf16));           
  } else {
    LOG(FATAL) << "do not support arch but SM_90.";
  }
}

RecTritonKernel::~RecTritonKernel() {

}

CUresult 
RecTritonKernel::create_tensor_map(const TensorDescriptor& tensor_descriptor, 
                                   CUtensorMap* tensorMap, 
                                   int32_t swizzle_type) {
  const auto& tensor = tensor_descriptor.base;
  const auto& metadata = tensor_descriptor.unified_metadata_;
  {
    // 1. 检查 CUDA Context
    CUcontext current_ctx;
    CUresult ctx_result = cuCtxGetCurrent(&current_ctx);
    if (ctx_result != CUDA_SUCCESS || current_ctx == nullptr) {
      LOG(ERROR) << "No valid CUDA context";
      return CUDA_ERROR_INVALID_CONTEXT;
    }
  }
  {
    // 1. 检查 TensorMap 对齐
    if (reinterpret_cast<uintptr_t>(tensorMap) % 64 != 0) {
      LOG(ERROR) << "TensorMap not 64-byte aligned: " << tensorMap;
      return CUDA_ERROR_INVALID_VALUE;
    }

    // 2. 检查张量维度
    cuuint32_t tensor_rank = tensor.dim();
    if (tensor_rank == 0 || tensor_rank > 5) {
      LOG(ERROR) << "Invalid tensor rank: " << tensor_rank;
      return CUDA_ERROR_INVALID_VALUE;
    }

    // 3. 检查设备指针
    void* tensor_addr = tensor.data_ptr();
    if (tensor_addr == nullptr) {
      LOG(ERROR) << "Null tensor address";
      return CUDA_ERROR_INVALID_VALUE;
    }
  }

  CUtensorMapDataType dtype = 
    get_target_tensor_map_dtype(tensor.scalar_type());
  LOG(INFO) << "dtype: " << dtype;
  cuuint32_t tensor_rank = tensor.dim();
  LOG(INFO) << "tensor_rank: " << tensor_rank;
  void* tensor_addr = tensor.data_ptr();
  const cuuint64_t* dims = metadata.shape;
  LOG(INFO) << "dims: " << metadata.shape[0];
  const cuuint64_t* strides = metadata.strides;
  LOG(INFO) << "strides: " << metadata.strides[0];
  const cuuint32_t* block_shape = metadata.block_shape;
  LOG(INFO) << "block_shape: " << metadata.block_shape[0];
  const cuuint32_t* element_strides = metadata.element_strides;
  CUresult result = cuTensorMapEncodeTiled(
        tensorMap,                    // 输出：填充这个结构体
        dtype,
        tensor_rank,                            // 2D 张量
        tensor_addr,                   // 基地址
        dims,                         // 维度信息
        strides,                      // 步长信息
        block_shape, 
        element_strides,             // 使用默认块大小
        CUtensorMapInterleave_enum::CU_TENSOR_MAP_INTERLEAVE_NONE,
        static_cast<CUtensorMapSwizzle_enum>(swizzle_type),   // 128B swizzle 优化
        // CUtensorMapSwizzle_enum::CU_TENSOR_MAP_SWIZZLE_128B, 
        CUtensorMapL2promotion_enum::CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CUtensorMapFloatOOBfill_enum::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
      );

  if (result != CUDA_SUCCESS) {
    const char* error_string;
    cuGetErrorString(result, &error_string);
    LOG(ERROR) << "cuTensorMapEncodeTiled failed: " << error_string 
               << " (code: " << result << ")";
  }

  return result;

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

void RecTritonKernel::grouped_cache_select(torch::Tensor beam_index, 
                                           std::vector<torch::Tensor> unshared_k_cache, 
                                           std::vector<torch::Tensor> unshared_v_cache, 
                                           torch::Tensor block_table, 
                                           uint32_t current_step) {
  CHECK_GE(unshared_k_cache.size(), 0) 
    << "kv_cache layer_num should be greater equal 0.";
  CHECK_GE(unshared_v_cache.size(), 0) 
    << "kv_cache layer_num should be greater equal 0.";

  LOG(INFO) << "beam_index.shape: " << beam_index.sizes();
  LOG(INFO) << "unshared_k_cache.shape: " << unshared_k_cache[0].sizes();
  LOG(INFO) << "unshared_v_cache.shape: " << unshared_v_cache[0].sizes();
  LOG(INFO) << "block_table.shape: " << block_table.sizes();
  
  uint32_t layer_num = unshared_k_cache.size();

  CUfunction grouped_cache_select_kernel;
  uint32_t grouped_cache_select_shared_mem_bytes = 0;

  
  uint32_t batch_size = unshared_k_cache[0].size(0);
  uint32_t beam_size = unshared_k_cache[0].size(1);
  LOG(INFO) << "batch_size: " << batch_size;
  LOG(INFO) << "current_step: " << current_step;
  LOG(INFO) << "beam_size: " << beam_size;

  switch (unshared_k_cache[0].scalar_type()) {
    case torch::kBFloat16: {
      LOG(INFO) << "inner unshared_k_cache is bf16.";
      const auto input_dim =
          std::array<int, 3>{static_cast<int>(batch_size),
                             static_cast<int>(beam_size), 
                             static_cast<int>(layer_num)};
      const auto& grouped_cache_select_config =
          bf16_grouped_cache_select_configs_.find_closest_index(
              bf16_grouped_cache_select_input_dim_array_, input_dim);
      grouped_cache_select_kernel = grouped_cache_select_config.kernel;

      grouped_cache_select_shared_mem_bytes =
          grouped_cache_select_config.shared_mem_bytes;
      break;
    }
    default: {
      LOG(FATAL) << "RecTritonKernel: grouped_cache_select_kernel get upsupported dtype "
                 << unshared_k_cache[0].scalar_type();
      break;
    }
  }

  std::vector<void*> _unshared_k_cache;
  _unshared_k_cache.reserve(layer_num);
  std::vector<void*> _unshared_v_cache;
  _unshared_v_cache.reserve(layer_num);
  for (std::size_t i = 0 ; i < layer_num ; ++i) {
    _unshared_k_cache.emplace_back(unshared_k_cache[i].data_ptr());
    _unshared_v_cache.emplace_back(unshared_v_cache[i].data_ptr());
  }

  CUdeviceptr beam_index_ptr =
      reinterpret_cast<CUdeviceptr>(beam_index.data_ptr());
  CUdeviceptr unshared_k_cache_ptr =
      reinterpret_cast<CUdeviceptr>(_unshared_k_cache.data());
  CUdeviceptr unshared_v_cache_ptr =
      reinterpret_cast<CUdeviceptr>(_unshared_v_cache.data());
  CUdeviceptr block_table_ptr =
      reinterpret_cast<CUdeviceptr>(block_table.data_ptr());

  LOG(INFO) << "before at::cuda::getCurrentCUDAStream().";
  // TODO
  auto stream = at::cuda::getCurrentCUDAStream();
  uint32_t head_num = unshared_k_cache[0].size(2);
  LOG(INFO) << "head_num: " << head_num;
  uint32_t max_decode_step = unshared_k_cache[0].size(3);
  LOG(INFO) << "max_decode_step: " << max_decode_step;
  uint32_t head_dim = unshared_k_cache[0].size(4);
  LOG(INFO) << "head_dim: " << head_dim;
  void* args[] = {(void*)&layer_num,
                  (void*)&batch_size,
                  (void*)&beam_index_ptr,
                  (void*)&unshared_k_cache_ptr,
                  (void*)&unshared_v_cache_ptr,
                  (void*)&block_table_ptr,
                  (void*)&current_step,
                  (void*)&beam_size,
                  (void*)&head_num, 
                  (void*)&max_decode_step, 
                  (void*)&head_dim};
  uint32_t thread_nums = get_default_thread_nums();
  LOG(INFO) << "thread_nums: " << thread_nums;
  LOG(INFO) << "before cuLaunchKernel.";
  CUDA_CHECK(cuLaunchKernel(grouped_cache_select_kernel,
                            layer_num,
                            batch_size,
                            head_num,
                            thread_nums,
                            1,
                            1,
                            grouped_cache_select_shared_mem_bytes,
                            stream,
                            args,
                            NULL));
  LOG(INFO) << "after cuLaunchKernel.";
  cudaStreamSynchronize(stream);
}

std::tuple<torch::Tensor, torch::Tensor> RecTritonKernel::attention_shared_wrapper_forward(torch::Tensor q, 
                                                       torch::Tensor shared_k_cache, 
                                                       torch::Tensor shared_v_cache, 
                                                       float sm_scale, 
                                                       uint32_t batch_size, 
                                                       uint32_t beam_width, 
                                                       uint32_t prompt_len) {
  // 获取各个维度的head dimension
  int64_t HEAD_DIM_Q = q.size(-1);
  int64_t HEAD_DIM_K = shared_k_cache.size(-1);
  int64_t HEAD_DIM_V = shared_v_cache.size(-1);

  // 断言检查head dimensions相等
  assert(HEAD_DIM_Q == HEAD_DIM_K && HEAD_DIM_K == HEAD_DIM_V);
  // 检查HEAD_DIM_K是否在支持的范围内
  std::set<int64_t> supported_dims = {16, 32, 64, 128, 256};
  assert(supported_dims.find(HEAD_DIM_K) != supported_dims.end());

  // 获取q张量的形状
  auto q_shape = q.sizes();
  LOG(INFO) << "q_shape: " << q_shape;
  int64_t Z = q_shape[0];
  int64_t H = q_shape[1]; 
  int64_t N_CTX = q_shape[2];
  int64_t HEAD_DIM = q_shape[3];

  // 获取k张量的形状
  auto k_shape = shared_k_cache.sizes();
  int64_t N_KTX = k_shape[2];

  // 设置块大小并计算填充后的N_CTX
  int64_t BLOCK_M = 64;
  int64_t padded_N_CTX = ((N_CTX + BLOCK_M - 1) / BLOCK_M) * BLOCK_M;

  if (padded_N_CTX > N_CTX) {
    int64_t pad_length = padded_N_CTX - N_CTX;
    // 使用torch::nn::functional::pad进行填充
    // 参数: (left, right, top, bottom) 对应最后两个维度
    q = torch::nn::functional::pad(q, 
        torch::nn::functional::PadFuncOptions({0, 0, 0, pad_length})
        .mode(torch::kConstant)
        .value(0.0));
  }

  // 获取q张量的形状
  q_shape = q.sizes();
  LOG(INFO) << "q_new_shape: " << q_shape;
  Z = q_shape[0];
  H = q_shape[1]; 
  // N_CTX = q_shape[2];
  HEAD_DIM = q_shape[3];

  // 创建输出张量
  auto o = torch::empty_like(q);

  // 创建M张量
  auto M = torch::empty({Z, H, q.size(2)}, 
      torch::TensorOptions()
          .device(q.device())
          .dtype(torch::kFloat32));

  uint32_t y_dim = q_shape[0] * q_shape[1] * q_shape[2];
  uint32_t k_dim = k_shape[0] * k_shape[1] * k_shape[2];

  LOG(INFO) << "shared_v_cache.shape: " << shared_v_cache.sizes();
  LOG(INFO) << "q.shape: " << q.sizes();
  LOG(INFO) << "shared_k_cache.shape: " << shared_k_cache.sizes();
  LOG(INFO) << "o.shape: " << o.sizes();
  
  LOG(INFO) << "y_dim: " << y_dim;
  LOG(INFO) << "k_dim: " << k_dim;

  LOG(INFO) << "HEAD_DIM: " << HEAD_DIM;

  LOG(INFO) << "desc_q: ";
  uint64_t q_shapes[2] = {y_dim, HEAD_DIM};
  uint64_t q_strides[2] = {HEAD_DIM * 2, 1};
  uint32_t q_block_shape[2] = {64, 64};
  uint32_t q_elem_strides[2] = {1, 1};
  q = q.view({y_dim, HEAD_DIM});
  LOG(INFO) << "q.scalar_type(): " << q.scalar_type();
  LOG(INFO) << "q.shape: " << q.sizes();
  TensorDescriptor desc_q{q, q_shapes, q_strides, q_block_shape, q_elem_strides, 2};

  LOG(INFO) << "desc_k: ";
  uint64_t k_shapes[2] = {k_dim, HEAD_DIM};
  uint64_t k_strides[2] = {HEAD_DIM * 2, 1};
  uint32_t k_block_shape[2] = {64, 64};
  uint32_t k_elem_strides[2] = {1, 1};
  shared_k_cache = shared_k_cache.view({k_dim, HEAD_DIM});
  TensorDescriptor desc_k{shared_k_cache, k_shapes, k_strides, k_block_shape, k_elem_strides, 2};

  LOG(INFO) << "desc_v: ";
  uint64_t v_shapes[2] = {k_dim, HEAD_DIM};
  uint64_t v_strides[2] = {HEAD_DIM * 2, 1};
  uint32_t v_block_shape[2] = {64, 64};
  uint32_t v_elem_strides[2] = {1, 1};
  shared_v_cache = shared_v_cache.view({k_dim, HEAD_DIM});
  TensorDescriptor desc_v{shared_v_cache, v_shapes, v_strides, v_block_shape, v_elem_strides, 2};

  LOG(INFO) << "desc_o: ";
  uint64_t o_shapes[2] = {y_dim, HEAD_DIM};
  uint64_t o_strides[2] = {HEAD_DIM * 2, 1};
  uint32_t o_block_shape[2] = {64, 64};
  uint32_t o_elem_strides[2] = {1, 1};
  o = o.view({y_dim, HEAD_DIM});
  TensorDescriptor desc_o{o, o_shapes, o_strides, o_block_shape, o_elem_strides, 2};

  LOG(INFO) << "sm_scale: " << sm_scale;
  LOG(INFO) << "M.sizes(): " << M.sizes();
  LOG(INFO) << "q.sizes(): " << q.sizes();
  LOG(INFO) << "shared_k_cache.sizes(): " << shared_k_cache.sizes();
  LOG(INFO) << "shared_v_cache.sizes(): " << shared_v_cache.sizes();
  LOG(INFO) << "o.sizes(): " << o.sizes();

  auto [o_shared, M_shared] = attn_fwd_shared(sm_scale, 
                                              M, 
                                              q_shape[0], 
                                              q_shape[1], 
                                              desc_q, 
                                              desc_k, 
                                              desc_v, 
                                              desc_o, 
                                              q_shape[2], 
                                              N_KTX, 
                                              HEAD_DIM, 
                                              batch_size, 
                                              beam_width, 
                                              prompt_len);
  // 返回两个张量
  return std::make_tuple(o_shared, M_shared);
}

std::tuple<torch::Tensor, torch::Tensor> RecTritonKernel::attn_fwd_shared(float sm_scale, 
                                      torch::Tensor M_shared,
                                      uint32_t Z,  // q_shape[0]
                                      uint32_t H,  // q_shape[1]
                                      const TensorDescriptor& desc_q, 
                                      const TensorDescriptor& desc_k, 
                                      const TensorDescriptor& desc_v, 
                                      const TensorDescriptor& desc_o, 
                                      uint32_t N_CTX,  // q_shape[2]
                                      uint32_t N_KTX, 
                                      uint32_t HEAD_DIM, // q_shape[3]
                                      uint32_t batch_size, 
                                      uint32_t beam_width, 
                                      uint32_t prompt_len) {
  LOG(INFO) << "sm_scale: " << sm_scale;
  LOG(INFO) << "Z: " << Z;
  LOG(INFO) << "H: " << H;
  LOG(INFO) << "N_CTX: " << N_CTX;
  LOG(INFO) << "N_KTX: " << N_KTX;
  LOG(INFO) << "HEAD_DIM: " <<HEAD_DIM;
  LOG(INFO) << "batch_size: " <<batch_size;
  LOG(INFO) << "beam_width: " <<beam_width;
  LOG(INFO) << "prompt_len: " <<prompt_len;
  // 获取kernel配置
  CUfunction attn_fwd_shared_kernel;
  uint32_t attn_fwd_shared_shared_mem_bytes = 0;
  
  switch (desc_k.base.scalar_type()) {
    case torch::kBFloat16: {
      const auto input_dim = 
        std::array<int, 3>{static_cast<int>(batch_size), 
                           static_cast<int>(beam_width), 
                           static_cast<int>(prompt_len)};
      const auto& kernel_config = bf16_attn_fwd_shared_configs_.find_closest_index(
        bf16_attn_fwd_shared_input_dim_array_, input_dim);
      attn_fwd_shared_kernel = kernel_config.kernel;
      attn_fwd_shared_shared_mem_bytes = kernel_config.shared_mem_bytes;
      break;
    }
    default:
      LOG(FATAL) << "Unsupported dtype for shared attention: " << desc_k.base.scalar_type();
  }

  LOG(INFO) << "before create_tensor_map.";
  // unified_metadata_

  LOG(INFO) << "tensor_map_q: ";
  alignas(64) CUtensorMap tensor_map_q;
  CUDA_CHECK(create_tensor_map(desc_q, &tensor_map_q, 3));
  LOG(INFO) << "tensor_map_k: ";
  alignas(64) CUtensorMap tensor_map_k;
  CUDA_CHECK(create_tensor_map(desc_k, &tensor_map_k, 3));
  LOG(INFO) << "tensor_map_v: ";
  alignas(64) CUtensorMap tensor_map_v;
  CUDA_CHECK(create_tensor_map(desc_v, &tensor_map_v, 3));
  LOG(INFO) << "tensor_map_o: ";
  alignas(64) CUtensorMap tensor_map_o;
  CUDA_CHECK(create_tensor_map(desc_o, &tensor_map_o, 3));

  LOG(INFO) << "after create_tensor_map.";

  CUdeviceptr M_shared_ptr = 
    reinterpret_cast<CUdeviceptr>(M_shared.data_ptr());
  
  // 计算grid dimensions
  uint32_t BLOCK_M = 64;  // 从Python配置中获取
  uint32_t grid_x = (N_CTX + BLOCK_M - 1) / BLOCK_M;
  uint32_t grid_y = Z * H;
  
  uint32_t q_shape_0 = static_cast<uint64_t>(desc_q.unified_metadata_.shape[0]);
  uint32_t q_shape_1 = static_cast<uint64_t>(desc_q.unified_metadata_.shape[1]);
  uint32_t k_shape_0 = static_cast<uint64_t>(desc_k.unified_metadata_.shape[0]);
  uint32_t k_shape_1 = static_cast<uint64_t>(desc_k.unified_metadata_.shape[1]);
  uint32_t v_shape_0 = static_cast<uint64_t>(desc_v.unified_metadata_.shape[0]);
  uint32_t v_shape_1 = static_cast<uint64_t>(desc_v.unified_metadata_.shape[1]);
  uint32_t o_shape_0 = static_cast<uint64_t>(desc_o.unified_metadata_.shape[0]);
  uint32_t o_shape_1 = static_cast<uint64_t>(desc_o.unified_metadata_.shape[1]);
  
  uint32_t q_stride_1{1};
  uint32_t k_stride_1{1};
  uint32_t v_stride_1{1};
  uint32_t o_stride_1{1};

  void* args[] = {
    (void*)&sm_scale,
    (void*)&M_shared_ptr,
    
    (void*)&tensor_map_q,
    (void*)&q_shape_0,
    (void*)&q_shape_1,
    (void*)&desc_q.unified_metadata_.strides[0],
    (void*)&q_stride_1,

    (void*)&tensor_map_k,
    (void*)&k_shape_0,
    (void*)&k_shape_1,
    (void*)&desc_k.unified_metadata_.strides[0],
    (void*)&k_stride_1,

    (void*)&tensor_map_v,
    (void*)&v_shape_0,
    (void*)&v_shape_1,
    (void*)&desc_v.unified_metadata_.strides[0],
    (void*)&v_stride_1,

    (void*)&tensor_map_o,
    (void*)&o_shape_0,
    (void*)&o_shape_1,
    (void*)&desc_o.unified_metadata_.strides[0],
    (void*)&o_stride_1,

    (void*)&N_CTX,
    (void*)&N_KTX,

    (void*)&N_KTX, // fake
    (void*)&N_KTX  // fake
  };


  
  auto stream = at::cuda::getCurrentCUDAStream();
  uint32_t thread_nums = get_default_thread_nums();
  
  // thread_nums *= 2;
  LOG(INFO) << "thread_nums: " << thread_nums;
  CUDA_CHECK(cuLaunchKernel(attn_fwd_shared_kernel,
                            grid_x, 
                            grid_y, 
                            1,  // grid dimensions
                            thread_nums, 
                            1, 
                            1,           // block dimensions (需要根据配置调整)
                            attn_fwd_shared_shared_mem_bytes,
                            stream,
                            args,
                            NULL));
  cudaStreamSynchronize(stream);

  
  auto o = desc_o.base;
  
  o = o.view({Z, H, N_CTX, HEAD_DIM});
  LOG(INFO) << "o.shape: " << o.sizes();
  // 对 o 进行切片: o[:, :, :N_CTX, :]
  auto o_sliced = o.slice(2, 0, N_CTX);  // 在第2维(seq_len)上从0切到N_CTX

  // 对 M 进行切片: M[:, :, :N_CTX]  
  auto M_sliced = M_shared.slice(2, 0, N_CTX);  // 

  return std::make_tuple(o_sliced, M_sliced);
}

void RecTritonKernel::attention_unshared_wrapper_forward(torch::Tensor q_unshared, 
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
                                                         uint32_t prompt_length) {
  LOG(INFO) << "desc_q: ";
  
  // uint64_t q_shapes[2] = {total_beams, num_heads * head_dim};
  // uint64_t q_strides[2] = {num_heads * head_dim * 2, 1};
  // uint32_t q_block_shape[2] = {2, 128};
  uint64_t q_shapes[2] = {num_heads * head_dim, total_beams};
  uint64_t q_strides[2] = {num_heads * head_dim * 2, num_heads * head_dim * total_beams * 2};
  uint32_t q_block_shape[2] = {128, 1};
  uint32_t q_elem_strides[2] = {1, 1};
  q_unshared = q_unshared.view({total_beams, num_heads * head_dim});

  TensorDescriptor desc_q{q_unshared, q_shapes, q_strides, q_block_shape, q_elem_strides, 2};

  LOG(INFO) << "desc_k: ";
  
  uint64_t k_shapes[2] = {kv_heads * head_dim, kv_total_size};
  uint64_t k_strides[2] = {k_shapes[0] * 2, k_shapes[0] * k_shapes[1] * 2};
  uint32_t k_block_shape[2] = {64, 64};
  
  uint32_t k_elem_strides[2] = {1, 1};
  unshared_k_cache = unshared_k_cache.view({kv_total_size, kv_heads * head_dim});
  TensorDescriptor desc_k{unshared_k_cache, k_shapes, k_strides, k_block_shape, k_elem_strides, 2};

  LOG(INFO) << "desc_v: ";
  
  uint64_t v_shapes[2] = {kv_heads * head_dim, kv_total_size};
  uint64_t v_strides[2] = {v_shapes[0] * 2, v_shapes[0] * v_shapes[1] * 2};
  uint32_t v_block_shape[2] = {64, 64};
  uint32_t v_elem_strides[2] = {1, 1};
  unshared_v_cache = unshared_v_cache.view({kv_total_size, kv_heads * head_dim});
  TensorDescriptor desc_v{unshared_v_cache, v_shapes, v_strides, v_block_shape, v_elem_strides, 2};

  LOG(INFO) << "desc_o: ";
  
  uint64_t o_shapes[2] = {num_heads * head_dim, total_beams};
  uint64_t o_strides[2] = {num_heads * head_dim * 2, num_heads * head_dim * total_beams * 2};
  uint32_t o_block_shape[2] = {128, 1};
  uint32_t o_elem_strides[2] = {1, 1};
  o_unshared = o_unshared.view({total_beams, num_heads * head_dim});
  TensorDescriptor desc_o{o_unshared, o_shapes, o_strides, o_block_shape, o_elem_strides, 2};

  attn_fwd_unshared(desc_q, 
                    desc_o, 
                    desc_k, 
                    desc_v, 
                    M_unshared, 
                    L_unshared, 
                    kv_heads, 
                    sm_scale, 
                    batch_size, 
                    total_beams, 
                    beam_size, 
                    max_decode_step, 
                    decode_step, 
                    num_heads, 
                    prompt_length);
}

void RecTritonKernel::attn_fwd_unshared(const TensorDescriptor& desc_q,
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
                                        uint32_t prompt_length) {

  // 获取kernel配置
  CUfunction attn_fwd_unshared_kernel;
  uint32_t attn_fwd_unshared_shared_mem_bytes = 0;
  
  switch (desc_unshared_k.base.scalar_type()) {
    case torch::kBFloat16: {
      const auto input_dim = 
        std::array<int, 3>{static_cast<int>(batch_size), 
                           static_cast<int>(beam_size), 
                           static_cast<int>(prompt_length)};
      const auto& kernel_config = bf16_attn_fwd_unshared_configs_.find_closest_index(
          bf16_attn_fwd_unshared_input_dim_array_, input_dim);
      attn_fwd_unshared_kernel = kernel_config.kernel;
      attn_fwd_unshared_shared_mem_bytes = kernel_config.shared_mem_bytes;
      break;
    }
    default:
      LOG(FATAL) << "Unsupported dtype for unshared attention: " << desc_unshared_k.base.scalar_type();
  }
  
  // 准备kernel参数
  LOG(INFO) << "tensor_map_q: ";
  alignas(64) CUtensorMap tensor_map_q;
  CUDA_CHECK(create_tensor_map(desc_q, &tensor_map_q, 0));
  LOG(INFO) << "tensor_map_k: ";
  alignas(64) CUtensorMap tensor_map_k;
  CUDA_CHECK(create_tensor_map(desc_unshared_k, &tensor_map_k, 3));
  LOG(INFO) << "tensor_map_v: ";
  alignas(64) CUtensorMap tensor_map_v;
  CUDA_CHECK(create_tensor_map(desc_unshared_v, &tensor_map_v, 3));
  LOG(INFO) << "tensor_map_o: ";
  alignas(64) CUtensorMap tensor_map_o;
  CUDA_CHECK(create_tensor_map(desc_o, &tensor_map_o, 0));

  LOG(INFO) << "after create_tensor_map.";
  
  CUdeviceptr M_unshared_ptr = reinterpret_cast<CUdeviceptr>(M_unshared.data_ptr());
  CUdeviceptr L_unshared_ptr = reinterpret_cast<CUdeviceptr>(L_unshared.data_ptr());
  
  // Grid dimensions: (beam_total_size, num_heads)
  uint32_t grid_x = total_beams;
  uint32_t grid_y = num_heads;

  uint32_t q_shape_0 = static_cast<uint64_t>(desc_q.unified_metadata_.shape[0]);
  uint32_t q_shape_1 = static_cast<uint64_t>(desc_q.unified_metadata_.shape[1]);
  uint32_t k_shape_0 = static_cast<uint64_t>(desc_unshared_k.unified_metadata_.shape[0]);
  uint32_t k_shape_1 = static_cast<uint64_t>(desc_unshared_k.unified_metadata_.shape[1]);
  uint32_t v_shape_0 = static_cast<uint64_t>(desc_unshared_v.unified_metadata_.shape[0]);
  uint32_t v_shape_1 = static_cast<uint64_t>(desc_unshared_v.unified_metadata_.shape[1]);
  uint32_t o_shape_0 = static_cast<uint64_t>(desc_o.unified_metadata_.shape[0]);
  uint32_t o_shape_1 = static_cast<uint64_t>(desc_o.unified_metadata_.shape[1]);
  
  uint32_t q_stride_1{1};
  uint32_t k_stride_1{1};
  uint32_t v_stride_1{1};
  uint32_t o_stride_1{1};
  
  void* args[] = {
    (void*)&sm_scale,
    (void*)&M_unshared_ptr,
    (void*)&L_unshared_ptr,

    (void*)&kv_heads,

    (void*)&tensor_map_q,
    (void*)&q_shape_0,
    (void*)&q_shape_1,
    (void*)&desc_q.unified_metadata_.strides[0],
    (void*)&q_stride_1,
    
    (void*)&tensor_map_k,
    (void*)&k_shape_0,
    (void*)&k_shape_1,
    (void*)&desc_unshared_k.unified_metadata_.strides[0],
    (void*)&k_stride_1,
    
    (void*)&tensor_map_v,
    (void*)&v_shape_0,
    (void*)&v_shape_1,
    (void*)&desc_unshared_v.unified_metadata_.strides[0],
    (void*)&v_stride_1,
    
    (void*)&tensor_map_o,
    (void*)&o_shape_0,
    (void*)&o_shape_1,
    (void*)&desc_o.unified_metadata_.strides[0],
    (void*)&o_stride_1,
    
    (void*)&total_beams,
    (void*)&beam_size,
    (void*)&max_decode_step,
    (void*)&current_step,

    (void*)&current_step,  // fake param
    (void*)&current_step       // fake param
  };
  
  auto stream = at::cuda::getCurrentCUDAStream();
  uint32_t thread_nums = get_default_thread_nums();
  
  CUDA_CHECK(cuLaunchKernel(attn_fwd_unshared_kernel,
                            grid_x, 
                            grid_y, 
                            1,  // grid dimensions
                            thread_nums, 
                            1, 
                            1,   // block dimensions
                            attn_fwd_unshared_shared_mem_bytes,
                            stream,
                            args,
                            NULL));
  cudaStreamSynchronize(stream);
}

void RecTritonKernel::combine_attention_kernel(torch::Tensor shared_out_2d, 
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
                                               uint32_t prompt_length) {
  // 获取kernel配置
  CUfunction combine_attention_kernel;
  uint32_t combine_attention_shared_mem_bytes = 0;
  
  switch (shared_out_2d.scalar_type()) {
    case torch::kBFloat16: {
      const auto input_dim = 
        std::array<int, 3>{static_cast<int>(batch_size), 
                           static_cast<int>(num_heads), 
                           static_cast<int>(prompt_length)};
      const auto& kernel_config = bf16_combine_attention_configs_.find_closest_index(
        bf16_combine_attention_input_dim_array_, input_dim);
      combine_attention_kernel = kernel_config.kernel;
      combine_attention_shared_mem_bytes = kernel_config.shared_mem_bytes;
      break;
    }
    default:
      LOG(FATAL) << "Unsupported dtype for combine attention: " << shared_out_2d.scalar_type();
  }
  
  // 准备kernel参数
  CUdeviceptr shared_out_2d_ptr = reinterpret_cast<CUdeviceptr>(shared_out_2d.data_ptr());
  CUdeviceptr unshared_out_2d_ptr = reinterpret_cast<CUdeviceptr>(unshared_out_2d.data_ptr());
  CUdeviceptr M_shared_ptr = reinterpret_cast<CUdeviceptr>(M_shared.data_ptr());
  CUdeviceptr L_shared_ptr = reinterpret_cast<CUdeviceptr>(L_shared.data_ptr());
  CUdeviceptr M_unshared_ptr = reinterpret_cast<CUdeviceptr>(M_unshared.data_ptr());
  CUdeviceptr L_unshared_ptr = reinterpret_cast<CUdeviceptr>(L_unshared.data_ptr());
  CUdeviceptr final_out_2d_ptr = reinterpret_cast<CUdeviceptr>(final_out_2d.data_ptr());
  
  // Grid dimensions based on BLOCK_BEAMS and BLOCK_H
  uint32_t BLOCK_BEAMS = 64;  // 从配置中获取，可以根据实际配置调整
  uint32_t BLOCK_H = 8;       // 从配置中获取，可以根据实际配置调整
  uint32_t grid_x = (total_beams + BLOCK_BEAMS - 1) / BLOCK_BEAMS;
  uint32_t grid_y = (num_heads + BLOCK_H - 1) / BLOCK_H;
  
  void* args[] = {
    (void*)&shared_out_2d_ptr,
    (void*)&unshared_out_2d_ptr,
    (void*)&M_shared_ptr,
    (void*)&L_shared_ptr,
    (void*)&M_unshared_ptr,
    (void*)&L_unshared_ptr,
    (void*)&final_out_2d_ptr,
    (void*)&total_beams,
    (void*)&num_heads,
    (void*)&num_heads,  // fake param
    (void*)&num_heads,  // fake param
  };
  
  auto stream = at::cuda::getCurrentCUDAStream();
  uint32_t thread_nums = get_default_thread_nums();
  
  CUDA_CHECK(cuLaunchKernel(combine_attention_kernel,
                            grid_x, 
                            grid_y, 
                            1,  // grid dimensions
                            thread_nums, 
                            1, 
                            1,   // block dimensions
                            combine_attention_shared_mem_bytes,
                            stream,
                            args,
                            NULL));
  cudaStreamSynchronize(stream);
}

torch::Tensor RecTritonKernel::xattention(torch::Tensor q, 
                                 torch::Tensor shared_k_cache, 
                                 torch::Tensor shared_v_cache, 
                                 torch::Tensor unshared_k_cache, 
                                 torch::Tensor unshared_v_cache, 
                                 uint32_t decode_step, 
                                 uint32_t beam_size, 
                                 float sm_scale, 
                                 uint32_t prompt_len) {
  LOG(INFO) << "q.shape: " << q.sizes();
  LOG(INFO) << "shared_k_cache.shape: " << shared_k_cache.sizes();
  LOG(INFO) << "shared_v_cache.shape: " << shared_v_cache.sizes();
  LOG(INFO) << "unshared_k_cache.shape: " << unshared_k_cache.sizes();
  LOG(INFO) << "unshared_v_cache.shape: " << unshared_v_cache.sizes();
  LOG(INFO) << "prompt_len: " << prompt_len;
  LOG(INFO) << "decode_step: " << decode_step;
  LOG(INFO) << "beam_size: " << beam_size;
  
  auto device = q.device();
  // ==================== 解析输入形状 ====================
  auto q_shape = q.sizes();
  uint32_t batch, num_heads, head_dim, total_beams;
  
  torch::Tensor q_shared;
  torch::Tensor q_unshared;

  if (q_shape.size() == 4) {
      // [batch, beamsize, num_heads, head_dim]
      batch = q_shape[0];
      beam_size = q_shape[1];
      num_heads = q_shape[2];
      head_dim = q_shape[3];
      total_beams = batch * beam_size;

      q_shared = q.permute({0, 2, 1, 3}).contiguous();
      q_unshared = q.reshape({total_beams, 1, num_heads, head_dim});

  } else if (q_shape.size() == 3) {
      total_beams = q_shape[0];
      num_heads = q_shape[1];
      head_dim = q_shape[2];
      batch = total_beams / beam_size;

      q_shared = q.reshape({batch, beam_size, num_heads, head_dim})
                 .permute({0, 2, 1, 3})
                 .contiguous();
      q_unshared = q.reshape({total_beams, 1, num_heads, head_dim});

  } else {
      LOG(FATAL) << "Unsupported query shape: " << q.sizes() << ". Expected 3D or 4D tensor.";
  }
  
  LOG(INFO) << "batch: " << batch;
  LOG(INFO) << "num_heads: " << num_heads;
  LOG(INFO) << "head_dim: " << head_dim;
  LOG(INFO) << "total_beams: " << total_beams;

  auto shared_k_shape = shared_k_cache.sizes();
  uint32_t kv_heads = shared_k_shape[1];

  LOG(INFO) << "kv_heads: " << kv_heads; 

  // float sm_scale = 0.08838834764831843;

  auto unshared_k_shape = unshared_k_cache.sizes();
  LOG(INFO) << "unshared_k_shape: " << unshared_k_shape;
  if (unshared_k_shape.size() == 5) {
    int64_t batch_cache = unshared_k_shape[0];
    int64_t beamsize_cache = unshared_k_shape[1];
    int64_t kv_heads_cache = unshared_k_shape[2];
    int64_t max_decode_step_cache = unshared_k_shape[3];
    int64_t head_dim_cache = unshared_k_shape[4];
    
    // Assert batch/beam size match
    assert(batch_cache == batch && beamsize_cache == beamsize && 
           "Batch/beam size mismatch");
    
    // Reshape from 5D to 4D
    unshared_k_cache = unshared_k_cache.reshape({
      batch * beam_size, kv_heads, max_decode_step_cache, head_dim_cache
    });
    unshared_v_cache = unshared_v_cache.reshape({
      batch * beam_size, kv_heads, max_decode_step_cache, head_dim_cache
    });
      
  } else if (unshared_k_shape.size() == 4) {
      int64_t beam_total_cache = unshared_k_shape[0];
      int64_t kv_heads_cache = unshared_k_shape[1];
      int64_t max_decode_step_cache = unshared_k_shape[2];
      int64_t head_dim_cache = unshared_k_shape[3];
      
      // Assert beam total match
      if (beam_total_cache != total_beams) {
          throw std::runtime_error(
              "Beam total mismatch: " + std::to_string(beam_total_cache) + 
              " vs " + std::to_string(total_beams)
          );
      }

      LOG(INFO) << "beam_total_cache: " << beam_total_cache;
      LOG(INFO) << "kv_heads_cache: " << kv_heads_cache;
      LOG(INFO) << "max_decode_step_cache: " << max_decode_step_cache;
      LOG(INFO) << "head_dim_cache: " << head_dim_cache;
      
  } else {
      std::string shape_str = "[";
      for (size_t i = 0; i < unshared_k_shape.size(); ++i) {
          shape_str += std::to_string(unshared_k_shape[i]);
          if (i < unshared_k_shape.size() - 1) shape_str += ", ";
      }
      shape_str += "]";
      
      throw std::runtime_error(
          "Unsupported unshared KV cache shape: " + shape_str + 
          ". Expected 4D or 5D."
      );
  }

  
  
  // ==================== Shared Attention ====================
  // if (num_heads != kv_heads) {
  //   int64_t repeat_factor = num_heads / kv_heads;
    
  //   shared_k_cache = shared_k_cache.unsqueeze(2).repeat({1, 1, repeat_factor, 1, 1});
    
  //   shared_k_cache = shared_k_cache.view({batch, num_heads, prompt_len, head_dim});
    
  //   shared_v_cache = shared_v_cache.unsqueeze(2).repeat({1, 1, repeat_factor, 1, 1});
  //   shared_v_cache = shared_v_cache.view({batch, num_heads, prompt_len, head_dim});
  // }

  
  
  LOG(INFO) << "q_shared.sizes(): " << q_shared.sizes();
  LOG(INFO) << "shared_k_cache.sizes(): " << shared_k_cache.sizes();
  LOG(INFO) << "shared_v_cache.sizes(): " << shared_v_cache.sizes();
  LOG(INFO) << "sm_scale: " << sm_scale;
  LOG(INFO) << "warp_specialize: " << warp_specialize;
  // Forward pass for shared attention
  // auto [o_shared, M_shared] = attention_shared_wrapper_forward(
  //     q, shared_k_cache, shared_v_cache, sm_scale, warp_specialize
  // );
  auto [o_shared, M_shared] = attention_shared_wrapper_forward(
                                  q_shared, shared_k_cache, shared_v_cache, sm_scale, 
                                  batch, beam_size, prompt_len
                              );
  LOG(INFO) << "o_shared.sizes(): " << o_shared.sizes();
  LOG(INFO) << "M_shared.sizes(): " << M_shared.sizes();
  // ==================== 2. Unshared Attention ====================
  // torch.empty_like(q_unshared)
  auto o_unshared = torch::empty_like(q_unshared);

  // torch.empty((num_heads, total_beams, 1), device=q.device, dtype=torch.float32)
  auto options = torch::TensorOptions()
      .device(q.device())
      .dtype(torch::kBFloat16);

  auto M_unshared = torch::empty({num_heads, total_beams, 1}, options);
  auto L_unshared = torch::empty({num_heads, total_beams, 1}, options);

  LOG(INFO) << "o_unshared.sizes(): " << o_unshared.sizes();
  LOG(INFO) << "M_unshared.sizes(): " << M_unshared.sizes();
  LOG(INFO) << "L_unshared.sizes(): " << L_unshared.sizes();
  LOG(INFO) << "unshared_k_cache.sizes(): " << unshared_k_cache.sizes();
  LOG(INFO) << "unshared_v_cache.sizes(): " << unshared_v_cache.sizes();

  uint32_t max_decode_step = 2;
  uint32_t kv_total_size = total_beams * max_decode_step;
  uint32_t N_KTX = max_decode_step;

  LOG(INFO) << "N_KTX: " << N_KTX;
  LOG(INFO) << "kv_total_size: " << kv_total_size;

  attention_unshared_wrapper_forward(q_unshared, 
                                     unshared_k_cache, 
                                     unshared_v_cache, 
                                     o_unshared, 
                                     M_unshared, 
                                     L_unshared, 
                                     total_beams, 
                                     num_heads, 
                                     head_dim, 
                                     kv_total_size, 
                                     sm_scale, 
                                     kv_heads, 
                                     beam_size, 
                                     max_decode_step, 
                                     decode_step, 
                                     N_KTX, 
                                     batch, 
                                     prompt_len);
  LOG(INFO) << "after attention_unshared_wrapper_forward.";

  LOG(INFO) << "o_unshared.sizes(): " << o_unshared.sizes() << ", scalar_type: " << o_unshared.scalar_type();
  LOG(INFO) << "M_unshared.sizes(): " << M_unshared.sizes() << ", scalar_type: " << M_unshared.scalar_type();
  LOG(INFO) << "L_unshared.sizes(): " << L_unshared.sizes() << ", scalar_type: " << L_unshared.scalar_type();


  // 重塑shared输出
  o_unshared = o_unshared.view({total_beams, num_heads, head_dim});

  M_unshared = M_unshared.view({num_heads, total_beams});
  L_unshared = L_unshared.view({num_heads, total_beams});
  
  // o_shared.permute(0, 2, 1, 3).reshape(total_beams, num_heads * head_dim)
  auto shared_out_2d = o_shared.permute({0, 2, 1, 3})
                              .reshape({total_beams, num_heads * head_dim});

  // o_unshared.reshape(total_beams, num_heads * head_dim)
  auto unshared_out_2d = o_unshared.reshape({total_beams, num_heads * head_dim});

  // torch.empty_like(shared_out_2d)
  auto final_out_2d = torch::empty_like(shared_out_2d);

  auto L_shared = 
    torch::ones_like(M_shared).reshape({num_heads, total_beams});

  LOG(INFO) << "o_unshared.sizes(): " << o_unshared.sizes() << ", scalar_type: " << o_unshared.scalar_type();
  LOG(INFO) << "M_unshared.sizes(): " << M_unshared.sizes() << ", scalar_type: " << M_unshared.scalar_type();
  LOG(INFO) << "L_unshared.sizes(): " << L_unshared.sizes() << ", scalar_type: " << L_unshared.scalar_type();
  LOG(INFO) << "shared_out_2d.sizes(): " << shared_out_2d.sizes() << ", scalar_type: " << shared_out_2d.scalar_type();
  LOG(INFO) << "unshared_out_2d.sizes(): " << unshared_out_2d.sizes() << ", scalar_type: " << unshared_out_2d.scalar_type();
  LOG(INFO) << "final_out_2d.sizes(): " << final_out_2d.sizes() << ", scalar_type: " << final_out_2d.scalar_type();
  LOG(INFO) << "L_shared.sizes(): " << L_shared.sizes() << ", scalar_type: " << L_shared.scalar_type();

  // // 调用combine kernel
  combine_attention_kernel(shared_out_2d, 
                           unshared_out_2d,
                           M_shared.view({num_heads, total_beams}), 
                           L_shared,
                           M_unshared, 
                           L_unshared, 
                           final_out_2d,
                           total_beams, 
                           num_heads, 
                           head_dim, 
                           batch, 
                           prompt_len);
  
  LOG(INFO) << "final_out_2d.shape: " << final_out_2d.sizes();
  return final_out_2d;
}

} // namespace xllm::kernel::cuda::triton