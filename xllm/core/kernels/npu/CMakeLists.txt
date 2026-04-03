include(cc_library)

add_subdirectory(xllm_ops)

cc_library(
  NAME
    npu_kernels
  HDRS
    genrec_attention.h
    npu_ops_api.h
  SRCS
    active.cpp
    attention.cpp
    fused_layernorm.cpp
    genrec_attention.cpp
    matmul.cpp
    npu_grouped_matmul.cpp
    npu_moe_gating_topk_softmax.cpp
    npu_moe_init_routing_v2.cpp
    npu_moe_token_unpermute.cpp
    rope.cpp
  DEPS
    :torch_npu_kernels
)

add_subdirectory(tests)
