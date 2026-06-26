include(cc_test)

if(NOT USE_NPU)
  return()
endif()

cc_test(
  NAME
    genrec_attention_structured_reference_test
  SRCS
    genrec_attention_structured_reference_test.cpp
  INCLUDES
    /usr/local/Ascend/ascend-toolkit/latest/include
    /usr/local/Ascend/cann-8.5.0/aarch64-linux/include
    ${CMAKE_SOURCE_DIR}/third_party/xllm_ops/build/autogen
  DEPS
    torch
    ascendcl
    nnopbase
    opapi
    GTest::gtest_main
    glog::glog
)

cc_test(
  NAME
    mtgr_attention_e2e_precision_test
  SRCS
    mtgr_attention_e2e_precision_test.cpp
    mtgr_attention_test.cpp
    ../utils.cpp
  INCLUDES
    /usr/local/Ascend/ascend-toolkit/latest/include
    /usr/local/Ascend/cann-8.5.0/aarch64-linux/include
    ${CMAKE_SOURCE_DIR}/third_party/xllm_ops/build/autogen
  DEPS
    npu_torch_layers
    torch
    ascendcl
    nnopbase
    opapi
    GTest::gtest_main
    glog::glog
)

#
# MTGR mainline tests
# - mtgr_attention_e2e_precision_test: single-side regression through the
#   test-side MTGRAttentionTestImpl::forward used by perf
# - mtgr_attention_one_vs_multi_stage_perf_test: one_stage vs current
#   no_matched best-path sweep with compact CSV output
#
cc_test(
  NAME
    mtgr_attention_one_vs_multi_stage_perf_test
  SRCS
    mtgr_attention_one_vs_multi_stage_perf_test.cpp
    mtgr_attention_test.cpp
  INCLUDES
    /usr/local/Ascend/ascend-toolkit/latest/include
    /usr/local/Ascend/cann-8.5.0/aarch64-linux/include
    ${CMAKE_SOURCE_DIR}/third_party/xllm_ops/build/autogen
  DEPS
    npu_torch_layers
    torch
    ascendcl
    nnopbase
    opapi
    GTest::gtest_main
    glog::glog
)
