include(cc_test)

if(NOT USE_NPU)
  return()
endif()

cc_test(
  NAME
    segmented_prefill_attention_test
  SRCS
    segmented_prefill_attention_test.cpp
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
    genrec_attention_test
  SRCS
    genrec_attention_test.cpp
  DEPS
    torch
    ascendcl
    nnopbase
    opapi
    GTest::gtest_main
    glog::glog
)

