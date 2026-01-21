#pragma once

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <mutex>

#ifndef check_cuda_error
#define check_cuda_error(call) C10_CUDA_CHECK(call)
#endif

namespace xllm::kernel::cuda {

inline int getMultiProcessorCount() {
  static int nSM{0};
  static std::once_flag flag;

  std::call_once(flag, []() {
    int deviceID{0};
    check_cuda_error(cudaGetDevice(&deviceID));
    check_cuda_error(
        cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, deviceID));
  });

  return nSM;
}

class NvtxRange {
 public:
  NvtxRange(const std::string& name) { nvtxRangePush(name.c_str()); }

  ~NvtxRange() { nvtxRangePop(); }
};

}  // namespace xllm::kernel::cuda
