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

#pragma once

#if defined(USE_CUDA)
#include <nvtx3/nvToolsExt.h>
#endif

#include "core/common/global_flags.h"

namespace xllm {

inline bool mtgr_nvtx_enabled(int level) {
  return FLAGS_mtgr_nvtx_level >= level;
}

class MtgrNvtxRange {
 public:
  MtgrNvtxRange(int level, const char* name)
      : enabled_(mtgr_nvtx_enabled(level)) {
#if defined(USE_CUDA)
    if (enabled_) {
      nvtxRangePushA(name);
    }
#else
    (void)name;
    enabled_ = false;
#endif
  }

  MtgrNvtxRange(const MtgrNvtxRange&) = delete;
  MtgrNvtxRange& operator=(const MtgrNvtxRange&) = delete;

  ~MtgrNvtxRange() {
#if defined(USE_CUDA)
    if (enabled_) {
      nvtxRangePop();
    }
#endif
  }

 private:
  bool enabled_ = false;
};

}  // namespace xllm

#define XLLM_MTGR_NVTX_CONCAT_IMPL(a, b) a##b
#define XLLM_MTGR_NVTX_CONCAT(a, b) XLLM_MTGR_NVTX_CONCAT_IMPL(a, b)
#define MTGR_NVTX_RANGE(level, name)                                      \
  [[maybe_unused]] ::xllm::MtgrNvtxRange XLLM_MTGR_NVTX_CONCAT(           \
      mtgr_nvtx_range_, __LINE__)((level), (name))
