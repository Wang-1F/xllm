/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "nvtx_range.h"

#if defined(USE_CUDA)
#include <nvtx3/nvToolsExt.h>
#endif

namespace xllm {
namespace util {

NvtxRange::NvtxRange(const char* name) : enabled_(false) {
#if defined(USE_CUDA)
  nvtxRangePushA(name);
  enabled_ = true;
#endif
}

NvtxRange::NvtxRange(const std::string& name) : NvtxRange(name.c_str()) {}

NvtxRange::NvtxRange(const char* name, uint32_t color) : enabled_(false) {
#if defined(USE_CUDA)
  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = color;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = name;
  nvtxRangePushEx(&eventAttrib);
  enabled_ = true;
#endif
}

NvtxRange::NvtxRange(const std::string& name, uint32_t color)
    : NvtxRange(name.c_str(), color) {}

NvtxRange::~NvtxRange() {
#if defined(USE_CUDA)
  if (enabled_) {
    nvtxRangePop();
  }
#endif
}

}  // namespace util
}  // namespace xllm
