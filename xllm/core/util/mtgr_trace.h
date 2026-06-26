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

#include <glog/logging.h>
#include <torch/torch.h>

#include <sstream>
#include <string>

#include "core/common/global_flags.h"

namespace xllm {

inline bool mtgr_trace_enabled(int level) {
  return FLAGS_mtgr_trace_log_level >= level;
}

inline std::string mtgr_trace_tensor_shape(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return "undefined";
  }
  std::ostringstream os;
  os << tensor.sizes();
  return os.str();
}

inline std::string mtgr_trace_tensor_desc(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return "undefined";
  }
  std::ostringstream os;
  os << tensor.sizes() << " dtype=" << tensor.dtype()
     << " device=" << tensor.device();
  return os.str();
}

}  // namespace xllm

#define MTGR_TRACE(level)                                                   \
  if (!::xllm::mtgr_trace_enabled(level)) {                                 \
  } else                                                                    \
    LOG(INFO) << "[MTGR_TRACE][L" << (level) << "] "
