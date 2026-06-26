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

#pragma once

#include "api_service/api_service_impl.h"

namespace xllm {

template <typename T>
class ServiceImplFactory {
 public:
  static std::unique_ptr<T> create_service_impl(
      LLMMaster* master,
      const std::vector<std::string>& model_names) {
    auto service_impl = std::make_unique<T>(master, model_names);
    if (!service_impl) {
      LOG(ERROR) << "handler is nullptr";
    }
    return service_impl;
  }

  static std::unique_ptr<T> create_service_impl(
      const std::vector<std::string>& model_names,
      const std::vector<std::string>& model_versions) {
    if (model_names.size() != model_versions.size()) {
      LOG(ERROR)
          << "Models and model_versions size mismatch: model_names.size()="
          << model_names.size()
          << ", model_versions.size()=" << model_versions.size();
    }
    auto service_impl = std::make_unique<T>(model_names, model_versions);
    if (!service_impl) {
      LOG(ERROR) << "handler is nullptr";
    }
    return service_impl;
  }
};

}  // namespace xllm
