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
#include "dit_cache_impl.h"

namespace xllm {

class DiTCache {
 public:
  DiTCache() = default;
  ~DiTCache() = default;

  DiTCache(const DiTCache&) = delete;
  DiTCache& operator=(const DiTCache&) = delete;
  DiTCache(DiTCache&&) = delete;
  DiTCache& operator=(DiTCache&&) = delete;

  static DiTCache& get_instance() {
    static DiTCache ditcache;
    return ditcache;
  }

  bool init(const DiTCacheConfig& cfg);

  bool on_before_block(const CacheBlockIn& blockin);
  CacheBlockOut on_after_block(const CacheBlockIn& blockin);

  bool on_before_step(const CacheStepIn& stepin);
  CacheStepOut on_after_step(const CacheStepIn& stepin);

 private:
  std::unique_ptr<DitCacheImpl> active_cache_;
};

}  // namespace xllm
