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

#include <cstdint>
#include <string>

namespace xllm {
namespace util {

// RAII-style NVTX range marker for profiling GPU/CPU operations
// Usage:
//   {
//     NvtxRange range("my_operation");
//     // your code here
//   } // range automatically ends when going out of scope
class NvtxRange {
 public:
  // Constructor: starts an NVTX range with the given name
  explicit NvtxRange(const char* name);
  explicit NvtxRange(const std::string& name);

  // Constructor with color: starts an NVTX range with name and color
  // color format: 0xAARRGGBB (alpha, red, green, blue)
  NvtxRange(const char* name, uint32_t color);
  NvtxRange(const std::string& name, uint32_t color);

  // Destructor: ends the NVTX range
  ~NvtxRange();

  // Non-copyable and non-movable
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
  NvtxRange(NvtxRange&&) = delete;
  NvtxRange& operator=(NvtxRange&&) = delete;

 private:
  bool enabled_;
};

// Predefined colors for common operations
namespace NvtxColor {
constexpr uint32_t kRed = 0xFFFF0000;
constexpr uint32_t kGreen = 0xFF00FF00;
constexpr uint32_t kBlue = 0xFF0000FF;
constexpr uint32_t kYellow = 0xFFFFFF00;
constexpr uint32_t kCyan = 0xFF00FFFF;
constexpr uint32_t kMagenta = 0xFFFF00FF;
constexpr uint32_t kOrange = 0xFFFFA500;
constexpr uint32_t kPurple = 0xFF800080;
constexpr uint32_t kPink = 0xFFFFC0CB;
constexpr uint32_t kLightBlue = 0xFFADD8E6;
}  // namespace NvtxColor

}  // namespace util
}  // namespace xllm
