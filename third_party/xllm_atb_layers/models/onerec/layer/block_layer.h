/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef ATB_SPEED_MODELS_ONEREC_BLOCK_LAYER_H
#define ATB_SPEED_MODELS_ONEREC_BLOCK_LAYER_H

#include <map>
#include <memory>
#include <vector>

#include "atb/atb_infer.h"
#include "atb_speed/base/hosttensor_binder.h"
#include "nlohmann/json.hpp"

namespace atb_speed {
namespace onerec {

// MLP linear layer indices (consistent with DeepSeek V2)
const uint64_t MLP_GATEUP_LINEAR_INDEX = 0;
const uint64_t MLP_DOWN_LINEAR_INDEX = 2;

// OneRec MoE configuration structure
struct OneRecMoEConfig {
  int moe_topk = 2;         // Number of experts to activate per token
  int moe_num_experts = 8;  // Total number of experts
  std::string moe_score_func =
      "softmax";                        // Gate function: "softmax" or "sigmoid"
  float moe_route_scale = 1.0f;         // Routing scale factor
  int moe_inter_dim = 1024;             // Expert intermediate dimension
  bool use_bf16 = true;                 // Use bfloat16 precision
  bool moe_use_shared_experts = false;  // Whether to use shared experts
  bool hasSharedExpertGate = false;     // Whether to use shared gate
  int moe_num_shared_experts = 0;       // Number of shared experts
};

struct BlockLayerParam {
  bool isFA = false;
  bool isPrefill = false;
  bool isBF16 = true;
  bool isPack = true;
  bool supportSwiGLU = false;
  bool supportLcoc = false;
  bool supportSpeculate = false;
  bool enableSplitFuse = false;
  bool supportLora = false;
  bool loraEnableGMM = false;
  bool enableLogN = false;
  bool kvQuant = false;
  bool enableIntraLayerAddNorm = false;
  bool enableInterLayerAddNorm = false;
  // OneRec position bias is now passed through attention_mask with ALIBI mask type
  // hasPositionBias parameter is no longer needed
  bool isDecoder = false;
  std::string backend = "lccl";
  int rank = 0;
  int worldSize = 1;
  int quantType = 0;
  int quantGroupSize = 64;
  int numAttentionHeadsPerRank = 0;
  int hiddenSizePerAttentionHead = 0;
  int numKeyValueHeadsPerRank = 0;
  float rmsNormEps = 0;
  int layerId = 0;
  int bs = 0;

  std::vector<int> seqLen;
  std::vector<int> tokenOffset;
  std::vector<int> packQuantType = {};
  // Two elements: first element represents QKV pack quantization type,
  // second element represents MLP pack quantization type
  // Seven elements: types for q, k, v, self attention out, gate, up, down
  // linear respectively
  std::vector<int> linearQuantType = {};
  std::vector<int> linearTransposeType;
  // Seven elements: description types for qkv, dense, gateup, down linear
  // respectively
  std::vector<int> linearDescs = {};
  // Four elements: quantization types for MoE router, gate, up, down linear
  // respectively
  std::vector<int> moeLinearQuantType = {};

  // OneRec specific parameters
  bool isOneRecEncoder = false;
  bool enableOneRecPrefillOnly = false;
  bool emptyCrossAttn = true;
  bool use_moe = false;

  // OneRec MoE configuration (only valid when use_moe is true)
  std::unique_ptr<OneRecMoEConfig> moe_config = nullptr;
};

atb::Status BlockLayer(const BlockLayerParam& param,
                       atb::Operation** operation);

// Function declarations for encoder and decoder self-attention
int64_t AddEncoderSelfAttention(atb::Node& selfAttentionNode,
                                const BlockLayerParam& param,
                                std::map<std::string, uint32_t>& tensorMap);
int64_t AddDecoderSelfAttention(atb::Node& selfAttentionNode,
                                const BlockLayerParam& param,
                                std::map<std::string, uint32_t>& tensorMap);

}  // namespace onerec
}  // namespace atb_speed
#endif  // ATB_SPEED_MODELS_ONEREC_BLOCK_LAYER_H
