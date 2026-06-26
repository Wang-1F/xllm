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
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "models/onerec/layer/block_layer.h"
#include "operations/fusion/attention/fusion_attention.h"
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "operations/fusion/mlp/mlp.h"
#include "operations/fusion/moe/moe_shared_expert.h"
#include "operations/fusion/moe/sparse_moe.h"
#include "operations/fusion/norm/norm_linear.h"

namespace atb_speed {
namespace onerec {

std::map<std::string, std::vector<std::string>>
GetOneRecLayerInTensorCandidates() {
  std::map<std::string, std::vector<std::string>>
      oneRecLayerInTensorCandidates = {
          {"base_weight",
           {// Layer norm weights
            "in_input_norm_weight",
            "in_input_norm_bias",
            "in_input_norm_new_weight",
            "in_input_norm_new_bias",
            // Self-attention QKV weights (packed) - OneRec specific names
            // mapped to standard names
            "in_weight_0",
            "in_bias_0",
            "in_descale_0",
            "in_offset_0",
            "in_scale_0",
            "in_compress_idx_0",
            "in_weight_1",
            "in_bias_1",
            "in_descale_1",
            "in_offset_1",
            "in_scale_1",
            "in_compress_idx_1",
            "in_weight_2",
            "in_bias_2",
            "in_descale_2",
            "in_offset_2",
            "in_scale_2",
            "in_compress_idx_2",
            // Self-attention output projection
            /*
            "in_dense_weight", "in_dense_bias", "in_dense_descale",
            "in_dense_offset", "in_dense_scale", "in_dense_compress_idx",
            */
            "in_weight_dense",
            "in_bias_dense",
            "in_descale_dense",
            "in_offset_dense",
            "in_scale_dense",
            "in_compress_idx_dense",
            "in_relative_attention_bias_weight",
            // Cross-attention layer norm
            "in_cross_attn_norm_weight",
            "in_cross_attn_norm_bias",
            "in_cross_attn_norm_new_weight",
            "in_cross_attn_norm_new_bias",
            // Cross-attention weights (for decoder only) - OneRec specific
            // names mapped to standard names
            "in_cross_weight_0",
            "in_cross_bias_0",
            "in_cross_descale_0",
            "in_cross_offset_0",
            "in_cross_scale_0",
            "in_cross_compress_idx_0",
            "in_cross_weight_1",
            "in_cross_bias_1",
            "in_cross_descale_1",
            "in_cross_offset_1",
            "in_cross_scale_1",
            "in_cross_compress_idx_1",
            "in_cross_weight_2",
            "in_cross_bias_2",
            "in_cross_descale_2",
            "in_cross_offset_2",
            "in_cross_scale_2",
            "in_cross_compress_idx_2",
            // Cross-attention output projection
            "in_cross_dense_weight",
            "in_cross_dense_bias",
            "in_cross_dense_descale",
            "in_cross_dense_offset",
            "in_cross_dense_scale",
            "in_cross_dense_compress_idx",

            // Post-attention layer norm
            "in_post_attn_norm_weight",
            "in_post_attn_norm_bias",
            "in_post_attn_norm_new_weight",
            "in_post_attn_norm_new_bias"}},
          {"mlp_weight",
           {// MLP weights
            "in_mlp_weight_0",
            "in_mlp_bias_0",
            "in_mlp_descale_0",
            "in_mlp_offset_0",
            "in_mlp_scale_0",
            "in_mlp_compress_idx_0",
            "in_mlp_weight_1",
            "in_mlp_bias_1",
            "in_mlp_descale_1",
            "in_mlp_offset_1",
            "in_mlp_scale_1",
            "in_mlp_compress_idx_1",
            "in_mlp_down_weight",
            "in_mlp_down_bias",
            "in_mlp_down_descale",
            "in_mlp_down_offset",
            "in_mlp_down_scale",
            "in_mlp_down_compress_idx"}},
          {"moe_weight",
           {// MoE gate weights
            "in_block_sparse_moe_gate_weight",
            "in_block_sparse_moe_gate_bias",
            "in_block_sparse_moe_gate_descale",
            "in_block_sparse_moe_gate_offset",
            "in_block_sparse_moe_gate_scale",
            "in_block_sparse_moe_gate_compress_idx",
            // Shared expert weights (OneRec MoE with shared experts)
            "in_shared_expert_gate_weight",
            "in_shared_expert_gate_bias",
            "in_shared_expert_gate_descale",
            "in_shared_expert_gate_offset",
            "in_shared_expert_gate_scale",
            "in_shared_expert_gate_compress_idx",
            "in_mlp_gateup_weight_shared_expert",
            "in_mlp_gateup_bias_shared_expert",
            "in_mlp_gateup_descale_shared_expert",
            "in_mlp_gateup_offset_shared_expert",
            "in_mlp_gateup_scale_shared_expert",
            "in_mlp_gateup_compress_idx_shared_expert",
            "in_mlp_down_weight_shared_expert",
            "in_mlp_down_bias_shared_expert",
            "in_mlp_down_descale_shared_expert",
            "in_mlp_down_offset_shared_expert",
            "in_mlp_down_scale_shared_expert",
            "in_mlp_down_compress_idx_shared_expert",
            // MoE expert weights
            "in_mlp_gateup_weight_expert",
            "in_mlp_gateup_bias_expert",
            "in_mlp_gateup_descale_expert",
            "in_mlp_gateup_offset_expert",
            "in_mlp_gateup_scale_expert",
            "in_mlp_gateup_compress_idx_expert",
            "in_mlp_down_weight_expert",
            "in_mlp_down_bias_expert",
            "in_mlp_down_descale_expert",
            "in_mlp_down_offset_expert",
            "in_mlp_down_scale_expert",
            "in_mlp_down_compress_idx_expert",
            // MoE routing tensors
            "in_expert_array",
            "in_expert_group",
            "in_one_hot",
            "in_zero_hot"}},
          {"kv_quant",
           {"in_k_quant_scale",
            "in_k_dequant_scale",
            "in_v_quant_scale",
            "in_v_dequant_scale",
            "in_k_quant_offset",
            "in_k_dequant_offset",
            "in_v_quant_offset",
            "in_v_dequant_offset"}},
          {"default",
           {"in_input",  // shape: FA: [batchSize, seqLen, hiddenSize] PA:
                         // [seqLen, hiddenSize] (OneRec: in_hidden_states ->
                         // in_input)
            "in_attention_mask",
            "in_k_cache",
            "in_v_cache",
            "in_seq_len",
            "in_token_offset",
            "in_layer_id",
            "in_block_tables",
            "in_slots",
            "in_encoder_output"}},
          {"onerec_encoder_input",
           {
               "in_input",
               "in_attention_mask",
               "in_token_offset",
               "in_layer_id",
               "in_seq_len",
           }},
          {"cross_attn",
           {"in_cross_attn_seq_len",
            "in_cross_attn_block_tables",
            "in_cross_attn_slots",
            "cross_kv_len"}},
          {"q_len", {"in_q_len"}},
          {"logn_enable", {"kv_cache_idx"}},
          {"lora_common", {"in_seq_len_cum_sum"}},
          {"lora_attn",
           {"in_lora_a_0",
            "in_lora_b_0",
            "in_lora_a_1",
            "in_lora_b_1",
            "in_lora_a_2",
            "in_lora_b_2",
            "in_dense_lora_a",
            "in_dense_lora_b"}},
          {"lora_mlp",
           {"in_mlp_lora_a_0",
            "in_mlp_lora_b_0",
            "in_mlp_lora_a_1",
            "in_mlp_lora_b_1",
            "in_mlp_down_lora_a",
            "in_mlp_down_lora_b"}}};
  return oneRecLayerInTensorCandidates;
}

std::map<std::string, std::vector<std::string>>
GetOneRecLayerIntermediateTensorCandidates() {
  std::map<std::string, std::vector<std::string>>
      oneRecLayerIntermediateTensorCandidates = {
          {"default",
           {"intermediate_self_attn_out", "intermediate_self_residual_out"}},
          {"decoder",
           {"intermediate_cross_attn_out", "intermediate_cross_residual_out"}},
          {"moe_input_norm", {"intermediate_selfattention_norm_out"}},
          {"moe_weight", {"intermediate_moe_out"}},
          {"mlp_weight", {"intermediate_mlp_out"}},
          {"shared_expert",
           {"intermediate_shared_expert_out",
            "intermediate_shared_expert_add_out"}},

      };
  return oneRecLayerIntermediateTensorCandidates;
}

std::map<std::string, uint32_t> ConstructTensorMap(
    const BlockLayerParam& param,
    uint32_t& inTensorNum,
    uint32_t& outTensorNum,
    uint32_t& internalTensorNum) {
  auto oneRecLayerInTensorCandidates = GetOneRecLayerInTensorCandidates();
  auto oneRecLayerIntermediateTensorCandidates =
      GetOneRecLayerIntermediateTensorCandidates();

  std::vector<std::string> inTensorList = {};
  std::vector<std::string> intermediateTensorList = {};
  std::vector<std::string> outTensorList = {"out"};

  // Add base weights (attention and layer normalization weights)
  atb_speed::common::AddTensorToList(
      oneRecLayerInTensorCandidates, "base_weight", inTensorList);

  // Add MLP weights based on mode
  if (param.use_moe) {
    // Traditional mode: add traditional MLP weights
    atb_speed::common::AddTensorToList(
        oneRecLayerInTensorCandidates, "moe_weight", inTensorList);
  } else {
    atb_speed::common::AddTensorToList(
        oneRecLayerInTensorCandidates, "mlp_weight", inTensorList);
  }

  // Add KV cache int8 feature tensors
  if (param.kvQuant) {
    atb_speed::common::AddTensorToList(
        oneRecLayerInTensorCandidates, "kv_quant", inTensorList);
  }
  // use_moe only supports decoder, not encoder
  if (param.use_moe) {
    // MoE mode only supports decoder
    atb_speed::common::AddTensorToList(
        oneRecLayerInTensorCandidates, "default", inTensorList);
  } else {
    if (param.isOneRecEncoder) {
      atb_speed::common::AddTensorToList(oneRecLayerInTensorCandidates,
                                         "onerec_encoder_input",
                                         inTensorList);
    } else {
      atb_speed::common::AddTensorToList(
          oneRecLayerInTensorCandidates, "default", inTensorList);
    }
  }
  atb_speed::common::AddTensorToList(oneRecLayerIntermediateTensorCandidates,
                                     "default",
                                     intermediateTensorList);
  // Add OneRec decoder cross-attention related tensors
  // use_moe forces decoder mode
  if (param.isDecoder) {
    atb_speed::common::AddTensorToList(
        oneRecLayerInTensorCandidates, "cross_attn", inTensorList);
    atb_speed::common::AddTensorToList(
        oneRecLayerIntermediateTensorCandidates,
        "decoder",
        intermediateTensorList);
  }
  if (param.use_moe) {
    atb_speed::common::AddTensorToList(
        oneRecLayerIntermediateTensorCandidates,
        "moe_weight",
        intermediateTensorList);
    atb_speed::common::AddTensorToList(
        oneRecLayerIntermediateTensorCandidates,
        "moe_input_norm",
        intermediateTensorList);
    // Add shared expert tensors if enabled
    if (param.moe_config && param.moe_config->moe_use_shared_experts) {
      atb_speed::common::AddTensorToList(
          oneRecLayerIntermediateTensorCandidates,
          "shared_expert",
          intermediateTensorList);
    }
  } else {
    atb_speed::common::AddTensorToList(
        oneRecLayerIntermediateTensorCandidates,
        "mlp_weight",
        intermediateTensorList);
  }

  // Add parallel decoding feature or SplitFuse tensors
  if (param.supportSpeculate || param.enableSplitFuse) {
    atb_speed::common::AddTensorToList(
        oneRecLayerInTensorCandidates, "q_len", inTensorList);
  }

  // Add lora feature tensors
  if (param.supportLora) {
    atb_speed::common::AddTensorToList(
        oneRecLayerInTensorCandidates, "lora_common", inTensorList);
    atb_speed::common::AddTensorToList(
        oneRecLayerInTensorCandidates, "lora_attn", inTensorList);
    atb_speed::common::AddTensorToList(
        oneRecLayerInTensorCandidates, "lora_mlp", inTensorList);
  }

  // Add logn feature tensors
  if (param.enableLogN) {
    atb_speed::common::AddTensorToList(
        oneRecLayerInTensorCandidates, "logn_enable", inTensorList);
  }

  inTensorNum = inTensorList.size();

  outTensorNum = outTensorList.size();
  internalTensorNum = intermediateTensorList.size();

  return atb_speed::common::GetTensorMap(
      inTensorList, outTensorList, intermediateTensorList);
}

void SetSelfAttentionParamPart(
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam>&
        fusionAttentionParam,
    const BlockLayerParam& param) {
  fusionAttentionParam.isGroupedQueryAttention =
      param.numAttentionHeadsPerRank != param.numKeyValueHeadsPerRank;
  fusionAttentionParam.isBF16 = param.isBF16;
  fusionAttentionParam.qkvHasBias =
      false;  // OneRec typically doesn't use bias in attention
  fusionAttentionParam.layerLinearQuantType = param.linearQuantType;
  fusionAttentionParam.layerLinearTransposeType = param.linearTransposeType;
  fusionAttentionParam.layerLinearDescs =
      param.linearDescs;  // Enable QKV projection
  fusionAttentionParam.packQuantType = param.packQuantType.at(0);
  fusionAttentionParam.quantGroupSize = param.quantGroupSize;
  fusionAttentionParam.supportLcoc = param.supportLcoc;
  fusionAttentionParam.supportLora = param.supportLora;
  fusionAttentionParam.loraEnableGMM = param.loraEnableGMM;
  fusionAttentionParam.isOneRecEncoder =
      !param.isDecoder;  // Set OneRec encoder flag: non-decoder is encoder
  fusionAttentionParam.isOneRecDecoder = param.isDecoder;
  *fusionAttentionParam.bs = param.bs;
  fusionAttentionParam.splitWithStride = false;

  // OneRec uses RMSNorm
  atb::infer::RmsNormParam attenRmsNormParam;
  attenRmsNormParam.layerType =
      atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
  attenRmsNormParam.normParam.epsilon = param.rmsNormEps;
  fusionAttentionParam.normParamType = attenRmsNormParam;

  atb::infer::RmsNormParam attenRmsNormQuantParam;
  attenRmsNormQuantParam.layerType =
      atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
  attenRmsNormQuantParam.normParam.epsilon = param.rmsNormEps;
  attenRmsNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
  fusionAttentionParam.normQuantParamType = attenRmsNormQuantParam;

  // OneRec uses relative position bias instead of RoPE
  fusionAttentionParam.rotaryType = atb_speed::common::RotaryType::NO_ROTARY;

  // OneRec position bias is now passed through attention_mask with ALIBI
  // mask type hasPositionBias parameter is no longer needed  // Disable
  // separate position bias input

  // OneRec KV cache update logic:
  // - OneRec Encoder: false (bidirectional attention, no KV cache needed)
  // - OneRec Decoder self-attention: true (always update for autoregressive
  // generation)
  fusionAttentionParam.needUpdateKVCache = param.isDecoder;
  if (param.enableOneRecPrefillOnly) {
    fusionAttentionParam.needUpdateKVCache = false;
  }

  if (param.enableLogN) {
    fusionAttentionParam.pageAttentionParam.scaleType =
        atb::infer::PagedAttentionParam::SCALE_TYPE_LOGN;
  }
}

void SetSelfAttentionParam(atb_speed::common::FusionAttentionParam<
                               atb::infer::RmsNormParam>& fusionAttentionParam,
                           const BlockLayerParam& param) {
  SetSelfAttentionParamPart(fusionAttentionParam, param);
  fusionAttentionParam.isFA = param.isFA;
  fusionAttentionParam.isPrefill = param.isPrefill;
  fusionAttentionParam.enableSplitFuse = param.enableSplitFuse;
  fusionAttentionParam.headDim = param.hiddenSizePerAttentionHead;
  fusionAttentionParam.selfAttentionParam.headNum =
      param.numAttentionHeadsPerRank;
  fusionAttentionParam.selfAttentionParam.kvHeadNum =
      param.numKeyValueHeadsPerRank;

  if (param.hiddenSizePerAttentionHead == 0) {
    std::stringstream ss;
    ss << "Cannot be divided by zero. Param hiddenSizePerAttentionHead is zero!"
       << std::endl;
    throw std::runtime_error(ss.str());
  }
  fusionAttentionParam.selfAttentionParam.qkScale = 1.0;

  // OneRec attention mask handling based on encoder/decoder type and stage
  bool is_oneRec_prefill = param.isPrefill;

  if (param.isDecoder) {
    // OneRec Decoder self-attention: always uses causal mask
    fusionAttentionParam.selfAttentionParam.isTriuMask = 0;
    if (param.isFA) {
      fusionAttentionParam.selfAttentionParam.calcType =
          is_oneRec_prefill
              ? atb::infer::SelfAttentionParam::CalcType::ENCODER
              : atb::infer::SelfAttentionParam::CalcType::DECODER;
    } else {
      fusionAttentionParam.selfAttentionParam.calcType =
          atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
    }
  } else {
    // OneRec Encoder self-attention: bidirectional (no causal mask)
    fusionAttentionParam.selfAttentionParam.isTriuMask = 0;
    fusionAttentionParam.selfAttentionParam.calcType =
        atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
  }

  // OneRec uses position bias through attention mask with ALIBI mask type
  fusionAttentionParam.selfAttentionParam.maskType =
      atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_ALIBI;

  fusionAttentionParam.pageAttentionParam.headNum =
      param.numAttentionHeadsPerRank;
  fusionAttentionParam.pageAttentionParam.kvHeadNum =
      param.numKeyValueHeadsPerRank;
  fusionAttentionParam.pageAttentionParam.qkScale = 1.0;

  if (param.supportSpeculate) {
    fusionAttentionParam.pageAttentionParam.calcType =
        atb::infer::PagedAttentionParam::CalcType::CALC_TYPE_SPEC;
    fusionAttentionParam.pageAttentionParam.maskType =
        atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_SPEC;
  }
  if (param.enableSplitFuse) {
    fusionAttentionParam.pageAttentionParam.calcType =
        atb::infer::PagedAttentionParam::CalcType::CALC_TYPE_SPEC;
    fusionAttentionParam.pageAttentionParam.maskType =
        atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_ALIBI;
  }
  fusionAttentionParam.selfOutLinearTensorParallelInfo = {
      param.rank, param.worldSize, param.backend};

  if (param.kvQuant) {
    fusionAttentionParam.pageAttentionParam.quantType =
        atb::infer::PagedAttentionParam::TYPE_DEQUANT_FUSION;
    fusionAttentionParam.pageAttentionParam.maskType =
        atb::infer::PagedAttentionParam::UNDEFINED;
    fusionAttentionParam.pageAttentionParam.hasQuantOffset = true;
  }
}

void SetCrossAttentionParam(atb_speed::common::FusionAttentionParam<
                                atb::infer::RmsNormParam>& fusionAttentionParam,
                            const BlockLayerParam& param) {
  SetSelfAttentionParamPart(fusionAttentionParam, param);
  fusionAttentionParam.attnBackend = atb_speed::common::OpBackend::ACLNN;
  fusionAttentionParam.isFA = param.isFA;
  fusionAttentionParam.isPrefill = param.isPrefill;
  fusionAttentionParam.enableSplitFuse = param.enableSplitFuse;
  fusionAttentionParam.headDim = param.hiddenSizePerAttentionHead;
  fusionAttentionParam.selfAttentionParam.headNum =
      param.numAttentionHeadsPerRank;
  fusionAttentionParam.selfAttentionParam.kvHeadNum =
      param.numKeyValueHeadsPerRank;
  fusionAttentionParam.aclnnFusedInferAttnParam.numHeads =
      param.numAttentionHeadsPerRank;
  fusionAttentionParam.aclnnFusedInferAttnParam.numKeyValueHeads =
      param.numKeyValueHeadsPerRank;
  fusionAttentionParam.aclnnFusedInferAttnParam.inputLayout = "BSND";

  fusionAttentionParam.isOneRecEncoder = false;
  *fusionAttentionParam.bs = param.bs;

  if (param.hiddenSizePerAttentionHead == 0) {
    std::stringstream ss;
    ss << "Cannot be divided by zero. Param hiddenSizePerAttentionHead is zero!"
       << std::endl;
    throw std::runtime_error(ss.str());
  }
  fusionAttentionParam.selfAttentionParam.qkScale = 1.0;

  // Cross-attention doesn't use causal mask
  fusionAttentionParam.selfAttentionParam.isTriuMask = 0;
  fusionAttentionParam.selfAttentionParam.calcType =
      atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
  fusionAttentionParam.selfAttentionParam.maskType =
      atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_UNDEFINED;

  fusionAttentionParam.pageAttentionParam.headNum =
      param.numAttentionHeadsPerRank;
  fusionAttentionParam.pageAttentionParam.kvHeadNum =
      param.numKeyValueHeadsPerRank;
  fusionAttentionParam.pageAttentionParam.qkScale = 1.0;
  // Cross-attention does not use position bias (handled via attention_mask for
  // self-attention only)

  // Cross-attention KV cache update logic based on OneRec stage:
  // - OneRec Decoder Prefill: true (update K/V cache with encoder output)
  // - OneRec Decoder Decode: false (use existing K/V cache, no update
  // needed)
  bool is_oneRec_prefill = param.isPrefill;
  // OneRec cross-attention: update KV cache only during prefill stage
  // During decode stage, reuse the encoder output KV cache from prefill
  // Special case: if enableOneRecPrefillOnly is true and emptyCrossAttn is
  // false, don't update KV cache (subsequent prefill)
  if (param.enableOneRecPrefillOnly) {
    fusionAttentionParam.needUpdateKVCache = false;
  } else {
    fusionAttentionParam.needUpdateKVCache = is_oneRec_prefill;
  }

  fusionAttentionParam.selfOutLinearTensorParallelInfo = {
      param.rank, param.worldSize, param.backend};

  if (param.kvQuant) {
    fusionAttentionParam.pageAttentionParam.quantType =
        atb::infer::PagedAttentionParam::TYPE_DEQUANT_FUSION;
    fusionAttentionParam.pageAttentionParam.maskType =
        atb::infer::PagedAttentionParam::UNDEFINED;
    fusionAttentionParam.pageAttentionParam.hasQuantOffset = true;
  }
}

int64_t AddEncoderSelfAttention(atb::Node& selfAttentionNode,
                                const BlockLayerParam& param,
                                std::map<std::string, uint32_t>& tensorMap) {
  // Encoder self-attention (no KV cache, no slots, no block tables)
  atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam>
      fusionAttentionParam;
  SetSelfAttentionParam(fusionAttentionParam, param);
  CHECK_OPERATION_STATUS_RETURN(
      Attention(fusionAttentionParam, &selfAttentionNode.operation));
  // OneRec Encoder tensor list - must match
  // GetOneRecLayerInTensorCandidates
  std::vector<std::string> selfAttnInTensorNames = {"in_input",
                                                    "in_input_norm_weight",
                                                    "in_input_norm_bias",
                                                    "in_input_norm_new_weight",
                                                    "in_input_norm_new_bias",
                                                    "in_weight_0",
                                                    "in_scale_0",
                                                    "in_offset_0",
                                                    "in_descale_0",
                                                    "in_bias_0",
                                                    "in_compress_idx_0",
                                                    "in_weight_1",
                                                    "in_scale_1",
                                                    "in_offset_1",
                                                    "in_descale_1",
                                                    "in_bias_1",
                                                    "in_compress_idx_1",
                                                    "in_weight_2",
                                                    "in_scale_2",
                                                    "in_offset_2",
                                                    "in_descale_2",
                                                    "in_bias_2",
                                                    "in_compress_idx_2",
                                                    "in_attention_mask",
                                                    "in_seq_len",
                                                    "in_weight_dense",
                                                    "in_scale_dense",
                                                    "in_offset_dense",
                                                    "in_descale_dense",
                                                    "in_bias_dense",
                                                    "in_compress_idx_dense"};

  if (param.supportSpeculate || param.enableSplitFuse) {
    selfAttnInTensorNames.push_back("in_q_len");
  }

  auto oneRecLayerInTensorCandidates = GetOneRecLayerInTensorCandidates();
  if (param.supportLora) {
    selfAttnInTensorNames.push_back("in_seq_len_cum_sum");
    for (std::string tensor :
         oneRecLayerInTensorCandidates.at("lora_attn")) {
      selfAttnInTensorNames.push_back(tensor);
    }
  }

  if (param.enableLogN) {
    selfAttnInTensorNames.push_back("kv_cache_idx");
  }

  selfAttentionNode.inTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, selfAttnInTensorNames);
  selfAttentionNode.outTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_self_attn_out"});
  return atb::NO_ERROR;
}

int64_t AddDecoderSelfAttention(atb::Node& selfAttentionNode,
                                const BlockLayerParam& param,
                                std::map<std::string, uint32_t>& tensorMap) {
  // Decoder self-attention (with KV cache, slots, block tables)
  atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam>
      fusionAttentionParam;
  SetSelfAttentionParam(fusionAttentionParam, param);
  CHECK_OPERATION_STATUS_RETURN(
      Attention(fusionAttentionParam, &selfAttentionNode.operation));
  std::vector<std::string> selfAttnInTensorNames = {
      "in_input",
      "in_input_norm_weight",
      "in_input_norm_bias",
      "in_input_norm_new_weight",
      "in_input_norm_new_bias",
      "in_weight_0",
      "in_scale_0",
      "in_offset_0",
      "in_descale_0",
      "in_bias_0",
      "in_compress_idx_0",
      "in_weight_1",
      "in_scale_1",
      "in_offset_1",
      "in_descale_1",
      "in_bias_1",
      "in_compress_idx_1",
      "in_weight_2",
      "in_scale_2",
      "in_offset_2",
      "in_descale_2",
      "in_bias_2",
      "in_compress_idx_2",
      "in_weight_dense",
      "in_scale_dense",
      "in_offset_dense",
      "in_descale_dense",
      "in_bias_dense",
      "in_compress_idx_dense",
      // Order must match fusion_attention.cpp "default" section:
      // "in_input", "in_attention_mask", "in_k_cache", "in_v_cache",
      // "in_seq_len", "in_token_offset", "in_layer_id", "in_block_tables",
      // "in_slots", "in_encoder_output"
      "in_attention_mask",
      "in_k_cache",
      "in_v_cache",
      "in_seq_len",
      "in_token_offset",
      "in_layer_id",
      "in_block_tables",
      "in_slots",
  };

  if (param.supportSpeculate || param.enableSplitFuse) {
    selfAttnInTensorNames.push_back("in_q_len");
  }

  auto oneRecLayerInTensorCandidates = GetOneRecLayerInTensorCandidates();
  if (param.kvQuant) {
    for (std::string tensor : oneRecLayerInTensorCandidates.at("kv_quant")) {
      selfAttnInTensorNames.push_back(tensor);
    }
  }
  if (param.supportLora) {
    selfAttnInTensorNames.push_back("in_seq_len_cum_sum");
    for (std::string tensor :
         oneRecLayerInTensorCandidates.at("lora_attn")) {
      selfAttnInTensorNames.push_back(tensor);
    }
  }

  if (param.enableLogN) {
    selfAttnInTensorNames.push_back("kv_cache_idx");
  }

  selfAttentionNode.inTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, selfAttnInTensorNames);
  selfAttentionNode.outTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_self_attn_out"});
  return atb::NO_ERROR;
}

int64_t AddCrossAttention(atb::Node& crossAttentionNode,
                          const BlockLayerParam& param,
                          std::map<std::string, uint32_t>& tensorMap) {
  // Cross-attention (for decoder only)
  atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam>
      fusionAttentionParam;
  SetCrossAttentionParam(fusionAttentionParam, param);
  CHECK_OPERATION_STATUS_RETURN(
      CrossAttention(fusionAttentionParam, &crossAttentionNode.operation));

  // Cross-attention inputs: self-attention output as query, weight tensors, and stage-specific tensors.
  std::vector<std::string> crossAttnInTensorNames = {
      "intermediate_self_residual_out",  // Use the self-attention output as the cross-attention query.
      "in_cross_attn_norm_weight",
      "in_cross_attn_norm_bias",
      "in_cross_attn_norm_new_weight",
      "in_cross_attn_norm_new_bias",
      "in_cross_weight_0",
      "in_cross_bias_0",
      "in_cross_descale_0",
      "in_cross_offset_0",
      "in_cross_scale_0",
      "in_cross_compress_idx_0",
      "in_cross_weight_1",
      "in_cross_bias_1",
      "in_cross_descale_1",
      "in_cross_offset_1",
      "in_cross_scale_1",
      "in_cross_compress_idx_1",
      "in_cross_weight_2",
      "in_cross_bias_2",
      "in_cross_descale_2",
      "in_cross_offset_2",
      "in_cross_scale_2",
      "in_cross_compress_idx_2",
      "in_cross_dense_weight",
      "in_cross_dense_scale",
      "in_cross_dense_offset",
      "in_cross_dense_descale",
      "in_cross_dense_bias",
      "in_cross_dense_compress_idx",
      "in_seq_len",
      "in_layer_id",
  };

  // Append stage-specific tensors based on prefill or decode mode.
  if (param.isPrefill) {
    // Prefill stage: use encoder_output to build K/V and use slots to update the KV cache.
    crossAttnInTensorNames.push_back("in_encoder_output");
    crossAttnInTensorNames.push_back("in_k_cache");
    crossAttnInTensorNames.push_back("in_v_cache");
    crossAttnInTensorNames.push_back("in_cross_attn_slots");
    crossAttnInTensorNames.push_back("in_cross_attn_seq_len");
    crossAttnInTensorNames.push_back("in_cross_attn_block_tables");
    crossAttnInTensorNames.push_back("cross_kv_len");
  } else {
    // Decode stage: read K/V directly from the KV cache without encoder_output or slots.
    crossAttnInTensorNames.push_back("in_k_cache");
    crossAttnInTensorNames.push_back("in_v_cache");
    crossAttnInTensorNames.push_back("in_cross_attn_seq_len");
    crossAttnInTensorNames.push_back("in_cross_attn_block_tables");
    crossAttnInTensorNames.push_back("cross_kv_len");
  }

  crossAttentionNode.inTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, crossAttnInTensorNames);
  crossAttentionNode.outTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_cross_attn_out"});
  return atb::NO_ERROR;
}

void SetMoeParam(atb_speed::common::SparseMoeParam& sparseMoeParam,
                 const BlockLayerParam& param) {
  if (!param.use_moe || !param.moe_config) {
    ATB_SPEED_LOG_ERROR("MoE configuration is missing when use_moe is true");
    return;
  }

  const auto& moe_config = *param.moe_config;

  // Basic routing parameters from OneRecMoEConfig
  sparseMoeParam.axes = {1};  // Apply softmax/sigmoid on expert dimension
  sparseMoeParam.num = {moe_config.moe_topk};
  sparseMoeParam.topkGroups = {moe_config.moe_topk};

  // Expert configuration
  sparseMoeParam.numOfExperts = moe_config.moe_num_experts;
  sparseMoeParam.numOfDeviceExperts = moe_config.moe_num_experts;
  sparseMoeParam.numOfGroups = 1;

  // Routing method based on gate function
  if (moe_config.moe_score_func == "sigmoid") {
    sparseMoeParam.routingMethod = "noAuxTc";
    sparseMoeParam.processLogits = "normalization";
  } else {
    sparseMoeParam.routingMethod = "softMaxTopK";
    sparseMoeParam.processLogits = "scaling";
  }

  // Routing scale factor
  sparseMoeParam.routedScalingFactor = moe_config.moe_route_scale;

  // Data type and optimization settings
  sparseMoeParam.isBF16 = moe_config.use_bf16;
  sparseMoeParam.supportSwiGLU = true;
  sparseMoeParam.transpose = true;
  sparseMoeParam.gateUpTransposeB = true;
  sparseMoeParam.downTransposeB = true;

  // Optimization flags
  sparseMoeParam.enableFusedRouting = true;
  sparseMoeParam.enableFusedTopk = false;
  sparseMoeParam.enableCVOverlap = false;
  sparseMoeParam.enableMoeParallel = false;
  sparseMoeParam.enableGMMSwigluQuant = false;
  sparseMoeParam.enableLoadBalance = false;

  // Communication settings
  sparseMoeParam.backend = param.backend;
  sparseMoeParam.hasMoeEp = false;
  sparseMoeParam.moeEpSize = 1;
  sparseMoeParam.moeEpRank = 0;

  // Bias and quantization settings
  sparseMoeParam.hasBias = false;
  sparseMoeParam.rounterHasBias = false;
  sparseMoeParam.packQuantType = atb_speed::common::PackQuantType::ALL_FP;
  sparseMoeParam.enableInitQuant = false;
  sparseMoeParam.enableSwigluQuant = false;
  sparseMoeParam.enableMoeDistribute = false;
  sparseMoeParam.routedScalingFactor = moe_config.moe_route_scale;

  // Initialize moeLinearQuantType vector with default values
  // This is critical to prevent core dump when accessing
  // moeLinearQuantType[ROUTER_IDX] 4 components: ROUTER, GATE, UP, DOWN
  sparseMoeParam.moeLinearQuantType = param.moeLinearQuantType;
}

void SetMlpParam(
    atb_speed::common::MlpParam<atb::infer::RmsNormParam>& mlpParam,
    const BlockLayerParam& param) {
  mlpParam.isBF16 = param.isBF16;
  mlpParam.layerLinearQuantType = param.linearQuantType;
  mlpParam.layerLinearTransposeType = param.linearTransposeType;
  mlpParam.packQuantType = param.packQuantType.at(1);
  mlpParam.quantGroupSize = param.quantGroupSize;
  // OneRec now uses gated activation (is_gated_act=True by default)
  mlpParam.mlpPackType = atb_speed::common::GATE_UP_WEIGHT_PACK;

  // OneRec now uses RMSNorm instead of LayerNorm
  atb::infer::RmsNormParam mlpRmsNormParam;
  mlpRmsNormParam.layerType =
      atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
  mlpRmsNormParam.normParam.epsilon = param.rmsNormEps;
  mlpParam.normParamType = mlpRmsNormParam;

  atb::infer::RmsNormParam mlpRmsNormQuantParam;
  mlpRmsNormQuantParam.layerType =
      atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
  mlpRmsNormQuantParam.normParam.epsilon = param.rmsNormEps;
  mlpRmsNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
  mlpParam.normQuantParamType = mlpRmsNormQuantParam;

  mlpParam.supportLora = param.supportLora;
  mlpParam.loraEnableGMM = param.loraEnableGMM;
  mlpParam.downLinearTensorParallelInfo = {
      param.rank, param.worldSize, param.backend};
  mlpParam.supportLcoc = param.supportLcoc;

  // OneRec now uses SwiGLU activation with gated structure
  if (param.supportSwiGLU) {
    mlpParam.activationParam.activationType =
        atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD;
    mlpParam.activationParam.dim = -1;
  } else {
    mlpParam.activationParam.activationType =
        atb::infer::ActivationType::ACTIVATION_SWISH;
  }
}

int64_t AddMlp(atb::Node& mlpNode,
               const BlockLayerParam& param,
               std::map<std::string, uint32_t>& tensorMap) {
  atb_speed::common::MlpParam<atb::infer::RmsNormParam> mlpParam;
  SetMlpParam(mlpParam, param);
  if (param.supportSwiGLU) {
    CHECK_OPERATION_STATUS_RETURN(MlpSwiGLU(mlpParam, &mlpNode.operation));
  } else {
    CHECK_OPERATION_STATUS_RETURN(Mlp(mlpParam, &mlpNode.operation));
  }

  std::vector<std::string> mlpInTensorNames;
  if (param.isDecoder) {
    mlpInTensorNames = {"intermediate_cross_residual_out",
                        "in_post_attn_norm_weight",
                        "in_post_attn_norm_bias",
                        "in_post_attn_norm_new_weight",
                        "in_post_attn_norm_new_bias",
                        "in_mlp_weight_0",
                        "in_mlp_scale_0",
                        "in_mlp_offset_0",
                        "in_mlp_descale_0",
                        "in_mlp_bias_0",
                        "in_mlp_compress_idx_0",
                        "in_mlp_weight_1",
                        "in_mlp_scale_1",
                        "in_mlp_offset_1",
                        "in_mlp_descale_1",
                        "in_mlp_bias_1",
                        "in_mlp_compress_idx_1",
                        "in_mlp_down_weight",
                        "in_mlp_down_scale",
                        "in_mlp_down_offset",
                        "in_mlp_down_descale",
                        "in_mlp_down_bias",
                        "in_mlp_down_compress_idx"};
  } else {
    mlpInTensorNames = {"intermediate_self_residual_out",
                        "in_post_attn_norm_weight",
                        "in_post_attn_norm_bias",
                        "in_post_attn_norm_new_weight",
                        "in_post_attn_norm_new_bias",
                        "in_mlp_weight_0",
                        "in_mlp_scale_0",
                        "in_mlp_offset_0",
                        "in_mlp_descale_0",
                        "in_mlp_bias_0",
                        "in_mlp_compress_idx_0",
                        "in_mlp_weight_1",
                        "in_mlp_scale_1",
                        "in_mlp_offset_1",
                        "in_mlp_descale_1",
                        "in_mlp_bias_1",
                        "in_mlp_compress_idx_1",
                        "in_mlp_down_weight",
                        "in_mlp_down_scale",
                        "in_mlp_down_offset",
                        "in_mlp_down_descale",
                        "in_mlp_down_bias",
                        "in_mlp_down_compress_idx"};
  }

  auto oneRecLayerInTensorCandidates = GetOneRecLayerInTensorCandidates();
  if (param.supportLora) {
    mlpInTensorNames.push_back("in_seq_len_cum_sum");
    for (std::string tensor : oneRecLayerInTensorCandidates.at("lora_mlp")) {
      mlpInTensorNames.push_back(tensor);
    }
  }

  std::vector<std::string> mlpOutTensorName = {"intermediate_mlp_out"};

  mlpNode.inTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, mlpInTensorNames);
  mlpNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, mlpOutTensorName);
  return atb::NO_ERROR;
}

int64_t AddSharedExpert(atb::Node& sharedExpertNode,
                        const BlockLayerParam& param,
                        std::map<std::string, uint32_t>& tensorMap) {
  // Use SharedExpert operation (similar to DeepSeek V2 implementation)
  atb_speed::common::SharedExpertParam sharedExpertParam;

  // Set shared expert parameters based on DeepSeek V2 implementation
  sharedExpertParam.isBF16 = param.isBF16;
  sharedExpertParam.transposeGateup =
      param.linearTransposeType[MLP_GATEUP_LINEAR_INDEX];
  sharedExpertParam.transposeDown =
      param.linearTransposeType[MLP_DOWN_LINEAR_INDEX];
  sharedExpertParam.hasSharedExpertGate = param.moe_config->hasSharedExpertGate;
  sharedExpertParam.mlpLinearQuantType = param.moeLinearQuantType;
  sharedExpertParam.mlpLinearTransposeType = param.linearTransposeType;
  sharedExpertParam.quantGroupSize = param.quantGroupSize;
  sharedExpertParam.packQuantType = param.packQuantType.at(1);
  sharedExpertParam.enableCVOverlap =
      false;  // OneRec doesn't use CV overlap
  sharedExpertParam.enableSwiGLUQuantForSharedExperts = param.supportSwiGLU;

  CHECK_OPERATION_STATUS_RETURN(atb_speed::common::CreateSharedExpertOperation(
      sharedExpertParam, &sharedExpertNode.operation));

  // Shared expert input tensors (based on DeepSeek V2 implementation)
  std::vector<std::string> sharedExpertInTensorNames;
  if (param.isDecoder) {
    sharedExpertInTensorNames = {"intermediate_cross_residual_out",
                                 "in_mlp_gateup_weight_shared_expert",
                                 "in_mlp_gateup_bias_shared_expert",
                                 "in_mlp_gateup_descale_shared_expert",
                                 "in_mlp_gateup_offset_shared_expert",
                                 "in_mlp_gateup_scale_shared_expert",
                                 "in_mlp_gateup_compress_idx_shared_expert",
                                 "in_mlp_down_weight_shared_expert",
                                 "in_mlp_down_bias_shared_expert",
                                 "in_mlp_down_descale_shared_expert",
                                 "in_mlp_down_offset_shared_expert",
                                 "in_mlp_down_scale_shared_expert",
                                 "in_mlp_down_compress_idx_shared_expert",
                                 "in_shared_expert_gate_weight",
                                 "in_shared_expert_gate_bias",
                                 "in_shared_expert_gate_descale",
                                 "in_shared_expert_gate_offset",
                                 "in_shared_expert_gate_scale",
                                 "in_shared_expert_gate_compress_idx"};
  } else {
    sharedExpertInTensorNames = {"intermediate_self_residual_out",
                                 "in_mlp_gateup_weight_shared_expert",
                                 "in_mlp_gateup_bias_shared_expert",
                                 "in_mlp_gateup_descale_shared_expert",
                                 "in_mlp_gateup_offset_shared_expert",
                                 "in_mlp_gateup_scale_shared_expert",
                                 "in_mlp_gateup_compress_idx_shared_expert",
                                 "in_mlp_down_weight_shared_expert",
                                 "in_mlp_down_bias_shared_expert",
                                 "in_mlp_down_descale_shared_expert",
                                 "in_mlp_down_offset_shared_expert",
                                 "in_mlp_down_scale_shared_expert",
                                 "in_mlp_down_compress_idx_shared_expert",
                                 "in_shared_expert_gate_weight",
                                 "in_shared_expert_gate_bias",
                                 "in_shared_expert_gate_descale",
                                 "in_shared_expert_gate_offset",
                                 "in_shared_expert_gate_scale",
                                 "in_shared_expert_gate_compress_idx"};
  }

  std::vector<std::string> sharedExpertOutTensorName = {
      "intermediate_shared_expert_out"};

  sharedExpertNode.inTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, sharedExpertInTensorNames);
  sharedExpertNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, sharedExpertOutTensorName);

  ATB_SPEED_LOG_DEBUG("Shared expert calculation success");
  return atb::NO_ERROR;
}

int64_t AddSharedExpertAdd(atb::Node& sharedExpertAddNode,
                           std::map<std::string, uint32_t>& tensorMap) {
  // Add operation to combine shared expert output with MoE output (based on
  // DeepSeek V2)
  atb::infer::ElewiseParam addParam;
  addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
  CHECK_OPERATION_STATUS_RETURN(
      atb::CreateOperation(addParam, &sharedExpertAddNode.operation));

  sharedExpertAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_moe_out", "intermediate_shared_expert_out"});
  sharedExpertAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_shared_expert_add_out"});

  ATB_SPEED_LOG_DEBUG("Shared expert add success");
  return atb::NO_ERROR;
}

int64_t AddMoeNode(atb::Node& moeNode,
                   const BlockLayerParam& param,
                   std::map<std::string, uint32_t>& tensorMap) {
  atb_speed::common::SparseMoeParam sparseMoeParam;
  SetMoeParam(sparseMoeParam, param);

  CHECK_OPERATION_STATUS_RETURN(atb_speed::common::CreateSparseMoeOperation(
      sparseMoeParam, &moeNode.operation));

  std::vector<std::string> moeInTensorNames;
  moeInTensorNames = {"intermediate_selfattention_norm_out",
                      "in_block_sparse_moe_gate_weight",
                      "in_block_sparse_moe_gate_bias",
                      "in_block_sparse_moe_gate_descale",
                      "in_block_sparse_moe_gate_offset",
                      "in_block_sparse_moe_gate_scale",
                      "in_block_sparse_moe_gate_compress_idx",
                      "in_mlp_gateup_weight_expert",
                      "in_mlp_gateup_bias_expert",
                      "in_mlp_gateup_descale_expert",
                      "in_mlp_gateup_offset_expert",
                      "in_mlp_gateup_scale_expert",
                      "in_mlp_gateup_compress_idx_expert",
                      "in_mlp_down_weight_expert",
                      "in_mlp_down_bias_expert",
                      "in_mlp_down_descale_expert",
                      "in_mlp_down_offset_expert",
                      "in_mlp_down_scale_expert",
                      "in_mlp_down_compress_idx_expert",
                      "in_expert_array",
                      "in_expert_group",
                      "in_one_hot",
                      "in_zero_hot"};

  moeNode.inTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, moeInTensorNames);
  moeNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_moe_out"});

  ATB_SPEED_LOG_DEBUG("MoE calculation success");
  return atb::NO_ERROR;
}

atb::Status addCrossNormNode(atb::Node& crossNormNode,
                             const BlockLayerParam& param,
                             std::map<std::string, uint32_t>& tensorMap) {
  atb::infer::RmsNormParam crossNormParam;
  crossNormParam.layerType =
      atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
  crossNormParam.normParam.epsilon = param.rmsNormEps;
  CHECK_OPERATION_STATUS_RETURN(
      atb::CreateOperation(crossNormParam, &crossNormNode.operation));
  crossNormNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap,
      {"intermediate_cross_residual_out", "in_post_attn_norm_weight"});
  crossNormNode.outTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_selfattention_norm_out"});
  ATB_SPEED_LOG_DEBUG("create post normEps");
  return atb::NO_ERROR;
}

atb::Status BlockLayer(const BlockLayerParam& param,
                       atb::Operation** operation) {
  atb::GraphParam opGraph;
  if (param.isDecoder) {
    opGraph.name = param.isPrefill ? "OneRec_Decoder_Prefill_layer"
                                   : "OneRec_Decoder_layer";
  } else {
    opGraph.name = param.isPrefill ? "OneRec_Encoder_Prefill_layer"
                                   : "OneRec_Encoder_layer";
  }

  std::map<std::string, uint32_t> tensorMap =
      ConstructTensorMap(param,
                         opGraph.inTensorNum,
                         opGraph.outTensorNum,
                         opGraph.internalTensorNum);
  atb::Node selfAttentionNode;
  atb::Node selfResidualAddNode;
  atb::Node crossAttentionNode;
  atb::Node crossResidualAddNode;
  atb::Node mlpNode;
  atb::Node crossNormNode;
  atb::Node moeNode;
  atb::Node sharedExpertNode;
  atb::Node sharedExpertAddNode;
  atb::Node mlpResidualAddNode;

  // Self-attention
  if (param.isDecoder) {
    CHECK_OPERATION_STATUS_RETURN(
        AddDecoderSelfAttention(selfAttentionNode, param, tensorMap));
  } else {
    CHECK_OPERATION_STATUS_RETURN(
        AddEncoderSelfAttention(selfAttentionNode, param, tensorMap));
  }
  opGraph.nodes.push_back(selfAttentionNode);

  // Self-attention residual connection
  atb::infer::ElewiseParam addParam;
  addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
  CHECK_OPERATION_STATUS_RETURN(
      atb::CreateOperation(addParam, &selfResidualAddNode.operation));
  // OneRec uses "in_input" instead of "in_hidden_states"
  selfResidualAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"in_input", "intermediate_self_attn_out"});
  selfResidualAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_self_residual_out"});
  opGraph.nodes.push_back(selfResidualAddNode);

  // Cross-attention (for decoder only)
  if (param.isDecoder) {
    CHECK_OPERATION_STATUS_RETURN(
        AddCrossAttention(crossAttentionNode, param, tensorMap));
    opGraph.nodes.push_back(crossAttentionNode);

    // Cross-attention residual connection
    CHECK_OPERATION_STATUS_RETURN(
        atb::CreateOperation(addParam, &crossResidualAddNode.operation));
    crossResidualAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap,
        {"intermediate_self_residual_out", "intermediate_cross_attn_out"});
    crossResidualAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_cross_residual_out"});
    opGraph.nodes.push_back(crossResidualAddNode);
  }

  // MLP or MoE
  if (param.use_moe) {
    // MoE is only supported for decoder mode
    if (!param.isDecoder) {
      ATB_SPEED_LOG_ERROR("MoE is only supported for decoder");
      return atb::ERROR_INVALID_PARAM;
    }
    // Cross-attention norm
    CHECK_OPERATION_STATUS_RETURN(
        addCrossNormNode(crossNormNode, param, tensorMap));
    opGraph.nodes.push_back(crossNormNode);

    CHECK_OPERATION_STATUS_RETURN(AddMoeNode(moeNode, param, tensorMap));
    opGraph.nodes.push_back(moeNode);

    // Add shared expert if enabled
    if (param.moe_config && param.moe_config->moe_use_shared_experts) {
      CHECK_OPERATION_STATUS_RETURN(
          AddSharedExpert(sharedExpertNode, param, tensorMap));
      opGraph.nodes.push_back(sharedExpertNode);

      CHECK_OPERATION_STATUS_RETURN(
          AddSharedExpertAdd(sharedExpertAddNode, tensorMap));
      opGraph.nodes.push_back(sharedExpertAddNode);

      // MoE + Shared Expert residual connection
      CHECK_OPERATION_STATUS_RETURN(
          atb::CreateOperation(addParam, &mlpResidualAddNode.operation));
      mlpResidualAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(
          tensorMap,
          {"intermediate_selfattention_norm_out",
           "intermediate_shared_expert_add_out"});
    } else {
      // MoE residual connection
      CHECK_OPERATION_STATUS_RETURN(
          atb::CreateOperation(addParam, &mlpResidualAddNode.operation));
      mlpResidualAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(
          tensorMap,
          {"intermediate_cross_residual_out", "intermediate_moe_out"});
    }
  } else {
    CHECK_OPERATION_STATUS_RETURN(AddMlp(mlpNode, param, tensorMap));
    opGraph.nodes.push_back(mlpNode);

    // MLP residual connection
    CHECK_OPERATION_STATUS_RETURN(
        atb::CreateOperation(addParam, &mlpResidualAddNode.operation));
    if (param.isDecoder) {
      mlpResidualAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(
          tensorMap,
          {"intermediate_cross_residual_out", "intermediate_mlp_out"});
    } else {
      mlpResidualAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(
          tensorMap,
          {"intermediate_self_residual_out", "intermediate_mlp_out"});
    }
  }
  mlpResidualAddNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"out"});
  opGraph.nodes.push_back(mlpResidualAddNode);

  opGraph.inferShapeFunc =
      [=](const atb::SVector<atb::TensorDesc>& inTensorDescs,
          atb::SVector<atb::TensorDesc>& outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
      };
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
  return atb::NO_ERROR;
}

}  // namespace onerec
}  // namespace atb_speed
