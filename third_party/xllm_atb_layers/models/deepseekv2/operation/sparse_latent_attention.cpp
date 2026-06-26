/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

#include "atb_speed/base/model.h"
#include "operations/fusion/utils.h"
#include "operations/aclrt/ops/aclrt_cmo_async.h"
#include "operations/aclnn/ops/aclnn_rope_operation.h"
#include "operations/aclnn/ops/concat_operation.h"
#include "operations/aclnn/ops/split_with_size_operation.h"
#include "operations/aclnn/ops/repeat_operation.h"
#include "operations/aclnn/ops/lightning_indexer_operation.h"
#include "operations/aclnn/ops/sparse_flash_attention_operation.h"
#include "operations/aclnn/ops/mla_preprocess_v2_operation.h"
#include <operations/aclnn/ops/multi_latent_attention.h>
#include "operations/fusion/utils.h"
#include "models/deepseekv2/operation/fa_update.h"
#include "models/deepseekv2/operation/ring_attention.h"
#include "models/deepseekv2/operation/sparse_latent_attention.h"
#include <gflags/gflags.h>
#include <glog/logging.h>

namespace atb_speed {
namespace deepseekV2 {
using namespace atb_speed::common;

namespace sparse {
template <typename NormParamType>
bool EnableFA3Quant(const LatentAttentionParam<NormParamType> &param)
{
    return param.pageAttentionParam.quantType == \
        atb::infer::PagedAttentionParam::QuantType::TYPE_QUANT_QKV_ONLINE;
}

std::map<std::string, std::vector<std::string>> GetLatentAttnInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>>  latentAttnInTensorCandidates = {
        {"default", {
            "in_input", "in_norm_weight", "in_norm_bias", "in_norm_new_weight", "in_norm_new_weight_bias",
            "in_q_proj_a_weight", "in_q_proj_a_bias", "in_q_proj_a_descale", "in_q_proj_a_offset", "in_q_proj_a_scale",
            "in_q_proj_a_compress_idx",
            "in_q_proj_a_layernorm_weight", "in_q_proj_a_layernorm_bias",
            "in_q_proj_b_weight", "in_q_proj_b_bias", "in_q_proj_b_descale", "in_q_proj_b_offset", "in_q_proj_b_scale",
            "in_q_proj_b_compress_idx",
            "in_kv_proj_with_mqa_weight", "in_kv_proj_with_mqa_bias", "in_kv_proj_with_mqa_descale",
            "in_kv_proj_with_mqa_offset", "in_kv_proj_with_mqa_scale", "in_kv_proj_with_mqa_compress_idx",
            "in_kv_proj_a_layernorm_weight", "in_kv_proj_a_layernorm_bias",
            "in_k_proj_b_for_q_weight", "in_k_proj_b_for_q_bias", "in_k_proj_b_for_q_descale",
            "in_k_proj_b_for_q_offset", "in_k_proj_b_for_q_scale", "in_k_proj_b_for_q_compress_idx",
            "in_v_proj_b_for_o_weight", "in_v_proj_b_for_o_bias", "in_v_proj_b_for_o_descale",
            "in_v_proj_b_for_o_offset", "in_v_proj_b_for_o_scale", "in_v_proj_b_for_o_compress_idx",
            "in_attn_out_weight", "in_attn_out_bias", "in_attn_out_descale", "in_attn_out_offset",
            "in_attn_out_scale", "in_attn_out_compress_idx",
            "in_indexer_proj_wq_b_weight", "in_indexer_proj_wq_b_bias", "in_indexer_proj_wq_b_descale",
            "in_indexer_proj_wq_b_offset", "in_indexer_proj_wq_b_scale","in_indexer_proj_wq_b_compress_idx",
            "in_indexer_proj_wk_weight", "in_indexer_proj_wk_bias", "in_indexer_proj_wk_descale",
            "in_indexer_proj_wk_offset", "in_indexer_proj_wk_scale","in_indexer_proj_wk_compress_idx",
            "in_indexer_proj_k_norm_weight", "in_indexer_proj_k_norm_bias",
            "in_indexer_proj_weight", "in_indexer_proj_bias", "in_indexer_proj_descale",
            "in_indexer_proj_offset", "in_indexer_proj_scale","in_indexer_proj_compress_idx",
            "in_q_proj_a_recompte_weight", "in_q_proj_a_recompte_bias", "in_q_proj_a_recompte_descale",
            "in_q_proj_a_recompte_offset", "in_q_proj_a_recompte_scale","in_q_proj_a_recompte_compress_idx",
            "in_cos_embed", "in_sin_embed", "in_seq_len", "in_k_cache", "in_k_rope_cache",
            "in_attention_mask", "in_q_len",
            "in_token_offset", "in_layer_id", "in_block_tables",
            "in_slots_in_pa_or_logn_in_fa", "in_attn_padding_idx", "in_k_cache_indexer",
            "in_seq_len_query"}
        },
        {"indexer", {
            "in_indexer_proj_wq_b_weight", "in_indexer_proj_wq_b_bias", "in_indexer_proj_wq_b_descale",
            "in_indexer_proj_wq_b_offset", "in_indexer_proj_wq_b_scale","in_indexer_proj_wq_b_compress_idx",
            "in_indexer_proj_wk_weight", "in_indexer_proj_wk_bias", "in_indexer_proj_wk_descale",
            "in_indexer_proj_wk_offset", "in_indexer_proj_wk_scale","in_indexer_proj_wk_compress_idx",
            "in_indexer_proj_k_norm_weight", "in_indexer_proj_k_norm_bias",
            "in_indexer_proj_weight", "in_indexer_proj_bias", "in_indexer_proj_descale",
            "in_indexer_proj_offset", "in_indexer_proj_scale","in_indexer_proj_compress_idx",
            "in_q_proj_a_recompte_weight", "in_q_proj_a_recompte_bias", "in_q_proj_a_recompte_descale",
            "in_q_proj_a_recompte_offset", "in_q_proj_a_recompte_scale","in_q_proj_a_recompte_compress_idx"}
        },
        {"fa3_quant", {
            "in_q_quant_scale", "in_k_quant_scale", "in_qk_descale",
            "kv_offset", "fa3_v_quant_scale"}
        },
        {"attn_cp_prefill", {
            // "in_seq_len_cp",
            // "in_cp_load_balance_idx_first", "in_cp_load_balance_idx_last",
            "cp_o_recover_idx", "cp_kv_recover_idx", "cp_load_balance_idx",
            "k_gather_index_prev", "k_gather_index_next",
            "actual_seq_lengths_query_prev", "actual_seq_lengths_query_next",
            "actual_seq_lengths_key_prev", "actual_seq_lengths_key_next"
        }},
        {"attn_inner_sp_decode", {"in_seq_len_sp"}},
        {"qkvdown_dp", {"in_ffn_unpadding_idx"}
        },
        {"prefixcache", {"in_history_compressed_kv", "in_history_k_rope", "ring_cur_seqlen", "ring_cache_seqlen"}
        },
        
    };
    return latentAttnInTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetLatentAttnIntermediateTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> latentAttnIntermediateTensorCandidates = {
        {"default",
            {
                "in_input_norm", "latent_q", "nope_q", "rope_q", "rope_k", "rope_q_o", "rope_k_o",
                "intermediate_kv", "intermediate_self_attention"
            }
        },
        {"prefill",
            {   // "intermediate_q",
                "intermediate_q_t" //, "nope_q_t"
            }
        },
        {"indexer",
            {
                "intermediate_indexer_qa_norm", "intermediate_indexer_qb", "indexer_rope_q", "indexer_nope_q",
                "intermediate_indexer_k", "intermediate_indexer_k_norm", "indexer_rope_k", "indexer_nope_k", "indexer_rope_q_o", "indexer_rope_k_o",
                "intermediate_indexer_q_out", "intermediate_indexer_k_out", "intermediate_indexer_input_norm",
                "intermediate_indexer_weight_out", "intermediate_topk_indices", "intermediate_reproj_t"
            }
        },
        {"orpoj_transpose", {"intermediate_sfa_out", "intermediate_reproj"}},
        {"decode", {"reproj_nope_q", "reproj_o"}},
        {"q_lora", {"latent_qkv", "latent_q_norm", "q_lora_out"}},
        {"no_q_lora", {"latent_kv"}},
        {"mla_preprocess", {"intermediate_q_nope", "intermediate_self_attention", "intermediate_q_rope"}},
        {"kv_quant_scale", {"intermediate_kv_int8"}},
        {"attn_cp_prefill", {
            "intermediate_kv_cp", "intermediate_kv_allgather", 
            "rope_k_o_cp", "rope_k_o_allgather",
            // "in_cos_embed_allgather", "in_cos_embed_allgather_s",
            // "in_sin_embed_allgather", "in_sin_embed_allgather_s",
            "indexer_k_out_allgather", "indexer_k_out_allgather_s",
            "indexer_q_out_balance", "indexer_weight_out_balance",
            "indexer_q_out_prev", "indexer_q_out_next",
            "indexer_weight_out_prev", "indexer_weight_out_next",
            "indexer_k_out_prev", "indexer_k_out_next",
            "intermediate_topk_indices_prev_cat", "intermediate_topk_indices_next_cat",
            "intermediate_topk_indices_balance", "intermediate_topk_indices_prev", "intermediate_topk_indices_next",
            "intermediate_topk_indices_draft",
            "intermediate_q_balance", "intermediate_q_prev", "intermediate_q_next",
            "rope_q_o_balance", "rope_q_o_prev", "rope_q_o_next",
            "intermediate_kv_prev", "intermediate_kv_next",
            "rope_k_o_prev", "rope_k_o_next",
            "intermediate_self_attention_prev", "intermediate_self_attention_next",
            "intermediate_self_attention_draft"
        }},
        {"attn_cp_decode", {"intermediate_go_lse_allgather"}},
        {"attn_inner_sp_decode", {
            "intermediate_q", "intermediate_q_allgather_sp", "intermediate_q_allgather_sp_t",
            "intermediate_q_sp_nope", "intermediate_q_sp_rope", "intermediate_go_lse_all2all"}},
        {"fa_update", {
            "intermediate_go", "intermediate_lse", "intermediate_go_lse",
            "intermediate_go_t", "intermediate_lse_t",
            "intermediate_go_fp32", "intermediate_lse_fp32", "intermediate_fa_update_out_fp32"
        }},
        {"extra_o_proj_tp", {"intermediate_self_attention_padding"}},
        {"extra_o_proj_tp_quant", {"intermediate_self_attention_padding_quant"}},
        {"qkvdown_dp", {"intermediate_qkv_all", "intermediate_qkv_unpadding_all"}},
        {"enable_preprocess_lcoc_tp", {"latent_kv", "intermediate_kv_all", "intermediate_kv_unpadding_all",
            "fused_latent_q_norm", "intermediate_q_b", "fused_q_lora_out"}},
        {"prefixcache",
            {
                "rope_k_o_repeat", "intermediate_k_nope", "intermediate_v_mha",
                "rope_k_o_repeat_history", "intermediate_k_nope_history", "intermediate_v_mha_history",
                "temp_v_proj_b", "cur_lse", "cache_lse", "cur_intermediate_self_attention"}},
        {"enable_fused_mla", {"rope_k_o_repeat", "intermediate_k_nope", "intermediate_v_mha", "temp_v_proj_b"}},
    };
    return latentAttnIntermediateTensorCandidates;
}

template <typename NormParamType>
bool UseExtraQuant(const LatentAttentionParam<NormParamType> &param, uint64_t linearIndex)
{
    LinearQuantType quantType = GetLinearQuantType(
        param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED \
            ? param.packQuantType : param.denseQuantType,
        param.attnLinearQuantType[linearIndex], true);
    if (quantType == LinearQuantType::LINEAR_W8A8_DEQUANT || \
        quantType == LinearQuantType::LINEAR_W8A8_SC_DEQUANT) {
        return true;
    } else {
        return false;
    }
}

template <typename NormParamType>
std::map<std::string, uint32_t> ConstructTensorMap(const LatentAttentionParam<NormParamType> &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum)
{
    std::vector<std::string> inTensorList = {};
    std::vector<std::string> outTensorList = {"out"};
    std::vector<std::string> intermediateTensorList = {};
    auto latentAttnInTensorCandidates = GetLatentAttnInTensorCandidates();
    auto latentAttnIntermediateTensorCandidates = GetLatentAttnIntermediateTensorCandidates();
    AddTensorToList(latentAttnInTensorCandidates, "default", inTensorList);  // translatedTensor
    if (param.enablePrefixCache) {
        AddTensorToList(latentAttnInTensorCandidates, "prefixcache", inTensorList);
    }
    if (EnableFA3Quant(param)) {  // translatedFA3translatedTensor
        AddTensorToList(latentAttnInTensorCandidates, "fa3_quant", inTensorList);
        if (!(param.qLoraRank > 0 && param.enableMlaPreprocess && !param.isPrefill)) {
            AddTensorToList(latentAttnIntermediateTensorCandidates, "kv_quant_scale", intermediateTensorList);
        }
    }
    if (param.enableQkvdownDp) {
        AddTensorToList(latentAttnInTensorCandidates, "qkvdown_dp", inTensorList);
        AddTensorToList(latentAttnIntermediateTensorCandidates, !param.enablePreprocessLcocTp ? \
            "qkvdown_dp" : "enable_preprocess_lcoc_tp", intermediateTensorList);
    }
    if (param.enableMlaPreprocess && !param.isPrefill) {
        AddTensorToList(latentAttnIntermediateTensorCandidates, "mla_preprocess", intermediateTensorList);
    } else {
        AddTensorToList(latentAttnIntermediateTensorCandidates, "default", intermediateTensorList);
        if (param.qLoraRank != 0) {
            if (!param.enablePreprocessLcocTp) {
                AddTensorToList(latentAttnIntermediateTensorCandidates, "q_lora", intermediateTensorList);
            }
        } else {
            AddTensorToList(latentAttnIntermediateTensorCandidates, "no_q_lora", intermediateTensorList);
        }
        if (param.isPrefill || !param.enableMlaPreprocess) {
            if (param.enablePrefixCache) {
                AddTensorToList(latentAttnIntermediateTensorCandidates, "prefixcache", intermediateTensorList);
            } else if (param.enableFusedMLA) {
                AddTensorToList(latentAttnIntermediateTensorCandidates, "enable_fused_mla", intermediateTensorList);
            } else {
                AddTensorToList(latentAttnIntermediateTensorCandidates, "prefill", intermediateTensorList);
            }
        } else {
            AddTensorToList(latentAttnIntermediateTensorCandidates, "decode", intermediateTensorList);
        }
    }
    AddTensorToList(latentAttnIntermediateTensorCandidates, "indexer", intermediateTensorList);    
    // using MATMUL_EIN_SUM, skip trans
    // AddTensorToList(latentAttnIntermediateTensorCandidates, "orpoj_transpose", intermediateTensorList);

    if (param.contextParallelInfo.IsEnabled()) {
        if (param.isPrefill) {
            AddTensorToList(latentAttnInTensorCandidates, "attn_cp_prefill", inTensorList);
            AddTensorToList(latentAttnIntermediateTensorCandidates, "attn_cp_prefill", intermediateTensorList);
        }
    }
    if (param.hasAttnInnerSp && !param.isPrefill) {
        AddTensorToList(latentAttnInTensorCandidates, "attn_inner_sp_decode", inTensorList);
        AddTensorToList(latentAttnIntermediateTensorCandidates, "attn_inner_sp_decode", intermediateTensorList);
    }
    if (param.enableExtraOprojTp || param.enableOutLcocTp) {
        AddTensorToList(latentAttnIntermediateTensorCandidates, "extra_o_proj_tp", intermediateTensorList);
        if (UseExtraQuant(param, O_LINEAR_INDEX) || param.enableOutLcocTp) {
            AddTensorToList(latentAttnIntermediateTensorCandidates, "extra_o_proj_tp_quant", intermediateTensorList);
        }
    }
    inTensorNum = inTensorList.size();
    outTensorNum = outTensorList.size();
    internalTensorNum = intermediateTensorList.size();
    return GetTensorMap(inTensorList, outTensorList, intermediateTensorList);
}

atb::Status AddKVQuantNode(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node kvQuantNode;
    atb::infer::ElewiseParam kvQuantParam;
    kvQuantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT_PER_CHANNEL;
    CREATE_OPERATION(kvQuantParam, &kvQuantNode.operation);
    kvQuantNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_kv"), GetTensorIdx(tensorMap, "in_k_quant_scale"),
        GetTensorIdx(tensorMap, "kv_offset")
    };
    kvQuantNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_kv_int8")};
    kvQuantNode.inTensorReshapeFuncs.resize(kvQuantNode.inTensorIds.size());
    opGraph.nodes.push_back(kvQuantNode);
    return atb::NO_ERROR;
}

void SqueezeHeadNumHeadDim(const atb::Dims &oldShape, atb::Dims &newShape)
{
    if (oldShape.dimNum == 4) {  // 4: FA
        newShape.dimNum = 3;  // 3: translatedshapetranslated3
        newShape.dims[0] = oldShape.dims[0];  // 0, 0: translatedshapetranslated0translated
        newShape.dims[1] = oldShape.dims[1];  // 1, 1: translatedshapetranslated1translated
        newShape.dims[2] =  oldShape.dims[2] * oldShape.dims[3];  // 2, 2, 3: translated
    } else {
        newShape.dimNum = 2;  // 2: translatedshapetranslated2
        newShape.dims[0] = oldShape.dims[0];  // 0, 0: translatedshapetranslated0translated
        newShape.dims[1] =  oldShape.dims[1] * oldShape.dims[2];  // 1, 1, 2: translated
    }
}

void UnSqueezeHeadNumHeadDim(const atb::Dims &oldShape, atb::Dims &newShape, int headDim)
{
    if (oldShape.dimNum == 2) {
        newShape.dimNum = 3;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1] / headDim;
        newShape.dims[2] = headDim;
    }
    else {
        newShape = oldShape;
    }
}

template <typename NormParamType>
atb::Status AddSfaAllGatherAndReorderCpNode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap, std::string in, std::string inter, std::string out, std::string idx)
{
    atb::Node allGatherNode;
    atb::infer::AllGatherParam allGatherParam;
    allGatherParam.rank = param.contextParallelInfo.rank;
    allGatherParam.rankSize = param.contextParallelInfo.rankIds.size();
    allGatherParam.backend = param.contextParallelInfo.defaultBackend;
    param.contextParallelInfo.InitCommDomain(allGatherParam.hcclComm, allGatherParam.commDomain);
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(allGatherParam, &allGatherNode.operation));
    allGatherNode.inTensorIds = GetTensorIdxList(tensorMap, {in});
    allGatherNode.outTensorIds = GetTensorIdxList(tensorMap, {inter});
    opGraph.nodes.push_back(allGatherNode);

    atb::Node nopeGatherNode;
    atb::infer::GatherParam nopeGatherParam;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(nopeGatherParam, &nopeGatherNode.operation));
    nopeGatherNode.inTensorIds = GetTensorIdxList(tensorMap, {inter, idx});
    nopeGatherNode.outTensorIds = {GetTensorIdx(tensorMap, out)};
    nopeGatherNode.inTensorReshapeFuncs.resize(nopeGatherNode.inTensorIds.size());
    nopeGatherNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        // Squeeze oldShape dim 0,1 into one dimension
        newShape.dimNum = oldShape.dimNum - 1;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        for (int i = 1; i < newShape.dimNum; ++i) {
            newShape.dims[i] = oldShape.dims[i + 1];
        }
    };
    opGraph.nodes.push_back(nopeGatherNode);

    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddSfaGatherCpNode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap, std::string in, std::string out, std::string idx)
{
    atb::Node nopeGatherNode;
    atb::infer::GatherParam nopeGatherParam;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(nopeGatherParam, &nopeGatherNode.operation));
    nopeGatherNode.inTensorIds = GetTensorIdxList(tensorMap, {in, idx});
    nopeGatherNode.outTensorIds = {GetTensorIdx(tensorMap, out)};
    opGraph.nodes.push_back(nopeGatherNode);

    return atb::NO_ERROR;
}

template<typename NormParamType>
atb::Status AddSfaSplitCpNode(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
                           std::map<std::string, uint32_t> &tensorMap, std::string in, std::string out_p, std::string out_n)
{
    atb::Node splitKNode;
    atb::infer::SplitParam splitKParam = {0, 2};
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(splitKParam, &splitKNode.operation));
    splitKNode.inTensorIds = {GetTensorIdx(tensorMap, in)};
    splitKNode.outTensorIds = {
        GetTensorIdx(tensorMap, out_p), GetTensorIdx(tensorMap, out_n)
    };
    opGraph.nodes.push_back(splitKNode);

    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddMlaPreprocessNode(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node mlaPreprocessNode;
    atb::infer::MlaPreprocessParam mlaPreprocessParam;
    mlaPreprocessParam.wdqDim = param.qLoraRank;
    mlaPreprocessParam.qRopeDim = param.qkRopeHeadDim;
    mlaPreprocessParam.kRopeDim = param.qkRopeHeadDim;
    if (EnableFA3Quant(param)) {
        mlaPreprocessParam.cacheMode = atb::infer::MlaPreprocessParam::CacheMode::INT8_NZCACHE;
    } else if (param.isNzCache) {
        mlaPreprocessParam.cacheMode = atb::infer::MlaPreprocessParam::CacheMode::NZCACHE;
    } else {
        mlaPreprocessParam.cacheMode = atb::infer::MlaPreprocessParam::CacheMode::KROPE_CTKV;
    }
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(mlaPreprocessParam, &mlaPreprocessNode.operation));
    mlaPreprocessNode.inTensorIds = {
        GetTensorIdx(tensorMap, "in_input"), GetTensorIdx(tensorMap, "in_norm_weight"),
        GetTensorIdx(tensorMap, "in_norm_bias"), GetTensorIdx(tensorMap, "in_q_proj_a_scale"),
        GetTensorIdx(tensorMap, "in_q_proj_a_offset"), GetTensorIdx(tensorMap, "in_q_proj_a_weight"),
        GetTensorIdx(tensorMap, "in_q_proj_a_descale"), GetTensorIdx(tensorMap, "in_q_proj_a_bias"),
        GetTensorIdx(tensorMap, "in_q_proj_a_layernorm_weight"), GetTensorIdx(tensorMap, "in_q_proj_a_layernorm_bias"),
        GetTensorIdx(tensorMap, "in_q_proj_b_scale"), GetTensorIdx(tensorMap, "in_q_proj_b_offset"),
        GetTensorIdx(tensorMap, "in_q_proj_b_weight"), GetTensorIdx(tensorMap, "in_q_proj_b_descale"),
        GetTensorIdx(tensorMap, "in_q_proj_b_bias"), GetTensorIdx(tensorMap, "in_kv_proj_a_layernorm_weight"),
        GetTensorIdx(tensorMap, "in_cos_embed"), GetTensorIdx(tensorMap, "in_sin_embed"),
        GetTensorIdx(tensorMap, "in_k_proj_b_for_q_weight"), GetTensorIdx(tensorMap, "in_k_cache"),
        GetTensorIdx(tensorMap, "in_k_rope_cache"), GetTensorIdx(tensorMap, "in_slots_in_pa_or_logn_in_fa"),
    };
    if (EnableFA3Quant(param)) {
        mlaPreprocessNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_k_quant_scale"));
        mlaPreprocessNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_q_quant_scale"));
    } else {
        mlaPreprocessNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_kv_proj_with_mqa_scale"));
        mlaPreprocessNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_q_proj_b_scale"));
    }
    mlaPreprocessNode.outTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_q_nope"),
        GetTensorIdx(tensorMap, "in_k_cache"),
        GetTensorIdx(tensorMap, "intermediate_q_rope"),
        GetTensorIdx(tensorMap, "in_k_rope_cache")
    };
    opGraph.nodes.push_back(mlaPreprocessNode);
    ATB_SPEED_LOG_DEBUG("MlaPreprocessNode calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddMlaPreprocessV2Node(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node mlaPreprocessV2Node;
    atb_speed::common::Mlapreprocessv2OperationParam mlaPreprocessV2Param;
    mlaPreprocessV2Param.wdqDim = param.qLoraRank;
    mlaPreprocessV2Param.qRopeDim = param.qkRopeHeadDim;
    mlaPreprocessV2Param.kRopeDim = param.qkRopeHeadDim;
    if (EnableFA3Quant(param)) {
        mlaPreprocessV2Param.cacheMode = atb::infer::MlaPreprocessParam::CacheMode::INT8_NZCACHE;
    } else if (param.isNzCache) {
        mlaPreprocessV2Param.cacheMode = atb::infer::MlaPreprocessParam::CacheMode::NZCACHE;
    } else {
        mlaPreprocessV2Param.cacheMode = atb::infer::MlaPreprocessParam::CacheMode::KROPE_CTKV;
    }

    mlaPreprocessV2Node.operation = new atb_speed::common::Mlapreprocessv2Operation(
        "AclNNMlaPreprocessV2Node", mlaPreprocessV2Param
    );

    mlaPreprocessV2Node.inTensorIds = {
        GetTensorIdx(tensorMap, "in_input"), GetTensorIdx(tensorMap, "in_norm_weight"),
        GetTensorIdx(tensorMap, "in_norm_bias"), GetTensorIdx(tensorMap, "in_q_proj_a_scale"),
        GetTensorIdx(tensorMap, "in_q_proj_a_offset"), GetTensorIdx(tensorMap, "in_q_proj_a_weight"),
        GetTensorIdx(tensorMap, "in_q_proj_a_descale"), GetTensorIdx(tensorMap, "in_q_proj_a_bias"),
        GetTensorIdx(tensorMap, "in_q_proj_a_layernorm_weight"), GetTensorIdx(tensorMap, "in_q_proj_a_layernorm_bias"),
        GetTensorIdx(tensorMap, "in_q_proj_b_scale"), GetTensorIdx(tensorMap, "in_q_proj_b_offset"),
        GetTensorIdx(tensorMap, "in_q_proj_b_weight"), GetTensorIdx(tensorMap, "in_q_proj_b_descale"),
        GetTensorIdx(tensorMap, "in_q_proj_b_bias"), GetTensorIdx(tensorMap, "in_kv_proj_a_layernorm_weight"),
        GetTensorIdx(tensorMap, "in_cos_embed"), GetTensorIdx(tensorMap, "in_sin_embed"),
        GetTensorIdx(tensorMap, "in_k_proj_b_for_q_weight"), GetTensorIdx(tensorMap, "in_k_cache"),
        GetTensorIdx(tensorMap, "in_k_rope_cache"), GetTensorIdx(tensorMap, "in_slots_in_pa_or_logn_in_fa"),
    };
    if (EnableFA3Quant(param)) {
        mlaPreprocessV2Node.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_k_quant_scale"));
        mlaPreprocessV2Node.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_q_quant_scale"));
    } else {
        mlaPreprocessV2Node.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_kv_proj_with_mqa_scale"));
        mlaPreprocessV2Node.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_q_proj_b_scale"));
    }
    mlaPreprocessV2Node.outTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_q_nope"),
        GetTensorIdx(tensorMap, "in_k_cache"),
        GetTensorIdx(tensorMap, "intermediate_q_rope"),
        GetTensorIdx(tensorMap, "in_k_rope_cache"),
        GetTensorIdx(tensorMap, "intermediate_indexer_qa_norm"),
    };
    opGraph.nodes.push_back(mlaPreprocessV2Node);
    ATB_SPEED_LOG_DEBUG("mlaPreprocessV2Node calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddLAttnPreNormNode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node normNode;
    
     if (UseExtraQuant(param, Q_PROJ_A_LINEAR_INDEX)) {  // W8A8
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.normQuantParamType, &normNode.operation));
        normNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_input"));
        normNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_norm_weight"));
        normNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_norm_bias"));
        normNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_q_proj_a_scale"));
        normNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_q_proj_a_offset"));
    } else {
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.normParamType, &normNode.operation));
        normNode.inTensorIds = {GetTensorIdx(tensorMap, "in_input"), GetTensorIdx(tensorMap, "in_norm_weight")};
    }
    
    normNode.outTensorIds = {GetTensorIdx(tensorMap, "in_input_norm")};
    opGraph.nodes.push_back(normNode);
    ATB_SPEED_LOG_DEBUG("SparseAttention PreNorm calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddLAttnQKVProjNode(const LatentAttentionParam<NormParamType> &param,
                                atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node qkvAProjNode;
    atb_speed::common::FusionLinearParam kvAProjNodeParam;
    kvAProjNodeParam.isBF16 = param.isBF16;
    kvAProjNodeParam.hasBias = param.selfAttnHasBias;
    kvAProjNodeParam.quantType = GetLinearQuantType(
        param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED ? \
            param.packQuantType : param.denseQuantType,
        param.attnLinearQuantType[Q_PROJ_A_LINEAR_INDEX], true);
    if (kvAProjNodeParam.quantType == LinearQuantType::LINEAR_W8A8_DYNAMIC_DEQUANT) {
        kvAProjNodeParam.quantType = LinearQuantType::LINEAR_W8A8_DYNAMIC_QUANT;
    }
    kvAProjNodeParam.quantGroupSize = param.quantGroupSize;
    qkvAProjNode.inTensorIds = {
        GetTensorIdx(tensorMap, "in_input_norm"),
        GetTensorIdx(tensorMap, "in_q_proj_a_weight"),
        GetTensorIdx(tensorMap, "in_q_proj_a_scale"),
        GetTensorIdx(tensorMap, "in_q_proj_a_offset"),
        GetTensorIdx(tensorMap, "in_q_proj_a_descale"),
        GetTensorIdx(tensorMap, "in_q_proj_a_bias"),
        GetTensorIdx(tensorMap, "in_q_proj_a_compress_idx"),
    };
    qkvAProjNode.outTensorIds = {GetTensorIdx(tensorMap, "latent_qkv")};
    CHECK_OPERATION_STATUS_RETURN(FusionLinear(kvAProjNodeParam, &qkvAProjNode.operation));
    opGraph.nodes.push_back(qkvAProjNode);
    ATB_SPEED_LOG_DEBUG("MLA proj_qkv_a calculation success");
    return atb::NO_ERROR;
}

template<typename NormParamType>
atb::Status AddSplitQKNode(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
                           std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node splitKNode;
    atb::infer::SplitParam splitKParam = {
        (param.isFA ? 2 : 1), 3, {param.kvLoraRank, param.qkRopeHeadDim, param.qLoraRank}};
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(splitKParam, &splitKNode.operation));
    splitKNode.inTensorIds = {GetTensorIdx(tensorMap, param.enableQkvdownDp ?
        "intermediate_qkv_unpadding_all" : "latent_qkv")};
    splitKNode.outTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_kv"), GetTensorIdx(tensorMap, "rope_k"), GetTensorIdx(tensorMap, "latent_q"),
    };
    
    if (param.contextParallelInfo.IsEnabled() && param.isPrefill) {
        splitKNode.outTensorIds = {
            GetTensorIdx(tensorMap, "intermediate_kv_cp"), GetTensorIdx(tensorMap, "rope_k"), GetTensorIdx(tensorMap, "latent_q"),
        };
    }
    opGraph.nodes.push_back(splitKNode);
    ATB_SPEED_LOG_DEBUG("MLA spilt_qk calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddLAttnQProjANode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node qAProjNode;
    atb_speed::common::FusionLinearParam qAProjNodeParam;
    qAProjNodeParam.isBF16 = param.isBF16;
    qAProjNodeParam.hasBias = param.selfAttnHasBias;
    qAProjNodeParam.quantType = GetLinearQuantType(
        param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED \
            ? param.packQuantType : param.denseQuantType,
        param.attnLinearQuantType[Q_PROJ_A_LINEAR_INDEX], true);
    qAProjNodeParam.quantGroupSize = param.quantGroupSize;
    qAProjNodeParam.transposeType = param.attnLinearTransposeType[Q_PROJ_A_LINEAR_INDEX];
    qAProjNode.inTensorIds = {
        GetTensorIdx(tensorMap, "in_input_norm"),
        GetTensorIdx(tensorMap, "in_q_proj_a_weight"),
        GetTensorIdx(tensorMap, "in_q_proj_a_scale"),
        GetTensorIdx(tensorMap, "in_q_proj_a_offset"),
        GetTensorIdx(tensorMap, "in_q_proj_a_descale"),
        GetTensorIdx(tensorMap, "in_q_proj_a_bias"),
        GetTensorIdx(tensorMap, "in_q_proj_a_compress_idx"),
    };
    qAProjNode.outTensorIds = {GetTensorIdx(tensorMap, "latent_q")};
    CHECK_OPERATION_STATUS_RETURN(FusionLinear(qAProjNodeParam, &qAProjNode.operation));
    opGraph.nodes.push_back(qAProjNode);
    ATB_SPEED_LOG_DEBUG("MLA proj_q_a calculation success");
    return atb::NO_ERROR;
}


template <typename NormParamType>
atb::Status AddLAttnQProjBNode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node qNormNode;
    if (UseExtraQuant(param, Q_PROJ_B_LINEAR_INDEX)) {  // W8A8
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.normQuantParamType, &qNormNode.operation));
        qNormNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "latent_q"));
        qNormNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_q_proj_a_layernorm_weight"));
        qNormNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_q_proj_a_layernorm_bias"));
        qNormNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_q_proj_b_scale"));
        qNormNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_q_proj_b_offset"));
    } else {
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.normParamType, &qNormNode.operation));
        qNormNode.inTensorIds = {GetTensorIdx(tensorMap, "latent_q"),
                                GetTensorIdx(tensorMap, "in_q_proj_a_layernorm_weight")};
    }
    qNormNode.outTensorIds = {GetTensorIdx(tensorMap, "latent_q_norm")};
    opGraph.nodes.push_back(qNormNode);

    atb::Node qBProjNode;
    atb_speed::common::FusionLinearParam qBProjNodeParam;
    qBProjNodeParam.isBF16 = param.isBF16;
    qBProjNodeParam.hasBias = param.selfAttnHasBias;
    qBProjNodeParam.quantType = GetLinearQuantType(
        param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED \
            ? param.packQuantType : param.denseQuantType,
        param.attnLinearQuantType[Q_PROJ_B_LINEAR_INDEX], true);
    if (qBProjNodeParam.quantType == LinearQuantType::LINEAR_W8A8_DYNAMIC_DEQUANT) {
        qBProjNodeParam.quantType = LinearQuantType::LINEAR_W8A8_DYNAMIC_QUANT;
    }
    qBProjNodeParam.quantGroupSize = param.quantGroupSize;
    qBProjNodeParam.transposeType = param.attnLinearTransposeType[Q_PROJ_B_LINEAR_INDEX];
    CHECK_OPERATION_STATUS_RETURN(FusionLinear(qBProjNodeParam, &qBProjNode.operation));
    qBProjNode.inTensorIds = {
        GetTensorIdx(tensorMap, "latent_q_norm"),
        GetTensorIdx(tensorMap, "in_q_proj_b_weight"),
        GetTensorIdx(tensorMap, "in_q_proj_b_scale"),
        GetTensorIdx(tensorMap, "in_q_proj_b_offset"),
        GetTensorIdx(tensorMap, "in_q_proj_b_descale"),
        GetTensorIdx(tensorMap, "in_q_proj_b_bias"),
        GetTensorIdx(tensorMap, "in_q_proj_b_compress_idx"),
    };
    qBProjNode.outTensorIds = {GetTensorIdx(tensorMap, "q_lora_out")};
    opGraph.nodes.push_back(qBProjNode);
    ATB_SPEED_LOG_DEBUG("MLA proj_q_b calculation success");
    return atb::NO_ERROR;
}


template <typename NormParamType>
atb::Status AddSplitQNode(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node splitQNode;
    atb::infer::SplitParam splitQParam = {(param.isFA ? 3 : 2), 2, {param.qkNopeHeadDim, param.qkRopeHeadDim}};
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(splitQParam, &splitQNode.operation));
    if (param.qLoraRank == 0) {    // translatedlite
        splitQNode.inTensorIds = {GetTensorIdx(tensorMap, "latent_q")};
    } else {
        splitQNode.inTensorIds = {GetTensorIdx(tensorMap, param.enablePreprocessLcocTp ?
            "fused_q_lora_out" : "q_lora_out")};
    }
    splitQNode.inTensorReshapeFuncs.resize(splitQNode.inTensorIds.size());
    // splitQNode.outTensorIds = {GetTensorIdxList(tensorMap, {"nope_q", "rope_q"})};
    splitQNode.outTensorIds = {
        GetTensorIdx(tensorMap, "nope_q"), GetTensorIdx(tensorMap, "rope_q"),
    };
    if (param.isFA) {
        splitQNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
                newShape.dimNum = 4; // 4: dimNum
                newShape.dims[0] = oldShape.dims[0];
                newShape.dims[1] = oldShape.dims[1];
                newShape.dims[2] = param.selfAttentionParam.headNum; // 2: dim id
                newShape.dims[3] = param.qkNopeHeadDim + param.qkRopeHeadDim; // 3: dim id
            };
    } else {
        splitQNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            newShape.dimNum = 3; // 3: dimNum
            newShape.dims[0] = oldShape.dims[0];
            newShape.dims[1] = param.selfAttentionParam.headNum;
            newShape.dims[2] = param.qkNopeHeadDim + param.qkRopeHeadDim; // 2: dim id
        };
    }
    opGraph.nodes.push_back(splitQNode);
    ATB_SPEED_LOG_DEBUG("MLA split q calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddReprojQNode(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node qReprojNode;
    atb_speed::common::FusionLinearParam qReprojNodeParam;
    qReprojNodeParam.isBF16 = param.isBF16;
    qReprojNodeParam.hasBias = param.selfAttnHasBias;
    qReprojNodeParam.quantType = GetLinearQuantType(
        param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED \
            ? param.packQuantType : param.denseQuantType,
        param.attnLinearQuantType[KV_PROJ_B_FOR_Q_LINEAR_INDEX], false);
    qReprojNodeParam.quantGroupSize = param.quantGroupSize;
    qReprojNodeParam.transposeType = false;
    qReprojNodeParam.enEin = true;
    CHECK_OPERATION_STATUS_RETURN(FusionLinear(qReprojNodeParam, &qReprojNode.operation));
    qReprojNode.inTensorIds = {
        GetTensorIdx(tensorMap, "nope_q"),
        GetTensorIdx(tensorMap, "in_k_proj_b_for_q_weight"),
        GetTensorIdx(tensorMap, "in_k_proj_b_for_q_scale"),
        GetTensorIdx(tensorMap, "in_k_proj_b_for_q_offset"),
        GetTensorIdx(tensorMap, "in_k_proj_b_for_q_descale"),
        GetTensorIdx(tensorMap, "in_k_proj_b_for_q_bias"),
        GetTensorIdx(tensorMap, "in_k_proj_b_for_q_compress_idx"),
    };
    qReprojNode.outTensorIds = {GetTensorIdx(tensorMap, "reproj_nope_q")};
    opGraph.nodes.push_back(qReprojNode);
    ATB_SPEED_LOG_DEBUG("MLA reproj q calculation success");
    return atb::NO_ERROR;
}


template <typename NormParamType>
atb::Status AddLAttnKVAProjNode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node kvAProjNode;
    atb_speed::common::FusionLinearParam kvAProjNodeParam;
    kvAProjNodeParam.isBF16 = param.isBF16;
    kvAProjNodeParam.hasBias = param.selfAttnHasBias;
    kvAProjNodeParam.quantType = GetLinearQuantType(
        param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED \
            ? param.packQuantType : param.denseQuantType,
        param.attnLinearQuantType[KV_PROJ_A_LINEAR_INDEX], true);
    kvAProjNodeParam.quantGroupSize = param.quantGroupSize;
    kvAProjNodeParam.transposeType = param.attnLinearTransposeType[KV_PROJ_A_LINEAR_INDEX];
    kvAProjNode.inTensorIds = {
        GetTensorIdx(tensorMap, "in_input_norm"),
        GetTensorIdx(tensorMap, "in_kv_proj_with_mqa_weight"),
        GetTensorIdx(tensorMap, "in_kv_proj_with_mqa_scale"),
        GetTensorIdx(tensorMap, "in_kv_proj_with_mqa_offset"),
        GetTensorIdx(tensorMap, "in_kv_proj_with_mqa_descale"),
        GetTensorIdx(tensorMap, "in_kv_proj_with_mqa_bias"),
        GetTensorIdx(tensorMap, "in_kv_proj_with_mqa_compress_idx"),
    };
    kvAProjNode.outTensorIds = {GetTensorIdx(tensorMap, "latent_kv")};
    CHECK_OPERATION_STATUS_RETURN(FusionLinear(kvAProjNodeParam, &kvAProjNode.operation));
    opGraph.nodes.push_back(kvAProjNode);
    ATB_SPEED_LOG_DEBUG("MLA proj_kv_a calculation success");
    return atb::NO_ERROR;
}


template <typename NormParamType>
atb::Status AddSplitKNode(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node splitKNode;
    atb::infer::SplitParam splitKParam = {(param.isFA ? 2 : 1), 2, {param.kvLoraRank, param.qkRopeHeadDim}};
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(splitKParam, &splitKNode.operation));
    splitKNode.inTensorIds = {GetTensorIdx(tensorMap, param.enablePreprocessLcocTp ?
        "intermediate_kv_unpadding_all" : "latent_kv")};
    splitKNode.outTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_kv"), GetTensorIdx(tensorMap, "rope_k"),
    };
    opGraph.nodes.push_back(splitKNode);
    ATB_SPEED_LOG_DEBUG("MLA spilt_k calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddLAttnKVNormNode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node kvNormNode;
    kvNormNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_kv"), GetTensorIdx(tensorMap, "in_kv_proj_a_layernorm_weight")
    };
    kvNormNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_kv")};
    if (!param.isFA) {
        kvNormNode.inTensorReshapeFuncs.resize(kvNormNode.inTensorIds.size());
        kvNormNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            UnSqueezeHeadNumHeadDim(oldShape, newShape, param.kvLoraRank);
        };
    }
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.normParamType, &kvNormNode.operation));
    opGraph.nodes.push_back(kvNormNode);
    ATB_SPEED_LOG_DEBUG("MLA kv norm calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddSfaRopeCpNode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node ropeQNode;
    atb_speed::common::AclnnRopeParam ropeParam;
    ropeParam.mode = 1;
    ropeQNode.operation = new atb_speed::common::AclnnRopeOperation("RopeQNode", ropeParam);
    ropeQNode.inTensorIds = {
        GetTensorIdx(tensorMap, "rope_q"),
        GetTensorIdx(tensorMap, "in_cos_embed"), GetTensorIdx(tensorMap, "in_sin_embed")
    };

    ropeQNode.outTensorIds = {
        GetTensorIdx(tensorMap, "rope_q_o")
    };
    ropeQNode.inTensorReshapeFuncs.resize(ropeQNode.inTensorIds.size());
    for(auto i = 0; i < ropeQNode.inTensorIds.size(); i++){
        ropeQNode.inTensorReshapeFuncs[i] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            UnSqueezeHeadNumHeadDim(oldShape, newShape, param.qkRopeHeadDim);
        };
    }
    opGraph.nodes.push_back(ropeQNode);
    atb::Node ropeKNode;
    ropeKNode.operation = new atb_speed::common::AclnnRopeOperation("RopeKNode", ropeParam);
    ropeKNode.inTensorIds = {
        GetTensorIdx(tensorMap, "rope_k"),
        GetTensorIdx(tensorMap, "in_cos_embed_allgather_s"), GetTensorIdx(tensorMap, "in_sin_embed_allgather_s")
    };

    ropeKNode.outTensorIds = {
        GetTensorIdx(tensorMap, "rope_k_o")
    };
    ropeKNode.inTensorReshapeFuncs.resize(ropeKNode.inTensorIds.size());
    for(auto i = 0; i < ropeKNode.inTensorIds.size(); i++){
        ropeKNode.inTensorReshapeFuncs[i] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            UnSqueezeHeadNumHeadDim(oldShape, newShape, param.qkRopeHeadDim);
        };
    }
    opGraph.nodes.push_back(ropeKNode);
    ATB_SPEED_LOG_DEBUG("Sfa rope calculation success");

    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddLAttnRopeNode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node ropeNode;
    atb::infer::RopeParam ropeParam;
    ropeParam.rotaryCoeff = param.ropeParam.rotaryCoeff;
    CreateOperation(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = {
        GetTensorIdx(tensorMap, "rope_q"), GetTensorIdx(tensorMap, "rope_k"),
        GetTensorIdx(tensorMap, "in_cos_embed"), GetTensorIdx(tensorMap, "in_sin_embed"),
        GetTensorIdx(tensorMap, "in_seq_len")
    };

    ropeNode.outTensorIds = {
        GetTensorIdx(tensorMap, "rope_q_o"), GetTensorIdx(tensorMap, "rope_k_o"),
    };
    if (param.contextParallelInfo.IsEnabled() && param.isPrefill) {
        ropeNode.outTensorIds = {
            GetTensorIdx(tensorMap, "rope_q_o"), GetTensorIdx(tensorMap, "rope_k_o_cp"),
        };
    }
    ropeNode.inTensorReshapeFuncs.resize(ropeNode.inTensorIds.size());
    ropeNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        SqueezeHeadNumHeadDim(oldShape, newShape);
    };
    opGraph.nodes.push_back(ropeNode);
    ATB_SPEED_LOG_DEBUG("MLA rope calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddReshapeAndCacheNode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node reshapeAndCacheNode;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.reshapeCacheParm, &reshapeAndCacheNode.operation));
    reshapeAndCacheNode.inTensorIds = {
        EnableFA3Quant(param) ? GetTensorIdx(tensorMap, "intermediate_kv_int8") :
        GetTensorIdx(tensorMap, "intermediate_kv"), GetTensorIdx(tensorMap, "rope_k_o"),
        GetTensorIdx(tensorMap, "in_k_cache"), GetTensorIdx(tensorMap, "in_k_rope_cache"),
        GetTensorIdx(tensorMap, "in_slots_in_pa_or_logn_in_fa"),
    };
    reshapeAndCacheNode.outTensorIds = {
        GetTensorIdx(tensorMap, "in_k_cache"),
        GetTensorIdx(tensorMap, "in_k_rope_cache"),
    };
    reshapeAndCacheNode.inTensorReshapeFuncs.resize(reshapeAndCacheNode.inTensorIds.size());
    reshapeAndCacheNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        UnSqueezeHeadNumHeadDim(oldShape, newShape, param.qkRopeHeadDim);
    };
    opGraph.nodes.push_back(reshapeAndCacheNode);
    return atb::NO_ERROR;
}


template <typename NormParamType>
atb::Status AddReprojVNode(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node vReprojNode;
    atb_speed::common::FusionLinearParam vReprojNodeParam;
    vReprojNodeParam.isBF16 = param.isBF16;
    vReprojNodeParam.hasBias = param.selfAttnHasBias;
    vReprojNodeParam.quantType = GetLinearQuantType(
        param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED \
            ? param.packQuantType : param.denseQuantType,
        param.attnLinearQuantType[KV_PROJ_B_FOR_V_LINEAR_INDEX], false);
    vReprojNodeParam.quantGroupSize = param.quantGroupSize;
    vReprojNodeParam.transposeType = false;
    CHECK_OPERATION_STATUS_RETURN(FusionLinear(vReprojNodeParam, &vReprojNode.operation));
    vReprojNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_sfa_out"), GetTensorIdx(tensorMap, "in_v_proj_b_for_o_weight"),
        GetTensorIdx(tensorMap, "in_v_proj_b_for_o_scale"), GetTensorIdx(tensorMap, "in_v_proj_b_for_o_offset"),
        GetTensorIdx(tensorMap, "in_v_proj_b_for_o_descale"), GetTensorIdx(tensorMap, "in_v_proj_b_for_o_bias"),
        GetTensorIdx(tensorMap, "in_v_proj_b_for_o_compress_idx"),
    };
    vReprojNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_reproj")};
    opGraph.nodes.push_back(vReprojNode);
    ATB_SPEED_LOG_DEBUG("MLA reproj v calculation success");
    return atb::NO_ERROR;
}


template <typename NormParamType>
atb::Status AddEinReprojVNode(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node vReprojNode;

    atb::infer::LinearParam linearParam;
    linearParam.transposeB = false;
    linearParam.outDataType = ACL_DT_UNDEFINED; // param.isBF16 ? ACL_BF16 : ACL_FLOAT16;
    linearParam.matmulType = atb::infer::LinearParam::MATMUL_EIN_SUM;

    // translatedLinear
    if (param.selfAttnHasBias) {
        linearParam.hasBias = true;
        vReprojNode.inTensorIds = GetTensorIdxList(tensorMap, {"intermediate_self_attention", "in_v_proj_b_for_o_weight", "in_v_proj_b_for_o_bias"});
    } else {
        linearParam.hasBias = false;
        vReprojNode.inTensorIds = GetTensorIdxList(tensorMap, {"intermediate_self_attention", "in_v_proj_b_for_o_weight"});
    }

    // translatedvReprojNode outTensor
    vReprojNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_reproj_t")};

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(linearParam, &vReprojNode.operation));

    opGraph.nodes.push_back(vReprojNode);
    ATB_SPEED_LOG_DEBUG("MLA reproj v calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddEINLAttnKProjBNode(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node kProjBNode;

    atb::infer::LinearParam linearParam;
    linearParam.transposeB = false;
    linearParam.outDataType = ACL_DT_UNDEFINED; // param.isBF16 ? ACL_BF16 : ACL_FLOAT16;
    linearParam.matmulType = atb::infer::LinearParam::MATMUL_EIN_SUM;

    if (param.selfAttnHasBias) {
        linearParam.hasBias = true;
        kProjBNode.inTensorIds = GetTensorIdxList(tensorMap, {"nope_q", "in_k_proj_b_for_q_weight", "in_k_proj_b_for_q_bias"});
    } else {
        linearParam.hasBias = false;
        kProjBNode.inTensorIds = GetTensorIdxList(tensorMap, {"nope_q", "in_k_proj_b_for_q_weight"});
    }
    kProjBNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_q_t")};

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(linearParam, &kProjBNode.operation));
    opGraph.nodes.push_back(kProjBNode);
    ATB_SPEED_LOG_DEBUG("MLA proj_k_b calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddLAttnKProjBNode(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node kProjBNode;
    atb_speed::common::FusionLinearParam kProjBNodeParam;
    kProjBNodeParam.isBF16 = param.isBF16;
    kProjBNodeParam.quantType = atb_speed::common::LinearQuantType::NO_QUANT;
    kProjBNodeParam.transposeType = 0;
    CHECK_OPERATION_STATUS_RETURN(FusionLinear(kProjBNodeParam, &kProjBNode.operation));
    kProjBNode.inTensorIds = {
        GetTensorIdx(tensorMap, "nope_q_t"),
        GetTensorIdx(tensorMap, "in_k_proj_b_for_q_weight"),
        GetTensorIdx(tensorMap, "in_k_proj_b_for_q_scale"),
        GetTensorIdx(tensorMap, "in_k_proj_b_for_q_offset"),
        GetTensorIdx(tensorMap, "in_k_proj_b_for_q_descale"),
        GetTensorIdx(tensorMap, "in_k_proj_b_for_q_bias"),
        GetTensorIdx(tensorMap, "in_k_proj_b_for_q_compress_idx"),
    };
    kProjBNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_q")};

    opGraph.nodes.push_back(kProjBNode);
    ATB_SPEED_LOG_DEBUG("MLA proj_k_b calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddLAttnKProjBHistoryNode(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node kProjBNode;
    atb_speed::common::FusionLinearParam kProjBNodeParam;
    kProjBNodeParam.isBF16 = param.isBF16;
    kProjBNodeParam.hasBias = param.selfAttnHasBias;
    kProjBNodeParam.quantType = GetLinearQuantType(
        param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED
            ? param.packQuantType
            : param.denseQuantType,
        param.attnLinearQuantType[KV_PROJ_B_FOR_Q_LINEAR_INDEX], false);
    kProjBNodeParam.quantGroupSize = param.quantGroupSize;
    kProjBNodeParam.transposeType = param.attnLinearTransposeType[KV_PROJ_B_FOR_Q_LINEAR_INDEX];
    CHECK_OPERATION_STATUS_RETURN(FusionLinear(kProjBNodeParam, &kProjBNode.operation));
    kProjBNode.inTensorIds = {
        GetTensorIdx(tensorMap, "in_history_compressed_kv"),
        GetTensorIdx(tensorMap, "in_k_proj_b_for_q_weight"),
        GetTensorIdx(tensorMap, "in_k_proj_b_for_q_scale"),
        GetTensorIdx(tensorMap, "in_k_proj_b_for_q_offset"),
        GetTensorIdx(tensorMap, "in_k_proj_b_for_q_descale"),
        GetTensorIdx(tensorMap, "in_k_proj_b_for_q_bias"),
        GetTensorIdx(tensorMap, "in_k_proj_b_for_q_compress_idx"),
    };
    kProjBNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_k_nope_history")};

    kProjBNode.inTensorReshapeFuncs.resize(kProjBNode.inTensorIds.size());
    kProjBNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2: dim id
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2]; // 2: dim id
    };

    opGraph.nodes.push_back(kProjBNode);
    ATB_SPEED_LOG_DEBUG("MLA proj_k_b AddLAttnKProjBHistoryNode calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddLAttnVProjBBeforeTransposeNode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node transposeVProjBeforeNode;
    atb::infer::TransposeParam transposeVProjBBeforeParam;
    transposeVProjBeforeNode.inTensorIds = {GetTensorIdx(tensorMap, "nope_q")};
    transposeVProjBeforeNode.outTensorIds = {GetTensorIdx(tensorMap, "nope_q_t")};
    transposeVProjBBeforeParam.perm = {1, 0, 2};
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(transposeVProjBBeforeParam,
        &transposeVProjBeforeNode.operation));
    opGraph.nodes.push_back(transposeVProjBeforeNode);
    ATB_SPEED_LOG_DEBUG("MLA proj_k_b input transpose calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddLAttnVProjBAfterTransposeNode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node transposeVProjBAfterNode;
    atb::infer::TransposeParam transposeVProjBAfterParam;
    transposeVProjBAfterNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_q")};
    transposeVProjBAfterNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_q_t")};
    transposeVProjBAfterParam.perm = {1, 0, 2};
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(transposeVProjBAfterParam,
        &transposeVProjBAfterNode.operation));
    opGraph.nodes.push_back(transposeVProjBAfterNode);
    ATB_SPEED_LOG_DEBUG("MLA proj_k_b output transpose calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddLAttnVProjBHistoryNode(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node vProjBNode;
    atb_speed::common::FusionLinearParam vProjBNodeParam;
    vProjBNodeParam.isBF16 = param.isBF16;
    vProjBNodeParam.hasBias = param.selfAttnHasBias;
    vProjBNodeParam.quantType = GetLinearQuantType(
        param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED
            ? param.packQuantType
            : param.denseQuantType,
        param.attnLinearQuantType[KV_PROJ_B_FOR_V_LINEAR_INDEX], false);
    vProjBNodeParam.quantGroupSize = param.quantGroupSize;
    vProjBNodeParam.transposeType = param.attnLinearTransposeType[KV_PROJ_B_FOR_V_LINEAR_INDEX];
    CHECK_OPERATION_STATUS_RETURN(FusionLinear(vProjBNodeParam, &vProjBNode.operation));
    vProjBNode.inTensorIds = {
        GetTensorIdx(tensorMap, "in_history_compressed_kv"),
        GetTensorIdx(tensorMap, "temp_v_proj_b"),
        GetTensorIdx(tensorMap, "in_v_proj_b_for_o_scale"),
        GetTensorIdx(tensorMap, "in_v_proj_b_for_o_offset"),
        GetTensorIdx(tensorMap, "in_v_proj_b_for_o_descale"),
        GetTensorIdx(tensorMap, "in_v_proj_b_for_o_bias"),
        GetTensorIdx(tensorMap, "in_v_proj_b_for_o_compress_idx"),
    };
    vProjBNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_v_mha_history")};
    vProjBNode.inTensorReshapeFuncs.resize(vProjBNode.inTensorIds.size());
    vProjBNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2: dim num
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2]; // 2: dim id
    };

    opGraph.nodes.push_back(vProjBNode);
    ATB_SPEED_LOG_DEBUG("MLA proj_v_b AddLAttnVProjBHistoryNode calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status SetSelfOutLinearParallelParam(const LatentAttentionParam<NormParamType> &param,
    atb_speed::common::LinearParallelParam &selfOutLinearParam)
{
    selfOutLinearParam.parallelType = atb_speed::common::ROW_PARALLEL;
    selfOutLinearParam.fusionLinearParam.isBF16 = param.isBF16;
    selfOutLinearParam.fusionLinearParam.hasBias = param.selfAttnHasBias && !selfOutLinearParam.biasAfterSync;
    selfOutLinearParam.fusionLinearParam.quantType = GetLinearQuantType(
        param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED \
            ? param.packQuantType : param.denseQuantType,
        param.attnLinearQuantType[O_LINEAR_INDEX], false);
    if (selfOutLinearParam.fusionLinearParam.quantType == LinearQuantType::LINEAR_W8A8_QUANT &&
        param.enableExtraOprojTp) {
        selfOutLinearParam.fusionLinearParam.quantType = LinearQuantType::LINEAR_W8A8_DEQUANT;
    }
    selfOutLinearParam.fusionLinearParam.quantGroupSize = param.quantGroupSize;
    selfOutLinearParam.fusionLinearParam.transposeType = param.attnLinearTransposeType[O_LINEAR_INDEX];
    selfOutLinearParam.tensorParallelInfo = param.selfOutLinearTensorParallelInfo;
    selfOutLinearParam.supportLcoc = param.enableLcoc;

    selfOutLinearParam.innerTensorParallelInfo = param.selfOutLinearInnerTensorParallelInfo;

    if (param.selfOutLinearInnerTensorParallelInfo.rankIds.size() == 0) {
        std::stringstream ss;
        ss << "Cannot be devided by zero. Param attnOprojTpSize is zero!" << std::endl;
        throw std::runtime_error(ss.str());
    }
    selfOutLinearParam.innerTpShape = \
        param.qkNopeHeadDim * param.selfAttentionParam.headNum / \
        param.selfOutLinearInnerTensorParallelInfo.rankIds.size();

    selfOutLinearParam.fusionLinearParam.isPrefill = param.isPrefill;
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddSelfOutLinearParallelNode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node selfOutLinearParallelNode;
    atb_speed::common::LinearParallelParam selfOutLinearParam;
    SetSelfOutLinearParallelParam(param, selfOutLinearParam);
    CHECK_OPERATION_STATUS_RETURN(LinearParallel(selfOutLinearParam, &selfOutLinearParallelNode.operation));
    selfOutLinearParallelNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_reproj_t"),
        GetTensorIdx(tensorMap, "in_attn_out_weight"),
        GetTensorIdx(tensorMap, "in_attn_out_scale"),
        GetTensorIdx(tensorMap, "in_attn_out_offset"),
        GetTensorIdx(tensorMap, "in_attn_out_descale"),
        GetTensorIdx(tensorMap, "in_attn_out_bias"),
        GetTensorIdx(tensorMap, "in_attn_out_compress_idx"),
    };
    selfOutLinearParallelNode.inTensorReshapeFuncs.resize(selfOutLinearParallelNode.inTensorIds.size());
    if (!param.isFA) {
        selfOutLinearParallelNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            SqueezeHeadNumHeadDim(oldShape, newShape);
        };
    }
    selfOutLinearParallelNode.outTensorIds = {GetTensorIdx(tensorMap, "out")};
    opGraph.nodes.push_back(selfOutLinearParallelNode);
    ATB_SPEED_LOG_DEBUG("MLA o_proj calculation success");
    return atb::NO_ERROR;
}

atb::Status AddQuantOprojNode(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    // quant
    atb::Node inputQuantNode;
    atb::infer::ElewiseParam inputQuantParam;
    inputQuantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT_PER_CHANNEL;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(inputQuantParam, &inputQuantNode.operation));
    inputQuantNode.inTensorIds = GetTensorIdxList(tensorMap, {"intermediate_self_attention_padding",
        "in_attn_out_scale", "in_attn_out_offset"});
    inputQuantNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_self_attention_padding_quant")};
    opGraph.nodes.push_back(inputQuantNode);
    ATB_SPEED_LOG_DEBUG("MLA Quant O calculation success");
    return atb::NO_ERROR;
}

atb::Status AddOprojAllToAllPaddingNode(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node gatherNode;
    atb::infer::GatherParam gatherParam;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode.operation));

    gatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_self_attention",
                                                                             "in_attn_padding_idx"});
    gatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_self_attention_padding"});
    opGraph.nodes.push_back(gatherNode);
    ATB_SPEED_LOG_DEBUG("MLA Wo AllToAll Padding calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status SetTPAllGatherNode(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> tensorMap)
{
    atb::Node allGatherNode;
    atb::infer::AllGatherParam allGatherParam;
    allGatherParam.rank = param.attnTpRank;
    allGatherParam.rankSize = param.attnTpSize;
    allGatherParam.backend = param.attnTpBackend;
    allGatherParam.commDomain = param.attnTpDomain;
    allGatherParam.rankTableFile = param.attnTpRankTableFile;
    allGatherParam.hcclComm = param.hcclComm;

    CreateOperation(allGatherParam, &allGatherNode.operation);

    allGatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {param.enablePreprocessLcocTp ?
        "latent_kv" : "latent_qkv"});
    allGatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {param.enablePreprocessLcocTp ?
        "intermediate_kv_all" : "intermediate_qkv_all"});

    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsBeforeComm(opGraph));
    opGraph.nodes.push_back(allGatherNode);
    CHECK_OPERATION_STATUS_RETURN(common::AddDapEventsAfterComm(opGraph));
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status SetFFNUnPadding(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> tensorMap)
{
    atb::Node gatherNode;
    atb::infer::GatherParam gatherParam;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode.operation));

    gatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, \
        {param.ffnAllGather ?  (param.enablePreprocessLcocTp ?
        "intermediate_kv_all" : "intermediate_qkv_all") : "latent_qkv",
        "in_ffn_unpadding_idx"});
    gatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, \
        {param.enablePreprocessLcocTp ? "intermediate_kv_unpadding_all" : "intermediate_qkv_unpadding_all"});
    // intermediate_layer_out
    if (param.ffnAllGather) {
        gatherNode.inTensorReshapeFuncs.resize(gatherNode.inTensorIds.size());
        gatherNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
                newShape.dimNum = 2; // 2: dimNum
                newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
                newShape.dims[1] = oldShape.dims[2]; // 2: dim 2
        };
    }
    opGraph.nodes.push_back(gatherNode);
    return atb::NO_ERROR;
}

atb::Status SetAttnOprojPrefetch(atb::GraphParam &opGraph, std::map<std::string, uint32_t> tensorMap)
{
    atb::Node computeRecordNode;
    computeRecordNode.inTensorIds = {};
    computeRecordNode.outTensorIds = {};
    CHECK_OPERATION_STATUS_RETURN(atb_speed::EventManager::GetInstance().RecordEvent(
        computeRecordNode.operation,
        atb_speed::EventAction::PUSH,
        atb_speed::common::CMO_OPROJ));
    opGraph.nodes.push_back(computeRecordNode);

    atb::Node commWaitNode;
    commWaitNode.inTensorIds = {};
    commWaitNode.outTensorIds = {};
    CHECK_OPERATION_STATUS_RETURN(atb_speed::EventManager::GetInstance().WaitEvent(
        commWaitNode.operation,
        atb_speed::EventAction::POP,
        atb_speed::common::CMO_OPROJ));
    atb::SetExecuteStreamId(commWaitNode.operation, 1);
    opGraph.nodes.push_back(commWaitNode);

    atb::Node cmoNode3;
    cmoNode3.operation = new atb_speed::common::AclrtCmoAsyncOperation("AclrtCmoAsync");
    cmoNode3.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {
        "in_attn_out_weight"
    });
    cmoNode3.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {});
    atb::SetExecuteStreamId(cmoNode3.operation, 1);
    opGraph.nodes.push_back(cmoNode3);

    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddLAttnQNormRecalNode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node qNormRecalNode;

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.normParamType, &qNormRecalNode.operation));
    qNormRecalNode.inTensorIds = {GetTensorIdx(tensorMap, "latent_q"),
                            GetTensorIdx(tensorMap, "in_q_proj_a_layernorm_weight")};
    qNormRecalNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_indexer_qa_norm")};
    opGraph.nodes.push_back(qNormRecalNode);
    ATB_SPEED_LOG_DEBUG("Sparse Latent Attention QNorm recalculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddIndexerInputNormRecalNode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node InputNormRecalNode;

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.normParamType, &InputNormRecalNode.operation));
    InputNormRecalNode.inTensorIds = {GetTensorIdx(tensorMap, "in_input"), GetTensorIdx(tensorMap, "in_norm_weight")};
    InputNormRecalNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_indexer_input_norm")};
    opGraph.nodes.push_back(InputNormRecalNode);
    ATB_SPEED_LOG_DEBUG("Sparse Latent Attention QNorm recalculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddLAttnQNormNode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node qNormNode;
    if (UseExtraQuant(param, Q_PROJ_B_LINEAR_INDEX)) {  // W8A8
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.normQuantParamType, &qNormNode.operation));
        qNormNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "latent_q"));
        qNormNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_q_proj_a_layernorm_weight"));
        qNormNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_q_proj_a_layernorm_bias"));
        qNormNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_q_proj_b_scale"));
        qNormNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_q_proj_b_offset"));
    } else {
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.normParamType, &qNormNode.operation));
        qNormNode.inTensorIds = {GetTensorIdx(tensorMap, "latent_q"),
                                GetTensorIdx(tensorMap, "in_q_proj_a_layernorm_weight")};
    }
    qNormNode.outTensorIds = {GetTensorIdx(tensorMap, "fused_latent_q_norm")};
    opGraph.nodes.push_back(qNormNode);
    ATB_SPEED_LOG_DEBUG("Sparse Latent Attention QNorm calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddFusedQBNode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node qBProjNode;
    atb::infer::LinearParallelParam qBProjNodeParam;
    qBProjNodeParam.transWeight = param.attnLinearTransposeType[Q_PROJ_B_LINEAR_INDEX] != -1 ?
        param.attnLinearTransposeType[Q_PROJ_B_LINEAR_INDEX] : true;
    qBProjNodeParam.type = atb::infer::LinearParallelParam::ParallelType::ALL_GATHER_LINEAR;
    qBProjNodeParam.rank = param.lcocAttnTpRank;
    qBProjNodeParam.rankSize = param.lcocAttnTpRankSize;
    qBProjNodeParam.hcclComm = param.lcocHcclComm;
    qBProjNodeParam.backend = param.lcocAttnTpBackend;
    qBProjNodeParam.commDomain = param.lcocAttnTpDomain;
    qBProjNodeParam.quantType = atb::infer::LinearParallelParam::QuantType::QUANT_TYPE_PER_CHANNEL;
    qBProjNodeParam.quantGroupSize = param.quantGroupSize;
    qBProjNodeParam.outDataType = aclDataType::ACL_FLOAT16;

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(qBProjNodeParam, &qBProjNode.operation));
    qBProjNode.inTensorIds = {
        GetTensorIdx(tensorMap, "fused_latent_q_norm"),
        GetTensorIdx(tensorMap, "in_q_proj_b_weight"),
        GetTensorIdx(tensorMap, "in_q_proj_b_bias"),
        GetTensorIdx(tensorMap, "in_q_proj_b_descale"),
    };
    qBProjNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_q_b")};
    opGraph.nodes.push_back(qBProjNode);
    ATB_SPEED_LOG_DEBUG("Fused QBNode calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status SetFFNQUnPadding(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> tensorMap)
{
    atb::Node gatherNode;
    atb::infer::GatherParam gatherParam;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode.operation));

    gatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, \
        {param.ffnAllGather ? "intermediate_q_b" : "fused_latent_q_norm", "in_ffn_unpadding_idx"});
    gatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"fused_q_lora_out"});
    opGraph.nodes.push_back(gatherNode);
    ATB_SPEED_LOG_DEBUG("FFN Q unpadding calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddSelfFusedOutLinearParallelNode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node selfFusedOutLinearParallelNode;
    atb::infer::LinearParallelParam selfFusedOutLinearParam;
    selfFusedOutLinearParam.transWeight = param.attnLinearTransposeType[Q_PROJ_B_LINEAR_INDEX] != -1 ?
        param.attnLinearTransposeType[Q_PROJ_B_LINEAR_INDEX] : true;
    selfFusedOutLinearParam.type = atb::infer::LinearParallelParam::ParallelType::LINEAR_REDUCE_SCATTER;
    selfFusedOutLinearParam.rank = param.lcocAttnTpRank;
    selfFusedOutLinearParam.rankSize = param.lcocAttnTpRankSize;
    selfFusedOutLinearParam.hcclComm = param.lcocHcclComm;
    selfFusedOutLinearParam.backend = param.lcocAttnTpBackend;
    selfFusedOutLinearParam.commDomain = param.lcocAttnTpDomain;
    selfFusedOutLinearParam.quantType = atb::infer::LinearParallelParam::QuantType::QUANT_TYPE_PER_CHANNEL;
    selfFusedOutLinearParam.quantGroupSize = param.quantGroupSize;
    selfFusedOutLinearParam.outDataType = aclDataType::ACL_FLOAT16;

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(selfFusedOutLinearParam,
        &selfFusedOutLinearParallelNode.operation));
    selfFusedOutLinearParallelNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_self_attention_padding_quant"),
        GetTensorIdx(tensorMap, "in_attn_out_weight"),
        GetTensorIdx(tensorMap, "in_attn_out_bias"),
        GetTensorIdx(tensorMap, "in_attn_out_descale"),
    };
    selfFusedOutLinearParallelNode.outTensorIds = {GetTensorIdx(tensorMap, "out")};
    selfFusedOutLinearParallelNode.inTensorReshapeFuncs.resize(selfFusedOutLinearParallelNode.inTensorIds.size());
    if (!param.isFA) {
        selfFusedOutLinearParallelNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            SqueezeHeadNumHeadDim(oldShape, newShape);
        };
    }
    opGraph.nodes.push_back(selfFusedOutLinearParallelNode);
    ATB_SPEED_LOG_DEBUG("Fused QBNode calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status PreprocessKV(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    CHECK_OPERATION_STATUS_RETURN(AddEINLAttnKProjBNode(param, opGraph, tensorMap));
    // CHECK_OPERATION_STATUS_RETURN(AddLAttnVProjBBeforeTransposeNode(param, opGraph, tensorMap));
    // CHECK_OPERATION_STATUS_RETURN(AddLAttnKProjBNode(param, opGraph, tensorMap));
    // CHECK_OPERATION_STATUS_RETURN(AddLAttnVProjBAfterTransposeNode(param, opGraph, tensorMap));
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddLAttnQProjRecalNode(const LatentAttentionParam<NormParamType> &param,
                                atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node qProjaRecalNode;
    atb_speed::common::FusionLinearParam qProjaRecalParam;
    qProjaRecalParam.isBF16 = param.isBF16;
    qProjaRecalParam.hasBias = param.selfAttnHasBias;
    qProjaRecalParam.quantType = GetLinearQuantType(
        param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED ? \
            param.packQuantType : param.denseQuantType,
        param.attnLinearQuantType[Q_PROJ_A_LINEAR_INDEX], true);
    if (qProjaRecalParam.quantType == LinearQuantType::LINEAR_W8A8_DYNAMIC_DEQUANT) {
        qProjaRecalParam.quantType = LinearQuantType::LINEAR_W8A8_DYNAMIC_QUANT;
    }
   qProjaRecalParam.quantGroupSize = param.quantGroupSize;
    qProjaRecalNode.inTensorIds = {
        GetTensorIdx(tensorMap, "in_input_norm"),
        GetTensorIdx(tensorMap, "in_q_proj_a_recompte_weight"),
        GetTensorIdx(tensorMap, "in_q_proj_a_recompte_scale"),
        GetTensorIdx(tensorMap, "in_q_proj_a_recompte_offset"),
        GetTensorIdx(tensorMap, "in_q_proj_a_recompte_descale"),
        GetTensorIdx(tensorMap, "in_q_proj_a_recompte_bias"),
        GetTensorIdx(tensorMap, "in_q_proj_a_recompte_compress_idx"),
    };
    qProjaRecalNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_indexer_qa")};
    CHECK_OPERATION_STATUS_RETURN(FusionLinear(qProjaRecalParam, &qProjaRecalNode.operation));
    opGraph.nodes.push_back(qProjaRecalNode);
    ATB_SPEED_LOG_DEBUG("MLA proj_qkv_a calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddIndexerQBNode(const LatentAttentionParam<NormParamType> &param,
                                atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node qProjbNode;
    atb_speed::common::FusionLinearParam qProjbParam;
    qProjbParam.isBF16 = param.isBF16;
    qProjbParam.quantType = atb_speed::common::LinearQuantType::NO_QUANT;
    qProjbParam.quantGroupSize = param.quantGroupSize;
    qProjbParam.transposeType = param.attnLinearTransposeType[Q_PROJ_B_LINEAR_INDEX];
    qProjbNode.inTensorIds = {
        GetTensorIdx(tensorMap, {"intermediate_indexer_qa_norm"}),
        GetTensorIdx(tensorMap, "in_indexer_proj_wq_b_weight"),
        GetTensorIdx(tensorMap, "in_indexer_proj_wq_b_scale"),
        GetTensorIdx(tensorMap, "in_indexer_proj_wq_b_offset"),
        GetTensorIdx(tensorMap, "in_indexer_proj_wq_b_descale"),
        GetTensorIdx(tensorMap, "in_indexer_proj_wq_b_bias"),
        GetTensorIdx(tensorMap, "in_indexer_proj_wq_b_compress_idx"),
    };
    qProjbNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_indexer_qb")};
    CHECK_OPERATION_STATUS_RETURN(FusionLinear(qProjbParam, &qProjbNode.operation));
    opGraph.nodes.push_back(qProjbNode);
    ATB_SPEED_LOG_DEBUG("MLA proj_qkv_a calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddIndexerQBSplitNode(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node splitQBNode;
    atb::infer::SplitParam splitQBParam = {(param.isFA ? 3 : 2), 2, {param.qkRopeHeadDim, param.index_head_dim - param.qkRopeHeadDim}};
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(splitQBParam, &splitQBNode.operation));
    splitQBNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_indexer_qb")};
    splitQBNode.inTensorReshapeFuncs.resize(splitQBNode.inTensorIds.size());
    splitQBNode.outTensorIds = {
        GetTensorIdx(tensorMap, "indexer_rope_q"), GetTensorIdx(tensorMap, "indexer_nope_q"),
    };
    splitQBNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // 3: dimNum
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = param.index_n_heads;
        newShape.dims[2] = param.index_head_dim; // 2: dim id
    };
    opGraph.nodes.push_back(splitQBNode);
    ATB_SPEED_LOG_DEBUG("Indexer split q calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddIndexerKNode(const LatentAttentionParam<NormParamType> &param,
                                atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node kProjNode;
    atb_speed::common::FusionLinearParam kProjNodeParam;
    kProjNodeParam.isBF16 = param.isBF16;
    kProjNodeParam.quantType = atb_speed::common::LinearQuantType::NO_QUANT;
    kProjNodeParam.quantGroupSize = param.quantGroupSize;
    kProjNodeParam.transposeType = param.attnLinearTransposeType[Q_PROJ_B_LINEAR_INDEX];
    kProjNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_indexer_input_norm"),
        GetTensorIdx(tensorMap, "in_indexer_proj_wk_weight"),
        GetTensorIdx(tensorMap, "in_indexer_proj_wk_scale"),
        GetTensorIdx(tensorMap, "in_indexer_proj_wk_offset"),
        GetTensorIdx(tensorMap, "in_indexer_proj_wk_descale"),
        GetTensorIdx(tensorMap, "in_indexer_proj_wk_bias"),
        GetTensorIdx(tensorMap, "in_indexer_proj_wk_compress_idx"),
    };
    kProjNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_indexer_k")};
    CHECK_OPERATION_STATUS_RETURN(FusionLinear(kProjNodeParam, &kProjNode.operation));
    opGraph.nodes.push_back(kProjNode);
    ATB_SPEED_LOG_DEBUG("MLA proj_qkv_a calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddIndexerKNormNode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node knormNode;
    atb::infer::LayerNormParam knormParam;
    knormParam.layerType = atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
    knormParam.normParam.epsilon = param.normEps;
    knormParam.normParam.beginNormAxis = 1;
    knormParam.normParam.beginParamsAxis = 1;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(knormParam, &knormNode.operation));
    knormNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_indexer_k"),
        GetTensorIdx(tensorMap, "in_indexer_proj_k_norm_weight"),
        GetTensorIdx(tensorMap, "in_indexer_proj_k_norm_bias")
    };
    knormNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_indexer_k_norm")};
    opGraph.nodes.push_back(knormNode);
    ATB_SPEED_LOG_DEBUG("Indexer kNorm calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddIndexerKSplitNode(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node splitKNode;
    atb::infer::SplitParam splitQParam = {1, 2, {param.qkRopeHeadDim, param.index_head_dim - param.qkRopeHeadDim}};
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(splitQParam, &splitKNode.operation));
    splitKNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_indexer_k_norm")};
    splitKNode.outTensorIds = {
        GetTensorIdx(tensorMap, "indexer_rope_k"), GetTensorIdx(tensorMap, "indexer_nope_k"),
    };
    opGraph.nodes.push_back(splitKNode);
    ATB_SPEED_LOG_DEBUG("Indexer split k calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddIndexerQKRopeNode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node QKropeNode;
    atb::infer::RopeParam QKropeParam;
    QKropeParam.rotaryCoeff = param.ropeParam.rotaryCoeff;
    CreateOperation(QKropeParam, &QKropeNode.operation);
    QKropeNode.inTensorIds = {
        GetTensorIdx(tensorMap, "indexer_rope_q"), GetTensorIdx(tensorMap, "indexer_rope_k"),
        GetTensorIdx(tensorMap, "in_cos_embed"), GetTensorIdx(tensorMap, "in_sin_embed"),
        GetTensorIdx(tensorMap, "in_seq_len")
    };
    QKropeNode.outTensorIds = {
        GetTensorIdx(tensorMap, "indexer_rope_q_o"), GetTensorIdx(tensorMap, "indexer_rope_k_o"),
    };
    QKropeNode.inTensorReshapeFuncs.resize(QKropeNode.inTensorIds.size());
    QKropeNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        SqueezeHeadNumHeadDim(oldShape, newShape);
    };
    opGraph.nodes.push_back(QKropeNode);
    ATB_SPEED_LOG_DEBUG("Indexwe qrope calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddIndexerQKCatNode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node indexQCatNode;
    atb::infer::ConcatParam indexQCatParam;
    indexQCatParam.concatDim = 2; // 2: dim id
    indexQCatNode.inTensorIds = {
        GetTensorIdx(tensorMap, "indexer_rope_q_o"), GetTensorIdx(tensorMap, "indexer_nope_q")};
    indexQCatNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_indexer_q_out")};
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(indexQCatParam, &indexQCatNode.operation));
    indexQCatNode.inTensorReshapeFuncs.resize(indexQCatNode.inTensorIds.size());
    indexQCatNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // 3: dimNum
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = param.index_n_heads;
        newShape.dims[2] = oldShape.dims[1] / param.index_n_heads;
    };
    opGraph.nodes.push_back(indexQCatNode);

    atb::Node indexKCatNode;
    atb::infer::ConcatParam indexKCatParam;
    indexKCatParam.concatDim = 1;
    indexKCatNode.inTensorIds = {
        GetTensorIdx(tensorMap, "indexer_rope_k_o"), GetTensorIdx(tensorMap, "indexer_nope_k")};
    indexKCatNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_indexer_k_out")};
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(indexKCatParam, &indexKCatNode.operation));
    opGraph.nodes.push_back(indexKCatNode);
    ATB_SPEED_LOG_DEBUG("Indexer qkCatNode calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddIndexerKReshapeAndCacheNode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node kreshapeAndCacheNode;
    atb::infer::ReshapeAndCacheParam kreshapeAndCache;
    kreshapeAndCache.kvCacheCfg = atb::infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_BYPASS;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(kreshapeAndCache, &kreshapeAndCacheNode.operation));
    kreshapeAndCacheNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_indexer_k_out"),
        GetTensorIdx(tensorMap, "in_k_cache_indexer"),
        GetTensorIdx(tensorMap, "in_slots_in_pa_or_logn_in_fa"),
    };
    if (param.contextParallelInfo.IsEnabled() && param.isPrefill) {
        kreshapeAndCacheNode.inTensorIds = {
            GetTensorIdx(tensorMap, "indexer_k_out_allgather_s"),
            GetTensorIdx(tensorMap, "in_k_cache_indexer"),
            GetTensorIdx(tensorMap, "in_slots_in_pa_or_logn_in_fa"),
        };
    }
    
    kreshapeAndCacheNode.outTensorIds = {
        GetTensorIdx(tensorMap, "in_k_cache_indexer"),
    };
    kreshapeAndCacheNode.inTensorReshapeFuncs.resize(kreshapeAndCacheNode.inTensorIds.size());
    kreshapeAndCacheNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // 3: dim num
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = 1;
        newShape.dims[2] = oldShape.dims[1]; // 2: dim id
    };
    opGraph.nodes.push_back(kreshapeAndCacheNode);
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddIndexerWeightNode(const LatentAttentionParam<NormParamType> &param,
                                atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node indexerWeightNode;
    atb_speed::common::FusionLinearParam indexerWeightParam;
    indexerWeightParam.isBF16 = param.isBF16;
    indexerWeightParam.quantType = atb_speed::common::LinearQuantType::NO_QUANT;
    indexerWeightParam.quantGroupSize = param.quantGroupSize;
    indexerWeightParam.transposeType = param.attnLinearTransposeType[Q_PROJ_B_LINEAR_INDEX];
    indexerWeightNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_indexer_input_norm"),
        GetTensorIdx(tensorMap, "in_indexer_proj_weight"),
        GetTensorIdx(tensorMap, "in_indexer_proj_scale"),
        GetTensorIdx(tensorMap, "in_indexer_proj_offset"),
        GetTensorIdx(tensorMap, "in_indexer_proj_descale"),
        GetTensorIdx(tensorMap, "in_indexer_proj_bias"),
        GetTensorIdx(tensorMap, "in_indexer_proj_compress_idx"),
    };
    indexerWeightNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_indexer_weight_out")};
    CHECK_OPERATION_STATUS_RETURN(FusionLinear(indexerWeightParam, &indexerWeightNode.operation));
    opGraph.nodes.push_back(indexerWeightNode);
    ATB_SPEED_LOG_DEBUG("MLA proj_qkv_a calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddLightIndexerCpNode(const LatentAttentionParam<NormParamType> &param,
                                atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node lightIndexerNode1;
    atb_speed::common::LightningIndexerParam lightIndexerParam;
    lightIndexerParam.keyLayout = "TND";
    lightIndexerNode1.operation = new atb_speed::common::LightningIndexerOperation(
        "AclNNLightningIndexerNode", lightIndexerParam
    );
    lightIndexerNode1.inTensorIds = {
        GetTensorIdx(tensorMap, "indexer_q_out_prev"),
        GetTensorIdx(tensorMap, "indexer_k_out_prev"),
        GetTensorIdx(tensorMap, "indexer_weight_out_prev"),
        GetTensorIdx(tensorMap, "actual_seq_lengths_query_prev"),
        GetTensorIdx(tensorMap, "actual_seq_lengths_key_prev")
    };
    lightIndexerNode1.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_topk_indices_prev_cat")};
    lightIndexerNode1.inTensorReshapeFuncs.resize(lightIndexerNode1.inTensorIds.size());
    lightIndexerNode1.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        UnSqueezeHeadNumHeadDim(oldShape, newShape, oldShape.dims[oldShape.dimNum - 1]);
    };
    opGraph.nodes.push_back(lightIndexerNode1);
    
    atb::Node lightIndexerNode2;
    lightIndexerNode2.operation = new atb_speed::common::LightningIndexerOperation(
        "AclNNLightningIndexerNode", lightIndexerParam
    );
    lightIndexerNode2.inTensorIds = {
        GetTensorIdx(tensorMap, "indexer_q_out_next"),
        GetTensorIdx(tensorMap, "indexer_k_out_next"),
        GetTensorIdx(tensorMap, "indexer_weight_out_next"),
        GetTensorIdx(tensorMap, "actual_seq_lengths_query_next"),
        GetTensorIdx(tensorMap, "actual_seq_lengths_key_next")
    };
    lightIndexerNode2.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_topk_indices_next_cat")};
    lightIndexerNode2.inTensorReshapeFuncs.resize(lightIndexerNode2.inTensorIds.size());
    lightIndexerNode2.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        UnSqueezeHeadNumHeadDim(oldShape, newShape, oldShape.dims[oldShape.dimNum - 1]);
    };
    opGraph.nodes.push_back(lightIndexerNode2);
    
    atb::Node aclnnConcatNode;
    atb_speed::common::AclNNConcatParam aclNNConcatParam;
    aclNNConcatParam.dim = 0;
    aclnnConcatNode.operation = new atb_speed::common::ConcatOperation("concatindex", aclNNConcatParam);
    aclnnConcatNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_topk_indices_prev_cat"),
        GetTensorIdx(tensorMap, "intermediate_topk_indices_next_cat")};
    aclnnConcatNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_topk_indices_draft")};
    opGraph.nodes.push_back(aclnnConcatNode);

    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddLightIndexerNode(const LatentAttentionParam<NormParamType> &param,
                                atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node lightIndexerNode;
    atb_speed::common::LightningIndexerParam lightIndexerParam;
    lightIndexerNode.operation = new atb_speed::common::LightningIndexerOperation(
        "AclNNLightningIndexerNode", lightIndexerParam
    );
    lightIndexerNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_indexer_q_out"),
        GetTensorIdx(tensorMap, "in_k_cache_indexer"),
        GetTensorIdx(tensorMap, "intermediate_indexer_weight_out"),
        GetTensorIdx(tensorMap, "in_seq_len_query"),
        GetTensorIdx(tensorMap, "in_seq_len"),
        GetTensorIdx(tensorMap, "in_block_tables"),
    };
    lightIndexerNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_topk_indices")};
    opGraph.nodes.push_back(lightIndexerNode);
    ATB_SPEED_LOG_DEBUG("LightningIndexer calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddSparseFlashAttentionNode(
    const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node sparseFlashAttentionNode;
    atb_speed::common::SparseFlashAttentionParam sparseFlashAttentionParam;
    sparseFlashAttentionParam.queryLayout = "TND";
    sparseFlashAttentionParam.kvLayout = "PA_BSND";
    sparseFlashAttentionParam.scaleValue = param.softmaxScale;
    sparseFlashAttentionParam.sparseBlockSize = 1;
    sparseFlashAttentionParam.hasBlockTable = true;
    sparseFlashAttentionNode.operation = new atb_speed::common::SparseFlashAttentionOperation(
        "AclNNSparseFlashAttentionNode", sparseFlashAttentionParam
    );
    auto q_nope = ""; 
    auto q_pe = ""; 
    if (param.isPrefill || !param.enableMlaPreprocess) {
        q_nope = "intermediate_q_t";
        q_pe = "rope_q_o";
    } else {
        q_nope = "intermediate_q_nope";
        q_pe = "intermediate_q_rope";
    }
    sparseFlashAttentionNode.inTensorIds = {
        GetTensorIdx(tensorMap, q_nope),
        GetTensorIdx(tensorMap, "in_k_cache"),
        GetTensorIdx(tensorMap, "in_k_cache"),
        GetTensorIdx(tensorMap, "intermediate_topk_indices"),
        GetTensorIdx(tensorMap, "in_block_tables"),
        GetTensorIdx(tensorMap, "in_seq_len_query"),
        GetTensorIdx(tensorMap, "in_seq_len"),
        GetTensorIdx(tensorMap, q_pe),
        GetTensorIdx(tensorMap, "in_k_rope_cache"),
    };
    sparseFlashAttentionNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_self_attention")};
    if (param.isPrefill || !param.enableMlaPreprocess) {
        sparseFlashAttentionNode.inTensorReshapeFuncs.resize(sparseFlashAttentionNode.inTensorIds.size());
        sparseFlashAttentionNode.inTensorReshapeFuncs[7] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            UnSqueezeHeadNumHeadDim(oldShape, newShape, param.qkRopeHeadDim);
        };
    }
    opGraph.nodes.push_back(sparseFlashAttentionNode);
    ATB_SPEED_LOG_DEBUG("SparseFlashAttention calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddSparseFlashAttentionCpNode(
    const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    // knope, kpe gathersplit - "k_gather_index"               | pre/next
    CHECK_OPERATION_STATUS_RETURN(AddSfaGatherCpNode(param, opGraph, tensorMap,
        "intermediate_kv", "intermediate_kv_prev", "k_gather_index_prev"));
    CHECK_OPERATION_STATUS_RETURN(AddSfaGatherCpNode(param, opGraph, tensorMap,
        "intermediate_kv", "intermediate_kv_next", "k_gather_index_next"));
    CHECK_OPERATION_STATUS_RETURN(AddSfaGatherCpNode(param, opGraph, tensorMap,
        "rope_k_o", "rope_k_o_prev", "k_gather_index_prev"));
    CHECK_OPERATION_STATUS_RETURN(AddSfaGatherCpNode(param, opGraph, tensorMap,
        "rope_k_o", "rope_k_o_next", "k_gather_index_next"));

    // qnope,qpe,topk gather&split - "cp_load_balance_idx"   | pre/next
    CHECK_OPERATION_STATUS_RETURN(AddSfaGatherCpNode(param, opGraph, tensorMap,
        "intermediate_q_t", "intermediate_q_balance", "cp_load_balance_idx"));
    CHECK_OPERATION_STATUS_RETURN(AddSfaSplitCpNode(param, opGraph, tensorMap,
        "intermediate_q_balance", "intermediate_q_prev", "intermediate_q_next"));
    CHECK_OPERATION_STATUS_RETURN(AddSfaGatherCpNode(param, opGraph, tensorMap,
        "rope_q_o", "rope_q_o_balance", "cp_load_balance_idx"));
    CHECK_OPERATION_STATUS_RETURN(AddSfaSplitCpNode(param, opGraph, tensorMap,
        "rope_q_o_balance", "rope_q_o_prev", "rope_q_o_next"));
    CHECK_OPERATION_STATUS_RETURN(AddSfaGatherCpNode(param, opGraph, tensorMap,
        "intermediate_topk_indices", "intermediate_topk_indices_balance", "cp_load_balance_idx"));
    //CHECK_OPERATION_STATUS_RETURN(AddSfaSplitCpNode(param, opGraph, tensorMap,
    //    "intermediate_topk_indices_balance", "intermediate_topk_indices_prev", "intermediate_topk_indices_next"));
    
    atb::Node aclnnSplitNode;
    atb_speed::common::AclNNSplitWithSizeParam aclNNSplitParam;
    aclNNSplitParam.dim = 0;
    aclNNSplitParam.num = 2;
    aclnnSplitNode.operation = new atb_speed::common::SplitWithSizeOperation("splitindex", aclNNSplitParam);
    aclnnSplitNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_topk_indices_balance")};
    aclnnSplitNode.outTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_topk_indices_prev"),
        GetTensorIdx(tensorMap, "intermediate_topk_indices_next")};
    opGraph.nodes.push_back(aclnnSplitNode);

    // get actual seq_lengths_query/seq_lengths_key 
    //      actual_seq_lengths_query_prev / actual_seq_lengths_key_prev

    atb_speed::common::SparseFlashAttentionParam sparseFlashAttentionParam;
    sparseFlashAttentionParam.queryLayout = "TND";
    sparseFlashAttentionParam.kvLayout = "TND";
    sparseFlashAttentionParam.scaleValue = param.softmaxScale;
    sparseFlashAttentionParam.sparseBlockSize = 1;
    sparseFlashAttentionParam.hasBlockTable = false;

    atb::Node sparseFlashAttentionPrevNode;
    sparseFlashAttentionPrevNode.operation = new atb_speed::common::SparseFlashAttentionOperation(
        "AclNNsparseFlashAttentionPrevNode", sparseFlashAttentionParam
    );
    sparseFlashAttentionPrevNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_q_prev"),
        GetTensorIdx(tensorMap, "intermediate_kv_prev"),
        GetTensorIdx(tensorMap, "intermediate_kv_prev"),
        GetTensorIdx(tensorMap, "intermediate_topk_indices_prev"),
        GetTensorIdx(tensorMap, "in_block_tables"),
        GetTensorIdx(tensorMap, "actual_seq_lengths_query_prev"),
        GetTensorIdx(tensorMap, "actual_seq_lengths_key_prev"),
        GetTensorIdx(tensorMap, "rope_q_o_prev"),
        GetTensorIdx(tensorMap, "rope_k_o_prev"),
    };
    sparseFlashAttentionPrevNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_self_attention_prev")};
    sparseFlashAttentionPrevNode.inTensorReshapeFuncs.resize(sparseFlashAttentionPrevNode.inTensorIds.size());
    sparseFlashAttentionPrevNode.inTensorReshapeFuncs[7] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        UnSqueezeHeadNumHeadDim(oldShape, newShape, param.qkRopeHeadDim);
    };
    sparseFlashAttentionPrevNode.inTensorReshapeFuncs[8] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        UnSqueezeHeadNumHeadDim(oldShape, newShape, param.qkRopeHeadDim);
    };
    opGraph.nodes.push_back(sparseFlashAttentionPrevNode);
    
    atb::Node sparseFlashAttentionNextNode;
    sparseFlashAttentionNextNode.operation = new atb_speed::common::SparseFlashAttentionOperation(
        "AclNNsparseFlashAttentionNextNode", sparseFlashAttentionParam
    );
    sparseFlashAttentionNextNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_q_next"),
        GetTensorIdx(tensorMap, "intermediate_kv_next"),
        GetTensorIdx(tensorMap, "intermediate_kv_next"),
        GetTensorIdx(tensorMap, "intermediate_topk_indices_next"),
        GetTensorIdx(tensorMap, "in_block_tables"),
        GetTensorIdx(tensorMap, "actual_seq_lengths_query_next"),
        GetTensorIdx(tensorMap, "actual_seq_lengths_key_next"),
        GetTensorIdx(tensorMap, "rope_q_o_next"),
        GetTensorIdx(tensorMap, "rope_k_o_next"),
    };
    sparseFlashAttentionNextNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_self_attention_next")};
    sparseFlashAttentionNextNode.inTensorReshapeFuncs.resize(sparseFlashAttentionNextNode.inTensorIds.size());
    sparseFlashAttentionNextNode.inTensorReshapeFuncs[7] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        UnSqueezeHeadNumHeadDim(oldShape, newShape, param.qkRopeHeadDim);
    };
    sparseFlashAttentionNextNode.inTensorReshapeFuncs[8] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        UnSqueezeHeadNumHeadDim(oldShape, newShape, param.qkRopeHeadDim);
    };
    opGraph.nodes.push_back(sparseFlashAttentionNextNode);
    
    atb::Node indexCatNode;
    atb::infer::ConcatParam sfaCatParam;
    sfaCatParam.concatDim = 0;
    indexCatNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_self_attention_prev"),
        GetTensorIdx(tensorMap, "intermediate_self_attention_next")};
    indexCatNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_self_attention_draft")};
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(sfaCatParam, &indexCatNode.operation));
    opGraph.nodes.push_back(indexCatNode);
    
    // attn & cat & rerank - "cp_o_recover_idx"
    CHECK_OPERATION_STATUS_RETURN(AddSfaGatherCpNode(param, opGraph, tensorMap,
        "intermediate_self_attention_draft", "intermediate_self_attention", "cp_o_recover_idx"));
        
    ATB_SPEED_LOG_DEBUG("SparseFlashAttention calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddsfaTransposeNode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node sfaTransposeNode;
    atb::infer::TransposeParam sfaTransposeParam;
    sfaTransposeNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_self_attention")};
    sfaTransposeNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_sfa_out")};
    sfaTransposeParam.perm = {1, 0, 2};
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(sfaTransposeParam,
        &sfaTransposeNode.operation));
    opGraph.nodes.push_back(sfaTransposeNode);
    ATB_SPEED_LOG_DEBUG("sfa output transpose calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddReprojVTransposeNode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node reprojVTransposeNode;
    atb::infer::TransposeParam reprojVTransposeParam;
    reprojVTransposeNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_reproj")};
    reprojVTransposeNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_reproj_t")};
    reprojVTransposeParam.perm = {1, 0, 2};
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(reprojVTransposeParam,
        &reprojVTransposeNode.operation));
    opGraph.nodes.push_back(reprojVTransposeNode);
    ATB_SPEED_LOG_DEBUG(" ReprojV output transpose calculation success");
    return atb::NO_ERROR;
}


template <typename NormParamType>
atb::Status Preprocess(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{   
    if (param.qLoraRank > 0 && param.enableMlaPreprocess && !param.isPrefill) {
        CHECK_OPERATION_STATUS_RETURN(AddMlaPreprocessV2Node(param, opGraph, tensorMap));
        // indexer
        // CHECK_OPERATION_STATUS_RETURN(AddLAttnPreNormNode(param, opGraph, tensorMap)); //strictly copy
        // CHECK_OPERATION_STATUS_RETURN(AddLAttnQProjRecalNode(param, opGraph, tensorMap));
        // CHECK_OPERATION_STATUS_RETURN(AddLAttnQNormRecalNode(param, opGraph, tensorMap));
    } else {
        CHECK_OPERATION_STATUS_RETURN(AddLAttnPreNormNode(param, opGraph, tensorMap));
        if (param.qLoraRank > 0 && param.enablePreprocessLcocTp) {
            CHECK_OPERATION_STATUS_RETURN(AddLAttnKVAProjNode(param, opGraph, tensorMap));
            CHECK_OPERATION_STATUS_RETURN(SetTPAllGatherNode(param, opGraph, tensorMap));
            CHECK_OPERATION_STATUS_RETURN(SetFFNUnPadding(param, opGraph, tensorMap));
            CHECK_OPERATION_STATUS_RETURN(AddSplitKNode(param, opGraph, tensorMap));
            CHECK_OPERATION_STATUS_RETURN(AddLAttnQProjANode(param, opGraph, tensorMap));
            CHECK_OPERATION_STATUS_RETURN(AddLAttnQNormNode(param, opGraph, tensorMap));
            CHECK_OPERATION_STATUS_RETURN(AddFusedQBNode(param, opGraph, tensorMap));
            CHECK_OPERATION_STATUS_RETURN(SetFFNQUnPadding(param, opGraph, tensorMap));
            CHECK_OPERATION_STATUS_RETURN(AddSplitQNode(param, opGraph, tensorMap));
        } else if (param.qLoraRank > 0) {
            CHECK_OPERATION_STATUS_RETURN(AddLAttnQKVProjNode(param, opGraph, tensorMap));
            if (param.enableQkvdownDp) {
                CHECK_OPERATION_STATUS_RETURN(SetTPAllGatherNode(param, opGraph, tensorMap));
                CHECK_OPERATION_STATUS_RETURN(SetFFNUnPadding(param, opGraph, tensorMap));
            }

            CHECK_OPERATION_STATUS_RETURN(AddSplitQKNode(param, opGraph, tensorMap));
            CHECK_OPERATION_STATUS_RETURN(AddLAttnQProjBNode(param, opGraph, tensorMap));
            //latent_q - > q_lora_out
            CHECK_OPERATION_STATUS_RETURN(AddSplitQNode(param, opGraph, tensorMap));
            //q_lora_out -> rope_q nope_q
        } else {
            CHECK_OPERATION_STATUS_RETURN(AddLAttnQProjANode(param, opGraph, tensorMap));
            CHECK_OPERATION_STATUS_RETURN(AddSplitQNode(param, opGraph, tensorMap));
            CHECK_OPERATION_STATUS_RETURN(AddLAttnKVAProjNode(param, opGraph, tensorMap));
            CHECK_OPERATION_STATUS_RETURN(AddSplitKNode(param, opGraph, tensorMap));
        }
        if (param.contextParallelInfo.IsEnabled() && param.isPrefill) {
            CHECK_OPERATION_STATUS_RETURN(AddSfaAllGatherAndReorderCpNode(param, opGraph, tensorMap,
                "intermediate_kv_cp", "intermediate_kv_allgather", "intermediate_kv", "cp_kv_recover_idx"));
        }
        // 2_0_14
        CHECK_OPERATION_STATUS_RETURN(AddLAttnKVNormNode(param, opGraph, tensorMap));
        //intermediate_kv -> intermediate_kv
        if (param.rotaryType != RotaryType::NO_ROTARY) {
            CHECK_OPERATION_STATUS_RETURN(AddLAttnRopeNode(param, opGraph, tensorMap));
            if (param.contextParallelInfo.IsEnabled() && param.isPrefill) {
                // CHECK_OPERATION_STATUS_RETURN(AddSfaRopeCpNode(param, opGraph, tensorMap));
                CHECK_OPERATION_STATUS_RETURN(AddSfaAllGatherAndReorderCpNode(param, opGraph, tensorMap,
                 "rope_k_o_cp", "rope_k_o_allgather", "rope_k_o", "cp_kv_recover_idx"));
            }
        }
        if (param.isPrefill || !param.enableMlaPreprocess) {
            CHECK_OPERATION_STATUS_RETURN(PreprocessKV(param, opGraph, tensorMap));
        }
        CHECK_OPERATION_STATUS_RETURN(AddReshapeAndCacheNode(param, opGraph, tensorMap));
        CHECK_OPERATION_STATUS_RETURN(AddLAttnQNormRecalNode(param, opGraph, tensorMap));
    }
    // indexer
    // CHECK_OPERATION_STATUS_RETURN(AddLAttnQNormRecalNode(param, opGraph, tensorMap));
    CHECK_OPERATION_STATUS_RETURN(AddIndexerQBNode(param, opGraph, tensorMap));
    CHECK_OPERATION_STATUS_RETURN(AddIndexerQBSplitNode(param, opGraph, tensorMap));
    
    CHECK_OPERATION_STATUS_RETURN(AddIndexerInputNormRecalNode(param, opGraph, tensorMap));
    CHECK_OPERATION_STATUS_RETURN(AddIndexerKNode(param, opGraph, tensorMap));
    CHECK_OPERATION_STATUS_RETURN(AddIndexerKNormNode(param, opGraph, tensorMap));
    CHECK_OPERATION_STATUS_RETURN(AddIndexerKSplitNode(param, opGraph, tensorMap));

    CHECK_OPERATION_STATUS_RETURN(AddIndexerQKRopeNode(param, opGraph, tensorMap));

    CHECK_OPERATION_STATUS_RETURN(AddIndexerQKCatNode(param, opGraph, tensorMap));
    // "intermediate_indexer_k_out"

    if (param.contextParallelInfo.IsEnabled() && param.isPrefill) {
        CHECK_OPERATION_STATUS_RETURN(AddSfaAllGatherAndReorderCpNode(param, opGraph, tensorMap,
            "intermediate_indexer_k_out", "indexer_k_out_allgather", "indexer_k_out_allgather_s", "cp_kv_recover_idx"));
    }
    CHECK_OPERATION_STATUS_RETURN(AddIndexerKReshapeAndCacheNode(param, opGraph, tensorMap));

    CHECK_OPERATION_STATUS_RETURN(AddIndexerWeightNode(param, opGraph, tensorMap));

    if (param.contextParallelInfo.IsEnabled() && param.isPrefill) {
        // q/weight rerank - "cp_load_balance_idx"
        CHECK_OPERATION_STATUS_RETURN(AddSfaGatherCpNode(param, opGraph, tensorMap,
            "intermediate_indexer_q_out", "indexer_q_out_balance", "cp_load_balance_idx"));
        CHECK_OPERATION_STATUS_RETURN(AddSfaGatherCpNode(param, opGraph, tensorMap,
            "intermediate_indexer_weight_out", "indexer_weight_out_balance", "cp_load_balance_idx"));
        // q/weight split                         | prev / next
        CHECK_OPERATION_STATUS_RETURN(AddSfaSplitCpNode(param, opGraph, tensorMap,
            "indexer_q_out_balance", "indexer_q_out_prev", "indexer_q_out_next"));
        CHECK_OPERATION_STATUS_RETURN(AddSfaSplitCpNode(param, opGraph, tensorMap,
            "indexer_weight_out_balance", "indexer_weight_out_prev", "indexer_weight_out_next"));
                
        // get actual seq_len_q / seq_len_kv      | prev / next
        //      actual_seq_lengths_query_prev / actual_seq_lengths_key_prev
        // k rerank - "k_gather_index"            | prev / next
        CHECK_OPERATION_STATUS_RETURN(AddSfaGatherCpNode(param, opGraph, tensorMap,
            "indexer_k_out_allgather_s", "indexer_k_out_prev", "k_gather_index_prev"));
        CHECK_OPERATION_STATUS_RETURN(AddSfaGatherCpNode(param, opGraph, tensorMap,
            "indexer_k_out_allgather_s", "indexer_k_out_next", "k_gather_index_next"));
        CHECK_OPERATION_STATUS_RETURN(AddLightIndexerCpNode(param, opGraph, tensorMap));
            //intermediate_topk_indices_prev_cat intermediate_topk_indices_next_cat

        // index & cat & rerank - "cp_o_recover_idx"
        CHECK_OPERATION_STATUS_RETURN(AddSfaGatherCpNode(param, opGraph, tensorMap,
            "intermediate_topk_indices_draft", "intermediate_topk_indices", "cp_o_recover_idx"));
    }
    else{
        CHECK_OPERATION_STATUS_RETURN(AddLightIndexerNode(param, opGraph, tensorMap));
    }
    return atb::NO_ERROR;
}

} // namespace sparse

template <typename NormParamType>
atb::Status SparseAttention(const LatentAttentionParam<NormParamType> &param, atb::Operation **operation)
{
    std::shared_ptr<int64_t> batchSizePtr = std::make_shared<int64_t>(0);
    atb::GraphParam opGraph;
    opGraph.name = "SparseAttention";
    std::map<std::string, uint32_t> tensorMap = sparse::ConstructTensorMap(param,
        opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);
    // ATB_SPEED_LOG_ERROR("opGraph.inTensorNum " << opGraph.inTensorNum);
    // ATB_SPEED_LOG_ERROR("opGraph.outTensorNum " << opGraph.outTensorNum);
    // ATB_SPEED_LOG_ERROR("opGraph.internalTensorNum " << opGraph.internalTensorNum);
    // Preprocess
    CHECK_OPERATION_STATUS_RETURN(sparse::Preprocess(param, opGraph, tensorMap));
    // PA or MLA
    
    if (param.contextParallelInfo.IsEnabled() && param.isPrefill) {
    // CP Attention
        CHECK_OPERATION_STATUS_RETURN(sparse::AddSparseFlashAttentionCpNode(param, opGraph, tensorMap));
    }
    else {
        CHECK_OPERATION_STATUS_RETURN(sparse::AddSparseFlashAttentionNode(param, opGraph, tensorMap));
    }
    // CHECK_OPERATION_STATUS_RETURN(sparse::AddsfaTransposeNode(param, opGraph, tensorMap)); 
    // CHECK_OPERATION_STATUS_RETURN(sparse::AddReprojVNode(param, opGraph, tensorMap));
    // CHECK_OPERATION_STATUS_RETURN(sparse::AddReprojVTransposeNode(param, opGraph, tensorMap));
    // using MATMUL_EIN_SUM
    CHECK_OPERATION_STATUS_RETURN(sparse::AddEinReprojVNode(param, opGraph, tensorMap));

    CHECK_OPERATION_STATUS_RETURN(sparse::AddSelfOutLinearParallelNode(param, opGraph, tensorMap));
    opGraph.inferShapeFunc = [=]
                (const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        if (param.enableQkvdownDp) {
            outTensorDescs.at(0).shape.dims[0] *= param.attnTpSize;
        }
        if (param.enableOutLcocTp) {
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(atb_speed::common::GetTensorIdx(tensorMap,
                "in_attn_padding_idx")).shape.dims[0] / param.attnTpSize;
        }
        return atb::NO_ERROR;
    };
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
    return atb::NO_ERROR;
}

template atb::Status SparseAttention(
    const LatentAttentionParam<atb::infer::RmsNormParam> &param, atb::Operation **operation);
} // namespace deepseekV2
} // namespace atb_speed
