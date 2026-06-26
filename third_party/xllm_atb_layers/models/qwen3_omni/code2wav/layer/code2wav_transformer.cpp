#include "models/qwen3_omni/code2wav/layer/code2wav_transformer.h"
#include "models/qwen3_omni/code2wav/operation/code2wav_operations.h"
#include <glog/logging.h>
#include <vector>

namespace atb_speed {
namespace qwen3_omni {

std::map<std::string, std::vector<std::string>>
Get_Transformer_LayerTensorCandidates() {
  std::map<std::string, std::vector<std::string>> qwenLayerTensorCandiadates = {
      {"default_weight",
       {"in_input_norm_weight",
        "in_qkv_weight",
        "in_attention_out_weight",
        "in_self_attn_layer_scale_scale",
        "in_selfattention_out_norm_weight",
        "in_mlp_gateup_weight",
        "in_mlp_down_weight",
        "in_mlp_layer_scale_scale",
        "in_mlp_gate_weight", // not use
        "in_mlp_up_weight", // not use
        "in_qkv_weight_q", // not use
        "in_qkv_weight_k", // not use
        "in_qkv_weight_v" // not use
        }},
      {"default_input",
       {"in_hidden_states",
        "in_cos_embedding",
        "in_sin_embedding",
        "in_seq_len",
        "in_attn_mask",
       }},
      {"out", {"out"}},
      {"intermediate_out",
       {"input_norm_out",
        "qkv_linear_out",
        "intermediate_q",
        "intermediate_k",
        "intermediate_v",
        "intermediate_rope_k",
        "intermediate_rope_q",
        "intermediate_atten_out",
        "intermediate_proj_out",
        "intermediate_atten_scale_out",
        "intermediate_add_out",
        "intermediate_post_atten_norm_out",
        "intermediate_gateup_out",
        "intermediate_silu_out",
        "intermediate_down_out",
        "intermediate_mlp_scale_out"}},
    };
  return qwenLayerTensorCandiadates;
}

static std::map<std::string, uint32_t> ConstructTensorMap(
    const Code2WavTransformerLayerParam& param,
    uint32_t& inTensorNum,
    uint32_t& outTensorNum,
    uint32_t& internalTensorNum) {
  auto qwenLayerTensorCandiadates = Get_Transformer_LayerTensorCandidates();
  std::vector<std::string> inTensorList = {};
  std::vector<std::string> intermediateTensorList = {};
  std::vector<std::string> outTensorList = {};
  // weight
  atb_speed::common::AddTensorToList(
      qwenLayerTensorCandiadates, "default_weight", inTensorList);
  // input
  atb_speed::common::AddTensorToList(
      qwenLayerTensorCandiadates, "default_input", inTensorList);
  // out
  atb_speed::common::AddTensorToList(
      qwenLayerTensorCandiadates, "out", outTensorList);
  // intermediate
  atb_speed::common::AddTensorToList(
      qwenLayerTensorCandiadates, "intermediate_out", intermediateTensorList);
  if (param.worldSize > 1) {
    atb_speed::common::AddTensorToList(qwenLayerTensorCandiadates,
                                       "parallel_intermediate_out",
                                       intermediateTensorList);
  }
  inTensorNum = inTensorList.size();
  outTensorNum = outTensorList.size();
  internalTensorNum = intermediateTensorList.size();

  return atb_speed::common::GetTensorMap(
      inTensorList, outTensorList, intermediateTensorList);
}

atb::Status BuildCode2WavAttentionBlock(
    const Code2WavTransformerLayerParam& param,
    const std::map<std::string, uint32_t>& tensorMap,
    atb::GraphParam& opGraph,
    bool isTp) {
  atb::Node inputNormNode;
  atb::Node qkvlinearNode;
  atb::Node splitNode;
  atb::Node ropeNode;
  atb::Node attnScaleNode;
  atb::Node selfAttentionNode;
  atb::Node outProjNode;
  atb::Node selfResidualAddNode;
  atb::Node proj_add_biasNode;

  // Input Norm
  CreateNormOperation(param,&inputNormNode.operation);
  inputNormNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"in_hidden_states", "in_input_norm_weight"});
  inputNormNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"input_norm_out"});
  opGraph.nodes.push_back(inputNormNode);

  // QKV Linear
  CreateLinearOperation(&qkvlinearNode.operation);
  qkvlinearNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"input_norm_out", "in_qkv_weight"});
  qkvlinearNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"qkv_linear_out"});
  opGraph.nodes.push_back(qkvlinearNode);

  // Split
  CreateSplitOperation(&splitNode.operation);
  splitNode.inTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"qkv_linear_out"});
  splitNode.outTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_q", "intermediate_k", "intermediate_v"});
  opGraph.nodes.push_back(splitNode);

  // Rope
  CreateRopeOperation(&ropeNode.operation);
  ropeNode.inTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap,
                                          {"intermediate_q",
                                           "intermediate_k",
                                           "in_cos_embedding",
                                           "in_sin_embedding",
                                           "in_seq_len"});
  ropeNode.outTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_rope_q", "intermediate_rope_k"});
  opGraph.nodes.push_back(ropeNode);

  // Self Attention
  CreateSelfAttentionOperation(param, &selfAttentionNode.operation);
  selfAttentionNode.inTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap,
                                          {"intermediate_rope_q",
                                           "intermediate_rope_k",
                                           "intermediate_v",
                                           "in_attn_mask",
                                           "in_seq_len"});
  selfAttentionNode.outTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_atten_out"});
  opGraph.nodes.push_back(selfAttentionNode);

  // Output Projection
  if (isTp) {
    CreateLinearParallelOperation(param, &outProjNode.operation);
    outProjNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_atten_out", "in_attention_out_weight"});
  } else {
    CreateLinearOperation(&outProjNode.operation);
    outProjNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap,
        {"intermediate_atten_out", "in_attention_out_weight"});
  }
  outProjNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_proj_out"});
  opGraph.nodes.push_back(outProjNode);

  // scale
  CreateMulOperation(&attnScaleNode.operation);
  attnScaleNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_proj_out", "in_self_attn_layer_scale_scale"});
  attnScaleNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_atten_scale_out"});
  opGraph.nodes.push_back(attnScaleNode);

  // Residual Add
  CreateAddOperation(&selfResidualAddNode.operation);
  std::string residualInput = "intermediate_atten_scale_out";
  selfResidualAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"in_hidden_states", residualInput});
  selfResidualAddNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_add_out"});
  opGraph.nodes.push_back(selfResidualAddNode);

  return atb::NO_ERROR;
}

atb::Status BuildCode2WavMLPBlock(const Code2WavTransformerLayerParam& param,
                          const std::map<std::string, uint32_t>& tensorMap,
                          atb::GraphParam& opGraph,
                          bool isTp) {
  atb::Node postNormNode;
  atb::Node gateUpNode;
  atb::Node siluNode;
  atb::Node downNode;
  atb::Node mlpScaleNode;
  atb::Node mlpResidualAddNode;

  // Post Norm
  CreateNormOperation(param,&postNormNode.operation);
  postNormNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_add_out", "in_selfattention_out_norm_weight"});
  postNormNode.outTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_post_atten_norm_out"});
  opGraph.nodes.push_back(postNormNode);

  // GateUp Linear
  CreateLinearOperation(&gateUpNode.operation);
  gateUpNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_post_atten_norm_out", "in_mlp_gateup_weight"});
  gateUpNode.outTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_gateup_out"});
  opGraph.nodes.push_back(gateUpNode);

  // silu Activation
  ActivationType act_type = ActivationType::ACTIVATION_SWIGLU_FORWARD;
  CreateActivateOperation(&siluNode.operation,act_type);
  siluNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_gateup_out"});
  siluNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_silu_out"});
  opGraph.nodes.push_back(siluNode);

  // Down Linear
  CreateLinearOperation(&downNode.operation);
  downNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap,
      {"intermediate_silu_out", "in_mlp_down_weight"});

  downNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_down_out"});
  opGraph.nodes.push_back(downNode);

  // scale
  CreateMulOperation(&mlpScaleNode.operation);
  mlpScaleNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_down_out", "in_mlp_layer_scale_scale"});
  mlpScaleNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_mlp_scale_out"});
  opGraph.nodes.push_back(mlpScaleNode);

  // MLP Residual Add
  std::string downInput = "intermediate_mlp_scale_out";
  CreateAddOperation(&mlpResidualAddNode.operation);
  mlpResidualAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {downInput, "intermediate_add_out"});
  mlpResidualAddNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"out"});
  opGraph.nodes.push_back(mlpResidualAddNode);

  return atb::NO_ERROR;
}

atb::Status Qwen3Omni_Code2wav_TransformerLayer(const Code2WavTransformerLayerParam& param,
                         atb::Operation** operation) {
  atb::GraphParam opGraph;
  opGraph.name = "Qwen3Omni_Code2wav_TransformerLayer";
  bool isTp = (param.worldSize > 1);

  // tensor names
  std::map<std::string, uint32_t> tensorMap =
      ConstructTensorMap(param,
                         opGraph.inTensorNum,
                         opGraph.outTensorNum,
                         opGraph.internalTensorNum);

  // Attention
  BuildCode2WavAttentionBlock(param, tensorMap, opGraph, isTp);
  // MLP
  BuildCode2WavMLPBlock(param, tensorMap, opGraph, isTp);
  opGraph.inferShapeFunc =
      [=](const atb::SVector<atb::TensorDesc>& inTensorDescs,
          atb::SVector<atb::TensorDesc>& outTensorDescs) {
        if (!inTensorDescs.empty() && !outTensorDescs.empty()) {
          outTensorDescs.at(0) = inTensorDescs.at(0);
        }
        return atb::NO_ERROR;
      };
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
  return atb::NO_ERROR;
}

}  // namespace qwen
}  // namespace atb_speed
