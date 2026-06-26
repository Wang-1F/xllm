#pragma once
#include "atb/atb_infer.h"
#include "atb_speed/utils/operation_util.h"
#include "models/qwen2_5/vision_encoder/encoder_layer.h"
#include "operations/fusion/utils.h"

namespace atb_speed {
namespace qwen3_omni {

struct Code2WavTransformerLayerParam {
  bool isBF16 = false;
  bool supportLcoc = false;
  bool supportLora = false;
  bool loraEnableGMM = false;
  bool enableLogN = false;
  std::string backend = "hccl";
  int rank = 0;
  int worldSize = 1;
  int quantType = 0;
  int quantGroupSize = 64;
  int numAttentionHeadsPerRank = 0;
  int hiddenSizePerAttentionHead = 0;
  int numKeyValueHeadsPerRank = 0;
  float rmsNormEps = 0;

  std::vector<int> seqLen;
  std::vector<int> tokenOffset;
  std::vector<int> packQuantType = {};
  std::vector<int> linearQuantType = {};
  std::vector<int> linearTransposeType;
};


using ActivationType = atb::infer::ActivationType;
atb::Status
CreateNormOperation(const Code2WavTransformerLayerParam &vision_encoder_param,
                    atb::Operation **op) ;

atb::Status
CreateLayerNormOperation(const Code2WavTransformerLayerParam &vision_encoder_param,
                    atb::Operation **op) ;

atb::Status CreateLinearOperation(atb::Operation **op, bool hasBias = false,
                                  bool istransB = true) ;

atb::Status
CreateAllReduceOperation(const Code2WavTransformerLayerParam &vision_encoder_param,
                         atb::Operation **op) ;

atb::Status CreateLinearParallelOperation(
    const Code2WavTransformerLayerParam &vision_encoder_param, atb::Operation **op,
    bool trans_weight = true) ;

atb::Status CreateActivateOperation(atb::Operation** op,ActivationType type) ;

atb::Status CreateMulOperation(atb::Operation **op) ;

atb::Status CreateSplitOperation(atb::Operation **op) ;

atb::Status CreateRopeOperation(atb::Operation **op) ;

atb::Status CreateSelfAttentionOperation(const Code2WavTransformerLayerParam &param,
                                         atb::Operation **op) ;

atb::Status CreateAddOperation(atb::Operation **op) ;
} // namespace qwen
} // namespace atb_speed
