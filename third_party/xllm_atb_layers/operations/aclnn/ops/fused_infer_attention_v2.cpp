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
#include "fused_infer_attention_v2.h"

#include <securec.h>

#include "acl/acl.h"
#include "aclnnop/aclnn_fused_infer_attention_score_v2.h"
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"

namespace atb_speed {
namespace common {

const int ACLNN_TENSOR_LIST_INDEX[8] = {-1, 1, 2, -1, -1, -1, -1, -1};
const int ACLNN_TENSOR_INDEX[8] = {0, 0, 0, 4, 6, 14, 12, 13};
FusedInferAttentionV2Operation::FusedInferAttentionV2Operation(
    const std::string& name,
    const AclNNFusedInferAttnParam& param)
    : AclNNOperation(name), param_(param) {
  ATB_SPEED_LOG_DEBUG(
      "FusedInferAttentionV2Operation, param: " << param_.ToString());
}

FusedInferAttentionV2Operation::~FusedInferAttentionV2Operation() {
  ATB_SPEED_LOG_DEBUG("~FusedInferAttentionV2Operation");
}

uint32_t FusedInferAttentionV2Operation::GetInputNum() const { return NUM4; }

uint32_t FusedInferAttentionV2Operation::GetOutputNum() const { return NUM1; }

atb::Status FusedInferAttentionV2Operation::InferShape(
    const atb::SVector<atb::TensorDesc>& inTensorDescs,
    atb::SVector<atb::TensorDesc>& outTensorDescs) const {
  ATB_SPEED_LOG_DEBUG(opName_ << " infer shape start");
  outTensorDescs.at(0) = inTensorDescs.at(0);
  ATB_SPEED_LOG_DEBUG(opName_ << " infer shape end");
  return 0;
}

int FusedInferAttentionV2Operation::CreateAclNNVariantPack(
    const atb::VariantPack& variantPack) {
  ATB_SPEED_LOG_DEBUG(opName_ << " CreateAclNNVariantPack start");
  int ret = 0;
  ret = CreateAclNNInTensorVariantPack(variantPack);
  if (ret != 0) {
    ATB_SPEED_LOG_ERROR(this->opName_
                        << " AclNNTensor CreateAclNNInTensorVariantPack fail");
    return ret;
  }
  ret = CreateAclNNOutTensorVariantPack(variantPack);
  if (ret != 0) {
    ATB_SPEED_LOG_ERROR(this->opName_
                        << " AclNNTensor CreateAclNNOutTensorVariantPack fail");
    return ret;
  }

  ATB_SPEED_LOG_DEBUG(opName_ << " CreateAclNNVariantPack end");
  return atb::NO_ERROR;
}

int FusedInferAttentionV2Operation::CreateAclNNInTensorVariantPack(
    const atb::VariantPack& variantPack) {
  AclNNVariantPack& aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
  uint32_t inputNum = GetInputNum();
  aclnnVariantPack.aclInTensors.resize(inputNum);
  for (size_t i = 0; i < inputNum; i++) {
    std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
    aclnnTensor->atbTensor = variantPack.inTensors.at(i);
    aclnnTensor->tensorIdx = ACLNN_TENSOR_INDEX[i];
    aclnnTensor->tensorListidx = ACLNN_TENSOR_LIST_INDEX[i];
    if (i == 4) {  // qSeqLens is 2nd last input tensor
      aclnnTensor->needUpdateTensorDataPtr = false;
      ConvertTensorToSeqLengths(aclnnTensor->atbTensor, actualSeqLengths_);
    } else if (i == 3) {  // kvSeqLens is the last input tensor
      aclnnTensor->needUpdateTensorDataPtr = false;
      aclnnTensor->intArrayHostData.dataSize = aclnnTensor->atbTensor.dataSize / NUM4; // int32 has 4 bytes
      aclnnTensor->intArrayHostData.data.resize(aclnnTensor->intArrayHostData.dataSize);
      aclnnTensor->intArrayHostData.dataOri.resize(aclnnTensor->intArrayHostData.dataSize);
      std::transform(
          static_cast<int32_t *>(aclnnTensor->atbTensor.hostData),
          static_cast<int32_t *>(aclnnTensor->atbTensor.hostData) + aclnnTensor->atbTensor.dataSize / NUM4,
          aclnnTensor->intArrayHostData.data.data(), [](int32_t value) {
              return static_cast<int64_t>(value);
      });
      std::copy(static_cast<int32_t *>(aclnnTensor->atbTensor.hostData),
          static_cast<int32_t *>(aclnnTensor->atbTensor.hostData) +
              aclnnTensor->atbTensor.dataSize / sizeof(int32_t),
          aclnnTensor->intArrayHostData.dataOri.data());
      aclnnTensor->intArrayHostData.intArray = aclCreateIntArray(
          static_cast<int64_t *>(aclnnTensor->intArrayHostData.data.data()),
          aclnnTensor->intArrayHostData.dataSize);
      actualSeqLengthsKv_ = aclnnTensor->intArrayHostData.intArray;
    } else {
      aclnnTensor->needUpdateTensorDataPtr = true;
      atb::Tensor atbTensor = variantPack.inTensors.at(i);
      aclnnTensor->strides = GetCopyTensorStride(atbTensor.desc.shape);
      aclnnTensor->tensor = aclCreateTensor(atbTensor.desc.shape.dims,
                                            atbTensor.desc.shape.dimNum,
                                            atbTensor.desc.dtype,
                                            aclnnTensor->strides.data(),
                                            0,
                                            atbTensor.desc.format,
                                            atbTensor.desc.shape.dims,
                                            atbTensor.desc.shape.dimNum,
                                            atbTensor.deviceData);
      if (aclnnTensor->tensor == nullptr) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " InTensor index " << i
                                          << " create fail");
        return atb::ERROR_INTERNAL_ERROR;
      }
    }
    aclnnVariantPack.aclInTensors[i] = aclnnTensor;
  }
  aclTensor* key = aclnnVariantPack.aclInTensors.at(1)->tensor;   // 1: key tensor index
  aclTensor* value = aclnnVariantPack.aclInTensors.at(2)->tensor; // 2: value tensor index
  std::vector<aclTensor *> keyList{key};
  std::vector<aclTensor *> valueList{value};
  auto tensorKeyList = aclCreateTensorList(keyList.data(), keyList.size());
  auto tensorValueList = aclCreateTensorList(valueList.data(), valueList.size());
  aclnnVariantPack.aclInTensorList.resize(3);
  aclnnVariantPack.aclInTensorList.clear();
  aclnnVariantPack.aclInTensorList.push_back(nullptr);
  aclnnVariantPack.aclInTensorList.push_back(tensorKeyList);
  aclnnVariantPack.aclInTensorList.push_back(tensorValueList);
  return atb::NO_ERROR;
}

int FusedInferAttentionV2Operation::CreateAclNNOutTensorVariantPack(
    const atb::VariantPack& variantPack) {
  AclNNVariantPack& aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
  aclnnVariantPack.aclOutTensors.resize(GetOutputNum());
  for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); i++) {
    std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
    aclnnTensor->tensorIdx = i;
    aclnnTensor->needUpdateTensorDataPtr = true;
    aclnnTensor->atbTensor = variantPack.outTensors.at(i);
    atb::Tensor squeezedAtbTensor =
        SqueezeBatchSeq(variantPack.outTensors.at(i));
    aclnnTensor->strides = GetCopyTensorStride(squeezedAtbTensor.desc.shape);
    aclnnTensor->tensor = aclCreateTensor(squeezedAtbTensor.desc.shape.dims,
                                          squeezedAtbTensor.desc.shape.dimNum,
                                          squeezedAtbTensor.desc.dtype,
                                          aclnnTensor->strides.data(),
                                          0,
                                          squeezedAtbTensor.desc.format,
                                          squeezedAtbTensor.desc.shape.dims,
                                          squeezedAtbTensor.desc.shape.dimNum,
                                          squeezedAtbTensor.deviceData);
    if (aclnnTensor->tensor == nullptr) {
      ATB_SPEED_LOG_ERROR(this->opName_ << " OutTensor index " << i
                                        << " create fail");
      return atb::ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack.aclOutTensors[i] = aclnnTensor;
  }
  return atb::NO_ERROR;
}

int FusedInferAttentionV2Operation::SetAclNNWorkspaceExecutor() {
  ATB_SPEED_LOG_DEBUG(opName_
                      << "FusedInferAttentionV2Operation GetWorkspace Start!");
  AclNNVariantPack& aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
  aclTensor* query = aclnnVariantPack.aclInTensors.at(0)->tensor;
  aclTensor* attnOut = aclnnVariantPack.aclOutTensors.at(0)->tensor;
  int ret = aclnnFusedInferAttentionScoreV2GetWorkspaceSize(
        query,        // 0: query index
        aclnnVariantPack.aclInTensorList.at(1),             // 1: key cache index
        aclnnVariantPack.aclInTensorList.at(2),             // 2: value cache index
        nullptr, nullptr,                    // 4: attenMask
        actualSeqLengths_, actualSeqLengthsKv_, // 6: seq length index
        nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr,                   // 12: antiquantScale
        nullptr,                  // 13: antiquantOffset
        nullptr,                            // 14: blocktable
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        param_.numHeads, param_.scaleValue, param_.preTokens, param_.nextTokens,
        const_cast<char*>(param_.inputLayout.c_str()),
        param_.numKeyValueHeads, param_.sparseMode, param_.innerPrecise,
        0, 0, false, 0, 0,
        attnOut,      // 0: out tensor
        nullptr, &this->aclnnOpCache_->workspaceSize, &this->aclnnOpCache_->aclExecutor);
  if (ret != 0) {
    ATB_SPEED_LOG_ERROR(opName_
                      << "FusedInferAttentionV2Operation GetWorkspace Run Failed!");
  }
  return ret;

}

int FusedInferAttentionV2Operation::ExecuteAclNNOp(uint8_t* workspace,
                                                   aclrtStream& stream) {
  auto ret = aclnnFusedInferAttentionScoreV2(workspace, this->aclnnOpCache_->workspaceSize,
                                           this->aclnnOpCache_->aclExecutor, stream);
  if (ret != 0) {
    ATB_SPEED_LOG_ERROR(this->opName_
                        << " aclnnFusedInferAttentionScoreV2 execute fail");
  }
  return ret;
}
}  // namespace common
}  // namespace atb_speed