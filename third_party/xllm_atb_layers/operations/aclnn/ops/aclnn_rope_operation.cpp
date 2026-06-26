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
#include "aclnn_rope_operation.h"

#include "acl/acl.h"
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"
#include "aclnnop/aclnn_rotary_position_embedding.h"

namespace atb_speed {
namespace common {

AclnnRopeOperation::AclnnRopeOperation(const std::string &name, AclnnRopeParam param)
    : AclNNOperation(name), param_(param) {}

AclnnRopeOperation::~AclnnRopeOperation()
{
    ATB_SPEED_LOG_DEBUG("AclnnRopeOperation deconstructor");
    this->DestroyOperation();
}

atb::Status AclnnRopeOperation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                           atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape start");
    outTensorDescs.at(0) = inTensorDescs.at(0);
    ATB_SPEED_LOG_DEBUG(opName_ << " infer shape end");
    return atb::NO_ERROR;
}

uint32_t AclnnRopeOperation::GetInputNum() const
{
    // inputs: x, cos, sin
    return NUM3;
}

uint32_t AclnnRopeOperation::GetOutputNum() const
{
    // output: out
    return NUM1;
}

atb::Status AclnnRopeOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = CreateTensor(variantPack.inTensors.at(i), i);
        if (aclnnTensor->tensor == nullptr) {
            ATB_SPEED_LOG_ERROR(this->opName_ << " InTensor aclCreateTensor index " << i << " fail");
            return atb::ERROR_INTERNAL_ERROR;
        }
        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

atb::Status AclnnRopeOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclOutTensors.resize(GetOutputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = CreateTensor(variantPack.outTensors.at(i), i);
        if (aclnnTensor->tensor == nullptr) {
            ATB_SPEED_LOG_ERROR(this->opName_ << " OutTensor aclCreateTensor index " << i << " fail");
            return atb::ERROR_INTERNAL_ERROR;
        }
        aclnnVariantPack.aclOutTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

int AclnnRopeOperation::SetAclNNWorkspaceExecutor()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor start");
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;

    int ret = aclnnRotaryPositionEmbeddingGetWorkspaceSize(
        aclnnVariantPack.aclInTensors.at(0)->tensor,  // x
        aclnnVariantPack.aclInTensors.at(1)->tensor,  // cos
        aclnnVariantPack.aclInTensors.at(2)->tensor,  // sin
        param_.mode,                                  // mode
        aclnnVariantPack.aclOutTensors.at(0)->tensor, // out
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);

    ATB_SPEED_LOG_DEBUG(opName_ << " SetAclNNWorkspaceExecutor end, ret:"
                                << ret << ", workspaceSize:" << this->aclnnOpCache_->workspaceSize
                                << ", aclExecutor:" << this->aclnnOpCache_->aclExecutor);
    return ret;
}

int AclnnRopeOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp start");
    int ret = aclnnRotaryPositionEmbedding(
        workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
    ATB_SPEED_LOG_DEBUG(opName_ << " ExecuteAclNNOp end, ret:" << ret);
    return ret;
}
} // namespace common
} // namespace atb_speed
