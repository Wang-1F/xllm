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
#ifndef ACLNN_FUSED_INFER_ATTN_V2_H
#define ACLNN_FUSED_INFER_ATTN_V2_H
#include <iostream>
#include <string>
#include <sstream>
#include "operations/aclnn/core/acl_nn_operation.h"
namespace atb_speed {
namespace common {
struct AclNNFusedInferAttnParam {
    /// A flag indicating whether the attention uses mask
    bool needMask = false;
    /// The number of query's heads
    int64_t numHeads;
    /// The scaling value
    double scaleValue = 1.0;
    /// The number of previous tokens related to attention's calculation, default to 214748647
    int64_t preTokens = 214748647;
    /// The number of post tokens related to attention's calculation, default to 214748647
    int64_t nextTokens = 214748647;
    /// The parameter indicating the layout of q, k, v input tensors (BSH、BSND、BNSD、BNSD_BSND)
    std::string inputLayout = "BNSD";
    /// The number of key and value's heads
    int64_t numKeyValueHeads;
    /// The parameter for the sparse mode
    /// 0: default mask, support no mask or full mask input (S1*S2)
    /// 1: allMask, only support full mask (S1*S2)
    /// 2: leftUpCausal mask, support improved mask (2048*2048)
    /// 3: rightDownCausal mask, support improved mask (2048*2048)
    /// 4: band mode mask, support improved mask (2048*2048)
    int64_t sparseMode = 0;
    /// The parameter for high precision/performance mode, default to 1
    /// 0: high precision mode without correction
    /// 1: high performance mode without correction
    /// 2: high precision mode with correction
    /// 3: high performance mode with correction
    int64_t innerPrecise = 0;
    std::string ToString() const
    {
        std::ostringstream oss;
        oss << "AclNNFlashAttentionParam {" << std::endl;
        oss << "  needMask: " << needMask << std::endl;
        oss << "  numHeads: " << numHeads << std::endl;
        oss << "  scaleValue: " << scaleValue << std::endl;
        oss << "  preTokens: " << preTokens << std::endl;
        oss << "  nextTokens: " << nextTokens << std::endl;
        oss << "  inputLayout: " << inputLayout << std::endl;
        oss << "  numKeyValueHeads: " << numKeyValueHeads << std::endl;
        oss << "  sparseMode: "  << sparseMode << std::endl;
        oss << "  innerPrecise: " << innerPrecise << std::endl;
        oss << "}";
        return oss.str();
    }
};


class FusedInferAttentionV2Operation : public AclNNOperation
{
public:
    explicit FusedInferAttentionV2Operation(const std::string &name, const AclNNFusedInferAttnParam& param);
    ~FusedInferAttentionV2Operation() override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) const override;

protected:
    int CreateAclNNVariantPack(const atb::VariantPack &variantPack) override;
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    int CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
    int CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;


private:
    AclNNFusedInferAttnParam param_;
    aclIntArray *actualSeqLengths_ = nullptr;
    aclIntArray *actualSeqLengthsKv_ = nullptr;
};

}  // namespace common
}  // namespace atb_speed

#endif