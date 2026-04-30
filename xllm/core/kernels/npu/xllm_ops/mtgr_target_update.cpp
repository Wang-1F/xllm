/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/torch_npu.h>

#include "acl/acl.h"
#include "aclnn_mtgr_target_update.h"
#include "core/common/macros.h"
#include "core/kernels/npu/utils.h"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {

void mtgr_target_update(const torch::Tensor& target_query,
                        const torch::Tensor& target_key,
                        const torch::Tensor& target_value,
                        const torch::Tensor& prefix_out,
                        const torch::Tensor& prefix_lse,
                        torch::Tensor& out) {
  constexpr const char* kFunc = "mtgr_target_update";
  check_tensor(target_query, "target_query", kFunc);
  check_tensor(target_key, "target_key", kFunc);
  check_tensor(target_value, "target_value", kFunc);
  check_tensor(prefix_out, "prefix_out", kFunc);
  check_tensor(prefix_lse, "prefix_lse", kFunc);
  check_tensor(out, "out", kFunc);
  check_tensor_shapes_equal(target_query, target_key, kFunc);
  check_tensor_shapes_equal(target_query, target_value, kFunc);
  check_tensor_shapes_equal(target_query, out, kFunc);
  CHECK_EQ(target_query.dim(), 4) << kFunc << ": target_query must be BSND";
  CHECK_EQ(prefix_out.dim(), 4) << kFunc << ": prefix_out must be BSND";
  CHECK_EQ(prefix_out.size(0), target_query.size(0));
  CHECK_GE(prefix_out.size(1), target_query.size(1));
  CHECK_EQ(prefix_out.size(2), target_query.size(2));
  CHECK_EQ(prefix_out.size(3), target_query.size(3));
  CHECK_EQ(prefix_lse.dim(), 4) << kFunc << ": prefix_lse must be BNS1";
  CHECK_EQ(prefix_lse.size(0), target_query.size(0));
  CHECK_EQ(prefix_lse.size(1), target_query.size(2));
  CHECK_EQ(prefix_lse.size(2), prefix_out.size(1));
  CHECK_EQ(prefix_lse.size(3), 1);
  CHECK_EQ(prefix_lse.scalar_type(), torch::kFloat32);

  aclTensor* q_acl = nullptr;
  aclTensor* k_acl = nullptr;
  aclTensor* v_acl = nullptr;
  aclTensor* prefix_out_acl = nullptr;
  aclTensor* prefix_lse_acl = nullptr;
  aclTensor* out_acl = nullptr;
  create_acltensor(&q_acl, target_query);
  create_acltensor(&k_acl, target_key);
  create_acltensor(&v_acl, target_value);
  create_acltensor(&prefix_out_acl, prefix_out);
  create_acltensor(&prefix_lse_acl, prefix_lse);
  create_acltensor(&out_acl, out);

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  CHECK_ACL_SUCCESS(aclnnMtgrTargetUpdateGetWorkspaceSize(q_acl,
                                                          k_acl,
                                                          v_acl,
                                                          prefix_out_acl,
                                                          prefix_lse_acl,
                                                          out_acl,
                                                          &workspace_size,
                                                          &executor),
                    "mtgr_target_update: failed to get workspace size");

  void* workspace_addr = nullptr;
  if (workspace_size > 0) {
    CHECK_ACL_SUCCESS(aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST),
                      "mtgr_target_update: failed to malloc workspace");
  }
  const int32_t device_id = target_query.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  CHECK_ACL_SUCCESS(aclnnMtgrTargetUpdate(workspace_addr, workspace_size, executor, stream),
                    "mtgr_target_update: failed to launch");
  CHECK_ACL_SUCCESS(aclrtSynchronizeStream(stream),
                    "mtgr_target_update: failed to synchronize stream");
  if (workspace_addr != nullptr) {
    CHECK_ACL_SUCCESS(aclrtFree(workspace_addr),
                      "mtgr_target_update: failed to free workspace");
  }

  aclDestroyTensor(q_acl);
  aclDestroyTensor(k_acl);
  aclDestroyTensor(v_acl);
  aclDestroyTensor(prefix_out_acl);
  aclDestroyTensor(prefix_lse_acl);
  aclDestroyTensor(out_acl);
}

}  // namespace xllm::kernel::npu
