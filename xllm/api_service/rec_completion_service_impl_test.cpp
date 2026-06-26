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

#include "rec_completion_service_impl.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace xllm {
namespace {

TEST(RecCompletionServiceImplTest, BuildResponseUsesIntTensorForTokenOutputs) {
  RequestOutput req_output;
  SequenceOutput output0;
  output0.index = 0;
  output0.text = "12";
  output0.token_ids = {1, 2};
  output0.finish_reason = "stop";
  req_output.outputs.push_back(output0);

  SequenceOutput output1;
  output1.index = 1;
  output1.text = "34";
  output1.token_ids = {3, 4};
  output1.finish_reason = "stop";
  req_output.outputs.push_back(output1);

  proto::CompletionResponse response;
  ASSERT_TRUE(rec_completion_service_internal::build_response(
      "req-token", 123, "mtgr", req_output, &response));

  ASSERT_EQ(response.output_tensors_size(), 1);
  const auto& tensor = response.output_tensors(0);
  EXPECT_EQ(tensor.name(), "rec_result");
  EXPECT_EQ(tensor.datatype(), proto::DataType::INT32);
  ASSERT_EQ(tensor.shape_size(), 2);
  EXPECT_EQ(tensor.shape(0), 2);
  EXPECT_EQ(tensor.shape(1), 2);
  ASSERT_EQ(tensor.contents().int_contents_size(), 4);
  EXPECT_EQ(tensor.contents().int_contents(0), 1);
  EXPECT_EQ(tensor.contents().int_contents(1), 2);
  EXPECT_EQ(tensor.contents().int_contents(2), 3);
  EXPECT_EQ(tensor.contents().int_contents(3), 4);
}

TEST(RecCompletionServiceImplTest,
     BuildResponseUsesFloatTensorForEmbeddingOutputs) {
  RequestOutput req_output;
  SequenceOutput output0;
  output0.index = 0;
  output0.finish_reason = "stop";
  output0.embeddings = std::vector<float>{1.0f, 2.0f, 3.0f};
  req_output.outputs.push_back(output0);

  SequenceOutput output1;
  output1.index = 1;
  output1.finish_reason = "stop";
  output1.embeddings = std::vector<float>{4.0f, 5.0f, 6.0f};
  req_output.outputs.push_back(output1);

  proto::CompletionResponse response;
  ASSERT_TRUE(rec_completion_service_internal::build_response(
      "req-emb", 456, "mtgr", req_output, &response));

  ASSERT_EQ(response.output_tensors_size(), 1);
  const auto& tensor = response.output_tensors(0);
  EXPECT_EQ(tensor.name(), "rec_result");
  EXPECT_EQ(tensor.datatype(), proto::DataType::FLOAT);
  ASSERT_EQ(tensor.shape_size(), 2);
  EXPECT_EQ(tensor.shape(0), 2);
  EXPECT_EQ(tensor.shape(1), 3);
  ASSERT_EQ(tensor.contents().fp32_contents_size(), 6);
  EXPECT_FLOAT_EQ(tensor.contents().fp32_contents(0), 1.0f);
  EXPECT_FLOAT_EQ(tensor.contents().fp32_contents(1), 2.0f);
  EXPECT_FLOAT_EQ(tensor.contents().fp32_contents(2), 3.0f);
  EXPECT_FLOAT_EQ(tensor.contents().fp32_contents(3), 4.0f);
  EXPECT_FLOAT_EQ(tensor.contents().fp32_contents(4), 5.0f);
  EXPECT_FLOAT_EQ(tensor.contents().fp32_contents(5), 6.0f);
}

TEST(RecCompletionServiceImplTest,
     BuildResponseRejectsInconsistentEmbeddingDims) {
  RequestOutput req_output;
  SequenceOutput output0;
  output0.index = 0;
  output0.embeddings = std::vector<float>{1.0f, 2.0f};
  req_output.outputs.push_back(output0);

  SequenceOutput output1;
  output1.index = 1;
  output1.embeddings = std::vector<float>{3.0f};
  req_output.outputs.push_back(output1);

  proto::CompletionResponse response;
  EXPECT_FALSE(rec_completion_service_internal::build_response(
      "req-bad", 789, "mtgr", req_output, &response));
}

}  // namespace
}  // namespace xllm
