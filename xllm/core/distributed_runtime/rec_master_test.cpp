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

#include "rec_master.h"

#include <gtest/gtest.h>

#include <optional>
#include <vector>

namespace xllm {
namespace {

TEST(RecMasterInternalTest,
     BuildMtgrPromptTokensKeepsOnlyPrefixCacheableForHopper) {
  const std::optional<std::vector<int32_t>> prompt_tokens =
      std::vector<int32_t>{10, 11, 12, 13, 14, 15};
  const auto built = rec_master_internal::build_mtgr_prompt_tokens(
      prompt_tokens,
      /*total_seq_len=*/6,
      /*cacheable_prefix_len=*/4,
      MTGRCachePolicy::kPrefixOnly,
      /*unique_salt=*/7);

  ASSERT_EQ(built.size(), 6u);
  EXPECT_EQ(built[0], 10);
  EXPECT_EQ(built[1], 11);
  EXPECT_EQ(built[2], 12);
  EXPECT_EQ(built[3], 13);
  EXPECT_NE(built[4], 14);
  EXPECT_NE(built[5], 15);
}

TEST(RecMasterInternalTest,
     BuildMtgrPromptTokensKeepsFullSequenceCacheableForFlashInferBase) {
  const std::optional<std::vector<int32_t>> prompt_tokens =
      std::vector<int32_t>{10, 11, 12, 13, 14, 15};
  const auto built = rec_master_internal::build_mtgr_prompt_tokens(
      prompt_tokens,
      /*total_seq_len=*/6,
      /*cacheable_prefix_len=*/4,
      MTGRCachePolicy::kFullSequence,
      /*unique_salt=*/7);

  ASSERT_EQ(built.size(), 6u);
  EXPECT_EQ(built[0], 10);
  EXPECT_EQ(built[1], 11);
  EXPECT_EQ(built[2], 12);
  EXPECT_EQ(built[3], 13);
  EXPECT_EQ(built[4], 14);
  EXPECT_EQ(built[5], 15);
}

TEST(RecMasterInternalTest, BuildMtgrPromptTokensWithoutIdsDisablesReuse) {
  const auto built = rec_master_internal::build_mtgr_prompt_tokens(
      std::nullopt,
      /*total_seq_len=*/5,
      /*cacheable_prefix_len=*/4,
      MTGRCachePolicy::kFullSequence,
      /*unique_salt=*/11);

  ASSERT_EQ(built.size(), 5u);
  for (size_t i = 0; i < built.size(); ++i) {
    EXPECT_LT(built[i], 0);
    for (size_t j = i + 1; j < built.size(); ++j) {
      EXPECT_NE(built[i], built[j]);
    }
  }
}

}  // namespace
}  // namespace xllm
