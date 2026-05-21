#include "rec_model_utils.h"

#include <gtest/gtest.h>

namespace xllm {

TEST(RecModelUtilsTest, MtgrUsesRecModelRoutingWithoutOneRecVocabPath) {
  EXPECT_FALSE(is_onerec_model_type("mtgr"));
  EXPECT_FALSE(is_llmrec_model_type("mtgr"));
  EXPECT_EQ(get_rec_model_kind("mtgr"), RecModelKind::kMtgr);
  EXPECT_EQ(get_rec_pipeline_type(get_rec_model_kind("mtgr")),
            RecPipelineType::kRecPrefillOnly);
}

TEST(RecModelUtilsTest, MtgrCachePolicyFollowsAttentionBackend) {
  const std::string old_backend = FLAGS_mtgr_attention_backend;

  FLAGS_mtgr_attention_backend = "flashinfer_token_mask";
  EXPECT_TRUE(is_mtgr_flashinfer_token_mask_backend());
  EXPECT_EQ(get_mtgr_cache_policy(), MTGRCachePolicy::kFullSequence);
  EXPECT_STREQ(current_mtgr_cache_policy_name(), "full_sequence");

  FLAGS_mtgr_attention_backend = "hopper";
  EXPECT_TRUE(is_mtgr_hopper_backend());
  EXPECT_EQ(get_mtgr_cache_policy(), MTGRCachePolicy::kPrefixOnly);
  EXPECT_STREQ(current_mtgr_cache_policy_name(), "prefix_only");

  FLAGS_mtgr_attention_backend = old_backend;
}

TEST(RecModelUtilsTest, MtgrCacheableLenFollowsBackendPolicy) {
  const std::string old_backend = FLAGS_mtgr_attention_backend;

  FLAGS_mtgr_attention_backend = "flashinfer_token_mask";
  EXPECT_EQ(mtgr_cacheable_len_for_policy(/*prefix_match_limit=*/40,
                                          /*logical_total_len=*/64),
            64);

  FLAGS_mtgr_attention_backend = "hopper";
  EXPECT_EQ(mtgr_cacheable_len_for_policy(/*prefix_match_limit=*/40,
                                          /*logical_total_len=*/64),
            40);

  FLAGS_mtgr_attention_backend = old_backend;
}

}  // namespace xllm
