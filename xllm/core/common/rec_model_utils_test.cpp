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
  const std::string old_policy = FLAGS_mtgr_kv_cache_policy;

  FLAGS_mtgr_kv_cache_policy = "auto";
  FLAGS_mtgr_attention_backend = "flashinfer_token_mask";
  EXPECT_TRUE(is_mtgr_flashinfer_token_mask_backend());
  EXPECT_EQ(get_mtgr_cache_policy(), MTGRCachePolicy::kFullSequence);
  EXPECT_STREQ(current_mtgr_cache_policy_name(), "full_cache");

  FLAGS_mtgr_attention_backend = "hopper";
  EXPECT_TRUE(is_mtgr_hopper_backend());
  EXPECT_EQ(get_mtgr_cache_policy(), MTGRCachePolicy::kPrefixOnly);
  EXPECT_STREQ(current_mtgr_cache_policy_name(), "prefix_only");

  FLAGS_mtgr_kv_cache_policy = "prefix_only_value_gated";
  FLAGS_mtgr_attention_backend = "flashinfer_token_mask";
  EXPECT_EQ(get_mtgr_cache_policy(), MTGRCachePolicy::kPrefixOnlyValueGated);
  EXPECT_STREQ(current_mtgr_cache_policy_name(), "prefix_only_value_gated");

  FLAGS_mtgr_kv_cache_policy = "dynamic_length_admission";
  EXPECT_EQ(get_mtgr_cache_policy(), MTGRCachePolicy::kDynamicLengthAdmission);
  EXPECT_STREQ(current_mtgr_cache_policy_name(), "dynamic_length_admission");

  FLAGS_mtgr_kv_cache_policy = "rha_writeback";
  EXPECT_EQ(get_mtgr_cache_policy(), MTGRCachePolicy::kPrefixOnlyReuseHorizon);
  EXPECT_STREQ(current_mtgr_cache_policy_name(),
               "prefix_only_reuse_horizon");

  FLAGS_mtgr_kv_cache_policy = old_policy;
  FLAGS_mtgr_attention_backend = old_backend;
}

TEST(RecModelUtilsTest, MtgrCacheableLenFollowsBackendPolicy) {
  const std::string old_backend = FLAGS_mtgr_attention_backend;
  const std::string old_policy = FLAGS_mtgr_kv_cache_policy;

  FLAGS_mtgr_kv_cache_policy = "auto";
  FLAGS_mtgr_attention_backend = "flashinfer_token_mask";
  EXPECT_EQ(mtgr_cacheable_len_for_policy(/*prefix_match_limit=*/40,
                                          /*logical_total_len=*/64),
            64);

  FLAGS_mtgr_attention_backend = "hopper";
  EXPECT_EQ(mtgr_cacheable_len_for_policy(/*prefix_match_limit=*/40,
                                          /*logical_total_len=*/64),
            40);

  FLAGS_mtgr_kv_cache_policy = old_policy;
  FLAGS_mtgr_attention_backend = old_backend;
}

TEST(RecModelUtilsTest, MtgrDynamicLengthAdmissionUsesFullCacheLengths) {
  const std::string old_policy = FLAGS_mtgr_kv_cache_policy;

  FLAGS_mtgr_kv_cache_policy = "dynamic_length_admission";

  const auto decision =
      mtgr_cache_policy_decision(/*prefix_match_limit=*/512,
                                 /*logical_total_len=*/2048);
  EXPECT_EQ(decision.policy, MTGRCachePolicy::kDynamicLengthAdmission);
  EXPECT_TRUE(decision.cache_admitted);
  EXPECT_TRUE(decision.runtime_admission);
  EXPECT_EQ(decision.cacheable_len, 2048);
  EXPECT_EQ(decision.writeback_len, 2048);

  FLAGS_mtgr_kv_cache_policy = old_policy;
}

TEST(RecModelUtilsTest, MtgrReuseHorizonUsesPrefixOnlyRuntimeWriteback) {
  const std::string old_policy = FLAGS_mtgr_kv_cache_policy;

  FLAGS_mtgr_kv_cache_policy = "prefix_only_reuse_horizon";

  const auto decision =
      mtgr_cache_policy_decision(/*prefix_match_limit=*/512,
                                 /*logical_total_len=*/2048);
  EXPECT_EQ(decision.policy, MTGRCachePolicy::kPrefixOnlyReuseHorizon);
  EXPECT_TRUE(decision.cache_admitted);
  EXPECT_TRUE(decision.runtime_admission);
  EXPECT_EQ(decision.cacheable_len, 512);
  EXPECT_EQ(decision.writeback_len, 512);

  FLAGS_mtgr_kv_cache_policy = "reuse_horizon";
  EXPECT_EQ(get_mtgr_cache_policy(), MTGRCachePolicy::kPrefixOnlyReuseHorizon);

  FLAGS_mtgr_kv_cache_policy = old_policy;
}

TEST(RecModelUtilsTest, MtgrDynamicLengthAdmissionThresholdUsesFreeBlocks) {
  EXPECT_EQ(mtgr_dynamic_length_admission_threshold(/*free_blocks=*/71,
                                                    /*total_blocks=*/100),
            1500);
  EXPECT_EQ(mtgr_dynamic_length_admission_threshold(/*free_blocks=*/70,
                                                    /*total_blocks=*/100),
            2000);
  EXPECT_EQ(mtgr_dynamic_length_admission_threshold(/*free_blocks=*/69,
                                                    /*total_blocks=*/100),
            2000);

  EXPECT_TRUE(mtgr_dynamic_length_admission_is_admitted(
      /*request_len=*/1501, /*free_blocks=*/71, /*total_blocks=*/100));
  EXPECT_FALSE(mtgr_dynamic_length_admission_is_admitted(
      /*request_len=*/1500, /*free_blocks=*/71, /*total_blocks=*/100));
  EXPECT_TRUE(mtgr_dynamic_length_admission_is_admitted(
      /*request_len=*/2001, /*free_blocks=*/70, /*total_blocks=*/100));
  EXPECT_FALSE(mtgr_dynamic_length_admission_is_admitted(
      /*request_len=*/2000, /*free_blocks=*/70, /*total_blocks=*/100));
}

TEST(RecModelUtilsTest, MtgrValueGatedDecisionUsesPrefixLenThreshold) {
  const std::string old_policy = FLAGS_mtgr_kv_cache_policy;
  const std::string old_value_fn = FLAGS_mtgr_kv_cache_value_fn;
  const int32_t old_threshold = FLAGS_mtgr_kv_cache_prefix_len_threshold;

  FLAGS_mtgr_kv_cache_policy = "prefix_only_value_gated";
  FLAGS_mtgr_kv_cache_value_fn = "prefix_len";
  FLAGS_mtgr_kv_cache_prefix_len_threshold = 2000;

  const auto low =
      mtgr_cache_policy_decision(/*prefix_match_limit=*/1999,
                                 /*logical_total_len=*/3000);
  EXPECT_EQ(low.policy, MTGRCachePolicy::kPrefixOnlyValueGated);
  EXPECT_FALSE(low.cache_admitted);
  EXPECT_FALSE(low.high_value);
  EXPECT_EQ(low.cacheable_len, 0);
  EXPECT_EQ(low.writeback_len, 0);

  const auto high =
      mtgr_cache_policy_decision(/*prefix_match_limit=*/2000,
                                 /*logical_total_len=*/3200);
  EXPECT_TRUE(high.cache_admitted);
  EXPECT_TRUE(high.high_value);
  EXPECT_EQ(high.cacheable_len, 2000);
  EXPECT_EQ(high.writeback_len, 2000);

  FLAGS_mtgr_kv_cache_policy = old_policy;
  FLAGS_mtgr_kv_cache_value_fn = old_value_fn;
  FLAGS_mtgr_kv_cache_prefix_len_threshold = old_threshold;
}

TEST(RecModelUtilsTest, MtgrHighValueClassificationIsPolicyIndependent) {
  const std::string old_policy = FLAGS_mtgr_kv_cache_policy;
  const std::string old_value_fn = FLAGS_mtgr_kv_cache_value_fn;
  const int32_t old_threshold = FLAGS_mtgr_kv_cache_prefix_len_threshold;

  FLAGS_mtgr_kv_cache_value_fn = "prefix_len";
  FLAGS_mtgr_kv_cache_prefix_len_threshold = 2000;

  FLAGS_mtgr_kv_cache_policy = "full_cache";
  const auto full_low =
      mtgr_cache_policy_decision(/*prefix_match_limit=*/1500,
                                 /*logical_total_len=*/2800);
  EXPECT_TRUE(full_low.cache_admitted);
  EXPECT_FALSE(full_low.high_value);
  EXPECT_EQ(full_low.cacheable_len, 2800);

  FLAGS_mtgr_kv_cache_policy = "prefix_only";
  const auto prefix_high =
      mtgr_cache_policy_decision(/*prefix_match_limit=*/2500,
                                 /*logical_total_len=*/3600);
  EXPECT_TRUE(prefix_high.cache_admitted);
  EXPECT_TRUE(prefix_high.high_value);
  EXPECT_EQ(prefix_high.cacheable_len, 2500);

  FLAGS_mtgr_kv_cache_policy = old_policy;
  FLAGS_mtgr_kv_cache_value_fn = old_value_fn;
  FLAGS_mtgr_kv_cache_prefix_len_threshold = old_threshold;
}

}  // namespace xllm
