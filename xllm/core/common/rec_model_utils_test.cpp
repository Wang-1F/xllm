#include "rec_model_utils.h"

#include <gtest/gtest.h>

namespace xllm {

TEST(RecModelUtilsTest, MtgrUsesRecModelRoutingWithoutOneRecVocabPath) {
  EXPECT_FALSE(is_onerec_model_type("mtgr"));
  EXPECT_FALSE(is_llmrec_model_type("mtgr"));
  EXPECT_EQ(get_rec_model_kind("mtgr"), RecModelKind::kOneRec);
  EXPECT_EQ(get_rec_pipeline_type(get_rec_model_kind("mtgr")),
            RecPipelineType::kOneRecDefault);
}

}  // namespace xllm
