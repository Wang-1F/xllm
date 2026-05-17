// Umbrella TU for Hopper ragged attention research code.
// Keep the research implementation physically split by role while preserving
// the existing single-file compile contract used by local extension builds.
#include "mtgr_ragged_hopper_attention_common.cu"
#include "mtgr_ragged_hopper_attention_research_entry.cu"
#include "mtgr_ragged_hopper_attention_unified_entry.cu"

}  // namespace xllm::kernel::cuda
