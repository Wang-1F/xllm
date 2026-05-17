// Dense Hopper research entrypoints that preserve the pre-unified behavior.

void mtgr_ragged_segment_attention_hopper_research_cuda(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    const torch::Tensor& segment_offsets_i32,
    const torch::Tensor& segment_rules_i32,
    int64_t max_request_len,
    double sm_scale,
    torch::Tensor output_snd) {
  MTGR_NVTX_RANGE(1, "MTGR/kernel/dense_research_entry");
  MTGR_TRACE(1) << "[KERNEL] dense_research_entry begin total_q="
                << query_snd.size(0) << " max_request_len=" << max_request_len
                << " head_dim=" << query_snd.size(2);
  MTGR_TRACE(2) << "[KERNEL] dense_research_entry shapes query="
                << query_snd.sizes() << " key=" << key_snd.sizes()
                << " value=" << value_snd.sizes()
                << " segment_offsets=" << segment_offsets_i32.sizes()
                << " segment_rules=" << segment_rules_i32.sizes();
  dispatch_mtgr_ragged_segment_attention_hopper_wgmma_tma_qk(
      query_snd,
      key_snd,
      value_snd,
      segment_offsets_i32,
      segment_rules_i32,
      max_request_len,
      sm_scale,
      output_snd);
}
