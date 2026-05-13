// Unified Hopper entrypoint. Public contract stays lean; packed-query launch
// metadata is derived here before dispatching into the WGMMA core.

void mtgr_ragged_segment_attention_hopper_unified_research_cuda(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    const torch::Tensor& segment_offsets_i32,
    const torch::Tensor& segment_rules_i32,
    const torch::Tensor& matched_prefix_lens_i32,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& block_table_i32,
    int64_t block_size,
    int64_t max_request_len,
    double sm_scale,
    torch::Tensor output_snd) {
  check_mtgr_ragged_segment_attention_hopper_unified_args(
      query_snd,
      key_snd,
      value_snd,
      segment_offsets_i32,
      segment_rules_i32,
      matched_prefix_lens_i32,
      key_cache,
      value_cache,
      block_table_i32,
      block_size,
      max_request_len,
      sm_scale,
      output_snd);
  // Preserve the no_match fast path: if the caller does not provide cache
  // metadata, unified dispatch is just a thin wrapper over the existing dense
  // Hopper research kernel.
  if (!mtgr_hopper_unified_uses_cache(
          key_cache, value_cache, block_table_i32)) {
    mtgr_ragged_segment_attention_hopper_research_cuda(query_snd,
                                                       key_snd,
                                                       value_snd,
                                                       segment_offsets_i32,
                                                       segment_rules_i32,
                                                       max_request_len,
                                                       sm_scale,
                                                       output_snd);
    return;
  }

  auto q_seq_starts_i32 = mtgr_hopper_build_unified_q_seq_starts(
      segment_offsets_i32, matched_prefix_lens_i32);
  dispatch_mtgr_ragged_segment_attention_hopper_unified_reference(
      query_snd,
      key_snd,
      value_snd,
      key_cache,
      value_cache,
      segment_offsets_i32,
      segment_rules_i32,
      q_seq_starts_i32,
      matched_prefix_lens_i32,
      block_table_i32,
      block_size,
      max_request_len,
      sm_scale,
      output_snd);
}
