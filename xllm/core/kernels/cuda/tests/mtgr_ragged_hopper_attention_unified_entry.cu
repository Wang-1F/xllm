// Unified Hopper entrypoint. Public contract stays lean; packed-query launch
// metadata is derived here before dispatching into the WGMMA core.

void mtgr_ragged_segment_attention_hopper_unified_research_cuda(
    const torch::Tensor& query_snd,
    const torch::Tensor& key_snd,
    const torch::Tensor& value_snd,
    const torch::Tensor& segment_offsets_i32,
    const torch::Tensor& segment_rules_i32,
    const torch::Tensor& q_seq_starts_i32,
    const torch::Tensor& matched_prefix_lens_i32,
    int64_t match_mode,
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
      q_seq_starts_i32,
      matched_prefix_lens_i32,
      match_mode,
      key_cache,
      value_cache,
      block_table_i32,
      block_size,
      max_request_len,
      sm_scale,
      output_snd);
  // Preserve the no_match fast path: dispatch mode is explicit and does not
  // depend on whether cache tensors are defined.
  if (match_mode == 0) {
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

  auto dispatch_unified =
      match_mode == 2
          ? dispatch_mtgr_ragged_segment_attention_hopper_unified_mixed
          : dispatch_mtgr_ragged_segment_attention_hopper_unified_partial_only;
  dispatch_unified(query_snd,
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
