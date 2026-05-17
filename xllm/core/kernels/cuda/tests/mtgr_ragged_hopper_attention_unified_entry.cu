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
  MTGR_NVTX_RANGE(1, "MTGR/kernel/unified_entry");
  MTGR_TRACE(1) << "[KERNEL] unified_entry begin match_mode=" << match_mode
                << " total_live_q=" << query_snd.size(0)
                << " max_request_len=" << max_request_len
                << " block_size=" << block_size;
  MTGR_TRACE(2) << "[KERNEL] unified_entry shapes query="
                << query_snd.sizes() << " key=" << key_snd.sizes()
                << " value=" << value_snd.sizes()
                << " segment_offsets=" << segment_offsets_i32.sizes()
                << " segment_rules=" << segment_rules_i32.sizes()
                << " q_seq_starts=" << q_seq_starts_i32.sizes()
                << " matched_prefix_lens=" << matched_prefix_lens_i32.sizes()
                << " key_cache=" << key_cache.sizes()
                << " block_table=" << block_table_i32.sizes();
  // Preserve the no_match fast path: dispatch mode is explicit and does not
  // depend on whether cache tensors are defined.
  if (match_mode == 0) {
    MTGR_NVTX_RANGE(1, "MTGR/kernel/no_match_dense_tma");
    MTGR_TRACE(1) << "[KERNEL] unified_entry route=no_match_dense_tma";
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
  MTGR_NVTX_RANGE(1,
                  match_mode == 2 ? "MTGR/kernel/mixed"
                                  : "MTGR/kernel/partial_only");
  MTGR_TRACE(1) << "[KERNEL] unified_entry route="
                << (match_mode == 2 ? "mixed" : "partial_only");
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
