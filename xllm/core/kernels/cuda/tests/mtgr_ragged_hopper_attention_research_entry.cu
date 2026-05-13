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
  check_mtgr_ragged_segment_attention_hopper_common_args(query_snd,
                                                         key_snd,
                                                         value_snd,
                                                         segment_offsets_i32,
                                                         segment_rules_i32,
                                                         output_snd,
                                                         max_request_len,
                                                         sm_scale);
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
