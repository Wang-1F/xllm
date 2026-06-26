/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "block_manager_pool.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <unordered_map>

#include "block_manager_impl.h"
#include "common/global_flags.h"
#include "common/metrics.h"
#include "common/rec_model_utils.h"
#include "concurrent_block_manager_impl.h"
#include "framework/xtensor/page_allocator.h"
#include "framework/xtensor/phy_page_pool.h"
#include "framework/xtensor/xtensor_block_manager_impl.h"
#include "util/mtgr_trace.h"

namespace xllm {
namespace {

constexpr const char* kMtgrSegmentOffsetsName = "segment_offsets";
constexpr const char* kMtgrEntityIdName = "entity_id";

struct MTGRRuntimeAdmissionDecision {
  bool applies = false;
  bool admitted = true;
  size_t request_len = 0;
  size_t free_blocks = 0;
  size_t total_blocks = 0;
  int32_t threshold = 0;
};

struct MTGRReuseHorizonFamilyState {
  uint64_t seen_count = 0;
  uint64_t hit_count = 0;
  uint64_t admitted_count = 0;
  uint64_t skipped_count = 0;
  uint64_t last_seen_index = 0;
};

struct MTGRReuseHorizonDecision {
  bool applies = false;
  bool admitted = true;
  bool pressure = false;
  bool reuse_evidence = false;
  bool cold_low_reuse = false;
  bool has_entity_id = false;
  bool entity_hot_skip_candidate = false;
  size_t request_len = 0;
  size_t matched_blocks = 0;
  size_t free_blocks = 0;
  size_t total_blocks = 0;
  double free_ratio = 0.0;
  uint64_t family_key = 0;
  uint64_t request_index = 0;
  uint64_t prior_seen_count = 0;
  uint64_t prior_hit_count = 0;
  int64_t entity_id = 0;
  const char* reason = "not_applicable";
};

std::mutex g_mtgr_rha_state_mu;
std::unordered_map<uint64_t, MTGRReuseHorizonFamilyState> g_mtgr_rha_state;
std::atomic<uint64_t> g_mtgr_rha_request_index{0};

uint64_t mtgr_reuse_horizon_prefix_family_key(const Slice<int32_t>& token_ids,
                                              size_t block_size) {
  constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
  constexpr uint64_t kFnvPrime = 1099511628211ULL;
  uint64_t hash = kFnvOffset;
  const size_t family_tokens =
      std::min(token_ids.size(), std::max<size_t>(block_size, 1));
  for (size_t i = 0; i < family_tokens; ++i) {
    const uint32_t token = static_cast<uint32_t>(token_ids[i]);
    hash ^= static_cast<uint64_t>(token);
    hash *= kFnvPrime;
    hash ^= static_cast<uint64_t>(token >> 16);
    hash *= kFnvPrime;
  }
  hash ^= static_cast<uint64_t>(family_tokens);
  hash *= kFnvPrime;
  return hash;
}

uint64_t mtgr_reuse_horizon_entity_key(int64_t entity_id) {
  constexpr uint64_t kEntityNamespace = 0x9e3779b97f4a7c15ULL;
  uint64_t value = static_cast<uint64_t>(std::hash<int64_t>{}(entity_id));
  value ^= value >> 33;
  value *= 0xff51afd7ed558ccdULL;
  value ^= value >> 33;
  value *= 0xc4ceb9fe1a85ec53ULL;
  value ^= value >> 33;
  return value ^ kEntityNamespace;
}

std::optional<int64_t> get_mtgr_entity_id(const Sequence* sequence) {
  if (sequence == nullptr) {
    return std::nullopt;
  }
  auto entity_opt =
      sequence->get_mm_data().get<torch::Tensor>(kMtgrEntityIdName);
  if (!entity_opt.has_value()) {
    return std::nullopt;
  }
  auto entity = entity_opt.value().to(torch::kInt64).cpu().contiguous();
  CHECK_GT(entity.numel(), 0) << "MTGR entity_id tensor must not be empty";
  return entity.view({-1}).data_ptr<int64_t>()[0];
}

size_t get_policy_cacheable_tokens(const Sequence* sequence,
                                   size_t default_num_tokens) {
  DCHECK(sequence != nullptr);
  if (sequence->rec_type() != RecType::kMtgr) {
    return default_num_tokens;
  }

  auto offsets_opt =
      sequence->get_mm_data().get<torch::Tensor>(kMtgrSegmentOffsetsName);
  CHECK(offsets_opt.has_value())
      << "MTGR sequence requires segment_offsets to derive cache policy";
  auto offsets =
      offsets_opt.value().to(torch::kInt32).cpu().contiguous();
  if (offsets.dim() == 2 && offsets.size(0) == 1) {
    offsets = offsets.view({offsets.size(1)}).contiguous();
  }
  CHECK_EQ(offsets.dim(), 1) << "MTGR segment_offsets must be 1-D or [1, N]";
  CHECK_GE(offsets.size(0), 2)
      << "MTGR segment_offsets must include at least begin and end";

  const auto* offsets_ptr = offsets.data_ptr<int32_t>();
  const int32_t prefix_match_limit =
      offsets_ptr[static_cast<int64_t>(offsets.size(0)) - 2];
  const int32_t logical_total_len =
      offsets_ptr[static_cast<int64_t>(offsets.size(0)) - 1];
  CHECK_EQ(static_cast<size_t>(logical_total_len), sequence->num_tokens())
      << "MTGR segment_offsets last element must match sequence length";
  const MTGRCachePolicyDecision decision =
      mtgr_cache_policy_decision(prefix_match_limit, logical_total_len);
  return static_cast<size_t>(decision.cacheable_len);
}

MTGRRuntimeAdmissionDecision get_mtgr_runtime_admission_decision(
    const Sequence* sequence,
    const BlockManager* block_manager,
    size_t request_len) {
  MTGRRuntimeAdmissionDecision decision;
  if (sequence == nullptr || block_manager == nullptr ||
      sequence->rec_type() != RecType::kMtgr ||
      !mtgr_uses_runtime_admission(get_mtgr_cache_policy())) {
    return decision;
  }

  decision.applies = true;
  decision.request_len = request_len;
  decision.free_blocks = block_manager->num_free_blocks();
  decision.total_blocks = block_manager->num_total_blocks();
  decision.threshold = mtgr_dynamic_length_admission_threshold(
      decision.free_blocks, decision.total_blocks);
  decision.admitted = mtgr_dynamic_length_admission_is_admitted(
      static_cast<int32_t>(request_len),
      decision.free_blocks,
      decision.total_blocks);
  return decision;
}

MTGRReuseHorizonDecision get_mtgr_reuse_horizon_admission_decision(
    Sequence* sequence,
    const BlockManager* block_manager,
    const Slice<int32_t>& token_ids) {
  MTGRReuseHorizonDecision decision;
  if (sequence == nullptr || block_manager == nullptr ||
      sequence->rec_type() != RecType::kMtgr ||
      get_mtgr_cache_policy() != MTGRCachePolicy::kPrefixOnlyReuseHorizon) {
    return decision;
  }

  CHECK_GE(FLAGS_mtgr_rha_pressure_free_block_ratio, 0.0)
      << "mtgr_rha_pressure_free_block_ratio must be non-negative";
  CHECK_LE(FLAGS_mtgr_rha_pressure_free_block_ratio, 1.0)
      << "mtgr_rha_pressure_free_block_ratio must be at most 1.0";
  CHECK_GE(FLAGS_mtgr_rha_min_reuse_count, 1)
      << "mtgr_rha_min_reuse_count must be at least 1";
  CHECK_GE(FLAGS_mtgr_rha_cold_prefix_len_threshold, 0)
      << "mtgr_rha_cold_prefix_len_threshold must be non-negative";
  CHECK_GE(FLAGS_mtgr_rha_entity_hot_skip_min_seen, 0)
      << "mtgr_rha_entity_hot_skip_min_seen must be non-negative";

  decision.applies = true;
  decision.request_len = token_ids.size();
  decision.matched_blocks = sequence->kv_state().shared_kv_blocks_num();
  decision.free_blocks = block_manager->num_free_blocks();
  decision.total_blocks = block_manager->num_total_blocks();
  decision.free_ratio =
      decision.total_blocks == 0
          ? 0.0
          : static_cast<double>(decision.free_blocks) /
                static_cast<double>(decision.total_blocks);
  decision.pressure =
      decision.total_blocks > 0 &&
      decision.free_ratio <= FLAGS_mtgr_rha_pressure_free_block_ratio;
  const std::optional<int64_t> entity_id = get_mtgr_entity_id(sequence);
  decision.has_entity_id = entity_id.has_value();
  if (decision.has_entity_id) {
    decision.entity_id = entity_id.value();
    decision.family_key = mtgr_reuse_horizon_entity_key(decision.entity_id);
  } else {
    decision.family_key = mtgr_reuse_horizon_prefix_family_key(
        token_ids, block_manager->block_size());
  }
  decision.request_index =
      g_mtgr_rha_request_index.fetch_add(1, std::memory_order_relaxed) + 1;

  {
    std::lock_guard<std::mutex> lock(g_mtgr_rha_state_mu);
    auto it = g_mtgr_rha_state.find(decision.family_key);
    if (it == g_mtgr_rha_state.end()) {
      const uint64_t max_entries = FLAGS_mtgr_rha_state_max_entries;
      if (max_entries > 0 && g_mtgr_rha_state.size() >= max_entries) {
        g_mtgr_rha_state.clear();
      }
      it = g_mtgr_rha_state.emplace(decision.family_key,
                                    MTGRReuseHorizonFamilyState{})
               .first;
    }

    MTGRReuseHorizonFamilyState& state = it->second;
    decision.prior_seen_count = state.seen_count;
    decision.prior_hit_count = state.hit_count;
    decision.reuse_evidence =
        decision.matched_blocks > 0 || state.hit_count > 0 ||
        state.seen_count >=
            static_cast<uint64_t>(FLAGS_mtgr_rha_min_reuse_count);
    if (decision.has_entity_id) {
      decision.entity_hot_skip_candidate =
          FLAGS_mtgr_rha_entity_hot_skip_min_seen > 0 && decision.pressure &&
          decision.matched_blocks > 0 &&
          state.seen_count >= static_cast<uint64_t>(
                                  FLAGS_mtgr_rha_entity_hot_skip_min_seen) &&
          FLAGS_mtgr_rha_cold_prefix_len_threshold > 0 &&
          decision.request_len <=
              static_cast<size_t>(FLAGS_mtgr_rha_cold_prefix_len_threshold);
      decision.admitted = !decision.entity_hot_skip_candidate;
      decision.reason =
          !decision.pressure
              ? "no_pressure"
              : decision.entity_hot_skip_candidate
                    ? "entity_hot_short"
                    : "entity_default_admit";
    } else {
      decision.cold_low_reuse =
          decision.matched_blocks == 0 && state.hit_count == 0 &&
          state.seen_count <
              static_cast<uint64_t>(FLAGS_mtgr_rha_min_reuse_count) &&
          FLAGS_mtgr_rha_cold_prefix_len_threshold > 0 &&
          decision.request_len <=
              static_cast<size_t>(FLAGS_mtgr_rha_cold_prefix_len_threshold);
      decision.admitted =
          !decision.pressure || decision.reuse_evidence ||
          !decision.cold_low_reuse;
      decision.reason = !decision.pressure
                            ? "no_pressure"
                            : decision.reuse_evidence
                                  ? "reuse_evidence"
                                  : decision.cold_low_reuse
                                        ? "cold_low_reuse"
                                        : "length_protected";
    }

    ++state.seen_count;
    if (decision.matched_blocks > 0) {
      ++state.hit_count;
    }
    if (decision.admitted) {
      ++state.admitted_count;
    } else {
      ++state.skipped_count;
    }
    state.last_seen_index = decision.request_index;
  }

  return decision;
}

}  // namespace

BlockManagerPool::BlockManagerPool(const Options& options, int32_t dp_size)
    : options_(options) {
  CHECK(dp_size > 0) << "dp_size must be greater than 0";
  block_managers_.reserve(dp_size);
  embedding_managers_.reserve(dp_size);

  BlockManager::Options block_options;
  block_options.num_blocks(options_.num_blocks())
      .block_size(options_.block_size())
      .enable_prefix_cache(options_.enable_prefix_cache())
      .enable_disagg_pd(options_.enable_disagg_pd())
      .enable_cache_upload(options_.host_num_blocks() > 0
                               ? false
                               : options_.enable_cache_upload());

  for (int32_t i = 0; i < dp_size; ++i) {
    if (options_.enable_xtensor()) {
      // Use XTensorBlockManagerImpl for xtensor mode
      CHECK_GT(options_.num_layers(), 0)
          << "num_layers must be set when enable_xtensor is true";
      CHECK_GT(options_.slot_size(), 0)
          << "slot_size must be set when enable_xtensor is true";
      size_t page_size = FLAGS_phy_page_granularity_size;
      // In the current implementation, K and V must be the same size, so we
      // divide by 2.
      size_t block_mem_size =
          static_cast<size_t>(options_.block_size()) * options_.slot_size() / 2;
      block_managers_.emplace_back(
          std::make_unique<XTensorBlockManagerImpl>(block_options,
                                                    options_.num_layers(),
                                                    block_mem_size,
                                                    page_size,
                                                    /*dp_rank=*/i,
                                                    options_.model_id()));
    } else if (options.enable_disagg_pd() || options_.enable_kvcache_store()) {
      block_managers_.emplace_back(
          std::make_unique<ConcurrentBlockManagerImpl>(block_options));
    } else {
      block_managers_.emplace_back(
          std::make_unique<BlockManagerImpl>(block_options));
    }
    // since one sequence only has one embedding block,
    // FLAGS_max_seqs_per_batch + 1 is enough.
    embedding_managers_.emplace_back(
        std::make_unique<EmbeddingManager>(FLAGS_max_seqs_per_batch + 2));
  }
  reset_transfer_infos();
}

int32_t BlockManagerPool::get_manager_with_max_free_blocks() const {
  if (block_managers_.empty()) {
    return 0;
  }

  size_t max_index = 0;
  size_t max_free = block_managers_[0]->num_free_blocks();

  for (size_t i = 1; i < block_managers_.size(); ++i) {
    const size_t current_free = block_managers_[i]->num_free_blocks();
    if (current_free > max_free) {
      max_free = current_free;
      max_index = i;
    }
  }
  return max_index;
}

int32_t BlockManagerPool::get_dp_rank(Sequence* sequence) const {
  int32_t dp_rank;
  if (sequence->dp_rank() >= 0) {
    dp_rank = sequence->dp_rank();
  } else {
    dp_rank = get_manager_with_max_free_blocks();
    sequence->set_dp_rank(dp_rank);
  }
  return dp_rank;
}

bool BlockManagerPool::allocate_embedding_id(Sequence* sequence,
                                             int32_t dp_rank) {
  CHECK(sequence != nullptr);
  CHECK_GE(dp_rank, 0);
  CHECK_LT(static_cast<size_t>(dp_rank), embedding_managers_.size());
  if (sequence->has_embedding_id()) {
    return true;
  }

  auto embedding_blocks = embedding_managers_[dp_rank]->allocate(1);
  if (embedding_blocks.empty()) {
    LOG(ERROR) << "Failed to allocate embedding block!";
    return false;
  }
  sequence->set_embedding_block(std::move(embedding_blocks[0]));
  return true;
}

void BlockManagerPool::deallocate_embedding_id(Sequence* sequence,
                                               int32_t dp_rank) {
  DCHECK(sequence != nullptr);
  CHECK_GE(dp_rank, 0);
  CHECK_LT(static_cast<size_t>(dp_rank), embedding_managers_.size());
  auto embedding_block = sequence->reset_embedding_block();
  if (!embedding_block.is_valid()) {
    return;
  }

  // std::vector<Block> embedding_blocks;
  // embedding_blocks.emplace_back(std::move(embedding_block));
  embedding_managers_[dp_rank]->deallocate({&embedding_block, 1});
}

void BlockManagerPool::deallocate(Request* request) {
  DCHECK(request != nullptr);
  for (auto& sequence : request->sequences()) {
    deallocate(sequence.get());
  }
}

void BlockManagerPool::deallocate(std::vector<Sequence*>& sequences) {
  for (auto* sequence : sequences) {
    deallocate(sequence);
  }
}

void BlockManagerPool::deallocate(Sequence* sequence) {
  DCHECK(sequence != nullptr);
  // add blocks to the prefix cache
  int32_t dp_rank = get_dp_rank(sequence);
  cache(sequence);
  block_managers_[dp_rank]->deallocate(sequence->kv_state().kv_blocks());
  deallocate_embedding_id(sequence, dp_rank);
  // release the blocks after prefix cache insertion
  sequence->reset();
}

std::vector<std::vector<BlockTransferInfo>>*
BlockManagerPool::get_swap_block_transfer_infos() {
  return &swap_block_transfer_infos_;
}

void BlockManagerPool::reset_transfer_infos() {
  swap_block_transfer_infos_.clear();
  swap_block_transfer_infos_.resize(block_managers_.size());
}

bool BlockManagerPool::allocate(Sequence* sequence) {
  DCHECK(sequence != nullptr);
  return allocate(sequence,
                  get_policy_cacheable_tokens(sequence, sequence->num_tokens()));
}

bool BlockManagerPool::allocate(std::vector<Sequence*>& sequences) {
  for (auto* sequence : sequences) {
    DCHECK(sequence != nullptr);
    if (!allocate(sequence,
                  get_policy_cacheable_tokens(sequence,
                                              sequence->num_tokens()))) {
      // should we gurantee the atomicity of the allocation? all or nothing?
      return false;
    }
  }
  return true;
}

bool BlockManagerPool::allocate(Sequence* sequence, size_t num_tokens) {
  AUTO_COUNTER(allocate_blocks_latency_seconds);
  DCHECK(sequence != nullptr);
  const size_t cacheable_num_tokens =
      get_policy_cacheable_tokens(sequence, num_tokens);
  int32_t dp_rank = get_dp_rank(sequence);
  const bool needs_embedding_id = !sequence->has_embedding_id();
  if (needs_embedding_id && !allocate_embedding_id(sequence, dp_rank)) {
    return false;
  }

  // first try to allocate shared blocks
  if (sequence->kv_state().num_kv_blocks() == 0) {
    BlockManagerPool::allocate_shared(sequence, cacheable_num_tokens);
  }

  const size_t num_blocks = sequence->kv_state().num_kv_blocks();
  // round up to the nearest block number
  const size_t block_size = options_.block_size();
  const size_t num_blocks_needed =
      (cacheable_num_tokens + block_size - 1) / block_size;
  if (num_blocks_needed <= num_blocks) {
    return process_beam_search(sequence, /*need_swap*/ true);
  }
  process_beam_search(sequence);

  const uint32_t num_additional_blocks = num_blocks_needed - num_blocks;

  const auto blocks = block_managers_[dp_rank]->allocate(num_additional_blocks);
  if (blocks.size() != num_additional_blocks) {
    // LOG(ERROR) << " Fail to allocate " << num_additional_blocks << "
    // blocks.";
    return false;
  }

  sequence->add_kv_blocks(blocks);

  return true;
}

bool BlockManagerPool::allocate(Sequence* sequence,
                                size_t num_tokens,
                                size_t needed_copy_in_blocks_num) {
  LOG(FATAL)
      << "allocate(Sequence* sequence, size_t num_tokens, size_t "
         "needed_copy_in_blocks_num) is not implemented in BlockManagerPool.";
  return false;
}

std::vector<Block> BlockManagerPool::allocate(size_t num_tokens,
                                              int32_t& dp_rank) {
  dp_rank = get_manager_with_max_free_blocks();
  const size_t block_size = options_.block_size();
  const size_t num_blocks_needed = (num_tokens + block_size - 1) / block_size;
  return block_managers_[dp_rank]->allocate(num_blocks_needed);
}

bool BlockManagerPool::try_allocate(Sequence* sequence) {
  DCHECK(sequence != nullptr);
  const size_t cacheable_num_tokens =
      get_policy_cacheable_tokens(sequence, sequence->tokens().size());
  int32_t dp_rank = get_dp_rank(sequence);
  const bool needs_embedding_id = !sequence->has_embedding_id();
  if (needs_embedding_id && !allocate_embedding_id(sequence, dp_rank)) {
    return false;
  }

  std::vector<Block> shared_blocks;
  size_t shared_num = 0;
  if (options_.enable_prefix_cache() && cacheable_num_tokens > 0) {
    const auto& existed_shared_blocks = sequence->kv_state().kv_blocks().slice(
        0, sequence->kv_state().shared_kv_blocks_num());
    // If the sequence holds shared_blocks, the hash values of these blocks do
    // not need to be recalculated and can be reused directly.
    const size_t match_tokens =
        std::min(cacheable_num_tokens, sequence->num_tokens());
    auto token_slice = sequence->tokens().slice(0, match_tokens);
    shared_blocks = block_managers_[dp_rank]->allocate_shared(
        token_slice, existed_shared_blocks);

    if (!shared_blocks.empty()) {
      sequence->add_shared_kv_blocks(std::move(shared_blocks));
      shared_num = sequence->kv_state().shared_kv_blocks_num();
    }
  }

  const size_t block_size = options_.block_size();
  const size_t shared_tokens =
      std::min(cacheable_num_tokens, shared_num * block_size);
  size_t num_tokens = cacheable_num_tokens - shared_tokens;

  const size_t num_blocks_needed = (num_tokens + block_size - 1) / block_size;
  if (num_blocks_needed > 0) {
    const auto blocks = block_managers_[dp_rank]->allocate(num_blocks_needed);
    if (blocks.size() != num_blocks_needed) {
      if (sequence->kv_state().num_kv_blocks() != 0) {
        block_managers_[dp_rank]->deallocate(sequence->kv_state().kv_blocks());
        sequence->reset();
      }
      if (needs_embedding_id) {
        deallocate_embedding_id(sequence, dp_rank);
      }
      return false;
    }

    sequence->add_kv_blocks(std::move(blocks));
  }

  const size_t cached_tokens = sequence->kv_state().kv_cache_tokens_num();
  CHECK_GE(cacheable_num_tokens, cached_tokens);
  sequence->kv_state().incr_kv_cache_tokens_num(cacheable_num_tokens -
                                                cached_tokens);
  return true;
}

bool BlockManagerPool::process_beam_search(Sequence* sequence, bool need_swap) {
  if (!sequence->check_beam_search()) {
    return true;
  }

  auto src_blocks = sequence->kv_state().src_blocks();
  if (src_blocks.size() == 0) {
    return true;
  }

  // when sequence need to swap the last block and no new block appended,
  // allocate a new block for this sequence
  if (need_swap && sequence->kv_state().need_swap()) {
    int32_t dp_rank = get_dp_rank(sequence);
    auto new_blocks = block_managers_[dp_rank]->allocate(1);
    if (new_blocks.size() == 0) {
      return false;
    }
    swap_block_transfer_infos_[dp_rank].emplace_back(src_blocks.back().id(),
                                                     new_blocks[0].id());
    sequence->kv_state().process_beam_search(new_blocks[0]);
  } else {
    sequence->kv_state().process_beam_search(std::nullopt);
  }
  return true;
}

void BlockManagerPool::allocate_shared(Sequence* sequence) {
  allocate_shared(sequence,
                  get_policy_cacheable_tokens(sequence, sequence->num_tokens()));
}

void BlockManagerPool::allocate_shared(Sequence* sequence, size_t num_tokens) {
  // only allocate shared blocks for prefill sequences
  if (options_.enable_prefix_cache() && num_tokens > 0) {
    int32_t dp_rank = get_dp_rank(sequence);
    const auto& existed_shared_blocks = sequence->kv_state().kv_blocks().slice(
        0, sequence->kv_state().shared_kv_blocks_num());
    // If the sequence holds shared_blocks, the hash values of these blocks do
    // not need to be recalculated and can be reused directly.
    const size_t match_tokens = std::min(num_tokens, sequence->num_tokens());
    auto token_slice = sequence->tokens().slice(0, match_tokens);
    std::vector<Block> shared_blocks =
        block_managers_[dp_rank]->allocate_shared(token_slice,
                                                  existed_shared_blocks);
    sequence->add_shared_kv_blocks(std::move(shared_blocks));
  }
}

void BlockManagerPool::cache(Sequence* sequence) {
  int32_t dp_rank = get_dp_rank(sequence);
  const auto token_ids = sequence->cached_tokens();
  auto* blocks = sequence->kv_state().mutable_kv_blocks();
  if (token_ids.empty() || blocks->empty()) {
    return;
  }
  auto* block_manager = block_managers_[dp_rank].get();
  const MTGRRuntimeAdmissionDecision admission =
      get_mtgr_runtime_admission_decision(sequence, block_manager,
                                          token_ids.size());
  if (admission.applies) {
    COUNTER_INC(mtgr_dynamic_admission_requests_total);
    const double free_ratio =
        admission.total_blocks == 0
            ? 0.0
            : static_cast<double>(admission.free_blocks) /
                  static_cast<double>(admission.total_blocks);
    MTGR_TRACE(1) << "[PREFIX_CACHE_ADMISSION] policy="
                  << current_mtgr_cache_policy_name()
                  << " request_len=" << admission.request_len
                  << " free_blocks=" << admission.free_blocks
                  << " total_blocks=" << admission.total_blocks
                  << " free_ratio=" << free_ratio
                  << " threshold=" << admission.threshold
                  << " admitted=" << admission.admitted;
    if (!admission.admitted) {
      COUNTER_INC(mtgr_dynamic_admission_skipped_requests_total);
      COUNTER_ADD(mtgr_dynamic_admission_skipped_tokens_total,
                  admission.request_len);
      return;
    }
    COUNTER_INC(mtgr_dynamic_admission_admitted_requests_total);
    COUNTER_ADD(mtgr_dynamic_admission_admitted_tokens_total,
                admission.request_len);
  }
  const MTGRReuseHorizonDecision reuse_horizon_admission =
      get_mtgr_reuse_horizon_admission_decision(sequence, block_manager,
                                                token_ids);
  if (reuse_horizon_admission.applies) {
    COUNTER_INC(mtgr_rha_admission_requests_total);
    if (reuse_horizon_admission.pressure) {
      COUNTER_INC(mtgr_rha_admission_pressure_requests_total);
    }
    if (reuse_horizon_admission.reuse_evidence) {
      COUNTER_INC(mtgr_rha_admission_reuse_evidence_requests_total);
    }
    if (reuse_horizon_admission.has_entity_id) {
      COUNTER_INC(mtgr_rha_admission_entity_id_requests_total);
    }
    if (reuse_horizon_admission.entity_hot_skip_candidate) {
      COUNTER_INC(mtgr_rha_admission_entity_hot_skip_candidates_total);
    }
    MTGR_TRACE(1) << "[RHA_WRITEBACK] policy="
                  << current_mtgr_cache_policy_name()
                  << " request_len=" << reuse_horizon_admission.request_len
                  << " matched_blocks="
                  << reuse_horizon_admission.matched_blocks
                  << " free_blocks=" << reuse_horizon_admission.free_blocks
                  << " total_blocks=" << reuse_horizon_admission.total_blocks
                  << " free_ratio=" << reuse_horizon_admission.free_ratio
                  << " pressure=" << reuse_horizon_admission.pressure
                  << " has_entity_id="
                  << reuse_horizon_admission.has_entity_id
                  << " entity_id=" << reuse_horizon_admission.entity_id
                  << " family_key=" << reuse_horizon_admission.family_key
                  << " prior_seen="
                  << reuse_horizon_admission.prior_seen_count
                  << " prior_hits=" << reuse_horizon_admission.prior_hit_count
                  << " reuse_evidence="
                  << reuse_horizon_admission.reuse_evidence
                  << " cold_low_reuse="
                  << reuse_horizon_admission.cold_low_reuse
                  << " entity_hot_skip_candidate="
                  << reuse_horizon_admission.entity_hot_skip_candidate
                  << " admitted=" << reuse_horizon_admission.admitted
                  << " reason=" << reuse_horizon_admission.reason;
    if (!reuse_horizon_admission.admitted) {
      COUNTER_INC(mtgr_rha_admission_skipped_requests_total);
      COUNTER_ADD(mtgr_rha_admission_skipped_tokens_total,
                  reuse_horizon_admission.request_len);
      return;
    }
    COUNTER_INC(mtgr_rha_admission_admitted_requests_total);
    COUNTER_ADD(mtgr_rha_admission_admitted_tokens_total,
                reuse_horizon_admission.request_len);
  }
  auto existed_shared_blocks_num = sequence->kv_state().shared_kv_blocks_num();
  block_managers_[dp_rank]->cache(
      token_ids, *blocks, existed_shared_blocks_num);
}

void BlockManagerPool::get_merged_kvcache_event(KvCacheEvent* event) const {
  for (int32_t i = 0; i < block_managers_.size(); ++i) {
    block_managers_[i]->get_merged_kvcache_event(event);
  }
}

float BlockManagerPool::get_gpu_cache_usage_perc() const {
  float perc = 0.0;
  for (int32_t i = 0; i < block_managers_.size(); ++i) {
    perc += block_managers_[i]->kv_cache_utilization();
  }
  return perc / block_managers_.size();
}

uint32_t BlockManagerPool::num_blocks() const { return options_.num_blocks(); }

int32_t BlockManagerPool::block_size() const { return options_.block_size(); }

std::vector<size_t> BlockManagerPool::num_blocks_in_prefix_cache() const {
  std::vector<size_t> num_blocks_in_prefix_cache(block_managers_.size());
  for (size_t dp_rank = 0; dp_rank < block_managers_.size(); ++dp_rank) {
    num_blocks_in_prefix_cache[dp_rank] =
        block_managers_[dp_rank]->num_blocks_in_prefix_cache();
  }
  return num_blocks_in_prefix_cache;
}

std::vector<size_t> BlockManagerPool::num_free_blocks() const {
  std::vector<size_t> num_free_blocks(block_managers_.size());
  for (size_t dp_rank = 0; dp_rank < block_managers_.size(); ++dp_rank) {
    num_free_blocks[dp_rank] = block_managers_[dp_rank]->num_free_blocks();
  }
  return num_free_blocks;
}

std::vector<size_t> BlockManagerPool::num_used_blocks() const {
  std::vector<size_t> num_used_blocks(block_managers_.size());
  for (size_t dp_rank = 0; dp_rank < block_managers_.size(); ++dp_rank) {
    num_used_blocks[dp_rank] = block_managers_[dp_rank]->num_used_blocks();
  }
  return num_used_blocks;
}

double BlockManagerPool::kv_cache_utilization() const {
  int32_t dp_rank = get_manager_with_max_free_blocks();
  return block_managers_[dp_rank]->kv_cache_utilization();
}

// currently use only for profile, which not need prefix cache.
// If more often used in the future, can be integrated into deallocate function.
void BlockManagerPool::deallocate_without_cache(Sequence* sequence) {
  DCHECK(sequence != nullptr);
  int32_t dp_rank = get_dp_rank(sequence);
  block_managers_[dp_rank]->deallocate(sequence->kv_state().kv_blocks());
  deallocate_embedding_id(sequence, dp_rank);
  sequence->reset();
}

void BlockManagerPool::reserve_xtensor_padding_blocks() {
  if (!options_.enable_xtensor()) {
    return;
  }

  // Reserve padding block on each XTensorBlockManagerImpl.
  for (auto& manager : block_managers_) {
    auto* xtensor_manager =
        dynamic_cast<XTensorBlockManagerImpl*>(manager.get());
    if (xtensor_manager) {
      xtensor_manager->reserve_xtensor_padding_blocks();
    }
  }

  // Start prealloc thread once (PageAllocator is shared by all managers)
  PageAllocator::get_instance().start_prealloc_thread();
}

}  // namespace xllm
