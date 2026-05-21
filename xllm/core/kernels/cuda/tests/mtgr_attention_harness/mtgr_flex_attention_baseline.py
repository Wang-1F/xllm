#!/usr/bin/env python3
"""FlexAttention baseline for MTGR attention odd-length benchmarks.

This script is intentionally Python-only. FlexAttention is a PyTorch
programmable-attention API, so keeping it out of the C++ harness avoids mixing
prototype Python compiler paths into the production-style C++ test contract.

Run it under Nsight Systems and use the generated labels CSV to join NVTX
ranges with request metadata.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)


LABEL_FIELDS = [
    "idx",
    "backend",
    "pair_id",
    "mode",
    "heads",
    "kv_heads",
    "head_dim",
    "history",
    "context",
    "realtime",
    "realtime_matched",
    "target",
    "total_q",
    "matched_prefix",
    "live_q",
    "repeat_id",
]


@dataclass(frozen=True)
class MTGRFlexCase:
    pair_id: int
    heads: int
    kv_heads: int
    head_dim: int
    history: int
    context: int
    realtime: int
    target: int
    matched_prefix: int
    block_size: int

    @property
    def total_len(self) -> int:
        return self.history + self.context + self.realtime + self.target

    @property
    def live_len(self) -> int:
        return self.total_len - self.matched_prefix

    @property
    def realtime_matched(self) -> int:
        if self.matched_prefix <= self.history + self.context:
            return 0
        return self.matched_prefix - self.history - self.context

    @property
    def mode_name(self) -> str:
        return "partial_match" if self.matched_prefix > 0 else "no_match"


@contextmanager
def nvtx_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def sample_odd_len(rng: random.Random, lo: int, hi: int) -> int:
    for _ in range(1024):
        value = rng.randint(lo, hi)
        if value % 32 != 0 and value % 64 != 0:
            return value
    value = rng.randint(lo, hi)
    return value - 1 if value == hi else value + 1


def shape_key(case: MTGRFlexCase) -> tuple[int, int, int, int, int]:
    return (
        case.heads,
        case.head_dim,
        case.history,
        case.realtime,
        case.target,
    )


def generate_odd_length_metadata_pairs(
    pair_count: int,
    seed: int,
    block_size: int,
) -> list[MTGRFlexCase]:
    rng = random.Random(seed)
    heads_all = [4, 8, 12]
    head_dims_all = [64, 128]
    cases: list[MTGRFlexCase] = []
    seen: set[tuple[int, int, int, int, int]] = set()

    attempt = 0
    while attempt < pair_count * 64 and len(cases) < pair_count * 2:
        attempt += 1
        pair_id = len(cases) // 2 + 1
        heads = rng.choice(heads_all)
        head_dim = rng.choice(head_dims_all)
        history = sample_odd_len(rng, 1350, 4096)
        context = 8
        realtime = sample_odd_len(rng, 100, 600)
        target = sample_odd_len(rng, 800, 2400)
        partial = MTGRFlexCase(
            pair_id=pair_id,
            heads=heads,
            kv_heads=heads,
            head_dim=head_dim,
            history=history,
            context=context,
            realtime=realtime,
            target=target,
            matched_prefix=history + context + (realtime * 4) // 5,
            block_size=block_size,
        )
        if shape_key(partial) in seen:
            continue
        seen.add(shape_key(partial))
        cases.append(
            MTGRFlexCase(
                pair_id=pair_id,
                heads=heads,
                kv_heads=heads,
                head_dim=head_dim,
                history=history,
                context=context,
                realtime=realtime,
                target=target,
                matched_prefix=0,
                block_size=block_size,
            )
        )
        cases.append(partial)

    if len(cases) != pair_count * 2:
        raise RuntimeError("failed to generate enough unique odd-length cases")
    return cases


def load_cases_from_labels(path: Path, block_size: int) -> list[MTGRFlexCase]:
    cases: list[MTGRFlexCase] = []
    seen: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["pair_id"], row["mode"])
            if key in seen:
                continue
            seen.add(key)
            cases.append(
                MTGRFlexCase(
                    pair_id=int(row["pair_id"]),
                    heads=int(row["heads"]),
                    kv_heads=int(row["kv_heads"]),
                    head_dim=int(row["head_dim"]),
                    history=int(row["history"]),
                    context=int(row["context"]),
                    realtime=int(row["realtime"]),
                    target=int(row["target"]),
                    matched_prefix=int(row["matched_prefix"]),
                    block_size=block_size,
                )
            )
    cases.sort(key=lambda case: (case.pair_id, case.matched_prefix != 0))
    return cases


def make_mtgr_mask_mod(case: MTGRFlexCase):
    history_end = case.history
    context_end = case.history + case.context
    realtime_end = context_end + case.realtime

    def mask_mod(batch, head, q_idx, kv_idx):
        del batch, head
        history_visible = (
            (q_idx < history_end) & (kv_idx < history_end) & (kv_idx <= q_idx)
        )
        context_visible = (
            (q_idx >= history_end) & (q_idx < context_end) & (kv_idx < context_end)
        )
        realtime_visible = (
            (q_idx >= context_end) & (q_idx < realtime_end) & (kv_idx <= q_idx)
        )
        target_visible = (q_idx >= realtime_end) & (
            (kv_idx < realtime_end) | (kv_idx == q_idx)
        )
        return (
            history_visible
            | context_visible
            | realtime_visible
            | target_visible
        )

    return mask_mod


def ranges_overlap(a_begin: int, a_end: int, b_begin: int, b_end: int) -> bool:
    return max(a_begin, b_begin) < min(a_end, b_end)


def mtgr_block_has_visible_token(
    q_begin: int,
    q_end: int,
    k_begin: int,
    k_end: int,
    case: MTGRFlexCase,
) -> bool:
    total_len = case.total_len
    q_end = min(q_end, total_len)
    k_end = min(k_end, total_len)
    if q_begin >= q_end or k_begin >= k_end:
        return False

    history_end = case.history
    context_end = case.history + case.context
    realtime_end = context_end + case.realtime

    hist_q_begin = max(q_begin, 0)
    hist_q_end = min(q_end, history_end)
    if hist_q_begin < hist_q_end and k_begin < history_end and k_begin <= hist_q_end - 1:
        return True

    ctx_q_begin = max(q_begin, history_end)
    ctx_q_end = min(q_end, context_end)
    if ctx_q_begin < ctx_q_end and k_begin < context_end:
        return True

    rt_q_begin = max(q_begin, context_end)
    rt_q_end = min(q_end, realtime_end)
    if rt_q_begin < rt_q_end and k_begin <= rt_q_end - 1:
        return True

    tgt_q_begin = max(q_begin, realtime_end)
    tgt_q_end = min(q_end, total_len)
    if tgt_q_begin < tgt_q_end:
        if k_begin < realtime_end:
            return True
        if ranges_overlap(tgt_q_begin, tgt_q_end, k_begin, k_end):
            return True

    return False


def mtgr_block_is_safely_full(
    q_begin: int,
    q_end: int,
    k_begin: int,
    k_end: int,
    case: MTGRFlexCase,
) -> bool:
    total_len = case.total_len
    if q_end > total_len or k_end > total_len:
        return False

    history_end = case.history
    context_end = case.history + case.context
    realtime_end = context_end + case.realtime

    if q_begin >= 0 and q_end <= history_end:
        return k_end <= q_begin and k_end <= history_end
    if q_begin >= history_end and q_end <= context_end:
        return k_end <= context_end
    if q_begin >= context_end and q_end <= realtime_end:
        return k_end <= q_begin
    if q_begin >= realtime_end and q_end <= total_len:
        return k_end <= realtime_end
    return False


def build_mtgr_direct_block_mask(
    case: MTGRFlexCase,
    device: torch.device,
) -> BlockMask:
    block_size = case.block_size
    total_len = case.total_len
    q_blocks = (total_len + block_size - 1) // block_size
    kv_blocks = q_blocks

    history_end = case.history
    context_end = case.history + case.context
    realtime_end = context_end + case.realtime

    q_begin = (np.arange(q_blocks, dtype=np.int32) * block_size)[:, None]
    q_end = q_begin + block_size
    k_begin = (np.arange(kv_blocks, dtype=np.int32) * block_size)[None, :]
    k_end = k_begin + block_size

    history_q_end = np.minimum(q_end, history_end)
    history_visible = (
        (q_begin < history_end)
        & (k_begin < history_end)
        & (k_begin <= history_q_end - 1)
    )
    context_visible = (
        (q_begin < context_end)
        & (q_end > history_end)
        & (k_begin < context_end)
    )
    realtime_q_end = np.minimum(q_end, realtime_end)
    realtime_visible = (
        (q_begin < realtime_end)
        & (q_end > context_end)
        & (k_begin <= realtime_q_end - 1)
    )
    target_q_begin = np.maximum(q_begin, realtime_end)
    target_q_end = np.minimum(q_end, total_len)
    target_visible = (q_begin < total_len) & (q_end > realtime_end) & (
        (k_begin < realtime_end)
        | (np.maximum(target_q_begin, k_begin) < np.minimum(target_q_end, k_end))
    )
    visible_blocks = (
        history_visible
        | context_visible
        | realtime_visible
        | target_visible
    )

    full_blocks = (
        (
            (q_begin >= 0)
            & (q_end <= history_end)
            & (k_end <= q_begin)
            & (k_end <= history_end)
        )
        | (
            (q_begin >= history_end)
            & (q_end <= context_end)
            & (k_end <= context_end)
        )
        | (
            (q_begin >= context_end)
            & (q_end <= realtime_end)
            & (k_end <= q_begin)
        )
        | ((q_begin >= realtime_end) & (q_end <= total_len) & (k_end <= realtime_end))
    ) & visible_blocks
    partial_blocks = visible_blocks & ~full_blocks

    partial_kv_num_blocks = partial_blocks.sum(axis=1, dtype=np.int32)
    full_kv_num_blocks = full_blocks.sum(axis=1, dtype=np.int32)
    partial_q_num_blocks = partial_blocks.sum(axis=0, dtype=np.int32)
    full_q_num_blocks = full_blocks.sum(axis=0, dtype=np.int32)
    partial_kv_indices = np.zeros((q_blocks, kv_blocks), dtype=np.int32)
    full_kv_indices = np.zeros((q_blocks, kv_blocks), dtype=np.int32)
    partial_q_indices = np.zeros((kv_blocks, q_blocks), dtype=np.int32)
    full_q_indices = np.zeros((kv_blocks, q_blocks), dtype=np.int32)

    for block_idx in range(q_blocks):
        indices = np.flatnonzero(partial_blocks[block_idx]).astype(np.int32)
        partial_kv_indices[block_idx, : indices.size] = indices
        indices = np.flatnonzero(full_blocks[block_idx]).astype(np.int32)
        full_kv_indices[block_idx, : indices.size] = indices
        indices = np.flatnonzero(partial_blocks[:, block_idx]).astype(np.int32)
        partial_q_indices[block_idx, : indices.size] = indices
        indices = np.flatnonzero(full_blocks[:, block_idx]).astype(np.int32)
        full_q_indices[block_idx, : indices.size] = indices

    flat_block_mask = np.concatenate(
        [
            partial_kv_num_blocks.ravel(),
            partial_kv_indices.ravel(),
            full_kv_num_blocks.ravel(),
            full_kv_indices.ravel(),
            partial_q_num_blocks.ravel(),
            partial_q_indices.ravel(),
            full_q_num_blocks.ravel(),
            full_q_indices.ravel(),
        ],
    ).astype(np.int32, copy=False)
    block_mask_tensor = torch.from_numpy(flat_block_mask).to(
        device=device,
        non_blocking=True,
    )

    offset = 0

    def take_view(element_count: int, shape: tuple[int, ...]) -> torch.Tensor:
        nonlocal offset
        out = block_mask_tensor[offset : offset + element_count].view(*shape)
        offset += element_count
        return out

    # The MTGR mask is head-independent, so a single-head BlockMask can be
    # broadcast by FlexAttention instead of duplicating metadata per head.
    kv_num_blocks_tensor = take_view(q_blocks, (1, 1, q_blocks))
    kv_indices_tensor = take_view(q_blocks * kv_blocks, (1, 1, q_blocks, kv_blocks))
    full_kv_num_blocks_tensor = take_view(q_blocks, (1, 1, q_blocks))
    full_kv_indices_tensor = take_view(
        q_blocks * kv_blocks,
        (1, 1, q_blocks, kv_blocks),
    )
    q_num_blocks_tensor = take_view(kv_blocks, (1, 1, kv_blocks))
    q_indices_tensor = take_view(kv_blocks * q_blocks, (1, 1, kv_blocks, q_blocks))
    full_q_num_blocks_tensor = take_view(kv_blocks, (1, 1, kv_blocks))
    full_q_indices_tensor = take_view(
        kv_blocks * q_blocks,
        (1, 1, kv_blocks, q_blocks),
    )

    return BlockMask(
        seq_lengths=(total_len, total_len),
        kv_num_blocks=kv_num_blocks_tensor,
        kv_indices=kv_indices_tensor,
        full_kv_num_blocks=full_kv_num_blocks_tensor,
        full_kv_indices=full_kv_indices_tensor,
        q_num_blocks=q_num_blocks_tensor,
        q_indices=q_indices_tensor,
        full_q_num_blocks=full_q_num_blocks_tensor,
        full_q_indices=full_q_indices_tensor,
        BLOCK_SIZE=(block_size, block_size),
        mask_mod=make_mtgr_mask_mod(case),
    )


def build_flex_block_mask(
    case: MTGRFlexCase,
    device: torch.device,
    mask_builder: str,
) -> BlockMask:
    if mask_builder == "direct":
        return build_mtgr_direct_block_mask(case, device)
    if mask_builder == "dense":
        return create_block_mask(
            make_mtgr_mask_mod(case),
            B=1,
            H=case.heads,
            Q_LEN=case.total_len,
            KV_LEN=case.total_len,
            device=str(device),
            BLOCK_SIZE=case.block_size,
        )
    raise ValueError(f"unknown mask builder: {mask_builder}")


def make_inputs(case: MTGRFlexCase, device: torch.device) -> tuple[torch.Tensor, ...]:
    if case.heads != case.kv_heads:
        raise ValueError("MTGR FlexAttention baseline currently keeps GQA out")
    shape = (1, case.heads, case.total_len, case.head_dim)
    options = {"device": device, "dtype": torch.bfloat16}
    query = torch.randn(shape, **options) * 0.05
    key = torch.randn(shape, **options) * 0.05
    value = torch.randn(shape, **options) * 0.05
    return query, key, value


def run_flex_case(
    case: MTGRFlexCase,
    compiled_flex_attention,
    device: torch.device,
    emit_nvtx: bool,
    mask_builder: str,
) -> torch.Tensor:
    query, key, value = make_inputs(case, device)
    scale = 1.0 / math.sqrt(case.head_dim)

    if emit_nvtx:
        with nvtx_range("MTGR/harness/mtgr_attention/flex/block_mask_build"):
            block_mask = build_flex_block_mask(case, device, mask_builder)
            torch.cuda.synchronize()
        with nvtx_range("MTGR/harness/mtgr_attention/flex/flex_attention"):
            output = compiled_flex_attention(
                query,
                key,
                value,
                block_mask=block_mask,
                scale=scale,
            )
            torch.cuda.synchronize()
        with nvtx_range("MTGR/harness/mtgr_attention/flex/device_sync"):
            torch.cuda.synchronize()
    else:
        block_mask = build_flex_block_mask(case, device, mask_builder)
        output = compiled_flex_attention(
            query,
            key,
            value,
            block_mask=block_mask,
            scale=scale,
        )
        torch.cuda.synchronize()

    return output[:, :, case.matched_prefix :, :].contiguous()


def write_label_row(writer: csv.DictWriter,
                    idx: int,
                    case: MTGRFlexCase,
                    repeat_id: int) -> None:
    writer.writerow(
        {
            "idx": idx,
            "backend": "flex_attention_base",
            "pair_id": case.pair_id,
            "mode": case.mode_name,
            "heads": case.heads,
            "kv_heads": case.kv_heads,
            "head_dim": case.head_dim,
            "history": case.history,
            "context": case.context,
            "realtime": case.realtime,
            "realtime_matched": case.realtime_matched,
            "target": case.target,
            "total_q": case.total_len,
            "matched_prefix": case.matched_prefix,
            "live_q": case.live_len,
            "repeat_id": repeat_id,
        }
    )


def run_warmup(cases: Iterable[MTGRFlexCase],
               compiled_flex_attention,
               device: torch.device,
               warmup: int,
               mask_builder: str) -> None:
    for _ in range(warmup):
        for case in cases:
            run_flex_case(
                case,
                compiled_flex_attention,
                device,
                emit_nvtx=False,
                mask_builder=mask_builder,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("/tmp/mtgr_flex_attention_baseline_labels.csv"),
    )
    parser.add_argument(
        "--input-labels",
        type=Path,
        help="Reuse shape metadata from an existing MTGR harness labels CSV.",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Call flex_attention directly instead of torch.compile(flex_attention).",
    )
    parser.add_argument(
        "--compile-dynamic",
        action="store_true",
        help="Use torch.compile(flex_attention, dynamic=True) for variable lengths.",
    )
    parser.add_argument(
        "--mask-builder",
        choices=("direct", "dense"),
        default="direct",
        help=(
            "Use direct MTGR BlockMask construction by default; 'dense' keeps "
            "PyTorch create_block_mask for comparison."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FlexAttention baseline")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.input_labels:
        cases = load_cases_from_labels(args.input_labels, args.block_size)
    else:
        cases = generate_odd_length_metadata_pairs(
            args.pairs,
            args.seed,
            args.block_size,
        )

    if args.no_compile:
        compiled_flex_attention = flex_attention
    else:
        compiled_flex_attention = torch.compile(
            flex_attention,
            dynamic=args.compile_dynamic,
        )

    run_warmup(
        cases,
        compiled_flex_attention,
        device,
        max(0, args.warmup),
        args.mask_builder,
    )

    args.labels.parent.mkdir(parents=True, exist_ok=True)
    with args.labels.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        idx = 0
        for case in cases:
            for repeat_id in range(1, max(1, args.repeat) + 1):
                idx += 1
                write_label_row(writer, idx, case, repeat_id)
                f.flush()
                root = f"MTGR/harness/mtgr_attention/flex/{case.mode_name}"
                with nvtx_range(root):
                    output = run_flex_case(
                        case,
                        compiled_flex_attention,
                        device,
                        emit_nvtx=True,
                        mask_builder=args.mask_builder,
                    )
                if output.size(2) != case.live_len:
                    raise RuntimeError("unexpected live output length")
            if case.mode_name == "partial_match" and case.pair_id % 50 == 0:
                print(
                    f"[MTGR][FlexAttention] completed_pairs="
                    f"{case.pair_id}/{args.pairs} labels={idx}",
                    flush=True,
                )

    print(
        f"[MTGR][FlexAttention] done cases={len(cases)} labels={idx} "
        f"labels_path={args.labels}",
        flush=True,
    )


if __name__ == "__main__":
    main()
