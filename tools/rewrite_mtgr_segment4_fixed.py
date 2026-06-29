#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

VOCAB_SIZE = 151936
TOKEN_LOW = 1000
TOKEN_HIGH = 140000
TOKEN_SPAN = TOKEN_HIGH - TOKEN_LOW + 1
MASK64 = (1 << 64) - 1
DEFAULT_SEED = 20260602
DEFAULT_SOURCE = Path(
    "/export/home/zhangshen/datas/mtgr_5000_user10x_short_heavy_jsonl_20260605/"
    "requests_access24_growth_5000_user10x_short_heavy_seg4_time_ordered_seed20260605.jsonl"
)
DEFAULT_OUT_DIR = Path("/tmp/mtgr_5000_seg4_fixed_len_variants_20260609")
DEFAULT_FIXED_TARGET_LENGTHS = [800, 1600, 2400]
ID_LEN_PATTERN = re.compile(r"_len\d{5}$")


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


def token_from(base: int, pos: int) -> int:
    return TOKEN_LOW + (splitmix64(base + pos) % TOKEN_SPAN)


def entity_int(entity: str) -> int:
    try:
        return int(entity)
    except ValueError:
        return int.from_bytes(
            hashlib.blake2b(entity.encode("utf-8"), digest_size=8).digest(),
            "little",
        )


def namespace_base(namespace: int, entity: str, extra: int = 0,
                   seed: int = DEFAULT_SEED) -> int:
    return (
        seed
        ^ (namespace * 0xD1B54A32D192ED03)
        ^ (entity_int(entity) * 0x94D049BB133111EB)
        ^ (extra * 0x9E3779B97F4A7C15)
    ) & MASK64


def make_sequence(base: int, length: int) -> list[int]:
    return [token_from(base, pos) for pos in range(length)]


def percentile(values: list[int], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * q
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stats(values: list[int]) -> dict[str, float]:
    return {
        "min": min(values),
        "avg": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def total_len_bucket(total_len: int) -> str:
    if total_len < 2000:
        return "<2000"
    if total_len < 2500:
        return "2000-2500"
    if total_len < 3000:
        return "2500-3000"
    if total_len < 3500:
        return "3000-3500"
    if total_len < 4000:
        return "3500-4000"
    if total_len < 4500:
        return "4000-4500"
    if total_len < 5000:
        return "4500-5000"
    return ">=5000"


def rewrite_id(old_id: str, new_total_len: int, fixed_target_len: int) -> str:
    replacement = f"_len{new_total_len:05d}_seg4f{fixed_target_len:04d}"
    if ID_LEN_PATTERN.search(old_id):
        return ID_LEN_PATTERN.sub(replacement, old_id)
    return f"{old_id}{replacement}"


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as src:
        for line_no, line in enumerate(src, start=1):
            obj = json.loads(line)
            offsets = obj["segment_offsets"]
            token_ids = obj["token_ids"]
            if len(offsets) != 5:
                raise ValueError(f"line {line_no}: expected 5 segment offsets")
            if len(token_ids) != offsets[-1]:
                raise ValueError(
                    f"line {line_no}: token_ids len {len(token_ids)} != offsets[-1] {offsets[-1]}"
                )
            rows.append(obj)
    return rows


def build_variant(rows: list[dict], fixed_target_len: int, *, out_dir: Path,
                  source: Path, seed: int) -> dict:
    out_path = out_dir / f"{source.stem}_seg4fixed{fixed_target_len}.jsonl"
    target_lengths = []
    total_lengths = []
    regenerated_prefix_mismatch_count = 0
    token_min = VOCAB_SIZE
    token_max = -1

    with out_path.open("w", encoding="utf-8") as out:
        for obj in rows:
            token_ids = obj["token_ids"]
            offsets = obj["segment_offsets"]
            prefix_end = offsets[3]
            old_total_len = offsets[4]
            old_target = token_ids[prefix_end:old_total_len]
            old_target_len = len(old_target)

            meta = dict(obj["meta"])
            entity = str(meta["entity_id"])
            original_idx = int(meta["original_idx"])
            base = namespace_base(4, entity, original_idx, seed=seed)
            generated_target = make_sequence(base, fixed_target_len)

            compare_len = min(old_target_len, fixed_target_len)
            if generated_target[:compare_len] != old_target[:compare_len]:
                regenerated_prefix_mismatch_count += 1
                if fixed_target_len <= old_target_len:
                    new_target = old_target[:fixed_target_len]
                else:
                    new_target = old_target + generated_target[old_target_len:]
            else:
                new_target = generated_target

            new_token_ids = token_ids[:prefix_end] + new_target
            new_total_len = len(new_token_ids)
            new_offsets = offsets[:4] + [new_total_len]

            new_meta = dict(meta)
            new_meta.update(
                {
                    "target_len": fixed_target_len,
                    "segment4_len": fixed_target_len,
                    "original_target_len_before_fixed_seg4": old_target_len,
                    "original_total_len_before_fixed_seg4": old_total_len,
                    "fixed_segment4_len": fixed_target_len,
                    "fixed_segment4_source_path": str(source),
                    "fixed_segment4_generation": (
                        "truncate_or_deterministic_extend_using_namespace_base_4"
                    ),
                    "fixed_segment4_seed": seed,
                    "total_len_bucket": total_len_bucket(new_total_len),
                }
            )

            row = {
                "id": rewrite_id(str(obj["id"]), new_total_len, fixed_target_len),
                "token_ids": new_token_ids,
                "segment_offsets": new_offsets,
                "segment_rules": list(obj["segment_rules"]),
                "meta": new_meta,
            }
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

            target_lengths.append(fixed_target_len)
            total_lengths.append(new_total_len)
            token_min = min(token_min, min(new_token_ids))
            token_max = max(token_max, max(new_token_ids))

    line_count = len(rows)
    sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()
    return {
        "output_path": str(out_path),
        "line_count": line_count,
        "sha256": sha256,
        "fixed_segment4_len": fixed_target_len,
        "segment4_len_stats": stats(target_lengths),
        "total_len_stats": stats(total_lengths),
        "regenerated_prefix_mismatch_count": regenerated_prefix_mismatch_count,
        "all_token_ids_within_vocab": 0 <= token_min and token_max < VOCAB_SIZE,
        "all_target_lens_equal_fixed_value": len(set(target_lengths)) == 1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite a 4-segment MTGR JSONL dataset so segment-4 length "
        "is fixed to one or more target lengths.")
    parser.add_argument("--source",
                        type=Path,
                        default=DEFAULT_SOURCE,
                        help="Source JSONL dataset path.")
    parser.add_argument("--out-dir",
                        type=Path,
                        default=DEFAULT_OUT_DIR,
                        help="Output directory for rewritten datasets and summary.")
    parser.add_argument(
        "--fixed-target-lengths",
        default=",".join(str(v) for v in DEFAULT_FIXED_TARGET_LENGTHS),
        help="Comma-separated segment-4 target lengths, for example 800,1600,2400.",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=DEFAULT_SEED,
                        help="Seed used when deterministic extension is needed.")
    return parser.parse_args()


def parse_fixed_lengths(raw: str) -> list[int]:
    values = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        value = int(piece)
        if value <= 0:
            raise ValueError(f"fixed target length must be positive: {value}")
        values.append(value)
    if not values:
        raise ValueError("--fixed-target-lengths must contain at least one value")
    return values


def main() -> int:
    args = parse_args()
    fixed_target_lengths = parse_fixed_lengths(args.fixed_target_lengths)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.source)
    variants = [
        build_variant(rows,
                      fixed_len,
                      out_dir=args.out_dir,
                      source=args.source,
                      seed=args.seed) for fixed_len in fixed_target_lengths
    ]
    summary = {
        "source_path": str(args.source),
        "output_dir": str(args.out_dir),
        "variant_count": len(variants),
        "variants": variants,
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
