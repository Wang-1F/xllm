#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


XLLM_BATCH_RE = re.compile(
    r"\[BATCH\] seq_idx=(?P<seq_idx>\d+).*?"
    r"total_input_len=(?P<total_input_len>\d+)\s+"
    r"cacheable_len=(?P<cacheable_len>\d+)\s+"
    r"(?:writeback_len=(?P<writeback_len>\d+)\s+)?"
    r"(?:cache_admitted=(?P<cache_admitted>\S+)\s+)?"
    r"(?:high_value=(?P<high_value>\S+)\s+)?"
    r"cache_policy=(?P<cache_policy>\S+)\s+"
    r"prefix_match_limit=(?P<prefix_match_limit>\d+)\s+"
    r"matched_prefix=(?P<matched_prefix>\d+)\s+"
    r"local_q_len=(?P<local_q_len>\d+)"
)
VLLM_FORWARD_RE = re.compile(
    r"MTGR prefix-cache forward req_id=(?P<req_id>\S+)\s+"
    r"prompt_len=(?P<prompt_len>\d+)\s+"
    r"(?:(?:cacheable_len=(?P<cacheable_len>\d+)\s+)?"
    r"(?:prefix_match_limit=(?P<prefix_match_limit>\d+)\s+)?)?"
    r"matched_prefix=(?P<matched_prefix>\d+)\s+"
    r"live_prompt_tokens=(?P<live_prompt_tokens>\d+)\s+"
    r"cache_policy=(?P<cache_policy>\S+)"
)
VLLM_NATIVE_RATE_RE = re.compile(r"Prefix cache hit rate: (?P<rate>[0-9.]+)%")
EVICT_TRACE_RE = re.compile(
    r"\[PREFIX_CACHE\].*?evict.*?"
    r"requested_blocks=(?P<requested_blocks>\d+).*?"
    r"evicted_blocks=(?P<evicted_blocks>\d+)"
)
XLLM_TS_RE = re.compile(
    r"^[IWEF](?P<date>\d{8})\s+"
    r"(?P<time>\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
)
VLLM_TS_RE = re.compile(
    r"\b(?P<month>\d{2})-(?P<day>\d{2})\s+"
    r"(?P<time>\d{2}:\d{2}:\d{2})\b"
)


@dataclass(frozen=True)
class PrefixEvent:
    source: str
    timestamp_epoch: float | None
    cache_policy: str
    total_input_len: int
    cacheable_len: int
    writeback_len: int
    cache_admitted: bool | None
    high_value: bool | None
    prefix_match_limit: int
    matched_prefix: int
    local_q_len: int


@dataclass(frozen=True)
class EvictEvent:
    source: str
    timestamp_epoch: float | None
    requested_blocks: int
    evicted_blocks: int


@dataclass(frozen=True)
class NativeRateSample:
    source: str
    timestamp_epoch: float | None
    hit_rate_percent: float


@dataclass(frozen=True)
class Window:
    qps: float
    start_epoch: float
    end_epoch: float


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return float("nan")
    return numerator / denominator


def parse_optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    return None


def parse_timestamp(line: str, *, year: int) -> float | None:
    xllm_match = XLLM_TS_RE.search(line)
    if xllm_match:
        text = f"{xllm_match.group('date')} {xllm_match.group('time')}"
        fmt = "%Y%m%d %H:%M:%S.%f" if "." in text else "%Y%m%d %H:%M:%S"
        return datetime.strptime(text, fmt).timestamp()

    vllm_match = VLLM_TS_RE.search(line)
    if vllm_match:
        text = (
            f"{year}-{vllm_match.group('month')}-{vllm_match.group('day')} "
            f"{vllm_match.group('time')}"
        )
        return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").timestamp()

    return None


def parse_logs(
    paths: list[Path],
    *,
    source: str,
    year: int,
) -> tuple[list[PrefixEvent], list[EvictEvent], list[NativeRateSample]]:
    events: list[PrefixEvent] = []
    evictions: list[EvictEvent] = []
    native_rates: list[NativeRateSample] = []

    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                ts = parse_timestamp(line, year=year)
                xllm_match = XLLM_BATCH_RE.search(line)
                if xllm_match and source in ("auto", "xllm"):
                    events.append(
                        PrefixEvent(
                            source="xllm",
                            timestamp_epoch=ts,
                            cache_policy=xllm_match.group("cache_policy"),
                            total_input_len=int(xllm_match.group("total_input_len")),
                            cacheable_len=int(xllm_match.group("cacheable_len")),
                            writeback_len=(
                                int(xllm_match.group("writeback_len"))
                                if xllm_match.group("writeback_len") is not None
                                else int(xllm_match.group("cacheable_len"))
                            ),
                            cache_admitted=parse_optional_bool(
                                xllm_match.group("cache_admitted")
                            ),
                            high_value=parse_optional_bool(
                                xllm_match.group("high_value")
                            ),
                            prefix_match_limit=int(
                                xllm_match.group("prefix_match_limit")
                            ),
                            matched_prefix=int(xllm_match.group("matched_prefix")),
                            local_q_len=int(xllm_match.group("local_q_len")),
                        )
                    )
                    continue

                vllm_match = VLLM_FORWARD_RE.search(line)
                if vllm_match and source in ("auto", "vllm"):
                    prompt_len = int(vllm_match.group("prompt_len"))
                    cacheable_len = (
                        int(vllm_match.group("cacheable_len"))
                        if vllm_match.group("cacheable_len") is not None
                        else prompt_len
                    )
                    prefix_match_limit = (
                        int(vllm_match.group("prefix_match_limit"))
                        if vllm_match.group("prefix_match_limit") is not None
                        else max(0, prompt_len - 1)
                    )
                    events.append(
                        PrefixEvent(
                            source="vllm",
                            timestamp_epoch=ts,
                            cache_policy=vllm_match.group("cache_policy"),
                            total_input_len=prompt_len,
                            cacheable_len=cacheable_len,
                            writeback_len=cacheable_len,
                            cache_admitted=None,
                            high_value=None,
                            prefix_match_limit=prefix_match_limit,
                            matched_prefix=int(vllm_match.group("matched_prefix")),
                            local_q_len=int(vllm_match.group("live_prompt_tokens")),
                        )
                    )
                    continue

                native_rate_match = VLLM_NATIVE_RATE_RE.search(line)
                if native_rate_match and source in ("auto", "vllm"):
                    native_rates.append(
                        NativeRateSample(
                            source="vllm",
                            timestamp_epoch=ts,
                            hit_rate_percent=float(native_rate_match.group("rate")),
                        )
                    )
                    continue

                evict_match = EVICT_TRACE_RE.search(line)
                if evict_match:
                    evictions.append(
                        EvictEvent(
                            source=source if source != "auto" else "xllm",
                            timestamp_epoch=ts,
                            requested_blocks=int(evict_match.group("requested_blocks")),
                            evicted_blocks=int(evict_match.group("evicted_blocks")),
                        )
                    )

    return events, evictions, native_rates


def summarize(
    *,
    scope: str,
    qps: float | None,
    events: list[PrefixEvent],
    evictions: list[EvictEvent],
    native_rates: list[NativeRateSample],
    block_size: int,
    value_prefix_len_threshold: int | None,
) -> dict[str, Any]:
    matched = [event.matched_prefix for event in events]
    total_input_tokens = sum(event.total_input_len for event in events)
    cacheable_tokens = sum(event.cacheable_len for event in events)
    writeback_tokens = sum(event.writeback_len for event in events)
    estimated_cache_blocks = sum(
        math.ceil(event.cacheable_len / block_size) for event in events
    )
    estimated_full_sequence_blocks = sum(
        math.ceil(event.total_input_len / block_size) for event in events
    )
    prefix_match_limit_tokens = sum(event.prefix_match_limit for event in events)
    matched_tokens = sum(event.matched_prefix for event in events)
    matched_tokens_capped_to_limit = sum(
        min(event.matched_prefix, event.prefix_match_limit) for event in events
    )
    matched_tokens_capped_to_cacheable = sum(
        min(event.matched_prefix, event.cacheable_len) for event in events
    )
    local_q_tokens = sum(event.local_q_len for event in events)
    policy_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for event in events:
        policy_counts[event.cache_policy] = policy_counts.get(event.cache_policy, 0) + 1
        source_counts[event.source] = source_counts.get(event.source, 0) + 1

    def event_is_high_value(event: PrefixEvent) -> bool:
        if event.high_value is not None:
            return event.high_value
        if value_prefix_len_threshold is not None:
            return event.prefix_match_limit >= value_prefix_len_threshold
        return True

    def event_is_admitted(event: PrefixEvent) -> bool:
        if event.cache_admitted is not None:
            return event.cache_admitted
        return event.cacheable_len > 0

    high_value_events = [event for event in events if event_is_high_value(event)]
    low_value_events = [event for event in events if not event_is_high_value(event)]
    admitted_events = [event for event in events if event_is_admitted(event)]

    def request_hit_rate(group: list[PrefixEvent]) -> float:
        return ratio(sum(1 for event in group if event.matched_prefix > 0), len(group))

    def token_match_rate_vs_limit(group: list[PrefixEvent]) -> float:
        matched_group_tokens = sum(
            min(event.matched_prefix, event.prefix_match_limit) for event in group
        )
        limit_group_tokens = sum(event.prefix_match_limit for event in group)
        return ratio(matched_group_tokens, limit_group_tokens)

    def token_match_rate_vs_cacheable(group: list[PrefixEvent]) -> float:
        matched_group_tokens = sum(
            min(event.matched_prefix, event.cacheable_len) for event in group
        )
        cacheable_group_tokens = sum(event.cacheable_len for event in group)
        return ratio(matched_group_tokens, cacheable_group_tokens)

    hit_events = sum(1 for event in events if event.matched_prefix > 0)
    full_hit_events = sum(
        1
        for event in events
        if event.prefix_match_limit > 0
        and event.matched_prefix >= event.prefix_match_limit
    )
    evict_calls = len(evictions)
    evicted_blocks = sum(event.evicted_blocks for event in evictions)
    requested_evict_blocks = sum(event.requested_blocks for event in evictions)
    evict_call_rate = ratio(evict_calls, len(events))
    requested_evict_blocks_per_request = ratio(requested_evict_blocks, len(events))
    evicted_blocks_per_request = ratio(evicted_blocks, len(events))
    evict_satisfaction_rate = ratio(evicted_blocks, requested_evict_blocks)

    native_last = native_rates[-1].hit_rate_percent if native_rates else float("nan")
    return {
        "scope": scope,
        "qps": qps,
        "sources": source_counts,
        "policies": policy_counts,
        "value_prefix_len_threshold": (
            "" if value_prefix_len_threshold is None else value_prefix_len_threshold
        ),
        "events": len(events),
        "cache_admitted_requests": len(admitted_events),
        "cache_skipped_requests": len(events) - len(admitted_events),
        "cache_admitted_ratio": ratio(len(admitted_events), len(events)),
        "high_value_requests": len(high_value_events),
        "low_value_requests": len(low_value_events),
        "request_hit_count": hit_events,
        "request_hit_rate": ratio(hit_events, len(events)),
        "high_value_request_hit_rate": request_hit_rate(high_value_events),
        "low_value_request_hit_rate": request_hit_rate(low_value_events),
        "request_full_hit_count": full_hit_events,
        "request_full_hit_rate": ratio(full_hit_events, len(events)),
        "matched_tokens": matched_tokens,
        "matched_tokens_capped_to_limit": matched_tokens_capped_to_limit,
        "matched_tokens_capped_to_cacheable": matched_tokens_capped_to_cacheable,
        "prefix_match_limit_tokens": prefix_match_limit_tokens,
        "token_match_rate_vs_limit": ratio(
            matched_tokens_capped_to_limit,
            prefix_match_limit_tokens,
        ),
        "cacheable_tokens": cacheable_tokens,
        "writeback_tokens": writeback_tokens,
        "token_match_rate_vs_cacheable": ratio(
            matched_tokens_capped_to_cacheable,
            cacheable_tokens,
        ),
        "high_value_token_match_rate_vs_limit": token_match_rate_vs_limit(
            high_value_events
        ),
        "low_value_token_match_rate_vs_limit": token_match_rate_vs_limit(
            low_value_events
        ),
        "high_value_token_match_rate_vs_cacheable": token_match_rate_vs_cacheable(
            high_value_events
        ),
        "low_value_token_match_rate_vs_cacheable": token_match_rate_vs_cacheable(
            low_value_events
        ),
        "total_input_tokens": total_input_tokens,
        "cacheable_token_ratio": ratio(cacheable_tokens, total_input_tokens),
        "saved_cacheable_tokens_vs_full_sequence": total_input_tokens
        - cacheable_tokens,
        "estimated_cache_blocks": estimated_cache_blocks,
        "estimated_cache_blocks_per_request": ratio(estimated_cache_blocks, len(events)),
        "saved_cache_blocks_estimate_vs_full_sequence": (
            estimated_full_sequence_blocks - estimated_cache_blocks
        ),
        "writeback_tokens_per_request": ratio(writeback_tokens, len(events)),
        "local_q_tokens": local_q_tokens,
        "avg_matched_prefix": ratio(matched_tokens, len(events)),
        "p50_matched_prefix": percentile([float(value) for value in matched], 50),
        "p90_matched_prefix": percentile([float(value) for value in matched], 90),
        "p95_matched_prefix": percentile([float(value) for value in matched], 95),
        "p99_matched_prefix": percentile([float(value) for value in matched], 99),
        "max_matched_prefix": max(matched) if matched else 0,
        "evict_calls": evict_calls,
        "evict_call_rate": evict_call_rate,
        "requested_evict_blocks": requested_evict_blocks,
        "requested_evict_blocks_per_request": requested_evict_blocks_per_request,
        "evicted_blocks": evicted_blocks,
        "evicted_blocks_per_request": evicted_blocks_per_request,
        "evict_satisfaction_rate": evict_satisfaction_rate,
        "native_reported_hit_rate_last": native_last,
    }


def load_windows(path: Path, *, phase: str, padding_s: float) -> list[Window]:
    windows: list[Window] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "wall_start_epoch" not in row or "wall_end_epoch" not in row:
                raise ValueError(
                    f"{path} does not contain wall-clock columns. Rerun "
                    "mtgr_qps_sweep.py after the wall-clock summary patch."
                )
            start_key = (
                "wall_measure_start_epoch"
                if phase == "measure" and row.get("wall_measure_start_epoch")
                else "wall_start_epoch"
            )
            start = float(row[start_key]) - padding_s
            end = float(row["wall_end_epoch"]) + padding_s
            qps_text = row.get("qps") or row.get("target_qps")
            if qps_text is None:
                raise ValueError(f"{path} does not contain qps or target_qps column")
            windows.append(
                Window(qps=float(qps_text), start_epoch=start, end_epoch=end)
            )
    return windows


def in_window(timestamp: float | None, window: Window) -> bool:
    return timestamp is not None and window.start_epoch <= timestamp <= window.end_epoch


def build_report(
    events: list[PrefixEvent],
    evictions: list[EvictEvent],
    native_rates: list[NativeRateSample],
    *,
    windows: list[Window],
    block_size: int,
    value_prefix_len_threshold: int | None,
    restrict_overall_to_windows: bool = False,
) -> dict[str, Any]:
    if restrict_overall_to_windows and windows:
        overall_events = [
            event
            for event in events
            if any(in_window(event.timestamp_epoch, window) for window in windows)
        ]
        overall_evictions = [
            event
            for event in evictions
            if any(in_window(event.timestamp_epoch, window) for window in windows)
        ]
        overall_native_rates = [
            sample
            for sample in native_rates
            if any(in_window(sample.timestamp_epoch, window) for window in windows)
        ]
    else:
        overall_events = events
        overall_evictions = evictions
        overall_native_rates = native_rates

    report: dict[str, Any] = {
        "overall": summarize(
            scope="overall",
            qps=None,
            events=overall_events,
            evictions=overall_evictions,
            native_rates=overall_native_rates,
            block_size=block_size,
            value_prefix_len_threshold=value_prefix_len_threshold,
        )
    }
    if windows:
        by_qps = []
        for window in windows:
            window_events = [
                event for event in events if in_window(event.timestamp_epoch, window)
            ]
            window_evictions = [
                event for event in evictions if in_window(event.timestamp_epoch, window)
            ]
            window_native_rates = [
                sample
                for sample in native_rates
                if in_window(sample.timestamp_epoch, window)
            ]
            by_qps.append(
                summarize(
                    scope="qps",
                    qps=window.qps,
                    events=window_events,
                    evictions=window_evictions,
                    native_rates=window_native_rates,
                    block_size=block_size,
                    value_prefix_len_threshold=value_prefix_len_threshold,
                )
            )
        report["by_qps"] = by_qps
    return report


def flatten_row(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "scope": summary["scope"],
        "qps": "" if summary["qps"] is None else summary["qps"],
        "value_prefix_len_threshold": summary["value_prefix_len_threshold"],
        "events": summary["events"],
        "cache_admitted_requests": summary["cache_admitted_requests"],
        "cache_skipped_requests": summary["cache_skipped_requests"],
        "cache_admitted_ratio": summary["cache_admitted_ratio"],
        "high_value_requests": summary["high_value_requests"],
        "low_value_requests": summary["low_value_requests"],
        "request_hit_rate": summary["request_hit_rate"],
        "high_value_request_hit_rate": summary["high_value_request_hit_rate"],
        "low_value_request_hit_rate": summary["low_value_request_hit_rate"],
        "request_full_hit_rate": summary["request_full_hit_rate"],
        "token_match_rate_vs_limit": summary["token_match_rate_vs_limit"],
        "high_value_token_match_rate_vs_limit": summary[
            "high_value_token_match_rate_vs_limit"
        ],
        "low_value_token_match_rate_vs_limit": summary[
            "low_value_token_match_rate_vs_limit"
        ],
        "token_match_rate_vs_cacheable": summary["token_match_rate_vs_cacheable"],
        "high_value_token_match_rate_vs_cacheable": summary[
            "high_value_token_match_rate_vs_cacheable"
        ],
        "low_value_token_match_rate_vs_cacheable": summary[
            "low_value_token_match_rate_vs_cacheable"
        ],
        "cacheable_token_ratio": summary["cacheable_token_ratio"],
        "avg_matched_prefix": summary["avg_matched_prefix"],
        "p50_matched_prefix": summary["p50_matched_prefix"],
        "p90_matched_prefix": summary["p90_matched_prefix"],
        "p95_matched_prefix": summary["p95_matched_prefix"],
        "p99_matched_prefix": summary["p99_matched_prefix"],
        "max_matched_prefix": summary["max_matched_prefix"],
        "matched_tokens": summary["matched_tokens"],
        "matched_tokens_capped_to_limit": summary[
            "matched_tokens_capped_to_limit"
        ],
        "matched_tokens_capped_to_cacheable": summary[
            "matched_tokens_capped_to_cacheable"
        ],
        "prefix_match_limit_tokens": summary["prefix_match_limit_tokens"],
        "cacheable_tokens": summary["cacheable_tokens"],
        "writeback_tokens": summary["writeback_tokens"],
        "writeback_tokens_per_request": summary["writeback_tokens_per_request"],
        "total_input_tokens": summary["total_input_tokens"],
        "saved_cacheable_tokens_vs_full_sequence": summary[
            "saved_cacheable_tokens_vs_full_sequence"
        ],
        "estimated_cache_blocks": summary["estimated_cache_blocks"],
        "estimated_cache_blocks_per_request": summary[
            "estimated_cache_blocks_per_request"
        ],
        "saved_cache_blocks_estimate_vs_full_sequence": summary[
            "saved_cache_blocks_estimate_vs_full_sequence"
        ],
        "evict_calls": summary["evict_calls"],
        "evict_call_rate": summary["evict_call_rate"],
        "requested_evict_blocks": summary["requested_evict_blocks"],
        "requested_evict_blocks_per_request": summary[
            "requested_evict_blocks_per_request"
        ],
        "evicted_blocks": summary["evicted_blocks"],
        "evicted_blocks_per_request": summary["evicted_blocks_per_request"],
        "evict_satisfaction_rate": summary["evict_satisfaction_rate"],
        "native_reported_hit_rate_last": summary["native_reported_hit_rate_last"],
        "sources": json.dumps(summary["sources"], sort_keys=True),
        "policies": json.dumps(summary["policies"], sort_keys=True),
    }


def write_csv(path: Path, report: dict[str, Any]) -> None:
    rows = [flatten_row(report["overall"])]
    rows.extend(flatten_row(row) for row in report.get("by_qps", []))
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_table(report: dict[str, Any]) -> None:
    rows = [report["overall"], *report.get("by_qps", [])]
    header = (
        "scope",
        "qps",
        "events",
        "admit",
        "req_hit",
        "hi_hit",
        "tok_hit",
        "hi_tok",
        "cacheable",
        "blocks",
        "avg_match",
        "evicted",
        "evict/r",
    )
    print(
        f"{header[0]:<8} {header[1]:>7} {header[2]:>8} {header[3]:>8} "
        f"{header[4]:>9} {header[5]:>9} {header[6]:>9} {header[7]:>9} "
        f"{header[8]:>10} {header[9]:>8} {header[10]:>10} "
        f"{header[11]:>8} {header[12]:>8}"
    )
    for row in rows:
        qps = "-" if row["qps"] is None else f"{row['qps']:g}"
        print(
            f"{row['scope']:<8} {qps:>7} {row['events']:>8} "
            f"{row['cache_admitted_ratio'] * 100:>7.2f}% "
            f"{row['request_hit_rate'] * 100:>8.2f}% "
            f"{row['high_value_request_hit_rate'] * 100:>8.2f}% "
            f"{row['token_match_rate_vs_limit'] * 100:>8.2f}% "
            f"{row['high_value_token_match_rate_vs_limit'] * 100:>8.2f}% "
            f"{row['cacheable_token_ratio'] * 100:>9.2f}% "
            f"{row['estimated_cache_blocks']:>8} "
            f"{row['avg_matched_prefix']:>10.1f} "
            f"{row['evicted_blocks']:>8} "
            f"{row['evicted_blocks_per_request']:>8.2f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize MTGR prefix-cache match metrics from service logs."
    )
    parser.add_argument("--log", type=Path, action="append", required=True)
    parser.add_argument("--source", choices=["auto", "xllm", "vllm"], default="auto")
    parser.add_argument("--qps-summary", type=Path)
    parser.add_argument("--phase", choices=["all", "measure"], default="measure")
    parser.add_argument(
        "--overall-scope",
        choices=["all", "windows"],
        default="all",
        help=(
            "Use all parsed log lines for the overall row, or only log lines "
            "inside --qps-summary windows."
        ),
    )
    parser.add_argument("--window-padding-s", type=float, default=0.0)
    parser.add_argument(
        "--block-size",
        type=int,
        default=1,
        help="KV-cache block size used for estimated cache block counts.",
    )
    parser.add_argument(
        "--value-prefix-len-threshold",
        type=int,
        help=(
            "Prefix length threshold used to classify high/low value events "
            "when the log line does not already contain high_value."
        ),
    )
    parser.add_argument("--year", type=int, default=datetime.now().year)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--print-table", action="store_true")
    args = parser.parse_args()
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")
    if (
        args.value_prefix_len_threshold is not None
        and args.value_prefix_len_threshold < 0
    ):
        raise ValueError("--value-prefix-len-threshold must be non-negative")

    events, evictions, native_rates = parse_logs(
        args.log,
        source=args.source,
        year=args.year,
    )
    windows = (
        load_windows(args.qps_summary, phase=args.phase, padding_s=args.window_padding_s)
        if args.qps_summary
        else []
    )
    report = build_report(
        events,
        evictions,
        native_rates,
        windows=windows,
        block_size=args.block_size,
        value_prefix_len_threshold=args.value_prefix_len_threshold,
        restrict_overall_to_windows=args.overall_scope == "windows",
    )

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if args.out_csv:
        write_csv(args.out_csv, report)
    if args.print_table or (not args.out_json and not args.out_csv):
        print_table(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
