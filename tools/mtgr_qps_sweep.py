#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp

from mtgr_prefix_cache_report import (
    build_report,
    load_windows,
    parse_logs,
    print_table as print_prefix_table,
    write_csv as write_prefix_csv,
)


@dataclass(frozen=True)
class DatasetItem:
    request_id: str
    token_ids: list[int]
    segment_offsets: list[int]
    segment_rules: list[int]


@dataclass(frozen=True)
class PreparedRequest:
    request_id: str
    body: bytes


@dataclass
class RequestResult:
    request_id: str
    scheduled_at: float
    start_at: float
    end_at: float
    phase: str
    status: int
    ok: bool
    error: str
    response_bytes: int

    @property
    def latency_ms(self) -> float:
        return (self.end_at - self.start_at) * 1000.0

    @property
    def schedule_lag_ms(self) -> float:
        return max(0.0, (self.start_at - self.scheduled_at) * 1000.0)


@dataclass
class QpsRun:
    results: list[RequestResult]
    wall_start_epoch: float
    wall_warmup_start_epoch: float
    wall_measure_start_epoch: float
    wall_end_epoch: float
    prime_request_count: int
    warmup_request_count: int
    measure_request_count: int


def load_dataset(path: Path) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            request_id = str(raw.get("id", f"line_{line_no:08d}"))
            token_ids = raw.get("token_ids")
            segment_offsets = raw.get("segment_offsets")
            segment_rules = raw.get("segment_rules")
            validate_item(
                request_id,
                token_ids,
                segment_offsets,
                segment_rules,
                line_no=line_no,
            )
            items.append(
                DatasetItem(
                    request_id=request_id,
                    token_ids=list(token_ids),
                    segment_offsets=list(segment_offsets),
                    segment_rules=list(segment_rules),
                )
            )
    if not items:
        raise ValueError(f"dataset is empty: {path}")
    return items


def validate_item(
    request_id: str,
    token_ids: Any,
    segment_offsets: Any,
    segment_rules: Any,
    *,
    line_no: int,
) -> None:
    if not isinstance(token_ids, list) or not token_ids:
        raise ValueError(f"line {line_no} {request_id}: token_ids must be a non-empty list")
    if not all(isinstance(value, int) for value in token_ids):
        raise ValueError(f"line {line_no} {request_id}: token_ids must be ints")
    if not isinstance(segment_offsets, list) or len(segment_offsets) < 2:
        raise ValueError(
            f"line {line_no} {request_id}: segment_offsets must contain at least 2 ints"
        )
    if not all(isinstance(value, int) for value in segment_offsets):
        raise ValueError(f"line {line_no} {request_id}: segment_offsets must be ints")
    if segment_offsets[0] != 0:
        raise ValueError(f"line {line_no} {request_id}: segment_offsets[0] must be 0")
    if segment_offsets[-1] != len(token_ids):
        raise ValueError(
            f"line {line_no} {request_id}: segment_offsets[-1]={segment_offsets[-1]} "
            f"must equal len(token_ids)={len(token_ids)}"
        )
    if any(end < start for start, end in zip(segment_offsets, segment_offsets[1:])):
        raise ValueError(
            f"line {line_no} {request_id}: segment_offsets must be monotonic"
        )
    if not isinstance(segment_rules, list):
        raise ValueError(f"line {line_no} {request_id}: segment_rules must be a list")
    if len(segment_rules) != len(segment_offsets) - 1:
        raise ValueError(
            f"line {line_no} {request_id}: len(segment_rules) must be "
            "len(segment_offsets)-1"
        )
    if any(rule not in (0, 1, 2) for rule in segment_rules):
        raise ValueError(
            f"line {line_no} {request_id}: segment_rules only supports 0, 1, 2"
        )


def build_xllm_payload(
    item: DatasetItem,
    *,
    model: str,
    max_tokens: int,
    beam_width: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": "",
        "max_tokens": max_tokens,
        "beam_width": beam_width,
        "stream": False,
        "input_tensors": [
            {
                "name": "token_ids",
                "data_type": "INT64",
                "shape": [len(item.token_ids)],
                "contents": {"int64_contents": item.token_ids},
            },
            {
                "name": "segment_offsets",
                "data_type": "INT32",
                "shape": [len(item.segment_offsets)],
                "contents": {"int_contents": item.segment_offsets},
            },
            {
                "name": "segment_rules",
                "data_type": "INT32",
                "shape": [len(item.segment_rules)],
                "contents": {"int_contents": item.segment_rules},
            },
        ],
    }


def build_vllm_payload(
    item: DatasetItem,
    *,
    model: str,
    max_tokens: int,
    beam_width: int,
    return_token_ids: bool = False,
) -> dict[str, Any]:
    del beam_width
    payload = {
        "model": model,
        "prompt": item.token_ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
        "segment_offsets": item.segment_offsets,
        "segment_rules": item.segment_rules,
    }
    if return_token_ids:
        payload["return_token_ids"] = True
    return payload


def prepare_requests(
    items: list[DatasetItem],
    *,
    adapter: str,
    model: str,
    max_tokens: int,
    beam_width: int,
    vllm_return_token_ids: bool = False,
) -> list[PreparedRequest]:
    prepared: list[PreparedRequest] = []
    for item in items:
        if adapter == "xllm":
            payload = build_xllm_payload(
                item,
                model=model,
                max_tokens=max_tokens,
                beam_width=beam_width,
            )
        elif adapter == "vllm":
            payload = build_vllm_payload(
                item,
                model=model,
                max_tokens=max_tokens,
                beam_width=beam_width,
                return_token_ids=vllm_return_token_ids,
            )
        else:
            raise ValueError(f"unsupported adapter: {adapter}")
        prepared.append(
            PreparedRequest(
                request_id=item.request_id,
                body=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            )
        )
    return prepared


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


def summarize_results(
    *,
    qps: float,
    prime_qps: float,
    prime_s: float,
    warmup_s: float,
    measure_s: float,
    results: list[RequestResult],
    target_tp99_ms: float | None,
    min_success_rate: float,
) -> dict[str, Any]:
    prime = [result for result in results if result.phase == "prime"]
    warmup = [result for result in results if result.phase == "warmup"]
    measured = [result for result in results if result.phase == "measure"]
    prime_ok = [result for result in prime if result.ok]
    warmup_ok = [result for result in warmup if result.ok]
    ok = [result for result in measured if result.ok]
    failed = [result for result in measured if not result.ok]
    latencies = [result.latency_ms for result in ok]
    lags = [result.schedule_lag_ms for result in measured]
    measure_start = min((result.scheduled_at for result in measured), default=0.0)
    measure_deadline = measure_start + measure_s
    issued_in_window = [
        result for result in measured if result.start_at <= measure_deadline
    ]
    completed_in_window = [
        result for result in ok if result.end_at <= measure_deadline
    ]
    measure_drain_s = (
        max((result.end_at for result in measured), default=measure_start)
        - measure_start
        if measured
        else 0.0
    )
    success_rate = len(ok) / len(measured) if measured else 0.0
    tp99 = percentile(latencies, 99.0)
    accept = success_rate >= min_success_rate
    if target_tp99_ms is not None:
        accept = accept and bool(latencies) and tp99 <= target_tp99_ms
    timeout_count = sum(1 for result in failed if result.error == "timeout")
    status_error_count = sum(
        1 for result in failed if not result.error and result.status != 200
    )
    other_error_count = len(failed) - timeout_count - status_error_count
    return {
        "qps": qps,
        "prime_qps": prime_qps,
        "prime_s": prime_s,
        "warmup_s": warmup_s,
        "measure_s": measure_s,
        "sent_total": len(results),
        "prime_total": len(prime),
        "prime_success": len(prime_ok),
        "prime_failed": len(prime) - len(prime_ok),
        "warmup_total": len(warmup),
        "warmup_success": len(warmup_ok),
        "warmup_failed": len(warmup) - len(warmup_ok),
        "measure_total": len(measured),
        "measure_success": len(ok),
        "measure_failed": len(failed),
        "timeout": timeout_count,
        "status_error": status_error_count,
        "other_error": other_error_count,
        "success_rate": success_rate,
        "scheduled_qps": len(measured) / measure_s if measure_s > 0 else 0.0,
        "issued_in_window": len(issued_in_window),
        "issued_qps": len(issued_in_window) / measure_s if measure_s > 0 else 0.0,
        "completed_in_window": len(completed_in_window),
        "completed_in_window_qps": (
            len(completed_in_window) / measure_s if measure_s > 0 else 0.0
        ),
        "measure_drain_s": measure_drain_s,
        "drain_completed_qps": len(ok) / measure_drain_s if measure_drain_s > 0 else 0.0,
        "completed_qps": len(ok) / measure_s if measure_s > 0 else 0.0,
        "latency_avg_ms": sum(latencies) / len(latencies) if latencies else float("nan"),
        "latency_tp50_ms": percentile(latencies, 50.0),
        "latency_tp90_ms": percentile(latencies, 90.0),
        "latency_tp95_ms": percentile(latencies, 95.0),
        "latency_tp99_ms": tp99,
        "latency_max_ms": max(latencies) if latencies else float("nan"),
        "schedule_lag_tp99_ms": percentile(lags, 99.0),
        "schedule_lag_max_ms": max(lags) if lags else float("nan"),
        "target_tp99_ms": target_tp99_ms,
        "accept": accept,
    }


async def send_one(
    session: aiohttp.ClientSession,
    *,
    url: str,
    request: PreparedRequest,
    scheduled_at: float,
    phase: str,
    timeout_s: float,
    semaphore: asyncio.Semaphore | None,
) -> RequestResult:
    headers = {"Content-Type": "application/json"}
    start_at = time.perf_counter()
    status = 0
    ok = False
    error = ""
    response_bytes = 0
    if semaphore is not None:
        async with semaphore:
            start_at = time.perf_counter()
            try:
                async with session.post(
                    url,
                    data=request.body,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout_s),
                ) as response:
                    data = await response.read()
                    status = response.status
                    response_bytes = len(data)
                    ok = status == 200
            except asyncio.TimeoutError:
                error = "timeout"
            except Exception as exc:  # noqa: BLE001 - preserve error string in CSV.
                error = f"{type(exc).__name__}:{exc}"
    else:
        start_at = time.perf_counter()
        try:
            async with session.post(
                url,
                data=request.body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout_s),
            ) as response:
                data = await response.read()
                status = response.status
                response_bytes = len(data)
                ok = status == 200
        except asyncio.TimeoutError:
            error = "timeout"
        except Exception as exc:  # noqa: BLE001 - preserve error string in CSV.
            error = f"{type(exc).__name__}:{exc}"
    end_at = time.perf_counter()
    return RequestResult(
        request_id=request.request_id,
        scheduled_at=scheduled_at,
        start_at=start_at,
        end_at=end_at,
        phase=phase,
        status=status,
        ok=ok,
        error=error,
        response_bytes=response_bytes,
    )


async def run_qps_once(
    prepared: list[PreparedRequest],
    *,
    url: str,
    qps: float,
    prime_qps: float,
    prime_s: float,
    prime_requests: int,
    prime_settle_s: float,
    warmup_s: float,
    measure_s: float,
    timeout_s: float,
    max_concurrency: int,
    order: str,
    seed: int,
    start_index: int = 0,
) -> QpsRun:
    if qps <= 0:
        raise ValueError("qps must be positive")
    if prime_qps <= 0:
        raise ValueError("prime_qps must be positive")
    if prime_s < 0 or prime_requests < 0 or prime_settle_s < 0:
        raise ValueError("invalid prime settings")
    if warmup_s < 0 or measure_s <= 0:
        raise ValueError("invalid warmup/measure duration")

    start_index = start_index % len(prepared)
    requests = list(prepared[start_index:]) + list(prepared[:start_index])
    if order == "shuffle":
        random.Random(seed).shuffle(requests)
    elif order != "sequential":
        raise ValueError(f"unsupported order: {order}")

    connector_limit = 0 if max_concurrency == 0 else max_concurrency
    connector = aiohttp.TCPConnector(limit=connector_limit, force_close=False)
    semaphore = None if max_concurrency == 0 else asyncio.Semaphore(max_concurrency)
    results: list[RequestResult] = []

    async def run_phase(
        session: aiohttp.ClientSession,
        *,
        request_offset: int,
        request_count: int,
        phase_qps: float,
        phase: str,
    ) -> int:
        if request_count <= 0:
            return request_offset
        start = time.perf_counter()
        tasks: list[asyncio.Task[RequestResult]] = []
        for idx in range(request_count):
            scheduled_at = start + idx / phase_qps
            now = time.perf_counter()
            if scheduled_at > now:
                await asyncio.sleep(scheduled_at - now)
            request = requests[(request_offset + idx) % len(requests)]
            task = asyncio.create_task(
                send_one(
                    session,
                    url=url,
                    request=request,
                    scheduled_at=scheduled_at,
                    phase=phase,
                    timeout_s=timeout_s,
                    semaphore=semaphore,
                )
            )
            tasks.append(task)
        if tasks:
            for task in asyncio.as_completed(tasks):
                results.append(await task)
        return request_offset + request_count

    async def run_warmup_measure(
        session: aiohttp.ClientSession,
        *,
        request_offset: int,
        warmup_request_count: int,
        measure_request_count: int,
    ) -> None:
        total_requests = warmup_request_count + measure_request_count
        if total_requests <= 0:
            return
        start = time.perf_counter()
        tasks: list[asyncio.Task[RequestResult]] = []
        for idx in range(total_requests):
            scheduled_at = start + idx / qps
            now = time.perf_counter()
            if scheduled_at > now:
                await asyncio.sleep(scheduled_at - now)
            phase = "warmup" if idx < warmup_request_count else "measure"
            request = requests[(request_offset + idx) % len(requests)]
            task = asyncio.create_task(
                send_one(
                    session,
                    url=url,
                    request=request,
                    scheduled_at=scheduled_at,
                    phase=phase,
                    timeout_s=timeout_s,
                    semaphore=semaphore,
                )
            )
            tasks.append(task)
        if tasks:
            for task in asyncio.as_completed(tasks):
                results.append(await task)

    prime_request_count = (
        prime_requests if prime_requests > 0 else math.ceil(prime_qps * prime_s)
    )
    warmup_request_count = math.ceil(qps * warmup_s)
    measure_request_count = max(1, math.ceil(qps * measure_s))

    wall_start_epoch = time.time()
    request_offset = 0
    async with aiohttp.ClientSession(connector=connector) as session:
        request_offset = await run_phase(
            session,
            request_offset=request_offset,
            request_count=prime_request_count,
            phase_qps=prime_qps,
            phase="prime",
        )
        if prime_request_count > 0 and prime_settle_s > 0:
            await asyncio.sleep(prime_settle_s)
        wall_warmup_start_epoch = time.time()
        wall_measure_start_epoch = (
            wall_warmup_start_epoch
            if warmup_request_count == 0
            else wall_warmup_start_epoch + warmup_request_count / qps
        )
        await run_warmup_measure(
            session,
            request_offset=request_offset,
            warmup_request_count=warmup_request_count,
            measure_request_count=measure_request_count,
        )
    wall_end_epoch = time.time()
    results.sort(key=lambda item: item.scheduled_at)
    return QpsRun(
        results=results,
        wall_start_epoch=wall_start_epoch,
        wall_warmup_start_epoch=wall_warmup_start_epoch,
        wall_measure_start_epoch=wall_measure_start_epoch,
        wall_end_epoch=wall_end_epoch,
        prime_request_count=prime_request_count,
        warmup_request_count=warmup_request_count,
        measure_request_count=measure_request_count,
    )


def write_request_results(path: Path, results: list[RequestResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "request_id",
                "phase",
                "status",
                "ok",
                "error",
                "latency_ms",
                "schedule_lag_ms",
                "response_bytes",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.request_id,
                    result.phase,
                    result.status,
                    int(result.ok),
                    result.error,
                    f"{result.latency_ms:.3f}",
                    f"{result.schedule_lag_ms:.3f}",
                    result.response_bytes,
                ]
            )


def parse_qps_list(value: str | None) -> list[float]:
    if not value:
        return []
    qps_values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not qps_values:
        raise ValueError("--qps-list did not contain any values")
    if any(qps <= 0 for qps in qps_values):
        raise ValueError("--qps-list values must be positive")
    return qps_values


def summary_csv_header() -> list[str]:
    return [
        "qps",
        "prime_qps",
        "prime_s",
        "warmup_s",
        "measure_s",
        "wall_start_epoch",
        "wall_warmup_start_epoch",
        "wall_measure_start_epoch",
        "wall_end_epoch",
        "wall_start_iso",
        "wall_warmup_start_iso",
        "wall_measure_start_iso",
        "wall_end_iso",
        "sent_total",
        "prime_total",
        "prime_success",
        "prime_failed",
        "warmup_total",
        "warmup_success",
        "warmup_failed",
        "measure_total",
        "measure_success",
        "measure_failed",
        "timeout",
        "status_error",
        "other_error",
        "success_rate",
        "scheduled_qps",
        "issued_in_window",
        "issued_qps",
        "completed_in_window",
        "completed_in_window_qps",
        "measure_drain_s",
        "drain_completed_qps",
        "completed_qps",
        "latency_avg_ms",
        "latency_tp50_ms",
        "latency_tp90_ms",
        "latency_tp95_ms",
        "latency_tp99_ms",
        "latency_max_ms",
        "schedule_lag_tp99_ms",
        "schedule_lag_max_ms",
        "target_tp99_ms",
        "accept",
    ]


def summary_row(summary: dict[str, Any]) -> list[Any]:
    row = []
    for key in summary_csv_header():
        value = summary[key]
        if isinstance(value, float):
            row.append(f"{value:.6f}")
        elif isinstance(value, bool):
            row.append(int(value))
        else:
            row.append(value)
    return row


async def async_main(args: argparse.Namespace) -> int:
    items = load_dataset(args.dataset)
    limit = args.dataset_limit if args.dataset_limit and args.dataset_limit > 0 else None
    if limit is not None:
        items = items[:limit]

    max_tokens = args.max_tokens
    if max_tokens is None:
        max_tokens = 1

    prepared = prepare_requests(
        items,
        adapter=args.adapter,
        model=args.model,
        max_tokens=max_tokens,
        beam_width=args.beam_width,
        vllm_return_token_ids=args.vllm_return_token_ids,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    qps_values = parse_qps_list(args.qps_list)
    if not qps_values:
        qps_values = [args.qps]

    summary_path = args.out_dir / "summary.csv"
    detail_dir = args.out_dir / "details"
    if args.write_details:
        detail_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(summary_csv_header())
        request_cursor = 0
        for qps in qps_values:
            prime_qps = args.prime_qps if args.prime_qps else qps
            print(
                f"[RUN] adapter={args.adapter} qps={qps:g} "
                f"prime_qps={prime_qps:g} prime_s={args.prime_s:g} "
                f"prime_requests={args.prime_requests}"
            )
            start_index = request_cursor if args.advance_between_qps else 0
            qps_run = await run_qps_once(
                prepared,
                url=args.url,
                qps=qps,
                prime_qps=prime_qps,
                prime_s=args.prime_s,
                prime_requests=args.prime_requests,
                prime_settle_s=args.prime_settle_s,
                warmup_s=args.warmup_s,
                measure_s=args.measure_s,
                timeout_s=args.timeout_s,
                max_concurrency=args.max_concurrency,
                order=args.order,
                seed=args.seed,
                start_index=start_index,
            )
            results = qps_run.results
            if args.advance_between_qps:
                request_cursor += max(
                    1,
                    qps_run.prime_request_count
                    + qps_run.warmup_request_count
                    + qps_run.measure_request_count,
                )
            summary = summarize_results(
                qps=qps,
                prime_qps=prime_qps,
                prime_s=(
                    args.prime_s
                    if args.prime_requests <= 0
                    else args.prime_requests / prime_qps
                ),
                warmup_s=args.warmup_s,
                measure_s=args.measure_s,
                results=results,
                target_tp99_ms=args.target_tp99_ms,
                min_success_rate=args.min_success_rate,
            )
            summary.update(
                {
                    "wall_start_epoch": qps_run.wall_start_epoch,
                    "wall_warmup_start_epoch": qps_run.wall_warmup_start_epoch,
                    "wall_measure_start_epoch": qps_run.wall_measure_start_epoch,
                    "wall_end_epoch": qps_run.wall_end_epoch,
                    "wall_start_iso": datetime.fromtimestamp(
                        qps_run.wall_start_epoch
                    ).isoformat(timespec="seconds"),
                    "wall_warmup_start_iso": datetime.fromtimestamp(
                        qps_run.wall_warmup_start_epoch
                    ).isoformat(timespec="seconds"),
                    "wall_measure_start_iso": datetime.fromtimestamp(
                        qps_run.wall_measure_start_epoch
                    ).isoformat(timespec="seconds"),
                    "wall_end_iso": datetime.fromtimestamp(
                        qps_run.wall_end_epoch
                    ).isoformat(timespec="seconds"),
                }
            )
            summaries.append(summary)
            writer.writerow(summary_row(summary))
            f.flush()
            if args.write_details:
                safe_qps = str(qps).replace(".", "p")
                write_request_results(detail_dir / f"qps_{safe_qps}.csv", results)
            print(
                "[SUMMARY] "
                f"qps={qps:g} success={summary['measure_success']}/"
                f"{summary['measure_total']} "
                f"prime={summary['prime_success']}/{summary['prime_total']} "
                f"warmup={summary['warmup_success']}/{summary['warmup_total']} "
                f"issued_qps={summary['issued_qps']:.3f} "
                f"drain_qps={summary['drain_completed_qps']:.3f} "
                f"completed_qps={summary['completed_qps']:.3f} "
                f"tp50={summary['latency_tp50_ms']:.3f}ms "
                f"tp95={summary['latency_tp95_ms']:.3f}ms "
                f"tp99={summary['latency_tp99_ms']:.3f}ms "
                f"accept={summary['accept']}"
            )

    best = None
    accepted = [summary for summary in summaries if summary["accept"]]
    if accepted:
        best = max(accepted, key=lambda item: item["qps"])
    result = {
        "adapter": args.adapter,
        "dataset": str(args.dataset),
        "dataset_count": len(items),
        "url": args.url,
        "model": args.model,
        "target_tp99_ms": args.target_tp99_ms,
        "min_success_rate": args.min_success_rate,
        "prime_s": args.prime_s,
        "prime_qps": args.prime_qps,
        "prime_requests": args.prime_requests,
        "prime_settle_s": args.prime_settle_s,
        "max_concurrency": args.max_concurrency,
        "open_loop_unbounded": args.max_concurrency == 0,
        "advance_between_qps": args.advance_between_qps,
        "best_qps": best["qps"] if best else None,
        "summaries": summaries,
    }
    result_path = args.out_dir / "summary.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {summary_path}")
    print(f"wrote {result_path}")
    if args.prefix_log:
        events, evictions, native_rates = parse_logs(
            args.prefix_log,
            source=args.prefix_source,
            year=args.prefix_year,
        )
        windows = load_windows(
            summary_path,
            phase=args.prefix_phase,
            padding_s=args.prefix_window_padding_s,
        )
        prefix_report = build_report(
            events,
            evictions,
            native_rates,
            windows=windows,
            block_size=args.prefix_block_size,
            value_prefix_len_threshold=args.prefix_value_threshold,
            restrict_overall_to_windows=True,
        )
        prefix_json_path = args.prefix_report_json or (
            args.out_dir / "prefix_cache_report.json"
        )
        prefix_csv_path = args.prefix_report_csv or (
            args.out_dir / "prefix_cache_report.csv"
        )
        prefix_json_path.parent.mkdir(parents=True, exist_ok=True)
        prefix_json_path.write_text(
            json.dumps(prefix_report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        write_prefix_csv(prefix_csv_path, prefix_report)
        print(f"wrote {prefix_json_path}")
        print(f"wrote {prefix_csv_path}")
        print_prefix_table(prefix_report)
    if best:
        print(
            f"best_accepted_qps={best['qps']:g} "
            f"tp99_ms={best['latency_tp99_ms']:.3f}"
        )
    else:
        print("best_accepted_qps=None")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run MTGR fixed-QPS load tests from a decoupled JSONL dataset."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--adapter", choices=["xllm", "vllm"], required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="Qwen3-0.6B")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--qps", type=float, default=1.0)
    parser.add_argument("--qps-list", help="Comma-separated fixed QPS values.")
    parser.add_argument("--target-tp99-ms", type=float)
    parser.add_argument("--min-success-rate", type=float, default=0.999)
    parser.add_argument(
        "--prime-s",
        type=float,
        default=0.0,
        help=(
            "Seconds of prime traffic to send before warmup/measure. Prime "
            "requests are excluded from latency acceptance and prefix report "
            "measure windows."
        ),
    )
    parser.add_argument(
        "--prime-qps",
        type=float,
        help="Prime traffic QPS. Defaults to the current measured QPS.",
    )
    parser.add_argument(
        "--prime-requests",
        type=int,
        default=0,
        help=(
            "Fixed number of prime requests. When positive, overrides "
            "--prime-s for request count and records prime_s=prime_requests/prime_qps."
        ),
    )
    parser.add_argument(
        "--prime-settle-s",
        type=float,
        default=0.0,
        help="Optional pause after prime requests finish and before warmup starts.",
    )
    parser.add_argument("--warmup-s", type=float, default=5.0)
    parser.add_argument("--measure-s", type=float, default=20.0)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=256,
        help=(
            "Client-side max in-flight HTTP requests. Set 0 for open-loop "
            "unbounded issue rate: requests are started on the QPS schedule "
            "without a client semaphore or aiohttp connector cap."
        ),
    )
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--beam-width", type=int, default=1)
    parser.add_argument(
        "--vllm-return-token-ids",
        action="store_true",
        help=(
            "Request token-id echo from the vLLM OpenAI endpoint. Disabled by "
            "default for MTGR prefill-only base measurements."
        ),
    )
    parser.add_argument("--order", choices=["sequential", "shuffle"], default="sequential")
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument(
        "--advance-between-qps",
        action="store_true",
        help=(
            "Advance the dataset cursor between QPS points so one sweep does not "
            "replay identical requests and contaminate prefix-cache state."
        ),
    )
    parser.add_argument("--write-details", action="store_true")
    parser.add_argument(
        "--prefix-log",
        type=Path,
        action="append",
        help=(
            "Service log to parse after the sweep for prefix-cache match metrics. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--prefix-source",
        choices=["auto", "xllm", "vllm"],
        default="auto",
    )
    parser.add_argument(
        "--prefix-phase",
        choices=["all", "measure"],
        default="measure",
        help="QPS window phase used for prefix-cache report rows.",
    )
    parser.add_argument("--prefix-window-padding-s", type=float, default=0.0)
    parser.add_argument("--prefix-block-size", type=int, default=1)
    parser.add_argument("--prefix-value-threshold", type=int)
    parser.add_argument("--prefix-year", type=int, default=datetime.now().year)
    parser.add_argument("--prefix-report-json", type=Path)
    parser.add_argument("--prefix-report-csv", type=Path)
    args = parser.parse_args()

    if args.max_concurrency < 0:
        raise ValueError("--max-concurrency must be non-negative; 0 means unbounded")
    if args.prefix_block_size <= 0:
        raise ValueError("--prefix-block-size must be positive")
    if args.prefix_value_threshold is not None and args.prefix_value_threshold < 0:
        raise ValueError("--prefix-value-threshold must be non-negative")
    if not (0.0 < args.min_success_rate <= 1.0):
        raise ValueError("--min-success-rate must be in (0, 1]")
    if args.prime_s < 0:
        raise ValueError("--prime-s must be non-negative")
    if args.prime_requests < 0:
        raise ValueError("--prime-requests must be non-negative")
    if args.prime_qps is not None and args.prime_qps <= 0:
        raise ValueError("--prime-qps must be positive")
    if args.prime_settle_s < 0:
        raise ValueError("--prime-settle-s must be non-negative")
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
