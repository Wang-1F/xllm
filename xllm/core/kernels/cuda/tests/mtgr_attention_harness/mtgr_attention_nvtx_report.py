#!/usr/bin/env python3
"""Summarize the prototype MTGR attention harness NVTX benchmark."""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import statistics
from dataclasses import dataclass, field
from pathlib import Path


NS_PER_MS = 1_000_000.0
ROOT_RE = re.compile(
    r"^MTGR/harness/mtgr_attention/(base|hopper)/(no_match|partial_match)$"
)

BASE_MASK_BUILD = "MTGR/harness/mtgr_attention/base/mask_build"
BASE_FLASHINFER = "MTGR/harness/mtgr_attention/base/one_stage_flashinfer"
HOPPER_FORWARD = "MTGR/harness/mtgr_attention/hopper_forward"
HOPPER_SYNC = "MTGR/harness/mtgr_attention/device_sync"
HOPPER_PREPARE_QKV = "MTGR/attention/prepare_qkv_snd"
HOPPER_KERNEL_CALL = "MTGR/kernel/hopper_unified_call"


@dataclass
class Event:
  idx: int
  start: int
  end: int
  text: str
  tid: int | None
  parent: Event | None = None
  children: list[Event] = field(default_factory=list)

  @property
  def ms(self) -> float:
    return max(0, self.end - self.start) / NS_PER_MS


def load_events(sqlite_path: Path) -> list[Event]:
  conn = sqlite3.connect(str(sqlite_path))
  conn.row_factory = sqlite3.Row
  try:
    has_string_ids = conn.execute(
        "select 1 from sqlite_master where type='table' and name='StringIds'"
    ).fetchone()
    if has_string_ids:
      query = """
        select e.start, e.end, coalesce(e.text, s.value) as text, e.globalTid
        from NVTX_EVENTS e
        left join StringIds s on e.textId = s.id
        where e.end is not null and e.end > e.start
        order by e.start asc, e.end desc
      """
    else:
      query = """
        select e.start, e.end, e.text as text, e.globalTid
        from NVTX_EVENTS e
        where e.end is not null and e.end > e.start
        order by e.start asc, e.end desc
      """
    events: list[Event] = []
    for idx, row in enumerate(conn.execute(query)):
      text = row["text"]
      if not text:
        continue
      text = str(text)
      if not (
          text.startswith("MTGR/harness/mtgr_attention/")
          or text.startswith("MTGR/bench/attention_base/")
          or text.startswith("MTGR/attention/")
          or text.startswith("MTGR/kernel/")
      ):
        continue
      events.append(
          Event(idx, int(row["start"]), int(row["end"]), text, row["globalTid"])
      )
    return events
  finally:
    conn.close()


def build_tree(events: list[Event]) -> None:
  stacks: dict[int | None, list[Event]] = {}
  for event in sorted(events, key=lambda e: (e.tid, e.start, -e.end, e.idx)):
    stack = stacks.setdefault(event.tid, [])
    event.children.clear()
    event.parent = None
    while stack and not (stack[-1].start <= event.start and event.end <= stack[-1].end):
      stack.pop()
    if stack:
      event.parent = stack[-1]
      stack[-1].children.append(event)
    stack.append(event)


def iter_tree(root: Event):
  stack = [root]
  while stack:
    node = stack.pop()
    yield node
    stack.extend(reversed(node.children))


def sum_named(root: Event, name: str) -> float:
  return sum(node.ms for node in iter_tree(root) if node.text == name)


def mean(values: list[float]) -> float:
  return statistics.fmean(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
  if not values:
    return 0.0
  ordered = sorted(values)
  if len(ordered) == 1:
    return ordered[0]
  pos = (len(ordered) - 1) * q
  lo = int(pos)
  hi = min(lo + 1, len(ordered) - 1)
  frac = pos - lo
  return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def read_labels(path: Path) -> list[dict[str, str]]:
  with path.open("r", encoding="utf-8", newline="") as f:
    return list(csv.DictReader(f))


def f6(value: float) -> str:
  return f"{value:.6f}"


def make_event_rows(sqlite_path: Path, labels_path: Path) -> list[dict[str, object]]:
  events = load_events(sqlite_path)
  build_tree(events)
  roots = sorted(
      [event for event in events if ROOT_RE.search(event.text)],
      key=lambda e: (e.start, e.end),
  )
  labels = read_labels(labels_path)
  if len(roots) != len(labels):
    raise SystemExit(
        f"root/label count mismatch: roots={len(roots)} labels={len(labels)}"
    )

  rows = []
  for root, label in zip(roots, labels):
    match = ROOT_RE.search(root.text)
    backend_kind = match.group(1)
    mode = match.group(2)
    expected_backend = "full_flashinfer_base" if backend_kind == "base" else "hopper_unified"
    if label["backend"] != expected_backend or label["mode"] != mode:
      raise SystemExit(
          f"label mismatch idx={label.get('idx')}: root={root.text} label={label}"
      )

    row: dict[str, object] = dict(label)
    row["total_ms"] = f6(root.ms)
    row["base_mask_build_ms"] = f6(sum_named(root, BASE_MASK_BUILD))
    row["base_one_stage_flashinfer_ms"] = f6(sum_named(root, BASE_FLASHINFER))
    row["hopper_forward_ms"] = f6(sum_named(root, HOPPER_FORWARD))
    row["hopper_device_sync_ms"] = f6(sum_named(root, HOPPER_SYNC))
    row["hopper_prepare_qkv_snd_ms"] = f6(sum_named(root, HOPPER_PREPARE_QKV))
    row["hopper_kernel_call_ms"] = f6(sum_named(root, HOPPER_KERNEL_CALL))
    rows.append(row)
  return rows


FIELDS = [
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
    "base_total_ms",
    "base_mask_build_ms",
    "base_one_stage_flashinfer_ms",
    "hopper_total_ms",
    "hopper_forward_ms",
    "hopper_device_sync_ms",
    "hopper_prepare_qkv_snd_ms",
    "hopper_kernel_call_ms",
    "base_over_hopper_speedup",
    "base_minus_hopper_ms",
]


def join_base_and_hopper_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
  by_key = {
      (row["pair_id"], row["mode"], row["repeat_id"], row["backend"]): row
      for row in rows
  }
  output: list[dict[str, object]] = []
  for row in rows:
    if row["backend"] != "hopper_unified":
      continue
    base = by_key.get(
        (row["pair_id"], row["mode"], row["repeat_id"], "full_flashinfer_base")
    )
    if not base:
      raise SystemExit(f"missing base row for hopper row: {row}")
    base_ms = float(base["total_ms"])
    hopper_ms = float(row["total_ms"])
    output.append(
        {
            "pair_id": row["pair_id"],
            "mode": row["mode"],
            "heads": row["heads"],
            "kv_heads": row["kv_heads"],
            "head_dim": row["head_dim"],
            "history": row["history"],
            "context": row["context"],
            "realtime": row["realtime"],
            "realtime_matched": row["realtime_matched"],
            "target": row["target"],
            "total_q": row["total_q"],
            "matched_prefix": row["matched_prefix"],
            "live_q": row["live_q"],
            "repeat_id": row["repeat_id"],
            "base_total_ms": f6(base_ms),
            "base_mask_build_ms": base["base_mask_build_ms"],
            "base_one_stage_flashinfer_ms": base["base_one_stage_flashinfer_ms"],
            "hopper_total_ms": f6(hopper_ms),
            "hopper_forward_ms": row["hopper_forward_ms"],
            "hopper_device_sync_ms": row["hopper_device_sync_ms"],
            "hopper_prepare_qkv_snd_ms": row["hopper_prepare_qkv_snd_ms"],
            "hopper_kernel_call_ms": row["hopper_kernel_call_ms"],
            "base_over_hopper_speedup": f6(base_ms / hopper_ms),
            "base_minus_hopper_ms": f6(base_ms - hopper_ms),
        }
    )
  return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
  with path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)


def nums(rows: list[dict[str, object]], field: str) -> list[float]:
  return [float(row[field]) for row in rows if row.get(field) not in ("", None)]


def summary(rows: list[dict[str, object]]) -> str:
  lines = [
      "# MTGR Attention Harness NVTX Summary",
      "",
      "Host timings are derived only from Nsight Systems NVTX ranges.",
      "",
      "| mode | count | avg base_total_ms | avg hopper_total_ms | p50 hopper_total_ms | p90 hopper_total_ms | avg base/hopper speedup | avg hopper_device_sync_ms |",
      "|---|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for mode in ["no_match", "partial_match"]:
    hopper = [row for row in rows if row["mode"] == mode]
    lines.append(
        f"| {mode} | {len(hopper)} | "
        f"{mean(nums(hopper, 'base_total_ms')):.6f} | "
        f"{mean(nums(hopper, 'hopper_total_ms')):.6f} | "
        f"{percentile(nums(hopper, 'hopper_total_ms'), 0.50):.6f} | "
        f"{percentile(nums(hopper, 'hopper_total_ms'), 0.90):.6f} | "
        f"{mean(nums(hopper, 'base_over_hopper_speedup')):.6f} | "
        f"{mean(nums(hopper, 'hopper_device_sync_ms')):.6f} |"
    )
  return "\n".join(lines)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--sqlite", required=True, type=Path)
  parser.add_argument("--labels", required=True, type=Path)
  parser.add_argument("--csv", required=True, type=Path)
  parser.add_argument("--summary", required=True, type=Path)
  args = parser.parse_args()

  rows = join_base_and_hopper_rows(make_event_rows(args.sqlite, args.labels))
  write_csv(args.csv, rows)
  args.summary.write_text(summary(rows), encoding="utf-8")


if __name__ == "__main__":
  main()
