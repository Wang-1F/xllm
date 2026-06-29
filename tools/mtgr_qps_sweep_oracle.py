#!/usr/bin/env python3
"""MTGR QPS client wrapper that can attach offline oracle admission tensors."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ORIGINAL = Path(__file__).with_name("mtgr_qps_sweep.py")


def pop_custom_args(argv: list[str]) -> tuple[list[str], Path | None]:
  cleaned = [argv[0]]
  mask_path: Path | None = None
  i = 1
  while i < len(argv):
    arg = argv[i]
    if arg == "--oracle-mask":
      if i + 1 >= len(argv):
        raise ValueError("--oracle-mask requires a path")
      mask_path = Path(argv[i + 1])
      i += 2
    elif arg.startswith("--oracle-mask="):
      mask_path = Path(arg.split("=", 1)[1])
      i += 1
    else:
      cleaned.append(arg)
      i += 1
  return cleaned, mask_path


def load_original():
  sys.path.insert(0, str(ORIGINAL.parent))
  spec = importlib.util.spec_from_file_location("mtgr_qps_sweep_original",
                                                ORIGINAL)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"failed to load {ORIGINAL}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def load_mask(path: Path | None) -> list[int] | None:
  if path is None:
    return None
  values: list[int] = []
  with path.open("r", encoding="utf-8") as src:
    for line_no, line in enumerate(src, start=1):
      if not line.strip():
        continue
      raw = json.loads(line)
      if int(raw.get("index", len(values))) != len(values):
        raise ValueError(f"mask index mismatch at line {line_no}: {path}")
      values.append(1 if int(raw["oracle_admit"]) != 0 else 0)
  if not values:
    raise ValueError(f"oracle mask is empty: {path}")
  return values


ARGV, MASK_PATH = pop_custom_args(sys.argv)
sys.argv = ARGV
mod = load_original()
MASK_VALUES = load_mask(MASK_PATH)
_original_build_xllm_payload = mod.build_xllm_payload


def load_dataset_with_oracle(path: Path) -> list[Any]:
  items: list[Any] = []
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
      mod.validate_item(request_id,
                        token_ids,
                        segment_offsets,
                        segment_rules,
                        line_no=line_no)
      index = len(items)
      oracle_admit = None
      if MASK_VALUES is not None:
        if index >= len(MASK_VALUES):
          raise ValueError(f"dataset has more rows than mask: index={index}")
        oracle_admit = MASK_VALUES[index]
      items.append(
          SimpleNamespace(request_id=request_id,
                          token_ids=list(token_ids),
                          segment_offsets=list(segment_offsets),
                          segment_rules=list(segment_rules),
                          oracle_admit=oracle_admit))
  if not items:
    raise ValueError(f"dataset is empty: {path}")
  if MASK_VALUES is not None and len(items) != len(MASK_VALUES):
    raise ValueError(
        f"dataset/mask length mismatch: dataset={len(items)} mask={len(MASK_VALUES)}")
  return items


def build_xllm_payload_with_oracle(item: Any, *, model: str, max_tokens: int,
                                   beam_width: int) -> dict[str, Any]:
  payload = _original_build_xllm_payload(item,
                                         model=model,
                                         max_tokens=max_tokens,
                                         beam_width=beam_width)
  def append_i64_tensor(name: str, value: Any) -> None:
    if value is None:
      return
    payload["input_tensors"].append({
        "name": name,
        "data_type": "INT64",
        "shape": [1],
        "contents": {
            "int64_contents": [int(value)],
        },
    })

  oracle_admit = getattr(item, "oracle_admit", None)
  append_i64_tensor("oracle_admit", oracle_admit)
  return payload


mod.load_dataset = load_dataset_with_oracle
mod.build_xllm_payload = build_xllm_payload_with_oracle


if __name__ == "__main__":
  raise SystemExit(mod.main())
