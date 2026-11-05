"""Split records into train/val/test and export to common fine-tuning formats.

Formats:
  messages  {"messages": [{"role": "system"|"user"|"assistant", "content": ...}]}
            Used by Hugging Face TRL, Axolotl, Unsloth, LLaMA-Factory, and most hosted
            fine-tuning services that accept chat JSONL.
  sharegpt  {"conversations": [{"from": "system"|"human"|"gpt", "value": ...}]}
  alpaca    {"instruction": ..., "input": "", "output": ..., "system": ...}  (single-turn only)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from .schema import write_jsonl

FORMATS = ("messages", "sharegpt", "alpaca")


def load_personas(path: str | Path = "configs/personas.yaml") -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_of(record_id: str, val_pct: int = 5, test_pct: int = 5) -> str:
    """Deterministic split from the id hash, so adding data never reshuffles old records."""
    bucket = int(hashlib.sha256(record_id.encode()).hexdigest(), 16) % 100
    if bucket < test_pct:
        return "test"
    if bucket < test_pct + val_pct:
        return "val"
    return "train"


def to_messages(rec: dict, personas: dict, include_system: bool = True) -> dict:
    msgs = [{"role": m["role"], "content": m["content"]} for m in rec["messages"]]
    if include_system:
        msgs.insert(0, {"role": "system", "content": personas[rec["persona"]]["system"].strip()})
    return {"messages": msgs}


def to_sharegpt(rec: dict, personas: dict, include_system: bool = True) -> dict:
    role_map = {"system": "system", "user": "human", "assistant": "gpt"}
    msgs = to_messages(rec, personas, include_system)["messages"]
    return {"conversations": [{"from": role_map[m["role"]], "value": m["content"]} for m in msgs]}


def to_alpaca(rec: dict, personas: dict, include_system: bool = True) -> dict | None:
    if len(rec["messages"]) != 2:
        return None  # Alpaca is single-turn
    out = {"instruction": rec["messages"][0]["content"], "input": "", "output": rec["messages"][1]["content"]}
    if include_system:
        out["system"] = personas[rec["persona"]]["system"].strip()
    return out


CONVERTERS = {"messages": to_messages, "sharegpt": to_sharegpt, "alpaca": to_alpaca}


def build(
    records: list[dict],
    out_dir: str | Path,
    personas: dict,
    formats: tuple[str, ...] = ("messages",),
    val_pct: int = 5,
    test_pct: int = 5,
    include_system: bool = True,
    personas_filter: tuple[str, ...] | None = None,
) -> dict[str, dict[str, int]]:
    """Write <split>.<format>.jsonl files. Returns counts[split][format]."""
    out_dir = Path(out_dir)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for rec in records:
        if personas_filter and rec["persona"] not in personas_filter:
            continue
        splits[split_of(rec["id"], val_pct, test_pct)].append(rec)

    counts: dict[str, dict[str, int]] = {}
    for split, recs in splits.items():
        counts[split] = {}
        for fmt in formats:
            conv = CONVERTERS[fmt]
            rows = [r for r in (conv(rec, personas, include_system) for rec in recs) if r is not None]
            counts[split][fmt] = write_jsonl(out_dir / f"{split}.{fmt}.jsonl", rows)
        # keep the raw records too, so evaluation can use held-out metadata
        write_jsonl(out_dir / f"{split}.records.jsonl", recs)
    return counts
