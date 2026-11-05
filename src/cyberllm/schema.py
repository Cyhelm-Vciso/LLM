"""Record schema, validation, and JSONL helpers.

One record = one training conversation. The system prompt is NOT stored in the
record; it is attached at export time from configs/personas.yaml so the persona
can be edited without regenerating data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

PERSONAS = ("ciso", "expert", "engineer")
DIFFICULTIES = ("foundational", "intermediate", "advanced")
TASK_TYPES = ("advisory", "explain", "analyze", "design", "write", "review", "boundary")
SOURCES = ("seed", "synthetic", "human")
ROLES = ("user", "assistant")

REQUIRED_FIELDS = ("id", "persona", "domain", "difficulty", "task_type", "messages", "source")


def validate_record(rec: Any) -> list[str]:
    """Return a list of human-readable problems. Empty list means valid."""
    if not isinstance(rec, dict):
        return ["record is not a JSON object"]

    errors = [f"missing field '{f}'" for f in REQUIRED_FIELDS if f not in rec]
    if errors:
        return errors

    if not isinstance(rec["id"], str) or not rec["id"].strip():
        errors.append("'id' must be a non-empty string")
    if rec["persona"] not in PERSONAS:
        errors.append(f"'persona' must be one of {PERSONAS}, got {rec['persona']!r}")
    if rec["difficulty"] not in DIFFICULTIES:
        errors.append(f"'difficulty' must be one of {DIFFICULTIES}, got {rec['difficulty']!r}")
    if rec["task_type"] not in TASK_TYPES:
        errors.append(f"'task_type' must be one of {TASK_TYPES}, got {rec['task_type']!r}")
    if rec["source"] not in SOURCES:
        errors.append(f"'source' must be one of {SOURCES}, got {rec['source']!r}")
    if "frameworks" in rec and not (
        isinstance(rec["frameworks"], list) and all(isinstance(x, str) for x in rec["frameworks"])
    ):
        errors.append("'frameworks' must be a list of strings")
    if "reviewed" in rec and not isinstance(rec["reviewed"], bool):
        errors.append("'reviewed' must be a boolean")

    msgs = rec["messages"]
    if not isinstance(msgs, list) or len(msgs) < 2:
        errors.append("'messages' must be a list with at least one user and one assistant turn")
        return errors
    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or m.get("role") not in ROLES:
            errors.append(f"messages[{i}].role must be one of {ROLES} (system is added at export)")
            continue
        if not isinstance(m.get("content"), str) or not m["content"].strip():
            errors.append(f"messages[{i}].content must be a non-empty string")
        expected = ROLES[i % 2]
        if m["role"] != expected:
            errors.append(f"messages[{i}].role should be '{expected}' (turns must alternate, starting with user)")
    if msgs and isinstance(msgs[-1], dict) and msgs[-1].get("role") != "assistant":
        errors.append("last message must be from the assistant")
    return errors


def load_jsonl(path: str | Path) -> Iterator[tuple[int, Any]]:
    """Yield (line_number, parsed_object). Raises ValueError with location on bad JSON."""
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e


def read_records(paths: Iterable[str | Path]) -> list[dict]:
    records: list[dict] = []
    for p in paths:
        records.extend(obj for _, obj in load_jsonl(p))
    return records


def write_jsonl(path: str | Path, records: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def validate_files(paths: Iterable[str | Path]) -> tuple[int, list[str]]:
    """Validate every record in the given files. Returns (count, problems)."""
    count, problems, seen = 0, [], {}
    for p in paths:
        for lineno, rec in load_jsonl(p):
            count += 1
            for err in validate_record(rec):
                problems.append(f"{p}:{lineno}: {err}")
            rid = rec.get("id") if isinstance(rec, dict) else None
            if rid is not None:
                if rid in seen:
                    problems.append(f"{p}:{lineno}: duplicate id '{rid}' (first seen at {seen[rid]})")
                else:
                    seen[rid] = f"{p}:{lineno}"
    return count, problems
