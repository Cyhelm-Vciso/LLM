"""Dataset quality gates: schema, length, boilerplate, near-duplicates, safety lint.

The safety lint is a heuristic tripwire, not a guarantee. Anything it flags goes to
a review file for a human to accept or reject; it is never silently kept.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from .schema import validate_record

MIN_ASSISTANT_CHARS = 200
MAX_ASSISTANT_CHARS = 16000

BOILERPLATE = [
    re.compile(p, re.I)
    for p in (
        r"\bas an ai (language )?model\b",
        r"\bi (?:am|'m) (?:just )?an ai\b",
        r"\bmy (?:training|knowledge) cut-?off\b",
        r"\bi cannot browse the internet\b",
        r"^\s*(?:sure|certainly|of course)[!,.]?\s*(?:here(?:'s| is))",
    )
]

# Patterns that suggest operational offensive content. A match sends the record to
# human review unless it is a 'boundary' example (where the assistant is expected to
# discuss, and decline, such requests).
SAFETY_PATTERNS = {
    "shellcode-bytes": re.compile(r"(?:\\x[0-9a-fA-F]{2}){16,}"),
    "av-disable": re.compile(r"Set-MpPreference\s+-Disable(?:RealtimeMonitoring|IOAVProtection)\s+\$?true", re.I),
    "credential-dump-cmd": re.compile(r"sekurlsa::(?:logonpasswords|wdigest)|lsadump::(?:sam|dcsync)", re.I),
    "ransom-note": re.compile(r"your (?:files|data) (?:have been|are) encrypted.{0,80}(?:bitcoin|btc|monero|xmr)", re.I | re.S),
    "keylogger-code": re.compile(r"(?:SetWindowsHookEx(?:A|W)?\s*\(\s*WH_KEYBOARD_LL|pynput\.keyboard\.Listener)", re.I),
    "self-propagation": re.compile(r"\b(?:self[- ]propagat|worm(?:able)? payload|spread to (?:all|every) host)", re.I),
    "edr-evasion": re.compile(r"\b(?:unhook(?:ing)? (?:ntdll|edr)|bypass(?:ing)? (?:edr|amsi) (?:by|with|using))", re.I),
}


@dataclass
class QualityReport:
    kept: list[dict] = field(default_factory=list)
    flagged: list[dict] = field(default_factory=list)  # needs human review
    dropped: list[tuple[str, str]] = field(default_factory=list)  # (id, reason)

    def summary(self) -> str:
        reasons: dict[str, int] = {}
        for _, r in self.dropped:
            key = r.split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
        lines = [f"kept={len(self.kept)} flagged={len(self.flagged)} dropped={len(self.dropped)}"]
        lines += [f"  dropped {k}: {v}" for k, v in sorted(reasons.items())]
        return "\n".join(lines)


def _assistant_text(rec: dict) -> str:
    return "\n".join(m["content"] for m in rec["messages"] if m["role"] == "assistant")


def _user_text(rec: dict) -> str:
    return "\n".join(m["content"] for m in rec["messages"] if m["role"] == "user")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def safety_hits(text: str) -> list[str]:
    return [name for name, pat in SAFETY_PATTERNS.items() if pat.search(text)]


# --- near-duplicate detection (MinHash + LSH, pure Python) -------------------

_NUM_PERM = 64
_BANDS = 16
_ROWS = _NUM_PERM // _BANDS
_PRIME = (1 << 61) - 1
# universal hash family h_i(x) = (a_i * x + b_i) mod p, deterministic across runs
_COEFFS = [
    (int(hashlib.sha256(f"a{i}".encode()).hexdigest(), 16) % (_PRIME - 1) + 1,
     int(hashlib.sha256(f"b{i}".encode()).hexdigest(), 16) % _PRIME)
    for i in range(_NUM_PERM)
]


def _shingles(text: str, k: int = 5) -> set[str]:
    words = _normalize(text).split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def _minhash(shingles: set[str]) -> list[int]:
    hashes = [int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") % _PRIME for s in shingles]
    if not hashes:
        return [0] * _NUM_PERM
    return [min((a * h + b) % _PRIME for h in hashes) for a, b in _COEFFS]


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def near_duplicates(texts: list[str], threshold: float = 0.8) -> set[int]:
    """Return indices to drop: each later item that is >= threshold similar to an earlier kept one."""
    sh = [_shingles(t) for t in texts]
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    drop: set[int] = set()
    for i, s in enumerate(sh):
        sig = _minhash(s)
        candidates: set[int] = set()
        keys = [(b, tuple(sig[b * _ROWS : (b + 1) * _ROWS])) for b in range(_BANDS)]
        for key in keys:
            candidates.update(buckets.get(key, ()))
        if any(_jaccard(s, sh[j]) >= threshold for j in candidates):
            drop.add(i)
            continue
        for key in keys:
            buckets.setdefault(key, []).append(i)
    return drop


# --- main entry ---------------------------------------------------------------


def run_quality(records: list[dict], dedupe_threshold: float = 0.8) -> QualityReport:
    report = QualityReport()
    candidates: list[dict] = []
    seen_exact: set[str] = set()

    for rec in records:
        rid = rec.get("id", "?") if isinstance(rec, dict) else "?"
        errs = validate_record(rec)
        if errs:
            report.dropped.append((rid, f"schema: {errs[0]}"))
            continue
        a = _assistant_text(rec)
        if len(a) < MIN_ASSISTANT_CHARS and rec["task_type"] != "boundary":
            report.dropped.append((rid, f"too-short: {len(a)} chars"))
            continue
        if len(a) > MAX_ASSISTANT_CHARS:
            report.dropped.append((rid, f"too-long: {len(a)} chars"))
            continue
        bp = next((p.pattern for p in BOILERPLATE if p.search(a)), None)
        if bp:
            report.dropped.append((rid, f"boilerplate: {bp}"))
            continue
        key = hashlib.sha1(_normalize(_user_text(rec)).encode()).hexdigest()
        if key in seen_exact:
            report.dropped.append((rid, "exact-duplicate: same user prompt"))
            continue
        seen_exact.add(key)
        candidates.append(rec)

    dup_idx = near_duplicates([_user_text(r) + "\n" + _assistant_text(r) for r in candidates], dedupe_threshold)
    for i, rec in enumerate(candidates):
        if i in dup_idx:
            report.dropped.append((rec["id"], "near-duplicate"))
            continue
        hits = safety_hits(_assistant_text(rec))
        if hits and rec["task_type"] != "boundary":
            report.flagged.append({**rec, "_flags": hits})
        else:
            report.kept.append(rec)
    return report
