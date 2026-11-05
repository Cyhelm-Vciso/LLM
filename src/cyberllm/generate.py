"""Synthetic data generation from the topic taxonomy using a 'teacher' model.

The teacher must be a model whose licence/terms allow using its outputs to train
other models (e.g. many open-weight models served locally through vLLM or Ollama).
Check the teacher's licence before you publish a dataset or model.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import yaml

from .llm import ChatBackend, extract_json
from .schema import load_jsonl, validate_record

GENERATOR_SYSTEM = """You write high-quality training conversations for a cybersecurity assistant.
Each conversation has a realistic, specific request from a practitioner and an expert answer.

Rules for the request ("user"):
- Concrete and situational: include the organisation type/size, constraints, and what the person needs.
- Vary tone and length. Never mention that this is training data.

Rules for the answer ("assistant"):
- Written in the voice of the persona described below. Lead with the answer or recommendation.
- Technically accurate. Reference framework identifiers only when you are certain they are correct
  (e.g. NIST CSF 2.0 function names, MITRE ATT&CK technique IDs). Never invent statistics, laws,
  deadlines, CVE numbers, or quotations. If something depends on jurisdiction or facts not given,
  say so and state the assumption.
- Practical: steps, owners, artefacts, queries or configs where useful, and how to verify.
- Defensive purpose only. No working malware, exploit chains, or evasion techniques.
- For task type 'boundary': the user drifts toward clearly harmful help. The assistant briefly says
  what it will not provide and why, without lecturing, then gives genuinely useful defensive help
  (detection, hardening, authorised testing approaches, or who to involve).

Return ONLY a JSON object: {"user": "<request>", "assistant": "<answer>"}"""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"user": {"type": "string"}, "assistant": {"type": "string"}},
    "required": ["user", "assistant"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Job:
    persona: str
    domain: str
    subtopic: str
    frameworks: tuple[str, ...]
    task_type: str
    difficulty: str
    variant: int

    @property
    def id(self) -> str:
        key = f"{self.persona}|{self.domain}|{self.subtopic}|{self.task_type}|{self.difficulty}|{self.variant}"
        return f"{self.persona}-syn-{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def load_taxonomy(path: str | Path = "configs/taxonomy.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def plan_jobs(
    taxonomy: dict,
    n: int,
    personas: tuple[str, ...] | None = None,
    seed: int = 0,
    boundary_share: float = 0.08,
) -> list[Job]:
    """Sample n jobs, balanced across personas and domains; ~boundary_share are safety examples."""
    rng = random.Random(seed)
    task_types = [t for t in taxonomy["task_types"] if t != "boundary"]
    cells = []
    for persona, domains in taxonomy["personas"].items():
        if personas and persona not in personas:
            continue
        for domain, spec in domains.items():
            for sub in spec["subtopics"]:
                cells.append((persona, domain, sub, tuple(spec.get("frameworks", []))))
    if not cells:
        raise ValueError("no taxonomy cells match the requested personas")

    jobs, counter = [], {}
    for i in range(n):
        persona, domain, sub, fw = cells[i % len(cells)] if i < len(cells) else rng.choice(cells)
        task = "boundary" if rng.random() < boundary_share else rng.choice(task_types)
        diff = rng.choice(taxonomy["difficulties"])
        key = (persona, domain, sub, task, diff)
        counter[key] = counter.get(key, 0) + 1
        jobs.append(Job(persona, domain, sub, fw, task, diff, counter[key]))
    return jobs


def job_prompt(job: Job, personas_cfg: dict, task_desc: str) -> str:
    persona = personas_cfg[job.persona]
    fw = ", ".join(job.frameworks) or "none in particular"
    return (
        f"Persona of the assistant: {persona['name']}\n"
        f"Persona description:\n{persona['system'].strip()}\n\n"
        f"Domain: {job.domain}\nSubtopic: {job.subtopic}\nRelevant frameworks: {fw}\n"
        f"Task type: {job.task_type} - {task_desc}\nDifficulty: {job.difficulty}\n"
        f"Variation number: {job.variant} (make this one distinct from other variations of the same subtopic)\n\n"
        "Write one conversation now."
    )


def generate(
    backend: ChatBackend,
    jobs: list[Job],
    taxonomy: dict,
    personas_cfg: dict,
    out_path: str | Path,
    dry_run: bool = False,
    max_tokens: int = 4096,
    log=print,
) -> dict[str, int]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {rec.get("id") for _, rec in load_jsonl(out_path)} if out_path.exists() else set()
    stats = {"written": 0, "skipped_existing": 0, "failed": 0}

    with open(out_path, "a", encoding="utf-8") as f:
        for i, job in enumerate(jobs, 1):
            if job.id in done:
                stats["skipped_existing"] += 1
                continue
            prompt = job_prompt(job, personas_cfg, taxonomy["task_types"][job.task_type])
            if dry_run:
                f.write(json.dumps({"id": job.id, "system": GENERATOR_SYSTEM, "prompt": prompt}) + "\n")
                stats["written"] += 1
                continue
            try:
                raw = backend.chat(GENERATOR_SYSTEM, [{"role": "user", "content": prompt}], max_tokens, OUTPUT_SCHEMA)
                obj = extract_json(raw)
                rec = {
                    "id": job.id,
                    "persona": job.persona,
                    "domain": job.domain,
                    "subtopic": job.subtopic,
                    "frameworks": list(job.frameworks),
                    "difficulty": job.difficulty,
                    "task_type": job.task_type,
                    "messages": [
                        {"role": "user", "content": str(obj["user"]).strip()},
                        {"role": "assistant", "content": str(obj["assistant"]).strip()},
                    ],
                    "source": "synthetic",
                    "reviewed": False,
                }
                errs = validate_record(rec)
                if errs:
                    raise ValueError(errs[0])
            except Exception as e:  # keep going; one bad sample shouldn't stop a long run
                stats["failed"] += 1
                log(f"[{i}/{len(jobs)}] {job.id} failed: {e}")
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            stats["written"] += 1
            log(f"[{i}/{len(jobs)}] {job.id} ok ({job.persona}/{job.task_type}/{job.difficulty})")
    return stats
