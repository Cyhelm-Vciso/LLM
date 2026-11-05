"""Command-line interface: cyberllm <command> [options]."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

from . import __version__


def _expand(patterns: list[str]) -> list[str]:
    files: list[str] = []
    for p in patterns:
        matches = sorted(glob.glob(p))
        if not matches and Path(p).exists():
            matches = [p]
        if not matches:
            raise SystemExit(f"no files match: {p}")
        files.extend(matches)
    return files


def cmd_validate(a: argparse.Namespace) -> int:
    from .schema import validate_files

    count, problems = validate_files(_expand(a.files))
    for p in problems:
        print(p)
    print(f"{count} records, {len(problems)} problems")
    return 1 if problems else 0


def cmd_generate(a: argparse.Namespace) -> int:
    from .export import load_personas
    from .generate import generate, load_taxonomy, plan_jobs
    from .llm import make_backend

    if a.teacher.startswith("anthropic"):
        raise SystemExit(
            "Refusing to use Claude as a data-generation teacher: Anthropic's terms restrict using outputs "
            "to train other AI models. Use an open-weight model whose licence permits this "
            "(see README: 'Choosing a teacher model'). Claude can still be used as an eval judge."
        )
    taxonomy = load_taxonomy(a.taxonomy)
    personas = load_personas(a.personas)
    jobs = plan_jobs(taxonomy, a.n, tuple(a.persona) if a.persona else None, a.seed, a.boundary_share)
    backend = None if a.dry_run else make_backend(a.teacher, a.base_url, a.api_key, a.temperature)
    stats = generate(backend, jobs, taxonomy, personas, a.out, dry_run=a.dry_run, max_tokens=a.max_tokens)
    print(stats)
    return 0


def cmd_clean(a: argparse.Namespace) -> int:
    from .quality import run_quality
    from .schema import read_records, write_jsonl

    records = read_records(_expand(a.inputs))
    rep = run_quality(records, a.dedupe_threshold)
    write_jsonl(a.out, rep.kept)
    write_jsonl(a.review, rep.flagged)
    if a.dropped_log:
        Path(a.dropped_log).parent.mkdir(parents=True, exist_ok=True)
        Path(a.dropped_log).write_text("".join(f"{i}\t{r}\n" for i, r in rep.dropped), encoding="utf-8")
    print(rep.summary())
    print(f"kept -> {a.out}\nflagged for human review -> {a.review}")
    return 0


def cmd_build(a: argparse.Namespace) -> int:
    from .export import build, load_personas
    from .schema import read_records

    records = read_records(_expand(a.inputs))
    if a.reviewed_only:
        records = [r for r in records if r.get("reviewed") or r.get("source") in ("seed", "human")]
    counts = build(
        records, a.out_dir, load_personas(a.personas), tuple(a.format), a.val_pct, a.test_pct,
        include_system=not a.no_system, personas_filter=tuple(a.persona) if a.persona else None,
    )
    for split, c in counts.items():
        print(f"{split:5s} " + " ".join(f"{fmt}={n}" for fmt, n in c.items()))
    print(f"-> {a.out_dir}")
    return 0


def cmd_train(a: argparse.Namespace) -> int:
    import yaml

    from .train import load_config, train

    cfg = load_config(a.config)
    for kv in a.set or []:
        k, _, v = kv.partition("=")
        cfg[k] = yaml.safe_load(v)  # "false" -> False, "2e-4" -> str, "3" -> 3
    print(f"saved adapter -> {train(cfg)}")
    return 0


def cmd_merge(a: argparse.Namespace) -> int:
    from .train import merge

    print(f"merged model -> {merge(a.adapter, a.out)}")
    return 0


def cmd_eval(a: argparse.Namespace) -> int:
    from .evaluate import run_mcq, run_scenarios, write_report
    from .export import load_personas
    from .llm import make_backend

    personas = load_personas(a.personas)
    target = make_backend(a.target, a.base_url, a.api_key, temperature=0.0)
    report: dict = {"target": a.target}
    if a.mcq:
        report["mcq"] = run_mcq(target, a.mcq, personas, not a.no_system)
    if a.scenarios:
        judge = make_backend(a.judge, a.judge_base_url, a.judge_api_key, temperature=0.0)
        report["judge"] = a.judge
        report["scenarios"] = run_scenarios(target, judge, a.scenarios, personas, not a.no_system)
    jpath, mpath = write_report(report, a.out_dir, a.name)
    print(mpath.read_text(encoding="utf-8"))
    print(f"full results -> {jpath}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cyberllm", description=__doc__)
    p.add_argument("--version", action="version", version=f"cyberllm {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("validate", help="validate record JSONL files against the schema")
    s.add_argument("files", nargs="+")
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("generate", help="generate synthetic conversations with a teacher model")
    s.add_argument("--teacher", required=True, help="openai:<model> (OpenAI-compatible server, e.g. vLLM/Ollama)")
    s.add_argument("--base-url", help="e.g. http://localhost:11434/v1 (Ollama) or http://localhost:8000/v1 (vLLM)")
    s.add_argument("--api-key", help="bearer token if the server needs one (or OPENAI_COMPAT_API_KEY)")
    s.add_argument("-n", type=int, default=100, help="number of conversations to plan")
    s.add_argument("--persona", action="append", choices=["ciso", "expert", "engineer"])
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--boundary-share", type=float, default=0.08, help="fraction of safety-boundary examples")
    s.add_argument("--temperature", type=float, default=0.8)
    s.add_argument("--max-tokens", type=int, default=4096)
    s.add_argument("--taxonomy", default="configs/taxonomy.yaml")
    s.add_argument("--personas", default="configs/personas.yaml")
    s.add_argument("--out", default="data/generated/synthetic.jsonl")
    s.add_argument("--dry-run", action="store_true", help="write the generation prompts instead of calling a model")
    s.set_defaults(func=cmd_generate)

    s = sub.add_parser("clean", help="schema/length/boilerplate/dedupe/safety gates")
    s.add_argument("inputs", nargs="+")
    s.add_argument("--out", default="data/processed/clean.jsonl")
    s.add_argument("--review", default="data/review/flagged.jsonl")
    s.add_argument("--dropped-log", default="data/review/dropped.tsv")
    s.add_argument("--dedupe-threshold", type=float, default=0.8)
    s.set_defaults(func=cmd_clean)

    s = sub.add_parser("build", help="split and export to fine-tuning formats")
    s.add_argument("inputs", nargs="+")
    s.add_argument("--out-dir", default="data/processed")
    s.add_argument("--format", action="append", choices=["messages", "sharegpt", "alpaca"])
    s.add_argument("--persona", action="append", choices=["ciso", "expert", "engineer"],
                   help="only include these personas (repeatable); default all three")
    s.add_argument("--val-pct", type=int, default=5)
    s.add_argument("--test-pct", type=int, default=5)
    s.add_argument("--no-system", action="store_true", help="omit persona system prompts")
    s.add_argument("--reviewed-only", action="store_true", help="drop synthetic records not marked reviewed")
    s.add_argument("--personas", default="configs/personas.yaml")
    s.set_defaults(func=cmd_build)

    s = sub.add_parser("train", help="LoRA/QLoRA fine-tune with Hugging Face TRL")
    s.add_argument("--config", default="configs/train_lora.yaml")
    s.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a top-level config value")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("merge", help="merge a LoRA adapter into its base model")
    s.add_argument("--adapter", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_merge)

    s = sub.add_parser("eval", help="evaluate a model on MCQ and judged scenarios")
    s.add_argument("--target", required=True, help="openai:<model> or anthropic[:<model>]")
    s.add_argument("--base-url")
    s.add_argument("--api-key")
    s.add_argument("--judge", default="anthropic", help="anthropic[:<model>] or openai:<model>")
    s.add_argument("--judge-base-url")
    s.add_argument("--judge-api-key")
    s.add_argument("--mcq", default="evals/mcq.jsonl")
    s.add_argument("--scenarios", default="evals/scenarios.jsonl")
    s.add_argument("--no-system", action="store_true", help="don't send persona system prompts to the target")
    s.add_argument("--personas", default="configs/personas.yaml")
    s.add_argument("--out-dir", default="reports")
    s.add_argument("--name", default="eval")
    s.set_defaults(func=cmd_eval)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "format", "unset") is None:
        args.format = ["messages"]
    if args.cmd == "eval" and args.mcq in ("", "none"):
        args.mcq = None
    if args.cmd == "eval" and args.scenarios in ("", "none"):
        args.scenarios = None
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
