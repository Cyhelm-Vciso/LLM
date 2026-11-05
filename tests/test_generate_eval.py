import json

from cyberllm.cli import main
from cyberllm.evaluate import judge_answer, parse_letter, run_mcq, run_scenarios
from cyberllm.export import load_personas
from cyberllm.generate import generate, load_taxonomy, plan_jobs
from cyberllm.llm import extract_json


class FakeBackend:
    name = "fake"

    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def chat(self, system, messages, max_tokens=4096, json_schema=None):
        self.calls.append((system, messages))
        return self.fn(system, messages)


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Here you go:\n```json\n{"a": "b}"}\n```') == {"a": "b}"}
    assert extract_json('preamble {"x": {"y": 2}} trailing') == {"x": {"y": 2}}


def test_parse_letter():
    assert parse_letter("B") == "B"
    assert parse_letter("(c)") == "C"
    assert parse_letter("The answer is D because...") == "D"
    assert parse_letter("A. Recover") == "A"


def test_plan_jobs_balanced_and_deterministic():
    tax = load_taxonomy()
    a = plan_jobs(tax, 60, seed=1)
    b = plan_jobs(tax, 60, seed=1)
    assert [j.id for j in a] == [j.id for j in b]
    assert len({j.id for j in a}) == 60
    assert {j.persona for j in a} == {"ciso", "expert", "engineer"}
    only = plan_jobs(tax, 10, personas=("ciso",))
    assert {j.persona for j in only} == {"ciso"}


def test_generate_with_fake_teacher_and_resume(tmp_path):
    tax, personas = load_taxonomy(), load_personas()
    jobs = plan_jobs(tax, 5, seed=3)
    reply = json.dumps({"user": "How should we do X?", "assistant": "Do Y. " * 60})
    backend = FakeBackend(lambda s, m: reply)
    out = tmp_path / "syn.jsonl"
    stats = generate(backend, jobs, tax, personas, out, log=lambda *_: None)
    assert stats["written"] == 5
    stats2 = generate(backend, jobs, tax, personas, out, log=lambda *_: None)
    assert stats2 == {"written": 0, "skipped_existing": 5, "failed": 0}
    rec = json.loads(out.read_text().splitlines()[0])
    assert rec["source"] == "synthetic" and rec["reviewed"] is False


def test_generate_refuses_claude_teacher(capsys):
    try:
        main(["generate", "--teacher", "anthropic", "-n", "1"])
    except SystemExit as e:
        assert "terms" in str(e)
    else:
        raise AssertionError("expected SystemExit")


def test_mcq_and_scenarios_with_fakes(tmp_path):
    personas = load_personas()
    gold = {json.loads(l)["question"]: json.loads(l)["answer"] for l in open("evals/mcq.jsonl")}

    def answer_mcq(system, messages):
        text = messages[0]["content"]
        q = text.split("Question: ", 1)[1].split("\n\n", 1)[0]
        return gold[q]

    res = run_mcq(FakeBackend(answer_mcq), "evals/mcq.jsonl", personas, log=lambda *_: None)
    assert res["accuracy"] == 1.0

    def judge(system, messages):
        n = messages[0]["content"].split("<rubric>")[1].count("\n") - 1
        return json.dumps({"scores": [{"criterion": str(i), "score": 4, "reason": "ok"} for i in range(n)],
                           "summary": "fine"})

    target = FakeBackend(lambda s, m: "An answer.")
    sres = run_scenarios(target, FakeBackend(judge), "evals/scenarios.jsonl", personas, log=lambda *_: None)
    assert sres["ungraded"] == 0
    assert abs(sres["mean_score"] - 0.75) < 1e-9
    assert sres["safety_mean"] is not None


def test_cli_clean_and_build(tmp_path):
    clean = tmp_path / "clean.jsonl"
    assert main(["clean", "data/seed/*.jsonl", "--out", str(clean), "--review", str(tmp_path / "flag.jsonl"),
                 "--dropped-log", str(tmp_path / "dropped.tsv")]) == 0
    assert len(clean.read_text().splitlines()) == 11
    assert main(["build", str(clean), "--out-dir", str(tmp_path / "p"), "--format", "messages",
                 "--format", "sharegpt"]) == 0
    assert (tmp_path / "p" / "train.messages.jsonl").exists()
