import glob

from conftest import make_record

from cyberllm.schema import validate_files, validate_record


def test_valid_record():
    assert validate_record(make_record()) == []


def test_rejects_bad_persona_and_roles():
    rec = make_record(persona="hacker")
    rec["messages"] = [{"role": "assistant", "content": "x"}, {"role": "user", "content": "y"}]
    errs = validate_record(rec)
    assert any("persona" in e for e in errs)
    assert any("alternate" in e for e in errs)
    assert any("last message" in e for e in errs)


def test_rejects_system_role_in_record():
    rec = make_record()
    rec["messages"].insert(0, {"role": "system", "content": "x"})
    assert any("system is added at export" in e for e in validate_record(rec))


def test_seed_and_eval_files_are_valid():
    count, problems = validate_files(sorted(glob.glob("data/seed/*.jsonl")))
    assert problems == []
    assert count >= 9
