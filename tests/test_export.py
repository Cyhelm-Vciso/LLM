import json

from conftest import make_record

from cyberllm.export import build, load_personas, split_of, to_alpaca, to_sharegpt


def test_split_is_deterministic_and_roughly_proportional():
    assert split_of("abc") == split_of("abc")
    splits = [split_of(f"id-{i}") for i in range(2000)]
    assert 0.85 < splits.count("train") / 2000 < 0.95


def test_formats(tmp_path):
    personas = load_personas()
    rec = make_record()
    sg = to_sharegpt(rec, personas)
    assert [c["from"] for c in sg["conversations"]] == ["system", "human", "gpt"]
    al = to_alpaca(rec, personas)
    assert al["instruction"] == rec["messages"][0]["content"] and "system" in al
    multi = make_record()
    multi["messages"] += [{"role": "user", "content": "more?"}, {"role": "assistant", "content": "yes"}]
    assert to_alpaca(multi, personas) is None

    recs = [make_record(id=f"r{i}", persona=p) for i, p in enumerate(["ciso", "expert", "engineer"] * 20)]
    counts = build(recs, tmp_path, personas, ("messages", "sharegpt"), personas_filter=("ciso", "engineer"))
    total = sum(c["messages"] for c in counts.values())
    assert total == 40
    first = json.loads((tmp_path / "train.messages.jsonl").read_text().splitlines()[0])
    assert first["messages"][0]["role"] == "system"
