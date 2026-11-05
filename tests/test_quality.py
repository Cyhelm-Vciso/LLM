from conftest import make_record

from cyberllm.quality import near_duplicates, run_quality, safety_hits


def test_drops_short_and_boilerplate():
    recs = [
        make_record(id="a", assistant="Too short."),
        make_record(id="b", user="q2", assistant="As an AI language model, I think " + "x " * 200),
        make_record(id="c", user="q3"),
    ]
    rep = run_quality(recs)
    assert [r["id"] for r in rep.kept] == ["c"]
    reasons = dict(rep.dropped)
    assert reasons["a"].startswith("too-short")
    assert reasons["b"].startswith("boilerplate")


def test_exact_and_near_duplicates():
    base = "Explain how to rotate AWS access keys safely across forty accounts using automation and alerts"
    long_answer = ("Start by pulling the IAM credential report in every account to inventory active keys and "
                   "their last-used dates. Create a second key for each workload, deploy it through your secret "
                   "manager, and confirm the application authenticates with the new key. Watch CloudTrail for any "
                   "remaining calls signed by the old key over a full business cycle, then deactivate it, wait "
                   "another week for surprises, and delete it. Automate the whole flow with a scheduled pipeline "
                   "and raise an alert whenever a key older than ninety days is still active anywhere.")
    recs = [
        make_record(id="a", user=base, assistant=long_answer),
        make_record(id="b", user=base, assistant=long_answer),          # exact duplicate prompt
        make_record(id="c", user=base + " please", assistant=long_answer + " Done."),  # near duplicate
        make_record(id="d", user="Something entirely different about Kubernetes RBAC design"),
    ]
    rep = run_quality(recs)
    kept = {r["id"] for r in rep.kept}
    assert kept == {"a", "d"}


def test_near_duplicates_function():
    texts = ["the quick brown fox jumps over the lazy dog again and again today",
             "the quick brown fox jumps over the lazy dog again and again today!",
             "completely unrelated sentence about detection engineering pipelines in ci"]
    assert near_duplicates(texts) == {1}


def test_safety_lint_flags_but_allows_boundary():
    bad = "Run this: " + "\\x90" * 20 + " then mimikatz sekurlsa::logonpasswords " + "pad " * 80
    rep = run_quality([make_record(id="x", assistant=bad),
                       make_record(id="y", user="other", assistant=bad, task_type="boundary")])
    assert [r["id"] for r in rep.flagged] == ["x"]
    assert set(rep.flagged[0]["_flags"]) >= {"shellcode-bytes", "credential-dump-cmd"}
    assert [r["id"] for r in rep.kept] == ["y"]
    assert safety_hits("normal defensive advice") == []
