import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _cwd_repo_root():
    old = os.getcwd()
    os.chdir(ROOT)
    yield
    os.chdir(old)


def make_record(id="t-1", persona="engineer", user="How do I harden SSH on Ubuntu servers?", assistant=None, **kw):
    assistant = assistant or ("Disable password authentication and root login, use key-based auth, "
                              "restrict AllowUsers, enable fail2ban, and log to a central SIEM. " * 3)
    rec = {"id": id, "persona": persona, "domain": "infrastructure-identity", "difficulty": "foundational",
           "task_type": "write", "source": "synthetic", "reviewed": False,
           "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}
    rec.update(kw)
    return rec
