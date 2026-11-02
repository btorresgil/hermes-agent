"""Selection-time accounting must survive reload, without a later pool mutation."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def auth_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _row(index):
    return {
        "id": f"entry-{index}", "label": f"synthetic-{index}",
        "auth_type": "api_key", "source": "manual", "priority": index,
        "access_token": f"synthetic-access-{index}", "request_count": 3 + index,
    }


def _write_store(home, rows, provider="openrouter"):
    store = {
        "version": 1, "active_provider": "unrelated",
        "providers": {"unrelated": {"marker": "preserve"}},
        "credential_pool": {provider: rows},
    }
    (home / "auth.json").write_text(json.dumps(store))
    return store


def _read_store(home):
    return json.loads((home / "auth.json").read_text())


@pytest.mark.parametrize("strategy", ["least_used", "fill_first", "random", "round_robin"])
@pytest.mark.parametrize("size", [1, 2])
def test_counted_selection_is_durable_before_other_mutations(auth_home, strategy, size):
    from agent.credential_pool import CredentialPool, PooledCredential
    from hermes_cli.auth import read_credential_pool

    rows = [_row(index) for index in range(size)]
    original = _write_store(auth_home, rows)
    pool = CredentialPool("openrouter", [PooledCredential.from_dict("openrouter", row) for row in rows])
    pool._strategy = strategy

    selected = pool.select()
    assert selected is not None

    expected = next(row["request_count"] for row in rows if row["id"] == selected.id) + 1
    disk = _read_store(auth_home)
    assert next(row for row in disk["credential_pool"]["openrouter"] if row["id"] == selected.id)["request_count"] == expected
    reloaded = CredentialPool("openrouter", [
        PooledCredential.from_dict("openrouter", row) for row in read_credential_pool("openrouter")
    ])
    assert next(entry for entry in reloaded.entries() if entry.id == selected.id).request_count == expected
    assert disk["providers"] == original["providers"]
    assert disk["active_provider"] == original["active_provider"]


@pytest.mark.parametrize("strategy", ["least_used", "fill_first", "random", "round_robin"])
@pytest.mark.parametrize("size", [1, 2])
def test_counted_selection_persists_once(auth_home, monkeypatch, strategy, size):
    from agent.credential_pool import load_pool

    _write_store(auth_home, [_row(index) for index in range(size)])
    pool = load_pool("openrouter")
    pool._strategy = strategy
    calls = []
    persist = pool._persist

    def record_persist(**kwargs):
        calls.append(kwargs)
        return persist(**kwargs)

    monkeypatch.setattr(pool, "_persist", record_persist)
    assert pool.select() is not None
    assert len(calls) == 1


@pytest.mark.parametrize("strategy", ["least_used", "fill_first", "random", "round_robin"])
@pytest.mark.parametrize("operation", ["peek", "reclaim", "count_false"])
def test_noncounting_operations_preserve_disk_counts(auth_home, strategy, operation):
    from agent.credential_pool import load_pool

    _write_store(auth_home, [_row(0), _row(1)])
    pool = load_pool("openrouter")
    pool._strategy = strategy
    before = (auth_home / "auth.json").read_bytes()
    if operation == "count_false":
        with pool._lock:
            selected, _pending = pool._select_unlocked(refresh=False, count=False)
    elif operation == "reclaim":
        selected = pool.reclaim("entry-0")
    else:
        selected = pool.peek()
    assert selected is not None
    assert {entry.id: entry.request_count for entry in pool.entries()} == {"entry-0": 3, "entry-1": 4}
    rows = _read_store(auth_home)["credential_pool"]["openrouter"]
    assert {row["id"]: row["request_count"] for row in rows} == {"entry-0": 3, "entry-1": 4}
    if operation == "count_false" and strategy == "round_robin":
        # Forced-refresh lookup historically rotates multi-entry round-robin;
        # preserve that write, but never count it as a request.
        assert [row["id"] for row in rows] == ["entry-1", "entry-0"]
    else:
        assert (auth_home / "auth.json").read_bytes() == before


@pytest.mark.parametrize("strategy", ["least_used", "fill_first", "random", "round_robin"])
def test_two_profiles_durably_count_only_in_root(auth_home, monkeypatch, strategy):
    from agent.credential_pool import load_pool

    root = auth_home.parent / ".hermes"
    root.mkdir()
    row = dict(_row(0), auth_type="oauth", source="manual:device_code",
               refresh_token="synthetic-refresh")
    original = _write_store(root, [row], provider="openai-codex")
    for name, expected in [("coder", 4), ("reviewer", 5)]:
        profile = root / "profiles" / name
        profile.mkdir(parents=True)
        _write_store(profile, [_row(9)])
        before = (profile / "auth.json").read_bytes()
        monkeypatch.setenv("HERMES_HOME", str(profile))
        pool = load_pool("openai-codex")
        pool._strategy = strategy
        assert pool.select().request_count == expected
        disk = _read_store(root)
        assert disk["credential_pool"]["openai-codex"][0]["request_count"] == expected
        assert load_pool("openai-codex").entries()[0].request_count == expected
        assert (profile / "auth.json").read_bytes() == before
        assert disk["providers"] == original["providers"]
        assert disk["active_provider"] == original["active_provider"]


@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("strategy", ["least_used", "round_robin"])
def test_selection_adopts_peer_generation(auth_home, monkeypatch, borrowed, strategy):
    from agent.credential_pool import load_pool

    owner = auth_home
    if borrowed:
        owner = auth_home.parent / ".hermes"
        owner.mkdir()
    rows = [dict(_row(index), auth_type="oauth", source="manual:device_code",
                 refresh_token=f"synthetic-refresh-{index}") for index in range(2)]
    _write_store(owner, rows, provider="openai-codex")
    monkeypatch.setenv("HERMES_HOME", str(owner))
    if borrowed:
        profile = owner / "profiles" / "coder"
        profile.mkdir(parents=True)
        _write_store(profile, [_row(9)])
        before = (profile / "auth.json").read_bytes()
        monkeypatch.setenv("HERMES_HOME", str(profile))
    pool = load_pool("openai-codex")
    pool._strategy = strategy
    # A real peer pool commits a new generation after this pool was loaded.
    # Its token baseline proves that the new pair replaces the disk pair.
    peer_pool = load_pool("openai-codex")
    peer = peer_pool._adopt(
        peer_pool.entries()[0], access_token="synthetic-peer-access",
        refresh_token="synthetic-peer-refresh", last_status="ok", status_cleared_at=2,
    )
    selected = pool.select()
    assert selected is not None
    assert selected.id == peer.id
    assert (selected.access_token, selected.refresh_token) == (peer.access_token, peer.refresh_token)
    disk = _read_store(owner)["credential_pool"]["openai-codex"]
    written = next(row for row in disk if row["id"] == selected.id)
    assert written["request_count"] == 4
    assert written["last_status"] != "exhausted"
    assert written["refresh_token"] == peer.refresh_token
    if borrowed:
        assert (profile / "auth.json").read_bytes() == before
