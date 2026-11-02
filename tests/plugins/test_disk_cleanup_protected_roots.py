"""Configured durable roots use the real config loader and temporary filesystem."""

import pytest

from tests.plugins.test_disk_cleanup_plugin import _load_lib, _load_plugin_init, _run_tool


@pytest.fixture
def home(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    return root


def configure(home, roots):
    import hermes_yaml as yaml

    (home / "config.yaml").write_text(yaml.safe_dump({
        "disk_cleanup": {"protected_roots": roots},
    }))


@pytest.mark.parametrize("root", ["workspaces", "scripts", "data"])
def test_configured_root_is_not_automatically_tracked(home, root):
    configure(home, [root])
    persistent = home / root / "nested" / "test_persistent.py"
    persistent.parent.mkdir(parents=True)
    plugin = _load_plugin_init()
    _run_tool(plugin, "write_file", {"path": str(persistent), "content": "keep"},
              create=lambda: persistent.write_text("keep"), task_id="t", session_id="s")
    cleanup = _load_lib()
    assert cleanup.guess_category(persistent) is None
    assert cleanup.load_tracked() == []
    scratch = home / "scratch" / "test_disposable.py"
    scratch.parent.mkdir()
    scratch.write_text("scratch")
    assert cleanup.guess_category(scratch) == "test"
    plugin._on_session_end(session_id="s", completed=True, interrupted=False)
    assert persistent.read_text() == "keep"


@pytest.mark.parametrize("root", ["workspaces", "scripts", "data"])
def test_configured_root_is_not_empty_swept(home, root):
    configure(home, [root])
    nested = home / root / "nested" / "empty"
    nested.mkdir(parents=True)
    scratch = home / "scratch" / "empty"
    scratch.mkdir(parents=True)
    result = _load_lib().quick()
    assert nested.is_dir()
    assert not scratch.parent.exists()
    assert result["empty_dirs"] == 2


def test_tracked_configured_directory_is_not_recursively_deleted(home):
    configure(home, ["workspaces"])
    directory = home / "workspaces" / "nested"
    directory.mkdir(parents=True)
    persistent = directory / "keep.txt"
    persistent.write_text("keep")
    cleanup = _load_lib()
    assert cleanup.track(str(directory), "temp", silent=True)
    entries = cleanup.load_tracked()
    entries[0]["timestamp"] = "2000-01-01T00:00:00+00:00"
    cleanup.save_tracked(entries)
    assert cleanup.dry_run() == ([], [])
    assert cleanup.quick()["deleted"] == 0
    assert persistent.read_text() == "keep"
    assert cleanup.load_tracked() == []


@pytest.mark.parametrize("root", ["workspaces", "scripts", "data"])
def test_stale_tracked_test_is_revalidated_before_preview_or_cleanup(home, root):
    configure(home, [root])
    persistent = home / root / "test_persistent.py"
    persistent.parent.mkdir()
    persistent.write_text("keep")
    cleanup = _load_lib()
    assert cleanup.track(str(persistent), "test", silent=True)
    assert cleanup.dry_run() == ([], [])
    assert cleanup.quick()["deleted"] == 0
    assert persistent.read_text() == "keep"
    assert cleanup.load_tracked() == []


@pytest.mark.parametrize("roots", [
    [" workspaces ", "scripts", 12, None, "", " ", ".", "..", "scratch/nested", "scratch\\nested"],
    {"workspaces", "scripts"},
])
def test_valid_names_are_trimmed_and_invalid_entries_ignored(home, roots):
    configure(home, roots)
    cleanup = _load_lib()
    for root in ("workspaces", "scripts"):
        persistent = home / root / "test_persistent.py"
        persistent.parent.mkdir()
        persistent.write_text("keep")
        assert cleanup.guess_category(persistent) is None
    scratch = home / "scratch" / "nested" / "test_disposable.py"
    scratch.parent.mkdir(parents=True)
    scratch.write_text("scratch")
    assert cleanup.guess_category(scratch) == "test"
    empty = home / "scratch" / "empty"
    empty.mkdir()
    cleanup.quick()
    assert not empty.exists()


@pytest.mark.parametrize("section", [None, [], "workspaces", 7,
    {"protected_roots": "workspaces"}, {"protected_roots": None},
    {"protected_roots": {"workspaces": True}}, {"protected_roots": 7}])
def test_invalid_configuration_retains_builtin_protection(home, section):
    import hermes_yaml as yaml

    (home / "config.yaml").write_text(yaml.safe_dump({"disk_cleanup": section}))
    cleanup = _load_lib()
    builtin = home / "workspace" / "empty"
    builtin.mkdir(parents=True)
    persistent = builtin.parent / "test_persistent.py"
    persistent.write_text("keep")
    scratch = home / "workspaces" / "test_disposable.py"
    scratch.parent.mkdir()
    scratch.write_text("scratch")
    assert cleanup.guess_category(persistent) is None
    assert cleanup.guess_category(scratch) == "test"
    cleanup.quick()
    assert builtin.is_dir()


def test_config_load_failure_retains_builtin_protection(home, monkeypatch):
    from hermes_cli import config_effective

    def fail_load():
        raise OSError("unreadable fixture config")

    monkeypatch.setattr(config_effective, "load_user_config_effective", fail_load)
    cleanup = _load_lib()
    persistent = home / "workspace" / "test_persistent.py"
    persistent.parent.mkdir()
    persistent.write_text("keep")
    assert cleanup.guess_category(persistent) is None
    cleanup.quick()
    assert persistent.exists()


def test_protection_follows_profile_scope_and_config_changes(home, tmp_path):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    other = tmp_path / "profiles" / "other"
    other.mkdir(parents=True)
    configure(home, ["workspaces"])
    configure(other, ["scripts"])
    cleanup = _load_lib()
    for current, protected, scratch in [(home, "workspaces", "scripts"),
                                         (other, "scripts", "workspaces"),
                                         (home, "workspaces", "scripts")]:
        token = set_hermes_home_override(current)
        try:
            for root, expected in [(protected, None), (scratch, "test")]:
                path = current / root / "test_persistent.py"
                path.parent.mkdir(exist_ok=True)
                path.write_text("keep")
                assert cleanup.guess_category(path) == expected
            empty = current / protected / "empty"
            empty.mkdir(exist_ok=True)
            cleanup.quick()
            assert empty.is_dir()
        finally:
            reset_hermes_home_override(token)
    configure(home, ["scripts"])
    assert cleanup.guess_category(home / "workspaces" / "test_persistent.py") == "test"
    assert cleanup.guess_category(home / "scripts" / "test_persistent.py") is None


def test_default_config_registers_empty_protected_roots(home):
    from hermes_cli.config import load_config_readonly

    assert load_config_readonly()["disk_cleanup"]["protected_roots"] == []


def test_explicit_temp_file_still_ages_out_in_configured_root(home):
    configure(home, ["data"])
    disposable = home / "data" / "old.bin"
    disposable.parent.mkdir()
    disposable.write_text("disposable")
    cleanup = _load_lib()
    assert cleanup.track(str(disposable), "temp", silent=True)
    entries = cleanup.load_tracked()
    entries[0]["timestamp"] = "2000-01-01T00:00:00+00:00"
    cleanup.save_tracked(entries)
    assert cleanup.dry_run()[0] == entries
    assert cleanup.quick()["deleted"] == 1
    assert not disposable.exists()
    assert disposable.parent.is_dir()
