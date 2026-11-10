"""Guards against borrowing incompatible dependencies or executing in the owner."""

from pathlib import Path

import pytest

from dev.scripts import session_runtime as runtime


@pytest.fixture
def roots(tmp_path):
    source, owner = tmp_path / "source", tmp_path / "owner"
    for root in (source, owner):
        for relative in runtime.CONTRACT_FILES:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('value = "same"\n', encoding="utf-8")
        prefix = root / ".pixi/envs/default"
        (prefix / "conda-meta").mkdir(parents=True)
        (prefix / "conda-meta/pixi").write_text("{}", encoding="utf-8")
        (prefix / "Library/bin").mkdir(parents=True)
        for dll in ("ffi.dll", "cairo.dll"):
            (prefix / "Library/bin" / dll).touch()
    return source, owner


@pytest.mark.parametrize("relative", runtime.CONTRACT_FILES)
def test_incompatible_contract_never_invokes_pixi(roots, relative, monkeypatch):
    source, owner = roots
    (source / relative).write_text('value = "changed"\n', encoding="utf-8")
    calls = []
    monkeypatch.setattr(runtime.subprocess, "call", lambda *a, **k: calls.append(a))
    assert (
        runtime.main(
            ["--source-root", str(source), "--runtime-root", str(owner), "--", "python"]
        )
        == 1
    )
    assert calls == []


def test_compatible_command_keeps_source_cwd_and_exit_status(roots, monkeypatch):
    source, owner = roots
    (source / "pixi.toml").write_bytes(b'# comment\r\nvalue = "same"\r\n')
    calls = []

    def execute(command, **kwargs):
        calls.append((command, kwargs))
        return 7

    monkeypatch.setattr(runtime.subprocess, "call", execute)
    assert (
        runtime.main(
            ["--source-root", str(source), "--runtime-root", str(owner), "--", "python"]
        )
        == 7
    )
    command, kwargs = calls.pop()
    assert kwargs["cwd"] == source
    assert command[:4] == ["pixi", "run", "--as-is", "--executable"]
    assert command[command.index("--manifest-path") + 1] == str(owner / "pixi.toml")
    assert command[command.index("--source-root") + 1] == str(source)


@pytest.mark.parametrize(
    "scope",
    [
        "tasks",
        "target.win-64.tasks",
        "feature.dev.tasks",
        "feature.dev.target.win-64.tasks",
    ],
)
@pytest.mark.parametrize("description", ['description = "New wording"', ""])
def test_task_description_changes_are_compatible(roots, scope, description):
    source, owner = roots
    task = f'[{scope}.check]\ncmd = "python check.py"\n'
    (owner / "pixi.toml").write_text(
        task + 'description = "Old wording"\n', encoding="utf-8"
    )
    (source / "pixi.toml").write_text(task + description + "\n", encoding="utf-8")
    runtime.check_compatibility(source, owner)
    runtime.check_compatibility(owner, source)


@pytest.mark.parametrize(
    "table, before, after",
    [
        ("tasks.check", 'cmd = "python old.py"', 'cmd = "python new.py"'),
        ("tasks.check.env", 'MODE = "old"', 'MODE = "new"'),
        ("dependencies", 'python = "3.12.*"', 'python = "3.13.*"'),
        ("activation.env", 'MODE = "old"', 'MODE = "new"'),
        ("feature.dev.dependencies", 'python = "3.12.*"', 'python = "3.13.*"'),
        ("target.win-64.activation", 'scripts = ["old.bat"]', 'scripts = ["new.bat"]'),
    ],
)
def test_runtime_settings_remain_strict(roots, table, before, after):
    source, owner = roots
    (owner / "pixi.toml").write_text(f"[{table}]\n{before}\n", encoding="utf-8")
    (source / "pixi.toml").write_text(f"[{table}]\n{after}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pixi.toml differs"):
        runtime.check_compatibility(source, owner)


def test_julia_task_description_is_not_exempt(roots):
    source, owner = roots
    (source / "Project.toml").write_text(
        '[tasks.check]\ndescription = "changed"\n', encoding="utf-8"
    )
    (owner / "Project.toml").write_text("[tasks.check]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Project.toml differs"):
        runtime.check_compatibility(source, owner)


def test_session_caches_and_context_replace_ambient_checkout(roots, monkeypatch):
    source, owner = roots
    monkeypatch.setenv("PYTHONPATH", str(owner))
    monkeypatch.setenv("JULIA_PROJECT", str(owner))
    monkeypatch.setenv("R_LIBS", str(owner))
    monkeypatch.delenv("JULIA_DEPOT_PATH", raising=False)
    env = runtime.command_environment(source)
    assert env["PYTHONPATH"] == str(source)
    assert env["JULIA_PROJECT"] == str(source)
    assert "R_LIBS" not in env
    for variable in ("MPLCONFIGDIR", "XDG_CACHE_HOME", "RUFF_CACHE_DIR"):
        assert Path(env[variable]).is_relative_to(source / ".tmp/session-runtime")
    assert env["PIXI_NO_INSTALL"] == "true"
    assert env["PIXI_FROZEN"] == "true"
    assert str(Path.home() / ".julia") in env["JULIA_DEPOT_PATH"]


def test_activation_cannot_bypass_interpreter_check(roots, capsys):
    source, owner = roots
    assert (
        runtime.main(
            [
                "--source-root",
                str(source),
                "--runtime-root",
                str(owner),
                "--activated",
                "--",
                "python",
            ]
        )
        == 1
    )
    assert "Wrong interpreter prefix" in capsys.readouterr().err


def test_activation_normalizes_inherited_pixi_booleans(roots, monkeypatch):
    source, owner = roots
    monkeypatch.setenv("PIXI_NO_INSTALL", "1")
    monkeypatch.setenv("PIXI_FROZEN", "1")
    calls = []

    def activate(command, **kwargs):
        calls.append((command, kwargs))
        return 0

    monkeypatch.setattr(runtime.subprocess, "call", activate)
    assert (
        runtime.main(
            ["--source-root", str(source), "--runtime-root", str(owner), "--", "python"]
        )
        == 0
    )
    assert len(calls) == 1
    assert calls[0][1]["env"]["PIXI_NO_INSTALL"] == "true"
    assert calls[0][1]["env"]["PIXI_FROZEN"] == "true"


def test_wrong_checkout_import_never_runs_requested_check(roots, monkeypatch, capsys):
    source, owner = roots
    monkeypatch.setattr(runtime.sys, "prefix", str(owner / ".pixi/envs/default"))
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *a, **k: runtime.subprocess.CompletedProcess(
            [], 0, stdout=str(owner / "blueearth_cst/__init__.py")
        ),
    )
    calls = []
    monkeypatch.setattr(runtime.subprocess, "call", lambda *a, **k: calls.append(a))
    assert (
        runtime.main(
            [
                "--source-root",
                str(source),
                "--runtime-root",
                str(owner),
                "--activated",
                "--",
                "python",
            ]
        )
        == 1
    )
    assert calls == []
    assert "Wrong Python source" in capsys.readouterr().err
