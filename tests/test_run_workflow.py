"""The individual CLI delegates to each workflow's existing owned runner."""

import sys
from types import SimpleNamespace

import pytest

from scripts import run_workflow


@pytest.mark.parametrize("workflow", ["generate_scenarios", "simulate_system"])
def test_dispatches_owned_runner(monkeypatch, workflow):
    calls = []

    def owned_main(argv):
        calls.append(argv)
        return 7

    monkeypatch.setitem(
        sys.modules, f"scripts.{workflow}", SimpleNamespace(main=owned_main)
    )
    arguments = [
        workflow,
        "--config",
        "case.yml",
        "--project-dir",
        "case-output",
        "--cores",
        "2",
        "--dry-run",
        "--keep-going",
    ]
    if workflow == "simulate_system":
        arguments += ["--target", "indicators"]
    arguments += ["--", "--rerun-incomplete"]
    assert run_workflow.main(arguments) == 7
    forwarded = calls[0]
    assert forwarded[:6] == [
        "--config",
        "case.yml",
        "--project-dir",
        "case-output",
        "--cores",
        "2",
    ]
    if workflow == "simulate_system":
        assert forwarded[6:8] == ["--target", "indicators"]
        assert forwarded[8:] == [
            "--dry-run",
            "--",
            "--keep-going",
            "--rerun-incomplete",
        ]
    else:
        assert forwarded[6:] == [
            "--",
            "--dry-run",
            "--keep-going",
            "--rerun-incomplete",
        ]


@pytest.mark.parametrize(
    "workflow", ["analyze_climate", "build_model", "analyze_projections"]
)
def test_existing_archive_dispatch_is_preserved(monkeypatch, workflow):
    calls = []
    monkeypatch.setattr(
        run_workflow,
        "run_workflow",
        lambda *args, **kwargs: calls.append((args, kwargs)) or 0,
    )
    assert (
        run_workflow.main(
            [workflow, "--config", "case.yml", "--project-dir", "case-output"]
        )
        == 0
    )
    assert calls[0][0][0] == workflow
    assert calls[0][1] == {
        "cores": 3,
        "targets": ["all"],
        "dry_run": False,
        "keep_going": False,
        "extra": [],
    }


def test_generation_rejects_unsupported_target_before_dispatch(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "scripts.generate_scenarios",
        SimpleNamespace(main=lambda _: pytest.fail("unsupported target dispatched")),
    )
    with pytest.raises(SystemExit) as error:
        run_workflow.main(
            [
                "generate_scenarios",
                "--config",
                "case.yml",
                "--project-dir",
                "case-output",
                "--target",
                "indicators",
            ]
        )
    assert error.value.code == 2
