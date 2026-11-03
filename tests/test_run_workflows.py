"""Contract tests for scripts/run_workflows.py (design §7(g) + ext1-03).

The six §7(g) assertions plus the ext1-03 enabled:false skip test. Every test
monkeypatches subprocess.run to capture the argv list -- no real snakemake runs.
"""

import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from blueearth_cst.experiment import simulation_record

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import run_workflows as rw  # noqa: E402


class FakeResult:
    def __init__(self, returncode, stdout=""):
        self.returncode = returncode
        self.stdout = stdout


@pytest.fixture(autouse=True)
def wrapper_only_admission(monkeypatch):
    """Keep these minimal orchestration fixtures outside scientific admission."""
    from types import SimpleNamespace

    from blueearth_cst.experiment import reservation_context

    monkeypatch.setattr(rw, "_console_width", lambda: 240)

    monkeypatch.setattr(
        reservation_context,
        "admit_context_owned",
        lambda *_args, **_kwargs: SimpleNamespace(
            name="test",
            action="continue",
            reservation_id=None,
            recheck=lambda: None,
            bind_selection=lambda: None,
            environment=lambda: {},
        ),
    )
    # The stubbed WF4 child publishes no indicator set; real named publication is
    # exercised by test_named_indicator_sets_workflow.
    from blueearth_cst.experiment import indicator_export

    monkeypatch.setattr(indicator_export, "publish_selected_sets", lambda *_args: [])


def _write_cfg(path, flags, project_dir=None):
    """Write a full-orchestration config with the given enabled flags dict.

    ``flags`` values are inserted verbatim into YAML so a caller can pass a raw
    string (e.g. '"true"' or 'yes') to exercise the parsed-value contract."""
    if project_dir is None:
        project_dir = path.parent / "project"
    lines = [
        "project:",
        f"  project_dir: {project_dir}",
        "workflows:",
    ]
    for name in rw.WORKFLOW_ORDER:
        lines.append(f"  {name}:")
        lines.append(f"    enabled: {flags[name]}")
    # Each stanza is closed; only simulation needs settings for the shared validator.
    text = "\n".join(lines) + "\n"
    text = text.replace(
        "  simulate_system:\n",
        f"  simulate_system:\n    config_path: {path.stem}-simulation.yml\n",
    )
    path.write_text(text, encoding="utf-8")
    (path.parent / f"{path.stem}-simulation.yml").write_text(
        "operation: simulate-and-indicators\nexperiment_name: test\n", encoding="utf-8"
    )


@pytest.fixture()
def capture_runs(monkeypatch):
    """Patch subprocess.run to record argv lists; default success exit 0."""
    calls = []
    exits = {}  # index -> returncode override
    # These wrapper-only fixtures deliberately use minimal v1-shaped mappings.
    # Archive capture is exercised with a valid project set separately.
    from blueearth_cst.shared import configuration_archive, workflow_archive_launch

    monkeypatch.setattr(
        workflow_archive_launch,
        "prepare_workflow",
        lambda _name, config, _root, **_kwargs: (Path(config), Path(config)),
    )
    monkeypatch.setattr(rw, "_bind_workflow_archive", lambda *_args: None)
    # This orchestration-only fake returns its source path as a context. The
    # lifecycle integration tests exercise cleanup of real private contexts.
    monkeypatch.setattr(
        configuration_archive, "remove_launch_context", lambda *_args: None
    )
    monkeypatch.setattr(rw, "_capture_legacy_wf3_attempt", lambda *_args: "a" * 64)

    def fake_run(cmd, cwd=None, **kwargs):
        if cmd[0] == "git":
            stdout = "abc123\n" if "rev-parse" in cmd else ""
            return FakeResult(0, stdout=stdout)
        idx = len(calls)
        calls.append(cmd)
        if (
            "-s" in cmd
            and Path(cmd[cmd.index("-s") + 1]).name == "build_model.smk"
            and exits.get(idx, 0) == 0
        ):
            import yaml

            cfg = yaml.safe_load(Path(cmd[cmd.index("--configfile") + 1]).read_text())
            for leaf in rw.LEAVES:
                artifact = Path(cfg["project"]["project_dir"]) / leaf
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.touch()
        return FakeResult(exits.get(idx, 0))

    monkeypatch.setattr(
        rw,
        "run_project_child",
        lambda cmd, **kwargs: fake_run(cmd, **kwargs).returncode,
    )
    from scripts import generate_scenarios

    monkeypatch.setattr(
        generate_scenarios,
        "run_owned",
        lambda config_path, _root, cores, extra, **_kwargs: (
            fake_run(generate_scenarios._command(config_path, cores, extra)).returncode
        ),
    )
    return calls, exits


def _snakefiles_invoked(calls):
    """Return the ordered list of Snakefile names across captured argv."""
    out = []
    for cmd in calls:
        i = cmd.index("-s")
        out.append(Path(cmd[i + 1]).name)
    return out


def _q90_config(tmp_path, *, build_model=True):
    """A named Q90 request whose historical output has not been produced."""
    import yaml

    cfg = tmp_path / "q90-project.yml"
    flags = {name: "true" for name in rw.WORKFLOW_ORDER}
    flags["build_model"] = "true" if build_model else "false"
    _write_cfg(cfg, flags)
    project = yaml.safe_load(cfg.read_text())
    project["model"] = {"outvars": ["river discharge"]}
    cfg.write_text(yaml.safe_dump(project), encoding="utf-8")
    workflow = tmp_path / "q90-project-simulation.yml"
    workflow.write_text(
        yaml.safe_dump(
            {
                "operations": ["simulations", "indicators"],
                "experiment_name": "test",
                "indicator_sets": ["crop-water"],
                "indicator_set_definitions": {
                    "crop-water": {"config_path": "crop-water-q90.yml"}
                },
                "indicator_selection": {
                    "crop-water": {"include": ["summer_low_flow_days"]}
                },
            }
        ),
        encoding="utf-8",
    )
    model = tmp_path / "project/models/hydrology/wflow"
    (tmp_path / "crop-water-q90.yml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": "crop-water",
                "title": "Q90",
                "description": "Historical",
                "selection": {
                    "registry": {
                        "set": "crop-water",
                        "overrides": {
                            "summer_low_flow_days": {
                                "season": "summer",
                                "thresholds": {
                                    "baseline_quantile": {
                                        "model_root": str(model),
                                        "source_id": "historical",
                                    }
                                },
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return cfg, model


@pytest.mark.parametrize("inline", [False, True])
def test_combined_q90_dry_run_leaves_wf4_pending_without_historical_output(
    tmp_path, capture_runs, monkeypatch, capsys, inline
):
    cfg, model = _q90_config(tmp_path)
    if inline:
        _inline_q90_config(tmp_path)
    calls, _ = capture_runs
    # A dry-run child must not manufacture WF1's outputs.
    monkeypatch.setattr(
        rw, "run_project_child", lambda cmd, **_kwargs: calls.append(cmd) or 0
    )
    assert rw.run(str(cfg), cores=3, extra=["--dry-run"]) == 0
    assert not model.exists()
    assert _snakefiles_invoked(calls) == [
        "analyze_climate.smk",
        "build_model.smk",
        "analyze_projections.smk",
        "generate_scenarios.smk",
    ]
    output = capsys.readouterr().out
    assert "Q90 preparation pending WF1" in output
    assert "DAG not inspected" in output
    history = _read_only_manifest(tmp_path / "project")
    assert "simulate_system" in history["pending_workflows"]


def _inline_q90_config(tmp_path):
    """Use the project's model context without a separate set file."""
    import yaml

    path = tmp_path / "q90-project-simulation.yml"
    settings = yaml.safe_load(path.read_text())
    settings.pop("indicator_set_definitions")
    settings["indicator_sets"] = [
        {
            "set": "crop-water",
            "summer_low_flow_days": {"thresholds": {"baseline_quantile": {}}},
        }
    ]
    path.write_text(yaml.safe_dump(settings), encoding="utf-8")


def test_inline_duplicate_indicator_keys_are_not_silently_overwritten(
    tmp_path, capture_runs
):
    cfg, _ = _q90_config(tmp_path)
    _inline_q90_config(tmp_path)
    path = tmp_path / "q90-project-simulation.yml"
    path.write_text(
        path.read_text().replace(
            "  summer_low_flow_days:",
            "  summer_low_flow_days: {}\n  summer_low_flow_days:",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate YAML key"):
        rw.run(str(cfg), cores=3, extra=[])
    assert capture_runs[0] == []


@pytest.mark.parametrize("inline", [False, True])
def test_combined_q90_execution_prepares_only_after_wf1(
    tmp_path, capture_runs, monkeypatch, inline
):
    import numpy as np
    import pandas as pd

    from blueearth_cst.experiment import baseline_qcrit

    cfg, model = _q90_config(tmp_path)
    if inline:
        _inline_q90_config(tmp_path)
    calls, _ = capture_runs
    child = rw.run_project_child
    prepare = baseline_qcrit.resolve_thresholds
    preparations = []

    def observed_prepare(parameters, path, **kwargs):
        if kwargs.get("prepare", True):
            assert "build_model.smk" in _snakefiles_invoked(calls)
            preparations.append(path)
        return prepare(parameters, path, **kwargs)

    def produce_historical(cmd, **kwargs):
        result = child(cmd, **kwargs)
        if Path(cmd[cmd.index("-s") + 1]).name == "build_model.smk":
            (model / "run_default").mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {"Q_9": np.arange(365)}, index=pd.date_range("1999-01-01", "1999-12-31")
            ).to_csv(model / "run_default/output.csv", index_label="time")
            (model / "wflow_sbm.toml").write_text(
                '[time]\ncalendar="standard"\ntimestepsecs=86400\n'
                'starttime="1998-12-31T00:00:00"\nendtime="1999-12-31T00:00:00"\n'
                '[input]\npath_forcing="forcing/inmaps_historical.nc"\n'
                '[[output.csv.column]]\nheader="Q"\nparameter="river_water__volume_flow_rate"\n',
                encoding="utf-8",
            )
        return result

    monkeypatch.setattr(baseline_qcrit, "resolve_thresholds", observed_prepare)
    monkeypatch.setattr(rw, "run_project_child", produce_historical)
    monkeypatch.setattr(
        simulation_record,
        "capture_simulation_sources_v2",
        lambda *_: tmp_path / "capture.json",
    )
    assert rw.run(str(cfg), cores=3, extra=[]) == 0
    assert preparations
    assert _snakefiles_invoked(calls)[-1] == "simulate_system.smk"


def test_q90_without_wf1_reports_required_historical_run(tmp_path, capture_runs):
    cfg, _ = _q90_config(tmp_path, build_model=False)
    with pytest.raises(
        ValueError, match="missing WF1.*historical simulation before WF4"
    ):
        rw.run(str(cfg), cores=3, extra=["--dry-run"])


def test_combined_q90_dry_run_does_not_hide_corrupt_existing_toml(
    tmp_path, capture_runs, monkeypatch, capsys
):
    cfg, model = _q90_config(tmp_path)
    calls, _ = capture_runs
    model.mkdir(parents=True)
    (model / "wflow_sbm.toml").write_text("[broken", encoding="utf-8")
    monkeypatch.setattr(
        rw, "run_project_child", lambda cmd, **_kwargs: calls.append(cmd) or 0
    )
    with pytest.raises(ValueError, match="Expected"):
        rw.run(str(cfg), cores=3, extra=["--dry-run"])
    assert "simulate_system.smk" not in _snakefiles_invoked(calls)
    assert "Q90 preparation pending WF1" not in capsys.readouterr().out


def test_combined_q90_refuses_malformed_rule_before_upstream_execution(
    tmp_path, capture_runs
):
    import yaml

    cfg, _ = _q90_config(tmp_path)
    path = tmp_path / "crop-water-q90.yml"
    document = yaml.safe_load(path.read_text())
    rule = document["selection"]["registry"]["overrides"]["summer_low_flow_days"][
        "thresholds"
    ]["baseline_quantile"]
    rule["quantile"] = 2
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid Qcrit"):
        rw.run(str(cfg), cores=3, extra=[])
    assert capture_runs[0] == []


def _manifests(project_dir: Path) -> list[Path]:
    """Return only parent invocation records, excluding workflow children."""
    paths = (project_dir / "config" / "_engine" / "invocations").glob("*.json")
    return sorted(
        path
        for path in paths
        if json.loads(path.read_text(encoding="utf-8"))["workflow"] is None
    )


def _read_only_manifest(project_dir: Path) -> dict:
    """Read the sole invocation manifest below a test project root."""
    manifests = _manifests(project_dir)
    assert len(manifests) == 1
    return json.loads(manifests[0].read_text(encoding="utf-8"))


# --- §7(g) assertion 1: all-true -> all five in fixed order ------------------


def test_all_true_invokes_five_in_fixed_order(tmp_path, capture_runs):
    calls, _ = capture_runs
    cfg = tmp_path / "c.yml"
    _write_cfg(cfg, {n: "true" for n in rw.WORKFLOW_ORDER})
    rc = rw.run(str(cfg), cores=3, extra=[])
    assert rc == 0
    assert _snakefiles_invoked(calls) == [
        "analyze_climate.smk",
        "build_model.smk",
        "analyze_projections.smk",
        "generate_scenarios.smk",
        "simulate_system.smk",
    ]


def test_preparation_is_announced_before_experiment_imports(
    tmp_path, capture_runs, monkeypatch, capsys
):
    import builtins

    original_import = builtins.__import__
    before_import = []

    def observe_import(name, *args, **kwargs):
        if name == "blueearth_cst.experiment.reservation_context":
            captured = capsys.readouterr()
            before_import.append(captured.out + captured.err)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", observe_import)
    cfg = tmp_path / "c.yml"
    _write_cfg(cfg, {name: "true" for name in rw.WORKFLOW_ORDER})
    assert rw.run(str(cfg), cores=1, extra=[]) == 0
    assert before_import
    assert "Preparing configuration and experiment checks" in before_import[0]
    output = "".join(before_import) + capsys.readouterr().out
    assert output.count("\n  run_workflows\n") == 1


def test_preparation_watchdog_stops_when_admission_fails(
    tmp_path, capture_runs, monkeypatch
):
    from blueearth_cst.experiment import reservation_context

    stopped = []
    original_stop = rw._PreparationHeartbeat.stop

    def stop(heartbeat):
        original_stop(heartbeat)
        stopped.append(not heartbeat._thread.is_alive())

    def refuse(*args, **kwargs):
        raise ValueError("changed request")

    monkeypatch.setattr(rw._PreparationHeartbeat, "stop", stop)
    monkeypatch.setattr(reservation_context, "admit_context_owned", refuse)
    cfg = tmp_path / "c.yml"
    _write_cfg(cfg, {name: "true" for name in rw.WORKFLOW_ORDER})
    with pytest.raises(ValueError, match="changed request"):
        rw.run(str(cfg), cores=1, extra=[])
    assert stopped == [True]
    assert _read_only_manifest(tmp_path / "project")["status"] == "failed"


def test_preparation_progress_reports_elapsed_time_in_the_introductory_row(
    monkeypatch, capsys
):
    monkeypatch.setenv("CST_HEARTBEAT_SECS", "0")
    clock = [100.0]
    monkeypatch.setattr(rw.time, "monotonic", lambda: clock[0])
    with rw._preparation_progress(True):
        initial = capsys.readouterr()
        assert initial.err == "\n  Preparing configuration and experiment checks..."
        clock[0] += 22
    assert capsys.readouterr().err == " (0:00:22 elapsed)\n"


@pytest.mark.parametrize("interactive", [False, True])
def test_preparation_timer_uses_one_row(monkeypatch, interactive):
    from io import StringIO

    stream = StringIO()
    monkeypatch.setattr(stream, "isatty", lambda: interactive)
    monkeypatch.setenv("CST_HEARTBEAT_SECS", "0")
    clock = [100.0]
    monkeypatch.setattr(rw.time, "monotonic", lambda: clock[0])
    timer = rw._PreparationHeartbeat(
        "Preparing configuration and experiment checks", stream
    ).start()
    for elapsed in (15, 30):
        clock[0] = 100 + elapsed
        timer._tick(clock[0])
    assert "\n" not in stream.getvalue()
    clock[0] = 143
    timer.stop()
    output = stream.getvalue()
    expected = "  Preparing configuration and experiment checks... (0:00:43 elapsed)\n"
    assert output.count("\n") == 1
    if interactive:
        assert output.split("\r")[-1] == "\033[2K" + expected
    else:
        assert output == expected


@pytest.mark.parametrize("failure", ["banner", "watchdog_start", "watchdog_stop"])
def test_preparation_reporting_never_masks_admission_errors(monkeypatch, failure):
    def unavailable(*args, **kwargs):
        raise OSError("console unavailable")

    class Watchdog:
        def __init__(self, *args):
            pass

        def start(self):
            if failure == "watchdog_start":
                unavailable()
            return self

        def stop(self):
            if failure == "watchdog_stop":
                unavailable()

    monkeypatch.setattr(rw, "_PreparationHeartbeat", Watchdog)
    if failure == "banner":
        monkeypatch.setattr(rw, "_banner", unavailable)
    with pytest.raises(ValueError, match="admission failed"):
        with rw._preparation_progress(True):
            raise ValueError("admission failed")


# --- §7(g) assertion 2: --keep-going on projections only (flag parity) -------


def test_keep_going_on_projections_only(tmp_path, capture_runs):
    calls, _ = capture_runs
    cfg = tmp_path / "c.yml"
    _write_cfg(cfg, {n: "true" for n in rw.WORKFLOW_ORDER})
    rw.run(str(cfg), cores=3, extra=[])
    by_sf = {Path(cmd[cmd.index("-s") + 1]).name: cmd for cmd in calls}
    assert "--keep-going" in by_sf["analyze_projections.smk"]
    assert "--keep-going" not in by_sf["build_model.smk"]
    assert "--keep-going" not in by_sf["simulate_system.smk"]
    assert "--keep-going" not in by_sf["analyze_climate.smk"]


# --- §7(g) assertion 3: missing enabled: key -> nonzero, named --------------


def test_missing_enabled_key_errors(tmp_path):
    cfg = tmp_path / "c.yml"
    cfg.write_text(
        "workflows:\n"
        # Every required section present but one: the closed set is all-or-
        # error, so an incomplete config would otherwise fail on whichever
        # section came first rather than on the one this test is about.
        "  analyze_climate:\n    enabled: true\n"
        "  generate_scenarios:\n    enabled: true\n"
        "  build_model:\n    enabled: true\n"
        "  analyze_projections:\n    enabled: true\n"
        "  simulate_system:\n    other: 1\n",  # no enabled:
        encoding="utf-8",
    )
    with pytest.raises(rw.ConfigError) as exc:
        rw.read_enabled_flags(str(cfg))
    assert "simulate_system.enabled" in str(exc.value)
    # And the CLI surfaces it as a nonzero exit.
    rc = rw.main(["--config", str(cfg)])
    assert rc != 0


def test_missing_workflows_section_errors(tmp_path):
    """A projections-only config (no workflows: section) is rejected (contract a/b)."""
    cfg = tmp_path / "proj.yml"
    cfg.write_text("data_sources: config/catalogs/cmip6_data.yml\n", encoding="utf-8")
    with pytest.raises(rw.ConfigError) as exc:
        rw.read_enabled_flags(str(cfg))
    assert "workflows" in str(exc.value)


# --- §7(g) assertion 4: parsed-value bool contract --------------------------


@pytest.mark.parametrize("bad", ['"true"', '"false"', "1", "0"])
def test_non_bool_enabled_rejected(tmp_path, bad):
    """Quoted strings and integers do NOT parse to bool -> rejected (contract c)."""
    cfg = tmp_path / "c.yml"
    flags = {n: "true" for n in rw.WORKFLOW_ORDER}
    flags["build_model"] = bad
    _write_cfg(cfg, flags)
    with pytest.raises(rw.ConfigError) as exc:
        rw.read_enabled_flags(str(cfg))
    assert "build_model.enabled" in str(exc.value)


@pytest.mark.parametrize("spelling", ["yes", "on", "true", "True"])
def test_unquoted_boolean_spellings_accepted(tmp_path, spelling):
    """Unquoted yes/on/true resolve to True under YAML 1.1 -> accepted (contract c)."""
    cfg = tmp_path / "c.yml"
    flags = {n: "true" for n in rw.WORKFLOW_ORDER}
    flags["build_model"] = spelling
    _write_cfg(cfg, flags)
    parsed = rw.read_enabled_flags(str(cfg))
    assert parsed["build_model"] is True


# --- §7(g) assertion 5: first nonzero -> stop, later not invoked, return code -


def test_first_nonzero_stops_and_returns_code(tmp_path, capture_runs):
    calls, exits = capture_runs
    exits[0] = 7  # first invoked workflow (analyze_climate) fails
    cfg = tmp_path / "c.yml"
    _write_cfg(cfg, {n: "true" for n in rw.WORKFLOW_ORDER})
    rc = rw.run(str(cfg), cores=3, extra=[])
    assert rc == 7
    # Only the first workflow was invoked; projections/experiment were not.
    assert _snakefiles_invoked(calls) == ["analyze_climate.smk"]


# --- §7(g) assertion 6: --cores / -- <extra> forwarded to EVERY invocation ---


@pytest.mark.parametrize("extra", [["--dry-run", "--unlock"], ["--unlock"]])
def test_cores_and_extra_forwarded_to_every_invocation(tmp_path, capture_runs, extra):
    calls, _ = capture_runs
    cfg = tmp_path / "c.yml"
    _write_cfg(cfg, {n: "true" for n in rw.WORKFLOW_ORDER})
    rw.run(str(cfg), cores=8, extra=extra)
    assert len(calls) == 5
    for cmd in calls:
        assert cmd[cmd.index("-c") + 1] == "8"
        for flag in extra:
            assert flag in cmd
        assert "--unlock" in cmd


def test_unlock_skips_execution_setup_and_publication(
    tmp_path, capture_runs, monkeypatch
):
    from blueearth_cst.experiment import indicator_export, simulation_runner
    from blueearth_cst.shared import workflow_archive_launch

    def unexpected(*args, **kwargs):
        pytest.fail("Unlock entered execution setup or publication")

    monkeypatch.setattr(rw, "_check_wf1_leaves", unexpected)
    monkeypatch.setattr(simulation_record, "capture_simulation_sources_v2", unexpected)
    monkeypatch.setattr(
        simulation_runner, "validate_retained_indicator_sets", unexpected
    )
    monkeypatch.setattr(indicator_export, "publish_selected_sets", unexpected)
    monkeypatch.setattr(workflow_archive_launch, "prepare_workflow", unexpected)
    calls, _ = capture_runs
    cfg = tmp_path / "c.yml"
    _write_cfg(cfg, {name: "true" for name in rw.WORKFLOW_ORDER})
    assert rw.run(str(cfg), cores=1, extra=["--unlock"]) == 0
    assert len(calls) == 5


def test_cli_strips_double_dash_sentinel(tmp_path, capture_runs):
    """The leading `--` sentinel is stripped before forwarding (contract e)."""
    calls, _ = capture_runs
    cfg = tmp_path / "c.yml"
    _write_cfg(cfg, {n: "true" for n in rw.WORKFLOW_ORDER})
    rw.main(["--config", str(cfg), "--cores", "2", "--", "--dry-run"])
    for cmd in calls:
        assert "--" not in cmd or "--dry-run" in cmd  # no bare sentinel forwarded
        assert "--dry-run" in cmd


def test_rebuild_clears_outputs_before_workflow_launch_and_retains_history(
    tmp_path, capture_runs
):
    calls, _ = capture_runs
    config = tmp_path / "project.yml"
    root = tmp_path / "project"
    flags = {name: "false" for name in rw.WORKFLOW_ORDER}
    flags["analyze_climate"] = "true"
    _write_cfg(config, flags, project_dir=str(root))
    config.write_text("schema_version: 3\n" + config.read_text(), encoding="utf-8")
    product = root / "experiments/old/results.csv"
    product.parent.mkdir(parents=True)
    product.write_text("old results", encoding="utf-8")
    log = root / "logs/previous/console.log"
    log.parent.mkdir(parents=True)
    log.write_text("old log", encoding="utf-8")
    assert rw.main(["--config", str(config), "--rebuild", "--yes"]) == 0
    assert not product.exists()
    assert log.read_text() == "old log"
    assert _snakefiles_invoked(calls) == ["analyze_climate.smk"]
    record = next(
        json.loads(path.read_text())
        for path in (root / "config/_engine/invocations").glob("*.json")
        if json.loads(path.read_text()).get("rebuild")
    )
    assert record["rebuild"] == {"status": "completed", "targets": ["experiments"]}
    assert record["status"] == "succeeded"


# --- ext1-03: enabled:false skip test, FRESH tmp_path, boundary assertion ----


def test_enabled_false_skips_at_subprocess_boundary(tmp_path, capture_runs):
    """Design §9 ext1-03: in a FRESH temp project_dir, disabling a workflow means
    the wrapper does NOT invoke its Snakefile, and DOES invoke the others.
    Asserted at the subprocess boundary (argv capture), not output presence --
    a reused dir could carry stale outputs."""
    calls, _ = capture_runs
    project_dir = tmp_path / "fresh_project"
    project_dir.mkdir()
    cfg = tmp_path / "c.yml"
    flags = {n: "true" for n in rw.WORKFLOW_ORDER}
    flags["analyze_projections"] = "false"
    _write_cfg(cfg, flags, project_dir=str(project_dir))

    rc = rw.run(str(cfg), cores=3, extra=[])
    assert rc == 0
    invoked = _snakefiles_invoked(calls)
    assert "analyze_projections.smk" not in invoked
    assert invoked == [
        "analyze_climate.smk",
        "build_model.smk",
        "generate_scenarios.smk",
        "simulate_system.smk",
    ]


def test_all_enabled_inverse_all_invoked(tmp_path, capture_runs):
    """The inverse of the skip test: all true -> all five invoked."""
    calls, _ = capture_runs
    cfg = tmp_path / "c.yml"
    _write_cfg(cfg, {n: "true" for n in rw.WORKFLOW_ORDER})
    rw.run(str(cfg), cores=3, extra=[])
    assert len(_snakefiles_invoked(calls)) == 5


def test_success_manifest_is_initialized_before_first_workflow_and_finalized(
    tmp_path, monkeypatch
):
    """The unique manifest exists as running before the child is launched."""
    project_dir = tmp_path / "project"
    cfg = tmp_path / "c.yml"
    _write_cfg(cfg, {n: "true" for n in rw.WORKFLOW_ORDER}, project_dir)
    calls = []

    def fake_run(cmd, cwd=None, **kwargs):
        if cmd[0] == "git":
            return FakeResult(0, stdout="abc123\n" if "rev-parse" in cmd else "")
        calls.append(cmd)
        running = _read_only_manifest(project_dir)
        assert running["status"] == "running"
        assert running["ended_at_utc"] is None
        if "build_model.smk" in cmd:
            _staged(project_dir, rw.LEAVES)
        return FakeResult(0)

    monkeypatch.setattr(
        rw,
        "run_project_child",
        lambda cmd, **kwargs: fake_run(cmd, **kwargs).returncode,
    )

    from scripts import generate_scenarios

    monkeypatch.setattr(
        generate_scenarios,
        "run_owned",
        lambda config_path, _root, cores, extra, **_kwargs: (
            fake_run(generate_scenarios._command(config_path, cores, extra)).returncode
        ),
    )

    assert rw.run(str(cfg), cores=5, extra=["--dry-run"]) == 0

    manifest = _read_only_manifest(project_dir)
    assert manifest["schema_version"] == "invocation/1"
    assert manifest["status"] == "succeeded"
    assert manifest["exit_code"] == 0
    assert manifest["ended_at_utc"].endswith("Z")
    assert manifest["mode"] == "dry_run"
    assert manifest["work_performed"] == "no"
    assert manifest["configuration"]["source_config_sha256"] == rw.file_sha256(cfg)
    assert manifest["configuration"]["effective_config_sha256"]
    # Derived from WORKFLOW_ORDER rather than restated: a literal list is what
    # made this the last of nine tests to fail when the set widened to four,
    # each for the same reason.
    assert [item["workflow"] for item in manifest["children"]] == list(
        rw.WORKFLOW_ORDER
    )
    assert len(calls) == len(rw.WORKFLOW_ORDER)


def test_no_op_invocations_each_get_an_immutable_manifest(tmp_path, capture_runs):
    """Repeated all-disabled calls create separate finalized records."""
    project_dir = tmp_path / "project"
    cfg = tmp_path / "c.yml"
    _write_cfg(cfg, {n: "false" for n in rw.WORKFLOW_ORDER}, project_dir)

    assert rw.run(str(cfg), cores=3, extra=[]) == 0
    assert rw.run(str(cfg), cores=3, extra=[]) == 0

    manifests = _manifests(project_dir)
    assert len(manifests) == 2
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["status"] == "succeeded"
        assert manifest["work_performed"] == "no"
        assert manifest["requested_workflows"] == []
        assert manifest["children"] == []


def test_failure_manifest_records_stop_boundary(tmp_path, capture_runs):
    """A child failure finalizes its result and marks later work not run."""
    _, exits = capture_runs
    exits[0] = 9
    project_dir = tmp_path / "project"
    cfg = tmp_path / "c.yml"
    _write_cfg(cfg, {n: "true" for n in rw.WORKFLOW_ORDER}, project_dir)

    assert rw.run(str(cfg), cores=3, extra=[]) == 9

    manifest = _read_only_manifest(project_dir)
    # analyze_climate leads WORKFLOW_ORDER, so it is the one exits[0] hits.
    assert manifest["exit_code"] == 9
    assert [item["workflow"] for item in manifest["children"]] == ["analyze_climate"]
    child = json.loads(
        (
            _manifests(project_dir)[0].parent
            / f"{manifest['children'][0]['invocation_id']}.json"
        ).read_text(encoding="utf-8")
    )
    assert child["status"] == "failed"
    assert child["exit_code"] == 9


def test_subprocess_exception_finalizes_failure_manifest(
    tmp_path, monkeypatch, capture_runs
):
    """Launch errors leave a terminal record rather than a stale running one."""
    project_dir = tmp_path / "project"
    cfg = tmp_path / "c.yml"
    _write_cfg(cfg, {n: "true" for n in rw.WORKFLOW_ORDER}, project_dir)

    def fake_run(cmd, cwd=None, **kwargs):
        if cmd[0] == "git":
            return FakeResult(0, stdout="abc123\n" if "rev-parse" in cmd else "")
        raise OSError("snakemake executable missing")

    monkeypatch.setattr(
        rw,
        "run_project_child",
        lambda cmd, **kwargs: fake_run(cmd, **kwargs).returncode,
    )

    with pytest.raises(OSError, match="snakemake executable missing"):
        rw.run(str(cfg), cores=3, extra=[])

    manifest = _read_only_manifest(project_dir)
    assert manifest["status"] == "failed"
    assert manifest["exit_code"] is None
    assert manifest["error"]["type"] == "OSError"
    assert [item["workflow"] for item in manifest["children"]] == ["analyze_climate"]


def test_manifest_sanitizes_sensitive_extra_args(tmp_path, capture_runs):
    """Sensitive flag and --config assignment values never reach the record."""
    project_dir = tmp_path / "project"
    cfg = tmp_path / "c.yml"
    _write_cfg(cfg, {n: "false" for n in rw.WORKFLOW_ORDER}, project_dir)
    extra = [
        "--config",
        "threshold=4",
        "api_token=token-value",
        "clientSecret=camel-secret-value",
        "--password",
        "password-value",
    ]

    rw.run(str(cfg), cores=3, extra=extra)

    manifest_text = _manifests(project_dir)[0].read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert "token-value" not in manifest_text
    assert "camel-secret-value" not in manifest_text
    assert "password-value" not in manifest_text
    assert "threshold=4" in manifest["command"]
    assert "api_token=<redacted>" in manifest["command"]


def test_sensitive_args_are_redacted_from_console(tmp_path, capture_runs, capsys):
    """The wrapper's command echo does not defeat manifest sanitization."""
    project_dir = tmp_path / "project"
    cfg = tmp_path / "c.yml"
    flags = {n: "false" for n in rw.WORKFLOW_ORDER}
    flags["build_model"] = "true"
    _write_cfg(cfg, flags, project_dir)

    rw.run(str(cfg), cores=3, extra=["--config", "api_token=visible-secret"])

    output = capsys.readouterr().out
    assert "visible-secret" not in output
    assert "api_token=<redacted>" in output


# --- contract (h): the wrapper's own console narration ----------------------


def test_command_display_wraps_long_options_without_changing_argv():
    argv = rw.build_command(
        "analyze_projections",
        "C:/cases/gabon-ntoum/project_config.yml",
        3,
        ["--config", "api_token=secret", "threshold=4"],
    )
    original = argv.copy()
    assert rw._format_command(argv).splitlines() == [
        "snakemake all -c 3 -s analyze_projections.smk",
        "              --configfile C:/cases/gabon-ntoum/project_config.yml",
        "              --keep-going",
        "              --config api_token=<redacted> threshold=4",
    ]
    assert argv == original


@pytest.mark.parametrize("width", [40, 60, 100])
def test_long_command_display_fits_terminal_without_mutating_arguments(width):
    argv = rw.build_command(
        "analyze_projections",
        "C:/cases/" + "very-long-path/" * 8 + "project config.yml",
        3,
        [],
    )
    original = argv.copy()
    lines = rw._format_command(argv, width=width).splitlines()
    assert all(len(line) <= width for line in lines)
    path_lines = lines[lines.index("              --configfile") + 1 : -1]
    assert all(line.startswith(" " * 18) for line in path_lines)
    assert argv == original


def test_narrow_settings_keep_continuations_in_the_value_column(monkeypatch):
    monkeypatch.setattr(rw, "_console_width", lambda: 60)
    value = "C:/cases/" + "regional-assessment/" * 4 + "project_config.yml"
    block = rw._console_block("settings preview", [("settings", [("config", value)])])
    lines = block.splitlines()
    assert all(len(line) <= 60 for line in lines)
    start = next(
        index for index, line in enumerate(lines) if line.startswith("    config")
    )
    assert all(line.startswith(" " * 12) for line in lines[start + 1 :])


def test_command_receipt_uses_owned_runners_and_preserves_quoted_arguments(tmp_path):
    config = tmp_path / "project config.yml"
    project = tmp_path / "project"
    receipt = rw._write_command_receipt(
        tmp_path / "invocation.json",
        config_path=str(config),
        project_dir=project,
        flags={name: True for name in rw.WORKFLOW_ORDER},
        cores=3,
        extra=["--config", "api_token=secret", "label=O'Brien $value"],
        simulation_targets=("all",),
    )
    text = receipt.read_text(encoding="utf-8")
    for name in rw.WORKFLOW_ORDER:
        assert f"scripts/run_workflow.py {name}" in text
    assert "secret" not in text
    assert "api_token=<redacted>" in text
    assert str(config.resolve()) in text
    if os.name == "nt":
        assert "'label=O''Brien $value'" in text
        assert " `\n" in text
    else:
        assert " \\\n" in text


@pytest.mark.skipif(
    os.name != "nt", reason="PowerShell quoting applies to Windows receipts"
)
def test_copyable_powershell_preserves_arguments_as_literal_ast_values(tmp_path):
    argv = [
        "pixi",
        "run",
        "python",
        "scripts/run_workflow.py",
        "analyze_climate",
        "--config",
        "C:/project files/config.yml",
        "--",
        "--config",
        "label=O'Brien $value",
        "expression=$(throw 'injected');literal`",
        "api_token=fake-secret",
    ]
    command_path = tmp_path / "command.txt"
    command_path.write_text(
        rw._copyable_command(argv, powershell=True), encoding="utf-8"
    )
    parser_path = tmp_path / "parse-command.ps1"
    parser_path.write_text(
        "param([string]$CommandPath)\n"
        "$tokens = $null; $parseErrors = $null\n"
        "$tree = [System.Management.Automation.Language.Parser]::ParseFile($CommandPath, [ref]$tokens, [ref]$parseErrors)\n"
        "if ($parseErrors.Count) { throw 'Command has parse errors' }\n"
        "$command = $tree.Find({param($node) $node -is [System.Management.Automation.Language.CommandAst]}, $true)\n"
        "@($command.CommandElements | ForEach-Object { $_.Value }) | ConvertTo-Json -Compress\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(parser_path), str(command_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == rw.sanitize_argv(argv)


def test_runner_restores_warning_tracking_after_a_launch_exception(
    tmp_path, capture_runs, monkeypatch, capsys
):
    from blueearth_cst.shared.run_log_core import note_warning

    config = tmp_path / "config.yml"
    project = tmp_path / "project"
    flags = {name: "false" for name in rw.WORKFLOW_ORDER}
    flags["analyze_climate"] = "true"
    _write_cfg(config, flags, project)
    monkeypatch.delenv(rw.RUN_WARNING_TALLY_ENV, raising=False)

    def failed_child(*_args, **_kwargs):
        note_warning("preflight")
        raise OSError("child launch failed")

    monkeypatch.setattr(rw, "run_project_child", failed_child)
    with pytest.raises(OSError, match="child launch failed"):
        rw.run(str(config), 3, [])
    assert rw.RUN_WARNING_TALLY_ENV not in os.environ
    assert "run_workflows FAILED with 1 warning" in capsys.readouterr().out
    assert not list((project / "config/_engine/invocations").glob(".*.warnings"))


@pytest.mark.parametrize("fails", [False, True])
def test_runner_warning_summary_counts_only_this_invocation(
    tmp_path, capture_runs, capsys, monkeypatch, fails
):
    from blueearth_cst.shared.run_log_core import note_warning

    config = tmp_path / "c.yml"
    project = tmp_path / "project"
    flags = {name: "false" for name in rw.WORKFLOW_ORDER}
    flags.update(analyze_climate="true", analyze_projections="true")
    _write_cfg(config, flags, project)
    (project / "logs").mkdir(parents=True)
    stale = project / "logs/.wf0_analyze_climate.log.tally"
    stale.write_text("old-warning\n" * 20, encoding="utf-8")
    original = rw.run_project_child
    inherited = str(tmp_path / "outer-tally")
    monkeypatch.setenv(rw.RUN_WARNING_TALLY_ENV, inherited)

    def emit_warnings(command, **kwargs):
        assert (
            kwargs["env"][rw.RUN_WARNING_TALLY_ENV]
            == os.environ[rw.RUN_WARNING_TALLY_ENV]
        )
        count = 1 if "analyze_climate.smk" in command else 2
        for _ in range(count):
            note_warning("fixture")
        result = original(command, **kwargs)
        return 1 if fails and count == 2 else result

    monkeypatch.setattr(rw, "run_project_child", emit_warnings)
    for _ in range(2):
        assert rw.run(str(config), 3, []) == int(fails)
        output = capsys.readouterr().out
        assert (
            f"run_workflows {'FAILED' if fails else 'done'} with 3 warnings" in output
        )
        assert _group_rows(output, "warnings") == {
            "wf0": "analyze_climate      1",
            "wf2": "analyze_projections  2",
        }
        assert os.environ[rw.RUN_WARNING_TALLY_ENV] == inherited
        assert not list((project / "config/_engine/invocations").glob(".*.warnings"))
    for path in _manifests(project):
        assert json.loads(path.read_text(encoding="utf-8"))["warning_counts"] == {
            "analyze_climate": 1,
            "analyze_projections": 2,
        }
    assert not Path(inherited).exists()
    assert stale.read_text(encoding="utf-8").count("old-warning") == 20


def test_experiment_selection_is_inside_opening_settings(
    tmp_path, capture_runs, capsys
):
    _, output, _ = _run_and_capture(
        tmp_path, capsys, {name: "true" for name in rw.WORKFLOW_ORDER}
    )
    assert _group_rows(output, "settings")["experiment"] == "test (continue)"
    assert "Selected experiment:" not in output


def test_region_description_keeps_unhandled_configuration():
    region = {"subbasin": [9.666, 0.4476], "uparea": 100, "strord": 4}
    assert rw._format_region(region) == str(region)
    assert rw._format_region({"bbox": [0, 1, 2, 3]}) == "{'bbox': [0, 1, 2, 3]}"


def _group_raw_lines(output, label):
    """The raw lines under one ``  <label>`` group, indentation intact."""
    lines = output.splitlines()
    start = lines.index(f"  {label}") + 1
    kept = []
    for line in lines[start:]:
        if not line.startswith("    "):
            break
        kept.append(line)
    return kept


def _group_lines(output, label):
    """The same lines, stripped -- for asserting on a continuation line."""
    return [line.strip() for line in _group_raw_lines(output, label)]


def _group_rows(output, label):
    """The ``key -> value`` rows under one ``  <label>`` group of a block."""
    rows = {}
    for line in _group_lines(output, label):
        key, value = line.split(maxsplit=1)
        rows[key] = value
    return rows


def _run_and_capture(tmp_path, capsys, flags, *, cores=3, extra=None):
    """Run the wrapper over a config with ``flags`` and return its stdout."""
    project_dir = tmp_path / "gabon_project"
    cfg = tmp_path / "c.yml"
    _write_cfg(cfg, flags, project_dir=str(project_dir))
    code = rw.run(str(cfg), cores=cores, extra=extra or [])
    return code, capsys.readouterr().out, project_dir


@pytest.mark.parametrize(
    ("identity", "expected"),
    [
        (
            {"commit": "d0802bf054", "release_tag": "v0.4.1", "dirty": False},
            "v0.4.1 (d0802bf0)",
        ),
        (
            {"commit": "d0802bf054", "release_tag": "v0.4.1", "dirty": True},
            "v0.4.1 (d0802bf0; local changes)",
        ),
        (
            {"commit": "d0802bf054", "dirty": None},
            "unversioned (d0802bf0; working tree unknown)",
        ),
        ({"commit": None}, "version unavailable"),
    ],
)
def test_toolbox_label(identity, expected):
    assert rw._toolbox_label(identity) == expected


def test_opening_block_states_the_project_and_the_config_in_one_group(
    tmp_path, capture_runs, capsys, monkeypatch
):
    """A console of four back-to-back Snakemake runs must say whose they are.

    Snakemake's own preamble names the host and the job counts and never the
    project, so before this block a scrolled-back or pasted console could not be
    attributed to a project or a config without asking.

    ONE group: what came from the command line and what came from the config
    were two, and that is a distinction the writer cares about rather than the
    reader. `cores` went with the split -- it is not a property of the project,
    and every hand-off band prints the `-c N` it forwarded.
    """
    monkeypatch.setattr(
        rw,
        "toolbox_identity",
        lambda: {"commit": "d0802bf054", "commit_source": "git", "dirty": False},
    )
    monkeypatch.setattr(rw, "toolbox_release_tag", lambda commit: "v0.4.1")
    _, out, project_dir = _run_and_capture(
        tmp_path, capsys, {n: "true" for n in rw.WORKFLOW_ORDER}, cores=7
    )
    assert "  run_workflows" in out.splitlines()  # the banner's head line
    assert "  run" not in out.splitlines()  # the old first group is gone
    rows = _group_rows(out, "settings")
    assert rows["project"] == "gabon_project"  # basename, no project_name key
    assert rows["repository"] == str(rw._REPO_ROOT_PATH).replace(os.sep, "/")
    assert rows["toolbox"] == "v0.4.1 (d0802bf0)"
    assert rows["folder"] == str(project_dir).replace(os.sep, "/")
    assert "c.yml" in rows["config"]
    assert "cores" not in rows


def test_opening_block_diagrams_the_sequence_and_marks_the_disabled(
    tmp_path, capture_runs, capsys
):
    """Every workflow appears; only the enabled ones are numbered.

    A disabled workflow silently absent from the list is indistinguishable from
    one the wrapper does not know about -- so it is shown, marked, and the
    positions count only what will actually be invoked.
    """
    flags = {n: "true" for n in rw.WORKFLOW_ORDER}
    flags["analyze_projections"] = "false"
    _, out, _ = _run_and_capture(tmp_path, capsys, flags)
    assert "  sequence\n" in out
    assert "[1/4]  wf0 analyze_climate" in out
    assert "[2/4]  wf1 build_model" in out
    assert "[3/4]  wf3 generate_scenarios" in out
    assert "[4/4]  wf4 simulate_system" in out
    # Present, marked, and NOT given a position.
    assert "wf2 analyze_projections  disabled" in out
    assert "[4/4]  wf2" not in out


def test_the_sequence_is_drawn_as_a_rail_of_nodes(tmp_path, capture_runs, capsys):
    """Keyed by node, and keyed DIFFERENTLY for what will not be invoked.

    The rows sat at the same indent as the `run` group's key/value pairs, so
    the diagram's shape did not say it was a diagram. A filled node for a
    workflow that will run and a hollow one for a workflow that will not is
    legible before any of the text is read -- and costs one character where the
    framed boxes this replaced cost three lines and forty columns each.
    """
    flags = {n: "true" for n in rw.WORKFLOW_ORDER}
    flags["analyze_projections"] = "false"
    _, out, _ = _run_and_capture(tmp_path, capsys, flags)
    lines = [line.strip() for line in out.splitlines()]

    # Nothing has run when the opening block prints, so every node is hollow.
    enabled = lines.index(next(ln for ln in lines if "[1/4]  wf0" in ln))
    assert lines[enabled].startswith(rw._NODE_OFF)
    assert lines[lines.index("sequence") + 1] == ""

    disabled = lines.index(next(ln for ln in lines if "wf2 analyze_projections" in ln))
    assert lines[disabled].startswith("–")
    # The nodes are threaded, not merely stacked.
    assert rw._RAIL in lines[enabled + 1 : disabled]


def test_the_opening_block_states_the_basin_settings(tmp_path, capture_runs, capsys):
    """What the run is ABOUT, next to where it is: bbox, extent, window.

    These are the values someone checks before letting a multi-hour run
    proceed, and reading them meant opening the config in another window.
    """
    project_dir = tmp_path / "gabon_project"
    region = project_dir / "data" / "spatial" / "geoms"
    region.mkdir(parents=True)
    (region / "region.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [9.6, 0.4],
                                    [9.8, 0.4],
                                    [9.8, 0.6],
                                    [9.6, 0.6],
                                    [9.6, 0.4],
                                ]
                            ],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = tmp_path / "c.yml"
    _write_cfg(
        cfg, {n: "true" for n in rw.WORKFLOW_ORDER}, project_dir=str(project_dir)
    )
    cfg.write_text(
        cfg.read_text(encoding="utf-8") + "basin:\n"
        "  region: \"{'subbasin': [9.666, 0.4476], 'uparea': 100}\"\n"
        "  resolution: 0.00833\n",
        encoding="utf-8",
    )
    rw.run(str(cfg), cores=3, extra=[])
    out = capsys.readouterr().out
    rows = _group_rows(out, "settings")
    lines = _group_lines(out, "settings")

    assert rows["region"] == "subbasin outlet (9.6660, 0.4476); uparea >= 100 km²"
    assert rows["resolution"].startswith("0.00833 deg")
    # The delineated box is the region row CONTINUING -- one setting, stated and
    # then answered -- so it carries no key of its own and is aligned into the
    # value column (`~22 x 22 km` at this latitude; a scale, hence `approx`).
    region_at = next(i for i, line in enumerate(lines) if line.startswith("region"))
    assert lines[region_at + 1] == (
        "lon 9.6000 .. 9.8000, lat 0.4000 .. 0.6000  (approx 22 x 22 km)"
    )
    raw = _group_raw_lines(out, "settings")
    assert raw[region_at + 1].index("lon ") == raw[region_at].index("subbasin outlet")


def test_the_bbox_says_it_is_absent_rather_than_deriving_one(
    tmp_path, capture_runs, capsys
):
    """A box from `{'subbasin': [lon, lat], 'uparea': 100}` would be invented.

    The specification names an outlet and an upstream-area threshold; the extent
    of the basin they delineate is not knowable from them, and the bbox row is
    exactly where a reader has no way to check a fabricated number.
    """
    cfg = tmp_path / "c.yml"
    project_dir = tmp_path / "fresh_project"
    _write_cfg(
        cfg, {n: "true" for n in rw.WORKFLOW_ORDER}, project_dir=str(project_dir)
    )
    cfg.write_text(
        cfg.read_text(encoding="utf-8") + "basin:\n"
        "  region: \"{'subbasin': [9.666, 0.4476], 'uparea': 100}\"\n",
        encoding="utf-8",
    )
    rw.run(str(cfg), cores=3, extra=[])
    out = capsys.readouterr().out
    lines = _group_lines(out, "settings")

    region_at = next(i for i, line in enumerate(lines) if line.startswith("region"))
    assert "9.666" in lines[region_at]  # the specification is still stated
    assert lines[region_at + 1].startswith("not delineated yet")
    assert "approx" not in out  # and no extent, since there is no box


def test_the_basin_rows_are_omitted_when_the_config_declares_no_basin(
    tmp_path, capture_runs, capsys
):
    """The wrapper validates `workflows:` and nothing else, so both basin rows
    are optional -- but project, repository, folder and config always answer."""
    _, out, _ = _run_and_capture(
        tmp_path, capsys, {n: "true" for n in rw.WORKFLOW_ORDER}
    )
    rows = _group_rows(out, "settings")
    assert set(rows) == {
        "project",
        "experiment",
        "toolbox",
        "repository",
        "folder",
        "config",
    }
    # No orphaned continuation either: with no region asked for, there is
    # nothing whose absence is worth a line.
    assert "not delineated yet" not in out


def test_an_extent_is_withheld_when_the_geometry_is_not_in_degrees(tmp_path):
    """A projected polygon multiplied by 111 would print a confident wrong number.

    The bbox itself is still shown: it carries its units on its face, so a
    reader can see what it is. The km scale cannot.
    """
    projected = (500000.0, 4649776.0, 520000.0, 4669776.0)
    assert rw._bbox_extent_km(projected) is None
    assert rw._bbox_extent_km((9.6, 0.4, 9.8, 0.6)) is not None


def test_a_declared_bbox_member_is_preferred_over_walking_the_coordinates(tmp_path):
    """It is what every other reader of the artifact sees."""
    path = tmp_path / "region.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "Feature",
                "bbox": [1.0, 2.0, 3.0, 4.0],
                "geometry": {"type": "Point", "coordinates": [99.0, 99.0]},
            }
        ),
        encoding="utf-8",
    )
    assert rw._geojson_bbox(path) == (1.0, 2.0, 3.0, 4.0)


def test_a_dry_run_says_so_in_the_opening_block(tmp_path, capture_runs, capsys):
    """A dry run's console is otherwise nearly identical to a real one's."""
    _, out, _ = _run_and_capture(
        tmp_path,
        capsys,
        {n: "true" for n in rw.WORKFLOW_ORDER},
        extra=["--dry-run"],
    )
    assert "mode" in out and "dry run -- nothing is executed" in out


def test_each_invoked_workflow_gets_a_hand_off_band_at_its_leading_edge(
    tmp_path, capture_runs, capsys
):
    """Position, identity and when it started -- but no closing band on success.

    Every workflow signs off with its own `wfN <name> done in <h:mm:ss>`, so a
    runner band repeating that was a second copy of one fact. The duration
    survives in the closing block's `ran` group.
    """
    flags = {n: "true" for n in rw.WORKFLOW_ORDER}
    flags["analyze_projections"] = "false"
    _, out, _ = _run_and_capture(tmp_path, capsys, flags)
    assert re.search(
        r"\[1/4]  WF0  ANALYZE CLIMATE  --  starting at \d{4}-\d{2}-\d{2} \d\d:\d\d:\d\d",
        out,
    )
    assert re.search(
        r"\[4/4]  WF4  SIMULATE SYSTEM  --  starting at \d{4}-\d{2}-\d{2} \d\d:\d\d:\d\d",
        out,
    )
    assert "  --  done in " not in out
    # Flush left, title and command both. A band has no group label and no rows,
    # so an indent would only make the line a reader scans for start one column
    # later than the rule announcing it.
    lines = out.splitlines()
    title_at = next(
        i for i, line in enumerate(lines) if "WF0  ANALYZE CLIMATE  --" in line
    )
    assert lines[title_at].startswith("[1/4]")
    assert lines[title_at - 1] == rw._RULE  # the rule it hangs from
    assert lines[title_at + 1] == ""  # a blank line parts title and command
    assert lines[title_at + 2].startswith("snakemake ")
    # A disabled workflow gets NO band -- the sequence diagram above already
    # named it, once, before anything ran.
    assert "WF2  ANALYZE PROJECTIONS  --  starting" not in out
    # WF4's band must show the snakemake invocation it actually launches, not
    # `simulate_system.py`'s own entry-point command -- the wrapper script is
    # an implementation detail of getting there, never what ran.
    wf4_title_at = next(
        i for i, line in enumerate(lines) if "WF4  SIMULATE SYSTEM  --" in line
    )
    wf4_command = lines[wf4_title_at + 2]
    assert "-m snakemake" in wf4_command
    assert "simulate_system.smk" in wf4_command
    assert "simulate_system.py" not in wf4_command


def test_every_wrapper_utterance_is_bounded_by_a_rule(tmp_path, capture_runs, capsys):
    """The rule is the wrapper's signature, and it is the whole point.

    Nested one inside the other, a wrapper block and a workflow's own block are
    the same label over the same-shaped rows — `run` over `project`/`config` in
    both — so a scrolled-back console could not be parsed into who said what.
    Every line the runner speaks first therefore sits under a rule, and nothing
    else in this console draws one.
    """
    flags = {n: "true" for n in rw.WORKFLOW_ORDER}
    flags["analyze_projections"] = "false"
    _, out, _ = _run_and_capture(tmp_path, capsys, flags)
    lines = out.splitlines()
    rule = "=" * 80
    # Opening banner (2), one per hand-off band (3 = 3 workflows, leading edge
    # only on a clean run), and the closing banner's pair. A bare count is the
    # assertion that would pass on any two extra rules, so check placement.
    # A blank line first parts the banner from the shell prompt.
    assert lines[0] == ""
    assert lines[1] == rule and lines[2] == "  run_workflows" and lines[3] == rule
    assert lines[-1] == rule
    for index, line in enumerate(lines):
        if line.startswith("  [") and "  --  " in line:
            assert lines[index - 1] == rule, f"unruled hand-off at {index}: {line}"
    # And the workflow-internal grammar never appears at this level.
    assert " - run_workflows - " not in out


def test_the_console_is_not_muted_by_the_rule_log_level(
    tmp_path, capture_runs, capsys, monkeypatch
):
    """`CST_LOG_LEVEL` quietens rule logs; the frame around them is not one.

    The wrapper deliberately does not wear `log_row`'s
    `HH:MM:SS - <module> - ...`, which every line reported from INSIDE a
    workflow wears — so it is not governed by the floor that grammar carries,
    and setting it must not delete the only statement of which project and
    which config a run was.
    """
    monkeypatch.setenv("CST_LOG_LEVEL", "WARNING")
    _, out, _ = _run_and_capture(
        tmp_path, capsys, {n: "true" for n in rw.WORKFLOW_ORDER}
    )
    assert "  run_workflows" in out.splitlines()
    assert "  sequence\n" in out
    assert re.search(
        r"\[1/5]  WF0  ANALYZE CLIMATE  --  starting at \d{4}-\d{2}-\d{2} \d\d:\d\d:\d\d",
        out,
    )
    assert "run_workflows done in" in out


def test_closing_block_names_what_ran_how_long_and_where_it_landed(
    tmp_path, capture_runs, capsys
):
    """The three answers a finished run owes: what, how long, and where."""
    flags = {n: "true" for n in rw.WORKFLOW_ORDER}
    flags["analyze_projections"] = "false"
    _, out, project_dir = _run_and_capture(tmp_path, capsys, flags)
    root = str(project_dir).replace(os.sep, "/")
    assert re.search(r"run_workflows done in \d+:\d{2}:\d{2}", out)
    wrote = _group_rows(out, "wrote")
    assert wrote["project"] == root
    # Rows under the project use the token the project row defines.
    assert wrote["logs"].startswith("<project>/logs/")
    assert wrote["logs"].count("/") == 3
    # Machine records under `_engine/` are not user-facing outputs.
    assert "invocation" not in wrote and "commands" not in wrote
    # `ran` lists what was invoked, in order, and nothing else -- a group headed
    # "ran" naming a workflow that did not is worse than not printing it.
    ran = out.split("\n  ran\n")[1].split("\n\n")[0].splitlines()
    assert [line.split()[0] for line in ran] == ["wf0", "wf1", "wf3", "wf4"]


def test_failure_console_carries_the_verdict_and_what_did_not_run(
    tmp_path, capture_runs, capsys
):
    """The stop boundary, stated: the exit code, and everything left behind."""
    _, exits = capture_runs
    exits[1] = 4  # build_model, the second invoked workflow
    code, out, _ = _run_and_capture(
        tmp_path, capsys, {n: "true" for n in rw.WORKFLOW_ORDER}
    )
    assert code == 4
    assert re.search(
        r"\[2/5\]  WF1  BUILD MODEL  --  FAILED \(exit 4\) after \d+:\d\d:\d\d", out
    )
    assert "stopping; later workflows not invoked" in out
    assert re.search(r"run_workflows FAILED in \d+:\d\d:\d\d", out)
    assert (
        "not run: wf2 analyze_projections, wf3 generate_scenarios, "
        "wf4 simulate_system" in out
    )
    assert "the failing workflow's own output is printed above" in out
    assert "exit code 4 is not one Snakemake's own CLI ever raises" in out


def test_ordinary_dag_failure_carries_no_unusual_exit_note(
    tmp_path, capture_runs, capsys
):
    """Exit 1 is Snakemake's own DAG/job-failure code -- no extra diagnosis fires.

    Snakemake's CLI exits 0 or 1 for everything it reports itself, so a plain
    ``1`` needs no cross-check against ``.snakemake/log``: the traceback or
    ``RuleException`` already printed above IS the complete account.
    """
    _, exits = capture_runs
    exits[1] = 1  # build_model, the second invoked workflow
    code, out, _ = _run_and_capture(
        tmp_path, capsys, {n: "true" for n in rw.WORKFLOW_ORDER}
    )
    assert code == 1
    assert re.search(
        r"\[2/5\]  WF1  BUILD MODEL  --  FAILED \(exit 1\) after \d+:\d\d:\d\d", out
    )
    assert "is not one Snakemake's own CLI ever raises" not in out


def test_unusual_exit_code_points_at_the_snakemake_log_for_cross_check(
    tmp_path, capture_runs, capsys
):
    """An exit code outside {0, 1} is a process-teardown clue, not a rule failure.

    A code Snakemake's own CLI never raises means whatever produced it ran
    AFTER Snakemake had already returned -- so the diagnosis names the newest
    ``.snakemake/log`` file, letting a reader check whether Snakemake's own
    account there ended cleanly despite the wrapper seeing a nonzero code.
    """
    _, exits = capture_runs
    exits[4] = 120  # simulate_system, the fifth invoked workflow
    code, out, _ = _run_and_capture(
        tmp_path, capsys, {n: "true" for n in rw.WORKFLOW_ORDER}
    )
    assert code == 120
    assert "[5/5]  WF4  SIMULATE SYSTEM  --  FAILED (exit 120) after " in out
    assert (
        "exit code 120 is not one Snakemake's own CLI ever raises (only 0 or 1)" in out
    )


def test_a_launch_error_still_closes_with_a_report(
    tmp_path, monkeypatch, capsys, capture_runs
):
    """An OSError out of subprocess.run must not end in a bare traceback.

    The manifest is finalized on this path for the same reason; the console
    report is the half a person actually reads.
    """
    project_dir = tmp_path / "project"
    cfg = tmp_path / "c.yml"
    _write_cfg(cfg, {n: "true" for n in rw.WORKFLOW_ORDER}, project_dir)

    def fake_run(cmd, cwd=None, **kwargs):
        if cmd[0] == "git":
            return FakeResult(0, stdout="abc123\n" if "rev-parse" in cmd else "")
        raise OSError("snakemake executable missing")

    monkeypatch.setattr(
        rw,
        "run_project_child",
        lambda cmd, **kwargs: fake_run(cmd, **kwargs).returncode,
    )
    with pytest.raises(OSError):
        rw.run(str(cfg), cores=3, extra=[])

    out = capsys.readouterr().out
    assert "run_workflows FAILED in" in out
    assert "wf0 analyze_climate  FAILED (OSError)" in out
    assert "not run: wf1 build_model, wf2 analyze_projections" in out
    # Startup diagnostics exist, but no child workflow log is claimed.
    assert "    logs " not in out
    assert "/summary.md" in out
    assert "printed above" not in out


def test_a_no_op_invocation_says_so_rather_than_printing_empty_groups(
    tmp_path, capture_runs, capsys
):
    """Nothing enabled: a sentence, never a group label with nothing under it."""
    _, out, _ = _run_and_capture(
        tmp_path, capsys, {n: "false" for n in rw.WORKFLOW_ORDER}
    )
    assert "nothing to invoke -- 0 of 5 workflows are enabled here" in out
    assert "nothing ran -- every workflow was disabled" in out
    assert "  ran" not in out.splitlines()
    assert "  sequence" not in out


def test_console_unicode_is_limited_to_the_reviewed_sequence_markers(
    tmp_path, capture_runs, capsys
):
    """Pending nodes, rails and disabled markers have distinct reviewed roles."""
    # One disabled workflow, so the sequence renders a hollow node as well as a
    # filled one. WHICH one is disabled is immaterial here; it is
    # `analyze_projections` rather than `build_model` only because disabling the
    # latter while `simulate_system` is enabled now trips the contract (i)
    # preflight, on a scratch project that has no wf1 leaves.
    flags = {n: "true" for n in rw.WORKFLOW_ORDER}
    flags["analyze_projections"] = "false"
    _, out, _ = _run_and_capture(tmp_path, capsys, flags)
    out.encode("utf-8")
    assert set(out) - set(map(chr, range(128))) == {
        rw._NODE_OFF,
        rw._RAIL,
        rw._NODE_DISABLED,
    }
    # The rest of the console still clears the bar it cleared before the rail.
    for line in out.splitlines():
        if not {rw._NODE_ON, rw._NODE_OFF, rw._RAIL} & set(line):
            line.encode("cp1252")


def test_the_runner_tells_its_children_the_workflow_is_already_named(
    tmp_path, capsys, monkeypatch, capture_runs
):
    """Each hand-off band names the workflow, so the child's block need not.

    On the CHILD's environment, and asserted together with the parent's
    staying clean. Setting `os.environ` in the runner covered both spawn paths
    in one line and leaked: nothing unsets it, so inside one process a single
    `run()` silenced the title for everything afterwards -- nine tests in
    `test_snake_utils.py`, visible only to the full suite because each file
    passes on its own.
    """
    monkeypatch.delenv(rw.console_style.ANNOUNCED_ENV, raising=False)
    envs = []

    def fake_run(cmd, cwd=None, **kwargs):
        if cmd[0] == "git":
            return FakeResult(0, stdout="abc123\n" if "rev-parse" in cmd else "")
        envs.append(kwargs.get("env"))
        return FakeResult(0)

    monkeypatch.setattr(
        rw,
        "run_project_child",
        lambda cmd, **kwargs: fake_run(cmd, **kwargs).returncode,
    )
    project_dir = tmp_path / "gabon_project"
    cfg = tmp_path / "c.yml"
    # `build_model` alone: it is the plain `subprocess.run` path, and enabling
    # `simulate_system` would trip the contract (i) preflight on a scratch
    # project with no wf1 leaves.
    flags = {n: "false" for n in rw.WORKFLOW_ORDER}
    flags["build_model"] = "true"
    _write_cfg(cfg, flags, project_dir=str(project_dir))
    rw.run(str(cfg), cores=3, extra=[])
    capsys.readouterr()

    assert [env[rw.console_style.ANNOUNCED_ENV] for env in envs] == ["1"]
    assert rw.console_style.ANNOUNCED_ENV not in os.environ  # never leaked


@pytest.mark.parametrize("published", [True, False])
def test_the_announcement_rides_on_the_simulation_runners_own_environment(
    tmp_path, capsys, monkeypatch, published
):
    """The other spawn path builds its own env, so the flag is added to it.

    Covered separately because this is the branch that MERGES: WF4 goes
    through `simulation_command`, whose environment carries the invocation id
    and the selected operation. Replacing it rather than adding to it would
    take those with it.
    """
    monkeypatch.delenv(rw.console_style.ANNOUNCED_ENV, raising=False)
    from blueearth_cst.experiment import indicator_export
    from blueearth_cst.shared.run_log_core import log_row

    def publication(*args):
        log_row(
            "Results saved" if published else "Publication failed",
            module="indicators",
            level="WARNING" if published else "ERROR",
        )
        return published

    monkeypatch.setattr(indicator_export, "report_publication", publication)
    envs = []

    def fake_run(cmd, cwd=None, **kwargs):
        if cmd[0] == "git":
            return FakeResult(0, stdout="abc123\n" if "rev-parse" in cmd else "")
        envs.append(kwargs.get("env"))
        log_row("Calculation warning", module="indicators", level="WARNING")
        if not kwargs["env"].get("CST_SIMULATION_FINALIZE"):
            print("wf4 simulate_system completed prematurely")
        return FakeResult(0)

    monkeypatch.setattr(
        rw,
        "run_project_child",
        lambda cmd, **kwargs: fake_run(cmd, **kwargs).returncode,
    )
    captures = []
    monkeypatch.setattr(
        simulation_record,
        "capture_simulation_sources_v2",
        lambda config, root, invocation: (
            captures.append((Path(config), Path(root), invocation))
            or tmp_path / "capture.json"
        ),
    )
    project_dir = tmp_path / "gabon_project"
    # The wf1 leaves the contract (i) preflight requires, without running wf1.
    for leaf in rw.LEAVES:
        artifact = project_dir / leaf
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.touch()
    cfg = tmp_path / "c.yml"
    flags = {n: "false" for n in rw.WORKFLOW_ORDER}
    flags["simulate_system"] = "true"
    _write_cfg(cfg, flags, project_dir=str(project_dir))
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", stream)
    assert rw.run(str(cfg), cores=3, extra=[]) == (0 if published else 1)
    output = stream.getvalue()
    verdict = "completed" if published else "FAILED"
    assert "completed prematurely" not in output
    publication_row = "Results saved" if published else "Publication failed"
    assert output.index(publication_row) < output.index(
        f"wf4 simulate_system {verdict}"
    )
    assert "2 warnings" in output
    assert output.count(f"wf4 simulate_system {verdict}") == 1
    assert ("- DONE all" in output) == published

    assert len(envs) == 1
    assert envs[0][rw.console_style.ANNOUNCED_ENV] == "1"
    assert envs[0]["CST_SIMULATION_FINALIZE"] == "1"
    # The runner's own keys survived the merge.
    assert envs[0]["CST_SIMULATION_OPERATION"] == "simulate-and-indicators"
    parent = _read_only_manifest(project_dir)
    assert (
        envs[0]["CST_SIMULATION_INVOCATION_ID"]
        == parent["children"][0]["invocation_id"]
    )
    assert captures == [
        (
            cfg.resolve(),
            project_dir.resolve(),
            parent["children"][0]["invocation_id"],
        )
    ]
    assert rw.console_style.ANNOUNCED_ENV not in os.environ


def test_the_runner_declares_utf8_on_its_own_stdout(monkeypatch):
    """`main` reconfigures before it prints, or the rail kills a redirected run.

    The glyphs are unreachable under a cp1252 locale without this call, and the
    failure is not a styling defect: `print` raises before the first workflow
    starts. Asserted on `main` rather than on a render, because that is the one
    entry point a user invokes and the only place the call can be forgotten.
    """
    seen: list[dict[str, str]] = []

    class _Stdout:
        def reconfigure(self, **kwargs):
            seen.append(kwargs)

        def write(self, _text):
            return 0

        def flush(self):
            pass

    monkeypatch.setattr(sys, "stdout", _Stdout())
    with pytest.raises(SystemExit):
        rw.main(["--help"])
    assert seen == [{"encoding": "utf-8"}]


def test_utf8_declaration_never_breaks_a_run_it_cannot_apply_to(monkeypatch):
    """A stdout without `reconfigure` costs the glyphs, never the run.

    pytest's own capture object is the common one, so this path is exercised by
    every other test in this file; it is pinned here because the alternative --
    letting the AttributeError out -- would trade a whole run for a banner.
    """

    class _Plain:
        def write(self, _text):
            return 0

        def flush(self):
            pass

    monkeypatch.setattr(sys, "stdout", _Plain())
    rw._enable_utf8_stdout()  # must not raise


def test_the_console_narration_never_leaks_a_secret(tmp_path, capture_runs, capsys):
    """Every new line that can carry `extra` goes through `sanitize_argv`."""
    # Exactly one enabled workflow, so `extra` reaches exactly one invocation.
    # `build_model` rather than `simulate_system`, for the contract (i) reason
    # given on the cp1252 test above -- the choice is immaterial to redaction.
    flags = {n: "false" for n in rw.WORKFLOW_ORDER}
    flags["build_model"] = "true"
    _, out, _ = _run_and_capture(
        tmp_path,
        capsys,
        flags,
        extra=["--config", "api_token=visible-secret", "--password", "pw-secret"],
    )
    assert "visible-secret" not in out
    assert "pw-secret" not in out
    assert "api_token=<redacted>" in out


# --- contract (i): the wf1 preflight -----------------------------------------


def _staged(project_dir: Path, leaves) -> None:
    """Create empty files at ``leaves`` under ``project_dir``."""
    for leaf in leaves:
        target = project_dir / leaf
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")


def test_wf4_without_wf1_is_refused_before_simulation_is_invoked(
    tmp_path, capture_runs
):
    """Missing model artifacts fail immediately before the WF4 consumer."""
    calls, _ = capture_runs
    project_dir = tmp_path / "project"
    cfg = tmp_path / "c.yml"
    flags = {n: "true" for n in rw.WORKFLOW_ORDER}
    flags["build_model"] = "false"
    _write_cfg(cfg, flags, project_dir=str(project_dir))

    with pytest.raises(rw.PrerequisiteError) as excinfo:
        rw.run(str(cfg), cores=3, extra=[])

    assert _snakefiles_invoked(calls) == [
        "analyze_climate.smk",
        "analyze_projections.smk",
        "generate_scenarios.smk",
    ]
    assert _read_only_manifest(project_dir)["status"] == "failed"
    message = str(excinfo.value)
    for leaf in rw.LEAVES:
        assert leaf in message, f"the message hides the missing leaf {leaf}"
    assert "build_model" in message, "the message must name the producer"


def test_the_preflight_names_only_what_is_actually_absent(tmp_path, capture_runs):
    """A partially built project reports its own two gaps, not a generic three.

    Snakemake's own failure names one leaf because rule 3.01 is merely the
    earliest to declare one -- naming the real set is the reason this reads
    `LEAVES` rather than restating paths.
    """
    project_dir = tmp_path / "project"
    cfg = tmp_path / "c.yml"
    flags = {n: "true" for n in rw.WORKFLOW_ORDER}
    flags["build_model"] = "false"
    _write_cfg(cfg, flags, project_dir=str(project_dir))
    _staged(project_dir, rw.LEAVES[:1])

    with pytest.raises(rw.PrerequisiteError) as excinfo:
        rw.run(str(cfg), cores=3, extra=[])

    message = str(excinfo.value)
    assert rw.LEAVES[0] not in message, "a present leaf is reported as missing"
    for leaf in rw.LEAVES[1:]:
        assert leaf in message
    assert "1 of 2" in message


def test_a_complete_wf1_tree_lets_wf3_run_with_build_model_disabled(
    tmp_path, capture_runs
):
    """The check is EXISTENCE, not freshness -- clause (i) against the docstring.

    Staged leaves are empty files: nothing here reads their content, and a
    check that did would be the freshness comparison the wrapper refuses to
    make.
    """
    calls, _ = capture_runs
    project_dir = tmp_path / "project"
    cfg = tmp_path / "c.yml"
    flags = {n: "true" for n in rw.WORKFLOW_ORDER}
    flags["build_model"] = "false"
    _write_cfg(cfg, flags, project_dir=str(project_dir))
    _staged(project_dir, rw.LEAVES)

    assert rw.run(str(cfg), cores=3, extra=[]) == 0
    assert "simulate_system.smk" in _snakefiles_invoked(calls)


@pytest.mark.parametrize(
    ("build_model", "simulate_system"),
    [("true", "true"), ("true", "false"), ("false", "false")],
)
def test_the_preflight_is_silent_on_every_other_flag_pair(
    tmp_path, capture_runs, build_model, simulate_system
):
    """Only wf3-enabled-without-wf1 is checked; the run produces or ignores.

    All three leaves are wf3-only -- wf2 declares none -- so no other pair can
    need them.
    """
    project_dir = tmp_path / "project"
    cfg = tmp_path / "c.yml"
    flags = {n: "true" for n in rw.WORKFLOW_ORDER}
    flags["build_model"] = build_model
    flags["simulate_system"] = simulate_system
    _write_cfg(cfg, flags, project_dir=str(project_dir))

    assert rw.run(str(cfg), cores=3, extra=[]) == 0


@pytest.mark.parametrize("extra", [[], ["--", "--dry-run"]])
def test_indicators_without_a_simulation_report_recovery(
    tmp_path, capture_runs, capsys, extra
):
    cfg = tmp_path / "project.yml"
    _write_cfg(
        cfg,
        {
            name: "true" if name == "simulate_system" else "false"
            for name in rw.WORKFLOW_ORDER
        },
    )
    (tmp_path / "project-simulation.yml").write_text(
        "experiment_name: test\noperations: [indicators]\nindicator_sets: [default]\n",
        encoding="utf-8",
    )
    assert rw.main(["--config", str(cfg), *extra]) == 2
    error = capsys.readouterr().err
    assert "No completed simulation" in error
    assert "test" in error and "_engine" in error and "simulation.json" in error
    assert "operations: [simulations, indicators]" in error
    assert "Traceback" not in error
    assert not (tmp_path / "project/experiments").exists()
    assert capture_runs[0] == []


def test_the_preflight_exits_two_through_main(tmp_path, capture_runs, capsys):
    """`main` catches PrerequisiteError beside ConfigError -- same exit 2."""
    project_dir = tmp_path / "project"
    cfg = tmp_path / "c.yml"
    flags = {n: "true" for n in rw.WORKFLOW_ORDER}
    flags["build_model"] = "false"
    _write_cfg(cfg, flags, project_dir=str(project_dir))

    assert rw.main(["--config", str(cfg)]) == 2
    assert "error:" in capsys.readouterr().err


def test_closing_block_paints_every_failed_line_red(tmp_path):
    """A failed run's verdict and its FAILED outcome row are the failure colour."""
    manifest = {
        "no_op": False,
        "workflows": {
            name: {"status": "failed" if i == 0 else "not_run", "exit_code": 1}
            for i, name in enumerate(rw.WORKFLOW_ORDER)
        },
    }
    kwargs = dict(
        project_dir=tmp_path,
        manifest_path=tmp_path / "config/_engine/invocations/x.json",
        manifest=manifest,
        ran=[(rw.WORKFLOW_ORDER[0], "FAILED (exit 1) after 0:00:01")],
        elapsed_seconds=1.0,
        failed=True,
    )
    painted = rw._closing_block(**kwargs, colour=True)
    red = f"\033[{rw._ANSI_FAIL}m"
    failed_lines = [line for line in painted.split("\n") if "FAILED" in line]
    assert len(failed_lines) == 2
    assert all(line.startswith(red) for line in failed_lines)
    assert "\033" not in rw._closing_block(**kwargs)
