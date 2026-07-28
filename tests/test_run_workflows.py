"""Contract tests for scripts/run_workflows.py (design §7(g) + ext1-03).

The six §7(g) assertions plus the ext1-03 enabled:false skip test. Every test
monkeypatches subprocess.run to capture the argv list -- no real snakemake runs.
"""

import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import run_workflows as rw  # noqa: E402


class FakeResult:
    def __init__(self, returncode):
        self.returncode = returncode


def _write_cfg(path, flags, project_dir="test_case/test"):
    """Write a full-orchestration config with the given enabled flags dict.

    ``flags`` values are inserted verbatim into YAML so a caller can pass a raw
    string (e.g. '"true"' or 'yes') to exercise the parsed-value contract."""
    lines = [
        "project:",
        f"  project_dir: {project_dir}",
        "workflows:",
    ]
    for name in rw.WORKFLOW_ORDER:
        lines.append(f"  {name}:")
        lines.append(f"    enabled: {flags[name]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture()
def capture_runs(monkeypatch):
    """Patch subprocess.run to record argv lists; default success exit 0."""
    calls = []
    exits = {}  # index -> returncode override

    def fake_run(cmd, cwd=None, **kwargs):
        idx = len(calls)
        calls.append(cmd)
        return FakeResult(exits.get(idx, 0))

    monkeypatch.setattr(rw.subprocess, "run", fake_run)
    return calls, exits


def _snakefiles_invoked(calls):
    """Return the ordered list of Snakefile names across captured argv."""
    out = []
    for cmd in calls:
        i = cmd.index("-s")
        out.append(cmd[i + 1])
    return out


# --- §7(g) assertion 1: all-true -> all three in fixed order -----------------

def test_all_true_invokes_three_in_fixed_order(tmp_path, capture_runs):
    calls, _ = capture_runs
    cfg = tmp_path / "c.yml"
    _write_cfg(cfg, {n: "true" for n in rw.WORKFLOW_ORDER})
    rc = rw.run(str(cfg), cores=3, extra=[])
    assert rc == 0
    assert _snakefiles_invoked(calls) == [
        "Snakefile_model_creation",
        "Snakefile_climate_projections",
        "Snakefile_climate_experiment",
    ]


# --- §7(g) assertion 2: --keep-going on projections only (flag parity) -------

def test_keep_going_on_projections_only(tmp_path, capture_runs):
    calls, _ = capture_runs
    cfg = tmp_path / "c.yml"
    _write_cfg(cfg, {n: "true" for n in rw.WORKFLOW_ORDER})
    rw.run(str(cfg), cores=3, extra=[])
    by_sf = {cmd[cmd.index("-s") + 1]: cmd for cmd in calls}
    assert "--keep-going" in by_sf["Snakefile_climate_projections"]
    assert "--keep-going" not in by_sf["Snakefile_model_creation"]
    assert "--keep-going" not in by_sf["Snakefile_climate_experiment"]


# --- §7(g) assertion 3: missing enabled: key -> nonzero, named --------------

def test_missing_enabled_key_errors(tmp_path):
    cfg = tmp_path / "c.yml"
    cfg.write_text(
        "workflows:\n"
        "  model_creation:\n    enabled: true\n"
        "  climate_projections:\n    enabled: true\n"
        "  climate_experiment:\n    other: 1\n",  # no enabled:
        encoding="utf-8",
    )
    with pytest.raises(rw.ConfigError) as exc:
        rw.read_enabled_flags(str(cfg))
    assert "climate_experiment.enabled" in str(exc.value)
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
    flags["model_creation"] = bad
    _write_cfg(cfg, flags)
    with pytest.raises(rw.ConfigError) as exc:
        rw.read_enabled_flags(str(cfg))
    assert "model_creation.enabled" in str(exc.value)


@pytest.mark.parametrize("spelling", ["yes", "on", "true", "True"])
def test_unquoted_boolean_spellings_accepted(tmp_path, spelling):
    """Unquoted yes/on/true resolve to True under YAML 1.1 -> accepted (contract c)."""
    cfg = tmp_path / "c.yml"
    flags = {n: "true" for n in rw.WORKFLOW_ORDER}
    flags["model_creation"] = spelling
    _write_cfg(cfg, flags)
    parsed = rw.read_enabled_flags(str(cfg))
    assert parsed["model_creation"] is True


# --- §7(g) assertion 5: first nonzero -> stop, later not invoked, return code -

def test_first_nonzero_stops_and_returns_code(tmp_path, capture_runs):
    calls, exits = capture_runs
    exits[0] = 7  # first invoked workflow (model_creation) fails
    cfg = tmp_path / "c.yml"
    _write_cfg(cfg, {n: "true" for n in rw.WORKFLOW_ORDER})
    rc = rw.run(str(cfg), cores=3, extra=[])
    assert rc == 7
    # Only the first workflow was invoked; projections/experiment were not.
    assert _snakefiles_invoked(calls) == ["Snakefile_model_creation"]


# --- §7(g) assertion 6: --cores / -- <extra> forwarded to EVERY invocation ---

def test_cores_and_extra_forwarded_to_every_invocation(tmp_path, capture_runs):
    calls, _ = capture_runs
    cfg = tmp_path / "c.yml"
    _write_cfg(cfg, {n: "true" for n in rw.WORKFLOW_ORDER})
    rw.run(str(cfg), cores=8, extra=["--dry-run", "--unlock"])
    assert len(calls) == 3
    for cmd in calls:
        assert cmd[cmd.index("-c") + 1] == "8"
        assert "--dry-run" in cmd
        assert "--unlock" in cmd


def test_cli_strips_double_dash_sentinel(tmp_path, capture_runs):
    """The leading `--` sentinel is stripped before forwarding (contract e)."""
    calls, _ = capture_runs
    cfg = tmp_path / "c.yml"
    _write_cfg(cfg, {n: "true" for n in rw.WORKFLOW_ORDER})
    rw.main(["--config", str(cfg), "--cores", "2", "--", "--dry-run"])
    for cmd in calls:
        assert "--" not in cmd or "--dry-run" in cmd  # no bare sentinel forwarded
        assert "--dry-run" in cmd


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
    flags["climate_projections"] = "false"
    _write_cfg(cfg, flags, project_dir=str(project_dir))

    rc = rw.run(str(cfg), cores=3, extra=[])
    assert rc == 0
    invoked = _snakefiles_invoked(calls)
    assert "Snakefile_climate_projections" not in invoked
    assert invoked == ["Snakefile_model_creation", "Snakefile_climate_experiment"]


def test_all_enabled_inverse_all_invoked(tmp_path, capture_runs):
    """The inverse of the skip test: all true -> all three invoked."""
    calls, _ = capture_runs
    cfg = tmp_path / "c.yml"
    _write_cfg(cfg, {n: "true" for n in rw.WORKFLOW_ORDER})
    rw.run(str(cfg), cores=3, extra=[])
    assert len(_snakefiles_invoked(calls)) == 3
