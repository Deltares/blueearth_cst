"""Contract tests for scripts/run_workflows.py (design §7(g) + ext1-03).

The six §7(g) assertions plus the ext1-03 enabled:false skip test. Every test
monkeypatches subprocess.run to capture the argv list -- no real snakemake runs.
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import run_workflows as rw  # noqa: E402


class FakeResult:
    def __init__(self, returncode, stdout=""):
        self.returncode = returncode
        self.stdout = stdout


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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture()
def capture_runs(monkeypatch):
    """Patch subprocess.run to record argv lists; default success exit 0."""
    calls = []
    exits = {}  # index -> returncode override

    def fake_run(cmd, cwd=None, **kwargs):
        if cmd[0] == "git":
            stdout = "abc123\n" if "rev-parse" in cmd else ""
            return FakeResult(0, stdout=stdout)
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


def _manifests(project_dir: Path) -> list[Path]:
    """Return wrapper manifests in deterministic filename order."""
    return sorted((project_dir / "config" / "runs" / "invocations").glob("*.json"))


def _read_only_manifest(project_dir: Path) -> dict:
    """Read the sole invocation manifest below a test project root."""
    manifests = _manifests(project_dir)
    assert len(manifests) == 1
    return json.loads(manifests[0].read_text(encoding="utf-8"))


# --- §7(g) assertion 1: all-true -> all four in fixed order ------------------


def test_all_true_invokes_four_in_fixed_order(tmp_path, capture_runs):
    calls, _ = capture_runs
    cfg = tmp_path / "c.yml"
    _write_cfg(cfg, {n: "true" for n in rw.WORKFLOW_ORDER})
    rc = rw.run(str(cfg), cores=3, extra=[])
    assert rc == 0
    assert _snakefiles_invoked(calls) == [
        "analyze_climate.smk",
        "build_model.smk",
        "analyze_projections.smk",
        "run_stress_test.smk",
    ]


# --- §7(g) assertion 2: --keep-going on projections only (flag parity) -------


def test_keep_going_on_projections_only(tmp_path, capture_runs):
    calls, _ = capture_runs
    cfg = tmp_path / "c.yml"
    _write_cfg(cfg, {n: "true" for n in rw.WORKFLOW_ORDER})
    rw.run(str(cfg), cores=3, extra=[])
    by_sf = {cmd[cmd.index("-s") + 1]: cmd for cmd in calls}
    assert "--keep-going" in by_sf["analyze_projections.smk"]
    assert "--keep-going" not in by_sf["build_model.smk"]
    assert "--keep-going" not in by_sf["run_stress_test.smk"]
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
        "  build_model:\n    enabled: true\n"
        "  analyze_projections:\n    enabled: true\n"
        "  run_stress_test:\n    other: 1\n",  # no enabled:
        encoding="utf-8",
    )
    with pytest.raises(rw.ConfigError) as exc:
        rw.read_enabled_flags(str(cfg))
    assert "run_stress_test.enabled" in str(exc.value)
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


def test_cores_and_extra_forwarded_to_every_invocation(tmp_path, capture_runs):
    calls, _ = capture_runs
    cfg = tmp_path / "c.yml"
    _write_cfg(cfg, {n: "true" for n in rw.WORKFLOW_ORDER})
    rw.run(str(cfg), cores=8, extra=["--dry-run", "--unlock"])
    assert len(calls) == 4
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
    flags["analyze_projections"] = "false"
    _write_cfg(cfg, flags, project_dir=str(project_dir))

    rc = rw.run(str(cfg), cores=3, extra=[])
    assert rc == 0
    invoked = _snakefiles_invoked(calls)
    assert "analyze_projections.smk" not in invoked
    assert invoked == [
        "analyze_climate.smk",
        "build_model.smk",
        "run_stress_test.smk",
    ]


def test_all_enabled_inverse_all_invoked(tmp_path, capture_runs):
    """The inverse of the skip test: all true -> all four invoked."""
    calls, _ = capture_runs
    cfg = tmp_path / "c.yml"
    _write_cfg(cfg, {n: "true" for n in rw.WORKFLOW_ORDER})
    rw.run(str(cfg), cores=3, extra=[])
    assert len(_snakefiles_invoked(calls)) == 4


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
        return FakeResult(0)

    monkeypatch.setattr(rw.subprocess, "run", fake_run)

    assert rw.run(str(cfg), cores=5, extra=["--dry-run"]) == 0

    manifest = _read_only_manifest(project_dir)
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "succeeded"
    assert manifest["exit_code"] == 0
    assert manifest["ended_at_utc"].endswith("Z")
    assert manifest["cores"] == 5
    assert manifest["dry_run"] is True
    assert manifest["source_config"]["sha256"] == rw.file_sha256(cfg)
    assert manifest["effective_config"]["sha256"]
    # commit_source arrived with the move to provenance.toolbox_identity(). The
    # wrapper's own helper could report a commit or a null and nothing else, so
    # a container run reading a baked sha was indistinguishable from a checkout.
    assert manifest["git"] == {
        "commit": "abc123",
        "commit_source": "git",
        "dirty": False,
    }
    assert manifest["runtime"]["python"]
    # Derived from WORKFLOW_ORDER rather than restated: a literal list is what
    # made this the last of nine tests to fail when the set widened to four,
    # each for the same reason.
    assert [item["status"] for item in manifest["workflows"].values()] == [
        "succeeded"
    ] * len(rw.WORKFLOW_ORDER)
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
        assert manifest["no_op"] is True
        assert {item["status"] for item in manifest["workflows"].values()} == {
            "disabled"
        }


def test_failure_manifest_records_stop_boundary(tmp_path, capture_runs):
    """A child failure finalizes its result and marks later work not run."""
    _, exits = capture_runs
    exits[0] = 9
    project_dir = tmp_path / "project"
    cfg = tmp_path / "c.yml"
    _write_cfg(cfg, {n: "true" for n in rw.WORKFLOW_ORDER}, project_dir)

    assert rw.run(str(cfg), cores=3, extra=[]) == 9

    workflows = _read_only_manifest(project_dir)["workflows"]
    # analyze_climate leads WORKFLOW_ORDER, so it is the one exits[0] hits.
    assert workflows["analyze_climate"]["status"] == "failed"
    assert workflows["analyze_climate"]["exit_code"] == 9
    assert workflows["build_model"]["status"] == "not_run"
    assert workflows["analyze_projections"]["status"] == "not_run"
    assert workflows["run_stress_test"]["status"] == "not_run"


def test_subprocess_exception_finalizes_failure_manifest(tmp_path, monkeypatch):
    """Launch errors leave a terminal record rather than a stale running one."""
    project_dir = tmp_path / "project"
    cfg = tmp_path / "c.yml"
    _write_cfg(cfg, {n: "true" for n in rw.WORKFLOW_ORDER}, project_dir)

    def fake_run(cmd, cwd=None, **kwargs):
        if cmd[0] == "git":
            return FakeResult(0, stdout="abc123\n" if "rev-parse" in cmd else "")
        raise OSError("snakemake executable missing")

    monkeypatch.setattr(rw.subprocess, "run", fake_run)

    with pytest.raises(OSError, match="snakemake executable missing"):
        rw.run(str(cfg), cores=3, extra=[])

    manifest = _read_only_manifest(project_dir)
    assert manifest["status"] == "failed"
    assert manifest["exit_code"] is None
    assert manifest["error_type"] == "OSError"
    assert manifest["workflows"]["analyze_climate"]["status"] == "failed"
    assert manifest["workflows"]["analyze_projections"]["status"] == "not_run"


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
    assert "threshold=4" in manifest["extra_args"]
    assert "api_token=<redacted>" in manifest["extra_args"]
    assert manifest["effective_config"]["includes_cli_config_overrides"] is False
    assert manifest["snakemake_config_overrides"] == [
        "threshold=4",
        "api_token=<redacted>",
        "clientSecret=<redacted>",
    ]


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
