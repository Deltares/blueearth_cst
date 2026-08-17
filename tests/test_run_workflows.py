"""Contract tests for scripts/run_workflows.py (design §7(g) + ext1-03).

The six §7(g) assertions plus the ext1-03 enabled:false skip test. Every test
monkeypatches subprocess.run to capture the argv list -- no real snakemake runs.
"""

import json
import os
import re
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


# --- contract (h): the wrapper's own console narration ----------------------


def _group_rows(output, label):
    """The ``key -> value`` rows under one ``  <label>`` group of a block."""
    lines = output.splitlines()
    start = lines.index(f"  {label}") + 1
    rows = {}
    for line in lines[start:]:
        if not line.startswith("    "):
            break
        key, value = line.strip().split(maxsplit=1)
        rows[key] = value
    return rows


def _run_and_capture(tmp_path, capsys, flags, *, cores=3, extra=None):
    """Run the wrapper over a config with ``flags`` and return its stdout."""
    project_dir = tmp_path / "gabon_project"
    cfg = tmp_path / "c.yml"
    _write_cfg(cfg, flags, project_dir=str(project_dir))
    code = rw.run(str(cfg), cores=cores, extra=extra or [])
    return code, capsys.readouterr().out, project_dir


def test_opening_block_states_the_project_the_config_and_the_cores(
    tmp_path, capture_runs, capsys
):
    """A console of four back-to-back Snakemake runs must say whose they are.

    Snakemake's own preamble names the host and the job counts and never the
    project, so before this block a scrolled-back or pasted console could not be
    attributed to a project or a config without asking.
    """
    _, out, project_dir = _run_and_capture(
        tmp_path, capsys, {n: "true" for n in rw.WORKFLOW_ORDER}, cores=7
    )
    assert "  run_workflows" in out.splitlines()  # the banner's head line
    rows = _group_rows(out, "run")
    assert rows["project"] == "gabon_project"  # basename, no project_name key
    assert rows["folder"] == str(project_dir).replace(os.sep, "/")
    assert rows["cores"] == "7"
    assert "c.yml" in rows["config"]


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
    assert "sequence -- 3 of 4 workflows enabled, invoked in this order" in out
    assert "[1/3]  wf0 analyze_climate" in out
    assert "[2/3]  wf1 build_model" in out
    assert "[3/3]  wf3 run_stress_test" in out
    # Present, marked, and NOT given a position.
    assert "wf2 analyze_projections  (disabled, not invoked)" in out
    assert "[3/3]  wf2" not in out


def test_a_dry_run_says_so_in_the_opening_block(tmp_path, capture_runs, capsys):
    """A dry run's console is otherwise nearly identical to a real one's."""
    _, out, _ = _run_and_capture(
        tmp_path,
        capsys,
        {n: "true" for n in rw.WORKFLOW_ORDER},
        extra=["--dry-run"],
    )
    assert "mode" in out and "dry run -- nothing is executed" in out


def test_each_invoked_workflow_gets_a_hand_off_band_at_both_its_edges(
    tmp_path, capture_runs, capsys
):
    """Position, identity, when it started and how long it took."""
    flags = {n: "true" for n in rw.WORKFLOW_ORDER}
    flags["analyze_projections"] = "false"
    _, out, _ = _run_and_capture(tmp_path, capsys, flags)
    assert re.search(r"\[1/3]  wf0 analyze_climate  --  starting \d\d:\d\d:\d\d", out)
    assert "[1/3]  wf0 analyze_climate  --  done in 0:00:0" in out
    assert "[3/3]  wf3 run_stress_test  --  done in 0:00:0" in out
    # A disabled workflow gets NO band -- the sequence diagram above already
    # named it, once, before anything ran.
    assert "wf2 analyze_projections  --  starting" not in out


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
    # Opening banner (2), one per hand-off band (6 = 3 workflows x 2 edges),
    # and the closing banner's pair. A bare count is the assertion that would
    # pass on any two extra rules, so check placement instead.
    assert lines[0] == rule and lines[1] == "  run_workflows" and lines[2] == rule
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
    assert "  sequence -- 4 of 4 workflows enabled, invoked in this order" in out
    assert "  [1/4]  wf0 analyze_climate  --  done in 0:00:0" in out
    assert "run_workflows done in" in out


def test_closing_block_names_what_ran_how_long_and_where_it_landed(
    tmp_path, capture_runs, capsys
):
    """The three answers a finished run owes: what, how long, and where."""
    flags = {n: "true" for n in rw.WORKFLOW_ORDER}
    flags["analyze_projections"] = "false"
    _, out, project_dir = _run_and_capture(tmp_path, capsys, flags)
    root = str(project_dir).replace(os.sep, "/")
    assert "run_workflows done in 0:00:0" in out
    wrote = _group_rows(out, "wrote")
    assert wrote["project"] == root
    assert wrote["logs"] == f"{root}/logs/"
    assert wrote["invocation"].startswith(f"{root}/config/runs/invocations/")
    # `ran` lists what was invoked, in order, and nothing else -- a group headed
    # "ran" naming a workflow that did not is worse than not printing it.
    ran = out.split("\n  ran\n")[1].split("\n\n")[0].splitlines()
    assert [line.split()[0] for line in ran] == ["wf0", "wf1", "wf3"]


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
    assert "[2/4]  wf1 build_model  --  FAILED (exit 4) after 0:00:0" in out
    assert "stopping; later workflows not invoked" in out
    assert "run_workflows FAILED in 0:00:0" in out
    assert "not run: wf2 analyze_projections, wf3 run_stress_test" in out
    assert "the failing workflow's own output is printed above" in out


def test_a_launch_error_still_closes_with_a_report(tmp_path, monkeypatch, capsys):
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

    monkeypatch.setattr(rw.subprocess, "run", fake_run)
    with pytest.raises(OSError):
        rw.run(str(cfg), cores=3, extra=[])

    out = capsys.readouterr().out
    assert "run_workflows FAILED in" in out
    assert "wf0 analyze_climate  FAILED (OSError)" in out
    assert "not run: wf1 build_model" in out
    # Two claims that are FALSE when no child ever launched.
    assert "/logs/" not in out
    assert "printed above" not in out


def test_a_no_op_invocation_says_so_rather_than_printing_empty_groups(
    tmp_path, capture_runs, capsys
):
    """Nothing enabled: a sentence, never a group label with nothing under it."""
    _, out, _ = _run_and_capture(
        tmp_path, capsys, {n: "false" for n in rw.WORKFLOW_ORDER}
    )
    assert "nothing to invoke -- 0 of 4 workflows are enabled here" in out
    assert "nothing ran -- every workflow was disabled" in out
    assert "  ran" not in out.splitlines()
    assert "  sequence" not in out


def test_the_whole_console_is_cp1252_encodable(tmp_path, capture_runs, capsys):
    """ASCII only: a Windows console defaults to cp1252 and RAISES on the rest.

    Box-drawing characters and arrows are the natural spelling for a sequence
    diagram and are exactly what would crash the wrapper on the platform it is
    most often run from -- the same constraint `rule_banner` records.
    """
    # One disabled workflow, so the sequence diagram renders a disabled row --
    # which is where the box-drawing characters would appear. WHICH one is
    # disabled is immaterial here; it is `analyze_projections` rather than
    # `build_model` only because disabling the latter while `run_stress_test`
    # is enabled now trips the contract (i) preflight, on a scratch project
    # that has no wf1 leaves.
    flags = {n: "true" for n in rw.WORKFLOW_ORDER}
    flags["analyze_projections"] = "false"
    _, out, _ = _run_and_capture(tmp_path, capsys, flags)
    out.encode("cp1252")  # raises UnicodeEncodeError on anything outside it
    assert out.isascii(), [line for line in out.splitlines() if not line.isascii()]


def test_the_console_narration_never_leaks_a_secret(tmp_path, capture_runs, capsys):
    """Every new line that can carry `extra` goes through `sanitize_argv`."""
    # Exactly one enabled workflow, so `extra` reaches exactly one invocation.
    # `build_model` rather than `run_stress_test`, for the contract (i) reason
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


def test_wf3_without_wf1_is_refused_before_anything_is_invoked(tmp_path, capture_runs):
    """The whole point: fail in one second, not after wf0 and wf2 have run.

    Measured 2026-08-17 before this check existed -- 4:14 of which wf3 was
    0:07, because Snakemake cannot discover a missing input until the DAG for
    that workflow is built, and the wrapper builds them in order.
    """
    calls, _ = capture_runs
    project_dir = tmp_path / "project"
    cfg = tmp_path / "c.yml"
    flags = {n: "true" for n in rw.WORKFLOW_ORDER}
    flags["build_model"] = "false"
    _write_cfg(cfg, flags, project_dir=str(project_dir))

    with pytest.raises(rw.PrerequisiteError) as excinfo:
        rw.run(str(cfg), cores=3, extra=[])

    assert calls == [], "the preflight must run BEFORE the first invocation"
    assert not _manifests(project_dir), (
        "a run that cannot start must not mint an invocation record"
    )
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
    assert "2 of 3" in message


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
    assert "run_stress_test.smk" in _snakefiles_invoked(calls)


@pytest.mark.parametrize(
    ("build_model", "run_stress_test"),
    [("true", "true"), ("true", "false"), ("false", "false")],
)
def test_the_preflight_is_silent_on_every_other_flag_pair(
    tmp_path, capture_runs, build_model, run_stress_test
):
    """Only wf3-enabled-without-wf1 is checked; the run produces or ignores.

    All three leaves are wf3-only -- wf2 declares none -- so no other pair can
    need them.
    """
    project_dir = tmp_path / "project"
    cfg = tmp_path / "c.yml"
    flags = {n: "true" for n in rw.WORKFLOW_ORDER}
    flags["build_model"] = build_model
    flags["run_stress_test"] = run_stress_test
    _write_cfg(cfg, flags, project_dir=str(project_dir))

    assert rw.run(str(cfg), cores=3, extra=[]) == 0


def test_the_preflight_exits_two_through_main(tmp_path, capture_runs, capsys):
    """`main` catches PrerequisiteError beside ConfigError -- same exit 2."""
    project_dir = tmp_path / "project"
    cfg = tmp_path / "c.yml"
    flags = {n: "true" for n in rw.WORKFLOW_ORDER}
    flags["build_model"] = "false"
    _write_cfg(cfg, flags, project_dir=str(project_dir))

    assert rw.main(["--config", str(cfg)]) == 2
    assert "error:" in capsys.readouterr().err
