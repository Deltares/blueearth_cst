"""Unit tests for the `--workflow` scope filter on check_baseline.py.

The filter (design ext1-1/ext2-1) tags each `TARGETS` entry with its owning
workflow. On `check --workflow`, it builds ONE selected path universe and applies
it symmetrically to both the recorded manifest and the current on-disk targets
before the missing/diff/orphan/count logic. On `record --workflow` (added for
ADR 0001 step 7), it records ONLY the selected workflow(s) and MERGES into the
existing manifest — the other workflows' rows are preserved, never clobbered — so
a wf1-slice re-record does not drag in a wf2/wf3 run.

The fixture builds a synthetic project dir under `tmp_path` with a valid file for
every real `TARGETS` template (including the workflow-1 discharge `output.csv`),
then records a manifest against it via `cmd_record` — the same code path the tool
uses, so path keys match `resolve()` byte-for-byte and the discharge reference
series lands under `<manifest_dir>/discharge_ref/`. No real Snakemake run.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev" / "scripts"))
import check_baseline as cb  # noqa: E402


def _write_indicator_plan_fixture(project_dir):
    """Synthetic ready-plan linkage; numeric target payloads are written separately."""
    from blueearth_cst.experiment.content_identity import (
        canonical_json_bytes,
        content_sha256,
        identity_segment,
    )

    root = Path(project_dir) / "experiments" / cb.EXPERIMENT_NAME / "results"
    request = {"fixture": "indicator request"}
    request_id = content_sha256(request)
    set_id = "a" * 64
    plan = {
        "schema_version": "indicator-request/3",
        "request": request,
        "indicator_request_id": request_id,
        "indicator_set_id": set_id,
        "response_inventory_sha256": "b" * 64,
    }
    plan["request_sha256"] = content_sha256(plan)
    path = (
        root.parent
        / "_engine"
        / "indicator_requests"
        / f"{identity_segment(request_id, 'indicator_request_id')}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(plan))
    marker = {
        "schema_version": "indicator-set/3",
        "status": "ready",
        "indicator_set_id": set_id,
        "response_inventory": {"sha256": "b" * 64},
        "result_root": f"results/indicator_sets/{set_id[:12]}",
    }
    marker["indicators_manifest_sha256"] = content_sha256(marker)
    target = (
        root.parent
        / "_engine"
        / "indicator_sets"
        / identity_segment(set_id, "indicator_set_id")
        / "indicators.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_json_bytes(marker))
    return path, target


def _write_target(path: str, kind: str) -> None:
    """Write a minimal but valid file for the given target kind."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if kind == "nc":
        import xarray as xr

        xr.Dataset({"v": (("t",), np.array([1.0, 2.0, 3.0]))}).to_netcdf(p)
    elif kind == "csv":
        p.write_text("a,b\n1,2\n")
    elif kind in ("yaml", "run_record"):
        p.write_text("loaded_config:\n  project:\n    project_dir: synthetic\n")
    elif kind == "png":
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    elif kind == "discharge":
        rows = ["time,Q_synthetic"]
        rows += [f"2000-01-{i + 2:02d}T00:00:00,{1.0 + 0.1 * i!r}" for i in range(10)]
        p.write_text("\n".join(rows) + "\n")
    elif kind == "indicator":
        # A long indicator table (R11 Q8). Two groups so the per-group tolerance
        # has something to group by; the comparator's own cases live in
        # tests/test_check_baseline_indicator.py.
        scope = "run" if p.name.endswith("_run_indicators.csv") else "group"
        rows = [f"indicator,location,{scope}_id,value"]
        for indicator, base in (("q_mean", 10.0), ("q_low", 0.5)):
            rows += [
                f"{indicator},101,{scope}_{i + 1:02d},{base * (1.0 + 0.1 * i)!r}"
                for i in range(5)
            ]
        p.write_text("\n".join(rows) + "\n")
    else:  # pragma: no cover - guard against a new untested kind
        raise ValueError(f"unhandled kind: {kind}")


def test_baseline_indicator_plan_refuses_legacy_fallback_and_ambiguity(tmp_path):
    legacy = tmp_path / "experiments" / cb.EXPERIMENT_NAME / "results/q_indicators.csv"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("indicator,location,st_id,rlz_id,value\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one retained indicator plan"):
        cb.resolve_indicator_set_dir(str(tmp_path))
    plan, marker = _write_indicator_plan_fixture(str(tmp_path))
    # The marker is in the engine bin; the gate compares the paired tables.
    experiment = marker.parents[3]
    tables = experiment / "results" / "indicator_sets" / marker.parent.name
    assert cb.resolve_indicator_set_dir(str(tmp_path)) == tables.as_posix()
    another = plan.parent / ("c" * 12 + ".json")
    another.write_bytes(plan.read_bytes())
    with pytest.raises(ValueError, match="found 2"):
        cb.resolve_indicator_set_dir(str(tmp_path))


@pytest.mark.parametrize("artifact", ["plan", "marker"])
def test_baseline_reader_refuses_legacy_schema_even_with_valid_digest(
    tmp_path, artifact
):
    from blueearth_cst.experiment.content_identity import (
        canonical_json_bytes,
        content_sha256,
    )

    plan, marker = _write_indicator_plan_fixture(str(tmp_path))
    path = plan if artifact == "plan" else marker
    document = json.loads(path.read_text(encoding="utf-8"))
    document["schema_version"] = (
        "indicator-request/2" if artifact == "plan" else "indicator-set/2"
    )
    digest_key = (
        "request_sha256" if artifact == "plan" else "indicators_manifest_sha256"
    )
    document[digest_key] = content_sha256(
        {k: v for k, v in document.items() if k != digest_key}
    )
    path.write_bytes(canonical_json_bytes(document))
    with pytest.raises(cb.BaselineFixtureError, match="regenerate the legacy fixture"):
        cb.resolve_indicator_set_dir(str(tmp_path))


def test_baseline_indicator_plan_refuses_a_filename_that_is_not_its_identity(tmp_path):
    """The identity assertion survived flattening as a STEM check, not a path one.

    Before t2609152104 the directory name carried the identity and was compared
    against the `indicator_request_id` inside the file. Flattened, the stem carries
    it. Dropping that comparison rather than rewriting it would have left a
    discovered file with nothing tying it to the identity it claims.
    """
    plan, _ = _write_indicator_plan_fixture(str(tmp_path))
    plan.rename(plan.with_name("0" * 12 + ".json"))
    with pytest.raises(ValueError, match="digest or request identity differs"):
        cb.resolve_indicator_set_dir(str(tmp_path))


def test_baseline_indicator_plan_refuses_changed_ready_marker(tmp_path):
    _, marker = _write_indicator_plan_fixture(str(tmp_path))
    doc = json.loads(marker.read_text())
    doc["response_inventory"]["sha256"] = "c" * 64
    from blueearth_cst.experiment.content_identity import canonical_json_bytes

    marker.write_bytes(canonical_json_bytes(doc))
    with pytest.raises(ValueError, match="marker differs"):
        cb.resolve_indicator_set_dir(str(tmp_path))


def _record_ns(project_dir, manifest_path, workflow=None, include_figures=False):
    return argparse.Namespace(
        cmd="record",
        project_dir=project_dir,
        manifest=manifest_path,
        workflow=workflow,
        include_figures=include_figures,
    )


def _check_ns(
    project_dir, manifest_path, workflow=None, tolerance=0.0, include_figures=False
):
    return argparse.Namespace(
        cmd="check",
        project_dir=project_dir,
        manifest=manifest_path,
        tolerance=tolerance,
        workflow=workflow,
        include_figures=include_figures,
    )


@pytest.fixture
def project(tmp_path):
    """A synthetic project dir with all 11 targets present + a recorded manifest.

    Returns (project_dir, manifest_path). Both point under tmp_path.
    """
    project_dir = str(tmp_path)
    _write_indicator_plan_fixture(project_dir)
    for _workflow, kind, template in cb.TARGETS:
        _write_target(cb.resolve(template, project_dir), kind)

    manifest_path = tmp_path / "manifest.json"
    # Recorded WITH figures so the fixture still exercises every target kind;
    # the default-exclusion behaviour gets its own tests below.
    rc = cb.cmd_record(_record_ns(project_dir, manifest_path, include_figures=True))
    assert rc == 0  # fixture sanity: all synthetic targets recorded
    return project_dir, manifest_path


def test_targets_tagged_with_expected_cardinality():
    """The shipping TARGETS carry the 4/5/2 workflow tags the count math relies on.

    `build_model` gained the beyond-`rule all` discharge target, then dropped
    one again on 2026-08-10: the evaluation hydrograph is keyed by `wflow_id`
    now, so no per-station figure has a config-invariant NAME and none can be a
    template target. Its `png` kind is still exercised by `elevation_basin.png` and
    `forcing_precip_map.png`, and the run's NUMBERS are covered by `output.csv`
    and `performance_indicators.csv`, which the baseline does track.
    Historically, `simulate_system` dropped from 3 to 2 at R11 CR-2: `basin_indicators.csv`
    no longer exists, and the seed config declares only `river discharge`, so it
    emits one indicator table. A project configuring more output variables gets
    more tables — but this list describes the SEED tree, which is why the number
    is pinned here rather than derived. It dropped to 1 on 2026-09-17 when
    `simulation.json` was removed (`t2609171739`); see the test below.
    Split run/group contracts now require two numeric targets.
    """
    counts = Counter(workflow for workflow, _kind, _template in cb.TARGETS)
    assert counts == {
        "build_model": 4,
        "analyze_projections": 5,
        "simulate_system": 2,
    }


def test_simulation_json_is_never_a_target():
    """It records an absolute path and a code fingerprint, so it cannot pass.

    A gate entry that differs by worktree and by branch BY CONSTRUCTION can only
    ever pass in the checkout that recorded it; everywhere else it is a standing
    FAIL that says nothing about the tree, which trains the reader to discount
    real failures. Measured before removal: 7 of 20 leaf keys differed between
    two successor trees and none was result-bearing, while `q_group_indicators.csv`
    over those same two trees was exactly equal.

    This is a negative assertion on purpose. The file is a reasonable-LOOKING
    target -- it is the experiment's frozen provenance document -- so the reason
    it is excluded has to live somewhere a re-adder will trip over.
    """
    assert not [
        template
        for _workflow, _kind, template in cb.TARGETS
        if "simulation" in template
    ]


def test_scoped_count_is_selected_not_full(project, capsys):
    """`--workflow build_model --workflow analyze_projections` reports nine
    of the eleven targets, not the full set."""
    project_dir, manifest_path = project
    rc = cb.cmd_check(
        _check_ns(
            project_dir,
            manifest_path,
            workflow=["build_model", "analyze_projections"],
            include_figures=True,
        )
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK - 9 target(s)" in out


def test_selected_missing_target_fails(project, capsys):
    """A selected target missing on disk -> non-zero, and named."""
    project_dir, manifest_path = project
    victim = cb.resolve(
        "{clim_project_dir}/summary/{clim_project}_change_factors_annual.csv",
        project_dir,
    )
    Path(victim).unlink()

    rc = cb.cmd_check(
        _check_ns(
            project_dir,
            manifest_path,
            workflow=["build_model", "analyze_projections"],
            include_figures=True,
        )
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert victim in out


def test_unselected_missing_target_ignored(project, capsys):
    """An unselected (workflow-3) target missing on disk is ignored by a scoped
    check -> returns 0, count stays nine."""
    project_dir, manifest_path = project
    victim = cb.resolve("{indicator_set_dir}/q_group_indicators.csv", project_dir)
    Path(victim).unlink()

    rc = cb.cmd_check(
        _check_ns(
            project_dir,
            manifest_path,
            workflow=["build_model", "analyze_projections"],
            include_figures=True,
        )
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK - 9 target(s)" in out


def test_unscoped_record_writes_all_targets(project):
    """An unscoped record with --include-figures writes all eleven (overwrite)."""
    project_dir, manifest_path = project
    written = json.loads(Path(manifest_path).read_text())
    assert len(written["targets"]) == 11
    assert written["version"] == cb.MANIFEST_VERSION


def test_record_workflow_merges_and_preserves_other_slices(project):
    """`record --workflow build_model` re-records only wf1 rows and preserves
    the wf2/wf3 rows verbatim (ADR step 7 merge semantics — no silent clobber)."""
    project_dir, manifest_path = project
    before = json.loads(Path(manifest_path).read_text())["targets"]

    cp_path = cb.resolve(
        "{clim_project_dir}/summary/{clim_project}_change_factors_annual.csv",
        project_dir,
    )
    exp_path = cb.resolve("{indicator_set_dir}/q_group_indicators.csv", project_dir)
    cp_before, exp_before = before[cp_path], before[exp_path]

    # Mutate a wf1 target AND a wf2 target on disk; then merge-record only wf1.
    # A correct merge must re-record wf1 yet leave the (now stale) wf2 row as it
    # was — proving it did not recompute or drop the unselected slice.
    disch_path = cb.resolve(
        "{project_dir}/models/hydrology/wflow/run_default/output.csv", project_dir
    )
    Path(cp_path).write_text("a,b\n9,9\n")  # wf2 content now differs from cp_before
    Path(disch_path).write_text(
        "time,Q_synthetic\n"
        + "\n".join(f"2000-01-{i + 2:02d}T00:00:00,{5.0 + i!r}" for i in range(10))
        + "\n"
    )

    rc = cb.cmd_record(_record_ns(project_dir, manifest_path, workflow=["build_model"]))
    assert rc == 0
    after = json.loads(Path(manifest_path).read_text())["targets"]

    assert len(after) == 11  # nothing dropped
    assert after[cp_path] == cp_before  # wf2 row preserved verbatim
    assert after[exp_path] == exp_before  # wf3 row preserved verbatim
    # wf1 discharge row re-recorded against the mutated series.
    assert after[disch_path]["type"] == "discharge"
    assert after[disch_path]["mean_ref"] != before[disch_path]["mean_ref"]


def test_unscoped_check_spans_all_targets(project, capsys):
    """`check --include-figures` with no `--workflow` spans all 11 targets."""
    project_dir, manifest_path = project
    rc = cb.cmd_check(
        _check_ns(project_dir, manifest_path, workflow=None, include_figures=True)
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK - 11 target(s)" in out


# --- figure targets are excluded by default (2026-08-03) ----------------------
# A figure is a terminal artifact and is fingerprinted by byte SIZE, so any
# cosmetic edit reddens the gate without indicating a defect.


def test_figure_targets_are_excluded_by_default():
    """The default universe drops every FIGURE_KINDS row and keeps the rest."""
    default = cb.active_targets()
    assert default, "the filter must not empty the target list"
    assert all(kind not in cb.FIGURE_KINDS for _w, kind, _t in default)
    assert len(default) == len(cb.TARGETS) - sum(
        1 for _w, kind, _t in cb.TARGETS if kind in cb.FIGURE_KINDS
    )


def test_include_figures_restores_the_full_universe():
    assert cb.active_targets(include_figures=True) == cb.TARGETS


def test_projection_figure_targets_match_baseline_analysis_window():
    import yaml

    from blueearth_cst.projections.get_change_climate_proj import analysis_windows

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "test_case/project_config_baseline_analyze_projections.yml").read_text()
    )
    horizon = config["future_windows"][0]
    window = analysis_windows({horizon["name"]: (horizon["start"], horizon["end"])})[
        horizon["name"]
    ]["window_id"]
    figures = {
        template
        for workflow, kind, template in cb.TARGETS
        if workflow == "analyze_projections" and kind in cb.FIGURE_KINDS
    }
    assert figures == {
        "{clim_project_dir}/plots/full_period/cmip6_precip_annual_overview_basin_avg.png",
        "{clim_project_dir}/plots/period_"
        + window.replace("-", "_")
        + "/cmip6_precip_temp_annual_change_cloud_basin_avg.png",
    }
    assert [kind for _, kind, _ in cb.active_targets({"analyze_projections"})] == [
        "csv",
        "csv",
        "run_record",
    ]


def test_projection_console_example_uses_suite_contract(monkeypatch):
    from console_sample_specs import WF2
    from render_console_sample import render

    from blueearth_cst.shared import console_style as cs
    from blueearth_cst.shared import snake_utils as su

    # The standalone renderer stages process-wide console state. Keep this
    # inspection local so later handler tests cannot inherit its workflow.
    monkeypatch.setattr(cs, "_RUN_HEADER", None)
    monkeypatch.setattr(cs, "_RULE_NUMBERS", {})
    monkeypatch.setattr(cs, "_RULE_SUMMARIES", {})
    monkeypatch.setenv(su._PATH_TOKENS_ENV, "")
    assert any(
        rule.number == "2.05b" and rule.name == "prepare_projection_analysis"
        for rule in WF2.rules
    )
    assert any(job.rule == "prepare_projection_analysis" for job in WF2.jobs)
    assert any(path.endswith("plots/figure_captions.md") for path in WF2.targets)
    assert any(path.endswith("_engine/figure_captions.json") for path in WF2.targets)
    transcript = render(WF2)
    assert "14 figures written" in transcript
    assert "climate_projections/figures" not in transcript


def test_the_workflow_and_figure_filters_compose():
    scoped = cb.active_targets({"build_model"})
    assert {w for w, _k, _t in scoped} == {"build_model"}
    assert all(kind not in cb.FIGURE_KINDS for _w, kind, _t in scoped)


def test_default_check_ignores_recorded_figure_rows(project, capsys):
    """A manifest recorded WITH figures still checks clean without them.

    The recorded side is filtered to the same universe, so the stale rows are
    neither compared nor counted -- rather than being reported as missing.
    """
    project_dir, manifest_path = project
    rc = cb.cmd_check(_check_ns(project_dir, manifest_path))
    out = capsys.readouterr().out
    assert rc == 0, out
    figures = sum(1 for _w, kind, _t in cb.TARGETS if kind in cb.FIGURE_KINDS)
    assert f"OK - {len(cb.TARGETS) - figures} target(s)" in out


def test_a_changed_figure_does_not_fail_the_default_check(project, capsys):
    """The whole point: repaint a figure, gate stays green."""
    project_dir, manifest_path = project
    for _workflow, kind, template in cb.TARGETS:
        if kind in cb.FIGURE_KINDS:
            path = pathlib.Path(cb.resolve(template, project_dir))
            path.write_bytes(path.read_bytes() + b"x" * 5000)

    assert cb.cmd_check(_check_ns(project_dir, manifest_path)) == 0
    capsys.readouterr()
    # ... and --include-figures still catches it, so the signal is not lost.
    rc = cb.cmd_check(_check_ns(project_dir, manifest_path, include_figures=True))
    assert rc == 1
    assert "size" in capsys.readouterr().out


# --------------------------------------------------------------------------
# "The gate did not run" is not "the gate passed" (2026-09-17).
#
# Two conditions used to be reported as something other than a failure: an
# empty scope printed `OK - 0 target(s) match manifest`, and an unusable
# fixture raised a traceback out of `main`. Both mean the same thing to a
# caller -- this run is not evidence -- and both now say so and exit 2.
# --------------------------------------------------------------------------


def test_an_empty_scope_is_not_a_pass(project, capsys):
    """`generate_scenarios` declares no non-figure target, so nothing compares.

    The old output was `OK - 0 target(s) match manifest`, which is the shape of
    pass this repo refuses everywhere else: a tool that bounds its own coverage
    has to say what it dropped.
    """
    project_dir, manifest_path = project

    rc = cb.cmd_check(
        _check_ns(project_dir, manifest_path, workflow=["generate_scenarios"])
    )

    out = capsys.readouterr().out
    assert rc == cb.EXIT_NOT_CHECKED
    assert rc != 0
    assert "NOT CHECKED" in out
    assert "generate_scenarios" in out
    assert "says nothing about the tree" in out
    assert "OK -" not in out


def test_an_empty_scope_names_the_figure_exclusion(project, capsys):
    """Figures are excluded by default, which is WHY the scope came out empty.

    A reader who does not know that reads the message as a broken tool.
    """
    project_dir, manifest_path = project

    cb.cmd_check(_check_ns(project_dir, manifest_path, workflow=["generate_scenarios"]))

    assert "figures excluded" in capsys.readouterr().out


def test_a_populated_scope_still_passes_at_zero(project, capsys):
    """The guard keys on an empty TARGET SET, never on a zero failure count."""
    project_dir, manifest_path = project

    rc = cb.cmd_check(_check_ns(project_dir, manifest_path, workflow=["build_model"]))

    out = capsys.readouterr().out
    assert rc == 0
    assert "OK -" in out
    assert "NOT CHECKED" not in out


def test_an_unusable_indicator_fixture_raises_a_typed_error(project, tmp_path):
    """`resolve` fails while RESOLVING, before any comparison runs.

    Typed so `main` can turn it into a diagnosis; a `ValueError` subclass so a
    caller already catching `ValueError` around resolution keeps working.
    """
    project_dir, _manifest_path = project
    plans = Path(project_dir) / "experiments" / cb.EXPERIMENT_NAME / "_engine"
    for plan in (plans / "indicator_requests").glob("*.json"):
        plan.unlink()

    with pytest.raises(cb.BaselineFixtureError):
        cb.resolve("{indicator_set_dir}/q_group_indicators.csv", project_dir)

    assert issubclass(cb.BaselineFixtureError, ValueError)


def test_the_three_exit_codes_are_distinct():
    """0 passed, 1 the tree differs, 2 nothing was checked.

    A caller that only tests `rc != 0` cannot tell a red gate from one that
    never ran, which is the confusion this whole block exists to remove.
    """
    assert cb.EXIT_NOT_CHECKED == 2
    assert cb.EXIT_NOT_CHECKED not in (0, 1)


# --- orphaned manifest rows (t2609171743) ----------------------------------
#
# A recorded row whose path no longer resolves used to be dropped by the
# `p in in_scope_paths` filter with nothing said. That is how the real manifest
# stopped comparing `q_group_indicators.csv`: `t2609152107` shortened every
# content-digest segment to twelve characters, the 64-hex row stopped matching,
# and `simulate_system` went on reporting a verdict computed from one target.
# The row is not out of scope -- it BELONGS to the scope and cannot be reached,
# which is a silent cap, not a filter.


def _orphan_the_indicator_row(manifest_path, keep_template=False):
    """Re-key the recorded indicator row under a longer digest segment.

    Mirrors the real defect exactly: same template, same directory, a segment of
    a different length. Returns the re-keyed path.

    `keep_template=False` also STRIPS the row's `template`, which is what makes
    it a genuine orphan -- it models a manifest recorded before 2026-09-17, the
    only kind that can still be orphaned. With the template kept, the row is
    re-keyed at check time and found; that is the fix, and it has its own test.
    """
    manifest = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))
    key = next(k for k in manifest["targets"] if k.endswith("q_group_indicators.csv"))
    parts = key.split("/")
    parts[-2] = "a" * 64
    orphan = "/".join(parts)
    row = manifest["targets"].pop(key)
    if not keep_template:
        row.pop("template", None)
    manifest["targets"][orphan] = row
    pathlib.Path(manifest_path).write_text(json.dumps(manifest), encoding="utf-8")
    return orphan


def test_a_renamed_identity_is_followed_not_orphaned(project, capsys):
    """THE FIX (t2609171743): a row carrying `template` survives a rename.

    The row is re-keyed to a path that does not exist, exactly as the 12-hex
    move did to the real manifest -- but it still carries the template it was
    recorded from, so `rekey_by_template` resolves it back to today's path and
    the comparison happens. No orphan, no re-record, no manual repointing.
    """
    project_dir, manifest_path = project
    orphan = _orphan_the_indicator_row(manifest_path, keep_template=True)

    rc = cb.cmd_check(
        _check_ns(project_dir, manifest_path, workflow=["simulate_system"])
    )

    out = capsys.readouterr().out
    assert rc == 0, out
    assert "OK -" in out
    assert "orphaned" not in out
    assert orphan not in out


def test_the_template_key_wins_over_the_literal_key(project):
    """Precedence is pinned, not emergent: `template` decides, the key is ignored."""
    project_dir, manifest_path = project
    _orphan_the_indicator_row(manifest_path, keep_template=True)

    recorded = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))
    rekeyed = cb.rekey_by_template(recorded["targets"], project_dir)

    resolved = cb.resolve("{indicator_set_dir}/q_group_indicators.csv", project_dir)
    assert resolved in rekeyed
    assert not [k for k in rekeyed if "a" * 64 in k]


def test_an_orphaned_row_fails_instead_of_vanishing(project, capsys):
    project_dir, manifest_path = project
    orphan = _orphan_the_indicator_row(manifest_path)

    rc = cb.cmd_check(
        _check_ns(project_dir, manifest_path, workflow=["simulate_system"])
    )

    out = capsys.readouterr().out
    assert rc == 1
    assert orphan in out
    assert "orphaned" in out
    assert "NOT compared" in out


def test_an_orphaned_row_is_never_reported_as_a_pass(project, capsys):
    """The whole point: absence must not read as agreement."""
    project_dir, manifest_path = project
    _orphan_the_indicator_row(manifest_path)

    rc = cb.cmd_check(
        _check_ns(project_dir, manifest_path, workflow=["simulate_system"])
    )

    out = capsys.readouterr().out
    assert rc != 0
    assert "OK -" not in out


def test_other_scopes_are_not_disturbed_by_an_orphan(project, capsys):
    """The orphan belongs to simulate_system; build_model must still pass.

    Guards the obvious over-correction -- reporting every unmatched manifest row
    regardless of scope would fail every scoped run on another scope's staleness.
    """
    project_dir, manifest_path = project
    _orphan_the_indicator_row(manifest_path)

    rc = cb.cmd_check(_check_ns(project_dir, manifest_path, workflow=["build_model"]))

    assert rc == 0
    assert "OK -" in capsys.readouterr().out


def test_a_pattern_accepts_any_segment_length(project):
    """The matcher must recognise the OLD length, which is the case it exists for."""
    project_dir, _ = project
    pattern = cb.target_pattern(
        "{indicator_set_dir}/q_group_indicators.csv", project_dir
    )
    # Posix-normalised, as the matcher normalises both sides -- on Windows the
    # recorded keys mix separators (see `target_pattern`).
    root = cb._posix(project_dir)
    base = f"{root}/experiments/{cb.EXPERIMENT_NAME}/results/indicator_sets"

    assert pattern.match(f"{base}/{'a' * 64}/q_group_indicators.csv")
    assert pattern.match(f"{base}/{'a' * 12}/q_group_indicators.csv")
    assert not pattern.match(f"{base}/{'a' * 12}/other.csv")
    internal = f"{root}/experiments/{cb.EXPERIMENT_NAME}/_engine/indicator_sets"
    assert pattern.match(f"{internal}/{'a' * 12}/data/q_group_indicators.csv")
    assert not pattern.match(f"{internal}/{'a' * 12}/q_group_indicators.csv")


def test_baseline_resolves_internal_indicator_payloads(tmp_path):
    from blueearth_cst.experiment.content_identity import (
        canonical_json_bytes,
        content_sha256,
    )

    _, path = _write_indicator_plan_fixture(str(tmp_path))
    marker = json.loads(path.read_bytes())
    marker["result_root"] = (
        f"_engine/indicator_sets/{marker['indicator_set_id'][:12]}/data"
    )
    marker["indicators_manifest_sha256"] = content_sha256(
        {
            key: value
            for key, value in marker.items()
            if key != "indicators_manifest_sha256"
        }
    )
    path.write_bytes(canonical_json_bytes(marker))
    assert (
        cb.resolve_indicator_set_dir(str(tmp_path)) == (path.parent / "data").as_posix()
    )


def test_a_scoped_record_does_not_clear_an_orphan(project, capsys):
    """A scoped re-record PRESERVES the orphaned row, so the FAIL persists.

    `cmd_record --workflow` prunes by RESOLVED in-scope path, not by owning
    workflow, so a row that no longer resolves is not in `selected_paths` and
    survives the merge -- landing beside the freshly recorded row rather than
    replacing it. This is the constraint on the obvious remedy: only an
    UNSCOPED record (which overwrites `targets` wholesale) clears an orphan.
    """
    project_dir, manifest_path = project
    orphan = _orphan_the_indicator_row(manifest_path)

    rc = cb.cmd_record(
        _record_ns(project_dir, manifest_path, workflow=["simulate_system"])
    )
    assert rc == 0

    targets = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))[
        "targets"
    ]
    assert orphan in targets, "scoped record unexpectedly pruned the orphan"

    capsys.readouterr()
    assert (
        cb.cmd_check(
            _check_ns(project_dir, manifest_path, workflow=["simulate_system"])
        )
        == 1
    )
    assert "orphaned" in capsys.readouterr().out


def test_an_unscoped_record_clears_an_orphan(project, capsys):
    """The counterpart: a full record overwrites `targets`, so the row is gone."""
    project_dir, manifest_path = project
    orphan = _orphan_the_indicator_row(manifest_path)

    assert cb.cmd_record(_record_ns(project_dir, manifest_path)) == 0

    targets = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))[
        "targets"
    ]
    assert orphan not in targets
