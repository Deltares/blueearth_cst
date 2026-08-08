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
import pathlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev" / "scripts"))
import check_baseline as cb  # noqa: E402


def _write_target(path: str, kind: str) -> None:
    """Write a minimal but valid file for the given target kind."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if kind == "nc":
        import xarray as xr

        xr.Dataset({"v": (("t",), np.array([1.0, 2.0, 3.0]))}).to_netcdf(p)
    elif kind == "csv":
        p.write_text("a,b\n1,2\n")
    elif kind == "yaml":
        p.write_text("project:\n  project_dir: synthetic\n")
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
        rows = ["metric,st_id,temp_change,precip_change,realization_id,location,value"]
        for metric, base in (("q_mean", 10.0), ("q_low", 0.5)):
            rows += [
                f"{metric},st_{i:02d},0.0,1.0,0,101,{base * (1.0 + 0.1 * i)!r}"
                for i in range(5)
            ]
        p.write_text("\n".join(rows) + "\n")
    else:  # pragma: no cover - guard against a new untested kind
        raise ValueError(f"unhandled kind: {kind}")


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
    """A synthetic project dir with all 13 targets present + a recorded manifest.

    Returns (project_dir, manifest_path). Both point under tmp_path.
    """
    project_dir = str(tmp_path)
    for _workflow, kind, template in cb.TARGETS:
        _write_target(cb.resolve(template, project_dir), kind)

    manifest_path = tmp_path / "manifest.json"
    # Recorded WITH figures so the fixture still exercises every target kind;
    # the default-exclusion behaviour gets its own tests below.
    rc = cb.cmd_record(_record_ns(project_dir, manifest_path, include_figures=True))
    assert rc == 0  # fixture sanity: all synthetic targets recorded
    return project_dir, manifest_path


def test_targets_tagged_with_expected_cardinality():
    """The shipping TARGETS carry the 5/6/2 workflow tags the count math relies on.

    `model_creation` gained the beyond-`rule all` discharge target.
    `climate_experiment` dropped from 3 to 2 at R11 CR-2: `basin_indicators.csv`
    no longer exists, and the seed config declares only `river discharge`, so it
    emits one indicator table. A project configuring more output variables gets
    more tables — but this list describes the SEED tree, which is why the number
    is pinned here rather than derived.
    """
    counts = Counter(workflow for workflow, _kind, _template in cb.TARGETS)
    assert counts == {
        "model_creation": 5,
        "climate_projections": 6,
        "climate_experiment": 2,
    }


def test_scoped_count_is_selected_not_full(project, capsys):
    """`--workflow model_creation --workflow climate_projections` reports 11
    of the 13 targets, not the full set."""
    project_dir, manifest_path = project
    rc = cb.cmd_check(
        _check_ns(project_dir, manifest_path,
                  workflow=["model_creation", "climate_projections"],
                  include_figures=True)
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK - 11 target(s)" in out


def test_selected_missing_target_fails(project, capsys):
    """A selected target missing on disk -> non-zero, and named."""
    project_dir, manifest_path = project
    victim = cb.resolve(
        "{clim_project_dir}/summary/{clim_project}_change_factors_annual.csv", project_dir
    )
    Path(victim).unlink()

    rc = cb.cmd_check(
        _check_ns(project_dir, manifest_path,
                  workflow=["model_creation", "climate_projections"],
                  include_figures=True)
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert victim in out


def test_unselected_missing_target_ignored(project, capsys):
    """An unselected (workflow-3) target missing on disk is ignored by a scoped
    check -> returns 0, count stays 11."""
    project_dir, manifest_path = project
    victim = cb.resolve("{exp_dir}/results/q_indicators.csv", project_dir)
    Path(victim).unlink()

    rc = cb.cmd_check(
        _check_ns(project_dir, manifest_path,
                  workflow=["model_creation", "climate_projections"],
                  include_figures=True)
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK - 11 target(s)" in out


def test_unscoped_record_writes_all_targets(project):
    """An unscoped record with --include-figures writes all 13 (overwrite)."""
    project_dir, manifest_path = project
    written = json.loads(Path(manifest_path).read_text())
    assert len(written["targets"]) == 13
    assert written["version"] == cb.MANIFEST_VERSION


def test_record_workflow_merges_and_preserves_other_slices(project):
    """`record --workflow model_creation` re-records only wf1 rows and preserves
    the wf2/wf3 rows verbatim (ADR step 7 merge semantics — no silent clobber)."""
    project_dir, manifest_path = project
    before = json.loads(Path(manifest_path).read_text())["targets"]

    cp_path = cb.resolve(
        "{clim_project_dir}/summary/{clim_project}_change_factors_annual.csv", project_dir
    )
    exp_path = cb.resolve("{exp_dir}/results/q_indicators.csv", project_dir)
    cp_before, exp_before = before[cp_path], before[exp_path]

    # Mutate a wf1 target AND a wf2 target on disk; then merge-record only wf1.
    # A correct merge must re-record wf1 yet leave the (now stale) wf2 row as it
    # was — proving it did not recompute or drop the unselected slice.
    disch_path = cb.resolve(
        "{project_dir}/models/hydrology/wflow/run_default/output.csv", project_dir
    )
    Path(cp_path).write_text("a,b\n9,9\n")  # wf2 content now differs from cp_before
    Path(disch_path).write_text(
        "time,Q_synthetic\n" + "\n".join(
            f"2000-01-{i + 2:02d}T00:00:00,{5.0 + i!r}" for i in range(10)
        ) + "\n"
    )

    rc = cb.cmd_record(_record_ns(project_dir, manifest_path, workflow=["model_creation"]))
    assert rc == 0
    after = json.loads(Path(manifest_path).read_text())["targets"]

    assert len(after) == 13                       # nothing dropped
    assert after[cp_path] == cp_before            # wf2 row preserved verbatim
    assert after[exp_path] == exp_before          # wf3 row preserved verbatim
    # wf1 discharge row re-recorded against the mutated series.
    assert after[disch_path]["type"] == "discharge"
    assert after[disch_path]["mean_ref"] != before[disch_path]["mean_ref"]


def test_unscoped_check_spans_all_targets(project, capsys):
    """`check --include-figures` with no `--workflow` spans all 13 targets."""
    project_dir, manifest_path = project
    rc = cb.cmd_check(
        _check_ns(project_dir, manifest_path, workflow=None, include_figures=True)
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK - 13 target(s)" in out


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


def test_the_workflow_and_figure_filters_compose():
    scoped = cb.active_targets({"model_creation"})
    assert {w for w, _k, _t in scoped} == {"model_creation"}
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
