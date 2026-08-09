"""Unit tests for the discharge comparator (ADR 0001 step 6) in check_baseline.py.

Covers the pure `compare_discharge` (structural + absolute + relative-low-flow
clauses, zero-safe handling) and the record -> check roundtrip plus the one-off
`compare` subcommand — all with synthetic `output.csv` series, no Wflow build.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev" / "scripts"))
import check_baseline as cb  # noqa: E402


def _times(n: int) -> list[str]:
    return [f"2000-01-{i + 1:02d}T00:00:00" for i in range(n)]


def _write_output_csv(
    path: Path, times: list[str], q: list[float], col: str = "Q_1"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"time,{col}"]
    lines += [f"{t},{float(v)!r}" for t, v in zip(times, q)]
    path.write_text("\n".join(lines) + "\n")


# --- pure comparator -------------------------------------------------------


def test_identical_series_pass():
    t = _times(20)
    q = np.linspace(1.0, 5.0, 20)
    rep = cb.compare_discharge(t, q, t, q.copy())
    assert rep["ok"] and not rep["structural"]
    assert rep["n_fail"] == 0


def test_within_absolute_tolerance_passes():
    t = _times(50)
    q = np.full(50, 10.0)
    # ATOL = 1e-3 * mean(10) = 0.01; perturb by 0.5*ATOL -> below both clauses.
    cur = q + 0.005
    rep = cb.compare_discharge(t, q, t, cur)
    assert rep["ok"], rep


def test_absolute_breach_is_material():
    t = _times(50)
    q = np.full(50, 10.0)
    cur = q.copy()
    cur[7] += 1.0  # >> ATOL(0.01) and >> 1% of 10
    rep = cb.compare_discharge(t, q, t, cur)
    assert not rep["ok"]
    assert rep["n_fail"] == 1
    assert rep["first_fail"] == t[7]
    assert rep["worst_fail"] == t[7]


def test_relative_low_flow_tightener_is_material():
    """A move small in absolute terms but >1% at a low-but-above-ATOL flow fails
    on the relative clause even though it passes the absolute clause."""
    t = _times(50)
    q = np.full(50, 10.0)
    q[3] = 0.05  # low flow; mean stays ~9.9 so ATOL ~ 0.0099
    cur = q.copy()
    cur[3] = 0.05 * 1.05  # +5% at a step where Q_ref (0.05) >= ATOL
    # |dQ| = 0.0025 < ATOL(~0.0099) -> absolute clause passes; relative (5%>1%) fails.
    rep = cb.compare_discharge(t, q, t, cur)
    assert not rep["ok"]
    assert rep["first_fail"] == t[3]
    assert rep["max_rel"] > cb.DISCHARGE_RTOL


def test_near_dry_below_atol_is_zero_safe():
    """Where Q_ref < ATOL the relative clause is skipped: a large *relative* wobble
    on a near-dry step is not material as long as the absolute move is < ATOL."""
    t = _times(50)
    q = np.full(50, 10.0)
    q[9] = 1e-6  # far below ATOL (~0.01)
    cur = q.copy()
    cur[9] = 5e-6  # +400% relative, but |dQ| = 4e-6 << ATOL
    rep = cb.compare_discharge(t, q, t, cur)
    assert rep["ok"], rep


def test_structural_duplicate_timestamps():
    t = _times(10)
    t_dup = t.copy()
    t_dup[5] = t_dup[4]  # duplicate
    q = np.ones(10)
    rep = cb.compare_discharge(t_dup, q, t, q)
    assert not rep["ok"]
    assert any("duplicate" in s for s in rep["structural"])


def test_structural_unequal_index():
    t = _times(10)
    q = np.ones(10)
    t2 = _times(9) + ["2001-05-05T00:00:00"]  # same length, different set
    rep = cb.compare_discharge(t, q, t2, q)
    assert not rep["ok"]
    assert any("time-index mismatch" in s for s in rep["structural"])


def test_structural_nan():
    t = _times(10)
    q = np.ones(10)
    cur = q.copy()
    cur[2] = np.nan
    rep = cb.compare_discharge(t, q, t, cur)
    assert not rep["ok"]
    assert any("non-finite" in s for s in rep["structural"])


def test_reordered_current_is_aligned_not_flagged():
    """Same index set in a different row order must align, not read as a mismatch."""
    t = _times(10)
    q = np.arange(10, dtype=float) + 1.0
    order = list(range(10))[::-1]
    t_rev = [t[i] for i in order]
    q_rev = q[order]
    rep = cb.compare_discharge(t, q, t_rev, q_rev)
    assert rep["ok"], rep


def test_reader_uses_outlet_crosswalk_when_registry_adds_q_columns(tmp_path):
    model_root = tmp_path / "hydrology_model"
    output = model_root / "run_default" / "output.csv"
    output.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "time": _times(2),
            "Q_101": [1.0, 2.0],
            "Q_102": [8.0, 9.0],
            "Q_101.1": [1.0, 2.0],
            "P_101": [3.0, 4.0],
        }
    ).to_csv(output, index=False)
    staticgeoms = model_root / "staticgeoms"
    staticgeoms.mkdir()
    pd.DataFrame(
        {
            "compat_station_name": ["wflow_1"],
            "subcatchment_id": [101],
        }
    ).to_csv(staticgeoms / "outlet_index.csv", index=False)

    times, q, column = cb.read_discharge_series(str(output))

    assert times == _times(2)
    assert q.tolist() == [1.0, 2.0]
    assert column == "Q_101"


def test_reader_rejects_multiple_q_columns_without_outlet_crosswalk(tmp_path):
    output = tmp_path / "hydrology_model" / "run_default" / "output.csv"
    output.parent.mkdir(parents=True)
    pd.DataFrame({"time": _times(2), "Q_101": [1.0, 2.0], "Q_102": [3.0, 4.0]}).to_csv(
        output, index=False
    )

    with pytest.raises(ValueError, match="missing .*outlet_index.csv"):
        cb.read_discharge_series(str(output))


# --- record -> check roundtrip + compare subcommand ------------------------


def _record_ns(project_dir, manifest_path, workflow=None):
    return argparse.Namespace(
        cmd="record", project_dir=project_dir, manifest=manifest_path, workflow=workflow
    )


def _check_ns(project_dir, manifest_path, workflow=None, tolerance=0.0):
    return argparse.Namespace(
        cmd="check",
        project_dir=project_dir,
        manifest=manifest_path,
        tolerance=tolerance,
        workflow=workflow,
    )


def _write_model_creation_targets(project_dir: str) -> str:
    """Materialize every workflow-1 target (3 PNGs, 1 yaml, the discharge csv).
    Returns the resolved discharge path."""
    disch = ""
    for workflow, kind, template in cb.TARGETS:
        if workflow != "model_creation":
            continue
        path = cb.resolve(template, project_dir)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if kind == "discharge":
            _write_output_csv(p, _times(30), list(np.linspace(2.0, 8.0, 30)))
            disch = path
        elif kind == "png":
            p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
        elif kind == "yaml":
            p.write_text("project:\n  project_dir: synthetic\n")
    return disch


@pytest.fixture
def discharge_only_project(tmp_path):
    """A project dir with the workflow-1 targets present, recorded via
    `record --workflow model_creation` — which merges into a manifest seeded with
    an unrelated wf2 row, exercising the merge path and the discharge sidecar."""
    project_dir = str(tmp_path)
    disch = _write_model_creation_targets(project_dir)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": cb.MANIFEST_VERSION,
                "project_dir": project_dir,
                "targets": {"sentinel/wf2.csv": {"type": "csv", "sha256": "deadbeef"}},
            }
        )
    )
    rc = cb.cmd_record(
        _record_ns(project_dir, manifest_path, workflow=["model_creation"])
    )
    assert rc == 0
    return project_dir, manifest_path, disch


def test_roundtrip_records_sidecar_and_checks_clean(discharge_only_project, capsys):
    project_dir, manifest_path, _disch = discharge_only_project
    targets = json.loads(Path(manifest_path).read_text())["targets"]
    # wf2 sentinel preserved; 5 wf1 rows recorded incl. discharge row + sidecar.
    assert "sentinel/wf2.csv" in targets
    disch_rows = [r for r in targets.values() if r.get("type") == "discharge"]
    assert len(disch_rows) == 1
    sidecar = Path(manifest_path).parent / disch_rows[0]["ref_series"]
    assert sidecar.exists()

    rc = cb.cmd_check(
        _check_ns(project_dir, manifest_path, workflow=["model_creation"])
    )
    out = capsys.readouterr().out
    assert rc == 0 and "OK - 2 target(s)" in out


def test_roundtrip_detects_material_move(discharge_only_project, capsys):
    project_dir, manifest_path, disch = discharge_only_project
    # Introduce a clearly material spike, then check.
    times, q, col = cb.read_discharge_series(disch)
    q = q.copy()  # read_discharge_series may return a read-only view
    q[10] += 5.0
    _write_output_csv(Path(disch), times, list(q), col=col)

    rc = cb.cmd_check(
        _check_ns(project_dir, manifest_path, workflow=["model_creation"])
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert disch in out
    assert "exceed tolerance" in out


def test_compare_subcommand_pass_and_fail(tmp_path, capsys):
    t = _times(40)
    q = list(np.linspace(1.0, 4.0, 40))
    ref = tmp_path / "ref.csv"
    cur_ok = tmp_path / "cur_ok.csv"
    cur_bad = tmp_path / "cur_bad.csv"
    _write_output_csv(ref, t, q)
    _write_output_csv(cur_ok, t, q)
    bad = list(q)
    bad[20] += 3.0
    _write_output_csv(cur_bad, t, bad)

    rc_ok = cb.cmd_compare(
        argparse.Namespace(cmd="compare", ref=str(ref), cur=str(cur_ok))
    )
    out_ok = capsys.readouterr().out
    assert rc_ok == 0 and "PASS" in out_ok

    rc_bad = cb.cmd_compare(
        argparse.Namespace(cmd="compare", ref=str(ref), cur=str(cur_bad))
    )
    out_bad = capsys.readouterr().out
    assert rc_bad == 1 and "FAIL (material)" in out_bad
