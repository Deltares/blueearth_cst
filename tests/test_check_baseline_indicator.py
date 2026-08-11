"""Unit tests for the indicator-table comparator (R11 Q8) in check_baseline.py.

Mirrors tests/test_check_baseline_discharge.py, which covers the other
REFERENCE_KIND. The two comparators share a rule -- structural checks first,
ATOL from the reference's own mean, RTOL as a large-value tightener -- so the
cases here are deliberately the same cases, plus the ones that only exist
because an indicator table stacks many series in one file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev" / "scripts"))
import check_baseline as cb  # noqa: E402

from blueearth_cst.shared.indicator_tables import INDICATOR_COLUMNS  # noqa: E402


def test_comparator_column_assumptions_match_the_writer():
    """`check_baseline` re-states the value column rather than importing it, so
    that a bare checkout can run the gate without the package installed. That
    duplication is only safe if something pins the two together -- this is it."""
    assert cb.INDICATOR_VALUE_COLUMN in INDICATOR_COLUMNS
    assert set(cb.INDICATOR_GROUP_COLUMNS).issubset(set(INDICATOR_COLUMNS))
    assert cb.INDICATOR_VALUE_COLUMN not in cb.INDICATOR_GROUP_COLUMNS


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _table(rows: list[dict]) -> pd.DataFrame:
    """Build a table in the canonical column order, values numeric."""
    df = pd.DataFrame(rows, columns=list(INDICATOR_COLUMNS))
    df[cb.INDICATOR_VALUE_COLUMN] = pd.to_numeric(df[cb.INDICATOR_VALUE_COLUMN])
    for c in df.columns:
        if c != cb.INDICATOR_VALUE_COLUMN:
            df[c] = df[c].astype(str)
    return df


def _row(
    metric="q_mean",
    st_id="st_00",
    temp="0.0",
    precip="1.0",
    rlz="0",
    location="101",
    value=10.0,
) -> dict:
    return {
        "metric": metric,
        "st_id": st_id,
        "temp_change": temp,
        "precip_change": precip,
        "rlz_id": rlz,
        "location": location,
        "value": value,
    }


def _grid(metric: str, location: str, base: float, n: int = 6) -> list[dict]:
    return [
        _row(
            metric=metric,
            location=location,
            st_id=f"st_{i:02d}",
            value=base * (1.0 + 0.1 * i),
        )
        for i in range(n)
    ]


# --------------------------------------------------------------------------
# The comparator, in isolation
# --------------------------------------------------------------------------


def test_identical_tables_pass():
    ref = _table(_grid("q_mean", "101", 10.0))
    report = cb.compare_indicator_table(ref, ref.copy())
    assert report["ok"] and report["n_fail"] == 0
    assert report["structural"] == []


def test_within_absolute_tolerance_passes():
    ref = _table(_grid("q_mean", "101", 10.0))
    cur = ref.copy()
    # mean|ref| ~ 12.5 -> ATOL ~ 1.25e-2. Nudge one row well under it.
    cur.loc[2, "value"] += 1e-3
    report = cb.compare_indicator_table(ref, cur)
    assert report["ok"], report


def test_absolute_breach_is_material():
    ref = _table(_grid("q_mean", "101", 10.0))
    cur = ref.copy()
    cur.loc[2, "value"] += 5.0
    report = cb.compare_indicator_table(ref, cur)
    assert not report["ok"] and report["n_fail"] == 1


def test_relative_tightener_catches_a_large_value_move():
    """A move can sit inside no absolute budget yet be 10% of its own row.

    The row's value is far above the group ATOL, so the relative clause governs
    -- the indicator analogue of the discharge low-flow tightener.
    """
    ref = _table(
        [
            _row(value=1000.0, st_id="st_00"),
            _row(value=1000.0, st_id="st_01"),
            _row(value=1000.0, st_id="st_02"),
        ]
    )
    cur = ref.copy()
    cur.loc[1, "value"] = 1100.0  # +10%, RTOL is 1%
    report = cb.compare_indicator_table(ref, cur)
    assert not report["ok"] and report["n_fail"] == 1
    assert report["max_rel"] == pytest.approx(0.1)


def test_near_zero_group_is_division_safe():
    """An all-zero group has ATOL == 0 and must not raise or divide by zero."""
    ref = _table([_row(value=0.0, st_id=f"st_{i:02d}") for i in range(3)])
    cur = ref.copy()
    report = cb.compare_indicator_table(ref, cur)
    assert report["ok"] and report["max_rel"] == 0.0


def test_tolerance_is_per_group_not_per_file():
    """The point of grouping: a big series must not set the threshold for a small one.

    Two metrics three orders of magnitude apart. A move that is material for the
    small one is far below a file-wide ATOL derived from the large one, so a
    global tolerance would pass this table and a grouped one must fail it.
    """
    ref = _table(_grid("gev_rl100", "101", 5000.0) + _grid("q_low", "101", 0.5))
    cur = ref.copy()
    small = cur.index[cur["metric"] == "q_low"]
    cur.loc[small[0], "value"] += 0.5  # +100% of a ~0.5 value

    file_wide_atol = cb.INDICATOR_ATOL_FRAC * float(ref["value"].abs().mean())
    assert 0.5 < file_wide_atol, "test premise: the move hides under a global ATOL"

    report = cb.compare_indicator_table(ref, cur)
    assert not report["ok"] and report["n_fail"] == 1
    assert report["failing_groups"] == ["q_low\x1f101"]


def test_locations_are_separate_groups():
    """Same metric, different gauges: catchment area makes their magnitudes differ."""
    ref = _table(_grid("q_mean", "101", 2000.0) + _grid("q_mean", "202", 0.4))
    cur = ref.copy()
    small = cur.index[cur["location"] == "202"]
    cur.loc[small[0], "value"] += 0.4
    report = cb.compare_indicator_table(ref, cur)
    assert not report["ok"]
    assert report["failing_groups"] == ["q_mean\x1f202"]


# --------------------------------------------------------------------------
# Structural checks — any hit is a FAIL, never a numeric pass
# --------------------------------------------------------------------------


def test_structural_column_added():
    ref = _table(_grid("q_mean", "101", 10.0))
    cur = ref.copy()
    cur["new_axis"] = "x"
    report = cb.compare_indicator_table(ref, cur)
    assert not report["ok"]
    assert any("column mismatch" in s for s in report["structural"])
    assert report["n_fail"] is None  # never scored numerically


def test_structural_column_reorder_is_caught_separately():
    ref = _table(_grid("q_mean", "101", 10.0))
    cur = ref[list(reversed(ref.columns))].copy()
    report = cb.compare_indicator_table(ref, cur)
    assert not report["ok"]
    assert any("column ORDER changed" in s for s in report["structural"])


def test_structural_duplicate_row_keys():
    rows = _grid("q_mean", "101", 10.0, n=3)
    rows.append(dict(rows[0]))  # exact duplicate key
    ref = _table(rows)
    report = cb.compare_indicator_table(ref, ref.copy())
    assert not report["ok"]
    assert any("duplicate row key" in s for s in report["structural"])


def test_structural_row_key_mismatch():
    ref = _table(_grid("q_mean", "101", 10.0, n=4))
    cur = _table(_grid("q_mean", "101", 10.0, n=3))
    report = cb.compare_indicator_table(ref, cur)
    assert not report["ok"]
    assert any("row-key mismatch" in s for s in report["structural"])


def test_structural_non_finite():
    ref = _table(_grid("q_mean", "101", 10.0))
    cur = ref.copy()
    cur.loc[1, "value"] = float("nan")
    report = cb.compare_indicator_table(ref, cur)
    assert not report["ok"]
    assert any("non-finite" in s for s in report["structural"])


def test_reordered_current_rows_are_aligned_not_flagged():
    """Row order is not part of the contract; the key alignment is."""
    ref = _table(_grid("q_mean", "101", 10.0))
    cur = ref.iloc[::-1].reset_index(drop=True)
    report = cb.compare_indicator_table(ref, cur)
    assert report["ok"], report


def test_a_filtered_frame_does_not_mis_index():
    """The group loop maps labels to positions, so a non-RangeIndex `ref` would
    silently mis-index. Both sides are normalized on entry; pin it from the
    REFERENCE side, which the reorder test above does not exercise."""
    full = _table(_grid("q_mean", "101", 10.0, n=6))
    ref = full[full["st_id"] != "st_00"]  # index 1..5, deliberately gappy
    cur = ref.copy()
    assert list(ref.index) != list(range(len(ref)))  # premise
    assert cb.compare_indicator_table(ref, cur)["ok"]

    cur = cur.copy()
    cur.loc[3, "value"] += 5.0
    report = cb.compare_indicator_table(ref, cur)
    assert not report["ok"] and report["n_fail"] == 1


def test_float_key_columns_are_compared_as_written():
    """`temp_change` reparsed as float would make 1.3 and 1.3000000000000003
    different groups and report a key-set mismatch. Read as strings, they are
    the same key iff they were written the same way."""
    ref = _table([_row(temp="1.3", st_id="st_00"), _row(temp="2.6", st_id="st_01")])
    assert cb.read_indicator_table.__doc__  # documented reason for dtype=str
    report = cb.compare_indicator_table(ref, ref.copy())
    assert report["ok"]


# --------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------


def test_reader_rejects_a_table_with_no_value_column(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("metric,st_id\nq_mean,st_00\n")
    with pytest.raises(ValueError, match="no 'value' column"):
        cb.read_indicator_table(str(p))


def test_reader_keeps_key_columns_as_written(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text(
        "metric,st_id,temp_change,precip_change,rlz_id,location,value\n"
        "q_mean,st_00,1.3000000000000003,1.0,0,101,12.5\n"
    )
    df = cb.read_indicator_table(str(p))
    assert df.loc[0, "temp_change"] == "1.3000000000000003"
    assert df.loc[0, "value"] == pytest.approx(12.5)


# --------------------------------------------------------------------------
# record / check round trip through the CLI entry points
# --------------------------------------------------------------------------


def _record_ns(project_dir, manifest_path, workflow=None):
    return argparse.Namespace(
        cmd="record",
        project_dir=project_dir,
        manifest=manifest_path,
        workflow=workflow,
        include_figures=False,
    )


def _check_ns(project_dir, manifest_path, workflow=None, tolerance=0.0):
    return argparse.Namespace(
        cmd="check",
        project_dir=project_dir,
        manifest=manifest_path,
        tolerance=tolerance,
        workflow=workflow,
        include_figures=False,
    )


def _write_climate_experiment_targets(project_dir: str) -> str:
    """Materialize every workflow-3 target. Returns the indicator table path."""
    indicator = ""
    for workflow, kind, template in cb.TARGETS:
        if workflow != "climate_experiment":
            continue
        path = cb.resolve(template, project_dir)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if kind == "indicator":
            _table(_grid("q_mean", "101", 10.0) + _grid("q_low", "101", 0.5)).to_csv(
                p, index=False
            )
            indicator = path
        elif kind == "yaml":
            p.write_text("project:\n  project_dir: synthetic\n")
        elif kind == "csv":
            p.write_text("a,b\n1,2\n")
        elif kind == "png":
            p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    return indicator


@pytest.fixture
def experiment_only_project(tmp_path):
    project_dir = str(tmp_path)
    indicator = _write_climate_experiment_targets(project_dir)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": cb.MANIFEST_VERSION,
                "project_dir": project_dir,
                "targets": {"sentinel/wf1.csv": {"type": "csv", "sha256": "deadbeef"}},
            }
        )
    )
    rc = cb.cmd_record(
        _record_ns(project_dir, manifest_path, workflow=["climate_experiment"])
    )
    assert rc == 0
    return project_dir, manifest_path, indicator


def test_roundtrip_records_reference_table_and_checks_clean(
    experiment_only_project, capsys
):
    project_dir, manifest_path, _ = experiment_only_project
    targets = json.loads(Path(manifest_path).read_text())["targets"]
    assert "sentinel/wf1.csv" in targets  # other workflows preserved
    rows = [r for r in targets.values() if r.get("type") == "indicator"]
    assert len(rows) == 1
    row = rows[0]
    assert row["n_rows"] == 12 and row["n_groups"] == 2
    assert (Path(manifest_path).parent / row["ref_table"]).exists()

    rc = cb.cmd_check(
        _check_ns(project_dir, manifest_path, workflow=["climate_experiment"])
    )
    assert rc == 0 and "OK -" in capsys.readouterr().out


def test_roundtrip_passes_an_immaterial_move(experiment_only_project, capsys):
    """The whole point of Q8: unrounded values nudge, and that is not a defect."""
    project_dir, manifest_path, indicator = experiment_only_project
    df = pd.read_csv(indicator)
    df["value"] = df["value"] * (1 + 1e-9)
    df.to_csv(indicator, index=False)

    rc = cb.cmd_check(
        _check_ns(project_dir, manifest_path, workflow=["climate_experiment"])
    )
    assert rc == 0, capsys.readouterr().out


def test_roundtrip_detects_a_material_move(experiment_only_project, capsys):
    project_dir, manifest_path, indicator = experiment_only_project
    df = pd.read_csv(indicator)
    df.loc[0, "value"] += 5.0
    df.to_csv(indicator, index=False)

    rc = cb.cmd_check(
        _check_ns(project_dir, manifest_path, workflow=["climate_experiment"])
    )
    out = capsys.readouterr().out
    assert rc == 1 and indicator in out and "exceed tolerance" in out


def test_record_refuses_when_the_table_is_missing(tmp_path, capsys):
    project_dir = str(tmp_path)
    indicator = _write_climate_experiment_targets(project_dir)
    Path(indicator).unlink()
    manifest_path = tmp_path / "manifest.json"
    rc = cb.cmd_record(
        _record_ns(project_dir, manifest_path, workflow=["climate_experiment"])
    )
    assert rc == 1
    assert "Missing targets" in capsys.readouterr().err
    assert not manifest_path.exists()


def test_indicator_is_not_fingerprinted_as_a_hash(tmp_path):
    """REFERENCE_KINDS must be skipped by compute_manifest.

    If the skip is forgotten the kind has no FINGERPRINTERS entry, so this
    would raise KeyError rather than silently hashing -- but pin the intent.
    """
    project_dir = str(tmp_path)
    indicator = _write_climate_experiment_targets(project_dir)
    manifest, _missing = cb.compute_manifest(
        project_dir, workflows={"climate_experiment"}
    )
    assert indicator not in manifest
    assert "indicator" in cb.REFERENCE_KINDS
    assert "indicator" not in cb.FINGERPRINTERS
