"""Tests for the per-workflow benchmark gather (blueearth_cst/shared/merge_benchmarks.py).

The merged output is a Markdown table (per-rule parts stay TSV = Snakemake's
fixed benchmark format).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from blueearth_cst.shared.merge_benchmarks import merge_benchmarks

_HEADER = "s\th:m:s\tmax_rss\tio_in\tcpu_time\tmean_load\n"


def _write(path, s, rss, io_in, cpu, load):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _HEADER + f"{s}\t0:00:00\t{rss}\t{io_in}\t{cpu}\t{load}\n", encoding="utf-8"
    )


@pytest.mark.integration
def test_gather_executes_through_snakemake_script_adapter(tmp_path):
    """Imported formatter tests do not exercise Snakemake's injected preamble."""
    parts = tmp_path / "parts"
    _write(parts / "1.03_create_model.tsv", 10, 100, 1, 8, 50)
    out = tmp_path / "wf1-benchmarks.md"
    repo = Path(__file__).resolve().parents[1]
    script = (repo / "blueearth_cst/shared/merge_benchmarks.py").as_posix()
    snakefile = tmp_path / "Snakefile"
    snakefile.write_text(
        f"rule all:\n    input: {str(out)!r}\n"
        f"rule gather:\n    output: {str(out)!r}, {str(out.with_suffix('.tsv'))!r}\n"
        f"    params: parts_dir={str(parts)!r}, workflow_num=1\n"
        f"    script: {script!r}\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "snakemake",
            "--cores",
            "1",
            "--snakefile",
            str(snakefile),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(repo)},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Create model" in out.read_text(encoding="utf-8")
    assert pd.read_csv(out.with_suffix(".tsv"), sep="\t")["s"].tolist() == [10]


def test_filters_by_workflow_prefix_and_retains_raw_measurements(tmp_path):
    # mirror the production layout: <project>/benchmarks/{_parts,wf1_benchmarks.md}
    bench = tmp_path / "gabon" / "benchmarks"
    parts = bench / "_parts"
    _write(parts / "1.03_create_model.tsv", 10.0, 100.0, 1.0, 8.0, 50.0)
    _write(parts / "1.09b_run_wflow.tsv", 20.0, 200.0, 3.0, 18.0, 70.0)
    _write(parts / "2.05_other_wf.tsv", 99.0, 999.0, 9.0, 9.0, 9.0)  # must be excluded
    out = bench / "wf1_benchmarks.md"
    merge_benchmarks(str(parts), "1", str(out))

    md = out.read_text(encoding="utf-8")
    # the same provenance header rule logs carry (project, dir, date), but
    # rendered as a fenced metadata box (not stacked H1s) and labelled for a
    # benchmark artifact rather than a log
    assert md.startswith("```text")  # code fence, so no H1 heading collision
    assert "BlueEarth-CST | project: gabon |" in md and "<project>:" in md
    assert "benchmark: wf1_benchmarks.md | generated " in md  # relabelled
    assert (
        "# BlueEarth-CST" not in md and "# log:" not in md
    )  # not headings, not "log:"
    assert "# WF1 benchmarks" in md  # the single H1 title follows the header
    assert "| rule" in md  # a Markdown table
    assert "1.03_create_model" in md and "1.09b_run_wflow" in md
    assert "2.05_other_wf" not in md  # WF2 excluded
    assert "TOTAL" not in md
    assert "Create model" in md and "Run wflow" in md
    assert "Workflow elapsed: unavailable" in md
    assert "Completeness: unknown" in md
    assert "<details>" in md and "`cpu_time`" in md
    raw = pd.read_csv(out.with_suffix(".tsv"), sep="\t")
    assert list(raw.columns) == ["rule", "sample", *_HEADER.strip().split("\t")]
    assert raw["s"].tolist() == [10.0, 20.0]
    assert raw["sample"].tolist() == [1, 1]
    # this workflow's merged parts are deleted; another workflow's part is kept
    assert not (parts / "1.03_create_model.tsv").exists()
    assert not (parts / "1.09b_run_wflow.tsv").exists()
    assert (parts / "2.05_other_wf.tsv").exists()  # WF2 part untouched
    assert parts.exists()  # the shared _parts dir itself is kept


def test_nested_wildcard_parts_get_relative_rule_label(tmp_path):
    parts = tmp_path / "_parts"
    _write(
        parts / "2.02_monthly_stats_hist" / "INM" / "GFDL.tsv",
        5.0,
        50.0,
        0.0,
        4.0,
        40.0,
    )
    out = tmp_path / "wf2_benchmarks.md"
    merge_benchmarks(str(parts), "2", str(out))
    assert "2.02_monthly_stats_hist/INM/GFDL" in out.read_text(encoding="utf-8")
    # merged parts and their now-empty _parts tree are removed
    assert not parts.exists()


def test_wf1s_gather_neither_rows_nor_deletes_wf3s_nested_parts(tmp_path):
    """WF3's parts sit INSIDE the `_parts/` WF1 and WF2 share (2026-08-11).

    The recursive glob reaches them, so the `W.` prefix filter is the only thing
    keeping WF1's gather off them -- and it has to gate the DELETION too, since
    `_remove_parts` works off the same list. If it ever gates only the table,
    WF1's run eats WF3's parts and WF3's next table reports "no part from this
    run" for rules that did run: silent, and only visible in a table nobody
    diffs.
    """
    parts = tmp_path / "_parts"
    _write(parts / "1.05_prepare_land_and_soil_maps.tsv", 5.0, 50.0, 0.0, 4.0, 40.0)
    wf3 = parts / "gabon_dry" / "3.16_derive_wflow_indicators.tsv"
    _write(wf3, 7.0, 70.0, 0.0, 6.0, 60.0)

    out = tmp_path / "wf1_benchmarks.md"
    merge_benchmarks(str(parts), "1", str(out))

    text = out.read_text(encoding="utf-8")
    assert "1.05_prepare_land_and_soil_maps" in text
    assert "3.16_derive_wflow_indicators" not in text
    assert "gabon_dry" not in text
    assert wf3.is_file(), "WF1's gather deleted WF3's benchmark part"

    # ...and WF3's own gather, pointed at its own subdir, still finds it.
    wf3_out = tmp_path / "wf3_benchmarks_gabon_dry.md"
    merge_benchmarks(str(parts / "gabon_dry"), "3", str(wf3_out))
    assert "3.16_derive_wflow_indicators" in wf3_out.read_text(encoding="utf-8")


def test_no_parts_writes_placeholder(tmp_path):
    out = tmp_path / "wf3_benchmarks.md"
    merge_benchmarks(str(tmp_path / "_parts"), "3", str(out))
    md = out.read_text(encoding="utf-8")
    assert md.startswith("```text")  # fenced header even on the empty placeholder
    assert "BlueEarth-CST" in md
    assert "# WF3 benchmarks" in md
    assert "No benchmark parts found" in md
    assert pd.read_csv(out.with_suffix(".tsv"), sep="\t").empty


def test_fanout_and_repeats_count_jobs_separately(tmp_path):
    parts = tmp_path / "_parts"
    first = parts / "2.04_fetch" / "model-a.tsv"
    _write(first, 10, 100, 1, 8, 50)
    with first.open("a", encoding="utf-8") as handle:
        handle.write("30\t0:00:30\t200\t3\t18\t60\n")
    _write(parts / "2.04_fetch" / "model-b.tsv", 40, 150, 2, 10, 20)
    out = tmp_path / "wf2-benchmarks.md"
    merge_benchmarks(parts, 2, out)
    overview = out.read_text(encoding="utf-8").split("<details>")[0]
    row = next(line for line in overview.splitlines() if "| Fetch " in line)
    cells = [value.strip() for value in row.strip("|").split("|")]
    assert cells[:6] == ["Fetch", "2", "3", "0:01:00", "0:00:30", "0:00:40"]
    assert float(cells[6]) == 200
    assert cells[7] == "0"
    raw = pd.read_csv(out.with_suffix(".tsv"), sep="\t")
    assert raw["sample"].tolist() == [1, 2, 1]


@pytest.mark.parametrize("bad", ["nan", "inf", "-1", "missing"])
def test_missing_measurements_do_not_become_zero_or_complete_sum(tmp_path, bad):
    parts = tmp_path / "_parts"
    _write(parts / "1.04_prepare" / "a.tsv", bad, bad, 1, 2, 3)
    _write(parts / "1.04_prepare" / "b.tsv", 10, 100, 1, 2, 3)
    out = tmp_path / "wf1-benchmarks.md"
    merge_benchmarks(parts, 1, out, remove_parts=False)
    overview = out.read_text(encoding="utf-8").split("<details>")[0]
    row = next(line for line in overview.splitlines() if "| Prepare " in line)
    cells = [value.strip() for value in row.strip("|").split("|")]
    assert cells[3] == "unavailable"
    assert cells[-1] == "1"
    assert (parts / "1.04_prepare" / "a.tsv").is_file()


def test_invocation_elapsed_and_refresh_preserve_raw_bytes(tmp_path):
    parts = tmp_path / "_parts"
    _write(parts / "1.04_prepare.tsv", 1000, 100, 1, 2, 3)
    record_path = tmp_path / "invocation.json"
    record = {"status": "running", "elapsed_seconds": 22, "ended_at_utc": None}
    record_path.write_text(json.dumps(record), encoding="utf-8")
    out = tmp_path / "wf1-benchmarks.md"
    merge_benchmarks(parts, 1, out, invocation_path=record_path)
    assert "Workflow elapsed: running (not finalized)" in out.read_text(
        encoding="utf-8"
    )
    raw_bytes = out.with_suffix(".tsv").read_bytes()
    record.update(
        status="failed", elapsed_seconds=22.5, ended_at_utc="2026-10-19T12:00:00Z"
    )
    record_path.write_text(json.dumps(record), encoding="utf-8")
    merge_benchmarks(parts, 1, out, invocation_path=record_path, refresh=True)
    text = out.read_text(encoding="utf-8")
    assert "Workflow elapsed: 0:00:22 (22.500 s)" in text
    assert "Outcome: failed" in text
    assert "0:16:40" in text  # independent job time retained
    assert out.with_suffix(".tsv").read_bytes() == raw_bytes


def test_julia_phases_keep_process_boundaries_and_failed_observations(tmp_path):
    resources = tmp_path / "resources"
    resources.mkdir()
    header = "phase\tstatus\twall_s\tcompile_s\n"
    (resources / "julia-1.tsv").write_text(
        header + "wflow_load\tok\t20\t10\nmember\tok\t300\t200\n"
        "member\tok\t100\t2\nmember\tfailed\t50\t\n",
        encoding="utf-8",
    )
    (resources / "julia-2.tsv").write_text(
        header + "member\tfailed\t50\t\nmember\tok\t120\t3\n", encoding="utf-8"
    )
    record_path = tmp_path / "invocation.json"
    record_path.write_text(
        json.dumps({"resources": {"directory": str(resources)}}), encoding="utf-8"
    )
    out = tmp_path / "wf4-benchmarks.md"
    merge_benchmarks(tmp_path / "parts", 4, out, invocation_path=record_path)
    phases = out.read_text(encoding="utf-8").split("## Julia phases")[1]
    assert "Wflow load" in phases and "0:00:20" in phases
    assert "First observed member compilation" in phases and "0:03:20" in phases
    assert "Later observed members" in phases and "0:01:50" in phases
    assert "Failed observations: 2" in phases
