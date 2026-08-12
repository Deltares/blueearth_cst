"""Tests for the per-workflow benchmark gather (blueearth_cst/shared/merge_benchmarks.py).

The merged output is a Markdown table (per-rule parts stay TSV = Snakemake's
fixed benchmark format).
"""

from blueearth_cst.shared.merge_benchmarks import merge_benchmarks  # noqa: E402

_HEADER = "s\th:m:s\tmax_rss\tio_in\tcpu_time\tmean_load\n"


def _write(path, s, rss, io_in, cpu, load):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _HEADER + f"{s}\t0:00:00\t{rss}\t{io_in}\t{cpu}\t{load}\n", encoding="utf-8"
    )


def test_filters_by_workflow_prefix_and_appends_total(tmp_path):
    # mirror the production layout: <project>/benchmarks/{_parts,wf1_benchmarks.md}
    bench = tmp_path / "gabon" / "benchmarks"
    parts = bench / "_parts"
    _write(parts / "1.03_create_model.tsv", 10.0, 100.0, 1.0, 8.0, 50.0)
    _write(parts / "1.09_run_wflow.tsv", 20.0, 200.0, 3.0, 18.0, 70.0)
    _write(parts / "2.05_other_wf.tsv", 99.0, 999.0, 9.0, 9.0, 9.0)  # must be excluded
    out = bench / "wf1_benchmarks.md"
    merge_benchmarks(str(parts), "1", str(out))

    md = out.read_text(encoding="utf-8")
    # the same provenance header rule logs carry (project, dir, date), but
    # rendered as a fenced metadata box (not stacked H1s) and labelled for a
    # benchmark artifact rather than a log
    assert md.startswith("```text")  # code fence, so no H1 heading collision
    assert "BlueEarth-CST | project: gabon |" in md and "project dir:" in md
    assert "benchmark: wf1_benchmarks.md | generated " in md  # relabelled
    assert (
        "# BlueEarth-CST" not in md and "# log:" not in md
    )  # not headings, not "log:"
    assert "# wf1 benchmarks" in md  # the single H1 title follows the header
    assert "| rule" in md  # a Markdown table
    assert "1.03_create_model" in md and "1.09_run_wflow" in md
    assert "2.05_other_wf" not in md  # WF2 excluded
    # the TOTAL row carries the right aggregations (sum s/io/cpu, max mem, mean load)
    total_line = next(ln for ln in md.splitlines() if "TOTAL" in ln)
    for value in ("30.00", "26.00", "4.00", "200.00", "60.00"):
        assert value in total_line
    # legend appended under the table, explaining the columns
    assert "How to read this" in md and "`max_rss`" in md and "`cpu_time`" in md
    # this workflow's merged parts are deleted; another workflow's part is kept
    assert not (parts / "1.03_create_model.tsv").exists()
    assert not (parts / "1.09_run_wflow.tsv").exists()
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
    _write(parts / "1.02_prepare_spatial_maps.tsv", 5.0, 50.0, 0.0, 4.0, 40.0)
    wf3 = parts / "gabon_dry" / "3.16_derive_wflow_indicators.tsv"
    _write(wf3, 7.0, 70.0, 0.0, 6.0, 60.0)

    out = tmp_path / "wf1_benchmarks.md"
    merge_benchmarks(str(parts), "1", str(out))

    text = out.read_text(encoding="utf-8")
    assert "1.02_prepare_spatial_maps" in text
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
    assert "# wf3 benchmarks" in md
    assert "no benchmark parts found" in md
