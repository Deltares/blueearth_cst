"""Tests for the workflow-log merge (blueearth_cst/shared/merge_logs.py).

The merge is shared by all three workflows: WF1 1.16, WF2 2.07, WF3 3.13. Its
input is an ordered list of rule LABELS; members of a fan-out rule are discovered
by listing the label's part dir.
"""

from blueearth_cst.shared.merge_logs import merge_logs  # noqa: E402

HEADER = (
    "# BlueEarth-CST | project: gabonx | 2026-07-31\n"
    "# project dir: C:/TESTS/CST/gabonx\n"
    "# log: 2.01_fetch_gcm_raw/modelA | started 14:12:37\n"
    "\n"
)


def _parts(tmp_path, layout):
    """Materialise ``{label: [member, ...] | None}`` under ``tmp_path/_parts``.

    ``None`` writes the single-job form ``<label>.log``; a member list writes
    ``<label>/<member>.log``. Returns the parts dir.
    """
    parts_dir = tmp_path / "_parts"
    for label, members in layout.items():
        targets = (
            [parts_dir / f"{label}.log"]
            if members is None
            else [parts_dir / label / f"{m}.log" for m in members]
        )
        for path, tag in zip(targets, [label] if members is None else members):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                HEADER + f"14:12:37 - cst - INFO - body of {tag}\n", encoding="utf-8"
            )
    return str(parts_dir)


def test_one_header_then_a_banner_per_rule(tmp_path):
    parts_dir = _parts(
        tmp_path,
        {
            "2.01_fetch_gcm_raw": ["modelA", "modelB"],
            "2.04_derive_change_factors": None,
        },
    )
    out = tmp_path / "logs" / "wf2_analyze_projections.log"
    merge_logs(
        ["2.01_fetch_gcm_raw", "2.04_derive_change_factors"], str(out), parts_dir
    )
    text = out.read_text(encoding="utf-8")

    # exactly one provenance header, and it is the merged log's own
    assert text.count("# BlueEarth-CST") == 1
    assert text.startswith("# BlueEarth-CST")
    assert "# log: wf2_analyze_projections.log | merged " in text
    # per-part headers are stripped, bodies survive
    assert "started 14:12:37" not in text
    assert "body of modelA" in text and "body of 2.04_derive_change_factors" in text

    # one `==` banner per rule, tagged as the console banner tags it
    assert "== 2.01  fetch_gcm_raw" in text
    assert "== 2.04  derive_change_factors" in text
    # fan-out members get their own sub-header; a single-job rule does not
    assert "-- modelA " in text and "-- modelB " in text
    assert "-- 2.04_derive_change_factors" not in text


def test_the_merged_log_defines_the_folder_tokens_its_rows_use(tmp_path, monkeypatch):
    """The merge DELETES the parts, so their headers go with them.

    A part defines `<model>` for its own rows; once merged and deleted, the only
    file left is this one, and the bodies it now holds are full of
    `<model>/staticmaps.nc`. An undefined token in the one durable artifact is
    exactly what the per-part definition exists to prevent, one level up.
    """
    import blueearth_cst.shared.snake_utils as su

    monkeypatch.setenv(su._PATH_TOKENS_ENV, "")
    su.declare_path_tokens(model=tmp_path / "models" / "hydrology" / "wflow")
    parts_dir = _parts(tmp_path, {"1.07_build_wflow_model": None})
    out = tmp_path / "logs" / "wf1_build_model.log"
    merge_logs(["1.07_build_wflow_model"], str(out), parts_dir)
    assert "# <model>: models/hydrology/wflow" in out.read_text(encoding="utf-8")


def test_wf3_lettered_rule_number_is_tagged(tmp_path):
    """WF3's first rule is `3.00b`, which a plain isdigit() check rejects."""
    parts_dir = _parts(tmp_path, {"3.00b_check_project_consistency": None})
    out = tmp_path / "merged.log"
    merge_logs(["3.00b_check_project_consistency"], str(out), parts_dir)
    assert "== 3.00b  check_project_consistency" in out.read_text(encoding="utf-8")


def test_sections_follow_the_label_order_not_disk_order(tmp_path):
    parts_dir = _parts(
        tmp_path,
        {"2.01_fetch_gcm_raw": ["modelA"], "2.04_derive_change_factors": None},
    )
    out = tmp_path / "merged.log"
    merge_logs(
        ["2.04_derive_change_factors", "2.01_fetch_gcm_raw"], str(out), parts_dir
    )
    text = out.read_text(encoding="utf-8")
    assert text.index("derive_change_factors") < text.index("fetch_gcm_raw")


def test_members_sort_naturally_not_lexicographically(tmp_path):
    """WF3 fans out to RLZ_NUM x ST_NUM, so `rlz_10` must not precede `rlz_2`."""
    members = [f"rlz_{n}_st_1" for n in (1, 2, 10, 11)]
    parts_dir = _parts(tmp_path, {"3.09_downscale_climate_realization": members})
    out = tmp_path / "merged.log"
    merge_logs(["3.09_downscale_climate_realization"], str(out), parts_dir)
    text = out.read_text(encoding="utf-8")
    order = [text.index(f"body of {m}") for m in members]
    assert order == sorted(order)


def test_member_id_keeps_a_nested_path(tmp_path):
    """A wildcard can carry a `/` (a CMIP6 {model} is `NOAA-GFDL/GFDL-ESM4`)."""
    parts_dir = tmp_path / "_parts"
    part = parts_dir / "2.01_fetch_gcm_raw" / "NOAA-GFDL" / "GFDL-ESM4.log"
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_text(HEADER + "14:12:37 - cst - INFO - nested body\n", encoding="utf-8")
    out = tmp_path / "merged.log"
    merge_logs(["2.01_fetch_gcm_raw"], str(out), str(parts_dir))
    text = out.read_text(encoding="utf-8")
    assert "== 2.01  fetch_gcm_raw" in text
    assert "-- NOAA-GFDL/GFDL-ESM4 " in text
    assert "nested body" in text


def test_part_without_a_header_survives_untouched(tmp_path):
    parts_dir = tmp_path / "_parts"
    parts_dir.mkdir()
    (parts_dir / "2.04_derive_change_factors.log").write_text(
        "first line, no header\nsecond line\n", encoding="utf-8"
    )
    out = tmp_path / "merged.log"
    merge_logs(["2.04_derive_change_factors"], str(out), str(parts_dir))
    text = out.read_text(encoding="utf-8")
    assert "first line, no header" in text and "second line" in text


def test_rule_without_a_part_is_reported_not_skipped(tmp_path):
    parts_dir = tmp_path / "_parts"
    parts_dir.mkdir()
    out = tmp_path / "merged.log"
    merge_logs(["2.11_extract_climate_grid"], str(out), str(parts_dir))
    text = out.read_text(encoding="utf-8")
    assert "== 2.11  extract_climate_grid" in text
    assert "no part from this run" in text


def test_remove_parts_clears_the_parts_tree(tmp_path):
    parts_dir = _parts(
        tmp_path,
        {"2.01_fetch_gcm_raw": ["modelA"], "2.04_derive_change_factors": None},
    )
    out = tmp_path / "logs" / "wf2_analyze_projections.log"
    merge_logs(
        ["2.01_fetch_gcm_raw", "2.04_derive_change_factors"],
        str(out),
        parts_dir,
        remove_parts=True,
    )
    assert out.exists()
    assert not (tmp_path / "_parts").exists()  # dir itself pruned once empty


def test_an_unlisted_orphan_dir_is_neither_merged_nor_deleted(tmp_path):
    """A part dir left by a renamed rule is not a label, so it is never read.

    `test_local` still holds `2.04_monthly_change/` from the pre-step-4d names.
    An orphan must not appear as a phantom section, and must not be swept up
    either -- deleting what this run does not own is not this rule's call.
    """
    parts_dir = _parts(tmp_path, {"2.01_fetch_gcm_raw": ["modelA"]})
    orphan = tmp_path / "_parts" / "2.04_monthly_change" / "old.log"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_text("stale\n", encoding="utf-8")

    out = tmp_path / "merged.log"
    merge_logs(["2.01_fetch_gcm_raw"], str(out), parts_dir, remove_parts=True)
    text = out.read_text(encoding="utf-8")
    assert orphan.exists()
    assert "monthly_change" not in text and "stale" not in text


def test_merge_creates_parent_dir(tmp_path):
    parts_dir = _parts(tmp_path, {"1.02_prepare_build_config": None})
    out = tmp_path / "nested" / "deep" / "merged.log"
    merge_logs(["1.02_prepare_build_config"], str(out), parts_dir)
    assert out.exists()
