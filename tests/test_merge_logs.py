"""Tests for the WF2 workflow-log merge (blueearth_cst/shared/merge_logs.py)."""

from blueearth_cst.shared.merge_logs import merge_logs  # noqa: E402

HEADER = (
    "# BlueEarth-CST | project: gabonx | 2026-07-31\n"
    "# project dir: C:/TESTS/CST/gabonx\n"
    "# log: 2.01_fetch_gcm_raw/modelA | started 14:12:37\n"
    "\n"
)


def _parts(tmp_path, layout):
    """Materialise ``{rule: [member, ...] | None}`` under ``tmp_path/_parts``.

    Returns ``(parts_dir, ordered_part_paths)``.
    """
    parts_dir = tmp_path / "_parts"
    paths = []
    for rule, members in layout.items():
        if members is None:
            p = parts_dir / f"{rule}.log"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(HEADER + f"14:12:37 - cst - INFO - body of {rule}\n", encoding="utf-8")
            paths.append(str(p))
            continue
        for member in members:
            p = parts_dir / rule / f"{member}.log"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(HEADER + f"14:12:37 - cst - INFO - body of {member}\n", encoding="utf-8")
            paths.append(str(p))
    return str(parts_dir), paths


def test_one_header_then_a_banner_per_rule(tmp_path):
    parts_dir, paths = _parts(
        tmp_path,
        {"2.01_fetch_gcm_raw": ["modelA", "modelB"], "2.04_derive_change_factors": None},
    )
    out = tmp_path / "logs" / "wf2_climate_projections.log"
    merge_logs(paths, str(out), parts_dir=parts_dir)
    text = out.read_text(encoding="utf-8")

    # exactly one provenance header, and it is the merged log's own
    assert text.count("# BlueEarth-CST") == 1
    assert text.startswith("# BlueEarth-CST")
    assert "# log: wf2_climate_projections.log | merged " in text
    # per-part headers are stripped, bodies survive
    assert "started 14:12:37" not in text
    assert "body of modelA" in text and "body of 2.04_derive_change_factors" in text

    # one `==` banner per rule, tagged as the console banner tags it
    assert "== 2.01  fetch_gcm_raw" in text
    assert "== 2.04  derive_change_factors" in text
    # fan-out members get their own sub-header inside the rule's section
    assert "-- modelA " in text and "-- modelB " in text
    # a single-job rule has no member sub-header
    assert "-- 2.04_derive_change_factors" not in text


def test_sections_follow_the_given_order(tmp_path):
    parts_dir, paths = _parts(
        tmp_path,
        {"2.01_fetch_gcm_raw": ["modelA", "modelB"], "2.04_derive_change_factors": None},
    )
    out = tmp_path / "merged.log"
    merge_logs(paths, str(out), parts_dir=parts_dir)
    text = out.read_text(encoding="utf-8")
    assert text.index("fetch_gcm_raw") < text.index("derive_change_factors")
    assert text.index("body of modelA") < text.index("body of modelB")


def test_part_without_a_header_survives_untouched(tmp_path):
    parts_dir = tmp_path / "_parts"
    parts_dir.mkdir()
    part = parts_dir / "2.04_derive_change_factors.log"
    part.write_text("first line, no header\nsecond line\n", encoding="utf-8")
    out = tmp_path / "merged.log"
    merge_logs([str(part)], str(out), parts_dir=str(parts_dir))
    text = out.read_text(encoding="utf-8")
    assert "first line, no header" in text and "second line" in text


def test_absent_part_is_reported_not_skipped(tmp_path):
    parts_dir = tmp_path / "_parts"
    parts_dir.mkdir()
    missing = parts_dir / "2.06_plot_climate_proj_timeseries.log"
    out = tmp_path / "merged.log"
    merge_logs([str(missing)], str(out), parts_dir=str(parts_dir))
    text = out.read_text(encoding="utf-8")
    assert "== 2.06  plot_climate_proj_timeseries" in text
    assert "no part from this run" in text


def test_remove_parts_clears_the_parts_tree(tmp_path):
    parts_dir, paths = _parts(
        tmp_path,
        {"2.01_fetch_gcm_raw": ["modelA"], "2.04_derive_change_factors": None},
    )
    out = tmp_path / "logs" / "wf2_climate_projections.log"
    merge_logs(paths, str(out), parts_dir=parts_dir, remove_parts=True)
    assert out.exists()
    assert not (tmp_path / "_parts").exists()  # dir itself pruned once empty


def test_remove_parts_leaves_an_orphan_dir_alone(tmp_path):
    """A part dir left by a renamed rule is neither merged nor deleted.

    `test_local` still holds `2.04_monthly_change/` from the pre-step-4d names.
    An orphan must not appear as a phantom section, and must not be swept up
    either -- deleting what this run does not own is not this rule's call.
    """
    parts_dir, paths = _parts(tmp_path, {"2.01_fetch_gcm_raw": ["modelA"]})
    orphan = tmp_path / "_parts" / "2.04_monthly_change" / "old.log"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_text("stale\n", encoding="utf-8")

    out = tmp_path / "merged.log"
    merge_logs(paths, str(out), parts_dir=parts_dir, remove_parts=True)
    assert orphan.exists()
    assert "monthly_change" not in out.read_text(encoding="utf-8")
    assert "stale" not in out.read_text(encoding="utf-8")


def test_merge_creates_parent_dir(tmp_path):
    part = tmp_path / "p.log"
    part.write_text("x\n", encoding="utf-8")
    out = tmp_path / "nested" / "deep" / "merged.log"
    merge_logs([str(part)], str(out))
    assert out.exists()
