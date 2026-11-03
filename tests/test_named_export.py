"""Named per-set exports: dictionary, ownership, refresh, rollback and recovery."""

import csv
import io
import shutil
import textwrap
from pathlib import Path

import pytest

from blueearth_cst.experiment import indicator_sets, named_export
from blueearth_cst.experiment.content_identity import (
    canonical_json_bytes,
    content_sha256,
    identity_segment,
    read_canonical_json,
)
from blueearth_cst.experiment.indicator_plan import (
    current_indicator_request,
    publish_indicator_set,
    write_indicator_plan,
)
from blueearth_cst.experiment.named_export import NamedExportError
from tests._v2_experiment import (
    FIXTURE_ENVIRONMENT,
    build_v2_experiment,
    copy_v2_experiment,
)

ALIAS = """\
schema_version: 1
id: depth_alias
title: Recharge depth alias
description: Authored alias.
selection:
  indicators:
    - id: recharge_depth
      title: Recharge depth
      description: Same calculation as gwr_annual_total.
      references: [Example citation]
      calculation:
        method: annual_summary/1
        parameters: {variable: gwr, statistic: sum}
"""


def _live(monkeypatch):
    from blueearth_cst.experiment import indicator_plan

    monkeypatch.setattr(
        indicator_plan, "resolve_indicator_environment", lambda: FIXTURE_ENVIRONMENT
    )


def _default():
    source = indicator_sets.legacy_selection({"operation": "indicators-only"}).sets[0]
    return indicator_sets.resolve_set(source, ["gwr"])


def _alias(text=ALIAS):
    return indicator_sets.resolve_set(
        indicator_sets.SetSource("depth_alias", Path("depth_alias.yml"), False, text)
    )


def _manifest(root, resolved):
    request = current_indicator_request(
        root, indicator_sets.calculation_records(resolved.indicators), "YE-DEC"
    )
    plan = write_indicator_plan(root, request)
    publish_indicator_set(root, plan)
    return Path(plan["targets"]["manifest"])


@pytest.fixture(scope="module")
def _published_once(tmp_path_factory):
    experiment = build_v2_experiment(tmp_path_factory.mktemp("named"))
    with pytest.MonkeyPatch.context() as patch:
        from blueearth_cst.experiment import indicator_plan

        patch.setattr(
            indicator_plan, "resolve_indicator_environment", lambda: FIXTURE_ENVIRONMENT
        )
        default = _manifest(experiment.root, _default())
        alias = _manifest(experiment.root, _alias())
    return (
        experiment,
        default.relative_to(experiment.root),
        alias.relative_to(experiment.root),
    )


@pytest.fixture
def ready(_published_once, tmp_path, monkeypatch):
    _live(monkeypatch)
    source, default, alias = _published_once
    root = copy_v2_experiment(source, tmp_path).root
    return root, root / default, root / alias


def _publish(root, resolved, manifest):
    return named_export.publish_named_set(
        root, resolved, manifest, water_year_start="Jan"
    )


def _dictionary(root, set_id):
    text = (root / "results/indicator_sets" / set_id / "indicators.csv").read_text(
        encoding="utf-8"
    )
    return list(csv.DictReader(io.StringIO(text)))


def test_pre_rename_named_publication_remains_readable(ready):
    root, default, _ = ready
    current = _publish(root, _default(), default)
    legacy = dict(current)
    legacy["schema_version"] = "named-metric-export/1"
    legacy["metric_set_id"] = legacy.pop("indicator_set_id")
    legacy["metric_manifest"] = legacy.pop("indicator_manifest")
    legacy["publication_id"] = content_sha256(
        {k: v for k, v in legacy.items() if k not in {"publication_id", "created_at"}}
    )
    from blueearth_cst.experiment.content_identity import atomic_record
    from blueearth_cst.shared.provenance import file_sha256

    names = root / "_engine/metric_sets/names/default"
    names.mkdir(parents=True)
    record_path = names / "publication.json"
    atomic_record(record_path, legacy)
    atomic_record(
        names / "current.json",
        {
            "schema_version": "named-metric-current/1",
            "publication_id": legacy["publication_id"],
            "publication": {
                "path": record_path.relative_to(root).as_posix(),
                "sha256": file_sha256(record_path),
            },
        },
    )
    shutil.copytree(
        root / "results/indicator_sets/default", root / "results/metric_sets/default"
    )
    # Keep all referenced immutable objects but remove the new current pointer,
    # as in a project which has only published the old layout.
    (root / "_engine/indicator_sets/names/default/current.json").unlink()
    before = {p: p.read_bytes() for p in names.rglob("*") if p.is_file()}
    loaded = named_export.read_current_publication(root, "default")
    assert loaded["indicator_set_id"] == current["indicator_set_id"]
    assert loaded["publication_id"] == legacy["publication_id"]
    assert before == {p: p.read_bytes() for p in names.rglob("*") if p.is_file()}
    export = root / "results/metric_sets/default/indicators.csv"
    export.write_bytes(export.read_bytes() + b"x")
    with pytest.raises(NamedExportError, match="differs from its committed inventory"):
        named_export.read_current_publication(root, "default")


def test_publication_layout_dictionary_and_no_machine_leakage(ready):
    root, default, _ = ready
    record = _publish(root, _default(), default)
    folder = root / "results/indicator_sets/default"
    names = {p.name for p in folder.iterdir()}
    assert names == {
        "gwr_run_indicators.csv",
        "gwr_group_indicators.csv",
        "group_members.csv",
        "group_metadata.csv",
        "indicators.csv",
    }
    assert not any(p.suffix == ".json" for p in folder.iterdir())
    rows = _dictionary(root, "default")
    assert list(rows[0]) == list(named_export.DICTIONARY_COLUMNS)
    (row,) = rows
    assert row["indicator"] == "gwr_annual_total"
    assert row["output_units"] == "mm" and row["input_units"] == "mm dt-1"
    assert "not a duration-normalized rate" in row["quantity"]
    assert row["equivalent_definition"] == "" and row["authored_references"] == ""
    assert row["grain"] == "run" and row["water_year_start"] == "Jan"
    assert row["block_assignment"].startswith("YE-DEC (right-closed, right-labelled")
    assert "min_count=0" in row["missingness"]
    text = folder.joinpath("indicators.csv").read_text(encoding="utf-8")
    for leak in (record["indicator_set_id"], record["publication_id"], "{", "sha256"):
        assert leak not in text
    current = named_export.read_current_publication(root, "default")
    assert current == record
    engine = root / "_engine/indicator_sets/names/default"
    assert {p.name for p in engine.iterdir()} == {
        "current.json",
        "definitions",
        "publications",
    }


def test_shared_definition_rows_are_identical_and_alias_differs_only_by_key(ready):
    root, default, alias = ready
    _publish(root, _default(), default)
    _publish(root, _alias(), alias)
    for scope in ("run", "group"):
        name = f"gwr_{scope}_indicators.csv"
        registered = (root / "results/indicator_sets/default" / name).read_text(
            encoding="utf-8"
        )
        authored = (root / "results/indicator_sets/depth_alias" / name).read_text(
            encoding="utf-8"
        )
        assert authored.replace("recharge_depth", "gwr_annual_total") == registered
    (row,) = _dictionary(root, "depth_alias")
    assert row["indicator"] == "recharge_depth"
    assert row["equivalent_definition"] == "gwr_annual_total"
    assert row["authored_references"] == "Example citation" and row["references"] == ""


def test_identical_requests_reuse_numbers_and_presentation_edits_refresh_visibly(ready):
    root, _, alias = ready
    first = _publish(root, _alias(), alias)
    again = _publish(root, _alias(), alias)
    assert again == first
    edited = _alias(ALIAS.replace("title: Recharge depth\n", "title: Edited title\n"))
    calculations = indicator_sets.calculation_records(edited.indicators)
    assert calculations == indicator_sets.calculation_records(_alias().indicators)
    refreshed = _publish(root, edited, alias)
    assert refreshed["indicator_set_id"] == first["indicator_set_id"]
    assert refreshed["publication_id"] != first["publication_id"]
    assert _dictionary(root, "depth_alias")[0]["title"] == "Edited title"


def test_block_coverage_lists_every_pair_in_deterministic_order(ready):
    root, default, _ = ready
    _publish(root, _default(), default)
    cell = _dictionary(root, "default")[0]["block_coverage"]
    evidence = read_canonical_json(default.parent / "coverage_evidence.json")["series"]
    pairs = sorted((item["run_id"], item["location_id"]) for item in evidence)
    for run, location in pairs:
        assert f'"{run} at location {location}" (evaluated)' in cell
    positions = [
        cell.index(f'"{run} at location {location}"') for run, location in pairs
    ]
    assert positions == sorted(positions)
    assert "blocks [" in cell and "complete label blocks" in cell


def test_coverage_text_separates_differing_windows_and_masks():
    def series(run, location, first, last, missing):
        return {
            "run_id": run,
            "variable": "q",
            "location_id": location,
            "calendar": "standard",
            "timestep_seconds": 86400,
            "time_label": "interval_end",
            "anchor": "YS-JAN",
            "first_label": first,
            "last_label": last,
            "blocks": [
                {
                    "block_label": first[:4] + "-01-01",
                    "first_label": first,
                    "last_label": last,
                    "observed": 10,
                    "expected": 365,
                    "missing": missing,
                    "all_missing": False,
                }
            ],
            "block_count": 1,
            "complete_label_blocks": 0,
            "missing_total": missing,
            "all_missing_blocks": 0,
        }

    evidence = [
        series("run_02", "1", "2001-01-01", "2001-01-10", 0),
        series("run_01", "1", "2000-01-01", "2000-01-10", 0),
        series("run_01", "2", "2000-01-01", "2000-01-10", 3),
    ]
    pairs = [
        ("run_01", "q", "1", "evaluated"),
        ("run_01", "q", "2", "evaluated"),
        ("run_02", "q", "1", "evaluated"),
    ]
    cell = named_export.block_coverage(evidence, pairs)
    patterns = cell.split(" | ")
    assert len(patterns) == 3
    assert patterns[0].startswith('"run_01 at location 1"')
    assert "missing 3" in patterns[1] and patterns[1].startswith(
        '"run_01 at location 2"'
    )
    assert patterns[2].startswith('"run_02 at location 1"')
    assert named_export.block_coverage(list(reversed(evidence)), pairs[::-1]) == cell


def test_series_coverage_keeps_window_and_mask_with_their_run_and_location():
    """Real series_coverage output, not hand-built evidence, through block_coverage."""
    from datetime import timedelta

    import numpy as np
    import pandas as pd

    from blueearth_cst.experiment.indicator_registry import series_coverage
    from blueearth_cst.experiment.response_series import ResponseSeries

    def series(run, location, start, end, missing_days=0):
        time = pd.date_range(start, end, freq="D")
        missing = np.zeros(len(time), dtype=bool)
        missing[:missing_days] = True
        values = np.where(missing, np.nan, 1.0)
        return ResponseSeries(
            run,
            "q",
            location,
            int(location),
            time,
            "standard",
            timedelta(days=1),
            "interval_end",
            "m3 s-1",
            values,
            missing,
        )

    responses = [
        series("run_02", "2", "2001-03-01", "2001-03-20"),
        series("run_01", "2", "2000-12-25", "2001-01-05", missing_days=3),
        series("run_02", "1", "2001-03-01", "2001-03-20"),
        series("run_01", "1", "2000-12-25", "2001-01-05"),
    ]
    evidence = [series_coverage(item, "YE-DEC") for item in responses]
    pairs = sorted(
        (item.run_id, "q", item.location_id, "evaluated") for item in responses
    )
    cell = named_export.block_coverage(evidence, pairs)
    patterns = cell.split(" | ")
    assert len(patterns) == 3
    assert patterns[0].startswith('"run_01 at location 1" (evaluated):')
    assert "labels 2000-12-25..2001-01-05; missing 0;" in patterns[0]
    assert "2 blocks" in patterns[0]
    assert patterns[1].startswith('"run_01 at location 2" (evaluated):')
    assert "labels 2000-12-25..2001-01-05; missing 3;" in patterns[1]
    assert patterns[2].startswith(
        '"run_02 at location 1" (evaluated), "run_02 at location 2" (evaluated):'
    )
    assert "labels 2001-03-01..2001-03-20; missing 0;" in patterns[2]
    assert all("anchor YE-DEC" in pattern for pattern in patterns)
    assert named_export.block_coverage(list(reversed(evidence)), pairs[::-1]) == cell


def test_unowned_and_legacy_claimed_folders_are_refused_with_bytes_preserved(ready):
    root, default, _ = ready
    folder = root / "results/indicator_sets/default"
    folder.mkdir(parents=True)
    with pytest.raises(NamedExportError, match="no committed publication"):
        _publish(root, _default(), default)
    (folder / "notes.csv").write_bytes(b"user")
    with pytest.raises(NamedExportError, match="no committed publication"):
        _publish(root, _default(), default)
    assert (folder / "notes.csv").read_bytes() == b"user"
    folder.joinpath("notes.csv").unlink()
    folder.rmdir()
    # A read-only /3 set claiming results/indicator_sets/<id> blocks that set id.
    claimed = root / "_engine/indicator_sets/aaaaaaaaaaaa"
    claimed.mkdir(parents=True)
    claimed.joinpath("indicators.json").write_bytes(
        canonical_json_bytes({"result_root": "results/indicator_sets/default"})
    )
    with pytest.raises(NamedExportError, match="legacy indicator set aaaaaaaaaaaa"):
        _publish(root, _default(), default)


def test_refresh_refuses_unknown_files_but_restores_owned_bytes(ready):
    root, default, _ = ready
    record = _publish(root, _default(), default)
    folder = root / "results/indicator_sets/default"
    (folder / "gwr_run_indicators.csv").write_bytes(b"user edit")
    with pytest.raises(NamedExportError, match="differs from its committed"):
        named_export.read_current_publication(root, "default")
    assert _publish(root, _default(), default) == record
    assert named_export.read_current_publication(root, "default") == record
    (folder / "extra.csv").write_bytes(b"x")
    with pytest.raises(NamedExportError, match="does not own"):
        _publish(root, _default(), default)
    assert (folder / "extra.csv").read_bytes() == b"x"


@pytest.mark.parametrize("failure", ["copy", "replace", "pointer"])
def test_failed_refresh_rolls_back_to_the_committed_export(ready, monkeypatch, failure):
    root, default, _ = ready
    first = _publish(root, _default(), default)
    folder = root / "results/indicator_sets/default"
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    edited = indicator_sets.resolve_set(
        indicator_sets.SetSource(
            "default",
            indicator_sets.BUNDLED_ROOT / "default.yml",
            True,
            (indicator_sets.BUNDLED_ROOT / "default.yml")
            .read_text(encoding="utf-8")
            .replace("title: Default indicators", "title: Renamed"),
        ),
        ["gwr"],
    )

    def fail(*args, **kwargs):
        raise OSError("injected")

    if failure == "copy":
        monkeypatch.setattr(named_export.shutil, "copyfile", fail)
    elif failure == "replace":
        real = named_export.os.replace
        calls = []

        def second(source, target):
            calls.append(target)
            if len(calls) == 2:
                fail()
            return real(source, target)

        monkeypatch.setattr(named_export.os, "replace", second)
    else:
        real_record = named_export.atomic_record

        def pointer(path, record, **kwargs):
            if Path(path).name == "current.json":
                fail()
            return real_record(path, record, **kwargs)

        monkeypatch.setattr(named_export, "atomic_record", pointer)
    with pytest.raises((OSError, NamedExportError)):
        _publish(root, edited, default)
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before
    assert named_export.read_current_publication(root, "default") == first
    monkeypatch.undo()
    _live(monkeypatch)
    assert _publish(root, edited, default)["publication_id"] != first["publication_id"]


def test_publication_record_is_closed_and_digest_bound(ready):
    root, default, _ = ready
    record = _publish(root, _default(), default)
    current = read_canonical_json(
        root / "_engine/indicator_sets/names/default/current.json"
    )
    path = root / current["publication"]["path"]
    tampered = dict(record, resolved_metadata=[])
    path.write_bytes(canonical_json_bytes(tampered))
    with pytest.raises(NamedExportError, match="digest differs"):
        named_export.read_current_publication(root, "default")
    semantic = {
        k: v for k, v in record.items() if k not in {"publication_id", "created_at"}
    }
    assert record["publication_id"] == content_sha256(semantic)
    assert (
        identity_segment(record["indicator_set_id"], "indicator_set_id")
        in record["indicator_manifest"]["path"]
    )
    assert textwrap.dedent(record["definition_source"]["path"]).endswith(".yml")


def test_sets_complete_independently_and_failures_keep_prior_publications(
    ready, capsys, monkeypatch
):
    from blueearth_cst.experiment import indicator_export, indicator_plan

    observations = []
    observe = indicator_plan._indicator_code_inventory

    def code(set_ids):
        observations.append(set_ids)
        return observe(set_ids)

    monkeypatch.setattr(indicator_plan, "_indicator_code_inventory", code)

    root, default, _ = ready
    q_only = indicator_sets.SetSource(
        "q_only",
        Path("q_only.yml"),
        False,
        "schema_version: 1\nid: q_only\ntitle: q\ndescription: d\n"
        "selection:\n  indicators:\n    - definition: q_annual_mean\n",
    )
    default_source = indicator_sets.legacy_selection(
        {"operation": "indicators-only"}
    ).sets[0]
    selection = indicator_sets.Selection(
        ("indicators",), (default_source, q_only), legacy=False
    )
    project = {"model": {}}
    settings = {"operation": "indicators-only"}
    outcomes = indicator_export.publish_selected_sets(
        root, project, settings, selection
    )
    assert observations == [("default",)]
    assert [(o["set_id"], o["status"]) for o in outcomes] == [
        ("default", "current"),
        ("q_only", "failed"),
    ]
    assert outcomes[1]["prior_publication_id"] is None
    assert not (root / "results/indicator_sets/q_only").exists()
    assert indicator_export.report_publication(root, outcomes) is False
    first = outcomes[0]["publication_id"]
    # A later failure of the same set keeps its committed export (mixed age).
    folder = root / "results/indicator_sets/default"
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    for plan in (root / "_engine/indicator_requests").glob("*.json"):
        plan.unlink()
    again = indicator_export.publish_selected_sets(root, project, settings, selection)
    assert observations == [("default",), ("default",)]
    assert again[0]["status"] == "failed"
    assert again[0]["prior_publication_id"] == first
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before
    indicator_export.report_publication(root, again)
    assert (
        "previous results kept; they have not been updated" in capsys.readouterr().out
    )


def test_a_failed_rollback_keeps_backups_and_reports_them(ready, monkeypatch):
    root, default, _ = ready
    _publish(root, _default(), default)
    real = named_export.os.replace
    calls = []

    def failing(source, target):
        # Fail the second export-file replacement and every rollback restore.
        if Path(source).name.endswith(".previous"):
            raise OSError("injected rollback failure")
        if (
            Path(target).parent.name == "default"
            and Path(target).parent.parent.name == "indicator_sets"
        ):
            calls.append(target)
            if len(calls) >= 2:
                raise OSError("injected")
        return real(source, target)

    source = indicator_sets.SetSource(
        "default",
        indicator_sets.BUNDLED_ROOT / "default.yml",
        True,
        (indicator_sets.BUNDLED_ROOT / "default.yml")
        .read_text(encoding="utf-8")
        .replace("title: Default indicators", "title: Renamed"),
    )
    renamed = indicator_sets.resolve_set(source, ["gwr"])
    monkeypatch.setattr(named_export.os, "replace", failing)
    with pytest.raises(NamedExportError, match="rollback failed; backups kept"):
        _publish(root, renamed, default)
    monkeypatch.undo()
    stages = list((root / "_engine/indicator_sets/names/default").glob(".stage-*"))
    assert len(stages) == 1
    assert any(p.name.endswith(".previous") for p in stages[0].iterdir())
