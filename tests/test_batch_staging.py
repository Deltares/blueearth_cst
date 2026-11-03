"""Staging keeps a failed batch's finished members for the retry (t2608071217)."""

from __future__ import annotations

import os

import pytest

from blueearth_cst.experiment.batch_staging import (
    BatchMember,
    members_from_records,
    prepare_staging,
    restore_staged,
)


def _members(tmp_path, ids):
    out = []
    for run in ids:
        toml = tmp_path / "settings" / f"{run}.toml"
        toml.parent.mkdir(parents=True, exist_ok=True)
        toml.write_text(f"run = {run}\n", encoding="utf-8")
        out.append(BatchMember(run, toml, tmp_path / "output" / f"{run}.csv"))
    return out


def _driver(staging, members, fail=()):
    """What run_wflow_batch.jl does per member, without Julia."""
    for member in members:
        if member.run_id in fail:
            member.native_output_path.parent.mkdir(parents=True, exist_ok=True)
            member.native_output_path.write_text("partial", encoding="utf-8")
            continue
        member.native_output_path.parent.mkdir(parents=True, exist_ok=True)
        member.native_output_path.write_text(f"csv {member.run_id}", encoding="utf-8")
        os.replace(member.native_output_path, staging / f"{member.run_id}.csv")
        os.replace(
            staging / f"{member.run_id}.expect",
            staging / f"{member.run_id}.ok",
        )


def test_records_parse_into_members(tmp_path):
    members = members_from_records(
        ["3", "run_01", "a.toml", "a.csv", "run_02", "b.toml", "b.csv"]
    )
    assert [m.run_id for m in members] == ["run_01", "run_02"]
    assert str(members[1].native_output_path) == "b.csv"


def test_a_retry_runs_only_the_failed_member(tmp_path):
    staging = tmp_path / "_engine" / "batch_staging"
    members = _members(tmp_path, ["run_01", "run_02", "run_03"])

    pending = prepare_staging(members, staging, "sim")
    assert pending == members
    _driver(staging, pending, fail={"run_02"})

    retry = prepare_staging(members, staging, "sim")
    assert [m.run_id for m in retry] == ["run_02"]
    _driver(staging, retry)
    restore_staged(members, staging)

    for member in members:
        assert (
            member.native_output_path.read_text(encoding="utf-8")
            == f"csv {member.run_id}"
        )
    assert not staging.exists()


def test_a_changed_toml_or_simulation_discards_the_staged_member(tmp_path):
    staging = tmp_path / "staging"
    members = _members(tmp_path, ["run_01", "run_02"])
    _driver(staging, prepare_staging(members, staging, "sim"))

    members[0].toml_path.write_text("run = changed\n", encoding="utf-8")
    assert [m.run_id for m in prepare_staging(members, staging, "sim")] == ["run_01"]
    assert len(prepare_staging(members, staging, "other-sim")) == 2


def test_a_csv_without_its_marker_is_not_trusted(tmp_path):
    """The process died between the CSV move and the marker rename."""
    staging = tmp_path / "staging"
    members = _members(tmp_path, ["run_01"])
    prepare_staging(members, staging, "sim")
    (staging / "run_01.csv").write_text("torn", encoding="utf-8")

    assert prepare_staging(members, staging, "sim") == members
    assert not (staging / "run_01.csv").exists()


def test_reuse_survives_regrouping_into_different_batches(tmp_path):
    """Batch size depends on --cores, so a retry may regroup the members."""
    staging = tmp_path / "staging"
    members = _members(tmp_path, ["run_01", "run_02", "run_03", "04"])
    first, second = members[:2], members[2:]
    _driver(staging, prepare_staging(first, staging, "sim"))
    _driver(staging, prepare_staging(second, staging, "sim"), fail={"04"})

    regrouped = [members[0], members[3]]
    assert [m.run_id for m in prepare_staging(regrouped, staging, "sim")] == ["04"]


def test_nothing_left_to_run_still_restores(tmp_path):
    staging = tmp_path / "staging"
    members = _members(tmp_path, ["run_01"])
    _driver(staging, prepare_staging(members, staging, "sim"))
    assert prepare_staging(members, staging, "sim") == []
    restore_staged(members, staging)
    assert members[0].native_output_path.is_file()


def test_restore_refuses_a_short_batch(tmp_path):
    staging = tmp_path / "staging"
    members = _members(tmp_path, ["run_01", "run_02"])
    _driver(staging, prepare_staging(members, staging, "sim"), fail={"run_02"})
    with pytest.raises(RuntimeError, match="1 member"):
        restore_staged(members, staging)
