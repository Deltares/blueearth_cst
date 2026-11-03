"""Gate 2 (a-h): pure script-unit tests for the wf3 drift-guard comparator.

These call ``compare_project_consistency`` directly on staged config/snapshot
pairs — NO Snakemake, no rerun-triggers (that is gate 2b,
``test_guard_invalidation.py``). Design: dev/records/milestones/p31/experiment-structure-design.md
§7 gate 2, §3b.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from blueearth_cst.experiment.check_project_consistency import (  # noqa: E402
    compare_project_consistency,
    guarded_paths,
    guarded_section_paths,
)

# A minimal but structurally faithful full config (R01 sectioned schema),
# mirroring test_case/project_config_baseline.yml at the guarded
# sections. The experiment sections that vary per experiment are present but
# deliberately NOT guarded.
_BASE_CFG = {
    "schema_version": 2,
    "project": {
        "project_dir": "test_case/test_local",
        # `C-07` deleted `static_dir`; `C-40` renamed `data_sources` to
        # `catalog`; `C-39` pushed the climate catalog DOWN to the WF2 file,
        # where it has its only reader.
        "catalog": "config/catalogs/deltares_data.yml",
    },
    "basin": {
        "region": "{'subbasin': [9.666, 0.4476], 'uparea': 100}",
        "resolution": 0.00833,
    },
    "climate": {
        "selected": "era5",
        "sources": ["era5"],
        # `C-70`: INCLUSIVE YEARS, not ISO timestamps.
        "window": {"start": 2000, "end": 2020},
    },
    # `C-19`: `wflow_outvars` is `model.outvars`. Still guarded as a LEAF, so a
    # post-build edit is refused at rule 3.01 — the mechanism is unchanged and
    # only the path moved.
    "model": {"outvars": ["river discharge"]},
    "workflows": {
        "build_model": {
            "enabled": True,
            # `C-22`: both paths sit under `engine:` now. The nesting matters to
            # this module specifically -- `_normalize_paths` recurses, so the
            # normalization map keys on the LEAF and has to find it at depth.
            "engine": {
                "build_config": "config/defaults/wflow_build_model.yml",
                "waterbodies_config": "config/defaults/wflow_update_waterbodies.yml",
            },
        },
        "analyze_projections": {
            "enabled": True,
            "clim_project": "cmip6",
            "models": ["NOAA-GFDL/GFDL-ESM4"],
            "scenarios": ["ssp245", "ssp585"],
        },
        "run_stress_test": {
            "experiment_name": "experiment",
            "realizations_num": 2,
        },
    },
}


def _write(path: Path, cfg: dict) -> Path:
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


#: Which `workflows.<name>` a given workflow's snapshot has actually COMPOSED.
#: Each Snakefile's projection names its own section and no other workflow's
#: settings, so this is one entry per snapshot the guard reads.
_COMPOSED_BY = {
    "build_model": {"build_model"},
    "analyze_projections": {"analyze_projections"},
}


def _snapshot_for(entry: str) -> dict:
    """What `copy_config_files` actually writes for ``entry``.

    A snapshot is that workflow's COMPOSED config, so every workflow outside its
    `R(entry)` survives as the bare `{enabled}` stanza the project file carried
    -- `compose_config` never opened those files.

    This fixture used to stage the FULL config as both snapshots, and that hid
    the one failure that would refuse every run of every project: under a rule
    that compares every key of the snapshot, a `{enabled: true}` stanza staged
    against a fully expanded live section diverges on sight. A synthetic
    snapshot that is more complete than a real one cannot fail that way.
    """
    doc = copy.deepcopy(_BASE_CFG)
    expanded = _COMPOSED_BY[entry]
    doc["workflows"] = {
        name: (
            section if name in expanded else {"enabled": section.get("enabled", True)}
        )
        for name, section in doc["workflows"].items()
    }
    return doc


@pytest.fixture()
def snapshots(tmp_path):
    """Write matching wf1 + wf2 snapshots from _BASE_CFG; return their paths."""
    wf1 = _write(
        tmp_path / "project_config_build_model.yml", _snapshot_for("build_model")
    )
    wf2 = _write(
        tmp_path / "project_config_analyze_projections.yml",
        _snapshot_for("analyze_projections"),
    )
    return wf1, wf2


def test_a_identical_sections_pass(snapshots):
    wf1, wf2 = snapshots
    diffs = compare_project_consistency(copy.deepcopy(_BASE_CFG), wf1, wf2)
    assert diffs == []


def test_b_mutated_basin_resolution_fails_naming_key(snapshots):
    wf1, wf2 = snapshots
    live = copy.deepcopy(_BASE_CFG)
    live["basin"]["resolution"] = 0.05
    diffs = compare_project_consistency(live, wf1, wf2)
    assert diffs
    assert any("basin" in d and "resolution" in d for d in diffs)


def test_c_mutated_build_model_fails(snapshots):
    wf1, wf2 = snapshots
    live = copy.deepcopy(_BASE_CFG)
    live["workflows"]["build_model"]["engine"]["waterbodies_config"] = "other.yml"
    diffs = compare_project_consistency(live, wf1, wf2)
    assert diffs
    assert any("workflows.build_model" in d for d in diffs)


def test_c2_mutated_shared_wflow_outvars_fails(snapshots):
    """The hoist must not WEAKEN this guard, and nothing else would catch it.

    ``wflow_outvars`` sat inside ``workflows.build_model`` until R13, so a
    post-build edit was refused here. Moved to ``shared:`` and left unguarded,
    the same edit would sail past rule 3.01 and first surface mid-experiment as
    ``export_wflow_results``' missing-column error -- a whole workflow away
    from its cause, and after the expensive part has run.

    P2 keeps the coverage and drops the leaf-by-leaf mechanism: `model:` is a
    top-level section of the snapshot, so the derived rule takes it whole. The
    narrowing existed to keep every comparand experiment-invariant while `seed`
    and `julia_threads` sat in the shared section, and `C-51`/`C-54` removed
    them.
    """
    wf1, wf2 = snapshots
    live = copy.deepcopy(_BASE_CFG)
    live["model"]["outvars"] = ["actual evapotranspiration"]
    diffs = compare_project_consistency(live, wf1, wf2)
    assert diffs
    assert any("model.outvars" in d for d in diffs), diffs


def test_d_flat_vs_binned_paths_pass(tmp_path):
    """Symmetric normalization: a flat-path experiment config vs a binned
    snapshot converges (design §3b, gate 2d)."""
    # Snapshot uses NEW binned catalog paths; experiment config uses OLD flat
    # paths for the mapped keys. Symmetric normalization makes them equal.
    snapshot_cfg = _snapshot_for("build_model")
    wf1 = _write(tmp_path / "project_config_build_model.yml", snapshot_cfg)

    live = copy.deepcopy(_BASE_CFG)
    live["project"]["catalog"] = "config\\deltares_data.yml"
    live["workflows"]["build_model"]["engine"]["build_config"] = (
        "config\\wflow_build_model.yml"
    )
    live["workflows"]["build_model"]["engine"]["waterbodies_config"] = (
        "config\\wflow_update_waterbodies.yml"
    )

    diffs = compare_project_consistency(live, wf1, wf2_snapshot_path=None)
    assert diffs == []


def test_e_missing_wf1_snapshot_fails_with_run_first_message(tmp_path):
    missing = tmp_path / "does_not_exist.yml"
    diffs = compare_project_consistency(copy.deepcopy(_BASE_CFG), missing)
    assert diffs
    assert any(
        "run build_model.smk first" in d.lower()
        or "run snakefile_build_model first" in d.lower()
        for d in diffs
    )


def test_f_mutated_climate_window_now_fails(snapshots):
    """INVERTED by P2 (design D-9.1 property 2): this is the hole it closes.

    Until now the climate window was outside the guard, so shortening it after
    a build was accepted and WF3 ran an experiment against a model forced over
    a different period -- the divergence the guard exists to catch, in the one
    section it did not cover. The assertion is kept rather than deleted, and
    inverted, so the change of behaviour is visible in the diff instead of
    looking like a test that quietly went away.
    """
    wf1, wf2 = snapshots
    live = copy.deepcopy(_BASE_CFG)
    live["climate"]["window"]["end"] = 2010
    diffs = compare_project_consistency(live, wf1, wf2)
    assert diffs
    assert any("climate.window" in d for d in diffs), diffs


def test_g_mutated_analyze_projections_with_wf2_snapshot_fails(snapshots):
    wf1, wf2 = snapshots
    live = copy.deepcopy(_BASE_CFG)
    live["workflows"]["analyze_projections"]["scenarios"] = ["ssp126"]
    diffs = compare_project_consistency(live, wf1, wf2)
    assert diffs
    assert any("workflows.analyze_projections" in d for d in diffs)


def test_h_mutated_analyze_projections_without_wf2_snapshot_passes(snapshots):
    wf1, _ = snapshots
    live = copy.deepcopy(_BASE_CFG)
    live["workflows"]["analyze_projections"]["scenarios"] = ["ssp126"]
    # No wf2 snapshot -> projections section unchecked + logged, passes.
    diffs = compare_project_consistency(live, wf1, wf2_snapshot_path=None)
    assert diffs == []


# ---------------------------------------------------------------------------
# P2: the derived rule's two structural exceptions, and the three sites
# ---------------------------------------------------------------------------


def test_i_an_unexpanded_stanza_is_not_compared(snapshots):
    """Exception 2, and the failure that would refuse every run everywhere.

    A WF1 snapshot carries `workflows.run_stress_test: {enabled: true}` and
    nothing more, because WF1 never opened that file. The live WF3 config has
    the section fully expanded. A rule that compares every key of the snapshot
    without this exception compares those two and diverges on the first run of
    any project -- and the failure would look like a config error rather than a
    guard defect, because the message names a real key that really does differ.
    """
    wf1, wf2 = snapshots
    live = copy.deepcopy(_BASE_CFG)
    live["workflows"]["run_stress_test"]["realizations_num"] = 40
    diffs = compare_project_consistency(live, wf1, wf2)
    assert diffs == [], diffs


def test_j_toggling_a_workflow_does_not_refuse(snapshots):
    """Exception 1: `enabled` is dropped from both operands.

    Switching the projections overlay off does not change what the model was
    built from, and the overlay is optional under the CST method. Without this
    the derived rule would refuse an experiment for a change that cannot affect
    a single number in it.
    """
    wf1, wf2 = snapshots
    live = copy.deepcopy(_BASE_CFG)
    live["workflows"]["analyze_projections"]["enabled"] = False
    live["workflows"]["build_model"]["enabled"] = False
    diffs = compare_project_consistency(live, wf1, wf2)
    assert diffs == [], diffs


def test_the_derived_list_covers_everything_the_maintained_tuples_did(snapshots):
    """No coverage is lost in the move from three literals to one rule.

    The old tuples are written out here as the thing to be beaten, not imported
    -- they no longer exist. `climate` and `schema_version` are the additions,
    and they are asserted separately so a future narrowing of the rule cannot
    quietly drop them while this test still passes on the old four.
    """
    wf1, wf2 = snapshots
    retired_wf1 = {
        "project",
        ("basin",),
        ("model", "outvars"),
        ("workflows", "build_model"),
    }
    covered = set(guarded_paths(yaml.safe_load(wf1.read_text(encoding="utf-8"))))

    # Leaf paths are covered by their SECTION now, which is wider, not narrower.
    assert "project" in covered
    assert "basin" in covered
    assert "model" in covered
    assert ("workflows", "build_model") in covered
    assert len(retired_wf1) == 4  # the tuple this replaces, for the reader

    assert "climate" in covered, "the hole D-9.1 exists to close"
    assert "schema_version" in covered

    # WF3's own section is never in a WF1 snapshot's list (design D-9.2), which
    # is why `compute:` needs no carve-out here.
    assert ("workflows", "run_stress_test") not in covered

    wf2_covered = set(guarded_paths(yaml.safe_load(wf2.read_text(encoding="utf-8"))))
    assert ("workflows", "analyze_projections") in wf2_covered


def test_the_parse_time_form_agrees_with_the_snapshot_derived_one(snapshots):
    """Sites (1) and (2)/(3) must name the same set, by construction.

    They cannot share an implementation -- the Snakefile has no snapshot to read
    at parse time -- so this is what holds them together. `D-9.6` records that
    the three sites were previously hand-kept and two of them disagreed: R13
    hoisted `wflow_outvars` out of `workflows.build_model`, taught the
    comparator about the leaf, and left the rerun trigger covering it only
    through the container it had just left.
    """
    wf1, wf2 = snapshots
    from_snapshots = set(guarded_paths(yaml.safe_load(wf1.read_text(encoding="utf-8"))))
    from_snapshots |= set(
        guarded_paths(yaml.safe_load(wf2.read_text(encoding="utf-8")))
    )
    # WF3's section appears in the WF2 snapshot no more than in the WF1 one.
    dotted = {p if isinstance(p, str) else ".".join(p) for p in from_snapshots}

    assert dotted == set(guarded_section_paths()), (
        "the guard compares one set of paths and the Snakefile threads another; "
        "an edit to the difference is refused without re-firing the guard"
    )
