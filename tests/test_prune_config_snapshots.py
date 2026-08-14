"""Unit tests for dev/scripts/prune_config_snapshots.py.

A one-shot migration for the config-snapshot redesign (2026-08-13): the
content-addressed bundles are no longer written by anything, and Snakemake
cannot clean a directory it no longer declares.

These pin the SCOPE, not the reporting. Deleting too little is a nuisance;
deleting one of the three protected classes destroys something no rule
regenerates, so each has its own test naming what would be lost.
"""

import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dev", "scripts"))
import prune_config_snapshots as pcs  # noqa: E402


@pytest.fixture()
def project(tmp_path):
    """A project carrying every shape the tool has to discriminate between."""
    root = tmp_path / "proj"

    bundles = [
        root / "config/runs/model_creation/1a22a14838f3",
        root / "config/runs/climate_projections/61868971c618",
        root / "experiments/exp1/config/runs/climate_experiment/278159763309",
    ]
    for bundle in bundles:
        bundle.mkdir(parents=True)
        (bundle / "source.yml").write_text("x\n", encoding="utf-8")

    keep = {
        # The GENERATED experiment catalog, written over generated forcing at
        # run time. The design keeps it, and a pattern match on
        # `config/catalogs/` across the whole tree would delete it.
        root / "experiments/exp1/config/catalogs/data_catalog_climate_experiment.yml",
        # Outside-repo files the predicate copies BY DESIGN. Losing these costs
        # the project its record of what it was evaluated against.
        root / "config/basin_data/output_locations.csv",
        # The wrapper's per-invocation manifests: a sibling of the bundles, not
        # one of them.
        root / "config/runs/invocations/20260811T142556.501Z-83c05db9c855.json",
        # A site-specific catalog that lives nowhere in the toolbox -- exactly
        # the file the copy policy exists to protect.
        root / "config/catalogs/my_site_catalog.yml",
    }
    for path in keep:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("site specific\n", encoding="utf-8")

    config = tmp_path / "cfg.yml"
    config.write_text(
        yaml.safe_dump({"project": {"project_dir": str(root)}}), encoding="utf-8"
    )
    return root, config, keep


def test_bundles_are_found_in_the_project_and_the_experiment(project):
    """Both trees accumulated them, so both have to be cleared."""
    root, _config, _keep = project

    found = {p.relative_to(root).as_posix() for p in pcs.find_bundles(root)}

    assert found == {
        "config/runs/model_creation/1a22a14838f3",
        "config/runs/climate_projections/61868971c618",
        "experiments/exp1/config/runs/climate_experiment/278159763309",
    }


def test_the_invocation_manifests_are_not_bundles(project):
    """`invocations/` sits beside the bundles and holds the wrapper's record.

    It has no digest level, so it cannot match the bundle shape -- but it lives
    under the same `config/runs/` parent, which is exactly the kind of
    near-miss a directory sweep gets wrong.
    """
    root, _config, _keep = project

    assert not any(
        "invocations" in p.relative_to(root).as_posix() for p in pcs.find_bundles(root)
    )


def test_the_generated_experiment_catalog_is_out_of_scope(project):
    """The one deletion that would destroy a kept artifact.

    `<exp_dir>/config/catalogs/` holds a catalog generated at run time over
    generated forcing. The recoverable-bin scan is scoped to `<project_dir>`
    exactly for this reason; a recursive glob would sweep it in.
    """
    root, _config, _keep = project
    # Pretend every file is byte-identical to something tracked, so the only
    # thing keeping the catalog alive is SCOPE rather than the hash test.
    everything = {p.name: {pcs._sha256(p)} for p in root.rglob("*") if p.is_file()}

    deletable, _reported = pcs.find_recoverable_copies(root, everything)

    # Relative to the project, never str(p): pytest derives tmp_path from the
    # TEST NAME, so an absolute-path substring check can match the temp
    # directory itself and pass for the wrong reason.
    assert not any(
        p.relative_to(root).as_posix().startswith("experiments/") for p in deletable
    )


def test_basin_data_is_never_deletable(project):
    """The predicate copies them because the toolbox cannot give them back."""
    root, _config, _keep = project
    everything = {p.name: {pcs._sha256(p)} for p in root.rglob("*") if p.is_file()}

    deletable, _reported = pcs.find_recoverable_copies(root, everything)

    # Named explicitly, not by substring: a `basin_data` bin that no longer
    # exists under that spelling would make a substring test pass vacuously,
    # which is exactly how the 2026-08-14 rename could have gone unnoticed.
    kept = root / "config/basin_data/output_locations.csv"
    assert kept.is_file()
    assert kept not in deletable
    assert not any("basin_data" in p.relative_to(root).as_posix() for p in deletable)


def test_a_file_the_repo_cannot_give_back_is_reported_never_deleted(project):
    """R4's whole point: a site-specific catalog must survive the migration."""
    root, _config, _keep = project

    deletable, reported = pcs.find_recoverable_copies(root, {})

    assert deletable == []
    assert any(p.name == "my_site_catalog.yml" for p in reported)


def test_git_being_unable_to_answer_deletes_nothing(project):
    """An exported tree or a container proves nothing recoverable.

    Failing closed matters here because the alternative -- treating an
    unanswerable query as "recoverable" -- deletes the copies precisely where
    they are the only record left.
    """
    root, _config, _keep = project

    deletable, _reported = pcs.find_recoverable_copies(root, {})

    assert deletable == []


def test_the_default_run_deletes_nothing(project, capsys):
    """The house contract of all three prune tools."""
    root, config, keep = project

    exit_code = pcs.main(["--config", str(config)])

    assert exit_code == 0
    assert "DRY RUN" in capsys.readouterr().out
    assert all(path.is_file() for path in keep)
    assert (root / "config/runs/model_creation/1a22a14838f3").is_dir()


def test_delete_removes_the_bundles_and_keeps_everything_protected(project):
    """The falsifier for "the cleanup deletes nothing it should keep"."""
    root, config, keep = project

    pcs.main(["--config", str(config), "--delete"])

    assert not (root / "config/runs/model_creation/1a22a14838f3").exists()
    assert not (
        root / "experiments/exp1/config/runs/climate_experiment/278159763309"
    ).exists()
    for path in keep:
        assert path.is_file(), f"{path} was deleted and must not have been"


def test_a_migrated_project_reports_nothing_to_do(tmp_path, capsys):
    """It is a one-shot migration, so a second run must find nothing."""
    root = tmp_path / "proj"
    (root / "config/runs/model_creation").mkdir(parents=True)
    (root / "config/runs/model_creation/run_record.yml").write_text(
        "schema_version: 2\n", encoding="utf-8"
    )
    config = tmp_path / "cfg.yml"
    config.write_text(
        yaml.safe_dump({"project": {"project_dir": str(root)}}), encoding="utf-8"
    )

    pcs.main(["--config", str(config)])

    assert "already in the new shape" in capsys.readouterr().out
    assert (root / "config/runs/model_creation/run_record.yml").is_file()
