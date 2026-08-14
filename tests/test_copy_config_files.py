"""R07 B9: the project config snapshot is routed by KIND, not to one dir.

This was a signature change rather than a rename -- `copy_config_files` derived
a single `output_dir` from the snake config's output path and wrote everything
beside it, so it could not serve four destinations (runs/, catalogs/,
templates/, generated/). These pin the new contract.
"""

import os
from pathlib import Path

import pytest
import yaml

from blueearth_cst.model import copy_config_files as ccf  # noqa: E402
from blueearth_cst.model.copy_config_files import copy_config_files  # noqa: E402


@pytest.fixture()
def sources(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    snake = src / "snake_config_model_creation.yml"
    snake.write_text("project:\n  project_dir: somewhere\n", encoding="utf-8")
    catalog = src / "deltares_data.yml"
    catalog.write_text("meta: {}\n", encoding="utf-8")
    template = src / "wflow_build_model.yml"
    template.write_text("steps: []\n", encoding="utf-8")
    return snake, catalog, template


def test_each_kind_lands_in_its_own_bin(tmp_path, sources):
    snake, catalog, template = sources
    cfg = tmp_path / "project" / "config"
    copy_config_files(
        config=str(snake),
        config_out_path=str(cfg / "runs" / "snake_config_model_creation.yml"),
        other_config_files={
            str(catalog): str(cfg / "catalogs"),
            str(template): str(cfg / "templates"),
        },
    )
    assert (cfg / "runs" / "snake_config_model_creation.yml").is_file()
    assert (cfg / "catalogs" / "deltares_data.yml").is_file()
    assert (cfg / "templates" / "wflow_build_model.yml").is_file()
    # nothing leaks into the parent bin
    assert not (cfg / "deltares_data.yml").exists()
    assert not (cfg / "snake_config_model_creation.yml").exists()


def test_content_is_copied_verbatim(tmp_path, sources):
    """A snapshot that mutates content would break the drift guard, which
    compares digests of these files across workflows."""
    snake, catalog, _ = sources
    cfg = tmp_path / "project" / "config"
    copy_config_files(
        config=str(snake),
        config_out_path=str(cfg / "runs" / "snake_config_model_creation.yml"),
        other_config_files={str(catalog): str(cfg / "catalogs")},
    )
    assert (cfg / "runs" / "snake_config_model_creation.yml").read_text(
        encoding="utf-8"
    ) == snake.read_text(encoding="utf-8")
    assert (cfg / "catalogs" / "deltares_data.yml").read_text(
        encoding="utf-8"
    ) == catalog.read_text(encoding="utf-8")


def test_missing_source_is_skipped_not_fatal(tmp_path, sources):
    """hydromt's predefined catalogs have no path on disk -- an absent entry
    must be skipped rather than crash the snapshot."""
    snake, catalog, _ = sources
    cfg = tmp_path / "project" / "config"
    copy_config_files(
        config=str(snake),
        config_out_path=str(cfg / "runs" / "snake_config_model_creation.yml"),
        other_config_files={
            str(catalog): str(cfg / "catalogs"),
            "artifact_data": str(cfg / "catalogs"),  # predefined, no file
        },
    )
    assert (cfg / "catalogs" / "deltares_data.yml").is_file()
    assert not (cfg / "catalogs" / "artifact_data").exists()


def test_destination_dirs_are_created(tmp_path, sources):
    """Snakemake creates parents for DECLARED outputs; the catalog/template
    bins are not declared outputs, so the script must create them itself."""
    snake, catalog, _ = sources
    cfg = tmp_path / "deep" / "nested" / "config"
    assert not cfg.exists()
    copy_config_files(
        config=str(snake),
        config_out_path=str(cfg / "runs" / "snake_config_model_creation.yml"),
        other_config_files={str(catalog): str(cfg / "catalogs")},
    )
    assert (cfg / "runs").is_dir() and (cfg / "catalogs").is_dir()


# --------------------------------------------------------------------------- #
# The observations bin (2026-08-01)
# --------------------------------------------------------------------------- #


def test_observations_land_in_their_own_bin(tmp_path, sources):
    """The two observation inputs are a fifth KIND, routed like the rest.

    They live outside the repo AND outside project_dir, referenced by absolute
    path (R07 O-01), so without this snapshot a finished project cannot say
    what it was evaluated against.
    """
    snake, _, _ = sources
    cfg = tmp_path / "project" / "config"
    locations = tmp_path / "src" / "output_locations.csv"
    locations.write_text("wflow_id,station_name,x,y\n", encoding="utf-8")
    series = tmp_path / "src" / "observations_timeseries.csv"
    series.write_text("time;101;102\n", encoding="utf-8")

    copy_config_files(
        config=snake,
        config_out_path=cfg / "runs" / "snake_config_model_creation.yml",
        other_config_files={
            str(locations): str(cfg / "observations"),
            str(series): str(cfg / "observations"),
        },
    )

    assert (cfg / "observations" / "output_locations.csv").is_file()
    assert (cfg / "observations" / "observations_timeseries.csv").is_file()
    # Routed, not duplicated into the other bins.
    assert not (cfg / "catalogs").exists()
    assert not (cfg / "templates").exists()


def test_the_snapshot_is_a_faithful_copy(tmp_path, sources):
    """A snapshot that silently truncated would be worse than none."""
    snake, _, _ = sources
    cfg = tmp_path / "project" / "config"
    series = tmp_path / "src" / "observations_timeseries.csv"
    body = "time;101;102\n2000-01-01T00:00:00;1.5;2.5\n2000-01-02T00:00:00;;3.0\n"
    series.write_text(body, encoding="utf-8")

    copy_config_files(
        config=snake,
        config_out_path=cfg / "runs" / "snake.yml",
        other_config_files={str(series): str(cfg / "observations")},
    )
    copied = (cfg / "observations" / "observations_timeseries.csv").read_text(
        encoding="utf-8"
    )
    assert copied == body


# --------------------------------------------------------------------------- #
# The copy predicate, the run record, and collision safety
# --------------------------------------------------------------------------- #

_ADVANCED = {
    "constraints": {"min_historical_years": 16},
    "defaults": {"julia_threads": 4},
    "runtime": {"julia_version": "1.11.7"},
}


def _record(tmp_path, sources, **overrides):
    """Run the writer with a run record and return the parsed record."""
    snake, catalog, _template = sources
    cfg = tmp_path / "project" / "config"
    record_path = cfg / "runs" / "model_creation" / "run_record.yml"
    kwargs = {
        "config": snake,
        "config_out_path": cfg / "runs" / "snake_config_model_creation.yml",
        "other_config_files": {str(catalog): str(cfg / "catalogs")},
        "run_record_path": record_path,
        "effective_config": {"project": {"project_dir": "somewhere"}},
        "advanced_settings": _ADVANCED,
        "workflow_name": "model_creation",
    }
    kwargs.update(overrides)
    copy_config_files(**kwargs)
    return yaml.safe_load(record_path.read_text(encoding="utf-8"))


def test_no_bundle_directory_is_created_by_any_path(tmp_path, sources):
    """The content-addressed bundle is gone; nothing may recreate it."""
    _record(tmp_path, sources)
    cfg = tmp_path / "project" / "config"

    assert not list(cfg.rglob("referenced-files.json"))
    assert not list(cfg.rglob("source.yml"))
    assert not list(cfg.rglob("effective.yml"))


def test_run_record_carries_the_design_schema(tmp_path, sources):
    """The record is the project's answer to 'what did the last run use'."""
    record = _record(tmp_path, sources)

    assert record["schema_version"] == 2
    assert record["workflow"] == "model_creation"
    assert record["advanced_settings"] == _ADVANCED
    assert record["effective_config"] == {"project": {"project_dir": "somewhere"}}
    assert len(record["effective_config_sha256"]) == 64
    assert len(record["configuration_inputs_sha256"]) == 64
    assert set(record["toolbox"]) == {"commit", "commit_source", "dirty"}
    assert set(record["environment"]) == {"pixi.lock", "Manifest.toml"}
    assert record["source_config"]["sha256"]


def test_run_record_leaves_no_temporary_behind(tmp_path, sources):
    """It is written temp-then-replace, so a reader never sees a partial file."""
    _record(tmp_path, sources)
    record_dir = tmp_path / "project" / "config" / "runs" / "model_creation"

    assert [p.name for p in record_dir.iterdir()] == ["run_record.yml"]


def test_a_file_outside_the_checkout_is_copied(tmp_path, sources):
    """R4's else-branch: the toolbox cannot give back what it never held."""
    _snake, catalog, _template = sources
    record = _record(tmp_path, sources)
    entry = next(e for e in record["referenced_inputs"] if e["origin"] == str(catalog))

    assert entry["recoverable"] is False
    assert entry["git_blob"] is None
    assert entry["archived_path"]
    assert Path(entry["archived_path"]).is_file()


def _fake_git(*, tracked=True, dirty=False, blob="b" * 40):
    """A stand-in for the three tracking queries, one branch at a time."""

    def query(command):
        if "ls-files" in command:
            return "" if tracked else None
        if "status" in command:
            return " M path\n" if dirty else ""
        if "rev-parse" in command:
            return f"{blob}\n"
        return None

    return query


def test_predicate_says_recoverable_only_when_tracked_and_clean(monkeypatch):
    """All three git branches, deterministically -- a real file cannot give them.

    A tracked file's cleanliness depends on what the working tree happens to
    look like when the suite runs, so driving the queries directly is the only
    way to reach every branch on every machine. The integration test below
    still proves the queries are wired to a real repository.
    """
    toolbox = {"commit": "a" * 40, "commit_source": "git", "dirty": False}
    inside = ccf._REPO_ROOT / "pyproject.toml"

    monkeypatch.setattr(ccf, "_git_query", _fake_git())
    assert ccf._tracked_blob(inside, toolbox) == "b" * 40

    monkeypatch.setattr(ccf, "_git_query", _fake_git(dirty=True))
    assert ccf._tracked_blob(inside, toolbox) is None, "a modified file must be copied"

    monkeypatch.setattr(ccf, "_git_query", _fake_git(tracked=False))
    assert ccf._tracked_blob(inside, toolbox) is None, "an untracked file is copied"

    monkeypatch.setattr(ccf, "_git_query", _fake_git())
    assert ccf._tracked_blob(inside, {"commit": None}) is None, "no commit, no claim"


def test_a_file_outside_the_checkout_is_never_recoverable(monkeypatch, tmp_path):
    """Even with every git query succeeding, an outside path cannot be tracked."""
    monkeypatch.setattr(ccf, "_git_query", _fake_git())
    outside = tmp_path / "site_specific_catalog.yml"
    outside.write_text("meta: {}\n", encoding="utf-8")

    assert ccf._tracked_blob(outside, {"commit": "a" * 40}) is None


def test_a_tracked_clean_toolbox_file_is_recorded_not_copied(tmp_path, sources):
    """The same predicate against the real repository, end to end.

    Picks a tracked file that git reports clean right now, so it exercises the
    actual queries. Skips when the tree offers none -- an exported tree has no
    tracking to query at all, and the predicate's contract there is to fall
    back to copying, which the degraded-mode test covers.
    """
    tracked = None
    for candidate in ("LICENSE", "pyproject.toml", "blueearth_cst/shared/gauges.py"):
        path = ccf._REPO_ROOT / candidate
        status = ccf._git_query(["git", "status", "--porcelain", "--", str(path)])
        if path.is_file() and status is not None and not status.strip():
            tracked = path
            break
    if tracked is None:
        pytest.skip("no clean tracked file available to exercise the predicate")

    cfg = tmp_path / "project" / "config"
    record = _record(
        tmp_path,
        sources,
        other_config_files={str(tracked): str(cfg / "templates")},
    )
    entry = record["referenced_inputs"][0]

    assert entry["recoverable"] is True
    assert entry["git_blob"]
    assert entry["archived_path"] is None
    # The repo-relative posix path exactly: this is the field a reader uses to
    # find the file back in the toolbox, so "some string with a slash" is not
    # the assertion worth making.
    assert entry["origin"] == tracked.relative_to(ccf._REPO_ROOT).as_posix()
    assert not (cfg / "templates").exists()


def _inside_the_checkout(monkeypatch, tmp_path):
    """Make ``tmp_path`` the toolbox root, with every git query answering yes.

    The tracked-and-clean branch cannot otherwise be reached hermetically: a
    file must resolve INSIDE ``_REPO_ROOT`` for the predicate to consult git at
    all, so a real tracked file would be the only alternative — and then the
    test's answer would depend on whether that file happens to be clean, and on
    whether `test_case/` was seeded into this worktree. Both faked together, so
    the assertion is about the POLICY and nothing else.
    """
    monkeypatch.setattr(ccf, "_REPO_ROOT", tmp_path.resolve())
    monkeypatch.setattr(ccf, "_git_query", _fake_git())


def test_observation_inputs_are_archived_even_when_recoverable(
    tmp_path, sources, monkeypatch
):
    """The exception the predicate does not get a say in.

    An observation input is the project's record of what the model was
    evaluated against, so it is archived whether or not the toolbox could hand
    it back. Before this, a project whose `gauge_points` pointed at a tracked
    CSV inside the checkout — the test fixture case — got no
    `config/observations/` bin at all, and nothing said so.
    """
    _inside_the_checkout(monkeypatch, tmp_path)
    cfg = tmp_path / "project" / "config"
    locations = tmp_path / "basin" / "gauges.csv"
    locations.parent.mkdir(parents=True, exist_ok=True)
    locations.write_text("wflow_id,station_name,x,y\n", encoding="utf-8")

    record = _record(
        tmp_path,
        sources,
        other_config_files={str(locations): str(cfg / "observations")},
        # The ROLE decides, not the filename -- which is why the source here is
        # called something else entirely.
        reference_roles={str(locations): "output_locations"},
    )
    entry = record["referenced_inputs"][0]

    assert (cfg / "observations" / "output_locations.csv").is_file()
    assert entry["archived_path"].endswith("config/observations/output_locations.csv")
    # `recoverable` and `archived_path` are now INDEPENDENT: this file is both.
    assert entry["recoverable"] is True
    assert entry["git_blob"] == "b" * 40


def test_a_non_observation_tracked_file_is_still_not_copied(
    tmp_path, sources, monkeypatch
):
    """The guard against the change being over-broad.

    Same tracked-and-clean setup, a role that is not an observation input: the
    record-by-blob-id branch still applies and the bin is still never created.
    """
    _inside_the_checkout(monkeypatch, tmp_path)
    cfg = tmp_path / "project" / "config"
    catalog = tmp_path / "site" / "site_specific_catalog.yml"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text("meta: {}\n", encoding="utf-8")

    record = _record(
        tmp_path,
        sources,
        other_config_files={str(catalog): str(cfg / "catalogs")},
    )
    entry = record["referenced_inputs"][0]

    assert entry["recoverable"] is True
    assert entry["archived_path"] is None
    assert not (cfg / "catalogs").exists()
    assert not (cfg / "templates").exists()


def test_a_case_only_destination_collision_still_raises(tmp_path, sources):
    """On Windows two destinations differing only in case are ONE file.

    Regression guard for an answer that used to depend on history: keyed on
    ``Path.resolve()``, a case-only pair collided on a re-run (the first file
    now existed, so resolve reported its real casing) but not on a fresh one,
    where the second silently overwrote the first.
    """
    snake, _catalog, _template = sources
    cfg = tmp_path / "project" / "config"
    first_dir = tmp_path / "a"
    second_dir = tmp_path / "b"
    first_dir.mkdir()
    second_dir.mkdir()
    (first_dir / "one.csv").write_text("first\n", encoding="utf-8")
    (second_dir / "two.csv").write_text("second\n", encoding="utf-8")

    def run():
        copy_config_files(
            config=snake,
            config_out_path=cfg / "runs" / "snake.yml",
            other_config_files={
                str(first_dir / "one.csv"): str(cfg / "observations"),
                str(second_dir / "two.csv"): str(cfg / "observations"),
            },
            reference_roles={
                str(first_dir / "one.csv"): "observations",
                str(second_dir / "two.csv"): "OBSERVATIONS",
            },
        )

    if os.path.normcase("A") != os.path.normcase("a"):
        pytest.skip("case-sensitive filesystem: the two destinations are distinct")

    with pytest.raises(ValueError, match="both map to"):
        run()
    # And again against a tree where the first destination already exists, the
    # case that used to be the only one that raised.
    with pytest.raises(ValueError, match="both map to"):
        run()


def test_degraded_mode_copies_everything(tmp_path, sources, monkeypatch):
    """A deployed image has no .git, so every referenced file must be copied.

    This is the predicate's else-branch reached by construction rather than by
    a special case: with no commit, step 2 can never be satisfied.
    """
    monkeypatch.setattr(
        ccf,
        "toolbox_identity",
        lambda *_, **__: {
            "commit": None,
            "commit_source": None,
            "dirty": None,
        },
    )
    tracked = Path(__file__).resolve()
    cfg = tmp_path / "project" / "config"
    record = _record(
        tmp_path,
        sources,
        other_config_files={str(tracked): str(cfg / "templates")},
    )

    assert record["toolbox"] == {
        "commit": None,
        "commit_source": None,
        "dirty": None,
    }
    entry = record["referenced_inputs"][0]
    assert entry["recoverable"] is False
    assert Path(entry["archived_path"]).is_file()


def test_two_references_sharing_a_destination_raise(tmp_path, sources):
    """The falsifier for 'no input is silently lost'.

    Two configured observation paths can share a basename -- they are arbitrary
    absolute paths. The old writer copied both to `dest / source.name` and the
    second overwrote the first, leaving a project that claimed to hold an input
    it had lost.
    """
    snake, _catalog, _template = sources
    cfg = tmp_path / "project" / "config"
    first_dir = tmp_path / "a"
    second_dir = tmp_path / "b"
    first_dir.mkdir()
    second_dir.mkdir()
    (first_dir / "data.csv").write_text("first\n", encoding="utf-8")
    (second_dir / "data.csv").write_text("second\n", encoding="utf-8")

    with pytest.raises(ValueError, match="both map to"):
        copy_config_files(
            config=snake,
            config_out_path=cfg / "runs" / "snake.yml",
            other_config_files={
                str(first_dir / "data.csv"): str(cfg / "observations"),
                str(second_dir / "data.csv"): str(cfg / "observations"),
            },
        )


def test_declared_roles_keep_same_named_files_apart(tmp_path, sources):
    """Roles are how two same-basename inputs both survive the snapshot."""
    snake, _catalog, _template = sources
    cfg = tmp_path / "project" / "config"
    first_dir = tmp_path / "a"
    second_dir = tmp_path / "b"
    first_dir.mkdir()
    second_dir.mkdir()
    (first_dir / "data.csv").write_text("locations\n", encoding="utf-8")
    (second_dir / "data.csv").write_text("series\n", encoding="utf-8")

    copy_config_files(
        config=snake,
        config_out_path=cfg / "runs" / "snake.yml",
        other_config_files={
            str(first_dir / "data.csv"): str(cfg / "observations"),
            str(second_dir / "data.csv"): str(cfg / "observations"),
        },
        reference_roles={
            str(first_dir / "data.csv"): "output_locations",
            str(second_dir / "data.csv"): "observations_timeseries",
        },
    )

    observations = cfg / "observations"
    assert (observations / "output_locations.csv").read_text(
        encoding="utf-8"
    ) == "locations\n"
    assert (observations / "observations_timeseries.csv").read_text(
        encoding="utf-8"
    ) == "series\n"


def test_a_logical_identifier_is_recorded_without_a_copy(tmp_path, sources):
    """hydromt's predefined catalogs are named, not pathed: nothing to copy."""
    cfg = tmp_path / "project" / "config"
    record = _record(
        tmp_path, sources, other_config_files={"artifact_data": str(cfg / "catalogs")}
    )
    entry = record["referenced_inputs"][0]

    assert entry["origin"] == "artifact_data"
    assert entry["archived_path"] is None
    assert entry["sha256"] is None


def test_run_record_requires_its_companions(tmp_path, sources):
    """A record missing its config would describe nothing."""
    snake, _catalog, _template = sources
    cfg = tmp_path / "project" / "config"

    with pytest.raises(ValueError, match="must be provided together"):
        copy_config_files(
            config=snake,
            config_out_path=cfg / "runs" / "snake.yml",
            run_record_path=cfg / "runs" / "run_record.yml",
        )


def test_the_runs_bin_carries_its_own_readme(tmp_path, sources):
    """The one genuine trap in this bin: it all looks like configuration.

    Everything under config/runs/ is written by the run, so an edit is
    silently overwritten on the next execution. Someone will eventually open
    the copy nearest the outputs and change it, which is why the warning ships
    beside the files rather than only in the repo README.
    """
    snake, _catalog, _template = sources
    cfg = tmp_path / "project" / "config"

    copy_config_files(
        config=snake,
        config_out_path=cfg / "runs" / "snake_config_model_creation.yml",
    )

    readme = (cfg / "runs" / "README.md").read_text(encoding="utf-8")

    assert "written by the run" in readme
    assert "journal.jsonl" in readme
    # The two claims a reader would otherwise get wrong.
    assert "lower bound on invocations" in readme
    assert "scientific data identity" in readme
    # The bin's SECOND trap, added 2026-08-13 after the owner hit it: the
    # per-workflow filename reads as a per-workflow scope, and the file is the
    # whole config. Asserted here because the mitigation for that confusion is
    # this text and nothing else -- the filename itself cannot be fixed (the
    # WF1 copy is a mandatory declared input of WF3's guard rule).
    assert "not that workflow's section of it" in readme
    assert "projection" in readme


def test_the_runs_readme_is_refreshed_not_preserved(tmp_path, sources):
    """It is shipped documentation, so an old project should get the new text."""
    snake, _catalog, _template = sources
    cfg = tmp_path / "project" / "config"
    stale = cfg / "runs" / "README.md"
    stale.parent.mkdir(parents=True)
    stale.write_text("notes I typed here\n", encoding="utf-8")

    copy_config_files(
        config=snake,
        config_out_path=cfg / "runs" / "snake_config_model_creation.yml",
    )

    assert "notes I typed here" not in stale.read_text(encoding="utf-8")
