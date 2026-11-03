"""Tests for deterministic shared CST provenance digests."""

import copy
from collections import OrderedDict
from pathlib import Path

import pytest

from blueearth_cst.shared import provenance
from blueearth_cst.shared.provenance import (
    SHORT_DIGEST_CHARS,
    TOOLBOX_COMMIT_FILE,
    append_journal_line,
    canonical_data,
    canonical_sha256,
    configuration_inputs_digest,
    effective_config_digest,
    effective_config_document,
    environment_file_hashes,
    file_sha256,
    lock_file_sha256,
    project_config,
    read_journal_lines,
    short_digest,
    snapshot_bundle_digest,
    toolbox_identity,
)

_CONFIG = {
    "project": {"project_dir": "out"},
    "shared": {"basin": "test", "julia_threads": 4},
    "workflows": {
        "build_model": {"resolution": 0.008},
        "run_stress_test": {"rlz_num": 10},
    },
}
_WF1_PROJECTION = ("project", "shared", "workflows.build_model")


def test_short_digest_is_a_prefix_of_the_full_digest() -> None:
    """The naming form must stay findable from the record it stands for."""
    digest = canonical_sha256({"a": 1})

    assert len(short_digest(digest)) == SHORT_DIGEST_CHARS
    assert digest.startswith(short_digest(digest))


def test_short_digest_rejects_a_value_that_is_not_a_digest() -> None:
    """Truncating a non-digest would name an artifact after nothing."""
    with pytest.raises(ValueError):
        short_digest("abc")
    with pytest.raises(TypeError):
        short_digest(None)


def test_canonical_digest_is_mapping_order_independent() -> None:
    """Equivalent mappings hash identically regardless of insertion order."""
    first = {"nested": {"b": 2, "a": 1}, "items": [True, None, 1.5]}
    second = OrderedDict([("items", [True, None, 1.5]), ("nested", {"a": 1, "b": 2})])

    assert canonical_sha256(first) == canonical_sha256(second)


def test_canonical_data_preserves_supported_types() -> None:
    """Type tags prevent distinct supported values from collapsing."""
    values = [None, False, 0, 0.0, "0", Path("data/input.nc"), [], ()]

    documents = [canonical_data(value) for value in values]

    assert len({canonical_sha256(value) for value in values}) == len(values)
    assert documents[-2]["type"] == "list"
    assert documents[-1]["type"] == "tuple"
    assert documents[5] == {"type": "path", "value": "data/input.nc"}


def test_canonical_data_rejects_unsupported_values() -> None:
    """Unsupported objects fail instead of acquiring unstable repr strings."""
    with pytest.raises(TypeError, match="unsupported provenance value"):
        canonical_data({"bad": object()})


def test_file_sha256_hashes_exact_bytes(tmp_path: Path) -> None:
    """File digests are byte-based rather than text-normalized."""
    source = tmp_path / "source.yml"
    source.write_bytes(b"key: value\r\n")

    assert (
        file_sha256(source)
        == "db5735d4d5b4974b003686308b1e5e4564d1e42e27187b13cabfd53b638cdb8f"
    )


def test_effective_config_digest_covers_advanced_settings() -> None:
    """Toolbox-wide settings participate in the scientific config identity."""
    config = {"project": {"project_dir": "out"}}
    first = {"defaults": {"julia_threads": 2}}
    second = {"defaults": {"julia_threads": 4}}

    assert effective_config_digest(config, first, None) != effective_config_digest(
        config, second, None
    )


def test_snapshot_bundle_digest_hashes_files_but_not_physical_paths(
    tmp_path: Path,
) -> None:
    """Local references use content identity and logical references use IDs."""
    source = tmp_path / "config.yml"
    source.write_text("project: test\n", encoding="utf-8")
    first_file = tmp_path / "first.yml"
    second_file = tmp_path / "second.yml"
    first_file.write_text("same: bytes\n", encoding="utf-8")
    second_file.write_text("same: bytes\n", encoding="utf-8")
    common = [{"kind": "catalog", "identifier": "era5"}]

    first = snapshot_bundle_digest(
        {},
        {},
        source,
        [*common, {"kind": "template", "identifier": "forcing", "path": first_file}],
    )
    second = snapshot_bundle_digest(
        {},
        {},
        source,
        [*common, {"kind": "template", "identifier": "forcing", "path": second_file}],
    )

    assert first == second


def test_snapshot_bundle_digest_rejects_ambiguous_reference() -> None:
    """Reference descriptors require explicit kind and logical identifier."""
    with pytest.raises(ValueError, match="kind.*identifier"):
        snapshot_bundle_digest({}, {}, __file__, [{"path": __file__}])


def test_project_config_selects_only_the_declared_paths() -> None:
    """The projection is what a workflow reads, not what the file happens to hold."""
    projected = project_config(_CONFIG, _WF1_PROJECTION)

    assert projected == {
        "project": {"project_dir": "out"},
        "shared": {"basin": "test", "julia_threads": 4},
        "workflows": {"build_model": {"resolution": 0.008}},
    }
    assert "run_stress_test" not in projected["workflows"]


def test_project_config_raises_on_a_path_the_config_lacks() -> None:
    """A declaration that selects nothing would silently stop recording.

    Omitting the missing path instead would give a digest that never moves when
    that section changes -- the failure mode is an absence, so it has to be
    loud at the point of declaration.
    """
    with pytest.raises(KeyError, match="analyze_projections"):
        project_config(_CONFIG, ("project", "workflows.analyze_projections"))


def test_project_config_rejects_overlapping_paths() -> None:
    """Declaring a section and its child leaves the covered scope ambiguous."""
    with pytest.raises(ValueError, match="overlap"):
        project_config(_CONFIG, ("shared", "shared.basin"))


def test_a_projection_can_exclude_a_child_of_a_selected_path() -> None:
    """`C-79`: the section is selected, one child is pruned from it.

    The alternative -- enumerating the section's OTHER children -- is unsafe
    here, because `project_config` raises on a declared path the config lacks
    and the remaining children are optional. A config that omitted one would
    stop parsing.
    """
    projected = project_config(_CONFIG, ("shared", "-shared.julia_threads"))

    assert projected == {"shared": {"basin": "test"}}


def test_excluding_a_path_the_config_lacks_is_a_no_op() -> None:
    """Deliberately NOT the rule selection follows, and the asymmetry is the point.

    Selection raises on an absent path because a projection DECLARES what a
    workflow reads, so a missing one is a typo or a config missing a section it
    needs. An exclusion declares what does not count as identity, and the keys
    worth excluding are the optional ones -- a config that never set one has
    nothing to disagree about and must still produce a digest.
    """
    projected = project_config(_CONFIG, ("project", "-project.compute"))

    assert projected == {"project": {"project_dir": "out"}}


def test_an_exclusion_outside_every_selected_path_is_refused() -> None:
    """It would prune nothing, which makes it a typo rather than a declaration.

    Silently accepting it is the failure worth refusing: the record would name
    an exclusion that never happened, so a reader would believe a key was
    outside configuration identity when it was inside it all along.
    """
    with pytest.raises(ValueError, match="would prune nothing"):
        project_config(_CONFIG, ("project", "-shared.basin"))


def test_pruning_does_not_reach_back_into_the_callers_config() -> None:
    """The projection is a view; building one must not edit the config.

    `config` is Snakemake's live mapping in the only caller that excludes
    anything, and every rule downstream reads it. A prune that mutated it would
    silently remove `compute:` from the batching code that resolves the run.
    """
    before = copy.deepcopy(_CONFIG)
    project_config(_CONFIG, ("shared", "-shared.julia_threads"))

    assert _CONFIG == before


def test_the_recorded_document_names_the_exclusion() -> None:
    """The record has to say what it left out, or it under-describes itself.

    This is the whole reason the exclusion travels inside the projection rather
    than being pruned from the config by the caller: a `projection` field that
    claimed a section while the document omitted one of its children would be a
    provenance record that quietly stops recording -- the failure the
    projection mechanism exists to prevent.
    """
    document = effective_config_document(
        _CONFIG, {}, ("shared", "-shared.julia_threads")
    )

    assert "-shared.julia_threads" in document["projection"]
    assert "julia_threads" not in document["project_config"]["shared"]


def test_an_excluded_key_does_not_move_the_digest() -> None:
    """`C-79` stated as the behaviour a user notices.

    Raising `batch_size` on a bigger machine must not change what the run is,
    and the digest is where "what the run is" is decided.
    """
    other = copy.deepcopy(_CONFIG)
    other["shared"]["julia_threads"] = 64

    assert effective_config_digest(
        _CONFIG, {}, ("shared", "-shared.julia_threads")
    ) == effective_config_digest(other, {}, ("shared", "-shared.julia_threads"))


def test_effective_config_digest_ignores_sections_outside_the_projection() -> None:
    """A WF3-only edit must not move WF1's configuration identity.

    This is the whole point of scoping by consumed keys: before it, an edit
    anywhere in the file re-fired every workflow's record.
    """
    other = {
        **_CONFIG,
        "workflows": {**_CONFIG["workflows"], "run_stress_test": {"rlz_num": 99}},
    }

    assert effective_config_digest(_CONFIG, {}, _WF1_PROJECTION) == (
        effective_config_digest(other, {}, _WF1_PROJECTION)
    )


def test_effective_config_digest_follows_keys_inside_the_projection() -> None:
    """The complement of the test above: in-scope edits must move the digest."""
    other = {
        **_CONFIG,
        "workflows": {**_CONFIG["workflows"], "build_model": {"resolution": 0.01}},
    }

    assert effective_config_digest(_CONFIG, {}, _WF1_PROJECTION) != (
        effective_config_digest(other, {}, _WF1_PROJECTION)
    )


def test_configuration_inputs_digest_splits_from_configuration_identity() -> None:
    """A referenced input moving is a run-input change, not a config change.

    The two digests answer different questions, so a change to one term must
    not be readable as a change to the other.
    """
    config_sha = effective_config_digest(_CONFIG, {}, _WF1_PROJECTION)
    toolbox = {"commit": "a" * 40, "commit_source": "git", "dirty": False}
    environment = {"pixi.lock": "lock-sha", "Manifest.toml": "manifest-sha"}

    first = configuration_inputs_digest(
        config_sha, toolbox, environment, [{"role": "obs", "sha256": "one"}]
    )
    second = configuration_inputs_digest(
        config_sha, toolbox, environment, [{"role": "obs", "sha256": "two"}]
    )

    assert first != second
    assert config_sha == effective_config_digest(_CONFIG, {}, _WF1_PROJECTION)


def test_configuration_inputs_digest_follows_toolbox_and_environment() -> None:
    """Params threading only refreshes the record if these terms move it."""
    config_sha = effective_config_digest(_CONFIG, {}, _WF1_PROJECTION)
    environment = {"pixi.lock": "lock-sha", "Manifest.toml": "manifest-sha"}
    base = {"commit": "a" * 40, "commit_source": "git", "dirty": False}

    baseline = configuration_inputs_digest(config_sha, base, environment)
    moved_commit = configuration_inputs_digest(
        config_sha, {**base, "commit": "b" * 40}, environment
    )
    flipped_dirty = configuration_inputs_digest(
        config_sha, {**base, "dirty": True}, environment
    )
    moved_lock = configuration_inputs_digest(
        config_sha, base, {**environment, "pixi.lock": "other"}
    )

    assert len({baseline, moved_commit, flipped_dirty, moved_lock}) == 4


def test_configuration_inputs_digest_ignores_referenced_input_order() -> None:
    """Descriptor order is an artifact of iteration, not an input difference."""
    config_sha = "config-sha"
    inputs = [{"role": "a", "sha256": "one"}, {"role": "b", "sha256": "two"}]

    assert configuration_inputs_digest(config_sha, {}, {}, inputs) == (
        configuration_inputs_digest(config_sha, {}, {}, list(reversed(inputs)))
    )


def test_toolbox_identity_prefers_git_over_a_baked_commit(tmp_path: Path) -> None:
    """A checkout must never read a stale baked sha in place of its own HEAD.

    This is the P5 falsifier: a .toolbox-commit sitting in a working checkout
    -- left by a build, or unpacked from an image -- must lose to git.
    """
    (tmp_path / TOOLBOX_COMMIT_FILE).write_text("f" * 40, encoding="utf-8")

    identity = toolbox_identity(repo_root=tmp_path)

    if identity["commit_source"] == "git":
        assert identity["commit"] != "f" * 40
    else:  # tmp_path sits outside any repository on this machine
        assert identity["commit_source"] == "baked"


def test_toolbox_identity_falls_back_to_the_baked_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deployed image has no .git, so the baked file is the only identity."""
    monkeypatch.setattr(provenance, "_run_metadata_command", lambda *_: None)
    (tmp_path / TOOLBOX_COMMIT_FILE).write_text("e" * 40 + "\n", encoding="utf-8")

    assert toolbox_identity(repo_root=tmp_path) == {
        "commit": "e" * 40,
        "commit_source": "baked",
        "dirty": None,
    }


def test_toolbox_identity_reports_nulls_when_nothing_can_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unwitnessed revision is recorded as unknown, never guessed."""
    monkeypatch.setattr(provenance, "_run_metadata_command", lambda *_: None)

    assert toolbox_identity(repo_root=tmp_path) == {
        "commit": None,
        "commit_source": None,
        "dirty": None,
    }


def test_toolbox_identity_ignores_an_empty_baked_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unset build ARG must yield no identity, not an empty-string commit."""
    monkeypatch.setattr(provenance, "_run_metadata_command", lambda *_: None)
    (tmp_path / TOOLBOX_COMMIT_FILE).write_text("\n  \n", encoding="utf-8")

    assert toolbox_identity(repo_root=tmp_path)["commit"] is None


def test_release_tag_excludes_milestones(monkeypatch: pytest.MonkeyPatch) -> None:
    def metadata(command, root):
        if command[1] == "tag":
            return "v0.4.1\nv0.2.0-alpha\nmilestone-12\nv0.4.2-checkpoint\n"
        assert command == [
            "git",
            "describe",
            "--tags",
            "--abbrev=0",
            "--match",
            "v0.4.1",
            "--match",
            "v0.2.0-alpha",
            "abc123",
        ]
        return "v0.4.1\n"

    monkeypatch.setattr(provenance, "_run_metadata_command", metadata)
    assert provenance.toolbox_release_tag("abc123") == "v0.4.1"


@pytest.mark.parametrize("tags", [None, "", "milestone-12\nv0.4.2-checkpoint\n"])
def test_release_tag_unavailable(tags, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provenance, "_run_metadata_command", lambda *_: tags)
    assert provenance.toolbox_release_tag("abc123") is None
    assert provenance.toolbox_release_tag(None) is None


def test_environment_file_hashes_records_absence_explicitly(tmp_path: Path) -> None:
    """A null says the file was not there; a missing key says nobody looked."""
    (tmp_path / "pixi.lock").write_text("version: 6\n", encoding="utf-8")

    hashes = environment_file_hashes(repo_root=tmp_path)

    assert hashes["pixi.lock"] == lock_file_sha256(tmp_path / "pixi.lock")
    assert hashes["Manifest.toml"] is None


def test_journal_accumulates_across_appends(tmp_path: Path) -> None:
    """The journal's value is its history, so a second run must not replace it."""
    journal = tmp_path / "runs" / "journal.jsonl"

    append_journal_line(journal, {"invocation_id": "one", "event": "started"})
    append_journal_line(journal, {"invocation_id": "one", "event": "success"})
    append_journal_line(journal, {"invocation_id": "two", "event": "started"})

    records = read_journal_lines(journal)

    assert [record["event"] for record in records] == ["started", "success", "started"]
    assert journal.read_text(encoding="utf-8").count("\n") == 3


def test_journal_reader_tolerates_a_torn_final_line(tmp_path: Path) -> None:
    """A hard-killed run can leave a partial line; it must not poison the rest."""
    journal = tmp_path / "journal.jsonl"
    append_journal_line(journal, {"invocation_id": "one", "event": "success"})
    with journal.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write('{"invocation_id": "two", "eve')

    records = read_journal_lines(journal)

    assert len(records) == 1
    assert records[0]["invocation_id"] == "one"


def test_journal_reader_reads_a_missing_file_as_empty(tmp_path: Path) -> None:
    """A young project has no journal yet; that is not an error."""
    assert read_journal_lines(tmp_path / "absent.jsonl") == []


def test_environment_file_hashes_ignore_the_checkout_line_ending(
    tmp_path: Path,
) -> None:
    lf, crlf = tmp_path / "lf", tmp_path / "crlf"
    for root, eol in ((lf, b"\n"), (crlf, b"\r\n")):
        root.mkdir()
        (root / "pixi.lock").write_bytes(b"version: 6" + eol + b"- a" + eol)
        (root / "Manifest.toml").write_bytes(b"[deps]" + eol)
    assert environment_file_hashes(repo_root=lf) == environment_file_hashes(
        repo_root=crlf
    )
