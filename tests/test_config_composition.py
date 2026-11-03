"""The T1+T2 loader composes to today's shape, and refuses what breaks the seam.

Unit surface for ``blueearth_cst/shared/config_composition.py`` (R13 §16.1).
Every case here is one row of the design's gate table, and the module is
deliberately self-contained: it builds its configs in ``tmp_path`` rather than
reading a shipped seed, so it keeps meaning while the seeds migrate in a later
commit and cannot be made green by a config edit.

Three groups carry most of the weight:

* **the composition invariant** — what comes out is what the pre-split config
  held, key for key. `enabled` merged, `config_path` not, an omitted
  `config_path` adding nothing at all;
* **the seam checks** — the four parse-time refusals, including the pair that is
  easy to invert: a broken workflow file *outside* this entry point's scope is
  skipped, and the identical file *inside* it is a hard error. Getting that
  backwards lets a broken WF3 config break a WF1 run, which is the inverse of
  what the milestone is for;
* **the static read scan** (D-9.6), which is checked in both directions and
  against a planted violation, so a green result means the scan looked rather
  than that it could not see.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from blueearth_cst.shared import config_composition as cc

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_retired_workflow_names_both_replacements(tmp_path):
    path = write_split(tmp_path, bodies={"run_stress_test": {}})
    with pytest.raises(ValueError, match="generate_scenarios.*simulate_system"):
        cc.load_composed_config(path)


@pytest.mark.parametrize("workflow", ["generate_scenario", "simulate", "unknown"])
def test_unknown_workflow_stanza_is_refused(tmp_path, workflow):
    path = write_split(tmp_path, bodies={workflow: {}})
    with pytest.raises(ValueError, match="unknown workflow stanza"):
        cc.load_composed_config(path)


# `sorted` is load-bearing, not tidiness. Both key collections are frozensets,
# whose iteration order depends on PYTHONHASHSEED and therefore differs between
# processes. Under `test-fast` (`-n auto`) each xdist worker collects these cases
# in its own order, and pytest aborts the whole run with "Different tests were
# collected between gw7 and gw8". Single-process runs -- which is what CI does --
# never expose it, so the documented pre-push gate was the only thing it broke.
@pytest.mark.parametrize(
    "owner,key,other",
    [
        ("simulate_system", key, "generate_scenarios")
        for key in sorted(cc.GENERATION_KEYS)
    ]
    + [
        ("generate_scenarios", key, "simulate_system")
        for key in sorted(cc.SIMULATION_KEYS)
    ],
)
def test_split_workflow_settings_cannot_cross_ownership(tmp_path, owner, key, other):
    path = write_split(tmp_path, bodies={owner: {key: {}}})
    with pytest.raises(ValueError, match=f"belong in {other}"):
        cc.load_composed_config(path)


#: The real ``CONFIG_PROJECTION`` of each entry point, and the ``R(entry)`` it
#: must produce (§8.3). Restated here rather than imported because a Snakefile
#: is not importable — which is exactly why the design passes the projection in
#: as an argument instead of restating it inside the loader.
PROJECTIONS = {
    "simulate_system": (
        ("project", "basin", "climate", "model", "workflows.simulate_system"),
        {"simulate_system"},
    ),
    "analyze_climate": (
        ("project", "basin", "climate", "model", "workflows.analyze_climate"),
        {"analyze_climate"},
    ),
    "build_model": (
        ("project", "basin", "climate", "model", "workflows.build_model"),
        {"build_model"},
    ),
    "analyze_projections": (
        ("project", "basin", "climate", "model", "workflows.analyze_projections"),
        {"analyze_projections"},
    ),
    "generate_scenarios": (
        (
            "project",
            "basin",
            "climate",
            "model",
            "workflows.analyze_projections",
            "workflows.build_model",
            "workflows.generate_scenarios",
        ),
        {"generate_scenarios", "build_model", "analyze_projections"},
    ),
}


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

#: The smallest T1 shared-section set that PASSES every R14 parse-time refusal.
#: `climate.selected` is set and is a member of `climate.sources` because
#: D-10.4 rows 3 and 4 refuse otherwise the moment WF1 or WF3 is enabled — and
#: nearly every case in this module enables one. Tests about the refusals
#: themselves pass `sections=` explicitly and take what they need away.
MINIMAL_T1_SECTIONS: dict = {
    "basin": {"region": "x"},
    "climate": {"sources": ["era5"], "selected": "era5"},
}


def test_plotting_is_project_owned_and_preserved(tmp_path):
    sections = {**MINIMAL_T1_SECTIONS, "plotting": {"annotation": "title"}}
    path = write_split(tmp_path, sections=sections, bodies={"build_model": {}})
    config = cc.load_composed_config(path)
    assert config["plotting"] == {"annotation": "title"}


@pytest.mark.parametrize("key", ["plotting", "annotation"])
def test_plotting_cannot_be_declared_in_workflow_file(tmp_path, key):
    path = write_split(tmp_path, bodies={"build_model": {key: "title"}})
    with pytest.raises(ValueError, match=key):
        cc.load_composed_config(path)


def test_invalid_annotation_is_refused_during_composition(tmp_path):
    path = write_split(tmp_path, sections={"plotting": {"annotation": "clean"}})
    with pytest.raises(ValueError, match="plotting.annotation"):
        cc.load_composed_config(path)


@pytest.mark.parametrize(
    "owner,body",
    [
        ("analyze_climate", {"diagnostics": {"captioned_figures": True}}),
        ("analyze_projections", {"captioned_figures": False}),
    ],
)
def test_old_caption_switch_names_project_replacement(tmp_path, owner, body):
    path = write_split(tmp_path, bodies={owner: body})
    with pytest.raises(ValueError, match="plotting.annotation: caption"):
        cc.load_composed_config(path)


def write_split(
    directory: Path,
    stem: str = "project_config_test",
    sections: dict | None = None,
    bodies: dict[str, dict] | None = None,
    enabled: dict[str, bool] | None = None,
    omit_path: tuple[str, ...] = (),
    schema_version: object = 2,
) -> Path:
    """Write a T1 file plus one T2 file per workflow body; return the T1 path.

    ``sections`` is the T1 SHARED sections — ``basin:``, ``climate:``,
    ``model:`` — as one mapping, written at the project file's top level. It
    replaced a ``shared=`` parameter when R14 dissolved that section (D-7.2);
    the callers below moved key for key, so a test that read ``shared={"basin":
    ...}`` now reads ``sections={"basin": ...}`` and asserts the same thing.

    ``omit_path`` names workflows whose stanza gets no ``config_path`` even
    though a body was given — the shape D-8.7 accepts and the migration never
    produces. ``schema_version`` is a parameter, and ``None`` omits the key
    entirely, so the v1 refusal can be exercised on a document this helper built
    rather than on a hand-pasted string.
    """
    directory.mkdir(parents=True, exist_ok=True)
    bodies = {} if bodies is None else bodies
    enabled = {} if enabled is None else enabled
    t1: dict = {}
    if schema_version is not None:
        t1["schema_version"] = schema_version
    t1["project"] = {"project_dir": (directory / "proj").as_posix()}
    t1.update(MINIMAL_T1_SECTIONS if sections is None else sections)
    t1["workflows"] = {}
    for name, body in bodies.items():
        stanza: dict[str, object] = {"enabled": enabled.get(name, True)}
        if name not in omit_path:
            t2_name = f"{stem}_{name}.yml"
            (directory / t2_name).write_text(
                yaml.safe_dump(body, sort_keys=False), encoding="utf-8"
            )
            stanza["config_path"] = t2_name
        t1["workflows"][name] = stanza
    t1_path = directory / f"{stem}.yml"
    t1_path.write_text(yaml.safe_dump(t1, sort_keys=False), encoding="utf-8")
    return t1_path


def compose(t1_path: Path, entry: str | None = "build_model") -> dict:
    """Compose ``t1_path`` for one entry point, using that entry's projection."""
    projection = PROJECTIONS[entry][0] if entry is not None else None
    return cc.load_composed_config(t1_path, entry, projection)


def test_captured_composition_does_not_reread_mutated_workflow_file(tmp_path):
    project = write_split(
        tmp_path / "cfg",
        bodies={"build_model": {"engine": {"build_config": "old.yml"}}},
    )
    workflow = project.with_name("project_config_test_build_model.yml")
    captured = cc.capture_configuration_sources(project, "build_model")
    workflow.write_text("engine:\n  build_config: new.yml\n", encoding="utf-8")
    composed, paths = cc.compose_captured_config(captured[0], captured, "build_model")
    assert composed["workflows"]["build_model"]["engine"]["build_config"] == "old.yml"
    assert "build_model" in paths


def test_capture_excludes_unrelated_workflow_files(tmp_path):
    project = write_split(
        tmp_path / "cfg",
        bodies={
            "build_model": {"engine": {}},
            "analyze_projections": {"ensemble": "cmip6"},
        },
    )
    captured = cc.capture_configuration_sources(project, "build_model")
    assert [source.id for source in captured] == ["project", "workflow_build_model"]


# ---------------------------------------------------------------------------
# The composition invariant (D-8.1, D-10.1, D-8.7, D-10.4)
# ---------------------------------------------------------------------------


def test_composed_document_equals_the_monolith_it_was_split_from(tmp_path):
    """D-8.1 in its plainest form: split then compose is the identity.

    The permanent digest-equality property test over the shipped seeds lands
    with the migration, once composed seeds exist to compare. This is the same
    claim on a config the test owns, so it holds from this commit on.
    """
    bodies = {
        "build_model": {
            "engine": {"build_config": "config/defaults/wflow_build_model.yml"},
            "simulation_window": {"start": 2000, "end": 2020},
        },
        "analyze_projections": {"ensemble": "cmip6", "scenarios": ["ssp245"]},
    }
    t1_path = write_split(tmp_path / "cfg", bodies=bodies)
    composed = compose(t1_path, "generate_scenarios")

    monolith = yaml.safe_load(t1_path.read_text(encoding="utf-8"))
    for name, body in bodies.items():
        monolith["workflows"][name].pop("config_path")
        monolith["workflows"][name].update(body)
    assert composed == monolith


def test_enabled_is_merged_and_config_path_is_not(tmp_path):
    """D-10.1, both halves.

    Dropping ``enabled`` would re-key every already-generated collection in
    every migrating project: the composed document feeds
    ``effective_config_digest``, so a key that is present rather than absent
    moves the identity even when the resolved value is unchanged, and nothing on
    disk matches any more. Adding ``config_path`` would do the mirror-image
    damage and make run identity depend on where a project stores its files.

    Argued through ``_frozen_differences``' key-union diff over a recorded
    ``experiment.yml`` until 2026-09-17, when that comparator was deleted as
    unreachable (``t2608290250``). The conclusion is unchanged; R12 regenerates
    where the comparator would have refused, so the failure is silent
    recomputation rather than an error.
    """
    t1_path = write_split(
        tmp_path / "cfg",
        bodies={"build_model": {"engine": {"build_config": "wflow.yml"}}},
        enabled={"build_model": False},
    )
    section = compose(t1_path)["workflows"]["build_model"]
    assert section == {"enabled": False, "engine": {"build_config": "wflow.yml"}}
    assert "config_path" not in section


def test_enabled_comes_first_so_recorded_bytes_are_stable(tmp_path):
    """``experiment.yml`` is dumped with ``sort_keys=False``, so order is bytes.

    Value-identity across migration is the guarantee; byte-identity holds
    whenever ``enabled`` was the block's first key, which this construction
    fixes by putting it there.
    """
    t1_path = write_split(
        tmp_path / "cfg", bodies={"generate_scenarios": {"n_realizations": 2}}
    )
    section = compose(t1_path, "generate_scenarios")["workflows"]["generate_scenarios"]
    assert list(section) == ["enabled", "n_realizations"]


def test_omitted_config_path_composes_to_an_empty_body(tmp_path):
    """D-8.7. An omitted key adds nothing — not an empty mapping, nothing.

    ``effective_config_document`` digests the config *mapping*, so a key that is
    present rather than absent moves ``effective_config_digest`` even when the
    resolved value is identical. This is what keeps the projections overlay
    optional: a project that never runs WF2 needs no WF2 file on disk.
    """
    t1_path = write_split(
        tmp_path / "cfg",
        bodies={"build_model": {}, "analyze_projections": {}},
        omit_path=("build_model", "analyze_projections"),
    )
    composed, paths = cc.compose_config(
        yaml.safe_load(t1_path.read_text(encoding="utf-8")),
        t1_path,
        "generate_scenarios",
        PROJECTIONS["generate_scenarios"][0],
    )
    assert composed["workflows"]["build_model"] == {"enabled": True}
    assert composed["workflows"]["analyze_projections"] == {"enabled": True}
    assert paths == {}


def test_empty_t2_file_is_accepted_as_no_settings(tmp_path):
    """A file that parses to ``None`` is a workflow with no settings, not an error."""
    directory = tmp_path / "cfg"
    t1_path = write_split(directory, bodies={"build_model": {"a": 1}})
    (directory / "project_config_test_build_model.yml").write_text("", encoding="utf-8")
    assert compose(t1_path)["workflows"]["build_model"] == {"enabled": True}


def test_no_section_is_hoisted_out_of_its_workflow(tmp_path):
    """R14 D-10.1. A T2 file's top-level section stays in that workflow.

    R13 D-10.4 carried `reporting:` OUT of ``workflows.generate_scenarios`` so a
    figure caption sat outside ``CONFIG_PROJECTION``, the effective-config
    digest and the experiment freeze. `C-77` removes `reporting:` from the
    config surface, which empties the map, and R14 retires the mechanism rather
    than leaving it empty.

    The generic claim is what matters, so this asserts on an arbitrary section
    name rather than on `reporting:` — a hoist reinstated for some OTHER
    section would pass a test written about the one that left.
    """
    t1_path = write_split(
        tmp_path / "cfg",
        bodies={
            "generate_scenarios": {
                "n_realizations": 2,
                "captions": {"title": "Gabon"},
                "anything_else": {"k": "v"},
            }
        },
    )
    composed = compose(t1_path, "generate_scenarios")
    section = composed["workflows"]["generate_scenarios"]
    assert section["captions"] == {"title": "Gabon"}
    assert section["anything_else"] == {"k": "v"}
    assert "captions" not in composed
    assert "anything_else" not in composed


def test_the_hoist_registry_is_retired_not_emptied(tmp_path):
    """Keep retired registries absent so they cannot grow again."""
    assert not hasattr(cc, "HOISTED_SECTIONS"), (
        "the hoist registry is retired (R14 D-10.1, amending R13 D-10.4); a "
        "reinstated one -- even empty -- would let a section be carried out of "
        "run identity by adding an entry beside it"
    )
    assert not hasattr(cc, "CROSS_WORKFLOW_READS"), (
        "cross-workflow values belong in the project file, not an allowlist"
    )


# ---------------------------------------------------------------------------
# R(entry) and the narrowing (§8.3, §8.1, §10.7)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", sorted(PROJECTIONS))
def test_r_entry_is_derived_from_the_declared_projection(tmp_path, entry):
    """§8.3's table, as executable form.

    ``R(entry)`` is derived from the Snakefile's own ``CONFIG_PROJECTION``, never
    restated inside the loader. WF3's three-workflow scope follows because its
    projection is itself derived from ``guarded_sections`` — the drift
    protection is preserved end to end, with the loader as one more consumer of
    the same literal rather than a second copy of it.
    """
    projection, expected = PROJECTIONS[entry]
    bodies = {name: {f"{name}_key": 1} for name in cc.WORKFLOW_NAMES}
    t1_path = write_split(tmp_path / "cfg", bodies=bodies)
    composed, paths = cc.compose_config(
        yaml.safe_load(t1_path.read_text(encoding="utf-8")),
        t1_path,
        entry,
        projection,
    )
    populated = {
        name
        for name, section in composed["workflows"].items()
        if set(section) - {"enabled"}
    }
    assert populated == expected
    assert set(paths) == expected


def test_a_section_outside_scope_is_present_but_unpopulated(tmp_path):
    """The narrowing (§8.1), and why nothing is affected by it.

    A section outside ``R(entry)`` keeps its stanza — so
    ``provenance.project_config`` finds the path and raises no ``KeyError`` —
    but carries no body. Every reader of another workflow's section is already
    declared, so a reader added later that reaches into an unloaded section gets
    a loud parse-time ``KeyError`` gated by ``test_cli``, which is the correct
    outcome: the fix is to declare the dependency in ``CONFIG_PROJECTION``.
    """
    t1_path = write_split(
        tmp_path / "cfg",
        bodies={"build_model": {"a": 1}, "analyze_climate": {"b": 2}},
    )
    composed = compose(t1_path, "build_model")
    assert composed["workflows"]["analyze_climate"] == {"enabled": True}


def test_an_absent_stanza_is_not_created(tmp_path):
    """D-9.1's last sentence: absence is not this check's business.

    ``tests/test_cli.py`` pops ``workflows.analyze_climate`` entirely and asserts
    a WF1 dry-run still succeeds with identical job counts — the additive-carve
    regression. A popped stanza is absent, not malformed, so nothing fires and
    composition must not invent one.
    """
    t1_path = write_split(tmp_path / "cfg", bodies={"build_model": {"a": 1}})
    composed = compose(t1_path, "build_model")
    assert set(composed["workflows"]) == {"build_model"}


# ---------------------------------------------------------------------------
# Path resolution (D-8.4)
# ---------------------------------------------------------------------------


def test_relative_config_path_is_anchored_at_t1_not_the_cwd(tmp_path, monkeypatch):
    """The reversal, on the case that distinguishes the two rules.

    Two files of the same name exist — one beside T1, one in the working
    directory — with different contents. CWD anchoring is what the migration
    could not implement for a production ``project_dir`` outside the repository
    tree, so this is the discriminating assertion, not a formality.
    """
    beside_t1 = tmp_path / "project"
    t1_path = write_split(beside_t1, bodies={"build_model": {"marker": "beside_t1"}})

    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    (elsewhere / "project_config_test_build_model.yml").write_text(
        yaml.safe_dump({"marker": "cwd"}), encoding="utf-8"
    )
    monkeypatch.chdir(elsewhere)

    assert compose(t1_path)["workflows"]["build_model"]["marker"] == "beside_t1"


def test_absolute_config_path_is_used_as_given(tmp_path):
    """An absolute path is never re-anchored."""
    directory = tmp_path / "cfg"
    directory.mkdir()
    t2 = tmp_path / "somewhere_else.yml"
    t2.write_text(yaml.safe_dump({"marker": "absolute"}), encoding="utf-8")
    t1_path = directory / "t1.yml"
    t1_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "project": {"project_dir": "p"},
                **MINIMAL_T1_SECTIONS,
                "workflows": {"build_model": {"enabled": True, "config_path": str(t2)}},
            }
        ),
        encoding="utf-8",
    )
    assert compose(t1_path)["workflows"]["build_model"]["marker"] == "absolute"


def test_tilde_in_config_path_is_expanded(tmp_path, monkeypatch):
    """``~`` is expanded before the relative test, so a home path is absolute."""
    home = tmp_path / "home"
    home.mkdir()
    (home / "wf1.yml").write_text(yaml.safe_dump({"marker": "home"}), encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    directory = tmp_path / "cfg"
    directory.mkdir()
    t1_path = directory / "t1.yml"
    t1_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "project": {"project_dir": "p"},
                **MINIMAL_T1_SECTIONS,
                "workflows": {
                    "build_model": {"enabled": True, "config_path": "~/wf1.yml"}
                },
            }
        ),
        encoding="utf-8",
    )
    assert compose(t1_path)["workflows"]["build_model"]["marker"] == "home"


def test_declared_path_is_cwd_relative_when_one_is_computable(tmp_path, monkeypatch):
    """``workflow_config_paths`` becomes rule inputs, so Snakemake must resolve it."""
    t1_path = write_split(tmp_path / "cfg", bodies={"build_model": {"a": 1}})
    monkeypatch.chdir(tmp_path)
    _, paths = cc.compose_config(
        yaml.safe_load(t1_path.read_text(encoding="utf-8")),
        t1_path,
        "build_model",
        PROJECTIONS["build_model"][0],
    )
    declared = paths["build_model"]
    assert not os.path.isabs(declared)
    assert os.path.isfile(declared)


def test_cross_drive_path_falls_back_to_absolute(tmp_path, monkeypatch):
    """On Windows ``os.path.relpath`` RAISES across drives — a caught error.

    A project on ``D:`` driven from a checkout on ``C:`` is an ordinary case
    here, and it is not reproducible by picking a path, so the failure is
    injected at the call the platform would fail at. An implementation that
    assumed a relative form always exists would raise instead of falling back.
    """
    t1_path = write_split(tmp_path / "cfg", bodies={"build_model": {"a": 1}})

    def _no_relative(*_args, **_kwargs):
        raise ValueError("path is on mount 'D:', start on mount 'C:'")

    monkeypatch.setattr(cc.os.path, "relpath", _no_relative)
    _, paths = cc.compose_config(
        yaml.safe_load(t1_path.read_text(encoding="utf-8")),
        t1_path,
        "build_model",
        PROJECTIONS["build_model"][0],
    )
    assert os.path.isabs(paths["build_model"])


# ---------------------------------------------------------------------------
# §8.4's failure table
# ---------------------------------------------------------------------------


def test_missing_t2_file_names_the_path_the_anchor_and_the_doc(tmp_path):
    """A dangling ``config_path`` is a hard error — a typo cannot pass.

    That is the distinction D-8.7 leans on: deleting the *file* while leaving
    the key fails loudly, while deleting the *key* is a two-step edit that says
    what it means.
    """
    directory = tmp_path / "cfg"
    t1_path = write_split(directory, bodies={"build_model": {"a": 1}})
    (directory / "project_config_test_build_model.yml").unlink()

    with pytest.raises(ValueError) as excinfo:
        compose(t1_path)
    message = str(excinfo.value)
    assert "project_config_test_build_model.yml" in message
    assert str(directory) in message
    assert cc.CONFIG_DOC in message


def test_t2_file_that_is_not_a_mapping_is_refused(tmp_path):
    directory = tmp_path / "cfg"
    t1_path = write_split(directory, bodies={"build_model": {"a": 1}})
    (directory / "project_config_test_build_model.yml").write_text(
        "- one\n- two\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not a mapping|list"):
        compose(t1_path)


def test_unparseable_t2_file_is_refused_with_the_parser_reason(tmp_path):
    directory = tmp_path / "cfg"
    t1_path = write_split(directory, bodies={"build_model": {"a": 1}})
    (directory / "project_config_test_build_model.yml").write_text(
        "a: [1, 2\nb: {\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not valid YAML"):
        compose(t1_path)


@pytest.mark.parametrize("value", [42, ["a"], {"a": 1}, None])
def test_non_string_config_path_is_refused(tmp_path, value):
    directory = tmp_path / "cfg"
    directory.mkdir()
    t1_path = directory / "t1.yml"
    t1_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "project": {},
                "workflows": {"build_model": {"enabled": True, "config_path": value}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be a path string"):
        compose(t1_path)


def test_empty_config_path_says_to_omit_the_key(tmp_path):
    """An empty string is a half-finished edit, and the fix is the D-8.7 shape."""
    directory = tmp_path / "cfg"
    directory.mkdir()
    t1_path = directory / "t1.yml"
    t1_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "project": {},
                "workflows": {"build_model": {"enabled": True, "config_path": "  "}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Omit the key"):
        compose(t1_path)


@pytest.mark.parametrize(
    "second",
    ["wf.yml", "./wf.yml", "WF.yml", ".\\wf.yml"],
    ids=["identical", "dot-prefixed", "case-only", "backslash-dot"],
)
def test_two_stanzas_sharing_one_file_are_refused(tmp_path, second):
    """The duplicate check compares ``normcase(abspath(...))``, not the raw string.

    A raw-string comparison is evaded by every spelling parametrized here, and
    no name-level check can substitute: with one file on disk there is only one
    file to compare. The repo has already paid for this bug class one layer
    down, in ``copy_config_files``' destination collision check — same
    comparison key, same reason.

    The case-only spelling only collides on a case-insensitive filesystem, which
    is where the silent-overwrite version of this bug lives.
    """
    if second == "WF.yml" and os.path.normcase("A") != "a":
        pytest.skip("case-only collision needs a case-insensitive filesystem")
    if second == ".\\wf.yml" and os.sep != "\\":
        pytest.skip("backslash separators are only a path on Windows")

    directory = tmp_path / "cfg"
    directory.mkdir()
    (directory / "wf.yml").write_text(yaml.safe_dump({"a": 1}), encoding="utf-8")
    t1_path = directory / "t1.yml"
    t1_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "project": {},
                "workflows": {
                    "build_model": {"enabled": True, "config_path": "wf.yml"},
                    "analyze_projections": {"enabled": True, "config_path": second},
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="resolve to the same file"):
        compose(t1_path, "generate_scenarios")


# ---------------------------------------------------------------------------
# The seam checks (D-9.1, D-9.2, D-9.3, D-9.5)
# ---------------------------------------------------------------------------


def test_a_third_stanza_key_is_refused_and_names_the_key(tmp_path):
    """D-9.1. The closure is the migration detector as well as the seam check."""
    directory = tmp_path / "cfg"
    directory.mkdir()
    t1_path = directory / "t1.yml"
    t1_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "project": {},
                "workflows": {
                    "build_model": {"enabled": True, "model_build_config": "x.yml"}
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as excinfo:
        compose(t1_path)
    message = str(excinfo.value)
    assert "model_build_config" in message
    assert cc.CONFIG_DOC in message


def test_stanza_closure_is_checked_outside_this_entry_points_scope(tmp_path):
    """D-9.1's scope, which the two readings answer oppositely.

    Checking closure only for ``R(entry)`` would make a half-split config legal:
    a project that split ``build_model`` and left the other three inline would
    run WF1 clean — the very defect the clean-break posture rejects, arriving
    through this design's own door.
    """
    directory = tmp_path / "cfg"
    directory.mkdir()
    t1_path = directory / "t1.yml"
    t1_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "project": {},
                "workflows": {
                    "build_model": {"enabled": True},
                    "generate_scenarios": {"enabled": True, "realizations_num": 2},
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="realizations_num"):
        compose(t1_path, "build_model")


def test_a_stray_t1_top_level_key_is_refused(tmp_path):
    """D-9.5, which is what makes the migration detector complete.

    A config whose only unmigrated element is a top-level ``reporting:``
    produces no extra key under any ``workflows.<name>``, so the stanza check
    alone sees nothing. Closing the top level also turns the hoist-collision
    question into a non-question: there is no precedence rule to define because
    a collision cannot be constructed.
    """
    directory = tmp_path / "cfg"
    directory.mkdir()
    t1_path = directory / "t1.yml"
    t1_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "project": {},
                "workflows": {"build_model": {"enabled": True}},
                "reporting": {"title": "left behind"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as excinfo:
        compose(t1_path)
    assert "reporting" in str(excinfo.value)


def test_a_shared_key_planted_in_a_t2_file_is_refused(tmp_path):
    """D-9.2. A key read by more than one workflow lives in T1, never in a T2."""
    t1_path = write_split(
        tmp_path / "cfg",
        sections={"basin": {"region": "x"}, "climate": {"selected": "era5"}},
        bodies={"build_model": {"selected": "chirps"}},
    )
    with pytest.raises(ValueError) as excinfo:
        compose(t1_path)
    message = str(excinfo.value)
    assert "selected" in message
    # The fix names the SECTION the leaf belongs to, which is what the R14
    # re-derivation buys: under the v1 flat set the message could only say
    # "the project file", leaving a user to find which of three sections.
    assert "`climate:`" in message


@pytest.mark.parametrize("workflow", ["analyze_climate", "build_model"])
def test_map_reference_policy_is_refused_in_workflow_files(tmp_path, workflow):
    """Historical map policy is project-owned even when omitted from the project."""
    project = write_split(
        tmp_path / "cfg",
        bodies={workflow: {"map_reference": {"max_missing_days_month": 3}}},
    )
    with pytest.raises(ValueError, match="map_reference"):
        compose(project, workflow)


def test_a_frozen_seam_key_absent_from_this_t1_is_still_refused(tmp_path):
    """Why ``SHARED_SEAM_KEYS`` exists rather than deriving from T1 alone.

    A T1 that omits an optional leaf would not reject a copy of it planted in a
    T2 file — the exact failure the rule is written against. ``basin.sources``
    is the case here: optional, absent from this T1, and still refused.
    """
    t1_path = write_split(
        tmp_path / "cfg",
        sections={"basin": {"region": "x"}, "climate": {"selected": "era5"}},
        bodies={"build_model": {"sources": {"lulc": "vito"}}},
    )
    t1_doc = yaml.safe_load(t1_path.read_text(encoding="utf-8"))
    assert "sources" not in t1_doc["basin"] and "sources" not in t1_doc["climate"]
    with pytest.raises(ValueError, match="sources"):
        compose(t1_path)


def test_an_independent_same_named_pair_across_two_t2_files_parses(tmp_path):
    """The r3 restriction: the seam is defined by CONSUMPTION, not spelling.

    Two workflows can legitimately own unrelated settings that happen to share a
    local name; their workflow namespaces distinguish them and neither reads the
    other's. Rejecting the pair would force artificial names or needless hoists,
    degrading the modular boundary this design exists to build. What the earlier
    unconditional rejection bought — first-symptom detection of a genuinely new
    cross-workflow key — is carried by the static scan instead.
    """
    t1_path = write_split(
        tmp_path / "cfg",
        bodies={
            "build_model": {"output_dir": "models/wflow"},
            "analyze_projections": {"output_dir": "climate/cmip6"},
        },
    )
    composed = compose(t1_path, "generate_scenarios")
    assert composed["workflows"]["build_model"]["output_dir"] == "models/wflow"
    assert composed["workflows"]["analyze_projections"]["output_dir"] == "climate/cmip6"


def test_a_shared_key_is_refused_in_a_file_outside_this_entry_points_scope(tmp_path):
    """D-9.3's every-file reach, which is what distinguishes it from D-9.2.

    Three of the four entry points have a singleton ``R(entry)``, so a check
    confined to the composed set has exactly one file to look at and could never
    see a shared name planted in another workflow's file.
    """
    t1_path = write_split(
        tmp_path / "cfg",
        sections={"basin": {"region": "x"}, "climate": {"selected": "era5"}},
        bodies={
            "build_model": {"a": 1},
            "generate_scenarios": {"basin": {"region": "planted"}},
        },
    )
    with pytest.raises(ValueError, match="basin"):
        compose(t1_path, "build_model")


@pytest.mark.parametrize(
    "breakage",
    ["missing", "unparseable", "not_mapping"],
)
def test_a_broken_file_outside_scope_is_skipped_and_inside_scope_is_fatal(
    tmp_path, breakage, capsys
):
    """The tolerance clause and its boundary, as one paired assertion.

    Load-bearing and easy to invert: without the clause a broken WF3 config
    would break a WF1 run, which is the inverse of this milestone's goal — a
    workflow's own file must not be able to fail another workflow's parse. With
    the clause applied too widely, a genuinely missing file this entry point
    needs would pass silently.
    """
    directory = tmp_path / "cfg"
    t1_path = write_split(
        directory,
        bodies={"build_model": {"a": 1}, "generate_scenarios": {"b": 2}},
    )
    wf3_file = directory / "project_config_test_generate_scenarios.yml"
    if breakage == "missing":
        wf3_file.unlink()
    elif breakage == "unparseable":
        wf3_file.write_text("a: [1, 2\n", encoding="utf-8")
    else:
        wf3_file.write_text("- a\n- b\n", encoding="utf-8")

    # WF1 does not load the WF3 file: skipped, logged, and the run proceeds.
    composed = compose(t1_path, "build_model")
    assert composed["workflows"]["build_model"]["a"] == 1
    assert "generate_scenarios" in capsys.readouterr().out

    # WF3 does load it, so the identical file is a hard error.
    with pytest.raises(ValueError):
        compose(t1_path, "generate_scenarios")


# ---------------------------------------------------------------------------
# The static cross-workflow read scan (D-9.6)
# ---------------------------------------------------------------------------


def test_scan_surfaces_cover_the_run_path_and_stop_at_dev_scripts():
    """The scan reads every authorized runtime-consumer surface, and only those.

    ``dev/scripts/`` stays outside on the repository's own invocation-model
    rule: it is never part of a run, and a ``dev/scripts/`` tool that acquires a
    run-path caller must move into the shipped package — where the scan sees it.
    """
    surfaces = cc.scan_surfaces(REPO_ROOT)
    for name in cc.WORKFLOW_NAMES:
        assert f"{name}.smk" in surfaces
    assert any(s.startswith("blueearth_cst/") for s in surfaces)
    assert any(s.startswith("scripts/") for s in surfaces)
    assert not any(s.startswith("dev/") for s in surfaces)
    assert not any("__pycache__" in s for s in surfaces)


def test_cross_workflow_reads_are_complete_and_minimal():
    """D-9.6, both directions, over the real tree.

    A grep sweep is a snapshot; this is the thing that keeps the inventory
    honest. **Completeness**: an undeclared cross-workflow read anywhere on the
    scanned surfaces turns this red — and the fix is to promote the key to
    ``shared:``, never to add an entry here. **Minimality**: a declared entry
    with no live site is equally a failure, so no enumeration can drift stale.

    The three enumerations are compared separately rather than merged, so that
    retiring ``CROSS_WORKFLOW_READS`` in the hoist commit cannot silently absorb
    an identity comparison into a value read.
    """
    value_reads, identity, ownerless = cc.partition_hits(
        cc.scan_cross_workflow_reads(REPO_ROOT)
    )
    assert value_reads == frozenset(), (
        "a cross-workflow VALUE read exists. There is no registry to add it to, "
        "on purpose: promote the key to `shared:` instead. An expandable "
        "allowlist cannot enforce shrink-only -- a read plus a matching entry "
        "keeps the test green -- so the assertion is a literal zero."
    )
    assert identity == cc.IDENTITY_COMPARISONS
    assert ownerless == cc.OWNERLESS_SECTION_READS


def test_the_scan_detects_a_planted_cross_workflow_read(tmp_path):
    """The scan can FAIL, which is what makes the assertion above evidence.

    A test that only ever ran against a clean tree would pass forever whether or
    not the patterns matched anything. This plants the violation the rule
    exists to catch — a stage module reading another workflow's settings — in a
    synthetic tree shaped like the real one.
    """
    for name in cc.WORKFLOW_NAMES:
        (tmp_path / f"{name}.smk").write_text("rule all:\n    input: []\n", "utf-8")
    stage = tmp_path / "blueearth_cst" / "projections"
    stage.mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (stage / "sneaky.py").write_text(
        textwrap.dedent(
            '''
            """A stage module reaching across the section boundary."""
            from blueearth_cst.shared.snake_utils import get_config


            def horizons(config):
                return get_config(config["workflows"]["build_model"], "simulation_window")
            '''
        ).lstrip(),
        encoding="utf-8",
    )
    hits = cc.scan_cross_workflow_reads(tmp_path)
    value_reads, _, _ = cc.partition_hits(hits)
    assert ("analyze_projections", "build_model", "simulation_window") in value_reads


def test_the_scan_ignores_prose_that_quotes_a_projection(tmp_path):
    """A docstring or comment naming a section is prose about a read, not a read.

    ``provenance.project_config``'s docstring quotes ``"workflows.build_model"``
    verbatim to explain what a projection is. A scan that could not tell them
    apart would force a prose sentence into a contract enumeration — which is
    how an allowlist starts absorbing things nobody reviewed.
    """
    for name in cc.WORKFLOW_NAMES:
        (tmp_path / f"{name}.smk").write_text("rule all:\n    input: []\n", "utf-8")
    stage = tmp_path / "blueearth_cst" / "shared"
    stage.mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (stage / "prose.py").write_text(
        textwrap.dedent(
            '''
            """A projection is a list of dotted paths -- "workflows.build_model"
            for WF1, for instance.
            """


            def helper(config):
                # config["workflows"]["build_model"] used to be read here
                return config
            '''
        ).lstrip(),
        encoding="utf-8",
    )
    assert cc.scan_cross_workflow_reads(tmp_path) == ()


# ---------------------------------------------------------------------------
# The binding invariant (D-8.6a)
# ---------------------------------------------------------------------------


@pytest.mark.workflow_contract
def test_the_composed_shape_is_visible_through_snakemake_workflow_config(tmp_path):
    """The rebinding is load-bearing, and nothing in the loader can enforce it.

    ``check_project_consistency`` takes its live config from ``sm.config`` —
    Snakemake's ``workflow.config`` — not from the Snakefile's local name. A
    ``composed = compose_config(...)`` refactor would leave the drift guard
    comparing ``{enabled, config_path}`` against the snapshot's full section and
    fail every WF3 run at rule 3.01, **after WF1 and WF2 have already run**,
    with a message blaming project drift rather than the binding.

    Checked here against a synthetic Snakefile rather than a shipped one, so the
    invariant is pinned from the commit that introduces the loader — before the
    four entry points are wired to call it.
    """
    t1_path = write_split(
        tmp_path / "cfg", bodies={"build_model": {"engine": {"build_config": "x.yml"}}}
    )
    snakefile = tmp_path / "probe.smk"
    snakefile.write_text(
        textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {str(REPO_ROOT)!r})
            from blueearth_cst.shared.config_composition import compose_config

            config, WORKFLOW_CONFIG_PATHS = compose_config(
                config, workflow.configfiles[0], entry="build_model",
                declared_sections=("project", "basin", "climate", "model", "workflows.build_model"),
            )
            LIVE = workflow.config["workflows"]["build_model"]
            assert LIVE == {{"enabled": True, "engine": {{"build_config": "x.yml"}}}}, LIVE

            rule all:
                run:
                    pass
            """
        ).lstrip(),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "snakemake",
            "all",
            "-c",
            "1",
            "-s",
            str(snakefile),
            "--configfile",
            str(t1_path),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")


# ---------------------------------------------------------------------------
# The tool-facing wrapper (D-12.0)
# ---------------------------------------------------------------------------


def test_load_composed_config_without_an_entry_loads_every_resolvable_file(tmp_path):
    """ "Whatever section I index" is the right scope for a caller that is not
    an entry point — the ``tree-check`` snapshot tool, the cache pruners, the
    DAG renderer. Each becomes a two-line change: import this, call it instead
    of ``yaml.safe_load``.
    """
    t1_path = write_split(
        tmp_path / "cfg",
        bodies={
            "build_model": {"a": 1},
            "analyze_projections": {"b": 2},
            "simulate_system": {"experiment_name": "gabon"},
        },
    )
    composed = cc.load_composed_config(t1_path)
    assert composed["workflows"]["simulate_system"]["experiment_name"] == "gabon"
    assert composed["workflows"]["build_model"]["a"] == 1
    assert composed["workflows"]["analyze_projections"]["b"] == 2


def test_a_broken_workflow_file_does_not_break_a_tool(tmp_path, capsys):
    """The tolerance clause is what keeps ``tree-check`` reporting on a project
    whose WF3 config is mid-edit, instead of reporting nothing at all."""
    directory = tmp_path / "cfg"
    t1_path = write_split(
        directory, bodies={"build_model": {"a": 1}, "generate_scenarios": {"b": 2}}
    )
    (directory / "project_config_test_generate_scenarios.yml").write_text(
        "a: [1, 2\n", encoding="utf-8"
    )
    composed = cc.load_composed_config(t1_path)
    assert composed["workflows"]["build_model"]["a"] == 1
    assert composed["workflows"]["generate_scenarios"] == {"enabled": True}
    assert "generate_scenarios" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# RETIRED 2026-08-27 — digest equality across the R13 migration (was D-10.2)
# ---------------------------------------------------------------------------
#
# `test_effective_config_digest_survives_the_migration` and
# `test_guarded_sections_digest_survives_the_migration` lived here, with the
# `MIGRATED` pairs and `PRESPLIT_DIR` they needed. Both asserted that R13's
# split did NOT move a digest, against frozen pre-split specimens under
# `tests/data/presplit/`.
#
# **They were removed rather than fixed, and the reason is that their subject no
# longer exists.** Two things ended them:
#
#   * R14 moves `effective_config_digest` ONCE, deliberately (`K3` — "a key
#     present-vs-absent moves the digest and refuses every already-run
#     experiment; that cost is payable once, which is why everything lands as
#     one bundle"). An assertion that it does NOT move is now an assertion
#     against the design, and making it pass would mean not doing R14.
#   * `presplit/` held PRE-R13 whole documents so R13's splitter could be
#     checked against them. P1 retired that splitter in `068b81dd`. The
#     fixtures had no remaining consumer once these two tests went.
#
# What replaced the coverage, so this is a deletion and not a hole:
#
#   * `test_write_config_round_trips_through_compose`, still below, holds the
#     composition helper to `compose(write(cfg)) == cfg`;
#   * the v1 -> v2 rewriter and its tests, which shipped through v0.3.0 and
#     were retired after it (t2610041227);
#   * P4's own falsifier re-runs the rewriter from the pre-migration commit and
#     diffs, which is a stronger statement than digest equality: it says the
#     shipped configs and the user's migration are byte-identical.
#
# The digest's new value is Gate 5's business (`check_baseline`), not this
# module's. Retired under owner ruling, R14 P4.
def test_write_config_round_trips_through_compose(tmp_path):
    """``compose_config(write_config(cfg)) == cfg`` — D-12.6's gate.

    ``tests/conftest.write_config`` is what let the suite's dominant config
    idiom survive the split unchanged: tests keep mutating one whole mapping and
    the helper writes it back as a T1 + T2 set. Without this property the helper
    would be a second, unchecked way to build a config, and every test that used
    it would be asserting against a shape no run can have.
    """
    from tests.conftest import write_config

    cfg = cc.load_composed_config(REPO_ROOT / "test_case/project_config_rapid.yml")
    # A workflow-owned section, not a top-level one: with the hoist retired
    # (R14 D-10.1) a top-level key outside `T1_TOP_LEVEL` is a parse error, so
    # the round trip is exercised where a section can actually live.
    #
    # `compute:` rather than `reporting:`, which `C-77` removed from the config
    # surface entirely -- the loader refuses it now, so it could no longer serve
    # as a payload. The property under test is the round trip, not the key.
    cfg["workflows"]["simulate_system"]["compute"] = {"batch_size": 4}
    assert cc.load_composed_config(write_config(tmp_path, cfg)) == cfg


def test_the_hoisted_key_is_refused_inside_a_workflow_file(tmp_path):
    """R13 D-9.7's placement, in its R14 spelling (`model.outvars`, `C-19`).

    The mechanism is unchanged and is the point: the key is covered by
    ``SHARED_SEAM_KEYS``, so a copy planted in any workflow file is refused by
    D-9.2/D-9.3 with no rule written for this key specifically. R14 changes
    only where the coverage comes from -- `outvars` is now a DERIVED member,
    a leaf of `model:`, rather than a hand-added name.
    """
    t1_path = write_split(
        tmp_path / "cfg",
        sections={
            "basin": {"region": "x"},
            "climate": {"selected": "era5"},
            "model": {"outvars": ["river discharge"]},
        },
        bodies={"build_model": {"outvars": ["snow"]}},
    )
    with pytest.raises(ValueError) as excinfo:
        compose(t1_path)
    message = str(excinfo.value)
    assert "outvars" in message
    assert "`model:`" in message


def test_the_hoisted_key_reaches_both_readers_from_shared(tmp_path):
    """Both workflows read the same value from one place, which is the point.

    WF1 builds the model with it and WF3 derives its indicator tables from it.
    Before the hoist that was the repository's ONE cross-workflow value read,
    and the registry existed to sanction it; now there is one authoring site and
    no registry.
    """
    t1_path = write_split(
        tmp_path / "cfg",
        sections={
            "basin": {"region": "x"},
            "climate": {"selected": "era5"},
            "model": {"outvars": ["river discharge"]},
        },
        bodies={
            "build_model": {"engine": {"build_config": "x.yml"}},
            "generate_scenarios": {},
        },
    )
    for entry in ("build_model", "generate_scenarios"):
        composed = compose(t1_path, entry)
        assert composed["model"]["outvars"] == ["river discharge"]


# ---------------------------------------------------------------------------
# R14: the seam set re-derived, and the parse-time refusals (D-10.3, D-10.4)
# ---------------------------------------------------------------------------


def test_a_bare_window_in_a_t2_file_is_refused(tmp_path):
    """THE falsifier for the R14 seam re-derivation (D-10.3).

    `SHARED_SEAM_KEYS` is a flat set of NAMES, so nesting moves seam coverage
    from leaf to group: `historical_window` was a `shared:` leaf and was in the
    set by name, but `climate.window` is a leaf of a GROUP, and a set derived
    from section names alone would carry `climate` and not `window`. A T2 file
    could then declare a bare `window:` uncaught -- and `basin` already has that
    property, which is exactly why the design calls for an explicit,
    re-DERIVED set rather than one edited name by name.

    Written before the implementation, and it failed then for the right reason:
    against the v1 set a bare `window:` composed cleanly into
    `workflows.build_model`.
    """
    t1_path = write_split(
        tmp_path / "cfg",
        sections={"basin": {"region": "x"}, "climate": {"window": {"start": 1990}}},
        bodies={"build_model": {"window": {"start": 1901}}},
    )
    with pytest.raises(ValueError) as excinfo:
        compose(t1_path)
    assert "window" in str(excinfo.value)


def test_the_seam_set_covers_every_declared_t1_leaf():
    """D-10.3's re-derivation, asserted as a PROPERTY rather than a snapshot.

    A snapshot (``assert SHARED_SEAM_KEYS == {...}``) would pass just as
    happily against a hand-edited set, which is the thing the design forbids.
    What matters is that every leaf a T2 file must not declare is covered BY
    NAME, because the set is flat — so this asserts the derivation, and the
    equality below asserts there is nothing else in it.
    """
    for section, leaves in cc.T1_SHARED_SECTIONS.items():
        assert section in cc.SHARED_SEAM_KEYS, (
            f"the section name {section!r} must be refused in a T2 file"
        )
        for leaf in leaves:
            assert leaf in cc.SHARED_SEAM_KEYS, (
                f"{section}.{leaf} is a T1 leaf, and the seam set is FLAT, so it "
                "must be covered by its own name -- nesting moves coverage from "
                "leaf to group and a bare copy in a T2 file would go uncaught"
            )
    derived = frozenset(cc.T1_SHARED_SECTIONS) | {
        leaf for leaves in cc.T1_SHARED_SECTIONS.values() for leaf in leaves
    }
    assert cc.SHARED_SEAM_KEYS == derived, (
        "SHARED_SEAM_KEYS must be exactly the derivation, with no name added by "
        "hand beside it -- a hand-added name is one the T1 shape does not know "
        "about, and it is how the two drift"
    )


def test_project_leaves_are_deliberately_outside_the_seam_set():
    """``catalog`` is a T1 leaf AND a legal WF2 key, on purpose (C-39/C-40).

    This is the boundary of the re-derivation, and it is worth a test because
    the obvious generalisation — derive from EVERY T1 section, ``project:``
    included — refuses a shape the design ships. ``project:`` was never part of
    ``shared:`` and is rejected in a T2 file as a section name, as it always
    was.
    """
    assert "project" not in cc.T1_SHARED_SECTIONS
    assert "catalog" not in cc.SHARED_SEAM_KEYS
    assert "project_dir" not in cc.SHARED_SEAM_KEYS
    assert "project" in cc.T1_TOP_LEVEL


def test_the_wf2_catalog_key_composes(tmp_path):
    """The same claim, exercised rather than asserted about the constants."""
    t1_path = write_split(
        tmp_path / "cfg",
        bodies={"analyze_projections": {"catalog": "config/catalogs/cmip6_data.yml"}},
    )
    composed = compose(t1_path, "analyze_projections")
    section = composed["workflows"]["analyze_projections"]
    assert section["catalog"] == "config/catalogs/cmip6_data.yml"


# --- D-10.4, one case per row. Each asserts the message names the FIX. -------


def test_a_v1_config_is_refused_and_names_the_migration_command(tmp_path):
    """Row 1. An absent ``schema_version`` is the v1 statement (C-05)."""
    t1_path = write_split(
        tmp_path / "cfg", bodies={"build_model": {}}, schema_version=None
    )
    with pytest.raises(ValueError) as excinfo:
        compose(t1_path)
    message = str(excinfo.value)
    assert cc.MIGRATION_COMMAND in message, "the fix is a command, so name it"
    assert str(t1_path) in message, "and name the file to run it on"
    assert cc.MIGRATION_DOC in message


def test_a_lower_schema_version_is_refused_the_same_way(tmp_path):
    """Row 1's other half: an explicit 1 says what an absent key says."""
    t1_path = write_split(
        tmp_path / "cfg", bodies={"build_model": {}}, schema_version=1
    )
    with pytest.raises(ValueError) as excinfo:
        compose(t1_path)
    assert cc.MIGRATION_COMMAND in str(excinfo.value)


def test_the_schema_check_runs_before_every_other_refusal(tmp_path):
    """Ordering is a contract, not an implementation detail.

    A v1 document trips half a dozen refusals at once. Reporting any of the
    others would send a user to fix ONE key in a file that needs migrating
    whole, and they would then meet the next one. This builds a document that
    is wrong in three ways and asserts which one it is told about.
    """
    directory = tmp_path / "cfg"
    directory.mkdir()
    t1_path = directory / "t1.yml"
    t1_path.write_text(
        yaml.safe_dump(
            {
                "project": {"project_dir": "p", "static_dir": "config"},
                "shared": {"basin": {"region": "x"}, "clim_historical": "era5"},
                "workflows": {"build_model": {"enabled": True}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as excinfo:
        compose(t1_path)
    message = str(excinfo.value)
    assert cc.MIGRATION_COMMAND in message
    assert "static_dir" not in message, (
        "a v1 document must be told to migrate, not sent to fix one of the many "
        "individual keys that migrating would fix"
    )


@pytest.mark.parametrize(
    ("path", "planted", "names"),
    [
        ("project.static_dir", {"project": {"static_dir": "config"}}, "deleted"),
        (
            "project.data_sources",
            {"project": {"data_sources": "catalog.yml"}},
            "project.catalog",
        ),
        (
            "basin.gauge_points",
            {"basin": {"gauge_points": "pts.csv"}},
            "basin.output_locations",
        ),
    ],
)
def test_a_retired_t1_key_is_refused_and_names_its_new_home(
    tmp_path, path, planted, names
):
    """Row 2, at T1. The message names WHERE IT WENT, not merely that it is bad.

    Reached only by a document already claiming ``schema_version: 2`` — a
    HALF-migrated one — which is what makes naming the individual key worth
    more here than a generic "run the migration".
    """
    directory = tmp_path / path.replace(".", "_")
    directory.mkdir(parents=True)
    document = {
        "schema_version": 2,
        "project": {"project_dir": "p"},
        **copy.deepcopy(MINIMAL_T1_SECTIONS),
        "workflows": {"build_model": {"enabled": True}},
    }
    for section, body in planted.items():
        document.setdefault(section, {}).update(body)
    t1_path = directory / "t1.yml"
    t1_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        compose(t1_path)
    message = str(excinfo.value)
    assert path in message
    assert names in message


def test_a_retired_t2_key_is_refused_and_names_its_new_home(tmp_path):
    """Row 2, in a workflow file."""
    t1_path = write_split(
        tmp_path / "cfg",
        bodies={"generate_scenarios": {"realizations_num": 2, "run_historical": True}},
    )
    with pytest.raises(ValueError) as excinfo:
        compose(t1_path, "generate_scenarios")
    message = str(excinfo.value)
    assert "n_realizations" in message, "name the key it became"
    assert "st_0" in message, "and say what deleting `run_historical` does"
    assert "generate_scenarios file" in message, "and which file to open"


def test_a_retired_key_is_refused_in_any_workflow_file(tmp_path):
    """The ``T2.*`` wildcard: some names are gone everywhere, not in one file."""
    for owner in ("build_model", "generate_scenarios"):
        t1_path = write_split(
            tmp_path / owner, bodies={owner: {"reporting": {"title": "anywhere"}}}
        )
        with pytest.raises(ValueError) as excinfo:
            compose(t1_path, owner)
        message = str(excinfo.value)
        assert "reporting" in message
        assert "deleted" in message and "C-77" in message


def test_unset_climate_selection_is_refused_when_wf1_or_wf3_runs(tmp_path):
    """Row 3. The message names the candidates AND how to choose between them."""
    t1_path = write_split(
        tmp_path / "cfg",
        sections={
            "basin": {"region": "x"},
            "climate": {"sources": ["era5", "chirps"]},
        },
        bodies={"build_model": {}},
    )
    with pytest.raises(ValueError) as excinfo:
        compose(t1_path)
    message = str(excinfo.value)
    assert "climate.selected" in message
    assert "era5" in message and "chirps" in message, "name the candidates"
    assert "analyze_climate" in message, "and the workflow that compares them"
    assert "comparison" in message, "and where it writes the comparison"


def test_unset_climate_selection_is_valid_with_only_wf0_enabled(tmp_path):
    """D-10.5, and it is a decision rather than a gap.

    "I have candidates, I have not chosen yet" is a legitimate project state,
    and it is exactly the state WF0 exists to resolve: it compares the
    candidates, and the comparison is what a user reads in order to choose.
    Refusing it would make the workflow that answers the question impossible to
    run until the question is answered.
    """
    t1_path = write_split(
        tmp_path / "cfg",
        sections={
            "basin": {"region": "x"},
            "climate": {"sources": ["era5", "chirps"]},
        },
        bodies={"analyze_climate": {}, "build_model": {}, "generate_scenarios": {}},
        enabled={
            "analyze_climate": True,
            "build_model": False,
            "generate_scenarios": False,
        },
    )
    composed = compose(t1_path, "analyze_climate")
    assert composed["climate"]["sources"] == ["era5", "chirps"]
    assert composed["climate"].get("selected") is None


def test_a_non_member_climate_selection_is_refused_and_names_the_set(tmp_path):
    """Row 4. A typo here otherwise builds a model on a dataset nobody listed."""
    t1_path = write_split(
        tmp_path / "cfg",
        sections={
            "basin": {"region": "x"},
            "climate": {"sources": ["era5", "chirps"], "selected": "era-5"},
        },
        bodies={"build_model": {}},
    )
    with pytest.raises(ValueError) as excinfo:
        compose(t1_path)
    message = str(excinfo.value)
    assert "era-5" in message
    assert "era5" in message and "chirps" in message, "name the member set"
    assert "add" in message, "and both ways out, not just one"


def test_an_observation_for_an_undeclared_outvar_is_refused(tmp_path):
    """Row 5, and D-7.3's whole reason for keying observations by variable.

    Without the check the typo is silent: the evaluation finds no modelled
    series to compare against and draws an empty panel, which reads as "the
    model is bad here" rather than "this key is misspelled".
    """
    t1_path = write_split(
        tmp_path / "cfg",
        sections={
            "basin": {"region": "x"},
            "climate": {"sources": ["era5"], "selected": "era5"},
            "model": {"outvars": ["river discharge"]},
        },
        bodies={"build_model": {"observations": {"river disharge": "obs.csv"}}},
    )
    with pytest.raises(ValueError) as excinfo:
        compose(t1_path)
    message = str(excinfo.value)
    assert "river disharge" in message
    assert "river discharge" in message, "name the outvars that ARE declared"
    assert "model.outvars" in message, "and where to add it if it belongs"


def test_an_observation_for_a_declared_outvar_composes(tmp_path):
    """The other direction, so the check cannot pass by refusing everything."""
    t1_path = write_split(
        tmp_path / "cfg",
        sections={
            "basin": {"region": "x"},
            "climate": {"sources": ["era5"], "selected": "era5"},
            "model": {"outvars": ["river discharge"]},
        },
        bodies={"build_model": {"observations": {"river discharge": "obs.csv"}}},
    )
    section = compose(t1_path)["workflows"]["build_model"]
    assert section["observations"] == {"river discharge": "obs.csv"}


@pytest.mark.parametrize("axis", ["temp", "precip"])
def test_a_perturbation_axis_without_a_trajectory_is_refused(tmp_path, axis):
    """Row 6. Both axes, because "either" is the claim and one is not both."""
    perturbations = {
        "temp": {"n_levels": 2, "trajectory": "transient"},
        "precip": {"n_levels": 3, "trajectory": "transient"},
    }
    perturbations[axis].pop("trajectory")
    t1_path = write_split(
        tmp_path / "cfg",
        bodies={"generate_scenarios": {"climate_perturbations": perturbations}},
    )
    with pytest.raises(ValueError) as excinfo:
        compose(t1_path, "generate_scenarios")
    message = str(excinfo.value)
    assert axis in message
    assert "NO default" in message, "say there is no default"
    assert "deliberately" in message, "and that this is a decision, not an omission"
    assert "constant" in message and "transient" in message, "and name the two values"


def test_both_axes_with_a_trajectory_compose(tmp_path):
    """The other direction for row 6."""
    t1_path = write_split(
        tmp_path / "cfg",
        bodies={
            "generate_scenarios": {
                "climate_perturbations": {
                    "temp": {"n_levels": 2, "trajectory": "constant"},
                    "precip": {"n_levels": 3, "trajectory": "transient"},
                }
            }
        },
    )
    section = compose(t1_path, "generate_scenarios")["workflows"]["generate_scenarios"]
    assert section["climate_perturbations"]["temp"]["trajectory"] == "constant"


def test_a_missing_config_path_names_deleting_the_key_as_the_fix(tmp_path):
    """Row 7. The fix a user does not think of: a workflow may have NO settings.

    D-8.7 makes an omitted ``config_path`` legal and distinct from an empty
    file, so "delete the key" is a real answer rather than a workaround — and
    it is the right one whenever the file was never written rather than
    mislaid.
    """
    directory = tmp_path / "cfg"
    t1_path = write_split(directory, bodies={"build_model": {"a": 1}})
    (directory / "project_config_test_build_model.yml").unlink()
    with pytest.raises(ValueError) as excinfo:
        compose(t1_path)
    message = str(excinfo.value)
    assert "does not exist" in message
    assert "delete the `config_path` key" in message
    assert "resolved against" in message, "and say what it was anchored at"


def _plant(document: dict, path: str, value=None):
    """Plant the key ``path`` names into ``document``, creating parents."""
    node = document
    parts = path.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = {"planted": True} if value is None else value


@pytest.mark.parametrize("path", sorted(cc.RETIRED_KEYS))
def test_every_retired_key_row_is_reachable(tmp_path, path):
    """No row of ``RETIRED_KEYS`` may be shadowed by an earlier refusal.

    This is the test that would have caught the phase's own worst bug. Seven of
    the twenty-six rows -- every one describing where ``shared:``'s contents
    went, which is the single most likely half-migration -- were UNREACHABLE
    when written: ``_check_t1_top_level`` answered a leftover ``shared:`` first,
    with a message naming the closed top level and none of ``basin:``,
    ``climate:`` or ``model:``. The table looked complete, the suite was green,
    and a third of the phase's headline deliverable ("every refusal message
    names its fix") was hollow.

    Parametrized over the table itself, so a row added without a reachable path
    to it fails here rather than being discovered by a user.
    """
    tier, _, rest = path.partition(".")
    directory = tmp_path / path.replace(".", "_").replace("*", "any")
    directory.mkdir(parents=True)

    document = {
        "schema_version": 2,
        "project": {"project_dir": "p"},
        **copy.deepcopy(MINIMAL_T1_SECTIONS),
        "workflows": {},
    }
    if tier == "T1":
        _plant(document, rest)
        owners = ["build_model"]
    else:
        owner, _, name = rest.partition(".")
        owners = list(cc.WORKFLOW_NAMES) if owner == "*" else [owner]
        body = {name: {"planted": True}}
        (directory / f"t2_{owners[0]}.yml").write_text(
            yaml.safe_dump(body), encoding="utf-8"
        )
    for owner in owners:
        stanza: dict = {"enabled": True}
        if tier != "T1" and owner == owners[0]:
            stanza["config_path"] = f"t2_{owners[0]}.yml"
        document["workflows"][owner] = stanza

    t1_path = directory / "t1.yml"
    t1_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        compose(t1_path, owners[0])
    message = str(excinfo.value)
    assert cc.RETIRED_KEYS[path] in message, (
        f"planting {path!r} did not produce its RETIRED_KEYS message -- some "
        f"earlier refusal shadows this row, so the destination it records is "
        f"never shown to anyone. Got:\n{message}"
    )
