"""wf3 startup drift guard: experiment config vs project snapshots.

An experiment config is a *full* config (approach A). Its project-level
sections must describe the same project the built model / overlay came from —
otherwise the experiment silently reuses a ``models/hydrology/wflow/`` built
under different settings. This rule (3.00b ``check_project_consistency``) runs
at wf3 rule time and fails loud on divergence, naming the diverging key.

WHICH sections is not written down anywhere: it is derived from the snapshot,
by the rule stated above :func:`guarded_paths`.

Current configuration ownership: dev/reference/toolbox-architecture.md.
Historical rationale: dev/records/milestones/p31/experiment-structure-design.md §3/§3a/§3b/§3d.

The comparator core (``compare_project_consistency``) is a PURE function of the
live config dict + snapshot paths (gate 2 a–h call it directly on staged
config/snapshot pairs with no Snakemake). The ``if "snakemake" in globals()``
harness wires it to the rule; the parse-time digest params (a separate rerun-
trigger layer, gate 2b) live in the Snakefile, not here.

NOTE: no ``from __future__`` import here — Snakemake's ``script:`` runner
prepends a preamble to the script file, so a ``__future__`` import would no
longer be the first statement and raises ``SyntaxError`` at rule time.
"""

import sys
from pathlib import Path

import yaml

# Import shared helpers regardless of the working directory (mirror the sibling
# experiment scripts): the Snakefile prepends its basedir to sys.path before
# invoking script: rules, but guard here so the module is import-clean for unit
# tests too.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from blueearth_cst.shared.config_composition import (  # noqa: E402
    T1_TOP_LEVEL,
    WORKFLOW_STANZA_KEYS,
)
from blueearth_cst.shared.snake_utils import log_row  # noqa: E402

# The guard's key list is DERIVED (P2, design D-9.1/D-9.6/D-9.7). There is no
# maintained tuple here, in the Snakefile's `guarded_sections`, or in its
# `guarded_sections_digest` -- the three used to be kept by hand and two of them
# disagreed with this one.
#
# THE RULE: compare every key a project snapshot carries, except two structural
# exceptions. Both are properties of what a snapshot IS, not entries on a list:
#
#  1. `workflows.<name>.enabled` is dropped from both operands. Toggling a
#     workflow off does not change what the model was built from, and an
#     experiment that switches the projections overlay off must not be refused
#     against a model built with it on.
#  2. A stanza the snapshot's own composition never EXPANDED is skipped whole.
#     A snapshot is that workflow's composed config, so a WF1 snapshot carries
#     `workflows.analyze_projections: {enabled: true}` and nothing more -- it
#     never loaded that file. Comparing that against the live config's fully
#     expanded section would refuse every run of every project. This exception
#     is also what keeps `workflows.run_stress_test` out of the WF1 comparison
#     (design D-9.2: `compute:` was never inside this guard, so carving it out
#     here would be a no-op) and what routes `analyze_projections` to the WF2
#     snapshot, which is the document that actually witnessed it.
#
# What the rule CHANGES, deliberately: `climate:` becomes guarded. The climate
# record is what the model was forced with, and editing it after a build is
# unguarded today (design D-9.1 property 2). `schema_version` becomes guarded
# too, as a plain consequence of "every key" -- a snapshot written under a
# different config shape is not a snapshot of this configuration, and the remedy
# the guard already prints (re-run the owning workflow) is the right one.
#
# What it stops REQUIRING: the leaf-by-leaf narrowing to `basin` and
# `model.outvars`. That existed to keep every comparand experiment-invariant
# while `seed` and `julia_threads` sat in the shared section; `C-51` and `C-54`
# removed them, so the exception has lost its cause. D2 is satisfied by removing
# the cause, not by asserting it away -- and `build_experiment_config` emits
# `{experiment_name, run_stress_test}` and nothing else, so no experiment
# overlay can reach a T1 key in the first place.

#: The workflows whose project snapshots the guard reads, in the order it
#: consults them. A path is compared against the FIRST snapshot that witnesses
#: it, so T1 lands on the WF1 snapshot and only what WF1 never expanded falls
#: through to WF2.
GUARD_SNAPSHOT_WORKFLOWS: tuple[str, ...] = ("build_model", "analyze_projections")

#: Exception 1. Named rather than inlined so the two sites that drop it -- the
#: comparator and the path derivation -- cannot disagree about which key it is.
_STANZA_TOGGLE = "enabled"


def _is_unexpanded(stanza) -> bool:
    """Whether a `workflows.<name>` stanza is one the snapshot never loaded.

    An unexpanded stanza carries nothing outside the closed stanza vocabulary
    (`enabled`, `config_path`), which is precisely the shape `compose_config`
    leaves behind for a workflow outside its own `R(entry)`. A workflow whose
    settings WERE composed always adds keys beyond those two.
    """
    return isinstance(stanza, dict) and set(stanza) <= WORKFLOW_STANZA_KEYS


def guarded_paths(snapshot) -> tuple:
    """The paths to compare against ``snapshot``, derived from its content.

    Returns top-level key strings and ``("workflows", name)`` tuples, in a
    stable order. Deriving from the snapshot rather than from the live config is
    what makes exception 2 self-enforcing: a section the snapshot cannot witness
    is one it does not contain, so it never enters the list.
    """
    if not isinstance(snapshot, dict):
        return ()
    paths = []
    for key in sorted(snapshot):
        if key != "workflows":
            paths.append(key)
            continue
        for name, stanza in sorted((snapshot.get("workflows") or {}).items()):
            if _is_unexpanded(stanza):
                continue
            paths.append(("workflows", name))
    return tuple(paths)


def guarded_section_paths() -> tuple[str, ...]:
    """The same rule as dotted strings, for a caller with no snapshot to read.

    The Snakefile needs this list at PARSE time, to thread the live values
    through rule 3.01's ``params:`` as a rerun trigger and to derive the
    consumed-key projection. It cannot call :func:`guarded_paths`: the snapshot
    lives under ``project_dir``, which is not known until the config is read,
    and on a first run it does not exist at all.

    So this is the same rule expressed over the two things that ARE knowable
    without a project: T1's closed top level, and the two workflows whose
    snapshots the guard reads. It is not a second maintained list --
    ``T1_TOP_LEVEL`` is the loader's own closure and
    ``GUARD_SNAPSHOT_WORKFLOWS`` is above. Presentation-only ``plotting`` is
    omitted from the snapshots; renderer params track it separately.
    """
    return tuple(
        [key for key in T1_TOP_LEVEL if key not in {"workflows", "plotting"}]
        + [f"workflows.{name}" for name in GUARD_SNAPSHOT_WORKFLOWS]
    )


# Directional OLD->NEW path map (pre-R6 flat -> post-R6 binned), mirroring
# dev/scripts/semantic_tree_diff.py COPIED_CONFIG_PATH_MAP (dev/ scripts are
# not importable from the shipped package, so the map is restated here). For
# the drift guard BOTH operands are contemporaneous post-R6 configs, so this
# matches nothing and is a no-op — plain section-scoped deep equality already
# gives the correct pass/fail. It is a DEFENSIVE layer for the one edge case
# where a hand-migrated flat-path experiment config is compared against a
# binned snapshot, and (unlike ``compare_copied_config``'s directional one-side
# application) it is applied SYMMETRICALLY to both operands, so a flat-vs-binned
# pair converges while two binned configs are unchanged (design §3b, gate 2d).
_COPIED_CONFIG_PATH_MAP: dict[str, dict[str, str]] = {
    # `C-40` renamed `project.data_sources` to `project.catalog` and `C-39`
    # pushed the climate catalog DOWN to `workflows.analyze_projections.catalog`.
    # Both leaves are spelled `catalog`, so the two former entries merge into
    # one -- `_normalize_paths` matches on the LEAF name at any depth, and the
    # union is what either tier can carry.
    "catalog": {
        "config\\deltares_data.yml": "config/catalogs/deltares_data.yml",
        "config\\deltares_data_linux.yml": "config/catalogs/deltares_data_linux.yml",
        "config\\deltares_data_analyze_projections.yml": "config/catalogs/deltares_data_analyze_projections.yml",
        "config\\deltares_data_analyze_projections_linux.yml": "config/catalogs/deltares_data_analyze_projections_linux.yml",
        "config\\cmip6_data.yml": "config/catalogs/cmip6_data.yml",
    },
    # TWO old spellings each, because these files have moved twice: pre-R6 flat,
    # then config/templates/, then config/defaults/ (2026-08-11, when templates/
    # was split so it holds only files you copy). Both must normalize onto the
    # current path or a config from either era fails against a fresh snapshot.
    # `C-22`: keyed by the CONFIG KEY, and the leaf under `engine:` is
    # `build_config`. `waterbodies_config` below kept its leaf name.
    "build_config": {
        "config\\wflow_build_model.yml": "config/defaults/wflow_build_model.yml",
        "config/templates/wflow_build_model.yml": "config/defaults/wflow_build_model.yml",
    },
    "waterbodies_config": {
        "config\\wflow_update_waterbodies.yml": "config/defaults/wflow_update_waterbodies.yml",
        "config/templates/wflow_update_waterbodies.yml": "config/defaults/wflow_update_waterbodies.yml",
    },
}


def _normalize_paths(doc):
    """Symmetric defensive OLD->NEW path normalization, in place, recursive.

    Rewrites a key's value only when it equals a documented OLD path exactly;
    every other value is left untouched (and fails the equality step if it
    diverges). Applied at any nesting depth, and to BOTH operands (see the map
    comment above).
    """
    if isinstance(doc, dict):
        for k, v in doc.items():
            if k in _COPIED_CONFIG_PATH_MAP and isinstance(v, str):
                doc[k] = _COPIED_CONFIG_PATH_MAP[k].get(v, v)
            else:
                _normalize_paths(v)
    elif isinstance(doc, list):
        for item in doc:
            _normalize_paths(item)
    return doc


def _get_section(cfg, path):
    """Return the (possibly nested) config section named by ``path``.

    ``path`` is a top-level key string (``"project"``) or a tuple of nested
    keys (``("shared", "basin")``). A missing intermediate key yields ``None``
    (a divergence the comparator reports, not a crash).
    """
    keys = (path,) if isinstance(path, str) else path
    node = cfg
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _section_label(path) -> str:
    return path if isinstance(path, str) else ".".join(path)


def _first_diff(snap, live, prefix: str) -> list[str]:
    """First-difference reporter for two parsed structures (dicts/lists/scalars)."""
    if isinstance(snap, dict) and isinstance(live, dict):
        diffs: list[str] = []
        for k in sorted(set(snap) | set(live)):
            p = f"{prefix}.{k}" if prefix else str(k)
            if k not in snap:
                diffs.append(f"{p}: present in experiment config, absent in snapshot")
            elif k not in live:
                diffs.append(f"{p}: present in snapshot, absent in experiment config")
            elif snap[k] != live[k]:
                diffs += _first_diff(snap[k], live[k], p)
        return diffs
    return [f"{prefix or '<root>'}: {live!r} (experiment) vs {snap!r} (snapshot)"]


def _without_toggle(section, path):
    """Drop `enabled` from a `workflows.<name>` section (exception 1).

    Applied to BOTH operands, and only to a workflow stanza -- a top-level
    section that happens to carry a key called `enabled` is untouched.
    """
    if not isinstance(section, dict):
        return section
    if not (isinstance(path, tuple) and len(path) == 2 and path[0] == "workflows"):
        return section
    return {k: v for k, v in section.items() if k != _STANZA_TOGGLE}


def _compare_section(live_cfg, snapshot, path) -> list[str]:
    """Section-scoped deep structural equality with symmetric normalization.

    Deep equality over the whole section, rather than key-by-key over the
    snapshot's keys: the union diff in :func:`_first_diff` is what catches a key
    the LIVE config added, which iterating the snapshot alone would miss.
    """
    live_section = _without_toggle(_normalize_paths(_get_section(live_cfg, path)), path)
    snap_section = _without_toggle(_normalize_paths(_get_section(snapshot, path)), path)
    if live_section == snap_section:
        return []
    label = _section_label(path)
    return _first_diff(snap_section, live_section, label)


def compare_project_consistency(
    live_cfg: dict,
    wf1_snapshot_path,
    wf2_snapshot_path=None,
) -> list[str]:
    """Compare an experiment config's project sections vs the project snapshots.

    PURE function — no Snakemake, no filesystem writes. Returns a list of
    human-readable divergence messages (empty ⇒ pass). Gate 2 (a–h) calls this
    directly on staged config/snapshot pairs.

    - The wf1 snapshot is MANDATORY: when missing, a single "run
      build_model.smk first" message is returned (the friendlier
      duplicate of the rule-level ``MissingInputException``, design §3b).
    - ``project``, ``shared.basin``, ``workflows.build_model`` are compared
      against the wf1 snapshot.
    - ``workflows.analyze_projections`` is compared against the wf2 snapshot
      ONLY when it exists; when absent (wf2 never ran) that section is logged
      unchecked and passes — it does not fall back to the wf1 copy.
    """
    wf1_path = Path(wf1_snapshot_path)
    if not wf1_path.is_file():
        return [f"No project snapshot at {wf1_path}; run build_model.smk first."]
    wf1_snapshot = yaml.safe_load(wf1_path.read_text(encoding="utf-8"))

    diffs: list[str] = []
    wf1_paths = guarded_paths(wf1_snapshot)
    for path in wf1_paths:
        diffs += _compare_section(live_cfg, wf1_snapshot, path)

    wf2_path = Path(wf2_snapshot_path) if wf2_snapshot_path is not None else None
    if wf2_path is not None and wf2_path.is_file():
        wf2_snapshot = yaml.safe_load(wf2_path.read_text(encoding="utf-8"))
        # Only what WF1 could not witness. Every T1 section appears in both
        # snapshots with the same value, so comparing them twice would report
        # one divergence as two -- and the WF1 snapshot is the mandatory one.
        seen = set(wf1_paths)
        for path in guarded_paths(wf2_snapshot):
            if path not in seen:
                diffs += _compare_section(live_cfg, wf2_snapshot, path)
    else:
        log_row(
            "wf2 snapshot absent; workflows.analyze_projections unchecked (passes)",
            module="guard",
        )
    return diffs


def check_project_consistency(
    live_cfg: dict,
    wf1_snapshot_path,
    sentinel_path,
    guard_ok_path,
    wf2_snapshot_path=None,
) -> None:
    """Run the guard; on pass write both guard artifacts, on divergence raise.

    A failing guard writes NEITHER artifact (and Snakemake removes a failed
    job's outputs), so both consumer classes — the four per-experiment roots
    (fresh sentinel) and ``extract_climate_datasets`` (``ancient()`` guard
    artifact) — stay blocked on failure (design §3a/§3b).
    """
    diffs = compare_project_consistency(live_cfg, wf1_snapshot_path, wf2_snapshot_path)
    if diffs:
        detail = "\n  - ".join(diffs)
        raise ValueError(
            "Experiment config diverges from the project snapshot(s); wf3 "
            "refuses to run against a model whose provenance no longer "
            "matches. Revert the experiment's project sections or re-run the "
            "owning workflow (wf1/wf2).\n  - " + detail
        )
    log_row("Project consistency OK.", module="guard")
    for out_path in (sentinel_path, guard_ok_path):
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("project consistency check passed\n", encoding="utf-8")


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import tee_to_log

        with tee_to_log(sm.log[0]):
            check_project_consistency(
                live_cfg=sm.config,
                wf1_snapshot_path=sm.input.wf1_snapshot,
                sentinel_path=sm.output.sentinel,
                guard_ok_path=sm.output.guard_ok,
                wf2_snapshot_path=sm.params.wf2_snapshot_path,
            )
