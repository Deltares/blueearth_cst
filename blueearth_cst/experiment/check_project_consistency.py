"""wf3 startup drift guard: experiment config vs project snapshots.

An experiment config is a *full* config (approach A). Its project-level
sections (``project``, ``shared.basin``, ``workflows.model_creation``,
``workflows.climate_projections``) must describe the same project the built
model / overlay came from — otherwise the experiment silently reuses a
``models/hydrology/wflow/`` built under different settings. This rule (3.00b
``check_project_consistency``) runs at wf3 rule time and fails loud on
divergence, naming the diverging key.

Design: dev/milestones/p31/experiment-structure-design.md §3/§3a/§3b/§3d.

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
from blueearth_cst.shared.snake_utils import log_row  # noqa: E402

# The four project-level sections the guard compares (design §3b "exact guarded
# key set"). Each is compared against the snapshot of the workflow that OWNS it;
# ``workflows.climate_projections`` is compared against the wf2 snapshot only
# when that snapshot exists (else unchecked + logged — the projections overlay
# is optional per the CST method and must not be force-required).
_WF1_GUARDED = ("project", ("shared", "basin"), ("workflows", "model_creation"))
_WF2_GUARDED = (("workflows", "climate_projections"),)

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
    "data_sources": {
        "config\\deltares_data.yml": "config/catalogs/deltares_data.yml",
        "config\\deltares_data_linux.yml": "config/catalogs/deltares_data_linux.yml",
        "config\\deltares_data_climate_projections.yml": "config/catalogs/deltares_data_climate_projections.yml",
        "config\\deltares_data_climate_projections_linux.yml": "config/catalogs/deltares_data_climate_projections_linux.yml",
        "config\\cmip6_data.yml": "config/catalogs/cmip6_data.yml",
    },
    "data_sources_climate": {
        "config\\cmip6_data.yml": "config/catalogs/cmip6_data.yml",
    },
    # TWO old spellings each, because these files have moved twice: pre-R6 flat,
    # then config/templates/, then config/defaults/ (2026-08-11, when templates/
    # was split so it holds only files you copy). Both must normalize onto the
    # current path or a config from either era fails against a fresh snapshot.
    "model_build_config": {
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


def _compare_section(live_cfg, snapshot, path) -> list[str]:
    """Section-scoped deep structural equality with symmetric normalization."""
    live_section = _normalize_paths(_get_section(live_cfg, path))
    snap_section = _normalize_paths(_get_section(snapshot, path))
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
      Snakefile_model_creation first" message is returned (the friendlier
      duplicate of the rule-level ``MissingInputException``, design §3b).
    - ``project``, ``shared.basin``, ``workflows.model_creation`` are compared
      against the wf1 snapshot.
    - ``workflows.climate_projections`` is compared against the wf2 snapshot
      ONLY when it exists; when absent (wf2 never ran) that section is logged
      unchecked and passes — it does not fall back to the wf1 copy.
    """
    wf1_path = Path(wf1_snapshot_path)
    if not wf1_path.is_file():
        return [
            f"No project snapshot at {wf1_path}; run Snakefile_model_creation first."
        ]
    wf1_snapshot = yaml.safe_load(wf1_path.read_text(encoding="utf-8"))

    diffs: list[str] = []
    for path in _WF1_GUARDED:
        diffs += _compare_section(live_cfg, wf1_snapshot, path)

    wf2_path = Path(wf2_snapshot_path) if wf2_snapshot_path is not None else None
    if wf2_path is not None and wf2_path.is_file():
        wf2_snapshot = yaml.safe_load(wf2_path.read_text(encoding="utf-8"))
        for path in _WF2_GUARDED:
            diffs += _compare_section(live_cfg, wf2_snapshot, path)
    else:
        log_row(
            "wf2 snapshot absent; workflows.climate_projections unchecked (passes)",
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
    (fresh sentinel) and ``extract_climate_grid`` (``ancient()`` guard
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
