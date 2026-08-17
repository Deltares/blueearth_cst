"""Stage the wf1 leaves WF2/WF3 declare and Snakemake will not satisfy on its own.

A CROSS-WORKFLOW LEAF is a file some rule in WF2 or WF3 declares as an input
while no rule in that workflow declares it as an output. Snakemake cannot
produce it, so anything driving WF2/WF3 in isolation — a dry-run test, the
layout scaffolder — has to put it on disk first. This module puts it there.

**The leaf LIST is not defined here.** It moved to
`blueearth_cst/shared/cross_workflow_leaves.py` on 2026-08-17, when
`scripts/run_workflows.py` needed it for a preflight: that is a run path, and
AGENTS.md's invocation-model split makes `dev/scripts/` never part of a run. The
names are re-exported below, so every consumer of this module is unaffected —
read the shared module for why the list is one definition rather than three, and
for the test that proves it complete and minimal.

**What stayed here is the STAGING**, which is test-fixture machinery with no
run-path caller: `stage`, `content_for`, the minimal file bodies, and the
`EXTRA_*` non-leaves below.

**Extras are not leaves.** Both test fixtures also stage files the DAG does not
require, for reasons that are real but separate — see `EXTRA_*` below. They are
passed explicitly so that "required by Snakemake" and "wanted by this caller"
stay legible; folding them into `LEAVES` is how the vestigial ones survived.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Re-exported so this module's own consumers (two test fixtures and
# `scaffold_project_tree.py`) keep importing the leaf set from the stager they
# already use. The definition is in `shared/`; this is a view onto it.
from blueearth_cst.shared.cross_workflow_leaves import (  # noqa: E402
    LEAF_MODEL_READY,
    LEAF_MODEL_TOML,
    LEAF_PRODUCER,
    LEAF_WF1_SNAPSHOT,
    LEAVES,
)

__all__ = [
    "EXTRA_REGION",
    "EXTRA_WF2_SNAPSHOT",
    "LEAF_MODEL_READY",
    "LEAF_MODEL_TOML",
    "LEAF_PRODUCER",
    "LEAF_WF1_SNAPSHOT",
    "LEAVES",
    "MINIMAL_REGION_GEOJSON",
    "MINIMAL_WFLOW_TOML",
    "content_for",
    "stage",
]

# --- Deliberate NON-leaves -------------------------------------------------
# Staged by some callers, required by no DAG. Each is here with its reason so
# the next reader does not have to decide whether it is drift.

#: NOT a declared input anywhere: WF3's drift guard reads it as a `params` path
#: and existence-checks it in the script, because the projections overlay is
#: optional and must not be force-required. `test_guard_invalidation` stages it
#: because its assertions are about the guard's COMPARISON, not the DAG.
EXTRA_WF2_SNAPSHOT = "config/runs/snake_config_analyze_projections.yml"

#: NOT read by either downstream workflow. R07 B1 retired the extraction's
#: `ancient(region.geojson)` input and ADR 0003 gave WF2 and WF3 their own
#: `delineate_region`, each declaring `data/spatial/geoms/region.geojson` as an
#: OUTPUT. Staged by the two test fixtures only to keep the scratch project
#: looking like a completed wf1 run.
EXTRA_REGION = "models/hydrology/wflow/staticgeoms/region.geojson"

#: Enough of a wflow TOML for rule 3.01c to read `input.path_static`.
MINIMAL_WFLOW_TOML = '[input]\npath_static = "staticmaps.nc"\n'

#: Minimal valid polygon, for callers that stage `EXTRA_REGION`.
MINIMAL_REGION_GEOJSON = """{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature", "properties": {"value": 1},
    "geometry": {"type": "Polygon", "coordinates": [[
      [11.3, -1.05], [13.6, -1.05], [13.6, 0.9], [11.3, 0.9], [11.3, -1.05]]]}
  }]
}
"""

#: Paths whose content must be the run's own config, so a snapshot-comparing
#: guard sees identical comparands and passes by construction.
_CONFIG_SNAPSHOTS = frozenset({LEAF_WF1_SNAPSHOT, EXTRA_WF2_SNAPSHOT})


def content_for(rel: str, config_text: str) -> str:
    """Return the file content to stage at project-relative path ``rel``."""
    if rel in _CONFIG_SNAPSHOTS:
        return config_text
    if rel.endswith("wflow_sbm.toml"):
        return MINIMAL_WFLOW_TOML
    if rel.endswith("region.geojson"):
        return MINIMAL_REGION_GEOJSON
    return ""


def stage(
    project_dir: Path,
    config_text: str,
    extras: Sequence[str] = (),
    leaves: Iterable[str] | None = None,
) -> tuple[Path, ...]:
    """Materialize the cross-workflow leaves (plus ``extras``) under ``project_dir``.

    Args:
        project_dir: Scratch project root. Created if absent.
        config_text: Serialized config to write at any snapshot path. Serialize
            it from the SAME parsed config the run consumes, or a snapshot-
            comparing guard will fail on a difference the caller invented.
        extras: Deliberate non-leaves — pass the ``EXTRA_*`` constants.
        leaves: Override the leaf set. For the minimality proof only; callers
            staging a project should leave it ``None``.

    Returns:
        The staged paths, in the order written.
    """
    staged: list[Path] = []
    for rel in (*(LEAVES if leaves is None else leaves), *extras):
        target = Path(project_dir) / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content_for(rel, config_text), encoding="utf-8")
        staged.append(target)
    return tuple(staged)
