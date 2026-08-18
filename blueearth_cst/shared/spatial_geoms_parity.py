"""Relationship contract between our spatial foundation and wflow's staticgeoms.

A built project carries two directories of GeoJSON with SIX COLLIDING
BASENAMES, and the collision is not drift -- it is misidentification. Measured
on ``test_case/test_local`` (2026-08-05, re-measured 2026-08-18), none of the
six pairs holds the same content:

``region``
    ours is the delineated basin; hydromt's is the model GRID EXTENT, a box.
``basins``
    ours is the basin as one polygon; hydromt's is its own per-subbasin
    decomposition. Union areas are identical, feature counts are not.
``rivers``
    different provenance -- ours carries 21 MERIT-derived attributes,
    hydromt's carries routing topology.
``subbasins`` / ``catchments`` / ``locations``
    true copies: identical geometry AND identical schema, ours included.

So ``basins.geojson`` means "the basin" in one tree and "the per-subbasin
polygons" in the other, with nothing on disk saying which is which. A future
rule, the GUI, or a later reader takes the wrong one and is SILENTLY wrong --
the output is plausible, not obviously broken. That is the exposure, and it is
why this module asserts a RELATIONSHIP per layer rather than equality.

**Why not** ``semantic_tree_diff.compare_geojson``. That function is the right
primitive for "did this file change between two trees", and this module reuses
its ideas -- CRS, feature count, non-geometry columns, topological geometry --
but it asserts EQUALITY, which is false for three of the six pairs BY DESIGN.
It also lives in ``dev/scripts/``, which is never part of a run.

**Why a relationship rather than a rerun rule.** Temporal drift is already
structurally impossible: ``data/spatial/`` is upstream of the model build, and
ADR 0004 makes rule 1.10 the terminal writer of the whole model root, so a
partially-failed build cannot leave one tree ahead of the other. What is NOT
guaranteed is that hydromt keeps deriving its ``region`` the way it does today
-- ``GridComponent._region_data`` returns ``box(*self.bounds)``
(``hydromt/model/components/grid.py:269``, hydromt 1.3.1), which is exactly
what makes ``ours <= theirs`` true. A hydromt upgrade that changed that would
break the containment with nothing watching. This module is the watcher.

Board item ``t2608071203`` (R9-1). The authority question -- which tree answers
which question -- is documented in
``dev/reference/contracts/hydrological-model-seam.md``.

Design invariants, mirroring ``interchange_contracts``:

1. Pure functions over PARSED objects (``geopandas.GeoDataFrame``), never
   paths. The caller owns all file I/O, so one function serves a synthetic
   unit test, a real-fixture test, and a future in-pipeline guard with no move.
2. ``-> list[str]`` divergence report; empty list means pass. Every violation
   is surfaced at once.
3. No ``assert`` in validator bodies: ``assert`` is stripped under ``python
   -O``, which would make an optimized in-pipeline guard fail OPEN.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Layers whose two versions are TRUE COPIES: identical geometry and identical
#: non-geometry schema, ours included (`delineation_method`, `subbasin_code`).
#: hydromt round-trips what we hand it, so a divergence here means the model
#: build stopped preserving our foundation.
COPIED_LAYERS: tuple[str, ...] = ("subbasins", "catchments", "locations")

#: Ours must be CONTAINED in hydromt's, not equal to it: theirs is the grid
#: extent, built to circumscribe the delineated basin. Equality would itself be
#: the surprise -- it would mean the grid stopped being a bounding box.
CONTAINED_LAYERS: tuple[str, ...] = ("region",)

#: Layers that share a basename and nothing else. Listed so the contract STATES
#: them rather than omitting them, which would read as "not yet checked".
#: Their reasons are the whole point of the module.
INCOMPARABLE_LAYERS: dict[str, str] = {
    "basins": (
        "ours is the delineated basin as one polygon; hydromt's is its own "
        "per-subbasin decomposition, keyed by a `value` column"
    ),
    "rivers": (
        "different provenance -- ours carries MERIT attributes, hydromt's "
        "carries routing topology (idx, idx_ds, pit, strord)"
    ),
}

#: Every basename that exists in both trees.
SHARED_BASENAMES: tuple[str, ...] = (
    *CONTAINED_LAYERS,
    *sorted(INCOMPARABLE_LAYERS),
    *COPIED_LAYERS,
)

#: Fractional slack on the containment test, as a share of OUR area. Float
#: coordinates round-tripping through GeoJSON leave a sliver on the boundary:
#: measured 1.4e-08 absolute against an area of 1.78e-02, i.e. ~8e-07 relative.
#: This sits three orders above that and still far below any real regression,
#: which would be a whole grid cell.
CONTAINMENT_RTOL = 1e-4

#: Vertex tolerance in DEGREES for the copied layers, applied as a Hausdorff
#: distance. Exact equality is the wrong test here and quietly so: shapely's
#: ``equals`` is topological but not tolerant, and a copy that round-trips
#: through GeoJSON comes back displaced by float formatting. Measured on
#: ``test_case/test_local`` (2026-08-18) the displacement is 4.7e-07 deg for
#: every feature of all three layers -- Polygon and Point alike, which is why
#: this is a distance rather than an area (``locations`` is points, whose
#: symmetric-difference area is 0 however far apart they are).
#:
#: 1e-5 is ~20x the measured noise and ~0.1% of the 0.00833 deg grid cell, so a
#: real divergence -- at minimum one cell -- is caught with three orders of
#: margin. ``equals_exact`` was rejected: it needs matching vertex STRUCTURE and
#: returns False for the polygons here even at 1e-6, because the ring start
#: point moves.
GEOMETRY_ATOL_DEG = 1e-5


def _union_area(gdf: Any) -> float:
    """Total area of a GeoDataFrame's geometry, as one unioned shape."""
    return float(gdf.geometry.union_all().area)


def _columns(gdf: Any) -> list[str]:
    return [c for c in gdf.columns if c != "geometry"]


def validate_copied_layer(ours: Any, theirs: Any, layer: str) -> list[str]:
    """Both sides must be the same layer: same schema, same geometry."""
    out: list[str] = []
    if str(ours.crs) != str(theirs.crs):
        out.append(f"{layer}: crs {ours.crs} vs {theirs.crs}")
    if len(ours) != len(theirs):
        out.append(f"{layer}: feature count {len(ours)} vs {len(theirs)}")
        return out
    our_cols, their_cols = _columns(ours), _columns(theirs)
    if our_cols != their_cols:
        out.append(f"{layer}: columns {our_cols} vs {their_cols}")
    else:
        for col in our_cols:
            if not ours[col].equals(theirs[col]):
                out.append(f"{layer}: column {col!r} values differ")
    for i, (a, b) in enumerate(zip(ours.geometry, theirs.geometry)):
        if a is None or b is None:
            if a is not b:
                out.append(f"{layer}: feature {i} geometry present on one side only")
            continue
        if a.equals(b):
            continue
        # Not exactly equal: fall through to the tolerant test rather than
        # reporting, because a GeoJSON round trip alone lands here.
        displacement = a.hausdorff_distance(b)
        if displacement > GEOMETRY_ATOL_DEG:
            out.append(
                f"{layer}: feature {i} geometry differs -- hausdorff "
                f"{displacement:.6g} deg exceeds {GEOMETRY_ATOL_DEG:g} "
                f"(symmetric difference area "
                f"{a.symmetric_difference(b).area:.6g})"
            )
    return out


def validate_contained_layer(ours: Any, theirs: Any, layer: str) -> list[str]:
    """Ours must lie inside theirs; report HOW MUCH escaped, not merely that it did.

    A boundary sliver reads differently from a grid that stopped circumscribing
    the basin, and the caller can only tell them apart if the number is in the
    message.
    """
    out: list[str] = []
    if str(ours.crs) != str(theirs.crs):
        out.append(f"{layer}: crs {ours.crs} vs {theirs.crs}")
    our_area = _union_area(ours)
    if our_area <= 0:
        out.append(f"{layer}: ours has zero area, so containment is vacuous")
        return out
    leaked = float(
        ours.geometry.union_all().difference(theirs.geometry.union_all()).area
    )
    if leaked > our_area * CONTAINMENT_RTOL:
        out.append(
            f"{layer}: ours is NOT contained in the model's -- {leaked:.6g} of "
            f"{our_area:.6g} lies outside ({leaked / our_area:.2%}). hydromt "
            f"derives this layer as the grid bounding box, so a break means "
            f"that box stopped circumscribing the delineated basin"
        )
    return out


def validate_spatial_geoms_parity(
    ours: Mapping[str, Any], theirs: Mapping[str, Any]
) -> list[str]:
    """Check every shared basename against its declared relationship.

    Args:
        ours: layer name -> GeoDataFrame, read from `data/spatial/geoms/`.
        theirs: layer name -> GeoDataFrame, read from the model's
            `staticgeoms/`.

    Returns:
        Divergence strings; empty means every relationship holds. A layer
        absent from either mapping is REPORTED rather than skipped, so a
        caller cannot pass an empty mapping and read the silence as a pass.
    """
    out: list[str] = []
    for layer in SHARED_BASENAMES:
        if layer not in ours:
            out.append(f"{layer}: absent from the spatial foundation")
        if layer not in theirs:
            out.append(f"{layer}: absent from the model's staticgeoms")
    if out:
        return out

    for layer in CONTAINED_LAYERS:
        out.extend(validate_contained_layer(ours[layer], theirs[layer], layer))
    for layer in COPIED_LAYERS:
        out.extend(validate_copied_layer(ours[layer], theirs[layer], layer))
    # INCOMPARABLE_LAYERS are deliberately NOT compared. They are in
    # SHARED_BASENAMES so their presence is still checked and their reasons
    # stay readable here, which is the documentation half of this contract.
    return out
