"""Spherical cell-area weights from midpoint edges (design D10, step 5a).

The spatial reduction was an unweighted ``mean([x_dim, y_dim])``, which treats a
cell near 60°N as equal in area to one at the equator. design-v2/v3 proposed
cos-latitude weighting behind a "1-D and strictly monotonic" geometry check;
round-2 review (ext2-02) faulted that pair, and the fault is worth restating
because it shapes this module:

    cos(lat) is a valid area weight only for *uniformly spaced* rectilinear
    grids, and "1-D + strictly monotonic" does not test spacing. A Gaussian grid
    passes the check and receives wrong weights — silently.

Revision 4 does not strengthen the check to match the scheme. It changes the
scheme so that **its validity condition is exactly the condition the check
tests**: each cell is weighted by the true spherical area implied by its edges,

    w = (sin φ_north − sin φ_south) × Δλ

with edges derived from adjacent-center midpoints. Midpoint derivation consumes
exactly one property — ordered, distinct 1-D centers — which is precisely what the
check establishes. No spacing assumption remains.

Two consequences worth knowing before editing:

* **This is a strict generalization, not a competitor.** On any uniformly spaced
  grid these weights are *exactly* proportional to cos φ, because
  ``sin(φ+d/2) − sin(φ−d/2) = 2·sin(d/2)·cos φ`` and the constant factor cancels in
  a weighted mean. Anything that breaks that equivalence is a bug in the edges, not
  a modelling choice (falsifier F1).
* **The residual approximation is a different, smaller one.** True cell edges are
  not always adjacent-center midpoints — a Gaussian grid's conventional edges
  differ slightly. Exact edges are unavailable here: the generated catalog sets
  ``drop_variables: [time_bnds, lat_bnds, lon_bnds, bnds]`` on every CMIP6 entry.
  So the residual is "true edges vs midpoints", not "cos φ vs area".

Falsifiers for every claim above: ``dev/working/2026-07-30_wf2-5a-falsifier.md``.
"""
from __future__ import annotations

import numpy as np

#: Recorded on every series reduced with these weights (design §5.3).
WEIGHTING_SCHEME = "spherical_cell_area_midpoint_edges"

#: What the pre-5a unweighted reduction recorded. Kept so the transition is
#: greppable and a stale label is detectable rather than merely absent.
WEIGHTING_SCHEME_PRE_5A = "unweighted_mean_pre_5a"


class GridGeometryError(ValueError):
    """A grid this reduction cannot weight without silently biasing the result.

    Raised for 2-D/curvilinear coordinates and non-monotonic axes only. A
    non-uniformly spaced 1-D axis — Gaussian latitudes among them — is **not** an
    error: handling those correctly is the entire point of D10 (falsifier F4, the
    over-refusal direction).
    """


def check_axis(values, name, source=""):
    """Return ``values`` as a 1-D float array, or raise naming the source.

    The exact condition, and nothing more: 1-D, finite, strictly monotonic.
    """
    where = f" ({source})" if source else ""
    arr = np.asarray(values, dtype="float64")
    if arr.ndim != 1:
        raise GridGeometryError(
            f"{name} is {arr.ndim}-D{where}: 2-D or curvilinear coordinates cannot "
            "be weighted by midpoint-edge cell areas. Refused rather than "
            "silently mis-weighted (design D10)."
        )
    if not np.all(np.isfinite(arr)):
        raise GridGeometryError(f"{name} contains non-finite values{where}.")
    if arr.size > 1:
        step = np.diff(arr)
        if not (np.all(step > 0) or np.all(step < 0)):
            raise GridGeometryError(
                f"{name} is not strictly monotonic{where}. A repeated or "
                "non-ordered axis — a dateline-wrapped longitude subset is the "
                "common case — has no midpoint edges. Refused rather than "
                "silently mis-weighted (design D10)."
            )
    return arr


def midpoint_edges(centers):
    """Cell edges from 1-D centers: interior midpoints, boundaries extrapolated.

    The boundary edges are ``center ± half the adjacent spacing``, i.e. the
    symmetric extrapolation the design specifies. This is the part worth testing
    against an independent quantity (falsifier F3: over a grid partitioning the
    sphere the resulting areas must sum to the sphere), because an interior-only
    check cannot see a boundary mistake.
    """
    centers = np.asarray(centers, dtype="float64")
    if centers.size == 1:
        raise ValueError("a length-1 axis has no midpoint edges; see cell_widths")
    interior = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - 0.5 * (centers[1] - centers[0])
    last = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return np.concatenate(([first], interior, [last]))


def latitude_weights(lat, name="latitude", source=""):
    """``sin φ_north − sin φ_south`` per cell, always positive.

    Edges are clamped to ``[-90, 90]``: a symmetric extrapolation on a
    pole-adjacent grid can place an edge beyond the pole, where ``sin`` turns back
    on itself and would yield a negative — i.e. nonsensical — cell area. Clamping
    is a no-op on every grid whose extrapolated edges already land inside the
    sphere, which includes every uniform global grid.
    """
    arr = check_axis(lat, name, source)
    if arr.size == 1:
        return np.ones(1)
    edges = np.clip(midpoint_edges(arr), -90.0, 90.0)
    return np.abs(np.diff(np.sin(np.deg2rad(edges))))


def longitude_weights(lon, name="longitude", source=""):
    """Cell width Δλ in radians, always positive."""
    arr = check_axis(lon, name, source)
    if arr.size == 1:
        return np.ones(1)
    return np.abs(np.diff(np.deg2rad(midpoint_edges(arr))))


def cell_area_weights(lat, lon, source=""):
    """Spherical cell areas (steradians) as a ``(lat, lon)`` array.

    A length-1 axis contributes the degenerate weight 1 rather than 0 or NaN —
    not an edge case here but the ordinary small-basin path, since at ``Amon``
    resolution (~1–2°) a buffered catchment bbox can select a single cell
    (falsifier F5).

    Separable by construction (``w = Δsinφ ⊗ Δλ``), so the outer product is exact
    rather than an approximation of a 2-D integral.
    """
    return np.outer(
        latitude_weights(lat, source=source),
        longitude_weights(lon, source=source),
    )


def geometry_check_label(lat, lon, source=""):
    """The geometry-check result, for ``cst_geometry_check`` on the series.

    Recorded because "which grid was accepted, and on what basis" is not
    recoverable from the numbers afterwards. Raises for a refused grid, so a
    series can only ever carry a passing label.
    """
    lat_arr = check_axis(lat, "latitude", source)
    lon_arr = check_axis(lon, "longitude", source)
    return f"1d_strictly_monotonic; lat={lat_arr.size} lon={lon_arr.size}"


def weighted_spatial_mean(da, x_dim, y_dim, source=""):
    """Area-weighted basin mean over the two spatial dims of ``da``.

    Uses xarray's ``.weighted()`` rather than a hand-rolled ``(da*w).sum()/w.sum()``
    for one reason that matters: it renormalises over the **valid** cells, so a
    masked or NaN cell drops out of both numerator and denominator. The hand-rolled
    form silently biases toward zero wherever the field has gaps.

    On a grid whose weights happen to be equal — the equatorial, latitude-symmetric
    case of this repo's fixture — this returns the unweighted mean exactly, which
    is the property step 5a's tree diff asserts (falsifier F6).
    """
    import xarray as xr

    weights = xr.DataArray(
        cell_area_weights(da[y_dim].values, da[x_dim].values, source=source),
        dims=(y_dim, x_dim),
        coords={y_dim: da[y_dim], x_dim: da[x_dim]},
    )
    reduced = da.weighted(weights).mean((y_dim, x_dim))

    # Preserve the input dtype. The weights are float64, so `.weighted()` upcasts
    # a float32 field and the series changes dtype -- which the step-5a tree diff
    # caught as 13 failing files reading `dtype float64 vs float32` and
    # `2.38 vs 2.380000114440918`. Those are the same number at two precisions,
    # not a weighting effect, but a silent schema change is exactly what a
    # value-neutral-on-this-fixture step must not smuggle in. Whether the series
    # should be float64 at all is a precision question, and it belongs to step 5c
    # (which drops the 2-decimal rounding), not here.
    return reduced.astype(da.dtype) if reduced.dtype != da.dtype else reduced
