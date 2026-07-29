"""P3-2a acceptance-gate comparison ladder (design §5.5, ext1-3).

.. warning::

   **SUPERSEDED AND NON-RUNNABLE SINCE R07.** This probe opens
   ``climate_historical/wf1_raw/``, which R07's B1 retired: the two climate
   stores were collapsed into one, so S1 below no longer exists on disk and
   this module now raises ``FileNotFoundError``.

   It is kept as the P3-2a milestone record, not as a tool. Deliberately NOT
   repointed at the surviving store -- doing so would silently change what it
   computes: the ladder's whole purpose was to characterise the difference
   between ``wf1_raw`` and the keyed store, and R07 proved those two
   element-wise identical via the declared merge comparison
   (``dev/r07/migration_project-layout.md`` §2e). The question this probe was
   built to answer has been answered; re-pointing it would leave it comparing a
   store against itself.


Computes the named, persisted ladder states and the component-decomposed
characterized diff for the wf1 subcatchment-climate re-source — the
milestone's single sanctioned value change:

- S0 = ``inmaps_historical.nc`` (rule 1.08): the old plot path's grid.
- S1 = ``climate_historical/wf1_raw/extract_historical.nc``: the raw
  branch-native extraction (not a ladder rung reduced by R — native-grid data
  cannot be grouped by the model-grid zone raster).
- S2 = regrid-only parity (corrections OFF)  -> ``<qa>/l1_regrid_only.nc``.
- S3 = full model parity (corrections ON)    -> ``<qa>/l2_parity.nc``.

R(.) = ``climate_forcing_by_subcatchment(., subcatchment)`` followed by a
per-subcatchment monthly climatology (12 months x index x {P, T, EP}).
A0 = R(S0), A1 = R(S2), A2 = R(S3).

Reported edges (every residual is assigned to a NAMED edge — the §5.5
attribution rule):

- ``A2 - A1``  correction component (temp lapse + pressure-through-PET).
- ``A2 - A1 (P)`` precip null-check — asserted EXACTLY zero.
- ``A2 - A0``  the end-to-end sanctioned change the user signs off.
- ``G``        grid-level parity check: S3 vs S0 on the model grid within the
  basin mask; era5 tolerances temp <= 0.05 degC, precip/pet <= 0.05 mm/d.
- ``wf1_raw vs keyed store`` allclose closure (ext1-5): both are outputs of
  the same function (staticmaps-bbox vs region-bbox), so equality proves the
  bbox swap changed nothing.

Also composes old/new side-by-side ``clim_*`` PNGs when ``--old-plots`` is
given. Exit 0 = all gates pass; 1 = a gate failed (details on stdout).

CLI (dev-side; no snakemake)::

    pixi run python dev/p32a/compare_climate_ladder.py \
        [--config config/workflows/snake_config_model_test.yml] \
        [--qa-dir dev/p32a/qa] [--old-plots <dir>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import hydromt  # noqa: E402  (registers the .raster accessor)
from blueearth_cst.climate_analysis.subcatchment_climate import (  # noqa: E402
    climate_forcing_by_subcatchment,
)
from blueearth_cst.model.climate_parity import model_parity_climate  # noqa: E402

# era5-branch grid-level parity tolerances (design §5.2: float32 +
# domain-clip headroom; the parity path and the build run the same functions
# on the same sources, so G residuals beyond these red the gate).
TOL = {"temp": 0.05, "precip": 0.05, "pet": 0.05}
VAR_LABEL = {"precip": "P", "temp": "T", "pet": "EP"}


def _reduce(ds, subcatchment):
    """R(.): per-subcatchment mean -> per-subcatchment monthly climatology."""
    agg = climate_forcing_by_subcatchment(ds, subcatchment)
    return agg.groupby("time.month").mean("time")


def _stats(da):
    """(mean, max-abs, rmse) over every element of a delta array."""
    v = np.asarray(da).ravel()
    v = v[np.isfinite(v)]
    return float(v.mean()), float(np.abs(v).max()), float(np.sqrt((v**2).mean()))


def _edge_rows(name, delta):
    """Per-variable summary rows for one named ladder edge."""
    rows = []
    for var in delta.data_vars:
        mean, maxabs, rmse = _stats(delta[var])
        rows.append((name, str(var), mean, maxabs, rmse))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--config", default=str(REPO / "config/workflows/snake_config_model_test.yml")
    )
    ap.add_argument("--qa-dir", default=str(REPO / "dev/p32a/qa"))
    ap.add_argument(
        "--old-plots",
        default=None,
        help="directory holding the OLD clim_*.png plots for side-by-side composites",
    )
    args = ap.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    project_dir = REPO / cfg["project"]["project_dir"]
    data_sources = str(REPO / cfg["project"]["data_sources"])
    clim_source = cfg["shared"]["clim_historical"]
    start = cfg["shared"]["historical_window"]["starttime"][:10].replace("-", "")
    end = cfg["shared"]["historical_window"]["endtime"][:10].replace("-", "")
    store_key = f"{clim_source}_{start}_{end}"

    qa = Path(args.qa_dir)
    qa.mkdir(parents=True, exist_ok=True)

    # --- Load states ------------------------------------------------------
    sm = xr.open_dataset(project_dir / "hydrology_model/staticmaps.nc")
    subcatchment = sm["subcatchment"]
    dem_model = sm["land_elevation"]
    mask = subcatchment != subcatchment.attrs.get("_FillValue", 0)

    s0 = xr.open_dataset(
        project_dir / "climate_historical/wflow_data/inmaps_historical.nc"
    )
    s1 = xr.open_dataset(project_dir / "climate_historical/wf1_raw/extract_historical.nc")

    if clim_source in ("chirps", "chirps_global"):
        dem_forcing = xr.open_dataarray(
            project_dir / "climate_historical/wf1_raw/orography.nc"
        )
    else:
        catalog = hydromt.DataCatalog(data_libs=data_sources)
        dem_forcing = catalog.get_rasterdataset(
            "era5_orography", geom=s1.raster.box, buffer=2, variables=["elevtn"]
        ).squeeze()

    # --- S2 / S3 (persisted) ---------------------------------------------
    s2 = model_parity_climate(s1, dem_model, dem_forcing, "debruin", corrections=False)
    s2 = s2.compute()
    s2.to_netcdf(qa / "l1_regrid_only.nc", mode="w")
    s3 = model_parity_climate(s1, dem_model, dem_forcing, "debruin", corrections=True)
    s3 = s3.compute()
    s3.to_netcdf(qa / "l2_parity.nc", mode="w")

    # --- Aggregated states ------------------------------------------------
    a0 = _reduce(s0, subcatchment)
    a1 = _reduce(s2, subcatchment)
    a2 = _reduce(s3, subcatchment)

    rows = []
    rows += _edge_rows("A2-A1 (correction component)", a2 - a1)
    rows += _edge_rows("A2-A0 (sanctioned change)", a2 - a0)

    failures = []

    # Precip null-check: EXACT zero (no correction touches precip on any
    # branch; S2 and S3 precip run the identical call).
    p_null = float(np.abs((a2 - a1)["P_subcatchment"]).max())
    if p_null != 0.0:
        failures.append(
            f"precip null-check FAILED on edge A2-A1: max-abs {p_null!r} != 0 "
            "(a pipeline bug, not a finding)"
        )

    # --- G: grid-level parity vs the build's own forcing ------------------
    g_rows = []
    times = np.intersect1d(s3["time"].values, s0["time"].values)
    for var in ("precip", "temp", "pet"):
        d = (
            s3[var].sel(time=times).where(mask)
            - s0[var].sel(time=times).where(mask)
        )
        mean, maxabs, rmse = _stats(d)
        g_rows.append(("G (S3-S0 grid, masked)", var, mean, maxabs, rmse))
        if maxabs > TOL[var]:
            failures.append(
                f"G exceeds tolerance for {var}: max-abs {maxabs:.4g} > {TOL[var]}"
            )
    rows += g_rows

    # --- wf1_raw vs keyed store closure (ext1-5) --------------------------
    key_path = project_dir / "climate_historical" / store_key / "extract_historical.nc"
    if key_path.exists():
        k = xr.open_dataset(key_path)
        for var in s1.data_vars:
            same_nan = bool(
                np.array_equal(
                    np.isnan(s1[var].values), np.isnan(k[var].values)
                )
            )
            close = bool(
                np.allclose(s1[var].values, k[var].values, equal_nan=True)
            )
            if not (same_nan and close):
                dmax = float(
                    np.nanmax(np.abs(s1[var].values - k[var].values))
                )
                failures.append(
                    f"wf1_raw vs keyed store NOT allclose for {var} "
                    f"(max abs {dmax:.4g})"
                )
        print(f"[edge: wf1_raw vs keyed store {store_key}] allclose checked "
              f"for {list(s1.data_vars)}")
    else:
        failures.append(f"keyed store extraction missing: {key_path}")

    # --- Report -----------------------------------------------------------
    lines = [
        "| edge | variable | mean | max-abs | rmse |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, var, mean, maxabs, rmse in rows:
        lines.append(
            f"| {name} | {VAR_LABEL.get(var, var)} ({var}) "
            f"| {mean:.6g} | {maxabs:.6g} | {rmse:.6g} |"
        )
    table = "\n".join(lines)
    print(table)
    (qa / "ladder_tables.md").write_text(
        f"# P3-2a ladder tables ({clim_source}, {store_key})\n\n{table}\n",
        encoding="utf-8",
    )
    print(f"precip null-check (A2-A1, P): max-abs == {p_null!r} (must be 0.0)")

    # --- Side-by-side composites -----------------------------------------
    if args.old_plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt

        new_dir = project_dir / "plots/wflow_model_performance"
        for old_png in sorted(Path(args.old_plots).glob("clim_*.png")):
            new_png = new_dir / old_png.name
            if not new_png.exists():
                continue
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
            for ax, path, title in zip(
                axes, [old_png, new_png], ["OLD (inmaps forcing)", "NEW (raw @ parity)"]
            ):
                ax.imshow(mpimg.imread(str(path)))
                ax.set_title(title, fontsize=10)
                ax.axis("off")
            fig.suptitle(old_png.name, fontsize=11)
            out = qa / f"side_by_side_{old_png.name}"
            fig.tight_layout()
            fig.savefig(out)
            plt.close(fig)
            print(f"side-by-side written: {out}")

    if failures:
        print("\nGATE FAILURES:")
        for f_ in failures:
            print(f"  - {f_}")
        return 1
    print("\nLADDER CLEAN: all edges within tolerance; precip null-check exact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
