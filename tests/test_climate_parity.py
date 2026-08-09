"""Exactly-once correction tests for the P3-2a model-parity transform.

Synthetic constant-value grids with known DEMs make the lapse algebra exact
(design §5.2 ext1-1): parity temperature must equal the single-step correction
``T_raw + lapse_rate * (dem_model - dem_forcing)``, and chaining the
chirps-style extraction correction with the sidecar-DEM parity step must land
on the same value — a stacked (double) correction misses by
``lapse_rate * dem_chirps``-scale (°C), far outside tolerance.
"""

import numpy as np
import pandas as pd
import xarray as xr
from hydromt.model.processes.meteo import temp as meteo_temp  # noqa: E402

from blueearth_cst.shared.climate_parity import model_parity_climate  # noqa: E402

LAPSE_RATE = -0.0065


def _raster(value, coords_1d, time=None, name=None):
    """A constant CRS-tagged (time, y, x) or (y, x) DataArray."""
    ny = nx = len(coords_1d)
    if time is None:
        da = xr.DataArray(
            np.full((ny, nx), float(value)),
            dims=("y", "x"),
            coords={"y": coords_1d, "x": coords_1d},
            name=name,
        )
    else:
        da = xr.DataArray(
            np.full((len(time), ny, nx), float(value)),
            dims=("time", "y", "x"),
            coords={"time": time, "y": coords_1d, "x": coords_1d},
            name=name,
        )
    da.raster.set_crs(4326)
    return da


def _time():
    return pd.date_range("2000-01-01", periods=4, freq="D")


# Grids: the coarse "forcing" grid covers the fine "model" grid with margin so
# reprojection has full coverage and no edge NaNs inside the model domain.
_COARSE = np.linspace(-0.5, 1.5, 6)  # ~0.4 deg cells
_MID = np.linspace(-0.2, 1.2, 8)  # ~0.2 deg cells (the "chirps" grid)
_FINE = np.linspace(0.0, 1.0, 11)  # 0.1 deg cells (the "model" grid)

T_RAW = 20.0
DEM_FORCING = 100.0  # "era5 orography"
DEM_CHIRPS = 300.0  # "MERIT-on-chirps-grid" sidecar DEM
DEM_MODEL = 500.0


def _ds_raw(temp_grid, coords, time):
    """Raw extraction-shaped dataset (temp + debruin PET inputs)."""
    return xr.Dataset(
        {
            "precip": _raster(5.0, coords, time),
            "temp": temp_grid,
            "press_msl": _raster(1013.25, coords, time),
            "kin": _raster(200.0, coords, time),
            "kout": _raster(50.0, coords, time),
        }
    )


def test_era5_exactly_once_analytic():
    """era5 branch: parity temp == T_raw + lapse * (dem_model - dem_forcing)."""
    time = _time()
    ds_raw = _ds_raw(_raster(T_RAW, _COARSE, time, name="temp"), _COARSE, time)
    dem_model = _raster(DEM_MODEL, _FINE)
    dem_forcing = _raster(DEM_FORCING, _COARSE)

    out = model_parity_climate(ds_raw, dem_model, dem_forcing, "debruin")

    expected = T_RAW + LAPSE_RATE * (DEM_MODEL - DEM_FORCING)  # 17.4
    temp_vals = out["temp"].values
    assert np.isfinite(temp_vals).all()
    assert np.allclose(temp_vals, expected, atol=1e-4), (
        f"parity temp {np.unique(temp_vals)} != analytic {expected}"
    )
    # precip is reprojected but never corrected (net-correction ledger: 0)
    assert np.allclose(out["precip"].values, 5.0, atol=1e-6)
    # pet exists (debruin on the corrected grid) and is physically plausible
    assert np.isfinite(out["pet"].values).all()
    assert (out["pet"].values >= 0).all()


def test_chirps_sidecar_chaining_exactly_once():
    """chirps branch: extraction correction + sidecar-DEM parity == single step.

    The extraction lapse-corrects era5 temp onto the chirps grid against the
    sidecar DEM (extract_historical_climate.py chirps branch); the parity step
    with dem_forcing = that same sidecar first inverts it (MSL shift), then
    re-corrects to the model DEM — net exactly one correction (ext1-1).
    """
    time = _time()
    t_raw = _raster(T_RAW, _COARSE, time, name="temp")
    dem_era5 = _raster(DEM_FORCING, _COARSE)
    dem_chirps = _raster(DEM_CHIRPS, _MID)
    dem_model = _raster(DEM_MODEL, _FINE)

    # Extraction-style step (mirrors extract_historical_climate.py:132-141)
    t_chirps = meteo_temp(
        t_raw,
        dem_chirps,
        dem_forcing=dem_era5,
        lapse_correction=True,
        freq=None,
        reproj_method="nearest_index",
        lapse_rate=LAPSE_RATE,
    )
    assert np.allclose(
        t_chirps.values, T_RAW + LAPSE_RATE * (DEM_CHIRPS - DEM_FORCING), atol=1e-4
    )

    ds_chirps = _ds_raw(t_chirps, _MID, time)
    out = model_parity_climate(ds_chirps, dem_model, dem_chirps, "debruin")

    single_step = T_RAW + LAPSE_RATE * (DEM_MODEL - DEM_FORCING)  # 17.4
    assert np.allclose(out["temp"].values, single_step, atol=1e-4), (
        f"chained parity temp {np.unique(out['temp'].values)} != "
        f"single-step {single_step}"
    )


def test_chirps_wrong_dem_double_corrects():
    """Sensitivity: era5 orography instead of the sidecar leaves the chirps
    correction unreversed — off by lapse * (dem_chirps - dem_era5) ~= 1.3 degC
    (the ext1-1 defect this test exists to catch)."""
    time = _time()
    t_raw = _raster(T_RAW, _COARSE, time, name="temp")
    dem_era5 = _raster(DEM_FORCING, _COARSE)
    dem_chirps = _raster(DEM_CHIRPS, _MID)
    dem_model = _raster(DEM_MODEL, _FINE)

    t_chirps = meteo_temp(
        t_raw,
        dem_chirps,
        dem_forcing=dem_era5,
        lapse_correction=True,
        freq=None,
        reproj_method="nearest_index",
        lapse_rate=LAPSE_RATE,
    )
    ds_chirps = _ds_raw(t_chirps, _MID, time)
    # WRONG dem_forcing on the chirps branch:
    out = model_parity_climate(ds_chirps, dem_model, dem_era5, "debruin")

    single_step = T_RAW + LAPSE_RATE * (DEM_MODEL - DEM_FORCING)
    offset = LAPSE_RATE * (DEM_CHIRPS - DEM_FORCING)  # -1.3 degC
    assert np.allclose(out["temp"].values, single_step + offset, atol=1e-4)
    assert not np.allclose(out["temp"].values, single_step, atol=0.5)


def test_regrid_only_variant_applies_no_correction():
    """corrections=False (ladder state S2): temp is reprojected, never
    lapse-corrected; press is renamed uncorrected inside meteo.pet."""
    time = _time()
    ds_raw = _ds_raw(_raster(T_RAW, _COARSE, time, name="temp"), _COARSE, time)
    dem_model = _raster(DEM_MODEL, _FINE)
    dem_forcing = _raster(DEM_FORCING, _COARSE)

    out = model_parity_climate(
        ds_raw, dem_model, dem_forcing, "debruin", corrections=False
    )
    assert np.allclose(out["temp"].values, T_RAW, atol=1e-6)
    assert np.allclose(out["precip"].values, 5.0, atol=1e-6)
