"""WF2 native-grid member files carry the monthly means the delta producer needs."""

import json

import numpy as np
import pandas as pd
import xarray as xr

from blueearth_cst.projections import series_identity, spatial_diagnostics
from blueearth_cst.projections.get_change_climate_proj import analysis_windows
from blueearth_cst.projections.variable_spec import VariableSpec

SPEC = {
    "temp": VariableSpec("temp", "temp", "level", "degC", "absolute"),
    "precip": VariableSpec("precip", "precip", "rate", "mm/day", "relative", 0.1),
}


def _raw(path, start, years, temp, precip):
    time = pd.date_range(start, periods=12 * years, freq="MS")
    shape = (time.size, 1, 2)
    xr.Dataset(
        {
            "temp": (("time", "lat", "lon"), np.full(shape, temp)),
            # Cell 0 is wet; cell 1 is dry (0.05 mm/day) in every month.
            "precip": (
                ("time", "lat", "lon"),
                np.broadcast_to(np.array([precip, 0.05]), shape).copy(),
            ),
        },
        coords={"time": time, "lat": [0.5], "lon": [10.5, 11.5]},
        attrs={"cst_calendar": "standard"},
    ).to_netcdf(path)


def test_member_grids_publish_monthly_reference_and_future(tmp_path, monkeypatch):
    monkeypatch.setattr(series_identity, "assert_raw_identity", lambda *a: None)
    monkeypatch.setattr(series_identity, "assert_raw_coverage", lambda *a: None)
    hist, fut = tmp_path / "hist.nc", tmp_path / "fut.nc"
    _raw(hist, "1980-01-01", 35, 10.0, 2.0)
    _raw(fut, "2015-01-01", 86, 12.0, 2.5)
    point = {
        "model": "TEST/M1",
        "scenario": "ssp245",
        "member": "r1",
        "raw_path_hist": str(hist),
        "raw_path_fut": str(fut),
        "raw_digest_hist": "h",
        "raw_digest_fut": "f",
    }
    windows = list(
        analysis_windows({"near": (2040, 2059)}, reference_window=(1985, 2004)).values()
    )
    out = tmp_path / "grids"
    inventory = spatial_diagnostics.prepare_spatial_changes(
        [point], windows, out, variable_spec=SPEC, min_reference={"precip": 0.1}
    )
    assert inventory["schema_version"] == 2
    assert json.loads((out / "inventory.json").read_text())["schema_version"] == 2
    with xr.open_dataset(out / inventory["members"][0]["path"]) as ds:
        reference = ds["precip_monthly_reference"]
        future = ds["precip_monthly_future"]
        ratio = ds["precip_monthly_ratio"]
        np.testing.assert_allclose(reference.isel(lon=0), 2.0)
        np.testing.assert_allclose(future.isel(lon=0), 2.5)
        # Dry cell: the ratio is withheld, but both means remain for consumers.
        assert ratio.isel(lon=1).isnull().all()
        np.testing.assert_allclose(reference.isel(lon=1), 0.05)
        np.testing.assert_allclose(ratio.isel(lon=0), 1.25)
        assert ratio.attrs["min_reference"] == 0.1
        assert reference.attrs["units"] == "mm/day"
        np.testing.assert_allclose(ds["temp_monthly"].isel(lon=0), 2.0)
        assert "temp_monthly_reference" not in ds
