"""Real HydroMT reads retain the widest overlap without changing daily values."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from blueearth_cst.climate_analysis.extract_historical_climate import (
    prep_historical_climate,
)
from blueearth_cst.model.add_climate_forcing import check_store_window


@pytest.mark.parametrize(
    "requested,expected,model_end",
    [
        (("1950-01-01", "2020-12-31"), ("1950-01-02", "2020-12-31"), "2020-12-31"),
        (("2000-01-01", "2025-12-31"), ("2000-01-01", "2023-11-30"), "2022-12-31"),
    ],
)
def test_era5_writes_the_actual_overlap(
    tmp_path, capsys, requested, expected, model_end
):
    dates = pd.date_range("1950-01-02", "2023-11-30", freq="D")
    shape = (len(dates), 2, 2)
    values = {
        "precip": np.broadcast_to(
            (np.arange(len(dates)) % 10).astype("float32")[:, None, None], shape
        ),
        **{
            name: np.full(shape, value, dtype="float32")
            for name, value in {
                "temp": 18,
                "temp_min": 10,
                "temp_max": 20,
                "kin": 100,
                "kout": 200,
                "press_msl": 1013,
            }.items()
        },
    }
    source = xr.Dataset(
        {
            name: (("time", "latitude", "longitude"), value)
            for name, value in values.items()
        },
        coords={"time": dates, "latitude": [0.0, 1.0], "longitude": [0.0, 1.0]},
    )
    source["precip"].attrs["units"] = "mm d**-1"
    source.raster.set_crs(4326)
    source_path = tmp_path / "source.nc"
    source.to_netcdf(source_path)
    catalog = tmp_path / "catalog.yml"
    catalog.write_text(
        yaml.safe_dump(
            {
                "era5": {
                    "data_type": "RasterDataset",
                    "uri": source_path.as_posix(),
                    "driver": {"name": "raster_xarray", "options": {"chunks": {}}},
                    "metadata": {
                        "crs": 4326,
                        "extent": {
                            "time_range": {"start": "1950-01-02", "end": "2023-11-30"}
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "extract_historical.nc"
    prep_historical_climate(
        region_fn=None,
        bbox=(0, 0, 1, 1),
        fn_out=output,
        data_libs=[str(catalog)],
        clim_source="era5",
        starttime=requested[0],
        endtime=requested[1],
    )
    expected_dates = pd.date_range(*expected, freq="D")
    with xr.open_dataset(output) as delivered:
        np.testing.assert_array_equal(delivered.time.values, expected_dates.values)
        elapsed_days = (expected_dates - dates[0]).days.to_numpy()
        expected_precip = np.broadcast_to(
            (elapsed_days % 10).astype("float32")[:, None, None],
            (len(expected_dates), 2, 2),
        )
        np.testing.assert_array_equal(delivered["precip"].values, expected_precip)
        assert set(values) <= set(delivered.data_vars)
    # Wflow's initial state is January 1; its first forcing update is January 2.
    assert check_store_window(output, requested[0], model_end, "era5") == (
        pd.Timestamp(expected[0]),
        pd.Timestamp(expected[1]),
    )
    outside_end = f"{pd.Timestamp(expected[1]).year + 1}-12-31"
    with pytest.raises(ValueError, match="does not cover the required updates"):
        check_store_window(output, requested[0], outside_end, "era5")
    log = capsys.readouterr().out
    assert "WARNING" in log
    assert f"extracting {expected[0]}–{expected[1]}" in log
