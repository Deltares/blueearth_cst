"""[R7-22] Unit coverage for `downscale_climate_forcing`'s extracted helpers.

The module used to run its whole body at import inside `with tee_to_log(...)`,
so none of this was reachable from a test. The conversion moved the body into
`downscale_climate_forcing(...)` and pulled three decisions out as pure
functions; those three are what a test can actually pin without a wflow model,
a data catalog and a forcing netCDF.

`forcing_window` has since moved to its own light module, and `C-72` reduced
it to rendering a DECLARED year pair -- WF3's Snakefile
needs it at parse time to size batches against the disk, and cannot pay this
module's `hydromt_wflow` import to get it. It is still rule 3.14's window, so
its tests stay here; only the import moved.

The end-to-end path stays covered where it always was -- the workflow contract
tests and a real run -- not here.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from blueearth_cst.experiment.downscale_climate_forcing import (
    forcing_chunksize,
    pet_method_for,
)
from blueearth_cst.experiment.forcing_window import forcing_window


class TestForcingWindow:
    def test_an_even_run_length_is_centred_on_the_horizon(self):
        start, end = forcing_window(2035, 2065)
        assert (start, end) == ("2035-01-01T00:00:00", "2065-12-31T00:00:00")

    def test_an_odd_run_length_puts_the_extra_year_at_the_end(self):
        """`ceil` backwards and `round` forwards, so 31 years split 15/16."""
        start, end = forcing_window(2034, 2066)
        assert start == "2034-01-01T00:00:00"
        assert end == "2066-12-31T00:00:00"

    def test_the_window_spans_whole_years(self):
        start, end = forcing_window(1990, 2010)
        assert start.endswith("-01-01T00:00:00")
        assert end.endswith("-12-31T00:00:00")

    def test_a_float_horizon_still_yields_integer_years(self):
        """`horizontime_climate` arrives from YAML and may parse as a float."""
        start, end = forcing_window(2035, 2065)
        assert (start, end) == ("2035-01-01T00:00:00", "2065-12-31T00:00:00")


class TestForcingChunksize:
    @pytest.mark.parametrize(
        "size, expected",
        [
            (2_000_000, 1),
            (1_000_001, 1),
            (1_000_000, 30),  # the > boundary, not >=
            (300_000, 30),
            (250_000, 100),
            (150_000, 100),
            (100_000, 365),
            (1, 365),
        ],
    )
    def test_thresholds(self, size, expected):
        assert forcing_chunksize(size) == expected

    def test_chunksize_never_increases_with_grid_size(self):
        sizes = [1, 1e5, 2.5e5, 1e6, 1e7]
        chunks = [forcing_chunksize(s) for s in sizes]
        assert chunks == sorted(chunks, reverse=True), chunks


class TestPetMethod:
    def test_eobs_takes_makkink(self):
        """E-OBS carries no radiation variable, so De Bruin is unavailable."""
        assert pet_method_for("eobs") == "makkink"

    @pytest.mark.parametrize("source", ["era5", "chirps", "era5_daily_zarr"])
    def test_every_other_source_takes_debruin(self, source):
        assert pet_method_for(source) == "debruin"


@pytest.mark.parametrize("longitude_shift", [0.0, 0.25])
def test_preparation_checks_wrapped_elevation_grid(
    tmp_path, monkeypatch, longitude_shift
):
    """Catalog wrapping preserves western cell values and rejects shifted grids."""
    import hydromt
    import yaml

    from blueearth_cst.experiment import downscale_climate_forcing as preparation
    from blueearth_cst.experiment.simulator_adapter import IncompatibleForcingError

    forcing = xr.Dataset(
        {"temp": (("time", "latitude", "longitude"), np.ones((1, 3, 4)))},
        coords={
            "time": [np.datetime64("2046-01-01")],
            "latitude": [7.0, 6.0, 5.0],
            "longitude": [-10.0, -9.0, -8.0, -7.0],
        },
    )
    forcing.raster.set_crs(4326)
    elevation = xr.Dataset(
        {"elevtn": (("latitude", "longitude"), np.tile(np.arange(360), (11, 1)))},
        coords={
            "latitude": np.arange(10, -1, -1, dtype=float),
            "longitude": np.arange(360, dtype=float) + longitude_shift,
        },
    )
    elevation.raster.set_crs(4326)
    catalog = {}
    for name, dataset in (("forcing", forcing), ("elevation", elevation)):
        path = tmp_path / f"{name}.nc"
        dataset.to_netcdf(path)
        catalog[name] = {
            "uri": str(path),
            "data_type": "RasterDataset",
            "driver": "raster_xarray",
            "metadata": {"crs": 4326},
        }
    catalog_path = tmp_path / "catalog.yml"
    catalog_path.write_text(yaml.safe_dump(catalog), encoding="utf-8")

    class PreparationReached(Exception):
        pass

    class Model:
        region = forcing.raster.box

        def __init__(self, **kwargs):
            self.staticmaps = SimpleNamespace(data=forcing)

        def setup_config(self, **kwargs):
            raise PreparationReached

    monkeypatch.setattr(preparation, "WflowSbmModel", Model)
    context = SimpleNamespace(
        catalogs=[SimpleNamespace(path=catalog_path)],
        ancillary=[SimpleNamespace(path=tmp_path / "elevation.nc")],
        forcing_entry="forcing",
        elevation_entry="elevation",
        pet_method="debruin",
    )
    run = SimpleNamespace(
        preparation_context=context,
        run_id="run_007",
        descriptor=SimpleNamespace(calendar="standard"),
    )
    settings = SimpleNamespace(
        forcing_path=tmp_path / "prepared.nc",
        toml_path=tmp_path / "wflow.toml",
        first_time="2046-01-01",
        last_time="2046-01-01",
        native_output_path=tmp_path / "output/discharge.csv",
        native_log_path=tmp_path / "output/log.txt",
    )
    expected = PreparationReached if longitude_shift == 0 else IncompatibleForcingError
    with pytest.raises(expected):
        preparation.prepare_model_forcing(run, SimpleNamespace(root=tmp_path), settings)
    if longitude_shift == 0:
        selected = hydromt.DataCatalog(data_libs=[str(catalog_path)]).get_rasterdataset(
            "elevation", geom=forcing.raster.box, buffer=2, variables=["elevtn"]
        )
        np.testing.assert_array_equal(
            selected.sel(longitude=forcing.longitude, latitude=forcing.latitude),
            np.tile([350, 351, 352, 353], (3, 1)),
        )
        selected.close()
