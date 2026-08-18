"""[R7-22] Unit coverage for `downscale_climate_forcing`'s extracted helpers.

The module used to run its whole body at import inside `with tee_to_log(...)`,
so none of this was reachable from a test. The conversion moved the body into
`downscale_climate_forcing(...)` and pulled three decisions out as pure
functions; those three are what a test can actually pin without a wflow model,
a data catalog and a forcing netCDF.

`forcing_window` has since moved to its own light module -- WF3's Snakefile
needs it at parse time to size batches against the disk, and cannot pay this
module's `hydromt_wflow` import to get it. It is still rule 3.14's window, so
its tests stay here; only the import moved.

The end-to-end path stays covered where it always was -- the workflow contract
tests and a real run -- not here.
"""

import pytest

from blueearth_cst.experiment.downscale_climate_forcing import (
    forcing_chunksize,
    pet_method_for,
)
from blueearth_cst.experiment.forcing_window import forcing_window


class TestForcingWindow:
    def test_an_even_run_length_is_centred_on_the_horizon(self):
        start, end = forcing_window(2050, 30)
        assert (start, end) == ("2035-01-01T00:00:00", "2065-12-31T00:00:00")

    def test_an_odd_run_length_puts_the_extra_year_at_the_end(self):
        """`ceil` backwards and `round` forwards, so 31 years split 15/16."""
        start, end = forcing_window(2050, 31)
        assert start == "2034-01-01T00:00:00"
        assert end == "2066-12-31T00:00:00"

    def test_the_window_spans_whole_years(self):
        start, end = forcing_window(2000, 20)
        assert start.endswith("-01-01T00:00:00")
        assert end.endswith("-12-31T00:00:00")

    def test_a_float_horizon_still_yields_integer_years(self):
        """`horizontime_climate` arrives from YAML and may parse as a float."""
        start, end = forcing_window(2050.0, 30)
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
