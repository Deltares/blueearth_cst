"""WF3 batch sizing — the disk ceiling P3-3 §6.1 calls the binding constraint.

The estimator is MEASURED rather than modelled (see the module docstring), so
the tests that matter most are the ones pinning it against real files: a
synthetic anchor with a known bytes-per-timestep, and the real fixture numbers
recorded from a live rapid run on 2026-08-18.
"""

import os
from datetime import date

import pytest

from blueearth_cst.experiment.batch_sizing import (
    BYTES_PER_GB,
    FORCING_ANCHOR,
    STATE_ANCHOR,
    MemberFootprint,
    disk_headroom_bytes,
    measure_member_footprint,
    resolve_batch_size,
)
from blueearth_cst.experiment.forcing_window import (
    forcing_window,
    forcing_window_years,
)

netCDF4 = pytest.importorskip("netCDF4")


def _write_forcing(path, steps, cells=384, nvars=3):
    """A forcing NC shaped like the real one: 3 vars, one chunk per timestep."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(str(path), "w") as ds:
        ds.createDimension("time", steps)
        ds.createDimension("latitude", cells)
        for name in list(("precip", "pet", "temp"))[:nvars]:
            var = ds.createVariable(
                name, "f4", ("time", "latitude"), zlib=True, complevel=4
            )
            var[:] = 0.0
    return path


def _model(tmp_path, hist_steps=2557):
    basin = tmp_path / "models" / "hydrology" / "wflow"
    _write_forcing(basin.joinpath(*FORCING_ANCHOR), hist_steps)
    state = basin.joinpath(*STATE_ANCHOR)
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_bytes(b"\0" * 106_086)
    return basin


class TestForcingWindow:
    """The window's own tests live with rule 3.14 in
    `test_downscale_climate_forcing.py`. What the DISK estimate depends on is
    narrower and is pinned here: the window is wider than `run_length`, so
    counting its days is not the same as multiplying by 365."""

    @pytest.mark.parametrize(
        "horizon, run_length, expected",
        [
            (2050, 30, (2035, 2065)),
            (2050, 8, (2046, 2054)),
        ],
    )
    def test_the_year_form_agrees_with_the_iso_form(
        self, horizon, run_length, expected
    ):
        assert forcing_window_years(horizon, run_length) == expected
        start, end = forcing_window(horizon, run_length)
        assert start.startswith(str(expected[0]))
        assert end.startswith(str(expected[1]))

    def test_the_window_is_wider_than_run_length(self):
        """The surprise the disk estimate must not miss: `run_length` 8 is NINE
        calendar years of forcing, so `run_length x 365` under-counts by 12 %."""
        start, end = forcing_window_years(2050, 8)
        days = (date(end, 12, 31) - date(start, 1, 1)).days + 1
        assert days == 3287
        assert days > 8 * 365


class TestMeasureMemberFootprint:
    def test_it_scales_the_historical_anchor_by_timestep_count(self, tmp_path):
        basin = _model(tmp_path, hist_steps=2557)
        hist_bytes = os.path.getsize(basin.joinpath(*FORCING_ANCHOR))

        fp = measure_member_footprint(basin, horizontime_climate=2050, run_length=8)

        # 2046..2054 inclusive = 3287 days.
        expected = round(hist_bytes / 2557 * 3287)
        assert fp.forcing_bytes == expected
        assert fp.forcing_bytes > hist_bytes  # a longer window costs more

    def test_state_is_not_scaled_at_all(self, tmp_path):
        """Outstates is one snapshot, so it is set by the grid and is
        independent of run length -- scaling it would be wrong, not merely
        imprecise."""
        basin = _model(tmp_path)
        short = measure_member_footprint(basin, 2050, 8)
        long = measure_member_footprint(basin, 2050, 40)
        assert short.state_bytes == long.state_bytes == 106_086
        assert long.forcing_bytes > short.forcing_bytes

    def test_total_is_both_temp_classes(self, tmp_path):
        fp = measure_member_footprint(_model(tmp_path), 2050, 8)
        assert fp.total_bytes == fp.forcing_bytes + fp.state_bytes

    @pytest.mark.parametrize("missing", [FORCING_ANCHOR, STATE_ANCHOR])
    def test_a_missing_anchor_yields_no_estimate_rather_than_an_error(
        self, tmp_path, missing
    ):
        """A fresh project, or a dry-run before WF1 has ever run. The cap must
        not become a new way for a run to fail."""
        basin = _model(tmp_path)
        basin.joinpath(*missing).unlink()
        assert measure_member_footprint(basin, 2050, 8) is None

    def test_an_absent_model_yields_no_estimate(self, tmp_path):
        assert measure_member_footprint(tmp_path / "nope", 2050, 8) is None

    def test_an_unreadable_forcing_file_yields_no_estimate(self, tmp_path):
        basin = _model(tmp_path)
        basin.joinpath(*FORCING_ANCHOR).write_bytes(b"not a netcdf")
        assert measure_member_footprint(basin, 2050, 8) is None

    def test_it_reproduces_the_measured_rapid_fixture(self, tmp_path):
        """The estimator's entire claim, against real numbers.

        Recorded 2026-08-18 from a live `snake_config_rapid.yml` run
        (horizon 2050, run_length 8, a 16x24 grid):

            <model>/forcing/inmaps_historical.nc   3_526_927 B over 2557 steps
            <exp>/…/forcing/inmaps_rlz_2_st_2.nc   4_530_000 B over 3287 steps

        Both files carry the same three float32 variables on the same grid at
        the same zlib level, chunked one timestep at a time -- which is WHY
        bytes-per-timestep transfers between them. If that ever stops holding,
        this test is the one that should notice.
        """
        basin = tmp_path / "wflow"
        forcing = basin.joinpath(*FORCING_ANCHOR)
        forcing.parent.mkdir(parents=True, exist_ok=True)
        forcing.write_bytes(b"\0" * 3_526_927)
        state = basin.joinpath(*STATE_ANCHOR)
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_bytes(b"\0" * 106_086)

        # The real file's time dimension, without needing a real netCDF here.
        import blueearth_cst.experiment.batch_sizing as mod

        original = mod._netcdf_timesteps
        mod._netcdf_timesteps = lambda _p: 2557
        try:
            fp = measure_member_footprint(basin, horizontime_climate=2050, run_length=8)
        finally:
            mod._netcdf_timesteps = original

        measured = 4_530_000
        assert fp.forcing_bytes == pytest.approx(measured, rel=0.01)


class TestDiskHeadroom:
    def test_an_absolute_budget_wins_outright(self, tmp_path):
        assert disk_headroom_bytes(tmp_path, fraction=0.25, headroom_gb=40) == (
            40 * BYTES_PER_GB
        )

    def test_absent_a_budget_it_is_a_share_of_free_space(self, tmp_path):
        import shutil

        free = shutil.disk_usage(str(tmp_path)).free
        got = disk_headroom_bytes(tmp_path, fraction=0.25)
        assert got == pytest.approx(free * 0.25, rel=0.05)

    def test_it_walks_up_to_a_volume_that_exists(self, tmp_path):
        """project_dir may not exist yet -- the run is about to create it."""
        future = tmp_path / "not" / "yet" / "created"
        assert disk_headroom_bytes(future, fraction=0.25) > 0


class TestResolveBatchSize:
    def _footprint(self, total):
        return MemberFootprint(forcing_bytes=total - 1, state_bytes=1, basis="test")

    def test_without_an_estimate_it_is_the_previous_behaviour(self):
        """The degradation path: min(batch_size_max, ceil(K / cores))."""
        got = resolve_batch_size(member_count=12, cores=3, batch_size_max=8)
        assert got.batch_size == 4
        assert got.bound_by == "parallelism"

    def test_batch_size_max_clamps_a_large_sweep(self):
        got = resolve_batch_size(member_count=60, cores=3, batch_size_max=8)
        assert got.batch_size == 8
        assert got.bound_by == "batch_size_max"

    def test_the_disk_ceiling_lowers_b_when_it_binds(self):
        """1 GB per member, 60 members, 3 cores, 12 GB headroom.

        The sweep needs 60 GB so the whole thing cannot be resident; the cap is
        then 12 GB / (3 cores x 1 GB) = 4.
        """
        got = resolve_batch_size(
            member_count=60,
            cores=3,
            batch_size_max=8,
            footprint=self._footprint(BYTES_PER_GB),
            headroom_bytes=12 * BYTES_PER_GB,
        )
        assert got.batch_size == 4
        assert got.bound_by == "disk"

    def test_it_never_raises_b_above_the_other_ceilings(self):
        """A machine with a huge disk must not get a bigger batch than the
        parallelism ceiling asked for."""
        got = resolve_batch_size(
            member_count=12,
            cores=3,
            batch_size_max=8,
            footprint=self._footprint(1024),
            headroom_bytes=10_000 * BYTES_PER_GB,
        )
        assert got.batch_size == 4
        assert got.bound_by == "parallelism"

    def test_a_sweep_that_fits_entirely_is_not_disk_constrained(self):
        """Once every member is resident at once, B has stopped controlling the
        footprint -- exactly the degenerate case P3-3 GN-3 measured."""
        got = resolve_batch_size(
            member_count=12,
            cores=3,
            batch_size_max=8,
            footprint=self._footprint(BYTES_PER_GB),
            headroom_bytes=20 * BYTES_PER_GB,
        )
        assert got.bound_by == "parallelism"
        assert got.batch_size == 4

    def test_b_never_falls_below_one(self):
        got = resolve_batch_size(
            member_count=1000,
            cores=8,
            batch_size_max=8,
            footprint=self._footprint(100 * BYTES_PER_GB),
            headroom_bytes=BYTES_PER_GB,
        )
        assert got.batch_size == 1

    def test_an_unfittable_sweep_warns_because_it_cannot_be_capped(self):
        got = resolve_batch_size(
            member_count=1000,
            cores=8,
            batch_size_max=8,
            footprint=self._footprint(100 * BYTES_PER_GB),
            headroom_bytes=BYTES_PER_GB,
        )
        assert got.warning is not None
        assert "exceeds" in got.warning
        # The console severity rule paints on this word; keep them in step.
        from blueearth_cst.shared.snake_utils import _ANSI_ALERT, _severity_code

        assert (
            _severity_code(f"12:00:00 - wflow - WARNING {got.warning}") == _ANSI_ALERT
        )

    def test_a_fitting_sweep_does_not_warn(self):
        got = resolve_batch_size(
            member_count=12,
            cores=3,
            batch_size_max=8,
            footprint=self._footprint(1024),
            headroom_bytes=BYTES_PER_GB,
        )
        assert got.warning is None

    def test_an_explicit_batch_size_wins_outright(self):
        """Unclamped on purpose: an operator who names B has said something the
        estimate cannot know."""
        got = resolve_batch_size(
            member_count=60,
            cores=3,
            batch_size_max=8,
            explicit=32,
            footprint=self._footprint(BYTES_PER_GB),
            headroom_bytes=BYTES_PER_GB,
        )
        assert got.batch_size == 32
        assert got.bound_by == "explicit"

    def test_an_explicit_batch_size_is_still_checked_against_the_disk(self):
        """Not enforced, but not silent either."""
        got = resolve_batch_size(
            member_count=60,
            cores=3,
            batch_size_max=8,
            explicit=32,
            footprint=self._footprint(BYTES_PER_GB),
            headroom_bytes=BYTES_PER_GB,
        )
        assert got.warning is not None

    def test_peak_is_capped_by_the_sweep_not_by_p_times_b(self):
        """`p x B` can exceed K; the sweep cannot hold more than K members."""
        got = resolve_batch_size(
            member_count=5,
            cores=8,
            batch_size_max=8,
            explicit=4,
            footprint=self._footprint(BYTES_PER_GB),
            headroom_bytes=100 * BYTES_PER_GB,
        )
        assert got.peak_bytes == 5 * BYTES_PER_GB

    def test_the_summary_names_the_binding_ceiling(self):
        got = resolve_batch_size(
            member_count=60,
            cores=3,
            batch_size_max=8,
            footprint=self._footprint(BYTES_PER_GB),
            headroom_bytes=12 * BYTES_PER_GB,
        )
        assert got.summary().startswith("B=4 (disk)")
        assert "headroom" in got.summary()

    def test_the_summary_is_short_without_an_estimate(self):
        got = resolve_batch_size(member_count=12, cores=3, batch_size_max=8)
        assert got.summary() == "B=4 (parallelism)"


class TestAdvancedSetting:
    def test_the_headroom_fraction_is_a_registered_default(self):
        from blueearth_cst.shared.snake_utils import ADVANCED_SETTINGS

        value = ADVANCED_SETTINGS["defaults"]["batch_disk_headroom_fraction"]
        assert 0 < value <= 1

    @pytest.mark.parametrize("bad", [0, -0.5, 1.5, "half", True, None])
    def test_an_out_of_range_fraction_is_rejected(self, bad):
        from blueearth_cst.shared.snake_utils import _unit_fraction

        with pytest.raises(ValueError):
            _unit_fraction(bad, "defaults.batch_disk_headroom_fraction")

    @pytest.mark.parametrize("good", [1, 0.25, 0.5])
    def test_an_in_range_fraction_is_accepted(self, good):
        from blueearth_cst.shared.snake_utils import _unit_fraction

        assert _unit_fraction(good, "where") == float(good)
