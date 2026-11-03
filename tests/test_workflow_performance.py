"""Performance contracts: reuse work without changing ensemble results."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from blueearth_cst.experiment.batch_sizing import measure_member_footprint
from blueearth_cst.projections import series_identity as si
from blueearth_cst.projections.derive_change_factors import derive_point_datasets
from blueearth_cst.projections.get_change_climate_proj_summary import (
    combine_changes,
    preprocess_coords,
)


@pytest.mark.parametrize("calendar", ["noleap", "360_day"])
def test_stage_b_reuses_series_and_reference_without_changing_merge(
    tmp_path, monkeypatch, calendar
):
    def write_series(name, scenario, shift):
        time = pd.date_range("2000-01-01", periods=96, freq="MS")
        steps = np.arange(len(time))
        ds = xr.Dataset(
            {
                "precip": ("time", 2 + steps % 12 / 12 + shift),
                "temp": ("time", 10 + np.sin(steps) + shift),
            },
            coords={"time": time},
        ).expand_dims(
            scenario=[scenario],
            member=["r1"],
            model=["long-model-name"],
            clim_project=["cmip6"],
        )
        components = {"scenario": scenario}
        ds.attrs.update(
            cst_calendar=calendar,
            cst_schema_version=si.SCHEMA_VERSION,
            cst_series_digest=si.series_digest(components, "region"),
        )
        path = tmp_path / name
        ds.to_netcdf(path)
        return path, components

    hist, hist_components = write_series("historical.nc", "historical", 0)
    future, fut_components = write_series("future.nc", "ssp245", 1)
    kwargs = dict(
        series_path_hist=hist,
        series_path=future,
        time_tuple_hist=("2000", "2003"),
        name_model="long-model-name",
        name_scenario="ssp245",
        region_fp="region",
        digest_components_hist=hist_components,
        digest_components_fut=fut_components,
        stats=["mean", "std", "q_90"],
        water_year_start="Oct",
        min_reference={"precip": 2.5},
    )
    series_cache, reference_cache = {}, {}
    annuals, monthlies, files = [], [], []
    opened = []
    cache_ids = {}
    original_open = xr.open_dataset

    def track_open(path, *args, **options):
        opened.append(Path(path))
        return original_open(path, *args, **options)

    with monkeypatch.context() as patch:
        patch.setattr(xr, "open_dataset", track_open)
        for horizon, window in [("near", ("2004", "2006")), ("far", ("2005", "2007"))]:
            annual, monthly, facts = derive_point_datasets(
                **kwargs,
                name_horizon=horizon,
                time_tuple_fut=window,
                series_cache=series_cache,
                reference_cache=reference_cache,
            )
            # The reference cache is populated once, then reused by horizon 2.
            if annuals:
                assert {
                    key: id(value)
                    for key, value in next(iter(reference_cache.values())).items()
                } == cache_ids
            cache_ids = {
                key: id(value)
                for key, value in next(iter(reference_cache.values())).items()
            }
            annuals.append(annual)
            monthlies.append(monthly)
            assert facts[2] == 3  # three complete October water years
        assert opened == [hist, future]
        with pytest.raises(RuntimeError, match="digest mismatch"):
            derive_point_datasets(
                **{**kwargs, "digest_components_hist": {"changed": True}},
                name_horizon="near",
                time_tuple_fut=("2004", "2006"),
                series_cache=series_cache,
                reference_cache=reference_cache,
            )

    for i, annual in enumerate(annuals):
        path = tmp_path / f"change-{i}.nc"
        annual.to_netcdf(path)
        files.append(path)
    with xr.open_mfdataset(
        files, coords="minimal", preprocess=preprocess_coords
    ) as old:
        xr.testing.assert_equal(combine_changes(annuals), old.load())
    for horizon, window, annual in zip(
        ["near", "far"], [("2004", "2006"), ("2005", "2007")], annuals
    ):
        uncached, uncached_monthly, _ = derive_point_datasets(
            **kwargs, name_horizon=horizon, time_tuple_fut=window
        )
        xr.testing.assert_identical(annual, uncached)
        xr.testing.assert_identical(
            monthlies[["near", "far"].index(horizon)], uncached_monthly
        )

    second, components = write_series("second.nc", "ssp585", 2)
    options = {
        **kwargs,
        "series_path": second,
        "name_scenario": "ssp585",
        "digest_components_fut": components,
    }
    cached = derive_point_datasets(
        **options,
        name_horizon="near",
        time_tuple_fut=("2004", "2006"),
        series_cache=series_cache,
        reference_cache=reference_cache,
    )
    fresh = derive_point_datasets(
        **options, name_horizon="near", time_tuple_fut=("2004", "2006")
    )
    assert len(reference_cache) == 1
    assert {
        key: id(value) for key, value in next(iter(reference_cache.values())).items()
    } == cache_ids
    for actual, expected in zip(cached[:2], fresh[:2]):
        xr.testing.assert_identical(actual, expected)
    monthlies.append(cached[1])
    monthly_files = []
    for i, monthly in enumerate(monthlies):
        path = tmp_path / f"monthly-{i}.nc"
        monthly.to_netcdf(path)
        monthly_files.append(path)
    with xr.open_mfdataset(monthly_files, coords="minimal") as old:
        merged = xr.combine_by_coords(
            monthlies,
            coords="minimal",
            compat="no_conflicts",
            join="outer",
            combine_attrs="override",
        )
        xr.testing.assert_equal(merged, old.load())
        pd.testing.assert_frame_equal(merged.to_dataframe(), old.to_dataframe())


def test_no_output_state_is_required_for_wf3(tmp_path):
    from blueearth_cst.shared.interchange_contracts import validate_hm4
    from tests.test_batch_sizing import _model
    from tests.test_interchange_contracts import _hm4_good

    basin = _model(tmp_path)
    (basin / "run_default/outstate/outstates.nc").unlink()
    assert measure_member_footprint(basin, 2046, 2054) is None
    footprint = measure_member_footprint(basin, 2046, 2054, write_states=False)
    assert footprint.state_bytes == 0
    assert footprint.total_bytes == footprint.forcing_bytes > 0
    cfg = _hm4_good()
    del cfg["state"]["path_output"]
    assert validate_hm4(cfg)  # WF1's default contract stays strict.
    assert validate_hm4(cfg, require_output_state=False) == []
    del cfg["state"]["path_input"]
    assert validate_hm4(cfg, require_output_state=False)
