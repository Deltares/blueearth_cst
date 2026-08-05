"""Two-layer tests for the interchange-contract validators (design §5.5).

- **Layer 1 — synthetic pass/fail, fixture-independent, ALWAYS executed.**
  Every validator ships a conforming in-memory object (report == []) and a
  deliberately broken one (report != []). No file I/O — objects are built
  directly here (the validators take parsed objects, never paths), so a
  fixtureless checkout (fresh clone, CI) still executes every validator's pass
  AND fail path. "Green" is never indistinguishable from "nothing checked".

- **Layer 2 — real-fixture integration, skipped VISIBLY when the fixture is
  absent.** Each case opens an artifact under the untracked ``test_case/test_local``
  tree and carries the repo fixture-absent guard (mirroring
  ``tests/test_store_region_bbox.py``): a module-level ``_FIXTURE_ABSENT``
  reason constant + ``@pytest.mark.skipif``. Absence is a NAMED, reported
  condition (read via ``pytest -rs``), never silence. The three temp() content
  validators (WG-4/WG-6/HM-6b) additionally skip with a documented reason when
  the temp artifact is absent (the default fixture state) — see the commit-4
  temp layer.

Source of record: ``dev/milestones/p32b/interchange-contracts-design.md`` §5.5 and the two
seam docs ``dev/reference/contracts/*-seam.md``.
"""
import os
from os.path import dirname, join, realpath

import pandas as pd
import pytest
import yaml

from blueearth_cst.shared import interchange_contracts as ic  # noqa: E402
from blueearth_cst.shared.snake_utils import stress_test_grid  # noqa: E402

# --- Fixture location + the single named skip reason -----------------------

TESTDIR = dirname(realpath(__file__))
SNAKEDIR = join(TESTDIR, "..")
_FIXTURE = join(SNAKEDIR, "test_case", "test_local")
_EXP = join(_FIXTURE, "experiments", "experiment")

_FIXTURE_ABSENT = (
    "untracked test_case/test_local fixture tree not present "
    "(interchange-contract integration layer skipped)"
)


def _fixture_present() -> bool:
    return os.path.exists(_FIXTURE)


# ===========================================================================
# Layer 1 — synthetic pass/fail (fixture-independent, always executed)
# ===========================================================================
#
# Objects are built with xarray/pandas in-memory. Each validator gets one
# conforming object (report == []) and one one-fault object (report != []).


def _wg1_good():
    import numpy as np
    import xarray as xr

    n = 3
    ds = xr.Dataset(
        {
            v: (
                ("time", "latitude", "longitude"),
                np.zeros((n, 2, 2), dtype="float32"),
                {"units": u},
            )
            for v, u in ic._WG1_VARS_UNITS.items()
        },
        coords={
            "time": pd.date_range("2000-01-01", periods=n),
            "latitude": np.array([1.0, 2.0], dtype="float32"),
            "longitude": np.array([1.0, 2.0], dtype="float32"),
            "spatial_ref": 0,
        },
        attrs={"crs": 4326, "category": "meteo"},
    )
    return ds


def test_wg1_synthetic_pass():
    assert ic.validate_wg1(_wg1_good()) == []


def test_wg1_synthetic_fail():
    ds = _wg1_good().drop_vars("precip")  # missing a pinned variable
    assert ic.validate_wg1(ds) != []


def _wg2_good():
    return pd.DataFrame(
        {
            "month": list(range(1, 13)),
            "temp_mean": [0.0] * 12,
            "precip_mean": [0.7] * 12,
            "precip_variance": [1.0] * 12,
        }
    )


def test_wg2_synthetic_pass():
    assert ic.validate_wg2(_wg2_good()) == []


def test_wg2_synthetic_fail():
    df = _wg2_good().iloc[:6]  # only 6 rows, month domain broken
    assert ic.validate_wg2(df) != []


def _wg3_good():
    return {
        "general": {"variables": ["precip", "temp"]},
        "generateWeatherSeries": {k: 0 for k in ic._WG3_GWS_KEYS},
    }


def test_wg3_synthetic_pass():
    assert ic.validate_wg3(_wg3_good()) == []


def test_wg3_synthetic_fail():
    cfg = _wg3_good()
    del cfg["generateWeatherSeries"]["seed"]  # a required key removed
    assert ic.validate_wg3(cfg) != []


def _catalog_entry_good(uri="X:/rlz.nc"):
    return {
        "uri": uri,
        "driver": {
            "name": "raster_xarray",
            "options": {"preprocess": "harmonise_dims", "lock": False},
        },
        "metadata": {"crs": 4326, "category": "meteo"},
        "data_type": "RasterDataset",
    }


def _wg5_good(keys=("rlz_1_cst_0", "rlz_1_cst_1")):
    return {k: _catalog_entry_good() for k in keys}


def test_wg5_synthetic_pass():
    assert ic.validate_wg5(_wg5_good()) == []


def test_wg5_synthetic_fail():
    cfg = _wg5_good()
    cfg["rlz_1_cst_1"]["driver"]["name"] = "wrong_driver"  # bad driver
    assert ic.validate_wg5(cfg) != []


def _hm1_good():
    import numpy as np
    import xarray as xr

    return xr.Dataset(
        {v: (("latitude", "longitude"), np.zeros((2, 2))) for v in ic._HM1_REFERENCED},
        coords={
            "latitude": np.array([1.0, 2.0], dtype="float64"),
            "longitude": np.array([1.0, 2.0], dtype="float64"),
            "spatial_ref": 0,
        },
    )


def test_hm1_synthetic_pass():
    assert ic.validate_hm1(_hm1_good()) == []


def test_hm1_synthetic_fail():
    ds = _hm1_good().drop_vars("outlets")  # a referenced name missing
    assert ic.validate_hm1(ds) != []


def _hm2_good():
    import numpy as np
    import xarray as xr

    n = 3
    ds = xr.Dataset(
        {
            "precip": (
                ("time", "latitude", "longitude"),
                np.zeros((n, 2, 2), dtype="float32"),
                {"units": "mm d**-1", "unit": "mm", "grid_mapping": "spatial_ref"},
            ),
            # pet: unit attr ABSENT on purpose — proves asserted-if-present
            # never blocks when the attr is missing.
            "pet": (
                ("time", "latitude", "longitude"),
                np.zeros((n, 2, 2), dtype="float32"),
                {"grid_mapping": "spatial_ref"},
            ),
            "temp": (
                ("time", "latitude", "longitude"),
                np.zeros((n, 2, 2), dtype="float32"),
                {"unit": "degree C.", "grid_mapping": "spatial_ref"},
            ),
        },
        coords={
            "time": pd.date_range("2000-01-01", periods=n),
            "latitude": np.array([1.0, 2.0], dtype="float64"),
            "longitude": np.array([1.0, 2.0], dtype="float64"),
            "spatial_ref": 0,
        },
    )
    return ds


def test_hm2_synthetic_pass():
    # A present-but-correct unit (precip) + an absent unit (pet) both pass.
    assert ic.validate_hm2(_hm2_good()) == []


def test_hm2_synthetic_fail():
    ds = _hm2_good()
    ds["temp"].attrs["unit"] = "kelvin"  # present-but-wrong unit attr
    assert ic.validate_hm2(ds) != []


def _hm3_good():
    import geopandas as gpd
    from shapely.geometry import Point, Polygon

    region = gpd.GeoDataFrame(
        {"geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]}, crs="EPSG:4326"
    )
    outlets = gpd.GeoDataFrame({"geometry": [Point(0.5, 0.5)]}, crs="EPSG:4326")
    outlet_index = pd.DataFrame(
        {"station_name": ["a"], "subcatchment_id": [1], "x": [0.5], "y": [0.5]}
    )
    return region, outlets, outlet_index


def test_hm3_synthetic_pass():
    assert ic.validate_hm3(*_hm3_good()) == []


def test_hm3_synthetic_fail():
    region, outlets, outlet_index = _hm3_good()
    region = region.to_crs("EPSG:3857")  # wrong CRS
    assert ic.validate_hm3(region, outlets, outlet_index) != []


def _hm4_good():
    return {
        "dir_output": ".",
        "model": {"cold_start__flag": True},
        "time": {
            "calendar": "standard",
            "starttime": "2070-01-01T00:00:00",
            "endtime": "2090-12-31T00:00:00",
            "timestepsecs": 86400,
        },
        "state": {"path_input": "in.nc", "path_output": "out.nc"},
        "input": {
            "path_static": "staticmaps.nc",
            "path_forcing": "inmaps.nc",
            "forcing": {
                "atmosphere_water__precipitation_volume_flux": "precip",
                "land_surface_water__potential_evaporation_volume_flux": "pet",
                "atmosphere_air__temperature": "temp",
            },
        },
        "output": {
            "csv": {
                "path": "output.csv",
                "column": [{"header": "Q", "map": "outlets", "parameter": "q"}],
            }
        },
    }


def test_hm4_synthetic_pass():
    assert ic.validate_hm4(_hm4_good()) == []


def test_hm4_synthetic_fail():
    cfg = _hm4_good()
    del cfg["time"]["timestepsecs"]  # a pinned rewrite field missing
    assert ic.validate_hm4(cfg) != []


def _hm5_good():
    return pd.DataFrame({"time": ["2070-01-01"], "Q_130000086": [1.0]})


def test_hm5_synthetic_pass():
    assert ic.validate_hm5(_hm5_good()) == []


def test_hm5_synthetic_fail():
    df = pd.DataFrame({"Q_130000086": [1.0]})  # no time column
    assert ic.validate_hm5(df) != []


def _hm7_good():
    qstats = pd.DataFrame(
        columns=["statistic", "temp_change", "precip_change", "Q_130000086"]
    )
    basin = pd.DataFrame(columns=["temp_change", "precip_change"])
    return qstats, basin


def test_hm7_synthetic_pass():
    assert ic.validate_hm7(*_hm7_good()) == []


def test_hm7_synthetic_fail():
    qstats, basin = _hm7_good()
    basin = pd.DataFrame(columns=["temp_change"])  # perturbation axis incomplete
    assert ic.validate_hm7(qstats, basin) != []


def test_hm7_accepts_the_shipped_template_default_basavg_columns():
    """The defect this phase must not inherit.

    The basin table's header is `temp_change`, `precip_change`, PLUS one column
    per configured `*_basavg` variable -- `export_wflow_results` builds it as
    `["temp_change", "precip_change"] + [c for c in sim.columns if "basavg" in c]`.
    The exact two-column assertion held only because the SEED CONFIG declares
    `wflow_outvars: ["river discharge"]` and so produces no basavg column.

    The SHIPPED TEMPLATE DEFAULT is
    `["river discharge", "actual evapotranspiration"]`, which does produce one.
    So the validator passed on the fixture and would have failed every project
    using the default -- a validator that only accepts the test's own shape.
    """
    qstats, _ = _hm7_good()
    basin = pd.DataFrame(
        columns=["temp_change", "precip_change", "actual_evapotranspiration_basavg"]
    )
    assert ic.validate_hm7(qstats, basin) == []


def test_hm7_accepts_the_unaggregated_realization_column():
    """The same class again, one config knob over.

    With `aggregate_rlz: false` the writer prepends a `realization` column
    (`col_names = ["realization", "temp_change", "precip_change"]`). An assertion
    pinned to the aggregated shape would reject every unaggregated run.
    """
    qstats, _ = _hm7_good()
    basin = pd.DataFrame(
        columns=["realization", "temp_change", "precip_change", "q_basavg"]
    )
    assert ic.validate_hm7(qstats, basin) == []


def test_hm7_still_rejects_a_foreign_basin_column():
    """Widening must not become 'accept anything'.

    Only the perturbation axis, the optional realization index, and
    `*_basavg` variables belong. A column that is none of those is still a
    contract violation and must be named.
    """
    qstats, _ = _hm7_good()
    basin = pd.DataFrame(columns=["temp_change", "precip_change", "Q_130000086"])
    diffs = ic.validate_hm7(qstats, basin)
    assert diffs and "Q_130000086" in diffs[0]


def test_hm7_rejects_the_pre_rename_axis_spelling():
    """The rename is a contract change, so the OLD header must now FAIL.

    `tavg` / `prcp` were the axis columns until 2026-08-05
    (`dev/milestones/r09/migration_indicator-axis-columns.md`). They were the
    repo's only violation of the `precip`/`temp` vocabulary naming.md §6 tier 2
    declares. A validator that accepted both spellings would let a stale writer
    keep emitting the old header undetected, which is exactly what the migration
    note exists to prevent -- so this pins the rejection, not just the acceptance.

    `tavg` also trips the foreign-column rule on the basin table, which is the
    right diagnosis to hand someone whose tree predates the rename.
    """
    qstats = pd.DataFrame(columns=["statistic", "tavg", "prcp", "Q_130000086"])
    basin = pd.DataFrame(columns=["tavg", "prcp"])
    diffs = ic.validate_hm7(qstats, basin)
    assert diffs
    assert any("temp_change" in d for d in diffs)
    assert any("precip_change" in d for d in diffs)


# --- Relational synthetic pass/fail (break exactly ONE member) -------------


def _gauge_identity_good():
    toml_cfg = {
        "output": {
            "csv": {
                "path": "output.csv",
                "column": [{"header": "Q", "map": "outlets", "parameter": "q"}],
            }
        }
    }
    output_rlz = pd.DataFrame({"time": ["2070-01-01"], "Q_130000086": [1.0]})
    qstats = pd.DataFrame(
        columns=["statistic", "temp_change", "precip_change", "Q_130000086"]
    )
    return toml_cfg, output_rlz, qstats


def test_gauge_identity_synthetic_pass():
    assert ic.validate_hm_gauge_column_identity(*_gauge_identity_good()) == []


def test_gauge_identity_synthetic_fail():
    toml_cfg, output_rlz, qstats = _gauge_identity_good()
    # Break exactly ONE member of the correlated set: rename the Qstats gauge
    # column so check-3 (list-equality) fires while TOML + output_rlz still agree.
    qstats = pd.DataFrame(
        columns=["statistic", "temp_change", "precip_change", "Q_999999999"]
    )
    assert ic.validate_hm_gauge_column_identity(toml_cfg, output_rlz, qstats) != []


def _catalog_grid_good():
    keys = [f"rlz_{n}_cst_{m}" for n in (1, 2) for m in range(0, 7)]
    catalog = {k: _catalog_entry_good() for k in keys}
    return catalog, 2, 6  # rlz_num=2, st_num=6


def test_catalog_grid_synthetic_pass():
    catalog, rlz_num, st_num = _catalog_grid_good()
    assert ic.validate_wg5_catalog_grid(catalog, rlz_num, st_num) == []


def test_catalog_grid_synthetic_fail():
    catalog, rlz_num, st_num = _catalog_grid_good()
    # Break exactly ONE member: drop a single expected catalog key.
    del catalog["rlz_1_cst_0"]
    assert ic.validate_wg5_catalog_grid(catalog, rlz_num, st_num) != []


# --- temp() content validators — synthetic pass/fail (commit-4 layer) -------
#
# WG-4 / WG-6 / HM-6b real artifacts are temp()-deleted and absent on the
# fixture; only their on-disk integration checks are skip-until-captured. Their
# logic is proven here on every checkout, fixture-independently.


def _wg4_good():
    import numpy as np
    import xarray as xr

    n = 3
    return xr.Dataset(
        {
            "precip": (("time", "lat", "lon"), np.zeros((n, 2, 2), dtype="float32")),
            "temp": (("time", "lat", "lon"), np.zeros((n, 2, 2), dtype="float32")),
        },
        coords={
            "time": pd.date_range("2070-01-01", periods=n),
            "lat": np.array([1.0, 2.0]),
            "lon": np.array([1.0, 2.0]),
            "spatial_ref": 0,
        },
        attrs={"crs": 4326, "category": "meteo"},
    )


def test_wg4_synthetic_pass():
    assert ic.validate_wg4(_wg4_good()) == []


def test_wg4_synthetic_fail():
    ds = _wg4_good().drop_vars("precip")  # missing a required variable
    assert ic.validate_wg4(ds) != []


def test_wg4_crs_category_absent_is_ok():
    """Empty global attrs must PASS — the real artifact's actual shape.

    Corrected 2026-07-25 on the first --notemp capture: the generator NC carries
    no global attrs at all. Its CRS lives in the spatial_ref coord (CF/rioxarray)
    and crs/category are catalog metadata that validate_wg5 pins. Requiring them
    here asserted the right values on the wrong surface.
    """
    ds = _wg4_good()
    ds.attrs = {}
    assert ic.validate_wg4(ds) == []


@pytest.mark.parametrize(
    "attrs",
    [
        {"crs": 3857},                          # contradictory crs
        {"category": "hydro"},                   # contradictory category
        {"crs": 4326, "category": "hydro"},      # one right, one wrong
    ],
)
def test_wg4_contradictory_crs_category_still_fails(attrs):
    """Asserted-if-present keeps its teeth: a PRESENT wrong value is a violation."""
    ds = _wg4_good()
    ds.attrs = attrs
    assert ic.validate_wg4(ds) != []


def test_wg6_synthetic_pass():
    # WG-6 shares HM-2's contract — reuse the conforming HM-2 object.
    assert ic.validate_wg6(_hm2_good()) == []


def test_wg6_synthetic_fail():
    ds = _hm2_good().drop_vars("pet")  # missing a required forcing variable
    assert ic.validate_wg6(ds) != []


def _hm6b_good():
    import numpy as np
    import xarray as xr

    return xr.Dataset(
        {"river_h": (("latitude", "longitude"), np.zeros((2, 2)))},
        coords={
            "latitude": np.array([1.0, 2.0]),
            "longitude": np.array([1.0, 2.0]),
        },
    )


def test_hm6b_synthetic_pass():
    assert ic.validate_hm6b(_hm6b_good()) == []


def test_hm6b_synthetic_fail():
    import xarray as xr

    ds = xr.Dataset()  # no grid axes, no state variables
    assert ic.validate_hm6b(ds) != []


# ===========================================================================
# Layer 2 — real-fixture integration (skipif _FIXTURE_ABSENT)
# ===========================================================================
#
# Each case opens a persisted fixture artifact and asserts the validator's
# report is empty. The 12 continuously-verified checks: 10 per-artifact
# (WG-1,2,3,5; HM-1,2,3,4,5,7) + 2 relational (gauge-identity — parametrized
# over the 12 (toml, output_rlz) pairs; catalog-grid).


def _open_ds(path):
    import xarray as xr

    return xr.open_dataset(path)


@pytest.mark.skipif(not _fixture_present(), reason=_FIXTURE_ABSENT)
def test_wg1_integration():
    key = "era5_20000101_20201231"
    path = join(_FIXTURE, "climate_historical", key, "extract_historical.nc")
    with _open_ds(path) as ds:
        assert ic.validate_wg1(ds) == []


@pytest.mark.skipif(not _fixture_present(), reason=_FIXTURE_ABSENT)
def test_wg2_integration():
    df = pd.read_csv(join(_EXP, "weather_generator", "_work", "cst_1.csv"))
    assert ic.validate_wg2(df) == []


@pytest.mark.skipif(not _fixture_present(), reason=_FIXTURE_ABSENT)
def test_wg3_integration():
    with open(join(_EXP, "weather_generator", "config", "weathergen_config.yml")) as f:
        cfg = yaml.safe_load(f)
    assert ic.validate_wg3(cfg) == []


@pytest.mark.skipif(not _fixture_present(), reason=_FIXTURE_ABSENT)
def test_wg5_integration():
    with open(join(_EXP, "data_catalog_climate_experiment.yml")) as f:
        cfg = yaml.safe_load(f)
    assert ic.validate_wg5(cfg) == []


@pytest.mark.skipif(not _fixture_present(), reason=_FIXTURE_ABSENT)
def test_wg5_catalog_grid_integration():
    with open(join(_EXP, "data_catalog_climate_experiment.yml")) as f:
        catalog = yaml.safe_load(f)
    with open(join(_EXP, "config", "snake_config_climate_experiment.yml")) as f:
        snap = yaml.safe_load(f)
    exp_cfg = snap["workflows"]["climate_experiment"]
    rlz_num = exp_cfg["realizations_num"]
    _, _, st_num = stress_test_grid(exp_cfg["stress_test"])
    assert ic.validate_wg5_catalog_grid(catalog, rlz_num, st_num) == []


@pytest.mark.skipif(not _fixture_present(), reason=_FIXTURE_ABSENT)
def test_hm1_integration():
    with _open_ds(join(_FIXTURE, "hydrology_model", "staticmaps.nc")) as ds:
        assert ic.validate_hm1(ds) == []


@pytest.mark.skipif(not _fixture_present(), reason=_FIXTURE_ABSENT)
def test_hm2_integration():
    path = join(_FIXTURE, "hydrology_model", "forcing", "inmaps_historical.nc")
    with _open_ds(path) as ds:
        assert ic.validate_hm2(ds) == []


@pytest.mark.skipif(not _fixture_present(), reason=_FIXTURE_ABSENT)
def test_hm3_integration():
    import geopandas as gpd

    geoms = join(_FIXTURE, "hydrology_model", "staticgeoms")
    region = gpd.read_file(join(geoms, "region.geojson"))
    outlets = gpd.read_file(join(geoms, "outlets.geojson"))
    outlet_index = pd.read_csv(join(geoms, "outlet_index.csv"))
    assert ic.validate_hm3(region, outlets, outlet_index) == []


@pytest.mark.skipif(not _fixture_present(), reason=_FIXTURE_ABSENT)
def test_hm4_integration():
    import tomllib

    with open(join(_FIXTURE, "hydrology_model", "wflow_sbm.toml"), "rb") as f:
        base = tomllib.load(f)
    assert ic.validate_hm4(base) == []
    with open(
        join(_EXP, "hydrology_runs", "rlz_1", "config", "cst_1.toml"), "rb"
    ) as f:
        percst = tomllib.load(f)
    assert ic.validate_hm4(percst) == []


@pytest.mark.skipif(not _fixture_present(), reason=_FIXTURE_ABSENT)
def test_hm5_integration():
    # wf1 output.csv + a wf3 per-cst output_rlz — both persist.
    wf1 = pd.read_csv(join(_FIXTURE, "hydrology_model", "run_default", "output.csv"))
    assert ic.validate_hm5(wf1) == []
    wf3 = pd.read_csv(join(_EXP, "hydrology_runs", "rlz_1", "output", "cst_1.csv"))
    assert ic.validate_hm5(wf3) == []


@pytest.mark.skipif(not _fixture_present(), reason=_FIXTURE_ABSENT)
def test_hm7_integration():
    qstats = pd.read_csv(join(_EXP, "results", "q_indicators.csv"))
    basin = pd.read_csv(join(_EXP, "results", "basin_indicators.csv"))
    assert ic.validate_hm7(qstats, basin) == []


def _gauge_identity_pairs():
    """The 12 fixture (toml, output_rlz) pairs (rlz {1,2} x cst {1..6})."""
    return [(n, m) for n in (1, 2) for m in range(1, 7)]


@pytest.mark.skipif(not _fixture_present(), reason=_FIXTURE_ABSENT)
@pytest.mark.parametrize("rlz,cst", _gauge_identity_pairs())
def test_gauge_identity_integration(rlz, cst):
    import tomllib

    with open(
        join(_EXP, "hydrology_runs", f"rlz_{rlz}", "config", f"cst_{cst}.toml"), "rb"
    ) as f:
        toml_cfg = tomllib.load(f)
    output_rlz = pd.read_csv(
        join(_EXP, "hydrology_runs", f"rlz_{rlz}", "output", f"cst_{cst}.csv")
    )
    qstats = pd.read_csv(join(_EXP, "results", "q_indicators.csv"))
    assert ic.validate_hm_gauge_column_identity(toml_cfg, output_rlz, qstats) == []


# --- temp() content integration cases — doubly skip-guarded (commit-4) ------
#
# Each carries BOTH the fixture-absent skipif (Layer-2 convention) AND a
# temp-absent runtime skip with the documented reason, since the temp()
# artifact is deleted after its consumer finishes and is absent on the default
# fixture. The ``--notemp`` capture procedure (both seam docs' validator
# indexes) un-skips these on disk without a design change.

_TEMP_ABSENT = "temp() artifact absent; capture via --notemp"

# Fixture temp() paths (present only after a --notemp capture run).
_WG4_NC = join(_EXP, "weather_generator", "output", "rlz_1_cst_1.nc")
_WG6_NC = join(_EXP, "hydrology_runs", "rlz_1", "forcing", "inmaps_cst_1.nc")
_HM6B_NC = join(_EXP, "hydrology_runs", "rlz_1", "output", "outstates_cst_1.nc")


@pytest.mark.skipif(not _fixture_present(), reason=_FIXTURE_ABSENT)
def test_wg4_integration():
    if not os.path.exists(_WG4_NC):
        pytest.skip(_TEMP_ABSENT)
    with _open_ds(_WG4_NC) as ds:
        assert ic.validate_wg4(ds) == []


@pytest.mark.skipif(not _fixture_present(), reason=_FIXTURE_ABSENT)
def test_wg6_integration():
    if not os.path.exists(_WG6_NC):
        pytest.skip(_TEMP_ABSENT)
    with _open_ds(_WG6_NC) as ds:
        assert ic.validate_wg6(ds) == []


@pytest.mark.skipif(not _fixture_present(), reason=_FIXTURE_ABSENT)
def test_hm6b_integration():
    if not os.path.exists(_HM6B_NC):
        pytest.skip(_TEMP_ABSENT)
    with _open_ds(_HM6B_NC) as ds:
        assert ic.validate_hm6b(ds) == []
