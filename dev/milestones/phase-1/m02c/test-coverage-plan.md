# M02c Test Coverage Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking. Implement task-by-task and check off as you go. Do not skip the verification commands — they are the contract.

**Goal:** Add unit-test coverage for four `src/` modules (`metrics_definition`, `setup_time_horizon`, `prepare_climate_data_catalog`, `extract_historical_climate`) with two strict xfails for documented bugs, replicating the `tests/test_stage_data.py` mocking pattern.

**Architecture:** One test file per source module under `tests/`. Heavy deps stubbed via `sys.modules.setdefault` so tests run without initializing hydromt / xarray / geopandas. Inline fixtures per file (no shared `conftest.py` changes). No Snakemake `script:` injection — tests call functions directly. `pytest.mark.xfail(strict=True)` for unfixed bugs.

**Tech Stack:** pytest 9.x (already in pixi env), pandas 3.x, numpy 2.x, pyyaml. xclim/hydromt are mocked, not imported. Pixi runs the suite via `pixi run pytest tests/`.

**Branch:** `milestone/02c-tests` (already created off `m02b-upgrades` tip).

**Spec:** `dev/milestones/phase-1/m02c/test-coverage-design.md`. Read it first if you haven't.

---

## Pre-task setup

### Step 0.1: Verify branch and clean tree

- [ ] Run: `git branch --show-current` → expect `milestone/02c-tests`
- [ ] Run: `git status --short` → expect empty output (clean tree)

### Step 0.2: Verify existing test suite passes

- [ ] Run: `pixi run pytest tests/ -v`
- [ ] Expect: `13 passed, 2 xfailed` (or similar — the 2 xfails come from `tests/test_cli.py`)
- [ ] If anything else fails, stop and investigate before adding new tests

---

## Task 1: `tests/test_metrics_definition.py`

**Files:**
- Create: `tests/test_metrics_definition.py`
- Target source: `src/metrics_definition.py` (69 lines, 11 functions; we test 7 of the pure-pandas ones)

**Mocking strategy:** None for the tested functions — they're pure pandas. The xclim-dependent functions (`returninterval`, `returninterval_Q7d`, `returnintervalmulti`) are deliberately skipped: xclim's `frequency_analysis` is heavy and edge-case territory, better suited to integration tests in M4.

**Why this module first:** It's the simplest. Establishes the file naming convention (`test_<module>.py`) and the docstring-comment pattern that documents the mocking discipline for future test files.

### Step 1.1: Create the file with the documented mocking-pattern header

- [ ] Create `tests/test_metrics_definition.py` with this content:

```python
"""Unit tests for src/metrics_definition.py — pure-pandas streamflow metrics.

Testing pattern (M02c convention; see dev/milestones/phase-1/m02c/test-coverage-design.md):
- One test file per source module: tests/test_<module>.py
- Heavy deps (hydromt, xarray, geopandas) stubbed via sys.modules.setdefault
  at the top of the file. See tests/test_stage_data.py for the canonical
  example. This file uses no stubs because the tested functions are pure
  pandas/numpy.
- Inline fixtures per file. Promote to conftest.py only if reused in 3+ files.
- No Snakemake script injection — call functions directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import metrics_definition  # noqa: E402


def _daily_df(values, start="2020-01-01"):
    """Build a 1-column daily-indexed DataFrame for metric inputs."""
    idx = pd.date_range(start=start, periods=len(values), freq="D")
    return pd.DataFrame({"q": values}, index=idx)


def _three_year_index():
    """Three full calendar years of daily timestamps (2020-01-01 → 2022-12-31)."""
    return pd.date_range("2020-01-01", "2022-12-31", freq="D")
```

### Step 1.2: Add Q7d_total / Q7d_min / Q7d_maxyear / BFI tests

- [ ] Append to the same file:

```python
def test_Q7d_total_returns_rolling_seven_day_mean():
    df = _daily_df(list(range(20)))
    result = metrics_definition.Q7d_total(df)

    # First 6 rows are NaN (insufficient window); row 7 onward equals the
    # trailing 7-day mean of the input.
    assert result["q"].iloc[:6].isna().all()
    assert result["q"].iloc[6] == pytest.approx(np.mean(range(7)))
    assert result["q"].iloc[19] == pytest.approx(np.mean(range(13, 20)))


def test_Q7d_min_constant_series_returns_constant():
    idx = _three_year_index()
    df = pd.DataFrame({"q": np.full(len(idx), 100.0)}, index=idx)

    result = metrics_definition.Q7d_min(df)

    # Series of length 1 (one entry per column).
    assert result.iloc[0] == pytest.approx(100.0)


def test_Q7d_maxyear_averages_yearly_rolling_maxes():
    # 2020 = constant 50, 2021 = constant 100, 2022 = constant 150.
    # Rolling 7-day mean is constant within each year (after the first 6
    # days), so yearly max == yearly value. Mean of [50, 100, 150] == 100.
    parts = []
    for year, value in [("2020", 50.0), ("2021", 100.0), ("2022", 150.0)]:
        idx = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        parts.append(pd.DataFrame({"q": np.full(len(idx), value)}, index=idx))
    df = pd.concat(parts)

    result = metrics_definition.Q7d_maxyear(df)

    assert result.iloc[0] == pytest.approx(100.0)


def test_BFI_constant_series_returns_one():
    idx = _three_year_index()
    df = pd.DataFrame({"q": np.full(len(idx), 100.0)}, index=idx)

    result = metrics_definition.BFI(df)

    # Q7d_min == annmean for a constant series → BFI == 1.0.
    assert result.iloc[0] == pytest.approx(1.0)
```

### Step 1.3: Add wetmonth_mean / drymonth_mean tests

- [ ] Append:

```python
def test_wetmonth_mean_finds_january_when_january_is_wet():
    # Construct a series where January is the only month with high values;
    # all other months are 0. wetmonth_mean should detect January as the
    # wet month and return January's annual mean (which equals the high
    # value, since the rest of the month doesn't dilute it).
    idx = _three_year_index()
    values = np.zeros(len(idx))
    jan_mask = idx.month == 1
    values[jan_mask] = 50.0
    df = pd.DataFrame({"q": values}, index=idx)

    result = metrics_definition.wetmonth_mean(df)

    # January mean per year == 50.0; mean across years == 50.0.
    assert result.iloc[0] == pytest.approx(50.0)


def test_drymonth_mean_finds_august_when_august_is_dry():
    # Mirror of wetmonth_mean: all months at 100.0 except August at 0.
    # August has the lowest monthlysum, so drymonth_mean returns August's
    # annual mean == 0.0.
    idx = _three_year_index()
    values = np.full(len(idx), 100.0)
    aug_mask = idx.month == 8
    values[aug_mask] = 0.0
    df = pd.DataFrame({"q": values}, index=idx)

    result = metrics_definition.drymonth_mean(df)

    assert result.iloc[0] == pytest.approx(0.0)
```

### Step 1.4: Run the new tests in isolation

- [ ] Run: `pixi run pytest tests/test_metrics_definition.py -v`
- [ ] Expect: 6 passed, 0 failed, 0 xfailed
- [ ] If any fail, fix the test (the source is treated as the contract for now)

### Step 1.5: Run the full suite to confirm no regressions

- [ ] Run: `pixi run pytest tests/ -v`
- [ ] Expect: previous count + 6 = `19 passed, 2 xfailed`

### Step 1.6: Commit

- [ ] Run:
```
git add tests/test_metrics_definition.py
git commit -m "$(cat <<'EOF'
m02c: add unit tests for metrics_definition (pure-pandas metrics)

Six tests exercising Q7d_total, Q7d_min, Q7d_maxyear, BFI, wetmonth_mean,
drymonth_mean on synthetic constant-per-year and known-peak-month series.
xclim-dependent metrics (returninterval*) deferred to M4 integration tests.
Establishes the M02c testing pattern and file-naming convention.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `tests/test_setup_time_horizon.py`

**Files:**
- Create: `tests/test_setup_time_horizon.py`
- Target source: `src/setup_time_horizon.py` — function `prep_hydromt_update_forcing_config`

**Mocking strategy:** Stub `hydromt_wflow.WflowSbmModel` via `sys.modules.setdefault` because the function imports it at module-load time. We construct fake model objects whose `.staticmaps.data.raster.size` returns predetermined values to exercise each chunksize branch.

**Branches under test:**
- `precip_source == "eobs"` → eobs / eobs_orography / makkink
- `precip_source == "era5"` (default) → era5 / era5_orography / debruin
- `wflow_root is None` → chunksize=30
- `wflow_root` provided + size > 1e6 → chunksize=1
- `wflow_root` provided + size > 2.5e5 → chunksize=30
- `wflow_root` provided + size > 1e5 → chunksize=100
- `wflow_root` provided + size <= 1e5 → chunksize=365

### Step 2.1: Create the file with stubs and helpers

- [ ] Create `tests/test_setup_time_horizon.py`:

```python
"""Unit tests for src/setup_time_horizon.py — forcing config YAML builder.

Note: module name is misleading. The function builds a hydromt update YAML
for adding forcing to a wflow model; testable surface is the chunksize
branching by staticmaps size and the precip-source branching (era5/eobs).
M3 may rename the module; tests pin behavior, not names.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import yaml


# Stub hydromt_wflow before importing the source module. The function uses
# WflowSbmModel(root=..., mode="r").staticmaps.data.raster.size — we need
# the stub deep enough to support that chain.
class _FakeRaster:
    def __init__(self, size):
        self.size = size


class _FakeStaticmaps:
    def __init__(self, size):
        self.data = types.SimpleNamespace(raster=_FakeRaster(size))


class _FakeWflowSbmModel:
    """Replaces hydromt_wflow.WflowSbmModel. Configured per test via _NEXT_SIZE."""

    def __init__(self, root, mode):
        # Read the size pre-set by the test on this class attribute.
        size = type(self)._NEXT_SIZE
        self.staticmaps = _FakeStaticmaps(size)

    _NEXT_SIZE = 0


sys.modules.setdefault(
    "hydromt_wflow",
    types.SimpleNamespace(WflowSbmModel=_FakeWflowSbmModel),
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import setup_time_horizon  # noqa: E402


def _read_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
```

### Step 2.2: Add precip-source branching tests

- [ ] Append:

```python
def test_default_precip_source_uses_era5_stack(tmp_path):
    fn = tmp_path / "out.yml"
    setup_time_horizon.prep_hydromt_update_forcing_config(
        starttime="2010-01-01T00:00:00",
        endtime="2010-12-31T00:00:00",
        fn_yml=fn,
        precip_source="era5",
    )

    cfg = _read_yaml(fn)
    steps_by_key = {next(iter(s)): next(iter(s.values())) for s in cfg["steps"]}

    assert steps_by_key["setup_precip_forcing"]["precip_fn"] == "era5"
    assert steps_by_key["setup_temp_pet_forcing"]["temp_pet_fn"] == "era5"
    assert steps_by_key["setup_temp_pet_forcing"]["dem_forcing_fn"] == "era5_orography"
    assert steps_by_key["setup_temp_pet_forcing"]["pet_method"] == "debruin"


def test_eobs_precip_source_swaps_orography_and_pet_method(tmp_path):
    fn = tmp_path / "out.yml"
    setup_time_horizon.prep_hydromt_update_forcing_config(
        starttime="2010-01-01T00:00:00",
        endtime="2010-12-31T00:00:00",
        fn_yml=fn,
        precip_source="eobs",
    )

    cfg = _read_yaml(fn)
    steps_by_key = {next(iter(s)): next(iter(s.values())) for s in cfg["steps"]}

    assert steps_by_key["setup_precip_forcing"]["precip_fn"] == "eobs"
    assert steps_by_key["setup_temp_pet_forcing"]["temp_pet_fn"] == "eobs"
    assert steps_by_key["setup_temp_pet_forcing"]["dem_forcing_fn"] == "eobs_orography"
    assert steps_by_key["setup_temp_pet_forcing"]["pet_method"] == "makkink"


def test_unknown_precip_source_falls_back_to_era5_stack(tmp_path):
    """The else branch catches anything that isn't 'eobs' — including typos."""
    fn = tmp_path / "out.yml"
    setup_time_horizon.prep_hydromt_update_forcing_config(
        starttime="2010-01-01T00:00:00",
        endtime="2010-12-31T00:00:00",
        fn_yml=fn,
        precip_source="not_a_real_source",
    )

    cfg = _read_yaml(fn)
    steps_by_key = {next(iter(s)): next(iter(s.values())) for s in cfg["steps"]}

    assert steps_by_key["setup_temp_pet_forcing"]["temp_pet_fn"] == "era5"
    assert steps_by_key["setup_temp_pet_forcing"]["pet_method"] == "debruin"
```

### Step 2.3: Add chunksize branching tests

- [ ] Append:

```python
def _chunksize_from_yaml(cfg):
    """Pull the chunksize value out of the setup_precip_forcing step."""
    for step in cfg["steps"]:
        if "setup_precip_forcing" in step:
            return step["setup_precip_forcing"]["chunksize"]
    raise AssertionError("setup_precip_forcing step missing")


def test_chunksize_defaults_to_30_when_wflow_root_is_none(tmp_path):
    fn = tmp_path / "out.yml"
    setup_time_horizon.prep_hydromt_update_forcing_config(
        starttime="2010-01-01T00:00:00",
        endtime="2010-12-31T00:00:00",
        fn_yml=fn,
        precip_source="era5",
        wflow_root=None,
    )
    assert _chunksize_from_yaml(_read_yaml(fn)) == 30


@pytest.mark.parametrize(
    "size,expected_chunksize",
    [
        (2_000_000, 1),     # > 1e6
        (500_000, 30),      # > 2.5e5
        (200_000, 100),     # > 1e5
        (50_000, 365),      # ≤ 1e5
        (1_000_000, 30),    # boundary: not > 1e6, but > 2.5e5
        (250_000, 100),     # boundary: not > 2.5e5, but > 1e5
    ],
)
def test_chunksize_branches_by_staticmaps_size(tmp_path, size, expected_chunksize):
    _FakeWflowSbmModel._NEXT_SIZE = size
    fn = tmp_path / "out.yml"
    setup_time_horizon.prep_hydromt_update_forcing_config(
        starttime="2010-01-01T00:00:00",
        endtime="2010-12-31T00:00:00",
        fn_yml=fn,
        precip_source="era5",
        wflow_root=tmp_path,  # any non-None path; fake model ignores it
    )
    assert _chunksize_from_yaml(_read_yaml(fn)) == expected_chunksize
```

### Step 2.4: Add YAML-structure tests

- [ ] Append:

```python
def test_output_yaml_has_modeltype_and_three_steps(tmp_path):
    fn = tmp_path / "out.yml"
    setup_time_horizon.prep_hydromt_update_forcing_config(
        starttime="2010-01-01T00:00:00",
        endtime="2010-12-31T00:00:00",
        fn_yml=fn,
        precip_source="era5",
    )

    cfg = _read_yaml(fn)
    assert cfg["modeltype"] == "wflow_sbm"
    assert len(cfg["steps"]) == 3
    step_keys = [next(iter(s)) for s in cfg["steps"]]
    assert step_keys == [
        "setup_config",
        "setup_precip_forcing",
        "setup_temp_pet_forcing",
    ]


def test_starttime_and_endtime_propagate_to_setup_config(tmp_path):
    fn = tmp_path / "out.yml"
    setup_time_horizon.prep_hydromt_update_forcing_config(
        starttime="1995-06-01T00:00:00",
        endtime="2015-06-30T23:00:00",
        fn_yml=fn,
        precip_source="era5",
    )

    cfg = _read_yaml(fn)
    setup_cfg_step = next(s for s in cfg["steps"] if "setup_config" in s)
    data = setup_cfg_step["setup_config"]["data"]

    assert data["time.starttime"] == "1995-06-01T00:00:00"
    assert data["time.endtime"] == "2015-06-30T23:00:00"
    assert data["time.timestepsecs"] == 86400


def test_creates_parent_dirs_for_output_yaml(tmp_path):
    fn = tmp_path / "deep" / "nested" / "out.yml"
    setup_time_horizon.prep_hydromt_update_forcing_config(
        starttime="2010-01-01T00:00:00",
        endtime="2010-12-31T00:00:00",
        fn_yml=fn,
        precip_source="era5",
    )
    assert fn.exists()
```

### Step 2.5: Run new tests

- [ ] Run: `pixi run pytest tests/test_setup_time_horizon.py -v`
- [ ] Expect: 13 passed (3 precip-source + 1 default chunksize + 6 parametrized chunksize + 3 YAML structure / starttime / parent-dirs).

### Step 2.6: Run full suite

- [ ] Run: `pixi run pytest tests/ -v`
- [ ] Expect: prior count + 13 = `32 passed, 2 xfailed`

### Step 2.7: Commit

- [ ] Run:
```
git add tests/test_setup_time_horizon.py
git commit -m "$(cat <<'EOF'
m02c: add unit tests for setup_time_horizon (forcing config builder)

Thirteen tests covering precip-source branching (era5/eobs/fallback),
chunksize selection by staticmaps size (parametrized over the four
branches plus boundary cases), default-when-wflow-root-None, and YAML
structure (modeltype, step keys, starttime/endtime propagation, parent
dir creation). hydromt_wflow.WflowSbmModel stubbed via sys.modules.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `tests/test_prepare_climate_data_catalog.py`

**Files:**
- Create: `tests/test_prepare_climate_data_catalog.py`
- Target source: `src/prepare_climate_data_catalog.py` — function `prepare_clim_data_catalog`

**Mocking strategy:** Stub `hydromt` via `sys.modules.setdefault`. The function uses `hydromt.DataCatalog(data_libs=...).to_dict()[source_like]` to inherit attributes. We provide a fake `DataCatalog` that returns a known dict.

**xfail this task introduces:** `test_hydromt_to_yml_round_trip_preserves_preprocess` — the underlying hydromt 1.3 issue that motivated the `yaml.safe_dump` workaround. Strict-xfailed until upstream fix lands.

### Step 3.1: Create the file with stubs

- [ ] Create `tests/test_prepare_climate_data_catalog.py`:

```python
"""Unit tests for src/prepare_climate_data_catalog.py.

The tested function builds a hydromt 1.x data catalog dict for R-generated
realization netCDFs, inheriting driver/metadata from a source_like entry.
Mocks hydromt.DataCatalog to return a controllable fake. The
yaml.safe_dump workaround (bypassing DataCatalog.to_yml because hydromt
1.3 strips driver.options.preprocess) is itself the subject of one of
the xfail regression tests below.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import yaml


# Fake hydromt.DataCatalog. Tests configure _CATALOG to control to_dict() output.
class _FakeDataCatalog:
    _CATALOG = {}  # set by test before instantiation

    def __init__(self, data_libs=None):
        self._data_libs = data_libs

    def to_dict(self):
        # Return a deep-ish copy so tests can assert the function doesn't
        # mutate the source catalog.
        return {k: dict(v) for k, v in type(self)._CATALOG.items()}

    def from_dict(self, d):
        return self


sys.modules.setdefault(
    "hydromt", types.SimpleNamespace(DataCatalog=_FakeDataCatalog)
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import prepare_climate_data_catalog as pcdc  # noqa: E402


@pytest.fixture
def era5_like_catalog():
    """A minimal era5 source dict shaped like hydromt 1.x DataCatalog.to_dict()."""
    _FakeDataCatalog._CATALOG = {
        "era5": {
            "data_type": "RasterDataset",
            "uri": "/some/source/era5.nc",
            "driver": {
                "name": "netcdf",
                "options": {"preprocess": "harmonise_dims", "lock": False},
            },
            "data_adapter": {"unit_mult": {"precip": 1000.0}},
            "root": "/data/root",
            "metadata": {"crs": 4326, "category": "meteo"},
        }
    }
    yield
    _FakeDataCatalog._CATALOG = {}


@pytest.fixture
def chirps_like_catalog():
    _FakeDataCatalog._CATALOG = {
        "chirps_global": {
            "data_type": "RasterDataset",
            "uri": "/some/source/chirps.nc",
            "driver": {"name": "netcdf"},
            "metadata": {"category": "meteo"},
        }
    }
    yield
    _FakeDataCatalog._CATALOG = {}
```

### Step 3.2: Add structural tests for non-chirps path

- [ ] Append:

```python
def test_single_fn_produces_one_entry_named_after_basename(tmp_path, era5_like_catalog):
    rlz_nc = tmp_path / "rlz_1_cst_0.nc"
    rlz_nc.write_bytes(b"")  # the function uses str(fn.resolve()), not contents
    fn_out = tmp_path / "catalog.yml"

    pcdc.prepare_clim_data_catalog(
        fns=[rlz_nc],
        data_libs_like="dummy_catalog.yml",
        source_like="era5",
        fn_out=fn_out,
    )

    written = yaml.safe_load(fn_out.read_text())
    assert list(written.keys()) == ["rlz_1_cst_0"]


def test_multiple_fns_produce_one_entry_per_input(tmp_path, era5_like_catalog):
    fns = []
    for name in ["rlz_1_cst_0.nc", "rlz_2_cst_0.nc", "rlz_3_cst_0.nc"]:
        p = tmp_path / name
        p.write_bytes(b"")
        fns.append(p)
    fn_out = tmp_path / "catalog.yml"

    pcdc.prepare_clim_data_catalog(
        fns=fns,
        data_libs_like="dummy_catalog.yml",
        source_like="era5",
        fn_out=fn_out,
    )

    written = yaml.safe_load(fn_out.read_text())
    assert set(written.keys()) == {"rlz_1_cst_0", "rlz_2_cst_0", "rlz_3_cst_0"}


def test_driver_options_preprocess_is_harmonise_dims(tmp_path, era5_like_catalog):
    """The function MUST set driver.options.preprocess='harmonise_dims' on every
    entry, regardless of what the source_like driver had. This is what the
    yaml.safe_dump workaround protects against hydromt 1.3 silently stripping."""
    rlz_nc = tmp_path / "rlz_1_cst_0.nc"
    rlz_nc.write_bytes(b"")
    fn_out = tmp_path / "catalog.yml"

    pcdc.prepare_clim_data_catalog(
        fns=[rlz_nc],
        data_libs_like="dummy_catalog.yml",
        source_like="era5",
        fn_out=fn_out,
    )

    written = yaml.safe_load(fn_out.read_text())
    entry = written["rlz_1_cst_0"]
    assert entry["driver"]["name"] == "raster_xarray"
    assert entry["driver"]["options"]["preprocess"] == "harmonise_dims"
    assert entry["driver"]["options"]["lock"] is False


def test_data_adapter_and_root_are_dropped(tmp_path, era5_like_catalog):
    """The R-generated NCs are already in standard units, so unit_mult /
    rename adapters from the source must be discarded. Same for `root`,
    which would point at the wrong filesystem."""
    rlz_nc = tmp_path / "rlz_1_cst_0.nc"
    rlz_nc.write_bytes(b"")
    fn_out = tmp_path / "catalog.yml"

    pcdc.prepare_clim_data_catalog(
        fns=[rlz_nc],
        data_libs_like="dummy_catalog.yml",
        source_like="era5",
        fn_out=fn_out,
    )

    written = yaml.safe_load(fn_out.read_text())
    entry = written["rlz_1_cst_0"]
    assert "data_adapter" not in entry
    assert "root" not in entry


def test_uri_resolves_to_absolute_path(tmp_path, era5_like_catalog):
    rlz_nc = tmp_path / "rlz_1_cst_0.nc"
    rlz_nc.write_bytes(b"")
    fn_out = tmp_path / "catalog.yml"

    pcdc.prepare_clim_data_catalog(
        fns=[rlz_nc],
        data_libs_like="dummy_catalog.yml",
        source_like="era5",
        fn_out=fn_out,
    )

    written = yaml.safe_load(fn_out.read_text())
    assert Path(written["rlz_1_cst_0"]["uri"]).is_absolute()
```

### Step 3.3: Add metadata.processing tests

- [ ] Append:

```python
def test_processing_metadata_for_era5_source(tmp_path, era5_like_catalog):
    rlz_nc = tmp_path / "rlz_1_cst_0.nc"
    rlz_nc.write_bytes(b"")
    fn_out = tmp_path / "catalog.yml"

    pcdc.prepare_clim_data_catalog(
        fns=[rlz_nc],
        data_libs_like="dummy_catalog.yml",
        source_like="era5",
        fn_out=fn_out,
    )

    written = yaml.safe_load(fn_out.read_text())
    proc = written["rlz_1_cst_0"]["metadata"]["processing"]
    assert "era5" in proc
    assert "weathergenr" in proc
    assert "chirps" not in proc  # the era5-branch message


def test_processing_metadata_mentions_chirps_and_era5_for_chirps_source(tmp_path, chirps_like_catalog):
    rlz_nc = tmp_path / "rlz_1_cst_0.nc"
    rlz_nc.write_bytes(b"")
    fn_out = tmp_path / "catalog.yml"

    pcdc.prepare_clim_data_catalog(
        fns=[rlz_nc],
        data_libs_like="dummy_catalog.yml",
        source_like="chirps_global",
        fn_out=fn_out,
    )

    written = yaml.safe_load(fn_out.read_text())
    proc = written["rlz_1_cst_0"]["metadata"]["processing"]
    # The chirps branch explicitly notes both precip source and era5 fallback.
    assert "chirps_global" in proc
    assert "era5" in proc
```

### Step 3.4: Add chirps-orography test

- [ ] Append:

```python
def test_chirps_source_adds_orography_entry(tmp_path, chirps_like_catalog):
    """Chirps branch adds an extra entry named '<source>_orography' pointing at
    a sibling DEM file under climate_historical/raw_data/."""
    # The function looks for `../../climate_historical/raw_data/<src>_orography.nc`
    # relative to the FIRST input file, so we need to build that directory shape.
    raw = tmp_path / "experiment" / "climate_realizations" / "rlz_1"
    raw.mkdir(parents=True)
    rlz_nc = raw / "rlz_1_cst_0.nc"
    rlz_nc.write_bytes(b"")
    oro_dir = tmp_path / "experiment" / "climate_historical" / "raw_data"
    oro_dir.mkdir(parents=True)
    oro_fn = oro_dir / "chirps_global_orography.nc"
    oro_fn.write_bytes(b"")
    fn_out = tmp_path / "catalog.yml"

    pcdc.prepare_clim_data_catalog(
        fns=[rlz_nc],
        data_libs_like="dummy_catalog.yml",
        source_like="chirps_global",
        fn_out=fn_out,
    )

    written = yaml.safe_load(fn_out.read_text())
    assert "chirps_global_orography" in written
    oro = written["chirps_global_orography"]
    assert oro["data_type"] == "RasterDataset"
    assert oro["driver"]["name"] == "raster_xarray"
    assert oro["metadata"]["crs"] == 4326
```

### Step 3.5: Add the strict-xfail for hydromt to_yml regression

- [ ] Append:

```python
@pytest.mark.xfail(
    strict=True,
    reason=(
        "hydromt 1.3 silently strips driver.options.preprocess on "
        "DataCatalog().from_dict(...).to_yml(...). "
        "src/prepare_climate_data_catalog.py works around this with "
        "yaml.safe_dump. When upstream fixes to_yml, this test will pass, "
        "strict=True will fail CI, and the workaround can be removed. "
        "See dev/milestones/phase-1/m02b/handoff.md for the upstream reproducer."
    ),
)
def test_hydromt_to_yml_round_trip_preserves_preprocess(tmp_path):
    """Regression test for the M2b workaround.

    Imports the REAL hydromt to test the upstream bug. Marked xfail until
    upstream fixes it. When this passes, remove the yaml.safe_dump
    bypass in src/prepare_climate_data_catalog.py.
    """
    # Pop the stub so we get the real hydromt for this one test.
    sys.modules.pop("hydromt", None)
    import hydromt as real_hydromt  # noqa: E402

    catalog = {
        "test_source": {
            "data_type": "RasterDataset",
            "uri": "/tmp/dummy.nc",
            "driver": {
                "name": "raster_xarray",
                "options": {"preprocess": "harmonise_dims", "lock": False},
            },
        }
    }
    out_yml = tmp_path / "round_trip.yml"
    real_hydromt.DataCatalog().from_dict(catalog).to_yml(out_yml)
    written = yaml.safe_load(out_yml.read_text())

    # Restore stub for any later tests in the file.
    sys.modules["hydromt"] = types.SimpleNamespace(DataCatalog=_FakeDataCatalog)

    # The assertion that fails until upstream fix:
    assert (
        written["test_source"]["driver"]["options"]["preprocess"]
        == "harmonise_dims"
    )
```

### Step 3.6: Run new tests in isolation

- [ ] Run: `pixi run pytest tests/test_prepare_climate_data_catalog.py -v`
- [ ] Expect: 8 passed, 1 xfailed
- [ ] If `test_hydromt_to_yml_round_trip_preserves_preprocess` passes (no longer xfails), it means hydromt has been upgraded; report and pause — that's the trigger to remove the workaround.

### Step 3.7: Run full suite

- [ ] Run: `pixi run pytest tests/ -v`
- [ ] Expect: prior count + 8 passed + 1 xfailed = `40 passed, 3 xfailed`

### Step 3.8: Commit

- [ ] Run:
```
git add tests/test_prepare_climate_data_catalog.py
git commit -m "$(cat <<'EOF'
m02c: add unit tests for prepare_climate_data_catalog + xfail regression

Eight passing tests exercising the function's catalog construction:
single/multi fn naming, preprocess=harmonise_dims set, data_adapter/root
dropped, uri resolution, processing metadata for era5 vs chirps, chirps
orography entry. One strict-xfail regression for the underlying hydromt
1.3 to_yml/preprocess strip — xfail flips to pass when upstream is
fixed, forcing removal of the yaml.safe_dump workaround in src/.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `tests/test_extract_historical_climate.py`

**Files:**
- Create: `tests/test_extract_historical_climate.py`
- Target source: `src/extract_historical_climate.py` — function `prep_historical_climate`

**Mocking strategy:** Heaviest of the four. Stub `hydromt`, `geopandas`, `dask.diagnostics`, and `hydromt.model.processes.meteo` via `sys.modules.setdefault`. The fake `DataCatalog` records calls and returns fake datasets that quack like hydromt rasterdatasets enough to satisfy the function body.

**xfail this task introduces:** `test_warns_when_extracted_window_is_shorter_than_requested` — the M3 followup ("silent truncation"). Strict-xfailed until M3 adds the warning.

**Coverage scope:** This module is heavily I/O-coupled. We test the testable seams:
- The hydromt 1.x driver-options chunks="auto" patching
- Driver-as-string → driver-as-dict normalization
- Branching on `clim_source` (chirps precip-only vs era5 full stack)
- Variable list passed to `get_rasterdataset`
- xfail for truncation warning

Deeper hydromt reprojection / temp() lapse-correction logic is not unit-testable without absurd mocking; defer to integration tests in M3+.

### Step 4.1: Create the file with stubs

- [ ] Create `tests/test_extract_historical_climate.py`:

```python
"""Unit tests for src/extract_historical_climate.py.

This module is heavily coupled to hydromt I/O; we test the function's
configuration logic (driver options, variable lists, clim_source
branching) and skip the deeper reprojection paths. The truncation
warning xfail captures the M3 followup bug from dev/followups.md.
"""
from __future__ import annotations

import sys
import types
import warnings
from pathlib import Path

import pytest


# --- Stubs for heavy deps (set up BEFORE importing the source module) ---

class _FakeRasterAccessor:
    """Mimics ds.raster on an xarray-like dataset."""

    def __init__(self, vars_, box=None):
        self.vars = list(vars_)
        self.box = box if box is not None else object()

    def reproject_like(self, *_args, **_kwargs):
        return _FakeDataset(["dummy"])


class _FakeDataArray:
    def __init__(self, name):
        self.name = name


class _FakeDataset:
    """Quacks enough like an xarray Dataset for prep_historical_climate."""

    def __init__(self, vars_, time_size=None):
        self._vars = list(vars_)
        self.raster = _FakeRasterAccessor(vars_)
        self.time = types.SimpleNamespace(size=time_size or 100)
        self._tonetcdf_calls = []

    def __getitem__(self, key):
        return _FakeDataArray(key)

    def __setitem__(self, key, value):
        if key not in self._vars:
            self._vars.append(key)
            self.raster.vars.append(key)

    def to_dataset(self):
        return self

    def squeeze(self):
        return self

    def to_netcdf(self, fn, **kwargs):
        self._tonetcdf_calls.append((fn, kwargs))

        class _Delayed:
            def compute(self_inner):
                return None
        return _Delayed()


class _RecordingDataCatalog:
    """Records calls so tests can assert what was requested from hydromt."""

    _CATALOG = {}
    _LAST_INSTANCE = None

    def __init__(self, data_libs=None):
        self.data_libs = data_libs
        self.get_rasterdataset_calls = []
        type(self)._LAST_INSTANCE = self

    def to_dict(self):
        return {k: dict(v) for k, v in type(self)._CATALOG.items()}

    def from_dict(self, d):
        type(self)._CATALOG = d
        return self

    def get_rasterdataset(self, source, **kwargs):
        self.get_rasterdataset_calls.append({"source": source, **kwargs})
        # Return a dataset shaped to satisfy the function body.
        vars_ = kwargs.get("variables", ["precip"])
        return _FakeDataset(vars_)


def _fake_temp(*_args, **_kwargs):
    return _FakeDataArray("temp_corrected")


# Stub modules. NOTE: 'geopandas' is stubbed without its read_file; we patch
# at call time within the test using monkeypatch.

_geopandas_stub = types.SimpleNamespace(
    read_file=lambda fn: types.SimpleNamespace(
        geometry=types.SimpleNamespace(total_bounds=(0.0, 0.0, 1.0, 1.0)),
    ),
)
sys.modules.setdefault("geopandas", _geopandas_stub)

_hydromt_stub = types.SimpleNamespace(DataCatalog=_RecordingDataCatalog)
sys.modules.setdefault("hydromt", _hydromt_stub)

_meteo_stub = types.SimpleNamespace(temp=_fake_temp)
sys.modules.setdefault("hydromt.model", types.SimpleNamespace(processes=types.SimpleNamespace(meteo=_meteo_stub)))
sys.modules.setdefault("hydromt.model.processes", types.SimpleNamespace(meteo=_meteo_stub))
sys.modules.setdefault("hydromt.model.processes.meteo", _meteo_stub)

_dask_diag_stub = types.SimpleNamespace(ProgressBar=lambda: _NoopCtx())
class _NoopCtx:
    def __enter__(self): return self
    def __exit__(self, *exc): return False
sys.modules.setdefault("dask", types.SimpleNamespace(diagnostics=_dask_diag_stub))
sys.modules.setdefault("dask.diagnostics", _dask_diag_stub)


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import extract_historical_climate as ehc  # noqa: E402


@pytest.fixture
def fake_era5_catalog():
    _RecordingDataCatalog._CATALOG = {
        "era5": {
            "data_type": "RasterDataset",
            "uri": "/data/era5.nc",
            "driver": {"name": "netcdf", "options": {"chunks": "default"}},
        }
    }
    yield
    _RecordingDataCatalog._CATALOG = {}


@pytest.fixture
def fake_era5_string_driver_catalog():
    """Source where 'driver' is a bare string (older catalog format)."""
    _RecordingDataCatalog._CATALOG = {
        "era5": {
            "data_type": "RasterDataset",
            "uri": "/data/era5.nc",
            "driver": "netcdf",
        }
    }
    yield
    _RecordingDataCatalog._CATALOG = {}
```

### Step 4.2: Add era5-path tests (driver patching, variables)

- [ ] Append:

```python
def _last_catalog():
    return _RecordingDataCatalog._LAST_INSTANCE


def test_era5_path_requests_full_seven_variable_stack(tmp_path, fake_era5_catalog):
    region = tmp_path / "region.geojson"
    region.write_text("{}")  # contents irrelevant; geopandas stub ignores it
    out_nc = tmp_path / "out.nc"

    ehc.prep_historical_climate(
        region_fn=region,
        fn_out=out_nc,
        data_libs="dummy.yml",
        clim_source="era5",
        starttime="2010-01-01T00:00:00",
        endtime="2010-12-31T00:00:00",
    )

    calls = _last_catalog().get_rasterdataset_calls
    assert len(calls) == 1
    assert calls[0]["source"] == "era5"
    assert sorted(calls[0]["variables"]) == sorted(
        ["precip", "temp", "temp_min", "temp_max", "kin", "kout", "press_msl"]
    )


def test_era5_path_patches_driver_options_chunks_auto(tmp_path, fake_era5_catalog):
    region = tmp_path / "region.geojson"
    region.write_text("{}")
    out_nc = tmp_path / "out.nc"

    ehc.prep_historical_climate(
        region_fn=region,
        fn_out=out_nc,
        data_libs="dummy.yml",
        clim_source="era5",
        starttime="2010-01-01T00:00:00",
        endtime="2010-12-31T00:00:00",
    )

    # The function calls from_dict on a patched catalog. Inspect what was set.
    patched = _RecordingDataCatalog._CATALOG["era5"]
    assert patched["driver"]["options"]["chunks"] == "auto"


def test_era5_path_normalizes_string_driver_to_dict(tmp_path, fake_era5_string_driver_catalog):
    """When the source's 'driver' is a bare string, the function must wrap it
    in {'name': <str>} before adding options.chunks. Regression for hydromt
    1.x catalog format support."""
    region = tmp_path / "region.geojson"
    region.write_text("{}")
    out_nc = tmp_path / "out.nc"

    ehc.prep_historical_climate(
        region_fn=region,
        fn_out=out_nc,
        data_libs="dummy.yml",
        clim_source="era5",
        starttime="2010-01-01T00:00:00",
        endtime="2010-12-31T00:00:00",
    )

    patched = _RecordingDataCatalog._CATALOG["era5"]
    assert isinstance(patched["driver"], dict)
    assert patched["driver"]["name"] == "netcdf"
    assert patched["driver"]["options"]["chunks"] == "auto"
```

### Step 4.3: Add chirps-path test

- [ ] Append:

```python
@pytest.fixture
def fake_chirps_catalog():
    _RecordingDataCatalog._CATALOG = {
        "chirps_global": {"data_type": "RasterDataset", "uri": "/data/chirps.nc",
                          "driver": {"name": "netcdf"}},
        "era5": {"data_type": "RasterDataset", "uri": "/data/era5.nc",
                 "driver": {"name": "netcdf"}},
        "merit_hydro": {"data_type": "RasterDataset", "uri": "/data/merit.nc",
                        "driver": {"name": "netcdf"}},
        "era5_orography": {"data_type": "RasterDataset", "uri": "/data/era5_oro.nc",
                           "driver": {"name": "netcdf"}},
    }
    yield
    _RecordingDataCatalog._CATALOG = {}


def test_chirps_global_branch_requests_precip_only_from_chirps(tmp_path, fake_chirps_catalog):
    region = tmp_path / "region.geojson"
    region.write_text("{}")
    out_nc = tmp_path / "out.nc"

    ehc.prep_historical_climate(
        region_fn=region,
        fn_out=out_nc,
        data_libs="dummy.yml",
        clim_source="chirps_global",
        starttime="2010-01-01T00:00:00",
        endtime="2010-12-31T00:00:00",
    )

    calls = _last_catalog().get_rasterdataset_calls
    chirps_calls = [c for c in calls if c["source"] == "chirps_global"]
    era5_calls = [c for c in calls if c["source"] == "era5"]

    assert len(chirps_calls) == 1
    assert chirps_calls[0]["variables"] == ["precip"]
    # era5 fallback fetches the rest, but NOT precip.
    assert len(era5_calls) == 1
    assert "precip" not in era5_calls[0]["variables"]
    assert "temp" in era5_calls[0]["variables"]
```

### Step 4.4: Add starttime/endtime propagation test

- [ ] Append:

```python
def test_starttime_and_endtime_passed_to_get_rasterdataset(tmp_path, fake_era5_catalog):
    """The function MUST pass its starttime/endtime params through to hydromt.
    Note: this tests the FUNCTION's behavior, not the Snakefile rule that
    invokes it. The rule-level bug (Snakefile_climate_experiment hardcoding
    dates) is separately tracked in dev/followups.md M5 and belongs to an
    integration test, not this unit."""
    region = tmp_path / "region.geojson"
    region.write_text("{}")
    out_nc = tmp_path / "out.nc"

    ehc.prep_historical_climate(
        region_fn=region,
        fn_out=out_nc,
        data_libs="dummy.yml",
        clim_source="era5",
        starttime="1995-06-15T00:00:00",
        endtime="2005-06-15T00:00:00",
    )

    calls = _last_catalog().get_rasterdataset_calls
    assert calls[0]["time_range"] == (
        "1995-06-15T00:00:00", "2005-06-15T00:00:00",
    )
```

### Step 4.5: Add the strict-xfail for the truncation warning

- [ ] Append:

```python
@pytest.mark.xfail(
    strict=True,
    reason=(
        "M3 followup (dev/followups.md): when the staged source covers a "
        "shorter time window than (starttime, endtime), prep_historical_climate "
        "silently produces a truncated netCDF without warning. Should emit a "
        "warnings.warn(...) when ds.time.size's date range is shorter than "
        "the requested span. xfail until M3 adds the check."
    ),
)
def test_warns_when_extracted_window_is_shorter_than_requested(
    tmp_path, fake_era5_catalog, monkeypatch
):
    """Drive a fake DataCatalog whose datasets have a narrow time dimension.
    Currently no warning is raised → xfail. M3 should add a warning that this
    test then verifies. monkeypatch updates ehc.hydromt.DataCatalog directly
    because `import hydromt` in the source module binds at import time —
    rewriting sys.modules['hydromt'] later does not affect that binding."""

    class _NarrowDataCatalog(_RecordingDataCatalog):
        def get_rasterdataset(self, source, **kwargs):
            self.get_rasterdataset_calls.append({"source": source, **kwargs})
            return _FakeDataset(kwargs.get("variables", ["precip"]), time_size=10)

    monkeypatch.setattr(ehc.hydromt, "DataCatalog", _NarrowDataCatalog)

    region = tmp_path / "region.geojson"
    region.write_text("{}")
    out_nc = tmp_path / "out.nc"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ehc.prep_historical_climate(
            region_fn=region,
            fn_out=out_nc,
            data_libs="dummy.yml",
            clim_source="era5",
            starttime="2000-01-01T00:00:00",
            endtime="2020-12-31T00:00:00",  # 21 years requested, ds returns 10 timesteps
        )

    assert any(
        "truncat" in str(w.message).lower() or "shorter" in str(w.message).lower()
        for w in caught
    ), "Expected a warning about time-window truncation; got none."
```

### Step 4.6: Run new tests in isolation

- [ ] Run: `pixi run pytest tests/test_extract_historical_climate.py -v`
- [ ] Expect: 4 passed, 1 xfailed
- [ ] If the xfail starts passing, M3 has added the warning — report and pause; remove the marker.

### Step 4.7: Run full suite

- [ ] Run: `pixi run pytest tests/ -v`
- [ ] Expect: prior count + 4 passed + 1 xfailed = `44 passed, 4 xfailed`

### Step 4.8: Commit

- [ ] Run:
```
git add tests/test_extract_historical_climate.py
git commit -m "$(cat <<'EOF'
m02c: add unit tests for extract_historical_climate + xfail truncation

Four passing tests covering the era5 path's variable stack and driver-
options chunks="auto" patching (incl. driver-as-string normalization),
the chirps-global precip-only branch, and starttime/endtime propagation
to hydromt.get_rasterdataset. One strict-xfail captures the M3 followup
where the function silently truncates without warning when the staged
source covers a shorter window than requested.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Final verification and seal M02c

### Step 5.1: Verify the full suite

- [ ] Run: `pixi run pytest tests/ -v`
- [ ] Expect: `44 passed, 4 xfailed` (the original `13 passed, 2 xfailed` + 31 new passes + 2 new xfails). Adjust counts if your numbers differ.
- [ ] Cross-check the xfail count: `pixi run pytest tests/ -v | grep -i xfail | wc -l` should report 4.

### Step 5.2: Lint-check the new test files

- [ ] Run: `pixi run python -c "import ast, pathlib; [ast.parse(p.read_text()) for p in pathlib.Path('tests').glob('test_*.py')]"` → no output = all parse cleanly.

### Step 5.3: Seal M02c in the roadmap

- [ ] Edit `dev/roadmap.md`. Find the M2c section header:
  ```markdown
  ## M2c — Test coverage (pre-M3)

  **Goal.** Establish unit-test infrastructure...
  ```
- [ ] Insert a status line between the heading and the goal:
  ```markdown
  ## M2c — Test coverage (pre-M3)

  **Status.** Sealed YYYY-MM-DD — four new test files, 31 new passing
  tests, 2 strict-xfails for documented bugs (hydromt to_yml preprocess
  strip; extract_historical_climate truncation warning).

  **Goal.** Establish unit-test infrastructure...
  ```
  Replace `YYYY-MM-DD` with today's date.

### Step 5.4: Tag the milestone

- [ ] Run: `git status --short` → expect only `dev/roadmap.md` modified.
- [ ] Commit the roadmap update:
  ```
  git add dev/roadmap.md
  git commit -m "$(cat <<'EOF'
  m02c: mark milestone sealed in roadmap

  Four new test files, 31 passing tests, 2 strict xfails. Pattern
  established for M3-M5 to inherit.

  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  EOF
  )"
  ```
- [ ] Tag: `git tag -a m02c-tests -m "m02c-tests: pre-M3 unit-test coverage for stable utilities"`
- [ ] Verify: `git tag -l "m02c*"` → `m02c-tests`

### Step 5.5: Report

- [ ] Summarize to the user:
  - Branch `milestone/02c-tests` complete, tagged `m02c-tests`.
  - `pixi run pytest tests/` → `44 passed, 4 xfailed`.
  - 4 new test files; xfails to watch as M3 progresses.

---

## Notes for the executing engineer

- **If a test fails for a reason other than the bug it's trying to catch** (e.g. an import error, a typo in a fixture), fix the test, not the source — M02c does not modify any `src/` module. Document any such fix in the commit body.
- **If a test reveals what looks like a real bug not covered by a planned xfail**, stop and surface it. Don't extend M02c scope unilaterally; that's a followups.md addition.
- **If the mocking gets out of hand on Task 4** (extract_historical_climate), the spec's fallback applies: drop the failing test for now, note it in `dev/followups.md` under M3, and continue. Better to seal M02c with 30 passing tests than to chase a brittle mock.
- **Tag placement.** Tag the milestone after Step 5.3's commit — that's the validated state. Don't tag earlier commits.

## Quick reference: file → test count → xfails

| Test file                                      | Tests | xfails |
| ---------------------------------------------- | ----- | ------ |
| `tests/test_metrics_definition.py`             |  6    | 0      |
| `tests/test_setup_time_horizon.py`             | 13    | 0      |
| `tests/test_prepare_climate_data_catalog.py`   |  8    | 1      |
| `tests/test_extract_historical_climate.py`     |  4    | 1      |
| **Total new**                                  | **31** | **2** |

Existing baseline (`13 passed, 2 xfailed`) → final state (`44 passed, 4 xfailed`).
