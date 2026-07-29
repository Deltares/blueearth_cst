"""Unit tests for blueearth_cst/shared/setup_time_horizon.py — forcing config YAML builder.

Note: module name is misleading. The function builds a hydromt update YAML
for adding forcing to a wflow model; testable surface is the chunksize
branching by staticmaps size and the precip-source branching (era5/eobs).
R3 may rename the module; tests pin behavior, not names.
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

    _NEXT_SIZE = 0

    def __init__(self, root, mode):
        # Read the size pre-set by the test on this class attribute.
        size = type(self)._NEXT_SIZE
        self.staticmaps = _FakeStaticmaps(size)


sys.modules.setdefault(
    "hydromt_wflow",
    types.SimpleNamespace(WflowSbmModel=_FakeWflowSbmModel),
)

from blueearth_cst.shared import setup_time_horizon  # noqa: E402


def _read_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


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
