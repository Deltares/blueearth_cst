"""Named models across WF1, WF4, configuration and layout (t2609281102)."""

import pytest

from blueearth_cst.model.registered_model import missing_output_declarations
from blueearth_cst.shared import config_composition as cc
from blueearth_cst.shared import layout_paths as L
from blueearth_cst.shared.model_registry import (
    ModelSelectionError,
    build_record,
    recipe_key,
    selected_model_root,
    write_record,
)
from tests.test_config_composition import MINIMAL_T1_SECTIONS, write_split

_COLUMNS = """
[output.csv]
path = "output.csv"

[[output.csv.column]]
header = "Q"
map = "outlets"
parameter = "river_water__volume_flow_rate"

[[output.csv.column]]
header = "Q"
map = "gauges_locations"
parameter = "river_water__volume_flow_rate"

[[output.csv.column]]
header = "gwr"
map = "subcatchment"
parameter = "soil_water_saturated_zone_top__net_recharge_volume_flux"
reducer = "mean"
"""


def _toml(tmp_path, body=_COLUMNS):
    path = tmp_path / "wflow_sbm.toml"
    path.write_text('[input]\npath_static = "staticmaps.nc"\n' + body, encoding="utf-8")
    return path


def test_output_declarations(tmp_path):
    toml = _toml(tmp_path)
    outvars = ["river discharge", "groundwater recharge"]
    assert missing_output_declarations(toml, outvars, None) == []
    missing = missing_output_declarations(toml, [*outvars, "precipitation"], {"x": 1})
    assert missing == [
        "output.csv column header='p' map='subcatchment'",
        "output.netcdf_grid (build_model.output requests a grid)",
    ]


def _model(project, name, origin):
    root = project / "models" / "hydrology" / name
    root.mkdir(parents=True)
    _toml(root)
    (root / "staticmaps.nc").write_bytes(b"S")
    write_record(root, build_record(root, name=name, origin=origin))
    return root


def test_recipe_key_and_selection(tmp_path):
    built = _model(tmp_path, "wflow", "built")
    assert selected_model_root(tmp_path, {}) == built
    assert recipe_key(built, None) is None
    registered = _model(tmp_path, "wflow_calibrated", "user_managed")
    assert recipe_key(registered, None) == "default"
    assert recipe_key(registered, "era5") == "era5"
    with pytest.raises(ModelSelectionError, match="several models"):
        selected_model_root(tmp_path, {"model": {}})
    assert (
        selected_model_root(tmp_path, {"model": {"name": "wflow_calibrated"}})
        == registered
    )


def test_model_name_coexists_with_the_scenario_set_name(tmp_path):
    sections = {**MINIMAL_T1_SECTIONS, "model": {"name": "wflow_calibrated"}}
    path = write_split(
        tmp_path, sections=sections, bodies={"generate_scenarios": {"name": "dry"}}
    )
    config = cc.load_composed_config(path)
    assert config["model"]["name"] == "wflow_calibrated"


def test_bind_covers_every_model():
    bound = L.bind(
        L.declarations(),
        clim_project="cmip6",
        experiments=["a"],
        model_name=["wflow", "wflow_calibrated"],
    )
    patterns = [pattern for _, pattern in bound]
    for path in (
        "models/hydrology/wflow/_engine/model.json",
        "models/hydrology/wflow_calibrated/wflow_sbm_default.toml",
        "models/hydrology/comparisons/calibration_effect/tables/daily.csv",
    ):
        assert any(p.fullmatch(path) for p in patterns), path
