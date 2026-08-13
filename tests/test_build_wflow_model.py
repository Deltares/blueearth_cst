"""Focused tests for the P1-to-Wflow public adapter boundary."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from shapely.geometry import Point

from blueearth_cst.model.build_wflow_model import (
    _BASE_CONFIG,
    _apply_parameter_steps,
    _record_kwargs,
    _step_call_kwargs,
    _validate_registry,
    arcgis_d8_to_wflow_ldd,
    read_parameter_steps,
)


def _template(path: Path, steps: list[dict]) -> Path:
    path.write_text(
        yaml.safe_dump({"modeltype": "wflow_sbm", "steps": steps}),
        encoding="utf-8",
    )
    return path


def _no_rivers():
    return gpd.GeoDataFrame(geometry=[], crs=4326)


def _p1_grid(lulc_source="vito", soil_source="soilgrids", river_upa=32.0):
    """A minimal P1 grid, with or without its stamped source attrs.

    ``river_upa=None`` omits the threshold attribute, and either source set to
    ``None`` omits that attr -- the three ways a grid can violate
    ``blueearth-cst-spatial-v1`` and so the three refusals worth pinning.
    """
    coords = {"y": [1.5, 0.5], "x": [0.5, 1.5]}
    shape = (2, 2)
    attrs = {}
    if lulc_source is not None:
        attrs["lulc_source"] = lulc_source
    if soil_source is not None:
        attrs["soil_source"] = soil_source
    river_attrs = (
        {} if river_upa is None else {"upstream_area_threshold_km2": river_upa}
    )
    maps = xr.Dataset(
        {
            "land_cover": (("y", "x"), np.ones(shape, dtype="int16")),
            "river_mask": (
                ("y", "x"),
                np.ones(shape, dtype="uint8"),
                river_attrs,
            ),
            # The four layers `_p1_hydrography` assembles for setup_rivers.
            "flow_direction": (("y", "x"), np.ones(shape, dtype="uint8")),
            "basin_id": (("y", "x"), np.ones(shape, dtype="int32")),
            "upstream_area": (("y", "x"), np.ones(shape)),
            "elevation": (("y", "x"), np.ones(shape)),
            "leaf_area_index": (
                ("month", "y", "x"),
                np.ones((12, *shape), dtype="float32"),
            ),
        },
        coords={**coords, "month": range(1, 13)},
        attrs=attrs,
    )
    maps.raster.set_crs(4326)
    return maps


def test_parameter_template_rejects_setup_basemaps(tmp_path):
    path = _template(tmp_path / "build.yml", [{"setup_basemaps": {}}])

    with pytest.raises(ValueError, match="forbidden after P1"):
        read_parameter_steps(path)


def test_direct_base_maps_have_required_toml_mapping():
    assert _BASE_CONFIG["input"]["static"]["land_surface__slope"] == "land_slope"


def test_shipped_template_retains_only_wflow_parameter_steps():
    path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "defaults"
        / "wflow_build_model.yml"
    )

    steps = read_parameter_steps(path)

    assert [name for name, _ in steps] == [
        "setup_rivers",
        "setup_lulcmaps",
        "setup_laimaps",
        "setup_soilmaps",
        "setup_constant_pars",
    ]


def test_arcgis_d8_conversion_preserves_grid_and_active_domain():
    flow = xr.DataArray(
        np.array([[1, 4], [64, 0]], dtype="uint8"),
        dims=("y", "x"),
        coords={"y": [1.5, 0.5], "x": [0.5, 1.5]},
    )
    basin = xr.DataArray(
        np.array([[1, 1], [1, 0]], dtype="int32"),
        coords=flow.coords,
        dims=flow.dims,
    )
    flow.raster.set_crs(4326)
    basin.raster.set_crs(4326)

    ldd = arcgis_d8_to_wflow_ldd(flow, basin)

    assert ldd.raster.shape == flow.raster.shape
    assert ldd.raster.bounds == flow.raster.bounds
    assert ldd.raster.crs.to_epsg() == 4326
    assert int(ldd.values[1, 1]) == 255
    assert set(np.unique(ldd.values[:1])).issubset(set(range(1, 10)))


def test_registry_requires_primary_ids_to_end_in_zero():
    registry = pd.DataFrame(
        {
            "subbasin_id": [101],
            "wflow_id": [999],
            "location_id": [1],
            "location_code": ["B001-S01-L01"],
            "location_role": ["control"],
        }
    )
    locations = gpd.GeoDataFrame(registry.copy(), geometry=[Point(1, 1)], crs=4326)

    with pytest.raises(ValueError, match="must end in 0"):
        _validate_registry(registry, locations)


def test_parameter_steps_use_p1_objects_and_keep_soil_catalog_owned():
    calls: list[tuple[str, dict]] = []

    class FakeModel:
        def setup_rivers(self, **kwargs):
            calls.append(("setup_rivers", kwargs))

        def setup_lulcmaps(self, **kwargs):
            calls.append(("setup_lulcmaps", kwargs))

        def setup_laimaps(self, **kwargs):
            calls.append(("setup_laimaps", kwargs))

        def setup_soilmaps(self, **kwargs):
            calls.append(("setup_soilmaps", kwargs))

        def setup_constant_pars(self, **kwargs):
            calls.append(("setup_constant_pars", kwargs))

    coords = {"y": [1.5, 0.5], "x": [0.5, 1.5]}
    shape = (2, 2)
    maps = xr.Dataset(
        {
            "flow_direction": (("y", "x"), np.ones(shape, dtype="uint8")),
            "basin_id": (("y", "x"), np.ones(shape, dtype="int32")),
            "upstream_area": (("y", "x"), np.ones(shape)),
            "elevation": (("y", "x"), np.ones(shape)),
            "land_cover": (("y", "x"), np.ones(shape, dtype="int16")),
            "leaf_area_index": (
                ("month", "y", "x"),
                np.ones((12, *shape), dtype="float32"),
            ),
            "river_mask": (
                ("y", "x"),
                np.ones(shape, dtype="uint8"),
                {"upstream_area_threshold_km2": 32.0},
            ),
        },
        coords={**coords, "month": range(1, 13)},
        attrs={"lulc_source": "vito", "soil_source": "soilgrids"},
    )
    maps.raster.set_crs(4326)
    rivers = gpd.GeoDataFrame(geometry=[], crs=4326)
    steps = [
        ("setup_rivers", {"hydrography_fn": "global", "river_upa": 32}),
        ("setup_lulcmaps", {"lulc_fn": "vito"}),
        ("setup_laimaps", {"lai_fn": "modis_lai"}),
        ("setup_soilmaps", {"soil_fn": "soilgrids"}),
        ("setup_constant_pars", {"snowpack__degree_day_coefficient": 3.7}),
    ]

    _apply_parameter_steps(FakeModel(), steps, maps, rivers)

    by_name = dict(calls)
    assert isinstance(by_name["setup_rivers"]["hydrography_fn"], xr.Dataset)
    assert by_name["setup_rivers"]["hydrography_fn"]["flwdir"].dtype == np.uint8
    assert by_name["setup_rivers"]["hydrography_fn"]["flwdir"].raster.nodata == 247
    assert by_name["setup_rivers"]["river_geom_fn"] is rivers
    assert isinstance(by_name["setup_lulcmaps"]["lulc_fn"], xr.DataArray)
    assert by_name["setup_lulcmaps"]["lulc_mapping_fn"] == "vito_mapping_default"
    assert isinstance(by_name["setup_laimaps"]["lai_fn"], xr.DataArray)
    assert by_name["setup_rivers"]["river_upa"] == 32.0
    assert by_name["setup_soilmaps"] == {"soil_fn": "soilgrids"}


def test_lulc_mapping_follows_P1_not_the_template():
    """Defect H: the mapping table must name the source the RASTER came from.

    Until 2026-08-13 the source name was taken from the template's `lulc_fn`,
    while the raster came from P1 — so a project selecting `corine` got CORINE
    land cover interpreted through `vito_mapping_default`. Wrong numbers, not a
    missing setting, which is why this is pinned rather than left to the
    template's absence.
    """
    calls: list[tuple[str, dict]] = []

    class FakeModel:
        def setup_lulcmaps(self, **kwargs):
            calls.append(("setup_lulcmaps", kwargs))

    maps = _p1_grid(lulc_source="corine")
    # A stale template still naming vito must NOT win.
    _apply_parameter_steps(
        FakeModel(), [("setup_lulcmaps", {"lulc_fn": "vito"})], maps, _no_rivers()
    )
    assert dict(calls)["setup_lulcmaps"]["lulc_mapping_fn"] == "corine_mapping_default"


def test_a_P1_grid_without_its_source_attr_is_refused():
    """A silent literal fallback is how the template's copy came to win.

    `blueearth-cst-spatial-v1` guarantees the attr, so its absence is a
    contract violation worth a loud failure rather than a guess at "vito".
    """

    class FakeModel:
        def setup_lulcmaps(self, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("should have refused before calling the model")

    maps = _p1_grid(lulc_source=None)
    with pytest.raises(ValueError, match="lulc_source"):
        _apply_parameter_steps(
            FakeModel(), [("setup_lulcmaps", {})], maps, _no_rivers()
        )


def test_river_threshold_follows_P1_not_the_template():
    """Defect H: one physical threshold, one owner.

    `river_upa` reached hydromt intact while `shared.basin.river_uparea_km2`
    drove P1's delineation, and nothing coupled them. Both default to 32, so a
    project moving one got a wflow river map and a P1 river mask that disagree
    about which cells are river -- with nothing reporting it.
    """
    calls: list[tuple[str, dict]] = []

    class FakeModel:
        def setup_rivers(self, **kwargs):
            calls.append(("setup_rivers", kwargs))

    maps = _p1_grid(river_upa=80.0)
    # A template still naming the old 32 must NOT win.
    _apply_parameter_steps(
        FakeModel(), [("setup_rivers", {"river_upa": 32})], maps, _no_rivers()
    )

    assert dict(calls)["setup_rivers"]["river_upa"] == 80.0


def test_soil_source_follows_P1_not_the_template():
    """Defect H: the model and the basin report must describe one dataset.

    hydromt reads the soil data itself, so `soil_fn` is a source NAME rather
    than a raster -- but P1 resamples the same choice into the `soil_*` maps
    rule 1.12 plots. Left free, the template could name a source the figures
    never showed.
    """
    calls: list[tuple[str, dict]] = []

    class FakeModel:
        def setup_soilmaps(self, **kwargs):
            calls.append(("setup_soilmaps", kwargs))

    maps = _p1_grid(soil_source="soilgrids_2020")
    _apply_parameter_steps(
        FakeModel(), [("setup_soilmaps", {"soil_fn": "soilgrids"})], maps, _no_rivers()
    )

    assert dict(calls)["setup_soilmaps"]["soil_fn"] == "soilgrids_2020"


def test_a_P1_grid_without_its_river_threshold_is_refused():
    """Same discipline as the source attrs: refuse rather than assume 32."""

    class FakeModel:
        def setup_rivers(self, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("should have refused before calling the model")

    maps = _p1_grid(river_upa=None)
    with pytest.raises(ValueError, match="upstream_area_threshold_km2"):
        _apply_parameter_steps(FakeModel(), [("setup_rivers", {})], maps, _no_rivers())


def test_the_record_shows_the_DERIVED_mapping_not_the_template_value():
    """The falsifier for "the record equals what hydromt received".

    R3 asks for the actual values used, and the template is not it:
    `lulc_mapping_fn` is derived at call time and appears in no file on disk.
    The decisive case is the defect this repo actually shipped -- a template
    naming `vito` while P1 supplied `corine` -- so a record built from the
    template would have shown `vito_mapping_default` for a run that used
    `corine_mapping_default`. Wrong numbers, recorded as if correct.
    """

    class FakeModel:
        def setup_lulcmaps(self, **kwargs):
            pass

    maps = _p1_grid(lulc_source="corine")
    records = _apply_parameter_steps(
        FakeModel(), [("setup_lulcmaps", {"lulc_fn": "vito"})], maps, _no_rivers()
    )

    recorded = dict(records)["setup_lulcmaps"]
    assert recorded["lulc_mapping_fn"] == "corine_mapping_default"
    # The template key that lost must not survive into the record either: it
    # never reached hydromt, so recording it would describe a call that did
    # not happen.
    assert "lulc_fn" not in recorded or recorded["lulc_fn"] != "vito"


def test_the_record_describes_injected_P1_objects_by_reference():
    """An xarray repr is not provenance -- it is unstable and unreadable.

    The record has to say WHICH P1 product was handed over, so a reader can go
    find it, and it has to stay diffable between two runs.
    """

    class FakeModel:
        def setup_rivers(self, **kwargs):
            pass

    records = dict(
        _apply_parameter_steps(
            FakeModel(),
            [("setup_rivers", {"hydrography_fn": "global", "min_rivwth": 30})],
            _p1_grid(),
            _no_rivers(),
        )
    )

    recorded = records["setup_rivers"]
    assert recorded["hydrography_fn"] == {
        "injected_from": "p1_spatial_maps",
        "product": "hydrography",
    }
    assert recorded["river_geom_fn"] == {
        "injected_from": "p1_spatial_catalog",
        "product": "rivers",
    }
    # Plain configured values pass through untouched, and the coupled
    # threshold is recorded as the number hydromt actually got.
    assert recorded["min_rivwth"] == 30
    assert recorded["river_upa"] == 32.0


def test_the_record_is_yaml_serializable():
    """It is written as YAML, so an unserializable value fails the rule.

    Cheap to assert here and expensive to discover in a run: safe_dump raises
    on any object it does not know, which would be a mid-build crash rather
    than a bad record.
    """

    class FakeModel:
        def setup_rivers(self, **kwargs):
            pass

        def setup_lulcmaps(self, **kwargs):
            pass

    records = _apply_parameter_steps(
        FakeModel(),
        [("setup_rivers", {}), ("setup_lulcmaps", {})],
        _p1_grid(),
        _no_rivers(),
    )

    dumped = yaml.safe_dump({name: kwargs for name, kwargs in records})

    assert "injected_from" in dumped


def test_the_SHIPPED_template_produces_a_serializable_record():
    """The fixture cannot reach this path, and safe_dump raises mid-build.

    The record is written after the setup steps have already mutated the
    model, so an unserializable value is not a bad record -- it is a crash
    with a half-built model on disk. The shipped template's steps are the ones
    that actually run, including `setup_constant_pars`' thirteen CSDMS-named
    floats through the untouched `else` branch, so they are what must be
    proven dumpable.
    """
    template = (
        Path(__file__).resolve().parents[1] / "config/defaults/wflow_build_model.yml"
    )
    maps = _p1_grid()

    records = [
        (name, _record_kwargs(_step_call_kwargs(name, configured, maps, _no_rivers())))
        for name, configured in read_parameter_steps(template)
    ]

    dumped = yaml.safe_dump([{name: kwargs} for name, kwargs in records])

    assert "lulc_mapping_fn" in dumped
    assert yaml.safe_load(dumped) is not None


def test_the_shipped_template_declares_neither_coupled_key():
    """The loser must be ABSENT, not merely ignored.

    A key that is popped and discarded still reads as a setting to whoever
    edits the file -- the whole defect class this bundle closes.
    """
    template = (
        Path(__file__).resolve().parents[1] / "config/defaults/wflow_build_model.yml"
    )
    text = template.read_text(encoding="utf-8")

    for key in ("river_upa:", "soil_fn:", "lulc_fn:", "lai_fn:"):
        assert key not in text, f"{key} is coupled to P1 and must not be declared"
