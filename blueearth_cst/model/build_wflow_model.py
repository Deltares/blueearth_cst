"""Build Wflow-SBM parameters from the engine-neutral spatial foundation."""

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from hydromt.gis import flw
from pyflwdir import core_d8, core_ldd

_BASE_CONFIG = {
    "input": {
        "path_static": "staticmaps.nc",
        "basin__local_drain_direction": "local_drain_direction",
        "subbasin_location__count": "subcatchment",
        "river_location__mask": "river_mask",
        "static": {"land_surface__slope": "land_slope"},
    }
}
_SUPPORTED_PARAMETER_STEPS = {
    "setup_rivers",
    "setup_lulcmaps",
    "setup_laimaps",
    "setup_soilmaps",
    "setup_constant_pars",
}


def _catalog_paths(value: str | os.PathLike[str] | Sequence[object]) -> list[str]:
    """Normalize one or several Snakemake catalog inputs."""
    if isinstance(value, (str, os.PathLike)):
        return [os.fspath(value)]
    return [os.fspath(item) for item in value]


def read_parameter_steps(
    path: str | os.PathLike[str],
) -> list[tuple[str, dict[str, Any]]]:
    """Read the Wflow-only setup steps and reject competing domain setup."""
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, Mapping) or config.get("modeltype") != "wflow_sbm":
        raise ValueError("Wflow parameter template must declare modeltype: wflow_sbm")
    raw_steps = config.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("Wflow parameter template must contain a non-empty steps list")

    steps: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for item in raw_steps:
        if not isinstance(item, Mapping) or len(item) != 1:
            raise ValueError("Each Wflow parameter step must be a single-key mapping")
        name, kwargs = next(iter(item.items()))
        if name == "setup_basemaps":
            raise ValueError(
                "setup_basemaps is forbidden after P1; build_wflow_model must consume "
                "the generated spatial foundation"
            )
        if name not in _SUPPORTED_PARAMETER_STEPS:
            raise ValueError(f"Unsupported Wflow parameter step: {name}")
        if name in seen:
            raise ValueError(f"Duplicate Wflow parameter step: {name}")
        seen.add(name)
        if kwargs is None:
            kwargs = {}
        if not isinstance(kwargs, Mapping):
            raise ValueError(f"Step {name} arguments must be a mapping")
        steps.append((str(name), dict(kwargs)))
    return steps


def arcgis_d8_to_wflow_ldd(
    flow_direction: xr.DataArray, basin_id: xr.DataArray
) -> xr.DataArray:
    """Convert the neutral ArcGIS D8 grid to Wflow's LDD encoding."""
    active = np.isfinite(basin_id.values) & (basin_id.values > 0)
    values = np.asarray(flow_direction.values)
    valid_codes = {0, 1, 2, 4, 8, 16, 32, 64, 128}
    active_codes = {int(value) for value in np.unique(values[active])}
    if not active_codes.issubset(valid_codes):
        raise ValueError(
            f"P1 flow_direction contains invalid ArcGIS D8 codes: {active_codes}"
        )
    d8 = xr.DataArray(
        np.where(active, values, core_d8._mv).astype("uint8"),
        coords=flow_direction.coords,
        dims=flow_direction.dims,
        name="flow_direction",
    )
    d8.raster.set_crs(flow_direction.raster.crs)
    d8.raster.set_nodata(core_d8._mv)
    active_da = xr.DataArray(active, coords=d8.coords, dims=d8.dims)
    flow_network = flw.flwdir_from_da(d8, ftype="d8", mask=active_da)
    ldd = xr.DataArray(
        flow_network.to_array(ftype="ldd").astype("uint8"),
        coords=d8.coords,
        dims=d8.dims,
        name="local_drain_direction",
        attrs={"long_name": "Wflow LDD flow direction", "_FillValue": core_ldd._mv},
    )
    ldd.raster.set_crs(flow_direction.raster.crs)
    ldd.raster.set_nodata(core_ldd._mv)
    return ldd


def _static_base_maps(maps: xr.Dataset) -> xr.Dataset:
    """Translate P1 neutral base layers to the Wflow component vocabulary."""
    required = {
        "flow_direction",
        "basin_id",
        "subbasin_id",
        "upstream_area",
        "river_order",
        "cell_area",
        "elevation",
        "slope",
        "river_mask",
    }
    missing = sorted(required.difference(maps.data_vars))
    if missing:
        raise ValueError(f"P1 spatial_maps lacks required Wflow base layers: {missing}")
    translated = xr.Dataset(
        {
            "local_drain_direction": arcgis_d8_to_wflow_ldd(
                maps["flow_direction"], maps["basin_id"]
            ),
            "subcatchment": maps["subbasin_id"].rename("subcatchment"),
            "meta_upstream_area": maps["upstream_area"].rename("meta_upstream_area"),
            "meta_streamorder": maps["river_order"].rename("meta_streamorder"),
            "meta_subgrid_area": maps["cell_area"].rename("meta_subgrid_area"),
            "land_elevation": maps["elevation"].rename("land_elevation"),
            "land_slope": maps["slope"].rename("land_slope"),
            "river_mask": maps["river_mask"].rename("river_mask"),
        }
    )
    translated.raster.set_crs(maps.raster.crs)
    return translated


def _validate_registry(registry: pd.DataFrame, locations: gpd.GeoDataFrame) -> None:
    """Validate P1 identities before exposing them to HydroMT-Wflow."""
    required = {
        "subbasin_id",
        "wflow_id",
        "location_id",
        "location_code",
        "location_role",
    }
    missing = sorted(required.difference(registry.columns))
    if missing:
        raise ValueError(f"P1 location_registry lacks columns: {missing}")
    if registry["wflow_id"].duplicated().any():
        raise ValueError("P1 location_registry contains duplicate wflow_id values")
    # ADR 0003 §12a: the old invariant here was "every primary location
    # wflow_id must equal its subbasin_id". That is REPEALED, not broken —
    # under §12 a primary is `basin_id*1000 + local_subbasin_number*10` while
    # its subbasin_id is `basin_id*100 + local_subbasin_number`, so the two are
    # deliberately different numbers and the old check made WF1 unbuildable.
    #
    # Replaced rather than deleted, because the property it protected is real:
    # a primary must be identifiable from its id alone. Under §12 that reads as
    # "ends in 0", which is also what tells a reader that 1010 is a subbasin
    # outlet and 1011 a gauge inside it.
    primary = registry[registry["location_id"].astype(int).eq(1)]
    misnumbered = primary.loc[
        primary["wflow_id"].astype(int) % 10 != 0, "location_code"
    ]
    if not misnumbered.empty:
        raise ValueError(
            "every primary location wflow_id must end in 0 (ADR 0003 §12: "
            "basin_id*1000 + local_subbasin_number*10 + m, m=0 for the "
            f"primary); offenders: {sorted(misnumbered.astype(str))}"
        )
    if set(locations["wflow_id"].astype(int)) != set(registry["wflow_id"].astype(int)):
        raise ValueError(
            "P1 locations geometry and location_registry wflow_id values disagree"
        )


def _p1_hydrography(maps: xr.Dataset) -> xr.Dataset:
    """Expose the public setup_rivers input names without changing the P1 grid."""
    active = np.isfinite(maps["basin_id"].values) & (maps["basin_id"].values > 0)
    flwdir = xr.DataArray(
        np.where(active, maps["flow_direction"].values, core_d8._mv).astype("uint8"),
        coords=maps["flow_direction"].coords,
        dims=maps["flow_direction"].dims,
        name="flwdir",
    )
    flwdir.raster.set_crs(maps.raster.crs)
    flwdir.raster.set_nodata(core_d8._mv)
    hydrography = xr.Dataset(
        {
            "flwdir": flwdir,
            "uparea": maps["upstream_area"].rename("uparea"),
            "elevtn": maps["elevation"].rename("elevtn"),
        }
    )
    hydrography.raster.set_crs(maps.raster.crs)
    return hydrography


def _apply_parameter_steps(
    model: Any,
    steps: list[tuple[str, dict[str, Any]]],
    maps: xr.Dataset,
    rivers: gpd.GeoDataFrame,
) -> None:
    """Run the Wflow-owned setup methods against the initialized P1 grid."""
    for name, configured in steps:
        kwargs = configured.copy()
        if name == "setup_rivers":
            kwargs.pop("hydrography_fn", None)
            kwargs.pop("river_geom_fn", None)
            model.setup_rivers(
                hydrography_fn=_p1_hydrography(maps),
                river_geom_fn=rivers,
                **kwargs,
            )
        elif name == "setup_lulcmaps":
            source_name = str(
                kwargs.pop("lulc_fn", maps.attrs.get("lulc_source", "vito"))
            )
            kwargs.setdefault("lulc_mapping_fn", f"{source_name}_mapping_default")
            model.setup_lulcmaps(
                lulc_fn=maps["land_cover"].rename("landuse"),
                **kwargs,
            )
        elif name == "setup_laimaps":
            kwargs.pop("lai_fn", None)
            lai = maps["leaf_area_index"].rename("LAI")
            if "month" in lai.dims:
                lai = lai.rename(month="time")
            model.setup_laimaps(lai_fn=lai, **kwargs)
        else:
            getattr(model, name)(**kwargs)


def _positive_ids(data: xr.DataArray) -> set[int]:
    values = np.asarray(data.values)
    return {
        int(value) for value in np.unique(values[np.isfinite(values) & (values > 0)])
    }


def _validate_written_model(
    root: Path,
    p1_maps: xr.Dataset,
    registry: pd.DataFrame,
) -> None:
    """Reopen the Wflow triplet and verify its P1 grid and identity joins."""
    from hydromt_wflow import WflowSbmModel

    required_paths = (
        root / "staticmaps.nc",
        root / "wflow_sbm.toml",
        root / "staticgeoms" / "region.geojson",
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise ValueError(f"Wflow build did not write its required triplet: {missing}")
    reopened = WflowSbmModel(root=root, mode="r")
    reopened.read()
    written = reopened.staticmaps.data
    if written.raster.shape != p1_maps.raster.shape:
        raise ValueError("Written Wflow grid shape differs from P1")
    if not np.allclose(written.raster.bounds, p1_maps.raster.bounds):
        raise ValueError("Written Wflow grid bounds differ from P1")
    p1_subbasins = _positive_ids(p1_maps["subbasin_id"])
    if _positive_ids(written["subcatchment"]) != p1_subbasins:
        raise ValueError("Written Wflow subcatchment IDs differ from P1")
    registry_ids = set(registry["wflow_id"].astype(int))
    if _positive_ids(written["gauges_locations"]) != registry_ids:
        raise ValueError("Written Wflow gauge IDs differ from P1 location_registry")
    if not _positive_ids(written["outlets"]).issubset(p1_subbasins):
        raise ValueError("Written Wflow outlet IDs are not inherited P1 subbasin IDs")
    reopened.close()


def build_wflow_model(
    spatial_catalog: str | os.PathLike[str],
    parameter_template: str | os.PathLike[str],
    data_catalogs: str | os.PathLike[str] | Sequence[object],
    wflow_root: str | os.PathLike[str],
) -> None:
    """Build and validate Wflow-SBM from the generated P1 spatial catalog."""
    from hydromt import DataCatalog
    from hydromt_wflow import WflowSbmModel

    spatial_catalog = Path(spatial_catalog)
    root = Path(wflow_root)
    steps = read_parameter_steps(parameter_template)
    p1_catalog = DataCatalog(data_libs=str(spatial_catalog))
    maps = p1_catalog.get_rasterdataset("spatial_maps", single_var_as_array=False)
    registry = p1_catalog.get_dataframe("location_registry")
    geoms = {
        name: p1_catalog.get_geodataframe(name)
        for name in ("basins", "subbasins", "catchments", "rivers", "locations")
    }
    if (
        maps is None
        or registry is None
        or any(value is None for value in geoms.values())
    ):
        raise ValueError("P1 spatial catalog contains an unreadable required entry")
    _validate_registry(registry, geoms["locations"])

    model_catalogs = [str(spatial_catalog), *_catalog_paths(data_catalogs)]
    model = WflowSbmModel(root=root, mode="w", data_libs=model_catalogs)
    model.staticmaps.set(_static_base_maps(maps))
    model.geoms.set(geoms["basins"], name="region")
    for name in ("subbasins", "catchments", "rivers", "locations"):
        model.geoms.set(geoms[name], name=name)
    model.setup_config(_BASE_CONFIG)
    model.set_flwdir(ftype="ldd")
    _apply_parameter_steps(model, steps, maps, geoms["rivers"])
    model.setup_gauges(
        gauges_fn=geoms["locations"],
        index_col="wflow_id",
        basename="locations",
        snap_to_river=False,
        derive_subcatch=False,
        toml_output=None,
    )
    model.setup_outlets(river_only=True, toml_output=None)
    model.write()
    model.close()
    _validate_written_model(root, maps, registry)
    maps.close()


if __name__ == "__main__" and "snakemake" in globals():
    sm = globals()["snakemake"]
    from blueearth_cst.shared.snake_utils import tee_to_log

    with tee_to_log(sm.log[0]):
        build_wflow_model(
            spatial_catalog=sm.input.spatial_catalog,
            parameter_template=sm.input.parameter_template,
            data_catalogs=sm.input.data_catalogs,
            wflow_root=Path(sm.output.staticmaps).parent,
        )
