"""Shared workflow opt-in for Wflow's native gridded output writer."""

from __future__ import annotations

import re
import tomllib
from math import prod
from pathlib import Path


def normalize_grid_output(output: object) -> dict | None:
    """Normalize the closed workflow output request; omission disables grids."""
    from blueearth_cst.shared.csv_output import normalize_csv_output

    normalize_csv_output(output)
    if output is None:
        return None
    grid = output.get("netcdf_grid")
    if grid is None:
        return None
    if not isinstance(grid, dict) or set(grid) - {"variables", "compressionlevel"}:
        raise ValueError(
            "netcdf_grid accepts variables and compressionlevel; no path override"
        )
    level = grid.get("compressionlevel", 1)
    if type(level) is not int or not 0 <= level <= 9:
        raise ValueError("netcdf_grid.compressionlevel must be an integer from 0 to 9")
    variables = grid.get("variables")
    if not isinstance(variables, dict) or not variables:
        raise ValueError("netcdf_grid.variables must be a nonempty mapping")
    reserved = {"x", "y", "lon", "lat", "time", "layer", "spatial_ref", "crs"}
    for key, value in variables.items():
        if not isinstance(key, str) or not re.fullmatch(
            r"[a-z][a-z0-9_]*__[a-z0-9_]+", key
        ):
            raise ValueError(
                f"Expected a Wflow standard-name identifier, received {key!r}"
            )
        if (
            not isinstance(value, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]*", value)
            or value in reserved
        ):
            raise ValueError(
                f"Grid variable {value!r} must be lowercase snake_case and not a coordinate"
            )
    if len(set(variables.values())) != len(variables):
        raise ValueError("netcdf_grid variable names must be unique")
    return {"compressionlevel": level, "variables": dict(sorted(variables.items()))}


def configure_grid_output(config: dict, request: dict | None, path: str) -> None:
    """Replace inherited declarations, including when this workflow disables grids."""
    output = config.setdefault("output", {})
    output.pop("netcdf_grid", None)
    if request is not None:
        output["netcdf_grid"] = {**request, "path": path}


def grid_path(csv_path: str | Path) -> Path:
    """Return the owned per-run grid path alongside its CSV."""
    return Path(csv_path).with_suffix(".nc")


def configured_grid_path(toml_path: str | Path) -> Path | None:
    """Resolve the native output anchor using Wflow's TOML path semantics."""
    path = Path(toml_path)
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    grid = config.get("output", {}).get("netcdf_grid")
    if grid is None:
        return None
    return (
        path.parent / config.get("dir_output", "run_default") / grid["path"]
    ).resolve()


def validate_grid(
    path: str | Path,
    toml_path: str | Path,
    *,
    expected_contract: dict | None = None,
    legacy: bool = False,
) -> dict:
    """Validate against native model expectations and the complete simulation clock.

    New admission loads the producer's TOML-bound contract and checks its static
    input digest. Retained readers may supply an inventory-authenticated contract
    without the original model paths. ``legacy`` preserves the original reader
    semantics only for checksum-verified inventories predating that contract.
    """
    import netCDF4
    import numpy as np
    import pandas as pd
    from pyproj import CRS

    from blueearth_cst.shared.provenance import file_sha256

    with Path(toml_path).open("rb") as handle:
        config = tomllib.load(handle)
    request = config["output"]["netcdf_grid"]
    clock = config["time"]
    contract = expected_contract
    if not legacy:
        if contract is None:
            with Path(str(toml_path) + ".grid.toml").open("rb") as handle:
                contract = tomllib.load(handle)
            static_path = (
                Path(toml_path).parent
                / config.get("dir_input", ".")
                / config["input"]["path_static"]
            ).resolve()
            if contract["static_sha256"] != file_sha256(static_path):
                raise ValueError("Grid contract differs from model staticmaps")
        if (
            contract["schema_version"] != "wflow-grid-contract/1"
            or contract["toml_sha256"] != file_sha256(toml_path)
            or {
                name: item["standard_name"]
                for name, item in contract["variables"].items()
            }
            != {name: standard for standard, name in request["variables"].items()}
        ):
            raise ValueError("Grid contract differs from native TOML/request")
    with netCDF4.Dataset(path) as ds:
        variables = request["variables"]
        coordinates = {"x", "y", "lon", "lat", "time", "layer", "spatial_ref"}
        if set(ds.variables) - coordinates != set(variables.values()):
            raise ValueError(f"Grid {path}: variable names differ from request")
        crs = ds.variables["spatial_ref"]
        actual_crs = CRS.from_wkt(
            getattr(crs, "crs_wkt", getattr(crs, "spatial_ref", ""))
        )
        if contract is not None:
            if actual_crs != CRS.from_wkt(contract["crs_wkt"]):
                raise ValueError(f"Grid {path}: CRS differs from model staticmaps")
            for name, expected_coordinate in contract["coordinates"].items():
                if name not in ds.variables or not np.array_equal(
                    ds[name][:], expected_coordinate
                ):
                    raise ValueError(
                        f"Grid {path}: {name} differs from model staticmaps"
                    )
        time = ds.variables["time"]
        if time.calendar != clock["calendar"]:
            raise ValueError(f"Grid {path}: calendar differs from simulation")
        dates = netCDF4.num2date(time[:], time.units, time.calendar)
        expected = pd.date_range(
            pd.Timestamp(clock["starttime"])
            + pd.Timedelta(seconds=clock["timestepsecs"]),
            pd.Timestamp(clock["endtime"]),
            freq=pd.Timedelta(seconds=clock["timestepsecs"]),
        )
        if [str(value) for value in dates] != [str(value) for value in expected]:
            raise ValueError(f"Grid {path}: incomplete or mismatched time coverage")
        metadata = {}
        for standard, name in variables.items():
            value = ds.variables[name]
            spatial = ("lat", "lon") if "lat" in ds.variables else ("y", "x")
            if set(value.dimensions) not in (
                set((*spatial, "time")),
                set((*spatial, "layer", "time")),
            ):
                raise ValueError(
                    f"Grid {name}: unsupported dimensions {value.dimensions}"
                )
            if (
                not getattr(value, "units", "")
                or getattr(value, "wflow_standard_name", None) != standard
                or getattr(value, "grid_mapping", None) != "spatial_ref"
            ):
                raise ValueError(f"Grid {name}: missing native physical metadata")
            support = None
            if contract is not None:
                native = contract["variables"][name]
                if value.units != native["units"]:
                    raise ValueError(f"Grid {name}: units differ from native Wflow")
                if list(value.dimensions) != native["dimensions"]:
                    raise ValueError(
                        f"Grid {name}: dimensions differ from native Wflow"
                    )
                if native["layers"] and not np.array_equal(
                    ds["layer"][:], native["layers"]
                ):
                    raise ValueError(f"Grid {name}: layers differ from native Wflow")
                support = np.zeros(
                    tuple(len(ds.dimensions[d]) for d in spatial), dtype=bool
                )
                indices = np.asarray(contract["supports"][native["support"]], dtype=int)
                if indices.size:
                    support[indices[:, 0] - 1, indices[:, 1] - 1] = True
            # Read every compressed chunk, batching small grids into roughly
            # 8 MiB blocks instead of paying one Python/NetCDF call per day.
            # Very large grids still require at least one native time slab.
            axis = value.dimensions.index("time")
            slab_bytes = value.dtype.itemsize * prod(
                size for index, size in enumerate(value.shape) if index != axis
            )
            block_steps = max(1, (8 << 20) // max(1, slab_bytes))
            for step in range(0, len(dates), block_steps):
                key = [slice(None)] * value.ndim
                key[axis] = slice(step, min(step + block_steps, len(dates)))
                block = np.ma.filled(value[tuple(key)], np.nan)
                if support is not None and not np.isfinite(block[..., support]).all():
                    raise ValueError(
                        f"Grid {name}: missing/non-finite values on native support"
                    )
                if np.isinf(block).any():
                    raise ValueError(f"Grid {name}: infinite values")
            metadata[name] = {
                "standard_name": standard,
                "units": value.units,
                "dimensions": list(value.dimensions),
                "shape": list(value.shape),
            }
        for name in spatial:
            coord = ds.variables[name][:]
            if len(coord) > 1 and not (
                np.all(np.diff(coord) > 0) or np.all(np.diff(coord) < 0)
            ):
                raise ValueError(f"Grid {path}: nonmonotonic {name}")
        result = {
            "variables": metadata,
            "calendar": time.calendar,
            "timesteps": len(dates),
        }
        if contract is not None:
            result["validation_contract"] = contract
        return result
