"""Numerical historical map preparation, identities and payload integrity."""

import hashlib
import importlib
import json
import math
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import xarray as xr

from blueearth_cst.climate_analysis.source_map_fields import SourceMapSpec
from blueearth_cst.shared.output_capture import captured_output
from blueearth_cst.shared.yaml_documents import parse_yaml_document

SCHEMA_VERSION = 1
POLICIES = {"preparation": 1, "units_availability": 1, "source_pet": 1}
PREPARATION_FILES = (
    "blueearth_cst/climate_analysis/source_map_fields.py",
    "blueearth_cst/climate_analysis/source_capabilities.py",
    "blueearth_cst/climate_analysis/diagnostic_maps.py",
    "blueearth_cst/climate_analysis/diagnostics.py",
    "blueearth_cst/climate_analysis/diagnostic_tables.py",
    "blueearth_cst/climate_analysis/diagnostic_settings.py",
    "blueearth_cst/climate_analysis/plot_climate_source.py",
    "blueearth_cst/shared/climate_parity.py",
    "blueearth_cst/shared/grid_cells.py",
    "blueearth_cst/shared/map_preparation.py",
    "blueearth_cst/shared/yaml_documents.py",
    "blueearth_cst/shared/variable_registry.py",
    "blueearth_cst/__init__.py",
    "blueearth_cst/climate_analysis/__init__.py",
    "blueearth_cst/shared/__init__.py",
)
# Environment narrowing is a separate scientific compatibility decision.
LOCK_FILES = ("pixi.lock", "Manifest.toml")
VARIABLES = ("precip", "temp", "pet")
SCIENTIFIC_ATTRS = (
    "units",
    "latitude_resolution",
    "longitude_resolution",
    "reporting_year_label",
    "period_start",
    "period_end",
)


def source_fields_path(spec: SourceMapSpec) -> Path:
    """Shared WF0/WF1 output; retain the existing annual-climatology location."""
    return spec.climate_nc.parent / "tables" / "annual_climatology.nc"


def read_source_fields(
    spec: SourceMapSpec, m0: int, completeness: Mapping[str, int]
) -> tuple[xr.Dataset, dict[str, Any]]:
    """Read current sealed fields without recalculating or repairing their identity."""
    target = source_fields_path(spec)
    expected = preparation_identity(spec, m0, completeness)
    with xr.open_dataset(target) as opened:
        fields = opened.load()
    entry = {
        "preparation_identity": expected,
        "preparation_id": fields.attrs.get("map_preparation_id"),
        "fields_digest": fields.attrs.get("map_fields_digest"),
    }
    validate_fields(fields, expected, entry, source=spec.source, target=target)
    return fields, entry


def canonical_json(value: Any) -> str:
    """Strict sorted JSON; preserve integer types and normalize negative zero."""

    def normalize(item: Any) -> Any:
        if isinstance(item, Mapping):
            if not all(isinstance(key, str) for key in item):
                raise TypeError("Identity JSON requires string object keys")
            return {key: normalize(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(val) for val in item]
        if isinstance(item, np.generic):
            return normalize(item.item())
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("Identity JSON forbids nonfinite numbers")
            return 0.0 if item == 0 else item
        if item is None or isinstance(item, (str, int, bool)):
            return item
        raise TypeError(f"Unsupported identity JSON type: {type(item).__name__}")

    return json.dumps(
        normalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def json_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_digest(path: str | Path) -> str:
    """Stream every byte; sizes and modification times are never identities."""
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def resolve_map_completeness(value: Any = None) -> dict[str, int]:
    """Resolve the closed project ``climate.map_reference`` policy."""
    defaults = {"max_missing_days_month": 3, "max_missing_days_year": 15}
    if value is None:
        return defaults
    if not isinstance(value, Mapping):
        raise ValueError("climate.map_reference must be a mapping")
    if set(value) - set(defaults):
        raise ValueError(
            f"climate.map_reference has unknown keys: {sorted(set(value) - set(defaults))}"
        )
    result = {**defaults, **value}
    for key, number in result.items():
        if (
            type(number) is not int
            or number < 0
            or (key.endswith("month") and number > 27)
        ):
            bound = (
                "an integer in [0, 27]"
                if key.endswith("month")
                else "a nonnegative integer"
            )
            raise ValueError(
                f"climate.map_reference.{key} must be {bound}; got {number!r}"
            )
    return result


def require_wf0_completeness(
    settings: Mapping[str, Any], reference: Mapping[str, int]
) -> None:
    """Fail DAG construction on differing WF0 and historical-reference policies."""
    for key, value in resolve_map_completeness(reference).items():
        if type(settings.get(key)) is not int or settings.get(key) != value:
            raise ValueError(
                f"workflows.analyze_climate.diagnostics.{key}={settings.get(key)!r} "
                f"must equal climate.map_reference.{key}={value}"
            )


def source_role(source: str) -> str:
    if source not in {"era5", "chirps", "chirps_global"}:
        raise ValueError(f"Historical map source is unsupported: {source!r}")
    return "temperature-carrier" if source == "era5" else "precipitation-only"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _code_identity(paths: Sequence[str]) -> list[dict[str, str]]:
    return [
        {"path": path, "sha256": file_digest(_repo_root() / path)} for path in paths
    ]


def environment_identity() -> dict[str, Any]:
    """Qualify actual imported versions and relevant Conda builds against Pixi v7."""
    root = _repo_root()
    lock = parse_yaml_document((root / "pixi.lock").read_bytes())
    if lock.get("version") != 7:
        raise ValueError("Map environment qualification requires Pixi lock schema 7")
    active_platform = (
        "win-64"
        if sys.platform == "win32"
        else "linux-64"
        if sys.platform.startswith("linux")
        else None
    )
    if active_platform is None:
        raise ValueError(f"Unsupported map environment platform: {sys.platform}")
    urls = {
        entry["conda"]
        for entry in lock["environments"]["default"]["packages"][active_platform]
        if "conda" in entry
    }
    expected = {}
    for url in urls:
        archive = url.rsplit("/", 1)[-1].removesuffix(".conda").removesuffix(".tar.bz2")
        name, version, build = archive.rsplit("-", 2)
        expected[name] = (version, build)
    modules = {
        "hydromt": "hydromt",
        "numpy": "numpy",
        "pandas": "pandas",
        "xarray": "xarray",
        "scipy": "scipy",
        "rasterio": "rasterio",
        "pyproj": "pyproj",
        "dask-core": "dask",
        "netcdf4": "netCDF4",
        "matplotlib-base": "matplotlib",
        "pyyaml": "yaml",
        "shapely": "shapely",
        "geopandas-base": "geopandas",
    }
    loaded = {
        name: str(importlib.import_module(module).__version__)
        for name, module in modules.items()
    }
    loaded["python"] = platform.python_version()
    rasterio = importlib.import_module("rasterio")
    pyproj = importlib.import_module("pyproj")
    netcdf = importlib.import_module("netCDF4")
    loaded.update(
        {
            "libgdal-core": rasterio.__gdal_version__,
            "proj": pyproj.__proj_version__,
            "libnetcdf": netcdf.__netcdf4libversion__.split()[0],
        }
    )
    packages = {}
    for name, version in sorted(loaded.items()):
        locked = expected.get(name)
        installed = None
        if locked is not None:
            metadata = (
                Path(sys.prefix) / "conda-meta" / f"{name}-{locked[0]}-{locked[1]}.json"
            )
            if metadata.is_file():
                entry = json.loads(metadata.read_text(encoding="utf-8"))
                installed = (entry["version"], entry["build"])
        if locked is None or version != locked[0] or installed != locked:
            raise ValueError(
                f"Map environment inconsistent with Pixi lock: {name}, loaded {version}, locked {locked}, installed {installed}"
            )
        packages[name] = {"version": version, "build": expected[name][1]}
    return {
        "environment": "default",
        "platform": active_platform,
        "packages": packages,
        "locks": _code_identity(LOCK_FILES),
    }


def _orography_descriptor(catalogs: Sequence[Path]) -> Any:
    """Qualify HydroMT's descriptor; preserve its root-relative logical URI."""
    import hydromt

    # Planning runs before rule-log capture, including runner preflights and DAG
    # parsing. Hide routine catalog chatter while retaining warnings and failures.
    with captured_output():
        source = hydromt.DataCatalog(
            data_libs=[str(path) for path in catalogs]
        ).get_source("era5_orography")
    if (
        source.data_type != "RasterDataset"
        or source.driver.name != "raster_xarray"
        or source.uri_resolver.name != "convention"
        or source.driver.filesystem.protocol != "file"
        or source.uri_resolver.filesystem.protocol != "file"
        or source.driver.options.preprocess is not None
        or source.driver.options.ext_override is not None
        or "://" in source.full_uri
        or Path(source.uri).is_absolute()
        or any(character in source.uri for character in "{}*?[")
        or Path(source.full_uri).suffix.lower() not in {".nc", ".netcdf"}
    ):
        raise ValueError(
            "era5_orography: unqualified concrete local input closure; return source/driver interface gap to G1"
        )
    return source


def resolve_source_orography(catalogs: Sequence[Path]) -> tuple[Path, ...]:
    """Resolve the qualified static local ERA5 NetCDF descriptor without reading it."""
    source = _orography_descriptor(catalogs)
    uris = source.uri_resolver.resolve(
        source.full_uri, variables=["z"], metadata=source.metadata
    )
    if len(uris) != 1 or not Path(uris[0]).is_file():
        raise ValueError(
            "era5_orography must resolve to one existing local NetCDF file"
        )
    return (Path(uris[0]),)


def source_spec_dict(spec: SourceMapSpec) -> dict[str, Any]:
    """Executable params only; absolute paths never enter semantic identities."""
    return {
        "source": spec.source,
        "climate_nc": spec.climate_nc.as_posix(),
        "basin_cells": spec.basin_cells.as_posix(),
        "catalogs": [path.as_posix() for path in spec.catalogs],
        "requested_window": list(spec.requested_window),
        "orography_nc": spec.orography_nc.as_posix()
        if spec.orography_nc is not None
        else None,
        "orography_inputs": [path.as_posix() for path in spec.orography_inputs],
    }


def source_spec_from_dict(value: Mapping[str, Any]) -> SourceMapSpec:
    return SourceMapSpec(
        value["source"],
        Path(value["climate_nc"]),
        Path(value["basin_cells"]),
        tuple(Path(path) for path in value["catalogs"]),
        tuple(value["requested_window"]),
        Path(value["orography_nc"]) if value["orography_nc"] is not None else None,
        tuple(Path(path) for path in value["orography_inputs"]),
    )


def source_map_inputs(spec: SourceMapSpec) -> dict[str, Any]:
    """Use identically on WF0 compute and the reference/renderer dependency sets."""
    result = {
        "climate_nc": spec.climate_nc.as_posix(),
        "basin_cells": spec.basin_cells.as_posix(),
        "catalogs": [path.as_posix() for path in spec.catalogs],
        "orography_inputs": [path.as_posix() for path in spec.orography_inputs],
        "preparation_code": [
            (_repo_root() / path).as_posix() for path in PREPARATION_FILES
        ],
        "environment_locks": [(_repo_root() / path).as_posix() for path in LOCK_FILES],
    }
    if spec.orography_nc is not None:
        result["oro_nc"] = spec.orography_nc.as_posix()
    return result


def _input_identity(spec: SourceMapSpec) -> list[dict[str, str]]:
    files = [
        (
            "climate",
            f"{spec.climate_nc.parent.name}/{spec.climate_nc.name}",
            spec.climate_nc,
        ),
        (
            "basin_cells",
            f"{spec.basin_cells.parent.name}/{spec.basin_cells.name}",
            spec.basin_cells,
        ),
    ]
    if spec.orography_nc is not None:
        files.append(
            (
                "orography_sidecar",
                f"{spec.orography_nc.parent.name}/{spec.orography_nc.name}",
                spec.orography_nc,
            )
        )
    for index, path in enumerate(spec.orography_inputs):
        descriptor = _orography_descriptor(spec.catalogs)
        resolved = resolve_source_orography(spec.catalogs)
        if tuple(item.resolve() for item in resolved) != tuple(
            item.resolve() for item in spec.orography_inputs
        ):
            raise ValueError(
                f"{spec.source}: catalog-backed orography dependency mismatch"
            )
        files.append((f"era5_orography:{index}", Path(descriptor.uri).as_posix(), path))
    return [
        {"role": role, "path": logical, "sha256": file_digest(path)}
        for role, logical, path in files
    ]


def preparation_identity(
    spec: SourceMapSpec, m0: int, completeness: Mapping[str, int]
) -> dict[str, Any]:
    """Expected current-byte identity, independent of candidate order or frame."""
    import hydromt  # noqa: F401 -- registers the raster CRS accessor

    if type(m0) is not int or not 1 <= m0 <= 12:
        raise ValueError("Map reporting month must be an integer in [1, 12]")
    if (
        source_role(spec.source) == "temperature-carrier"
        and spec.orography_nc is None
        and not spec.orography_inputs
    ):
        raise ValueError(
            f"{spec.source}: concrete source PET orography inputs are required"
        )
    with xr.open_dataset(spec.climate_nc) as raw:
        if raw.raster.crs is None or raw.raster.crs.to_epsg() != 4326:
            raise ValueError(
                f"{spec.source}: source-map preparation requires EPSG:4326"
            )
        index = pd.DatetimeIndex(raw.time.values)
        if (
            index.empty
            or index.has_duplicates
            or not index.is_monotonic_increasing
            or (index != index.normalize()).any()
        ):
            raise ValueError(
                f"{spec.source}: ordered unique midnight daily Gregorian dates required"
            )
        first, last = index.min(), index.max()
        labels = [
            year
            for year in range(first.year - 1, last.year + 1)
            if pd.Timestamp(year, m0, 1) >= first
            and pd.Timestamp(year + 1, m0, 1) - pd.Timedelta(days=1) <= last
        ]
        present = [variable for variable in VARIABLES if variable in raw]
        units_original = {
            variable: raw[variable].attrs.get("units") for variable in present
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "source": spec.source,
        "role": source_role(spec.source),
        "scientific_variables": ["precip"]
        if source_role(spec.source) == "precipitation-only"
        else ["precip", "temp", "pet"],
        "extracted_variables": present,
        "units_original": units_original,
        "crs": "EPSG:4326",
        "requested_window": list(spec.requested_window),
        "extraction_span": [str(first.date()), str(last.date())],
        "full_reporting_years": labels,
        "reporting_month": m0,
        "completeness": resolve_map_completeness(completeness),
        "policies": {
            key: POLICIES[key]
            for key in ("preparation", "units_availability", "source_pet")
        },
        "catalogs": [
            {"role": f"catalog:{i}", "name": path.name, "sha256": file_digest(path)}
            for i, path in enumerate(spec.catalogs)
        ],
        "inputs": _input_identity(spec),
        "code": _code_identity(PREPARATION_FILES),
        "environment": environment_identity(),
    }


def _array_digest(values: Any, kind: str) -> str:
    array_input = np.asanyarray(values)
    if np.iscomplexobj(array_input) or np.ma.getmaskarray(array_input).any():
        raise ValueError(
            "Payload arrays must be real and explicitly represent missing values"
        )
    values = np.asarray(array_input)
    if not (np.issubdtype(values.dtype, np.number) or values.dtype == np.dtype(bool)):
        raise ValueError("Payload arrays must be numeric or boolean")
    if kind == "float":
        array = np.array(values, dtype="<f8", order="C", copy=True)
        array[array == 0] = 0.0
        array.view("<u8")[np.isnan(array)] = np.uint64(0x7FF8000000000000)
    elif kind == "count":
        if (
            not np.isfinite(values).all()
            or not np.equal(values, values.astype("<i8")).all()
            or (values < 0).any()
        ):
            raise ValueError("Eligible-year counts must be finite nonnegative integers")
        array = np.ascontiguousarray(values, dtype="<i8")
    else:
        if not np.isin(values, [0, 1]).all():
            raise ValueError("Unavailable masks must contain only boolean values")
        array = np.ascontiguousarray(values, dtype="u1")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def fields_digest(fields: xr.Dataset) -> str:
    """Actual payload digest includes every retained auxiliary value/count/mask."""
    inventory = {}
    allowed = {
        f"{variable}{suffix}"
        for variable in VARIABLES
        for suffix in ("", "_valid_years", "_unavailable")
    }
    if set(fields.data_vars) - allowed:
        raise ValueError(
            f"Unknown source-map payload variables: {sorted(set(fields.data_vars) - allowed)}"
        )
    for variable in VARIABLES:
        group = {variable, f"{variable}_valid_years", f"{variable}_unavailable"}
        if group & set(fields.data_vars) and not group <= set(fields.data_vars):
            raise ValueError(
                f"{variable}: map values require matching counts and unavailable mask"
            )
        if variable in fields:
            expected_units = "degC" if variable == "temp" else "mm/year"
            if fields[variable].attrs.get("units") != expected_units:
                raise ValueError(
                    f"{variable}: source-map payload units must be {expected_units}"
                )
    for name in sorted(fields.variables):
        array = fields[name]
        if name in fields.data_vars:
            if set(array.dims) != {"latitude", "longitude"}:
                raise ValueError(
                    f"{name}: annual map dimensions must be latitude/longitude"
                )
            array = array.transpose("latitude", "longitude")
        kind = (
            "count"
            if name.endswith("_valid_years")
            else "mask"
            if name.endswith("_unavailable")
            else "float"
        )
        inventory[name] = {
            "dims": list(array.dims),
            "shape": list(array.shape),
            "sha256": _array_digest(array.values, kind),
            "attrs": {
                key: array.attrs[key] for key in SCIENTIFIC_ATTRS if key in array.attrs
            },
        }
    return json_digest(
        {
            "arrays": inventory,
            "attrs": {
                key: fields.attrs[key]
                for key in SCIENTIFIC_ATTRS
                if key in fields.attrs
            },
        }
    )


def seal_fields(fields: xr.Dataset, expected: Mapping[str, Any]) -> dict[str, Any]:
    """Attach computed preparation and actual payload identity to existing fields."""
    entry = {
        "preparation_identity": dict(expected),
        "preparation_id": json_digest(expected),
        "fields_digest": fields_digest(fields),
    }
    fields.attrs.update(
        map_preparation_schema=SCHEMA_VERSION,
        map_preparation_identity=canonical_json(expected),
        map_preparation_id=entry["preparation_id"],
        map_fields_digest=entry["fields_digest"],
    )
    return entry


def validate_fields(
    fields: xr.Dataset,
    expected: Mapping[str, Any],
    reference_entry: Mapping[str, Any],
    *,
    source: str,
    target: str | Path = "annual_climatology.nc",
) -> dict[str, Any]:
    """Reject missing/stale/tampered fields before the adapter publishes figures."""
    prefix = f"{source}: {target}"
    attrs = fields.attrs
    if (
        type(attrs.get("map_preparation_schema")) not in (int, np.int32, np.int64)
        or attrs.get("map_preparation_schema") != SCHEMA_VERSION
    ):
        raise ValueError(
            f"{prefix}: missing or unsupported map preparation schema; regenerate WF0 compute"
        )
    identity_text = attrs.get("map_preparation_identity")
    if not isinstance(identity_text, str):
        raise ValueError(
            f"{prefix}: missing preparation identity; regenerate WF0 compute"
        )
    identity = json.loads(identity_text)
    entry = {
        "preparation_identity": identity,
        "preparation_id": attrs.get("map_preparation_id"),
        "fields_digest": attrs.get("map_fields_digest"),
    }
    if canonical_json(identity) != canonical_json(expected) or entry[
        "preparation_id"
    ] != json_digest(expected):
        raise ValueError(
            f"{prefix}: stale preparation identity; regenerate WF0 compute and reference"
        )
    if canonical_json(entry) != canonical_json(reference_entry):
        raise ValueError(f"{prefix}: fields/reference preparation mismatch")
    if entry["fields_digest"] != fields_digest(fields):
        raise ValueError(f"{prefix}: source-map payload corruption")
    return entry


def publish_source_fields(
    path: str | Path,
    fields: xr.Dataset,
    spec: SourceMapSpec,
    m0: int,
    completeness: Mapping[str, int],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal, reopen and validate a private sibling before atomic publication."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.nc")
    entry = seal_fields(fields, expected)
    try:
        fields.to_netcdf(temporary)
        with xr.open_dataset(temporary) as reopened:
            validate_fields(
                reopened, expected, entry, source=spec.source, target=target
            )
        if canonical_json(
            preparation_identity(spec, m0, completeness)
        ) != canonical_json(expected):
            raise ValueError(
                f"{spec.source}: inputs changed during preparation; publication aborted"
            )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return entry
