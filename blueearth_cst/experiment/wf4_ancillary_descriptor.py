"""WF4 physical description of retained preparation ancillary data."""

import math
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import yaml

from blueearth_cst.experiment.content_identity import content_sha256
from blueearth_cst.experiment.forcing_descriptor import (
    _metadata_value,
    _missing_encoding,
    _spatial_description,
)
from blueearth_cst.experiment.shared_orography import (
    publish_shared_orography,
    relative_hydromt_uri,
    resolve_hydromt_uri,
)
from blueearth_cst.shared.workflow_config_snapshot import (
    file_reference,
    resolve_file_reference,
)

_CITATION_KEYS = frozenset(
    {
        "license",
        "notes",
        "paper_doi",
        "paper_ref",
        "url",
        "source_license",
        "source_url",
        "source_version",
    }
)
_METADATA_KEYS = (
    frozenset(
        {
            "crs",
            "nodata",
            "units",
            "extent",
            "processing",
            "category",
            "cst_unit_interpretation",
        }
    )
    | _CITATION_KEYS
)


def _closed(value: Any, keys: set[str] | frozenset[str], name: str) -> dict:
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"{name} requires exactly {sorted(keys)}")
    return value


def _extent(value: Any) -> dict | None:
    if value is None:
        return None
    if type(value) is not dict or set(value) - {"bbox", "time_range"}:
        raise ValueError("reader extent has unknown fields")
    bbox = value.get("bbox")
    if bbox is not None:
        _closed(bbox, {"West", "South", "East", "North"}, "reader bbox")
        if any(
            type(v) not in {int, float} or not math.isfinite(v) for v in bbox.values()
        ):
            raise ValueError("reader bbox requires finite numbers")
        if bbox["West"] > bbox["East"] or bbox["South"] > bbox["North"]:
            raise ValueError("reader bbox has reversed bounds")
    interval = value.get("time_range")
    if interval is not None:
        _closed(interval, {"start", "end"}, "reader time range")
        parsed = []
        for endpoint in (interval["start"], interval["end"]):
            if type(endpoint) is not str:
                raise ValueError("reader time endpoint requires ISO 8601 text")
            try:
                parsed.append(
                    datetime.fromisoformat(endpoint)
                    if "T" in endpoint
                    else date.fromisoformat(endpoint)
                )
            except ValueError as exc:
                raise ValueError("reader time endpoint requires ISO 8601 text") from exc
        if type(parsed[0]) is not type(parsed[1]) or parsed[0] > parsed[1]:
            raise ValueError("reader time range has incomparable or reversed bounds")
    return {
        "bbox": dict(bbox) if bbox is not None else None,
        "time_range": dict(interval) if interval is not None else None,
    }


def reader_identity_v2(native: dict, *, elevation: bool) -> dict:
    """Project an installed HydroMT raster reader with closed controls."""
    if type(native) is not dict or set(native) - {
        "data_type",
        "driver",
        "data_adapter",
        "metadata",
        "uri",
        "root",
    }:
        raise ValueError("HydroMT reader has unclassified controls")
    if native.get("data_type") != "RasterDataset":
        raise ValueError("unsupported HydroMT reader data type")
    driver = native.get("driver")
    if driver == "raster_xarray":
        driver = {"name": "raster_xarray", "options": {}}
    if (
        type(driver) is not dict
        or set(driver) - {"name", "options"}
        or driver.get("name") != "raster_xarray"
    ):
        raise ValueError("unsupported HydroMT raster driver")
    options = driver.get("options") or {}
    if type(options) is not dict or set(options) - {"chunks", "lock", "preprocess"}:
        raise ValueError("unclassified HydroMT driver options")
    chunks = options.get("chunks")
    if chunks is not None and (
        type(chunks) is not dict
        or any(
            type(k) is not str or not k or type(v) is not int or v < 1
            for k, v in chunks.items()
        )
    ):
        raise ValueError("invalid HydroMT chunks")
    lock = options.get("lock")
    if lock is not None and type(lock) is not bool:
        raise ValueError("invalid HydroMT lock")
    preprocess = options.get("preprocess")
    if preprocess is not None and preprocess not in {"harmonise_dims", "round_latlon"}:
        raise ValueError("unsupported HydroMT preprocess")
    adapter = native.get("data_adapter")
    if adapter is not None:
        if type(adapter) is not dict or set(adapter) - {
            "rename",
            "unit_mult",
            "unit_add",
        }:
            raise ValueError("unclassified HydroMT adapter")
        adapter = {
            key: adapter.get(key, {}) for key in ("rename", "unit_mult", "unit_add")
        }
        if any(type(v) is not dict for v in adapter.values()):
            raise ValueError("invalid HydroMT adapter map")
        if any(
            type(k) is not str or type(v) is not str
            for k, v in adapter["rename"].items()
        ):
            raise ValueError("invalid HydroMT rename")
        if any(
            type(k) is not str or type(v) not in {int, float} or not math.isfinite(v)
            for key in ("unit_mult", "unit_add")
            for k, v in adapter[key].items()
        ):
            raise ValueError("invalid HydroMT unit adapter")
    metadata = native.get("metadata") or {}
    if type(metadata) is not dict or set(metadata) - _METADATA_KEYS:
        raise ValueError("unclassified HydroMT reader metadata")
    unit = metadata.get("cst_unit_interpretation")
    if elevation:
        if unit is not None:
            raise ValueError("elevation reader has climate unit interpretation")
        unit_identity = None
    else:
        _closed(
            unit, {"revision", "evidence", "variables"}, "climate unit interpretation"
        )
        if (
            type(unit["revision"]) is not str
            or type(unit["evidence"]) is not str
            or type(unit["variables"]) is not list
        ):
            raise ValueError("invalid climate unit interpretation")
        variables = unit["variables"]
        if any(
            type(row) is not list
            or len(row) != 2
            or any(type(x) is not str for x in row)
            for row in variables
        ):
            raise ValueError("invalid climate unit variables")
        unit_identity = {
            "revision": unit["revision"],
            "variables": [
                dict(zip(("name", "units"), row)) for row in sorted(variables)
            ],
        }
    return {
        "schema_version": "forcing-reader-identity/1",
        "data_type": "RasterDataset",
        "driver": {
            "name": "raster_xarray",
            "options": {"chunks": chunks, "lock": lock, "preprocess": preprocess},
        },
        "data_adapter": adapter,
        "metadata": {
            "crs": metadata.get("crs"),
            "nodata": metadata.get("nodata"),
            "units": metadata.get("units"),
            "extent": _extent(metadata.get("extent")),
        },
        "unit_interpretation": unit_identity,
    }


def preparation_identity_v2(
    document: dict, *, project_root: Path, experiment_root: Path
) -> dict:
    """Rebuild the path-neutral identity from checked preparation artifacts."""
    _closed(
        document,
        {
            "schema_version",
            "ancillary",
            "catalog",
            "forcing_elevation",
            "generated_forcing_reader",
            "pet_method",
        },
        "WF4 preparation",
    )
    if document["schema_version"] != "forcing-preparation/2":
        raise ValueError("unsupported WF4 preparation version")
    ancillary = document["ancillary"]
    if type(ancillary) is not list or not ancillary:
        raise ValueError("WF4 preparation ancillary is empty")
    projected = []
    by_id = {}
    for item in ancillary:
        _closed(item, {"id", "role", "file", "descriptor"}, "WF4 ancillary")
        if type(item["id"]) is not str or item["id"] in by_id:
            raise ValueError("WF4 ancillary ID is invalid or repeated")
        path = resolve_file_reference(item["file"], {"project_root": project_root})
        if item["file"]["path_base"] != "project_root":
            raise ValueError("WF4 ancillary must bind the project root")
        if describe_ancillary(path) != item["descriptor"]:
            raise ValueError("WF4 ancillary physical descriptor differs")
        by_id[item["id"]] = item
        projected.append(
            {
                "role": item["role"],
                "sha256": item["file"]["sha256"],
                "size_bytes": item["file"]["size_bytes"],
                "descriptor": item["descriptor"],
            }
        )
    if [item["id"] for item in ancillary] != sorted(by_id):
        raise ValueError("WF4 ancillary IDs are not sorted")
    selection = _closed(
        document["forcing_elevation"],
        {"catalog_key", "artifact_id"},
        "WF4 elevation selection",
    )
    if selection["catalog_key"] != "elevation" or selection["artifact_id"] not in by_id:
        raise ValueError("WF4 elevation selection is invalid")
    selected = by_id[selection["artifact_id"]]
    if selected["role"] != "forcing_elevation":
        raise ValueError("WF4 elevation role differs")
    catalog_path = resolve_file_reference(
        document["catalog"], {"project_root": project_root}
    )
    if document["catalog"][
        "path_base"
    ] != "project_root" or not catalog_path.is_relative_to(experiment_root):
        raise ValueError("WF4 elevation catalog is outside experiment")
    native_catalog = yaml.safe_load(catalog_path.read_bytes())
    _closed(native_catalog, {"elevation"}, "WF4 elevation catalog")
    native_reader = native_catalog["elevation"]
    checked = resolve_hydromt_uri(
        catalog_path, native_reader["uri"], selected["file"], project_root
    )
    if checked != resolve_file_reference(
        selected["file"], {"project_root": project_root}
    ):
        raise ValueError("WF4 elevation catalog resolves another artifact")
    return {
        "schema_version": "forcing-preparation-identity/2",
        "ancillary": projected,
        "catalog": {
            "elevation": {
                "content": {
                    "role": selected["role"],
                    "sha256": selected["file"]["sha256"],
                    "size_bytes": selected["file"]["size_bytes"],
                },
                "reader": reader_identity_v2(native_reader, elevation=True),
            }
        },
        "forcing_elevation": selection,
        "generated_forcing_reader": reader_identity_v2(
            document["generated_forcing_reader"], elevation=False
        ),
        "pet_method": document["pet_method"],
    }


def describe_ancillary(path: Path) -> dict[str, Any]:
    """Observe a packaged NetCDF ancillary without a live catalog."""
    with xr.open_dataset(path) as ds:
        spatial = _spatial_description(ds, require_crs=False)
        variables = []
        for name in sorted(set(ds.data_vars) - {"spatial_ref"}):
            variable = ds[name]
            variables.append(
                {
                    "name": name,
                    "units": variable.attrs.get("units"),
                    "dimensions": list(variable.dims),
                    "attributes": _metadata_value(variable.attrs),
                    "missing_value": _missing_encoding(variable),
                    "missing_count": int((~np.isfinite(variable)).sum().item()),
                }
            )
        if not variables:
            raise ValueError("ancillary has no physical variables")
        return {
            "crs": spatial["crs"],
            "spatial_representation_sha256": content_sha256(spatial),
            "variables": variables,
        }


def resolve_wf4_preparation(
    project_root: Path,
    experiment_root: Path,
    *,
    climate_source: str,
    catalogs: list[str | Path],
    unit_interpretation,
    oro_path: Path | None = None,
) -> dict[str, Any]:
    """Bind verified shared elevation and its relative HydroMT catalog to WF4."""
    import hydromt

    from blueearth_cst.climate_analysis.prepare_climate_data_catalog import (
        resolve_preparation_payloads,
    )

    project = Path(project_root).resolve()
    experiment = Path(experiment_root).resolve()
    if not experiment.is_relative_to(project / "experiments"):
        raise ValueError("WF4 preparation must belong to this project")
    predecessor, catalog_bytes, payloads = resolve_preparation_payloads(
        catalogs, climate_source, unit_interpretation, oro_path
    )
    if len(payloads) != 1:
        raise ValueError("WF4 preparation has an unexpected ancillary closure")
    relative, elevation_source = next(iter(payloads.items()))
    if relative != f"ancillary/elevation/{elevation_source.name}":
        raise ValueError("WF4 elevation payload path differs")
    shared_reference, observed_descriptor = publish_shared_orography(
        elevation_source,
        project,
        source_name=climate_source,
        basename=elevation_source.name,
        describe=describe_ancillary,
    )
    if observed_descriptor != predecessor["ancillary"][0]["descriptor"]:
        raise ValueError("shared elevation physical descriptor differs")
    catalog_path = (
        experiment / "hydrology/wflow/run_settings/forcing_elevation_catalog.yml"
    )
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    if (
        catalog_path.parent.resolve() != catalog_path.parent
        or catalog_path.is_symlink()
    ):
        raise ValueError("WF4 elevation catalog path is aliased")
    native = yaml.safe_load(catalog_bytes)
    if set(native) != {"elevation"}:
        raise ValueError("WF4 elevation catalog must contain only elevation")
    native["elevation"]["uri"] = relative_hydromt_uri(
        catalog_path, shared_reference, project
    )
    catalog_data = yaml.safe_dump(native, sort_keys=True).encode("utf-8")
    if catalog_path.exists():
        if catalog_path.read_bytes() != catalog_data:
            raise ValueError("retained WF4 elevation catalog differs")
    else:
        with catalog_path.open("xb") as handle:
            handle.write(catalog_data)
            handle.flush()
            os.fsync(handle.fileno())
    selected = hydromt.DataCatalog(data_libs=[str(catalog_path)]).get_source(
        "elevation"
    )
    if resolve_hydromt_uri(
        catalog_path, native["elevation"]["uri"], shared_reference, project
    ) != Path(selected.full_uri).resolve(strict=True):
        raise ValueError("HydroMT elevation URI differs from checked shared file")
    return {
        "schema_version": "forcing-preparation/2",
        "ancillary": [
            {
                "id": "elevation",
                "role": "forcing_elevation",
                "file": shared_reference,
                "descriptor": observed_descriptor,
            }
        ],
        "catalog": file_reference(catalog_path, "project_root", project),
        "forcing_elevation": predecessor["forcing_elevation"],
        "generated_forcing_reader": predecessor["generated_forcing_reader"],
        "pet_method": predecessor["pet_method"],
    }
