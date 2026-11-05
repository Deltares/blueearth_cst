"""Declared historical map dependencies, preparation identities and common display.

All identities use current bytes and logical roles. Executable paths belong only
to source specs and rule declarations. Shared sealed numerical fields feed display.
"""

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap

from blueearth_cst.climate_analysis.source_map_fields import (
    SourceMapSpec,
)
from blueearth_cst.plotting.prepared_maps import MapFrame, MapScale
from blueearth_cst.shared.map_preparation import (  # noqa: F401 -- compatible import surface
    LOCK_FILES,
    PREPARATION_FILES,
    SCIENTIFIC_ATTRS,
    VARIABLES,
    _array_digest,
    _code_identity,
    _input_identity,
    _orography_descriptor,
    _repo_root,
    canonical_json,
    environment_identity,
    fields_digest,
    file_digest,
    json_digest,
    preparation_identity,
    publish_source_fields,
    read_source_fields,
    require_wf0_completeness,
    resolve_map_completeness,
    resolve_source_orography,
    seal_fields,
    source_fields_path,
    source_map_inputs,
    source_role,
    source_spec_dict,
    source_spec_from_dict,
    validate_fields,
)
from blueearth_cst.shared.map_preparation import POLICIES as PREPARATION_POLICIES
from blueearth_cst.shared.plot_colors import HISTORICAL_MAP_COLORS, MAP_FEATURE_COLORS
from blueearth_cst.shared.wf3_science import (
    DEFAULT_BASIN_INDEX,
    DEFAULT_HYDROGRAPHY,
    climate_store_rule,
)

SCHEMA_VERSION = 1
POLICIES = {
    **PREPARATION_POLICIES,
    "frame_support": 1,
    "ticks": 1,
    "palette": 1,
    "classes": 1,
}
REFERENCE_FILES = (
    "config/plotting/colors.yml",
    "blueearth_cst/shared/plot_colors.py",
    "blueearth_cst/shared/plot_setup.py",
    *PREPARATION_FILES,
    "blueearth_cst/shared/map_reference.py",
    "blueearth_cst/shared/plot_style.py",
    "blueearth_cst/shared/plot_preset.py",
    "blueearth_cst/plotting/fundamentals.py",
    "blueearth_cst/plotting/preset.py",
    "blueearth_cst/plotting/prepared_maps.py",
    "blueearth_cst/plotting/__init__.py",
    "blueearth_cst/shared/produce_map_reference.py",
)


def reference_arguments(params: Mapping[str, Any]) -> dict[str, Any]:
    """Translate shared executable rule params for producer and both adapters."""
    return {
        "specs": tuple(
            source_spec_from_dict(value) for value in params["source_specs"]
        ),
        "selected": params["selected"],
        "m0": params["m0"],
        "completeness": dict(params["completeness"]),
        "shared_inputs": dict(params["shared_inputs"]),
    }


@dataclass(frozen=True)
class HistoricalMapReferenceRule:
    script: str
    inputs: Mapping[str, Any]
    outputs: Mapping[str, str]
    params: Mapping[str, Any]
    source_specs: tuple[SourceMapSpec, ...]


def historical_map_reference_rule(
    *,
    project_dir: str | Path,
    model_region: Any,
    climate: Mapping[str, Any],
    data_sources: str | Sequence[str],
    region_geojson: str | Path,
    basin_geojson: str | Path,
    m0: int,
    hydrography: str = DEFAULT_HYDROGRAPHY,
    basin_index: str = DEFAULT_BASIN_INDEX,
) -> HistoricalMapReferenceRule:
    """Resolve both DAGs' identical producer without reading not-yet-built stores."""
    if type(m0) is not int or not 1 <= m0 <= 12:
        raise ValueError("Map reporting month must be an integer in [1, 12]")
    selected = climate["selected"]
    declared = climate["sources"]
    if (
        not isinstance(declared, (list, tuple))
        or not declared
        or selected not in declared
    ):
        raise ValueError(
            "climate.sources must be a nonempty list containing climate.selected"
        )
    names = tuple(dict.fromkeys([selected, *declared]))
    catalogs = tuple(
        Path(path)
        for path in ([data_sources] if isinstance(data_sources, str) else data_sources)
    )
    completeness = resolve_map_completeness(climate.get("map_reference"))
    orography = resolve_source_orography(catalogs) if "era5" in names else ()
    specs = []
    for name in names:
        source_role(name)
        store = climate_store_rule(
            project_dir,
            model_region,
            name,
            climate["window"],
            [str(path) for path in catalogs],
            hydrography=hydrography,
            basin_index=basin_index,
            enforce_min_years=name == selected,
            forcing_required=name == selected
            or name not in {"chirps", "chirps_global"},
        )
        specs.append(
            SourceMapSpec(
                name,
                Path(store.outputs["climate_nc"]),
                Path(store.outputs["basin_cells"]),
                catalogs,
                (store.params["starttime"][:10], store.params["endtime"][:10]),
                Path(store.outputs["oro_nc"]) if "oro_nc" in store.outputs else None,
                orography if name == "era5" else (),
            )
        )
    shared = {
        "region": Path(region_geojson).as_posix(),
        "basins": Path(basin_geojson).as_posix(),
    }
    inputs = {
        "prepared_fields": [source_fields_path(spec).as_posix() for spec in specs],
        "climate_nc": [spec.climate_nc.as_posix() for spec in specs],
        "basin_cells": [spec.basin_cells.as_posix() for spec in specs],
        "catalogs": [path.as_posix() for path in catalogs],
        "orography": [
            path.as_posix()
            for spec in specs
            for path in (
                *spec.orography_inputs,
                *((spec.orography_nc,) if spec.orography_nc is not None else ()),
            )
        ],
        "reference_code": [
            (_repo_root() / path).as_posix() for path in REFERENCE_FILES
        ],
        "environment_locks": [(_repo_root() / path).as_posix() for path in LOCK_FILES],
        **shared,
    }
    params = {
        "source_specs": [source_spec_dict(spec) for spec in specs],
        "selected": selected,
        "m0": m0,
        "completeness": completeness,
        "shared_inputs": shared,
        "preparation_code_digest": json_digest(_code_identity(PREPARATION_FILES)),
        "reference_code_digest": json_digest(_code_identity(REFERENCE_FILES)),
        "environment": environment_identity(),
        "policies": dict(POLICIES),
    }
    return HistoricalMapReferenceRule(
        "blueearth_cst/shared/produce_map_reference.py",
        inputs,
        {
            "reference": (
                Path(project_dir) / "data/climate/historical/_engine/map_reference.json"
            ).as_posix()
        },
        params,
        tuple(specs),
    )


def native_edges(coordinates: Any, resolution: Any = None) -> np.ndarray:
    """Regular monotonic native edges, including explicit one-cell resolution."""
    original = np.asanyarray(coordinates)
    if (
        not np.issubdtype(original.dtype, np.number)
        or np.iscomplexobj(original)
        or np.ma.getmaskarray(original).any()
    ):
        raise ValueError("Native coordinates must be real and unmasked")
    values = np.asarray(original)
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Native grid coordinates must be a finite nonempty vector")
    if len(values) == 1:
        if (
            isinstance(resolution, bool)
            or resolution is None
            or not np.isfinite(resolution)
            or resolution <= 0
        ):
            raise ValueError("One-cell maps require positive finite native resolution")
        step = float(resolution)
    else:
        steps = np.diff(values.astype(np.float64))
        precision = 2 * float(np.max(np.abs(np.spacing(values))))
        if not (np.all(steps > 0) or np.all(steps < 0)) or not np.allclose(
            steps, steps[0], rtol=0, atol=precision
        ):
            raise ValueError(
                "Maps require regular monotonic rectilinear native coordinates"
            )
        step = values[1] - values[0]
    return np.r_[
        values[0] - step / 2, (values[:-1] + values[1:]) / 2, values[-1] + step / 2
    ]


def field_edges(field: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    return (
        native_edges(field.longitude.values, field.attrs.get("longitude_resolution")),
        native_edges(field.latitude.values, field.attrs.get("latitude_resolution")),
    )


def geographic_ticks(low: float, high: float) -> list[float]:
    target = (high - low) / 4
    if not math.isfinite(target) or target <= 0:
        raise ValueError("Geographic tick span must be positive and finite")
    power = 10 ** math.floor(math.log10(target))
    step = next(
        factor * power for factor in (1, 2, 2.5, 5, 10) if factor * power >= target
    )
    return [
        0.0 if (value := round(index * step, 12)) == 0 else value
        for index in range(math.ceil(low / step), math.floor(high / step) + 1)
    ]


def frame_record(field: xr.DataArray, source: str) -> dict[str, Any]:
    x, y = field_edges(field)
    west, east, south, north = (
        float(x.min()),
        float(x.max()),
        float(y.min()),
        float(y.max()),
    )
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError(
            "Historical reference frame must be geographic without antimeridian crossing"
        )
    cosine = math.cos(math.radians((south + north) / 2))
    if cosine <= 1e-12:
        raise ValueError("Historical reference frame is singular at the poles")
    return {
        "selected_source": source,
        "extent": [west, east, south, north],
        "longitude_ticks": geographic_ticks(west, east),
        "latitude_ticks": geographic_ticks(south, north),
        "aspect": 1 / cosine,
        "crs": "EPSG:4326",
    }


def frame_intersection(field: xr.DataArray, extent: Sequence[float]) -> np.ndarray:
    """Closed cell/frame intersection includes boundary-only touching cells."""
    x, y = field_edges(field)
    west, east, south, north = extent
    columns = (np.minimum(x[:-1], x[1:]) <= east) & (np.maximum(x[:-1], x[1:]) >= west)
    rows = (np.minimum(y[:-1], y[1:]) <= north) & (np.maximum(y[:-1], y[1:]) >= south)
    return rows[:, None] & columns[None, :]


def scale_record(
    variable: str, low: float | None, high: float | None
) -> dict[str, Any]:
    """Approved increments, outward integer-index rounding and resolved palette."""
    if variable not in VARIABLES:
        raise ValueError(f"Unknown map variable: {variable}")
    temperature = variable == "temp"
    palette = HISTORICAL_MAP_COLORS["temp" if temperature else "precip_pet"]
    result = {
        "units": "degC" if temperature else "mm/year",
        "status": "unavailable" if low is None else "available",
        "reason": "No finite historical reference cells" if low is None else None,
        "extrema": [low, high],
        "boundaries": [],
        "ticks": [],
        "colors": [],
        "under_color": palette["under"],
        "over_color": palette["over"],
        "missing_color": MAP_FEATURE_COLORS["missing"],
    }
    if low is None:
        if high is not None:
            raise ValueError("Unavailable reference extrema must both be null")
        return result
    if high is None or not math.isfinite(low) or not math.isfinite(high) or high < low:
        raise ValueError("Reference extrema must be finite and ordered")
    working_high = high if high > low else low + (0.2 if temperature else 1)
    span = working_high - low

    def within(threshold):
        # Subtraction can widen a translated decimal interval by a few ULPs.
        # Comparing its endpoint keeps the next representable wider span strict.
        return working_high <= low + threshold and (
            span <= threshold
            or math.isclose(
                span,
                threshold,
                rel_tol=0,
                abs_tol=math.ulp(low) + math.ulp(working_high),
            )
        )

    step = 100 if not temperature else 0.2 if within(1.6) else 0.5 if within(4) else 1
    first, last = math.floor(low / step), math.ceil(working_high / step)
    boundaries = [round(index * step, 12) for index in range(first, last + 1)]
    ramp = palette["colors"]
    classes = len(boundaries) - 1
    colors = (
        [ramp[-1]]
        if classes == 1
        else LinearSegmentedColormap.from_list("historical", ramp)
        .resampled(classes)(np.arange(classes))
        .tolist()
    )
    result.update(boundaries=boundaries, ticks=list(boundaries), colors=colors)
    return result


def current_reference_identity(
    specs: Sequence[SourceMapSpec],
    selected: str,
    m0: int,
    completeness: Mapping[str, int],
    shared_inputs: Mapping[str, str],
) -> dict[str, Any]:
    if (
        not specs
        or specs[0].source != selected
        or len({spec.source for spec in specs}) != len(specs)
    ):
        raise ValueError("Reference sources must be unique and selected-first")
    preparations = {
        spec.source: preparation_identity(spec, m0, completeness) for spec in specs
    }
    return {
        "sources": [
            {"source": spec.source, "role": source_role(spec.source)} for spec in specs
        ],
        "selected": selected,
        "reporting_month": m0,
        "completeness": resolve_map_completeness(completeness),
        "policies": dict(POLICIES),
        "inputs": [
            {"role": role, "sha256": file_digest(path)}
            for role, path in sorted(shared_inputs.items())
        ],
        "code": _code_identity(REFERENCE_FILES),
        "environment": environment_identity(),
        "source_preparations": {
            source: {
                "preparation_identity": identity,
                "preparation_id": json_digest(identity),
            }
            for source, identity in preparations.items()
        },
    }


def build_reference(
    specs: Sequence[SourceMapSpec],
    selected: str,
    m0: int,
    completeness: Mapping[str, int],
    shared_inputs: Mapping[str, str],
) -> dict[str, Any]:
    """Build display classes from shared, integrity-checked numerical fields."""
    identity = current_reference_identity(
        specs, selected, m0, completeness, shared_inputs
    )
    expected = json.loads(canonical_json(identity))
    pooled = {
        variable: {"low": None, "high": None, "count": 0, "sources": [], "excluded": []}
        for variable in VARIABLES
    }
    frame = None
    for spec in specs:
        fields, entry = read_source_fields(spec, m0, completeness)
        if (
            entry["preparation_identity"]
            != identity["source_preparations"][spec.source]["preparation_identity"]
        ):
            raise ValueError(
                f"{spec.source}: source inputs changed during reference preparation"
            )
        identity["source_preparations"][spec.source] = entry
        if frame is None:
            frame = frame_record(fields.precip, selected)
        prep = entry["preparation_identity"]
        grid = fields.precip.transpose("latitude", "longitude")
        x, y = field_edges(grid)
        support = frame_intersection(grid, frame["extent"])
        grid_summary = {
            "grid_dimensions": {
                "latitude": grid.sizes["latitude"],
                "longitude": grid.sizes["longitude"],
            },
            "cell_edges_digest": json_digest(
                {
                    "longitude": _array_digest(x, "float"),
                    "latitude": _array_digest(y, "float"),
                }
            ),
            "intersecting_cells": int(support.sum()),
            "finite_cells": 0,
            "unavailable_cells": int(support.sum()),
            "eligible_year_counts": None,
        }
        scientific = [
            variable
            for variable in VARIABLES
            if variable in fields
            and (
                variable == "precip"
                or source_role(spec.source) == "temperature-carrier"
            )
        ]
        for variable in VARIABLES:
            pool = pooled[variable]
            summary = {
                **grid_summary,
                "source": spec.source,
                "role": source_role(spec.source),
                "effective_dates": prep["extraction_span"],
                "scientific_variables": scientific,
                "full_reporting_years": prep["full_reporting_years"],
                "preparation_id": entry["preparation_id"],
                "fields_digest": entry["fields_digest"],
            }
            if variable not in scientific:
                pool["excluded"].append(
                    {
                        **summary,
                        "reason": "Precipitation-only source role"
                        if variable != "precip"
                        and source_role(spec.source) == "precipitation-only"
                        else "Variable absent",
                    }
                )
                continue
            field = fields[variable].transpose("latitude", "longitude")
            support = frame_intersection(field, frame["extent"])
            finite = support & np.isfinite(field.values)
            values = field.values[finite]
            x, y = field_edges(field)
            counts = (
                fields[f"{variable}_valid_years"]
                .transpose("latitude", "longitude")
                .values[finite]
            )
            summary.update(
                grid_dimensions={
                    "latitude": field.sizes["latitude"],
                    "longitude": field.sizes["longitude"],
                },
                cell_edges_digest=json_digest(
                    {
                        "longitude": _array_digest(x, "float"),
                        "latitude": _array_digest(y, "float"),
                    }
                ),
                intersecting_cells=int(support.sum()),
                finite_cells=int(finite.sum()),
                unavailable_cells=int((support & ~np.isfinite(field.values)).sum()),
                eligible_year_counts=[int(counts.min()), int(counts.max())]
                if len(counts)
                else None,
            )
            if not len(values):
                pool["excluded"].append(
                    {**summary, "reason": "No finite cells intersect reference frame"}
                )
                continue
            low, high = float(values.min()), float(values.max())
            pool["low"] = low if pool["low"] is None else min(pool["low"], low)
            pool["high"] = high if pool["high"] is None else max(pool["high"], high)
            pool["count"] += len(values)
            pool["sources"].append(summary)
        fields.close()
        del fields
    if canonical_json(
        current_reference_identity(specs, selected, m0, completeness, shared_inputs)
    ) != canonical_json(expected):
        raise ValueError(
            "Historical reference inputs changed during preparation; publication aborted"
        )
    record = {
        "schema_version": SCHEMA_VERSION,
        "identity": identity,
        "frame": frame,
        "variables": {
            variable: {
                **scale_record(variable, pool["low"], pool["high"]),
                "finite_pool_count": pool["count"],
                "sources": pool["sources"],
                "excluded_sources": pool["excluded"],
            }
            for variable, pool in pooled.items()
        },
        "provenance": {
            "method": "Native source fields; raw precipitation/temperature; de Bruin source-grid PET",
            "policies": dict(POLICIES),
            "source_inputs": {
                source: entry["preparation_identity"]["inputs"]
                for source, entry in identity["source_preparations"].items()
            },
            "environment": identity["environment"],
        },
    }
    record["reference_id"] = json_digest(record)
    return record


def validate_reference(
    record: Mapping[str, Any],
    specs: Sequence[SourceMapSpec],
    selected: str,
    m0: int,
    completeness: Mapping[str, int],
    shared_inputs: Mapping[str, str],
) -> None:
    """Reject unknown schema, corrupt payload and current-byte identity drift."""
    if not isinstance(record, Mapping):
        raise ValueError("Historical map reference must be a JSON object")
    if (
        set(record)
        != {
            "schema_version",
            "reference_id",
            "identity",
            "frame",
            "variables",
            "provenance",
        }
        or type(record.get("schema_version")) is not int
        or record["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("Unknown historical map reference schema")
    if record["reference_id"] != json_digest(
        {key: value for key, value in record.items() if key != "reference_id"}
    ):
        raise ValueError("Historical map reference payload corruption")
    if set(record["variables"]) != set(VARIABLES):
        raise ValueError(
            "Historical reference variables must be exactly precip/temp/pet"
        )
    if (
        set(record["frame"])
        != {
            "selected_source",
            "extent",
            "longitude_ticks",
            "latitude_ticks",
            "aspect",
            "crs",
        }
        or record["frame"]["crs"] != "EPSG:4326"
    ):
        raise ValueError("Unknown historical reference frame schema/CRS")
    frame = record["frame"]
    extent = frame["extent"]
    if (
        not isinstance(extent, list)
        or len(extent) != 4
        or any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in extent
        )
    ):
        raise ValueError("Historical reference frame has invalid extent")
    west, east, south, north = extent
    if (
        not (-180 <= west < east <= 180 and -90 <= south < north <= 90)
        or frame["selected_source"] != selected
    ):
        raise ValueError(
            "Historical reference has an incompatible geographic frame authority"
        )
    expected_frame = {
        "longitude_ticks": geographic_ticks(west, east),
        "latitude_ticks": geographic_ticks(south, north),
        "aspect": 1 / math.cos(math.radians((south + north) / 2)),
    }
    for key, value in expected_frame.items():
        if canonical_json(frame[key]) != canonical_json(value):
            raise ValueError(f"Historical reference has inconsistent frame {key}")
    expected = current_reference_identity(
        specs, selected, m0, completeness, shared_inputs
    )
    actual = dict(record["identity"])
    entries = actual["source_preparations"]
    actual["source_preparations"] = {}
    for source, entry in entries.items():
        if (
            set(entry) != {"preparation_identity", "preparation_id", "fields_digest"}
            or not isinstance(entry["fields_digest"], str)
            or len(entry["fields_digest"]) != 64
        ):
            raise ValueError(f"{source}: unknown source-preparation schema")
        actual["source_preparations"][source] = {
            key: value for key, value in entry.items() if key != "fields_digest"
        }
    if canonical_json(actual) != canonical_json(expected):
        raise ValueError(
            "Historical reference is stale: current inputs/code/environment/settings differ"
        )
    provenance = record["provenance"]
    expected_provenance = {
        "method": "Native source fields; raw precipitation/temperature; de Bruin source-grid PET",
        "policies": dict(POLICIES),
        "source_inputs": {
            source: entry["preparation_identity"]["inputs"]
            for source, entry in entries.items()
        },
        "environment": expected["environment"],
    }
    if canonical_json(provenance) != canonical_json(expected_provenance):
        raise ValueError("Historical reference has inconsistent provenance")
    variable_keys = set(scale_record("precip", None, None)) | {
        "finite_pool_count",
        "sources",
        "excluded_sources",
    }
    for variable, scale in record["variables"].items():
        if set(scale) != variable_keys:
            raise ValueError(
                f"{variable}: unknown historical reference variable schema"
            )
        for key, value in scale_record(variable, *scale["extrema"]).items():
            if canonical_json(scale[key]) != canonical_json(value):
                raise ValueError(f"{variable}: inconsistent historical scale {key}")
        if (
            type(scale["finite_pool_count"]) is not int
            or scale["finite_pool_count"] < 0
        ):
            raise ValueError(f"{variable}: invalid finite cell count")
        if (scale["status"] == "available") != (scale["finite_pool_count"] > 0):
            raise ValueError(
                f"{variable}: reference status disagrees with finite pool count"
            )
        source_names = [
            summary["source"]
            for summary in [*scale["sources"], *scale["excluded_sources"]]
        ]
        if len(set(source_names)) != len(source_names) or set(source_names) != set(
            entries
        ):
            raise ValueError(f"{variable}: source contribution inventory mismatch")
        if (
            sum(summary["finite_cells"] for summary in scale["sources"])
            != scale["finite_pool_count"]
        ):
            raise ValueError(f"{variable}: source counts disagree with pool")
        for summary in [*scale["sources"], *scale["excluded_sources"]]:
            required = {
                "source",
                "role",
                "effective_dates",
                "scientific_variables",
                "full_reporting_years",
                "preparation_id",
                "fields_digest",
                "grid_dimensions",
                "cell_edges_digest",
                "intersecting_cells",
                "finite_cells",
                "unavailable_cells",
                "eligible_year_counts",
            }
            excluded = summary in scale["excluded_sources"]
            if set(summary) != required | ({"reason"} if excluded else set()):
                raise ValueError(f"{variable}: unknown source summary schema")
            entry = entries.get(summary["source"])
            if (
                entry is None
                or summary["preparation_id"] != entry["preparation_id"]
                or summary["fields_digest"] != entry["fields_digest"]
            ):
                raise ValueError(f"{variable}: source summary identity mismatch")
            if (
                summary["role"] != source_role(summary["source"])
                or any(
                    type(summary[key]) is not int or summary[key] < 0
                    for key in (
                        "intersecting_cells",
                        "finite_cells",
                        "unavailable_cells",
                    )
                )
                or summary["intersecting_cells"]
                != summary["finite_cells"] + summary["unavailable_cells"]
            ):
                raise ValueError(f"{variable}: invalid source role/cell counts")
            if (
                summary["role"] == "precipitation-only"
                and variable != "precip"
                and not excluded
            ):
                raise ValueError(
                    f"{variable}: precipitation-only source cannot contribute"
                )
            if (
                excluded
                and summary["finite_cells"] != 0
                or not excluded
                and summary["finite_cells"] == 0
            ):
                raise ValueError(f"{variable}: invalid contribution status")
            bounds = summary["eligible_year_counts"]
            if bounds is None:
                if summary["finite_cells"]:
                    raise ValueError(
                        f"{variable}: finite source needs eligible-year count range"
                    )
            elif (
                not isinstance(bounds, list)
                or len(bounds) != 2
                or any(type(value) is not int or value < 0 for value in bounds)
                or bounds[1] < bounds[0]
            ):
                raise ValueError(f"{variable}: invalid eligible-year count range")


def load_reference(path: str | Path, **expected: Any) -> dict[str, Any]:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_reference(record, **expected)
    return record


def publish_reference(
    path: str | Path, record: Mapping[str, Any], **expected: Any
) -> None:
    """Validate current inputs then atomically replace from a private sibling."""
    validate_reference(record, **expected)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write((canonical_json(record) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        validate_reference(
            json.loads(temporary.read_text(encoding="utf-8")), **expected
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def drawing_frame(record: Mapping[str, Any]) -> MapFrame:
    frame = record["frame"]
    return MapFrame(
        tuple(frame["extent"]),
        frame["longitude_ticks"],
        frame["latitude_ticks"],
        frame["aspect"],
    )


def drawing_scale(record: Mapping[str, Any], variable: str) -> MapScale:
    scale = record["variables"][variable]
    return MapScale(
        **{
            key: scale[key]
            for key in (
                "units",
                "boundaries",
                "ticks",
                "colors",
                "under_color",
                "over_color",
                "missing_color",
                "status",
            )
        }
    )
