"""Physical forcing metadata, separate from scenario meaning and native labels."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from pyproj import CRS

from blueearth_cst.experiment.content_identity import content_sha256


def _metadata_value(value: Any) -> Any:
    """Encode native numeric metadata without nonstandard JSON NaN literals."""
    if isinstance(value, np.ndarray):
        return _metadata_value(value.tolist())
    if isinstance(value, np.generic):
        return _metadata_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return {
            "nonfinite": "nan" if np.isnan(value) else ("inf" if value > 0 else "-inf")
        }
    if isinstance(value, (list, tuple)):
        return [_metadata_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _metadata_value(item) for key, item in value.items()}
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError(f"unsupported physical metadata type: {type(value).__name__}")


def _spatial_description(ds: xr.Dataset, *, require_crs: bool = True) -> dict[str, Any]:
    """Describe the actual grid, including auxiliary coordinates and CRS metadata."""
    if "spatial_ref" not in ds and require_crs:
        raise ValueError("missing spatial_ref")
    attrs = ds.spatial_ref.attrs if "spatial_ref" in ds else {}
    wkt = attrs.get("crs_wkt") or attrs.get("spatial_ref")
    if not wkt and require_crs:
        raise ValueError("spatial_ref has no CRS WKT")
    coordinates = {}
    for name, coordinate in sorted(ds.coords.items()):
        if name in {"time", "spatial_ref"} or "time" in coordinate.dims:
            continue
        coordinates[name] = {
            "dimensions": list(coordinate.dims),
            "dtype": coordinate.dtype.name,
            "values": _metadata_value(coordinate.values),
            "attributes": _metadata_value(coordinate.attrs),
        }
    if not coordinates:
        raise ValueError("physical artifact has no spatial coordinates")
    return {
        "crs": CRS.from_wkt(wkt).to_string() if wkt else None,
        "spatial_ref": _metadata_value(attrs),
        "dimensions": {
            name: size for name, size in sorted(ds.sizes.items()) if name != "time"
        },
        "coordinates": coordinates,
    }


def _missing_encoding(variable: xr.DataArray) -> dict[str, Any] | None:
    """Retain the on-disk fill declarations separately from decoded missingness."""
    result = {
        key: _metadata_value(variable.encoding.get(key, variable.attrs.get(key)))
        for key in ("_FillValue", "missing_value")
        if key in variable.encoding or key in variable.attrs
    }
    return result or None


def reader_unit_interpretation(reader: dict[str, Any]) -> "UnitInterpretation":
    """Read the explicit persisted binding; never infer units from native labels."""
    try:
        binding = reader["metadata"]["cst_unit_interpretation"]
        if set(binding) != {"revision", "evidence", "variables"}:
            raise ValueError("unexpected unit interpretation fields")
        result = UnitInterpretation(
            binding["revision"],
            binding["evidence"],
            tuple((name, units) for name, units in binding["variables"]),
        )
        if not all(
            isinstance(value, str) for value in (result.revision, result.evidence)
        ):
            raise ValueError("unit interpretation evidence must be text")
        if not all(
            isinstance(value, str) for pair in result.variables for value in pair
        ):
            raise ValueError("unit interpretation variables must be text")
        result.validate({name for name, _ in result.variables})
        return result
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid persisted unit interpretation: {exc}") from exc


def collection_forcing_descriptor(path: Path, reader: dict[str, Any]) -> dict[str, Any]:
    """Extract collection/1 metadata from a real forcing and evidenced unit binding."""
    descriptor = describe_forcing(
        ClimateArtifact(path.stem, path),
        reader_unit_interpretation(reader),
        time_label="interval_end",
    )
    seconds = descriptor.timestep_seconds
    timestep = "P1D" if seconds == 86400 else f"PT{seconds:g}S"
    with xr.open_dataset(path) as ds:
        return {
            "source_calendar": descriptor.calendar,
            "timestep": timestep,
            "time_label": descriptor.time_label,
            "start": descriptor.first_time,
            "end": descriptor.last_time,
            "spatial_representation_sha256": content_sha256(_spatial_description(ds)),
            "variables": [
                {
                    "name": item.name,
                    "units": item.units,
                    "missing_value": _missing_encoding(ds[item.name]),
                }
                for item in descriptor.variables
            ],
        }


@dataclass(frozen=True)
class ClimateArtifact:
    """A forcing artifact addressed by an opaque run id."""

    run_id: str
    path: Path


@dataclass(frozen=True)
class UnitInterpretation:
    """Explicit effective-unit binding justified by source/preparation provenance."""

    revision: str
    evidence: str
    variables: tuple[tuple[str, str], ...]

    def validate(self, observed: set[str]) -> None:
        """Require one evidenced physical interpretation per observed variable."""
        names = [name for name, _ in self.variables]
        if not self.revision or not self.evidence:
            raise ValueError(
                "forcing units require interpretation revision and evidence"
            )
        if len(set(names)) != len(names) or set(names) != observed:
            raise ValueError(
                f"unit interpretation variables {names} do not equal {sorted(observed)}"
            )
        if any(not units for _, units in self.variables):
            raise ValueError("effective forcing units must be explicit")


@dataclass(frozen=True)
class VariableDescriptor:
    """One variable's physical units and unmodified native metadata."""

    name: str
    units: str
    dimensions: tuple[str, ...]
    native_attributes: tuple[tuple[str, object], ...]
    missing_count: int


@dataclass(frozen=True)
class ForcingDescriptor:
    """Observed spatial/time axes and explicitly interpreted physical units."""

    variables: tuple[VariableDescriptor, ...]
    crs: str
    dimensions: tuple[tuple[str, int], ...]
    calendar: str
    timestep_seconds: float
    time_label: str
    first_time: str
    last_time: str
    unit_interpretation: UnitInterpretation


def describe_forcing(
    artifact: ClimateArtifact,
    interpretation: UnitInterpretation,
    *,
    time_label: str,
) -> ForcingDescriptor:
    """Inspect forcing without treating native unit labels as conversions.

    Args:
        artifact: Existing generated forcing; never modified by inspection.
        interpretation: Versioned, evidenced units for every physical variable.
        time_label: Explicit interval convention supplied by the provider binding.

    Raises:
        ValueError: Missing metadata, invalid axes or incomplete interpretation.
    """
    if time_label not in {"interval_end", "instant"}:
        raise ValueError(f"invalid forcing time_label {time_label!r}")
    with xr.open_dataset(artifact.path) as ds:
        if "time" not in ds.coords or ds.sizes["time"] < 2:
            raise ValueError(
                f"run_id={artifact.run_id}: forcing needs at least two times"
            )
        index = ds.indexes["time"]
        if not index.is_unique or not index.is_monotonic_increasing:
            raise ValueError(f"run_id={artifact.run_id}: time must increase uniquely")
        steps = np.array(
            [
                (right - left).total_seconds()
                for left, right in zip(index[:-1], index[1:])
            ]
        )
        if not np.all(steps == steps[0]) or steps[0] <= 0:
            raise ValueError(
                f"run_id={artifact.run_id}: forcing timestep is not uniform"
            )
        if "spatial_ref" not in ds:
            raise ValueError(f"run_id={artifact.run_id}: missing spatial_ref")
        spatial = ds["spatial_ref"].attrs
        wkt = spatial.get("crs_wkt") or spatial.get("spatial_ref")
        if not wkt:
            raise ValueError(f"run_id={artifact.run_id}: spatial_ref has no CRS WKT")
        crs = CRS.from_wkt(wkt).to_string()
        names = set(ds.data_vars) - {"spatial_ref"}
        interpretation.validate(names)
        units = dict(interpretation.variables)
        variables = tuple(
            VariableDescriptor(
                name=name,
                units=units[name],
                dimensions=tuple(ds[name].dims),
                native_attributes=tuple(sorted(ds[name].attrs.items())),
                missing_count=int((~np.isfinite(ds[name])).sum().item()),
            )
            for name in sorted(names)
        )
        return ForcingDescriptor(
            variables=variables,
            crs=crs,
            dimensions=tuple((str(name), int(size)) for name, size in ds.sizes.items()),
            calendar=str(ds.time.dt.calendar),
            timestep_seconds=float(steps[0]),
            time_label=time_label,
            first_time=str(index[0]),
            last_time=str(index[-1]),
            unit_interpretation=interpretation,
        )
