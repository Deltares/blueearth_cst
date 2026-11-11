"""Source qualification and readers for climate-index results."""

import json
from pathlib import Path

from blueearth_cst.shared.provenance import canonical_sha256, file_sha256


def implementation_digest() -> str:
    """Identify the index kernels, source qualification and table calculation."""
    package = Path(__file__).resolve().parents[1]
    paths = sorted((package / "indicators" / "climate").glob("*.py"))
    paths += [
        Path(__file__),
        Path(__file__).with_name("climate_index_table.py"),
        Path(__file__).with_name("source_capabilities.py"),
        Path(__file__).with_name("diagnostic_settings.py"),
        package / "shared" / "grid_cells.py",
    ]
    return canonical_sha256(
        {str(p.relative_to(package)): file_sha256(p) for p in paths}
    )


def qualified_source_data(climate_path, source, catalogs=()):
    """Load source-owned variables using the established unit qualification rules."""
    import numpy as np
    import xarray as xr
    import yaml

    from blueearth_cst.climate_analysis.source_capabilities import source_climate_vars

    with xr.open_dataset(climate_path) as opened:
        original = opened.load()
    catalog_paths = [catalogs] if isinstance(catalogs, str) else list(catalogs)
    offsets = {}
    for path in catalog_paths:
        entry = yaml.safe_load(Path(path).read_text(encoding="utf-8")).get(source, {})
        if entry:
            offsets = entry.get("data_adapter", entry).get("unit_add", {})
    data = xr.Dataset(coords=original.coords, attrs=dict(original.attrs))
    unavailable = {}
    for variable in ("precip", "temp", "temp_max", "temp_min"):
        if variable not in original or (
            variable != "precip" and "temp" not in source_climate_vars(source)
        ):
            unavailable[variable] = "not available from this source"
            continue
        field = original[variable].copy(deep=False)
        units = str(field.attrs.get("units", "")).lower()
        if variable == "precip":
            verified = units in {
                "mm",
                "mm/day",
                "mm/d",
                "mm d-1",
                "mm d**-1",
                "mm day-1",
                "mm day**-1",
            }
            canonical = "mm/day"
        else:
            verified = units in {
                "degc",
                "celsius",
                "degree_celsius",
                "degrees_celsius",
                "°c",
            }
            verified |= (
                units in {"k", "kelvin"}
                and offsets.get(variable) == -273.15
                and bool(original.attrs.get("region_geojson_sha256"))
            )
            finite = field.values[np.isfinite(field.values)]
            verified &= (
                bool(len(finite))
                and (finite.min() >= -100 if len(finite) else False)
                and (finite.max() <= 70 if len(finite) else False)
            )
            canonical = "degC"
        if not verified:
            unavailable[variable] = "units or values could not be verified"
            continue
        field.attrs = {**field.attrs, "units": canonical, "source": source}
        data[variable] = field
    if not data.data_vars:
        raise ValueError("No source-owned climate variables have verified units")
    return data, unavailable


def load_index_results(record_path):
    """Read stable CSV results only when their record and payload hashes agree."""
    import pandas as pd

    record_path = Path(record_path)
    root = record_path.parent.parent.resolve()
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("schema_version") != "climate-index-results/1":
        raise ValueError("Unsupported climate-index results record")
    if record.get("implementation") != implementation_digest():
        raise ValueError(
            "Climate-index results are stale; recalculate with current code"
        )
    products = record.get("products", {})
    if set(products) != {"table", "spells"}:
        raise ValueError("Incomplete climate-index results record")
    paths = {}
    for key, reference in products.items():
        path = root / reference["path"]
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(root)
            or file_sha256(path) != reference["sha256"]
        ):
            raise ValueError(f"Climate-index result failed integrity check: {key}")
        paths[key] = path
    identifiers = (
        "spatial_id",
        "subbasin_id",
        "wflow_id",
        "spatial_location_id",
        "location_code",
        "period",
    )
    table = pd.read_csv(
        paths["table"],
        dtype=dict.fromkeys(identifiers, str),
        keep_default_na=False,
        na_values={"value": [""], "valid_days": [""], "x": [""], "y": [""]},
    )
    return table, pd.read_csv(paths["spells"]), record


def load_publication(pointer_path: str | Path) -> tuple[Path, dict]:
    """Read legacy retained bundles for compatibility; new runs write stable CSVs."""
    pointer_path = Path(pointer_path)
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    root = pointer_path.parent.parent.resolve()
    manifest_path = (root / pointer["manifest"]).resolve()
    if not any(
        manifest_path.is_relative_to(directory)
        for directory in (root / "_engine" / "climate_indices", root / "indices")
    ):
        raise ValueError("Climate-index manifest escapes its source store")
    if file_sha256(manifest_path) != pointer["sha256"]:
        raise ValueError("Climate-index manifest digest mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("complete") is not True:
        raise ValueError("Unsupported or incomplete climate-index publication")
    for name, digest in manifest["products"].items():
        path = (manifest_path.parent / name).resolve()
        if not path.is_relative_to(manifest_path.parent) or file_sha256(path) != digest:
            raise ValueError(f"Climate-index product failed integrity check: {name}")
    return manifest_path.parent, manifest
