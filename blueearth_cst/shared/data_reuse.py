"""Reuse extracted climate data from other project folders.

A development convenience: a new trial of a project can take its historical
climate store and its CMIP6 raw slices from a sibling trial that already holds
matching data, instead of reading the catalog sources again. Sources are other
projects' ``project_dir`` folders, named by ``project.reuse_data_from`` and by
the ``CST_REUSE_DATA_FROM`` environment variable that ``run_workflows.py
--reuse-data-from`` sets. They are only ever read.

Matching is strict and fails closed. A historical store is reused only when its
extraction record (:data:`RECORD_RELPATH`, written beside every new extraction)
names the same extraction code, source, forcing recipe, hydrography entry,
region geometry and catalog entries (compared without their root), and its
stored time axis covers the requested window. Stores extracted before records
existed, or by other extraction code, are never reused. A CMIP6 raw
slice is reused only when its ``cst_raw_digest`` equals the expected one.

Reuse never enters a Snakemake rule's ``params`` or declared outputs, so
setting or clearing a source changes no rerun decision and no run identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

#: Environment variable the all-workflow runner sets for ``--reuse-data-from``;
#: ``os.pathsep``-separated project directories.
REUSE_ENV = "CST_REUSE_DATA_FROM"

#: Extraction record beside ``extract_historical.nc``, relative to the store.
RECORD_RELPATH = "_engine/extraction_record.json"

RECORD_SCHEMA = "climate-store-record/2"

#: Record fields that must be equal for a store to be reused. The window is
#: matched against the stored time axis instead.
MATCH_FIELDS = (
    "schema",
    "code",
    "clim_source",
    "forcing_required",
    "extraction_fingerprint",
    "hydrography",
    "region_geometry_sha256",
    "datasets",
)

HISTORICAL_STORE_DIR = Path("data") / "climate" / "historical"


def reuse_sources(config: Mapping[str, Any] | None) -> list[Path]:
    """Project directories to read reusable data from, config first, then env.

    ``project.reuse_data_from`` is a path or a list of paths, already resolved
    by :mod:`blueearth_cst.shared.config_paths`. Duplicates keep their first
    position.
    """
    declared: list[str] = []
    project_cfg = (config or {}).get("project") or {}
    value = project_cfg.get("reuse_data_from")
    if isinstance(value, str):
        declared.append(value)
    elif isinstance(value, Sequence):
        declared.extend(str(item) for item in value)
    env = os.environ.get(REUSE_ENV, "")
    declared.extend(item for item in env.split(os.pathsep) if item.strip())
    sources: list[Path] = []
    for item in declared:
        path = Path(item).expanduser().resolve()
        if path not in sources:
            sources.append(path)
    return sources


def region_geometry_sha256(region) -> str:
    """sha256 of a region's geometry as WKB, in feature order.

    Content, not file bytes: two region files written by different runs or
    catalogs that describe the same polygon match.
    """
    digest = hashlib.sha256()
    for wkb in region.geometry.to_wkb():
        digest.update(wkb)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def catalog_dataset_identity(
    catalogs: str | os.PathLike | Sequence[str | os.PathLike],
    names: Iterable[str],
) -> dict[str, str | None] | None:
    """Root-free identity of the named catalog entries.

    Each entry is hashed as written in its YAML, after later catalogs override
    earlier ones, with the catalog ``meta`` block (and so its root) left out.
    Entry paths are relative to the root, so the same dataset under a P-drive
    root and a local mirror has the same identity. An alias also hashes its
    target. Returns None when a catalog is not a readable YAML file, such as a
    predefined catalog name, which disables reuse.
    """
    paths = [catalogs] if isinstance(catalogs, (str, os.PathLike)) else catalogs
    entries: dict[str, Any] = {}
    for item in paths:
        path = Path(item)
        if not path.is_file():
            return None
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for key, value in document.items():
            if key != "meta":
                entries[key] = value

    def resolved(name: str, seen: frozenset[str] = frozenset()) -> Any:
        entry = entries.get(name)
        if isinstance(entry, Mapping) and isinstance(entry.get("alias"), str):
            target = entry["alias"]
            if target not in seen:
                return {"entry": entry, "alias": resolved(target, seen | {name})}
        return entry

    identity: dict[str, str | None] = {}
    for name in sorted(set(names)):
        entry = resolved(name)
        identity[name] = None if entry is None else _canonical_sha256(entry)
    return identity


def store_record(
    *,
    catalogs,
    clim_source: str,
    forcing_required: bool,
    recipe,
    hydrography: str,
    region,
    starttime: str,
    endtime: str,
    code: str,
) -> dict[str, Any] | None:
    """The extraction record of one historical store, or None if unrecordable.

    ``recipe`` is the normalized forcing recipe; every catalog entry the
    extraction may read is identified, so a changed companion, orography or
    PET entry also prevents reuse. ``code`` identifies the extraction code, so
    a store written before a code change is not handed to a later trial.
    """
    names = {clim_source, "era5", "era5_orography", hydrography}
    names.update(
        name
        for name in (
            recipe.precip_source,
            recipe.temp_pet_source,
            recipe.orography_source,
            recipe.pet_source,
        )
        if name
    )
    datasets = catalog_dataset_identity(catalogs, names)
    if datasets is None:
        return None
    return {
        "schema": RECORD_SCHEMA,
        "code": code,
        "clim_source": clim_source,
        "forcing_required": bool(forcing_required),
        "extraction_fingerprint": recipe.extraction_fingerprint,
        "hydrography": hydrography,
        "region_geometry_sha256": region_geometry_sha256(region),
        "datasets": datasets,
        "window": {"start": starttime, "end": endtime},
    }


def write_store_record(store_dir: str | os.PathLike, record: Mapping[str, Any]) -> Path:
    """Write ``record`` atomically under ``store_dir``."""
    path = Path(store_dir) / RECORD_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)
    return path


def _read_record(store_dir: Path) -> dict[str, Any] | None:
    try:
        return json.loads((store_dir / RECORD_RELPATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _stored_time_bounds(climate_nc: Path):
    import xarray as xr

    with xr.open_dataset(climate_nc) as ds:
        if "time" not in ds.coords or ds.sizes.get("time", 0) == 0:
            return None
        return pd.Timestamp(ds["time"].values[0]), pd.Timestamp(ds["time"].values[-1])


#: Days between a store's requested start and its first label: 0 for a source
#: labelled at the interval start, 1 for era5's yearly files, which label a day
#: at its end and put 1 January in the previous year's file.
_START_OFFSETS = (pd.Timedelta(0), pd.Timedelta(days=1))


def find_covering_store(
    sources: Sequence[Path],
    record: Mapping[str, Any],
    *,
    starttime: str,
    endtime: str,
    exclude: str | os.PathLike | None = None,
) -> tuple[Path | None, pd.Timestamp | None, list[str]]:
    """First source store whose record matches and whose data covers the window.

    Returns the store directory, the first label to slice from, and one short
    reason per source store considered and rejected. The slice starts where a
    fresh extraction of the request would: the requested start plus the
    store's own offset from its requested start, which is how the source labels
    a window starting on 1 January. A store with any other offset was clipped
    by its source and is not used.
    """
    excluded = Path(exclude).resolve() if exclude is not None else None
    want_start = pd.Timestamp(starttime).normalize()
    want_end = pd.Timestamp(endtime).normalize()
    reasons: list[str] = []
    for source in sources:
        root = source / HISTORICAL_STORE_DIR
        if not root.is_dir():
            reasons.append(f"{source.name}: no historical stores")
            continue
        for store in sorted(path for path in root.iterdir() if path.is_dir()):
            if excluded is not None and store.resolve() == excluded:
                continue
            if not store.name.startswith(f"{record['clim_source']}_"):
                continue
            label = f"{source.parent.name}/{store.name}"
            candidate = _read_record(store)
            if candidate is None:
                reasons.append(f"{label}: no extraction record")
                continue
            differing = [f for f in MATCH_FIELDS if candidate.get(f) != record.get(f)]
            if differing:
                reasons.append(f"{label}: differs in {', '.join(differing)}")
                continue
            climate_nc = store / "extract_historical.nc"
            if not climate_nc.is_file():
                reasons.append(f"{label}: extract missing")
                continue
            bounds = _stored_time_bounds(climate_nc)
            if bounds is None:
                reasons.append(f"{label}: no time axis")
                continue
            have_start, have_end = (bound.normalize() for bound in bounds)
            offset = have_start - pd.Timestamp(candidate["window"]["start"]).normalize()
            if offset not in _START_OFFSETS:
                reasons.append(f"{label}: source starts {have_start.date()}")
                continue
            slice_start = want_start + offset
            if have_start > slice_start or have_end < want_end:
                reasons.append(f"{label}: holds {have_start.date()}..{have_end.date()}")
                continue
            return store, slice_start, reasons
    return None, None, reasons


def project_label(path: str | os.PathLike) -> str:
    """``<trial>/<project_dir>`` for a path under a project's ``data/climate``."""
    parts = Path(path).parts
    for index in range(len(parts) - 2, 0, -1):
        if parts[index] == "data" and parts[index + 1] == "climate":
            return "/".join(parts[max(index - 2, 0) : index])
    return Path(path).name


def raw_slice_candidates(
    sources: Sequence[Path],
    project_dir: str | os.PathLike,
    raw_nc_out: str | os.PathLike,
) -> list[Path]:
    """The same raw-slice path under each source project directory."""
    relative = Path(
        os.path.relpath(Path(raw_nc_out).resolve(), Path(project_dir).resolve())
    )
    if relative.parts and relative.parts[0] == "..":
        return []
    own = Path(raw_nc_out).resolve()
    return [
        candidate
        for candidate in (source / relative for source in sources)
        if candidate.resolve() != own and candidate.is_file()
    ]


def adopt_matching_file(
    candidates: Sequence[Path],
    destination: str | os.PathLike,
    matches: Callable[[Path], bool],
) -> Path | None:
    """Copy the first candidate ``matches`` accepts to ``destination`` atomically."""
    for candidate in candidates:
        if matches(candidate):
            destination = Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            tmp = destination.with_name(destination.name + ".reuse.tmp")
            shutil.copyfile(candidate, tmp)
            os.replace(tmp, destination)
            return candidate
    return None
