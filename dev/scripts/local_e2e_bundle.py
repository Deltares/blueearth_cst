"""Seal and verify source-only inputs for the rapid all-workflow E2E gate.

Stage the hydromt mirror with ``stage_data.py --config dev/scripts/local_e2e_data.yml``
and copy the four pinned CMIP6 raw slices into ``<bundle>/cmip6/raw`` first.
``seal`` checks coverage and writes the file inventory; ``verify`` refuses missing,
changed, or unexpected files and changed catalog/config contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import xarray as xr

REPO = Path(__file__).resolve().parents[2]
MANIFEST = "bundle.json"
RAW_NAMES = tuple(
    f"cmip6_{model}_{experiment}_r1i1p1f1.nc"
    for model in ("INM_INM-CM4-8", "NOAA-GFDL_GFDL-ESM4")
    for experiment in ("historical", "ssp245")
)
ERA5_VARIABLES = {"tp", "t2m", "tmin", "tmax", "ssrd", "tisr", "msl"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contract_files() -> list[Path]:
    return [
        REPO / "dev/scripts/local_e2e_data.yml",
        REPO / "dev/scripts/stage_data.py",
        REPO / "dev/scripts/stage_cmip6.yml",
        REPO / "dev/scripts/stage_cmip6.py",
        REPO / "config/catalogs/deltares_data_pdrive.yml",
        REPO / "config/catalogs/deltares_era5_daily_zarr.yml",
        REPO / "config/catalogs/cmip6_data.yml",
        REPO / "config/catalogs/cmip6_store_index.json",
        *sorted((REPO / "test_case").glob("project_config_rapid*.yml")),
    ]


def _contracts() -> dict[str, str]:
    return {
        path.relative_to(REPO).as_posix(): _sha256(path) for path in _contract_files()
    }


def _inventory(bundle: Path) -> dict[str, dict[str, int | str]]:
    result = {}
    for parent, dirs, files in os.walk(bundle, followlinks=False):
        for name in dirs:
            if (Path(parent) / name).is_symlink():
                raise ValueError(f"bundle contains a directory link: {name}")
        for name in files:
            path = Path(parent) / name
            if path.is_symlink():
                raise ValueError(f"bundle contains a file link: {path}")
            rel = path.relative_to(bundle).as_posix()
            if rel in (MANIFEST, f".{MANIFEST}.tmp"):
                continue
            result[rel] = {"size": path.stat().st_size, "sha256": _sha256(path)}
    return dict(sorted(result.items()))


def _check_sources(bundle: Path) -> str:
    hydromt = bundle / "hydromt"
    raw = bundle / "cmip6/raw"
    if {p.name for p in raw.glob("*.nc")} != set(RAW_NAMES):
        raise ValueError("bundle must contain exactly the four rapid CMIP6 raw slices")

    years = range(1999, 2017)
    chirps = hydromt / "meteo/chirps_africa_daily_v2.0"
    if {p.name for p in chirps.glob("*.nc")} != {
        f"CHIRPS_rainfall_{year}.nc" for year in years
    }:
        raise ValueError("CHIRPS does not cover the 1999 boundary and 2000–2016")
    for year in years:
        with xr.open_dataset(chirps / f"CHIRPS_rainfall_{year}.nc") as data:
            expected_days = 1 if year == 1999 else (366 if year % 4 == 0 else 365)
            if "precipitation" not in data or data.time.size != expected_days:
                raise ValueError(f"CHIRPS {year} lacks daily precipitation")
            if data["precipitation"].attrs.get("units") != "mm":
                raise ValueError(f"CHIRPS {year} units differ from the source")
            first = f"{year}-12-31" if year == 1999 else f"{year}-01-01"
            if (
                str(data.time.values[0])[:10] != first
                or str(data.time.values[-1])[:10] != f"{year}-12-31"
            ):
                raise ValueError(f"CHIRPS {year} has the wrong calendar window")

    with xr.open_zarr(hydromt / "meteo/era5_daily.zarr") as data:
        if set(data.data_vars) != ERA5_VARIABLES:
            raise ValueError("ERA5 variables differ from the rapid catalog request")
        era5_units = {
            "msl": "Pa",
            "ssrd": "J m**-2",
            "t2m": "K",
            "tisr": "J m**-2",
            "tmax": "K",
            "tmin": "K",
            "tp": "mm d**-1",
        }
        if {name: data[name].attrs.get("units") for name in data} != era5_units:
            raise ValueError("ERA5 native units differ from the pinned source")
        if (
            str(data.time.values[0])[:10] > "2000-01-01"
            or str(data.time.values[-1])[:10] < "2016-12-31"
        ):
            raise ValueError("ERA5 does not cover the rapid historical window")

    required = (
        "topography/merit_hydro/basin_index.fgb",
        "hydrography/rivers_lin2019/rivers_ge30m.fgb",
        "landuse/globcover/GLOBCOVER_L4_200901_200912_V2.3.tif",
        "hydrography/lakes/lake-db.fgb",
        "hydrography/reservoirs/reservoir-db.fgb",
        "hydrography/rgi/rgi.fgb",
        "meteo/era5/meta/era5_orography_2018.nc",
    )
    for rel in required:
        if not (hydromt / rel).is_file():
            raise ValueError(f"missing pinned source: {rel}")
    raster_counts = {
        "topography/merit_hydro_ihu/30sec": 14,
        "landuse/modis/MODIS_MCD15A3H_LAI": 12,
        "soil/soilgrids_v1.0": 45,
    }
    for rel, count in raster_counts.items():
        found = len(list((hydromt / rel).glob("*.tif")))
        if found != count:
            raise ValueError(
                f"pinned raster family {rel} has {found}, expected {count}"
            )

    fingerprints = set()
    for name in RAW_NAMES:
        with xr.open_dataset(raw / name) as data:
            if set(data.data_vars) != {"precip", "temp"}:
                raise ValueError(f"CMIP6 variables differ: {name}")
            units = {variable: data[variable].attrs.get("units") for variable in data}
            if units != {"precip": "mm/day", "temp": "degC"}:
                raise ValueError(f"CMIP6 units differ: {name}")
            expected_months = 780 if "_historical_" in name else 1032
            if data.time.size != expected_months:
                raise ValueError(f"CMIP6 acquisition window is incomplete: {name}")
            if not data.attrs.get("cst_raw_digest"):
                raise ValueError(f"CMIP6 raw digest absent: {name}")
            if not data.attrs.get("cst_calendar"):
                raise ValueError(f"CMIP6 source calendar absent: {name}")
            fingerprints.add(data.attrs.get("cst_region_fingerprint"))
    if len(fingerprints) != 1 or None in fingerprints:
        raise ValueError("CMIP6 raw slices disagree on region fingerprint")
    return fingerprints.pop()


def seal_bundle(bundle: Path) -> None:
    """Validate source coverage and write a deterministic bundle manifest."""
    bundle = bundle.resolve(strict=True)
    fingerprint = _check_sources(bundle)
    manifest = {
        "schema": 1,
        "region_fingerprint": fingerprint,
        "contracts": _contracts(),
        "files": _inventory(bundle),
    }
    target = bundle / MANIFEST
    if target.is_symlink():
        raise ValueError("bundle manifest is a link")
    temporary = bundle / f".{MANIFEST}.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def verify_bundle(bundle: Path) -> dict:
    """Refuse a missing, stale, incomplete, or modified local input bundle."""
    bundle = bundle.resolve(strict=True)
    target = bundle / MANIFEST
    if target.is_symlink() or not target.is_file():
        raise ValueError("bundle manifest is missing or linked")
    manifest = json.loads(target.read_text(encoding="utf-8"))
    if manifest.get("schema") != 1 or manifest.get("contracts") != _contracts():
        raise ValueError("bundle schema or rapid config/catalog contracts changed")
    if manifest.get("files") != _inventory(bundle):
        raise ValueError("bundle file inventory or checksums changed")
    if manifest.get("region_fingerprint") != _check_sources(bundle):
        raise ValueError("bundle CMIP6 region identity changed")
    return manifest


def main() -> None:
    """Seal or verify a bundle staged outside Git."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("seal", "verify"))
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    if args.operation == "seal":
        seal_bundle(args.bundle)
    else:
        verify_bundle(args.bundle)
    print(f"{args.operation} OK: {args.bundle}")


if __name__ == "__main__":
    main()
