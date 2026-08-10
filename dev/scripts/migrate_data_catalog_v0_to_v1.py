"""Migrate a hydromt v0.x data catalog YAML to the v1.x schema.

Mapping derived from the hydromt test fixture
tests/data/test_v0_data_catalog{,_upgraded}.yml in the upstream repo.

v0 → v1 transformations (per source):
- ``path:``                              → ``uri:``
- ``meta:``                              → ``metadata:``
- top-level ``crs:``, ``nodata:``,
  ``attrs:``                             → moved under ``metadata:``
- ``driver: <string>``                   → ``driver: {name: <new>, options: {...}}``
  with name mapping vector→pyogrio (GeoDataFrame),
  csv (GeoDataFrame)→geodataframe_table, raster→rasterio,
  netcdf→raster_xarray for RasterDataset / geodataset_xarray for GeoDataset,
  zarr→raster_xarray, raster_tindex→rasterio (with mosaic options).
- top-level ``kwargs:``                  → ``driver.options:``
- top-level ``unit_add:``, ``unit_mult:`` → ``data_adapter: {...}``

v0 → v1 transformations (top-level meta):
- ``meta.root: <p>`` → ``roots: [<p>]``
- other meta keys preserved as-is under top-level (informational only).

Usage:
    pixi run python dev/scripts/migrate_data_catalog_v0_to_v1.py <in.yml> <out.yml>
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

_NETCDF_DRIVER_BY_TYPE = {
    "RasterDataset": "raster_xarray",
    "GeoDataset": "geodataset_xarray",
}


def _migrate_driver(
    v0_driver: str | None, data_type: str, kwargs: dict | None
) -> dict | None:
    """Return a v1 driver dict, or None if no driver was specified."""
    if v0_driver is None and not kwargs:
        return None

    if v0_driver == "vector":
        name = "pyogrio" if data_type == "GeoDataFrame" else v0_driver
    elif v0_driver in ("csv", "xlsx", "xls", "parquet") and data_type == "GeoDataFrame":
        name = "geodataframe_table"
    elif v0_driver == "raster":
        name = "rasterio"
    elif v0_driver == "raster_tindex":
        name = "rasterio"
    elif v0_driver == "netcdf":
        name = _NETCDF_DRIVER_BY_TYPE.get(data_type, "raster_xarray")
    elif v0_driver == "zarr":
        name = "raster_xarray"
    else:
        name = v0_driver  # already-v1 / passthrough

    out: dict = {"name": name}
    if kwargs:
        out["options"] = kwargs
    return out


def _migrate_source(name: str, v0: dict) -> dict:
    v0 = dict(v0)  # shallow copy
    data_type = v0.get("data_type", "RasterDataset")

    out: dict = {"data_type": data_type}

    # uri
    if "path" in v0:
        out["uri"] = v0.pop("path")

    # driver + driver.options + driver.filesystem
    driver = _migrate_driver(v0.pop("driver", None), data_type, v0.pop("kwargs", None))
    filesystem = v0.pop("filesystem", None)
    if filesystem is not None:
        if driver is None:
            driver = {}
        driver["filesystem"] = filesystem
    if driver is not None:
        out["driver"] = driver

    # data_adapter (unit_add / unit_mult / rename)
    data_adapter: dict = {}
    if "unit_add" in v0:
        data_adapter["unit_add"] = v0.pop("unit_add")
    if "unit_mult" in v0:
        data_adapter["unit_mult"] = v0.pop("unit_mult")
    if "rename" in v0:
        data_adapter["rename"] = v0.pop("rename")
    if data_adapter:
        out["data_adapter"] = data_adapter

    # metadata: rename `meta` and absorb top-level crs/nodata/attrs
    metadata: dict = dict(v0.pop("meta", {}) or {})
    for moved in ("crs", "nodata", "attrs"):
        if moved in v0:
            metadata[moved] = v0.pop(moved)
    if metadata:
        out["metadata"] = metadata

    # alias should be fully resolved by _resolve_aliases at this point;
    # drop any leftover alias key defensively (v1 doesn't support it).
    v0.pop("alias", None)

    # everything else: preserve so we don't silently drop fields
    for leftover_key, leftover_val in v0.items():
        out[leftover_key] = leftover_val

    return out


def _resolve_aliases(d: dict) -> dict:
    """Inline alias entries: hydromt 1.x doesn't support `alias:` so we copy
    the target's full body into the alias entry. Override keys on the alias
    entry shadow the target."""
    resolved = dict(d)
    changed = True
    safety = 0
    while changed:
        changed = False
        safety += 1
        if safety > 50:
            raise RuntimeError("Alias resolution did not converge — possible cycle")
        for name, body in list(resolved.items()):
            if isinstance(body, dict) and "alias" in body:
                target = body["alias"]
                if target not in resolved:
                    raise RuntimeError(f"alias {name} → {target} but {target} missing")
                if isinstance(resolved[target], dict) and "alias" in resolved[target]:
                    continue  # let target resolve first
                merged = dict(resolved[target])
                overrides = {k: v for k, v in body.items() if k != "alias"}
                merged.update(overrides)
                resolved[name] = merged
                changed = True
    return resolved


def _migrate_top(d: dict) -> dict:
    out: dict = {}
    if "meta" in d:
        meta = d.pop("meta")
        if isinstance(meta, dict) and "root" in meta:
            meta["roots"] = [meta.pop("root")]
        if meta:
            out["meta"] = meta

    d = _resolve_aliases(d)

    for name, body in d.items():
        if isinstance(body, dict):
            out[name] = _migrate_source(name, body)
        else:
            out[name] = body
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            "usage: migrate_data_catalog_v0_to_v1.py <in.yml> <out.yml>",
            file=sys.stderr,
        )
        return 2
    in_fn = Path(argv[1])
    out_fn = Path(argv[2])

    with in_fn.open("r", encoding="utf-8") as f:
        v0 = yaml.safe_load(f)

    v1 = _migrate_top(v0)

    out_fn.parent.mkdir(parents=True, exist_ok=True)
    with out_fn.open("w", encoding="utf-8") as f:
        yaml.safe_dump(v1, f, sort_keys=False, allow_unicode=True)

    print(f"wrote {out_fn} ({len(v1)} top-level keys)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
