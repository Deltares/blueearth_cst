# Current guidance: dev/reference/toolbox-forcing-units.md
import hashlib
import os
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import List, Union

import hydromt
import yaml

from blueearth_cst.experiment.forcing_descriptor import UnitInterpretation
from blueearth_cst.shared.provenance import file_sha256

_REVIEWED_DAILY_BINDING = {
    "unit_add": {"temp": -273.15, "temp_min": -273.15, "temp_max": -273.15},
    "unit_mult": {"kin": 0.000277778, "kout": 0.000277778, "press_msl": 0.01},
    "rename": {
        "msl": "press_msl",
        "ssrd": "kin",
        "t2m": "temp",
        "tisr": "kout",
        "tmax": "temp_max",
        "tmin": "temp_min",
        "tp": "precip",
    },
}

# The seven variables the toolbox consumes, named as they are AFTER the rename.
# The guard below reasons about this set and ignores everything outside it.
_REVIEWED_VARIABLES = frozenset(_REVIEWED_DAILY_BINDING["rename"].values())


def _departs_from_reviewed_binding(adapter):
    """Report whether an adapter's arithmetic over the reviewed variables differs.

    The reviewed binding (the forcing-admission contract) is a statement about seven variables, not
    about the size of a catalog entry: what P1 established is the arithmetic
    `extract_historical_climate.py` applies to `precip`, `temp`, `temp_min`,
    `temp_max`, `press_msl`, `kin` and `kout`. An upstream catalog that also
    describes variables this toolbox never reads says nothing about those seven,
    so it is accepted; an upstream catalog that touches one of them in a way the
    review did not cover is rejected.

    Concretely, an adapter departs when any of these holds:

    - a reviewed conversion is missing or has a different value;
    - it converts a reviewed variable the review left unconverted (an extra
      `unit_add`/`unit_mult` entry keyed on one of the seven) — e.g. a
      `unit_mult: {precip: 1000}` that would silently rescale precipitation;
    - it renames some other native variable ONTO a reviewed name, which would
      shadow the reviewed source of that variable.

    Everything else is surplus description and is ignored. Rejecting it was the
    behaviour before the forcing-admission contract, and it made the guard a version pin on one
    catalog file rather than a check on the arithmetic: hydromt's own
    `deltares_data` catalog states the identical seven conversions and adds
    `d2m`/`u10`/`v10` renames plus an `ssr` scale, and was refused for it.
    """
    for key in ("unit_add", "unit_mult"):
        reviewed = _REVIEWED_DAILY_BINDING[key]
        found = adapter.get(key, {})
        if any(found.get(name) != value for name, value in reviewed.items()):
            return True
        if any(name in _REVIEWED_VARIABLES for name in found if name not in reviewed):
            return True
    reviewed_rename = _REVIEWED_DAILY_BINDING["rename"]
    found_rename = adapter.get("rename", {})
    if any(found_rename.get(src) != dst for src, dst in reviewed_rename.items()):
        return True
    return any(
        dst in _REVIEWED_VARIABLES
        for src, dst in found_rename.items()
        if src not in reviewed_rename
    )


def resolved_unit_interpretation(data_libs, precip_source):
    """Verify the reviewed daily source arithmetic before interpreting labels.

    This checks the consumed adapter, not merely its catalog name. CHIRPS keeps
    its precipitation/time-shift binding and inherits the six ERA5 auxiliaries
    through the existing extraction path. E-OBS remains unsupported by WF1.

    The primary source is checked over the reviewed variables only; see
    :func:`_departs_from_reviewed_binding`. The CHIRPS branch below is
    deliberately left as exact equality: its assertion is that precipitation
    carries the time shift and NO scaling at all, and `unit_mult` being empty is
    the substance of that, not an incidental shape.
    """
    paths = (
        [os.fspath(data_libs)]
        if isinstance(data_libs, (str, os.PathLike))
        else [os.fspath(item) for item in data_libs]
    )
    entries = hydromt.DataCatalog(data_libs=paths).to_dict()
    hybrid = precip_source in {"chirps", "chirps_global"}
    selected = entries["era5" if hybrid else precip_source]
    adapter = selected.get("data_adapter", {})
    if _departs_from_reviewed_binding(adapter):
        raise ValueError(
            f"UnverifiedForcingUnits: {precip_source} catalog arithmetic differs from reviewed daily binding"
        )
    if hybrid:
        adapter = entries[precip_source].get("data_adapter", {})
        if (
            adapter.get("rename") != {"precipitation": "precip"}
            or adapter.get("unit_add") != {"time": 86400}
            or adapter.get("unit_mult", {})
        ):
            raise ValueError(
                f"UnverifiedForcingUnits: {precip_source} precipitation/time adapter differs from reviewed binding"
            )
    # The evidence path below is a historical citation serialized into catalog metadata.
    # Keep that provenance stable across documentation moves; the decision-history
    # migration map resolves it. Current admission rules: dev/reference/toolbox-forcing-units.md.
    return UnitInterpretation(
        revision="daily-catalog-hydromt1.3.1-weathergenr2.0.0/2",
        evidence=(
            "dev/records/milestones/r12/implementation/evidence/p1-forcing-units.md; "
            "dev/records/decisions/0010-check-the-reviewed-variables-not-the-catalog-shape.md; "
            f"selected={precip_source}; catalogs={[(path, file_sha256(path)) for path in paths]}; "
            f"extraction_sha256={file_sha256(Path(__file__).parents[1] / 'climate_analysis' / 'extract_historical_climate.py')}"
        ),
        variables=(
            ("precip", "mm/day"),
            ("temp", "degC"),
            ("temp_min", "degC"),
            ("temp_max", "degC"),
            ("press_msl", "hPa"),
            ("kin", "W/m2"),
            ("kout", "W/m2"),
        ),
    )


def generated_forcing_reader(source_entry, source_like):
    """Retain the established generated-file reader without a source URI."""
    entry = deepcopy(source_entry)
    entry.pop("uri", None)
    entry.pop("root", None)
    entry.pop("data_adapter", None)
    entry["driver"] = {
        "name": "raster_xarray",
        "options": {"preprocess": "harmonise_dims", "lock": False},
    }
    suffix = (
        " for precipitation and era5"
        if source_like in ("chirps", "chirps_global")
        else ""
    )
    entry.setdefault("metadata", {})["processing"] = (
        f"Climate data generated from {source_like}{suffix} using Deltares/weathergenr"
    )
    return entry


def chirps_elevation_entry(source_like, oro_path):
    """Use the predecessor reader for the extraction-owned CHIRPS sidecar."""
    if oro_path is None:
        raise ValueError(f"oro_path is required for source_like={source_like!r}")
    return {
        "data_type": "RasterDataset",
        "uri": str(Path(oro_path).resolve()),
        "driver": {
            "name": "raster_xarray",
            "options": {"chunks": {"latitude": 100, "longitude": 100}, "lock": False},
        },
        "metadata": {
            "category": "topography",
            "processing": (
                f"Resampled DEM from MERIT Hydro to the resolution of {source_like}"
            ),
            "crs": 4326,
        },
    }


class IncompletePreparationContext(ValueError):
    """The selected reader's physical dependencies cannot be retained completely."""


def resolve_preparation_payloads(
    data_libs, source_like, unit_interpretation, oro_path=None
):
    """Refuse an incomplete portable closure before any collection is claimed."""
    try:
        return _resolve_preparation_payloads(
            data_libs, source_like, unit_interpretation, oro_path
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise IncompletePreparationContext(f"{source_like}: {exc}") from exc


def _resolve_preparation_payloads(
    data_libs, source_like, unit_interpretation, oro_path=None
):
    """Resolve current local catalog inputs once, preserving native elevation adapters.

    HydroMT resolves the physical URI. Read the selected local YAML entry directly
    because its serialization drops executable preprocessing options in 1.3.
    Variants require an explicit resolved entry instead of guessing their meaning.
    """
    paths = [data_libs] if isinstance(data_libs, (str, Path)) else list(data_libs)
    catalog = hydromt.DataCatalog(data_libs=paths)
    source_entry = catalog.to_dict()[source_like]
    if source_like in {"chirps", "chirps_global"}:
        elevation_entry = chirps_elevation_entry(source_like, oro_path)
        elevation_path = Path(elevation_entry["uri"])
    else:
        key = f"{source_like}_orography"
        elevation_path = Path(catalog.get_source(key).full_uri)
        elevation_entry = None
        for path in paths:
            document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
            if key in document:
                elevation_entry = document[key]
        if elevation_entry is None or "variants" in elevation_entry:
            raise ValueError(
                f"preparation requires an explicit local catalog entry for {key}"
            )
    return plan_preparation_payloads(
        source_entry, source_like, elevation_entry, elevation_path, unit_interpretation
    )


def plan_preparation_payloads(
    source_entry, source_like, elevation_entry, elevation_path, unit_interpretation
):
    """Plan the portable reader and unchanged elevation bytes before claiming a collection.

    Entries are resolved by the producer's catalog planning step. The elevation
    entry must retain its original driver and adapters, including preprocessing.
    Return the canonical context, reduced catalog bytes, and relative payload map.
    This function neither writes a collection nor changes ancillary values.
    """
    from blueearth_cst.experiment.content_identity import confined_path
    from blueearth_cst.experiment.wf4_ancillary_descriptor import describe_ancillary
    from blueearth_cst.shared.provenance import file_sha256

    path = Path(elevation_path).resolve(strict=True)
    relative = f"ancillary/elevation/{path.name}"
    confined_path(path.parent, relative)
    reader = generated_forcing_reader(source_entry, source_like)
    reader["metadata"]["cst_unit_interpretation"] = asdict(unit_interpretation)
    # JSON has arrays, not tuples; this same representation goes into YAML metadata.
    reader["metadata"]["cst_unit_interpretation"]["variables"] = [
        list(pair) for pair in unit_interpretation.variables
    ]
    elevation = deepcopy(elevation_entry)
    elevation.pop("root", None)
    elevation["uri"] = relative
    catalog = yaml.safe_dump({"elevation": elevation}, sort_keys=True).encode("utf-8")
    context = {
        "schema_version": "forcing-preparation/1",
        "generated_forcing_reader": reader,
        "pet_method": "makkink" if source_like == "eobs" else "debruin",
        "forcing_elevation": {"catalog_key": "elevation", "artifact_id": "elevation"},
        "catalog": {
            "path": "preparation_catalog.yml",
            "sha256": hashlib.sha256(catalog).hexdigest(),
        },
        "ancillary": [
            {
                "id": "elevation",
                "role": "forcing_elevation",
                "path": relative,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
                "descriptor": describe_ancillary(path),
            }
        ],
    }
    return context, catalog, {relative: path}


def prepare_clim_data_catalog(
    fns: List[Union[str, Path]],
    data_libs_like: Union[str, Path],
    source_like: str,
    fn_out: Union[str, Path] = None,
    oro_path: Union[str, Path] = None,
):
    """
    Prepares a data catalog for files path listed in fns using the same attributes as source_like
    in data_libs_like.
    If fn_out is provided writes the data catalog to that path.

    Parameters
    ----------
    fns: list(path)
        Path to the new data sources files.
    data_libs_like: str or list(str)
        Path to the existing data catalog where source_like is stored.
    source_like: str
        Data sources with the same attributes as the new sources in fns.
    fn_out: str, Optional
        If provided, writes the new data catalog to the corresponding path.
    oro_path: str or Path, Optional
        Absolute path to the ``<source_like>_orography.nc`` sidecar produced by
        the historical extraction, under the dataset+window keyed store dir. Used
        only for the chirps/chirps_global branch. Passed explicitly by the caller
        (design Â§4a) so the path is not reconstructed by fragile ``../..``
        walking from a realization NC â€” which broke on the store keying and on
        every subsequent move of the realization NC dir (now
        ``experiments/<name>/climate/weathergenr/output/``; R07 B5 moved it,
        R9 P2 moved it again, which is the point). Required
        when ``source_like`` is chirps/chirps_global.

    Returns
    -------
    climate_data_catalog: hydromt.DataCatalog
        Data catalog of the new sources in fns.
    """

    data_catalog = hydromt.DataCatalog(data_libs=data_libs_like)
    dc_like = data_catalog.to_dict()[source_like]

    climate_data_dict = dict()

    for fn in fns:
        fn = Path(fn).resolve()
        name = os.path.basename(fn).split(".")[0]
        dc_fn = generated_forcing_reader(dc_like, source_like)
        dc_fn["uri"] = str(fn)
        # The R-generated NetCDFs are written one realization per file with
        # variables already in their standard units, so override the inherited
        # driver to the netcdf-friendly raster_xarray with a harmonise pass
        # and drop any unit/rename adapters that came from the source.
        climate_data_dict[name] = dc_fn

    # Add local orography for chirps resolution
    if source_like in ("chirps", "chirps_global"):
        # The sidecar lives beside extract_historical.nc under the keyed store
        # dir; the caller passes its absolute path (design Â§4a). No ../.. walk.
        if oro_path is None:
            raise ValueError(
                f"oro_path is required for source_like={source_like!r} "
                "(the <source>_orography.nc sidecar under the keyed store dir)"
            )
        climate_data_dict[f"{source_like}_orography"] = chirps_elevation_entry(
            source_like, oro_path
        )

    if fn_out is not None:
        # Dump the dict directly with PyYAML rather than via
        # `DataCatalog().from_dict(...).to_yml(...)` because hydromt 1.3
        # silently strips driver.options.preprocess on `to_dict`/`to_yml`,
        # which would lose the `harmonise_dims` step the R-generated nc
        # files rely on (their dim order is longitude/latitude/time).
        # The downstream reader (DataCatalog(data_libs=...)) parses YAML
        # via from_dict, which DOES preserve preprocess.
        with open(fn_out, "w") as f:
            yaml.safe_dump(climate_data_dict, f, sort_keys=False)


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import tee_to_log

        with tee_to_log(sm.log[0]):
            # Read the two list of nc files and combine
            nc_fns = sm.input.st_nc
            nc_fns2 = sm.input.rlz_nc
            nc_fns.extend(nc_fns2)

            prepare_clim_data_catalog(
                fns=nc_fns,
                data_libs_like=sm.params.data_sources,
                source_like=sm.params.clim_source,
                fn_out=sm.output.clim_data,
                oro_path=getattr(sm.params, "oro_path", None),
            )
    else:
        raise ValueError("This script should be run from a snakemake environment")
