"""WF3-owned interpretation of the historical climate extraction binding."""

from pathlib import Path
from typing import Sequence

from blueearth_cst.experiment.forcing_descriptor import UnitInterpretation
from blueearth_cst.shared.provenance import file_sha256

_BINDING = {
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
_VARIABLES = frozenset(_BINDING["rename"].values())


def _departs_from_binding(adapter: dict) -> bool:
    for key in ("unit_add", "unit_mult"):
        required = _BINDING[key]
        found = adapter.get(key, {})
        if any(found.get(name) != value for name, value in required.items()):
            return True
        if any(name in _VARIABLES for name in found if name not in required):
            return True
    found = adapter.get("rename", {})
    if any(found.get(name) != value for name, value in _BINDING["rename"].items()):
        return True
    return any(
        value in _VARIABLES
        for name, value in found.items()
        if name not in _BINDING["rename"]
    )


def generation_interpretation(
    catalogs: Sequence[str | Path], climate: str
) -> UnitInterpretation:
    """Resolve only WF3-consumed units; refuse changed extraction arithmetic."""
    import hydromt

    paths = [str(Path(path).resolve()) for path in catalogs]
    entries = hydromt.DataCatalog(data_libs=paths).to_dict()
    hybrid = climate in {"chirps", "chirps_global"}
    primary = entries["era5" if hybrid else climate].get("data_adapter", {})
    if _departs_from_binding(primary):
        raise ValueError(
            f"UnverifiedForcingUnits: {climate} catalog arithmetic differs"
        )
    if hybrid:
        precipitation = entries[climate].get("data_adapter", {})
        if (
            precipitation.get("rename") != {"precipitation": "precip"}
            or precipitation.get("unit_add") != {"time": 86400}
            or precipitation.get("unit_mult", {})
        ):
            raise ValueError(
                f"UnverifiedForcingUnits: {climate} precipitation binding differs"
            )
    return UnitInterpretation(
        revision="daily-catalog-hydromt1.3.1-weathergenr2.0.0/2",
        evidence=(
            f"selected={climate}; catalogs={[(path, file_sha256(path)) for path in paths]}; "
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
