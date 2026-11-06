"""WF3-owned interpretation of the historical climate extraction binding."""

from pathlib import Path
from typing import Sequence

from blueearth_cst.experiment.forcing_descriptor import UnitInterpretation
from blueearth_cst.shared.forcing_data import (
    departs_from_reviewed_binding as _departs_from_binding,
)
from blueearth_cst.shared.provenance import file_sha256


def generation_interpretation(
    catalogs: Sequence[str | Path], climate: str, *, recipe=None, climate_nc=None
) -> UnitInterpretation:
    """Resolve only WF3-consumed units; refuse changed extraction arithmetic."""
    import hydromt

    from blueearth_cst.shared.forcing_recipe import normalize_forcing_recipe

    if (
        recipe is not None
        and recipe.extraction_fingerprint
        != normalize_forcing_recipe(climate).extraction_fingerprint
    ):
        from blueearth_cst.shared.forcing_data import recipe_unit_interpretation

        return recipe_unit_interpretation(catalogs, recipe, climate_nc)

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
