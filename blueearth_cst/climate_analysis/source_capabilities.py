"""Scientific variable ownership, independent of figure declarations."""

from blueearth_cst.shared.variable_registry import CLIMATE_VARS

PRECIP_ONLY_SOURCES = ("chirps", "chirps_global")


def source_climate_vars(clim_source: str) -> tuple[str, ...]:
    """Return only variables belonging to this source, excluding borrowed fields."""
    if clim_source in PRECIP_ONLY_SOURCES:
        return ("precip",)
    return tuple(CLIMATE_VARS)
