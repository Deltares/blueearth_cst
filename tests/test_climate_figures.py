"""The declared climate figure names (blueearth_cst/climate_analysis/climate_figures).

Pure-Python registries the Snakefiles read at parse time; drawing lives in
``model/forcing_figures.py`` and ``climate_analysis/diagnostic_render.py``.
"""

import pytest

from blueearth_cst.climate_analysis import climate_figures as cf

# --- the declared name set -------------------------------------------------


def test_figure_names_is_the_full_cross_product():
    names = cf.figure_names("source")
    assert len(names) == len(cf.CLIMATE_VARS) * len(cf.FIGURE_KINDS)
    assert len(set(names)) == len(names)
    assert all(name.startswith("source_") and name.endswith(".png") for name in names)


def test_every_dataset_gets_its_own_prefix():
    """The prefix is what makes a figure self-identifying once copied out of
    its directory, and what makes the two directories comparable."""
    source, forcing = cf.figure_names("source"), cf.figure_names("forcing")
    assert not set(source) & set(forcing)
    assert [n.replace("source_", "", 1) for n in source] == [
        n.replace("forcing_", "", 1) for n in forcing
    ]


def test_unknown_dataset_is_rejected():
    with pytest.raises(ValueError, match="unknown dataset"):
        cf.figure_names("wflow")


# --- the per-source subset -------------------------------------------------


@pytest.mark.parametrize("source", ["chirps", "chirps_global"])
def test_a_precip_only_source_draws_precipitation_only(source):
    """Candidate stores omit ERA5 fields; forcing stores must not plot them."""
    assert cf.source_climate_vars(source) == ("precip",)

    names = cf.figure_names("source", variables=cf.source_climate_vars(source))
    assert len(names) == len(cf.FIGURE_KINDS)
    assert all("precip" in name for name in names)
    assert not any("temp" in name or "pet" in name for name in names)


def test_a_full_source_still_draws_the_whole_set():
    assert cf.source_climate_vars("era5") == tuple(cf.CLIMATE_VARS)
    assert cf.figure_names("source", variables=cf.source_climate_vars("era5")) == (
        cf.figure_names("source")
    )


def test_variable_order_follows_the_canonical_set_not_the_caller():
    """Two callers naming the same variables must declare the same filenames."""
    assert cf.figure_names("source", variables=["pet", "precip"]) == cf.figure_names(
        "source", variables=["precip", "pet"]
    )


def test_an_unknown_variable_is_rejected():
    with pytest.raises(ValueError, match="unknown climate variables"):
        cf.figure_names("source", variables=["precip", "humidity"])


def test_an_empty_variable_set_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        cf.figure_names("source", variables=[])
