"""The WF3 consumer still resolves and verifies forcing adapter arithmetic."""

import sys
from types import SimpleNamespace

import pytest

from blueearth_cst.experiment.generation_sources import generation_interpretation


@pytest.mark.parametrize("changed_units", [False, True])
def test_consumer_loads_catalog_and_refuses_changed_arithmetic(
    tmp_path, monkeypatch, changed_units
):
    catalog = tmp_path / "catalog.yml"
    catalog.write_text("era5: {}\n", encoding="utf-8")
    adapter = {
        "rename": {
            "msl": "press_msl",
            "ssrd": "kin",
            "t2m": "temp",
            "tisr": "kout",
            "tmax": "temp_max",
            "tmin": "temp_min",
            "tp": "precip",
        },
        "unit_add": {"temp": -273.15, "temp_min": -273.15, "temp_max": -273.15},
        "unit_mult": {"press_msl": 0.01, "kin": 1 / 3600, "kout": 1 / 3600},
    }
    # Use the catalog's declared conversion precision.
    adapter["unit_mult"].update(kin=0.000277778, kout=0.000277778)
    if changed_units:
        adapter["unit_mult"]["press_msl"] = 1
    reads = []

    def load_catalog(*, data_libs):
        reads.append(data_libs)
        return SimpleNamespace(to_dict=lambda: {"era5": {"data_adapter": adapter}})

    monkeypatch.setitem(
        sys.modules, "hydromt", SimpleNamespace(DataCatalog=load_catalog)
    )
    if changed_units:
        with pytest.raises(ValueError, match="UnverifiedForcingUnits"):
            generation_interpretation([catalog], "era5")
    else:
        interpretation = generation_interpretation([catalog], "era5")
        assert dict(interpretation.variables)["temp"] == "degC"
        assert dict(interpretation.variables)["press_msl"] == "hPa"
    assert reads == [[str(catalog.resolve())]]
