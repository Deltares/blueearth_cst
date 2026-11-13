"""The local-first overlay catalog stays generated from its two sources."""

import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "dev/scripts"))

import write_staged_catalog as staged  # noqa: E402


def test_the_committed_overlay_is_current():
    assert staged.OUTPUT.read_text(encoding="utf-8") == staged.staged_catalog(), (
        "run python dev/scripts/write_staged_catalog.py"
    )


def test_overlay_entries_are_verbatim_and_local_first():
    overlay = yaml.safe_load(staged.OUTPUT.read_text(encoding="utf-8"))
    vendored = yaml.safe_load(staged.VENDORED.read_text(encoding="utf-8"))
    assert overlay["meta"]["roots"] == [
        staged.LOCAL_ROOT,
        *vendored["meta"]["roots"],
    ]
    entries = {name: entry for name, entry in overlay.items() if name != "meta"}
    assert entries
    assert all(entry == vendored[name] for name, entry in entries.items())


def test_only_files_the_staging_tool_copies_are_overlaid():
    datasets = [
        {"path": "topography/merit_hydro_ihu/30sec", "type": "raster_glob"},
        {"path": "meteo/era5/meta/era5_orography_2018.nc", "type": "netcdf"},
    ]
    assert staged._staged("topography/merit_hydro_ihu/30sec/*.tif", datasets)
    assert not staged._staged("topography/merit_hydro_ihu/30sec/basins.gpkg", datasets)
    assert not staged._staged(
        "meteo/era5/{variable}/era5_{variable}_{year}_hourly.nc", datasets
    )


def test_the_rapid_project_lists_the_overlay_after_the_base_catalog():
    rapid = yaml.safe_load(
        (REPO / "test_case/project_config_rapid.yml").read_text(encoding="utf-8")
    )
    catalogs = rapid["project"]["catalog"]
    assert catalogs.index("toolbox:config/catalogs/deltares_data_staged.yml") == (
        catalogs.index("toolbox:config/catalogs/deltares_data_pdrive.yml") + 1
    )
