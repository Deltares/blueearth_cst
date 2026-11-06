"""Real catalog reads expose recipe selection, overlap and unit-admission errors."""

import hydromt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from blueearth_cst.climate_analysis.extract_historical_climate import (
    _extract_recipe_fields,
)
from blueearth_cst.shared.forcing_data import (
    CANONICAL_UNITS,
    recipe_unit_interpretation,
    recipe_unit_profiles,
    source_unit_profile,
    validate_daily_fields,
)
from blueearth_cst.shared.forcing_recipe import normalize_forcing_recipe
from blueearth_cst.shared.wf3_science import climate_store_rule


@pytest.fixture
def sources(tmp_path):
    coords = {"latitude": [0.3, 0.2, 0.1], "longitude": [0.1, 0.2, 0.3]}

    def field(name, value, dates=None):
        da = xr.DataArray(
            np.full((3, 3), value, dtype="float32"),
            dims=("latitude", "longitude"),
            coords=coords,
            name=name,
        )
        da.raster.set_crs(4326)
        if dates is not None:
            da = da.expand_dims(time=dates)
            da.attrs["units"] = CANONICAL_UNITS[name]
        return da

    dates = pd.date_range("2000-01-01", periods=8)
    values = {
        "precip": 5,
        "temp": 20,
        "temp_min": 15,
        "temp_max": 25,
        "press_msl": 1013.25,
        "kin": 200,
        "kout": 400,
    }
    primary = xr.Dataset(
        {name: field(name, value, dates) for name, value in values.items()}
    )
    companion = primary.drop_vars("precip").isel(time=slice(1, -1)).copy(deep=True)
    companion["temp"] += 2
    companion.temp.attrs["units"] = "degC"
    datasets = {
        "era5": primary,
        "fixture_companion": companion,
        "fixture_pet": field("pet", 7, dates[2:]).to_dataset(),
        "fixture_orography": field("elevtn", 100).to_dataset(),
        "fixture_hydrography": field("elevtn", 300).to_dataset(),
    }
    entries = {}
    for source, dataset in datasets.items():
        path = tmp_path / f"{source}.nc"
        dataset.to_netcdf(path)
        entries[source] = {
            "data_type": "RasterDataset",
            "driver": "raster_xarray",
            "uri": str(path),
            "metadata": {"crs": 4326, "time_label": "interval_end"},
        }
    return hydromt.DataCatalog().from_dict(entries), entries


def extract(catalog, recipe, tmp_path):
    return _extract_recipe_fields(
        catalog,
        recipe,
        bbox=(0.05, 0.05, 0.35, 0.35),
        requested=("2000-01-01", "2000-01-08"),
        hydrography="fixture_hydrography",
        oro_out=tmp_path / "orography.nc",
        domain=None,
    )


@pytest.mark.parametrize("correction,expected", [(True, 20.7), (False, 22)])
def test_catalog_selects_distinct_companion_and_applies_one_lapse(
    sources, tmp_path, correction, expected
):
    recipe = normalize_forcing_recipe(
        "era5",
        {
            "temp_pet_source": "fixture_companion",
            "orography_source": "fixture_orography",
            "temp_correction": correction,
            "pet": {"method": "makkink"},
        },
    )
    result = extract(sources[0], recipe, tmp_path)
    assert set(result.data_vars) == set(recipe.required_variables)
    np.testing.assert_allclose(result.temp, expected, atol=1e-5)
    np.testing.assert_allclose(result.precip, 5)
    assert result.sizes["time"] == 6
    assert pd.Timestamp(result.time.values[0]) == pd.Timestamp("2000-01-02")
    assert result.attrs["forcing_extraction_sha256"] == recipe.extraction_fingerprint


def test_direct_pet_intersects_three_sources_without_replacing_values(
    sources, tmp_path
):
    recipe = normalize_forcing_recipe(
        "era5",
        {
            "temp_pet_source": "fixture_companion",
            "orography_source": "fixture_orography",
            "pet": {"mode": "direct", "source": "fixture_pet"},
        },
    )
    result = extract(sources[0], recipe, tmp_path)
    assert set(result.data_vars) == {"precip", "temp", "pet"}
    assert result.sizes["time"] == 5
    np.testing.assert_allclose(result.pet, 7)
    path = tmp_path / "extract.nc"
    result.to_netcdf(path)
    import yaml

    catalog_path = tmp_path / "catalog.yml"
    catalog_path.write_text(yaml.safe_dump(sources[1]), encoding="utf-8")
    interpretation = recipe_unit_interpretation(catalog_path, recipe, path)
    assert dict(interpretation.variables)["pet"] == "mm/day"


def test_direct_pet_missing_outside_common_time_is_not_required(sources, tmp_path):
    entries = sources[1]
    with xr.open_dataset(entries["fixture_pet"]["uri"]) as opened:
        supplied = opened.load()
    supplied.pet.values[-1] = np.nan  # Jan 8 is beyond the companion's last day.
    path = tmp_path / "trailing-missing.nc"
    supplied.to_netcdf(path)
    entries["fixture_pet"]["uri"] = str(path)
    recipe = normalize_forcing_recipe(
        "era5",
        {
            "temp_pet_source": "fixture_companion",
            "orography_source": "fixture_orography",
            "pet": {"mode": "direct", "source": "fixture_pet"},
        },
    )
    result = extract(hydromt.DataCatalog().from_dict(entries), recipe, tmp_path)
    assert result.sizes["time"] == 5
    np.testing.assert_allclose(result.pet, 7)


@pytest.mark.parametrize("frequency", ["D", "2D", "12h"])
def test_pet_daily_total_mm_requires_a_one_day_axis(sources, tmp_path, frequency):
    entries = sources[1]
    with xr.open_dataset(entries["fixture_pet"]["uri"]) as opened:
        supplied = opened.load()
    supplied.pet.attrs["units"] = "mm"
    supplied = supplied.assign_coords(
        time=pd.date_range("2000-01-01", periods=6, freq=frequency)
    )
    path = tmp_path / "pet-total.nc"
    supplied.to_netcdf(path)
    entries["fixture_pet"]["uri"] = str(path)
    recipe = normalize_forcing_recipe(
        "era5", {"pet": {"mode": "direct", "source": "fixture_pet"}}
    )
    catalog = hydromt.DataCatalog().from_dict(entries)
    if frequency == "D":
        result = extract(catalog, recipe, tmp_path)
        np.testing.assert_allclose(result.pet, 7)
    else:
        with pytest.raises(ValueError, match="contiguous daily"):
            extract(catalog, recipe, tmp_path)


def test_direct_pet_domain_masks_only_exclude_outside_cells():
    ds = xr.Dataset(
        {"pet": (("time", "y", "x"), np.ones((4, 2, 2)))},
        coords={"time": pd.date_range("2000-01-01", periods=4)},
    )
    mask = xr.DataArray([[True, False], [True, False]], dims=("y", "x"))
    ds.pet.values[:, :, 1] = np.nan
    validate_daily_fields(ds, ("pet",), source="fixture", domain=mask)
    for bad in (np.nan, np.inf, -1):
        ds.pet.values[1, 0, 0] = bad
        with pytest.raises(ValueError, match="invalid values"):
            validate_daily_fields(ds, ("pet",), source="fixture", domain=mask)


def test_admitted_canonical_temperature_label_remains_available_in_diagnostics(
    sources, tmp_path
):
    from blueearth_cst.climate_analysis.diagnostic_tables import daily_series

    recipe = normalize_forcing_recipe(
        "era5",
        {
            "temp_pet_source": "fixture_companion",
            "orography_source": "fixture_orography",
            "pet": {"method": "makkink"},
        },
    )
    result = extract(sources[0], recipe, tmp_path)
    result.temp.attrs = {"unit": "degree C."}
    path = tmp_path / "diagnostics.nc"
    result.to_netcdf(path)
    cells = tmp_path / "basin_cells.csv"
    cells.write_text("latitude,longitude\n0.2,0.2\n", encoding="utf-8")
    daily, lineage = daily_series(path, cells, "era5", recipe=recipe)
    np.testing.assert_allclose(daily.temp, 20.7, atol=1e-5)
    assert not lineage["warnings"]


def test_daily_gap_and_unstated_time_semantics_are_rejected(sources):
    with pytest.raises(ValueError, match="time_label"):
        source_unit_profile({"metadata": {}}, ("temp",), source="fixture")
    ds = sources[0].get_rasterdataset("fixture_companion")
    with pytest.raises(ValueError, match="contiguous daily"):
        validate_daily_fields(ds.isel(time=[0, 1, 3]), ("temp",), source="fixture")


def test_roles_sharing_source_keep_separate_unit_evidence():
    from blueearth_cst.shared.forcing_data import _DAILY_BINDING

    recipe = normalize_forcing_recipe(
        "era5", {"pet": {"mode": "direct", "source": "era5"}}
    )
    evidence = recipe_unit_profiles(
        {
            "era5": {
                "data_adapter": _DAILY_BINDING,
                "metadata": {"time_label": "interval_end"},
            }
        },
        recipe,
    )
    assert evidence["companion"]["profile"] == "reviewed-era5-daily/2"
    assert evidence["pet"]["profile"] == "canonical-daily/1"
    assert evidence["pet"]["variables"] == ["pet"]


def test_shared_reviewed_source_and_canonical_pet_survive_real_catalog_read(
    sources, tmp_path
):
    import yaml

    from blueearth_cst.shared.forcing_data import _DAILY_BINDING

    entries = sources[1]
    with xr.open_dataset(entries["era5"]["uri"]) as opened:
        native = opened.load()
    native["pet"] = native.precip.copy()
    native.pet.attrs["units"] = "mm/day"
    for name, offset in _DAILY_BINDING["unit_add"].items():
        native[name] -= offset
    for name, scale in _DAILY_BINDING["unit_mult"].items():
        native[name] /= scale
    native = native.rename(
        {value: key for key, value in _DAILY_BINDING["rename"].items()}
    )
    native_path = tmp_path / "native.nc"
    native.to_netcdf(native_path)
    entries["era5"]["uri"] = str(native_path)
    entries["era5"]["data_adapter"] = _DAILY_BINDING
    catalog_path = tmp_path / "shared.yml"
    catalog_path.write_text(yaml.safe_dump(entries), encoding="utf-8")
    recipe = normalize_forcing_recipe(
        "era5", {"pet": {"mode": "direct", "source": "era5"}}
    )
    result = extract(hydromt.DataCatalog(data_libs=[catalog_path]), recipe, tmp_path)
    np.testing.assert_allclose(result.temp, 20, atol=1e-5)
    np.testing.assert_allclose(result.pet, 5)
    store = tmp_path / "shared.nc"
    result.to_netcdf(store)
    assert (
        dict(recipe_unit_interpretation(catalog_path, recipe, store).variables)["pet"]
        == "mm/day"
    )


def test_companion_and_supplied_pet_attribution_preserves_default_credit():
    from blueearth_cst.climate_analysis.prepare_climate_data_catalog import (
        generated_forcing_reader,
    )
    from blueearth_cst.shared.map_frames import forcing_attribution, map_caption

    recipe = normalize_forcing_recipe(
        "era5",
        {
            "temp_pet_source": "fixture_temperature",
            "orography_source": "fixture_dem",
            "pet": {"mode": "direct", "source": "fixture_pet"},
        },
    )
    assert "ERA5" in forcing_attribution(recipe, "precip")
    assert forcing_attribution(recipe, "temp") == "fixture_temperature"
    assert "fixture_pet (supplied reference PET)" in map_caption(
        "Annual PET", recipe=recipe, variable="pet"
    )
    reader = generated_forcing_reader({}, "era5", recipe=recipe, historical=True)
    assert "fixture_temperature" in reader["metadata"]["processing"]
    assert "fixture_pet" in reader["metadata"]["processing"]
    assert "weathergenr" not in reader["metadata"]["processing"]
    assert generated_forcing_reader({}, "era5") == generated_forcing_reader(
        {}, "era5", recipe=normalize_forcing_recipe("era5")
    )


def test_store_params_and_path_exclude_preparation_only_changes():
    args = ("project", {}, "era5", {"start": 2000, "end": 2016}, "catalog.yml")
    original = climate_store_rule(*args)
    pressure = climate_store_rule(*args, forcing={"press_correction": False})
    temperature = climate_store_rule(*args, forcing={"temp_correction": False})
    makkink = climate_store_rule(*args, forcing={"pet": {"method": "makkink"}})
    assert original == pressure == temperature
    assert original.store_dir != makkink.store_dir
    assert original.inputs == makkink.inputs
    assert "press_correction" not in makkink.params["forcing"]


def test_native_grid_projection_preserves_explicit_companion_admission():
    recipe = normalize_forcing_recipe(
        "chirps",
        {
            "temp_pet_source": "chirps",
            "orography_source": "fixture_elevation",
            "pet": {"method": "makkink"},
        },
    )
    store = climate_store_rule(
        "project",
        {},
        "chirps",
        {"start": 2000, "end": 2016},
        "catalog.yml",
        forcing=recipe.settings(),
    )
    transmitted = normalize_forcing_recipe("chirps", store.params["forcing"])
    assert transmitted.extraction_dict() == recipe.extraction_dict()
    assert "orography_source" not in transmitted.extraction_dict()


def test_nondefault_preparation_v3_retains_method_flags_and_rejects_tampering(
    sources, tmp_path, monkeypatch
):
    import copy

    import yaml

    from blueearth_cst.climate_analysis.prepare_climate_data_catalog import (
        resolve_preparation_payloads,
    )
    from blueearth_cst.experiment.wf4_ancillary_descriptor import (
        preparation_identity_v2,
        resolve_wf4_preparation,
    )

    recipe = normalize_forcing_recipe(
        "era5",
        {
            "temp_pet_source": "fixture_companion",
            "orography_source": "fixture_orography",
            "pet": {"method": "makkink"},
            "press_correction": False,
        },
    )
    extracted = extract(sources[0], recipe, tmp_path)
    climate_path = tmp_path / "extracted.nc"
    extracted.to_netcdf(climate_path)
    catalog_path = tmp_path / "catalog.yml"
    catalog_path.write_text(yaml.safe_dump(sources[1]), encoding="utf-8")
    units = recipe_unit_interpretation(catalog_path, recipe, climate_path)
    precursor, _, _ = resolve_preparation_payloads(
        [catalog_path], "era5", units, tmp_path / "orography.nc", recipe=recipe
    )
    assert precursor["pet_method"] == "makkink"
    project = tmp_path / "project"
    experiment = project / "experiments/fixture"
    preparation = resolve_wf4_preparation(
        project,
        experiment,
        climate_source="era5",
        catalogs=[catalog_path],
        unit_interpretation=units,
        oro_path=tmp_path / "orography.nc",
        recipe=recipe,
    )
    assert preparation["schema_version"] == "forcing-preparation/3"
    assert preparation["forcing_recipe"] == recipe.to_dict()
    assert preparation["pet_method"] == "makkink"
    assert preparation["press_correction"] is False
    identity = preparation_identity_v2(
        preparation, project_root=project, experiment_root=experiment
    )
    assert identity["schema_version"] == "forcing-preparation-identity/3"
    # Exercise the named flags at the neutral adapter boundary: positional
    # temperature/pressure order differs from the recipe's field order.
    from blueearth_cst.experiment import scenario_collection_v2
    from blueearth_cst.experiment.downscale_climate_forcing import (
        collection_run_forcing,
    )
    from blueearth_cst.shared.workflow_config_snapshot import file_reference

    generated = project / "scenarios/series/run_01.nc"
    generated.parent.mkdir(parents=True)
    generated.write_bytes(climate_path.read_bytes())
    marker = project / "scenarios/_engine/collections/fixture/collection.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}", encoding="utf-8")
    collection = {
        "collection_id": "a" * 64,
        "collection_revision": "b" * 64,
        "series": [
            {
                "run_id": "run_01",
                "file": file_reference(generated, "project_root", project),
            }
        ],
    }
    monkeypatch.setattr(
        scenario_collection_v2, "read_collection_v2", lambda _: collection
    )
    prepared = collection_run_forcing(
        marker, "run_01", project / "transient.yml", preparation=preparation
    )
    assert prepared.preparation_context.pet_method == "makkink"
    assert prepared.preparation_context.press_correction is False
    assert prepared.preparation_context.temp_correction is True
    wrong = copy.deepcopy(preparation)
    wrong["press_correction"] = True
    with pytest.raises(ValueError, match="press_correction differs"):
        preparation_identity_v2(wrong, project_root=project, experiment_root=experiment)


def test_composition_normalizes_defaults_and_refuses_direct_generation_before_staging():
    from copy import deepcopy
    from pathlib import Path

    import yaml

    from blueearth_cst.experiment.generation_plan import generation_configuration
    from blueearth_cst.shared.config_composition import compose_config

    path = Path(__file__).parent / "project_config_fixture.yml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    original, _ = compose_config(document, path, entry="build_model")
    explicit = deepcopy(document)
    explicit["climate"]["forcing"] = normalize_forcing_recipe(
        document["climate"]["selected"]
    ).settings()
    assert compose_config(explicit, path, entry="build_model")[0] == original
    direct = deepcopy(original)
    direct["climate"]["forcing"] = {"pet": {"mode": "direct", "source": "fixture_pet"}}
    with pytest.raises(
        ValueError, match="generate_scenarios does not support direct PET"
    ):
        generation_configuration(direct, path.parents[1])
