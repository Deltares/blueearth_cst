"""Neutral forcing compatibility and explicit execution association."""

from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from blueearth_cst.experiment.forcing_descriptor import (
    ForcingDescriptor,
    UnitInterpretation,
    VariableDescriptor,
)
from blueearth_cst.experiment.simulator_adapter import (
    ArtifactReference,
    ForcingRequirement,
    IncompatibleForcingError,
    ModelReference,
    PreparationContext,
    PreparationSettings,
    PreparedRun,
    ResponseRequest,
    RunForcing,
    batch_arguments,
    execute,
    prepare,
    requirements,
    validate_ancillary_grid,
    validate_forcing,
)
from blueearth_cst.shared.provenance import file_sha256


@pytest.fixture
def neutral_forcing(tmp_path):
    forcing = tmp_path / "unrelated-forcing.nc"
    forcing.write_bytes(b"synthetic physical input")
    catalog = tmp_path / "physical.yml"
    catalog.write_text("physical: fixture\n", encoding="utf-8")
    elevation = tmp_path / "elevation.nc"
    elevation.write_bytes(b"synthetic elevation")
    context = PreparationContext(
        (ArtifactReference(catalog, file_sha256(catalog)),),
        (ArtifactReference(elevation, file_sha256(elevation)),),
        "forcing",
        "elevation",
        "debruin",
        True,
        True,
        ("cftime_to_datetime64", "clip_to_configured_window", "refresh_toml_endpoints"),
    )
    interpretation = UnitInterpretation(
        "fixture/1", "synthetic fixture", (("temp", "degC"),)
    )
    descriptor = ForcingDescriptor(
        (VariableDescriptor("temp", "degC", ("time", "y", "x"), (), 0),),
        "EPSG:4326",
        (("time", 3), ("y", 2), ("x", 2)),
        "noleap",
        86400,
        "interval_end",
        "2001-01-01 00:00:00",
        "2001-01-03 00:00:00",
        interpretation,
    )
    requirement = ForcingRequirement(
        (("temp", "degC"),),
        "EPSG:4326",
        ("time", "y", "x"),
        ("noleap",),
        86400,
        "interval_end",
        "2001-01-01T00:00:00",
        "2001-01-03T00:00:00",
        False,
    )
    return RunForcing(
        "run_007", forcing, file_sha256(forcing), descriptor, context, None, None
    ), requirement


def test_neutral_record_reaches_dummy_preparation_and_execution(
    neutral_forcing, tmp_path, monkeypatch
):
    run, requirement = neutral_forcing
    assert not {
        "scenario_type",
        "rlz",
        "st_id",
        "derived_from",
        "pairing",
        "lookup",
    } & {field.name for field in fields(RunForcing)}
    model = ModelReference(tmp_path, requirement)
    assert requirements(model, ResponseRequest(("q",))) == requirement
    settings = PreparationSettings(
        tmp_path / "arbitrary-config.toml",
        tmp_path / "grid.nc",
        tmp_path / "arbitrary-response.csv",
        tmp_path / "arbitrary-log.log",
        requirement.first_time,
        requirement.last_time,
    )
    seen = []

    def dummy_prepare(forcing, model_reference, preparation):
        seen.append(forcing.run_id)
        assert forcing is run
        assert model_reference is model
        preparation.toml_path.write_text("fixture=true\n", encoding="utf-8")
        preparation.forcing_path.write_bytes(b"dummy")

    monkeypatch.setattr(
        "blueearth_cst.experiment.downscale_climate_forcing.prepare_model_forcing",
        dummy_prepare,
    )
    prepared = prepare("run_007", validate_forcing(run, requirement), model, settings)

    def dummy_execute(command, log_path):
        assert command[-4:] == [
            "run_007",
            "run_007",
            str(settings.toml_path),
            str(settings.native_output_path),
        ]
        settings.native_output_path.write_text("Q_1\n1\n", encoding="utf-8")
        return 0

    native = execute(
        prepared,
        julia_command=("julia",),
        log_path=tmp_path / "run.log",
        run_command=dummy_execute,
    )
    assert seen == ["run_007"]
    assert native.csv_path == settings.native_output_path
    assert native.toml_path == settings.toml_path


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"calendar": "360_day"}, "calendar"),
        ({"timestep_seconds": 3600}, "timestep_seconds"),
        ({"time_label": "instant"}, "time_label"),
        ({"first_time": "2001-01-02 00:00:00"}, "first_time"),
        ({"last_time": "2001-01-02 00:00:00"}, "last_time"),
        ({"crs": "EPSG:3857"}, "crs"),
        ({"variables": ()}, "variables.temp"),
    ],
)
def test_physical_mismatch_names_run_field_and_artifact(neutral_forcing, change, field):
    run, requirement = neutral_forcing
    run = replace(run, descriptor=replace(run.descriptor, **change))
    with pytest.raises(IncompatibleForcingError, match=field) as error:
        validate_forcing(run, requirement)
    assert "run_id=run_007" in str(error.value)
    assert str(run.forcing_path) in str(error.value)
    assert "required=" in str(error.value) and "observed=" in str(error.value)


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"units": "K"}, "units.temp"),
        ({"missing_count": 1}, "missing_count.temp"),
        ({"dimensions": ("time", "station")}, "dimensions.temp"),
    ],
)
def test_variable_metadata_refuses_without_conversion(neutral_forcing, change, field):
    run, requirement = neutral_forcing
    variable = replace(run.descriptor.variables[0], **change)
    changed = replace(run, descriptor=replace(run.descriptor, variables=(variable,)))
    with pytest.raises(IncompatibleForcingError, match=field):
        validate_forcing(changed, requirement)


@pytest.mark.parametrize("artifact", ["forcing", "catalog", "elevation"])
def test_input_freshness_is_separate_from_compatible_metadata(
    neutral_forcing, artifact
):
    run, requirement = neutral_forcing
    path = {
        "forcing": run.forcing_path,
        "catalog": run.preparation_context.catalogs[0].path,
        "elevation": run.preparation_context.ancillary[0].path,
    }[artifact]
    path.write_bytes(b"changed")
    with pytest.raises(IncompatibleForcingError, match="sha256"):
        validate_forcing(run, requirement)


def test_batch_records_keep_ids_with_arbitrary_paths(neutral_forcing, tmp_path):
    run, _ = neutral_forcing
    first = PreparedRun(
        "run_007",
        Path("b.toml"),
        Path("x.csv"),
        Path("grid.nc"),
        run.preparation_context,
    )
    second = replace(
        first,
        run_id="run_012",
        toml_path=Path("a.toml"),
        native_output_path=Path("z.csv"),
    )
    assert batch_arguments("batch-4", [first, second]) == [
        "batch-4",
        "run_007",
        "b.toml",
        "x.csv",
        "run_012",
        "a.toml",
        "z.csv",
    ]
    with pytest.raises(ValueError, match="increasing"):
        batch_arguments("batch-4", [second, first])
    with pytest.raises(ValueError, match="prefixed"):
        batch_arguments("batch-4", [replace(first, run_id="007")])


def test_incompatible_ancillary_grid_refuses_before_preparation():
    forcing = xr.DataArray(
        np.ones((2, 2)), dims=("y", "x"), coords={"y": [1.0, 0.0], "x": [0.0, 1.0]}
    )
    forcing.raster.set_crs(4326)
    validate_ancillary_grid("run_007", forcing, forcing.copy(), Path("dem.nc"))
    shifted = forcing.assign_coords(x=[10.0, 11.0])
    with pytest.raises(
        IncompatibleForcingError, match="run_id=run_007: ancillary_grid"
    ):
        validate_ancillary_grid("run_007", forcing, shifted, Path("dem.nc"))
