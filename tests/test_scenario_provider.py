"""P1 provider preserves R arguments and selects the declared ancestor bytes."""

from pathlib import Path
from subprocess import CalledProcessError

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import CRS

from blueearth_cst.experiment.forcing_descriptor import (
    ClimateArtifact,
    UnitInterpretation,
)
from blueearth_cst.experiment.scenario_provider import (
    SourceInputs,
    describe,
    generate_roots,
    transform,
)
from blueearth_cst.experiment.scenario_rows import ScenarioRow, stochastic_rows


def test_provider_preserves_root_and_transform_arguments(tmp_path):
    rows = stochastic_rows(2, 1, unit_id_capacity=10)
    source = SourceInputs(
        tmp_path / "historical.nc",
        tmp_path / "wg.yml",
        tmp_path / "cells.csv",
        tmp_path,
        1,
        1,
    )
    seen = []

    def execute(command, log):
        seen.append((command, log))
        if command[2].endswith("generate_weather.R"):
            (tmp_path / "run_01.nc").write_bytes(b"first draw")
            (tmp_path / "run_03.nc").write_bytes(b"second draw")
        else:
            # Distinct contents catch a correctly labelled but wrongly routed ancestor.
            assert Path(command[3]).read_bytes() == b"second draw"
            Path(command[6]).write_bytes(
                Path(command[3]).read_bytes() + b" transformed"
            )

    roots = generate_roots(
        (rows[0], rows[2]), source, log_path=tmp_path / "root.log", execute=execute
    )
    assert seen[0][0] == [
        "Rscript",
        "--vanilla",
        "blueearth_cst/weathergen/generate_weather.R",
        str(source.historical_climate),
        str(source.weathergen_config),
        "1",
        "1",
        str(source.basin_cells),
        "run_01,run_03",
    ]
    output = tmp_path / "arbitrary-name.nc"
    result = transform(
        rows[3],
        roots[1],
        weathergen_config=source.weathergen_config,
        lookup_csv=tmp_path / "grid.csv",
        output_path=output,
        log_path=tmp_path / "transform.log",
        execute=execute,
    )
    assert result == ClimateArtifact("run_04", output)
    assert output.read_bytes() == b"second draw transformed"
    assert seen[1][0] == [
        "Rscript",
        "--vanilla",
        "blueearth_cst/weathergen/impose_climate_change.R",
        str(roots[1].path),
        str(source.weathergen_config),
        str(tmp_path / "grid.csv"),
        str(output),
        "1",
    ]
    assert ScenarioRow.from_record(rows[3].as_record()) == rows[3]
    with pytest.raises(ValueError, match="ancestor id"):
        transform(
            rows[3],
            roots[0],
            weathergen_config=source.weathergen_config,
            lookup_csv=tmp_path / "grid.csv",
            output_path=output,
            log_path=tmp_path / "x.log",
            execute=execute,
        )
    assert len(seen) == 2


def test_provider_propagates_failure_and_refuses_missing_output(tmp_path):
    rows = stochastic_rows(1, 1, unit_id_capacity=2)
    source = SourceInputs(
        tmp_path / "in.nc", tmp_path / "wg.yml", tmp_path / "cells.csv", tmp_path, 1, 1
    )

    def fail(command, log):
        raise CalledProcessError(7, command)

    with pytest.raises(CalledProcessError) as exc:
        generate_roots(rows[:1], source, log_path=tmp_path / "x.log", execute=fail)
    assert exc.value.returncode == 7
    with pytest.raises(FileNotFoundError, match="root missing"):
        generate_roots(
            rows[:1],
            source,
            log_path=tmp_path / "x.log",
            execute=lambda command, log: None,
        )


def _forcing(path):
    ds = xr.Dataset(
        {
            "temp": (
                ("time", "latitude", "longitude"),
                np.full((3, 2, 2), 25.0),
                {"units": "K"},
            )
        },
        coords={
            "time": pd.date_range("2046-01-01", periods=3),
            "latitude": [1.0, 0.0],
            "longitude": [9.0, 10.0],
        },
    )
    ds["spatial_ref"] = xr.DataArray(0, attrs={"crs_wkt": CRS.from_epsg(4326).to_wkt()})
    ds.to_netcdf(path)


def test_descriptor_keeps_native_attributes_separate_from_effective_units(tmp_path):
    path = tmp_path / "neutral.nc"
    _forcing(path)
    before = path.read_bytes()
    interpretation = UnitInterpretation(
        "fixture/1", "explicit fixture construction", (("temp", "degC"),)
    )
    descriptor = describe(ClimateArtifact("run_01", path), interpretation)
    variable = descriptor.variables[0]
    assert variable.units == "degC"
    assert dict(variable.native_attributes)["units"] == "K"
    assert descriptor.calendar == "proleptic_gregorian"
    assert descriptor.timestep_seconds == 86400
    assert descriptor.time_label == "interval_end"
    assert descriptor.crs == "EPSG:4326"
    assert variable.missing_count == 0
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="interpretation"):
        describe(
            ClimateArtifact("run_01", path),
            UnitInterpretation("", "", (("temp", "degC"),)),
        )
    with pytest.raises(ValueError, match="variables"):
        describe(
            ClimateArtifact("run_01", path),
            UnitInterpretation("fixture/1", "fixture", ()),
        )


def test_a_root_is_transformed_from_itself_at_the_identity_member(tmp_path):
    """t2608151154: the baseline takes the grid's own perturbation path."""
    rows = stochastic_rows(2, 1, unit_id_capacity=10)
    raw = tmp_path / "raw_01.nc"
    raw.write_bytes(b"raw root")
    seen = []

    def execute(command, log):
        seen.append(command)
        Path(command[6]).write_bytes(Path(command[3]).read_bytes() + b" mapped")

    output = tmp_path / "run_01.nc"
    transform(
        rows[0],
        ClimateArtifact("run_01", raw),
        weathergen_config=tmp_path / "wg.yml",
        lookup_csv=tmp_path / "grid.csv",
        output_path=output,
        log_path=tmp_path / "t.log",
        execute=execute,
    )
    assert seen[0][-1] == "identity"
    assert output.read_bytes() == b"raw root mapped"
    with pytest.raises(ValueError, match="ancestor id mismatch"):
        transform(
            rows[0],
            ClimateArtifact("run_03", raw),
            weathergen_config=tmp_path / "wg.yml",
            lookup_csv=tmp_path / "grid.csv",
            output_path=tmp_path / "other.nc",
            log_path=tmp_path / "t.log",
            execute=execute,
        )
