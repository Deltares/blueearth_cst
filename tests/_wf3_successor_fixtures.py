"""Test-only non-stochastic provider and non-Wflow simulator; never registered."""

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from blueearth_cst.experiment.forcing_descriptor import (
    ClimateArtifact,
    ForcingDescriptor,
    UnitInterpretation,
    VariableDescriptor,
)
from blueearth_cst.experiment.response_series import ResponseSeries
from blueearth_cst.experiment.scenario_rows import ScenarioRow
from blueearth_cst.experiment.simulator_adapter import RunForcing
from blueearth_cst.shared.provenance import file_sha256


class SyntheticProvider:
    """A fixture scenario type without stochastic payload or native filenames."""

    @staticmethod
    def enumerate_scenarios():
        return (
            ScenarioRow("run_01", "", True, "fixture", (("level", "1"),)),
            ScenarioRow("run_02", "", True, "fixture", (("level", "2"),)),
        )

    @staticmethod
    def generate(row, path):
        np.savez(
            path,
            time=pd.date_range("2046-01-01", periods=365).values,
            precip=np.full(365, float(dict(row.payload)["level"])),
        )
        return ClimateArtifact(row.run_id, path)

    @staticmethod
    def describe(artifact):
        with np.load(artifact.path) as data:
            times = pd.DatetimeIndex(data["time"])
            missing = int((~np.isfinite(data["precip"])).sum())
        return ForcingDescriptor(
            (VariableDescriptor("precip", "mm/day", ("time",), (), missing),),
            "fixture-point",
            (("time", len(times)),),
            "standard",
            86400,
            "interval_end",
            str(times[0]),
            str(times[-1]),
            UnitInterpretation(
                "fixture/1", "explicit synthetic values", (("precip", "mm/day"),)
            ),
        )


@dataclass(frozen=True)
class DummyPreparedRun:
    run_id: str
    forcing_path: Path
    output_path: Path


class DummySimulator:
    """Minimal fixture implementation of the same five logical operations."""

    @staticmethod
    def requirements(model_reference, response_request):
        assert response_request.variables == ("q",)
        return ("precip", "mm/day")

    @staticmethod
    def validate_forcing(run_forcing: RunForcing, requirement):
        assert file_sha256(run_forcing.forcing_path) == run_forcing.forcing_sha256
        variable = run_forcing.descriptor.variables[0]
        assert (variable.name, variable.units) == requirement
        return run_forcing

    @staticmethod
    def prepare(run_id, validated_forcing, model_reference, settings):
        assert run_id == validated_forcing.run_id
        return DummyPreparedRun(run_id, validated_forcing.forcing_path, settings)

    @staticmethod
    def execute(prepared_run):
        with np.load(prepared_run.forcing_path) as data:
            np.savez(
                prepared_run.output_path, time=data["time"], values=2 * data["precip"]
            )
        return prepared_run.output_path

    @staticmethod
    def open_responses(run_id, native_artifacts, response_request):
        assert response_request.variables == ("q",)
        with np.load(native_artifacts) as data:
            values = data["values"]
            return (
                ResponseSeries(
                    run_id,
                    "q",
                    "point",
                    0,
                    pd.DatetimeIndex(data["time"]),
                    "standard",
                    timedelta(days=1),
                    "interval_end",
                    "m3 s-1",
                    values,
                    np.isnan(values),
                ),
            )
