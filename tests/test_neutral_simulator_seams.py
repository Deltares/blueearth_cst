"""A fixture without stochastic fields or Wflow output reaches the same reducer."""

from blueearth_cst.experiment.indicator_registry import declarations, reduce_run
from blueearth_cst.experiment.simulator_adapter import PreparationContext, RunForcing
from blueearth_cst.experiment.wflow_response_reader import ResponseRequest
from blueearth_cst.shared.provenance import file_sha256
from tests._wf3_successor_fixtures import DummySimulator, SyntheticProvider


def test_synthetic_provider_dummy_simulator_neutral_indicator(tmp_path):
    provider, simulator = SyntheticProvider(), DummySimulator()
    request = ResponseRequest(("q",))
    values = []
    for ordinal, row in enumerate(provider.enumerate_scenarios()):
        assert not {"rlz", "st_id", "derived_from"} & dict(row.payload).keys()
        artifact = provider.generate(row, tmp_path / f"source-{ordinal}.npz")
        # This simulator needs no catalog, elevation or PET operations.
        context = PreparationContext((), (), "", "", "not_applicable", False, False, ())
        forcing = RunForcing(
            row.run_id,
            artifact.path,
            file_sha256(artifact.path),
            provider.describe(artifact),
            context,
            None,
            None,
        )
        validated = simulator.validate_forcing(
            forcing, simulator.requirements(None, request)
        )
        prepared = simulator.prepare(
            row.run_id, validated, None, tmp_path / f"result-{ordinal}.npz"
        )
        native = simulator.execute(prepared)
        series = simulator.open_responses(row.run_id, native, request)
        values.append(
            reduce_run(declarations(("q",))[0], series, anchor="YE-DEC")["point"]
        )
    assert values == [2.0, 4.0]
