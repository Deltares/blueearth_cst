"""Wrap weathergenr operations behind the stochastic provider seam."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from subprocess import CalledProcessError

from blueearth_cst.experiment.forcing_descriptor import (
    ClimateArtifact,
    ForcingDescriptor,
    UnitInterpretation,
    describe_forcing,
)
from blueearth_cst.experiment.scenario_rows import (
    ScenarioRow,
    enumerate_stochastic,
)


@dataclass(frozen=True)
class SourceInputs:
    """Current R generation inputs, including its explicit filename widths."""

    historical_climate: Path
    weathergen_config: Path
    basin_cells: Path
    output_dir: Path
    rlz_width: int
    st_width: int


def enumerate_scenarios(
    generation_spec: Mapping, *, unit_id_capacity: int
) -> tuple[ScenarioRow, ...]:
    """Enumerate the production stochastic binding without source I/O."""
    return enumerate_stochastic(generation_spec, unit_id_capacity=unit_id_capacity)


def _execute(command: list[str], log_path: Path) -> None:
    """Preserve the existing live log and child exit-status behavior."""
    from blueearth_cst.shared.run_log_core import run_and_tee

    result = run_and_tee(command, log_path)
    if result:
        raise CalledProcessError(result, command)


def generate_roots(
    root_rows: Sequence[ScenarioRow],
    source_inputs: SourceInputs,
    *,
    log_path: Path,
    execute: Callable[[list[str], Path], None] = _execute,
) -> tuple[ClimateArtifact, ...]:
    """Generate roots directly under their allocated run IDs."""
    if not root_rows:
        raise ValueError("generate_roots needs nonempty root rows")
    expected_draws = [
        f"{i:0{source_inputs.rlz_width}d}" for i in range(1, len(root_rows) + 1)
    ]
    if [dict(row.payload).get("rlz") for row in root_rows] != expected_draws:
        raise ValueError("root rows must cover the configured realization order")
    if any(
        row.scenario_type != "stochastic"
        or row.derived_from
        or dict(row.payload).get("st_id") != ""
        for row in root_rows
    ):
        raise ValueError("generate_roots accepts stochastic roots only")
    run_ids = [row.run_id for row in root_rows]
    if len(set(run_ids)) != len(run_ids) or any(
        not item.startswith("run_")
        or not item[4:].isascii()
        or not item[4:].isdecimal()
        for item in run_ids
    ):
        raise ValueError("root run IDs must be unique prefixed ASCII decimal strings")
    command = [
        "Rscript",
        "--vanilla",
        "blueearth_cst/weathergen/generate_weather.R",
        str(source_inputs.historical_climate),
        str(source_inputs.weathergen_config),
        str(source_inputs.rlz_width),
        str(source_inputs.st_width),
        str(source_inputs.basin_cells),
        ",".join(run_ids),
    ]
    execute(command, log_path)
    artifacts = tuple(
        ClimateArtifact(
            row.run_id,
            source_inputs.output_dir / f"{row.run_id}.nc",
        )
        for row in root_rows
    )
    for artifact in artifacts:
        if not artifact.path.is_file():
            raise FileNotFoundError(
                f"run_id={artifact.run_id}: root missing {artifact.path}"
            )
    return artifacts


#: The member token of the unperturbed baseline. It has no lookup row; the
#: perturbation step synthesizes zero change for it (t2608151154).
IDENTITY_MEMBER = "identity"


def transform(
    derived_row: ScenarioRow,
    ancestor_artifact: ClimateArtifact,
    *,
    weathergen_config: Path,
    lookup_csv: Path,
    output_path: Path,
    log_path: Path,
    execute: Callable[[list[str], Path], None] = _execute,
) -> ClimateArtifact:
    """Consume the exact ancestor selected by the validated row's edge.

    A ROOT row (no ``st_id``) is transformed too, from its own raw generated
    series at unit factors, so the baseline reaches the collection through the
    same perturbation step as every grid member (t2608151154).
    """
    payload = dict(derived_row.payload)
    if derived_row.scenario_type != "stochastic":
        raise ValueError("transform requires a stochastic row")
    root = not payload.get("st_id")
    ancestor_id = derived_row.run_id if root else derived_row.derived_from
    if ancestor_id != ancestor_artifact.run_id:
        raise ValueError(f"run_id={derived_row.run_id}: ancestor id mismatch")
    if not ancestor_artifact.path.is_file():
        raise FileNotFoundError(ancestor_artifact.path)
    if output_path.resolve() == ancestor_artifact.path.resolve():
        raise ValueError("transform cannot overwrite its ancestor")
    command = [
        "Rscript",
        "--vanilla",
        "blueearth_cst/weathergen/impose_climate_change.R",
        str(ancestor_artifact.path),
        str(weathergen_config),
        str(lookup_csv),
        str(output_path),
        IDENTITY_MEMBER if root else payload["st_id"],
    ]
    execute(command, log_path)
    if not output_path.is_file():
        raise FileNotFoundError(
            f"run_id={derived_row.run_id}: transform missing {output_path}"
        )
    return ClimateArtifact(derived_row.run_id, output_path)


def describe(
    artifact: ClimateArtifact, interpretation: UnitInterpretation
) -> ForcingDescriptor:
    """Describe the provider's right-labelled daily forcing without conversion."""
    return describe_forcing(artifact, interpretation, time_label="interval_end")
