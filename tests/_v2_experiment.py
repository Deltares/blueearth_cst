"""A retained v2 experiment built by the real WF3/WF4 record producers.

Every record here is written by the production v2 producer and is read back
through the production readers with nothing stubbed: `read_collection_v2`,
`read_simulation_intent_v2`, `read_response_inventory_v2` and
`read_simulation_v2`. Only probes outside the record contract are replaced --
the stage environment (a synthetic package set, as the indicator tests have
always used) and HydroMT's elevation payload resolution, which would otherwise
read a real data catalog.

`build_v2_experiment` is slow enough to share: build it once per module with
`tmp_path_factory` and hand each test a `copy_v2_experiment` of it. Every
reference is project-relative, so a copied project is a complete one.

Not a test module, so it is never collected and no test file imports a
fixture from another (the coupling that made the v1 records hard to retire).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

#: The stage environment indicator planning and reduction observe (see
#: `indicator_plan.resolve_indicator_environment`).
FIXTURE_ENVIRONMENT = {"packages": {"numpy": "fixture"}}

EXPERIMENT = "indicator_experiment"
RUN_IDS = ("run_1", "run_2")
INVOCATION_ID = "f" * 32
CLIMATE_SOURCE = "era5"

#: The two retained native runs: one recharge series at one location, daily
#: over two days, labelled at interval end.
NATIVE_CSV = "time,Q_9,Q_2,gwr_9\n2046-01-02,1,2,3\n2046-01-03,4,,6\n"
NATIVE_TOML = (
    '[time]\ncalendar="standard"\ntimestepsecs=86400\n'
    'starttime="2046-01-01T00:00:00"\nendtime="2046-01-03T00:00:00"\n'
    '[[output.csv.column]]\nheader="Q"\n'
    'parameter="river_water__volume_flow_rate"\n'
    '[[output.csv.column]]\nheader="gwr"\n'
    'parameter="soil_water_saturated_zone_top__net_recharge_volume_flux"\n'
)
TEMPORAL = {
    "source_calendar": "proleptic_gregorian",
    "prepared_forcing_calendar": "proleptic_gregorian",
    "response_calendar": "standard",
    "operations": ["clip_to_configured_window", "refresh_toml_endpoints"],
    "prepared_start": "2046-01-01T00:00:00",
    "prepared_end": "2046-01-03T00:00:00",
    "response_start": "2046-01-02 00:00:00",
    "response_end": "2046-01-03 00:00:00",
    "time_label": "interval_end",
}


@dataclass(frozen=True)
class Responses:
    """What the retained native runs hold, and what the frozen request asks."""

    csv: dict[str, bytes]
    toml: str
    temporal: dict
    variables: list[dict]


def default_responses() -> Responses:
    """Two days of recharge at one location, the same in both runs."""
    return Responses(
        csv={run: NATIVE_CSV.encode("utf-8") for run in RUN_IDS},
        toml=NATIVE_TOML,
        temporal=TEMPORAL,
        variables=[
            {
                "variable": "gwr",
                "locations": ["9"],
                "units": "mm dt-1",
                "calendar": "standard",
                "timestep": "P1D",
                "time_label": "interval_end",
                "start": TEMPORAL["response_start"],
                "end": TEMPORAL["response_end"],
                "missing_value": "NaN",
            }
        ],
    )


@dataclass(frozen=True)
class V2Experiment:
    """Where the parts of one retained v2 experiment live."""

    project: Path
    config: Path
    root: Path
    collection_marker: Path


def collection_intent_v2(capacity: int = 2) -> dict:
    """One complete two-member scientific intent, assembled without a producer.

    ``capacity`` is the run-group id space; a bundle-grain indicator needs one id
    per run plus one per bundle.
    """
    from blueearth_cst.experiment.content_identity import (
        automatic_seed_v2,
        collection_id_v2,
        content_sha256,
        generation_seed_material_v2,
    )
    from blueearth_cst.experiment.generation_plan import (
        GENERATOR_SEED_FIELDS,
        generator_seed_projection,
    )

    digest = sha256(b"x").hexdigest()
    sources = [
        {"role": role, "sha256": digest, "size_bytes": 1}
        for role in ("basin_cells", "historical_climate")
    ]
    code = [{"path": "blueearth_cst/experiment/scenario_provider.py", "sha256": digest}]
    month = [0.0] * 12
    range12 = {"min": month, "max": month}
    perturbations = {
        "temp": {"n_levels": 1, "trajectory": "constant", "mean": range12},
        "precip": {
            "n_levels": 1,
            "trajectory": "constant",
            "mean": range12,
            "variance": range12,
        },
        "spell_factors": {"dry": [1.0] * 12, "wet": [1.0] * 12},
    }
    window = {"start": 2046, "end": 2054}
    generator = {
        section: {key: 1 for key in included | excluded}
        for section, (included, excluded) in GENERATOR_SEED_FIELDS.items()
    }
    generator["apply_climate_perturbations"]["diagnostic"] = False
    generator["generate_weather"]["out_dir"] = None
    for section, keys in (
        ("generate_weather", ("plot_dpi", "plot_device")),
        ("write_netcdf", ("file_suffix", "out_dir")),
    ):
        for key in keys:
            generator[section][key] = {"present": False, "value": None}
    generator_settings = generator_seed_projection(generator)
    material = generation_seed_material_v2(
        n_realizations=1,
        simulation_window=window,
        climate_perturbations=perturbations,
        water_year_start="JAN",
        provider_revision=content_sha256(code),
        sources=sources,
        generator_settings=generator_settings,
    )
    seed = automatic_seed_v2(material)
    generator["generate_weather"]["seed"] = seed
    resolution = {
        "seed_request": "auto",
        "resolved_seed": seed,
        "projection": material,
        "projection_sha256": content_sha256(material),
    }
    resolution["seed_resolution_id"] = content_sha256(resolution)
    generation = {
        "seed": {"requested": "auto", "resolved": seed},
        "seed_resolution": resolution,
        "run_group_id_capacity": capacity,
        "weathergen": generator,
        "climate_perturbations": perturbations,
    }
    source_inventory = {
        "sources": [
            {
                "role": role,
                "original_path": role,
                "file": {
                    "schema_version": "artifact-reference/1",
                    "path_base": "project_root",
                    "path": f"data/{role}.dat",
                    "sha256": digest,
                    "size_bytes": 1,
                },
                "metadata": {},
            }
            for role in ("basin_cells", "historical_climate")
        ],
        "interpretation": {
            "revision": "fixture/1",
            "variables": [{"name": "temp", "units": "degC"}],
            "calendar": "standard",
            "time_label": "interval_end",
        },
    }
    documents = {
        "generation_config": generation,
        "source_inventory": source_inventory,
        "provider_code": code,
        "environment": {
            "packages": {"python": "fixture"},
            "locks": {"Manifest.toml": digest},
        },
    }
    projections = {
        "generation_config": {
            "schema_version": "generation-config-identity/2",
            "resolved_seed": seed,
            "generator_settings": material["generator_settings"],
            "climate_perturbations": perturbations,
            "simulation_window": window,
            "n_realizations": 1,
            "water_year_start": "JAN",
        },
        "source_inventory": {
            "schema_version": "generation-sources-identity/2",
            "sources": sources,
            "interpretation": source_inventory["interpretation"],
        },
    }
    intent = {
        "schema_version": "scenario-collection-intent/3",
        "canonicalization_id": "collection-canon/1",
        "collection_id": digest,
        "provider": {"name": "weathergenr", "revision": content_sha256(code)},
        "scenario_type": "stochastic",
        "scenario_spec": {
            "scenario_type": "stochastic",
            "n_realizations": 1,
            "n_design_points": 1,
            "unperturbed_per_realization": 1,
            "expected_run_count": 2,
            "simulation_window": window,
            "pairing": "paired_across_design_points",
            "row_order": "rlz-major/unperturbed-first/st-id-ascending",
        },
        "scenario_semantics_sha256": content_sha256(
            [
                {"evaluate": "true", "type": "stochastic", "rlz": "1", "st_id": ""},
                {"evaluate": "true", "type": "stochastic", "rlz": "1", "st_id": "1"},
            ]
        ),
        "run_count": 2,
        "run_group_id_capacity": capacity,
        "run_group_id_width": len(str(capacity)),
        "documents": documents,
        "document_digests": {
            name: content_sha256(value) for name, value in documents.items()
        },
        "identity_projections": projections,
        "identity_digests": {
            name: content_sha256(projections[name] if name in projections else value)
            for name, value in documents.items()
        },
    }
    intent["collection_id"] = collection_id_v2(intent)
    return intent


def _write_configs(base: Path, project: Path) -> Path:
    """A project config set: the project file plus one file per workflow."""
    (base / "generate_scenarios.yml").write_text(
        yaml.safe_dump({"seed": "auto", "n_realizations": 1}), encoding="utf-8"
    )
    (base / "simulate_system.yml").write_text(
        yaml.safe_dump(
            {
                "experiment_name": EXPERIMENT,
                "operation": "indicators-only",
                "indicators": ["gwr"],
            }
        ),
        encoding="utf-8",
    )
    config = base / "project.yml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "project": {
                    "project_dir": project.as_posix(),
                    "catalog": (base / "unused-catalog.yml").as_posix(),
                },
                "basin": {},
                "climate": {"selected": CLIMATE_SOURCE},
                "model": {},
                "workflows": {
                    "generate_scenarios": {
                        "enabled": True,
                        "config_path": "generate_scenarios.yml",
                    },
                    "simulate_system": {
                        "enabled": True,
                        "config_path": "simulate_system.yml",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return config


def _archive(
    project, config, workflow, projection, owner_kind, owner_id, target, generated=()
):
    """Publish one creator archive exactly as the workflow producers do."""
    from blueearth_cst.shared.config_composition import (
        capture_configuration_sources,
        compose_captured_config,
    )
    from blueearth_cst.shared.provenance import (
        environment_file_hashes,
        toolbox_identity,
    )
    from blueearth_cst.shared.snake_utils import ADVANCED_SETTINGS
    from blueearth_cst.shared.workflow_config_snapshot import (
        publish_archive,
        run_record_document,
    )

    sources = capture_configuration_sources(config, workflow, projection)
    loaded, _ = compose_captured_config(sources[0], sources, workflow, projection)
    record, payloads = run_record_document(
        workflow=workflow,
        invocation_id=INVOCATION_ID,
        owner_kind=owner_kind,
        owner_id=owner_id,
        loaded_config=loaded,
        advanced_settings=ADVANCED_SETTINGS,
        projection=projection,
        toolbox=toolbox_identity(),
        environment=environment_file_hashes(),
        invocation={
            "entry_point": "tests/_v2_experiment.py",
            "command": [workflow],
            "targets": ["all"],
            "working_directory": str(project),
            "overrides": {},
        },
        sources=sources,
        generated_inputs=list(generated),
    )
    publish_archive(project, owner_id[:12], target, record, payloads)


def _publish_collection(
    project: Path, config: Path, capacity: int, *, legacy_storage: bool = False
) -> Path:
    """Write a ready scenario-collection/3 the way WF3 publication does."""
    from blueearth_cst.experiment.content_identity import (
        atomic_record,
        collection_revision_v2,
        identity_segment,
    )
    from blueearth_cst.experiment.generation_publication import PROJECTION
    from blueearth_cst.shared.workflow_config_snapshot import file_reference

    intent = collection_intent_v2(capacity)
    segment = identity_segment(intent["collection_id"], "collection_id")
    record = project / "scenarios/_engine/collections" / segment
    data_relative = (
        f"scenarios/{segment}"
        if legacy_storage
        else f"scenarios/_engine/data/{segment}"
    )
    data = project / data_relative
    record.mkdir(parents=True)
    (data / "series").mkdir(parents=True)
    (data / "weathergenr/output").mkdir(parents=True)
    for role in ("basin_cells", "historical_climate"):
        path = project / "data" / f"{role}.dat"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"x")
    products = []
    for role, name in (
        ("sim_dates", "output/sim_dates.csv"),
        ("resampled_dates", "output/resampled_dates.csv"),
        ("weather_generation_input", "weather_generation_input.yml"),
    ):
        path = data / "weathergenr" / name
        path.write_bytes(role.encode())
        products.append(
            {"role": role, "file": file_reference(path, "project_root", project)}
        )
    _archive(
        project,
        config,
        "generate_scenarios",
        PROJECTION,
        "scenario_collection",
        intent["collection_id"],
        f"{data_relative}/config",
        generated=[products[2]],
    )
    atomic_record(record / "collection_intent.json", intent)
    lookup = data / "scenario_run_lookup.csv"
    lookup.write_text(
        "run_id,evaluate,type,rlz,st_id\nrun_1,true,stochastic,1,\nrun_2,true,stochastic,1,1\n",
        encoding="utf-8",
    )
    perturbation = data / "perturbation_lookup.csv"
    perturbation.write_text(
        "st_id,month,temp_change,precip_change,precip_variance_change\n"
        + "".join(f"1,{month},0,0,0\n" for month in range(1, 13)),
        encoding="utf-8",
    )
    series = []
    for run_id in RUN_IDS:
        path = data / "series" / f"{run_id}.nc"
        path.write_bytes(run_id.encode())
        series.append(
            {
                "run_id": run_id,
                "file": file_reference(path, "project_root", project),
                "descriptor": {
                    "source_calendar": "proleptic_gregorian",
                    "timestep": "daily",
                    "time_label": "interval_end",
                    "start": "2046-01-01",
                    "end": "2054-12-31",
                    "spatial_representation_sha256": sha256(b"spatial").hexdigest(),
                    "variables": [
                        {"name": "temp", "units": "degC", "missing_value": None}
                    ],
                },
            }
        )
    marker = {
        "schema_version": "scenario-collection/3",
        "canonicalization_id": "collection-canon/1",
        "status": "ready",
        "collection_id": intent["collection_id"],
        "collection_revision": "0" * 64,
        "data_root": data_relative,
        "intent": file_reference(
            record / "collection_intent.json", "record_directory", record
        ),
        "archive": file_reference(
            data / "config/run_record.yml", "project_root", project
        ),
        "scenario_run_lookup": file_reference(lookup, "project_root", project),
        "perturbation_lookup": file_reference(perturbation, "project_root", project),
        "series": series,
        "provider_products": products,
        "diagnostics": [],
    }
    marker["collection_revision"] = collection_revision_v2(marker, intent)
    atomic_record(record / "collection.json", marker)
    return record / "collection.json"


def _preparation(project: Path, root: Path, base: Path) -> dict:
    """Bind shared forcing elevation as WF4 does, without a real data catalog."""
    from blueearth_cst.climate_analysis import prepare_climate_data_catalog
    from blueearth_cst.experiment.forcing_descriptor import UnitInterpretation
    from blueearth_cst.experiment.wf4_ancillary_descriptor import (
        resolve_wf4_preparation,
    )

    source = base / "external/era5_orography_2018.nc"
    source.parent.mkdir(parents=True)
    xr.Dataset(
        {"z": (("latitude", "longitude"), np.ones((2, 2)))},
        coords={"latitude": [1.0, 2.0], "longitude": [3.0, 4.0]},
    ).to_netcdf(source)
    interpretation = UnitInterpretation("fixture/1", "evidence", (("temp", "degC"),))
    planned = prepare_climate_data_catalog.plan_preparation_payloads(
        {
            "data_type": "RasterDataset",
            "uri": "unused.nc",
            "driver": "raster_xarray",
            "metadata": {"crs": 4326},
        },
        CLIMATE_SOURCE,
        {
            "data_type": "RasterDataset",
            "uri": str(source),
            "driver": "raster_xarray",
            "data_adapter": {"rename": {"z": "elevtn"}},
            "metadata": {"crs": 4326},
        },
        source,
        interpretation,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            prepare_climate_data_catalog,
            "resolve_preparation_payloads",
            lambda *_args: planned,
        )
        return resolve_wf4_preparation(
            project,
            root,
            climate_source=CLIMATE_SOURCE,
            catalogs=[base / "unused-catalog.yml"],
            unit_interpretation=interpretation,
        )


def _freeze_simulation(project, config, root, marker_path, base, responses):
    """Freeze the simulation intent and its creator archive (rule 4.03)."""
    from blueearth_cst.experiment.content_identity import content_sha256
    from blueearth_cst.experiment.response_inventory import make_response_request
    from blueearth_cst.experiment.scenario_collection_v2 import read_collection_v2
    from blueearth_cst.experiment.simulation_record import (
        capture_simulation_sources_v2,
        freeze_simulation_v2,
        simulation_intent_v2,
    )
    from blueearth_cst.experiment.wf4_ancillary_descriptor import (
        preparation_identity_v2,
    )
    from blueearth_cst.experiment.write_model_reference import write_model_reference
    from blueearth_cst.shared.workflow_config_snapshot import file_reference

    model = project / "models/hydrology/wflow"
    model.mkdir(parents=True)
    (model / "wflow_sbm.toml").write_bytes(responses.toml.encode("utf-8"))
    reference = write_model_reference(
        model, project, root / "_engine/model_reference.yml"
    )
    collection = read_collection_v2(marker_path)
    run_ids = list(responses.csv)
    request = make_response_request(run_ids, responses.variables)
    preparation = _preparation(project, root, base)
    environment = {"packages": {"julia:Wflow": "1.0.2:fixture"}, "locks": {}}
    documents = {
        "model_reference": reference,
        "settings": {"simulation_window": {"start": 2046, "end": 2054}},
        "environment": environment,
        "simulator_adapter_code": [{"path": "fixture.py", "sha256": "a" * 64}],
        "response_request": request,
        "preparation": preparation,
    }
    if run_ids != list(RUN_IDS):
        documents["settings"]["include_unperturbed"] = False
    intent = simulation_intent_v2(
        root.name,
        {
            "resolution_mode": "explicit-manifest",
            "manifest": file_reference(marker_path, "project_root", project),
            "collection_id": collection["collection_id"],
            "collection_revision": collection["collection_revision"],
            "run_ids": run_ids,
        },
        {
            "name": "wflow",
            "revision": content_sha256(
                {"julia:Wflow": environment["packages"]["julia:Wflow"]}
            ),
        },
        documents,
        preparation_identity_v2(
            preparation, project_root=project, experiment_root=root
        ),
    )
    capture_simulation_sources_v2(config, project, INVOCATION_ID)
    return freeze_simulation_v2(
        root, intent, invocation_id=INVOCATION_ID, command=["simulate_system"]
    )


def native_runs(root: Path, run_ids=RUN_IDS) -> dict:
    """The retained native artifacts of every run, where rule 4.04/4.05 put them."""
    from blueearth_cst.experiment.wflow_response_reader import NativeRunArtifacts

    wflow = Path(root) / "hydrology/wflow"
    return {
        run: NativeRunArtifacts(
            wflow / f"output/{run}.csv",
            wflow / f"run_settings/{run}.toml",
            wflow / f"run_settings/{run}.temporal.json",
        )
        for run in run_ids
    }


def _write_native_runs(root: Path, responses: Responses) -> dict:
    """Retain both native runs as Wflow and the forcing preparation leave them."""
    from blueearth_cst.experiment.content_identity import canonical_json_bytes

    native = native_runs(root, responses.csv)
    for run, artifacts in native.items():
        artifacts.csv_path.parent.mkdir(parents=True, exist_ok=True)
        artifacts.toml_path.parent.mkdir(parents=True, exist_ok=True)
        # Bytes, not text: the inventory binds exact bytes on every platform.
        artifacts.csv_path.write_bytes(responses.csv[run])
        artifacts.toml_path.write_bytes(responses.toml.encode("utf-8"))
        artifacts.temporal_path.write_bytes(canonical_json_bytes(responses.temporal))
    return native


def _publish_responses(root: Path, responses: Responses) -> dict:
    """Publish inventory then readiness over the retained runs (rule 4.06)."""
    from blueearth_cst.experiment.response_inventory import (
        publish_response_inventory_v2,
    )
    from blueearth_cst.experiment.simulation_record import publish_simulation_v2

    inventory = publish_response_inventory_v2(
        root, native_runs(root, responses.csv), responses.temporal
    )
    return publish_simulation_v2(root, inventory)


def build_v2_experiment(
    base: Path,
    responses: Responses | None = None,
    *,
    capacity: int = 2,
    publish: bool = True,
    legacy_storage: bool = False,
    include_unperturbed: bool = True,
) -> V2Experiment:
    """Build one retained v2 experiment under ``base``.

    ``publish=False`` stops after the native runs are retained, before the
    response inventory and the readiness marker -- the state rule 4.06 sees.
    """
    responses = responses or default_responses()
    base = Path(base).resolve()
    project = base / "project"
    root = project / "experiments" / EXPERIMENT
    root.mkdir(parents=True)
    config = _write_configs(base, project)
    marker = _publish_collection(
        project, config, capacity, legacy_storage=legacy_storage
    )
    if not include_unperturbed:
        from blueearth_cst.experiment.scenario_collection_v2 import read_collection_v2
        from blueearth_cst.experiment.simulation_runs import selected_run_ids

        runs = selected_run_ids(read_collection_v2(marker), marker, include=False)
        responses = replace(responses, csv={run: responses.csv[run] for run in runs})
        workflow = config.parent / "simulate_system.yml"
        workflow.write_text(
            workflow.read_text(encoding="utf-8") + "\ninclude_unperturbed: false\n",
            encoding="utf-8",
        )
    _freeze_simulation(project, config, root, marker, base, responses)
    _write_native_runs(root, responses)
    if publish:
        _publish_responses(root, responses)
    return V2Experiment(project, config, root, marker)


def copy_v2_experiment(source: V2Experiment, base: Path) -> V2Experiment:
    """Copy a built experiment, config set and all, to a new base directory.

    The project config names its project directory absolutely, as a real one
    does, so the copy's project file is rewritten to point at the copy.
    """
    base = Path(base).resolve()
    origin = source.config.parent
    for path in origin.iterdir():
        target = base / path.name
        if path.is_dir():
            shutil.copytree(path, target)
        else:
            shutil.copyfile(path, target)
    project = base / source.project.relative_to(origin)
    config = base / source.config.name
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["project"]["project_dir"] = project.as_posix()
    document["project"]["catalog"] = (base / "unused-catalog.yml").as_posix()
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    return V2Experiment(
        project,
        config,
        project / source.root.relative_to(source.project),
        project / source.collection_marker.relative_to(source.project),
    )
