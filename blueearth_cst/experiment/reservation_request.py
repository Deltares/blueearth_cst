"""Early scientific request projection, independent of prepared source contents."""

import stat
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path

from blueearth_cst.experiment.catalog_identity import catalog_input
from blueearth_cst.experiment.content_identity import content_sha256
from blueearth_cst.shared.config_composition import (
    capture_configuration_sources,
    compose_captured_config,
)
from blueearth_cst.shared.provenance import file_sha256
from blueearth_cst.shared.yaml_documents import parse_yaml_document


def _ordinary_file(locator: str | Path) -> Path:
    """Reject redirected captured inputs before resolving their original locator."""
    path = Path(locator).absolute()
    for part in (path, *path.parents):
        if part.exists() or part.is_symlink():
            attributes = getattr(part.lstat(), "st_file_attributes", 0)
            if part.is_symlink() or attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                raise ValueError(f"captured input is redirected: {part}")
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"captured input is not an ordinary file: {path}")
    return path


def _json_value(value):
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError(f"unsupported semantic input value: {type(value).__name__}")


def _catalog_value(path):
    """Capture source semantics and root context without loading HydroMT."""
    original = _ordinary_file(path)
    raw = parse_yaml_document(original.read_bytes())
    return _json_value(catalog_input(raw, original))


def _local_inputs(basin):
    inputs = []
    region = basin.get("region")
    for role, locator in (
        ("region", region),
        ("output_locations", basin.get("output_locations")),
    ):
        if not isinstance(locator, str):
            continue
        path = Path(locator)
        if role == "region" and not path.is_file():
            continue  # Upstream region expressions and mappings are not files.
        original = _ordinary_file(path)
        inputs.append(
            {"role": role, "path": original.as_posix(), "sha256": file_sha256(original)}
        )
    return inputs


def generation_request(config, repository, *, captures=None):
    """Reuse the effective generator builder and its closed scientific projection."""
    from blueearth_cst.experiment.generation_plan import (
        generation_configuration,
        generator_seed_projection,
    )
    from blueearth_cst.experiment.prepare_weathergen_config import (
        build_weathergen_config,
    )
    from blueearth_cst.shared.wf3_science import (
        DEFAULT_BASIN_INDEX,
        DEFAULT_HYDROGRAPHY,
        DEFAULT_SEED,
        validate_spell_factor,
        window_year_pair,
    )

    settings = generation_configuration(config, repository, stage=False)
    if captures is not None:
        template = _ordinary_file(settings["template"])
        captures.append(("generator_template", template, template.read_bytes()))
    cfg = settings["config"]
    perturbations = deepcopy(cfg["climate_perturbations"])
    spells = perturbations.get("spell_factors") or {}
    dry = validate_spell_factor(spells.get("dry"), "spell_factors.dry")
    wet = validate_spell_factor(spells.get("wet"), "spell_factors.wet")
    perturbations["spell_factors"] = {"dry": dry, "wet": wet}
    generator = build_weathergen_config(
        settings["n_realizations"],
        perturbations,
        "",
        "run_",
        settings["template"],
        settings["end"],
        0,
        settings["request"]["water_year_start"],
        dry,
        wet,
        settings["historical_end"],
    )
    basin = config["basin"]
    climate = config["climate"]
    sources = basin.get("sources") or {}
    start, end = window_year_pair(climate["window"], "climate.window")
    return {
        "schema_version": "reservation-generation-request/1",
        "n_realizations": settings["n_realizations"],
        "simulation_window": {"start": settings["start"], "end": settings["end"]},
        "climate_perturbations": perturbations,
        "water_year_start": settings["request"]["water_year_start"],
        "requested_seed": cfg.get("seed", DEFAULT_SEED),
        "generator_settings": generator_seed_projection(generator),
        "run_group_id_capacity": settings["capacity"],
        "source_request": {
            "region": basin["region"],
            "climate_source": climate["selected"],
            "historical_window": {"start": start, "end": end},
            "hydrography": sources.get("hydrography", DEFAULT_HYDROGRAPHY),
            "basin_index": sources.get("basin_index", DEFAULT_BASIN_INDEX),
            "local_inputs": _local_inputs(basin),
        },
        "provider_code": settings["code"],
    }


def capture_request(
    config_path, repository, *, execute_generation=False, execute_model=False
):
    """Return approved Request and exact (role, path, bytes) captures before producers."""
    from blueearth_cst.experiment.content_identity import confined_path
    from blueearth_cst.shared.snake_utils import (
        DEFAULT_WFLOW_OUTVARS,
        resolve_simulation_window,
    )
    from blueearth_cst.shared.wf3_science import window_year_pair
    from blueearth_cst.shared.workflow_config_snapshot import file_reference

    path = _ordinary_file(config_path)
    raw = parse_yaml_document(path.read_bytes())
    simulation_stanza = raw["workflows"]["simulate_system"]
    simulation = parse_yaml_document(
        (path.parent / simulation_stanza["config_path"]).read_bytes()
    )
    need_generation = execute_generation or "scenario_collection" not in simulation
    for name in ("simulate_system", "generate_scenarios", "build_model"):
        if name == "generate_scenarios" and not need_generation:
            continue
        if name == "build_model" and not execute_model:
            continue
        _ordinary_file(path.parent / raw["workflows"][name]["config_path"])
    sections = ["project", "basin", "climate", "model", "workflows.simulate_system"]
    if need_generation:
        sections.append("workflows.generate_scenarios")
    if execute_model:
        sections.append("workflows.build_model")
    buffers = capture_configuration_sources(path, "simulate_system", sections)
    config, _ = compose_captured_config(
        buffers[0], buffers, "simulate_system", sections
    )
    root = Path(config["project"]["project_dir"]).resolve()
    captures = []
    for source in buffers:
        role = (
            "project" if source.id == "project" else source.id.removeprefix("workflow_")
        )
        captures.append((role, source.original_path, source.data))
    catalogs = config["project"]["catalog"]
    catalogs = [catalogs] if isinstance(catalogs, str) else list(catalogs)
    semantic_catalogs = [_catalog_value(item) for item in catalogs]
    for index, locator in enumerate(catalogs):
        original = _ordinary_file(locator)
        captures.append((f"catalog_{index}", original, original.read_bytes()))
    basin = config.get("basin") or {}
    local = _local_inputs(basin) if need_generation or execute_model else []
    for item in local:
        original = Path(item["path"])
        captures.append((item["role"], original, original.read_bytes()))
    generated = (
        generation_request(config, repository, captures=captures)
        if need_generation
        else None
    )
    model_request = None
    if execute_model:
        model_settings = config["workflows"]["build_model"]
        allowed = {"enabled", "engine", "observations", "simulation_window"}
        unknown = set(model_settings) - allowed
        if unknown:
            raise ValueError(f"unclassified build_model fields: {sorted(unknown)}")
        engine = model_settings.get("engine") or {}
        if set(engine) - {"build_config", "waterbodies_config"}:
            raise ValueError("unclassified build_model.engine fields")
        documents = {}
        for key, role, default in (
            ("build_config", "build_template", "config/defaults/wflow_build_model.yml"),
            (
                "waterbodies_config",
                "waterbodies_template",
                "config/defaults/wflow_update_waterbodies.yml",
            ),
        ):
            original = _ordinary_file(engine.get(key, default))
            data = original.read_bytes()
            documents[key] = parse_yaml_document(data)
            captures.append((role, original, data))
        start, end = window_year_pair(
            resolve_simulation_window(config["climate"], model_settings),
            "build_model.simulation_window",
        )
        model_request = {
            "basin": dict(basin),
            "climate": {
                "selected": config["climate"]["selected"],
                "window": config["climate"]["window"],
            },
            "engine": documents,
            "simulation_window": {"start": start, "end": end},
            "local_inputs": local,
        }
    workflow = config["workflows"]["simulate_system"]
    unknown = set(workflow) - {
        "enabled",
        "experiment_name",
        "operation",
        "scenario_collection",
        "indicators",
        "compute",
        "include_unperturbed",
        # Analysis intent, never simulation identity.
        "operations",
        "indicator_sets",
        "indicator_set_definitions",
        "indicator_selection",
        "indicator_evaluation",
    }
    if unknown:
        raise ValueError(f"unclassified simulate_system fields: {sorted(unknown)}")
    compute = workflow.get("compute") or {}
    unknown_compute = set(compute) - {
        "julia_threads",
        "max_parallel_batches",
        "batch_size_max",
        "batch_size",
        "disk_headroom_gb",
    }
    if unknown_compute:
        raise ValueError(
            f"unclassified simulate_system.compute fields: {sorted(unknown_compute)}"
        )
    manifest = None
    mode = "project-generation"
    if "scenario_collection" in workflow:
        from blueearth_cst.experiment.collection_resolution import (
            resolve_explicit_collection_v2,
        )

        selected, _ = resolve_explicit_collection_v2(workflow["scenario_collection"])
        manifest_path = Path(selected["manifest_path"]).resolve(strict=True)
        if not manifest_path.is_relative_to(root):
            raise ValueError("selected collection must lie in the project root")
        manifest = file_reference(manifest_path, "project_root", root)
        mode = "explicit-manifest"
    request = {
        "schema_version": "experiment-request/2",
        "generation": generated,
        "selection": {"mode": mode, "manifest": manifest},
        "input_catalogs": semantic_catalogs,
        "simulation": {
            "climate_source": config["climate"]["selected"],
            "model_root": confined_path(root, "models/hydrology/wflow")
            .relative_to(root)
            .as_posix(),
            "outvars": list(
                config.get("model", {}).get("outvars", DEFAULT_WFLOW_OUTVARS)
            ),
            "model_request": model_request,
        },
    }
    from blueearth_cst.experiment.simulation_runs import include_unperturbed

    # Omission and true preserve the existing reservation identity.
    if not include_unperturbed(workflow):
        request["simulation"]["include_unperturbed"] = False
    request = _json_value(request)
    content_sha256(request)  # Reject non-finite/unsupported values before allocation.
    for role, original, data in captures:
        if original.read_bytes() != data:
            raise ValueError(f"{role} source changed during request capture")
    return request, captures
