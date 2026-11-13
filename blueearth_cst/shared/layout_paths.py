"""Declared output layout of a CST project folder, with a kind for every path.

One table of what the toolbox writes under ``project_dir`` today, per output
unit, so the tree inventory (`dev/scripts/semantic_tree_diff.py`) and the
placement check derive from one source instead of restating it (t017 step 1).
Declarations describe the CURRENT layout; t017's later steps change them in the
same commit as the writers.

This module imports no repository code, on purpose: identity-bearing WF3/WF4
code will import it, and their identities hash the transitive import closure.
Grammars that mirror a constant elsewhere (digest length, experiment and
indicator-set names, forcing IDs) are pinned against their owners by
``tests/test_layout_paths.py``.

Template syntax. A declaration path is literal text plus ``{...}`` fields:

- ``{name}`` -- a grammar from :data:`GRAMMARS`, or a value the caller binds
  (:data:`BOUND`: ``experiment``, ``clim_project``, ``model``);
- ``{a|b|c}`` -- one of the listed literals;
- ``{?text}`` -- optional literal text.

A trailing ``/`` declares an open directory: any path beneath it belongs to
the declaration. Use it only where the contents are genuinely open.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

#: Readable kinds: what a person reads (data, tables, figures, settings, views).
READABLE_KINDS = ("product", "settings", "view")
#: Machine kinds, whose place is a unit's ``_engine/`` folder (rule R10).
ENGINE_KINDS = ("record", "marker", "pin", "figure_data", "log")
#: Files an engine (HydroMT, Wflow, Snakemake) writes where the toolbox does
#: not control the location (rule R3a); exempt from placement rules.
ENGINE_OWNED = "engine_owned"
KINDS = (*READABLE_KINDS, *ENGINE_KINDS, ENGINE_OWNED)

UNITS = ("project", "spatial", "wf0", "wf1", "wf2", "wf3", "wf4")

#: Layout version this toolbox writes (rule R18). A project folder without a
#: stamp is layout 1. Bumped once per landed t017 migration step.
LAYOUT_VERSION = 4

#: Mirrors ``blueearth_cst.shared.provenance.SHORT_DIGEST_CHARS``.
SHORT_DIGEST_CHARS = 12

_OUTPUT_VARIABLES = "(?:q|p|pet|aet|gwr|qof|swe|temp)"
_LOG_SCOPE = r"(?:all-workflows|workflows|[a-z]+(?:-[a-z]+)+|wf[0-4](?:-wf[0-4])*)"

#: Field grammars, as regular expressions matching within one path.
GRAMMARS: dict[str, str] = {
    "hex12": f"[0-9a-f]{{{SHORT_DIGEST_CHARS}}}",
    "hex32": "[0-9a-f]{32}",
    "hex64": "[0-9a-f]{64}",
    "segment": "[^/]+",
    "rest": ".+",
    "number": "[0-9]+",
    "run": "run_[0-9]+",
    # `execution_outputs._directory_name`: `<YYYY-MM-DD_HH-MM-SSZ>__<scope>`,
    # an optional `__dry-run` and an `__<id12>` collision suffix; a child
    # without a recorded directory falls back to `<stamp>-<invocation id>`.
    "execution": (
        r"(?:[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}"
        rf"|[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}_[0-9]{{2}}-[0-9]{{2}}-[0-9]{{2}}Z__{_LOG_SCOPE}"
        r"(?:__dry-run)?(?:__[0-9a-f]{12})?)"
    ),
    # `snake_utils.validate_experiment_name`.
    "experiment_name": "[a-z0-9_]+",
    # `named_forcings.FORCING_ID`.
    "recipe": "[a-z][a-z0-9_]{0,31}",
    # `indicator_sets.SET_ID_PATTERN` within `SET_ID_MAX_LENGTH`.
    "set_id": "[a-z][a-z0-9_-]{0,63}",
    "slug": "[a-z][a-z0-9_]*",
    "store": "(?!_engine/)[^/]+",
    # `delta_scenarios` set folder: never `_engine` and never 8+ hex characters,
    # so it cannot match the engine directory or a retained identity segment.
    "set_name": "(?![0-9a-f]{8,}/)[A-Za-z0-9][A-Za-z0-9._-]*",
    "netcdf_name": "[a-zA-Z0-9][a-zA-Z0-9_.-]*",
    "output_variable": _OUTPUT_VARIABLES,
    "indicator_table": (
        "(?:group_(?:members|metadata)|[a-z][a-z0-9_]*_(?:run|group)_indicators)"
    ),
    "captured_role": (
        r"(?:project|simulate_system|generate_scenarios|generator_template|build_model|"
        r"build_template|waterbodies_template|region|output_locations|catalog_(?:0|[1-9][0-9]*))"
    ),
}

#: Fields the caller binds to literal values when generating rules.
BOUND = ("experiment", "clim_project", "model")

_FIELD = re.compile(r"\{([^{}]*)\}")


@dataclass(frozen=True)
class Declaration:
    """One project-relative path shape the toolbox writes, and what it is."""

    unit: str
    path: str
    kind: str

    def __post_init__(self) -> None:
        if self.unit not in UNITS:
            raise ValueError(f"unknown unit {self.unit!r} in {self.path!r}")
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r} in {self.path!r}")
        for field in _FIELD.findall(self.path):
            if "|" in field or field.startswith("?"):
                continue
            if field not in GRAMMARS and field not in BOUND:
                raise ValueError(f"unknown field {{{field}}} in {self.path!r}")

    @property
    def is_open(self) -> bool:
        return self.path.endswith("/")

    @property
    def in_engine(self) -> bool:
        """Whether the declared path lies inside an ``_engine/`` folder."""
        return "_engine/" in self.path

    def pattern(self, values: Mapping[str, str]) -> str:
        """Regular expression matching every path of this declaration."""
        out: list[str] = []
        position = 0
        for match in _FIELD.finditer(self.path):
            out.append(re.escape(self.path[position : match.start()]))
            out.append(_field_regex(match[1], values))
            position = match.end()
        tail = self.path[position:]
        out.append(re.escape(tail))
        if self.is_open:
            out.append(".+")
        return "".join(out)


def _field_regex(field: str, values: Mapping[str, str]) -> str:
    if field.startswith("?"):
        return f"(?:{re.escape(field[1:])})?"
    if "|" in field:
        return "(?:" + "|".join(re.escape(part) for part in field.split("|")) + ")"
    if field in BOUND:
        if field not in values:
            raise KeyError(f"layout field {{{field}}} needs a value")
        return re.escape(values[field])
    return GRAMMARS[field]


def _declare(unit: str, kind: str, *paths: str) -> list[Declaration]:
    return [Declaration(unit, path, kind) for path in paths]


def declarations() -> tuple[Declaration, ...]:
    """Every path shape the toolbox writes today, grouped by output unit.

    The model folder is the bound field ``{model}`` (rule R16), so a further
    model is a binding, not a new declaration. Experiment-scoped declarations
    contain ``{experiment}`` and are bound once per experiment by :func:`bind`.
    """
    d: list[Declaration] = []

    # -- project: logs, benchmarks, config/ ----------------------------------
    d += _declare(
        "project",
        "log",
        "logs/{execution}/{summary.md|console.log|wf0.log|wf1.log|wf2.log|wf3.log|wf4.log"
        "|wf0-benchmarks.md|wf1-benchmarks.md|wf2-benchmarks.md|wf3-benchmarks.md|wf4-benchmarks.md"
        "|wf0-benchmarks.tsv|wf1-benchmarks.tsv|wf2-benchmarks.tsv|wf3-benchmarks.tsv"
        "|wf4-benchmarks.tsv|wf4-indicators.log}",
        "logs/{execution}/_parts/",
        "logs/{wf0|wf1|wf2}_{segment}.log",
        "logs/wf3_generate_scenarios_{hex12}.log",
        "logs/wf4_simulate_system_{experiment_name}.log",
        "logs/.{wf0|wf1|wf2}_{segment}.log.tally",
        "logs/.wf3_generate_scenarios_{hex12}.log.tally",
        "logs/.wf4_simulate_system_{experiment_name}.log.tally",
        "logs/_parts/",
        "logs/dag/",
        "logs/benchmarks/{wf0|wf1|wf2}_benchmarks.md",
        "logs/benchmarks/_parts/",
        "benchmarks/wf3_benchmarks_{hex12}.md",
        "benchmarks/wf4_benchmarks_{experiment_name}.md",
        "benchmarks/_parts/",
    )
    archived = "{analyze_climate|build_model|analyze_projections}"
    d += _declare(
        "project",
        "settings",
        f"config/{archived}/sources/",
        f"config/{archived}/{{?catalogs/}}{{segment}}",
        "config/catalogs/",
        "config/templates/",
        "config/basin_data/",
    )
    d += _declare("project", "view", "config/README.md")
    d += _declare(
        "project",
        "record",
        "config/_engine/execution_configs/",
        "config/_engine/archive_transactions/",
        "config/_engine/snapshots/",
        "config/_engine/invocations/",
        "config/_engine/journal.jsonl",
        # Layout version stamp (rule R18; `layout_stamp`).
        "config/_engine/layout.json",
        "experiments/_engine/allocation.json",
        "experiments/_engine/transactions/{hex32}/transaction.json",
        "experiments/_engine/transactions/{hex32}/experiment/experiment.yml",
        "experiments/_engine/transactions/{hex32}/experiment/_engine/reservation.json",
    )
    d += _declare(
        "project",
        "pin",
        "experiments/_engine/transactions/{hex32}/sources/{captured_role}.bin",
    )

    # -- spatial foundation ---------------------------------------------------
    d += _declare(
        "spatial",
        "product",
        "data/spatial/spatial_maps.nc",
        "data/spatial/location_registry.csv",
        "data/spatial/hydrography.nc",
        "data/spatial/geoms/",
        "data/spatial/plots/",
    )
    d += _declare(
        "spatial",
        "record",
        "data/spatial/_engine/{spatial_catalog|spatial_report}.yml",
        "data/spatial/_engine/plots/",
    )

    # -- WF0: historical climate stores and their comparison ------------------
    d += _declare("wf0", "record", "data/climate/historical/{store}/_engine/{rest}")
    d += _declare("wf0", "product", "data/climate/historical/{store}/{rest}")
    d += _declare(
        "wf0",
        "record",
        "data/climate/historical/_engine/map_reference.json",
    )
    d += _declare(
        "wf0", "product", "data/climate/ancillary/{slug}/{hex12}/{netcdf_name}.nc"
    )

    # -- WF2: projections -----------------------------------------------------
    cmip = "data/climate/projections/{clim_project}"
    d += _declare(
        "wf2",
        "product",
        *(
            f"{cmip}/{tier}/"
            for tier in ("raw", "scalar", "summary", "analysis", "plots")
        ),
        f"{cmip}/report.md",
    )
    d += _declare(
        "wf2",
        "record",
        f"{cmip}/_engine/{{stage_b_settings.json|analysis_record.json|provenance.json}}",
        f"{cmip}/_engine/plots/{{figure_captions.json|figure_titles.json}}",
        f"{cmip}/_engine/execution/production.json",
        f"{cmip}/_engine/imported/{{hex12}}/{{manifest|import_record}}.json",
    )
    d += _declare(
        "wf2",
        "pin",
        f"{cmip}/_engine/imported/{{hex12}}/dependencies/region.geojson",
        f"{cmip}/_engine/imported/{{hex12}}/configuration/{{hex12}}/{{rest}}",
    )

    # -- WF1: model, recipe-keyed historical runs, retained runs, comparison --
    m = "models/hydrology/{model}"
    d += _declare(
        "wf1",
        ENGINE_OWNED,
        f"{m}/staticmaps.nc",
        f"{m}/wflow_sbm.toml",
        f"{m}/wflow_sbm_{{recipe}}.toml",
        f"{m}/hydromt.log",
        f"{m}/hydromt_data.yml",
        f"{m}/staticgeoms/",
    )
    d += _declare(
        "wf1",
        "marker",
        f"{m}/_engine/markers/.model_built",
        f"{m}/.outputs_configured",
        f"{m}/_engine/markers/.model_final",
    )
    d += _declare(
        "wf1",
        "record",
        f"{m}/wflow_sbm.toml.grid.toml",
        f"{m}/wflow_sbm_{{recipe}}.toml.grid.toml",
        f"{m}/_engine/hydromt_build_config.yml",
        f"{m}/_engine/hydromt_update_waterbodies.yml",
        f"{m}/_engine/run_metadata.json",
        f"{m}/_engine/{{reservoirs_lakes_glaciers.txt|outlet_index.csv}}",
    )
    # One folder per forcing recipe, `default` for a single recipe (R16,
    # layout 4); the recipe TOML stays at the model root (open point 2).
    h = f"{m}/historical/{{recipe}}"
    d += _declare(
        "wf1",
        "record",
        f"{h}/_engine/{{build_historical_forcing|climate_store_catalog}}.yml",
        f"{h}/_engine/historical_forcing.json",
        f"{h}/forcing/_engine/plots/",
    )
    d += _declare(
        "wf1",
        "product",
        f"{h}/forcing/inmaps_historical.nc",
        f"{h}/forcing/plots/",
        f"{h}/run/",
        f"{h}/evaluation/",
    )
    # Recipe runs compared within their model (R5, R16, D18).
    c = f"{m}/historical_comparison"
    d += _declare(
        "wf1",
        "product",
        f"{c}/tables/{{segment}}.csv",
        f"{c}/tables/supporting/support.csv",
        f"{c}/plots/",
    )
    d += _declare(
        "wf1", "record", f"{c}/_engine/comparison.json", f"{c}/_engine/plots/"
    )
    d += _declare("wf1", ENGINE_OWNED, f"{c}/.snakemake_timestamp")

    # -- WF3: scenario collections --------------------------------------------
    # Retained collection data under `_engine/data/` and named sets in
    # `scenarios/<name>/` (t016), indexed by `scenarios/collections.csv`.
    for root in ("scenarios/_engine/data/{hex12}", "scenarios/{set_name}"):
        d += _declare(
            "wf3",
            "settings",
            f"{root}/config/sources/{{segment}}",
            f"{root}/config/{{?catalogs/}}{{segment}}",
            f"{root}/weathergenr/weather_generation_input.yml",
        )
        d += _declare(
            "wf3",
            "product",
            f"{root}/{{scenario_run_lookup|perturbation_lookup}}.csv",
            f"{root}/weathergenr/output/{{sim_dates|resampled_dates}}.csv",
            f"{root}/series/{{run}}.nc",
            f"{root}/weathergenr/output/{{run}}.nc",
            f"{root}/weathergenr/evaluation/plots/{{segment}}.{{png|pdf}}",
        )
    d += _declare(
        "wf3",
        "product",
        "scenarios/collections.csv",
        "scenarios/{set_name}/factor_lookup.csv",
        "scenarios/{set_name}/delta/factors/{run}.nc",
    )
    d += _declare(
        "wf3",
        "record",
        "scenarios/{set_name}/_engine/{collection.json|collection_intent.json}",
    )
    d += _declare("wf3", "pin", "scenarios/{set_name}/_engine/producer.py")
    d += _declare(
        "wf3",
        "pin",
        "scenarios/_engine/data/{hex12}/inputs/{reference|future}_{number}.nc",
        "scenarios/_engine/data/{hex12}/inputs/historical.nc",
        "scenarios/_engine/data/{hex12}/producer.py",
        "scenarios/_engine/generation_inputs/{hex12}/{segment}.{yml|yaml}",
        "scenarios/_engine/generation_inputs/{hex12}/{basin_cells.csv|extract_historical.nc}",
    )
    d += _declare(
        "wf3",
        "record",
        "scenarios/_engine/current.json",
        "scenarios/_engine/collections/{hex12}/{collection.json|collection_intent.json}",
        "scenarios/_engine/requests/{hex12}/request.json",
        "scenarios/_engine/requests/{hex12}/{hex64}.json",
        "scenarios/_engine/requests/{hex12}/initializations/{hex32}.json",
        "scenarios/_engine/requests/{hex12}/generation/config/stress_test_lookup.csv",
    )

    # -- WF4: experiments ------------------------------------------------------
    e = "experiments/{experiment}"
    d += _declare(
        "wf4",
        "settings",
        f"{e}/experiment.yml",
        f"{e}/config/sources/",
        f"{e}/config/{{?catalogs/}}{{segment}}",
    )
    d += _declare("wf4", "marker", f"{e}/.model_reference_ok")
    d += _declare(
        "wf4",
        "record",
        f"{e}/_engine/{{reservation.json|simulation-freeze.json|model_reference.yml"
        "|simulation.json|simulation_intent.json|response_inventory.json}",
        f"{e}/_engine/indicator_requests/{{hex12}}.json",
        f"{e}/_engine/indicator_sets/{{hex12}}/{{indicators.json|indicator_environment.json"
        "|indicator_diagnostics.json|return_level_benchmark.json|return_level_evidence.json"
        "|coverage_evidence.json}",
        f"{e}/_engine/indicator_sets/names/{{set_id}}/current.json",
        f"{e}/_engine/indicator_sets/names/{{set_id}}/definitions/{{hex64}}.yml",
        f"{e}/_engine/indicator_sets/names/{{set_id}}/publications/{{hex64}}.json",
        f"{e}/_engine/batch_staging/{{run}}.{{csv|expect|ok}}",
    )
    d += _declare(
        "wf4",
        "settings",
        f"{e}/_engine/freeze-payloads/run_record.yml",
        f"{e}/_engine/freeze-payloads/sources/{{?catalogs/}}{{segment}}",
        f"{e}/_engine/freeze-payloads/{{?catalogs/}}{{segment}}",
    )
    indicator_tables = "{indicator_table}.csv"
    d += _declare(
        "wf4",
        "product",
        # Copies inside `_engine/` that R1 removes.
        f"{e}/_engine/indicator_sets/{{hex12}}/data/{indicator_tables}",
        f"{e}/results/indicator_sets/{{hex12}}/{indicator_tables}",
        f"{e}/results/indicator_sets/{{set_id}}/{indicator_tables}",
        f"{e}/results/indicator_sets/{{set_id}}/indicators.csv",
    )
    native = f"{e}/hydrology/wflow"
    d += _declare(
        "wf4",
        "record",
        f"{native}/run_settings/{{run}}.{{toml|toml.grid.toml|yml|temporal.json}}",
        f"{native}/run_settings/forcing_elevation_catalog.yml",
        f"{native}/output/{{run}}/_engine/response_tables.json",
    )
    d += _declare(
        "wf4",
        "product",
        f"{native}/forcing/inmaps_{{run}}.nc",
        f"{native}/output/{{run}}.{{csv|nc}}",
        f"{native}/output/{{run}}/output_{{output_variable}}.csv",
        f"{native}/output/outstates_{{run}}.nc",
    )
    d += _declare("wf4", "log", f"{native}/output/_log/{{run}}.log")
    return tuple(d)


def bind(
    declared: Iterable[Declaration],
    *,
    clim_project: str,
    experiments: Iterable[str],
    model_name: str = "wflow",
) -> list[tuple[Declaration, re.Pattern[str]]]:
    """Compile each declaration, once per experiment for experiment-scoped ones."""
    base = {"clim_project": clim_project, "model": model_name}
    names = sorted(set(experiments))
    out: list[tuple[Declaration, re.Pattern[str]]] = []
    for item in declared:
        if "{experiment}" in item.path:
            for name in names:
                values = {**base, "experiment": name}
                out.append((item, re.compile(item.pattern(values))))
        else:
            out.append((item, re.compile(item.pattern(base))))
    return out


def misplaced(declared: Iterable[Declaration]) -> list[Declaration]:
    """Declarations that break rule R10's placement by kind.

    A machine kind outside ``_engine/`` (top-level ``logs/`` excepted, rule
    R13), or a readable kind inside it. Engine-owned paths are exempt (R3a).
    """
    out = []
    for item in declared:
        if item.kind == ENGINE_OWNED:
            continue
        if item.kind in ENGINE_KINDS:
            allowed = item.in_engine or (
                item.kind == "log" and item.path.startswith("logs/")
            )
        else:
            allowed = not item.in_engine
        if not allowed:
            out.append(item)
    return out
