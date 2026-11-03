"""Shared output paths must keep the same producer across workflow invocations."""

import asyncio
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueearth_cst.shared.config_composition import load_composed_config
from tests.conftest import write_config
from tests.test_climate_store_contract import _COMPARED, _parse_workflow

CONFIG_FN = Path(__file__).with_name("project_config_fixture.yml")
pytestmark = pytest.mark.workflow_contract


@pytest.mark.parametrize(
    "snakefile",
    [
        "analyze_climate.smk",
        "build_model.smk",
        "analyze_projections.smk",
        "generate_scenarios.smk",
    ],
)
def test_workflow_defers_header_and_summarizes_requested_outputs(
    snakefile, monkeypatch, capsys
):
    from blueearth_cst.shared import console_style

    if snakefile == "analyze_projections.smk":
        from blueearth_cst.projections import reference_window

        monkeypatch.setattr(reference_window, "SHORT_WINDOW_YEARS", 30)
    monkeypatch.setattr(console_style, "_DEFERRED_WARNINGS", [])
    capsys.readouterr()
    workflow = _parse_workflow(snakefile, CONFIG_FN)
    parsed = capsys.readouterr()
    if snakefile == "analyze_projections.smk":
        # Queued at run start, not DAG parse, so a reused analysis stays quiet.
        assert "Reference period is short" not in parsed.out + parsed.err
        assert not any(
            "Reference period is short" in row[2]
            for row in console_style._DEFERRED_WARNINGS
        )
        with monkeypatch.context() as patch:
            for name in ("install_console_style", "_header", "_journal"):
                patch.setitem(workflow.globals, name, lambda *args: None)
            workflow._onstart(None)
        assert any(
            "Reference period is short" in row[2]
            for row in console_style._DEFERRED_WARNINGS
        )
    monkeypatch.setattr(console_style, "_CONSOLE_STYLE_ACTIVE", True)
    monkeypatch.setattr(console_style, "_RUN_HEADER", None)
    capsys.readouterr()
    if snakefile == "build_model.smk":
        from blueearth_cst.experiment import batch_sizing

        monkeypatch.setattr(batch_sizing, "count_active_cells", lambda path: None)
    workflow.globals["_header"]()
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    details = console_style._RUN_HEADER[3]
    if snakefile != "generate_scenarios.smk":
        # Source entries are sorted by ID, so the project need not be first.
        source = str(CONFIG_FN.parent / "user-project.yml")
        monkeypatch.setitem(
            workflow.globals,
            "CAPTURE_RECORD",
            {
                "source_config": "project",
                "source_files": [
                    {"id": "catalog", "original_path": "catalog.yml"},
                    {"id": "project", "original_path": source},
                ],
            },
        )
        workflow.globals["_header"]()
        assert console_style._RUN_HEADER[2] == source
    else:
        assert Path(console_style._RUN_HEADER[2]).resolve() == CONFIG_FN.resolve()
    header = console_style.run_header(*console_style._RUN_HEADER[:3], **details)
    paths = header.splitlines()[header.splitlines().index("Paths") + 1 :]
    assert paths[0].split()[0] == "config"
    if snakefile == "analyze_climate.smk":
        window = workflow.globals["historical_window"]
        period = f"{window['start']}–{window['end']}"
        assert details == {
            "climate data": ", ".join(workflow.globals["CANDIDATE_SOURCES"]),
            "analysis period": period,
            "water year": f"Starts in {workflow.globals['WATER_YEAR_START']}",
        }
        # A distinct climatological baseline must remain visible when configured.
        monkeypatch.setitem(
            workflow.globals,
            "DIAGNOSTIC_SETTINGS",
            workflow.globals["DIAGNOSTIC_SETTINGS"]
            | {"reference_period": {"start": 2001, "end": 2010}},
        )
        monkeypatch.setitem(workflow.globals, "WATER_YEAR_START", "Oct")
        workflow.globals["_header"]()
        assert console_style._RUN_HEADER[3] == {
            "climate data": details["climate data"],
            "analysis period": period,
            "reference period": "2001–2010",
            "water year": "Starts in Oct",
        }
    elif snakefile == "build_model.smk":
        assert details["hydrological model"] == "Wflow-SBM"
        assert details["climate forcing"] == workflow.globals["clim_source"]
        window = workflow.globals["simulation_window"]
        assert details["simulation period"] == f"{window['start']}–{window['end']}"
        assert details["grid spacing"] == f"{workflow.globals['model_resolution']:g}°"
        assert details["model grid"] == "Cell count available after model build"
        assert details["water year"].startswith("Starts in")
        monkeypatch.setattr(batch_sizing, "count_active_cells", lambda path: 1234)
        workflow.globals["_header"]()
        assert (
            console_style._RUN_HEADER[3]["model grid"]
            == "1,234 active cells in existing model"
        )

        def unreadable_grid(path):
            raise OSError("existing model is unreadable")

        monkeypatch.setattr(batch_sizing, "count_active_cells", unreadable_grid)
        workflow.globals["_header"]()
        assert (
            console_style._RUN_HEADER[3]["model grid"]
            == "Cell count unavailable from existing model"
        )
    elif snakefile == "analyze_projections.smk":
        assert (
            details["reference period"]
            == workflow.globals["REFERENCE_WINDOW"].effective_label
        )
        assert details["future periods"]
        assert details["emissions scenarios"]
        assert details["climate models"]
        assert details["water year"].startswith("Starts in")
    elif snakefile == "generate_scenarios.smk":
        generation = workflow.globals["GENERATION"]
        assert details["historical climate"] == generation["source"]
        assert details["historical period"]
        assert (
            details["scenario period"] == f"{generation['start']}–{generation['end']}"
        )
        assert details["weather realizations"] == generation["n_realizations"]
        assert details["climate perturbations"] == generation["n_design_points"]
        assert details["scenario series"] == len(workflow.globals["SCENARIO_ROWS"])
        assert details["water year"].startswith("Starts in")

    rule = workflow.get_rule("all")
    # Input functions resolve at DAG time; the console recounts those inputs.
    count = sum(1 for item in rule.input if not callable(item))
    assert rule.message == f"Requested outputs: {count}/{count} provided"


def _differences(left, right):
    """Compare the existing exhaustive directive contract, allowing fan-out names."""
    return [
        field
        for field, extract in _COMPARED.items()
        if field != "name" and extract(*left) != extract(*right)
    ]


def _resolved_annotation_params(rule):
    """Resolve WF2's series-key lambdas instead of comparing function addresses."""
    values = []
    for value in rule.params:
        if callable(value):
            series = value.__globals__["SERIES"]
            values.append(
                {key: value(SimpleNamespace(series_key=key)) for key in sorted(series)}
            )
        else:
            values.append(value)
    return values


@pytest.mark.parametrize(
    "snakefile,plot_rules",
    [
        (
            "analyze_climate.smk",
            {
                "plot_climate_diagnostics_era5",
                "plot_climate_diagnostics_chirps",
                "plot_climate_comparison",
            },
        ),
        (
            "build_model.smk",
            {
                "plot_basin_map",
                "plot_model_forcing",
                "plot_model_evaluation",
            },
        ),
        ("analyze_projections.smk", {"plot_climate_projections"}),
    ],
)
def test_annotation_changes_only_plot_rule_parameters(tmp_path, snakefile, plot_rules):
    """Presentation edits keep every upstream input/output/parameter unchanged."""
    cfg = load_composed_config(CONFIG_FN)
    cfg["climate"]["sources"] = ["era5", "chirps"]
    path = write_config(tmp_path, cfg)
    original = _parse_workflow(snakefile, path)
    originals = {rule.name: rule for rule in original.rules}
    for mode in ("caption", "title"):
        cfg["plotting"] = {"annotation": mode}
        write_config(tmp_path, cfg)
        changed = _parse_workflow(snakefile, path)
        changed_rules = set()
        for rule in changed.rules:
            differences = _differences(
                (original, originals[rule.name]), (changed, rule)
            )
            if "params" in differences and _resolved_annotation_params(
                originals[rule.name]
            ) == _resolved_annotation_params(rule):
                differences.remove("params")
            if differences:
                assert differences == ["params"], (rule.name, differences)
                assert rule.params.annotation == mode
                changed_rules.add(rule.name)
        assert changed_rules == plot_rules


@pytest.mark.parametrize("source", ["era5", "chirps", "chirps_global"])
def test_wf0_exclusively_owns_source_diagnostics_at_store_root(tmp_path, source):
    cfg = load_composed_config(CONFIG_FN)
    cfg["climate"]["selected"] = source
    cfg["climate"]["sources"] = [source]
    path = write_config(tmp_path, cfg)
    wf0 = _parse_workflow("analyze_climate.smk", path)
    wf1 = _parse_workflow("build_model.smk", path)
    canonical = wf0.get_rule(f"plot_climate_diagnostics_{source}")
    assert "plot_climate_datasets" not in {rule.name for rule in wf1.rules}
    store = Path(
        wf0.get_rule(f"extract_climate_datasets_{source}").output.climate_nc
    ).parent
    assert all(Path(p).is_relative_to(store) for p in canonical.output)
    assert not any(
        "diagnostics" in Path(p).relative_to(store).parts for p in canonical.output
    )
    assert all(
        Path(p).parent == store / "plots"
        for p in canonical.output
        if str(p).endswith(".png")
    )
    assert not any(
        Path(p).is_relative_to(store / "plots")
        for rule in wf1.rules
        for p in rule.output
    )
    assert not any(p.is_directory for p in canonical.output)


def test_all_shared_outputs_have_equivalent_producers():
    """Discover overlaps across WF0–WF3, including future plot rules.

    WF4 consumes a published collection and owns experiment-local outputs;
    parsing it requires that collection and is covered by its lifecycle tests.
    """
    owners = defaultdict(list)
    for snakefile in (
        "analyze_climate.smk",
        "build_model.smk",
        "analyze_projections.smk",
        "generate_scenarios.smk",
    ):
        workflow = _parse_workflow(snakefile, CONFIG_FN)
        for rule in workflow.rules:
            for output in rule.output:
                owners[str(output)].append((workflow, rule))
    shared = {path: entries for path, entries in owners.items() if len(entries) > 1}
    assert any(path.endswith("extract_historical.nc") for path in shared)
    failures = []
    for path, entries in shared.items():
        for left, right in combinations(entries, 2):
            differences = _differences(left, right)
            if differences:
                failures.append((path, left[1].name, right[1].name, differences))
    assert not failures, failures


def test_repeated_wf0_diagnostics_preserve_provenance_but_detect_changes(tmp_path):
    """Repeated source plotting preserves metadata; changed presentation invalidates."""
    from snakemake.api import DAGSettings, ExecutionSettings
    from snakemake.io import IOCache
    from snakemake.jobs import Job
    from snakemake.persistence import Persistence

    cfg = load_composed_config(CONFIG_FN)
    cfg["project"]["project_dir"] = (tmp_path / "project").as_posix()
    cfg["climate"]["water_year_start"] = "Jan"
    path = write_config(tmp_path, cfg)
    jobs = []
    for snakefile, name in (
        ("analyze_climate.smk", "plot_climate_diagnostics_era5"),
        ("analyze_climate.smk", "plot_climate_diagnostics_era5"),
    ):
        workflow = _parse_workflow(snakefile, path)
        workflow.execution_settings = ExecutionSettings()
        workflow.dag_settings = DAGSettings()
        workflow.iocache = IOCache(0)
        jobs.append(
            Job(workflow.get_rule(name), SimpleNamespace(workflow=workflow), {})
        )
    persistence = Persistence(path=tmp_path / "metadata", dag=jobs[0].dag)
    for previous, current in ((jobs[0], jobs[1]), (jobs[1], jobs[0])):
        asyncio.run(persistence.finished(previous))
        # Each invocation opens a fresh metadata reader; Snakemake caches reads
        # within an invocation, including the pre-completion empty record.
        persistence = Persistence(path=tmp_path / "metadata", dag=current.dag)
        assert persistence.record_format_version(previous.output[0]) is not None
        assert previous.params.m0 == 1
        assert not list(persistence.input_changed(current))
        assert not persistence.params_changed(current)
        assert not list(persistence.code_changed(current))
        assert not list(persistence.software_stack_changed(current))
    # A genuine presentation change must still invalidate the saved job.
    cfg["climate"]["water_year_start"] = "Oct"
    path = write_config(tmp_path / "changed", cfg)
    workflow = _parse_workflow("analyze_climate.smk", path)
    workflow.execution_settings = ExecutionSettings()
    workflow.dag_settings = DAGSettings()
    changed = Job(
        workflow.get_rule("plot_climate_diagnostics_era5"),
        SimpleNamespace(workflow=workflow),
        {},
    )
    assert changed.params.m0 == 10
    assert persistence.params_changed(changed), (
        persistence.params(changed.output[0]),
        list(changed.non_derived_params),
    )
