"""Indicator sets over a real retained v2 experiment, end to end and unstubbed.

`tests/_v2_experiment.py` builds the experiment with the production WF3/WF4
record producers. Nothing on the reading side is replaced here: planning,
reduction, publication and reading all go through `indicator_plan`'s v2 branch.
The live indicator-stage environment is presented as `FIXTURE_ENVIRONMENT`
wherever a test plans against it, because publication re-observes it.
"""

import csv
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from blueearth_cst.experiment import indicator_sets
from blueearth_cst.experiment.content_identity import (
    canonical_json_bytes,
    content_sha256,
    identity_segment,
    read_canonical_json,
)
from blueearth_cst.experiment.indicator_plan import (
    ImmutableIndicatorSetError,
    IndicatorPlanStale,
    build_indicator_plan,
    current_indicator_request,
    publish_indicator_set,
    read_indicator_set,
    verify_indicator_plan,
    write_indicator_plan,
)
from blueearth_cst.experiment.indicator_sets import default_calculations
from blueearth_cst.experiment.simulation_record import read_simulation_intent_v2
from blueearth_cst.shared.indicator_tables import VARIABLE_TOKENS
from blueearth_cst.shared.provenance import file_sha256
from tests._v2_experiment import (
    FIXTURE_ENVIRONMENT,
    RUN_IDS,
    Responses,
    build_v2_experiment,
    copy_v2_experiment,
)


def _live(monkeypatch, observed=FIXTURE_ENVIRONMENT):
    from blueearth_cst.experiment import indicator_plan

    monkeypatch.setattr(
        indicator_plan, "resolve_indicator_environment", lambda: observed
    )


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return build_v2_experiment(tmp_path_factory.mktemp("v2-experiment"))


@pytest.fixture
def bare(built, tmp_path):
    """A private copy, with the live indicator environment left as it really is."""
    return copy_v2_experiment(built, tmp_path)


@pytest.fixture
def experiment(bare, monkeypatch):
    _live(monkeypatch)
    return bare


@pytest.fixture(scope="module")
def _published_once(built, tmp_path_factory):
    """One experiment with its gwr indicator set already published.

    The tamper tests only need a ready set to corrupt, and planning plus
    publication is most of their cost, so they share this one and each gets a
    private copy. Every reference inside a set is project-relative.
    """
    experiment = copy_v2_experiment(built, tmp_path_factory.mktemp("published"))
    with pytest.MonkeyPatch.context() as patch:
        from blueearth_cst.experiment import indicator_plan

        patch.setattr(
            indicator_plan, "resolve_indicator_environment", lambda: FIXTURE_ENVIRONMENT
        )
        plan = _plan(experiment.root)
        publish_indicator_set(experiment.root, plan)
    return experiment, identity_segment(plan["indicator_set_id"], "indicator_set_id")


@pytest.fixture
def published(_published_once, tmp_path, monkeypatch):
    """A private copy: (root, engine marker, result directory)."""
    _live(monkeypatch)
    source, short = _published_once
    root = copy_v2_experiment(source, tmp_path).root
    marker = root / "_engine/indicator_sets" / short / "indicators.json"
    return root, marker, marker.parent / "data"


def _plan(root, tokens=("gwr",)):
    return build_indicator_plan(
        root,
        current_indicator_request(root, default_calculations(list(tokens)), "YS-JAN"),
    )


def _engine(root, plan):
    short = identity_segment(plan["indicator_set_id"], "indicator_set_id")
    return root / "_engine/indicator_sets" / short


def _result(root, plan):
    short = identity_segment(plan["indicator_set_id"], "indicator_set_id")
    return root / "_engine/indicator_sets" / short / "data"


def test_indicators_use_only_simulated_perturbed_members(tmp_path, monkeypatch):
    _live(monkeypatch)
    experiment = build_v2_experiment(tmp_path, include_unperturbed=False)
    root = experiment.root
    assert not (root / "hydrology/wflow/output/run_1.csv").exists()
    plan = _plan(root)
    assert {row["run_id"] for row in plan["units"]} == {"run_2"}
    manifest = publish_indicator_set(root, plan)
    assert read_indicator_set(root, _engine(root, plan) / "indicators.json") == manifest
    for scope, unit in (("run", "run_2"), ("group", "group_1")):
        table = pd.read_csv(_result(root, plan) / f"gwr_{scope}_indicators.csv")
        assert set(table[f"{scope}_id"]) == {unit}
        assert np.allclose(table["value"], 9.0)


def test_request_identity_is_stable_before_simulation_publication(experiment):
    """The combined DAG names requests after freezing, before responses exist."""
    root = experiment.root
    calculations = default_calculations(["gwr"])
    completed = current_indicator_request(root, calculations, "YE-DEC")
    (root / "_engine/simulation.json").unlink()
    (root / "_engine/response_inventory.json").unlink()

    pending = current_indicator_request(root, calculations, "YE-DEC")

    assert pending == completed
    assert content_sha256(pending) == content_sha256(completed)


def test_named_requests_share_provenance_but_new_observations_detect_drift(
    experiment, monkeypatch
):
    from blueearth_cst.experiment import indicator_plan

    calls = {"environment": 0, "code": 0}

    def environment():
        calls["environment"] += 1
        return {"revision": calls["environment"]}

    def code(set_ids):
        assert set_ids == ("default",)
        calls["code"] += 1
        return [{"path": "indicator.py", "sha256": str(calls["code"]) * 64}]

    monkeypatch.setattr(indicator_plan, "resolve_indicator_environment", environment)
    monkeypatch.setattr(indicator_plan, "_indicator_code_inventory", code)
    calculations = [default_calculations(["gwr"]), default_calculations(["q"])]
    requests = indicator_plan.current_indicator_requests(
        experiment.root, calculations, "YE-DEC"
    )
    assert calls == {"environment": 1, "code": 1}
    for records, request in zip(calculations, requests):
        expected = indicator_plan.indicator_request(
            request["simulation_id"],
            records,
            "YE-DEC",
            request["indicator_environment"],
            request["return_level_validation"],
            read_simulation_intent_v2(experiment.root)["documents"]["response_request"],
            code_inventory=request["code_inventory"],
        )
        expected["simulation_schema_version"] = "simulation/3"
        assert request == expected

    fresh = indicator_plan.current_indicator_requests(
        experiment.root, calculations, "YE-DEC"
    )
    assert calls == {"environment": 2, "code": 2}
    assert all(content_sha256(a) != content_sha256(b) for a, b in zip(requests, fresh))


@pytest.mark.parametrize("set_id", ["default", "flow_regime", "flood_risk"])
def test_request_identity_tracks_only_contributing_sets(tmp_path, monkeypatch, set_id):
    from blueearth_cst.experiment import indicator_plan, return_level_validation
    from blueearth_cst.indicators import loader

    repo = Path(indicator_plan.__file__).resolve().parents[2]
    shutil.copytree(
        repo / "blueearth_cst",
        tmp_path / "blueearth_cst",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    monkeypatch.setattr(
        indicator_plan,
        "__file__",
        str(tmp_path / "blueearth_cst/experiment/indicator_plan.py"),
    )
    execution_sources = loader.execution_sources

    def copied_sources(set_ids=None):
        return tuple(
            tmp_path / path.relative_to(repo) for path in execution_sources(set_ids)
        )

    monkeypatch.setattr(loader, "execution_sources", copied_sources)
    validation = return_level_validation.build_declaration()
    path = indicator_sets.BUNDLED_ROOT / f"{set_id}.yml"

    def request():
        resolved = indicator_sets.resolve_set(
            indicator_sets.SetSource(
                set_id, path, True, path.read_text(encoding="utf-8")
            ),
            ["gwr"],
        )
        return indicator_plan.indicator_request(
            "simulation",
            indicator_sets.calculation_records(resolved.indicators),
            "YE-DEC",
            FIXTURE_ENVIRONMENT,
            validation,
            {
                "variables": [
                    {"variable": token, "locations": ["1"]}
                    for token in VARIABLE_TOKENS.values()
                ]
            },
        )

    before = request()
    for relative in (
        "plotting/export.py",
        "shared/snake_utils.py",
        "shared/config_composition.py",
        "shared/workflow_config_snapshot.py",
        "experiment/reservation_context.py",
        "experiment/generation_plan.py",
        "experiment/simulator_adapter.py",
        "experiment/simulation_record.py",
    ):
        unrelated = tmp_path / "blueearth_cst" / relative
        unrelated.write_bytes(unrelated.read_bytes() + b"\n# unrelated edit\n")
        assert request() == before, relative

    for relative in (
        "experiment/indicator_groups.py",
        "experiment/wflow_response_reader.py",
        "experiment/export_wflow_results.py",
    ):
        scientific = tmp_path / "blueearth_cst" / relative
        original = scientific.read_bytes()
        scientific.write_bytes(original + b"\n# contributing edit\n")
        assert request() != before, relative
        scientific.write_bytes(original)

    root = tmp_path / "blueearth_cst/indicators"
    new_set = root / "sets/new_unrelated_set"
    new_set.mkdir()
    (new_set / "__init__.py").write_text("", encoding="utf-8")
    (new_set / "registry.py").write_text("INDICATORS = ()\n", encoding="utf-8")
    assert request() == before

    unselected = "flood_risk" if set_id != "flood_risk" else "flow_regime"
    unrelated = root / f"sets/{unselected}/calculations.py"
    unrelated.write_text(
        unrelated.read_text(encoding="utf-8") + "\n# unrelated edit\n", encoding="utf-8"
    )
    assert request() == before

    # Flow-regime aliases really depend on the default package as well.
    selected = "default" if set_id == "flow_regime" else set_id
    contributing = root / f"sets/{selected}/calculations.py"
    contributing.write_text(
        contributing.read_text(encoding="utf-8") + "\n# contributing edit\n",
        encoding="utf-8",
    )
    assert request() != before

    after_selected = request()
    shared = root / "temporal.py"
    shared.write_text(
        shared.read_text(encoding="utf-8") + "\n# shared edit\n", encoding="utf-8"
    )
    assert request() != after_selected


def test_simulations_only_does_not_resolve_indicator_requirements(
    experiment, monkeypatch
):
    from blueearth_cst.experiment import indicator_plan, simulation_runner

    monkeypatch.setattr(
        indicator_plan,
        "current_indicator_request",
        lambda *_: pytest.fail("simulations_only must not resolve indicators"),
    )
    monkeypatch.setattr(
        simulation_runner,
        "resolve_targets",
        lambda _, targets, **_kwargs: (
            "simulate-and-indicators",
            simulation_runner.current_targets(targets),
        ),
    )
    _, environment = simulation_runner.simulation_command(
        experiment.config, ["responses"], 1
    )
    assert environment["CST_SIMULATION_TARGET"] == "simulations_only"


def test_the_fixture_is_a_ready_v2_experiment_wherever_it_is_copied(experiment):
    from blueearth_cst.experiment.response_inventory import read_response_inventory_v2
    from blueearth_cst.experiment.scenario_collection_v2 import read_collection_v2
    from blueearth_cst.experiment.simulation_record import read_simulation_v2

    root = experiment.root
    # indicator_plan takes its v1 branch whenever the v2 marker is absent, so a
    # broken fixture would otherwise test the legacy path without saying so.
    assert (root / "_engine/simulation.json").is_file()
    assert not (root / "config/simulation.json").exists()
    assert read_simulation_v2(root)["status"] == "ready"
    assert read_response_inventory_v2(root)["schema_version"] == (
        "response-inventory/3"
    )
    marker = read_collection_v2(experiment.collection_marker)
    assert [item["run_id"] for item in marker["series"]] == list(RUN_IDS)


def test_a_v2_indicator_set_publishes_once_and_is_reused_byte_for_byte(
    experiment, monkeypatch
):
    from blueearth_cst.experiment import indicator_plan
    from blueearth_cst.indicators import loader

    calculate = loader.evaluate
    calls = []

    def counted(indicator, series, **context):
        calls.append(tuple(item.run_id for item in series))
        return calculate(indicator, series, **context)

    monkeypatch.setattr(loader, "evaluate", counted)

    root = experiment.root
    plan = _plan(root)
    assert plan["schema_version"] == "indicator-request/5"
    assert len(plan["expected_result_keys"]) == 4
    assert plan["units"] == [
        {"run_group_id": run, "grain": "run", "run_id": run} for run in RUN_IDS
    ] + [
        {"run_group_id": f"group_{i}", "grain": "bundle", "run_id": run}
        for i, run in enumerate(RUN_IDS, 1)
    ]
    manifest = publish_indicator_set(root, plan)
    assert calls == [(run,) for run in RUN_IDS], "group means must reuse run values"
    marker = _engine(root, plan) / "indicators.json"
    assert (_result(root, plan) / "gwr_run_indicators.csv").is_file()
    assert not (root / "results/indicator_sets").exists()
    assert (
        plan["targets"]["unit_index"]
        == (_result(root, plan) / "group_members.csv").as_posix()
    )
    files = sorted(p for p in marker.parent.rglob("*") if p.is_file())
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in files}
    assert read_indicator_set(root, marker) == manifest

    repo = Path(indicator_plan.__file__).resolve().parents[2]
    changed = {
        repo / "blueearth_cst" / relative
        for relative in (
            "plotting/export.py",
            "shared/snake_utils.py",
            "shared/config_composition.py",
            "experiment/generation_plan.py",
            "experiment/simulator_adapter.py",
        )
    }
    read_bytes = Path.read_bytes

    def edited_bytes(path):
        data = read_bytes(path)
        return data + b"\n# unrelated edit\n" if path in changed else data

    monkeypatch.setattr(Path, "read_bytes", edited_bytes)
    monkeypatch.setattr(
        indicator_plan,
        "reduce_indicator_plan",
        lambda *args, **kwargs: pytest.fail("ready reuse must not reduce again"),
    )
    assert publish_indicator_set(root, _plan(root)) == manifest
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in files}


def test_flow_regime_group_diagnostics_publish_as_json_without_changing_values(
    experiment,
):
    """The bundled recharge alias retains tuple membership metadata in memory."""
    root = experiment.root
    baseline_plan = _plan(root)
    publish_indicator_set(root, baseline_plan)
    baseline_files = {
        path: path.read_bytes()
        for path in _engine(root, baseline_plan).rglob("*")
        if path.is_file()
    }
    selection = indicator_sets.parse_selection(
        {"operations": ["indicators"], "indicator_sets": ["flow_regime"]},
        experiment.config,
    )
    resolved = indicator_sets.resolve_set(selection.sets[0])
    calculations = indicator_sets.calculation_records(
        [item for item in resolved.indicators if item.indicator == "recharge_depth"]
    )
    assert len(calculations) == 1
    plan = build_indicator_plan(
        root, current_indicator_request(root, calculations, "YS-JAN")
    )

    manifest = publish_indicator_set(root, plan)

    engine = _engine(root, plan)
    diagnostics_path = engine / "indicator_diagnostics.json"
    diagnostics = read_canonical_json(diagnostics_path)
    assert canonical_json_bytes(diagnostics) == diagnostics_path.read_bytes()
    grouped = [
        item for item in diagnostics["results"] if item["unit"].startswith("group_")
    ]
    assert len(grouped) == 2
    for item in grouped:
        expected = [
            row["run_id"]
            for row in plan["units"]
            if row["run_group_id"] == item["unit"]
        ]
        assert item["metadata"]["member_run_ids"] == expected
    for table in manifest["tables"]:
        with (root / table["file"]["path"]).open(
            encoding="utf-8", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))
        assert rows
        assert all(
            float(row["value"]) == 9.0 for row in rows
        )  # 3 + 6 daily recharge depths
    assert read_indicator_set(root, engine / "indicators.json") == manifest
    before = {path: path.read_bytes() for path in engine.rglob("*") if path.is_file()}
    assert publish_indicator_set(root, plan) == manifest
    assert before == {
        path: path.read_bytes() for path in engine.rglob("*") if path.is_file()
    }
    assert baseline_files == {path: path.read_bytes() for path in baseline_files}


def _relocate_to_legacy_root(root, marker, result, schema):
    manifest = read_indicator_set(root, marker)
    legacy = root / "results/indicator_sets" / marker.parent.name
    legacy.parent.mkdir(parents=True)
    shutil.move(result, legacy)
    # Sets before indicator-set/6 carry four-column tables.
    for item in manifest["tables"]:
        path = legacy / item["file"]["path"].split("/")[-1]
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text(
            "".join(",".join(line.split(",")[:4]) + "\n" for line in lines),
            encoding="utf-8",
        )
        item["file"]["sha256"] = file_sha256(path)
    for field in ("unavailable_calculations", "unavailable_results"):
        manifest.pop(field)
    manifest["schema_version"] = schema
    manifest["result_root"] = legacy.relative_to(root).as_posix()
    for reference in [
        manifest["group_members"],
        manifest["group_metadata"],
        *(item["file"] for item in manifest["tables"]),
    ]:
        reference["path"] = (
            f"{manifest['result_root']}/{reference['path'].split('/')[-1]}"
        )
    manifest["indicators_manifest_sha256"] = content_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key != "indicators_manifest_sha256"
        }
    )
    marker.write_bytes(canonical_json_bytes(manifest))
    return manifest, legacy


def test_a_legacy_v3_results_root_stays_readable(published):
    """Read-only /3 support: a legacy results/<digest>/ root is still read."""
    root, marker, result = published
    manifest, legacy = _relocate_to_legacy_root(root, marker, result, "indicator-set/3")
    files = [marker, *legacy.iterdir()]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in files}
    assert read_indicator_set(root, marker) == manifest
    assert before == {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in files
    }


def test_a_v4_set_never_adopts_a_legacy_results_root(published):
    root, marker, result = published
    _relocate_to_legacy_root(root, marker, result, "indicator-set/4")
    with pytest.raises(ImmutableIndicatorSetError, match="result identity"):
        read_indicator_set(root, marker)
    plan = _plan(root)
    assert plan["targets"]["unit_index"] == (result / "group_members.csv").as_posix()


@pytest.mark.parametrize("changed", ["root", "lookup", "table"])
def test_mixed_or_escaping_indicator_layout_is_refused(published, changed):
    root, marker, _ = published
    manifest = read_indicator_set(root, marker)
    if changed == "root":
        manifest["result_root"] = "../outside"
    elif changed == "lookup":
        manifest["group_members"]["path"] = (
            f"results/indicator_sets/{marker.parent.name}/group_members.csv"
        )
    else:
        manifest["tables"][0]["file"]["path"] = "../outside.csv"
    manifest["indicators_manifest_sha256"] = content_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key != "indicators_manifest_sha256"
        }
    )
    marker.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(ImmutableIndicatorSetError, match="identity|reference"):
        read_indicator_set(root, marker)


def test_the_indicator_environment_changes_only_the_indicator_identity(experiment):
    root = experiment.root
    request = current_indicator_request(root, default_calculations(["gwr"]), "YS-JAN")
    before = build_indicator_plan(root, request)
    changed = deepcopy(request)
    changed["indicator_environment"]["packages"]["numpy"] = "another"
    after = build_indicator_plan(root, changed)
    assert after["indicator_set_id"] != before["indicator_set_id"]
    assert after["response_inventory_sha256"] == before["response_inventory_sha256"]
    assert after["request"]["simulation_id"] == before["request"]["simulation_id"]


def test_an_xclim_revision_changes_only_the_indicator_identity(bare, monkeypatch):
    from blueearth_cst.experiment import content_identity

    root = bare.root
    revision = "first"

    def environment(roots):
        return {
            "packages": {
                name: revision if name == "xclim" else "fixed" for name in roots
            }
        }

    monkeypatch.setattr(content_identity, "stage_environment", environment)
    before = _plan(root)
    revision = "second"
    after = _plan(root)
    assert before["indicator_set_id"] != after["indicator_set_id"]
    assert before["request"]["simulation_id"] == after["request"]["simulation_id"]
    assert before["response_inventory_sha256"] == after["response_inventory_sha256"]


def test_a_changed_native_response_byte_refuses_planning(experiment):
    native = experiment.root / f"hydrology/wflow/output/{RUN_IDS[0]}.csv"
    original = native.read_bytes()
    native.write_bytes(original.replace(b",3\n", b",4\n"))
    assert native.read_bytes() != original
    with pytest.raises(ValueError, match="bytes differ"):
        _plan(experiment.root)


@pytest.mark.parametrize("document", ["intent", "response_inventory", "collection"])
def test_verified_metadata_still_requires_the_exact_referenced_bytes(
    experiment, monkeypatch, document
):
    from blueearth_cst.experiment import indicator_plan

    root = experiment.root
    request = current_indicator_request(root, default_calculations(["gwr"]), "YS-JAN")
    validate = indicator_plan.read_simulation_v2

    def changed_after_validation(path):
        ready = validate(path)
        if document == "collection":
            target = experiment.collection_marker
        else:
            target = root.parent.parent / ready[document]["path"]
        target.write_bytes(target.read_bytes() + b"\n")
        return ready

    monkeypatch.setattr(indicator_plan, "read_simulation_v2", changed_after_validation)
    with pytest.raises(ValueError, match="bytes differ|size differs"):
        build_indicator_plan(root, request, _observed_code=request["code_inventory"])


def test_a_tampered_published_table_is_refused_on_read(published):
    root, marker, result = published
    read_indicator_set(root, marker)
    table = result / "gwr_run_indicators.csv"
    table.write_bytes(table.read_bytes() + b"gwr_mean,9,1,0\n")
    with pytest.raises(ImmutableIndicatorSetError):
        read_indicator_set(root, marker)


@pytest.mark.parametrize(
    "defect", ["legacy", "scope", "missing_indicator", "membership", "metadata"]
)
def test_rehashed_contract_corruption_is_refused(published, defect):
    """Valid file hashes cannot disguise invalid semantics or old layouts."""
    from blueearth_cst.shared.provenance import file_sha256

    root, marker, result = published
    manifest = read_indicator_set(root, marker)
    if defect == "legacy":
        manifest["schema_version"] = "indicator-set/2"
    elif defect in {"scope", "missing_indicator"}:
        entry = next(item for item in manifest["tables"] if item["scope"] == "run")
        path = result / "gwr_run_indicators.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        if defect == "scope":
            rows[0]["run_id"] = "group_1"
        else:
            rows.pop()
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        entry["file"]["sha256"] = file_sha256(path)
    else:
        name = "group_members" if defect == "membership" else "group_metadata"
        path = result / f"{name}.csv"
        path.write_text(
            path.read_text(encoding="utf-8").replace("run_1", "run_99")
            if defect == "membership"
            else path.read_text(encoding="utf-8").replace("true", "false"),
            encoding="utf-8",
        )
        manifest[name]["sha256"] = file_sha256(path)
    manifest["indicators_manifest_sha256"] = content_sha256(
        {k: v for k, v in manifest.items() if k != "indicators_manifest_sha256"}
    )
    marker.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(ImmutableIndicatorSetError):
        read_indicator_set(root, marker)


def test_tampered_return_level_evidence_is_refused_on_read(published):
    from blueearth_cst.experiment.indicator_plan import RETURN_LEVEL_EVIDENCE_FILENAME

    root, marker, _ = published
    evidence = marker.parent / RETURN_LEVEL_EVIDENCE_FILENAME
    assert read_canonical_json(evidence)["bundles"] == [], "gwr fits no return level"
    read_indicator_set(root, marker)
    evidence.write_bytes(b'{"bundles": [], "schema_version": "other"}\n')
    with pytest.raises(ImmutableIndicatorSetError, match="indicator artifact differs"):
        read_indicator_set(root, marker)


def test_new_sets_require_return_level_evidence(published):
    """The /3 schema requires explicit return-level evidence, even when empty."""
    root, marker, _ = published
    manifest = read_canonical_json(marker)
    del manifest["return_level_evidence"]
    manifest["indicators_manifest_sha256"] = content_sha256(
        {k: v for k, v in manifest.items() if k != "indicators_manifest_sha256"}
    )
    marker.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(ImmutableIndicatorSetError):
        read_indicator_set(root, marker)


def test_only_indicators_json_is_read_as_the_engine_marker(published):
    root, marker, _ = published
    renamed = marker.with_name("alternate.json")
    renamed.write_bytes(marker.read_bytes())
    with pytest.raises(ImmutableIndicatorSetError, match="named indicators.json"):
        read_indicator_set(root, renamed)


def test_a_table_renamed_in_the_manifest_is_refused(published):
    root, marker, result = published
    manifest = read_indicator_set(root, marker)
    (result / "alternate.csv").write_bytes(
        (result / "gwr_run_indicators.csv").read_bytes()
    )
    manifest["tables"][0]["file"]["path"] = (
        f"results/indicator_sets/{result.name}/alternate.csv"
    )
    manifest["indicators_manifest_sha256"] = content_sha256(
        {k: v for k, v in manifest.items() if k != "indicators_manifest_sha256"}
    )
    marker.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(ImmutableIndicatorSetError, match="table reference"):
        read_indicator_set(root, marker)


def test_the_published_set_carries_and_verifies_its_benchmark_report(published):
    from blueearth_cst.experiment import return_level_validation

    root, marker, _ = published
    manifest = read_canonical_json(marker)
    report = marker.parent / return_level_validation.REPORT_FILENAME
    assert report.read_bytes() == return_level_validation.load_report_bytes()
    assert manifest["return_level_benchmark"]["path"] == report.name
    assert read_indicator_set(root, marker) == manifest
    # A tampered copy must fail the reader rather than downgrade silently.
    report.write_bytes(b"{}\n")
    with pytest.raises(ImmutableIndicatorSetError, match="indicator artifact differs"):
        read_indicator_set(root, marker)


def test_a_stale_live_environment_refuses_reduction_before_any_fit(
    experiment, monkeypatch
):
    """D5: drift between planning and execution aborts before fitting."""
    from blueearth_cst.experiment import gev_lmoments
    from blueearth_cst.experiment import indicator_plan as plan_module

    root = experiment.root
    plan = _plan(root)
    _live(monkeypatch, {"packages": {"numpy": "moved"}})
    calls = []
    real = gev_lmoments.fit_case
    monkeypatch.setattr(
        gev_lmoments, "fit_case", lambda *a, **k: (calls.append(a), real(*a, **k))[1]
    )
    with pytest.raises(IndicatorPlanStale, match="live indicator-stage environment"):
        plan_module.reduce_indicator_plan(root, plan)
    assert calls == [], "the estimator ran despite a stale environment"


def test_a_stale_live_environment_leaves_no_ready_marker(experiment, monkeypatch):
    root = experiment.root
    plan = _plan(root)
    _live(monkeypatch, {"packages": {"numpy": "moved"}})
    with pytest.raises(IndicatorPlanStale):
        publish_indicator_set(root, plan)
    assert not (_engine(root, plan) / "indicators.json").exists()


def _moved_plan(experiment, tmp_path_factory):
    """Write the gwr plan, then copy the experiment to another project root."""
    request = current_indicator_request(
        experiment.root, default_calculations(["gwr"]), "YS-JAN"
    )
    write_indicator_plan(experiment.root, request)
    moved = copy_v2_experiment(experiment, tmp_path_factory.mktemp("moved")).root
    (stored,) = (moved / "_engine/indicator_requests").glob("*.json")
    return moved, request, stored


def test_a_plan_copied_from_another_root_verifies_at_the_new_root(
    experiment, tmp_path_factory
):
    """t2610042152: a stored plan's targets name the root it was written at."""
    moved, request, stored = _moved_plan(experiment, tmp_path_factory)
    assert experiment.root.as_posix() in stored.read_text(encoding="utf-8")
    plan = verify_indicator_plan(moved, request)
    assert plan == build_indicator_plan(moved, request)
    assert all(
        target.startswith(moved.as_posix() + "/")
        for target in [*plan["targets"].values(), *plan["tables"].values()]
    )


def _other_field(plan, origin):
    plan["expected_result_keys"] = plan["expected_result_keys"][:1]


def _mixed_roots(plan, origin):
    """One target keeps its tail but under a third root."""
    target = plan["tables"]["gwr_run"]
    plan["tables"]["gwr_run"] = "C:/elsewhere" + target[len(origin.as_posix()) :]


@pytest.mark.parametrize(
    "tamper, redigest",
    [(_other_field, True), (_mixed_roots, True), (lambda plan, origin: None, False)],
    ids=["another-field", "mixed-roots", "stale-self-digest"],
)
def test_a_moved_plan_still_refuses_any_other_difference(
    experiment, tmp_path_factory, tamper, redigest
):
    moved, request, stored = _moved_plan(experiment, tmp_path_factory)
    plan = read_canonical_json(stored)
    tamper(plan, experiment.root)
    body = {key: value for key, value in plan.items() if key != "request_sha256"}
    plan["request_sha256"] = content_sha256(body) if redigest else "0" * 64
    stored.write_bytes(canonical_json_bytes(plan))
    with pytest.raises(IndicatorPlanStale):
        verify_indicator_plan(moved, request)


@pytest.mark.parametrize("mismatch", [False, True])
def test_indicators_only_reads_the_anchor_and_asserts_the_retained_collection(
    experiment, capsys, mismatch
):
    from blueearth_cst.experiment import indicator_plan
    from blueearth_cst.experiment.simulation_record import read_simulation_intent_v2

    retained = read_simulation_intent_v2(experiment.root)["collection"]
    asserted = {
        "collection_id": retained["collection_id"],
        "collection_revision": "0" * 64
        if mismatch
        else retained["collection_revision"],
    }
    config = experiment.config
    workflow = config.parent / "simulate_system.yml"
    workflow.write_text(
        yaml.safe_dump(
            {
                "experiment_name": experiment.root.name,
                "indicators": ["gwr"],
                "scenario_collection": asserted,
                "seed": 123,
                "simulation_window": {"start": 9990, "end": 9999},
            }
        ),
        encoding="utf-8",
    )
    project = yaml.safe_load(config.read_text(encoding="utf-8"))
    project["climate"] = {"water_year_start": "oct"}
    project["model"] = {"name": "unavailable-model"}
    config.write_text(yaml.safe_dump(project), encoding="utf-8")
    if mismatch:
        with pytest.raises(ValueError, match="collection_revision"):
            indicator_plan.indicators_only_configuration(config)
        return
    root, sets, anchor = indicator_plan.indicators_only_configuration(config)
    assert root == experiment.root
    assert [item.indicator for item in sets[0].indicators] == ["gwr_annual_total"]
    assert anchor == "YE-SEP"
    output = capsys.readouterr().out
    assert "seed" in output and "simulation_window" in output
    assert "model.name" in output and "_engine/simulation.json" in output


# --- thirty years of discharge: real GEV fits and class-C month means --------

_TIME = pd.date_range("2046-01-01", "2075-12-31", freq="D")
_FACTORS = {"run_1": 1.0, "run_2": 1.5}


def _discharge_frames():
    rng = np.random.default_rng(12345)
    yearly = rng.uniform(0.8, 1.2, 30)[_TIME.year - 2046]
    base = np.where(_TIME.month == 7, 100.0, np.where(_TIME.month == 1, 5.0, 30.0))
    return {
        run: pd.DataFrame(
            {"Q_9": f * base * yearly, "Q_2": f * (120 - base) * yearly}, index=_TIME
        )
        for run, f in _FACTORS.items()
    }


def _discharge_responses(frames):
    temporal = {
        "source_calendar": "proleptic_gregorian",
        "prepared_forcing_calendar": "proleptic_gregorian",
        "response_calendar": "standard",
        "operations": ["clip_to_configured_window", "refresh_toml_endpoints"],
        "prepared_start": "2045-12-31 00:00:00",
        "prepared_end": str(_TIME[-1]),
        "response_start": str(_TIME[0]),
        "response_end": str(_TIME[-1]),
        "time_label": "interval_end",
    }
    return Responses(
        csv={
            run: frame.to_csv(index_label="time", lineterminator="\n").encode()
            for run, frame in frames.items()
        },
        toml=(
            '[time]\ncalendar="standard"\ntimestepsecs=86400\n'
            'starttime="2045-12-31T00:00:00"\nendtime="2075-12-31T00:00:00"\n'
            '[[output.csv.column]]\nheader="Q"\n'
            'parameter="river_water__volume_flow_rate"\n'
        ),
        temporal=temporal,
        variables=[
            {
                "variable": "q",
                "locations": ["9", "2"],
                "units": "m3 s-1",
                "calendar": "standard",
                "timestep": "P1D",
                "time_label": "interval_end",
                "start": str(_TIME[0]),
                "end": str(_TIME[-1]),
                "missing_value": "NaN",
            }
        ],
    )


def test_q_bundles_and_class_c_use_exact_units_and_shared_reference(
    tmp_path, monkeypatch
):
    """Thirty full years exercise actual GEV fits and independent month means."""
    _live(monkeypatch)
    frames = _discharge_frames()
    root = build_v2_experiment(tmp_path, _discharge_responses(frames), capacity=9).root
    plan = _plan(root, ("q",))
    assert plan["bundle_run_group_ids"] == {"": "group_1", "1": "group_2"}
    assert plan["units"] == [
        {"run_group_id": unit, "grain": grain, "run_id": run}
        for unit, grain, run in (
            ("run_1", "run", "run_1"),
            ("run_2", "run", "run_2"),
            ("group_1", "bundle", "run_1"),
            ("group_2", "bundle", "run_2"),
        )
    ]
    for which, month in (("wet", 7), ("dry", 1)):
        reference = plan["resolved_references"][which]
        assert reference["month"] == month
        assert reference["reference_location_id"] == "9"
        assert reference["run_ids"] == ["run_1"]
    publish_indicator_set(root, plan)
    with open(
        _result(root, plan) / "q_run_indicators.csv", encoding="utf-8", newline=""
    ) as handle:
        results = list(csv.DictReader(handle))
    keys = [[r["indicator"], r["location"], r["run_id"]] for r in results]
    assert sorted(keys) == [
        key for key in plan["expected_result_keys"] if key[2].startswith("run_")
    ]
    assert len(keys) == len({tuple(key) for key in keys})
    for suffix, month in (("wettest_month_mean", 7), ("driest_month_mean", 1)):
        selected = [row for row in results if row["indicator"] == f"q_{suffix}"]
        assert {(r["run_id"], r["location"]) for r in selected} == {
            (run, location) for run in RUN_IDS for location in ("9", "2")
        }
        for row in selected:
            daily = frames[row["run_id"]][f"Q_{row['location']}"]
            expected = (
                daily[daily.index.month == month].resample("YS-JAN").mean().mean()
            )
            # The retained CSV contract rounds float32 values to four digits.
            assert float(row["value"]) == float(f"{np.float32(expected):.4g}")
    # Return levels are fitted per bundle, never per run.
    fitted = {
        r["run_id"] for r in results if r["indicator"].startswith("q_return_level")
    }
    assert fitted == set()
    with (_result(root, plan) / "q_group_indicators.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        group_results = list(csv.DictReader(handle))
    group_keys = [[r["indicator"], r["location"], r["group_id"]] for r in group_results]
    assert sorted(group_keys) == [
        key for key in plan["expected_result_keys"] if key[2].startswith("group_")
    ]
    assert {
        r["group_id"]
        for r in group_results
        if r["indicator"].startswith("q_return_level")
    } == {"group_1", "group_2"}
    member_to_group = {"run_1": "group_1", "run_2": "group_2"}
    for row in results:
        grouped = next(
            r
            for r in group_results
            if (r["indicator"], r["location"], r["group_id"])
            == (row["indicator"], row["location"], member_to_group[row["run_id"]])
        )
        assert grouped["value"] == row["value"]
    manifest = read_canonical_json(_engine(root, plan) / "indicators.json")
    assert manifest["run_groups"] == plan["units"]
    # Every fitted return level keeps its provenance: one record per bundle and
    # indicator, each location with its usable-block total and fitted parameters.
    from blueearth_cst.experiment.indicator_plan import RETURN_LEVEL_EVIDENCE_FILENAME

    assert manifest["return_level_evidence"]["path"] == RETURN_LEVEL_EVIDENCE_FILENAME
    evidence = read_canonical_json(_engine(root, plan) / RETURN_LEVEL_EVIDENCE_FILENAME)
    assert evidence["schema_version"] == "return-level-evidence/1"
    assert {item["run_group_id"] for item in evidence["bundles"]} == {
        "group_1",
        "group_2",
    }
    for item in evidence["bundles"]:
        assert {loc["location_id"] for loc in item["locations"]} == {"9", "2"}
        for location in item["locations"]:
            assert location["total"] == 30
            assert location["total"] >= location["required"]
            assert location["parameters"], "the fitted parameters are retained"


def test_a_pre_release_v1_experiment_is_refused_by_name(tmp_path):
    """No release wrote v1 records; one is named as such, not read as missing."""
    from blueearth_cst.experiment.simulation_record import (
        IndicatorsOnlySimulationUnavailable,
    )

    root = tmp_path / "project/experiments/old_experiment"
    (root / "config").mkdir(parents=True)
    (root / "config/simulation.json").write_bytes(b"{}\n")
    with pytest.raises(IndicatorsOnlySimulationUnavailable, match="pre-release v1"):
        current_indicator_request(root, default_calculations(["gwr"]), "YS-JAN")
    with pytest.raises(IndicatorsOnlySimulationUnavailable, match="pre-release v1"):
        build_indicator_plan(root, {})
