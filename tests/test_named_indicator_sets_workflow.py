"""Named indicator sets through the real runner and indicator checkpoint.

Indicators-only over a retained v2 simulation: several sets plan in one
invocation, identical numerical requests share one immutable set, and an
authored alias publishes the registered calculation under its own indicator key.
"""

import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from blueearth_cst.experiment.content_identity import read_canonical_json
from tests._v2_experiment import RUN_IDS, Responses, build_v2_experiment

pytestmark = [
    pytest.mark.workflow_contract,
    pytest.mark.usefixtures("isolated_snakemake"),
]

REPO = Path(__file__).resolve().parents[1]

ALIAS = """\
schema_version: 1
id: depth_alias
title: Recharge depth alias
description: Authored alias of the registered recharge depth.
selection:
  indicators:
    - id: recharge_depth
      title: Recharge depth
      description: Same calculation as gwr_annual_total.
      references: [Example citation]
      calculation:
        method: annual_summary/1
        parameters: {variable: gwr, statistic: sum}
"""

SAME_AS_DEFAULT = """\
schema_version: 1
id: same_numbers
title: Same numbers as default
description: Selects the default's single gwr definition explicitly.
selection:
  indicators:
    - definition: gwr_annual_total
"""


def _named(experiment, settings):
    workflow = experiment.config.parent / "simulate_system.yml"
    (experiment.config.parent / "sets").mkdir(exist_ok=True)
    (experiment.config.parent / "sets/depth_alias.yml").write_text(
        ALIAS, encoding="utf-8"
    )
    (experiment.config.parent / "sets/same_numbers.yml").write_text(
        SAME_AS_DEFAULT, encoding="utf-8"
    )
    workflow.write_text(
        yaml.safe_dump({"experiment_name": experiment.root.name, **settings}),
        encoding="utf-8",
    )


def _simulate(experiment, *extra):
    environment = os.environ.copy()
    environment.pop("SNAKEMAKE_WORKFLOW_PROFILE", None)
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/simulate_system.py"),
            "--config",
            str(experiment.config),
            "--cores",
            "1",
            *extra,
        ],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        # Several requests import and fingerprint their execution environment
        # in separate workers; allow cold Windows runs without a speed claim.
        timeout=1200,
    )
    return result.returncode, result.stdout + result.stderr


SETTINGS = {
    "operations": ["indicators"],
    "indicator_sets": ["default", "depth_alias", "same_numbers"],
    "indicator_set_definitions": {
        "depth_alias": {"config_path": "sets/depth_alias.yml"},
        "same_numbers": {"config_path": "sets/same_numbers.yml"},
    },
}


def test_named_sets_dry_run_plans_each_distinct_request(tmp_path):
    experiment = build_v2_experiment(tmp_path)
    _named(experiment, SETTINGS)
    code, output = _simulate(experiment, "--dry-run")
    assert code == 0, output
    # default (outvars absent -> retained gwr) and same_numbers are one request.
    assert re.search(r"prepare_indicator_plan\s+2", output), output
    assert not (experiment.root / "_engine/indicator_requests").exists()


def test_named_sets_refuse_an_explicit_target(tmp_path):
    experiment = build_v2_experiment(tmp_path)
    _named(experiment, SETTINGS)
    code, output = _simulate(experiment, "--target", "indicators", "--dry-run")
    assert code != 0
    assert "omit --target" in output, output


@pytest.mark.skipif(os.name != "nt", reason="project writers require Windows")
def test_erfa_set_executes_and_rechecks_retained_discharge(tmp_path):
    """Exercise public hyphenated selection, publication and all numerical checkers."""
    time = pd.date_range("2046-01-02", "2048-12-31")
    temporal = {
        "source_calendar": "proleptic_gregorian",
        "prepared_forcing_calendar": "proleptic_gregorian",
        "response_calendar": "standard",
        "operations": ["clip_to_configured_window", "refresh_toml_endpoints"],
        "prepared_start": "2046-01-01T00:00:00",
        "prepared_end": str(time[-1]),
        "response_start": str(time[0]),
        "response_end": str(time[-1]),
        "time_label": "interval_end",
    }
    responses = Responses(
        csv={
            run: pd.DataFrame(
                {"Q_9": (np.arange(len(time)) % 31 + 1) * scale}, index=time
            )
            .to_csv(index_label="time")
            .encode()
            for run, scale in zip(RUN_IDS, (1, 2), strict=True)
        },
        toml=(
            '[time]\ncalendar="standard"\ntimestepsecs=86400\n'
            'starttime="2046-01-01T00:00:00"\nendtime="2048-12-31T00:00:00"\n'
            '[[output.csv.column]]\nheader="Q"\n'
            'parameter="river_water__volume_flow_rate"\n'
        ),
        temporal=temporal,
        variables=[
            {
                "variable": "q",
                "locations": ["9"],
                "units": "m3 s-1",
                "calendar": "standard",
                "timestep": "P1D",
                "time_label": "interval_end",
                "start": str(time[0]),
                "end": str(time[-1]),
                "missing_value": "NaN",
            }
        ],
    )
    experiment = build_v2_experiment(tmp_path, responses)
    _named(experiment, {"operations": ["indicators"], "indicator_sets": ["eflow-erfa"]})
    code, output = _simulate(experiment, "--dry-run")
    assert code == 0, output
    code, output = _simulate(experiment)
    assert code == 0, output
    export = experiment.root / "results/indicator_sets/eflow-erfa"
    dictionary = pd.read_csv(export / "indicators.csv")
    assert len(dictionary) == 70
    assert set(dictionary.set_id) == {"eflow-erfa"}
    rows = pd.read_csv(export / "q_run_indicators.csv")
    maxima = rows[rows.indicator == "q_amax1d_med"]
    assert sorted(maxima.value) == [31.0, 62.0]
    (marker,) = (experiment.root / "_engine/indicator_sets").glob("*/indicators.json")
    checked = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/inspect_indicators.py"),
            "--experiment",
            str(experiment.root),
            "--manifest",
            str(marker),
            "--recheck",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr


@pytest.mark.skipif(
    os.name != "nt",
    reason="run_project_child refuses project writers off Windows",
)
@pytest.mark.parametrize("include", [False, True])
def test_named_sets_publish_independent_immutable_sets(tmp_path, include):
    experiment = build_v2_experiment(tmp_path, include_unperturbed=include)
    root = experiment.root
    _named(experiment, SETTINGS)
    code, output = _simulate(experiment)
    assert code == 0, output
    markers = sorted((root / "_engine/indicator_sets").glob("*/indicators.json"))
    assert len(markers) == 2, markers
    manifests = [read_canonical_json(marker) for marker in markers]
    assert {m["schema_version"] for m in manifests} == {"indicator-set/6"}
    by_indicator = {m["calculations"][0]["indicator"]: m for m in manifests}
    assert set(by_indicator) == {"gwr_annual_total", "recharge_depth"}
    alias = by_indicator["recharge_depth"]["calculations"][0]
    assert alias["definition"] == "gwr_annual_total"

    def rows(manifest, scope):
        path = root / next(
            item["file"]["path"]
            for item in manifest["tables"]
            if item["scope"] == scope
        )
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    for scope in ("run", "group"):
        registered = rows(by_indicator["gwr_annual_total"], scope)
        authored = rows(by_indicator["recharge_depth"], scope)
        assert registered and len(registered) == len(authored)
        if not include:
            assert len(registered) == 1
            assert all(float(row["value"]) == 9.0 for row in registered)
        # Only the indicator key differs; values and keys match exactly.
        assert [
            {**row, "indicator": "gwr_annual_total"} for row in authored
        ] == registered


@pytest.mark.skipif(
    os.name != "nt",
    reason="run_project_child refuses project writers off Windows",
)
def test_named_sets_export_one_readable_folder_per_set(tmp_path):
    from blueearth_cst.experiment.named_export import read_current_publication

    experiment = build_v2_experiment(tmp_path)
    root = experiment.root
    _named(experiment, SETTINGS)
    code, output = _simulate(experiment)
    assert code == 0, output
    records = {
        set_id: read_current_publication(root, set_id)
        for set_id in ("default", "depth_alias", "same_numbers")
    }
    # Identical normalized requests reuse one immutable indicator set across names.
    assert (
        records["default"]["indicator_set_id"]
        == records["same_numbers"]["indicator_set_id"]
    )
    assert (
        records["default"]["indicator_set_id"]
        != records["depth_alias"]["indicator_set_id"]
    )
    for set_id in records:
        assert (root / "results/indicator_sets" / set_id / "indicators.csv").is_file()
    assert "depth_alias: results saved" in output, output
    assert "[default, same_numbers]" in output, output
    assert "1 indicator;" in output, output
    assert "expected result keys" not in output, output
    assert "calculations complete" not in output, output
    assert output.index("depth_alias: results saved") < output.index(
        "wf4 simulate_system completed"
    ), output
    assert "Updating checkpoint dependencies." not in output, output
    # A metadata-only edit refreshes the publication but not the numbers.
    alias = experiment.config.parent / "sets/depth_alias.yml"
    alias.write_text(
        alias.read_text(encoding="utf-8").replace(
            "title: Recharge depth\n", "title: Renamed\n"
        ),
        encoding="utf-8",
    )
    code, output = _simulate(experiment)
    assert code == 0, output
    refreshed = read_current_publication(root, "depth_alias")
    assert "default: results unchanged; reusing saved results" in output, output
    assert refreshed["indicator_set_id"] == records["depth_alias"]["indicator_set_id"]
    assert refreshed["publication_id"] != records["depth_alias"]["publication_id"]
    assert "Renamed" in (
        root / "results/indicator_sets/depth_alias/indicators.csv"
    ).read_text(encoding="utf-8")
