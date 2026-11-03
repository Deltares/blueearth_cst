"""The v2 project config shape parses, composes and builds a DAG.

P1's acceptance surface (R14 `config-shape-p1-loader.task.md`). The fixture is
`tests/data/v2/project_config_v2_probe.yml`, hand-written rather than migrated:
P4 owns `tests/project_config_fixture.yml` and migrates it WITH the rewriter, so
a P1 test that edited it would be doing P4's job with a worse tool.

**This module is P1b's acceptance surface as well as P1's.** P1 moved the T1
reads — the `project:`, `basin:`, `climate:` and `model:` sections, which is
what `shared:` dissolved into — and two entry points dry-ran clean. P1b moved
the T2 reads, the per-workflow key renames, one workflow per commit: WF1
(`C-22`, `C-56`), WF2 (`C-25`, `C-59`, `C-60`, `C-63`) and WF3 (`C-68`, `C-31`,
`C-32`, `C-33`, `C-67`, `C-69`, `C-29`, `C-34`). All four are in ``V2_CLEAN``
now, so the ``V2_BLOCKED`` boundary marker this module carried is gone.

Still owed, and NOT by P1b: `C-57` and `C-64` need a per-variable registry that
does not exist, and `C-66` cannot dissolve `relative_change:` without it. Those
are P1c. The long form of `variables:` is not retired, so the fixture uses it.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from blueearth_cst.shared.config_composition import load_composed_config

pytestmark = pytest.mark.usefixtures("isolated_snakemake")


REPO_ROOT = Path(__file__).resolve().parent.parent
V2_CONFIG = REPO_ROOT / "tests" / "data" / "v2" / "project_config_v2_probe.yml"

#: The entry points that dry-run clean on v2. P1 delivered the first two by
#: moving the T1 reads; P1b added `analyze_projections` by moving WF2's T2 key
#: readers (`C-25`, `C-59`, `C-60`, `C-63`).
V2_CLEAN = (
    "analyze_climate",
    "build_model",
    "analyze_projections",
    "generate_scenarios",
)


def _dry_run(snakefile: str) -> subprocess.CompletedProcess[str]:
    """Dry-run one entry point on the v2 config; return the completed process.

    ``generate_scenarios.smk`` refuses a bare dry-run since P5 (`c83ead70`):
    it requires ``CST_GENERATION_PHASE`` to name which of the launcher's two
    owned phases is asking, and rejects anything else with `"WF3 requires the
    owned source or generation phase"` (pinned by
    ``tests/test_cli.py::test_snakefile_cli_generate_scenarios``). ``source``
    is enough to reach a clean DAG here, mirroring
    ``test_log_rules_contract.py``'s ``_parse_workflow``.
    """
    env = {**os.environ}
    if snakefile == "generate_scenarios":
        env["CST_GENERATION_PHASE"] = "source"
    return subprocess.run(
        f"snakemake all -c 1 -s {snakefile}.smk --configfile {V2_CONFIG} --dry-run",
        shell=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )


def test_the_v2_fixture_composes():
    """The shape parses and composes before any DAG is asked for.

    Cheapest of the three claims, and the one whose failure is least ambiguous:
    a composition failure is the loader, a DAG failure is a rule.
    """
    composed = load_composed_config(V2_CONFIG)
    assert composed["schema_version"] == 2
    assert set(composed) == {
        "schema_version",
        "project",
        "basin",
        "climate",
        "model",
        "workflows",
    }
    assert "shared" not in composed
    assert composed["climate"]["selected"] == "era5"
    assert composed["climate"]["window"] == {"start": 2000, "end": 2020}


@pytest.mark.workflow_contract
@pytest.mark.parametrize("snakefile", V2_CLEAN)
def test_an_entry_point_dry_runs_clean_on_v2(snakefile):
    """P1's acceptance criterion, for the entry points it reaches."""
    result = _dry_run(snakefile)
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, combined[-3000:]
    assert "Building DAG of jobs" in combined, combined[-3000:]
    assert "Error in file" not in combined, combined[-3000:]


def test_every_declared_t2_file_exists():
    """D-7.9: the file and its ``config_path`` stanza travel together.

    An empty T2 file is legal and a MISSING one with ``config_path`` still
    declared is a hard parse error, so a fixture that lost a sibling would fail
    with a refusal about the fixture rather than about the shape under test.
    """
    directory = V2_CONFIG.parent
    import yaml

    t1 = yaml.safe_load(V2_CONFIG.read_text(encoding="utf-8"))
    for name, stanza in t1["workflows"].items():
        declared = stanza.get("config_path")
        assert declared, f"workflows.{name} declares no config_path"
        assert (directory / declared).is_file(), (
            f"workflows.{name}.config_path names {declared!r}, which is not "
            f"beside {V2_CONFIG.name} — a relative config_path anchors at the "
            "project file's own directory, not the working directory"
        )
        assert os.path.basename(declared) == declared, (
            "the fixture's config_path values are bare sibling filenames, so "
            "the set stays relocatable"
        )


def test_reference_window_takes_no_water_year_offset(tmp_path):
    """`C-74` / D-7.4: WF2's reference window is CALENDAR years.

    `hydrological_year_bounds()` trims to complete water years one layer down,
    so a `reference_window` routed through the water-year path would have the
    offset applied twice and every change factor would move. Nothing else
    catches that: `check_baseline` is out of scope for the phase that wrote
    this reader, and the probe fixture declares `water_year_start: Jan`, under
    which a double offset is exactly zero and therefore invisible.

    So the assertion is a COMPARISON against a non-Jan water year, not a value.
    """
    import shutil

    import yaml

    from blueearth_cst.projections import reference_window as _rw
    from blueearth_cst.shared.snake_utils import window_year_pair

    def _effective(water_year_start):
        for source in V2_CONFIG.parent.glob("project_config_v2_probe*.yml"):
            shutil.copy(source, tmp_path / source.name)
        target = tmp_path / V2_CONFIG.name
        doc = yaml.safe_load(target.read_text(encoding="utf-8"))
        doc["climate"]["water_year_start"] = water_year_start
        target.write_text(yaml.safe_dump(doc), encoding="utf-8")
        composed = load_composed_config(target)
        window = composed["workflows"]["analyze_projections"]["reference_window"]
        pair = window_year_pair(window, "reference_window")
        return _rw.clip_reference_window(pair).effective

    assert _effective("Jan") == _effective("Oct")


def test_n_levels_is_step_num_plus_one():
    """`C-31` is a RETYPE, and this is the assertion that says which one.

    The brief's falsifier: a config with ``step_num: 1`` must produce the SAME
    grid as one with ``n_levels: 2``. Asserting the MEMBER COUNT rather than the
    key is the whole point — a migration that renamed the key without adding one
    would pass any spelling check and silently drop a level from every axis,
    halving the stress-test grid.

    ``st_num`` is the product of the two axes, so an off-by-one on either shows
    up here: 2x2 = 4 members, where the v1 ``step_num: 1`` pair gave (1+1)x(1+1).
    """
    from blueearth_cst.shared.snake_utils import stress_test_grid

    v1_equivalent = {
        "temp": {"n_levels": 1 + 1, "trajectory": "transient", "mean": {}},
        "precip": {"n_levels": 1 + 1, "trajectory": "transient", "mean": {}},
    }
    assert stress_test_grid(v1_equivalent) == (2, 2, 4)

    # And the retired spelling is refused rather than read as a level count,
    # which is what stops a bare rename from being accepted.
    with pytest.raises(ValueError, match="n_levels"):
        stress_test_grid(
            {
                "temp": {"step_num": 1, "transient_change": True, "mean": {}},
                "precip": {"step_num": 1, "transient_change": True, "mean": {}},
            }
        )
