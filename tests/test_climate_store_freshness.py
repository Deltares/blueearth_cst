"""R07 B1 / ext2-01: the data catalog is the store's freshness boundary.

The producer declares exactly one input in both DAGs — the ``project.catalog``
catalog, plain (never ``ancient()``). Two properties follow, and this module
pins both against a real Snakemake invocation:

1. **Freshness.** Editing the catalog in place schedules ``extract_climate_datasets``
   **exactly once**. Pre-R07 the catalog rode only as a ``params`` path string,
   so an in-place edit re-triggered nothing — a staleness gap that predates R07.
2. **No oscillation.** Once the store is fresh again, ``--dry-run`` schedules
   nothing in **either** workflow. The ext1-02 oscillation needed *asymmetric*
   input sets; the symmetric singleton cannot reproduce it, and this is the
   empirical check on that claim.

Mechanics: the store outputs are fabricated and provenance is seeded with
``snakemake --touch`` (which records mtimes, not the full code/params metadata),
then every check is edit-then-``--dry-run``. Executing the rule for real is not
an option in a unit suite — it would fetch era5. The property under test is the
input **edge**, which the mtime trigger exercises directly.

Data behind an unchanged catalog entry is deliberately OUT of scope (see
``dev/records/milestones/r07/migration_project-layout.md`` §2f); the escape hatch is
``snakemake --forcerun extract_climate_datasets``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from blueearth_cst.shared.config_composition import load_composed_config  # noqa: E402
from tests.conftest import phase_env, write_config  # noqa: E402

pytestmark = [
    pytest.mark.workflow_contract,
    pytest.mark.usefixtures("isolated_snakemake"),
]

TESTDIR = Path(__file__).resolve().parent
SNAKEDIR = TESTDIR.parent
CONFIG_FN = TESTDIR / "project_config_fixture.yml"
CATALOG_FN = TESTDIR / "data" / "tests_data_catalog.yml"

SNAKEFILES = ("build_model.smk", "generate_scenarios.smk")

_JOB_COUNT_RE = re.compile(r"^extract_climate_datasets\s+(\d+)\s*$", re.MULTILINE)


@pytest.fixture()
def staged_store(tmp_path):
    """A project_dir with a fabricated store and a tmp-owned catalog copy.

    The catalog is copied so the test can edit it without touching the tracked
    fixture. Returns (config_path, catalog_path, store_target).
    """
    cfg = load_composed_config(CONFIG_FN)
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    catalog = tmp_path / "catalog.yml"
    shutil.copy(CATALOG_FN, catalog)

    cfg["project"]["project_dir"] = project_dir.as_posix()
    cfg["project"]["catalog"] = catalog.as_posix()
    cfg_path = write_config(tmp_path, cfg, stem="project_config_staged")

    # Store key exactly as climate_store_rule builds it.

    from blueearth_cst.shared.snake_utils import climate_store_rule

    spec = climate_store_rule(
        project_dir=project_dir.as_posix(),
        model_region=cfg["basin"]["region"],
        clim_source=cfg["climate"]["selected"],
        historical_window=cfg["climate"]["window"],
        data_sources=catalog.as_posix(),
    )
    store = Path(spec.store_dir)
    store.mkdir(parents=True)
    for path in spec.outputs.values():
        Path(path).write_bytes(b"")
    return cfg_path, catalog, Path(spec.outputs["climate_nc"])


def _run(args, snakefile, cfg_path):
    """Invoke snakemake with the repo workflow profile disabled (it hides reasons)."""
    cmd = (
        f"snakemake {args} --workflow-profile none -c 1 "
        f'-s {snakefile} --configfile "{cfg_path}"'
    )
    return subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=str(SNAKEDIR),
        env=phase_env(snakefile),
    )


def _scheduled_count(snakefile, cfg_path, target):
    """How many ``extract_climate_datasets`` jobs a dry-run would run (0 when clean).

    A missing job-stats row is read as 0, so every ``== 0`` assertion below
    pairs this with an explicit up-to-date check (Snakemake's "Nothing to be done",
    restated by `console_style` as "Up to date") — otherwise a
    changed job-stats format would silently turn the no-oscillation assertions
    into no-ops.
    """
    result = _run(f'--dry-run "{target.as_posix()}"', snakefile, cfg_path)
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, combined[-3000:]
    match = _JOB_COUNT_RE.search(combined)
    return (int(match.group(1)) if match else 0), combined


def _assert_clean(snakefile, cfg_path, target, why):
    """Assert a dry-run schedules nothing, on positive evidence not absence."""
    count, out = _scheduled_count(snakefile, cfg_path, target)
    assert count == 0, f"{snakefile} {why}\n{out}"
    assert "Up to date" in out, f"{snakefile} {why}\n{out[-2000:]}"


def _seed_provenance(cfg_path, target):
    for snakefile in SNAKEFILES:
        result = _run(f'--touch "{target.as_posix()}"', snakefile, cfg_path)
        assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")


def test_the_catalog_is_the_one_freshness_edge_in_both_workflows(staged_store):
    """Both properties, in one ordered sequence over one staged store.

    A ``--dry-run`` changes no workflow state, so every dry-run below observes
    the same tree the previous one did. That is what lets a single pass cover
    the ext1-02 regression check -- wf1 then wf3 on an unchanged store schedule
    nothing -- which an earlier version ran twice over a second store for an
    identical observation (t2610041227 cut it from 16 Snakemake runs to 10).
    """
    cfg_path, catalog, target = staged_store
    _seed_provenance(cfg_path, target)

    # No oscillation: a fresh store is clean in each DAG, in turn.
    for snakefile in SNAKEFILES:
        _assert_clean(snakefile, cfg_path, target, "re-scheduled the producer")

    # Freshness: an in-place catalog edit is the supported signal for a data
    # change, and it schedules the producer exactly once in each DAG.
    catalog.write_text(
        catalog.read_text(encoding="utf-8") + "\n# freshness probe\n", encoding="utf-8"
    )
    for snakefile in SNAKEFILES:
        count, out = _scheduled_count(snakefile, cfg_path, target)
        assert count == 1, (
            f"{snakefile} scheduled {count} extract_climate_datasets job(s) after a "
            f"catalog edit; expected exactly 1\n{out}"
        )
        assert "Updated input files" in out, out[-2000:]

    # Store re-made -> both DAGs clean again. No alternation re-fires it.
    _seed_provenance(cfg_path, target)
    for snakefile in SNAKEFILES:
        _assert_clean(
            snakefile,
            cfg_path,
            target,
            "still schedules the producer after the store was remade — the "
            "cross-DAG oscillation is back",
        )
