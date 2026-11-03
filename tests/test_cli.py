"""Test some snake command line interface (CLI) for validity of snakefiles."""

import json
import os
import subprocess
import sys
from os.path import dirname, join, realpath
from pathlib import Path

import pytest
import yaml

TESTDIR = dirname(realpath(__file__))
SNAKEDIR = join(TESTDIR, "..")

sys.path.insert(0, join(SNAKEDIR, "dev", "scripts"))
import cross_workflow_inputs as cwi  # noqa: E402

from blueearth_cst.shared.config_composition import load_composed_config  # noqa: E402
from tests.conftest import write_config  # noqa: E402

pytestmark = pytest.mark.usefixtures("isolated_snakemake")


config_fn = join(TESTDIR, "project_config_fixture.yml")
linux_config_fn = join(SNAKEDIR, "test_case", "project_config_baseline_linux.yml")


def _dry_run(snakefile, cfg=config_fn, env=None):
    """Dry-run a Snakefile on a config; return the completed process.

    stdout/stderr are captured as text so callers can match on the DAG-build
    exception class name. Snakemake writes these diagnostics to stderr, but we
    match on the combined stream so a stream change does not silently break a
    ratchet assertion below.
    """
    cmd = f"snakemake all -c 1 -s {snakefile} --configfile {cfg} --dry-run"
    return subprocess.run(
        cmd, shell=True, capture_output=True, text=True, env=env, cwd=SNAKEDIR
    )


@pytest.fixture()
def config_with_staged_region(tmp_path):
    """Config whose project_dir is a temp dir pre-staged with the wf1 leaves.

    Historically analyze_projections declared `{project_dir}/hydrology_model/
    staticgeoms/region.geojson` as an `ancient(...)` input produced by
    build_model — a cross-workflow contract Snakemake will not satisfy on its
    own — so the fixture staged a minimal valid region file under a **test-owned
    tmp project_dir** (never the tracked baseline dir). Since ADR 0003 the extent
    is model-free and BOTH downstream workflows delineate their own
    `data/spatial/geoms/region.geojson`, declaring it as an OUTPUT, so the staged
    region is no longer load-bearing for either dry-run; it is retained only
    because removing it belongs with the wider staging consolidation (R9 P5 F3).
    The staged path follows the R9 model root. tmp_path is torn down by pytest.

    Since P3-1 commit 1, run_stress_test's drift guard (rule
    check_project_consistency) additionally declares the wf1 config snapshot
    `{project_dir}/config/build_model/composed_config.yml` as a mandatory
    `ancient(...)` input — the same class of cross-workflow contract, staged
    the same way. The staged snapshot is serialized from the SAME parsed
    config the dry-run consumes, so the guard's comparands match by
    construction.
    """
    cfg = load_composed_config(config_fn)
    cfg["project"]["project_dir"] = str(tmp_path).replace("\\", "/")

    # ONE definition of the leaf set, shared with `test_guard_invalidation` and
    # `scaffold_project_tree` and proved complete-and-minimal against the real
    # DAG by `tests/test_cross_workflow_inputs.py`. It was three hand-kept
    # copies until R9 P5 F3; R9 P4's rule 3.01c added the two model leaves, two
    # of the three were updated, and the third went red looking like a defect
    # in the thing it tested. EXTRA_REGION is a deliberate non-leaf, kept for
    # the reason given in the docstring above.
    cwi.stage(tmp_path, yaml.safe_dump(cfg), extras=(cwi.EXTRA_REGION,))

    cfg_path = write_config(tmp_path, cfg, stem="project_config_staged")
    return cfg_path


def _job_counts(result):
    """Parse Snakemake's `Job stats:` table into {rule: count}.

    The table is alphabetical and says WHAT will run, never in what order --
    which is exactly what a set comparison wants.
    """
    counts = {}
    lines = (result.stdout or "").splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("job") and "count" in line:
            for row in lines[i + 2 :]:
                parts = row.split()
                if len(parts) != 2 or not parts[1].isdigit():
                    break
                counts[parts[0]] = int(parts[1])
            break
    return counts


@pytest.mark.workflow_contract
def test_analyze_climate_adds_no_job_to_build_model(tmp_path):
    """THE ADDITIVITY CLAIM, checked rather than asserted in a design doc.

    The fourth workflow was designed to be purely ADDITIVE: it declares the
    same shared producer rules WF1 already declares, so WF1 keeps every rule it
    had and gains no edge. If that is true, WF1's planned job set against a
    fresh project_dir is identical whether or not `analyze_climate` exists --
    and since the two are separate files, the only way it could change is
    through the config, which is the thing this checks.

    A regression here means the carve stopped being additive and WF1 has
    started depending on WF0 -- the exact failure mode that would make
    `analyze_climate.enabled: false` break the model build.
    """
    cfg = load_composed_config(config_fn)
    cfg["project"]["project_dir"] = (tmp_path / "proj").as_posix()

    with_wf0 = write_config(tmp_path / "with_wf0", cfg)

    # The same config with the fourth workflow's section removed entirely --
    # the state every project config was in before 2026-08-14.
    without = load_composed_config(config_fn)
    without["project"]["project_dir"] = (tmp_path / "proj").as_posix()
    without["workflows"].pop("analyze_climate")
    without_wf0 = write_config(tmp_path / "without_wf0", without)

    a = _dry_run("build_model.smk", cfg=str(with_wf0))
    b = _dry_run("build_model.smk", cfg=str(without_wf0))
    assert a.returncode == 0, (a.stdout or "") + (a.stderr or "")
    assert b.returncode == 0, (b.stdout or "") + (b.stderr or "")

    counts = _job_counts(a)
    assert counts, "could not parse a Job stats table from the WF1 dry-run"
    assert counts == _job_counts(b)


@pytest.mark.workflow_contract
def test_snakefile_cli_analyze_climate():
    """Workflow 0 dry-run builds a clean DAG on the test config.

    NO staged cross-workflow leaves, deliberately, and that is the assertion:
    this workflow is model-free, so unlike WF2 and WF3 it needs nothing on disk
    that it does not itself produce. If it ever gained a leaf, the fixture
    config would stop resolving here rather than at someone's first real run.
    """
    result = _dry_run("analyze_climate.smk")
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")


@pytest.mark.workflow_contract
def test_snakefile_cli_build_model():
    """Workflow 1 dry-run builds a clean DAG on the test config."""
    result = _dry_run("build_model.smk")
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")


@pytest.mark.workflow_contract
def test_snakefile_cli_build_model_linux_config(tmp_path):
    """The Linux config must still build a DAG after O-01 retired `data/`.

    R07 deletes the tracked `data/` tree, whose only live consumers were this
    config and the Docker runner. Linux *end-to-end* validation stays parked
    (no Linux machine), but parse-level consistency is cheap and is exactly
    what a silently-broken config would fail: DAG build resolves every
    config-declared path, so a dangling `gauge_points` would surface here.
    Runs on both CI legs.
    """
    # Keep all Linux workflow/settings/path bindings; qualify the shared map
    # producer against a real tiny local DEM rather than requiring the P-drive
    # production mirror on both CI platforms.
    cfg = load_composed_config(linux_config_fn)
    cfg["project"]["catalog"] = (
        Path(SNAKEDIR, "tests/data/tests_data_catalog.yml").resolve().as_posix()
    )
    staged = write_config(tmp_path, cfg, stem="project_config_linux_parse")
    result = _dry_run("build_model.smk", cfg=str(staged))
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")


def test_the_linux_baseline_set_is_a_twin_of_the_windows_one():
    """One `baseline` name, one experiment — the ratchet for `t2608241414`.

    The two sets had drifted on NINE keys across four files while the Linux
    project file's header claimed they were the same basin: the resolution
    (0.0062475 against 0.00833), the historical window (21 years against 17),
    the emitted variables, the output locations and observed series (unset
    against the tracked CSVs), the wf1 engine block (absent), one of the three
    CMIP6 models, the reference window, the future horizons (near+far against
    far), and the WF3 simulation window — which sat inside the Linux set's
    `near` horizon while the Windows one sat inside its `far`.

    Nothing caught it because nothing compared them. End-to-end validation on
    Linux is deferred, so this set's whole obligation is to parse cleanly and to
    stay a twin, and any cross-platform reading of baseline numbers is unsound
    the moment it is not one.

    Compared as PARSED documents, not as text, so a comment or a key reordering
    is free while a value change is not.
    """
    import copy

    def _load(name):
        path = Path(SNAKEDIR, "test_case", name)
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    # The three keys that MUST differ, and the only ones. `project_dir` and
    # `catalog` are the reason the Linux set exists; `config_path` names the
    # sibling files, which are different files by construction.
    windows = _load("project_config_baseline.yml")
    linux = _load("project_config_baseline_linux.yml")
    for doc in (windows, linux):
        doc["project"].pop("project_dir")
        doc["project"].pop("catalog")
        for stanza in doc["workflows"].values():
            stanza.pop("config_path", None)
    assert linux == windows, (
        "project_config_baseline_linux.yml has drifted from its twin; only "
        "project.project_dir, project.catalog and the config_path names may "
        "differ"
    )

    for workflow in (
        "build_model",
        "analyze_projections",
        "generate_scenarios",
        "simulate_system",
    ):
        a = _load(f"project_config_baseline_{workflow}.yml")
        b = _load(f"project_config_baseline_linux_{workflow}.yml")
        assert b == copy.deepcopy(a), (
            f"project_config_baseline_linux_{workflow}.yml has drifted from "
            f"project_config_baseline_{workflow}.yml; the two baseline sets "
            f"describe one experiment and differ only in the project file's "
            f"paths"
        )


@pytest.mark.workflow_contract
def test_in_repo_project_dir_warning_reaches_the_stream(tmp_path):
    """O-22 end to end: the parse-time warning is actually surfaced.

    The unit cases in test_snake_utils.py pin the decision; this pins that a
    real `snakemake` invocation shows it, which is the only thing a user sees.
    project_dir points at an in-repo scratch dir (NOT test_case/, the one
    exemption), so the warning must fire -- and the run must still succeed,
    because O-22 warns and never raises.
    """
    scratch = Path(SNAKEDIR, "_o22_probe_project")
    scratch.mkdir(exist_ok=True)
    try:
        cfg = load_composed_config(config_fn)
        cfg["project"]["project_dir"] = scratch.as_posix()
        cfg_path = write_config(tmp_path, cfg, stem="project_config_in_repo")

        result = _dry_run("build_model.smk", cfg=str(cfg_path))
        combined = (result.stdout or "") + (result.stderr or "")
        assert "inside the repo tree" in combined, combined[-3000:]
        assert result.returncode == 0, combined[-3000:]
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
        scratch.rmdir()


@pytest.mark.workflow_contract
def test_baseline_seed_config_does_not_warn(tmp_path):
    """The exemption holds for the config the baseline gate actually runs.

    That is test_case/project_config_baseline.yml (project_dir:
    test_case/test_local) -- NOT tests/project_config_fixture.yml, which
    points at tests/test_project and therefore warns correctly: it is an
    in-repo project_dir outside the single exemption. The exemption exists
    because the baseline seed config is TRACKED and a tracked config cannot
    carry a machine-specific absolute path; it does not extend to every
    convenient in-repo scratch dir.
    """
    seed_cfg = join(SNAKEDIR, "test_case", "project_config_baseline.yml")
    cfg = load_composed_config(seed_cfg)
    cfg["project"]["project_dir"] = Path(SNAKEDIR, "test_case/test_local").as_posix()
    # Keep the baseline project_dir and gauge request, but isolate the warning
    # contract from the P-drive data mirror unavailable on both CI platforms.
    cfg["project"]["catalog"] = (
        Path(SNAKEDIR, load_composed_config(config_fn)["project"]["catalog"])
        .resolve()
        .as_posix()
    )
    hermetic_cfg = write_config(tmp_path, cfg, stem="project_config_baseline")
    result = _dry_run("build_model.smk", cfg=str(hermetic_cfg))
    combined = (result.stdout or "") + (result.stderr or "")
    assert "inside the repo tree" not in combined, combined[-3000:]

    # The exemption above is what this test is named for, and it holds in every
    # checkout. BUILDING the DAG is a different matter, and it is NOT a property
    # of the config alone: the seed config's `gauge_points` CSV lives under the
    # untracked `test_case/test_data/` (`.gitignore` line `test_case/*`), so the
    # dry-run resolves only where that file exists OR where the outputs it feeds
    # are already on disk and no job needs it. A bare `returncode == 0` asserted
    # neither, and both CI legs failed on it from 2026-08-10 to 2026-08-12 with a
    # MissingInputException that says nothing about the exemption.
    #
    # So classify the failure rather than predict it. Skipping was the obvious
    # alternative and is worse: it makes "the gauge CSV is untracked here"
    # indistinguishable from "the DAG is broken" -- the skip-on-absence false
    # green this repo has been bitten by before (R9-4, and the fixture layer
    # AGENTS.md describes). Deriving the path from the config rather than
    # spelling it is the same lesson the store-key literal in
    # test_interchange_contracts.py taught on 2026-08-12.
    if result.returncode != 0:
        with open(seed_cfg) as f:
            gauge_points = yaml.safe_load(f)["basin"]["output_locations"]
        norm = combined.replace("\\", "/")
        assert "MissingInputException" in norm, combined[-3000:]
        assert gauge_points in norm, combined[-3000:]
        assert not os.path.exists(join(SNAKEDIR, gauge_points)), combined[-3000:]


def test_observation_configs_use_yaml_null():
    """Every shipped config spells "not provided" as a real YAML null.

    This reverses an earlier ratchet that pinned the STRING "None". Unquoted
    `None` parses to the Python string, not to null -- it reads as a null
    without being one, and that gap is what produced the `gauges_None`
    layer-name bug (O-08). ec92ae6 converted all four config/workflows/*.yml in
    one sweep; this finishes the job and pins the direction.

    The legacy spelling stays TOLERATED -- see
    `test_both_sentinel_spellings_are_treated_as_unset` -- because project
    configs in the wild still carry it. Tolerated on the way in, not emitted on
    the way out.
    """
    # Checked over EVERY shipped seed plus the fixture, and as a SPELLING rather
    # than as an absence. It asserted `is None` on two hand-picked configs until
    # `t2608241414` gave the Linux set a real observations path — at which point
    # a test whose docstring is about how "not provided" is written started
    # failing because something WAS provided. The property survives a config
    # gaining a value; the earlier phrasing did not.
    # The glob also matches the per-workflow files beside each project file, and
    # those are not composable entry points — `load_composed_config` refuses one
    # for declaring no `schema_version`. Selected by CONTENT rather than by a
    # name pattern, so a seed added with a different suffix is still covered.
    shipped = [
        path
        for path in sorted(Path(SNAKEDIR, "test_case").glob("project_config_*.yml"))
        if "workflows:" in path.read_text(encoding="utf-8")
    ]
    assert shipped, "no shipped seed configs found; this test would pass vacuously"
    for cfg_path in [Path(config_fn), *shipped]:
        cfg = load_composed_config(cfg_path)
        mc = cfg["workflows"]["build_model"]
        values = {
            "basin.output_locations": cfg["basin"]["output_locations"],
            # `C-56`: a mapping keyed by variable. The property is unchanged --
            # an unset observation is YAML null, not the string "None" -- but
            # it now lives one level down, under the outvar it belongs to.
            "workflows.build_model.observations[river discharge]": (
                (mc.get("observations") or {}).get("river discharge")
            ),
        }
        for key, value in values.items():
            assert value is None or (
                isinstance(value, str) and value.strip().lower() != "none"
            ), (
                f"{cfg_path}:{key} is {value!r}; shipped configs spell "
                f"'not provided' as YAML null, never as the string 'None'"
            )


def test_both_sentinel_spellings_are_treated_as_unset():
    """The guards must accept `null` AND the legacy string, identically.

    This is what makes the migration above safe rather than merely tidy: an
    existing project config saying `output_locations: None` has to keep
    working. Each consumer is checked against the predicate it actually uses --
    plot_map derives a layer NAME (an explicit string check, O-08), while the
    other two guard on file existence.
    """
    from blueearth_cst.shared.gauges import gauges_layer_name

    geoms = {"basins", "outlets", "gauges_my-stations"}
    for unset in (None, "None"):
        assert gauges_layer_name(geoms, unset) is None, unset
        # The existence-based guards: neither spelling names a real file, so
        # both take the skip branch. `os.path.isfile(None)` would raise, which
        # is why the `is not None` half has to come first in those callers.
        assert not (unset is not None and os.path.isfile(unset)), unset

    # And a real path is still recognised, so the assertions above are not just
    # "everything is falsy".
    # And a configured file still resolves — hydromt spells it with a HYPHEN.
    assert gauges_layer_name(geoms, "gauges/my_stations.csv") == "gauges_my-stations"


@pytest.mark.workflow_contract
def test_eobs_config_fails_wf1_dry_run_at_parse_time(tmp_path):
    """`clim_historical: eobs` must red the wf1 dry-run at DAG-parse time.

    Rehomed from the retired `tests/test_extract_climate_wf1.py` (R07 commit 7
    retired rule 1.10's wf1-only wrapper, but NOT this guard): the rejection
    exists because rule 1.11's model-parity transform maps eobs to a different
    PET method, which B1 does not touch. The other test in that module compared
    the two pre-R07 bbox derivations and is superseded by
    `tests/test_store_region_bbox.py`.
    """
    cfg = load_composed_config(config_fn)
    # `C-43`: `selected` names a MEMBER of `sources`, and the loader refuses
    # one that is not. Switching the source therefore means declaring it as a
    # candidate as well -- otherwise this test reds on the membership rule
    # rather than on the eobs rejection it exists to check.
    cfg["climate"]["selected"] = "eobs"
    cfg["climate"]["sources"] = ["eobs"]
    cfg_path = write_config(tmp_path, cfg, stem="project_config_eobs")

    result = _dry_run("build_model.smk", cfg=str(cfg_path))
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode != 0, "eobs config must fail the wf1 dry-run"
    assert (
        "clim_historical: eobs is not supported by the P3-2a wf1 raw-climate "
        "path; supported sources: era5, chirps, chirps_global"
    ) in combined, combined


@pytest.mark.parametrize(
    "end_year, label",
    [
        # `C-70` retyped the window to INCLUSIVE YEARS, so the old sub-year
        # case is no longer expressible -- the shortest window a v2 config can
        # state is one year, which is what this now exercises.
        (2000, "one-year"),
        # The case the UNIFIED floor added: WF1 used to build a model happily on
        # ten years and let WF3 discover the problem inside weathergenr.
        (2009, "ten-year"),
    ],
)
@pytest.mark.workflow_contract
def test_short_window_fails_wf1_dry_run_at_parse_time(tmp_path, end_year, label):
    """A `climate.window` under MIN_HISTORICAL_YEARS must red the dry-run.

    Same parse-time stance, and same test shape, as the eobs rejection above:
    no execution can rescue the config, so the earliest failure is the most
    legible one. Pre-guard, a sub-year window reached rule 1.11 and died with
    MissingOutputException nine rules and one hydromt build past the cause
    (pre-board record 45e3a564 R7-6), and a ten-year window ran WF1 to
    completion before failing a whole workflow away.
    """
    cfg = load_composed_config(config_fn)
    cfg["climate"]["window"] = {"start": 2000, "end": end_year}
    cfg_path = write_config(tmp_path, cfg, stem=f"project_config_{label}")

    result = _dry_run("build_model.smk", cfg=str(cfg_path))
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode != 0, f"{label} window must fail the wf1 dry-run"
    assert "16-year minimum" in combined, combined[-3000:]
    # The message must be actionable without opening the Snakefile.
    assert "weathergenr" in combined, combined[-3000:]


@pytest.mark.workflow_contract
def test_snakefile_cli_analyze_projections(config_with_staged_region):
    """Workflow 2 dry-run builds a clean DAG once its WF1 region input is staged.

    analyze_projections declares region.geojson (a build_model output) as an
    `ancient(...)` input Snakemake will not build itself — correct behavior. R3
    stages it in a test-owned tmp project_dir (see the fixture) rather than
    weakening the contract; workflow 2's Snakefile is untouched (R4 territory).
    Was a MissingInputException ratchet pre-R3 (dev/tasks/).
    """
    result = _dry_run("analyze_projections.smk", cfg=config_with_staged_region)
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")


@pytest.mark.workflow_contract
def test_wf2_dag_accepts_shared_newest_store_pins(config_with_staged_region, tmp_path):
    """Older tas revisions do not block the fetcher's shared newest store."""
    cfg = load_composed_config(config_with_staged_region)
    wf2 = cfg["workflows"]["analyze_projections"]
    wf2["models"] = ["CAS/FGOALS-g3", "CCCR-IITM/IITM-ESM"]
    wf2["scenarios"] = ["ssp126"]
    wf2["members"] = {"preference": ["r1i1p1f1"]}
    config_path = write_config(tmp_path / "shared_newest", cfg)
    result = _dry_run("analyze_projections.smk", cfg=config_path)
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")
    counts = _job_counts(result)
    assert counts["fetch_cmip6_projections"] == 4
    assert counts["reduce_to_basin_averages"] == 4


@pytest.mark.workflow_contract
@pytest.mark.parametrize("only_ciesm", [False, True])
def test_wf2_dag_excludes_ciesm_before_acquisition(
    config_with_staged_region, tmp_path, only_ciesm
):
    cfg = load_composed_config(config_with_staged_region)
    wf2 = cfg["workflows"]["analyze_projections"]
    wf2["models"] = ["THU/CIESM"] + ([] if only_ciesm else ["NOAA-GFDL/GFDL-ESM4"])
    wf2["scenarios"] = ["ssp245"]
    wf2["members"] = {
        "preference": ["r1i1p1f1"],
        "overrides": {"THU/CIESM": ["r1i1p1f1"]},
    }
    config_path = write_config(tmp_path / "ciesm_exclusion", cfg)
    result = _dry_run("analyze_projections.smk", cfg=config_path)
    if only_ciesm:
        assert result.returncode != 0
        assert "source_quality_excluded" in result.stdout + result.stderr
        assert "all variables" in result.stdout + result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        counts = _job_counts(result)
        # One valid model, its historical series and one SSP; no CIESM fetch/reduce.
        assert counts["fetch_cmip6_projections"] == 2
        assert counts["reduce_to_basin_averages"] == 2
        stamp = json.loads(
            (
                Path(cfg["project"]["project_dir"])
                / "data/climate/projections/cmip6/_engine/stage_b_settings.json"
            ).read_text(encoding="utf-8")
        )
        assert len(stamp["source_quality_exclusions"]) == 1
        assert stamp["source_quality_exclusions"][0]["dataset"] == "THU/CIESM"
        assert all("CIESM" not in key for key in stamp["points"])


def test_analyze_projections_owns_its_region():
    """Pin WF2's region contract to what it actually is.

    This guard used to assert that `staticgeoms/region.geojson` appears in
    analyze_projections.smk, standing in for a wf1 -> wf2 cross-workflow
    input. That input is gone: since ADR 0003 the extent is model-free and WF2
    delineates its OWN `data/spatial/geoms/region.geojson`. The literal string
    still occurred in the file — in a comment recording that WF2 was *freed*
    from it — so the assertion kept passing while guarding nothing, found by the
    R9 P5 stale-path sweep.

    What is worth pinning is the current shape: WF2 produces the model-free
    region, and reads nothing from the wf1 model root.
    """
    text = Path(SNAKEDIR, "analyze_projections.smk").read_text()
    assert "data/spatial/geoms/region.geojson" not in text, (
        "the region path belongs in snake_utils.region_rule, not inline here"
    )
    assert "region_rule(" in text and "REGION.region_geojson" in text
    # The model root may only appear as history, never as a declared dependency.
    for line in text.splitlines():
        if "hydrology_model/" in line or "models/hydrology/wflow" in line:
            assert line.lstrip().startswith("#"), f"WF2 reads the model root: {line}"


@pytest.mark.workflow_contract
def test_snakefile_cli_generate_scenarios(config_with_staged_region):
    """WF3 requires ownership, then builds the source-only cold DAG."""
    refused = _dry_run("generate_scenarios.smk", cfg=config_with_staged_region)
    assert refused.returncode != 0
    assert "owned source or generation phase" in refused.stdout + refused.stderr
    result = _dry_run(
        "generate_scenarios.smk",
        cfg=config_with_staged_region,
        env={
            **os.environ,
            "CST_GENERATION_PHASE": "source",
            "CST_GENERATION_OWNED": "test",
        },
    )
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")


@pytest.mark.workflow_contract
def test_simulation_requires_validated_runner(config_with_staged_region):
    result = _dry_run("simulate_system.smk", cfg=config_with_staged_region)
    assert result.returncode != 0
    assert "scripts/simulate_system.py" in result.stdout + result.stderr


#: Every file that writes a run banner to `sys.stderr` under a broad `except`.
#: The wrapper joined the four entry points on 2026-08-17 -- it had the same two
#: sites and the same defect, and being a `.py` file made it no less exposed.
BANNER_SOURCES = (
    "analyze_climate.smk",
    "analyze_projections.smk",
    "build_model.smk",
    "scripts/run_workflows.py",
)


@pytest.mark.parametrize("snakefile", BANNER_SOURCES)
def test_banner_fallback_cannot_raise(snakefile):
    """`_summary` / `_header` must not report a banner failure on the failed stream.

    Both write the block with `sys.stderr.write` under a broad `except`, and the
    docstring on each promises it never raises. The fallback inside that except
    used to be a bare `print(..., file=sys.stderr)` — the SAME stream — so when
    stderr was what failed, the handler raised again and the second exception
    escaped. Observed 2026-08-17 on wf0: `[Errno 22] Invalid argument` came out
    of `onerror` and was reported as an OSError in analyze_climate.smk, masking
    the two `extract_historical_climate_*` rules that had actually failed.
    `_header` is the worse half — it runs from `onstart`, so a raise there ends
    the run before any rule executes.

    A source assertion because these live in the Snakefiles, which are not
    importable Python. It is deliberately shape-only: that each fallback sits
    directly under its own `try:` and that the `try` has an `except`. What the
    fallback prints is not pinned here.
    """
    lines = Path(SNAKEDIR, snakefile).read_text(encoding="utf-8").splitlines()
    sites = [
        i
        for i, line in enumerate(lines)
        if "run header unavailable: {exc}" in line
        or "run summary unavailable: {exc}" in line
    ]
    assert len(sites) == 2, (
        f"{snakefile}: expected the _summary and _header fallbacks, found {len(sites)}"
    )
    for i in sites:
        before = next(
            line
            for line in reversed(lines[:i])
            if line.strip() and not line.lstrip().startswith("#")
        )
        assert before.strip() == "try:", (
            f"{snakefile}:{i + 1}: the banner fallback writes to sys.stderr outside a "
            "try -- and sys.stderr may be exactly what failed above it"
        )
        after = next(line for line in lines[i + 1 :] if line.strip())
        assert after.strip().startswith("except Exception"), (
            f"{snakefile}:{i + 1}: the banner fallback's try has no except"
        )
