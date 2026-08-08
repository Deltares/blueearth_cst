"""Test some snake command line interface (CLI) for validity of snakefiles."""

import os
import sys
from os.path import join, dirname, realpath
from pathlib import Path
import subprocess

import yaml
import pytest

TESTDIR = dirname(realpath(__file__))
SNAKEDIR = join(TESTDIR, "..")

sys.path.insert(0, join(SNAKEDIR, "dev", "scripts"))
import cross_workflow_inputs as cwi  # noqa: E402

config_fn = join(TESTDIR, "snake_config_model_test.yml")
linux_config_fn = join(
    SNAKEDIR, "config", "workflows", "snake_config_model_test_linux.yml"
)

def _dry_run(snakefile, cfg=config_fn):
    """Dry-run a Snakefile on a config; return the completed process.

    stdout/stderr are captured as text so callers can match on the DAG-build
    exception class name. Snakemake writes these diagnostics to stderr, but we
    match on the combined stream so a stream change does not silently break a
    ratchet assertion below.
    """
    os.chdir(SNAKEDIR)
    cmd = f"snakemake all -c 1 -s {snakefile} --configfile {cfg} --dry-run"
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


@pytest.fixture()
def config_with_staged_region(tmp_path):
    """Config whose project_dir is a temp dir pre-staged with the wf1 leaves.

    Historically climate_projections declared `{project_dir}/hydrology_model/
    staticgeoms/region.geojson` as an `ancient(...)` input produced by
    model_creation — a cross-workflow contract Snakemake will not satisfy on its
    own — so the fixture staged a minimal valid region file under a **test-owned
    tmp project_dir** (never the tracked baseline dir). Since ADR 0003 the extent
    is model-free and BOTH downstream workflows delineate their own
    `data/spatial/geoms/region.geojson`, declaring it as an OUTPUT, so the staged
    region is no longer load-bearing for either dry-run; it is retained only
    because removing it belongs with the wider staging consolidation (R9 P5 F3).
    The staged path follows the R9 model root. tmp_path is torn down by pytest.

    Since P3-1 commit 1, climate_experiment's drift guard (rule
    check_project_consistency) additionally declares the wf1 config snapshot
    `{project_dir}/config/runs/snake_config_model_creation.yml` as a mandatory
    `ancient(...)` input — the same class of cross-workflow contract, staged
    the same way. The staged snapshot is serialized from the SAME parsed
    config the dry-run consumes, so the guard's comparands match by
    construction.
    """
    with open(config_fn) as f:
        cfg = yaml.safe_load(f)
    cfg["project"]["project_dir"] = str(tmp_path).replace("\\", "/")

    # ONE definition of the leaf set, shared with `test_guard_invalidation` and
    # `scaffold_project_tree` and proved complete-and-minimal against the real
    # DAG by `tests/test_cross_workflow_inputs.py`. It was three hand-kept
    # copies until R9 P5 F3; R9 P4's rule 3.01c added the two model leaves, two
    # of the three were updated, and the third went red looking like a defect
    # in the thing it tested. EXTRA_REGION is a deliberate non-leaf, kept for
    # the reason given in the docstring above.
    cwi.stage(tmp_path, yaml.safe_dump(cfg), extras=(cwi.EXTRA_REGION,))

    cfg_path = tmp_path / "snake_config_staged.yml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


@pytest.mark.workflow_contract
def test_snakefile_cli_model_creation():
    """Workflow 1 dry-run builds a clean DAG on the test config."""
    result = _dry_run("Snakefile_model_creation")
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")


@pytest.mark.workflow_contract
def test_snakefile_cli_model_creation_linux_config():
    """The Linux config must still build a DAG after O-01 retired `data/`.

    R07 deletes the tracked `data/` tree, whose only live consumers were this
    config and the Docker runner. Linux *end-to-end* validation stays parked
    (no Linux machine), but parse-level consistency is cheap and is exactly
    what a silently-broken config would fail: DAG build resolves every
    config-declared path, so a dangling `gauge_points` would surface here.
    Runs on both CI legs.
    """
    result = _dry_run("Snakefile_model_creation", cfg=linux_config_fn)
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")


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
        with open(config_fn) as f:
            cfg = yaml.safe_load(f)
        cfg["project"]["project_dir"] = "_o22_probe_project"
        cfg_path = tmp_path / "snake_config_in_repo.yml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        result = _dry_run("Snakefile_model_creation", cfg=str(cfg_path))
        combined = (result.stdout or "") + (result.stderr or "")
        assert "inside the repository tree" in combined, combined[-3000:]
        assert result.returncode == 0, combined[-3000:]
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
        scratch.rmdir()


@pytest.mark.workflow_contract
def test_baseline_seed_config_does_not_warn():
    """The exemption holds for the config the baseline gate actually runs.

    That is config/workflows/snake_config_model_test.yml (project_dir:
    test_case/test_local) -- NOT tests/snake_config_model_test.yml, which
    points at tests/test_project and therefore warns correctly: it is an
    in-repo project_dir outside the single exemption. The exemption exists
    because the baseline seed config is TRACKED and a tracked config cannot
    carry a machine-specific absolute path; it does not extend to every
    convenient in-repo scratch dir.
    """
    seed_cfg = join(SNAKEDIR, "config", "workflows", "snake_config_model_test.yml")
    result = _dry_run("Snakefile_model_creation", cfg=seed_cfg)
    combined = (result.stdout or "") + (result.stderr or "")
    assert "inside the repository tree" not in combined, combined[-3000:]
    assert result.returncode == 0, combined[-3000:]


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
    for cfg_path in (config_fn, linux_config_fn):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        basin = cfg["shared"]["basin"]
        mc = cfg["workflows"]["model_creation"]
        values = {
            "shared.basin.gauge_points": basin["gauge_points"],
            "workflows.model_creation.observations_timeseries": mc[
                "observations_timeseries"
            ],
        }
        for key, value in values.items():
            assert value is None, (
                f"{cfg_path}:{key} is {value!r}; shipped configs use YAML "
                f"null, not the legacy 'None' string"
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
    with open(config_fn) as f:
        cfg = yaml.safe_load(f)
    cfg["shared"]["clim_historical"] = "eobs"
    cfg_path = tmp_path / "snake_config_eobs.yml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    result = _dry_run("Snakefile_model_creation", cfg=str(cfg_path))
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode != 0, "eobs config must fail the wf1 dry-run"
    assert (
        "clim_historical: eobs is not supported by the P3-2a wf1 raw-climate "
        "path; supported sources: era5, chirps, chirps_global"
    ) in combined, combined


@pytest.mark.parametrize(
    "endtime, label",
    [
        ("2000-06-01T00:00:00", "sub-year"),
        # The case the UNIFIED floor added: WF1 used to build a model happily on
        # ten years and let WF3 discover the problem inside weathergenr.
        ("2010-01-01T00:00:00", "ten-year"),
    ],
)
@pytest.mark.workflow_contract
def test_short_window_fails_wf1_dry_run_at_parse_time(tmp_path, endtime, label):
    """A historical_window under MIN_HISTORICAL_YEARS must red the dry-run.

    Same parse-time stance, and same test shape, as the eobs rejection above:
    no execution can rescue the config, so the earliest failure is the most
    legible one. Pre-guard, a sub-year window reached rule 1.11 and died with
    MissingOutputException nine rules and one hydromt build past the cause
    (dev/followups-archive.md R7-6), and a ten-year window ran WF1 to completion before
    failing a whole workflow away.
    """
    with open(config_fn) as f:
        cfg = yaml.safe_load(f)
    cfg["shared"]["historical_window"] = {
        "starttime": "2000-01-01T00:00:00",
        "endtime": endtime,
    }
    cfg_path = tmp_path / f"snake_config_{label}.yml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    result = _dry_run("Snakefile_model_creation", cfg=str(cfg_path))
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode != 0, f"{label} window must fail the wf1 dry-run"
    assert "16-year minimum" in combined, combined[-3000:]
    # The message must be actionable without opening the Snakefile.
    assert "weathergenr" in combined, combined[-3000:]


@pytest.mark.workflow_contract
def test_snakefile_cli_climate_projections(config_with_staged_region):
    """Workflow 2 dry-run builds a clean DAG once its WF1 region input is staged.

    climate_projections declares region.geojson (a model_creation output) as an
    `ancient(...)` input Snakemake will not build itself — correct behavior. R3
    stages it in a test-owned tmp project_dir (see the fixture) rather than
    weakening the contract; workflow 2's Snakefile is untouched (R4 territory).
    Was a MissingInputException ratchet pre-R3 (dev/tasks/).
    """
    result = _dry_run(
        "Snakefile_climate_projections", cfg=config_with_staged_region
    )
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")


def test_climate_projections_owns_its_region():
    """Pin WF2's region contract to what it actually is.

    This guard used to assert that `staticgeoms/region.geojson` appears in
    Snakefile_climate_projections, standing in for a wf1 -> wf2 cross-workflow
    input. That input is gone: since ADR 0003 the extent is model-free and WF2
    delineates its OWN `data/spatial/geoms/region.geojson`. The literal string
    still occurred in the file — in a comment recording that WF2 was *freed*
    from it — so the assertion kept passing while guarding nothing, found by the
    R9 P5 stale-path sweep.

    What is worth pinning is the current shape: WF2 produces the model-free
    region, and reads nothing from the wf1 model root.
    """
    text = Path(SNAKEDIR, "Snakefile_climate_projections").read_text()
    assert "data/spatial/geoms/region.geojson" not in text, (
        "the region path belongs in snake_utils.region_rule, not inline here"
    )
    assert "region_rule(" in text and "REGION.region_geojson" in text
    # The model root may only appear as history, never as a declared dependency.
    for line in text.splitlines():
        if "hydrology_model/" in line or "models/hydrology/wflow" in line:
            assert line.lstrip().startswith("#"), f"WF2 reads the model root: {line}"


@pytest.mark.workflow_contract
def test_snakefile_cli_climate_experiment(config_with_staged_region):
    """Workflow 3 dry-run builds a clean DAG on the test config (R5 fixed the cycle).

    Pre-R5 this tripped a CyclicGraphException at rule
    generate_climate_stress_test: its output wildcard rlz_{rlz_num}_st_{st_num}.nc
    could resolve st_num to 0, making the rule a second eligible producer of
    st_0.nc (a self-loop). R5 removed it with a rule-local
    `wildcard_constraints: st_num=[1-9][0-9]*` on that rule. Once the cycle is
    gone the ancient(region.geojson) input existence is checked, so this reuses
    the same staged-region fixture as workflow 2 (region.geojson is the sole
    unbuilt cross-workflow leaf). Was a CyclicGraphException ratchet pre-R5
    (dev/tasks/ § R3).
    """
    result = _dry_run(
        "Snakefile_climate_experiment", cfg=config_with_staged_region
    )
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")
