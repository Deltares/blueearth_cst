"""Test some snake command line interface (CLI) for validity of snakefiles."""

import os
from os.path import join, dirname, realpath
from pathlib import Path
import subprocess

import yaml
import pytest

TESTDIR = dirname(realpath(__file__))
SNAKEDIR = join(TESTDIR, "..")

config_fn = join(TESTDIR, "snake_config_model_test.yml")
linux_config_fn = join(
    SNAKEDIR, "config", "workflows", "snake_config_model_test_linux.yml"
)

# Minimal valid GeoJSON standing in for the workflow-1 region output that
# climate_projections consumes as a cross-workflow input (see the fixture).
_MINIMAL_REGION_GEOJSON = (
    '{"type": "FeatureCollection", "features": [{"type": "Feature", '
    '"properties": {}, "geometry": {"type": "Polygon", "coordinates": '
    "[[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]}}]}"
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

    climate_projections declares `{project_dir}/hydrology_model/staticgeoms/
    region.geojson` as an `ancient(...)` input produced by model_creation — a
    correct cross-workflow contract Snakemake will not satisfy on its own. To
    dry-run workflow 2 in isolation we stage a minimal valid region file under
    a **test-owned tmp project_dir** (never the tracked baseline dir) and point
    a copy of the test config at it. tmp_path is torn down by pytest.

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

    region = tmp_path / "hydrology_model" / "staticgeoms" / "region.geojson"
    region.parent.mkdir(parents=True)
    region.write_text(_MINIMAL_REGION_GEOJSON, encoding="utf-8")

    wf1_snapshot = tmp_path / "config" / "runs" / "snake_config_model_creation.yml"
    wf1_snapshot.parent.mkdir(parents=True)
    wf1_snapshot.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    cfg_path = tmp_path / "snake_config_staged.yml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def test_snakefile_cli_model_creation():
    """Workflow 1 dry-run builds a clean DAG on the test config."""
    result = _dry_run("Snakefile_model_creation")
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")


def test_snakefile_cli_model_creation_linux_config():
    """The Linux config must still build a DAG after O-01 retired `data/`.

    R07 deletes the tracked `data/` tree, whose only live consumers were this
    config and the Docker runner. Linux *end-to-end* validation stays parked
    (no Linux machine), but parse-level consistency is cheap and is exactly
    what a silently-broken config would fail: DAG build resolves every
    config-declared path, so a dangling `output_locations` would surface here.
    Runs on both CI legs.
    """
    result = _dry_run("Snakefile_model_creation", cfg=linux_config_fn)
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")


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
        mc = cfg["workflows"]["model_creation"]
        for key in ("output_locations", "observations_timeseries"):
            assert mc[key] is None, (
                f"{cfg_path}:{key} is {mc[key]!r}; shipped configs use YAML "
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
    from blueearth_cst.shared.plot_map import gauges_layer_name

    for unset in (None, "None"):
        assert gauges_layer_name(unset) is None, unset
        # The existence-based guards: neither spelling names a real file, so
        # both take the skip branch. `os.path.isfile(None)` would raise, which
        # is why the `is not None` half has to come first in those callers.
        assert not (unset is not None and os.path.isfile(unset)), unset

    # And a real path is still recognised, so the assertions above are not just
    # "everything is falsy".
    assert gauges_layer_name("gauges/my_stations.csv") == "gauges_my_stations"


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
def test_short_window_fails_wf1_dry_run_at_parse_time(tmp_path, endtime, label):
    """A historical_window under MIN_HISTORICAL_YEARS must red the dry-run.

    Same parse-time stance, and same test shape, as the eobs rejection above:
    no execution can rescue the config, so the earliest failure is the most
    legible one. Pre-guard, a sub-year window reached rule 1.11 and died with
    MissingOutputException nine rules and one hydromt build past the cause
    (dev/followups.md R7-6), and a ten-year window ran WF1 to completion before
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


def test_snakefile_cli_climate_projections(config_with_staged_region):
    """Workflow 2 dry-run builds a clean DAG once its WF1 region input is staged.

    climate_projections declares region.geojson (a model_creation output) as an
    `ancient(...)` input Snakemake will not build itself — correct behavior. R3
    stages it in a test-owned tmp project_dir (see the fixture) rather than
    weakening the contract; workflow 2's Snakefile is untouched (R4 territory).
    Was a MissingInputException ratchet pre-R3 (dev/followups.md).
    """
    result = _dry_run(
        "Snakefile_climate_projections", cfg=config_with_staged_region
    )
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")


def test_climate_projections_declares_wf1_region_input():
    """Pin the fixture to the real contract it stands in for.

    The dry-run fixture stages `staticgeoms/region.geojson`; this guards that
    Snakefile_climate_projections still declares that exact workflow-1 output
    path as an input, so the fixture can never silently diverge from the
    cross-workflow contract.
    """
    text = Path(SNAKEDIR, "Snakefile_climate_projections").read_text()
    assert "staticgeoms/region.geojson" in text


def test_snakefile_cli_climate_experiment(config_with_staged_region):
    """Workflow 3 dry-run builds a clean DAG on the test config (R5 fixed the cycle).

    Pre-R5 this tripped a CyclicGraphException at rule
    generate_climate_stress_test: its output wildcard rlz_{rlz_num}_cst_{st_num}.nc
    could resolve st_num to 0, making the rule a second eligible producer of
    cst_0.nc (a self-loop). R5 removed it with a rule-local
    `wildcard_constraints: st_num=[1-9][0-9]*` on that rule. Once the cycle is
    gone the ancient(region.geojson) input existence is checked, so this reuses
    the same staged-region fixture as workflow 2 (region.geojson is the sole
    unbuilt cross-workflow leaf). Was a CyclicGraphException ratchet pre-R5
    (dev/followups.md § R3).
    """
    result = _dry_run(
        "Snakefile_climate_experiment", cfg=config_with_staged_region
    )
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")
