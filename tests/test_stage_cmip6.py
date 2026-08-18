"""Contract tests for `dev/scripts/stage_cmip6.py`.

The tool's whole claim is that its slices are WF2-cache-compatible — drop one
into a project's ``raw/`` and the pipeline accepts it instead of re-opening the
remote store. That claim rests entirely on the tool rebuilding the same digest
the Snakefile builds, from a recipe that lives in two places
(``analyze_projections.smk::series_digest_components`` and
``stage_cmip6.digest_components``) and therefore can drift.

Two layers, deliberately:

* NAMING and RECIPE-SHAPE cases run on every checkout, including a bare CI leg.
  They pin the filename grammar and that the raw components carry no reducer
  hash — the exclusion the whole stage-A split exists for.
* One FIXTURE case is the real guard: it recomputes the digest for a slice WF2
  itself wrote and asserts equality with the ``cst_raw_digest`` on the file. A
  drift in either recipe turns it red. It skips without ``test_case/``, and
  AGENTS.md records that a worktree lacking that tree downgrades rather than
  fails — so a green run here is not evidence this case ran.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, os.path.join(str(_REPO_ROOT), "dev", "scripts"))

import stage_cmip6 as sc  # noqa: E402

from blueearth_cst.projections import series_identity as _si  # noqa: E402

FIXTURE = _REPO_ROOT / "test_case" / "test_local"
RAW_DIR = FIXTURE / "data" / "climate" / "projections" / "cmip6" / "raw"
REGION = FIXTURE / "data" / "spatial" / "geoms" / "region.geojson"

#: The variables the seed config asks for, POST-rename — what the catalog's
#: adapter calls them and what the digest therefore carries. `pr`/`tas` are the
#: raw CMIP6 names and would not match; that mistake is the reason this constant
#: is spelled out here rather than inlined.
SEED_VARIABLES = {"precip": {"units": "kg m-2 s-1"}, "temp": {"units": "K"}}


def _cfg():
    return {
        "clim_project": "cmip6",
        "catalog": str(_REPO_ROOT / "config/catalogs/cmip6_data.yml"),
        "store_index": str(_REPO_ROOT / "config/catalogs/cmip6_store_index.json"),
        "buffer_degrees": sc.DEFAULT_BUFFER_DEGREES,
        "variables": SEED_VARIABLES,
    }


# --- naming grammar ----------------------------------------------------------


def test_series_key_matches_the_snakefile_grammar():
    """A slash in a model id becomes an underscore; everything else is verbatim.

    This IS the filename a staged slice gets, so a change here silently stops
    the files dropping into `raw/` under the name WF2 looks for.
    """
    assert (
        sc.series_key("cmip6", "NOAA-GFDL/GFDL-ESM4", "historical", "r1i1p1f1")
        == "cmip6_NOAA-GFDL_GFDL-ESM4_historical_r1i1p1f1"
    )


def test_catalog_entry_keeps_the_member_placeholder_unresolved():
    """The generated catalog expands `{member}` at generation time.

    `fetch_raw_slice` resolves it through the catalog's own grammar, so the
    entry handed in must still carry the literal placeholder.
    """
    entry = sc.catalog_entry_name("cmip6", "INM/INM-CM4-8", "ssp245")
    assert entry == "cmip6_INM/INM-CM4-8_ssp245_{member}"
    assert "{member}" in entry


def test_the_model_slash_survives_in_the_entry_but_not_the_filename():
    """The two grammars differ, and conflating them is the obvious mistake."""
    key = sc.series_key("cmip6", "INM/INM-CM4-8", "ssp245", "r1i1p1f1")
    entry = sc.catalog_entry_name("cmip6", "INM/INM-CM4-8", "ssp245")
    assert "/" not in key
    assert "/" in entry


# --- recipe shape ------------------------------------------------------------


@pytest.mark.skipif(
    not (_REPO_ROOT / "config/catalogs/cmip6_data.yml").is_file(),
    reason="needs the generated cmip6 catalog",
)
def test_components_carry_no_reducer_hash():
    """The exclusion the stage-A split exists for.

    If a reducer hash reached these components, editing a reduction formula
    would change the RAW digest and re-download every slice — which is the
    precise cost the split was built to remove (open 1142 s vs reduce 0.2 s).
    """
    components = sc.digest_components(_cfg(), "INM/INM-CM4-8", "historical", "r1i1p1f1")
    assert "reducer_module_hash" not in components


# --- the real guard ----------------------------------------------------------


@pytest.mark.skipif(
    not (RAW_DIR.is_dir() and REGION.is_file()),
    reason="needs the built test_case/test_local fixture with WF2 raw slices",
)
def test_the_tool_reproduces_a_digest_wf2_itself_wrote():
    """Recompute the digest for slices the pipeline wrote, and require equality.

    This is what makes the cache-compatibility claim checkable rather than
    asserted. It reads `cst_raw_digest` off files produced by a real WF2 run
    and rebuilds it from the tool's own recipe; the two must agree exactly, or
    a staged slice would be re-fetched instead of reused.
    """
    xr = pytest.importorskip("xarray")

    slices = sorted(RAW_DIR.glob("cmip6_*.nc"))
    assert slices, "the fixture's raw/ is empty; this test would prove nothing"

    region_fp = _si.region_fingerprint(str(REGION))
    checked = 0
    for path in slices:
        with xr.open_dataset(path) as ds:
            written = ds.attrs.get("cst_raw_digest")
            variables = {name: {"units": ""} for name in ds.data_vars}
        if not written:
            continue
        # stem: cmip6_<model with _ for />_<experiment>_<member>
        stem = path.stem[len("cmip6_") :]
        model_experiment, member = stem.rsplit("_", 1)
        model_us, experiment = model_experiment.rsplit("_", 1)
        model = model_us.replace("_", "/", 1)

        cfg = _cfg() | {"variables": variables}
        recomputed = _si.raw_digest(
            sc.digest_components(cfg, model, experiment, member), region_fp
        )
        assert recomputed == written, (
            f"{path.name}: the tool's recipe no longer reproduces the digest WF2 "
            f"wrote — a staged slice would be re-fetched, not reused"
        )
        checked += 1

    assert checked, "no slice carried a cst_raw_digest; nothing was verified"


# --- the parallel machinery --------------------------------------------------


def _write_cfg(tmp_path, models, target):
    """A minimal config file pointing at the fixture region."""
    import yaml

    cfg = {
        "region": str(REGION),
        "target_root": str(target),
        "models": models,
        "scenarios": ["historical"],
        "members": ["r1i1p1f1"],
        "variables": SEED_VARIABLES,
    }
    path = tmp_path / "stage_cmip6.yml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.mark.skipif(not REGION.is_file(), reason="needs the test_local fixture region")
def test_stage_one_returns_the_error_instead_of_raising():
    """One unavailable source must not end a run with hours of work in it.

    A model the catalog does not carry is an ordinary fact, not a crash, so the
    worker reports it as a value. Uses a deliberately absent model, which fails
    long before any network call.
    """
    cfg = _cfg() | {"region": str(REGION), "clim_project": "cmip6"}
    job = {
        "key": "cmip6_NO_SUCH_MODEL_historical_r1i1p1f1",
        "model": "NO/SUCH-MODEL",
        "experiment": "historical",
        "member": "r1i1p1f1",
        "out": "unused.nc",
    }
    key, error, elapsed = sc.stage_one(cfg, job)
    assert key == job["key"]
    assert error, "an absent model must be reported, not silently succeed"
    assert elapsed >= 0, "every slice reports a duration, failures included"


# --- pre-filtering against the catalog ---------------------------------------


def test_plan_refuses_a_scenario_the_model_never_published():
    """19 of 65 models with `historical` never submitted `ssp245`.

    The catalog knows, so the request is refused before a worker is spawned.
    """
    # `plan` never opens the region -- it only consults the catalog -- so these
    # cases need no fixture and run on a bare checkout, which is where a
    # regression in the pre-filter would otherwise go unseen.
    cfg = _cfg() | {
        "region": "unused",
        "target_root": "unused",
        "models": ["NCAR/CESM2-FV2"],
        "scenarios": ["ssp245"],
        "members": ["r1i1p1f1"],
    }
    jobs, skipped = sc.plan(cfg)
    assert jobs == []
    assert len(skipped) == 1
    assert "published no ssp245" in skipped[0][1]


def test_plan_refuses_a_member_the_entry_does_not_have_and_names_the_real_ones():
    """The failure this exists to prevent, and the worst error in the old log.

    UKESM1-0-LL publishes the `f2` forcing variant from realisation 13, so
    `r1i1p1f1` does not exist. Unfiltered it reached hydromt, which could not
    resolve the name, treated it as a LOCAL PATH and reported a NoDataException
    about finding no files -- with the model's slash read as a directory
    separator, and nothing anywhere saying "wrong member". 70 of the catalog's
    289 entries lack `r1i1p1f1`, so this is the common case, not an oddity.
    """
    # `plan` never opens the region -- it only consults the catalog -- so these
    # cases need no fixture and run on a bare checkout, which is where a
    # regression in the pre-filter would otherwise go unseen.
    cfg = _cfg() | {
        "region": "unused",
        "target_root": "unused",
        "models": ["NIMS-KMA/UKESM1-0-LL"],
        "scenarios": ["historical"],
        "members": ["r1i1p1f1"],
    }
    jobs, skipped = sc.plan(cfg)
    assert jobs == []
    reason = skipped[0][1]
    assert "member r1i1p1f1 not available" in reason
    assert "r13i1p1f2" in reason, "the reason must name what the entry DOES have"


def test_plan_keeps_a_combination_the_catalog_really_carries():
    """The filter must not be over-eager -- a real request still plans."""
    # `plan` never opens the region -- it only consults the catalog -- so these
    # cases need no fixture and run on a bare checkout, which is where a
    # regression in the pre-filter would otherwise go unseen.
    cfg = _cfg() | {
        "region": "unused",
        "target_root": "unused",
        "models": ["INM/INM-CM4-8"],
        "scenarios": ["historical"],
        "members": ["r1i1p1f1"],
    }
    jobs, skipped = sc.plan(cfg)
    assert skipped == []
    assert [job["key"] for job in jobs] == ["cmip6_INM_INM-CM4-8_historical_r1i1p1f1"]


# --- the parallel machinery, offline -----------------------------------------


def _local_catalog(tmp_path):
    """A catalog whose one entry passes the filter but resolves to nothing.

    Cloned from a real entry so the spec is realistic, with the URI pointed at
    a local path that does not exist -- so `fetch_raw_slice` fails FAST and
    without touching the network, which is what makes a pool test cheap.
    """
    import copy

    import yaml

    with open(_REPO_ROOT / "config/catalogs/cmip6_data.yml", encoding="utf-8") as h:
        real = yaml.safe_load(h)
    spec = copy.deepcopy(real["cmip6_INM/INM-CM4-8_historical_{member}"])
    # `file://` on purpose: a bare Windows path is handed to gcsfs, which
    # treats `C:\...` as a bucket name and spends ~40 s retrying an HTTP 400
    # before giving up. An explicit local scheme fails immediately and keeps
    # the case genuinely offline.
    spec["uri"] = (tmp_path / "no_such_store" / "{variable}").as_uri()
    out = {}
    for name in ("FAKE/MODEL-A", "FAKE/MODEL-B"):
        entry = copy.deepcopy(spec)
        entry.setdefault("placeholders", {})["member"] = ["r1i1p1f1"]
        out[f"cmip6_{name}_historical_{{member}}"] = entry
    path = tmp_path / "catalog.yml"
    path.write_text(yaml.safe_dump(out), encoding="utf-8")
    return path


@pytest.mark.skipif(not REGION.is_file(), reason="needs the test_local fixture region")
def test_the_worker_pool_round_trips_and_reports_every_failure(tmp_path, capsys):
    """Exercise the ProcessPoolExecutor path itself, with no network.

    What this proves is the machinery AROUND the fetch: that `cfg` and a job
    pickle to a worker, that a worker starts and imports the module, and that
    both failures come back and are summarised. The fetch itself is covered by
    the digest guard, not here.
    """
    import yaml

    cfg = {
        "region": str(REGION),
        "target_root": str(tmp_path / "out"),
        "catalog": str(_local_catalog(tmp_path)),
        "models": ["FAKE/MODEL-A", "FAKE/MODEL-B"],
        "scenarios": ["historical"],
        "members": ["r1i1p1f1"],
        "variables": SEED_VARIABLES,
    }
    cfg_path = tmp_path / "stage_cmip6.yml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    code = sc.main(["--config", str(cfg_path), "--workers", "2"])
    out = capsys.readouterr().out
    assert code == 1, "a run where every slice failed must exit nonzero"
    assert "attempted   2" in out, "both slices must survive the pre-filter"
    assert "staged      0 of 2" in out
    assert "could not be downloaded  (2)" in out


def test_resolve_workers_caps_at_the_slice_count_and_floors_at_one():
    """Each worker costs ~7 s of imports and ~311 MiB, so idle ones are a cost.

    Extracted from `main` precisely so this can be asserted without starting a
    pool -- the arithmetic is the claim, and running a fetch to check it added
    36 s to the suite for nothing.
    """
    assert sc.resolve_workers(8, 1) == 1
    assert sc.resolve_workers(2, 5) == 2
    assert sc.resolve_workers(0, 5) == 1


#: A syntactically valid polygon, written into tmp_path. `load_config` only
#: checks that the region FILE exists, and any test that stubs the fetch never
#: reads its geometry -- so depending on `test_case/` for it would tie the case
#: to a fixture no bare checkout has. That is not hypothetical: this test
#: originally used the fixture region, passed everywhere `test_case/` exists,
#: and failed on BOTH CI legs, which is the one place it could.
MINIMAL_REGION = """{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature", "properties": {},
    "geometry": {"type": "Polygon", "coordinates": [[
      [9.6, 0.3], [9.9, 0.3], [9.9, 0.5], [9.6, 0.5], [9.6, 0.3]]]}
  }]
}
"""


def _standalone_region(tmp_path):
    path = tmp_path / "region.geojson"
    path.write_text(MINIMAL_REGION, encoding="utf-8")
    return path


def test_one_worker_stays_in_process(tmp_path, capsys, monkeypatch):
    """`--workers 1` must not spin up a pool.

    That is what keeps a traceback and any attached debugger pointing at the
    real failure, which is the whole reason the flag accepts 1. Asserted by
    making the pool EXPLODE if touched, and stubbing the fetch -- both work
    here only because this path never leaves the process, which is the point.

    Runs on EVERY checkout: it needs no fixture, which is what lets CI catch a
    regression in it.
    """
    import yaml

    def _explode(*_a, **_k):
        raise AssertionError("--workers 1 must not construct a process pool")

    monkeypatch.setattr(sc, "ProcessPoolExecutor", _explode)
    monkeypatch.setattr(sc, "stage_one", lambda cfg, job: (job["key"], None, 0.5))

    cfg = {
        "region": str(_standalone_region(tmp_path)),
        "target_root": str(tmp_path / "out"),
        "models": ["INM/INM-CM4-8"],
        "scenarios": ["historical"],
        "members": ["r1i1p1f1"],
        "variables": SEED_VARIABLES,
    }
    cfg_path = tmp_path / "stage_cmip6.yml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    assert sc.main(["--config", str(cfg_path), "--workers", "1"]) == 0
    assert "workers     1" in capsys.readouterr().out
