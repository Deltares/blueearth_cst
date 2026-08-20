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

import io
import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, os.path.join(str(_REPO_ROOT), "dev", "scripts"))

import stage_cmip6 as sc  # noqa: E402

import blueearth_cst.shared.snake_utils as su  # noqa: E402
from blueearth_cst.projections import series_identity as _si  # noqa: E402
from blueearth_cst.shared.snake_utils import log_row  # noqa: E402

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
        "buffer_cells": sc.DEFAULT_BUFFER_CELLS,
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

    Slices recorded under an OLDER `cst_schema_version` are passed over, not
    failed. A digest is only comparable within one schema: the pipeline itself
    rejects such a slice on the schema check (`cache_hit`) and re-fetches it
    before any digest is consulted, so an inequality here would say nothing
    about the tool's recipe. Skipping keeps a deliberate bump — 4->5 renamed the
    `buffer_degrees` component to `buffer_cells` (t2608182238) — from reading as
    a defect in this tool until the fixture is re-run. The whole-fixture skip
    below is what stops that concession from hiding a real break: once a run
    refreshes the tree, every slice is current and every slice is checked again.

    Slices whose `cst_entry_identity_digest` no longer matches the live catalog
    are passed over on the same reasoning, and this second case is why that
    attribute exists. A schema bump is not the only way to strand a fixture: a
    catalog edit moves `entry_identity`, and therefore every raw digest, while
    `cst_schema_version` sits still — so the skip above cannot see it and the
    suite goes red until someone re-runs WF2. That is a stale artifact, not a
    broken recipe, and t2608201134 records the occurrence that proved it
    (`b0963e9` added tasmin/tasmax to `cmip6_data.yml` 72 minutes after this
    fixture was written). What still FAILS is a mismatch the recorded
    provenance cannot account for: same schema, same catalog entry, different
    digest. That is the recipe breaking, which is what this test is for.

    A slice predating the attribute has nothing to compare and falls through to
    the assertion, which is the behaviour this test had before.
    """
    xr = pytest.importorskip("xarray")

    slices = sorted(RAW_DIR.glob("cmip6_*.nc"))
    assert slices, "the fixture's raw/ is empty; this test would prove nothing"

    region_fp = _si.region_fingerprint(str(REGION))
    checked = 0
    stale = 0
    recatalogued = 0
    for path in slices:
        with xr.open_dataset(path) as ds:
            written = ds.attrs.get("cst_raw_digest")
            schema = ds.attrs.get("cst_schema_version")
            recorded_entry = ds.attrs.get("cst_entry_identity_digest")
            variables = {name: {"units": ""} for name in ds.data_vars}
        if not written:
            continue
        if schema != _si.SCHEMA_VERSION:
            stale += 1
            continue
        # stem: cmip6_<model with _ for />_<experiment>_<member>
        stem = path.stem[len("cmip6_") :]
        model_experiment, member = stem.rsplit("_", 1)
        model_us, experiment = model_experiment.rsplit("_", 1)
        model = model_us.replace("_", "/", 1)

        cfg = _cfg() | {"variables": variables}
        components = sc.digest_components(cfg, model, experiment, member)
        recomputed = _si.raw_digest(components, region_fp)
        if recomputed != written and recorded_entry:
            # The catalog entry this slice was built from is not the one on disk
            # now, so the digest SHOULD differ and the artifact is simply stale.
            # Only claim that when the recorded provenance says so.
            if recorded_entry != _si.entry_identity_digest(components, member):
                recatalogued += 1
                continue
        assert recomputed == written, (
            f"{path.name}: the tool's recipe no longer reproduces the digest WF2 "
            f"wrote — a staged slice would be re-fetched, not reused"
        )
        checked += 1

    if not checked and (stale or recatalogued):
        why = []
        if stale:
            why.append(f"{stale} predate schema {_si.SCHEMA_VERSION}")
        if recatalogued:
            why.append(
                f"{recatalogued} were built from a catalog entry that has since "
                "changed, so their digests moved with it"
            )
        pytest.skip(
            f"none of the fixture's {len(slices)} raw slice(s) is comparable: "
            + "; ".join(why)
            + ". Re-run WF2 stage A against test_case/test_local to restore "
            "this check"
        )
    assert checked, "no slice carried a cst_raw_digest; nothing was verified"


# --- entry-identity attribution (t2608201134) --------------------------------


def _components(rename):
    """Minimal digest components carrying one member's catalog entry."""
    return {
        "entry_identity": {
            "r1i1p1f1": {
                "uri": "gs://cmip6/.../{variable}/gn/v1".replace("{variable}", "tas"),
                "driver": "zarr",
                "data_adapter": {"rename": dict(rename)},
                "metadata": {"crs": 4326},
            }
        }
    }


def test_entry_identity_digest_moves_when_the_catalog_entry_moves():
    """The attribute has to notice the edit that stranded the fixture.

    `b0963e9` added `tasmin`/`tasmax` to the catalog's rename map, which moved
    `entry_identity` and so every raw digest. An attribute that did not move
    with it could not tell a stale artifact from a broken recipe, which is its
    only job.
    """
    before = _components({"pr": "precip", "tas": "temp"})
    after = _components({"pr": "precip", "tas": "temp", "tasmin": "temp_min"})
    assert _si.entry_identity_digest(before, "r1i1p1f1") != _si.entry_identity_digest(
        after, "r1i1p1f1"
    )


def test_entry_identity_digest_ignores_mapping_order():
    """Canonical, like every other digest here -- order is not an edit."""
    a = _components({"pr": "precip", "tas": "temp"})
    b = _components({"tas": "temp", "pr": "precip"})
    assert _si.entry_identity_digest(a, "r1i1p1f1") == _si.entry_identity_digest(
        b, "r1i1p1f1"
    )


def test_entry_identity_digest_is_absent_for_an_unknown_member():
    """A member the components do not carry hashes the empty mapping.

    Not an error: `digest_components` is built for the members a run asked for,
    and a reader passing another one should get a value that simply never
    matches rather than an exception inside a diagnostic path.
    """
    assert _si.entry_identity_digest(_components({"pr": "precip"}), "r9i9p9f9") == (
        _si.entry_identity_digest({"entry_identity": {}}, "r1i1p1f1")
    )


def test_a_diagnostic_attribute_does_not_change_a_cache_decision(tmp_path):
    """The falsifier for "no SCHEMA_VERSION bump is owed".

    `cache_hit` compares the schema version and the digest attribute and nothing
    else, so stamping an extra `cst_*` key must leave an otherwise-current slice
    a HIT. If this ever fails, adding the attribute became a contract change and
    the bump is owed after all.
    """
    xr = pytest.importorskip("xarray")

    path = tmp_path / "slice.nc"
    ds = xr.Dataset({"temp": ("time", [1.0, 2.0])})
    ds.attrs = {
        "cst_schema_version": _si.SCHEMA_VERSION,
        "cst_raw_digest": "deadbeef",
        "cst_entry_identity_digest": "an attribute cache_hit has never heard of",
    }
    ds.to_netcdf(path)

    assert _si.cache_hit([path], "deadbeef", digest_attr="cst_raw_digest")
    assert not _si.cache_hit([path], "a different digest", digest_attr="cst_raw_digest")


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
def test_stage_one_returns_the_error_instead_of_raising(tmp_path):
    """One unavailable source must not end a run with hours of work in it.

    A model the catalog does not carry is an ordinary fact, not a crash, so the
    worker reports it as a value. Uses a deliberately absent model, which fails
    long before any network call.
    """
    cfg = _cfg() | {
        "region": str(REGION),
        "clim_project": "cmip6",
        "target_root": str(tmp_path),
    }
    job = {
        "key": "cmip6_NO_SUCH_MODEL_historical_r1i1p1f1",
        "model": "NO/SUCH-MODEL",
        "experiment": "historical",
        "member": "r1i1p1f1",
        "out": "unused.nc",
    }
    key, error, elapsed, _notices = sc.stage_one(cfg, job)
    assert key == job["key"]
    assert error, "an absent model must be reported, not silently succeed"
    # Named, so the test cannot pass on an unrelated failure -- a missing config
    # key would raise inside the same `try` and satisfy a bare truthiness check.
    assert "SUCH-MODEL" in error or "KeyError" in error
    assert elapsed >= 0, "every slice reports a duration, failures included"
    # The failure has a log part, and it holds the traceback `stage_one`
    # flattened to one line for the console.
    part = Path(sc.slice_log_path(cfg["target_root"], job["key"]))
    assert part.is_file(), "a failed slice must leave the log that explains it"
    assert "Traceback" in part.read_text(encoding="utf-8")


def test_stage_one_routes_the_fetch_through_the_workflow_tee(tmp_path, capsys):
    """The staging tool is a WRAPPER around WF2's fetch, console included.

    What the fetch says goes through `tee_to_log` -- the same context manager
    the Snakemake rule uses -- so the mute table, the compaction and the log
    part are the pipeline's rather than a second implementation living in
    Pinned by driving three rows through a stubbed fetch: one the shared mute
    table drops, one INFO that only this tool's console filter drops, and one
    WARNING that must survive both. All three must reach the log part.
    """

    def _fake_fetch(**kwargs):
        log_row("gcsfs extended-filesystem switch = 'false'", module="fetch")
        log_row("fetching cmip6_X/Y_historical_r1i1p1f1", module="fetch")
        log_row(
            "irregular grid, applying the bbox directly: cmip6_X/Y_historical_r1i1p1f1",
            module="fetch",
            level="WARNING",
        )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(sc, "fetch_raw_slice", _fake_fetch)
    # Stubbed too: it is resolved as an ARGUMENT to the fetch, so a made-up
    # (model, scenario) fails in the catalog before the stub is ever reached.
    # The recipe itself is pinned by the fixture case; this test is about where
    # the fetch's output goes.
    monkeypatch.setattr(sc, "digest_components", lambda *a, **k: {})
    try:
        cfg = _cfg() | {
            "region": "unused.geojson",
            "target_root": str(tmp_path),
        }
        job = {
            "key": "cmip6_X_Y_historical_r1i1p1f1",
            "model": "X/Y",
            "experiment": "historical",
            "member": "r1i1p1f1",
            "out": str(tmp_path / "slice.nc"),
        }
        _key, error, _elapsed, notices = sc.stage_one(cfg, job)
    finally:
        monkeypatch.undo()
    assert error is None, error
    captured = capsys.readouterr()
    out = captured.out + captured.err
    logged = Path(sc.slice_log_path(cfg["target_root"], job["key"])).read_text(
        encoding="utf-8"
    )
    # NOTHING the fetch says reaches the console live: the parent prints one
    # numbered line per slice, and a worker cannot number itself.
    assert "extended-filesystem switch" not in out
    assert "fetching cmip6_X/Y_historical" not in out
    assert "irregular grid" not in out
    # the WARNING comes back as a notice for the parent to attach, with its
    # stamp, module and the entry name it repeats all stripped
    assert notices == ["WARNING - irregular grid, applying the bbox directly"]
    # the log part is the durable copy of all three rows, entry name included
    for row in ("extended-filesystem switch", "fetching cmip6_X/Y", "irregular grid"):
        assert row in logged, row


def test_slice_log_path_puts_the_logs_anchor_where_the_tee_looks_for_it():
    """`_log_path_parts` splits on a `logs` component; a flat name loses that.

    Without the anchor the log gets no project root, so its header and every
    path inside it stop being relativized against `target_root`.
    """
    path = sc.slice_log_path("C:/data/cmip6_slices", "cmip6_X_Y_historical_r1i1p1f1")
    root, log_id = su._log_path_parts(path)
    assert Path(root) == Path("C:/data/cmip6_slices")
    assert log_id == "cmip6_X_Y_historical_r1i1p1f1.log"


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
    assert "attempted     2" in out, "both slices must survive the pre-filter"
    assert "staged        0 of 2" in out
    assert "could not be downloaded (2):" in out
    # The restyled frame, pinned: a failing slice reports on STDOUT like every
    # other entry (it went to stderr until the console surface was aligned with
    # `stage_data.py`, which interleaved unpredictably out of a worker pool),
    # and the run states its verdict once.
    assert "x [1/2] " in out or "x [2/2] " in out, "entries carry the counter"
    assert "FAILED — 2 slice(s) did not stage" in out
    assert "written: 0" in out and "failed: 2" in out


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
    out = capsys.readouterr().out
    assert "workers      1" in out
    # The regions `stage_data.py` also prints, in order. Asserted here rather
    # than in a test of its own because this is the one case that runs a whole
    # `main` on every checkout, fixture or not.
    #
    # "Description" is deliberately still in the candidate tuple after the
    # block was dropped: the assertion is an equality on the whole list, so a
    # heading that comes BACK fails here rather than passing unnoticed.
    #
    # Matched on the WHOLE LINE: a region heading is bold Title Case with no
    # decoration, and under capture `bold()` is the identity, so the heading is
    # its bare label. Substring matching would be wrong even now that the
    # description is gone -- "Stage" occurs in ordinary output text.
    regions = ("Description", "Parameters", "Stage", "Dry Run", "Total")
    assert [line for line in out.splitlines() if line in regions] == [
        "Parameters",
        "Stage",
        "Total",
    ]


# --- the one-line entry row ---------------------------------------------------


def test_an_outcome_is_one_line_with_its_detail_trailing_dim(capsys):
    """161 slices used to print 322 rows, half of them saying only a size."""
    sc._entry("+", lambda t: t, "cmip6_X_Y_ssp585_r1i1p1f1", "72.3 KB  47.6s", "[8/9]")
    out = capsys.readouterr().out
    assert out.count("\n") == 1, out
    assert "cmip6_X_Y_ssp585_r1i1p1f1" in out and "72.3 KB" in out and "[8/9]" in out


def test_a_failure_keeps_its_reason_on_a_line_of_its_own(capsys):
    """The multi-version message runs to ~380 characters.

    Inlining it would push the name off the first screen-width and defeat the
    collapse for every row around it.
    """
    sc._entry(
        "x",
        lambda t: t,
        "cmip6_CAS_CAS-ESM2-0_historical_r1i1p1f1",
        "1m02s",
        "[6/6]",
        reason="RuntimeError: " + "the store index records " * 12,
    )
    lines = [ln for ln in capsys.readouterr().out.split("\n") if ln.strip()]
    assert len(lines) == 2
    assert "1m02s" in lines[0] and "cmip6_CAS" in lines[0]
    assert lines[1].startswith("      ")


# --- the worker's console filter ----------------------------------------------


@pytest.mark.parametrize(
    "row",
    [
        "16:24:31 - fetch - fetching cmip6_NCC/NorESM2-MM_ssp245_r1i1p1f1\n",
        "16:24:33 - fetch - pinned gn/v20191108, no bucket listing\n",
        "16:24:33 - fetch - store calendar=noleap (tas)\n",
        "16:24:33 - data_source - Reading cmip6_X from <cmip6>/a/b\n",
    ],
)
def test_the_workers_info_rows_do_not_reach_the_console(row):
    """They say the same slice is being worked on, from a process that cannot
    number it -- and the parent prints one numbered line per slice already."""
    sink = io.StringIO()
    sc._SliceConsole(sink).write(row)
    assert sink.getvalue() == ""


@pytest.mark.parametrize(
    "row,expected",
    [
        (
            "16:25:09 - fetch - WARNING - irregular grid, applying the bbox: x\n",
            "WARNING - irregular grid, applying the bbox: x",
        ),
        (
            "16:26:41 - fetch - ERROR - something went wrong\n",
            "ERROR - something went wrong",
        ),
    ],
)
def test_a_warning_is_collected_rather_than_printed(row, expected):
    """Collected so the PARENT can print it under the slice it belongs to.

    A worker cannot number its own row, so a warning it printed itself landed
    between two unrelated entries and left the reader matching names across
    them. The stamp and module go: the entry above supplies the identity, and
    the level is what makes the line scannable without a coloured glyph.
    """
    sink = io.StringIO()
    console = sc._SliceConsole(sink)
    console.write(row)
    assert sink.getvalue() == "", "a notice must not reach the console live"
    assert console.take_notices() == [expected]
    assert console.take_notices() == [], "reading resets, or it follows the next slice"


def test_a_painted_info_row_is_still_recognised():
    """`_Tee` colours body text on its way to a live console, so this sink sees
    escape codes around the very fields it has to read."""
    sink = io.StringIO()
    sc._SliceConsole(sink).write("\x1b[38;5;245m16:24:31 - fetch - fetching x\x1b[0m\n")
    assert sink.getvalue() == ""


@pytest.mark.parametrize(
    "text",
    [
        "   ... cmip6_X_ssp585_r1i1p1f1: still running, 4m00s elapsed\n",
        "Traceback (most recent call last):\n",
        "some library printing whatever it likes\n",
        "\n",
    ],
)
def test_anything_not_positively_an_info_row_is_printed(text):
    """Same direction of failure as `snake_utils._muted_on_console`: a filter
    over someone else's output lets an unfamiliar line through rather than
    eating it. The heartbeat's stall notice is the one that matters."""
    sink = io.StringIO()
    sc._SliceConsole(sink).write(text)
    assert sink.getvalue() == text


def test_the_console_wrapper_is_reused_for_every_slice_in_a_worker():
    """`tee_to_log` repoints library log handlers by STREAM IDENTITY.

    hydromt binds a StreamHandler the first time a worker parses a catalog, and
    the tee hands it back to whatever `sys.stdout` was on entry. A fresh wrapper
    per slice would leave it pointing at the previous slice's object, which the
    next slice's identity check cannot match -- so hydromt's records bypass the
    tee, arriving uncompacted and unmuted AND missing from the log. Observed on
    a 6-slice run before this was memoized.
    """
    sc._CONSOLE_WRAPPERS.clear()
    try:
        console = io.StringIO()
        first = sc._wrapped_console("stdout", console)
        # same stream -> same wrapper, which is what keeps hydromt teed
        assert sc._wrapped_console("stdout", console) is first
        assert sc._wrapped_console("stderr", console) is not first
        # a DIFFERENT stream gets a new wrapper, or the first test to call
        # `stage_one` would own every later test's captured output
        assert sc._wrapped_console("stdout", io.StringIO()) is not first
    finally:
        sc._CONSOLE_WRAPPERS.clear()


def test_the_heartbeat_is_flushed_so_it_lands_while_the_stall_is_happening():
    """A worker is a separate process, block-buffered once the run is captured.

    The heartbeat's `... still running` is the ONLY thing on the console during
    a twenty-minute open, and a notice that arrives at process exit reports a
    stall the reader has already sat through. Warnings no longer take this path
    -- they are collected and attached to their slice -- so what is left here is
    exactly the text that must not wait.
    """

    class _Counting(io.StringIO):
        def __init__(self):
            super().__init__()
            self.flushes = 0

        def flush(self):
            self.flushes += 1

    sink = _Counting()
    console = sc._SliceConsole(sink)
    console.write("   ... cmip6_X_ssp585_r1i1p1f1: still running, 4m00s elapsed\n")
    assert sink.flushes == 1
    console.write("16:25:09 - fetch - fetching x\n")  # dropped, nothing to flush
    console.write("16:25:09 - fetch - WARNING - collected, not printed\n")
    assert sink.flushes == 1
