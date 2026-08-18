"""Stage CMIP6 slices for one region, outside any project.

What it does
------------
The same thing WF2's rule 2.04 does — open the remote CMIP6 store, clip to a
polygon plus a buffer, slice the acquisition window, write a small netCDF — but
driven by a YAML file and writing wherever you point it, with no project and no
Snakemake run.

**The slices are WF2-cache-compatible.** Each file is stamped with the same
``cst_raw_digest`` the pipeline computes, so dropping one into a project's
``data/climate/projections/<clim_project>/raw/`` makes WF2 treat it as a cache
hit and skip the download. That is the point of the tool: WF2's raw cache is
project-scoped, so every new project on the same basin re-pays the remote open,
measured at **1142 s per source** (``fetch_gcm_raw`` docstring). Stage once
here, reuse everywhere.

For that to work the REGION must be the same polygon the project uses — the
digest covers it (``series_identity.region_fingerprint``). A different polygon
is a different slice by design, and its file will simply be re-fetched by the
project rather than silently accepted.

Why this is not a `stage_data.py` dataset type
----------------------------------------------
``stage_data.py`` mirrors a directory tree under one ``source_root`` by joining
``path``. CMIP6 is ``(model, scenario, member, variable)`` tuples resolved
through a GENERATED catalog with ``{member}`` / ``{variable}`` placeholders, on
an object store. The addressing has nothing in common, so this is a sibling
tool rather than a new ``type:``.

There is exactly ONE fetch implementation: ``fetch_gcm_raw.fetch_raw_slice``,
which WF2's rule 2.04 also calls. This module contributes the config and the
digest recipe, never a second way to read the store.

Configuration
-------------
All knobs live in a YAML file (default: ``dev/scripts/stage_cmip6.yml``)::

    region: C:/basins/gabon/region.geojson
    target_root: C:/data/cmip6_slices
    catalog: config/catalogs/cmip6_data.yml
    store_index: config/catalogs/cmip6_store_index.json
    clim_project: cmip6
    buffer_degrees: 1.0
    models:    [NOAA-GFDL/GFDL-ESM4, INM/INM-CM4-8]
    scenarios: [historical, ssp245]
    members:   [r1i1p1f1]
    variables:                       # POST-rename names, not pr/tas
      precip: {units: kg m-2 s-1}
      temp:   {units: K}

The acquisition window is NOT a knob: it is derived per scenario by
``series_identity.acquisition_window``, exactly as the Snakefile derives it, so
a staged slice covers the window the pipeline expects.

Usage
-----
    python dev/scripts/stage_cmip6.py
    python dev/scripts/stage_cmip6.py --config my_basin.yml
    python dev/scripts/stage_cmip6.py --dry-run          # list jobs, fetch none
    python dev/scripts/stage_cmip6.py --models NOAA-GFDL/GFDL-ESM4
    python dev/scripts/stage_cmip6.py --workers 8        # 4 slices at once by default
    python dev/scripts/stage_cmip6.py --workers 1        # serial, for a real traceback

Slices are staged in parallel PROCESSES because the cost is the remote open, not
compute -- see `DEFAULT_WORKERS` for the measurement and for why threads are the
wrong tool here. Progress is reported in completion order, and one unavailable
source is reported without ending the run.

Not part of a run: this is an authoring/staging helper
(see AGENTS.md, "Three homes for executables").
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from blueearth_cst.projections import series_identity as _si  # noqa: E402
from blueearth_cst.projections.fetch_gcm_raw import fetch_raw_slice  # noqa: E402
from blueearth_cst.shared.snake_utils import format_elapsed  # noqa: E402

CONFIG_DEFAULT = Path(__file__).resolve().parent / "stage_cmip6.yml"

#: Mirrors `analyze_projections.smk:402`'s REGION_BUFFER_DEGREES. Duplicated
#: rather than imported because a Snakefile is not an importable module. It IS a
#: digest component, so a value differing from the pipeline's yields slices the
#: pipeline will not accept -- which `tests/test_stage_cmip6.py`'s fixture case
#: catches, since it recomputes against files WF2 actually wrote.
DEFAULT_BUFFER_DEGREES = 1.0

#: Mirrors the `clim_project` config key. Only `cmip6` has a generated catalog
#: today, but the grammar below is the Snakefile's, not a cmip6 literal.
DEFAULT_CLIM_PROJECT = "cmip6"


def series_key(clim_project, model, experiment, member):
    """The pipeline's filename stem for one slice.

    Verbatim from `analyze_projections.smk`'s `SERIES` comprehension — the
    slash in a model id becomes an underscore. Staged files carry this name so
    they can be copied into `raw/` unchanged.
    """
    return f"{clim_project}_{model.replace('/', '_')}_{experiment}_{member}"


def catalog_entry_name(clim_project, model, experiment):
    """The catalog entry, `{member}` placeholder still unresolved.

    Also verbatim from the Snakefile: the generated catalog expands the member
    at generation time, so the entry NAME carries it and `fetch_raw_slice`
    resolves it through the catalog's own grammar.
    """
    return f"{clim_project}_{model}_{experiment}_{{member}}"


def digest_components(cfg, model, experiment, member):
    """Rebuild the pipeline's raw-digest components for one slice.

    This is the one place the tool must agree with
    `analyze_projections.smk::series_digest_components`, and
    `tests/test_stage_cmip6.py` pins that agreement rather than trusting it.

    `reducer_module_hash` is passed as the empty string on purpose:
    `raw_components` strips it, and inventing a value here would suggest the
    raw slice depends on a reduction it has never seen.
    """
    entry_name = catalog_entry_name(cfg["clim_project"], model, experiment)
    entry = _si.load_catalog_entry(cfg["catalog"], entry_name)
    return _si.raw_components(
        _si.digest_components(
            catalog_entry=entry_name,
            entry=entry,
            members=[member],
            pins_by_member={
                member: _si.load_pins(cfg["store_index"], entry_name, member)
            },
            buffer_degrees=cfg["buffer_degrees"],
            variable_spec=list(cfg["variables"]),
            experiment=experiment,
            reducer_module_hash="",
        )
    )


def load_config(path):
    """Read the YAML and apply defaults; raise on anything required and absent."""
    with open(path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    for key in ("region", "target_root", "models", "scenarios", "members", "variables"):
        if not cfg.get(key):
            raise SystemExit(f"{path}: '{key}' is required and missing or empty")

    cfg.setdefault("clim_project", DEFAULT_CLIM_PROJECT)
    cfg.setdefault("buffer_degrees", DEFAULT_BUFFER_DEGREES)
    cfg.setdefault("catalog", str(_REPO_ROOT / "config/catalogs/cmip6_data.yml"))
    cfg.setdefault(
        "store_index", str(_REPO_ROOT / "config/catalogs/cmip6_store_index.json")
    )
    if not Path(cfg["region"]).is_file():
        raise SystemExit(f"{path}: region not found: {cfg['region']}")
    return cfg


def plan(cfg):
    """Every (model, scenario, member) slice the config asks for, in a stable order."""
    jobs = []
    for model in cfg["models"]:
        for experiment in cfg["scenarios"]:
            for member in cfg["members"]:
                key = series_key(cfg["clim_project"], model, experiment, member)
                jobs.append(
                    {
                        "key": key,
                        "model": model,
                        "experiment": experiment,
                        "member": member,
                        "out": str(Path(cfg["target_root"]) / f"{key}.nc"),
                    }
                )
    return jobs


#: Slices staged at once. PROCESSES, not threads, and the reason is measured:
#: `fetch_gcm_raw`'s benchmark puts one source's OPEN at 1142 s against 19 s to
#: transfer and 0.2 s to reduce, so the job is dominated by waiting on store
#: metadata -- and a worker's whole startup cost, importing geopandas + hydromt
#: + xarray, was measured at 6.7-7.4 s. Paying 7 s to remove every thread-safety
#: question is a 0.6 % overhead on the thing being parallelised.
#:
#: Threads were the alternative and are rejected on THIS workload: netCDF4/HDF5
#: serialises writes behind a global lock (the same lock `fetch_gcm_raw`'s
#: eager-`load()` comment records a deadlock against), and hydromt, GDAL and
#: gcsfs would all have to be thread-safe together. Each slice writes its own
#: file and shares nothing, so processes are the natural fit.
#:
#: 4 rather than cpu_count(): the limit is the remote store and ~311 MiB of
#: resident memory per worker, not cores.
DEFAULT_WORKERS = 4


def stage_one(cfg, job):
    """Stage a single slice. Runs in a worker process; returns (key, error, seconds).

    Returns rather than raises so one unavailable source cannot end a staging
    run that may have hours of other work in it -- a model missing a member is
    an ordinary catalog fact. The error is stringified HERE because an arbitrary
    exception may not survive the trip back to the parent process intact.

    The duration is measured INSIDE the worker, so it is the slice's own cost
    and excludes process startup and queue wait. That is the number worth
    reporting: it is dominated by the remote store OPEN (benchmarked at 19 s to
    transfer against a far larger open), so a slow slice means a slow store, not
    a slow machine or a saturated pool.
    """
    started = time.perf_counter()
    units = {
        name: (spec or {}).get("units", "") for name, spec in cfg["variables"].items()
    }
    try:
        fetch_raw_slice(
            region_path=cfg["region"],
            raw_nc_out=job["out"],
            catalog_path=cfg["catalog"],
            catalog_entry=catalog_entry_name(
                cfg["clim_project"], job["model"], job["experiment"]
            ),
            member=job["member"],
            variables=list(cfg["variables"]),
            variable_units=units,
            buffer=cfg["buffer_degrees"],
            acquisition_window=tuple(_si.acquisition_window(job["experiment"])),
            components=digest_components(
                cfg, job["model"], job["experiment"], job["member"]
            ),
        )
    except Exception as exc:  # noqa: BLE001 -- reported per slice, never fatal
        return job["key"], f"{type(exc).__name__}: {exc}", time.perf_counter() - started
    return job["key"], None, time.perf_counter() - started


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Stage WF2-cache-compatible CMIP6 slices for one region."
    )
    parser.add_argument("--config", default=str(CONFIG_DEFAULT))
    parser.add_argument("--region", help="override the config's region polygon")
    parser.add_argument("--target-root", help="override where slices are written")
    parser.add_argument("--models", nargs="+", help="override the model list")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list the slices and their destinations; open nothing",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            f"slices to stage at once (default: {DEFAULT_WORKERS}). "
            "1 runs in this process, which is what to use when a failure needs "
            "a readable traceback"
        ),
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.region:
        cfg["region"] = args.region
    if args.target_root:
        cfg["target_root"] = args.target_root
    if args.models:
        cfg["models"] = args.models

    jobs = plan(cfg)
    print(f"region      {cfg['region']}")
    print(f"target      {cfg['target_root']}")
    print(f"buffer      {cfg['buffer_degrees']} deg")
    print(f"variables   {', '.join(cfg['variables'])}")
    print(f"slices      {len(jobs)}")

    if args.dry_run:
        for job in jobs:
            state = "present" if Path(job["out"]).exists() else "would fetch"
            print(f"  {state:11s} {job['key']}")
        return 0

    os.makedirs(cfg["target_root"], exist_ok=True)
    workers = max(1, min(args.workers, len(jobs)))
    print(f"workers     {workers}")

    done = 0
    failed = {}
    seconds = {}
    wall_started = time.perf_counter()

    out_by_key = {job["key"]: Path(job["out"]) for job in jobs}

    def _report(key, error, elapsed):
        """One line per finished slice, in COMPLETION order.

        Completion order, not config order: with several workers in flight the
        useful signal is which slice just landed, and a staging run is long
        enough that waiting to sort would mean a silent console for minutes.
        """
        nonlocal done
        done += 1
        seconds[key] = elapsed
        if error:
            failed[key] = error
            print(
                f"[{done}/{len(jobs)}] FAILED  {key}  ({format_elapsed(elapsed)})",
                file=sys.stderr,
            )
            print(f"          {error}", file=sys.stderr)
        else:
            path = out_by_key[key]
            mb = path.stat().st_size / 1e6 if path.exists() else 0.0
            print(
                f"[{done}/{len(jobs)}] ok      {key}  "
                f"({mb:.2f} MB, {format_elapsed(elapsed)})"
            )

    if workers == 1:
        # Deliberately NOT a one-worker pool: staying in-process keeps
        # tracebacks and any debugger attached to the real failure.
        for job in jobs:
            _report(*stage_one(cfg, job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(stage_one, cfg, job): job["key"] for job in jobs}
            for future in as_completed(futures):
                try:
                    _report(*future.result())
                except Exception as exc:  # noqa: BLE001 -- a worker died outright
                    # Distinct from a slice that failed: the worker never got to
                    # return, so `stage_one`'s own handler never ran. Most likely
                    # a killed process (memory) rather than a bad source.
                    key = futures[future]
                    # No slice duration exists -- the worker never returned one.
                    _report(key, f"worker died: {type(exc).__name__}: {exc}", 0.0)

    wall = time.perf_counter() - wall_started
    slice_total = sum(seconds.values())

    print("")
    print(f"wall        {format_elapsed(wall)}")
    if seconds:
        # Slice time SUMMED across workers, against wall clock. The ratio is the
        # speed-up the pool actually delivered -- worth printing because this
        # workload is bound by the remote store's open, so more workers help
        # until the store rate-limits and then stop helping. A ratio well below
        # `workers` says adding more will not pay.
        print(
            f"slice time  {format_elapsed(slice_total)} summed over {workers} worker(s)"
        )
        if wall > 0:
            print(f"speed-up    {slice_total / wall:.1f}x")
        slowest = max(seconds.items(), key=lambda kv: kv[1])
        print(f"slowest     {slowest[0]}  ({format_elapsed(slowest[1])})")

    if failed:
        # Repeated at the end, sorted, because the per-slice rows above are in
        # completion order and scroll past: on a long run the summary is the
        # only place the full failure set is legible.
        print("", file=sys.stderr)
        print(f"{len(failed)} of {len(jobs)} failed:", file=sys.stderr)
        for key, error in sorted(failed.items()):
            print(f"  {key}", file=sys.stderr)
            print(f"    {error}", file=sys.stderr)
        return 1
    print(f"staged {len(jobs)} slice(s) into {cfg['target_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
