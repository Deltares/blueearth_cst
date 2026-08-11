# Config templates

Two different kinds of file share this directory. The distinction matters,
because only one kind is read by a running workflow.

## Consumed by the pipeline

Referenced from a config and read by a rule. Changing one changes a run.

| File | Consumer |
| --- | --- |
| `wflow_build_model.yml` | `Snakefile_model_creation` — default for `workflows.model_creation.model_build_config`; rule 1.06 `prepare_spatial_maps` and rule 1.07 `build_wflow_model` |
| `wflow_update_waterbodies.yml` | `Snakefile_model_creation` — default for `waterbodies_config`; rule 1.08 `add_reservoirs_lakes_glaciers` |
| `weathergen_config.yml` | `Snakefile_climate_experiment` — `default_config` for rule 3.10 `prepare_weathergen_config` |

## Scaffolds you copy

Never read by a run. Copy, fill in, point a config at your copy.

| File | Purpose |
| --- | --- |
| `snake_config.template.yml` | The annotated starting point for a new project's config. A filled-in worked example is `test_case/snake_config_model_test.yml`. |
| `output_locations_template.csv` | Header-only schema for gauge/output locations |
| `observed_daily_discharge_template.csv` | Header-only schema for observed discharge |
| `archive/` | Unmaintained single-workflow configs; see its own README |

`wflow_sbm.toml` sits here as a **reference copy only** — no Snakefile, script or
test reads it. Rule 1.07 has hydromt generate the project's own TOML from
hydromt_wflow's defaults. Treat this file as documentation, and expect it to lag:
measured 2026-08-10, it was 126 lines against the 149 a real build emitted.

**Rename after copying.** Layer names inside the model are derived from your
file's basename (`blueearth_cst/shared/gauges.py`), so a file still called
`output_locations_template.csv` yields layers named `output-locations-template`.
Drop the `_template` suffix when you copy.

---

# Why the template's defaults are what they are

`snake_config.template.yml` states only types and allowed values. The reasoning
behind the non-obvious defaults lives here.

### `historical_window` — minimum 16 years

Enforced at WF1 parse time and again at extraction. weathergenr's wavelet
decomposition needs at least 16 annual observations, so a shorter record cannot
support a stress test at all. The floor is `constraints.min_historical_years` in
`config/advanced_settings.yml` and is **not** overridable per project. This
window was 6 years until 2026-08-01 and would now be rejected.

### `members` — why four labels, not one

Not every model publishes `r1i1p1f1`. CNRM-\*, MIROC-ES2L, UKESM1-0-LL and
MCM-UA-1-0 use `f2`; HadGEM3-GC31-\* use `f3`; CanESM5-CanOE uses `p2f1`. Pinning
`r1i1p1f1` alone silently drops them — 38 of 45 usable models. A label a model
does not publish is simply skipped, because the catalog declares exactly what
exists.

The four shipped labels were chosen empirically over the generated catalog: they
reach 45 models across the tier-1 SSPs. Adding further physics variants (`p3f1`,
`p5f1`, `r1i1000p1f1`, …) reaches *fewer*, because they make a model's historical
and future member sets differ. When those differ, the change-factor stage raises
`asymmetric hist/clim members` rather than quietly using the intersection. With
this list that affects CAMS-CSM1-0, CanESM5, EC-Earth3 (ssp245), GISS-E2-1-G,
GISS-E2-1-H, IPSL-CM6A-LR (ssp434/460), MCM-UA-1-0 and NorESM2-LM (ssp245) — for
those, narrow `members:` to the one label the model shares across your scenarios.

Per-model member counts: `dev/reference/workflows/wf2-cmip6-monthly-members.csv`.

### `historical_year_range` — why 1985–2014

Thirty years ending at the last year the CMIP6 historical experiment covers
(owner ruling OQ-4, 2026-07-29).

- 30 years is the WMO climatological normal, and the workflow warns below 20. A
  shorter reference makes every derived statistic noisier and the tail quantiles
  unsupportable — which is why those are opt-in via `stats:`.
- 2014 ends the historical experiment. Asking for more is not an error: the
  window is **clipped** to 2014 and the run says so on stderr. Scenario data is
  never spliced in to fill the gap, so the extra years simply do not arrive.
- The range is inclusive, so 1985–2014 is thirty calendar years — and with the
  default `start_month_hyd_year: Jan`, thirty complete hydrological years. Any
  other start month yields 29, the partial years at both ends dropped. Every
  artifact reports the effective window and count beside the nominal one.

### `experiment_name` — why it is absent rather than set

Left unset, WF3 names the directory from the project's own name plus the date the
experiment was first created, and later runs **reuse** that directory rather than
minting today's date — which is what keeps incremental reruns idempotent. A
placeholder value in the template would instead put every project copied from it
into one shared `experiments/experiment/`.

To pin a deliberate name, run this once before the first climate-experiment run:

```console
pixi run python scripts/suggest_experiment_name.py <your config>
pixi run python scripts/suggest_experiment_name.py <your config> --name dry_scenario
```

It reserves the directory atomically, versions a generated collision to `_v2`,
and refuses to overwrite a name already set — which would strand a completed
experiment's outputs under a name nothing points at.

### `julia_threads` — how it interacts with `--cores`

Wflow parallelizes over grid **cells**, so raising it pays on a large basin and
does nothing on a small one. It is not Snakemake's `--cores`: the two multiply,
so keep `--cores N × julia_threads <= logical CPUs`.

### `batch_size` / `batch_size_max`

WF3 groups stress-test members into batches for the Wflow run. Disk is the
binding constraint on large `RLZ_NUM × ST_NUM` sweeps, because concurrent batches
are resident at once — so `batch_size_max` (default 8) bounds the footprint,
while an explicit `batch_size` wins outright. Both fail at parse time, naming the
offending key, if set below 1.

### `hydrography` / `basin_index`

Catalog **entry names**, not paths. They must match `setup_basemaps` in the
`model_build_config` template, or rule 1.02 fails loudly naming both files and
both values.

---

# Observation inputs

The two CSV scaffolds above are the optional observation inputs of workflow 1
(`Snakefile_model_creation`). Copy them next to your basin data and point the
config at the copies by **absolute path** — real basin data lives in the project
folder, never in this repository (see `AGENTS.md` § Repo Map, the two-tier
`project_dir` rule).

Both inputs are optional. To run without them, set the config keys to `null`:

```yaml
shared:
  basin:
    gauge_points: null
workflows:
  model_creation:
    observations_timeseries: null
```

**They work as a pair.** `observations_timeseries` without `gauge_points` has
nothing to key against: the series columns are matched by resolved `wflow_id`,
and those ids only exist once gauge points have driven the delineation.

**Config migration — the old key no longer works on its own.**
`shared.basin.gauge_points` replaces `workflows.model_creation.output_locations`
because the points now control the model-neutral basin/subbasin layout as well
as Wflow outputs, and **only the canonical key reaches the rule that delineates
it** (1.03 `delineate_spatial_units`, whose params are `shared.basin` alone per
ADR 0003 §8b — it is declared by all three workflows and the other two carry no
`workflows.model_creation` section).

A config that sets ONLY the legacy key therefore fails at parse time with the
migrated key spelled out. This used to be a `FutureWarning` that returned the
path anyway, which was worse than useless: the points still reached the
evaluation rule, so delineation quietly used the automatic fallback and the run
failed a whole model build later, comparing observation station IDs against a
registry built without them. If both keys are set they must name the same path,
so a staged migration can carry both; conflicting values fail at parse time.

Older configs may also write an unquoted `None`, which YAML parses to the Python
**string** `"None"` rather than to null. That remains an accepted unset spelling
during the compatibility release. Prefer a real YAML `null` in new configs.

## `output_locations_template.csv`

Gauge/output locations, **comma**-separated:

| Column | Meaning |
| --- | --- |
| `wflow_id` | optional integer station id; when supplied it must exactly match the deterministic ID generated from the resolved basin/subbasin hierarchy |
| `station_name` | free-text label used in figure titles and metric tables |
| `x`, `y` | longitude, latitude in EPSG:4326 |
| `location_role` | optional role: `control` (default) defines a subbasin; `observation` is tracked without controlling delineation |

### How `wflow_id` is built (changed 2026-08-06 — **existing files must be renumbered**)

```
wflow_id = basin_id*1000 + local_subbasin_number*10 + m
```

`m = 0` for the subbasin's own primary location and `1`–`9` for additional
points inside it. Basin 1 reads `1010, 1011, 1020, 1030…`; basin 2 reads
`2010, 2011, …`. Ids therefore group by basin, order by subbasin, and keep the
subbasin legible in the flat integer.

**This replaces the previous scheme, and old files will not work.** Before
2026-08-06 a primary location took its `subbasin_id` verbatim (`101`, `102`, …)
while any additional location took `1_000_000 + subbasin_id*100 + n` — so a
station and its neighbour sat four orders of magnitude apart in the same column.

| location | before | after |
| --- | --- | --- |
| basin 1, subbasin 1, primary | `101` | `1010` |
| basin 1, subbasin 1, second point | `1010102` | `1011` |
| basin 1, subbasin 2, primary | `102` | `1020` |

**What you have to do.** Both files are keyed by `wflow_id`, so **renumber the
locations file's `wflow_id` column and the discharge file's column headers
together.** Neither failure is silent: a pinned `wflow_id` that no longer matches
the resolved hierarchy stops preparation with an explicit old-ID → resolved-ID
crosswalk, and an observation header carrying ids the registry does not know
fails the WF1 header check by name.

The simplest migration is to **delete the `wflow_id` column**, run WF1 once, and
read the assigned ids out of `data/spatial/location_registry.csv` — the column is
optional, and pinning it is only worth doing when you need the ids to stay fixed
across rebuilds.

`location_code` is unchanged (`B001-S01-L01`): codes are for reading, `wflow_id`
is the integer for joining and for scanning a CSV header.

## `observed_daily_discharge_template.csv`

Observed discharge, **semicolon**-separated — deliberately a different
separator from the locations file; both are read with explicit `sep=`
arguments, so keep each file's separator as shipped.

- First column `time`, ISO-8601 timestamps (`2000-01-01T00:00:00`).
- One further column per station, named by the resolved **`wflow_id`** value in
  `spatial/location_registry.csv` — not by `station_name`.
- Missing values: leave the field empty.

The shipped header (`time;1010;1020`) is illustrative — replace `1010` and `1020`
with your own `wflow_id` values and add one column per station. The two files
must be changed **together**. Before plotting, Workflow 1 checks the raw header
against `spatial/location_registry.csv`: duplicate or unknown IDs fail
explicitly, as does a missing series for any user-provided control or
observation location. Automatically generated outlets may be included but do
not require an observation series.

## What consumes these

The spatial-preparation phase reads gauge points to control subbasin
delineation and writes `spatial/location_registry.csv`. The Wflow adapter then
uses that registry for gauge/output IDs; `plot_results.py` uses the same IDs for
observation joins. A configured path is a declared Snakemake input, so a typo
fails as a missing input instead of silently dropping observation outputs.

## Where they end up

Both files are snapshotted into `<project_dir>/config/observations/` by rule
1.01, alongside the run's config (`config/runs/`), catalogs (`config/catalogs/`)
and build templates (`config/templates/`). They are referenced by **absolute
path** from wherever you keep them, so without that copy a finished project
could not say what it was evaluated against — the metrics table would cite
gauges and observations that exist only on the machine that ran it.
