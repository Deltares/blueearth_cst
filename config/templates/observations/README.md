# Observation input templates

Header-only schemas for the two optional observation inputs of workflow 1
(`Snakefile_model_creation`). Copy them next to your basin data and point the
config at the copies by **absolute path** — real basin data lives in the project
folder, never in this repository (see `AGENTS.md` § Repo Map, the two-tier
`project_dir` rule).

Both inputs are optional. To run without them, set the config keys to `null`:

```yaml
workflows:
  model_creation:
    output_locations: null
    observations_timeseries: null
```

**Legacy spelling.** Older configs write an unquoted `None`, which YAML parses
to the Python **string** `"None"` rather than to null. That still works and is
not something you need to migrate: every consumer guards on file existence, so
a path that is not a file is skipped either way, and `plot_map.py` recognises
the string explicitly. Prefer `null` in new configs — it means what it looks
like, whereas a bare `None` reads as a null and is not one. That gap is what
produced the `gauges_None` layer-name bug the explicit check now guards.

## `output_locations.csv`

Gauge/output locations, **comma**-separated:

| Column | Meaning |
| --- | --- |
| `wflow_id` | integer station id — **start at 100** (100, 101, 102, …); the column names in the timeseries file must match these |
| `station_name` | free-text label used in figure titles and metric tables |
| `x`, `y` | longitude, latitude in EPSG:4326 |

**Why `wflow_id` starts at 100.** These ids end up as wflow output columns
(`Q_101`) and as burned-in values in the derived
`staticgeoms/subcatchment_<name>.geojson`, sharing a namespace with two other
numbering schemes: the model's own outlet subcatchment ids (large, from the
hydrography — e.g. `130000086`) and the positional station labels the evaluation
figures generate for outlets (`wflow_1`, `wflow_2`, …). Small ids make `Q_1`
ambiguous with the first positional outlet on sight. Starting at 100 keeps a
user gauge visibly a user gauge. It is a convention, not a validated
constraint — nothing rejects lower ids, so an existing dataset keeps working.

## `observations_timeseries.csv`

Observed discharge, **semicolon**-separated — deliberately a different
separator from `output_locations.csv`; both are read with explicit `sep=`
arguments, so keep each file's separator as shipped.

- First column `time`, ISO-8601 timestamps (`2000-01-01T00:00:00`).
- One further column per station, named by the **`wflow_id`** value from
  `output_locations.csv` — not by `station_name`.
- Missing values: leave the field empty.

The shipped header (`time;101;102`) is illustrative — replace `101` and `102`
with your own `wflow_id` values and add one column per station. The two files
must be changed **together**: the join is on these ids, and a mismatch drops the
station from the metrics without failing the run.

## What consumes these

`blueearth_cst/model/setup_gauges_and_outputs.py` (gauge setup) and
`blueearth_cst/model/plot_results.py` (evaluation figures and
`performance_metrics.csv`). Both check file existence before reading, so a
`null` or a legacy `None` skips the observation-dependent outputs rather than
failing the run. A configured path that is not a file is a different case and
now RAISES in rule 1.01 — a typo used to be skipped in silence, taking the
gauges, the signature plots and the metrics table with it.

## Where they end up

Both files are snapshotted into `<project_dir>/config/observations/` by rule
1.01, alongside the run's config (`config/runs/`), catalogs (`config/catalogs/`)
and build templates (`config/templates/`). They are referenced by **absolute
path** from wherever you keep them, so without that copy a finished project
could not say what it was evaluated against — the metrics table would cite
gauges and observations that exist only on the machine that ran it.
