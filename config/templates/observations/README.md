# Observation input templates

Header-only schemas for the two optional observation inputs of workflow 1
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

**Config migration.** `shared.basin.gauge_points` replaces
`workflows.model_creation.output_locations` because the points now control the
model-neutral basin/subbasin layout as well as Wflow outputs. The old key is
accepted for one compatibility release with a warning. If both keys are set,
they must name the same path; conflicting values fail at parse time.

Older configs may also write an unquoted `None`, which YAML parses to the Python
**string** `"None"` rather than to null. That remains an accepted unset spelling
during the compatibility release. Prefer a real YAML `null` in new configs.

## `output_locations.csv`

Gauge/output locations, **comma**-separated:

| Column | Meaning |
| --- | --- |
| `wflow_id` | optional integer station id; when supplied it must exactly match the deterministic ID generated from the resolved basin/subbasin hierarchy |
| `station_name` | free-text label used in figure titles and metric tables |
| `x`, `y` | longitude, latitude in EPSG:4326 |
| `location_role` | optional role: `control` (default) defines a subbasin; `observation` is tracked without controlling delineation |

Primary locations inherit their subbasin ID: basin 1 subbasins are 101, 102,
and so on, giving location codes such as `B001-S01-L01`. Additional
non-controlling locations use a reserved range beginning at 1,000,000. The
current Gabon IDs 101–104 remain valid only when the resolved hierarchy assigns
the same stations to `B001-S01` through `B001-S04`; otherwise preparation fails
with an explicit old-ID → resolved-ID crosswalk rather than silently preserving
stale IDs.

## `observations_timeseries.csv`

Observed discharge, **semicolon**-separated — deliberately a different
separator from `output_locations.csv`; both are read with explicit `sep=`
arguments, so keep each file's separator as shipped.

- First column `time`, ISO-8601 timestamps (`2000-01-01T00:00:00`).
- One further column per station, named by the resolved **`wflow_id`** value in
  `spatial/location_registry.csv` — not by `station_name`.
- Missing values: leave the field empty.

The shipped header (`time;101;102`) is illustrative — replace `101` and `102`
with your own `wflow_id` values and add one column per station. The two files
must be changed **together**: the join is on these ids, and a mismatch drops the
station from the metrics without failing the run.

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
