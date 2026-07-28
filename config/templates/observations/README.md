# Observation input templates

Header-only schemas for the two optional observation inputs of workflow 1
(`Snakefile_model_creation`). Copy them next to your basin data and point the
config at the copies by **absolute path** — real basin data lives in the project
folder, never in this repository (see `AGENTS.md` § Repo Map, the two-tier
`project_dir` rule).

Both inputs are optional. To run without them, leave the config keys at the
`None` sentinel:

```yaml
workflows:
  model_creation:
    output_locations: None
    observations_timeseries: None
```

**Write `None` unquoted and exactly so.** Unquoted `None` parses to the Python
string `"None"`, not YAML `null`; the consumers guard on file existence, and a
real `null` is a different code path. Do not "fix" it to `null` or `~`.

## `output_locations.csv`

Gauge/output locations, **comma**-separated:

| Column | Meaning |
| --- | --- |
| `wflow_id` | integer station id; the column names in the timeseries file must match these |
| `station_name` | free-text label used in figure titles and metric tables |
| `x`, `y` | longitude, latitude in EPSG:4326 |

## `observations_timeseries.csv`

Observed discharge, **semicolon**-separated — deliberately a different
separator from `output_locations.csv`; both are read with explicit `sep=`
arguments, so keep each file's separator as shipped.

- First column `time`, ISO-8601 timestamps (`2000-01-01T00:00:00`).
- One further column per station, named by the **`wflow_id`** value from
  `output_locations.csv` — not by `station_name`.
- Missing values: leave the field empty.

The shipped header (`time;1;2`) is illustrative — replace `1` and `2` with your
own `wflow_id` values and add one column per station.

## What consumes these

`blueearth_cst/model/setup_gauges_and_outputs.py` (gauge setup) and
`blueearth_cst/model/plot_results.py` (evaluation figures and
`performance_metrics.csv`). Both check file existence before reading, so an
absent or `None`-sentinel path skips the observation-dependent outputs rather
than failing the run.
