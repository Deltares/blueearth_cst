# WF2 — rule-level overview (`Snakefile_climate_projections`)

Working aid for the planned efficiency/modularity rework of workflow 2. Records
**current** behavior, rule by rule: what each rule does, what it consumes, what
it writes, and which rule consumes that output.

Scope split — this file is strictly **rule/DAG level**. The behavioral contract
(owned config keys, the `precip`/`temp` unit split, `save_grids` semantics, the
known metadata regression, downstream-consumer semantics) lives in
`dev/workflows/climate_projections.md` and is **not** repeated here.

Grounded in `Snakefile_climate_projections`,
`blueearth_cst/projections/*.py`, `blueearth_cst/shared/merge_{logs,benchmarks}.py`,
`blueearth_cst/model/copy_config_files.py`, and the seed config
`config/workflows/snake_config_model_test.yml`.

Path shorthand used below:

- `PD` = `project.project_dir`
- `CPD` = `{PD}/climate_projections/{clim_project}` (seed: `.../cmip6`)
- `BD` = `{PD}/hydrology_model` (workflow-1 product)

---

## 1. Rule inventory

11 rules. "Banner" is the `W.NN` reference number in the message/log/benchmark
names — **definition order, not execution order**.

| Banner | Rule | Script | Fan-out |
| --- | --- | --- | --- |
| 2.00 | `all` | — | — |
| 2.01 | `fetch_gcm_raw` | `projections/fetch_gcm_raw.py` | `{series_key}` |
| 2.02 | `reduce_gcm_series` | `projections/get_stats_climate_proj.py` | `{series_key}` |
| 2.03 | `copy_config` | `model/copy_config_files.py` | — |
| 2.04 | `derive_change_factors` | `projections/derive_change_factors.py` | — (one job) |
| 2.06 | `plot_climate_proj_timeseries` | `projections/plot_proj_timeseries.py` | — (gather) |
| 2.07 | `gather_logs` | `shared/merge_logs.py` | — (gather) |
| 2.10 | `gather_benchmarks` | `shared/merge_benchmarks.py` | — (gather) |
| 2.11 | `extract_climate_grid` | `model/extract_historical_climate.py` | — (shared store) |

**Eight rules plus `all`**, matching the design's §8 count again. It briefly ran
to nine: `gather_raw_logs` arrived with the fetch/reduce split (design revision
6) after that count was written, and left when the two per-stage log gathers
collapsed into the single workflow-level `gather_logs`.

What went, and where it went:

| Retired | Replaced by |
| --- | --- |
| `monthly_stats_hist`, `monthly_stats_fut` | `reduce_gcm_series` over `{series_key}` (step 3) |
| `monthly_change`, `monthly_change_scalar_merge` | `derive_change_factors`, one job (step 4d) |
| `gather_stats_hist_logs`, `gather_stats_fut_logs` | `gather_series_logs` (step 3) |
| `gather_change_logs` | nothing — stage B is one job and writes its own log (step 4d) |
| `gather_series_logs`, `gather_raw_logs` | `gather_logs`, one merge for the whole workflow |
| the `ruleorder:` directive | nothing — it named two rules that no longer exist (step 4d) |

**Job count on the seed config** (3 models × 2 scenarios × 1 horizon = 9 series):
1 (2.03 `copy_config`) + 9 (2.01) + 9 (2.02) + 1 (2.04) + 1 (2.06)
+ 1 (2.07) + 1 (2.10) + 1 (2.11) = **24 jobs**, plus `all`.

Stage A dominates the count and is where the parallelism is; everything from
stage B down is single-job.

---

## 2. Rule detail

### 2.00 `all` — target aggregator

Declares the workflow's terminal targets. Note that the merged log and the
benchmark table are **targets**, not optional bookkeeping — WF2 is not complete
until 2.07 and 2.10 have run.

| Direction | Item | Producer |
| --- | --- | --- |
| in | `{CPD}/summary/{proj}_change_factors_{annual,monthly}.csv` | 2.04 |
| in | `{CPD}/plots/{proj}_change_factor_cloud.png` | 2.04 |
| in | `{CPD}/plots/{proj}_{precip,temp}_annual_absolute.png` | 2.06 |
| in | `{PD}/config/runs/snake_config_climate_projections.yml` | 2.01 |
| in | `{PD}/logs/wf2_climate_projections.log` | 2.07 |
| in | `{PD}/benchmarks/wf2_benchmarks.md` | 2.10 |

### 2.01 `fetch_gcm_raw` — `{series_key}`

The **only** rule that opens the remote store. Acquires one bbox+buffer, time-
windowed slice per `(model, experiment, member)` and writes it to local disk.

- **In:** `{CPD}/store_region.geojson` (the model-free delineated polygon).
- **Params:** `catalog_path`, `catalog_entry`, `member`, `variables`,
  `variable_units`, `buffer_degrees`, `acquisition_window`,
  `raw_digest_components`.
- **Out:** `update({CPD}/raw/{series_key}.nc)` — dims `(time, lat, lon)`, PERSISTENT.
- **Why the split exists:** measured 2026-07-30 — opening one source ~1142 s,
  transferring ~19 s, reducing ~0.2 s. `raw_digest_components` deliberately
  **excludes** the reducer hash, so editing a formula re-reads local disk instead
  of re-downloading nine sources.
- **Consumed by:** 2.02.

### 2.02 `reduce_gcm_series` — `{series_key}`

Collapses the slice to a basin scalar: area-weighted spatial mean per month.
Makes **no** network call — it reads 2.01's local file and checks the digest
recorded on it.

- **In:** `{CPD}/store_region.geojson`, `{CPD}/raw/{series_key}.nc`.
- **Params:** the identity set (`digest_components`, `acquisition_window`,
  `store_index`, `buffer_degrees`), plus `variables`, `variable_units`,
  `series_nc_out`.
- **Out:** `update({CPD}/scalar/{series_key}.nc)` — dims
  `(clim_project, model, scenario, member, time)`, PERSISTENT. Same filename as
  its raw slice: the directory carries the tier, the filename the identity.
- **Consumed by:** 2.04, 2.06, 2.07.

### 2.03 `copy_config`

Snapshots the run's `--configfile` YAML plus the CMIP6 data catalog, so a finished
project records what produced it.

- **In:** `config_path`. **Out:** `{PD}/config/runs/snake_config_climate_projections.yml`.
- Side effect, not declared: each catalog copied to `{PD}/config/catalogs/`.
- **Connections:** an isolated leaf consumed only by `all`; never gates compute.

### 2.04 `derive_change_factors` — ONE job, no fan-out

Stage B. Reads the explicit expanded scalar list, derives annual and monthly
change factors for every `(point, horizon)`, and writes every result artifact.

- **In:** the expanded `{CPD}/scalar/*.nc` set (never a glob — a model dropped
  from the config cannot rejoin through a leftover file), plus `store_region.geojson`.
- **Out:** `{CPD}/summary/{proj}_change_factors_{annual,monthly}.csv`,
  `{CPD}/summary/composition.csv`, `{CPD}/summary/provenance.json`,
  `{CPD}/report.md`, `{CPD}/plots/{proj}_change_factor_cloud.png`.
- **Job-internal:** the per-point change netCDFs and the wide
  `annual_change_scalar_stats_summary.nc` live in a `TemporaryDirectory`. The wide
  file is written and read back so the tidy table describes what was persisted;
  it is not an artifact (S8-05).
- **CSV number format:** all three CSVs go through `change_factor_table.csv_value`
  — floats fixed to `CSV_DECIMALS` (3) places, non-floats untouched. Serialization
  only: the rows stay exact in memory and `scalar/*.nc` keeps full precision, so
  this is not a partial revert of 5c's stored-series de-quantisation. It removes
  the 17-significant-digit reprs that made Excel prompt to convert the file on
  every open, and guarantees no cell is ever in exponent form. Both change-factor
  CSVs are `check_baseline.py` targets, so the format change required a scoped
  `record --workflow climate_projections`.

### 2.06 `plot_climate_proj_timeseries` — gather

Reopens all scalar series and renders the eight figures. Figure-only since S8-02.

- **In:** `{CPD}/summary/{proj}_change_factors_annual.csv` (declared, unread — an
  ordering edge), all historical and scenario `scalar/*.nc`.
- **Out (all declared since 7-i):** the eight
  `{CPD}/plots/{proj}_{precip,temp}_{annual,monthly}_{absolute,change}.png`.

### 2.07 `gather_logs`

**Every** WF2 rule that logs writes a part under `logs/_parts/`; this rule merges
all of them into ONE `logs/wf2_climate_projections.log` and then deletes the
parts, pruning the emptied dirs (so a clean full run leaves no `logs/_parts/`).
That is the only WF2 file left in `logs/` — the deal `benchmarks/wf2_benchmarks.md`
already had.

The merged file carries **one** provenance header (the per-part headers are
stripped: a near-identical three-line block per rule was bulk, not information),
then one `== W.NN  rule_name` banner per rule, and inside a fan-out rule's section
one `-- {series_key}` sub-header per member.

`params.rules` is an ordered list of rule **labels**, not part paths; `merge_logs`
lists each label's part dir to find its members, so a fan-out width lives only in
the rule that owns it. Scoping discovery to the label list is what keeps it from
being a glob: an orphan dir from a renamed rule (`test_local` still holds
`2.04_monthly_change/`) is not a label, so it is neither merged as a phantom
section nor deleted. Section order is **rule number**, matching the rule map and
the benchmark table, not execution order (2.11 runs first). `input:` is the
terminal artifact set, so the rule is scheduled after every logging rule.

The rule is shared verbatim by all three workflows — WF1 1.16
(`logs/wf1_model_creation.log`), WF3 3.13
(`{exp_dir}/logs/wf3_climate_experiment.log`) — differing only in the label list,
the parts dir and the output name.

Same partial-re-run caveat as 2.10: only the rules that re-ran have parts, so the
rewritten log marks the rest `# (no part from this run — rule was already up to date)`. The artifact
describes the run that produced it, not an accumulated history.

### 2.10 `gather_benchmarks`

Merges `benchmarks/_parts/**.tsv` into `benchmarks/wf2_benchmarks.md` (Markdown
table, `rule` column + `TOTAL` row). Takes a summary CSV and two figures as inputs
so it runs after everything else.

### 2.11 `extract_climate_grid`

The shared climate-store producer, declared identically in all three workflows and
pinned by `tests/test_climate_store_contract.py`. WF2 declares it to obtain
`store_region.geojson`, so the workflow no longer depends on wf1's model build.

---

## 3. DAG shape

```
        2.03 copy_config ──────────────────────────────────────────────┐
                                                                       │
        2.11 extract_climate_grid ─► store_region.geojson              │
                     │                                                 │
                     ▼                                                 │
   ┌─► 2.01 fetch_gcm_raw {series_key} ─► raw/{key}.nc                 │
   │        (the ONLY remote read)                                     │
   │             │                                                     │
   │             ▼                                                     │
   └─► 2.02 reduce_gcm_series {series_key} ─► scalar/{key}.nc          │
                 │                                                     │
                 ▼                                                     │
        2.04 derive_change_factors  (ONE job)                          │
                 ├─► summary/{proj}_change_factors_{annual,monthly}.csv│
                 ├─► summary/composition.csv, summary/provenance.json  │
                 ├─► report.md                                         │
                 └─► plots/{proj}_change_factor_cloud.png              │
                 │                                                     │
                 ▼ (ordering edge; the CSV is declared, unread)        │
        2.06 plot_climate_proj_timeseries ─► plots/*.png ──────────────┤
                 │                                                     │
                 ├─► 2.07 gather_logs ─► logs/wf2_climate_projections.log
                 │        (+ deletes logs/_parts/)                     │
                 │                                                     │
                 └─► 2.10 gather_benchmarks ─► benchmarks/wf2_benchmarks.md
                                                                       │
        (2.07 and 2.10 are `all` inputs too) ──────────────────────────┘
```

Both gathers hang off the same terminal artifact set, which is what schedules
them last; neither takes the parts it merges as `input:` (a `log:`/`benchmark:`
file is not a DAG node).

**Fan-out width on the seed config** (3 models × 3 experiments = 9 series):
9 (2.01) → 9 (2.02) → 1 (2.04) → 1 (2.06). Stage A fans out at full width — since
step 2b there is **no edge between series at all**, so the historical→future
ordering constraint is gone.

**No `temp()` intermediates remain in WF2.** `raw/` and `scalar/` are both
PERSISTENT with `update()`, which is load-bearing: Snakemake's `Job.prepare()`
removes existing outputs before every job, so without the flag the revalidate-and-
skip cache could never fire and every scheduled job would re-download. Stage B's
per-point files are job-internal (a `TemporaryDirectory`), not `temp()`.

**No `ruleorder:`.** It named two rules that no longer exist, and the merge that
retired them also removed what it insured against.

---

## 4. Observations relevant to efficiency / modularity

> **Historical.** These were written **before** the v2.0 rework, against the rule
> set §1's "what went, and where it went" table now lists as retired. Several
> describe rules that no longer exist (2.03 `monthly_stats_fut`, 2.05
> `monthly_change_scalar_merge`) or problems the rework fixed — the hist→fut
> ordering edge, the two-scripts-one-computation duplication, the `temp()`
> lifetimes. Kept as the record of what motivated the rework, **not** as a
> description of current behaviour: §§1–3 above are that. Do not read a rule
> number in this section as a live one.

Findings from the code as it stood pre-v2.0 — recorded to think against, not
proposals.

1. **2.03's dependency on 2.02's output is ordering-only.** The `__main__` block
   of `get_stats_climate_proj.py` reads only `input.region_path`; the historical
   netCDF is never opened by the future job. The edge serializes hist→fut across
   the entire fan-out for no data reason. *Hypothesis to test, not an established
   cause:* the script does `os.mkdir(folder_out)` guarded by `os.path.exists`
   (lines 179–180) rather than `makedirs(exist_ok=True)`, so concurrent hist/fut
   jobs would race on creating `{CPD}`.

2. **2.02 and 2.03 are the same script with different params.** The only
   structural difference is the output naming, which is inconsistent between
   them: `historical_stats_time_{model}.nc` (underscore separator, no scenario)
   vs `stats_time-{model}_{scenario}.nc` (dash separator). Any consolidation into
   a single `{model}×{scenario}` rule has to reconcile that naming first — the
   inconsistency is what forces two rules, not the computation.

3. **`members` is looped inside the script, not a wildcard.** The per-member
   remote reads (the dominant cost — remote GCS/zarr fetch + `.load()`) run
   serially within one job and cannot be scheduled by Snakemake. With the seed's
   single member this is invisible; with several it caps parallelism.

4. **`ancient()` on intra-workflow temps.** 2.04 marks both of its stats inputs
   `ancient`, and 2.05 marks its whole `expand()` `ancient`. Unlike the
   documented `region.geojson` case (a deliberate cross-workflow staleness
   exemption), these suppress rebuild propagation *inside* WF2: regenerated stats
   will not retrigger change factors, and regenerated change factors will not
   retrigger the merge. Combined with `temp()`, an incremental re-run is
   effectively an all-or-nothing re-run.

5. **A whole file-dependency layer sits outside the DAG when `save_grids: true`.**
   The grid netCDFs (`historical_stats_{model}.nc`, `stats-{model}_{scenario}.nc`,
   `monthly_change_mean_grid-*.nc`) are written by the scripts but declared
   nowhere; 2.04 and 2.06 receive their paths through `params`. Snakemake neither
   schedules, tracks, nor cleans them, and a missing one fails at runtime rather
   than at DAG build.

6. **~~The three `gather_*_logs` rules are identical modulo wildcards.~~
   RESOLVED.** They were the same script and shape differing only in which
   `expand()` fed the barrier and which `_parts` dir fed their part list — which
   is why they collapsed into the single `gather_logs` (2.07), now shared by all
   three workflows.

7. **Both gather rules are invisible in the cost picture.** 2.07 and 2.10 declare
   neither `log:` nor `benchmark:`, so their contribution never reaches
   `wf2_benchmarks.md`. Both delete the parts they merged, so a later partial
   re-run produces a table — and now a log — covering only the rules that re-ran.

8. **`{model}` contains a `/`** (e.g. `NOAA-GFDL/GFDL-ESM4`), so the vendor
   becomes a real path segment in every output filename, log part, and benchmark
   part. Relevant to any change touching path templates or wildcard constraints.

9. **2.06 is a monolith.** One script does: reopen all series → anomaly
   statistics → render 8 PNGs → optionally render the gridded maps. It is the
   last consumer of the `temp()` series (see §3), and the natural split point if
   figure generation should become independently re-runnable. It no longer writes
   a merged timeseries netCDF — S8-02 deleted `gcm_timeseries.nc`, leaving 2.06 a
   figure-only rule.

10. **Mislabeled outputs.** The precip anomaly figure is saved without an
    extension (`plots/precipitation_anomaly_projections_{n}`, matplotlib appends
    `.png`), while its temp counterpart passes `.png` explicitly.

11. **Doc discrepancy (not fixed here):** `dev/workflows/climate_projections.md`
    lists the config snapshot at `{PD}/config/snake_config_climate_projections.yml`;
    the Snakefile writes `{PD}/config/runs/snake_config_climate_projections.yml`.
    The Snakefile is authoritative.
