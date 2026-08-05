# WF3 output change requests

Opened 2026-08-05, from observations on the first full WF3 run of the migrated
R9 tree. **Collect-then-implement**: the requests are specified here and built as
one batch, so the contract, the validators, the migration note and the baseline
re-record move once rather than per request.

Status: **CR-1 landed. CR-2 specified, NOT implemented — open questions below.**

---

## CR-1 — indicator-table axis columns (LANDED)

`tavg` → `temp_change`, `prcp` → `precip_change`. Full record:
`migration_indicator-axis-columns.md`. Commits `ba6f2b2`, `e159ca7`.

**Owes two gates**, both cleared by one WF3 run from the primary checkout:
`pytest tests/` (the fixture tree still holds pre-rename headers, so
`test_hm7_integration` and the 12 `test_gauge_identity_integration` cases fail
*correctly*) and `check_baseline.py check`. Tracked as followups R9-2.

**CR-2 supersedes this table's shape entirely.** CR-1 is not wasted work — it
fixed the vocabulary, and CR-2 carries the corrected names forward — but if CR-2
lands before the CR-1 re-run, do the WF3 run **once, after both**, and record a
single baseline move. Do not re-run twice.

---

## CR-2 — `q_indicators.csv` wide → long

### The shape

**One table per variable** (ruled 2026-08-05), named from the token table below:
`q_indicators.csv`, `aet_indicators.csv`, `recharge_indicators.csv`,
`overland_flow_indicators.csv`, `snow_indicators.csv`, `precip_indicators.csv`.
Only the variables in `wflow_outvars` are emitted. `q_indicators.csv` keeps its
current name.

**Six fixed columns**, independent of location count. There is no `variable`
column — the filename carries it, and so does the composite `metric`:

| Column | Domain |
| --- | --- |
| `metric` | **composite** `<variable>_<statistic>` — e.g. `q_mean_annual_7day_min` |
| `temp_change` | perturbation axis, absolute degC |
| `precip_change` | perturbation axis, relative % |
| `realization_id` | **integer**; `1..RLZ_NUM`, or `0` = pooled over all realizations |
| `location` | **bare** gauge id — `130000086`, not `Q_130000086` |
| `value` | the metric value, **`float32`, not rounded** |

Rows-not-columns for locations is the point: the header stops growing with gauge
count, which is what forces `validate_hm_gauge_column_identity`'s check 3 to
compare column *sets* today. Post-CR-2 that check compares the `location`
column's value set instead — same invariant, simpler expression.

`location` is retained even where it is constant today (`basin` for the
basin-scalar variables). It costs one column, keeps every table unionable with a
plain `concat`, and is what CR-3 would populate with subbasin ids — a schema
change avoided by not dropping it.

#### Accepted costs of the per-variable split

Ruled with these known and accepted, not discovered during implementation:

1. **The output set becomes config-dependent.** Three currently-fixed things
   become derived: `WF3_TARGETS` in `Snakefile_climate_experiment`,
   `check_baseline.py`'s literal `TARGETS` entries, and the R9 path map (which
   needs a pattern row rather than two literal paths).
2. **WF3 must read `workflows.model_creation.wflow_outvars`** — a new
   cross-workflow config coupling. Today WF3 never reads it; it discovers what
   exists at *runtime* from the wflow CSV's columns
   (`basavg_vars = [x for x in sim.columns if "basavg" in x]`). Snakemake needs
   output paths at DAG-construction time, so runtime discovery is not an option.
   `check_model_reference.py` already exists to catch the case where the model on
   disk was built from a different list than the config now declares.

#### The size argument does NOT hold — measured, not assumed

Per-variable splitting was partly motivated by file size. It does not deliver
that. At the stated scale — 10 locations, 100 stress-test combinations, 10
realizations, class A per-realization and class B pooled:

| table | rows | approx size |
| --- | --- | --- |
| `q_indicators.csv` | 9 × 100 × 10 × 10 + 2 × 100 × 10 = **92,000** | ~5.5 MB |
| each basin variable (1 metric, 1 location) | 1 × 100 × 10 = **1,000** | ~55 KB |

`q` is ~98% of the volume, because it is the only variable that is both
multi-location and multi-metric. The split moves five ~55 KB files away from a
5.5 MB one that stays 5.5 MB. **The decision stands on its other two grounds**
(no `variable` column; each variable gets its natural geometry) — this is
recorded so a 5.5 MB CSV is not a surprise later.

If q's size becomes a real problem the levers are elsewhere: fewer metrics, or
dropping per-realization rows for class A and emitting only pooled values, or a
binary format.

### Decision — `realization_id = 0` means pooled

Ruled 2026-08-05. Keeps the column integer. Realizations are 1-indexed
(`rlz_1..rlz_N`), so `0` is free.

`cst_0` is NOT revisited. It is a genuine grid member (a real run, not a
sentinel) and is pinned in `naming.md` §4, in
`wildcard_constraints: st_num=[1-9][0-9]*` on `generate_climate_stress_test`, and
in `validate_wg5_catalog_grid`'s `m in 0..st_num`. Nothing about
`realization_id` justifies moving it.

**A numeric sentinel is safe here ONLY because of ruling (b1) below.** The hazard
of a numeric sentinel in a numeric key column is that `groupby("realization_id")`
folds pooled rows in as another realization. That requires both grains to
coexist *for one metric*, which under (b1) never happens — see the class table.
**If (b1) is ever reversed, the sentinel must become a string (`"all"`).** This
is a dependency, not a preference.

Since `0` cannot announce itself, `validate_hm7` asserts the invariant:
every class-B row has `realization_id == 0`; no class-A row does;
`realization_id` ∈ {0} ∪ `1..RLZ_NUM`.

### Decision — metrics split three ways

| Class | Metrics | Grain |
| --- | --- | --- |
| **A — linear in years** | `mean`, `max`, `min`, `q95`, `Q7day_max`, `Q7day_min`, `BaseFlowIndex` | Per realization (`1..N`). All are "annual statistic → mean over years"; realizations are equal-length, so per-realization values average back to the pooled value **exactly**. Nothing is lost. |
| **B — non-linear fit** | `returninterval`, `returninterval_min_7day` | Pooled (`0`) only. `frequency_analysis(freq="YS")` fits a GEV to annual blocks; pooling multiplies the block sample by `RLZ_NUM`. A per-realization fit over a short record is ill-conditioned — the owner's stated reason for this CR. |
| **C — selects a category** | `wetmonth_mean`, `drymonth_mean` | `groupby(index.month).sum().idxmax()` picks ONE month from the record, so different realizations can pick different months. Realization axis ruled: **pick once from the pooled record**. Stress-test axis still open (Q5). |

### Decision — retire `aggregate_rlz` (ruling b1)

Ruled 2026-08-05. In the long shape "aggregated" is no longer a *shape* choice —
which is the only reason the flag existed — so the table always carries the
finest grain (class A per realization, class B pooled) and downstream aggregates
as it likes.

Consequences:
- The key is read at `Snakefile_climate_experiment:811` via
  `get_config(my_cfg, "aggregate_rlz", True)` and appears in
  `snake_config.template.yml:194` and `snake_config_model_test.yml:82`.
- Workflow configs **silently ignore unread keys**, so a user's stale
  `aggregate_rlz` would quietly do nothing while they believed it was in effect.
  Removal policy under open Q7.
- One `aggr_rlz` parameter drives **both** tables, so `basin_indicators.csv`
  must change in the same commit — open Q6.

### Decision — CR-3 folded in: pool the sample, not the spliced series

Was a separate CR; merged 2026-08-05 after the worked example narrowed it.

`export_wflow_results.py:173-181` pools by `pd.concat`-ing realization series and
overwriting the index with a synthetic continuous `date_range` — butt-splicing
them into one fictitious record. A `rolling(7)` window then crosses each splice,
producing 7-day flows **that occurred in no realization**:

```
A's last 7 days:   20 20 20 20 | 2 2 2      -> A's true 7-day min = 12.3
B's first 7 days:   2  2  2  2 | 20 20 20   -> B's true 7-day min =  9.7
spliced window:              2 2 2 | 2 2 2 2 -> 2.0   <- never occurred
```

Each realization has a short 3-4 day low spell too brief to fill a 7-day window;
the splice manufactures a 7-day one. It becomes that year's annual minimum and
enters the GEV block sample for the low-flow RP.

**Scope is one metric, not the pooling method.** Which metrics touch `rolling(7)`
on the spliced series:

| Metric | Rolling? | Class | Fixed by |
| --- | --- | --- | --- |
| `returninterval` (high flow) | No — `frequency_analysis` on raw daily | B | Nothing needed; only block alignment matters |
| `returninterval_min_7day` | **Yes** | B | **This item** |
| `Q7day_max`, `Q7day_min`, `BaseFlowIndex` | Yes | A | **(b1) for free** — per-realization means no `concat`, no splice |
| `mean`, `max`, `min`, `q95` | No | A | n/a |

So the fix is: for `returninterval_min_7day`, extract each realization's annual
7-day minima **within** that realization, then pool the *blocks* and fit the GEV
on `RLZ_NUM × N` of them. The synthetic calendar disappears.

**Magnitude, stated honestly.** 6 contaminated windows per splice ×
`RLZ_NUM − 1` splices — 6 rows out of ~21,900 for 2 realizations × 30 years.
They matter only when a fabricated window becomes an annual extremum, which
needs the basin's low-flow season to coincide with the calendar year boundary.
Snowmelt and dry-winter basins: plausible. Wet-winter basins: inert. The
unconditional argument is different — both methods give `RLZ_NUM × N` blocks
**only if** every realization is a whole number of years on the same calendar
boundary, and nothing checks that. Sample-pooling does not depend on it.

### Decision — `value` is `float32`, unrounded (was Q1)

Ruled 2026-08-05. The `.round(2)` / `.round(4)` calls go away entirely; no
per-metric precision policy is needed.

The original request was a flat 2 digits, which would have been lossy: precision
today is per-statistic, and the `.round(4)` set is exactly the low-flow metrics
plus BFI. At 2 dp a `Q7day_min` of `0.0034` becomes `0.00` and the low-flow half
of the response surface flattens to zero.

`float32` avoids the tradeoff rather than resolving it. **Measured** in the pixi
env — pandas writes a float32 column as the shortest round-tripping repr, so the
output is short *and* keeps low-flow resolution:

| value | plain `to_csv` (float32) | `float_format="%.2f"` |
| --- | --- | --- |
| `12.3` | `12.3` | `12.30` |
| `0.0034567` | `0.0034567` | **`0.00`** |
| `0.87654321` | `0.8765432` | `0.88` |

Read back as `float64` (pandas default) — a lossless widening. `float32` carries
~7 significant digits, ample for discharge and every statistic derived from it.
The basin table is already `dtype="float32"`, so this also makes the two tables
consistent.

**Consequence — the baseline gate gets brittle (see Q8).** `.round(2)` was
incidentally a numeric-drift buffer: a 1e-6 change could not move a 2-decimal
file. Unrounded float32 exposes the last bit to a byte-exact `sha256`.

### Decision — `location` is the bare id (was Q2)

Ruled 2026-08-05. `130000086`, not `Q_130000086`.

The `Q_` prefix filter at `export_wflow_results.py:108` **stays** — it reads the
wflow output CSV, which is upstream of our table and keeps wflow's naming.
Stripping is a write-time transform (`col.removeprefix("Q_")`).
`validate_hm_gauge_column_identity` strips explicitly on one side of its
comparison rather than relying on the two strings being equal.

`location` is declared a **string** in both tables, read with
`dtype={"location": str}`. Bare ids would otherwise load as `int64` in
`q_indicators` and `object` in `basin_indicators` (where `location = "basin"`),
so two files claiming one schema would disagree on their shared column's dtype.
Ids are identifiers, not numbers; nothing arithmetic is done with them.

The row key is therefore
`(variable, metric, temp_change, precip_change, realization_id, location)`.
`validate_hm7` asserts its **uniqueness** — cheap, and it catches an id collision
between the `outlets` and `gauges_locations` maps, which in the wide table would
surface only as duplicate columns and go unnoticed.

### Decision — `variable` is a live discriminator, not future-proofing (was Q3)

Ruled 2026-08-05, on evidence rather than intent.

`setup_gauges_and_outputs.py:79-87`: when a `location_registry` is configured,
wflow is set up with `header=["Q", "P"]` on `gauges_locations`, so the run CSV
already carries **`Q_<id>` and `P_<id>` at the same gauge**.
`export_wflow_results.py:108` filters `startswith("Q_")` and **silently drops
every `P_` column**.

With bare-id locations those two are distinguished naturally by `variable` ∈
{`q`, `p`} at one `location` — which the wide format could not express without a
second column set. Whether to stop dropping `P_` is a separate call; leave it
dropped for now, and note that the schema no longer blocks it.

### Decision — `metric` is a composite, `<variable>_<statistic>` (was Q4, Q9)

Ruled 2026-08-05, **against the recommendation on this page**, deliberately and
with the reason recorded: a self-contained CSV, with the variable already merged
into the metric name, is handier for plotting and for sharing a result file
outside the project tree.

The redundancy is real and accepted — any two of {variable, statistic,
composite} determine the third, so `metric: q_mean_annual_7day_min` makes
`variable: q` derivable. The alternative considered and rejected was
`variable` + a variable-neutral `statistic` (`q` / `mean_annual_7day_min`), which
is fully normalised and composes the display name at read time.

**Consequence: the validator must assert what normalisation would have given for
free.** `validate_hm7` checks `metric.startswith(variable + "_")` and that the
statistic suffix matches the known pattern set. Without it, `variable` and
`metric` can silently disagree.

#### Do NOT borrow the terse conventional abbreviations

The appeal of a composite is that `Q95`, `Q10`, `7Q2`, `BFI` are established
hydrological names. **Established names carry established meanings, and two of
ours differ:**

- `q95` here is `sim.resample("YE").quantile(0.95).mean()` — the mean annual
  **95th percentile**, a HIGH flow. Conventional **Q95** is the flow *exceeded*
  95% of the time, from a flow-duration curve — a LOW-flow drought index. They
  are opposite ends of the distribution. The existing name is already a misnomer
  in domain terms; do not propagate it. Renamed to `..._p95`.
- `Q10` would be readable as either a 10-year flood or a 10%-exceedance flow.

Compose systematically instead. Verbose beats wrong.

#### Vocabulary

| current | `metric` |
| --- | --- |
| `mean` | `q_annual_mean` |
| `max` | `q_mean_annual_max` |
| `min` | `q_mean_annual_min` |
| `q95` | `q_mean_annual_p95` |
| `Q7day_max` | `q_mean_annual_7day_max` |
| `Q7day_min` | `q_mean_annual_7day_min` |
| `wetmonth_mean` | `q_wettest_month_mean` |
| `drymonth_mean` | `q_driest_month_mean` |
| `BaseFlowIndex` | `q_baseflow_index` |
| `returninterval` | `q_return_level_10yr_max` |
| `returninternval_min_7day` | `q_return_level_2yr_7day_min` |

`mean_annual_` is not padding: every one of these is "annual statistic, then mean
over years", which the current names hide. The shipped `returninternval` typo
(`export_wflow_results.py:239`) dies here.

**This closes Q9 for free.** `Tpeak` / `Tlow` are config values that appear in no
column and no name today, so two runs with different `Tpeak` produce
identical-looking rows meaning different things — a 10-year and a 20-year flood
level, indistinguishable once the file leaves the project folder. A composite
name absorbs them at no structural cost. The vocabulary is therefore partly
config-derived, so `validate_hm7` matches a **pattern**, not an enumeration.

#### `variable` token vocabulary

Short tokens, chosen over snake-cased `wflow_outvars` labels
(`river_discharge`, `actual_evapotranspiration`) — those would avoid a third
vocabulary but produce metrics like `actual_evapotranspiration_annual_total`,
undercutting the readability that motivated the composite. **Cost: a third
spelling** alongside the CSDMS names and the Tier 2 display labels, so this
mapping is the contract and belongs in the seam doc.

Authoritative source for the left two columns:
`dev/reference/workflows/model_creation.md:213-220` (the `WFLOW_VARS` map).
**Six entries, not five** — `precipitation` is one, emitted at registry
locations with header `P` (see the `variable` decision above).

| semantic name (`wflow_outvars`) | token | CSDMS name | unit |
| --- | --- | --- | --- |
| river discharge | `q` | `river_water__volume_flow_rate` | m³ s⁻¹ |
| precipitation | `precip` | `atmosphere_water__precipitation_volume_flux` | mm Δt⁻¹ |
| actual evapotranspiration | `aet` | `land_surface__evapotranspiration_volume_flux` | mm Δt⁻¹ |
| groundwater recharge | `recharge` | `soil_water_saturated_zone_top__net_recharge_volume_flux` | mm Δt⁻¹ |
| overland flow | `overland_flow` | `land_surface_water__volume_flow_rate` | m³ s⁻¹ |
| snow | `snow` | `snowpack_liquid_water__depth` | mm |

**The rule, so future tokens are not minted ad hoc: where the repo already has a
canonical short name for the quantity, use it; only mint a token where none
exists; and disambiguate against names already in use.** Three consequences:

- **`precip`, not `p`.** `naming.md` §6 tier 2 declares `precip` the canonical
  cross-tool name. `p` would be a seventh spelling for precipitation — the exact
  inconsistency CR-1 fixed.
- **`aet`, not `et`.** `pet` is already canonical here (one of the three HM-2
  forcing variables). `et` one letter from `pet` in the same result file is a
  misreading waiting to happen; `aet`/`pet` is the standard pairing.
- **`snow`, not `swe`.** The code comment calls it "snow water equivalent", but
  the CSDMS name is `snowpack_liquid_water__depth` — snowpack *liquid water*,
  not total water equivalent. Minting `swe` would assert a physical claim the
  upstream name does not support, and re-adjudicating wflow physics is outside
  this repo's scope (`AGENTS.md` hard constraint). Keep the label's own word.

---

## CR-3 — subbasin-mean precipitation instead of point precipitation

Raised 2026-08-05 alongside the per-variable ruling: for CST's purposes a
subbasin mean is more useful than precipitation at an exact gauge point.

This is a **model-output** change, not a table-shape one, so it is its own CR.
It splits into a free half and a real one:

| Want | Status |
| --- | --- |
| **Basin-average precip** | **Available today, no code change.** Put `precipitation` in `wflow_outvars` and it is emitted as `precipitation_basavg` via `mapname="subcatchment", reducer=["mean"]`. `plot_results.py:314` explicitly drops it from the basin plots, so someone already anticipated it appearing. |
| **Per-*subbasin* precip** | **Not available.** `reducer=["mean"]` collapses the whole subcatchment map to one value. One value per subbasin means dropping the reducer in `setup_gauges_and_outputs.py` — a model_creation change. |

**Consequence if the reducer is dropped:** every basin-scalar variable gains
*multiple* locations, so `location` stops being constant and the row count of
each basin table multiplies by the subbasin count. This is the case where the
size argument dismissed above *does* start to bite — and it is the reason
`location` is retained in the basin tables even though it is constant today.

Note this also removes the last point-geometry variable other than `q`: with
precip aggregated, the point/basin split collapses to `q` vs everything else,
which is what the owner originally described.

### CR-3b — basin variables reported per contributing subbasin

Ruled 2026-08-05: the basin-scalar variables use **the exact same location set
and names as the q table**, so a gauge's discharge metrics and its catchment's
ET / recharge / snow sit on one join key.

**No crosswalk is needed — the ids are already one namespace.**
`write_outlet_index.py:3-4` records that hydromt_wflow 1.x labels outlets with
**basin-derived subcatchment IDs** (e.g. `130000086`), and `build_outlet_index`
merges `subcatchment_id` to `subbasin_id` with `validate="one_to_one"`. So the
map id in `Q_130000086` *is* the subcatchment id, and dropping the `reducer`
emits `<header>_<subcatchment_id>` columns in the same namespace the q table
already uses.

`location` stays the **numeric id**. `outlet_index.csv` also carries
`station_name`, `location_code`, `subbasin_code` and `compat_station_name`
(`wflow_1..N`), but those are a presentation join — the numeric id is what wflow
emits, so nothing can drift between the two tables.

#### OPEN — nested or incremental subcatchments?

This decides whether the requested overall basin value can be *derived*:

| | consequence |
| --- | --- |
| **Incremental** — each gauge's own area, tiling the basin, non-overlapping | the basin value is a valid area-weighted mean of the per-location values |
| **Nested** — each gauge's full upstream contributing area, so a downstream gauge contains upstream ones | the areas overlap; area-weighting double-counts and the basin value **cannot** be derived |

"Contributing subbasin" reads as nested.

**Recommendation regardless: do not derive it.** Emit the overall basin value
independently by keeping the existing `reducer=["mean"]` output alongside the new
per-subcatchment one, under a reserved `location = basin`. A derived value that
is silently wrong under nesting is the worse failure, and the reduced output
already exists — one extra `setup_config_output_timeseries` call, not new
computation.

---

## Rulings closing Q5, Q8, Q10 (2026-08-05)

**Q5 — class C stress-test axis: fixed month from `cst_0`.** The wettest and
driest months are picked once, from the `cst_0` baseline pooled over
realizations, and that month is evaluated for every stress-test member. The
surface then shows how flow in a given month responds to perturbation rather
than conflating that with the month itself moving. **This makes `cst_0` rows
mandatory** — the aggregated path drops them today (`st_nb = i + 1`), and the
month cannot be picked from a record that is not there.

**Q8 — tolerance comparator.** The indicator tables move off byte-exact `sha256`
onto `check_baseline.py`'s existing `compare_discharge`-style tolerance
comparator, the one already used for the wf1 discharge anchor. Byte-exactness
became untenable when `.round(2)` was dropped (see the `float32` decision): it
was the accidental drift buffer, and without it every harmless numeric nudge
fails the gate without indicating a defect — the same argument that excludes
`FIGURE_KINDS` from the baseline by default.

**Q10 — keep native wflow units.** Overland flow reduces with a
**unit-preserving** reduction (annual mean, m³ s⁻¹) instead of a sum. ET and
recharge keep `annual_total` in mm/yr: a daily sum of a mm Δt⁻¹ flux is a
legitimate time-integral of a flux, not a unit error, which is precisely why
overland flow was the odd one out. Per-variable tables make the differing metric
vocabularies natural rather than awkward.

> **Reading to confirm at implementation:** this is scoped to the overland-flow
> defect. If the intent was that *all* basin variables report in native
> per-timestep units (ET as mm/day rather than mm/yr), that is a factor of 365 on
> two variables and needs saying explicitly.

---

## Open questions

Q1-Q4, Q6 and Q9 are closed in the decision sections above; Q5, Q8 and Q10 in
the ruling block immediately above. **Two remain.**

| # | Question | Recommendation |
| --- | --- | --- |
| **Q7** | Stale `aggregate_rlz` in an existing user config — silently ignored today, because workflow configs never read unknown keys. The user believes it is still in effect. | Hard error naming the migration note, following the `variable_spec.parse` precedent (it refuses the pre-5e list shape and states the migration). |
| **Q11** | **Nested or incremental subcatchments?** Decides whether the overall basin value can be derived from the per-location values. See CR-3b. | Do not derive it either way — emit it independently under a reserved `location = basin`, by keeping the existing `reducer=["mean"]` output alongside the new per-subcatchment one. |

---

## Cost, once the batch lands

- **§7 migration note** — required twice over: output-table column labels (the
  bullet added to `naming.md` §7 on 2026-08-05) and removal of a checked-in
  user-facing config key.
- **`validate_hm7`** — rewritten. Simpler: the header is now fixed and
  gauge-independent. Gains the class↔sentinel invariant.
- **`validate_hm_gauge_column_identity`** — check 3 moves from column-set
  equality to `location`-value-set equality.
- **Baseline** — both tables are byte-exact `sha256` entries in
  `dev/baseline/manifest.json`. One WF3 run from the primary checkout, then
  re-record. **Combine with CR-1's outstanding run** rather than running twice.
- **`pytest tests/`** from the primary checkout is the merge gate; it fails until
  that run lands.
