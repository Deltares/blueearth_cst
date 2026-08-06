# WF3 change requests

Opened 2026-08-05, from observations on the first full WF3 run of the migrated
R9 tree. **Collect-then-implement**: the requests are specified here and built as
one batch, so the contract, the validators, the migration note and the baseline
re-record move once rather than per request.

**Scope: WF3 as a whole**, not only its output tables — that is simply where the
discussion started. CR-1..CR-3 all concern the result tables; later CRs need not.

This is the implementation specification, dense with file and line references.
The reviewable, plain-language companion is `wf3-changes-proposal.md`, which
carries the stable **C / F / O** reference numbers. **The two must be kept in
step:** a decision changed here without updating the proposal leaves the
reviewable version lying.

Status: **CR-1 landed. CR-2 specified, NOT implemented. Unit D DEFERRED.**
Collection closed for sequencing purposes — see the batch plan below.

---

## Batch plan and sequencing (2026-08-05)

The register grew from three requests to thirty-four changes and sixteen findings
because tracing each request surfaced adjacent ones. That is a collection, not a
plan. **It is four units**, and they do not all belong in this milestone.

| unit | changes | work | risk |
| --- | --- | --- | --- |
| **A — results tables** | C2–C21, C28 | rewrite `export_wflow_results.py`, rework `validate_hm7` + the relational validator, update tests | medium — moves numbers, needs the baseline |
| **B — run identification** | C22–C27 | `cst_`→`st_` rename, the design table, DAG-time enumeration | medium — many filenames, catalog keys, path map |
| **C — generator plumbing** | C29, C34 | delete rule 3.05, audit both weathergenr call sites | small |
| **D — config surfaces** | C30–C33 | redistribute the weathergen template, restructure `advanced_settings.yml` | **highest** — breaking migration for every existing project config |

### Unit D is DEFERRED to its own milestone

Nothing is broken today; it is tidiness plus the unreachable dimensions in F8. It
is the only unit with a breaking migration (the hydrological-year unification),
it touches every project config, and it is not what this thread set out to fix.
The specification stays here and is complete — it just does not land with A–C.

**Consequence to hold onto:** F7 (undeclared template input) was to be disposed of
by C31. With D deferred, F7 needs its own one-line fix — declare the template as
an input to rule 3.04 — or it stays open. Do not let it fall between the two.

### Order

| # | step | why here |
| --- | --- | --- |
| 1 | **Clear the owed WF3 re-run** (CR-1 / followups R9-2) | It already blocks `pytest tests/` on the primary checkout. Every unit below adds to that debt; running once now stops it compounding. |
| 2 | **Rule C29** | One yes/no. It is R10's only blocker — see below. |
| 3 | **R10** (rule identifier renames) | ACCEPTED and fully designed, so zero decision cost remains. Small bounded work should not queue behind large work with six unruled changes. |
| 4 | **Unit A** | The original request. |
| 5 | **Unit B**, with **Unit C** alongside | B is what makes A durable past two stress dimensions; C touches the same rules B does. |
| — | Unit D | deferred |

### Interaction with R10 — exactly one collision

R10's scope is rule **identifiers only** (bodies, inputs, outputs and numbering
are explicitly out), so there is no renumbering cascade. Checked rename by
rename against this register:

| R10 rename | interaction |
| --- | --- |
| 1.07, 1.11, 1.12, 2.01, 2.06 | none — WF1/WF2, untouched here |
| 3.03 → `prepare_stress_grid` | rule survives; only its outputs change (C22/C23). Name stays correct |
| 3.08 → `write_climate_catalog` | survives; catalog *keys* change, the rule does not |
| 3.04 → `prepare_weathergen_config` | survives; C31 changes its inputs, name stays correct |
| **3.05 → `prepare_weathergen_config_perturbed`** | **C29 deletes this rule** |

Renaming 3.05 would enter R10's `migration_rule-names.md` as a CLI-surface rename
and then vanish a milestone later. **If C29 is ruled, drop 3.05 from R10's
scope** — R10's own design already applies this principle in reverse, folding
`export_wflow_results` into R9 because "a milestone renames what it falsifies".
If C29 is declined, 3.05 stays and R10 renames it as designed.

**Shared hazard, worth keeping apart rather than merging:** both R10 and units
A–C edit `LOG_RULES`. R10's design records that a missed entry makes a log
section *silently vanish* while its parts stay on disk. Two separate careful
passes beat one tangled one.

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

**That collision is not hypothetical — see F17 below. It is live in the shipped
fixture.** This assertion is the only thing in the register that catches it.

### F17 — `Q_<id>` is emitted twice when an outlet is also a registry gauge

Observed 2026-08-05 in `test_case/test_local`, inside the failure message of an
unrelated test.

```
run CSV: time,Q_101,Q_105,Q_104,Q_102,Q_103,Q_101,P_105,P_104,P_102,P_103,P_101
                 ^^^^^                        ^^^^^   both from wflow, one name
wflow_sbm.toml [output.csv]:
    header = "Q"   map = "outlets"
    header = "Q"   map = "gauges_locations"
    header = "P"   map = "gauges_locations"
```

`setup_gauges_and_outputs.py` registers `Q` on **both** maps — `setup_outlets`
(`:55-58`) and `setup_config_output_timeseries` (`:79-87`). Outlet 101 is also
registry gauge 101, so wflow writes that series twice under one header.

**Propagation.** `pd.read_csv` renames the second to `Q_101.1`;
`export_wflow_results.py:108`'s `startswith("Q_")` filter takes both; so
`q_indicators.csv` carries **six gauge columns for five gauges**, in both wf1's
`output.csv` and every wf3 run CSV.

**Measured severity today: benign.** The two series are byte-identical
(`(a-b).abs().max() == 0.0`). So no number is wrong; a per-location aggregation
just double-counts station 101.

**Severity ceiling: not benign.** They agree because both map entries resolve to
the same cell. **Nothing asserts that.** An outlet id and a gauge id that coincide
numerically while pointing at different cells would put two different series
under one name, silently.

**`validate_hm_gauge_column_identity` does NOT catch it, in any of its three
checks:**

| check | why it passes |
| --- | --- |
| 1 — every produced column traces to a declared entry | a duplicate still matches the `Q_*` pattern |
| 2 — map-typed gauge columns carry the `Q_` prefix | only requires ≥1 such column |
| 3 — q_indicators gauge set == output_rlz gauge set | **both files carry the duplicate identically**, so they agree |

**Addressed by C17's row-key uniqueness assertion**, and by nothing else.
Structural for any project whose basin outlet is also a registered gauge — the
normal case, not an edge case.

**Also confirms F4 with data:** `P_101`..`P_105` are present in the same output
and dropped by the `Q_` filter. That finding was inferred from the code; this is
the artifact.

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

## CR-4 — run identification and the stress-test design table

Proposal-side: **Part B, C22–C27**, finding **F5**, open item **O3**.

### C22 — `cst_` → `st_` (`rlz_` unchanged)

Ruled 2026-08-05. `cst` is the tool's own name, so it says nothing as a member
token. The decisive evidence is that **the code already says `st`** and only the
filenames disagree:

| layer | token |
| --- | --- |
| Snakemake wildcard (naming.md §4, pinned vocabulary) | `st_num` |
| config-derived count | `ST_NUM` |
| shared helper | `stress_test_grid()` |
| config section | `stress_test:` |
| rule input variable | `st_csv_fns` |
| **filenames / catalog keys** | **`cst_<m>`** |

`Snakefile_climate_experiment:805` builds `f"{wg_dir}/_work/cst_{st_num}.csv"` —
a `cst_` filename from an `st_num` wildcard. So this **removes an existing
inconsistency**, and needs no new word.

Surfaces to move: `cst_<m>.csv`, `rlz_<n>_cst_<m>.{nc,csv,toml,log}`, the WG-5
catalog entry keys (`rlz_<n>_cst_<m>`), and the reserved baseline `cst_0` →
`st_0`. `wildcard_constraints: st_num=[1-9][0-9]*` on
`generate_climate_stress_test` is unaffected — the wildcard name does not change.

`rlz_` stays: unlike `cst`, it abbreviates a *correct* term (CMIP's `r1i1p1f1`
uses `r` for realization) and collides with nothing. Renaming it would cost the
same migration for no correctness gain. **Considered and rejected:** `mem_`,
which is climate-standard but is already WF2's word for CMIP members — a
genuinely different thing, so it would create the cross-workflow ambiguity this
whole thread has been removing.

Also rejected for the design token, each on a concrete collision: `scn_` /
`scenario_` (AGENTS.md forbids coupling the experiment to CMIP scenarios;
"scenario" already means an SSP here), `grid_` (the spatial/model grid), `point_`
(Part A's `location`).

### C23–C27 — the design table

- **C23** — `stress_test_design.csv`: one row per design point, one column per
  stress dimension, plus a row for `st_0` with every change zero. The artifact
  that is missing today — `st_<m>.csv` holds twelve monthly rows and there is no
  single place that answers "what is run 37?".
- **C24** — two id spaces, not one. `st_id` is the *designed* axis (enumerable,
  worth looking up); `realization` is the *sampled* axis (draw 7 has no
  parameters). Run identity stays `(rlz, st)`. Four reasons: C10 pools over
  realization but not design, and one opaque id cannot express that; adding
  realizations would otherwise renumber the design; the P3-3 batching work groups
  by realization via wildcard patterns; and a failing run's log should name what
  broke.
- **C25** — ids are **experiment-scoped**. The table lives in `experiments/<id>/`
  beside the config snapshot, which is already where settings are pinned. Rejected:
  content-hashed ids (stable but opaque and unsortable) and positional ids
  (`t2_p5_v1` — self-describing but grows a segment per dimension, which is what
  this change exists to avoid).
- **C26** — one enumeration, two consumers. The routine that expands the DAG also
  writes the table, so they cannot disagree. Without this there is a
  chicken-and-egg: Snakemake needs the ids at DAG-construction time, before any
  rule has written a file. `stress_test_grid()` is the natural place to extend.
- **C27** — id width derived from the count, not fixed at three digits. 10 rlz ×
  a 5×5×3 grid is 750 before a fourth dimension exists.

### C28 — results tables carry `st_id` ALONGSIDE the perturbation columns

Ruled 2026-08-05, **"at this stage"** — against the recommendation, which was to
replace. The owner chose plottable-without-a-join now, with an explicit revisit
when a third dimension arrives. The seven-column shape is therefore:

    metric, st_id, temp_change, precip_change, realization_id, location, value

**Two obligations follow, and neither is optional — they are what stop an interim
decision from rotting into a wrong one:**

1. **A consistency check.** `temp_change` / `precip_change` are now a cached copy
   of the design table's row for that `st_id`. `validate_hm7` must assert they
   agree. Same class as C2's `metric.startswith(variable + "_")` check: a
   denormalised copy that nothing verifies is a copy that eventually lies.
2. **A hard stop at the third dimension.** When `stress_test:` gains a third
   axis, the results writer raises, naming this decision, rather than silently
   adding a column. The `variable_spec.parse` precedent again — refuse and
   explain. Without it, CR-2's fixed-shape property degrades one column at a
   time with nothing noticing.

Recorded plainly: *alongside* re-couples the results header to the stress
dimension count, which is exactly what CR-2 removed on the location axis. That is
tolerable at two dimensions and is not a permanent position.

### Machinery CR-4 touches

`naming.md` §4 (wildcard vocabulary) and §7 (a rename note is required —
filenames *and* catalog source names are both listed there);
`validate_wg5_catalog_grid`, whose expected key set is literally
`rlz_<n>_cst_<m>`; the R9 path map; `check_baseline.py` targets; every WF3 log
and TOML name.

---

## CR-5 — retire the per-run weather-generator config — **RULED AND LANDED 2026-08-05**

Implemented as specified below, plus two things folded in because they live in
the same two rules:

| item | what landed |
| --- | --- |
| **C29** | rule 3.05 deleted; 3.07 takes the output path as its 4th CLI arg and reads the shared config from 3.04 |
| **F6** | only the two `transient_change` flags moved to the shared config — the `stress_test` step counts and monthly ranges did **not**, so the file no longer implies values that had no part in the run |
| **F7** | `config/templates/weathergen_config.yml` is now a **declared input** to 3.04, not a params-only read |

**Design choice worth recording.** `impose_climate_change.R` derives `out_dir`,
`file_prefix` and `file_suffix` from the single output path rather than taking
them as three separate arguments. `weathergenr::write_netcdf` composes
`<prefix>_<suffix>.nc`, so the stem is split at its LAST underscore. That keeps
one source of truth — the Snakemake `output:` declaration — and is
naming-agnostic: `rlz_1_cst_2` and `rlz_1_st_2` both split correctly, so **C22's
member-token rename touches nothing here**. It refuses loudly if the stem has no
underscore.

**`LOG_RULES` was updated in the same edit**, which matters as much as deleting
the rule: `merge_logs` discovers a section by listing the directory named after
its label, so a label with no producer contributes an empty section forever.
R10's design records the mirror hazard for renames on this same list.

**Gates:** `pytest tests/test_cli.py` (dry-runs all three Snakefiles — the DAG is
valid without 3.05) + the two affected module suites: **62 passed, 26 skipped**.
`validate_wg3` now pins the two flags, with a synthetic pass/fail pair per flag.

**NOT verified: the R side.** Nothing here exercises
`impose_climate_change.R` — no R test harness exists (`dev/followups.md`:
"R testthat coverage. Decided at the start of R5 — Python helpers only"). The new
arity, the path split and the flag read all first execute at rule 3.07 in a real
WF3 run. **Run WF3 in the primary checkout before treating this as done.**

### Original proposal

Proposal-side: **Part C, C29**, finding **F6**.

Rule 3.05 `prepare_weagen_config_st` emits one
`{wg_dir}/_work/weathergen_config_rlz_{rlz_num}_cst_{st_num}.yml` per member,
each with its own log and benchmark file. At RLZ_NUM=10, ST_NUM=88 that is 880
YAMLs + 880 logs + 880 benchmark TSVs.

**The file carries no per-member information beyond its own output name.**
`build_weagen_config`'s stress-test branch (`prepare_weagen_config.py:69-81`)
emits exactly:

| key | varies? | read by `impose_climate_change.R`? |
| --- | --- | --- |
| `imposeClimateChanges.output.path` | no — constant `{wg_dir}/output/` | yes (`:29`) |
| `imposeClimateChanges.nc.file.prefix` | **yes** — `rlz_<m>_cst` | yes (`:30`) |
| `imposeClimateChanges.nc.file.suffix` | **yes** — `<n>` | yes (`:31`) |
| `temp` (whole config block) | no | **only `$transient_change`** (`:34`) |
| `precip` (whole config block) | no | **only `$transient_change`** (`:35`) |

The prefix/suffix split exists only because `weathergenr::write_netcdf` takes
`file_prefix` and `file_suffix` separately — and Snakemake already knows that
path: it is rule 3.07's own declared output `rlz_st_nc`.

**F6:** the copied `temp:` / `precip:` blocks carry `step_num` and the monthly
`mean.min` / `mean.max` ranges, none of which the R script reads. The values that
actually perturb the run come from `cst_<n>.csv`. So the file presents
plausible-looking perturbation settings that had no part in the run it names —
worse than carrying nothing.

**Proposal:** pass the output path (or prefix/suffix) to `impose_climate_change.R`
as CLI args alongside the two it already takes, move `temp.transient_change` /
`precip.transient_change` into the single `weathergen_config.yml` from rule 3.04,
and delete rule 3.05. Removes one rule, one wildcard-expanded artifact class, and
2,640 files from a production sweep.

**Not yet ruled** — awaiting owner decision.

---

## CR-5b — audit both weathergenr call sites against v1.2.0 (C34, PROPOSED)

Signatures read from the **installed** package, not from docs:
`tanerumit/weathergenr@v1.2.0` (pinned in `dev/scripts/install_weathergenr.R`),
via the primary checkout's env — this worktree has had only `pixi install`, so
weathergenr is absent here.

```
Rscript --vanilla -e "print(args(weathergenr::generate_weather))"
Rscript --vanilla -e "print(args(weathergenr::apply_climate_perturbations))"
```

### F13 — nothing we pass is retired; the dead keys are pre-1.2.0 vestiges

**`generate_weather()` takes 24 args; `generate_weather.R:39-58` passes 19.**
Every one still exists in v1.2.0 — there is no retired parameter in either call.
Unpassed:

| arg | default | note |
| --- | --- | --- |
| `save_plots` | `TRUE` | **the live control that `evaluate.model` used to be** |
| `warm_filter_bounds` | `list()` | new in 1.2.0 — acceptance bounds on the generated annual series |
| `relax_priority` | `c("wavelet","sd","tail_low","tail_high","mean")` | new — which distributional criterion is sacrificed when the warm-pool filter cannot be met. A **scientific** choice currently made by an upstream default |
| `n_cores` | `NULL` | we pass `parallel` but never `n_cores` |
| `verbose` | `FALSE` | |

v1.2.0 splits evaluation out into its own exports (`evaluate_weather_generator`,
`prepare_evaluation_data`), which is why `evaluate.model` / `evaluate.grid.num`
(F10) reach nothing. Plot emission is now `save_plots`, unset and defaulting
`TRUE` — so setting `evaluate.model: FALSE` today does **not** stop the plots.

### F14 — three perturbation dimensions exist upstream and are unreachable

**`apply_climate_perturbations()` takes 25 args;
`impose_climate_change.R:45-57` passes 11.** Fourteen unpassed:

| arg | default | class |
| --- | --- | --- |
| `precip_occurrence_factor` | `NULL` | **STRESS DIMENSION** — wet/dry day frequency |
| `precip_occurrence_transient` | `TRUE` | its transient flag |
| `precip_intensity_threshold` | `0` | wet-day threshold |
| `exaggerate_extremes` | `FALSE` | **STRESS DIMENSION** — extreme intensification |
| `extreme_prob_threshold` / `extreme_k` | `0.95` / `1.2` | its controls |
| `precip_cap_mm_day` / `precip_floor_mm_day` / `precip_cap_quantile` | `NULL` | physical bounds on perturbed precip |
| `scale_var_with_mean` / `enforce_target_mean` | `TRUE` / `TRUE` | how the perturbation is conditioned |
| `pet_method` | `"hargreaves"` | see F16 |
| `seed` | `NULL` | see F15 |
| `verbose` | `FALSE` | |

With `dry_spell_factor` / `wet_spell_factor` on the generate side (F8), that is
**three additional stress dimensions already installed and working**: occurrence
frequency, extreme intensification, spell length. Ties directly to CR-4/C23 —
the design table's extra columns are plumbing, not new science.

### F15 — the perturbation step is unseeded while generation is seeded

`generate_weather.R:57` passes `seed = yaml$...$seed` (`123` from the template).
`impose_climate_change.R` passes no `seed`, so `apply_climate_perturbations` gets
`NULL`. The function accepting a seed implies something stochastic in it
(quantile-mapping fit / resampling). **Confirm against the package source before
asserting non-determinism** — but the asymmetry is unchosen either way.

### F16 — PET computed twice, by two methods, first result probably discarded

`impose_climate_change.R:54` passes `compute_pet = TRUE` and lets `pet_method`
default to `"hargreaves"`. Rule 3.09's `downscale_climate_forcing.py:117-122`
then calls `setup_temp_pet_forcing(..., pet_method=pet_method)` with `makkink`
(eobs) or `debruin`, recomputing PET from the perturbed temperature. HM-2/WG-6
pins the wflow forcing's `pet`; WG-4 pins only `precip`/`temp` on the generator
NC. **Verify the generator's `pet` really is unused before removing the work** —
but the chain carries two PET methods and neither was chosen at the first step.

### C34 — one recorded decision per argument

Not "expose everything" — most defaults are right and surfacing them would bloat
the project config. The rule is that each of the 19 unpassed args gets one
recorded decision: **surface it, or accept the default deliberately**. An
unexamined default is not a choice. Minimum set to surface, on this evidence:
`save_plots`, `pet_method`, `seed` (perturbation side), and the three stress
dimensions once CR-4's design table can carry them.

---

## F7 — `config/templates/weathergen_config.yml` is an UNDECLARED input to 3.04

Found 2026-08-05 while checking the appendix's dependency arrows.
`Snakefile_climate_experiment:588` passes it as
`params.default_config = "config/templates/weathergen_config.yml"`, never as an
`input:`. `build_weagen_config` reads it as the seed dict for the `generate`
branch (`prepare_weagen_config.py:57`), so it carries `general.variables`,
`warm.sample.num`, `warm.variable` — all behaviour-affecting.

**Failure mode:** edit the template, and 3.04 does not re-run. The generated
`{wg_dir}/config/weathergen_config.yml` stays stale, and 3.06 keeps generating
realizations from the superseded settings. It propagates silently because the
*generated* config **is** declared and consumed by 3.06 (`:629`), so every
downstream timestamp is consistent.

**Not the same as the neighbouring `ancient()` usage.** `ancient(config_path)` in
3.03 and the params-passed `snake_config` in 3.04/3.05 have an evident reason:
`suggest_experiment_name.py` rewrites the config as text, so an mtime-sensitive
edge would invalidate the whole pipeline on every name pin. The template has no
counterpart — not declared, not marked ancient, no recorded decision.

**Fix:** one line, declare it as an input to 3.04. Fold into whichever CR next
touches this rule — CR-5 (C29) is the natural carrier, since it rewrites 3.04's
neighbourhood anyway.

---

## CR-6 — configuration surfaces — **DEFERRED to its own milestone**

Proposal-side: **Part D, C30–C33**, findings **F8–F12**.

**Deferred 2026-08-05** (see the batch plan). The specification below is complete
and stands; it simply does not land with units A–C. Reasons: it is the only unit
with a breaking migration, it touches every project config, nothing is broken
today, and it is not what this thread set out to fix.

**Do not lose F7 on the way out.** C31 was to dispose of it by removing the
template entirely; deferred, it needs its own one-line fix (declare
`config/templates/weathergen_config.yml` as an input to rule 3.04) or it stays
open indefinitely.

### The three surfaces, and why two of them overlap

| file | charter | schema |
| --- | --- | --- |
| `config/workflows/snake_config_*.yml` | per-project, the `--configfile` target | per-workflow `get_config` |
| `config/advanced_settings.yml` | *"toolbox-wide knobs a normal project never touches"* | **CLOSED** — `snake_utils._ADVANCED_SETTINGS_SCHEMA`, unknown key/section rejected at parse time, `tests/test_advanced_settings.py` |
| `config/templates/weathergen_config.yml` | *"Weather Generator Advanced settings"* | **none** |

Two files, one charter. The mature one does not hold the generator knobs.

### C30 — split by OWNERSHIP, not change frequency

- project config: describes *this* basin/experiment;
- `advanced_settings.yml`: applies to every project, under `constraints:` /
  `defaults:` / `runtime:`;
- `config/templates/`: the SHAPE of a generated artifact — not a settings
  surface. Using it as one is the root cause.

"Rarely changed" is unfalsifiable. Ownership is decidable.

### C31 — redistribute `config/templates/weathergen_config.yml`

Consumer evidence: `generate_weather.R:39-58` reads 16 keys; `:100` reads
`nc.file.prefix`. `impose_climate_change.R` reads only `imposeClimateChanges.*`
plus the two `transient_change` flags.

| key | destination | rationale |
| --- | --- | --- |
| `dry.spell.change`, `wet.spell.change` | project config, under `stress_test:` | F8 — perturbation params, passed to `apply_climate_perturbations` beside temp/precip factors |
| `month.start` | project config, unified with `climate_projections.start_month_hyd_year` | F9 |
| `seed` | project config (`climate_experiment.seed`) | reproducibility is an experiment property |
| `general.variables` | project config or derived from the store | must be a subset of what `extract_historical.nc` carries; nothing checks it today |
| `warm.signif.level`, `warm.sample.num`, `knn.sample.num`, `mc.wet.quantile`, `mc.extreme.quantile` | `advanced_settings.defaults.weathergen.*` | true generator tuning; gains schema validation |
| `compute.parallel` | `advanced_settings.defaults.weathergen.*` | resource sibling of `julia_threads` |
| `evaluate.model`, `evaluate.grid.num` | **delete** | F10 — read by nothing |
| `output.path`, `sim.year.start`, `sim.year.num`, `nc.file.prefix`, `realizations_num` | not settings | already injected per-run by `prepare_weagen_config.py:60-66` |

Rule 3.04 survives — it still assembles the generated `weathergen_config.yml`,
just from `advanced_settings` + project config instead of a template seed.
**Disposes of F7** (no template → no undeclared input) and composes cleanly with
CR-5/C29, which retires the *per-run* config (rule 3.05), a different file.

### C32 — write the RESOLVED toolbox settings into the project under their own name

Not a provenance fix — provenance is already intact. `Snakefile_climate_experiment:392`
passes `advanced_settings = ADVANCED_SETTINGS` to `copy_config_files.py`, and
`_write_snapshot_bundle` writes them into `<snapshot_dir>/effective.yml` under an
`advanced_settings:` key, which also feeds `effective_config_digest` and
`snapshot_bundle_digest`. Changing the file changes the digest, which changes the
snapshot directory name.

It is a **findability** fix, and the evidence is the strongest kind: the tool's
own author looked for these settings in a project folder and concluded they were
not saved. They are nested under a key, inside a differently-named file, in a
digest-named directory. Write them as `<exp_dir>/config/advanced_settings.yml`
beside the existing `snake_config_climate_experiment.yml` snapshot — **resolved
values, not a copy of the source**, since resolved is what applied.

### C33 — group `advanced_settings.yml` by KIND first, COMPONENT second

    defaults:
      weathergen: {...}
      wflow:      {julia_threads: 4}

The three sections encode **override semantics**, which is what the closed schema
enforces and the first question a reader has. Component-first buries it.

**The case that settles it:** `min_historical_years` *originates* with weathergenr
(the wavelet decomposition's minimum annual record) but is *enforced* against wf1
at parse time and against the shared climate-store producer, so it binds wf2 and
wf3 too. Component-first would file it under `weathergen` and misdescribe where
it binds. Kind-first keeps it a constraint with its origin in the comment — what
it does today.

**Category gap this opens.** Generator tuning is neither a hard limit
(`constraints:`) nor a toolchain pin (`runtime:`), leaving `defaults:` — which
means "a project config MAY override", so each needs an override key in the
project schema. **Recommended:** accept that, mirroring `julia_threads` ←
`shared.julia_threads`; an unused optional override key costs nothing and lets a
project with a real reason differ without editing the toolbox. **Rejected:** a
fourth `tuning:` section — it adds a concept to a split documented as "the
point", and hard-blocks a legitimate per-project need.

### Migration for existing project configs

Additive except one item:

| change | impact on an existing config |
| --- | --- |
| `stress_test.dry_spell` / `wet_spell` | **optional**, default `[1.0]*12` — untouched configs produce identical results |
| `climate_experiment.seed` | **optional**, default `123` |
| generator tuning → `advanced_settings` | **none** — toolbox file to toolbox file |
| template deleted | **none** |
| **hydrological-year start unified** | **BREAKING** — see below |

`climate_projections.start_month_hyd_year: Jan` (wf2, month *name*) and the
template's `month.start: 1` (wf3, *integer*) become one key in one place —
`shared:` is the natural home, since it is cross-workflow. A config still
declaring the old location must be **refused with an explanation** (the
`variable_spec.parse` precedent, same as Q7/O2); silently ignoring it would change
results while the file still claims otherwise.

**Verify before merging, do not assume:** wf2 uses it as an *aggregation
boundary* for hydrological-year statistics; weathergenr uses it as
`year_start_month`, a *simulation start*. Same concept, but confirm the two
consumers agree on semantics before collapsing them into one key.

### F11 — the WG-3 contract under-covers

`interchange_contracts._WG3_GWS_KEYS` pins 11 keys; `generate_weather.R` reads 16
from `generateWeatherSeries`. Missing: `warm.signif.level`, `warm.sample.num`,
`mc.wet.quantile`, `mc.extreme.quantile`, `compute.parallel`. Drop one from the
config and `validate_wg3` passes while R hands `NULL` to `generate_weather`. The
docstring claims the set is "the key set the R side reads". Widen to 16 whenever
C31 lands, since C31 changes where those keys come from.

### F12 — hardcoded historical anchor

`prepare_weagen_config.compute_nr_years` returns
`ceil((middle_year + wflow_run_length/2) - 2010 + 2)`, and `:62` writes
`sim.year.start: 2010`. The docstring calls 2010 "the historical-end anchor".
Nothing reconciles it with the actual span of `extract_historical.nc`. A project
whose record ends in 2005, or runs to 2020, gets a wrong anchor silently. Fix is
to derive it from the store — moves numbers, so it needs its own ruling.

---

## CR-7 — invert the wf3 batch-size default (C35, F18) — PROPOSED, not ruled

Proposal-side: **Part E, C35**, finding **F18**. Supersedes the *framing* of
`dev/followups.md` § Post-P3-3's disk item — see below.

### F18 — `B` keys off sweep size; only per-run cost should set it

`Snakefile_climate_experiment`:

```python
batch_size = min(batch_size_max, max(1, ceil(K / cores)))   # batch_size_max = 8
```

What batching amortizes is **fixed**: `F + S_cold − S_warm` = `24 + 92 − 35` ≈
**81 s per amortized member** (measured, `dev/milestones/p33/batching-results.md`
§ Decomposition). That is Julia start + Wflow-code warm-up — **compilation, not
simulation** — so it does not scale with basin size or run length.

| per-run sim time | solo | warm | saving |
| --- | --- | --- | --- |
| 35 s (seed fixture) | 116 s | 35 s | **70 %** |
| 1 h | 3,681 s | 3,600 s | **2.2 %** |
| 6 h | 21,681 s | 21,600 s | **0.4 %** |

**At the owner's stated production scale** — 1–6 h per run, 3–5 rlz × 25 cst, so
K ≈ 125 — at `-c 3`: `ceil(125/3) = 42`, clamped to **B = 8**, the maximum. The
sweep gets the largest batch on exactly the runs where batching is worth least:

| | at 1 h/run |
| --- | --- |
| total work, B=1 | `125 × (3600+81)` = 460,125 s → **42.6 h** at `-c 3` |
| total work, B=8 | `125 × 3600 + 16 × 81` = 451,296 s → **41.8 h** |
| gain | **1.9 %** |
| blast radius | up to **7 completed runs** discarded — 7 h at 1 h/run, **42 h at 6 h/run** |

Three second-order costs also grow with run length: scheduling granularity (125
jobs load-balance better than 16, and P3-3 chose LPT precisely because member
costs vary), restart cost (1 member lost vs up to 8), and progress visibility
(per-member Snakemake jobs vs driver log lines, over a 40 h sweep).

**Why this was invisible.** On the seed fixture K=12 at `-c 3` gives
`min(ceil(12/3), 8) = 4`, so every P3-3 measurement stands and the 35.4 % win is
real. The clamp only binds from K > 24 — the followup already records that — and
the *timing* divergence appears in the same region.

### C35 — default `batch_size: 1`, batching opt-in

Mechanism unchanged; only the default side flips. `config/workflows/snake_config_model_test.yml`
sets `batch_size: 4` so the fast-sweep behaviour and the batched code path stay
exercised, and so P3-3's measurements remain reproducible from the seed config.

**It dissolves the Post-P3-3 disk item rather than solving it.** That followup
asks for a cap computed from a stated disk headroom and a per-run
forcing+state size estimate, and records that the estimate is the hard part —
at parse time the forcing NCs are `temp()` and do not exist. At B=1 the `p × B ×
(forcing + state)` term loses its `B`, so there is no cap left to compute. The
followup's *observation* stands and is strengthened; its proposed remedy becomes
unnecessary.

**Interim, no code needed:** set `batch_size: 1` in any production config.

### Adjacent, NOT proposed here

The frozen resource triple is `-c 3, --threads 4`. Wflow parallelizes over grid
**cells**, so a real basin gains from threads in a way the fixture cannot show,
and the `-c N × julia_threads <= logical CPUs` budget probably wants rebalancing
toward threads at production scale. That is a **measurement**, not an inference —
do not change it on the strength of this note.

---

## Open questions

Q1-Q4, Q6 and Q9 are closed in the decision sections above; Q5, Q8 and Q10 in
the ruling block immediately above. **Two remain.**

| # | Question | Recommendation |
| --- | --- | --- |
| **Q7** | Stale `aggregate_rlz` in an existing user config — silently ignored today, because workflow configs never read unknown keys. The user believes it is still in effect. | Hard error naming the migration note, following the `variable_spec.parse` precedent (it refuses the pre-5e list shape and states the migration). |
| **Q11** | **Nested or incremental subcatchments?** Decides whether the overall basin value can be derived from the per-location values. See CR-3b. | Do not derive it either way — emit it independently under a reserved `location = basin`, by keeping the existing `reducer=["mean"]` output alongside the new per-subcatchment one. |
**Q12 (= proposal O3) is CLOSED — ruled *alongside*, 2026-08-05.** See C28 below.
**CR-2 is no longer blocked.**

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
