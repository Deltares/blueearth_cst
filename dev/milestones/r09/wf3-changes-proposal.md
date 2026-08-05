# Proposal: changes to Workflow 3 (the climate stress test)

Plain-language summary of what we propose to change in WF3, and why. Written to
be read, not parsed — the precise specification, with file and line references,
lives in `wf3-change-requests.md`.

Opened 2026-08-05, and **still being added to**. Nothing here is implemented yet
except **C1**.

Everything is numbered for reference: **C**hanges, **F**indings (problems we
turned up along the way), **O**pen decisions. **Numbers are stable** — once
assigned they are not reused or renumbered, so a later note can point at C14 and
mean the same thing it means today. New items append.

This document covers WF3 as a whole. Part A is the result tables, which is where
the discussion started; further parts are added as topics come up.

---

## Index

### Part A — result tables

| | |
|---|---|
| **Naming** | C1 rename perturbation columns · C2 variable in the metric name · C3 spell out statistic names · C4 say "mean annual" · C5 return period in the name |
| **Shape** | C6 locations as rows · C7 one file per variable · C8 keep `location` even when constant |
| **Realizations** | C9 retire the pooling switch · C10 pool per statistic, not globally · C11 `realization_id = 0` means pooled |
| **Seasonal statistics** | C12 choose the month once · C13 include the unperturbed baseline |
| **Numbers** | C14 single precision, unrounded · C15 keep the model's units · C16 pool yearly extremes, not the stitched record |
| **Locations** | C17 plain gauge number · C18 same locations for basin variables · C19 basin figure produced directly · C20 precipitation as a subbasin mean |
| **Checks** | C21 tolerance comparison instead of exact match |

### Part B — run identification

| | |
|---|---|
| **Naming** | C22 `cst_` becomes `st_` (`rlz_` unchanged) |
| **Design table** | C23 write a master design table · C24 keep two identifiers, not one · C25 numbers are experiment-scoped · C26 enumerate once, use twice · C27 width follows the count · C28 results carry `st_id` alongside the perturbation columns |

### Part C — generator plumbing

| | |
|---|---|
| | C29 retire the per-run generator configuration file *(proposed, not ruled)* |

### Across the document

| | |
|---|---|
| **Findings** | F1 January-only perturbation · F2 overland flow units · F3 seam in the 7-day window · F4 gauge rainfall discarded · F5 run numbers silently change meaning · F6 per-run generator config is empty and misleading |
| **Open** | O1 do subbasins overlap · O2 stale configuration switch · ~~O3~~ closed → C28 |

An appendix at the end sketches how a stress test is built today, step by step.

---

# Part A — result tables

## The short version

Today the stress test writes two result tables whose **shape depends on the
basin** — one column per gauge, so a five-gauge basin and a fifty-gauge basin
produce structurally different files. We propose switching to **one table per
variable**, each with the same six columns regardless of how many gauges there
are, with locations as rows instead of columns.

Along the way we settled a naming vocabulary, made explicit which statistics are
pooled across realizations and which are not, and found four numerical problems
that were not visible before.

---

## What is wrong with the tables today

- **The header grows with the basin.** Each gauge is its own column, so no two
  projects produce the same file structure and any script that reads them has to
  discover the columns first.
- **Two names for the same thing.** The perturbation columns were called `tavg`
  and `prcp`, while everywhere else in the toolbox the same two quantities are
  `temp` and `precip`.
- **All-or-nothing pooling.** A single switch decided whether *every* statistic
  was averaged over realizations or *none* was. There was no way to say "pool the
  return periods, because a single realization is too short to estimate a 10-year
  flood, but keep everything else per realization" — which is what you actually
  want.
- **The return period is invisible.** The table records a flood level but not
  whether it is a 10-year or a 20-year one. Two runs with different settings
  produce files that look identical and mean different things.
- **Inconsistent metric names,** one of them misspelled (`returninternval`) and
  shipped to users.

---

## What the tables will look like

One file per variable, named after the variable:

    q_indicators.csv
    aet_indicators.csv
    recharge_indicators.csv
    overland_flow_indicators.csv
    snow_indicators.csv
    precip_indicators.csv

Only the variables actually requested in the model configuration get a file.
Every file has the same seven columns:

| column | what it holds |
|---|---|
| `metric` | what was calculated, e.g. `q_mean_annual_7day_min` |
| `st_id` | which stress-test design point — the key into the design table (C23) |
| `temp_change` | the temperature perturbation, in °C |
| `precip_change` | the precipitation perturbation, in % |
| `realization_id` | which realization — or `0`, meaning "pooled over all of them" |
| `location` | the gauge or subbasin number |
| `value` | the number |

`temp_change` and `precip_change` are a **deliberate duplicate** of what the
design table already holds — kept so the file can be plotted without a join. See
C28, which is why they are here and what has to happen when a third stress
dimension arrives.

---

## Naming

**C1 — Rename the perturbation columns to `temp_change` and `precip_change`.**
*(Already done.)* They were `tavg` and `prcp`, the only place in the whole
repository using different names for temperature and precipitation. Not renamed
to bare `temp`/`precip` because these columns hold the *perturbation applied*,
not the temperature or rainfall itself — and the two axes are not even the same
kind of number (°C added, versus % change). A column called `temp` in a results
table would read as a temperature.

**C2 — Put the variable into the metric name.** `q_mean_annual_max`, not `max`.
This makes each row self-explanatory when a file is opened on its own or emailed
to someone, which is worth the small redundancy with the filename.

**C3 — Spell out the statistic names instead of borrowing the conventional
abbreviations.** No `Q95`, no `Q10`, even though they are standard. The reason is
specific: what this tool computes as "q95" is the mean annual **95th
percentile**, a high flow — whereas `Q95` in hydrology conventionally means the
flow *exceeded* 95% of the time, which is a low-flow drought index. They are
opposite ends of the distribution, so borrowing the abbreviation would mislead
exactly the reader who knows what it usually means. We write
`q_mean_annual_p95`.

**C4 — Say "mean annual" where that is what it is.** Almost every statistic here
is "compute it for each year, then average across years", and the old names hid
that. `Q7day_max` becomes `q_mean_annual_7day_max`, which is what it has always
been. This also retires the shipped misspelling.

**C5 — Put the return period into the metric name.** `q_return_level_10yr_max`.
Costs nothing and stops the file being ambiguous about which return period it
holds.

## Shape

**C6 — Locations become rows, not columns.** This is the main change. The file
structure stops depending on the basin.

**C7 — One file per variable.** Discharge, evapotranspiration, recharge and so on
each get their own table. This removes the need for a "which variable is this"
column and lets each variable use the metric names and units that suit it.

**C8 — Keep a `location` column even where there is only one value today.** It
costs one column, means the tables can be stacked together without editing, and
is where subbasin identifiers will go under C20.

## Realizations

**C9 — Retire the pool-everything switch.** In the new shape, "pooled" is no
longer a different file layout — it is just extra rows — so the switch has
nothing left to switch.

**C10 — Give each statistic the pooling that suits it,** in three groups:

- *Per realization:* mean, max, min, 95th percentile, 7-day maximum and minimum,
  baseflow index. These are all "yearly value, then average over years", and
  because realizations are the same length, averaging the per-realization numbers
  gives exactly the pooled number back. Nothing is lost by reporting them
  separately, and you gain the spread.
- *Pooled only:* the two return-period statistics. Fitting a 10-year flood to one
  short realization is unreliable; pooling multiplies the sample by the number of
  realizations. Reporting a per-realization version alongside would just invite
  someone to use it.
- *Special case:* the wettest-month and driest-month statistics, which have to
  choose a month before they can average anything — see C12.

**C11 — `realization_id = 0` means pooled.** A number rather than a word keeps
the column numeric. It is safe only because, under C10, no statistic ever appears
in both forms — if that ever changes, this has to become a word, and that
condition is written down in the specification.

## Seasonal statistics

**C12 — Choose the wettest and driest month once, from the unperturbed baseline,
and use it everywhere.** Otherwise each stress-test member could pick a different
month and the response surface would be comparing January against July without
saying so. If the *shift* in wettest month is itself interesting, that is a
different indicator and can be added later.

**C13 — Include the unperturbed baseline run in the results table.** It currently
does not appear when realizations are pooled, so the response surface has no
origin point — and C12 cannot pick a month from a record that is not there.

## Numbers

**C14 — Store values at single precision and do not round them.** The original
request was to round everything to two decimals, but that would have destroyed
the low-flow half of the results: a 7-day minimum flow of `0.0034` becomes
`0.00`. Single precision gives short, readable numbers *and* keeps small values,
so no rounding rule is needed at all.

**C15 — Keep the model's own units.** Where a unit conversion is needed it is
done properly or not at all. Concretely this fixes F2: overland flow is
summarised as a yearly average flow rate rather than added up, while
evapotranspiration and recharge keep their yearly totals, because adding up a
daily depth is a legitimate total and adding up a flow rate is not.

**C16 — Pool the yearly extremes, not the stitched-together record.** Compute
each realization's yearly maxima and minima within that realization, then combine
those before fitting the return period. This fixes F3 and is simpler than what it
replaces — the artificial calendar disappears entirely.

## Locations

**C17 — Use the plain gauge number** (`130000086`), not `Q_130000086`. The `Q_`
was only ever there because it was a column header.

**C18 — Give the basin-averaged variables the same location numbers as
discharge.** This needs no translation work: the model already labels its gauges
with the subbasin identifier, so discharge at a gauge and evapotranspiration over
that gauge's catchment already share a key. It means a gauge's flow statistics and
its catchment's water balance can sit side by side.

**C19 — Produce the overall basin figure directly, not by averaging the
per-location ones.** Whether that averaging would even be valid depends on O1, and
producing the basin figure directly is both safer and already possible.

**C20 — Report precipitation as a subbasin mean rather than at gauge points.**
More useful for this tool than rainfall at an exact point, and it also stops F4.
Basin-wide averaging needs only a configuration change; per-subbasin averaging
needs a small change to how the model is set up.

## Checks

**C21 — Move the reference check for these tables to a tolerance comparison**
instead of an exact match. Without rounding (C14), an exact match would fail on
numerically meaningless differences. The comparison this needs already exists and
is used for the discharge reference.

---

# Part B — how stress-test runs are identified

## The short version

Every run is currently identified by its filename — `rlz_2_cst_37` — and what
`cst_37` actually *is* exists nowhere except inside `cst_37.csv`, as twelve
monthly rows. We propose writing a **master design table**: one row per
stress-test point, listing what climate perturbation it applies. Runs keep a
readable two-part identity, but the meaning finally lives somewhere you can look
it up.

We also rename the stress-test token from `cst_` to `st_`.

## Why this matters more than it looks

Part A's main achievement was making the results table shape independent of the
basin. But under **multi-dimensional** stress testing, the current design breaks
that from the other end: `temp_change` and `precip_change` are columns, so adding
a seasonality or spell-length dimension adds a column, and the header depends on
the configuration again.

A design table with an id fixes that permanently — the results tables reference a
point instead of describing it, and their shape stops depending on how many
stress dimensions there are. That is the real argument for this change, and it is
why it belongs alongside Part A rather than after it.

## The changes

**C22 — Rename the stress-test token from `cst_` to `st_`.** `cst` is the name of
the whole tool, so using it for one member of a grid inside that tool says
nothing. More to the point, **the code already says `st`** — the workflow
wildcard is `st_num`, the count is `ST_NUM`, the helper is `stress_test_grid()`,
the configuration section is `stress_test:`. Only the *files* say `cst_`. The
Snakefile even builds a filename called `cst_...` out of a wildcard called
`st_num`. So this removes an inconsistency that already exists rather than
introducing churn. `rlz_` stays as it is: unlike `cst`, it is a terse
abbreviation of a *correct* term — realization is the standard word for a
stochastic replicate, and there is no collision to fix.

**C23 — Write a stress-test design table.** One row per design point, listing the
perturbation it applies, with a row for the unperturbed baseline where every
change is zero. This is the artifact that is missing today: a single place that
answers "what was run 37?". It also becomes the natural place to add a dimension
— a new column, rather than a new naming convention.

**C24 — Keep two identifiers, not one.** The design point and the realization are
different kinds of thing and should not collapse into a single counter:

- the **design point** is *designed* — enumerable, meaningful, worth looking up;
- the **realization** is *sampled* — realization 7 has no parameters, it is
  simply draw 7, and there is nothing to look up.

Four practical consequences of keeping them apart. The return-period statistics
pool over realizations but *not* over design points (C10), which a single
identifier cannot express. Adding realizations would otherwise renumber the whole
design. The workflow engine can still select "everything for realization 2" as a
pattern, which the batching work depends on. And when a run fails, the log name
tells you what broke instead of sending you to a lookup table.

**C25 — Design-point numbers are meaningful within one experiment, not across
experiments.** A sequential number over an enumerated grid changes meaning the
moment the grid changes — add one temperature step and everything after it
shifts. Rather than pretend otherwise, the design table lives in the experiment
folder next to the configuration snapshot that produced it, which is already the
place where settings are pinned. A different configuration is a different
experiment, which is how the tool already works.

*Considered and rejected:* numbering derived from the parameter values
themselves, which would be stable across grid changes but unreadable and
unsortable; and numbering that encodes the grid position, which is
self-describing but grows a segment for every dimension added — the thing this
change exists to avoid.

**C26 — The list of design points is worked out once and used twice.** The same
routine that tells the workflow engine which runs to plan also writes the design
table, so the two cannot disagree. Without this there is a circularity — the
engine needs to know the runs before any of them has produced a file.

**C27 — The number width follows the count rather than being fixed.** Three
digits caps at 999, and ten realizations across a 5 × 5 × 3 grid is already 750
before a fourth dimension is added.

**C28 — The results tables carry `st_id` *alongside* `temp_change` and
`precip_change`, not instead of them.** Decided 2026-08-05, **for this stage**.

The alternative was to drop the two perturbation columns and let `st_id` be the
only link to what a run was. That would have kept the results shape fixed at any
number of stress dimensions — the property Part A exists for — at the cost of
needing a join before anything can be plotted. Keeping both means the files stay
readable on their own, which is what has driven most of the Part A decisions.

**What this costs, stated plainly.** The two columns are now a *cached copy* of
what the design table holds, so they can disagree with it if anything goes wrong
in writing them, and the results shape is again tied to the number of stress
dimensions — adding a third adds a column. That is acceptable while stress
testing is two-dimensional. It stops being acceptable the moment it is not, which
is the whole reason Part B exists.

**So the decision carries a trigger rather than a hope.** Two things follow:

- the writer must check, for every row, that `temp_change` and `precip_change`
  match the design table's entry for that `st_id` — a cached copy that nothing
  verifies is a copy that eventually lies;
- when the stress-test configuration gains a **third dimension**, the results
  writer should stop and say so, naming this decision, rather than quietly adding
  a column. At that point the choice is made again with the extra dimension
  actually in hand.

---

# Part C — generator plumbing

**C29 — Retire the per-run weather-generator configuration file.** *(Proposed
2026-08-05, not yet ruled.)*

Every perturbed run currently gets its own small configuration file,
`weathergen_config_rlz_<m>_cst_<n>.yml`, written by its own workflow step with
its own log and timing record. On a ten-realization, eighty-eight-point sweep
that is 880 configuration files, 880 logs and 880 timing files.

They contain almost nothing. The only thing that varies between them is **the
name of the file the run should write** — split into a prefix and a suffix
because the generator's write routine takes them separately. Everything else is
identical in all 880: two on/off switches, and a copy of the perturbation
settings that the generator does not read (see F6).

The workflow engine already knows the output filename — it is the step's own
declared output. So the filename can be passed straight to the R script as an
argument, the two switches belong in the single shared generator configuration
that already exists, and the whole per-run step disappears.

---

# Findings, open items and cost

These span the whole document, not just Part A.

## Findings

None of these were the point of the exercise; all were found by reading the code
the changes touch. None are fixed yet.

**F1 — The perturbation columns hold January, not the year.** The stress-test
files contain twelve monthly values, and the results table reads only the first
and labels it as the member's perturbation. This has never been visibly wrong
because the shipped settings use the same value for all twelve months — but a
project using a seasonal perturbation would get a response surface secretly
indexed by its January figure. **Not addressed by any change here:** deciding what
twelve months should collapse to (a mean? an annual total?) is a method question,
so it is recorded rather than patched.

**F2 — Overland flow is added up in the wrong units.** Evapotranspiration and
recharge are produced as millimetres per day, so adding up a year of them
correctly gives millimetres per year. Overland flow is produced as cubic metres
per *second*, and adding those up gives a quantity that is not millimetres, not
cubic metres, and not anything — off by a factor of 86400. Any project asking for
overland flow currently gets a wrong column under a wrong stated unit.
**Addressed by C15.**

**F3 — The 7-day averaging window runs across the join between realizations.**
When realizations are pooled they are currently stapled end to end into one
artificial record. A 7-day average then spans the seam and can manufacture a low
flow that occurred in none of the realizations — which becomes that year's
minimum and feeds the low-flow return period. It matters most in basins whose dry
season falls near the turn of the year. **Addressed by C16.**

**F4 — Rainfall at gauges is calculated and then thrown away.** The model is
already set up to report it at gauge locations, and the results step silently
discards those columns. **Addressed by C20.**

**F5 — Run numbers silently change meaning when the configuration changes.**
`cst_37` is the thirty-seventh point of an enumerated grid, so adding a single
temperature step shifts everything after the first block — the same filename then
refers to a different climate. This is true today and always has been; nothing
warns about it, and results carried between runs can be compared under the
assumption that the numbers mean the same thing. It becomes more frequent once
stress testing is multi-dimensional. **Addressed by C25**, which scopes the
numbering to one experiment and writes down what each number meant.

**F6 — The per-run generator configuration carries no per-run information, and
the part of it that looks informative is ignored.** Each
`weathergen_config_rlz_<m>_cst_<n>.yml` copies in the full temperature and
precipitation perturbation settings from the project configuration — the step
counts and the monthly minimum and maximum ranges. The R script reads **only the
two transient on/off switches** from those blocks. The numbers that actually
perturb the run come from `cst_<n>.csv`, not from here.

So anyone opening one of these files to find out what a run did would read
plausible-looking perturbation ranges that had no part in it. The rest of the
file is a filename. **Addressed by C29.**

---

## Open decisions

**O1 — Do the subbasins overlap?** If each gauge's area is its full upstream
catchment, they nest inside one another; if each is only its own local area, they
tile. This decides whether an overall basin figure could be averaged from them —
though C19 recommends not doing that regardless.

**O2 — What should happen to an old configuration file** that still contains the
retired pooling switch (C9). It would currently be ignored in silence, leaving
the user believing it still does something. The recommendation is to refuse to
run and say so.

**O3 — CLOSED 2026-08-05 → see C28.** Ruled *alongside*: the results tables carry
`st_id` **and** the perturbation columns, for this stage. The recommendation had
been *replace*, for a permanently fixed shape; the owner chose readability now
with an explicit revisit when a third dimension arrives. C28 records the cost and
the trigger.

---

## What this costs

- **The stress-test workflow has to be re-run once** on the test project, and the
  reference fingerprints re-recorded. C1 is already waiting on that same run, so
  it should be done once at the end, not per change.
- **A written record of the old and new names** is required by the repository's
  own conventions, covering both the column renames and the removal of the
  pooling switch (C9).
- **The results workflow starts depending on the model configuration** to know
  which variable files to produce (C7). It currently discovers this while
  running, which is too late for the workflow engine to plan the job.

---

# Appendix — how a stress test is actually built today

Reference for reviewing the changes above. Traced from
`Snakefile_climate_experiment`, 2026-08-05. Step numbers are the workflow's own.

```
config stress_test:          extract_historical.nc        WF1 model
        │                            │                    (staticmaps, toml)
        ▼                            ▼                        │
  3.03 climate_stress_parameters                              │
   prepare_cst_parameters.py                                  │
        │                                                     │
        ├──► _work/cst_1..N.csv    ◄── 12 monthly rows each   │
        │      (temp_mean, precip_mean, precip_variance)      │
        │                                                     │
        │    3.04 prepare_weagen_config                       │
        │      └──► config/weathergen_config.yml  (ONE file)  │
        │                     │                               │
        │                     ▼                               │
        │        3.06 generate_weather_realization            │
        │           generate_weather.R  (weathergenr)         │
        │              └──► output/rlz_1..R_cst_0.nc   temp() │
        │                     │                               │
        │  3.05 prepare_weagen_config_st                      │
        │    └──► _work/weathergen_config_rlz_m_cst_n.yml     │
        │                     │                               │
        ▼                     ▼                               │
      3.07 generate_climate_stress_test                       │
         impose_climate_change.R                              │
         weathergenr::apply_climate_perturbations             │
           └──► output/rlz_m_cst_n.nc              temp()     │
                        │                                     │
                        ▼                                     │
              3.08 climate_data_catalog                       │
                └──► data_catalog_climate_experiment.yml      │
                        │                                     │
                        ▼                                     ▼
              3.09 downscale_climate_realization ◄────────────┘
                 hydromt setup_precip_forcing
                        + setup_temp_pet_forcing
                ├──► forcing/inmaps_rlz_m_cst_n.nc   temp()
                └──► config/rlz_m_cst_n.toml
                        │
                        ▼
              3.10 run_wflow_batch_<b>   (B members per Julia session)
                ├──► output/rlz_m_cst_n.csv
                └──► output/outstates_rlz_m_cst_n.nc   temp()
                        │
                        ▼
              3.11 derive_wflow_indicators ◄─── also reads cst_1..N.csv
                └──► results/q_indicators.csv, basin_indicators.csv
```

**Where the perturbation is actually applied:** step 3.07, inside R.
`apply_climate_perturbations` receives `precip_mean_factor` (multiplicative),
`precip_var_factor` (multiplicative) and `temp_delta` (additive °C) directly from
the twelve rows of `cst_<n>.csv`, together with `compute_pet = TRUE` — so
potential evaporation is computed there by the weather generator, and then
computed again from temperature by hydromt at step 3.09.

**`cst_<n>.csv` — one producer, two consumers that read it differently:**

| | |
|---|---|
| written by | 3.03, from the project configuration's `stress_test:` block |
| read by 3.07 | the R generator — uses **all twelve monthly values** |
| read by 3.11 | the results reducer — uses **the first row only**, i.e. January |

That asymmetry is F1. The file is correct and the generator reads it correctly;
only the results reducer under-reads it. The defect is one line in one consumer,
not a data problem.

**`weathergen_config_rlz_<m>_cst_<n>.yml` — see F6 and C29.** The R script reads
five values from it: an output directory, a filename prefix, a filename suffix,
and two transient on/off switches. Only the prefix and suffix vary between runs.
