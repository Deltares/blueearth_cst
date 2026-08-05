# Proposal: reshaping the WF3 result tables

Plain-language summary of what we propose to change about the stress-test result
tables, and why. Written to be read, not parsed — the precise specification,
with file and line references, lives in `wf3-output-change-requests.md`.

Decided 2026-08-05. Nothing here is implemented yet except the first item.

---

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
  `temp` and `precip`. *(Already fixed — see the first decision below.)*
- **All-or-nothing pooling.** A single switch decided whether *every* statistic
  was averaged over realizations or *none* was. There was no way to say "pool
  the return periods, because a single realization is too short to estimate a
  10-year flood, but keep everything else per realization" — which is what you
  actually want.
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
Every file has the same six columns:

| column | what it holds |
|---|---|
| `metric` | what was calculated, e.g. `q_mean_annual_7day_min` |
| `temp_change` | the temperature perturbation, in °C |
| `precip_change` | the precipitation perturbation, in % |
| `realization_id` | which realization — or `0`, meaning "pooled over all of them" |
| `location` | the gauge or subbasin number |
| `value` | the number |

---

## The decisions, and why

### Naming

- **`tavg` → `temp_change`, `prcp` → `precip_change`.** These were the only place
  in the whole repository that used different names for temperature and
  precipitation. *(Done — this part has already landed.)*
- **`_change`, not bare `temp`/`precip`.** These columns hold the *perturbation
  applied*, not the temperature or rainfall itself — and the two are not even the
  same kind of number (°C added, versus % change). Calling a column `temp` in a
  results table would read as a temperature.
- **The metric name carries the variable** — `q_mean_annual_max`, not `max`. This
  makes each row self-explanatory when the file is opened on its own or emailed
  to someone, which is worth the small redundancy with the filename.
- **We do *not* use the short conventional abbreviations** like `Q95` or `Q10`,
  even though they are standard. The reason is specific: what this tool computes
  as "q95" is the mean annual **95th percentile**, a high flow — whereas `Q95` in
  hydrology conventionally means the flow *exceeded* 95% of the time, which is a
  low-flow drought index. They are opposite ends of the distribution. Borrowing
  the abbreviation would mislead exactly the reader who knows what it usually
  means. We spell it out instead: `q_mean_annual_p95`.
- **The names say "mean annual".** Almost every statistic here is "compute it for
  each year, then average across years" — the old names hid that. `Qmax` becomes
  `q_mean_annual_max`, which is what it has always been.
- **The return period goes into the name** — `q_return_level_10yr_max`. This costs
  nothing and stops the file from being ambiguous about what it contains.

### Shape

- **Locations become rows, not columns.** This is the main change. The file
  structure stops depending on the basin.
- **One file per variable.** Discharge, evapotranspiration, recharge and so on
  each get their own table, which removes the need for a "which variable is this"
  column and lets each variable use the metric names and units that suit it.
- **We keep a `location` column even where there is only one value today.** It
  costs one column and means the tables can be stacked together without editing,
  and it is where subbasin identifiers will go if we go ahead with reporting per
  subbasin.

### Realizations

- **The single pool-everything switch is retired.** In the new shape, "pooled" is
  no longer a different file layout — it is just extra rows — so the switch has
  nothing left to switch.
- **Each statistic now gets the treatment that suits it**, in three groups:
  - *Per realization:* mean, max, min, 95th percentile, 7-day maximum and
    minimum, baseflow index. These are all "yearly value, then average over
    years", and because realizations are the same length, averaging the
    per-realization numbers gives exactly the pooled number back. Nothing is
    lost by reporting them separately, and you gain the spread.
  - *Pooled only:* the two return-period statistics. Fitting a 10-year flood to
    one short realization is unreliable; pooling multiplies the sample by the
    number of realizations. Reporting a per-realization version alongside would
    just invite someone to use it.
  - *Special case:* the wettest-month and driest-month statistics, which have to
    choose a month before they can average anything.
- **`realization_id = 0` means pooled.** A number rather than a word keeps the
  column numeric. It is safe here only because no statistic ever appears in both
  forms — if that ever changes, this has to become a word, and that is written
  down.

### The wettest / driest month

- **The month is chosen once, from the unperturbed baseline, and used
  everywhere.** Otherwise each stress-test member could pick a different month
  and the response surface would be comparing January against July without
  saying so. If the *shift* in wettest month is itself interesting, that is a
  different indicator and can be added later.
- **A consequence:** the unperturbed baseline run must appear in the results
  table. It currently does not, when realizations are pooled — so the response
  surface has no origin point. That gets fixed as part of this.

### Numbers

- **Values are stored at single precision and not rounded.** The original request
  was to round everything to two decimals, but that would have destroyed the
  low-flow half of the results — a 7-day minimum flow of 0.0034 becomes 0.00.
  Single precision gives short, readable numbers *and* keeps small values, so no
  rounding rule is needed at all.
- **Units stay as the model produces them.** Where a conversion is needed it is
  done properly or not at all; see the overland-flow problem below.

### Locations

- **The location is the plain gauge number** (`130000086`), not `Q_130000086`.
  The `Q_` was only there because it was a column header.
- **The basin-averaged variables will use the same location numbers as
  discharge.** This turned out to need no translation work at all: the model
  already labels its gauges with the subbasin identifier, so discharge at a gauge
  and evapotranspiration over that gauge's catchment already share a key. It
  means you can put a gauge's flow statistics and its catchment's water balance
  side by side.
- **An overall basin figure will be produced separately**, not calculated from
  the per-location ones. Whether that calculation would even be valid depends on
  whether the subbasins overlap, and producing the basin figure directly is both
  safer and already possible.

---

## Problems found along the way

None of these were the point of the exercise; all were found by reading the code
the changes touch. None are fixed yet.

- **The perturbation columns hold January, not the year.** The stress-test files
  contain twelve monthly values, and the results table reads only the first one
  and labels it as the member's perturbation. This has never been visibly wrong
  because the shipped settings use the same value for all twelve months — but a
  project using a seasonal perturbation would get a response surface secretly
  indexed by its January figure. Fixing it is a method question (should twelve
  months become a mean? an annual total?), so it is recorded rather than patched.
- **Overland flow is added up in the wrong units.** Evapotranspiration and
  recharge are produced as millimetres per day, so adding up a year of them
  correctly gives millimetres per year. Overland flow is produced as cubic metres
  per *second*, and adding those up gives a quantity that is not millimetres, not
  cubic metres, and not anything — off by a factor of 86400. Any project that
  asks for overland flow currently gets a wrong column with a wrong stated unit.
- **The 7-day averaging window runs across the join between realizations.** When
  realizations are pooled, they are currently stapled end to end into one
  artificial record. A 7-day average then spans the seam and can manufacture a
  low flow that occurred in none of the realizations — which then becomes that
  year's minimum and feeds the low-flow return period. It matters most in basins
  whose dry season falls near the turn of the year. The fix is to compute each
  realization's yearly extremes first and pool *those*, which is also simpler.
- **Precipitation at gauges is calculated and then thrown away.** The model is
  already set up to report rainfall at gauge locations, and the results step
  silently discards those columns.

---

## Still to decide

- **Whether the subbasins overlap.** If each gauge's area is its full upstream
  catchment, they nest inside each other; if each is only its own local area,
  they tile. This decides whether an overall basin figure can be averaged from
  them — though the recommendation is not to do that anyway.
- **What should happen to an old configuration file** that still contains the
  retired pooling switch. It would currently be ignored in silence, leaving the
  user believing it still does something. The recommendation is to refuse to run
  and say so.

---

## What this costs

- **The stress-test workflow has to be re-run once** on the test project, and the
  reference fingerprints re-recorded. Two changes are already waiting on that same
  run, so it should be done once at the end, not per change.
- **The reference check for these two tables moves to a tolerance comparison**
  rather than an exact-match one. Without rounding, an exact match would fail on
  numerically meaningless differences.
- **A written record of the old and new names** is required by the repository's
  own conventions, and covers both the column renames and the removal of the
  pooling switch.
- **The results workflow starts depending on the model configuration** to know
  which variable files to produce. It currently discovers this while running,
  which is too late for the workflow engine to plan the job.
