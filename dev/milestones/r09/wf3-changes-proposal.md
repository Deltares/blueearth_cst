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

### Across the document

| | |
|---|---|
| **Findings** | F1 January-only perturbation · F2 overland flow units · F3 seam in the 7-day window · F4 gauge rainfall discarded |
| **Open** | O1 do subbasins overlap · O2 stale configuration switch |

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
