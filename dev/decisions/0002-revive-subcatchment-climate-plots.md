Status: superseded by 0006-retire-subcatchment-climate-plots.md (2026-08-09)
Date: 2026-07-21
Deciders: Ümit Taner
Consulted: Observation 4 investigation + correction (test/pre06 gabon run;
           plot_results.log); commit af38068 (honest-message fix that surfaced
           the dormant path); existing plot_forcing precedent (plot_map_forcing.py)
Supersedes: none
Superseded-by: 0006-retire-subcatchment-climate-plots.md — the canonical
           climate figure set now answers this question for both gridded
           products, so the subcatchment figures were duplication. The
           diagnosis below (the branch was dead and reported its skip
           misleadingly) remains accurate and is why it was fixed rather
           than deleted at the time.
Revisions:
  - 2026-07-21: initial draft (scoping only). Proposed wiring wflow OUTPUT
    variables (extend WFLOW_VARS, emit *_subcatchment CSV columns, opt-in).
  - 2026-07-21: revised after user correction — the climate plots must use the
    climate data INPUT to wflow (the ERA5-derived forcing inmaps_historical.nc),
    not wflow output variables. Data-source mechanism reversed; the old output-
    variable approach demoted to a rejected alternative. Still proposed, no code.
  - 2026-07-21: IMPLEMENTED and accepted (test/pre06). New `src/climate_forcing.py`
    (`climate_forcing_by_subcatchment`) aggregates the gridded forcing to a
    per-subcatchment mean `(index, time)` dataset with the `P/T/EP_subcatchment`
    keys `plot_clim` expects; `src/plot_results.py` §4 now builds `ds_clim` from
    `mod.forcing.data` instead of wflow outputs; `Snakefile_model_creation`
    declares `inmaps_historical.nc` as a `plot_results` input. Unit tests in
    `tests/test_climate_forcing.py`. Verified end-to-end on the real gabon model
    (both `clim_wflow_<id>_year/month.png` render; physically sensible
    climatology). Full visual QA still occurs on the user's next WF1 run.

# ADR 0002 — Revive the subcatchment climate plots from the wflow forcing input

### Context

`src/plot_results.py` §4 draws yearly and monthly climate figures (temperature,
precipitation, potential ET) at the subcatchment scale via
`src/func_plot_signature.py::plot_clim`, which reads three series keyed
`T_subcatchment` (averaged), `P_subcatchment` and `EP_subcatchment` (summed to
period accumulations). Today it builds `ds_clim` from the wflow **output** CSV,
looking for columns literally named `P/T/EP_subcatchment`. No configuration
produces those columns, so `ds_clim` is always empty and §4 always skips. On the
gabon run this printed a misleading "less than 1 year of data" line even though
`output.csv` holds 20 years; commit `af38068` corrected the **message**.

The correct data source is the climate **input** to wflow, not its output. Those
three variables are precisely the forcing that drives the model, written by the
`add_forcing` rule into `climate_historical/wflow_data/inmaps_historical.nc`
(ERA5 in the test case). Inspection of the gabon file confirms it carries
`precip` (mm d⁻¹), `pet`, and `temp` on the model grid (time × latitude ×
longitude; 2000–2020 daily). Reading wflow *outputs* to recover P/T/PET would
make the model echo its own inputs back out — redundant, and it would perturb the
manifest-fingerprinted `output.csv`.

A precedent already exists in this repo: the `plot_forcing` rule
(`src/plot_map_forcing.py`) reads the forcing via `mod.forcing.data`, masks it
with `staticmaps["subcatchment"] >= 0`, and reduces precip/pet by sum and temp by
mean — the same variables, access path, and reduction logic `plot_clim` needs. So
the fix is to aggregate the forcing to a per-subcatchment timeseries, not to wire
any new outputs.

### Decision

We will **revive** the climate plots by sourcing P/T/EP from the wflow forcing
input and feeding them to the existing `plot_clim`:

1. In `plot_results.py`, obtain the gridded forcing from the already-instantiated
   model via `mod.forcing.data` (variables `precip`, `temp`, `pet` =
   `inmaps_historical.nc`). Declare `inmaps_historical.nc` as an explicit input of
   the `plot_results` rule so the DAG dependency is honest.
2. Reduce each variable spatially to a per-subcatchment timeseries using
   `staticmaps["subcatchment"]` (group by subcatchment id so the `index`
   dimension matches the discharge stations used elsewhere), yielding an
   `index × time` dataset. Map `precip → P_subcatchment`, `temp → T_subcatchment`,
   `pet → EP_subcatchment` so `plot_clim` stays **unchanged** (it keeps summing
   P/EP and averaging T over the resample period).
3. Keep it **default-on**: the source data already exists after `add_forcing`, and
   nothing here touches `output.csv`, so the baseline manifest is unaffected and no
   opt-in gate or config surface is needed.

The already-committed honest message in `plot_results.py` §4 remains as a guard for
the rare case the forcing cannot be read (e.g. `mod.forcing.data` empty).

### Consequences

*Positive*
- The yearly/monthly climate QA figures render by default from the actual model
  forcing — the intended, non-redundant source.
- Zero change to `output.csv`, `WFLOW_VARS`, or `setup_gauges_and_outputs` → the
  sealed discharge baseline manifest is untouched; no re-record, no opt-in gate.
- Reuses the proven `plot_forcing` access/masking pattern (`mod.forcing.data` +
  `staticmaps["subcatchment"]`), lowering implementation and review risk.

*Negative*
- `plot_results` gains a dependency on the forcing file; the rule's `input:` must
  add `inmaps_historical.nc` (and the plotting must degrade gracefully if forcing
  is absent — the existing guard message covers this).
- Per-subcatchment spatial aggregation is new code to write and test (grouping the
  forcing grid by the subcatchment map, handling the nodata mask).

*Neutral*
- Units carry from the forcing: `precip` mm d⁻¹ (summed to period totals), `pet`
  same (units attr unset in the file — assume mm d⁻¹, verify), `temp` °C
  (averaged). Matches `plot_clim`'s existing sum/mean split.
- The `ds_clim.time < 365` branch stays meaningful (short forcing → skip yearly
  plots); the "absent" branch becomes a rare fallback, not the normal path.

### Alternatives considered

- **Wire wflow OUTPUT variables** (the original draft): extend `WFLOW_VARS` with
  temperature + potential-ET CSDMS names, emit `*_subcatchment` CSV columns via
  `setup_config_output_timeseries`, opt-in/default-off. **Rejected** on the user
  correction: it makes wflow re-emit its own forcing inputs (redundant), perturbs
  the manifest-fingerprinted `output.csv` (forcing a re-record or an opt-in gate),
  and needs a `WFLOW_VARS` extension the forcing-input path avoids entirely. Would
  only be preferred if the desired series were genuine model *state/flux* outputs
  rather than input forcing.
- **Remove the dead §4 code** (and `plot_clim`). Simpler, but discards legitimate
  forcing-climatology QA the user wants; the forcing-input fix is cheap and reuses
  an existing pattern. Preferred only if the plots were judged redundant with the
  projection workflow.
- **Basin-mean instead of per-subcatchment**: aggregate the whole basin to one
  series (as `plot_forcing` does for its maps). Simpler, but loses the per-station
  `index` alignment `plot_clim` iterates; acceptable as a first cut for
  single-outlet basins, but per-subcatchment is the general form.

### Related

- `src/plot_results.py` §4 and `src/func_plot_signature.py::plot_clim` — the plot
  contract (`P/T/EP_subcatchment`).
- `src/plot_map_forcing.py::plot_forcing` — the precedent: forcing via
  `mod.forcing.data`, masking via `staticmaps["subcatchment"]`, sum/mean reduction.
- `src/setup_time_horizon.py` / `add_forcing` rule — produces
  `inmaps_historical.nc` (`precip`, `temp`, `pet`).
- Commit `af38068` — the honest-message fix that exposed this dormant path.
- Baseline manifest (`dev/scripts/check_baseline.py`; memory: baseline-manifest-*)
  — unaffected here because `output.csv` is not touched (the key win over the
  rejected output-variable approach).
