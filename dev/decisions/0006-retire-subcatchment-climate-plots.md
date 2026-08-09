Status: accepted
Date: 2026-08-09
Deciders: Ümit Taner
Consulted: the canonical climate figure set (`climate_analysis/climate_figures.py`)
           and the cartographic template (`shared/cartographic_map.py`), both of
           which reached their current form during the 2026-08 plotting work
Supersedes: 0002-revive-subcatchment-climate-plots.md

# ADR 0006 — Retire the subcatchment climate plots

### Context

ADR 0002 revived `plot_results.py` §4, which drew per-subcatchment yearly and
monthly climate figures (`evaluation/plots/clim_wflow_<id>_{month,year}.png`)
via `func_plot_signature.plot_clim`. That decision was correct for its moment:
the branch was dead — it looked for `P/T/EP_subcatchment` columns no config
produced, so it always skipped, and it reported the skip as "less than 1 year of
data" on a twenty-year run. ADR 0002 rewired it to read the model's climate
INPUT and made it render.

What has changed is not that decision but its surroundings. Since 2026-08 the
toolbox has ONE canonical climate figure set (`climate_figures.py`) applied to
both gridded products — the source extraction and the wflow forcing — drawn
through the shared cartographic template. Each variable now gets a map, an
annual series with its trend, and a monthly climatology with its interannual
spread, at `models/hydrology/wflow/forcing/plots/` and
`data/climate/historical/<key>/plots/`, framed alike and sharing colour scales
so the two are directly comparable.

The subcatchment figures answer a question that set already answers, from the
same underlying climate, in a folder whose other contents are per-location
DISCHARGE evaluation: `hydro_<station>.png` and `signatures_<station>.png`. A
reader opening `evaluation/plots/` to judge model performance meets climate
figures that are neither performance nor located where the rest of the climate
lives.

### Decision

Retire the subcatchment climate plots. `evaluation/plots/` holds hydrological
evaluation and signatures per location, and nothing else.

Removed with them, because nothing else used them:

- `plot_results.py` §4 and the `ds_clim` construction feeding it
- `climate_analysis/subcatchment_climate.py`
  (`climate_forcing_by_subcatchment`) and `tests/test_climate_forcing.py`
- `func_plot_signature.plot_clim` and `_plot_clim_year`
- the two declared outputs on rule 1.15, and the climate-store inputs and
  params that only served this branch

Rule 1.15 no longer reads the climate store at all, so it no longer waits on
rule 1.04. This also retires a failure mode that declaring the figures
introduced (O-24): a config with a sub-year `historical_window` failed the rule
with `MissingOutputException` where it had previously logged a skip.

### Consequences

- The per-SUBCATCHMENT view of climate is gone. The new set is per-GRID and
  domain-mean, so a user who wants climate for one subcatchment specifically no
  longer has a figure for it. Accepted: no analysis in the toolbox consumed
  those figures, and the boxes on the monthly chart carry the variability the
  per-subcatchment split was mostly being read for.
- `plot_results.py` shrinks to what its name says — it evaluates a run against
  observations.
- Anything reproducing an old run's output tree will find two fewer figures per
  station. They were declared outputs, so `--delete-all-output` removed them;
  a stale tree keeps them until cleaned.

### Alternatives considered

**Relocate rather than delete** — move the figures under `forcing/plots/`
beside the rest of the climate set. Rejected: it keeps two code paths drawing
the same quantities from the same data, which is the duplication this decision
exists to remove, and the relocated figures would not share the new set's
framing, palettes or colour scales without being rewritten into it anyway.

**Stop declaring them, keep writing them** — leaves undeclared outputs, which
O-24 exists to prevent: they are not cleaned on rerun and are absent from the
baseline.
