# Post-R7 self-check — challenge register

Live register of the owner's own self-check after the **R7 project-layout**
milestone sealed. Opened 2026-07-29 against `43f8dfa` (`main`).

Unlike a defect log, this register is about **decisions**: places where an
earlier design or implementation choice — R7's or any earlier milestone's —
now looks questionable in hindsight. Each entry records the challenge, the
reasoning behind the original choice, the alternatives, and where it landed.
Entries are worked one at a time, in conversation, and written up here as they
are resolved.

**Scope and boundary.**

- **In scope:** any decision or implementation from Phase 2 (R1–R7) or Phase 3
  planning that the owner now wants to re-examine — not just R7's surface.
- **Out of scope / do not re-log:** the `## Post-R7` items already in
  [`../followups.md`](../followups.md) (`R7-1`, `R7-2`, `R7-3`, `R7-21`) and the
  post-R6 observations in
  [`2026-07-25_post-r6-assessment.md`](2026-07-25_post-r6-assessment.md)
  (`O-nn`). Cross-reference those by ID; never duplicate them here.
- **Promotion:** an entry that needs work in a later milestone goes to
  `../followups.md`; one needing tracked multi-session work gets a `../TODO.md`
  row; one that changes a standing design position gets a `../decisions/` ADR.
  The **Disposition** field keeps the pointer, so nothing lives in two registers.

**ID prefix:** `S7-nn` (self-check, opened post-R7). Distinct from `O-nn`
(post-R6 register) and `R7-nn` (followups).

**Status vocabulary:** `open` (recorded, not yet discussed) · `discussed`
(reasoning and alternatives worked through, decision pending) · `resolved`
(decision made; disposition names the outcome) · `promoted` (routed to
followups/TODO/ADR) · `no-change` (challenge considered, original decision
stands — with the sharper reason recorded).

**How to add an entry.** Append the next `S7-nn` index row, then add the matching
detail block below with all five headings filled in. A block must read standalone
once the conversation that produced it is gone.

---

## Index

| ID | Challenge | Area | Status | Created | Updated | Disposition |
|---|---|---|---|---|---|---|
| S7-01 | wf1 figures are scattered across three `plots/` dirs; consolidate to one per product area | layout | discussed | 2026-07-29 | 2026-07-29 | Owner proposes 2 buckets; recommendation = adopt as a pure path/name move. Pending ruling |

---

## Entries

### S7-01 — wf1 figures are scattered across three `plots/` dirs

- **Area:** layout
- **Status:** discussed
- **Created / Updated:** 2026-07-29 / 2026-07-29
- **Rev:** `43f8dfa`

**Challenge.** Browsing a finished wf1 run, climate figures show up in two
places — `climate_historical/era5_20000101_20201231/plots/` and
`hydrology_model/forcing/plots/` — with near-identical names. Opening question
was "why?"; on discussion the owner sharpened it into a **simplification
proposal**: too much scatter degrades usability, so wf1 should expose **two**
plot locations, one for climate-data figures (under the climate store) and one
for model figures (under `hydrology_model/`).

Two supporting claims were raised and are assessed below: (a) the
`forcing/plots/` P/T/PET maps could be made from the raw source data, and
(b) `clim_wflow_1_{month,year}.png` could be made from the forcing data and does
not belong under `evaluation/`.

**What we decided, and why.** R07 §B4 (`dev/r07/project-layout-design.md:611`)
recorded three figure families, each answering a different question, and R07
principle **P1** placed each artifact beside the data it describes:

| Product | Question | Grid | Needs a model? | Home (post-R7) |
|---|---|---|---|---|
| Source climate (rule 1.15) | what does the source climate look like? | source | **no** | `climate_historical/<key>/plots/source_*.png` |
| Model-parity climate (rule 1.11) | what climate did the model see, per station/period? | model | yes | `hydrology_model/evaluation/plots/clim_wflow_1_*.png` |
| Forcing / model-input QA (rule 1.13) | did the downscaling to the model grid behave? | model | yes | `hydrology_model/forcing/plots/{precip,temp,pet}.png` |

The `source_` prefix is deliberate (design risk-9): a bare `pet.png` copied out
of its directory loses its parent, and the two `pet.png` values genuinely
differ. The source family exists so the **P4 assertion** holds — those three
PNGs build with neither `hydrology_model/` nor
`config/templates/wflow_build_model.yml` on disk (pinned by
`tests/test_plot_climate_source.py`). P1 was applied per-artifact; **no one
checked the resulting count of `plots/` dirs per product area.** That is the
gap this entry names.

**Assessment of the two supporting claims.**

- *(b) is correct, and stronger than stated.* `clim_wflow_1_*` never reads the
  wflow run. `plot_results.py:198-227` takes the **store extraction**
  (`climate_nc`) through `climate_parity.model_parity_climate` against the model
  DEM and masks it with `staticmaps["subcatchment"]`. It is a climate figure on
  the model grid, not an evaluation product; it sits under `evaluation/plots/`
  only because it shares a module with `hydro_wflow_1.png`, which *is* an
  evaluation product (sim-vs-obs discharge from `run_default/output.csv`).
- *(a) is true but costs a real product.* The `forcing/plots/` maps read
  `inmaps_historical.nc` — what hydromt **actually wrote** — masked to the
  subcatchment. Regenerating them from source data (or from the parity
  transform) replaces the observed model input with *our reimplementation* of
  the build's regrid/PET chain. That deletes the only figure that would reveal a
  bad hydromt forcing step, which is the highest-consequence silent failure in a
  no-calibration pipeline. Placement can be fixed without touching the input.

**Supporting evidence the proposal is right: wf1 is the outlier.** Every other
product area in the tree already has exactly one `plots/`:
`climate_projections/<clim_project>/plots/`,
`experiments/<exp>/weather_generator/plots/` — and `hydrology_model/plots/`
(holding `basin_area.png`). Only wf1's model area has three. The owner's
proposal is not a new convention; it is **making wf1 conform to the grammar the
rest of the tree already follows**: *one `plots/` per product area.*

**Alternatives.**

1. **Keep as-is, document better** (the pre-discussion recommendation). Add a
   user-facing output-tree doc plus the three-family table. Cheapest, zero
   baseline churn — but leaves wf1 inconsistent with the rest of the tree and
   does not address the usability complaint, only explains it. *Standard
   practice.*
2. **Two buckets, pure path + filename move** (owner's proposal, recommended).
   Producers and inputs untouched; only output paths and figure filenames
   change. Preserves all three products and the P4 assertion; wf1 gains the
   one-`plots/`-per-area grammar. Costs a baseline re-record and a docs pass.
   *Standard practice.*
3. **Two buckets + collapse producers** (owner's claim (a) taken literally).
   Also re-source the model-grid maps from the store/parity transform and drop
   the inmaps-based ones, removing a rule and an input edge. Genuinely simpler
   DAG — the model-grid climate figures would then need the built model but not
   the run, mirroring P4 one level up — but it deletes the downscaling QA
   product. *Speculative; not recommended as part of this change.* Separable,
   and should be ruled on its own merits after (2) lands.

**Recommendation.** Alternative 2, split into the core move (what was asked) and
two optional additions (proposed here, not requested — rule on each separately).

*Core — two buckets, no renames.* Move `forcing/plots/{precip,temp,pet}.png` and
`evaluation/plots/{clim_wflow_1_month,clim_wflow_1_year,hydro_wflow_1}.png`
(plus the undeclared `signatures_*`, `basavg_*`) into `hydrology_model/plots/`.
**No filename collisions:** `source_*` stays in the other bucket, and
`clim_wflow_1_*` / `hydro_wflow_1` / `basin_area` are already distinct. So the
core move needs no renames at all.

```text
climate_historical/<key>/plots/   source_precip.png  source_temp.png  source_pet.png
hydrology_model/plots/            basin_area.png
                                  precip.png  temp.png  pet.png
                                  clim_wflow_1_month.png  clim_wflow_1_year.png
                                  hydro_wflow_1.png
                                  signatures_<station>.png  basavg_<var>.png  (undeclared)
hydrology_model/evaluation/       performance_metrics.csv
```

*Optional A — the `forcing_` prefix.* Rename the three model-grid maps to
`forcing_{precip,temp,pet}.png`. R07 disambiguated only one of three families;
`source_precip.png` and `clim_wflow_1_month.png` are self-describing, bare
`precip.png` is not, and once the families share a directory the risk-9 argument
(a figure copied out of its directory loses its parent) applies to it with full
force. Costs a rename on top of the move.

*Optional B — dissolve `evaluation/`.* With the figures gone it holds one CSV;
move `performance_metrics.csv` to the model root. **This reverses an explicit
R07 P1 decision** (`plots/` holds figures only, the table sits one level up in
`evaluation/`) that was not part of the challenge — hence optional.

*Verified edit surface* (grep over `tests/ dev/ docs/ scripts/ blueearth_cst/`,
2026-07-29 — larger than the rule declarations):

| File | What pins the paths |
|---|---|
| `Snakefile_model_creation` | rules 1.11, 1.13 outputs; rule `all` and 1.14's gather inputs |
| `blueearth_cst/model/plot_results.py:116` | `Folder_plots = f"{Folder_eval}/plots"` — the script decides the directory, not the rule |
| `blueearth_cst/model/plot_map_forcing.py:193` | `Folder_plots` module global — same |
| `tests/test_wf1_plot_outputs.py:35-42,78` | hard-codes all 8 paths incl. the undeclared `signatures_wflow_1.png` |
| `dev/scripts/check_baseline.py:162-164` | comparator table entries |
| `dev/scripts/semantic_tree_diff.py:260-273` | the pre-R7 → post-R7 rename map; a second move makes it a 2-hop map unless updated |
| `dev/baseline/manifest.json` | 3 hits + a full re-record |
| `plot_climate_source.py:14-22`, `plot_results.py:113-118,358` | the three-family docstring table and P1 comments |

Validation: `pytest tests/test_cli.py`, then `tests/test_wf1_plot_outputs.py`, a
`check_baseline` run, and a baseline re-record.
`tests/test_plot_climate_source.py` and the P4 assertion are unaffected — bucket
1 does not move.

**Open question blocking sign-off — asymmetric across the three parts.** CST is
the engine of a three-part platform. If the CST-API backend or the frontend GUI
collects figures **by glob** (`hydrology_model/**/plots/*.png`), the core move is
harmless and only Optional A breaks it. If it reads **exact paths**, the core
move breaks it too. Optional A is blocked hard either way; the core move is
blocked only under exact-path collection. This cannot be checked from this repo.

**Disposition.** Pending owner ruling on Core / Optional A / Optional B.

**Disposition.** Pending owner ruling.

<!--
Template — copy per entry:

### S7-nn — <one-line challenge>

- **Area:** <config | layout | workflow | tests | docs | tooling | method>
- **Status:** open
- **Created / Updated:** 2026-07-29 / 2026-07-29
- **Rev:** `<short sha the challenge was raised against>`

**Challenge.** What now looks wrong, and how it shows up. Include the exact
command / configfile / path if it is reproducible.

**What we decided, and why.** The original choice and its stated rationale,
sourced to a design doc, commit, or ADR — so it is clear whether the decision
was deliberate or incidental.

**Alternatives.** Two or three options with real tradeoffs. Mark each as
standard practice, emerging practice, or speculation.

**Recommendation.** The proposed way forward, including "keep as-is" when that
is the answer.

**Disposition.** Where it landed: sha, followups ID, TODO row, ADR number, or
`no-change`.
-->
