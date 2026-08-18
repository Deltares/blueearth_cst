---
title: Give WF0 its forcing-selection evaluation layer — rules 0.07-0.09
type: todo-item
status: backlog
effort: 2
area: wf0 / evaluation layer
origin: t2608131847a split (2026-08-18)
queue: 1
created: 2026-08-18
updated: 2026-08-18
---

> [!note] Overview
> **What** — Add the rules that let WF0 *evaluate* candidate forcing datasets
> rather than only characterise them: station/subregion sampling with
> observation comparison, and Budyko screening. The rule numbering already
> reserves the space — WF0 runs 0.01–0.06 and then jumps to 0.10, so **0.07,
> 0.08 and 0.09 are an empty gap**, which is the clearest statement of what is
> missing.
> **Why** — WF0 exists to answer *which forcing dataset should this basin use*,
> and that question matters here precisely because CST does no local
> calibration, so forcing choice is the dominant lever on the historical run.
> Today WF0 extracts, plots and compares the candidates against **each other**;
> nothing compares them against **observations**. A user can see that CHIRPS and
> ERA5 disagree, and still cannot see which is closer to the truth.
> **Effort** — Large. Two new config keys with templates, a breaking config
> move already ruled, and three rules.

## Progress

- [ ] **Station/subregion sampling + observation comparison** (rules 0.07–0.08).
      Needs two new WF0-owned config keys, `climate_locations` and
      `climate_locations_timeseries`, plus their csv templates under
      `config/templates/`.
- [ ] **Budyko screening** (rule 0.09), carrying the ruling below.
- [ ] The fourth notebook under `docs/notebooks/`, inherited from the closed
      `t2608131847` — there are three today. Note
      [[t2608132140-make-the-notebooks-run-helper-fail-loudly-on-a-nonzero-exit]]:
      the new notebook will copy the `run()` helper, so fixing that wart first
      stops it propagating.
- [ ] Follow-ons, only if wanted after the above lands: SPI / dry-day / heat-day
      indices; MODIS snow cover.

## The one ruling this inherits

Carried verbatim from `t2608131847a`, because it is an owner decision and
re-deciding it would be waste:

> **O-2 RULED 2026-08-15: option A** — `observations_timeseries` moves from
> `workflows.build_model` to `shared:`, beside `shared.basin.gauge_points`.
> Rejected: having WF0 read WF1's section (breaks the `CONFIG_PROJECTION` model,
> so a WF1-only edit would re-fire WF0's run record) and duplicating the key
> (the `output_locations` / `gauge_points` scar). Fold the move into the rename
> migration note — one breaking config change, not two.

That last clause has aged and should be re-read before acting: the rename
migration landed on 2026-08-14 (`docs/migration-workflow-names.md`), so there
is no longer an open migration note to fold into. The intent survives — make
this a single breaking config change, documented in one place — but it now
needs its own migration entry rather than a ride-along.

## Why this is a separate item

`t2608131847a` bundled a MIGRATION that is finished with a FEATURE that is
unstarted, which is why it read as stale while sitting at `active`. The split
half shipped in full on 2026-08-14: `analyze_climate.smk` exists and runs, the
rename landed, `run_workflows.py` orders WF0 first, and `AGENTS.md` documents
it. None of the evaluation layer exists — verified 2026-08-18, and not only from
the checklist: rules 0.07–0.09 are absent from the Snakefile, and nothing under
`blueearth_cst/climate_analysis/` mentions Budyko or subregion sampling.

**Explicitly NOT in scope: multi-forcing model runs.** Deferred out by the
2026-08-14 ruling, and still deferred — it belongs to WF1, not here. If it is
wanted, it takes its own note.

## Refs

- `dev/working/2026-08-14_climate-workflow-split/design.md` — the design the
  split was executed from; its later sections describe this layer.
- `dev/reviews/2026-08-13_fao-branch-assessment.md` §2 — where the item
  originated.
- `dev/LOG.md` — `t2608131847a`'s closure row, which points here.
