---
title: Verify the newly-honoured water year, and sweep for projects whose results it moves
type: todo-item
area: wf2 projections / config
origin: shared.water_year_start promotion (2026-08-12)
created: 2026-08-12
updated: 2026-08-12
---

> [!note] Overview
> **What** — `shared.water_year_start` now reaches WF2's change-factor arithmetic, which it never did before. Two loose ends: the non-Jan path has never been run against real data, and no one has checked which existing projects it changes.
> **Why** — The fix is landed and gated only at the Jan default, where it is provably a no-op. Everything the change actually *enables* is untested, and any project that already set a non-Jan value has recorded results that silently used Jan.
> **Effort** — S for the verification run; unknown for the sweep, since production project roots live outside this repo.

## Background — what was fixed, so this is not re-diagnosed

`workflows.climate_projections.start_month_hyd_year` was **inert**. The
Snakefile read it and passed it to rule 2.06; `derive_change_factors.py` never
read the param. Every change factor ever produced was computed Jan–Dec,
whatever the config said. The module's own comment recorded the deferral
("forwarding the key would change results for any non-Jan config and belongs
in its own commit with its own gate").

Fixed 2026-08-12 in `086ba7b`: the value now reaches both
`hydrological_year_bounds` calls and `get_change_annual_clim_proj`, the key
moved to `shared.water_year_start`, and the legacy key is **refused** at parse
time rather than quietly starting to work — because honouring it silently
would change a project's numbers as a side effect of an upgrade.

**This item is not "fix the inert key".** That is done. It is the two things
the fix left open.

## The sweep half is DESCOPED — owner ruling 2026-08-17

> **Ruling (owner).** Do not run the sweep; go straight to the verification run.
> **No project inventory was taken** — this is a scope decision, not a recorded
> finding that every project uses `Jan`. Read it as "not worth enumerating",
> which is why the parse-time refusal below is what makes it safe.

The per-project re-run-versus-pin decision the sweep would have fed therefore
does not arise now. It would arise again if a non-Jan project surfaces.

So this item reduces to its **verification half**, which the descoping does not
touch: the non-Jan path has still never been run, and the gate that passed proves
nothing about the path this work exists to enable, because at `Jan` the change is
provably a no-op.

**Why descoping is safe without the inventory.** A project that sets the legacy
key is **refused at parse time**, with the replacement block in the message
(`analyze_projections.smk`). So an affected project cannot silently produce
Jan-computed change factors while claiming a water year — it stops, loudly, and
the sweep's question gets asked then, about a project that is in front of you.

## Progress

- [x] ~~Sweep for existing projects carrying a non-Jan value~~ — **descoped**,
      owner ruling 2026-08-17. No inventory was taken; the parse-time refusal is
      what carries the risk instead.
- [x] ~~Decide, per affected project, whether to re-run WF2 or pin
      `water_year_start: Jan`~~ — descoped with the sweep. Re-opens per project
      if one ever hits the parse-time refusal.
- [ ] **Run WF2 with a non-Jan water year against real data.** Everything so
      far is DAG-parse, unit tests, and value-neutrality at `Jan` — where the
      change is provably a no-op, so the gate that passed proves nothing about
      the path this work exists to enable. Use the fixture, set
      `shared.water_year_start: Oct`, and confirm the change factors move, that
      `n_hyd_years_reference` is the Oct→Sep count, and that
      `reference_window_effective` reports the window actually used.
- [ ] **Check the Oct boundary against `hydrological_year_bounds`' known
      off-by-one.** `dev/milestones/r08/2026-07-30_wf2-5f-hydyear-offbyone.md`
      records that the complete-year count was wrong for exactly the
      October-start case until 2026-07-30. That fix has never been exercised
      end-to-end with a non-Jan config, because no non-Jan config ever reached
      the arithmetic. This is the first run that would.
The two sweep steps that stood here are closed above. Their original text, kept
because it states the hazard that would return if a project ever adopts a
non-Jan value: *"They now hard-error until the key is moved, and once moved
their change factors change — correctly, but every recorded result for that
project becomes non-comparable."* That remains true; it simply has no subject
today.

## Refs

- `086ba7b` — the fix. `5a881d0` — the same key reaching the WF3 indicators and
  WF1 figures.
- `blueearth_cst/projections/derive_change_factors.py` — the two
  `hydrological_year_bounds` calls and the `get_change_annual_clim_proj` call.
- `Snakefile_climate_projections` — the refusal, with the replacement block in
  its message.
- `dev/milestones/r08/2026-07-30_wf2-5f-hydyear-offbyone.md` — the
  October-start off-by-one this run would exercise for the first time.
- Related: [[t2608121742-run-weather-generator-does-not-forward-relax-priority]]
  — the other parameter this session found reaching nothing.
