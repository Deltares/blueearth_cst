---
title: Reconcile test_rapid's WF1 snapshot with the rapid config so WF3 can run on the default config
type: todo-item
status: backlog
effort: 1
area: test fixtures
queue: 2
created: 2026-08-11
updated: 2026-08-11
---

> [!note] Overview
> **What** — The WF1 project snapshot in test_case/test_rapid was written by a different config than the tracked test_case/snake_config_rapid.yml, so rule 3.01 check_project_consistency refuses every WF3 run against it. Nine fields diverge, including observation paths pointing at an absolute C:/TESTS/CST/observations/gabon location, a simulation end date of 2016 against the snapshot's 2020, two CMIP6 models against three, and one emissions scenario against two. Decide whether to re-run WF1 to regenerate the snapshot or to re-point the config at what the tree already holds.
> **Why** — WF3 cannot be run at all on the config AGENTS.md names as the default, so the cheap tree meant for watching rules execute is the one tree that cannot execute them. Anyone reaching for the documented default hits a nine-line consistency refusal that reads as a code defect and is not one.
> **Effort** — Small if the answer is to re-point the config at what the tree already holds; a 16-job WF1 rebuild if the snapshot has to be regenerated instead. The open question is which side is actually right, and whether the data the config names is still reachable from this machine — the snapshot's observation paths are absolute and point outside the repo.

## Progress

- [x] Establish which side is authoritative — was `test_rapid` built by an earlier revision of `snake_config_rapid.yml`, or seeded by copying a gabon tree that never matched it?
- [x] Confirm the `deltares_data` catalog and the observation CSVs the config names are reachable, since a rebuild needs both
- [x] Decide: re-run WF1 into `test_rapid`, or re-point the config at the tree's actual provenance
- [x] Apply the decision, and confirm a WF3 `--dry-run` gets past rule 3.01
- [x] Run WF3 end to end and confirm all three indicator tables land populated
- [x] Re-check that AGENTS.md's "which config to run" table still tells the truth

## Outcome (2026-08-11)

**Neither side was "an earlier revision of the rapid config".** All 207 files
dated from 2026-08-11 14:45, so `test_rapid` was a real WF1 build — but from an
ad-hoc config: an older revision of the seed (its header still says
`examples/test_local`, the pre-R9 path) with `project_dir` swapped and
observations pointed at `C:/TESTS/CST/observations/gabon/`. Only one invocation
record exists and it is the 14:25 **dry** run; the real build bypassed
`run_workflows.py`, which is why the provenance had to be reconstructed by hand.

**The tracked config wins, and it was never really a choice.** The invariant is
visible in the healthy tree: `test_local`'s WF1 snapshot is byte-identical to
its tracked `snake_config_model_test.yml`. `test_rapid` violated it. And
"re-point the config at the tree" fails on its own terms — no tracked config can
name `C:/TESTS/CST/...` and stay reproducible for anyone else, so a WF1 re-run
was forced either way. The only open question was which settings it used.

Rebuilt all three workflows through `scripts/run_workflows.py` (so this time
there IS an invocation record). Both snapshots now match the tracked config
byte-for-byte, and WF3 completes 34/34 with `q`/`aet`/`gwr` indicator tables.

**A second defect, found while costing the rebuild and NOT in the original
diagnosis.** WF2's snapshot regenerated but its *products* did not: after the
full wrapper run the summary still carried 3 models, the `far` horizon and a
1990-2010 reference window, against the config's 2 models / `mid` / 2000-2014.
`raw/` and `scalar/` are keyed by (model, scenario, member) and are genuinely
window-independent, so only stage B (2.06) had to re-run — and it did not,
because this `.snakemake` has no metadata for the 14:45 build (`12 jobs have
missing provenance/metadata`), so the params trigger cannot fire and mtime says
`summary/` is newer than `scalar/`.

Rule 3.01 cannot catch this: it compares **config sections**, not products. So
the tree passed the guard while its CMIP6 overlay described a different
experiment than the config asked for. Fixed with
`--forcerun derive_change_factors plot_gcm_timeseries` (network-free — the raw
slices stay valid), then `prune_series_cache.py --delete` for the 5 series the
model/scenario narrowing orphaned. The rebuilt tree also exposed two tree-inventory
map gaps, boarded separately as
[[t2608112047-cover-the-two-declared-artifacts-the-post-r9-tree-inventory-misses]].

`experiments/experiment_rapid_v2/` was dropped on the owner's ruling: its
purpose (the `gwr` rename verification) is spent and committed, and it pinned a
model digest the rebuild destroyed.

## Refs

Re-confirmed 2026-08-11 19:48 on `fix/improvements`, after the WF3 run-record
move: `snakemake all -s Snakefile_climate_experiment --configfile
test_case/snake_config_rapid.yml` still dies at rule 3.01 on the same nine
fields, 2 of 34 steps in. Nothing has drifted and nothing new is blocked — the
end-to-end check that run was for was taken on `snake_config_model_test.yml`
instead, which passes 3.01 and completed 39/39.

Surfaced 2026-08-11 while verifying the `recharge` → `gwr` indicator-token rename
on `fix/improvements`. The rename itself was verified another way (the reducer
driven directly over the eight real member CSVs in
`test_rapid/experiments/experiment_rapid_v2`), so nothing is blocked on this —
but the run that was *supposed* to verify it could not start.

The refusal, from
`experiments/experiment_rapid/logs/_parts/3.01_check_project_consistency.log`
(that partial experiment directory has since been removed). Experiment config on
the left, WF1 snapshot on the right:

| field | config | snapshot |
| --- | --- | --- |
| `project.project_dir` | `test_case/test_rapid` | `<repo>/test_case/test_rapid` |
| `shared.basin.gauge_points` | `test_case/test_data/output_locations.csv` | `C:/TESTS/CST/observations/gabon/output_locations.csv` |
| `…model_creation.observations_timeseries` | `test_case/test_data/observations_timeseries.csv` | `C:/TESTS/CST/observations/gabon/observations_timeseries_workflow.csv` |
| `…simulation_window.endtime` | `2016-12-31` | `2020-12-31` |
| `…climate_projections.historical_year_range` | `[2000, 2014]` | `[1990, 2010]` |
| `…climate_projections.models` | 2 models | 3 models |
| `…climate_projections.scenarios` | `[ssp245]` | `[ssp245, ssp585]` |
| `…future_horizons` | `mid` only | `far` only |

The absolute `C:/TESTS/CST/observations/gabon/…` paths are the strongest clue:
they are not a path any tracked config in this repo uses, which points at the
tree having been built by a private config rather than by the one now shipped.

`experiments/experiment_rapid_v2` is a complete, consistent experiment under the
*old* provenance and still carries its own `.project_consistency_ok`, so
whatever changed, changed after 2026-08-11 14:52.

Queue rank 2 is a suggestion, not a ruling — it ties with `t2608071203`. It sits
high because it blocks the config `AGENTS.md` designates as the default, not
because it is urgent in itself.
