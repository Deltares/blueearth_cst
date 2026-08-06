# Migration — Snakemake rule identifiers (R10)

Required by `dev/reference/naming.md` §7: a rule identifier is the CLI target
surface (`snakemake <rule> -s …`, `--forcerun <rule>`), so renaming one breaks a
command someone has in their shell history.

Date: 2026-08-06 · Milestone: R10 step 6 · Design:
`dev/milestones/r10/rule-naming-design.md` · Published map:
`dev/reference/workflows/rule-index.md` § *What changed*

**This record covers the twelve NAME changes only.** Every rule also changed
NUMBER in the same sweep (`dev/followups.md` `[R10-5]`); the number map is in
`rule-index.md` and is not repeated here, because a number is a reading aid and
a name is a CLI contract.

## The twelve

| old | new | workflow(s) | why |
|---|---|---|---|
| `add_gauges_and_outputs` | `declare_wflow_outputs` | WF1 | **adds no gauges** — `build_wflow_model` does, via `setup_gauges`/`setup_outlets` with `toml_output=None`. This rule only writes the `[output.csv]` block and re-checks the gauge IDs. The job moved in the P1/P2 restructuring and the name did not follow |
| `add_forcing` | `add_climate_forcing` | WF1 | thin noun beside its siblings — did not say *which* forcing |
| `extract_climate_grid` | `extract_historical_climate` | WF1 + WF3 | "grid" collided with the stress-test grid. **The one cross-workflow rename**: this rule is splatted into both files from one producer contract |
| `plot_results` | `plot_wflow_evaluation` | WF1 | vague noun beside the specific `plot_climate_source` / `plot_forcing` |
| `plot_map` | `plot_basin_map` | WF1 | vague noun |
| `fetch_gcm_raw` | `fetch_gcm_slice` | WF2 | dangling adjective; "slice" is the code's own word for the artifact |
| `plot_climate_proj_timeseries` | `plot_gcm_timeseries` | WF2 | `proj` is an ad-hoc contraction; `gcm` puts all three WF2 rules on one noun |
| `climate_stress_parameters` | `prepare_stress_test_grid` | WF3 | no verb; matches the `stress_test:` config section it reads |
| `prepare_weagen_config` | `prepare_weathergen_config` | WF3 | `weagen` appears in no path or directory |
| `generate_weather_realization` | `generate_weather_realizations` | WF3 | one invocation produces ALL `RLZ_NUM` of them and the output is a list; singular was simply wrong |
| `generate_climate_stress_test` | `perturb_climate_realization` | WF3 | **claimed another rule's job.** `prepare_stress_test_grid` creates the stress test; this applies ONE point of it to ONE realization |
| `climate_data_catalog` | `write_climate_data_catalog` | WF3 | no verb |

## Three that left scope, so the count reconciles against the original ten

| rule | disposition |
|---|---|
| `export_wflow_results` → `derive_wflow_indicators` | **landed with R9**, on the principle that a milestone renames what it falsifies — R9 renamed the rule's outputs |
| `prepare_weagen_config_st` | **the rule no longer exists** — deleted by C29, which found its per-member config carried nothing that varied except its own output filename |
| `setup_runtime` → ~~`prepare_runtime_window`~~ | **withdrawn** — `[R10-1]` merged the rule into the forcing rule instead. A recipe that never leaves the pair needs no name of its own, so the naming drift disappears with the rule rather than being renamed around |

## What a caller has to change

**Only an explicit rule target.** No output path, no output value, and no config
key moves.

```bash
# before                                    # after
snakemake plot_results -s Snakefile_model_creation
snakemake plot_wflow_evaluation -s Snakefile_model_creation

snakemake --forcerun extract_climate_grid   snakemake --forcerun extract_historical_climate
```

The `--forcerun extract_climate_grid` form is the one most likely to be in
somebody's history: it is the documented escape hatch for an in-place data
mutation behind an unchanged catalog entry (`[R7-2]`,
`dev/milestones/r07/migration_project-layout.md` §2f).

## Six call sites per rule, and the one a checklist skips

| # | site | example |
|---|---|---|
| 1 | `LOG_RULES` entry | `"1.08_add_climate_forcing"` |
| 2 | `W.NN` comment header | `# 1.08  add_climate_forcing — …` |
| 3 | the `rule` identifier | `rule add_climate_forcing:` |
| 4 | **`rule_banner`'s second argument** | `rule_banner("1.08", "add_climate_forcing")` |
| 5 | `log:` path | `{LOG_PARTS_DIR}/1.08_add_climate_forcing.log` |
| 6 | `benchmark:` path | `benchmarks/_parts/1.08_add_climate_forcing.tsv` |

**Site 4 is the one to miss**, and missing it is cosmetic rather than breaking —
the banner announces the old name while the rule carries the new one — which is
exactly why it would survive a green test run and reach a user.

**Site 1 is the one that fails silently.** `merge_logs` discovers a rule's parts
by listing the directory named after its `LOG_RULES` label and is deliberately
scoped to that list, so a missed rename produces a log section that vanishes
from the merged log while its parts accumulate on disk forever. Neither
direction raises. `tests/test_log_rules_contract.py` now asserts set-equality
between the list and the labels derived from every rule's `log:` path, in both
directions, for all three workflows.

**One rule is exempt.** The batched Wflow rule's identifiers are
`run_wflow_batch_<b>` — parse-time loop-generated, one per batch — while its log
label and `LOG_RULES` entry are the singular `<W.NN>_run_wflow`. That divergence
is deliberate (P3-3 keys logs by batch id, not by rule identifier). Applying the
six-call-site rule mechanically there would rename a `LOG_RULES` entry that has
no rule to match, and break the merge. None of the twelve touches it; stated so
a future sweep does not "fix" it.

## Script modules did NOT move

The scope is the rule **identifier**. `blueearth_cst/model/plot_results.py`,
`plot_map.py`, `fetch_gcm_raw.py`, `prepare_weagen_config.py` and
`prepare_climate_data_catalog.py` keep their names, so five of the twelve rules
no longer share a name with the module they execute. That is a deliberate
narrowing, not an oversight: renaming a module is a separate import-surface
change with its own blast radius.

It is also the trap the sweep had to be written around — a blanket text
substitution of `plot_map` would have hit `plot_map_forcing.py`, and one of
`prepare_weagen_config` would have hit its own `script:` line. Each rename was
applied to the six call sites above by exact pattern, with the match count
asserted per site.

## Gates

- `pytest tests/test_cli.py` — all three Snakefiles parse and dry-run.
- `pytest tests/` — the full suite; nine tests hardcoded old rule identifiers
  and were updated with the sweep.
- `check_baseline.py check` — **must pass with NO diff, and every target is in
  scope.** The manifest is current: `[R10-6]` §11 and §12's re-record landed at
  `ea5ac59`, which is in this branch's history, and it moved exactly four
  targets (the three config snapshots, sharing a hash because all three copy the
  seed config, plus `q_indicators.csv`). Nothing is owed. **Any** target moving
  here is this sweep's damage and is a rollback trigger.

  This was briefly written the other way, and the correction is the point worth
  keeping: `dev/followups.md` still said "re-record owed" under §11 and §12
  after the re-record had landed, so reading the item rather than the manifest
  produced a gate that would have accepted six diffs as expected. **Check what
  the manifest records, not what a followup says is owed** — a followup is a
  statement about the past, and `ea5ac59` is the fact.

  **Why a rename cannot reach any target.** Rule labels reach exactly two
  durable *contents* — the merged log's section banners and the benchmark
  table's rule column — and the manifest fingerprints neither. It has no
  `logs/`- or `benchmarks/`-shaped target at all: its fourteen are three config
  snapshots, two CMIP6 summary tables, two indicator tables, the run CSV, and
  **six** figures — which are excluded by default under `FIGURE_KINDS`, so the
  gate compares **eight**.

  **RESULT 2026-08-06: `OK - 8 target(s) match manifest.`** No target moved.

  **What that does and does not prove, because the tool says so itself.** It
  warned: *"manifest recorded by `main@a1d9993` +dirty, checking from
  `HEAD@6ad6a2e` +dirty. The baseline fixture is untracked and therefore SHARED
  BY EVERY BRANCH, so a pass here may mean the tree matches another branch's
  code."* That is exactly the situation. `test_case/test_local` still holds the
  artifacts a pre-sweep run produced; no workflow has been re-run under the
  renamed rules. So this pass establishes that **the sweep did not perturb the
  recorded artifacts** — real, since a careless edit to a `script:` path or an
  `output:` block would show here — but it is **not** evidence that a fresh run
  under the new identifiers reproduces them.

  The structural argument is what carries that claim, and it is checkable
  without a run: no manifest target is log- or benchmark-shaped. The empirical
  confirmation is a three-workflow run followed by a re-check, which is owed
  regardless for the `[R10-1]` merge's subprocess plumbing.

- `pixi run tree-check` — **`MAP CLEAN: 186 paths, 0 moved, 186 identity, 0
  unmapped`** (2026-08-06, primary checkout). No rule identifier reaches a
  durable path, which this confirms from the other direction: the tree the
  pre-sweep run produced is still fully declared under the post-migration
  inventory.

## Old part directories will persist, and that is correct

The fixture tree currently holds **no** `logs/_parts/` at all — the last run
completed, and both merges delete the parts they consume. But after the next
*partial* run under the new identifiers, `logs/_parts/` will hold directories
named for the old labels — `2.01_fetch_gcm_raw/`,
`3.07_generate_climate_stress_test/`, `3.10_run_wflow/` and the rest —
alongside the new ones. `merge_logs` is deliberately scoped to `LOG_RULES`, so
it never reads an unlisted directory and never deletes one; that scoping is what
keeps an orphan out of the merged log in the first place.

`tree-check` will not flag them: the post-migration inventory covers
`logs/_parts/` as a prefix. So the "a stranded part shows up as an unmapped
path" check does **not** fire for this class, and a clean `tree-check` is not
evidence that no part was stranded. Stated because it looks like a defect and
is not — delete them by hand, or leave them; they are transient either way.

## Do not rewrite the archives

`dev/milestones/`, `DEVLOG.md`, `dev/decisions/` and the dated migration records
under `docs/` name these rules as they were on their date. A hit there is
evidence, not drift. The live surfaces — the three Snakefiles, `AGENTS.md`,
`docs/` guides, `dev/reference/`, `blueearth_cst/`, `dev/scripts/`, `tests/` —
carry the new names, with this record and `rule-index.md` § *What changed* the
two places that name the old ones on purpose.
