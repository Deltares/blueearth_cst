# Snakemake rule naming — R10 design

Status: **ACCEPTED** by the owner, 2026-08-04. **AMENDED THREE TIMES, 2026-08-06.**
Not implemented.

## Amendment 3 — 2026-08-06, `[R10-2]` dropped and `evaluate_` withdrawn

Owner ruled `[R10-2]` **DROPPED** (option 3 of the three the item records). The
split of rule 1.11 into a metrics rule and a figure rule does not happen.

**Consequence for this design: `evaluate_` is withdrawn as the 19th verb.** It
was ruled in *for* `evaluate_wflow_run`, the metrics half of that split, and no
other rule was ever going to carry it — grep confirms zero occurrences across the
three Snakefiles. A verb in the table with no user is a trap for the next
author: it reads as an available option that some rule must already justify.

Nothing else moves. 1.11's own rename to `plot_wflow_evaluation` was always the
FIGURE half's name, so it survives the drop unchanged and step 6 renames 1.11 to
it exactly as this design already says.

## Amendment 2 — 2026-08-06, from the rule-index name-vs-body audit

Ruled by the owner after `dev/reference/workflows/rule-index.md` checked all 34
identifiers against their **script or shell bodies**, rather than against the
naming grammar. Amendment 1 below re-audited the *inventory*; this one re-audits
the *claims*.

**The method gap this exposed.** Amendment 1's "Conforming (22), do not touch"
list is correct on grammar — every one of the 22 satisfies `verb_object`. It was
never a check that the verb is still **true**. Two rules passed it while
describing work that had moved out from under them. Grammar conformance and body
conformance are two different checks, and only the first had been run.

| # | audit found | ruled |
| --- | --- | --- |
| 1.05 | `add_gauges_and_outputs` **adds no gauges** — rule 1.03 does, via `setup_gauges`/`setup_outlets` with `toml_output=None`. 1.05 only writes the `[output.csv]` block and re-checks the gauge IDs. The "add gauges" half moved in the P1/P2 restructuring and the name did not follow | **rename → `declare_wflow_outputs`**, and add `declare_` to the verb table |
| 1.07 | `setup_runtime` writes a hydromt **forcing build recipe**, not a runtime. Nothing reaches the model TOML until 1.08 applies it. The designed target `prepare_runtime_window` is no more accurate than the name it replaces | **drop the rename — merge 1.07 into 1.08** (see *Out of scope* below) |
| 1.02 | `prepare_spatial_maps` names one of nine outputs; the rule is the delineation-and-identity engine | **keep the name.** `build_spatial_foundation` reads less clearly to the owner than the slightly narrow name it would replace, and `build_` is defined here as "construct a model from inputs" — the spatial foundation is not a model |

**Why `declare_` earns an 18th verb.** `add_` was available at no vocabulary
cost: its entry reads "mutate an existing model in place (a hydromt `update`)",
which is mechanically what 1.05 does. It was rejected because the action class
differs — 1.04 and 1.08 add model **data** (waterbody layers, forcing grids),
while 1.05 adds none and only changes what the engine will emit. Amendment 1 set
this precedent by adding `delineate_` rather than forcing a worse name onto
`derive_` or `extract_`.

**Counts reconcile at 34, unchanged:**

| class | count | change |
| --- | --- | --- |
| renames | 12 | 1.07 leaves, 1.05 joins — net zero |
| conforming, do not touch | 21 | 1.05 leaves for the rename list |
| pending removal (merges into 1.08) | 1 | 1.07 |

**Out of scope for R10: two structural changes.** This milestone is
identifier-only — "Rule bodies, inputs, outputs, numbering, and the artifacts
they produce are out of scope". The same audit accepted two body changes, both
recorded in `dev/followups.md`:

| | change | effect on this design |
| --- | --- | --- |
| `[R10-1]` | merge 1.07 into 1.08 | 1.07's rename is withdrawn — see above. Cost: 1.07 is a Python `script:`, 1.08 a `shell:` hydromt CLI call, and Snakemake allows one per rule |
| `[R10-2]` | ~~split 1.11 into `evaluate_wflow_run` + `plot_wflow_evaluation`~~ **DROPPED 2026-08-06** | no effect. 1.11's rename to `plot_wflow_evaluation` was always the FIGURE half's name and stands unchanged. `evaluate_` is withdrawn with the split — see Amendment 3 |

**Sequencing:** either order works for both. If a merge or split lands first, the
affected rename is already moot; if R10 lands first, the rules keep their names
until the structural change removes or adds one. What must not happen is renaming
1.07 in passing, or inventing the metrics rule's name during the sweep instead of
ruling on it.

Two further merges were **rejected** by the same audit — 1.06 into 1.05, and the
paired `gather_*` rules — for structural reasons recorded in `[R10-3]`. Noted
here because both would have changed which identifiers exist.

**Riding in the same sweep, but not a rule rename:** `[R10-7]` renames the three
shared-rule helper functions in `snake_utils.py` from `_spec` to `_rule`
(`region_spec` → `region_rule`, `climate_store_spec` → `climate_store_rule`, and
the new `spatial_units_rule` from `[R10-6]`). These are **Python identifiers, not
Snakemake rule identifiers**, so they are outside this design's scope proper —
they ride along only because the sweep already edits all three Snakefiles.
Counted separately: the 34 identifiers above are unaffected. `naming.md` should
gain the `_rule` convention in the same edit that gains the verb vocabulary.

## Amendment 1 — 2026-08-06

Ruled by the owner after a full re-audit against the code as it stands. The
design was accepted before R9's followups landed, and two of its rules moved
underneath it.

**Scope is now TWELVE renames, not ten** — the original ten, less two that left
scope, plus four newly ruled in. The arithmetic is stated because the original
total is close enough to look untouched.

Two dropped, for opposite reasons:

| # | rule | why it is no longer in scope |
| --- | --- | --- |
| R9 | `export_wflow_results` → `derive_wflow_indicators` | **already landed** with R9, as this design intended |
| 3.05 | `prepare_weagen_config_st` | **the rule no longer exists** — deleted by C29 (`dev/milestones/r09/wf3-change-requests.md` CR-5), which found its per-member config carried nothing that varied except its own output filename. Renaming it would have entered `migration_rule-names.md` as a CLI-surface rename for a rule that then vanished. Same principle this design applies in reverse when it folds the R9 rename out. |

**Three renames adjusted:**

| # | as designed | as ruled | reason |
| --- | --- | --- | --- |
| 3.03 | `prepare_stress_grid` | **`prepare_stress_test_grid`** | matches the config section it reads (`stress_test:`) |
| 3.08 | `write_climate_catalog` | **`write_climate_data_catalog`** | matches the artifact (`data_catalog_climate_experiment.yml`) |
| 2.06 | `plot_projection_timeseries` | **`plot_gcm_timeseries`** | shorter, and puts all three WF2 rules on one noun — `fetch_gcm_slice`, `reduce_gcm_series`, `plot_gcm_timeseries`. `projection` would have introduced a second noun for the same thing. Verified accurate: 2.06 consumes the per-member GCM series `reduce_gcm_series` writes |

**Four renames added**, all from reading what the rules actually do:

| # | current | added as | why |
| --- | --- | --- | --- |
| 1.08 | `add_forcing` | `add_climate_forcing` | thin noun beside its siblings |
| 1.10 / 3.02 | `extract_climate_grid` | `extract_historical_climate` | "grid" now collides with `prepare_stress_test_grid`. The new name already agrees with four existing things: the script `extract_historical_climate.py`, the artifact `extract_historical.nc`, the store path `data/climate/historical/<key>/`, and it drops "grid" entirely. **The one cross-workflow rename** — this rule is splatted into WF1 and WF3 from one producer contract, so it touches both Snakefiles plus `climate_store_spec` in `snake_utils.py` |
| 3.06 | `generate_weather_realization` | `generate_weather_realizations` | one invocation produces ALL `RLZ_NUM` of them and the output is a list; singular was simply wrong. It also makes number carry meaning — 3.06 plural (all in one job), 3.09 singular (wildcarded, one job per member) |
| 3.07 | `generate_climate_stress_test` | `perturb_climate_realization` | **the name claims the wrong job.** 3.03 creates the stress test; 3.07 applies ONE point of it to ONE realization via `apply_climate_perturbations`. With 3.03 becoming `prepare_stress_test_grid`, two rules saying "stress test" for different things is worse than today. Pairs with 3.09 `downscale_climate_realization` — same object, next transform |

**Seven corrections to the design itself**, each below in place:

1. the verb table gains `delineate_`, which `delineate_region` conformed to
   without it being listed;
2. the `prepare_`/`write_` split is re-cut on a testable criterion — the old one
   could not decide `write_experiment_config` or `write_climate_data_catalog`;
3. the conforming list becomes a full **34-identifier audit** — the old one
   omitted five rules that do conform, so silence read as clearance;
4. the implementation trap gains the one rule it does **not** apply to (3.10);
5. it also gains the call-site count, which was **six, not three** — the missing
   one is `rule_banner`'s label argument;
6. validation item 4 is rescoped — as written it could not pass;
7. the §9 finding's gap numbers are corrected — 1.14 and 3.12 are occupied, not
   gaps.

Date: 2026-08-04

Decider: Ümit Taner

## Purpose

Bring the Snakemake rule identifiers across the three `Snakefile_*` entry points
onto one verb-and-noun scheme, without changing what any rule does.

**Counts as amended 2026-08-06: 34 identifiers — 12 move, 21 already conform and
must not, 1 (rule 1.07) is removed by a merge rather than renamed.** (Originally
written as "twenty-eight … ten move; eighteen conform", from an inventory that
was never exhaustive — see the full audit below.)

Scope is the rule **identifier** only. Rule bodies, inputs, outputs, numbering,
and the artifacts they produce are out of scope — R9 owns the artifact tree.

## Why this is its own milestone

Rule names are a **contract surface**. `naming.md` §9 states that rule
identifiers are the CLI target surface (`snakemake create_model -s …`) and that
renaming one is a §7 contract rename, so this milestone owes a
`dev/milestones/r10/migration_rule-names.md` record and touches `README.rst`,
`AGENTS.md`, and the vendored docs.

It was deliberately **not** folded into R9. The coupling is weaker than it looks:
rule names appear in `logs/_parts/<W.NN>_<rule>/` and the benchmark part paths,
both **transient** — merged and deleted every run, never baseline-pinned — and in
the merged log's section banners and the benchmark table's rule column, which are
durable *content* rather than paths. No durable artifact path carries a rule name.
R9 already carries a tree migration, a naming rule, a fingerprint redesign, an
experiment lifecycle, a code defect and a baseline re-record; this adds a
CLI-surface rename to that pile for no shared cost.

**One rename was folded into R9** on the principle that a milestone renames what
it falsifies: R9 renames rule 3.11's outputs to `q_indicators.csv` and
`basin_indicators.csv`, so leaving `export_wflow_results` would entrench a
mismatch R9 itself creates. That rename lands with R9; the rest land here — see
the amendments for the current count.

## The convention

Every rule is `<verb>_<noun>`, verb first, always. Verbs come from this list —
one verb per action class, so two rules doing the same kind of work read the
same:

| Verb | Action class |
| --- | --- |
| `fetch_` | acquire from an external source |
| `extract_` | subset or derive from a larger source already present |
| `delineate_` | derive a catchment boundary from hydrography and an outlet |
| `prepare_` | **compute or assemble** something a later rule needs |
| `build_` | construct a model from inputs |
| `add_` | mutate an existing model in place by adding **data** (a hydromt `update`) |
| `declare_` | change what an engine will **emit**, adding no model data |
| `write_` | **emit a record or index** — the emission *is* the work |
| `generate_` | stochastic or synthetic production |
| `downscale_` | resolution transform |
| `perturb_` | apply a climate perturbation to an existing series |
| `run_` | invoke an external engine |
| `reduce_` | **intermediate** aggregation that feeds a later rule |
| `derive_` | compute a workflow's **terminal product** from reduced inputs |
| `plot_` | render a figure |
| `check_` | validate, fail loud |
| `snapshot_` | copy inputs for provenance |
| `gather_` | merge parts |

`reduce_` versus `derive_` is the distinction that needed care, since a reduction
rule and a product rule both turn many inputs into few outputs. The split is by
**position in the workflow, not by operation**: `reduce_gcm_series` feeds a later
rule, while `derive_change_factors` and `derive_wflow_indicators` each produce
their workflow's final answer. That makes the terminal rule of WF2 and WF3 read
alike, which they should.

### `prepare_` versus `write_` — re-cut 2026-08-06

The original split was by *what is produced* — "a config or intermediate" versus
"one small table or index". That is not decidable: `write_experiment_config`
produces a config, and `write_climate_data_catalog` produces an index that a
later rule consumes. Both readings applied to both rules, which is why neither
appeared in the renames or the conforming list.

Split on **where the work is** instead:

> **The test: if you deleted the file-writing, would there be work left?**
> **Yes → `prepare_`. No → `write_`.**

| rule | work left without the write? | verb |
| --- | --- | --- |
| `write_outlet_index` | no — a crosswalk join, then emit | `write_` |
| `write_experiment_config` | no — records what was configured | `write_` |
| `write_model_reference` | no — records which model was used | `write_` |
| `write_climate_data_catalog` | no — enumerates entries | `write_` |
| `prepare_stress_test_grid` | **yes** — `np.linspace` over 12-month vectors | `prepare_` |
| `prepare_weathergen_config` | **yes** — template merge plus `compute_nr_years` | `prepare_` |
| ~~`prepare_runtime_window`~~ | **yes** — assembles the forcing recipe | withdrawn by amendment 2; rule 1.07 merges into 1.08 |
| `prepare_spatial_maps` | **yes** — derives the maps | `prepare_` |

Every previously-accepted choice survives this cut. Note what is **not** the
criterion: whether a later rule consumes the output. `write_climate_data_catalog`
is consumed by 3.09 and is still `write_`, because enumerating entries is all it
does.

### `delineate_` — why a sixteenth verb rather than a rename

`delineate_region` was listed as conforming, but `delineate_` was not in the
table — it conformed to a list it was not on. Basin delineation is the field's
own word for deriving a catchment boundary from hydrography and an outlet;
`derive_region` or `extract_region` would be vocabulary purity bought with a
worse name. Added to the table with its action class stated, so the next reader
finds it rather than inferring it.

Nouns are full words. Only the established domain set abbreviates — `gcm`,
`cmip6`, `wflow`, `rlz`, `cst` — and those are tier-1/tier-2 identifiers under
`naming.md` §6. Ad-hoc contractions (`weagen`, `proj`) are not. Qualifiers are
trailing full words, never two-letter suffixes.

## The renames

Twelve, as amended 2026-08-06.

| # | Current | New | Defect |
| --- | --- | --- | --- |
| 1.05 | `add_gauges_and_outputs` | `declare_wflow_outputs` | **added by amendment 2** — adds no gauges; 1.03 does. The name kept a job that moved away from it |
| 1.08 | `add_forcing` | `add_climate_forcing` | thin noun beside its siblings — does not say *which* forcing |
| 1.11 | `plot_results` | `plot_wflow_evaluation` | vague noun beside the specific `plot_climate_source` / `plot_forcing` |
| 1.10 / 3.02 | `extract_climate_grid` | `extract_historical_climate` | "grid" collides with the stress-test grid; **the one cross-workflow rename** |
| 1.12 | `plot_map` | `plot_basin_map` | vague noun |
| 2.01 | `fetch_gcm_raw` | `fetch_gcm_slice` | dangling adjective; "slice" is the code's own word for the artifact |
| 2.06 | `plot_climate_proj_timeseries` | `plot_gcm_timeseries` | `proj` contraction; `gcm` puts all three WF2 rules on one noun |
| 3.03 | `climate_stress_parameters` | `prepare_stress_test_grid` | no verb |
| 3.04 | `prepare_weagen_config` | `prepare_weathergen_config` | `weagen` appears in no path or directory |
| 3.06 | `generate_weather_realization` | `generate_weather_realizations` | produces ALL realizations in one job; the output is a list |
| 3.07 | `generate_climate_stress_test` | `perturb_climate_realization` | claims 3.03's job; it applies one grid point to one realization |
| 3.08 | `climate_data_catalog` | `write_climate_data_catalog` | no verb |

**Out of scope, and why** — see amendment 1: `export_wflow_results` →
`derive_wflow_indicators` **landed with R9**; `prepare_weagen_config_st` **no
longer exists** (C29 deleted rule 3.05). Per amendment 2, **1.07 `setup_runtime`
is no longer renamed** — it merges into 1.08 (`dev/followups.md` `[R10-1]`), so
its designed target `prepare_runtime_window` is withdrawn rather than replaced.

## The full audit — every identifier, 2026-08-06

The original "eighteen already conform" was illustrative, not exhaustive: it
omitted `gather_benchmarks`, `write_model_reference`, `check_model_reference`,
`write_experiment_config` and the dynamic `run_wflow_batch_<b>`. All five do
conform, but silence in that list read as clearance when it was really absence.
Every identifier in the three Snakefiles is now accounted for.

**Conforming (21), do not touch:** `all`, `snapshot_config`, `delineate_region`,
`prepare_spatial_maps`, `build_wflow_model`, `add_reservoirs_lakes_glaciers`,
`write_outlet_index`, `run_wflow`, `plot_forcing`,
`plot_climate_source`, `gather_logs`, `gather_benchmarks`, `reduce_gcm_series`,
`derive_change_factors`, `derive_wflow_indicators`, `check_project_consistency`,
`check_model_reference`, `write_model_reference`, `write_experiment_config`,
`downscale_climate_realization`, plus the dynamic `run_wflow_batch_<b>`.

`add_forcing` was carried here as an optional candidate and is now **ruled into
the renames** as `add_climate_forcing`. `climate_` rather than `historical_`: it is
already the established noun across rule names — `extract_climate_grid`,
`plot_climate_source`, `downscale_climate_realization`,
`write_climate_data_catalog` — whereas `historical_` appears in none, so it would
have introduced a word for one rule's benefit. There is no ambiguity to resolve
within WF1 anyway: the perturbed forcing is WF3's, added by a differently-named
rule in a different workflow.

**Counts reconcile:** 12 renames + 21 conforming + 1 pending removal (1.07) =
34 identifiers.

**Still 34 after `[R10-6]` §8, by coincidence rather than design.** The merge
removed 1.07 and the vector/raster split added `delineate_spatial_units`, so the
audit now reads 12 renames + **22** conforming = 34. The new rule conforms
without being touched: `delineate_` is already in the verb table (amendment 1),
and `spatial_units` is the noun ADR 0003 §8 uses for the layer set. It is
declared once and splatted into all three workflows, exactly like
`delineate_region`, so it is **one** identifier here and three rows in
`rule-index.md`'s renumber map.

That is the whole of the 34-vs-47 difference: this document counts distinct
identifiers, `rule-index.md` counts declarations. Seven rules are declared in
more than one workflow — `all`, `snapshot_config`, `delineate_region`,
`delineate_spatial_units`, `gather_logs`, `gather_benchmarks` (three times each)
and `extract_historical_climate` (twice) — which is 13 declarations beyond their
first. 47 − 13 = 34. Neither count is wrong; state which one is meant.

**What this list does and does not certify (amendment 2).** It certifies that
each name satisfies `verb_object` against the verb table. It does **not** certify
that the verb is still true of the body — `add_gauges_and_outputs` sat here while
adding no gauges, and it took reading the script to find that. Anything added to
this list in future should be checked both ways, and the check that was run
should be stated.

## The implementation trap

**Every rename must update `LOG_RULES` in the same edit.** That list is the merge
order for `merge_logs`, which discovers a rule's parts by listing the directory
named after its label. An unlisted label is not an error — `merge_logs` is
deliberately scoped to the list so a renamed rule's orphan directory is never
read — so a missed rename produces a log section that silently vanishes from the
merged log while its parts stay on disk forever. The same applies to
`merge_benchmarks`.

**SIX call sites move together, not three (corrected 2026-08-06).** The original
count was taken from memory rather than from a rule; counting them on 1.08
`add_forcing` gives:

| # | site | example |
| --- | --- | --- |
| 1 | `LOG_RULES` entry | `"1.08_add_forcing"` |
| 2 | comment header carrying `W.NN` | `# 1.08  add_forcing — …` |
| 3 | the `rule` identifier | `rule add_forcing:` |
| 4 | **`rule_banner`'s second argument** | `rule_banner("1.08", "add_forcing")` |
| 5 | `log:` path | `{LOG_PARTS_DIR}/1.08_add_forcing.log` |
| 6 | `benchmark:` path | `benchmarks/_parts/1.08_add_forcing.tsv` |

**Site 4 is the one the original list missed**, and it is the one a checklist
would cause someone to skip: `rule_banner` prints the human-facing label in the
run output, so missing it leaves the banner announcing the old name while the
rule carries the new one. Cosmetic rather than breaking — which is exactly why it
would survive a green test run and reach a user.

Sites 5 and 6 are also two lines, not one "path prefix".

**The one rule this does NOT apply to (added 2026-08-06).** Rule 3.10's
identifiers are `run_wflow_batch_0`, `run_wflow_batch_1`, … — parse-time
loop-generated, one per batch — while its log label and its `LOG_RULES` entry are
both the singular `3.10_run_wflow`. That divergence is **deliberate** (P3-3 keys
logs by batch id, not by rule identifier), so identifier and label are not
supposed to match there. Applying the three-call-sites rule mechanically to 3.10
would rename a `LOG_RULES` entry that has no rule to match and break the merge.
None of the twelve renames touches 3.10 — this is stated so a sweep does not
"fix" it.

**A deletion is the same hazard as a rename.** When C29 removed rule 3.05 its
`LOG_RULES` entry went in the same edit, because a label with no producer
contributes an empty section forever — the mirror of the missed-rename case
above. Whatever changes the rule set, the list changes with it.

## Separate finding — the `W.NN` scheme is violated in WF2

> **Superseded in part.** The recommendation below (amend §9, do not renumber)
> was **overruled 2026-08-06** — see the subsection at the end. The finding
> itself stands; the remedy changed.

`naming.md` §9 defines `NN` as "the zero-padded step in **definition order**".
WF2 defines its rules in the order 2.00, 2.03b, 2.03, 2.01, 2.02, 2.04, 2.06,
2.07 — numbers out of order relative to definition — and the series has gaps at
1.14, 2.05 and 3.12 left by merged or removed rules.

**Recommendation: amend §9, do not renumber.** Renumbering churns every log and
benchmark part path across all three workflows, invalidates every `W.NN`
cross-reference in comments and docs, and buys nothing. §9 should say that the
number is a **stable identifier assigned at rule creation**, not a position: it
disambiguates logs across workflows and gives comments a short handle, and
neither purpose needs contiguity or order. Gaps then record history instead of
looking like mistakes.

This is a documentation fix and could land independently of the renames.

### OVERRULED 2026-08-06 — renumber

The owner ruled the other way after the rule-index audit reordered the DAG
diagrams into `data → model → run → records`: with the stages made explicit, the
numbers contradicting them was the more visible defect. **`W.NN` becomes
positional** — contiguous per workflow, every dependency pointing from a lower
number to a higher one. The full old→new map for all **47** identifiers lives in
`dev/reference/workflows/rule-index.md` § *What changed*; the item is
`dev/followups.md` `[R10-5]`.

**Regenerated 2026-08-06, and the count moved twice.** The first published map
carried 45. It reconciles as −1 (`evaluate_wflow_run` never exists — `[R10-2]`
dropped) +3 (`delineate_spatial_units`, one declaration per workflow, from
`[R10-6]` §8). The map was deliberately regenerated at step 6 of the landing
order rather than being the target from the start, precisely so the rule set had
stopped changing first.

**"Every dependency low→high" must be checked against `input:`, `ancient()`
included.** The first draft of the map put `write_outlet_index` and
`plot_basin_map` before `add_climate_forcing` — both declare
`ancient(<model>/.model_final)`, which that rule writes, so both edges pointed
high→low. `ancient()` suppresses the timestamp rerun-trigger and nothing else;
the DAG edge stands. ADR 0004 moved that sentinel onto the forcing rule after
the draft was written, and `ancient()` is why it read as no dependency at all.

The cost above stands and was accepted knowingly. One part of it is worse than
this section anticipated: because the new numbering is contiguous, **retired
numbers get reused** — new 1.07 is `build_wflow_model` where old 1.07 was
`setup_runtime`, and new 3.05 is `write_model_reference` where old 3.05 was the
rule C29 deleted. Sharpest, new 3.10 is `prepare_weathergen_config` where old
3.10 was `run_wflow`. Under the policy this section recommended, a stale `W.NN`
reference merely dangled and was obvious; now it silently resolves to a
different rule. Read every `W.NN` in `dev/milestones/`, `DEVLOG.md` and
`dev/decisions/` **as of its date**, and do not rewrite archived records to the
new numbers — the same rule validation item 4 already applies to old rule names.

**Land it in the same sweep as the renames.** Both touch the same six call sites
per rule and want the same validation, so splitting them pays that cost twice.
Going forward, insert with a letter suffix (`1.09b`) rather than renumbering
again.

## Validation

1. `pytest tests/test_cli.py` — dry-runs all three Snakefiles; a broken rule
   reference fails at parse time.
2. `pytest tests/` once before merging.
3. A full three-workflow run, then confirm the merged log contains a section for
   **every** rule in `LOG_RULES` and that no `_parts/` directory survives — the
   direct check on the trap above.
4. `grep` for each old rule name after the sweep — **scoped to LIVE surfaces**:
   the three `Snakefile_*`, `README.rst`, `AGENTS.md`, `docs/`, `dev/reference/`,
   `blueearth_cst/`, `dev/scripts/` and `tests/` (comments and test IDs
   included). Zero hits there, excluding `migration_rule-names.md`.

   **Rescoped 2026-08-06.** As written this said `dev/` and demanded zero hits
   outside the migration record — which cannot pass and should not.
   `dev/milestones/` is an **archive**: the p31, p32b, r05, r07 and r09 records
   legitimately name `prepare_weagen_config_st`, `export_wflow_results` and the
   rest as historical fact, because that is what those rules were called when
   those milestones ran. Rewriting them to satisfy a grep would falsify the
   record. A hit in an archived milestone document is evidence, not drift; a hit
   in a live surface is drift.

The baseline is **not** affected — no renamed rule changes an output path or
value, and part paths are transient. `check_baseline.py check` should pass
unchanged, which is itself worth asserting.

## Consequences

- **Twelve** rule identifiers change on the CLI surface; a
  `migration_rule-names.md` record is mandatory under `naming.md` §7. It should
  also record the **three** that left scope — one landed with R9, one had its
  rule deleted (3.05), and one is withdrawn in favour of a merge (1.07) — so the
  count reconciles against the original ten.
- `declare_wflow_outputs` is the only rename that also **adds a verb**, so
  `naming.md`'s vocabulary section must gain `declare_` in the same edit, not
  just the new name.
- The merged log and benchmark table gain new section labels — durable content
  changes with no path or value change.
- `naming.md` should carry the verb vocabulary above, so the next rule is named
  from a list rather than by analogy. It currently has no rule-naming section.
