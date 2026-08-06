# Snakemake rule naming — R10 design

Status: **ACCEPTED** by the owner, 2026-08-04. **AMENDED 2026-08-06.** Not implemented.

## Amendment — 2026-08-06

Ruled by the owner after a full re-audit against the code as it stands. The
design was accepted before R9's followups landed, and two of its rules moved
underneath it.

**Scope is now EIGHT renames, not ten.** Two dropped for opposite reasons:

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

**Four corrections to the design itself**, each below in place: the verb table
gains `delineate_`; the `prepare_`/`write_` split is re-cut on a testable
criterion; the conforming list becomes a full 34-rule audit; and the
implementation trap gains the one rule it does not apply to. Validation item 4 is
rescoped — as written it could not pass.

Date: 2026-08-04

Decider: Ümit Taner

## Purpose

Bring the twenty-eight Snakemake rule identifiers across the three `Snakefile_*`
entry points onto one verb-and-noun scheme, without changing what any rule does.
Ten rules move; eighteen already conform and must not.

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
mismatch R9 itself creates. That rename lands with R9; the other nine land here.

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
| `add_` | mutate an existing model in place (a hydromt `update`) |
| `write_` | **emit a record or index** — the emission *is* the work |
| `generate_` | stochastic or synthetic production |
| `downscale_` | resolution transform |
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
| `prepare_runtime_window` | **yes** — computes the window | `prepare_` |
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

Eight, as amended 2026-08-06.

| # | Current | New | Defect |
| --- | --- | --- | --- |
| 1.07 | `setup_runtime` | `prepare_runtime_window` | `setup_` duplicates `prepare_`; "runtime" alone says nothing |
| 1.11 | `plot_results` | `plot_wflow_evaluation` | vague noun beside the specific `plot_climate_source` / `plot_forcing` |
| 1.12 | `plot_map` | `plot_basin_map` | vague noun |
| 2.01 | `fetch_gcm_raw` | `fetch_gcm_slice` | dangling adjective; "slice" is the code's own word for the artifact |
| 2.06 | `plot_climate_proj_timeseries` | `plot_gcm_timeseries` | `proj` contraction; `gcm` puts all three WF2 rules on one noun |
| 3.03 | `climate_stress_parameters` | `prepare_stress_test_grid` | no verb |
| 3.04 | `prepare_weagen_config` | `prepare_weathergen_config` | `weagen` appears in no path or directory |
| 3.08 | `climate_data_catalog` | `write_climate_data_catalog` | no verb |

**Out of scope, and why** — see the amendment: `export_wflow_results` →
`derive_wflow_indicators` **landed with R9**; `prepare_weagen_config_st` **no
longer exists** (C29 deleted rule 3.05).

## The full audit — every identifier, 2026-08-06

The original "eighteen already conform" was illustrative, not exhaustive: it
omitted `gather_benchmarks`, `write_model_reference`, `check_model_reference`,
`write_experiment_config` and the dynamic `run_wflow_batch_<b>`. All five do
conform, but silence in that list read as clearance when it was really absence.
Every identifier in the three Snakefiles is now accounted for.

**Conforming (24), do not touch:** `all`, `snapshot_config`, `delineate_region`,
`prepare_spatial_maps`, `build_wflow_model`, `add_reservoirs_lakes_glaciers`,
`add_gauges_and_outputs`, `write_outlet_index`, `run_wflow`,
`extract_climate_grid`, `plot_forcing`, `plot_climate_source`, `gather_logs`,
`gather_benchmarks`, `reduce_gcm_series`, `derive_change_factors`,
`derive_wflow_indicators`, `check_project_consistency`, `check_model_reference`,
`write_model_reference`, `write_experiment_config`,
`generate_weather_realization`, `generate_climate_stress_test`,
`downscale_climate_realization`, plus the dynamic `run_wflow_batch_<b>`.

**One optional candidate, not ruled:** `add_forcing` has a thin noun beside its
siblings `add_gauges_and_outputs` and `add_reservoirs_lakes_glaciers` — it does
not say *which* forcing. It adds the historical climate forcing
(`inmaps_historical.nc`), so `add_historical_forcing` would match the artifact and
draw the contrast with WF3's per-member forcing. Left out of the eight
deliberately; take it or leave it, but decide rather than inherit.

## The implementation trap

**Every rename must update `LOG_RULES` in the same edit.** That list is the merge
order for `merge_logs`, which discovers a rule's parts by listing the directory
named after its label. An unlisted label is not an error — `merge_logs` is
deliberately scoped to the list so a renamed rule's orphan directory is never
read — so a missed rename produces a log section that silently vanishes from the
merged log while its parts stay on disk forever. The same applies to
`merge_benchmarks`.

Three call sites move together for each rename: the `rule` identifier, its
`log:`/`benchmark:` path prefix, and its `LOG_RULES` entry. The comment header
carrying the `W.NN` number moves too.

**The one rule this does NOT apply to (added 2026-08-06).** Rule 3.10's
identifiers are `run_wflow_batch_0`, `run_wflow_batch_1`, … — parse-time
loop-generated, one per batch — while its log label and its `LOG_RULES` entry are
both the singular `3.10_run_wflow`. That divergence is **deliberate** (P3-3 keys
logs by batch id, not by rule identifier), so identifier and label are not
supposed to match there. Applying the three-call-sites rule mechanically to 3.10
would rename a `LOG_RULES` entry that has no rule to match and break the merge.
None of the eight renames touches 3.10 — this is stated so a sweep does not
"fix" it.

**A deletion is the same hazard as a rename.** When C29 removed rule 3.05 its
`LOG_RULES` entry went in the same edit, because a label with no producer
contributes an empty section forever — the mirror of the missed-rename case
above. Whatever changes the rule set, the list changes with it.

## Separate finding — the `W.NN` scheme is violated in WF2

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

- **Eight** rule identifiers change on the CLI surface; a
  `migration_rule-names.md` record is mandatory under `naming.md` §7. It should
  also record the two that left scope — one landed with R9, one had its rule
  deleted — so the count reconciles against the original ten.
- The merged log and benchmark table gain new section labels — durable content
  changes with no path or value change.
- `naming.md` should carry the verb vocabulary above, so the next rule is named
  from a list rather than by analogy. It currently has no rule-naming section.
