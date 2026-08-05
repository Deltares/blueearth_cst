# Snakemake rule naming — R10 design

Status: **ACCEPTED** by the owner, 2026-08-04. Not implemented.

> **Note added 2026-08-05 — read before implementing. This does not change the
> accepted scope.** Rule 3.05 `prepare_weagen_config_st` is proposed for
> **deletion** by a pending WF3 change (C29 in
> `dev/milestones/r09/wf3-change-requests.md`), which found the per-run config it
> writes carries no per-run information. If C29 is ruled, **drop 3.05's rename
> from the nine** rather than recording a CLI-surface rename for a rule that then
> disappears — the same principle this design already applies in reverse when it
> folds `export_wflow_results` into R9. If C29 is declined, rename it as
> designed. The other eight renames are unaffected; that register's batch plan
> has the rename-by-rename check.

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
| `prepare_` | produce a config or intermediate for a later rule |
| `build_` | construct a model from inputs |
| `add_` | mutate an existing model in place (a hydromt `update`) |
| `write_` | emit one small table or index |
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

Nouns are full words. Only the established domain set abbreviates — `gcm`,
`cmip6`, `wflow`, `rlz`, `cst` — and those are tier-1/tier-2 identifiers under
`naming.md` §6. Ad-hoc contractions (`weagen`, `proj`) are not. Qualifiers are
trailing full words, never two-letter suffixes.

## The renames

| # | Current | New | Defect |
| --- | --- | --- | --- |
| 3.03 | `climate_stress_parameters` | `prepare_stress_grid` | no verb |
| 3.08 | `climate_data_catalog` | `write_climate_catalog` | no verb |
| 3.04 | `prepare_weagen_config` | `prepare_weathergen_config` | `weagen` appears in no path or directory |
| 3.05 | `prepare_weagen_config_st` | `prepare_weathergen_config_perturbed` | same, plus `_st` reads as a truncation — `st_num` is the combination index, not "stress test" |
| 1.07 | `setup_runtime` | `prepare_runtime_window` | `setup_` duplicates `prepare_`; "runtime" alone says nothing |
| 1.11 | `plot_results` | `plot_wflow_evaluation` | vague noun beside the specific `plot_climate_source` / `plot_forcing` |
| 1.12 | `plot_map` | `plot_basin_map` | vague noun |
| 2.01 | `fetch_gcm_raw` | `fetch_gcm_slice` | dangling adjective; "slice" is the code's own word for the artifact |
| 2.06 | `plot_climate_proj_timeseries` | `plot_projection_timeseries` | `proj` contraction |
| *(R9)* | `export_wflow_results` | `derive_wflow_indicators` | wrong verb — the rule aggregates *and* computes; `derive_` is the terminal-product verb. **Lands with R9** |

**Conforming, do not touch:** `all`, `snapshot_config`, `delineate_region`,
`prepare_spatial_maps`, `build_wflow_model`, `add_reservoirs_lakes_glaciers`,
`add_gauges_and_outputs`, `write_outlet_index`, `add_forcing`, `run_wflow`,
`extract_climate_grid`, `plot_forcing`, `plot_climate_source`, `gather_logs`,
`reduce_gcm_series`, `derive_change_factors`, `check_project_consistency`,
`generate_weather_realization`, `generate_climate_stress_test`,
`downscale_climate_realization`.

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
4. `grep` the repository for each old rule name after the sweep: `README.rst`,
   `AGENTS.md`, `dev/`, `docs/`, comments and test IDs. Zero hits outside the
   migration record.

The baseline is **not** affected — no renamed rule changes an output path or
value, and part paths are transient. `check_baseline.py check` should pass
unchanged, which is itself worth asserting.

## Consequences

- Ten rule identifiers change on the CLI surface; a `migration_rule-names.md`
  record is mandatory under `naming.md` §7.
- The merged log and benchmark table gain new section labels — durable content
  changes with no path or value change.
- `naming.md` should carry the verb vocabulary above, so the next rule is named
  from a list rather than by analogy. It currently has no rule-naming section.
