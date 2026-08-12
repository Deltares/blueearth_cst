# Task Brief — Configuration parameter review (planning only, no code)

### Context

Canonical ruleset: `AGENTS.md`. Inventory and problem statement already done:
`dev/working/parameter-placement.md` (DRAFT — its §5 rule is a proposal, not a
premise; do not treat it as decided).

- Surface is **three tiers**: toolbox (36 tracked config files), project (one
  `--configfile`, 55 leaf keys, 17 required), generated-into-`project_dir` (a
  record of a run, never an input).
- Defaults live in two places with no rule: 3 in
  `config/advanced_settings.yml`, 6 as Python `DEFAULT_*` backing config keys.
- Four inert or partly-inert parameters were found **by hand** in one session
  (WF2 `start_month_hyd_year`, `relax_priority`, `static_dir` in WF3, C34's
  `evaluate.model`). Nothing detects this class mechanically.
- Engine-native schemas (`config/defaults/*.yml`) are fixed by AGENTS.md's hard
  constraint — hydromt / wflow / weathergenr vocabulary is used verbatim. They
  are in scope to *describe*, out of scope to *restructure*.

### Goal

Answer four questions with evidence, so that a later design decision on
parameter organisation rests on a complete and checked picture rather than on
precedent. Produce findings and recommendations only.

### Non-goals

- **No code, config, schema or test changes.** Not a single edit to a runtime
  file. Recommendations are written down, not applied.
- No decision on where defaults should live — that is the *output* of this
  review, informed by Q4, not an input.
- No restructuring proposal for engine-native templates.
- Not a design document. If the answers imply an architecture change, say so
  and stop; `design-document` owns that.

### Allowed scope

**Permitted (read only):** the whole repository, and the generated config tree
under `test_case/test_local/**` as evidence of tier 3.

**Write:** exactly one new file —
`dev/working/2026-08-12_config-parameter-review.md`. Update this brief's
`Progress` section as work advances.

**Forbidden:** every other path. Specifically no edits to `config/**`,
`blueearth_cst/**`, `Snakefile_*`, `tests/**`, `AGENTS.md`, or
`dev/working/parameter-placement.md` (the draft is an input; supersede it in
the review's own conclusions rather than editing it).

### Required changes (checklist)

The four questions, sharpened so they do not overlap. Answer in order — Q4
must not reorganise parameters that Q1–Q3 would delete.

1. **Q1 · Reach — which declared parameters never reach the computation?**
   Classify every one of the 55 project-config leaf keys and the
   `config/defaults/*.yml` keys into exactly one of:
   - **(a) never read** — no code reads the key at all;
   - **(b) read, unused** — bound to a variable or a rule `params:` and never
     consumed (`static_dir` in `Snakefile_climate_experiment:41`);
   - **(c) forwarded, dropped** — reaches a call boundary and is discarded
     before the arithmetic (WF2's water year until 2026-08-12; `relax_priority`
     at the `run_weather_generator` wrapper);
   - **(d) live** — a traced path to the consumer that uses it.
   For (a)–(c) give the exact point where the chain stops, as `file:line`.

2. **Q2 · Necessity — of the LIVE parameters, which should not be user-facing?**
   Distinct from Q1: a parameter can work perfectly and still not belong in a
   user's config. Flag each of:
   - only one value is ever valid (`static_dir` can only be `config`, because
     the fallbacks it feeds resolve to in-repo toolbox files);
   - never varied — identical across all shipped configs, and no stated reason
     a project would change it;
   - an implementation detail exposed by accident;
   - superseded by another key.
   Evidence: the set of values the key takes across the four `test_case/`
   configs, the template and `tests/snake_config_fixture.yml`, plus whether any
   other value is admissible.

3. **Q3 · Duplication — where is one concept declared more than once?**
   Four sub-cases, each needing both locations and **which one wins at
   runtime**:
   - the same value defined twice (`DEFAULT_ANCHOR` in
     `metrics_definition.py:18` and `climate_figures.py:120`);
   - one concept under two names, spellings or units (the water year was
     `start_month_hyd_year` as a month name and `year_start_month` as an
     integer);
   - a default in code *and* in config, so the effective value depends on which
     is consulted;
   - a value derivable from another already present (state the derivation).

4. **Q4 · Organisation — is the hierarchy right for a user?**
   Answer each, with a recommendation:
   - Are three tiers the right tiers, or does the split hide something?
   - Within the project config, is `project` / `shared` / `workflows.*` the
     right axis — and is `shared` a coherent category or a leftovers bin?
   - Is nesting depth justified? `shared.basin.automatic_subbasins.max_per_basin`
     is four levels for one integer.
   - Is grouping by *kind*? `shared.basin` currently mixes basin definition,
     catalog bindings and delineation tolerances.
   - **The user-oriented test:** list every key a user must set or review to run
     a NEW BASIN. Is that set contiguous in the file, or scattered? If
     scattered, that is the finding — state the contiguous grouping that would
     replace it.
   - Where should a key's default be *visible* to the user? Give a
     recommendation and the argument against it.

5. **Rank every finding** by consequence, not by count: what could produce a
   wrong number, versus what is only untidy.

6. **Answer the P2 question the draft left open:** could Q1's classification be
   produced *mechanically* rather than by reading? Say whether a "declared keys
   ⊆ read keys" check is feasible against Snakemake's `params:` indirection,
   and if not, what the cheapest partial check would be. A judgement with
   reasons is an acceptable answer; an untested claim that it works is not.

### Progress

- [ ] Q1 reach classification
- [ ] Q2 necessity
- [ ] Q3 duplication
- [ ] Q4 organisation + user-oriented test
- [ ] Ranking and P2 feasibility

### Validation

No test suite applies — nothing executes. Validation is **evidence per claim**:

1. **Every claim carries `file:line`.** A parameter asserted inert names the
   line where its chain stops.
2. **Falsifier for each inertness claim.** The claim "X never reaches the
   computation" is disproved by exhibiting a call path from the config key to a
   consumer. State the search that would find one — e.g.
   `grep -rn "X" blueearth_cst/ Snakefile_*` plus the `sm.params.X` read — and
   report that it was run and returned nothing. Absence claims are the ones no
   amount of reading proves by itself; run the search that would refute them.
3. **Coverage is complete, not sampled.** All 55 project keys and all 14
   `DEFAULT_*` constants appear in the classification, including the
   uninteresting ones. A partial pass silently omits the inert parameter it was
   commissioned to find.
4. **Cross-check against the draft.** Where a conclusion contradicts
   `parameter-placement.md`, say so explicitly — the draft is unreviewed and
   may be wrong.

### Acceptance criteria

- All four questions answered, each finding evidenced and ranked by
  consequence.
- Complete coverage per Validation 3; no key silently skipped.
- Every recommendation states its cost and whether it breaks existing project
  configs.
- Open questions the review cannot settle are named as such, with what would
  settle them — not resolved by assertion.
- **Zero changes to any file outside `dev/working/`.** `git status` shows only
  the review document and this brief.

### Output requirements

One markdown file: `dev/working/2026-08-12_config-parameter-review.md`.

Structure: findings per question (Q1–Q4), then a single ranked recommendation
table — *finding · consequence · proposed action · cost · breaking?* — then
open questions.

No Results delta: nothing executes, so no results change.

### Task constraints

- Planning and review only. The first edit to a runtime file is a scope
  violation, however obvious the fix looks. Record it and move on.
- Do not treat `parameter-placement.md` §5 as decided; it is one input.
- Report assumptions and residual risk.

**Human gates**

- **Gate 1** — after Q1's classification, PAUSE. If it finds inert parameters
  beyond the four already known, the owner decides whether the review widens to
  cover them or records and continues.
- **Gate 2** — before writing Q4's recommendations, PAUSE for the owner to
  confirm whether breaking changes to the project-config schema may be
  proposed. (Latitude was granted for the earlier draft; confirm it still
  holds.)
- **Gate 3** — on completion, PAUSE. No follow-on implementation without an
  explicit new instruction.
