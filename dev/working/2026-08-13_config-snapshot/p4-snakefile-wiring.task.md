# Task Brief — P4: Snakefile wiring

### Context

`AGENTS.md`; design `design-v3.md` §5.2, §5.3, §5.7, §5.8. Depends on P0, P1,
P2, P3. **This is the phase where the design's blocking defect is fixed.**

- Rule `X.01`'s rerun triggers today are the configfile, referenced inputs, its
  script and its params — **toolbox identity is none of them**, so a code-only
  commit change leaves the record stamped with the previous commit.
- **Rule-number correction (binding):** the design says `1.16`/`3.17`; both are
  taken (WF1 `1.16 gather_benchmarks`, `1.17 gather_logs`; WF3 `3.17`, `3.18`
  the same pair). Use **`1.15b`** and **`3.16b`**, before the gather rules, per
  `dev/reference/naming.md` §8b. Do not renumber anything.

### Goal

Wire the three Snakefiles so the record stays fresh, the journal records every
invocation, and the bundle is gone from the DAG.

### Non-goals

No changes to the writer or the provenance helpers (P1/P2 own them).

### Allowed scope

- **Permitted:** `Snakefile_model_creation`, `Snakefile_climate_projections`,
  `Snakefile_climate_experiment`, `tests/test_snapshot_config_rules.py`,
  `tests/test_cli.py`.
- **Forbidden:** `dev/scripts/semantic_tree_diff.py` (P6 owns the inventory).

### Required changes (checklist)

1. Drop `CONFIG_SNAPSHOT_DIR` and the `snapshot_bundle` output from all three.
2. Declare the consumed-key projection per workflow. **WF3's is derived** from
   the same `guarded_sections` tuple the drift guard hashes — not written out
   beside it, which is proximity rather than enforcement.
3. Compute both digests at parse time; thread `configuration_inputs_sha256`
   through X.01's `params:` as a string digest so the params rerun-trigger
   refires the rule when the checkout, lock files, or referenced bytes move.
4. Pass the **projection**, not the whole config, as the `effective_config`
   param.
5. Add `onstart:` / `onsuccess:` / `onerror:` handlers calling
   `append_journal_line`. WF3 lines carry `experiment`.
   **The journal is never a declared output** — Snakemake deletes declared
   outputs before a job runs, which would truncate it to one line every time,
   silently.
   **Scope, per the narrowed R5 (§4):** these handlers fire only when at least
   one job executes — a no-op invocation records nothing, and that is the
   accepted behaviour, not a defect to engineer around (`p0-probe-result.md`,
   design §7 item 11). Do **not** add a parse-time or `atexit` emitter to reach
   past it; the owner ruled against that on 2026-08-13.
6. Add rules **`1.15b`** and **`3.16b`** `write_run_metadata` emitting
   `run_metadata.json` sidecars; extend WF2's existing `provenance.json`.

### Validation

- Rung 1: `pytest tests/test_snapshot_config_rules.py`.
- Rung 2 (new behavioural tests):
  - **journal is not a declared output** — assert it appears in no rule's
    `output:`;
  - WF3 projection equals the derived union of `guarded_sections` +
    `workflows.climate_experiment`;
  - per-workflow mutation tests: every key in the projection moves
    `effective_config_sha256`; a sibling key outside it does not.
- Rung 3: `pytest tests/test_cli.py` — dry-runs all three Snakefiles; the only
  gate that catches a malformed rule.
- Rung 4: `pixi run test-full` at phase merge — this phase touches Snakefiles
  and `script:` signatures, the paths that tier exists to guard.
- Rung 5: end-to-end `scripts/run_workflows.py --config
  test_case/snake_config_rapid.yml` from the **primary checkout**.

**Falsifiers, both required — each targets an absence, which no ordinary test
reaches:**
- *"the record refreshes when the checkout moves"* — run WF1, commit an
  unrelated code change, re-run without touching the config; a `run_record.yml`
  still carrying the **old** commit disproves it.
- *"the journal accumulates"* — **amended 2026-08-13 after the P0 probe; the
  original wording tested the one case that cannot work.** It said "run any
  workflow twice with no config change", but with no change the second run is a
  "Nothing to be done" no-op, which fires no handler and appends nothing
  (`p0-probe-result.md`) — so the original falsifier would fire on correct code.
  Run it instead as: **run any workflow, then re-run it with `--forcerun` on any
  rule** (or after touching one input), so the second invocation executes at
  least one job. A `journal.jsonl` with one line, or with a terminal line
  missing for the second run, disproves accumulation.
  The **complementary** check, now that R5 is narrowed (§4): a genuine no-op
  re-run must append **nothing** — a line appearing there would mean the
  journal is being written from somewhere other than the handlers.

### Acceptance criteria

Both falsifiers fail to fire; `test_cli.py` green on all three; no bundle
directory appears in a fresh run.

### Task constraints

Rule numbering is `W.NN` positional per `naming.md`; a new rule takes a letter
suffix and every dependency must point from a lower number to a higher one.
