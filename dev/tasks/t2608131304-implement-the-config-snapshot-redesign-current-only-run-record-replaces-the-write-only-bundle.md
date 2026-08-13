---
title: Implement the config-snapshot redesign — current-only run record replaces the write-only bundle
type: todo-item
status: active
effort: 2
area: config / provenance
origin: config-snapshot design review (2026-08-13)
queue:
created: 2026-08-13
updated: 2026-08-13  # P0 done, Gate 1 tripped
---

> [!note] Overview
> **What** — Replace the content-addressed config bundle with a current-only run record that stays fresh when the toolbox moves, records post-normalization hydromt values, and copies only what the repo cannot give back.
> **Why** — The bundle has zero readers and an imprecise digest; R3's 'actual values used' is satisfied today only by an unprotected debug log; and the toolbox revision is recorded only by the wrapper, so a direct snakemake run records nothing.
> **Effort** — large. Seven phases, most in `lane/pipeline`; design accepted, so this is execution, not scoping.

## The design is accepted — do not re-derive it

`dev/working/2026-08-13_config-snapshot/`:

| File | What it is |
|---|---|
| `design-v3.md` | the accepted design (831 lines) |
| `master.task.md` | program brief: subsystem map, sequencing, gates |
| `p0…p7-*.task.md` | one brief per phase, each independently verifiable |
| `driver-verification.md` | facts checked in-tree, incl. a reviewer tie-break |
| `review-{gpt,fable}-r2.md`, `review-ledger-r1.md` | the review record |

Two external review rounds (GPT `gpt-5.6-sol` and Fable, independent, same
brief) returned `revise` on both v1 and v2; the owner invoked arbitration at
the round cap and Fable authored v3. **No further design review** — the next
artifact is code.

## What the reviews actually caught

The recurring defect had one shape: the design kept specifying a **field**
where a **trigger** was needed. Worth carrying into implementation, because
the same mistake is available at every step.

- **Rule X.01's rerun triggers do not include toolbox identity**, so a
  code-only commit left the record stamped with the previous commit and wrote
  no journal line. Both reviewers found this independently. Fixed by two
  complementary mechanisms — params threading refreshes the record when the
  checkout moves; lifecycle hooks record invocations that execute nothing.
- **The values-used record must serialize POST-normalization kwargs.**
  `build_wflow_model.py:237-268` pops configured args and derives
  `lulc_mapping_fn` at call time, so serializing the template records the
  wrong thing. The reviewers disagreed here; code broke the tie
  (`driver-verification.md`). The decisive case is that function's own comment
  about CORINE read through `vito_mapping_default` — *"Wrong numbers, not a
  missing setting."*
- **The journal must never be a declared output** — Snakemake deletes declared
  outputs before a job runs, so it would truncate to one line every run,
  silently. A one-line journal looks exactly like a young journal.
- **Production carries no `.git`** (the Dockerfile ADDs sources individually),
  so every identity mechanism needs a degraded mode.

## Correction to the design, binding on P4

Design §6 proposes rules `1.16`/`3.17`. **Both are taken** — WF1
`1.16 gather_benchmarks` / `1.17 gather_logs`, WF3 `3.17` / `3.18` the same
pair. Use **`1.15b`** and **`3.16b`**, before the gather rules, per
`dev/reference/naming.md` §8b ("DO NOT RENUMBER TO INSERT A RULE").

## Progress

Sequencing and blocking edges are in `master.task.md`; this is the surface.

- [x] **P0** — probe Snakemake lifecycle handlers on the pinned version.
      Done 2026-08-13, `p0-probe-result.md`. **Gate 1 TRIPPED — stopped.**
      On Snakemake 9.6.2 a "Nothing to be done" no-op fires *no* handler:
      `workflow.py:1375-1377` returns before `_onstart`, unguarded by any flag.
      Hooks therefore cover only invocations in which ≥1 job executed — the set
      params threading already covers — so R5's "record every invocation" is
      unmet exactly where it was needed. Awaiting an owner decision on the
      journal mechanism; P1–P3 are unaffected and can proceed.
- [x] **P1** — `shared/provenance.py`: projection document, two digests,
      `toolbox_identity()`, `environment_file_hashes()`,
      `append_journal_line()`. Landed `906bd69`.
      `projection` defaults to `None` (whole config) rather than being
      required: the two pre-existing callers acquire theirs in P2 and P4, and
      each phase has to leave the tree working on its own. The default is the
      safe direction — over-inclusive, so noisy rather than blind.
- [x] **P2** — `copy_config_files.py`: bundle removed, `run_record.yml`,
      R4 per-file predicate, collision refusal. Landed `4d4243a` + `3a1d270`.
      **Carries P4's checklist item 1** (drop `CONFIG_SNAPSHOT_DIR` and the
      `snapshot_bundle` output from all three Snakefiles), pulled forward on an
      owner ruling — without it every workflow fails at its FIRST rule with
      `MissingOutputException` for the whole time P4 stays blocked.
- [x] **P3** — values-used records from rules 1.07 and 1.08. Landed `27754cd`
      + `61d38d7`, after defect H's remainder cleared the collision in
      `_apply_parameter_steps` (that item is now closed; see `dev/LOG.md`).
      Each step's call kwargs are built ONCE and consumed twice — unwrapped for
      hydromt, rendered for the record. The three falsifiers were written
      before the refactor and observed to fail first.
- [ ] **P4** — Snakefile wiring: projections, parse-time digests, params
      threading, hooks, rules `1.15b`/`3.16b`. Needs P3. Item 1 already landed
      with P2; the hooks are now scoped by the narrowed R5 above.
      **Until it lands, `run_record.yml` carries `projection: null`** — the
      rules declare no `config_projection` param yet, so the record covers the
      whole config. Over-inclusive, not wrong; not a bug to chase.
- [x] **P5 (devmeta half)** — `.gitignore` entry for `.toolbox-commit`,
      `e006e34`. The `Dockerfile` half stays open in `lane/pipeline`.
- [ ] **P6** — cleanup tool + tree inventory + fixture. **Gate 2** before any
      `--delete`. Carries the program's one baseline check.
- [ ] **P7** — `README.md` bundle section, new `config/runs/README.md`.

## Before starting

**`lane/pipeline` was claimed 2026-08-13T03:10** by another session (config
defects, [[t2608130215-fix-the-confirmed-config-defects-found-by-the-dual-parameter-review]]).
Check `.lane-claim` and report if still held — nearly all of this lands there.

That item's **defect H** touches the same function P3 refactors
(`_apply_parameter_steps`), and its `lulc` half already landed in `b31b9a3`.
Land H's remainder first or accept the conflict; do not run them concurrently.

## Refs

- `AGENTS.md` — validation ladder, lane routing, primary-checkout rule.
- `dev/reference/naming.md` §8, §8b — generated-output names, rule numbering.
- Related: [[t2608112239-rename-the-project-dir-config-templates-snapshot-bin-to-say-what-it-holds]]
  — this design **deletes** that bin rather than renaming it, so closing this
  item supersedes that one.
