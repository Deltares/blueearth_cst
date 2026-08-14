# Git conventions — branches, tags, and commits

Two things: an inventory of the repo's **durable** refs and what each one is
for, then the conventions that govern new branches, tags, and commit messages.
Transient branches (`exp/*`, `feat/*`, `pr/*`) don't belong in the inventory.

The conventions below moved here from `dev/roadmap.md` on 2026-08-02 — this
file used to point *back* at the roadmap for them, which meant the rules and
the inventory lived apart. `roadmap.md` is now the phase narrative only.

Inventory last updated: 2026-08-04.

## Durable branches

| Branch                           | Role                                                                                       |
| -------------------------------- | ------------------------------------------------------------------------------------------ |
| `main`                           | Moving trunk and GitHub default. All milestones merge here. The only branch that is pushed routinely. |
| `upstream-deltares`              | Frozen upstream Deltares state at fork-renaming time. **Never commit to it.**              |
| `base/v0.1.0-alpha`              | Frozen historical starting point of the fork.                                              |
| `milestone/01-replication`       | Sealed Phase 1 milestone (kept alive for late patches / PR prep).                          |
| `milestone/02-pixi-installation` | Sealed Phase 1 milestone.                                                                  |
| `milestone/02b-library-upgrades` | Sealed Phase 1 milestone.                                                                  |
| `milestone/02c-tests`            | Sealed Phase 1 milestone (local tip carries a late followups patch).                       |
| `milestone/r01-contracts`        | **Sealed** 2026-07-18 (tag `r01-contracts`) — R1 config-contract migration; merged to `main` 2026-07-18. Kept alive as a durable phase marker. |
| `milestone/r02-naming`           | **Sealed** 2026-07-19 (tag `r02-naming`) — R2 naming style guide; merged to `main` 2026-07-19. Kept alive as a durable phase marker. |
| `milestone/r03-model-builder`    | **Sealed** 2026-07-19 (tag `r03-model-builder`) — R3 workflow-1 cleanup; behavior-preserving (14/14). Merged to `main` 2026-07-19. Kept alive as a durable phase marker. |
| `milestone/r04-projections`      | **Sealed** 2026-07-20 (tag `r04-projections`) — R4 workflow-2 cleanup. Merged to `main`. Kept alive as a durable phase marker. |
| `milestone/r05-experiment`       | **Sealed** 2026-07-20 (tag `r05-experiment`) — R5 workflow-3 + weathergen cleanup. Merged to `main`. Kept alive as a durable phase marker. |
| `milestone/r06-refactor`         | **Sealed** 2026-07-23 (tag `r06-refactor`) — R6 structural refactor; behavior-preserving. Merged to `main`. Kept alive as a durable phase marker. |
| `milestone/p31-experiments`      | **Sealed** 2026-07-24 (tag `p31-experiments`) — P3-1 project/experiment structure. Merged to `main`. Kept alive as a durable phase marker. |
| `milestone/p32a-climate-analysis` | **Sealed** 2026-07-24 (tag `p32a-climate-analysis`) — P3-2a model-independent climate analysis. Merged to `main`. Kept alive as a durable phase marker. |
| `milestone/p32b-interchange-contracts` | **Sealed** 2026-07-24 (tag `p32b-interchange-contracts`) — P3-2b model-swap interchange contracts. Merged to `main`. Kept alive as a durable phase marker. |
| `milestone/p33-performance`      | **Sealed** 2026-07-25 (tag `p33-performance`) — P3-3 performance passes. Merged to `main`. Kept alive as a durable phase marker. |
| `milestone/r07-layout`           | **Sealed** 2026-07-29 (tag `r07-layout`) — R7 project layout; behaviour-preserving. Merged to `main`. Kept alive as a durable phase marker. |
| `milestone/r08-wf2-projections`  | **Sealed** 2026-07-31 (tag `r08-wf2-projections`) — R8 WF2 v2.0 GCM projections analysis. Merged to `main`. Kept alive as a durable phase marker. |
| `milestone/r09-project-tree`     | **Sealed** 2026-08-07 (tag `r09-project-tree`) — R9 project-tree migration; all five phases, landing gate nine of nine. Merged to `main`. Kept alive as a durable phase marker. |
| `milestone/r10-rule-naming`      | **Sealed** 2026-08-07 (tag `r10-rule-naming`) — R10 rule identifiers. Cut **retroactively** at `7164d83`, R10's completion point: the work ran on `fix/r09-followups` because it began as R9 followups and grew into the milestone, so this branch records where R10 finished rather than how it got there. Kept alive as a durable phase marker. |
| `milestone/r11-wf3-artifacts`    | **Sealed** 2026-08-08 (tag `r11-wf3-artifacts`) — R11 WF3 result tables and run identification. Cut at `94c5bea`, main's tip: P1 and P2 landed through task branches and P3 ran in the primary checkout (the run and the single baseline re-record require it), so like `r10` this branch records where the milestone finished rather than how it got there. Kept alive as a durable phase marker. |
| `docs/wf3-redesign`              | **Frozen input, do not merge.** The `wf3-experiment-v2` design run (2026-08-01→04), 14,855 lines under `dev/working/design-runs/`. G2 ratified 2026-08-08 as an *architectural input, not an implementable spec* — the tree it was designed against no longer exists. Durable record: `dev/reference/workflows/wf3-experiment-v2-design-review-record.md`. Kept as the citable home of the raw scratch, per the WF2 precedent. |
| `origin/fao` (remote-only)       | Inherited upstream project branch (FAO / DCRM work, ~32 commits off-trunk). Not tracked locally; review before ever deleting. |

## Tags

Tags are permanent rollback points; they never move.

| Tag               | Date       | Meaning                                                                          |
| ----------------- | ---------- | -------------------------------------------------------------------------------- |
| `v0.1.0-alpha`    | 2024-09-26 | Upstream release state at the fork point (same commit as `base/v0.1.0-alpha`).    |
| `m01-replication` | 2026-05-07 | Phase 1 seal: replication baseline + fingerprint manifest.                        |
| `m02-pixi`        | 2026-05-07 | Phase 1 seal: pixi env + install.                                                 |
| `m02b-upgrades`   | 2026-05-08 | Phase 1 seal: hydromt/wflow/Python-stack library upgrades.                        |
| `v0.2.0-alpha`    | 2026-05-09 | Release: foundation phase sealed, Phase 2 designed.                               |
| `m02c-tests`      | 2026-07-17 | Phase 1 seal: unit-test coverage for 4 `src/` modules.                            |
| `pre-r01`         | 2026-07-18 | Checkpoint before R1: last flat-config-schema commit; green suite (47/3/2); all three workflow smoke tests verified. |
| `r01-contracts`   | 2026-07-18 | Phase 2 seal: sectioned config schema (project/shared/workflows); suite 51/3/2. Sealed on invariance-by-construction — M2b baseline left untouched (stale; see `dev/milestones/r01/baseline_diffs.md`). |
| `r02-naming`      | 2026-07-19 | Phase 2 seal: naming style guide (`dev/reference/naming.md`). Docs-only; existing names grandfathered; suite 51/3/2. |
| `r03-model-builder` | 2026-07-19 | Phase 2 seal: workflow-1 (model builder) cleanup — shared `snake_utils` (`get_config`/`tee_to_log`), per-rule log/benchmark, `outlet_index.csv`, gauges hardening, structured waterbodies sentinel. Behavior-preserving (14/14); suite 73/3/2. |
| `r04-projections` | 2026-07-20 | Phase 2 seal: workflow-2 (climate projections) cleanup, inheriting the R3 patterns. Contract doc `dev/reference/workflows/analyze_projections.md`; 11 commits. |
| `r05-experiment`  | 2026-07-20 | Phase 2 seal: workflow-3 (climate experiment) + the R weathergen layer cleaned up, inheriting the R3/R4 patterns. Contract doc `dev/reference/workflows/run_stress_test.md`; 12 commits. |
| `r06-refactor`    | 2026-07-23 | Phase 2 seal: structural refactor — `src/` → `blueearth_cst/` package, `config/` three-bin split, runners → `scripts/`, `enabled:`-aware wrapper, `MIGRATION.md` (51 renames). Behavior-preserving (run-relative baseline + full-tree semantic diff clean). |
| `p31-experiments` | 2026-07-24 | Phase 3 seal: project/experiment structure. |
| `p32a-climate-analysis` | 2026-07-24 | Phase 3 seal: model-independent climate analysis. |
| `p32b-interchange-contracts` | 2026-07-24 | Phase 3 seal: model-swap interchange contracts. |
| `p33-performance` | 2026-07-25 | Phase 3 seal: performance passes; user-signed milestone gate. |
| `r07-layout`      | 2026-07-29 | Phase 4 seal: project layout — single climate store with a model-free shared producer, engine subtrees under `project_dir`, config split four ways, project-level `plots/` and `data/` retired. Behaviour-preserving (full-tree semantic diff clean, discharge bit-identical). |
| `r08-wf2-projections` | 2026-07-31 | Phase 5 seal: WF2 v2.0 GCM projections analysis; all seven §8 migration steps implemented. User migration note in `docs/migration-r08-wf2.md`. |
| `r09-project-tree` | 2026-08-07 | Phase 6 seal: generated project tree — six semantic roots, fan-out members keyed by filename, result tables renamed, pointer-derived model fingerprint with a drift guard, experiment freezing. Landing gate nine of nine; closing record `dev/milestones/r09/closing-record.md`. Work completed 2026-08-05; sealed two days later, which is why the seal is now a named step in the roadmap's cross-cutting principles. |
| `r10-rule-naming` | 2026-08-07 | Phase 7 seal: rule identifiers on one `<verb>_<noun>` scheme — twelve renames, `W.NN` renumbered to follow the DAG, `LOG_RULES` made a conformance test, and ADR 0003 §8–12's spatial-units split. Gates: suite 1526 from the primary, a full three-workflow run, `check_baseline` 8/8 after it, `tree-check` 186/0. Script modules deliberately not renamed. |

| `r11-wf3-artifacts` | 2026-08-08 | Phase 8 seal (R11 of R11–R12): WF3 result tables wide→long with one table per output variable, `cst_`→`st_` member identification with zero-padded ids, the stress-test design table and `st_id`, and `aggregate_rlz` retired as a hard error. Gates: suite 1707 from the primary, a full three-workflow run, exactly one baseline re-record with `check` green after it, `tree-check` 221/0. **The run found three defects nothing else could** — two metrics silently absent, a design table recording perturbations never applied, and a test that had asserted nothing since R9; see `dev/milestones/r11/phase-3-run-report.md`. |

`r05-experiment`, `p31-experiments`, `p32a-climate-analysis`, and
`p32b-interchange-contracts` are **lightweight** tags; every other tag above is
annotated and carries a message.

Planned (cut at seal): `r09-project-tree` (Phase 6),
`r10-rule-naming` (Phase 7).

## Using a checkpoint tag (e.g. `pre-r01`)

```bash
git diff pre-r01 -- <path>            # what changed since the checkpoint
git checkout pre-r01 -- <path>        # restore one file from the checkpoint
git reset --hard pre-r01              # throw away all commits on the current
                                      # branch since the checkpoint (destructive)
```

Tags protect committed state only — not the working tree, and not run outputs
under `project_dir` (the baseline manifest covers those).

## Maintenance

Update this file (and its date) whenever a durable branch or tag is created,
sealed, or retired. Local tags/branches reach `origin` only on explicit push
(`git push origin <tag>` / `--tags`).

---
## Branching and tagging conventions

| Branch type   | Pattern                       | Purpose                                                                  |
| ------------- | ----------------------------- | ------------------------------------------------------------------------ |
| Frozen base   | `base/<start-point>`          | Historical starting point of the fork (e.g. `base/v0.1.0-alpha`).        |
| Phase 1 milestone | `milestone/<NN>-<topic>`  | Sealed; pattern preserved on existing branches (`milestone/02c-tests`).  |
| Phase 2 milestone | `milestone/r<NN>-<topic>` | Active; example `milestone/r01-contracts`, `milestone/r03-model-builder`. |
| Phase 3 sub-milestone | `milestone/p3<N><letter?>-<topic>` | Sealed; pattern preserved on existing branches (`milestone/p32a-climate-analysis`). |
| Experiment    | `exp/r<NN>-<topic>`           | Messy trial branch off a Phase 2 milestone.                              |
| Feature       | `feat/r<NN>-<topic>`          | Cleaner implementation off a Phase 2 milestone, intended to be merged in. |
| Pull request  | `pr/<NN>-<topic>`             | Clean branch prepared for upstream review.                               |

**Tags.** Phase 1 tags use `m##-<topic>` and stay frozen
(`m01-replication`, `m02-pixi`, `m02b-upgrades`, `m02c-tests`). Phase 2
onward use `r##-<topic>`; Phase 3's sub-milestones use `p3#-<topic>`.
The tags themselves are inventoried in § Tags above — that table is the
single list, so don't restate it here; a second copy is what went stale
last time. Tags are permanent rollback points; milestone branches stay
alive after their tag for late patches or PR prep.

**Stacked, not parallel.** Each milestone branches from the previous
milestone's tip (not from `base/`). Phase 2 starts from the
`m02c-tests` tag. R1, R2 are pre-workflow contracts and conventions
that R3-R5 inherit; R6 is the cross-cutting structural refactor.
Once a milestone has merged, `main`'s tip *is* the previous milestone's
tip, so later milestones are cut from `main` — verify rather than assume
(`git log --oneline main..milestone/<previous>` must be empty). R9 was
cut this way on 2026-08-04, with R7 and R8 both confirmed empty.

**Remotes.**
- `origin` — your fork (`github.com/tanerumit/blueearth_cst`).
- `upstream` — the original Deltares repo
  (`github.com/Deltares/blueearth_cst`), fetch-only.

The branch `upstream-deltares` (formerly `main`) freezes the upstream
Deltares state the fork tracked at renaming time; never commit to it.
`main` is the moving trunk and the GitHub default branch.

**PRs back to upstream** go from `pr/<NN>-<topic>` branches, not
directly from milestone branches. One PR per milestone is the default;
only stack PRs when maintainers explicitly agree to review them in
series.

---

## Sealing a milestone

Alongside merging, tagging, and updating the two tables above:

- [ ] **Ask which reference documents this milestone superseded**, and seal
      each one: a `> **SUPERSEDED — … (sealed YYYY-MM-DD).**` banner at its
      head naming what replaced it and where current truth lives, plus an
      entry in `dev/reference/sealed-records.yml`.

Seal them; do **not** migrate their paths. A superseded document is kept
because it is the baseline that milestone's commits were checked against, and
freshening its paths leaves its line numbers, rule names and module locations
lying just as loudly while making the document *look* maintained — strictly
worse than leaving it obviously old.

This step is a judgment and cannot be automated: nothing can infer from a
document's content that it is a record. `tests/test_sealed_records.py` enforces
only what the registry already lists — that each entry keeps its banner and
still hashes to what was sealed. Skipping the question is how
`run_stress_test.md` spent four milestones reading as a live WF3 contract
(R9 P5 F2).

---

## Commit strategy

Branch and tag naming live in "Branching and tagging conventions"
above. This section covers commit messages only.

**Subject format.** `<prefix>: <imperative subject, ≤72 chars>`. The
`<prefix>` matches the milestone the commit belongs to:

- Phase 1 (sealed): `m01:`, `m02:`, `m02b:`, `m02c:` — historical
  prefix on existing commits, do not rewrite.
- Phase 2 (active): `r01:`, `r02:`, `r03:`, `r04:`, `r05:`, `r06:`.
- Phase 3 (active): `p31:` (P3-1 experiment structure), `p32a:` (P3-2a
  model-independent climate analysis), `p32b:` (P3-2b model-swap
  interchange contracts), `p33:` (P3-3 performance passes).
- Repo housekeeping that doesn't belong to a milestone: `chore:`
  (e.g. updating this roadmap, `.gitignore`, fixing typos in
  unrelated docs).

Examples:

- `r01: migrate test config + 3 Snakefiles to sectioned schema`
- `r02: add dev/reference/naming.md + CLAUDE.md pointer`
- `r03: collapse get_config into src/snake_utils.py`
- `r04: fix calendar handling in get_stats_climate_proj.py`
- `r05: extract stress-test grid into tested helper`
- `chore(dev): split roadmap into phase-1 / phase-2 sections`

**Body.** Optional. Include only when the *why* isn't obvious from
the diff. Wrap at ~72 chars. Don't restate what the diff shows.

**Granularity.** One logical change per commit. If the subject needs
the word "and", split it.

**Never commit.**
- Outputs under `project_dir/`.
- Files matching `*_local.yml` or other local-only configs.
- Secrets, credentials, large binary fixtures.
- Generated baselines other than `dev/baseline/manifest.json` itself.

If any of these slip in, update `.gitignore` first, then remove from
history if the commit hasn't been pushed.

**Merges and tags.** Default merge-commit messages are fine — don't
hand-craft them. Tag messages should restate the milestone goal in
one line (e.g. `r03-model-builder: model creation workflow + scripts
cleaned`).

---
