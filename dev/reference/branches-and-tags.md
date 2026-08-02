# Branches & tags — quick reference

Living inventory of the repo's **durable** refs and what each one is for.
Conventions and lifecycle rules live in `dev/roadmap.md` ("Branching and
tagging conventions"); this file only answers "what is this ref?".
Transient branches (`exp/*`, `feat/*`, `pr/*`) don't belong here.

Last updated: 2026-07-19.

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

Planned (cut at each Phase 2 milestone seal):
`r04-projections`, `r05-experiment`, `r06-refactor`.

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
