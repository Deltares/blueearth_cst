# Git conventions

Last updated: 2026-10-20

Current branch, tag and commit conventions. Branch/worktree lifecycle belongs to the `git-workflow` skill and
repository configuration. Inspect `git branch -a` and `git tag` for existing refs; historical ref inventory
(private-checkout-only citation: `../records/milestones/git-ref-inventory.md`) preserves their earlier
context.

## Using a checkpoint tag (e.g. `pre-r01`)

```bash
git diff pre-r01 -- <path>            # what changed since the checkpoint
git checkout pre-r01 -- <path>        # restore one file from the checkpoint
git reset --hard pre-r01              # throw away all commits on the current
                                      # branch since the checkpoint (destructive)
```

Tags protect committed state only — not the working tree, and not run outputs
under `project_dir` (the baseline manifest covers those).

## Branching and tagging conventions

**Branches are `<type>/<slug>`** — the Conventional Commits type, then a short kebab-case description
of the work: `fix/fixture-digest-drift-attribution`, `docs/config-modularization`,
`refactor/wf0-rule-label-constants`. No milestone number in the name; the branch is short-lived and its
milestone, if any, is recorded on the board item and in the commit.

Historical patterns preserved on existing refs, not used for new work:

| Pattern | Was |
|---|---|
| `base/<start-point>` | Frozen historical starting point of the fork |
| `milestone/<NN>-<topic>`, `milestone/r<NN>-<topic>`, `milestone/p3<N><letter?>-<topic>` | Sealed phase markers, one per milestone |
| `lane/<territory>` | The two standing territory lanes, retired in favour of allocator-managed session slots |

**Creating and landing a branch is the `git-workflow` skill's contract**, not this file's: a task claims
a session slot, works on its own branch, and the integrator lands it. Do not cut a branch by hand in the
primary checkout.

**Tags.** Phase 1 tags use `m##-<topic>`; Phase 2 onward `r##-<topic>`; Phase 3 sub-milestones
`p3#-<topic>`; archived design runs and deleted branches `archive/<topic>`. Use `git tag` for the current inventory. Tags
are permanent rollback points and never move. A milestone branch stays alive after its tag.

Not every piece of work takes a tag or a milestone branch. Both are for a deliberate seal; ordinary work
lands on `main` and is found through `dev/LOG.md`.

**Remotes.**

- `origin` — the fork (`github.com/tanerumit/blueearth_cst`), where CI runs.
- `upstream` — the original Deltares repo (`github.com/Deltares/blueearth_cst`), fetch-only.

`gh` resolves to `upstream` until told otherwise — see `dev/reference/dev-validation.md`.

The branch `upstream-deltares` freezes the upstream Deltares state the fork tracked at renaming time;
never commit to it. `main` is the moving trunk and the GitHub default.

**PRs back to upstream** go from a dedicated branch, not from `main`. One PR per milestone is the
default; only stack PRs when maintainers explicitly agree to review them in series.

---

## Sealing a milestone

Follow [record lifecycle](dev-record-lifecycle.md#milestone-records-and-historical-recovery) for
preserving superseded specifications and their seal metadata.

## Commit strategy

Branch and tag naming live in § Branching and tagging conventions above. This section is commit messages
only.

**Subject format — Conventional Commits.** `<type>(<scope>): <imperative subject>`, subject ≤72 chars,
no trailing period. Append `!` before the colon for a breaking change (`refactor(workflows)!:`).

| Type | For |
|---|---|
| `feat` | new capability |
| `fix` | a defect corrected |
| `docs` | documentation, including `dev/` records and the board |
| `refactor` | behaviour-preserving restructuring |
| `test` | tests only |
| `chore` | housekeeping — env, ignore rules, tooling |
| `perf` | a measured performance change |
| `style` | formatting only, no code change |

The scope is the area touched, lowercase: a workflow (`wf0`, `wf2`, `wf3`), a subsystem (`console`,
`stage`, `climate`, `plotting`, `runner`), or a documentation surface (`board`, `dev`, `ref`, `agents`,
`tests`). Omit it when the change is genuinely repo-wide.

Examples, all from this repo's history:

- `fix(wf2): refuse to attribute drift against a mismatched fixture`
- `refactor(tests): read LOG_RULES from the parsed workflow, and retire the second parser`
- `docs(ref): state the rules in dev-naming-conventions.md, drop the archaeology`
- `chore(board): close t2608201134, and record the ruling`
- `refactor(workflows)!: rename the workflows.<name> config keys and every derived path`

**Body.** Optional. Include only when the *why* is not obvious from the diff. Do not restate what the
diff shows.

**Granularity.** One logical change per commit. If the subject needs the word "and" to join two
unrelated changes, split it.

**Scope every commit by explicit pathspec** to the files the task authored. A bare `git commit -a`
sweeps whatever else is dirty in the checkout, which is how another session's half-finished work lands
in an unrelated commit.

**Never commit.**

- Outputs under `project_dir/`.
- Files matching `*_local.yml` or other local-only configs.
- Secrets, credentials, large binary fixtures.
- Generated baselines other than `dev/baseline/manifest.json` itself.
- Hand-edited `pixi.lock` or `Manifest.toml`.

If any of these slip in, update `.gitignore` first, then remove from history if the commit has not been
pushed.

**Merges and tags.** Default merge-commit messages are fine — do not hand-craft them. A tag message
restates the milestone goal in one line.
