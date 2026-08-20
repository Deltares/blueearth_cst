# Atomic task lanes and worktrees

The stop-signal rules live in `AGENTS.md` § Git task boundary and § Worktrees; this
file is the long form they point to — slot mechanics, scope declaration, the
lifecycle, and the worktree seeding rules with their measurements. Lifted from
`AGENTS.md` @ 46f9df2 during the 2026-08-20 slim-down, unedited except for this
header and the section heading that joins the two source blocks.

### Atomic task lanes — a reusable worktree, one fresh session per task

This repo keeps three persistent worktrees as allocator-managed task lanes. A
lane is detached at `main` while `EMPTY`; `task-start` atomically claims one,
checks out a short-lived branch, and launches a fresh agent process carrying
the claim token. The `.pixi` environment and copied fixtures survive cleanup.

| Slot | Worktree | State |
|---|---|---|
| `session-1` | `.worktrees/blueearth_cst/session-1` | allocator managed; detached when `EMPTY` |
| `session-2` | `.worktrees/blueearth_cst/session-2` | allocator managed; detached when `EMPTY` |
| `session-3` | `.worktrees/blueearth_cst/session-3` | allocator managed; detached when `EMPTY` |

Three slots since 2026-08-19, up from two, because several sessions routinely
work this repo at once — the same reason `worktree_policy: always` is set, and
relaxing that has collided three times (`git log -- .git-workflow.yml`). The
count is a **dial, not a design**: raise it when tasks queue behind occupied
slots, at a one-time cost of one `worktree_seed` copy plus one `.pixi` build per
slot added. The pool is **capacity, not taxonomy**: no slot means anything, and
a task takes whichever is idle. Do not name a slot after a workflow or a kind of
work.

**Why slots and not a worktree per task.** `worktree_policy: always` means every
modifying task builds a worktree, so every task would otherwise pay a full
`worktree_seed` copy *and* a `.pixi` solve — and this repo's `.pixi` is the
expensive half. The slots amortize both into a one-time cost per slot that every
later task reuses. That payoff is independent of concurrency: it arrives on
purely sequential work too.

**Why slots and not the standing lanes they replaced.** From 2026-08-12 to
2026-08-17 this repo ran `lane/devmeta` and `lane/pipeline`, partitioned by
territory. The partition itself was healthy — 45%/55% of traffic, neither a
catch-all — but **37 of 162 commits touched both territories**, almost all "fix
the code, then close the board note". That is 23% of tasks paying a second
worktree visit, and the routing table needed a rule for spanning tasks because
they were the ordinary case rather than the exception. A territory split that
23% of work refuses to respect has stopped predicting the work. Under one slot a
spanning task is simply one branch touching both trees, with no visit and no
split.

The trade is deliberate: lanes made merge order irrelevant **by construction**,
because two territories cannot collide. A slot has no territory, so that
guarantee is gone and the declaration below replaces it.

**Declare the expected write set before editing.** `task-start` begins in
`CLAIMED`, which is read-only. Inspect first, then run `task-scope --path ...`
for every path the task may write; this moves the lane to `OCCUPIED`. Include:

- implementation paths and workflow entry points (`*.smk`, `blueearth_cst/**`);
- shared seams — `blueearth_cst/shared/`, above all `snake_utils.py` (parses
  every Snakefile's config) and `interchange_contracts.py` (**is** the
  wf1→wf2/wf3 seam). 72–85% of each workflow's commits touch `shared/`, so two
  workflow-scoped tasks are **not** independent by default;
- `config/**`, `test_case/snake_config_*.yml`, and the schema they validate;
- the exact test modules each task expects to edit; and
- mutable state outside the checkout — `project_dir`, `.snakemake` locks, the
  shared Julia depot under `~/.julia`.

The registry refuses overlapping path declarations. A `git diff` cannot replace
scope: an untouched file may still be the next file two tasks intend to change.

Declare mutable state outside the checkout with `--resource`, including a
shared `project_dir`, `.snakemake` lock, Julia depot, or board renderer. When
paths or resources overlap, serialize the tasks. Land a shared contract as the
smallest valid base change before allocating dependent work.

Because spanning tasks are the ordinary case here (37 of 162 commits), the
common shape is one task in one slot touching both code and `dev/**` — that is
fine and needs no declaration. The declaration is for the moment a slot is
claimed while another one is still occupied.

**Ownership is the non-expiring filesystem claim, not the branch or process
age.** `lane-banner-hook --current` prints slot, task, branch, state, and scope
before every prompt. If it names a different task, or reports `UNCLAIMED TASK
LANE`, stop; do not switch branches or continue the new task in that session.
Use `task-status` from the primary checkout for the full registry. Recovery is
an explicit `git-steward` action; never delete registry files manually.

**Lifecycle.** Allocate, scope, implement, freeze, then hand off:

```bash
S=.agents/skills/git-workflow/scripts/worktree-session.py
python $S task-start --task <task> --type <type> --base main -- codex
# In the fresh child session, before the first edit:
python $S task-scope --path <expected-path> [--resource <shared-resource>]
# ... edit, run the testing-policy check, and commit ...
python $S task-ready --validation "<checks and result>" [--risk <flag>]
# Stop. git-integrator independently reviews, integrates, verifies, and cleans.
```

`task-start` compares each `worktree_seed` against the primary at allocation
time: a copy the primary has moved past is refreshed, and a copy that *leads*
the primary is left untouched and reported — the fixture-drift failure recorded
in `t2608121258` (closed; see `dev/LOG.md`) is that second case, and it must never be
overwritten. `integration-complete` proves the task branch is ancestral to
`main` before parking the worktree and deleting the local branch.

**Adding a slot.** `slot-create --slot <name> --base main` adds the worktree
detached, applies `worktree_seed` and `worktree_link`, then runs
`worktree_provision` — `pixi install` plus `pixi run install`, several minutes
and a whole `.pixi` prefix on disk. That cost is why the pool is small and
standing rather than per-task, and `--no-provision` defers it (the slot then
cannot run WF3 or the fixture layer until the env is built). `task-start`
creates a missing slot implicitly, so `slot-create` is only how you pay that
cost *before* you need the slot instead of while waiting for it. **Add the new
slot to `session_slots` and the table above in the same commit. Adding capacity
is an owner decision; normal tasks never select a slot directly.

**Routing a task.**

| Situation | Action |
|---|---|
| Any normal modifying task | Run `task-start` from the primary checkout; the allocator chooses the slot and launches a fresh session. |
| Another lane is active | Declare full paths/resources; the registry permits disjoint work and refuses overlap. |
| All slots occupied | Report it and wait or postpone; do not reuse a running session or create a parallel branch there. |
| Primary checkout | Read-only inspection, board reservation, and `git-integrator` integration only. |

`todoboard render` regenerates `dev/TODO.md`; declare
`--resource todo-board-render` so no two lanes run it concurrently.

**Why the pipeline was never split further.** Kept from the lane analysis
because it still governs how work is scoped: only 8 of 68 package-touching
commits touch more than one workflow, but 72–85% of each workflow's commits also
touch `blueearth_cst/shared/` — wf2 has *zero* commits that don't. `shared/` is
145 file-touches, larger than any single workflow territory, because
`snake_utils.py` parses every Snakefile's config and `interchange_contracts.py`
*is* the wf1→wf2/wf3 seam. Likewise 15 of the 35 plot-touching commits (43%)
also touch non-plot code — `cartographic_map.py` is drawn through by rules 1.12
and 1.13, so a change there is never figure-local (see *Figures are terminal
artifacts*). Standardizing every figure is a **sweep**: it edits call sites
everywhere by definition. Both facts are why no territory partition held here.

## Worktrees — seeding, environments, and what a worktree cannot gate

**Run the workflows from the PRIMARY checkout, not from a task worktree.**
Snakemake keeps its "what is up to date" metadata in `.snakemake/` under the
*working directory*, so one `project_dir` driven from two checkouts gets two
independent stores and they disagree — measured 2026-08-02, the same config and
the same project planned 12 jobs from one checkout and 2 from the other.
Snakemake also locks its working directory, so two checkouts running against one
`project_dir` each hold their own lock while writing the same outputs: a
corruption risk, not just confusion. Worktrees are for editing code and running
`pytest`; pipeline runs belong in the one checkout `worktree_policy: always`
already reserves for integration.

**Every worktree builds its OWN pixi env.** This said the opposite until R9 —
that a worktree "resolves to the primary's copy instead of building its own" —
and that is not what happens: each worktree carries its own tracked `pixi.toml`,
so pixi creates a separate `.pixi/` beside it.

The practical consequence, measured in R9 P2: `pixi install` alone is not enough
to run WF3 in a fresh worktree. `weathergenr` comes from `pixi run install`
(remotes), so a worktree that has only had `pixi install` fails at rule 3.06
with `there is no package called 'weathergenr'`. **Run `pixi run install` in a
worktree before running WF3 there** — and prefer running the pipeline from the
primary checkout anyway, for the `.snakemake` reason above.

**A worktree carries no `test_case/`, and that silently downgrades the test
suite.** `test_case/` is untracked, so `git worktree add` does not bring it. The
fixture-dependent layer then **skips instead of failing** — measured 2026-08-07:
`pytest tests/` in a fresh worktree reported *1567 passed, 31 skipped* and looked
like a clean gate, while **15 of those skips were the fixture layer** this file
already names as the one no worktree can exercise. A branch whose
change crosses the project tree (an R9-style move, a `MODEL_DIRNAME` edit) is
exactly the case that layer exists to catch, and exactly the case a worktree
cannot report.

**Seed a new worktree with the fixture subtrees it needs, by COPY:**

```bash
# 46 MB — the tree named by 18 of the 25 fixture references in tests/
cp -r <primary>/test_case/test_local        <worktree>/test_case/
# 248 KB — only dev/scripts/preview_basin_map.py reads it
cp -r <primary>/test_case/basin_map_fixture <worktree>/test_case/
```

Copy those two subtrees, not all of `test_case/` — the whole directory is 361 MB
and most of it is superseded reference trees (`ref_wf2_pre_*`, `test_local_pre_*`,
`_pruned_*`). `test_case/test` and `test_case/gabon` appear in test source but do
not exist on disk; their tests skip on the primary too, so they need nothing.

**Never symlink or junction it.** `tests/test_model_rebuild_cascade.py` runs a
real `snakemake all -c 1` against the fixture, so a link would drive one
`project_dir` from two checkouts — the same `.snakemake` divergence and
concurrent-lock corruption this section already warns about, arriving through the
test suite instead of through a deliberate run. A copy is an independent
`project_dir`; a link is a shared one.

**The agent-config directories are the opposite case, and are SYMLINKED.**
`.claude/`, `.codex/` and `.agents/` are gitignored per-user state, so no
worktree gets them either — but they are read, not written, and one shared
definition is the whole point. A copy at any level would dereference into a
private fork that then answers with last week's version.

**The links are PER SKILL, not per directory**, and this file said otherwise
until 2026-08-19. What a worktree gets is the three top-level directories as
links to the primary's (`worktree_link:`). Inside them, `skills/` is an
ordinary directory whose ENTRIES are one symlink each into
`~/workspace/brain/artifacts/skills/<name>` — 14 of the brain's 60, a curated
selection rather than the whole shelf. So a skill is present or absent one
entry at a time, a missing one is a missing ENTRY rather than a broken
directory link, and a skill can be reachable without being linked here at all:
`todo-board` is absent from both `skills/` dirs and still works, because
`dev/scripts/todoboard.py` searches those two, then the brain path, then
`~/.claude/skills`.

Their absence does not fail, it **downgrades**: measured 2026-08-11, a worktree
session resolved only 4 of the 18 project skills — every generic process skill
still came from `~/.claude/skills`, so the 14 domain ones (`hydromt`,
`snakemake`, `wflow`, `cst-run-control`, `climate-stress-testing`, …) were
missing with nothing reporting it. Codex is worse: `.codex/agents/*.toml` are
regular files with no brain fallback, so a Codex session there has no personas
at all.

Both lists are declared in `.git-workflow.yml` (`worktree_seed:` copies,
`worktree_link:` links) and applied by the launcher at `git worktree add` time.
For a worktree created before that, reapply both from the primary's config:

```bash
python ~/workspace/brain/artifacts/skills/git-workflow/scripts/worktree-session.py sync
```

**Do NOT borrow the primary checkout to run a branch's gate.** The obvious
alternative to seeding — `git checkout --detach <branch>` in the primary, run,
`git checkout main` — parses as safe and is not, because a long test run holds
the checkout for fifteen minutes and nothing reserves it.

Tried 2026-08-07 and it failed exactly that way. Another session merged its own
branch **onto the detached HEAD**, noticed, checked out `main`, and redid the
merge properly — all while the suite was running. `config/basemap/` exists only on
the branch, so it vanished from the tree mid-run and six basemap tests failed. The
failures were pure artifact: the branch was fine, and the run had to be discarded.
Cost was 15 minutes and a false defect report. A stale checkout is recoverable; a
gate result you have to *decide whether to believe* is worse than no gate.

Seeding is therefore not the cheap option, it is the correct one — a seeded
worktree cannot be moved by another session. The residual difference is small and
worth stating: a copied fixture proves the code runs, the primary's tree is the
one the baseline was recorded from. When that distinction actually matters — a
baseline re-record — take the primary deliberately, with no other session live,
which `worktree_policy: always` is what enforces.

`.pixi/` self-ignores through a `.gitignore` the tool writes itself, so it needs
no repo rule. The pytest and ruff caches were redirected out of the root on
2026-08-11 (`pyproject.toml` `cache_dir` / `cache-dir`) and now sit under the
ignored `.tmp/`. **The root carries no cache directory of any kind, and
`tests/test_cache_dir_hygiene.py` fails if one appears** -- matched by shape, so
a tool nobody has added yet is caught too. Ignoring was never the guard: it
keeps such a directory out of a commit, not out of the root, and the two the
redirect left behind sat there unread for a week because nothing failed.
The one invocation that still writes one is `ruff check --isolated` (which
discards config by definition, and both `pyproject.toml` and `ci.yml` invite it
as a rule-set diagnostic); run it as `RUFF_CACHE_DIR=.tmp/ruff_cache ruff check
--isolated`, since the variable outranks the config-less default. That variable
does NOT belong in `pixi.toml` `[activation.env]`: pixi does not expand values
there, so it would have to be relative -- and a relative `RUFF_CACHE_DIR`
resolves against the CWD, scattering `.tmp/ruff_cache` into whichever directory
ruff ran from, while `cache-dir` already resolves correctly from any of them.

