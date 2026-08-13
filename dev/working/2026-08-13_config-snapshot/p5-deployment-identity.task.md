# Task Brief — P5: deployment identity

### Context

`AGENTS.md`; design `design-v3.md` §5.2 (resolution step 2), open item 9.
Depends on P1's `toolbox_identity()` contract.

- Verified: the `Dockerfile` `ADD`s `src` and each Snakefile individually — the
  image carries **no `.git`**. So `git rev-parse` in a deployed container
  returns nothing and every provenance guarantee degrades exactly where outputs
  are quoted from.
- `run_workflows._git_metadata()` already returns `{"commit": None,
  "dirty": None}` on failure, so the degraded path exists in code and was
  simply unspecified.

### Goal

Give a deployed image a commit identity without a git checkout.

### Non-goals

Not per-file tracking queries — those stay unavailable in a container, which is
why the P2 predicate falls back to copy. Baking a tracked-file hash manifest is
explicitly **not** in scope (design open item 9).

### Allowed scope

- **Permitted (`lane/pipeline`):** `Dockerfile`.
- **Permitted (`lane/devmeta`):** `.gitignore`.
- **Forbidden:** `shared/provenance.py` (P1 owns the reader).

### Required changes (checklist)

1. `Dockerfile`: `ARG TOOLBOX_COMMIT`, written to `<repo_root>/.toolbox-commit`
   at image build.
2. Document the build invocation
   (`--build-arg TOOLBOX_COMMIT=$(git rev-parse HEAD)`) beside it — an unset ARG
   must yield an absent file, not an empty one.
3. `.gitignore`: `.toolbox-commit` — it must be absent in a normal checkout, so
   resolution step 1 (git) wins there.

### Validation

- Rung 1: `pytest tests/test_shared_provenance.py -k toolbox_identity` (P1's
  tests already cover the `baked` branch).
- Rung 4: `pixi run test-fast`.
- Docker build is **not** run as part of this phase's gate unless the owner asks
  — it is slow and not in CI. State plainly that it was not run.

**Falsifier for "a normal checkout never reads the baked file":** create a
`.toolbox-commit` containing a bogus sha in a working checkout and confirm
`toolbox_identity()` still returns `commit_source: git`. A `baked` result
disproves the resolution order.

### Acceptance criteria

`.toolbox-commit` is gitignored; resolution order is git → baked → nulls; the
Dockerfile change is two lines plus a comment.

### Task constraints

This phase spans two lanes. Land the `Dockerfile` edit in `lane/pipeline` and
the `.gitignore` edit in `lane/devmeta`, as separate commits — do not edit
`.gitignore` from the pipeline worktree.
