ADR 0008 — Ship `blueearth_cst` unpackaged; `pyproject.toml` stays tool-config-only

Status: accepted
Date: 2026-08-17
Deciders: Ümit Taner
Consulted: —
Supersedes: none — this is O-14 **decision 2**, which O-14 decision 1 (`ab781a5`)
  deliberately left open rather than pre-empting. Nothing is reversed; the open
  half is closed.
Revisions:
  - 2026-08-17: initial record and **ACCEPTED** the same day. Raised from board
    item `t2608071209` (R7-16), which had been `blocked` since 2026-08-07 on
    exactly this ruling and named the superseding record as its own deliverable.

### Context

`pyproject.toml` in this repository carries **tool configuration only** — no
`[build-system]`, no `[project]`, no `[tool.pixi]`. That was O-14 decision 1,
landed in `ab781a5`, and it was explicitly a decision about *what the file is
for*, not about whether the package should ever be installable. The second half —
**real packaging** — was left open, and `dev/reviews/2026-07-25_post-r6-assessment.md`
recorded a consequence of leaving it open: O-16, `flit = ">=3.2"` declared in
`pixi.toml:16` with no build system for it to drive. Its disposition was written
as "drop it, or give it a job — depends on the O-14 build decision", so O-16
could not close while O-14 decision 2 stayed open.

Two things about this repository bear on the question, and both point the same
way.

**Nothing here is imported as a package.** `blueearth_cst/` is a tree of modules
invoked by Snakemake `script:` directives (Python) and `Rscript --vanilla`
`shell:` bodies (R). `AGENTS.md` states it plainly: *"none is a standalone CLI"*
and *"there is no package CLI"*. The four `*.smk` files at the repo root are the
only entry points. A `script:` module reads `snakemake.input/output/params`; it
is executed by path, never imported by name from outside the workflow.

**The environment already has an owner.** `pixi` resolves the whole polyglot
stack — conda-forge Python, the R toolchain, snakemake, plus `pixi run install`
for `weathergenr` via remotes and the Julia environment via `Pkg.instantiate`.
Adding a Python build backend would put a second, partial environment mechanism
beside a complete one, covering the smallest of the three languages involved.

The counter-case is real and was weighed: packaging buys a clean
`import blueearth_cst` for any out-of-repo consumer, plus versioned release
artifacts. What it does **not** buy is the in-repo import — `tests/` already
writes `from blueearth_cst.shared import snake_utils` and the like, resolved by
`pythonpath = ["."]` in the same tool-config-only `pyproject.toml`, with the 34
`sys.path.insert` shims that predated it already removed (`tests/conftest.py:8-11`).
So the usual first argument for packaging a source tree — "the tests have to
hack the path" — does not apply here; it was answered by O-14 decision 1.

### Decision

**`blueearth_cst` is not an installable Python package, and `pyproject.toml`
stays tool-config-only.** No `[build-system]`, no `[project]`, no build backend.

Consequently, **O-16 resolves to "drop it"**: `flit` is removed from `pixi.toml`,
because the job it was declared for is one this repository has now decided not to
have. That removal is the closing step of O-16 and requires a `pixi.lock`
regeneration through pixi — never a hand edit — because CI runs with
`locked: true` and a stale lock fails the gate on both platform legs.

This is a decision about *this* repository, not a position on packaging. The
platform's other two parts (CST-API, CST-frontend) are separate codebases and
this record says nothing about them.

### Consequences

- **O-16 closes**, and `t2608071209` closes with it. The tooling contract from
  R7 — O-14, O-15, O-16 — is complete for the first time.
- **`pythonpath = ["."]` becomes the permanent in-repo import mechanism**, not a
  stopgap until packaging arrives. It already gives `tests/` ordinary
  `from blueearth_cst...` imports; this record makes that the arrangement rather
  than an interim state. `dev/scripts/semantic_tree_diff.py` and
  `dev/scripts/cross_workflow_inputs.py` are reached the same way, and `AGENTS.md`
  already calls them contract surfaces with test consumers.
- **An out-of-repo Python consumer has no supported import path.** This is the
  cost, and it is the thing to watch: if one ever appears, it reopens this record
  rather than working around it with a `sys.path` shim in a foreign codebase. The
  known consumers (CST-API, the frontend, `csthelpers`) drive the Snakefiles
  server-side and re-implement from the seam contracts, which is why none of them
  needs an import today.
- **No release artifact exists other than a git checkout**, plus the Docker image,
  which `ADD`s sources individually and carries no `.git` — the degraded-identity
  case the config-snapshot design (`t2608131304`) already handles with
  `ARG TOOLBOX_COMMIT`. Packaging would have offered a third identity mechanism;
  declining it keeps the count at two.
- **Residual risk.** The argument above rests on "nothing imports it as a
  package", and that is a property of today's tree rather than a constraint
  anything enforces. Nothing fails if a module grows an out-of-repo importer; it
  would simply be discovered late. The mitigation is this record, read when
  someone next proposes an import path.

### Alternatives considered

- **Package with flit.** Rejected. It buys an import path nothing currently asks
  for, at the cost of the tool-config-only `pyproject.toml` that O-14 decision 1
  deliberately chose, and of a second environment mechanism beside pixi. If the
  decision is ever reversed, flit remains the natural backend — the file is
  already the right shape to grow `[build-system]` and `[project]`.
- **Defer again with a stated trigger** (convert `t2608071209` to a watch-item
  naming an out-of-repo importer as the trigger). Rejected as the worse form of
  the same answer: it leaves O-16 formally blocked and `flit` declared with
  nothing to build, which is the exact state the review flagged. Declining
  packaging *records* the trigger in Consequences above without keeping a board
  item open to hold it.

### References

- `dev/reviews/2026-07-25_post-r6-assessment.md` § O-16 — `flit` declared with
  nothing to build, and the disposition that made it depend on this decision.
- `dev/milestones/r07/project-layout-design.md:1412` — O-14/O-15/O-16 named as
  open decisions unrelated to layout.
- `ab781a5` — O-14 decision 1, the tool-config-only `pyproject.toml`.
- ADR 0005 — the other half of the tooling contract, `ruff format`; O-15 (ruff as
  the lint gate) is recorded there and in `85d3178`…`518151b`.
- `AGENTS.md` § Repo Map, § Key Commands — the "no package CLI" statement and
  pixi's role as the single environment owner.
