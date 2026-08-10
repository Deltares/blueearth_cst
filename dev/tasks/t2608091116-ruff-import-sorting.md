---
title: Adopt ruff's I rule family — import sorting — in two stages, like ADR 0005
type: todo-item
status: backlog
effort: 1
area: formatting
origin: R7
queue:
created: 2026-08-09
updated: 2026-08-09
---

> [!note] Overview
> **What** — Add `I` to `[tool.ruff.lint] select`, run `ruff check --select I
> --fix`, and let the existing `pixi run lint` gate hold it.
> **Why** — It is the last rule family O-15 deferred that has a clear answer.
> The churn is now measured and small, and ADR 0005 already built the machinery
> for landing exactly this shape of change.
> **Effort** — Small, and `--fix` does the edit. The scheduling is the work.

## Progress

**HALF LANDED 2026-08-09 (`b124a98`). The 42 files outside `blueearth_cst/` are
sorted, and `I` is armed in `select`** — with the deferred half carved out as
`"blueearth_cst/**" = ["I001"]`, the same shape `[tool.ruff.format]` uses.

That is a deviation from this note's original plan, which said to arm `I` only
once both halves were clean. Arming it early with an explicit carve-out is
strictly better: it stops the 42 drifting back while the 20 wait, and waiting
was the failure mode this item exists to prevent. What remains is deleting one
line from `pyproject.toml` and running `--fix` on 20 files.

*Not blocked on a decision — nobody has objected to sorted imports. The
remainder is queued behind ADR 0005 stage 2 because the two share a cost.*

**Measured on `main` at `b46ccc8` (2026-08-09): 62 files, all auto-fixable.**

| Scope | Files |
|---|---|
| `tests/` | 34 |
| **`blueearth_cst/`** | **20** |
| `dev/` | 7 |
| `scripts/` | 1 |

**The 20 are the point.** They are the Snakemake `script:` layer, so sorting
their imports fires the `code` rerun trigger on every `project_dir` exactly as
ADR 0005 stage 2 does — and paying that invalidation twice, once for formatting
and once for imports, is pure waste. Land the 42 non-`script:` files whenever;
land the 20 **in the same sitting as stage 2**, ideally as the commit
immediately before or after it, so one full re-run absorbs both.

- [x] Land `I` for `tests/`, `dev/`, `scripts/` (42 files) — `b124a98`
- [x] Add `I` to `select`, carving out `blueearth_cst/**` rather than waiting
- [ ] Land `I` for `blueearth_cst/` (20 files) beside `t2608090907a` stage 2
- [ ] Delete the `"blueearth_cst/**" = ["I001"]` per-file-ignore in the same
      commit — it is the only thing left holding the rule back

## Refs

- `dev/decisions/0005-adopt-ruff-format-in-two-stages.md` — the precedent. Its
  Consequences already park this: "Import sorting (`I001`, 63 findings) stays
  out of scope. It is a separate rule family, a separate reviewable commit."
  This item is that commit, plus the observation that the split ADR 0005 chose
  applies here for the same reason.
- `dev/tasks/t2608090907a-adr0005-stage2.md` — the sitting to land the 20 in.
- `pyproject.toml` `[tool.ruff.lint]` — the `select` list, and O-15's reasoning
  for pinning it explicitly rather than inheriting ruff's ~415-rule default.
- Still deferred, no measurement taken: `UP` (pyupgrade), `B`, `SIM`, `PERF`,
  `RUF`. Those are taste, not correctness, and this item is not a precedent for
  adopting them.
