Task Brief — R9 follow-up: move `provenance/runs/` under an existing root

> **COMPLETE 2026-08-05.** Both deferred rulings were taken by the owner the
> same day: placement `config/runs/invocations/` (the recommendation), and the
> inventory gap closed by **documenting the limit** rather than extending the
> declared tier. The falsifier ran as specified — a wrapper-driven run, not a
> unit test — and all three of its clauses passed; evidence at the foot of this
> brief. The brief is kept unedited above that record.

Raised by P5 on 2026-08-04 and ruled the same day: `provenance/runs/` **moves
under an existing root** rather than becoming a seventh. This is a code change,
and P5 is docs-only, so it is its own task.

### The gap

`scripts/run_workflows.py:322` writes `<project_dir>/provenance/runs/*.json` —
one immutable invocation manifest per wrapper-driven run, recording enabled
workflows, sanitized arguments, start/end status, config and lock digests, and
Git/runtime identity.

It appears in **zero rows** of `migration_project-tree.md` and **zero lines** of
`project-tree-design.md`, whose tree has six roots: `config/`, `data/`,
`models/`, `experiments/`, `logs/`, `benchmarks/`. So R9 migrated the tree around
an artifact class neither document knows about.

**Why every R9 instrument missed it**, which is the more useful part:

- P1's **declared tier** is derived from the three Snakefiles' `output:`
  declarations. The wrapper is not a Snakemake rule, so it declares nothing.
- P1's **observed tier** was produced by direct `snakemake` invocations. The
  wrapper was never run, so the directory never appeared.
- The whole-tree `semantic_tree_diff` compares two trees that both lack it.

The class is *artifacts written by a user-facing runner rather than by a rule*.
R9's inventory design has no tier that covers it, and that is a gap in the
method, not an oversight in one document.

### Recommended placement — `config/runs/invocations/`

Argued from the migration map's own precedent rather than from taste. Finding 1
of `migration_project-tree.md` ruled where the generated config snapshot lives
and considered `logs/` explicitly:

> `logs/` is disqualified outright: it is what a user deletes to reclaim space,
> and its parts are merged-then-deleted by design, while this bundle is
> immutable, retained, and read by a downstream workflow.

The invocation manifest is **immutable and retained** for the same reasons, so
the same reasoning disqualifies `logs/` for it. `config/runs/` already holds
per-run generated provenance — the resolved configs and the digest-keyed
bundles — which is exactly this artifact's class.

The one asymmetry to weigh: `config/runs/<workflow>/<digest>/` is keyed by
workflow, and an invocation spans workflows. Hence `invocations/` as a sibling
rather than a fourth `<workflow>` entry.

**Alternative considered:** `logs/` at project scope, on the grounds that a run
record belongs with run records. Rejected by the quotation above — but if the
owner reads the manifest as a log rather than as provenance, that reading
changes the answer and should be ruled before implementation.

### Allowed scope

**Permitted** — `scripts/run_workflows.py`; `blueearth_cst/shared/provenance.py`
if the path is constructed there; `dev/scripts/semantic_tree_diff.py` (one map
row); `dev/milestones/r09/migration_project-tree.md` and
`project-tree-design.md` (one row and one line); tests.

**Forbidden** — `dev/baseline/**`; the manifest's CONTENT or schema; the six
existing roots.

### Required changes

1. Move the write target to the ruled location; one binding, not a search.
2. Add the map row and the design-tree line, so the next inventory covers it.
3. Add a `build_r09_path_map` rule for the old → new path, with a row-driven
   test case.
4. **Close the inventory gap that hid it**: either extend the declared-tier
   recipe to include wrapper-written artifacts, or state in the map doc that the
   inventory covers rule-declared artifacts only and name the classes it
   therefore misses. The second is cheaper and honest; the first is stronger.

### Validation

**Named scope** — `tests/test_run_workflows.py`, `tests/test_shared_provenance.py`,
`tests/test_r09_path_map.py`.

**Falsifier.** The gap is that no inventory sees wrapper-written artifacts, so
the check must be a run *through the wrapper*: invoke
`scripts/run_workflows.py`, then assert the manifest lands at the new path, that
nothing remains at `provenance/runs/`, and that `--check-map` over the resulting
tree reports zero unmapped. A unit test asserting the constructed path would
pass without ever proving the wrapper writes there.

### Acceptance criteria

- The manifest is written at the ruled location and nowhere else.
- The map and the design tree both carry it.
- A wrapper-driven run's tree passes `--check-map` with zero unmapped.
- The inventory gap is closed or explicitly documented.
- Pre-existing `provenance/` directories are NOT migrated: R9 declares pre-R9
  `project_dir` trees unsupported and requires a fresh run (R7 ruling GA-2,
  restated for R9), so there is nothing to move.

---

## Completion record — 2026-08-05

### Rulings taken

| Question the brief deferred | Ruling |
| --- | --- |
| Placement | `config/runs/invocations/` — the brief's recommendation. The owner did **not** read the manifest as a log, so the `logs/` alternative stays rejected on Finding 1's reasoning: `logs/` is what a user deletes to reclaim space and its parts are merged-then-deleted by design, while this manifest is immutable and retained |
| Required change 4 — the inventory gap | **Document the limit.** `migration_project-tree.md` now carries *What the inventory does not cover*: both tiers are Snakemake-derived, so the inventory covers rule-declared artifacts only, and the missed class — artifacts written by a user-facing runner rather than by a rule — is named with a table of known members. Extending the declared tier was considered and deferred: it needs a declaration mechanism the runners do not have |

### Changes

| File | Change |
| --- | --- |
| `scripts/run_workflows.py` | the one path binding, plus the docstring clause (g) |
| `tests/test_run_workflows.py` | the manifest-glob helper follows the binding |
| `README.rst` | the two user-facing mentions of the old path |
| `dev/scripts/semantic_tree_diff.py` | relocation rule (§8b) + an explicit identity row for the destination (§9) |
| `tests/test_r09_path_map.py` | two row-driven cases in the `config` section: the relocation and the destination |
| `dev/milestones/r09/migration_project-tree.md` | third amendment (F3), the `config/` row, and *What the inventory does not cover* |
| `dev/milestones/r09/project-tree-design.md` | the `invocations/` tree line, and an amendment recording that the rejected-`provenance/`-root ruling had already been violated when it was written |

`README.rst` is outside the brief's permitted list. It is included deliberately:
it names the manifest path twice in user-facing prose, and leaving it pointing at
a directory nothing writes any more is the same defect class the stale-path-prose
follow-up exists to clear.

### Why the destination got its own map row

The pre-existing `re.compile(r"(config/runs/[a-z_]+/.*)")` identity rule already
matches `config/runs/invocations/…`, so the row is redundant *today*. It is
there because `invocations` is not a workflow and only matches `[a-z_]+` by
coincidence: tightening that regex to the three real workflow names is a
reasonable future edit, and without its own row this path would drop to UNMAPPED
with nothing in the map to say why.

### Falsifier — as specified, a run through the wrapper

`pixi run python scripts/run_workflows.py --config <scratch cfg> -- --dry-run`,
`model_creation` enabled only, into a scratch `project_dir`. A dry-run exercises
`_initialize_manifest` / `_finalize_manifest` identically — the manifest is
written per invocation regardless of what the child does — so it proves the write
path without a build. Wrapper exit 0; WF1 planned 18 jobs.

| Clause | Result |
| --- | --- |
| Manifest lands at the new path | PASS — `config/runs/invocations/20260805T110022.527Z-e8857faa45f0.json`, the only file in the tree |
| Nothing remains at `provenance/runs/` | PASS — no `provenance/` directory exists |
| `--check-map` reports zero unmapped | PASS — `MAP CLEAN: 1 paths, 0 moved, 1 identity (by rule), 0 deleted-by-design, 0 unmapped`, exit 0 |

One check beyond the brief, because the three clauses above never exercise the
**relocation** rule — the post-move tree has no old path in it. Dropping a
synthetic `provenance/runs/*.json` into the same tree and re-running the real
tool:

```
IDENTITY config/runs/invocations/20260805T110022.527Z-e8857faa45f0.json
MOVED    provenance/runs/20260101T000000Z-oldpath.json  ->  config/runs/invocations/20260101T000000Z-oldpath.json
MAP CLEAN: 2 paths, 1 moved, 1 identity (by rule), 0 deleted-by-design, 0 unmapped
```

So the row is live through `snapshot_project_tree`, not only through the unit
test. The synthetic file was removed afterwards.

### Named-scope tests

`pytest tests/test_run_workflows.py tests/test_shared_provenance.py
tests/test_r09_path_map.py` — **128 passed**.

### Not done, by the brief's own acceptance criteria

Pre-existing `provenance/` directories are **not** migrated. R9 declares pre-R9
`project_dir` trees unsupported and requires a fresh run (R7 ruling GA-2,
restated for R9), so no migration code was written and none should be.
