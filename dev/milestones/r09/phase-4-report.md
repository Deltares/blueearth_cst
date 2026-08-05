# R9 P4 report — model fingerprint and experiment lifecycle

Date: 2026-08-04. Branch: `feat/r09-p4-fingerprint`, cut from
`milestone/r09-project-tree` after P3. Brief:
[`phase-4-fingerprint-task-brief.md`](phase-4-fingerprint-task-brief.md).

**Status: complete.** Commits 1–3 (the model fingerprint) landed first; commit 4
(experiment lifecycle) stopped mid-phase because its checklist diverged from the
code in four ways, was re-specified against reality in
[`phase-4-commit-4-task-brief.md`](phase-4-commit-4-task-brief.md), ruled by the
owner, and then implemented. *Commit 4: why it stopped* is kept below as the
record of that pause.

---

## What landed

| # | Commit | Note |
| --- | --- | --- |
| 1 | `r09: add the pointer-derived model digest` | pure, no caller — testable in isolation |
| 2 | `r09: write model_reference.yml per experiment` | the reference exists before anything reads it |
| — | `fix(wf2): add 2.03b_delineate_region to LOG_RULES` | pre-existing; third instance of one defect |
| 3 | `r09: fail WF3 on model drift before simulating` | the guard, plus the defect class closed mechanically |
| 4a | `r09: reject colliding experiment names and version generated ones` | scope exception for `scripts/`, owner-ruled |
| 4b | `r09: write experiment.yml and freeze it after the first successful run` | |
| — | `fix(r09): keep the suggest-name tests out of the working tree` | my defect; caught by the phase gate |

## The digest

Pointer-derived, not a fixed file list. Seed model's file set, computed in
**13 ms**:

| Path | State |
| --- | --- |
| `wflow_sbm.toml` | hashed |
| `staticmaps.nc` | hashed |
| `forcing/inmaps_historical.nc` | hashed |
| `instate/instates.nc` | `<absent>` |
| `output.csv` | `<absent>` |
| `outstate/outstates.nc` | `<absent>` |

**Cost caveat:** 13 ms is the 384-cell fixture. The cost is dominated by
hashing `staticmaps.nc`, so it scales with basin size; quoting 13 ms as if it
were universal would be wrong.

Two design decisions, both visible above:

- **`dir_output` is deliberately not applied.** Wflow resolves output pointers
  against `dirname(toml) + dir_output`, so the three output keys resolve to
  paths that normally do not exist and are marked absent. Intended: the
  fingerprint stays **stable across runs** instead of moving every time the
  historical run rewrites its own outputs. Applying `dir_output` would break
  every experiment's reference after any WF1 re-run.
- **The exclusions are structural, not a blocklist.** `staticgeoms/`,
  `hydromt.log` and `hydromt_data.yml` stay out because nothing points at them.
  If a future TOML did point at one, Wflow would read it, it would be a runtime
  input, and it *should* enter the digest — a hardcoded blocklist would be wrong
  in exactly that case.

The absent warm state independently corroborates P2's F2: `instate/` does not
exist in a built tree.

## Falsifiers

| Falsifier | Result |
| --- | --- |
| **Pointer discovery** — a new TOML key brings a file in; editing that file's content alone moves the digest | **pass** |
| …**shown to fail against a fixed-file-list implementation** | **pass** — the rejected implementation is built alongside in the test and shown blind to the same edit |
| **Exclusions hold** — `staticgeoms/`, `hydromt.log`, `hydromt_data.yml` do not move the digest | pass |
| **Optional state** — presence vs absence of `instate/instates.nc` differ, via the marker | pass |
| …and "target missing" ≠ "no such key" | pass |
| **Containment** — a pointer escaping the model root raises | pass |
| **Determinism** — same model at two absolute locations agrees; entries sorted | pass |
| **End to end** — an old experiment fails after the model changes; a new one succeeds | pass **at unit level**; not yet demonstrated in a real run |
| **Resume is not a collision** | pass — a resume allocates nothing |
| **User-supplied duplicate rejected**, naming the existing experiment | pass; never silently versioned |
| **Generated collision → `_v2` then `_v3`** | pass — the third case is the discriminator; gaps are filled, not counted |
| **Reservation is atomic** | pass — **demonstrated by racing** eight threads for one name: one winner, seven clean errors |
| **Immutability, both directions** | pass — writable *before* the first successful run, refused *after* |

The pointer-discovery falsifier carries a second assertion: if the fixed-list
stand-in ever *starts* catching the edit, the test fails. Otherwise "discovery
works" would be asserted against nothing.

## Ordering is the guard

A check that runs after the work is a post-mortem. So rule 3.09 — the first rule
that touches the model — declares the guard's sentinel as an **input**, and a
test parses the Snakefile to pin that edge. Without it the guard could run after
the members had simulated and every other test in the module would still pass.
That is the brief's rollback condition, so it is the one property made
mechanical rather than described.

**The `ancient()` is load-bearing.** Rule 3.01c declares its model inputs
`ancient()`, so a rebuilt model does not re-trigger the writer. If the reference
were rewritten whenever the model changed, it would always match and the guard
would be decorative. A test pins it.

Rule 3.00b is untouched — its inputs and sentinel paths are approval-gated, so
the new guard uses its own sentinel, with a test confirming 3.00b was not
co-opted.

## Findings

### F1 — WF3 had no declared dependency on the model at all

Rule 3.01c is the **first WF3 rule to declare model files as inputs**. Until now
WF3 reached the model only through `params`, so the DAG could not see that WF3
depends on it. Making the edge real is the point — P2's F5 was a rule reading an
undeclared file and being ordered correctly by luck until a cold `-c 3` run
scheduled it badly.

The cost: WF3's dry-run now needs `wflow_sbm.toml` and `.outputs_configured`
staged, as it already needed the WF1 config snapshot. Updating that fixture also
revealed its staged region was still at `hydrology_model/` — a path P2 deleted.
It passed anyway because ADR 0003 lets WF3 produce the region itself, so the
stale staging was inert.

### F2 — P1's F7 was one instance of a class, not a bug

`delineate_region` is splatted into all three workflows from **one** producer
contract, but each workflow owns its own `log:` label and its own `LOG_RULES` —
so registering the label is a **per-file obligation the shared definition does
not carry**, and nothing checked it.

| Workflow | Label | Fixed |
| --- | --- | --- |
| WF1 | `1.01b_delineate_region` | P1 (F7) |
| WF3 | `3.01b_delineate_region` | P4, commit 2 |
| WF2 | `2.03b_delineate_region` | P4, own commit (out of scope, owner-ruled) |

All three had been silently dropping their section and stranding their part on
every run since ADR 0003 landed.

**The class is now closed mechanically**: a test parses every Snakefile and
asserts each declared log-part label appears in that file's `LOG_RULES`. Its
scope is stated in the test rather than overclaimed — it matches the form every
current declaration uses, counts 14/3/14 labels today, and would not see a label
assembled from a variable.

Found because an automated `LOG_RULES` insert printed `False` and I went looking
rather than assuming the edit had landed.


### F3 — adding a side effect invalidated an existing command's test fixtures

**My defect, caught by `pixi run test-fast` and not by the narrow scope.**

Reservation gave `suggest_experiment_name.py` a filesystem side effect it never
had: it now creates `experiments/<id>/` to claim the name. The existing tests
point `project_dir` at the repo-relative `examples/Gabon`, so running them wrote
real directories into the working tree and accumulated them across runs — the
failure read `gabon_20260728_v5`, four versions deep, having resurrected
`examples/`, a tree retired at R7.

I ran those two test files *before* wiring reservation into the runner and read
that green as covering the change. Only the full gate exercised the runner end
to end.

The lesson generalises past the fix: **a command that only read the filesystem
now writes to it, and every existing test of it was written under the old
assumption.** Their fixtures were invalidated even though their assertions still
looked right.

**Known trade-off, stated rather than left to be discovered:** reservation
creates the directory, so an abandoned setup leaves an empty experiment dir that
will collide next time. Inherent to reserving up front; the alternative is a
window in which two sessions own one name.
## Commit 4: why it stopped

The checklist diverges from the code in four ways, each verified:

| Brief | Code |
| --- | --- |
| default is `stress_test_<YYYYMMDD>`, suffixed `_v2`/`_v3` | default is project-basename + date (`test_local_20260804`); `stress_test_` appears nowhere in the codebase |
| `experiment.yml` becomes immutable at first successful run | **nothing writes `experiment.yml`** |
| "reservation must be atomic" | there is **no reservation step**; `suggest_experiment_name` proposes a string and the directory appears when a rule first writes into it |
| scope is `blueearth_cst/experiment/**` + a new shared module | the creation path is `scripts/suggest_experiment_name.py`, outside permitted scope |

Delivering it would have meant choosing a default-name shape, inventing
`experiment.yml`'s contents, designing an atomicity mechanism, and widening
scope — four owner decisions. The brief's own stance on an analogous case is to
stop and report, and its commit plan made the lifecycle rules separately
revertible precisely so this could stop here.

**Resolution.** The brief's items 5–6 were superseded by
[`phase-4-commit-4-task-brief.md`](phase-4-commit-4-task-brief.md), which leads
with a verified ground-truth table and leaves the four decisions to the owner
rather than settling them. All four were ruled on 2026-08-04 — keep the existing
`<project-basename>_<YYYYMMDD>` default and suffix only generated names;
`experiment.yml` is the generated experiment section; atomicity is scoped to one
machine (`os.mkdir`); and `scripts/suggest_experiment_name.py` is in scope by
exception — and commit 4 was then implemented against those rulings.

## Validation

| Rung | Command | Result |
| --- | --- | --- |
| 1 Narrow | `tests/test_model_digest.py`, `tests/test_model_reference.py` | 28 passed |
| 2 Integration | `pixi run test-cli` | 12 passed — three clean dry-runs |
| 3 Phase gate | `pixi run test-fast` | **1271 passed**, 30 skipped, 42 deselected, 1 xfailed — and the working tree is clean of stray directories afterwards, which is the property F3 is about |

`ruff` clean on the new modules. `check_baseline` is green (P3 re-recorded it);
this phase changes no baseline-pinned artifact.

## Carried forward

- The **end-to-end drift falsifier** in a real run, not just at unit level.
- From P2: the concurrency falsifier has still not been shown to **fail** with
  `path_log` unset; WF2's nondeterministic fetch provenance (F4); rule 1.04's
  undeclared write to `staticmaps.nc` (F5's root cause); `AGENTS.md`'s incorrect
  shared-env claim (F6).
- **P5's queue**: `AGENTS.md`'s stale DAG-render path and shared-env claim, and
  the design tree's silence on `spatial/geoms/region.geojson`.
