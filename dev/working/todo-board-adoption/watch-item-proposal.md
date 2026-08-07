# Gate 1 draft — a `watch-item` type for `todo-board`

*Drafted 2026-08-07. **Proposal only — nothing applied.** Target artifact:
`~/workspace/brain/artifacts/skills/todo-board/` (canonical; never a vendored
copy). Releases Phase 1 of `task-brief.md` in this folder.*

## Why

`todo-board` has exactly one item type, and its admission gate says work failing
the gate "creates no board item" without saying where it goes. A repo with a
standing set of *known but not scheduled* facts therefore has two bad options:
put them on the board as items that never advance, or keep a second backlog file.

blueearth_cst took the second and it failed measurably: `dev/followups.md` grew
to 2,038 lines with a parallel `R<n>-<n>` ID scheme cited from 14 code sites,
hand-synced against `dev/TODO.md` — which drifted (a fixed item still listed
`backlog`) and gave single items two IDs. Of its 39 open entries, ~13 fail the
admission gate **on purpose**: parked rulings, knowingly-accepted debt, upstream
defects a local workaround already covers.

`status: blocked` does not cover these. The skill requires a blocked item to say
*what must clear first*; a parked ruling is not blocked, it is **decided**.

## What changes

Additive and backward compatible: notes already carry `type: todo-item`, so no
existing note migrates and no view breaks.

### 1. Frontmatter description (trigger surface)

Add a trigger clause after "…setting backlog, active, or blocked state":

> …recording a known issue or accepted trade-off that needs visibility but no
> action;

### 2. New bullet in `## The model`, after **Note-per-item**

> - **Two item types.** `type:` in frontmatter says what a note is for.
>   - **`todo-item`** (default) — work intended to be done. Overview carries
>     **What / Why / Effort**; `## Progress` required; participates in `queue`,
>     `next`, and `backlog`/`active`/`blocked`.
>   - **`watch-item`** — something true about the system that needs durable
>     visibility but no action: a knowingly-accepted trade-off, a parked ruling,
>     an upstream defect a local workaround already covers. Overview carries
>     **What / Why / Trigger**; no `## Progress`, no `queue`, no `status`, and
>     `next` never returns one.
>
>   **Trigger replaces Effort, and it is what keeps the type honest.** It names
>   the observable condition that would turn this into work — "hydromt ships a
>   fixed `to_yml`", "a Linux runner exists", "the owner rules on X". If no such
>   condition can be written, it is not a watch-item: it is either work (make it
>   a `todo-item`) or nothing (delete it — the diff and Git history carry it).
>
>   A watch-item ends two ways: its trigger fires and it is **converted** to a
>   `todo-item` under the same ID; or it stops being true and
>   `done --outcome superseded` files it in `LOG.md`.

### 3. Admission gate — route, don't only exclude

Append to the existing bullet:

> **Failing the gate does not mean untracked.** Route it: durable knowledge
> nobody intends to act on becomes a `watch-item`; only genuinely ephemeral
> detail becomes nothing. A project with no such route grows a second backlog
> file beside the board and then hand-syncs the two — the failure this type
> exists to prevent.

### 4. Views

- **Obsidian** — add a **Watch** view beside Active / Backlog / Blocked,
  filtered `type == "watch-item"`, columns `ID · Item · Area · Trigger`. The
  existing three keep their `type == "todo-item"` filter, so "what's next" stays
  uncontaminated.
- **Non-Obsidian** — `render` emits a second table, **Watching**, below the open
  table, same four columns. One file to read, two questions answered separately.

### 5. Task-lifecycle cadence

Add one sentence: watch-item triggers are re-checked at a **milestone close or
periodic review**, not per session — `todoboard list --type watch-item`. Without
a named review point, watch-items accumulate unread, which is this design's main
risk.

### 6. Verification signal

> - **Types stay separated.** `next` never returns a `watch-item`; every
>   `watch-item` names an observable Trigger; no `watch-item` carries
>   `## Progress`, `queue`, or `status`.

### 7. `HISTORY.md` + version

One row, `v0.11.0` (minor — additive, backward compatible).

## This is not doc-only — the CLI changes too

Sizing it honestly, because the task brief listed it as one open decision:

| Verb | Change |
|---|---|
| `add` | `--type watch-item` scaffolds a different body: Trigger instead of Effort, no `## Progress`, no `queue`, no `status` |
| `next` | must skip `watch-item` (today it skips only `blocked`) |
| `list` | add `--type` |
| `render` | emit the second **Watching** table |
| `done` | unchanged — `--outcome superseded` already exists |

Plus tests per verb. `python -m pytest todoboard/` must stay green and
`grep -rn "brain" todoboard/` must stay empty (the vendorability check).

## The alternative I rejected, and the risk I accept

**Alternative: a separate `dev/notes/` store with its own index.** Cleaner
separation of concerns — a task board stays purely about work. Rejected because
it reproduces exactly what blueearth_cst already has (two stores, two ID
schemes, hand-synced) and the owner's stated goal is *one* scheme. The `type:`
field and the `.base` filter already exist precisely to let one store hold more
than one kind of note.

**Risk accepted: the board becomes a dumping ground.** Mitigated by the Trigger
requirement (no writable trigger → not a watch-item) and by §5's named review
point. If watch-items still accumulate unread after a milestone or two, that is
evidence for the rejected alternative, not for loosening the Trigger rule.

## Open questions for the owner

1. **Name.** `watch-item` vs `known-issue` vs `observation`. `watch-item` reads
   as "something to watch", which fits a trigger; `known-issue` is narrower than
   the set (a parked ruling is not an issue).
2. **Does a watch-item take `area:`?** Proposed yes — it is how the Watch view
   groups, and blueearth_cst's set spans upstream / platform / test-hygiene.
3. **Version.** `v0.11.0` assumed. Brain's release policy (ADR 0020) makes the
   bump an owner-approved gate, so confirm before it lands.

## Applying this

- Canonical artifact only — a vendored pin is overwritten on resync.
- **Verify in a NEW session.** Skill content is snapshotted at session start, so
  a same-session subagent reads the pre-edit file and reports guidance that no
  longer exists.
- Brain runs `mode: lite` (trunk only, auto-commit verified work), so this lands
  on brain's trunk once the CLI tests are green.
