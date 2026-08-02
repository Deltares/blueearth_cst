# R07 — Project layout — design-review intake

**Status.** Driver-authored intake for the `r07-project-layout`
design-review-loop run, 2026-07-27. Normalized from an existing change request
and its provenance chain; no `design-scoping` dialogue was run (see § Why no
scoping dialogue).

**Genre mapping.** `decision-record` — a milestone/refactor design (goal,
what-changes, commit plan, alternatives) maps to the nearest
`design-document` genre per `run-artifacts.md` § `status.md` manifest. The
design doc already declares this genre itself.

## Change request (verbatim)

> Lets process R07 project layout design (folder tree structure and
> organization of project files) through design-review-loop

Target: `dev/milestones/r07/project-layout-design.md` (DRAFT 2026-07-26, 414 lines).

## Provenance

The design under review was **authored interactively with the owner** across
the 2026-07-26 layout review — not by a stage-1 author spawn. It carries a
sixteen-ruling question log with rationale, two logged reversals (the §9 bbox
recommendation, reversed after the P4 ruling; the `stress_test/` provenance
argument, withdrawn on `Qstats.csv` evidence), and one superseded first-draft
principle (P3). Chain:

- `dev/reviews/2026-07-25_post-r6-assessment.md` — observation register
  O-01 … O-24, repository-side, with a routing note for which observations R7
  owns.
- `dev/milestones/r07/2026-07-26_project-output-layout.md` — the working note the
  design supersedes; artifact-side items, question log, cost analysis.
- `dev/roadmap.md` § "Phase 4 — Layout consolidation" — the milestone entry,
  exit criteria, and tag reservation (`r07-layout`).
- Prior accepted layouts this design revises: `dev/milestones/p31/experiment-structure-design.md`
  §2 (`project_dir` tree) and `dev/milestones/r06/structural-refactor-design.md`
  (repository tree).

## Problem

R6 closed the repository restructure and P3-1/P3-2a closed the experiment
restructure. Each left residue the other could not see, and R6's own lock list
deferred the artifact tree explicitly. The residue spans both halves of the
system and is one class of problem:

- Basin-specific data in the repository root (`data/`), contradicting the rule
  that a run writes outside the repository tree.
- Generated artifacts in the repository root (`dag/`, a stray `dag_model.png`).
- The test fixture named `examples/` while the real examples live in
  `docs/notebooks/`.
- Inside `project_dir`, three different figure conventions (centralized for
  wf1, distributed for wf2, root-dumped for wf3).
- Two directories holding the same extracted climate grid, with a shipped
  `allclose` check proving it.

Cheapest to fix at once: almost every item moves a path that
`dev/baseline/manifest.json` fingerprints, so batching buys a single baseline
re-record instead of two.

## Constraints

- **Behaviour-preserving, not re-record-free.** No computational path changes.
  17 of 18 baseline targets move path; 4 also change content. The manifest is
  re-recorded **exactly once**, at the end, after the phase-B gate.
- **`AGENTS.md` Hard Constraints bind**, in particular *stay within CST's
  automation scope*: hydromt / hydromt_wflow / Wflow conventions are consumed
  verbatim, never re-engineered.
- **The `"None"` sentinel.** Unquoted `None` in the configs parses to the Python
  string `"None"`, not YAML `null`; `null` raises `TypeError` downstream. Every
  `None` written must stay byte-identical.
- **`hydrology_model/` remains the hydromt `model_root`** (design option A) — no
  nested `model/` subfolder.
- No change to a computed value, a Wflow physics parameter, or hydromt
  internals.
- Forbidden to touch: `pixi.lock`, `Manifest.toml`, `Project.toml`, vendored
  upstream packages, `.pixi/`.

## Decision criteria

- **Four stated principles** govern, rather than accretion: P1 figures attach to
  what they depict (no project-level `plots/`); P2 one producer *definition* per
  artifact; P3 engine-shaped artifacts live inside their engine's subtree, every
  engine subtree sharing `config/ output/ plots/ _work/` except where an upstream
  tool owns the directory contract; P4 a full climate analysis must be possible
  with no wflow setup or run.
- A reader can tell what produced a file from where it sits.
- ~~A second modelling engine can be added without inventing a new layout.~~
  **Withdrawn 2026-07-28** — see the amendment note below.
- Engine-shaped artifacts are separable from generic ones, so an engine's subtree
  can be relocated, rebuilt, or replaced without moving generic climate data.
- Migration cost is bounded and the baseline consequence is paid once.

### Amendment note — 2026-07-28 (owner-ruled at the second G1 return)

The internal panel produced three findings that changed criteria approved at the
first G1, so they went back to the owner before external review rather than
riding to G2. All three were accepted; this section is amended to match the
design rather than leaving `intake.md` and `design-v2.md` in conflict.

- **P1 reworded** (risk-4, major): "attach to their producer" gave
  `basin_area.png` no home — rule 1.12 produces it from `staticgeoms/outlets.geojson`
  and it depicts the model, not its evaluation, so v1 filed it in a topic bucket.
  "Attach to what they depict" decides the case.
- **P2 reworded** (repo-2, blocking; implied by ruling GA-1): the owner-selected
  fix declares one producer in both Snakefiles, which v1's "computed twice by two
  workflows" wording forbade. The hazard P2 exists to prevent is two *definitions*
  disagreeing about content, not two declarations of one definition. The G1-return
  option table stated this restatement as the route's precondition, so the ruling
  carried it.
- **Extensibility criterion withdrawn** (arch-8, major): the delivered tree cannot
  honour "a second modelling engine can be added without inventing a new layout" —
  hydrology appears twice, in two shapes, at two levels, and a second hydrology
  engine collides on the domain-descriptive name `hydrology_model/`. Writing the
  placement rule that would honour it requires deciding the engine-naming question
  (OQ-1) the owner parked at the first G1. Replaced by the separability criterion
  above; the gap is recorded in the design as a stated limitation, and the
  structural half of the naming question defers with the naming half.

Also ruled at the same return: the commit count moves **13 → 15**, with the delta
named (arch-9's machinery-first split, plus B9 and B10 — two items v1 drew in the
tree but assigned to no commit). Content scope is unchanged; ruling GA-1's
"scope unchanged" is read as content scope.

## Success criteria

Mirrors the roadmap's R7 exit criteria:

- Design accepted; the implementation lands as `r07:` commits off a task brief,
  each leaving the tree runnable.
- All three Snakefiles `--dry-run` clean; `pytest tests/` green; CI baselines
  unmoved.
- A full three-workflow run on the seed config completes.
- Full-`project_dir` `semantic_tree_diff` against the R7 path map clean modulo a
  written, justified MISSING/EXTRA allowlist; all values identical.
- The **P4 assertion** demonstrated: climate figures produced with no
  `hydrology_model/` present.
- Manifest re-recorded exactly once; `check_baseline` green.

## Non-goals

**Of the milestone.**

- Tooling-contract decisions O-14 (`pyproject.toml`), O-15 (`ruff`), O-16
  (`flit`) — open, unrelated to layout.
- Docker (O-06) and Linux end-to-end (O-18, O-19) — parked, no Linux machine.
- Promoting climate analysis to a fourth Snakefile — separate milestone; R7 only
  ensures the layout does not obstruct it.
- Engine-named subtrees (`models/wflow/`) — parked, explicitly non-gating.

**Of this review run.** Three artifacts derive from the design and go stale the
moment review changes it. They are **out of scope for every author spawn** and
are regenerated after G2, not reviewed here:

| Artifact | Post-G2 action |
|---|---|
| `dev/milestones/r07/project-layout-task-brief.md` | Regenerate from the accepted version via `task-brief` |
| `dev/milestones/r07/migration_project-layout.md` | Mechanical derivation of the accepted path map; regenerate at stage 7 |
| `dev/roadmap.md` § R7 "design DRAFT 2026-07-26" | Update status line at stage 7 |

Author spawns must also **not** edit `dev/milestones/r07/project-layout-design.md` in
place — the version series lives in the run dir; stage 7 lands it.

## Why no scoping dialogue

`design-scoping` fires at stage 0 only for new-direction or underspecified
work. Neither holds: the change request names an existing, fully-specified
414-line design whose scope was settled through sixteen recorded owner rulings.
The intake normalizes that record rather than re-eliciting it.

## Open questions carried to G1

The design's § "Risks and open questions" leaves four open. One is parked; the
other three are selected-layout content, so leaving them open risks an external
reviewer forcing the choice and bouncing the run back to G1 after a spent
round. They are put to the owner at G1:

1. **Engine-named subtrees** (`models/wflow/` vs `hydrology_model/`) — parked,
   non-gating. Not put to the owner.
2. **`MIGRATION.md`'s home** (O-12) — `docs/`, `dev/milestones/r06/`, or root with a stated
   `naming.md` §7 exemption.
3. **`blueearth_cst.Rproj`** (O-13) — delete, or move beside the R sources.
   Factual: depends on whether the owner uses it.
4. **Where the weathergen date CSVs settle** — `weather_generator/output/` as
   designed, or `_work/` since they are diagnostics rather than products.
