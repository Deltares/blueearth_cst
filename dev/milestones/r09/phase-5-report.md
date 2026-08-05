# R9 P5 report — conventions and documentation

Date: 2026-08-05. Branch: `docs/r09-p5-conventions-docs`, cut from
`milestone/r09-project-tree` after P4. Brief:
[`phase-5-conventions-docs-task-brief.md`](phase-5-conventions-docs-task-brief.md).

**Status: complete.** The last phase, so it also carries the program's single
full-suite run — which found a P4 escape no earlier gate could have seen.

---

## What landed

| # | Commit | Note |
| --- | --- | --- |
| 1 | `give naming.md a generated-outputs rule and fix two AGENTS.md defects` | checklist 1–4, 6, 7, 8, 10 |
| 2 | `correct the design's region.geojson claim; brief the provenance move` | two owner rulings |
| 3 | `finish the seam repoint the previous commit claimed, and the README DAG` | the overclaim finding, F1 |
| 4 | `seal the WF3 baseline, migrate WF1's contract, close two followups` | scope extension, ruled; the P4 escape |

Two follow-up briefs written rather than implemented:
[`followup-provenance-root-task-brief.md`](followup-provenance-root-task-brief.md)
and
[`followup-stale-path-prose-task-brief.md`](followup-stale-path-prose-task-brief.md).

## `naming.md` by section

| § | Change |
| --- | --- |
| **§6** | Tier 2's "user-facing output / config names: `Qstats`, `Tlow`, `Tpeak`" is now config keys and table labels only. P3 renamed the file; the labels inside it still rely on the tier. |
| **§7** | The scientific-abbreviation carve-out narrows from *user-facing output filenames* to config keys and column/row labels — **narrowed to where it is load-bearing, not repealed**. Records that R9's rename note is `migration_project-tree.md` and that its scope is exactly two files. |
| **§8** | "Owning workflow contract (R01) — **varies**" replaced with the real rule: lowercase `snake_case` for locally minted names under `project_dir`, two exemptions (upstream-owned names, embedded tier-1 identifiers). "Varies" was not a rule — it let each workflow answer differently, which is how the tree drifted from the convention. Explicitly **class-scoped**: read as a repo-wide sweep it would rename `dev/` markdown and Python modules the same guide protects. |
| **§9** | The numbering claim was **false** and is corrected with evidence. It said `NN` is a "step in definition order" and that inserting a rule "renumbers the contiguous comments below it". The code has never done that — gaps at 1.14 / 2.05 / 3.12; WF2 defines 2.03b, 2.03, 2.01, 2.02 out of order; P4 inserted 3.01c/d/e renumbering nothing. `NN` is a **stable identifier assigned at creation**; a letter suffix is how you insert. Renumbering would be actively harmful: the number appears in `LOG_RULES`, in log and benchmark paths and in prose, and an unlisted `LOG_RULES` label drops its log section **silently** — which happened three times in R9 alone. |
| **§4** | The rule list carried `export_wflow_results`; P3 renamed it `derive_wflow_indicators`. |

## The grep falsifier

Claim: *no current documentation still describes the old tree*. A read-through
cannot establish that, so the instrument is a grep with every surviving hit
justified.

**The brief's term list over-matched and was refined.** `RT_` hit
`SHORT_DIGEST_CHARS`, `HISTORICAL_START_YEAR` and `_MAX_VERT_EXAG` — three
constants with nothing to do with return-period tables. Refined to
`\bRT_[0-9A-Za-z]`. Per checklist item 11 the list also gained
`export_wflow_results`, `basin.csv`, `config/dag/` and `indicators/`.

**133 files** match. Every one falls in a class below.

### Justified classes

The brief named three. Seven were needed — and the extra four are not
bookkeeping, they are places where a naive sweep would have done damage.

| Class | Why old paths are correct there |
| --- | --- |
| **Sealed milestone records** — `dev/milestones/**` | Records of what was done. The brief's glob `dev/milestones/r0[1-8]/` **undercounts**: `p31/`, `p32a/`, `p32b/`, `p33/`, `phase-1/` and `r10/` are equally sealed and match none of it. |
| **R9's own reports and briefs** | They discuss the old tree deliberately and at length; `phase-1-report.md` is about little else. |
| **The comparator's source side** — `build_r09_path_map`, both inventories, `test_r09_path_map.py`, `test_semantic_tree_diff.py` | A rename map that did not name the old path would map nothing. |
| **ADRs** — `dev/decisions/000*.md` | Decision records, dated. |
| **Dated review records** — `dev/reviews/2026-07-*.md` | Same. |
| **Prior-release migration guides** — `docs/migration-r06.md`, `docs/migration-r08-wf2.md` | They exist to describe the tree a user is coming *from*. |
| **`config/runs/<workflow>/` — current and correct** | `Snakefile_climate_projections:61` and `wf2_climate_projections_overview.md:122` carry `config/runs/climate_projections/<digest>/`. The snapshot directory is keyed by **workflow name**, so the term `climate_projections/` matches a path that is right. Naming this class explicitly is the point: without it, the next sweep "fixes" a correct path. |
| **Rename records** — `naming.md` §6/§7, `check_baseline.py:185`, `plot_workflow_dag.py:16`, closed `followups.md` entries | "X was renamed to Y" must name X. |
| **Task briefs** — `dev/working/wf1-spatial-decoupling/*.md` | Records of what was asked at the time. |

### Real hits — what was fixed

| Where | Was | Now |
| --- | --- | --- |
| `AGENTS.md` DAG path | `config/dag/` | `logs/dag/` + `experiments/<id>/logs/dag/` |
| `AGENTS.md` pixi env | "a worktree resolves to the primary's copy" | each worktree builds its own; `weathergenr` needs `pixi run install` |
| `hydrological-model-seam.md` | ten pre-P2 member paths, `indicators/`, the old rule name | the declared shapes (see F1) |
| `weather-generator-seam.md` | five member paths, the loose catalog path | flattened shapes, `config/catalogs/` |
| `naming.md` | §4 rule name, §6 tier-2 example | current |
| `README.rst` | `--dag` piped to `test_case/test_local/dag/`, "create `<project_dir>/dag/`" | `scripts/plot_workflow_dag.py` in all three blocks |
| `model_creation.md` | `hydrology_model/`, `spatial/`, `climate_historical/wflow_data/` | `models/hydrology/wflow/`, `data/spatial/`, `{basin_dir}/forcing/` |
| `project-tree-design.md` | `region.geojson` "exists only as…" (×2) | corrected, and drawn in the tree |
| `followups.md` | `[R7-8]`, `[R7-15]` reading as open | both closed with what R9 delivered |

### Real hits — deferred, with a brief

Owner-ruled 2026-08-05. `blueearth_cst/**` prose (≈15 modules) and a 30-line
`Snakefile_climate_experiment` layout block; `dev/scripts/scaffold_project_tree.py`
+ `scaffold_extras.yml`; and three more `dev/scripts/` files. All in
[`followup-stale-path-prose-task-brief.md`](followup-stale-path-prose-task-brief.md).

Two of those deserve naming here:

- **`scaffold_project_tree.py` is functional, not prose** — it stages a fixture
  at a path P2 deleted, so it would fail if run. Verified **inert**: it is not
  what produced `declared_inventory.txt` (that recipe is three raw
  `snakemake --summary` calls) and nothing references it, so P1's evidence chain
  is untouched. Verified independently rather than inherited from P4's adjacent
  "stale staging was inert" verdict, which was about a different file.
- **`prune_climate_store.py` is R9's own.** `STORE_ROOT = "data/climate/historical"`
  is correct; three docstrings around it still say `climate_historical/`. P1
  shipped the tool, P2 repointed the constant, and nobody repointed the prose
  describing the constant.

### Not fixed, by judgment

`docs/notebooks/*.ipynb` reference `examples/myModel/hydrology_model/…`.
Patching the R9 half would leave `examples/`, retired at **R7**, still wrong —
a still-broken notebook that now looks maintained. Reported as one class: pre-R9
debt needing its own pass. The acceptance criterion is that hits are *accounted
for*, not that all are fixed.

## Findings

### F1 — a sweep claimed complete on the edits made, not on the instrument re-run

**The fourth instance of one pattern in R9, and mine.**

Commit `52856ac`'s message says the two seam contracts were "repointed at the
v10 tree, **including the member-path flattening**." They were not. Ten passages
still carried the pre-P2 shape, and the flattening is precisely what survived: I
edited the parts I had read and wrote the message from intent.

| # | Where | The claim |
| --- | --- | --- |
| 1 | P2 commit 1 | `grep … \| head -20`, then "every model-internal path is built from `basin_dir`, verified". Three sites sorted past the cut; one failed the run. |
| 2 | P2 → P3 | `indicators_dir` → `results_dir` replaced across one file; the consumer's `sm.params.indicators_dir` lived in another. |
| 3 | P3 | the same rename pattern, again. |
| 4 | P5 commit 1 | this one. |

The class: **a sweep is declared complete on the strength of the edits made
rather than on a re-run of the instrument that found the work.** The fix is
mechanical — re-run the finder before writing the message — and it is the one
habit R9 has failed to acquire four times.

What did **not** catch it: `tests/test_interchange_contracts.py` passes 37 tests
against these documents' subject matter, because it pins artifacts, not prose.
For documentation the grep is the only instrument, exactly as the brief said.

Two corrections beyond path strings, both of which would have misled a reader
reasoning from cause:

- HM-4 explained wflow pointer mechanics as "R07 B5 moved the run TOMLs one
  level deeper, so every pointer gains one `../`". The depth changed at P2; the
  `../` is now the `config/` → `output/` **sibling** hop, riding in the pointers
  because `dir_output` stays `"."`.
- HM-7's producer is `derive_wflow_indicators` since P3, but the **module** is
  still `export_wflow_results.py` — so the `export_wflow_results.py:61` citations
  are current. Stated inline, since the mismatch otherwise reads as a miss.

### F2 — two of the three "current workflow contracts" were sealed records

The brief's checklist item 5 and my own triage both treated
`dev/reference/workflows/{model_creation,climate_experiment,climate_projections}.md`
as live contracts describing the old tree. That premise was wrong for two of
them, and I found out the expensive way.

`climate_projections.md` opens with a **SUPERSEDED banner** — *"sealed
2026-07-31 … kept unedited because it is the baseline the R4 commits were
checked against; rewriting it would destroy the record it exists to be."* I had
migrated the whole document before seeing it, because I grepped the file rather
than reading its head. Fully reverted.

`climate_experiment.md` is the same kind of document — *"R5's opening act,
written before any code change… the baseline the R5 code commits are checked
against"* — and had **no banner**. So it read as a live WF3 contract while being
four milestones stale in paths, rule names, `src/` module locations and every
Snakefile line number. Owner-ruled: revert and seal. It now carries the WF2 twin
banner naming what superseded it and where current truth lives.

**The lesson is about the failure mode, not the two files.** Migrating
`climate_experiment.md`'s paths would have left its line numbers and module
paths lying just as loudly while making the document *look* freshly maintained —
strictly worse than leaving it obviously old. The hazard was that it looked
current; a banner addresses that and a path sweep does not.

Two process consequences:

1. **Read a document's head before editing it.** A grep with a path pattern
   cannot see a seal banner, because a seal banner does not contain paths.
2. **The seal is a convention with one instance and no enforcement.** Only WF2's
   doc had a banner; the identical WF3 doc went four milestones without one.
   Worth an `AGENTS.md` line or a test, but that is not P5's scope.

`model_creation.md` has no such framing, is a live contract, and was migrated.

### F3 — a P4 escape, caught by the gate assigned to P5

`pixi run test-full` returned **1 failed, 1311 passed** on first run:
`test_guard_invalidation.py::test_2c_fresh_project_missing_wf1_snapshot`.

Not a P5 regression. P4's rule 3.01c `write_model_reference` is the **first WF3
rule ever to declare model files as inputs** (P4's own finding F1), so
`models/hydrology/wflow/{wflow_sbm.toml,.outputs_configured}` joined the wf1
config snapshot as leaves `--unlock` needs on disk. Gate 2c(iii) asserts that
with *every* leaf present `--unlock` succeeds; the leaf set grew and this
fixture staged only the two config snapshots. P4 updated `test_cli.py`'s fixture
for exactly this reason and missed this one.

**P4's gate could not have caught it.** The module is marked `workflow_contract`;
`pixi run test-fast` runs `-m "not workflow_contract and not process_isolation"`,
so it collects **0 of 2** tests here. P4's green test-fast was truthful about
what it ran. The full gate the program deliberately assigned to its last phase is
what surfaced this, which is the argument for having assigned it.

Fixed by staging the two leaves; the assertions are untouched. The fixture
docstring now says why "every leaf" is load-bearing — a forgotten leaf turns
2c(iii) into a failure that reads like a guard defect and is not one.

Related, and left as debt rather than swept in: this is the **third** copy of
cross-workflow staging logic (`test_cli.py`, this fixture,
`scaffold_project_tree.py`), and the scaffold copy is currently stale in the same
way. Whether they should share one helper is raised in the follow-up brief.

### F4 — a seventh project root nothing in R9 could see

`scripts/run_workflows.py:322` writes `<project_dir>/provenance/runs/*.json` —
an immutable per-invocation manifest — in **zero rows** of the migration map and
**zero lines** of a six-root design tree.

Why every R9 instrument missed it is the reusable part: the **declared tier**
comes from Snakefile `output:` declarations and the wrapper is not a rule; the
**observed tier** came from direct `snakemake` invocations so the wrapper never
ran; and the whole-tree diff compares two trees that both lack it. The class is
*artifacts written by a user-facing runner rather than by a rule*, and R9's
inventory design has no tier that covers it.

Owner-ruled to move under an existing root; briefed with
`config/runs/invocations/` recommended, argued from the map's **own** Finding 1
precedent (`logs/` was disqualified for the config snapshot because it is what a
user deletes, while the bundle is immutable and retained — the same is true
here), and with the alternative reading that would change the answer recorded so
it can be overruled on the point that decides it.

### F5 — `project-tree-design.md` asserted a file does not exist where it does

It claimed `region.geojson` "exists only as
`models/hydrology/wflow/staticgeoms/region.geojson` and the store's
`store_region.geojson`". It also exists at `data/spatial/geoms/region.geojson`,
written by rule `delineate_region` (ADR 0003), and is in both inventory tiers and
the built tree.

The error was **counting one rule's outputs as the whole subtree** when a
different rule writes another into it — the root of P1's F1a, where the map
enumerated five geoms layers against six emitted. The map was amended at P1; the
design, which is the more authoritative document, was not until now.

Corrected in **both** places it appeared. The second was found only because the
first edit's assertion aborted before writing — a blind replace would have left
it, which is F1's pattern surfacing a fifth time in a form that got caught.

## Validation

| Rung | Command | Result |
| --- | --- | --- |
| 1 Narrow | the grep falsifier, per edit | all 133 matching files classed |
| 1 Narrow | `tests/test_interchange_contracts.py` | 37 passed, 26 skipped |
| 1 Narrow | `tests/test_guard_invalidation.py` | 2 passed |
| 2 Integration | `pixi run test-cli` | 12 passed — **at commit 1**; superseded by rung 3, which subsumes it and ran against the final tree |
| 3 **Full gate** | `pixi run test-full` | **first run: 1 failed** (F3), 1311 passed, 31 skipped, 1 xfailed, 5m04s |
| 3 **Full gate, re-run** | `pixi run test-full` | **1312 passed**, 31 skipped, 1 xfailed, 4m26s — **green** |

The count rises by one because F3's fix turned a failing test into a passing
one; nothing was added or skipped to get there.

The rung-2 caveat is deliberate. Naming test-cli as this phase's validation when
it ran three commits earlier would be F1's pattern — a check claimed on intent
rather than on a re-run — in the report that names the class. It is recorded as
what it is, and the full gate is what actually covers the final tree.

Sealed records under `dev/milestones/**` are unmodified — checked, not assumed.

## Carried forward

- **The concurrency falsifier has still never been shown to FAIL** with
  `path_log` unset. The cheap half (distinct pointers per member) is unit tested;
  content attribution under a real concurrent batch needs a run. Now recorded in
  `followups.md` under `[R7-8]` rather than only in phase reports.
- The **end-to-end model-drift falsifier** in a real run (from P4).
- WF2's nondeterministic fetch provenance (P2 F4); rule 1.04's undeclared write
  to `staticmaps.nc` (P2 F5's root cause).
- The **seal convention** has one banner and no enforcement (F2).
- Three copies of cross-workflow staging logic, one of them stale (F3).
- Two follow-up briefs: the provenance root, and the stale-path prose sweep.
