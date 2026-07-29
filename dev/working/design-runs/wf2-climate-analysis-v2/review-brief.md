# External review brief — WF2 v2.0 GCM projections analysis, round 2

## Role

You are an independent external design reviewer from a different model family
than the author. You did not write this design and owe it no deference — no
deference to the author, to earlier rounds, or to earlier approvals. Your value
is adversarial pressure: challenge framing, feasibility, and completeness. Do
not copyedit prose.

## Task

Review exactly one document:

- `C:\Users\taner\workspace\blueearth_cst\dev\working\design-runs\wf2-climate-analysis-v2\design-v3.md`

The repository is **BlueEarth Climate Stress Test** — a Snakemake-orchestrated
scientific toolbox (Python + R + Julia) with three workflow entry points run in
order: workflow 1 builds a distributed Wflow-SBM hydrological model from global
datasets via hydromt; workflow 2 (the subject of this design) computes CMIP6
change factors as a *plausibility overlay*; workflow 3 is the bottom-up climate
stress test, where a stochastic weather generator produces realizations
perturbed across a temperature × precipitation grid and run through Wflow. The
design under review proposes restructuring workflow 2 into a **monthly GCM
projections analysis** workflow with an optional gridded branch (ruling R4
below narrowed an earlier, broader "climate data & projections analysis"
ambition). It is a **draft**: the owner has asked for review before finalizing.

**Settled framing - out of scope for your review.** Every item below is an
owner ruling recorded in `status.md`. Round 1 reviewed an earlier version; these
rulings were made in response to it.

1. Workflow 2 is a plausibility overlay. Workflow 3 is never driven by CMIP6
   scenarios (repository hard constraint).
2. `plot_climate_source` (workflow 1 rule 1.15) stays where it is; workflow 2
   composes rather than relocates it. A sealed milestone's test pins it.
3. No new third-party dependency is adopted.
4. hydromt / hydromt_wflow / Wflow conventions are consumed verbatim.
5. **R1 - clip, never splice.** The GCM reference window clips to the CMIP6
   historical experiment (ends 2014); the 2015-2020 gap is never filled from
   scenario data. The workflow warns when a configured period overruns 2014.
   The overlay reference period and the project's stress-test baseline remain
   different periods, and the owner accepts that.
6. **R2 - the gridded option is retained, default off**, with declared outputs.
7. **R3' - no aggregation at any level.** Members are never averaged. Each
   (model, scenario, member) is a single data point with its own dT and dP. No
   aggregation across models, scenarios, or members. Member availability is a
   union of what the store publishes; differing member counts and missing SSP
   scenarios are normal, not errors.
8. **R3'' - cross-combination statistics are ex-post**, computed downstream and
   out of v2.0 scope. Per-series statistics (computed within one
   (model, scenario, member, horizon)) remain in scope - they are what the
   change factor is.
9. **R4 - v2.0 is monthly GCM projections analysis**, with a gridded expansion.
   No comparison against observed data at this stage; `extract_historical.nc` is
   not reduced.
10. **R5 - the basin-averaged monthly series per run is a declared
    deliverable**, alongside the change-factor table and the composition record.
11. **D2 -> A1** - WF2 declares the full `climate_store_spec`, accepting the
    gridded observed extraction on a fresh projections-only run.
12. **OQ-4 -> 30 years, 1985-2014** for the reference window.

Do not spend findings arguing these should have been decided differently; **do**
raise a finding if a ruling creates a downstream inconsistency in the document,
or if the document's implementation of a ruling does not actually satisfy it.

Also read, **after** forming your own view of the design:

- `ledger.md` - dispositions of all 19 round-1 findings plus round-2 rows
- `review-index.md` - the round-1 aggregation, including driver premise checks

**Regression duty.** Verify that findings marked resolved are actually resolved
in this version, that no accepted fix introduced a new defect, and that
rejections' rationales hold. Re-raise anything that fails.

**Explicitly open - input wanted.** Section 10's remaining open questions
(including OQ-12 config-key naming, OQ-13 expressing many members without a
fan-out surprise, OQ-14 catalog snapshot cadence, OQ-15 guaranteed-variable
coverage) are *not* settled framing; substantive input is welcome, filed as
findings at whatever severity you judge.

One context fact the design relies on: `config/catalogs/cmip6_data.yml` is now
**generated** from a live `gs://cmip6` crawl - 289 entries, one per
(model, scenario), each listing the members that exist with both `pr` and `tas`
at Amon; 2426 sources total. It is never hand-edited.

## Authority boundary

Read-only. Read the document listed above; you may skim files it directly cites
if needed for context, but do not read broadly through the repository and do not
modify anything.

## Review lenses (in priority order)

1. **Operational feasibility** — would this design work as specified? Ambiguous
   contracts, unimplementable steps, missing inputs, undefined behavior.
2. **Failure modes missed** — realistic ways the designed system degrades that
   the design does not cover.
3. **Scientific methodology** — the change-factor method, window choices,
   spatial reduction, ensemble treatment, and what the resulting numbers can
   honestly support.
4. **Over-engineering** — components whose cost exceeds their value in this
   repo's context; simplifications that lose little.
5. **Gaps** — anything a design of this genre should cover and doesn't,
   particularly for the widened "climate data & projections analysis" ambition
   versus the narrower change-factor product it replaces.

## Evidence burden

Every `blocking` or `major` finding must state an observable consequence — what
fails, degrades, or costs — not a preference. Cite the design section it
targets. A verdict of `approve` may not coexist with any `blocking` or `major`
finding.

## Output contract (mandatory)

Return ONLY a markdown document with this structure — no preamble:

    ## Verdict
    verdict: approve | revise | reject
    doc_version: design-v3.md

    ## Findings
    ### ext2-<seq>  [blocking | major | minor]
    - section: <design heading the finding targets>
    - finding: <one-paragraph claim>
    - rationale: <why it matters — observable consequence>
    - suggested_fix: <concrete change, or "none">

Severity calibration: `blocking` = the design as specified would fail, produce
wrong results, or cannot be implemented; `major` = meaningful degradation, cost,
or risk with a clear fix; `minor` = worth noting, author's discretion. List
findings in severity order, blocking first. Aim for the findings that matter; do
not pad. If the design is sound, say so — an empty findings list with
`verdict: approve` is a valid review.
