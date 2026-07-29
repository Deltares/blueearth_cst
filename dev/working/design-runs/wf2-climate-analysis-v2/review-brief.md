# External review brief — WF2 v2.0 climate data & projections analysis, round 1

## Role

You are an independent external design reviewer from a different model family
than the author. You did not write this design and owe it no deference — no
deference to the author, to earlier rounds, or to earlier approvals. Your value
is adversarial pressure: challenge framing, feasibility, and completeness. Do
not copyedit prose.

## Task

Review exactly one document:

- `C:\Users\taner\workspace\blueearth_cst\dev\working\design-runs\wf2-climate-analysis-v2\design-v1.md`

The repository is **BlueEarth Climate Stress Test** — a Snakemake-orchestrated
scientific toolbox (Python + R + Julia) with three workflow entry points run in
order: workflow 1 builds a distributed Wflow-SBM hydrological model from global
datasets via hydromt; workflow 2 (the subject of this design) computes CMIP6
change factors as a *plausibility overlay*; workflow 3 is the bottom-up climate
stress test, where a stochastic weather generator produces realizations
perturbed across a temperature × precipitation grid and run through Wflow. The
design under review proposes restructuring workflow 2 and widening it from a
change-factor calculator into a general climate data and projections analysis
workflow. It is a **draft**: the owner has asked for review before finalizing.

**Settled framing — out of scope for your review.**

1. Workflow 2 is a plausibility overlay only. Workflow 3 is never driven by
   CMIP6 scenarios (repository hard constraint). The design's §5.8 slot S5 is a
   one-way advisory that emits a figure and a warning and never writes workflow
   3 config; do not argue for a tighter coupling.
2. `plot_climate_source` (workflow 1's rule 1.15) stays where it is and workflow
   2 composes rather than relocates it (§6.2). A sealed milestone's test
   (`tests/test_plot_climate_source.py`) pins the assertion those figures build
   without a hydrological model on disk; relocating invalidates it.
3. No new third-party dependency is adopted in this design. `xclim`,
   `regionmask`, `intake-esm` are recorded as owner asks (OQ-7), not as
   proposals.
4. hydromt / hydromt_wflow / Wflow conventions are consumed verbatim; the design
   does not re-engineer upstream behavior.
5. Two facts were measured, not assumed, and you may rely on them: (a) the two
   candidate region polygons have identical bounds `[9.658333, 0.35, 9.858333,
   0.483333]` on the test fixture; (b) the CMIP6 catalog sources are `Amon`
   (monthly), so the current `resample("MS").sum()` vs `.mean()` dispatch is a
   no-op.

Do not spend findings arguing these should have been decided differently; **do**
raise a finding if a ruling creates a downstream inconsistency in the document,
or if the document's implementation of a ruling does not actually satisfy it.

**Explicitly open — input wanted.** §10 records OQ-1 through OQ-8 as unresolved
(entry point, roadmap placement, output layout, window length, extremes data,
ensemble weighting, dependencies, `save_grids`). These are *not* settled framing;
substantive input on them is welcome and should be filed as findings at whatever
severity you judge.

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
    doc_version: design-v1.md

    ## Findings
    ### ext1-<seq>  [blocking | major | minor]
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
