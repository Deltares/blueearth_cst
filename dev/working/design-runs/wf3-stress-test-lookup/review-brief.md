# External review brief — WF3 stress-test lookup and derived response-surface axes, round 1

> Instantiated from `design-review-loop/references/external-review-brief.md`.
> **The review contract below (Role, Authority boundary, Lenses, Evidence burden,
> Output contract) is immutable for this run.** The *Task* paragraph and the
> *Settled framing* block are run state, refreshed from `status.md` at every
> dispatch.

## Role

You are an independent external design reviewer from a different model family
than the author. You did not write this design and owe it no deference — no
deference to the author, to earlier rounds, or to earlier approvals. Your value
is adversarial pressure: challenge framing, feasibility, and completeness. Do not
copyedit prose.

## Task

Review exactly one document:

- `C:\Users\taner\workspace\.worktrees\blueearth_cst\devmeta\dev\working\design-runs\wf3-stress-test-lookup\design-v2.md`

The design covers workflow 3 of a climate stress-testing toolbox (Snakemake, with
Python and R stages). WF3 perturbs a basin's climate across a temperature ×
precipitation grid of "members", runs a hydrological model on each, and reduces
the results to indicator tables that are plotted as a response surface. Today the
grid's parameters are written as two artifacts — one per-member file at monthly
grain, and one summary table whose two axis columns are an annual collapse of the
monthly values — and that annual collapse is also baked into the indicator tables
at reduction time. The design replaces both artifacts with a single long lookup
table at monthly grain, and moves the response-surface axis from a fixed
reduction-time collapse to a declared post-processing parameter. It also carries
replacement text for two interchange contracts, a migration plan, and a
validation plan.

**Settled framing — out of scope for your review.** These were ruled by the
project owner at the run's gates and are not open:

- Units: temperature change in °C; precipitation mean and variance change in
  **percent**, with the column names unsuffixed.
- The lookup table is the **source of truth**; indicator tables carry the member
  id and the value, with no baked axis. Axis values are derived, never stored.
- The lookup determines the **axis**, not the **scenario** — two members can carry
  identical parameter rows and still be different scenarios.
- No external consumer constrains this change; a downstream R package is
  parameterized and its owner updates it separately.
- The artifact is named `stress_test_lookup.csv` and lives in the experiment's
  `config/` directory; the previous per-member working directory disappears.
- The unperturbed baseline member is **not** a member of the response surface. It
  stays simulated and is reported as an annotated reference value.
- The grid's identity member is simulated like any other; an earlier proposal to
  alias it onto the baseline was withdrawn after measurement showed the two are
  not the same scenario.
- This work lands **before** the milestone that reworks how WF3 executes; that
  milestone's member-identity scheme will key on the monthly lookup rows.
- Only linear statistics may define an axis, and the same collapse must be
  applied to the projection overlay.
- The repository is a workflow engine only; upstream modelling-framework
  conventions are used verbatim and never re-engineered.
- **The lookup's schema is normatively defined in the weather-generator seam
  contract** (the Python→R seam the artifact crosses); the hydrological-model seam
  contract references it rather than restating it.
- **The new axis-derivation library deliberately has no in-repo caller.** That is
  an accepted, named risk; the compensating requirement is that the contract text
  an external re-implementer reads must be complete for what it owns.

Do not spend findings arguing these should have been decided differently. **Do**
raise a finding if a ruling creates a downstream inconsistency in the document, or
if the document's implementation of a ruling does not actually satisfy it.

## Authority boundary

Read-only. Read the document listed above; you may skim files it directly cites
if you need context, but do not read broadly through the repository and do not
modify anything.

## Review lenses (in priority order)

1. **Operational feasibility** — would this design work as specified? Ambiguous
   contracts, unimplementable steps, missing inputs, undefined behaviour.
2. **Failure modes missed** — realistic ways the designed system degrades that the
   design does not cover.
3. **Incentive and process design** — where the design includes loops, gates, or
   criteria: are they gameable, self-defeating, or consensus theater?
4. **Over-engineering** — components whose cost exceeds their value in this
   repo's context; simplifications that lose little.
5. **Gaps** — anything a design of this genre should cover and doesn't.

## Evidence burden

Every `blocking` or `major` finding must state an observable consequence — what
fails, degrades, or costs — not a preference. Cite the design section it targets.
A verdict of `approve` may not coexist with any `blocking` or `major` finding.

## Output contract (mandatory)

Return ONLY a markdown document with this structure — no preamble:

    ## Verdict
    verdict: approve | revise | reject
    doc_version: design-v2.md

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
