# Intake — wf2-climate-analysis-v2

## Change request (verbatim)

> I would like to do some scripting changes for increased efficiency and
> modularity in WF2.

> If you were to redesign WF2, how would you do? Consider scientific
> methodology, computational efficiency, and user-friendliness

> Yes, lets turn this into a design document but one remark: while WF2 is
> currently on interested in delta factors, I would like to develop it further
> into a handy climate data & projections analysis workflow. So lets keep this
> in mind while we are desiging WF2_V2.0

> Before that, lets send this design document to Fable & GPT 5.6 SOL so that
> they both individually review intention and the design document and provide
> input. We can then finalize the design

## Problem

WF2 (`Snakefile_climate_projections`) computes CMIP6 change factors and is
structured against its own cost profile: it fans out where computation is free
and serializes where the network is expensive, marks every intermediate
`temp()` so a re-run re-downloads the archive, discards the monthly seasonality
it computes, and cannot run without a built hydrological model. The owner wants
it to become a general climate data and projections analysis workflow.

## Constraints (standing, from `AGENTS.md` + sealed milestones)

- WF2 is a plausibility overlay; WF3 is never driven by CMIP6 scenarios.
- `climate_store_spec` declarations must stay symmetric across Snakefiles.
- hydromt / hydromt_wflow / Wflow conventions are consumed verbatim.
- No new dependency without explicit owner approval.
- `check_baseline.py` on `test_case/test_local` is a local-only gate.

## Decision criteria

Recorded in the design doc §4: tiered value-neutrality; cost follows the
network; sealed acceptance gates are not reopened; explicit beats inferred; the
extension surface is designed once.

## Success criteria

An owner-accepted design that (a) is implementable in commit-sized units with a
value-neutral prefix, (b) widens WF2 to observed + projected climate analysis
without reopening P3-2a, and (c) resolves or explicitly defers OQ-1..OQ-8.

## Non-goals

Implementation (a task brief for migration steps 1–2 already exists at
`dev/working/2026-07-29_wf2-v2-decouple-and-cache.md`); a 4th Snakefile entry
point; new dependencies; bias correction or downscaling.

## Genre mapping

`workflow-spec` per `design-document`. The doc also carries decision-record
content (named decisions D1, alternatives, consequences); recorded here rather
than splitting the artifact.

## Derived-artifact register

What depends on this design, and what regenerates after G2:

| Artifact | Relationship |
|---|---|
| `dev/working/2026-07-29_wf2-v2-decouple-and-cache.md` | Task brief for steps 1–2; **must be re-checked** if the review changes step ordering or the value-neutrality classification |
| `dev/workflows/wf2_climate_projections_overview.md` | Current-state map; unaffected by the design, but its findings feed it |
| `dev/workflows/climate_projections.md` | Current behavioral contract; §5.4 of the design records a contract/code discrepancy that lands here at implementation |
| `dev/roadmap.md` | OQ-2 (Phase 4 or close) would add a section |
