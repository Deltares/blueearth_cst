# Process observations — p32a-climate-analysis design run

Process-friction log per design-review-loop operating rule. Process notes
only, never design content. Feeds the post-run retrospective.

- 2026-07-24 — stage 3 revision spawn died on a stream idle timeout (API
  transport, not content) after ~350 s / 15 tool uses. Skeleton-first held:
  design-v2.md was on disk as a v1 copy, so nothing was lost. Escalation
  ladder rung 1 taken: same-thread resume with "stop reading, write the
  deliverables now." Confirms the p31 evidence for the skeleton-first rule.
- 2026-07-24 — run opened. Stage 0 was near-null by construction: the
  authoritative intake (`dev/milestones/p32a/climate-analysis-intake.md`) was produced
  and user-confirmed in a prior session via `design-scoping`, then landed and
  pushed before the loop started. Run-dir `intake.md` is a verbatim copy; no
  friction. Note for the skill: the "intake lands with the design at
  finalize" checklist item is already satisfied before the run — finalize
  should detect this and not double-land.

- 2026-07-24 — external rounds: both GPT reviews reduced to the verdict schema
  first-try (no re-dispatch needed); both faulted prior-round fix resolutions,
  so the Fable escalation rule fired on both revision spawns (r2 and 6a) —
  matching the r04/p31 pattern that the external-reviewer-faults-the-fix locus
  is where Opus revisions run out of road.
- 2026-07-24 — round cap → arbitration → stage 6a worked as specified: three
  AskUserQuestion rulings (with fix-variant choices embedded as options)
  translated cleanly into a confined 6a brief, and the heading-based diff
  scope-check passed first try. Offering the fix VARIANTS as arbitration
  options (tolerance vs direct-hop; exclude vs verify) made the rulings
  actionable without a second round-trip.
- 2026-07-24 — finalize (lean path): landing edits were pure citation
  repoints + status swap; the pre-landed intake meant no intake landing step.
  One wrinkle: the v4 header cited run-dir artifacts (ledger, external
  reviews) that the retention policy prunes — the landing repoint to the
  consolidated review record is what keeps the landed doc dangling-free.
  Worth a skill note: the author should cite run artifacts knowing they will
  be repointed at landing.
