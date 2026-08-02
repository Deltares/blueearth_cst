# Process observations — p31-experiment-structure

Skill/process friction log per design-review-loop SKILL.md operating rule.
Process notes only, never design content. User meta-objective for this run:
test and solidify the design-review-loop skill; note process, performance,
and logic improvement opportunities.

- (2026-07-23, stage 0) Intake handoff from `design-scoping` worked as
  designed: the scoping session had already written `intake.md` into the run
  dir the loop expects, so stage 0 reduced to creating `status.md` — no
  re-normalization needed. Positive signal for the scoping→loop seam.
- (2026-07-23, run start) Minor reference ambiguity: `run-artifacts.md` says
  `review-brief.md` is "instantiated from external-review-brief.md at run
  start; immutable for the run", while SKILL.md § References says to load
  `external-review-brief.md` "at the first external round". Driver chose:
  instantiate at the first external round (lazy), since nothing consumes the
  brief earlier. Candidate skill edit: align the two statements.
- (2026-07-23, stage 1) Spawn-transport failure, rung 1 taken: the stage-1
  author spawn (Opus, cst-architect) died on "API Error: Stream idle timeout —
  partial response received" after ~6 min / 20 tool uses, no `design-v1.md`
  on disk. Same failure mode as the r04 pilot's author-stream timeouts —
  recurring pattern for long read-heavy author spawns. Escalation ladder rung
  1 applied: same-thread resume with the explicit "stop reading, write the
  deliverable now" instruction. Rung 1 SUCCEEDED — full deliverable written
  on resume. Candidate skill note: for read-heavy author spawns, consider
  pre-authorizing the `[UNVERIFIED: ...]` marker convention in the initial
  brief (it was granted only at the resume) so a first spawn can trade
  residual verification for landing the deliverable before the stream idles.
- (2026-07-23, stage 2) Panel performance signal: three clean-room lenses on
  Opus each produced schema-conformant verdicts, zero re-dispatches, strong
  complementarity — risk found the blocking runtime break (chirps sidecar),
  architecture+repo-fit independently converged on the same core defect
  (guard not threaded into the DAG), near-zero finding overlap otherwise
  (2 duplicate minor pairs across 22 findings). Self-containment held on all
  three (no lens asked for missing context). The clean-room + verdict-schema
  + severity-consequence-rule combination is doing real work; no process
  friction to log for stage 2.
- (2026-07-23, stage 5/r1) Fable-escalation rule ambiguity: the rule fires
  when an external review "re-raises a finding a prior round already
  raised". Round-1 external findings all fault NEW mechanisms the v2
  revision introduced (sentinel freshness, comparator cross-root defect) —
  they are faults-in-the-fix, not literal re-raises of the panel's finding
  IDs. Driver interpreted narrowly (no Fable; r2 stays Opus). Candidate
  skill clarification: state whether (a) "prior round" includes the internal
  panel, and (b) a defect in the fix for finding X counts as a re-raise of
  X. Current wording lets both readings through.
- (2026-07-23, stage 6) SECOND author-spawn stream idle timeout (r2, Opus,
  ~8 min, 17 tool uses, no deliverable despite the progressive-write
  instruction — it died before starting to write). Pattern now 2-for-3 on
  author spawns in this run (draft + r2; r1 survived at 42 tool uses).
  Rung 1 same-thread resume applied again. Candidate skill edit is now
  stronger than the earlier note: author briefs should mandate
  write-the-skeleton-first (create the deliverable file with section stubs
  BEFORE the read phase), not just "write progressively" — a stub on disk
  turns every timeout into a resumable partial instead of a total loss.
- (2026-07-23, stage 4) codex dispatch mechanics worked first-try under the
  adapter's exact recipe (empty stdin, -c approval_policy=never, background,
  -o capture): no stream timeout, no hang, read-only held. The r04-pilot
  failure mode (foreground default timeout) did not recur. Adapter guidance
  is sufficient; no edit needed.
- (2026-07-23, stage 6a) Skeleton-first VALIDATED: the 6a spawn's mandated
  first action (copy v3 → v4 before any reading) plus 45 tool uses on
  Fable produced zero transport loss, vs 2 total-loss timeouts on
  read-first Opus author spawns. Confirms the candidate skill edit: author
  briefs should mandate deliverable-file-on-disk as the first action.
- (2026-07-23, stage 6a) Fable escalation earned its cost: the Fable spawn
  empirically REFUTED the v3 Opus author's probe design (which tested
  content-change on one path, missing the input-set identity trigger),
  re-probed the correct A→B shape, and replaced the mechanism. This is the
  exact failure locus the tiering rule predicts (Opus fix fails reviewer
  re-raise → Fable). Rule works as designed when the trigger is read to
  include faults-in-the-fix (see the stage-5 ambiguity note — this run
  suggests resolving the ambiguity toward the BROADER reading).
- (2026-07-23, G2 prep) Driver scope-check mechanics: mapping diff hunks to
  nearest markdown heading via a small PowerShell script worked well
  (32 hunks → 16 sections, exact match to the author's declared list).
  Candidate skill addition: give the driver a concrete recipe for the 6a
  scope-check (hunk→heading mapping) instead of leaving the method open.
- (2026-07-23, stage 7) G2 approved with ZERO editorial edits, making the
  stage-7 author spawn an identity transform. Driver landed mechanically
  (copy + ACCEPTED status header swap, logged as the sole editorial edit)
  instead of spending a ~150k-token spawn on a header change. Candidate
  skill edit: stage 7 should name this lean path explicitly ("with zero
  approved editorial edits and zero pending accepted-editorial minors, the
  driver lands mechanically; the status-header swap is a logged editorial
  edit, not design content").
- (2026-07-23, stage 7) Retention gap found in the R6 precedent: the landed
  R6 design's scope-authority pointer cites the PRUNED run-dir intake
  (dev/working/design-runs/r06-structural-refactor/intake.md — gone). Fixed
  for this run by landing the intake beside the design
  (experiment-structure-intake.md) and repointing. Candidate skill edit:
  stage 7 landing checklist should include "land the intake beside the
  design (or fold it in) — a pruned run dir must not leave dangling
  citations in the landed doc".
