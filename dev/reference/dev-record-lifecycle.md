# Development documentation lifecycle

Last updated: 2026-10-30

This reference owns the boundary between current guidance, proposed changes, reusable templates and retained
evidence. File placement and scratch cleanup are detailed in [repository layout](dev-repository-layout.md);
task coordination is in development workflow (private-checkout-only citation:
`private/dev-task-and-session-workflow.md`).

## Current guidance

`dev/reference/` is the single maintained home for development practices and toolbox architecture.
Each topic has one owner document, shared by developers and agents. Elsewhere, use a pointer and a short
explanation of when to read it. `AGENTS.md` routes agents and carries essential execution constraints;
`dev/README.md` provides orientation.

Choose the existing topic from the reference index before writing. Update that document for improvements
and revisions; do not create a new file per decision, date, task, review round or implementation change.
A new topic file is justified only when it answers a distinct reader question with its own maintenance
boundary. Add it to the owning area's index and remove any overlapping account in the same change.

Write the applicable rule or design, its scope, brief rationale where useful, and pointers to implementation
and verification. Keep exact machine-owned defaults, schemas and inventories in their owning code or
configuration. Update the relevant section in place; remove superseded wording. Git preserves versions.
Do not append dated updates, old/new accounts, revision logs or a new ADR for every change.

Use a lightweight format: a title, `Last updated: YYYY-MM-DD`, a short scope statement, and the sections
the topic needs for current rules/design, rationale and implementation/verification pointers.
The date means the latest substantive content revision, not a verification certificate. Do not change it
for relocation, formatting or index maintenance alone. When introducing the field, use the latest relevant
Git revision date if no more precise content date is available.

Guidance describes the effective policy and implementation of its checkout. Main carries the integrated
account; session changes to code and guidance merge together. Do not share reference files through the
board's cross-worktree links.

## Proposed changes and promotion

Unfinished designs and implementation plans belong with their live task on main. Acceptance of a design
does not establish runtime implementation. Promote a design to current guidance when its behavior or
policy takes effect. If a requirement is effective before implementation catches up, state the gap
explicitly and track it on the board; do not describe intended behavior as already available.

Before closing a task or removing old material, check filename citations and cited details in code,
tests, configuration and prose. Extract current requirements into their owner document and preserve
only evidence with a future use. Keep historical citations distinct from compliance pointers.

| Material | Home |
|---|---|
| Current rules, procedures and technical contracts | `dev/reference/` |
| Reusable templates and their usage instructions | `dev/templates/` |
| Proposals, implementation plans and unresolved choices | Main's `dev/tasks/` |
| Decision rationale or measurements needed beyond a task | Appropriate existing home under `dev/records/` |
| User-facing installation, configuration and usage | `docs/` |
| Disposable notes, logs and probes | Local scratch under the layout policy |

`dev/records/decision-history/` preserves significant historical choices and rationale, including
accepted designs whose adoption was partial or unverified. Developers and agents may consult it to
understand past reasoning, but must use `dev/reference/` for current decisions and compliance.
Historical acceptance labels and imperative wording do not establish present requirements.

Before retaining a decision, place effective requirements in their existing reference owner and
unresolved implementation work with its task. Put measured verification in `records/validation/` or
the relevant milestone. Keep historical IDs and cited sections when consumers need them; otherwise
retain concise rationale and recover superseded implementation instructions from Git.
The history index maps each record to its current owner. Operational instructions must not require
reading history to discover a rule. Optional links must identify their purpose as historical rationale
or evidence. Do not create a new history record for every reference update or retain review bundles
merely because a decision was made.

Live scripts, workflow comments and parameter guidance cite the owning `dev/reference/` topic rather
than an ADR or historical section as authority. The reference may link to history for rationale.
Do not rewrite a historical citation embedded in persisted provenance merely to rename documentation;
retain its identity and resolve it through the current reference's recovery pointer.

## Retention

Delete a retained record only after confirming that needed findings have another home, unresolved
work is tracked, and no citations require its body or identifiers. Check cited detail as well as filenames.
A newer summary or age alone does not establish disposability. Preserve measurements, tolerances and
individual findings when consumers depend on them.

Sealed evidence is immutable under sealed-records.yml (private-checkout-only citation:
`../records/sealed-records.yml`). An authorized relocation may move it unchanged and update the registry path;
never rewrite its body or recalculate its hash to make an edited record pass. Update editable consumers and
record old-to-new paths for immutable citations. Keep intentionally historical IDs and anchors resolvable.

Reviews belong with their task on main; disposable drafts and renders follow the scratch policy.
Persistent task closure retains authored notes and supporting review or acceptance evidence in place.
Current requirements still belong in their owning reference; reusable knowledge may be promoted when it
has an identified consumer. Retained task evidence needs no move into `dev/records/` merely for closure.
Scratch cleanup is a separate explicit operation; closure never authorizes deletion.
Approved skill or role changes must reach their canonical infrastructure repository before discarding
their only evidence. Decisions and milestone evidence receive the same citation review, with sealing
taking precedence.

`dev/records/reviews/` and `dev/records/review/` are retired and must not be recreated.
`dev/scripts/private/check_activation.py` rejects those directories and current-reference links to them, as well as
dependencies on transient tasks. The retirement map (private-checkout-only citation:
`../records/migrations/review-retirement-2026-10-20.md`) resolves historical citations.

## Closure-ledger entries

`dev/LOG.md` is the historical pre-adoption ledger; persistent task closure no longer appends to it.
The retained task note owns its explicit result and append-only closure history.

When maintaining an existing historical row, keep it at most three short sentences: what closed, why, and where durable outcomes
live. Promote measurements, findings, tolerances and decisions first, then link to their home.
This limit does not authorize dropping unique evidence.

Before shortening a row, confirm removed findings survive elsewhere. Preserve cited item IDs and identify
the Git commit from which full text is recoverable. Older rows are reviewed individually, not shortened
by a blanket sweep. Shared board and ledger changes are committed on main separately.

## Milestone records and historical recovery

Milestone records (private-checkout-only citation: `../records/milestones/README.md`) own milestone evidence
and seal dates. The historical roadmap (private-checkout-only citation: `../records/milestones/roadmap.md`)
preserves phase lineage; it is not a source of current requirements. Seal superseded specifications when they
are needed as acceptance evidence and retain them under records, while current references continue to be
updated.

For pre-board item identifiers cited by code, tests or records, recover the original source with
`git show 45e3a564:dev/followups-archive.md`. For the earlier development policy narrative, use
`git show cda3900a:dev/README.md`. Git history supports investigation; it is not another active rulebook.
