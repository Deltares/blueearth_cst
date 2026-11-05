# Development documentation lifecycle

Last updated: 2026-11-05

This reference separates current development guidance, proposals, reusable templates and retained evidence.
See [repository layout](dev-repository-layout.md) for placement and temporary files.

## Current guidance

`dev/reference/` holds maintained development practices and toolbox architecture. Find the existing topic
in the [reference index](README.md), then update that document in place. The
[developer overview](../README.md) orients contributors and links to the detailed references.
Other pages should link to the topic owner with a short reason to read it. Create a new topic only when it
answers a distinct reader question with a separate maintenance boundary; index it and remove overlap.

Write the effective rule or design, its scope, useful rationale, and pointers to implementation and
verification. Keep exact schemas, defaults and inventories in their owning code or configuration. Remove
superseded wording instead of appending dated old/new accounts or a decision record for every revision.
Git history retains earlier versions.

Use a title, `Last updated: YYYY-MM-DD`, a short scope statement and sections suited to the topic. The date
marks a substantive content revision, not a verification certificate; relocation, formatting or index
maintenance alone need no date change. If introducing the field, use the latest relevant Git revision
date when a more precise content date is unavailable. Guidance must describe the effective policy and
implementation in its checkout, and state any known gap explicitly.

## Proposed changes and promotion

Keep unfinished designs, implementation plans and unresolved choices with the work they concern. Design
acceptance alone does not establish implementation. Promote a requirement into its current reference when
the policy or behavior takes effect; if implementation still lags, describe the gap rather than claiming
the intended behavior works.

Before removing material, search filename and section citations in code, tests, configuration and prose.
Move effective requirements to their topic owner; keep evidence that has a future use. A live script or
workflow comment should cite the current topic for compliance, while historical evidence may explain why.
Persisted provenance can retain its original citation identity; give readers a current recovery pointer.

| Material | Home |
|---|---|
| Current rules, procedures and technical contracts | `dev/reference/` or the established component reference |
| Reusable templates and their usage instructions | `dev/templates/` |
| Proposals, implementation plans and unresolved choices | Work-specific notes |
| Decision rationale or measurements needed beyond the work | Retained evidence record |
| User-facing installation, configuration and usage | `docs/` |
| Disposable notes, logs and probes | Local scratch under the layout policy |

Historical acceptance labels and imperative wording do not establish present requirements. Retain
significant rationale, measurements, tolerances and identifiers when a consumer needs them; use the
current reference for the actionable rule. Do not create a history record for each reference update.

## Retention

Delete a retained record only after checking that needed findings have another home, unresolved work
remains visible, and no citation requires its body or identifiers. Search cited details as well as file
names. Age or a newer summary alone does not make evidence disposable. Preserve individual findings,
measurements and tolerances that readers rely on.

If an evidence record is sealed, keep its body unchanged. Relocation requires byte-preserving movement
and an updated registry path; never edit the body and recalculate its hash to make the change pass.
Update editable consumers and retain old-to-new path resolution for immutable citations. Keep deliberately
historical IDs and anchors resolvable.

Keep accepted work notes and supporting review or verification evidence while they have readers. Current
requirements still belong in the maintained reference. Review temporary drafts and renders separately;
completion does not itself authorize deletion. Preserve evidence until its consumers no longer need it.
Do not recreate retired review directories as a parallel home for current guidance.

## Closure-ledger entries

An older closure ledger is historical evidence, not the place for new requirements or routine completion
entries. When maintaining an existing row, keep it brief: what closed, why and where durable outcomes live.
First promote unique findings, tolerances and decisions, then shorten. Preserve cited IDs and identify a
Git commit from which removed detail can be recovered. Review rows individually, never by blanket sweep.

## Milestone records and historical recovery

Milestone records preserve evidence and seal dates; they do not override current guidance. Retain a
superseded specification when needed to interpret acceptance, after promoting its effective requirements.
For an older file or rule, inspect the relevant commit with `git log -- <path>` and
`git show <commit>:<path>`. Interpret historical rule numbers in their dated context because numbers can
be reused. Git history supports investigation; the current reference supplies the present rule.
