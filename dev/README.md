# Development workspace

Start with [the current reference](reference/README.md) for development practices and toolbox design.
Developers and agents use the same guidance.

Private coordination and historical records are cited as private-checkout-only locations; they are not
public install or CI prerequisites. Shared current toolbox guidance remains available through the
reference index, with user setup and execution instructions in `docs/`.

| Location | Purpose |
|---|---|
| [reference/](reference/README.md) | Current rules, architecture and contracts, maintained in place |
| `tasks/` and `TODO.md` | Persistent task records and generated Work view; canonical board on main |
| [templates/](templates/README.md) | Reusable report/review templates and usage instructions |
| `LOG.md` | Historical pre-adoption closure ledger |
| [scripts/](scripts/README.md) | Developer maintenance and verification tools |
| `baseline/` | Replication fingerprints and reference series |

- records/ (private-checkout-only citation: `records/README.md`): Selected historical evidence and design
  context

`dev/tasks/` links to main in configured sessions; views and the historical ledger remain main-only.
References, templates and records remain versioned with each branch's code. See worktree coordination
(private-checkout-only citation: `reference/private/dev-task-and-session-workflow.md`) before setup or handoff.

Use [repository layout](reference/dev-repository-layout.md) for file placement and local scratch,
[record lifecycle](reference/dev-record-lifecycle.md) for retention, and
[validation ladder](reference/dev-validation.md) for checks.
User-facing documentation belongs in `docs/`.

Environment-specific recipes live in [scripts/environment-notes.md](scripts/environment-notes.md).
