# Agent and skill activation

How roles and skills become available to Claude Code and Codex **in this
repository**, and why the two runtimes differ.

The canonical spec is in the brain repo — `dev/decisions/0003-skill-activation.md`
(§3 D1/D2) and the `brain-agent-system` skill. This note records only how that
spec resolves here, plus the repo-specific consequences. Do not restate the
brain's rules; link to them.

## Where activation is declared

| File | Scope | Materializes |
|---|---|---|
| `.claude/agent-manifest.yml` | this repo | `.claude/agents/`, `.claude/skills/`, `.agents/skills/`, `.codex/agents/*.toml` |
| `brain/config/ai/agent-manifest.yml` | user | the same trees under `~/` |

Both are tracked in their own repo; every materialized tree above is a
**gitignored symlink farm** rebuilt by `brain refresh --agent-system`. A skill
or role on disk in the brain is inert until a manifest lists it.

`.claude/agent-manifest.yml` has two blocks: `roles:` (the 9 active here) and
`skills:` (the *manifest-explicit* list). Roles additionally carry their own
`skills:` frontmatter, each entry either unconditional (`always` or bare) or
conditional (`when …`).

## The rule

Activation closure is computed **per platform** — the same manifest resolves to
different sets:

| Binding | Claude | Codex |
|---|---|---|
| manifest-explicit | `.claude/skills/` — global | `.agents/skills/` — root catalog |
| role-bound `always` / bare | `.claude/skills/` — global | per-agent route only |
| role-bound `when …` | **nothing — unreachable** | per-agent on-demand route |

Two consequences drive every decision below.

**A `when …` binding is inert on Claude.** Claude has no on-demand read path, so
a conditionally-bound skill is not symlinked and therefore reachable by nobody —
not the main loop, and not the subagent whose own frontmatter names it. On Claude
the manifest is the *only* way to reach such a skill.

**Role-bound skills are invisible to the Codex root agent.** Codex publishes only
the manifest-explicit set to the root catalog; a role's bindings — including
`always` ones — are appended to that role's generated
`.codex/agents/<role>.toml` as `read this skill when … — read: <abs path>`.
Codex *does* have an on-demand read path, so the role route works there.

Listing a skill in `skills:` is not "loading" it. It publishes the skill's
`description` (~280 chars each) into the catalog; the body loads only when the
trigger fires.

## When to add a manifest entry

The brain's criteria, applied here:

- **(a)** No active role binds it unconditionally, and Claude needs it — the
  manifest is the only thing keeping it in scope.
- **(b)** The Codex *root* agent invokes it directly, even though a role already
  binds it `always`. Costs root-catalog budget, so state the reason in a comment.

Skills a specialist role binds `always` are deliberately **not** relisted.

## How it resolves in this repo

As of 2026-07-31 — 14 manifest-explicit skills, 9 roles:

- **Claude global (19)** = the 14 explicit + 5 pulled unconditionally by roles
  (`scientific-appraisal`, `scientific-workflows`, `reproducible-computing`,
  `scientific-visualization`, `data-visualization`).
- **Codex root catalog (14)** = the manifest-explicit set only. The 5 above are
  *not* here; Codex reaches them per-agent.

Of the 14 explicit entries, 12 are load-bearing under (a): eight are bound
`when`-only somewhere (`snakemake`, `hydromt`, `hydromt-wflow`, `wflow`,
`climate-projections`, `climate-stress-testing`, `cst-run-control`,
`python-geospatial`) and four are claimed by no active role at all (`pixi-env`,
`design-review-loop`, `design-scoping`, `python-publication-figures`). The
remaining two — `git-workflow` (`always` via `git-steward`) and
`python-discipline` (`always` via `python-engineer`) — are redundant on Claude
and kept under (b), because `.codex/` drives this repo too.

### Why most of this repo runs un-roled

The routing gate dispatches a subagent only for substantial, self-contained work.
Small mid-flow edits, lookups, and thread-dependent follow-ups are handled inline
by the main loop, which carries **no role** — its behavior comes from
`AGENTS.md` plus whichever skills triggered. That is why the explicit `skills:`
list matters so much here: it is the main loop's only domain equipment, and a
`when …` binding never reaches it.

## Gotchas

- **Codex root invocation is implicit.** Root routing is decided from injected
  name + description metadata. It is not guaranteed. The deterministic lever is
  naming the skill in `AGENTS.md`, which Codex reads directly.
- **Over-budget catalogs truncate.** Past the discovery floor (8,000 chars when
  the context window is unknown) Codex may shorten or omit descriptions — a
  trigger that silently stops firing. Budget is reliability, not tidiness.
- **`brain status --agent-system` reports the brain's own scope**, not this
  repo's, when run from the brain root. Its `catalog_budget` finding is about
  `brain/.claude/agent-manifest.yml`, where the overage is a recorded accepted
  decision.
- **Manifest edits do nothing until refreshed.** A listed-but-not-materialized
  skill is a silent gap.

## Verifying

```bash
# activate this repo only (targeted repair; the bare form sweeps every project)
brain refresh --agent-system --project /c/Users/taner/workspace/blueearth_cst

# what Claude sees (19) vs what Codex root sees (14)
ls .claude/skills/ ; ls .agents/skills/

# a role's generated on-demand routes
grep "read:" .codex/agents/model-builder.toml
```

The Codex side of the table above was verified from generated files. The Claude
side follows ADR 0003 D1 and the observed `.claude/skills/` contents matching
`explicit + always` exactly; it has not been probed by spawning a subagent and
asking what it can load.

## Open

- `r-discipline` and `pipeline-regression-testing` are bound `when`-only across
  all 9 roles, so they are absent from Claude scope entirely. Both are live
  surfaces here (six tracked `.R` files; `check_baseline*`). Addition proposed,
  not decided.
- `weathergenr` was considered and declined — generator work routes through
  `model-builder`, which binds it.
- `AGENTS.md` currently names **no** skills, so criterion (b) for `git-workflow`
  and `python-discipline` rests on implicit routing alone.
