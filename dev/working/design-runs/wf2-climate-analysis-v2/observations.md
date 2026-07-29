# Observations — wf2-climate-analysis-v2

Process friction only. Never design content. Feeds the post-run retrospective
and any skill edits made *after* this run ends (one skill version per run).

## 2026-07-29 — run-dir path assumes `dev/drafts/`

`references/run-artifacts.md` § Run directory gives
`<target-repo>/<working-docs>/design-runs/<slug>/` with the parenthetical
"e.g. `dev/drafts/design-runs/<slug>/`". This repo's `dev/README.md` documents a
seven-folder grammar whose ephemeral-working folder is `dev/working/`, with no
`dev/drafts/`. The generic form is correct and was followed
(`dev/working/design-runs/wf2-climate-analysis-v2/`), but the example reads as
the default at a glance.

Same defect class as the one found and fixed in `design-document` earlier this
session (brain `c63f130f`): a hardcoded folder example competing with a
documented repo grammar. Candidate: make the example grammar-agnostic here too.
**Not applied during the run** — the skill stays frozen for the run's duration.

## 2026-07-29 — stage 1 seeded, not authored

The design already existed (`dev/workflows/wf2-climate-analysis-v2-design.md`,
landed f5cd5ff) before the review was requested, so stage 1 used the seeding
path rather than a fresh author spawn. Seed is the **corrected** version — it
includes the §5.3 fix reclassifying the `historical_year_range` retirement as
value-changing, found while writing the steps 1–2 task brief. Seeding the
pre-correction version would have handed reviewers a known defect.

## 2026-07-29 — Fable lens is an owner-requested tier deviation

The skill rations Fable to revision spawns answering an external review that
re-raised a prior finding. The owner asked for a Fable review directly, which
overrides. Logged in `status.md` as `dispatches.fable: 1` and flag
`owner-requested-fable-lens` so the quota meter stays honest. No skill change
implied — an explicit owner request is exactly the kind of override the default
should yield to.

## 2026-07-29 — codex preflight held

`-c approval_policy=never` produced a banner reading `approval: never`,
`sandbox: read-only`, `model: gpt-5.6-sol`. Worth noting that the adapter's
warning is real on this machine: `~/.codex/config.toml` sets no
`approval_policy`, so the default `on-request` would have applied without the
explicit `-c` flag. The preflight probe cost ~24.5k tokens.

Bash-form invocation used (`- < brief.md`, stdin closed via file redirect,
background dispatch) — the PowerShell `'' |` form in the adapter fails in Bash,
as its own note says.
