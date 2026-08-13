---
title: Repair and extend the workflow notebooks onto the fao pattern
type: todo-item
status: backlog
effort: 1
area: docs / notebooks
origin: fao branch assessment (2026-08-13)
queue:
created: 2026-08-13
updated: 2026-08-13
---

> [!note] Overview
> **What** — Repair the three broken notebooks in `docs/notebooks/` against the current tree, then restructure them onto the pattern the upstream `fao` branch uses: config authored in the notebook, input schemas shown, a rule-by-rule narrative naming the config keys that tune each rule, and results read rather than merely displayed.
> **Why** — They are the only user-facing "how do I actually run this" artifact, and all three are currently non-runnable — they write a config path the repo does not use and cite pre-R9 result paths. `fao`'s five are markedly better and cost nothing to learn from.
> **Effort** — Medium. Mostly prose and a real run to render against; no code outside `docs/`.

## They are broken, not merely stale

Verified 2026-08-13 against the working tree. All three (`Model building`,
`Climate projections`, `Climate Stress Test`) share the same three defects:

- They `%%writefile ./config/my-project-settings.yml` and then run against it.
  The repo's shipped seed configs live at `test_case/snake_config_*.yml`
  (`AGENTS.md` § Repo Map), so the notebook teaches a config location the repo
  does not use — and one that will never carry the `snake_config_` prefix the
  `.gitignore` un-ignore rule depends on.
- The DAG is written to `../../test_case/test_local/dag/dag_*.png` and then
  displayed from `./dag_*.png`. Mismatched, so the display fails; the directory
  does not exist; and the repo's actual convention is
  `<project_dir>/logs/dag/...` via `scripts/plot_workflow_dag.py`.
- Result paths are pre-R9: `examples/myModel/hydrology_model/plots/basin_area.png`
  (now under `data/spatial/plots/`) and `.../evaluation/plots/hydro_wflow_1.png`
  (now `models/hydrology/wflow/evaluation/plots/hydrograph_<wflow_id>.png`).

The last two are the R9-class failure `AGENTS.md` already names: a tree move
that no gate could catch, because nothing executes these files.

## The pattern to adopt

From `dev/reviews/2026-08-13_fao-branch-assessment.md` §6.1. In order:

1. Intro naming the Snakefile and numbering what it does.
2. **Config authored in the notebook**, every option commented in place — the
   settings file *is* the tutorial rather than being described by it.
3. Input-schema cells: `pd.read_csv(...)` on the observation and station files,
   so the required columns are shown rather than specified.
4. DAG render, then a **rule-by-rule narrative in which each rule names the
   config keys that tune it**. Main's docs have no equivalent of this anywhere.
5. `--unlock`, `--dryrun`, then the real run.
6. DAG render after, results-tree walk, then figures **with interpretation** —
   what the reader should conclude, not just what is plotted.
7. Forward link to the next notebook, so the set reads as one narrative.

Point 6 is what makes them worth more than our current docs: they teach the
*reading* of the output, which is the part a rapid-assessment tool most needs to
transfer.

Deviate from `fao` on two points. Its config cell writes to a path that does not
match our seed-config convention — point 2 must target a real
`test_case/snake_config_*.yml` shape. And `os.chdir(r'c:\repos\blueearth_cst')`
is hardcoded in all five; do not carry that across.

## Rot control — RULED, do not re-litigate

Owner ruling 2026-08-13 (assessment §6.3): **outputs are committed**, each
notebook carries a dated **`rendered against <sha>`** banner, and a periodic
re-render is a board item. Staleness is made *visible* rather than prevented.

CI cannot execute these — a bare checkout has no `test_case/test_local` and no
data access — so the alternative was silent rot, which is how the current three
got here. Render from the primary checkout against
`test_case/snake_config_rapid.yml` (`AGENTS.md` § Which config to run: anything
you want to watch EXECUTE), not the baseline config.

## Progress

- [ ] Repair the three existing notebooks against the current tree — config
      path, DAG path, result paths. Verify by running them, not by reading them.
- [ ] Restructure onto the §6.1 pattern; add the rule-by-rule narrative naming
      config keys per rule.
- [ ] Render against `snake_config_rapid.yml` from the primary checkout; commit
      outputs with the `rendered against <sha>` banner.
- [ ] Add a `docs/notebooks/README.md` index. `fao`'s `README.rst` is the model,
      including its dataset-provenance citation table — worth copying on its own.
- [ ] Open the periodic re-render item the ruling calls for.
- [ ] Consider a fourth notebook once [[t2608131847a-split-historical-climate-out-of-wf1]] lands.

## Notes

Runs belong in the **primary checkout**, not this lane's worktree
(`AGENTS.md`: `.snakemake` metadata diverges between checkouts driving one
`project_dir`, and both hold their own lock). Editing prose here is fine;
rendering is not.

## Refs

- `dev/reviews/2026-08-13_fao-branch-assessment.md` §6 — the pattern, the
  defects, and the rot-control ruling.
- `upstream/fao:docs/notebooks/` — five notebooks + `README.rst`.
- `AGENTS.md` § Which config to run; § Figures are terminal artifacts.
- Related: [[t2608131847a-split-historical-climate-out-of-wf1]] — that split
  adds a fourth workflow and therefore a fourth notebook. This item's output is
  effectively its specification, so do this one first.
