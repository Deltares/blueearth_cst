---
title: Land feat/signature-figure-redesign once the rendered figure has been judged
type: todo-item
status: blocked
branch: feat/signature-figure-redesign
effort: 1
area: plotting
queue:
created: 2026-08-09
updated: 2026-08-09
---

> [!note] Overview
> **What** — The branch (a04fcde, +172/-0 to shared/func_plot_signature.py) redesigns the climate-year signature figure -- Theil-Sen slope with a Mann-Kendall test replacing OLS + R2, magnitude-per-decade annotation, panel labels, shared x-axis, Okabe-Ito palette at 180 mm publication width. Its own commit message says NOT verified and NOT ready to land, because rendering needed a built WF1 model. That render has now been done. Before and after were produced from the SAME 21-year forcing series at station wflow_1 (index 101), so the pair isolates the design change from the data. Published for inspection at https://claude.ai/code/artifact/fa17e051-48a4-453a-a6f2-669816dbae46 . Remaining step is the owner visual verdict, then merge to main and rebase the plotting worktree onto it.
> **Why** — Parked 2026-08-09 at the owner request, mid-review. Recorded because everything that closed the gate -- the render, the comparison, the artifact URL and the caveat below -- otherwise exists only in one chat session. Reproduce with the render recipe in Refs. CAVEAT worth re-reading before judging: the render aggregates the model own inmaps_historical.nc per subcatchment rather than reproducing WF1 elevation-parity transform, which needs the Deltares mirror and a full rebuild. Structure, record length and station are real; exact axis values would shift slightly in a production run. Fine for judging a design, not a numerical check.
> **Effort** — small

## Progress

- [ ] <first step>

## Refs

**Render recipe** — reproduces both panels without a WF1 run, from the seeded fixture:

1. Inputs, both already in `test_case/test_local/models/hydrology/wflow/`:
   `forcing/inmaps_historical.nc` (`precip`/`pet`/`temp`, 7671 steps) and
   `staticmaps.nc` (for `subcatchment`).
2. `climate_forcing_by_subcatchment(forcing, static["subcatchment"])` →
   `ds_clim` with `index` × `time`; take `ds_clim.sel(index=101)`.
3. Render *before* by calling `plot_clim(ds_i, out_dir, "wflow_1", "year")` on
   `main`'s `shared/func_plot_signature.py`.
4. Render *after* by writing the branch's copy of that one file
   (`git show feat/signature-figure-redesign:blueearth_cst/shared/func_plot_signature.py`)
   to a scratch path and loading it with `importlib.util.spec_from_file_location`.
   The rest of the package comes from the checkout — the branch's diff touches
   this file only, so the pair differs by exactly the design change.

Why `importlib` rather than checking the branch out: the branch's worktree has no
pixi env, and building one costs minutes to render two PNGs.

**Related**
- [[t2608091006]] — the toolbox-wide plotting standardization this blocks. That
  sweep will touch `func_plot_signature.py`, so landing this first is what keeps
  172 lines of figure code from needing a rebase across a style refactor.
- `AGENTS.md` § *Figures are terminal artifacts* — the gate this follows: render
  it, publish it as an Artifact, never byte-compare or run the baseline.
