---
title: Make the notebooks' run() helper fail loudly on a nonzero exit
type: todo-item
status: backlog
effort: 1
area: docs / notebooks
origin: t2608131847 residue
created: 2026-08-13
updated: 2026-08-13
---

> [!note] Overview
> **What** — The `run()` helper in all three `docs/notebooks/*.ipynb` streams a command's output and returns its exit code, and nothing acts on the return value. Give it `check: bool = True` raising `RuntimeError` on a nonzero exit, with `check=False` on the `--unlock` cell, which legitimately fails when there is no lock.
> **Why** — A failed Snakemake call prints its error into the cell output and the cell still renders as successful. A reader following the notebook only discovers it when a later `display.Image` raises on a file that was never written — an error that points at the wrong place. These are teaching notebooks, so the failure mode is aimed squarely at someone who cannot yet tell the difference.
> **Effort** — Small. One cell per notebook, plus a re-render.

## Why it was not done in the same pass

Changing notebook source invalidates the `rendered against ca4c9df` banner the
render was stamped with, so the fix wants to ride along with the next re-render
rather than force one of its own. It is a wart in what landed, not a defect in
it: every results cell still fails loudly on a missing file, so a failed run
cannot be read as a *complete* notebook — only as a confusing one.

## Do it together with

Whatever next triggers
[[t2608132100-re-render-the-workflow-notebooks-when-their-banner-sha-falls-behind]],
or [[t2608181139-give-wf0-its-forcing-selection-evaluation-layer-rules-0-07-0-09]]'s fourth notebook,
which will need the helper copied into it anyway — and copying the current one
propagates the wart.

The check the render is verified with is separate and stays either way: no cell
carries an `error` output, and no captured text contains `Error in rule` or
`Exiting because a job execution failed` (`docs/notebooks/README.md`
§ Re-rendering). That check exists *because* `run()` swallows the exit code;
closing this item does not retire it, since a Snakemake failure that a notebook
raises on still has to be noticed by whoever re-renders.
