---
title: "Use the workflow notebooks"
subtitle: "Inspect and adapt analyses interactively"
---

The toolbox includes Jupyter notebooks for readers who want to inspect Python code, explore
intermediate results or adapt an analysis interactively. For a first run, use the maintained
[Quick-start commands](quick-start.md). To inspect completed results without executing code,
follow [Reading the outputs](outputs.md).

## Choose a notebook

The [notebook index](../notebooks/README.md) links three walkthroughs:

| Notebook | Intended use |
|---|---|
| Model building | Inspect historical model preparation and its outputs |
| Climate projections | Explore projection processing and climate change factors |
| Climate stress test | Examine scenario and indicator analysis code |

**These notebooks are advanced examples, not verified current end-to-end tutorials.** Some cells
still use older configuration, path or indicator interfaces. Read each notebook's opening status
note before executing it. In particular, the stress-test notebook is not a verified procedure
for constructing response surfaces with the current toolbox.

## Open a notebook

Complete the [installation and data-access checks](install.md), then run this command from the
`blueearth_cst` folder used for your case:

```powershell
pixi run jupyter lab docs/notebooks
```

JupyterLab opens the notebook folder. Select a notebook and inspect its configuration reads,
commands and output paths before running cells. Use the toolbox environment for its kernel.
The supplied notebooks contain source code rather than saved results, so empty outputs are expected.

## Work safely with your case

Notebook cells can launch real workflows, download data and write model outputs. Changing a
configuration path alone may not update every command or output location in an older notebook.
Keep your assessment outputs outside the toolbox repository and check those paths before execution.

Use the [documented runners](running.md) for scenario generation and simulation, and allow only
one active process to write to a case's output directory. Inspect each cell's output before
continuing; a plot appearing successfully does not establish that all underlying workflows completed.

When a cell uses an outdated interface, compare it with the current configuration and output guides
before adapting it. If you only need to run an assessment, return to the command-line route rather
than relying on the notebook to reproduce it.

The documentation build does not execute these notebooks. See the
[notebook index](../notebooks/README.md) for their current status and source-handling details.
