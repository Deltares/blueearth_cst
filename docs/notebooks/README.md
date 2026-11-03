# Workflow notebooks

Optional interactive sources for advanced readers:

- [Model building](<Model building.ipynb>): historical model preparation and inspection.
- [Climate projections](<Climate projections.ipynb>): projection composition and change-factor reading.
- [Climate stress test](<Climate Stress Test.ipynb>): retained scenario/indicator exploration.

## Current status

The cells execute real workflows and need installed tools, external data and sufficient storage/compute.
Some code uses older configuration reads, result paths and indicator interfaces.
Markdown notes identify those limitations; code has not been repaired or requalified in this prose pass.
Use the [command-line quick start](../guide/quick-start.md) for the maintained execution route.
Supported response-surface construction remains deferred; do not treat the third notebook as a verified
current assessment tutorial.

## Running them

Complete [installation/data checks](../guide/install.md), then open from your selected toolbox checkout:

```powershell
pixi run jupyter lab docs/notebooks
```

Inspect every command and path before executing. Use one checkout for a case and one writer per output root.
WF1 and WF3 independently supply WF4; WF2 is optional plausibility context.
Changing `CONFIG` alone is not sufficient to qualify these notebooks for another case.

## Outputs are NOT committed

The `.ipynb` file is the sole editable tutorial source. Keep outputs and execution counts cleared.
`dev/scripts/notebook_outputs.py --check` reports stored outputs; `--strip` removes them.
`tests/test_notebook_outputs.py` enforces source-only notebooks; a local hook is additional protection.
No rendered artifact or measured result is supplied by an empty source notebook.

### Re-rendering

Rendering with execution runs the real pipeline; it is not part of the documentation build.
Do not delete scientific outputs merely to make a notebook display fresh work.
Use the owned runner and an explicit output root for any separately authorized execution.
Afterwards inspect errors, workflow exit outcomes and expected tables/figures, not only nbconvert's exit code.
Do not commit generated outputs or execution counts.

## Data

The rapid example references external catalogs under `config/catalogs/`.
Check actual source entries and coverage; software installation does not supply model inputs.
Use the [dataset reference](../technical-manual/datasets.md) and the case's retained provenance.
The example is uncalibrated; execution and source agreement do not establish basin skill.

## Related reading

- [Notebook user guide](../guide/notebooks.md).
- [Read indicator tables](../guide/outputs.md).
- [Independent indicator module](../../blueearth_cst/indicators/README.md) for advanced calculations.
