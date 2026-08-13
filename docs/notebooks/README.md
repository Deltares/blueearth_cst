# Workflow notebooks

Three notebooks, one per workflow, meant to be read in order. Each names the
Snakefile it drives, walks the settings file that controls it, renders the job
graph, runs it, and then *reads* the results rather than only displaying them.

| # | Notebook | Workflow | What it covers |
|---|---|---|---|
| 1 | [Model building](<Model building.ipynb>) | `Snakefile_model_creation` | Delineates the basin, extracts a historical climate store, builds and forces a Wflow-SBM model with hydromt, runs it once, and evaluates it against observed discharge. |
| 2 | [Climate projections](<Climate projections.ipynb>) | `Snakefile_climate_projections` | Fetches CMIP6 slices for the basin and derives monthly and annual change factors per model, scenario and horizon — the plausibility overlay, not a driver of the stress test. |
| 3 | [Climate stress test](<Climate Stress Test.ipynb>) | `Snakefile_climate_experiment` | Generates stochastic weather realizations, perturbs them across a temperature × precipitation grid, runs Wflow for every combination, and reduces the result to a response surface. |

Run them in order. Notebook 3 does not rebuild the model — it binds to the one
notebook 1 left behind, and refuses to run against a stale build.

## Running them

The notebooks execute the real pipeline, so they need the toolbox installed, not
just a Python kernel:

```bash
pixi install       # Python stack, R toolchain, snakemake, graphviz
pixi run install   # + weathergenr (R) and the Julia environment
```

Julia is **not** in the pixi environment — it is juliaup-managed and must
already be on `PATH`. See `docs/install.md` if setup misbehaves.

Then start Jupyter (or VS Code) from inside that environment:

```bash
pixi run jupyter lab docs/notebooks
```

Every notebook locates the repository root itself by walking up from the
kernel's working directory, so it does not matter where you start it — there is
no install path to edit.

All three run against `test_case/snake_config_rapid.yml`, the cheap
end-to-end config. To point one at your own project, change the `CONFIG`
constant in the setup cell; everything downstream is derived from the config.
For a new project, copy `config/templates/snake_config.template.yml`.

Run the pipeline from the **primary checkout**, not from a task worktree.
Snakemake keeps its up-to-date metadata under the working directory, so one
project driven from two checkouts gets two stores that disagree, and each holds
its own lock while writing the same outputs.

## Committed outputs, and how they go stale

Each notebook is committed **with its outputs** and carries a dated
*rendered against `<sha>`* banner at the top. That is deliberate: the
interpretation these notebooks exist to teach needs the rendered figures and
tables in front of the reader, and nothing in CI can produce them — a bare
checkout has neither the project tree nor the data access.

So staleness is made **visible** rather than prevented. The banner tells you
which commit the numbers came from; if the pipeline has moved since, the prose
is still current but the numbers are not. Re-rendering is a tracked board item,
not an automated gate.

If you edit a notebook's prose without re-running it, leave the banner alone —
it describes the outputs, not the text.

### Re-rendering

From the primary checkout, on a commit that already carries the prose you want
rendered:

```bash
pixi run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 "docs/notebooks/Model building.ipynb"
```

Then update each banner to name the commit that was executed, and commit the
outputs with it.

Two things to know before you run it:

- A run against an already up-to-date project reports "nothing to be done", so
  the run cells render empty of work. Deleting the terminal outputs first — the
  `plots/` directories, `performance_metrics.csv`, `experiments/*/results/` —
  makes those rules re-execute without redoing the model build or the CMIP6
  fetches.
- Deleting `performance_metrics.csv` re-runs Wflow. Rule 1.15's input is
  `run_default/output.csv`, which rule 1.14 declares as `temp()`, so a completed
  run has already removed it. That is expected, not a cascade bug.

Verify by checking the executed notebooks rather than by reading them: no cell
carries an `error` output, and no cell's captured text contains `Error in rule`
or `Exiting because a job execution failed`. The `run()` helper returns the
subprocess exit code and nothing acts on it, so a failed Snakemake call inside a
notebook otherwise reads as a successful cell.

To read a rendered notebook as a web page. Pass `--output-dir`, or nbconvert
writes a ~7 MB untracked `.html` beside the notebook, inside the tracked
`docs/notebooks/`:

```bash
pixi run jupyter nbconvert --to html --embed-images \
  --output-dir .tmp/notebooks-html "docs/notebooks/<name>.ipynb"
```

Add `--template basic` for a fragment with no external references at all; the
default template pulls MathJax, mermaid and require.js from a CDN, which is
harmless in a browser but not offline-clean.

## Data

The rapid config builds a small test basin from global datasets, registered in
`config/catalogs/deltares_data.yml` (physiography, land surface, climate) and
`config/catalogs/cmip6_data.yml` (projections, generated from a live listing of
the public CMIP6 store).

Sources are never hardcoded in a rule: they are named in a catalog and handed to
hydromt with `-d`, so retargeting a project at different inputs is a config
edit. The table below is what the *default* configuration uses; a specific run's
inputs are whatever its config named, and the catalog entry is the authoritative
citation.

| Name | Catalog entry | Type | Reference |
|---|---|---|---|
| MERIT Hydro IHU | `merit_hydro_ihu` | Hydrography | Eilander et al. (2020). doi:10.5281/zenodo.5166932 |
| Reach-level bankfull river width | `rivers_lin2019_v1` | Hydrography | Lin et al. (2019). doi:10.5281/zenodo.3552776 |
| Copernicus Global Land Cover 100 m | `vito` | Land cover | Buchhorn et al. (2020). doi:10.5281/zenodo.3939038 |
| MODIS/Terra+Aqua Leaf Area Index | `modis_lai` | Leaf area index | Myneni et al. (2015). doi:10.5067/MODIS/MCD15A3H.006 |
| SoilGrids | `soilgrids` | Soil properties | Hengl et al. (2017). doi:10.1371/journal.pone.0169748 |
| GRanD v1.1 + HydroLAKES v10 + JRC 2016 | `hydro_reservoirs` | Reservoirs | Lehner et al. (2011). doi:10.1890/100125 |
| HydroLAKES v10 | `hydro_lakes` | Lakes | Messager et al. (2016). doi:10.1038/ncomms13603 |
| Randolph Glacier Inventory v6 | `rgi` | Glaciers | Pfeffer et al. (2014). doi:10.3189/2014JoG13J176 |
| ERA5 reanalysis | `era5` | Climate | Hersbach et al. (2019). doi:10.1002/qj.3803 |
| CMIP6 | `config/catalogs/cmip6_data.yml` | Climate projections | Eyring et al. (2016). doi:10.5194/gmd-9-1937-2016 |

Which land-cover, LAI and soil products a run actually used is set by
`shared.basin.spatial_sources.{lulc,lai,soil}`; the waterbody sources come from
`config/defaults/wflow_update_waterbodies.yml`.

*The basin, the model and the results in these notebooks are for illustration.
This is a rapid, uncalibrated, global-data deployment — see notebook 1's
evaluation section for what that means when reading the numbers.*

## Related reading

- `README.md` — how the three workflows fit together.
- `docs/cst-toolbox-technical-note-2025.md` — the stress-test method and the
  design rationale behind it. Read this before changing *what* a workflow
  computes.
- `docs/install.md`, `docs/env_setup_notes.md` — when pixi, R or Julia setup
  misbehaves.
