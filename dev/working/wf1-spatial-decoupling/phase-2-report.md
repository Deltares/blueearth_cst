# Phase 2 report — Wflow-SBM build from the spatial foundation

## Result

Phase 2 is implemented and paused at master Gate 2. Workflow 1 now has a
distinct `prepare_spatial_maps` rule (1.02) and `build_wflow_model` rule (1.03).
The latter declares all nine P1 products as inputs and writes the standard
Wflow triplet plus its rebuild sentinel. No baseline has been recorded.

## Selected adapter route

The implementation uses the Gate 1-approved project-owned adapter in
`blueearth_cst/model/build_wflow_model.py`. It opens the generated P1 catalog,
loads the P1 grid, five vector layers, and registry, and initializes
`WflowSbmModel` through public component APIs. It then:

1. translates P1 ArcGIS D8 to Wflow LDD;
2. sets the inherited subbasin IDs as `subcatchment` before outlet setup;
3. calls public Wflow parameter methods for rivers, LULC, LAI, soil, and
   constant parameters;
4. calls `setup_gauges(..., index_col="wflow_id", basename="locations")` and
   `setup_outlets()`;
5. writes, reopens, and validates the Wflow grid, triplet, and IDs.

`setup_basemaps` is forbidden by the adapter and removed from
`config/templates/wflow_build_model.yml`. The adapter supplies P1 hydrography,
rivers, LULC, and LAI objects directly. The pinned public `setup_soilmaps` API
accepts a catalog source name rather than an in-memory P1 soil object, so soil
pedotransfer still reads `soilgrids` from the project catalog. Wflow constant
parameters remain unchanged in P2.

The project-owned script is not a HydroMT build/update YAML and therefore has
no applicable `hydromt check` entry point. Validation instead uses the pinned
public API to write and reopen the model; the subsequent HydroMT forcing update
and Wflow.jl run both completed.

## Produced contract and identities

The clean integration model contains `staticmaps.nc`, `wflow_sbm.toml`, and
`staticgeoms/`, followed downstream by historical forcing, runtime state paths,
and `run_default/output.csv`.

Automatic fallback is now constrained to active P1 river cells as well as the
configured maximum. The integration basin therefore resolves to five
subbasins/primary locations (IDs 101–105), not twenty below-threshold drainage
cells. Every `gauges_locations` value joins once to `location_registry.csv`;
the model outlet retains P1 subbasin ID 101. HydroMT-Wflow parses the raw CSV
into distinct `Q_outlets`, `Q_gauges_locations`, and `P_gauges_locations`
arrays even though outlet 101 also appears in the gauge map.

`outlet_index.csv` now preserves the compatibility label and adds deterministic
`basin_code`, `subbasin_id`, `subbasin_code`, `location_code`, `station_name`,
and `wflow_id` fields. Observation headers are validated against the resolved
registry before HydroMT reads the series: missing, duplicate, and unexpected
IDs are reported explicitly. User-provided controls/observations require one
series each; synthetic automatic outlets are optional.

## Gate 2 result comparison

The pre-split model and clean split model have identical 16 × 24 grids, bounds,
EPSG:4326 CRS, LDD, elevation, river mask, river width/depth/slope, and active
soil parameter maps. The new model adds only `gauges_locations` to the static
variable set.

Selected active-cell differences are:

| Map | Equal cells | Mean absolute difference | Maximum absolute difference | Cause |
|---|---:|---:|---:|---|
| `meta_upstream_area` | 0.00% | 0.6718 km² | 3.2885 km² | P1 recomputes accumulation on the accepted analysis grid |
| `meta_streamorder` | 49.03% | 0.5525 | 2 | P1 derives order from that grid-local flow network |
| `land_slope` | 98.05% | 0.000101 | 0.01413 | P1 model-neutral elevation/slope resampling |
| `river_length` | 98.83% | 0.00646 m | 0.8278 m | river geometry derived from P1 hydrography |
| `meta_landuse` | 78.99% | 3.2607 class units | 86 | P1 first resolves categorical LULC to the shared grid |
| monthly LAI | 94.78% | 0.06416 | 4.7272 | P1 first resolves LAI to the shared grid |

All checked soil maps are exactly equal on active cells. The TOML has only two
logical differences: it registers `input.gauges_locations` and adds discharge/
precipitation columns for the registry gauge map. All Wflow physics constants
and other configuration values are identical.

The primary outlet comparison aligns 7,670 daily values from 2000-01-02 through
2020-12-31:

| Statistic | Pre-split | Split |
|---|---:|---:|
| Mean discharge | 10.9460 m³/s | 10.9477 m³/s |
| Standard deviation | 11.5398 m³/s | 11.5453 m³/s |
| Median | 9.7058 m³/s | 9.6935 m³/s |
| 95th percentile | 29.2906 m³/s | 29.3058 m³/s |
| 99th percentile | 42.2865 m³/s | 42.5036 m³/s |
| Maximum | 206.7375 m³/s | 207.2110 m³/s |

Mean bias is +0.00164 m³/s (+0.01496%), RMSE is 0.1616 m³/s (1.476% of the
pre-split mean), and Pearson correlation is 0.999902. This small numerical
delta is attributable to the documented P1 grid-local accumulation/order and
P1-first LULC/LAI resampling decisions; it is not caused by changed Wflow
constants.

The repository's stricter per-timestep comparator classifies the change as
material: 6,343 of 7,670 timesteps exceed its tolerance (absolute threshold
0.01095 m³/s, equal to 0.1% of the reference mean, or 1% relative where the
reference is above that threshold). Maximum absolute difference is 29.73% of
mean reference flow; maximum relative difference is 254.1% at low flow. The
first offending date is 2000-01-05 and the largest absolute difference occurs
on 2019-11-22. Aggregate agreement therefore does not justify an automatic
baseline update.

## Validation record

- Focused lint: passed.
- Focused spatial/adapter batch: 25 passed.
- Focused adapter/identity/observation/DAG batch: 31 passed; one fixture-only
  rebuild-cascade test skipped in the isolated worktree.
- Baseline discharge comparator batch: 14 passed.
- Full repository suite: 1,014 passed, 31 skipped, one expected xfail.
- Targeted dry-run through `run_wflow`: seven rules with the required P1→P2
  dependency.
- Final dry-runs: Workflow 1 (17 jobs), Workflow 2 (25 jobs), and Workflow 3
  (50 jobs) all built valid DAGs against the completed absolute fixture.
- Clean integration through `run_wflow`: seven of seven rules completed.
- Wflow.jl 1.0.2 historical run: completed the full 2000–2020 simulation.
- HydroMT output reopen: found `Q_outlets`, `Q_gauges_locations`, and
  `P_gauges_locations` with the expected IDs.
- Real Gabon four-location check: P1 and P2 targets completed in a disposable
  latest-schema project; stations resolve to IDs 101–104 and the actual
  observation header validates exactly.
- Existing Workflow 1 baseline fixture: all five recorded targets still match
  its manifest. No record command was run.
- Split-versus-reference comparator: FAIL (material), as quantified above.

## Remaining risks and gate decision

- The aggregate physics delta is small and explained, but the repository's
  per-timestep gate classifies it as material. Owner approval is required
  before the baseline is recorded.
- The external Gabon config still uses the one-release compatibility key
  `workflows.model_creation.output_locations`; new configs should use
  `shared.basin.gauge_points`.
- Windows HydroMT processes continue to print the known benign empty
  `Error in sys.excepthook` shutdown cascade while returning success.

Gate 2 should approve or reject the documented resampling and discharge delta.
The branch must not record a baseline until that decision is explicit.
