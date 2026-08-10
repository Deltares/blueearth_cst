Status: accepted
Date: 2026-08-09
Deciders: Ümit Taner
Consulted: the shared spatial foundation (`data/spatial/`, ADR 0003) and the
           cartographic template (`shared/cartographic_map.py`)
Supersedes: none

# ADR 0007 — Draw basin_area from the spatial foundation, not the model

### Context

`basin_area.{png,pdf}` lived at `models/hydrology/wflow/plots/` and was drawn
from the wflow model: the DEM from `staticmaps.nc`, the vectors from
`staticgeoms/`. But the figure depicts ELEVATION and a basin outline — data
about the catchment, not a result the model produced. Its location said
otherwise, and a reader looking for the study area's map found it inside the
engine subtree.

Everything it needs already exists outside the model. `data/spatial/` carries
`hydrography.nc` with an `elevation` variable on the same grid, and
`geoms/` carries basins, subbasins, rivers and locations — the same layers the
climate maps already read.

`AGENTS.md` recorded this proposal as considered and rejected, on three
grounds: waterbodies come from rule 1.04, the gauge layer from 1.05, and
`SpatialUnits` carries neither. Two of the three are now stale —
`locations.geojson` carries the gauge points with `wflow_id`, and the basin
outlet is one of them. Only waterbodies remain.

### Decision

Draw `basin_area` from `data/spatial/` and write it to
`data/spatial/plots/`. Rule 1.12 reads `hydrography.nc` and the four geom
layers; `plot_basin_map_from_spatial` replaces `plot_basin_map_from_model` as
its entry point.

The figure is therefore model-independent: it renders before a wflow build
exists, the same property rule 1.05's source-climate figures have.

### Consequences

- **Rule 1.12's HDF5 race workaround is retired wholesale.** It read
  `staticmaps.nc` straight off disk while rule 1.08 wrote that file as an
  undeclared side effect, so at `-c 3` the two overlapped and the rule aborted
  BELOW Python with no traceback (`HDF5_USE_FILE_LOCKING=FALSE` in the pixi
  env). The fix was a sentinel ordering edge plus an `ancient()` declaration on
  `staticmaps.nc`. Not opening the file removes the race and both mitigations,
  and drops 1.12's dependency on the model build entirely.
- **Waterbodies are gone from the figure.** Lakes, reservoirs and glaciers
  reach `staticgeoms/` from rule 1.08, a model rule; the foundation carries
  none. This basin has none so nothing changes here, but on a basin with a
  major reservoir `basin_area` would silently omit it. Tracked as
  `dev/tasks/t2608091730-*`: produce those layers data-side and have 1.08
  consume them. The shortcut — 1.08 also writing into `data/spatial/` — is
  rejected: a model rule writing there makes the tree model-dependent and
  undoes this decision.
- The R9 path map (`tests/test_r09_path_map.py`) still maps the figure to its
  old `models/hydrology/wflow/plots/` home. That is correct: it is a one-way
  migration map describing the move R9 performed, and this is a later,
  separate move.

### Alternatives considered

**Move the location only**, keeping the model as the producer. Rejected: a path
under `data/` that only exists after a model build is misleading, and the
figure would still go stale on a model rebuild that changed nothing it draws.

**Do the waterbody producer relocation first**, so nothing is lost. Rejected as
sequencing, not as direction: it is a producer relocation with its own design
question, and holding a correct, small move behind it buys nothing. The gap is
recorded and visible rather than discovered later.
