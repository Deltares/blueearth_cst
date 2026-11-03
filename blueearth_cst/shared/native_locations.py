"""Label every Wflow CSV column with the location registry's ``wflow_id``.

Wflow names a mapped column ``<header>_<map value>``, and the map decides what
the value means. A gauge map burns in the ``wflow_id`` itself (``Q_1010``); the
``outlets`` and ``subcatchment`` maps burn in the SUBBASIN id (``Q_101``,
``aet_101``). One run's results therefore spoke three id spaces at once -- q on
101, 1020, 1030, 1040 and aet/gwr on 101..104, beside WF1 figures keyed
1010..1040 (t2609251515).

Every derived result is keyed by ``wflow_id`` instead:

* a column on a gauge map keeps its value, already a ``wflow_id``;
* a column on the outlets or an area map takes its subbasin's PRIMARY
  ``wflow_id``, the one ``plot_results.resolve_stations`` names the outlet by;
* two point columns on one model cell -- a gauge snapped onto the outlet -- are
  one series, and the GAUGE is kept, since its id is already the label; so is
  an outlet whose primary gauge sits a cell off it, being that gauge's point.

The subbasin -> primary crosswalk is read from the model's own gauge layer
(``staticgeoms/<map>.geojson``), which WF1 writes from the location registry in
the same build as ``staticmaps.nc``. ``staticgeoms/`` is outside the model
digest, so every crosswalk ``wflow_id`` is checked against the digested gauge
map before it is trusted. A model without that layer, or whose layer lacks the
registry columns, keeps its native ids. The native CSVs are never rewritten.
"""

from __future__ import annotations

import json
from pathlib import Path

#: The model's outlet point map. Its values are subbasin ids, not wflow_ids.
OUTLETS_MAP = "outlets"

#: Registry columns a gauge layer must carry to act as the crosswalk.
_CROSSWALK_COLUMNS = ("subbasin_id", "is_primary", "wflow_id")


def _point_cells(values):
    """``{id: (row, col)}`` when ``values`` is a point map, else None."""
    import numpy as np

    ids, counts = np.unique(
        values[np.isfinite(values) & (values > 0)], return_counts=True
    )
    if len(ids) == 0 or counts.max() != 1:
        return None  # an area map (subcatchment), not points
    return {
        int(value): tuple(int(i) for i in np.argwhere(values == value)[0])
        for value in ids
    }


def primary_wflow_ids(geoms_dir, gauge_maps, point_ids) -> dict[int, int]:
    """``{subbasin_id: primary wflow_id}`` from the model's gauge layers.

    ``point_ids`` maps each gauge map to the ids ``staticmaps.nc`` burns into it;
    a primary absent from its map is a layer that no longer describes this
    model, and raises rather than labelling a series with it.
    """
    primary = {}
    for name in gauge_maps:
        path = Path(geoms_dir) / f"{name}.geojson"
        if not path.is_file():
            continue
        features = json.loads(path.read_text(encoding="utf-8"))["features"]
        rows = [feature["properties"] for feature in features]
        if not rows or any(
            column not in row for row in rows for column in _CROSSWALK_COLUMNS
        ):
            continue
        for row in rows:
            if not row["is_primary"]:
                continue
            subbasin, wflow_id = int(row["subbasin_id"]), int(row["wflow_id"])
            if wflow_id not in point_ids[name]:
                raise ValueError(
                    f"{path.name} names wflow_id {wflow_id} primary for subbasin "
                    f"{subbasin}, but staticmaps map {name!r} has no such point"
                )
            if primary.setdefault(subbasin, wflow_id) != wflow_id:
                raise ValueError(
                    f"subbasin {subbasin} has two primary wflow_ids: "
                    f"{primary[subbasin]} and {wflow_id}"
                )
    return primary


def native_location_labels(
    columns, headers, static_path, dropped=None
) -> dict[str, str]:
    """``{native header: location id}`` for every mapped column kept.

    ``columns`` is the TOML's ``output.csv.column`` list, ``headers`` Wflow's
    ordered header names. A mapped header missing from the result is a
    same-cell duplicate of one that is kept; ``dropped``, when given, is filled
    with ``{that header: the label it is kept under}``. Headers of unmapped
    columns are not keys. Raises when two kept columns of one variable end on
    one label; warns when an outlet or area id has no primary wflow_id to take.
    """
    import numpy as np
    import xarray as xr

    maps_of = {}
    for item in columns:
        if "map" in item:
            maps_of.setdefault(item["header"], []).append(item["map"])
    points, area_maps = {}, set()
    with xr.open_dataset(static_path) as dataset:
        for name in {name for maps in maps_of.values() for name in maps}:
            cells = _point_cells(np.asarray(dataset[name].values))
            if cells is None:
                area_maps.add(name)
            else:
                points[name] = cells
    gauge_maps = sorted(name for name in points if name != OUTLETS_MAP)
    primary = primary_wflow_ids(
        Path(static_path).parent / "staticgeoms", gauge_maps, points
    )

    # Candidates in precedence order: gauges first, then outlets, then areas,
    # each in Wflow's header order. A point cell already taken is a duplicate.
    order = {name: i for i, name in enumerate(headers)}
    candidates = []
    for header, maps in maps_of.items():
        for name in maps:
            rank = 2 if name in area_maps else 1 if name == OUTLETS_MAP else 0
            if name in points:
                natives = list(points[name])
            else:
                suffixes = (
                    key[len(header) + 1 :]
                    for key in headers
                    if key.startswith(header + "_")
                )
                natives = [int(s) for s in suffixes if s.isdigit()]
            candidates += [(rank, header, name, native) for native in natives]
    labels, taken, used, unresolved = {}, {}, set(), set()
    dropped = {} if dropped is None else dropped
    for rank, header, name, native in sorted(
        candidates,
        key=lambda c: (c[0], order.get(f"{c[1]}_{c[3]}", len(order))),
    ):
        key = f"{header}_{native}"
        if key not in order or key in labels:
            continue
        if name in points:
            cell = (header, points[name][native])
            if cell in taken:
                dropped[key] = taken[cell]
                continue
        if rank and native not in primary:
            unresolved.add(key)
        label = str(native if rank == 0 else primary.get(native, native))
        if name in points:
            taken[cell] = label
        if rank == 1 and (header, label) in used:
            # The outlet's primary is a gauge off the outlet cell (a snapped
            # location a cell away): that gauge IS the subbasin's discharge
            # point, so the outlet column is the duplicate.
            dropped[key] = label
            continue
        used.add((header, label))
        labels[key] = label
    if unresolved and gauge_maps:
        from blueearth_cst.shared.snake_utils import listed, log_row

        log_row(
            "No primary wflow_id in the model's gauge layer for "
            f"{listed(sorted(unresolved))}; these keep their native subbasin id",
            module="locations",
            level="WARNING",
        )
    for header in maps_of:
        kept = [
            label for key, label in labels.items() if key.rpartition("_")[0] == header
        ]
        if len(set(kept)) != len(kept):
            raise ValueError(
                f"{header} columns collide after relabelling to wflow_id: {kept}"
            )
    return labels
