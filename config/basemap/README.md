# config/basemap/

`natural_earth_50m.gpkg` — the only geographic data committed to this
repository. It exists so the basin map's **locator inset** can say where in the
world a basin is without a network call.

## Why it is here and not fetched

The basin figure used to pull live satellite tiles at render time. That was
removed (see `blueearth_cst/shared/plot_map.py`) because a network dependency
inside WF1 is a licence question, a reproducibility question, and a failure
mode on an offline machine. Cartopy's own `cartopy.feature.COASTLINE` has the
same problem: it downloads Natural Earth on first use and caches it per user,
so a fresh machine hits the network and two machines can disagree.

Vendoring the three layers the inset actually draws costs 1.8 MB and removes
the question. This is a static cartographic asset, like a font — **not** an
exception to the rule that real basin data lives outside the repository.

## Provenance

| | |
|---|---|
| Source | Natural Earth, 1:50m scale — <https://www.naturalearthdata.com/> |
| Layers | `ne_50m_land`, `ne_50m_admin_0_boundary_lines_land`, `ne_50m_populated_places_simple` |
| Licence | Public domain. No attribution required, though the project credits it. |
| CRS | EPSG:4326 |
| Retrieved | 2026-08-03 from `https://naciscdn.org/naturalearth/50m/...` |

## Layers in the GeoPackage

| Layer | Geometry | Features | Columns kept |
|---|---|---|---|
| `land` | Polygon | 1420 | geometry only — its boundary IS the coastline, so no separate coastline layer is stored |
| `borders` | LineString | 390 | geometry only |
| `places` | Point | 1251 | `name`, `pop_max`, `scalerank` |

`scalerank` is Natural Earth's own prominence ranking, 0 being most prominent;
`plot_map.py` filters on it so a locator does not fill with town names.

## Rebuilding it

Only needed to move to a different Natural Earth release. Download the three
zips from the URL above, unpack, then:

```python
import geopandas as gpd
from pathlib import Path

ne, out = Path("<unpacked>"), Path("config/basemap/natural_earth_50m.gpkg")
out.unlink(missing_ok=True)
gpd.read_file(ne / "ne_50m_land.shp")[["geometry"]].to_file(
    out, layer="land", driver="GPKG")
gpd.read_file(ne / "ne_50m_admin_0_boundary_lines_land.shp")[["geometry"]].to_file(
    out, layer="borders", driver="GPKG", mode="a")
gpd.read_file(ne / "ne_50m_populated_places_simple.shp")[
    ["name", "pop_max", "scalerank", "geometry"]].to_file(
    out, layer="places", driver="GPKG", mode="a")
```

Dropping the unused columns is what keeps this to 1.8 MB; the full
`populated_places` carries 38 of them.
