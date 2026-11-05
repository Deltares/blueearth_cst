"""Draw prepared geographic rasters without workflow IO or scientific preparation.

The map template (``blueearth_cst/plotting/docs/cst-style.md``, Spatial map template).
The caller supplies common frame, ticks, resolved colors and loaded GeoJSON
features (``basins``, ``divides``, ``rivers``, ``locations``), and owns style
context, captions, backend selection and export. Feature styling comes from the
active preset's ``maps`` tokens.
"""

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.path import Path
from matplotlib.patheffects import Normal, Stroke
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Bbox
from numpy.typing import NDArray

from blueearth_cst.plotting.fundamentals import style_colourbar
from blueearth_cst.plotting.location_labels import location_label, wrap_label
from blueearth_cst.plotting.preset import active

#: Stream-order attribute names, in lookup order (hydromt spells it ``strord``).
RIVER_ORDER_KEYS = ("order", "strord")
#: Kilometres per degree of latitude; longitude scales by cos(latitude).
KM_PER_DEGREE = 111.32

# Side panel geometry, in map-axes fractions (>1 lies right of the frame).
PANEL_LEFT = 1.03
PANEL_GAP = 0.04
#: Locator inset width in the side panel, and inside the map on overview maps;
#: its height keeps it square on the page.
PANEL_LOCATOR_WIDTH = 0.22
MAP_LOCATOR_WIDTH, MAP_LOCATOR_MAX_HEIGHT = 0.28, 0.4
#: Colourbar length, its floor when the locator and title crowd it, and the
#: room its two-line title takes above it.
COLORBAR_HEIGHT, COLORBAR_MIN, COLORBAR_TITLE_ROOM = 0.7, 0.25, 0.12
#: Smallest key text a class legend shrinks to before warning (pt).
MIN_KEY_FONT = 4.0
MISSING_LABEL = "No data"

# Maximum physical map frame; keys and coordinate labels extend the canvas.
MAP_WIDTH_MM, MAP_HEIGHT_MM = 150.0, 95.0
MAP_PADDING_MM = 3.0


def _map_figure(frame: "MapFrame") -> tuple[Figure, Axes]:
    """Fit the geographic aspect inside a common physical map envelope."""
    west, east, south, north = frame.extent
    ratio = frame.aspect * (north - south) / (east - west)
    width = min(MAP_WIDTH_MM, MAP_HEIGHT_MM / ratio) / 25.4
    height = width * ratio
    fig = plt.figure(figsize=(width + 1, height + 1), layout="none")
    ax = fig.add_axes(
        (
            0.5 / (width + 1),
            0.5 / (height + 1),
            width / (width + 1),
            height / (height + 1),
        )
    )
    return fig, ax


def _fit_map_canvas(fig: Figure, ax: Axes) -> None:
    """Measure decorations and extend the canvas without resizing the map."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    frame = ax.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
    bounds = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
    padding = MAP_PADDING_MM / 25.4
    width, height = bounds.width + 2 * padding, bounds.height + 2 * padding
    fig.set_size_inches(width, height)
    ax.set_position(
        (
            (frame.x0 - bounds.x0 + padding) / width,
            (frame.y0 - bounds.y0 + padding) / height,
            frame.width / width,
            frame.height / height,
        )
    )
    fig.canvas.draw()


@dataclass(frozen=True)
class MapFrame:
    """Common west/east/south/north viewport and explicit geographic display."""

    extent: tuple[float, float, float, float]
    longitude_ticks: Sequence[float]
    latitude_ticks: Sequence[float]
    aspect: float


@dataclass(frozen=True)
class MapScale:
    """Resolved continuous classes; unavailable scales have no boundaries/ticks.

    ``extend`` names the colourbar's rectangular extensions (``both``, ``min``,
    ``max`` or ``neither``); with ``neither`` values beyond the outer
    boundaries fall into the end classes.
    """

    units: str
    boundaries: Sequence[float]
    ticks: Sequence[float]
    colors: Sequence[Any]
    under_color: Any
    over_color: Any
    missing_color: Any
    status: str
    extend: str = "both"


@dataclass(frozen=True)
class MapClasses:
    """Nominal classes: raster values are ``codes``, keyed by ``labels``.

    Only classes present in the drawn area enter the key, and of those only
    classes covering at least ``key_min_share`` of the drawn classified cells
    (smaller ones are still drawn; ``unkeyed_classes`` names them for the
    caption). Codes the table does not declare draw in ``unclassified_color``
    and are keyed as such.
    """

    title: str
    codes: Sequence[float]
    colors: Sequence[Any]
    labels: Sequence[str]
    missing_color: Any
    unclassified_color: Any = "#bdbdbd"
    unclassified_label: str = "Unclassified"
    alpha: float = 1.0
    key_min_share: float = 0.0


@dataclass(frozen=True)
class LocatorMap:
    """A "where is this" inset from supplied basemap features.

    ``window`` is west/east/south/north; ``land`` holds polygon features,
    ``borders`` line features and ``places`` ``(x, y, name)`` tuples. The main
    map's frame is outlined on it.
    """

    window: tuple[float, float, float, float]
    land: Sequence[Mapping[str, Any]] = ()
    borders: Sequence[Mapping[str, Any]] = ()
    places: Sequence[tuple[float, float, str]] = ()


@dataclass(frozen=True)
class RasterMap:
    """Native latitude × longitude values and one-more-element cell-edge arrays.

    ``rgba`` optionally replaces the classified fill with an adapter-prepared
    image of the same rows and columns (a hillshaded relief), drawn over the
    classes; transparent pixels show the classes or missing colour beneath.
    The key still comes from the scale.
    """

    values: NDArray[np.number]
    longitude_edges: NDArray[np.number]
    latitude_edges: NDArray[np.number]
    inside: NDArray[np.bool_] | None = None
    cell_edges: bool = True
    rgba: NDArray[np.floating] | None = None


class _ClosedUpperBoundaryNorm(BoundaryNorm):
    """Keep the final endpoint in the last class; strict excess stays over-range."""

    def __call__(self, value: Any, clip: bool | None = None) -> Any:
        result = super().__call__(value, clip)
        closed = np.ma.where(
            np.ma.asarray(value) == self.boundaries[-1], self.Ncmap - 1, result
        )
        return closed[()] if np.ndim(value) == 0 else closed


def _numeric_vector(value: Any, name: str) -> NDArray[np.number]:
    array = np.asanyarray(value)
    if (
        array.ndim != 1
        or not np.issubdtype(array.dtype, np.number)
        or np.iscomplexobj(array)
        or not np.isfinite(array).all()
        or np.ma.getmaskarray(array).any()
    ):
        raise ValueError(f"{name} must be a finite unmasked numeric vector")
    return array


def _validate(
    raster: RasterMap, frame: MapFrame, scale: "MapScale | MapClasses"
) -> None:
    values = np.asanyarray(raster.values)
    if (
        values.ndim != 2
        or 0 in values.shape
        or not np.issubdtype(values.dtype, np.number)
        or np.iscomplexobj(values)
    ):
        raise ValueError("Raster values must be a nonempty numeric 2D array")
    for edges, name, length in (
        (raster.longitude_edges, "longitude edges", values.shape[1] + 1),
        (raster.latitude_edges, "latitude edges", values.shape[0] + 1),
    ):
        array = _numeric_vector(edges, name)
        if len(array) != length:
            raise ValueError(f"{name} must contain {length} values")
        steps = np.diff(array.astype(np.float64))
        precision = (
            2 * float(np.max(np.abs(np.spacing(array))))
            if np.issubdtype(array.dtype, np.floating)
            else 0
        )
        if not (np.all(steps > 0) or np.all(steps < 0)) or not np.allclose(
            steps, steps[0], rtol=0, atol=precision
        ):
            raise ValueError(f"{name} must describe a regular monotonic grid")
    if raster.inside is not None:
        inside = np.asanyarray(raster.inside)
        if (
            inside.shape != values.shape
            or inside.dtype != np.dtype(bool)
            or np.ma.getmaskarray(inside).any()
        ):
            raise ValueError("inside must be an unmasked boolean array matching values")
    if not isinstance(raster.cell_edges, bool):
        raise ValueError("cell_edges must be boolean")
    if raster.rgba is not None and np.shape(raster.rgba) != (*values.shape, 4):
        raise ValueError("rgba must be a rows × columns × 4 array matching values")
    extent = _numeric_vector(frame.extent, "frame extent")
    if len(extent) != 4:
        raise ValueError("Frame extent must contain west/east/south/north")
    west, east, south, north = extent
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("Frame must be geographic and not cross the antimeridian")
    if not np.isfinite(frame.aspect) or frame.aspect <= 0:
        raise ValueError("Frame aspect must be finite and positive")
    for ticks, name, low, high in (
        (frame.longitude_ticks, "longitude ticks", west, east),
        (frame.latitude_ticks, "latitude ticks", south, north),
    ):
        array = _numeric_vector(ticks, name)
        if not np.all(np.diff(array) > 0) or np.any((array < low) | (array > high)):
            raise ValueError(f"{name} must increase within the closed frame")
    if isinstance(scale, MapClasses):
        codes = _numeric_vector(scale.codes, "class codes")
        if (
            len(set(codes.tolist())) != len(codes)
            or len(scale.colors) != len(codes)
            or len(scale.labels) != len(codes)
        ):
            raise ValueError("Classes need distinct codes, one colour and label each")
        for color in (*scale.colors, scale.missing_color, scale.unclassified_color):
            to_rgba(color)
        return
    if not isinstance(scale.units, str) or not scale.units:
        raise ValueError("Map scale units must be a nonempty string")
    if scale.extend not in {"both", "min", "max", "neither"}:
        raise ValueError("Map scale extend must be both, min, max or neither")
    if scale.status not in {"available", "unavailable"}:
        raise ValueError("Map scale status must be available or unavailable")
    boundaries = _numeric_vector(scale.boundaries, "scale boundaries")
    ticks = _numeric_vector(scale.ticks, "scale ticks")
    for color in (
        *scale.colors,
        scale.under_color,
        scale.over_color,
        scale.missing_color,
    ):
        to_rgba(color)
    if scale.status == "unavailable":
        if len(boundaries) or len(ticks) or len(scale.colors):
            raise ValueError("Unavailable scales must have no classes or ticks")
    elif (
        len(boundaries) < 2
        or not np.all(np.diff(boundaries) > 0)
        or len(scale.colors) != len(boundaries) - 1
        or not np.all(np.diff(ticks) > 0)
        or np.any((ticks < boundaries[0]) | (ticks > boundaries[-1]))
    ):
        raise ValueError(
            "Available scale needs increasing boundaries, one colour per class "
            "and increasing ticks within the boundaries"
        )


def _geometry_lines(geometry: Mapping[str, Any]) -> Sequence[Any]:
    kind, coordinates = geometry["type"], geometry["coordinates"]
    if kind == "Polygon":
        return coordinates
    if kind == "MultiPolygon":
        return [ring for polygon in coordinates for ring in polygon]
    if kind == "LineString":
        return [coordinates]
    return coordinates if kind == "MultiLineString" else []


def _river_order(properties: Mapping[str, Any]) -> float | None:
    for key in RIVER_ORDER_KEYS:
        value = properties.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            return float(value)
    return None


def _draw_rivers(ax: Axes, rivers: Sequence[Mapping[str, Any]]) -> None:
    """Blue reaches stepped by stream order; thin reaches first, round joins.

    The layer's lowest order takes the first step and orders past the last
    step reuse it. A layer without an order attribute draws one uniform weight.
    """
    maps = active().maps
    orders = [_river_order(f.get("properties") or {}) for f in rivers]
    known = [order for order in orders if order is not None]
    lowest = min(known) if known else 0.0
    steps = maps["river_order"]
    styled = []
    for feature, order in zip(rivers, orders):
        if order is None:
            width, color = maps["river_width"], maps["river"]
        else:
            width, color = steps[min(max(int(order - lowest), 0), len(steps) - 1)]
        styled.append((width, color, feature))
    # A thin white halo keeps reaches legible over any ramp, blue ones included.
    halo = maps["halo"]
    for width, color, feature in sorted(styled, key=lambda item: item[0]):
        for coordinates in _geometry_lines(feature["geometry"]):
            points = np.asarray(coordinates)
            ax.plot(
                points[:, 0],
                points[:, 1],
                color=color,
                lw=width,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=3,
                path_effects=[
                    Stroke(
                        linewidth=width + maps["river_halo_width"],
                        foreground=halo,
                        capstyle="round",
                        joinstyle="round",
                    ),
                    Normal(),
                ],
            )


def _draw_overlays(
    ax: Axes, overlays: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[tuple[float, float, str]]:
    """Draw rivers, divides, the basin outline and locations; return labels."""
    maps = active().maps
    halo = maps["halo"]
    _draw_rivers(ax, overlays.get("rivers", []))
    for key, color, width, dashes, zorder in (
        ("divides", maps["divide"], maps["divide_width"], maps["divide_dash"], 4),
        ("basins", maps["basin_outline"], maps["basin_outline_width"], None, 4.5),
    ):
        for feature in overlays.get(key, []):
            for coordinates in _geometry_lines(feature["geometry"]):
                points = np.asarray(coordinates)
                (artist,) = ax.plot(
                    points[:, 0], points[:, 1], color=color, lw=width, zorder=zorder
                )
                if dashes is not None:
                    artist.set_dashes(dashes)
                artist.set_path_effects(
                    [Stroke(linewidth=width + 1.0, foreground=halo), Normal()]
                )
    labels = []
    for feature in overlays.get("locations", []):
        geometry = feature["geometry"]
        if geometry["type"] != "Point":
            continue
        x, y = geometry["coordinates"][:2]
        ax.plot(
            x,
            y,
            maps["location_marker"],
            color=maps["location"],
            mec=maps["marker_edge"],
            mew=0.6,
            ms=maps["location_size"],
            zorder=5,
        )
        label = location_label(feature.get("properties") or {})
        if label:
            labels.append((x, y, label))
    return labels


def _nice_length(target_km: float) -> float:
    """The largest 1/2/2.5/5 x 10^k kilometres not above ``target_km``."""
    exponent = math.floor(math.log10(target_km))
    for step in (5, 2.5, 2, 1):
        if step * 10**exponent <= target_km:
            return step * 10**exponent
    return 10 ** (exponent - 1) * 5


def _draw_scale_bar(ax: Axes, frame: MapFrame) -> None:
    """Alternating kilometre segments in the lower left, latitude-corrected."""
    maps = active().maps
    preset = active()
    west, east, south, north = frame.extent
    km_per_degree = KM_PER_DEGREE * math.cos(math.radians((south + north) / 2))
    length_km = _nice_length(0.25 * (east - west) * km_per_degree)
    segments = maps["scale_bar_segments"]
    length = length_km / km_per_degree
    x0 = west + 0.04 * (east - west)
    y0 = south + 0.05 * (north - south)
    height = 0.012 * (north - south)
    ink = preset.colors["ink"]
    for index in range(segments):
        ax.add_patch(
            Rectangle(
                (x0 + index * length / segments, y0),
                length / segments,
                height,
                facecolor=ink if index % 2 == 0 else "white",
                edgecolor=ink,
                linewidth=0.5,
                zorder=6,
            )
        )
    halo = [Stroke(linewidth=2.0, foreground=maps["halo"]), Normal()]
    for step in (0, segments // 2, segments):
        value = length_km * step / segments
        ax.text(
            x0 + length * step / segments,
            y0 + height + 0.01 * (north - south),
            f"{value:g} km" if step == segments else f"{value:g}",
            ha="center",
            va="bottom",
            fontsize=preset.fonts["tick"],
            color=ink,
            zorder=6,
            path_effects=halo,
        )


def _place_point_labels(
    fig: Figure, ax: Axes, labels: Sequence[tuple[float, float, str]]
) -> None:
    """Haloed, optionally two-line labels at the clearest in-frame offset."""
    maps = active().maps
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    radius = 4 * fig.dpi / 72
    # An in-map locator is occupied ground; side-panel insets lie outside the frame.
    occupied = [
        box
        for child in ax.child_axes
        if ax.bbox.contains(*(box := child.get_window_extent(renderer)).p0 + 1)
    ]
    for x, y, _ in labels:
        px, py = ax.transData.transform((x, y))
        occupied.append(
            Bbox.from_extents(px - radius, py - radius, px + radius, py + radius)
        )
    candidates = (
        (8, 8),
        (-8, 8),
        (8, -8),
        (-8, -8),
        (16, 0),
        (-16, 0),
        (0, 16),
        (0, -16),
        (24, 16),
        (-24, 16),
        (24, -16),
        (-24, -16),
    )
    for x, y, label in labels:
        if not ax.get_xlim()[0] <= x <= ax.get_xlim()[1] or not (
            ax.get_ylim()[0] <= y <= ax.get_ylim()[1]
        ):
            continue
        text = ax.annotate(
            wrap_label(label, maps["location_label_wrap"]),
            (x, y),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=maps["location_label_size"],
            fontweight="bold",
            color=active().colors["ink"],
            zorder=6,
            path_effects=[Stroke(linewidth=2.2, foreground=maps["halo"]), Normal()],
            arrowprops={
                "arrowstyle": "-",
                "color": active().colors["muted"],
                "lw": 0.5,
                "shrinkA": 1.5,
                "shrinkB": 4,
                "path_effects": [
                    Stroke(linewidth=1.5, foreground=maps["halo"]),
                    Normal(),
                ],
            },
        )
        text.set_in_layout(False)
        options = []
        for dx, dy in candidates:
            text.set_position((dx, dy))
            align = "left" if dx > 0 else "right" if dx < 0 else "center"
            text.set_ha(align)
            text.set_multialignment(align)
            text.set_va("bottom" if dy > 0 else "top" if dy < 0 else "center")
            text.update_positions(renderer)
            box = text.get_window_extent(renderer).expanded(1.08, 1.15)
            outside = not (
                ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1)
            )
            score = 1000 * outside + sum(box.overlaps(other) for other in occupied)
            options.append((score, dx, dy, box))
        _, dx, dy, box = min(options, key=lambda option: option[0])
        text.set_position((dx, dy))
        align = "left" if dx > 0 else "right" if dx < 0 else "center"
        text.set_ha(align)
        text.set_multialignment(align)
        text.set_va("bottom" if dy > 0 else "top" if dy < 0 else "center")
        occupied.append(box)


def _polygon_paths(geometry: Mapping[str, Any]) -> list[Path]:
    """Compound paths (exterior plus holes) for a Polygon or MultiPolygon."""
    kind, coordinates = geometry["type"], geometry["coordinates"]
    polygons = [coordinates] if kind == "Polygon" else coordinates
    if kind not in {"Polygon", "MultiPolygon"}:
        return []
    paths = []
    for rings in polygons:
        vertices, codes = [], []
        for ring in rings:
            ring = np.asarray(ring, dtype=float)[:, :2]
            vertices.extend(ring.tolist())
            codes.extend(
                [Path.MOVETO, *[Path.LINETO] * (len(ring) - 2), Path.CLOSEPOLY]
            )
        paths.append(Path(vertices, codes))
    return paths


def _in_map_corner(
    ax: Axes,
    frame: MapFrame,
    overlays: Mapping[str, Sequence[Mapping[str, Any]]],
    width: float,
    height: float,
) -> tuple[float, float]:
    """Lower-left of the in-map corner box covering the least basin and locations.

    The lower-left corner is the scale bar's, so it is never offered. Ties keep
    the upper left.
    """
    west, east, south, north = frame.extent
    points = [
        point
        for key in ("basins", "locations")
        for feature in overlays.get(key, [])
        for line in (
            _geometry_lines(feature["geometry"])
            if feature["geometry"]["type"] != "Point"
            else [[feature["geometry"]["coordinates"]]]
        )
        for point in line
    ]
    xy = np.asarray(points, dtype=float)[:, :2] if points else np.empty((0, 2))
    fx = (xy[:, 0] - west) / (east - west)
    fy = (xy[:, 1] - south) / (north - south)
    corners = ((0.0, 1.0 - height), (1.0 - width, 1.0 - height), (1.0 - width, 0.0))
    return min(
        corners,
        key=lambda c: int(
            (
                (fx >= c[0])
                & (fx <= c[0] + width)
                & (fy >= c[1])
                & (fy <= c[1] + height)
            ).sum()
        ),
    )


def _draw_locator(
    ax: Axes,
    locator: LocatorMap,
    frame: MapFrame,
    top: float,
    overlays: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> float:
    """Draw the locator inset; return the side-panel height it took.

    In the side panel by default. With ``overlays`` it sits inside the map, in
    the clearest upper or lower-right corner, flush with the frame, and takes
    no panel height (the overview-map exception).
    """
    maps = active().maps
    west, east, south, north = frame.extent
    axes_ratio = frame.aspect * (north - south) / (east - west)
    if overlays is not None:
        # Square on the page and small: a wide basin's short frame caps the height.
        height = min(MAP_LOCATOR_WIDTH / axes_ratio, MAP_LOCATOR_MAX_HEIGHT)
        width = height * axes_ratio
        x0, y0 = _in_map_corner(ax, frame, overlays, width, height)
        inset = ax.inset_axes((x0, y0, width, height))
        inset.set_zorder(7)
        taken = 0.0
    else:
        width = PANEL_LOCATOR_WIDTH
        height = min(width / axes_ratio, top)
        inset = ax.inset_axes((PANEL_LEFT, top - height, width, height))
        taken = height
    w0, e0, s0, n0 = locator.window
    inset.set_facecolor(maps["locator_ocean"])
    for feature in locator.land:
        for path in _polygon_paths(feature["geometry"]):
            inset.add_patch(
                PathPatch(
                    path,
                    facecolor=maps["locator_land"],
                    edgecolor=maps["locator_coast"],
                    linewidth=0.35,
                )
            )
    for feature in locator.borders:
        for coordinates in _geometry_lines(feature["geometry"]):
            points = np.asarray(coordinates)
            inset.plot(points[:, 0], points[:, 1], color=maps["locator_border"], lw=0.3)
    halo = [Stroke(linewidth=1.6, foreground=maps["halo"]), Normal()]
    # Places arrive largest first; one whose label would hit an earlier one is
    # dropped rather than overprinted.
    inset.set_xlim(w0, e0)
    inset.set_ylim(s0, n0)
    renderer = ax.figure.canvas.get_renderer()
    placed: list[Bbox] = []
    for x, y, name in locator.places:
        (marker,) = inset.plot(x, y, "o", ms=1.8, color=maps["locator_city"])
        text = inset.annotate(
            name,
            (x, y),
            xytext=(2, 0),
            textcoords="offset points",
            fontsize=4.5,
            color=maps["locator_city"],
            va="center",
            path_effects=halo,
            annotation_clip=True,
        )
        box = Bbox.union(
            [text.get_window_extent(renderer), marker.get_window_extent(renderer)]
        ).expanded(1.05, 1.2)
        if any(box.overlaps(other) for other in placed):
            text.remove()
            marker.remove()
            continue
        placed.append(box)
    inset.add_patch(
        Rectangle(
            (west, south),
            east - west,
            north - south,
            fill=False,
            edgecolor=maps["locator_mark"],
            linewidth=0.9,
            zorder=5,
        )
    )
    # The box is square on the page, so the longitude span that fills it is the
    # latitude span times the midpoint-latitude aspect.
    aspect = 1 / math.cos(math.radians((s0 + n0) / 2))
    half = (n0 - s0) * aspect / 2
    centre = (w0 + e0) / 2
    inset.set_xlim(centre - half, centre + half)
    inset.set_ylim(s0, n0)
    inset.set_aspect(aspect, adjustable="box")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_linewidth(0.6)
        spine.set_edgecolor(active().colors["ink"])
    return taken


def _side_legend(ax: Axes, handles: list, title: str | None, fontsize: float) -> Any:
    """A key in the side panel, its lower-left corner on the frame's bottom."""
    preset = active()
    return ax.legend(
        handles=handles,
        title=title,
        loc="lower left",
        bbox_to_anchor=(PANEL_LEFT, 0.0),
        bbox_transform=ax.transAxes,
        borderaxespad=0.0,
        borderpad=0.0,
        frameon=False,
        fontsize=fontsize,
        title_fontsize=preset.fonts["tick"],
        alignment="left",
        handlelength=1.4,
        handleheight=1.0,
        labelcolor=preset.colors["ink"],
    )


def _fit_legend(fig: Figure, ax: Axes, handles, title, available: float) -> Any:
    """Shrink a side legend's text until it fits ``available`` of the frame height.

    Warns when it still does not fit at the smallest permitted size.
    """
    fontsize = active().fonts["tick"]
    legend = _side_legend(ax, handles, title, fontsize)
    for _ in range(6):
        fig.canvas.draw()
        height = legend.get_window_extent().height / ax.bbox.height
        if height <= available:
            return legend
        fontsize *= 0.97 * available / height
        if fontsize < MIN_KEY_FONT:
            break
        legend = _side_legend(ax, handles, title, fontsize)
    warnings.warn(
        f"Map key ({len(handles)} entries) exceeds the frame height", stacklevel=3
    )
    return legend


def _swatch(color: Any, label: str, alpha: float = 1.0) -> Patch:
    return Patch(
        facecolor=color,
        edgecolor=active().colors["axis"],
        linewidth=0.4,
        alpha=alpha,
        label=label,
    )


def _class_shares(
    raster: RasterMap, frame: MapFrame, classes: MapClasses
) -> tuple[np.ma.MaskedArray, dict[int, float]]:
    """Class-index raster and each drawn class's share of drawn classified cells.

    Index ``len(codes)`` stands for undeclared codes.
    """
    values = np.ma.masked_invalid(raster.values)
    codes = np.asarray(classes.codes, dtype=float)
    index = np.full(values.shape, len(codes), dtype=float)
    for position, code in enumerate(codes):
        index[np.ma.filled(values, np.nan) == code] = position
    index = np.ma.array(index, mask=np.ma.getmaskarray(values))
    inside = np.ones(values.shape, bool) if raster.inside is None else raster.inside
    drawn = index[inside & _in_frame(raster, frame) & ~index.mask].astype(int)
    found, counts = np.unique(drawn, return_counts=True)
    total = max(int(counts.sum()), 1)
    return index, {int(i): c / total for i, c in zip(found, counts)}


def unkeyed_classes(
    raster: RasterMap, frame: MapFrame, classes: MapClasses
) -> list[str]:
    """Labels of drawn classes left out of the key for covering too little area."""
    _, shares = _class_shares(raster, frame, classes)
    return [
        classes.labels[i] if i < len(classes.codes) else classes.unclassified_label
        for i, share in sorted(shares.items())
        if share < classes.key_min_share
    ]


def _in_frame(raster: RasterMap, frame: MapFrame) -> NDArray[np.bool_]:
    """Cells whose footprint intersects the frame."""
    west, east, south, north = frame.extent
    x, y = np.asarray(raster.longitude_edges), np.asarray(raster.latitude_edges)
    xs = (np.maximum(x[:-1], x[1:]) > west) & (np.minimum(x[:-1], x[1:]) < east)
    ys = (np.maximum(y[:-1], y[1:]) > south) & (np.minimum(y[:-1], y[1:]) < north)
    return ys[:, None] & xs[None, :]


def missing_shown(raster: RasterMap, frame: MapFrame) -> bool:
    """Whether the map draws missing cells, which is when it keys "No data"."""
    inside = (
        np.ones(np.shape(raster.values), dtype=bool)
        if raster.inside is None
        else np.asarray(raster.inside)
    )
    missing = ~np.isfinite(np.asarray(raster.values, dtype=float))
    return bool((inside & _in_frame(raster, frame) & missing).any())


def draw_raster_map(
    raster: RasterMap,
    frame: MapFrame,
    scale: MapScale | MapClasses,
    overlays: Mapping[str, Sequence[Mapping[str, Any]]],
    colorbar_title: str | None = None,
    locator: LocatorMap | None = None,
    show_key: bool = True,
    locator_in_map: bool = False,
) -> Figure:
    """Draw supplied classes or honest unscaled hatching; never change input arrays.

    A ``MapScale`` draws a classified ramp with its colourbar; ``MapClasses`` a
    nominal raster with a class legend. Finite unscaled cells receive neutral
    hatching; missing cells take the missing colour and are keyed "No data" when
    any are drawn; cells outside ``inside`` stay white. The side panel stacks
    the optional ``locator`` at the top and the key at the bottom;
    ``locator_in_map`` instead places it inside the map's clearest corner (the
    overview maps' exception). Geographic grids use the supplied ticks/aspect.
    ``colorbar_title`` defaults to the scale's units; ``show_key=False`` omits
    the key. The map frame fits within 150 × 95 mm at its geographic aspect;
    labels, keys and external locators extend the canvas without shrinking it.
    """
    _validate(raster, frame, scale)
    fig, ax = _map_figure(frame)
    values = np.ma.masked_invalid(raster.values)
    inside = (
        np.ones(values.shape, dtype=bool)
        if raster.inside is None
        else np.asarray(raster.inside)
    )
    missing = missing_shown(raster, frame)
    classes = scale if isinstance(scale, MapClasses) else None
    present: list[int] = []
    if classes is not None:
        codes = np.asarray(classes.codes, dtype=float)
        index, shares = _class_shares(raster, frame, classes)
        present = sorted(
            i for i, share in shares.items() if share >= classes.key_min_share
        )
        cmap = ListedColormap([*classes.colors, classes.unclassified_color])
        cmap.set_bad(classes.missing_color)
        norm = BoundaryNorm(np.arange(len(codes) + 2) - 0.5, cmap.N)
        displayed = index
    elif scale.status == "available":
        cmap = ListedColormap(scale.colors)
        cmap.set_under(scale.under_color)
        cmap.set_over(scale.over_color)
        cmap.set_bad(scale.missing_color)
        clip = scale.extend == "neither"
        norm = _ClosedUpperBoundaryNorm(scale.boundaries, cmap.N, clip=clip)
        displayed = values
    else:
        cmap = ListedColormap(["#eeeeee"])
        cmap.set_bad(scale.missing_color)
        norm = None
        displayed = np.ma.array(np.zeros(values.shape), mask=np.ma.getmaskarray(values))
    mesh = ax.pcolormesh(
        raster.longitude_edges,
        raster.latitude_edges,
        displayed,
        cmap=cmap,
        norm=norm,
        shading="flat",
        edgecolors=(0.25, 0.25, 0.25, 0.12) if raster.cell_edges else "none",
        linewidth=0.25 if raster.cell_edges else 0,
        alpha=classes.alpha if classes is not None else None,
    )
    if raster.rgba is not None:
        x, y = np.asarray(raster.longitude_edges), np.asarray(raster.latitude_edges)
        north_first = y[0] > y[-1]
        ax.imshow(
            raster.rgba,
            extent=(x[0], x[-1], y[-1], y[0])
            if north_first
            else (x[0], x[-1], y[0], y[-1]),
            origin="upper" if north_first else "lower",
            interpolation="none",
            aspect="auto",
            zorder=1,
        )
    if classes is None and scale.status == "unavailable":
        xedges, yedges = raster.longitude_edges, raster.latitude_edges
        for i, j in np.argwhere(inside & ~np.ma.getmaskarray(values)):
            ax.add_patch(
                Rectangle(
                    (min(xedges[j : j + 2]), min(yedges[i : i + 2])),
                    abs(xedges[j + 1] - xedges[j]),
                    abs(yedges[i + 1] - yedges[i]),
                    facecolor="none",
                    edgecolor="#b0b0b0",
                    linewidth=0,
                    hatch="///",
                )
            )
    if raster.inside is not None:
        ax.pcolormesh(
            raster.longitude_edges,
            raster.latitude_edges,
            np.ma.masked_where(inside, np.zeros(values.shape)),
            cmap=ListedColormap(["white"]),
            shading="flat",
            edgecolors="none",
            linewidth=0,
        )
    labels = _draw_overlays(ax, overlays)
    west, east, south, north = frame.extent
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect(frame.aspect)
    ax.set_xticks(frame.longitude_ticks)
    ax.set_yticks(frame.latitude_ticks)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    formatter = FuncFormatter(lambda value, _: f"{value:g}".replace("-", "−"))
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.grid(color="#777777", alpha=0.15, linewidth=0.4)
    _draw_scale_bar(ax, frame)

    # --- side panel: locator on top, key on the frame's bottom ---------------
    top = 1.0
    if locator is not None and locator_in_map:
        _draw_locator(ax, locator, frame, top, overlays)
    elif locator is not None:
        top -= _draw_locator(ax, locator, frame, top) + PANEL_GAP
    missing_color = scale.missing_color
    if show_key and classes is not None:
        handles = [
            _swatch(
                classes.colors[i]
                if i < len(classes.codes)
                else classes.unclassified_color,
                classes.labels[i]
                if i < len(classes.codes)
                else classes.unclassified_label,
                classes.alpha,
            )
            for i in present
        ]
        if missing:
            handles.append(_swatch(missing_color, MISSING_LABEL))
        _fit_legend(fig, ax, handles, classes.title, top)
    elif show_key and scale.status == "available":
        bottom = 0.0
        if missing:
            swatch = _side_legend(
                ax,
                [_swatch(missing_color, MISSING_LABEL)],
                None,
                active().fonts["tick"],
            )
            fig.canvas.draw()
            bottom = swatch.get_window_extent().height / ax.bbox.height + PANEL_GAP
        # Placed in the map axes' own fractions rather than through
        # ``fig.colorbar(ax=ax)``: the locked aspect leaves the axes shorter than
        # its layout cell, so a cell-anchored bar overhangs the frame bottom.
        # Width keeps matplotlib's default 20:1 bar in physical units; the bar
        # gives way to the locator and its own title.
        height = max(
            min(COLORBAR_HEIGHT, top - bottom - COLORBAR_TITLE_ROOM), COLORBAR_MIN
        )
        axes_ratio = frame.aspect * (north - south) / (east - west)
        cax = ax.inset_axes((PANEL_LEFT, bottom, height / 20 * axes_ratio, height))
        colorbar = fig.colorbar(
            mesh,
            cax=cax,
            ticks=scale.ticks,
            extend=scale.extend,
            extendrect=True,
        )
        style_colourbar(
            colorbar, scale.units if colorbar_title is None else colorbar_title
        )
    elif show_key and missing:
        _side_legend(
            ax, [_swatch(missing_color, MISSING_LABEL)], None, active().fonts["tick"]
        )
    _place_point_labels(fig, ax, labels)
    _fit_map_canvas(fig, ax)
    return fig
