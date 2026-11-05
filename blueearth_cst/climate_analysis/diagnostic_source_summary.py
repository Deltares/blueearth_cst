"""Retained WF0 extracted-source summary; independent of diagnostic plot periods."""

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from blueearth_cst.shared.snake_utils import PRECIP_ONLY_SOURCES, log_row

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr

TABLE_STEM = "dataset_comparison"
DISPLAY_HEADERS = {
    "source": "Dataset",
    "temporal_resolution": "Time step",
    "time_window": "Time window",
    "spatial_resolution": "Grid size",
    "reference": "Reference",
}
FOOTNOTE_COLUMN = "remarks"
MISSING = "—"
_SPATIAL_DIM_PAIRS = (("latitude", "longitude"), ("y", "x"), ("lat", "lon"))


def _attr(ds: "xr.Dataset", name: str) -> str:
    value = ds.attrs.get(name)
    if value is None or (isinstance(value, str) and (not value.strip())):
        return MISSING
    return str(value).strip()


def _temporal_resolution(ds: "xr.Dataset") -> str:
    import numpy as np
    import pandas as pd

    time = pd.DatetimeIndex(ds["time"].values)
    if time.size < 2:
        return MISSING
    step = pd.Series(time).diff().dropna().mode()
    if step.empty:
        return MISSING
    days = pd.Timedelta(step.iloc[0]).total_seconds() / 86400.0
    if np.isclose(days, 1.0):
        return "daily"
    if days < 1.0:
        hours = days * 24.0
        return f"{hours:g}-hourly"
    if 28.0 <= days <= 31.0:
        return "monthly"
    return f"{days:g}-daily"


def _spatial_dims(ds: "xr.Dataset") -> Optional[tuple]:
    for pair in _SPATIAL_DIM_PAIRS:
        if all((dim in ds.dims for dim in pair)):
            return pair
    return None


def _grid_step(ds: "xr.Dataset", dim: str) -> Optional[float]:
    import numpy as np

    if dim not in ds.coords or ds[dim].size < 2:
        return None
    return float(np.median(np.abs(np.diff(ds[dim].values))))


def _spatial_resolution(ds: "xr.Dataset") -> str:
    import numpy as np

    dims = _spatial_dims(ds)
    if dims is None:
        return MISSING
    lat, lon = (_grid_step(ds, dim) for dim in dims)
    if lat is None or lon is None:
        return MISSING
    if np.isclose(lat, lon, rtol=0.001):
        return f"{lat:g}°"
    return f"{lat:g}° × {lon:g}°"


def _time_window(ds: "xr.Dataset") -> str:
    import pandas as pd

    time = pd.DatetimeIndex(ds["time"].values)
    if not time.size:
        return MISSING
    return f"{time.min().date().isoformat()} → {time.max().date().isoformat()}"


def _remarks(source: str, ds: "xr.Dataset") -> str:
    parts = []
    note = _attr(ds, "notes")
    if note != MISSING:
        parts.append(note)
    if source in PRECIP_ONLY_SOURCES:
        parts.append("precipitation only")
        if "temp" in ds:
            parts.append("ERA5 temperature companion is present")
        else:
            parts.append("no ERA5 companion fields; promotion requires re-extraction")
    return "; ".join(parts) if parts else MISSING


def summarize_sources(stores: Mapping) -> "pd.DataFrame":
    """Describe extracted spans and WG-1 conformance, not scientific suitability."""
    if len(stores) < 2:
        raise ValueError("A comparison requires at least two sources")
    import pandas as pd
    import xarray as xr

    from blueearth_cst.shared.interchange_contracts import validate_wg1

    rows = []
    for source, path in stores.items():
        with xr.open_dataset(path) as ds:
            diffs = validate_wg1(ds)
            rows.append(
                {
                    "source": source,
                    "temporal_resolution": _temporal_resolution(ds),
                    "time_window": _time_window(ds),
                    "spatial_resolution": _spatial_resolution(ds),
                    "reference": _attr(ds, "paper_ref"),
                    FOOTNOTE_COLUMN: _remarks(source, ds),
                    "wg1_status": "not ready" if diffs else "ready",
                    "wg1_diffs": "\n".join(diffs),
                }
            )
    return pd.DataFrame(rows)


def write_comparison_table(table: "pd.DataFrame", out_dir: Union[str, Path]) -> list:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{TABLE_STEM}.csv"
    md_path = out_dir / f"{TABLE_STEM}.md"
    table.to_csv(csv_path, index=False)
    columns = [name for name in DISPLAY_HEADERS if name in table.columns]
    rendered = table[columns].copy()
    rendered.columns = [DISPLAY_HEADERS[name] for name in columns]
    notes = ""
    if FOOTNOTE_COLUMN in table.columns:
        lines = [
            f"- **{row['source']}** — {row[FOOTNOTE_COLUMN]}"
            for _, row in table.iterrows()
            if row[FOOTNOTE_COLUMN] and row[FOOTNOTE_COLUMN] != MISSING
        ]
        notes = "\n" + "\n".join(lines) + "\n" if lines else ""
    if "wg1_status" in table.columns:
        notes += "\n## WG-1 readiness of extracted stores\n\n"
        notes += "Readiness checks the existing full forcing contract; it does not block available-variable comparisons. Precipitation-only candidates require re-extraction with ERA5 companions when selected. This is structural and metadata conformance, not a check of record length, missing values or scientific suitability.\n\n"
        for _, row in table.iterrows():
            notes += f"- **{row['source']}** — {row['wg1_status']}\n"
            for diff in row["wg1_diffs"].splitlines():
                notes += f"  - {diff}\n"
    md_path.write_text(
        "# Gridded climate datasets compared\n\n"
        + rendered.to_markdown(index=False)
        + "\n"
        + notes,
        encoding="utf-8",
    )
    log_row(
        f"Source overview written for {len(table)} datasets → {csv_path.name}, {md_path.name}",
        module="compare",
    )
    return [csv_path, md_path]
