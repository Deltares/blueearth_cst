"""Inspect ERA5 Zarr geometry and model representative basin reads.

The script is report-only: it opens metadata and chunk files read-only and does
not create a rechunked store. Candidate byte estimates preserve each variable's
sampled bytes-per-cell ratio; they are geometry models, not measured candidate
stores.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import threading
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

WF0_VARIABLES = ("t2m", "tmin", "tmax", "tp", "ssrd", "tisr", "msl")
WINDOWS = {
    "ntoum": (9.30, 0.30, 9.65, 0.65),
    "medium_10deg": (5.0, -5.0, 15.0, 5.0),
    "boundary_10deg": (55.0, -5.0, 65.0, 5.0),
    "large_40deg": (0.0, -20.0, 40.0, 20.0),
}
PERIODS = {
    "short_30d": (date(2000, 1, 2), date(2000, 1, 31)),
    "normal_2000_2016": (date(2000, 1, 2), date(2016, 12, 31)),
}
CANDIDATES = {
    "current": (365, 103, 480),
    "longitude_half": (365, 103, 240),
    "latitude_half": (365, 52, 480),
    "time_half": (180, 103, 480),
    "balanced_half": (180, 103, 240),
}


@dataclass(frozen=True)
class Grid:
    """Coordinate-grid properties needed by the geometry model."""

    shape: tuple[int, int, int]
    time_origin: date
    latitude_start: float
    latitude_step: float
    longitude_start: float
    longitude_step: float


def _chunk_length(size: int, chunk: int, index: int) -> int:
    """Return the cell count in one regular or edge chunk."""
    return min(chunk, size - index * chunk)


def _chunk_indices(first: int, last: int, chunk: int) -> range:
    """Return chunk indices intersecting an inclusive cell-index interval."""
    return range(first // chunk, last // chunk + 1)


def _ascending_cell_bounds(
    lower: float,
    upper: float,
    start: float,
    step: float,
    size: int,
    buffer_cells: int,
) -> tuple[int, int]:
    """Map coordinate bounds to inclusive cell indices on an ascending axis."""
    first = math.ceil((lower - start) / step) - buffer_cells
    last = math.floor((upper - start) / step) + buffer_cells
    return max(0, first), min(size - 1, last)


def _latitude_cell_bounds(
    south: float,
    north: float,
    grid: Grid,
    buffer_cells: int,
) -> tuple[int, int]:
    """Map latitude bounds to indices on the store's descending axis."""
    positive_step = abs(grid.latitude_step)
    first = math.ceil((grid.latitude_start - north) / positive_step) - buffer_cells
    last = math.floor((grid.latitude_start - south) / positive_step) + buffer_cells
    return max(0, first), min(grid.shape[1] - 1, last)


def _selection_indices(
    grid: Grid,
    bbox: tuple[float, float, float, float],
    period: tuple[date, date],
    buffer_cells: int,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Return inclusive time, latitude, and longitude cell-index bounds."""
    west, south, east, north = bbox
    time_bounds = (
        (period[0] - grid.time_origin).days,
        (period[1] - grid.time_origin).days,
    )
    latitude_bounds = _latitude_cell_bounds(south, north, grid, buffer_cells)
    longitude_bounds = _ascending_cell_bounds(
        west,
        east,
        grid.longitude_start,
        grid.longitude_step,
        grid.shape[2],
        buffer_cells,
    )
    return time_bounds, latitude_bounds, longitude_bounds


def _sample_chunk_paths(
    store: Path,
    variable: str,
    shape: tuple[int, int, int],
    chunks: tuple[int, int, int],
) -> list[Path]:
    """Build the bounded, deterministic 3x3x3 metadata sample."""
    counts = [math.ceil(size / chunk) for size, chunk in zip(shape, chunks)]
    axes = [sorted({0, count // 2, count - 1}) for count in counts]
    return [
        store / variable / f"{t}.{y}.{x}"
        for t in axes[0]
        for y in axes[1]
        for x in axes[2]
    ]


def _sample_compression(
    store: Path,
    grid: Grid,
    chunks: tuple[int, int, int],
) -> dict[str, list[float]]:
    """Sample compressed bytes per cell for each wf0 variable."""
    samples = {}
    for variable in WF0_VARIABLES:
        values = []
        for path in _sample_chunk_paths(store, variable, grid.shape, chunks):
            t, y, x = (int(part) for part in path.name.split("."))
            cells = (
                _chunk_length(grid.shape[0], chunks[0], t)
                * _chunk_length(grid.shape[1], chunks[1], y)
                * _chunk_length(grid.shape[2], chunks[2], x)
            )
            try:
                compressed_bytes = path.stat().st_size
            except FileNotFoundError:
                # Zarr v2 represents an all-fill chunk by omitting its key.
                continue
            values.append(compressed_bytes / cells)
        samples[variable] = values
    return samples


def _candidate_cost(
    grid: Grid,
    selection: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    chunks: tuple[int, int, int],
    bytes_per_cell: dict[str, float],
) -> tuple[int, float, float]:
    """Model request count, compressed MiB, and byte amplification."""
    axes = [
        _chunk_indices(bounds[0], bounds[1], chunk)
        for bounds, chunk in zip(selection, chunks)
    ]
    chunk_cells = []
    for t in axes[0]:
        for y in axes[1]:
            for x in axes[2]:
                chunk_cells.append(
                    _chunk_length(grid.shape[0], chunks[0], t)
                    * _chunk_length(grid.shape[1], chunks[1], y)
                    * _chunk_length(grid.shape[2], chunks[2], x)
                )
    requests = len(chunk_cells) * len(WF0_VARIABLES)
    compressed_bytes = sum(
        sum(cells * bytes_per_cell[variable] for cells in chunk_cells)
        for variable in WF0_VARIABLES
    )
    selected_cells = math.prod(last - first + 1 for first, last in selection)
    requested_bytes = selected_cells * 4 * len(WF0_VARIABLES)
    return requests, compressed_bytes / 2**20, compressed_bytes / requested_bytes


def _current_exact_cost(
    store: Path,
    selection: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    chunks: tuple[int, int, int],
) -> tuple[int, float]:
    """Sum exact compressed current-chunk sizes for one selection."""
    axes = [
        _chunk_indices(bounds[0], bounds[1], chunk)
        for bounds, chunk in zip(selection, chunks)
    ]
    paths = [
        store / variable / f"{t}.{y}.{x}"
        for variable in WF0_VARIABLES
        for t in axes[0]
        for y in axes[1]
        for x in axes[2]
    ]
    compressed_bytes = 0
    for path in paths:
        try:
            compressed_bytes += path.stat().st_size
        except FileNotFoundError:
            # Missing keys are implicit all-fill chunks and contribute no bytes.
            continue
    return len(paths), compressed_bytes / 2**20


def _peak_rss(stop: threading.Event, readings: list[int]) -> None:
    """Sample this process's resident memory until the read probe completes."""
    import psutil

    process = psutil.Process()
    while not stop.wait(0.05):
        readings.append(process.memory_info().rss)


def _probe_current(store: Path) -> None:
    """Measure one non-redundant 30-day Ntoum read through HydroMT."""
    import hydromt
    import psutil

    catalog = hydromt.DataCatalog(
        data_libs=[
            "config/catalogs/deltares_data.yml",
            "config/catalogs/deltares_era5_daily_zarr.yml",
        ]
    )
    process = psutil.Process()
    rss_readings = [process.memory_info().rss]
    stop = threading.Event()
    sampler = threading.Thread(target=_peak_rss, args=(stop, rss_readings), daemon=True)
    sampler.start()
    bytes_before = process.io_counters().read_bytes
    started = time.perf_counter()
    dataset = catalog.get_rasterdataset(
        "era5",
        bbox=list(WINDOWS["ntoum"]),
        time_range=("2000-01-02", "2000-01-31"),
        buffer=2,
        variables=[
            "precip",
            "temp",
            "temp_min",
            "temp_max",
            "kin",
            "kout",
            "press_msl",
        ],
    ).compute()
    elapsed = time.perf_counter() - started
    stop.set()
    sampler.join()
    read_bytes = process.io_counters().read_bytes - bytes_before
    print(
        "\nMeasured current-store probe: Ntoum + 2-cell buffer, 2000-01-02..31, "
        f"7 variables; {read_bytes / 2**20:.1f} MiB process read bytes, "
        f"{elapsed:.2f} s, {max(rss_readings) / 2**20:.1f} MiB peak RSS, "
        f"output sizes={dict(dataset.sizes)}."
    )
    print(
        "Timing is a single P-drive observation; process read bytes may be lower "
        "than chunk-file bytes when the operating-system cache is warm."
    )
    dataset.close()


def main() -> None:
    """Print metadata, bounded compression samples, and candidate costs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store",
        type=Path,
        default=Path(r"P:\wflow_global\hydromt\meteo\era5_daily.zarr"),
    )
    parser.add_argument(
        "--probe-current",
        action="store_true",
        help="also execute one read-only 30-day Ntoum HydroMT read",
    )
    args = parser.parse_args()

    import zarr

    metadata = json.loads((args.store / ".zmetadata").read_text(encoding="utf-8"))
    arrays = metadata["metadata"]
    time_meta = arrays["time/.zarray"]
    time_attrs = arrays["time/.zattrs"]
    latitude_meta = arrays["latitude/.zarray"]
    longitude_meta = arrays["longitude/.zarray"]
    variable_meta = arrays["t2m/.zarray"]
    shape = tuple(variable_meta["shape"])
    chunks = tuple(variable_meta["chunks"])
    group = zarr.open_group(args.store, mode="r")
    time_values = group["time"][[0, 1, -1]]
    latitude_values = group["latitude"][[0, 1, -1]]
    longitude_values = group["longitude"][[0, 1, -1]]
    time_origin = date.fromisoformat(time_attrs["units"].removeprefix("days since "))
    grid = Grid(
        shape=shape,
        time_origin=time_origin,
        latitude_start=float(latitude_values[0]),
        latitude_step=float(latitude_values[1] - latitude_values[0]),
        longitude_start=float(longitude_values[0]),
        longitude_step=float(longitude_values[1] - longitude_values[0]),
    )

    print(f"store={args.store}")
    print(
        "metadata: Zarr v2, consolidated=yes "
        f"(format {metadata['zarr_consolidated_format']}), sharding=none; "
        f"shape={shape}, chunks={chunks}"
    )
    print(
        f"coordinates: time={time_origin + timedelta(days=int(time_values[0]))}"
        f"..{time_origin + timedelta(days=int(time_values[-1]))} "
        f"by {int(time_values[1] - time_values[0])} day, "
        f"calendar={time_attrs['calendar']}; "
        f"latitude={latitude_values[0]}..{latitude_values[-1]} "
        f"by {grid.latitude_step}; "
        f"longitude={longitude_values[0]}..{longitude_values[-1]} "
        f"by {grid.longitude_step}; CRS=EPSG:4326 from catalog"
    )
    print(
        "wf0 arrays: all float32, C order, fill=NaN, Blosc/Zstd level 3 "
        "with byte shuffle; geometry is identical across all seven variables."
    )
    print(
        f"coordinate arrays: time chunks={time_meta['chunks']} dtype={time_meta['dtype']}; "
        f"latitude chunks={latitude_meta['chunks']}; "
        f"longitude chunks={longitude_meta['chunks']}."
    )

    samples = _sample_compression(args.store, grid, chunks)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    raw_current_mib = math.prod(chunks) * 4 / 2**20
    print(
        "\nCompression sample: exactly 27 chunks/variable at first, middle, and "
        "last chunk on every axis (189 keys total; edge and absent all-fill "
        "chunks included)."
    )
    print("variable  sample compressed MiB min/median/max  median ratio")
    for variable, values in samples.items():
        full_chunk_mib = [value * math.prod(chunks) / 2**20 for value in values]
        ratio = 4 / statistics.median(values)
        print(
            f"{variable:8s} {min(full_chunk_mib):6.1f}/"
            f"{statistics.median(full_chunk_mib):6.1f}/"
            f"{max(full_chunk_mib):6.1f} {ratio:10.2f}:1 "
            f"present={len(values)}/27"
        )
    print(f"current full chunk uncompressed={raw_current_mib:.1f} MiB")

    print(
        "\nAccess model: bboxes are inclusive EPSG:4326 bounds, expanded by the "
        "production two-cell (0.5 degree) HydroMT buffer. Candidate bytes use "
        "the sampled per-variable median compressed bytes/cell, then calibrate "
        "all candidates to the exact current-chunk bytes for that access. They "
        "exclude filesystem/protocol overhead and compression-ratio changes."
    )
    print("window/period              candidate          requests  cal.MiB amp(raw)")
    for window_name, bbox in WINDOWS.items():
        for period_name, period in PERIODS.items():
            selection = _selection_indices(grid, bbox, period, buffer_cells=2)
            exact_requests, exact_mib = _current_exact_cost(
                args.store, selection, chunks
            )
            _, current_model_mib, _ = _candidate_cost(grid, selection, chunks, medians)
            calibration = exact_mib / current_model_mib
            selected_shape = tuple(last - first + 1 for first, last in selection)
            print(
                f"\n{window_name}/{period_name}: selected cells={selected_shape}; "
                f"current exact={exact_requests} requests, {exact_mib:.1f} MiB"
            )
            for candidate_name, candidate_chunks in CANDIDATES.items():
                requests, mib, amplification = _candidate_cost(
                    grid, selection, candidate_chunks, medians
                )
                mib *= calibration
                amplification *= calibration
                print(
                    f"{window_name[:10]:10s}/{period_name[:10]:10s} "
                    f"{candidate_name:18s} {requests:8d} "
                    f"{mib:7.1f} {amplification:9.1f}x"
                )

    print("\nCandidate full-chunk compressed MiB (median model by variable):")
    for candidate_name, candidate_chunks in CANDIDATES.items():
        values = [
            math.prod(candidate_chunks) * medians[variable] / 2**20
            for variable in WF0_VARIABLES
        ]
        print(
            f"{candidate_name:18s} chunks={candidate_chunks} "
            f"range={min(values):.1f}..{max(values):.1f} MiB"
        )

    if args.probe_current:
        _probe_current(args.store)


if __name__ == "__main__":
    main()
