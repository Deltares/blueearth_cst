"""Shared inputs and output handling for the synthetic observation generators."""

import argparse
import csv
import json
from datetime import date
from hashlib import file_digest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter


def observation_parser(
    description: str, reference_help: str
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.epilog = (
        f"Example: pixi run python dev/scripts/{parser.prog} "
        "--locations location_registry.csv --start 2000-01-02 --end 2001-01-01 "
        "--seed 42 --out synthetic-fixture"
    )
    parser.add_argument(
        "--locations", type=Path, required=True, help="location_registry.csv"
    )
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        required=True,
        help="first daily interval end (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        required=True,
        help="last daily interval end (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="nonnegative random seed"
    )
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument(
        "--wet-month",
        type=int,
        choices=range(1, 13),
        default=8,
        help="month of peak synthetic rainfall (default: 8)",
    )
    parser.add_argument("--reference", type=Path, help=reference_help)
    return parser


def read_locations(path: Path) -> pd.DataFrame:
    locations = pd.read_csv(path, dtype={"wflow_id": str}, float_precision="round_trip")
    if "wflow_id" not in locations or locations.empty:
        raise ValueError(
            "Location registry requires wflow_id and at least one location"
        )
    ids = locations.wflow_id
    if not ids.str.fullmatch(r"[1-9][0-9]*", na=False).all() or ids.duplicated().any():
        raise ValueError(
            "Location registry requires unique positive integer wflow_id values"
        )
    return locations.set_index("wflow_id").sort_index()


def observation_dates(start: date, end: date) -> pd.DatetimeIndex:
    if end < start:
        raise ValueError("--end must be on or after --start")
    return pd.date_range(start, end, name="time")


def read_reference(
    path: Path, date_column: str, columns: list[str], dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Read the requested interval ends and reject gaps in the reference."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    needed = [date_column, *columns]
    if not set(needed).issubset(header) or len(header) != len(set(header)):
        raise ValueError(f"{path}: expected unique CSV columns including {needed}")
    frame = pd.read_csv(path, usecols=needed, float_precision="round_trip").set_index(
        date_column
    )
    index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="raise"))
    if (
        index.tz is not None
        or index.hasnans
        or index.has_duplicates
        or not index.equals(index.normalize())
    ):
        raise ValueError(
            f"{path}: reference dates must be unique midnight UTC labels without a timezone suffix"
        )
    frame.index = index
    frame = frame.reindex(dates)[columns].apply(pd.to_numeric, errors="raise")
    values = frame.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(
            f"{path}: reference needs complete finite, nonnegative values over the requested dates"
        )
    return frame


def correlated_noise(
    rng: np.random.Generator, size: int, persistence: float
) -> np.ndarray:
    innovations = rng.standard_normal(size)
    innovations[1:] *= np.sqrt(1 - persistence**2)
    return lfilter([1], [1, -persistence], innovations)


def wet_season(dates: pd.DatetimeIndex, wet_month: int) -> np.ndarray:
    covered_days = dates - pd.Timedelta(days=1)
    peak_day = date(2001, wet_month, 15).timetuple().tm_yday
    return (
        1 + np.cos(2 * np.pi * (covered_days.dayofyear.to_numpy() - peak_day) / 365.25)
    ) / 2


def storm_depths(rng: np.random.Generator, season: np.ndarray) -> np.ndarray:
    weather = correlated_noise(rng, len(season), persistence=0.6)
    wet = weather > 1.15 - 1.35 * season
    return np.where(wet, rng.gamma(0.9, 4 + 12 * season), 0.0)


def write_bundle(
    args: argparse.Namespace,
    kind: str,
    tables: dict[str, pd.DataFrame],
    separator: str = ",",
) -> None:
    """Publish labelled CSVs and provenance to a new, exclusively created directory."""
    inputs = {}
    for label, path in (("locations", args.locations), ("reference", args.reference)):
        if path is not None:
            with path.open("rb") as stream:
                inputs[label] = {
                    "path": str(path),
                    "sha256": file_digest(stream, "sha256").hexdigest(),
                }
    metadata = {
        "synthetic": True,
        "product": kind,
        "seed": args.seed,
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "time_label": "interval_end",
        "wet_month": args.wet_month,
        "inputs": inputs,
        "files": list(tables),
    }
    manifest = args.out / "synthetic-observations.json"
    try:
        args.out.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f"Choose a new output directory: {args.out}") from exc
    written = []
    try:
        for filename, table in tables.items():
            path = args.out / filename
            with path.open("x", encoding="utf-8", newline="") as stream:
                written.append(path)
                table.to_csv(stream, index=False, sep=separator, date_format="%Y-%m-%d")
        with manifest.open("x", encoding="utf-8") as stream:
            written.append(manifest)
            stream.write(json.dumps(metadata, indent=2) + "\n")
    except (Exception, KeyboardInterrupt):
        for path in written:
            path.unlink(missing_ok=True)
        if not any(args.out.iterdir()):
            args.out.rmdir()
        raise
