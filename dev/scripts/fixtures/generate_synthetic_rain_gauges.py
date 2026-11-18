"""Generate synthetic rain-gauge metadata and daily rainfall for comparison checks."""

import numpy as np
import pandas as pd
from synthetic_observation_fixtures import (
    correlated_noise,
    observation_dates,
    observation_parser,
    read_locations,
    read_reference,
    storm_depths,
    wet_season,
    write_bundle,
)


def gauge_stations(locations: pd.DataFrame) -> pd.DataFrame:
    if not {"original_x", "original_y"} <= set(locations.columns):
        raise ValueError(
            "Rain gauges require original_x and original_y in the location registry"
        )
    x = pd.to_numeric(locations.original_x, errors="raise")
    y = pd.to_numeric(locations.original_y, errors="raise")
    if not (x.between(-180, 180) & y.between(-90, 90)).all():
        raise ValueError(
            "Original coordinates must be finite WGS84 longitude and latitude"
        )
    names = locations.get("station_name", locations.index.to_series()).fillna(
        locations.index.to_series()
    )
    return pd.DataFrame(
        {
            "id": locations.index,
            "name": "SYNTHETIC " + names.astype(str),
            "x": x,
            "y": y,
        }
    )


def synthetic_precipitation(
    dates: pd.DatetimeIndex,
    ids: list[str],
    rng: np.random.Generator,
    wet_month: int,
    reference: pd.DataFrame | None,
) -> pd.DataFrame:
    season = wet_season(dates, wet_month)
    base = (
        storm_depths(rng, season) if reference is None else reference.precip.to_numpy()
    )
    precipitation = pd.DataFrame(index=dates)
    for gauge_id in ids:
        local_storms = storm_depths(rng, season)
        error = correlated_noise(rng, len(dates), persistence=0.45)
        rain = (
            rng.uniform(0.8, 1.2) * base * np.exp(0.48 * error - 0.48**2 / 2)
            + 0.35 * local_storms
        )
        rain[(rain < 2) & (rng.random(len(dates)) < 0.35)] = 0
        precipitation[gauge_id] = rain.round(3)
    if not np.isfinite(precipitation.to_numpy()).all():
        raise ValueError("Synthetic rainfall exceeds the supported numeric range")
    return precipitation.reset_index()


def main() -> None:
    parser = observation_parser(
        __doc__, "optional daily_basin.csv (date and precip columns, mm/day)"
    )
    args = parser.parse_args()
    try:
        dates = observation_dates(args.start, args.end)
        stations = gauge_stations(read_locations(args.locations))
        reference = (
            read_reference(args.reference, "date", ["precip"], dates)
            if args.reference
            else None
        )
        rain = synthetic_precipitation(
            dates,
            stations.id.to_list(),
            np.random.default_rng(args.seed),
            args.wet_month,
            reference,
        )
        write_bundle(
            args,
            "rain-gauges",
            {
                "synthetic-rain-gauge-stations.csv": stations,
                "synthetic-rain-gauge-precip.csv": rain,
            },
        )
    except ValueError as exc:
        parser.error(str(exc))
    except OSError as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    print(
        f"Wrote {len(stations)} synthetic rain gauges and {len(dates)} days to {args.out}"
    )


if __name__ == "__main__":
    main()
