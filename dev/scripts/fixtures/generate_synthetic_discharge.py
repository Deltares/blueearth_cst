"""Generate synthetic daily discharge for testing model evaluation plots and tables."""

import numpy as np
import pandas as pd
from scipy.signal import lfilter
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


def synthetic_discharge(
    dates: pd.DatetimeIndex,
    ids: list[str],
    rng: np.random.Generator,
    wet_month: int,
    reference: pd.DataFrame | None,
) -> pd.DataFrame:
    season = wet_season(dates, wet_month)
    shared_storms = storm_depths(rng, season)
    shared_error = correlated_noise(rng, len(dates), persistence=0.8)
    days = np.arange(len(dates))
    discharge = pd.DataFrame(index=dates)
    for location_id in ids:
        rain = 0.7 * shared_storms + 0.5 * storm_depths(rng, season)
        fast = lfilter([0.62], [1, -0.38], rain)
        slow = lfilter([0.035], [1, -0.965], rain)
        runoff = 0.1 + 0.7 * fast + 0.3 * slow
        runoff /= runoff.mean()
        if reference is None:
            discharge[location_id] = rng.uniform(6, 20) * runoff
            continue

        flow = reference[location_id].to_numpy()
        scale = flow.mean()
        if scale <= 0 or not np.isfinite(scale):
            raise ValueError(
                f"Reference discharge for {location_id} needs positive mean flow"
            )
        lag = rng.integers(-2, 3)
        shifted = flow[np.clip(days - lag, 0, len(days) - 1)]
        peak_weight = shifted / (shifted + scale)
        error = 0.55 * shared_error + 0.84 * correlated_noise(
            rng, len(dates), persistence=0.65
        )
        # Event-dependent spread, persistent residuals and a distinct low-flow floor.
        spread = 0.22 + 0.43 * peak_weight
        gain = rng.uniform(0.75, 1.3) * np.exp(
            0.35 * (season - 0.5) + spread * error - spread**2 / 2
        )
        discharge[location_id] = shifted * gain + scale * (0.015 + 0.16 * runoff)
    if not np.isfinite(discharge.to_numpy()).all():
        raise ValueError("Synthetic discharge exceeds the supported numeric range")
    return discharge.reset_index()


def main() -> None:
    parser = observation_parser(
        __doc__, "optional Wflow output_q.csv (time and wflow_id columns)"
    )
    args = parser.parse_args()
    try:
        dates = observation_dates(args.start, args.end)
        ids = read_locations(args.locations).index.to_list()
        reference = (
            read_reference(args.reference, "time", ids, dates)
            if args.reference
            else None
        )
        discharge = synthetic_discharge(
            dates, ids, np.random.default_rng(args.seed), args.wet_month, reference
        )
        write_bundle(
            args,
            "discharge",
            {"synthetic-observed-discharge.csv": discharge},
            separator=";",
        )
    except ValueError as exc:
        parser.error(str(exc))
    except OSError as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    print(
        f"Wrote synthetic discharge for {len(ids)} locations and {len(dates)} days to {args.out}"
    )


if __name__ == "__main__":
    main()
