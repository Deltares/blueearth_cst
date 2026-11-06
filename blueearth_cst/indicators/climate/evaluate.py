"""Evaluate shared climate indices and retain continuous event evidence."""

import numpy as np
import pandas as pd
import xarray as xr

from blueearth_cst.indicators.climate.absolute import absolute_indices
from blueearth_cst.indicators.climate.daily import daily_array
from blueearth_cst.indicators.climate.events import detect_events
from blueearth_cst.indicators.climate.periods import Period
from blueearth_cst.indicators.climate.reference import (
    calendar_day,
    exceedance_probability,
)

EVENT_COLUMNS = [
    "location_id",
    "event_type",
    "start",
    "end",
    "duration",
    "left_censored",
    "right_censored",
    "mean_temperature",
    "peak_temperature",
]


def evaluate(
    data: xr.Dataset,
    reference: xr.Dataset,
    periods: tuple[Period, ...],
    *,
    max_missing_days_month: int = 3,
    max_missing_days_period: int = 15,
    mean_temperature_threshold_degc: float = 25.0,
) -> tuple[xr.Dataset, pd.DataFrame]:
    """Evaluate explicit periods on a grid or location axis without publishing.

    TX90p/TN90p use replacement-year bootstrap in the reference period. WSDI and
    heatwaves use fixed-reference thresholds. Events are detected before period
    slicing. Period lengths are overlapping observed days; full event lengths
    stay in the catalogue. Exact heatwave counts are unavailable if missing days
    or censored qualifying segments prevent establishing their physical count.
    Reference and evaluation spatial coordinates must match exactly.
    """
    output = xr.Dataset()
    rows = []
    wet = float(reference.attrs["wet_day_threshold_mm"])
    for variable in ("precip", "temp", "temp_max", "temp_min"):
        if variable not in data:
            continue
        field = daily_array(
            data[variable], units="mm/day" if variable == "precip" else "degC"
        )
        spatial = tuple(dim for dim in field.dims if dim != "time")
        field = field.transpose("time", *spatial)
        ref = reference.copy(deep=False)
        for dim in spatial:
            if dim not in ref.coords or not field.get_index(dim).equals(
                ref.get_index(dim)
            ):
                raise ValueError(f"Reference coordinate mismatch: {dim}")
        base = absolute_indices(
            field,
            periods,
            variable=variable,
            wet_day_threshold_mm=wet,
            mean_temperature_threshold_degc=mean_temperature_threshold_degc,
            max_missing_days_month=max_missing_days_month,
            max_missing_days_period=max_missing_days_period,
        )
        base = base.rename(
            {
                name: f"{variable}_{name}"
                for name in ("valid_days", "expected_days", "eligible")
            }
        )
        output = xr.merge([output, base], compat="no_conflicts")
        if variable == "temp":
            continue
        dates = pd.DatetimeIndex(field.time.values)
        values = field.values
        shape = values.shape[1:]
        flat = values.reshape((len(dates), -1))
        if variable == "precip":
            names = {
                "cdd": "days",
                "cwd": "days",
                "r95p_amount": "mm",
                "r95p_fraction": "%",
                "wet_spell_mean": "days",
                "wet_spell_median": "days",
                "wet_spell_p90": "days",
            }
            threshold = ref["precip_p95"].transpose(*spatial).values.reshape(-1)
            probabilities = None
        else:
            name = "tx90p" if variable == "temp_max" else "tn90p"
            names = {name: "%"}
            if variable == "temp_max":
                names.update(
                    wsdi="days",
                    heatwave_count="events",
                    heatwave_days="days",
                    longest_heatwave="days",
                )
            ref = ref.transpose("reference_year", "calendar_day", *spatial)
            probabilities = exceedance_probability(
                values, dates, ref, variable
            ).reshape(flat.shape)
            threshold = ref[f"{variable}_p90"].values.reshape((365, -1))[
                calendar_day(dates)
            ]
        metrics = {
            name: np.full((len(periods), flat.shape[1]), np.nan) for name in names
        }
        eligible = (
            base[f"{variable}_eligible"]
            .transpose("period", *spatial)
            .values.reshape((len(periods), -1))
        )
        for location in range(flat.shape[1]):
            observations = flat[:, location]
            valid = np.isfinite(observations)
            events = {}
            if variable == "precip":
                events["dry"] = detect_events(observations < wet, valid)
                events["wet"] = detect_events(observations >= wet, valid)
            elif variable == "temp_max":
                q = threshold[:, location]
                known = valid & np.isfinite(q)
                warm_segments = detect_events(observations > q, known)
                events["warm_spell"] = detect_events(
                    observations > q, known, minimum_duration=6
                )
                events["heatwave"] = detect_events(
                    observations > q, known, minimum_duration=3
                )
            coordinates = (
                dict(zip(spatial, np.unravel_index(location, shape), strict=True))
                if spatial
                else {}
            )
            location_values = {
                dim: field[dim].values[index].item()
                for dim, index in coordinates.items()
            }
            for event_type, catalogue in events.items():
                for event in catalogue:
                    temperature = (
                        observations[event.start : event.stop]
                        if variable == "temp_max"
                        else None
                    )
                    rows.append(
                        {
                            "location_id": location,
                            **location_values,
                            "event_type": event_type,
                            "start": dates[event.start].isoformat(),
                            "end": dates[event.stop - 1].isoformat(),
                            "duration": event.duration,
                            "left_censored": event.left_censored,
                            "right_censored": event.right_censored,
                            "mean_temperature": float(temperature.mean())
                            if temperature is not None
                            else np.nan,
                            "peak_temperature": float(temperature.max())
                            if temperature is not None
                            else np.nan,
                        }
                    )
            for pindex, period in enumerate(periods):
                if not eligible[pindex, location]:
                    continue
                selected = np.asarray((dates >= period.start) & (dates < period.stop))
                lo, hi = dates.searchsorted([period.start, period.stop])
                obs = observations[selected]
                if variable == "precip":
                    metrics["cdd"][pindex, location] = max(
                        (e.overlap(lo, hi) for e in events["dry"]), default=0
                    )
                    metrics["cwd"][pindex, location] = max(
                        (e.overlap(lo, hi) for e in events["wet"]), default=0
                    )
                    complete = [
                        e.duration
                        for e in events["wet"]
                        if lo <= e.start < hi
                        and not (e.left_censored or e.right_censored)
                    ]
                    if complete:
                        for key, quantile in (
                            ("wet_spell_median", 0.5),
                            ("wet_spell_p90", 0.9),
                        ):
                            metrics[key][pindex, location] = np.quantile(
                                complete, quantile, method="median_unbiased"
                            )
                        metrics["wet_spell_mean"][pindex, location] = np.mean(complete)
                    if np.isfinite(threshold[location]):
                        amount = np.nansum(obs[obs > threshold[location]])
                        denominator = np.nansum(obs[obs >= wet])
                        metrics["r95p_amount"][pindex, location] = amount
                        if denominator > 0:
                            metrics["r95p_fraction"][pindex, location] = (
                                100 * amount / denominator
                            )
                else:
                    probability = probabilities[selected, location]
                    if np.isfinite(probability[valid[selected]]).all():
                        metrics[name][pindex, location] = (
                            100 * np.nansum(probability) / valid[selected].sum()
                        )
                    if (
                        variable == "temp_max"
                        and np.isfinite(
                            threshold[selected, location][valid[selected]]
                        ).all()
                    ):
                        for event_type, metric in (
                            ("warm_spell", "wsdi"),
                            ("heatwave", "heatwave_days"),
                        ):
                            metrics[metric][pindex, location] = sum(
                                e.overlap(lo, hi) for e in events[event_type]
                            )
                        catalogue = events["heatwave"]
                        metrics["longest_heatwave"][pindex, location] = max(
                            (e.overlap(lo, hi) for e in catalogue), default=0
                        )
                        censored = any(
                            e.overlap(lo, hi) and (e.left_censored or e.right_censored)
                            for e in warm_segments
                        )
                        if (
                            valid[selected].all()
                            and len(obs) == (period.stop - period.start).days
                            and not censored
                        ):
                            metrics["heatwave_count"][pindex, location] = sum(
                                lo <= e.start < hi for e in catalogue
                            )
        for name, matrix in metrics.items():
            output[name] = xr.DataArray(
                matrix.reshape((len(periods), *shape)),
                dims=("period", *spatial),
                coords={
                    "period": [p.label for p in periods],
                    **{d: field[d] for d in spatial},
                },
                attrs={"units": names[name]},
            )
    output.attrs.update(
        method_version="climate-indices/1",
        wsdi_reference="fixed; unbootstrapped",
        temperature_percentages="Zhang replacement-year bootstrap in reference years",
        event_period_allocation="known starts for counts; overlapping observed days for durations",
        spell_lengths="period-overlap lengths; censored observations are lower bounds",
        reference_qualification=reference.attrs["qualification"],
    )
    events = pd.DataFrame(rows)
    if events.empty:
        events = pd.DataFrame(columns=EVENT_COLUMNS)
    events.insert(0, "event_id", np.arange(len(events)))
    return output, events
