# Daily climate indicators

This component fits climate references and evaluates daily xarray datasets without project configuration,
acquisition, plotting or filesystem publication. The historical-climate adapter owns those operations.
It requires NumPy, pandas and xarray; event catalogues are pandas DataFrames.

```python
from blueearth_cst.indicators.climate.reference import fit_reference
from blueearth_cst.indicators.climate.evaluate import evaluate
from blueearth_cst.indicators.climate.periods import annual_periods

# daily: Dataset with precip [mm/day], temp, temp_max and/or temp_min [degC].
reference = fit_reference(daily, 1991, 2020)
indices, events = evaluate(daily, reference, annual_periods(2021, 2025))
```

Inputs have a Gregorian, timezone-free, increasing midnight `time` coordinate. Missing dates become
NaN; duplicate dates, infinities, noncanonical units and negative precipitation are refused. Arbitrary
spatial dimensions are supported, including a station/location axis or no spatial axis. Reference and
evaluation coordinates must match. Non-Gregorian climate-model calendars are not supported yet.

`daily.py` validates inputs; `periods.py` constructs half-open reporting periods; `events.py` detects
observed runs with gap/edge censoring. `absolute.py` evaluates absolute indices, `reference.py` fits
serializable thresholds and retains temperature samples for bootstrap, and `evaluate.py` combines
indices with continuous event detection. `definitions.py` owns display labels and units.

The [historical climate manual](../../../docs/technical-manual/workflow-analyze-climate.md#climate-indices)
owns formulas, reference conventions, missingness, period allocation and interpretation. In particular,
TX90p/TN90p are bootstrap-corrected within reference years; WSDI and heatwaves use fixed thresholds.
Daily mean `temp` supplies observed-day counts below zero and above
`mean_temperature_threshold_degc` (default 25°C), passed to `evaluate` or `absolute_indices`.
These counts do not require percentile support and never substitute TX/TN for the mean.
Calling evaluation on a subset does not refit thresholds. Complete reference years, including absent
years, remain in expected support counts. Unsupported bootstrap replicates make a result unavailable.

The returned Dataset retains variable-specific valid/expected days and eligibility alongside values.
NaN denotes insufficient observation/reference support or an undefined statistic; zero denotes an
eligible period with no qualifying observed days/events. Full event lengths and inclusive start/end
dates stay in the catalogue; period duration metrics count overlap. Censored lengths are lower bounds.
Heatwave counts require complete period observations and no overlapping censored qualifying event.

Qualification uses independent synthetic arithmetic and boundary cases in
`tests/test_climate_indicator_*.py`. These checks establish implementation behavior, not adequacy of a
particular climate dataset or full equivalence to every ETCCDI implementation. Spatial blocks bound
bootstrap window expansion without dropping locations. Input arrays and the event catalogue are held
in memory; large grids and long reference periods can still require substantial memory and compute.
