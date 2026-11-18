---
title: "Meteorological data"
format: html
---

The toolbox uses ERA5 or CHIRPS precipitation for historical climate analysis, hydrological forcing
and weather-generator conditioning. Both pathways use ERA5 for temperature, pressure and radiation.

| Dataset | Model contribution | Spatial representation | Time representation |
|---|---|---|---|
| [ERA5][era5] | Preprocessed daily meteorological forcing | Global; ~31 km native, 0.25° CDS atmospheric grid | Hourly source; 1940 onward; daily toolbox input |
| [CHIRPS v2][chirps] | Precipitation, combined with ERA5 temperature, pressure and radiation | Land between 50°S and 50°N; 0.05° (~5.6 km at the equator) | Daily, pentadal and monthly products; 1981 onward; daily toolbox input |

**ERA5.** This global reanalysis from ECMWF / Copernicus Climate Change Service provides the full set
of meteorological variables used by the toolbox.
The catalog accesses preprocessed daily fields; their grid and available years may differ from the
hourly provider archive described above.

**CHIRPS.** This rainfall product from the UCSB Climate Hazards Center combines satellite estimates
with station observations, providing
an alternative precipitation source alongside ERA5's other variables. The configured version is v2;
the provider plans to end v2 production after December 2026, and adopting v3 requires a compatibility
review. Its finer grid does not by itself guarantee more accurate basin rainfall.

Select at least 16 complete calendar years covered by all required inputs. The historical model-run
period is chosen separately within that window. Use historical climate analysis and available local
observations to compare
seasonal patterns, variability and biases before selecting a forcing source.

## Checks before assessment

Historical meteorology serves two purposes: forcing the hydrological model and conditioning the stochastic
weather generator. This makes source selection unusually consequential in the current toolbox, which does not
locally calibrate the model. A structurally complete climate store is not, by itself, a scientific validation
of the forcing. Compare source behavior with local stations or hydrological evidence where available; gridded
products may smooth local extremes or miss topographic gradients even when basin-average statistics look
plausible.

| Source | Checks before assessment |
|---|---|
| ERA5 daily forcing | Coverage over the basin and time window, variable units, precipitation magnitude and seasonality, temperature, and missing values |
| CHIRPS precipitation | Geographic coverage, rainfall seasonality and extremes, and consistency with the other forcing variables needed by the model and generator |
| Local observed discharge | Gauge IDs, dates, units and known regulation |

[era5]: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview
[chirps]: https://www.chc.ucsb.edu/data/chirps
