---
title: This is a TOML configuration file for Wflow.
source: "https://deltares.github.io/Wflow.jl/stable/user_guide/required_files.html"
reference: "Wflow.jl team. (n.d.). *This is a TOML configuration file for Wflow.*. Wflow documentation. Retrieved May 25, 2026, from sources/tools/wflow/wflow-user-guide/02-required-files.md"
topic: tools
type: documentation
promoted: 2026-05-25
tags:
  - tools
  - wflow
accessed: 2026-05-25
---

To run wflow, several files are required. These include a settings file and input data. The input data is typically separated into static maps and forcing data, and both are provided in netCDF files, except for reservoir storage and rating curves that are supplied via CSV files. Below is a brief overview of the different files:

- The `settings.toml` file contains information on the simulation period, links to the input files (and their names in the netCDF files), and connect the correct variable names in the netCDF files to the variables and parameters of wflow.
- The `staticmaps.nc` file contains spatial information such as elevation, gauge locations, land use, and drainage direction, etc. This file can also contain maps with parameter values.
- The `forcing.nc` file contains time series data for precipitation, temperature and potential evaporation (as a 3D array).

## The configuration file (settings.toml)

The configuration file contains all relevant settings for running wflow, such as the simulation period, the model settings, the mapping between input files and (internal) model parameters. More details and explanations can be found [here](../user_guide/toml_file.html). An example configuration file is presented below.

Click to show example.toml file

```toml
# This is a TOML configuration file for Wflow.
# Relative file paths are interpreted as being relative to this TOML file.
# Wflow documentation https://deltares.github.io/Wflow.jl/dev/
# TOML documentation: https://github.com/toml-lang/toml

dir_input = "data/input"
dir_output = "data/output"

[time]
endtime = 2000-01-10T00:00:00

[logging]
loglevel = "info"

[input]
path_forcing = "forcing-moselle.nc"
path_static = "staticmaps-moselle.nc"

# these are not directly part of the model
basin__local_drain_direction = "wflow_ldd"
river_location__mask = "wflow_river"
reservoir_area__count = "wflow_reservoirareas"
reservoir_location__count = "wflow_reservoirlocs"
subbasin_location__count = "wflow_subcatch"

[input.forcing]
atmosphere_water__precipitation_volume_flux = "precip"
land_surface_water__potential_evaporation_volume_flux = "pet"
atmosphere_air__temperature = "temp"

[input.static]
atmosphere_air__snowfall_temperature_threshold = "TT"
atmosphere_air__snowfall_temperature_interval = "TTI"

land_water_covered__area_fraction = "WaterFrac"

snowpack__melting_temperature_threshold = "TTM"
snowpack__degree_day_coefficient = "Cfmax"

soil_layer_water__brooks_corey_exponent = "c"
soil_surface_water__infiltration_reduction_parameter = "cf_soil"
soil_surface_water__vertical_saturated_hydraulic_conductivity = "KsatVer"
soil_water__vertical_saturated_hydraulic_conductivity_scale_parameter = "f"
compacted_soil_surface_water__infiltration_capacity = "InfiltCapPath"
soil_water__residual_volume_fraction = "thetaR"
soil_water__saturated_volume_fraction = "thetaS"
soil_water_saturated_zone_bottom__max_leakage_volume_flux = "MaxLeakage"
compacted_soil__area_fraction = "PathFrac"
soil_wet_root__sigmoid_function_shape_parameter = "rootdistpar"
soil__thickness = "SoilThickness"

vegetation_canopy_water__mean_evaporation_to_mean_precipitation_ratio = "EoverR"
vegetation_canopy__light_extinction_coefficient = "Kext"
vegetation__specific_leaf_storage = "Sl"
vegetation_wood_water__storage_capacity = "Swood"
vegetation_root__depth = "RootingDepth"

river__length = "wflow_riverlength"
river_water_flow__manning_n_parameter = "N_River"
river__slope = "RiverSlope"
river__width = "wflow_riverwidth"

land_surface_water_flow__manning_n_parameter = "N"
land_surface__slope = "Slope"

reservoir_surface__area = "reservoir_area"
reservoir_water_demand__required_downstream_volume_flow_rate = "ResDemand"
reservoir_water_release_below_spillway__max_volume_flow_rate = "ResMaxRelease"
reservoir_water__max_volume = "ResMaxVolume"
reservoir_water__target_full_volume_fraction = "ResTargetFullFrac"
reservoir_water__target_min_volume_fraction = "ResTargetMinFrac"
reservoir_water_surface__initial_elevation = "waterlevel_reservoir"
reservoir_water__rating_curve_type_count = "outflowfunc"
reservoir_water__storage_curve_type_count = "storfunc"

subsurface_water__horizontal_to_vertical_saturated_hydraulic_conductivity_ratio = "KsatHorFrac"

[input.cyclic]
vegetation__leaf_area_index = "LAI"

[model]
soil_layer__thickness = [100, 300, 800]
type = "sbm"

[output.csv]
path = "output_moselle_simple.csv"

[[output.csv.column]]
coordinate.x = 7.378
coordinate.y = 50.204
header = "Q"
parameter = "river_water__volume_flow_rate"

[[output.csv.column]]
header = "recharge"
parameter = "soil_water_saturated_zone_top__recharge_volume_flux"
reducer = "mean"
```

## The static input data (staticmaps.nc)

The list below contains a brief overview of several essential static maps required to run wflow. These NC variables names refer to the example data of the wflow\_sbm + kinematic wave model (see [here](../getting_started/download_example_models.html#wflow_sbm-kinematic-wave)). Example data for the other model configurations can be found [here](../getting_started/download_example_models.html).

| `basin__local_drain_direction` | Local drain direction (1-9) | `wflow_ldd` | \- |
| --- | --- | --- | --- |
| `river_location__mask` | River mask (0-1) | `wflow_river` | \- |
| `river__length` | River length | `wflow_riverlength` | m |
| `river__width` | River width | `wflow_riverwidth` | m |
| `subbasin_location__count` | Subbasin ids | `wflow_subcatch` | \- |
| `land_surface__slope` | Land slope | `Slope` | m m \\(^{-1}\\) |
| `river__slope` | River slope | `RiverSlope` | m m \\(^{-1}\\) |

As mentioned before, the model parameters can also be defined as spatial maps. They can be included in the same netCDF file, as long as their variable names are correctly mapped in the TOML settings file. See the section on [example models](../getting_started/download_example_models.html) on how to use this functionality. The time unit of input flux parameters should be day (the model base time step size), these input parameters are converted to the user-defined model time step size during initialization.

Important

When using cyclic data, three different options are supported:

- 12 (monthly)
- 365 (daily, where Feb. 29 is not present, so the value for Feb. 28 is taken. Dec. 31 is represented by DOY 365)
- 366 (where Feb. 29 represents DOY 60, and Dec. 31 DOY 366)

In contrast to the right-labelling of the forcing data (see below), the DOY/month of the current time step is used. For example: to simulate 2023-06-14 00:00:00 (with a daily time step), the DOY value at position 6 (when 12 values are provided), 165 (when 365 values are provided) or 166 (when 366 values are provided).

## The forcing data (forcing.nc)

The forcing data typically contains the meteorological boundary conditions. This data is provided as a single netCDF file, with several variables containing the forcing data for precipitation, temperature and potential evaporation. The time unit of forcing data should be equal to the user-defined model time step size. The code snippet below shows the contents of the example file (downloaded [here](../getting_started/download_example_models.html#wflow_sbm-kinematic-wave)), and displaying the content with `NCDatasets` in Julia. As can be seen, each forcing variable (`precip`, `pet` and `temp`) consists of a three-dimensional dataset (`x`, `y`, and `time`), and each timestep consists of a two-dimensional map with values at each grid cell. Only values within the basin are required.

Important

Wflow expects the forcing to be “right-labelled”. This means that e.g. daily precipitation at 2023-06-15 00:00:00 is the accumulated total precipitation between 2023-06-14 00:00:00 and 2023-06-15 00:00:00.

Note

For optimal computational performance, it is recommended to have chunks in the time dimension. This way, only part of the forcing file needs to be read and kept in memory. We recommend using a chunk size in the time dimension with size 1. Using larger chunks can largely degrade computational performance.

Click to show example forcing.nc file

```bash
Group: /

Dimensions
  time = 366
  y = 313
  x = 291

Variables
  time   (366)
    Datatype:    Int64
    Dimensions:  time
    Attributes:
    units                = days since 2000-01-02 00:00:00
    calendar             = proleptic_gregorian

  y   (313)
    Datatype:    Float64
    Dimensions:  y
    Attributes:
    _FillValue           = NaN

  x   (291)
    Datatype:    Float64
    Dimensions:  x
    Attributes:
    _FillValue           = NaN

  spatial_ref
    Attributes:
    crs_wkt              = GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]
    x_dim                = x
    y_dim                = y
    dim0                 = time

  precip   (291 × 313 × 366)
    Datatype:    Float32
    Dimensions:  x × y × time
    Attributes:
    _FillValue           = NaN
    unit                 = mm
    precip_fn            = era5
    coordinates          = idx_out spatial_ref mask

  idx_out   (291 × 313)
    Datatype:    Int32
    Dimensions:  x × y

  mask   (291 × 313)
    Datatype:    UInt8
    Dimensions:  x × y

  pet   (291 × 313 × 366)
    Datatype:    Float32
    Dimensions:  x × y × time
    Attributes:
    _FillValue           = NaN
    unit                 = mm
    pet_fn               = era5
    pet_method           = debruin
    coordinates          = idx_out spatial_ref mask

  temp   (291 × 313 × 366)
    Datatype:    Float32
    Dimensions:  x × y × time
    Attributes:
    _FillValue           = NaN
    unit                 = degree C.
    temp_fn              = era5
    temp_correction      = True
    coordinates          = idx_out spatial_ref mask

Global attributes
  unit                 = mm
  precip_fn            = era5
```
