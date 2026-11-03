---
title: HydroMT-Wflow user guide
source: "https://deltares.github.io/hydromt_wflow/stable/user_guide"
reference: "HydroMT team. (n.d.). *HydroMT-Wflow user guide*. HydroMT-Wflow documentation. sources/tools/hydromt-wflow/user-guide.md"
topic: tools
type: documentation
promoted: 2026-05-17
tags:
  - tools
  - hydromt-wflow
accessed: 2026-05-17
---

# HydroMT-Wflow user guide

Source note: the project changelog is intentionally not archived here. For
release-specific details, use the live changelog:
<https://deltares.github.io/hydromt_wflow/latest/changelog.html>.

## 1. User guide

The user guide is organised through the following sections:

With the **HydroMT-Wflow plugin**, users can easily benefit from the rich set of tools of the [HydroMT package](https://deltares.github.io/hydromt/latest/index.html) to build and update [Wflow](https://deltares.github.io/Wflow.jl/stable/) models from available global and local data.

This plugin assists Wflow modellers in:

- Quickly setting up a base Wflow model and default parameter values
- Making maximum use of the best available global or local data
- Adjusting and updating components of a Wflow model and their associated parameters in a consistent way
- Clipping existing Wflow models for a smaller extent
- Analysing Wflow model outputs

Two Wflow model classes are currently available:

- `wflow_sbm` ([`WflowSbmModel`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.html#hydromt_wflow.WflowSbmModel "hydromt_wflow.WflowSbmModel")): for the **wflow\_sbm + kinematic** and **wflow\_sbm + local inertial** concepts
- `wflow_sediment` ([`WflowSedimentModel`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.html#hydromt_wflow.WflowSedimentModel "hydromt_wflow.WflowSedimentModel")): for the **wflow\_sediment** concept

## 2. Model methods and components

The HydroMT-Wflow plugin helps you preparing or updating several inputs of a Wflow model such as topography information, landuse, soil or forcing using setup methods. The main interactions are available from the HydroMT Command Line Interface and allow you to configure HydroMT in order to build or update or clip Wflow models.

When building or updating a model from command line a [model region](https://deltares.github.io/hydromt/latest/user_guide/models/model_region.html); a model setup [configuration](https://deltares.github.io/hydromt_wflow/stable/user_guide/2_sbm_model/2_build.html#model-config) (.yml file) with model methods and options and, optionally, a [data catalog](https://deltares.github.io/hydromt/latest/user_guide/data_catalog/data_overview.html) (.yml) file should be prepared.

### 2.1. Model setup methods

An overview of the available Wflow model setup methods is provided in the tables below. When using HydroMT from the command line only the setup methods are exposed. Click on a specific method to see its documentation.

#### 2.1.1. Configuration (TOML)

Defines and manages model configuration, global parameters, and output settings.

| Method | Explanation | Required Setup Method |
| --- | --- | --- |
| [`setup_config()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_config.html#hydromt_wflow.WflowSbmModel.setup_config "hydromt_wflow.WflowSbmModel.setup_config") | Update config with a dictionary |  |
| [`setup_config_output_timeseries()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_config_output_timeseries.html#hydromt_wflow.WflowSbmModel.setup_config_output_timeseries "hydromt_wflow.WflowSbmModel.setup_config_output_timeseries") | Add new variable/column to the netcdf/csv output section of the toml based on a selected gauge/area map. | [`setup_basemaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_basemaps.html#hydromt_wflow.WflowSbmModel.setup_basemaps "hydromt_wflow.WflowSbmModel.setup_basemaps") |
| [`setup_constant_pars()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_constant_pars.html#hydromt_wflow.WflowSbmModel.setup_constant_pars "hydromt_wflow.WflowSbmModel.setup_constant_pars") | Setup constant parameter maps for all active model cells. |  |

#### 2.1.2. Topography and Rivers

Prepares elevation maps, drainage networks, and river-related features used to simulate flow routing and floodplain processes.

| Method | Explanation | Required Setup Method |
| --- | --- | --- |
| [`setup_basemaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_basemaps.html#hydromt_wflow.WflowSbmModel.setup_basemaps "hydromt_wflow.WflowSbmModel.setup_basemaps") | Set the region of interest and res (resolution in degrees) of the model. |  |
| [`setup_rivers()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_rivers.html#hydromt_wflow.WflowSbmModel.setup_rivers "hydromt_wflow.WflowSbmModel.setup_rivers") | Set all river parameter maps. | [`setup_basemaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_basemaps.html#hydromt_wflow.WflowSbmModel.setup_basemaps "hydromt_wflow.WflowSbmModel.setup_basemaps") |
| [`setup_river_roughness()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_river_roughness.html#hydromt_wflow.WflowSbmModel.setup_river_roughness "hydromt_wflow.WflowSbmModel.setup_river_roughness") | Set river Manning roughness coefficient. | [`setup_rivers()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_rivers.html#hydromt_wflow.WflowSbmModel.setup_rivers "hydromt_wflow.WflowSbmModel.setup_rivers") |
| [`setup_floodplains()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_floodplains.html#hydromt_wflow.WflowSbmModel.setup_floodplains "hydromt_wflow.WflowSbmModel.setup_floodplains") | Add floodplain information (can be either 1D or 2D). | [`setup_rivers()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_rivers.html#hydromt_wflow.WflowSbmModel.setup_rivers "hydromt_wflow.WflowSbmModel.setup_rivers") |

#### 2.1.3. Reservoirs and Glaciers

Adds reservoirs and glaciers, and defines their impact on hydrological storage and flow regulation.

| Method | Explanation | Required Setup Method |
| --- | --- | --- |
| [`setup_reservoirs_no_control()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_reservoirs_no_control.html#hydromt_wflow.WflowSbmModel.setup_reservoirs_no_control "hydromt_wflow.WflowSbmModel.setup_reservoirs_no_control") | Generate maps of uncontrolled reservoirs (lakes, weirs) areas and outlets as well as parameters with average reservoir area, depth a discharge values. | [`setup_rivers()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_rivers.html#hydromt_wflow.WflowSbmModel.setup_rivers "hydromt_wflow.WflowSbmModel.setup_rivers") |
| [`setup_reservoirs_simple_control()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_reservoirs_simple_control.html#hydromt_wflow.WflowSbmModel.setup_reservoirs_simple_control "hydromt_wflow.WflowSbmModel.setup_reservoirs_simple_control") | Generate maps of controlled reservoir areas and outlets as well as parameters with average reservoir area, demand, min and max target storage capacities and discharge capacity values. | [`setup_rivers()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_rivers.html#hydromt_wflow.WflowSbmModel.setup_rivers "hydromt_wflow.WflowSbmModel.setup_rivers") |
| [`setup_glaciers()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_glaciers.html#hydromt_wflow.WflowSbmModel.setup_glaciers "hydromt_wflow.WflowSbmModel.setup_glaciers") | Generate maps of glacier areas, area fraction and volume fraction, as well as tables with temperature threshold, melting factor and snow-to-ice conversion fraction. | [`setup_basemaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_basemaps.html#hydromt_wflow.WflowSbmModel.setup_basemaps "hydromt_wflow.WflowSbmModel.setup_basemaps") |

#### 2.1.4. Land Use and Vegetation

Defines land use and vegetation properties, including LULC and LAI maps, which influence evapotranspiration and interception processes.

| Method | Explanation | Required Setup Method |
| --- | --- | --- |
| [`setup_lulcmaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_lulcmaps.html#hydromt_wflow.WflowSbmModel.setup_lulcmaps "hydromt_wflow.WflowSbmModel.setup_lulcmaps") | Derive several wflow maps based on landuse- landcover (LULC) raster data. | [`setup_basemaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_basemaps.html#hydromt_wflow.WflowSbmModel.setup_basemaps "hydromt_wflow.WflowSbmModel.setup_basemaps") |
| [`setup_lulcmaps_from_vector()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_lulcmaps_from_vector.html#hydromt_wflow.WflowSbmModel.setup_lulcmaps_from_vector "hydromt_wflow.WflowSbmModel.setup_lulcmaps_from_vector") | Derive several wflow maps based on landuse- landcover (LULC) vector data. | [`setup_basemaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_basemaps.html#hydromt_wflow.WflowSbmModel.setup_basemaps "hydromt_wflow.WflowSbmModel.setup_basemaps") |
| [`setup_lulcmaps_with_paddy()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_lulcmaps_with_paddy.html#hydromt_wflow.WflowSbmModel.setup_lulcmaps_with_paddy "hydromt_wflow.WflowSbmModel.setup_lulcmaps_with_paddy") | Derive several wflow maps based on landuse- landcover (LULC) raster data with paddy rice. | [`setup_soilmaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_soilmaps.html#hydromt_wflow.WflowSbmModel.setup_soilmaps "hydromt_wflow.WflowSbmModel.setup_soilmaps") |
| [`setup_laimaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_laimaps.html#hydromt_wflow.WflowSbmModel.setup_laimaps "hydromt_wflow.WflowSbmModel.setup_laimaps") | Set leaf area index (LAI) climatology maps per month. | [`setup_basemaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_basemaps.html#hydromt_wflow.WflowSbmModel.setup_basemaps "hydromt_wflow.WflowSbmModel.setup_basemaps") |
| [`setup_laimaps_from_lulc_mapping()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_laimaps_from_lulc_mapping.html#hydromt_wflow.WflowSbmModel.setup_laimaps_from_lulc_mapping "hydromt_wflow.WflowSbmModel.setup_laimaps_from_lulc_mapping") | Set leaf area index (LAI) climatology maps per month based on landuse mapping. | [`setup_basemaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_basemaps.html#hydromt_wflow.WflowSbmModel.setup_basemaps "hydromt_wflow.WflowSbmModel.setup_basemaps") |
| [`setup_rootzoneclim()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_rootzoneclim.html#hydromt_wflow.WflowSbmModel.setup_rootzoneclim "hydromt_wflow.WflowSbmModel.setup_rootzoneclim") | Derive an estimate of the rooting depth from hydroclimatic data (as an alternative from the look-up table). The method can be applied for current conditions and future climate change conditions. | [`setup_soilmaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_soilmaps.html#hydromt_wflow.WflowSbmModel.setup_soilmaps "hydromt_wflow.WflowSbmModel.setup_soilmaps"), [`setup_laimaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_laimaps.html#hydromt_wflow.WflowSbmModel.setup_laimaps "hydromt_wflow.WflowSbmModel.setup_laimaps") or equivalent |

#### 2.1.5. Soil

Sets up soil-related data including soil maps and hydraulic properties.

| Method | Explanation | Required Setup Method |
| --- | --- | --- |
| [`setup_soilmaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_soilmaps.html#hydromt_wflow.WflowSbmModel.setup_soilmaps "hydromt_wflow.WflowSbmModel.setup_soilmaps") | Derive several (layered) soil parameters based on a database with physical soil properties using available point-scale (pedo)transfer functions (PTFs) from literature with upscaling rules to ensure flux matching across scales. | [`setup_basemaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_basemaps.html#hydromt_wflow.WflowSbmModel.setup_basemaps "hydromt_wflow.WflowSbmModel.setup_basemaps") |
| [`setup_ksathorfrac()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_ksathorfrac.html#hydromt_wflow.WflowSbmModel.setup_ksathorfrac "hydromt_wflow.WflowSbmModel.setup_ksathorfrac") | Prepare the saturated hydraulic conductivity horizontal ratio from an existing map. | [`setup_basemaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_basemaps.html#hydromt_wflow.WflowSbmModel.setup_basemaps "hydromt_wflow.WflowSbmModel.setup_basemaps") |
| [`setup_ksatver_vegetation()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_ksatver_vegetation.html#hydromt_wflow.WflowSbmModel.setup_ksatver_vegetation "hydromt_wflow.WflowSbmModel.setup_ksatver_vegetation") | Prepare ksatver from soil and vegetation parameters. | [`setup_soilmaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_soilmaps.html#hydromt_wflow.WflowSbmModel.setup_soilmaps "hydromt_wflow.WflowSbmModel.setup_soilmaps"), [`setup_laimaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_laimaps.html#hydromt_wflow.WflowSbmModel.setup_laimaps "hydromt_wflow.WflowSbmModel.setup_laimaps") or equivalent. |

#### 2.1.6. Water demands and Allocation

Defines domestic, irrigation, and other water demand maps and allocation parameters.

| Method | Explanation | Required Setup Method |
| --- | --- | --- |
| [`setup_allocation_areas()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_allocation_areas.html#hydromt_wflow.WflowSbmModel.setup_allocation_areas "hydromt_wflow.WflowSbmModel.setup_allocation_areas") | Create water demand allocation areas. | [`setup_rivers()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_rivers.html#hydromt_wflow.WflowSbmModel.setup_rivers "hydromt_wflow.WflowSbmModel.setup_rivers") |
| [`setup_allocation_surfacewaterfrac()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_allocation_surfacewaterfrac.html#hydromt_wflow.WflowSbmModel.setup_allocation_surfacewaterfrac "hydromt_wflow.WflowSbmModel.setup_allocation_surfacewaterfrac") | Create fraction of surface water used for allocation of the water demands. | [`setup_basemaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_basemaps.html#hydromt_wflow.WflowSbmModel.setup_basemaps "hydromt_wflow.WflowSbmModel.setup_basemaps") |
| [`setup_domestic_demand()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_domestic_demand.html#hydromt_wflow.WflowSbmModel.setup_domestic_demand "hydromt_wflow.WflowSbmModel.setup_domestic_demand") | Create domestic water demand from grid. | [`setup_basemaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_basemaps.html#hydromt_wflow.WflowSbmModel.setup_basemaps "hydromt_wflow.WflowSbmModel.setup_basemaps") |
| [`setup_domestic_demand_from_population()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_domestic_demand_from_population.html#hydromt_wflow.WflowSbmModel.setup_domestic_demand_from_population "hydromt_wflow.WflowSbmModel.setup_domestic_demand_from_population") | Create domestic water demand using demand per capita and gridded population. | [`setup_basemaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_basemaps.html#hydromt_wflow.WflowSbmModel.setup_basemaps "hydromt_wflow.WflowSbmModel.setup_basemaps") |
| [`setup_other_demand()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_other_demand.html#hydromt_wflow.WflowSbmModel.setup_other_demand "hydromt_wflow.WflowSbmModel.setup_other_demand") | Create other water demand (eg industry, livestock). | [`setup_basemaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_basemaps.html#hydromt_wflow.WflowSbmModel.setup_basemaps "hydromt_wflow.WflowSbmModel.setup_basemaps") |
| [`setup_irrigation()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_irrigation.html#hydromt_wflow.WflowSbmModel.setup_irrigation "hydromt_wflow.WflowSbmModel.setup_irrigation") | Create irrigation areas and trigger for paddy and nonpaddy crops from a raster file. | [`setup_lulcmaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_lulcmaps.html#hydromt_wflow.WflowSbmModel.setup_lulcmaps "hydromt_wflow.WflowSbmModel.setup_lulcmaps") or equivalent, [`setup_laimaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_laimaps.html#hydromt_wflow.WflowSbmModel.setup_laimaps "hydromt_wflow.WflowSbmModel.setup_laimaps") or equivalent |
| [`setup_irrigation_from_vector()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_irrigation_from_vector.html#hydromt_wflow.WflowSbmModel.setup_irrigation_from_vector "hydromt_wflow.WflowSbmModel.setup_irrigation_from_vector") | Create irrigation areas and trigger for paddy and nonpaddy crops from a vector file. | [`setup_lulcmaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_lulcmaps.html#hydromt_wflow.WflowSbmModel.setup_lulcmaps "hydromt_wflow.WflowSbmModel.setup_lulcmaps") or equivalent, [`setup_laimaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_laimaps.html#hydromt_wflow.WflowSbmModel.setup_laimaps "hydromt_wflow.WflowSbmModel.setup_laimaps") or equivalent |

#### 2.1.7. Forcing

Sets up meteorological forcing inputs such as precipitation, temperature, and potential evapotranspiration.

| Method | Explanation | Required Setup Method |
| --- | --- | --- |
| [`setup_precip_forcing()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_precip_forcing.html#hydromt_wflow.WflowSbmModel.setup_precip_forcing "hydromt_wflow.WflowSbmModel.setup_precip_forcing") | Setup gridded precipitation forcing at model resolution. | [`setup_basemaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_basemaps.html#hydromt_wflow.WflowSbmModel.setup_basemaps "hydromt_wflow.WflowSbmModel.setup_basemaps") |
| [`setup_precip_from_point_timeseries()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_precip_from_point_timeseries.html#hydromt_wflow.WflowSbmModel.setup_precip_from_point_timeseries "hydromt_wflow.WflowSbmModel.setup_precip_from_point_timeseries") | Setup precipitation forcing from station data at model resolution. | [`setup_basemaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_basemaps.html#hydromt_wflow.WflowSbmModel.setup_basemaps "hydromt_wflow.WflowSbmModel.setup_basemaps") |
| [`setup_temp_pet_forcing()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_temp_pet_forcing.html#hydromt_wflow.WflowSbmModel.setup_temp_pet_forcing "hydromt_wflow.WflowSbmModel.setup_temp_pet_forcing") | Setup gridded temperature and optionally compute reference evapotranspiration forcing at model resolution. | [`setup_basemaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_basemaps.html#hydromt_wflow.WflowSbmModel.setup_basemaps "hydromt_wflow.WflowSbmModel.setup_basemaps") |
| [`setup_pet_forcing()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_pet_forcing.html#hydromt_wflow.WflowSbmModel.setup_pet_forcing "hydromt_wflow.WflowSbmModel.setup_pet_forcing") | Setup gridded reference evapotranspiration forcing at model resolution. | [`setup_basemaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_basemaps.html#hydromt_wflow.WflowSbmModel.setup_basemaps "hydromt_wflow.WflowSbmModel.setup_basemaps") |

#### 2.1.8. States

Defines initial hydrological state variables such as soil moisture and groundwater storage.

| Method | Explanation | Required Setup Method |
| --- | --- | --- |
| [`setup_cold_states()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_cold_states.html#hydromt_wflow.WflowSbmModel.setup_cold_states "hydromt_wflow.WflowSbmModel.setup_cold_states") | Setup wflow cold states based on data in staticmaps. | [`setup_soilmaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_soilmaps.html#hydromt_wflow.WflowSbmModel.setup_soilmaps "hydromt_wflow.WflowSbmModel.setup_soilmaps"), [`setup_constant_pars()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_constant_pars.html#hydromt_wflow.WflowSbmModel.setup_constant_pars "hydromt_wflow.WflowSbmModel.setup_constant_pars"), [`setup_reservoirs_no_control()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_reservoirs_no_control.html#hydromt_wflow.WflowSbmModel.setup_reservoirs_no_control "hydromt_wflow.WflowSbmModel.setup_reservoirs_no_control"), [`setup_reservoirs_simple_control()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_reservoirs_simple_control.html#hydromt_wflow.WflowSbmModel.setup_reservoirs_simple_control "hydromt_wflow.WflowSbmModel.setup_reservoirs_simple_control"), [`setup_glaciers()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_glaciers.html#hydromt_wflow.WflowSbmModel.setup_glaciers "hydromt_wflow.WflowSbmModel.setup_glaciers"), [`setup_irrigation()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_irrigation.html#hydromt_wflow.WflowSbmModel.setup_irrigation "hydromt_wflow.WflowSbmModel.setup_irrigation") or equivalent |

#### 2.1.9. Output Locations

Defines outlets, gauges, and spatial masks used for reporting model results.

| Method | Explanation | Required Setup Method |
| --- | --- | --- |
| [`setup_outlets()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_outlets.html#hydromt_wflow.WflowSbmModel.setup_outlets "hydromt_wflow.WflowSbmModel.setup_outlets") | Set the default gauge map based on basin outlets. | [`setup_rivers()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_rivers.html#hydromt_wflow.WflowSbmModel.setup_rivers "hydromt_wflow.WflowSbmModel.setup_rivers") |
| [`setup_gauges()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_gauges.html#hydromt_wflow.WflowSbmModel.setup_gauges "hydromt_wflow.WflowSbmModel.setup_gauges") | Set the default gauge map based on a gauges\_fn data. | [`setup_rivers()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_rivers.html#hydromt_wflow.WflowSbmModel.setup_rivers "hydromt_wflow.WflowSbmModel.setup_rivers") |
| [`setup_areamap()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_areamap.html#hydromt_wflow.WflowSbmModel.setup_areamap "hydromt_wflow.WflowSbmModel.setup_areamap") | Setup area map from vector data to save wflow outputs for specific area. | [`setup_basemaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_basemaps.html#hydromt_wflow.WflowSbmModel.setup_basemaps "hydromt_wflow.WflowSbmModel.setup_basemaps") |

#### 2.1.10. Other Setup Methods

Additional high-level utilities to modify model geometry, link external models, or upgrade model versions.

| Method | Explanation | Required Setup Method |
| --- | --- | --- |
| [`setup_1dmodel_connection()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_1dmodel_connection.html#hydromt_wflow.WflowSbmModel.setup_1dmodel_connection "hydromt_wflow.WflowSbmModel.setup_1dmodel_connection") | Setup subbasins and gauges to save results from wflow to be used in 1D river models. | [`setup_rivers()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_rivers.html#hydromt_wflow.WflowSbmModel.setup_rivers "hydromt_wflow.WflowSbmModel.setup_rivers") |
| [`setup_grid_from_raster()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_grid_from_raster.html#hydromt_wflow.WflowSbmModel.setup_grid_from_raster "hydromt_wflow.WflowSbmModel.setup_grid_from_raster") | Setup staticmaps from raster to add parameters from direct data. | [`setup_basemaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_basemaps.html#hydromt_wflow.WflowSbmModel.setup_basemaps "hydromt_wflow.WflowSbmModel.setup_basemaps") |
| [`upgrade_to_v1_wflow()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.upgrade_to_v1_wflow.html#hydromt_wflow.WflowSbmModel.upgrade_to_v1_wflow "hydromt_wflow.WflowSbmModel.upgrade_to_v1_wflow") | Upgrade a model from a Wflow.jl 0.x to 1.0. |  |
| [`clip()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.clip.html#hydromt_wflow.WflowSbmModel.clip "hydromt_wflow.WflowSbmModel.clip") | Clip a sub-region of an existing model. |  |

### 2.2. Model components

The following table provides an overview of which [`WflowSbmModel`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.html#hydromt_wflow.WflowSbmModel "hydromt_wflow.WflowSbmModel") components contains which Wflow in- and output files. The files are read and written with the associated read- and write- methods, i.e. `read()` and `write()` for the `config` component.

| [`WflowSbmModel`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.html#hydromt_wflow.WflowSbmModel "hydromt_wflow.WflowSbmModel") model | Wflow files |
| --- | --- |
| `config` | wflow\_sbm.toml |
| `staticmaps` | staticmaps.nc |
| `geoms` | geometries from the staticgeoms folder (basins.geojson, rivers.geojson etc.) |
| `forcing` | inmaps.nc |
| `states` | instates.nc |
| `tables` | tabular data (csv format, e.g. lake\_hq.csv, lake\_sh.csv) |
| `output_grid` | output.nc (defined in \[output.netcdf\_grid\] TOML section) |
| `output_scalar` | output\_scalar.nc (defined in \[output.netcdf\_scalar\] TOML section) |
| `output_csv` | output.csv (defined in \[output.csv\] TOML section) |

## 3. Building a model

This plugin allows to build a complete model from available data. Once the configuration and data libraries are set, you can build a model by using:

```
hydromt build wflow_sbm path/to/built_model -i wflow_build.yml -d data_sources.yml -vvv
```

Note

From HydroMT-wflow version 1.0 onwards (Hydromt version > 1.0.0), the region argument (-r & –region) has been moved to `setup_basemaps` function arguments and is no longer available via cli.

The recommended [region options](https://deltares.github.io/hydromt/latest/user_guide/models/model_region.html) for a proper implementation of this model are:

- basin
- subbasin

The coordinate reference system (CRS) of the model will be the same as the one of the input hydrography data. If the region is specified using point coordinates or a bounding box, the coordinates used should match the CRS of the hydrography data. If the user wants to use a different CRS, we advise to reproject the hydrography data to the desired CRS before building the model. You can find some examples on how to do this in the notebook: [here](https://deltares.github.io/hydromt_wflow/stable/_examples/prepare_ldd.html#example-prepare-ldd).

### 3.1. Configuration file

Settings to build or update a Wflow model are managed in a configuration file. In this file, every option from each [model method](https://deltares.github.io/hydromt_wflow/stable/user_guide/2_sbm_model/1_methods_components.html#model-methods) can be changed by the user in its corresponding section.

Note that the order in which the components are listed in the configuration file is important:

- setup\_basemaps should always be run first to determine the model domain
- setup\_rivers should be run right after setup\_basemaps as it influences several other setup components (reservoirs, riverwidth, gauges)

Below is an example configuration file that can be used to build a complete Wflow model[`.yml file`](https://deltares.github.io/hydromt_wflow/stable/_downloads/91e3b6c950554d08e2fce9e28e2a82f9/wflow_build.yml). Each section corresponds to a model component with the same name.

```yaml
steps:
  - setup_config:                 # options parsed to wflow toml file <section>.<option>
      data:
        time.starttime: 2010-01-01T00:00:00
        time.endtime: 2010-03-31T00:00:00
        time.timestepsecs: 86400
        input.path_forcing: inmaps-era5-2010.nc
        output.netcdf_grid.path: output.nc
        output.netcdf_grid.compressionlevel: 1
        output.netcdf_grid.variables.river_water__volume_flow_rate: river_q

  - setup_basemaps:
      hydrography_fn: merit_hydro   # source hydrography data {merit_hydro, merit_hydro_1k}
      basin_index_fn: merit_hydro_index # source of basin index corresponding to hydrography_fn
      upscale_method: ihu           # upscaling method for flow direction data, by default 'ihu'
      res: 0.00833           # build the model at a 30 arc sec (~1km) resolution
      region: # model region
        subbasin: [12.2051, 45.8331] # derive a subbasin with its outlet at the given x,y coordinates
        strord: 4 # snapped to a river with minimum stream order (strord) of 4
        bounds: [11.70, 45.35, 12.95, 46.70] # within the given bounds [xmin, ymin, xmax, ymax] (WGS84)

  - setup_rivers:
      hydrography_fn: merit_hydro      # source hydrography data, should correspond to hydrography_fn in setup_basemaps
      river_geom_fn: hydro_rivers_lin # river source data with river width and bankfull discharge
      river_upa: 30               # minimum upstream area threshold for the river map [km2]
      rivdph_method: powlaw           # method to estimate depth {'powlaw', 'manning', 'gvf'}
      min_rivdph: 1                # minimum river depth [m]
      min_rivwth: 30               # minimum river width [m]
      slope_len: 2000             # length over which tp calculate river slope [m]
      smooth_len: 5000             # length over which to smooth river depth and river width [m]
      river_routing: kinematic_wave   # {'kinematic_wave', 'local_inertial'}

  - setup_river_roughness:

  # - setup_floodplains: # if 2D floodplains are required
  #     hydrography_fn: merit_hydro      # source hydrography data, should correspond to hydrography_fn in setup_basemaps
  #     floodplain_type: 2d #  # If two-dimensional floodplains are required
  #     elevtn_map: land_elevation  # {'land_elevation', 'meta_subgrid_elevation'}

  # - setup_floodplains: # if 1D floodplains are required
  #     hydrography_fn: merit_hydro      # source hydrography data, should correspond to hydrography_fn in setup_basemaps
  #     floodplain_type: 1d    # If one-dimensional floodplains are required
  #     flood_depths: # flood depths at which a volume is derived
  #       - 0.5
  #       - 1.0
  #       - 1.5
  #       - 2.0
  #       - 2.5
  #       - 3.0
  #       - 4.0
  #       - 5.0

  - setup_reservoirs_simple_control:
      reservoirs_fn: hydro_reservoirs  # source for reservoirs shape and attributes
      timeseries_fn: gww           # additional source for reservoir are timeseries to compute reservoirs, Either 'gww' using gwwapi or 'jrc' using hydroengine.
      min_area: 1.0           # minimum lake area to consider [km2]

  - setup_reservoirs_no_control:
      reservoirs_fn: hydro_lakes   # source for uncontrolled reservoirs (e.g. lakes) based on hydroLAKES: {hydro_lakes}
      min_area: 10.0          # minimum reservoir area to consider [km2]

  - setup_glaciers:
      glaciers_fn: rgi           # source for glaciers based on Randolph Glacier Inventory {rgi}
      min_area: 0.0           # minimum glacier area to consider [km2]

  - setup_lulcmaps:
      lulc_fn : globcover_2009     # source for lulc maps: {globcover, vito, corine}
      lulc_mapping_fn: globcover_mapping_default  # default mapping for lulc classes

  - setup_laimaps:
      lai_fn: modis_lai     # source for Leaf Area Index: {modis_lai}

  - setup_soilmaps:
      soil_fn: soilgrids     # source for soilmaps: {soilgrids}
      ptf_ksatver: brakensiek    # pedotransfer function to calculate hydraulic conductivity: {brakensiek, cosby}

  - setup_outlets:
      river_only: True

  - setup_gauges:
      gauges_fn: grdc          # if not None add gaugemap. Either a path or known gauges_fn: {grdc}
      snap_to_river: True          # if True snaps gauges from source to river
      derive_subcatch: False         # if True derive subcatch map based on gauges.

  - setup_precip_forcing:
      precip_fn: era5          # source for precipitation.
      precip_clim_fn:          # source for high resolution climatology to correct precipitation if any.

  - setup_temp_pet_forcing:
      temp_pet_fn: era5          # source for temperature and potential evapotranspiration.
      press_correction: True          # if True temperature is corrected with elevation lapse rate.
      temp_correction: True          # if True pressure is corrected with elevation lapse rate.
      dem_forcing_fn: era5_orography # source of elevation grid corresponding to temp_pet_fn. Used for lapse rate correction.
      pet_method: debruin       # method to compute PET: {debruin, makkink}
      skip_pet: False         # if True, only temperature is prepared.

  - setup_constant_pars:
      "subsurface_water__horizontal_to_vertical_saturated_hydraulic_conductivity_ratio": 100
      "snowpack__degree_day_coefficient": 3.75653
      "soil_surface_water__infiltration_reduction_parameter": 0.038
      "vegetation_canopy_water__mean_evaporation_to_mean_precipitation_ratio": 0.11
      "compacted_soil_surface_water__infiltration_capacity": 10
      "soil_water_saturated_zone_bottom__max_leakage_volume_flux": 0
      "soil_wet_root__sigmoid_function_shape_parameter": -500
      "atmosphere_air__snowfall_temperature_threshold": 0
      "atmosphere_air__snowfall_temperature_interval": 2
      "snowpack__melting_temperature_threshold": 0
      "snowpack__liquid_water_holding_capacity": 0.1
      "glacier_ice__degree_day_coefficient": 3
      "glacier_firn_accumulation__snowpack_dry_snow_leq_depth_fraction": 0.001
      "glacier_ice__melting_temperature_threshold": 0
```

For an example of how to build a model using this configuration file, see the example notebook: [here](https://deltares.github.io/hydromt_wflow/stable/_examples/build_model.html#example-build-model).

### 3.2. Selecting data

Data sources in HydroMT are provided in one of several yaml libraries. These libraries contain required information on the different data sources so that HydroMT can process them for the different models. There are three ways for the user to select which data libraries to use:

- For testing and examples purposes, HydroMT can use the data stored in the [hydromt-artifacts](https://github.com/DirkEilander/hydromt-artifacts) which contains an extract of global data for a small region around the Piave river in Northern Italy. to use this predefined catalog, the user can add **\-d artifact\_data** to the build / update command line.
- Another options for Deltares users is to select the deltares\_data library (requires access to the Deltares P-drive). In the command lines examples below, this is done by adding **\-d deltares\_data** predefined catalog to the build / update command line.
- Finally, the user can prepare its own yaml catalog (see [HydroMT documentation](https://deltares.github.io/hydromt/latest/index) to check the guidelines). These user libraries can be added either in the command line using the **\-d** option and path/to/yaml or in the **ini file** with the **data\_libs** option in the \[global\] sections.

### 3.3. Examples

To know more about building a Wflow-SBM model from scratch, check the following examples:

- [Optional first step: preparing flow directions from DEM](https://deltares.github.io/hydromt_wflow/stable/_examples/prepare_ldd.html#example-prepare-ldd)
- [Building a Wflow SBM model from command line](https://deltares.github.io/hydromt_wflow/stable/_examples/build_model.html#example-build-model)

## 4. Updating a model

To add or change one or more components of an existing Wflow model the `update` method can be used.

**Steps in brief:**

1. You have an **existing model** schematization. This model does not have to be complete.
2. Prepare or use a pre-defined **data catalog** with all the required data sources, see [working with data](https://deltares.github.io/hydromt/latest/user_guide/data_catalog/data_overview.html).
3. Prepare a **model configuration** with the methods that you want to use to add or change components of your model: see [model configuration](https://deltares.github.io/hydromt/latest/user_guide/models/model_workflow.html).
4. **Update** your model using the CLI or Python interface.

```
activate hydromt-wflow
hydromt update wflow_sbm path/to/model_to_update -o path/to/updated_model -i wflow_update.yml -d data_sources.yml -vvv
```

Note

By default, the updated model will overwrite your existing one. To save the updated model in a different folder, use the -o path/to/updated\_model option of the CLI.

Tip

By default all model data is written at the end of the update method. If your update however only affects a certain model data (e.g. staticmaps or forcing) you can add a write\_\* method (e.g. write\_grid, write\_forcing) to the.yml file and only these data will be written.

Note that the model config is often changed as part of the a model method and write\_config should thus be added to the.yml file to keep the model data and config consistent.

### 4.1. Examples

Several examples of updating different input data for a Wflow SBM model are available:

- [Update land use](https://deltares.github.io/hydromt_wflow/stable/_examples/update_model_landuse.html#example-update-model-landuse)
- [Update meteo forcing](https://deltares.github.io/hydromt_wflow/stable/_examples/update_model_forcing.html#example-update-model-forcing)
- [Add gauges locations](https://deltares.github.io/hydromt_wflow/stable/_examples/update_model_gauges.html#example-update-model-gauges)
- [Add water demands and allocations](https://deltares.github.io/hydromt_wflow/stable/_examples/update_model_water_demand.html#example-update-model-water-demand)
- [Connect Wflow SBM to a 1D (hydraulic) model](https://deltares.github.io/hydromt_wflow/stable/_examples/connect_to_1d_model.html#example-connect-to-1d-model)

## 5. Clipping a model

This plugin allows to clip the following parts of an existing model for a smaller region from command line:

- staticmaps
- forcing
- states
- geoms
- config (update reservoir settings)
- tables (update reservoir rating curves)

To clip a smaller model from an existing one use the `update` CLI command with the **clip** method:

```
activate hydromt_wflow
hydromt update wflow_sbm path/to/model_to_clip -o path/to/clipped_model -i clip_config.yml -v
```

Here is an example of the clip config:

```yaml
steps:
  - clip:
      region: {"basin": [x, y]} # region to clip the model too
      inverse_clip: false # whether to clip outside or inside the region
      clip_states: true # whether to clip states
      clip_forcing: true # whether to clip forcing
```

As for building, the recommended [region options](https://deltares.github.io/hydromt/latest/user_guide/models/model_region.html) for a proper implementation of the clipped model are:

- basin
- subbasin

See the following model API:

- [`clip()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.clip.html#hydromt_wflow.WflowSbmModel.clip "hydromt_wflow.WflowSbmModel.clip")

### 5.1. Examples

To know more about clipping a Wflow-SBM model, check the following example:

- [Clipping a Wflow SBM model](https://deltares.github.io/hydromt_wflow/stable/_examples/clip_model.html#example-clip-model)

## 6. Model methods and components

This plugin helps you preparing or updating several components of a Wflow Sediment model such as topography information, landuse or soil. The main interactions are available from the HydroMT Command Line Interface and allow you to configure HydroMT in order to build or update or clip Wflow Sediment models.

When building or updating a model from command line a [model region](https://deltares.github.io/hydromt/latest/user_guide/models/model_region.html); a model setup [configuration](https://deltares.github.io/hydromt_wflow/stable/user_guide/2_sbm_model/2_build.html#model-config-sed) (.yml file) with model components and options and, optionally, a [data sources](https://deltares.github.io/hydromt/latest/user_guide/data_catalog/data_overview.html) (.yml) file should be prepared.

### 6.1. Model setup methods

An overview of the available Wflow Sediment model setup components is provided in the tables below. When using HydroMT from the command line only the setup components are exposed. Click on a specific method see its documentation.

#### 6.1.1. Configuration (TOML)

| Method | Explanation | Required Setup Method |
| --- | --- | --- |
| [`setup_config()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.setup_config.html#hydromt_wflow.WflowSedimentModel.setup_config "hydromt_wflow.WflowSedimentModel.setup_config") | Update config with a dictionary |  |
| [`setup_config_output_timeseries()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.setup_config_output_timeseries.html#hydromt_wflow.WflowSedimentModel.setup_config_output_timeseries "hydromt_wflow.WflowSedimentModel.setup_config_output_timeseries") | This method add a new variable/column to the netcf/csv output section of the toml based on a selected gauge/area map. | [`setup_basemaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.setup_basemaps.html#hydromt_wflow.WflowSedimentModel.setup_basemaps "hydromt_wflow.WflowSedimentModel.setup_basemaps") |
| [`setup_constant_pars()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.setup_constant_pars.html#hydromt_wflow.WflowSedimentModel.setup_constant_pars "hydromt_wflow.WflowSedimentModel.setup_constant_pars") | Setup constant parameter maps. |  |

#### 6.1.2. Topography and Rivers

| Method | Explanation | Required Setup Method |
| --- | --- | --- |
| [`setup_basemaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.setup_basemaps.html#hydromt_wflow.WflowSedimentModel.setup_basemaps "hydromt_wflow.WflowSedimentModel.setup_basemaps") | This component sets the region of interest and res (resolution in degrees) of the model. |  |
| [`setup_rivers()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.setup_rivers.html#hydromt_wflow.WflowSedimentModel.setup_rivers "hydromt_wflow.WflowSedimentModel.setup_rivers") | This component sets the all river parameter maps. | [`setup_basemaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.setup_basemaps.html#hydromt_wflow.WflowSedimentModel.setup_basemaps "hydromt_wflow.WflowSedimentModel.setup_basemaps") |
| [`setup_riverbedsed()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.setup_riverbedsed.html#hydromt_wflow.WflowSedimentModel.setup_riverbedsed "hydromt_wflow.WflowSedimentModel.setup_riverbedsed") | Setup sediments based river bed characteristics maps. | [`setup_rivers()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.setup_rivers.html#hydromt_wflow.WflowSedimentModel.setup_rivers "hydromt_wflow.WflowSedimentModel.setup_rivers") |

#### 6.1.3. Reservoirs

| Method | Explanation | Required Setup Method |
| --- | --- | --- |
| [`setup_natural_reservoirs()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.setup_natural_reservoirs.html#hydromt_wflow.WflowSedimentModel.setup_natural_reservoirs "hydromt_wflow.WflowSedimentModel.setup_natural_reservoirs") | This component generates maps of lake (natural reservoirs) areas and outlets as well as parameters such as average area. | [`setup_rivers()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.setup_rivers.html#hydromt_wflow.WflowSedimentModel.setup_rivers "hydromt_wflow.WflowSedimentModel.setup_rivers") |
| [`setup_reservoirs()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.setup_reservoirs.html#hydromt_wflow.WflowSedimentModel.setup_reservoirs "hydromt_wflow.WflowSedimentModel.setup_reservoirs") | This component generates maps of reservoir areas and outlets as well as parameters such as average area. | [`setup_rivers()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.setup_rivers.html#hydromt_wflow.WflowSedimentModel.setup_rivers "hydromt_wflow.WflowSedimentModel.setup_rivers") |

#### 6.1.4. Land Use and Vegetation

| Method | Explanation | Required Setup Method |
| --- | --- | --- |
| [`setup_lulcmaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.setup_lulcmaps.html#hydromt_wflow.WflowSedimentModel.setup_lulcmaps "hydromt_wflow.WflowSedimentModel.setup_lulcmaps") | This component derives several wflow maps based on landuse- landcover (LULC) raster data. | [`setup_basemaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.setup_basemaps.html#hydromt_wflow.WflowSedimentModel.setup_basemaps "hydromt_wflow.WflowSedimentModel.setup_basemaps") |
| [`setup_lulcmaps_from_vector()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.setup_lulcmaps_from_vector.html#hydromt_wflow.WflowSedimentModel.setup_lulcmaps_from_vector "hydromt_wflow.WflowSedimentModel.setup_lulcmaps_from_vector") | This component derives several wflow maps based on landuse- landcover (LULC) vector data. | [`setup_basemaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.setup_basemaps.html#hydromt_wflow.WflowSedimentModel.setup_basemaps "hydromt_wflow.WflowSedimentModel.setup_basemaps") |
| [`setup_canopymaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.setup_canopymaps.html#hydromt_wflow.WflowSedimentModel.setup_canopymaps "hydromt_wflow.WflowSedimentModel.setup_canopymaps") | Setup sediments based canopy height maps. | [`setup_basemaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.setup_basemaps.html#hydromt_wflow.WflowSedimentModel.setup_basemaps "hydromt_wflow.WflowSedimentModel.setup_basemaps") |

#### 6.1.5. Soil

| Method | Explanation | Required Setup Method |
| --- | --- | --- |
| [`setup_soilmaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.setup_soilmaps.html#hydromt_wflow.WflowSedimentModel.setup_soilmaps "hydromt_wflow.WflowSedimentModel.setup_soilmaps") | Setup sediments based soil parameter maps. | [`setup_basemaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.setup_basemaps.html#hydromt_wflow.WflowSedimentModel.setup_basemaps "hydromt_wflow.WflowSedimentModel.setup_basemaps") |

#### 6.1.6. Output Locations

| Method | Explanation | Required Setup Method |
| --- | --- | --- |
| [`setup_outlets()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.setup_outlets.html#hydromt_wflow.WflowSedimentModel.setup_outlets "hydromt_wflow.WflowSedimentModel.setup_outlets") | This method sets the default gauge map based on basin outlets. | [`setup_rivers()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.setup_rivers.html#hydromt_wflow.WflowSedimentModel.setup_rivers "hydromt_wflow.WflowSedimentModel.setup_rivers") |
| [`setup_gauges()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.setup_gauges.html#hydromt_wflow.WflowSedimentModel.setup_gauges "hydromt_wflow.WflowSedimentModel.setup_gauges") | This method sets the default gauge map based on a gauges\_fn data. | [`setup_rivers()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.setup_rivers.html#hydromt_wflow.WflowSedimentModel.setup_rivers "hydromt_wflow.WflowSedimentModel.setup_rivers") |
| [`setup_areamap()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.setup_areamap.html#hydromt_wflow.WflowSedimentModel.setup_areamap "hydromt_wflow.WflowSedimentModel.setup_areamap") | Setup area map from vector data to save wflow outputs for specific area. | [`setup_basemaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.setup_basemaps.html#hydromt_wflow.WflowSedimentModel.setup_basemaps "hydromt_wflow.WflowSedimentModel.setup_basemaps") |

#### 6.1.7. Other methods

| Method | Explanation | Required Setup Method |
| --- | --- | --- |
| [`setup_grid_from_raster()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.setup_grid_from_raster.html#hydromt_wflow.WflowSedimentModel.setup_grid_from_raster "hydromt_wflow.WflowSedimentModel.setup_grid_from_raster") | Setup staticmaps from raster to add parameters from direct data. | [`setup_basemaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.setup_basemaps.html#hydromt_wflow.WflowSedimentModel.setup_basemaps "hydromt_wflow.WflowSedimentModel.setup_basemaps") |
| [`upgrade_to_v1_wflow()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.upgrade_to_v1_wflow.html#hydromt_wflow.WflowSedimentModel.upgrade_to_v1_wflow "hydromt_wflow.WflowSedimentModel.upgrade_to_v1_wflow") | Upgrade a model from a Wflow.jl 0.x to 1.0. |  |
| [`clip()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.clip.html#hydromt_wflow.WflowSedimentModel.clip "hydromt_wflow.WflowSedimentModel.clip") | Clip a sub-region of an existing model. |  |

### 6.2. Model components

The following table provides an overview of which [`WflowSedimentModel`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.html#hydromt_wflow.WflowSedimentModel "hydromt_wflow.WflowSedimentModel") components contains which Wflow Sediment in- and output files. The files are read and written with the associated read- and write- methods, i.e. `read_config()` and `write_config()` for the `config` component.

| [`WflowSedimentModel`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.html#hydromt_wflow.WflowSedimentModel "hydromt_wflow.WflowSedimentModel") model | Wflow sediment files |
| --- | --- |
| `config` | wflow\_sediment.toml |
| `staticmaps` | staticmaps.nc |
| `geoms` | geometries from the staticgeoms folder (basins.geojson, rivers.geojson etc.) |
| `forcing` | inmaps.nc |
| `states` | instates.nc |
| `output_grid` | output.nc (defined in \[output.netcdf\_grid\] TOML section) |
| `output_scalar` | output\_scalar.nc (defined in \[output.netcdf\_scalar\] TOML section) |
| `output_csv` | output.csv (defined in \[output.csv\] TOML section) |

## 7. Building a model

This plugin allows to build a complete Wflow Sediment model from available data. Once the configuration and data libraries are set, you can build a model by using:

```
activate hydromt-wflow
hydromt build wflow_sediment path/to/built_model -i wflow_sediment_build.yml -d data_sources.yml -vvv
```

Note

From HydroMT version 0.7.0 onwards the region argument is optional and should be preceded by a -r or --region flag. The resolution (previously -r) argument has been moved to the setup\_basemaps section in the.yml configuration file. From HydroMT version 1.0 onwards, the region argument has been moved to `setup_basemaps` function arguments and is no longer available via cli.

The recommended [region options](https://deltares.github.io/hydromt/latest/user_guide/models/model_region.html) for a proper implementation of the Wflow Sediment model are:

- basin
- subbasin

### 7.1. Configuration file

Settings to build or update a Wflow Sediment model are managed in a configuration file. In this file, every option from each [model component](https://deltares.github.io/hydromt_wflow/stable/user_guide/3_sediment_model/1_methods_components.html#model-methods-sed) can be changed by the user in its corresponding section.

Note that the order in which the components are listed in the configuration file is important:

- setup\_basemaps should always be run first to determine the model domain
- setup\_rivers should be run right after setup\_basemaps as it influences several other setup components (lakes, reservoirs, riverbedsed, floodplains, gauges)

Below is an example configuration file that can be used to build a complete Wflow Sediment model[`.yml file`](https://deltares.github.io/hydromt_wflow/stable/_downloads/998603dba710c7a5357833971adbd4bb/wflow_sediment_build.yml). Each section corresponds to a model component with the same name.

```yaml
steps:
  - setup_config:                     # options parsed to wflow toml file <section>.<option>
      data:
        time.starttime: "2010-02-02T00:00:00"
        time.endtime: "2010-02-10T00:00:00"
        time.timestepsecs: 86400
        output.netcdf_grid.path: output.nc
        output.netcdf_grid.compressionlevel: 1
        output.netcdf_grid.variables.soil_erosion__mass_flow_rate: soil_loss
        output.netcdf_grid.variables.river_water_sediment__suspended_mass_concentration: suspended_solids

  - setup_basemaps:
      hydrography_fn: merit_hydro      # source hydrography data {merit_hydro, merit_hydro_1k}
      basin_index_fn: merit_hydro_index # source of basin index corresponding to hydrography_fn
      upscale_method: ihu              # upscaling method for flow direction data, by default 'ihu'
      res: 0.00833           # build the model at a 30 arc sec (~1km) resolution
      region:
        subbasin: [12.2051, 45.8331]
        strord: 4
        bounds: [11.70, 45.35, 12.95, 46.70]

  - setup_rivers:
      hydrography_fn: merit_hydro      # source hydrography data, should correspond to hydrography_fn in setup_basemaps
      river_geom_fn: hydro_rivers_lin # river source data with river width and bankfull discharge
      river_upa: 30               # minimum upstream area threshold for the river map [km2]
      min_rivwth: 30               # minimum river width [m]
      slope_len: 2000             # length over which tp calculate river slope [m]
      smooth_len: 5000             # length over which to smooth river depth and river width [m]

  - setup_reservoirs:
      reservoirs_fn: hydro_reservoirs  # source for reservoirs shape and attributes
      min_area: 1.0           # minimum lake area to consider [km2]
      trapping_default: 1.0   # default trapping efficiency for reservoirs [0-1]

  - setup_natural_reservoirs:
      reservoirs_fn: hydro_lakes      # source for lakes based on hydroLAKES: {hydro_lakes}
      min_area: 1.0              # minimum reservoir area to consider [km2]

  - setup_riverbedsed:

  - setup_lulcmaps:
      lulc_fn: globcover_2009        # source for lulc maps: {globcover, vito, corine}
      lulc_mapping_fn: globcover_mapping_default  # default mapping for lulc classes

  - setup_canopymaps:
      canopy_fn: simard           # source for vegetation canopy height: {simard}

  - setup_soilmaps:
      soil_fn: soilgrids        # source for soilmaps: {soilgrids}
      usle_k_method: renard      # method to compute the USLE K factor: {renard, epic}
      add_aggregates: True      # if True add small and large aggregates to soil composition

  - setup_outlets:
      river_only: True

  - setup_gauges:
      gauges_fn: grdc             # If not None add gaugemap. Either a path or known gauges_fn: {grdc}
      snap_to_river: True             # If True snaps gauges from source to river
      derive_subcatch: False            # if True derive subcatch map based on gauges.

  - setup_constant_pars:             # constant parameters values
      river_water_sediment__bagnold_transport_capacity_coefficient: 0.0000175
      river_water_sediment__bagnold_transport_capacity_exponent: 1.4
      soil_erosion__answers_overland_flow_factor: 0.9
      soil_erosion__eurosem_exponent: 2.0
      sediment__particle_density: 2650
      clay__mean_diameter: 2
      silt__mean_diameter: 10
      "sediment_small_aggregates__mean_diameter": 30
      sand__mean_diameter: 200
      "sediment_large_aggregates__mean_diameter": 500
      gravel__mean_diameter: 2000
```

### 7.2. Selecting data

Data sources in HydroMT are provided in one of several yaml libraries. These libraries contain required information on the different data sources so that HydroMT can process them for the different models. There are three ways for the user to select which data libraries to use:

- For testing and examples purposes, HydroMT can use the data stored in the [hydromt-artifacts](https://github.com/DirkEilander/hydromt-artifacts) which contains an extract of global data for a small region around the Piave river in Northern Italy. to use this predefined catalog, the user can add **\-d artifact\_data** to the build / update command line.
- Another options for Deltares users is to select the deltares\_data library (requires access to the Deltares P-drive). In the command lines examples below, this is done by adding **\-d deltares\_data** predefined catalog to the build / update command line.
- Finally, the user can prepare its own yaml catalog (see [HydroMT documentation](https://deltares.github.io/hydromt/latest/index) to check the guidelines). These user libraries can be added either in the command line using the **\-d** option and path/to/yaml or in the **ini file** with the **data\_libs** option in the \[global\] sections.

### 7.3. Extending a Wflow (SBM) model with a Wflow Sediment model

If you already have a Wflow model and you want to extend it in order to include sediment as well, then you do not need to build the Wflow Sediment model from scratch. You can instead `update` the Wflow model with the additional components needed by Wflow Sediment. These components are available in a template [`.yml file`](https://deltares.github.io/hydromt_wflow/stable/_downloads/a86c6c62a10521074a9063267a17721b/wflow_extend_sediment.yml) and shown below. The corresponding command line would be:

```
activate hydromt-wflow
hydromt update wflow_sediment path/to/wflow_model_to_extend -o path/to/wflow_sediment_model -i wflow_extend_sediment.yml -d data_sources.yml -vvv
```

```yaml
steps:
  - config.read: # read a template config file for wflow sediment to be able to extend from a sbm model
      filename: data/wflow_sediment.toml

  - setup_config: # options parsed to wflow toml file <section>.<option>
      data:
        time.starttime: 2010-01-01T00:00:00
        time.endtime: 2010-03-31T00:00:00
        time.timestepsecs: 86400
        output.netcdf_grid.path: output.nc
        output.netcdf_grid.compressionlevel: 1
        output.netcdf_grid.variables.soil_erosion__mass_flow_rate: soil_loss
        output.netcdf_grid.variables.river_water_sediment__suspended_mass_concentration: suspended_solids

  - setup_riverbedsed:
      bedsed_mapping_fn:       # path to a mapping csv file from streamorder to river bed particles characteristics if any, else default is used

  # If you do have reservoirs, you can re-run setup_natural_reservoirs and setup_reservoirs to add it
  - setup_reservoirs:
      reservoirs_fn: hydro_reservoirs  # source for reservoirs shape and attributes
      min_area: 1.0           # minimum lake area to consider [km2]
      trapping_default: 1.0   # default trapping efficiency for reservoirs [0-1]

  - setup_natural_reservoirs:
      reservoirs_fn: hydro_lakes   # source for uncontrolled reservoirs (e.g. lakes) based on hydroLAKES: {hydro_lakes}
      min_area: 10.0          # minimum reservoir area to consider [km2]

  - setup_lulcmaps:
      lulc_fn: globcover_2009        # source for lulc maps: {globcover, vito, corine}
      lulc_mapping_fn: globcover_mapping_default  # default mapping for lulc classes

  - setup_canopymaps:
      canopy_fn: simard           # source for vegetation canopy height: {simard}

  - setup_soilmaps:
      soil_fn: soilgrids        # source for soilmaps: {soilgrids}
      usle_k_method: renard           # method to compute the USLE K factor: {renard, epic}
      add_aggregates: True      # if True add small and large aggregates to soil composition

  - setup_outlets:
      river_only: True

  - setup_gauges:
      gauges_fn: grdc             # If not None add gaugemap. Either a path or known gauges_fn: {grdc}
      snap_to_river: True             # If True snaps gauges from source to river
      derive_subcatch: False            # if True derive subcatch map based on gauges.

  - setup_constant_pars:               # constant parameters values
      river_water_sediment__bagnold_transport_capacity_coefficient: 0.0000175
      river_water_sediment__bagnold_transport_capacity_exponent: 1.4
      soil_erosion__answers_overland_flow_factor: 0.9
      soil_erosion__eurosem_exponent: 2.0
      sediment__particle_density: 2650
      clay__mean_diameter: 2
      silt__mean_diameter: 10
      "sediment_small_aggregates__mean_diameter": 30
      sand__mean_diameter: 200
      "sediment_large_aggregates__mean_diameter": 500
      gravel__mean_diameter: 2000
      "reservoir_water_sediment__bedload_trapping_efficiency": 1.0
```

### 7.4. Examples

To know more about building a Wflow-Sediment model from scratch, check the following examples:

- [Building a Wflow Sediment model from command line](https://deltares.github.io/hydromt_wflow/stable/_examples/build_sediment.html#example-build-sediment)

## 8. Updating a model

To add or change one or more components of an existing Wflow Sediment model the `update` method can be used.

**Steps in brief:**

1. You have an **existing model** schematization. This model does not have to be complete.
2. Prepare or use a pre-defined **data catalog** with all the required data sources, see [working with data](https://deltares.github.io/hydromt/latest/user_guide/data_catalog/data_overview.html).
3. Prepare a **model configuration** with the methods that you want to use to add or change components of your model: see [model configuration](https://deltares.github.io/hydromt/latest/user_guide/models/model_workflow.html).
4. **Update** your model using the CLI or Python interface.

```
activate hydromt-wflow
hydromt update wflow_sediment path/to/model_to_update -o path/to/updated_model -i wflow_sediment_update.yml -d data_sources.yml -vvv
```

Note

By default, the updated model will overwrite your existing one. To save the updated model in a different folder, use the -o path/to/updated\_model option of the CLI.

Tip

By default all model data is written at the end of the update method. If your update however only affects a certain model data (e.g. staticmaps or forcing) you can add a write\_\* method (e.g. write\_grid, write\_forcing) to the.yml file and only these data will be written.

Note that the model config is often changed as part of the a model method and write\_config should thus be added to the.yml file to keep the model data and config consistent.

## 9. Clipping a model

This plugin allows to clip the following parts of an existing model for a smaller region from command line:

- staticmaps
- forcing
- states
- geoms
- config (update reservoir settings)

To clip a smaller model from an existing one use the `update` CLI command with the **clip** method:

```
activate hydromt_wflow
hydromt update wflow_sediment -o path/to/model_to_clip path/to/clipped_model -i clip_config.yml -v
```

Here is an example of the clip config:

```yaml
steps:
  - clip:
      region: {"basin": [x, y]} # region to clip the model too
      inverse_clip: false # whether to clip outside or inside the region
      clip_states: true # whether to clip states
      clip_forcing: true # whether to clip forcing
```

As for building, the recommended [region options](https://deltares.github.io/hydromt/latest/user_guide/models/model_region.html) for a proper implementation of the clipped model are:

- basin
- subbasin

See the following model API:

- `clip()`

## 10. Pre and postprocessing and visualization

The Hydromt-Wflow plugin provides several possibilities to postprocess and visualize the model data and model results:

- [Prepare flow directions](https://deltares.github.io/hydromt_wflow/stable/_examples/prepare_ldd.html#example-prepare-ldd) using the [flw methods of HydroMT](https://deltares.github.io/hydromt/latest/api/gis.html#flow-direction-methods)
- [Convert static maps to mapstack](https://deltares.github.io/hydromt_wflow/stable/_examples/convert_staticmaps_to_mapstack.html#example-convert-staticmaps-to-mapstack) for further processing and analysis
- Plot [static maps](https://deltares.github.io/hydromt_wflow/stable/_examples/plot_wflow_staticmaps.html#example-plot-wflow-staticmaps), [forcing](https://deltares.github.io/hydromt_wflow/stable/_examples/plot_wflow_forcing.html#example-plot-wflow-forcing) and [results](https://deltares.github.io/hydromt_wflow/stable/_examples/plot_wflow_results.html#example-plot-wflow-results) by means of additional python packages
- Use the [statistical methods of HydroMT](https://deltares.github.io/hydromt/latest/api/stats.html#statistics-and-performance-metrics) to statistically analyze the model results
- Upgrade your old Wflow model to the Wflow.jl version 1 format using the [upgrade](https://deltares.github.io/hydromt_wflow/stable/_examples/upgrade_to_wflow_v1.html#example-upgrade-to-wflow-v1) example.

Here are a few examples of how these methods can be used:

- [Preparing flow directions from DEM](https://deltares.github.io/hydromt_wflow/stable/_examples/prepare_ldd.html#example-prepare-ldd)
- [Convert static maps to mapstack](https://deltares.github.io/hydromt_wflow/stable/_examples/convert_staticmaps_to_mapstack.html#example-convert-staticmaps-to-mapstack)
- [Plot static maps](https://deltares.github.io/hydromt_wflow/stable/_examples/plot_wflow_staticmaps.html#example-plot-wflow-staticmaps)
- [Plot forcing](https://deltares.github.io/hydromt_wflow/stable/_examples/plot_wflow_forcing.html#example-plot-wflow-forcing)
- [Plot results](https://deltares.github.io/hydromt_wflow/stable/_examples/plot_wflow_results.html#example-plot-wflow-results)
- [Upgrade to Wflow.jl version 1](https://deltares.github.io/hydromt_wflow/stable/_examples/upgrade_to_wflow_v1.html#example-upgrade-to-wflow-v1)

## 11. Setup\_lulcmaps and related methods

### 11.1. Description

To prepare land use / land cover related maps for Wflow, HydroMT provides several methods. The basis of these methods is that they use lookup tables with parameters values for each land use / land cover class and then map these values to the model grid using land use / land cover maps.

The parameters are mapped at the original LULC map resolution before being resampled to the model grid resolution (using either averaging or majority mapping depending on the parameter type). This ensures that most of the details of the original resolution of the LULC map are preserved in the final model maps.

![setup_lulcmaps](https://deltares.github.io/hydromt_wflow/stable/_images/setup_lulcmaps.png)

The following methods are available:

- [`setup_lulcmaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_lulcmaps.html#hydromt_wflow.WflowSbmModel.setup_lulcmaps "hydromt_wflow.WflowSbmModel.setup_lulcmaps") and [`setup_lulcmaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.setup_lulcmaps.html#hydromt_wflow.WflowSedimentModel.setup_lulcmaps "hydromt_wflow.WflowSedimentModel.setup_lulcmaps"): Main method to setup LULC maps using lookup tables.
- [`setup_lulcmaps_from_vector()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_lulcmaps_from_vector.html#hydromt_wflow.WflowSbmModel.setup_lulcmaps_from_vector "hydromt_wflow.WflowSbmModel.setup_lulcmaps_from_vector") and [`setup_lulcmaps_from_vector()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.setup_lulcmaps_from_vector.html#hydromt_wflow.WflowSedimentModel.setup_lulcmaps_from_vector "hydromt_wflow.WflowSedimentModel.setup_lulcmaps_from_vector"): Similar to the above but starts with rasterizing the LULC vector data to a user-defined resolution.
- [`setup_lulcmaps_with_paddy()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_lulcmaps_with_paddy.html#hydromt_wflow.WflowSbmModel.setup_lulcmaps_with_paddy "hydromt_wflow.WflowSbmModel.setup_lulcmaps_with_paddy"): Specific method if paddies are present in your catchment. The LULC map can directly contain a paddy class or an additional paddy map can be provided and will be merged into the landuse map before deriving parameters. Additional parameters related to paddy management (minimum/optimal/maximum water levels) are also added based on user defined values. Finally, to allow for water to pool on the surface (for paddy/rice fields), the layers in the model can be updated to new depths, such that we can allow a thin layer with limited vertical conductivity. These updated layers means that the `soil_brooks_corey_c` parameter needs to be calculated again. Next, the soil\_ksat\_vertical\_factor layer corrects the vertical conductivity (by multiplying) such that the bottom of the layer corresponds to a target\_conductivity for that layer.

### 11.2. Parameter lookup tables

The parameter values for each land use / land cover class need to be defined in lookup tables. HydroMT provides some default lookup tables for the following LULC classification systems:

- **corine**: [CORINE Land Cover (CLC)](https://land.copernicus.eu/en/products/corine-land-cover)
- **glcnmo**: [GLCNMO](https://globalmaps.github.io/glcnmo.html)
- **globcover**: [GlobCover](https://due.esrin.esa.int/page_globcover.php)
- **esa\_worldcover**: [ESA WorldCover](https://esa-worldcover.org/en)
- **vito**: [Copernicus Global Dynamic Land Cover](https://land.copernicus.eu/en/products/global-dynamic-land-cover)
- **paddy**: specific lookup table for paddy fields (if using the `setup_lulcmaps_with_paddy` method)

You can find these tables in the [HydroMT-Wflow repository](https://github.com/Deltares/hydromt_wflow/tree/main/hydromt_wflow/data/lulc) or create your own based on these examples or literature values to better reflect the specific vegetation and soil characteristics of your study area.

The lookup tables are simple CSV files with the first column containing the land use / land cover class identifiers (matching those in the LULC map), the second column containing the `description` of the class and the other columns containing the parameter values for each class. The **last line of the table should contain the nodata value** for the LULC map (e.g. -9999) and the corresponding parameter values for nodata areas.

The columns names should match the HydroMT names of each Wflow parameter. These are:

- **landuse**: landuse class ID
- **vegetation\_kext**: Extinction coefficient in the canopy gap fraction equation \[-\]
- **land\_manning\_n**: Manning Roughness \[m-1/3 s\]
- **soil\_compacted\_fraction**: The fraction of compacted or urban area per grid cell \[-\]
- **vegetation\_root\_depth**: Length of vegetation roots \[mm\]
- **vegetation\_leaf\_storage**: Specific leaf storage \[mm\]
- **vegetation\_wood\_storage**: Fraction of wood in the vegetation/plant \[-\]
- **land\_water\_fraction**: The fraction of open water per grid cell \[-\]
- **vegetation\_crop\_factor**: Crop coefficient \[-\]
- **vegetation\_feddes\_alpha\_h1**: Root water uptake reduction at soil water pressure head h1 (0 or 1) \[-\]
- **vegetation\_feddes\_h1**: Soil water pressure head h1 at which root water uptake is reduced (Feddes) \[cm\]
- **vegetation\_feddes\_h2**: Soil water pressure head h2 at which root water uptake is reduced (Feddes) \[cm\]
- **vegetation\_feddes\_h3\_high**: Soil water pressure head h3 (high) at which root water uptake is reduced (Feddes) \[cm\]
- **vegetation\_feddes\_h3\_low**: Soil water pressure head h3 (low) at which root water uptake is reduced (Feddes) \[cm\]
- **vegetation\_feddes\_h4**: Soil water pressure head h4 at which root water uptake is reduced (Feddes) \[cm\]
- **erosion\_usle\_c** (sediment): USLE cover management factor \[-\]

Example lookup table (for ESA WorldCover):

| esa | description | landuse | vegetation\_kext | land\_manning\_n | soil\_compacted\_fraction | vegetation\_root\_depth | vegetation\_leaf\_storage | vegetation\_wood\_storage | land\_water\_fraction | vegetation\_crop\_factor | vegetation\_feddes\_alpha\_h1 | vegetation\_feddes\_h1 | vegetation\_feddes\_h2 | vegetation\_feddes\_h3\_high | vegetation\_feddes\_h3\_low | vegetation\_feddes\_h4 | erosion\_usle\_c |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | Tree cover | 10 | 0.8 | 0.5 | 0 | 406 | 0.23 | 0.09 | 0 | 1.1 | 1 | 0 | \-100 | \-400 | \-1000 | \-16000 | 0.0012 |
| 20 | Shrubland | 20 | 0.7 | 0.5 | 0 | 410 | 0.1 | 0.05 | 0 | 1.05 | 1 | 0 | \-100 | \-400 | \-1000 | \-16000 | 0.06 |
| 30 | Grassland | 30 | 0.6 | 0.2 | 0 | 106.8 | 0.1 | 0.01 | 0 | 1 | 1 | 0 | \-100 | \-400 | \-1000 | \-16000 | 0.04 |
| 40 | Cropland | 40 | 0.6 | 0.15 | 0 | 390.4 | 0.077 | 0.005 | 0 | 1.1 | 0 | 0 | \-100 | \-400 | \-1000 | \-16000 | 0.3 |
| 50 | Built-up | 50 | 0.6 | 0.015 | 0.9 | 257.4 | 0.1 | 0.03 | 0 | 1 | 1 | 0 | \-100 | \-400 | \-1000 | \-16000 | 0.001 |
| 60 | Bare / sparse vegetation | 60 | 0.6 | 0.015 | 0 | 10.7 | 0.1 | 0.03 | 0 | \-999 | 1 | 0 | \-100 | \-400 | \-1000 | \-16000 | 0.35 |
| 70 | Snow and Ice | 70 | 0 | 0.01 | 0 | 0 | 0 | 0 | 0 | \-999 | 1 | 0 | \-100 | \-400 | \-1000 | \-16000 | 0 |
| 80 | Permanent water bodies | 80 | 0 | 0.01 | 0 | 0 | 0 | 0 | 1 | \-999 | 1 | 0 | \-100 | \-400 | \-1000 | \-16000 | 0 |
| 90 | Herbaceous wetland | 90 | 0.6 | 0.125 | 0 | 106.8 | 0.1 | 0.01 | 0 | 1.2 | 1 | 0 | \-100 | \-400 | \-1000 | \-16000 | 0.001 |
| 95 | Mangroves | 95 | 0.8 | 0.5 | 0 | 369 | 0.23 | 0.09 | 0.5 | 1.05 | 1 | 0 | \-100 | \-400 | \-1000 | \-16000 | 0.008 |
| 100 | Moss and lichen | 100 | 0.6 | 0.085 | 0 | 136.9 | 0.09 | 0 | 0 | 1.05 | 1 | 0 | \-100 | \-400 | \-1000 | \-16000 | 0.001 |
| 0 | No data | 0 | \-999 | \-999 | \-999 | \-999 | \-999 | \-999 | \-999 | \-999 | \-999 | \-999 | \-999 | \-999 | \-999 | \-999 | \-999 |

### 11.3. Example usage

Here is an example of how to use the `setup_lulcmaps` and related methods when updating a model:

The definition of the method and the arguments is done in a workflow file (YAML format). The workflow file can then be used to build or update a model from the command line interface. For example, using the pre-defined `artifact_data` catalog:

```
$ hydromt update wflow_sbm "./path/to/model_to_update" -o "./path/to/model_with_landuse" -d "artifact_data" -i "./path/to/update_landuse.yaml" -v
```

A minimal example of how to use the `setup_lulcmaps` method in a workflow file:

```yaml
steps:
  - setup_lulcmaps:
      lulc_fn: globcover
      lulc_mapping_fn: globcover_mapping_default
```

Another example using the `setup_lulcmaps_from_vector`. Here we will also save the rasterized version of the landuse map to a file and only prepare a couple of parameters:

```yaml
steps:
  - setup_lulcmaps_from_vector:
      lulc_vector_fn: local_landuse_vector.shp
      lulc_mapping_fn: local_landuse_mapping.csv
      lulc_res: 50
      save_raster_lulc: true
      lulc_vars:
        - land_manning_n
        - vegetation_crop_factor
```

Final example using the `setup_lulcmaps_with_paddy` method to include paddy fields. Here another raster file is provided with the paddy field locations (GLCNMO where paddy class is number 12). As class numbers for irrigated and rainfed cropland are 11 and 14 in globcover we can keep twelve as the paddy class value in the merged landuse map.

Adding paddies also requires to add extra parameters related to paddy management and update some of the soil parameters.

```yaml
steps:
  - setup_lulcmaps_with_paddy:
      lulc_fn: globcover
      lulc_mapping_fn: globcover_mapping_default
      paddy_fn: glcnmo
      paddy_mapping_fn: paddy_mapping_default
      paddy_class: 12
      output_paddy_class: 12
      paddy_waterlevels:
        demand_paddy_h_min: 20
        demand_paddy_h_opt: 50
        demand_paddy_h_max: 80
      wflow_thicknesslayers: [50, 100, 50, 200, 800]
      target_conductivity: [null, null, 5, null, null]
```

For python, you need to first instantiate a Wflow model and then call the setup methods directly:

```python
from hydromt_wflow import WflowSbmModel

model = WflowSbmModel(
  root="path/to/model_to_update",
  mode="r+",
  data_libs=["artifact_data"]
)
```

A minimal example of how to use the `setup_lulcmaps` method:

```python
model.setup_lulcmaps(
  lulc_fn="globcover",
  lulc_mapping_fn="globcover_mapping_default"
)
```

Another example using the `setup_lulcmaps_from_vector`. Here we will also save the rasterized version of the landuse map to a file and only prepare a couple of parameters:

```python
model.setup_lulcmaps_from_vector(
  lulc_vector_fn="local_landuse_vector.shp",
  lulc_mapping_fn="local_landuse_mapping.csv",
  lulc_res=50,
  save_raster_lulc=True,
  lulc_vars=[
    "land_manning_n",
    "vegetation_crop_factor"
  ]
)
```

Final example using the `setup_lulcmaps_with_paddy` method to include paddy fields. Here another raster file is provided with the paddy field locations (GLCNMO where paddy class is number 12). As class numbers for irrigated and rainfed cropland are 11 and 14 in globcover we can keep twelve as the paddy class value in the merged landuse map.

Adding paddies also requires to add extra parameters related to paddy management and update some of the soil parameters.

```python
model.setup_lulcmaps_with_paddy(
  lulc_fn="globcover",
  lulc_mapping_fn="globcover_mapping_default",
  paddy_fn="glcnmo",
  paddy_mapping_fn="paddy_mapping_default",
  paddy_class=12,
  output_paddy_class=12,
  paddy_waterlevels={
    "demand_paddy_h_min": 20,
    "demand_paddy_h_opt": 50,
    "demand_paddy_h_max": 80
  },
  wflow_thicknesslayers=[50, 100, 50, 200, 800],
  target_conductivity=[None, None, 5, None, None]
)
```

More examples can be found in the following notebooks:

- [Update land use](https://deltares.github.io/hydromt_wflow/stable/_examples/update_model_landuse.html#example-update-model-landuse)
- [Add water demands and allocations (with paddy landuse)](https://deltares.github.io/hydromt_wflow/stable/_examples/update_model_water_demand.html#example-update-model-water-demand)

### 11.4. Parameter estimation

The estimates in the above table are based on literature reviews done by [Imhoff et al., 2020](https://doi.org/10.1029/2019WR026807)

Here are some references to help you estimate parameter values for your own lookup tables.

#### 11.4.1. Interception parameters

Parameters related to vegetation interception and storage of rainfall on leaves and branches.

| Parameter | Description | Range | Reference |
| --- | --- | --- | --- |
| kext | Extinction coefficient in the canopy gap fraction equation \[-\] | 0.2-0.9 | [Van Dijk and Bruijnzeel (2001)](https://doi.org/10.1016/S0022-1694\(01\)00392-4) [Van Heemst (1988)](https://edepot.wur.nl/218353) |
| leaf\_storage | Specific leaf storage \[mm\] | 0.02-0.2 | [Zhong et al. (2022)](https://doi.org/10.5194/hess-26-5647-2022) |
| wood\_storage | Fraction of wood in the vegetation/plant \[-\] | 0.0-0.5 | [Zhong et al. (2022)](https://doi.org/10.5194/hess-26-5647-2022) |

##### 11.4.1.1. kext

Extract from Van Dijk and Bruijnzeel (2001):

> The value of kext for a particular radiation wavelength depends on leaf distribution and inclination angle and for PAR usually ranges between 0.6 and 0.8 in forests (Ross, 1975). For a number of agricultural crops, van Heemst (1988) reported kext values between 0.2 and 0.8 with values of 0.5-0.7 being the most common.

Values for different crops from van Heemst (1988):

| Crop | kext |
| --- | --- |
| Wheat | 0.42 - 0.54 |
| Barley | 0.44 |
| Rice | 0.29 - 0.43 |
| Millet | 0.5 - 0.6 |
| Sorghum | 0.4 - 0.7 |
| Maize | 0.6 - 0.64 |
| Soybean | 0.787 - 0.804 |
| Peanut | 0.6 |
| Oilseed rape | 0.54 |
| Sunflower | 0.8 - 0.9 |
| Cassava | 0.7 - 0.88 |
| Sweet Potato | 0.45 |
| Potato | 0.48 |
| Sugar beet | 0.65 |
| Sugar cane | 0.48 |
| Cotton | 0.62 |

##### 11.4.1.2. Leaf and wood storage

Previous values were derived from [Pitman (1989)](https://doi.org/10.5194/hess-15-3355-2011) and [Liu (1998)](https://doi.org/10.1016/S0022-1694\(98\)00115-2). Starting from version 1, the default lookup tables use updated values based on a literature review by [Zhong et al. (2022)](https://doi.org/10.5194/hess-26-5647-2022) (supplement values with more details are available).

Note that for land use types with mixed (e.g urban) or sparse vegetation, the actual values will be scaled with LAI.

| Vegetation / Crop type | Leaf storage \[mm\] | Wood storage \[-\] |
| --- | --- | --- |
| Needleleaf forest | 0.29 | 0.09 |
| Evergreen broadleaf forest | 0.20 | 0.09 |
| Deciduous broadleaf forest | 0.18 | 0.09 |
| Mixed forest | 0.20 | 0.09 |
| All forest | 0.23 | 0.09 |
| Short vegetation (crops, grass, shrub) | 0.10 | 0.03 (0.01 - 0.05) |
| Maize | 0.077 | 0.005 |
| Rice | 0.042 | 0.005 |

#### 11.4.2. Evapotranspiration parameters

Parameters related to vegetation evaporation and transpiration.

| Parameter | Description | Range | Reference |
| --- | --- | --- | --- |
| crop\_factor | Crop coefficient \[-\] | 0.3 - 1.25 | [Allen et al. (1998)](https://www.fao.org/4/x0490e/x0490e0c.htm) |
| root\_depth | Length of vegetation roots \[mm\] | 100 - 5000 | [Fan et al. (2016)](https://www.mdpi.com/2077-0472/14/4/532) [Schenk and Jackson (2002)](https://doi.org/10.1890/0012-9615\(2002\)072[0311:TGBOR]2.0.CO;2) |
| feddes\_alpha\_h1 | Root water uptake reduction at pressure head h1 \[-\] | 0 (crop) - 1 (other) | [van Dam et al. (1997)](https://edepot.wur.nl/222782) [Singh et al. (2003)](https://www.academia.edu/download/102602419/19325.pdf) |
| feddes\_h1 | Critical pressure head h1 - anorexic condition \[cm\] | 100 (paddy) - 0 (other) | [van Dam et al. (1997)](https://edepot.wur.nl/222782) [Singh et al. (2003)](https://www.academia.edu/download/102602419/19325.pdf) |
| feddes\_h2 | Critical pressure head h2 - field capacity \[cm\] | 55 (paddy) - -100 (other) | [van Dam et al. (1997)](https://edepot.wur.nl/222782) [Singh et al. (2003)](https://www.academia.edu/download/102602419/19325.pdf) |
| feddes\_h3\_high | Critical pressure head h3 (high) \[cm\] | \-160 (paddy) - -400 (other) | [van Dam et al. (1997)](https://edepot.wur.nl/222782) [Singh et al. (2003)](https://www.academia.edu/download/102602419/19325.pdf) |
| feddes\_h3\_low | Critical pressure head h3 (low) \[cm\] | \-250 (paddy) - -1000 (other) | [van Dam et al. (1997)](https://edepot.wur.nl/222782) [Singh et al. (2003)](https://www.academia.edu/download/102602419/19325.pdf) |
| feddes\_h4 | Critical pressure head h4 - wilting point \[cm\] | \-15000 (paddy) - -16000 (other) | [van Dam et al. (1997)](https://edepot.wur.nl/222782) [Singh et al. (2003)](https://www.academia.edu/download/102602419/19325.pdf) |

##### 11.4.2.1. Crop factor

The factor or FAO-56 crop coefficient the crop factor is used to scale reference evapotranspiration (\\(ET\_0\\)) to crop evapotranspiration (ETc) as follows: \\(ET\_c = (K\_{cb} + K\_e) \* ET\_0\\), where \\(K\_{cb}\\) is the basal crop coefficient and \\(K\_e\\) is the soil evaporation coefficient. As Wflow takes care of the soil evaporation component, the crop coefficient needed is then \\(K\_{cb full}\\) which is the basal crop coefficient during the mid-season (at peak plant size or height) for vegetation having full ground cover or LAI > 3. Within Wflow, \\(K\_{cb full}\\) will be scaled further based on the actual vegetation cover fraction (using LAI) to get the actual crop coefficient used for \\(ET\_c\\) calculation.

In sub-humid and calm wind conditions, \\(K\_{cb full}\\) is equal to the FAO-56 mid-season crop coefficient \\(K\_{cb mid}\\). Detailed values of \\(K\_{cb mid}\\) can be found for different crop types in the [FAO guidelines](https://www.fao.org/4/x0490e/x0490e0c.htm). As most LULC maps do not distinguish between crop types, an average value representing the most common crops in your study area should be used. In the default lookup tables, 1.15 is used for cropland areas (based on an average value for cereals and oil crops), and 1.2 for paddy/rice fields.

Detailed values of kc can be found for different crop types in the [FAO guidelines](https://www.fao.org/4/x0490e/x0490e0b.htm). As most LULC maps do not distinguish between crop types, an average value representing the most common crops in your study area should be used. In the default lookup tables, 1.10 is used for cropland areas (based on an average value for cereals and oil crops), and 1.15 for paddy/rice fields.

For natural vegetation, \\(K\_{cb full}\\) can be estimated from the vegetation height and climate conditions. For example, the [FAO guidelines](https://www.fao.org/4/x0490e/x0490e0f.htm) provide the following equations to estimate \\(K\_{cb full}\\) for natural vegetation:

..math:

```
K_{cb full} = K_{cb,h} + [0.04(u_2 - 2) - 0.004(RH_{min} - 45)] * (h/3)^0.3
K_{cb,h} = 1.0 + 0.1 * h \quad \text{for } h \leq 2 \text{ m}
```

where \\(u\_2\\) is the wind speed at 2 m height \[m/s\], \\(RH\_{min}\\) is the minimum relative humidity \[%\] and \\(h\\) is the mean maximum plant height \[m\].

Finally, for land use type with no vegetation (e.g. bare soil, waterbodies), the nodata value should be used (e.g. -9999) to avoid underestimation of \\(K\_{cb full}\\) during reprojection of the land use parameter at original resolution to the model grid resolution.

For mixed land use types (e.g. urban areas), the value should be scaled based on the vegetation type in the area (e.g. sparse vegetation should use the value for grass or shrubland). In the default lookup tables, such areas (urban, sparse vegetation) use a value of 1.0 for grass.

##### 11.4.2.2. Root depth

Values for different crops from Fan et al. (2016) and different other vegetation from Schenk and Jackson (2002):

| Vegetation / Crop type | Root depth D50 \[mm\] | Root depth D95 \[mm\] |
| --- | --- | --- |
| Tundra | 90 | 290 |
| Boreal forest | 120 | 580 |
| Cool temperate forest | 210 | 1040 |
| Warm temperate forest | 230 | 1210 |
| Meadows | 50 | 400 |
| Prairie | 70 | 910 |
| Semi arid steppe | 160 | 1200 |
| Temperate savanna | 230 | 1400 |
| Mediterranean woodland and shrub | 190 | 1710 |
| Semi-desert shrubland | 280 | 1310 |
| Desert | 270 | 1120 |
| Dry tropical savannas | 280 | 1440 |
| Humid tropical savannas | 140 | 940 |
| Tropical semi-deciduous and deciduous forest | 160 | 950 |
| Tropical evergreen forest | 150 | 910 |
| Wheat | 168 | 1038 |
| Maize | 144 | 889 |
| Oat | 112 | 777 |
| Barley | 115 | 996 |
| Cereals | 141 | 929 |
| Soybean | 109 | 1380 |
| Oilseed crops | 94 | 1063 |
| All crops | 146 | 1027 |

##### 11.4.2.3. Feddes root water uptake

Critical pressure heads for rice are taken after Singh et al. (2003). For other vegetation, the default values from Wflow.jl are used. These are now vegetation independent and are taken as the default complete saturation (h1=0 cm), field capacity (h2=-100 cm) and wilting point (h4=-16000 cm). The h3 values are set to -400 cm (high) and -1000 cm (low) but these are largely dependent on the type of vegetation.

Examples can be found in annexes C and D of Van Dam et al. (1997). Here are examples for the most common crops \[cm\]:

| Crop | h1 | h2 | h3\_high | h3\_low | h4 |
| --- | --- | --- | --- | --- | --- |
| Potatoes | \-10 | \-25 | \-320 | \-600 | \-16000 |
| Sugar beet | \-10 | \-25 | \-320 | \-600 | \-16000 |
| Wheat | 0 | \-1 | \-500 | \-900 | \-16000 |
| Pasture | \-10 | \-25 | \-200 | \-800 | \-8000 |
| Corn | \-15 | \-30 | \-325 | \-600 | \-8000 |

#### 11.4.3. Manning Roughness

Manning’s N values are used to represent roughness of the land surface for overland flow. Estimations per landuse class can be found in literature such as:

| Parameter | Description | Range | Reference |
| --- | --- | --- | --- |
| manning\_n | Manning Roughness \[m-1/3 s\] | 0.008-0.96 | [Engman (1986)](https://doi.org/10.1061/\(ASCE\)0733-9437\(1986\)112:1\(39\)) [Kilgore (1997)](http://hdl.handle.net/10919/35777) [Cronshey (1986)](https://www.nrc.gov/docs/ML1421/ML14219A437.pdf) |

Example of values from different sources:

| Landuse | Cronshey | Kilgore | Engman |
| --- | --- | --- | --- |
| Smooth surfaces (concrete, gravel, bare) | 0.011 | 0.015 (residential/commercial) / 0.020 (gravel road) | 0.01 (smooth bare soil or bare sand) / 0.011 (concrete) - 0.020 (gravel) |
| Fallow (no residue) | 0.05 | 0.05 | 0.05 |
| Cropland | 0.06 - 0.17 (depending on residue cover) | 0.032 (wheat) / 0.08 (corn) - 0.2 (depending on tillage) | 0.1 - 0.4 (small grain) / 0.07 - 0.2 (row crops) |
| Grassland | 0.15 (short) - 0.24 (dense) | 0.046 (grass) / 0.1 (pasture) | 0.15 (short) - 0.24 (dense) |
| Forest | 0.4 - 0.8 (depending on underbrush) | 0.6 |  |
| Range (natural) | 0.13 |  | 0.13 |
| Wetland |  | 0.125 |  |
| Waterway, pond |  | 0.08 |  |

#### 11.4.4. Land cover parameters

The land cover parameters represent fractions of different land cover types within each grid cell. For these parameters, it may matter to take into account the resolution of the original LULC map when estimating the values. E.g. a coarse resolution may for example represent a cell that is majority urban but still contains a significant fraction of vegetation or water.

| Parameter | Description | Estimate |
| --- | --- | --- |
| soil\_compacted\_fraction | Fraction of compacted or paved area per grid cell \[-\] | \> 0 if urban or compacted / paved surfaces are present |
| land\_water\_fraction | Fraction of open water per grid cell \[-\] | \> 0 if water (water bodies, ponds, waterways etc.) is present |

#### 11.4.5. Soil erosion

For soil erosion, the soil cover-management factor USLE C can be estimated for different land use / vegetation type.

| Parameter | Description | Range | Reference |
| --- | --- | --- | --- |
| erosion\_usle\_c | USLE cover management factor \[-\] | 0.001 - 1.0 | [Panagos et al. (2015)](https://doi.org/10.1016/j.landusepol.2015.05.021) [Bosco et al. (2015)](https://doi.org/10.5194/nhess-15-225-2015) [Gericke et al. (2015)](https://doi.org/10.1080/15715124.2014.1003302) |

Examples of USLE C values for different land use types different sources:

| Land use | Panagos | Bosco | Gericke |
| --- | --- | --- | --- |
| Wheat | 0.20 |  |  |
| Maize | 0.38 |  |  |
| Rice | 0.15 | 0.15 | 0.05 |
| Potatoes or sugar beet | 0.34 |  |  |
| Oilseeds | 0.28 |  |  |
| All crops | 0.233 (0.2 - 0.5) | 0.2 (irrigated) / 0.335 (rainfed) | 0.18 - 0.24 (irrigated) / 0.3 - 0.4 (rainfed) |
| Vineyards | 0.3527 (0.15 - 0.45) | 0.45 | 0.5 |
| Fruit trees and berries | 0.2188 (0.1 - 0.3) | 0.35 | 0.4 |
| Olive groves | 0.2273 (0.1 - 0.3) | 0.35 | 0.4 |
| Agro-forestry areas | 0.0881 (0.03 - 0.13) | 0.2 | 0.23 - 0.3 |
| Broad-leaved forest | 0.0013 (0.0001 - 0.003) | 0.0025 | 0.005 - 0.008 |
| Coniferous forest | 0.0011 (0.0001 - 0.003) | 0.0015 | 0.005 - 0.008 |
| Mixed forest | 0.0011 (0.0001 - 0.003) | 0.002 | 0.005 - 0.008 |
| Pastures | 0.0903 (0.05 - 0.15) | 0.01 | 0.01 - 0.005 |
| Natural grasslands | 0.0435 (0.01 - 0.08) | 0.005 | 0.01 - 0.05 |
| Moors and heathland | 0.0420 (0.01 - 0.1) | 0.05 | 0.01 - 0.05 |
| Shrubland | 0.0623 (0.01 - 0.1) | 0.04 | 0.01 - 0.05 |
| Bare rocks | 0 |  | 0 |
| Sparse vegetation | 0.2652 (0.1 - 0.45) | 0.3 | 0.35 |
| Burnt areas | 0.3427 (0.1 - 0.55) | 0.3 | 0.35 |
| Glaciers and perpetual snow | 0 | 0.001 | 0 |

### 11.5. References

- Allen RG, Pereira LS, Raes D, Smith M (1998) Crop evapotranspiration guidelines for computing crop water requirements. FAO Irrig Drain Pap 56. FAO, Rome, p 300
- Bosco, C., de Rigo, D., Dewitte, O., Poesen, J., and Panagos, P. (2015). Modelling soil erosion at European scale: towards harmonization and reproducibility, Nat. Hazards Earth Syst. Sci., 15, 225–245, [https://doi.org/10.5194/nhess-15-225-2015](https://doi.org/10.5194/nhess-15-225-2015)
- Corbari, C., Ravazzani, G., Galvagno, M., Cremonese, E., & Mancini, M. (2017). Assessing crop coefficients for natural vegetated areas using satellite data and eddy covariance stations. Sensors, 17(11), 2664.
- Cronshey, R. (1986). Urban hydrology for small watersheds (No. 55). US Department of Agriculture, Soil Conservation Service, Engineering Division.
- van Dam, J.C., Huygen, J., Wesseling, J.G., Feddes, R.A., Kabat, P., van Walsum, P.E.V., Groenendijk, P., and van Diepen, C.A., 1997. Theory of SWAP version 2.0: Simulation of water flow, solute transport and plant growth in the soil-water-atmosphere-plant environment. Wageningen Agricultural University, The Netherlands, Report 71.
- van Dijk, A. I. J. M., & Bruijnzeel, L. A. (2001). Modelling rainfall interception by vegetation of variable density using an adapted analytical model. Part 2. Model validation for a tropical upland mixed cropping system. Journal of Hydrology, 247(3-4), 239–262.
- Engman, E. (1986). Roughness coefficients for routing surface runoff. Journal of Irrigation and Drainage Engineering, 112(1), 39-53. [https://doi.org/10.1061/(ASCE)0733-9437(1986)112:1(39](https://doi.org/10.1061/\(ASCE\)0733-9437\(1986\)112:1\(39))
- Fan, J., McConkey, B., Wang, H., & Janzen, H. (2016). Root distribution by depth for temperate agricultural crops. Field Crops Research, 189, 68–74. [https://doi.org/10.1016/j.fcr.2016.02.013](https://doi.org/10.1016/j.fcr.2016.02.013)
- Feddes, R.A., Kowalik, P.J. and Zaradny, H., 1978, Simulation of field water use and crop yield, Pudoc, Wageningen, Simulation Monographs.
- Gericke, A. (2015). Soil loss estimation and empirical relationships for sediment delivery ratios of European river catchments. International Journal of River Basin Management, 13(2), 179–202. [https://doi.org/10.1080/15715124.2014.1003302](https://doi.org/10.1080/15715124.2014.1003302)
- van Heemst, H.D.J. (1988). Plant data values required for simple crop growth simulation models, review and bibliography. Simulation report CABO-TT No 17. Wageningen.
- Imhoff, R.O, van Verseveld, W.J., van Osnabrugge, B., Weerts, A.H., 2020. Scaling Point-Scale (Pedo)transfer Functions to Seamless Large-Domain Parameter Estimates for High-Resolution Distributed Hydrologic Modeling: An Example for the Rhine River. Water Resources Research, 56, e2019WR026807. [https://doi.org/10.1029/2019WR026807](https://doi.org/10.1029/2019WR026807).
- Kilgore, J. L. (1997). Development and evaluation of a GIS-based spatially distributed unit hydrograph model (MSc thesis). Retrieved from [http://hdl.handle.net/10919/35777](http://hdl.handle.net/10919/35777)
- Liu, S. (1998). Estimation of rainfall storage capacity in the canopies of cypress wet lands and slash pine uplands in North-Central Florida. Journal of Hydrology, 207(1-2), 32–41.[https://doi.org/10.1016/S0022-1694(98)00115-2](https://doi.org/10.1016/S0022-1694\(98\)00115-2)
- Panagos, P., Borrelli, P., Meusburger, K., Alewell, C., Lugato, E., & Montanarella, L. (2015). Estimating the soil erosion cover-management factor at the European scale. Land Use Policy, 48, 38–50. [https://doi.org/10.1016/j.landusepol.2015.05.021](https://doi.org/10.1016/j.landusepol.2015.05.021)
- Pereira, L.S., Paredes, P. & Espírito-Santo, D. (2024a). Crop coefficients of natural wetlands and riparian vegetation to compute ecosystem evapotranspiration and the water balance. Irrig Sci 42, 1171–1197. [https://doi.org/10.1007/s00271-024-00923-9](https://doi.org/10.1007/s00271-024-00923-9)
- Pereira, L.S., Paredes, P., Espírito-Santo, D. et al. (2024b). Actual and standard crop coefficients for semi-natural and planted grasslands and grasses: a review aimed at supporting water management to improve production and ecosystem services. Irrig Sci 42, 1139–1170. [https://doi.org/10.1007/s00271-023-00867-6](https://doi.org/10.1007/s00271-023-00867-6)
- Pereira, L.S., Paredes, P., Oliveira, C.M. et al. (2024c). Single and basal crop coefficients for estimation of water use of tree and vine woody crops with consideration of fraction of ground cover, height, and training system for Mediterranean and warm temperate fruit and leaf crops. Irrig Sci 42, 1019–1058. [https://doi.org/10.1007/s00271-023-00901-7](https://doi.org/10.1007/s00271-023-00901-7)
- Pitman, J. (1989). Rainfall interception by bracken in open habitats—Relations between leaf area, canopy storage and drainage rate. Journal of Hydrology, 105(3-4), 317–334.[https://doi.org/10.1016/0022-1694(89)90111-X](https://doi.org/10.1016/0022-1694\(89\)90111-X)
- Schenk, H. J., & Jackson, R. B. (2002). The global biogeography of roots. Ecological Monographs, 72(3), 311–328. [https://doi.org/10.1890/0012-9615(2002)072\[0311:TGBOR\]2.0.CO;2](https://doi.org/10.1890/0012-9615\(2002\)072[0311:TGBOR]2.0.CO;2)
- Singh, R., Van Dam, J. C., & Jhorar, R. K. (2003). Water and salt balances at farmer fields. Water productivity of irrigated crops in Sirsa district, India. Integration of remote sensing, crop and soil models and geographical information systems.
- Zhong, F., Jiang, S., van Dijk, A. I. J. M., Ren, L., Schellekens, J., and Miralles, D. G. (2022). Revisiting large-scale interception patterns constrained by a synthesis of global experimental data, Hydrol. Earth Syst. Sci., 26, 5647–5667. [https://doi.org/10.5194/hess-26-5647-2022](https://doi.org/10.5194/hess-26-5647-2022)

## 12. Migrating to HydroMT v1

HydroMT v1 introduces a component-based architecture to replace the previous inheritance model. Instead of all model functionality being defined in a single `Model` class, a model is now composed of modular `ModelComponent` classes such as `GridComponent`, `VectorComponent`, or `ConfigComponent`. This structure makes models more flexible, extensible, and easier to maintain. For detailed guidance, refer to the official [HydroMT migration guide](https://deltares.github.io/hydromt/latest/user_guide/migration_guide/index.html).

For HydroMT-Wflow, the names and arguments of a few setup methods have changed. The names of the maps in staticmaps have been updated to reflect better which process they belong to and maps that are not needed by Wflow.jl now start with `meta_` to indicate they are metadata only. The names of maps in staticmaps is now also customizable to avoid having to duplicate fully staticmaps.nc for small changes or scenario analysis.

**For HydroMT-Wflow, one of the major change is that** `wflow` **and** `WflowModel` **are now** `wflow_model` **and** `WflowSbmModel`**.**

### 12.1. YAML Configuration Changes

#### 12.1.1. Information

The HydroMT model configuration format has been overhauled and the ini format is not supported anymore. The root YAML file now includes three main keys: `modeltype`, `global`, and `steps`.

- `modeltype` (optional): Defines which model plugin is being used (e.g. `wflow_sbm` or `wflow_sediment`).
- `global`: Defines model-wide configuration, including data catalog(s), name of the model configuration toml file etc.
- `steps`: Replaces the old numbered dictionary format with a sequential list of function calls.

Some of the functions (component specific read and write) are now explicitly mapped to model or component methods using the <component>.<method> syntax.

For a complete example of the new configuration format, see the Wflow v1 YAML template: [`wflow_build.yml`](https://deltares.github.io/hydromt_wflow/stable/_downloads/91e3b6c950554d08e2fce9e28e2a82f9/wflow_build.yml).

For more information on the format changes, see this section in the HydroMT migration guide: [Changes to the yaml HydroMT configuration file format](https://deltares.github.io/hydromt/latest/user_guide/migration_guide/model_workflow.html).

#### 12.1.2. How to upgrade

In general, we advise to switch to the new YAML format by re-using and adapting the new template files provided in the HydroMT-Wflow examples folder:

- [`Wflow-SBM Build yml`](https://deltares.github.io/hydromt_wflow/stable/_downloads/91e3b6c950554d08e2fce9e28e2a82f9/wflow_build.yml)
- [`Wflow-Sediment Build yml`](https://deltares.github.io/hydromt_wflow/stable/_downloads/998603dba710c7a5357833971adbd4bb/wflow_sediment_build.yml)

If you do wish to update an existing file, you will have to do this manually. Here are some things you should be aware of:

- The `global` section remains the same. `config_fn` is renamed to `config_filename`.
- Add the `steps` section and convert the numbered dictionary to a list of function calls. Be careful with indents.
- Specific `read` and `write` functions are now at the component level. For example, `write_forcing` becomes `forcing.write`.
- `setup_config` now needs a `data` argument rather than directly the Wflow model options.
- The names of the setup methods have changed only for the reservoirs and lakes components: - `setup_lakes` becomes `setup_reservoirs_no_control` (some of the arguments have then also changed) - `setup_reservoirs` becomes `setup_reservoirs_simple_control`
- `setup_basemaps` must include the `region` (no longer available from CLI)
- `setup_rivers` method is now split into two methods: - `setup_rivers`: sets up the river network and cross-sections - `setup_river_roughness`: sets up manning roughness of the river
- `setup_lulcmaps` and equivalent: the parameters have changed. User-defined land use mapping tables will need to update the name of the columns.
- `setup_constant_pars`: the paramaters are now the Wflow.jl parameter names rather than a name that HydroMT adds to the staticmaps.
- `setup_other_demand` and `setup_lulcmaps_with_paddy`: some of the variables have been renamed (e.g. `industry_net`, `demand_paddy_h_min`).

### 12.2. Data Catalog Format Changes

#### 12.2.1. Information

The data catalog structure has been refactored to introduce a more modular design and clearer separation of responsibilities across several new classes (DataSource, Driver, URIResolver, and DataAdapter).

Key format changes:

- `path` renamed to `uri`
- `filesystem` or `driver_kwargs` moved under `driver`
- `unit_add`, `unit_mult`, `rename`, etc. moved under `data_adapter`
- `crs` and `nodata` moved under `metadata` (renamed from `meta`)
- A single catalog entry can now reference multiple data variants or versions

For detailed information about the format changes, see this section in the HydroMT migration guide: [Changes to the data catalog yaml file format](https://deltares.github.io/hydromt/latest/user_guide/migration_guide/data_catalog.html)

#### 12.2.2. How to upgrade

All existing pre-defined catalogs have been updated to the new format. For your own catalogs, you can upgrade easily with the HydroMT `check` command:

```bash
hydromt check -d /path/to/data_catalog.yml --format v0 --upgrade -v
```

### 12.3. Main Changes for Python Users

In HydroMT-Wflow v1, the internal data structure and API were redesigned to improve consistency and maintainability. Most changes affect how model components (such as `staticmaps` and `forcing`) are accessed and how model data is read and written. Another change is that to better differentiate between wflow SBM and flow Sediment, the `WflowModel` class is now `WflowSbmModel`.

#### 12.3.1. Component Changes

The model components are now **dedicated classes** rather than raw data objects (e.g., `xarray`, `dict`, or `geopandas`). Each component can be accessed via the `model` instance and exposes its underlying data through the `.data` property.

| v0.x | v1 |
| --- | --- |
| `model.grid` | `model.staticmaps` |
| `model.results` | `model.output_grid` `model.output_scalar` `model.output_csv` |
| `model.<component>` | `model.<component>.data` |
| `model.write_<component>()` | `model.<component>.write()` |
| `model.read_<component>()` | `model.<component>.read()` |
| `model.set_<component>()` | `model.<component>.set()` |

#### 12.3.2. Example: Accessing Component Data

Each component provides structured access to its data via the `.data` property.

```python
from hydromt_wflow import WflowSbmModel

model = WflowSbmModel(root="path/to/model", mode="r")

# Access xarray.Dataset of static maps
staticmaps = model.staticmaps.data

# Access geometries (GeoDataFrames)
geoms = model.geoms.data

# Access forcing data (xarray.Dataset)
forcing = model.forcing.data
```

#### 12.3.3. Example: Writing Components

Read and write operations are now handled at the **component level**.

```python
# Write configuration file
model.config.write()

# Write updated staticmaps to disk
model.staticmaps.write()

# Read forcing component explicitly
model.forcing.read()
```

These changes provide a clearer and more modular interface, making it easier to manipulate model components independently.

## 13. Migrating to Wflow.jl v1.0.0

The Wflow.jl v1 update mostly introduces new organisation of the model configuration (TOML), renamed or new (for sediment only) parameters and merging of lakes and reservoirs. For a complete overview of the changes, refer to the official [Wflow.jl changelog](https://deltares.github.io/Wflow.jl/dev/changelog.html).

To convert an existing v0.x wflow sbm model with hydromt, you can use the cli command:

```bash
hydromt update <model_type> <model_root_v0> -o <model_root_v1> -i <upgrade_v1.yml>  -v
```

Where

- `<model_type>` is `wflow_sbm` or `wflow_sediment`
- `<model_root>` is the folder containing your model
- `<upgrade_v1.yml>` is a configuration file specifying how to handle the migration.
- `<model_root_v1>` is the output folder for the migrated model.
- For sediment: `-d data_catalog.yml` to specify a data catalog for preparing the extra parameters of wflow sediment.

Template upgrade configuration files:

- [`Wflow-SBM Upgrade yml`](https://deltares.github.io/hydromt_wflow/stable/_downloads/a90c53160ab9df6d759b422d4dc7d9ae/wflow_update_v1_sbm.yml)
- [`Wflow-Sediment Upgrade yml`](https://deltares.github.io/hydromt_wflow/stable/_downloads/c74fe8e3446ba14f398b6584daec8492/wflow_update_v1_sediment.yml)

An example migration workflow notebook, is available [here](https://deltares.github.io/hydromt_wflow/stable/_examples/upgrade_to_wflow_v1.html#example-upgrade-to-wflow-v1).
