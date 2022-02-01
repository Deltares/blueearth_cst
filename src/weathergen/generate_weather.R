

#library(weathergenr)
devtools::install_github("Deltares/weathergenr")

yaml <- yaml::read_yaml(snakemake@params[["weagen_config"]])

# General settings
weathergen_output_path <- snakemake@params[["output_path"]]
historical_realizations_num <- yaml$general$historical_realizations_num
variables <- yaml$general$variables
weathergen_input_ncfile <- snakemake@input[["climate_nc"]]

# STEP 1) READ BASELINE HISTORICAL DATA FROM NETCDF
ncdata <- weathergenr::readNetcdf(weathergen_input_ncfile)

# STEP 2) GENERATE NEW REALIZATIONS OF HISTORICAL WEATHER DATA
stochastic_weather <- weathergenr::generateWeatherSeries(
     output.path = weathergen_output_path,
     realization.num = historical_realizations_num,
     variable.names = variables,
     weather.data = ncdata$data,
     weather.grid = ncdata$grid,
     weather.date = ncdata$date,
     month.start = yaml$generateWeatherSeries$month.start,
     warm.variable = yaml$generateWeatherSeries$warm.variable,
     warm.signif.level = yaml$generateWeatherSeries$warm.signif.level,
     warm.sample.num = yaml$generateWeatherSeries$warm.sample.num,
     warm.subset.criteria= yaml$generateWeatherSeries$warm.subset.criteria,
     knn.sample.num = yaml$generateWeatherSeries$knn.sample.num,
     mc.wet.threshold = yaml$generateWeatherSeries$mc.wet.threshold,
     mc.extreme.quantile = yaml$generateWeatherSeries$mc.extreme.quantile,
     evaluate.model = yaml$generateWeatherSeries$evaluate.model,
     evaluate.grid.num = yaml$generateWeatherSeries$evaluate.grid.num,
     seed = yaml$generateWeatherSeries$seed
)

# STEP 3) SAVE EACH GENERATED REALIZATION TO A NETCDF FILE

for (n in 1:historical_realizations_num) {

  # Resample order
  day_order <- match(stochastic_weather$resampled[[n]], ncdata$date)

  # Obtain stochastic series by re-ordering historical data
  stochastic_rlz <- lapply(ncdata$data, function(x) x[day_order,])

  # save to netcdf
  weathergenr::writeNetcdf(
        data = stochastic_rlz,
        coord.grid = ncdata$grid,
        output.path = paste0(weathergen_output_path,"realization_",n,"/"),
        origin.date =  stochastic_weather$dates[1],
        calendar.type = "noleap",
        nc.template.file = weathergen_input_ncfile,
        nc.compression = 4,
        nc.spatial.ref = "spatial_ref",
        nc.file.prefix = "rlz",
        nc.file.suffix = paste0(n,"_historical")
  )

}

