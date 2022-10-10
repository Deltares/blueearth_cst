
# Install required packages -- ONLY ONCE!
# source("./src/weathergen/install_rpackages.r")

# General R settings and prequisites
source("./src/weathergen/global.R")

# Read parameters from the snkae yaml file
yaml <- yaml::read_yaml(snakemake@params[["snake_config"]])

# Read parameters from the defaults yaml file
yaml_defaults <- yaml::read_yaml(snakemake@params[["weagen_config"]])

historical_realizations_num <- yaml$realizations_num
variables <- yaml_defaults$general$variables

# Parameters set through snakemake
weathergen_output_path <- snakemake@params[["output_path"]]
weathergen_input_ncfile <- snakemake@input[["climate_nc"]]
sim_year_start <- snakemake@params[["start_year"]]
sim_year_num <- snakemake@params[["sim_years"]]
nc_file_prefix <- snakemake@params[["nc_file_prefix"]]

# Step 1) Read weather data from the netcdf file
ncdata <- weathergenr::readNetcdf(weathergen_input_ncfile)

# Step 2) Generate new weather realizations
stochastic_weather <- weathergenr::generateWeatherSeries(
     output.path = paste0(weathergen_output_path, "plots/"),
     realization.num = historical_realizations_num,
     variable.names = variables,
     weather.data = ncdata$data,
     weather.grid = ncdata$grid,
     weather.date = ncdata$date,
     sim.year.num = sim_year_num,
     sim.year.start = sim_year_start,
     month.start = yaml_defaults$generateWeatherSeries$month.start,
     warm.variable = yaml_defaults$generateWeatherSeries$warm.variable,
     warm.signif.level = yaml$warm.signif.level,
     warm.sample.num = yaml$warm.sample.num,
     warm.subset.criteria = yaml_defaults$generateWeatherSeries$warm.subset.criteria,
     knn.sample.num = yaml$knn.sample.num,
     mc.wet.quantile = yaml_defaults$generateWeatherSeries$mc.wet.quantile,
     mc.extreme.quantile = yaml_defaults$generateWeatherSeries$mc.extreme.quantile,
     evaluate.model = yaml_defaults$generateWeatherSeries$evaluate.model,
     evaluate.grid.num = yaml_defaults$generateWeatherSeries$evaluate.grid.num,
     seed = yaml_defaults$generateWeatherSeries$seed,
     compute.parallel = yaml_defaults$generateWeatherSeries$compute.parallel,
     num.cores = yaml_defaults$generateWeatherSeries$num.cores
)

# STEP 3) Save each stochastic realization back to a netcdf file

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
        nc.file.prefix = nc_file_prefix,
        nc.file.suffix = paste0(n,"_cst_0")
  )

}

