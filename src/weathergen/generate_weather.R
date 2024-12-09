# General R settings and prequisites
source("./src/weathergen/global.R")

# Install required packages -- ONLY ONCE!
# weathergen is assumed to be installed in R-environment
library(yaml)

args <- commandArgs(trailingOnly = TRUE)

# Pass command line options
yaml <- yaml::read_yaml(args[2])
weathergen_input_ncfile <- args[1]

# Parse global parameters from the yaml configuration file
historical_realizations_num <- yaml$generateWeatherSeries$realizations_num
weathergen_output_path <- yaml$generateWeatherSeries$output.path

# Step 1) Read weather data from the netcdf file
ncdata <- weathergenr::readNetcdf(weathergen_input_ncfile)

# Step 2) Generate new weather realizations
stochastic_weather <- weathergenr::generateWeatherSeries(
    weather.data = ncdata$data,
    weather.grid = ncdata$grid,
    weather.date = ncdata$date,
    variable.names = yaml$general$variables,
    sim.year.num = yaml$generateWeatherSeries$sim.year.num,
    sim.year.start = yaml$generateWeatherSeries$sim.year.start,
    month.start = yaml$generateWeatherSeries$month.start,
    realization.num = historical_realizations_num,
    warm.variable = yaml$generateWeatherSeries$warm.variable,
    warm.signif.level = yaml$generateWeatherSeries$warm.signif.level,
    warm.sample.num = yaml$generateWeatherSeries$warm.sample.num,
    # warm.subset.criteria = yaml$generateWeatherSeries$warm.subset.criteria, #not needeed
    knn.sample.num = yaml$generateWeatherSeries$knn.sample.num,
    mc.wet.quantile = yaml$generateWeatherSeries$mc.wet.quantile,
    mc.extreme.quantile = yaml$generateWeatherSeries$mc.extreme.quantile,
    dry.spell.change = yaml$generateWeatherSeries$dry.spell.change,
    wet.spell.change = yaml$generateWeatherSeries$wet.spell.change,
    # evaluate.model = yaml$generateWeatherSeries$evaluate.model,
    # evaluate.grid.num = yaml$generateWeatherSeries$evaluate.grid.num,
    output.path = weathergen_output_path,
    seed = yaml$generateWeatherSeries$seed,
    compute.parallel = yaml$generateWeatherSeries$compute.parallel
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
        nc.file.prefix = yaml$generateWeatherSeries$nc.file.prefix,
        nc.file.suffix = paste0(n,"_cst_0")
  )

}
