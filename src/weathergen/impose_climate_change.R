

# GENERAL STRESS TEST PARAMETERS ###############################################

# General R settings and prequisites
source("./src/weathergen/global.R")

args <- commandArgs(trailingOnly = TRUE)

# Pass command line options
# Config file
yaml <- yaml::read_yaml(args[2])
# Stochastic weather realization to be perturbed
rlz_fn <- args[1]
rlz_input <- weathergenr::readNetcdf(rlz_fn, leap.days = FALSE)
# Climate stress file
cst_data <- read.csv(args[3])


# General stress test parameters
output_path <- yaml$imposeClimateChanges$output.path
nc_file_prefix <- yaml$imposeClimateChanges$nc.file.prefix
nc_file_suffix <- yaml$imposeClimateChanges$nc.file.suffix

# temp_change_type/precip_change_type [string]
temp_change_transient = yaml$temp$transient_change
precip_change_transient = yaml$precip$transient_change


# PARAMETERS CHANGING PER RUN ##################################################

# Apply climate changes to baseline weather data stored in the nc file
rlz_future <- weathergenr::imposeClimateChanges(
   climate.data = rlz_input$data,
   climate.grid = rlz_input$grid,
   sim.dates = rlz_input$date,
   change.factor.precip.mean = cst_data$precip_mean ,
   change.factor.precip.variance = cst_data$precip_variance,
   change.factor.temp.mean = cst_data$temp_mean,
   transient.temp.change = temp_change_transient,
   transient.precip.change = precip_change_transient,
   calculate.pet = TRUE,
   compute.parallel = FALSE,
   num.cores = NULL,
   fit.method = "mme"
)

# Save to netcdf file
weathergenr::writeNetcdf(
   data = rlz_future,
   coord.grid = rlz_input$grid,
   output.path = output_path,
   origin.date =  rlz_input$date[1],
   calendar.type = "noleap",
   nc.template.file = rlz_fn,
   nc.compression = 4,
   nc.spatial.ref = "spatial_ref",
   nc.file.prefix = nc_file_prefix,
   nc.file.suffix = nc_file_suffix
)


################################################################################

