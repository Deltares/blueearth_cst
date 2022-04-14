

# GENERAL STRESS TEST PARAMETERS ###############################################

# General R settings and prequisites
source("./src/weathergen/global.R")

# Read parameters from the snkae yaml file
yaml <- yaml::read_yaml(snakemake@params[["snake_config"]])

# Read parameters from the defaults yaml file
yaml_defaults <- yaml::read_yaml(snakemake@params[["weagen_config"]])

# General stress test parameters
output_path <- snakemake@params[["output_path"]]
nc_file_prefix <- snakemake@params[["nc_file_prefix"]]

# temp_change_type/precip_change_type [string]
temp_change_type = yaml$temp$change_type
precip_change_type = yaml$precip$change_type


# PARAMETERS CHANGING PER RUN ##################################################

# Stochastic weather realization to be perturbed
stochastic_nc <- snakemake@input[["rlz_nc"]]
rlz_input <- weathergenr::readNetcdf(stochastic_nc)

# scenario run identifier
nc_file_suffix <- snakemake@params[["nc_file_suffix"]]

# Climate stress file
cst_csv_fn <- snakemake@input[["st_csv"]]
cst_data <- read.csv(cst_csv_fn)

# Apply climate changes to baseline weather data stored in the nc file
rlz_future <- weathergenr::imposeClimateChanges(
   climate.data = rlz_input$data,
   climate.grid = rlz_input$grid,
   sim.dates = rlz_input$date,
   change.factor.precip.mean = cst_data$precip_mean ,
   change.factor.precip.variance = cst_data$precip_variance,
   change.factor.temp.mean = cst_data$temp_mean,
   change.type.temp = temp_change_type,
   change.type.precip = precip_change_type,
   calculate.pet = TRUE
)

# Save to netcdf file
weathergenr::writeNetcdf(
 data = rlz_future,
 coord.grid = rlz_input$grid,
 output.path = output_path,
 origin.date =  rlz_input$date,
 calendar.type = "noleap",
 nc.template.file = stochastic_nc,
 nc.compression = 4,
 nc.spatial.ref = "spatial_ref",
 nc.file.prefix = nc_file_prefix,
 nc.file.suffix = nc_file_suffix
)


################################################################################

