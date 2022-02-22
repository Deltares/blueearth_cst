
# General R settings and prequisites
source("./src/weathergen/global.R")

# Read config
yaml <- yaml::read_yaml(snakemake@params[["weagen_config"]])

# path to the base nc file [string]
output_path <- snakemake@params[["output_path"]]

# What prefix we want to attach to the final scenario name when writing back to nc? [string]
nc_file_prefix <- snakemake@params[["nc_file_prefix"]]

# What suffix we want to attach to the final scenario name when writing back to nc? [string]
# (index to keep track of both natural variability realization and current climate change run)
nc_file_suffix <- snakemake@params[["nc_file_suffix"]]

# temp_change_type [string]
temp_change_type = yaml$temp$change_type

# precip_change_type [string]
precip_change_type = yaml$precip$change_type

### RUN SPECIFIC INPUTS (changes in the loop)

# stochastic_nc = Name of the gridded historical realization nc file [string]
stochastic_nc <- snakemake@input[["rlz_nc"]]

# Climate stress file
cst_csv_fn <- snakemake@input[["st_csv"]]
cst_data <- read.csv(cst_csv_fn)

# vector of monthly precip mean change factors [numeric vector with 12 values]
current_precip_mean_change <- cst_data$temp_mean

# strtest_matrix_precip_variance = vector of monthly precip variance change factors [numeric vector with 12 values]
current_precip_variance_change <- cst_data$precip_mean

# strtest_matrix_temp_mean = vector of monthly temperature mean changes, ad DegC [numeric vector with 12 values]
current_temp_mean_change <- cst_data$precip_variance


################################################################################
################################################################################
################################################################################


# THIS IS THE MAIN WORKFLOW TO BE CALLED FROM R #

#rlz_input_name <- paste0(output_path, "/", stochastic_nc)
rlz_input <- readNetcdf(stochastic_nc)

# Apply climate changes to baseline weather data stored in the nc file
rlz_future <- imposeClimateChanges(
   climate.data = rlz_input$data,
   climate.grid = rlz_input$grid,
   sim.dates = rlz_input$date,
   change.factor.precip.mean = current_precip_mean_change,
   change.factor.precip.variance = current_precip_variance_change,
   change.factor.temp.mean = current_temp_mean_change,
   change.type.temp = temp_change_type,
   change.type.precip = precip_change_type)

 # Save to netcdf file
 writeNetcdf(
   data = rlz_future,
   coord.grid = rlz_input$grid,
   output.path = output_path,
   origin.date =  rlz_input$date,
   calendar.type = "noleap",
   nc.template.file = stochastic_nc,
   nc.compression = 4,
   nc.spatial.ref = "spatial_ref",
   nc.file.prefix = nc_file_prefix,
   nc.file.suffix = nc_file_suffix)


################################################################################

