

# GENERAL STRESS TEST PARAMETERS ###############################################

# General R settings and prerequisites
source("./blueearth_cst/weathergen/global.R")

# Bind positional CLI args to named locals with an arity check (see
# generate_weather.R). Placed after source(global.R) so the arity stop() is the
# first thing to touch args.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4L) {
  stop("impose_climate_change.R expects 4 args: <realization_nc> <weagen_config_yaml> <stress_csv> <output_nc>")
}
rlz_path           <- args[[1]]
weagen_config_path <- args[[2]]
stress_csv_path    <- args[[3]]
output_nc_path     <- args[[4]]

# Config file — the ONE shared weathergen config from rule 3.04. C29 retired the
# per-member weathergen_config_rlz_<n>_cst_<m>.yml: it carried nothing that
# varied except the output filename, which Snakemake already knows because it is
# this rule's own declared output, so it now arrives as args[[4]].
yaml <- yaml::read_yaml(weagen_config_path)
# Stochastic weather realization to be perturbed
message("[impose_climate_change] Reading realization: ", rlz_path)
rlz_input <- weathergenr::read_netcdf(rlz_path, keep_leap_day = FALSE)
# Climate stress file
cst_data <- read.csv(stress_csv_path)


# General stress test parameters, derived from the declared output path.
# weathergenr::write_netcdf composes its filename as <prefix>_<suffix>.nc, so the
# stem is split at its LAST underscore. Deriving rather than passing prefix and
# suffix separately keeps ONE source of truth -- the Snakemake output
# declaration -- and is naming-agnostic: rlz_1_cst_2 and rlz_1_st_2 both split
# correctly, so a future member-token rename touches nothing here.
output_path    <- paste0(dirname(output_nc_path), "/")
output_stem    <- sub("\\.nc$", "", basename(output_nc_path))
if (!grepl("_", output_stem, fixed = TRUE)) {
  stop("cannot split '", output_stem, "' into weathergenr's <prefix>_<suffix>: ",
       "the declared output name carries no underscore")
}
nc_file_prefix <- sub("_[^_]+$", "", output_stem)
nc_file_suffix <- sub("^.*_", "", output_stem)

# temp_change_type / precip_change_type [boolean]
temp_change_transient   <- yaml$temp$transient_change
precip_change_transient <- yaml$precip$transient_change


# PARAMETERS CHANGING PER RUN ##################################################

# Apply climate changes to baseline weather data stored in the nc file.
# `diagnostic = FALSE` makes the return shape compatible with write_netcdf
# directly (a list of data.frames, one per grid cell — same as the old
# imposeClimateChanges return).
message("[impose_climate_change] Applying climate perturbations")
rlz_future <- weathergenr::apply_climate_perturbations(
   data               = rlz_input$data,
   grid               = rlz_input$grid,
   date               = rlz_input$date,
   precip_mean_factor = cst_data$precip_mean,
   precip_var_factor  = cst_data$precip_variance,
   temp_delta         = cst_data$temp_mean,
   temp_transient     = temp_change_transient,
   precip_transient   = precip_change_transient,
   compute_pet        = TRUE,
   qm_fit_method      = "mme",
   diagnostic         = FALSE
)

# Save to netcdf file
message("[impose_climate_change] Saving perturbed netcdf to: ", output_path)
weathergenr::write_netcdf(
   data          = rlz_future,
   grid          = rlz_input$grid,
   out_dir       = output_path,
   origin_date   = rlz_input$date[1],
   calendar      = "noleap",
   template_path = rlz_path,
   compression   = 4,
   spatial_ref   = "spatial_ref",
   file_prefix   = nc_file_prefix,
   file_suffix   = nc_file_suffix
)


################################################################################
