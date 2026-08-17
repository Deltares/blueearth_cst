

# GENERAL STRESS TEST PARAMETERS ###############################################

# General R settings and prerequisites
source("./blueearth_cst/weathergen/global.R")
# The R side of WG-2: one member's twelve monthly rows, carrying the
# postcondition that makes a mis-keyed slice loud instead of silently short.
source("./blueearth_cst/weathergen/read_member_grid.R")

# Bind positional CLI args to named locals with an arity check (see
# generate_weather.R). Placed after source(global.R) so the arity stop() is the
# first thing to touch args.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5L) {
  stop("impose_climate_change.R expects 5 args: <realization_nc> <weagen_config_yaml> <lookup_csv> <output_nc> <st_id>")
}
rlz_path           <- args[[1]]
weagen_config_path <- args[[2]]
lookup_csv_path    <- args[[3]]
output_nc_path     <- args[[4]]
# The member token, from rule 3.12's {st_num} wildcard. Already zero-padded to
# ST_WIDTH by that rule's own wildcard_constraints, so it is textually identical
# to the lookup's `st_id` -- C27's one-token identity, now checked on the
# CONSUMING side because the pad width is no longer carried by a filename.
st_id_token        <- args[[5]]

# Config file — the ONE shared weathergen config from rule 3.10. C29 retired the
# per-member weathergen_config_rlz_<n>_cst_<m>.yml: it carried nothing that
# varied except the output filename, which Snakemake already knows because it is
# this rule's own declared output, so it now arrives as args[[4]].
yaml <- yaml::read_yaml(weagen_config_path)
# Stochastic weather realization to be perturbed
message("[impose_climate_change] Reading realization: ", rlz_path)
rlz_input <- weathergenr::read_netcdf(rlz_path, keep_leap_day = FALSE)
# This member's slice of the experiment's stress-test lookup: twelve rows in
# month order, or a stop() naming the token.
cst_data <- read_member_grid(lookup_csv_path, st_id_token)

# PERCENT -> the generator's multiplier form (WG-2). The rule is stated over the
# percent COLUMNS rather than per column, so a future column inherits it:
# `precip_variance_change` converts on a path no shipped config exercises
# (variance is flat at 1.0, i.e. 0.0 percent, everywhere), and omitting it here
# would hand apply_climate_perturbations a variance factor of ZERO rather than
# the identity 1.0 on the default configuration.
#
# `1 + p/100` and not `(100 + p)/100`: measured over 200k random float32
# multipliers, the first fails to reproduce the level in 19.9% of cases and the
# second in 32.9%. The round trip is NOT exact in general and cannot be made so
# -- 1,155 of 50,000 levels admit no float64 percent that inverts them exactly
# -- but it is within one float64 ulp for every multiplier >= 0.5, which the
# producer refuses to go below (D35), and bit-identical on every level of every
# shipped config.
precip_mean_factor     <- 1 + cst_data$precip_change / 100
precip_variance_factor <- 1 + cst_data$precip_variance_change / 100
# temp_change is additive degC and crosses unconverted in both directions, so
# there is nothing to reconstruct and nothing that can be out of bound.


# General stress test parameters, derived from the declared output path.
# weathergenr::write_netcdf composes its filename as <prefix>_<suffix>.nc, so the
# stem is split at its LAST underscore. Deriving rather than passing prefix and
# suffix separately keeps ONE source of truth -- the Snakemake output
# declaration -- and is naming-agnostic: rlz_1_cst_2 and rlz_1_st_2 both split
# correctly, which R11 P2's member-token rename then demonstrated -- it changed
# the declared name and touched nothing here.
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

# Section and key names are weathergenr's own function and argument names
# (renamed 2026-08-12 from `generateWeatherSeries`; tracking 2.0.0 since
# 2026-08-17), so these are pass-throughs.
#
# 2.0.0 makes a year-varying `n_years x 12` factor MATRIX an error when its
# transient flag is TRUE, because only row 1 would be read. This call is
# unaffected: every factor below is a length-12 monthly VECTOR read from the
# member's twelve lookup rows, which weathergenr expands to identical rows and
# accepts under either flag.
acp <- yaml$apply_climate_perturbations
wnc <- yaml$write_netcdf


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
   precip_mean_factor = precip_mean_factor,
   precip_var_factor  = precip_variance_factor,
   temp_delta         = cst_data$temp_change,
   temp_transient     = temp_change_transient,
   precip_transient   = precip_change_transient,
   precip_occurrence_transient = acp$precip_occurrence_transient,
   precip_intensity_threshold  = acp$precip_intensity_threshold,
   compute_pet        = acp$compute_pet,
   qm_fit_method      = acp$qm_fit_method,
   scale_var_with_mean = acp$scale_var_with_mean,
   enforce_target_mean = acp$enforce_target_mean,
   exaggerate_extremes = acp$exaggerate_extremes,
   extreme_prob_threshold = acp$extreme_prob_threshold,
   extreme_k          = acp$extreme_k,
   precip_cap_mm_day  = acp$precip_cap_mm_day,
   precip_floor_mm_day = acp$precip_floor_mm_day,
   precip_cap_quantile = acp$precip_cap_quantile,
   verbose            = acp$verbose,
   # LOAD-BEARING, and the config says so. `diagnostic = FALSE` makes the return
   # a list of per-cell data.frames, which write_netcdf below consumes directly
   # (the same shape the old imposeClimateChanges returned). TRUE -- weathergenr's
   # own default -- returns a diagnostic structure and the next call fails.
   diagnostic         = acp$diagnostic,
   # C34/F15. Generation is seeded and the perturbation was not, so the two
   # halves of one experiment had different reproducibility guarantees and
   # nobody chose that. Passing the SAME seed the generator uses makes the whole
   # chain reproducible; if the function turns out to be deterministic this is a
   # no-op, and either way the asymmetry is now a decision rather than an
   # oversight. There is deliberately no seed key in the
   # `apply_climate_perturbations` config section -- one seed cannot diverge.
   seed               = yaml$generate_weather$seed,
   # C34/F16. PET is computed twice in this chain -- here, and again from the
   # perturbed temperature by rule 3.14's setup_temp_pet_forcing -- by two
   # different methods, neither of which was chosen. Surfaced at weathergenr's
   # own default so this step's method is now stated; whether the first result
   # is used at all is the open half of F16 and is NOT settled here.
   pet_method         = acp$pet_method
)

# Save to netcdf file
message("[impose_climate_change] Saving perturbed netcdf to: ", output_path)
weathergenr::write_netcdf(
   data          = rlz_future,
   grid          = rlz_input$grid,
   out_dir       = output_path,
   origin_date   = rlz_input$date[1],
   calendar      = wnc$calendar,
   template_path = rlz_path,
   compression   = wnc$compression,
   spatial_ref   = wnc$spatial_ref,
   signif_digits = wnc$signif_digits,
   verbose       = wnc$verbose,
   # Derived from this rule's declared output above, NOT from
   # write_netcdf.file_prefix -- that key carries the generation step's prefix
   # (rule 3.11), and the perturbed series is named for its own member.
   file_prefix   = nc_file_prefix,
   file_suffix   = nc_file_suffix
)


################################################################################
