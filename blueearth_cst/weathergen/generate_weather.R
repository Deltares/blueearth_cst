# General R settings and prerequisites
source("./blueearth_cst/weathergen/global.R")
# The pure decisions on our side of the seam (output split, basin subset).
source("./blueearth_cst/weathergen/member_helpers.R")

# weathergenr is assumed to be installed in R-environment.
# See dev/scripts/install_weathergenr.R for the install path.
library(yaml)

# Bind positional CLI args to named locals with an arity check, so a wrong
# number of args fails loudly here rather than surfacing as a cryptic NA
# downstream. Placed after source(global.R) so the arity stop() is the first
# thing to touch args.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 6L) {
  stop("generate_weather.R expects 6 args: <climate_nc> <weathergen_config_yaml> ",
       "<rlz_index_width> <st_index_width> <basin_cells_csv> <root_run_ids>")
}
climate_nc_path    <- args[[1]]
weathergen_config_path <- args[[2]]
# Member indices are zero-padded to a width derived from the COUNT (C27). The
# widths are computed ONCE, in the Snakefile, and passed in -- never re-derived
# here. Re-deriving would mean reimplementing stress_test_grid's arithmetic in R,
# and a cross-language copy of a filename rule is exactly the kind of
# producer/declaration disagreement that --dry-run cannot see.
rlz_index_width    <- as.integer(args[[3]])
st_index_width     <- as.integer(args[[4]])
if (is.na(rlz_index_width) || is.na(st_index_width) ||
    rlz_index_width < 1L || st_index_width < 1L) {
  stop("generate_weather.R needs positive integer index widths, got: ",
       args[[3]], " / ", args[[4]])
}
# The store's basin-cell mask (rule 1.04/3.08 writes it beside the extraction).
# Passed as a path rather than recomputed here: R has no geometry library in
# this env, and the producer is the only place holding both the grid and the
# region polygon.
basin_cells_path   <- args[[5]]
root_run_ids <- strsplit(args[[6]], ",", fixed = TRUE)[[1]]
if (length(root_run_ids) == 0L || any(!grepl("^run_[0-9]+$", root_run_ids)) ||
    anyDuplicated(root_run_ids)) {
  stop("root_run_ids must be a nonempty, comma-separated list of unique prefixed run IDs")
}

yaml <- yaml::read_yaml(weathergen_config_path)

# Parse global parameters from the yaml configuration file. Section and key
# names are weathergenr's own function and argument names (renamed 2026-08-12
# from the pre-1.2.0 `generateWeatherSeries` spelling; tracking 2.0.0 since
# 2026-08-17), so the assignments below are a pass-through rather than a
# translation table.
gw <- yaml$generate_weather
wnc <- yaml$write_netcdf
rwg <- yaml$run_weather_generator
historical_realizations_num <- gw$n_realizations
if (length(root_run_ids) != historical_realizations_num) {
  stop("root_run_ids count ", length(root_run_ids),
       " does not match n_realizations ", historical_realizations_num)
}
# `out_dir` is the generator subtree ROOT --
# experiments/<id>/climate/weathergenr/ -- not a write directory (R07 B5 set
# the split; R9 P2 moved the subtree under climate/ and gave it the engine's
# own name).
# weathergenr::generate_weather writes BOTH its diagnostic figures and its two
# date CSVs into a single out_dir; the R07 layout separates products from
# figures, so the split is done here -- on our side of the seam -- rather than
# by asking upstream for two output directories.
weathergen_paths_ <- weathergen_paths(gw$out_dir)
weathergen_output_path <- weathergen_paths_$output
weathergen_plots_path <- weathergen_paths_$plots

# Step 1) Read weather data from the netcdf file
log_row("Reading weather netcdf: ", climate_nc_path)
ncdata <- weathergenr::read_netcdf(climate_nc_path)

# Step 1b) Restrict the RESAMPLING to the cells the basin touches, with the
# subset's ids renumbered to index its data list. The why -- an unweighted
# spatial mean upstream, and the rule 3.11 id mismatch -- is on
# subset_basin_cells in member_helpers.R.
basin <- subset_basin_cells(
  ncdata, utils::read.csv(basin_cells_path), source_label = climate_nc_path
)
log_row("Resampling on ", length(basin$keep), " basin cell(s) of ",
        length(ncdata$data), " in the store")
obs_data_basin <- basin$data
obs_grid_basin <- basin$grid

# Step 2) Generate new weather realizations
log_row("Generating ", historical_realizations_num,
        " weather realization(s)")
# run_weather_generator, not generate_weather: the wrapper runs the SAME
# generation and then the evaluation pass (prepare_evaluation_data +
# evaluate_weather_generator), whose diagnostic plots are the point of using it.
# It takes the generation arguments as ONE `config` list, and because the config
# section is named for generate_weather's arguments, that list IS the section --
# no translation table, which is what the 1.2.0 rename bought.
#
# weathergenr 2.0.0 (2026-08-17) changed how that config list is consumed, in
# our favour: an entry the config does not name is now DROPPED from the call
# instead of forwarded as an explicit NULL that overwrote the receiving
# function's default. It also forwards `relax_order` -- 1.2.0's one unforwarded
# argument, spelled `relax_priority` then -- so the key is back in the config
# and pinned in the contract (t2608121742).
#
# `config$out_dir` is not read by the wrapper (out_dir is its own argument), but
# it is not dead either: OUR code above derives the output/ and plots/ split
# from it.
#
# C34. weathergenr 1.2.0 split evaluation into its own exports, so the config's
# old `evaluate.model` reached NOTHING -- plot emission is `save_plots`, which
# defaulted TRUE. Setting evaluate.model: FALSE therefore did not stop the
# plots, which is what the key claimed to do. It now governs BOTH the generation
# figures and the evaluation ones, since the wrapper forwards it to each.
# Read presentation tokens in this R process, after identity resolution.
source("./blueearth_cst/weathergen/plot_style.R")
if (!all(c("export_weather_plot", "style_weather_plot", "arrange_weather_panels") %in%
         getNamespaceExports("weathergenr"))) {
  stop("WF3 shared styling requires the weathergenr plotting interface ",
       "(t2610241523c).")
}
save_weather_plots <- isTRUE(gw$save_plots)
gw$return_plots <- save_weather_plots
gw$save_plots <- FALSE
gw$plot_style <- read_cst_weather_style()

weathergen_run <- weathergenr::run_weather_generator(
    # The BASIN subset: this call decides WHICH DAYS get resampled, and that
    # decision should reflect the basin's climate, not the buffer's. The full
    # grid is re-attached below, where the realizations are built.
    obs_data       = obs_data_basin,
    obs_grid       = obs_grid_basin,
    obs_dates      = ncdata$date,
    out_dir        = weathergen_output_path,
    config         = gw,
    eval_max_grids = rwg$eval_max_grids,
    log_messages   = rwg$log_messages
)
# The wrapper returns list(gen_output=, evaluation=, log_path=); everything
# below consumes the generation half, which is what generate_weather returned
# directly before the swap.
stochastic_weather <- weathergen_run$gen_output

# Export diagnostics directly into the figure directory.
if (save_weather_plots) {
  export_cst_weather_plots(weathergen_run, weathergen_plots_path)
}

# STEP 3) Save each stochastic realization back to a netcdf file
for (n in 1:historical_realizations_num) {

  log_row("Saving realization ", n, " of ",
          historical_realizations_num)

  # New return: $resampled is a data.frame with columns rlz_1, rlz_2, ...
  rlz_dates <- stochastic_weather$resampled[[paste0("rlz_", n)]]
  day_order <- match(rlz_dates, ncdata$date)

  # Obtain stochastic series by re-ordering historical data.
  #
  # The FULL grid, deliberately -- not the basin subset the resampling ran on.
  # The day order is a basin decision; the cells carried through are a
  # downscaling requirement, because rule 3.14 regrids these realizations onto
  # the wflow grid and needs the surrounding ring for the same reason rule 1.10
  # does. Subsetting here instead would fix the climate signal and break the
  # downscaling.
  stochastic_rlz <- lapply(ncdata$data, function(x) x[day_order, ])

  # Write directly to the declared temporary member path. Python owns the
  # row allocation; the ordered IDs here bind its roots to generator draws.
  rlz_out_dir <- weathergen_output_path
  weathergenr::write_netcdf(
        data          = stochastic_rlz,
        grid          = ncdata$grid,
        out_dir       = rlz_out_dir,
        origin_date   = stochastic_weather$dates[1],
        calendar      = wnc$calendar,
        template_path = climate_nc_path,
        compression   = wnc$compression,
        spatial_ref   = wnc$spatial_ref,
        signif_digits = wnc$signif_digits,
        verbose       = wnc$verbose,
        file_prefix   = "run",
        # Upstream joins prefix and suffix with '_'; IDs already contain it.
        file_suffix   = sub("^run_", "", root_run_ids[[n]])
  )

  # A `spatial_ref` re-patching block stood here until the weathergenr 2.0.0
  # upgrade (2026-08-17, t2608071225). weathergenr::write_netcdf did not carry
  # the template's spatial_ref attributes into its output, and downstream
  # (impose_climate_change.R) uses the realization file as ITS template and
  # needs `x_dim` / `y_dim` on that variable -- without them it failed with
  # "attempt to select less than one element" -- so the attributes were copied
  # back from the historical template by hand.
  #
  # 2.0.0's write_netcdf copies the spatial_ref VALUE and every atomic
  # attribute from the template unconditionally, which is the removal condition
  # the block carried. Verified rather than assumed, on the era5 store fixture:
  # a round trip through read_netcdf + write_netcdf returns all 13 template
  # attributes, `x_dim` and `y_dim` among them, each with hasatt=TRUE.

}
