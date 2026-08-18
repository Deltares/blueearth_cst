# General R settings and prerequisites
source("./blueearth_cst/weathergen/global.R")

# weathergenr is assumed to be installed in R-environment.
# See dev/scripts/install_weathergenr.R for the install path.
library(yaml)

# Bind positional CLI args to named locals with an arity check, so a wrong
# number of args fails loudly here rather than surfacing as a cryptic NA
# downstream. Placed after source(global.R) so the arity stop() is the first
# thing to touch args.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5L) {
  stop("generate_weather.R expects 5 args: <climate_nc> <weagen_config_yaml> ",
       "<rlz_index_width> <st_index_width> <basin_cells_csv>")
}
climate_nc_path    <- args[[1]]
weagen_config_path <- args[[2]]
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
pad <- function(value, width) sprintf(paste0("%0", width, "d"), as.integer(value))
# The reserved unperturbed baseline this rule writes, padded like any member.
st_baseline <- pad(0L, st_index_width)

yaml <- yaml::read_yaml(weagen_config_path)

# Parse global parameters from the yaml configuration file. Section and key
# names are weathergenr's own function and argument names (renamed 2026-08-12
# from the pre-1.2.0 `generateWeatherSeries` spelling; tracking 2.0.0 since
# 2026-08-17), so the assignments below are a pass-through rather than a
# translation table.
gw <- yaml$generate_weather
wnc <- yaml$write_netcdf
rwg <- yaml$run_weather_generator
historical_realizations_num <- gw$n_realizations
# `out_dir` is the generator subtree ROOT --
# experiments/<id>/climate/weathergenr/ -- not a write directory (R07 B5 set
# the split; R9 P2 moved the subtree under climate/ and gave it the engine's
# own name).
# weathergenr::generate_weather writes BOTH its diagnostic figures and its two
# date CSVs into a single out_dir; the R07 layout separates products from
# figures, so the split is done here -- on our side of the seam -- rather than
# by asking upstream for two output directories.
weathergen_root <- gw$out_dir
weathergen_output_path <- paste0(weathergen_root, "output/")
weathergen_plots_path <- paste0(weathergen_root, "plots/")

# Step 1) Read weather data from the netcdf file
log_row("Reading weather netcdf: ", climate_nc_path)
ncdata <- weathergenr::read_netcdf(climate_nc_path)

# Step 1b) Restrict the RESAMPLING to the cells the basin touches.
#
# weathergenr picks which years to resample from a spatial mean of every cell it
# is handed (`compute_area_averages`: sum over n_grids, divided by n_grids -- no
# mask, no weights). The store is a bbox read plus a buffer, so those cells
# include neighbouring climate the basin never sees. On gabon_1008 the basin
# spans 0.80 x 0.53 ERA5 cells and touches 2 of the store's cells, so most of
# the signal steering that stress test came from outside the basin.
#
# The mask is a FILTER, not a weighting (owner ruling 2026-08-10): a cell either
# touches the basin or it does not, and the ones that do count equally. That is
# exactly what weathergenr's own unweighted mean computes -- once it is given
# the right subset -- so nothing upstream needs changing, which matters because
# weathergenr is a vendored package we do not patch.
#
# Coordinates are matched, never indices: both sides enumerate the grid their
# own way and an index convention would break silently.
basin_cells <- utils::read.csv(basin_cells_path)
grid_key <- paste(round(ncdata$grid$y, 6), round(ncdata$grid$x, 6))
mask_key <- paste(round(basin_cells$latitude, 6), round(basin_cells$longitude, 6))
keep <- which(grid_key %in% mask_key)
if (length(keep) == 0L) {
  stop("basin_cells.csv matched no cell in ", climate_nc_path,
       " -- the mask and the store disagree about the grid")
}
log_row("Resampling on ", length(keep), " basin cell(s) of ",
        length(ncdata$data), " in the store")
obs_data_basin <- ncdata$data[keep]
obs_grid_basin <- ncdata$grid[keep, , drop = FALSE]

# Renumber the subset's cell ids to 1..n, matching the data list handed over
# with it.
#
# `read_netcdf` stamps `id = seq_len(nrow(grid))` across the FULL store grid, so
# a row subset keeps the ORIGINAL ids -- cells 7 and 12 of 20 stay 7 and 12 --
# while `ncdata$data[keep]` is renumbered 1..2 by the subset itself. The two
# then disagree.
#
# It went unnoticed until the 2026-08-13 swap to `run_weather_generator`:
# `generate_weather` never reads `grid$id` (it assigns its own when absent), but
# the wrapper's evaluation pass derives `grid_ids` from that column and requires
# them to index `obs_data`. Rule 3.11 failed with "'grid_ids' must match names
# or indices of obs_data" on any basin that is not the whole store -- i.e. every
# real basin. Reproduced and fixed against a 20-cell grid subset to 2.
if ("id" %in% names(obs_grid_basin)) {
  obs_grid_basin$id <- seq_len(nrow(obs_grid_basin))
}

# Assert the contract the renumbering above exists to hold: the grid describes
# exactly the cells in the data list, in the same order. weathergenr states this
# only as a downstream failure -- "'grid_ids' must match names or indices of
# obs_data" -- raised deep inside the evaluation pass, AFTER ~54 seconds of
# generation, with nothing naming the subset as the cause. Checking it here
# costs nothing and fails at the point the mistake is made.
if (nrow(obs_grid_basin) != length(obs_data_basin)) {
  stop("basin subset is inconsistent: ", nrow(obs_grid_basin),
       " grid rows vs ", length(obs_data_basin), " data cells")
}
if ("id" %in% names(obs_grid_basin) &&
    !identical(as.integer(obs_grid_basin$id), seq_along(obs_data_basin))) {
  stop("basin subset grid ids do not index its data list: ids are ",
       paste(utils::head(obs_grid_basin$id, 5), collapse = ", "),
       " for ", length(obs_data_basin), " cells")
}

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

# Step 2b) Move the diagnostic figures into plots/. weathergenr writes figures
# and products into ONE out_dir; the R07 layout separates them, so the split is
# done here rather than by asking upstream for two output directories.
#
# Globbed by EXTENSION, not by a name list. The four generation figures were
# named literally until the run_weather_generator swap (2026-08-12); the
# evaluation pass that swap added builds its filenames dynamically, so there is
# no list to extend -- and a name list silently leaves behind anything upstream
# adds or renames. Everything that is not an image stays put: the two date CSVs
# (sim_dates.csv, resampled_dates.csv) and the realization .nc files are
# generator PRODUCTS and belong in output/.
dir.create(weathergen_plots_path, recursive = TRUE, showWarnings = FALSE)
weathergen_figures <- list.files(
  weathergen_output_path, pattern = "[.](png|pdf)$", ignore.case = TRUE,
  full.names = FALSE
)
log_row("Moving ", length(weathergen_figures),
        " figure(s) to ", weathergen_plots_path)
for (fig in weathergen_figures) {
  file.rename(file.path(weathergen_output_path, fig),
              file.path(weathergen_plots_path, fig))
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

  # save to netcdf. Every realization NC lands flat in
  # climate/weathergenr/output/, its index carried by the file name -- R07 B5
  # dissolved the realization_<n>/ level, R9 P2 renamed the subtree.
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
        file_prefix   = wnc$file_prefix,
        file_suffix   = paste0(pad(n, rlz_index_width), "_st_", st_baseline)
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
