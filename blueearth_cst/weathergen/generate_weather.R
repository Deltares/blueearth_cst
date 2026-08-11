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

# Parse global parameters from the yaml configuration file
historical_realizations_num <- yaml$generateWeatherSeries$realizations_num
# `output.path` is the generator subtree ROOT --
# experiments/<id>/climate/weathergenr/ -- not a write directory (R07 B5 set
# the split; R9 P2 moved the subtree under climate/ and gave it the engine's
# own name).
# weathergenr::generate_weather writes BOTH its diagnostic figures and its two
# date CSVs into a single out_dir; the R07 layout separates products from
# figures, so the split is done here -- on our side of the seam -- rather than
# by asking upstream for two output directories.
weathergen_root <- yaml$generateWeatherSeries$output.path
weathergen_output_path <- paste0(weathergen_root, "output/")
weathergen_plots_path <- paste0(weathergen_root, "plots/")

# Step 1) Read weather data from the netcdf file
message("[generate_weather] Reading weather netcdf: ", climate_nc_path)
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
message("[generate_weather] Resampling on ", length(keep), " basin cell(s) of ",
        length(ncdata$data), " in the store")
obs_data_basin <- ncdata$data[keep]
obs_grid_basin <- ncdata$grid[keep, , drop = FALSE]

# Step 2) Generate new weather realizations
message("[generate_weather] Generating ", historical_realizations_num,
        " weather realization(s)")
stochastic_weather <- weathergenr::generate_weather(
    # The BASIN subset: this call decides WHICH DAYS get resampled, and that
    # decision should reflect the basin's climate, not the buffer's. The full
    # grid is re-attached below, where the realizations are built.
    obs_data         = obs_data_basin,
    obs_grid         = obs_grid_basin,
    obs_dates        = ncdata$date,
    vars             = yaml$general$variables,
    n_years          = yaml$generateWeatherSeries$sim.year.num,
    start_year       = yaml$generateWeatherSeries$sim.year.start,
    year_start_month = yaml$generateWeatherSeries$month.start,
    n_realizations   = historical_realizations_num,
    warm_var         = yaml$generateWeatherSeries$warm.variable,
    warm_signif      = yaml$generateWeatherSeries$warm.signif.level,
    warm_pool_size   = yaml$generateWeatherSeries$warm.sample.num,
    annual_knn_n     = yaml$generateWeatherSeries$knn.sample.num,
    wet_q            = yaml$generateWeatherSeries$mc.wet.quantile,
    extreme_q        = yaml$generateWeatherSeries$mc.extreme.quantile,
    dry_spell_factor = yaml$generateWeatherSeries$dry.spell.change,
    wet_spell_factor = yaml$generateWeatherSeries$wet.spell.change,
    out_dir          = weathergen_output_path,
    seed             = yaml$generateWeatherSeries$seed,
    parallel         = yaml$generateWeatherSeries$compute.parallel,
    # C34. weathergenr 1.2.0 split evaluation into its own exports, so the
    # config's old `evaluate.model` reached NOTHING -- plot emission is
    # `save_plots`, which defaulted TRUE. Setting evaluate.model: FALSE
    # therefore did not stop the plots, which is what the key claimed to do.
    save_plots       = yaml$generateWeatherSeries$save.plots
)

# Step 2b) Move the generator's diagnostic figures into plots/. The two date
# CSVs (sim_dates.csv, resampled_dates.csv) are generator PRODUCTS and stay in
# output/, where generate_weather already wrote them. Each ggsave upstream sits
# in its own tryCatch, so a missing figure is a legitimate state, not an error.
weathergen_figures <- c("obs_power_spectra.png", "warm_annual_precip.png",
                        "warm_annual_stats.png", "warm_annual_wavelet.png")
dir.create(weathergen_plots_path, recursive = TRUE, showWarnings = FALSE)
for (fig in weathergen_figures) {
  src <- file.path(weathergen_output_path, fig)
  if (file.exists(src)) {
    file.rename(src, file.path(weathergen_plots_path, fig))
  }
}

# STEP 3) Save each stochastic realization back to a netcdf file
for (n in 1:historical_realizations_num) {

  message("[generate_weather] Saving realization ", n, " of ",
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
        calendar      = "noleap",
        template_path = climate_nc_path,
        compression   = 4,
        spatial_ref   = "spatial_ref",
        file_prefix   = yaml$generateWeatherSeries$nc.file.prefix,
        file_suffix   = paste0(pad(n, rlz_index_width), "_st_", st_baseline)
  )

  # Workaround (load-bearing): weathergenr::write_netcdf does NOT propagate
  # spatial_ref attributes from template_path to the output. Downstream
  # (impose_climate_change.R) uses the realization file as its own template and
  # needs `x_dim` / `y_dim` on its spatial_ref; without them it crashes with
  # "attempt to select less than one element". Copy them here from the
  # historical template.
  # REMOVAL CONDITION: drop this block only once tanerumit/weathergenr's
  # write_netcdf propagates spatial_ref (and its ncatt_get check asserts
  # hasatt=TRUE) — tracked in dev/tasks/ § R5. Removing it before the
  # upstream fix lands breaks the pipeline.
  # Match THIS realization only: all realizations now share one output dir, so
  # an index-free pattern would re-patch realization 1 on every iteration.
  rlz_files <- list.files(
    rlz_out_dir, pattern = paste0("_", pad(n, rlz_index_width), "_st_", st_baseline, "\\.nc$"), full.names = TRUE
  )
  if (length(rlz_files) >= 1) {
    src <- ncdf4::nc_open(climate_nc_path)
    dst <- ncdf4::nc_open(rlz_files[1], write = TRUE)
    src_atts <- ncdf4::ncatt_get(src, "spatial_ref")
    for (an in names(src_atts)) {
      try(
        ncdf4::ncatt_put(dst, "spatial_ref", an, src_atts[[an]]),
        silent = TRUE
      )
    }
    ncdf4::nc_close(src)
    ncdf4::nc_close(dst)
  }

}
