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
if (length(args) != 2L) {
  stop("generate_weather.R expects 2 args: <climate_nc> <weagen_config_yaml>")
}
climate_nc_path    <- args[[1]]
weagen_config_path <- args[[2]]

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

# Step 2) Generate new weather realizations
message("[generate_weather] Generating ", historical_realizations_num,
        " weather realization(s)")
stochastic_weather <- weathergenr::generate_weather(
    obs_data         = ncdata$data,
    obs_grid         = ncdata$grid,
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
    parallel         = yaml$generateWeatherSeries$compute.parallel
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

  # Obtain stochastic series by re-ordering historical data
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
        file_suffix   = paste0(n, "_st_0")
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
    rlz_out_dir, pattern = paste0("_", n, "_st_0\\.nc$"), full.names = TRUE
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
