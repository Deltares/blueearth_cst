################################################################################
#################### PURE HELPERS FOR THE WEATHERGEN RULES #####################
#
# The decisions generate_weather.R and impose_climate_change.R make on their own
# side of the weathergenr seam: which cells the generator resamples on, where
# its products land, how a member's percent changes become multipliers, and how
# the declared output name splits into weathergenr's <prefix>_<suffix>.
#
# Extracted for the same reason read_member_grid.R was: inline, each is
# reachable only after weathergenr has loaded a netCDF, so a defect surfaced
# only minutes into a WF3 run. Two of them did ship that way -- the basin-subset
# id mismatch (rule 3.11, 2026-08-13) and the trailing-slash out_dir
# (2026-09-05). Here each is one `Rscript -e 'source(...)'` away from a test
# (tests/test_weathergen_member_helpers.py).
#
# SOURCES NOTHING and calls no weathergenr function, so a test needs neither.

#' The generator subtree's output/ and evaluation/plots/ directories.
#'
#' No trailing slash survives: weathergenr joins with file.path(), so a slash
#' on the root landed as `output//rlz_1_st_0.nc` on every path row it printed.
weathergen_paths <- function(out_dir) {
  root <- sub("/+$", "", out_dir)
  list(
    output = file.path(root, "output"),
    plots = file.path(root, "evaluation", "plots")
  )
}

#' Restrict the store to the cells the basin touches, ids renumbered 1..n.
#'
#' weathergenr picks which years to resample from an unweighted mean of every
#' cell it is handed, so it must be handed the basin's cells only. The mask is a
#' FILTER, not a weighting (owner ruling 2026-08-10). Coordinates are matched,
#' never indices: both sides enumerate the grid their own way.
#'
#' `read_netcdf` stamps `id = seq_len(nrow(grid))` across the FULL store grid,
#' so a row subset keeps the original ids -- cells 7 and 12 of 20 stay 7 and 12
#' -- while `data[keep]` is renumbered 1..2. run_weather_generator's evaluation
#' pass derives `grid_ids` from that column and requires them to index
#' `obs_data`; before the renumbering, rule 3.11 failed with "'grid_ids' must
#' match names or indices of obs_data" on every basin smaller than its store.
#'
#' The postcondition is asserted here because weathergenr states it only as a
#' failure deep in the evaluation pass, after generation has already run.
#'
#' @param ncdata `read_netcdf`'s list: `data` (per-cell frames) and `grid`
#'   (with `x`, `y` and optionally `id`).
#' @param basin_cells data.frame with `latitude` and `longitude`.
#' @return list(data =, grid =, keep =) -- `keep` indexes the full store.
subset_basin_cells <- function(ncdata, basin_cells, source_label = "the store") {
  grid_key <- paste(round(ncdata$grid$y, 6), round(ncdata$grid$x, 6))
  mask_key <- paste(round(basin_cells$latitude, 6), round(basin_cells$longitude, 6))
  keep <- which(grid_key %in% mask_key)
  if (length(keep) == 0L) {
    stop("basin_cells.csv matched no cell in ", source_label,
         " -- the mask and the store disagree about the grid")
  }
  data <- ncdata$data[keep]
  grid <- ncdata$grid[keep, , drop = FALSE]
  if ("id" %in% names(grid)) {
    grid$id <- seq_len(nrow(grid))
  }

  if (nrow(grid) != length(data)) {
    stop("basin subset is inconsistent: ", nrow(grid),
         " grid rows vs ", length(data), " data cells")
  }
  if ("id" %in% names(grid) &&
      !identical(as.integer(grid$id), seq_along(data))) {
    stop("basin subset grid ids do not index its data list: ids are ",
         paste(utils::head(grid$id, 5), collapse = ", "),
         " for ", length(data), " cells")
  }
  list(data = data, grid = grid, keep = keep)
}

#' The twelve zero-change rows of the unperturbed `identity` baseline.
#'
#' The baseline has NO lookup row -- st_0 is absent from the parameter grid on
#' purpose -- so its rows are synthesized here and it then takes the SAME
#' perturbation path as every grid member (t2608151154): skipping it moved
#' low-flow indicators by -70% to +128% through a processing step alone.
identity_member_grid <- function() {
  data.frame(
    month = 1:12, temp_change = 0, precip_change = 0,
    precip_variance_change = 0
  )
}

#' PERCENT -> the generator's multiplier form (WG-2).
#'
#' `1 + p/100`, not `(100 + p)/100`: over 200k random float32 multipliers the
#' first fails to reproduce the level in 19.9% of cases and the second in 32.9%.
#' Within one float64 ulp for every multiplier >= 0.5, which the producer
#' refuses to go below (D35). Applied to the variance column too: omitting it
#' would hand apply_climate_perturbations a factor of ZERO, not the identity 1.0.
percent_to_factor <- function(percent) {
  1 + percent / 100
}

#' Split a declared output name into weathergenr's <prefix>_<suffix>.
#'
#' write_netcdf composes `<prefix>_<suffix>.nc`, so the stem splits at its LAST
#' underscore. Deriving from the Snakemake output declaration keeps one source
#' of truth and is naming-agnostic (rlz_1_cst_2 and rlz_1_st_2 both split).
split_output_stem <- function(output_nc_path) {
  stem <- sub("\\.nc$", "", basename(output_nc_path))
  if (!grepl("_", stem, fixed = TRUE)) {
    stop("cannot split '", stem, "' into weathergenr's <prefix>_<suffix>: ",
         "the declared output name carries no underscore")
  }
  list(
    stem = stem,
    prefix = sub("_[^_]+$", "", stem),
    suffix = sub("^.*_", "", stem)
  )
}

#' One change column for a log row: a single value, or its range.
#'
#' Every shipped grid holds one value for all twelve months, but the lookup
#' schema permits monthly variation, and printing `[1]` of a varying vector
#' would be a confidently wrong row.
fmt_change <- function(values, unit) {
  lo <- min(values)
  hi <- max(values)
  if (isTRUE(all.equal(lo, hi))) {
    sprintf("%+.1f%s", lo, unit)
  } else {
    sprintf("%+.1f..%+.1f%s", lo, hi, unit)
  }
}

#' The perturbation summary a 3.12 job prints: temp, precip, and the variance
#' change only where it actually varies the run (flat 0.0 on every shipped
#' config, where naming it would be a constant column).
format_perturbation <- function(member_grid) {
  text <- paste0(
    "temp ", fmt_change(member_grid$temp_change, " degC"),
    ", precip ", fmt_change(member_grid$precip_change, "%")
  )
  if (any(member_grid$precip_variance_change != 0)) {
    text <- paste0(
      text, ", precip var ",
      fmt_change(member_grid$precip_variance_change, "%")
    )
  }
  text
}
