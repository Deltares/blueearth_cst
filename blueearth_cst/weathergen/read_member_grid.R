#################### THE R SIDE OF WG-2 ########################################
#
# One member's twelve monthly rows, sliced out of the experiment's stress-test
# lookup. This is the consumer half of WG-2, so it exists as a named counterpart
# to the Python producer -- one function per language for one contract.
#
# WHY THIS IS ITS OWN FILE. Before the lookup, rule 3.12 read a file that WAS
# the member: twelve rows, no id, and Snakemake guaranteed its existence with a
# loud structural MissingInputException. After it, the same values arrive
# through a filter-and-order join, and a join that matches nothing yields a
# ZERO-LENGTH vector while one that matches partially yields a short one. R's
# recycling rules then make a silent wrong answer at least as likely as an
# error. So the migration converts a structural failure into a quiet data
# condition inside a script that had no guard -- and this is that guard.
#
# Extracted rather than inlined so the guard has an executable falsifier. Inside
# impose_climate_change.R the malformed-input path is reachable only after the
# weathergen YAML is read and a realization netCDF is loaded through
# weathergenr, so a negative test would have to stand up the whole chain to
# assert a stop() -- and a WF3 run on a VALID config is green whether the guard
# exists or not. Here the falsifier is one `Rscript -e 'source(...)'` per
# fixture.
#
# SOURCES NOTHING -- not even global.R, which is options-only. A negative test
# therefore needs neither weathergenr nor a netCDF.

read_member_grid <- function(lookup_path, st_id_token) {

  # `st_id` is zero-padded TEXT whose width is the member filename's (C27), so
  # the two are ONE token. Read as anything else and `01` becomes `1`, the
  # comparison below matches nothing, and the failure presents as a missing
  # member rather than as a type error.
  lookup <- utils::read.csv(
    lookup_path,
    colClasses = c(st_id = "character")
  )

  required <- c("st_id", "month", "temp_change", "precip_change",
                "precip_variance_change")
  missing_cols <- setdiff(required, names(lookup))
  if (length(missing_cols) > 0L) {
    stop("read_member_grid: '", lookup_path, "' is missing the WG-2 column(s) ",
         paste(missing_cols, collapse = ", "),
         ". Expected header: ", paste(required, collapse = ","),
         call. = FALSE)
  }

  grid <- lookup[lookup$st_id == st_id_token, , drop = FALSE]
  grid <- grid[order(grid$month), , drop = FALSE]
  rownames(grid) <- NULL

  # The postcondition, at the point of use. Ordering happens BEFORE the check on
  # purpose: unordered months are normalised, not rejected -- the lookup's sort
  # order is pinned by WG-2, but a consumer that re-emits it unsorted is
  # producing the same twelve values and this function's job is to hand the
  # caller a month-ordered frame, not to police upstream formatting.
  if (nrow(grid) != 12L || !identical(as.integer(grid$month), 1:12)) {
    stop("read_member_grid: member '", st_id_token, "' does not resolve to the ",
         "twelve calendar months in '", lookup_path, "' -- got ", nrow(grid),
         " row(s) with month(s) [",
         paste(grid$month, collapse = ", "), "]. ",
         "The member token must match an `st_id` in the lookup EXACTLY, ",
         "including its zero padding.",
         call. = FALSE)
  }

  grid
}
