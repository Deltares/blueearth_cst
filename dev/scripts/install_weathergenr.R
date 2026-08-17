# Install weathergenr v2.0.0 from GitHub via remotes::install_github.
#
# Invoked at env-setup time by:
#     pixi run install-rdeps
# (also pulled in transitively by `pixi run install`).
#
# Why remotes and not pak:
#   conda-forge's r-pak is dysfunctional on win-64 — loading it warns
#   "Wrong OS or architecture, pak is probably dysfunctional" and
#   pkg_install() dies with "object 'clic_start_thread' not found". pak
#   bought us nothing here anyway: weathergenr is a pure-R package
#   (no src/, NeedsCompilation: no) and all 16 of its Imports are already
#   present as conda binaries in the pixi env, so there is no binary
#   resolution or compilation to do — remotes just fetches the tarball
#   and copies the R sources into the library.
#
# Where it lands:
#   .libPaths()[1], which under pixi is the env's R site-lib
#   (.pixi/envs/<env>/lib/R/library) on both win-64 and linux-64. There
#   is no separate user-level R lib on the path under `Rscript --vanilla`.
#
# Why dependencies = FALSE / upgrade = "never":
#   every Import is already installed and conda-managed. Installing or
#   upgrading anything from CRAN here would shadow a conda binary with a
#   source build against a different toolchain — the exact ABI mismatch
#   ("Mingw-w64 ... 32 bit pseudo relocation out of range") we want to
#   avoid. So we touch nothing but weathergenr itself.
#
# Idempotent: returns early if weathergenr at the right version is
# already present anywhere visible to R.
#
# The version is written ONCE, below, and the GitHub ref is derived from it.
# It used to be spelled twice — here and in the `@v<x>` tag — which is one
# edit away from installing a version the check then rejects on every run.

required_version <- "2.0.0"

# Read from the library's DESCRIPTION rather than via packageVersion(), which
# answers from the namespace loaded at the top of this script and would report
# the version we just replaced.
installed_version <- function() {
  path <- tryCatch(
    find.package("weathergenr", quiet = TRUE),
    error = function(e) character()
  )
  if (!length(path)) {
    return(NA_character_)
  }
  desc <- file.path(path[[1]], "DESCRIPTION")
  if (!file.exists(desc)) {
    return(NA_character_)
  }
  as.character(read.dcf(desc, fields = "Version")[1, 1])
}

if (requireNamespace("weathergenr", quietly = TRUE)) {
  found_version <- as.character(packageVersion("weathergenr"))
  if (found_version == required_version) {
    cat("weathergenr ", found_version, " already installed at\n", sep = "")
    cat("  ", find.package("weathergenr"), "\n", sep = "")
    quit(save = "no", status = 0)
  }
  cat(
    "weathergenr ", found_version, " present but ", required_version,
    " required; reinstalling.\n", sep = ""
  )
}

if (!requireNamespace("remotes", quietly = TRUE)) {
  stop("r-remotes is missing from the env. Declare it in pixi.toml [dependencies].")
}

remotes::install_github(
  paste0("Deltares-research/weathergenr@v", required_version),
  dependencies = FALSE,   # all Imports are already conda-managed in the env
  upgrade = "never",      # never shadow a conda binary with a CRAN source build
  build_vignettes = FALSE,
  force = TRUE            # reached only when a wrong version is present (see above)
)

# Verify, because install_github does not always fail the script when it fails
# to install. A transient GitHub 503 on the tarball download (seen 2026-08-17
# during the 1.2.0 -> 2.0.0 upgrade) printed a download error, left the OLD
# version in place, and still exited 0 — so `pixi run install` reported success
# and the next WF3 run would have used the superseded package.
final_version <- installed_version()
if (!identical(final_version, required_version)) {
  stop(
    "weathergenr ", required_version, " was NOT installed (library holds: ",
    if (is.na(final_version)) "nothing" else final_version,
    "). Check the download error above and re-run `pixi run install-rdeps`; ",
    "a GitHub outage is the usual cause and a retry is the usual fix.",
    call. = FALSE
  )
}

cat("--- weathergenr installed via remotes ---\n")
cat("Version:    ", final_version, "\n", sep = "")
cat("Location:   ", find.package("weathergenr"), "\n", sep = "")
