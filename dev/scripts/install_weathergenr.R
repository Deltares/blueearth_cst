# Install the verified weathergenr commit via remotes::install_github.
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
# Idempotent: returns early only when both version and GitHub commit match.
#
# The v2.1.0 tag is not published. Pin the verified, published implementation
# by immutable commit rather than depending on a moving branch or missing tag.

required_version <- "2.1.0"
required_sha <- "b023e65e5237f681cb96178754488bba84afb2ae"

# Read DESCRIPTION without loading the package, so post-install verification
# cannot report metadata from a previously loaded namespace.
installed_metadata <- function() {
  path <- tryCatch(
    find.package("weathergenr", quiet = TRUE),
    error = function(e) character()
  )
  if (!length(path)) {
    return(c(Version = NA_character_, RemoteSha = NA_character_))
  }
  desc <- file.path(path[[1]], "DESCRIPTION")
  if (!file.exists(desc)) {
    return(c(Version = NA_character_, RemoteSha = NA_character_))
  }
  read.dcf(desc, fields = c("Version", "RemoteSha"))[1, ]
}

found <- installed_metadata()
if (identical(unname(found[["Version"]]), required_version) &&
      identical(unname(found[["RemoteSha"]]), required_sha)) {
  cat("weathergenr ", required_version, " at ", required_sha,
      " already installed at\n", sep = "")
  cat("  ", find.package("weathergenr"), "\n", sep = "")
  quit(save = "no", status = 0)
}
if (!is.na(found[["Version"]])) {
  cat(
    "weathergenr ", found[["Version"]], " present, but version ",
    required_version, " at ", required_sha, " is required; reinstalling.\n",
    sep = ""
  )
}

if (!requireNamespace("remotes", quietly = TRUE)) {
  stop(
    "r-remotes is missing from the env. Declare it in pixi.toml [dependencies]."
  )
}

remotes::install_github(
  paste0("Deltares-research/weathergenr@", required_sha),
  dependencies = FALSE,   # all Imports are already conda-managed in the env
  upgrade = "never",      # never shadow a conda binary with a CRAN source build
  build_vignettes = FALSE,
  force = TRUE            # reached only when version or source differs
)

# Verify, because install_github does not always fail the script when it fails
# to install. A transient GitHub 503 on the tarball download (seen 2026-08-17
# during the 1.2.0 -> 2.0.0 upgrade) printed a download error, left the OLD
# version in place, and still exited 0 — so `pixi run install` reported success
# and the next WF3 run would have used the superseded package.
final <- installed_metadata()
if (!identical(unname(final[["Version"]]), required_version) ||
      !identical(unname(final[["RemoteSha"]]), required_sha)) {
  stop(
    "weathergenr ", required_version, " at ", required_sha,
    " was NOT installed (library version: ", final[["Version"]],
    "; commit: ", final[["RemoteSha"]],
    "). Check the download error above and re-run `pixi run install-rdeps`; ",
    "a GitHub outage is the usual cause and a retry is the usual fix.",
    call. = FALSE
  )
}

cat("--- weathergenr installed via remotes ---\n")
cat("Version:    ", final[["Version"]], "\n", sep = "")
cat("Commit:     ", final[["RemoteSha"]], "\n", sep = "")
cat("Location:   ", find.package("weathergenr"), "\n", sep = "")
