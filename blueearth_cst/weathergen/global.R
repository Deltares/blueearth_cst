#################### GLOBAL R SETTINGS AND LOG GRAMMAR #########################

# Trust R's default .libPaths(): user lib first (where weathergenr +
# its native-code dependencies are installed against the system
# toolchain), then the conda site lib. Forcing conda site lib first
# breaks weathergenr's load on Windows because its imports resolve
# from a conda r-base build with an incompatible mingw runtime ABI.
# R3 followup: build weathergenr against the conda toolchain so the
# user-lib dependency goes away.

# General options
options(warn = -1) # Disable warnings

# Disable S3 method overwritten message
Sys.setenv(`_R_S3_METHOD_REGISTRATION_NOTE_OVERWRITES_` = "false")


#################### LOG GRAMMAR ###############################################
#
# The R side of `snake_utils.log_row`. One row is
#
#     HH:MM:SS - <module> - <message>
#
# with the level shown only when it is not INFO -- the same four fields
# `_log_row_text` assembles for every Python `script:` rule and every compacted
# hydromt record, so an R rule's own messages sit uniformly among the library
# lines instead of as bare, timestamp-less `[tag] ...` text. These rules run
# through `run_logged.py`, whose tee compacts hydromt records and relativizes
# project paths but does NOT stamp anything: a line that arrives unstamped stays
# unstamped, which is why the stamp is emitted here rather than left to it.
#
# Three emitters with three spellings of one grammar is how the grammar stops
# being one, so this mirrors the Python function's contract rather than
# approximating it -- including the `CST_LOG_LEVEL` floor, which would otherwise
# silence a WF3 run's Python rows and leave its R ones talking.

#: Mirrors `snake_utils._LOG_LEVEL_RANK`.
CST_LOG_LEVEL_RANK <- c(
  DEBUG = 10L, INFO = 20L, WARNING = 30L, WARN = 30L, ERROR = 40L, CRITICAL = 50L
)

#' The minimum rank `log_row` will emit, from `CST_LOG_LEVEL`.
#'
#' Unset or unrecognized means DEBUG -- emit everything, the behaviour every
#' caller had before the floor existed. Read per call, not cached, so the
#' variable can be set mid-session.
log_level_floor <- function() {
  rank <- unname(CST_LOG_LEVEL_RANK[toupper(trimws(Sys.getenv("CST_LOG_LEVEL")))])
  if (is.na(rank)) 10L else rank
}

#' Emit one log row in the toolbox's standard compact format.
#'
#' `...` is pasted together exactly as `message()` concatenates its arguments,
#' so a multi-part call reads the same as the one it replaces. It comes FIRST on
#' purpose: R matches arguments after `...` by exact name only, so a positional
#' `log_row("read ", n, " cells")` cannot silently bind `n` to `module` and
#' render a mangled row.
#'
#' Rows below `CST_LOG_LEVEL` are dropped; an unrecognized `level` is never
#' suppressed, because a filter that swallowed an unfamiliar level would hide
#' exactly the unusual row worth seeing.
#'
#' Written to stderr via `message()`, like the calls it replaces: `run_logged.py`
#' merges the child's stdout and stderr, and R buffers the two differently, so
#' keeping one stream keeps the rows in the order they were emitted.
log_row <- function(..., module = "weathergen", level = "INFO") {
  level_text <- toupper(trimws(level))
  rank <- unname(CST_LOG_LEVEL_RANK[level_text])
  if (!is.na(rank) && rank < log_level_floor()) {
    return(invisible(NULL))
  }
  stamp <- format(Sys.time(), "%H:%M:%S")
  body <- paste0(...)
  if (level_text == "INFO") {
    message(stamp, " - ", module, " - ", body)
  } else {
    message(stamp, " - ", module, " - ", level_text, " - ", body)
  }
  invisible(NULL)
}
