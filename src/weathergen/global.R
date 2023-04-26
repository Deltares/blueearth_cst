#################### GLOBAL R SETTINGS #########################################

# Set library location (prevents errors if another R.exe is already installed in the system)
.libPaths(paste0(Sys.getenv("R_HOME"),"/library"))

# General options
options(warn = -1) # Disable warnings

# Disable S3 method overwritten message
Sys.setenv(`_R_S3_METHOD_REGISTRATION_NOTE_OVERWRITES_` = "false")
