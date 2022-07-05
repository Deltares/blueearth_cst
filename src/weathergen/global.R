
#################### GLOBAL R SETTINGS #########################################

# General options
options(warn = -1) # Disable warnings

# Disable S3 method overwritten message
Sys.setenv(`_R_S3_METHOD_REGISTRATION_NOTE_OVERWRITES_` = "false")

# Install Rlang package from the correct repo
install.packages("rlang", repos = "http://cran.rstudio.com", dependencies = TRUE)

# Install weathergenr package, but not don't update the dependencies
devtools::install_github("Deltares/weathergenr", upgrade = "never")
