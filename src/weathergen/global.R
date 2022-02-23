

# General options
options(warn = -1) # Disable warnings

# Disable S3 method overwritten message
#Sys.setenv(`_R_S3_METHOD_REGISTRATION_NOTE_OVERWRITES_` = "false")

# Install and load libraries
devtools::install_github("Deltares/weathergenr", ask = "never")
library(weathergenr)
