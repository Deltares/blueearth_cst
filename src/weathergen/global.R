

# General options
options(warn = -1) # Disable warnings
Sys.setenv(`_R_S3_METHOD_REGISTRATION_NOTE_OVERWRITES_` = "false") # Disable S3 method overwritten message

# Install and load libraries
devtools::install_github("Deltares/weathergenr", ask = "never")
library(weathergenr)
