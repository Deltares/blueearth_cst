# Install Rlang package from the correct repo
install.packages("rlang", repos = "http://cran.rstudio.com", dependencies = TRUE)

# Install weathergenr package, but not don't update the dependencies
devtools::install_github("Deltares/weathergenr", ref="wg-server", upgrade = "never")
