
#Use reticulate package to run python code from R

# Specify the virtual python environment
Sys.setenv(RETICULATE_PYTHON = "C:/Users/taner/Anaconda3/envs/blueearth-cst")
reticulate::use_condaenv(condaenv = "C:/Users/taner/Anaconda3/envs/blueearth-cst",
  required = TRUE)
