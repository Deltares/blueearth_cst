"""Regression for shared-axis conditional weather diagnostic exports."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript is required")
def test_conditional_grid_export_preserves_shared_axes(tmp_path):
    script = r'''
library(ggplot2)
source("blueearth_cst/weathergen/plot_style.R")
dat <- expand.grid(variable=c("precip:temp", "precip:press"),
  regime=c("all", "wet", "dry"), id=1:5, rlz=1:2)
dat$Observed <- seq_len(nrow(dat))/nrow(dat)
dat$Simulated <- dat$Observed + ifelse(dat$regime == "dry", 0.3, -0.1)
p <- ggplot(dat, aes(Observed, Simulated)) + geom_abline() +
  geom_point(alpha=0) +
  stat_summary(geom="linerange", fun.min=min, fun.max=max) +
  stat_summary(geom="point", fun=median) +
  facet_grid(regime~variable, scales="free")
run <- list(gen_output=list(plots=list(precip_cond_cor=p)), evaluation=list())
out <- commandArgs(trailingOnly=TRUE)[[1]]
result <- export_cst_weather_plots(run, out)$precip_cond_cor
old <- ggplot_build(p)
new <- ggplot_build(result)
stopifnot(identical(old$data, new$data), nrow(new$layout$layout) == 6L)
for (i in seq_along(old$layout$panel_params)) {
  for (axis in c("x.range", "y.range")) {
    stopifnot(identical(old$layout$panel_params[[i]][[axis]],
      new$layout$panel_params[[i]][[axis]]))
  }
}
stopifnot(file.info(file.path(out, "precip_conditional_correlations.png"))$size > 0)
'''
    source = tmp_path / "conditional.R"
    source.write_text(script, encoding="utf-8")
    result = subprocess.run(
        ["Rscript", "--vanilla", str(source), str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
