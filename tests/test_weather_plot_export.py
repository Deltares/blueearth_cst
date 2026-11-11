"""Regression for shared-axis conditional weather diagnostic exports."""

import shutil
import subprocess
from pathlib import Path

import pytest


def _weathergenr_available() -> bool:
    """Whether Rscript can load weathergenr, which the plot export calls."""
    if shutil.which("Rscript") is None:
        return False
    probe = "quit(status=as.integer(!requireNamespace('weathergenr', quietly=TRUE)))"
    result = subprocess.run(
        ["Rscript", "--vanilla", "-e", probe], capture_output=True, timeout=60
    )
    return result.returncode == 0


@pytest.mark.skipif(
    not _weathergenr_available(), reason="Rscript with weathergenr is required"
)
def test_conditional_grid_export_preserves_shared_axes(tmp_path):
    script = r"""
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
"""
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


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript is required")
def test_time_series_styling_builds_and_leaves_the_source_plot(tmp_path):
    """ggproto(NULL, layer) copies recursed without bound under ggplot2 4."""
    script = r"""
library(ggplot2)
source("blueearth_cst/weathergen/plot_style.R")
style <- yaml::read_yaml("config/plotting/style.yml")
tokens <- style$charts$time_series
dat <- data.frame(year=rep(2000:2009, 2), value=c(1:10, 2:11),
  series=rep(c("observed", "simulated"), each=10))
p <- ggplot(dat, aes(year, value, group=series)) +
  geom_line(data=dat[dat$series == "observed", ], colour="black") +
  geom_line(data=dat[dat$series == "simulated", ], colour="grey50",
    linetype="dashed") + geom_point()
before <- lapply(p$layers, function(layer) layer$aes_params)
styled <- style_cst_time_series(p, tokens, "black")
built <- ggplot_build(styled)
stopifnot(length(built$data) == 3L)
stopifnot(identical(lapply(p$layers, function(layer) layer$aes_params), before))
pt <- 25.4 / 72.27
stopifnot(styled$layers[[1]]$aes_params$linewidth == tokens$observed_width * pt,
  styled$layers[[2]]$aes_params$linetype == "solid")
ggsave(file.path(commandArgs(trailingOnly=TRUE)[[1]], "styled.png"), styled,
  width=80, height=50, units="mm", dpi=72)
"""
    source = tmp_path / "time_series.R"
    source.write_text(script, encoding="utf-8")
    result = subprocess.run(
        ["Rscript", "--vanilla", str(source), str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
