# Translate presentation tokens after scientific identity resolution.
read_cst_weather_style <- function(root = "config/plotting") {
  style <- yaml::read_yaml(file.path(root, "style.yml"))
  colors <- yaml::read_yaml(file.path(root, "colors.yml"))
  weather <- yaml::read_yaml(file.path(root, "weathergenr.yml"))
  fonts <- style$charts$fonts
  list(
    theme = ggplot2::theme(
      text = ggplot2::element_text(size = fonts$label,
        family = style$charts$rcparams[["font.family"]]),
      axis.title = ggplot2::element_text(size = fonts$label,
        colour = colors$neutral$ink),
      axis.text = ggplot2::element_text(size = fonts$tick,
        colour = colors$neutral$muted),
      strip.text = ggplot2::element_text(size = fonts$label),
      plot.title = ggplot2::element_blank(),
      plot.subtitle = ggplot2::element_blank(),
      legend.text = ggplot2::element_text(size = fonts$note),
      panel.grid.major = ggplot2::element_line(colour = colors$neutral$grid),
      panel.grid.major.y = ggplot2::element_line(colour = colors$neutral$grid),
      panel.grid.minor = ggplot2::element_blank(),
      axis.ticks = ggplot2::element_line(colour = colors$neutral$axis)
    ),
    colors = c(observed = colors$default_cycle[[colors$evaluation$observed]],
      simulated = weather$colors$simulated, reference = weather$colors$reference)
  )
}

export_cst_weather_plots <- function(run, output_dir, root = "config/plotting") {
  style <- yaml::read_yaml(file.path(root, "style.yml"))
  weather <- yaml::read_yaml(file.path(root, "weathergenr.yml"))
  plot_style <- read_cst_weather_style(root)
  plots <- c(run$gen_output$plots, unclass(run$evaluation))
  if (length(plots) == 0L || is.null(names(plots)) || anyDuplicated(names(plots))) {
    stop("Weather diagnostics must have nonempty, unique plot IDs.")
  }
  filenames <- c(crossgrid = "crossgrid_correlations",
    intergrid = "intergrid_correlations",
    precip_cond_cor = "precip_conditional_correlations")
  layout <- weather$panel_layout
  layout$preferred_columns <- as.integer(unlist(layout$preferred_columns))
  layout$dpi <- style$page$raster_dpi
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  for (id in names(plots)) {
    effective <- plot_style
    if (id == "obs_power_spectra") {
      effective$colors[["observed"]] <- weather$colors$wavelet_observed
    }
    p <- weathergenr::style_weather_plot(plots[[id]], id,
      plot_style = effective, panel_layout = layout)
    stem <- if (id %in% names(filenames)) filenames[[id]] else id
    faceted <- !is.null(attr(p, "weathergenr_panel_layout"))
    weathergenr::export_weather_plot(p,
      file.path(output_dir, paste0(stem, ".png")),
      width = if (faceted) NULL else style$page$figure_width_mm,
      height = if (faceted) NULL else style$charts$heights$single,
      dpi = style$page$raster_dpi)
    plots[[id]] <- p
  }
  invisible(plots)
}
