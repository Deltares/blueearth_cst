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
    temporal <- id %in% c("annual_precip", "warm_annual_precip", "monthly_cycle")
    if (temporal) {
      effective$colors[["observed"]] <- yaml::read_yaml(
        file.path(root, "colors.yml"))$neutral$time_series_observed
    }
    if (id == "obs_power_spectra") {
      effective$colors[["observed"]] <- weather$colors$wavelet_observed
    }
    if (id == "precip_cond_cor") {
      # The native facet_grid shares x scales by variable and y scales by regime.
      # weathergenr 2.1.0's panel arranger retrains each panel independently,
      # then fails its axis-range assertion. Preserve the grid and its scales.
      p <- plots[[id]] + effective$theme
    } else {
      p <- weathergenr::style_weather_plot(plots[[id]], id,
        plot_style = effective, panel_layout = layout)
    }
    if (temporal) {
      p <- style_cst_time_series(p, style$charts$time_series,
        effective$colors[["observed"]], root)
    }
    stem <- if (id %in% names(filenames)) filenames[[id]] else id
    faceted <- !is.null(attr(p, "weathergenr_panel_layout"))
    height <- style$charts$heights$single
    if (id == "precip_cond_cor") {
      panels <- ggplot2::ggplot_build(p)$layout$layout
      height <- length(unique(panels$ROW)) * layout$panel_height_mm
    }
    weathergenr::export_weather_plot(p,
      file.path(output_dir, paste0(stem, ".png")),
      width = if (faceted) NULL else style$page$figure_width_mm,
      height = if (faceted) NULL else height,
      dpi = style$page$raster_dpi)
    plots[[id]] <- p
  }
  invisible(plots)
}

# Apply the shared template after native styling; keep ensemble opacity and data.
style_cst_time_series <- function(plot, tokens, observed_color,
                                  root = "config/plotting") {
  pt_to_mm <- 25.4 / 72.27
  if (inherits(plot, "patchwork")) {
    plot$patches$plots <- lapply(plot$patches$plots, function(panel) {
      style_cst_time_series(panel, tokens, observed_color, root)
    })
  }
  for (i in seq_along(plot$layers)) {
    # Copy the layer so the source plot keeps its styling. A ggproto(NULL, layer)
    # child recurses without bound when ggplot2 4 (S7) builds the plot.
    source_layer <- plot$layers[[i]]
    layer <- rlang::env_clone(source_layer)
    class(layer) <- class(source_layer)
    if (inherits(layer$geom, "GeomLine")) {
      observed <- identical(layer$aes_params$colour, observed_color)
      layer$mapping$linetype <- NULL
      layer$aes_params$linetype <- "solid"
      layer$aes_params$linewidth <- pt_to_mm * if (observed) {
        tokens$observed_width
      } else {
        tokens$line_width
      }
    } else if (inherits(layer$geom, "GeomPoint")) {
      layer$aes_params$size <- tokens$marker_size * pt_to_mm
    }
    plot$layers[[i]] <- layer
  }
  colors <- yaml::read_yaml(file.path(root, "colors.yml"))$neutral
  plot + ggplot2::theme(
    panel.grid.major.x = ggplot2::element_blank(),
    panel.grid.minor = ggplot2::element_blank(),
    panel.grid.major.y = ggplot2::element_line(
      colour = grDevices::adjustcolor(colors$time_series_grid,
        alpha.f = tokens$grid_alpha),
      linewidth = tokens$grid_width * pt_to_mm)
  )
}
