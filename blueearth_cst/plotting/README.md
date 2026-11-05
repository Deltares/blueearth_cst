# Plotting code

This directory contains Matplotlib drawing and export helpers used by BlueEarth CST. Developers
building a figure should prepare scientific data in its owning workflow, select a style, draw with
the appropriate helper and let that workflow own captions, output paths and interpretation.

## Find the right component

| Need | Source |
|---|---|
| Active style and neutral defaults | [preset.py](preset.py) |
| Common axes, marks and labels | [fundamentals.py](fundamentals.py) |
| Annual series and monthly bands from prepared arrays | [prepared_charts.py](prepared_charts.py) |
| Fixed physical grid geometry | [panel_layout.py](panel_layout.py) and the [layout contract][layout] |
| Save a supplied figure without workflow logging | [export.py](export.py) |

These helpers separate drawing from scientific preparation. `prepared_maps.py` draws the CST map
template from supplied fields and features; `location_labels.py` supports its climate and spatial
map labels. `figure_titles.py` and `figure_captions.py` write CST figure inventories. Their
workflow-specific conventions should not be mistaken for a general plotting package API.

## CST integration and checks

[`shared/plot_preset.py`](../shared/plot_preset.py) builds the CST preset from tracked
[`style.yml`](../../config/plotting/style.yml) and [`colors.yml`](../../config/plotting/colors.yml).
Figure families own their scientific preparation, preset scope, export and captions. The
[plot style reference](../../dev/reference/toolbox-plot-style.md) describes current CST appearance;
the [weather profile](../../dev/reference/weather-diagnostic-style.md) describes R weather plots
and their use of the panel contract.

For a new fixed-size grid, read the [layout contract](docs/panel-layout.md) and the
[Python layout tests](../../tests/test_panel_layout.py), including the shared
[planner cases](../../tests/fixtures/panel_layout.csv). These files document the implemented Python
behavior and comparison expectations; they do not provide a standalone generic ggplot2 adapter.

[layout]: docs/panel-layout.md
