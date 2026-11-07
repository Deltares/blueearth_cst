# Plotting code

This directory contains Matplotlib drawing and export helpers used by BlueEarth CST. Prepare scientific
data in its owning workflow, then supply it to a drawing helper. The workflow owns aggregation, figure
meaning, captions and output paths. This module tree contains both reusable helpers and CST-specific
figure code; it is not a separately distributed plotting package.

## Find the right component

| Need | Source |
|---|---|
| Active style and neutral defaults | [preset.py](preset.py) |
| Common axes, marks and labels | [fundamentals.py](fundamentals.py) |
| Annual series and monthly bands from prepared arrays | [prepared_charts.py](prepared_charts.py) |
| Fixed physical grid geometry | [panel_layout.py](panel_layout.py) and the [layout contract][layout] |
| Shared filename rules and workflow profiles | [Figure naming conventions](docs/figure-naming.md) |
| Save a supplied figure without workflow logging | [export.py](export.py) |

These helpers separate drawing from scientific preparation. [prepared_maps.py](prepared_maps.py) draws
the CST map template from supplied fields and features; [location_labels.py](location_labels.py)
supports its climate and spatial labels. [figure_titles.py](figure_titles.py) and
[figure_captions.py](figure_captions.py) write CST figure inventories. Those modules carry workflow
conventions and are not general plotting APIs.

`prepared_maps.draw_raster_panels(rasters, headings, frame, scale, overlays)` composes native raster
maps with the same coordinates, geographic aspect, overlays and scale bars as `draw_raster_map`.
Supply a common resolved `MapScale`; drawing does not calculate pooled limits. Panels fill row by row
in up to three columns, with one shared key and optional within-frame unavailable notes. Each map fits
a 65 × 50 mm envelope by default; measured labels and keys extend the canvas without shrinking fonts
or frames. The caller owns pagination, captions, preset scope and export. The single-map API retains
its 150 × 95 mm envelope and validation before figure allocation.

## Prepared drawing, presets and export

The drawing helpers accept prepared data. [prepared_charts.py](prepared_charts.py) provides `annual_series`
and `monthly_cycle`, which return a Matplotlib `Figure`. Use `draw_annual_lines` and
`draw_annual_trends` on an existing axis when a composition owns its ticks or legend. The adapter owns
aggregation, completeness, source order, units, reporting-year alignment and fitted trends.
`AnnualLine` holds values aligned to supplied years; `TrendSegment` holds evaluated endpoints and a
formatted label; `MonthlyBand` holds mean and lower/upper arrays in the supplied twelve-month order.
Pass the band's statistical meaning through `band_label`. Drawing does not calculate quantiles, fit
trends, rotate months, mutate inputs or select a backend. Missing values retain gaps.

[preset.py](preset.py) provides `use_preset(preset)` and a built-in `NEUTRAL` default. Drawing helpers
read the active preset at call time; the core does not load `config/plotting/`. Keep the scope active
through export. `use_preset` applies its rcParams unless a caller that owns them passes
`rcparams=False`. Outside a scope the core draws with `NEUTRAL`.

[export.py](export.py) saves an explicitly supplied figure without workflow logging. Preserve the
caller's save options and close the figure after its final export:

```python
from blueearth_cst.plotting.export import export_figure

export_figure(fig, "plots/precip_annual_ts.png", dpi=600, metadata={"Software": None})
```

Existing workflow callers use `shared/snake_utils.py::save_figure` for workflow bundles and logging.
The [standalone demo](../../dev/scripts/plotting_standalone_demo.py) exercises the core outside CST.

## CST integration and checks

[`shared/plot_preset.py`](../shared/plot_preset.py) builds the CST preset from tracked
[`style.yml`](../../config/plotting/style.yml) and [`colors.yml`](../../config/plotting/colors.yml).
Every CST render entry point must use `use_preset(CST_PRESET)` through export; a caller that owns
rcParams uses `use_preset(CST_PRESET, rcparams=False)`. CST composition code reads token names from
`shared.plot_preset`. Figure families own scientific preparation, export and captions. The
[CST style reference](docs/cst-style.md) owns CST appearance and figure
interpretation. The [figure naming conventions](docs/figure-naming.md) collect shared rules and workflow profiles;
[weathergenr figure styling](docs/weathergenr-style.md) covers the package's
ggplot2 figures as used in CST.

For a new fixed-size grid in Matplotlib or ggplot2, read the [layout contract](docs/panel-layout.md) and the
[Python layout tests](../../tests/test_panel_layout.py), including the shared
[planner cases](../../tests/fixtures/panel_layout.csv). These files document implemented Python behavior
and comparison expectations. The repository has no standalone generic ggplot2 adapter.

[layout]: docs/panel-layout.md
