# Technical-manual illustrations

## Workflow diagrams

`workflow-wf0.svg` through `workflow-wf4.svg` are original editable vector sources for the five workflow
pages. They summarize current Snakefiles, their included rule modules and runner boundaries; the pages'
tables provide the exhaustive rule inventories. They are explanatory groupings, not exported execution DAGs.

Workflow diagrams follow the Workflow steps sector of `toolbox-conceptual-overview-illustrated.svg`:
regular Segoe UI text (Arial/sans-serif fallback), `#292d34` labels, white boxes with 6-pixel corner radii,
and `#4f555e` borders and orthogonal arrows at 1.5 pixels. Main labels are 16 pixels; supporting labels
are 13 pixels. Arrowheads use an 8-by-6 triangle. No coloured labels are needed.
Dashed `#f6f6f4` boxes retain the distinction for runner actions in scenario generation and hydrological
simulation. Conditional branches are labeled in text. The SVGs contain accessible titles/descriptions
and require no external fonts. The Quick start overview uses the same visual tokens in
`docs/guide/assets/workflow-overview.drawio` and its PNG export; its arrows show the documented reading order.
Edit the SVG text directly and keep each page's diagram, caption and rule table consistent. Validate XML,
local links, and geometry; inspect publication rendering when changing these assets.

## Toolbox overview figure

`toolbox-conceptual-overview-illustrated.svg` (PNG export alongside) is the overview-page figure. It is
generated, not hand-edited: the source, design specification, sources and arrow audit live with the
plain and illustrated variants on branch `docs/technical-manual-conceptual-figure` (commits `17404c9a`,
`61438282`, `docs/technical-manual/assets/toolbox-conceptual-overview*.md`). The HydroMT and Wflow
logos are embedded unaltered from the projects' own repositories (`Deltares/hydromt`
`docs/_static/hydromt-icon.svg`; `Deltares/Wflow.jl` `docs/_static/logo.png`), both MIT-licensed
repositories with no separate logo terms found. Display against a white background.

## Modeling tool illustrations

Original embedded images extracted without modification from the user-supplied
`CST Development Phase Report_CLEAN_Feb14_2025 V2.docx`.

Source document SHA-256: `52373ccabfd542da013d14bd9dbfd8bed914ce91401e701c0fbd6a157cf414bb`.

| Asset | Embedded source | Report figure |
|---|---|---|
| `hydromt-data-to-model.png` | `word/media/image4.png` | Figure 2-2, HydroMT overview (Eilander et al. 2023) |
| `wflow-sbm-processes.png` | `word/media/image5.png` | Figure 2-3, Wflow-SBM hydrological processes |

The modeling chapter supplies captions and alternative text. The HydroMT diagram retains its original
INI label; the caption explains that current CST model-build recipes use YAML. The Wflow diagram is
conceptual, with feature activation determined by the model setup. Preserve the original image bytes
and transparency; display against a white background so labels remain readable in dark-mode HTML.
