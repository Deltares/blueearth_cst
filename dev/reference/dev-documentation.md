# CST documentation guidelines

Last updated: 2026-11-05

Use this guide for CST user guides, scientific explanations, toolbox reference and technical notes.
It owns CST content and presentation conventions; the [validation ladder](dev-validation.md) owns
check selection, and the [record lifecycle](dev-record-lifecycle.md) owns development references.

## Audience and terminology

- Write for water-resources, hydrology and climate-risk practitioners. Use familiar domain terms;
  explain toolbox-specific meanings. Follow the [indicator glossary](toolbox-indicator-glossary.md)
  and [data conventions](../../docs/technical-manual/data-formats-and-conventions.md).
- Call the full process **climate risk assessment** or **bottom-up climate risk assessment**.
  Reserve **stress testing** for scenario generation and assessment of hydrological responses to
  climate perturbations. The toolbox supplies evidence for decisions; it does not make them.
- Use descriptive workflow names in user-facing prose, headings, tables and diagrams. Omit `WF0`,
  `WF1` and `WF` shorthand; preserve executable identifiers, keys and filenames verbatim.
- Describe historical climate analysis as context for baseline variability, hydrological conditions
  and forcing selection. Keep its conceptual role distinct from whether execution is required.
  Distinguish climate diagnostics from simulated responses.
- Under “Models and tools”, distinguish Wflow software, the Wflow-SBM model, the HydroMT framework,
  its HydroMT-Wflow plugin and the weathergenr package. “Indicator module” and “visualization module”
  describe toolbox functions, not necessarily individual Python files.
- Use neutral Technical Manual labels such as “Files supplied by the user” and “Files prepared by
  the toolbox”. Use direct instructions in procedures; the User Guide may address readers directly.

## Content and sources

- Keep the homepage focused on purpose and navigation. Introduce climate vulnerability and
  water-system robustness before software or datasets. Put the five workflows' responsibilities,
  dependencies and execution order in the User Guide. Do not add a homepage Downloads button or PDF
  prompt without revisiting that presentation decision.
- Describe workflows by purpose and data needs, independently of supported datasets. Separate dataset
  descriptions from access and preparation. Keep qualifications beside the relevant method or result;
  a separate first-version data-quality chapter is not needed.
- Document HydroMT and CMIP6 catalogs in the Technical Manual. The ERA5 daily Zarr override is a
  development aid, outside its user-facing catalog inventory.
- Keep processing-rule tables in the dedicated workflow-rule reference, grouped by workflow. Link
  from workflow pages; omit `all`, `gather_benchmarks` and `gather_logs` rows.
- Use official HydroMT and HydroMT-Wflow manuals for upstream schemas, handling and setup methods;
  explain CST choices locally. Check current runners, seed configurations and contracts before
  treating older technical material as supported behavior.
- Use Quarto citations and a maintained bibliography for scientific claims. Numerical examples need
  source provenance, configuration and relevant retained run IDs.
- Public pages must be usable from the public repository. Cite tracked sources or published references;
  do not depend on private records, ignored instructions or machine-specific paths.

Preserve these scientific distinctions:

- Scenario-neutral stressors define the experiment independently of GCM selection. Projection changes
  are plausibility overlays, not generation inputs or probabilities assigned to stress-grid cells.
- Historical forcing, historical simulation and generated unperturbed weather serve different roles.
  Name each comparison's reference; generated weather may use a different reference period.
- State additive temperature changes, precipitation factors, monthly conventions, calendars,
  reference windows and spatial aggregation when they affect interpretation.
- Separate weather variability from forcing, parameter, structural and scenario uncertainty.
  Ensemble frequency is conditional on the experiment and sampling, not automatically a future-climate
  probability.
- For extremes, explain return-level fitting, effective record length, pooling and extrapolation.
  Distinguish screening indicators from design estimates. Present thresholds as user choices or
  externally supported criteria, not universal defaults.

## Presentation

- Keep website figures unnumbered; use direct references such as “the diagram below”. Retain useful
  captions; the top-down/bottom-up comparison has none. PDF numbering is separate. Scientific figures
  follow [CST style](../../blueearth_cst/plotting/docs/cst-style.md).
- In diagnostic summaries, explain why each variable matters, then name the main method if needed.
  Use at most two short sentences per entry; explain symbols and shading in captions or results guidance.
- Label highlighted configuration excerpts with Quarto `filename="project_config.yml (excerpt)"`.
  Add a path only to distinguish files; do not repeat the filename in the heading.
- Show directory trees in fenced `text` blocks with aligned `├──`, `│` and `└──` branches and
  four-space nesting. In the quick start, give a compact folder overview and the exact result path;
  link to the output guide for schemas and provenance. Preserve prerequisites and unavailable values.
- Use substantial-task headings. Fold brief prerequisites into the introduction when useful; short
  cues such as **While it runs** and **When it finishes** can organize a task. Keep commands, paths
  and success checks visible, with a link to the full execution guide.
- For overview diagrams, use short functional labels without workflow numbers or definitions in
  boxes. Use white backgrounds and neutral borders; colour must convey a defined distinction.
  Explain arrows in the caption: convenience order is not dependency order, projections do not drive
  generation, and simulation requires both model and scenario inputs. Update editable source,
  exported image and alt text together; inspect clipping, legibility and page consistency.

## Publishing and checks

- Edit canonical sources, not staged site copies or `_site/` HTML:

  | Content | Source |
  |---|---|
  | Homepage | `docs/site/index.qmd` |
  | User guides, including the quick start | `docs/guide/` |
  | Configuration examples users copy | `config/templates/` |
  | Defaults that affect runs | `config/defaults/` |
  | Maintainer procedures, rationale and retained evidence | `dev/` |

- Link configuration owners instead of maintaining another specification. Keep prose within 110
  characters per line; URLs and code blocks are exempt.
- Use `pixi run docs-build` and `pixi run docs-preview` for publication checks when appropriate.
  Inspect execution settings first: `freeze: auto` does not guarantee that rendering cannot run
  computations or fetch data. Rebuild affected pages for review; rebuild the site for shared
  navigation or style changes. Check citations, equations, tables, navigation and figure legibility.
- Follow the [validation ladder](dev-validation.md) for changed commands, numerical examples and
  figures. This reference is outside the published site; source, link and consistency checks suffice
  for its prose changes. Report edits, build success and visual inspection separately.
