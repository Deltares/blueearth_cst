# Documentation guidelines

Last updated: 2026-10-20

These guidelines govern authoring and review of the scientific approach, user guide,
toolbox reference, and technical notes. They apply to human and agent authors.

For the agreed editorial preferences and collaborative page-review approach, also follow
[Documentation revision guidance](documentation-revision-guidance.md).

## Choose the page's job

Give each page one primary reader need. Preserve the site's existing navigation;
these are content distinctions, not a requirement for four new directories.

| Reader need | Content | Completion criterion |
|---|---|---|
| Learn | A guided example in the user guide | The reader completes a task and interprets its result |
| Act | How-to or troubleshooting | Prerequisites, actions, outcome, and recovery are explicit |
| Understand | Method page or technical note | Method, implementation, assumptions, and limits connect |
| Look up | Toolbox reference | Supported inputs, outputs, conventions, and failures are precise |

Use `technical-writer` for the documentation journey and editorial integration;
`academic-writer` for substantial scientific explanations and technical notes;
`api-documenter` for configuration, CLI, and output-schema reference. Resolve missing
scientific evidence through `academic-researcher`. Use `critical-thinker` for a bounded
critique of assumptions, and `model-validator` for claims about empirical adequacy.
`academic-reviewer` is appropriate for manuscript-style technical-note review, not
routine guide edits. Use `dataviz-designer` when a figure needs analytical design.

These assignments do not require every role on every page. The writing owner can
perform bounded source checks; unresolved method or interface decisions go to the
relevant scientific or engineering owner before documentation asserts a resolution.

Use the shared `scientific-software-documentation` skill for the method-to-code-to-evidence
connection, alongside `tutorial-authoring`, `quarto`, or `citation-management` as needed.
Canonical shared guidance lives at
`infrastructure:artifacts/skills/scientific-software-documentation/SKILL.md`;
resolve it through `BRAIN_INFRA_ROOT`, never copy its body into this repository.

## Write for the reader

- Write the user guide for water resources, hydrology, and climate-risk
  specialists. Use domain terms such as *forcing*, *natural variability*,
  *response surface*, and *return level* where they carry useful meaning.
  Explain a term briefly when its meaning in this toolbox may be unclear.
- Use plain language for software operations. Tell readers what to choose, what
  command to run, where to find the result, and how to check it. Introduce file
  names and configuration keys when the reader needs them to act.
- Lead with the scientific question, interpretation or reader action before explaining software mechanics.
  Prefer "identifiers" to "keys", "climate scenarios used in the experiment" to "retained collection",
  and "simulations included in each group" to "membership" in explanatory prose. Preserve the exact
  filenames, column names and technical terms needed to follow commands or code examples. Simplify the
  explanation without losing distinctions needed to interpret results or trace their source.
- Show directory layouts in fenced `text` blocks using `├──`, `│` and `└──`, with
  four-space nesting and aligned descriptions. Show nested folders as branches;
  keep individual paths and executable commands in their appropriate formats.
- State the intended use and limits of an output. Distinguish a screening
  indicator from a design estimate, and a climate-projection plausibility
  comparison from a stress-test input. Avoid claiming that the toolbox makes a
  decision or validates a model when it only supplies evidence to assess.

## Put information in the right place

- `docs/site/index.qmd` introduces the assessment approach and directs readers to
  tasks. `docs/site/guide/` explains setup choices, running workflows, and
  interpreting outputs. Keep each page focused on a reader's task and link to
  detail instead of repeating it.
- `docs/` holds user-facing instructions and references. Put implementation
  rationale, review history, milestone status, test evidence, and maintainer
  procedures under `dev/`. Do not place development commentary in the user
  guide.
- `config/templates/` is the source for configuration examples users copy;
  `config/defaults/` contains settings read by runs. Refer to these files rather
  than maintaining a second, divergent configuration specification in prose.
- Give each concept one authoritative explanation and cross-link procedures and
  references to it. Define toolbox-specific terminology consistently with
  [the indicator glossary](toolbox-indicator-glossary.md) and the published conventions page.
- Refer readers to the official HydroMT and HydroMT-Wflow manuals for upstream spatial handling,
  catalog schemas, readers and setup-method details instead of explaining those interfaces again.
  Document CST-specific choices, conventions and scientific implications here, with upstream links
  at the relevant point of use.
- Keep technical notes beside the relevant scientific explanation when they describe
  current supported methods. Keep historical evidence and internal design decisions
  in their existing `dev/` homes. Do not publish legacy technical-note chapters merely
  because they exist: reconcile them with the current implementation first.

## Ground scientific explanations

For each substantive method, cover the following where relevant; use the reader's
question to choose headings rather than applying a rigid paper structure:

1. The assessment question, intended use, and reason for choosing the method.
2. Mathematical definitions, notation, dimensions, units, and aggregation conventions.
3. The implemented algorithm, important approximations, and differences from the cited method.
4. User choices, defaults, and consequences, linked to their configuration reference.
5. An interpretable example connecting inputs, computation, and outputs.
6. Supporting evaluation, assumptions, failure regimes, and limits of applicability.

Use primary literature for scientific rationale, upstream documentation for upstream
capabilities, and current source, tests, schemas, and retained run evidence for toolbox
behavior. A publication about an upstream model does not prove that this toolbox enables
every feature or that a particular basin model is accurate. Report conflicts between
these authorities instead of silently choosing a convenient account.

Distinguish implemented capabilities, evaluated behavior, illustrative examples, and
planned extensions. Tie claims of accuracy or fitness for use to the actual variables,
locations, periods, regimes, and stressor domain evaluated. Where observations or
validation evidence are missing, say what remains unevaluated; physical plausibility
and successful execution do not establish predictive skill.

Preserve the toolbox's scientific distinctions:

- Scenario-neutral stressors define the experiment independently of selected GCMs.
  Projection changes provide a plausibility overlay; they do not assign probabilities
  to stress-test grid cells.
- Historical forcing, the historical hydrological simulation, and generated unperturbed
  realizations serve different purposes. Name the reference used in every comparison.
- State additive temperature changes, precipitation factors, monthly conventions,
  calendars, reference windows, and spatial aggregation when they affect interpretation.
- Separate weather variability from forcing, parameter, structural, and scenario
  uncertainty. Ensemble frequencies are conditional on the experiment and sampling;
  they are not automatically probabilities of future climate conditions.
- Explain return-level fitting, effective record length, pooling assumptions, and
  extrapolation limits when reporting extremes. Identify thresholds as user decisions
  or externally supported criteria rather than universal defaults.

Use Quarto citations with a maintained bibliography for scientific claims. Check that
the cited source supports the particular statement, including qualifications; validating
a DOI alone is insufficient. Define symbols before use and link equations, tables,
and figures through stable cross-references.

## Keep instructions current

- Check workflow order, prerequisites, commands, config keys, and output paths
  against the current runners, example project files, and output contracts.
  The five workflows are separate: historical climate, model build, projections,
  scenario generation, and simulation. Projections provide context; they do not
  drive scenario generation.
- Use examples that a reader can adapt. State external data-access requirements
  and important limits of the example. Mark placeholders clearly and never
  present a machine-specific path as generally available.
- When behavior or a path changes, update every live user-facing reference in
  the same task. Keep historical migration records as records rather than
  silently rewriting their past instructions.
- Identify the supported toolbox version or commit for version-sensitive examples.
  Record data provenance, configuration, and relevant retained run IDs with numerical
  examples. Link to existing evidence rather than inventing a separate receipt system.
- Reference pages describe required and optional fields, defaults and precedence,
  path-resolution rules, units, constraints, outputs, side effects, and actionable
  failures as applicable. Keep exhaustive machine contracts in their owning schemas
  or generators; explain consequences in prose.

## Make examples and figures usable

- Separate a walkthrough of retained outputs from a tutorial that executes workflows.
  State which steps run computations and which inspect supplied results. Never imply
  that a documentation render reran a model or reproduced its numerical results.
- State data access, environment, expected runtime and disk needs when material.
  Qualify estimates by the example and machine; do not promise a universal runtime.
  Include a visible success condition and recovery from common failures.
- Explain why each substantive command is needed and what its output means. Put
  optional parameter exploration after the reader has one working result.
- Give figures a clear analytical question, readable units and labels, a self-contained
  caption, and alternative text. Avoid color-only distinctions. Describe uncertainty
  and baselines in the caption when they affect interpretation; link accessible static
  alternatives for interactive material. Follow [plot style](toolbox-plot-style.md).

## Check a revision

Select checks through [the validation ladder](dev-validation.md), based on the
changed claims and affected consumers. For this maintainer reference, inspect links,
consistency, and any affected documentation checks; it is outside the site render scope.

For published pages, read the result as a practitioner following its links and commands.
Check local links, citation resolution, equations, tables, navigation, and figure
legibility in rendered output. Build with `pixi run docs-build` and use
`pixi run docs-preview` for visual review when appropriate. Inspect execution settings
first: a site render must not launch expensive workflows or fetch production data.
Do not treat `freeze: auto` as an unconditional guarantee against execution.

Verify changed commands and examples with proportionate existing checks. Reuse qualified
run evidence when inputs, implementation, and claims remain applicable; a prose-only
edit does not require rerunning the model. Changed numerical claims need their own
support, and figure changes need visual inspection under the validation ladder.

Report the actual checks, evidence reused, skipped checks, and remaining limitations.
If rendering is blocked, retain the error and complete independent text, link, and
source checks. Never claim visual review, reproduction, or empirical validation from
inspection of source text alone.
