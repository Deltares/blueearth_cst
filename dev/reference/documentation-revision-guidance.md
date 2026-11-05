# Documentation revision guidance

Last updated: 2026-11-05

This living guide records the editorial preferences agreed during the homepage and quick-start review.
Apply them when revising other pages, adapting the level of detail to each page's purpose.
[Documentation guidelines](dev-documentation.md) remain authoritative for scientific accuracy,
source ownership, authoring conventions and verification.

## Lead with purpose

- Describe workflows through their purpose and data requirements, independently of the datasets
  currently supported. Keep available dataset options and source-specific requirements in the dataset
  reference; do not present first-version options as intrinsic to the assessment method.
- Use **climate risk assessment**, or **bottom-up climate risk assessment**, for the full toolbox process.
  Reserve **stress testing** for scenario generation and assessment of hydrological responses across
  perturbed climate conditions; do not use it as a synonym for the entire assessment.
- Introduce historical climate analysis through its purpose: understanding baseline climate variability
  and the climatic context of hydrological conditions, and selecting a gridded forcing dataset.
  Do not label it optional in conceptual descriptions. Explain execution dependencies only where needed
  for running the workflows; distinguish climate diagnostics from simulated hydrological responses.
- Introduce the toolbox through climate vulnerability and understanding water-system robustness under
  an uncertain future. Explain what readers can learn or do before describing implementation.
- Keep model and dataset names out of the general homepage introduction. Include them where readers
  need to understand a method, configure a run or interpret an example.
- Mention climate projections alongside interpretation rather than adding a separate technical
  explanation to the homepage. Preserve the distinction between projection context and stress testing
  wherever the method is explained.
- Keep the homepage focused on purpose and navigation. Introduce the five workflows in the user guide;
  do not repeat their IDs, dependencies and execution order on the homepage.
- Focus navigation on live documentation. Do not reintroduce a Downloads button or homepage PDF prompt
  without a new decision. This preference does not itself retire publication tooling or old artifacts.

## Make pages easy to read on screen

- Public documentation must stand alone in the public repository. Avoid links or path-only citations to
  private-checkout records, including labels such as "private-checkout-only citation". Use public source,
  configuration, tests or published references to support current guidance. Preserve any necessary
  explanation in the public document; keep private history in its existing private home. If a claim
  depends on unavailable evidence, qualify it rather than presenting that evidence as publicly accessible.
  Apply this rule as each page is revised.

- Keep website figures unnumbered. Refer directly to nearby illustrations with wording such as
  "the diagram below". Retain useful explanatory captions where needed; the top-down/bottom-up
  comparison is displayed without a caption. This website convention does not change PDF numbering.
- Keep processing rule tables on the dedicated workflow rule reference page, organized by workflow.
  Link to the relevant section from each workflow page. Omit the `all` target, `gather_benchmarks`
  and `gather_logs` entries from these tables.
- Label highlighted configuration excerpts with the source filename above the code block, using Quarto's
  `filename` attribute, for example: `filename="project_config.yml (excerpt)"`. Include a relative path
  only when needed to distinguish files; avoid repeating the filename in the section heading.
- In workflow diagnostic summaries, explain why each variable or diagnostic matters to the assessment,
  then name the main method if needed. Keep each entry to at most two concise sentences; figure symbols,
  shading and other display conventions belong in captions or the results-reading guidance.
- Use neutral reference language in the Technical manual, such as "Files supplied by the user" and
  "Files prepared by the toolbox". Describe requirements and behavior directly; reserve reader-directed
  instructions for procedural steps. The User guide can retain its practical, direct-address style.
- Under "Models and tools", distinguish external software tools from toolbox modules. Wflow is
  hydrological modelling software, Wflow-SBM is the model, HydroMT is a model-building framework,
  HydroMT-Wflow is its Wflow-specific plugin, and weathergenr is a weather-generation package.
  Use "indicator module" and "visualization module" for the toolbox's corresponding functionality;
  "module" here denotes a functional component, not necessarily a single Python file.
- Use descriptive workflow names in user-facing prose, headings, tables and diagrams. Omit short IDs
  such as `WF0` and `WF1`, and do not introduce the abbreviation `WF`. Preserve executable identifiers,
  configuration keys and filenames verbatim. Apply this rule as each page is revised.
- Prefer short paragraphs with one clear purpose. Remove repeated explanations and unnecessary detail
  before adding more headings or callouts.
- Group reference topics by their role. Separate datasets from data access and preparation, and mirror
  that distinction in overview pages and navigation. The first manual version omits a standalone data
  quality and limitations chapter; retain relevant qualifications alongside methods and results.
- Document the HydroMT and CMIP6 catalogs in the manual. The ERA5 daily Zarr override is a development
  aid, not a separate user-facing catalog; keep its configuration out of the catalog chapter inventory.
- Use headings for substantial reader tasks. Merge brief prerequisite guidance into the introduction
  when a separate "Before you start" section adds little value.
- Use short bold cues such as "While it runs" and "When it finishes" within a task section when they
  make instructions easier to scan. Avoid deeply nested sections.
- Keep commands, important paths and success checks easy to find. Explain only the options needed for
  the current task; link to the execution guide for fuller detail.
- Show a compact folder overview at quick-start level. Give the exact path to the result being inspected,
  then link to the output guide for detailed schemas, joins and provenance.
- Keep qualifications where readers need them, without repeating the same warning in every paragraph.
  Shortening must preserve prerequisites, unavailable-value handling and limits on interpretation.
- Use verified historical and projection reference periods when describing climate changes. Distinguish
  the historical record from generated unperturbed weather; their reference periods may differ.
- Include runtime estimates only when supported and qualified by the relevant run and machine.

## Keep overview diagrams simple

- Use short functional labels. Omit workflow numbers and explanatory definitions from the introductory
  workflow diagram; retain detailed descriptions in nearby prose or later pages.
- Do not repeat "Workflow" above every label. Use one caption such as "Toolbox workflows".
- Use white backgrounds and neutral borders for this overview. Add color only when it communicates an
  agreed distinction; this preference does not replace the scientific plotting palette.
- Explain arrow meaning in the caption. Execution order must not imply that each workflow depends on
  the preceding one, or that projections drive scenario generation.
- Keep essential prerequisites clear in accompanying text, including the model and generated scenarios
  required for simulation. Avoid adding filenames, output inventories or technical definitions to boxes.
- Update the editable diagram source, exported image and alternative text together. Inspect the image
  for clipping, legibility and consistency with the surrounding text.

## Review together

The user reads rendered pages and edits or annotates their canonical Markdown or Quarto source.
Follow the repository's source ownership: staged site pages and `_site/` HTML are generated artifacts.
For example, edit `docs/guide/quick-start.md` for the quick start and `docs/site/index.qmd` for the homepage.

Use these conversational shortcuts:

- **Apply it:** apply the latest agreed revision to the page or diagram under discussion, preserving
  unrelated edits and source formatting. A request for an opinion alone is discussion, not approval to edit.
- **Render it:** rebuild the current page so the user can refresh it. Rebuild the site when a shared
  navigation or style change affects every page. Report the result and link to the generated page.
- Render affected pages automatically after each revision, without waiting for a separate request.
  Include the rendered-page links so the user can review the latest version immediately.
- **REVIEW comments:** treat `<!-- REVIEW: ... -->` as an actionable note. Address it, verify any factual
  claim and delete the comment once resolved. Leave unresolved comments with a clear explanation.

If a quoted comment is absent from saved source, say so and check the file path or unsaved editor state.
Do not claim to have removed a comment that was not found. Avoid overwriting concurrent user edits.

Preserve readable source line breaks, following the repository's 110-character prose limit. When proposing
copy-ready replacement text in chat, use a fenced Markdown block with explicit line breaks. Do not assume
the user's terminal interface has a Copy button. Directly applying approved wording avoids copy/paste issues.

Distinguish source edits, successful rendering and visual inspection in completion messages. Do not claim
the browser view is current until the page has been rebuilt. Follow the existing Git/session policy and
explicit holds for commits and landing; "Apply it" and "Render it" are not publishing instructions.

## Improve this guide

Add newly agreed, reusable preferences here as the review continues. Revise an existing rule when a decision
changes rather than appending conflicting instructions. Keep page-specific corrections in their source
pages and scientific or build rules in their existing authoritative references. Do not treat an unaccepted
suggestion or a one-off wording choice as a universal rule.
