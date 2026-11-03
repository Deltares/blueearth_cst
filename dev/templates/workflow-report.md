# Workflow report template

[workflow-report.html](workflow-report.html) is an offline layout for reviewing and sharing one workflow's
outputs. It is a template, not an automatic reporting stage. The workflow engine and its outputs are
unchanged. Start with WF0; reuse the same sections for the other workflows.

The reader moves from an overview to findings, then follows a finding to its supporting figure.
Methods, limitations, output inventory and provenance remain available in the same document.
Variable filters and search narrow the gallery. Figures open with fit-to-view and actual-pixel inspection.
Print / PDF expands methods and includes all figures, regardless of the active filter. Browser print
behavior needs visual verification before using an exported PDF as a deliverable.

## Populate a report

Copy the template into `.tmp/tasks/<id>/` while developing. Replace the JSON
inside `<script type="application/json" id="report-data">` with report data. Leave the renderer unchanged.
Serialize JSON and escape every `<` as `\u003c` so a caption cannot close the script element.
Report text is inserted with `textContent`, not interpreted as HTML.

The empty template displays setup guidance. It contains no example outputs or machine-specific paths.
Use PNG, JPEG or WebP data URLs for figures: the self-contained report accepts embedded images only.
The gallery and methods operate offline. External reference links explicitly require the internet.
Do not embed private absolute paths, credentials, or an entire project archive in a shareable report.

```json
{
  "title": "Historical climate",
  "context": "Example basin / WF0",
  "workflow": "WF0 / ANALYZE CLIMATE",
  "eyebrow": "Basin assessment · historical climate",
  "summary": "What was done and what the reader can conclude.",
  "metadata": ["Example basin", "Historical period", "Data source"],
  "notice": {"label": "Draft", "text": "Interpretation awaits analyst review."},
  "indicators": [{"value": "Value", "label": "Quantity", "basis": "Units and evidence source"}],
  "findingsIntro": "Findings with links to their supporting outputs.",
  "findings": [{
    "title": "A specific finding", "status": "Draft interpretation",
    "text": "Evidence, qualification and practical meaning.", "figure": "F01"
  }],
  "decision": {"label": "Implication", "text": "The next assessment decision supported by these outputs."},
  "steps": [{"title": "Prepare inputs", "text": "What was done.", "detail": "Inputs and outputs"}],
  "methods": [{
    "id": "method-one", "title": "Method name", "text": "Definition and settings used.",
    "note": "Assumptions or missing information.", "references": ["R1"]
  }],
  "outputNote": "Included scope and omissions.",
  "figures": [{
    "id": "F01", "title": "Figure title", "group": "Precipitation", "kind": "Seasonality",
    "image": "data:image/png;base64,...", "filename": "figure.png",
    "alt": "A meaningful figure description.", "caption": "Publication-style caption.",
    "note": "Optional qualification", "featured": true, "sha256": "Image byte digest"
  }],
  "limitations": [{"title": "Scope of inference", "text": "What the evidence cannot establish."}],
  "inventoryNote": "A selected evidence bundle; enumerate omitted output groups.",
  "inventory": [{"name": "Output group", "role": "Purpose", "availability": "Embedded / omitted"}],
  "provenance": [{"label": "Generating run", "value": "Recorded identifier or explicitly unknown"}],
  "references": [{"id": "R1", "title": "Method reference", "text": "Citation and offline summary"}],
  "footer": "Project · workflow · report date"
}
```

Use unique figure IDs and reference IDs. `findings[].figure` identifies an included figure;
`methods[].references` identifies references. An optional reference `url` accepts HTTP(S) links.
`sha256` is optional provenance retained in the data; the renderer does not verify it automatically.
The example image value above is a placeholder; replace it with actual base64 image bytes.
The schema is a prototype authoring convention, not a stable workflow output contract.

## Scientific reporting rules

- Keep calculations, exported values and analyst interpretation distinct. Identify rounded readings from
  figure annotations; do not present them as independently calculated statistics.
- Identify the generating run, config identity, source data, period and spatial support. An unknown
  method or revision remains explicitly unknown, rather than inferred from current defaults.
- Describe the estimator, aggregation, missing-data treatment and uncertainty that produced each result.
  Prefer retained method metadata and numerical tables over reverse-engineering plots.
- Link findings to evidence. Significance, source ranking and causality need more than visual appearance.
- Embed retained figures unchanged and retain meaningful captions; report omitted products explicitly.
- Keep caveats relevant to the decision. Execution success alone does not establish model quality.
- Follow [toolbox-plot-style.md](../reference/toolbox-plot-style.md) when revising the scientific figures themselves.

## Reuse across workflows

The shell stays the same; content and evidence change with the workflow's actual scope.

| Workflow | Summary focus | Typical evidence |
|---|---|---|
| WF0 | Historical climate and forcing selection | Coverage, seasonality, extremes, source comparisons |
| WF1 | Model construction and historical behavior | Build settings, forcing QA, hydrographs, performance |
| WF2 | Projection plausibility | Model/scenario inventory, monthly changes, ensemble spread |
| WF3 | Stochastic scenario collection | Perturbation grid, realizations, generator diagnostics |
| WF4 | System response and vulnerability | Run IDs, retained responses, indicators, response surfaces |

WF2 contextualizes projection plausibility; it does not drive WF3 scenario generation.
WF3 and model construction are independent prerequisites for WF4. Do not imply all workflows were run
merely because the template supports them. A future adapter should read retained outputs and metadata
without recomputing scientific analysis in the presentation layer.
