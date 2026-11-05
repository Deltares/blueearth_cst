# Publishing pilot

This historical pilot is retained as a test fixture at its approved V9 revision, `fb9aca81`.
Its live-source excerpt boundaries predate the current documentation. Reproduce the commands below
from that revision; build current deliverables with [the complete publication builder](../publication/README.md).
Tests use the frozen inputs under `tests/fixtures/docs-pilot/source/`, independently of current page headings.

This isolated P3 sample assembles the frozen indicator-selection excerpts from live sources using
`manifest.json`. It does not migrate chapters or change the production site. Generated prose is refreshed
on every build and must not be edited independently. `format.yml` supplies document metadata for both formats;
Quarto project configuration is ignored beneath hidden directories, so this pilot uses no staged project file.

From the session root, borrow the compatible stable runtime through the documented helper:

```powershell
$runtime = '<stable-toolbox-checkout>'
& "$runtime/.pixi/envs/default/python.exe" -B dev/scripts/session_runtime.py --runtime-root $runtime -- python dev/scripts/build_docs_pilot.py --output .tmp/tasks/t2610211349/p3/build
```

Use normal user permissions on Windows if Quarto's Sass subprocess reports an invalid handle under the
restricted agent token. No dependency installation is performed. Quarto and its Typst PDF engine must already
be installed. The command runs `quarto render <source> --to html|typst --no-execute`; metadata also sets
`engine: markdown` and `execute.enabled: false`. It invokes no workflows, notebooks or data acquisition.

Outputs beneath the selected root:

- `index.html` and `index.pdf`: the same manifest-driven sample, with internal section links.
- `assets/shape-of-a-run.html`: the existing interactive asset, copied verbatim.
- `assets/shape-of-a-run.svg`: the existing SVG with its host styles inlined.
- `toolbox-reference/workflow-simulate-system.html#configuration`: preserved rendered compatibility route.
- `technical-manual/indicator-requests.html#configuration`: selected canonical configuration destination.
- `source-hashes.json`: byte hashes of consumed inputs for unchanged-source checks.

The tracked isolated fixture under `tests/fixtures/docs-pilot/` demonstrates old `.qmd` source navigation
before rendering. Its wrapper links to the represented canonical Markdown and preserves `#configuration`.
Staging rebases that link to the corresponding HTML destination. The live page is untouched.

The default README contributes only its reader conventions before `Advanced implementation details`.
That advanced section remains in the canonical README and is excluded from both pilot formats.
Its reader-facing link resolves to the existing module README's `Add a calculation` section.
The pilot subtitle and input hashes identify revised working-tree content; unchanged external reference
targets remain pinned to the earlier repository revision, without claiming online availability.

The PDF and the HTML static fallback include all five workflow explanations and the order/dependency legend
derived from the interactive asset. The HTML also retains the interactive iframe. Source links outside the
sample are explicitly labelled repository/online references and pinned to the specification's source revision.
They require network access; the indicator module README is not included in the offline sample.
The HTML's Other Formats link uses Quarto `format-links` metadata to label the Typst-produced output as PDF.

Demonstrated features: prose, headings, inline code and notation, fenced command examples, attribution,
internal links, repository references, SVG and an interactive/static annex. Formal equations, citations,
notebook publication, full manuals and response-surface construction are outside this pilot's coverage.
