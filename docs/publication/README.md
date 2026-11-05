# Build the publications

The [manifest](manifest.json) selects canonical sources in reading order.
Edit those sources, this manifest or [format metadata](format.yml); do not edit generated chapters.
Chapter images under `docs/technical-manual/assets/` are bundled locally in the website and complete
manuals. Keep their provenance beside the source assets and supply captions and alternative text in
the chapter Markdown.
The technical manual follows the manifest's reading order, with separate pages for the modeling
overview, model preparation, hydrological simulation, weather generation and indicator calculations.
Data pages cover requirements, spatial inputs, historical meteorology, projections, catalogs and input formats.
Detailed references follow the main chapters: configuration rules, location matching and daily responses,
indicator settings, saved-result tables, run records, resource measurements, transfers and custom indicators.
The website sidebar uses the same sequence for previous/next navigation. Indicator appendices are excluded;
their available sources are declared in
[`local-source-appendix.yml`](../technical-manual/local-source-appendix.yml).
No notebook, model or data acquisition runs during publication.

## Full HTML and PDF build

Run from the toolbox root with the locked environment installed:

```powershell
pixi run python dev/scripts/build_publications.py --source-root . --output .tmp/publications/build
New-Item -ItemType Directory -Force docs/site/manuals | Out-Null
Copy-Item .tmp/publications/build/manuals/*.html docs/site/manuals/
Copy-Item .tmp/publications/build/manuals/*.pdf docs/site/manuals/
pixi run docs-build
```

The first command stages the selected chapters and renders the two complete volumes with Quarto HTML
and Typst PDF. After copying the downloads, `pixi run docs-build` renders the navigable website
and retained compatibility routes.
Both explicitly disable computation. Quarto and its Typst engine must already be installed.
Neither command installs packages, starts workflows or acquires external datasets.

For a session borrowing a compatible main runtime, use its helper instead of installing an environment:

```powershell
$runtime = '<stable-toolbox-checkout>'
& "$runtime/.pixi/envs/default/python.exe" -B dev/scripts/session_runtime.py --runtime-root $runtime -- python dev/scripts/build_publications.py --source-root . --output .tmp/publications/build
New-Item -ItemType Directory -Force docs/site/manuals | Out-Null
Copy-Item .tmp/publications/build/manuals/*.html docs/site/manuals/
Copy-Item .tmp/publications/build/manuals/*.pdf docs/site/manuals/
& "$runtime/.pixi/envs/default/python.exe" -B dev/scripts/session_runtime.py --runtime-root $runtime -- quarto render docs/site --no-execute
```

Windows Quarto/Sass rendering may require normal user permissions rather than a restricted agent token.
Use the existing runtime; do not change permissions or install another toolchain to mask an access error.
For staging without rendering, add `--stage-only` to the builder command.

## Generated outputs

- `docs/site/guide/`: derived guide chapters and workflow assets.
- `docs/site/technical-manual/`: derived reference chapters.
- `docs/site/manuals/user-guide.html` and `.pdf`: complete user guide.
- `docs/site/manuals/technical-manual.html` and `.pdf`: complete technical manual, including detailed references.
- `docs/site/_site/`: navigable website, retained routes and downloadable volumes after the site build.

The intermediate complete publication tree is `.tmp/publications/build/`.
Its `manuals/` directory contains the volume sources and rendered outputs; copy only HTML/PDF downloads
to the site, not generated Quarto sources or caches.

The generated directories are ignored. Do not commit them or edit their prose independently.
Keep both volumes of your chosen format together for offline reading; the PDFs need no HTML companions.
Images are embedded in the complete HTML manuals without links to separate image files.
Website chapters retain links to their full-size image assets.
Links between PDFs may open the volume without jumping to the section, depending on your PDF reader.
Use the contents if needed.
The workflow overview and chapter illustrations are bundled from the canonical local image assets.
Required inputs must exist; a successful smaller publication is not an acceptable missing-source fallback.

## Source identity and external material

`source-hashes.json` records consumed input hashes; retain it with review artifacts.
The manifest's source label identifies a local build; the hashes identify its exact consumed sources.
Repository links point to the maintained `dev` branch in `Deltares/blueearth_cst`.
Its repository URL is a locator for external source evidence, not proof that an unpushed candidate
exists online. Network availability is not verified by rendering.

Only the selected guide/manual chapters and module/set sources are included in the complete volumes.
Other linked repository evidence, configuration files, optional notebooks and upstream references require
a separate checkout or network access. Existing scientific pages remain available on the website;
this publication does not claim a completed new scientific-approach volume or a supported
response-surface construction tutorial.

The retained [P3 pilot](../publishing-pilot/README.md) remains a separate review fixture.
Full publication assembly does not replace its source or retained review evidence.
