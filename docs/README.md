# Documentation

Choose the task you want to complete:

- [Install](guide/install.md), then [run the example](guide/quick-start.md).
- [Creating a project](guide/new-case.md).
- [Run/resume](guide/running.md), [choose indicators](guide/new-case.md#choose-indicators)
  and [read tables](guide/outputs.md).
- [Read timings](technical-manual/runtime-and-resources.md), [diagnose a failure](guide/troubleshooting.md), or
  [use optional notebooks](guide/notebooks.md).
- [Technical manual](technical-manual/overview.md): implemented interfaces and advanced extension.
- [Methodology](site/scientific-approach/overview.qmd): foundations of climate stress testing.

Canonical guide Markdown is in `guide/`; its asset sources remain in `guide/assets/`.
The technical manual and local component READMEs own reference detail.
The site configuration selects published sources and stages derived guide copies; edit canonical sources,
never those generated copies. Repository source compatibility pages remain at existing setup/reference routes.

The [publication manifest](publication/manifest.json) selects seven User Guide chapters and 32
Technical Manual chapters for shared HTML/PDF assembly.
See [Build the publications](publication/README.md) for exact commands and output locations.
The [site configuration](site/_quarto.yml) owns navigation and retained routes.
Build the complete manuals before rendering the site so its downloads are available.

Notebooks remain editable `.ipynb` sources and are not executed by documentation rendering.
[Upstream references](references/) are repository-only consultation material; the 2025 note describes
the earlier platform, not the current workflow contract. No supported response-surface construction
walkthrough or completed new scientific-approach volume is claimed by this tranche.
