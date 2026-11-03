# Interactive figure comparison template

The tracked [HTML template](figure-comparison.html) is the `artifact-comparison.html` starter of the shared
`web-ui-mockups` skill (brain-infrastructure `057df6b`, 2026-10-29) with its sample records removed. It
provides figure navigation, side-by-side and guarded wipe comparison, linked zoom and pan, captions inside
the canvas, optional canvas boundaries, consistent physical font scaling, per-figure file names, source
folder links and image metadata. It remembers the selected figure and view options in browser-local
storage scoped to the page; navigation still works when storage is unavailable. Improve the shared starter
first, then re-copy it here; keep only the empty `const items=[];` and placeholder header text.

Copy it into the owning task's temporary folder from the worktree root (replace `<id>`):

```powershell
New-Item -ItemType Directory -Force '.tmp/tasks/<id>/figure-review'
Copy-Item dev/templates/figure-comparison.html '.tmp/tasks/<id>/figure-review/comparison.html'
```

Edit the copy's `<title>`, `<h1>`, the header's project name, location and rendered date, and the
provenance paragraph under "Plot information and render provenance". Replace `const items=[];` with a JSON
array of figure records. Use JSON serialization when generating the page programmatically; do not
interpolate unescaped filenames or captions into JavaScript. Escape `<` as `\u003c` in embedded JSON to
prevent a caption containing `</script>` from terminating the script element.

Example record (image URLs resolve relative to the copied HTML file):

```javascript
const items=[{
  "label": "CHIRPS – Spell",
  "existing_path": "C:/project/data/climate/historical/chirps/plots/spell.png",
  "after_path": "revised/plots/spell.png",
  "before": "before/plots/spell.png",
  "after": "revised/plots/spell.png",
  "before_folder": "file:///C:/project/data/climate/historical/chirps/plots/",
  "after_folder": "revised/plots/",
  "before_bytes": 412345,
  "after_bytes": 398765,
  "version": "revision-1",
  "dpi": 600,
  "after_label": "revised",
  "before_caption": "Caption for the existing figure.",
  "after_caption": "Caption for the revised figure.",
  "before_edges": [0.12, 0.95],
  "after_edges": [0.14, 0.95]
}];
```

- `existing_path` and `after_path` identify the source files; the cards show their file names, and
  `existing_path` (or an explicit `id`) keys the remembered figure.
- `*_folder` fill each card's "Open folder" link; `*_bytes` appear in the metadata line.
- `version` refreshes the revised image after rerendering. `after_label` names the revised version in
  headings (default "Revised").
- Caption and edge fields are optional. Edge pairs are the plotting area's left and right positions as
  fractions of the image width and align captions to it; they default to the full width.
- `dpi` defaults to 600 and must be positive; use the same export dpi for both versions of a pair. The
  overview scale uses a 140 × 80 mm reference canvas, keeping smaller figures smaller and text sizes
  proportional. Zoom 1× means this overview scale, not actual screen millimetres.

Unequal canvases default to side by side; wipe needs "Allow aligned wipe" under Options, aligns top-left
corners at a common scale and pads the smaller image without stretching it. Keyboard shortcuts, listed
under Options, apply when focus is outside controls: Left/Right navigate, W switches mode, +/− zoom,
0 resets, and holding E in wipe mode peeks at the existing image. The layout stacks the pair on narrow
screens.

Keep figure images and populated pages untracked in scratch. Preserve scientific style decisions in
[toolbox-plot-style.md](../reference/toolbox-plot-style.md). This template is a development aid and is not
part of workflow outputs.
