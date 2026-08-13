# Driver premise-verification — round 2

Facts checked against the repository by the driver (not by either reviewer),
recorded so the v3 author does not have to re-derive them. Every line below was
run against `C:/Users/taner/workspace/blueearth_cst` on 2026-08-13.

## Tie-break: the reviewers DISAGREE on `gpt-2` — GPT is right

Fable's round-2 regression check marks `gpt-2` (the values-used record)
**resolved**. GPT's marks it **not resolved**. Primary evidence settles it in
GPT's favour, and v3 must treat it as UNRESOLVED.

`blueearth_cst/model/build_wflow_model.py:237-268` does not hand hydromt the
template's values. It mutates them per step:

```python
for name, configured in steps:
    kwargs = configured.copy()
    if name == "setup_rivers":
        kwargs.pop("hydrography_fn", None)
        kwargs.pop("river_geom_fn", None)
        model.setup_rivers(hydrography_fn=_p1_hydrography(maps), river_geom_fn=rivers, **kwargs)
    elif name == "setup_lulcmaps":
        source_name = _p1_source(maps, "lulc_source")
        kwargs.pop("lulc_fn", None)
        kwargs.setdefault("lulc_mapping_fn", f"{source_name}_mapping_default")   # DERIVED
        model.setup_lulcmaps(lulc_fn=maps["land_cover"].rename("landuse"), **kwargs)
    elif name == "setup_laimaps":
        kwargs.pop("lai_fn", None)
        ...
```

So serializing `read_parameter_steps()` output records the **input template**,
which is what R3 explicitly does not want. `lulc_mapping_fn` in particular is
*derived at call time* and appears in no file on disk.

The decisive illustration is that function's own comment: until 2026-08-13 the
mapping source was taken from the template's `lulc_fn`, so
`shared.basin.spatial_sources.lulc: corine` produced CORINE land cover read
through `vito_mapping_default` — "Wrong numbers, not a missing setting." A
values-used record that serializes the template would NOT have caught it,
because the template looked correct throughout. This is the case the record
exists for.

## Verified: Fable ext2-4 — the embedding carriers do not exist

- WF1's evaluation terminal is `{basin_dir}/evaluation/performance_metrics.csv`
  (`Snakefile_model_creation:379`, `:931`). There is no "evaluation summary".
  It is **not** in `dev/baseline/manifest.json`.
- WF3's terminal is `experiments/experiment/results/q_indicators.csv`, and it
  **IS** baseline-fingerprinted (confirmed present in `dev/baseline/manifest.json`).
  There is no "results metadata" artifact.

Consequence: embedding a digest into WF3's existing CSV changes a fingerprinted
target and falsifies v2's "no baseline re-record" claim. New declared sidecar
outputs avoid that; they must be named, added to the tree inventory, and costed.

## Verified: Fable ext2-3 — production has no git checkout

`Dockerfile` lines 20-23 `ADD src src` and each Snakefile individually. There is
no `.git` in the image. (`git` appears at line 34 only as an apt build
dependency.) So `git rev-parse HEAD` inside a deployed container yields nothing,
and `scripts/run_workflows.py:424-433` (`_git_metadata`) already returns
`{"commit": None, "dirty": None}` on failure — the degraded path exists in code
and is unspecified in the design.

## Verified: GPT ext2-6 — `toolbox.version` has no source

`pyproject.toml` states it is "DELIBERATELY NOT a packaging manifest: there is
no `[build-system]` and no `[project]` table". There is no `__version__`.
Any `toolbox.version` field must define its derivation or be dropped.

## Verified: both reviewers' blocking finding — X.01 rerun triggers

`Snakefile_model_creation` rule `snapshot_config` declares:

```
params:
    data_catalogs = DATA_SOURCES,
    workflow_name = "model_creation",
    config_dir  = f"{project_dir}/config",
    effective_config = config,
    advanced_settings = ADVANCED_SETTINGS,
```

Toolbox identity is absent from inputs and params, so a commit change alone
cannot re-fire the rule.

## Verified: GPT ext2-2 — v2's claim about the current digest was false

`provenance.py:137-141` — `effective_config_document` returns
`{"schema_version": 1, "project_config": config, "advanced_settings": advanced_settings}`.
Today's digest **does** include `advanced_settings`. v2 §5.4's statement that
`effective_config_sha256` is "unchanged in meaning from today" is wrong, and v3
must state where advanced settings sit in the two-digest split.

## Verified in round 1, still standing

- `Snakefile_climate_experiment:348` guards `"project"`, `"shared.basin"`,
  `"workflows.model_creation"`; `:480-495` reads `wflow_outvars` from
  `workflows.model_creation`.
- `copy_config_files.py:81` copies with `source_path.name` (collision-prone);
  `:144` used hash-prefixed names in the bundle.
- `dev/baseline/manifest.json` fingerprints exactly three `snake_config_*`
  copies, including `experiments/experiment/config/snake_config_climate_experiment.yml`.
- `tests/test_snapshot_config_rules.py:51,67` assert the bundle wiring.
- `README.md` ~170-186 documents the bundle and `referenced-files.json`.
- `short_digest` / `snapshot_bundle_digest` have call sites only in code the
  change removes.
