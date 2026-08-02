# R02 Naming Conventions Review

**Date.** 2026-05-09.

**Scope.** Review of `dev/milestones/r02/naming-conventions-design.md` against the
current repository, the R01 config-contract plan, and the R3-R6 cleanup
roadmap.

## Summary

The R02 design is useful and well-timed. A short prescriptive naming guide
before R3-R5 will prevent workflow refactors from inventing local conventions
one file at a time. The strongest parts are the grandfathering policy, the
explicit upstream/domain escape hatch, the `_path` direction for new path
variables, and the "do not rename without migration note" rule for baseline
and user-facing surfaces.

I recommend tightening the design before execution in four areas:

1. Clarify file naming by file class. The design says "files snake_case" but
   the design file itself and most dev docs use hyphenated Markdown names.
2. Do not call config-derived workflow dimensions "true constants".
   `RLZ_NUM`, `ST_NUM`, and `DATA_SOURCES` are current globals, but they are
   derived from config and should probably become lowercase in new code.
3. Split suffix rules for file paths vs loaded data objects. `_nc` and `_csv`
   currently risk meaning both "path to file" and "loaded object".
4. Add explicit exceptions for YAML files owned by upstream tools, especially
   `config/weathergen_config.yml`, whose keys intentionally use dotted names
   such as `warm.signif.level`.

## What works well

- **Grandfathering is the right policy.** R02 should not perform renames.
  The repo has baseline-tracked output filenames, copied config snapshots,
  cross-language scripts, and user-facing examples. A pure docs milestone
  avoids accidental churn.
- **The upstream-contract framing is necessary.** Wflow variables, HydroMT
  catalog source names, CMIP IDs, CSDMS names, and weathergenr names should
  remain externally compatible even when they violate local style.
- **`_path` as the new canonical path suffix is a good direction.** The
  current code mixes `_fn`, `_fid`, `_file`, and unlabeled strings. R3-R5 can
  improve readability incrementally without broad renames.
- **The wildcard vocabulary table is valuable.** It gives R3-R5 a target for
  cleaning the existing `st_num` / `st_num2` inconsistency and for avoiding
  new ad hoc wildcard names.
- **The migration-note requirement is exactly the right safeguard.** Rule-all
  outputs, config keys, data catalog entries, and fixture paths should not be
  renamed casually.

## Recommended changes before authoring `naming.md`

### 1. Separate file naming rules by file type

The design currently says:

- Universal case: snake_case for variables, functions, modules, files.
- File names: snake_case for `.py` and `.R`; exceptions include
  `Snakefile_*`, `CLAUDE.md`, `README.*`, `Dockerfile`, `LICENSE`.

But the existing dev docs use hyphenated names:

- `dev/milestones/r01/modularity-contracts-design.md`
- `dev/milestones/r01/modularity-contracts-plan.md`
- `dev/milestones/r02/naming-conventions-design.md`

Recommendation: make the final guide explicit:

- Python modules and R scripts: `snake_case`.
- Snakemake entry points: keep existing `Snakefile_<workflow>` until R6
  decides on workflow layout.
- Markdown planning docs under `dev/`: allow `kebab-case` because that is
  already the dominant milestone-doc convention.
- Config/data/catalog files: governed by existing user-facing and tool-facing
  contracts; rename only with migration note.
- Generated outputs under `project_dir`: not governed by local file-style
  cleanup unless the owning workflow contract is updated.

This avoids the awkward result where the new guide declares its own
`naming-conventions-design.md` filename non-conforming.

### 2. Refine "true constants"

The table lists `RLZ_NUM`, `ST_NUM`, and `DATA_SOURCES` as true constants.
In the current Snakefiles these are module-level globals, but not true
constants: they are derived from config and change per run.

Recommendation: distinguish:

- **True constants:** immutable lookup tables and fixed values, e.g.
  `WFLOW_VARS`, `XDIMS`, `YDIMS`, `VOLATILE_NC_ATTRS`.
- **Config-derived run settings:** lowercase in new code, e.g.
  `rlz_count`, `stress_test_count`, `data_sources_path`,
  `climate_catalog_path`.

Existing `RLZ_NUM`, `ST_NUM`, and `DATA_SOURCES` can remain grandfathered.
For new R3+ code, avoid blessing that pattern.

### 3. Do not let suffixes mean both path and object

The suffix table currently says:

- `_nc` means netCDF file path / dataset object.
- `_csv` means CSV file path / DataFrame.
- `_yml` means YAML file path / dict.

That ambiguity is exactly what the naming guide should remove. A path string
and a loaded object behave very differently.

Recommendation:

- Use `_path` for any filesystem path, regardless of extension:
  `forcing_path`, `summary_path`, `catalog_path`.
- Use `_ds` for xarray datasets.
- Use `_df` for pandas DataFrames.
- Use `_gdf` for GeoDataFrames.
- Use `_cfg` for parsed config dictionaries.
- Reserve extension-like suffixes (`_nc`, `_csv`, `_yml`, `_png`) for
  Snakemake `input:` / `output:` labels only if the label is intentionally
  mirroring a file product.

This keeps new Python code clearer while still allowing current Snakemake
labels such as `climate_nc` and `st_csv` to be grandfathered.

### 4. Expand YAML exceptions

"YAML keys snake_case throughout" is correct for the new R01 snake config,
but not for all YAML in this repo. `config/weathergen_config.yml` is consumed
by weathergenr conventions and currently uses dotted keys:

- `warm.signif.level`
- `warm.sample.num`
- `knn.sample.num`
- `evaluate.model`

HydroMT / Wflow config files also contain upstream method names and
parameter names that should not be locally normalized.

Recommendation: write the rule as:

- BlueEarth-owned YAML keys MUST be snake_case.
- Upstream tool config keys MUST preserve upstream spelling.
- Data catalog source names and adapter fields follow HydroMT's schema.

This also protects future R01 config work from accidentally normalizing
external schemas.

### 5. Include boolean casing in config guidance

Existing YAML uses uppercase `TRUE` / `FALSE` in places; the R01 template
uses lowercase `true` / `false`. YAML accepts both, but a style guide should
pick one for BlueEarth-owned configs.

Recommendation: require lowercase YAML booleans (`true`, `false`) in
BlueEarth-owned configs. Treat uppercase booleans in existing files as
grandfathered until touched.

### 6. Add examples of good names and bad names

Because this guide is meant for future agents and contributors, examples
will be more effective than rules alone. Add a short table:

| Instead of | Use | Reason |
|---|---|---|
| `config_fn` | `config_path` | New path suffix |
| `stats_nc` | `stats_path` or `stats_ds` | Path/object distinction |
| `ST_NUM` | `stress_test_count` | Config-derived setting |
| `st_num2` | `st_num` | Stable wildcard vocabulary |
| `cmip6Models` | `cmip6_models` | Lowercase acronym + snake_case |

Keep the examples sparse so the final guide stays under 250 lines.

### 7. Decide what happens to `dev/conventions-review.md`

The design references `dev/conventions-review.md` as pre-design feedback.
That file currently says review was blocked because `dev/conventions/` did
not exist. After this review, either:

- supersede it with this file and remove the stale blocker reference from
  the final design, or
- update it after `dev/conventions/naming.md` exists.

Leaving a stale "blocked" note as the main reference will confuse future
readers.

## Minor comments

- "Single prescriptive style guide" is a good goal, but the guide should not
  try to own branch/tag/commit naming. The design already keeps those in the
  roadmap, which is correct.
- The `member` wildcard is already a likely future addition because CMIP
  members exist in config. Consider listing it as a reserved-but-currently-
  unused wildcard rather than waiting for a later workflow to invent it.
- The outlet station naming decision in `dev/followups.md` is not purely a
  style issue; it affects output filenames and user interpretation. Keep that
  as an R3 behavioral/contract decision, not a naming-guide-only decision.
- The guide should say that external scientific abbreviations are allowed in
  user-facing output filenames when they are established domain vocabulary,
  e.g. `Qstats.csv`, `Tlow`, `Tpeak`.

## Suggested design edits

Before R02 execution, update `dev/milestones/r02/naming-conventions-design.md` to:

1. Replace "files snake_case" with a file-class table.
2. Reclassify `RLZ_NUM`, `ST_NUM`, and `DATA_SOURCES` as grandfathered
   config-derived globals, not examples of true constants.
3. Split suffix vocabulary into paths (`_path`), data objects (`_ds`, `_df`,
   `_gdf`), configs (`_cfg`), and Snakemake labels.
4. Add YAML exceptions for weathergenr, HydroMT, Wflow, and data catalogs.
5. Add lowercase YAML boolean guidance for BlueEarth-owned configs.
6. Add a compact "instead of / use / reason" examples table.
7. Clarify whether `dev/conventions-review.md` remains a live reference or
   is superseded by this R02 review.

## Bottom line

Proceed with R02, but tighten the guide boundaries before writing
`dev/conventions/naming.md`. The naming guide should be strict for new
BlueEarth-owned code and config keys, but careful not to normalize external
schemas, generated products, or scientific identifiers that are already part
of the workflow contract.

