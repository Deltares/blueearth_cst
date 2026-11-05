# Naming conventions

Last updated: 2026-11-05

Naming rules for new identifiers, rules and files in `blueearth_cst`. `MUST`, `SHOULD` and `MAY` have
their usual normative weight.

Existing names may remain until a substantive change needs them. Do not rename only for style. A
contract-surface rename needs a migration note (§7).

Upstream and established BlueEarth contracts keep their prescribed spelling (§6).

## 1. Universal rules

- Use snake_case for variables, functions and modules (MUST). File naming depends on class (§8).
- Lowercase acronyms inside identifiers (MUST): `cmip6_models`, `era5_orography`, `csdms_name`.
- Use `UPPER_SNAKE_CASE` only for true constants: fixed, non-config-derived values or lookup tables
  never reassigned or mutated at runtime (MUST). Config-derived settings stay lowercase.
- Verbs for functions, nouns for variables and data (SHOULD).

## 2. Per language

**Python:** PEP 8 snake_case variables, functions and modules; PascalCase classes; uppercase true constants.

**R:** snake_case, not `dot.case`; verb-noun functions such as `read_climate_data`.

**Snakemake:** snake_case rule names (MUST); action rules use `verb_noun` (SHOULD). Non-action targets
such as `rule all` may use a noun. See §8b.

**YAML:** follow the consuming contract, including for generated files:

- BlueEarth-owned config sections `project`, `basin`, `climate`, `model` and `workflows.<name>` use
  snake_case keys and lowercase booleans `true` / `false` for new keys. Existing uppercase booleans remain.
- YAML under an upstream schema preserves its spelling (MUST), including weathergenr
  (`warm.signif.level`), HydroMT and Wflow parameter names, and HydroMT data catalogs.

## 3. Path-identifier suffix (`_path` canonical)

Use `_path` for a file-path string: `region_path`, `forcing_path`, `csv_path` (MUST). Existing `_fn`,
`_fid` and `_file` names may remain; do not introduce them or rename them without a migration note.

## 4. Snakemake wildcards (stable vocabulary)

Cross-Snakefile wildcards MUST use this vocabulary. Update it when adding one.

| Wildcard | Status | Meaning |
| --- | --- | --- |
| `model` | active | climate model id (CMIP6 model name) |
| `scenario` | active | climate scenario (`historical`, `ssp245`, …) |
| `horizon` | active | future horizon name (`near`, `far`) |
| `rlz_num` | active | weather realization number (`1..rlz_count`) |
| `st_num` | active | stress-test combination: `1..stress_test_count` perturbed; `0` is the reserved unperturbed baseline (`st_0`), also run through Wflow |
| `member` | reserved (CMIP ensemble) | ensemble member id (`r1i1p1f1`, …). Config-only today; becomes a wildcard if per-member rules are added |

Filename and catalog-key member tokens use `st_`: `st_<m>.csv` and
`rlz_<n>_st_<m>.{nc,csv,toml,log}`. `st_0` is the baseline; `rlz_` remains the realization token.

Pad member indices from their own counts so lexical order matches run order: `st_01 … st_12` for
12 points, `st_001` above 99, and no padding below 10. `rlz_` and `st_` pad independently.
`snake_utils.index_width` defines width; `member_index_regex` builds `wildcard_constraints` so an
unpadded name raises `MissingRuleException`. Keep that regex anchor-free: Snakemake embeds it in the
whole path regex, where `$` would bind to the path end and void the condition.

WF3 generation rules key files on scenario `run_id`, not `rlz_` or `st_`.
`perturb_climate_realizations` constrains `run_id` to perturbed plan rows, preventing a second producer
of a root.

`cst` also appears in the package name and WF2 netCDF provenance attributes (`cst_calendar`,
`cst_raw_digest`, `cst_source_paths`). These attributes mean “written by CST”; they are not member tokens.

## 5. Suffix vocabulary — path vs. object

A suffix identifies either a filesystem path or a loaded object, never both.

**Paths:** `_dir` for a directory (`project_dir`); `_path` for a file (`region_path`).

**Loaded objects:** `_ds` for xarray Dataset, `_df` for pandas DataFrame, `_gdf` for GeoDataFrame,
and `_cfg` for a parsed config dict. Use `project_cfg`, `shared_cfg` and `my_cfg`; `my_cfg` denotes the
Snakefile's own `workflows.<name>` section. Do not invent a per-workflow variant.

**Extension suffixes** (`_nc`, `_csv`, `_yml`, `_png`) are for Snakemake `input:` and `output:`
labels mirroring a product (`climate_nc`, `st_csv`). New Python code uses `_path` for the string
or an object suffix for loaded data. Existing labels may remain.

**Deprecated path suffixes** (grandfathered; not for new code): `_fn`, `_fid`, `_file` → `_path`.

**`_rule` denotes a shared Snakemake rule definition.** The helper returns a frozen dataclass holding
`script`, `inputs`, `outputs` and `params`; `message`, `log` and `benchmark` stay workflow-local.
Name both function and dataclass accordingly: `region_rule` → `RegionRule`.

Do not spell this `_contract`: this repo reserves "contract" for interchange surfaces
(the engine-interface references and `SPATIAL_CONTRACT_VERSION`).

## 6. Domain identifiers — three tiers

Apply three tiers of domain identifiers. Do not normalize them casually.

**Tier 1 — opaque upstream identifiers. Preserve verbatim; no local rename path, not even a migration note.**

- Wflow/CSDMS names consumed by hydromt_wflow, such as
  `land_surface__evapotranspiration_volume_flux`.
- HydroMT data-catalog *schema* — adapter fields and structure.
- CMIP model IDs (`NOAA-GFDL/GFDL-ESM4`, `INM/INM-CM5-0` — keep hyphens, slashes, mixed case).
- weathergenr R function names.

**Tier 2 — established BlueEarth contracts. Grandfather; rename only with a migration note (§7).**

- User-facing table labels: `BFI`, return-period tokens such as `10yr` / `2yr` inside an indicator name.
- HydroMT catalog *source names* (`era5`, `merit_hydro`,
  `cmip6_<model>_<scenario>_<member>`): BlueEarth lookup keys, distinct from the tier-1 schema.
- User-facing Wflow output *labels* mapped to CSDMS names in `setup_gauges_and_outputs.py`, such as
  `actual evapotranspiration` and `groundwater recharge`.
- Cross-tool scientific variable stems `precip` and `temp`, used by every producer.

Preserve schema aliases at their adaptation seams: CMIP6 `tas`/`pr` are renamed by
`data_adapter.rename`; Wflow `[output.csv]` uses `Q`/`P`; `WFLOW_VARS` displays `precipitation`;
weathergenr uses `temp_delta`/`precip_mean_factor`.

**Tier 3 — new local identifiers.** Follow §1 unless an external schema dictates spelling.

## 7. Rename only with a migration note

Renaming any of these requires a migration note with an old → new mapping, affected consumers and
checks. Keep the note with the change's task or retained development record, following the
[record lifecycle](dev-record-lifecycle.md):

- `rule all` output filenames (baseline manifest contract).
- Snakemake rule identifiers: CLI targets (`snakemake <rule> -s …`, `--forcerun <rule>`) and
  references in docs and comments.
- `rule all` output-table columns: each header is its own tier-2 contract, independent of its filename.
- Checked-in example config keys (user-facing).
- HydroMT data-catalog source names in `config/catalogs/*.yml` (§6 tier 2).
- Test fixture paths read by `tests/conftest.py`, `dev/scripts/check_baseline.py`, or other scripts.

Tier-1 identifiers (§6) are not renameable at all, so they are omitted here.

An internal mapping records the contract change. A user guide is needed only if users must act:

| Class | Location | Required? | Audience |
| --- | --- | --- | --- |
| Internal rename record | With the task or retained development record | Required for every rename listed above | Old → new mapping, consumers and checks |
| User-facing migration guide | Relevant guide under `docs/` | Write when users must act | Existing installs and project folders |

Do not create a user-facing guide when no user action is needed.
Use a descriptive kebab-case name for a new standalone note, such as `migration-figure-names.md`.

Scientific abbreviations such as `T2`, `T10` and `BFI` remain valid in config keys and data labels.
This exception does not extend to generated filenames (§8).

## 8. File naming by class

Different file classes follow different conventions; this guide does not unify them.

| File class | Convention | Examples |
| --- | --- | --- |
| Python modules / R scripts | snake_case | `prepare_climate_data_catalog.py`, `generate_weather.R` |
| Snakemake entry points | `<verb>_<noun>.smk` | `build_model.smk`, `analyze_climate.smk` |
| Markdown planning docs under `dev/` | kebab-case | `naming-conventions-design.md` |
| Standard root-level files | upstream | `CLAUDE.md`, `README.md`, `LICENSE` |
| Config / data / catalog YAML | tool contract | `project_config_rapid.yml`, `deltares_data.yml` |
| Outputs in `project_dir/` | `snake_case`; exemptions below | `q_group_indicators.csv`, `inmaps_run_01.nc` |

Do not rename existing `dev/` docs.

### Generated outputs under `project_dir/`

Locally minted files and directories use lowercase `snake_case`: no hyphens, capitals or spaces.
The two exceptions below preserve upstream names and domain labels.

Adopt one `_engine/` per cohesive output unit, directly beneath that unit's root. A cohesive unit is
a set of products managed and consumed together, such as `data/climate/projections/cmip6/`.
Keep its internal settings, analysis records and machine plot manifests in that single `_engine/`;
do not create additional `_engine/` folders beneath its `analysis/` or `plots/` directories.
This rule does not imply one project-wide `_engine/`: independently managed output units retain
their own engine directory. Document the base directory used to resolve relative paths in each record.

Generated graphics use `plots/`. Readable captions belong to `plots/figure_captions.md`, while
user-facing JSON data and provenance remain with their data products. The
[figure naming guide](../../blueearth_cst/plotting/docs/figure-naming.md) owns shared rules and workflow
profiles. WF2's current names remain in `projection_figures.FIGURE_FILENAMES` until its profile is added.

WF2 uses `data/climate/projections/cmip6/_engine/` for `stage_b_settings.json`, `analysis_record.json`
and `figure_captions.json`; manifest figure paths resolve relative to the sibling `plots/` directory.
Existing WF0 nested engine paths await explicit migration.

1. **Upstream names pass through.** Preserve engine filenames (`wflow_sbm.toml`, `staticmaps.nc`,
   `instates.nc`, `hydromt_data.yml`) and path identifiers such as `NOAA-GFDL/GFDL-ESM4` verbatim.
2. **Config keys and data labels are outside this rule.** Labels such as `BFI` and `T10` keep their
   domain spelling (§7).

This rule applies only to generated output names under `project_dir/`. `dev/` Markdown stays
kebab-case, Python modules snake_case, and standard root files keep their established names.

## 8b. Rule naming — `<verb>_<noun>`, verb first, always

Action rules use `<verb>_<noun>`; aggregate targets such as `all` may use a noun (§2).
Choose the verb by action class from this table.

| Verb | Action class |
| --- | --- |
| `fetch_` | acquire from an external source |
| `extract_` | subset or derive from a larger source already present |
| `delineate_` | derive a catchment boundary from hydrography and an outlet |
| `prepare_` | **compute or assemble** something a later rule needs |
| `build_` | construct a model from inputs |
| `add_` | mutate an existing model in place by adding **data** (a hydromt `update`) |
| `declare_` | change what an engine will **emit**, adding no model data |
| `write_` | **emit a record or index** — the emission *is* the work |
| `generate_` | stochastic or synthetic production |
| `downscale_` | resolution transform |
| `perturb_` | apply a climate perturbation to an existing series |
| `run_` | invoke an external engine |
| `reduce_` | **intermediate** aggregation that feeds a later rule |
| `produce_` | shared preparation and publication consumed by more than one workflow |
| `derive_` | compute a workflow's **terminal product** from reduced inputs |
| `plot_` | render a figure |
| `check_` | validate, fail loud |
| `publish_` | **validate a finished product and mark it ready** — the ready marker is written last |
| `claim_` | reserve a durable identity before anything is written under it |
| `snapshot_` | freeze or copy inputs for provenance, so a later step cannot drift from them |
| `gather_` | merge parts |

Use these distinctions when choosing a verb:

- **`reduce_` vs `derive_`:** both aggregate inputs, but `reduce_` feeds another rule while
  `derive_` produces the terminal product (`derive_change_factors`, `derive_system_indicators`).
- **`prepare_` vs `write_`:** use `prepare_` when computation or assembly remains beyond file writing;
  use `write_` when producing the record or index is the work. Downstream use does not decide the verb:
  `write_climate_data_catalog` remains `write_` because emitting the catalog is the work.

Use full nouns except established `gcm`, `cmip6`, `wflow`, `rlz` and `st`. Do not introduce ad hoc
contractions such as `weagen` or `proj`; existing `proj` names await their own migration. Put
qualifiers after the noun and spell them out.

Add a verb only for a distinct action class with a rule that uses it.

Check the verb against the rule's script or shell body when work moves; grammatical naming alone can
misdescribe the action.

## 9. Rule numbering (`W.NN` reference scheme)

Each rule in the five workflow definitions has a `W.NN` reference number. `W` is the workflow ID:
`0` analyze_climate, `1` build_model, `2` analyze_projections, `3` generate_scenarios or
`4` simulate_system. `NN` is its zero-padded position in logical order: data, build, run, records.

It exists in exactly two places:

- A comment header above each rule, such as `# 1.06  build_wflow_model`.
- **The `log:` / `benchmark:` filename prefix** — `1.06_build_wflow_model.log` or `.tsv`.
  Managed launches place parts under `logs/<execution>/_parts/wfN-<child-id>/{logs,benchmarks}/`.
  For wildcard rules the prefix goes on the subdirectory (`4.05_run_wflow_simulations/batch_{b}.log`).
  The `W` digit keeps rule labels disambiguated and sorts them by workflow then step within each namespace.

Numbering conventions:

- **Logical order**, with gaps after removals and letter suffixes for insertions; do not close gaps by
  renumbering surviving rules. Target rules (`all`, `simulations_only`, `simulations_and_indicators`)
  carry no number: they aggregate work rather than execute a pipeline step.
- Every dependency points from a lower to a higher number, including `ancient()` inputs.
  `ancient()` suppresses a timestamp rerun trigger, not the DAG edge.

Rule numbers have been reused. Interpret an older `W.NN` citation against the Snakefile revision it
describes (`git show <commit>:<snakefile>`); do not read it as a current rule ID or rewrite retained evidence.

Rules:

- Rule *identifiers* are not numbered (MUST). Snakemake identifiers cannot contain a leading digit or
  dot, and they are CLI targets. Put `W.NN` only in comments and log or benchmark paths.
- `W.NN` is a reference and reading aid, not execution order. Snakemake uses the DAG; independent
  branches can run concurrently. Low-to-high means a lower-numbered rule cannot depend on a higher one.
- Source definition order may differ from numbering because module-level code depends on placement.
  `LOG_RULES` must remain in number order (`tests/test_log_rules_contract.py`): it controls workflow
  log merge order.
- In prose and commits use “Rule 1.3”; use padded `1.03` in sortable filenames.
- **Do not renumber to insert a rule.** Use a letter suffix (`1.09b`) until a deliberate workflow-wide
  migration. Numbers appear in `LOG_RULES`, log and benchmark paths, `rule_banner`, comments and prose.
  An omitted `LOG_RULES` label silently drops its log section. A suffix retains sorting order:
  `"1.09" < "1.09b" < "1.10"`.
- "Rule 1.5" decimals are review shorthand for *talking about* an insert, never a permanent identifier.

## 10. Examples

> Illustrative names only; existing identifiers do not need a style-only rename.

| Instead of | Use | Reason |
| --- | --- | --- |
| `config_fn` | `config_path` | Canonical path suffix |
| `stats_nc` (a path) | `stats_path` | Path / object distinction |
| `stats_nc` (a Dataset) | `stats_ds` | Path / object distinction |
| `ST_NUM` | `stress_test_count` | Config-derived setting, not a true constant |
| `RLZ_NUM` | `rlz_count` | Same |
| `st_num2` | `st_num` | Stable wildcard vocabulary |
| `cmip6Models` | `cmip6_models` | Lowercase acronym + snake_case |
| `TRUE` / `FALSE` (BlueEarth YAML) | `true` / `false` | Lowercase YAML booleans |
