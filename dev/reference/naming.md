# Naming conventions

Prescriptive style guide for naming identifiers and files in `blueearth_cst`. `MUST` / `SHOULD` / `MAY` carry their usual normative weight.

**Grandfathered today, applied tomorrow.** This guide governs *new* code. Existing non-conforming names stay until the milestone that owns them refactors them — do not rename an identifier just to conform. Renaming a *contract* surface needs a migration note (§7).

**Local style yields to external contracts.** Identifiers governed by an upstream tool or an established BlueEarth contract follow those contracts, not the rules here (§6).

## 1. Universal rules

- snake_case for variables, functions, and modules (MUST). File names are governed by class — see §8, not this rule.
- Lowercase acronyms inside identifiers (MUST): `cmip6_models`, `era5_orography`, `csdms_name` — never `CMIP6Models`.
- `UPPER_SNAKE_CASE` only for true constants: fixed, non-config-derived values or lookup tables never reassigned or mutated at runtime (MUST). Config-derived run settings are lowercase.
- Verbs for functions, nouns for variables and data (SHOULD).

## 2. Per language

**Python** — PEP 8: snake_case variables / functions / modules; PascalCase classes; `UPPER_SNAKE_CASE` for module-level true constants only.

**R** — snake_case, not `dot.case` (aligns with tidyverse and weathergenr). Verb-noun functions (`read_climate_data`, not `climate_data`).

**Snakemake** — rule names snake_case (MUST); `verb_noun` for action rules (SHOULD); noun-only is acceptable for non-action rules like `rule all` (MAY). Full grammar: §8b.

**YAML** — discriminate by the *consuming contract*, never by authorship or by whether the file is checked in or generated:

- BlueEarth-owned configs consumed locally — the `project` / `shared` / `workflows.<name>` snake config — use snake_case keys and lowercase booleans `true` / `false` (MUST for new keys). Existing `TRUE` / `FALSE` are grandfathered.
- Any YAML consumed under an upstream schema preserves the upstream spelling (MUST), **even when BlueEarth generates the file**: weathergenr (`warm.signif.level`), HydroMT / Wflow parameter names, and HydroMT data catalogs.

## 3. Path-identifier suffix (`_path` canonical)

New code MUST use `_path` for a variable holding a file-path string: `region_path`, `forcing_path`, `csv_path`. The deprecated suffixes `_fn`, `_fid`, `_file` are grandfathered — do not use them in new code, and rename an existing one only with a migration note.

## 4. Snakemake wildcards (stable vocabulary)

Wildcards used across Snakefiles MUST come from this list. Adding one requires updating this file in the same commit.

| Wildcard | Status | Meaning |
| --- | --- | --- |
| `model` | active | climate model id (CMIP6 model name) |
| `scenario` | active | climate scenario (`historical`, `ssp245`, …) |
| `horizon` | active | future horizon name (`near`, `far`) |
| `rlz_num` | active | weather realization number (`1..rlz_count`) |
| `st_num` | active | stress-test combination: `1..stress_test_count` perturbed; `0` = reserved unperturbed baseline (`st_0`), run through Wflow only when `run_historical` sets `ST_START = 0` |
| `member` | reserved (CMIP ensemble) | ensemble member id (`r1i1p1f1`, …). Config-only today; becomes a wildcard if per-member rules are added |

**The member token in filenames and catalog keys is `st_`**, the same word as the wildcard: `st_<m>.csv` and `rlz_<n>_st_<m>.{nc,csv,toml,log}`, with `st_0` the reserved baseline. `rlz_` stays as-is — it abbreviates a correct term and collides with nothing.

**Member indices are zero-padded to a width derived from the count**, so lexical order matches run order: `st_01 … st_12` for a twelve-point grid, `st_001` past ninety-nine, no padding below ten. `rlz_` and `st_` pad independently, each from its own count. `snake_utils.index_width` owns the width; `member_index_regex` builds the matching `wildcard_constraints` so an unpadded name raises `MissingRuleException` rather than routing silently. That regex MUST stay anchor-free: Snakemake embeds a constraint in the whole path's regex, so a `$` inside one binds to the end of the path and silently voids the condition.

Only `perturb_climate_realization` carries a rule-local `wildcard_constraints` barring the all-zeros baseline, so it cannot become a second producer of `st_0`. Downstream rules keep the default match that admits `0`.

**Three different things spell themselves `cst`**, so a bare `cst_` grep is never the right tool: the package `blueearth_cst`, the historical member token (now `st_`), and the WF2 netCDF provenance attributes in `blueearth_cst/projections/` (`cst_calendar`, `cst_raw_digest`, `cst_source_paths`, …), which mean "written by CST" and are part of WF2's on-disk output.

## 5. Suffix vocabulary — path vs. object

A suffix means EITHER a filesystem path OR a loaded object, never both.

**Paths:** `_dir` (directory — `project_dir`, `basin_dir`), `_path` (file, any extension — `region_path`, `catalog_path`).

**Loaded objects:** `_ds` (xarray Dataset), `_df` (pandas DataFrame), `_gdf` (GeoDataFrame), `_cfg` (parsed config dict). `project_cfg`, `shared_cfg` and `my_cfg` are the blessed idiom — `my_cfg` for a Snakefile's own `workflows.<name>` section, uniform across all Snakefiles. Use it; do not invent a per-workflow variant.

**Extension suffixes** (`_nc`, `_csv`, `_yml`, `_png`) are reserved for Snakemake `input:` / `output:` labels that mirror a file product (`climate_nc`, `st_csv`, `output_png`). New Python code uses `_path` for the string or `_ds` / `_df` for the object. Existing non-conforming labels are grandfathered.

**Deprecated path suffixes** (grandfathered; not for new code): `_fn`, `_fid`, `_file` → `_path`.

**`_rule` — a shared Snakemake rule definition.** A helper returning a frozen dataclass that holds a rule's `script`, `inputs`, `outputs` and `params` — everything content- or execution-determining, leaving only `message` / `log` / `benchmark` workflow-local — so one rule can be splatted into more than one Snakefile without the declarations drifting. Function and dataclass both carry it: `region_rule` → `RegionRule`, `climate_store_rule` → `ClimateStoreRule`, `spatial_units_rule` → `SpatialUnitsRule`.

Do not spell this `_contract`: this repo reserves "contract" for interchange surfaces (`dev/reference/contracts/`, `SPATIAL_CONTRACT_VERSION`).

## 6. Domain identifiers — three tiers

Domain identifiers carry different kinds of contract, so treat them in three tiers rather than one flat "external" bucket. None are normalized casually.

**Tier 1 — opaque upstream identifiers. Preserve verbatim; no local rename path, not even a migration note.**

- Wflow / CSDMS variable names consumed by hydromt_wflow (e.g. `land_surface__evapotranspiration_volume_flux`).
- HydroMT data-catalog *schema* — adapter fields and structure.
- CMIP model IDs (`NOAA-GFDL/GFDL-ESM4`, `INM/INM-CM5-0` — keep hyphens, slashes, mixed case).
- weathergenr R function names.

**Tier 2 — established BlueEarth contracts. Grandfather; rename only with a migration note (§7).**

- User-facing table labels: `BFI`, return-period tokens such as `10yr` / `2yr` inside a metric name.
- HydroMT data-catalog *source names* (`era5`, `merit_hydro`, `cmip6_<model>_<scenario>_<member>`) — BlueEarth-minted lookup keys forming a catalog-lookup contract. Their schema is tier 1.
- User-facing Wflow output *labels* mapped to CSDMS names in `setup_gauges_and_outputs.py` (`actual evapotranspiration`, `groundwater recharge`) — display names, not upstream IDs.
- Cross-tool scientific variable names: `precip` and `temp` are the canonical stems, and every producer uses them.

Aliases that look like drift but are NOT — each owned by an external schema and adapted at a named seam, so leave them alone: `tas` / `pr` (CMIP6, renamed by the catalog's `data_adapter.rename`), `Q` / `P` (wflow `[output.csv]` headers), `precipitation` (a `WFLOW_VARS` display label, tier 2), and weathergenr's `temp_delta` / `precip_mean_factor` (tier 1).

**Tier 3 — new locally owned scientific identifiers.** Follow local style (§1) unless an explicit external schema dictates a spelling.

## 7. Rename only with a migration note

Renaming any of these requires a `dev/<milestone>/migration_<topic>.md` note listing the old → new mapping:

- `rule all` output filenames (baseline manifest contract).
- **Snakemake rule identifiers.** They are the CLI target surface (`snakemake <rule> -s …`, `--forcerun <rule>`) and are referenced across `docs/`, `dev/reference/` and Snakefile comments, so a rename breaks a command someone has in their shell history.
- **Column labels in `rule all` output tables.** A header is a tier-2 contract in its own right (§6), separately from the filename carrying it: a consumer that survived a file rename can still break on a header.
- Checked-in example config keys (user-facing).
- HydroMT data-catalog source names in `config/*.yml` (§6 tier 2).
- Test fixture paths read by `tests/conftest.py`, `dev/scripts/check_baseline.py`, or other scripts.

Tier-1 identifiers (§6) are not renameable at all, so they are omitted here.

**Two artifact classes, distinguished.** A rename record and a user-facing migration guide are different documents with different audiences:

| Class | Location | Required? | Audience |
| --- | --- | --- | --- |
| Internal rename record | `dev/<milestone>/migration_<topic>.md` | **Required** for every rename listed above | Whoever implements or audits the milestone: the old → new table, the machinery to update, the gate evidence |
| User-facing migration guide | `docs/migration-<milestone>.md` | **Optional** — write one only when users must act | Someone with an existing install or project folder |

A milestone that changes nothing a user must act on ships the internal record and no guide.

**The mandated `migration_<topic>.md` filename overrides §8's kebab-case rule for `dev/` markdown.** The form is fixed by this section.

**Scientific abbreviations are allowed in config keys and column/row labels** even though they break the acronym-lowercase rule: return-period `T2` / `T10`, `BFI`. These are established domain vocabulary; keep them. The carve-out covers labels and config keys only — §8's generated-outputs rule governs filenames.

## 8. File naming by class

Different file classes follow different conventions; this guide does not unify them.

| File class | Convention | Examples |
| --- | --- | --- |
| Python modules / R scripts | snake_case | `prepare_climate_data_catalog.py`, `generate_weather.R` |
| Snakemake entry points | `<verb>_<noun>.smk` | `build_model.smk`, `analyze_climate.smk` |
| Markdown planning docs under `dev/` | kebab-case | `naming-conventions-design.md` |
| Standard root-level files | upstream | `CLAUDE.md`, `README.md`, `Dockerfile`, `LICENSE` |
| Config / data / catalog YAML | tool contract | `snake_config_rapid.yml`, `deltares_data.yml` |
| Generated outputs under `project_dir/` | lowercase `snake_case`, two exemptions below | `q_indicators.csv`, `basin_indicators.csv`, `inmaps_rlz_1_st_2.nc` |

Do not rename existing `dev/` docs.

### Generated outputs under `project_dir/`

Locally minted file and directory names are lowercase `snake_case`: no hyphens, no capitals, no spaces. Two exemptions, both narrow, both stated so a reader does not "correct" them:

1. **Upstream-owned names pass through verbatim.** Engine-mandated filenames (`wflow_sbm.toml`, `staticmaps.nc`, `instates.nc`, `hydromt_data.yml`) and upstream identifiers embedded in a path — CMIP model IDs such as `NOAA-GFDL/GFDL-ESM4`, carrying hyphens, slashes and mixed case — are never normalized. These are §6 tier-1 identifiers.
2. **Config keys and data labels are out of reach.** The rule governs file and directory names only. Column and row labels (`BFI`, `T10`) and config keys keep their domain spelling (§7).

**The rule is CLASS-SCOPED and must not be generalised.** It governs generated output names under `project_dir/` and nothing else: `dev/` markdown stays kebab-case, Python modules stay snake_case because they must be importable, and root-level files keep their upstream names. Reading it as a repo-wide sweep would rename documents this guide explicitly protects.

## 8b. Rule naming — `<verb>_<noun>`, verb first, always

Every Snakemake rule identifier is `<verb>_<noun>`. The verb comes from this list — **one verb per action class**, so two rules doing the same kind of work read the same. Name a new rule from the table, not by analogy with whichever rule sits above it.

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
| `derive_` | compute a workflow's **terminal product** from reduced inputs |
| `plot_` | render a figure |
| `check_` | validate, fail loud |
| `snapshot_` | copy inputs for provenance |
| `gather_` | merge parts |

Two distinctions are the ones a new rule gets wrong:

- **`reduce_` vs `derive_` splits by POSITION, not by operation.** Both turn many inputs into few outputs. `reduce_gcm_series` feeds a later rule; `derive_change_factors` and `derive_wflow_indicators` each produce their workflow's final answer.
- **`prepare_` vs `write_` splits on where the work is.** The test:

  > **If you deleted the file-writing, would there be work left? Yes → `prepare_`. No → `write_`.**

  Whether a later rule consumes the output is *not* the criterion. `write_climate_data_catalog` is consumed downstream and is still `write_`, because enumerating entries is all it does.

**Nouns are full words.** Only the established domain set abbreviates — `gcm`, `cmip6`, `wflow`, `rlz`, `st` — and those are tier-1/tier-2 identifiers under §6. Ad-hoc contractions (`weagen`, `proj`) are not. Qualifiers are trailing full words, never two-letter suffixes.

**Adding a verb is allowed, and cheaper than a bad name.** The bar is that the action class is genuinely distinct *and* that the verb has a rule using it. A verb in this table with no rule behind it reads as an available option that some rule must already justify.

**Grammar conformance is not body conformance.** A name satisfying `<verb>_<noun>` can still be false — a rule can keep a verb after the work moves elsewhere. Check the verb against the rule's script or shell body, and say which check you ran.

## 9. Rule numbering (`W.NN` reference scheme)

Each rule in the four `*.smk` entry points carries a `W.NN` reference number. `W` is the **workflow id** — `0` analyze_climate, `1` build_model, `2` analyze_projections, `3` run_stress_test — not a position, so ids need not start at 1. `NN` is the zero-padded **position in that workflow's logical order**: data first, then model build, then run, then records.

It exists in exactly two places:

- **A comment header above each rule** — `# 1.07  build_wflow_model — parameterize Wflow-SBM on the spatial foundation`.
- **The `log:` / `benchmark:` filename prefix** — `logs/1.07_build_wflow_model.log`, `benchmarks/_parts/1.07_build_wflow_model.tsv`. For wildcard rules the prefix goes on the subdirectory (`logs/3.15_run_wflow/batch_{b}.log`). All workflows share `project_dir/logs`, so the `W` digit keeps their logs disambiguated and a single `ls logs/` sorts by workflow then step.

Two properties hold:

- **Contiguous** within each workflow, from `W.00` (`rule all`).
- **Every dependency points from a lower number to a higher one**, checked against each rule's `input:` block — **`ancient()` included**. `ancient()` suppresses the timestamp rerun-trigger, not the DAG edge.

**Numbers are REUSED, so a stale reference resolves to a different rule.** Read every `W.NN` in `dev/milestones/`, `dev/decisions/`, `dev/LOG.md` and the dated migration records **as of its date**, and translate through `dev/reference/workflows/rule-index.md` § *What changed*. Do not rewrite those archives to current numbers.

Rules:

- **Rule *identifiers* are NOT numbered** (MUST). Snakemake rule names are Python identifiers (no leading digit, no dot) and are the CLI target surface referenced across docs — a `W.NN` identifier would be both illegal-as-typed and a §7 contract rename. The number lives only in the comment and the log/benchmark path.
- The number is a **reference and reading aid, not execution order** (MUST keep this framing). Snakemake executes from the DAG, so rules on separate branches run concurrently — WF1's `1.11`–`1.13` are parallel leaves and WF3 fans out over `rlz_num × st_num`. Low-to-high means **"cannot depend on"**, not "runs before". Each Snakefile states this in a header comment.
- **Definition order in the file need not match the number.** Module-level code is interleaved between rule blocks and depends on its position, so reordering blocks is a behaviour risk taken for cosmetics. `W.NN` is the rule's place in the workflow, not its offset in the file. `LOG_RULES` *is* asserted to read in number order (`tests/test_log_rules_contract.py`), because that list is the merge order for the workflow log.
- **Reference in prose and commits as "Rule 1.3"** (drop the pad); the padded `1.03` form is for sortable filenames.
- **DO NOT RENUMBER TO INSERT A RULE. Use a letter suffix** (`1.09b`) until the next deliberate sweep, and take the whole workflow in one commit when that sweep comes. Renumbering is a migration, not an edit: the number appears in `LOG_RULES`, in log and benchmark paths, in `rule_banner`, in comment headers and in prose across `dev/`, and it has a silent failure mode — an unlisted `LOG_RULES` label drops its log section without erroring. A letter suffix sorts correctly against the padded numbers (`"1.09" < "1.09b" < "1.10"`), so an inserted rule does not break the `LOG_RULES` ordering assertion.
- "Rule 1.5" decimals are review shorthand for *talking about* an insert, never a permanent identifier.

## 10. Examples

> **Illustrative targets only.** This is not a rename list — existing identifiers are grandfathered until their owning milestone touches them.

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
