# R02 — Naming conventions (pre-R3) — design

**Date.** 2026-05-09 (revised 2026-07-19 after internal + GPT-5.6 + Fable reviews).

## Goal

Single prescriptive style guide for naming identifiers and files
across the repo, delivered as `dev/reference/naming.md`. Pure docs
addition. **No code refactoring in R2** — existing names are
grandfathered. R3+ apply the conventions when touching code; R6 may
do bulk file moves under the structural-refactor milestone.

The intent is: when a future workflow, agent, or contributor adds new
identifiers, the style guide tells them what to choose without re-
arguing the convention each time.

## Why now (and why before R3)

R3-R5 each refactor a workflow's scripts and add new identifiers
along the way (helper functions in `snake_utils.py`, new fixtures,
new wildcards, new config keys under R01's sectioned schema). Doing
those refactors against an undocumented convention means each
milestone has to re-decide naming on the fly — and the decisions
accumulate inconsistently. Locking the convention first lets R3-R5
inherit it.

This doc does not refactor existing names. It is the contract for
*new* code from R3 onward.

## Approach

Prescriptive style guide with `MUST` / `SHOULD` / `MAY` voice,
opinionated where the codebase is currently mixed and lenient where
external conventions take precedence. Two important framings:

1. **Local style vs external / established contracts.** The local style
   guide bends for identifiers that carry an external or established
   contract — upstream tool identifiers (Wflow/CSDMS names, HydroMT
   catalog keys, CMIP model IDs, weathergenr functions) and
   BlueEarth-owned-but-established names (`precip`, `temp`, `Qstats`).
   §6 splits these into three tiers; none are normalized casually.
2. **Grandfathered today, applied tomorrow.** Existing names that
   don't conform stay as-is until the owning milestone refactors
   them. R2 itself produces zero code diffs.

## Decisions baked in (from brainstorming + R2 review)

| Topic                          | Decision                                                                |
| ------------------------------ | ----------------------------------------------------------------------- |
| Voice                          | `MUST` / `SHOULD` / `MAY`                                               |
| Universal case                 | snake_case for variables, functions, modules                            |
| Acronyms in identifiers        | Always lowercase (`cmip6_models`, `era5_orography`, `csdms_name`)       |
| True constants                 | `UPPER_SNAKE_CASE` for fixed, non-config-derived values / lookup tables not reassigned or mutated at runtime (`WFLOW_VARS`, `XDIMS`, `YDIMS`, `VOLATILE_NC_ATTRS`). `WFLOW_VARS` is a mutable dict used read-only, so the rule is intent (read-only lookups), not language enforcement; `XDIMS`/`YDIMS` (tuples) and `VOLATILE_NC_ATTRS` (frozenset) are genuinely immutable. |
| Config-derived run settings    | lowercase snake_case for new code (`rlz_count`, `stress_test_count`, `data_sources_path`). Existing `RLZ_NUM`, `ST_NUM`, `DATA_SOURCES` grandfathered. |
| Python                         | PEP 8: snake_case, PascalCase classes, UPPER_SNAKE for true constants only |
| R                              | snake_case (not `dot.case`); aligns with tidyverse + weathergenr        |
| Snakemake rule names           | snake_case; verb_noun for action rules; noun-only acceptable for `rule all` |
| YAML keys (by consuming contract) | snake_case for BlueEarth-owned configs consumed locally (the R01 snake config sections). Discriminate by the *consuming contract*, not authorship. |
| YAML keys (upstream schema)    | preserve upstream spelling for any YAML consumed under an upstream schema — weathergenr's `warm.signif.level`, HydroMT/Wflow parameter names, and HydroMT data catalogs **even when BlueEarth-generated** |
| YAML booleans                  | lowercase `true` / `false` for BlueEarth-owned configs (existing `TRUE`/`FALSE` grandfathered) |
| Path-identifier suffix         | `_path` is canonical for new code; `_fn`, `_fid`, `_file` deprecated    |
| File names by class            | See "File naming by class" section below                                |

## Style guide outline (`dev/reference/naming.md`)

The guide itself gets authored during R2 execution. Section list:

### 1. Universal rules

- snake_case for variables, functions, and modules (MUST). File names
  are governed by class — see §8, not this universal rule.
- Lowercase acronyms in identifiers (MUST).
- `UPPER_SNAKE_CASE` for true constants (MUST).
- Verbs for functions, nouns for variables / data (SHOULD).

### 2. Per-language

**Python**: PEP 8. snake_case for variables / functions / modules.
PascalCase for classes. UPPER_SNAKE_CASE for module-level constants.

**R**: snake_case (not `dot.case`). Aligns with tidyverse style guide
and the weathergenr package conventions. Verb-noun for functions
(`read_climate_data`, not `climate_data`).

**Snakemake**: rule names snake_case (MUST). Verb_noun for action
rules (`build_model`, `add_gauges`, `run_wflow`) (SHOULD). Noun-only
acceptable for non-action rules (`rule all`) (MAY).

**YAML**: snake_case keys for BlueEarth-owned configs. Group under
R01 sections: `project`, `shared`, `workflows.<name>`. Lowercase
booleans (`true` / `false`).

**YAML — upstream tool configs are exempt.** Configs consumed by
external tools preserve upstream key spelling even when it conflicts
with snake_case:

- `config/weathergen_config.yml` uses dotted keys (`warm.signif.level`,
  `warm.sample.num`, `knn.sample.num`, `evaluate.model`) per
  weathergenr conventions.
- HydroMT / Wflow build and update configs use upstream method names
  and parameter names.
- Data catalog source names and adapter fields follow HydroMT's
  schema — including catalogs BlueEarth generates (e.g. the CMIP6
  catalog): the consuming contract governs, not who authored the file.

These are external-API contracts; do not normalize. The discriminator
is the consuming contract (upstream schema vs. BlueEarth-local), never
authorship or whether the file is checked in vs. generated.

### 3. Path-identifier suffix (`_path` canonical)

New code MUST use `_path` for variables holding file path strings:
`region_path`, `forcing_path`, `csv_path`. Deprecated suffixes still
present in the codebase (`_fn`, `_fid`, `_file`) are grandfathered;
do not rename existing usages without a migration note.

Document why `_path` was chosen: explicit, language-neutral, and works
naturally with `pathlib.Path`.

### 4. Snakemake wildcards (stable vocabulary)

Wildcards used across Snakefiles MUST come from this list:

| Wildcard      | Status                  | Meaning                                          |
| ------------- | ----------------------- | ------------------------------------------------ |
| `model`       | active                  | climate model id (CMIP6 model name)              |
| `scenario`    | active                  | climate scenario (`historical`, `ssp245`, ...)   |
| `horizon`     | active                  | future horizon name (`near`, `far`)              |
| `rlz_num`     | active                  | weather realization number (1..rlz_count)        |
| `st_num`      | active                  | stress test combination number. `1..stress_test_count` = perturbed combinations; `0` = reserved unperturbed baseline (`cst_0`), run through Wflow only when `run_historical` sets `ST_START = 0`. |
| `member`      | reserved (CMIP ensemble)| ensemble member id (`r1i1p1f1`, ...). Currently config-only; will become a wildcard if ensemble per-member rules are added. |

Adding a new wildcard requires updating `dev/reference/naming.md`
in the same commit. The current `st_num2` variant in
`Snakefile_climate_experiment` (it admits `0` under `run_historical`,
where `st_num` starts at `1`) is a known inconsistency — fold into
`st_num` during **R5** Snakefile cleanup. `Snakefile_climate_experiment`
is R5 territory; per the roadmap, no milestone touches another's
Snakefile.

### 5. Suffix vocabulary — split paths from data objects

A suffix means EITHER a filesystem path OR a loaded object. Not both.
This is the single biggest readability improvement R3+ can make
incrementally.

**Paths (filesystem strings):**

| Suffix   | Use for                       | Example                                   |
| -------- | ----------------------------- | ----------------------------------------- |
| `_dir`   | directory path                | `project_dir`, `basin_dir`, `exp_dir`     |
| `_path`  | file path (any extension)     | `region_path`, `forcing_path`, `summary_path`, `catalog_path` |

**Loaded data objects:**

| Suffix   | Use for                       | Example                                   |
| -------- | ----------------------------- | ----------------------------------------- |
| `_ds`    | xarray Dataset                | `forcing_ds`, `change_ds`                 |
| `_df`    | pandas DataFrame              | `qstats_df`, `signature_df`               |
| `_gdf`   | geopandas GeoDataFrame        | `region_gdf`, `outlets_gdf`               |
| `_cfg`   | parsed config dict (R01)     | `project_cfg`, `shared_cfg`, `my_cfg`     |

> `my_cfg` is the blessed idiom for a Snakefile's own workflow section
> (`config["workflows"][<name>]`) — established in R01 and uniform across
> all three Snakefiles. Use it; don't invent a per-workflow variant.

**Snakemake input/output labels (grandfathered, not for new Python):**

| Suffix   | Status                                              | Example       |
| -------- | --------------------------------------------------- | ------------- |
| `_nc`    | Reserved for Snakemake `input:`/`output:` labels mirroring a netCDF product | `climate_nc`, `rlz_nc` |
| `_csv`   | Same, for CSV products                              | `st_csv`      |
| `_yml`   | Same, for YAML products                             | `weathergen_config_yml` |
| `_png`   | Same, for PNG products                              | `output_png` (the existing `precip_plt` label is grandfathered non-conforming) |

In new Python code, prefer `_path` (for the path string) or `_ds`/`_df`
(for the loaded object) over the file-extension suffixes. The
extension suffixes are for Snakemake label hygiene only.

**Deprecated path suffixes (grandfathered; do not use in new code):**

| Suffix   | Replacement | Example of grandfathered usage              |
| -------- | ----------- | ------------------------------------------- |
| `_fn`    | `_path`     | `region_fn`, `csv_fn`                       |
| `_fid`   | `_path`     | `gauges_fid`, `region_fid`, `forcing_fid`   |
| `_file`  | `_path`     | `csv_file`                                  |

### 6. Domain identifiers — three tiers, none normalized casually

Domain identifiers carry different kinds of contract, so the guide
treats them in three tiers rather than one flat "external API" bucket.

**Tier 1 — opaque upstream identifiers (preserve verbatim).** Owned by
an external system; renaming breaks downstream tools or catalog lookups.

- **Wflow / CSDMS variable names** consumed by hydromt_wflow (e.g. the
  `land_surface__evapotranspiration_volume_flux`-style Standard Names in
  `setup_constant_pars`).
- **HydroMT data-catalog *schema*** — adapter fields and structure follow
  HydroMT's schema (the source-*name* strings are tier 2, below).
- **CMIP model IDs** (`NOAA-GFDL/GFDL-ESM4`, `INM/INM-CM5-0` —
  preserve hyphens, slashes, mixed case).
- **weathergenr R function names.**

Tier 1 has no local rename path at all — not even a migration note.

**Tier 2 — established BlueEarth cross-tool / output / scientific
contracts (grandfather + migration-gate).** BlueEarth-owned, but already
a contract with baseline manifests, downstream consumers, or users. Keep
as-is; rename only with a migration note (§7).

- User-facing output / config names: `Qstats`, `Tlow`, `Tpeak`.
- **HydroMT data-catalog source names** (`era5`, `merit_hydro`,
  `cmip6_<model>_<scenario>_<member>`) — BlueEarth-minted lookup keys (the
  CMIP6 ones are generated by `prepare_climate_data_catalog.py`) that form
  a catalog-lookup contract: migration-gated, not verbatim-frozen. The
  catalog *schema* they sit in is tier 1.
- The user-facing Wflow output *labels* mapped to CSDMS names in
  `setup_gauges_and_outputs.py` (`actual evapotranspiration`,
  `groundwater recharge`) — semantic display names, not the upstream IDs.
- Cross-tool scientific variable names: `precip`, `temp`.

**Tier 3 — new locally owned scientific identifiers.** Follow local
naming (snake_case, lowercase acronyms) unless an explicit external
schema dictates a spelling.

Rationale: tier 1 renames break external tools/catalogs; tier 2 renames
break baseline or user contracts (hence the migration gate); tier 3 is
free to follow local style because nothing downstream depends on it yet.

### 7. Do not rename without migration note

The following surfaces have downstream / baseline / user contracts.
Any renaming requires a `dev/<milestone>/migration_<topic>.md` note
listing the old → new mapping:

- `rule all` output filenames (baseline manifest contract).
- Checked-in example config keys (user-facing).
- Data catalog source names in `config/*.yml` (§6 tier 2 — catalog contract).
- Test fixture paths referenced by `tests/conftest.py`,
  `dev/scripts/check_baseline.py`, or other scripts.

(§6 tier-1 identifiers — CMIP IDs, Wflow/CSDMS names, weathergenr
function names — are not renameable at all, not even with a migration
note, so they are deliberately omitted from this list.)

**Scientific abbreviations in user-facing output filenames are
allowed.** `Qstats.csv`, `Tlow`, `Tpeak`, `BFI`, return-period
`T2` / `T10` etc. are established domain vocabulary. They violate the
acronym-lowercase rule but are user-facing and well understood;
keep as-is.

### 8. File naming by class

Different file classes follow different conventions. The local style
guide does not unify these.

| File class                                  | Convention   | Examples                                                 |
| ------------------------------------------- | ------------ | -------------------------------------------------------- |
| Python modules / R scripts                  | snake_case   | `prepare_climate_data_catalog.py`, `generate_weather.R`  |
| Snakemake entry points                      | `Snakefile_<workflow>` (existing) | `Snakefile_model_creation`             |
| Markdown planning docs under `dev/`         | kebab-case   | `naming-conventions-design.md`, `modularity-contracts-plan.md` |
| Standard root-level files                   | upstream     | `CLAUDE.md`, `README.rst`, `Dockerfile`, `LICENSE`        |
| Config / data / catalog YAML                | tool contract | `snake_config_model_test.yml`, `deltares_data.yml` (rename only with migration note) |
| Generated outputs under `project_dir/`      | governed by owning workflow contract (R01) | varies                     |

The `kebab-case` rule for `dev/*.md` recognizes the dominant
convention already in use across milestone planning docs. Don't
rename existing dev docs.

### 9. Examples

Compact "instead of / use" table to anchor the rules. Keep the final
guide's table sparse so the doc stays under 250 lines.

> **Illustrative future targets only.** R2 renames nothing — every
> existing identifier is grandfathered until its owning milestone touches
> it. Do not read this table as an R2 rename list.

| Instead of   | Use                       | Reason                                   |
| ------------ | ------------------------- | ---------------------------------------- |
| `config_fn`  | `config_path`             | Canonical path suffix.                   |
| `stats_nc` (when it's a path) | `stats_path` | Path/object distinction.                  |
| `stats_nc` (when it's a Dataset) | `stats_ds` | Path/object distinction.                  |
| `ST_NUM`     | `stress_test_count`       | Config-derived setting, not a true constant. |
| `RLZ_NUM`    | `rlz_count`               | Same.                                    |
| `st_num2`    | `st_num`                  | Stable wildcard vocabulary.              |
| `cmip6Models` | `cmip6_models`           | Lowercase acronym + snake_case.          |
| `TRUE` / `FALSE` (in BE-owned YAML) | `true` / `false` | Lowercase YAML booleans.            |

## Out of scope (what R2 does not deliver)

- Branch / commit / PR conventions (already in `dev/roadmap.md`).
- Output path conventions inside `project_dir` (in R01 workflow
  contract docs).
- Refactoring existing names to conform — explicitly grandfathered.
- Linter or CI enforcement — manual review for now. A future linter
  is a possible R3+ followup.
- Per-language style guides (e.g., function-length limits, comment
  conventions). Future `dev/reference/python-style.md`,
  `dev/reference/r-style.md` if needed.

## Verification

- `dev/reference/naming.md` exists and is < 250 lines.
- `AGENTS.md` has a one-line pointer to the naming doc (canonical
  instruction source; `CLAUDE.md` inherits it via its `@AGENTS.md`
  import — do not add the rule only to `CLAUDE.md`).
- `pixi run pytest tests/` unchanged: 51 passed, 3 skipped, 2 xfailed
  (the sealed R01 state).
- The R2 changeset touches documentation only — `dev/`, the new guide,
  `AGENTS.md`, `dev/roadmap.md`, `dev/reference/git-conventions.md`. No
  `Snakefile_*`, `src/`, `tests/`, config YAML, lockfile, manifest, or
  generated output appears in the diff.
- `dev/reference/git-conventions.md` records the `r02-naming` branch/tag at seal
  (it currently lists `r02-naming` as planned).

## Migration notes for existing names

R2 writes the contract; it does not enforce it on existing code.
R3-R5 may opportunistically rename when refactoring nearby code.
R6 (structural refactor) may do bulk file / module moves with a
`MIGRATION.md` mapping per the existing roadmap.

The deprecated path suffixes (`_fn`, `_fid`, `_file`) are the most
visible non-conformance. The roadmap entry for R3 should note that
incidental renames from these to `_path` are acceptable under R3's
"shared helper" deliverable when touching the affected code.

## Risks and open questions

- **Style guide rot**: prescriptive guides drift if not enforced.
  Mitigation: R3-R5 commit messages reference `dev/reference/
  naming.md` when adding new identifiers. A future linter would
  catch drift mechanically; defer until needed.
- **Domain-identifier boundary**: section 6's list will grow as new
  upstream tools enter scope (e.g., adding a different climate model
  family). Keep the list living; R3-R5 update it as new external
  identifiers appear.
- **Wildcard vocabulary growth**: section 4's table is small today.
  Future workflows may need new wildcards (e.g., a `member` wildcard
  for ensemble members). The "update naming.md in the same commit"
  rule keeps this honest.

## Tag

`r02-naming`. Sequenced after R01 seals. Branch `milestone/r02-naming`
off `main` (which carries the post-R01 baseline rebuild), not the bare
`r01-contracts` tag.

## Estimated commits (~3)

1. `r02: tighten naming design after independent review`
2. `r02: add naming guide and agent-instruction pointers`
3. `r02: seal naming milestone in roadmap and durable refs` (+ tag `r02-naming`)

(The old first commit, "open milestone with design spec", is dropped:
the design already exists in the `r01-contracts` ancestry.)

## Reference

- User review at `dev/milestones/r02/naming-conventions-review.md` (substantive
  review of the first design draft; surfaced the file-class table,
  the suffix path/object split, the YAML upstream exceptions, the
  examples table, and the lowercase-boolean policy). Supersedes the
  earlier blocker note that previously lived at
  `dev/conventions-review.md`.
- R01 sectioned config schema: `dev/milestones/r01/modularity-contracts-design.md`.

## Related but separate

- The outlet station naming decision in `dev/followups.md` (M2b
  carryover) affects output filenames and downstream interpretation.
  It's a behavioral / contract decision for R3, not a naming-guide
  decision.
