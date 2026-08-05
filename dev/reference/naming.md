# Naming conventions

Prescriptive style guide for naming identifiers and files in
`blueearth_cst`. `MUST` / `SHOULD` / `MAY` carry their usual normative
weight.

**Grandfathered today, applied tomorrow.** This guide governs *new* code
from R3 onward. Existing non-conforming names stay as-is until the
milestone that owns them refactors them — do not rename an identifier
just to conform. Renaming a *contract* surface (§7) needs a migration
note.

**Local style yields to external / established contracts.** Identifiers
governed by an upstream tool or an established BlueEarth contract follow
those contracts, not the rules here — see §6.

## 1. Universal rules

- snake_case for variables, functions, and modules (MUST). File names
  are governed by class — see §8, not this rule.
- Lowercase acronyms inside identifiers (MUST): `cmip6_models`,
  `era5_orography`, `csdms_name` — never `CMIP6Models`.
- `UPPER_SNAKE_CASE` only for true constants — fixed, non-config-derived
  values or lookup tables that are not reassigned or mutated at runtime
  (MUST). Config-derived run settings are lowercase (§9).
- Verbs for functions, nouns for variables and data (SHOULD).

## 2. Per language

**Python** — PEP 8: snake_case variables / functions / modules;
PascalCase classes; `UPPER_SNAKE_CASE` for module-level true constants
only.

**R** — snake_case, not `dot.case` (aligns with tidyverse and the
weathergenr package). Verb-noun functions (`read_climate_data`, not
`climate_data`).

**Snakemake** — rule names snake_case (MUST); `verb_noun` for action
rules (`build_model`, `add_gauges`, `run_wflow`) (SHOULD); noun-only is
acceptable for non-action rules like `rule all` (MAY).

**YAML** — discriminate by the *consuming contract*, never authorship or
whether the file is checked in vs. generated:

- BlueEarth-owned configs consumed locally — the R01 `project` /
  `shared` / `workflows.<name>` snake config — use snake_case keys and
  lowercase booleans `true` / `false` (MUST for new keys).
- Any YAML consumed under an upstream schema preserves the upstream
  spelling (MUST), **even when BlueEarth generates the file**:
  weathergenr (`warm.signif.level`), HydroMT / Wflow parameter names,
  and HydroMT data catalogs.

Existing `TRUE` / `FALSE` in BlueEarth configs are grandfathered.

## 3. Path-identifier suffix (`_path` canonical)

New code MUST use `_path` for a variable holding a file-path string:
`region_path`, `forcing_path`, `csv_path`. `_path` is explicit,
language-neutral, and works naturally with `pathlib.Path`. The deprecated
suffixes `_fn`, `_fid`, `_file` are grandfathered — do not use them in
new code, and rename an existing one only with a migration note.

## 4. Snakemake wildcards (stable vocabulary)

Wildcards used across Snakefiles MUST come from this list. Adding a new
wildcard requires updating this file in the same commit.

| Wildcard   | Status                   | Meaning                                                                 |
| ---------- | ------------------------ | ----------------------------------------------------------------------- |
| `model`    | active                   | climate model id (CMIP6 model name)                                     |
| `scenario` | active                   | climate scenario (`historical`, `ssp245`, …)                            |
| `horizon`  | active                   | future horizon name (`near`, `far`)                                     |
| `rlz_num`  | active                   | weather realization number (`1..rlz_count`)                             |
| `st_num`   | active                   | stress-test combination: `1..stress_test_count` perturbed; `0` = reserved unperturbed baseline (`cst_0`), run through Wflow only when `run_historical` sets `ST_START = 0` |
| `member`   | reserved (CMIP ensemble) | ensemble member id (`r1i1p1f1`, …). Config-only today; becomes a wildcard if per-member rules are added |

The `st_num2` variant formerly used in `Snakefile_climate_experiment`'s
downstream rules (it admitted `0` under `run_historical`, where `st_num`
starts at `1`) was **folded into `st_num` in R5**. The downstream rules
(`downscale_climate_realization`, `run_wflow`, `export_wflow_results`) now use
the single `st_num` vocabulary and keep the default match that admits `0`;
only `generate_climate_stress_test` carries a rule-local
`wildcard_constraints: st_num=[1-9][0-9]*` that bars `0` (so it cannot be a
second producer of the `cst_0` baseline).

## 5. Suffix vocabulary — path vs. object

A suffix means EITHER a filesystem path OR a loaded object, never both.
This is the single biggest readability win R3+ can make incrementally.

**Paths:** `_dir` (directory path — `project_dir`, `basin_dir`),
`_path` (file path, any extension — `region_path`, `catalog_path`).

**Loaded objects:** `_ds` (xarray Dataset), `_df` (pandas DataFrame),
`_gdf` (GeoDataFrame), `_cfg` (parsed config dict). `project_cfg`,
`shared_cfg`, and `my_cfg` are the blessed R01 idiom — `my_cfg` for a
Snakefile's own `workflows.<name>` section, uniform across all three
Snakefiles; use it, don't invent a per-workflow variant.

**Extension suffixes** (`_nc`, `_csv`, `_yml`, `_png`) are reserved for
Snakemake `input:` / `output:` labels that mirror a file product
(`climate_nc`, `st_csv`, `weathergen_config_yml`, `output_png`). New
Python code uses `_path` (the string) or `_ds` / `_df` (the object)
instead. Existing non-conforming labels (e.g. `precip_plt`) are
grandfathered.

**Deprecated path suffixes** (grandfathered; do not use in new code):
`_fn`, `_fid`, `_file` → `_path`.

## 6. Domain identifiers — three tiers

Domain identifiers carry different kinds of contract, so treat them in
three tiers rather than one flat "external" bucket. None are normalized
casually.

**Tier 1 — opaque upstream identifiers. Preserve verbatim; no local
rename path, not even a migration note.**

- Wflow / CSDMS variable names consumed by hydromt_wflow (e.g.
  `land_surface__evapotranspiration_volume_flux` in `setup_constant_pars`).
- HydroMT data-catalog *schema* — adapter fields and structure.
- CMIP model IDs (`NOAA-GFDL/GFDL-ESM4`, `INM/INM-CM5-0` — keep hyphens,
  slashes, mixed case).
- weathergenr R function names.

**Tier 2 — established BlueEarth contracts. Grandfather; rename only with
a migration note (§7).**

- User-facing output / config names: `Qstats`, `Tlow`, `Tpeak`.
- HydroMT data-catalog *source names* (`era5`, `merit_hydro`,
  `cmip6_<model>_<scenario>_<member>`) — BlueEarth-minted lookup keys
  that form a catalog-lookup contract. (Their schema is tier 1.)
- User-facing Wflow output *labels* mapped to CSDMS names in
  `setup_gauges_and_outputs.py` (`actual evapotranspiration`,
  `groundwater recharge`) — display names, not the upstream IDs.
- Cross-tool scientific variable names: `precip`, `temp`.

**Tier 3 — new locally owned scientific identifiers.** Follow local
style (§1) unless an explicit external schema dictates a spelling.

## 7. Rename only with a migration note

Renaming any of these requires a `dev/<milestone>/migration_<topic>.md`
note listing the old → new mapping:

- `rule all` output filenames (baseline manifest contract).
- Checked-in example config keys (user-facing).
- HydroMT data-catalog source names in `config/*.yml` (§6 tier 2).
- Test fixture paths read by `tests/conftest.py`,
  `dev/scripts/check_baseline.py`, or other scripts.

Tier-1 identifiers (§6) are not renameable at all, so they are omitted
here.

**R9's record is `dev/milestones/r09/migration_project-tree.md`**, and its §7
scope is exactly two files: `Qstats.csv` → `q_indicators.csv` and `basin.csv` →
`basin_indicators.csv`. Everything else R9 moved is a directory relocation or a
non-`rule all` artifact, and `series/` → `output/` is a directory rename, so §7
does not extend to them. Stated so the scope is not re-derived.

**Two artifact classes, distinguished (R07).** The rename note above and a
user-facing migration guide are different documents with different audiences,
and conflating them is what made `MIGRATION.md`'s home ambiguous:

| Class | Location | Required? | Audience |
| --- | --- | --- | --- |
| Internal rename record | `dev/<milestone>/migration_<topic>.md` | **Required** for every rename listed above | Whoever implements or audits the milestone: the old → new table, the machinery to update, the gate evidence |
| User-facing migration guide | `docs/migration-<milestone>.md` | **Optional** — write one only when users must act | Someone with an existing install or project folder |

A milestone that changes nothing a user must act on ships the internal record
and no guide. R07 is such a milestone: it declares pre-R07 `project_dir` trees
unsupported and requires a fresh run, so there is nothing for a user to
migrate, and it publishes no guide.

**The mandated `migration_<topic>.md` filename overrides §8's kebab-case rule
for `dev/` markdown.** The form is fixed by this section; §8 does not apply to
it. (Stated because two consecutive milestones hit the ambiguity.)

**Scientific abbreviations are allowed in config keys and column/row labels**
even though they break the acronym-lowercase rule: `Tlow`, `Tpeak`,
return-period `T2` / `T10`, `BFI`. These are established domain vocabulary;
keep them.

**Narrowed at R9 (was: "in user-facing output filenames").** The carve-out
existed for `Qstats.csv`, and R9 renamed it to `q_indicators.csv` — after which
no *filename* relied on it. The exemption is not repealed, because the labels
inside those tables still need it; it is scoped to where it is actually load-
bearing. §8's generated-outputs rule now governs the filenames.

## 8. File naming by class

Different file classes follow different conventions; this guide does not
unify them.

| File class                             | Convention                        | Examples                                             |
| -------------------------------------- | --------------------------------- | ---------------------------------------------------- |
| Python modules / R scripts             | snake_case                        | `prepare_climate_data_catalog.py`, `generate_weather.R` |
| Snakemake entry points                 | `Snakefile_<workflow>` (existing) | `Snakefile_model_creation`                           |
| Markdown planning docs under `dev/`    | kebab-case                        | `naming-conventions-design.md`                       |
| Standard root-level files              | upstream                          | `CLAUDE.md`, `README.rst`, `Dockerfile`, `LICENSE`   |
| Config / data / catalog YAML           | tool contract                     | `snake_config_model_test.yml`, `deltares_data.yml`   |
| Generated outputs under `project_dir/` | lowercase `snake_case`, two exemptions (below) | `q_indicators.csv`, `basin_indicators.csv`, `model_reference.yml`, `inmaps_rlz_1_cst_2.nc` |

Don't rename existing `dev/` docs.

### Generated outputs under `project_dir/` (R9)

Locally minted file and directory names are lowercase `snake_case`: no hyphens,
no capitals, no spaces. This replaces "owning workflow contract — varies", which
was not a rule and let each workflow answer differently.

Two exemptions, both narrow, both stated so a reader does not "correct" them:

1. **Upstream-owned names pass through verbatim.** Engine-mandated filenames
   (`wflow_sbm.toml`, `staticmaps.nc`, `instates.nc`, `hydromt_data.yml`) and
   upstream identifiers embedded in a path — CMIP model IDs such as
   `NOAA-GFDL/GFDL-ESM4`, which carry hyphens, slashes and mixed case — are
   never normalized. These are §6 tier-1 identifiers.
2. **Config keys and data labels are out of reach.** The rule governs filenames
   and directory names only. Column and row labels (`Tlow`, `Tpeak`, `BFI`) and
   config keys keep their domain spelling — see §7's narrowed carve-out.

**The rule is CLASS-SCOPED and must not be generalised.** It governs generated
output names under `project_dir/` and nothing else: `dev/` markdown stays
kebab-case (the row above), Python modules stay `snake_case` because they must
be importable, and root-level files keep their upstream names. Reading this as a
repo-wide sweep would rename documents this guide explicitly protects.

## 9. Rule numbering (`W.NN` reference scheme)

Each rule in the three `Snakefile_*` entry points carries a `W.NN`
reference number — `W` = workflow (`1` model_creation, `2`
climate_projections, `3` climate_experiment), `NN` = a zero-padded
**stable identifier assigned when the rule is created** — NOT a position. It exists in exactly two
places, both cheap:

- **A comment header above each rule** —
  `# 1.03  create_model — build the Wflow-SBM model (hydromt build wflow_sbm)`.
- **The `log:` / `benchmark:` filename prefix** —
  `logs/1.03_create_model.log`, `benchmarks/1.03_create_model.tsv`; for
  wildcard rules the prefix goes on the subdirectory
  (`logs/3.10_run_wflow/rlz_{rlz_num}_cst_{st_num}.log`). All three
  workflows share `project_dir/logs`, so the `W` digit keeps their logs
  disambiguated and a single `ls logs/` sorts globally by workflow then
  step.

Rules:

- **Rule *identifiers* are NOT numbered** (MUST). Snakemake rule names are
  Python identifiers (no leading digit, no dot) and are the CLI target
  surface (`snakemake create_model -s …`) referenced across docs — a
  `W.NN` identifier would be both illegal-as-typed and a §7 contract
  rename. The number lives only in the comment and the log/benchmark path.
- The number is a **reference and reading aid, not execution order**
  (MUST keep this framing). Snakemake executes from the DAG; e.g.
  WF1 `1.10`–`1.12` are parallel plot leaves and WF3 fans out over
  `rlz_num × st_num`. Each Snakefile states this in a header comment.
- **Reference in prose/commits as "Rule 1.3"** (drop the pad); the padded
  `1.03` form is for the sortable filenames.
- **Inserting a rule takes the next free number, or a letter suffix; it does
  NOT renumber anything below.** Corrected at R9, because the previous wording
  ("renumbers the contiguous comments below it… use contiguous numbers, not
  gaps") described a practice the code has never followed:

  | Claim | Reality |
  | --- | --- |
  | contiguous, no gaps | gaps at **1.14**, **2.05**, **3.12** |
  | definition order | WF2 defines `2.03b`, `2.03`, `2.01`, `2.02` — out of numeric order |
  | renumber on insert | R9 P4 inserted `3.01c`, `3.01d`, `3.01e` and renumbered nothing |

  A letter suffix (`1.01b`, `3.00b`) is the established way to insert between
  two numbers, and it is preferable to renumbering: the number appears in
  `LOG_RULES`, in log and benchmark paths, and in prose across `dev/`, so a
  sweep is a wide edit with a silent failure mode — an unlisted `LOG_RULES`
  label drops its log section without erroring, which happened three times in
  R9 alone. Gaps are the cost of that safety and are expected.

- "Rule 1.5" decimals remain review shorthand for *talking about* an insert,
  never a permanent identifier.

## 10. Examples

> **Illustrative future targets only.** This is not a rename list —
> existing identifiers are grandfathered until their owning milestone
> touches them.

| Instead of                          | Use                 | Reason                                       |
| ----------------------------------- | ------------------- | -------------------------------------------- |
| `config_fn`                         | `config_path`       | Canonical path suffix.                        |
| `stats_nc` (a path)                 | `stats_path`        | Path / object distinction.                    |
| `stats_nc` (a Dataset)              | `stats_ds`          | Path / object distinction.                    |
| `ST_NUM`                            | `stress_test_count` | Config-derived setting, not a true constant.  |
| `RLZ_NUM`                           | `rlz_count`         | Same.                                         |
| `st_num2`                           | `st_num`            | Stable wildcard vocabulary.                   |
| `cmip6Models`                       | `cmip6_models`      | Lowercase acronym + snake_case.               |
| `TRUE` / `FALSE` (BlueEarth YAML)   | `true` / `false`    | Lowercase YAML booleans.                      |
