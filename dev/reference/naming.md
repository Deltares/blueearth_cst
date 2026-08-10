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
| `st_num`   | active                   | stress-test combination: `1..stress_test_count` perturbed; `0` = reserved unperturbed baseline (`st_0`), run through Wflow only when `run_historical` sets `ST_START = 0` |
| `member`   | reserved (CMIP ensemble) | ensemble member id (`r1i1p1f1`, …). Config-only today; becomes a wildcard if per-member rules are added |

The `st_num2` variant formerly used in `Snakefile_climate_experiment`'s
downstream rules (it admitted `0` under `run_historical`, where `st_num`
starts at `1`) was **folded into `st_num` in R5**. The downstream rules
(`downscale_climate_realization`, `run_wflow`, `derive_wflow_indicators`) now use
the single `st_num` vocabulary and keep the default match that admits `0`;
only `perturb_climate_realization` carries a rule-local
`wildcard_constraints: st_num=member_index_regex(ST_WIDTH)` that bars the
all-zeros baseline (so it cannot be a second producer of `st_0`). That was the
literal `[1-9][0-9]*` until R11 P2 zero-padded the index — a leading-zero-hostile
pattern rejects `st_01`, so it is now built from the width; see below.

**The member token in filenames and catalog keys is `st_`, the same word as the
wildcard.** R11 P2 renamed it from `cst_` (C22): `cst` is the toolbox's own
name, so it said nothing as a member token, while every layer that mattered
already said `st` — the `st_num` wildcard above, `ST_NUM`, `stress_test_grid()`,
the `stress_test:` config section. Only the filenames and the WG-5 catalog keys
disagreed, and `Snakefile_climate_experiment` built a `cst_` filename out of an
`st_num` wildcard on one line. So a member is `st_<m>.csv` and
`rlz_<n>_st_<m>.{nc,csv,toml,log}`, with `st_0` the reserved unperturbed
baseline. `rlz_` deliberately stays: it abbreviates a *correct* term and
collides with nothing. Record: `dev/milestones/r11/migration_indicator-tables.md`.

**Member indices are ZERO-PADDED to a width derived from the count** (C27),
so lexical order matches run order: `st_01 … st_12` for a twelve-point grid,
`st_001` past ninety-nine, and no padding at all below ten, where `st_1 … st_6`
already sort correctly. `rlz_` and `st_` pad independently, each from its own
count. The width is a pure function of `ST_NUM` / `RLZ_NUM`, both of which live
in the `climate_experiment` section `experiment.yml` freezes, so a grid change
that would move the width already forces a new experiment — no tree is ever
renamed underneath itself. One helper owns it (`snake_utils.index_width`), and
`member_index_regex` builds the matching `wildcard_constraints` so an *unpadded*
name raises `MissingRuleException` instead of routing silently. That regex MUST
stay anchor-free: Snakemake embeds a constraint in the whole path's regex, so a
`$` inside one binds to the end of the path and silently voids the condition.

**Three different things spell themselves `cst`**, which is why a bare `cst_`
grep is never the right tool: the package `blueearth_cst`, the member token
(now `st_`), and the WF2 netCDF provenance attributes in
`blueearth_cst/projections/` (`cst_calendar`, `cst_raw_digest`, `cst_source_paths`,
…), which mean "written by CST" and are part of WF2's on-disk output.

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

**`_rule` — a shared Snakemake rule definition.** A helper that returns a
frozen dataclass holding a rule's `script`, `inputs`, `outputs` and `params`
— everything content- or execution-determining, with only
`message`/`log`/`benchmark` left workflow-local — so the same rule can be
splatted into more than one Snakefile without the declarations drifting.
Function and dataclass both carry it: `region_rule` → `RegionRule`,
`climate_store_rule` → `ClimateStoreRule`, `spatial_units_rule` →
`SpatialUnitsRule`.

The suffix was `_spec` until `[R10-7]` (2026-08-06). `spec` reads as jargon
to a non-programmer and the object specifies nothing abstract — it *is* a
rule definition minus its labels. Two alternatives were rejected:
`_contract`, because this repo already uses "contract" for interchange
surfaces (`dev/reference/contracts/`, `SPATIAL_CONTRACT_VERSION`,
`test_climate_store_contract.py`) and overloading it is worse than the
jargon; and `_definition`, on verbosity at the call sites.

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

- User-facing config keys and table labels: `Tlow`, `Tpeak`. (`Qstats` was the
  third example until R9 P3 renamed `Qstats.csv` to `q_indicators.csv`; no
  filename relies on this tier any more — see §7 and §8.)
- HydroMT data-catalog *source names* (`era5`, `merit_hydro`,
  `cmip6_<model>_<scenario>_<member>`) — BlueEarth-minted lookup keys
  that form a catalog-lookup contract. (Their schema is tier 1.)
- User-facing Wflow output *labels* mapped to CSDMS names in
  `setup_gauges_and_outputs.py` (`actual evapotranspiration`,
  `groundwater recharge`) — display names, not the upstream IDs.
- Cross-tool scientific variable names: `precip`, `temp`. **These are the
  canonical stems and every producer now uses them** — the WG-1 extraction, the
  HM-2 / WG-6 wflow forcing, the `stress_test` config block, the `st_<m>.csv`
  perturbation files, and (since 2026-08-05) the two indicator tables. The one
  exception was `q_indicators.csv` / `basin_indicators.csv`, whose axis columns
  read `tavg` / `prcp`; see §7. Aliases that look like drift but are NOT — each is
  owned by an external schema and adapted at a named seam, so leave them alone:
  `tas` / `pr` (CMIP6, renamed by the catalog's `data_adapter.rename`),
  `Q` / `P` (wflow
  `[output.csv]` headers), `precipitation` (a `WFLOW_VARS` display label, tier 2
  above), and weathergenr's `temp_delta` / `precip_mean_factor` (tier 1).

**Tier 3 — new locally owned scientific identifiers.** Follow local
style (§1) unless an explicit external schema dictates a spelling.

## 7. Rename only with a migration note

Renaming any of these requires a `dev/<milestone>/migration_<topic>.md`
note listing the old → new mapping:

- `rule all` output filenames (baseline manifest contract).
- **Snakemake rule identifiers.** They are the CLI target surface
  (`snakemake <rule> -s …`, `--forcerun <rule>`) and are referenced across
  `docs/`, `dev/reference/` and the Snakefile comments, so a rename breaks a
  command someone has in their shell history. §9's "rule identifiers are not
  numbered" clause already called this a §7 event; it is listed here now
  because the enumeration, not the cross-reference, is what gets read.
  **R10's record is `dev/milestones/r10/migration_rule-names.md`.**
- **Column labels in `rule all` output tables** — a header is a tier-2 contract
  in its own right (§6), separately from the filename that carries it: a
  consumer that survived a file rename can still break on a header. Added
  2026-08-05, when the `tavg` → `temp_change` rename found this list enumerated
  only filenames.
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

**A second R9 record covers the COLUMNS inside those two files**:
`dev/milestones/r09/migration_indicator-axis-columns.md` (2026-08-05), scope
`tavg` → `temp_change` and `prcp` → `precip_change`. Two files renamed and their
axis columns renamed are separate §7 events with separate records, because a
table label is a tier-2 contract in its own right (§6) — a consumer that survived
the filename rename can still break on the header.

**R11's record is `dev/milestones/r11/migration_indicator-tables.md`**, and it
carries two §7 events: P1's indicator-table reshape (files and columns), and
P2's member-token rename `cst_` → `st_` (§4), which moves member filenames and
the WG-5 catalog *source names* — the tier-2 clause above. One document rather
than two because one milestone re-records the baseline once.

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
| Generated outputs under `project_dir/` | lowercase `snake_case`, two exemptions (below) | `q_indicators.csv`, `basin_indicators.csv`, `model_reference.yml`, `inmaps_rlz_1_st_2.nc` |

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

## 8b. Rule naming — `<verb>_<noun>`, verb first, always

Every Snakemake rule identifier is `<verb>_<noun>`. The verb comes from this
list — **one verb per action class**, so two rules doing the same kind of work
read the same. Name a new rule from the table, not by analogy with whichever
rule happens to sit above it.

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

Two distinctions needed care and are the ones a new rule gets wrong:

- **`reduce_` vs `derive_` splits by POSITION, not by operation.** Both turn
  many inputs into few outputs. `reduce_gcm_series` feeds a later rule;
  `derive_change_factors` and `derive_wflow_indicators` each produce their
  workflow's final answer. That makes WF2's and WF3's terminal rules read alike,
  which they should.
- **`prepare_` vs `write_` splits on where the work is.** The original criterion
  was "a config or intermediate" versus "one small table or index", which could
  not decide `write_experiment_config` or `write_climate_data_catalog` — both
  readings applied to both. The testable form:

  > **If you deleted the file-writing, would there be work left?
  > Yes → `prepare_`. No → `write_`.**

  Note what is *not* the criterion: whether a later rule consumes the output.
  `write_climate_data_catalog` is consumed downstream and is still `write_`,
  because enumerating entries is all it does.

**Nouns are full words.** Only the established domain set abbreviates — `gcm`,
`cmip6`, `wflow`, `rlz`, `st` — and those are tier-1/tier-2 identifiers under
§6. (`st` replaced `cst` at R11 P2; see §4.) Ad-hoc contractions (`weagen`, `proj`) are not. Qualifiers are trailing full
words, never two-letter suffixes.

**Adding a verb is allowed, and cheaper than a bad name.** `delineate_` and
`declare_` were both added rather than forcing their rules onto `derive_` or
`add_`: basin delineation is the field's own word, and `declare_wflow_outputs`
changes what the engine *emits* while `add_` is defined as adding model *data*.
The bar is that the action class is genuinely distinct — and that the verb has a
user. `evaluate_` was ruled in for a rule that then never existed and was
withdrawn, because a verb in this table with no rule behind it reads as an
available option that some rule must already justify.

**Grammar conformance is not body conformance.** A name satisfying
`<verb>_<noun>` can still be false: `add_gauges_and_outputs` passed the grammar
check for three milestones while adding no gauges — the job had moved to another
rule and the name did not follow. Check the verb against the rule's script or
shell body, and say which check you ran.

Full rationale and the twelve-rename audit:
`dev/milestones/r10/rule-naming-design.md`.

## 9. Rule numbering (`W.NN` reference scheme)

Each rule in the three `Snakefile_*` entry points carries a `W.NN`
reference number — `W` = workflow (`1` model_creation, `2`
climate_projections, `3` climate_experiment), `NN` = a zero-padded
**position in that workflow's logical order**: data first, then model
build, then run, then records. It exists in exactly two places, both
cheap:

- **A comment header above each rule** —
  `# 1.07  build_wflow_model — parameterize Wflow-SBM on the spatial foundation`.
- **The `log:` / `benchmark:` filename prefix** —
  `logs/1.07_build_wflow_model.log`,
  `benchmarks/_parts/1.07_build_wflow_model.tsv`; for wildcard rules the
  prefix goes on the subdirectory
  (`logs/3.15_run_wflow/batch_{b}.log`). All three
  workflows share `project_dir/logs`, so the `W` digit keeps their logs
  disambiguated and a single `ls logs/` sorts globally by workflow then
  step.

**Positional since 2026-08-06 (`[R10-5]`), and this reverses the previous
rule.** `NN` used to be "a stable identifier assigned when the rule is
created — NOT a position", which left it uncorrelated with everything: gaps
where rules had been removed, WF2 defining its rules out of numeric order,
five letter suffixes stacked beside one WF3 number, and `gather_benchmarks`
answering to 2.10 beside siblings at 1.14 and 3.12. The rule-index audit
made the workflow stages explicit and the numbers contradicting them became
the more visible defect. Two properties now hold and are worth stating
separately:

- **Contiguous** within each workflow, from `W.00` (`rule all`).
- **Every dependency points from a lower number to a higher one**, checked
  against each rule's `input:` block — **`ancient()` included**. `ancient()`
  suppresses the timestamp rerun-trigger, not the DAG edge; missing that is
  how the first draft of the map put two rules ahead of something they
  depend on.

**The cost was accepted knowingly: numbers are REUSED.** Under the old
policy a retired number stayed a gap, so a stale reference merely dangled
and was obvious. Now it silently resolves to a *different rule* — new 3.10
is `prepare_weathergen_config` where old 3.10 was `run_wflow`. Read every
`W.NN` in `dev/milestones/`, `DEVLOG.md`, `dev/decisions/` and the dated
migration records **as of its date**, and translate through
`dev/reference/workflows/rule-index.md` § *What changed*. Do not rewrite
those archives to the new numbers.

Rules:

- **Rule *identifiers* are NOT numbered** (MUST). Snakemake rule names are
  Python identifiers (no leading digit, no dot) and are the CLI target
  surface (`snakemake create_model -s …`) referenced across docs — a
  `W.NN` identifier would be both illegal-as-typed and a §7 contract
  rename. The number lives only in the comment and the log/benchmark path.
- The number is a **reference and reading aid, not execution order**
  (MUST keep this framing, and it survives the change to positional
  numbering). Snakemake executes from the DAG, so rules on separate branches
  run concurrently — WF1's `1.11`–`1.13` are parallel leaves and WF3 fans out
  over `rlz_num × st_num`. Low-to-high means **"cannot depend on"**, not
  "runs before". Each Snakefile states this in a header comment.
- **Definition order in the file need not match the number.** It does not
  today: module-level code is interleaved between rule blocks and depends on
  its position, so reordering the blocks would be a behaviour risk taken for
  cosmetics. `W.NN` is the rule's place in the workflow, not its offset in the
  file. (`LOG_RULES` *is* asserted to read in number order —
  `tests/test_log_rules_contract.py` — because that list is the merge order
  for the workflow log.)
- **Reference in prose/commits as "Rule 1.3"** (drop the pad); the padded
  `1.03` form is for the sortable filenames.
- **DO NOT RENUMBER TO INSERT A RULE. Use a letter suffix** (`1.09b`) until
  the next deliberate sweep, and take the whole workflow in one commit when
  that sweep comes. Renumbering is a migration, not an edit: the number
  appears in `LOG_RULES`, in log and benchmark paths, in `rule_banner`, in
  the comment headers and in prose across `dev/`, so a sweep is a wide edit
  with a silent failure mode — an unlisted `LOG_RULES` label drops its log
  section without erroring, which happened four times before it was made
  mechanically checkable.

  A letter suffix sorts correctly against the padded numbers
  (`"1.09" < "1.09b" < "1.10"`), so an inserted rule does not break the
  `LOG_RULES` ordering assertion.

  R9's version of this bullet said inserting "does NOT renumber anything
  below" and treated gaps as the accepted cost. `[R10-5]` accepted that cost
  once, deliberately, to buy contiguity and dependency order; the
  *steady-state* rule is unchanged and is the one above.

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
