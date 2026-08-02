# R02 design review — GPT-5.6 (2026-07-18)

## 1. Verdict

**Verdict: ready-with-changes.** The design is well bounded, substantially follows through on the prior review, and is detailed enough to execute without a separate plan. Its core repository claims are mostly accurate. Before execution, however, it needs targeted corrections to the universal file rule, wildcard semantics and milestone ownership, constant terminology, YAML/domain-identifier boundaries, and the stale `m02e:` commit decomposition. These are documentation fixes, not a wholesale redesign.

## 2. Prior-review follow-through

| Prior recommendation | Status | Assessment |
|---|---|---|
| File-class table | **Partially incorporated** | Section 8 adds the requested class-specific table and preserves `Snakefile_*`, root-standard names, tool-contract YAML, and generated outputs. However, section 1 still says “snake_case for … files (MUST)” (`dev/milestones/r02/naming-conventions-design.md:68–70`), contradicting section 8’s kebab-case rule for `dev/` planning documents. The prior review asked to replace that universal file rule, not merely supplement it. |
| Reclassify `RLZ_NUM`, `ST_NUM`, `DATA_SOURCES` | **Incorporated** | The decisions table distinguishes true constants from config-derived run settings, grandfathers the existing uppercase globals, and proposes lowercase replacements (`dev/milestones/r02/naming-conventions-design.md`, “Decisions baked in”). Section 9 reinforces this with `ST_NUM` → `stress_test_count` and `RLZ_NUM` → `rlz_count`. |
| Split path and object suffixes | **Incorporated** | Section 5 cleanly separates `_path`/`_dir`, `_ds`/`_df`/`_gdf`/`_cfg`, Snakemake product labels, and deprecated path suffixes. |
| YAML upstream exceptions | **Incorporated, with one conflict** | Section 2 exempts weathergenr, HydroMT/Wflow configurations, and HydroMT data-catalog schemas. The decisions table nevertheless classifies “generated catalogs” as BlueEarth-owned snake_case YAML, which conflicts with the HydroMT-schema exception. |
| Lowercase YAML booleans | **Incorporated** | The decisions table, section 2, and section 9 consistently prescribe lowercase `true`/`false` for new BlueEarth-owned YAML while grandfathering existing uppercase values. |
| Examples table | **Incorporated** | Section 9 supplies the requested sparse “instead of / use / reason” table and covers paths, loaded objects, config-derived settings, wildcards, acronyms, and booleans. |
| Resolve stale `dev/conventions-review.md` reference | **Incorporated** | The “Reference” section names `dev/milestones/r02/naming-conventions-review.md` as the substantive review and explicitly says it supersedes the earlier blocker formerly at `dev/conventions-review.md`. |

## 3. Correctness pass

### Required corrections

1. **The universal file rule contradicts the file-class decision.** Section 1 requires snake_case for all files, while section 8 prescribes kebab-case for Markdown planning documents. The repository supports the class-specific rule: current R01/R02 planning documents include `dev/milestones/r01/modularity-contracts-design.md`, `dev/milestones/r01/modularity-contracts-plan.md`, `dev/milestones/r02/naming-conventions-design.md`, and `dev/milestones/r02/naming-conventions-review.md`. Historical `dev/` artifacts remain mixed, including `package_inventory.md` and `baseline_diffs.md`, so the guide should not make a universal claim about all `dev/*.md`. Replace section 1’s rule with “snake_case for variables, functions, and modules”; delegate filenames entirely to section 8.

2. **The `st_num` range and cleanup ownership are misstated.** Section 4 describes `st_num` as a stress-test combination numbered `0..ST_NUM`. In the current workflow, perturbation CSVs and `st_num` products use `1..ST_NUM`; `cst_0` is the unperturbed realization. The separate `st_num2` wildcard can include zero only when `run_historical` makes `ST_START = 0` (`Snakefile_climate_experiment:38,84,122,140,153–180`). The guide should define:

   - `0`: reserved unperturbed baseline, included in Wflow results only when configured;
   - `1..stress_test_count`: perturbed combinations.

   Section 4 also says to fold `st_num2` into `st_num` during “R3–R5 Snakefile cleanup.” `Snakefile_climate_experiment` belongs specifically to R5 under `dev/roadmap.md`, while the roadmap forbids milestones from touching one another’s territory. Assign this cleanup to **R5 only**.

3. **Not all proposed “true constants” are immutable.** The examples are present, but their implementation differs:

   - `VOLATILE_NC_ATTRS` is an immutable `frozenset` (`dev/scripts/check_baseline.py:43`).
   - `XDIMS` and `YDIMS` are tuples (`src/get_change_climate_proj.py:212–213`; `src/get_stats_climate_proj.py:112–113`), although the latter pair is function-local rather than module-level.
   - Both `WFLOW_VARS` definitions are mutable dictionaries (`src/setup_gauges_and_outputs.py:13`; `src/func_plot_signature.py:24`). Current targeted uses read rather than mutate them, but immutability is not enforced.

   Replace “immutable lookups” with “fixed, non-config-derived values and lookup tables that are not reassigned or mutated at runtime,” or use only genuinely immutable objects as examples.

4. **The domain-identifier rationale conflates upstream identifiers with local scientific contracts.** Section 6 says all listed names are external-system API contracts. That is inaccurate:

   - `setup_gauges_and_outputs.py:10–18` identifies strings such as `actual evapotranspiration` as **user-facing semantic names** mapped to actual Wflow/CSDMS identifiers such as `land_surface__evapotranspiration_volume_flux`.
   - `Qstats`, `Tlow`, and `Tpeak` are current BlueEarth output/config contracts (`Snakefile_climate_experiment:51,182,188–189`), not upstream identifiers.
   - `precip` and `temp` are cross-tool scientific variable names, but their preservation rationale differs from an opaque upstream API identifier.

   Preserve all existing names, but split the rule into three categories:

   - opaque upstream identifiers: preserve verbatim;
   - established BlueEarth cross-tool, output, or scientific contracts: grandfather and migration-gate;
   - new locally owned scientific identifiers: follow local naming unless an explicit schema dictates otherwise.

5. **YAML ownership is not a sufficient discriminator.** The decisions table includes “generated catalogs” under BlueEarth-owned snake_case YAML, while section 2 correctly says data-catalog source names and adapter fields follow HydroMT’s schema. A locally generated file can still be consumed under an upstream schema. The rule should be based on the consuming contract: BlueEarth snake configs use local conventions; any HydroMT/Wflow/weathergenr-consumed YAML preserves the upstream schema, whether checked in or generated.

6. **The estimated commit prefixes are stale.** The design proposes three `m02e:` subjects (`dev/milestones/r02/naming-conventions-design.md:300–304`). The active Phase-2 scheme requires `r02:` (`dev/roadmap.md:478–496`); `m02e:` is not a recognized milestone prefix. The branch and tag names—`milestone/r02-naming` and `r02-naming`—are correct.

   The first proposed commit, “open naming-conventions milestone with design spec,” is also obsolete because the design already exists in the `r01-contracts` ancestry. A suitable replacement is:

   1. `r02: tighten naming design after independent review`
   2. `r02: add naming guide and agent-instruction pointers`
   3. `r02: seal naming milestone in roadmap and durable refs`

### Verified repository claims

- `config/weathergen_config.yml:10–12,18` contains `warm.signif.level`, `warm.sample.num`, `knn.sample.num`, and `evaluate.model`; the dotted-key claim is correct. It also contains upstream-owned uppercase booleans, consistent with the proposed exemption.
- `st_num2` exists in `Snakefile_climate_experiment:153–180`.
- `DATA_SOURCES` is config-derived in all three Snakefiles; `RLZ_NUM` and `ST_NUM` are config-derived in `Snakefile_climate_experiment:25–36`. Reclassifying them away from true constants is correct.
- The current canonical/test/Linux configurations use the R01 `project` / `shared` / `workflows.<name>` schema referenced in section 2.
- The CMIP identifier examples are real: `INM/INM-CM5-0` and `NOAA-GFDL/GFDL-ESM4` occur in `config/cmip6_data.yml` and the canonical snake config.
- The guide’s kebab-case observation is supported for current milestone planning documents, but not as a universal rule for the entire historical `dev/` tree.

## 4. Completeness pass

After the corrections above, the design is detailed enough to execute directly. Its nine-section outline is effectively a content specification for `dev/reference/naming.md`, including normative voice, examples, exceptions, migration boundaries, and the 250-line target.

The following execution details should be made explicit:

1. **Add a changed-path allowlist.** “No code files modified” is weaker than the stated pure-docs boundary. At sealing, require the diff from `r01-contracts` to contain only the approved documentation paths, such as:

   - `dev/milestones/r02/naming-conventions-design.md`
   - `dev/reference/naming.md`
   - `AGENTS.md` and/or `CLAUDE.md`
   - `dev/roadmap.md`
   - `dev/reference/branches-and-tags.md`

   No `Snakefile_*`, `src/`, `tests/`, configuration YAML, lockfile, manifest, or generated output should appear.

2. **Specify how the guide is made canonical and discoverable.** `AGENTS.md` says it is the canonical cross-runtime instruction source and that `CLAUDE.md` is only a thin importer. A pointer added only to `CLAUDE.md` would not equally serve runtimes that read `AGENTS.md` directly. Put the canonical naming-guide pointer in `AGENTS.md`; if the roadmap’s direct `CLAUDE.md` criterion is retained, add only a matching reference line there without duplicating rules.

3. **Make “suite unchanged” concrete.** The sealed R01 state is recorded as **51 passed, 3 skipped, 2 xfailed** in `dev/roadmap.md:137–142,199`. Use that as the expected comparison for `pixi run pytest tests/`. No Snakemake workflow or scientific baseline run is warranted for this docs-only milestone.

4. **Update the durable-ref inventory at sealing.** `dev/reference/branches-and-tags.md:39–40` still lists `r02-naming` as planned. The sealing commit should record the branch/tag there as well as marking R2 sealed in `dev/roadmap.md`.

5. **State the final mechanical gates.** Before tagging, confirm the guide exists, is under 250 lines, contains all nine specified sections, the suite matches the sealed R01 state, and the changed-path allowlist contains documentation only.

## 5. Risk pass

1. **External and scientific identifiers are either normalized accidentally or exempted too broadly.** The current single “external API” bucket mixes opaque upstream names with BlueEarth-owned scientific/output contracts.

   **Mitigation:** Use the three-tier taxonomy above. Require verbatim preservation for upstream identifiers, migration notes for established local contracts, and normal local style for newly introduced locally owned names.

2. **Examples intended for later milestones are executed as an R2 rename list.** Section 9 includes `ST_NUM`, `RLZ_NUM`, and `st_num2`, while section 4 directs a future wildcard fold. An executor could interpret these as current actions despite the grandfathering language.

   **Mitigation:** Add an explicit banner above the examples: “Illustrative future naming targets only; R2 MUST NOT rename any existing identifier.” Assign `st_num2` solely to R5 and require that milestone’s dry-run/test gates.

3. **The milestone acquires non-documentation drift while being treated as harmless.** Editing config keys, output names, or generated artifacts would violate both grandfathering and the zero-code-diff promise even if tests remained green.

   **Mitigation:** Enforce the tag-to-branch changed-path allowlist before every R02 commit and again before `r02-naming` is cut. Treat any path outside the approved documentation set as a blocker.

## 6. Minor findings

- Section 5’s `_yml` example, `weagen_config_yml`, preserves the repository’s existing `weagen` abbreviation/typo and does not currently exist as a label. Prefer `weathergen_config_yml` as the new-code example.
- The `_png` row uses `precip_plt`, which does not end in `_png`; it is an existing Snakemake label (`Snakefile_climate_projections:51,154`) but is not an example of the proposed suffix. Use an actual `_png` example or label it explicitly as grandfathered.
- `_cfg` uses `my_cfg` as an example. A prescriptive guide should prefer a domain-specific name such as `model_creation_cfg` or `climate_experiment_cfg`.
- “Matches Python’s pathlib convention” overstates the rationale for `_path`; `pathlib` provides `Path` objects but does not prescribe identifier suffixes. “Works naturally with `pathlib.Path` and is explicit across languages” is more accurate.
