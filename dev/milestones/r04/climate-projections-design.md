# R04 — Workflow 2: climate projections — design

**Status.** Accepted 2026-07-20 (design-review-loop run
`r04-climate-projections`, gate G2). Review trail: 3-lens internal panel, 3
external cross-vendor (GPT) rounds, round-cap arbitration by the user on the
two surviving majors — 24/24 findings accepted and closed (full dispositions
and verbatim external verdicts: `climate-projections-design-reviews.md`, this
directory). This final text applies the two arbitration-mandated fixes
(ext3-1, ext3-2) on top of the round-3-reviewed v4, carrying every prior
accepted resolution forward. G1-approved framing: behavior-preserving
refactor, zero manifest edits by default, split-don't-absorb audit policy,
attrs out-by-default (user-ratified).

**Date.** 2026-07-20.

**What v5 changes vs v4 (scope of this revision, two arbitrated findings only):**

- **ext3-1 (major, user-arbitrated).** v4's matrix rows V and P stated the norm
  as "fail loudly **or** be explicitly reconciled," leaving two permitted
  behaviors — so the normative tests could not distinguish the current
  silent-loss defect from every permitted fix, and a correct fail-loud fix
  would have stayed an expected `xfail` instead of tripping the strict gate.
  v5 pins **fail-loud as the single norm for both rows** (a `ValueError`
  naming the asymmetry), specifies the exception contract the normative
  acceptance tests assert, and **separates those normative tests from
  current-behavior characterization tests**, naming the exact condition under
  which each marker/test is removed. See §5 (missing-data bullets) and §7.
- **ext3-2 (major, user-arbitrated).** v4's probe checkpoints P3 and P4 each
  bundled several potentially attr-dropping operations, so an empty-attrs
  reading there still localized only to a multi-operation region; and the
  probe's mechanism ("inside the production function") was unspecified,
  risking ad-hoc instrumentation of computational code in a
  behavior-preserving milestone. v5 splits the compound checkpoints into
  **one checkpoint per operation** (P3a/P3b, P4a/P4b) and specifies the
  **probe mechanism and artifact ownership**: a standalone stepwise
  diagnostic script under `dev/scripts/`, no production-code instrumentation
  by default, and a removed-and-verified-by-diff rule for any temporary
  instrumentation. See §5 "M2b attrs diagnostic probe".

## Goal

Clean up `Snakefile_climate_projections` (workflow 2) and the four `src/`
scripts it invokes — orchestration *and* analytical code — by inheriting the
patterns R3 minted on workflow 1 (shared `get_config`/`tee_to_log` from
`src/snake_utils.py`, per-rule `log:`/`benchmark:`, contract-doc-before-code,
config-path forwarding). It also delivers the one workflow-2-specific
scientific deliverable the roadmap names: an end-to-end audit of the
`monthly_stats_hist → monthly_stats_fut → monthly_change` change-factor chain
for unit consistency, calendar handling, and missing-data behavior — an audit
that produces **executable evidence** (§7 test matrix) with **normative
pass/fail criteria** (ext1-2, ext2-2), not inspection alone.

**R4 is a behavior-preserving refactor by default.** Like R3, it aims to make
**zero manifest edits** and keep `check_baseline check` clean throughout. Any
finding from the chain audit that would change an output value is either
explicitly chartered here with its baseline consequence, or split to a
dedicated task with a named owner and activation condition — never absorbed
silently (decision criteria §1, intake). Note the interaction with the
executable-evidence requirement: a test that *reveals* a defect
**characterizes** it; it does **not** authorize an inline fix. A revealed
calendar/variable-dropping defect still routes through the defect class (split,
don't absorb). Executable evidence sharpens the audit; it does not weaken
behavior-preservation.

One honest caveat, stated once and relied on throughout: a clean
`check_baseline` does **not** by itself prove workflow 2's scientific outputs
are unchanged in every respect, and — for the M2b attrs case (§"Candidate
absorptions" #1) — does not prove they are *correct*. The manifest fingerprints
workflow 2 with **7 targets** (`dev/scripts/check_baseline.py` `TARGETS`,
lines 60–66 — the workflow-2 subset of the full `TARGETS` list at lines 53–71):
the merged `annual_change_scalar_stats_summary.nc` (per-variable summary stats +
`attrs`), two summary CSVs (normalized SHA256), three PNGs (**size-only, ±10%
tolerance** — `PNG_TOLERANCE_FRAC = 0.10`, line 40), and the
`snake_config_climate_projections.yml` snapshot (SHA256 of full parsed YAML).
Critically, the **per-model intermediate** netCDFs
(`historical_stats_time_{model}.nc`, `stats_time-{model}_{scenario}.nc`,
`annual_change_scalar_stats-{model}_{scenario}_{horizon}.nc`) are wrapped in
`temp(...)` and are **not** manifest targets — they are deleted after
consumers finish. So the gate sees the *merged summary* and the *end plots*,
not the per-model change factors that feed them. R4's guarantee for the
intermediates rests on *not editing any computational path*; the gate is
necessary, not sufficient. Unlike R3, though, the summary `.nc` fingerprint
**does** capture `attrs` (`check_baseline.py` line 116 records per-variable
`attrs`), which is exactly the surface the M2b attrs-loss defect touches — so
if R4 restored those attrs, `check` would go red (see §"Candidate absorptions"
#1).

## Why now

R1 sectioned this Snakefile's config and R3 moved it onto the shared
`get_config` import + `sys.path` bootstrap (both already present at the top of
`Snakefile_climate_projections`, lines 1–34 — verified). But everything else
in workflow 2 predates the Phase-2 discipline:

- No per-workflow contract doc — `dev/reference/workflows/climate_projections.md` was
  deferred from R1 to R4's opening act (roadmap R4 deliverables).
- **Zero** `log:`/`benchmark:` directives on any rule in
  `Snakefile_climate_projections` (verified: the file has none). The exhaustive
  M1 warnings triage (`dev/followups.md` § R3) closes only once all three
  Snakefiles emit logs; R3 swept workflow 1, R4 sweeps workflow 2.
- The `ruleorder:` at line 53 is uncommented, and `AGENTS.md` (lines 95–96)
  asserts it is load-bearing ("wildcard inference is ambiguous without it") — a
  claim this design's dry-run refuted (§3), so the AGENTS.md correction is now
  part of R4's scope (Group-B fix below).
- `src/get_stats_climate_proj.py` and the
  `monthly_stats_hist → monthly_stats_fut → monthly_change` chain have never had
  a correctness / units / calendar / missing-data review.
- Zero unit-test coverage for any of the four workflow-2 Python helpers.
- One open defect in workflow-2 territory: CMIP6 `precip`/`temp` `.attrs` lost
  on `monthly_change_scalar_merge` under hydromt 1.3 (`dev/followups.md` § R3+,
  lines 316–322) — a **documented regression** (pre-M2b these carried correct
  CF metadata; M2b lost them to `{}`), not a neutral baseline. Whether R4 fixes
  it is a design decision, ruled below (out-by-default, user-ratified).

R4 is where R3's inherited patterns get copied for the first time; R5 copies
them a second time. Getting the inheritance clean here is what makes R5
mechanical.

## Approach

Same contract-first, behavior-preserving discipline R3 used, applied to
workflow 2:

1. **Contract before code.** `dev/reference/workflows/climate_projections.md` (format per
   `dev/milestones/r01/modularity-contracts-design.md` §4, mirroring the R3-authored
   `dev/reference/workflows/model_creation.md`) is R4's first commit, written against
   *current* behavior. Code commits then change behavior against a recorded
   contract, not from memory.
2. **Behavior-preserving throughout; `check_baseline check` stays clean, zero
   manifest edits by default.** Every hygiene commit (log/benchmark directives,
   ruleorder comment + AGENTS.md correction, label renames, the `tee_to_log`
   wraps, the import guards) leaves the 7 workflow-2 manifest targets
   byte-identical and needs no `record`. The one bounded exception is
   error-behavior hardening surfaced by the script review (§"What changes" 4) —
   output-neutral on a *valid* config, an error-behavior change only on an
   invalid one, mirroring R3's §7.1 posture.
3. **Audit-then-classify, don't audit-then-edit.** The chain audit
   (§"What changes" 5) is a deliverable that produces a written findings table
   **backed by executable evidence with normative pass/fail criteria** (§7
   audit-evidence matrix, ext1-2/ext2-2). Each finding is classified **before**
   any code moves (finding-classification policy, §"Finding-classification
   policy"). Noise is documented and left; a defect that would move an output is
   split to a dedicated task **with a named owner and activation condition**
   unless explicitly chartered; only an intentional, chartered change moves the
   baseline with a `dev/milestones/r04/baseline_diffs.md` entry.
4. **Milestone boundaries beat followup tags, with two deliberate exceptions.**
   Findings outside workflow 2 go to `dev/followups.md`; R4 does not touch
   `Snakefile_model_creation` or `Snakefile_climate_experiment` content
   (roadmap iron rule; the shared-helper inheritance already landed in R3).
   **The one repo-root file R4 does edit is `AGENTS.md`** — a one-line
   correction of the refuted ruleorder invariant the milestone itself disproved
   (§3, Group-B fix). **The one dev-script file R4 edits is
   `dev/scripts/check_baseline.py`** — a `--workflow` scope filter chartered to
   make the milestone gate decidable (§"Verification plan", ext1-1/ext2-1); `dev/`
   is tooling territory, allowed. Both are instruction/tooling-surface changes,
   not workflow-content changes, and both are scoped explicitly rather than
   smuggled in.
5. **Naming: applied here.** `dev/reference/naming.md` names R4 as the owning
   milestone for workflow-2 identifiers. Deprecated `_fid` labels *inside this
   Snakefile's rules and its scripts* (`region_fid`, `yml_fid`, `stats_nc_hist`
   as a *path* param) rename to `_path` per naming §3/§5 (lines 57–60, 103–104);
   §7 contract surfaces (output filenames, config keys, catalog names) stay
   grandfathered. `region_fn` inside `get_stats_climate_proj.py` is the local
   read of `snakemake.input.region_fid` — renamed with its label in the same
   commit.

## What changes

### 1. Contract doc `dev/reference/workflows/climate_projections.md` (opening act)

Format per `dev/milestones/r01/modularity-contracts-design.md` §4, mirroring
`dev/reference/workflows/model_creation.md`, target < 100 lines. Content grounded in
`Snakefile_climate_projections` and `config/snake_config_model_test.yml`:

- **Owned config keys** (`workflows.climate_projections.*`, verified against the
  Snakefile's reads, lines 23–33): `clim_project`, `models`, `scenarios`,
  `members`, `variables`, `start_month_hyd_year` (default `"Jan"`),
  `historical_year_range` (read as `time_horizon_hist`), `future_horizons`,
  `save_grids` (default `False`). Note: `enabled:` is present in the config
  (`config/snake_config_model_test.yml` line 36) but is **not read** by the
  Snakefile — documented as a known dormant key (operationalizing it is R6,
  roadmap; non-goal here).
- **Reads from `project`** (lines 17–21): `project.project_dir`,
  `project.data_sources_climate` (read as `DATA_SOURCES`, the hydromt catalog,
  e.g. `config/cmip6_data.yml`). *Note the divergence from workflow 1:* workflow
  2 reads `data_sources_climate`, not `data_sources` — documented so the
  contract doc does not imply a shared key.
- **Input contract** — the cross-workflow input `{basin_dir}/staticgeoms/
  region.geojson` (produced by workflow 1, consumed as `ancient(...)` by
  `monthly_stats_hist`/`_fut`); and the CMIP6 catalog sources named
  `{clim_project}_{model}_{scenario}_{member}` resolved from
  `config/cmip6_data.yml` (enumerated during drafting from the catalog, not
  invented here).
- **Output contract**, split by role:
  - *Direct `rule all` targets* (Snakefile lines 44–51):
    `annual_change_scalar_stats_summary.{nc,csv}`,
    `annual_change_scalar_stats_summary_mean.csv`,
    `plots/projected_climate_statistics.png`,
    `plots/precipitation_anomaly_projections_abs.png`,
    `plots/temperature_anomaly_projections_abs.png`,
    `{project_dir}/config/snake_config_climate_projections.yml`.
  - *Intermediate `temp(...)` artifacts* (deleted after consumers finish; NOT
    manifest targets): `historical_stats_time_{model}.nc`,
    `stats_time-{model}_{scenario}.nc`,
    `annual_change_scalar_stats-{model}_{scenario}_{horizon}.nc`.
  - *Side-effect artifacts*: `gcm_timeseries.nc` (written by
    `plot_climate_proj_timeseries` but declared as `timeseries_csv` output,
    line 154 — the label/extension mismatch is a documented known state, not
    fixed in R4 unless the audit charters it); per-rule logs/benchmarks
    (ephemeral, not tracked, not committed — inherited convention).
- **Downstream consumers** — the merged summary `.nc`/`.csv` and the response
  plots are terminal for workflow 2. Per the method invariant (`AGENTS.md`),
  these are a **plausibility overlay only**; nothing in workflow 3 consumes
  them to drive stress-test runs. The contract doc states this explicitly so a
  future reader cannot mistake the change factors for stress-test forcing.
- **`save_grids` behavior** — documented: when `False` (the seed default) the
  gridded branches in `get_stats_climate_proj.py` / `get_change_climate_proj.py`
  / `plot_proj_timeseries.py` are skipped and several params
  (`stats_nc_hist`/`stats_nc` grid paths, `change_grids`) are never written.
  The contract records which outputs are grid-gated.
- **Known metadata regression flagged in the contract** — the contract doc
  records that `annual_change_scalar_stats_summary.nc` currently ships **empty
  CF `attrs`** (`{}`) on `precip`/`temp` as a consequence of the M2b hydromt-1.3
  regression (candidate #1), so a reader of the sealed workflow does not mistake
  the empty metadata for correct output. This is a documentation entry, not a
  fix.

### 2. Per-rule `log:` + `benchmark:` (inherited R3 pattern)

Apply the R3 convention verbatim (`Snakefile_model_creation` lines 71–89 as the
shape reference; path convention from `dev/milestones/r03/model-builder-design.md` §6).
**Exclusion criterion (restated to be principled):** every rule whose `script:`
performs computation or non-trivial IO gains `log:`/`benchmark:`; the sole
exclusion is `copy_config`, whose `script:` (`src/copy_config_files.py`)
performs only a **deterministic verbatim file copy**. Note that `copy_config`'s
output *is* a baseline-gated target (`snake_config_climate_projections.yml`,
`TARGETS` line 66, fingerprinted as the yaml snapshot); it is excluded on the
"no computation to profile" ground, not because it is untracked. (arch-4:
resolved — the exclusion now agrees with the criterion and acknowledges the
gated output.) Rules that gain `log:`/`benchmark:`:

- `monthly_stats_hist` (`src/get_stats_climate_proj.py`)
- `monthly_stats_fut` (`src/get_stats_climate_proj.py`)
- `monthly_change` (`src/get_change_climate_proj.py`)
- `monthly_change_scalar_merge` (`src/get_change_climate_proj_summary.py`)
- `plot_climate_proj_timeseries` (`src/plot_proj_timeseries.py`)

Path convention (inherited verbatim): `log:` →
`{project_dir}/logs/{rule}.log`; **for the wildcard rules**
(`monthly_stats_hist` on `{model}`, `monthly_stats_fut` and `monthly_change` on
`{model}_{scenario}(_{horizon})`), the log path **must** embed the wildcards —
`{project_dir}/logs/{rule}/{model}...{scenario}...log` — so concurrent jobs
(`snakemake -c 3`) never collide on one path. `benchmark:` →
`{project_dir}/benchmarks/{rule}.tsv` (Snakemake native TSV; also wildcard-keyed
for wildcard rules). Logs/benchmarks are **ephemeral run artifacts** — under
gitignored `{project_dir}/`, never fingerprinted, never committed.

All five rules use `script:` (verified — none use `shell:`), so **none** get
Snakemake auto-redirection. Each script wraps its top-level body in
`tee_to_log(snakemake.log[0])` from `src/snake_utils.py` (already implemented
and unit-tested in R3, `src/snake_utils.py` lines 76–110). This is a mechanical
wrap of existing module-level code; it changes no computed value. Adding this
wrap requires the scripts to reference `snakemake.log[0]`, so the `log:`
directive and the wrap land in the **same commit per script** to keep each
commit runnable.

### 3. `ruleorder:` — reproduce-or-refute at implementation, then a DETERMINED action (ext1-3)

`Snakefile_climate_projections` line 53:
`ruleorder: monthly_stats_hist > monthly_stats_fut > monthly_change >
monthly_change_scalar_merge`. `AGENTS.md` (lines 95–96) records this directive
as load-bearing ("wildcard inference is ambiguous without it").

**Empirical finding (verified during this design, 2026-07-19, scratchpad
dry-run on the current pinned Snakemake; independently reproduced by the
internal architecture reviewer):** commenting line 53 out and dry-running
workflow 2 on both the tests fixture config (3 models, 2 scenarios,
slash-bearing CMIP6 model names) **and** a reduced config (2 simple model names,
1 scenario) **builds a clean 19-job DAG with exit 0 in both cases — no
`AmbiguousRuleException`.** The `AGENTS.md` claim did not reproduce.

**Operationally decidable resolution (ext1-3).** The prior draft (v2) left a
two-branch fork — "keep-and-comment **or** remove behind the gate" — that let
two implementers reach opposite DAG-semantics decisions from the same evidence.
v3 collapsed this to a **single deterministic action** (carried forward
unchanged in v4):

- **R4 always RETAINS the directive and comments it in place.** R4 does **not**
  remove it. Removal is out of R4 scope; it is chartered to a dedicated future
  task whose activation condition is *encoding the ambiguity-sensitive
  configuration shapes as regression tests* — work R4 does not perform. This
  removes the discretion the finding flagged: on this milestone the directive
  stays, full stop.
- **The supported configuration-shape matrix is named, not left as "any
  supported config".** The reproduce-or-refute step dry-runs exactly two shapes
  (both already exercised in this design): **(M1)** the tests fixture
  (`config/snake_config_model_test.yml`: 3 models incl. two slash-bearing CMIP6
  names `NOAA-GFDL/GFDL-ESM4`, `INM/INM-CM4-8`, `INM/INM-CM5-0` × scenarios
  `[ssp245, ssp585]` × 1 horizon `far` × 1 member) and **(M2)** a reduced shape
  (2 simple non-slash model names × 1 scenario × 1 horizon). This is the
  representative matrix; it spans the underscore/slash boundary conditions most
  likely to trip wildcard inference.
- **The reproduce step still runs — but only to fix the *comment text*, never
  the *decision*.** Two outcomes, each with **exact AGENTS.md text specified**:
  - **Reproduce case** (a real `AmbiguousRuleException` is produced with the
    directive removed on M1 or M2): capture the exact message (it names the
    conflicting rules and the contested file); that verbatim message becomes the
    in-place Snakefile comment. **AGENTS.md lines 95–96 are confirmed correct and
    left byte-for-byte unchanged** ("The `ruleorder:` directive in
    `Snakefile_climate_projections` is load-bearing; wildcard inference is
    ambiguous without it.").
  - **Refute case** (no exception on either M1 or M2 — the outcome this design's
    dry-run already observed): the in-place Snakefile comment reads
    *"Retained as stale insurance pending removal task; dry-run on the current
    pinned Snakemake shows this directive constrains nothing on the tests fixture
    and a reduced config (see `dev/milestones/r04/...`). Removal deferred to a task that
    first encodes the ambiguity-sensitive config shapes as regression tests."*
    **AGENTS.md lines 95–96 are replaced with exactly:** *"The `ruleorder:`
    directive in `Snakefile_climate_projections` is retained as stale insurance,
    not confirmed load-bearing: a 2026-07 dry-run on the pinned Snakemake showed
    it constrains nothing on the tests fixture and a reduced config. Removal is
    deferred to a task that first encodes any ambiguity-sensitive config shapes
    as regression tests (see `dev/milestones/r04/...`)."*

There is now **no branch in which R4 removes the directive** and **no
undefined "any supported config"** — the matrix is the two named shapes, and the
only thing the evidence selects is which of two pre-written comment/AGENTS.md
texts is committed. **Tightening via a `wildcard_constraints` rewrite is not
attempted in R4** — that *would* be a DAG-semantics change (which paths a rule
can match), the class R3 protected as territory (§2, R3 design), and it is out
of scope here.

**`AGENTS.md` correction is part of this commit (Group-B fix; risk-4 / arch-2 /
repo-3).** In the refute case, a Snakefile-local comment does **not** reach a
future reader of `AGENTS.md`, the repo's canonical single-source-of-truth
instruction file for every runtime (its own header). Leaving line 95 asserting
"load-bearing … ambiguous without it" — a claim R4 recorded as empirically
false — would mislead R5 (which inherits these patterns) and any future agent
into preserving dead insurance or reintroducing an equivalent directive: exactly
the stale-claim propagation §3 exists to end. Therefore the refute-case AGENTS.md
edit above is mandatory in that outcome; the reproduce case leaves AGENTS.md
untouched. **Open question (below):** which of {stale-from-a-pre-migration-version
/ config-shape-triggered-only / inaccurate-claim} explains the non-reproduction
is resolved (documented, not gated) at implementation — but the *action* R4
takes (retain-and-comment) is the same regardless.

### 4. `src/get_stats_climate_proj.py` — correctness / vectorization / units review

The roadmap names this script specifically. Documented review targets, grounded
in the current code:

1. **Bare `except:` masking (correctness).** Lines 199 and 214 use bare
   `except:` around `data_catalog.get_rasterdataset(...)`. The outer one falls
   back to a per-variable loop; the inner one prints "not found" and continues.
   A bare `except:` swallows `KeyboardInterrupt`/`SystemExit` and any real bug
   (typo, dependency error) as "data not found," producing a silently empty
   result. **Classification: this is a correctness defect, but narrowing it
   (to `except Exception` or a specific hydromt exception) is behavior-changing
   on the error path** — it could turn a currently-swallowed condition into a
   hard failure and move which models produce empty datasets, hence move the
   baseline. So R4 **documents** it and narrows only if the narrowing is
   provably output-neutral on the seed config (i.e. the seed run never hits the
   fallback). If it does hit the fallback, the fix is split to a dedicated task
   with a named owner + activation condition (finding-classification policy).
   Recorded either way, not silently rewritten.
2. **Units handling (documentation + assertion).** Precip is resampled with
   `.sum` and temp with `.mean` (lines 127–131) — the multiplicative/additive
   split that the whole change-factor method depends on. The `precip` vs
   `else`-is-temp dispatch is **string-name-based** (`if var == "precip"`), so a
   catalog that named the precip variable anything else would silently be
   treated as temp. R4 records the required variable naming (`precip`, `temp`
   per the seed `variables: [precip, temp]`) in the contract doc and adds a unit
   table (precip flux → summed to monthly total; temp → monthly mean) as
   documentation. **No computed value changes.**
3. **Vectorization (advisory).** The member loop (lines 179–235) and the
   per-variable fallback are Python-level loops over small dimensions
   (`members: [r1i1p1f1]`, 2 variables) — not a hot path; the `.load()` on line
   198 is the real cost and is already the M2b throughput fix (retired
   followup, `dev/followups.md` lines 332–337). R4 leaves the loops as-is and
   records that the eager `.load()` is deliberate. No rewrite.

Outcome of this review is **documentation + at most an output-neutral guard**,
never a silent semantic change.

### 5. The change-factor chain audit (the scientific deliverable)

Audit `monthly_stats_hist → monthly_stats_fut → monthly_change` end-to-end for
unit consistency, calendar handling, and missing-data behavior. Named rules and
their outputs (all verified in the Snakefile):

- **`monthly_stats_hist`** → `historical_stats_time_{model}.nc` (`temp`) via
  `get_stats_climate_proj.py` with `name_scenario="historical"`.
- **`monthly_stats_fut`** → `stats_time-{model}_{scenario}.nc` (`temp`), same
  script, `name_scenario={scenario}`; takes the historical file as an input to
  force ordering.
- **`monthly_change`** → `annual_change_scalar_stats-{model}_{scenario}_
  {horizon}.nc` (`temp`) via `get_change_climate_proj.py`.
- (chain terminus, for baseline context) **`monthly_change_scalar_merge`** →
  the merged `annual_change_scalar_stats_summary.{nc,csv}` +
  `_summary_mean.csv` via `get_change_climate_proj_summary.py`.

**This audit is not inspection-only (ext1-2).** Each audit question below is
paired with a **mandatory executable check** in the §7 audit-evidence matrix,
and each findings-table row must cite **either a passing synthetic test or a
concrete traced result** — a row backed by "read the code and it looks fine" is
not an acceptable end state for the high-risk questions (calendar boundaries,
variable intersection, partial members, dummy datasets). **Every matrix row
carries a normative expected outcome; a row whose observed behavior differs from
its norm FAILS and yields a defect-class finding (ext2-2).** Specific audit
questions, each grounded in the code:

- **Unit consistency.** `get_stats_climate_proj.py` sums precip and means temp
  (lines 127–131); `get_change_climate_proj.py` applies **multiplicative**
  change for precip (`(clim-hist)/hist*100`, lines 47–59, 160–161) and
  **additive** for temp (lines 62, 162–163). Confirm the hist/fut monthly stats
  fed to `monthly_change` are in the same units the change formula assumes (precip
  as a monthly total to be summed to annual before the percent change on lines
  116–130; temp as a monthly mean). Confirm the percent-vs-degree split in the
  merge/plot layer (`get_change_climate_proj_summary.py` axis labels lines
  120–123; `plot_proj_timeseries.py` `%` vs `degC`) matches. *Evidence:* the
  precip-multiplicative / temp-additive unit test (§7, matrix row U).
- **Calendar handling (now unconditional, ext1-2).** `get_change_climate_proj.py`
  builds hydrological-year boundaries with
  `pd.to_datetime(f"{year}-{start_month_hyd_year}")` and resamples with
  `YS-{MON}` (lines 100–148). CMIP6 uses `cftime` (`Datetime360Day`,
  `DatetimeNoLeap`, …) — the code comment at `get_stats_climate_proj.py` line 191
  acknowledges this, and the once-considered `to_datetimeindex()` conversion is
  present in `plot_proj_timeseries.py` (lines 36–43) but **not** in
  `get_change_climate_proj.py` (its commented-out block, lines 224–232, shows it
  was considered and dropped). The v2 conditionality ("audit whether … a
  calendar mishandling could silently drop or misalign months", exercised **only
  if the reading flags it**) is exactly what ext1-2 rejects. **The calendar test
  is mandatory and multi-calendar:** the §7 matrix rows C1–C3 feed
  `get_change_annual_clim_proj` synthetic hist/fut datasets on a
  `Datetime360Day`, a `DatetimeNoLeap`, and a proleptic-Gregorian/`datetime64`
  index respectively, and assert the number of retained hydrological years and
  the month membership per year against a hand-computed expected value. This is
  the highest-risk audit item; it is now *tested*, not merely *read*.
- **Non-January hydrological-year boundary (now explicit, ext1-2).** The
  `start_month_hyd_year` boundary math (`pd.to_datetime(f"{year}-{mon}")` minus
  one month for the end; `resample("YS-{MON}")`) is exercised for the default
  `"Jan"` **and** at least one non-January month (`"Oct"`) — §7 matrix row H — to
  assert that a non-January hydrological year keeps the right whole-year window
  and does not silently drop the leading/trailing partial year incorrectly.
- **Missing-data behavior (now tested with normative outcomes, ext1-2/ext2-2).**
  Both `get_change_climate_proj.py` (lines 254, 283) and
  `get_change_climate_proj_summary.py` (lines 54–59) handle **dummy empty
  netCDFs** (`len(ds)==0`) written when a catalog entry is absent. The audit must
  assert, with executable evidence **and a normative pass/fail criterion**, three
  distinct behaviors:
  - a missing model/scenario produces a dummy that the merge correctly skips
    (the `list_files_not_empty` filter in `summary_climate_proj`,
    `get_change_climate_proj_summary.py` lines 54–59) — §7 matrix row D
    (dummy-skip integration test, synthetic empty-vs-nonempty pair). **Norm: the
    empty file is excluded and the non-empty file is retained; this is the
    intended, correct behavior — row D PASSES on the current code.**
  - an **asymmetric-variable** case where `ds_hist` carries `{precip, temp}` but
    `ds_clim` carries only `{precip}` (or vice-versa). The `intersection()` at
    `get_change_climate_proj.py` lines 18–19, 45, 98 (`list(set(a) & set(b))`)
    means the change loop iterates **only the shared variables**, so `temp` is
    **silently dropped from the result with no error and no warning**. **Norm
    (ext2-2, pinned single-policy by the 2026-07-20 arbitration, ext3-1):
    silent variable-dropping is NOT acceptable for a plausibility overlay whose
    summary must carry the configured variable set — an asymmetric-variable
    input MUST FAIL LOUDLY: raise `ValueError` with a message naming the
    variables present on each side and the missing one(s) (exception contract
    in §7). Explicit reconciliation is NOT the norm; adopting it later would be
    a norm revision owned by the split task, with the §7 normative test revised
    there. §7 matrix row V asserts the raise; because the current code silently
    intersects (no raise), row V is EXPECTED TO FAIL, and that failure yields a
    defect-class finding** (split, don't absorb — see below). See §7 for the
    exact synthetic input and surviving coordinates.
  - a **partial-member** case (a member present in one dataset, absent in the
    other). In `get_change_annual_clim_proj` the change is computed as
    `clim_stat - hist_stat` (temp, line 163) / `(clim_stat - hist_stat)/hist_stat`
    (precip, line 161) on xarray objects that still carry the `member` dimension;
    xarray's default binary-op alignment (`arithmetic_join="inner"`) **silently
    intersects the `member` coordinate**, so a member present in only one side is
    **dropped without error**, and the downstream member-mean in the merge is
    then taken over the surviving intersection only. **Norm (ext2-2, pinned
    single-policy by the 2026-07-20 arbitration, ext3-1): a member present in
    one dataset and absent in the other must NOT be silently dropped — the
    asymmetry MUST FAIL LOUDLY: raise `ValueError` with a message naming the
    unshared member(s) and each side's member set (exception contract in §7).
    Explicit alignment is NOT the norm; adopting it later would be a norm
    revision owned by the split task, with the §7 normative test revised there.
    §7 matrix row P asserts the raise; because the current code silently
    inner-joins (no raise), row P is EXPECTED TO FAIL, yielding a defect-class
    finding** (split; ext1-2 evidence characterizes, it does not authorize the
    fix). See §7 for the exact synthetic input and surviving coordinates. *(The precise xarray join semantics on the
    `member` dim are asserted by the §7 characterization test rather than by
    inspection — if the run reveals a different mechanism than inner-join, the
    characterization assertion is corrected to the observed coordinates; the
    normative row is unaffected, since its criterion is the raise (ext3-1).
    Recorded as an open question.)*
- **M2b attrs localization (unconditional obligation of this audit; ext1-4).**
  The §5 audit **must** localize the M2b attrs drop (candidate #1) via a
  **runtime diagnostic probe** (§"M2b attrs diagnostic probe" below), not by
  source inspection — attribute propagation is runtime- and
  operation-dependent, so reading the code cannot reliably say where `.attrs`
  vanish (ext1-4). Localization is defined as **the first transformation at
  which the expected `.attrs` are observed empty** (ext2-3 sharpens this from
  v3's "first pipeline stage"). The acceptable end states are (a) a pinned single
  workflow-2 line/operation, or (b) a documented **dependency reproducer**
  showing the loss occurs inside an isolated hydromt/xarray operation (e.g.
  `open_mfdataset` with `coords="minimal"`/`preprocess`) rather than in
  workflow-2 code. "Cause not localized to *either* a workflow-2 stage/operation
  *or* an isolated dependency op" is not an acceptable end state; "we localized
  it to hydromt with a reproducer" is.

**The audit produces a written findings table in the contract doc (or a
`dev/milestones/r04/chain-audit.md` if it exceeds the contract doc's scope), with an
`Evidence` column citing the §7 test name or the traced result per row, and an
`Outcome` column recording pass/fail against the row's norm.** It does not, by
itself, change code. Whether any finding moves to code is decided by the
finding-classification policy below; a failing row (V, P, or any other) routes
to the defect class.

#### M2b attrs diagnostic probe (ext1-4, ext2-3)

The probe records `precip`/`temp` `.attrs` at a chain of checkpoints. **v4
extends the checkpoint set INSIDE per-model generation (ext2-3):** v3's first
checkpoint was already *after* the entire per-model change computation and write,
so an empty-attrs reading there localized the loss only to a large upstream
region while the design simultaneously demanded a single workflow-2 operation or
an isolated reproducer. v4 threads checkpoints through the per-model computation
so the *first transformation* that drops attrs is pinnable.

**Per-model generation checkpoints (new in v4; inside
`get_change_climate_proj.py` → `get_change_annual_clim_proj`, run against a
synthetic or `--notemp`-preserved input carrying known non-empty CF attrs on
`precip`/`temp`):**

- **P0 — known-attrs input, before change computation.** Assert the input
  `ds_hist_time` / `ds_clim_time` carry the expected `precip`/`temp` `.attrs`
  (establishes the baseline the loss is measured against; if attrs are already
  empty here the loss is upstream in `get_stats_climate_proj.py` / the catalog
  read, which is then the localization).
- **P1 — after the `.sel(time=slice(...))` window trim** (lines 255–256, and
  the per-var `.sel(time=slice(...))` at 118/127/135/144).
- **P2 — after `.resample(...).sum("time")` / `.resample(...).mean("time")`**
  (the precip/temp reduction, lines 119/128/137/146) — a reduction is a classic
  attr-dropping operation under `keep_attrs` defaults.
- **P3a — after the statistic reduction alone** (`getattr(hist, stat_name)("time")`
  / `.quantile(qvalue, "time")`, lines 154–158) — reductions drop attrs under
  `keep_attrs` defaults. *(ext3-2: one operation per checkpoint — the v4 P3
  bundled reduction + arithmetic and could localize only to the pair.)*
- **P3b — after the change arithmetic alone**
  (`change = (clim_stat - hist_stat)/hist_stat*100` for precip /
  `clim_stat - hist_stat` for temp, lines 160–163) — binary ops drop attrs
  unless `keep_attrs` is set.
- **P4a — after the coordinate operations alone**
  (`.assign_coords({"stats": stat_name}).expand_dims("stats")` plus the
  conditional `.drop("quantile")`, lines 164–167) — coord assignment can drop
  variable-level attrs. *(The chained `assign_coords`/`expand_dims`/`drop` are
  one checkpoint of the "coordinate operation" category per ext3-2's
  suggested_fix; the stepwise script below can further split them at zero cost
  if this checkpoint is where attrs vanish.)*
- **P4b — after `.to_dataset()` collection and `xr.merge(ds)`** (lines
  168–170) — `xr.merge` is a known attr-dropping site (`combine_attrs`
  defaults).
- **P5 — immediately before `to_netcdf`** (line 278) — the last in-memory state.
- **P6 — after reopening the written per-model file**
  (`annual_change_scalar_stats-{model}_{scenario}_{horizon}.nc`) — captures any
  loss on serialization/encoding.

**Localization (ext2-3): the first checkpoint in the chain P0, P1, P2, P3a,
P3b, P4a, P4b, P5, P6 at which the expected attrs are empty is the localizing
operation** — with the compound v4 checkpoints split (ext3-2), every checkpoint
now brackets a single operation (or, for P4a, a single coordinate-op category
that the stepwise script can subdivide on demand), so the first empty reading
pins one operation with no further bisection needed. The audit then requires an
**isolated reproducer for exactly that operation** — a minimal synthetic
`DataArray`/`Dataset` carrying known attrs fed through only that operation (e.g.
just `.resample(...).mean("time")`, or just the binary subtraction) showing the
attrs vanish with no other workflow-2 code in the loop. That reproducer is the
ext2-3 evidentiary threshold for a per-model loss, and it also discharges
ext1-4's "dependency reproducer" end state when the operation is a
library/xarray default (e.g. `keep_attrs=False` on a reduction).

**Probe mechanism and artifact ownership (ext3-2, arbitration-mandated).** The
probe is a **standalone diagnostic script at `dev/scripts/probe_attrs_chain.py`**
(dev-territory, committed with the milestone as audit evidence, like the
commit-2b `check_baseline` work). It does **not** instrument
`src/get_change_climate_proj.py`: it **re-executes the operation sequence of
`get_change_annual_clim_proj` stepwise** — the same xarray calls with the same
arguments as lines 98–171, one statement per checkpoint — on the synthetic
known-attrs input, recording `.attrs` after each step. A final **cross-check**
runs the *intact* production function on the same input and asserts its
end-state attrs match the stepwise chain's P4b/P5 state, guarding against the
mirror diverging from production. The M-checkpoints likewise drive
`open_mfdataset(coords="minimal", preprocess=preprocess_coords)` + `to_netcdf`
directly. Under this mechanism **no computational-code edit exists at all** —
the milestone's diagnostic-only, output-neutral claim holds by construction. If
implementation nevertheless chooses the `--notemp` real-run corroboration path
with temporary `print`/log instrumentation inside `src/`, that instrumentation
**must be removed before the milestone gate and verified removed by
`git status --short` / `git diff` showing a clean `src/` tree** — a dirty
`src/` at the gate is a gate failure. Probe outputs (the per-checkpoint attrs
table) land in the audit findings table / `dev/milestones/r04/chain-audit.md`.

**Post-generation (merge) checkpoints (retained from v3, for losses that occur
after per-model generation):**

- **M1 — after `open_mfdataset(coords="minimal", preprocess=preprocess_coords)`**
  — the merged in-memory `ds` in `summary_climate_proj`
  (`get_change_climate_proj_summary.py` lines 60–62), inspected **before** the
  `to_netcdf`.
- **M2 — after merge write + reopen** — the on-disk
  `annual_change_scalar_stats_summary.nc` re-opened (this is the manifest
  target `check_baseline.py` line 116 fingerprints).

If the whole per-model chain (P0 through P6) shows non-empty attrs but M1/M2
show empty, the loss is post-generation and localizes to the merge path
(`open_mfdataset` + `coords="minimal"`/`preprocess`, or the summary
`to_netcdf`) — the dependency-reproducer end state applies there, exactly as v3
specified.

**Operational trap the probe must handle:** the per-model
`annual_change_scalar_stats-*.nc` files are `temp(...)` and are **deleted after
the merge consumes them**, so a naive post-hoc probe of P6/M-side files finds
nothing on disk. The probe **must** run either (a) under `snakemake --notemp` so
the intermediates survive for inspection, or (b) the default mechanism above:
the **synthetic** known-attrs input driven through the **stepwise operation
chain** mirroring `get_change_annual_clim_proj` (for the P-checkpoints, with
the intact-function cross-check) and, separately, through
`open_mfdataset(coords="minimal", preprocess=...)` + `to_netcdf` (for the
M-checkpoints). **The synthetic path is sufficient for per-model localization
(ext2-3): the stepwise P-checkpoints reproduce the per-model operation
sequence on a known-attrs input, so they localize a loss during per-model
generation without needing the real merge path** — closing
v3's gap where the synthetic alternative exercised only the merge path. The probe
is a **reading/diagnostic finding with a concrete traced result**; it restores
nothing and moves no baseline (the attrs absorption remains out-by-default,
candidate #1). It lives in the audit table / `dev/milestones/r04/chain-audit.md`; it is not
a manifest-touching commit.

### 6. Naming: `_fid` → `_path` in workflow-2 rules and scripts

Per `dev/reference/naming.md` §3/§5 (R4 is the owning milestone), rename
deprecated `_fid`/path-as-`_nc` labels **inside this Snakefile's rules and the
scripts it calls**, with the paired script reads updated in the same commit:

- `region_fid` (Snakefile lines 71, 90; `get_stats_climate_proj.py` line 29
  `region_fn`) → `region_path`.
- `yml_fid` (Snakefile lines 76, 94; `get_stats_climate_proj.py` line 30
  `path_yml`) → a `_path` name (e.g. `catalog_path`).
- Path-valued `params` currently suffixed `_nc` that are *paths* not datasets
  (`get_change_climate_proj.py` `stats_nc_hist`/`stats_nc` params, lines
  179–180) — reviewed against naming §5's path/object rule
  (`dev/reference/naming.md` line 181 explicitly lists `stats_nc` (a path) →
  `stats_path`). Renamed where they are paths; left where they are the netCDF
  *input* labels the DAG keys on (those are §7-adjacent contract labels the
  merge/plot rules match on — grandfathered unless the rename is proven not to
  break wildcard matching). **§7 contract surfaces — output filenames
  (`annual_change_scalar_stats_summary.nc`, etc.), config keys, catalog
  names — stay grandfathered.** No output filename changes → no migration note.

### 7. Unit + integration tests, and the audit-evidence matrix (M02c discipline + ext1-2/ext2-2)

Per the M02c testing discipline (`dev/followups.md` § R3+): **monkeypatch over
`sys.modules.setdefault`** for shared heavy deps (hydromt, xarray, dask), no
network, no full builds. New tests target the pure/near-pure functions:

- `get_change_climate_proj.get_change_annual_clim_proj` (line 72) — the core
  change formula: multiplicative for precip, additive for temp, on a
  **synthetic in-memory** hist/fut dataset with a known answer; **and** the
  calendar / boundary / variable / member cases in the audit-evidence matrix
  below (no longer conditional on the reading audit flagging them, ext1-2).
- `get_change_climate_proj.get_change_clim_projections` (line 22) — the
  grid-change branch (multiplicative/additive), synthetic input.
- `get_change_climate_proj._to_str_tuple` (line 193) — the R01
  list-vs-comma-string horizon parser; falsey/edge inputs.
- `get_change_climate_proj_summary.preprocess_coords` (line 15) — a real,
  cleanly importable function; synthetic coord input.
- `get_stats_climate_proj.get_stats_clim_projections` — the sum-precip /
  mean-temp resample and the dim-name detection (`XDIMS`/`YDIMS`), synthetic
  input. (This is the vectorization/units function the roadmap names.)

**Audit-evidence matrix (ext1-2/ext2-2) — the minimum executable evidence the §5
audit must produce.** Each row is a synthetic test with a **hand-computed
normative expected outcome** and an **Outcome** column (`PASS` = code matches
norm; `FAIL` = code violates norm → defect-class finding). Each maps to a §5
findings-table row. **A row whose expected outcome is `FAIL` is a design
prediction that the current code violates the norm; the test must be written to
assert the norm (not the current behavior), so it fails against today's code and
that failure is the executable evidence the finding is a defect (ext2-2). It
does NOT authorize an inline fix — the failing row routes to the defect class
(split, don't absorb).**

| Row | Target function | Synthetic input | Asserts (the norm) | Expected outcome |
| --- | --------------- | --------------- | ------------------ | ---------------- |
| **U** | `get_change_annual_clim_proj` | precip + temp hist/fut, known values | precip change multiplicative `(c−h)/h*100`, temp additive `c−h` | **PASS** — exact known scalar per var |
| **C1** | `get_change_annual_clim_proj` | hist/fut on `cftime.Datetime360Day` index | # retained hydro-years + month membership = hand-computed window | **PASS** if window matches; **FAIL** (defect) if months silently dropped/misaligned |
| **C2** | `get_change_annual_clim_proj` | hist/fut on `cftime.DatetimeNoLeap` index | same | same |
| **C3** | `get_change_annual_clim_proj` | hist/fut on proleptic-Gregorian `datetime64` | same | same |
| **H** | `get_change_annual_clim_proj` | `start_month_hyd_year="Oct"` (+ `"Jan"` control) | correct Oct→Sep whole-year window retained | **PASS** if window correct; **FAIL** (defect) otherwise |
| **V** | `get_change_annual_clim_proj` | `ds_hist={precip,temp}`, `ds_clim={precip}`, else identical known values | **fail-loud (ext3-1): raises `ValueError` whose message names the per-side variable sets and the missing variable(s)** — e.g. matches `r"asymmetric.*variables.*temp"` | **FAIL (defect, ext2-2)** — current `intersection()` raises nothing and yields a result with **only `precip`** (`temp` silently absent; that observed set is asserted by the *characterization* test, below, not by this row). No raise → normative test fails against today's code → variable-drop defect, split. |
| **P** | `get_change_annual_clim_proj` | hist members `[r1i1p1f1, r2i1p1f1]`, fut members `[r1i1p1f1]`, else identical | **fail-loud (ext3-1): raises `ValueError` whose message names the unshared member(s) and each side's member set** — e.g. matches `r"asymmetric.*members.*r2i1p1f1"` | **FAIL (defect, ext2-2)** — current default `arithmetic_join="inner"` raises nothing and yields a result whose `member` = `[r1i1p1f1]` only (`r2i1p1f1` silently dropped; asserted by the characterization test). No raise → normative test fails against today's code → partial-member defect, split. |
| **D** | `summary_climate_proj` dummy-skip path | synthetic empty-vs-nonempty netCDF pair | empty file excluded from merge, non-empty kept | **PASS** — empty dropped; nonempty kept (intended behavior) |

Rows **C1–C3, H, V, P** discharge ext1-2's "minimum audit evidence matrix …
supported calendar classes, non-January hydrological-year boundaries,
partial-member absence, dummy datasets, and asymmetric hist/future variables";
rows **V and P** now carry **normative pass/fail criteria with exact surviving
coordinates** (ext2-2): the norm is "no silent drop," the current code drops
silently, so both rows are **predicted to FAIL** and each failure is a
defect-class finding routed through split-don't-absorb. Row **U** anchors the
unit split; row **D** is the dummy-skip integration test (a genuine PASS —
dropping empties is correct).

**On rows V and P: one norm, two test kinds (ext2-2 + ext3-1, the latter
user-arbitrated 2026-07-20).** The v3 phrasing ("records observed set /
observed behavior") let an implementer encode whatever the code does as the
assertion and pass; v4 made the assertions norms but left **two** permitted
norms ("fail loudly *or* be explicitly reconciled"), so no test could
distinguish the defect from every permitted fix, and a correct fail-loud fix
would have raised inside the test and stayed an expected `xfail` forever
(ext3-1). v5 closes both gaps:

- **The single norm is fail-loud** (chosen at arbitration to match the repo's
  R3 hardening posture — cf. `setup_gauges` raising on unknown
  `wflow_outvars`). **Exception contract:** `ValueError`, message naming the
  asymmetric dimension (variables for V, members for P), the offending
  element(s), and both sides' sets, so a user can act on it without reading
  source. Explicit reconciliation (`join="outer"`, declared alignment) is
  **not** a permitted alternative under this design; if the owning split task
  later argues reconciliation is scientifically preferable, that is a **norm
  revision decided in that task**, which must revise the normative tests in
  the same change — it cannot silently satisfy them.
- **Normative acceptance tests** (rows V and P) assert the raise:
  `pytest.raises(ValueError, match=...)` on the hardcoded asymmetric synthetic
  inputs (V: `hist={precip,temp}`, `fut={precip}`; P: hist members
  `[r1i1p1f1, r2i1p1f1]`, fut `[r1i1p1f1]`). Against today's
  silently-intersecting code nothing raises, so both **fail
  deterministically**; they are wired `xfail(strict=True, reason=<split-task
  id>)`. **Removal condition:** the marker is removed by the owning split
  task's fix commit — enforced mechanically, because a landed fail-loud fix
  makes the raise happen, flips strict-`xfail` to `xpass`, and fails the
  suite until the marker is deliberately removed (the tripwire ext3-1
  demanded).
- **Characterization tests** (separate functions, plain PASS today, no xfail)
  pin the *current* defective behavior with exact surviving coordinates —
  V-char: result carries only `precip`; P-char: result `member ==
  ["r1i1p1f1"]` — documenting the silent loss as durable evidence.
  **Removal condition:** deleted in the owning split task's fix commit (they
  fail loudly the moment the fix lands, so they cannot linger). *(The precise
  xarray join mechanism on the `member` dim is asserted by the
  characterization test rather than by inspection; if the run reveals a
  different mechanism than inner-join, the characterization assertion is
  corrected to the observed coordinates — the normative row is unaffected,
  since its criterion is the raise.)*

The failures are **already classified as defects** (silent scientific data
loss in a plausibility-overlay summary is not acceptable behavior) and are
**split** — not fixed in R4 — with a named owner + activation condition,
enumerated at the seal (risk-3), because the fail-loud fix is
behavior-changing on the missing-data path — an error-path change on invalid
input, but one that could also surface latent production asymmetries — and
its baseline consequence belongs to the owning task. Whether the defect
*manifests in production* — e.g. an upstream guarantee that real hist/fut
always share variables and members — is a **severity/priority** input for the
owning split task, **not** a path by which the synthetic row passes. The point
ext2-2 requires is that the *norm* is decided in the design; the point ext3-1
adds is that it is decided **uniquely**, with the test mechanics able to tell
defect, fix, and norm-revision apart.

**Dummy-file skip (integration-style, not a unit — row D).** The "empty datasets
are dropped from the merge" behavior lives in the `list_files_not_empty` **local
list variable** inside `summary_climate_proj`
(`get_change_climate_proj_summary.py` lines 54–59) — it is **not** a callable,
so it has no unit entry point (repo-2). R4 covers it with an integration-style
test that drives the dummy-skip path with a synthetic empty-vs-nonempty netCDF
pair and asserts the empty file is excluded, rather than naming a nonexistent
unit. No extract-to-function refactor is chartered for this (parsimony
criterion); if implementation finds the integration test too heavy to mock
(`open_mfdataset` + seaborn `JointGrid` + `savefig`), the fallback is a small
output-neutral `filter_nonempty(clim_files)` extract — but that is a documented
fallback, not the plan.

**Two scripts are not import-clean today, and BOTH need an import guard
(repo-1):**

- `get_stats_climate_proj.py` runs module-level code against `snakemake.*` on
  import (lines 28–74).
- `get_change_climate_proj.py` **also** runs module-level `snakemake.params.*`
  reads on import (lines 178–187). Importing it to reach *any* of its three
  named test targets raises `NameError: name 'snakemake' is not defined` at
  line 178. Critically, `_to_str_tuple` is **defined at line 193, after** the
  exec block begins at line 178 — so a naive "guard the trailing body" wrap
  would leave `_to_str_tuple` inside the guard and still unimportable. The guard
  refactor for this script **must relocate `_to_str_tuple` (and the two
  `get_change_*` functions if not already above) above the guarded
  `snakemake.*` exec block.** (Both `get_change_*` functions are already above
  the exec block at lines 22 and 72 — verified; the relocation obligation is
  binding for `_to_str_tuple`.)

The **exact guard shape** both new guards must copy is the *nested* form both
sibling scripts already use (`copy_config_files.py` lines 59–60;
`get_change_climate_proj_summary.py` lines 129–130 — verified) — **not** an
either/or (repo-4):

```python
if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        ...   # module-level snakemake.* reads and the main call
    else:
        ...   # standalone fallback / raise
```

Both guard refactors are **output-neutral structural commits** — they make the
functions importable without changing any computed value. The guard preference
over a `runpy`+mock harness stands: a `runpy` harness driving module-level
`snakemake.*` code is brittle and was part of the M02c mocking pain the intake
calls out. (Both guard refactors are asserted output-neutral but **unverified
until run** — recorded as an open question.)

## Finding-classification policy (pre-committed, before the audit runs)

Every audit finding (§5) and every script-review finding (§4) is assigned one
of three classes **before** any code is written, and the class determines what
happens. **The §7 audit-evidence matrix is the input to this policy, not a
bypass of it:** a matrix row that fails still routes through the defect class —
executable evidence characterizes the behavior, it never authorizes an inline
fix (ext1-2/ext2-2). **Rows V and P are pre-classified as defects here (their
expected outcome is FAIL against the norm), so their disposition is decided
before implementation, not discovered by it.**

| Class | Definition | Action | Moves baseline? |
| ----- | ---------- | ------ | --------------- |
| **Noise** | Cosmetic / stylistic / already-correct; or a difference that cannot change any output value (e.g. a clarifying comment, a documented dormant key, a loop that is provably equivalent). | Document in the contract doc / audit table. Leave code as-is. | No. |
| **Defect** | A real correctness bug whose fix **would change an output value** or an error-path outcome (e.g. a calendar misalignment that shifts a monthly mean; a silent variable drop on asymmetric variables — row V; a silent partial-member inner-join — row P; narrowing a bare `except` that currently swallows a real error). | **Do not fix inline** (this holds even when a §7 matrix row *proves* the defect — evidence characterizes, it does not license the fix). Split to a dedicated task (à la R3's `t260719a`) that **names a concrete owning milestone or task and an explicit activation condition** — recorded in `dev/followups.md` **and** `dev/TODO.md`, **and enumerated in the roadmap R4 seal** (see below). "Split" means deferral-with-an-owner, never deferral-into-the-void. R4 stays behavior-preserving. **Exception:** an output-neutral hardening (loud error on an *invalid* config, no change on the *valid* seed config) may land in R4 as an error-behavior change, mirroring R3 §7.1 — classified as such in the commit message. | No, unless explicitly re-chartered as an intentional change (next row). |
| **Intentional change** | A finding R4 *elects* to fix in-scope with eyes open (e.g. restoring the M2b attrs — see below), accepting the output move. | Fix in a dedicated commit; record the exact baseline delta in `dev/milestones/r04/baseline_diffs.md`; run `check_baseline record` for **only** the affected workflow-2 target(s); state the scientific justification. | **Yes** — this is the only class that edits the manifest. |

The governing rule: **the audit is a reading+testing exercise that produces a
classified findings table backed by executable evidence; only the "intentional
change" class edits the baseline, and only with a `baseline_diffs.md` entry.**
This is the mechanism that keeps R4 behavior-preserving while still delivering a
real scientific audit. The default disposition for a genuine bug is **split,
don't absorb** — matching R3's treatment of the CSDMS constants.

**Forcing function against silent deferral (risk-3).** A "defect" split out of
R4 is only honest if it is (a) enumerated at the seal, in the roadmap R4 section
itself, not only buried in `dev/followups.md`; and (b) tagged with a concrete
owner (a named future milestone or a `t2607xx` task) and an activation
condition. The seal language (commit 10) must distinguish **"audited, no
defects"** from **"audited, defects found and deferred"** — a future reader of
the roadmap R4 section must not be able to infer "chain audit delivered" as
"change-factor chain validated clean" when a monthly-mean-shifting defect (or the
row-V variable-drop / row-P member-drop defects) was in fact found and parked.
R3's `t260719a` split a *scoped, characterized* constant change; R4's split
policy inherits that standard — a split defect must be characterized (localized,
baseline-consequence stated; and, where a §7 matrix row applies, backed by the
failing test) before it is parked, never left fuzzy.

## Candidate absorptions — rulings

The intake requires an explicit in/out decision on three items.

**1. CMIP6 `.attrs` loss on `monthly_change_scalar_merge` (M2b;
`dev/followups.md` lines 316–322) — RULING: OUT of R4's default scope
(user-ratified at G1); split to a dedicated task with a named owner and
activation condition.**

**Framing correction (risk-1).** This target is a **documented regression**,
not a neutral baseline. Pre-M2b, `annual_change_scalar_stats_summary.nc` carried
correct CF metadata (`cell_methods`, `standard_name`, `units`, `long_name`,
`original_name`, …); M2b *lost* those to `{}` under hydromt 1.3, recorded in
`dev/milestones/phase-1/m02b/baseline_diffs.md`. Because `check_baseline.py` line 116
fingerprints per-variable `attrs`, restoring the correct metadata makes `check`
go **red**, so restoration is an "intentional change that moves the baseline."
The deferral is therefore a **scientific-cost decision** — R4 elects, for this
milestone, not to open an un-localized interop fix inside a behavior-preserving
refactor — **not** a gate-color decision. The design states plainly: the default
disposition **preserves a known metadata regression**; "behavior-preserving"
here means "regression-preserving" for this one target, and the honest caveat in
the Goal (a clean `check` proves *unchanged*, not *correct*) explicitly extends
to cover it. The seal must flag this (see below) so the green gate is never
mistaken for a validated/correct output.

Rationale for deferring. Restoring the dropped `attrs` moves the baseline (as
above); more importantly, absorbing it well depends on the §5 audit having
localized the cause — which is now an **unconditional audit obligation** via the
runtime diagnostic probe with the per-model checkpoint chain (risk-2 + ext1-4 +
ext2-3, §5), *decoupled* from the absorption decision. Fixing an un-diagnosed
interop drop inside a refactor risks a fragile manual `attrs` re-attach or an
hydromt-version-specific patch — a scientific/interop decision, not mechanical
cleanup, which is exactly the class R3's precedent forbids absorbing silently.

Disposition: **split to a dedicated task** (a `t2607xx`-style task in
`dev/TODO.md`, cross-referenced from the M2b section of `dev/followups.md`,
**and enumerated at the R4 seal** with its owner + activation condition),
mirroring R3's `t260719a` split of the CSDMS constants.

**Decoupled absorption call (risk-2 + ext1-4 + ext2-3).** Localization (§5
diagnostic probe) and absorption are now **two separate decisions**. The §5 probe
localizes the drop unconditionally to a workflow-2 transformation **or** an
isolated dependency op — that is a diagnostic finding with a concrete traced
result and zero baseline cost, and "we didn't localize it" is not an available
end state. *Given* a localization outcome, the absorption decision is argued
separately: R4 **may** elect to absorb the fix **only if** localization pins it
to a single, provably correct `attrs`-preserving change in workflow-2 code (e.g.
a per-model reduction dropping attrs under `keep_attrs=False`, or the
`coords="minimal"`/`preprocess` interaction in `open_mfdataset`,
`get_change_climate_proj_summary.py` lines 60–62, shown by the probe to strip
attrs where a workflow-2-side fix restores them) **and** the milestone
deliberately charters the baseline move with a `baseline_diffs.md` entry and a
targeted `record`. If the probe's outcome is instead the **dependency-reproducer**
end state (the loss is inside an isolated hydromt/xarray op with no workflow-2
code the milestone can correctly change), absorption is *not* attempted in R4 —
the fix belongs upstream or to a version-pin task. Default remains **out**
(ratified), with the probe obliged to localize either way.

**2. Parked per-rule `message:` progress directives (`dev/followups.md` lines
54–64) — RULING: OUT of R4; leave parked.**

Rationale. The followup itself marks this **[PARKED 2026-07-19]** and
cross-cutting ("apply across all three `Snakefile_*` as a consistent pattern"),
explicitly "deferred by choice, not a blocker." Absorbing it for workflow 2
alone would create the exact inconsistency the followup warns against — one
Snakefile with `message:` directives, two without — and R4 must not touch the
other two Snakefiles' content (iron rule). The R3 design added `log:`/
`benchmark:` **without** `message:`, so absorbing `message:` here would also
break inheritance fidelity (decision criteria §3: apply R3 patterns as-is unless
a workflow-2 reason to deviate — there is none). Disposition: **leave parked**;
the followup entry already names R4/R5 Snakefile work as a natural pickup point,
so no re-tag is needed — the parking note stands. R4 adds `log:` only, matching
R3.

**3. Workflow-2 `test_cli` xfail resolution (`dev/followups.md` lines 70–94) —
RULING: already resolved in R3; confirm no residual ratchet work lands in R4.**

Rationale. R3 already fixed the workflow-2 `MissingInputException` ratchet from
the *test* side by pre-staging a minimal valid `region.geojson` in a test-owned
`tmp_path` fixture and flipping the assertion to `returncode == 0` (verified in
`tests/test_cli.py`: `config_with_staged_region` fixture lines 38–59,
`test_snakefile_cli_climate_projections` lines 68–80, and the contract-pin
`test_climate_projections_declares_wf1_region_input` lines 83–92). Those tests
currently pass. Disposition: **no ratchet work in R4.** R4's *obligation* is
narrower and stated as a verification gate (below): when R4 renames workflow-2
labels (§6) or comments the `ruleorder:` (§3), the pin test asserting
`"staticgeoms/region.geojson"` still appears in the Snakefile
(`test_cli.py` line 92) must stay green — i.e. R4 must not rename that
cross-workflow **input path** (it is a §7 contract surface, grandfathered). If a
friendlier production error for "workflow 2 run before workflow 1" is wanted,
that is new UX scope, recorded as a followup, not ratchet work — R3's design
explicitly left "a friendlier production error" to R4 territory but it is a
non-goal here (intake non-goals: no new operational behavior). Confirmed out.

## Behavior-preservation stance and exact baseline consequence

**R4 is a behavior-preserving refactor. Default: zero manifest edits.** The
gating is **two-tier** (arch-3):

- **Per-commit gate:** dry-run (`--dry-run`) + `pytest tests/test_cli.py` after
  every Snakefile-touching or script-signature-touching commit. This is what
  actually runs at each commit boundary; it does **not** regenerate the manifest
  targets.
- **Milestone-level gate:** the full baseline comparison is run **once,
  end-to-end at the milestone**, not per commit — because it compares
  fingerprints of *regenerated* targets, which requires a full workflow run. The
  concrete, decidable procedure (which workflows to run and which targets to
  check) is specified in the Verification plan below (ext1-1/ext2-1). It is
  expected clean precisely because no hygiene commit changes a computational path
  or a fingerprinted byte. (The v1 phrasing "14/14 clean after every commit"
  overclaimed a per-commit invariant that would be either trivially true on stale
  files or infeasible; corrected in v2 and retained here.)

Discriminating self-check: walk the commit plan below — none of the hygiene
commits (contract doc, log/benchmark, ruleorder comment + AGENTS.md correction,
`tee_to_log` wraps, both import guards, label renames, unit tests, the
`check_baseline.py --workflow` filter) alters a computed value or a fingerprinted
byte, so none requires `record`.

**Exact consequence, stated precisely:** the 7 workflow-2 manifest targets
(`check_baseline.py` lines 60–66) stay byte-for-byte / stat-for-stat identical
across all hygiene commits. The one place a commit *could* move the baseline is
an **intentional change** elected under the finding-classification policy — of
which the only live candidate is the M2b attrs restoration (candidate #1), and
that is **out of scope by default (ratified)**. **If** the probe localizes it to
workflow-2 code *and* the milestone charters it in, exactly one target moves —
`annual_change_scalar_stats_summary.nc`, on its `attrs` field only — recorded in
`dev/milestones/r04/baseline_diffs.md` with a targeted `record`. No other target can move
under this design. (The row-V/row-P defects do **not** move the baseline in R4:
they are split, not fixed here — their fixes' baseline consequences land in the
owning task, not this milestone.)

**Gate-coverage honesty (risk-5).** The three PNG targets are fingerprinted
**size-only at ±10% tolerance** (`PNG_TOLERANCE_FRAC = 0.10`) — the **weakest
link in the gate**. A per-model or plotting-branch change that shifts a plotted
anomaly but keeps the summary `.nc` stats within 10 significant figures (e.g. a
member-ordering or plot-only effect on `gcm_timeseries.nc` or the anomaly plots)
could leave `check` green while the actual figures differ. R4 does not lean on
the merged-summary fingerprint as if it fully covers the plot outputs: the
guarantee for plot-affecting intermediate changes rests on **code-path
inspection**, not the gate. No manifest change needed — this is a scope-honesty
correction.

**`temp(...)` intermediates (repeated caveat).** The per-model intermediates are
**not** fingerprinted, so a clean `check` proves the merged summary + plots
(within the PNG tolerance above) are unchanged, not that every per-model change
factor is — R4's guarantee for those rests on not editing the computational
path.

## Verification plan

- **Per-commit dry-run:** after every Snakefile-touching commit,
  `snakemake all -c 1 -s Snakefile_climate_projections --configfile
  config/snake_config_model_test.yml --dry-run` builds a clean DAG. (The
  `ancient(region.geojson)` input means a *bare* dry-run against an empty
  project dir raises `MissingInputException` — correct behavior; the DAG
  validity is proven by `tests/test_cli.py`'s staged-region fixture, not a bare
  dry-run.)
- **`pytest tests/test_cli.py`** after every commit touching a Snakefile or a
  script signature (per `AGENTS.md`). The three workflow-2 tests
  (`test_snakefile_cli_climate_projections`,
  `test_climate_projections_declares_wf1_region_input`, and the shared
  `test_snakefile_cli_model_creation`) must stay green; the workflow-3
  `CyclicGraphException` ratchet (lines 111–120) must still report its exact
  known exception — a change there means a shared-helper regression.
- **New unit + audit-evidence tests** (§7, including matrix rows U/C1–C3/H/V/P/D)
  under `pixi run pytest tests/`; **rows U, C1–C3, H, D are expected to PASS;
  rows V and P (the fail-loud normative acceptance tests, ext3-1) are expected
  to FAIL against the current code (ext2-2) and their failures are the
  executable evidence for the two split defects.** The suite is wired so those
  two predicted failures are recorded as `xfail(strict=True)` with a reason
  string naming the split task — so the suite is green overall, the defect
  evidence is durable, and a landed fail-loud fix flips `xfail`→`xpass` and
  trips the strict gate (forcing the split task to remove the marker
  deliberately). The paired **characterization tests** (V-char/P-char, plain
  PASS today) pin the current silent-drop coordinates and are deleted by the
  owning split task's fix commit (§7). Final counts recorded in the roadmap
  seal.

### End-to-end baseline gate — the decidable procedure (arch-1 + ext1-1 + ext2-1)

**The `project_dir` (arch-1, blocking, resolved in v2).** The gate runs at
**`project_dir = examples/test_local`**, the exact directory the manifest was
recorded against (`PROJECT_DIR_DEFAULT`, `check_baseline.py` line 33; the
committed `config/snake_config_model_test.yml` line 12 sets
`project_dir: examples/test_local`; the recorded manifest's `project_dir` field
is `examples/test_local` — verified against `dev/baseline/manifest.json`). This is
mandatory, not preference, because **the yaml snapshot target is
`project_dir`-sensitive**: `copy_config_files.py` copies the `--configfile`
**verbatim, byte-for-byte** (lines 41–44), the config embeds
`project.project_dir`, and `check_baseline.py::fingerprint_yaml` (lines 146–152)
hashes the **full parsed YAML content** with no path-stripping. Running the gate
at any *other* `project_dir` (e.g. a differently-named `examples/test1`) changes
the snapshot content → its SHA256 → **RED on
`snake_config_climate_projections.yml`**, even though nothing workflow-2
computed. Because the uncommitted `examples/test1` edit has been **reverted**
(working tree clean per G1/status), the committed config already specifies
`examples/test_local`, so the run needs no edit.

**Which workflows to run and which targets to check (ext1-1 → ext2-1,
BLOCKING).** The v2 procedure — *wipe `examples/test_local`, run wf1 → wf2, then
`check_baseline.py check`* — could not produce a clean result because `TARGETS`
(lines 53–71) spans all three workflows, including three **workflow-3** targets
(`{exp_dir}/model_results/Qstats.csv`, `{exp_dir}/model_results/basin.csv`,
`{project_dir}/config/snake_config_climate_experiment.yml`, lines 67–70); after a
wf1 → wf2-only run those three are absent, and `cmd_check` (lines 260–262) turns
any recorded target **missing on disk** into a failure. v3 charted a `--workflow`
filter but scoped it **only on the source `TARGETS` list** — which reduces the
*current* side but leaves `cmd_check` loading all 14 recorded entries into
`rec_targets`. **ext2-1 correctly shows that leaves the gate incoherent** (see
the exact mechanics below). v4 respecifies the filter so **one selected target
universe is applied symmetrically to both the recorded and current sides**.

**The exact `check_baseline.py` mechanics ext2-1 turns on (verified against the
current source).** In `cmd_check` (lines 250–281):
- Line 255–256 loads the **full** recorded manifest into `rec_targets` (14
  entries), *unconditionally* — no filtering.
- Line 257 computes `current, missing = compute_manifest(project_dir)`, which
  walks the module-level `TARGETS`. Scoping only `TARGETS` (v3's fix) makes
  `current` and `missing` cover 11 targets, but **`rec_targets` is still 14**.
- Line 260–262 (`for p in missing: if p in rec_targets`) — with `TARGETS` scoped
  to 11, the 3 wf3 templates never enter `missing`, so they never fail here.
- Line 263–265 (`for path, rec in rec_targets.items(): if path not in current:
  continue`) — the 3 wf3 recorded entries are not in the 11-entry `current`, so
  they are **silently skipped**.
- Line 269 (`sorted(set(current) - set(rec_targets))`) — `current` (11) ⊆
  `rec_targets` (14), so the orphan set is empty (no false orphan — good).
- Line 273–274 (`print(f"OK - {len(rec_targets)} target(s) match manifest")`) —
  **reports `len(rec_targets)` = 14**, contradicting the promised "OK — 11
  target(s)". The gate claims 14 matched after comparing only 11.

So v3's source-only scoping produces a *passing but misleading* count and
silently skips the unselected recorded entries — exactly ext2-1's blocking
finding.

**v4's respecification — symmetric selected universe (ext2-1).** Add a
`--workflow {model_creation,climate_projections,climate_experiment}` option
(**repeatable**) to the **`check` subcommand only** (see the record decision
below). The implementation:

1. **Tag each `TARGETS` entry with its owning workflow.** Change `TARGETS` from
   `list[tuple[str, str]]` to carry a workflow tag per entry (e.g.
   `list[tuple[str, str, str]]` = `(workflow, kind, template)`, or a small
   dataclass). The three groups already sit under the existing
   `# Snakefile_*` comments (lines 54/59/67), so the tagging is mechanical and
   the group boundaries are already correct (4 / 7 / 3).
2. **Derive one `selected` target list** in `cmd_check`: if `--workflow` is
   given, `selected = [t for t in TARGETS if t.workflow in chosen]`; else
   `selected = TARGETS` (unchanged default = all 14, full backward
   compatibility).
3. **Apply `selected` symmetrically to BOTH sides before any missing / diff /
   orphan / count logic:**
   - `compute_manifest` takes `selected` and walks only those templates → the
     `current` and `missing` sets cover only the selected universe.
   - **`rec_targets` is filtered to the selected universe too:** build
     `rec_selected = {path: rec for path, rec in recorded["targets"].items() if
     path in selected_resolved_paths}` (where `selected_resolved_paths =
     {resolve(t.template, project_dir) for t in selected}`). Every subsequent use
     of `rec_targets` in `cmd_check` (lines 261, 263, 269, 274) uses
     `rec_selected` instead.
   - Result: the missing-check (260–262), the diff loop (263–268), the orphan
     check (269–270), and the reported count (274) **all operate on the same
     reduced 11-target universe.** The 3 wf3 recorded entries are excluded from
     `rec_selected`, so they are neither checked nor counted, and they cannot be
     falsely flagged as orphans (they are not in `current` either).
4. **The reported count is `len(rec_selected)` (= 11 for the R4 gate), not
   `len(rec_targets)`** — so "OK — 11 target(s) match manifest" is truthful.

**Scoped `record` is OMITTED entirely (ext2-1's dangerous case).** The reviewer
flags that a scoped `record` would write a partial manifest to the canonical
manifest path and truncate the authoritative baseline. v4's simplest safe answer:
**the `--workflow` option is added to `check` only, never to `record`.** `record`
continues to walk the full unfiltered `TARGETS` (all 14) and refuses to write an
incomplete manifest (its existing `missing`-guard, lines 234–238, is retained
unchanged). There is therefore **no code path by which a scoped run can truncate
the canonical manifest** — the truncation risk is eliminated by construction, not
by convention. (The only `record` R4 ever runs is the optional, chartered
intentional-change attrs commit, which is out-by-default; if it runs, it is a
full-manifest `record` after a full wf1→wf2→wf3 regeneration, or it uses a
targeted per-target update path that R4 does not build here — but since attrs is
out-by-default, R4 plans no `record` at all.)

**The gate R4 actually runs:** seed `examples/test_local` clean (wipe, run
workflow 1, then workflow 2), then
`python dev/scripts/check_baseline.py check --workflow model_creation
--workflow climate_projections` (default `--project-dir` is already
`examples/test_local`). This checks **11 targets: the 4 workflow-1 targets
(shared-helper / wf1 regression coverage) + the 7 workflow-2 targets**;
workflow 3 is excluded because it was not run and is out of R4's scope.
Expected: **OK — 11 target(s) match manifest.**

**How wf1 / shared-helper regressions are covered (ext1-1's "state
separately").** Two independent gates cover the shared helper: (1) the
per-commit `pytest tests/test_cli.py`, whose
`test_snakefile_cli_model_creation` and the workflow-3 `CyclicGraphException`
ratchet fail loudly on any `get_config`/`tee_to_log` regression; and (2) the
milestone gate's **`--workflow model_creation`** targets (the 4 wf1 manifest
targets), which fingerprint workflow-1's regenerated outputs and go red if the
shared-helper inheritance perturbed a workflow-1 result. R4 runs workflow 1 as
part of seeding (workflow 2 needs its `region.geojson`), so those 4 targets are
regenerated and genuinely checked, not skipped.

**Chartered pytest coverage for the `--workflow` filter (ext2-1).** The filter
is a new `check_baseline.py` capability that **does not exist today**
(`_add_common` lines 284–288 add only `--project-dir`/`--manifest`; `check` adds
only `--tolerance`, line 299). It is chartered as an explicit R4 deliverable
(commit 2b) with its own test module in `tests/` (e.g.
`tests/test_check_baseline_scope.py`) that, against a small **synthetic fixture
manifest** (a 3-entry manifest tagged across two workflows + on-disk matching
files under a `tmp_path` project dir), asserts:
1. **Reported count is the selected count, not the full manifest count.**
   `check --workflow climate_projections` on a full 14-tagged synthetic manifest
   reports "OK — N target(s)" where **N = number of selected (e.g. 7), not 14** —
   the exact bug ext2-1 names.
2. **A selected target missing on disk FAILS.** Delete one selected file →
   `check --workflow climate_projections` returns non-zero and names that target.
3. **An unselected recorded target is ignored.** With the 3 wf3 recorded entries
   present in the manifest but their files absent on disk,
   `check --workflow model_creation --workflow climate_projections` returns
   **0** (passes) — the unselected wf3 entries neither fail nor inflate the count.
4. **Scoped `record` cannot truncate the canonical manifest.** Assert `record`
   has **no `--workflow` argument** (argparse rejects it), so there is no code
   path to a partial `record`; and that an unscoped `record` still writes all
   tagged targets. (This is a structural assertion — the safety is that the flag
   does not exist on `record`.)
5. **Unscoped `check` is unchanged** — `check` with no `--workflow` still spans
   all tagged targets and reports the full count, preserving backward
   compatibility with the recorded manifest.

- **Clean-baseline discipline:** no hygiene commit runs `record`. The only
  `record` in R4, if any, is the chartered intentional-change commit (attrs, iff
  localized to workflow-2 code and elected in), and by the decision above that is
  out-by-default, so R4 plans no `record`.

## Commit plan (roadmap `r04:` prefix; one logical change per commit; tag `r04-projections`)

1. `r04: add dev/reference/workflows/climate_projections.md contract doc`
   (current behavior; owned keys, save_grids gating, plausibility-overlay note,
   the documented M2b attrs regression flag, the chain-audit findings table with
   its `Evidence`/`Outcome` columns if it fits else a pointer to
   `dev/milestones/r04/chain-audit.md`).
2. `r04: resolve the ruleorder in Snakefile_climate_projections + correct AGENTS.md`
   (§3; reproduce-or-refute on the two named config shapes M1/M2, then the
   **determined** action — always retain-and-comment; commit the reproduce-case
   or refute-case Snakefile comment + AGENTS.md text per the pre-written texts;
   no removal, no `wildcard_constraints` rewrite).
2b. `r04: add --workflow scope filter to dev/scripts/check_baseline.py`
   (ext1-1/ext2-1; repeatable `--workflow` on **`check` only**, tagging the
   source `TARGETS` list per workflow and applying the selected universe
   **symmetrically to both `rec_targets` and `current`** so the count and every
   check operate on the reduced set; `record` unchanged and unscoped so it cannot
   truncate the manifest; unscoped `check` default unchanged at 14 targets; new
   `tests/test_check_baseline_scope.py` per the five assertions above; touches no
   manifest, no `record`).
3. `r04: guard get_stats_climate_proj.py module body for importability`
   (output-neutral nested `if __name__/if "snakemake" in globals()` guard
   enabling §7 tests; dry-run + test_cli gate).
4. `r04: guard get_change_climate_proj.py module body for importability`
   (repo-1: output-neutral nested guard; **relocate `_to_str_tuple` above the
   guarded `snakemake.*` exec block** — the two `get_change_*` functions are
   already above it; dry-run + test_cli gate).
5. `r04: add log and benchmark directives + tee_to_log to monthly_stats rules`
   (`monthly_stats_hist`, `monthly_stats_fut` → `get_stats_climate_proj.py`,
   already guarded by commit 3; wildcard-keyed log paths).
6. `r04: add log and benchmark directives + tee_to_log to change/merge/plot rules`
   (`monthly_change` → `get_change_climate_proj.py` already guarded by commit 4;
   `monthly_change_scalar_merge`, `plot_climate_proj_timeseries`).
7. `r04: rename deprecated _fid path labels in workflow-2 rules and scripts`
   (§6; paired script reads updated same commit; §7 surfaces untouched).
8. `r04: document get_stats_climate_proj units + narrow error handling if neutral`
   (§4; error-behavior change only if provably output-neutral on the seed).
9. `r04: add unit + audit-evidence + integration tests for workflow-2 python helpers`
   (§7; unit tests for the importable functions + the audit-evidence matrix rows
   U/C1–C3/H/D as PASS + rows V/P as strict-`xfail` fail-loud normative tests
   with their paired V-char/P-char characterization tests (ext3-1) + the
   dummy-skip integration test D).
10. `r04: seal milestone — dev/roadmap.md R4 section + durable refs`
    (+ tag `r04-projections`; **the roadmap R4 section enumerates any
    defect-class findings the audit deferred — including the row-V variable-drop
    and row-P member-drop defects — each with owner + activation condition, and
    flags the encoded M2b attrs regression** — distinguishing "audited, clean"
    from "audited, defects deferred", risk-1/risk-3).

**Commit ordering rationale (repo-5).** The two import guards (commits 3, 4)
land **before** the `tee_to_log` wraps (commits 5, 6) so each script's
module body is restructured once (guard first) and then wrapped on the
already-guarded body — no region of `get_stats_climate_proj.py` or
`get_change_climate_proj.py` is re-touched across two commits. Commit 2b (the
`check_baseline.py` scope filter) lands early so the milestone-gate procedure is
runnable throughout; it is independent of the workflow-2 script commits.

An **intentional-change** commit for the M2b attrs (candidate #1) is added
**only if** the probe localizes it to workflow-2 code *and* the milestone
charters it in — as a dedicated commit paired with `dev/milestones/r04/baseline_diffs.md`
and a targeted `record`, inserted before the seal. Default plan carries no such
commit and writes no `baseline_diffs.md` (nothing moves).

**Scope statement (corrected for Group B + ext1-1/ext2-1).** R4 touches
workflow-2 files (`Snakefile_climate_projections`, the four `src/` scripts),
`tests/`, `dev/`, **one repo-root instruction file — `AGENTS.md` (lines 95–96
only), the ruleorder-invariant correction (§3)**, **and one dev-script —
`dev/scripts/check_baseline.py` (the `--workflow` scope filter, ext1-1/ext2-1)**.
The v1 "Most cross-cutting: none" and "touches only workflow-2 files plus tests/
and dev/" claims were made false by the Group-B fix and the ext1-1 dev-script
edit, and are corrected here: the cross-cutting edits are the `AGENTS.md`
ruleorder correction (an instruction surface) and the `check_baseline.py` scope
filter (a dev tooling surface), both deliberate. Riskiest commits: **5/6**
(`tee_to_log` stream redirection inside `script:` processes — R3's identified
sharp edge, though the manager is already tested) and the calendar audit finding
if it promotes to a task.

## Alternatives considered

- **ext2-1: keep v3's source-`TARGETS`-only scoping.** v3's posture. *Rejected
  (this round):* it reduces only the current side; `cmd_check` still loads all 14
  recorded entries into `rec_targets`, silently skips the 3 unselected recorded
  entries (line 263–265), and reports `len(rec_targets)` = 14 while comparing 11
  — a passing but misleading gate. v4 filters `rec_targets` to the same selected
  universe so count and all checks agree on 11.
- **ext2-1: add `--workflow` to `record` too (a scoped `record`).** Symmetric
  with `check`, and convenient for regenerating one workflow's slice. *Rejected:*
  a scoped `record` writes a partial manifest to the canonical path and truncates
  the authoritative 14-target baseline that later milestones depend on. v4 omits
  scoped `record` entirely (the flag exists on `check` only), eliminating the
  truncation path by construction. A separate-manifest-path variant
  (`record --workflow X --manifest other.json`) was considered and set aside as
  unneeded scope — R4 needs only scoped `check`.
- **ext1-1: run wf1→wf2→wf3 before an unscoped 14-target check.** The
  zero-code path — satisfy the full manifest by regenerating all three workflows'
  targets. *Rejected as primary:* it drags workflow 3's heavy stress-test
  ensemble (weathergenr × Wflow × `RLZ_NUM`×`ST_NUM`) into a workflow-2 milestone
  gate, contradicting the milestone boundary and making the gate slow and
  scope-inflated. The `--workflow` scope filter (adopted) checks exactly the 4
  wf1 + 7 wf2 targets that R4 actually regenerates, is faster, and matches the
  milestone's territory. Kept as a fallback only if the scope filter proved
  infeasible (it is a small change).
- **ext1-1: filter only `rec_targets` inside `cmd_check`.** Simpler than tagging
  the source `TARGETS`. *Rejected:* `cmd_check` line 269
  (`set(current) - set(rec_targets)`) would then flag every on-disk wf1/wf2
  target as "present but not in manifest," reintroducing the failure from the
  other side. The filter must apply the **same selected universe** to both the
  source `TARGETS` (so `compute_manifest`, `missing`, and the orphan check share
  it) **and** `rec_targets` (so the diff loop and count share it) — the symmetric
  fix v4 specifies.
- **ext2-2: leave rows V and P as "records observed behavior".** v3's posture.
  *Rejected:* an implementer can encode whatever the code does as the assertion
  and pass, sealing the milestone without ever deciding whether silent
  variable/member loss is acceptable. v4 states the **norm** (both variables
  survive; no member silently dropped), predicts the current code **FAILS** it,
  and pre-classifies both as split defects — the norm is decided in the design,
  not by the code.
- **ext2-2: elect to FIX the row-V / row-P defects in R4.** Restore both
  variables / align members explicitly so the summary is complete. *Rejected as
  default:* the fix (fail-loud or `join="outer"`/explicit alignment) is
  behavior-changing on the missing-data path and could change which (model,
  scenario, member) combinations contribute to the summary — a baseline move
  inside a behavior-preserving milestone. Handled by split-don't-absorb; the fix
  belongs to the owning task with its own baseline_diffs entry. (Available as an
  intentional-change path only if the milestone deliberately charters the move,
  which R4 does not.)
- **ext2-3: keep the probe's first checkpoint after the whole per-model
  write.** v3's posture. *Rejected:* an empty-attrs reading there localizes the
  loss only to the entire per-model region, yet the design demands a single
  operation or an isolated reproducer, and the synthetic alternative exercised
  only the merge path. v4 threads checkpoints P0→P6 through
  `get_change_annual_clim_proj` (input → window trim → resample-reduce →
  stat/arithmetic → merge/assign_coords → pre-`to_netcdf` → reopen), defines
  localization as the first checkpoint where attrs vanish, and requires an
  isolated reproducer for that transformation; the synthetic path now localizes a
  per-model loss without the real merge.
- **Tighten the `ruleorder:` via a `wildcard_constraints` rewrite (§3).** Let
  inference resolve hist/fut/change ordering so the directive can be dropped
  cleanly. *Rejected for R4:* it is a DAG-semantics change (which paths a rule
  can match), the class R3 protected as territory; it risks breaking the
  `temp(...)` intermediate wiring; and it is unnecessary given the empirical
  finding that the directive already constrains nothing on the test configs.
  Weighed and set aside as possible R6 territory if a cross-workflow cleanup
  revisits it.
- **ext1-3: leave the ruleorder decision as a keep-or-remove fork.** v2's
  posture. *Rejected:* the finding correctly shows two implementers could make
  opposite DAG-semantics calls from the same evidence. v3 fixes R4's action as
  **always retain-and-comment**, defers removal to a task gated on regression
  tests, and pre-writes the exact Snakefile/AGENTS.md text for both the reproduce
  and refute outcomes — the evidence selects only the comment wording, never the
  action. (Carried forward unchanged in v4.)
- **Blindly comment the `ruleorder:` with the `AGENTS.md`-stated reason, and
  leave `AGENTS.md` alone (§3).** The fastest path to the roadmap deliverable.
  *Rejected:* the scratchpad dry-run (independently reproduced by the panel)
  refuted the stated ambiguity on both test configs, so shipping the `AGENTS.md`
  reason verbatim would enshrine an unverified (likely stale) claim in a
  permanent code comment, *and* leaving `AGENTS.md` uncorrected would leave the
  canonical instruction file asserting a claim the milestone proved false.
- **Leave `AGENTS.md` out of scope and only track its correction via a
  followup (arch-2 option b).** Keeps R4 strictly workflow-2-scoped.
  *Rejected:* the correction is a single line, the milestone owns the evidence
  that refutes it, and orphaning a known-false invariant to a followup risks it
  propagating into R5/R6 before the followup is picked up. A deliberate,
  explicitly-scoped one-line edit is more honest than a tracked-but-deferred
  falsehood. (The followup path remains the fallback if the reproduce step
  unexpectedly *confirms* the invariant — in which case AGENTS.md is right and
  nothing changes.)
- **ext1-2: keep the calendar/variable/member checks inspection-only and
  conditional.** v2's posture ("audit whether … if the audit flags one").
  *Rejected:* a reviewer could complete the findings table without ever
  exercising 360-day / noleap / Gregorian boundaries, non-January hydrological
  years, partial members, or asymmetric variables — sealing "audited, no defects"
  while the silent month-/variable-dropping paths stay untested. v3 makes the §7
  audit-evidence matrix (rows C1–C3/H/V/P/D) mandatory with hand-computed
  expected outcomes, and requires each findings-table row to cite a passing test
  or a traced result; v4 adds normative pass/fail criteria to V and P.
  Behavior-preservation is intact: evidence characterizes, the defect class still
  splits.
- **ext1-4: treat M2b localization as a zero-cost source-reading finding.** v2's
  framing. *Rejected:* attrs propagation is runtime- and operation-dependent, so
  source inspection alone may misattribute the loss. v3 charters a runtime
  diagnostic probe; v4 threads it through per-model generation (ext2-3) so the
  first attr-dropping transformation is pinnable and an isolated reproducer is
  required for it.
- **Absorb the M2b attrs fix in-scope by default (candidate #1).** Cleaner for
  the end user (CF metadata restored now). *Rejected as default (user-ratified):*
  moves the baseline, and absorbing it well requires localization (now the
  unconditional §5 probe obligation, decoupled from the absorption call); fixing
  an un-diagnosed interop drop inside a refactor is exactly the mechanical
  scientific edit R3's precedent forbids. Kept as a *separately-argued*
  conditional in-scope path gated on the probe localizing it to workflow-2 code.
- **Fix the bare `except:` blocks in `get_stats_climate_proj.py` (§4.1)
  unconditionally.** Correct in principle. *Rejected as unconditional:* narrowing
  the exception is behavior-changing on the error path and could move which
  models yield empty datasets — a baseline move. Handled via the classification
  policy: narrow only if provably output-neutral on the seed, else split with an
  owner.
- **Skip the import-guard refactors (§7) and test via `runpy` + heavy mocking.**
  Avoids touching the scripts. *Rejected as primary:* `runpy` harnesses that
  drive module-level `snakemake.*` code are brittle and were part of the M02c
  mocking pain the intake calls out; two small output-neutral nested guards (the
  pattern two sibling scripts already use) make the functions cleanly importable
  and are more honest. Kept as fallback only if a guard proves to change
  behavior.
- **Charter a `filter_nonempty` extract to unit-test the dummy-skip logic
  (§7).** More consistent with the import-guard pattern. *Rejected as primary
  (parsimony):* the dummy-skip logic is a local list filter, not a callable, and
  the behavior is genuinely integration-shaped (empty-vs-nonempty netCDF pair);
  an integration-style test (matrix row D) covers it without adding a refactor
  commit. Kept as the documented fallback if the integration test proves too
  heavy to mock.
- **Defer the whole chain audit to R5/R6 and ship R4 as hygiene-only.** Smallest
  change set. *Rejected:* the audit is an explicit roadmap R4 deliverable, and
  the calendar/missing-data questions are workflow-2-specific — no later
  milestone owns them. Deferring would leave the scientific core unreviewed.
- **Do the audit *and* fix everything it finds, in R4.** Maximally thorough.
  *Rejected:* violates behavior-preservation and the "split, don't absorb"
  precedent; conflates a reading+testing deliverable with a physics-change
  milestone. The classification policy (with the risk-3 forcing function and the
  ext1-2/ext2-2 evidence matrix feeding it) is the deliberate middle path.

## Risks and open questions

- **The `ruleorder:` may be stale insurance, not load-bearing (open question,
  but the R4 action is now decided — ext1-3).** Verified during this design
  (scratchpad dry-run, current pinned Snakemake; independently reproduced by the
  panel): commenting out line 53 builds a clean 19-job DAG with exit 0 on both
  the tests fixture config (M1) and a reduced simple-name config (M2). Which of
  {stale-from-a-pre-migration-version / config-shape-triggered-only /
  inaccurate-`AGENTS.md`-claim} holds is **unresolved in this doc** and is
  documented (not gated) at implementation. **What is *no longer* open is R4's
  action:** R4 retains-and-comments in every case, deferring removal to a
  regression-test-gated task — so the open mechanism question cannot produce
  divergent DAG-semantics decisions (ext1-3).
- **Calendar handling is the riskiest audit item — and now tested (ext1-2).**
  `get_change_climate_proj.py` mixes `pd.to_datetime` boundary construction with
  `cftime`-indexed CMIP6 data and does **not** convert to a datetime index
  (unlike `plot_proj_timeseries.py` lines 36–43); the once-considered conversion
  is commented out (lines 224–232). The §7 matrix rows C1–C3/H now *exercise*
  this on 360-day, noleap, and Gregorian indices plus a non-January boundary,
  with hand-computed expected windows. If a matrix row reveals a real
  misalignment, it is a **defect** that moves an output → split to a task with a
  named owner + activation condition, enumerated at the seal (risk-3), not fixed
  in R4.
- **Rows V and P are predicted defects, not open questions about behavior
  (ext2-2 + ext3-1).** The design has decided the norm **uniquely** — fail-loud
  via the §7 `ValueError` exception contract (arbitrated 2026-07-20) —
  predicted the current `intersection()` / default `arithmetic_join="inner"`
  code violates it (nothing raises), and pre-classified both as split defects.
  Both normative rows fail **deterministically** on their hardcoded asymmetric
  synthetic inputs — the failure is certain (hence the strict-`xfail` wiring is
  safe, and a landed fail-loud fix flips them `xpass`, tripping the gate). The
  one residual open item is the **exact xarray join mechanism on the `member`
  dim**: v4 asserts inner-join drop from the default `arithmetic_join="inner"`;
  that mechanism is pinned by the paired **characterization test**, so if the
  true mechanism differs, the characterization assertion is corrected to the
  observed coordinates while the normative row is unaffected (its criterion is
  the raise). Whether the defect reaches production outputs (e.g. an upstream
  guarantee that real hist/fut always share variables and members, making the
  intersection a no-op on real data) is a **severity/priority** input for the
  owning split task, assessed with evidence at implementation — **not** a path by
  which the synthetic rows pass.
- **Both `get_stats_climate_proj.py` and `get_change_climate_proj.py` are not
  import-clean.** Module-level `snakemake.*` execution
  (`get_stats_climate_proj.py` lines 28–74; `get_change_climate_proj.py` lines
  178–187) blocks direct unit-testing of their functions, and in the change
  script `_to_str_tuple` sits *after* the exec block (line 193 vs 178) so the
  guard must relocate it (repo-1). The two guard refactors (§7, commits 3–4) are
  asserted output-neutral but **unverified until run** — recorded as an open
  question: confirm each guard changes no computed value before relying on it.
- **Baseline `project_dir` for the end-to-end gate is DECIDED (arch-1), not
  open.** The gate runs at `examples/test_local` (the recorded manifest dir; the
  committed config already sets it), because the yaml snapshot target embeds
  `project_dir` and is hashed verbatim — running elsewhere would turn the
  snapshot target red for a non-computational reason. The only implementation
  step is to seed `examples/test_local` clean and run wf1 → wf2 before the
  scoped `check`.
- **The milestone gate needs the scoped-check capability that does not exist
  today, and the scoping must be symmetric (ext1-1 → ext2-1, DECIDED).**
  `check_baseline.py` has no target-filtering option (only
  `--project-dir`/`--manifest`/`--tolerance`), so an unscoped `check` after a
  wf1→wf2-only run would FAIL on the three missing workflow-3 targets. R4 charters
  the `--workflow` filter (commit 2b) applying **one selected universe to both
  `rec_targets` and `current`** (v3's source-only scoping was incoherent — it
  reported 14 while comparing 11 and silently skipped the unselected recorded
  entries), omits scoped `record` (so the canonical manifest cannot be
  truncated), and runs
  `check --workflow model_creation --workflow climate_projections` for the 11
  relevant targets, with `tests/test_check_baseline_scope.py` asserting the
  count, the selected-missing failure, the unselected-ignored pass, and the
  no-scoped-`record` structural guarantee. This is a decided dev-script
  deliverable, not an assumed flag.
- **M2b attrs localization is an unconditional audit obligation via a runtime
  probe with per-model checkpoints (risk-2 + ext1-4 + ext2-3), not an open
  gate.** Whether the drop is in a per-model transformation inside
  `get_change_annual_clim_proj` (a reduction/arithmetic dropping attrs under
  `keep_attrs=False`), the `open_mfdataset(coords="minimal", preprocess=...)` in
  `get_change_climate_proj_summary.py` (lines 60–62), or the merge `to_netcdf`
  must be pinned by the §5 probe (the per-operation per-model checkpoint chain
  P0–P2/P3a/P3b/P4a/P4b/P5–P6 + M1/M2 merge checkpoints, run as the stepwise
  `dev/scripts/probe_attrs_chain.py` diagnostic with no production-code
  instrumentation, handling the `temp(...)`-deletion trap), with an isolated
  reproducer required for the first attr-dropping operation, regardless of
  the absorption decision. The out-by-default ruling (#1) is decoupled from
  localization: the probe localizes; the milestone separately decides to defer
  (default, ratified) or absorb.
- **`gcm_timeseries.nc` label/extension mismatch** (`plot_climate_proj_timeseries`
  declares `timeseries_csv = ...gcm_timeseries.nc`, Snakefile line 154) is a
  known cosmetic wart — classified **noise** (documented, not renamed: the output
  filename is a §7 contract surface).
- **`tee_to_log` in `script:` processes** is R3's identified sharp edge (stream
  redirection, empty-log-on-swallowed-error). The manager is already tested
  (R3, `src/snake_utils.py` lines 76–110), so the residual risk is only in
  wiring `snakemake.log[0]` correctly per script — reviewed hardest in
  commits 5/6.

## Tag

`r04-projections`. Branch `milestone/r04-projections` off `main` (which carries
the sealed R3 state).

## References

- `dev/roadmap.md` — R4 section (lines 355–390), cross-cutting principles
  (baseline discipline, territory rule), commit strategy (`r04:` prefix).
- `dev/milestones/r03/model-builder-design.md` — the sibling design R4 mirrors
  (log/benchmark convention §6, `tee_to_log` contract, split-don't-absorb
  precedent §5, behavior-preservation posture).
- `dev/reference/workflows/model_creation.md` — the R3-authored contract-doc precedent
  R4's opening act mirrors.
- `dev/milestones/r01/modularity-contracts-design.md` §4 — contract-doc format.
- `dev/reference/naming.md` — §3/§5 path-suffix rules (R4 is the owning
  milestone for workflow-2 identifiers); §7 grandfathering.
- `dev/followups.md` — M2b attrs item (lines 316–322), parked `message:`
  directives (lines 54–64), the resolved workflow-2 `test_cli` ratchet (lines
  70–94), retired CMIP6 throughput followup (lines 332–337).
- `dev/milestones/phase-1/m02b/baseline_diffs.md` — records the M2b attrs regression
  (correct CF metadata → `{}`), the reference for candidate #1's
  "documented regression" framing.
- `dev/scripts/check_baseline.py` — `TARGETS` (lines 53–71; workflow-2 subset
  60–66, workflow-3 subset 67–70): exactly what the manifest fingerprints;
  `compute_manifest` (163–172) / `cmd_check` (250–281, esp. the full-manifest
  load into `rec_targets` at 255–256, the missing-target failure 260–262, the
  diff loop 263–268, the orphan check 269, and the count print 274) — the
  mechanics ext1-1/ext2-1 turn on; `cmd_record` (232–247, esp. the missing-guard
  234–238 that keeps `record` from writing an incomplete manifest);
  `fingerprint_yaml` (146–152) hashes full parsed YAML (project_dir-sensitive,
  arch-1); attrs at line 116; `PNG_TOLERANCE_FRAC` line 40 (risk-5);
  `PROJECT_DIR_DEFAULT` line 33; `_add_common` (284–288) shows only
  `--project-dir`/`--manifest` exist today (ext1-1 charter); and the honest
  caveat about `temp(...)` intermediates.
- `src/copy_config_files.py` — copies the `--configfile` verbatim (lines 41–44),
  which is why the yaml snapshot is project_dir-sensitive (arch-1).
- `src/get_change_climate_proj.py` — `intersection` (18–19), the change formula
  `get_change_annual_clim_proj` (72–171, esp. the variable loop over
  `intersection()` at line 98, the resample/reduce at 119/128/137/146, the
  arithmetic at 161–163, the merge at 170, the write at 278), and the
  module-level `snakemake.*` exec block (178–199) with `_to_str_tuple` at 193 —
  the code rows V/P/C/H, the probe checkpoints, and the guard obligation are
  grounded in.
- `AGENTS.md` — lines 95–96 carry the ruleorder "load-bearing" claim R4 corrects
  in the refute case (§3, Group B).
- `Snakefile_climate_projections`, `src/get_stats_climate_proj.py`,
  `src/get_change_climate_proj_summary.py`,
  `src/plot_proj_timeseries.py`, `src/snake_utils.py`, `tests/test_cli.py` — the
  code every claim above is grounded in.

## Revision log

- **v5 (2026-07-20).** Arbitration revision — applies the two fixes mandated
  by the user at the round-cap arbitration (external rounds capped at 3;
  surviving majors ext3-1 and ext3-2 both ruled ACCEPTED, fix-in-v5; rulings
  logged in the run's status.md). *ext3-1:* rows V/P norm pinned **uniquely to
  fail-loud** — `ValueError` exception contract (message names the asymmetric
  dimension, offending element(s), and both sides' sets); normative acceptance
  tests (assert the raise, strict-`xfail` today, marker removed by the owning
  split task's fix commit via the xpass tripwire) separated from
  characterization tests (plain PASS pinning today's silent-drop coordinates,
  deleted by the same fix commit); explicit reconciliation demoted from
  permitted norm to a norm-revision path owned by the split task. *ext3-2:*
  probe checkpoints P3/P4 split per-operation (P3a statistic reduction, P3b
  change arithmetic, P4a coordinate ops, P4b to_dataset+merge) so the first
  empty-attrs reading pins one operation; probe mechanism specified as a
  standalone stepwise diagnostic script (`dev/scripts/probe_attrs_chain.py`)
  with an intact-function cross-check and **no production-code
  instrumentation** by default — any temporary instrumentation on the
  `--notemp` corroboration path must be removed and verified by a clean
  `git status`/`git diff` on `src/` before the milestone gate. Authored by the
  driver session under explicit user authorization (delegated-author transport
  failures), deviating from the loop's driver-never-writes rule; deviation
  logged in observations.md.
- **v4 (2026-07-19).** Revision r3 — resolves external review round 2's three
  findings (ext2-1 blocking; ext2-2, ext2-3 major), all re-raises of
  incompletely-fixed round-1 findings; all three accepted. Carries forward every
  prior accepted resolution (internal 15 + external-r1 4) unchanged in substance
  (reviewer holds regression duty).
  - *ext2-1 (blocking).* v3's `--workflow` filter scoped only the source
    `TARGETS` list, so `cmd_check` still loaded all 14 recorded entries into
    `rec_targets`, silently skipped the 3 unselected recorded entries
    (line 263–265), and reported `len(rec_targets)` = 14 while comparing 11 — and
    a scoped `record` would truncate the canonical manifest. Fixed by
    respecifying the filter to build **one selected target universe applied
    symmetrically to both `rec_targets` and `current`** before missing/diff/orphan
    /count logic (count = `len(rec_selected)` = 11), and by **omitting scoped
    `record` entirely** (flag on `check` only; `record` stays full-manifest, so
    truncation is impossible by construction). Chartered
    `tests/test_check_baseline_scope.py` with five assertions (count = selected,
    selected-missing fails, unselected ignored, no scoped `record`, unscoped
    unchanged). See "End-to-end baseline gate — the decidable procedure" and
    commit 2b.
  - *ext2-2 (major).* Matrix rows V and P asserted only "records observed
    behavior," letting current behavior satisfy the test. Fixed by giving both
    rows **normative expected outcomes with exact surviving coordinates**: row V's
    norm is that the result carries BOTH `precip` and `temp` (current
    `intersection()` yields only `precip` → **row V FAILS → variable-drop
    defect, split**); row P's norm is no silent member drop (current default
    `arithmetic_join="inner"` drops the unshared member → **row P FAILS →
    partial-member defect, split**). Both pre-classified as defects in the
    finding-classification policy; wired as strict-`xfail` so the evidence is
    durable and an accidental fix trips the gate. See §5, §7.
  - *ext2-3 (major).* The probe's first checkpoint was already after the whole
    per-model computation and write, localizing an empty-attrs result only to a
    large region while the design demanded a single operation / isolated
    reproducer. Fixed by adding **per-model checkpoints P0→P6 inside
    `get_change_annual_clim_proj`** (known-attrs input → window trim → resample
    reduce → stat/arithmetic → merge/assign_coords → pre-`to_netcdf` → reopen),
    defining localization as the first checkpoint where attrs vanish, requiring an
    isolated reproducer for that transformation, and making the synthetic path
    sufficient for per-model localization. Merge checkpoints M1/M2 and the
    dependency-reproducer end state retained for post-generation losses. See §5
    "M2b attrs diagnostic probe".
- **v3 (2026-07-19).** Revision r2 — resolves the external cross-vendor review's
  four findings (ext1-1 blocking; ext1-2, ext1-3, ext1-4 major); all four
  accepted. Carries forward every v2 resolution of the internal panel's 15 IDs
  (risk-1..6, arch-1..4, repo-1..5) unchanged in substance.
  - *ext1-1 (blocking).* Verified in `check_baseline.py` that `TARGETS` spans all
    three workflows (wf3 at lines 67–70) and `cmd_check` (260–262) fails on any
    recorded target missing on disk — so the v2 "wipe, run wf1→wf2, then check"
    procedure could not yield a clean result (3 wf3 targets missing). Fixed by
    chartering a **`--workflow` scope filter** on `check_baseline.py` (new
    commit 2b) that scopes the **source `TARGETS`** list; the milestone gate now
    runs `check --workflow model_creation --workflow climate_projections`
    (11 targets: 4 wf1 + 7 wf2), wf3 excluded. wf1/shared-helper coverage stated
    separately (per-commit `test_cli` + the 4 wf1 manifest targets). Charter notes
    that the flag does not exist today. (Superseded in v4 by the symmetric-scoping
    respecification — ext2-1.)
  - *ext1-2 (major).* The chain audit's calendar/variable/member checks were
    conditional and inspection-only. Added a mandatory **audit-evidence matrix**
    (§7, rows U/C1–C3/H/V/P/D) with hand-computed expected outcomes covering
    360-day/noleap/Gregorian calendars, a non-January hydrological-year boundary,
    partial members, asymmetric hist/future variables, and dummy datasets; each
    §5 findings-table row must cite a passing test or a traced result. Reconciled
    with split-don't-absorb: evidence characterizes, it does not authorize an
    inline fix (a failing row still splits). (Rows V/P sharpened in v4 — ext2-2.)
  - *ext1-3 (major).* The reproduce-or-refute gate was not operationally
    decidable (undefined "any supported config"; keep-or-remove fork). Fixed by
    naming the **two-shape config matrix** (M1 fixture / M2 reduced), fixing R4's
    action to **always retain-and-comment** (removal deferred to a
    regression-test-gated task), and pre-writing the **exact Snakefile comment and
    AGENTS.md text** for both the reproduce and refute outcomes — evidence selects
    only the wording, never the action. See §3.
  - *ext1-4 (major).* M2b localization was framed as a zero-cost reading finding
    with no diagnostic procedure. Added a **runtime diagnostic probe** recording
    `.attrs` at three checkpoints (post per-model gen / post
    `open_mfdataset`+preprocess / post merge-write+reopen); localization = first
    checkpoint where attrs vanish; handled the `temp(...)`-deletion trap
    (`--notemp` or a synthetic pair); admitted a **dependency-reproducer** end
    state. (Extended in v4 with per-model checkpoints — ext2-3.)
- **v2 (2026-07-19).** Revision r1 — resolves the internal panel's 15 finding
  IDs (risk-1..6, arch-1..4, repo-1..5; 1 blocking / 8 major / 6 minor); all 15
  accepted. Structural changes by group:
  - *Group A (arch-1, blocking).* End-to-end baseline gate now runs at the
    recorded dir `examples/test_local` (mandatory, not preference) — the yaml
    snapshot target embeds `project_dir` and is hashed verbatim
    (`copy_config_files.py` 41–44 → `fingerprint_yaml` 146–152), so any other dir
    turns it red for a non-computational reason. The false "path-independent for
    … yaml" claim is deleted; the "expected to hold" hedge is removed.
  - *Group B (risk-4 / arch-2 / repo-3 — one defect, one fix).* The ruleorder
    commit (commit 2) now also corrects `AGENTS.md` lines 95–96 to the verified
    status; the "Most cross-cutting: none" and "only workflow-2 files" scope
    claims are corrected to name the deliberate one-line `AGENTS.md` edit.
  - *Group C (risk-1, risk-2).* Candidate #1 reframed: the default preserves a
    **documented regression**, not a neutral baseline; deferral is a
    scientific-cost decision, ratified out-by-default, with the seal flagging the
    encoded regression. Localization is now an **unconditional §5 audit
    obligation**, decoupled from the (separately-argued) absorption decision.
  - *Group D (risk-3).* The classification policy gains a forcing function: a
    split defect must name an owner + activation condition and be enumerated in
    the roadmap R4 section at the seal, which distinguishes "audited, clean" from
    "audited, defects deferred".
  - *Group E (repo-1, repo-2).* A **second** import-guard commit is chartered for
    `get_change_climate_proj.py` (with `_to_str_tuple` relocated above the guard);
    the nonexistent `list_files_not_empty` unit is dropped for an integration-style
    dummy-skip test. Commit plan renumbered (10 commits).
  - *Minors.* PNG ±10% weakest-link stated (risk-5); Goal cites lines 60–66 as
    the workflow-2 subset of 53–71 (risk-6); two-tier per-commit vs milestone
    gating clarified (arch-3); `copy_config` exclusion criterion restated as
    "no computation to profile," acknowledging its gated output (arch-4); exact
    nested guard shape specified (repo-4); guards sequenced before wraps so no
    file region is re-touched (repo-5).
- **v1 (2026-07-19).** Initial draft for design-review-loop. Rules on the three
  intake candidate absorptions (attrs: out-by-default w/ conditional path;
  `message:`: leave parked; `test_cli` ratchet: resolved in R3, confirm no
  residual). Pre-commits the finding-classification policy (noise / defect /
  intentional-change) before the chain audit runs. Behavior-preserving stance
  with zero-manifest-edit default and the single conditional baseline move
  (attrs) fully scoped. §3 ruleorder rationale corrected after a scratchpad
  dry-run empirically refuted the initially-guessed ambiguity mechanism *and*
  showed the directive is not required for DAG construction on either test
  config on the current Snakemake — §3 now mandates reproduce-then-comment with
  a quoted exception rather than asserting an inferred reason.
