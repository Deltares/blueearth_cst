# WF3 — stress-test lookup and derived response-surface axes: Design (v1, DRAFT)

> **Status: DRAFT (v1)** — stage 1 of the `design-review-loop` run
> `wf3-stress-test-lookup`. Not accepted; gates G1 (framing) and G2 (design) are
> both `pending` (`status.md`).
> **Date:** 2026-08-15. **Genre:** workflow-spec with a method component (which
> statistic may define a response-surface axis, and why only affine ones), written
> in this repo's own design house style — structure precedent
> `dev/milestones/p32b/interchange-contracts-design.md`, per the `design-document`
> skill's software-system clause.
> **Author role:** cst-architect. **Scope authority:**
> `dev/working/design-runs/wf3-stress-test-lookup/intake.md`. **Seed source (six
> owner rulings, two same-day revisions):**
> `dev/working/2026-08-15_wf3-scenario-generation-trace/stress-test-design-and-surface-axes.md`,
> with companions `trace.md` (measured cost profile) and `wf3-rule-reference.md`
> (rules, scripts, file shapes).
> **Size budget for the normative body:** **1,250 lines**, set against this
> repo's accepted designs (p32b 1,066; p33 1,192; p32a 1,372). v1 is ~1,230. A
> revision that would push the body past the budget **relocates** superseded text
> into this run's review record rather than stacking new text on top of it —
> per-finding argument belongs there, cited by decision id, not here.
> **Self-contained:** a reviewer needs this file plus the cited paths.

---

## 1. Problem statement

WF3 fuses two things that are conceptually distinct.

| | what it is | what it owns |
|---|---|---|
| **The experiment** | perturbed climatology → simulated hydrology → simulated indicators | what was imposed, and what the system did |
| **The response surface** | a post-processed *view* of those indicators | how a member is summarised into an axis value, and how the plot is labelled |

Today the second is baked into the first, and that costs twice.

**A correctness cost.** `export_wflow_results.annual_perturbation` collapses each
member's twelve monthly perturbation values to one month-length-weighted annual
figure *at reduction time* and writes it into the indicator tables as
`temp_change` / `precip_change`. No other axis is recoverable from the results.
For a seasonal design the fixed annual collapse **misreports what was explored**:

```
(92 x 1.30 + 273 x 1.00) / 365 = 1.0757…  ->  +7.6%
```

so a member that imposed **+30% in JJA** is plotted at **+7.6%**. The more
concentrated the perturbation, the worse — a single-month perturbation compresses
to roughly a twelfth of its magnitude. Stress testing here is extended
sensitivity analysis: the axis must report *the range that was explored*, so for
a seasonal design this is a correctness defect, not a presentation preference.

**A duplication cost.** The grid is written as two artifacts:

- `<wg>/_work/st_<id>.csv` — twelve monthly rows, `month, temp_mean, precip_mean,
  precip_variance`; precip as a **multiplier**;
- `<exp>/config/stress_test_design.csv` — one row per member, `st_id,
  temp_change, precip_change, precip_variance_change`; precip as a **percent**,
  and the annual collapse of the first.

The second is a materialized cache of the first, derived independently by the
writer (`prepare_cst_parameters.py:175-189` writes the member CSV, reads it back
off disk and calls the same `perturbation_axes` the reduction calls), which is
why `validate_hm7` exists to police the drift between them.

This design replaces both artifacts with **one long lookup table at monthly
grain**, deletes the baked axis from the indicator tables, and makes every
response-surface axis a **derivation** performed at reporting time from that
lookup. The mechanism is largely unwritten in the seed note; six gaps are closed
here in normative text.

## 2. Goals / Non-goals

### Goals

- **G1 — one parameter artifact.** The monthly grid is a single table; the cache
  and one of its two derivations are gone, and with them the drift class
  `validate_hm7` guards (§5.1).
- **G2 — a declarable, validated axis.** A response-surface axis is a declared
  triple `{variable, months, statistic}` with a config schema, a tier story, and
  parse-time refusal of a malformed declaration (§5.2).
- **G3 — an assigned consumer.** Something owns joining lookup x indicators and
  deriving the axis; the decision names it and says why it is not a rule (§5.3).
- **G4 — enforcement for two asserted constraints.** "Only linear statistics may
  define an axis" and "varying months must carry the same `(min, max)`" both
  become executable checks with named homes and named failure behaviour (§5.4).
- **G5 — a derived caption**, specified as an algorithm, including the two cases
  the seed leaves open: months held at a non-zero offset, and a design with no
  varying months at all (§5.5).
- **G6 — a migration that lands in one commit**, with every live reference
  updated and a `naming.md` §7 record (§8), plus an HM-7 replacement that drops
  into `dev/reference/contracts/hydrological-model-seam.md` (§5.7).

### Non-goals

Carried verbatim from `intake.md`; none is re-opened here.

- **The projection overlay (Q6).** Deferred deliberately; its constraint is
  pinned as S10 so it cannot drift.
- **A third stress-test axis.** C28 refuses one deliberately; removing the
  *shape* barrier does not remove the *contract* barrier.
- **Members varying seasonal pattern independently** — a second design dimension,
  colliding with C28.
- **R12's execution model** — manifest, ledger, `member_hash`, resumable sweeps,
  epochs, quarantine, atomic publication. Owned by `t2608082036`.
- **Fixing `st_0`'s comparability** — `t2608151154`, `origin: R12`.
- **`precip_variance` in `member_hash`** — G1 retention ruling, followup `R9-F1`.
- Any change to CST-API, CST-frontend, or `csthelpers`.

## 3. Settled framing — carried forward, not re-opened

Eleven items arrive as settled. A reviewer may note a consequence; none may be
re-litigated. Each is restated as a decision of this design and given an `S` id
so later sections can cite it.

| id | ruling | source |
|---|---|---|
| **S1** | **Percent everywhere.** `temp_change` in degC; `precip_change` and `precip_variance_change` in **percent**. Column names stay unsuffixed (`precip_change`, not `precip_change_pct`) | Q1, owner 2026-08-15 |
| **S2** | **The lookup is the source of truth.** Indicator tables carry `st_id` + `value`; no baked axis; axis values are derived, never stored | Q2, owner 2026-08-15 |
| **S3** | **The lookup determines the AXIS, not the SCENARIO** | §3 qualifier, owner 2026-08-15 |
| **S4** | **No external consumer constrains this.** CST-API / frontend out of scope; `csthelpers` is parameterized and its owner updates it | Q3, owner 2026-08-15 |
| **S5** | **Name: `stress_test_lookup.csv`**, in `<exp>/config/`; `_work/` disappears | Q4, owner 2026-08-15 |
| **S6** | **`st_0` is not a surface member** — baseline reference only, reported as an annotated value; it stays simulated | §5, owner 2026-08-15, standing with a caveat |
| **S7** | **The identity member is simulated like any other.** The alias is withdrawn; `st_id` stays dense | §5, owner 2026-08-15 (withdrawal) |
| **S8** | **The lookup lands before R12's member-identity re-derivation**, which then keys `member_hash` on the monthly rows | §7b, owner 2026-08-15 |
| **S9** | **Only linear statistics may define an axis**, or HM-7's evenly-spaced guarantee breaks | §3, inherited from HM-7 |
| **S10** | **The same collapse must apply to the projection overlay** | HM-7; treatment deferred as Q6 |
| **S11** | Workflow engine only; hydromt / wflow conventions used verbatim, never re-engineered | `AGENTS.md` § Hard Constraints |

**The reasoning that keeps them alive** — carried because it is why each survives
review, not as decoration:

- **S1** was settled on cross-artifact consistency, not internal convenience. WF2
  already emits percent with an explicit `relative_units` column (`%` for precip,
  `degC` for temp — `projections/change_factor_table.py:65,88,154`); the WF3
  design table and the indicator tables are percent; the member files are the
  sole multiplier outlier. HM-7 requires the stress-test axes to match WF2's
  definition **because the GCM dots are overlaid on them**, so those two cannot
  diverge. The counter-argument — "store what is applied, because a rounding
  error changes the science" — does not survive: the incident behind it
  (`float32(0.7)` -> `-30.000001%`) was a float32-vs-float64 CSV round-trip
  problem, not a unit-choice one, and §5.1 D7 resolves it directly. Consequence
  worth knowing: no-change becomes `0.0` rather than `1.0`.
- **S2** rests on the lookup being a **sufficient statistic** for any collapse:
  it holds all twelve months for every member, so annual, seasonal, single-month
  and non-linear collapses are all projections of it. Keeping `temp_change` /
  `precip_change` in the results would privilege one collapse and re-create the
  drift `validate_hm7` polices, and it spares no one a join — a consumer needs
  the lookup regardless. The general principle: **store the finest grain actually
  imposed; derive every summary.** The same principle withdrew an earlier
  proposal to materialize a per-surface `axes.csv`.
- **S3** is the limit of S2. `st_0` and the grid's identity member carry
  *identical* all-zero rows (`stress_test_design.csv` shows `2,0.0,0.0,0.0` on
  the baseline config) and are demonstrably different climates — measured 70%
  apart on `q_mean_annual_min`. `st_0` is the raw generated series; every member
  is that series round-tripped through `apply_climate_perturbations`, which is
  **not the identity at unit factors** (weathergenr 1.2.0 calls
  `adjust_precipitation_qm` unconditionally; probed: temperature identity exact,
  every wet day changed, all twelve monthly means preserved to +0.0000%, single
  max day -32.9%). So the lookup cannot distinguish two rows that are not the
  same scenario. §5.1 D4 turns that from a caveat into a structural marker.
- **S6/S7** were settled together by the precondition test the seed note demanded:
  because the perturbation is not the identity at unit factors, there was never a
  duplicate to alias away. `st_0` stays simulated because two of the eleven `q`
  metrics are derived *from* it (the class-C month selection) and
  `run_historical: false` drops them with nothing reporting it.
- **S8** is a dependency with a direction, not a boundary dispute. R12's archived
  `member_hash` tuple includes `tavg`, `prcp`, `precip_variance` — field-noted as
  "the annual scalars the response surface is indexed by, derived exactly as the
  reduction derives them today", i.e. exactly the collapse this design abolishes.
  Re-deriving member identity against an artifact about to change spends the work
  twice. The replacement is strictly more faithful: a digest over the member's
  twelve lookup rows.
- **S9** is what lets a consumer rely on the axis being evenly spaced. Members
  are `min + (j/n)(max - min)` month by month, so any affine collapse is affine in
  the step index and the surface stays rectilinear; a max or a quantile is not.

## 4. Decision criteria

From `intake.md`, restated so §6's rejections can cite them.

1. **C1 — correctness first.** The axis must report the range that was explored.
   A design that keeps the misreport is rejected regardless of other merits.
2. **C2 — store the finest grain imposed; derive every summary.**
3. **C3 — no new cache of a derivation, at any layer.**
4. **C4 — a new perturbation parameter should be a column, not a file shape** —
   while respecting C28's deliberate refusal of a third *axis*.
5. **C5 — the migration is a rename plus a shape change**, executable in one
   commit with every live reference updated.
6. **C6 — gate-ability.** Every claimed runtime property must have an observation
   that would falsify it.

## 5. Selected approach

### 5.1 The lookup table — schema, units, placement

**D1 — one long table at monthly grain**, replacing both `<wg>/_work/st_<id>.csv`
and `<exp>/config/stress_test_design.csv`. Written by rule 3.09
(`prepare_stress_test_grid`), which keeps its current job: it is the same loop
that enumerates the members, so the enumeration that names a member and the one
that describes it still cannot disagree (C26's property, preserved).

**D2 — path and header.** `<exp>/config/stress_test_lookup.csv` (S5), beside the
config snapshot whose settings produced it — it is a record of what ran, not
scratch. Header, in this order:

```
st_id,month,temp_change,precip_change,precip_variance_change
```

- `st_id` — the member id, **zero-padded to `index_width(ST_NUM)`**, textually
  identical to the member filename token (C27). **Read it as a string**;
  `pd.read_csv` with no `dtype` turns `01` into `1` and silently breaks the join.
  R readers pass `colClasses = c(st_id = "character")` for the same reason.
- `month` — integer `1..12`, calendar months.
- `temp_change` — additive degC (S1).
- `precip_change`, `precip_variance_change` — **percent** (S1).

Rows are sorted by `(st_id, month)`; the file carries `12 x ST_NUM` rows.

**D3 — the config keeps its multiplier convention.** `stress_test.precip.mean`
and `.variance` stay 12-element multiplier vectors in the project config
(`min: [0.7 …]`, `max: [1.3 …]`). S1 rules the *artifact* units, and the seed's
own taxonomy is written in multiplier terms as "already expressible in the
current config". Changing the config surface would be a user-facing break with no
ruling behind it. Two conversion sites, and only two:

| where | conversion |
|---|---|
| rule 3.09 writing the lookup | `precip_change = precip_mean * 100 - 100` |
| `impose_climate_change.R` applying it | `precip_mean = 1 + precip_change / 100` |

The multiplier survives only as the generator's operation form. The formula is
written as `f * 100 - 100` and **not** as `(f - 1) * 100`: measured,
`1.3*100-100 == 30.0` exactly while `(1.3-1.0)*100 == 30.000000000000004`. The
inverse restores `1.3` exactly. This is the formula
`export_wflow_results.perturbation_axes` already uses; it is preserved, not
invented.

**D4 — `st_0` has no row in the lookup.** The table covers members `1..ST_NUM`
only. Three reasons, in order:

1. It is the **parameter grid**, and `st_0` has no parameters. It is produced by
   rule 3.11, not by perturbation; rule 3.12 never runs for it.
2. An all-zero `st_0` row is **indistinguishable from an identity member's row
   while denoting a differently-processed climate** (S3). The seed records the
   need inverted — what wants marking is "these identical rows are *not* the same
   scenario" — and **absence is the strongest available marking**: a consumer
   joining results to the lookup finds no axis for `st_0` and cannot place it on
   the surface by accident. That implements S6 mechanically rather than by
   convention.
3. It keeps the caption algorithm honest. With an all-zero `st_0` row present,
   case 3 (months held at -20%) reads *every* month as varying, and the caption
   for the exact case the seed says is uncovered comes out wrong.

This **amends C23's recorded rationale**, and the amendment is deliberate:
`prepare_cst_parameters.py:117` justifies the `st_0` design row as "a response
surface missing its own origin forces every downstream consumer to reconstruct
it", i.e. it assumed `st_0` **is** the surface origin. S6 rules that it is not.
The comment must be corrected in the same commit rather than left asserting a
superseded intent.

Consumers still see `st_0`: the indicator tables carry `st_id = 0` rows, and
§5.3's join returns them as the **annotated baseline**, separately from the
surface rows.

**D5 — the column vocabulary is closed, and the refusal stays in rule 3.09.**
The five columns above are the whole vocabulary. A new perturbation parameter is
a **column** (C4), and adding one requires a C28 ruling — removing the *shape*
barrier does not remove the *contract* barrier. The existing guard is already in
the right place and keeps its job unchanged: `prepare_cst_parameters._KNOWN_AXES`
refuses an unknown key under `stress_test:` at write time. What retires with the
baked axis is the *second* half of C28's obligation — the reduction's refusal of
"a design table carrying an axis this header cannot express" — because the
indicator header no longer expresses any axis (§5.6 D14, §5.7).

**D6 — no `alias_of` column, and no per-surface `axes.csv`.** The first is
withdrawn with the alias (S7); the second is withdrawn by C3 — given the lookup,
an axis table stores something already fully determined. The one case for
materializing derived axis values, recorded so it stays a decision rather than an
oversight: archiving a *published* figure, where the exact plotted numbers should
sit beside it. That is publication provenance, served by an export-on-demand, not
by writing a file every run.

**D7 — value precision: the float32 shortest-repr round trip is preserved.**
Today `prepare_cst_parameters` builds each member frame as `float32`, writes it,
and **re-reads it off disk** before deriving the design row, because a design row
computed from the in-memory `float32` records a perturbation nobody applied
(`float32(0.7) = 0.69999998807` -> `-30.000001%`, while the run imposes the
round-tripped `0.7`). With one artifact there is no second derivation and no
disk round trip to perform, so the same property is obtained directly: the
lookup's values are the **shortest text that round-trips the `float32` level**
(`float(str(np.float32(v)))`, which is exactly what `to_csv` writes today),
converted by D3's formula. R11 P3's read-it-back hack disappears with the cache
it existed to serve.

The `float32` quantization itself is **kept**, deliberately. It is what the
generator receives today, so the migration's numerical effect on the forcing —
and therefore on every indicator value — is **nil**. That is what makes §9's
strongest falsifier available: after migration, `value` must be unchanged within
the baseline comparator's tolerance, so the re-record differs from the current
baseline in the **column set only**.

### 5.2 The surface declaration — config schema and tiers

**D8 — a response surface is declared in the project config, in a new top-level
`reporting:` section.**

```yaml
reporting:
  surfaces:
    - id: jfm                          # required; [a-z0-9_]+, unique in the list
      x: {variable: temp}              # axis declaration
      y: {variable: precip, months: [1, 2, 3], statistic: mean}
```

An axis declaration is the seed's triple `{variable, months, statistic}`:

| field | required | domain | default |
|---|---|---|---|
| `variable` | yes | closed enum `temp` \| `precip` | — |
| `months` | no | list of ints `1..12`, non-empty, unique | **the member-varying month set**, derived from the lookup (D11) |
| `statistic` | no | closed enum, today `mean` only (D12) | `mean` |

`variable` admits no third member. `precip_variance` is a lookup column, not a
grid dimension — its levels are indexed by the *precip* step, so an axis over it
would be a relabelling of the precip axis — and admitting it would be the third
axis C28 refuses.

**Why a new top-level section rather than `workflows.run_stress_test`, measured
rather than argued.** `write_experiment_config._frozen_differences` is a
**key-union diff over the resolved `run_stress_test` section**, and
`experiment.yml` is immutable once the experiment has run
(`ExperimentConfigFrozenError`). A `surfaces:` key under `run_stress_test`
therefore makes **relabelling a surface trip the experiment freeze** — which
inverts the entire experiment/surface separation this design exists to create:
the one thing a derived axis must permit is re-describing a completed run without
re-running it. Probed both ways against the shipped rapid config (P2, §9):

| home | `effective_config_digest` moves? | `_frozen_differences` |
|---|---|---|
| top-level `reporting:` | **no** | `[]` |
| `workflows.run_stress_test.surfaces` | yes | `['surfaces']` — the freeze refuses |

`CONFIG_PROJECTION` is `('project', 'shared', 'workflows.analyze_projections',
'workflows.build_model', 'workflows.run_stress_test')`, so a top-level
`reporting:` section is outside configuration identity by construction. That is
the **correct** semantics, not a loophole: the effective-config digest answers
"the settings the workflow was asked to run under", and a caption is not one.
`reporting:` is also outside `guarded_sections`, so it cannot thrash rule 3.01's
shared guard artifact. It is still **recorded**: rule 3.02 byte-copies the config
as run into `<exp>/config/snake_config_run_stress_test.yml`, so S2's obligation
that the collapse and the caption are a *choice that must be recorded* is met
without paying the freeze.

The section is named `reporting:` rather than `surfaces:` so the name declares
the tier — everything under it is post-processing, outside run identity — and so
Q6's overlay collapse has an obvious home when it is designed.

**D9 — the toolbox tier does not exist, and `advanced_settings.yml` is not
touched.** `config/advanced_settings.yml` has a **closed** schema
(`snake_utils._ADVANCED_SETTINGS_SCHEMA`) with three sections whose validators
are scalar-typed (`positive_int`, `nonnegative_int`, `month_abbrev`,
`version_string`), and the standing rule is that a setting is added to the file
and to the schema *together*. The surface declaration does not belong in any of
its three sections:

- `constraints:` is for **hard limits no project config may relax**. The surface
  default is not a limit.
- `defaults:` is for **starting values a project config may override**. This is
  the near miss, and it fails on substance rather than on typing: a toolbox-wide
  default surface that a user could edit would silently change the axis labels of
  *every* project on the machine, and there is no reason for one machine's
  projects to disagree with another's about what "the axis" means when nothing is
  declared.
- `runtime:` is for external toolchain pins.

So the design **meets the closed schema by not extending it**, and the "no
declaration" case is defined in code instead: `shared/surface_axes.DEFAULT_SURFACE`
is one surface, `id: annual`, `x: {variable: temp}`, `y: {variable: precip}`,
both axes taking D11's derived month set and `statistic: mean`. It is a constant
of the toolbox in the same sense `_PERTURBATION_AXIS` is, and it needs no new
validator class for a structured value.

**D10 — `surfaces:` is a list of zero or more.** Q5 is closed by S2: the lookup
is a sufficient statistic, so N surfaces need nothing beyond N derivations, and a
list of one costs nothing more than a scalar. Refusing N would be an arbitrary
barrier given the mechanism. Absent or empty ⇒ `DEFAULT_SURFACE`.

The **limit worth stating so nobody rediscovers it as a bug**: within one
experiment every member lies on the line from `min` to `max`, so every affine
axis is an affine image of every other — **two surfaces from one experiment
differ in magnitude and label, not in shape or member ordering.** That is not a
limitation to fix; it is what makes the seasonal case worth having. Reporting
"+30% over JFM" instead of "+7.6% annual" is the *same* surface, correctly
labelled. Genuinely different response *shapes* need members varying seasonal
pattern independently — a second design dimension, out of scope.

**D11 — `months` defaults to the member-varying set, derived from the lookup.**
A month is **varying** when its value differs across the surface members
(`max - min > 0` over `st_id` in `1..ST_NUM`) and **held** otherwise. The default
is the varying set; when nothing varies (D17's degenerate axis) it is all twelve.

This is the single most consequential default in the design, and it follows from
C1. Today's axis is the twelve-month mean, so a JFM design plots at +7.6%.
With this default:

- a **uniform** design (every shipped config) has all twelve months varying, so
  the derived axis is **identical to today's** — the default is
  behaviour-preserving exactly where today's behaviour is correct;
- a **seasonal** design reports the imposed value automatically, without the user
  having to remember to declare anything.

Requiring an explicit declaration would leave the misreport as the default and
make correctness opt-in, which C1 rejects. The triple remains fully declarable —
only its default changes, which is also the seed's own "the caption should be
derived, not typed" applied one level up.

**D12 — `statistic` is a closed enum with one admitted member: `mean`**, the
month-length-weighted mean over the declared months (`_MONTH_LENGTHS` in the
weather generator's `noleap` calendar, matching `annual_perturbation` and WF2's
`get_change_climate_proj._annual`). Enforcement of S9 is by **closed
vocabulary**, not by inspecting a callable: an unknown statistic is refused at
parse time. Extension rule, stated so a future addition is not a judgement call:
**a statistic may be added only if it is affine in the member's step index, with
the proof recorded in this document's revision log.** A max, a quantile or a
variance is not, and admitting one breaks HM-7's evenly-spaced guarantee.

Under D14's homogeneity constraint the weighting is immaterial — every value
being averaged is equal — so month-length weighting is chosen for compatibility
with WF2 and the S10 overlay rather than for its arithmetic effect. It is
nonetheless specified exactly, because the overlay must apply *the same*
collapse.

**D13 — the declaration is validated at Snakefile parse time.** `run_stress_test.smk`
calls `surface_axes.parse_surfaces(config)` beside the existing
`refuse_retired_experiment_keys(my_cfg)` — the repo's established parse-time
refusal pattern. Consequences: a malformed declaration fails `--dry-run`, so
`pytest tests/test_cli.py` is its gate; no rule declares it, so no DAG edge and
no rerun-trigger hazard is created (§5.3). Each surface entry has a **closed key
set** (`id`, `x`, `y`) and each axis a closed key set (`variable`, `months`,
`statistic`), so a typo *inside* the declaration is refused rather than ignored —
the R11 Q7 posture. A typo in the section name itself is not catchable: the
config root has no section whitelist (verified — nothing in the Snakefiles or
`snake_utils` enumerates allowed top-level sections). Recorded as a residual risk
in §7, not papered over.

### 5.3 The consumer side — what derives an axis

**D14 — the derivation is a library, not a rule.** `blueearth_cst/shared/surface_axes.py`,
a pure module with no Snakemake dependency:

```python
DEFAULT_SURFACE: Surface                      # id "annual", temp x precip, derived months
def parse_surfaces(config) -> list[Surface]   # parse-time; refuses (D13)
def read_lookup(path) -> pd.DataFrame         # dtype={"st_id": str}
def month_classes(lookup, variable) -> tuple[list[int], dict[int, float]]   # varying, held->level
def axis_values(lookup, axis) -> pd.Series    # st_id -> axis value, refuses per D15/D16
def axis_caption(lookup, axis) -> str         # §5.5
def join_axes(indicators, lookup, surface) -> tuple[pd.DataFrame, pd.DataFrame]
```

`join_axes` returns `(surface_rows, baseline_rows)`: indicator rows whose `st_id`
is in the lookup, with the two derived axis columns attached, and — separately —
the rows whose `st_id` is absent from the lookup, which is exactly `st_0`. That
is S6 implemented as a partition rather than as a documented convention, and it
is why D4's omission of `st_0` from the lookup is load-bearing rather than tidy.

**The derived columns keep the old names**, `temp_change` and `precip_change`,
named once as `surface_axes.AXIS_COLUMN[variable]`. What changes for a consumer
is that it must join the lookup to obtain them; what does not change is the
column it then plots, so an existing call site keeps working and simply receives
values that are now correct for a seasonal design. Two declared surfaces produce
two frames, each carrying that pair — the surface `id` names the frame, not the
columns.

**Why not a rule.** Four reasons, in order of weight:

1. **There is no in-repo consumer.** WF3 has no plotting rule — rules 3.01–3.18
   end at the reduction and the record gathers. The consumers of a response
   surface are CST-API, the frontend and `csthelpers`, all out of scope by S4. A
   rule would therefore produce an artifact nothing in this repo reads.
2. **A rule that wrote axis values would be a cache of a derivation** (C3), which
   is the exact proposal S2's third consequence already withdrew one layer up.
3. **R12 owns how WF3 executes** (`t2608082036`). Adding a rule, its rerun
   triggers, its log part and its benchmark part is run mechanics, and this design
   has no justification to spend R12's budget on a file nothing reads.
4. **It would inherit a live hazard.** Rule 3.09 declares `config = ancient(...)`
   and carries no `params:`, so it is deaf to `stress_test` edits (E5); a new rule
   reading a config section would face the same choice and the same trap.

**D15 — the derivation is specified, not only implemented.** An R or JavaScript
consumer cannot import a Python module, and S10 requires the *same* collapse over
WF2's monthly change factors. The collapse is therefore written into the HM-7
contract text (§5.7) as a formula, with `surface_axes.py` as its **reference
implementation**. `csthelpers::plot_climate_surface` needs no in-repo change:
verified this run (E9, §9) that it takes `x_var`/`y_var` as required arguments,
validates them against `names(data)`, and derives its breaks from
`sort(unique(data[[x_var]]))`; `precip_change` occurs **nowhere in the package's
`R/` sources** — only in `inst/examples/usage_climate_surface.R` and two vendored
snapshot CSVs under `data/`. A caller that joins the lookup and passes a derived
column name is already supported.

### 5.4 Enforcement — the two asserted constraints

Two properties the design asserts and nothing checks today. They are independent:
the probe in §9 shows a heterogeneous design whose axis is nevertheless perfectly
evenly spaced, so neither check substitutes for the other.

**D16 — heterogeneous varying months: warn at the design, refuse at the axis.**
The axis is interpretable only when every varying month shares the same `(min,
max)`; then the mean over those months *equals* the change applied to each, and
the axis reports the imposed value rather than an average of unlike things.

- **Design tier — a WARNING at parse time.** `parse_surfaces` (D13) also inspects
  `stress_test:` and emits a warning when the varying months carry differing
  `(min, max)` pairs. It does **not** refuse: such an experiment is legitimate and
  runnable — the members exist, the response is real — only its *scalar summary*
  is dishonest, and refusing would forbid a legal experiment to prevent a bad
  label.
- **Axis tier — a REFUSAL.** `axis_values` / `axis_caption` raise
  `HeterogeneousAxisError` when the declared months are not homogeneous. A caption
  is a claim, and no caption can honestly describe a mean of unlike
  perturbations. The refusal is not a dead end: it names the homogeneous subsets,
  so the user declares one and gets an honest axis.

**A declared month set must not contain a held month.** Held months contribute a
constant to the mean, so including one reproduces exactly the misreport C1
rejects — declaring `months: [1..12]` on a JFM-varying, Apr–Dec-held-at-`-20%`
design returns `-15%` for a member that imposed `-30%` in JFM. Rule: the declared
set must be a **non-empty subset of the varying set**; a held month raises
`HeldMonthInAxisError`. A proper subset of a homogeneous varying set is
admissible and returns the same value.

**D17 — non-affine statistics: closed vocabulary plus a postcondition.** D12
closes the vocabulary, which is the static half. The dynamic half:
`axis_values` asserts that the **distinct axis levels are evenly spaced** to
within a relative tolerance and raises `NonRectilinearAxisError` otherwise. This
turns HM-7's "a consumer may rely on the axis staying evenly spaced" from a
documented promise into a checked postcondition, and it is the check that would
catch an affineness argument that turns out to be wrong — the failure mode a
closed enum cannot see. Two or fewer distinct levels pass trivially.

**Where the checks run, and why not in rule 3.09.** All of D16's axis tier and
D17's postcondition live in the library, which is where the axis is computed.
The design-tier warning runs at Snakefile parse time. Rule 3.09 is deliberately
not given the job: it is `ancient()` with no `params:`, so it is deaf to a
`stress_test` edit and a check placed there would not re-run when the thing it
checks changes. That gap is already covered from the other side — 3.07 carries
`experiment_cfg` in `params:`, so an in-place `stress_test` edit on an
already-run experiment is refused by `ExperimentConfigFrozenError` — and its
repair belongs to R12, which inherits `file_digest_or_absent` threading through
`params:` for exactly this.

### 5.5 The derived caption — algorithm

**D18 — captions are derived from the lookup, never typed.** A typed label can
drift from the design it describes; a derived one cannot. This is the seed's
strongest argument for the merged table beyond simplification.

Inputs: the lookup, one `variable`, and the declared month set `M` (D11). All
classification is over the **surface members only** — `st_0` has no row (D4), so
no exclusion step is needed.

**Step 1 — classify months.** For each month `m`, over `st_id` in `1..ST_NUM`:
varying if `max - min > tol`, else held at level `L_m`.

**Step 2 — label a month set.** Deterministic, in this order:

1. all twelve -> `the year`;
2. a contiguous run in **circular** month order of length <= 3 -> the initials
   (`JFM`, `JJA`, `DJF`);
3. a contiguous circular run of length >= 4 -> `<first>–<last>` (`Apr–Dec`,
   `Sep–May`);
4. otherwise -> a comma list of three-letter abbreviations (`Jan, Mar, Jul`).

Rule 2 subsumes the meteorological seasons without a season table: `{12,1,2}`
renders `DJF` and `{6,7,8}` renders `JJA` by construction. Rule 3 exists because
the initials of a nine-month run (`AMJJASOND`) are unreadable, and because the
seed's own worked caption spells that complement `Apr–Dec`.

**Step 3 — format a level.** `+3 degC` / `-20%`, signed, three significant
digits, unit per `variable`.

**Step 4 — compose.** Let `H` be the held months and `G` the distinct held
levels.

| case | condition | caption |
|---|---|---|
| 1 — uniform | `H` empty | `mean change over the year` |
| 2 — some vary, rest unchanged | `G == {0}` | `mean change over JFM; Apr–Dec unchanged` |
| 3 — some vary, rest held at one offset | `\|G\| == 1`, non-zero | `mean change over JFM; Apr–Dec held at -20%` |
| **3b — rest held at several offsets** | `2 <= \|G\| <= 3` | `mean change over JFM; Apr–Sep held at -20%; Oct–Dec held at -10%` |
| **3c — more than three held levels** | `\|G\| > 3` | `mean change over JFM; remaining months held at declared monthly offsets` |
| **4 — nothing varies** | varying set empty | see D19 |

Cases 3b and 3c are the seed's uncovered "case 3" generalised. The cap at three
groups is a legibility rule, not a correctness one: beyond three, the honest
statement is that the held pattern is not summarisable in a caption and the
reader should look at the lookup.

**D19 — an axis with no varying months is degenerate, not an error.** Every
member shares one value, so the axis has a single level. Refusing would break a
**legitimate** experiment: a temperature-only stress test (`precip` flat, `temp`
stepped) is exactly this on its precip axis, and it is a one-dimensional design,
not a malformed one. So `axis_values` returns the constant and marks the axis
`degenerate = True`, and the caption is:

| held levels | caption |
|---|---|
| all zero | `unchanged` |
| one non-zero level `L` | `held at -20%` |
| several levels | `held at declared monthly offsets` |

A degenerate axis is an **annotation**, not a plot dimension; a consumer that
receives `degenerate = True` renders it in the caption rather than as an axis.
The rectilinearity postcondition (D17) passes trivially on one level.

All six cases and both hazard cases were executed against `np.linspace` member
matrices this run; the rendered captions are in §9.

### 5.6 Rule-level changes, and what leaves the DAG

**D20 — rule 3.09 `prepare_stress_test_grid`.** Outputs collapse from
`ST_NUM` member CSVs plus the design table to **one** `lookup_csv`. The
`_KNOWN_AXES` refusal, the `index_width` padding and the single-loop enumeration
are unchanged. `ancient(config_path)` and the absence of `params:` are
unchanged — that is R12's territory (§5.4).

**D21 — rule 3.12 `perturb_climate_realization`.** Its `st_csv` input becomes the
**constant** `lookup_csv`; the member id reaches the script as a new positional
argument derived from the `{st_num}` wildcard. `impose_climate_change.R` filters
`st_id == <padded token>` (reading `colClasses = c(st_id = "character")`), orders
by `month`, and converts percent to the generator's factor form
(`1 + precip_change/100`). Everything else in that script — the
prefix/suffix split from the declared output path, the transient flags, the
`apply_climate_perturbations` call — is untouched.

**`wildcard_constraints: st_num=member_index_regex(ST_WIDTH)` stays, and stays
load-bearing.** Probed (P1, §9): with the input no longer carrying the member
wildcard, removing the constraint still yields `CyclicGraphException in rule
perturb — Cyclic dependency on rule perturb`, exactly as the rule's own comment
records; keeping it yields a clean DAG with the fan-out unchanged.

**D22 — rule 3.16 `derive_wflow_indicators` reads no parameter artifact at all.**
Both `st_csv_fns` and `design_csv` are dropped from its `input:`. It needed the
first for axis values (now derived downstream, S2) and the second for the id
width and C28's refusal. The width comes from `index_width(st_num)` — the same
shared helper rule 3.09 pads with, so the two spellings still cannot diverge —
and the refusal moves wholly to 3.09 (D5).

**The run-time coverage check is preserved, not dropped.** Today the reducer
raises when the declared parameter files do not cover `1..st_num`. That check
moves to what actually matters and gets stronger: **the member set recovered from
the run CSVs must equal `range(ST_START, st_num + 1)`**, checked from `params`
before any reduction work. It now verifies what *ran* rather than what was
*declared*, and it survives at the run-time tier, which matters because
`validate_hm7` is a test-time validator — the HM-7 record notes its "no rows"
check "is never invoked at run time". §5.7's completeness check adds the
lookup↔tables tier on top; it does not replace this one.

**D23 — the `rule all` target entry stays, renamed.** `WF3_TARGETS` keeps an
entry, `stress_test_lookup` -> `<exp>/config/stress_test_lookup.csv`. The lookup
*is* now demanded by 3.12, so it is reachable during a fresh run — but on a
**completed** experiment every 3.12 output is `temp()` and already deleted, so
nothing would re-demand a lookup deleted from disk. That is precisely the
reachability argument C23 made for the design table, and it still holds.

**D24 — the fan-out is unchanged.** 3.12 remains `RLZ_NUM x ST_NUM` jobs and 3.14
remains `RLZ_NUM x (ST_NUM+1)`. The per-file split bought no invalidation
granularity in the first place: rule 3.09 declares **all** member files as one
job's outputs (`run_stress_test.smk:817-836`), so any config change that re-fires
it rewrites all of them and re-fires every 3.12 job anyway.

### 5.7 HM-7 replacement text (drop-in)

Replaces the `## HM-7` section of `dev/reference/contracts/hydrological-model-seam.md`.
Everything not restated below is unchanged (path pattern, producer, consumer,
variable tokens, `rlz_id` grain, `location`, `basin` reservation, `aggregate_rlz`
retirement, `temp()` lifecycle, the `RT_*.csv` removal note, and the
HM-4 -> HM-5 -> HM-7 gauge-column invariant, which does not touch these columns).

> **Pinned surface:** every table carries **exactly five columns, in this
> order**:
>
>     metric, location, st_id, rlz_id, value
>
> *What* (`metric`, `location`), *which member* (`st_id`, `rlz_id`), then the
> number. The header does not grow with the gauge count — locations are ROWS —
> and it no longer grows with the stress-dimension count either. `value` is
> `float32` and unrounded.
>
> **`temp_change` / `precip_change` were removed on `<DATE — the implementation
> commit's date>`**, and the removal is
> the point rather than a simplification. They held a **month-length-weighted
> annual mean** of the member's twelve monthly perturbations, taken at reduction
> time, which **misreports any seasonal design**: +30% imposed in JJA is
> `(92 x 1.30 + 273 x 1.00)/365 = +7.6%` on the axis, and a single-month
> perturbation compresses to roughly a twelfth of its magnitude. Baking one
> collapse into the results also made every other axis unrecoverable from them.
> Record: `dev/working/design-runs/wf3-stress-test-lookup/design-v1.md`.
>
> **The response-surface axis is now a derivation, defined here.** Its source is
> `<exp>/config/stress_test_lookup.csv` (`st_id, month, temp_change,
> precip_change, precip_variance_change`; `temp_change` in degC, the other two in
> percent; `st_id` zero-padded text; twelve rows per member; **no `st_0` row** —
> `st_0` is the reserved unperturbed baseline, is not a surface member, and is
> reported as an annotated reference value beside the surface). For a declared
> axis `{variable, months M, statistic mean}`:
>
>     axis(st) = sum_{m in M} w_m * v(st, m) / sum_{m in M} w_m
>     w_m      = the month's length in the noleap calendar (31, 28, 31, …)
>     v(st, m) = the lookup's <variable>_change value for member st, month m
>
> `M` defaults to the **member-varying months** — those whose value differs
> across `st_id` — which is what makes the default axis report the range actually
> explored. Two constraints a consumer may rely on and an implementation must
> enforce:
>
> - **Only affine statistics.** Members are `min + (j/n)(max - min)` month by
>   month, so an affine collapse is affine in the step index and **the axis is
>   evenly spaced across the grid**. A max or a quantile is not, and the surface
>   stops being a regular grid. The admitted vocabulary is `mean` alone.
> - **`M` must be a non-empty subset of the varying months, and those months must
>   share the same `(min, max)`.** Otherwise the mean averages unlike
>   perturbations and no caption can describe it honestly. Including a held month
>   reproduces the annual misreport this contract removed.
>
> **The same collapse must be applied to the projection overlay.** The CMIP6 dots
> are placed on these axes, so two different collapses would compare two
> different quantities. WF2 emits monthly change factors in percent
> (`cmip6_change_factors_monthly.csv`), so the same month-set collapse runs over
> the GCM table and over the lookup with no unit conversion between them.
> Overlay treatment is deferred (Q6); the constraint is not.
>
> **`st_id` (C28, R11 P2).** The design point's id, zero-padded to the same
> count-derived width as the member filename, so the two are ONE token. **Read
> it as a string** — `pd.read_csv` with no `dtype` returns `01` as `1` and the
> join to the lookup silently misses. C28's second obligation — the writer
> refusing a design table carrying an axis the header cannot express — retires
> with the axis columns: the header expresses no axis, so a third perturbation
> parameter no longer needs a results column. The **contract** barrier stands:
> a new column in the lookup requires a C28 ruling, refused today by
> `prepare_cst_parameters._KNOWN_AXES`.
>
> **Validator:** `validate_hm7`. The **cache-drift check retires with the
> cache** — with one artifact there is no second derivation to disagree with, so
> the drift class is eliminated structurally rather than merely unchecked. What
> replaces the guarantee it was providing:
>
> 1. **Completeness, both directions, kept and re-pointed at the lookup.** Every
>    `st_id` in the lookup appears in every indicator table, and every non-zero
>    `st_id` in a table appears in the lookup. This half of the old check was
>    added because its absence hid a defect (R11 P3): a seed config with
>    `run_historical: false` dropped the `st_0` baseline and with it two of eleven
>    metrics — 180 rows — with the validator green.
> 2. **The `st_0` partition.** `st_0` rows are expected in the tables and
>    expected **absent** from the lookup; either violated is a divergence. Two
>    identical all-zero rows would otherwise be indistinguishable from an identity
>    member's, and they are not the same scenario: `st_0` is the raw generated
>    series while every member is that series round-tripped through a perturbation
>    that is not the identity at unit factors.
> 3. **The axis postcondition, at the point of derivation.** The reference
>    implementation asserts the distinct axis levels are evenly spaced, which is
>    the check the evenly-spaced guarantee never had.

## 6. Alternatives considered

**6.1 Keep the two artifacts and fix only the collapse.** Make
`annual_perturbation` seasonal-aware and leave the design table in place.
Rejected by C1 + C3: it keeps a cache of a derivation, keeps `validate_hm7`'s
drift check, and — decisively — leaves the *choice of collapse* frozen at
reduction time, so a second surface still requires a re-run. It would become
preferable only if the monthly grain were genuinely unavailable at reporting
time, which it is not.

**6.2 Materialize a per-surface `axes.csv`.** Proposed in an earlier draft of the
seed note and withdrawn by the owner: given the lookup, an axis table stores
something already fully determined — the same cache-of-a-derivation this design
removes, one layer up (C3). It would become preferable for *publication
provenance*, where the exact plotted numbers should sit beside an archived
figure; that is better served by an export-on-demand than by writing a file every
run (D6).

**6.3 Put the surface declaration under `workflows.run_stress_test`.** The
obvious home — it is WF3's own setting, it rides the existing config snapshot,
and it needs no new section. Rejected on measurement: `_frozen_differences`
reports `['surfaces']`, so the experiment freeze refuses any relabel after the
first successful run, which inverts the separation this design exists to create;
and the effective-config digest moves, so a caption edit re-fires the record
rules and marks a run stale that is not. Making it work needs a carve-out list in
the freeze — new machinery whose only job is to say "this key is not really an
experiment parameter", which is what a separate section says for free. It would
become preferable if the freeze ever gained a principled reporting/parameters
split for other reasons.

**6.4 A new rule that writes derived axis values.** Rejected by C3 (a cache), by
the absence of any in-repo consumer, and by the R12 boundary — the rule's rerun
triggers, log part and benchmark part are run mechanics owned by `t2608082036`.
It would become preferable the moment WF3 grows its own plotting rule, at which
point that rule consumes the library directly and still stores no axis.

**6.5 Enforce S9 by inspecting the statistic.** Accept an arbitrary callable or
named function and test it for linearity. Rejected: linearity of an arbitrary
callable is not statically decidable, and a sampled test would be a probabilistic
gate on a correctness property. A closed vocabulary plus a rectilinearity
postcondition gives an exact static check and an exact dynamic one (D12, D17).

**6.6 Refuse a heterogeneous design outright at parse time.** Symmetrical and
simple. Rejected: such an experiment is legal and runnable, and refusing it would
forbid a valid run to prevent a bad label. The two-tier split — warn at the
design, refuse at the axis — puts the refusal where the false claim would be made
(D16). It would become preferable if such designs were found to be always
accidental, which nothing shows.

**6.7 Keep `st_0` in the lookup as twelve zero rows.** Symmetrical with today's
design table, and it makes `st_0` joinable. Rejected: `st_0` has no parameters
and is not produced by perturbation; an all-zero row asserts a scenario identity
the lookup cannot support (S3); and it corrupts the caption's varying/held
classification for exactly the case the seed leaves open (D4). It would become
preferable if `st_0` ever became a surface member, which S6 rules out.

**6.8 Rename to `experiment_design.csv` / keep `stress_test_design.csv`.**
Rejected by the owner's Q4 ruling: `stress_test` keeps the file in the same
vocabulary as its own key column (`st_id`, `stress_test:`, `ST_NUM`,
`rlz_1_st_4`), and `lookup` rather than `design` because the rename's job is to
**signal that the shape moved** to long form — "design" describes the old
artifact just as well, so keeping it would pay a migration and buy nothing.

## 7. Consequences and risks

**Falsifiable consequences.**

1. **A relabel needs no re-run and no new experiment name.** Editing
   `reporting.surfaces` leaves `effective_config_digest` unchanged and
   `_frozen_differences` empty (measured, §9). Falsifier: an edit that trips
   `ExperimentConfigFrozenError` or re-fires 3.02/3.16b.
2. **Indicator values do not move.** The forcing is bit-identical (D7), so the
   re-recorded baseline differs from the current one in the **column set only**.
   Falsifier: `compare_indicator_table` reporting a numeric failure — as opposed
   to the expected structural column-set mismatch — between a pre-change and a
   post-change run of `snake_config_baseline.yml`.
3. **The default axis is unchanged for a uniform design and corrected for a
   seasonal one.** Falsifier: a uniform design whose derived axis differs from
   today's `annual_perturbation` output, or a JFM design still reporting +7.6%.
4. **The DAG shape is unchanged.** 3.12 fans out `RLZ_NUM x ST_NUM`, 3.14
   `RLZ_NUM x (ST_NUM+1)`. Falsifier: a `--dry-run` job count that differs from
   the pre-change run on the same config.
5. **A stale `_work/` or `stress_test_design.csv` reports.** After §8's inventory
   change, `pixi run tree-check` classifies both as undeclared (measured, §9).
   Falsifier: either classifying IDENTITY.

**Risks.**

- **R1 — the baseline gate fails by design and must be re-recorded first.**
  Dropping two columns changes the `indicator` target's column set, and
  `compare_indicator_table` treats every non-`value` column as part of the row
  key, so the comparison is a **structural** failure. Two open board items say
  the current baseline cannot serve as the "before": `t2608131718` (the
  baseline's two flat config copies stale since 2026-08-12) and `t2608121258`
  (the `test_local` fixture predates the weathergenr 1.2.0 rename). **A
  re-record must happen before the first implementation commit**, from
  `snake_config_baseline.yml`, in the primary checkout, with `--notemp` on WF1
  and no other session live — otherwise every step landing before it is
  permanently ungateable. A **second** re-record closes the migration (§8 step
  7), because the stored reference otherwise keeps seven columns and the gate
  stays red forever; the numeric question in between is answered by §9's V4
  procedure, not by the gate. This is the highest-cost item in the plan and it is
  sequencing, not code.
- **R2 — the artifact being replaced is outside the numerical gate entirely.**
  `stress_test_design.csv` was deliberately kept out of the baseline manifest by
  an R11 ruling, so `check_baseline` says nothing about the correctness of the
  replacement. Coverage comes from unit tests and `validate_hm7`, not from the
  baseline. Named here so G2 prices it.
- **R3 — the fixture-dependent test layer cannot run in a worktree.** It *skips*
  rather than fails, and this is a tree-shape change — the case `AGENTS.md`
  records as surviving every gate a branch can run. Implementation gating must
  happen in the primary checkout.
- **R4 — a typo in the `reporting:` section name is silent.** The config root has
  no section whitelist, so `reportng:` yields the default surface with nothing
  reported. Mitigated only inside the section (closed key sets, D13). Symptom is
  visible rather than dangerous: the caption renders as the default annual one.
- **R5 — the surface declaration is outside configuration identity.** Intended
  (D8), but it means `run_metadata.json` does not witness which surface a figure
  was drawn under. The config snapshot does record it, one level less directly.
  If publication provenance later needs a tighter tie, D6's export-on-demand is
  the place, not the digest.
- **R6 — three notebooks reference `stress_test_design`** and must be re-rendered
  after implementation (`t2608132100`).
- **R7 — S8's ordering is a real dependency, not a preference.** R12's
  `member_hash` is defined over the annual collapse this design deletes; landing
  R12 first would spend the member-identity work twice.

## 8. Migration plan

**One commit for the shape change**, per C5 — every live reference updated
together, because a stale path in a document someone reads to do their job is a
defect. Sequenced as:

- **Step 0 (prerequisite, not a commit): re-record the baseline** from
  `snake_config_baseline.yml` in the primary checkout, `--notemp` on WF1, no other
  session live (R1). This is the "before"; §9 V4 states how it is compared
  across the column change, and **step 7 re-records the "after"** — without that
  closing step the stored reference keeps seven columns and
  `check_baseline.py check` stays permanently red.
- **Step 1 — the artifact.** `prepare_cst_parameters.py` (one output, percent,
  no `st_0` row, D7's precision discipline, the corrected C23 comment), rule 3.09,
  `DESIGN_COLUMNS` -> `LOOKUP_COLUMNS`.
- **Step 2 — the consumer of the artifact.** Rule 3.12's input and the new
  positional argument; `impose_climate_change.R`'s filter and percent->factor
  conversion.
- **Step 3 — the reduction.** `export_wflow_results.py` loses `perturbation_axes`
  and `annual_perturbation` from its public surface, the `axes` map, the design
  read and the extra-axis refusal; gains the run-coverage check (D22).
  `INDICATOR_COLUMNS` in `shared/indicator_tables.py` loses two entries and
  `DESIGN_AXES` is **deleted** (its only consumer is the refusal being retired);
  rule 3.16's `input:` shrinks.
- **Step 4 — the library.** `shared/surface_axes.py` and the parse-time call in
  `run_stress_test.smk`.
- **Step 5 — contracts and inventory.** `validate_hm7` in
  `shared/interchange_contracts.py`; the HM-7 section (§5.7);
  `dev/scripts/semantic_tree_diff.py`; `dev/reference/workflows/rule-index.md`.
  `_PERTURBATION_AXIS` is **deleted**: verified this run that it is already
  vestigial — defined at `interchange_contracts.py:741` and referenced nowhere
  but the stale docstring at line 1141, since `validate_hm_gauge_column_identity`
  check 3 compares the `location` value SET post-CR-2 rather than subtracting
  axis columns from a wide header. Fix that docstring in the same commit; the
  relational validator's *logic* needs no change, which is what §5.7's
  non-interaction claim rests on.
- **Step 5b — sweep the test suite for the old roots.** `AGENTS.md` records that
  a task moving the project tree must do this, and that R9 — the same migration
  class — left 22 failures, three of them behind an `os.path.exists` guard that
  turned a wrong path into a **silent skip**. The sweep is
  `rg -n "_work/|stress_test_design" tests/` (the trailing slash matters:
  bare `_work` false-positives on `_workflow`). It currently returns:

  | file | what it holds |
  |---|---|
  | `tests/test_prepare_cst_parameters.py` (3 sites) | writes/reads a `stress_test_design.csv` under `tmp_path` |
  | `tests/test_export_wflow_results.py` (3 sites) | a design-table fixture "as 3.09 writes it", plus a re-read |
  | `tests/test_interchange_contracts.py` (3 sites) | a design fixture, a `validate_hm7(design=…)` call, and the real-fixture path `<exp>/config/stress_test_design.csv` |
  | `tests/test_project_tree_inventory.py` (1 site) | asserts `…/climate/weathergenr/_work/st_4.csv` classifies as declared — it must flip to reporting after §8's prefix narrowing |

  Re-run the sweep after the edits; a non-empty result is the failure. The
  fixture-dependent layer in `test_interchange_contracts.py` uses the *named*
  guard form (`_FIXTURE_ABSENT` + `skipif`), so its absence is reported rather
  than silent — but it still **cannot run in a worktree** (R3), so this step is
  gated in the primary checkout.
- **Step 6 — docs and seeds.** `config/templates/snake_config.template.yml`
  (a commented `reporting:` block), `test_case/snake_config_*.yml` if a seed
  declares a surface, `dev/reference/indicator-glossary.md`, and the
  `read_indicator_table` docstring in `dev/scripts/check_baseline.py`, which
  names `temp_change`/`precip_change` as its reason for string-parsing every
  non-`value` column and goes stale in this commit.
- **Step 7 — re-record the baseline "after".** Same procedure as step 0, once the
  shape change has landed and V4 has been evaluated. The stored reference
  otherwise still carries seven columns and every later `check_baseline.py check`
  fails structurally on a difference that is no longer a defect.

**Tree-shape changes**, all in step 5:

| what moves | from | to |
|---|---|---|
| the member grid | `<exp>/climate/weathergenr/_work/st_<m>.csv` (ST_NUM files) | `<exp>/config/stress_test_lookup.csv` (one file, 12 x ST_NUM rows) |
| the design table | `<exp>/config/stress_test_design.csv` | absorbed; deleted |
| the directory | `<exp>/climate/weathergenr/_work/` | gone |

Machinery that must move with it, checked file by file:

- **`dev/scripts/semantic_tree_diff.py`** — two edits. Rename the enumerated leaf
  `config/stress_test_design.csv` -> `config/stress_test_lookup.csv` (measured:
  without it the new artifact classifies UNMAPPED on every run). And **narrow the
  `climate/weathergenr/` whole-directory prefix to `config/`, `output/` and
  `plots/`** — the fixture holds exactly those three plus `_work/`, so the
  narrowing is exact, and it is what makes a leftover `_work/` report as
  undeclared instead of riding the prefix (§9, P3). `tests/test_project_tree_inventory.py`
  pins these rows and moves with them.
- **`dev/scripts/scaffold_project_tree.py`** — no change: it creates no `_work/`
  (verified). Its `scaffold_extras.yml` mentions `_work/` only in a retired-entry
  comment about rule 3.05, which is historical record and stays.
- **`dev/scripts/cross_workflow_inputs.py`** — no change: `LEAVES` is the WF1
  leaf set (`LEAF_WF1_SNAPSHOT`, `LEAF_MODEL_TOML`, `LEAF_MODEL_READY`) and this
  design adds no cross-workflow input. `tests/test_cross_workflow_inputs.py`
  proves minimality against the real DAG and is the falsifier.
- **`dev/scripts/check_baseline.py`** — no target row to rename (`TARGETS` carries
  `{exp_dir}/results/q_indicators.csv` and the config snapshot for WF3; the design
  table was deliberately never a target). The `read_indicator_table` docstring is
  the live reference that must change.

**The `naming.md` §7 record**, drop-in. §7 mandates
`dev/<milestone>/migration_<topic>.md`; the milestone directory is **unresolved**
(§10 OQ-1).

> # Migration — the stress-test lookup
>
> §7 record for the WF3 lookup redesign. Two §7 events in one commit.
>
> **Event 1 — a `rule all` output filename.**
>
> | old | new |
> |---|---|
> | `<exp>/config/stress_test_design.csv` | `<exp>/config/stress_test_lookup.csv` |
>
> Not a pure rename: the shape moves from one row per member to twelve rows per
> member, absorbing `<exp>/climate/weathergenr/_work/st_<m>.csv`, which is
> deleted along with the `_work/` directory. `lookup` rather than `design`
> precisely to signal that the shape moved — "design" would have described the
> old artifact just as well and bought nothing for the migration.
>
> **Event 2 — column labels in `rule all` output tables.**
>
> | table | old header | new header |
> |---|---|---|
> | `<exp>/results/<token>_indicators.csv` | `metric, location, st_id, rlz_id, temp_change, precip_change, value` | `metric, location, st_id, rlz_id, value` |
> | the member grid | `month, temp_mean, precip_mean, precip_variance` (multiplier) | `st_id, month, temp_change, precip_change, precip_variance_change` (percent) |
>
> The axis columns are removed rather than renamed: they are derived at reporting
> time from the lookup (`shared/surface_axes.py`; specification in HM-7).
>
> **Machinery updated in the same commit:** `prepare_cst_parameters.py`,
> `export_wflow_results.py`, `shared/indicator_tables.py`,
> `shared/interchange_contracts.py` (`validate_hm7`, `_PERTURBATION_AXIS`),
> `weathergen/impose_climate_change.R`, `run_stress_test.smk` (rules 3.09 / 3.12 /
> 3.16 and `WF3_TARGETS`), `dev/scripts/semantic_tree_diff.py`,
> `dev/scripts/check_baseline.py` (docstring), `dev/reference/contracts/hydrological-model-seam.md`,
> `dev/reference/workflows/rule-index.md`, `config/templates/snake_config.template.yml`.
>
> **For an existing project tree:** delete `<exp>/climate/weathergenr/_work/` and
> `<exp>/config/stress_test_design.csv`, then re-run WF3. `pixi run tree-check`
> reports both as undeclared until they are removed. No user action is needed
> beyond that, so no `docs/migration-*.md` guide is published.
>
> **Gate evidence:** `pixi run test-full`, `pixi run tree-check`, and a
> `check_baseline.py check` against a baseline re-recorded **before** the first
> implementation commit.

## 9. Validation plan

### Claim -> falsifier

| # | claim | falsifier | tier |
|---|---|---|---|
| V1 | The lookup is the only parameter artifact | any rule declaring an `st_<m>.csv` or a design table; `--dry-run` job list | `test_cli.py`, `--dry-run` |
| V2 | The DAG shape is unchanged | 3.12 job count != `RLZ_NUM x ST_NUM`, or 3.14 != `RLZ_NUM x (ST_NUM+1)` | `--dry-run` on the rapid config |
| V3 | `st_num >= 1` is still enforced on 3.12 | removing the constraint does **not** produce `CyclicGraphException` | probe P1 (below) |
| V4 | Indicator values do not move | the **column-aligned** comparison below reports any failing group | one-off, §9 "V4 procedure" |
| V5 | A uniform design's axis equals today's | derived axis != `annual_perturbation` output on a flat vector | unit test |
| V6 | A JFM design reports the imposed value | derived axis returns +7.6% for a +30% JFM member | unit test |
| V7 | Heterogeneous varying months are refused at the axis | `axis_values` returns a value instead of `HeterogeneousAxisError` | unit test |
| V8 | A held month in `M` is refused | `axis_values` returns instead of `HeldMonthInAxisError` | unit test |
| V9 | The axis is evenly spaced | `NonRectilinearAxisError` not raised on a deliberately non-affine collapse | unit test |
| V10 | A degenerate axis annotates rather than raises | a temp-only design failing on its precip axis | unit test |
| V11 | Captions match the six cases | rendered caption != the §5.5 table | unit test |
| V12 | A relabel does not trip the freeze | `_frozen_differences` non-empty after a `reporting:` edit | probe P2 (below) |
| V13 | Every lookup member appears in every table, and `st_0` appears in the tables but not the lookup | `validate_hm7` green while a member is missing | `test_interchange_contracts.py` |
| V14 | Stale `_work/` and `stress_test_design.csv` report | either classifies IDENTITY | `tree-check`, probe P3 |

### V4 procedure — how to compare values across a column change

V4 is D7's headline consequence and it needs its own procedure, because
`compare_indicator_table` **cannot** deliver it directly: its structural checks
run first and any hit is a structural FAIL that never reaches the numeric
comparison, and `_indicator_key_columns` treats **every non-`value` column** as
part of the row key. Dropping two columns therefore changes both the column set
and the key set, and a raw before/after comparison fails structurally with no
numeric verdict at all.

The executable form, run once at step 7:

1. Take the step-0 reference table and **drop `temp_change` and `precip_change`
   from the stored copy**, leaving `metric, location, st_id, rlz_id, value`.
2. Confirm `(metric, location, st_id, rlz_id)` is unique per row in both tables —
   otherwise the duplicate-key structural check fires, and a non-unique key would
   itself be a finding. It is expected to hold: the dropped columns are a function
   of `st_id`, so they added no discriminating power.
3. Run `compare_indicator_table(ref_trimmed, current)`. Pass is zero failing
   groups, which is the claim; a failing group means the forcing moved and D7's
   float32 round-trip discipline was not preserved.

The trimmed copy is a **one-off comparison input, not a stored artifact** — step
7's re-record replaces the reference outright.

### Gates

| gate | verdict |
|---|---|
| `pytest tests/test_prepare_cst_parameters.py`, `test_export_wflow_results.py`, `test_interchange_contracts.py`, `test_stress_test_grid.py` | runnable — the narrow tier, and these own the changed surfaces |
| `pytest tests/test_cli.py` | **required** — a rule's declared input changes, and it is the gate for D13's parse-time refusal |
| `pixi run test-fast` at the merge; `pixi run test-full` before the push | **`test-full` required**: this touches a Snakefile, a `script:` signature *and* `shared/` |
| `pixi run tree-check` | runnable, **needs the `semantic_tree_diff.py` change in the same commit** |
| `check_baseline.py check` | **needs a pre-change re-record first** (R1) and a **post-change re-record** to become usable again (§8 step 7). Between them it fails structurally on the column set; that failure is the design working, and the numeric question is answered by the V4 procedure instead |
| `validate_hm7` | part of the deliverable, not an independent check |
| the fixture-dependent layer | cannot run in a worktree — gate in the primary checkout (R3) |

### Framework-feasibility probes — results

All three were **executed** this run; none is settled by prose.

**P1 — does collapsing twelve declared outputs into one change rule 3.12's
fan-out or its `wildcard_constraints`?** **No.** Synthetic Snakefile (Snakemake
9.6.2, the pinned version) reproducing 3.09/3.11/3.12/3.16 with `RLZ_NUM = 2`,
`ST_NUM = 4`, the rule-3.12 shape (two wildcards, a constant `lookup.csv` input,
and `member_index_regex(1) = '[1-9]'` verbatim):

| variant | result |
|---|---|
| constant lookup input, constraint present | plans; `perturb` = **8 jobs** (unchanged); `st_0` routed to the generator |
| same at width 2 (`(?:[1-9][0-9]{1}\|0{1}[1-9])`) | identical, 8 jobs |
| constraint removed, generator present | still plans — so on 9.6.2 the constraint is not the *only* thing routing `st_0` in this shape |
| **generator removed, constraint present** | `MissingInputException in rule perturb` — the rule is barred from `st_0` |
| **generator removed, constraint removed** | **`CyclicGraphException in rule perturb — Cyclic dependency on rule perturb`** |

The last two rows are the decisive pair: the constraint remains load-bearing
after the input stops carrying the member wildcard. Keep it (D21).

**P2 — does the axis-derivation consumer need a rule, and does it re-fire
correctly on an axis-declaration edit?** **No rule is needed** (D14), so the
re-fire question is answered by placement instead. Measured against
`test_case/snake_config_rapid.yml` with the repo's own
`effective_config_digest` / `_frozen_differences`:

```
CONFIG_PROJECTION = ('project', 'shared', 'workflows.analyze_projections',
                     'workflows.build_model', 'workflows.run_stress_test')
digest unchanged by a TOP-LEVEL `reporting:` edit : True
digest unchanged by a run_stress_test.surfaces edit: False
freeze differences, top-level home : []
freeze differences, inside WF3     : ['surfaces']
```

So the top-level home neither re-fires the record rules nor trips the experiment
freeze, and the `run_stress_test` home does both. The `ancient()`/no-`params:`
trap (E5) is not inherited, because no rule reads the declaration.

**P3 — does removing `_work/` leave a declared-but-unwritten directory anywhere
in the scaffold or inventory?** **No declared entry is orphaned — and that is the
problem.** `build_project_tree_rules` carries exactly one weathergenr row and it
is the whole-directory prefix `experiments/<e>/climate/weathergenr/`, so `_work/`
has no row to go stale. Classifying a synthetic path list:

| path | current inventory | proposed inventory |
|---|---|---|
| `…/weathergenr/_work/st_1.csv` | **IDENTITY** (silently accepted) | **UNMAPPED** |
| `…/weathergenr/{config,output,plots}/…` | IDENTITY | IDENTITY |
| `…/config/stress_test_design.csv` | IDENTITY | **UNMAPPED** |
| `…/config/stress_test_lookup.csv` | **UNMAPPED** | IDENTITY |

Hence §8's two inventory edits: the leaf rename is **mandatory** (without it the
new artifact reports undeclared on every future run), and narrowing the prefix to
the three surviving subdirectories is what makes `tree-check` the migration's own
gate. `scaffold_project_tree.py` and `cross_workflow_inputs.py` need no change.

### Evidence settled this run

- **E9 — `csthelpers::plot_climate_surface` is parameterized: SETTLED, verified.**
  `x_var`/`y_var` default to `NULL` and `stop()` when unset, are validated with
  `if (!x_var %in% names(data)) stop(...)` and a numeric check, and breaks come
  from `sort(unique(data[[x_var]]))`. Stronger than the seed asserted:
  `precip_change` occurs **nowhere in `csthelpers/R/`** — only in
  `inst/examples/usage_climate_surface.R` and two vendored snapshot CSVs under
  `data/`. Dropping the axis columns is not a live-integration break. (S4 rules
  it out of scope either way; recorded so a reviewer does not inherit it.)
- **E10 — the three interpretable designs are expressible today: SETTLED,
  executed.** `np.linspace(min_vector, max_vector, step_num + 1, axis=1)` takes
  arbitrary 12-vectors, so cases 1–3 differ only in the vectors' contents; run at
  `step_num: 2` each produces a `(12, 3)` member matrix with no design-side code
  change. The residual, stated because "expressible" and "interpretable" are
  different: interpretability is a property of the vectors, and it is what D16
  enforces. Executed captions, with the axis levels each yields:

  | case | varying | homogeneous | caption | axis levels |
  |---|---|---|---|---|
  | 1 uniform `[0.7]x12 -> [1.3]x12` | all 12 | yes | `mean change over the year` | -30, 0, +30 |
  | 2 JFM, rest `1.0` | 1–3 | yes | `mean change over JFM; Apr–Dec unchanged` | -30, 0, +30 |
  | 3 JFM, rest `0.8` | 1–3 | yes | `mean change over JFM; Apr–Dec held at -20%` | -30, 0, +30 |
  | 3b JFM, rest `0.8`/`0.9` | 1–3 | yes | `mean change over JFM; Apr–Sep held at -20%; Oct–Dec held at -10%` | -30, 0, +30 |
  | 4 nothing varies | — | — | `unchanged` (degenerate) | — |
  | JJA seasonal | 6–8 | yes | `mean change over JJA; Sep–May unchanged` | -30, 0, +30 |
  | heterogeneous Jan ±30 / Feb 0→+50 | 1–2 | **no** | refused (D16) | -15, 12.5, 40 — **evenly spaced** |

  The last row is the empirical argument for D16 and D17 being **independent**
  checks: a heterogeneous design produces a perfectly rectilinear axis whose
  values (-15, +12.5, +40) match no month's imposed change. Even spacing does not
  imply interpretability.

**Evidence still carried as verified-elsewhere** (E1–E8, `intake.md`): the cache
mechanics at `prepare_cst_parameters.py:175-189`; the +7.6% arithmetic; WF2's
percent units; rule 3.09 declaring all member files as one job's outputs; rule
3.09's `ancient()`/no-`params:` deafness; weathergenr 1.2.0's non-identity at
unit factors; the measured `st_0` -> identity-member indicator differences
(magnitudes provisional — the fixture predates the 1.2.0 rename); and R12's
`member_hash` indexing members by the annual collapse this design deletes.

## 10. Open questions

- **OQ-1 — where the migration note lands.** `naming.md` §7 mandates
  `dev/<milestone>/migration_<topic>.md`, but this work lands **before** R12 (S8)
  and carries no milestone directory of its own; the board item is
  `t2608152230`. The note's full content is in §8, drop-in ready; only its path
  is unresolved. Options: create `dev/milestones/r12/` early and file it there;
  file it under a new directory named for this change; or extend §7's grammar to
  admit a board-item-scoped record. **Owner ruling needed** — this is a
  self-containment gap in the run's inputs, not a design choice this document
  should make alone.
- **OQ-2 — Q6, the projection overlay.** Deferred deliberately. The constraint is
  pinned (S10) and the mechanism is cheap because WF2 already emits monthly
  factors in percent, so the same collapse runs over both tables with no unit
  conversion. `reporting:` is where its declaration will live.
- **OQ-3 — should `st_0`'s annotated reference carry its health warning in the
  data?** S6 stands, with the caveat that `st_0` and the surface differ by a
  **processing step** and not only by a perturbation (five of eleven `q` metrics
  move by a factor). `join_axes` returns the baseline rows separately, which
  makes the distinction visible; whether the warning text belongs in the library,
  the caption, or `t2608151154`'s own resolution is not settled here.
- **OQ-4 — a `basin`-scalar or a third parameter later.** Both are refused today
  (Q11, C28) and neither is re-opened. Noted because D5's closed vocabulary is
  where the first one would land.

## 11. Revision log

- **v1 (2026-08-15)** — first authored version. Seeded from
  `dev/working/2026-08-15_wf3-scenario-generation-trace/stress-test-design-and-surface-axes.md`
  (six owner rulings + two same-day revisions, carried as S1–S11) and restructured
  to this repo's design house style, which the seed note — a design *conversation
  record* — did not have. New normative content closing the six intake gaps:
  the lookup schema and its `st_0` omission (D1–D7), the `reporting:` config
  schema and its tier story (D8–D13), the library consumer (D14–D15), the two
  enforcement checks (D16–D17), the caption algorithm including the two
  previously-uncovered cases (D18–D19), the rule-level changes (D20–D24), HM-7's
  replacement text (§5.7) and the migration record (§8). P1, P2 and P3 executed;
  E9 and E10 settled.
