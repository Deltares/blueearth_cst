# WF3 — stress-test lookup and derived response-surface axes: Design (v4, DRAFT)

> **Status: DRAFT (v4)** — stage 6a (**arbitration revision**) of the
> `design-review-loop` run `wf3-stress-test-lookup`. Not accepted; G1 (framing) is
> **approved**, with a **G1-return** ruling two forks after the internal panel; G2
> (design) is `pending` (`status.md`).
> **Revision basis:** cumulative. v2 answered the 26 internal-panel findings
> (both blocking — `architecture-1`, `repo-fit-1` — closed by §5.7); v3 answered
> external round 1's six (`external-review-r1.md`), whose two blocking findings
> were the caption's false annual label (`ext1-1`, closed in §5.5/§5.8 by **D31**)
> and the notebook contradicting the settled no-caller ruling (`ext1-2`, closed in
> §8 step 6); **v4 answers external round 2's three** (`external-review-r2.md`) —
> the round-2 cap, so there is no external verdict on this version and the three
> fixes land under the **owner's arbitration of 2026-08-15** (`status.md`
> § arbitration), which accepted all three and ruled the *shape* of the third.
> All 35 are dispositioned one row each in `ledger.md`, which remains
> the run record for per-finding argument.
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
> **Size budget for the normative body:** **raised at v4 from 2,600 to 3,000
> lines** (v2 raised it from 1,250 to 2,150 and landed at 2,133; v3 raised it to
> 2,600 and landed at 2,615, 15 over and disclosed; v4 is **2,998**). Raised rather
> than exceeded silently, for the same reason as at v2 and v3. Where v4's
> **+383 net lines** go — measured after the fact, not estimated before it:
>
> - **~150 lines** — `ext2-3`, the only fix that adds a rule with no existing home.
>   **D35** (the admitted multiplier domain, its mechanism table, its enforcement
>   site), the WG-2 domain bullet, **E16**'s dense-sweep evidence, **V23**, and the
>   widening of **V20**. E16 is the larger half of that, and deliberately so: the
>   finding is about a claim standing on a sweep, so the answer is an argument
>   plus directed confirmation, not a bigger sweep.
> - **~95 lines** — `ext2-1`, blocking. D28's rule 2 becomes three ordered checks
>   with set equality, mirrored into §5.8's report-time join semantics, plus the
>   V18 widening.
> - **~70 lines** — `ext2-2`, blocking. §5.2's cross-axis distinctness rule and
>   what refusing it costs, D13's parser clause, D33's keying note, **V22**. Each
>   fix also touches §8's machinery list, since a rule stated and never invoked is
>   the gap that cost this run a round.
> - **~70 lines** — the v4 revision-log entry (§11), where the mechanism table a
>   later reader re-reads lives and where the arbitration authority is recorded,
>   plus this accounting block.
>
> Where v3's
> **+482 net lines** go — measured after the fact, not estimated before it:
>
> - **~150 lines** — `ext1-1`, blocking. **D31**, the caption's one clause builder
>   and two new cases in §5.5, **mirrored into §5.8** — the fix is normative text
>   in two documents, because under Fork B the contract is what the real consumers
>   execute. E15 is its measurement.
> - **~90 lines** — `ext1-3`. **D32** (the degenerate scalar) and **D33** (the
>   result objects and the inferred key width), each landing in §5.3 as an API and
>   in §5.8 as semantics an R re-implementer can act on.
> - **~90 lines** — `ext1-2`, blocking. The notebook realignment reaches six
>   places (§5.3 reason 1, D15, 6.9, R6, R9, §8 step 6) because v2's contradiction
>   reached six places.
> - **~55 lines** — `ext1-5`. **D34** and the migration, test-file and gate rows
>   that make V17's negative executions possible at all.
> - **~55 lines** — `ext1-4`. The claim narrowing in D25, §7-2, the V4 diagnosis
>   rule and **R13** — which deletes an overstated bound as well as adding.
> - **~65 lines** — the v3 revision-log entry (§11), which is where the mechanism
>   table a later round re-reads lives.
>
> The original budget was set against this repo's accepted designs
> (p32b 1,066; p33 1,192; p32a 1,372) for a deliverable carrying **one** contract
> replacement. Where the ~880 added lines went:
>
> - **~210 lines** — the second contract. The G1-return ruling on Fork A makes the
>   deliverable **two** drop-in sections (§5.7 WG-2, §5.8 HM-7) instead of one, and
>   Fork B adds a completeness obligation to both, so the tolerances, the
>   degenerate rule and the caption case table now appear in normative contract
>   text as well as in the design's own §5.5.
> - **~380 lines** — six new decisions (D25–D30), five new risks (R8–R12), six new
>   validation claims (V15–V20), a second probe (P2-b) and four new measurements
>   (E11–E14). All are normative or evidentiary; none is argument.
> - **~290 lines** — corrections in place, where a finding showed a claim false
>   rather than missing (D3, D7, D11, D14, §7-1, §5.8's scope clause, §8 step 5b).
>
> **Per-finding argument is NOT here.** It relocates to this run's review record —
> which, for an author spawn barred from every file but two, is **`ledger.md`**,
> whose rationale column carries it, cited by finding id. That is the budget's own
> release valve, used as intended.
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
  updated and a `naming.md` §7 record (§8), plus **two** contract replacements
  that drop in as written: **WG-2** into
  `dev/reference/contracts/weather-generator-seam.md` (§5.7, owning the lookup's
  schema) and **HM-7** into
  `dev/reference/contracts/hydrological-model-seam.md` (§5.8, owning the axis
  derivation). v1 wrote only the second, which is what both blocking findings
  caught; the split follows the G1-return ruling on Fork A.

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
  error changes the science" — is **partly right, and v1 dismissed it too
  cleanly.** v1 argued the incident behind it (`float32(0.7)` ->
  `-30.000001%`) was a float32-vs-float64 CSV round-trip problem rather than a
  unit-choice one, "and §5.1 D7 resolves it directly". The first clause holds;
  the second does not. Storing the percent means the multiplier the generator
  receives is **reconstructed** rather than read, and that reconstruction is not
  exactly invertible for every value — measured, and quantified in **D25**, which
  replaces D7's exactness claim with a stated bound. S1 the *ruling* stands on
  cross-artifact consistency and the overlay, which is what it was ruled on; what
  changes is that the design now prices the residual instead of denying it.
  Consequence worth knowing: no-change becomes `0.0` rather than `1.0`.
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
scratch.

**Its normative home is WG-2, not this section and not HM-7.** Ruled at the
G1-return (Fork A): the lookup is the **Python → R parameter handoff** — rule
3.09 writes it, `impose_climate_change.R` reads it — which is the
weather-generator seam by definition, and WG-2 is the contract that already pins
the artifact it replaces. §5.7 carries WG-2's replacement text and is the
authority for every clause below; HM-7 (§5.8) **references** it rather than
restating it, so one artifact is described once. What follows is a summary for
readers of this document, not a second definition — where the two disagree, §5.7
wins.

Header, in this order:

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
ruling behind it.

**One conversion site per direction, covering BOTH multiplier columns.** v1 said
"two conversion sites, and only two" and then tabulated `precip_change` alone,
which asserted completeness while omitting a column. `precip_variance` is also a
multiplier the generator consumes — `impose_climate_change.R:70` passes
`precip_var_factor = cst_data$precip_variance` — and S1 makes its lookup column a
**percent**. Implemented as v1 wrote it, the R would hand
`apply_climate_perturbations` a variance factor of `precip_variance_change`,
which on **every shipped config** (variance `min = max = 1.0`, i.e. `0.0` percent)
is a variance factor of **zero** rather than the identity `1.0`. The rule is
therefore stated over the percent columns, not per column, so a future column
inherits it:

| direction | where | rule |
|---|---|---|
| forward | rule 3.09 writing the lookup | for each multiplier column `c`: `<c>_change = c * 100 - 100` |
| inverse | `impose_climate_change.R` applying it | for each percent column: `precip_mean = 1 + precip_change / 100`; `precip_variance = 1 + precip_variance_change / 100` |

`temp_change` is additive °C and crosses unconverted in both directions.

The multiplier survives only as the generator's operation form. The forward
formula is written as `f * 100 - 100` and **not** as `(f - 1) * 100`: measured,
`1.3*100-100 == 30.0` exactly while `(1.3-1.0)*100 == 30.000000000000004`. This
is the formula `export_wflow_results.perturbation_axes` already uses; it is
preserved, not invented. The **inverse** spelling is pinned by **D25**, which
also states what the round trip does and does not guarantee — v1's claim that
"the inverse restores `1.3` exactly" is true of `1.3` and false in general.

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
indicator header no longer expresses any axis (§5.6 D14, §5.8).

**D6 — no `alias_of` column, and no per-surface `axes.csv`.** The first is
withdrawn with the alias (S7); the second is withdrawn by C3 — given the lookup,
an axis table stores something already fully determined. The one case for
materializing derived axis values, recorded so it stays a decision rather than an
oversight: archiving a *published* figure, where the exact plotted numbers should
sit beside it. That is publication provenance, served by an export-on-demand, not
by writing a file every run.

**D7 — value precision: the `float32` grid quantization is preserved.**
Today `prepare_cst_parameters` builds each member frame as `float32`, writes it,
and **re-reads it off disk** before deriving the design row, because a design row
computed from the in-memory `float32` records a perturbation nobody applied
(`float32(0.7) = 0.69999998807` -> `-30.000001%`, while the run imposes the
round-tripped `0.7`). With one artifact there is no second derivation and no disk
round trip to perform, so R11 P3's read-it-back hack disappears with the cache it
existed to serve.

What is preserved is the **quantization of the grid itself**: the member levels
are the `float32` shortest-repr values `float(str(np.float32(v)))` — exactly what
`to_csv` writes today — so the *grid the user asked for* is unchanged. What is
**not** preserved, and what v1 wrongly claimed was, is bit-identity of the
multiplier the generator receives. That is D25.

**D25 — the percent text's precision, the pinned inverse, and the residual
(replaces D7's exactness claim).** The generator no longer reads the multiplier;
it **reconstructs** it from the percent. Three normative rules and one measured
bound.

1. **The percent is written at `float64` shortest repr of the exact conversion of
   the `float32` level**, and **not** as a `float32` column. Spelled as two steps
   on a Python `float`, because the one-liner is easy to misread and this is the
   line the whole bound rests on:

   ```python
   level = float(str(np.float32(v)))   # the grid level, as D7 defines it
   text  = repr(level * 100 - 100)     # what rule 3.09 writes
   ```

   `repr` of a **Python float** is float64 shortest-repr; `repr` of a *numpy*
   scalar yields `np.float32(0.82)`, and applying it before the round-trip yields
   the raw binary value rather than the level. Both are wrong and both are one
   keystroke away. This is a deliberate departure from "just keep
   writing `float32`", and it is the single highest-leverage line in this section:
   re-quantizing the *percent* to `float32` degrades the reconstructed multiplier
   by roughly **eight orders of magnitude** (measured: worst relative
   |Δ| = **5.98e-08**, i.e. one `float32` ulp, against ~1e-16 at `float64` text).
   D7's whole purpose is fidelity of the multiplier, so the column that must stay
   coarse is the *grid*, and the column that must stay fine is the *text*.
2. **The inverse is `1 + p/100`, not `(100 + p)/100`.** `risk-1`'s suggested fix
   asks for the second spelling; measured, it is **worse**. Over 200,000 random
   `float32` multipliers in [0.5, 1.6]: `1 + p/100` fails to reproduce the level
   in 19.9% of cases, `(100 + p)/100` in 32.9%. Allowing a search over
   neighbouring `float64` percents, the number with **no** exact solution at all
   is 1,155/50,000 for the first spelling and 6,778/50,000 for the second. The
   first spelling is also what a reader of the percent will write unprompted.
3. **The round trip is not exact, and cannot be made exact.** Both reviewers
   offered "specify a search over candidate texts, verified at write time" as the
   way to keep an exactness claim. Measured, that fix is **unattainable**:
   **1,155 of 50,000** `float32` multipliers admit no `float64` percent whatever
   that reconstructs them under `1 + p/100`, so no choice of written text can
   deliver exactness for every grid. Option (a) — state the bound — is therefore
   forced, not chosen.

**The bound, stated normatively — with the precondition D35 enforces.** Every
admitted multiplier is **≥ 0.5**, refused at parse time otherwise (**D35**). Given
that, the multiplier the generator receives after migration differs from the
pre-migration multiplier by **at most one `float64` ulp of the level** (relative
~2.2e-16), and is bit-identical for the majority of levels including every level in
every shipped config. The bound is stated **unconditionally over the admitted
domain**, which is why the domain is a refusal and not a caveat: a bound with an
escape clause is one an R or JavaScript re-implementer must re-derive, and this is
a cross-language contract. Worked counter-examples, measured over `np.linspace`:

| grid | level | percent written | reconstructed |
|---|---|---|---|
| 0.7–1.3, `step_num: 5` | `0.82` | `-18.0` | `0.8200000000000001` |
| 0.6–1.4, `step_num: 3` | `1.1333333` | `13.33333` | `1.1333332999999999` |
| 0.7–1.3, `step_num` 1, 2, 4, 6, 8 | — | — | **exact, every level** |

The shipped configs (`precip` `step_num` 1 and 2) are in the exact set, which is
why §9's V4 procedure is a valid gate **on those configs and only on those** — a
`step_num: 5` project is not observable through it. §9 adds a unit test over a
non-round grid for exactly that reason, and V4's failure interpretation is
corrected. The corrected rule is **sharper** than v2's, because V4 runs on
`snake_config_baseline.yml` alone: on that config the reconstructed multiplier is
**bit-identical**, so D25's arithmetic cannot explain a failing group *at all*,
and the diagnosis is a lost quantization discipline or a real behaviour change.
v2 wrote the general rule ("moved by more than one ulp"), which silently assumed
that a one-ulp forcing difference bounds the output difference — the conflation
`ext1-4` names. On a non-exact grid V4 is not run and no such inference is
offered.

**The bound is on the MULTIPLIER, and on nothing downstream of it.** `ext1-4` is
right that v2 slid from one to the other, and the slide is not cosmetic: between
the multiplier and an indicator sit `apply_climate_perturbations`' quantile
mapping, its wet-day occurrence and intensity thresholds, the caps and floors,
and then a distributed hydrological simulation with its own thresholds and
storage states. None of those is Lipschitz in the forcing parameter with a
constant this design has measured, and **this run's own history is the
precedent**: E6 measured that a perturbation at *exactly* unit factors — no
parameter difference at all — moves single-day and 7-day precipitation extremes
by tens of percent, and E7 that five of eleven `q` indicators then move by a
factor. A tiny parameter difference is not the same case, but it is the same
mechanism, and nothing here bounds it.

So the normative claim is: **the reconstructed multiplier is within one `float64`
ulp of the pre-migration level (V16, V20), and is bit-identical on every level of
every shipped config.** Indicator-level equality is claimed **only** for the
exactly-reconstructing configs, where the forcing is bit-identical and the
question does not arise. For a project on a non-exact grid this design states no
output bound at all — see §7-2 for the claim as restated and **R13** for the
residual.

One further column inherits the same residual: `precip_variance` converts on a
path no shipped config exercises (variance is flat at 1.0 everywhere), which is
why D3 states the conversion as a rule over columns rather than per column.

**D35 — the admissible multiplier domain is `multiplier ≥ 0.5`, refused at parse
time.** `ext2-3` is right that v3 stated D25's one-ulp bound as an unqualified
normative claim while its evidence covered `[0.5, 1.6]` and the design admitted any
positive multiplier whatever. The claim is false outside the measured region — at
the `float32`-shortest level `0.013596006` the specified conversion writes
`-98.6403994` and reconstructs `0.013596005999999883`, **68 ulps** against a bound
of one. Since WG-2 pins that bound as a cross-language contract and V16/V20 use it
as an acceptance threshold, a low-but-positive multiplier could follow every
specified formula exactly and still fail the contract and the migration gate.

**The domain, and what it applies to.** Every element of the `min` and `max`
12-vectors of **both** percent-converted config keys —
`stress_test.precip.mean` and `stress_test.precip.variance` — must be `≥ 0.5`. The
grid levels are `np.linspace` between those endpoints and are therefore monotone
between them, so validating the endpoints validates every level: no separate
per-level check is needed. `stress_test.temp` carries **no** domain — it is
additive °C and crosses unconverted in both directions (D3), so no reconstruction
happens and none can be out of bound.

**Both columns, not just `mean`, and the cost is recorded rather than discovered.**
D3's whole correction at v2 (`architecture-2` / `risk-7`) was that the conversion
is a rule over the **percent columns**, not a formula for one of them; a domain
covering `precip.mean` alone would re-create exactly that defect one layer up. So a
variance multiplier below 0.5 is refused too. No shipped config declares one —
all four seeds and `config/templates/snake_config.template.yml` carry
`variance: min = max = 1.0` and `mean: 0.7 → 1.3`, verified this revision — so the
refusal costs nothing today and is stated as a decision rather than left to be
found by whoever first writes one.

**Why 0.5, and why nothing above.** The error is **not** empirical noise; it has a
mechanism, and the mechanism is what makes the threshold pinnable rather than
sweep-fitted. The reconstruction's absolute error is set by the rounding of the
*percent*, at scale `ulp(|100(level − 1)|)`, which the division by 100 then carries
into level space. Measured against `ulp(level)`:

| region | `ulp(level)` | `|percent|` reaches | error in ulps of the level |
|---|---|---|---|
| `level ≥ 1` | grows with the level | ≤ `100 · level`, i.e. within 7 binades of it | bounded **forever** — the two scales grow together |
| `0.5 ≤ level < 1` | 2⁻⁵³ | ≤ 50, so `ulp(percent) ≤ 2⁻⁴⁷` | **≤ 1** |
| `0.36 ≲ level < 0.5` | 2⁻⁵⁴ | ≤ 64, `ulp(percent)` still 2⁻⁴⁷ | ≤ 1 — one binade of headroom below the floor |
| `0.25 ≤ level ≲ 0.36` | 2⁻⁵⁴ | crosses 64, `ulp(percent)` **jumps** to 2⁻⁴⁶ | **2** |
| `level → 0` | halves each binade | pinned near `ulp(100)` | diverges: 18 at 0.05, 72 at 0.015, 574 at 0.0016 |

So the bound fails **downward only**, at the first percent-binade crossing that
`ulp(level)` no longer keeps up with, and it fails nowhere upward: the percent
grows in step with the level, so the ratio stays bounded however large the
multiplier gets. E16 confirms the argument by dense `float32` sweeps across every
percent-binade crossing in the domain — not by random draws, which is the evidence
shape `ext2-3` faulted — and by random confirmation out to `1e6`.

**Hence a floor and no ceiling.** `0.5` is a `float64` binade boundary and is where
the argument above stops needing a case, with the first 2-ulp level a full binade
below it. An **upper** bound would be an invention: the measurement shows the one-ulp
bound holding to `1e6`, so capping the multiplier would refuse configurations that
are numerically fine and would tell a re-implementer reading WG-2 that the bound
fails above the cap, which is false. Whether a multiplier of 8 is a sensible stress
test is a question for the analyst, not for a numerical guard.

**Where it is enforced, and why parse time.** `MULTIPLIER_DOMAIN = (0.5, None)`
and `refuse_out_of_domain_multipliers(stress_test_cfg)` live in
`blueearth_cst/experiment/prepare_cst_parameters.py` — the module that already owns
the `stress_test:` vocabulary (`_KNOWN_AXES`) and performs the forward conversion,
so the domain and the conversion it protects cannot drift into different files.
`run_stress_test.smk` calls it **at parse time**, beside
`refuse_retired_experiment_keys(my_cfg)` (`:520`) and `parse_surfaces(config)`
(D13). The module is import-clean at parse time by its own construction — its
lines 11–17 insert the repo root on `sys.path` and state the intent — and
`run_stress_test.smk:14` already imports from `blueearth_cst.experiment` at parse
time, so this needs no new pattern.

Parse time rather than write time is the point: the refusal must land **before the
DAG is built**, so `--dry-run` fails and `pytest tests/test_cli.py` is its gate
(the same gate D13 names), and no member file, no realization and no wflow run is
produced under a config whose reconstruction the contract cannot bound. Note this
is a **second** call path, not a merge of two: `_KNOWN_AXES` refuses at *write*
time off `prep_cst_parameters`' own YAML read, and this one refuses at *parse* time
off `my_cfg`. Raises `MultiplierDomainError` naming the key, the offending months
and their values. **V23** is its falsifier and **V20** is extended to the domain's
lower boundary.

**What this does NOT do.** It bounds the reconstructed **multiplier** and nothing
downstream of it. **R13** stands exactly as written: on a non-exactly-reconstructing
grid nothing here bounds the *output* difference, and narrowing the admitted domain
does not change that — a one-ulp forcing difference inside `[0.5, ∞)` is still a
forcing difference with a quantile mapping and a hydrological model between it and
an indicator.

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

**The two axes must declare DIFFERENT variables, and a surface that repeats one is
refused at parse time** (`ext2-2`, amending D8). The rule is exactly

    {x.variable, y.variable} == {"temp", "precip"}

so **orientation reversal stays legal** — `x: {variable: precip}, y: {variable:
temp}` is an ordinary declaration and nothing here constrains which variable takes
which axis — while `x: {variable: temp}, y: {variable: temp}` is refused with the
surface `id` in the message.

The reason is that the rest of the design **cannot represent** the repeated case,
not that it is unwise. `SurfaceJoin.axes` is a `dict` keyed by variable (D33), so
one `AxisResult` would overwrite the other; and both axes name their derived column
through `AXIS_COLUMN[variable]` (D28), so both would target `temp_change`. An
implementation handed such a declaration must either discard an axis or return an
object that violates its own declared API — so the schema was admitting a
configuration no conforming implementation could serve. A per-axis closed enum with
no cross-axis rule is exactly how that gets missed: each field is individually
valid.

**What the refusal costs, stated so it is a decision rather than a side effect.**
The thing a user might have wanted from `x: temp[JFM], y: temp[JJA]` — two month
windows of the *same* variable on two axes — is not expressible, and this design
does not make it so. It would need `axes` keyed by something other than the
variable and a second derived column per variable, which is a shape change to D33's
result objects and to HM-7's `AXIS_COLUMN` naming. It is also **not a response
surface over this experiment**: D10's affinity argument means every member lies on
one line from `min` to `max`, so two temperature axes over one experiment are
affine images of each other and the "surface" would be a line embedded in a plane.
Refusing is correct here and remains correct until members vary seasonal pattern
independently — the second design dimension D10 already places out of scope.

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
shared guard artifact.

**How the choice is recorded, stated without the circularity v1 had.** v1 claimed
both that "a relabel needs no re-run" and that the choice is recorded because
"rule 3.02 byte-copies the config as run" — which cannot both hold for the same
edit. The honest version, verified in the tree rather than inferred:

- Rule 3.02 declares `config_snake = config_path` as a **plain** input
  (`run_stress_test.smk:604-605`), not `ancient()`. So **any** edit to the config
  file makes it newer than `<exp>/config/snake_config_run_stress_test.yml` and
  Snakemake re-runs 3.02 on the next invocation, by mtime. That is not a defect —
  it is precisely **how** the relabel gets recorded.
- `copy_config_files.py:222` is a `shutil.copyfile`, so the snapshot carries the
  `reporting:` section verbatim. S2's obligation is met.
- Therefore: **a relabel made without any WF3 invocation is unrecorded.** The
  recording costs one cheap re-invocation, not a re-run of the experiment — 3.02
  and the record rules re-fire; 3.16b does not (its `params:` carry only the two
  digests, `run_stress_test.smk:1146-1148`), and the freeze does not trip.

**And it has a gate cost the design must price.** `dev/scripts/check_baseline.py:326-329`
carries the WF3 config snapshot as a **`yaml` baseline target**, and
`fingerprint_yaml` (`:449-455`) `yaml.safe_load`s the **whole unprojected
document** and hashes a canonical JSON of it. Verified: a top-level `reporting:`
section is inside that hash. So on the baseline tree, editing `reporting:` and
re-running WF3 turns `check_baseline.py check` red — for an edit D8's tier story
says costs nothing.

This is scoped, not alarming: it is a **pre-existing property of that target**
(the snapshot is a byte copy and the fingerprint is unprojected, so *any* config
edit moves it), and the design's contribution is only that it makes a **caption**
edit a config edit. It is recorded as risk **R8** and it constrains §8 step 6:
a seed config that declares a surface arms it, so either the seeds ship
`reporting:` from the start — inside the same re-record §8 already sequences — or
they ship none at all and rely on `DEFAULT_SURFACE`. This design takes the second
option; §8 step 6 is amended accordingly.

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
is one surface, **`id: default`**, `x: {variable: temp}`, `y: {variable: precip}`,
both axes taking D11's derived month set and `statistic: mean`. It is a constant
of the toolbox in the same sense `_PERTURBATION_AXIS` is, and it needs no new
validator class for a structured value.

**The id is `default`, not `annual`.** D18's whole premise is that a typed label
drifts from the design it describes and a derived one cannot — and the surface
`id` is a typed label that D14 makes a consumer's handle on the frame. Since both
its axes take D11's *member-varying* month set, `annual` would be accurate only
for a uniform design: a JFM design's default surface would be identified as
`annual` while captioned "mean change over JFM", asserting exactly the collapse
D11 exists to stop being the default. `annual` is **reserved** for a surface a
user declares explicitly with `months: [1..12]` — which D16's subset rule then
correctly refuses on a seasonal design.

**D10 — `surfaces:` is a list of zero or more.** Q5 is closed by S2: the lookup
is a sufficient statistic, so N surfaces need nothing beyond N derivations, and a
list of one costs nothing more than a scalar. Refusing N would be an arbitrary
barrier given the mechanism. Absent or empty ⇒ `DEFAULT_SURFACE`.

The **limit worth stating so nobody rediscovers it as a bug**: within one
experiment every member lies on the line from `min` to `max`, so every affine
axis is an affine image of every other — **two surfaces from one experiment
differ in magnitude and label, not in shape or member ordering.** That claim is
about the **surface members only** and does **not** extend to the S10 projection
overlay: a GCM's monthly change factors are not homogeneous, so the same collapse
over a different `M` moves the GCM cloud non-affinely, and two GCMs at the same
annual mean can sit at opposite ends of a JFM axis. Declaring N surfaces
therefore places the same GCM cloud N different ways against what is, on the
stress side, one surface relabelled — and because D8 puts `reporting:` outside
run identity, that is a change in a **plausibility judgement** made by a config
edit that leaves no run-identity trace. §5.8 carries the caveat normatively and
OQ-2 carries what Q6 must decide about it; risk **R12** names it. That is not a
limitation to fix; it is what makes the seasonal case worth having. Reporting
"+30% over JFM" instead of "+7.6% annual" is the *same* surface, correctly
labelled. Genuinely different response *shapes* need members varying seasonal
pattern independently — a second design dimension, out of scope.

**D11 — `months` defaults to the member-varying set, derived from the lookup.**
A month is **varying** when its value differs across the surface members and
**held** otherwise. The default is the varying set; when nothing varies (**D19**'s
degenerate axis — v1 mis-cited D17, which is the rectilinearity postcondition) it
is all twelve.

**The classification threshold, fixed once and cited everywhere.** A month is
varying **iff `max - min > 0` over `st_id` in `1..ST_NUM`, exactly** — no
tolerance. v1 spelled it `> 0` here and `> tol` in §5.5 step 1 with `tol` defined
nowhere. Exact-zero is the right threshold and not merely the simpler one: the
values compared are `float64` reads of the same written text for every member
whose level is held, so a held month's twelve values are **bit-identical by
construction**, not merely close. A tolerance would buy nothing and would create a
band in which a month is neither varying nor held. `risk-6`'s hazard — a config
with `min: 0.8` / `max: 0.8000000001` classifying as varying and then tripping
D16's homogeneity refusal — is real and is the **correct** outcome: those are two
different perturbations, and the refusal names the homogeneous subsets so the user
sees what they actually declared. This threshold is normative in **§5.8** as part
of the axis-derivation contract, so an R re-implementer gets it too.

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

**D26 — the flat-vector short-circuit is normative, not an implementation
detail.** The statistic is defined as: *when the declared months' values are
exactly equal, return that value; otherwise return the month-length weighted
mean.* v1 omitted the short-circuit and asserted the opposite — "under D16's
homogeneity constraint the weighting is immaterial, every value being averaged is
equal" — which is the claim the incumbent code was written to refute. That
sentence is **deleted**. `export_wflow_results.annual_perturbation:122-124`
carries `if np.ptp(values) == 0: return float(values[0])`, documented as "flat
vectors short-circuit on exact equality rather than falling through the weighted
mean, which would round twelve identical values to something a unit in the last
place away from them".

This matters **more** under this design, not less, and the measurement inverts
which space is dangerous:

| space | values | flat-vector weighted mean != value |
|---|---|---|
| multiplier (today's path, being deleted) | 50,000 random `float32` in [0.5, 1.6] | **0** |
| **percent (the lookup's precip columns)** | 200,000 random in [-90, 200] | **97,628 (49%)** |
| **°C (the lookup's temp column)** | 100,000 random `float32` in [0, 6] | **48,294 (48%)** |

So the case v1 called immaterial is the one the unit change creates. Realistic
grids hit it: `0.6–1.4` at `step_num: 3` gives `-13.33333` -> `-13.333330000000002`,
and `0.5–1.5` at `step_num: 7` gives `35.71428` -> `35.71428000000002`. Under
D16, homogeneous is the **normal** path for every admissible axis, so without the
short-circuit the derived axis would be a ulp away from the imposed value on
roughly half of all non-round grids — and the repo already treats that ulp as
load-bearing (`tests/test_check_baseline_indicator.py:244` exists because `1.3`
and `1.3000000000000003` must not be conflated).

The short-circuit is therefore part of the contract text in §5.8, so an R or
JavaScript re-implementer inherits it, and §7 consequence 3 / §9 V5 keep their
**equality** form rather than being relaxed to a tolerance.

Month-length weighting is retained for compatibility with WF2 and the S10 overlay
— it is what the non-flat branch must be for the overlay to compare like with
like — and is specified exactly for that reason.

**D13 — the declaration is validated at Snakefile parse time.** `run_stress_test.smk`
calls `surface_axes.parse_surfaces(config)` beside the existing
`refuse_retired_experiment_keys(my_cfg)` — the repo's established parse-time
refusal pattern.

**The section is read through `get_config`**, per `AGENTS.md` § Conventions, and
the spelling is stated because the null case is otherwise ambiguous:

```python
reporting_cfg = get_config(config, "reporting", {}, optional=True) or {}
surfaces_cfg  = get_config(reporting_cfg, "surfaces", [], optional=True) or []
```

The trailing `or {}` / `or []` is the point, not noise: `get_config` returns a
present key's value **as-is**, so `reporting:` written with no body yields `None`
and would `TypeError` on subscript. The precedent is `run_stress_test.smk:526-529`,
which guards a nested optional key the same way. With this spelling a
present-but-null section resolves to `DEFAULT_SURFACE` rather than raising, which
is what D10's "absent or empty ⇒ `DEFAULT_SURFACE`" already promised.

Consequences: a malformed declaration fails `--dry-run`, so
`pytest tests/test_cli.py` is its gate; no rule declares it, so no DAG edge and
no rerun-trigger hazard is created (§5.3). Each surface entry has a **closed key
set** (`id`, `x`, `y`) and each axis a closed key set (`variable`, `months`,
`statistic`), so a typo *inside* the declaration is refused rather than ignored —
the R11 Q7 posture. **`parse_surfaces` also enforces §5.2's cross-axis
distinctness rule** and raises `DuplicateAxisVariableError` (`ext2-2`, V22). It is
called out because it is the one constraint in the schema that **no per-field
validator can reach**: both `variable` values are individually inside their closed
enum, and only the *pair* is illegal — which is how a closed key set plus a closed
value enum still admitted an unimplementable declaration. A typo in the section
name itself is not catchable: the
config root has no section whitelist (verified — nothing in the Snakefiles or
`snake_utils` enumerates allowed top-level sections). Recorded as a residual risk
in §7, not papered over.

### 5.3 The consumer side — what derives an axis

**D14 — the derivation is a library, not a rule.** `blueearth_cst/shared/surface_axes.py`,
a pure module with no Snakemake dependency:

```python
DEFAULT_SURFACE: Surface                          # id "default", temp x precip, derived months
def parse_surfaces(config) -> list[Surface]       # parse-time; refuses (D13)
def read_lookup(lookup_path) -> pd.DataFrame      # dtype={"st_id": str}
def read_indicators(indicators_path) -> pd.DataFrame   # dtype={"st_id": str}  (D28)
def key_width(lookup_df) -> int                   # inferred from st_id, D33
def month_classes(lookup_df, variable) -> tuple[list[int], dict[int, float]]  # varying, held->level
def derive_axis(lookup_df, axis) -> AxisResult    # D33; refuses per D16/D17
def axis_values(lookup_df, axis) -> pd.Series     # derive_axis(...).values
def axis_caption(lookup_df, axis) -> str          # derive_axis(...).caption, §5.5
def join_axes(indicators_df, lookup_df, surface) -> SurfaceJoin   # D33

@dataclass(frozen=True)
class AxisResult:
    values: pd.Series          # st_id (padded str) -> axis value
    caption: str               # §5.5
    degenerate: bool           # D19 — render as annotation, not as a dimension
    months: tuple[int, ...]    # M after defaulting (D11), so the consumer sees what was collapsed
    variable: str

@dataclass(frozen=True)
class SurfaceJoin:
    surface_df: pd.DataFrame   # indicator rows on the surface, axis columns attached
    baseline_df: pd.DataFrame  # the st_0 rows, partitioned out (S6)
    axes: dict[str, AxisResult]   # keyed by variable — carries degenerate + months
    key_width: int             # the width both frames were normalised to (D33)
```

The `_path` and `_df` suffixes are `naming.md` §3/§5, which is not advisory —
"new code MUST use `_path` for a variable holding a file-path string" — and this
block is normative API text an implementer transcribes verbatim, in a module HM-7
cites as its reference implementation. The `_csv` suffixes on the Snakemake labels
(`lookup_csv`, D20/D23) are correct and stay; §5 reserves extension suffixes for
that position.

`join_axes` returns `surface_df` and `baseline_df`: indicator rows whose `st_id`
is in the lookup, with the two derived axis columns attached, and — separately —
the rows whose `st_id` is absent from the lookup, which is exactly `st_0`. That is
S6 implemented as a partition rather than as a documented convention, and it is
why D4's omission of `st_0` from the lookup is load-bearing rather than tidy.

**D33 — the results are OBJECTS, and the key width is INFERRED.** `ext1-3` is
right that v2's signatures could not carry what v2's own text required of them.
Three defects, one fix each:

1. **`degenerate` had no channel.** D19 says a consumer receiving
   `degenerate = True` renders the axis as an annotation rather than as a plot
   dimension, and `axis_values` returned a bare `pd.Series` — so the flag existed
   in prose and nowhere in the API. `AxisResult` carries it, beside the caption
   and the effective `M`, which is the other thing a consumer needs and could not
   see (D11 derives `M`, so a caller that did not declare it cannot otherwise know
   what was collapsed). `axis_values` and `axis_caption` survive as one-line
   accessors, because `repo-fit-4` pinned those names and they are the ergonomic
   90% case.
2. **`ST_NUM` was required and not available.** D28 as v2 wrote it normalises both
   key columns "at `index_width(ST_NUM)`" — but `ST_NUM` is a config value, and
   `index_width` lives in `shared/snake_utils.py`, which D14 declares this module
   must not depend on. The width is therefore **inferred from the lookup**:
   `key_width(lookup_df)` returns the common length of the `st_id` strings and
   raises `LookupKeyWidthError` if they are not all equal. That is sound because
   WG-2 pins `st_id` as zero-padded text at one width for the whole table (§5.7),
   so the inference reads a pinned property rather than guessing. The baseline
   token is then `"0".zfill(width)` and `ST_NUM` is `max(int(st_id))` where
   anything needs it. The pure-module claim survives intact, which is the
   coherence win: v2's spelling would have forced either a `snake_utils` import or
   a config argument into a library specified as having neither.
3. **`join_axes` returned two frames and dropped everything else.** `SurfaceJoin`
   keeps `surface_df` and `baseline_df` under those exact names — `repo-fit-4`'s
   disposition survives as field names — and adds `axes` (one `AxisResult` per
   variable) and `key_width`, so the caller that draws the figure has the caption,
   the degeneracy flag and the padding it must use to render `st_0`'s annotation
   beside the surface.

   **Keying `axes` by variable is well-defined only because §5.2 refuses a surface
   whose two axes repeat a variable** (`ext2-2`). Without that rule the dict drops
   one axis with no error, and `AXIS_COLUMN[variable]` points both axes at one
   derived column. The schema constraint and this representation are **one decision
   recorded in two places**, and neither half is safe on its own: this is the
   representation, §5.2 is the refusal that keeps it total.

**Padding happens in ONE place, and it is the join.** `AxisResult` carries no
`key_width` and `derive_axis` re-pads nothing: it reads one table, so its index is
whatever the lookup holds, which WG-2 already pins. Only `join_axes` sees a second
frame with a second provenance, so only `join_axes` normalises — and it reports
the width it used. Said explicitly because the obvious implementer's error is to
pad defensively in both, which produces a token padded twice on any consumer that
composes them.

The R and JavaScript re-implementers get the same three facts as **semantics**
rather than as a Python object: §5.8's HM-7 text states that a derivation returns
values *plus* a caption *plus* a degeneracy flag *plus* the effective month set,
and that the join key's width is read from the lookup. What it does not state is
how a language packages them.

**D28 — the partition carries a postcondition and raises.** Making *absence* the
sole marker of the baseline has a failure mode the design named elsewhere and
then walked into: D2 and §5.7 both warn at length that `pd.read_csv` with no
`dtype` turns `01` into `1` and "the join silently misses". Under an
absence-means-baseline encoding that miss does not surface as an empty result —
it surfaces as **"every row is baseline"**, which is a shape the partition is
designed to produce and therefore looks plausible. Two rules close it:

1. **The library owns both reads.** `read_indicators(indicators_path)` sits beside
   `read_lookup(lookup_path)` and forces `dtype={"st_id": str}`, and `join_axes`
   **normalises both key columns to zero-padded strings** at the width
   `key_width(lookup_df)` infers from the lookup (D33 — v2 said
   `index_width(ST_NUM)`, which this module cannot reach) before partitioning, so
   a caller who loaded the frame some other way is repaired rather than silently
   mis-partitioned.
2. **`join_axes` asserts its partition, in three ordered checks.** Let `I` be the
   `st_id` set of the indicator frame and `L` the lookup's, both after
   normalisation, and `b = "0".zfill(key_width)`.

   | # | assertion | raises |
   |---|---|---|
   | a | `I \ L == {b}` — what the tables carry and the lookup does not is **exactly** the baseline token | `BaselinePartitionError` |
   | b | **`I \ {b} == L`** — set EQUALITY between the surface members and the lookup's members | `SurfaceMemberMismatchError` |
   | c | `surface_df` is non-empty | `BaselinePartitionError` |

   **Check b is added at v4 and is the one that matters** (`ext2-1`). v3 stated
   only a and c, which constrain the *extra* direction and say nothing about the
   *missing* one: a stale or partial indicator table whose members are a strict
   **subset** of the lookup's satisfies both — every id it does carry is in the
   lookup, the only absent id is still `st_0`, and the surface is still non-empty —
   so `join_axes` returned a **silently incomplete** response surface. That is a
   worse failure than the one D28 was introduced to catch: a mis-keyed join
   produces a visibly wrong shape, while a short join produces a plausible surface
   with holes in it, or a biased one if the missing members sit at one end of the
   grid. Given a, the only way b can fail is that a lookup member is missing from
   the indicators, so the message names the **missing ids** and the count.

   **Why a second error class rather than widening the first.** A missing member is
   not a baseline problem, and `BaselinePartitionError` would name the wrong thing
   in a diagnostic an out-of-repo consumer reads — the mis-citation class this run
   has caught twice already (D11 → D17, `architecture-7`'s scope clause). The
   error names are contract surface here (`LookupKeyWidthError`,
   `HeterogeneousAxisError`, `NonRectilinearAxisError`), so one more is cheap and a
   misleading one is not.

   Check c now catches only the degenerate residue — an **empty lookup**, which b
   cannot see because `I \ {b} == L == ∅` holds vacuously. It is kept for that.

All three are cheap, and all three are checks the pre-change code did not need,
because the axis was a column on the row.

The run-time coverage check D22 adds does **not** cover this: it lives in the
reducer, upstream of the join, so it cannot see a stale or mismatched lookup at
report time — and `validate_hm7` is test-time only, as §5.6 notes. D28 is the
report-time tier. **With check b, that sentence is true rather than aspirational:**
b is the same predicate as `validate_hm7`'s completeness check 1 (§5.8, "every
`st_id` in the lookup appears in every indicator table, and every non-zero `st_id`
in a table appears in the lookup"), evaluated at report time instead of at test
time. The two are deliberately the same statement in two tiers, not two different
checks — which is the answer to "why assert it twice": the test-time one runs in
this repo and never at a consumer, and the consumers are where the surface is
actually drawn (R9).

**The derived columns keep the old names**, `temp_change` and `precip_change`,
named once as `surface_axes.AXIS_COLUMN[variable]`. What changes for a consumer
is that it must join the lookup to obtain them; what does not change is the
column it then plots, so an existing call site keeps working and simply receives
values that are now correct for a seasonal design. Two declared surfaces produce
two frames, each carrying that pair — the surface `id` names the frame, not the
columns.

**Why not a rule.** Four reasons, in order of weight:

1. **No in-repo *rule* consumes a response surface.** WF3 has no plotting rule —
   rules 3.01–3.18 end at the reduction and the record gathers — so a rule would
   produce an artifact no *rule* reads. v1 stated this as "there is no in-repo
   consumer", which is **false and had to be corrected**:
   `docs/notebooks/Climate Stress Test.ipynb` is a shipped, user-facing consumer
   that reads `EXP_DIR / "config" / "stress_test_design.csv"` (line 500) and
   builds the surface with
   `.groupby(["temp_change", "precip_change"])["value"] … .unstack("precip_change")`
   (lines 683-685), plus two narrative sites (line 352, lines 481-482).

   **v2 then over-corrected**, and `ext1-2` is right that it broke a settled
   ruling doing so: it made the notebook a caller of `read_lookup` +
   `read_indicators` + `join_axes` + `axis_caption`, while D15, alternative 6.9
   and R9 all state the library has **no in-repo caller** — the G1-return's Fork B.
   A document cannot instruct both. The ruling stands and the notebook is realigned
   (§8 step 6): it is rewritten as a **contract-based external-consumer example
   that imports nothing from `surface_axes`**, deriving the axis from HM-7's text
   exactly as the R and JavaScript consumers must.

   That is a coherence win rather than a concession. Fork B's compensating
   requirement is that the contract text be complete enough to re-implement from
   (D15); the notebook is then the **only in-repo demonstration that it is** — a
   claim v2 asserted and could not check, because its worked example was a library
   call. R9 carries the point, and V21 is the check.
2. **A rule that wrote axis values would be a cache of a derivation** (C3), which
   is the exact proposal S2's third consequence already withdrew one layer up.
3. **R12 owns how WF3 executes** (`t2608082036`). Adding a rule, its rerun
   triggers, its log part and its benchmark part is run mechanics, and this design
   has no justification to spend R12's budget on a file nothing reads.
4. **It would inherit a live hazard.** Rule 3.09 declares `config = ancient(...)`
   and carries no `params:`, so it is deaf to `stress_test` edits (E5); a new rule
   reading a config section would face the same choice and the same trap.

**D15 — the derivation is specified, not only implemented, and the specification
must be COMPLETE.** An R or JavaScript consumer cannot import a Python module, and
S10 requires the *same* collapse over WF2's monthly change factors.

The G1-return ruling on Fork B makes this obligation sharper than v1 treated it.
Fork B accepts that the library has **no in-repo caller** (see R9), which means
`axis_values`, `axis_caption` and `join_axes` execute only in unit tests and in
out-of-repo consumers — so the normative documents are not a summary of the code,
they are **the only thing the real consumers read**. What must therefore appear in
normative contract text, not merely in this design:

| owned by | content |
|---|---|
| **WG-2** (§5.7) | the lookup's schema — columns, dtypes, `st_id` padding and string domain, `12 × ST_NUM` row count, `(st_id, month)` ordering and completeness, percent semantics, no `st_0` row |
| **HM-7** (§5.8) | the axis derivation — the collapse formula **including D26's flat-vector short-circuit**, the varying/held classification and its exact-zero threshold (D11), the affine-statistic and subset/homogeneity constraints, the **degenerate-axis rule** (D19/D27), and the **caption case table** (§5.5) |

Each document is complete for what it owns; neither restates the other's half.
`surface_axes.py` remains the **reference implementation** of HM-7's half.

One in-repo artifact reads those documents rather than the module:
`docs/notebooks/Climate Stress Test.ipynb`, rewritten at §8 step 6 as a
contract-based consumer that imports nothing from `surface_axes` (`ext1-2`). It is
an in-repo **re-implementer**, not a caller, so R9's gap is unchanged — but it is
the one place where "the contract is complete enough to re-implement from" is
exercised instead of asserted, and V21 compares its output against the reference
implementation once, during the migration.

`csthelpers::plot_climate_surface` needs no in-repo change:
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

**D27 — precedence: classify first, and a degenerate axis short-circuits both
rules.** D16 and D19 collide as v1 wrote them, and D11's own default is the
collision's worked example. When nothing varies, the varying set is empty, so no
non-empty subset exists and D16 refuses the case D19 declares legal — while D11's
default for that same case is *twelve held months*, the precise input D16 says
must raise. The evaluation order is now normative:

1. **Classify** every month varying/held per D11's exact-zero threshold.
2. **If the varying set is empty, the axis is degenerate (D19).** Neither the
   subset rule nor the homogeneity rule applies. An explicit `months:`
   declaration is **admitted**; the default is all twelve. The value is the
   collapse over `M` per **D32**. This is what keeps a temperature-only stress test
   legal on its precip axis — a one-dimensional design, not a malformed one.
3. **Otherwise** the declared set must be a non-empty subset of the varying set
   (`HeldMonthInAxisError`) and those months must be homogeneous
   (`HeterogeneousAxisError`).

**D32 — a degenerate axis bypasses the CONSTRAINTS, not the FORMULA.** v2 said a
degenerate axis "returns the constant for those months" and routed around step 3,
where the collapse is defined — so when `M`'s months are held at *different*
offsets (D19's own third row, and D27's admitted explicit-`M` case) the design
named no value at all. `ext1-3` is right that two conforming implementations could
then legitimately return different scalars: the first month's level, the
unweighted mean, the weighted mean, or a refusal.

The rule, which costs nothing because it reuses what step 3 already defines:

    axis(st) = collapse(M)   — the same weighted mean, with D26's exact-equality
                               short-circuit, over exactly the same months

Because nothing varies, the result is **constant across members** — that is what
makes the axis degenerate — but it is *not* necessarily equal to any one month's
level, and the two are different statements. When the held levels over `M` are
equal (the common case: a flat precip vector), the short-circuit returns that
level exactly and the axis reads as the number the user wrote. When they are not,
the axis is the weighted mean of them, the caption says so (§5.5 case 4c), and
nothing is silently invented.

This also removes the last place where a value could be produced by a path the
contract text does not define — which matters more here than in most designs,
because under Fork B the contract text is what the real consumers execute.

Step 2 is not a carve-out that weakens C1. C1's misreport is *a held month
diluting a varying one*; when nothing varies there is nothing to dilute and the
axis reports the one value every member shares, which is exactly what happened.
This ordering is normative in §5.8, where v1's HM-7 text stated the subset rule
unconditionally and would have exported the collision to every re-implementer.

**D17 — non-affine statistics: closed vocabulary plus a postcondition.** D12
closes the vocabulary, which is the static half. The dynamic half:
`axis_values` asserts that the **distinct axis levels are evenly spaced** and
raises `NonRectilinearAxisError` otherwise. **The tolerance is named, not left to
the implementer:** consecutive gaps must agree to `rtol = 1e-9` relative to the
mean gap, compared with `math.isclose`. This is a genuine tolerance rather than
D11's exact-zero, and for the opposite reason: the levels being compared are
*different* values arrived at by different arithmetic, so `float64` noise at the
1e-16 scale is expected and must not be a failure — while a non-affine statistic
breaks spacing by orders of magnitude, not by ulps. 1e-9 sits between the two by
seven decades in each direction. It is normative in §5.8. This
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
varying if `max - min > 0` **exactly** (D11's single threshold; v1 wrote `> tol`
here and `> 0` in D11, with `tol` defined nowhere), else held at level `L_m`.

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

**Step 4 — compose.** Let `V` be the varying months, `M` the declared or derived
month set, `E = V \ M` the varying months the axis does **not** summarise, `H` the
held months (all outside `M`, by D16's subset rule), and `G` the distinct held
levels over `H`.

**D31 — the leading phrase is derived from `M`, and from nothing else.** v2
selected case 1 on `H` being empty and emitted `mean change over the year`, which
is a **false statement about the plotted quantity** whenever every month varies
and the user declared a proper subset — a case D16 explicitly admits and which is
the *normal* way to get an honest axis out of a heterogeneous design (D16's
refusal names the homogeneous subsets so the user declares one). The failure is
not cosmetic and not confined to the label: the same `M` collapses the projection
overlay (§5.8), so an overlay computed over JFM would be captioned, and read, as
annual.

Worked: twelve varying months, JFM at 0.7–1.3 and Apr–Dec at 0.9–1.1, `step_num:
2`, `months: [1,2,3]`. The axis is **−30 / 0 / +30**; the annual collapse of the
same members is **−14.931507 / 0 / +14.931507**. v2 would have labelled the first
`mean change over the year` — a name for a quantity that differs from the plotted
one by **2.01×** (E15).

The composition, in one rule whose only exception is the degenerate branch
(D19, reached first per D27):

    caption = "mean change over " + label(M)
              + clauses(E, "also vary")      if E non-empty
              + clauses(H, "held at")        if H non-empty

**One clause builder, used twice.** `clauses(S, phrasing)` partitions `S` by a
per-month key — the level `L_m` for held months, the `(min, max)` pair over
members for varying ones — labels each group's month set with step 2, formats its
value with step 3, and caps at **three** groups, beyond which it emits the
catch-all. The cap and the catch-all are step 4's existing legibility rule (v2's
cases 3b/3c), now stated once and inherited by both clause kinds rather than
written for held months alone.

**The cap is per clause group, not over their sum** — so case 1c may carry up to
six groups. Stated because it is otherwise ambiguous and two implementations would
split it differently: a combined cap would let a busy `E` swallow the held-month
clause, which is the more informative of the two (a held month is a decision the
user made about a month the axis does not show). Six groups is at the edge of
legible; a design that reaches it is one the reader should be looking at the
lookup for anyway, which is what the catch-all already says.

| case | condition | caption |
|---|---|---|
| 1 — uniform, whole year | `H` and `E` both empty | `mean change over the year` |
| **1b — explicit subset of an all-varying design** | `H` empty, `E` non-empty | `mean change over JFM; Apr–Dec also vary, -10% to +10%` |
| 2 — some vary, rest unchanged | `E` empty, `G == {0}` | `mean change over JFM; Apr–Dec unchanged` |
| 3 — some vary, rest held at one offset | `E` empty, `\|G\| == 1`, non-zero | `mean change over JFM; Apr–Dec held at -20%` |
| 3b — rest held at several offsets | `E` empty, `2 <= \|G\| <= 3` | `mean change over JFM; Apr–Sep held at -20%; Oct–Dec held at -10%` |
| 3c — more than three held levels | `E` empty, `\|G\| > 3` | `mean change over JFM; remaining months held at declared monthly offsets` |
| **1c — both kinds outside `M`** | `E` and `H` non-empty | `mean change over JFM; Apr–Jun also vary, -10% to +10%; Jul–Dec held at -20%` |
| 4 / 4b / 4c — nothing varies | varying set empty | see D19; reached before D16 applies, per D27 |

In case 1 the condition already implies `M` is all twelve — `H` empty makes every
month varying, and `E` empty with D16's subset rule forces `M = V` — so the phrase
`the year` is derived, not special-cased. Cases 1b and 1c are new at v3 and are
the ones v2 mislabelled.

**Why `E` is described rather than refused.** Refusing an `M` that omits varying
months would forbid the exact declaration D16's refusal message tells the user to
write. It is admissible, and on a lookup written by rule 3.09 it is also safe in
the way that matters for a surface: every month of member `j` sits at step `j` of
its own range, so a collapse over any non-empty `M` is injective in `j` and two
members never collide on the axis. That guarantee is a property of 3.09's
`np.linspace` construction, not of the schema — a hand-edited lookup is outside
it, and the rectilinearity postcondition (D17) is what would notice.

**D19 — an axis with no varying months is degenerate, not an error.** Every
member shares one value, so the axis has a single level. Refusing would break a
**legitimate** experiment: a temperature-only stress test (`precip` flat, `temp`
stepped) is exactly this on its precip axis, and it is a one-dimensional design,
not a malformed one. So `derive_axis` returns D32's collapse over `M` with
`degenerate = True`, and the caption — over the held levels **within `M`**, per
D31's rule that the phrase describes what was collapsed — is:

| held levels over `M` | caption |
|---|---|
| all zero | `unchanged` |
| one non-zero level `L` | `held at -20%` |
| several levels | `held at declared monthly offsets (weighted mean -13.3%)` |

When `M` is not all twelve, ` in <label(M)>` is appended (`unchanged in JFM`), for
D31's reason: the caption may not describe months the axis did not collapse. The
third row states the value D32 returns rather than leaving the reader to guess
which of several offsets was plotted.

A degenerate axis is an **annotation**, not a plot dimension; a consumer that
receives `degenerate = True` renders it in the caption rather than as an axis.
The rectilinearity postcondition (D17) passes trivially on one level.

**D27 governs how this reaches D16.** Degeneracy is decided at classification
time, *before* the subset and homogeneity rules are consulted, so neither applies
here — including when the user declares an explicit `months:` set on a degenerate
axis, which is **admitted** and returns the constant. Without that ordering, D16's
"non-empty subset of the varying set" refuses every case D19 admits, and D11's own
all-twelve default for the degenerate case is the precise input D16 says must
raise.

Eight of the ten caption cases and both hazard cases were executed against
`np.linspace` member matrices; the rendered captions are in §9. (v1 said "six
cases": the degenerate row was one case in the §5.5 table and three rows in
D19's, which §5.8 now reconciles as 4 / 4b / 4c.) **Cases 1b and 1c are v3's and
are argued, not rendered** — the closed-form arithmetic behind 1b is E15 — because
the ad-hoc rendering harness that produced §9's table belongs to v1's run. §8 step
4 promotes all ten into `tests/test_surface_axes.py` fixtures, which is where the
rendering stops being ad hoc.

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
by `month`, and converts **both** percent columns to the generator's factor form
per D3 — `precip_mean = 1 + precip_change/100` **and**
`precip_variance = 1 + precip_variance_change/100`. Omitting the second was
`architecture-2`'s concrete defect: every shipped config sets variance
`min = max = 1.0`, i.e. `0.0` percent, so a literal implementation of v1 would
have handed `apply_climate_perturbations` a variance factor of **zero** on the
default configuration.

**Two script-level facts v1's "everything else is untouched" got wrong**, both
verified in the file:

- **The arity check moves from 4 to 5.** `impose_climate_change.R:12-14` hard-fails
  on `length(args) != 4L`; a new positional argument makes that a guaranteed
  `stop()`. The message enumerates the arguments and moves with it.
- **D29 — the filter carries a postcondition, on the R side, at the point of
  use.** Today the script reads a file that *is* the member — twelve rows, no id —
  and passes `cst_data$precip_mean` straight into `apply_climate_perturbations`
  (`:27, :68-70`). After D21 that becomes a filter-and-order join, and a join that
  matches nothing yields a **zero-length** vector while one that matches partially
  yields a short one; R's recycling rules make a silent wrong answer at least as
  likely as an error. This is the one seam the migration moves across a language
  boundary, and it converts what is today a loud, structural
  `MissingInputException` from Snakemake into a quiet data condition inside a
  script with no guard. So the script **stops** unless, after the filter and sort,
  `nrow(cst_data) == 12L` and `identical(cst_data$month, 1:12)`, with the member
  token named in the message. That mirrors the assertion the Python side already
  makes (`export_wflow_results.py:108-121`) and is what keeps C27's
  lookup-token ≡ filename-token identity checked on the consuming side after the
  pad width stops being carried by the filename.

Everything else in that script — the prefix/suffix split from the declared output
path, the transient flags, the `apply_climate_perturbations` call — is untouched.

**D34 — the read-filter-assert block is EXTRACTED, so D29's guard is
executable in a test.** `ext1-5` found that v2 gave D29 a falsifier no proposed
check could reach: V17's claim is about what the script does with a malformed
member slice, and its assigned check was a WF3 run on the *valid* rapid config —
green whether the guard exists or not. A negative execution is the only falsifier
of a guard, and it cannot be written against `impose_climate_change.R` as it
stands, because the malformed-input path is reached only after the script has read
the weathergen YAML and loaded a realization netCDF through `weathergenr`.

So the block becomes a sourced function in a new file,
`blueearth_cst/weathergen/read_member_grid.R`:

    read_member_grid(lookup_path, st_id_token) -> data.frame  # 12 rows, month 1..12
        reads with colClasses = c(st_id = "character")
        filters st_id == st_id_token, orders by month
        stop()s unless nrow == 12L and identical(month, 1:12), naming the token
        returns the frame; the caller converts percents per D3

`impose_climate_change.R` sources it beside `global.R` and calls it. Nothing about
D29's semantics changes — this is where the assertion lives, not whether it
exists.

Three properties make it the right shape rather than a testing convenience: the
function is the **R-side implementation of WG-2's consumer half**, so the contract
has one named counterpart per language; `global.R` is options-only, so sourcing
costs no package load and a negative test needs neither `weathergenr` nor a
netCDF; and the extraction is what lets V17 assert the **member-specific
diagnostic**, which is the part of D29 that makes a cross-language failure
readable at all.

The alternative — reorder `impose_climate_change.R` so the lookup read precedes
the netCDF read, and drive the whole script with dummy paths — was considered and
is weaker: it tests the negative cases only, since a *valid* slice then fails at
the next stage for an unrelated reason, so "the guard does not reject a good
member" would be asserted by the absence of a string in stderr.

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
check "is never invoked at run time". §5.8's completeness check adds the
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

### 5.7 WG-2 replacement text (drop-in) — the lookup's schema

**This section closes the two blocking findings** (`architecture-1`,
`repo-fit-1`). v1 deleted the artifact WG-2 pins and never mentioned WG-2: the
strings "WG-2", "weather-generator-seam" and "validate_wg2" appeared nowhere in
it. After that migration `validate_wg2` would still have asserted the header
`month,temp_mean,precip_mean,precip_variance` on a file that no longer exists, and
`test_wg2_synthetic_pass` would have gone on **positively asserting green that a
retired contract holds** — worse than R9's silent skips, because it is an
assertion rather than an absence.

**D30 — WG-2 keeps its id and re-points at the lookup; it is not retired and no
new WG entry is minted.** Ruled at the G1-return (Fork A). The reasoning is that
WG-2 is *the perturbation grid the weather generator consumes*, and that artifact
did not disappear — it moved and changed shape. Its consumer (rule 3.12 /
`impose_climate_change.R`) and its producer (rule 3.09) are unchanged. Retiring
the id and minting a WG-7 would assert a discontinuity that did not happen, and
would orphan every citation of WG-2 in the bounded-substitution walkthrough.

**Disposition per passage, not an exclusion clause.** `weather-generator-seam.md`
touches WG-2 in four places, and v1's §5.7 was caught by `architecture-7` for
exactly the defect of writing "everything not restated is unchanged" over passages
that were not:

| passage | disposition |
|---|---|
| `## WG-2` section, `:110-135` | **replaced** by the blockquote below, in full |
| the `input:` on rule 3.16 clause, inside WG-2's path bullet `:113-115` | **struck** — D22 drops both parameter inputs from 3.16; the lookup is a 3.12 input only |
| validator-index row, `:330` | **amended** — fixture path becomes `<exp>/config/stress_test_lookup.csv`; "continuously verified? yes (persists)" **still holds** and is re-asserted, since the lookup is not `temp()` and persists in `rule all` |
| bounded-substitution walkthrough, `:298` and `:310`/`:313` | **amended** — "WG-2 (the `st_<m>.csv` perturbation grid)" becomes "WG-2 (the `stress_test_lookup.csv` perturbation grid)"; the `validate_wg2` acceptance-check entry stands unchanged |

> ## WG-2 — stress-test perturbation grid
>
> - **path pattern:** `<exp>/config/stress_test_lookup.csv` — **one file for the
>   whole experiment**. Replaced the per-member
>   `<exp>/climate/weathergenr/_work/st_<m>.csv` on `<DATE — the implementation
>   commit's date>`, together with the derived cache
>   `<exp>/config/stress_test_design.csv`, which it absorbs. `_work/` is deleted.
>   The rename to `lookup` signals that **the shape moved**, not merely the path.
>   It sits beside the config snapshot whose settings produced it: it is a record
>   of what ran, not scratch.
> - **producer:** rule 3.09 `prepare_stress_test_grid`
>   (`blueearth_cst/experiment/prepare_cst_parameters.py`).
> - **consumer:** rule 3.12 `perturb_climate_realization` (weathergenr
>   `impose_climate_change.R`), passed in as `lookup_csv` — a **constant** input,
>   no longer carrying the member wildcard. The member id arrives as a positional
>   argument and the script filters on it.
> - **shape:** a CSV with **header exactly**
>   `st_id,month,temp_change,precip_change,precip_variance_change`, and
>   **`12 × ST_NUM` rows** — twelve per member, `month ∈ 1..12`, members
>   `1..ST_NUM`. The `(st_id, month)` grid is **complete and duplicate-free**, and
>   rows are sorted by `(st_id, month)`.
> - **semantics:** `temp_change` additive (°C); `precip_change` and
>   `precip_variance_change` **percent**, not multipliers — `0.0` means no change,
>   `-30.0` means a 0.7 factor. The multiplier convention survives only inside the
>   generator: the R side reconstructs `1 + <col>/100` for **both** percent
>   columns. The project config keeps its 12-element multiplier vectors; this is
>   an artifact-unit rule, not a config-surface change.
> - **precision:** the member levels are `float32` shortest-repr quantized (the
>   grid the user asked for); the percent text is written at **`float64` shortest
>   repr of the exact conversion**, so the reconstructed multiplier is within one
>   `float64` ulp of the level. It is **not** bit-identical for every level, and
>   cannot be made so — measured, 1,155 of 50,000 `float32` multipliers admit no
>   `float64` percent that reconstructs them exactly under `1 + p/100`. A consumer
>   may rely on the bound, not on exactness.
> - **admissible multiplier domain: `multiplier ≥ 0.5`, with no upper bound.** This
>   is the **precondition of the bound above**, not a caveat on it: the producer
>   refuses a configuration declaring a precipitation mean or variance multiplier
>   below `0.5` before the DAG is built, so every lookup this contract describes
>   was written from admitted multipliers and the one-ulp bound holds over the whole
>   table, unconditionally. Outside the domain it does not: at level
>   `0.013596006` the specified conversion reconstructs `0.013596005999999883`,
>   68 ulps out, because the percent's rounding scale stops shrinking with the
>   level once `|percent|` crosses 64. There is deliberately **no ceiling** — the
>   bound was measured to hold out to `1e6`, so an upper cap would refuse
>   configurations the arithmetic serves correctly. A re-implementer needs the
>   floor to *validate its own producer*; a consumer reading a lookup needs only the
>   bound.
> - **`st_id`:** the member id, **zero-padded** to a width derived from `ST_NUM`
>   (C27: `01 … 12` at twelve points, unpadded below ten), **textually identical**
>   to the member filename token, so the two are ONE token. **Read it as a
>   string** — `pd.read_csv` with no `dtype` returns `01` as `1` and every join
>   silently misses. R readers pass `colClasses = c(st_id = "character")`.
>   **Every `st_id` in one table has the same width**, which is what lets a
>   consumer infer the join key's width from the table itself rather than needing
>   `ST_NUM` passed alongside it. A table mixing widths is malformed.
> - **`st_0` has NO row.** The table covers members `1..ST_NUM` only. `st_0` is
>   the reserved unperturbed baseline (naming.md §4): it has no parameters, is
>   produced by rule 3.11 rather than by perturbation, and rule 3.12 never runs
>   for it. Its **absence is load-bearing**, not incidental — it is what makes
>   "not on the surface" a structural fact rather than a convention, and an
>   all-zero `st_0` row would be indistinguishable from an identity member's row
>   while denoting a differently-processed climate (the raw generated series, not
>   that series round-tripped through a perturbation that is not the identity at
>   unit factors).
> - **column vocabulary:** **closed**. A new perturbation parameter is a new
>   COLUMN, and adding one requires a C28 ruling — the shape barrier is gone, the
>   contract barrier is not. Refused at write time by
>   `prepare_cst_parameters._KNOWN_AXES`.
> - **temp() lifecycle:** not `temp()`; a `rule all` target (`WF3_TARGETS` entry
>   `stress_test_lookup`), so it persists.
> - **pinned surface:** the exact header and column order; the `12 × ST_NUM` row
>   count with a complete, duplicate-free `(st_id, month)` grid; the `(st_id,
>   month)` sort order; `st_id` as zero-padded TEXT at the filename's width, **one
>   width for the whole table**; `st_0` ABSENT; the additive-vs-percent column
>   semantics.
> - **deliberately unpinned:** the numeric values themselves (they are the
>   experiment), and the percent text's digit count beyond the `float64`
>   shortest-repr rule.
> - **validator:** `validate_wg2`.

**`validate_wg2` changes mechanism, not just constants**, and §8 step 5 carries
it: `_WG2_HEADER` becomes the five-column lookup header; the `n != 12` row
assertion becomes `n == 12 × ST_NUM` **plus** a `(st_id, month)` grid-completeness
check (every declared member present, twelve distinct months each, no duplicates);
and a new assertion that **no row carries the `st_0` token**. The `st_id` dtype is
checked as text **and its width asserted uniform** (D33 — the whole join-key
inference rests on that being true of every table, so the validator is where it
becomes a checked property rather than a documented one), since a validator that
reads the frame with inferred dtypes would itself be subject to the `01` -> `1`
hazard the contract names. `ST_NUM` reaches
the validator as an argument, the way the relational validators already take their
second input.

### 5.8 HM-7 replacement text (drop-in) — the axis derivation

Replaces the `## HM-7` section of `dev/reference/contracts/hydrological-model-seam.md`.
**Per-passage disposition, not an exclusion clause** — v1's scope clause
("everything not restated below is unchanged") was false on three live passages,
and `architecture-7` is right that a drop-in whose scope clause is wrong cannot be
dropped in:

| passage | disposition |
|---|---|
| path pattern, producer, consumer, variable tokens, `rlz_id` grain, `location`, `basin` reservation, `aggregate_rlz` retirement, `temp()` lifecycle, `RT_*.csv` removal note | **unchanged** |
| **"axis-column rename (2026-08-05)"** bullet, `:346-354` | **DELETED.** It documents the naming of two columns that no longer exist, and ends "Both spellings are named once in code, as `interchange_contracts._PERTURBATION_AXIS`" — a symbol §8 step 5 deletes. Its content is subsumed by the removal note below. The `dev/milestones/r09/migration_indicator-axis-columns.md` record it cites stays where it is; history is not edited |
| **"axis VALUE, not just its name (2026-08-07, [R9-3])"** bullet, `:355-368` | **DELETED.** It defines the annual collapse *as the columns' definition*, which is the defect this design removes. Its two surviving claims — the WF2 overlay tie and the evenly-spaced guarantee — are restated below as properties of the **derivation** rather than of a stored column |
| **HM-4 → HM-5 → HM-7 relational check 3**, `:420-422` | **AMENDED**, and v1 wrongly listed it as untouched. It reads "`qstats_df`'s gauge columns (header minus `statistic` and the `_PERTURBATION_AXIS` columns `temp_change,precip_change`, ordered per `export_wflow_results.py:66-67`)". It becomes "header minus `statistic`", with the `_PERTURBATION_AXIS` clause struck. The validator's **logic** needs no change — `validate_hm_gauge_column_identity` check 3 compares the `location` value SET post-CR-2 rather than subtracting axis columns from a wide header — which is what the non-interaction claim actually rests on, and is why this is a prose amendment rather than a code change |

The blockquote below is the replacement `## HM-7` body.

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
> **The response-surface axis is now a derivation, and this section defines it
> COMPLETELY.** The reference implementation is
> `blueearth_cst/shared/surface_axes.py`, but no rule in this repo calls it —
> the consumers that draw a surface are out-of-repo (CST-API, the frontend,
> `csthelpers`) and re-implement from this text, as does the repository's own
> `docs/notebooks/Climate Stress Test.ipynb`, which is deliberately written
> against this section rather than against the module. So everything an
> implementer needs is here, including the tolerances and the edge cases.
>
> **Source.** `<exp>/config/stress_test_lookup.csv`, whose schema is pinned by
> **WG-2** (`dev/reference/contracts/weather-generator-seam.md`) and is not
> restated here. The two facts this section leans on: `st_id` is zero-padded
> **text** and there is **no `st_0` row**, because `st_0` is the reserved
> unperturbed baseline — it is not a surface member and is reported as an
> annotated reference value beside the surface, never placed on it.
>
> **A declared axis is a triple `{variable, months M, statistic}`.** Evaluate it
> in this order; the order is normative, because steps 1 and 2 decide whether
> step 3's refusals apply at all.
>
> **What a derivation returns — four things, not one.** The values (member →
> number), the **caption**, a **degenerate** flag, and the **effective `M`** after
> defaulting. A consumer needs all four: the flag decides whether the axis is a
> plot dimension or an annotation, and `M` is derived rather than declared in the
> default case, so a caller cannot otherwise know which months were collapsed.
> How a language packages them is its own business; that all four cross the
> boundary is not.
>
> **1. Classify months.** For each month `m`, over members `st_id ∈ 1..ST_NUM`:
> the month is **varying** iff `max − min > 0` **exactly**, and **held at level
> `L_m`** otherwise. The threshold is exact zero, with **no tolerance**: a held
> month's values are bit-identical across members by construction (one written
> text, read by every member), so a tolerance would buy nothing and would create
> a band in which a month is neither varying nor held.
>
> **2. If the varying set is empty, the axis is DEGENERATE.** Every member shares
> one value, so the axis has a single level. This is a **legitimate** design, not
> an error — a temperature-only stress test is exactly this on its precip axis.
> Mark the axis `degenerate`. `M` defaults to all twelve months; an explicit `M`
> is **admitted**. Neither the subset rule nor the homogeneity rule of step 3
> applies — but **the collapse below does**: the value is the same weighted mean,
> with the same exact-equality short-circuit, over exactly the same `M`. A
> degenerate axis bypasses step 3's *constraints*, never its *formula*, and an
> implementation that returns some other scalar (the first month's level, an
> unweighted mean) is non-conforming. The result is constant across **members** —
> that is what degenerate means — which is a different statement from being equal
> across **months**: when `M`'s months are held at different offsets the axis is
> their weighted mean, and the caption says so. A degenerate axis is an
> **annotation**, not a plot dimension: a consumer renders it in the caption
> rather than as an axis.
>
> **3. Otherwise, constrain `M` and collapse.** `M` defaults to the **varying
> months**, which is what makes the default axis report the range actually
> explored rather than a diluted annual figure. Two constraints a consumer may
> rely on and an implementation must enforce:
>
> - **`M` must be a non-empty subset of the varying months, and those months must
>   share the same `(min, max)`.** A held month contributes a constant and
>   reproduces the annual misreport this contract removed — `months: [1..12]` on a
>   JFM-varying, Apr–Dec-held-at-−20% design returns −15% for a member that
>   imposed −30%. Heterogeneous varying months make the mean an average of unlike
>   perturbations that no caption can describe honestly. Refuse both.
> - **Only affine statistics.** Members are `min + (j/n)(max − min)` month by
>   month, so an affine collapse is affine in the step index and **the axis is
>   evenly spaced across the grid**. A max or a quantile is not, and the surface
>   stops being a regular grid. The admitted vocabulary is `mean` alone. An
>   implementation additionally **asserts** the distinct axis levels are evenly
>   spaced — consecutive gaps agreeing to **`rtol = 1e-9`** relative to the mean
>   gap — the postcondition the evenly-spaced guarantee never had. Two or fewer
>   distinct levels pass trivially. Note the two thresholds in this section are
>   deliberately different: month classification (step 1) is **exact zero**
>   because a held month's values are bit-identical by construction, while this
>   one is a true tolerance because it compares different values reached by
>   different arithmetic.
>
>   The two are **independent**, and one does not imply the other: a heterogeneous
>   design (Jan ±30%, Feb 0→+50%) yields axis levels −15, +12.5, +40 — perfectly
>   evenly spaced, and matching no month's imposed change.
>
> **The collapse, with its flat-vector rule.** For `statistic: mean`:
>
>     axis(st) = v(st, m*)                             if all v(st, m), m in M, are EXACTLY equal
>              = sum_{m in M} w_m * v(st, m) / sum_{m in M} w_m   otherwise
>     w_m      = the month's length in the noleap calendar (31, 28, 31, …)
>     v(st, m) = the lookup's <variable>_change value for member st, month m
>     m*       = ANY m in M -- they are equal by the branch's own guard, so the
>                choice is immaterial and an implementation need not fix one
>
> **The exact-equality short-circuit is normative, not an optimization.** Under
> the homogeneity constraint above, equal values are the **normal** path for every
> admissible axis, and a weighted mean of twelve identical values does not
> generally return that value: measured, `np.average` over the noleap month
> lengths differs from the input in **49%** of random percents and **48%** of
> random °C values. Realistic grids hit it — a 0.6–1.4 precip range at
> `step_num: 3` gives `−13.33333` → `−13.333330000000002`. Without the
> short-circuit, the derived axis would sit one ulp off the imposed value on
> roughly half of all non-round grids, and this repo treats that ulp as
> load-bearing.
>
> **The caption is derived from the lookup, never typed.** A typed label drifts
> from the design it describes; a derived one cannot.
>
> **The leading phrase names `M`, always** — `mean change over <label(M)>` — and
> is never selected from the global varying/held classification. A design whose
> twelve months all vary but whose declared `M` is JFM is captioned *over JFM*,
> because JFM is what was collapsed and what the plotted number is. Labelling it
> `over the year` asserts a quantity that was not computed, and the same `M`
> collapses the projection overlay below, so the error would propagate from the
> label into the comparison.
>
> Two trailing clause groups follow the phrase, in this order, over the months
> **outside** `M`: `E = V \ M`, the varying months the axis does not summarise;
> then `H`, the held months. Both are built by **one rule**: partition the month
> set by its per-month key — the held level for `H`, the `(min, max)` pair over
> members for `E` — label each group's months, format its value, and **cap at
> three groups**, beyond which emit the catch-all. The cap applies **per clause
> group and not to their sum**, so a caption carrying both may show up to six; a
> combined cap would let a busy `E` swallow the held-month clause, which is the
> more informative of the two. The cap is a legibility rule, not a correctness
> one: beyond three, the honest statement is that the pattern is not summarisable
> and the reader should look at the lookup.
>
> With `G` the distinct held levels over `H`:
>
> | case | condition | caption |
> |---|---|---|
> | 1 — uniform, whole year | `H`, `E` both empty | `mean change over the year` |
> | 1b — explicit subset of an all-varying design | `H` empty, `E` non-empty | `mean change over JFM; Apr–Dec also vary, -10% to +10%` |
> | 2 — some vary, rest unchanged | `E` empty, `G == {0}` | `mean change over JFM; Apr–Dec unchanged` |
> | 3 — rest held at one offset | `E` empty, `\|G\| == 1`, non-zero | `mean change over JFM; Apr–Dec held at -20%` |
> | 3b — rest held at several offsets | `E` empty, `2 <= \|G\| <= 3` | `mean change over JFM; Apr–Sep held at -20%; Oct–Dec held at -10%` |
> | 3c — more than three held levels | `E` empty, `\|G\| > 3` | `mean change over JFM; remaining months held at declared monthly offsets` |
> | 1c — both | `E`, `H` non-empty | `mean change over JFM; Apr–Jun also vary, -10% to +10%; Jul–Dec held at -20%` |
> | 4 — degenerate, all held at zero | varying set empty, `G == {0}` | `unchanged` |
> | 4b — degenerate, one non-zero level | varying set empty, `\|G\| == 1` | `held at -20%` |
> | 4c — degenerate, several levels | varying set empty, `\|G\| > 1` | `held at declared monthly offsets (weighted mean -13.3%)` |
>
> In case 1 the condition already implies `M` is all twelve, so `the year` is
> derived rather than special-cased. In the degenerate cases the levels and the
> reported mean are those **within `M`**, and ` in <label(M)>` is appended when `M`
> is not all twelve (`unchanged in JFM`) — for the same reason as the leading
> phrase: a caption may not describe months the axis did not collapse.
>
> A month set is labelled deterministically, in this order: **all twelve** →
> `the year`; a contiguous run in **circular** month order of length ≤ 3 → the
> initials (`JFM`, `JJA`, `DJF`, which subsumes the meteorological seasons with no
> season table); a contiguous circular run of length ≥ 4 → `<first>–<last>`
> (`Apr–Dec`, `Sep–May`); otherwise a comma list of three-letter abbreviations
> (`Jan, Mar, Jul`). A level is formatted signed, to three significant digits, with
> the unit of `variable` (`+3 °C`, `-20%`); a `(min, max)` range is formatted as
> two such levels joined by ` to `.
>
> **The same collapse must be applied to the projection overlay.** The CMIP6 dots
> are placed on these axes, so two different collapses would compare two
> different quantities. WF2 emits monthly change factors in percent
> (`cmip6_change_factors_monthly.csv`), so the same month-set collapse runs over
> the GCM table and over the lookup with no unit conversion between them.
>
> **The transfer is arithmetic, not semantic, and an overlay implementation must
> say so.** On the lookup side the constraints above make the mean a mean over
> **equal** values, so `axis(st)` *is* the change imposed in each declared month.
> A GCM's monthly change factors are not homogeneous, so the same formula over the
> same `M` computes a genuine average of unlike months — precisely the quantity
> this contract refuses to put on the stress axis. It also moves **non-affinely**
> as `M` changes: two GCMs at the same annual mean can sit at opposite ends of a
> JFM axis. So the overlay dot is a **summary** placed against an axis of
> **imposed** values, and narrowing `M` to the varying set (the default) sharpens
> the asymmetry rather than reducing it. Overlay treatment is deferred (Q6); the
> constraint and this caveat are not.
>
> **`st_id` (C28, R11 P2).** Spelled here exactly as WG-2 spells it — zero-padded
> TEXT at the member filename's width, so the two are ONE token — because it is
> the join key between this table and the lookup. **Read it as a string** in both:
> `pd.read_csv` with no `dtype` returns `01` as `1` and the join silently misses,
> and under the `st_0`-absent encoding that miss presents as "every row is the
> baseline" rather than as an empty result.
>
> **The key's width is READ FROM THE LOOKUP, not passed in.** WG-2 pins every
> `st_id` in one lookup to a single width, so an implementation takes that common
> length as the join width — refusing a lookup whose widths differ — and
> **re-pads the indicator tables' `st_id` to it** before joining. The baseline
> token is that width's zero-padded `0`. No implementation needs `ST_NUM` to
> perform the join, which is deliberate: a derivation that had to be told the
> member count could be told the wrong one.
>
> An implementation then **asserts its partition**, in three checks. Writing `I`
> for the `st_id` set the indicator tables carry, `L` for the lookup's, and `b` for
> the baseline token:
>
> 1. `I \ L` is **exactly** `{b}` — what the tables carry and the lookup does not
>    is the baseline and nothing else.
> 2. `I \ {b}` **equals** `L` — set equality, both directions. The surface members
>    present in the tables are exactly the members the lookup declares.
> 3. the surface partition is non-empty.
>
> Any of the three violated is an **error, not a shape**. Check 2 is the one an
> implementation is most likely to omit and the one that is least visible when it
> is: checks 1 and 3 alone are all satisfied by a stale or partial indicator table
> whose members are a strict **subset** of the lookup's, and a join under those
> conditions returns a response surface that is silently missing grid cells — or
> biased, if the missing members sit at one end of the grid — rather than
> reporting a mismatch. Report a missing member by naming it; a surface drawn from
> an incomplete join is indistinguishable from a smaller experiment.
>
> This is the **same predicate** as completeness check 1 under *Validator* below,
> evaluated when the surface is drawn rather than when the tables are validated.
> That is deliberate duplication across tiers, not redundancy: the validator runs
> in the toolbox repository and never in a consumer, and the consumer is where the
> surface is drawn.
>
> C28's second obligation — the writer refusing a design table carrying an axis
> the header cannot express — retires with the axis columns: the header expresses
> no axis, so a third perturbation parameter no longer needs a results column. The
> **contract** barrier stands: a new column in the lookup requires a C28 ruling,
> refused today by `prepare_cst_parameters._KNOWN_AXES`.
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

**6.9 Give the library a real in-repo caller.** Raised by `architecture-6`: with
no caller, D16's axis-tier refusal, D17's postcondition and the whole caption
algorithm execute only in unit tests, while the out-of-repo consumers that will
actually draw a surface re-implement from contract prose. The cheapest caller
proposed was a `--dry-run`-invisible assertion at the end of rule 3.16 that every
declared surface's axis derives without raising. **Ruled out at the G1-return
(Fork B):** it costs the lookup back as a 3.16 input, which reopens D22, and it
puts a *reporting* concern inside the *reduction* — a rule that would fail an
otherwise-correct experiment because a caption is malformed. It would also spend
R12's budget on rule mechanics for a check with no artifact. The accepted answer
is to make the contract text complete instead (D15, §5.7, §5.8) and carry the gap
as a named risk (R9).

**The notebook is not a second route to the same thing, and v2 tried to make it
one.** v2's §8 step 6 had `docs/notebooks/Climate Stress Test.ipynb` call
`read_lookup`, `read_indicators`, `join_axes` and `axis_caption` — a caller inside
the repository, which is what Fork B ruled out, and which contradicted D15, this
alternative and R9 in the same document (`ext1-2`). v3 realigns it: the notebook
re-implements from the contract text and imports nothing from `surface_axes`. The
only in-repo importers remain `tests/test_surface_axes.py` and the parse-time
`parse_surfaces` call, exactly as R9 states.

This alternative would become preferable the moment WF3 grows its own
plotting rule, at which point that rule is a genuine consumer and the assertion
is a by-product of drawing the figure rather than an extra input on the reducer.

**6.10 Retire WG-2 and absorb the lookup into HM-7.** The third option at Fork A:
one contract instead of two, with HM-7 owning both the results header and the
parameter table. Rejected by the owner's Fork A ruling and on the merits: the
lookup is the **Python → R** handoff and HM-7's declared consumer is the
CST-API/GUI, so absorbing it would leave the seam the artifact actually crosses
with no contract, and would leave `validate_wg2` and the bounded-substitution
walkthrough — which tells a drop-in generator what it must consume — describing a
deleted file. It would become preferable if the weather generator ever stopped
reading the perturbation grid directly.

## 7. Consequences and risks

**Falsifiable consequences.**

1. **A relabel needs no re-run of the EXPERIMENT, and no new experiment name.**
   Editing `reporting.surfaces` leaves `effective_config_digest` unchanged, leaves
   `_frozen_differences` empty, and does not re-fire 3.16b (measured, §9).
   Falsifier: an edit that trips `ExperimentConfigFrozenError`, moves the digest,
   or re-fires 3.16b.
   **v1 also named "re-fires 3.02" as a falsifier, and that was wrong** — it fires
   on correct behaviour, because 3.02's `config_snake` is a plain input and any
   config edit moves its mtime. A stated falsifier that fires on a working design
   is worse than none: whoever runs V12 would see 3.02 re-run and have to decide
   whether to believe the gate. 3.02 re-firing is **how the relabel gets
   recorded**, and it is cheap. See R8 for the cost it does carry.
2. **The reconstructed FORCING moves by at most one `float64` ulp of the
   perturbation level — and on every shipped config, not at all.** **The bound's
   precondition is D35's admitted domain** (`multiplier ≥ 0.5`, refused at parse
   time), added at v4 because `ext2-3` showed v3 asserting the bound over
   configurations where it is false by up to 574 ulps. Inside the domain the bound
   is unconditional. v1 claimed the
   forcing was bit-identical; D25 measures that it is not, in general, and cannot
   be made so. The grid quantization is preserved, so the re-recorded baseline
   differs from the current one in the **column set** and at most in the last bits
   of the reconstructed multiplier. Falsifiers: V20 (a reconstructed multiplier
   off a `float32` level by more than one ulp on a non-round grid) and V16 (the
   R-side factors differing from a pre-migration `st_<m>.csv`).

   **Indicator values are claimed unchanged only where the forcing is
   bit-identical**, which is every level of every shipped config, including
   `snake_config_baseline.yml`. Falsifier there: `compare_indicator_table`
   reporting a numeric failure — as opposed to the expected structural column-set
   mismatch — between a pre-change and a post-change run of that config.

   **No output bound is claimed for a non-exact grid, and v2 claimed one.** v2
   wrote this consequence as "indicator values move by at most one ulp … within
   the comparator's tolerance", which silently promoted a bound on a *forcing
   parameter* into a bound on *simulated indicators*. `ext1-4` is right that
   nothing supports the promotion: quantile mapping, wet-day occurrence and
   intensity thresholds, caps and floors, and then the hydrological model's own
   thresholds and storage states sit between them, and none is Lipschitz with a
   measured constant. E6 is the standing demonstration in this repo — a
   perturbation at exactly unit factors, i.e. **zero** parameter difference, moves
   the single-day precipitation maximum by −32.9% — that the map from forcing
   parameters to indicators is not gently behaved. See **R13**.
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
- **R5** — superseded at v2 by **R11**, which adds architecture-4's correction
  that the recording requires a WF3 invocation. Number retained so §9's and §8's
  citations do not renumber.
- **R6 — `docs/notebooks/Climate Stress Test.ipynb` needs a REWRITE, not a
  re-render, and it lands in the migration commit.** v1 filed the three notebooks
  as a stale-path chore deferred to `t2608132100`. That under-reads it: the
  notebook reads `<exp>/config/stress_test_design.csv` (line 500) and groups on
  `["temp_change", "precip_change"]` (lines 683-685), so after the column drop the
  cell raises `KeyError` — a **new breakage this design causes**, on top of a
  pre-existing staleness. It cannot be closed by re-rendering. C5 requires every
  live reference updated in the one commit, and `AGENTS.md` § Conventions calls a
  stale path in a document someone reads to do their job a defect. It is also the
  repo's only worked example of the thing this design exists to fix, so deferring
  it would leave no in-repo demonstration that the corrected axis is obtainable.
  §8 step 6 carries it. **At v3 the rewrite is contract-based rather than
  library-based** (`ext1-2`): v2 had it importing `surface_axes`, which is the
  in-repo caller Fork B ruled out.
- **R7 — S8's ordering is a real dependency, not a preference.** R12's
  `member_hash` is defined over the annual collapse this design deletes; landing
  R12 first would spend the member-identity work twice.
- **R8 — a `reporting:` edit turns the baseline's `yaml` target red.** Verified:
  `check_baseline.py:326-329` carries the WF3 config snapshot as a `yaml` target,
  and `fingerprint_yaml:449-455` hashes the **whole unprojected document**, into
  which `shutil.copyfile` puts `reporting:` verbatim. So on the baseline tree,
  editing a caption and re-running WF3 fails a numerical gate. This is a
  pre-existing property of that target — *any* config edit moves it — and the
  design's contribution is that it makes a caption edit a config edit. Mitigation
  is procedural and cheap: the shipped seeds declare **no** `reporting:` section
  (§8 step 6), so the fingerprint is unarmed until a project opts in, and a
  project that does re-records once. Recorded because v1 discussed the baseline
  gate at length entirely through the `indicator` target and never reached this
  one.
- **R9 — the library has no in-repo caller, and two of its checks are therefore
  unit-test-only.** Accepted deliberately at the G1-return (Fork B; alternative
  6.9). `parse_surfaces` is called at Snakefile parse time and runs only the
  design-tier warning; `axis_values`, `axis_caption` and `join_axes` execute on no
  repo execution path. So D16's axis-tier refusal, D17's rectilinearity
  postcondition, D27's precedence and D28's partition assertion are exercised by
  `tests/test_surface_axes.py` and by out-of-repo consumers that re-implement
  them — **not** by any run. The mitigation is that the normative documents are
  complete rather than summary (D15, §5.7, §5.8), which is why the tolerance, the
  degenerate rule, the short-circuit and the caption case table are in contract
  text rather than only here. The residual is real: an R consumer that skips those
  clauses re-introduces C1's misreport, and nothing in this repo would report it.

  **At v3 the mitigation gains one in-repo exercise of the contract itself.** The
  rewritten notebook (§8 step 6) re-implements the derivation from HM-7's text
  without importing the library, so it is a re-implementer of exactly the kind
  R9's residual is about — and V21 compares it against the reference
  implementation once, at migration time. That does not close the gap (it is a
  one-off, not a gate, and it exercises Python rather than R), but it converts
  "the contract text is complete enough to re-implement from" from an assertion
  into something that was tried at least once. It does **not** make the notebook a
  caller: the comparison is run by the migrating engineer in a scratch session,
  and the notebook itself imports nothing from `surface_axes`.
- **R10 — a widened test sweep is part of the migration, because the failure mode
  is a PASSING test.** Three sites keep asserting a retired contract after the
  change rather than failing: `tests/test_interchange_contracts.py:143-160`
  (`_wg2_good()`, built in memory, matching neither of v1's search strings),
  `tests/test_check_baseline_scope.py:56` and
  `tests/test_check_baseline_indicator.py:62-63,268,272` (synthetic seven-column
  headers). A fourth, `test_wg2_integration` (`:826-834`), is a stale fixture path
  behind a `skipif` — R9's silent-skip class exactly. §8 step 5b widens the sweep
  and re-derives its table.
- **R11 — the surface declaration is outside configuration identity.** Intended
  (D8), but `run_metadata.json` does not witness which surface a figure was drawn
  under; the config snapshot records it one level less directly, and only after a
  WF3 invocation. If publication provenance later needs a tighter tie, D6's
  export-on-demand is the place, not the digest. (v1's R5, restated with
  architecture-4's correction that the recording requires the invocation.)
- **R12 — the overlay collapse is asymmetric, and a second implementation of it
  already exists in-repo.** On the lookup side the collapse averages equal values;
  on the GCM side it averages unlike months, so an overlay dot is a summary placed
  against an axis of imposed values, and it moves non-affinely with `M`. §5.8
  carries the caveat. Separately,
  `dev/scripts/preview_wf2_projection_plots.py:299-302,319-322,364-367` already
  computes `precip_change = (precip - precip_ref)/precip_ref * 100` and
  `temp_change = temp - temp_ref` off an annual reference and plots GCM dots in
  exactly this axis space. It is a dev script, not a WF3 rule, so it is not a
  counter-example to D14 and changes no decision here — but Q6 will find a second,
  divergent implementation of the quantity §5.8 now specifies, and it should defer
  a **known** site rather than an unknown one.
- **R13 — on a non-exactly-reconstructing grid, nothing bounds the OUTPUT
  difference, and no gate here will.** D25 bounds the reconstructed multiplier to
  one `float64` ulp; between that and an indicator sit quantile mapping,
  occurrence and intensity thresholds, caps and floors, and a hydrological
  simulation. A one-ulp forcing difference is expected to be immaterial and is
  **not demonstrated** to be. The reviewer's alternative fix — an end-to-end WF3
  run on a deliberately non-round grid, with an empirically justified output
  tolerance — is **not taken**: it needs a full stress-test run on a config no
  fixture carries, `check_baseline` cannot gate a tree it has no reference for
  (R1/R2), and a tolerance chosen from a single run would be an assertion wearing
  a number. Instead the claim is narrowed to what is measured (§7-2), and the
  exposure is stated: a project running a `step_num: 5`-class grid gets a forcing
  that differs in its last bits from what the pre-migration code would have
  imposed, with no bound on what the indicators do. This is a **migration-once**
  exposure, not a standing one — after the change there is no second code path to
  disagree with. Escalate to G2 if a project on a non-round grid is expected to
  re-run and compare across the migration.

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
  `DESIGN_COLUMNS` -> `LOOKUP_COLUMNS`. **Plus D35's domain guard**, which lives in
  this module because it protects this module's conversion: `MULTIPLIER_DOMAIN`,
  `refuse_out_of_domain_multipliers`, `MultiplierDomainError`, and its cases in the
  existing `tests/test_prepare_cst_parameters.py` (V20's widened grids, V23). The
  **call** is a parse-time one and lands in step 4 with the other parse-time call —
  named in both places because a guard defined here and never invoked there is
  exactly the one-place-stated, other-place-absent gap that cost this run a round.
- **Step 2 — the consumer of the artifact.** Rule 3.12's input and the new
  positional argument; `impose_climate_change.R`'s arity check (4 → 5), its
  percent->factor conversion for **both** columns (D3/D21), and the new
  `blueearth_cst/weathergen/read_member_grid.R` carrying D29's assertion (D34),
  sourced from the script. **Plus its negative tests** — V17's four malformed
  fixtures — which land in a new `tests/test_read_member_grid.py` rather than in
  `tests/test_r_scripts.py`, whose module docstring declares itself syntax-only
  ("`Rscript -e parse(...)` — syntax only, no evaluation, no side effects",
  `:57`); quietly evaluating R inside a file that says it does not is the same
  class of defect as a scope clause that is false. `read_member_grid.R` **sources
  nothing** — not even `global.R` — so the test is one `Rscript --vanilla -e
  'source(...)'` per fixture; it pins cwd to the repo root, since this repo's R
  scripts are sourced by repo-relative path, and carries the same
  `shutil.which("Rscript")` skipif `test_r_scripts.py` uses.
- **Step 3 — the reduction.** `export_wflow_results.py` loses `perturbation_axes`
  and `annual_perturbation` from its public surface, the `axes` map, the design
  read and the extra-axis refusal; gains the run-coverage check (D22).
  `INDICATOR_COLUMNS` in `shared/indicator_tables.py` loses two entries and
  `DESIGN_AXES` is **deleted** (its only consumer is the refusal being retired);
  rule 3.16's `input:` shrinks.
- **Step 4 — the library and its tests.** `shared/surface_axes.py` (with D28's
  `read_indicators` and the `naming.md` §5 signatures) and the parse-time call in
  `run_stress_test.smk`. **Both parse-time refusals land here**, on the two lines
  beside `refuse_retired_experiment_keys(my_cfg)`: `parse_surfaces(config)` (D13,
  now also enforcing §5.2's cross-axis distinctness rule) and
  `refuse_out_of_domain_multipliers` (D35, defined in step 1). **Plus
  `tests/test_surface_axes.py`**, which v1 omitted
  from every list: `surface_axes` is the largest new surface in the design —
  `parse_surfaces`, `month_classes`, `axis_values`, `axis_caption`, `join_axes`,
  four exception types **plus the two v4 adds** (`SurfaceMemberMismatchError`,
  D28 check b; `DuplicateAxisVariableError`, D13/§5.2),
  `DEFAULT_SURFACE`, `AXIS_COLUMN` — and none of the four
  test files §9 named as "the narrow tier" owns any of it. Under this repo's own
  ladder ("only the tests covering the file you changed"), an unnamed test file is
  an ungated file. V5–V11, V18–V19 and **V22** map onto named functions there (§9), and the
  `np.linspace` member matrices and rendered captions in §9's E10 block are
  promoted from run evidence into its fixtures. **All ten caption cases**
  (1, 1b, 1c, 2, 3, 3b, 3c, 4, 4b, 4c) are fixtures, including
  `test_caption_explicit_subset_of_all_varying` — the all-months-varying design
  with an explicit `M = JFM` that `ext1-1` names, which no v2 case covered and
  which v2 would have captioned `mean change over the year`.
- **Step 5 — contracts and inventory.** **Two** contracts, not one:
  - **WG-2** — `dev/reference/contracts/weather-generator-seam.md`: the `## WG-2`
    section (§5.7's blockquote), the `input:`-on-3.16 clause, the validator-index
    row at `:330`, and the bounded-substitution walkthrough at `:298`/`:310`.
    Code: `_WG2_HEADER` and `validate_wg2` in `shared/interchange_contracts.py`
    (header, row-count and grid-completeness mechanism, `st_0`-absent assertion,
    `ST_NUM` argument).
  - **HM-7** — `dev/reference/contracts/hydrological-model-seam.md`: the `## HM-7`
    section (§5.8's blockquote), the **deletion** of the two axis-column bullets
    at `:346-354` and `:355-368`, and the **amendment** of the
    HM-4 → HM-5 → HM-7 relational check-3 sentence at `:420-422`, which names both
    deleted columns and the deleted `_PERTURBATION_AXIS` symbol. Code:
    `validate_hm7`.

  Then `dev/scripts/semantic_tree_diff.py`; `dev/reference/workflows/rule-index.md`.
  `_PERTURBATION_AXIS` is **deleted**: verified this run that it is already
  vestigial — defined at `interchange_contracts.py:741` and referenced nowhere
  but the stale docstring at line 1141, since `validate_hm_gauge_column_identity`
  check 3 compares the `location` value SET post-CR-2 rather than subtracting
  axis columns from a wide header. Fix that docstring in the same commit; the
  relational validator's *logic* needs no change, which is what **§5.8**'s
  check-3 amendment rests on — the amendment is to the seam document's PROSE,
  which names `_PERTURBATION_AXIS` and both deleted columns, not to the validator.
- **Step 5b — sweep the test suite for the old roots, and for the old COLUMNS.**
  `AGENTS.md` records that a task moving the project tree must do this, and that
  R9 — the same migration class — left 22 failures, three of them behind an
  `os.path.exists` guard that turned a wrong path into a **silent skip**.

  **v1's sweep was `rg -n "_work/|stress_test_design" tests/`, and it was wrong in
  two ways it also nominated as its own acceptance test.** (a) The trailing slash
  it explicitly defended — "bare `_work` false-positives on `_workflow`" — is
  exactly what hid `tests/test_interchange_contracts.py:830-831`, which spells the
  path as `join(_WG_DIR, "_work", "st_1.csv")`, segments, no slash. (b) It covered
  only migration Event 1 (paths); **Event 2, the column removal, had no sweep at
  all**, and its live references keep *passing* rather than failing. Both classes
  are R10.

  The widened sweep, over three trees rather than one:

  ```
  rg -n "_work|stress_test_design|temp_change|precip_change|precip_mean|temp_mean" \
     tests/ dev/scripts/ dev/reference/ docs/notebooks/
  ```

  `_workflow` false positives are cheap and accepted; the WF2 hits
  (`tests/test_get_change_climate_proj_summary.py`, where the names are WF2
  change-factor variables) are excluded **by inspection, not by pattern**. Its
  result:

  | file | what it holds | after the change |
  |---|---|---|
  | `tests/test_prepare_cst_parameters.py` (3 sites) | writes/reads a `stress_test_design.csv` under `tmp_path` | fails |
  | `tests/test_export_wflow_results.py` (3 sites) | a design-table fixture "as 3.09 writes it", plus a re-read | fails |
  | `tests/test_interchange_contracts.py` (3 sites) | a design fixture, a `validate_hm7(design=…)` call, the real-fixture path `<exp>/config/stress_test_design.csv` | fails |
  | `tests/test_project_tree_inventory.py` (1 site) | asserts `…/weathergenr/_work/st_4.csv` classifies as declared | must flip to reporting after §8's prefix narrowing |
  | **`tests/test_interchange_contracts.py:143-160`** | `_wg2_good()`, built in memory | **PASSES** — positively asserts a retired contract holds |
  | **`tests/test_interchange_contracts.py:826-834`** | `test_wg2_integration`, real-fixture `_work/st_1.csv` with a `cst_1.csv` legacy fallback | **SKIPS** behind `_FIXTURE_ABSENT` — R9's silent class |
  | **`tests/test_check_baseline_scope.py:56`** | the literal seven-column header | **PASSES** while asserting a dead shape |
  | **`tests/test_check_baseline_indicator.py:62-63,244,268,272`** | a five-site fixture incl. `assert df.loc[0, "temp_change"] == "1.3000000000000003"` | **PASSES**; `test_float_key_columns_are_compared_as_written` exists *because* `temp_change` is a float key column, and after the change every remaining key column is non-numeric, so its stated subject no longer exists |
  | **`docs/notebooks/Climate Stress Test.ipynb`** (4 sites) | lines 352, 481-482, 500, 683-685 | **`KeyError` on execution** — step 6 |

  The last five rows are what a paths-only sweep cannot reach, and four of them
  are **false greens** rather than failures — which is why the exit condition
  changes too. v1's was "a non-empty result is the failure"; that cannot work for
  a test whose stale content is a literal. The exit condition is: **the widened
  sweep returns only sites an inspector has classified as WF2 or as
  deliberately-historical**, and `_member_artifact`'s legacy `cst_1.csv` fallback
  (`:831`) moves or retires with whatever replaces `test_wg2_integration`.

  The fixture-dependent layer uses the *named* guard form (`_FIXTURE_ABSENT` +
  `skipif`), so its absence is reported rather than silent — but it still **cannot
  run in a worktree** (R3), so this step is gated in the primary checkout.
- **Step 6 — docs, seeds and the notebook.**
  `config/templates/snake_config.template.yml` gains a **commented-out**
  `reporting:` block; `dev/reference/indicator-glossary.md`; and the
  `read_indicator_table` docstring in `dev/scripts/check_baseline.py`, which names
  `temp_change`/`precip_change` as its reason for string-parsing every non-`value`
  column and goes stale in this commit.

  **The `test_case/snake_config_*.yml` seeds declare NO `reporting:` section**, and
  that is a decision rather than an omission (R8): a seed that declares a surface
  arms the `yaml` baseline target, so every later caption edit on the baseline
  tree would turn `check_baseline.py check` red. Unset, they exercise
  `DEFAULT_SURFACE`, which D11 makes behaviour-preserving on a uniform design —
  so the seeds still cover the path.

  **`docs/notebooks/Climate Stress Test.ipynb` is rewritten in this commit** (R6),
  not deferred and not merely re-rendered: it reads the retired artifact and
  groups on the retired columns, so it raises `KeyError` after step 3.

  **The rewrite is a CONTRACT-BASED external-consumer example, and it imports
  nothing from `surface_axes`.** v2 specified the opposite — `read_lookup` +
  `read_indicators` + `join_axes` + `axis_caption` — which is an in-repo caller,
  the thing the G1-return ruled out at Fork B and which D15, alternative 6.9 and
  R9 all state does not exist. `ext1-2` filed it blocking and it is: the two
  instructions cannot both be followed, and an implementer would have had to pick
  one silently. The notebook therefore does what an R or JavaScript consumer must:

  1. `pd.read_csv(lookup_path, dtype={"st_id": str})` and the same for the
     indicator table — the WG-2 half, including why the dtype is not optional;
  2. classify months, take `M`, collapse with the weighted mean and its
     exact-equality short-circuit, and build the caption — the HM-7 half,
     transcribed from the contract text;
  3. re-pad the indicator `st_id` to the lookup's width, partition `st_0` out,
     join, and pivot to the surface.

  Roughly fifteen lines, and they are the point rather than the cost: the notebook
  becomes the **only in-repo evidence that HM-7 is complete enough to
  re-implement from**, which is precisely the compensating requirement Fork B
  attached to accepting no caller. V21 checks it once against the reference
  implementation. It is *not* a demonstration of the library's API — under Fork B
  there is no in-repo demonstration of that, and R9 says so.

  The two remaining notebooks and the re-render pass stay with `t2608132100`.
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

**The `naming.md` §7 record**, drop-in at
**`dev/milestones/r12/migration_stress-test-lookup.md`**. v1 carried the path as
an open question on the premise that this work lands before R12 and so has no
milestone directory. That premise is **false on disk**: `dev/milestones/r12/`
already exists and holds `g2-assessment.md`. §7's grammar is satisfied by an
existing directory, so no directory is created early, no new grammar is needed,
and no owner ruling is required — OQ-1 closes on a fact (§10). Filing it under
`r12/` is also correct on substance: S8 makes this design R12's prerequisite, so
the migration belongs to R12's record even though it lands first.

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
> **Event 3 — a unit change on the parameter grid.** `precip_mean` and
> `precip_variance` crossed the Python→R seam as **multipliers** and now cross as
> **percent** (`precip_change`, `precip_variance_change`). The R side
> reconstructs `1 + <col>/100` for both. The reconstruction is within one
> `float64` ulp of the pre-migration level and is bit-identical for every level in
> every shipped config; it is not exact in general, and cannot be made so.
>
> **Machinery updated in the same commit:** `prepare_cst_parameters.py`,
> `export_wflow_results.py`, `shared/indicator_tables.py`,
> `shared/interchange_contracts.py` (`validate_hm7`, **`validate_wg2`**,
> **`_WG2_HEADER`**, `_PERTURBATION_AXIS`), `shared/surface_axes.py` (new),
> `weathergen/impose_climate_change.R`, **`weathergen/read_member_grid.R`** (new —
> the member slice and its twelve-ordered-months assertion),
> `run_stress_test.smk` (rules 3.09 / 3.12 /
> 3.16 and `WF3_TARGETS`), `dev/scripts/semantic_tree_diff.py`,
> `dev/scripts/check_baseline.py` (docstring),
> **`dev/reference/contracts/weather-generator-seam.md`**,
> `dev/reference/contracts/hydrological-model-seam.md`,
> `dev/reference/workflows/rule-index.md`,
> `config/templates/snake_config.template.yml`,
> **`docs/notebooks/Climate Stress Test.ipynb`**, and the test sweep of §8 step 5b
> including **`tests/test_surface_axes.py`** and
> **`tests/test_read_member_grid.py`** (both new).
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
| V5 | A uniform design's axis **equals** today's, exactly | derived axis != `annual_perturbation` output on a flat vector | `test_surface_axes.py::test_uniform_axis_matches_annual_perturbation` |
| V6 | A JFM design reports the imposed value | derived axis returns +7.6% for a +30% JFM member | `test_surface_axes.py::test_seasonal_axis_reports_imposed_value` |
| V7 | Heterogeneous varying months are refused at the axis | `axis_values` returns a value instead of `HeterogeneousAxisError` | `test_surface_axes.py::test_heterogeneous_axis_refused` |
| V8 | A held month in `M` is refused **on a non-degenerate axis** | `axis_values` returns instead of `HeldMonthInAxisError` | `test_surface_axes.py::test_held_month_refused` |
| V9 | The axis is evenly spaced | `NonRectilinearAxisError` not raised on a deliberately non-affine collapse | `test_surface_axes.py::test_rectilinearity_postcondition` |
| V10 | A degenerate axis annotates rather than raises, **including under an explicit `months:`** | a temp-only design raising on its precip axis, by either the default or an explicit set (D27) | `test_surface_axes.py::test_degenerate_axis_admits_explicit_months` |
| V11 | Captions match the **ten** cases, and the leading phrase names `M` in every non-degenerate one | rendered caption != the §5.5 / §5.8 table; specifically, an all-months-varying design declared `months: [1,2,3]` captioned `mean change over the year` (D31) | `test_surface_axes.py::test_caption_cases` + `::test_caption_explicit_subset_of_all_varying` |
| V12 | A relabel does not trip the freeze, move the digest, or re-fire 3.16b | `_frozen_differences` non-empty, digest moved, or 3.16b's params changed after a `reporting:` edit | probe P2 (below) |
| V13 | Every lookup member appears in every table, and `st_0` appears in the tables but not the lookup | `validate_hm7` green while a member is missing | `test_interchange_contracts.py` |
| V14 | Stale `_work/` and `stress_test_design.csv` report | either classifies IDENTITY | `tree-check`, probe P3 |
| **V15** | **WG-2 validates the lookup, not a deleted file** | `validate_wg2` **not** green on a valid `12 × ST_NUM` lookup, or green on a table containing an `st_0` row, or green on a `(st_id, month)` grid with a gap, or green on a table mixing `st_id` widths (D33) | `test_interchange_contracts.py` (rewritten `_wg2_good()` + a failing twin per clause) |
| **V16** | **The R side reconstructs the pre-migration factors** | for one member, `1 + precip_change/100` / `1 + precip_variance_change/100` differing from that member's pre-migration `st_<m>.csv` values by more than one `float64` ulp | one-off at step 7, beside V4 |
| **V17** | **The R side refuses a malformed member slice, loudly and by member** | `read_member_grid` **returning** instead of `stop()`ing on any of three fixtures — a member missing month 7; a member carrying month 7 twice; a token matching no row (`"9"` against a four-member lookup) — where each must exit **nonzero** with the member token in the message. Plus two positive fixtures, distinct files: **(i)** a well-formed lookup, whose slice returns 12 rows in month order; **(ii)** the same lookup with that member's rows **shuffled**, which must return the identical frame in month order rather than stopping — D21 sorts before asserting, so unordered input is normalised, not rejected | `tests/test_read_member_grid.py` — three negative + two positive `Rscript -e 'source(...)'` executions, cwd pinned to the repo root, `Rscript` skipif (D34) |
| **V18** | **`join_axes` refuses a mis-keyed partition, and an INCOMPLETE one** | `join_axes` returning an empty `surface_df`, or classifying a non-baseline `st_id` as baseline, instead of `BaselinePartitionError`; **and — added at v4 (`ext2-1`) — `join_axes` returning a `SurfaceJoin` instead of `SurfaceMemberMismatchError` when one valid lookup member is absent from the indicator table**, the case checks a and c both pass (D28 check b). The missing-member fixture is the load-bearing one: it is a plausible-looking surface, not a visibly broken one, and its diagnostic must name the missing ids | `test_surface_axes.py::test_partition_postcondition` + `::test_missing_lookup_member_refused` |
| **V19** | **The flat-vector short-circuit holds in percent space** | derived axis != the imposed value on a homogeneous 0.6–1.4 `step_num: 3` grid, where the weighted mean is a ulp off (D26) | `test_surface_axes.py::test_flat_vector_short_circuit` |
| **V20** | **A non-round grid's forcing is reconstructed within the stated bound, ACROSS THE WHOLE ADMITTED DOMAIN** — a claim about the MULTIPLIER only, and about no indicator (§7-2, R13) | reconstructed multiplier differing from the `float32` level by more than one `float64` ulp on a 0.6–1.4 `step_num: 3` grid (D25) — the case the shipped configs cannot exercise. **Widened at v4** (`ext2-3`): also on a grid anchored at the domain **floor** (`0.5 → 1.5`), and at the percent-binade crossings inside the domain where the error is *attained* rather than slack — levels near `0.5, 0.68, 1.32, 1.64, 2.0` (D35, E16). Random draws inside the domain are **not** an acceptable substitute: they are the evidence shape `ext2-3` faulted | `test_prepare_cst_parameters.py` |
| **V22** | **The parser refuses two axes declaring the same variable** (`ext2-2`) | `parse_surfaces` accepting `x: {variable: temp}, y: {variable: temp}` — or, the twin that must NOT be refused, rejecting the legal orientation reversal `x: {variable: precip}, y: {variable: temp}` | `test_surface_axes.py::test_duplicate_axis_variable_refused` + `::test_orientation_reversal_admitted` |
| **V23** | **The parser refuses an out-of-domain multiplier bound, before the DAG is built** (`ext2-3`, D35) | `refuse_out_of_domain_multipliers` admitting a `precip.mean` **or** `precip.variance` vector with any element below `0.5`; or refusing any of the four shipped seed configs and `snake_config.template.yml`, all of which are inside the domain; or a `temp` vector being refused, which carries no domain at all. The refusal must reach `--dry-run`, so `pytest tests/test_cli.py` is the second half of this claim | `test_prepare_cst_parameters.py::test_multiplier_domain_refused`, `test_cli.py` |
| **V21** | **HM-7 is complete enough to re-implement from** | the rewritten notebook's contract-transcribed derivation differing from `surface_axes` on any member's axis value, or rendering a different caption, on the rapid experiment | one-off at step 6, beside V4/V16 — run by the migrating engineer in a scratch session against a library run. The **notebook imports nothing** from `surface_axes` (§8 step 6); the comparison does, the way a test does, which is what R9 already admits |

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
   groups, which is the claim.

**What a failing group means — corrected.** v1 pre-committed the implementer to
"the forcing moved and D7's float32 round-trip discipline was not preserved". That
diagnosis is wrong on its face for a general grid, because **the arithmetic this
design specifies is itself a sufficient cause** of a small move (D25). The
corrected rule: a failing group means the forcing moved by **more than one
`float64` ulp of the perturbation level**, which D25's conversion cannot explain
and a lost quantization discipline can. On `snake_config_baseline.yml`
specifically the distinction is moot — its grid levels are in D25's
exactly-invertible set — and **that is precisely why V4 alone is not enough**: run
once on that config, it structurally cannot observe the conversion residual at
all. **V20 covers the gap with a non-round grid**, and V16 checks the
reconstruction directly against a pre-migration member file rather than through
eleven indicators.

The trimmed copy is a **one-off comparison input, not a stored artifact** — step
7's re-record replaces the reference outright.

### Gates

| gate | verdict |
|---|---|
| `pytest tests/test_prepare_cst_parameters.py`, `test_export_wflow_results.py`, `test_interchange_contracts.py`, `test_stress_test_grid.py`, **`test_surface_axes.py`** (new), **`test_read_member_grid.py`** (new), **`test_check_baseline_indicator.py`**, **`test_check_baseline_scope.py`** | runnable — the narrow tier. Four were missing from v1's list: `surface_axes` had no owning test file at all, the R-side guard had no executable falsifier until D34 extracted it, and the two `check_baseline` files pin the retired seven-column header by literal, so they pass while asserting a dead contract (R10). `test_read_member_grid.py` **skips without `Rscript`**, so it is a real gate only inside the pixi env — named here rather than assumed |
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

So the top-level home does not trip the experiment freeze and does not move the
effective-config digest, while the `run_stress_test` home does both. The
`ancient()`/no-`params:` trap (E5) is not inherited, because no rule reads the
declaration.

**P2-b — what P2 did NOT measure, added at v2.** Both `risk-9` and
`architecture-4` are right that P2 measured **repo digest helpers**
(`effective_config_digest`, `_frozen_differences`), not **Snakemake's rerun
triggers**, so the executed probe never reached the layer §7 consequence 1 was
about. The missing layer, established by reading the rules and the fingerprinter
rather than by another synthetic — these are structural facts, not timing
questions:

| question | answer | source |
|---|---|---|
| Does a `reporting:` edit re-fire rule 3.02? | **Yes**, by mtime | `run_stress_test.smk:604-605` — `config_snake = config_path`, a plain input, not `ancient()` |
| Does that cascade in the DAG? | **No.** No rule in any `*.smk` declares `snake_config_run_stress_test.yml` or `run_record.yml` as an input; `:535` is a `WF3_TARGETS` entry and `:620-621` the outputs, and that is the whole set | grep over `*.smk` |
| Does the snapshot then carry `reporting:`? | **Yes**, verbatim | `copy_config_files.py:222` — `shutil.copyfile` |
| Does that move a baseline fingerprint? | **Yes** | `check_baseline.py:326-329` carries the snapshot as a `yaml` target; `fingerprint_yaml:449-455` `yaml.safe_load`s the **whole unprojected document** and hashes canonical JSON of it |
| Does 3.16b re-fire? | **No** — its `params:` carry only the two digests | `run_stress_test.smk:1146-1148` |

Consequences: §7-1 loses "re-fires 3.02" as a falsifier (it fires on correct
behaviour), gains 3.16b as the discriminator it actually is, and the design gains
risk **R8** plus the step-6 decision that the shipped seeds declare no
`reporting:` section.

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

### Measured at v2 — the arithmetic the panel disputed

Four things v1 asserted and v2 measured. All are reproducible from the formulas
stated in D25 and D26; none needs a fixture.

- **E11 — the percent round trip is not exact, and cannot be made exact.**
  Over 200,000 random `float32` multipliers in [0.5, 1.6]: `1 + p/100` fails to
  reproduce the level in **19.9%** of cases, `(100 + p)/100` in **32.9%** — so
  `risk-1`'s suggested inverse spelling is the **worse** one and is not adopted.
  Allowing a search over neighbouring `float64` percents, **1,155 of 50,000**
  levels admit **no** exact solution under `1 + p/100` (6,778 under the
  alternative), so the "search at write time" fix both reviewers proposed is
  unattainable. Worked failures: 0.7–1.3 at `step_num: 5` (`0.82` →
  `-18.0` → `0.8200000000000001`) and 0.6–1.4 at `step_num: 3` (`1.1333333` →
  `13.33333` → `1.1333332999999999`). `step_num` 1, 2, 4, 6 and 8 on 0.7–1.3 are
  exact at every level, which covers every shipped config. **→ D25.**
- **E12 — writing the percent as `float32` would cost eight orders of
  magnitude.** Worst relative |Δ| in the reconstructed multiplier is **5.98e-08**
  (one `float32` ulp) when the percent is re-quantized, against ~1e-16 at
  `float64` shortest-repr text. This is why D25 rule 1 departs from "keep writing
  `float32`" — the *grid* stays coarse, the *text* must be fine. **→ D25.**
- **E13 — the flat-vector short-circuit matters in percent and °C, and not in
  multipliers.** Weighted mean over the noleap month lengths differs from twelve
  identical inputs in **0/50,000** random `float32` multipliers in [0.5, 1.6],
  **97,628/200,000** random percents in [-90, 200], and **48,294/100,000** random
  `float32` °C in [0, 6]. So the unit change *creates* the exposure v1 called
  immaterial, and D16 makes the flat case the normal path. Realistic grid hits:
  `-13.33333` → `-13.333330000000002` (0.6–1.4, `step_num: 3`) and `35.71428` →
  `35.71428000000002` (0.5–1.5, `step_num: 7`). **→ D26.**
- **E14 — the `reporting:` section is inside the baseline's `yaml` fingerprint.**
  `fingerprint_yaml` `yaml.safe_load`s the whole document unprojected and hashes
  canonical JSON of it, and the snapshot is a `shutil.copyfile`. Read, not
  inferred. **→ R8, §8 step 6.**

### Measured at v3 — the caption defect's magnitude

- **E15 — v2's caption case 1 mislabels an all-varying design by a factor of
  two.** Twelve varying months, JFM at `0.7 → 1.3` and Apr–Dec at `0.9 → 1.1`,
  `step_num: 2`, declared `months: [1,2,3]`. Computed over the noleap month
  lengths: the JFM axis is **−30 / 0 / +30**, the annual collapse of the same
  members is **−14.931507 / 0 / +14.931507**. v2's rule (case 1 fires when `H` is
  empty) captions the first `mean change over the year` — a statement about a
  quantity that differs from the plotted one by 2.01×, on a declaration D16
  explicitly admits. Closed form, reproducible from the formulas in §5.5 with
  `np.linspace(min_vec, max_vec, 3, axis=1)` and `np.average` over `[31, 28, 31,
  …]`; no fixture. **→ D31, §5.5 cases 1b/1c, V11.**

### Measured at v4 — where the one-ulp bound actually holds

- **E16 — the reconstruction bound holds for `multiplier ≥ 0.5`, unboundedly
  above, and fails only downward.** Executed with D25's conversion verbatim
  (`level = float(str(np.float32(v)))`; `text = repr(level*100-100)`;
  `back = 1 + float(text)/100`), reporting `|back − level| / ulp(level)`.

  **Dense `float32` sweeps** — every representable value in a ±2e-5 band — across
  every percent-binade crossing in the domain, which is where the error is
  *attained* rather than slack, and across the level binades:

  | probed | worst |
  |---|---|
  | levels near `0.5` (level binade 2⁻¹, the floor) | **1 ulp** |
  | percent crossings `−32, −16, −8, −4, −2` (levels 0.68, 0.84, 0.92, 0.96, 0.98) | **1 ulp** |
  | percent crossings `+2 … +16` (levels 1.02–1.16) | **0 ulp** |
  | percent crossings `+32, +64, +128` (levels 1.32, 1.64, 2.28) | **1 ulp** |
  | level binades `1.0`, `2.0`, `4.0` | **1 ulp** |

  **Random confirmation inside the domain:** worst 1 ulp over 300k `float32` draws
  in each of `[0.5, 10]`, `[0.5, 1e3]` and `[0.5, 1e6]` — so there is no upper
  bound to impose, which is why D35 imposes none.

  **Below the floor**, log-uniform draws, 200k each: `[0.36, 0.5)` still **1**;
  `[0.25, 0.36)` **2** — the first failure, at the `|percent| = 64` crossing the
  mechanism table in D35 predicts; `[0.05, 0.25)` **18**; `[0.01, 0.05)` **72**;
  `[0.001, 0.01)` **574**. The reviewer's counter-example reproduces exactly:
  level `0.013596006` → text `-98.6403994` → `0.013596005999999883`, **68 ulps**.
  So the floor is conservative by roughly one binade, and the refusal fires at
  parse time — before any partial experiment exists.

  Recorded this way on purpose: `ext2-3`'s actual complaint is a **normative claim
  standing on a domain-restricted random sweep**, and answering it with a second
  random sweep would repeat the defect. The argument is the mechanism table in
  D35 — `ulp(|percent|)` versus `ulp(level)` — and these sweeps confirm it at the
  points the mechanism identifies as extremal. Closed form and reproducible from
  the formulas in D25; no fixture.

  **Config side, verified rather than assumed — the half that makes the refusal
  safe to add.** All four shipped seeds (`snake_config_rapid.yml`,
  `snake_config_baseline.yml`, its `_linux` twin, `snake_config_wf2_fast.yml`) and
  `config/templates/snake_config.template.yml` declare `precip.mean` `0.7 → 1.3`
  and `precip.variance` `1.0 → 1.0`, so every one is inside D35's domain and none
  trips the new parse-time refusal. `test_cli.py` dry-runs all four entry points
  and §9 marks it **required**, so a domain that excluded a shipped config would
  turn a required gate red — which is why this was checked before the refusal was
  specified rather than after. **→ D35, WG-2 `precision:`/domain bullets,
  V20, V23.**

**Evidence still carried as verified-elsewhere** (E1–E8, `intake.md`): the cache
mechanics at `prepare_cst_parameters.py:175-189`; the +7.6% arithmetic; WF2's
percent units; rule 3.09 declaring all member files as one job's outputs; rule
3.09's `ancient()`/no-`params:` deafness; weathergenr 1.2.0's non-identity at
unit factors; the measured `st_0` -> identity-member indicator differences
(magnitudes provisional — the fixture predates the 1.2.0 rename); and R12's
`member_hash` indexing members by the annual collapse this design deletes.

## 10. Open questions

- **OQ-1 — CLOSED at v2, on a fact rather than a ruling.** It asked where the
  migration note lands, on the premise that this work carries no milestone
  directory. The premise was **false on disk**: `dev/milestones/r12/` already
  exists and holds `g2-assessment.md`. The note files at
  `dev/milestones/r12/migration_stress-test-lookup.md` (§8); `naming.md` §7's
  grammar is satisfied by an existing directory, nothing is created early, and no
  owner ruling is needed. Retained as a closed entry rather than deleted, because
  it was carried to G1 unruled and the closure is the record of why it stopped
  needing one.
- **OQ-2 — Q6, the projection overlay.** Deferred deliberately. The constraint is
  pinned (S10) and the arithmetic is cheap because WF2 already emits monthly
  factors in percent, so the same collapse runs over both tables with no unit
  conversion. `reporting:` is where its declaration will live. Three things Q6
  must decide, sharpened at v2 so it defers known questions rather than unknown
  ones:
  1. **The transfer is arithmetic, not semantic.** §5.8 now carries the caveat:
     on the lookup side the collapse averages equal values and reports an imposed
     change; on the GCM side it averages unlike months and reports a summary, and
     it moves non-affinely with `M` (R12).
  2. **Whether the overlay is recomputed per declared surface, and whether that
     recomputation is recorded anywhere.** D10 makes `M` a freely editable
     per-surface choice and D8 puts it outside run identity, so N surfaces place
     the same GCM cloud N ways with no run-identity trace.
  3. **Which implementation survives.** `dev/scripts/preview_wf2_projection_plots.py:299-302,319-322,364-367`
     already computes `precip_change`/`temp_change` off an annual reference and
     plots GCM dots in exactly this axis space. Q6 must reconcile it with §5.8's
     formula rather than discover it.
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

- **v2 (2026-08-15)** — revision r1, answering the internal panel's 26 findings.
  Dispositions and per-finding argument: `ledger.md`.

  **Both blocking findings closed** (`architecture-1`, `repo-fit-1`): v1 deleted
  the artifact WG-2 pins and wrote replacement text for HM-7 only. §5.7 is now a
  **WG-2 replacement**, and per the G1-return ruling on Fork A the lookup's schema
  is normatively defined **there** — the seam the artifact actually crosses
  (Python → R) — with HM-7 (§5.8) referencing it. **D30** records that WG-2 keeps
  its id rather than being retired or renumbered. §8 step 5 carries
  `weather-generator-seam.md`, `_WG2_HEADER`, `validate_wg2` and the WG-2 test
  sites; §8 step 5b widens the sweep to reach them.

  **The G1-return ruling on Fork B — accept no in-repo caller — is implemented as
  a completeness obligation on both contracts** (D15): WG-2 owns the schema, HM-7
  owns the axis derivation *including* the classification threshold, the
  degenerate-axis rule, the short-circuit and the caption case table, so an
  out-of-repo re-implementer's document is complete. The gap itself is named as
  risk **R9**, and alternative **6.9** records why the rule-3.16 caller was
  rejected.

  **Mechanism changes, not wording** — these are the ones a later round should
  re-read rather than assume unchanged:

  | id | change |
  |---|---|
  | **D25** | replaces D7's "the forcing is bit-identical" with a **measured bound**. Percent text at `float64` shortest repr (not `float32`); inverse pinned to `1 + p/100`; exactness shown **unattainable** (E11), so both reviewers' fix option (b) is rejected on measurement |
  | **D26** | the flat-vector exact-equality short-circuit becomes normative, in the contract text; the unit change moves the exposure from ~0% to ~49% (E13) |
  | **D27** | precedence between D16 and D19 — classify first; a degenerate axis short-circuits both the subset and the homogeneity rule |
  | **D28** | `join_axes` gains a partition postcondition and `BaselinePartitionError`; the library owns the indicator read and normalises both keys |
  | **D29** | the R side asserts twelve ordered months after the filter, and the arity moves 4 → 5 |
  | **D30** | WG-2 keeps its id and re-points; `validate_wg2`'s row check becomes `12 × ST_NUM` plus grid completeness plus `st_0`-absent |
  | D3 | the conversion is stated over the percent **columns**, adding the missing `precip_variance` inverse — a variance factor of **zero** on every shipped config, as v1 was written |
  | D11 | one threshold (`max − min > 0`, exact), replacing `> 0` / `> tol`; the D17 → D19 citation fixed |
  | D9 | `DEFAULT_SURFACE.id` `annual` → `default` |

  **Corrections to premises that were false**, each of which had a decision
  standing on it: D14's "there is no in-repo consumer" (the notebook is one, which
  strengthens rather than weakens the library choice, and makes R6 a **rewrite in
  the migration commit** rather than a deferred re-render); §7-1's "re-fires 3.02"
  falsifier, which fires on correct behaviour; §5.8's scope clause, now a
  per-passage disposition after `architecture-7` caught three live passages v1
  called unchanged; §3's S1 rationale, which claimed D7 resolved the unit
  round-trip question.

  **Measured this revision:** E11–E14 (the round-trip rate and its
  unattainability, the `float32`-percent cost, the flat-vector rate in three
  spaces, the `yaml` fingerprint's scope) and probe **P2-b**, the rerun-trigger
  layer P1–P3 never reached. **OQ-1 closed on a fact** — `dev/milestones/r12/`
  already exists. Everything v1 got right is carried: P1/P2/P3's results, E9/E10's
  settlements, and S1–S11 unaltered.

  New risks **R8–R12**; **V15–V20** added to the claim → falsifier table and
  V5–V11 given named test functions in `tests/test_surface_axes.py`, which v1
  omitted from every list. Size budget raised 1,250 → 1,600 lines for the second
  contract section (header).

- **v3 (2026-08-15)** — revision r2, answering external round 1's six findings
  (`external-review-r1.md`, reviewer `gpt-5.6-sol`, clean-room on `design-v2.md`).
  Dispositions: `ledger.md`, rows `ext1-1` … `ext1-6`. All six **accepted**; none
  deferred, none rejected, so nothing goes to owner arbitration.

  **Both blocking findings closed.**

  `ext1-1` — **the caption made a false statement about the plotted quantity.**
  v2 selected its uniform case on `H` being empty, so a design whose twelve months
  all vary but which declares `months: [1,2,3]` was labelled
  `mean change over the year` while its axis reported the JFM change. Measured
  (E15): −30 / 0 / +30 on the axis against −14.93 / 0 / +14.93 annually, a factor
  of two, on a declaration D16 explicitly admits and which D16's own refusal
  message tells the user to write. **D31** derives the leading phrase from `M` in
  every non-degenerate case, adds cases **1b** and **1c** for varying months
  outside `M`, and states the clause builder **once** for both held and
  excluded-varying months instead of twice. Mirrored into §5.8, since the
  re-implementers read that and not §5.5.

  `ext1-2` — **the design contradicted a settled ruling and itself.** v2's §8
  step 6 made the notebook call `read_lookup` + `read_indicators` + `join_axes` +
  `axis_caption`, while D15, alternative 6.9 and R9 all state the library has no
  in-repo caller — the G1-return's Fork B. The ruling stands; the notebook is
  realigned as a **contract-based external-consumer example importing nothing
  from `surface_axes`**, which is also the only in-repo exercise of Fork B's
  compensating requirement that the contract text be re-implementable. §5.3
  reason 1, §5.8's D15 note, alternative 6.9, R6, R9, §8 step 6 and V21 all carry
  it, so the no-caller claim is now consistent everywhere it appears.

  **Mechanism changes, not wording** — the list a later round should re-read
  rather than assume unchanged:

  | id | change |
  |---|---|
  | **D31** | the caption's leading phrase derives from `M`, always; two new cases (1b, 1c); one clause builder applied to held **and** excluded-varying months |
  | **D32** | a degenerate axis bypasses step 3's *constraints*, not its *formula* — its scalar is the same weighted collapse over `M`, with D26's short-circuit. v2 named no value for a multi-offset degenerate axis |
  | **D33** | `derive_axis` → `AxisResult` and `join_axes` → `SurfaceJoin`, carrying `degenerate`, the effective `M` and the key width; the join width is **inferred from the lookup** rather than taken as `index_width(ST_NUM)`, which a module declared free of `snake_utils` could not reach |
  | **D34** | the R-side read-filter-assert block is extracted to `weathergen/read_member_grid.R`, which is what makes V17's negative executions possible at all |
  | §7-2 | the one-ulp bound is stated for the **multiplier**; indicator equality is claimed only where the forcing is bit-identical. New risk **R13** |
  | V17 | four fixtures and a positive twin in a new `tests/test_read_member_grid.py`, replacing "a WF3 run on the valid rapid config" — which was green whether the guard existed or not |
  | V15 | first clause inverted: **not** green on a valid lookup |

  **`ext1-4` deserves its own note**, because it is the finding most easily
  answered wrongly. v2 wrote a bound on a *forcing parameter* and then spent it as
  a bound on *indicator outputs*. Nothing supports that: quantile mapping,
  occurrence and intensity thresholds, caps and floors, and a hydrological
  simulation sit in between. This run's own origin is the precedent — it began
  because two code paths were assumed equivalent on exact parameters and
  measurement showed the transform moving low-flow indicators by a factor (E6,
  E7). The reviewer's alternative fix (an end-to-end non-round-grid run with an
  empirical tolerance) is **not taken**, and R13 says why: it needs a full WF3 run
  on a config no fixture carries and a tolerance that one run cannot justify. The
  measurements stay; only the claim narrows.

  **Measured this revision:** E15 (closed form). Everything earlier is carried
  unchanged: P1/P2/P3 and P2-b, E9–E14, D25's and D26's measured rates, S1–S11,
  and all 26 internal-panel dispositions. Size budget raised 2,150 → 2,600 lines,
  accounted in the header against the measured +482 rather than an estimate.

- **v4 (2026-08-15)** — **arbitration revision** (stage 6a), answering external
  round 2's three findings (`external-review-r2.md`, reviewer `gpt-5.6-sol`,
  non-clean-room with a regression duty, on `design-v3.md`). Dispositions:
  `ledger.md`, rows `ext2-1` … `ext2-3`.

  **Round 2 was the cap, so no reviewer verdict names this version.** The three
  fixes land under the **owner's arbitration of 2026-08-15** (`status.md`
  § arbitration), which accepted all three, required a fix for each, and ruled the
  *shape* of `ext2-3`'s. That is the authority for v4 and it is recorded here
  rather than left to the run log, because a reader of this document otherwise
  finds three mechanism changes with no verdict behind them.

  **Both blocking findings closed.**

  `ext2-1` — **the report-time partition check was one-directional.** D28 required
  that the ids *absent* from the lookup be exactly the baseline and that the
  surface be non-empty, and neither constrains the *missing* direction: a stale or
  partial indicator table whose members are a strict subset of the lookup's
  satisfies both and yields a silently incomplete surface — holes in the grid, or a
  bias if the missing members sit at one end. D28 rule 2 becomes **three ordered
  checks** with **set equality** between the lookup's members and the non-baseline
  indicator members as the new middle one, raising a new
  `SurfaceMemberMismatchError` (not `BaselinePartitionError`, which would name the
  wrong side). Mirrored into §5.8's report-time join semantics, where the
  re-implementers read it. A by-product worth naming: §5.3's standing claim that
  "`validate_hm7` is test-time only; D28 is the report-time tier" is now **true**,
  because check b is that validator's completeness check evaluated at report time.

  `ext2-2` — **the schema admitted a declaration no implementation could serve.**
  `variable` was a closed enum *per axis* with no cross-axis rule, so
  `x: temp, y: temp` passed while `SurfaceJoin.axes` (keyed by variable) would drop
  one axis and `AXIS_COLUMN[variable]` would point both at `temp_change`. §5.2 now
  requires `{x.variable, y.variable} == {"temp", "precip"}` at parse time, with
  **orientation reversal still legal**, and records what the refusal costs. D13
  names the refusal and its error class; D33 records that keying `axes` by variable
  is total only because of it.

  **Mechanism changes, not wording** — the list a later reader should re-read
  rather than assume unchanged:

  | id | change |
  |---|---|
  | **D35** | **new.** An admitted multiplier domain, `multiplier ≥ 0.5` with **no** upper bound, over `precip.mean` **and** `precip.variance`, refused at Snakefile **parse** time by `refuse_out_of_domain_multipliers` in `prepare_cst_parameters.py`. D25's one-ulp bound becomes unconditional **inside** it |
  | D28 | rule 2 becomes three ordered checks; **set equality** added; new `SurfaceMemberMismatchError`; check c narrows to the empty-lookup case |
  | D8 | the two axes must declare **different** variables; orientation reversal stays legal |
  | D13 | `parse_surfaces` enforces the cross-axis rule and raises `DuplicateAxisVariableError` — the one constraint no per-field validator can reach |
  | D25 | the bound is stated with D35 as a **precondition**, not as a qualification |
  | §5.7 | WG-2 gains a domain bullet, stated as the precondition of the `precision:` bullet above it |
  | §5.8 | the "asserts its partition" paragraph becomes three checks, with the incomplete-join failure spelled out |
  | V18 / V20 | widened: the missing-lookup-member case; the domain floor and the percent-binade crossings where the error is *attained* |
  | V22 / V23 | **new.** The duplicate-variable parser refusal (with its must-not-refuse twin); the out-of-domain refusal, plus the requirement that it reach `--dry-run` |

  **`ext2-3` deserves its own note, because the owner ruled the shape of the fix
  and the alternative was explicitly declined.** The finding is that D25's one-ulp
  bound was normative and unqualified while its evidence covered `[0.5, 1.6]`, and
  the design admitted any positive multiplier — at level `0.013596006` the error is
  **68 ulps**, and it reaches **574** near `0.0016`. The two available repairs were
  to **impose a validated domain** or to **domain-qualify the bound**. The second
  was **declined**: WG-2 makes this a cross-language contract, and a bound with an
  escape clause is one an R or JavaScript re-implementer has to re-derive. So D35
  imposes the domain and the bound stays unqualified inside it.

  Two things about D35 that are decisions rather than measurements. It applies to
  **both** percent-converted columns, because D3's v2 correction was precisely that
  the conversion is a rule over the percent columns and a `mean`-only domain would
  re-create `architecture-2`'s defect one layer up. And it has **no ceiling**: the
  bound was measured to hold to `1e6`, so a cap would refuse configurations the
  arithmetic serves and would tell a re-implementer the bound fails where it does
  not. The floor is `0.5` because that is a `float64` binade boundary one full
  binade above the first 2-ulp level (~0.36, the `|percent| = 64` crossing) — a
  mechanism, not a curve fit.

  **R13 is unchanged and deliberately so.** D35 bounds the reconstructed
  *multiplier*; R13 is about *outputs*, which nothing here bounds and the domain
  does not touch.

  **Measured this revision:** **E16** — dense `float32` sweeps at every
  percent-binade crossing in the domain (1 ulp), random confirmation to `1e6`
  (1 ulp), and the degradation below the floor (2 → 18 → 72 → 574 ulps) — plus the
  config-side verification that all four shipped seeds and the template are inside
  the domain, so the new refusal cannot turn `test_cli.py` red. Everything earlier
  is carried unchanged: P1/P2/P3 and P2-b, E9–E15, S1–S11, and all 32 prior
  dispositions. Size budget raised 2,600 → 2,950 lines, accounted in the header
  against the measured delta rather than an estimate.
