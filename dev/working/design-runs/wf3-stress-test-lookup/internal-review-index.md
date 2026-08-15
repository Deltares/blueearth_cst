# Internal review index — design-v1.md

Driver-authored aggregation of stage 2. Groups findings **by concern**; preserves
every original ID, severity and text **by reference**. Nothing here is deleted,
merged or re-graded — the per-lens files are authoritative, and a finding is
dispositioned at *its own* filed severity.

**Do not paraphrase from this file.** The grouping is by concern, so a claim
appearing under one group routinely originates in a different lens's finding.
Cite `internal-review-<lens>.md` for text.

## Verdicts

| Lens | Role | Verdict | doc_version | blocking | major | minor |
|---|---|---|---|---|---|---|
| risk | `critical-thinker` | `revise` | design-v1.md | 0 | 5 | 5 |
| architecture | `cst-architect` | `revise` | design-v1.md | 1 | 6 | 3 |
| repo fit | `python-engineer` | `revise` | design-v1.md | 1 | 2 | 3 |
| **total** | | | | **2** | **13** | **11** |

26 findings. All three lenses returned schema-valid verdicts naming
`design-v1.md`; no `IN_PROGRESS` placeholder survived.

## Groups

### G1 — the WG-2 contract is deleted and never replaced *(2 blocking)*

`architecture-1` (blocking, §5.7 / §8) · `repo-fit-1` (blocking, §5.7)

Two lenses independently. `<exp>/climate/weathergenr/_work/st_<m>.csv` is the
subject of WG-2 in `dev/reference/contracts/weather-generator-seam.md`, with its
own header constant, validator (`validate_wg2`) and tests; §5.7 replaces HM-7
only, and the strings "WG-2" / "weather-generator-seam" / "validate_wg2" appear
nowhere in the design. `architecture-1` carries a **second limb the other does
not**: seam *placement* — the lookup is the Python→R parameter handoff, i.e. the
weather-generator seam by definition, while the design puts its full schema in
HM-7, whose declared consumer is the CST-API/GUI. See § Gate return.

### G2 — D7's "bit-identical forcing" claim is false in general *(2 major)*

`risk-1` (major, §5.1 D7 / §7-2 / §9 V4) · `architecture-3` (major, same sections)

Independently measured, same counter-example: `0.82 → -18.0% → 0.8200000000000001`.
`risk-1` adds that §9's V4, run once on `snake_config_baseline.yml`, structurally
cannot see it (the shipped 0.7/1.0/1.3 grids all round-trip) and that V4 step 3
pre-commits the implementer to the wrong diagnosis. `architecture-3` adds a rate:
~17% of random float32 multipliers fail, and names the affected grid levels
(6/8/10/11).

### G3 — the relabel-is-free claim, and what P2 did not measure *(2 major)*

`risk-9` (major, §7-1 / §9 P2 / §5.2 D8) · `architecture-4` (major, §7-1 / §5.2 D8 / §6.3)

Rule 3.02 declares `config_snake = config_path` plain (`run_stress_test.smk:605`),
so any config edit re-fires it by mtime. `risk-9` traces the consequence to
`dev/scripts/check_baseline.py:326-329`, where the WF3 config snapshot is a `yaml`
baseline target — so on the baseline tree a caption edit turns
`check_baseline.py check` red. Both note that **P2 measured repo digest helpers,
not Snakemake rerun triggers**, so the executed probe never reached this layer.

### G4 — D16 and D19 collide on the degenerate axis *(severity divergence)*

`architecture-5` (**major**, §5.4 D16 / §5.5 D19 / §5.2 D11) · `risk-6` (**minor**, same decisions)

D16 requires the declared month set to be a non-empty subset of the *varying* set;
D19 says an axis with no varying months is degenerate rather than an error. When
nothing varies no non-empty subset exists, so D16 refuses the case D19 admits —
and D11's own default for that case is twelve *held* months, the precise input
D16 says must raise. `architecture-5` also catches a cross-reference error (D11
cites "D17's degenerate axis"; D17 is the rectilinearity postcondition, D19 is the
degenerate axis) and argues that mis-citation is what makes the collision easy to
miss on a read. **Graded differently by two lenses; each is dispositioned at its
own severity.** See § Conflicts.

### G5 — who actually calls the library *(2 major, 1 minor — see § Conflicts)*

`risk-2` (major, §5.3 D14 reason 1 / §7 R6 / §8 step 6) · `architecture-6` (major, §5.3 D14 / §5.4 D16-D17 / §5.5 D18) · `repo-fit-6` (minor, §10)

D14's first and heaviest reason for a library rather than a rule is "there is no
in-repo consumer". The three lenses reach that claim from three directions and do
**not** agree about it; the readings are preserved below rather than resolved.

### G6 — the migration sweep is incomplete *(severity divergence, different misses)*

`repo-fit-2` (**major**, §8) · `risk-10` (**minor**, §8 step 5b) · and the step-5b limb of `architecture-1` (blocking)

Three lenses, **three different sets of missed files** — the union is the finding,
so no member of this group substitutes for another:

- `repo-fit-2` — `dev/reference/contracts/weather-generator-seam.md` §WG-2 and its
  validator-table row; `tests/test_interchange_contracts.py:826-834`
  (`test_wg2_integration`), invisible to §8's own sweep because the path is spelled
  `join(_WG_DIR, "_work", "st_1.csv")` — segments, no slash — so the trailing slash
  the design explicitly defends is what hides it; and Event 2 (column removal) has
  no sweep at all.
- `risk-10` — `tests/test_check_baseline_scope.py:56` and
  `tests/test_check_baseline_indicator.py:62-63,268,272`, which pin the retired
  seven-column header by literal and will keep **passing** while asserting a dead
  contract.
- `architecture-1` — `tests/test_interchange_contracts.py:143-160`, where
  `_wg2_good()` is built in memory and contains neither search string, so
  `test_wg2_synthetic_pass` would keep asserting green that a retired contract
  holds.

### G7 — D3/D21 conversion completeness *(1 major, 1 minor)*

`architecture-2` (major, §5.1 D3 / §5.6 D21) · `risk-7` (minor, same sections)

`architecture-2` is the sharper claim and reaches a concrete defect:
`impose_climate_change.R:70` passes `precip_var_factor = cst_data$precip_variance`
as a **multiplier**, while D2/S1 make `precip_variance_change` a **percent**. D3
asserts "two conversion sites, and only two" and both rows name `precip_change`
alone. Implemented literally, every shipped config (variance min = max = 1.0 →
0.0 percent) hands the generator **a variance factor of zero**.

### G8 — §5.7's scope clause is wrong on three live passages *(1 major)*

`architecture-7` (major, §5.7)

"Everything not restated below is unchanged" is false against
`hydrological-model-seam.md:420-422` (the HM-4→HM-5→HM-7 check-3 sentence, which
names both deleted columns *and* `_PERTURBATION_AXIS`, a symbol §8 step 5 deletes),
plus two further HM-7 bullets on neither list. A drop-in whose scope clause is
wrong cannot be dropped in.

### G9 — the validation plan's homes and gaps *(2 major, 1 minor)*

`repo-fit-3` (major, §9) · `risk-4` (major, §5.2 D12 / §7-3 / §9 V5) · `architecture-8` (minor, §5.6 D21 / §9)

- `repo-fit-3` — V5–V11 are routed to "unit test" with **no test file named**;
  `tests/test_surface_axes.py` appears in neither §8 step 4 nor §9's gate table,
  so the design's most novel logic is an ungated file under the repo's own ladder.
- `risk-4` — `annual_perturbation` short-circuits flat vectors precisely because
  the weighted mean returns `1.3000000000000003`; D12 never mentions the
  short-circuit and D16 makes the flat case the *normal* path.
- `architecture-8` — the R side gains a filter-and-order join with no post-join
  assertion and no V-claim; a zero-row join hands `apply_climate_perturbations` a
  zero-length vector, where R's recycling makes a silent wrong answer at least as
  likely as an error. The Python side keeps its arity guard; the R side has none.
  Also: D21's "everything else is untouched" is literally false —
  `impose_climate_change.R:12-14` hard-fails on `length(args) != 4L`.

### G10 — `join_axes` has no postcondition on its partition *(1 major)*

`risk-3` (major, §5.3 D14 / §5.1 D4)

D4 makes *absence from the lookup* the sole marker of the baseline. Any `st_id`
representation mismatch makes the membership test all-False, and `join_axes` then
returns an empty surface while classifying **every** row as baseline, without
raising. The design names the `01`→`1` dtype hazard at length, then picks an
encoding that converts that exact miss from a visible empty result into a
plausible-looking one.

### G11 — repo conventions *(2 minor)*

`repo-fit-4` (minor, §5.3) — library signatures use bare names where
`naming.md` §5 requires `_path` / `_df` suffixes, in a block written as normative
API text an implementer will transcribe verbatim.

`repo-fit-5` (minor, §5.2) — D8/D13 never state how `reporting:` is obtained from
`config`, and never mention `get_config`; the contract is satisfiable but unstated,
leaving `reporting: null` ambiguous.

### G12 — `DEFAULT_SURFACE` is misnamed *(1 minor)*

`architecture-9` (minor, §5.2 D9 / D11) — `id: annual` on a surface whose axes take
D11's member-varying month set means a JFM design's default surface is identified
as `annual` while captioned "mean change over JFM", asserting exactly the collapse
D11 exists to stop being the default.

### G13 — the overlay deferral *(2 minor)*

`risk-5` (minor, §5.2 D10 / §5.7 S10 clause) · `repo-fit-6` (minor, §10)

`repo-fit-6` names an existing in-repo implementation of the collapse S10 pins —
`dev/scripts/preview_wf2_projection_plots.py:299-302,319-322,364-367` — so OQ-2
defers a *known* site rather than an unknown one.

### G14 — OQ-1's premise is false on disk *(1 minor)*

`architecture-10` (minor, §10) — OQ-1 says this work "carries no milestone
directory of its own" and offers "create `dev/milestones/r12/` early" as its first
option. **That directory already exists and holds `g2-assessment.md`.**

### G15 — the R-side member filter *(1 minor)*

`risk-8` (minor, §5.6 D21)

## Conflicts

Two independent lenses reaching different conclusions marks where the design is
under-determined. Both readings are kept intact; neither is resolved by the driver.

### Severity divergences

| Concern | Graded | By |
|---|---|---|
| D16/D19 degenerate-axis collision (G4) | **major** vs **minor** | `architecture-5` vs `risk-6` |
| Migration sweep incompleteness (G6) | **major** vs **minor** | `repo-fit-2` vs `risk-10` |
| D3/D21 conversion completeness (G7) | **major** vs **minor** | `architecture-2` vs `risk-7` |

Each original ID is dispositioned at its own filed severity. The "never re-grade"
rule forbids harmonizing them; this table is where that is made visible instead of
silently preserved.

### Factual: does D14's "no in-repo consumer" hold? (G5)

Three lenses, three readings, and they are **not** the same claim:

- **`risk-2` says the premise is false.** `docs/notebooks/Climate Stress Test.ipynb`
  reads `stress_test_design.csv` (line 500) and builds the surface with
  `.groupby(["temp_change", "precip_change"])["value"] … .unstack(...)`
  (lines 683–685). It cannot be re-rendered as §7 R6 assumes — the cell raises
  `KeyError` once the columns are dropped — and §8 step 6 lists no notebooks.
- **`architecture-6` says the premise is true and that is the problem.** D13 gives
  the library exactly one in-repo call site (`parse_surfaces` at parse time, which
  runs only the design-tier warning), so `axis_values`, `axis_caption` and
  `join_axes` are never invoked on any repo execution path: D16's axis-tier
  refusal, D17 and the whole caption algorithm execute only in unit tests, while
  the real R/JS consumers re-implement from HM-7's weaker prose.
- **`repo-fit-6` explicitly declines to treat its site as a counter-example** —
  `preview_wf2_projection_plots.py` is a dev script, not a WF3 rule, and D14 is
  about rules.

**The cheap test that would settle it:** open
`docs/notebooks/Climate Stress Test.ipynb` at the cited cells and confirm the read
and the `groupby`.

> **Driver fact-check, 2026-08-15 — `risk-2`'s premise HOLDS, and is broader than
> filed.** Verified in the notebook, not inferred: line 500 reads
> `EXP_DIR / "config" / "stress_test_design.csv"`; lines 683–685 carry
> `.groupby(["temp_change", "precip_change"])["value"] … .unstack("precip_change")`.
> Two further sites the finding does not cite: line 352 documents rule 3.09 as
> emitting `stress_test_design.csv`, and lines 481–482 describe its columns in
> prose. So the notebook depends on the retired artifact in **four** places, three
> of them narrative rather than executable.
>
> **This resolves the conflict as compatible, not contradictory.** The notebook is
> a consumer of the *artifact*, not a caller of the *library*, so `risk-2` and
> `architecture-6` are both correct and both fixes are owed: D14's stated reason is
> false as written, *and* the library still has no in-repo caller. `repo-fit-6`'s
> decision to scope its own site out stands on the same distinction.
>
> Graded **pre-existing condition → regression**: the notebook is stale today for
> unrelated reasons (`t2608132100`), but the column drop converts a stale render
> into a `KeyError`, which is new breakage this design causes.

> **Driver fact-check, 2026-08-15 — `architecture-10`'s premise HOLDS.**
> `dev/milestones/r12/` exists and contains `g2-assessment.md`. OQ-1's first option
> is therefore already available and needs no owner ruling to create anything.

### Adjacent, not contradictory: the baseline gate

`repo-fit` checked §8/§9's baseline ordering against `dev/baseline/manifest.json`
and **deliberately raised no finding** — the pre-record/post-record sequencing and
the trimmed-copy comparison check out for the `indicator` target. `risk-9` then
found a baseline failure through a **different** target: the WF3 config snapshot,
carried as a `yaml` baseline target.

Recorded here so the clearance is not read as covering the whole gate. Same gate,
different targets; both readings stand.

## Gate return to G1 — recommended before the revision is dispatched

Per `stage-contracts.md` § *Gate return from the panel*: when the panel's findings
admit resolutions differing in **scope**, **constraints**, or the **selected
alternative**, return to G1 before spending an author dispatch. Two do.

**Fork A — where the lookup's schema is normatively defined** (from
`architecture-1`'s second limb). The lookup is the Python→R handoff, i.e. the
weather-generator seam; the design defines it in HM-7, the hydrological seam.
Resolutions differ in the deliverable's scope: (i) keep the schema in HM-7 and add
a WG-2 replacement alongside; (ii) move the schema to WG-2 and have HM-7 reference
it; (iii) retire WG-2 into HM-7 deliberately, with a stated rationale.

**Fork B — whether the library gets a real in-repo caller** (from
`architecture-6`'s suggested fix, and bearing on `risk-2`). Resolutions differ in
the **selected alternative**: (i) add an assertion at the end of rule 3.16 that
every declared surface's axis derives without raising — which costs the lookup
back as a 3.16 input and reopens D22; (ii) accept no caller, add it to §7 as a
named risk, and move the classification tolerance, the degenerate rule and the
caption case table into HM-7 so the R re-implementer's normative document is
complete; (iii) make the notebook the caller, which turns `risk-2` from a breakage
into the design's own validation.

`architecture-10` also makes **OQ-1 answerable without a ruling**: its premise is
false, `dev/milestones/r12/` already exists.
