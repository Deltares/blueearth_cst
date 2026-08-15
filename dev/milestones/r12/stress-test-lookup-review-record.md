# WF3 stress-test lookup — consolidated review record

> **The durable audit trail** for the `design-review-loop` run
> `wf3-stress-test-lookup`, which produced
> `dev/milestones/r12/stress-test-lookup-design.md` (ACCEPTED 2026-08-15).
>
> This file exists so the per-round scratch is prunable: it carries the verdict
> table, every reviewer's verbatim output, the aggregation index with its
> conflicts section, and the closed ledger. **Rationale belongs here, not in the
> design** — superseded alternatives and round-by-round argument accrue in this
> record while the design's body carries the normative contract.

## The run

| | |
|---|---|
| Variant | full (three internal lenses, external rounds capped at 2) |
| Versions | v1 1231 → v4 2998 lines |
| Dispatches | 7 author/lens spawns (all Opus), 2 headless `codex exec` rounds, 0 Fable |
| Findings | **35** — 26 internal panel, 6 external r1, 3 external r2 |
| Re-raised | 2 |
| Dispositions | 35 accepted, **0 rejected, 0 deferred**, no blocking finding unfixed |
| Gates | G1 approved; G1-return (2 forks ruled); arbitration (3 rulings); G2 approved |
| Expensive checks | 6 |

**Fable was never spent, and that is a decision rather than an absence:** the
stage-6 revision qualified for escalation under the tier rule (external round 1
faulted the resolution of a prior-round finding), and the owner ruled Opus.

## Verdict table

| Round | Reviewer | Verdict | On | blocking | major | minor |
|---|---|---|---|---|---|---|
| Internal — risk | `critical-thinker` | `revise` | design-v1 | 0 | 5 | 5 |
| Internal — architecture | `cst-architect` | `revise` | design-v1 | 1 | 6 | 3 |
| Internal — repo fit | `python-engineer` | `revise` | design-v1 | 1 | 2 | 3 |
| External r1 (clean-room) | `gpt-5.6-sol` | `revise` | design-v2 | 2 | 3 | 1 |
| External r2 (regression duty) | `gpt-5.6-sol` | `revise` | design-v3 | 2 | 1 | 0 |

Round 2 was the cap, so **design-v4 carries no reviewer verdict.** Its three
fixes stand on the owner's arbitration; the driver's scope check and scoped
verification pass ran clean before G2 and stand where a verdict would.

## What the loop caught that the author and driver did not

Recorded because the run's cost is only justifiable against what it found:

1. **The WG-2 contract deletion** — two internal lenses independently. The design
   deleted the artifact one interchange contract pins while writing replacement
   text for the other; the contract's name appeared nowhere in the draft.
2. **A settled owner ruling violated in the migration plan** (`ext1-2`) — and the
   author had reasoned about the tension explicitly rather than overlooking it.
   Closing it needed five further sites: the violation had spread.
3. **An accepted fix that did not hold** (`ext2-3`) — the precision bound accepted
   at round 1 was normative, unqualified, and false outside the domain it was
   measured on, by up to 574 ulps. This is what round 2's regression duty exists
   for, and it is why the cap was not wasted.
4. **A driver bookkeeping defect**, found by an author spawn's self-containment
   check rather than by any gate — three stage entries silently lost to a
   `str.replace()` whose anchor did not match. See `observations.md` O4.

## Driver fact-checks

Premises verified against the repo rather than relayed:

- `risk-2`'s notebook consumer — **holds, and broader than filed** (four sites).
  Resolved the panel's one factual conflict as *compatible*: the notebook consumes
  the artifact, the library still has no caller, both fixes owed.
- `architecture-10` — **holds**; `dev/milestones/r12/` already existed, closing an
  open question on a fact rather than a ruling.
- `ext2-1`, `ext2-2`, `ext2-3` — all three **confirmed** before arbitration;
  `ext2-3` reproduced at 68 ulps and found worse on a wider sweep (564).
- **D35's domain claim** — reproduced independently at the scoped verification
  pass: 1 ulp throughout `multiplier ≥ 0.5` including to `1e6`, first break at
  2 ulp in `[0.25, 0.36)`. All five shipped configs sit inside the domain.

---

# Ledger


# Finding ledger — wf3-stress-test-lookup design run

**Append-only.** One row per original finding ID, at the severity **its own lens
filed**. Three concerns were graded differently by two lenses (`internal-review-index.md`
§ *Severity divergences*: G4, G6, G7); those rows keep their divergent grades even
where the resolution text is identical, because harmonizing them would erase the
record that the panel disagreed.

This file is also the run's **review record** in the sense of `design-v2.md`'s size
budget: per-finding argument lives in the Resolution column, cited by finding id,
rather than in the design body.

Round: `internal-panel` for all 26 (stage 2, `design-v1.md`).

Appended at v3: round `ext1` — the six external-review findings (stage 4, filed
against `design-v2.md`). Sections below the horizontal rule cover that round; the
26 rows above it are unchanged.

Appended at v4: round `ext2` — the three external-review findings from the capped
second round (stage 4, filed against `design-v3.md`), dispositioned under the
**owner's arbitration** rather than under a reviewer verdict. Its section is the
last one in this file; the 32 rows above it are unchanged.

## Dispositions

| ID | Round | Severity | Disposition | Resolution or rationale | Doc version |
|---|---|---|---|---|---|
| risk-1 | internal-panel | major | **accepted (claim)** / **rejected (suggested fix b, and the inverse spelling in fix a)** | **Claim accepted, measured and confirmed:** the composition is not the identity; `0.82 → -18.0 → 0.8200000000000001` on 0.7–1.3 at `step_num: 5`, and the shipped grids all round-trip so V4 cannot see it. D7's exactness claim is deleted; **D25** replaces it with a stated bound, §7-2 is restated as "within the comparator's tolerance", and V4's diagnosis rule is corrected so it no longer pre-commits the implementer to the wrong cause. **V20** adds the non-round-grid unit test the finding asks for. **Two parts of the suggested fix are rejected on measurement.** (i) "Pin the inverse spelling `(100 + precip_change)/100`": measured worse, not better — 32.9% vs 19.9% failures over 200k random `float32` multipliers, and 6,778 vs 1,155 with no exact solution at all. D25 pins `1 + p/100`. (ii) Fix option (b), "make it exact via a search over shortest-repr candidates verified at write time", is **unattainable**: 1,155 of 50,000 `float32` levels admit *no* `float64` percent that reconstructs them under any spelling. So option (a) is forced rather than chosen, and D25 says so with the number. E11 records both measurements. | design-v2.md |
| risk-2 | internal-panel | major | **accepted** | Premise confirmed by the driver fact-check and broader than filed — the notebook depends on the retired artifact at **four** sites (lines 352, 481-482, 500, 683-685), three narrative and one executable. D14 reason 1 is corrected to "no in-repo **rule** consumes a response surface; the one in-repo consumer is a notebook, which can import the library directly", which strengthens the decision rather than weakening it. **R6 is re-graded from a re-render chore to a rewrite that lands in the migration commit** (§8 step 6), and the notebook becomes the design's worked example of `join_axes` + `axis_caption` — the only in-repo demonstration that the corrected axis is obtainable. | design-v2.md |
| risk-3 | internal-panel | major | **accepted** | **D28** added. `join_axes` gains a stated postcondition and raises `BaselinePartitionError`: the `st_id` set absent from the lookup must be **exactly** the baseline token and `surface_df` must be non-empty. Both halves of the suggested fix are taken rather than one — the library owns the indicator read (`read_indicators(indicators_path)`, beside `read_lookup`) **and** `join_axes` normalises both key columns to zero-padded strings before partitioning, so a caller who loaded the frame elsewhere is repaired rather than silently mis-partitioned. The finding's sharpest point is recorded verbatim in the decision: the design names the `01` → `1` hazard at length and then picks an encoding that converts that exact miss from a visible empty result into a plausible-looking one. Mirrored into §5.8's contract text, since the out-of-repo consumers are the ones who will implement the partition. | design-v2.md |
| risk-4 | internal-panel | major | **accepted** | **D26** added; the "the weighting is immaterial" sentence is **deleted** as the finding asks — it is the claim `annual_perturbation:105-107,122-124` was written to refute. The short-circuit becomes part of the statistic's normative definition *and* of §5.8's contract text. Re-measured, and the finding is **stronger under this design than as filed**: the counter-example it cites (`1.3` → `1.3000000000000003`) is in *multiplier* space, and the multiplier column is being deleted — 0/50,000 random `float32` multipliers mismatch. In the spaces the lookup actually holds, the rate is **49%** (percent, 200k samples) and **48%** (°C, 100k). Realistic grids hit it: 0.6–1.4 at `step_num: 3` and 0.5–1.5 at `step_num: 7`. So the unit change *creates* this exposure. V5 keeps its **equality** form (the second half of the suggested fix, relaxing it to a tolerance, is therefore not needed); **V19** is added for the percent-space case. E13 records the rates. | design-v2.md |
| risk-5 | internal-panel | minor | **accepted** | Both requested clauses added and a third. §5.8's overlay paragraph now states the asymmetry explicitly — on the lookup side the collapse averages **equal** values by D16 and reports an imposed change; on the GCM side it averages unlike months, reports a summary, and moves **non-affinely** with `M`, so two GCMs at the same annual mean can sit at opposite ends of a JFM axis. D10's "magnitude and label only" claim is scoped to the surface members. OQ-2 gains the two questions the finding names (recompute per surface? recorded where?), and the consequence — a plausibility judgement changed by a config edit that leaves no run-identity trace — is risk **R12**. Filed minor and dispositioned as minor: the asymmetry is pre-existing and no overlay code lands here. | design-v2.md |
| risk-6 | internal-panel | minor | **accepted** | Same concern as `architecture-5`, **kept at its own filed severity** (see the header note). Both requested closures made. The degenerate/explicit-`months:` collision is resolved by **D27**'s precedence rule — classify first; an empty varying set makes the axis degenerate and neither the subset nor the homogeneity rule applies, so an explicit `months:` on a temp-only design's precip axis is admitted and returns the constant. The threshold is fixed **once** as `max − min > 0` exactly, in D11, and §5.5 step 1 cites it instead of an undefined `tol`. **The finding's `0.8` / `0.8000000001` hazard is answered rather than removed:** classifying that as varying and then refusing on homogeneity is the *correct* outcome — those are two different perturbations — and the refusal names the homogeneous subsets so the user sees what they declared. D11 records that reasoning, and D17 explains why the *rectilinearity* check does take a real tolerance (`rtol = 1e-9`) while classification does not. | design-v2.md |
| risk-7 | internal-panel | minor | **accepted** | Same concern as `architecture-2`, **kept at its own filed severity**. D3's table gains the `precip_variance` inverse and the conversion is restated as a **rule over the percent columns** rather than a per-column formula, so a future column inherits it; "two conversion sites, and only two" becomes "one site per direction, covering both multiplier columns". D21 names both columns. The finding's own framing is carried into the design — every shipped config sets variance flat at 1.0, so the omission could not fail any gate the repo can run, which is exactly why it belongs in normative text rather than in the implementer's head — and D25 notes that the variance path inherits the same conversion residual on a path no fixture exercises. | design-v2.md |
| risk-8 | internal-panel | minor | **accepted** | **D29** added, taking the suggested assertion verbatim: `impose_climate_change.R` stops unless the filtered frame has `nrow == 12L` and `identical(cst_data$month, 1:12)`, with the member token in the message — mirroring `export_wflow_results.py:108-121` on the Python side. The finding's framing is recorded as the decision's rationale: the change converts a loud, structural `MissingInputException` into a quiet data condition inside a script that has no guard, and R's recycling makes a silent wrong answer at least as likely as an error. **V17** is its falsifier. Overlaps `architecture-8`, which adds the arity limb; both are dispositioned separately. | design-v2.md |
| risk-9 | internal-panel | major | **accepted** | Both limbs. **(a) The falsifier was wrong:** rule 3.02 declares `config_snake = config_path` as a plain input (`run_stress_test.smk:604-605`), so any config edit re-fires it by mtime — §7-1 loses "re-fires 3.02" and keeps the freeze, the digest and 3.16b, which are the real discriminators. **(b) The gate consequence is confirmed by a measurement the finding did not have:** `fingerprint_yaml:449-455` `yaml.safe_load`s the **whole unprojected document** and hashes canonical JSON, and `copy_config_files.py:222` is a `shutil.copyfile` — so `reporting:` is inside the baseline `yaml` target's hash, and a caption edit on the baseline tree does turn `check_baseline.py check` red. Recorded as risk **R8**, scoped honestly (it is a pre-existing property of that target; the design's contribution is making a caption edit a config edit), and **decided** as the finding asks: §8 step 6 rules that the shipped seeds declare **no** `reporting:` section, so the fingerprint is unarmed until a project opts in. **P2-b** is added to §9 as the rerun-trigger layer P2 never reached. | design-v2.md |
| risk-10 | internal-panel | minor | **accepted** | Same concern as `repo-fit-2`, **kept at its own filed severity**; the two name **different** missed files and the union is the finding, so neither substitutes for the other. §8 step 5b's sweep is widened past the pattern this finding proposes — to `_work|stress_test_design|temp_change|precip_change|precip_mean|temp_mean` over `tests/ dev/scripts/ dev/reference/ docs/notebooks/` — and its table is re-derived, naming `test_check_baseline_scope.py:56` and `test_check_baseline_indicator.py:62-63,244,268,272` with the annotation this finding supplies: they keep **passing** while asserting a dead contract, and `test_float_key_columns_are_compared_as_written` exists solely because `temp_change` is a float key column, which after the change no remaining key column is. The WF2 hits are excluded **by inspection, not by pattern**, as suggested. The step's exit condition is rewritten, because "a non-empty result is the failure" cannot work for a test whose stale content is a literal. Risk **R10**. | design-v2.md |
| architecture-1 | internal-panel | blocking | **accepted (limb 1: the missing WG-2 replacement)** · **accepted (limb 2: seam placement)** | **Limb 1 — the missing replacement.** New **§5.7** is WG-2's drop-in replacement text, and **D30** records that WG-2 **keeps its id** and re-points at `<exp>/config/stress_test_lookup.csv` rather than being retired or renumbered: the producer and consumer are unchanged and the artifact moved rather than disappeared, so a WG-7 would assert a discontinuity that did not happen. `validate_wg2` changes **mechanism**: `_WG2_HEADER` becomes the five-column header, `n != 12` becomes `n == 12 × ST_NUM` plus a `(st_id, month)` grid-completeness check plus an `st_0`-absent assertion, with `ST_NUM` passed in. §8 step 5 carries `weather-generator-seam.md`, `_WG2_HEADER`, `validate_wg2`; step 5b's widened sweep reaches `tests/test_interchange_contracts.py:143-160` and `:826-834`; **V15** is the falsifier. The finding's sharpest point is recorded: `test_wg2_synthetic_pass` is a **positive assertion** that a retired contract holds, which is worse than R9's silent skips. **Limb 2 — seam placement.** The finding's reading is upheld and was ruled at the G1-return (Fork A): the lookup is the Python → R handoff, so its schema is normatively defined in **WG-2**, and HM-7 (§5.8) **references** it rather than restating it. §5.1 D2 is re-pointed accordingly. The **third** limb — widening step 5b's pattern to the column names — is dispositioned under `repo-fit-2`/`risk-10` and implemented once. | design-v2.md |
| architecture-2 | internal-panel | major | **accepted** | Same concern as `risk-7`, **kept at its own filed severity**; this lens reached the concrete defect and its framing is the one carried into D3 and D21 — implemented literally, v1 hands `apply_climate_perturbations` a variance factor of **zero** on every shipped config (variance `min = max = 1.0` ⇒ 0.0 percent), not merely an unstated inverse. Confirmed by reading `impose_climate_change.R:70`. Both halves of the suggested fix taken: D3 states the conversion as a rule over the percent columns so a future column inherits it, and **V16** is added — the R-side factors reconstructed from the lookup are checked against the pre-migration `st_<m>.csv` values on one member, at step 7 beside V4. | design-v2.md |
| architecture-3 | internal-panel | major | **accepted (claim and fix a)** / **rejected (fix b)** | Same concern as `risk-1`, filed at the same severity by both lenses, and this lens's framing is the one D25 adopts: the **unit conversion** introduces its own non-invertibility, independent of `float32`, because the design deletes the artifact that stored the multiplier the generator receives and replaces it with one from which that multiplier is **reconstructed**. §3's S1 rationale is corrected accordingly — it claimed "§5.1 D7 resolves it directly", and it does not. Fix option (a) is taken and made precise rather than left as "state the exactness condition": D25 gives the measured bound (one `float64` ulp of the level), the worked failing grids, and the fact that every shipped config is in the exactly-invertible set — which is why V4 is valid *on those configs only* and **V20** covers the rest. **Fix option (b) is rejected on measurement**, as under `risk-1`: no written text can deliver exactness, since 1,155/50,000 levels admit none. The finding's own suggestion to write `float(str(np.float32(f*100-100)))` is specifically rejected — measured, `float32`-quantizing the *percent* costs ~8 orders of magnitude (5.98e-08 vs ~1e-16), so D25 rule 1 writes `float64` shortest repr instead. E11, E12. | design-v2.md |
| architecture-4 | internal-panel | major | **accepted** | Both limbs, and the suggested restatement is adopted almost verbatim. **Limb 1:** §7-1 no longer names 3.02 as a falsifier — the finding is right that a falsifier firing on correct behaviour is worse than none, because whoever runs V12 would have to decide whether to believe the gate. 3.16b and the freeze are named as the discriminators they are, and §6.3's rejection sentence is scoped to 3.16b and the digest. **Limb 2 — the circularity:** D8 now states the honest version. 3.02 re-firing by mtime **is** how the relabel gets recorded; the snapshot is a `shutil.copyfile` so it carries `reporting:` verbatim; therefore **a relabel made without any WF3 invocation is unrecorded**, and the recording costs one cheap re-invocation rather than a re-run of the experiment. R5 is superseded by **R11**, which carries that correction. **P2-b** records the layer measurement. | design-v2.md |
| architecture-5 | internal-panel | major | **accepted** | Same concern as `risk-6`, **kept at its own filed severity** — this lens graded it major and that grade stands unharmonized. The suggested precedence is adopted as **D27** and stated in the order given: classify first; empty varying set ⇒ degenerate (D19), neither the subset nor the homogeneity rule applies; otherwise the subset and homogeneity rules apply. Mirrored into §5.8's contract text, which in v1 stated the subset rule **unconditionally** and would have exported the collision to every re-implementer — the finding is right that this is the load-bearing half. D11's mis-citation ("D17's degenerate axis"; D17 is the rectilinearity postcondition) is fixed to D19, and the finding's point that the mis-citation is what makes the collision easy to miss is recorded. | design-v2.md |
| architecture-6 | internal-panel | major | **accepted (as the G1-return ruled it)** | The finding is upheld on fact — `parse_surfaces` is the library's only in-repo call site and runs only the design-tier warning, so `axis_values`, `axis_caption` and `join_axes` execute on no repo execution path. Its **first** suggested fix (a rule-3.16 assertion) is **not** taken: the G1-return ruled Fork B in favour of the second, and **§6.9** records why (it costs the lookup back as a 3.16 input, reopening D22; it puts a reporting concern inside the reduction; it spends R12's budget on a check with no artifact). The **second** fix is taken in full: (a) risk **R9** names the gap, stating plainly that D16-axis, D17, D27 and D28 are unit-test-only and that an R consumer skipping those clauses re-introduces C1's misreport with nothing in-repo reporting it; (b) the classification threshold, the degenerate rule **and** the full caption case table move into normative contract text — with the split following the G1-return's seam rule, so the schema half lands in WG-2 (§5.7) and the derivation half in HM-7 (§5.8). D15 is rewritten around that obligation. | design-v2.md |
| architecture-7 | internal-panel | major | **accepted** | The scope clause is **replaced by a per-passage disposition table**, exactly as suggested, and the finding's standard — "a drop-in whose scope clause is wrong cannot be dropped in" — is applied to the *new* WG-2 section too, which has four touched passages of its own. Dispositions: the two axis-column bullets (`:346-354`, `:355-368`) are **DELETED**, with their two surviving claims (the WF2 overlay tie, the evenly-spaced guarantee) restated as properties of the derivation rather than of a stored column; the HM-4 → HM-5 → HM-7 check-3 sentence (`:420-422`) is **AMENDED** to "header minus `statistic`" with the `_PERTURBATION_AXIS` clause struck. §8 step 5 gains the relational section alongside the docstring. The distinction the finding draws is preserved in §8: the validator's *logic* needs no change (check 3 compares the `location` value SET post-CR-2), so this is a prose amendment, not a code change. | design-v2.md |
| architecture-8 | internal-panel | minor | **accepted** | Both limbs, both verbatim. The arity check moves **4 → 5** (`impose_climate_change.R:12-14` hard-fails on `length(args) != 4L`, so v1's "everything else is untouched" was literally false), and **D29** specifies the post-filter assertion. **V17** is the R-side claim the finding notes V1–V14 entirely lacked, which C6 requires for the one seam this migration moves across a language boundary. Overlaps `risk-8` on the assertion limb; both dispositioned separately at their own severities. | design-v2.md |
| architecture-9 | internal-panel | minor | **accepted** | `DEFAULT_SURFACE.id` becomes **`default`**, and the finding's reasoning is carried into D9: D18's premise is that a typed label drifts and a derived one cannot, the surface `id` **is** a typed label that D14 makes a consumer's handle on the frame, and `annual` is accurate only for the uniform case — so a JFM design's default surface would be identified as `annual` while captioned "mean change over JFM", asserting the exact collapse D11 exists to stop being the default. The suggested reservation is taken too: `annual` is reserved for a surface a user declares with `months: [1..12]`, which D16's subset rule then correctly refuses on a seasonal design. | design-v2.md |
| architecture-10 | internal-panel | minor | **accepted** | Premise verified by the driver fact-check: `dev/milestones/r12/` exists and holds `g2-assessment.md`. **OQ-1 is closed on a fact rather than a ruling** and the migration note files at `dev/milestones/r12/migration_stress-test-lookup.md` (§8). The finding's framing is what makes the closure clean — the option set OQ-1 put to the owner had one option already satisfied, which turns "create a directory early" into "file into an existing one", i.e. no decision at all. §10 keeps OQ-1 as a **closed** entry rather than deleting it, since it was carried to G1 unruled and the closure is the record of why it stopped needing a ruling. Filing under `r12/` is also right on substance: S8 makes this design R12's prerequisite. | design-v2.md |
| repo-fit-1 | internal-panel | blocking | **accepted** | Same concern as `architecture-1` limb 1, filed blocking by both lenses, and this lens's evidence is the fuller one — it enumerates the four WG-2 passages (`:110-117` path bullet incl. the "declared `input:` on rule 3.16" clause D22 falsifies; `:330` validator-index row incl. "continuously verified? **yes (persists)**"), the code sites (`interchange_contracts.py:223,227`) and the three test sites, and establishes that `validate_wg2` is called from **nowhere but those tests**, which is what bounds the fix. New §5.7 is the drop-in replacement, taking the finding's own preferred branch — **re-point WG-2 at the lookup** rather than retire it (D30) — because the lookup *is* the weather generator's input contract, moved and reshaped. The `input:`-on-3.16 clause is struck; the validator-index row is amended with a re-asserted "persists", which still holds since the lookup is a `rule all` target and not `temp()`; the bounded-substitution walkthrough is amended at `:298`/`:310`. §8 **step 5** names all four passages plus `_WG2_HEADER` and `validate_wg2`; the test sites, including `test_wg2_integration`, land in **step 5b**'s widened-sweep table, which is where this run keeps test-suite work. **V15** is the claim → falsifier row the finding asks for. | design-v2.md |
| repo-fit-2 | internal-panel | major | **accepted** | Same concern as `risk-10`, **kept at its own filed severity**; the two lenses found different files and the union is what §8 step 5b now carries. Both of this finding's limbs are taken. **(a) The sweep's blind spot:** the trailing slash the design explicitly defended is what hid `tests/test_interchange_contracts.py:830-831` (`join(_WG_DIR, "_work", "st_1.csv")` — segments, no slash), and that test is behind a `skipif`, i.e. R9's silent class. The widened pattern drops the slash and accepts `_workflow` false positives as cheap, exactly as suggested. **(b) Event 2 had no sweep at all:** the pattern now covers the column names and the retired header tokens, and step 5b's table is re-derived from the widened result — it names `test_wg2_integration`, `test_check_baseline_indicator.py` and `test_check_baseline_scope.py`, with the finding's observation that four of them are **false greens** rather than failures. The note about `_member_artifact`'s legacy `cst_1.csv` fallback (`:831`) is carried into the step's exit condition. Risk **R10**. | design-v2.md |
| repo-fit-3 | internal-panel | major | **accepted** | `tests/test_surface_axes.py` is added to **§8 step 4** and to **§9's narrow-tier gate row**, and V5–V11 are mapped onto **named test functions** in it. The finding's framing is carried into step 4: under this repo's own ladder ("only the tests covering the file you changed"), an unnamed test file is an ungated file — and none of the four files §9 called "the narrow tier" owns `surface_axes`, whose surface is the largest new thing in the design. The suggested promotion is taken literally: §9's E10 `np.linspace` member matrices and rendered captions move from run evidence into the test file's fixtures, so the caption cases stop being "a table rendered ad hoc during the design run". §9's gate row also gains the two `check_baseline` test files, which `risk-10`/`repo-fit-2` show are covering a shape that is about to retire. | design-v2.md |
| repo-fit-4 | internal-panel | minor | **accepted** | Signatures renamed exactly as suggested: `read_lookup(lookup_path)`, `month_classes(lookup_df, variable)`, `axis_values(lookup_df, axis)`, `axis_caption(lookup_df, axis)`, `join_axes(indicators_df, lookup_df, surface)`, and the return is `(surface_df, baseline_df)`. `risk-3`'s new reader inherits the convention as `read_indicators(indicators_path)`. The finding's two reasons are recorded in D14: the block is normative API text an implementer transcribes verbatim, and `surface_axes.py` becomes the reference implementation HM-7 cites — a contract surface under `naming.md` §5's own framing. The `_csv` suffixes on the Snakemake labels (`lookup_csv`, D20/D23) are kept, as the finding correctly notes §5 reserves that position for them. | design-v2.md |
| repo-fit-5 | internal-panel | minor | **accepted** | D13 now states the read, in the suggested spelling and for the suggested reason: `get_config(config, "reporting", {}, optional=True) or {}`, and `surfaces` from it the same way. The trailing `or {}` / `or []` is called out as load-bearing rather than incidental — `get_config` returns a present key's value as-is, so `reporting:` with no body yields `None` and would `TypeError` on subscript — with the in-file precedent (`run_stress_test.smk:526-529`) cited. This closes the `reporting: null` ambiguity in favour of `DEFAULT_SURFACE`, which is what D10 already promised for "absent or empty". | design-v2.md |
| repo-fit-6 | internal-panel | minor | **accepted** | `dev/scripts/preview_wf2_projection_plots.py:299-302,319-322,364-367` is named in **OQ-2** as the existing in-repo implementation of the collapse S10 pins, and in **§7 R12** as the place a second, divergent implementation currently lives. The finding's own scoping is preserved verbatim in both places: it is a **dev script, not a WF3 rule**, so it is *not* a counter-example to D14's argument and changes no decision here — the point is that Q6 defers a **known** site rather than an unknown one. | design-v2.md |
| ext1-1 | ext1 | blocking | **accepted** | Confirmed by arithmetic before it was fixed (**E15**): twelve varying months, JFM at 0.7–1.3 and Apr–Dec at 0.9–1.1, `step_num: 2`, `months: [1,2,3]` — the axis is **−30 / 0 / +30**, the annual collapse is **−14.931507 / 0 / +14.931507**, and v2's case 1 (fires when `H` is empty) captions the first `mean change over the year`. A factor of two, on a declaration D16 explicitly admits — and which D16's *own refusal message* instructs the user to write, so the defect sits on the design's recommended repair path rather than in a corner. **D31** takes the suggested fix in full: the leading phrase is `mean change over <label(M)>` in every non-degenerate case, derived from `M` alone and never from the global varying/held classification. The finding's second requirement — "define how varying months outside `M` are described or omitted" — is met by **describing** them: `E = V \ M` gets its own clause group (`Apr–Dec also vary, -10% to +10%`), which adds cases **1b** (H empty, E non-empty) and **1c** (both non-empty). Rather than write a second grouping rule, the held-months rule is generalised into **one clause builder** used twice, so the three-group legibility cap (v2's 3b/3c) is stated once and inherited. Degenerate captions become `M`-scoped for the same reason (` in JFM` appended when `M` is not all twelve). Mirrored into **§5.8**, which is the half a re-implementer executes, and the finding's own test case is named: `test_surface_axes.py::test_caption_explicit_subset_of_all_varying`, in §8 step 4 and in V11. §5.5 marks 1b/1c **argued, not rendered** — v1's rendering harness is not this run's — rather than quietly extending §9's "executed captions" table with rows nobody executed. | design-v3.md |
| ext1-2 | ext1 | blocking | **accepted** | **Not a judgement call, and not argued.** The G1-return's Fork B is settled framing: accept no in-repo caller, do not restore the lookup as a 3.16 input, do not reopen D22 — and do not make the notebook the caller. v2's §8 step 6 did exactly the last, specifying the rewrite as `surface_axes.read_lookup` + `read_indicators` + `join_axes` + `axis_caption` while D15, alternative 6.9 and R9 each assert the library executes on no repo path. The finding is right that an implementer could not satisfy both. **Resolution as the run's framing determines it:** the notebook becomes a **contract-based external-consumer example that imports nothing from `surface_axes`**, re-implementing from HM-7's text as the R and JavaScript consumers must (§8 step 6, three numbered steps). It is a coherence win rather than a concession — Fork B's compensating requirement is that the contract text be complete enough to re-implement from, and the notebook is now the only in-repo place that is *exercised* instead of asserted; **V21** compares it against the reference implementation once at step 6, scoped so it cannot read as reopening the ruling (the comparison is a migration-time one-off in a scratch session; the notebook imports nothing). Closing it required touching **six** sites, because the contradiction reached six: §5.3 D14 reason 1 (the "can import the library" argument, replaced), D15's completeness note, alternative **6.9**, risk **R6**, risk **R9** (which *gains* the notebook as evidence for its stated mitigation), and §8 step 6. **This row also supersedes the last clause of `risk-2`'s v2 resolution** — "the notebook becomes the design's worked example of `join_axes` + `axis_caption`". That row is not rewritten (the ledger is append-only) and its substantive half stands: the notebook is a real consumer, R6 is a rewrite in the migration commit, not a deferred re-render. Only the *form* of the rewrite changes. | design-v3.md |
| ext1-3 | ext1 | major | **accepted** (three parts, one disposition each: degenerate scalar — accepted; result object — accepted; key-width inference — accepted) | All three parts hold on reading v2. **(a) The degenerate scalar was undefined.** D27 step 2 said "return the constant for those months" and routed around step 3, where the collapse lives, so a degenerate axis whose `M` spans several held offsets had no specified value — the finding is right that conforming implementations could return the first month's level, an unweighted mean, a weighted mean or a refusal. **D32** takes the suggested fix verbatim: the same weighted collapse over `M`, with D26's exact-equality short-circuit, and the design now states the distinction the finding drew — constant across **members** (what degenerate means) is not the same as equal across **months**. Caption case 4c reports the value rather than hiding it. **(b) The API had no channel for `degenerate`.** **D33** adds `AxisResult` (values, caption, `degenerate`, effective `M`, variable) behind `derive_axis`, with `axis_values`/`axis_caption` surviving as accessors so `repo-fit-4`'s pinned names are kept, and `SurfaceJoin` for the join. **This narrows `repo-fit-4`'s pinned `(surface_df, baseline_df)` return, deliberately**: both names survive as field names, and the change is recorded here rather than left to read as an untracked regression against a closed disposition. §5.8 states the same four facts as *semantics* — a derivation returns values plus caption plus flag plus effective `M` — so an R consumer gets the requirement without the Python object. **(c) Key-width inference was the sharpest point and repairs a latent defect.** D28 required normalisation "at `index_width(ST_NUM)`" inside a module D14 declares pure and free of `snake_utils` — so v2's own text needed either an import it forbids or an argument no signature carries. `key_width(lookup_df)` infers the width from the lookup's `st_id` strings and raises when they disagree; WG-2 gains the pinning sentence that makes the inference sound ("every `st_id` in one table has the same width"), and `validate_wg2` gains the check, so a documented property becomes a validated one. `ST_NUM` is no longer needed for a join at all. | design-v3.md |
| ext1-4 | ext1 | major | **accepted** (the "limit the claim" branch of an either/or fix; the end-to-end-experiment branch is **not taken**, with the reason recorded) | The conflation is real and v2's wording made it: §7-2 read "indicator values move by at most one `float64` ulp of the perturbation level, and are expected identical within the baseline comparator's tolerance" — a bound measured on the **reconstructed multiplier**, spent as a bound on **simulated indicators**. The finding's mechanism list is right, and this run's own origin is the standing precedent: it opened because two code paths were assumed equivalent on exact parameters, and E6 measured the transform at *unit* factors — zero parameter difference — moving the single-day precipitation maximum by **−32.9%** and E7 five of eleven `q` indicators by a factor. §7-2 is split accordingly: the ulp bound is stated for the multiplier (falsifiers V16, V20), indicator equality is claimed **only** where the forcing is bit-identical, which is every level of every shipped config including `snake_config_baseline.yml`, and for a non-exact grid **no output bound is claimed at all**. D25's closing paragraph is rewritten to the same scope. **The suggested fix's other branch — an end-to-end non-round-grid experiment with an empirically justified tolerance — is not taken**, and **R13** states why rather than leaving it implied: it requires a full WF3 run on a config no fixture carries, `check_baseline` cannot gate a tree it holds no reference for (R1/R2), and a tolerance derived from one run is an assertion wearing a number. The exposure is stated instead, scoped as **migration-once** rather than standing, with a G2 escalation condition. One by-product is a **stronger** V4 rule than v2's: because V4 runs on the baseline config alone, where the multiplier is bit-identical, D25's arithmetic cannot explain a failing group *at all* — v2's general "moved by more than one ulp" diagnosis embedded the same conflation in the opposite direction. No measurement is deleted; only the claim narrows. | design-v3.md |
| ext1-5 | ext1 | major | **accepted** | The finding is right that V17 validated nothing: its claim is about the guard's behaviour on a malformed slice and its check was a WF3 run on the **valid** rapid config, which is green whether D29's `stop()` exists, is misspelled, or is absent. Worse, the negative executions it asks for **could not be written against v2's design** — the malformed-input path in `impose_climate_change.R` is reached only after the script reads the weathergen YAML and loads a realization netCDF through `weathergenr`, so every negative fixture would have needed the heaviest fixture in the repo. **D34** removes that: the read-filter-assert block becomes `blueearth_cst/weathergen/read_member_grid.R`, sourcing nothing (verified — `global.R` is options-only, so it is not even needed), which makes each fixture one `Rscript --vanilla -e 'source(...)'` with no netCDF and no package load. D29's semantics are untouched; only its location moves. V17 is rewritten to enumerate the fixtures rather than gesture at them, asserting **nonzero exit and the member-token diagnostic**. **One of the four suggested fixtures is re-classified rather than adopted as filed:** "unordered months" is not a failure case under D21, which orders by `month` before asserting — so it is specified as a **positive** twin (a shuffled valid slice returns the same frame in month order), leaving three negatives. Recorded here rather than silently dropped. The tests land in a **new** `tests/test_read_member_grid.py`, not in `tests/test_r_scripts.py`, whose module docstring declares itself syntax-only ("no evaluation, no side effects") — quietly violating a file's stated contract is the class of defect this run keeps catching. §8 step 2, §9's narrow-tier gate row and the migration note's machinery list all carry the new file, mirroring what `repo-fit-3` earned for `test_surface_axes.py`, and the gate row notes it **skips without `Rscript`**. | design-v3.md |
| ext1-6 | ext1 | minor | **accepted** | Taken exactly as suggested and no further: the first clause becomes "`validate_wg2` **not** green on a valid `12 × ST_NUM` lookup". The other two clauses were correct falsifiers as filed and are kept verbatim. One clause is added while the row is open — green on a table mixing `st_id` widths — because D33 makes uniform width a pinned schema property that `validate_wg2` now checks, and a pinned property with no falsifier is what V15 exists to prevent. | design-v3.md |
| ext2-1 | ext2 | blocking | **accepted per owner ruling** (arbitration 2026-08-15: *accepted, fix required*) | Premise confirmed by the driver before arbitration and re-read here at `design-v3.md:838-842`: D28 stated exactly two conditions — the `st_id` set **absent** from the lookup equals the baseline token, and `surface_df` non-empty — and neither constrains the *missing* direction. A stale or partial indicator table whose members are a strict **subset** of the lookup's satisfies both, so `join_axes` returned a **silently incomplete** response surface. That is worse than the failure D28 was introduced to catch: a mis-keyed join produces a visibly wrong shape, an incomplete one produces a plausible surface with holes — or a biased surface, if the missing members sit at one end of the grid. **Fix:** D28 rule 2 becomes **three ordered checks** — (a) `I \ L == {b}` → `BaselinePartitionError`; (b) **`I \ {b} == L`, set equality** → `SurfaceMemberMismatchError`; (c) `surface_df` non-empty → `BaselinePartitionError`, which now catches only the empty-lookup residue b cannot see. Given (a), the only way (b) fails is a lookup member missing from the indicators, so the message names the missing ids. **A second error class rather than widening the first**, because a missing member is not a baseline problem and a misnamed diagnostic is the mis-citation class this run has caught twice (D11 → D17, `architecture-7`'s scope clause); error names are contract surface here. **Mirrored into §5.8's report-time join semantics** as the same three checks, per the reviewer's suggested fix — HM-7 owns report-time join semantics under Fork A, and §5.7 is untouched. **V18 widened** with the missing-member case and `::test_missing_lookup_member_refused`, which is the fixture the reviewer asked for. One coherence by-product, recorded because it converts an aspiration into a fact: §5.3's standing sentence "`validate_hm7` is test-time only … D28 is the report-time tier" is now **true** — check b is that validator's completeness check 1 evaluated at report time. The duplication across tiers is deliberate and said so, since the validator runs in this repo and never at a consumer, and the consumer is where the surface is drawn (R9). **No new decision id:** this is a mechanism correction to an existing rule, which this document amends in place (the D3/D9/D11 precedent at v2); splitting one rule across two ids is the mis-citation hazard itself. | design-v4.md |
| ext2-2 | ext2 | blocking | **accepted per owner ruling** (arbitration 2026-08-15: *accepted, fix required; orientation reversal stays legal*) | Confirmed on three sites of `design-v3.md`: `:490` makes `variable` a closed enum **per axis** with no cross-axis rule, `:760` keys `SurfaceJoin.axes` by variable, and `:850` names derived columns through `AXIS_COLUMN[variable]`. So `x: {variable: temp}, y: {variable: temp}` passed the schema while one `AxisResult` overwrote the other and both targeted `temp_change` — an **admitted configuration no conforming implementation can serve**. The finding's framing is what makes it easy to miss and is carried into the design: a closed key set plus a closed value enum still admits it, because both values are individually legal and only the **pair** is not. **Fix taken exactly as suggested:** §5.2 requires `{x.variable, y.variable} == {"temp", "precip"}` at parse time; **orientation reversal (`x: precip, y: temp`) stays legal** and gets its own must-not-refuse test twin, so the fix cannot be over-applied. D13 names the refusal (`DuplicateAxisVariableError`) and states why no per-field validator can reach it; D33 records that keying `axes` by variable is total **only** because of the rule, so the schema constraint and the result representation are one decision written in two places. **V22** is the negative parser test the finding asks for. The design also records **what the refusal costs** rather than letting it read as free: two month windows of the same variable on two axes become inexpressible — which needs a shape change to D33 and to `AXIS_COLUMN`, and is in any case not a response surface over this experiment, since D10's affinity argument makes two temperature axes affine images of each other (a line embedded in a plane). **Not mirrored into §5.7 or §5.8**: this is a project-config schema rule, not a property of either contract artifact, and restating it there would breach Fork A's one-artifact-described-once split. **No new decision id** — D8's schema and D13's parser are amended in place. | design-v4.md |
| ext2-3 | ext2 | major | **accepted per owner ruling**, with the **shape of the fix ruled by the owner** (arbitration 2026-08-15: *impose a validated multiplier domain*; the domain-qualified-bound branch **declined**) | The finding holds and the driver's pre-arbitration check found it **worse than filed**: reproducing D25's own conversion verbatim, worst error over `[0.001, 2.0]` is **564 ulps** against a normative bound of one, the reviewer's counter-example at level `0.013596006` is **68 ulps** (`-98.6403994` → `0.013596005999999883`), and the bound is true exactly over `[0.5, 1.6]`, the region D25 measured. This is the evidence register's own warning realised — an unqualified normative claim standing on a domain-restricted measurement — and it matters because WG-2 pins the bound as a cross-language contract and V16/V20 use it as an acceptance threshold. **The suggested fix is an either/or and the owner ruled which branch**: impose and parse-time validate a domain (**taken**), rather than replace the bound with a domain-qualified one (**declined** — it weakens the contract for the R and JavaScript re-implementers, who would each have to re-derive it). The two are not blended, and the bound inside the domain is left **unqualified** with the domain as a **precondition**. **D35** is the new decision: `multiplier ≥ 0.5`, applied to the `min`/`max` vectors of **both** percent-converted keys (`precip.mean` and `precip.variance` — a `mean`-only domain would re-create `architecture-2`/`risk-7`'s defect one layer up, since D3 states the conversion as a rule over the percent **columns**), with **no upper bound**, refused at Snakefile **parse** time so `--dry-run` fails and no partial experiment exists. Enforcement lives in `prepare_cst_parameters.py` beside `_KNOWN_AXES` and the forward conversion, called from `run_stress_test.smk` beside `refuse_retired_experiment_keys`; the module is import-clean by its own lines 11–17 and `:14` is the parse-time import precedent. Validating the endpoints suffices because the levels are `np.linspace` between them; `temp` carries no domain, being additive. **The bounds were established by this revision's own measurement, not adopted**: the floor is `0.5` because the error is `ulp(|percent|)/100` against `ulp(level)`, which stays bounded for every `level ≥ 0.5` and upward forever, and first breaks at the `|percent| = 64` crossing near **0.36** — so `0.5` is a `float64` binade boundary one full binade above the first 2-ulp level, a mechanism rather than a curve fit. **E16** confirms it with **dense `float32` sweeps at every percent-binade crossing in the domain** (worst 1 ulp at 0.5, 0.68, 1.32, 1.64, 2.0, 2.28 and the level binades) plus random confirmation to `1e6` — deliberately not another random sweep, since "a normative claim standing on a sweep" is the defect being answered — and records the degradation below the floor (2 ulps in `[0.25, 0.36)`, 18 at 0.05, 72 at 0.015, 574 at 0.0016). **No ceiling is imposed**, and that is a decision: the bound measures 1 ulp to `1e6`, so a cap would refuse configurations the arithmetic serves and would tell a WG-2 reader the bound fails above it, which is false. **V20 is widened across the admitted domain including its lower boundary**, as the finding asks, and **V23** covers the refusal itself plus its must-not-fire cases. The refusal was verified safe before being specified: all four shipped seeds and `config/templates/snake_config.template.yml` declare `precip.mean` `0.7 → 1.3` and `precip.variance` `1.0`, so `pytest tests/test_cli.py` — a **required** gate — cannot turn red on it. **R13 is untouched**: it bounds nothing about outputs, and D35 bounds the multiplier. | design-v4.md |

## Tally

| Disposition | blocking | major | minor | total |
|---|---|---|---|---|
| accepted | 2 | 13 | 11 | **26** |
| accepted-in-part (a suggested fix rejected on measurement) | 0 | 2 | 0 | *(2 of the above: `risk-1`, `architecture-3`)* |
| rejected | 0 | 0 | 0 | 0 |
| deferred | 0 | 0 | 0 | 0 |
| withdrawn | 0 | 0 | 0 | 0 |

**No finding is rejected.** Two — `risk-1` and `architecture-3`, the same concern
from two lenses — have their **claim accepted** and part of their **suggested fix
rejected on measurement**, which is recorded in the Disposition column rather than
smoothed away: the fix they propose (make the round trip exact by searching for a
representable percent) is **unattainable**, and one of them additionally pins the
inverse spelling that measures *worse*. Neither is a `blocking` finding, so no
owner arbitration is triggered.

**No `blocking` finding is deferred**, and both are closed in `design-v2.md` §5.7.

## Severity divergences, preserved

| Concern | Grades | Lenses | Both rows dispositioned at |
|---|---|---|---|
| D16/D19 degenerate-axis collision | major / minor | `architecture-5` / `risk-6` | their own filed grade |
| Migration sweep incompleteness | major / minor | `repo-fit-2` / `risk-10` | their own filed grade |
| D3/D21 conversion completeness | major / minor | `architecture-2` / `risk-7` | their own filed grade |

The resolutions are shared — one fix closes each pair — but the grades are not
harmonized, per `internal-review-index.md` § *Conflicts*.

## Multi-part findings

`architecture-1` is the only finding whose limbs needed separate dispositions
(the missing WG-2 replacement; seam placement). Both are `accepted`; the second
was ruled at the G1-return (Fork A) rather than decided by this revision. Its
third limb — widening step 5b's pattern to the column names — is implemented once
and dispositioned under `repo-fit-2` / `risk-10`, which own that concern.

`architecture-6` and `architecture-4` each carry two limbs resolved by a single
disposition, noted inline.

---

# Round `ext1` — external review 1 (stage 4, `design-v2.md`)

**Appended at v3.** Reviewer `gpt-5.6-sol`, clean-room. Six findings: 2 blocking,
3 major, 1 minor, at the severities `external-review-r1.md` filed — not re-graded.
The sections above are the `internal-panel` round and are unchanged; the rows for
this round sit in the same table, keyed by `ext1`.

## Tally — `ext1`

| Disposition | blocking | major | minor | total |
|---|---|---|---|---|
| accepted | 2 | 3 | 1 | **6** |
| rejected | 0 | 0 | 0 | 0 |
| deferred | 0 | 0 | 0 | 0 |
| withdrawn | 0 | 0 | 0 | 0 |

**No `blocking` finding is deferred or rejected**, so this round triggers no owner
arbitration. Both blocking findings are closed in `design-v3.md`: `ext1-1` by D31
(§5.5, §5.8), `ext1-2` by §8 step 6 and the five other sites the contradiction
had reached.

Three rows accept the finding while **declining one branch or one clause of its
suggested fix**, recorded in the Disposition or Resolution column rather than
smoothed away, and none of the three is blocking:

- `ext1-4` — the suggested fix is an *either/or*; the "limit the claim" branch is
  taken in full and the end-to-end-experiment branch is not, with **R13** stating
  the cost that makes it unaffordable rather than leaving it implied.
- `ext1-5` — three of the four proposed negative fixtures are negatives; the
  fourth (unordered months) is **not a failure case** under D21, which sorts
  before asserting, so it is specified as a positive twin instead.
- `ext1-3` — accepted in all three parts, but the fix **narrows `repo-fit-4`'s
  closed disposition** (the `(surface_df, baseline_df)` return becomes a
  `SurfaceJoin` carrying those two under the same names). Flagged so a later round
  does not read it as drift.

## Cumulative — both rounds

| Disposition | blocking | major | minor | total |
|---|---|---|---|---|
| accepted | 4 | 16 | 12 | **32** |
| rejected | 0 | 0 | 0 | 0 |
| deferred | 0 | 0 | 0 | 0 |
| withdrawn | 0 | 0 | 0 | 0 |

32 unique finding ids, 32 rows, no id answered twice and none unanswered.

## What `ext1` says about how v2 was written

Recorded because it is the run's own lesson rather than a finding's content.
`ext1-2` is not a design disagreement: v2 was handed Fork B as **settled framing**
and wrote a step that violates it, having noticed the tension and argued through
it (`design-v2.md:750-757`) instead of stopping. The framing gate is the stop
condition for exactly that case, and returning to it costs a driver decision;
not returning to it cost a full external round. `ext1-1` is the same shape one
level down — v2 introduced case 3b/3c for held months and did not re-derive the
*leading phrase* the new cases sit under.

v3 changes nothing in `intake.md` § Constraints or § Non-goals and contradicts no
gate record in `status.md`; where a fix would have, it is not made (see `ext1-4`'s
untaken branch and `ext1-2`'s resolution, which implements Fork B rather than
re-arguing it).

---

# Round `ext2` — external review 2 (stage 4, `design-v3.md`), closed by ARBITRATION

**Appended at v4.** Reviewer `gpt-5.6-sol`, non-clean-room with a regression duty.
Three findings: 2 blocking, 1 major, at the severities `external-review-r2.md`
filed — not re-graded.

**Round 2 was the cap, so there is no third external round and no reviewer verdict
on `design-v4.md`.** Convergence failed (verdict `revise`, 2 blocking), which under
the loop's own rule sends the surviving findings to the owner. All three premises
were verified against the repo before arbitration, and **none is a pre-existing
condition — each is a defect in new design content**. The **owner's arbitration of
2026-08-15** accepted all three, required a fix for each, and additionally ruled the
*shape* of `ext2-3`'s fix. Those rulings stand in for the reviewer verdict the cap
forecloses, and each row's Disposition column records the ruling rather than an
author judgement.

## Tally — `ext2`

| Disposition | blocking | major | minor | total |
|---|---|---|---|---|
| accepted (per owner ruling) | 2 | 1 | 0 | **3** |
| rejected | 0 | 0 | 0 | 0 |
| deferred | 0 | 0 | 0 | 0 |
| withdrawn | 0 | 0 | 0 | 0 |

**Nothing was rejected and no `blocking` finding is deferred**, so no finding
proceeds unfixed. Both blocking findings are closed in `design-v4.md`: `ext2-1` by
D28's three ordered checks (§5.3) and their mirror in §5.8; `ext2-2` by §5.2's
cross-axis distinctness rule, with D13 and D33 carrying it.

One row takes a **ruled branch of an either/or suggested fix** and the other branch
is **declined by the owner, not by the author** — recorded here because a later
reader must not mistake it for an author's choice:

- `ext2-3` — the finding offers "impose and parse-time validate a domain" **or**
  "replace the bound with a domain-qualified one". The owner ruled the first and
  declined the second, on the ground that a domain-qualified bound weakens the
  cross-language contract for the R and JavaScript re-implementers. The two are
  not blended: **D35** imposes the domain and the bound stays **unqualified inside
  it**, with the domain as a precondition.

## Cumulative — all three rounds

Appended rather than edited: the two-round table above is accurate as of `ext1` and
this file is append-only.

| Disposition | blocking | major | minor | total |
|---|---|---|---|---|
| accepted | 6 | 17 | 12 | **35** |
| rejected | 0 | 0 | 0 | 0 |
| deferred | 0 | 0 | 0 | 0 |
| withdrawn | 0 | 0 | 0 | 0 |

35 unique finding ids, 35 rows, no id answered twice and none unanswered.

---

# Aggregation index — internal panel

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

---

# Internal review — risk lens (verbatim)

```yaml
verdict: revise
doc_version: design-v1.md
findings:
  - id: risk-1
    severity: major
    section: "5.1 D7 (with §7 consequence 2 and §9 V4)"
    finding: >-
      D7 asserts that keeping the float32 quantization makes "the migration's numerical
      effect on the forcing — and therefore on every indicator value — nil", and §7
      consequence 2 restates it as "the forcing is bit-identical (D7)". That is false for
      ordinary grids. Today the generator receives the shortest
      float32 text of the multiplier, read back as float64. After the change it receives
      `1 + precip_change/100` computed in R from the percent text. The concrete case:
      `step_num: 5` on the shipped 0.7–1.3 precip range yields the member value `0.82`,
      and `1 + (-18.0)/100 = 0.8200000000000001` — not the `0.82` the generator gets today.
      So does `0.3` (`1 + (-70.0)/100 = 0.30000000000000004`). Sweeping every
      `np.linspace(lo, hi, step_num+1)` grid for step_num 1..12 across six realistic
      (min,max) pairs plus 200k random multipliers in [0.2, 2.5], 48,410 of 200,540 values
      fail to reproduce the original float64 bit-for-bit, at ~1 ulp (max |Δ| = 4.44e-16).
      That fraction is over a sampled range, not over grids anyone runs — the load-bearing
      fact is that a plausible `step_num` on the shipped range is one of the failures and
      the shipped `step_num` values are not.
      The design's own preferred spelling argument was applied to the forward
      formula (`f*100-100` over `(f-1)*100`) and then not applied to the inverse: writing
      it as `(100 + precip_change)/100` cuts the mismatch count from 48,410 to 32,277 but
      still does not reach exactness. Critically, the shipped grids
      (`snake_config_baseline.yml` precip 0.7/1.0/1.3, temp 0.0/3.0) are all in the subset
      that *does* round-trip, so §9's V4 procedure — run once, on that config — cannot
      observe this. A user with `step_num: 5` gets a different forcing than before the
      migration, and V4 will have certified that nothing moved.
    rationale: >-
      This is the same shape as the incident that opened this run: an assumed no-op between
      two code paths. Two things break. (a) The claim is stated unconditionally in a
      normative decision (D7) and in a falsifiable consequence (§7-2), and the ledger will
      carry it forward. (b) §9's V4 step 3 says "a failing group means the forcing moved
      and D7's float32 round-trip discipline was not preserved" — so if V4 ever does fail,
      the design has pre-committed the implementer to the wrong diagnosis: the arithmetic
      the design itself specifies is a sufficient cause. The magnitude is ~1e-16 relative
      and is probably immaterial for most indicators, but "probably immaterial" is not what
      D7 claims, and the repo already treats ulp-level axis differences as load-bearing
      (`tests/test_check_baseline_indicator.py:244,272` exists precisely because
      `1.3` and `1.3000000000000003` must not be conflated). Measured with
      `blueearth_cst/experiment/prepare_cst_parameters.py:143,155,175` (float32 frame,
      `to_csv`, read-back) and `blueearth_cst/weathergen/impose_climate_change.R:27,68-70`
      (`read.csv` then `precip_mean_factor = cst_data$precip_mean`) as the incumbent path.
    suggested_fix: >-
      Either (a) drop the exactness claim: restate D7 and §7-2 as "the forcing changes by at
      most 1 ulp of the float32 multiplier; indicator values are expected to match within the
      baseline comparator's tolerance", and change V4's failure interpretation accordingly;
      or (b) make it exact and keep the claim: specify that rule 3.09 writes the shortest
      decimal `s` for which the R-side inverse reproduces the original float32 multiplier
      exactly (a search over shortest-repr candidates, verified at write time), and pin the
      inverse spelling `(100 + precip_change)/100` in D3/D21 rather than `1 + precip_change/100`.
      Either way, add a unit test over a non-round grid (e.g. 0.6–1.4 at `step_num: 3`,
      which yields 0.8666667) — the shipped grids cannot exercise this.

  - id: risk-2
    severity: major
    section: "5.3 D14 (reason 1), with §7 R6 and §8 step 6"
    finding: >-
      D14's first and heaviest reason for choosing a library over a rule is "There is no
      in-repo consumer. WF3 has no plotting rule … The consumers of a response surface are
      CST-API, the frontend and `csthelpers`, all out of scope by S4." That premise is false.
      `docs/notebooks/Climate Stress Test.ipynb` is a shipped, user-facing in-repo consumer
      that builds the response surface directly: it reads
      `EXP_DIR / "config" / "stress_test_design.csv"` (line 500), documents the seven-column
      indicator header (lines 481-482), and constructs the surface with
      `.groupby(["temp_change", "precip_change"])["value"] … .unstack("precip_change")`
      (lines 683-685). The design does see the notebook, but only as a stale-path chore:
      R7/R6 says "three notebooks reference `stress_test_design` and must be **re-rendered**
      after implementation (`t2608132100`)", and §8 step 6 ("docs and seeds") does not list
      the notebooks at all.
    rationale: >-
      Two consequences. First, D14's conclusion survives — a notebook consuming
      `shared/surface_axes.py` is an argument *for* the library — but the stated reason must
      be corrected, because a false premise in the load-bearing position of a decision is
      what the review is for. Second, and more practically, this notebook cannot be
      "re-rendered": the cell that produces the surface will raise `KeyError` on
      `temp_change` after the columns are dropped, so it needs a rewrite (read the lookup,
      call `surface_axes.join_axes`, group on the derived columns). C5 requires the migration
      to be one commit with every live reference updated, and `AGENTS.md` § Conventions calls
      a stale path in a document someone reads to do their job a defect. Deferring the only
      worked example of the thing this design exists to fix to a separate board item leaves
      the repo with no demonstration that the corrected axis is obtainable at all.
    suggested_fix: >-
      Correct D14 reason 1 to "no in-repo *rule* consumes a response surface; the one in-repo
      consumer is a notebook, which can import the library directly" — which strengthens the
      decision. Move the notebook rewrite into §8 step 6 as a required part of the commit,
      and make it the design's worked example of `join_axes` + `axis_caption`.

  - id: risk-3
    severity: major
    section: "5.3 D14 (join_axes), with 5.1 D4"
    finding: >-
      The design makes absence from the lookup the *sole* marker distinguishing the baseline
      from a surface member: D4 argues "absence is the strongest available marking", and D14
      implements S6 as a partition — `join_axes` returns "indicator rows whose `st_id` is in
      the lookup" and, separately, "the rows whose `st_id` is absent from the lookup, which is
      exactly `st_0`". No postcondition is specified on that partition. The design guards the
      *lookup* read (`read_lookup` forces `dtype={"st_id": str}`, D2/D14) but `join_axes`
      accepts `indicators` as an already-loaded frame and constrains nothing about how it was
      read. Any `st_id` representation mismatch — the caller reading the indicator table with
      default dtypes, an `st_id` written at a different pad width, a lookup from a different
      experiment — makes the membership test all-False, and the function then returns an
      empty surface and classifies **every** row as "baseline" without raising.
    rationale: >-
      D2 and §5.7 both warn, at length, that `pd.read_csv` with no `dtype` turns `01` into `1`
      and "the join to the lookup silently misses" — the design knows the hazard exists and
      names it, then chooses an encoding (absence = baseline) that converts that exact miss
      from a visible empty result into a plausible-looking one, because "all rows are baseline"
      is a shape the partition is designed to produce. Whether it is silent depends on an
      unpinned implementation choice: an `isin`-style membership partition — which is how
      D14's prose describes it ("rows whose `st_id` is in the lookup") — returns all-False
      without raising, while a `pd.merge` on object-vs-int64 keys raises. Pinning which one,
      plus the postcondition below, is the fix. The pre-change code has no equivalent
      failure: the axis is a column on the row. And the run-time coverage check D22 adds lives
      in the *reducer*, upstream of the join, so it cannot catch a stale or mismatched lookup
      at report time; `validate_hm7` is test-time only, as §5.6 itself notes.
    suggested_fix: >-
      Give `join_axes` a stated postcondition and make it raise: the set of `st_id` absent from
      the lookup must be exactly the baseline token, and the surface partition must be
      non-empty. Either have the library own the indicator read (a `read_indicators(path)`
      alongside `read_lookup`) or normalise both key columns to zero-padded strings inside
      `join_axes` before partitioning.

  - id: risk-4
    severity: major
    section: "5.2 D12, with §7 consequence 3 and §9 V5"
    finding: >-
      D12 defines the statistic as "the month-length-weighted mean over the declared months
      (`_MONTH_LENGTHS` … matching `annual_perturbation`)" and argues that "Under [D16]'s
      homogeneity constraint the weighting is immaterial — every value being averaged is
      equal". The incumbent function says the opposite, in code and in a comment written for
      this exact case: `export_wflow_results.annual_perturbation` short-circuits
      `if np.ptp(values) == 0: return float(values[0])`, documented as "Flat vectors
      short-circuit on exact equality rather than falling through the weighted mean, which
      would round twelve identical values to something a unit in the last place away from
      them" (`blueearth_cst/experiment/export_wflow_results.py:105-107,122-124`). I measured
      it: `np.average(np.full(12, 1.3), weights=_MONTH_LENGTHS)` returns
      `1.3000000000000003`, and the same for `-6.666665999999992` → `-6.666665999999991`.
      Under D16 the flat case is not an edge case — it is the *normal* path for every
      admissible axis. The design never mentions the short-circuit, so a faithful
      implementation of D12 as written drops it.
    rationale: >-
      §7 consequence 3 claims "The default axis is unchanged for a uniform design", with
      falsifier "a uniform design whose derived axis differs from today's
      `annual_perturbation` output" (V5). Stated as an equality, that falsifier fires for any
      grid whose values are not exactly representable — e.g. a 0.6–1.4 grid at `step_num: 3`
      gives 0.8666667. It does *not* fire for the shipped 0.7/1.0/1.3 grids, so, as with
      risk-1, the fixture cannot see it. The repo already treats this ulp as significant:
      `tests/test_check_baseline_indicator.py:244` exists because `temp_change` reparsed as
      float would conflate `1.3` and `1.3000000000000003`.
    suggested_fix: >-
      State the short-circuit as part of D12's normative definition ("when the declared months'
      values are exactly equal, the statistic returns that value; otherwise the month-length
      weighted mean"), or retire it explicitly and restate V5 as equality within a stated
      tolerance. Delete the "the weighting is immaterial" sentence — it is the claim the
      incumbent code was written to refute.

  - id: risk-5
    severity: minor
    section: "5.2 D10 and 5.7 (the S10 overlay clause)"
    finding: >-
      D10 states the limit it wants nobody to rediscover as a bug: "two surfaces from one
      experiment differ in magnitude and label, not in shape or member ordering." That is true
      of the stress-test members and false of the CMIP6 overlay S10 binds to the same collapse.
      The stress members are homogeneous by D16, so `axis(st)` equals the change imposed in
      each declared month, and rescaling the month set rescales every member identically. A
      GCM's monthly change factors are *not* homogeneous, so the weighted mean over M is a
      genuine average of unlike months, and it moves non-affinely as M changes: two GCMs that
      sit at the same annual mean can sit at opposite ends of a JFM axis. So under D10 a user
      may declare several surfaces from one run, and each one places the same GCM cloud
      differently against the same response surface. Because D8 deliberately puts `reporting:`
      outside `CONFIG_PROJECTION` and outside `effective_config_digest`, and R5 records that
      `run_metadata.json` does not witness which surface a figure was drawn under, that is a
      change in a plausibility judgement made by a config edit that leaves no run-identity
      trace.
    rationale: >-
      I am not re-opening S10 or D10 — I am naming a consequence the document has not faced.
      §5.7's normative contract text asserts the transfer is cheap ("the same month-set collapse
      runs over the GCM table and over the lookup with no unit conversion between them"), which
      reads as "the collapse transfers cleanly". It transfers arithmetically and not
      semantically: the very quantity D16 refuses to put on the stress axis — "the mean averages
      unlike perturbations and no caption can describe it honestly" — is exactly what the GCM
      side of that formula computes, by construction. The asymmetry exists today too (annual
      mean of 12 unequal GCM months vs. a flat stress design), but D11's default shrinks M to
      the varying set, which reduces the averaging and amplifies the sensitivity; and D10 makes
      "which M" a per-surface, freely editable choice rather than a fixed one. This lands in
      normative text now even though the overlay is deferred to Q6.
    suggested_fix: >-
      Add one clause to §5.7's overlay paragraph stating the asymmetry explicitly (on the lookup
      side the mean is over equal values by D16; on the GCM side it is not, so the overlay dot is
      a summary and the axis is an imposed value), and add a sentence to D10 limiting its
      "magnitude and label only" claim to the surface members. Note in OQ-2 that Q6 must decide
      whether the overlay is recomputed per declared surface and whether that recomputation is
      recorded anywhere. Filed minor because the asymmetry is pre-existing (today's annual mean
      of twelve unequal GCM months against a flat stress design has the same shape) and no
      overlay code lands here; the actionable content is one caveat clause in §5.7 and one in D10.

  - id: risk-6
    severity: minor
    section: "5.4 D16 and 5.5 D19 (and 5.2 D11 / §5.5 step 1)"
    finding: >-
      Two normative rules collide on a config the design calls legitimate. D19 rules that an axis
      with no varying months is "degenerate, not an error", using a temperature-only stress test as
      the worked example. D16 rules that "the declared set must be a **non-empty subset of the
      varying set**; a held month raises `HeldMonthInAxisError`". On a degenerate axis the varying
      set is empty, so *every* explicit `months:` declaration on that axis raises — only the D11
      default path (which special-cases the degenerate case to all twelve) survives. A user who
      writes `y: {variable: precip, months: [1,2,3]}` on a temp-only design gets an error for a
      design D19 declares legal. Separately, the varying/held threshold is spelled two ways and
      never fixed: D11 says `max - min > 0`, §5.5 step 1 says `max - min > tol`, and `tol` is
      defined nowhere.
    rationale: >-
      Both are cheap to close and both are the kind of gap that surfaces as a support question
      rather than a test failure. The `tol` question is not cosmetic once percent conversion is in
      play: a config with `min: 0.8` / `max: 0.8000000001` classifies as varying under `> 0`, joins
      the axis, then trips D16's homogeneity refusal — a confusing path for what the user meant as a
      held month.
    suggested_fix: >-
      State that on a degenerate axis an explicit `months:` declaration is admitted and returns the
      constant (the subset rule applies only when the varying set is non-empty), and fix one
      threshold — `max - min > 0` — in both places, or define `tol` once in D11 and cite it from §5.5.

  - id: risk-7
    severity: minor
    section: "5.1 D3 and 5.6 D21"
    finding: >-
      D3 states "Two conversion sites, and only two" and tabulates the `precip_change` conversion
      alone; D21 says `impose_climate_change.R` "converts percent to the generator's factor form
      (`1 + precip_change/100`)". But `precip_variance` is also a multiplier consumed by the
      generator — `impose_climate_change.R:69` passes `precip_var_factor = cst_data$precip_variance`
      — and S1/D2 make its lookup column `precip_variance_change` a percent. The normative text
      therefore leaves one of the three lookup value columns without a stated inverse, and the
      "only two" claim is wrong.
    rationale: >-
      Every shipped config sets `variance` `min: [1.0…]` / `max: [1.0…]`, so the column is
      identically zero and converts back to 1.0 exactly on the fixture — the omission cannot fail any
      gate the repo can run. That is precisely why it belongs in the normative text rather than in the
      implementer's head, and it compounds risk-1: the untested variance path inherits the same
      inverse-formula question.
    suggested_fix: >-
      Add the `precip_variance` row to D3's conversion table and name it in D21 alongside
      `precip_change`; change "two conversion sites" to "one site per direction, covering both
      multiplier columns".

  - id: risk-8
    severity: minor
    section: "5.6 D21 (the R-side filter)"
    finding: >-
      D21 replaces rule 3.12's per-member `st_csv` with the constant `lookup_csv` and has
      `impose_climate_change.R` "filter `st_id == <padded token>` … order by `month`". No
      postcondition is specified on the filter. `impose_climate_change.R` currently reads a file
      that *is* the member (12 rows, no id) and passes `cst_data$precip_mean` straight into
      `apply_climate_perturbations` (lines 27, 68-70); after the change, a filter that matches
      nothing yields a zero-length vector, and a filter that matches partially yields a short one.
      R will pass either into `apply_climate_perturbations` without complaint from this script.
      The token equality D2 asserts (lookup `st_id` textually identical to the member filename
      token, C27) is exactly the thing that would silently stop holding under a pad-width change,
      and it is now checked nowhere on the consuming side.
    rationale: >-
      Today a wrong member path is a `MissingInputException` from Snakemake — loud, and structural.
      After D21 the same class of error becomes a data condition inside an R script that has no
      guard for it. This is the seam where a silent wrong-perturbation run would originate, and the
      whole point of §5.4's two-tier enforcement is that a correctness property gets an executable
      check at the point of use.
    suggested_fix: >-
      Specify in D21 that the R script stops unless the filtered frame has exactly 12 rows whose
      `month` values are the twelve calendar months — the same assertion `annual_perturbation`
      already makes on the Python side (`export_wflow_results.py:110-121`), and cheap to mirror.

  - id: risk-9
    severity: major
    section: "§7 consequence 1 (and §9 P2, §5.2 D8)"
    finding: >-
      Consequence 1 claims "A relabel needs no re-run", with falsifier "an edit that trips
      `ExperimentConfigFrozenError` or **re-fires 3.02**/3.16b". The 3.02 half of that falsifier is
      met by construction: rule 3.02 declares `config_snake = config_path` as a plain (non-`ancient`)
      input (`run_stress_test.smk:605`), so any edit to the config file — including a top-level
      `reporting:` edit — makes it newer than
      `<exp>/config/snake_config_run_stress_test.yml` and Snakemake re-runs 3.02 on the next
      invocation. P2 measured `effective_config_digest` and `_frozen_differences`, which are repo
      helper functions, not Snakemake's rerun triggers; the probe as run cannot answer the question
      the claim is about.
    rationale: >-
      Verified that the re-fire does not cascade in the DAG: no rule in any `*.smk` declares
      `snake_config_run_stress_test.yml` or `run_record.yml` as an input — `run_stress_test.smk:535`
      is a `WF3_TARGETS` entry and `:620-621` the outputs, and that is the whole set. But it is not
      free, because a non-rule consumer does read it:
      `dev/scripts/check_baseline.py:326-329` carries
      `("run_stress_test", "yaml", "{exp_dir}/config/snake_config_run_stress_test.yml")` as a
      baseline target. So on the baseline tree, editing `reporting:` re-fires 3.02, the snapshot
      gains or changes the section, its `yaml` fingerprint moves, and `check_baseline.py check`
      reports a difference — for a pure relabel that D8's whole tier story says costs nothing. The
      design discusses the baseline gate at length (R1, R2, §8 steps 0 and 7) entirely in terms of
      the `indicator` target and never reaches this one; §8 step 6 contemplates a seed declaring a
      surface, which is precisely the configuration that arms it. This upgrades what would otherwise
      be a wording defect: the new section's placement has a gate consequence the design has not
      priced.
    suggested_fix: >-
      Narrow consequence 1 to what P2 measured ("a relabel does not trip the experiment freeze and
      does not move `effective_config_digest`, so 3.16b and the record's staleness verdict are
      untouched"), and add the two costs it does carry: 3.02 re-fires by mtime on any config edit,
      and because the WF3 config snapshot is a `yaml` baseline target, a `reporting:` edit on the
      baseline tree turns `check_baseline.py check` red. Decide explicitly whether the recorded
      snapshot should carry `reporting:` at all — recording it satisfies S2's "the collapse is a
      choice that must be recorded", but it is what couples a caption to the numerical gate.

  - id: risk-10
    severity: minor
    section: "§8 step 5b (sweep the test suite for the old roots)"
    finding: >-
      Step 5b specifies the sweep as `rg -n "_work/|stress_test_design" tests/` and presents its
      four-file result as the affected surface, with "a non-empty result is the failure" as the exit
      condition. The pattern misses test files that pin the retired seven-column header by literal
      without naming either token: `tests/test_check_baseline_scope.py:56`
      (`"metric,st_id,temp_change,precip_change,rlz_id,location,value"`) and
      `tests/test_check_baseline_indicator.py:62-63,268,272` (a five-site fixture including
      `assert df.loc[0, "temp_change"] == "1.3000000000000003"`). Both build synthetic tables, so
      they will keep passing after the change while asserting a contract that no longer exists.
    rationale: >-
      §8 step 5b is written against the R9 precedent that a tree-shape migration left 22 stale-path
      failures, three of them silent. The failure mode here is the milder cousin — a test that passes
      while encoding the retired shape — but it defeats the step's own exit condition, and
      `check_baseline.py`'s `read_indicator_table` (which the design already flags as going stale in
      this commit) is the module those two files cover.
    suggested_fix: >-
      Widen the sweep to `rg -n "_work/|stress_test_design|temp_change|precip_change" tests/` and
      exclude the WF2 hits (`tests/test_get_change_climate_proj_summary.py`, where the names are WF2
      change-factor variables and are unaffected) by inspection rather than by pattern.
```

---

# Internal review — architecture lens (verbatim)

```yaml
verdict: revise
doc_version: design-v1.md
findings:
  - id: architecture-1
    severity: blocking
    section: "5.7 HM-7 replacement text / 8 Migration plan"
    finding: >-
      The design deletes the artifact that the WG-2 contract pins, and never mentions WG-2
      anywhere. `<exp>/climate/weathergenr/_work/st_<m>.csv` is a NAMED SEAM ARTIFACT with its
      own contract section (`dev/reference/contracts/weather-generator-seam.md:110-135`), its own
      pinned header constant (`interchange_contracts.py:224` `_WG2_HEADER`), its own validator
      (`interchange_contracts.py:227` `validate_wg2`), and three tests. §5.7 replaces HM-7 only;
      §8 step 5 names `interchange_contracts.py` but scopes it to `validate_hm7` and
      `_PERTURBATION_AXIS`; no step touches `weather-generator-seam.md` at all; and step 5b's
      sweep pattern `rg -n "_work/|stress_test_design" tests/` provably cannot find
      `tests/test_interchange_contracts.py:143-160`, whose `_wg2_good()` fixture is built
      in-memory and contains neither string. So after the migration `validate_wg2` still asserts
      the header `month,temp_mean,precip_mean,precip_variance` on a file that no longer exists,
      two synthetic tests keep passing GREEN against a dead contract, and the seam document still
      documents the deleted artifact as live. Second half of the same defect: the lookup IS the
      Python-to-R parameter handoff (rule 3.09 writes it, `impose_climate_change.R` reads it),
      which is the weather-generator seam by definition — WG-2's own text calls the member CSV
      "the only record of the `precip_variance` axis and of the monthly structure the reduction
      collapses". The design instead puts the lookup's full schema into HM-7, whose declared
      consumer is "CST-API / GUI (terminal in-repo)". The response-surface seam is the wrong home
      for a cross-language parameter contract, and the correct home is left describing a deleted
      file.
    rationale: >-
      C5 makes "every live reference updated in one commit" a decision criterion, and §8 presents
      a file-by-file check ("Machinery that must move with it, checked file by file"). That check
      is incomplete on a whole contract, a whole validator and a whole seam document. The failure
      mode is exactly the one `AGENTS.md` records from R9 — a test that keeps passing while the
      thing it validates is gone — and here it is worse than a silent skip, because
      `test_wg2_synthetic_pass` is a POSITIVE assertion that a retired contract holds.
      `test_wg2_integration` (`tests/test_interchange_contracts.py:827-834`) would fail, but it
      is behind `_FIXTURE_ABSENT` and so skips in any worktree (R3). Verified:
      `dev/reference/contracts/weather-generator-seam.md:110-135`,
      `blueearth_cst/shared/interchange_contracts.py:224,227-247`,
      `tests/test_interchange_contracts.py:143-160,827-834`, and grep for `WG-2`/`validate_wg2`
      over `design-v1.md` returns nothing.
    suggested_fix: >-
      Add a §5.8 giving WG-2's replacement text (either a rewritten WG-2 pinning
      `<exp>/config/stress_test_lookup.csv` — header, `12 x ST_NUM` rows, `st_id` string domain,
      percent semantics, no `st_0` row — or an explicit retirement of WG-2 with the lookup
      registered as a new WG entry), and add `weather-generator-seam.md`, `_WG2_HEADER`,
      `validate_wg2` and `tests/test_interchange_contracts.py:143-160` to §8 step 5/5b. Then
      broaden step 5b's sweep: the pattern must include the COLUMN names and the header tokens
      (`temp_mean|precip_mean|precip_variance|temp_change|precip_change`), not only the two path
      strings. That widened pattern also surfaces `tests/test_check_baseline_scope.py:56` and
      `tests/test_check_baseline_indicator.py:62,244,268,272`, which build synthetic seven-column
      indicator tables and are likewise absent from §8 and from §9's narrow-tier gate list.
  - id: architecture-2
    severity: major
    section: "5.1 D3 / 5.6 D21"
    finding: >-
      `precip_variance_change` is converted to percent on the way out and never converted back on
      the way in. D3 states "Two conversion sites, and only two", and both rows name
      `precip_change` alone; D21 repeats only "converts percent to the generator's factor form
      (`1 + precip_change/100`)" and then says "Everything else in that script … is untouched".
      But `impose_climate_change.R:70` passes `precip_var_factor = cst_data$precip_variance`, a
      MULTIPLIER, and S1/D2 make the lookup column a percent. Implemented literally, the R hands
      `apply_climate_perturbations` a variance factor of `precip_variance_change`, and every
      shipped config sets `precip.variance` min = max = 1.0 twelve times
      (`test_case/snake_config_baseline.yml:82-84`), i.e. 0.0 percent — so the generator receives
      a variance factor of ZERO on the default configuration rather than the identity 1.0.
    rationale: >-
      The three-column unit change is normative (S1, restated in §5.7's HM-7 text as "temp_change
      in degC, the other two in percent"), but the inverse is specified for one of the two
      converted columns. D3's own framing — "Two conversion sites, and only two" — is what makes
      this a defect rather than an elision: it asserts completeness. The forcing consequence is
      not subtle, and the only thing that would catch it is V4, which is a one-off manual
      procedure run at step 7, after the shape change has landed.
    suggested_fix: >-
      Give D3's table a third row (`precip_variance = 1 + precip_variance_change / 100`) and state
      the conversion as a rule over the two percent columns rather than as a per-column formula,
      so a future column inherits it. Add a V-claim for it: the R-side factors reconstructed from
      the lookup equal the pre-migration `st_<m>.csv` values, checked on one member.
  - id: architecture-3
    severity: major
    section: "5.1 D7 / 7 Consequences 2 / 9 V4"
    finding: >-
      D7's "the migration's numerical effect on the forcing — and therefore on every indicator
      value — is **nil**" is false in general, and the evidence offered for it is a single value.
      D7 measures `1.3*100-100 == 30.0` exactly and concludes "The inverse restores 1.3 exactly".
      Measured this review over `np.linspace([0.7]*12, [1.3]*12, n, axis=1)` for n = 2..11, the
      composition `f -> f*100-100 -> 1 + p/100` is NOT the identity for n = 6, 8, 10 and 11: at
      n = 6 the level `0.82` returns as `0.8200000000000001`; at n = 8, `0.78571427` returns as
      `0.7857142699999999`; at n = 10, `0.76666665` returns as `0.7666666499999999`. Over 200,000
      random float32 multipliers in [0.5, 1.6], 17% fail to round-trip. The shipped configs
      (`precip.step_num` 1 and 2, i.e. 2 and 3 levels) DO round-trip exactly, so the migration's
      own gate passes — but a `step_num: 5` project does not.
    rationale: >-
      This is a consequence of S1, not a re-litigation of it: S1's carried rationale dismisses the
      "store what is applied" counter-argument on the ground that the incident behind it "was a
      float32-vs-float64 CSV round-trip problem, not a unit-choice one, and §5.1 D7 resolves it
      directly." The measurement above shows the UNIT CONVERSION introduces its own
      non-invertibility, independent of float32, and D7 does not resolve it — because the design
      deletes the only artifact that stored the multiplier the generator actually receives, and
      replaces it with one from which that multiplier is reconstructed. Two downstream statements
      inherit the error: §7 consequence 2 ("the forcing is bit-identical (D7)") and §9's V4
      diagnosis rule ("a failing group means the forcing moved and D7's float32 round-trip
      discipline was not preserved"), which would mis-attribute a genuine 1-ulp conversion
      artifact on a 6-level grid to a lost round-trip discipline.
    suggested_fix: >-
      State the exactness condition rather than the absolute: the round trip is exact when the
      float32 shortest repr is a 2-decimal value, which covers every shipped config and is what
      makes V4 a valid gate ON THOSE CONFIGS. Then either (a) accept the residual and say in §7
      that a general grid may move the forcing by ~1 ulp in the factor, with V4's diagnosis rule
      qualified accordingly, or (b) remove the residual by specifying that rule 3.09 writes the
      percent as `float(str(np.float32(f*100-100)))` AND that the R reconstruction is asserted
      against the pre-migration value in the step-7 check.
  - id: architecture-4
    severity: major
    section: "7 Consequences 1 / 5.2 D8 / 6.3"
    finding: >-
      Falsifiable consequence 1 names "re-fires 3.02/3.16b" as a falsifier, and that falsifier
      fires on correct behaviour. Rule 3.02 declares `config_snake = config_path` as a PLAIN
      `input:` (`run_stress_test.smk:604-605`), not `ancient()`, so editing the config file to
      change `reporting.surfaces` makes the input newer than the snapshot output and Snakemake
      re-runs 3.02 on the mtime trigger — regardless of `effective_config_digest`, which is what
      P2 actually measured. The same edit leaves 3.16b alone (its params carry only the two
      digests, `run_stress_test.smk:1146-1148`), so 3.16b and the freeze ARE discriminators
      between the two homes; 3.02 is not, and §6.3's rejection sentence "the effective-config
      digest moves, so a caption edit re-fires the record rules" should be scoped to 3.16b.
      The second limb is a circularity in D8: it claims both that "a relabel needs no re-run" and
      that S2's obligation to record the collapse choice "is met" because "rule 3.02 byte-copies
      the config as run". Those cannot both hold for the same edit — either you re-run WF3, and
      3.02 re-fires (contradicting consequence 1), or you do not, and `<exp>/config/
      snake_config_run_stress_test.yml` still carries the OLD `reporting:` section, so the
      relabel is precisely the thing NOT recorded.
    rationale: >-
      A stated falsifier that fires on a working design is worse than no falsifier: whoever runs
      V12 will see 3.02 re-run and have to decide whether to believe the gate. And the recording
      argument is the whole answer D8 gives to S2's "the collapse and the caption are a choice
      that must be recorded", so its circularity leaves the tier story with a hole. Checked at
      `run_stress_test.smk:602-623` (3.02's inputs and outputs) and `:1140-1155` (3.16b's params).
    suggested_fix: >-
      Restate consequence 1 as "the effective-config digest and `_frozen_differences` are
      unchanged; 3.02 re-writes the snapshot on the config file's mtime, which is the intended
      way the relabel gets recorded, and 3.16b does not re-fire". Then say explicitly that the
      surface declaration is recorded only in a snapshot refreshed by a (cheap) re-invocation of
      WF3, and that a relabel made without any WF3 invocation is unrecorded — that is the honest
      version of R5.
  - id: architecture-5
    severity: major
    section: "5.4 D16 / 5.5 D19 / 5.2 D11"
    finding: >-
      D16 and D19 give contradictory rulings on the degenerate axis, and D11's own default
      violates D16 exactly there. D16 states "the declared set must be a non-empty subset of the
      varying set; a held month raises `HeldMonthInAxisError`" — restated normatively in §5.7's
      HM-7 text as "`M` must be a non-empty subset of the varying months". D19 states that an
      axis with no varying months "is degenerate, not an error" and returns the constant. When
      nothing varies the varying set is empty, so no non-empty subset exists and D16 refuses the
      case D19 admits. D11 then makes it concrete: its default "when nothing varies … is all
      twelve" months, i.e. the design's own default month set for a degenerate axis is twelve
      HELD months — the precise input D16 says must raise. The design nowhere states which check
      runs first. The same ambiguity covers an EXPLICIT declaration on a degenerate axis
      (`y: {variable: precip, months: [1,2,3]}` on a temp-only design): D16 says raise, D19 says
      return the constant.
    rationale: >-
      D19 exists to keep a temperature-only stress test legal, which is a real and common design;
      D16 exists to keep C1's misreport from returning. Both are right and their interaction is
      unspecified, so an implementer picks one and the other silently stops holding. D11 also
      mis-cites the degenerate case as "D17's degenerate axis" — D17 is the rectilinearity
      postcondition; the degenerate axis is D19. That cross-reference error is what makes the
      collision easy to miss on a read.
    suggested_fix: >-
      State the precedence explicitly in D16: classify months first; if the varying set is empty
      the axis is degenerate (D19) and neither the subset rule nor the homogeneity rule applies;
      otherwise the declared set must be a non-empty subset of the varying set. Mirror that
      ordering into §5.7's HM-7 text, which currently states the subset rule unconditionally, and
      fix D11's citation.
  - id: architecture-6
    severity: major
    section: "5.3 D14 / 5.4 D16-D17 / 5.5 D18"
    finding: >-
      The enforcement G4 promises and the caption algorithm G5 specifies have no caller. D14's
      first and heaviest reason for a library rather than a rule is "There is no in-repo
      consumer"; D13 gives the library exactly one in-repo call site, `parse_surfaces(config)` at
      Snakefile parse time, which per §5.4 runs only the DESIGN-tier warning. So `axis_values`,
      `axis_caption` and `join_axes` are never invoked on any repo execution path: D16's axis-tier
      refusal, D17's rectilinearity postcondition and the whole §5.5 caption algorithm execute
      only in the unit tests V7-V11 and in a hypothetical Python consumer that does not exist.
      Meanwhile the actual consumers are R and JavaScript (S4), and what they get is §5.7's HM-7
      text — which carries the collapse formula and the two constraints as prose, but NOT the
      caption algorithm, NOT the varying/held classification tolerance (`max - min > tol`, with
      `tol` undefined), and NOT the degenerate-axis behaviour. The consequence for C1 is direct:
      the check that stops the annual misreport from returning through a held month in `M`
      (`HeldMonthInAxisError`) is unreachable for the only consumers that will ever draw a
      surface.
    rationale: >-
      The intake's gap 2 is "the consumer side is unassigned … something must join lookup x
      indicators at plot time". The design assigns it to a Python library and then establishes
      that nothing in Python will call it. That is internally consistent but it means the
      derivation is now specified in two voices at two strengths — a full algorithm in §5.5 for a
      caller that does not exist, and a weaker prose formula in HM-7 for the callers that do —
      which is the drift condition. §7 carries R1-R7 and none of them is this. Confirmed:
      `run_stress_test.smk` rules 3.01-3.18 contain no plotting rule, and §5.5's caption cases
      are rendered nowhere in-repo.
    suggested_fix: >-
      Either give the library one real in-repo caller — the cheapest is a `--dry-run`-invisible
      assertion at the end of rule 3.16 that every declared surface's axis derives without
      raising, which costs the lookup back as a 3.16 input and should be weighed against D22 — or
      accept it and (a) add it to §7 as a named risk stating that D16-axis/D17 are unit-test-only
      and the R consumer is unguarded, and (b) move the classification tolerance, the degenerate
      rule and at least the caption's case table into §5.7's HM-7 text, so the normative document
      an R re-implementer reads is complete.
  - id: architecture-7
    severity: major
    section: "5.7 HM-7 replacement text"
    finding: >-
      §5.7's opening non-interaction claim is false against the document it replaces into.
      It lists as unchanged "the HM-4 -> HM-5 -> HM-7 gauge-column invariant, which does not touch
      these columns" — but that section's own check-3 text reads "`qstats_df`'s gauge columns
      (header minus `statistic` and the `_PERTURBATION_AXIS` columns `temp_change,precip_change`,
      ordered per `export_wflow_results.py:66-67`)"
      (`dev/reference/contracts/hydrological-model-seam.md:420-422`, in the HM-4->HM-5->HM-7
      relational-validator section), naming
      both the deleted columns and the symbol §8 step 5 deletes. Two further HM-7 bullets are on
      neither list — neither restated in the replacement nor named as unchanged: "axis-column
      rename (2026-08-05)" (`:346-354`), which ends "Both spellings are named once in code, as
      `interchange_contracts._PERTURBATION_AXIS`", and "axis VALUE, not just its name (2026-08-07,
      [R9-3])" (`:355-368`), which documents the annual collapse as the columns' definition. Read
      literally ("Everything not restated below is unchanged"), the drop-in leaves all three
      passages in place describing removed columns and a deleted symbol.
    rationale: >-
      §5.7 is a deliverable, not a summary — the intake's success criterion is "an HM-7
      replacement that can be dropped into `hydrological-model-seam.md`" — and C5 makes every live
      reference the commit's obligation. A drop-in whose scope clause is wrong on three passages
      cannot be dropped in. The `_PERTURBATION_AXIS` case is the sharp one: step 5 deletes the
      symbol and fixes only the code docstring at `interchange_contracts.py:1141`, while the seam
      document's own prose names it twice more.
    suggested_fix: >-
      Change §5.7's scope clause from an exclusion list to an explicit disposition per bullet, and
      state that the two axis-column bullets are DELETED (their content is subsumed by the removal
      note and the derivation spec) and that the HM-4->HM-5->HM-7 check-3 sentence is amended to
      "header minus `statistic`" with the `_PERTURBATION_AXIS` clause struck. Add
      `hydrological-model-seam.md`'s relational section to §8 step 5's edit list alongside the
      docstring.
  - id: architecture-8
    severity: minor
    section: "5.6 D21 / 9 Claim -> falsifier"
    finding: >-
      The R side of the new seam is under-specified and carries no falsifier. D21 says the member
      id arrives "as a new positional argument" and that "Everything else in that script … is
      untouched" — but `impose_climate_change.R:12-14` hard-fails on `length(args) != 4L`, so the
      arity check must go to 5 and the claim is literally false. More substantively, the script
      today receives a file that IS the twelve rows; the design replaces that with a filter-and-
      order join, and specifies no post-join assertion. A join that returns 0 rows (the dtype
      hazard D2 names, a width mismatch, a wrong wildcard) hands `apply_climate_perturbations` a
      zero-length `precip_mean_factor`/`temp_delta`, where R's recycling rules make a silent wrong
      answer at least as likely as an error. V1-V14 contain no R-side claim at all.
    rationale: >-
      C6 requires every claimed runtime property to have a falsifier, and this is the one seam the
      migration moves across a language boundary. The Python side keeps its arity guard
      (`export_wflow_results.annual_perturbation` raises when a column has other than twelve rows
      and again when the `month` domain is not 1..12, `export_wflow_results.py:108-120`); the R
      side, after this change, has none.
    suggested_fix: >-
      Specify in D21 that the R asserts `nrow(cst_data) == 12L` and
      `identical(cst_data$month, 1:12)` after the filter and sort, with a `stop()` naming the
      member token; update the arity to 5; and add a V-claim whose falsifier is the join returning
      a non-twelve row count.
  - id: architecture-9
    severity: minor
    section: "5.2 D9 / D11"
    finding: >-
      `DEFAULT_SURFACE` is given `id: annual` while both its axes take D11's member-varying month
      set. For a JFM design the default surface is therefore identified as `annual` and captioned
      "mean change over JFM; Apr-Dec unchanged" — the id asserts the collapse D11 exists to stop
      being the default.
    rationale: >-
      D18's premise is that a typed label drifts from the design it describes and a derived one
      cannot; the surface `id` is a typed label, and D14 says "the surface `id` names the frame",
      so it is what a consumer will use to identify the figure. `annual` is accurate only for the
      uniform case.
    suggested_fix: >-
      Name it `default` (or `varying`), and reserve `annual` for a surface a user declares with
      `months: [1..12]` — which D16's subset rule will refuse on a seasonal design, correctly.
  - id: architecture-10
    severity: minor
    section: "10 OQ-1"
    finding: >-
      OQ-1's premise is false on disk. It states this work "lands **before** R12 (S8) and carries
      no milestone directory of its own", and offers as its first option "create
      `dev/milestones/r12/` early and file it there". That directory already exists and already
      holds `dev/milestones/r12/g2-assessment.md`.
    rationale: >-
      OQ-1 was put to the owner at G1 and carried forward unruled; the option set it presents is
      what the owner will rule on. One of the three options is already satisfied, which changes
      the ruling from "create a directory early" to "file into an existing milestone directory" —
      a materially cheaper decision. Verified by listing `dev/milestones/`.
    suggested_fix: >-
      Restate OQ-1 with the observation, so the owner rules on where in an EXISTING `r12/` the
      note lands rather than on whether to open the directory.
```

---

# Internal review — repo-fit lens (verbatim)

```yaml
verdict: revise
doc_version: design-v1.md
findings:
  - id: repo-fit-1
    severity: blocking
    section: "5.7"
    finding: >-
      The design breaks TWO interchange contracts and writes replacement text for only
      one. §5.7 gives a careful drop-in replacement for HM-7, but the artifact being
      deleted — `<exp>/climate/weathergenr/_work/st_<m>.csv` — is the subject of
      **WG-2**, a contract of its own in `dev/reference/contracts/weather-generator-seam.md`,
      with a pinned header constant, a validator (`validate_wg2`), a synthetic test pair,
      and a real-fixture integration test. The words "WG-2", "weather-generator-seam" and
      "validate_wg2" do not appear anywhere in design-v1.md, and §8's step-5 file list
      ("`validate_hm7` in `shared/interchange_contracts.py`; the HM-7 section (§5.7);
      `semantic_tree_diff.py`; `rule-index.md`") does not name any of them. The design
      therefore cannot land in one commit as C5/G6 require, because it does not yet know
      one of the two contracts it retires.
    rationale: >-
      `dev/reference/contracts/weather-generator-seam.md:110-117` declares WG-2 as
      "`<exp>/climate/weathergenr/_work/st_<m>.csv` (`m >= 1`) … Demoted to `_work/` by
      R07 B6 but **retained**: it is the only record of the `precip_variance` axis and of
      the monthly structure the reduction collapses. Also a **declared `input:` on rule
      3.16**" — every clause of which this design falsifies (D20 collapses the outputs,
      D22 drops the 3.16 input, S5 deletes `_work/`). Line 330 of the same file lists
      `validate_wg2` in the validator table with fixture path
      `<exp>/climate/weathergenr/_work/st_<m>.csv` and "continuously verified? **yes
      (persists)**" — a claim that becomes false. The code side is
      `blueearth_cst/shared/interchange_contracts.py:223` (`#: WG-2 stress-test CSV
      header, exact and ordered.`) and `:227` (`def validate_wg2`). Test sites:
      `tests/test_interchange_contracts.py:155,160` (synthetic) and `:826-834`
      (`test_wg2_integration`, real fixture). I grepped `validate_wg2` across all `*.py`
      and it is called from nowhere but those tests — no `script:` module invokes it — so
      the fix is bounded: contract text, validator, header constant, three test sites. It
      is bounded but it is not optional, and the design's own §5.7 sets the standard for
      what "handled" looks like.
    suggested_fix: >-
      Add a WG-2 disposition alongside §5.7 — either a drop-in replacement pointing WG-2
      at `<exp>/config/stress_test_lookup.csv` with the new five-column header and the
      `st_0`-absent rule (the more likely correct answer: the lookup IS the weather
      generator's input contract, it merely moved and changed shape), or an explicit
      retirement recording that WG-2's guarantees are absorbed by HM-7's lookup
      specification. Name `weather-generator-seam.md`, `validate_wg2`, the WG-2 header
      constant and `test_wg2_integration` in §8 step 5, and add a WG-2 row to §9's
      claim → falsifier table.
  - id: repo-fit-2
    severity: major
    section: "8"
    finding: >-
      §8 step 5b nominates a single command as the step's own acceptance test — "Re-run
      the sweep after the edits; a non-empty result is the failure" — and that sweep,
      `rg -n "_work/|stress_test_design" tests/`, provably misses the WG-2 fixture site
      that repo-fit-1 is about, for exactly the reason the design gives for writing it
      that way. Separately, the sweep covers only migration Event 1 (paths). Event 2 of
      the design's own migration note (the indicator-table column removal) has **no sweep
      at all**, and it has live references in `tests/` that step 5b's table does not list.
    rationale: >-
      I ran the design's exact pattern. It returns 10 hits over 4 files, matching §8's
      3+3+3+1 table precisely — so the table is an accurate record of what the sweep
      finds, and the sweep is what bounds the step. But `tests/test_interchange_contracts.py:830-831`
      spells the path as `join(_WG_DIR, "_work", "st_1.csv")` / `join(_WG_DIR, "_work",
      "cst_1.csv")` — segments, no slash — so the trailing slash the design explicitly
      defends ("the trailing slash matters: bare `_work` false-positives on `_workflow`")
      is the thing that hides it. That test is `@pytest.mark.skipif(not _fixture_present())`
      (line 826), i.e. a stale fixture path behind a skip guard: the precise failure class
      `AGENTS.md` records R9 losing 22 tests to, three of them silently. On Event 2:
      `tests/test_check_baseline_indicator.py:59-66,243-247,265-273` and
      `tests/test_check_baseline_scope.py:56` build fixtures on the seven-column header,
      and `test_float_key_columns_are_compared_as_written` exists solely because
      `temp_change` is a float key column — after the change, every remaining key column
      (`metric, location, st_id, rlz_id`) is non-numeric, so that test keeps passing while
      its stated subject no longer exists. That is a false green, not a failure, which is
      why only a widened sweep catches it.
    suggested_fix: >-
      Widen the sweep to cover both migration events and both path spellings, e.g.
      `rg -n "_work|stress_test_design|temp_change|precip_change" tests/ dev/scripts/ dev/reference/`,
      accept the `_workflow` false positives as cheap, and re-derive step 5b's table from
      the widened result so it names `test_wg2_integration`, `test_check_baseline_indicator.py`
      and `test_check_baseline_scope.py`. Note in step 5b that `_member_artifact`'s legacy
      `cst_1.csv` fallback (line 831) moves or retires with whatever replaces that test.
  - id: repo-fit-3
    severity: major
    section: "9"
    finding: >-
      Seven of the fourteen claims in §9's claim → falsifier table (V5–V11) are routed to
      the tier "unit test", and no test file is named for any of them. `shared/surface_axes.py`
      is the largest new surface in the design — `parse_surfaces`, `month_classes`,
      `axis_values`, `axis_caption`, `join_axes`, three exception types, `DEFAULT_SURFACE`,
      `AXIS_COLUMN` — and §8 step 4 lists only "`shared/surface_axes.py` and the parse-time
      call in `run_stress_test.smk`". No `tests/test_surface_axes.py` appears in step 4,
      in step 5b, or in §9's gate table, which otherwise names an owning test file for every
      changed surface.
    rationale: >-
      §9's gate table names `test_prepare_cst_parameters.py`, `test_export_wflow_results.py`,
      `test_interchange_contracts.py`, `test_stress_test_grid.py` (all four exist — I
      checked `tests/`) as "the narrow tier, and these own the changed surfaces". None of
      them owns `surface_axes`, and none of them would run V5–V11. The result is that the
      design's most novel logic — the caption algorithm's six cases, the two refusals
      (`HeterogeneousAxisError`, `HeldMonthInAxisError`), the rectilinearity postcondition —
      has no declared home, and the §9 evidence for it is a table of captions rendered ad
      hoc during the design run rather than a check anyone can re-run. Under the repo's
      own ladder ("only the tests covering the file you changed"), an unnamed test file is
      an ungated file.
    suggested_fix: >-
      Add `tests/test_surface_axes.py` to §8 step 4 and to §9's narrow-tier gate row, and
      map V5–V11 onto named test functions in it. The `np.linspace` member matrices and the
      six rendered captions already in §9's E10 block are the fixtures — promote them from
      run evidence into the test file.
  - id: repo-fit-4
    severity: minor
    section: "5.3"
    finding: >-
      The library signatures in D14 use bare names where `dev/reference/naming.md` §5
      requires a suffix: `read_lookup(path)` for a file path, and `lookup`, `indicators`
      for pandas DataFrames in `month_classes`, `axis_values`, `axis_caption` and
      `join_axes`. §5 is not advisory — "New code MUST use `_path` for a variable holding
      a file-path string" (§3), and "`_df` (pandas DataFrame)" (§5).
    rationale: >-
      `dev/reference/naming.md:55-62` and `:118-133`. This matters more than usual here
      because D14's block is written as normative API text that an implementer will
      transcribe verbatim, and because `surface_axes.py` becomes the reference
      implementation HM-7 cites — a contract surface, per §5's own framing. The `_csv`
      suffixes the design uses for Snakemake labels (`lookup_csv` in D20/D23) are correct
      and should stay; §5 reserves extension suffixes for exactly that position.
    suggested_fix: >-
      `read_lookup(lookup_path)`, `month_classes(lookup_df, variable)`,
      `axis_values(lookup_df, axis)`, `axis_caption(lookup_df, axis)`,
      `join_axes(indicators_df, lookup_df, surface)`.
  - id: repo-fit-5
    severity: minor
    section: "5.2"
    finding: >-
      D8 introduces a new top-level config section and D13 says it is read by
      `surface_axes.parse_surfaces(config)`, but neither says how the section itself is
      obtained from `config`, and neither mentions `get_config`. `AGENTS.md` § Conventions
      states that each Snakefile "parses one `--configfile` YAML via a shared
      `get_config(config, key, default, optional)` helper" and that "a new config key must
      mirror that contract (raise on missing required, return the default for optional)".
      `reporting:` is optional with a default (D10: absent or empty ⇒ `DEFAULT_SURFACE`),
      so the contract is satisfiable — it just is not stated.
    rationale: >-
      `blueearth_cst/shared/snake_utils.py:437-467`; the in-file precedent is
      `run_stress_test.smk:526-529`, which reads a nested optional key as
      `get_config(config.get("workflows", {}).get("build_model", {}) or {}, "wflow_outvars",
      DEFAULT_WFLOW_OUTVARS, optional=True)`. Saying so in D13 costs one clause and closes
      the question of whether `reporting: null` (a key present with a null value) yields the
      default or a TypeError — `get_config` returns `None` as-is for a present key, which is
      exactly the `or {}` case the precedent above guards against.
    suggested_fix: >-
      State in D13 that the section is read as
      `get_config(config, "reporting", {}, optional=True) or {}` and that `surfaces` is read
      from it the same way, so a present-but-null section resolves to `DEFAULT_SURFACE`
      rather than raising.
  - id: repo-fit-6
    severity: minor
    section: "10"
    finding: >-
      S10 pins "the same collapse must be applied to the projection overlay" and OQ-2
      defers the mechanism, which is correct and not re-opened here. But there is already
      an in-repo implementation of that collapse, and the design does not name it:
      `dev/scripts/preview_wf2_projection_plots.py` computes its own `temp_change` /
      `precip_change` from WF2 series and plots GCM dots in exactly the axis space HM-7
      pins. Q6's eventual design will find a second, divergent implementation of the
      quantity the new HM-7 text specifies.
    rationale: >-
      `dev/scripts/preview_wf2_projection_plots.py:299-302, 319-322, 364-367` each compute
      `precip_change = (precip - precip_ref)/precip_ref * 100` and `temp_change = temp -
      temp_ref` off an annual reference, and `:527, 535` place labels at
      `(row["precip_change"], row["temp_change"])`. This is not a counter-example to D14's
      "no in-repo consumer" argument — D14 is about WF3 *rules*, and a dev preview script
      is not one — and it does not change any decision. It is worth one line so the
      deferral is a deferral of a known site rather than of an unknown one.
    suggested_fix: >-
      Add `dev/scripts/preview_wf2_projection_plots.py` to OQ-2 as the existing overlay
      implementation Q6 must reconcile with §5.7's formula, and to §7's risk list as the
      place the second collapse currently lives.
```

---

# External review — round 1 (verbatim)

## Verdict
verdict: revise
doc_version: design-v2.md

## Findings
### ext1-1  [blocking]
- section: 5.5 The derived caption — algorithm
- finding: The caption algorithm ignores an explicitly declared month subset when all months vary. D16 permits `M` to be a proper subset of the varying set, but §5.5 selects case 1 whenever `H` is empty and emits `mean change over the year`. Thus a uniform experiment declared with `months: [1, 2, 3]` is labelled annual even though its axis—and eventually its projection overlay—is collapsed over JFM.
- rationale: The resulting figure makes a false statement about the plotted quantity and violates C1. The error is especially consequential for the projection overlay because its JFM collapse can differ materially from its annual collapse even when the stress-test member values do not.
- suggested_fix: Derive the leading phrase from `M` in every non-degenerate case (`mean change over <label(M)>`), independently of the global varying/held classification. Define how varying months outside `M` are described or omitted, and add a test for an all-month-varying design with explicit `M = JFM`.

### ext1-2  [blocking]
- section: 8. Migration plan
- finding: The design contradicts the settled ruling that the new axis-derivation library has no in-repo caller. D15, alternative 6.9, and R9 assert that `axis_values`, `axis_caption`, and `join_axes` execute on no repository path, while migration step 6 requires the repository notebook to call `surface_axes.read_lookup`, `read_indicators`, `join_axes`, and `axis_caption`.
- rationale: The implementation cannot satisfy both instructions. Following step 6 violates the owner-approved boundary and invalidates R9’s risk analysis; following the ruling leaves the notebook migration specified by R6 incomplete and potentially broken after removal of the old columns.
- suggested_fix: Align the notebook migration with the settled no-caller ruling, such as by making it a contract-based external-consumer example that does not import the library. Otherwise return the proposed exception to the owner gate before continuing.

### ext1-3  [major]
- section: 5.3 The consumer side — what derives an axis
- finding: The degenerate-axis contract is not implementable unambiguously. D27 admits degenerate axes whose months are held at several different offsets, but says to “return the constant for those months” and bypasses step 3, where the weighted-collapse formula is defined. In addition, `axis_values` returns only a `pd.Series` and `join_axes` only two data frames, although D19 requires the consumer to receive `degenerate = True`; D28 also requires normalization using `ST_NUM`, which none of the normative signatures accepts.
- rationale: Independent implementations can legitimately choose different scalar values for a multi-offset degenerate axis, and the specified Python caller has no defined channel for the metadata needed to render that value as an annotation rather than a plot dimension. Key-width inference is likewise left implicit, weakening the partition check intended to prevent silent misjoins.
- suggested_fix: Define the degenerate scalar explicitly by applying the same flat-vector/weighted-mean formula over `M`, noting that the result is constant across members rather than necessarily equal across months. Replace the Series-only API with an explicit result object carrying values, caption, `degenerate`, and key-width context, or add equivalent explicit parameters and return values.

### ext1-4  [major]
- section: 7. Consequences and risks
- finding: Consequence 2 conflates the bound on the reconstructed precipitation multiplier with a bound on hydrological indicator values. D25 can bound the forcing-parameter difference to one `float64` ulp, but it cannot establish that indicator outputs move by at most that amount or remain within the baseline comparator’s tolerance.
- rationale: Weather-generation transformations, thresholds, quantile mapping, and hydrological simulation can amplify or discontinuously respond to a tiny parameter change. V20 tests only reconstruction, while V4 uses shipped levels that reconstruct exactly, so no validation falsifies the stated output claim for a non-exact grid.
- suggested_fix: State the one-ulp guarantee only for the reconstructed multiplier. Limit indicator-equality claims to exactly reconstructing shipped configurations, or add an end-to-end non-round-grid experiment with an empirically justified output tolerance.

### ext1-5  [major]
- section: 9. Validation plan
- finding: V17 does not test the failure behaviour it claims to validate. Its falsifier is the R script proceeding with a missing, partial, duplicate, or unordered member slice, but its assigned check is only a WF3 run on the valid rapid configuration.
- rationale: The guard can be absent or incorrectly expressed while the proposed gate remains green. A malformed or mismatched lookup can then reach R vector recycling and produce silently wrong climate perturbations—the exact cross-language failure D29 is intended to prevent.
- suggested_fix: Add negative executions of `impose_climate_change.R` using lookup fixtures with a missing month, duplicate month, wrong member token, and unordered months, asserting a nonzero exit and the member-specific diagnostic.

### ext1-6  [minor]
- section: 9. Validation plan
- finding: V15 lists "`validate_wg2` green on a `12 × ST_NUM` lookup" as a falsifier even though that is the expected valid case.
- rationale: Read literally, the gate treats correct validator acceptance as failure, making its pass criterion internally inconsistent.
- suggested_fix: Change the first clause to "`validate_wg2` not green on a valid `12 × ST_NUM` lookup"; retain green results on the malformed variants as falsifiers.

---

# External review — round 2 (verbatim)

## Verdict
verdict: revise
doc_version: design-v3.md

## Findings
### ext2-1 [blocking]
- section: §5.3 The consumer side — what derives an axis (D28); §5.8 HM-7 replacement text
- finding: The report-time partition check is only one-directional. `join_axes` requires indicator ids absent from the lookup to equal the baseline token and requires a non-empty surface, but never requires every lookup member to appear in the indicator table. `validate_hm7` specifies bidirectional completeness, but it is test-time-only and therefore does not close this runtime gap.
- rationale: If one or more surface members are missing from a stale or partial indicator table, every remaining nonzero id still belongs to the lookup, the absent-id set is still exactly the baseline, and the surface is non-empty. `join_axes` therefore returns an incomplete response surface silently, producing missing grid cells or a biased surface rather than the mismatch error D28 was introduced to provide.
- suggested_fix: Make `join_axes` assert equality between the lookup member-id set and the nonbaseline indicator member-id set before joining. Mirror that requirement in HM-7’s report-time join semantics and add a V18 case where one valid lookup member is missing from the indicators.

### ext2-2 [blocking]
- section: §5.2 The surface declaration — config schema and tiers; §5.3 The consumer side — what derives an axis (D14/D33)
- finding: The schema permits `x` and `y` to declare the same `variable`, but the result representation cannot express that configuration. Each axis independently accepts `temp | precip`; no distinctness rule exists, while `SurfaceJoin.axes` is keyed by variable and derived columns are named through `AXIS_COLUMN[variable]`.
- rationale: A declaration such as JFM temperature on `x` and JJA temperature on `y` passes the specified schema, but one `AxisResult` overwrites the other in the dictionary and both target the same `temp_change` column. The implementation must either discard an axis or return an object that violates its declared API, so an admitted configuration cannot be implemented correctly.
- suggested_fix: Require at parse time that `{x.variable, y.variable} == {"temp", "precip"}`—allowing orientation reversal but refusing duplicate variables. State this in the surface schema and add a negative parser test.

### ext2-3 [major]
- section: §5.1 The lookup table — D25; §5.7 WG-2 replacement text
- finding: The normative “at most one `float64` ulp of the level” reconstruction bound is unqualified, although its evidence covers only multipliers in `[0.5, 1.6]` and the design specifies no matching admissible range. The bound is false over positive values otherwise permitted by the document: for the float32-shortest level `0.013596006`, the specified conversion writes `-98.6403994` and reconstructs `0.013596005999999883`, a difference of 68 float64 ulps.
- rationale: WG-2 makes this bound a pinned cross-language contract, while V16 and V20 use it as an acceptance threshold. A low but positive multiplier can therefore follow the specified formulas exactly and still fail the contract and migration gate; the accepted resolution of the prior precision finding is not valid over its declared domain.
- suggested_fix: Either impose and parse-time validate a multiplier domain for which the one-ulp bound is proved, or replace it with a domain-qualified numerical-error bound. Extend V20 across the full admitted domain, including values near its lower boundary.

---

# External review brief (the immutable contract, as dispatched)

# External review brief — WF3 stress-test lookup and derived response-surface axes, round 2

> Instantiated from `design-review-loop/references/external-review-brief.md`.
> **The review contract below (Role, Authority boundary, Lenses, Evidence burden,
> Output contract) is immutable for this run.** The *Task* paragraph and the
> *Settled framing* block are run state, refreshed from `status.md` at every
> dispatch.

## Role

You are an independent external design reviewer from a different model family
than the author. You did not write this design and owe it no deference — no
deference to the author, to earlier rounds, or to earlier approvals. Your value
is adversarial pressure: challenge framing, feasibility, and completeness. Do not
copyedit prose.

## Task

Review exactly one document:

- `C:\Users\taner\workspace\.worktrees\blueearth_cst\devmeta\dev\working\design-runs\wf3-stress-test-lookup\design-v3.md`

The design covers workflow 3 of a climate stress-testing toolbox (Snakemake, with
Python and R stages). WF3 perturbs a basin's climate across a temperature ×
precipitation grid of "members", runs a hydrological model on each, and reduces
the results to indicator tables that are plotted as a response surface. Today the
grid's parameters are written as two artifacts — one per-member file at monthly
grain, and one summary table whose two axis columns are an annual collapse of the
monthly values — and that annual collapse is also baked into the indicator tables
at reduction time. The design replaces both artifacts with a single long lookup
table at monthly grain, and moves the response-surface axis from a fixed
reduction-time collapse to a declared post-processing parameter. It also carries
replacement text for two interchange contracts, a migration plan, and a
validation plan.

**Settled framing — out of scope for your review.** These were ruled by the
project owner at the run's gates and are not open:

- Units: temperature change in °C; precipitation mean and variance change in
  **percent**, with the column names unsuffixed.
- The lookup table is the **source of truth**; indicator tables carry the member
  id and the value, with no baked axis. Axis values are derived, never stored.
- The lookup determines the **axis**, not the **scenario** — two members can carry
  identical parameter rows and still be different scenarios.
- No external consumer constrains this change; a downstream R package is
  parameterized and its owner updates it separately.
- The artifact is named `stress_test_lookup.csv` and lives in the experiment's
  `config/` directory; the previous per-member working directory disappears.
- The unperturbed baseline member is **not** a member of the response surface. It
  stays simulated and is reported as an annotated reference value.
- The grid's identity member is simulated like any other; an earlier proposal to
  alias it onto the baseline was withdrawn after measurement showed the two are
  not the same scenario.
- This work lands **before** the milestone that reworks how WF3 executes; that
  milestone's member-identity scheme will key on the monthly lookup rows.
- Only linear statistics may define an axis, and the same collapse must be
  applied to the projection overlay.
- The repository is a workflow engine only; upstream modelling-framework
  conventions are used verbatim and never re-engineered.
- **The lookup's schema is normatively defined in the weather-generator seam
  contract** (the Python→R seam the artifact crosses); the hydrological-model seam
  contract references it rather than restating it.
- **The new axis-derivation library deliberately has no in-repo caller.** That is
  an accepted, named risk; the compensating requirement is that the contract text
  an external re-implementer reads must be complete for what it owns.

Do not spend findings arguing these should have been decided differently. **Do**
raise a finding if a ruling creates a downstream inconsistency in the document, or
if the document's implementation of a ruling does not actually satisfy it.

Also read, **after forming your own view of the design**:

- `C:\Users\taner\workspace\.worktrees\blueearth_cst\devmeta\dev\working\design-runs\wf3-stress-test-lookup\ledger.md`
  — dispositions of every prior finding (32 rows: 26 from an internal panel, 6
  from external round 1)
- `C:\Users\taner\workspace\.worktrees\blueearth_cst\devmeta\dev\working\design-runs\wf3-stress-test-lookup\internal-review-index.md`
  — the internal panel's findings, grouped by concern

**Regression duty.** Verify that findings marked resolved are **actually**
resolved in this version, that no accepted fix introduced a new defect, and that
the rationales given for declining part of a suggested fix hold. Re-raise anything
that fails. Your own round-1 findings may be withdrawn only by you, here.

Three specifics worth your attention, stated as facts rather than as conclusions:
round 1 filed six findings and all six are dispositioned `accepted`; three of
those rows accept the finding while declining one branch or clause of the
suggested fix; and the previous version was found to contradict an owner ruling in
a place the author had reasoned about explicitly rather than overlooked.

## Authority boundary

Read-only. Read the document listed above; you may skim files it directly cites
if you need context, but do not read broadly through the repository and do not
modify anything.

## Review lenses (in priority order)

1. **Operational feasibility** — would this design work as specified? Ambiguous
   contracts, unimplementable steps, missing inputs, undefined behaviour.
2. **Failure modes missed** — realistic ways the designed system degrades that the
   design does not cover.
3. **Incentive and process design** — where the design includes loops, gates, or
   criteria: are they gameable, self-defeating, or consensus theater?
4. **Over-engineering** — components whose cost exceeds their value in this
   repo's context; simplifications that lose little.
5. **Gaps** — anything a design of this genre should cover and doesn't.

## Evidence burden

Every `blocking` or `major` finding must state an observable consequence — what
fails, degrades, or costs — not a preference. Cite the design section it targets.
A verdict of `approve` may not coexist with any `blocking` or `major` finding.

## Output contract (mandatory)

Return ONLY a markdown document with this structure — no preamble:

    ## Verdict
    verdict: approve | revise | reject
    doc_version: design-v3.md

    ## Findings
    ### ext2-<seq>  [blocking | major | minor]
    - section: <design heading the finding targets>
    - finding: <one-paragraph claim>
    - rationale: <why it matters — observable consequence>
    - suggested_fix: <concrete change, or "none">

Severity calibration: `blocking` = the design as specified would fail, produce
wrong results, or cannot be implemented; `major` = meaningful degradation, cost,
or risk with a clear fix; `minor` = worth noting, author's discretion. List
findings in severity order, blocking first. Aim for the findings that matter; do
not pad. If the design is sound, say so — an empty findings list with
`verdict: approve` is a valid review.

---

# Process observations

# Process observations — wf3-stress-test-lookup

Driver-appended. Process friction only, never design content. Feeds the post-run
retrospective; the skill stays unchanged for the whole run.

## O1 — the seed path's "reshaping, not a redraft" under-determines stage 1 (stage 1, 2026-08-15)

`stage-contracts.md` § *Seeding from an existing doc* gives two paths, keyed on
structural checks. Ours fail, so the rule reads:

> spawn the author scoped to *restructure to the genre contract, preserving all
> content verbatim*. That is a reshaping, not a redraft.

Taken literally that produces a `design-v1.md` that is genre-shaped and **still
missing everything the run exists to write** — `intake.md` declares six scope
gaps (axis-declaration schema, the unassigned consumer side, two unenforced
constraints, HM-7's replacement text, the caption spec, migration + tree shape)
that are *undesigned*, not merely unstructured. The internal panel would then
spend three lens dispatches re-reporting six gaps the driver had already
enumerated at stage 0.

The rule's *purpose* is clear and worth keeping: protect owner-ruled content from
being silently re-authored. Its wording generalizes from a case where the seed was
a complete design needing reshaping.

**Driver's reading, applied to the stage-1 brief:** preserve every ruled decision
verbatim, *and* write the declared scope gaps as new normative content. The
protection attaches to what has been ruled, not to the document's silences.

Candidate for the post-run retrospective: the seed path could split its structural
check from a **completeness** check — a seed can be genre-shaped and still not
cover the intake's declared scope, and those are different repairs.

## O2 — dispatch authorization is a session-level gate the loop does not model (stage 0→1, 2026-08-15)

This session runs under a standing instruction not to dispatch agents unless the
user asks. The loop assumes the driver may spawn freely once entry criteria are
met, so stage 1 blocked on something no stage contract names. Handled by executing
stage 0 driver-only (it needs no dispatch), recording the block in `status.md`,
and putting the dispatch plan with its floor/cap counts to the user as an explicit
choice.

Not a defect in the skill — but the run-start checklist could usefully ask whether
the driving session is *permitted* to dispatch, alongside the entry criteria it
already checks. A loop authorized in principle and blocked in practice at stage 1
wastes the intake if it is discovered later.

## O3 — the primary checkout cannot host author spawns here (stage 1, 2026-08-15)

`blueearth_cst` runs `worktree_policy: always`: a PreToolUse guard denies native
edit tools whenever the session's cwd is the primary checkout, and a spawned agent
inherits that cwd. The run directory is `dev/**`, which the repo's lane partition
assigns to `lane/devmeta`.

So the driver must **enter the lane worktree before dispatching**, not merely
before editing. Recorded because it is invisible until a spawn fails on its first
write, which under the skeleton-first rule is its first action — the failure would
look like a transport fault and earn the retry ladder, which would repeat it.

## O4 — a driver bookkeeping defect the AUTHOR caught, not the driver (stage 6, 2026-08-15)

The stage-4 outcome, the stage-5 convergence entry and the stage-6 dispatch entry
never reached `status.md`. The driver had written them with a Python
`str.replace()` against a multi-line anchor; the anchor did not match, and
`str.replace()` **returns the string unchanged rather than raising**, so the write
succeeded, the commit succeeded, and the log silently lost three stages.

Nothing in the loop caught it. It surfaced because the stage-6 author reported
"`status.md` has no stage-5 convergence entry" under *what the input set failed to
give me* — i.e. the spawn discipline's self-containment check did the work the
driver's own bookkeeping should have.

Three things follow, in increasing generality:

1. **Repaired**, and the manifest carries `status-log-gap-repaired` so a later
   reader does not read the reconstructed entries as contemporaneous.
2. **The resume rule assumed a completeness this defect breaks.**
   `run-artifacts.md` says a resuming driver "compares artifacts on disk against
   the stage log and re-runs any stage whose outputs are missing **or
   unrecorded**". Here the artifacts existed and the *log* was missing, so a
   resume would have re-run external round 1 — spending a **capped** round to
   regenerate a file already on disk. Write-then-mark protects against a crash
   between artifact and mark; it does not protect against a mark that silently
   no-ops.
3. **The general rule, which is this run's own recurring theme:** prefer an edit
   mechanism that fails loudly. The repair used a Python block that `sys.exit`s on
   any unmatched anchor, and the `Edit` tool errors on a non-matching
   `old_string` — either is safe. A bare `str.replace()` on generated prose is the
   same defect class as the run's own findings: a check that passes for a reason
   unrelated to what it claims to verify.

## O5 — an author disclosure worth keeping (stage 6, 2026-08-15)

The stage-6 author volunteered that it ran one read-only `git status --porcelain`
against the brief's "do not run git", and said it should have been a filesystem
check. Harmless in substance — it is how it verified v1/v2 were untouched.

Recorded because the disclosure is the valuable behaviour: an author that reports
a boundary it brushed is worth more than one that quietly stays inside. The brief
wording is the thing to fix — "do not run git" is aimed at *state changes*, and a
blanket ban pushes a spawn into either violating it or skipping a verification it
should do. Candidate for the retrospective: phrase author authority boundaries as
"no git operation that changes state (add/commit/checkout/stash)" rather than a
blanket prohibition.
