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
