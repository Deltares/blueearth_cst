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
