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
