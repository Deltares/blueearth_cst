# P3-2b — Interchange contracts design: consolidated review record

Durable audit trail of the `p32b-interchange-contracts` design-review-loop
run (2026-07-24). The run directory was pruned at landing; this record
preserves the verdicts, the aggregated internal index, the verbatim internal
lens reviews, both verbatim external rounds, and the final disposition
ledger. Accepted design: `interchange-contracts-design.md` (v3). Scoping
authority: `climate-interchange-intake.md` (landed pre-run).

## Run summary

| Item | Value |
| --- | --- |
| Run | p32b-interchange-contracts (full variant) |
| Author binding | cst-architect |
| Gates | G1 approved 2026-07-24; G2 approved 2026-07-24 |
| Versions | v1 (draft) -> v2 (post-panel) -> v3 (post-ext-r1; ACCEPTED) |
| External rounds | r1 revise (2 major) on v2; r2 APPROVE (zero findings) on v3 |
| Convergence | approve + 0 blocking/major on v3; ledger closure 18/18 |
| Dispatches | opus: 5, fable: 1 (revision r2 — ext1-2 faulted the v2 repo-1 fix), external GPT: 2 |
| Arbitration events | none (no rejected findings; converged inside the cap) |
| Verdict table | risk: revise 0/2/3 - architecture: revise 0/3/4 - repo-fit: revise 0/2/2 - ext r1: revise 0/2/0 - ext r2: approve 0/0/0 |

---

## Internal review index (driver aggregation)

# Internal review index — p32b-interchange-contracts (panel on design-v1.md)

Driver aggregation only: groups reference the original findings; every ID,
severity, and full text lives in the immutable per-lens files
(`internal-review-{risk,architecture,repo-fit}.md`). Nothing is deleted or
re-graded here.

## Verdicts

| Lens | Verdict | blocking | major | minor |
| --- | --- | --- | --- | --- |
| risk (`critical-thinker`) | revise | 0 | 2 (risk-1, risk-2) | 3 (risk-3..5) |
| architecture (`cst-architect`) | revise | 0 | 3 (arch-1..3) | 4 (arch-4..7) |
| repo-fit (`python-engineer`) | revise | 0 | 2 (repo-1, repo-2) | 2 (repo-3, repo-4) |

Totals: 0 blocking / 7 major / 9 minor. All three verdicts name
`doc_version: design-v1.md`; consistency rule satisfied (no approve with
majors).

## Groups (duplicates and near-duplicates; author must still ledger every ID)

- **A — HM-6 accounting + path (majors risk-1, arch-1; major arch-3).**
  risk-1 and arch-1 are the same root defect from two angles: HM-6 appears in
  both the green and skip buckets, so the G1-anchored 10-green/3-skip tally
  is internally contradictory (arch-1 verifies the honest per-instance count
  is 11 green + 3 skip). arch-3 compounds it: the HM-6 wf1 path as stated
  (`outstate/outstates.nc`) is unopenable — the real path is
  `run_default/outstate/outstates.nc` (derived from `dir_output` +
  `[state].path_output`). Fix set: id-split HM-6 (per risk-1's suggestion,
  mirroring HM-2/WG-6), correct the path + record its derivation, and restate
  ONE counting axis consistently across §5.5/§7/§9/§11. Note risk-1 also
  flags HM-6-wf1 as a redundant existence check (HM-4 already pins
  `[state].path_output`) — the author should decide pinned-vs-dropped
  explicitly, not silently.
- **B — units pinned wrong or under-specified vs fixture (major arch-2;
  minors repo-3, risk-4).** arch-2: HM-2 pins pet/temp units that are ABSENT
  (None) on the fixture artifact and that no consumer reads (wflow is
  name-keyed via `[input.forcing]`) — a validator per the row is red on the
  design's own green list and violates the §5.1 tiering. repo-3: HM-2 precip
  unit is `mm` (not `mm d**-1`), and WG-1's attr key is `unit` (singular).
  risk-4: within HM-2 the attr KEY is heterogeneous (`units` for precip,
  `unit` for pet/temp) and the row must say which key is pinned. Fix set: one
  coherent units contract per artifact, grounded on observed attrs, with
  unpopulated/unconsumed units moved to deliberately-unpinned or
  asserted-if-present.
- **C — fixture-absent suite red (major repo-1, standalone).** All 13
  validator tests need the repo's `skipif(fixture absent)` convention, not
  only the 3 temp cases; restate coverage as green-when-fixture-present.
- **D — validator error idiom (major repo-2, standalone).** `AssertionError`
  matches no house idiom and is stripped under `-O`, undercutting C5
  liftability; choose `list[str]` report (drift-guard analog) or `ValueError`
  (validate_experiment_name analog).
- **E — temp-validator correctness never executed (major risk-2,
  standalone).** The acceptance gate only checks that the 3 temp validators
  skip; their assertion logic runs against nothing. Preferred fix (a):
  synthetic in-memory pass+fail unit tests per temp validator (no fixture
  edit, no behavior change).
- **F — minors, walkthrough/doc precision.** arch-4 (HM-4 missing
  `time.timestepsecs` rewrite field; wf3 calendar rewritten to "standard" vs
  wf1 proleptic_gregorian — document), arch-5 (WG-5 uri is absolute,
  machine-scoped — state unpinned), arch-6 (~39 not ~30 unpinned staticmaps
  vars; `[input.forcing]` mapping direction inverted), arch-7
  (considered-and-excluded note for `wf1_raw/extract_historical.nc` —
  completeness audit otherwise CONFIRMS the inventory), risk-3 (WG-5
  "indirect coverage" of WG-4/WG-6 oversold — bookkeeping only), risk-5
  (sim_dates.csv / resampled_dates.csv exclusion note), repo-4 (placement
  audited clean — no action required).


---

## Internal lens review — risk (verbatim)

```yaml
verdict: revise
doc_version: design-v1.md
findings:
  - id: risk-1
    severity: major
    section: "5.5 Validator design / 11 Revision log / 9 Validation plan (the 10-green/3-skip tally)"
    finding: >
      The headline "13 validators = 10 continuously green + 3 skip-until-captured"
      does not reconcile with the design's own §5.5 green list. §5.5 enumerates the
      continuously-verified set and explicitly includes "HM-6 wf1 outstate/outstates.nc"
      as green, while the skip list includes "HM-6 wf3 outstates_rlz_*.nc". The single
      HM-6 artifact-id therefore appears in BOTH buckets. If HM-6 counts as green the
      real tally is 11-green/2-skip; if it counts as skip, §5.5 must not list HM-6-wf1
      under "green now". The design claims the accounting credit of both at once. The
      design already solved the identical wf1/wf3 split cleanly for forcing by giving
      the wf3 twin its own id (HM-2 wf1 vs WG-6 wf3); it simply failed to apply that
      same id-split to HM-6.
    rationale: >
      The 10-green/3-skip reading is the G1-accepted interpretation of success
      criterion 2 and the acceptance number the milestone gate (§9) and the roadmap
      close-out note (commit 5) will assert. As written, an implementer cannot produce
      a validator index whose green/skip counts sum consistently, and a reviewer
      checking "10 green" against the delivered suite will find an off-by-one whose
      resolution the design left ambiguous — the milestone's own pass/fail number is
      undefined until this is decided. This is a failure to deliver the accepted
      anchor cleanly, not a relitigation of it.
    suggested_fix: >
      Split HM-6 into two ids exactly as HM-2/WG-6 were split — e.g. HM-6a
      (wf1 outstate/outstates.nc, persisted) and HM-6b (wf3 outstates_rlz_*.nc,
      temp()) — then restate the tally against the 14-id inventory. Note this also
      resolves risk-2's redundancy point: HM-6a's only pinned surface is name+location,
      which HM-4's TOML-rewrite fields ([state].path_output) already pin, so HM-6a is
      an existence check that pads the green count rather than verifying an
      independent contract.
  - id: risk-2
    severity: major
    section: "5.5 Validator design (temp() problem) / 9 Validation plan (acceptance criteria)"
    finding: >
      §5.5 asserts the three temp()-content validators (WG-4, WG-6, HM-6-wf3)
      "exist and are correct," but the §9 acceptance criterion for them is only that
      they "skip (not error) on the default fixture." Their assertion logic is never
      executed against any artifact — real or synthetic — anywhere in the milestone,
      because every temp NC is absent and the design explicitly does not run the
      --notemp capture. A validator whose body contains a wrong assertion (transposed
      dim, wrong var name, mis-typed unit) would PASS acceptance (it skips cleanly) and
      the defect would surface only at some future --notemp capture run — i.e. exactly
      when someone finally trusts it during a real swap.
    rationale: >
      This is a concrete gap between the acceptance check and success criterion 2's
      "validator per contract." "Correct" is claimed for code with zero execution
      evidence; the acceptance gate cannot detect an incorrect skipped validator, so
      the observable consequence is a suite that reports "3 correct validators, gated"
      when it has actually verified nothing about those three functions' logic.
      Risk R3 (temp-capture rot) names the drift risk but not this correctness-never-
      established risk, and R3's mitigation ("documented procedure") does not test the
      logic either.
    suggested_fix: >
      Either (a) add synthetic-Dataset unit tests that build a tiny in-memory
      xarray/geopandas object and exercise each temp validator's pass AND fail paths
      (no fixture edit, no behavior change, keeps the real-artifact case skipping), so
      the assertion logic is proven while the on-disk check stays gated; or (b)
      downgrade the §5.5 wording from "exist and are correct" to "written; correctness
      not established until --notemp capture," and reflect that honestly in the
      coverage statement. Option (a) is preferred — it costs one small test function
      per temp validator and closes the criterion-2 gap the strict reading demands.
  - id: risk-3
    severity: minor
    section: "5.5 (partial mitigations) / 5.6 (WG-seam bounded substitution)"
    finding: >
      §5.5 offers "WG-4/WG-6 producer-side surface is still checkable indirectly via
      WG-5" as a partial mitigation. WG-5 (verified against the fixture catalog) pins
      the entry-key grid and driver options (uri, driver.name, preprocess, crs,
      category, data_type). It pins nothing about the NC's dims, variable names, units,
      or grid — the content WG-4/WG-6 are supposed to guarantee. So WG-5 constrains
      that a catalog ENTRY exists per realization×cst, not that the NC it points at
      satisfies WG-4's contract.
    rationale: >
      The mitigation as phrased oversells indirect coverage: a reader may conclude the
      WG-4 content contract is partially checked when in fact only the catalog
      bookkeeping is. In a real generator swap the acceptance question is "does the
      candidate NC load and carry precip/temp on an EPSG:4326 grid" — precisely what
      WG-5 cannot answer. Honesty about coverage (C6) is the design's own stated
      priority, so the overstatement is worth correcting even though it is not
      load-bearing to the approach.
    suggested_fix: >
      Reword to "WG-5 checks the catalog bookkeeping (entry-key grid + driver
      options), not WG-4/WG-6 NC content; the NC-content contract is
      skip-until-captured with no indirect proxy." This is a one-line honesty
      correction, consistent with §6.3's own rejection of re-derivation proxies.
  - id: risk-4
    severity: minor
    section: "5.3 HM-2 (data_vars units) / 5.2 units note"
    finding: >
      HM-2 pins the model-grid forcing as "precip (mm d**-1), pet (mm), temp
      (degree C.)". Fixture inspection confirms the VALUES but shows precip carries
      its unit under the attr key `units` (`mm d**-1`) while pet and temp carry theirs
      under a DIFFERENT attr key `unit` (`mm`, `degree C.`) — precip has no `unit`
      key with the bare "mm", pet/temp have no `units` key at all. The contract states
      the unit strings but not which attr key holds each, so "reproduce HM-2" is
      under-specified at the attribute level and a naive validator asserting
      ds['temp'].attrs['units'] would raise KeyError against the real fixture.
    rationale: >
      This is discoverable at implementation time and the design's job is to pin the
      contract, not write the validator, so it is minor. But because the design
      elsewhere leans hard on the K-vs-degreeC cross-seam unit story as a named
      contract fact (§5.2 units note, §5.3 units divergence), the attr-key
      inconsistency inside the SAME artifact is a lurking trap the contract doc should
      record so the validator author and any future swapper reproduce the exact
      (heterogeneous) attribute layout the downstream reliance actually depends on
      (or explicitly declare the attr-key immaterial and pin only the value).
    suggested_fix: >
      In the HM-2 row, note precip uses attr `units` while pet/temp use attr `unit`
      (a known upstream hydromt heterogeneity), and state whether the validator keys
      on the attr name or reads either — so the pinned surface is unambiguous.
  - id: risk-5
    severity: minor
    section: "5.2 (weather-generator seam inventory completeness) / 9 (complete-inventory acceptance)"
    finding: >
      The fixture carries experiments/experiment/{sim_dates.csv, resampled_dates.csv}
      that map to no WG/HM id. Verified: neither name appears as a produced or consumed
      path in any Snakefile, Python module, or R script (they are internal diagnostic
      side-products of the weathergenr run, not DAG-tracked handoffs), so their
      omission from the interchange inventory is defensible. However success criterion
      1 is "every interchange artifact ... has a doc row," and the design nowhere
      states that these two persisted, non-temp CSVs were examined and deliberately
      excluded as non-interchange.
    rationale: >
      An auditor of "complete inventory" will see two persisted CSVs in the experiment
      tree with no contract row and cannot tell from the design whether they were
      missed or judged out-of-scope. A one-line exclusion note converts a silent gap
      into a documented, defensible boundary — cheap insurance for the completeness
      acceptance criterion, and consistent with §5.1's stated goal that a reviewer can
      see each omission is intentional, not an oversight.
    suggested_fix: >
      Add a short "excluded non-interchange artifacts" note to the weather-generator
      seam doc (or §5.2) stating that sim_dates.csv/resampled_dates.csv are
      weathergenr-internal diagnostics with no downstream consumer, hence not seam
      contracts.

```

---

## Internal lens review — architecture (verbatim)

```yaml
verdict: revise
doc_version: design-v1.md
findings:
  - id: arch-1
    severity: major
    section: "§5.5 (Validator design) / §7 / §9 / §11 — the 10-green vs 11-green count"
    finding: >
      The "continuously verified (persisted survivors)" bullet in §5.5 enumerates
      ELEVEN artifact instances: WG-1, WG-2, WG-3, WG-5, HM-1, HM-2 (wf1
      inmaps_historical), HM-3, HM-4, HM-5, HM-6 (wf1 outstate/outstates.nc), and
      HM-7. But every other place the count appears — §7 ("three of thirteen
      validators skip", implying ten green), §9 ("all ten persisted-artifact
      validators green"), §11 ("10 continuously green, 3 skip-until-captured"), and
      the G1-fixed 10-green/3-skip acceptance anchor — says TEN. The discrepancy is
      HM-6: the design counts its wf1 instance as green AND its wf3 instance as
      skip, so a per-instance tally is 11 green + 3 skip while the anchor is stated
      per-id as 10 + 3. I verified the fixture (examples/test_local): all eleven of
      those artifacts persist and are openable (HM-6 wf1 outstate/outstates.nc does
      exist on disk), so the "11" enumeration is the factually complete one and the
      "10" figure is wrong, not the other way around.
    rationale: >
      This is an internal contradiction against a fixed acceptance anchor, which the
      task puts explicitly in scope. Observable consequence: the milestone gate in
      §9 ("ten persisted validators pass, three temp validators skip") is
      unverifiable as written — an implementer who builds validators for every §5.5
      green-bullet artifact produces ELEVEN green cases and the gate that checks for
      exactly ten cannot be satisfied without either dropping a real, checkable
      artifact or admitting eleven. The reviewer/user checking the milestone cannot
      tell whether "10" is a miscount or whether one persisted artifact was silently
      dropped from coverage.
    suggested_fix: >
      Reconcile the counting axis in one place. Either (a) state the tally
      per-artifact-id and note HM-6 spans two instances (wf1 green, wf3 skip), giving
      "11 persisted instances green across 10 pinned ids, 3 temp instances skip", or
      (b) keep "10/3" but define it as per-id and list HM-6's split status explicitly.
      Whichever axis is chosen, make §5.5, §7, §9, §11, and the acceptance criteria
      all use the same axis and the same number.
  - id: arch-2
    severity: major
    section: "§5.3 (HM-2 contract row) / §5.5 (green-on-fixture claim) / §5.1 R1"
    finding: >
      The HM-2 row (§5.3) puts variable UNITS in the pinned contract column:
      "precip (mm d**-1), pet (mm), temp (degree C.), all float32". I opened the
      fixture's wf1 forcing (climate_historical/wflow_data/inmaps_historical.nc):
      only precip carries units="mm d**-1"; pet.attrs['units'] is None and
      temp.attrs['units'] is None — the attribute is simply not populated on the
      artifact hydromt writes. A validator that asserts pet units == "mm" and temp
      units == "degree C." on this artifact fails RED on the era5 fixture, the exact
      artifact §5.5 lists as "continuously verified (green now)".
    rationale: >
      Two observable consequences, both forcing a revise. (1) Criterion C1 /
      success-criterion 2 (green-on-fixture) is not met for HM-2 as specified:
      validate_hm2 built to the §5.3 row is red on era5, not green. (2) It violates
      the design's own §5.1 tier rule and R1 mitigation: wflow maps forcing by
      variable NAME via the TOML [input.forcing] block (verified: base wflow_sbm.toml
      lines 127-130 map atmosphere_*/land_surface_* CSDMS names to "precip"/"pet"/
      "temp") — it never reads the netCDF units attribute of pet/temp. So pinning
      those units is pinning a property no consumer relies on, precisely the R1
      over-constraint the design says its "pinned vs unpinned" column exists to
      prevent. The pin is both wrong-on-fixture and self-inconsistent with the
      governing tiering principle.
    suggested_fix: >
      Move pet/temp units to the "deliberately unpinned" column for HM-2 (consumer
      is name-keyed, not units-keyed), or pin them only as "asserted-if-present".
      Keep precip units="mm d**-1" pinned (it IS populated). If a units contract is
      wanted at all, pin it where a consumer actually reads it, not on the inmaps
      artifact. Re-derive the WG-1 temp=K vs HM-2 temp=degC "documented divergence"
      note accordingly — on the fixture, HM-2 temp has no units attribute at all, so
      the stated "degree C." fact is not fixture-grounded (C4 violation for that cell).
  - id: arch-3
    severity: major
    section: "§5.5 (green bullet) / §8 commit 2 / §9 validator index — HM-6 wf1 path"
    finding: >
      §5.5 names the wf1 warm state as "outstate/outstates.nc" and lists it green.
      The actual on-disk path is run_default/outstate/outstates.nc: the base
      wflow_sbm.toml has dir_output = "run_default" (line 1) and
      [state].path_output = "outstate/outstates.nc" (line 32), so wflow writes it
      under run_default/. The fixture confirms the file lives at
      hydrology_model/run_default/outstate/outstates.nc, not
      hydrology_model/outstate/outstates.nc.
    rationale: >
      Observable consequence: a validator or validator-index entry pointed at the
      §5.5-stated path cannot open the file (FileNotFoundError) → the HM-6 wf1 case
      is not green → it perturbs the count in arch-1 (an implementer taking the path
      literally gets a red/error case where the design claims green). This is the
      concrete openability failure the task asked to be checked ("can a validator
      even open each named artifact given what persists on the fixture").
    suggested_fix: >
      Correct the HM-6 wf1 path to run_default/outstate/outstates.nc in §5.3 (row
      HM-6), §5.5, and the commit-2 warm-state finding. It is derived from
      dir_output + [state].path_output, so state that derivation so a swapper that
      changes dir_output knows the path moves with it.
  - id: arch-4
    severity: minor
    section: "§5.3 (HM-4) / §5.6 (hydrological-model bounded-substitution walkthrough)"
    finding: >
      HM-4 lists the TOML rewrite fields our downscale step sets as
      "[time].{calendar,starttime,endtime}, dir_output, [state].{path_input,
      path_output}, [input].{path_static,path_forcing}, [output.csv].path". The
      actual rewrite in downscale_climate_forcing.py (setup_config, lines 55-84)
      ALSO sets time.timestepsecs (86400). The per-cst fixture TOML confirms
      timestepsecs=86400 is written (wflow_sbm_rlz_1_cst_1.toml line 8). Also note
      the calendar is rewritten to "standard" (line 62), while HM-2 pins the inmaps
      forcing time axis as proleptic_gregorian and the base TOML is
      proleptic_gregorian — the wf3 forcing/TOML calendar handling diverges from the
      wf1 pin, undocumented.
    rationale: >
      A swapper reading §5.6's "run-config rewrite fields it must replace" gets an
      incomplete field list — the walkthrough is meant to be the complete file-touch
      map, and a missing rewrite field (timestepsecs) plus an unstated calendar
      value change means a replacement model's config-rewrite could omit a field the
      current pipeline sets. Low blast radius (timestepsecs is a fixed constant), so
      minor, but it undercuts §5.6's "concrete and reviewable" claim.
    suggested_fix: >
      Add time.timestepsecs to the HM-4 rewrite-field list and the §5.6 walkthrough.
      Add a one-line note that wf3 rewrites calendar to "standard" (distinct from the
      wf1 base proleptic_gregorian) so the cross-artifact calendar difference is a
      documented contract fact, not an apparent inconsistency with HM-2.
  - id: arch-5
    severity: minor
    section: "§5.2 (WG-5) / §5.6 (weather-generator walkthrough)"
    finding: >
      WG-5's catalog entries carry absolute machine-specific uri values (verified:
      the fixture data_catalog_climate_experiment.yml uri is
      C:\Users\taner\workspace\blueearth_cst\examples\test_local\...\rlz_1_cst_1.nc).
      The design correctly scopes the WG-5 pinned surface to the entry-key grid +
      driver contract (not uri resolution), so the validator design is fine — but
      the absolute-uri fact is unstated, and any future lift that resolves uri to
      open the referenced NC (e.g. the --notemp capture path, or an in-pipeline
      guard per OQ-5) would be non-portable.
    rationale: >
      Minor: does not break this milestone's validators (they check keys+driver, not
      uri targets). Worth recording because §5.5's "WG-5 pins the driver contract the
      temp NCs must satisfy to load" and the --notemp un-skip path implicitly assume
      the catalog can be followed to the NC — which only works on the machine that
      produced the fixture. A swapper emitting its own catalog (WG-5, per §5.6) should
      know the uri is absolute in the current emitter.
    suggested_fix: >
      Note in WG-5 that uri is an absolute path emitted by
      prepare_climate_data_catalog.py, is deliberately unpinned (portability is not a
      current contract), and that any uri-resolving validator/guard is
      machine-scoped.
  - id: arch-6
    severity: minor
    section: "§5.3 (HM-1 unpinned count) / §5.3 (HM-4 [input.forcing] direction)"
    finding: >
      Two small grounding imprecisions. (1) HM-1 says "~30 wflow vars ... unpinned"
      beyond the pinned set; the fixture staticmaps.nc has 44 data_vars, so
      44 - 5 pinned = ~39 unpinned, not ~30. (2) HM-4 describes [input.forcing] as
      "maps precip/pet/temp -> forcing var names"; the actual block maps CSDMS
      Standard Names (LHS keys, e.g. atmosphere_water__precipitation_volume_flux) to
      the forcing var names precip/pet/temp (RHS values). The design's phrasing
      inverts which side the var names sit on.
    rationale: >
      Both minor and non-blocking, but C4 (grounded-in-the-tree) asks every fact to
      trace to an observed artifact; a swapper reading HM-4 could misread which side
      of [input.forcing] their forcing variable names must match. Naming the mapping
      direction wrong is a real trip hazard for the model-seam substitution the doc
      exists to enable.
    suggested_fix: >
      Update HM-1 to "~39 unpinned wflow vars (44 total minus the pinned set)". Fix
      HM-4 to state that [input.forcing] keys are wflow CSDMS Standard Names and the
      VALUES are the forcing netCDF variable names precip/pet/temp — the tie to HM-2
      is on the RHS values.
  - id: arch-7
    severity: minor
    section: "§2 Non-goals / §5.2 inventory completeness (audit note, not a defect)"
    finding: >
      Inventory-completeness audit (walked both rule graphs): the WG-1..6 / HM-1..7
      enumeration DOES cover every interchange handoff at the two seams. Non-seam
      intermediates are correctly excluded — wf1 build configs
      (wflow_build_model_run.yml rule 1.02, wflow_build_forcing_historical.yml rule
      1.07), the reservoirs_lakes_glaciers.txt sequencing sentinel (rule 1.04), the
      .project_consistency_ok / .guard_ok guard sentinels, and the log/benchmark
      gather outputs are pipeline-internal, not substitution-seam artifacts. One
      artifact deserves an explicit "considered and excluded" line: wf1_raw/
      extract_historical.nc (rule 1.10 extract_climate_grid_wf1) shares WG-1's
      extraction schema but feeds the wf1 model-parity PLOTS, not either substitution
      seam — it is correctly out of both seam inventories.
    rationale: >
      No defect; recorded so the "complete inventory" verdict is auditable rather
      than asserted. The only actionable nit: the design nowhere states that
      wf1_raw/extract_historical.nc was considered-and-excluded, so a reviewer cannot
      distinguish "deliberately out" from "overlooked" — a one-line note closes that.
    suggested_fix: >
      Add a one-line "considered and excluded (feeds wf1 plots, not a seam):
      wf1_raw/extract_historical.nc" note to the §5.2 scope/method preamble.

```

---

## Internal lens review — repo-fit (verbatim)

```yaml
verdict: revise
doc_version: design-v1.md
findings:
  - id: repo-1
    severity: major
    section: "5.5 Validator design / 8 commit-3 / 9 Validation plan"
    finding: >
      The design guards `pytest.skip` ONLY for the three temp()-deleted content
      contracts (WG-4, WG-6, HM-6-wf3), and asserts the ten persisted-artifact
      validators are "green now under pixi run pytest" (§5.5), with §8 commit-3
      and §9 both stating "all ten persisted validators pass". But
      `examples/test_local` is UNTRACKED (verified: `git ls-files
      examples/test_local` -> 0 files). On any environment lacking the fixture —
      a fresh clone or CI — the ten persisted validators call `xr.open_dataset`
      / `open` on missing paths and ERROR, turning the suite red. The design
      never states a fixture-presence condition and prescribes no
      fixture-absent skip for the persisted validators.
    rationale: >
      Observable consequence: on a machine without `examples/test_local`, the
      per-commit gate "full suite green" (G5 / C2) FAILS at commits 3-5, which
      directly breaks the suite-green anchor the design must deliver. The repo
      already established the correct convention and the design silently
      departs from it: `tests/test_extract_climate_wf1.py` and
      `tests/test_check_project_consistency.py` are the only two tests that read
      `examples/test_local`, and the former guards every fixture-dependent test
      with `@pytest.mark.skipif(not os.path.exists(...), reason="untracked
      examples/test_local fixture tree not present")`. The design's
      "10 continuously green" is therefore local-only green; as written it does
      not faithfully deliver the 10-green criterion because it omits the
      fixture-presence precondition the repo convention exists to handle.
    suggested_fix: >
      Apply a fixture-absent `skipif` guard to ALL 13 validator tests (not only
      the 3 temp cases), mirroring `test_extract_climate_wf1.py`'s
      `skipif(not os.path.exists(<fixture path>))`. Restate the coverage claim
      as "10 continuously green WHEN the fixture is present, else skip; 3
      skip-until-captured" and adjust §5.5/§8/§9 accordingly so the suite is
      green (via skips) on a fixtureless checkout.
  - id: repo-2
    severity: major
    section: "5.5 Validator design / 6.2 Validator placement / C5"
    finding: >
      The design specifies validators as `validate_wgN_...(ds) -> None` that
      `raise AssertionError` with a specific message on violation (§5.5), and
      leans on C5 "liftable into a future in-pipeline guard without a move".
      This error-reporting idiom matches NEITHER house validator analog. The
      closest analog, `compare_project_consistency`
      (`blueearth_cst/experiment/check_project_consistency.py`), is a documented
      PURE function returning `list[str]` divergence messages (empty => pass);
      the other, `validate_experiment_name`
      (`blueearth_cst/shared/snake_utils.py`), returns the validated value or
      raises `ValueError`. Neither raises bare `AssertionError`.
    rationale: >
      Observable consequence for the load-bearing liftability claim (C5):
      `assert`/`AssertionError` is stripped under `python -O` / `PYTHONOPTIMIZE`,
      so a validator that signals failure only via `AssertionError` becomes a
      silent no-op if a future in-pipeline guard ever runs optimized — the exact
      lift the design is architected around would fail silently rather than
      block a bad artifact. It also breaks idiom consistency: this repo signals
      input-validation failure with `ValueError` (domain exception) or a
      returned report list, not `AssertionError`. A validator raising
      `AssertionError` reads as a test-only construct, undercutting the
      "written to be a real guard" rationale.
    suggested_fix: >
      Choose one house idiom and state it in §5.5: either a pure
      `-> list[str]` report (mirrors `compare_project_consistency`, the drift
      guard) or `-> None` raising `ValueError` on violation (mirrors
      `validate_experiment_name`). Avoid bare `assert`/`AssertionError` for the
      guardable path; if the test file wants `assert`, let the test wrap the
      validator's return/exception rather than the validator relying on `assert`.
  - id: repo-3
    severity: minor
    section: "5.2 WG-1 units / 5.3 HM-2 pinned surface / C4"
    finding: >
      Two grounded-fact transcription slips against the on-disk fixture, both
      touching the C4 "grounded in the tree" claim. (a) §5.3 HM-2 states
      `precip (mm d**-1)`, but `inmaps_historical.nc` on disk carries precip
      unit `mm` (same as `pet`); the design copied WG-1's `mm d**-1` onto the
      model-grid forcing. (b) On WG-1 `extract_historical.nc` the per-variable
      unit is stored under the attribute key `unit` (singular), not `units`;
      the design's prose says "units". A validator written literally to the
      doc's stated strings/keys would fail an equality assertion (a) or KeyError
      (b) against the real fixture.
    rationale: >
      C4 asserts every contract fact traces to an observed artifact; these two
      details do not, and a faithful validator author following the doc verbatim
      would write checks that fail on the very fixture the design says they are
      green against. Low blast radius (both are validator-writing details a
      careful implementer catches by opening the file), hence minor — but worth
      pinning so the contract doc records the actual on-disk strings.
    suggested_fix: >
      Correct HM-2 precip unit to `mm` and note the attribute key is `unit`
      (singular) in WG-1's row; have validators read the observed attr key/value
      rather than the design's paraphrase. (Verified green facts that need NO
      change: WG-1 temp*/temp_min/temp_max in K, kin/kout `J m**-2`, press_msl
      `Pa`, crs=4326/category=meteo, proleptic_gregorian; HM-2 temp
      `degree C.`, lat float64, on-disk calendar proleptic_gregorian; Qstats
      `statistic,tavg,prcp,Q_130000086`; basin `tavg,prcp`; output_rlz
      `time,Q_130000086`; cst_1 header + cold_start__flag=true;
      `naming.py:10 -> "elevtn": "land_elevation"`.)
  - id: repo-4
    severity: minor
    section: "5.4 Contract-doc placement / OQ-2"
    finding: >
      The design places the two seam docs in a NEW `dev/contracts/` subdir
      (verified absent today) and cites `dev/workflows/climate_experiment.md` as
      the one-file-per-workflow precedent. The actual precedent dir is
      `dev/workflows/` (contains `model_creation.md`, `climate_projections.md`,
      `climate_experiment.md`). Introducing `dev/contracts/` is defensible (the
      docs are cross-workflow, per OQ-2) and stays on the correct `dev/` side of
      the AGENTS.md dev-vs-docs boundary (audience = future swappers / R6, not
      end users). This is a taste call the design already flags as OQ-2, not a
      defect — recorded only to confirm repo-fit is clean either way.
    rationale: >
      The dev-vs-docs boundary is correctly honored (dev/, not docs/), the
      kebab-case filenames match naming.md, and neither placement breaks any
      tooling (no code globs `dev/`). No observable consequence; noted so the
      panel sees the placement was audited and passes.

```

---

## External review round 1 (verbatim; doc_version design-v2.md)

## Verdict
verdict: revise
doc_version: design-v2.md

## Findings
### ext1-1 [major]
- section: §5.3 HM-4–HM-7; §5.5 Validator design
- finding: The two stated cross-artifact invariants have no specified validator interface or test wiring. The proposed `validate_<id>(obj) -> list[str]` functions and one-per-artifact tests can validate each TOML, CSV, or catalog independently, but cannot establish that `[output.csv].column` identities match both `output_rlz_*.csv` and `Qstats.csv`, nor that catalog entries exactly match the intended realization×CST grid.
- rationale: A swapped model can emit syntactically valid TOML and CSV artifacts with a renamed, omitted, or mismatched gauge column; all individual validators can pass while rule 3.11 fails or silently produces an incorrect response surface. This leaves the design’s load-bearing “single degree of freedom” uncheckable, contrary to C1.
- suggested_fix: Specify dedicated relational validators, with explicit multi-path/object inputs, for the HM-4→HM-5→HM-7 chain and WG-5’s expected grid. Add pass/fail tests that deliberately break one member of each correlated set, and include these validators in the coverage/counting model.

### ext1-2 [major]
- section: §5.5 Test driver + fixture-absent skip; §8 Migration + commit plan; §9 Validation plan
- finding: All ten persisted-artifact validator tests silently skip whenever the untracked `examples/test_local` tree is absent. Thus a normal clean checkout or CI run can report the suite green without executing any of the milestone’s primary real-artifact validators; only the three temp-artifact validators receive fixture-independent synthetic coverage.
- rationale: A regression in validator logic or expected persisted-artifact contract can land undetected in the environment most likely to enforce per-commit checks. “Green” becomes indistinguishable from “nothing was checked,” weakening the claimed machine-checked and per-commit-runnable deliverable.
- suggested_fix: Provide fixture-independent pass/fail coverage for all persisted validators and retain the real-fixture checks as a separately reported integration target; alternatively add a small committed contract fixture. If the real-fixture check must remain optional, make its absence a clearly reported unmet integration gate rather than an indistinguishable all-skip success.

---

## External review round 2 (verbatim; doc_version design-v3.md)

## Verdict
verdict: approve
doc_version: design-v3.md

## Findings

---

## Final disposition ledger (verbatim)

# Disposition ledger — p32b-interchange-contracts (author r1, v1 → v2; author r2, v2 → v3)

One row per finding ID (18 total: risk-1..5, arch-1..7, repo-1..4 from the
internal panel; ext1-1..2 from external round 1). Round column names the
source round; doc version names the revision that dispositioned it.

| ID | Round | Severity | Disposition | Resolution or rationale | Doc version |
| --- | --- | --- | --- | --- | --- |
| arch-1 | internal-panel | major | accepted | HM-6 split into HM-6a (wf1 persisted) / HM-6b (wf3 temp), mirroring HM-2/WG-6. Single counting axis written once in §5.5 and reused verbatim in §7/§9/§11: 13 validator functions across 14 doc rows; fixture present → 10 green + 3 skip-until-captured on disk; fixtureless → all 13 skip, suite green; 3 temp validators synthetic-proven. | design-v2.md |
| risk-1 | internal-panel | major | accepted | Same root as arch-1: HM-6 id-split applied. Redundancy point decided explicitly — HM-6a carries NO validator (its name+location surface is already pinned by HM-4 `[state].path_output`); kept as a doc row, existence guaranteed transitively via HM-4. Keeps the tally at 13 validators / 10-green. | design-v2.md |
| arch-3 | internal-panel | major | accepted | HM-6a wf1 path corrected to `run_default/outstate/outstates.nc` (fixture-verified) in §5.3/§5.5/§8, with derivation recorded (`dir_output="run_default"` + `[state].path_output="outstate/outstates.nc"`) so a swapper changing `dir_output` knows the path moves. | design-v2.md |
| arch-2 | internal-panel | major | accepted (with primary-source correction) | HM-2 units DEMOTED from pinned to asserted-if-present (wflow name-keyed via `[input.forcing]`; no consumer reads unit attrs — R1). CORRECTION: arch-2's parenthetical that HM-2 temp "has no units attribute at all" is wrong — fixture shows `temp.attrs['unit']='degree C.'`, so the K→°C divergence note IS fixture-grounded; only the `units` (plural) key is absent on pet/temp. Units contract rebuilt on observed attrs; §5.5 encodes asserted-if-present semantics. | design-v2.md |
| repo-3 | internal-panel | minor | accepted-in-part (primary-source correction) | (a) HM-2 precip unit corrected to `mm` (under the `unit` key) — accepted. (b) repo-3's claim that WG-1 uses `unit` (singular) is DISPROVEN by fixture probe: WG-1 stores units under `units` (plural); v1 prose was right. WG-1 row now states the `units`-plural key explicitly. The verified-green facts repo-3 lists (temp* K, kin/kout J m**-2, press_msl Pa, crs=4326, proleptic_gregorian, degree C., float64 lat, Qstats/basin/output headers) need no change and were preserved. | design-v2.md |
| risk-4 | internal-panel | minor | accepted | HM-2 row records the heterogeneous attr-key layout verbatim (precip carries both `units` and `unit`; pet/temp carry only `unit`). Since units are asserted-if-present (not hard-pinned), the validator reads the `unit`/`units` key if present and never KeyErrors on absence. | design-v2.md |
| repo-1 | internal-panel | major | accepted | Fixture-absent `skipif(not os.path.exists(<fixture>))` guard applied to ALL 13 validator tests (not just the 3 temp cases), mirroring `tests/test_extract_climate_wf1.py`. Coverage restated: green when fixture present, skip otherwise; suite green on fixtureless checkouts. §5.5/§8/§9 updated. | design-v2.md |
| repo-2 | internal-panel | major | accepted | Validator idiom fixed to pure `-> list[str]` divergence report (empty ⇒ pass), mirroring `compare_project_consistency` (the drift-guard analog). Chosen over `ValueError` (surfaces all violations at once; composes with test+guard) and over `assert`/`AssertionError` (stripped under `-O`, would fail open in a lifted guard). `assert`/`AssertionError` banned in validator bodies. New §6.5 records the decision; §5.5/§8/§9 updated. | design-v2.md |
| risk-2 | internal-panel | major | accepted | Synthetic in-memory pass+fail unit tests added per temp()-gated validator (WG-4/WG-6/HM-6b): build a conforming object → `== []`, a broken object → `!= []`. Run fixture-independently every suite, so assertion logic is proven this milestone without a fixture edit or behavior change. §9 acceptance updated to skip-on-fixture AND synthetic-proven; R3 mitigation strengthened. | design-v2.md |
| arch-4 | internal-panel | minor | accepted | HM-4 rewrite-field list gains `time.timestepsecs` (=86400, fixture-verified) in §5.3 and §5.6 walkthrough. Added note: wf3 rewrites `time.calendar` to `"standard"` (distinct from wf1/base/HM-2 `proleptic_gregorian`) with the code's noleap rationale — documented cross-artifact fact, not an inconsistency. | design-v2.md |
| arch-5 | internal-panel | minor | accepted | WG-5 row notes the `uri` value is an absolute machine-scoped path emitted by `prepare_climate_data_catalog.py`, deliberately unpinned (portability not a contract), and any uri-resolving validator/guard is machine-scoped. | design-v2.md |
| arch-6 | internal-panel | minor | accepted | HM-1 unpinned count corrected to ~39 (44 total staticmaps data_vars minus pinned set; fixture-verified). HM-4 `[input.forcing]` direction corrected: block KEYS are wflow CSDMS Standard Names, VALUES are forcing var names precip/pet/temp — tie to HM-2 is on the RHS values. | design-v2.md |
| arch-7 | internal-panel | minor | accepted | Considered-and-excluded note added to §5.2 for `wf1_raw/extract_historical.nc` (feeds wf1 parity plots, not a seam). Completeness audit's confirmation that WG-1..6/HM-1..7 cover every seam handoff recorded. | design-v2.md |
| risk-3 | internal-panel | minor | accepted | §5.5 mitigation reworded honestly: WG-5 checks catalog bookkeeping (entry-key grid + driver options) only, NOT WG-4/WG-6 NC content; the NC-content contract is skip-until-captured with no indirect proxy. The oversold "indirect coverage" framing dropped, consistent with §6.3's rejection of proxies. | design-v2.md |
| risk-5 | internal-panel | minor | accepted | Excluded-non-interchange note added to §5.2 for `sim_dates.csv`/`resampled_dates.csv` (weathergenr-internal diagnostics, no DAG-tracked consumer). Converts a silent gap into a documented boundary for the complete-inventory criterion. | design-v2.md |
| repo-4 | internal-panel | minor | accepted (no doc change) | Placement audit is clean — `dev/contracts/` is the correct `dev/` side of the AGENTS.md boundary, kebab-case matches naming.md, no tooling globs `dev/`. Ledger row only, as instructed; OQ-2 already flags the taste call. No design-v2 edit. | design-v2.md (no change) |
| ext1-1 | external-r1 | major | accepted | Two dedicated RELATIONAL validators added (same pure `-> list[str]` idiom, explicit multi-object signatures; §5.5): `validate_hm_gauge_column_identity(toml_cfg, output_rlz_df, qstats_df)` — the HM-4→HM-5→HM-7 gauge-column chain, grounded on rule 3.11's hard-coded `Q_` prefix filter + first-file indexing (`export_wflow_results.py:60-61,123,136` — the silent-corruption path made concrete), C3-bounded (outlets-map id derivation stays wflow-owned: documented, not asserted); `validate_wg5_catalog_grid(catalog_cfg, rlz_num, st_num)` — entry-key set vs the intended rlz 1..N × cst 0..M grid (cst_0 included, `Snakefile_climate_experiment:318-319`), intent derived from the fixture's P3-1 config snapshot via `stress_test_grid` (`snake_utils.py:336`). All relational inputs persist on the fixture (12 per-cst TOMLs, 12 output_rlz CSVs, Qstats, 14-entry catalog) → both continuously verified; synthetic fail cases break one member of each correlated set (renamed Qstats gauge column; dropped catalog key). Counting model 13 → 15 validators; G2/§5.2/§5.3/§5.5/§5.6/§7/§8/§9 updated. | design-v3.md |
| ext1-2 | external-r1 | major | accepted (prior fix defective) | Defect in the v2 repo-1 fix named: skipif-guarding all 13 fixture tests made a fixtureless checkout (fresh clone/CI — where per-commit checks run) report GREEN having executed ZERO validator logic; all-skip green indistinguishable from nothing-checked. Repair combines two reviewer directions: (a) the risk-2 synthetic pass/fail mechanism GENERALIZED from the 3 temp validators to ALL 15 validators (30 fixture-independent in-memory tests; validators sharpened to take parsed objects only — the test layer owns I/O); (b) real-fixture tests retained as the integration layer under the repo skipif convention, with a module-level `_FIXTURE_ABSENT` reason constant + a documented `pytest -rs` acceptance step making fixture absence a named, reported unmet integration gate. (c) A committed contract fixture REJECTED: generated run outputs (repo hard constraint), multi-MB binaries, provenance drift vs the regenerated fixture (new §6.6 records the trade). Counting axis stated once in §5.5, referenced (never re-quoted) from §7/§8/§9. | design-v3.md |

