Task Brief — R9 P3: result tables, rule 3.11, and the single baseline re-record

### Context

Canonical ruleset: `AGENTS.md`. Master brief:
`dev/milestones/r09/project-tree-task-brief.md`. Runs after P2's tree is settled.

- `export_wflow_results.py` writes three products today: `Qstats.csv` (gauge-point
  discharge statistics), `basin.csv` (basin-averaged fluxes and states), and
  per-location `RT_*.csv` return-period tables.
- `RT_*.csv` is written into `indicators_dir` via `params` and is **not a declared
  output** — invisible to `--dry-run`. It has no in-repo consumer and the
  interchange contract already marks it unpinned.
- `validate_hm7` (`blueearth_cst/shared/interchange_contracts.py:660`) asserts the
  basin table's header is **exactly** `["tavg", "prcp"]`. That holds only when no
  basin-average outputs are configured — true for the test fixture, **false for
  the shipped template default** `wflow_outvars: ["river discharge", "actual
  evapotranspiration"]`. A pre-existing defect this phase must not inherit.
- Both tables are baseline-pinned (`dev/baseline/manifest.json`). This phase owns
  the program's **single** allowed re-record.

### Goal

`q_indicators.csv` and `basin_indicators.csv` in place, value-identical to the
tables they replace; `RT_*.csv` gone; rule 3.11 renamed; `validate_hm7` fixed;
baseline re-recorded once.

### Non-goals

- **The nine R10 rule renames.** Exactly one rule identifier changes here.
- No change to how any indicator is computed. Column order, dtypes and rounding
  stay as they are.
- No new return-period artifact to replace `RT_*.csv`.

### Allowed scope

**Permitted** — `Snakefile_climate_experiment` (rule 3.11 only),
`blueearth_cst/experiment/export_wflow_results.py`,
`blueearth_cst/shared/interchange_contracts.py`, affected tests.

**Approval-gated** — `dev/baseline/manifest.json`, released only by master
**Gate 2**.

**Forbidden** — any other rule identifier; the indicator computations themselves.

### Required changes (checklist)

1. `Qstats.csv` → `q_indicators.csv`; `basin.csv` → `basin_indicators.csv`, in the
   rule's `output:` and in the script.
2. Remove `RT_*.csv` generation, including the `Q_rps` accumulation and its
   `xr.concat` — dead once nothing consumes it.
3. Rename rule 3.11 `export_wflow_results` → `derive_wflow_indicators`, updating
   its `log:`/`benchmark:` prefixes **and its `LOG_RULES` entry in the same edit**.
4. Fix `validate_hm7`: the basin table's header is `tavg`, `prcp`, **plus one
   column per configured `*_basavg` variable**. Assert the prefix and the
   perturbation-axis columns, not an exact two-column list.
5. Update the interchange contract's names and drop its `RT_*.csv` sentence.

### Commit plan

A contract rename must ride with the consumers it breaks, and the re-record must
be attributable on its own.

| # | Subject | Paths | Invariant preserved |
|---|---|---|---|
| 1 | `r09: fix the basin-table header assertion` | `interchange_contracts.py`, tests | the validator is correct **before** the rename moves it — otherwise a fixture-shaped assertion is carried into new names |
| 2 | `r09: rename the result tables and rule 3.11` | WF3 Snakefile, `export_wflow_results.py`, `interchange_contracts.py`, tests | names and every consumer change together; `LOG_RULES` included |
| 3 | `r09: drop the RT_*.csv side tables` | `export_wflow_results.py` | removal is separately revertible from the rename |
| 4 | `r09: re-record the baseline manifest` | `dev/baseline/manifest.json` | **after Gate 2 only**; one commit, so the re-record is attributable to nothing else |

### Validation

**Named scope — run this and nothing else:** `tests/test_export_wflow_results.py`,
`tests/test_interchange_contracts.py`, `tests/test_check_baseline*.py`. This
phase touches no WF1 or WF2 code; their suites are not upstream of anything here.

1. **Narrow** — the named scope (per edit), plus the new `validate_hm7` case with
   `*_basavg` columns present, which must fail on today's assertion.
2. **Integration** — `pixi run test-cli` after the rule rename (rung 2's trigger:
   a rule identifier changed).
3. **Phase gate** — `pixi run test-fast` once, at phase end. **Not** the full
   suite.
4. **Non-regression** — `check_baseline.py check`, expected **green after**
   commit 4 and red before it.

**Falsifier — value identity.** Claim: *the renamed tables are byte-identical in
content to the ones they replace*. Capture `Qstats.csv` / `basin.csv` from a
pre-P3 run, then diff against `q_indicators.csv` / `basin_indicators.csv`
element-wise. A green baseline after a re-record proves nothing about identity —
the re-record is what makes it green. The comparison must be against the
**retained pre-P3 artifacts**.

### Acceptance criteria

- Element-wise identity against the retained pre-P3 tables, demonstrated.
- No `RT_*.csv` produced anywhere in a full run.
- `validate_hm7` passes under both `wflow_outvars` shapes — the fixture's and the
  template default's.
- Baseline re-recorded exactly once, its manifest diff limited to the two renamed
  keys, and `check_baseline check` green afterwards.
- Rollback: any non-identical value reverts commits 2–4 and reports at Gate 2.

### Output requirements

A phase report with the element-wise comparison result, the manifest diff, and a
Results delta section. If the delta is empty — the expected outcome — say so
explicitly; an absent section reads as an unrun check.

### Task constraints

- Do not run `check_baseline.py record` before master Gate 2 authorises it.
- Do not rename any rule other than 3.11.
- A missing `LOG_RULES` update is silent, not an error — verify the merged log
  contains a `derive_wflow_indicators` section after a full run.
