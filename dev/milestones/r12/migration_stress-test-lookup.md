# Migration — the stress-test lookup

`dev/reference/naming.md` §7 record for the WF3 lookup redesign. Landed
2026-08-16. Three §7 events.

Filed under `r12/` because S8 makes this design R12's prerequisite, so the
migration belongs to R12's record even though it lands first.

**Event 1 — a `rule all` output filename.**

| old | new |
|---|---|
| `<exp>/config/stress_test_design.csv` | `<exp>/config/stress_test_lookup.csv` |

Not a pure rename: the shape moves from one row per member to twelve rows per
member, absorbing `<exp>/climate/weathergenr/_work/st_<m>.csv`, which is deleted
along with the `_work/` directory. `lookup` rather than `design` precisely to
signal that **the shape moved** — "design" would have described the old artifact
just as well and bought nothing for the migration.

**Event 2 — column labels in `rule all` output tables.**

| table | old header | new header |
|---|---|---|
| `<exp>/results/<token>_indicators.csv` | `metric, location, st_id, rlz_id, temp_change, precip_change, value` | `metric, location, st_id, rlz_id, value` |
| the member grid | `month, temp_mean, precip_mean, precip_variance` (multiplier) | `st_id, month, temp_change, precip_change, precip_variance_change` (percent) |

The axis columns are **removed rather than renamed**: they are derived at
reporting time from the lookup (`blueearth_cst/shared/surface_axes.py`;
specification in HM-7).

**Event 3 — a unit change on the parameter grid.** `precip_mean` and
`precip_variance` crossed the Python→R seam as **multipliers** and now cross as
**percent** (`precip_change`, `precip_variance_change`). The R side reconstructs
`1 + <col>/100` for both. The reconstruction is within one `float64` ulp of the
pre-migration level and is bit-identical for every level in every shipped config;
it is **not** exact in general, and cannot be made so — 1,155 of 50,000 `float32`
multipliers admit no `float64` percent that inverts them exactly. The bound holds
unconditionally over the admitted domain (`multiplier ≥ 0.5`), which the producer
refuses below at parse time.

**Machinery updated in the same landing:** `prepare_cst_parameters.py`,
`export_wflow_results.py`, `shared/indicator_tables.py`,
`shared/interchange_contracts.py` (`validate_hm7`, `validate_wg2`,
`_WG2_HEADER`, and the **deletion** of `_PERTURBATION_AXIS`),
`shared/surface_axes.py` (new), `weathergen/impose_climate_change.R`,
`weathergen/read_member_grid.R` (new — the member slice and its
twelve-ordered-months assertion), `run_stress_test.smk` (rules 3.09 / 3.12 /
3.16, `WF3_TARGETS`, and two parse-time refusals),
`dev/scripts/semantic_tree_diff.py`, `dev/scripts/check_baseline.py`
(docstring), `dev/reference/contracts/weather-generator-seam.md`,
`dev/reference/contracts/hydrological-model-seam.md`,
`dev/reference/workflows/rule-index.md`,
`config/templates/snake_config.template.yml`,
`docs/notebooks/Climate Stress Test.ipynb`, and the test sweep including
`tests/test_surface_axes.py` and `tests/test_read_member_grid.py` (both new).

**For an existing project tree:** delete `<exp>/climate/weathergenr/_work/` and
`<exp>/config/stress_test_design.csv`, then re-run WF3. `pixi run tree-check`
reports both as undeclared until they are removed. No user action is needed
beyond that, so no `docs/migration-*.md` guide is published.

**Gate evidence:** `pixi run test-full`, `pixi run tree-check`, and a
`check_baseline.py check` against a baseline re-recorded **before** the first
implementation commit (`11dacdc`, 2026-08-16) and again after the shape change.

## One correction to the accepted design, ruled during implementation

The design's caption rule ("a contiguous circular run of length ≤ 3 → the
initials") contradicted two of its own illustrative captions, which spelled
three-month runs `Oct–Dec` and `Apr–Jun`. **Ruled 2026-08-16: the rule stands**,
so those sets render `OND` and `AMJ`. The rule is stated normatively in both
§5.5 and the §5.8 drop-in text and is what a re-implementer reads; the table
entries were prose that had drifted from the rule beside them — the same drift
that motivated deriving captions instead of typing them. The HM-7 text below
carries the corrected examples.
