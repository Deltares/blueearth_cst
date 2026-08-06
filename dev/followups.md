# Followups

Issues surfaced during pre-M1 cleanup that belong to later milestones.
Per the roadmap's "no milestone touches the next milestone's territory" rule,
captured here and resolved in passing when the relevant milestone starts.

Convention: one bullet per item. Keep the diagnosis date and reproducible
context so future-you can confirm the issue still applies before fixing.

---

## Carried over from the roadmap (moved 2026-08-02)

These sat in `roadmap.md` as a third backlog, referenced by neither
`TODO.md` nor this file. Content unchanged; only the location is new.

### Minor open items

Small decisions that don't justify a section of their own. Resolve
in passing as the relevant milestone starts.

- ~~**CI.**~~ **DONE 2026-07-25 (first Phase-4 item)** —
  `.github/workflows/ci.yml` runs the unit suite on push to `main` and on PRs,
  across both supported pixi platforms (`ubuntu-latest` + `windows-latest`,
  `fail-fast: false`), with `locked: true` so `pixi.lock` drift fails the run.
  Scope set by measurement: a bare checkout gives 386 passed / 30 skipped /
  1 xfailed in ~100 s, every skip principled (~27 need the untracked
  `examples/test_local` fixture, 3 are the `--run-integration` end-to-end
  tests). **`check_baseline.py` turned out NOT to be the natural fit this entry
  assumed** — it fingerprints targets inside that untracked fixture tree, so it
  cannot run on a runner and stays a local gate, as does `semantic_tree_diff`
  whole-tree diffing. The ubuntu leg is also the first time the linux-64 half of
  `pixi.lock` has been resolved anywhere, so it de-risks the parked Linux work
  below.
- **R testthat coverage.** Decided at the start of R5 — Python
  helpers only by default; adding R testing infrastructure is a
  separate call.
- **Linter for naming conventions.** R2 establishes the convention
  but does not enforce it. A future linter (ruff custom rule, or a
  small ad-hoc script) would mechanically catch drift. Add as an
  R3+ followup if drift becomes a real problem.

---

### Deferred: Linux replication

Currently parked because no Linux machine is available locally. Not
abandoned — to be picked up when a Linux box, WSL setup, or Deltares
P-drive mount becomes available.

**What this covers when reactivated.**
- Reproducing the M1 baseline on Linux using
  `config/snake_config_model_test_linux.yml`.
- Rebuilding the Docker image on top of the M2 env manager and
  validating `run_snake_docker.sh`.
- Confirming the M2 env file resolves cleanly on Linux (it was
  authored cross-platform during M2).
- Sorting out the Deltares P-drive mount
  (`/mnt/p/wflow_global/hydromt`): whether the baseline is captured
  natively or only inside the container.
- Collapsing the OS-specific data catalog split (`*_linux.yml`) into
  a single parameterized catalog or config selection.
- Once green, recording Linux-specific fingerprints alongside the
  Windows ones in `dev/baseline/` (separate manifest, not a
  replacement).

**Where it slots in.** Likely a small dedicated Phase 2 milestone
when picked up (`r0X-linux-parity` between two existing R milestones)
so that subsequent milestones can assume both platforms work.

**Until then.** All milestone exit criteria refer to Windows only.
Linux-specific files (`*_linux.yml`, `run_snake_docker.sh`, the
Dockerfile) must continue to build / parse but are not exercised
end-to-end. Don't delete them.

---

## Post-R10-design (surfaced 2026-08-06 during the rule-index name-vs-body audit)

Every rule identifier in the three Snakefiles was checked against its **script or
shell body** while writing `dev/reference/workflows/rule-index.md`. Three names
were ruled on directly and are recorded in `dev/milestones/r10/rule-naming-design.md`
amendment 2.

The audit also raised four **structural** candidates — three merges and one
split. All four were ruled on 2026-08-06: **M1 and S1 accepted, M2 and M3
rejected.** None is an R10 item; that milestone is identifier-only.

- **[R10-1] Merge rule 1.07 `setup_runtime` into 1.08 `add_forcing`.**
  *Accepted 2026-08-06; not implemented.* 1.07 writes a hydromt forcing build
  recipe (`<model>/config/build_historical_forcing.yml`) whose **only** consumer
  is 1.08, which runs `hydromt update wflow_sbm -i` against it. Two rules, one
  job. The merge is the reason 1.07's R10 rename was withdrawn rather than
  replaced: a recipe that never leaves the pair needs no name of its own, so the
  naming drift disappears with the rule instead of being renamed around.

  **The implementation cost is real.** Snakemake allows one of `script:` /
  `shell:` per rule. 1.07 is a Python `script:` (`setup_time_horizon.py`, which
  also opens the model's staticmaps to size a chunksize); 1.08 is a `shell:`
  invoking the hydromt CLI. The merged rule must either call hydromt's Python API
  or shell out from inside a script. Decide at the same time whether
  `build_historical_forcing.yml` stays a **declared** output — it is kept today
  as provenance of the model it built (`Snakefile_model_creation` rule 1.07's
  comment, design v10), and demoting it to an undeclared side-write would lose
  that without saying so.

  **Sequencing against R10:** either order works. If the merge lands first, 1.07
  has no rename to skip; if R10 lands first, 1.07 keeps `setup_runtime` until the
  merge deletes it. What must not happen is renaming 1.07 in passing.

- **[R10-2] Split rule 1.11 into a metrics rule and a figure rule.**
  *Accepted 2026-08-06; not implemented.* Today one rule writes both
  `<model>/evaluation/performance_metrics.csv` — **baseline-covered data** — and
  the evaluation figures, which `check_baseline.py` **excludes by default**
  (`FIGURE_KINDS`). The DAG therefore cannot express the distinction the
  `AGENTS.md` validation ladder turns on: "do not run the validation suite or the
  baseline for a figure-only change" is written guidance that the rule graph
  contradicts.

  Target shape: **1.11 `evaluate_wflow_run`** (metrics) → **1.11b
  `plot_wflow_evaluation`** (figures), using the letter-suffix convention already
  set by 1.01b / 2.03b / 3.00b / 3.01c–e. The figure half keeps the R10 target
  name, so that rename is unaffected either way.

  **Honest about the size of the win.** The harm today is a wasted re-run and a
  needless baseline comparison, not a wrong number: a plotting edit re-runs 1.11
  and rewrites identical metrics, so the gate passes. The gain is that a
  figure-only change becomes *visible as one* to Snakemake.

  Two open sub-decisions for whoever implements it:

  1. **How the halves share data.** `plot_results.py` loads the run, the climate
     store and the observations once and uses them for both. The split needs
     either a re-read of `output.csv` in the figure rule or a declared
     intermediate. Prefer the re-read unless it measures badly — an intermediate
     is a new artifact in a tree R9 just finished settling.
  2. **The metrics rule's verb.** `evaluate_` would be a **19th verb**;
     `derive_wflow_metrics` reuses an existing one, but `derive_` is explicitly
     reserved for a workflow's *terminal* product and the metrics table is not
     WF1's. Rule on this before the sweep, not during it.

- **[R10-3] Two consolidations REJECTED, recorded so they are not re-raised.**
  Both looked obvious and both are wrong; the reasons are structural, not
  aesthetic.

  **M2 — merge 1.06 `write_outlet_index` into 1.05. Rejected.** The two were
  paired thematically ("both wire the model to named stations"), and the
  structure does not agree. 1.06's inputs are `outlets.geojson` (rule 1.03) and
  `location_registry.csv` (rule 1.02) — nothing else — while 1.05 waits on
  `reservoirs_lakes_glaciers.txt` from 1.04. So 1.06 runs in parallel with 1.04
  and 1.05 today, and merging would serialise a cheap deterministic pandas join
  behind the waterbody update *and* a hydromt `r+` model mutation. Re-deriving
  the crosswalk after a registry fix would then re-run the hydromt update.
  **The merge adds a DAG edge rather than removing one.**

  **M3 — merge `gather_benchmarks` + `gather_logs` per workflow. Rejected.**
  Both merge functions call `_remove_parts`: they **delete the parts they
  consumed**. In one rule that becomes a partial-failure hazard — the log merge
  succeeds and deletes its parts, the benchmark merge raises, Snakemake removes
  both declared outputs as a failed job, and the re-run finds no log parts and
  rewrites the merged log with "no part from this run" for every rule. Today
  `gather_logs` succeeding independently means its output survives a
  `gather_benchmarks` failure. Three rules out of forty is not worth a path that
  silently degrades a durable artifact. Revisit only if the deletion is made safe
  first (both merges complete before either deletes), which is work the current
  split gets for free.

  **Also do not merge 3.01c `write_model_reference` and 3.01d
  `check_model_reference`.** They read as an obvious pair and merging them
  destroys the guard. 3.01c's model inputs are `ancient()` *on purpose* — if the
  reference were rewritten whenever the model changed it would always match, and
  3.01d's comparison would be decorative. 3.01d's sentinel is `temp()` for the
  mirror reason: a persisted verdict satisfies 3.09's edge after the model has
  drifted. The asymmetry *is* the mechanism.

  **Generalizes:** two rules being small, adjacent and thematically similar is
  not an argument for merging them. Check what each actually depends on, and
  whether either destroys its own inputs.

- **[R10-7] Rename the three shared-rule helpers from `_spec` to `_rule`.**
  *Ruled 2026-08-06; not implemented.* `region_spec` → `region_rule`,
  `climate_store_spec` → `climate_store_rule`, and the new one from `[R10-6]` is
  born as `spatial_units_rule`. Dataclasses follow (`RegionRule`,
  `ClimateStoreRule`, `SpatialUnitsRule`), as does `tests/test_region_spec.py` →
  `tests/test_region_rule.py`.

  **Why:** `spec` reads as jargon to a non-programmer, and the object is not a
  specification of anything abstract — it holds a rule's `script`, `inputs`,
  `outputs` and `params`, i.e. a rule definition minus its `message`/`log`/
  `benchmark` labels. "The region rule" says what it is. `_contract` was
  rejected: this repo already uses "contract" for interchange surfaces
  (`dev/reference/contracts/`, `SPATIAL_CONTRACT_VERSION`,
  `test_climate_store_contract.py`), and overloading it would be worse than the
  jargon. `_definition` was the runner-up, rejected on verbosity at the call
  sites.

  **Scope:** all three, not just the new one — three sibling helpers under two
  suffixes is the inconsistency the shared-rule pattern exists to prevent.
  **21 files** reference the current names (`snake_utils.py`, all three
  Snakefiles, eight tests, two `dev/scripts/`, and several `dev/reference/`
  docs). Mechanical, but note `climate_store_spec` appears inside an error
  *message string* in `snake_utils.py` and in module docstrings — a
  symbol-only rename misses both.

  **Land it in the R10 sweep**, which already edits all three Snakefiles.
  `dev/reference/naming.md` documents no `_spec` convention (its suffix rules
  cover `_path` / `_ds` / `_cfg`), so nothing there contradicts this — but the
  sweep should *add* the `_rule` convention so the next helper is named from a
  rule rather than by analogy. Do **not** rewrite `dev/milestones/` archives,
  per R10's validation item 4.

- **[R10-6] Split `prepare_spatial_maps` so WF2 and WF3 can consume basin and
  subbasin boundaries.** *Designed 2026-08-06 as ADR 0003 §8–12 (**proposed**,
  not accepted); not implemented.* WF2 and WF3 declare `delineate_region` and no
  other spatial rule, and neither workflow's scripts read a vector layer today.
  The split puts the vector half — basins, subbasins, catchments, rivers,
  locations, registry — behind a third shared spec declared in all three
  workflows, and leaves the thematic raster stack (`vito`, `modis_lai`,
  `soilgrids`) in a WF1-only rule.

  Full context, decision, consequences, alternatives, validation and four open
  questions in `dev/decisions/0003-one-shared-region-artifact.md`. **Land it
  before `[R10-5]`** — it adds a WF1 rule, so renumbering first moves the numbers
  twice.

  The record also carries two **identity** changes ruled the same day, which ride
  with the split because they live in the same vector half:

  - **§11 — `automatic_subbasins.max_count` becomes per-basin, default 20 → 11.**
    Today it is one global budget, area-weighted across parents, that *raises*
    when parents exceed it. Per-basin removes that failure and deletes
    `allocate_automatic_subbasin_budgets` outright. Safe because
    `select_automatic_subbasins` treats the count as an upper bound.
  - **§12 — `wflow_id` becomes a per-basin block of 100** (basin 1 → 100, 101,
    102 …). Today a subbasin primary gets `basin*100 + n` while any additional
    point gets `1_000_000 + subbasin_id*100 + n`, so basin 1's second gauge is
    `1_010_102` — a seven-digit id beside a three-digit one in the same column.

  **§12 is a baseline event and must not share a commit with the rest.** `wflow_id`
  values name Wflow's gauge output columns, so renumbering renames every
  `Q_<id>` / `P_<id>` in `output.csv`; `check_baseline.py check` fails until
  re-recorded. §8–11 are behaviour-preserving, §12 is not — landing them together
  destroys the ability to tell an intended diff from a regression.

  Two things not to lose: the hydrography-read cost §8 adds to WF2 is asserted,
  **not measured**; and `wflow_id == subbasin_id` stops holding for primary
  locations under §12, so grep for code relying on that identity first.

- **[R10-5] Renumber every rule so `W.NN` follows the logical order.**
  *Accepted 2026-08-06; not implemented.* Numbers become **positional**: data
  first, then model build, then run, then records, contiguous within each
  workflow, with every dependency pointing from a lower number to a higher one.
  The full old→new map for all 45 identifiers is in
  `dev/reference/workflows/rule-index.md` § *What changed*.

  **This overrides `rule-naming-design.md` §9**, which recommended amending the
  convention to "a stable identifier assigned at rule creation" and *not*
  renumbering. That recommendation was made on cost grounds and the owner ruled
  the other way, so §9 is amended to record the reversal rather than the advice.

  **The cost, and it is the reason §9 said no: numbers are REUSED.** New 1.07 is
  `write_outlet_index`; old 1.07 was `setup_runtime`. New 3.05 is
  `check_model_reference`; old 3.05 was the deleted `prepare_weagen_config_st`.
  Under the old policy a retired number stayed a gap, so a stale reference merely
  dangled and was obvious. Now it silently resolves to a **different rule**.
  Every `W.NN` in `dev/milestones/`, `DEVLOG.md`, `dev/decisions/` and the
  Snakefile comments predates the map and must be read as of its date. Do not
  "fix" archived milestone documents to the new numbers — the same reasoning
  R10's validation item 4 already applies to old rule *names*.

  **Migration surface**, per renumbered rule — the same six call sites the
  rename sweep touches (`LOG_RULES` entry, `W.NN` comment header, `rule`
  identifier where a rename coincides, `rule_banner`'s first argument, the `log:`
  path, the `benchmark:` path). Two extra hazards specific to renumbering:

  1. **`LOG_RULES` order is the merge order.** Renumbering changes both the
     labels and their intended sequence; update the list wholesale, not entry by
     entry, or the merged log comes out in a mixed order.
  2. **Rule 3.14 keeps a singular log label** (`3.14_run_wflow`) while its
     identifiers stay `run_wflow_batch_<b>`. The divergence is deliberate and
     survives renumbering — do not "fix" it.

  **Do it in the same sweep as R10's renames.** They touch the same six call
  sites per rule, want the same validation (`pytest tests/test_cli.py`, then a
  full three-workflow run confirming the merged log has a section per
  `LOG_RULES` entry and no `_parts/` survives), and splitting them means paying
  that cost twice. The baseline is unaffected either way — part paths are
  transient and no output path or value changes.

  **Going forward:** do not renumber to insert a rule. Use a letter suffix
  (`1.09b`) until the next deliberate sweep.

- **[R10-4] Stale rule references in Snakefile comments.** Cosmetic, found by the
  same read. `Snakefile_climate_experiment` names the **deleted** rule 3.05 twice
  — the 3.00b comment still lists `prepare_weagen_config_st` as one of the four
  per-experiment roots, and 3.13's comment says "3.05/3.07/3.09 write one part
  per (rlz, cst)". Separately, all three `gather_benchmarks` comments describe
  their output as `wf<N>_benchmarks.tsv`; the declared output is `.md`. Gate is
  `pytest tests/test_cli.py` (comments only, but the files are Snakefiles).

---

## Post-R9 (surfaced 2026-08-05 during the R9 self-test)

- **[R9-1] Six geojson basenames collide across `data/spatial/geoms/` and
  `models/hydrology/wflow/staticgeoms/`, meaning different things in each.**
  Raised as "how do we prevent drift between the files that are the same"; the
  measurement says none of the six pairs *are* the same, which makes
  misidentification the exposure rather than drift. Measured on
  `test_case/test_local`, 2026-08-05:

  | layer | features (`data/spatial` / `staticgeoms`) | relationship |
  |---|---|---|
  | `region.geojson` | 1 / 1 | **different objects** — ours 0.017847, hydromt's 0.026667, IoU 0.67, ours ⊂ hydromt's exactly (`a\b` = 0). Ours is the delineated basin; hydromt's is the model grid extent. |
  | `basins.geojson` | **1 / 5** | hydromt's own per-subbasin polygons (`value` column); union area identical |
  | `rivers.geojson` | **3 / 4** | different provenance — 21 MERIT-style attrs vs `idx, idx_ds, pit, strord` |
  | `subbasins` / `catchments` / `locations` | 5 / 5 | true copies — identical geometry AND identical column schema, incl. our own `delineation_method`, `subbasin_code` |

  **Temporal drift is already structurally impossible** and needs nothing: there
  is no second independent producer. `data/spatial/` is upstream —
  `Snakefile_model_creation:385-392` declares all six as inputs to rule 1.03,
  so the model rebuilds whenever they change. That is ADR 0003's fix, and
  `spatial/delineate_region.py:7-14` records the pre-ADR state this replaced
  ("agreed exactly — agreement maintained by coincidence, since nothing
  compared them").

  Three things are exposed, none of them drift:

  1. **Name collision.** `basins.geojson` is "the basin" in one directory and
     "five grid-derived polygons" in the other, with nothing in the tree saying
     so. A future rule, the GUI, or a later reader takes the wrong one and is
     silently wrong.
  2. **Eight of ten `staticgeoms/` files are undeclared outputs.** Rule 1.03
     declares only `region.geojson` and `outlets.geojson`
     (`Snakefile_model_creation:400-401`); the rest are hydromt side effects
     Snakemake does not track, so a partially-failed build can leave stale ones.
  3. **The `region` containment relationship is unrecorded.** `ours ⊆ hydromt's`
     held exactly here; a hydromt upgrade that changed how the model extent is
     derived would break it with nothing watching.

  Options when this is picked up, in preference order: **(a)** a contract check
  asserting the *relationship* rather than equality — containment for `region`,
  topological equality for the three copy layers — reusing
  `dev/scripts/semantic_tree_diff.py:1250` `compare_geojson`, which already
  compares CRS, row count, non-geometry columns and geometry topologically
  (it exists because byte comparison is wrong for this format); **(b)** document
  in the seam contract and the R9 path map which directory is authoritative for
  which question; **(c)** declare the eight undeclared `staticgeoms/` outputs on
  rule 1.03 — closes the stale-artifact gap but couples our DAG harder to
  hydromt's output surface.

  **Do not rename or suppress anything under `staticgeoms/`** — it is
  hydromt_wflow's own output surface and `AGENTS.md`'s hard constraint puts it
  off-limits.

  Reproduce the table with `gpd.read_file` on the two directories of any built
  project; no run needed if `test_case/test_local` is present.

- **[R9-2] Baseline re-record owed for the indicator-table axis-column rename.**
  Raised 2026-08-05 as "climate variable terminology is inconsistent across
  workflows" (`temp`/`tavg`, `precip`/`prcp`). The screening found exactly one
  real violation: `q_indicators.csv` / `basin_indicators.csv` spelled the two
  perturbation-axis columns `tavg` / `prcp`, while every other producer already
  used the `precip` / `temp` stems `naming.md` §6 tier 2 declares. Renamed to
  `temp_change` / `precip_change`; full rationale, the old → new map, and the
  alias list that only *looks* like drift are in
  `dev/milestones/r09/migration_indicator-axis-columns.md`.

  **CLOSED 2026-08-05.** WF3 re-run from the primary checkout on `main@03e546c`;
  `pytest tests/` green (1356 passed) and the baseline re-recorded after the
  ADR 0001 step-7 attribution came back conclusive: **zero numeric movement**.
  Reverting the header line alone reproduced both recorded `sha256` exactly, body
  bytes compared equal, and both sizes moved by exactly +16 — the header delta and
  nothing else. Only rule 3.11 re-ran, so the reduction consumed byte-identical
  inputs. Full evidence in
  `dev/milestones/r09/migration_indicator-axis-columns.md` §5. It also closed the
  mixed-provenance residual `check_baseline.py` had carried since the restoration:
  the wf1 delta demonstrably does not survive the wf3 reduction, and that
  docstring is now a result rather than a warning.

  <details><summary>What was outstanding (kept for the record)</summary>

  **TWO gates, one fix: re-run WF3 from the primary checkout.**
  Code, tests, contract doc and `naming.md` are done; both remaining gates fail
  only because the fixture tree still holds pre-rename output.

  1. **`pytest tests/` from the primary checkout FAILS** — the branch's merge
     gate, not just a reporting nicety. `test_hm7_integration` and the 12
     `test_gauge_identity_integration` cases parse the real
     `q_indicators.csv` / `basin_indicators.csv`, which still carry `tavg` /
     `prcp`. They fail *correctly*: `validate_hm7` is meant to reject the old
     spelling. This worktree passed only because it has no `test_case/test_local`
     and those 26 cases skip.
  2. **`check_baseline.py check` FAILS** — `manifest.json` fingerprints both
     tables byte-exact. Expected re-record diff is exactly two entries'
     `sha256` / `size_bytes`; a third entry moving means something else changed
     too. **But those two can move for TWO reasons at once** — the header rename
     *and* the pre-restoration wf3 provenance finally catching up with the
     restored wf1 slice (`check_baseline.py` module docstring; the wf3 rows were
     deliberately never re-recorded because the discharge move was immaterial).
     Do not re-record on sight: follow ADR 0001 step 7's immaterial branch —
     confirm the movement is consistent with the recorded wf1 diff first. Full
     procedure and commands in
     `dev/milestones/r09/migration_indicator-axis-columns.md` §5.

  Order: WF3 run → `pytest tests/` → ADR 0001 step-7 consistency check →
  baseline re-record.

  </details>

- **[R9-4] R9 moved the project tree but never re-pointed the interchange
  contract tests. FIXED 2026-08-05.** The whole Layer-2 block of
  `tests/test_interchange_contracts.py` still used pre-R9 paths —
  `climate_historical/`, `hydrology_model/`, `weather_generator/`,
  `hydrology_runs/rlz_<n>/`, and the loose `data_catalog_climate_experiment.yml`
  at the experiment root — plus the pre-flattening member naming
  (`rlz_<n>/config/cst_<m>.toml` rather than `config/rlz_<n>_cst_<m>.toml`).
  **22 failures** on the first post-R9 `pytest tests/` in the primary checkout.

  **Why it stayed invisible, which is the part worth keeping.** The block is
  `skipif(not _fixture_present())` and `test_case/test_local` is untracked, so
  it is absent in every worktree and on CI — `AGENTS.md` already says CI covers
  only what a bare checkout can run. R9's gates were `semantic_tree_diff` and
  `check_baseline`, which validate the tree's SHAPE; neither reads the code that
  reads the tree. So the only check that could have caught it is the one only
  the primary checkout can run, and it had not been run since R9 landed.

  Three of the paths were worse than merely broken: `_WG4_NC`, `_WG6_NC` and
  `_HM6B_NC` sit behind a runtime `os.path.exists` guard, so a wrong path reads
  as "temp() artifact absent" and **skips silently** — indistinguishable from a
  normal run, forever.

  Fixed by deriving four roots (`_MODEL_DIR`, `_STORE_ROOT`, `_WG_DIR`,
  `_RUNS_DIR`) named after the Snakefile variables they mirror, so the next tree
  move is a one-line edit rather than a dozen literals. **Verified only against
  the fixture's real layout on disk, not by a green run** — this worktree has no
  fixture. Confirm with `pytest tests/test_interchange_contracts.py` in the
  primary checkout.

  **Generalizes:** any milestone that moves the project tree must grep the test
  suite for the old roots, because the suite's fixture-dependent layer cannot
  fail in CI.

- **[R9-3] The response-surface axis columns hold JANUARY, not an annual value.**
  Surfaced 2026-08-05 while writing R9-2's rename, reading the code the rename
  touched. `export_wflow_results.py` does `df_st["temp_mean"].iloc[0]`, but
  `cst_<m>.csv` has **twelve rows, one per month** — `prepare_cst_parameters`
  builds them from the config's 12-element `min` / `max` vectors. So the value
  labelled `temp_change` / `precip_change` for a stress-test member is its
  January perturbation.

  **Never observed wrong, and that is the whole problem.** Both the shipped
  template and the seed config use flat vectors (`min: [0.0]*12`,
  `max: [3.0]*12`), so January *is* the annual figure there. A project with a
  seasonal perturbation vector — which the config schema explicitly supports,
  and which `transient_change: true` invites — gets a response surface silently
  indexed by one month. Same class as the fixture-shaped `validate_hm7`
  assertion R9 P3 fixed: correct on the fixture, wrong for the general config.

  **Not fixed in passing, deliberately.** Collapsing 12 months to one axis value
  is a method question, not a typo — mean? annual total (right for precip,
  wrong for temp)? or does a seasonally-perturbed run need a different response
  surface altogether? It also moves numbers, so it is a baseline event. Recorded
  in the code at the read site so nobody re-derives it. Reproduce by setting a
  non-flat `stress_test.temp.mean.max` and comparing the emitted axis column
  against the intended annual mean.

---

## Post-R8 (surfaced 2026-08-02 during the Post-R7 triage)

- **[R8-1] The ruff gate is red on `main`.** *Row `t260802a`.* `pixi run ruff
  check .` reports **10 findings**, 7 of them auto-fixable:

  | File | Finding |
  |---|---|
  | `projections/get_change_climate_proj.py` | F401 ×5 — `os`, `series_identity`, `dry_month.FLAGGED_STATUS`, `dry_month.is_flagged`, `snake_utils.log_row` |
  | `projections/get_stats_climate_proj.py:96` | F841 — `ds` assigned, never used |
  | `projections/resolution.py:29` | F401 — `os` |
  | `tests/test_monthly_change.py:5` | F401 — `pytest` |
  | `tests/test_variable_spec.py:77,99` | E702 ×2 — semicolon statements |

  This is not cosmetic: `.github/workflows/ci.yml:75` runs exactly this command
  on both legs, so **CI fails on an untouched checkout**. R7-16 recorded the
  gate as adopted with "all 96 findings cleared" (`85d3178` → `81e0096`); every
  finding above is in an R8-era file, so the gate went red during the WF2 v2.0
  rework and the seal did not catch it.

  Verified to predate this triage: the identical 10 findings reproduce at
  `207c449`, the commit before any `dev/` tidy work. Most are dead imports left
  by refactoring — likely `ruff check --fix` plus two hand edits, but the F841
  in `get_stats_climate_proj.py` should be read before deleting, in case the
  assignment was meant to be used.

  Worth asking as part of the fix: the seal ran CI green (R7-19 cites run
  30450296441), so either the gate was added after that run or a later red was
  not acted on. The answer decides whether anything beyond the ten fixes is
  needed.

---

## Post-R7 (surfaced 2026-07-28/29 during the R7 project-layout milestone)

R7 landed as 15 `r07:` commits with a clean full-tree diff, a green
`check_baseline`, and the P4 assertion demonstrated. The items below are what it
deliberately did **not** fix, plus what implementation surfaced along the way.
Provenance: `dev/milestones/r07/migration_project-layout.md` §§7a–7d,
`dev/milestones/r07/project-layout-design.md`, and the `r07:` commit messages.

### Defects — worth fixing

- **[R7-1] ~~`wflow_sbm.toml` is written by five rules and declared by one.~~
  FIXED 2026-07-29** — rule 1.03 now emits a `touch()` completion sentinel
  (`hydrology_model/.model_built`) that rule 1.04 consumes as a **non-ancient**
  input, so a rebuild re-fires the whole toml-writing chain
  (1.04 → `.txt` → 1.05 → `outlets.geojson` → 1.07 → forcing yml → 1.08).
  The obvious fix — dropping the `ancient()` on staticmaps — was **wrong**:
  1.04/1.05 commit writes back into staticmaps themselves via
  `mod.write()`/`mod.close()`, so a plain edge would re-trigger them on their
  own execution forever. Regression test falsified both ways.

  *Original diagnosis, kept for the record* (map §7c): rule 1.03 `create_model`
  creates it; rules 1.04–1.09 update it
  **in place** while taking `ancient(f"{basin_dir}/staticmaps.nc")`, which
  suppresses exactly the mtime trigger a rebuilt staticmaps would fire. So
  **anything that re-fires `create_model` alone leaves the TOML stripped** of
  every section the later rules added, and the next wflow run dies on the
  missing key. Bit three times during R7 (commits 7, 10, and once more during
  the config split); each recovery needed `--forceall`. Pre-existing, not an R7
  regression — R7 is simply the first thing in a while to re-fire the build.
  Fix is a rule-shape change (declare the TOML on every rule that writes it, or
  make the update chain depend on it), which is why a behaviour-preserving
  milestone could not take it. **Highest-value item in this list.**

- **[R7-2] ~~The store's freshness boundary stops at the catalog file.~~
  CLOSED — WON'T FIX, 2026-07-29 (owner-ruled).** Editing the catalog
  mtime-triggers exactly one re-extraction (R7 closed *that* gap, which
  pre-dated the milestone). Data *behind* an unchanged catalog entry — a local
  file the entry points at, or a remote store — participates in no trigger, and
  it is staying that way. Three reasons, and the ruling records them so this is
  not re-opened as an oversight:

  1. **It is outside CST's automation scope.** Enumerating catalog-resolved
     sources as DAG inputs means parsing hydromt catalog semantics at
     DAG-parse time — re-implementing how hydromt resolves data. `AGENTS.md`
     Hard Constraints put that off-limits: consume hydromt conventions
     verbatim, never re-engineer them.
  2. **It is not implementable for the general case anyway.** Remote sources
     (the CMIP6 GCS store, any URI-backed entry) expose no usable mtime, so
     even a correct implementation would cover only local-file entries and
     silently miss the rest — a gate that looks complete and is not, which is
     worse than a documented gap.
  3. **The catalog-conventional signal already exists.** The supported way to
     record a data change behind a stable entry is to edit the entry (path,
     version, or meta), which the R7 input edge now picks up. For a genuine
     in-place data mutation the escape hatch is
     `snakemake --forcerun extract_climate_grid`, documented in
     `dev/milestones/r07/migration_project-layout.md` §2f.

  Revisit only if hydromt gains a first-class "resolve this entry to its
  concrete sources" API — at which point this becomes consuming an upstream
  convention rather than re-implementing one.

- **[R7-3] ~~`basin_area.png`: which change produced the 134,828-byte figure?~~
  ANSWERED 2026-07-29.** It was written by a **different branch**.
  `feat/outputs-figures` carries `e917a8e` *"redesign basin_area.png as a
  self-contained basin map"* and `c2f4881` *"degree-aware gridline locators"*,
  both dated 2026-07-25 and **neither in `main`'s history**. That branch's
  `plot_map.py` defines `_add_scale_bar`, `_add_north_arrow`, an `area_km2`
  title and a `YlOrBr` colormap — precisely the figure found in the fixture;
  HEAD's `plot_map.py` contains none of them and cannot produce it.

  So the earlier reading was wrong in its premise: `plot_map.py` **has** changed
  since R6, just not on this line of history. Nobody's figure was corrupt —
  someone ran wf1 from `feat/outputs-figures` into the shared fixture, and the
  artifact outlived the branch checkout. The manifest (recorded from main-line
  code) and the fixture (written by feature-branch code) then disagreed, exactly
  as observed. Closed as **explained, not a defect**; the commit-14 re-record
  already restored agreement. Generalised as R7-21 below, which is the part that
  matters.

- **[R7-21] ~~The baseline fixture is branch-shared mutable state.~~
  MITIGATED 2026-07-29** — candidate (a) implemented: `record` stamps
  `recorded_by` (branch, commit, dirty) into the manifest and `check` prints a
  provenance line **before** the verdict, warning loudly when the recording
  branch differs from the checking one. Advisory only: it never changes the exit
  code, because the failure mode is silent misattribution rather than
  corruption, and a deliberate cross-branch check is legitimate. A pre-stamp
  manifest says so rather than pretending. The R7-3 scenario is simulated in
  `tests/test_check_baseline_provenance.py`. The underlying *sharing* is
  unchanged — candidates (b) branch-derived fixture paths and (c) per-branch
  regeneration remain open if misattribution recurs despite the warning.

  *Original diagnosis, kept for the record.* `test_case/test_local` is
  **untracked**, so it is not part of any branch: every branch, worktree and
  session that runs a workflow writes into the *same* tree. Consequences, all
  observed rather than hypothesised:

  - A figure produced by `feat/outputs-figures` sat in the fixture for days and
    was read as the pre-R7 baseline reference (R7-3).
  - `check_baseline check` therefore answers "does the tree match the manifest"
    for **whichever branch ran last**, not for the branch you are on. A green
    check can mean someone else's code is consistent with your manifest.
  - The R7 milestone's own gate captured that contamination into its
    pre-R07 reference tree at commit 1 and had to allowlist it at Gate 3.

  Nothing here is a bug in the gate's logic — it is a scoping gap: the fixture
  has no owner and no provenance. Candidate fixes, cheapest first:
  **(a)** have `record`/`check` stamp the writing branch + commit into the
  manifest and warn when they disagree with `HEAD`; **(b)** make the fixture
  path branch-derived, so branches cannot collide; **(c)** treat it as
  disposable and regenerate per branch, accepting the runtime. (a) is probably
  enough, since the failure mode is silent misattribution rather than
  corruption. **Worth doing before the next milestone runs a gate.**

### Design debt accepted knowingly

- **[R7-4] ~~Import direction in the model-free producer.~~ FIXED 2026-07-29** —
  `climate_parity.py` moved `model/` → `shared/`, where it belongs: it imports
  only `typing`/`pandas`/`xarray`/`hydromt`, never touches a model object (the
  P3-2a C1 criterion its own docstring claims), and now has two callers on
  opposite sides — `model/plot_results.py` at model parity and
  `climate_analysis/plot_climate_source.py` on the source grid. It was misfiled,
  not miscoupled. `climate_analysis/` now imports nothing from
  `blueearth_cst.model`, pinned by a test that walks the package's ASTs so the
  convention cannot drift back silently.

- **[R7-5] O-24 is partially closed; its premise was wrong.** *Remaining half
  is row `t260802b`.* *Basin-average
  half FIXED 2026-08-01.* Rule 1.11 now derives `plot_basavg`'s PNGs from
  `wflow_outvars` and declares them — excluding `river discharge` (rule 1.05
  filters it out of the basin-average setup) and `precipitation` (whose column
  `plot_results` drops before plotting). Verified reaching
  `--delete-all-output`, including the fact that the derived filename carries
  the config's spelling **with spaces** (`actual evapotranspiration_basavg.png`)
  and that Snakemake handles it as a declared output and an explicit target.

  **The rest cannot be closed the way this entry assumed.** It claimed all
  three families were derivable "at parse time from `wflow_outvars` /
  `output_locations`". They are not: `hydro_{station}.png` and
  `clim_{station}_{period}.png` are counted by the model's OUTLETS and
  SUBCATCHMENTS — a rule-1.03 product read back through `Q_outlets` / the
  subcatchment map, unknown until the model is built, with `output_locations`
  contributing only the extra gauge stations on top. `signatures_{station}.png`
  is narrower still: it also needs observations AND a run longer than a year
  (`plot_results.do_signatures`), so it is data-conditional, not merely
  config-conditional. Closing those needs a `checkpoint` or a `directory()`
  output — a real rule-shape change, not the enumeration this entry imagined.
  Consequence, unchanged for those families: on a config with extra gauges or
  observations, `--delete-all-output` still cannot clean them and stale figures
  survive a rerun.

- **[R7-6] ~~Declaring `clim_wflow_1_*` made rule 1.11 newly able to fail.~~
  FIXED 2026-08-01** — the failure still happens, but no longer at rule 1.11 and
  no longer as `MissingOutputException`. Rule 1.11's ≥365-timestep requirement
  is subsumed by `snake_utils.MIN_HISTORICAL_YEARS` (16) — ONE floor for the
  whole toolbox (owner ruling 2026-08-01) — checked twice: against the
  REQUESTED window at WF1 parse time (`validate_historical_window`, so a short
  config reds the dry-run before any rule executes) and against the ACTUAL
  extracted span in the shared store producer
  (`extract_historical_climate._check_window_coverage`, which raises naming the
  requested window, the covered span and the floor). The owner ruling that
  failing loudly beats an incomplete figure set is preserved — only *where* and
  *how legibly* it fails changed. Original entry: those figures are written only
  when the extraction spans ≥365 days, so a config with a sub-year
  `historical_window` died with `MissingOutputException` instead of logging a
  skip.

- **[R7-7] The contract-equality test pins Snakemake 9.6.2's directive set.**
  **ACCEPTED — NO ACTION, 2026-08-02 (triage).** It asserts the compared /
  allowed-local / structural buckets partition `RuleInfo`'s fields exactly, so a
  Snakemake upgrade fails it loudly rather than silently widening the hole. That
  is the designed behaviour: the maintenance touchpoint at every version bump is
  the price of the loud failure, and paying it is cheaper than the silent hole.
  No row — the work it implies is "read the failure when you bump Snakemake."

### Cosmetic / low priority

- **[R7-8] ~~wflow writes `log.txt` beside the run TOML.~~ FIXED 2026-08-03 by
  R9 P2 commit 3.** `downscale_climate_forcing.py` now sets
  `logging.path_log = f"{out_prefix}{run_name}.log"` beside the other run-TOML
  pointers, so each member logs to its own file under
  `hydrology/wflow/output/`. It shipped **in the same commit as the `rlz_<n>/`
  flattening** and not after: flattening put every member's TOML in one shared
  `config/` directory, where the wflow default `"log.txt"` would have had all
  members writing one file concurrently. What was a cosmetic while each
  realization owned a directory became a correctness problem the moment they
  did not.
  Triaged 2026-08-02 as "fold into the next task that runs wf3 anyway", which is
  exactly what happened.
  **Still owed:** the concurrency falsifier has never been shown to FAIL with
  `path_log` unset. The cheap half — distinct pointers per member — is unit
  tested via `snake_utils.member_pointer_base`; the expensive half, content
  attribution under a real concurrent batch, still needs a run.
- **[R7-9] ~~Stale benchmark parts survive a rule rename.~~ CLOSED — NO ACTION,
  2026-07-29.** Investigated: `merge_benchmarks` deletes every part it merges
  (`_remove_parts`, called at the end of the merge), so a stale part from a
  renamed rule is consumed and removed on the **first** merge after the rename.
  The phantom row therefore appears exactly once, in one report, and the
  condition is self-healing. Adding a rule-name guard would mean teaching
  `merge_benchmarks` the current rule list, which it has no other reason to
  know. Not worth the coupling.
- **[R7-10] ~~Old-path references in documents commit 15 did not own.~~
  FIXED 2026-07-29.** `dev/reference/workflows/model_creation.md`'s `rule all` target list
  repointed to the B10 homes (and gains B4's three `source_*` figures, which it
  never listed); two notebook figure paths repointed.
  `dev/milestones/p32a/compare_climate_ladder.py` turned out to be a **live** probe, not a
  historical note — it opens `wf1_raw/`, which B1 retired, so it raises
  `FileNotFoundError`. Marked SUPERSEDED with the reason rather than repointed:
  the ladder existed to characterise the difference between `wf1_raw` and the
  keyed store, and R07's merge comparison proved those two element-wise
  identical, so re-pointing it would leave it comparing a store against itself.
- **[R7-11] ~~`plot_map_forcing.py:199` carries the same `"None"`-string
  shape as O-08.~~ CLOSED — NO ACTION, 2026-07-29.** Two independent reasons,
  both checked. The derived name is consumed by `if gauges_name in geoms:`
  (`plot_map_forcing.py:91`) — a *membership test*, which is itself the guard: a
  bogus `"None"` simply is not in `geoms` and nothing is drawn. And rule 1.13
  passes `{basin_dir}/staticgeoms/outlets.geojson`, a real declared path, never
  the sentinel, so the case cannot arise from the workflow at all. This is the
  structural difference from O-08, where `plot_map.py` *built a layer name and
  used it* with no membership check.
- **[R7-12] CLOSED — WORKING AS INTENDED, 2026-07-29. The tests config warns on
  every dry-run.**
  `tests/snake_config_model_test.yml` uses `project_dir: tests/test_project`,
  which is in-repo and outside the single `test_case/` exemption, so O-22's
  warning fires correctly but routinely. Not widened to silence it — the
  exemption exists because the *baseline seed* config is tracked, not as a
  general licence for in-repo scratch dirs.
- **[R7-13] ~~Map §2c's depth arithmetic is off by one.~~ FIXED 2026-07-29** —
  corrected against the emitted TOMLs, with the four verified pointer values
  tabulated and the two the original omitted (`state.path_output`,
  `output.csv.path`) added. Recorded as a correction rather than quietly
  amended: the map deferring to the comparator is precisely what kept the wrong
  arithmetic harmless, and that lesson is worth more than a tidy table.
- **[R7-14] `tests/test_stage_data_incremental.py` fails intermittently** *(row
  `t260802c`; still present and still flaky, confirmed 2026-08-02)* under
  some orderings; passes in isolation and on re-run. Another workstream's
  module, predates the R7 branch. Test-isolation issue, not a product defect.

### Parked by ruling — not defects

- **[R7-15] ~~Engine-named subtrees~~ DELIVERED by R9 P2.** Parked at R7's G1
  and explicitly deferred beyond R7; R9 is the milestone that took it. The tree
  is now `models/hydrology/wflow/` — domain, then engine — and the experiment
  carries the symmetric pair `experiments/<id>/{climate/weathergenr,hydrology/wflow}/`,
  so a second engine slots in beside the first at both scopes rather than
  needing a new root. That answers arch-8's structural half: a build subtree
  lives under `models/<domain>/<engine>/` and a run subtree under
  `experiments/<id>/<domain>/<engine>/`.
  **Not claimed:** that a second engine has been *tried*. The placement rule
  exists; nothing has exercised it. R7's narrowing from extensibility to
  **separability** (ruling GB-1) still describes what is actually proven.
- **[R7-16] Tooling contract** *(O-14 decision 2 + O-16 are row `t260802d`)*: O-14 `pyproject.toml`, O-15 `ruff`, O-16 `flit`
  — open decisions, unrelated to layout.
  **O-14 decision 1 RESOLVED** (ab781a5): tool-config-only `pyproject.toml`, no
  `[build-system]` / `[project]` / `[tool.pixi]`. Decision 2 (real packaging)
  still needs a superseding record in `dev/decisions/`.
  **O-15 RESOLVED** (85d3178 → 81e0096): ruff adopted as the lint gate,
  `select = ["E4","E7","E9","F"]`, all 96 findings cleared, and the PR
  template's unfounded "Black formatting pass" checkbox now names
  `pixi run ruff check .` — a command that exists. Two things worth carrying
  forward: ruff 0.16's *default* selection is ~415 rules (409 findings here
  under `--isolated`), so `select` is pinned explicitly and must stay pinned;
  and `ruff format` is configured but deliberately **not** enforced — see
  R7-23. **O-16 still open** and still gated on O-14 decision 2.
- **[R7-17] Docker (O-06) and Linux end-to-end (O-18, O-19)** — parked, no Linux
  machine. Linux *parse-level* consistency is now covered: the Linux config
  dry-runs on both CI legs.
- **[R7-18] Climate analysis as a fourth Snakefile** — a separate milestone. R7
  only ensured the layout does not obstruct it, and the model-free store plus
  rule 1.15 are the enabling pieces.

### Milestone housekeeping

- **[R7-19] ~~Branch unmerged, tag unapplied, roadmap stale.~~ RESOLVED
  2026-07-29.** Merged `--no-ff` (`0ea3918`), tagged `r07-layout`, and both
  pushed to `origin`. `dev/roadmap.md` Phase 4 now reads SEALED. CI green on
  both legs for the sealed tree (run 30450296441) — which was also the first
  CI run to see any of R7, since the milestone sat unpushed until the seal.
- **[R7-22]** *(row `t260802e`; re-confirmed 2026-08-02 — the bare reads and the
  `F821` per-file-ignore are both still in place)* **`downscale_climate_forcing.py` is the last module that reads the
  bare `snakemake` global at import time.** The other 22 `script:`-invoked
  modules use the guarded `if "snakemake" in globals(): sm = globals()["snakemake"]`
  idiom, which keeps them importable for unit tests; `prepare_weagen_config.py`'s
  docstring records that conversion as a deliberate past fix ("made it
  un-importable for unit tests"). Surfaced by ruff F821, and confirmed
  independently: a `pkgutil.walk_packages` sweep imports every module under
  `blueearth_cst/` cleanly **except** this one, which raises `NameError`.
  It currently carries an F821 per-file-ignore in `pyproject.toml` — that entry
  should be **deleted, not extended**, once the module is converted. Converting
  it would also make it unit-testable, which is the actual prize; note the whole
  module body sits inside a `with tee_to_log(...)` block, so the conversion is
  not purely mechanical.
- **[R7-23] `ruff format` is configured but not enforced.** *Row `t260802f`
  (blocked on an owner ruling).* **Re-measured 2026-08-02: now 136 of 276 files**
  would be reformatted (118 at R7) — the churn grows with every unformatted
  commit, so deferring has a running cost. Original entry: 118 of 262 files
  would be reformatted, ~7.8k diff lines. That is a churn decision on its own
  merits and was deliberately kept out of the O-15 lint adoption. If it is ever
  taken, it should be a single mechanical commit with no other change in it, so
  the diff stays reviewable — and note it would rewrite files the baseline gate
  reads, so re-record afterwards. Likewise, the rule families left out of
  `select` (`I` import sorting, `UP` pyupgrade, `B`/`SIM`/`PERF`/`RUF`) can each
  be added later as its own reviewable commit; `I` alone is ~62 files.
- **[R7-20]** *(row `t260802g`)* **Precondition met — R7 sealed 2026-07-29, so
  this is simply unexecuted; the tree is still on disk at 48 MB (checked
  2026-08-02).** The pre-R7 reference tree at
  `C:/Users/taner/workspace/.r07-reference/` (219 files + the discharge anchor)
  can be retired once the milestone seals — the re-recorded manifest is the
  regression detector again.

---

## Post-P3-3 (surfaced 2026-07-25 during the P3-3 batching milestone)

- **[2026-08-06] The item below may not need solving — see CR-7 / F18** in
  `dev/milestones/r09/wf3-change-requests.md`. Its *observation* is confirmed and
  strengthened: `B` keys off sweep size when only per-run cost should set it. But
  the same defect applies to the TIME economics, and more sharply — what batching
  amortizes is a fixed ~81 s per member, worth 70% of a run on the seed fixture
  and **2.2% at 1 h/run, 0.4% at 6 h/run**, which is the owner's stated production
  scale (1–6 h per run, 3–5 rlz × 25 cst). At K≈125 the default clamps to B=8,
  buying ~1.9% wall-clock for up to **seven completed runs** discarded on one
  failure — 42 h of compute at 6 h/run. **C35 proposes defaulting to B=1 with
  batching opt-in**, which removes the `B` from `p × B × (forcing + state)` and
  leaves no cap to compute. Rule on C35 before investing in the disk estimator
  below.

- **Make the wf3 batch-size default genuinely disk-aware.** Design §6.1 names
  three ceilings on `B` and calls the **disk ceiling the BINDING constraint** on
  large `RLZ_NUM×ST_NUM` runs, capped so `p × B × (forcing_size + state_size)`
  stays inside a stated headroom. The landed default implements only the
  *parallelism* ceiling (`ceil(K / -c N)`), which scales `B` **up** with sweep
  size and therefore grows peak temp disk as the sweep grows — backwards from
  what §6.1 asks. Commit `3392587` bounds it with an overridable
  `batch_size_max` (default 8); that caps the blast radius but is a constant, not
  a disk computation. A real cap needs (a) a stated disk-headroom config key and
  (b) a per-run forcing+state size estimate, and (b) is the hard part: at parse
  time the forcing NCs are `temp()` and do not exist yet, so the estimate has to
  come from the wflow grid dimensions × run length × variable count, or from a
  measured prior run recorded in config. Verified 2026-07-25: fixture (K=12,
  `-c 3`) is unaffected — `min(ceil(12/3), 8) = 4`, so every P3-3 measurement
  stands; the clamp only binds from K > 24 at `-c 3`. Confirm the hazard still
  applies before fixing (it is scale-dependent and invisible on the seed
  fixture, whose peak footprint is 120 MB).
- **Consider recovering per-cst persistence isolation under batching.** C5 is
  DEGRADED by design (blast radius `B`): one failing cst causes Snakemake to
  delete the `B−1` completed sibling CSVs and re-run the whole batch, and rule
  3.11 is blocked sweep-wide until it succeeds. Measured exactly as documented
  (`dev/milestones/p33/batching-results.md` GN-4). §6.1 names the mechanism worth probing:
  the `--keep-incomplete` ↔ `--keep-going` interaction (does `--keep-incomplete`
  preserve successfully-written sibling CSVs across a failed batch job, and does
  the sweep then re-run only the failed cst?), with **accept-the-degradation as
  the explicit fallback** if the probe fails. Only worth doing if the blast
  radius actually bites in practice.

---

## Post-R6 (surfaced 2026-07-23 during the R6 milestone validation)

- ~~**Make the projections summary CSV column order deterministic.**~~
  **CLOSED 2026-07-25 — code fix landed AND manifest re-recorded.** wf2 was run
  to completion (12/12 jobs, 140 s), the delta was proven column-order-only, and
  `record --workflow climate_projections` updated exactly the two affected rows.
  Evidence, in the order it was established:
  - **Delta is ordering, nothing else.** Both CSVs: identical column *set*,
    identical shape (48×9 and 6×9), header moved `temp,precip` →
    `precip,temp`, and **every value identical when matched by label**
    (`DataFrame.equals` after realigning the after-frame to the before-frame's
    column order). Checked before recording, because a value change would have
    meant `sorted()` did more than reorder — that would have blocked the record.
  - **Scope was exactly the 2 predicted rows.** `check` pre-record reported
    `FAIL - 2 target(s)`, both summary CSVs; the sibling
    `annual_change_scalar_stats_summary.nc` and the other five wf2 targets were
    unaffected.
  - **Manifest diff is 2 lines.** `git diff dev/baseline/manifest.json` = 2
    insertions / 2 deletions, both `sha256` values, `size_bytes` unchanged
    (a pure column swap preserves total width).
  - **Post-state:** full `check_baseline check` **OK 18/18**, and wf2 dry-runs
    to "Nothing to be done" — the queued-jobs tripwire left by the earlier
    killed run is gone.

  Historical detail retained below.

  **CODE FIX LANDED 2026-07-25; MANIFEST RE-RECORD (now done, see above).**
  `intersection()` in `blueearth_cst/projections/get_change_climate_proj.py`
  returned `list(set(lst1) & set(lst2))` — hash-order dependent, so
  `annual_change_scalar_stats_summary{,_mean}.csv` flipped `precip`/`temp`
  column order run-to-run (values identical by label, sibling `.nc` unaffected,
  consumers read by name). Now `sorted(...)`, with 7 regression tests including
  a sub-process `PYTHONHASHSEED` sweep — the only form that actually catches
  hash-order dependence, since the seed is fixed for an interpreter's life.
  Verified: all 7 fail on the pre-fix line, pass on the fixed one.

  **Chose alphabetical over "preserve the first sequence's order"** so the
  guarantee is self-contained rather than silently dependent on upstream
  dataset-variable ordering. On the seed config the two coincide
  (`variables: [precip, temp]`).

  **Outstanding — the baseline gate the R6 note anticipated.** The recorded
  fixture CSVs carry the columns as `…,temp,precip,…`, which is *neither*
  candidate deterministic order, so the next wf2 run will emit
  `…,precip,temp,…` and the **2 manifested CSV rows will mismatch**
  (`check_baseline` fingerprints them by sha256 of normalized bytes —
  `check_baseline.py` `fingerprint_csv`). `check` is green **18/18 today** only
  because a code change cannot move on-disk outputs. So this fix has planted a
  guaranteed, expected failure that will look like a regression to whoever next
  runs wf2. To close: re-run wf2 (`~1172 s` summed job time, so roughly 3–6 min
  wall at `-c 3` on a quiet box — and that figure was measured during the
  contaminated window, so likely faster), confirm the delta is column order
  only, then `check_baseline.py record --workflow climate_projections`. That is
  a deliberate manifest update per the roadmap's "no silent updates" rule.

  **STATE AS LEFT 2026-07-25 — read before the next wf2 run.** A forced wf2
  re-run (`--forcerun monthly_change monthly_change_scalar_merge`, 22 jobs) was
  started and **killed at 10/22**, deliberately not retried. Consequences:
  - The 2 summary CSVs are **still the original bytes** (`…,temp,precip,…`) —
    rule 2.05 never ran. The manifest was **not** touched; `check_baseline`
    reports **OK 18/18**.
  - The workdir was left locked and has since been **unlocked** (no locks
    remain).
  - **A partial regeneration is now QUEUED.** wf2 went from "Nothing to be
    done" to **12 pending jobs** (3 `monthly_stats_fut`, 3 `monthly_change`,
    `monthly_change_scalar_merge`, `plot_climate_proj_timeseries`, the log/
    benchmark gathers, `all`), because 6 of the 9 stats jobs completed and their
    `temp()` intermediates were reclaimed. So **the next wf2 invocation — any
    invocation, not just a forced one — will complete the merge and rewrite the
    two CSVs in the new `…,precip,temp,…` order**, at which point those 2
    manifest rows go red until re-recorded. That is expected, not a regression.
  - Cost note: wf2's 2.02/2.03 read CMIP6 **over the network** from
    `gs://cmip6/...`, so its wall time is bandwidth-bound, not CPU-bound. The
    killed run managed 10/22 jobs in ~3.5 min. An earlier "3-6 min from the
    1172 s summed benchmark" estimate was wrong in kind for that reason.
  - Snakemake may also require `--rerun-incomplete` if it flags any output left
    half-written by the kill.

- **Snakemake's `code` rerun-trigger does NOT reach wf2's rule 2.04.**
  Discovered 2026-07-25 while trying to propagate the `intersection()` fix.
  Rule 2.04 `monthly_change` names `get_change_climate_proj.py` directly as its
  `script:`, so an edit to it should have re-run the rule — it did not, even
  under an explicit `--rerun-triggers code`. Cause is structural: 2.04's output
  is `temp()` (already reclaimed) and rule 2.05's inputs are wrapped in
  `ancient(...)`, which tells Snakemake to ignore their timestamps; once 2.05's
  outputs exist the whole 2.04 layer leaves the DAG, so there is no job whose
  code hash could be compared. Same family as the P3-3 finding that
  `--forcerun generate_weather_realization` does not cascade the wf3 sweep.
  **Practical rule: after fixing computational code in this repo, `--dry-run`
  first to confirm the affected rules are actually in the DAG, and reach for an
  explicit `--forcerun <rule>` rather than trusting the code trigger.** Worth
  considering whether the `ancient()` wrappers on 2.05's inputs are still
  earning their keep, or whether they are over-broad insurance that now hides
  real staleness.
- ~~**`semantic_tree_diff.py` exclusion refinement.**~~ **CLOSED 2026-07-25 —
  already fixed, no action taken.** The cited defect (stray `.log`/`.txt` under
  `hydrology_model/` reaching the hash comparator as benign FAILs) was resolved
  by P3-1 commit `576b6a6` ("exclude run-log files (5b)"), which added a
  file-level rule to `_is_excluded`: `rel.suffix == ".log" or rel.name ==
  "log.txt"`. Verified 2026-07-25 by calling `_is_excluded` directly on the
  three exact paths — `hydrology_model/hydromt.log`,
  `hydrology_model/run_default/log.txt`,
  `experiments/experiment/model_runs/log.txt` — all three EXCLUDED. The only
  other non-standard-extension file in the tree,
  `hydrology_model/staticgeoms/reservoirs_lakes_glaciers.txt`, is correctly
  hash-compared: it is content-bearing and deterministic, and passed as one of
  the 102 CLEAN files in the P3-3 value-identity gate. **The residual
  suggestion (a generic extension-level volatile class / per-tree exclude
  globs) is deliberately NOT implemented:** this is a *gate* tool, and widening
  its exclusions without a demonstrated false FAIL trades real detection for
  nothing. Reopen only with a concrete benign-FAIL case.
- ~~**Dead-fixture audit: `tests/wflow_build_model.yml`.**~~ **CLOSED 2026-07-25
  — confirmed dead, removed.** Evidence: (1) no config points at it — every
  `model_build_config:` in the repo, **including `tests/snake_config_model_test.yml`
  and `tests/test_project/`**, resolves to `config/templates/wflow_build_model.yml`
  in the shared config tree (the R6 review already established this as finding
  arch-1); (2) no test loads it by name and nothing globs `tests/*.yml`; (3) it
  was itself **broken** — its `read_config.config_fn: "../config/wflow_sbm.toml"`
  dangles, since R6 moved that file to `config/templates/wflow_sbm.toml` and the
  fixture was never updated, so anything that *did* load it would fail; (4) last
  touched in m02b (`95c4163`), predating the R6 config split. Removed rather than
  wired up: a second build template would be a duplicate maintenance surface with
  no consumer. Full suite green after removal.
- ~~**`scripts/run_snake_test.cmd` modernization.**~~ **CLOSED 2026-07-25 —
  ported, not retired.** `scripts/` is a documented user-facing entry point
  (`AGENTS.md` Repo Map), so the default was to preserve the surface and fix the
  hostility rather than delete it. Changes: every call goes through `pixi run`
  (drops `call activate cst`, and incidentally fixes the graphviz complaint —
  `dot` resolves from the pixi env, verified graphviz 14.1.2, all three DAGs
  render); `pause` removed; stops on the first failing workflow with a nonzero
  exit, matching `run_workflows.py`'s contract; arguments forward to every
  `snakemake all` call, so `scripts\run_snake_test.cmd --dry-run` validates the
  whole script in seconds; DAG renders moved out of the repo root into a
  gitignored `dag/`, and the render step is best-effort so a graphviz failure
  cannot abort a run. Verified: `--dry-run` exits 0 across all three workflows
  in the required order; a bogus flag exits 1 after workflow 1 and never reaches
  2 or 3. Two cmd.exe traps hit and documented in-file while porting — an
  unescaped `)` inside a parenthesised `if/else` block, and `shift` **not**
  rebasing `%*` (forwarded args are captured once into `%FWD%` instead).
  A stale `dag_model.png` from the old script may still sit in the repo root;
  it is gitignored and safe to delete by hand.

---

## Cross-cutting — baseline manifest integrity

- **[RESOLVED 2026-07-18] Baseline rebuilt from a tracked seed config.**
  `dev/baseline/manifest.json` was re-recorded from the now-tracked
  `config/snake_config_model_test.yml` (project_dir `examples/test_local`,
  3 models, single `far` horizon, current libraries), after a fresh run of
  all three workflows. `record` → `check` round-trips clean (14/14). The
  untracked `snake_config_model_test_local.yml` that seeded the stale M2b
  baseline is retired, so the divergence cannot recur. The model-independent
  workflow-1 PNG drift noted below was not separately investigated — it is
  moot now that the whole baseline is re-recorded from a known, tracked
  config; revisit only if a future `check` shows unexplained PNG drift.
  Original diagnosis retained below for provenance.

- **Rebuild `dev/baseline/manifest.json` against current libraries with a
  tracked seed config.** *Surfaced 2026-07-18 during R01 Task 5.* The M2b
  manifest (last recorded 2026-05-08, commit `159e197`) was recorded from
  an **untracked, 3-model local config**, while the tracked canonical
  (`config/snake_config_model_test.yml`) has used an **8-model** list since
  before `pre-r01`. The two have been silently divergent since M2b — the
  workflow smoke tests never compare against the manifest, so nobody caught
  it. R01 Task 5 was the first `check_baseline check` against the canonical
  model set and exposed the mismatch (see `dev/milestones/r01/baseline_diffs.md`).

  A fresh 8-model canonical run also shows **model-independent** drift on
  workflow-1 plots (`basin_area/hydro_wflow_1/precip.png`, 19–32% size) —
  workflow 1 never reads the climate model list, so the M2b local config
  must have differed from the canonical in ways beyond model count, and
  **cannot be reconstructed**.

  *Fix:* choose a deliberate model set, commit a **tracked** seed config
  (or parameterize `check_baseline` with one) so the baseline is
  reproducible, run all three workflows on current libraries, and record a
  fresh manifest. Investigate whether the workflow-1 PNG drift is
  rendering-only (matplotlib) or real content before blessing it. Until
  then the M2b manifest remains the contract of record, with R01 sealed on
  invariance-by-construction rather than a re-record.

---

## Cross-cutting — workflow ergonomics

- **[FIXED 2026-08-01] The user's gauges were dropped in silence, everywhere.**
  *Found on a real basin run (`C:/TESTS/CST/gabon_0108`), reported by Ümit as
  "output locations missing from the spatial plots".* hydromt_wflow's
  `setup_gauges` normalizes the basename — `.replace("_", "-")`,
  `wflow_base.py` — so `output_locations.csv` becomes `output-locations` in the
  staticgeoms layer, the wflow TOML `map`, and the parsed output columns. Three
  of our readers derived that name from the FILENAME and missed the
  substitution; every lookup was a membership test used as a guard
  (`if name in geoms:`), so all three failed **silently**. Damage: gauges absent
  from `basin_area.png`, no gauge hydrographs, no signature plots, and an EMPTY
  `performance_metrics.csv` on a config that supplied observations — while
  `output.csv` held all four stations correctly, because wflow reads the TOML
  instead of guessing. Rule 1.13 had a second, independent instance: it passed
  `staticgeoms/outlets.geojson` as its gauges param, so a configured
  `output_locations` was never even attempted there.
  *Fixed* by `blueearth_cst/shared/gauges.py`: resolve the layer and variable
  from the MODEL, and WARN (never skip) when a configured file cannot be
  resolved. **This is the second instance of the class** — R07 O-08 was the
  same shape with a different cause (the `"None"` sentinel). Both times a
  membership test doubled as a guard. Worth a sweep: any `if <derived name> in
  <mapping>:` where the name comes from config is a silent-drop candidate.

- **`tee_to_log` does not capture the traceback of a failing `script:` rule.**
  *Surfaced 2026-08-01 while landing the canonical climate figure set.* A rule
  that raises writes every `log_row`/INFO line to
  `logs/_parts/<W.NN>_<rule>.log` and then stops **without the exception**. The
  merged workflow log therefore ends mid-rule with no reason, and
  `check log file(s) for error details` — which is the only thing Snakemake
  prints — points at a file that does not contain them. The traceback does reach
  Snakemake's own captured stderr, so it is visible on an interactive console
  run and invisible in the artifact a user would send you. Cost a full
  reproduce-outside-pytest cycle to recover a one-line `KeyError`.
  *Fix:* have `tee_to_log` catch, write the formatted traceback into the log
  part, and re-raise. Cheap and self-contained (`snake_utils.tee_to_log`), and
  it improves every `script:` rule in all three workflows at once. Owner:
  `python-engineer`. Activation: next time WF logging is touched.

- **[PARKED 2026-07-19] Per-rule progress messages.** Add `message:`
  directives to the long-running rules so each announces itself in plain
  language when it starts (e.g. "Building Wflow model from global data…"),
  layered on top of Snakemake's built-in `N of M steps (X%) done` counter and
  the per-rule timestamps. Snakemake cannot show progress *inside* an external
  step (hydromt build, Julia) — only start/end — but the tool's own streamed
  output (now visible via `tee`) covers the in-between. Cross-cutting: apply
  across all three `Snakefile_*` as a consistent pattern; R4/R5 would inherit
  it. Per-rule wall-clock is already captured by the `benchmark:` TSVs added in
  R3. Deferred by choice, not a blocker — pick up when convenient (a natural
  fit alongside R4/R5 Snakefile work or R6 polish).

- **[RESOLVED 2026-07-21, commit `d13ba37` (t260721a, `fix/pre-r6-followups`).]**
  wf1's three shell rules now route through `src/run_logged.py` (a CLI over
  `snake_utils.run_and_tee`), a portable Python tee wrapper that keeps live
  console output, writes the log, and exits with the child's own return code.
  Verified end-to-end: a deliberately-failing child propagates its non-zero code
  (the old `| tee` masked it to 0 under cmd.exe). Original diagnosis retained
  below for provenance.

- **[Latent robustness, not a blocker] wf1's `| tee {log}` shell rules mask the
  exit code on failure.** *Surfaced 2026-07-20 during R5 (design §2 ruling).*
  `Snakefile_model_creation`'s three shell rules (lines 89, 167, 182 — `hydromt
  build`, `hydromt update`, Julia `Wflow.run()`) use `... 2>&1 | tee {log}`.
  A bare `cmd | tee` pipeline returns `tee`'s exit status, not `cmd`'s, unless
  bash `pipefail` is active. On **Windows/cmd.exe** Snakemake injects **no**
  `set -euo pipefail` prefix (that prefix is bash-only — verified against
  Snakemake 9.6.2 `shell.py`), so a genuine `cmd` failure is reported as
  success. Verified empirically 2026-07-20 in a scratch Snakefile: a
  deliberately-failing command under `| tee` → Snakemake reports success;
  under `> {log} 2>&1` → Snakemake fails ("command exited with non-zero exit
  code"). On POSIX/bash the `pipefail` prefix protects, and on **success** the
  wf1 `| tee` rules run correctly (R3 sealed via a full `--forceall` wf1 rebuild
  that wrote all three tee logs and passed 14/14 on this machine) — so this is a
  **latent** failure-masking gap that only bites if a wf1 rule actually fails
  mid-run, **not** a gate blocker.

  *Fix:* migrate wf1's three shell rules to the exit-preserving `> {log} 2>&1`
  form R5 adopted for workflow 3's shell rules, **or** adopt a portable Python
  tee wrapper repo-wide if live console streaming must return (the tee form was
  deliberately chosen in commit `4a67d79` to restore live output). Owner:
  `cst-architect`. Activation: **next time wf1 shell-rule robustness is worked
  on.** wf3's own new shell rules already use `> {log} 2>&1` (R5 commit 8), so
  R5 introduces no new instance of the masking.

---

## R3 — Workflow 1: model builder

- ~~**Resolve test_cli xfails.**~~ **CLOSED 2026-07-25 — both resolved, each by
  one of the options this entry proposed.** Verified on the current tree:
  `tests/test_cli.py` contains **no `xfail` marker at all**; all three cases
  assert `returncode == 0` and the file runs **4 passed**. The
  `MissingInputException` case was fixed the fixture way — `test_snakefile_cli_
  climate_projections` takes the `config_with_staged_region` fixture (R3), so
  `region.geojson` is pre-staged rather than the Snakefile being refactored. The
  `CyclicGraphException` case was fixed the `wildcard_constraints` way in R5, and
  the fix is self-documenting at `Snakefile_climate_experiment:297` — a
  rule-local `st_num=r"[1-9][0-9]*"` constraint whose comment names this exact
  exception and explains why `cst_0` must not be resolvable by rule 3.07. No
  `ruleorder` was needed. Original entry retained below for provenance.

  Two of the three parametrizations in
  `tests/test_cli.py` are marked `xfail` since M2:
  - `Snakefile_climate_projections`: dry-run trips
    `MissingInputException` because the workflow expects
    `staticgeoms/region.geojson` (produced by Snakefile_model_creation)
    even when only dry-running. Either change the test fixture to
    pre-stage that file, or refactor `Snakefile_climate_projections` so
    `--dry-run` doesn't require it.
  - `Snakefile_climate_experiment`: dry-run trips
    `CyclicGraphException` at `rule generate_climate_stress_test`. The
    rule's wildcard pattern `rlz_{rlz_num}_cst_{st_num}.nc` overlaps
    with `generate_weather_realization`'s output `rlz_{rlz_num}_cst_0.nc`.
    Production configs (`config/snake_config_model_test_local.yml`) work
    fine because Snakemake disambiguates from concrete paths in
    `expand(...)`, but the `--dry-run` resolver flags the cycle on the
    test config. Add a wildcard constraint (`{st_num,[1-9][0-9]*}` or
    similar) or a `ruleorder:` directive.

  These are pre-M2 failures masked by the fact that M1 closure didn't
  actually run pytest.

  *Split 2026-07-19 (`dev/milestones/r03/model-builder-design.md` §2), by where the fix
  lives:* the `MissingInputException` is a workflow-2 **test-fixture** defect
  (dry-run against an empty project dir) — **fixed in R3** by pre-staging a
  minimal valid `region.geojson` and flipping that ratchet. The
  `CyclicGraphException` fix is a `wildcard_constraints`/`ruleorder` edit
  **inside `Snakefile_climate_experiment`** (R5 territory, entangled with the
  `st_num2 → st_num` fold that `dev/reference/naming.md` §4 already assigns
  to R5) — **deferred to R5**; the ratchet is retained until then.

- **[RESOLVED 2026-07-21, t260716a′ (`fix/pre-r6-followups`).] Redo M1 warnings
  triage exhaustively.** Swept 82 captured `.log` files across all three workflows
  (per-rule `log:` directives now present via R3/R4/R5). **Bucket 3 (our-code):
  empty** — no warnings framed in `src/`, the Snakefiles, `dev/scripts/`, or the R
  layer. **Bucket 2:** one item, intended hydromt behavior (the `0.00833` vs native
  `0.008333333333325754` resolution snap) — won't-fix (a config match is fragile +
  would drift the tracked snake-config fingerprint for zero model change).
  **Bucket 1:** hydromt CRS/forcing/model-dir warnings + a new-but-captured 62×
  `Error in sys.excepthook` shutdown cascade from `hydromt build -vv` (post-success;
  upstream subprocess, not our tee wrapper — absent from the Julia/hydromt-update
  logs that use the same wrapper). No code changes. Full re-triage recorded in
  `dev/milestones/phase-1/m01/warnings.md` § "Exhaustive re-triage — 2026-07-21".

- **~~`extract_climate_grid` silently truncates the historical range.~~
  CLOSED 2026-08-01.** Both halves are now resolved. *Truncation WARNING*
  resolved 2026-07-21, commit `ce56bc3` (t260716a, `fix/pre-r6-followups`):
  `prep_historical_climate` emits an advisory when the extracted span falls
  short of the requested window. *Config staleness* resolved 2026-07-21
  (t260716a′, see the nested entry below — R5 wired the window in as `params`,
  so Snakemake's default rerun-trigger re-extracts on an edit). *The
  "optionally, fail the rule" half of the fix below* landed 2026-08-01, as a
  UNIFIED floor rather than a per-workflow one (owner ruling, same day):
  `extract_historical_climate._check_window_coverage` keeps shortfall-vs-
  requested advisory with its 31-day tolerance, and RAISES below
  `snake_utils.MIN_HISTORICAL_YEARS` (16). A WF1 parse-time guard
  (`validate_historical_window`) applies the same floor to the requested window
  before any rule runs. The floor is set by the most demanding consumer
  (weathergenr's wavelet minimum) and enforced in WF1 and WF2 too: a first
  revision enforced 365 days hard and 16 years advisory, which let WF1 build a
  model on a record WF3 would reject — moving the failure to the workflow least
  able to explain it. **Consequence:** two shipped configs held windows under
  the new floor and were widened to 2000–2020 — `snake_config.template.yml` (was
  6 years) and `tests/snake_config_model_test.yml` (was 6). **Still open, and
  tracked separately:** the 16-year gate in WF3 itself, where weathergenr is
  invoked — a store built before this change can still reach it. See
  "weathergenr's wavelet minimum surfaces as a cryptic error" in the R5
  section.*
  When the snake config's `historical:` window asks for years that the
  staged source doesn't cover, the rule produces a shorter `extract_historical.nc`
  without any warning. Downstream rules then fail in cryptic ways far from
  the actual cause.

  *Observed 2026-05-07:* config asked `historical: 1980, 2010` (31 years),
  the staged era5 only had data from 2000-01-01 onward, so the extracted
  netCDF held 2000–2010 = **11 years**. That fell below weathergenr's
  16-year wavelet minimum and crashed `generate_weather_realization`
  with `'series' must have at least 16 observations`.

  *Fix:* in `extract_climate_grid` (or its underlying script), log a warning
  when the extracted time span is shorter than the requested span. Optionally,
  fail the rule if the shortfall is large enough to break a downstream step
  (e.g. < 16 historical years when weathergenr is in the pipeline).

  *Related Snakemake-staleness issue:* **[RESOLVED 2026-07-21, t260716a′ — by R5's
  wiring + verification, no new code.]** The 2026-05-07 repro (changing `historical:`
  didn't re-extract) predates R5, when the dates were hardcoded and `historical:`
  was **never read** by `extract_climate_grid` — so of course the edit had no effect.
  R5 wired `shared.historical_window` into the rule as `params`
  (`starttime`/`endtime`, `Snakefile_climate_experiment:78-82`), and Snakemake 9.6.2
  applies its default `params` rerun-trigger (no `--rerun-triggers` override in the
  repo). **Verified empirically 2026-07-21:** a dry-run against the built
  `examples/test_local` with `historical_window.endtime` changed 2020→2019 schedules
  `extract_climate_grid` with reason *"Params have changed since last execution:
  before '2020-12-31…' now '2019-12-31…'"*. So config edits to the historical window
  now propagate automatically; `--forcerun` is no longer needed. Declaring the whole
  config as an input (the original suggestion) is unnecessary and coarser (would
  re-run on any unrelated config edit). The broader "audit every rule whose behavior
  depends on an unread config key" remains a general note (see R5 section below), not
  part of this item.

  *Workaround applied 2026-05-07:* `historical: 2000, 2020` in the local test
  config + `--forcerun extract_climate_grid` for the immediate run. Treats
  the symptom; doesn't fix either of the two underlying issues.

---

## R5 — Workflow 3: climate experiment

- **`extract_climate_grid` ignores the `historical:` config and hardcodes
  the date range.** Pre-R5 unblocking edit on 2026-05-07 changed the
  hardcoded `starttime="2000-01-01"` / `endtime="2020-12-31"` in
  `src/extract_historical_climate.py`. The snake config's `historical:`
  key is read by `Snakefile_climate_projections` (workflow 2) but never
  by `Snakefile_climate_experiment` (workflow 3) — the rule
  `extract_climate_grid` only receives `data_sources` and `clim_source`
  as params; `historical:` is silently ignored.

  *Proper fix:* in `Snakefile_climate_experiment`, parse `historical:`
  from the config and pass `starttime` / `endtime` as rule params; in
  `src/extract_historical_climate.py`, replace the hardcoded date
  strings with `sm.params.starttime` / `sm.params.endtime`. While at it,
  drop the misleading function defaults at lines 20–21 (currently still
  say `1980` / `2010`).

  *Also touches the R3 followup:* this is the same shape as the
  config-key-not-wired pattern — fixing it should be paired with a
  general audit of every rule whose behavior depends on a config key
  that isn't actually read.

- **`weathergenr::write_netcdf` does not propagate `spatial_ref` attributes
  from `template_path` to the output.** Confirmed 2026-05-07: the historical
  template (`extract_historical.nc`) has `x_dim='longitude'` and
  `y_dim='latitude'` on its `spatial_ref` variable, but the realization
  files written by `write_netcdf` (`rlz_*_cst_0.nc`) have an *empty*
  attribute list on their `spatial_ref` variable. Downstream
  (`impose_climate_change.R`) then crashes when it uses the realization
  as its own template, because `write_netcdf`'s `x_dim` lookup returns
  `0` (numeric, from `ncatt_get` on a missing attr) — which slips past
  the existence check and causes
  `Error in nc_in$dim[[x_dim_name]] : attempt to select less than one element`.

  *Workaround applied 2026-05-07:* in `src/weathergen/generate_weather.R`,
  after each `write_netcdf` call, manually copy `spatial_ref` attributes
  from the historical input file to the just-written realization file
  via `ncdf4::ncatt_get` / `ncatt_put`. Marked clearly so it can be
  removed when weathergenr is fixed.

  *Proper fix:* in `tanerumit/weathergenr` `R/io_netcdf.R`, the
  attribute-copy loop in `write_netcdf` looks correct on the surface
  (`ncatt_get(nc_in, spatial_ref)` → `ncatt_put`) but evidently isn't
  executing or isn't writing through. Investigate why the loop produces
  zero attributes on the output. Separately, the missing-attribute check
  should also assert `hasatt = TRUE` on the `ncatt_get` result, not just
  test the value for NA / NULL — the current check accepts the numeric
  `0` returned for a missing attribute and crashes one line later.

- **weathergenr's wavelet minimum surfaces as a cryptic error.**
  `wavelet_cwt.R` enforces `length(series) >= 16` on the *annual* aggregate
  (i.e. ≥ 16 historical years), but the user-facing error is just
  `'series' must have at least 16 observations` — no mention of years,
  wavelet, or how to remedy.

  *Fix:* improve the error in `tanerumit/weathergenr` (upstream of this repo).
  Suggested message: *"historical period (N years) is below weathergenr's
  wavelet minimum of 16 years; extend the historical range or reduce the
  wavelet decomposition depth."*

  *Note:* this fix lives in the weathergenr package, not this repo. Mention
  in R5 deliverables if R5 is also touching the R layer; otherwise track as
  a separate weathergenr issue.

---

## R3+ — Surfaced during M02c (test coverage)

Lessons learned writing the M02c unit tests. Not bugs — testing-discipline
notes for R3-R5 to inherit when they add their own test files.

- **Test pollution between `sys.modules.setdefault` files.** pytest collects
  test files in alphabetical order. The first file to call
  `sys.modules.setdefault("hydromt", <stub>)` (or any heavy dep) wins, and
  later files using `setdefault` for the same key get a silent no-op —
  their import of the source module then binds to the *previous* test
  file's stub. Symptom: tests pass when run in isolation, fail in the full
  suite with `KeyError` on fixture-set catalog data.

  *Pattern:* don't rely on `setdefault` alone for shared keys. Use
  `monkeypatch.setattr(<source_module>.<dep>, "<attr>", <fake>)` inside
  fixtures so each test gets a clean override regardless of collection
  order. See `tests/test_prepare_climate_data_catalog.py` for the
  reference implementation; commit `f65244e` for the diagnosis.

- **dask cannot be stubbed at module level.** pandas does a lazy
  `import dask` and accesses `dask.__spec__` during type compatibility
  checks. A `types.SimpleNamespace` stub for dask there raises
  `ValueError: dask.__spec__ is not set` during collection of *any* test
  file that imports pandas. dask is in the env via pixi; let it import
  normally. If the cost matters, mock the specific dask object at call
  time within the test, not at module level.

---

## R3+ — Surfaced during M2b (library upgrades)

Items surfaced during the hydromt 0.x → 1.x / hydromt_wflow 0.x → 1.x /
Wflow.jl 0.7 → 1.0.2 / pandas 3.x / Python 3.12 jump. See `dev/milestones/phase-1/m02b/`
for the full M2b record.

- **`hydromt 1.x` `to_dict` / `to_yml` silently strips `driver.options.preprocess`.**
  Round-tripping a catalog dict through `DataCatalog().from_dict(...).to_yml(path)`
  loses the preprocess hook even though `from_dict` preserves it on read.
  *Workaround applied:* `src/prepare_climate_data_catalog.py` bypasses
  `to_yml` and uses `yaml.safe_dump` directly.
  *Proper fix:* file upstream against `hydromt`. Reproducer is the
  three-line snippet in `dev/milestones/phase-1/m02b/handoff.md` decision section.

- **conda-forge does not ship `julia` for win-64 at all.** linux-64 / osx-64
  have 1.10.x and 1.12.x but skip 1.11.x; win-64 has nothing. This blocks the
  "single env via pixi" goal on Windows for the Julia layer. Today juliaup
  manages Julia 1.11.7 outside pixi.
  *Possible fixes:* (a) wait for conda-forge to ship win-64 Julia; (b) wrap
  juliaup in a pixi `[tasks]` step that calls `juliaup install 1.11.7` at
  env-setup time; (c) move to a different distribution channel.

- **[RESOLVED 2026-07-17] weathergenr crashed loading on Windows —
  root cause was conda-forge's r45 `r-waveslim` build, not `ncdf4`.**
  `library(weathergenr)` (and the install's lazy-load step) died with
  `Mingw-w64 runtime failure: 32 bit pseudo relocation ... out of range`.
  Isolated by loading each Import in turn: the first 15 loaded fine and
  only `waveslim` overflowed — its Fortran DLL carries a 32-bit
  pseudo-relocation to libgfortran that lands ~2.7 GB away (past the
  signed 2 GB range), so load order can't dodge it. The earlier "likely
  ncdf4" and "user lib on Windows" notes were both wrong: under pixi
  `Rscript --vanilla` the only libPath is the conda site-lib, and ncdf4
  is fine. The bug is specific to the **r45** (R 4.5) waveslim build; the
  **r44** build loads and runs `modwt` cleanly.
  *Fix applied:* pin `r-base = "4.4.*"` in `pixi.toml` so the solver picks
  the working r44 waveslim (and r44 builds of the other Fortran deps).
  Also switched `install_weathergenr.R` from `pak` (conda-forge `r-pak` is
  separately broken on win-64 — "Wrong OS or architecture") to
  `remotes::install_github(dependencies=FALSE, upgrade="never")`, which
  touches nothing but weathergenr itself. Verified: `pixi run install-rdeps`
  installs and `library(weathergenr)` loads.
  *Revisit when:* conda-forge ships a fixed r45 `r-waveslim` (or R 4.6)
  Fortran build — then the `r-base` pin can move forward again.

- **`setup_constant_pars` short names → CSDMS Standard Names.**
  *Re-tagged 2026-07-19: standalone scientific-review task `t260719a`, split
  out of R3.* hydromt_wflow 1.x's `setup_constant_pars` rejects the short
  parameter names from 0.x and requires CSDMS Standard Names instead. M2b
  dropped **14 of the 15** originally-set constants under the "intentional
  drift, re-baseline aggressively" policy and kept only `KsatHorFrac` (which
  the build errors without).

  **Authoritative inventory (the handoff prose miscounts — its explicit
  parenthesized list of 15 names controls, not its "14 constant pars / other
  13" prose):** 15 original / 1 retained (`KsatHorFrac`) / **14 dropped**,
  where the 14 dropped = **8 known CSDMS mappings** (`Cfmax`, `WHC`, `TT`,
  `TTI`, `TTM`, `G_Cfmax`, `MaxLeakage`, `InfiltCapPath`) + **`InfiltCapSoil`**
  (deprecated, `wflow_v1: None` in `hydromt_wflow.naming` → stays dropped) +
  **5 unresolved** (`cf_soil`, `EoverR`, `rootdistpar`, `G_SIfrac`, `G_TT`)
  whose CSDMS mapping or deprecation status is not yet confirmed.

  Restoring physics parameters is a scientific decision, not a mechanical
  rename, so it is **out of R3** (which is a behavior-preserving refactor,
  per `dev/milestones/r03/model-builder-design.md`). The dedicated task owns
  `config/wflow_build_model.yml` and the resulting baseline move. Its scope
  must carry: a **parameter-reconciliation table** (per param: old value,
  Wflow 1.x effective default, units, semantics, storage location, observed
  built value, and a restore / adopt-new-default / drop-deprecated
  classification); a **direct staticmaps.nc/TOML assertion** that each
  restored value actually lands (a name accepted but silently no-op'd is the
  failure mode); a **data-level workflow-1 discharge comparison** (not PNG
  size — the manifest does not fingerprint discharge); and a **clean
  dedicated project-dir re-record** with a freshness check on every recorded
  target (existence-based Snakemake timestamps + `ancient()` inputs can bless
  stale artifacts). The task should also ADD staticmaps/TOML fingerprints to
  the manifest, since workflow 1's slice is currently only 3 size-only PNGs +
  a snake-config snapshot. CSDMS lookup tables in `hydromt_wflow.naming` and
  `hydromt_wflow.version_upgrade`. Concrete 8-mapping remap in
  `dev/milestones/phase-1/m02b/handoff.md` decision #3.

- **[RESOLVED 2026-07-21, t260720e — does-not-reproduce, no fix.] CMIP6 `precip` /
  `temp` `.attrs` lost on `monthly_change_scalar_merge`.** Under the current pinned
  env (hydromt 1.3.1) the merged `annual_change_scalar_stats_summary.nc` carries the
  **full CF set** (`cell_measures`, `cell_methods`, `comment`, `long_name`,
  `original_name`, `standard_name`, `units`) on both `precip` and `temp` — verified
  on the real-CMIP6-read output in `examples/test_local` AND in the recorded manifest
  fingerprint (`check --workflow climate_projections` passes on the `.nc`). R4's
  `probe_attrs_chain.py` proved no wf2 code drops attrs, and the values are
  CMIP6-native, so the hydromt read preserves them. The M2b `{}` diagnosis no longer
  reproduces; original root cause not re-litigated (moot). Absorbed the old t260716c
  "CMIP6 attr loss on merge" item. Full disposition: `dev/milestones/r04/chain-audit.md`
  § D-ATTRS.

- **Outlet station naming convention decision.** hydromt_wflow 1.x's
  `setup_outlets` uses subcatchment IDs (e.g. `130000086`, `1`, `2`, …) for
  outlet stations rather than the contiguous `1..N` of 0.x. The CSV column
  also renamed `Q_gauges` → `Q_outlets`. M2b's `src/plot_results.py`
  rebuilds `station_name` as `1..N` to keep `hydro_wflow_1.png` visually
  stable; R3 should pick a consistent project-wide convention (real
  subcatchment IDs vs `1..N` rebuild) and document it.

- ~~**Retire the "CMIP6 GCS throughput regression" follow-up.**~~ **CLOSED
  2026-07-25 — retired, and independently re-confirmed.** The original
  M2b mid-flight estimate was ~6 h for the full 3-model × 2-scenario fetch;
  the as-shipped run completed in 24 min after the eager `.load()` patch
  in `src/get_stats_climate_proj.py` (now
  `blueearth_cst/projections/get_stats_climate_proj.py`). The followup line item
  was based on the slow path and no longer applies; no file lists it separately,
  so this entry was the last trace and is now closed rather than carried
  indefinitely as a "reminder if it resurfaces".

  **Fresh corroboration 2026-07-25:** a wf2 run observed 2.02/2.03 reading
  `gs://cmip6/CMIP6/ScenarioMIP/...` live and clearing **10 of 22 jobs in ~3.5
  min** at `-c 3` — i.e. ~100 s per model-scenario stats job, entirely
  consistent with the fast path and nowhere near the regressed one. Recorded
  because it also corrects a cost-model error made the same day: wf2's wall is
  **bandwidth-bound on GCS**, not CPU-bound, so estimates derived from its summed
  CPU benchmark (1172 s) are wrong in kind. See the Post-R6 CSV-determinism entry.

## R6 — Functional modularization (capability boundaries)

- **Climate analysis/visualization as a model-independent subworkflow.**
  *Direction raised by Ümit 2026-07-21 (test/pre06, Observation 4 follow-up).*
  We should be able to analyze and visualize climate data — gridded meteo
  diagnostics, forcing climatology, projection change factors — **without**
  building a hydrology model. Today the WF1 climate QA plots
  (`src/plot_results.py` §4) are coupled to the built wflow model
  (`mod.forcing.data`, `staticmaps["subcatchment"]`), and the forcing itself
  (`inmaps_historical.nc`) is a *product* of the model build. Yet the natural
  minimal dependency for climate analysis is a region/AOI geometry + data
  catalog — which WF3's `extract_climate_grid` (rule 3.02: `region.geojson` +
  clim source → `extract_historical.nc`) and WF2's `monthly_stats_*` already
  demonstrate (both depend only on `region.geojson`, not the full model).
  Direction: factor a shared **climate-analysis subworkflow/component** whose
  inputs are (region/AOI, gridded climate dataset) and whose outputs are
  climate diagnostics/plots, consumed by WF1 QA, WF2, and WF3 alike; degrade
  gracefully (region-only → basin-level; + subcatchment map → per-subcatchment).
  This is *functional* decomposition (capability boundaries), a **new axis**
  beyond the R6 roadmap's current layout/`enabled:` pain points (roadmap §R6) —
  add it to the R6 lock list when R6 scoping begins.
  **Tension to resolve:** ADR 0002
  (`dev/decisions/0002-revive-subcatchment-climate-plots.md`) currently sources
  the climate plots from `mod.forcing.data` (re-couples to the build); a modular
  design would source raw gridded climate (catalog + region) instead. Keep this
  in mind when ADR 0002 is implemented — it may argue for sourcing from
  `extract_climate_grid`-style extraction rather than the model forcing. To be
  discussed at R6 scoping; not to be designed or implemented yet.

- **Reconsider the WF1 rule arrangement — bundle/split + rename.**
  *Direction raised by Ümit 2026-07-21 (test/pre06, Observation re: WF1's 11
  rules).* NOT covered by R6's current lock list (which is repo/directory
  layout + `enabled:`); this is rule-level composition *within* a workflow — a
  new R6 axis. WF1 today has 12 rules (1.01–1.12; see `Snakefile_model_creation`
  and naming.md §9): copy_config, prepare_build_config, create_model,
  add_reservoirs_lakes_glaciers, add_gauges_and_outputs, write_outlet_index,
  setup_runtime, add_forcing, run_wflow, plot_results, plot_map, plot_forcing.
  Candidates to weigh:
  - **Plotting is three separate rules** (plot_results 1.10, plot_map 1.11,
    plot_forcing 1.12), each a `script:` emitting PNGs and now sharing
    `save_figure`. Consider consolidating into fewer rules (or one parameterized
    "plots" rule / a plotting sub-component) and a shared plotting module.
  - **Model-update chain is finely split** (create_model → add_reservoirs… →
    add_gauges… → write_outlet_index → setup_runtime → add_forcing). Some splits
    are historical: `add_reservoirs_lakes_glaciers`'s own comment says it "can be
    moved back to create_model when hydromt is updated" — a standing re-merge
    candidate.
  - **Verb standardization**: rules mix `create_`/`add_`/`setup_`/`prepare_`/
    `write_`/`plot_`/`run_`; `prepare_build_config` vs `setup_runtime` vs
    `create_model` overlap semantically. Align on a small verb vocabulary
    (naming.md §2 already prescribes `verb_noun`).
  **Key tradeoff — do not bundle blindly:** separate rules give Snakemake
  parallelism and *targeted* re-runs (edit forcing → only `plot_forcing` reruns);
  bundling coarsens the DAG and re-runs more on any change. Weigh granularity vs.
  readability per rule. Interactions: any reorg renumbers the `W.NN` scheme
  (naming.md §9 documents this as a mechanical cost), touches CLI target names
  (a naming.md §7 contract-surface rename → migration note), and overlaps the
  climate-subworkflow item above (plotting may move out of WF1 entirely). Same
  lens applies to WF3 (also 11 rules). To be discussed at R6 scoping; not to be
  designed or implemented yet.
