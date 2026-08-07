# Followups

Issues surfaced during pre-M1 cleanup that belong to later milestones.
Per the roadmap's "no milestone touches the next milestone's territory" rule,
captured here and resolved in passing when the relevant milestone starts.

**This file holds OPEN items only.** Closed ones live in
[`followups-archive.md`](followups-archive.md), one brief entry each, with
their IDs intact — those IDs are cited from code, tests and Snakefiles.

## Conventions

- **One bullet per item.** Keep the diagnosis date and reproducible context so
  future-you can confirm the issue still applies before fixing.
- **Closing an item is a brief note, not a post-mortem.** State what it was,
  how it ended, and the date — a few lines. If the fix taught something that
  generalizes, **promote the lesson to the file that governs the behaviour**
  (`AGENTS.md`, `dev/reference/`, a contract doc, a test or code docstring) and
  cite that from here. A lesson kept only in a backlog is read by nobody; the
  2026-08-07 sweep found that nearly every lesson worth keeping had *already*
  been promoted, so 953 lines of closure prose were duplicating guidance that
  lived somewhere better.
- **Move a closed item to the archive.** An item with anything left to do is
  not closed — it stays here, with its finished part compressed to a few lines.
- **Keep the index below in step** when an item is added, closed or moved.

## Open items

| # | Area | Item |
|---|---|---|
| R10-12 | wf1 / drift guard | `inmaps_historical.nc` is not byte-reproducible, so every model rebuild trips WF3's drift guard |
| R10-13 | logging | A failing `check_model_reference` writes an empty log part, so the file the error points at is useless |
| R10-14 | rerun triggers | A comment-only edit to a shared-rule script invalidates two whole workflows |
| R10-9 | rule identifiers | `LOG_RULES` contract test DONE; the one-label-constant-per-rule refactor is not taken |
| R10-6 | spatial rules | ADR 0003 §8–12 landed; the WF2 hydrography read cost is asserted, not measured |
| R9-1 | naming | Six geojson basenames collide across `data/spatial/geoms/` and the wflow `staticgeoms/` |
| R9-2 | baseline | Re-record owed for the indicator-table axis-column rename |
| R9-5 | wf3 results | The unperturbed baseline member is present under `aggregate_rlz: false`, absent under `true` |
| R8-1 | lint / CI | Ruff gate FIXED; open question is why a red gate went unnoticed (pre-push hook?) |
| R7-5 | wf1 rule shape | Figure families are not parse-time enumerable, so `--delete-all-output` leaves stale figures |
| R7-8 | wf3 logging | Per-member wflow logs FIXED; the concurrency falsifier has never been shown to fail |
| R7-14 | test hygiene | `tests/test_stage_data_incremental.py` fails intermittently under some orderings |
| R7-16 | packaging | O-14 decision 2 (real packaging) + O-16 (flit) still open |
| R7-17 | platform | Docker and Linux end-to-end, parked — no Linux machine |
| R7-18 | architecture | Climate analysis as a fourth Snakefile — a separate milestone |
| R7-22 | importability | `downscale_climate_forcing.py` is the last module reading the bare `snakemake` global at import |
| R7-23 | formatting | `ruff format` configured but not enforced — needs an owner ruling |
| R7-20 | housekeeping | Retire the 48 MB pre-R7 reference tree |
| — | wf3 batching | Disk-aware batch-size default; per-cst persistence isolation under batching |
| — | wf2 | Snakemake's `code` rerun-trigger does not reach rule 2.04 |
| — | ergonomics | `tee_to_log` does not capture a failing `script:` rule's traceback; per-rule progress messages |
| — | wf3 / weathergenr | `write_netcdf` drops `spatial_ref`; the wavelet minimum surfaces as a cryptic error |
| — | tests | `sys.modules.setdefault` pollution; dask cannot be stubbed at module level |
| — | upstream / hydromt | `to_yml` strips `driver.options.preprocess`; `setup_constant_pars` CSDMS names; outlet station naming |
| — | R6 direction | Model-independent climate-analysis subworkflow; WF1 rule arrangement |

---

## Carried over from the roadmap (moved 2026-08-02)

These sat in `roadmap.md` as a third backlog, referenced by neither
`TODO.md` nor this file. Content unchanged; only the location is new.

### Minor open items

Small decisions that don't justify a section of their own. Resolve
in passing as the relevant milestone starts.

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

**Reviewed 2026-08-06** by a `cst-architect` pass over the whole stack against
the code. It confirmed the M2/M3 rejections and the renumber arithmetic, and
found several defects the design had asserted its way past — recorded in the
items below and in ADR 0003. The landing order it recommended, adopted:

| # | step | why here |
|---|---|---|
| 1 | ~~`[R10-8]` stale WF2 `LOG_RULES` entry · `[R10-4]` comments · rule-index diagram fixes~~ **DONE 2026-08-06** | no sequencing dependency; one is a live defect |
| 2 | ~~`[R10-9]` the `LOG_RULES` conformance test~~ **DONE 2026-08-06** (`tests/test_log_rules_contract.py`, 9 passed) | the sweep's highest-risk surface, verified *before* the sweep edits it |
| 3 | ~~`[R10-1]` merge~~ **DONE 2026-08-06** · ~~`[R10-2]` split~~ **DROPPED 2026-08-06** — no seam worth its price; `evaluate_` withdrawn with it | the merge was small and behaviour-preserving; the split turned out not to have the seam it assumed |
| 4 | ~~`[R10-6]` §8–10 — the vector/raster split~~ **DONE 2026-08-06** (rules `1.01c` / `2.03c` / `3.01f`; nine WF1 artifacts byte-identical; ADR §8–10 now **accepted**). Baseline gate still open | changes the rule count of all three workflows |
| 5 | ~~`[R10-6]` §11a then §11b~~ **DONE 2026-08-06 as ONE landing** — measurement collapsed the split: the fixture's partition saturates at 5 subbasins from ceiling 5 up, so §11b moves nothing and only the key rename shows. **Baseline re-recorded `ea5ac59`** | §11b turned out NOT to be a baseline event on this fixture |
| 6 | ~~R10 renames + `[R10-5]` renumber + `[R10-7]` + `[R10-10]`~~ **DONE 2026-08-06** — eight commits; the number map was regenerated FIRST and owner-approved before any code moved against it. `[R10-9]`'s deferred ordering assertion landed with the renumber that makes it true. Gates: `pytest tests/` 1526 passed (primary, fixture layer included); **a full three-workflow run** — WF1 17/17, WF2 14/14, WF3 34/34, every merged-log section present in number order and no `_parts/` surviving, including the batch and fan-out labels no test reaches; `check_baseline.py check` **OK 8/8 AFTER that run**; `pixi run tree-check` **186 paths, 0 unmapped** | against a rule set that is finally stable; regenerate the number map from it. `[R10-10]` rides here because it and `[R10-9]`'s ordering assertion touch the same test file |
| — | ~~`[R10-11]` tree-check on a post-migration tree~~ **DONE 2026-08-06** — post-migration inventory is the default; 186/186 identity on the live tree | done before step 6, so the sweep has a working tree-shape gate while it runs |
| 7 | ~~`[R10-6]` §12~~ **DONE 2026-08-06** — landed with §11's re-record still pending, so ONE re-record covered both: **`ea5ac59`**, 4 targets (3 config snapshots + `q_indicators.csv`). Nothing further owed | standalone, last |

**This resolves the double-renumber contradiction.** `[R10-6]` says land the
split before `[R10-5]`, but `rule-index.md` publishes a 45-identifier map that
excludes §8's new rules. Renumbering now moves to **step 6**, after the rule set
stops changing, and the published map is regenerated at that point rather than
being the target from the start.

As of 2026-08-06 that is no longer hypothetical: step 4 ADDED three identifiers
the published map does not carry — `1.01c` / `2.03c` / `3.01f`
`delineate_spatial_units`, one per workflow. Step 6 regenerates the map, so it
must ADD rows, not only renumber the 45 that were there; `3.01f` in particular
exists only because `3.01c`–`3.01e` were taken, and renumbering is what removes
that awkwardness.

- **[R10-12] `inmaps_historical.nc` is not byte-reproducible, so every model
  rebuild trips WF3's drift guard.** *Found 2026-08-06 by the R10 step-6
  three-workflow run — the first time WF1 had been re-run end to end since the
  guard was built.*

  WF1 rebuilt the model; `check_model_reference` then failed at job 1 of 34
  naming exactly one changed input: `forcing/inmaps_historical.nc`.
  `staticmaps.nc` and `wflow_sbm.toml` were unchanged.

  **The values did not move — only the bytes.** The old file was overwritten so
  it cannot be diffed directly, but `run_default/output.csv` is a baseline
  target, is a deterministic function of (forcing, model, TOML), and matched
  byte-for-byte. Had the forcing numbers changed, that would have moved. The NC
  carries no global attributes, so it is not an embedded timestamp; most likely
  HDF5 chunk/encoding layout from the hydromt write.

  **The guard is behaving correctly and must not be "fixed" by loosening it.**
  `write_model_reference` declares its model inputs `ancient()` *on purpose* —
  a reference that refreshed whenever the model changed would always match and
  the comparison would be decorative. That asymmetry IS the mechanism.

  **The cost is operational and lands on a real project, not the fixture:** any
  WF1 rebuild blocks every existing experiment until its reference is
  re-recorded, including rebuilds that changed nothing numeric. On the fixture
  the recovery is to delete `<exp>/config/model_reference.yml` and let
  `write_model_reference` regenerate it; on a real project that is a decision
  ("this experiment now accepts the rebuilt model"), not a chore.

  Options when picked up, in preference order: **(a)** make the digest ignore
  the forcing NC's storage layout — hash its *values* (variable-wise checksums)
  rather than the file, which is what `compare_model_digest` would need;
  **(b)** make the hydromt forcing write byte-deterministic by pinning
  encoding/chunking, if hydromt exposes that; **(c)** accept it and document the
  re-record step as normal after any rebuild. (a) is the honest fix — the guard
  should fire on physics changes, not on chunk layout.

- **[R10-13] A failing `check_model_reference` writes an empty log part, so the
  file the error points at is useless.** *Same run.* Snakemake reported
  `Error in rule check_model_reference … log: …/3.06_check_model_reference.log
  (check log file(s) for error details)`, and that file contained the three
  header lines and nothing else. The `ModelDriftError` — which names the changed
  input, the useful part — went to the Snakemake log instead.

  It is a `script:` rule, so `tee_to_log` captures Python-level streams; an
  exception propagating out of the script is not written through the tee before
  the process dies. Cheap fix: catch, log the diff lines to the rule's own log,
  re-raise. **The message actively misdirects** — it tells an operator to read a
  file that will not explain the failure, which is worse than saying nothing.
  Likely applies to every `script:` rule that raises, not just this one; check
  before scoping.

- **[R10-14] A comment-only edit to a shared-rule script invalidates two whole
  workflows.** *Same run.* The `[R10-7]` rename touched one line inside
  `blueearth_cst/spatial/delineate_region.py`. Snakemake's `code` rerun-trigger
  hashes the entire script text, comments included, so `delineate_region`
  re-ran, `region.geojson` was rewritten, and all 17 WF1 jobs plus all 25 WF2
  jobs were scheduled from it.

  Correct Snakemake behaviour, and convenient here — it gave a full run with no
  `--forceall`. But it is the same over-invalidation the WF2 design already
  engineered around: `series_identity.kernel_hash` hashes the *behaviour* of
  enumerated reduction functions rather than file bytes, precisely so an
  error-handling edit does not re-derive nine series over the network (the 4c
  incident). The three shared-rule scripts have no equivalent protection, and
  they are the highest-fan-out scripts in the repo — one of them re-running
  cascades into every workflow that declares it.

  Not urgent, and possibly not worth fixing: the safe direction for a cache is
  to over-invalidate. Recorded because the *asymmetry* is surprising — WF2's
  reducer is protected and the shared region/vector/store producers are not —
  and because anyone editing a comment in those three scripts should know it
  costs a full re-run of up to three workflows.

- **[R10-9] Make the `LOG_RULES` contract a test.** **The test is DONE
  2026-08-06** — `tests/test_log_rules_contract.py` asserts set-equality between
  each Snakefile's `LOG_RULES` and the labels derived from every rule's `log:`
  path, plus, since the `[R10-5]` renumber, that the list reads in rule-number
  order. It confirmed `[R10-8]`'s deletion left no orphan. The
  unlisted-or-stale-label defect had occurred four times before it existed.

  **Still open: one label constant per rule.** Defining
  `_L = "1.10_add_climate_forcing"` and building `LOG_RULES`, `rule_banner`,
  `log:` and `benchmark:` from it would collapse four of the six call sites per
  rule and make the next rename a one-line edit. The R10 sweep was the moment
  this was free and dropped it anyway, so the reason matters: **`LOG_RULES` must
  stay a module-level list literal.** Three consumers read it out of the source
  text without executing the Snakefile — a Snakefile is not valid Python, so
  `ast.literal_eval` on the lifted block is the only way to read it. Building the
  list from per-rule constants makes it a list of *names* and `literal_eval`
  raises. Closing this needs the checker to parse the executed workflow's globals
  instead — a change to the instrument, so do it as its own task, instrument
  first.

- **[R10-6] Split `prepare_spatial_maps` so WF2 and WF3 can consume basin and
  subbasin boundaries.** *ADR 0003 §8–12.* **§8–12 all landed 2026-08-06.** The
  vector half moved behind a third shared rule (`delineate_spatial_units`,
  `1.01c` / `2.03c` / `3.01f`) declared in all three workflows; the thematic
  raster stack stayed WF1-only. The seam is `data/spatial/hydrography.nc`,
  deliberately absent from `spatial_catalog.yml`. §11 made
  `automatic_subbasins.max_count` a per-basin `max_per_basin` (default 20 → 11);
  §12 changed `wflow_id` to `basin_id*1000 + subbasin*10 + m`, which uncovered a
  live defect — `output.csv` had shipped `Q_101` twice, a collision that had
  already leaked into WF3's surface as a column named `Q_101.1`. One baseline
  re-record covered both (`ea5ac59`), so **the manifest is CURRENT**. Full
  decision, consequences and landed state:
  `dev/decisions/0003-one-shared-region-artifact.md`.

  **Still open, both from ADR 0003:**

  - The hydrography-read cost §8 adds to WF2 is **asserted, not measured** —
    validation item 7 makes measuring it the acceptance gate.
  - The nine-additional-locations cap was never checked against a real gauge
    list: the fixture runs `gauge_points: null` and real basin data lives
    outside the repo. The cap raises by name if tripped, so failure is loud.

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

- **[R9-5] The unperturbed baseline member is in the response tables under
  `aggregate_rlz: false` and absent under `true`.** Surfaced 2026-08-07 while
  fixing [R9-3], in the branch immediately above the axis read.

  `export_wflow_results.py` guards the baseline with `if st_nb == "0"`. In the
  `aggr_rlz=False` branch `st_nb` is a **string** parsed off the run CSV's name,
  so the guard fires and `cst_0` gets a row with both axes at 0. In the
  `aggr_rlz=True` branch `st_nb = i + 1` is an **int**, so `st_nb == "0"` is
  false for a reason that has nothing to do with the baseline — the comparison
  is int-to-str and can never be true. It is harmless only because that loop
  runs over `1..st_num` and never reaches the baseline at all: the aggregated
  table simply has no `cst_0` row.

  So the two shapes of the same table disagree on whether the baseline is a
  member of the response surface, and the dead comparison hides it. Which is
  correct is a question about the artifact's contract (HM-7 pins the columns,
  not the row set) rather than a typo — the baseline is a legitimate (0, 0)
  point on the surface, but it is also the only member with no perturbation to
  average over realizations. `ST_START = 0 if run_hist else 1` means the
  baseline runs exist either way when `run_hist` is set, so the data is there
  in both shapes.

  Deliberately not folded into [R9-3]'s fix: adding or dropping a row moves the
  artifact, which the axis-collapse change explicitly did not.

---

## Post-R8 (surfaced 2026-08-02 during the Post-R7 triage)

- **[R8-1] ~~The ruff gate is red on `main`.~~ FIXED 2026-08-07** — `pixi run
  ruff check .` reports **All checks passed!**, the exact command
  `.github/workflows/ci.yml` runs on both legs. It had grown from 10 findings to
  14 by the time it was cleared. Nine were `ruff check --fix`; five needed
  judgment, and two are worth remembering: the `F841` was genuinely dead, and an
  `import sys` that looked used appeared only inside a **docstring**, so a grep
  would have kept a dead import where ruff was right. Gates: ruff clean;
  `pytest tests/` 1503 passed / 31 skipped / 1 xfailed.

  **Still open, which is why this is not archived:** whether the gate went red
  after the R7-19 seal's green CI run (30450296441), or a later red went unacted
  on. The four newcomers say nobody was watching — an argument for a pre-push
  hook rather than more fixes.

---

## Post-R7 (surfaced 2026-07-28/29 during the R7 project-layout milestone)

R7 landed as 15 `r07:` commits with a clean full-tree diff, a green
`check_baseline`, and the P4 assertion demonstrated. The items below are what it
deliberately did **not** fix, plus what implementation surfaced along the way.
Provenance: `dev/milestones/r07/migration_project-layout.md` §§7a–7d,
`dev/milestones/r07/project-layout-design.md`, and the `r07:` commit messages.


### Design debt accepted knowingly

- **[R7-5] O-24 is partially closed; its premise was wrong.** *Remaining half is
  row `t260802b`.* The basin-average half was FIXED 2026-08-01 — rule 1.11
  derives `plot_basavg`'s PNGs from `wflow_outvars` and declares them, verified
  reaching `--delete-all-output`.

  **The rest cannot be closed the way this entry assumed.** It claimed all three
  families were derivable "at parse time from `wflow_outvars` /
  `output_locations`". They are not: `hydro_{station}.png` and
  `clim_{station}_{period}.png` are counted by the model's OUTLETS and
  SUBCATCHMENTS — a rule-1.03 product, unknown until the model is built — and
  `signatures_{station}.png` also needs observations AND a run longer than a
  year, so it is data-conditional, not merely config-conditional. Closing those
  needs a `checkpoint` or a `directory()` output, a real rule-shape change.
  Consequence, unchanged: on a config with extra gauges or observations,
  `--delete-all-output` cannot clean them and stale figures survive a rerun.

### Cosmetic / low priority

- **[R7-8] ~~wflow writes `log.txt` beside the run TOML.~~ FIXED 2026-08-03 by
  R9 P2 commit 3.** `downscale_climate_forcing.py` sets `logging.path_log` per
  member, so each logs to its own file. It shipped in the same commit as the
  `rlz_<n>/` flattening and not after: flattening put every member's TOML in one
  shared `config/`, where the wflow default `"log.txt"` would have had all
  members writing one file concurrently — a cosmetic became a correctness
  problem the moment realizations stopped owning a directory.

  **Still owed:** the concurrency falsifier has never been shown to FAIL with
  `path_log` unset. Distinct pointers per member are unit tested via
  `snake_utils.member_pointer_base`; content attribution under a real concurrent
  batch still needs a run.

- **[R7-14] `tests/test_stage_data_incremental.py` fails intermittently** *(row
  `t260802c`; still present and still flaky, confirmed 2026-08-02)* under
  some orderings; passes in isolation and on re-run. Another workstream's
  module, predates the R7 branch. Test-isolation issue, not a product defect.

### Parked by ruling — not defects

- **[R7-16] Tooling contract** *(O-14 decision 2 + O-16 are row `t260802d`)*.
  **Resolved:** O-14 decision 1 (`ab781a5`) — a tool-config-only
  `pyproject.toml`, no `[build-system]` / `[project]` / `[tool.pixi]`; and O-15
  (`85d3178` → `81e0096`) — ruff adopted as the lint gate with
  `select = ["E4","E7","E9","F"]` pinned explicitly, because ruff 0.16's
  *default* selection is ~415 rules and must not be inherited by accident.
  **Open:** O-14 decision 2 (real packaging) needs a superseding record in
  `dev/decisions/`; O-16 (flit) stays gated on it. `ruff format` is configured
  but deliberately unenforced — see R7-23.

- **[R7-17] Docker (O-06) and Linux end-to-end (O-18, O-19)** — parked, no Linux
  machine. Linux *parse-level* consistency is now covered: the Linux config
  dry-runs on both CI legs.
- **[R7-18] Climate analysis as a fourth Snakefile** — a separate milestone. R7
  only ensured the layout does not obstruct it, and the model-free store plus
  rule 1.15 are the enabling pieces.

### Milestone housekeeping

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
---


---

## Cross-cutting — workflow ergonomics

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

---


---

## R5 — Workflow 3: climate experiment

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

- **Outlet station naming convention decision.** hydromt_wflow 1.x's
  `setup_outlets` uses subcatchment IDs (e.g. `130000086`, `1`, `2`, …) for
  outlet stations rather than the contiguous `1..N` of 0.x. The CSV column
  also renamed `Q_gauges` → `Q_outlets`. M2b's `src/plot_results.py`
  rebuilds `station_name` as `1..N` to keep `hydro_wflow_1.png` visually
  stable; R3 should pick a consistent project-wide convention (real
  subcatchment IDs vs `1..N` rebuild) and document it.

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
