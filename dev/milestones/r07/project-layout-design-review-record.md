# R07 — Project layout — design review record

**The audit trail behind `dev/milestones/r07/project-layout-design.md` (ACCEPTED 2026-07-28).**
Produced by a `design-review-loop` run, slug `r07-project-layout`, 2026-07-27 →
2026-07-28. The per-round scratch directory has been pruned; this file is the
durable record. The framing the run was approved against is
`dev/milestones/r07/project-layout-intake.md`.

Everything below the gate record is **verbatim reviewer output** or the
append-only finding ledger. Nothing has been re-graded, merged away, or edited
for presentation.

## Outcome

| | |
|---|---|
| Verdict at close | **ACCEPTED** by the owner at gate G2, 2026-07-28 |
| Versions | 4 (`design-v1` … `design-v4`) |
| Findings | **44 — 7 blocking, 20 major, 7 minor internal; 2 blocking, 4 major, 1 minor + 1 blocking, 1 major, 1 minor external** |
| Dispositions | **44 accepted, 0 rejected, 0 deferred** |
| Reviews | 3 internal lenses (1 round) + 2 external cross-vendor rounds |
| Gate rulings | 9, across G1, two G1 returns, and round-cap arbitration |
| Author dispatches | 7 Opus + 2 Fable (Fable spent on the two revisions answering faulted resolutions) |

### Verdict table

| Round | Reviewer | Doc version | Verdict | Findings |
|---|---|---|---|---|
| Internal — risk & assumptions | `critical-thinker` | `design-v1` | `revise` | 10 — 2 blocking, 6 major, 2 minor |
| Internal — architecture & consistency | `cst-architect` | `design-v1` | `revise` | 11 — 2 blocking, 8 major, 1 minor |
| Internal — repo fit & conventions | `python-engineer` | `design-v1` | `revise` | 13 — 3 blocking, 6 major, 4 minor |
| External round 1 | headless `codex exec` (GPT, clean-room) | `design-v2` | `revise` | 7 — 2 blocking, 4 major, 1 minor |
| External round 2 | headless `codex exec` (GPT, + regression duty) | `design-v3` | `revise` | 3 — 1 blocking, 1 major, 1 minor |
| **Close** | **Owner arbitration (round cap reached)** | **`design-v4`** | **ACCEPTED** | all 3 survivors accepted, fix required |

**The final version carries no external reviewer verdict.** The cap of two
external rounds was exhausted with round 2 unconverged; per the loop's
failure-mode contract that triggers owner arbitration rather than a third round.
`design-v4` implements the three arbitrated fixes under the owner's authority,
scope-checked by the driver against those finding IDs. This is recorded here and
on the design's own face rather than left for a reader to infer.

### What review actually changed

The findings that altered the design materially, as opposed to sharpening it:

- **B1 had no named producer.** All three internal lenses found this
  independently (risk-1, arch-1, repo-2, all blocking) and each showed every
  available assignment broke a stated commitment. The owner selected the
  shared-rule route at a gate return.
- **The `ancient()` guard mechanism was wrong.** External round 1 (ext1-02)
  claimed `ancient()` suppresses only mtime-triggering, not input-set
  membership; verified against Snakemake 9.6.2 source and confirmed. The
  producer's input contract was rebuilt.
- **The phase-B verification gate crashed on its own most substantive move.**
  `semantic_tree_diff.diff_trees` is strictly one-to-one and raises on the
  many-to-one collapse B1 performs (risk-2, arch-2, both blocking).
- **"The wf1 discharge target is unchanged" was false** (repo-3, blocking) — all
  18 manifest keys move, and the sidecar name is `sha1(resolved_path)[:16]`, so a
  re-record would have silently re-blessed drift on the milestone's strongest
  numeric anchor.
- **The `"None"` sentinel risk was misdiagnosed at both cited lines** (repo-4) —
  both short-circuit on `is not None`, so the design's only runtime assertion
  would have passed while proving nothing.
- **`get_region_preview.py`, cited as the model-free escape hatch, does not
  import** on the pinned hydromt 1.3.1 (`hydromt.cli.api` removed in hydromt
  1.x). Found during revision; logged as observation O-25.
- **Two moves drawn in the tree were assigned to no commit** (arch-5, repo-5),
  and the baseline gate was down for nine of thirteen commits with no stated
  substitute (repo-3, risk-3, arch-9).

## Gate and arbitration record

Reproduced from the run's `status.md`. Every ruling is the owner's; the driver
recorded them and never decided one.

### G1 — framing, approved 2026-07-27

Provisional alternative approved: one batched milestone across both halves,
governed by P1–P4, `hydrology_model/` kept as the hydromt `model_root`
(option A). Four open questions ruled:

| # | Question | Ruling |
|---|---|---|
| OQ-1 | Engine-named subtrees (`models/wflow/`) | **Parked, non-gating** — stays an open question, explicitly deferred beyond R07 |
| OQ-12 | `MIGRATION.md`'s home | **Move to `docs/`** — its audience is users. Diverges from `naming.md` §7; the divergence must be stated and §7 reconciled or an exemption recorded |
| O-13 | `blueearth_cst.Rproj` | **Delete** — unreferenced, not used by the owner |
| OQ-4 | weathergenr date CSVs | **`weather_generator/output/` as designed** — a product of the generator, not a `_work/` diagnostic |

### G1 return — 2026-07-28

Triggered by the internal panel: the three candidate resolutions for B1's
blocking group differed in *milestone scope*, which is a material change to the
selected alternative.

| # | Question | Ruling |
|---|---|---|
| GA-1 | B1's producer assignment (risk-1 / arch-1 / repo-2, all blocking) | **The shared-rule fix** — one producer rule over **region + catalog only**, declared identically in both Snakefiles. Keeps P2 and P4; scope unchanged. Accepted cost: rule 1.10's input changes from `staticmaps.nc` to the region, so the bbox derivation genuinely changes. Must be named as a **third exception** to the behaviour-preservation stance and **proven empirically**, not assumed |
| GA-2 | What depends on current `project_dir` artifact paths (risk-5) | **Only the test fixture.** No production trees to preserve; no CST-API or CST-frontend consumer reads artifact paths. Pre-R07 trees **unsupported** — fresh run required, no `mv` script shipped |

### Second G1 return — 2026-07-28

Triggered by revision r1: three accepted findings changed decision criteria the
owner had approved at G1. `intake.md` was amended to match rather than left in
conflict with the design.

| # | Change | Ruling |
|---|---|---|
| GB-1 | **Goal narrowed** (arch-8) — extensibility → separability | **Accepted.** The delivered tree cannot honour "a second modelling engine can be added without inventing a new layout"; the rule that would honour it requires deciding OQ-1, which is parked. Ships as a stated limitation |
| GB-2 | **Commit count 13 → 15** | **Accepted.** GA-1's "scope unchanged" reads as *content* scope, which is identical. The delta is arch-9's machinery-first split plus B9 and B10 — two items v1 drew but assigned to no commit |
| GB-3 | **P1 and P2 reworded** | **Accepted, consequentially.** P1 "attach to their producer" → "attach to what they depict"; P2 "one producer per artifact" → "one producer *definition* per artifact" (the stated precondition of the GA-1 route) |

### Round-cap arbitration — 2026-07-28

External round 2 returned `revise` with three findings and the cap was reached.
Driver evidence supplied at arbitration, beyond both parties' rationales:
**ext2-01 describes a pre-existing gap, not an R07 regression** — today's rule
3.02 already carries the catalog as `params: data_sources = DATA_SOURCES`, not
as a declared input, so catalog-content edits do not retrigger extraction today
either.

| ID | Severity | Ruling |
|---|---|---|
| ext2-01 | blocking | **Accepted; fix = symmetric catalog input** declared identically in both DAGs (the round-1 oscillation came from *asymmetric* inputs, so a symmetric one cannot reproduce it). **Owner-mandated verification:** confirm the catalog set is genuinely identical in both workflows; if not, fall back to a catalog digest in params |
| ext2-02 | major | **Accepted; fix required** — compare every content- or execution-affecting directive and fail on unknown per-workflow fields |
| ext2-03 | minor | **Accepted; fix required** — correct the blackout start and resolve the contradictory `check_baseline.py` commit ownership |

**Outcome of the mandated verification: passed, symmetric input adopted.** Both
Snakefiles bind `DATA_SOURCES` from the same `project.data_sources` key
(`Snakefile_model_creation:31`, `Snakefile_climate_experiment:34`) — a single
path, not a list — and today's two extraction rules already carry that identical
value. The composed catalog at `Snakefile_climate_experiment:344` belongs to rule
3.09, not the producer. The digest-in-params fallback was not needed and is
recorded in the design's alternatives as the on-file route if a producer catalog
set ever diverges.

### G2 — approval, 2026-07-28

`design-v4.md` approved as-is. Zero editorial change requests, so the loop's lean
finalize path applied: the driver landed the design mechanically and the
ACCEPTED status-header swap is a logged editorial edit. Driver scope-check of the
arbitration revision: **pass** — 18 hunks across 10 sections, every one mapping to
an arbitrated finding's section, a declared forced cross-reference, or the
revision log.

---

# Internal panel — aggregated index

*Driver-written aggregation of the three lens reviews, verbatim as produced
during the run. Grouping only: every original finding ID, severity, and text is
preserved; nothing was deleted or re-graded.*

Stage 2 of the `r07-project-layout` design-review-loop run. Three lenses reviewed
`design-v1.md` in parallel, clean-room (no shared context, no ledger), each given
the four G1 rulings as settled framing.

**Driver-written index.** Grouping only. Every original finding ID, severity, and
text is preserved verbatim in the per-lens files; nothing here deletes, merges
away, or re-grades a finding. Where two lenses disagree on severity or on the
facts, both readings are recorded as a conflict for the author to adjudicate with
evidence — the driver does not pick a winner.

| Lens | Reviewer | Verdict | Findings |
|---|---|---|---|
| risk & assumptions | `critical-thinker` | `revise` | 10 — 2 blocking, 6 major, 2 minor |
| architecture & consistency | `cst-architect` | `revise` | 11 — 2 blocking, 8 major, 1 minor |
| repo fit & conventions | `python-engineer` | `revise` | 13 — 3 blocking, 6 major, 4 minor |
| **total** | | **`revise`** | **34 — 7 blocking, 20 major, 7 minor** |

Sources: `internal-review-risk.md`, `internal-review-architecture.md`,
`internal-review-repo-fit.md`.

---

## Group A — B1 has no named producer (3 blocking, all three lenses)

**risk-1** (blocking) · **arch-1** (blocking) · **repo-2** (blocking)

The single highest-confidence result of the panel: three lenses, working
independently, reached the same conclusion by three different routes. B1 says
"one producer, both workflows consume" and never assigns the producer to a
Snakefile; each lens enumerated the available assignments and found every one
breaks a stated commitment.

- wf1 produces → P4 fails (the store needs wf1), and B1's own justification is P4.
- wf3 produces → `Snakefile_model_creation` gains a dependency on a wf3 artifact,
  inverting the documented wf1→wf2→wf3 order; a wf1-only run dies with
  `MissingInputException`.
- both declare → P2 breached; two rules with different inputs produce one path
  across two DAGs.

Distinct contributions worth carrying separately into the ledger:

- **risk-1** finds the design reverses a *prior accepted design* without citing it:
  `dev/milestones/p32a/climate-analysis-design.md:389-405` rejects the single store
  explicitly and by name ("Why a separate wf1 extraction and NOT reuse of the
  P3-1 keyed store"). It also notes B4 does not dissolve wf1's dependency on the
  store, so the wf3-produces branch is not rescued by B4.
- **arch-1** finds `get_region_preview.py` — cited as the model-free escape — is a
  standalone `argparse` CLI wired into no rule, with its own hydrography default
  (`merit_hydro_ihu`); adopting it is a new rule, not a path move.
- **repo-2** adds that B1's region-derived bbox ruling already forces rule 1.10's
  input to change, and that the P3-1 guard contract (`{store_dir}/.guard_ok`) is
  keyed to the store dir.

**The three lenses propose three different resolutions**, and they differ in
milestone scope — this is the run's main decision point, not an authoring choice:

| Source | Proposed resolution | Scope effect |
|---|---|---|
| risk-1 (i) | Pull a minimal standalone extraction rule into R07 | Widens scope; breaks the "no computational path changes" constraint |
| risk-1 (ii) | Drop B1 to a follow-on milestone, keep B2/B3/B5–B8 | Narrows scope; B2 does not depend on the collapse |
| arch-1 (alt) | Keep B1 but restate the P4 assertion as what R07 actually proves | Preserves scope; abandons a named success criterion |
| repo-2 | One rule over region + catalog, declared in **both** Snakefiles | Preserves scope; needs a P2 reading that permits it |

## Group B — the phase-B verification gate cannot express B1 (2 blocking)

**risk-2** (blocking) · **arch-2** (blocking)

Independent convergence. `semantic_tree_diff.diff_trees` is strictly one-to-one
and raises `ValueError("path map collision: … both map to …")` on many-to-one.
B1 is a many-to-one collapse (`wf1_raw/extract_historical.nc` and
`<key>/extract_historical.nc`, plus two `orography.nc` on the chirps branch), so
the milestone's single most-cited proof of behaviour preservation aborts before
it can report.

- **risk-2** adds that the path map is not parameterized at all —
  `build_p31_path_map()` / `build_p31_allowlist()` are hardcoded milestone code and
  `main()` exposes no `--map`, so the gate cannot run until commit 12 regardless.
  It also names the semantic that is actually needed: the survivor must be
  compared against *each* collapsed source, and a merge is proven by two passing
  comparisons, not by allowlisting one as MISSING.
- **arch-2** cites the derived migration map (`dev/milestones/r07/migration_project-layout.md:154`,
  "**MISSING** — none by design") as evidence the mapping-rule route was the
  intended one, and proposes either a declared many-to-one merge class or an
  explicit `--retire` set.

Note the repo-fit lens independently reports that the design **overstates** one
adjacent claim: the TOML comparator already covers all five pointer fields and is
generic over the path map, so B5 needs a new `build_r07_path_map`, not a
comparator change.

## Group C — the regression gate is down for most of the milestone (1 blocking, 2 major)

**repo-3** (blocking) · **risk-3** (major) · **arch-9** (major)

All three lenses independently identify a blackout window: `manifest.json` keys
are literal paths prefixed `examples/test_local/`, so commit 3's fixture rename
invalidates all 18 keys at once, and the machinery + re-record land only at
commit 12. Nine of thirteen commits — including every value-touching one — run
with no working regression detector, after which the re-record freezes whatever
is on disk.

- **repo-3** escalates this to blocking on a specific mechanism the others miss:
  the wf1 discharge target is compared with a *tolerance comparator against a
  stored series*, not a self-contained hash, and `check_baseline.py:384-385`
  derives the sidecar filename as `sha1(resolved_path)[:16]` — so the rename
  changes the sidecar name, `record` writes a new series from the post-R07 run,
  and drift is silently re-blessed. It also states the design's claim that the
  discharge target "is unchanged" is **false**.
- **risk-3** adds the cut-line contradiction: § Risks says the artifact half is the
  coherent unit if the milestone must be cut, but the commit plan lands the
  *repository* half first, so a mid-flight stop leaves the deferrable half done
  and the coherent half half-done.
- **arch-9** proposes moving the mechanical machinery update *before* the moves,
  keeping only the manifest re-record at the end.

## Group D — the baseline inventory arithmetic is wrong (2 major, 1 minor)

**risk-6** (major) · **arch-10** (major) · **repo-13** (minor)

Triple convergence on the same facts: `manifest.json` holds 18 rows but
`check_baseline.TARGETS` holds 15 live templates; three rows are pre-P3-1 orphans
with no producer (`climate_experiment/model_results/{Qstats,basin}.csv` and the
root-level `config/snake_config_climate_experiment.yml`) that a full `record`
silently drops. The design's "four copied-config snapshots" is therefore three,
and the doc's own tree keeps the wf3 experiment snapshot inside
`experiments/<id>/` rather than moving it into `config/runs/`. The listed movers
sum to 16, not 18.

Consequence all three state: the exit adjudication ("path-and-snapshot-only") will
see three unexplained *deletions* it cannot mechanically account for.

**Severity divergence, preserved:** filed `major` by risk and architecture,
`minor` by repo-fit.

## Group E — the four principles are contradicted by the tree they govern (2 major)

**risk-4** (major) · **arch-7** (major)

P3 states without qualification that every engine subtree shares
`config/ output/ plots/ _work/`. The tree satisfies it in exactly one place:
`weather_generator/` has all four, `hydrology_runs/rlz_<r>/` has two,
`hydrology_model/` — the largest — has none. Both lenses reach the same fix
shape (state the exemption inline, naming hydromt's ownership of `model_root`),
and both note the design records the cause only in an *Alternatives considered*
entry for a rejected nesting, never as an exemption to P3.

- **risk-4** additionally faults **P1**: `basin_area.png` is produced by rule 1.12
  from `staticgeoms/outlets.geojson` and has nothing to do with evaluation, yet
  lands in `hydrology_model/evaluation/plots/` — `evaluation/` is a topic bucket,
  not a producer. It also finds P3 gives no answer for the *generated* wflow build
  configs (`wflow_build_model_run.yml`, `wflow_build_forcing_historical.yml`),
  which the tree files beside verbatim template snapshots.
- **arch-8** (major, filed separately) carries the same defect up to the **Goal**:
  "a second modelling engine can be added without inventing a new layout" is not
  delivered, because hydrology appears twice, in two shapes, at two levels. The G1
  ruling parks the *naming* question; arch-8 argues the structural question is
  upstream of naming and unraised.

## Group F — moves drawn in the tree but absent from B1–B8 and the commit plan (2 major)

**arch-5** (major) · **repo-5** (major)

Two separate items appear in the § B tree with no substantive-move entry, no
commit, and no verification row:

- **arch-5** — `hydrology_model/evaluation/`. This retires
  `{project_dir}/plots/wflow_model_performance/`, home of three manifest targets
  and the outputs of rules 1.11, 1.12, 1.13 plus rule 1.14's gather inputs. B4
  covers only rule 1.13.
- **repo-5** — the `config/` → `runs/ catalogs/ templates/` split. Not a path move
  at all: `copy_config_files.py` derives a single `output_dir` and writes all kinds
  into it, so routing three kinds needs a signature change — which falsifies the
  "every item is a path move, a rename, a declaration fix, or an added warning"
  claim. It also notes the two *generated* build configs are filed under
  `templates/` beside copied source templates.

## Group G — the contract/edit surface is larger than the doc's `--dry-run` concession (1 major, 1 minor)

**arch-4** (major) · **risk-10** (minor)

The doc concedes only `params:`-string paths and R `shell:` bodies. Both lenses
find the moved paths are also hardcoded inside `script:`-directive Python modules
(`plot_results.py:108`, `plot_map.py:34`, `plot_map_forcing.py:201`,
`export_wflow_results.py:161,281`, `setup_time_horizon.py:51`,
`downscale_climate_forcing.py:72`, `generate_weather.R:68`) and across at least
seven test modules that no commit touches — while "`pytest tests/` green" is a
stated success criterion. **arch-4** additionally names the `dev/contracts/*-seam.md`
documents, which pin the same paths and are unlisted.

**Severity divergence, preserved:** `major` (architecture) vs `minor` (risk).

## Group H — `COPIED_CONFIG_PATH_MAP` omitted from the machinery list (1 major, 1 minor)

**repo-6** (major) · **arch-11(a)** (minor)

Both lenses find the design's machinery list names only the directory-prefix map
and the TOML comparator, omitting `semantic_tree_diff.py`'s third normalization
table. **repo-6** works out the consequence in detail: O-20 changes
`project.project_dir` inside every copied snapshot and `compare_copied_config`
FAILs on any residual difference, so the phase-B gate goes red for pure path
bookkeeping — indistinguishable from a real content regression. `MIGRATION.md:167-172`
records that this table is kept in lockstep with the migration map.
**arch-11(a)** adds that `_is_copied_config` matches any YAML with a `config` path
part, so the new `weather_generator/config/weathergen_config.yml` is newly swept
into that directional policy.

**Severity divergence, preserved:** `major` (repo-fit) vs `minor` (architecture).

## Group I — the `MIGRATION.md` ruling leaves `naming.md` §7 unreconciled (1 major, 2 minor)

**risk-8** (major) · **repo-11** (minor) · **arch-11(b)** (minor)

The downstream inconsistency the G1 ruling creates — which the panel brief
explicitly asked for, and which v1 addresses nowhere.

- **risk-8** sharpens it into a paradox: by the ruling's own audience test, R07's
  own map is *more* user-facing than `MIGRATION.md`, because it maps production
  `project_dir` paths that users' artifact trees sit on. The design places a
  user-facing map under `dev/` in the same milestone that moves a user-facing map
  out of it, on opposite reasoning.
- **repo-11** finds a stated divergence is **not sufficient**: §7 makes
  `dev/<milestone>/migration_<topic>.md` the *required artifact* of a contract
  rename, so moving the R06 note to `docs/` leaves §7 unsatisfied for R06 unless
  §7 itself is amended. It also flags `docs/` casing consistency.
- Both propose the same shape: amend §7 to distinguish an internal rename record
  (`dev/<milestone>/`) from an optional user-facing guide (`docs/`).

**Severity divergence, preserved:** `major` (risk) vs `minor` (repo-fit,
architecture).

## Group J — figure families proliferate beyond what B4 describes (1 major, 1 minor)

**arch-6** (major) · **risk-9** (minor)

B4's table names two figure families; **arch-6** finds three — the
`clim_wflow_1_{month,year}.png` pair, produced by rule 1.11 at *model parity*, is
named nowhere in B4 and lands in `hydrology_model/evaluation/plots/`. It asks
whether `clim_wflow_1_*` survives the new producer at all. **risk-9** finds the
collision is a *set* of basenames, not one file, and that it compounds with the
two-PET risk: `pet.png` is exactly the pair the design says someone will compare
and report as a defect, published under one filename. Both propose disambiguating
filenames rather than relying on the parent directory. risk-9 also notes the new
producer needs `era5_orography` from the catalog on the era5 branch — compatible
with P4 but absent from B4's input list.

## Group K — singletons

Each raised by one lens, none duplicated.

| ID | Sev | Claim |
|---|---|---|
| **repo-1** | blocking | The two stores hold the orography sidecar under **different filenames** (`orography.nc` in wf1 vs `{clim_source}_orography.nc` in wf3); B1's tree draws one. Either choice breaks a declared rule I/O, and because the sidecar exists only on the chirps branch while the seed config is era5, no gate in the repo can see it |
| **arch-3** | major | The region-vs-staticmaps bbox change is an unlisted **third exception** to the behaviour-preservation stance, reaching `clim_wflow_1_*` figures whose only comparators are size-only with a 10% band — see § Conflicts |
| **risk-5** | major | No migration procedure for **existing production `project_dir` trees**. Post-R07 wf3 against a pre-R07 tree raises `MissingInputException` at the guard rule; old directories become untracked orphans outside any `.gitignore`. Asks whether the CST-API/GUI reads artifact paths |
| **risk-7** | major | B6's safety argument is wrong on two checkable points (`precip_variance` denormalised nowhere; `.iloc[0]` takes the **January** row, not a member scalar) and the move relocates an **undeclared runtime input** (`export_wflow_results.py:161`) that `--dry-run` cannot see |
| **repo-4** | major | The `"None"` sentinel risk is **misdiagnosed at both cited lines** — both short-circuit on `is not None`, so `null` raises nothing. The real defect is the string reaching `plot_map.py:28` → layer `gauges_None`, which is drive-by O-08. The milestone's only runtime assertion would pass while proving nothing |
| **repo-7** | major | B8's `basename(project_dir)` is not guaranteed to satisfy `^[a-z0-9][a-z0-9_]*$`; `examples/Gabon` is live in six configs and would raise `ValueError` at parse time |
| **repo-8** | major | The O-22 verification claim is false — **no test in `tests/test_cli.py` asserts on output text**; the three CLI tests assert only `returncode == 0`. O-22 would ship with zero coverage |
| **repo-9** | major | `inmaps_rlz_*_cst_*.nc` are filed under `weather_generator/output/` but are **wflow-grid downscaled forcing** from rule 3.09 — the per-realization twin of the artifact B2 moves *into* the wflow subtree. P3 violated in the one place the design also fixes it |
| **repo-10** | minor | O-24's undeclared-output inventory is complete only for the seed config (`plot_basavg`, `plot_signatures`, per-station `clim_{station}_{period}.png` are config-dependent) |
| **repo-12** | minor | O-22's exemption cannot work as described — `snake_utils.py` has no notion of repo root; proposes `warn_if_project_dir_in_repo(project_dir, repo_root)` with `workflow.basedir` passed from each Snakefile |

---

## Conflicts between lenses — for the author to adjudicate, not the driver

**1. B1's numerics (arch-3 vs the risk lens's explicit non-finding).** The
architecture lens files the bbox change as a `major` unlisted third exception to
the behaviour-preservation stance. The risk lens checked the same question and
**deliberately declined to raise it**, recording its evidence: `prep_historical_climate`
passes `buffer=1` (one *source* cell, ≥0.05° chirps / 0.25° era5) against a
staticmaps-vs-region bbox shift of ≤ ~2 model cells, and
`dev/milestones/p32a/climate-analysis-design.md:380-388` records an empirical `allclose`
closure on the fixture; separately, rule 1.08 builds `inmaps_historical.nc` via
`hydromt update` from the catalog rather than from the extraction, so B1 cannot
move `run_default/output.csv`. The risk lens's conclusion is that B1's problem is
producer assignment, not numerics — and that the design mis-identifies which.
Both readings stand as filed; arch-3's suggested fix (put the collapsed
`extract_historical.nc` on the element-wise `compare_nc` path and check whether the
two extractions land on identical coordinate arrays) is a cheap empirical test
that would settle it either way.

**2. Three severity divergences**, all preserved unchanged: group D (major/major/minor),
group G (major/minor), group H (major/minor), group I (major/minor/minor). The
author dispositions each original ID at its filed severity; any change needs a
logged ledger entry naming who changed it and why.

**3. Three incompatible resolutions for group A** (table above). These differ in
milestone scope, so the choice is an owner decision at the gate, not an author
decision in revision.

---

## Claims the panel verified as correct

Recorded so the author does not re-check them. From the repo-fit lens unless noted:

- 31 `script:` directives (11/10/10) and three `sys.path.insert` calls — the
  Snakefile-move rejection cost is accurate.
- `README.rst:269,285,298`, six notebook cells, `run_snake_test.cmd:32`,
  `.gitignore:124,136` — O-02's inventory is accurate.
- `docs/config/` holds exactly 16 files; `data/` is 653 KiB in two CSVs referenced
  only by the Linux config.
- CI baselines 386/30/1 (windows), 385/31/1 (ubuntu) are correct.
- B2: `path_forcing` is the only TOML pointer; full edit set is five places. The
  new pointer `forcing/inmaps_historical.nc` is a strictly better shape than
  today's `../climate_historical/…`.
- B6: `Qstats.csv` is `statistic,tavg,prcp,Q_130000086` and `basin.csv` is exactly
  `tavg,prcp` — the denormalisation argument holds (but see risk-7 on variance).
- B8's grammar claim is correct; the derivation feeding it is not (repo-7).
- O-24's core claim is correct (inventory incomplete — repo-10).
- Nothing in the milestone touches the forbidden set (`pixi.lock`, `Manifest.toml`,
  `Project.toml`, `.pixi/`, vendored packages). The Julia project root stays the
  repo root regardless of `project_dir`.
- The new identifiers and paths are `naming.md`-conformant.
- **Architecture lens:** the two-tier `project_dir` rule, B8's rejection of a
  runtime-generated `experiment_id`, the P3 rewrite, and the B6 withdrawal on
  `Qstats.csv` evidence are all sound. The "batch both halves for one re-record"
  argument is correct and should not be disturbed — arch-9 attacks the *machinery*
  sequencing, not the single re-record.

## Scope check

No finding proposes re-engineering hydromt, hydromt_wflow, or Wflow. arch-7
resolves the `hydrology_model/` shape deviation *in favour of* the upstream
`model_root` contract by stating an exemption rather than reshaping the directory;
arch-1's fix keeps `get_region_preview.py` inside `blueearth_cst/`. The
`AGENTS.md` Hard Constraint holds across all 34 findings.

---

# Internal lens reviews — verbatim


## Risk & assumptions (`critical-thinker`)

```yaml
verdict: revise
doc_version: design-v1.md
findings:
  - id: risk-1
    severity: blocking
    section: "What changes — B. The artifacts (`project_dir`)"
    finding: >-
      B1 states "One producer, both workflows consume" but never names the producer,
      and every possible assignment is blocked by constraints the *accepted* P3-2a
      design already recorded — which design-v1 does not engage anywhere, not in B1,
      not in `## Alternatives considered`. `dev/milestones/p32a/climate-analysis-design.md:389-405`
      rejects the single store explicitly and by name ("Why a separate wf1 extraction
      and NOT reuse of the P3-1 keyed store (C3, C4)"): (a) if **wf3** produces it,
      wf1 rules 1.10/1.11 would take an `input:` on a downstream, guard-gated artifact
      that does not exist in a fresh project — `MissingInputException`, and it inverts
      the fixed workflow order in `AGENTS.md` / `scripts/run_workflows.py`; (b) if
      **wf1** produces it, that re-points rule 3.02's output, which P3-2a records as
      breaking wf3's value-identity, and orphans both the `.guard_ok` co-location and
      the §3d input-set-invariance argument that rule 3.00b/3.02 depend on
      (`Snakefile_climate_experiment`, rule `check_project_consistency` outputs
      `sentinel` + `guard_ok = {store_dir}/.guard_ok`). Worse, (b) is
      self-contradicting: B1's stated justification is P4 ("the store must be
      buildable without wf1"), yet making wf1 the producer means the store *requires*
      wf1. The only assignment that satisfies P4 is a third, workflow-independent
      producer — i.e. the standalone climate-analysis Snakefile that design-v1 lists
      under "Explicitly out of scope." Note that B4 does **not** dissolve wf1's
      dependency on the store and so does not rescue the wf3-produces branch: O-24
      keeps rule 1.11 `plot_results` declaring `clim_wflow_1_{month,year}.png`, the
      design's own tree files them under `hydrology_model/evaluation/plots/`, and rule
      1.11's climate inputs (`_wf1_plot_clim_inputs` in `Snakefile_model_creation`)
      remain the extraction. wf1 still consumes the store after B4.
    rationale: >-
      B1 is the anchor of the artifact half (B2 and P4 both derive from
      "climate_historical/ becomes purely generic"), and as specified it is not
      implementable inside R07's declared scope. An implementer following commit 6
      will discover this at DAG-parse time, mid-milestone, with commits 7-11 stacked
      behind it. The design also reverses a prior accepted design's reasoned decision
      without citing or rebutting it, which is the failure mode the `dev/r0#/` decision
      -record convention exists to prevent.
    suggested_fix: >-
      Name the producer explicitly in B1 and rebut `dev/milestones/p32a/climate-analysis-design.md`
      §"Why a separate wf1 extraction" point by point. If no in-scope producer works,
      either (i) pull the minimal standalone extraction rule into R07 scope and say so,
      or (ii) drop B1 to a follow-on milestone and keep B2/B3/B5-B8 — B2 does not
      depend on the stores being collapsed, only on the forcing moving.
  - id: risk-2
    severity: blocking
    section: "Verification plan"
    finding: >-
      The plan's sole proof of baseline preservation for B1-B7 is a full-`project_dir`
      `semantic_tree_diff.py` pre/post run "with the R07 path map." That tool cannot
      express B1. `diff_trees()` keys the reference tree by mapped relpath and
      **raises `ValueError("path map collision: … both map to …")`** when two reference
      files map onto one target (`dev/scripts/semantic_tree_diff.py`, the `translated`
      loop in `diff_trees`). B1 is precisely a many-to-one merge: pre-R07 the reference
      tree contains both `climate_historical/wf1_raw/extract_historical.nc` and
      `climate_historical/<key>/extract_historical.nc` (and, on the chirps branch, two
      `orography.nc`), and any prefix rule `climate_historical/wf1_raw/ ->
      climate_historical/<key>/` collides them. Separately, the path map is not
      parameterized: it is hardcoded milestone code (`build_p31_path_map()` /
      `build_p31_allowlist()`), and `main()` exposes only `--experiment-name`,
      `--dataset-key`, `--no-path-map`, `--allow` — there is no way to supply an R07
      map without writing new code, which the commit plan defers to commit 12.
    rationale: >-
      The verification stage that is supposed to certify B1-B7 aborts with a
      `ValueError` before emitting a report, and it cannot be run at all until commit
      12 lands. The design lists only "directory-prefix path map and its path-aware
      TOML comparator" as machinery to update — it does not identify that B1 requires a
      *new comparison semantic* (one-to-many: the single surviving store must be
      compared against **both** reference stores, and the second comparison recorded,
      not allowlisted away).
    suggested_fix: >-
      Add the many-to-one requirement to the "Machinery to update" list, and specify
      the intended semantic (compare the survivor against each collapsed source; a
      merge is only proven by two passing comparisons, not by allowlisting one as
      MISSING). Also move the R07 path map to an early commit, or add a generic
      `--map old=new` CLI option, so the gate exists before the moves it must police.
  - id: risk-3
    severity: major
    section: "Commit plan"
    finding: >-
      The milestone disables its own primary regression gate across exactly the commits
      most likely to change a value. `dev/baseline/manifest.json` stores
      `"project_dir": "examples/test_local"` and full-path target keys; commit 3
      (O-20, `examples/` -> `test_case/`) invalidates every key, so `check_baseline`
      fails by construction from commit 3 through commit 12. The value-touching commits
      — 6 (B1), 7 (B2), 9 (B5/B6/B7), 10 (B4 + O-24) — all land inside that window, and
      the substitute gate is specified once, as a lump, at the "Output-tree moves
      (B1-B7)" row rather than per commit (and per risk-2 is unavailable until commit
      12 anyway). The final re-record at commit 12 therefore cannot distinguish an
      intended path move from an unintended value change; it enshrines whatever is on
      disk. The stated cut-line compounds this: § Risks says "if it must be cut, the
      artifact half (B1-B8) is the coherent unit; the repository half can be deferred"
      — but the commit plan lands the repository half **first** (commits 1-5), so a
      mid-flight stop leaves the deferrable half done and the coherent half half-done,
      with the baseline dead and no gate to restart from.
    rationale: >-
      A silent numerical regression introduced in commit 6 or 9 is undetectable until
      after the re-record, at which point it becomes the new reference. That is the
      exact failure the baseline machinery exists to prevent, and the sole stated
      rationale for batching thirteen commits ("one re-record instead of two") is what
      creates the blackout.
    suggested_fix: >-
      Capture a full-tree reference snapshot of the fixture `project_dir` **before**
      commit 1, land the R07 path map early, and require a path-mapped
      `semantic_tree_diff` pass after each of commits 6, 7, 9, 10 individually. Then
      either reorder so O-20 lands last (keeping `check_baseline` alive through the
      artifact half), or state plainly that `check_baseline` is expected red for N
      commits and name the substitute gate per commit.
  - id: risk-4
    severity: major
    section: "Principles"
    finding: >-
      P1 and P3 are described as invariants ("Four invariants. Most of the tree follows
      from them") but the tree in the same document violates both, which means they
      cannot be used to decide the cases R07 actually has to decide. P3 asserts "Every
      engine subtree has the same internal shape: `config/`, `output/`, `plots/`,
      `_work/`" — yet `hydrology_model/`, the flagship engine subtree, has **none** of
      the four (it has `forcing/`, `run_default/`, `evaluation/`), and
      `hydrology_runs/rlz_<r>/` has two of four. Consequently P3 gives no answer for the
      generated wflow build configs: `wflow_build_model_run.yml` (produced by rule 1.02,
      consumed by 1.03) and `wflow_build_forcing_historical.yml` (produced by 1.07,
      consumed by 1.08) are engine-shaped *generated* artifacts, but the design files
      them in project-level `config/templates/` beside verbatim template snapshots —
      conflating "copy of a shipped template" with "generated build config" and
      contradicting P3's own placement rule. P1 ("figures attach to their producer")
      fails likewise: `basin_area.png` is produced by rule 1.12 from
      `staticgeoms/outlets.geojson` and has nothing to do with evaluation, yet lands in
      `hydrology_model/evaluation/plots/`. `evaluation/` is a topic bucket, not a
      producer.
    rationale: >-
      If the principles do not decide the boundary cases, they are a post-hoc
      description of a chosen tree rather than a decision procedure — and the design's
      stated goal is a layout "governed by stated principles rather than accretion."
      The practical cost is that the next person adding an artifact gets no answer and
      accretion resumes, which is the problem R07 exists to end.
    suggested_fix: >-
      Either weaken P3's second sentence to a *permitted* shape rather than a universal
      one and say why `hydrology_model/` is exempt (hydromt owns `model_root`'s
      immediate children), or restructure it to comply. Give the generated build
      configs an explicit home and state the rule that put them there. Re-word P1 as
      "figures attach to the subtree they describe" if that is what is meant, or move
      `basin_area.png`.
  - id: risk-5
    severity: major
    section: "Behaviour-preservation stance and baseline consequence"
    finding: >-
      The design defines no migration procedure for existing production `project_dir`
      trees, and its success criteria are entirely fixture-scoped. The concrete failure
      is deterministic: after the `config/` -> `config/runs/` split, rule 3.00b
      `check_project_consistency` in `Snakefile_climate_experiment` declares
      `input: wf1_snapshot = ancient(wf1_snapshot_path)` where `wf1_snapshot_path =
      f"{project_dir}/config/snake_config_model_creation.yml"` — a *mandatory* rule
      input. Running post-R07 wf3 against any pre-R07 `project_dir` raises
      `MissingInputException` at the guard, before anything else. Everything else in
      the tree is worse-behaved rather than better: with 17 of 18 fingerprinted paths
      moved, Snakemake sees the new paths as absent and re-derives them, leaving the
      old `wf1_raw/`, `wflow_data/`, `stress_test/`, `realization_*/`, `model_runs/`,
      `model_results/` directories as untracked orphans in a tree that by design lives
      **outside** the repository and is therefore covered by no `.gitignore` and no
      cleanup. `MIGRATION.md` and `dev/milestones/r07/migration_project-layout.md` are both
      repository-side documents; neither is stated to cover artifact trees. Separately
      and less certainly: `AGENTS.md` declares this repo the workflow engine of a
      three-part platform with a CST-API backend and a CST-frontend GUI, and the
      design's own `## Alternatives considered` invokes a GUI globbing
      `**/plots/*.png` — so the doc assumes an artifact-path consumer exists while
      saying nothing about coordinating a wholesale rename with it.
    rationale: >-
      The milestone can pass every stated success criterion while leaving the only real
      `project_dir` unusable, and the first symptom a user sees is an opaque
      `MissingInputException` from a guard rule, not a migration message. Whether the
      downstream API/GUI reads artifact paths is a question only the owner can answer;
      the design should answer it rather than leave it implicit.
    suggested_fix: >-
      Add a short "Migrating an existing project_dir" section: either (i) declare
      pre-R07 trees unsupported and require a fresh run (cheap to state, expensive to
      execute — say which), or (ii) ship the path map as an executable `mv` script and
      name the orphan directories to delete. State explicitly whether any downstream
      consumer reads `project_dir` paths, and if so what the coordination is.
  - id: risk-6
    severity: major
    section: "Behaviour-preservation stance and baseline consequence"
    finding: >-
      The baseline arithmetic is wrong in a way that will surface at the exit
      adjudication. `dev/baseline/manifest.json` has 18 target entries but
      `dev/scripts/check_baseline.py` `TARGETS` has only 15 templates; three manifest
      entries — `climate_experiment/model_results/Qstats.csv`,
      `climate_experiment/model_results/basin.csv`, and
      `config/snake_config_climate_experiment.yml` — are **pre-P3-1 legacy paths with
      no current producer** (mixed-provenance manifest). So "Of the eighteen targets"
      counts three dead entries. The follow-on claim "The four copied-config snapshots
      change *content* … **and** *path*" also contradicts the design's own tree: that
      tree puts exactly **two** files in `config/runs/`, the experiment-level
      `experiments/<id>/config/snake_config_climate_experiment.yml` keeps its path
      (content only), and the third project-level snapshot is the legacy entry.
    rationale: >-
      The design's exit test is that the pre/post manifest diff is
      "path-and-snapshot-only, adjudicated by the normalize-then-compare policy."
      That adjudication will instead see three unexplained **deletions** plus a
      path-change count that does not match the doc, and whoever runs it will either
      allowlist them blind or spend a review round rediscovering the mixed-provenance
      state that is already on record.
    suggested_fix: >-
      Correct the counts (15 live / 3 legacy), and either purge the three legacy
      entries in a separate pre-R07 re-record so the R07 diff is clean, or list them
      as expected deletions in the migration map.
  - id: risk-7
    severity: major
    section: "What changes — B. The artifacts (`project_dir`)"
    finding: >-
      B6's justification is wrong on two checkable points, and the move it proposes
      relocates an **undeclared runtime input**. (1) `cst_*.csv` carries three axes —
      `prepare_cst_parameters.py` writes `temp_mean`, `precip_mean` **and
      `precip_variance`** — while the indicators carry two (`tavg`, `prcp`). The
      variance axis is denormalised nowhere. (2) "a scalar per member" understates the
      reduction: `export_wflow_results.py:162-163` takes `df_st["temp_mean"].iloc[0]`
      and `df_st["precip_mean"].iloc[0]` — the **January** row, not a member-level
      scalar — so the monthly-structure loss the doc frames as hypothetical is already
      live. (3) Most consequential: `cst_*.csv` is read at run time from a path
      *constructed inside the script* (`export_wflow_results.py:161`,
      `f"{exp_dir}/stress_test/cst_{st_nb}.csv"`), while rule 3.11
      `export_wflow_results` declares only `rlz_csv_fns` as `input:`. Moving the
      directory therefore breaks a dependency that Snakemake does not know about and
      `--dry-run` cannot see — the design's own caveat that dry-run is blind to
      `params:`-string paths applies here and B6 is not in the list of moves flagged as
      needing a real run.
    rationale: >-
      The same milestone that fixes undeclared *outputs* (O-24, on the grounds that
      undeclared artifacts are not cleaned and are absent from the baseline) silently
      relocates an undeclared *input*, and its safety argument rests on a
      denormalisation that is incomplete (variance) and lossier than claimed
      (January-only). If the string at line 161 is missed, wf3 fails at rule 3.11 after
      the full RLZ x ST run has completed — the most expensive possible place to
      discover it.
    suggested_fix: >-
      Correct the caveat to name `precip_variance` and the `.iloc[0]` month-1
      reduction. Declare `cst_*.csv` as a real `input:` on rule 3.11 while moving it
      (the paths are already enumerable from `ST_NUM`), and add B6 to the "needs a real
      run, not a dry-run" list beside B4/B5.
  - id: risk-8
    severity: major
    section: "Risks and open questions"
    finding: >-
      The G1 ruling that `MIGRATION.md` moves to `docs/` (audience = users) creates two
      classes of migration document with no stated rule for which class a new note
      joins, and design-v1 does not address the divergence at all.
      `dev/conventions/naming.md` §7 requires "a `dev/<milestone>/migration_<topic>.md`
      note" for renames of `rule all` output filenames and of "Test fixture paths read
      by `tests/conftest.py`, `dev/scripts/check_baseline.py`" — R07 triggers both — and
      the design duly files its map at `dev/milestones/r07/migration_project-layout.md`. But by the
      ruling's own audience test, R07's map is *more* user-facing than `MIGRATION.md`:
      it maps production `project_dir` paths that users' own artifact trees sit on
      (risk-5). The design therefore places a user-facing map under `dev/` in the same
      milestone that moves a user-facing map out of it, on opposite reasoning, and
      records neither.
    rationale: >-
      The ruling explicitly requires the design to state the §7 divergence and either
      reconcile §7 or record an exemption. v1 does neither, so the milestone lands a
      convention conflict that the next migration note inherits with no rule to resolve
      it — and R07 is itself the second instance of the ambiguity.
    suggested_fix: >-
      Add a line to §7 (or an explicit exemption in the design) distinguishing the two
      classes: dev-facing rename maps stay at `dev/<milestone>/migration_<topic>.md`;
      user-affecting migrations — anything that moves paths inside a user's
      `project_dir` — go to `docs/`. Then say which class `dev/milestones/r07/migration_project-layout.md`
      is, given that it is partly both.
  - id: risk-9
    severity: minor
    section: "What changes — B. The artifacts (`project_dir`)"
    finding: >-
      The "Two `precip.png`" risk understates its own scope, and it compounds with the
      "Two PET values" risk rather than sitting beside it. O-24 records that
      `plot_map_forcing.py` writes **three** PNGs (precip / temp / pet); the new B4
      climate producer will write the analogous set from the same variables. So the
      collision is a *set* of identical basenames across sibling `plots/` trees, and one
      of them is `pet.png` — the exact pair the two-PET risk says someone will compare
      and report as a defect. Relying on the parent directory to disambiguate means the
      two divergent PET figures are published under the same filename. Secondarily, the
      tree annotates `orography.nc` as a "chirps-branch sidecar", so on the era5 branch
      the store carries no orography; the new source-grid PET producer must fetch
      `era5_orography` from the catalog (as `extract_historical_climate.py` /
      `climate_parity.py` already do), which is compatible with P4's "region + catalog
      alone" but is not in B4's stated input list.
    rationale: >-
      Two files named `pet.png` holding deliberately different values is the most
      confusable outcome the milestone can produce, and the design's mitigation ("the
      figures must say approximate on their face") does not survive a file being copied
      out of its directory — into a report, a GUI collector, or a chat message.
    suggested_fix: >-
      Prefix the source-grid set (e.g. `source_precip.png` / `source_pet.png`) rather
      than relying on the parent directory, and add the catalog dependency to B4's
      input list so the P4 assertion test is specified correctly.
  - id: risk-10
    severity: minor
    section: "Commit plan"
    finding: >-
      The test suite's own hardcoded path bindings are not in the "Machinery to update"
      list nor in any commit. `tests/test_model_creation.py:26-28` asserts three exact
      paths — `{project_dir}/config/snake_config_model_creation.yml`,
      `{project_dir}/config/wflow_build_model.yml`,
      `{project_dir}/config/tests_data_catalog.yml` — all of which move under the
      `config/{runs,catalogs,templates}` split, and `tests/test_interchange_contracts.py`
      binds `_FIXTURE = join(SNAKEDIR, "examples", "test_local")` (line 39) plus
      `climate_historical/<key>/extract_historical.nc` (484) and
      `climate_historical/wflow_data/inmaps_historical.nc` (529), which O-20, B1 and B2
      move respectively. The commit plan mentions `tests/` only as a "config fix" in
      commit 1 and lists `check_baseline.py` in commit 3.
    rationale: >-
      `intake.md` names "`pytest tests/` green" as a success criterion and the commit
      plan claims each commit leaves the tree runnable; as sequenced, commits 3, 6 and
      7 land red tests. The design also asserts CI counts must not move (386/30/1), an
      assertion no commit is tasked with maintaining.
    suggested_fix: >-
      Add `tests/` path bindings to the "Machinery to update alongside" list and name
      the affected test files in commits 3, 6 and 7.
```

## Notes on what I did *not* raise

- The four G1-settled questions are treated as settled; risk-8 raises only the
  downstream inconsistency ruling 2 creates, as directed.
- **Grid equality under B1 is *not* a finding.** I checked it and the design is
  defensible: `prep_historical_climate` passes `buffer=1` (one *source* cell, ≥0.05°
  chirps / 0.25° era5) against a staticmaps-vs-region bbox shift of ≤ ~2 model cells,
  and `dev/milestones/p32a/climate-analysis-design.md:380-388` records an empirical `allclose`
  closure on the fixture. Discharge is likewise safe: rule 1.08 builds
  `inmaps_historical.nc` via `hydromt update` from the catalog, not from the
  extraction, so B1 cannot move `hydrology_model/run_default/output.csv`. The B1
  problem is producer assignment and workflow order (risk-1), not numerics — and the
  design mis-identifies which one it is.
- The five named risks in § Risks and open questions are not restated; risk-3 adds the
  gate-blackout consequence the scope risk misses, risk-9 adds the compounding the
  `precip.png` and PET risks miss, and the TOML-pointer risk I judged adequately
  covered (though note B2 also changes `path_forcing` — three literal occurrences in
  `Snakefile_model_creation` plus `blueearth_cst/shared/setup_time_horizon.py:51` plus
  `tests/test_interchange_contracts.py:529` — so "the only pointer edit is
  `path_forcing`" is true of the TOML key but understates the edit surface).
- Test-suite path bindings are filed as `risk-10` (minor).

## Architecture & internal consistency (`cst-architect`)

```yaml
verdict: revise
doc_version: design-v1.md
findings:
  - id: arch-1
    severity: blocking
    section: "The substantive moves"
    finding: >-
      B1 collapses the two climate stores to "one producer, both workflows consume" but
      never names which Snakefile owns the surviving producer rule, and every available
      answer breaks a stated commitment. The repo's established idiom for a cross-workflow
      artifact is wf1-produces / downstream-consumes-via-`ancient()`
      (`Snakefile_climate_projections:90,114`; `Snakefile_climate_experiment:197`), so the
      default reading is that `climate_historical/<key>/` keeps a `hydrology_model/` input.
      Under that reading the verification plan's B4 row — "New figures produced from
      `<key>/extract_historical.nc` with **no** `hydrology_model/` present — the P4
      assertion" — is false, because the store's own producer cannot run without
      `hydrology_model/staticgeoms/region.geojson` (today's wf3 rule 3.02 input) or
      `staticmaps.nc` (today's wf1 rule 1.10 input, `extract_climate_wf1.py:59`). The other
      two horns fail differently: if wf3 owns the producer, wf1 rule 1.11 `plot_results`
      (`Snakefile_model_creation:261-269`) consumes a wf3 output and a wf1-only run dies
      with `MissingInputException`, inverting the documented wf1→wf2→wf3 order; if both
      Snakefiles declare it, P2 is breached. `get_region_preview.py`, cited as the escape,
      is a standalone `argparse` CLI wired into no rule, with its own hydrography default
      (`merit_hydro_ihu`) and an output that concatenates river geometries into
      `region.geojson` — adopting it is a new rule, not a path move, and the design specs
      neither the rule nor its inputs.
    rationale: >-
      The P4 assertion is a named success criterion in `intake.md` and the headline
      verification row for B4. As specified, the milestone cannot demonstrate its own exit
      criterion, and the implementer has no way to choose among three resolutions that
      break the workflow order, P2, or the behaviour-preservation stance respectively.
    suggested_fix: >-
      Name the owning Snakefile and the producer rule's exact inputs. If the model-free
      route is intended, spec it as a first-class change: a new rule taking `region` +
      catalog (not `staticgeoms/region.geojson`), with `hydrography_fn` pinned to whatever
      `config/templates/wflow_build_model.yml` `setup_basemaps` uses, and move it out of the
      "no computational path changes" list. If it is not intended, restate the P4 assertion
      as what R07 actually proves (the layout does not obstruct a future model-free store)
      and drop the "no `hydrology_model/` present" verification row.

  - id: arch-2
    severity: blocking
    section: "Verification plan"
    finding: >-
      The headline output-tree gate — "`semantic_tree_diff.py` full-`project_dir` pre/post
      comparison with the R07 path map: MISSING/EXTRA empty modulo a written allowlist" —
      cannot be expressed for B1, because `semantic_tree_diff.diff_trees` is strictly
      one-to-one and hard-fails on many-to-one. Lines 641-647 raise
      `ValueError("path map collision: … both map to …")` when two reference relpaths
      translate to the same key. B1 is a many-to-one collapse: the reference tree holds both
      `climate_historical/wf1_raw/extract_historical.nc` (`Snakefile_model_creation:231-232`)
      and `climate_historical/<key>/extract_historical.nc`, and any rule mapping `wf1_raw/`
      onto `<key>/` makes them collide (plus `orography.nc` on the chirps branch,
      `Snakefile_model_creation:233-234`). The tool raises before it can report, so the gate
      aborts rather than passing or failing. That the mapping-rule route is the intended one —
      rather than the allowlist route — is visible in the derived migration map
      (`dev/milestones/r07/migration_project-layout.md:154`, "**MISSING** — none by design"), cited here
      as evidence of authorial intent, not reviewed.
    rationale: >-
      The single most-cited proof of behaviour preservation in the milestone crashes on the
      single most substantive move. The only escape without touching the tool — omitting the
      rule and allowlisting `wf1_raw/*` as MISSING-by-design — means the gate proves nothing
      about the store that disappeared, which is precisely what arch-3 says needs proving.
    suggested_fix: >-
      Add a machinery item alongside the path-map update: either teach `diff_trees` a
      declared many-to-one merge class (reference file X and Y both compared against the same
      current file, both required to pass) or add an explicit `--retire <relpath>` set
      excluded from translation, and state which of the two retired files is content-compared
      against the survivor.

  - id: arch-3
    severity: major
    section: "Behaviour-preservation stance and baseline consequence"
    finding: >-
      "No computational path changes … the two exceptions, both additive" lists only the new
      climate plot producer and the parse-time warning. B1's bbox ruling is an unlisted third
      exception. Today wf1's store is cut to the staticmaps bounds
      (`extract_climate_wf1.py:29-37`, whose own docstring records that each edge "can sit up
      to ~one model cell outside the region's tight bounds"), while the wf3 store is cut to
      `region.geometry.total_bounds` (`extract_historical_climate.py:88-91`). Ruling the
      collapsed store "region-derived" therefore changes the grid extent fed to
      `climate_parity.model_parity_climate`, whose `nearest_index` reprojection onto
      `dem_model` can draw different source cells at the basin margin. The affected artifacts
      are the `clim_wflow_1_{month,year}.png` figures — and both available comparators for
      PNGs are size-only with a 10% band (`check_baseline.diff_png`,
      `PNG_TOLERANCE_FRAC = 0.10`; `semantic_tree_diff.compare_png` delegates to it).
    rationale: >-
      The verification plan promises "every value identical" for the B1–B7 stage, but for the
      exact artifacts B1 can perturb the gate is a file-size check with a 10% tolerance. A
      real change in the climate figures would pass silently and then be frozen into the
      single end-of-milestone re-record, with no earlier commit to bisect against.
    suggested_fix: >-
      Add the collapsed `extract_historical.nc` to the element-wise `compare_nc` path so the
      store itself, not the downstream PNGs, carries the proof. Note that the two *bboxes*
      cannot be equal by construction (the staticmaps bounds snap outward), so the check to
      run is whether the two extractions land on identical coordinate arrays on the seed
      fixture — `get_rasterdataset(..., buffer=1)` adds a source cell on each side and may
      absorb the difference. If they do not match, list the extent change as a third named
      exception to the behaviour-preservation stance and declare `clim_wflow_1_*` as
      expected-to-move.

  - id: arch-4
    severity: major
    section: "Verification plan"
    finding: >-
      The design's blind-spot concession — "`--dry-run` is blind to `params:`-string paths and
      to R `shell:` bodies" — understates the surface. The paths that move are hardcoded
      inside `script:`-directive Python modules and inside the test suite, neither of which is
      a `params:` string or an R body: `plot_results.py:108`, `plot_map.py:34`, and
      `plot_map_forcing.py:201` each hardcode
      `f"{project_dir}/plots/wflow_model_performance"`; `export_wflow_results.py:161`
      hardcodes `stress_test/cst_{n}.csv` (B6) and `:281` hardcodes `model_results` (B7);
      `setup_time_horizon.py:51` hardcodes the `path_forcing` string (B2);
      `downscale_climate_forcing.py:72` computes a relative prefix against `model_runs/`
      depth (B5); `generate_weather.R:68` builds `realization_<n>/` (B5). Separately, the
      moved paths are hardcoded in at least seven test modules —
      `tests/test_extract_climate_wf1.py:24,26`, `tests/test_interchange_contracts.py:39,484,529,570`,
      `tests/test_check_baseline_scope.py:131,160`, `tests/test_semantic_tree_diff.py:332-388`,
      `tests/test_workflow_climate_experiment.py:114`, `tests/test_guard_invalidation.py:97`,
      `tests/test_check_project_consistency.py:30` — none named in the commit plan, which
      mentions `tests/` only as a "config fix" in commit 1. The `dev/contracts/*-seam.md`
      documents pin the same paths (`hydrological-model-seam.md:74,353`,
      `weather-generator-seam.md:56,71,248,294`) and are also unlisted.
    rationale: >-
      Two observable consequences. First, a rule's declared `output:` and its script's write
      path can diverge, and O-24 (declaring the currently-undeclared PNGs) makes the divergence
      load-bearing rather than cosmetic. Second, the verification plan's own B1–B7 gate says
      "`pytest tests/` green" while the test suite encodes the pre-move paths, so that gate is
      unsatisfiable as sequenced unless the test updates ride in the same commits — which no
      commit assigns.
    suggested_fix: >-
      Replace the concession with a contract inventory table: for each move, list the
      Snakefile rule, the script-module constant, the test modules, and the seam doc that
      must change together, and attach each row to its commit.

  - id: arch-5
    severity: major
    section: "What changes — B. The artifacts (`project_dir`)"
    finding: >-
      `hydrology_model/evaluation/` appears in the tree (holding `performance_metrics.csv` and
      `plots/hydro_wflow_1, basin_area, clim_wflow_1_*`) but has no entry in B1–B8 and no
      commit in the 13-commit plan. This move retires
      `{project_dir}/plots/wflow_model_performance/` — the home of three of the manifest's
      wf1 targets and the outputs of three distinct rules (1.11 `plot_results`, 1.12
      `plot_map`, 1.13 `plot_forcing`, `Snakefile_model_creation:265-315`) plus rule 1.14's
      gather inputs (`:324-326`). B4 covers only rule 1.13's redirection to
      `hydrology_model/forcing/plots/`; nothing covers 1.11, 1.12, or 1.14.
    rationale: >-
      An implementer working the commit plan has no instruction to make the largest wf1 move
      in the milestone, and no commit boundary at which the three `Folder_plots` constants
      (arch-4) and rule 1.14's hardcoded gather inputs get repointed. Left unassigned, either
      the move silently rides inside commit 10 (whose title is about climate figures) or it is
      omitted and the delivered tree does not match the design's own diagram.
    suggested_fix: >-
      Add a B-item ("wf1 evaluation outputs move into the engine subtree") naming rules 1.11,
      1.12, 1.13, 1.14, the three module constants, and `performance_metrics.csv`; assign it
      an explicit commit.

  - id: arch-6
    severity: major
    section: "The substantive moves"
    finding: >-
      B4's product table enumerates two figure families (source-grid "climate figures" and
      model-grid "forcing / model-input QA figures") but the tree shows three: the
      `clim_wflow_1_{month,year}.png` pair, produced by rule 1.11 `plot_results` from the raw
      extraction at *model parity* (`climate_parity.py`, `Snakefile_model_creation:254-263`),
      lands in `hydrology_model/evaluation/plots/` and is named nowhere in B4. The design's
      "two `precip.png`" risk therefore understates the situation: after B4 the project holds
      three climate-figure families on two different grids from two different producers, with
      no rule saying which is authoritative for a reader or a GUI collector, and no statement
      of how the new source-grid family relates to `clim_wflow_1_*`.
    rationale: >-
      This adds a consequence the named risk misses. The stated goal is that "a reader can
      tell what made a file by where it sits"; three climate-figure families whose only
      discriminator is their parent directory, two of them documented and one undocumented,
      defeats it. It also leaves open whether `clim_wflow_1_*` is now redundant with the new
      producer and should be retired — a decision the design took explicitly for the
      forcing/QA figures but not for these.
    suggested_fix: >-
      Extend B4's table to three rows (source-grid climate, model-parity climate
      `clim_wflow_1_*`, forcing/model-input QA), state the question each answers, and decide
      whether `clim_wflow_1_*` survives. If all three survive, add the disambiguating-filename
      decision the "two `precip.png`" risk defers.

  - id: arch-7
    severity: major
    section: "Principles"
    finding: >-
      P3 states without qualification that "Every engine subtree has the same internal shape:
      `config/`, `output/`, `plots/`, `_work/`", and the tree satisfies it in exactly one
      place. `weather_generator/` has all four. `hydrology_runs/rlz_<r>/` has two (`config/`,
      `output/`). `hydrology_model/` — the largest engine subtree — has none of the four: it
      has loose `wflow_sbm.toml` / `staticmaps.nc` / `staticgeoms/` at its root plus
      `forcing/`, `run_default/`, `evaluation/`. (`climate_projections/<clim_project>/` is a
      further distinct shape but is *not* P3-governed — the design classifies it as generic,
      engine-independent data — so it is noted only because a reader must still learn it.) The
      `hydrology_model/` deviation has a real cause — it is the hydromt `model_root`, and
      `intake.md` pins that as a constraint (option B rejected) — but the design records that
      cause only in the *Alternatives considered* entry for a rejected nesting, never as an
      exemption to P3.
    rationale: >-
      P3 is one of the four invariants the design says the tree "follows from". As written it
      is contradicted by the tree it governs, so it cannot be used to settle the next layout
      question — which is exactly what a decision-record is for. An implementer or a later
      author reading P3 has no way to know that `hydrology_model/` is exempt rather than
      simply not yet conformed.
    suggested_fix: >-
      Restate P3 with the exemption inline: engine subtrees share the four-part shape *except*
      where an upstream tool owns the directory contract, and name `hydrology_model/` as the
      one such case with `model_root` as the reason. Say explicitly that `plots/` and `_work/`
      are optional-when-empty so `hydrology_runs/rlz_<r>/` conforms.

  - id: arch-8
    severity: major
    section: "Goal"
    finding: >-
      The Goal states the artifacts are organised "so … a second modelling engine can be added
      without inventing a new layout", and `intake.md` lists it as a decision criterion. The
      proposed shape does not deliver it. Hydrology appears twice, in two shapes, at two
      levels: `hydrology_model/` at the `project_dir` root (upstream-shaped, no four-part
      structure) and `hydrology_runs/` inside `experiments/<id>/` (two-part). Adding a second
      hydrology engine requires deciding, with no rule to appeal to, which of those two shapes
      to copy, whether the new engine gets a root-level build subtree at all, and how it
      coexists with a name — `hydrology_model/` — that is descriptive-by-domain rather than
      by engine, so a second hydrology engine collides on it. The G1 ruling parks the
      *naming* question, but the structural question is upstream of naming and the design does
      not raise it.
    rationale: >-
      A stated goal that the delivered structure does not support will be discovered at the
      first attempt to add an engine, when the layout is already frozen by a baseline
      re-record and a migration map. Better to state the limitation now than to imply a
      guarantee the tree cannot honour.
    suggested_fix: >-
      Either narrow the Goal to what the tree does deliver ("engine-shaped artifacts are
      separable from generic ones, so an engine can be relocated without moving generic data")
      or add the rule the claim needs: how a second engine's build subtree and run subtree are
      placed and named, noting that this is the structural half of the parked naming question.

  - id: arch-9
    severity: major
    section: "Commit plan"
    finding: >-
      The commit plan's sequencing claim is "each commit leaves the tree runnable", which is
      true but is not the claim that matters. `dev/baseline/manifest.json` keys are fully
      resolved paths beginning `examples/test_local/`, so commit 3's `examples/` → `test_case/`
      rename invalidates all eighteen recorded keys at once; commits 8, 9 and 10 then move
      wf2, wf3 and wf1 targets respectively. The machinery and the re-record land only at
      commit 12. Between commits 3 and 12 — nine of the thirteen — `check_baseline check` is
      red by construction, and the `semantic_tree_diff` path map is likewise not updated until
      12, so the second gate is unavailable too. The design never states this window or how it
      is covered.
    rationale: >-
      For nine commits the milestone runs with no working regression detector, and the single
      end-of-milestone re-record then freezes whatever the tree contains. A value change
      introduced anywhere in commits 6–11 (see arch-3) is undetectable and un-bisectable, and
      is recorded as the new baseline. The single re-record is the right call; the missing
      piece is what substitutes for the gate while it is down.
    suggested_fix: >-
      State the blackout window explicitly and give it a substitute: move the mechanical
      machinery update (TARGETS templates, `PROJECT_DIR_DEFAULT`, the path map) to a commit
      *before* the moves rather than after, so `check --workflow <name>` can be re-run
      per-slice against a pre-milestone reference tree kept on disk; keep only the manifest
      *re-record* at the end.

  - id: arch-10
    severity: major
    section: "Behaviour-preservation stance and baseline consequence"
    finding: >-
      The baseline arithmetic does not reconcile with the artifacts. `manifest.json` holds 18
      keys but `check_baseline.TARGETS` holds only 15 rows; three recorded keys are stale
      pre-P3-1 orphans (`climate_experiment/model_results/{Qstats,basin}.csv` and
      `config/snake_config_climate_experiment.yml`) that a full `record` will silently drop.
      The stated breakdown also mis-scopes: there are three live config snapshots, not four —
      the fourth recorded one is an orphan, and the live wf3 snapshot at
      `experiments/<id>/config/snake_config_climate_experiment.yml` does **not** join the
      `config/runs/` split, since the design's own tree keeps it inside the experiment
      (line 179), contradicting the claim that all four "change … *path* (the `config/runs/`
      split)". And of the "six wf2 summary/plot targets", the three PNGs under
      `climate_projections/<clim_project>/plots/` do not move under B3 at all. The listed
      movers sum to 16, not 18.
    rationale: >-
      The exit gate is that "the diff against the pre-R07 manifest is path-and-snapshot-only,
      adjudicated by the normalize-then-compare policy". An adjudicator comparing pre- and
      post-R07 manifests will see three unexplained key *deletions* and a set of moves that
      does not match the design's inventory, and cannot mechanically conclude
      path-and-snapshot-only. The mis-scoping also hides that the wf3 config snapshot needs no
      repoint.
    suggested_fix: >-
      Re-derive the inventory from `TARGETS` rather than from the manifest file, list the three
      orphan keys as expected deletions in the migration map, correct the wf2 count to the
      three summary files, and drop the claim that the wf3 experiment snapshot moves into
      `config/runs/`.

  - id: arch-11
    severity: minor
    section: "Behaviour-preservation stance and baseline consequence"
    finding: >-
      Two smaller inventory gaps. (a) `semantic_tree_diff.py` carries a *third* path mechanism
      the design does not name: `COPIED_CONFIG_PATH_MAP` (lines 90-110), the config-key →
      {old value: new value} table driving copied-config normalization; the design names only
      the directory-prefix map and the path-aware TOML comparator. If O-01's
      `output_locations` / `observations_timeseries` values or O-05's catalog paths change
      value in any snapshotted config, a new entry is required and the design does not say so.
      Relatedly, `_is_copied_config` (line 576) matches any YAML with a `config` path part, so
      the new `weather_generator/config/weathergen_config.yml` is newly swept into that
      directional policy. (b) The G1 ruling moves `MIGRATION.md` to `docs/` while
      `dev/conventions/naming.md` §7 (lines 141-150) places migration notes under
      `dev/<milestone>/` — and R07 writes its own map to `dev/milestones/r07/migration_project-layout.md`,
      so the repo will hold two migration notes under two conventions. v1 addresses this
      nowhere; it lists the home as an open question and cites §7 only as a requirement it
      satisfies.
    rationale: >-
      (a) is a gate-time surprise rather than a design error, but the inventory is presented as
      complete. (b) leaves `naming.md` §7 with an undocumented exception, so the next milestone
      author has no rule to follow for where a migration note goes.
    suggested_fix: >-
      Add `COPIED_CONFIG_PATH_MAP` to the machinery list with a yes/no on whether R07 needs new
      entries. For §7, state the divergence explicitly and pick one: amend §7 to distinguish
      user-facing migration guides (`docs/`) from milestone path maps (`dev/<milestone>/`), or
      record a named exemption.
```

## Notes

**What holds up well.** The two-tier `project_dir` rule, B8's rejection of a runtime-generated
`experiment_id`, the P3 rewrite after the owner's challenge to the "upstream-governed" claim,
and the B6 withdrawal on `Qstats.csv` evidence are all sound and well-evidenced. The
"batch both halves for one re-record" argument is correct and I would not disturb it —
arch-9 attacks the *machinery* sequencing, not the single re-record.

**Not raised, deliberately.** The five named risks (`"None"` sentinel, TOML relative pointers,
two `precip.png`, two PET values, milestone scope) are restated only where I could add a
consequence they miss: arch-6 extends the `precip.png` risk from two families to three, and
arch-3 shows the TOML-pointer risk has a PNG-comparator sibling the risk list does not cover.
The four G1-settled questions are treated as settled; arch-8 addresses only the structural
half of the parked naming question, and arch-11(b) is the downstream inconsistency ruling 2
creates, which the brief asks for. The weathergen date CSVs in `weather_generator/output/`
are consistent with P3 as the design applies it elsewhere — a generator product, and
`_work/` is reserved for the per-member configs — so no finding.

**Scope check.** No finding proposes changing hydromt, hydromt_wflow, or Wflow behaviour.
arch-7 explicitly resolves the `hydrology_model/` shape deviation *in favour of* the upstream
`model_root` contract, by stating the exemption rather than reshaping the directory. arch-1's
suggested fix keeps `get_region_preview.py` inside `blueearth_cst/` and only pins its
hydrography argument to match the build config.

## Repo fit & conventions (`python-engineer`)

verdict: revise
doc_version: design-v1.md
findings:
  - id: repo-1
    severity: blocking
    section: "What changes — B. The artifacts (`project_dir`)"
    finding: >-
      B1 collapses `climate_historical/wf1_raw/` and `climate_historical/<key>/`
      into one store, but the two stores hold the orography sidecar under two
      DIFFERENT filenames and the design does not reconcile them. wf1's
      `blueearth_cst/model/extract_climate_wf1.py:62-67` relocates the sidecar to
      the stable, clim_source-independent name `orography.nc` (declared as rule
      1.10's `oro_nc` output, `Snakefile_model_creation:234`, and consumed as rule
      1.11's declared input at `Snakefile_model_creation:263`). wf3 writes
      `{clim_source}_orography.nc` (`blueearth_cst/climate_analysis/extract_historical_climate.py:155`)
      and reads it back as a `params:` string at
      `Snakefile_climate_experiment:331` (`oro_path = f"{store_dir}/{clim_source}_orography.nc"`).
      The design's tree (design-v1.md:159) draws the collapsed store with
      `orography.nc`.
    rationale: >-
      Implemented as drawn, rule 3.08 `climate_data_catalog` writes a hydromt
      catalog whose orography entry points at a file that no longer exists on the
      chirps / chirps_global branch; the reverse choice breaks rule 1.10's
      declared output and rule 1.11's declared input. Because the sidecar exists
      only on the chirps branch and the seed / baseline config is `era5`, neither
      `pytest tests/`, `--dry-run`, nor `check_baseline` can see the breakage —
      it surfaces first on a real chirps basin. `oro_path` is a `params:` string,
      which the design itself notes `--dry-run` is blind to.
    suggested_fix: >-
      Pick one sidecar filename for the collapsed store (`orography.nc` is the
      better one — it is already the clim_source-independent form P3-2a ext2-1
      standardized), add the `Snakefile_climate_experiment:331` `oro_path` repoint
      to B1's edit list, and add a chirps-branch check to the verification plan
      (a unit test over `prepare_climate_data_catalog.py` is enough; a chirps run
      is not required).

  - id: repo-2
    severity: blocking
    section: "What changes — B. The artifacts (`project_dir`)"
    finding: >-
      B1 states "One producer, both workflows consume" but never assigns the
      producer to a Snakefile, and both available assignments break a stated
      principle. Today the two extractions have different bbox derivations: rule
      1.10 takes `basin_nc = ancient(f"{basin_dir}/staticmaps.nc")`
      (`Snakefile_model_creation:239`), rule 3.02 takes
      `prj_region = ancient(f"{basin_dir}/staticgeoms/region.geojson")`
      (`Snakefile_climate_experiment:197`). B1 rules the bbox region-derived,
      which already forces rule 1.10's input to change. If wf1 then owns the sole
      producer, a model-free climate analysis cannot produce the store — P4 fails.
      If wf3 owns it, `Snakefile_model_creation` gains a dependency on a wf3
      artifact, inverting the documented model → projections → experiment order
      (AGENTS.md § Key Commands) and making wf1 non-standalone. If both Snakefiles
      keep a rule writing the same path, two rules with different inputs produce
      one artifact across two DAGs — a stale-content hazard Snakemake cannot
      adjudicate.
    rationale: >-
      This is the pivot of the entire artifact half: B4 (climate figures from the
      store), P4 (model-free climate analysis), and the P4 verification assertion
      ("new figures produced with no `hydrology_model/` present") all rest on it,
      and the guard-artifact sharing contract from P3-1 §3a/§3d
      (`{store_dir}/.guard_ok`, `Snakefile_climate_experiment:171,202`) is keyed to
      the store dir too. An implementer handed the design as written must invent
      the answer, and either invention silently breaks an accepted principle.
    suggested_fix: >-
      State the producer assignment explicitly in B1. The only branch that
      satisfies P4 without inverting the workflow order is a producer whose inputs
      are region + catalog only (per `blueearth_cst/model/get_region_preview.py`),
      declared in BOTH Snakefiles as the same rule over the same inputs — then say
      so, and say what happens to rule 1.10's `ancient(staticmaps.nc)` input and to
      rule 3.02's `ancient(region.geojson)` input.

  - id: repo-3
    severity: blocking
    section: Behaviour-preservation stance and baseline consequence
    finding: >-
      "The wf1 discharge target (`hydrology_model/run_default/output.csv`) is
      unchanged" (design-v1.md:286) is false, and the consequence is that the
      milestone's strongest numeric anchor is silently re-blessed. Every key in
      `dev/baseline/manifest.json` is a literal path prefixed
      `examples/test_local/`, so O-20 changes ALL 18 keys, the discharge row
      included. Worse, `dev/scripts/check_baseline.py:384-385` derives the stored
      reference-series filename as `sha1(resolved_path)[:16]`, so the rename also
      changes the sidecar name — `record` writes a NEW
      `dev/baseline/discharge_ref/<newhash>.csv` from the CURRENT `output.csv`
      (`record_discharge`, :404-415) and orphans
      `dev/baseline/discharge_ref/1f9f30a367de162f.csv`. Between commit 3 and
      commit 12 `check` cannot run at all against either tree: recorded keys are
      old paths (`Path(path).exists()` False → "target missing on disk") and every
      current path reports "target present but not in manifest"
      (`cmd_check`, :569-582).
    rationale: >-
      Unlike the 17 fingerprint targets, the discharge target is compared with a
      tolerance comparator against a stored series, not against a self-contained
      hash. Re-recording it regenerates that series from the post-R07 run, so if
      R07 does perturb discharge — a genuine risk given B2 relocates the forcing
      and rewrites `input.path_forcing` — the re-record accepts the drift and
      `check` goes green. The design's whole behaviour-preservation claim rests on
      this one target, and the plan as written cannot prove it.
    suggested_fix: >-
      Add to the commit plan, before commit 3: copy
      `examples/test_local/hydrology_model/run_default/output.csv` to a run-local
      holding path. Add to commit 12, as a gate before `record`:
      `python dev/scripts/check_baseline.py compare --ref <saved> --cur test_case/test_local/hydrology_model/run_default/output.csv`
      must exit 0. Also delete the orphaned
      `dev/baseline/discharge_ref/1f9f30a367de162f.csv` in the same commit, and
      state that `check_baseline check` is expected to be red from commit 3 to
      commit 12 (so a red gate mid-milestone is not misread as a regression).

  - id: repo-4
    severity: major
    section: Risks and open questions
    finding: >-
      The `"None"` sentinel risk is misdiagnosed at both cited line numbers.
      `blueearth_cst/model/setup_gauges_and_outputs.py:55` reads
      `if gauges_fn is not None and os.path.isfile(gauges_fn):` and
      `blueearth_cst/model/plot_results.py:127` reads
      `if observations_fn is not None and os.path.exists(observations_fn):`.
      Both short-circuit on the first clause, so YAML `null` raises nothing —
      neither line can produce the claimed `TypeError`. (Verified at those two call
      sites only; I did not audit every consumer for `null`-tolerance.) The value
      that actually misbehaves is the STRING: `blueearth_cst/shared/plot_map.py:28-29`
      guards only `if gauges_fn is not None:` and then computes
      `gauges_name = f'gauges_{basename(gauges_fn).split(".")[0]}'`, yielding the
      bogus layer name `gauges_None` — which is exactly drive-by O-08. The
      empirical evidence that the guards are existence-based rather than
      None-based is `tests/snake_config_model_test.yml:32-33`, which points at a
      `tests/data/observations/` tree that does not exist (O-04) and passes today.
    rationale: >-
      This is the design's single named runtime assertion (Verification plan,
      stage 1: "both … parse to the **string** `"None"`, not YAML `null`"). As
      written it pins the value that causes the real defect, and it will pass while
      proving nothing — burning the milestone's only runtime sentinel check on a
      non-risk while the genuine one is demoted to an unsequenced drive-by.
    suggested_fix: >-
      Rewrite the risk as "the string sentinel `"None"` reaches
      `plot_map.py:28` unguarded", promote O-08 out of the drive-by list into the
      commit that touches `plot_map.py` (commit 10), and replace the stage-1
      assertion with a positive one: with `output_locations: None`,
      `basin_area.png` must not be produced from a `gauges_None` layer.

  - id: repo-5
    severity: major
    section: "What changes — B. The artifacts (`project_dir`)"
    finding: >-
      The `<project_dir>/config/` split into `runs/` + `catalogs/` + `templates/`
      (design-v1.md:148-152) appears only in the tree diagram: it has no entry in
      "The substantive moves" (B1–B8), no commit in the 13-commit plan, and no
      verification row. It is not a path move — it requires a behaviour change in
      `blueearth_cst/model/copy_config_files.py`, which derives a SINGLE
      `output_dir = dirname(config_snake_out)` (:68) and writes the snake config,
      the two build templates, and the catalog into that one directory (:47-56,
      :80-81). Routing three kinds to three subdirectories means passing three
      destinations, not renaming one. Separately, `wflow_build_model_run.yml` and
      `wflow_build_forcing_historical.yml` are filed under `templates/` in the
      tree, but they are rule-GENERATED runtime configs — declared outputs of rules
      1.02 and 1.07 (`Snakefile_model_creation:98,180`) and declared inputs of
      rules 1.03 and 1.08 (:113,196) — not copied source templates.
    rationale: >-
      Three concrete consequences. (a) The commit plan is incomplete: no commit
      leaves the tree in the drawn shape, so the phase-B `semantic_tree_diff`
      gate compares against a tree the plan never produces. (b) The design's
      "no computational path changes — every item is a path move, a rename, a
      declaration fix, or an added warning" is false for this item. (c) Filing
      generated runtime configs under `templates/` directly contradicts the stated
      goal that "a reader can tell what made a file by where it sits", and moving
      them changes four rule path strings (1.02 output, 1.03 input, 1.07 output,
      1.08 input) that the tree diagram does not surface.
    suggested_fix: >-
      Promote the config split to a numbered substantive move with its own commit,
      list the `copy_config_files.py` signature change it needs, and either give
      the two generated configs their own bin (`config/generated/` or
      `config/runs/`) or drop them out of `templates/` and say where they land.

  - id: repo-6
    severity: major
    section: Behaviour-preservation stance and baseline consequence
    finding: >-
      The "machinery to update alongside" list names `check_baseline.py` TARGETS,
      `semantic_tree_diff.py`'s directory-prefix path map, and its path-aware TOML
      comparator — but omits `COPIED_CONFIG_PATH_MAP`
      (`dev/scripts/semantic_tree_diff.py:90-110`), the third normalization table.
      O-20 changes `project.project_dir` inside every copied config snapshot from
      `examples/test_local` to `test_case/test_local`, and O-01 changes
      `workflows.model_creation.output_locations` /
      `observations_timeseries` in `config/workflows/snake_config_model_test_linux.yml:25-26`.
      `compare_copied_config` (:428-447) normalizes only values whose key appears
      in that map and FAILs on any residual difference. The walk is recursive
      (`_normalize_config_paths`, :409-425), so a `project_dir` entry will reach
      the nested `project:` section — the fix works, it is simply not listed.
      `MIGRATION.md:167-172` records that this table is kept "in lockstep" with
      the migration map's config-path section.
    rationale: >-
      Without the added entries, the phase-B gate — "full-`project_dir` pre/post
      comparison … every value identical" — fails on all three (or four) copied
      config snapshots for a reason that is pure path bookkeeping, and the failure
      is indistinguishable from a real content regression. Since the design makes
      that gate the sole proof for B1–B7, a false red there is expensive.
    suggested_fix: >-
      Add `COPIED_CONFIG_PATH_MAP` to the machinery list with the required entries
      (`project_dir`, and the two observation keys if their values change), and add
      the corresponding rows to `dev/milestones/r07/migration_project-layout.md`'s config-path
      table so the documented lockstep holds.

  - id: repo-7
    severity: major
    section: "What changes — B. The artifacts (`project_dir`)"
    finding: >-
      B8's `project_name` is `basename(project_dir)`, but that value is not
      guaranteed to satisfy the grammar the Snakefile then enforces.
      `validate_experiment_name` (`blueearth_cst/shared/snake_utils.py:181,232`)
      matches `^[a-z0-9][a-z0-9_]*$` and its docstring states "Uppercase is
      REJECTED (never silently lowercased)". Real project dirs in this repo
      already violate it: `config/workflows/snake_config_model_test_linux.yml:8`
      and five `snake_config_projections_*.yml` files carry
      `project_dir: examples/Gabon`, whose basename is `Gabon`. Production
      `project_dir` values live outside the repo tree (AGENTS.md § Repo Map) and
      routinely carry uppercase, hyphens, or spaces.
    rationale: >-
      The helper writes `experiment_name: Gabon_20260726` into the config; the very
      next `Snakefile_climate_experiment` parse raises `ValueError` at line 41,
      before any rule runs. The feature whose stated purpose is to remove friction
      would introduce a hard parse failure on the majority of realistic project
      dirs, and the design's supporting evidence ("both `gabon260725` and
      `gabon_20260726` already satisfy the grammar") tests only already-conforming
      names.
    suggested_fix: >-
      Specify the slugification the helper applies (lowercase, non-`[a-z0-9]` →
      `_`, strip leading non-alnum, truncate to 64) and note in the design that
      this deliberately differs from `validate_experiment_name`'s
      never-silently-lowercase stance because the helper is a *suggestion* writer,
      not a validator — then state that the suggested value is re-validated through
      `validate_experiment_name` before being written.

  - id: repo-8
    severity: major
    section: Verification plan
    finding: >-
      The O-22 row's proof is "`tests/test_cli.py` matches on combined
      stdout+stderr — confirm its assertions are undisturbed." No test in
      `tests/test_cli.py` asserts on output text. The three CLI tests assert only
      `result.returncode == 0`, with the combined stream used as the assertion
      *message* argument; the historical ratchet assertions on
      `MissingInputException` / `CyclicGraphException` were removed in R3 and R5,
      and only the `_dry_run` docstring still describes them. The one text
      assertion in the file (`test_climate_projections_declares_wf1_region_input`)
      reads the Snakefile from disk, not process output.
    rationale: >-
      The design's only stated verification for `warn_if_project_dir_in_repo()` is
      structurally incapable of detecting whether the warning fires, is silent, or
      was never wired in — the tests pass identically in all three cases. O-22
      would ship with zero coverage, and the exemption logic (the case most likely
      to regress) is never exercised.
    suggested_fix: >-
      Correct the claim, and add a real assertion — three cases in
      `tests/test_snake_utils.py` calling `warn_if_project_dir_in_repo()`
      directly (in-repo path → warns; `<repo_root>/test_case/...` → silent;
      absolute out-of-tree path → silent) plus one `test_cli.py` case asserting the
      warning text appears in the combined stream for the in-repo fixture config.

  - id: repo-9
    severity: major
    section: "What changes — B. The artifacts (`project_dir`)"
    finding: >-
      The tree places `inmaps_rlz_*_cst_*.nc (temp)` under
      `experiments/<id>/weather_generator/output/` (design-v1.md:186-187). Those
      files are not weathergenr output: they are wflow-grid downscaled forcing,
      produced by rule 3.09 `downscale_climate_realization`
      (`Snakefile_climate_experiment:346`) via
      `blueearth_cst/experiment/downscale_climate_forcing.py`, and consumed by the
      rule 3.10 batch rules (:417-418). They are the per-realization twin of
      `inmaps_historical.nc` — the exact artifact class B2 moves INTO the wflow
      engine subtree. B5 also does not say where the weathergenr-native
      `rlz_{r}_cst_{c}.nc` realizations land, nor where rule 3.10's
      `outstates_rlz_*_cst_*.nc` (:424-425) goes under
      `hydrology_runs/rlz_<r>/{config,output}/`.
    rationale: >-
      P3 is the principle the whole artifact half is built on, and the tree
      violates it in the one place the design also fixes it (B2). The stated payoff
      — "a reader can tell what made a file by where it sits" — is lost for the
      largest file class in the tree, and a future reader tracing rule 3.09's
      output lands in the wrong engine's subtree. It also makes the `_work/` vs
      `output/` boundary undecidable for the two file classes B5 leaves unplaced.
    suggested_fix: >-
      Move `inmaps_rlz_*_cst_*.nc` to `hydrology_runs/rlz_<r>/forcing/` (mirroring
      B2's `hydrology_model/forcing/`), keep only the weathergenr-native
      `rlz_{r}_cst_{c}.nc` and the date CSVs under `weather_generator/output/`,
      and add a row for `outstates_*.nc`.

  - id: repo-10
    severity: minor
    section: What changes — A. The repository
    finding: >-
      O-24's undeclared-output inventory is complete only for the seed config.
      Beyond `clim_wflow_1_{month,year}.png` and `performance_metrics.csv`,
      `plot_results.py` also drives `plot_basavg` (:262 →
      `func_plot_signature.py`, `save_figure(f"{dvar}.png")`) whenever
      `ds_basin.data_vars` is non-empty — one PNG per basin-average entry in
      `wflow_outvars`, whose Snakefile default is
      `['river discharge', 'actual evapotranspiration']` — and `plot_signatures`
      (:320 → `signatures_{station_name}.png`) whenever observations exist and
      `nb_years >= 5`. The `clim_{station}_{period}.png` names are per-station
      (:246-256), so a project with a real `output_locations` CSV produces
      `clim_wflow_2..N_*`. The current fixture happens to have an empty
      `ds_basin` and no observations, which is why exactly 8 files sit in
      `examples/test_local/plots/wflow_model_performance/`.
    rationale: >-
      The verification row claims `snakemake --delete-all-output` will remove the
      newly-declared set "which it cannot do today". On any config with gauges,
      observations, or basin-average outvars it still cannot — stale figures from a
      previous config survive a rerun, which is the defect O-24 exists to close.
    suggested_fix: >-
      Either derive the declared output list at parse time from `wflow_outvars` /
      `output_locations` (both already read in `Snakefile_model_creation:51,54`),
      or state explicitly that O-24 declares the config-invariant subset and name
      the remainder as knowingly undeclared.

  - id: repo-11
    severity: minor
    section: Risks and open questions
    finding: >-
      The G1 ruling to move `MIGRATION.md` to `docs/` leaves three loose ends the
      design must close. (a) `naming.md` §7 does not merely "put migration notes
      under `dev/<milestone>/`" — it makes
      `dev/<milestone>/migration_<topic>.md` the required artifact of a contract
      rename. Moving the R06 note to `docs/` does not create a `dev/milestones/r06/` note, so
      §7 stays unsatisfied for R06 unless §7 itself is amended. (b) After the move
      the repo has `docs/MIGRATION.md` (R06) and
      `dev/milestones/r07/migration_project-layout.md` (R07) — two consecutive milestones,
      two locations, two naming schemes. (c) `docs/` is uniformly lowercase
      (`install.md`, `env_setup_notes.md`, `cst-toolbox-technical-note-2025.md`);
      `MIGRATION.md` is a root-level upstream convention (naming.md §8 row 4) that
      carries no exemption inside `docs/`.
    rationale: >-
      Left as-is, the very next milestone re-opens the same question, and a user
      following `docs/MIGRATION.md` after R07 lands on paths R07 renamed again
      (its config-path table at :160-172 references `config/` targets and its
      `docs/config/` mirror note at :173 describes a directory O-05 deletes).
    suggested_fix: >-
      Amend `naming.md` §7 to distinguish two artifact classes — an internal
      rename record at `dev/<milestone>/migration_<topic>.md` (required) and an
      optional user-facing migration guide under `docs/` derived from it — rename
      the moved file to `docs/migration-r06.md` for `docs/` consistency, and state
      in commit 13 whether R07 publishes a user-facing guide or `docs/` carries an
      R06-only historical record.

  - id: repo-12
    severity: minor
    section: What changes — A. The repository
    finding: >-
      O-22's exemption is described as "`<repo_root>/test_case`, held in a
      module-level constant", but `blueearth_cst/shared/snake_utils.py` has no
      notion of the repository root — a module-level constant can hold only the
      relative segment `"test_case"`. The three call sites do have the root
      (`workflow.basedir`, already used for the `sys.path` insert and the
      `run_logged` path at `Snakefile_model_creation:7,21`).
    rationale: >-
      Deriving the root inside the module (`Path(__file__).parents[2]`) silently
      breaks if the package is ever installed rather than imported from the repo,
      and hardcoding an absolute path in a constant is not portable across
      machines. Getting the signature wrong makes the exemption either
      non-functional (never fires) or over-broad (any dir named `test_case`).
    suggested_fix: >-
      Give the helper the signature
      `warn_if_project_dir_in_repo(project_dir, repo_root)` with a module-level
      `_PROJECT_DIR_EXEMPT_NAMES = frozenset({"test_case"})` — matching the
      existing private-constant style (`_EXPERIMENT_NAME_RE`,
      `_WINDOWS_RESERVED_NAMES`) — and pass `workflow.basedir` from each Snakefile.

  - id: repo-13
    severity: minor
    section: Behaviour-preservation stance and baseline consequence
    finding: >-
      The per-group baseline accounting does not reconcile. The design lists four
      copied-config snapshots + three wf1 plots + six wf2 targets + two wf3 targets
      + one unchanged discharge = 16, against "the eighteen targets". The gap is
      that `dev/baseline/manifest.json` holds 18 rows while
      `dev/scripts/check_baseline.py`'s `TARGETS` (:96-117) holds 15 live
      templates. Three manifest rows have no live producer and are silently
      ignored by `check` (they are neither in `current` nor in `missing`, so the
      `cmd_check` loop at :573-575 skips them):
      `climate_experiment/model_results/Qstats.csv`,
      `climate_experiment/model_results/basin.csv`, and the root-level
      `config/snake_config_climate_experiment.yml` — all pre-P3-1 paths superseded
      by `experiments/<name>/`. So there are three live config snapshots, not four.
    rationale: >-
      A full `record` at commit 12 overwrites `targets` wholesale
      (`cmd_record`, :532-534), so those three rows vanish from the manifest. The
      design promises the re-record diff will be "path-and-snapshot-only"; three
      unexplained deletions in that diff are indistinguishable from an R07 target
      that failed to be produced, which is precisely what the adjudication step
      must rule on.
    suggested_fix: >-
      Correct the counts (15 live targets: 14 move path, 1 discharge whose
      manifest key also moves per repo-3), and note in commit 12 that the
      re-record additionally drops three stale pre-P3-1 rows, with their paths
      listed so the diff is fully accounted for.

---

## Reviewer notes (repo fit & conventions lens)

**Verdict rationale.** `revise`, not `reject`. The milestone's structure is sound
and every finding above has a bounded fix that fits inside the existing commit
plan. Two things must change before implementation: B1 needs an explicit
producer assignment (repo-2) and a sidecar-name reconciliation (repo-1), and the
baseline plan needs the wf1 discharge anchor carried across the fixture rename
(repo-3). The remaining ten are corrections and completions, not redesigns.

**Claims that check out.** Verified against the code and recorded here so the
author does not re-check them:

- 31 `script:` directives (11 / 10 / 10 across the three Snakefiles), three
  `sys.path.insert` calls — the Snakefile-move rejection cost is accurate.
- `README.rst:269,285,298` are exactly the three `--dag | dot -Tpng > dag_*.png`
  lines; six notebook cells (two per notebook × three notebooks) repeat the
  pattern. `scripts/run_snake_test.cmd:32` is `set DAGDIR=dag`. `.gitignore:124`
  is `examples/`, `:136` is `dag/`.
- `docs/config/` holds exactly 16 files; `data/` is 653 KiB in two CSVs,
  referenced only by `config/workflows/snake_config_model_test_linux.yml:25-26`
  (plus the `docs/config/` copies O-05 deletes).
- CI baselines 386/30/1 (windows) and 385/31/1 (ubuntu) are correct
  (`.github/workflows/ci.yml:68-69`).
- B2: `path_forcing` is indeed the only TOML pointer, hardcoded at
  `blueearth_cst/shared/setup_time_horizon.py:51`. The full edit set is five
  places — that line, `Snakefile_model_creation:198,210,305`, and
  `tests/test_interchange_contracts.py:529`. Because the target moves *inside*
  the hydromt model root (`basin_dir = f"{project_dir}/hydrology_model"`), the
  new pointer is `forcing/inmaps_historical.nc`, which is a strictly better
  shape than today's `../climate_historical/...`.
- B6: `Qstats.csv` header is `statistic,tavg,prcp,Q_130000086` and `basin.csv`
  is exactly `tavg,prcp` — the denormalisation argument holds.
- B8's grammar claim: `_EXPERIMENT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")`
  at `blueearth_cst/shared/snake_utils.py:181`, and both cited example names pass.
  (The derivation that feeds it does not — see repo-7.)
- O-24's core claim: `plot_map_forcing.py:167-187` loops
  `{precip, pet, temp}` writing three PNGs against rule 1.13's single declared
  `precip.png`; `plot_results.py` writes `clim_{station}_{period}.png` and
  `performance_metrics.csv` undeclared. (Inventory incomplete — see repo-10.)
- `semantic_tree_diff.py`'s TOML comparator already covers all five pointer
  fields including `csv.path` and `state.path_output`, and is generic over the
  path map — so B5 needs a new `build_r07_path_map`, not a comparator change.
  The design's phrasing ("the comparator must be updated") overstates it.

**Forbidden set.** Nothing in the milestone touches `pixi.lock`,
`Manifest.toml`, `Project.toml`, `.pixi/`, or a vendored upstream package. The
one adjacency worth watching: `Snakefile_model_creation:220` and the rule 3.10
batch driver both run `julia --project=.`, so the Julia project root stays the
repo root regardless of where `project_dir` points — no R07 item disturbs that.

**`naming.md` compliance of the new surface.** The proposed identifiers and
paths are conformant: `hydrology_runs/`, `weather_generator/`, `indicators/`,
`_work/`, `rlz_<r>/` are snake_case (§1, §8); `warn_if_project_dir_in_repo` is
verb-first (§1); `test_case/` is a §7 contract rename (test fixture path read by
`check_baseline.py`) and correctly carries a migration map. Two notes: the
`config/templates/observations/{output_locations.csv, observations_timeseries.csv}`
templates rename the existing kebab-case `output-locations-test.csv` to
snake_case, which §8 permits (config/data files follow the tool contract) but
which the migration map should record since a user config may reference the old
name; and `dev/milestones/r07/migration_project-layout.md` mixes a snake_case prefix with a
kebab-case topic — that is §7's own mandated `migration_<topic>.md` form, so it
is correct despite §8's kebab-case rule for `dev/` markdown. Worth a one-line
note in §7 that its filename form overrides §8, since this is the second
milestone to hit it.

**On the four G1 rulings.** Not re-litigated. The only downstream inconsistency
a ruling creates is the `MIGRATION.md` one, filed as repo-11 — the ruling to
place it in `docs/` is defensible on audience grounds, but §7 needs an actual
amendment (not just a stated divergence) because R06 would otherwise be left
with no `dev/milestones/r06/` note at all, and the R06/R07 pair would sit in two locations
under two naming schemes.


---

# External review — round 1 (clean-room, `design-v2.md`)

*Verbatim output of a headless `codex exec` dispatch, read-only sandbox.*

## Verdict
verdict: revise
doc_version: design-v2.md

## Findings
### ext1-01  [blocking]
- section: Principles — P4 / The substantive moves — B1
- finding: B1 is not a producer “over region + catalog only”: `climate_store_spec(...)` requires `build_config`, and delineation reads dataset identifiers from `config/templates/wflow_build_model.yml`’s `setup_basemaps`. This directly contradicts both P4 and the settled B1 producer assignment.
- rationale: A climate-only execution remains coupled to a Wflow/HydroMT build template. Without that template it cannot derive the store region, while edits to model-build configuration can change a supposedly model-independent climate artifact.
- suggested_fix: Define the delineation inputs in a model-independent shared region specification consumed by both climate extraction and the Wflow build. Remove `build_config` from the climate producer contract and verify climate analysis with model templates and `hydrology_model/` absent.

### ext1-02  [blocking]
- section: The substantive moves — B1 / Verification plan — Rerun-triggering across the two DAGs
- finding: The claim that `ancient(guard_ok)` prevents both mtime and input-set triggering is incorrect. `ancient()` suppresses timestamp-based freshness; it does not make an input disappear from Snakemake’s recorded input set. An output first produced by wf1 with no inputs is subsequently encountered in wf3 with `guard_ok` added, so the documented input-set rerun trigger can schedule extraction again.
- rationale: The required “wf1 then wf3 reports nothing to be done” verification is expected to fail, retaining the duplicate native-resolution extraction B1 is intended to eliminate. The design provides a test for this outcome but no implementable branch if it occurs.
- suggested_fix: Keep the producer’s declared inputs identical in both DAGs. Enforce the wf3 guard through a separate readiness sentinel or downstream dependency that depends on both the guard and store, without adding the guard to the shared producer rule.

### ext1-03  [major]
- section: Behaviour-preservation stance and baseline consequence — Claim 1
- finding: The unqualified no-value-change claim is supported only for the seed fixture. Equality of three bbox derivations there does not establish equivalence for other regions, resolutions, hydrography datasets, or polygons near source-grid cell boundaries.
- rationale: For another supported configuration, the new delineation bounds can select a different climate-cell extent despite `buffer=1`, changing weather-generator inputs and downstream indicators. The design’s failure branch will never activate if only the seed fixture is checked.
- suggested_fix: Either scope the behaviour-preservation claim explicitly to the baseline fixture or add a configuration-independent argument plus parameterized boundary-sensitive tests across representative region forms and climate grids. State how non-seed divergence is classified.

### ext1-04  [major]
- section: Principles — P2 / The substantive moves — B1
- finding: `climate_store_spec()` shares only outputs and params; it does not generate the complete producer declaration. The two Snakefiles can still drift in script target, inputs, resources, execution directives, or rule-body behavior, so the stated mechanism does not ensure “one rule definition” or satisfy P2(a).
- rationale: A later edit to one declaration can make the same artifact path execute different producer behavior depending on the entry-point Snakefile, causing repeated work or inconsistent content while the shared-spec equality test still passes.
- suggested_fix: Centralize the entire producer contract—including executable, inputs, outputs, params, and relevant directives—in one reusable rule module or generated declaration, and test equality of the complete normalized contract rather than paths and params alone.

### ext1-05  [major]
- section: What changes — A. The repository — O-01 / Risks and open questions — Explicitly out of scope
- finding: O-01 deletes the basin data still consumed by the Linux configuration and Docker runner while merely declaring those consumers “parked.” Parking their validation does not prevent the refactor from breaking them.
- rationale: The documented Linux/Docker entry path will fail on missing observation files after the deletion, making the milestone behavior-breaking outside the Windows seed path without an explicit support decision.
- suggested_fix: Retarget those consumers to mounted or explicitly configured external observation paths and add a parse-level check, or formally retire/remove the affected runner and configuration with clear documentation in this milestone.

### ext1-06  [major]
- section: Commit plan — Cut line, corrected
- finding: Commit 6 is called a safe stopping point even though the design states `check_baseline check` is necessarily red from commits 4 through 14. “One re-record still owed” means commit 6 is not a releasable or completed cut line.
- rationale: Stopping there leaves the repository’s baseline contract invalid indefinitely, and subsequent failures cannot be distinguished from the intentionally stale manifest using the normal gate.
- suggested_fix: Define an alternate cut-at-6 closure step that performs and commits an interim manifest re-record, or remove commit 6 as a safe cut line and describe it only as a temporary pause requiring preservation of the holding artifacts.

### ext1-07  [minor]
- section: Contract inventory
- finding: The inventory claims to cover moved hardcoded paths and attach every row to a commit, but it omits B3, B4, and B8 entirely despite B4 introducing a producer and B8 introducing a configuration-writing interface.
- rationale: The derived implementation brief can omit affected rules, modules, tests, or user-facing invocation details, leaving discovery to implementation-time inspection.
- suggested_fix: Add rows for B3, B4, and B8, including producer/rule ownership, modules, tests, and the exact invocation and target-config contract for the experiment-name suggestion helper.


---

# External review — round 2 (with regression duty, `design-v3.md`)

*Verbatim output of a headless `codex exec` dispatch, read-only sandbox.*

## Verdict
verdict: revise
doc_version: design-v3.md

## Findings
### ext2-01  [blocking]
- section: The substantive moves — B1 / Verification plan — Rerun-triggering across the two DAGs
- finding: Emptying both producer input sets fixes the cross-DAG input-set oscillation, but the replacement freshness contract is incomplete. The producer reads the data catalog and catalog-resolved hydrography and climate sources, while `params` records only identifiers and the catalog path. Editing the catalog in place—or changing data referenced by an unchanged catalog entry—does not change any recorded input, parameter, code, or environment trigger. The existing store can therefore remain silently stale despite the claim that “everything content-determining rides in params.”
- rationale: A supported configuration change can leave `store_region.geojson` and `extract_historical.nc` representing the old catalog contents, producing climate figures and stress-test indicators from stale data. The alternation test cannot detect this because both workflows will correctly report the stale artifact up to date.
- suggested_fix: Keep the producer declarations identical but give both the same regular catalog input, or carry a stable digest/version of the relevant catalog definitions in params. Define how local or mutable catalog-resolved sources participate in freshness, and add a test that changing a relevant catalog definition schedules extraction exactly once without restoring the cross-DAG oscillation.

### ext2-02  [major]
- section: Principles — P2 / The substantive moves — B1
- finding: The proposed “complete producer contract” test is still not complete: it compares rule name, script, inputs, outputs, and params, but omits execution-affecting directives such as `conda`, `container`, `envmodules`, `wrapper`/`notebook`, `shadow`, threads, and resources. The document asserts that only `message`, `log`, and `benchmark` may differ, but the test does not enforce that allowed-field set.
- rationale: A later one-DAG environment or execution-directive change can make the shared output run under different software or execution semantics, potentially changing content or firing Snakemake’s software-environment trigger, while the advertised contract-equality test still passes.
- suggested_fix: Normalize and compare every content- or execution-affecting rule directive, and fail on unknown or unexpected per-workflow fields; alternatively, generate both rule declarations from one shared rule module so only explicitly permitted presentation fields remain local.

### ext2-03  [minor]
- section: Commit plan / The gate blackout, stated
- finding: The blackout starts at commit 1, not commit 4 as repeatedly stated. Commit 1 changes `check_baseline.py`’s `TARGETS` and `PROJECT_DIR_DEFAULT` to the future `test_case/` paths, while the fixture remains under `examples/` until commit 4; commit 4 also independently lists `check_baseline.py`, leaving ownership of that edit contradictory.
- rationale: `check_baseline check` will report missing targets immediately after commit 1, so the documented baseline-valid interval and pause/recovery reasoning are inaccurate for commits 1–3.
- suggested_fix: Keep commit 1 to path-map, merge-class, comparison machinery, and reference capture; move `TARGETS` and `PROJECT_DIR_DEFAULT` atomically into commit 4 with the fixture rename, or explicitly redefine and cover the longer blackout.

## Round-1 regression check
- ext1-01: resolved
- ext1-02: re-raised as ext2-01
- ext1-03: resolved
- ext1-04: re-raised as ext2-02
- ext1-05: resolved
- ext1-06: resolved
- ext1-07: resolved


---

# Finding ledger

*Append-only, one row per original finding ID. Verbatim as closed.*

Append-only. One row per original finding ID from the stage-2 internal panel
(`internal-review-risk.md`, `internal-review-architecture.md`,
`internal-review-repo-fit.md`; grouped in `internal-review-index.md`).

**Severities are as filed.** Where a finding is graded differently by two lenses
(groups D, G, H, I in the index), each ID is dispositioned at *its own* filed
severity and no severity is changed here — that is the driver's to log, not the
author's. Where I judge a grade arguable, it is said in the rationale column and
the severity is left alone.

**Disposition counts.** 34 accepted, 0 rejected, 0 deferred — 7/7 blocking,
20/20 major, 7/7 minor accepted. **Read this as "no finding was rejected", not as
"every suggested fix landed verbatim": eight findings are resolved by a route
other than their own suggested fix**, listed in § Notes below. Several accepted
findings are fixed by a route
*other* than their own suggested fix (owner rulings GA-1 / GA-2, or a
better-evidenced alternative); the rationale column names the adopted route in
every such case, so "accepted" is never read as "the suggested fix landed".

| ID | Round | Severity | Disposition | Resolution or rationale | Doc version |
|---|---|---|---|---|---|
| risk-1 | internal-panel | blocking | accepted | B1 now names the producer: one rule definition over region + catalog, declared in both Snakefiles (owner ruling GA-1). **Neither of risk-1's own fixes adopted** — (i) pulling a standalone extraction rule into R07 and (ii) deferring B1 were both rejected at G1-return; alternatives now record why. The finding's second demand *is* met: B1 carries a point-by-point rebuttal table against `dev/milestones/p32a/climate-analysis-design.md` §"Why a separate wf1 extraction", which v1 engaged nowhere, and the scope-authority header lists p32a as explicitly superseded. | design-v2.md |
| risk-2 | internal-panel | blocking | accepted | Both halves adopted. `diff_trees` gains a **declared many-to-one merge class** (`--merge <survivor>=<src1>,<src2>`) with exactly risk-2's semantic: the survivor is compared against *each* collapsed source and the merge passes only if all comparisons pass — allowlisting one as MISSING is explicitly rejected as proving nothing. The hardcoded-map half is also fixed: machinery list item 2 adds `build_r07_path_map()` plus a generic `--map old=new` CLI, and the machinery moves to commit 1 so the gate exists before the moves it polices. | design-v2.md |
| risk-3 | internal-panel | major | accepted | Blackout window now stated explicitly ("The gate blackout, stated") with three named substitutes: per-slice `semantic_tree_diff` against a retained pre-R07 reference tree after commits 7/8/11/12, the comparator-based discharge anchor, and a commit-message note that a red `check` is expected. The cut-line contradiction is resolved the other way from risk-3's suggestion: O-20 must precede the machinery-dependent moves, so the repository half stays first and the **cut line is restated to match** — safe stopping points are after commit 6 and after commit 14. | design-v2.md |
| risk-4 | internal-panel | major | accepted | Three separate fixes. P3 restated with the `hydrology_model/` exemption inline (hydromt owns `model_root`) and with `plots/`/`_work/` optional-when-empty. P1 reworded to "figures attach to **what they depict**", and `basin_area.png` consequently moves to `hydrology_model/plots/` rather than `evaluation/plots/`. Generated build configs get an explicit home and a stated rule: `config/generated/` for run-time-generated configs, `config/templates/` for verbatim template snapshots. | design-v2.md |
| risk-5 | internal-panel | major | accepted | Resolved by owner ruling **GA-2**: only the test fixture depends on artifact paths; no CST-API/CST-frontend consumer reads them. New § "Migrating an existing `project_dir`" states the non-support explicitly, names the deterministic first failure (rule 3.00b's mandatory `wf1_snapshot` input), and records that the downstream-consumer question was asked and answered "no consumer". **Horn (ii) — an executable `mv` script — is forbidden by the ruling and is now a rejected alternative.** One addition beyond the finding: non-support is scoped to *running* against a pre-R07 tree, not to *retaining* one, because the phase-B gate and the discharge anchor both need a preserved reference. | design-v2.md |
| risk-6 | internal-panel | major | accepted | Counts corrected and the derivation source changed: the inventory is now re-derived from `check_baseline.TARGETS` (15 live) rather than from `manifest.json` (18 rows), the three pre-P3-1 orphans are named, and the "four copied-config snapshots" claim is corrected to three with the wf3 experiment snapshot explicitly *not* joining the `config/runs/` split. The three orphan rows are listed in the migration map as expected deletions so the exit adjudication can account for them. Same facts as arch-10 and repo-13; dispositioned identically at each filed severity. | design-v2.md |
| risk-7 | internal-panel | major | accepted | All three points. B6's caveat now names `precip_variance` (denormalised nowhere) and the `.iloc[0]` January-row reduction, and the conclusion is qualified — `_work/` is retained-not-deleted and `cst_*.csv` remains the only record of both. `cst_*.csv` becomes a **declared `input:` on rule 3.11** while moving, and B6 joins B4/B5/B6 on the "needs a real run, not a dry-run" list. Also recorded in Claim 2 as one of the four items that are not pure path moves. | design-v2.md |
| risk-8 | internal-panel | major | accepted | Reconciliation written into § Risks and assigned to commit 15: `naming.md` §7 is **amended** (not merely diverged from) to distinguish a required internal rename record at `dev/<milestone>/migration_<topic>.md` from an optional user-facing guide under `docs/`. The paradox risk-8 sharpens is answered directly: R07 publishes **no** user-facing guide, because GA-2 declares pre-R07 trees unsupported, so its map stays internal and the two milestones become consistent rather than opposite. Filed `major` by risk and `minor` by repo-fit/architecture; dispositioned at each filed severity, one fix. | design-v2.md |
| risk-9 | internal-panel | minor | accepted | Both halves. The source-grid family is **filename-disambiguated** (`source_precip.png`, `source_temp.png`, `source_pet.png`) rather than relying on the parent directory, precisely because a `pet.png` copied into a report loses its directory. B4's input list gains the data catalog, since on the era5 branch source-grid PET needs `era5_orography` from the catalog — which also makes the P4 assertion test correctly specified. Not deferred: the fix is one line in the tree plus one in the B4 table, and it lands in the same commit as B4. | design-v2.md |
| risk-10 | internal-panel | minor | accepted | `tests/` path bindings promoted to machinery list item 6, with all eight affected modules and line numbers, and attached to commits via the new § "Contract inventory" table. Not deferred: "`pytest tests/` green" is a stated success criterion, so leaving the bindings unassigned makes commits 4, 7 and 8 land red. Overlaps arch-4's larger surface; both dispositioned at their filed severity with one shared fix. | design-v2.md |
| arch-1 | internal-panel | blocking | accepted | Same blocking group as risk-1/repo-2, resolved by GA-1's shared-rule route. **arch-1's own alternative — restating the P4 assertion as what R07 actually proves — was rejected at G1-return** and is now a recorded alternative, because it preserves scope by abandoning a named success criterion. Its first fix *is* adopted in full: B1 names the owning declarations, the exact inputs, and pins `hydrography_fn`/`basin_index_fn` to `config/templates/wflow_build_model.yml` `setup_basemaps`. arch-1's `get_region_preview.py` observation is not only confirmed but strengthened — verified this revision that the module **does not import** on the pinned hydromt 1.3.1 (`hydromt.cli.api` removed in hydromt 1.x); it is logged as O-25 and retired, and `hydromt.model.processes.region.parse_region_basin` replaces it. | design-v2.md |
| arch-2 | internal-panel | blocking | accepted | Merge class adopted, `--retire` set rejected on the merits and recorded as a rejected alternative: retiring `wf1_raw/*` and allowlisting it as MISSING lets the gate go green while proving nothing about the store that disappeared — exactly where GA-1 demands proof. Verified against the code that `.nc` already dispatches to the element-wise `compare_nc` (`semantic_tree_diff.py:592`), so the merge class is simultaneously the fix for the `ValueError` and the executable form of GA-1's bbox proof; the design states this connection rather than treating them as two work items. | design-v2.md |
| arch-3 | internal-panel | major | accepted | Resolves the index's preserved conflict (arch-3 vs the risk lens's explicit non-finding) **in the architecture lens's favour, by owner ruling GA-1**: the bbox change is now the third named exception to the behaviour-preservation stance. arch-3's suggested test is adopted, via the Group B merge class rather than as a separate mechanism. The risk lens's `buffer=1` reasoning is preserved as a reason to *expect* a match, not as a substitute for checking — and this revision adds a bounds probe on the seed fixture (R07 bbox bit-identical to today's wf1 bbox; ≤3.4e-07° from today's wf3 bbox, i.e. GeoJSON 6-dp rounding). The design also writes the **stated branch** if the coordinate arrays do *not* match, including the wf3-indicator tail arch-3 did not reach and a hard stop-and-escalate if discharge moves. | design-v2.md |
| arch-4 | internal-panel | major | accepted | The `--dry-run` concession is replaced by a **contract inventory table**, exactly the shape arch-4 asked for: for each move, the rule(s), the script-module constant(s), the test modules, and the seam doc, with each row attached to a commit. All cited script constants and both `dev/contracts/*-seam.md` documents are listed. Filed `major` here and `minor` as risk-10; both dispositioned at their filed severity, one fix. | design-v2.md |
| arch-5 | internal-panel | major | accepted | Promoted to a numbered move, **B10**, with its own commit (12). It names rules 1.11, 1.12, 1.13 and 1.14, the three module constants (`plot_results.py:108`, `plot_map.py:34`, `plot_map_forcing.py:201`), and `performance_metrics.csv`, and gives a per-rule destination table. One deviation from the suggested fix, driven by P1's restatement: `basin_area.png` goes to `hydrology_model/plots/`, not `evaluation/plots/`, because it depicts the model rather than the run (risk-4's P1 half). | design-v2.md |
| arch-6 | internal-panel | major | accepted | B4's table extended to three rows with a "question it answers" column, and the open decision arch-6 flags is **taken**: `clim_wflow_1_*` survives, on the same reasoning the owner used to retain the forcing/QA figures (it answers a model-parity question the source-grid family cannot). The disambiguating-filename decision the v1 risk deferred is also taken here rather than deferred again — the source-grid set is prefixed `source_*` (risk-9). | design-v2.md |
| arch-7 | internal-panel | major | accepted | P3 restated with the exemption inline and `model_root` named as the reason, plus the optional-when-empty clause that makes `hydrology_runs/rlz_<r>/` conform. One addition beyond the finding, needed by repo-9's relocation: engine subtrees may **add** engine-mandated directories (`forcing/`, `run_default/`, `evaluation/`), which is what lets `hydrology_runs/rlz_<r>/forcing/` exist without re-breaking P3. The `climate_projections/` shape stays out of P3's scope, as arch-7 itself notes. | design-v2.md |
| arch-8 | internal-panel | major | accepted | **Narrow-the-Goal horn taken**, not the add-the-rule horn. The Goal now claims separability ("an engine's subtree can be relocated, rebuilt, or replaced without moving generic climate data") rather than extensibility, and states the limitation explicitly: hydrology appears twice, in two shapes, at two levels, and a second hydrology engine would collide on the domain-descriptive name. The structural half of the parked naming question is deferred with the naming half and said to be so. Adding the placement rule was rejected for this milestone: it would decide the engine-naming question the owner explicitly parked at G1 (OQ-1). | design-v2.md |
| arch-9 | internal-panel | major | accepted | Mechanical machinery moves to **commit 1**, before the moves it must police; only the manifest re-record stays at the end. The single re-record is left untouched — arch-9 attacks sequencing, not the batching argument, and the design says so in the alternatives entry. Consequence carried through: `check --workflow <name>` and per-slice `semantic_tree_diff` runs become possible during the blackout, which is what makes risk-3's and repo-3's substitutes executable. Commit count grows 13 → 15 as a result; the delta is named rather than absorbed. | design-v2.md |
| arch-10 | internal-panel | major | accepted | Same facts as risk-6/repo-13, same fix, dispositioned at each filed severity. arch-10's two extra corrections are both taken: the wf2 count is corrected to the **three summary files** (the three `climate_projections/<clim_project>/plots/` PNGs do not move under B3), and the claim that the wf3 experiment snapshot joins `config/runs/` is dropped — B9 states it keeps its path and changes content only. | design-v2.md |
| arch-11 | internal-panel | minor | accepted | Both halves. (a) `COPIED_CONFIG_PATH_MAP` added to the machinery list as item 4, with a yes/no answer (yes, R07 needs new entries — `project_dir` under O-20 and the snapshot paths under B9), plus the `_is_copied_config` note that the new `weather_generator/config/weathergen_config.yml` is newly swept into the directional policy. (b) The §7 divergence is not merely stated but **amended**, in the same resolution as risk-8/repo-11. Filed `minor` here and `major` as repo-6; dispositioned at each filed severity. | design-v2.md |
| repo-1 | internal-panel | blocking | accepted | Suggested fix adopted in full: the collapsed store standardises on **`orography.nc`** (the clim_source-independent form P3-2a ext2-1 introduced), rule 3.08's `oro_path` params-string repoint joins B1's edit list, and B1 ships a unit test over `prepare_climate_data_catalog.py` asserting the chirps-branch catalog entry resolves to the emitted filename. repo-1's invisibility argument is recorded verbatim in the design — the sidecar exists only on the chirps branch while the seed config is era5, so no dry-run, test, or baseline check in the repo can see the breakage. | design-v2.md |
| repo-2 | internal-panel | blocking | accepted | **This is the fix the owner selected at G1-return (GA-1).** B1 specifies it concretely: one rule definition over region + catalog only, declared in both Snakefiles, generated from a shared `climate_store_spec()` helper so the declarations cannot drift. repo-2's closing question is answered explicitly — rule 1.10's `ancient(staticmaps.nc)` and rule 3.02's `ancient(staticgeoms/region.geojson)` are **both removed**, and wf3 keeps only `guard_ok = ancient(...)` as an ordering-only edge. P2 is restated to one producer *definition* per artifact, which is the "P2 reading that permits it" the index's table said the route needs — without that restatement the fix would have shipped a principle its own tree contradicts. The P3-1 store-dir-keyed guard contract is stated as untouched. | design-v2.md |
| repo-3 | internal-panel | blocking | accepted | The false claim is **retracted in the design's own words**: "The wf1 discharge target is *not* unchanged." All three mechanisms are recorded — every manifest key is prefixed `examples/test_local/` so O-20 moves all 18; the sidecar name is `sha1(resolved_path)[:16]` so `record` writes a new series from the post-R07 run and orphans `1f9f30a367de162f.csv`; and discharge is a tolerance comparator against a stored series, not a self-contained hash, so a re-record silently re-blesses drift. The substitute gate is repo-3's: save `output.csv` in commit 1, gate commit 14 on `check_baseline.py compare --ref <saved> --cur ...` exiting 0 *before* `record`, delete the orphan sidecar in the same commit, and state that a red `check` from commit 4 to 14 is expected. | design-v2.md |
| repo-4 | internal-panel | major | accepted | **Verified independently against the code this revision, as instructed.** `setup_gauges_and_outputs.py:55` reads `if gauges_fn is not None and os.path.isfile(gauges_fn):` and `plot_results.py:127` reads `if observations_fn is not None and os.path.exists(observations_fn):` — both short-circuit, so YAML `null` raises no `TypeError` at either site. `plot_map.py:28-31` guards only `is not None` and then computes `gauges_{basename(...)}`, yielding `gauges_None`. repo-4 is correct on both counts. The risk is rewritten around the string reaching `plot_map.py`, **O-08 is promoted out of the drive-by list into commit 12**, and the stage-1 verification assertion is replaced by the positive one repo-4 proposes. The intake constraint that every written `None` stays byte-identical still holds — it is what the existence-based guards depend on — and the design says so, so the correction does not read as licence to switch to `null`. | design-v2.md |
| repo-5 | internal-panel | major | accepted | Promoted to numbered move **B9** with its own commit (10), naming the `copy_config_files.py` signature change (one derived `output_dir` → four destinations). The generated-vs-copied conflation is fixed by a stated principle rather than an ad-hoc bin: P3 gains a rule putting run-time-generated configs in `config/generated/` and verbatim template snapshots in `config/templates/`, and the four affected rule path strings (1.02 output, 1.03 input, 1.07 output, 1.08 input) are listed. repo-5's (b) consequence drives the split of the behaviour-preservation stance into two claims, since "every item is a path move" was false. | design-v2.md |
| repo-6 | internal-panel | major | accepted | `COPIED_CONFIG_PATH_MAP` added as machinery item 4 with the required entries and with the `MIGRATION.md:167-172` lockstep obligation carried into the migration map's config-path table. repo-6's false-red argument is recorded as the reason it matters: without the entries the phase-B gate fails on pure path bookkeeping, indistinguishable from a real content regression, on the one gate the design makes the sole proof for B1–B10. Same table as arch-11(a); dispositioned at each filed severity. | design-v2.md |
| repo-7 | internal-panel | major | accepted | Suggested fix adopted. B8 now specifies the slugification (lowercase; non-`[a-z0-9]` → `_`; strip leading non-alphanumerics; collapse runs; truncate to 64), states why it deliberately differs from `validate_experiment_name`'s never-silently-lowercase stance (the helper is a suggestion writer, not a validator), and requires the suggested value to be re-validated through `validate_experiment_name` before being written. v1's supporting evidence is explicitly labelled as testing only already-conforming names, and `examples/Gabon` is named as the live counterexample. | design-v2.md |
| repo-8 | internal-panel | major | accepted | The false verification claim is corrected and replaced. The O-22 row now specifies three unit cases in `tests/test_snake_utils.py` calling `warn_if_project_dir_in_repo()` directly (in-repo warns; `<repo_root>/test_case/...` silent; absolute out-of-tree silent) plus one `test_cli.py` case asserting the warning text in the combined stream. Without this the feature would have shipped with zero coverage and the exemption branch — the case most likely to regress — never exercised. | design-v2.md |
| repo-9 | internal-panel | major | accepted | Suggested fix adopted in full. `inmaps_rlz_*_cst_*.nc` move to `hydrology_runs/rlz_<r>/forcing/inmaps_cst_<c>.nc`, mirroring B2's `hydrology_model/forcing/` and keeping `temp()`; the weathergenr-native `rlz_<r>_cst_<c>.nc` and the date CSVs stay in `weather_generator/output/` (consistent with G1 ruling OQ-4); and `outstates_*.nc` gets its row at `hydrology_runs/rlz_<r>/output/outstates_cst_<c>.nc`. B5 also picks up the two edit-surface items this implies (`downscale_climate_forcing.py:72` relative-prefix depth, `generate_weather.R:68`). This is what required P3's "may add engine-mandated directories" clause. | design-v2.md |
| repo-10 | internal-panel | minor | accepted | **State-the-subset horn taken**, not the parse-time-derivation horn. O-24 now says explicitly that it declares the config-invariant subset, names the remainder (`plot_basavg` per `wflow_outvars` entry, `plot_signatures`, per-station `clim_{station}_{period}.png`), explains why the fixture shows exactly 8 files, and narrows the `--delete-all-output` verification claim to the seed-config class. Deriving the list at parse time is recorded as a rejected alternative: it is a rule-shape change, not a declaration fix, and belongs outside a behaviour-preservation milestone. | design-v2.md |
| repo-11 | internal-panel | minor | accepted | All three loose ends closed. (a) §7 is **amended**, not merely diverged from, since a stated divergence leaves §7 unsatisfied for R06 — the amendment distinguishes the required internal rename record from the optional user-facing guide. (b) The two-locations/two-schemes problem is resolved by R07 publishing no user-facing guide (GA-2 leaves nothing to migrate), so the pair is consistent. (c) The moved file is renamed **`docs/migration-r06.md`** for `docs/` casing consistency. repo-11's separate note that §7's `migration_<topic>.md` form overrides §8's kebab-case rule is also added as a §7 line. Same resolution as risk-8/arch-11(b); dispositioned at each filed severity. | design-v2.md |
| repo-12 | internal-panel | minor | accepted | Signature corrected to `warn_if_project_dir_in_repo(project_dir, repo_root)` with `_PROJECT_DIR_EXEMPT_NAMES = frozenset({"test_case"})` and `workflow.basedir` passed from each Snakefile, matching the existing private-constant style. The reasoning is recorded in the O-22 note: deriving the root inside the module breaks under installation and an absolute constant is not portable, so getting the signature wrong makes the exemption either non-functional or over-broad. Not deferred — the wrong signature is what would make the exemption silently useless. | design-v2.md |
| repo-13 | internal-panel | minor | accepted | Same facts and same fix as risk-6/arch-10, dispositioned at its filed `minor` severity per the index's preserved divergence. repo-13's specific contribution is carried: the counts are stated as 15 live targets with the discharge row's manifest key also moving (per repo-3), and commit 14 is told to drop three stale pre-P3-1 rows *with their paths listed*, so the re-record diff is fully accounted for at adjudication. | design-v2.md |
| ext1-01 | external-r1 | blocking | accepted | Correct: v2's producer contract included `build_config`, and the delineation read `wflow_build_model.yml` `setup_basemaps` — arch-1's suggested pin, which put a model-build template inside a supposedly model-independent artifact's contract, contradicting P4 and GA-1's "region + catalog only". Resolved **within the GA-1 route** by relocating the two dataset names to new optional `shared.basin.hydrography`/`basin_index` keys (catalog entry names, defaults = the shipped template's values): the producer's contract becomes `shared.basin` + catalog, `build_config` leaves the spec, wf3 never opens the template. arch-1's intent (store and build cannot disagree about the basin) survives via a loud cross-check in rule 1.02's merge script plus the existing `shared.basin` guard digest; injection into the generated build config is a recorded rejected alternative. The P4 assertion is strengthened to the reviewer's form: figures build with neither `hydrology_model/` nor the build template on disk. | design-v3.md |
| ext1-02 | external-r1 | blocking | accepted | **Verified against the pinned Snakemake 9.6.2 source before accepting**: `persistence._input()` iterates every `job.input` with no `is_ancient` exclusion and `_input_changed()` fires on `recorded != _input(job)` — `ancient()` suppresses only the mtime trigger, exactly as the reviewer claims; the repo's own comment at `Snakefile_climate_experiment:198-201` gets its input-set protection from path-invariance, not `ancient()`, and v2 misread it. Resolved by making the producer's input set **identical and empty in both DAGs**: wf3's guard edge removed, P2(b) tightened to forbid per-workflow edge asymmetry. The reviewer's readiness-sentinel mechanism was evaluated and not adopted (recorded alternative): wf3 store consumers are already transitively guard-gated via the per-experiment sentinel chain (3.00b → 3.04 → 3.06), and store integrity moves to the params trigger (region/hydrography/window/source all ride in params), which also closes a latent staleness gap the old `ancient()` bbox input had. Both alternation directions added to the verification row. | design-v3.md |
| ext1-03 | external-r1 | major | accepted | Correct that the evidence is fixture-only and does not generalize (raster-snapped staticmaps bounds vs polygon bounds genuinely differ at other resolutions/hydrographies). Scoping horn taken: Claim 1 is restated as holding for the baseline seed-fixture class, and the scoping is shown **lossless under GA-2** — the fixture is the only pre-R07 tree in existence, so no other configuration has a "today" to diverge from. Non-seed divergence is classified explicitly as the GA-1-accepted derivation change (documented in the migration map), not a regression; the per-edge tolerance unit test is named as the configuration-independent invariant. The parameterized multi-region test horn is a recorded rejected alternative (each case needs a full hydromt build — a fixture program, not a layout milestone). | design-v3.md |
| ext1-04 | external-r1 | major | accepted | Correct: a spec sharing only outputs and params leaves script, inputs, and directives free to drift, so v2's mechanism did not enforce P2(a). `climate_store_spec()` becomes the **complete producer contract** — script path, outputs, params, plus the two fields the rule grammar cannot splat (rule name, absence of `input:`), which are test-enforced instead: the contract-equality test parses both workflows and compares the full normalized contract (name, script, input set, outputs, params). Per-workflow fields are confined to `message`/`log`/`benchmark`, none of which participates in any rerun trigger. | design-v3.md |
| ext1-05 | external-r1 | major | accepted | Correct: "parked" is a validation status, not a licence to break — O-01 deleted files the Linux config references and O-20 renames a directory the Docker runner mounts. Resolved by the retain-and-retarget horn with an explicit support decision: Linux config observation keys → the `None` sentinel (commit 2), Docker runner drops the `data/` mount (commit 2) and follows `examples/` → `test_case/` (commit 4), and `tests/test_cli.py` gains a Linux-config dry-run that executes on the `ubuntu-latest` CI leg, with a stated fallback if the dry-run proves blocked. End-to-end validation remains a non-goal (no Linux machine — intake), now stated as a decision rather than implied; the behavioural delta (Linux test config loses observation-driven extras) is documented. | design-v3.md |
| ext1-06 | external-r1 | major | accepted | Correct: a stopping point with the baseline gate red by construction is not "safe". The demote horn taken: the milestone has exactly one completed state (after commit 14), commit 6 is a pause point requiring preservation of the holding artifacts and a temporary-pause flag in the commit message. The reviewer's other horn — an interim re-record at 6 — is rejected because the intake constraint is one re-record exactly once (the batched milestone exists to avoid paying it twice); recorded as an alternative. Added beyond the ask: the abandonment path (revert the landed `r07:` commits; the untouched pre-R07 manifest is then valid again with no re-record in either direction). | design-v3.md |
| ext1-07 | external-r1 | minor | accepted | Rows added for B3 (rules 2.05/2.06 with the module constants `get_change_climate_proj_summary.py:80,88,93`, `plot_proj_timeseries.py:223`), B4 (producer named: new rule 1.15 `plot_climate_source` in `Snakefile_model_creation`, new `climate_analysis/plot_climate_source.py` module, P4-assertion + source-PET tests), and B8 (helper in `snake_utils.py` + new `scripts/suggest_experiment_name.py` CLI with a pinned invocation and target-config contract: reads `project.project_dir`, writes `workflows.climate_experiment.experiment_name` only-if-absent, value re-validated before write). B7's empty tests cell also corrected (`test_interchange_contracts.py:570-571,592`). | design-v3.md |
| ext2-01 | external-r2 | blocking | accepted | **Owner arbitration, 2026-07-28** (external round cap of 2 exhausted; the ruling stands in place of a further reviewer verdict): accepted, fix = **symmetric catalog input declared identically in both DAGs** — the reviewer's first suggestion, chosen because the ext1-02 oscillation came from *asymmetric* input sets, so a symmetric one cannot reproduce it, and it closes a staleness gap that predates R07 (today's rule 3.02 carries the catalog only as a `params` path string, `Snakefile_climate_experiment:204`, so an in-place catalog edit retriggers nothing today either — driver evidence at arbitration). The **owner-mandated verification ran and settled the route**: wf1 and wf3 read the *same* `project.data_sources` key (`Snakefile_model_creation:31`, `Snakefile_climate_experiment:34`), a single catalog path whose cross-config agreement the rule 3.00b drift guard already enforces (`project` is a guarded section); the experiment-level catalog composed at `Snakefile_climate_experiment:344` belongs to rule 3.09 `downscale_climate_realization`, not the producer — the sets are identical, so the digest-in-params fallback was not needed and is recorded as the on-file alternative should a future producer catalog set diverge. The freshness boundary is defined (catalog file in; data behind an unchanged entry out, with the catalog-edit convention and the `--forcerun` escape hatch documented), and the mandated test added: a catalog edit schedules extraction exactly once, after which both workflows' `--dry-run`s schedule nothing. | design-v4.md |
| ext2-02 | external-r2 | major | accepted | **Owner arbitration, 2026-07-28**: accepted, fix required — the author had independently flagged the same gap as its own residual risk, so the finding was not in dispute. Fix per the ruling's named form (the reviewer's first option): the contract-equality test normalizes and compares **every** content- or execution-affecting directive (`conda`, `container`, `envmodules`, `wrapper`/`notebook`, `shadow`, `threads`, `resources`, `priority`, `retries`, `group`, `cache`, `wildcard_constraints`) and **fails on unknown or unexpected per-workflow fields** — the allowed-local set is exactly {`message`, `log`, `benchmark`}, deny-by-default, so a future Snakemake version adding a directive surfaces as a loud test failure rather than silently widening the hole. The sweep also protects the trigger family: `conda`/`container` feed the software-environment rerun trigger, so a one-DAG environment change would re-fire extraction on alternation exactly as ext1-02's asymmetry did. Verified that neither Snakefile uses any such directive today (sweep starts absent-equals-absent). The reviewer's alternative — one shared rule module via `include:` — was evaluated and not adopted (first `include:` in the repo; the `W.NN` banner and workflow-scoped log/benchmark paths would need per-workflow parameterization channels that need the same policing; the deny-by-default test is needed for the allowed-local set anyway); recorded in Alternatives. | design-v4.md |
| ext2-03 | external-r2 | minor | accepted | **Owner arbitration, 2026-07-28**: accepted, fix required. The reviewer's first horn taken: commit 1 is confined to path-map, merge-class, comparison machinery, and reference capture; the `check_baseline.py` `TARGETS` + `PROJECT_DIR_DEFAULT` retarget moves **atomically into commit 4** with the fixture rename, and commit 4 is named that edit's sole owner — resolving the contradictory dual ownership. The document's repeated "red from commit 4 to commit 14" statements are thereby made true rather than reworded, and the gate-blackout section now states the boundary explicitly (`check` green through commit 3, red from 4). Checked against ext1-06's accepted resolution as the arbitration required: commit 6 remains a pause point with the gate red because it sits inside the commit-4-to-14 window — the corrected start boundary leaves that reasoning intact. | design-v4.md |

## Notes

**Nothing rejected, nothing deferred.** Every finding survived checking, and the
two I was directed to verify myself both held: repo-4's rediagnosis of the
`"None"` sentinel is correct at both cited lines (checked in
`setup_gauges_and_outputs.py`, `plot_results.py`, `plot_map.py`), and arch-1's
`get_region_preview.py` observation understates the problem — the module does not
import at all on the pinned hydromt.

**Where the adopted route differs from the suggested fix**, at a glance:
risk-1 (GA-1's shared rule, not (i) or (ii)); risk-3 (cut line restated rather
than commits reordered); risk-5 (GA-2's horn (i); horn (ii) forbidden); arch-1
(its fix adopted, its alternative rejected by GA-1); arch-2 (merge class over
`--retire`); arch-5 (`basin_area.png` to `hydrology_model/plots/`, not
`evaluation/plots/`); arch-8 (narrow-the-Goal horn); repo-10 (state-the-subset
horn).

**Findings that produced changes beyond their own ask**, flagged so a later
reviewer can trace them: repo-2 forced the P2 restatement (a principle change,
not just a B1 change); repo-9 forced P3's "may add engine-mandated directories"
clause; repo-5 and risk-7 together forced the behaviour-preservation stance to
split into two claims; arch-9 forced the commit count from 13 to 15.

**External round 1 (rows ext1-01 … ext1-07, appended by the r2 revision):
7 accepted, 0 rejected, 0 deferred — 2/2 blocking, 4/4 major, 1/1 minor.** No
rejection, so no arbitration halt. Both blocking findings were fact-checked
before acceptance: ext1-02's `ancient()` claim against the pinned Snakemake
9.6.2 source (the reviewer is right; v2's mechanism was wrong), and ext1-01's
coupling claim against `climate_store_spec`'s v2 signature and the template
read (textually and substantively right). Three findings are resolved by a
route other than their own suggested fix, each recorded in its rationale
column and in the design's alternatives: ext1-02 (no-inputs producer instead
of a readiness-sentinel rule), ext1-06 (demote horn, not the interim-re-record
horn, which the re-record-once intake constraint forbids), ext1-03 (scoping
horn with a GA-2 losslessness argument, not parameterized multi-region
tests). ext1-01 and ext1-04 are resolved **within** owner ruling GA-1's
shared-rule route, as required — the route survives; its contract is repaired.

**One new observation logged during revision, from no finding:** O-25 —
`blueearth_cst/model/get_region_preview.py` raises `ModuleNotFoundError` on
import under hydromt 1.3.1 (`hydromt.cli.api` was removed in hydromt 1.x), has no
rule, no test, and no other module referencing it. It is retired in commit 7.


---

# Post-acceptance verification pass — 2026-07-28

*Dispatched after acceptance, at the owner's request, to close the audit gap the
round cap created: the accepted version's final changes were made under owner
arbitration and no external reviewer had verified them. **Not external round 3** —
the cap stands; this pass had no author waiting to revise and its findings are
owner decisions, not automatic rework.*

*Reviewer: the same headless `codex exec` (`gpt-5.6-sol`) that ran rounds 1 and 2,
read-only sandbox. Scoped to three priorities: verify the three arbitrated fixes,
check the two post-acceptance editorial corrections nobody had reviewed, and make
a fresh pass for the failure mode that produced those corrections — counts and
inventories that do not reconcile.*

**Outcome.** The arbitration delta is substantially verified: ext2-01 and ext2-03
confirmed resolved, and both editorial corrections confirmed correct (the reviewer
independently enumerated the same ten within-tree movers). ext2-02's fix is sound
in principle but its enumerated directive universe is incomplete (pv-3). Two new
`major` findings surfaced — a commit-count contradiction (pv-1) and a manifest
inventory contradiction (pv-2) — both of the same class as the defects that
survived the full loop, and neither previously raised by any round.

**Disposition of pv-4:** accepted and fixed immediately — the 14 → 10 editorial
correction had not been propagated to `project-layout-task-brief.md`, and this
map's note read present-tense. Both corrected 2026-07-28; design, map, and task
brief now agree.

## Verdict
verdict: revise
doc_version: dev/milestones/r07/project-layout-design.md (accepted 2026-07-28)

## Arbitration-delta verification
- ext2-01: verified resolved. Both workflows resolve the same `project.data_sources` path, and identical plain inputs cannot reproduce the prior input-set oscillation. The catalog-file freshness boundary is coherent and explicitly excludes unchanged backing data.
- ext2-02: defective — see pv-3. Deny-by-default is sound in principle, but the stated pinned-Snakemake directive universe is incomplete.
- ext2-03: verified resolved. Commit 1 leaves `check_baseline.py` untouched; commit 4 solely and atomically owns its retarget with the fixture rename. Nothing else in commits 1–3 starts the blackout.
- editorial 14->10: correct. The 10 movers are three wf1 plots, two project-level config snapshots, three wf2 summaries, and two wf3 indicators.
- editorial CI invariant: correct. Added tests invalidate fixed pass counts; zero failures plus comparison of skip counts/reasons against the `-rs` pre-R07 reference is checkable on both CI legs.

## Findings

### pv-1  [major]
- section: Commit plan
- finding: The plan requires exactly 15 commits, but after enumerating commits 1–15 it says O-07, O-09, and O-10 must be landed as “their own small commits.” None is assigned to an enumerated commit.
- rationale: The implementation cannot simultaneously preserve the settled 15-commit framing and give all three drive-bys separate commits. Following the prose literally produces 18 commits; following the numbered plan silently omits the accepted fixes.
- suggested_fix: Assign O-07 to commit 11, O-09 to commit 12, and O-10 to commit 15, or explicitly defer them; remove the “own small commits” instruction.

### pv-2  [major]
- section: Verification plan
- finding: The declaration-fixes row requires newly declared targets to be added to the manifest, while the commit-14 row says the manifest diff is “path-and-snapshot-only plus three stated orphan deletions.” These cannot both hold. The migration map exposes the unresolved scope: five O-24 outputs and three B4 figures are still only “candidates,” with `TARGETS` membership left to implementation.
- rationale: If the new rows are added, the stated commit-14 diff assertion fails; if they are omitted, the newly declared outputs receive no promised baseline coverage. The implementer also lacks an exact post-R07 `TARGETS` inventory.
- suggested_fix: Specify the exact new `TARGETS` and manifest rows, then amend the commit-14 expected diff to include those additions alongside path changes, snapshot-content changes, and the three deletions.

### pv-3  [minor]
- section: The substantive moves
- finding: Snakemake 9.6.2’s rule grammar includes execution-affecting directives absent from the claimed complete universe: `containerized`, `handover`, `localrule`, `default_target`, `template_engine`, and `cwl`. The generic unknown-non-default rejection still prevents silent asymmetry if implemented exactly as stated, but these fields would be forbidden rather than normalized and compared.
- rationale: A legitimate symmetric use of one of these pinned directives would fail the contract test, and an implementation derived only from the enumerated fields could miss it. The document therefore does not substantiate its “every directive” claim.
- suggested_fix: Extend the normalized universe to the omitted pinned directives and define unknown detection against `RuleInfo` defaults plus effective workflow-level rule state.

### pv-4  [minor]
- section: Behaviour-preservation stance and baseline consequence
- finding: The corrected value 10 was not propagated to the implementation handoff: `project-layout-task-brief.md` still says 14 within-tree movers. The supposedly authoritative migration map also still says the design prose reports 14.
- rationale: An implementer using the committed task brief receives contradictory inventory totals and may report or validate against the superseded count.
- suggested_fix: Change the task brief’s 14 to 10 and rewrite the migration-map note as historical rather than present-tense.
