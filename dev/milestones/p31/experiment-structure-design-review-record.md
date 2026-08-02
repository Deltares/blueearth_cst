# P3-1 — Experiment structure — design review record

Consolidated audit trail of the `p31-experiment-structure` design-review-loop
run (started 2026-07-23, closed 2026-07-23; full variant; genre
decision-record; author binding `cst-architect`; driver = interactive session
per `workflow-driver`). The accepted design is
`experiment-structure-design.md` beside this file; the authoritative scoping
record is `experiment-structure-intake.md`.

## Verdict table

| Round | Reviewer | Doc version | Verdict | Findings |
| --- | --- | --- | --- | --- |
| internal (risk) | critical-thinker (Opus) | design-v1 | revise | 1 blocking, 3 major, 3 minor |
| internal (architecture) | cst-architect lens (Opus) | design-v1 | revise | 1 major, 8 minor |
| internal (repo-fit) | python-engineer lens (Opus) | design-v1 | revise | 2 major, 4 minor |
| external r1 (clean-room) | GPT via codex exec | design-v2 | revise | 1 blocking, 3 major |
| external r2 (regression) | GPT via codex exec | design-v3 | revise | 2 major, 1 minor |
| arbitration | user (2026-07-23) | design-v3 → v4 | ext2-1/-2/-3 accepted, fix required; cap stands | — |
| G2 | user (2026-07-23) | design-v4 | APPROVED (zero editorial edits) | — |

Version series: v1 (draft) → v2 (answers internal panel, 22 findings) → v3
(answers external r1; incl. the 5-case Snakemake trigger-matrix probe) → v4
(arbitration revision, confined to ext2-1..3; driver scope-check: 32/32 diff
hunks within the arbitrated sections). Dispatches: 6 Opus + 1 Fable
(fable-escalation-6a — ext2-1 re-raised ext1-2). Two author-spawn stream
timeouts recovered at escalation rung 1 (same-thread resume); no driver
authorship of design content (landing status-header swap logged as the sole
editorial edit).

Convergence: reached via round-cap arbitration — the user's rulings stand in
for the reviewer verdict the cap forecloses (stage-6a authority); the G2
approval covers design-v4 unchanged.

---

## Internal panel — aggregation index

(Verbatim per-lens reviews follow below.)

# Internal review index — p31-experiment-structure

Driver aggregation of the stage-2 lens panel over `design-v1.md`. Grouping
only — every original ID, severity, and full text lives in the per-lens
artifacts (`internal-review-risk.md`, `internal-review-architecture.md`,
`internal-review-repo-fit.md`), verbatim and immutable. No finding is
deleted or re-graded here.

**Verdicts:** risk = revise · architecture = revise · repo-fit = revise
(all on `doc_version: design-v1.md`).
**Totals:** 22 findings — 1 blocking, 6 major, 15 minor.
**Self-containment:** all three lenses confirmed the doc reviewable
stand-alone; citation spot-checks passed (repo-fit, architecture).

## Blocking

- **risk-1** — §4: the CHIRPS orography sidecar has a tested consumer
  (`prepare_climate_data_catalog.py:77-84`) hardcoding the old path at the
  old depth; the store move breaks it, and every proposed gate runs ERA5 so
  the break escapes the design's own verification.

## Cluster A — drift-guard DAG wiring (the panel's core theme)

- **arch-1** (major) + **repo-1** (major) — same defect, found
  independently: the guard rule as specified is an orphan — wf3 has five
  independent root rules and no chokepoint, so "the DAG threads through the
  sentinel" is false without per-root `input:` edits that the commit plan's
  "additive, existing paths unchanged" claim excludes. As written the guard
  passes every gate while gating nothing.
- **risk-2** (major, distinct claim, same section) — the guard's `input:`
  consumes only the wf1 snapshot while §2a claims wf2's
  `workflows.climate_projections` section is guarded; wrong-snapshot/wrong-
  section comparison, false-fire by edit order.
- Related minors: **repo-5** + **arch-5** (Cluster A2) — the reused
  directional `COPIED_CONFIG_PATH_MAP` normalization is a no-op between two
  post-R6 configs; normalization side/direction unpinned, gate 2(d)
  behavior undefined.

## Remaining majors (standalone)

- **risk-3** — §4 staleness: the store key omits catalog source-data
  identity; a repointed/updated catalog `uri` silently reuses a stale
  extraction (`ancient()` blocks rebuild).
- **risk-4** — §6: the path-map re-record can mask a value regression —
  `semantic_tree_diff` keys on identical relpaths, so an omitted/wrong
  mapping degrades to benign MISSING+EXTRA; the gate must assert those
  empty modulo an explicit allowlist.
- **repo-2** — §2a/commit 2: `gather_benchmarks` `params.parts_dir`
  (`benchmarks/_parts`) is a dry-run-blind string path the benchmarks move
  never calls out and no gate smokes; silent empty/stale `wf3_benchmarks.md`.

## Minor clusters

- **Cluster B (key rendering):** risk-5 + arch-4 — `slugify_window`
  truncates to day resolution; sub-day-divergent windows collide to one key
  and silently share an extraction. Unstated invariant.
- **Cluster C (path-map over-scope):** repo-6 + arch-3 —
  `semantic_tree_diff.py` already excludes `logs`/`benchmarks`; §6's
  path-map layer over-scopes.

## Standalone minors

- **risk-6** — §5's cross-subtree audit misses `setup_time_horizon.py:51`;
  safe only via C1's unstated pin of `wflow_data/`.
- **risk-7** — §5a commits to absolute `path_static` while its enabling
  assumption is `[UNVERIFIED]`; hydromt's `config_root=...resolve()` is the
  exact re-relativization knob.
- **arch-2** — `params.output_path` (132/151) and `params.exp_dir` (265)
  absent from the change inventory; safe only if `exp_dir` redefinition is
  the mechanism, never stated.
- **arch-6** — `check_baseline.py` TARGET line 114 (config snapshot) is
  `{project_dir}`-rooted; needs a different repoint than the two results
  rows.
- **arch-7** — "pure hydrology_model" claim needs a gate-3 assertion that
  the downscale `r+` open doesn't flush into model_root.
- **arch-8** — the §4 `[UNVERIFIED]` has a nameable hit: dead `__main__`
  fallback `extract_historical_climate.py:210-217`; closable now.
- **arch-9** — "seven collision classes verified" is six live + one
  reserved (no wf3 plots producer).
- **repo-3** — `p31:` commit prefix not registered in the roadmap's commit
  strategy; needs a one-line registration.
- **repo-4** — root `MIGRATION.md` extension conflicts with naming.md §7
  (`dev/<milestone>/migration_<topic>.md`); R06-scoped and git-ref-anchored.


---

## Internal review — risk lens (verbatim)

# Internal design review — RISK & ASSUMPTIONS lens

Target: `dev/working/design-runs/p31-experiment-structure/design-v1.md`
Lens: risk / assumptions / failure-modes (adversarial). Reviewer never edits the design.

```yaml
verdict: revise
doc_version: design-v1.md
findings:
  - id: risk-1
    severity: blocking
    section: "4. Historical-climate store: dataset keying vs the window (Chirps orography sidecar)"
    finding: >
      The design asserts "no script edit needed for the sidecar" and its [UNVERIFIED]
      grep only checks the extract producer and the Snakefile (lines 101/165). It misses
      a second, tested consumer — prepare_climate_data_catalog.py:77-84 — which hardcodes
      `../../climate_historical/raw_data/<src>_orography.nc` relative to the realization NC.
      That path breaks twice under the new tree: the target moves from raw_data/ to the
      dataset-key dir, AND the `../..` depth is now one level short (realization dir goes
      from 2 to 3 levels below project_dir).
    rationale: >
      For any chirps / chirps_global experiment, wf3 rule 3.08 would write a
      data_catalog_climate_experiment.yml whose `<src>_orography` entry points at a
      nonexistent file, causing a runtime failure when downscale_climate_realization
      (3.09) resolves that catalog source. The failure is invisible to the design's OWN
      verification: every gate (3–6) runs the era5 seed config, and the chirps branch is
      only reached when clim_historical is chirps — an anchor-1 experiment-variable value.
      The design ships a false "no edit needed" claim that would land a broken chirps path.
    suggested_fix: >
      Add prepare_climate_data_catalog.py:77-84 to the store-keying edit surface: rewrite
      the orography lookup to derive from the same slugify_window key (not flat raw_data/)
      and from the correct depth (or make it absolute from a passed param). Add a chirps
      smoke to the gate ladder, or at minimum a unit test on the rewritten lookup, since
      the era5 gates cannot exercise it.
  - id: risk-2
    severity: major
    section: "3. Drift guard mechanics / 2a. Shared config"
    finding: >
      §2a states the wf1 AND wf2 snapshots "become the drift-guard baseline," and §3's
      compare set includes `workflows.climate_projections`. But §3's rule `input:` lists
      ONLY `snake_config_model_creation.yml` (wf1's snapshot). So the experiment's
      climate_projections section is validated against wf1's copy of it, not wf2's.
    rationale: >
      Concrete failure by edit order: a user edits workflows.climate_projections and
      re-runs wf2 (not wf1). wf1's snapshot still holds the OLD projections section, so
      the guard fires against a legitimately-current experiment config — a false block of
      a valid wf3 run. Conversely, an edit that never re-runs wf1 leaves the guard
      comparing against a stale snapshot for a section wf1 does not even own. Compounding
      this: wf3 does not consume workflows.climate_projections at all (projections are a
      plausibility overlay, per AGENTS.md Background), so guarding it — against the wrong
      snapshot — adds a false-positive surface with no provenance benefit to wf3.
    suggested_fix: >
      Either (a) drop workflows.climate_projections from the guarded set (it does not gate
      any wf3 input), or (b) if kept, add snake_config_climate_projections.yml as a second
      `input:` and compare that section against the wf2 snapshot. Reconcile §2a and §3 so
      the snapshot(s) and the compared sections line up one-to-one. Also resolve open
      question 4 (§12) here: state which `shared.*` keys wf3 actually consumes.
  - id: risk-3
    severity: major
    section: "4. Historical-climate store (Staleness semantics)"
    finding: >
      §4 claims "the key encodes the full identity of the extraction inputs (dataset +
      window + region) ... There is no hidden mutable input, so a present keyed file is
      never stale within a project." The extraction also depends on the catalog DATA
      behind the source (DATA_SOURCES -> get_rasterdataset), which is NOT in the key.
    rationale: >
      If a catalog `uri` is repointed to updated source data (or the underlying staged
      file changes) while dataset+window stay the same, the key is unchanged; Snakemake
      finds the keyed file present and skips re-extraction, and `ancient()` on the
      consumer guarantees no rebuild cascade. Result: silently reused stale extraction ->
      wrong stress-test inputs, with no diagnostic. The drift guard does not catch this
      either — it normalizes and compares catalog PATHS, never hashes catalog CONTENT. The
      "never stale within a project" claim is the load-bearing justification for zero-copy
      reuse and is overstated.
    suggested_fix: >
      Weaken the claim to "never stale for a fixed catalog," and state the assumption
      explicitly (catalog source data is immutable within a project's lifetime — true for
      the CST rapid-assessment target, but it is an assumption, not a guarantee). Note the
      escape hatch: manual key-dir deletion forces re-extraction. Defer any content-hash
      staleness check to reproducible-computing, consistent with the §3 snapshot-freshness
      deferral.
  - id: risk-4
    severity: major
    section: "6. Behavior-preservation stance (path-map re-record)"
    finding: >
      The value-identity proof depends on a NEW old->new path-map layer for
      semantic_tree_diff.py, which today keys strictly on identical relpaths (diff_trees,
      lines 356-361: `ref_files & cur_files`). A file with no path-map entry, or a wrong
      entry, does not register as a value diff — it registers as MISSING (in ref) + EXTRA
      (in cur).
    rationale: >
      This is precisely how a genuine value regression can hide inside a path move. If the
      path map omits or mis-maps a moved output, semantic_tree_diff never content-compares
      the pair; the operator sees a benign-looking missing/extra pair (expected during a
      restructuring milestone) and waves it through, while the numbers may in fact have
      changed. The design treats the path map as a mechanical translation but does not
      require a completeness check (every moved output has exactly one mapping, and no
      residual missing/extra survives except by documented exception).
    suggested_fix: >
      Make the milestone gate assert: after applying the path map, the missing/extra sets
      are EMPTY (modulo an explicitly enumerated, justified allowlist). A nonempty
      unexplained missing/extra set is a gate failure, not a pass — otherwise a moved-file
      value regression is indistinguishable from a benign move. Spec the path map as
      total over the wf3 output set in the task-brief.
  - id: risk-5
    severity: minor
    section: "4 / 9. Dataset-key rendering"
    finding: >
      slugify_window drops time-of-day (`...T00:00:00` -> `20000101`). Key distinctness
      therefore assumes windows never differ only sub-daily, and the design asserts the
      rendering is "deterministic and reversible enough" without stating that invariant.
    rationale: >
      For CST's daily forcing this is safe, but two experiments whose windows differ only
      in the (currently always-midnight) time-of-day would collide onto one key dir and
      silently share an extraction — a wrong reuse. It is latent, not active, but the
      determinism claim omits the assumption that makes it true.
    suggested_fix: >
      State the invariant explicitly (window endpoints are day-resolution; time-of-day is
      dropped by contract) and, cheaply, have slugify assert HH:MM:SS == 00:00:00 or fold
      it in, so a future sub-daily window fails loud instead of colliding.
  - id: risk-6
    severity: minor
    section: "5. Run-dir move / 1a. C1 (unstated cross-subtree dependency)"
    finding: >
      A grep of blueearth_cst/ for cross-subtree relative paths surfaces a wf1 consumer
      the design never enumerates: setup_time_horizon.py:51 hardcodes
      `../climate_historical/wflow_data/inmaps_historical.nc` for the run_default toml.
      It is unaffected (both endpoints stay project-level per C1), but the design's
      cross-subtree-path audit (§5 toml table) is presented as exhaustive and is not.
    rationale: >
      The audit's credibility rests on completeness. This path happens to be safe ONLY
      because C1 keeps wflow_data/inmaps_historical.nc exactly put — i.e. C1's
      preservation is load-bearing for a wf1 path the design never connects to it. If a
      later reader "tidied" wflow_data/ under the dataset store (a plausible future move),
      this wf1 default run would break silently. The dependency should be recorded so the
      constraint is visible.
    suggested_fix: >
      Add setup_time_horizon.py:51 to §1a C1 as the concrete wf1 consumer that pins
      wflow_data/inmaps_historical.nc in place, closing the "confirm no other consumer"
      [UNVERIFIED] with a positive result rather than an open grep.
  - id: risk-7
    severity: minor
    section: "5a. Absolute path_static (residual UNVERIFIED)"
    finding: >
      The design proceeds on an absolute-path decision for input.path_static while flagging
      [UNVERIFIED: hydromt_wflow config.write may re-relativize absolute values]. This
      unknown is load-bearing for the run-dir move's correctness, and the design defers it
      to the gate-3 smoke rather than resolving it before committing the approach.
    rationale: >
      If config.write DOES re-relativize (or normalizes against config_root), the chosen
      absolute-path resolution silently produces a wrong path_static and Wflow reads the
      wrong / no staticmaps — the exact failure the design says gate 3 will catch, but
      only after the toml-rewrite commit (commit 3) is written. The risk is bounded (the
      smoke catches it) but the design commits to "prefer absolute" as settled in §5a while
      the mechanism that could invalidate it is unverified. This is acceptable to proceed
      ONLY because a cheap, decisive smoke exists — worth stating as such.
    suggested_fix: >
      Keep the decision but label it provisional-pending-gate-3, and pre-commit a 5-line
      isolated check (write one toml via mod.config.write with an absolute path_static,
      read it back) before commit 3, so the approach is not built on an unverified upstream
      behavior. Note config.write here already passes config_root=...resolve() (line 116),
      which is the exact knob that could trigger re-relativization.
```

---

## Elaboration

### Framing check — does the design earn its "values identical; paths move" thesis?

Mostly yes, and the by-construction argument in §6 is sound for the computational core:
no scientific script logic is edited. The thesis is NOT refuted at the level of the wflow
run or the weathergen chain. It is undermined at two seams the design under-weights:

1. **Generated artifacts that bake paths in.** The design reasons about *source scripts*
   moving cleanly, but wf3 emits a **generated data catalog** (rule 3.08,
   `data_catalog_climate_experiment.yml`) whose contents are computed from realization
   paths. Moving the realization subtree changes what that catalog points at. For era5 the
   catalog only lists realization NCs (which move together, absolute `uri`s — safe). For
   chirps it additionally computes a cross-subtree orography path with a hardcoded
   `../../climate_historical/raw_data/` literal (risk-1) — the one place "paths move
   cleanly" is false. The design's self-audit looked at hand-written scripts and missed a
   generated-artifact consumer.

2. **The verification plan is confident but era5-only.** §7 is admirably honest about
   dry-run blindness and adds execution smokes — but every smoke runs the seed config
   (era5). The plan is structurally blind to the chirps branch, which is exactly where the
   one genuine path break lives. "Honest about dry-run blindness" is not the same as
   "covers the config space that anchor 1 admits."

### On the dismissed alternatives (§8) — are any rejections strawmen?

Checked A1–A6; none are strawmen and the reasoning is factually grounded:

- **A1 (layered configs):** correctly rejected on the real `configfiles[0]`-to-R
  forwarding contract (verified: Snakefile line 17, forwarded as `config_path` at rule
  params 130/149). The claim that the drift guard is the "cheaper substitute" is fair.
- **A6 (`dir_input` vs absolute path_static):** the rejection rationale — `dir_input` also
  reroutes `path_forcing`/`path_output` and would entangle the working forcing relpath — is
  consistent with the script (line 73 computes path_forcing as a relpath from the toml dir;
  a `dir_input` change would indeed disturb its resolution). Not a strawman.
- **A4 (full-period store + subset):** correctly grounded — the extraction already subsets
  at read time (`get_rasterdataset(time_range=...)`, verified lines 91/159), so there is no
  full-period artifact to subset without a contract change. Honest.

The §8 rejections are the strongest part of the document. No finding there.

### On the `state.path_input` "inert" claim (task pressure point) — HOLDS.

Verified: template `reinit = true` (`config/templates/wflow_sbm.toml:26`), and
`downscale_climate_forcing.py`'s `setup_config` does not override `model.reinit`, so wf3
cold-starts and never reads input states. `state.path_input` is set (line 70) but unread.
The design's claim is correct, and rewriting it "for future warm-state safety" is a
reasonable low-risk hedge. No finding — but note this inertness is a *current-template*
property; if a future config enabled warm states, the path would become live, so the
design's instinct to fix it anyway is right.

### Self-containment of the doc

Adequate. I could verify every load-bearing claim from the cited paths without needing
context the doc withheld; the line-number citations were accurate throughout (spot-checked
exp_dir line 62, run-dir lines 227/245/247/248/260, check_baseline EXPERIMENT_NAME line 65
/ resolve line 124 / TARGETS 112-114, semantic_tree_diff diff_trees keying). One stale
external note (AGENTS.md still says wf3 scripts live in `src/*.py`; they are under
`blueearth_cst/experiment/`) is an AGENTS.md defect, not a design defect — the design uses
the correct `blueearth_cst/` paths. Not a finding against this doc.

### Verdict rationale

`revise`, not `approve` (one blocking finding forbids approve) and not `reject` (the core
architecture is sound and the blocking finding has a contained, mechanical fix). risk-1
must be closed before implementation; risk-2/3/4 are meaningful and each has a clear fix;
risk-5/6/7 are worth folding in but do not gate the milestone.


---

## Internal review — architecture lens (verbatim)

verdict: revise
doc_version: design-v1.md
findings:
  - id: arch-1
    severity: major
    section: "3. Drift guard mechanics / 10. Commit plan (commit 1)"
    finding: >
      §3 states the guard is a first rule that "every downstream wf3 rule depends
      on (via a sentinel output the DAG threads through)", while §10 commit-1
      states the guard is "additive; existing wf3 paths unchanged." These cannot
      both hold. The wf3 DAG has five independent root rules — copy_config,
      extract_climate_grid, climate_stress_parameters, prepare_weagen_config,
      prepare_weagen_config_st — and no single chokepoint (nothing consumes
      copy_config's snapshot output except `rule all`). For a sentinel to gate
      every downstream rule, it must be added as an `input:` to each root rule.
      That edit is neither "additive/unchanged" nor enumerated in the §1 table or
      the §2 change inventory.
    rationale: >
      As specified, the design is self-contradictory and the guard does not gate
      what it claims to. If the sentinel is left additive (commit-1 wording), it
      is an orphan output: extract_climate_grid, climate_stress_parameters, and
      the prepare_weagen rules run in parallel with — or before — the guard,
      producing the extraction netCDF, stress-test parameter grid, and weagen
      configs against a project whose consistency was never verified. That is
      exactly the divergence the guard exists to catch. If instead the sentinel is
      genuinely wired into every root (as §3 requires), then commit-1's
      "existing wf3 paths unchanged" is false and the change inventory is
      incomplete, so an implementer working from the doc will under-scope the edit
      and the milestone verification (gate 1 dry-run) will pass while the guard
      silently fails to gate the roots.
    suggested_fix: >
      Specify the sentinel wiring explicitly: add the guard sentinel as an
      `input:` to each of the five wf3 root rules (or, at minimum, to the
      model-consuming rules downscale_climate_realization + run_wflow AND the
      three data-producing roots extract_climate_grid / climate_stress_parameters
      / prepare_weagen_config). Mark those per-rule `input:` additions in the §1/§2
      change inventory, and correct commit-1's "existing wf3 paths unchanged"
      claim — commit 1 does touch existing rule inputs. Alternatively, drop the
      "every downstream rule depends on it" claim and gate only the model-run
      rules, but then §3's provenance-for-all-artifacts framing must be softened.
  - id: arch-2
    severity: minor
    section: "2. Decision — the target tree / 1. Context (inventory)"
    finding: >
      The change inventory (§1 table, §2) does not enumerate the two params-level
      path expressions that carry realization output locations:
      prepare_weagen_config `params.output_path = f"{exp_dir}/"` (Snakefile line
      132) and prepare_weagen_config_st `params.output_path =
      f"{exp_dir}/realization_.../"` (line 151), nor the export_wflow_results
      `params.exp_dir = exp_dir` (line 265). These flow automatically only if the
      implementation redefines the single `exp_dir` binding (line 62); the design
      never states that redefinition is the mechanism.
    rationale: >
      §7 correctly warns that --dry-run is blind to params: string paths. If the
      implementer edits per-rule `output:` expressions individually rather than
      redefining `exp_dir`, these params silently keep writing to the old
      `climate_{experiment}/` location, and the miss is invisible to the cheap
      gates — the weathergen R script writes realization netCDFs to the stale dir
      and only an execution + collision smoke (gates 4/5) would surface it.
    suggested_fix: >
      State the implementation mechanism explicitly: "redefine `exp_dir =
      f'{project_dir}/experiments/{experiment}'` at line 62; all exp_dir-derived
      output and params paths (incl. lines 132, 151, 265) move with it." Then the
      only manual edits are the project_dir-rooted wf3 log/benchmark/config paths
      (§2a) and the run-dir/basin_dir paths (§5).
  - id: arch-3
    severity: minor
    section: "6. Behavior-preservation stance (step 2, path map)"
    finding: >
      semantic_tree_diff.py excludes `logs/` and `benchmarks/` from the tree walk
      entirely (`EXCLUDED_DIR_NAMES = frozenset({"logs","benchmarks",".snakemake"})`,
      line 91; `_is_excluded` matches any path part, line 307). §6 step 2 describes
      the path map as handling "a moved file" generically without noting that the
      wf3 logs/benchmarks move (§2a) is invisible to the diff and therefore needs
      NO path-map entry. The path map only needs entries for the config snapshot,
      results CSVs, the run-dir tomls/csvs, and the extraction netCDF.
    rationale: >
      Over-claiming the path-map scope risks the implementer wasting effort adding
      (or debugging the absence of) log/benchmark entries, or misreading a clean
      diff as a bug. Scoping it correctly also confirms the log/benchmark move is
      provably behavior-neutral at the gate for free.
    suggested_fix: >
      Add one sentence to §6 step 2: "logs/ and benchmarks/ are excluded from the
      tree walk (EXCLUDED_DIR_NAMES), so their move needs no path-map entry; the
      map covers only the config snapshot, results, run-dir tomls/csvs, and the
      keyed extraction netCDF."
  - id: arch-4
    severity: minor
    section: "4. Historical-climate store — key rendering"
    finding: >
      The store-reuse contract (§4) treats file existence at the slugified key
      path as extraction identity, but `slugify_window()` strips time-of-day
      (`YYYY-MM-DDTHH:MM:SS` → `YYYYMMDD`). Two windows differing only below the
      day boundary render to the same key yet call
      get_rasterdataset(time_range=(starttime,endtime)) with different bounds, so
      the second experiment would reuse the first's extraction against a different
      requested window.
    rationale: >
      In practice windows are daily (`T00:00:00`), so the collision is unlikely —
      but the reuse mechanism has zero guard against it: a present keyed file is
      "never stale" by §4's own claim, so a sub-day-divergent window silently gets
      the wrong netCDF and the experiment's config snapshot (which records the true
      window) would then disagree with the on-disk data it consumed.
    suggested_fix: >
      Either constrain the key to assert windows are day-aligned (reject/normalize
      time-of-day at slug time), or include the full ISO instants in the key hash.
      A short deterministic hash of the exact (clim_source, starttime, endtime)
      tuple sidesteps both the `:` illegality and the truncation collision.
  - id: arch-5
    severity: minor
    section: "3. Drift guard — Normalization"
    finding: >
      §3 cites reuse of `_normalize_config_paths` + `COPIED_CONFIG_PATH_MAP`
      (semantic_tree_diff.py:238-254) so "flat compares equal to binned," but that
      map is directional (OLD flat → NEW binned) and, in compare_copied_config, is
      applied to exactly one side after a reflexivity guard. For the drift guard
      both operands are post-R6 configs, so the mapping is a no-op on both and the
      cited flat-vs-binned rationale describes a case that should not arise between
      two live configs. Gate 2(d) ("flat-vs-binned ⇒ pass") therefore pins a
      behavior the design does not specify a side/direction for.
    rationale: >
      Under-specified normalization direction means the new
      check_project_consistency.py could be implemented to normalize the wrong
      side (or only one side) and still pass its own author-written unit test while
      diverging from the intended semantics — an internal-contract ambiguity, not
      a correctness bug per se.
    suggested_fix: >
      State that the guard applies the OLD→NEW normalization to BOTH operands
      before deep compare (symmetric), which makes a flat-vs-binned pair converge
      while leaving two binned configs unchanged; note this differs from
      compare_copied_config's directional one-side application.
  - id: arch-6
    severity: minor
    section: "1a. C2 / 6. step 3 (baseline manifest repoint)"
    finding: >
      §6 step 3 says to repoint "the three wf3 TARGETS templates
      (check_baseline.py:112-114)". Two of the three (Qstats, basin) use the
      `{exp_dir}` template and follow the resolve() change, but the third (line
      114, the config snapshot) is templated on `{project_dir}/config/...`, not
      `{exp_dir}/`. Under the new tree the wf3 config snapshot moves to
      `experiments/<name>/config/`, so line 114 must change its template root from
      `{project_dir}/config/` to `{exp_dir}/config/` — a different edit than the
      resolve()/EXPERIMENT_NAME repoint the two results rows need.
    rationale: >
      If step 3 is read as "just repoint resolve()/EXPERIMENT_NAME," the config
      snapshot TARGET keeps pointing at the now-empty `{project_dir}/config/`
      location and `check --workflow climate_experiment` fails on a MISSING file at
      the milestone gate — a self-inflicted gate failure, not a real regression.
    suggested_fix: >
      Split the note: resolve()/EXPERIMENT_NAME + the two `{exp_dir}` results
      TARGETS move together; the config-snapshot TARGET (line 114) additionally
      changes its template from `{project_dir}/config/...` to
      `{exp_dir}/config/...`.
  - id: arch-7
    severity: minor
    section: "11. Consequences (Positive — pure hydrology_model)"
    finding: >
      §11 headlines "hydrology_model/ is a pure wf1 artifact again — unblocks
      P3-2." downscale_climate_forcing.py:43 opens WflowSbmModel(root=model_root,
      mode="r+") with model_root = basin_dir (hydrology_model) and calls
      mod.close() (line 118) to "commit any deferred writes." Moving the run dir
      out does not by itself prove the r+ open/close does not flush artifacts back
      into hydrology_model/. This is pre-existing behavior, not introduced here,
      so it is not blocking — but the claimed benefit depends on it.
    rationale: >
      If the r+ session writes anything back into model_root, the "pure wf1
      artifact" claim (and the P3-2 unblock premise) is overstated. The same
      execution smoke that gate 3 already mandates would settle it.
    suggested_fix: >
      Add a one-line check to gate 3: after the run-dir smoke, assert
      hydrology_model/ gained no wf3-written files (diff its contents pre/post
      wf3). If it did, either switch the open to mode="r" or document the residual
      write as a known non-purity.
  - id: arch-8
    severity: minor
    section: "1. Context (inventory) / 4 [UNVERIFIED]"
    finding: >
      The §4 [UNVERIFIED] note ("confirm no other consumer hardcodes
      climate_historical/raw_data/extract_historical.nc") has one confirmable hit
      the doc does not name: extract_historical_climate.py:210-217, the non-
      Snakemake `__main__` else-branch, hardcodes
      examples/my_project/climate_historical/raw_data/extract_historical.nc as its
      standalone fallback fn_out. It is dead under the Snakemake path (the script
      always runs via the snakemake global), so it is not a functional blocker.
    rationale: >
      Leaving it unnamed keeps an open [UNVERIFIED] that is trivially closable; a
      reviewer or implementer re-greps and finds it anyway. Naming it (and its
      dead-path status) discharges the uncertainty in the doc rather than deferring
      it to review.
    suggested_fix: >
      Note in §4 that the only other occurrence is the dead standalone fallback in
      extract_historical_climate.py __main__ (unreached under Snakemake), so no
      live consumer hardcodes the path; close the [UNVERIFIED].
  - id: arch-9
    severity: minor
    section: "1. Context (inventory, plots class) / 11 (seven classes)"
    finding: >
      The §1 table lists "plots" as a collision class but marks it "(no wf3 plot
      rule today; class reserved)" — confirmed: Snakefile_climate_experiment has
      no rule writing under plots/. The intake success criterion and §11 both
      claim "zero collisions across all seven classes," but one of the seven has
      no producer, so it is vacuously collision-free rather than demonstrably so.
    rationale: >
      Minor traceability wrinkle: gate 5 ("assert zero file collisions across all
      seven classes") cannot exercise the plots class because nothing writes it,
      so the "seven classes verified" claim is really "six verified + one
      reserved." Harmless but worth stating so the gate's coverage is not
      overclaimed.
    suggested_fix: >
      In §11 / gate 5, note the plots class is reserved (no wf3 producer yet), so
      collision verification covers six live classes and the plots root is
      pre-partitioned by construction for the future rule.


---

## Internal review — repo-fit lens (verbatim)

# Internal review — REPO-FIT & CONVENTIONS lens

```yaml
verdict: revise
doc_version: design-v1.md
findings:
  - id: repo-1
    severity: major
    section: "3. Drift guard mechanics / 10. Commit plan (commit 1)"
    finding: >
      The guard rule can only gate the DAG by being threaded into an existing
      rule's `input:` or into `rule all`, but commit 1 claims the guard is
      "additive; existing wf3 paths unchanged" and enumerates no threading edit.
      As written, commit 1 lands an orphan rule Snakemake never invokes — an
      inert guard.
    rationale: >
      `rule all` (Snakefile_climate_experiment:70-75) currently declares four
      inputs (Qstats.csv, basin.csv, the config snapshot, wf3_benchmarks.md). A
      Snakemake rule whose output is not consumed by `rule all` or by any rule
      reachable from it is simply never scheduled. So if commit 1 adds
      `check_project_consistency` + its `.project_consistency_ok` sentinel
      WITHOUT adding that sentinel to a downstream rule's `input:` (or to `rule
      all`), then `pytest tests/test_cli.py` (dry-run), the guard unit tests, and
      `--dry-run` all pass green while the guard never runs on any real wf3
      invocation — the exact "green gates, dead code" failure §7 warns about.
      Conversely, wiring the sentinel in IS an edit to an existing rule, which
      falsifies commit 1's "existing wf3 paths unchanged / additive" claim. Both
      readings are defects: either the guard is inert, or the commit-1 isolation
      claim is false and commit 1 is not independently the clean additive commit
      it says it is.
    suggested_fix: >
      Make the sentinel-threading explicit in the commit plan. Either (a) add
      `experiments/<name>/.project_consistency_ok` as an input to `rule all`
      (or to the first data-producing rule, e.g. copy_config / extract) in the
      SAME commit that adds the guard rule, and restate commit 1 as "adds guard
      rule + wires its sentinel into rule all" (not "additive, nothing existing
      changes"); or (b) fold the wiring into commit 2. Name the exact rule whose
      `input:` gains the sentinel, and state the DAG consequence (the guard now
      runs before any downstream rule fires).
  - id: repo-2
    severity: major
    section: "2a / 10. Commit plan (commit 2)"
    finding: >
      The `gather_benchmarks` rule (3.12) carries a params-string path
      `parts_dir = f"{project_dir}/benchmarks/_parts"` and an output
      `{project_dir}/benchmarks/wf3_benchmarks.md`. §2a moves wf3 benchmarks
      under `experiments/<name>/benchmarks/`, but neither §2a nor commit 2 calls
      out that this `parts_dir` PARAM must move too, and no gate smokes the
      benchmark gather.
    rationale: >
      Verified at Snakefile_climate_experiment:287-290: `parts_dir` is a
      `params:` string, and every per-rule `benchmark:` writes under
      `{project_dir}/benchmarks/_parts/3.*`. §7 itself names `params:` string
      paths as the canonical dry-run-blind surface. If commit 2 moves the per-rule
      `benchmark:` paths under `experiments/<name>/benchmarks/_parts/` but leaves
      `gather_benchmarks`'s `parts_dir` param pointing at the old
      `{project_dir}/benchmarks/_parts`, the gather silently finds no parts and
      emits an empty/stale `wf3_benchmarks.md`. That passes `--dry-run`, `pytest
      tests/test_cli.py`, and every named execution smoke (gates 3-5 exercise the
      run-dir move, dataset reuse, and collision — none run the gather). The wf3
      benchmark report is non-scientific, so this is not wrong results, but it is
      a silent-degradation failure mode the design's own §7 principle says must be
      caught and the gate ladder does not.
    suggested_fix: >
      In commit 2, list the `gather_benchmarks` `parts_dir` param and output as
      part of the "benchmarks move" edit surface (both the per-rule `_parts/`
      writes AND the gather's `parts_dir`/output must move together, or the
      gather stops finding parts). Add a cheap assertion to a gate that
      `wf3_benchmarks.md` is non-empty / has the expected rule rows after a run.
  - id: repo-3
    severity: minor
    section: "10. Commit plan / 13. Related"
    finding: >
      The design states "Prefix `p31:` per roadmap commit strategy" (§10) but the
      roadmap's commit-prefix convention registers only `m0*`, `r01:`..`r06:`,
      and `chore:` — there is no `p31:` prefix defined.
    rationale: >
      Verified at dev/roadmap.md:689-697: the `<prefix>` list is Phase-1 `m*`,
      Phase-2 `r01..r06`, and `chore:`. The milestone-matching PATTERN clearly
      implies `p31:` is the right prefix for a P3-1 commit, so the intent is
      unambiguous — but "per roadmap commit strategy" overstates existing sanction
      (the roadmap does not yet register a Phase-3 prefix). A reviewer following
      the citation literally finds no `p31:` there.
    suggested_fix: >
      Either add a commit to the plan (or a note) that registers the `p31:`
      prefix in dev/roadmap.md's commit-prefix list in the same milestone, or
      soften the wording to "following the roadmap's milestone-matching prefix
      pattern (`p31:`, to be registered)."
  - id: repo-4
    severity: minor
    section: "6 / 10 (commit 7) / 13. Related — MIGRATION.md"
    finding: >
      The design plans to "extend MIGRATION.md" with the P3-1 output-path map,
      but the existing root MIGRATION.md is a single-milestone R06 document, and
      naming.md §7 prescribes migration notes at
      `dev/<milestone>/migration_<topic>.md`, not appended to the root R06 file.
    rationale: >
      Verified: MIGRATION.md:1-14 is titled "MIGRATION — R06 structural refactor",
      scoped to milestone `r06-refactor` with a pre-R6 git ref (`e33ee45`) baked
      in. naming.md §7 (lines 141-150) says renaming `rule all` output filenames
      (which P3-1 does — Qstats/basin results and the config snapshot move) and
      test-fixture/dev-script paths "requires a `dev/<milestone>/migration_<topic>.md`
      note." Appending a P3-1 section to a root file titled and ref-anchored to R06
      conflicts with that convention and muddies the R06 doc's single-milestone
      provenance. Anchor 4b (a project intake decision) mandates a MIGRATION doc,
      which is satisfied either way — this is only about WHERE it lands.
    suggested_fix: >
      Land the P3-1 old->new output-path map at `dev/milestones/p31/migration_experiment_tree.md`
      per naming.md §7, and reference it from the roadmap/related section; if the
      root MIGRATION.md is deliberately being promoted to a multi-milestone index,
      say so explicitly and retitle it, rather than silently appending under an
      R06 header.
  - id: repo-5
    severity: minor
    section: "3. Drift guard mechanics — Normalization"
    finding: >
      §3 presents reuse of `_normalize_config_paths` + `COPIED_CONFIG_PATH_MAP`
      as load-bearing for the guard ("the guard normalizes catalog-path values
      before comparing"), but that map is a DIRECTIONAL pre-R6-flat -> post-R6-binned
      rewrite that is a no-op on two contemporaneous (both post-R6) configs.
    rationale: >
      Verified at semantic_tree_diff.py:238-254 and 67-87: the map only rewrites a
      value that exactly equals a documented OLD (pre-R6 flat) path; any other
      value is left untouched. Both the live experiment config and the wf1
      snapshot in the NEW tree are post-R6 binned paths, so the map matches
      nothing and normalization is a no-op — a plain section-scoped deep-equal
      already gives the correct pass/fail. The design's normalization rationale
      ("so a config on binned paths compares equal to one on flat paths") does not
      apply to the guard's compare (there is no old-vs-new axis between two
      same-era configs). No FALSE result arises — the deep-equal is correct
      regardless — so this is a rationale that misfits, not a bug.
    suggested_fix: >
      Either drop the normalization claim from the guard (a section-scoped deep
      structural equality on the four guarded sections suffices for two same-era
      full configs), or, if you keep it, state that normalization is defensive
      (handles the edge case of a hand-migrated flat-path experiment config
      compared against a binned snapshot) rather than the core mechanism.
  - id: repo-6
    severity: minor
    section: "6. Behavior-preservation — milestone gate (semantic_tree_diff)"
    finding: >
      The §6 milestone-diff plan does not note that `semantic_tree_diff.py`
      already EXCLUDES `logs`, `benchmarks`, and `.snakemake` from the tree walk,
      so the wf3 logs/benchmarks move (§2a) is invisible to the diff and needs no
      path-map entry — narrowing what the P3-1 path-map layer must cover.
    rationale: >
      Verified at semantic_tree_diff.py:91 (`EXCLUDED_DIR_NAMES = {"logs",
      "benchmarks", ".snakemake"}`) and 307-308/339. The §6 UNVERIFIED note and
      open question 1 correctly identify that the diff keys on identical relpaths
      (confirmed at 352-353, 356) and that a path-map layer is needed. But the
      design frames the path map as covering moved files broadly; in fact the
      moved logs/benchmarks are already skipped, so the path map only needs
      old->new entries for the CONTENT-bearing moves (results CSVs, config
      snapshot, run tomls, the keyed extraction nc). This is a scoping refinement
      that makes the path-map layer smaller and the §6 claim more precise.
    suggested_fix: >
      Note in §6 that logs/benchmarks are excluded by `EXCLUDED_DIR_NAMES`, so the
      P3-1 path map covers only the content-bearing relocations (results, config
      snapshot, run tomls, keyed extraction). This also confirms the wf3
      benchmark move cannot be validated by the tree diff — reinforcing repo-2's
      need for a separate gather smoke.
```

---

## Summary

Citations spot-checked and **verified**: `exp_dir`/`basin_dir` (Snakefile 61-62),
the run-dir paths (227, 245, 247-248, 260, 262-263, 287), `copy_config` out (86),
`extract_climate_grid` out (101) + consumer (165); `downscale_climate_forcing.py`
toml fields (`dir_output`=. line 69, `state.path_input` 70, `state.path_output` 71,
`input.path_static` 72, `input.path_forcing` 73, `output.csv.path` 74; the two
`os.path.relpath`/`.resolve()` writes 113/116); `extract_historical_climate.py`
`time_range` (91 chirps / 159 else) and the orography sidecar via
`os.path.dirname(fn_out)` (143); `check_baseline.py` `EXPERIMENT_NAME="experiment"`
(65), `resolve()` `exp_dir` template (124), the three wf3 TARGETS (112-114);
`semantic_tree_diff.py` `_normalize_config_paths`/`COPIED_CONFIG_PATH_MAP` (238-254,
67-87), `run_default` special-case (328), `diff_trees` identical-relpath keying
(344-361); C1's wf1 forcing path (Snakefile_model_creation:185). The intake anchors
are faithfully restated, and C1/C2 are accurate refinements the driver should see at
G1.

The doc is **self-contained** for review (I needed only the cited paths). The design
is technically sound on the toml-path mechanics (§5), the dataset-keying reuse
semantics (§4), and the dry-run-blindness stance (§7). The blocking issue is not in
the citations — it is in the **guard-wiring / commit-1 isolation** gap (repo-1) and
the **unaccounted benchmark-gather params path** (repo-2): both are silent-failure
modes that pass every green gate the design names. Neither reverses a decision; both
are fixable by making the edit surface explicit in the commit plan and adding one
gather smoke. `approve` is invalid with two major findings; verdict **revise**.


---

## External review round 1 — clean-room on design-v2 (verbatim)

## Verdict
verdict: revise
doc_version: design-v2.md

## Findings
### ext1-1  [blocking]
- section: 3. Drift guard mechanics
- finding: The persistent `.project_consistency_ok` sentinel does not reliably re-run the guard after either the live experiment config or a project snapshot changes. The live config and optional wf2 snapshot are only `params`, which do not participate in Snakemake timestamp freshness, while the mandatory wf1 snapshot is wrapped in `ancient()`. Once the sentinel exists, a later invocation can therefore reuse it without executing the comparison.
- rationale: A user can change a guarded project section after the first successful run and then execute wf3 with the existing sentinel; the workflow proceeds against a mismatched model or overlay even though fail-loud drift detection is the design’s core safety contract.
- suggested_fix: Define explicit invalidation semantics. Prefer making the guard an always-recomputed or disposable prerequisite, or keying its output to a deterministic digest of the guarded live sections and relevant snapshot sections. Add a test that runs the guard successfully, mutates each comparand in turn without deleting the sentinel, and verifies the next workflow invocation fails.

### ext1-2  [major]
- section: 4b. Reuse / staleness semantics
- finding: Threading each experiment’s newly created guard sentinel into the shared `extract_climate_grid` rule conflicts with the claimed dataset-store reuse. For experiment B, its sentinel is newer than the existing dataset+window extraction produced for experiment A, so ordinary Snakemake freshness evaluation can schedule extraction again even though the shared output key is unchanged.
- rationale: The designed “extracted once, referenced by N experiments” behavior can degrade into repeated extraction for every new experiment, adding substantial I/O and runtime and invalidating gate 4’s expectation that B skips `extract_climate_grid`.
- suggested_fix: Keep the guard as a required ordering prerequisite while excluding its timestamp from shared-cache freshness, for example by marking that sentinel input `ancient()` specifically on the extraction rule, or restructure the guarded dependency so experiment provenance does not make the project-level cache stale. Specify and test the exact DAG behavior for A then B.

### ext1-3  [major]
- section: 6a. The milestone gate — R3/R5-style value-identical re-record
- finding: The proposed path-aware TOML comparison resolves each pointer against its own TOML directory and then compares the resulting normalized paths directly. Because the reference and current trees occupy different project roots, equivalent targets such as `<ref-root>/hydrology_model/staticmaps.nc` and `<current-root>/hydrology_model/staticmaps.nc` remain unequal.
- rationale: The milestone comparison will report every correctly relocated run TOML as a failure, so the stated clean-diff gate cannot pass without an undocumented exemption or manual interpretation.
- suggested_fix: Compare resolved targets after translating both to a common semantic namespace, such as project-root-relative paths followed by the same old-to-new path mapping used for file pairs. Add positive and negative comparator tests: equivalent targets under different roots must pass, while a pointer to a different project-relative target must fail.

### ext1-4  [major]
- section: 2. Decision — the target tree
- finding: `experiment_name` is interpolated directly into filesystem paths without a validation or containment contract. The design does not reject separators, `..`, absolute-path forms, Windows-invalid characters, reserved device names, empty values, or names that normalize to the same directory.
- rationale: A malformed or adversarial configuration can write outside `experiments/`, collide with another experiment, or fail partway through the workflow after creating a partial tree, contradicting the isolation and zero-collision guarantees.
- suggested_fix: Specify centralized validation before DAG path construction: allow a conservative portable slug grammar, reject empty/reserved names and path components, and verify the resolved experiment directory remains a direct child of `<project_dir>/experiments`. Add validation tests for traversal, separators, Windows-invalid/reserved names, and normalization collisions.

---

## External review round 2 — regression on design-v3 (verbatim)

## Verdict
verdict: revise
doc_version: design-v3.md

## Findings
### ext2-1  [major]
- section: 3d. The guard-sentinel / shared-store interaction — verified DAG behavior
- finding: The accepted ext1-2 fix is not verified for the behavior it actually introduces. Its probe changes the content and mtime of one `ancient()` input, whereas experiments A and B give `extract_climate_grid` different input paths (`sentinel(A)` versus `sentinel(B)`). Snakemake’s persisted-job triggers can distinguish a changed set of input files independently of timestamp freshness, so `ancient()` does not establish that substituting B’s sentinel path leaves A’s keyed output up to date.
- rationale: If the input-identity change triggers the rule, experiment B re-extracts the shared dataset/window despite key reuse, defeating the stated extracted-once contract and invalidating gate 4’s expected A→B skip. The current empirical evidence therefore does not resolve ext1-2 on its merits.
- suggested_fix: Probe the exact A→B case with two distinct sentinel paths resolving to one shared output and record Snakemake’s reason. If it reruns, make the shared extraction depend on a stable project/key-level guard artifact rather than an experiment-specific sentinel, while retaining a separate per-experiment guard record for provenance.

### ext2-2  [major]
- section: 3b. Rule shape and the two snapshots
- finding: Computing the mandatory wf1 snapshot digest “at parse” conflicts with the specified rule-time missing-input behavior. Hashing a nonexistent snapshot during Snakefile evaluation raises before Snakemake can construct the DAG, report the intended `MissingInputException`, or run `--unlock`; the script’s promised friendly diagnostic is likewise unreachable.
- rationale: A fresh or partially recovered project without the wf1 snapshot fails during parsing instead of through the designed guard rule, contradicting §3’s operational rationale and breaking the documented lock-recovery path in exactly the missing-project-state case the design claims to handle.
- suggested_fix: Derive the digest without eagerly opening the file during Snakefile parsing—for example, use a checkpoint or digest sidecar produced from the mandatory snapshot, or otherwise structure the rule so absence remains a normal missing input while snapshot content changes still invalidate the persistent sentinel. Add a missing-snapshot `--dry-run` and `--unlock` regression test.

### ext2-3  [minor]
- section: 2b. `experiment_name` validation — centralized, before any path construction
- finding: The validation test matrix says `experiment_A` passes unchanged, but the fixed grammar `^[a-z0-9][a-z0-9_]*$` rejects uppercase `A`.
- rationale: Implementing both requirements literally makes the mandatory validation test fail and leaves the intended case-normalization policy ambiguous.
- suggested_fix: Change the passing fixture to `experiment_a`, or explicitly broaden the grammar if uppercase names are intended.

---

## Final finding ledger (29 rows)

# P3-1 — design-review-loop ledger

Append-only. One row per original finding ID from the stage-2 internal lens panel
(risk-1..7, arch-1..9, repo-1..6 = 22). Round = `internal`. Doc version =
`design-v2.md` (the revision that resolves them).

| ID | Round | Severity | Disposition | Resolution or rationale | Doc version |
| --- | --- | --- | --- | --- | --- |
| risk-1 | internal | blocking | accepted | §4a: `prepare_climate_data_catalog.py:76-84` named as an edit site (chirps orography lookup hardcodes old `../../climate_historical/raw_data/` — breaks on both moved target AND changed depth). Rewritten to derive from the passed dataset+window key param at correct depth. Non-era5 chirps-path gate 8 (+ mandatory unit-test minimum) added to close the era5-only escape class. Verified against the tree. | design-v2.md |
| risk-2 | internal | major | accepted | §3b: wf1 snapshot = hard `input:` (ancient, mandatory); wf2 snapshot = params path, existence-checked in-script (NOT a rule input, so the optional projections overlay is not force-required). `workflows.climate_projections` compared against the wf2 snapshot when present, unchecked+logged when absent. Each guarded section now compared against the snapshot of the workflow that owns it. `climate_projections` kept guarded (anchor 3 names it; dropping would reopen an anchor). | design-v2.md |
| risk-3 | internal | major | accepted | §4b: "never stale" corrected to "never stale for a fixed catalog." Catalog source-data immutability stated as an explicit assumption (fits CST rapid-assessment scope); manual key-dir deletion documented as the invalidation escape hatch; content-hash staleness deferred to reproducible-computing (§12 row 3 + dev/followups.md). | design-v2.md |
| risk-4 | internal | major | accepted | §6a step 3: milestone gate now asserts MISSING+EXTRA EMPTY modulo an explicit, justified allowlist (nonempty unexplained set = gate FAILURE). Allowlist lives beside the path map in `dev/milestones/p31/migration_experiment-structure.md`; entries require written justification and milestone review; scoped to genuine presence exemptions (e.g. the `.project_consistency_ok` sentinel EXTRA-by-design). Path map spec'd total over the wf3 content-bearing output set. NB: the toml pointer diffs land in the separate `failures` list (file content), handled by the risk-7 path-aware comparator, not this allowlist. | design-v2.md |
| risk-5 | internal | minor | accepted | §4c: day-resolution invariant stated ("window endpoints are day-resolution; time-of-day dropped by contract"); `slugify_window()` asserts `HH:MM:SS==00:00:00` and fails loud, so a future sub-daily window errors instead of colliding onto a shared key. (Merged with arch-4 as Cluster B.) | design-v2.md |
| risk-6 | internal | minor | accepted | §1a C1: `setup_time_horizon.py:51` named as the concrete wf1 consumer that pins `wflow_data/inmaps_historical.nc` in place (hardcodes `../climate_historical/wflow_data/inmaps_historical.nc`). Unaffected because C1 keeps `wflow_data/` put; recorded so the load-bearing constraint is visible. Closes the open grep with a positive result. | design-v2.md |
| risk-7 | internal | minor | accepted | §5/§5a: verified against the vendored hydromt_wflow source — `config.write` → `make_config_paths_relative`/`_relpath` re-relativizes absolute same-mount paths against the toml dir. Absolute `path_static` in ⇒ correct relative on disk (depth-robust). `[UNVERIFIED]` removed; gate-3 smoke demoted to confirmation. v1's "stored verbatim"/"non-portable" rationale corrected. Downstream consequence (toml pointer diffs) handled in §6a by a path-aware toml comparator in `semantic_tree_diff.py` — resolves each path field against its own toml dir, passes same-target pointer moves, catches a mis-repoint. These diffs land in the `failures` list (file content), NOT MISSING/EXTRA, so a bare allowlist could not exempt them; the comparator fix (~10 lines, commit 5) makes the §6 clean-diff thesis true rather than asserted. | design-v2.md |
| arch-1 | internal | major | accepted | §3a: the guard sentinel is now an explicit `input:` of all five wf3 root rules (copy_config, extract_climate_grid, climate_stress_parameters, prepare_weagen_config, prepare_weagen_config_st — enumerated with line numbers). The self-contradictory "additive/unchanged" claim is dropped; the five per-rule edits are in the §2 inventory and §10 commit 1. (Cluster A, same defect as repo-1.) | design-v2.md |
| arch-2 | internal | minor | accepted | §1/§2: implementation mechanism stated as `exp_dir` redefinition (line 62), so the params-level realization paths (`output_path` 132/151, `exp_dir` param 265) move with it automatically; called out because they are dry-run-blind `params:` strings. | design-v2.md |
| arch-3 | internal | minor | accepted | §6a step 2: path-map scope narrowed — `semantic_tree_diff.py` excludes logs/benchmarks/.snakemake (`EXCLUDED_DIR_NAMES`), so the wf3 logs/benchmarks move needs no path-map entry; the map covers only content-bearing relocations (results CSVs, config snapshot, run tomls/CSVs, keyed extraction nc). (Cluster C, with repo-6.) | design-v2.md |
| arch-4 | internal | minor | accepted | §4c: same sub-day-window collision as risk-5; resolved by the day-resolution invariant + fail-loud assertion in `slugify_window()`. (Cluster B.) | design-v2.md |
| arch-5 | internal | minor | accepted | §3b (normalization): `COPIED_CONFIG_PATH_MAP` is directional and a no-op between two post-R6 configs, so the guard's core mechanism is a plain section-scoped deep-equal; if normalization is kept it is applied SYMMETRICALLY to both operands (explicitly unlike compare_copied_config's directional one-side application). Gate 2(d) pins this. (Cluster A2, with repo-5.) | design-v2.md |
| arch-6 | internal | minor | accepted | §1a C2 / §6a step 4: the config-snapshot TARGET (`check_baseline.py:114`) is `{project_dir}/config/`-rooted and needs a DIFFERENT repoint (template root → `{exp_dir}/config/`) than the two `{exp_dir}` results TARGETS + `resolve()`/`EXPERIMENT_NAME`. Both edits split out explicitly. | design-v2.md |
| arch-7 | internal | minor | accepted | §7 gate 3 / §11: "pure hydrology_model" now asserted, not argued — gate 3 diffs `hydrology_model/` pre/post the wf3 smoke to prove the `r+` open/close (downscale line 43/118) flushes nothing back; remedy (switch to mode="r" or document) if it does. | design-v2.md |
| arch-8 | internal | minor | accepted | §4d: the §4 `[UNVERIFIED]` closed — the only other occurrence of the old path is the dead `__main__` fallback `extract_historical_climate.py:210-217`, unreached under Snakemake. Named; marker removed. | design-v2.md |
| arch-9 | internal | minor | accepted | §1/§7 gate 5/§11: the plots class has no wf3 producer today, so collision verification covers six live classes + one reserved (pre-partitioned by construction), not "seven verified." Overclaim corrected. | design-v2.md |
| repo-1 | internal | major | accepted | §3a/§10 commit 1: same guard-wiring defect as arch-1 — an orphan sentinel is never scheduled; wiring it into a downstream `input:` falsifies commit-1's "additive" claim. Resolved by the five-root sentinel `input:` wiring, named rules, dropped "additive" claim, and the DAG consequence stated. (Cluster A.) | design-v2.md |
| repo-2 | internal | major | accepted | §1/§2a/§10 commit 2: `gather_benchmarks` `params.parts_dir` (line 289) + output (287) added to the change inventory as part of the benchmarks move (per-rule `_parts/` writes AND the gather's `parts_dir` must move together or the gather emits an empty report). Gate 7 (gather smoke) added — the tree diff cannot cover it (benchmarks are diff-excluded). | design-v2.md |
| repo-3 | internal | minor | accepted | §10: `p31:` prefix is NOT yet registered in the roadmap's commit-prefix list (`m*`, `r01..r06`, `chore:` at :689-697); commit 7 registers it, and the wording is softened to "the roadmap's milestone-matching prefix pattern (`p31:`, registered in commit 7)." | design-v2.md |
| repo-4 | internal | minor | accepted | §6a/§10 commit 7/§13: the migration note moves to `dev/milestones/p31/migration_experiment-structure.md` per naming.md §7 (NOT appended to the root R06-scoped, git-ref-anchored `MIGRATION.md`). Convention honored, not fought. | design-v2.md |
| repo-5 | internal | minor | accepted | §3b (normalization): same as arch-5 — directional map is a no-op between two same-era configs; normalization reframed as a defensive layer over a section-scoped deep-equal core, applied symmetrically if kept. (Cluster A2.) | design-v2.md |
| repo-6 | internal | minor | accepted | §6a step 2: same scope refinement as arch-3 — logs/benchmarks excluded by `EXCLUDED_DIR_NAMES`, so the path map covers only content-bearing moves; reinforces repo-2's need for a separate gather smoke. (Cluster C.) | design-v2.md |
| ext1-1 | external-r1 | blocking | accepted | §3c: sentinel-invalidation semantics rebuilt on an EMPIRICALLY VERIFIED pinned-Snakemake-9.6.2 rerun-trigger matrix (throwaway dry-run/scratch probe this session; verbatim reasons in §3c). Verified: (a) params-only change RE-TRIGGERS ("Params have changed since last execution: before 'live_A' now 'live_B'"); (b) ancient() input CONTENT change does NOT re-trigger ("Nothing to be done"); (c) both → re-triggers (params); (fix) a params-carried SHA-256 digest of the snapshot RE-TRIGGERS on content change ("Params have changed: before '74e1d2f7924c' now '0b6a6f768348'") AND is content-addressed not mtime-addressed (reverting to byte-identical content → "Nothing to be done"). RESOLUTION: a SHA-256 digest of the guarded live-config sections (string) threaded as a guard `param` (covers live-config drift, case a) + wf1 snapshot kept `ancient()` input for the mandatory missing-fail/ordering BUT its content-change tracked via a SHA-256 digest `param` (covers case b) + wf2 snapshot digest `param` when present. Every comparand change now re-triggers via a params trigger; none relies on ancient() mtime. Reviewer's mutate-each-comparand test adopted as gate 2b (a named integration/`--dry-run` check, distinct from the gate-2 comparator unit tests; i–l). **Reviewer factually wrong for this repo:** the premise "params do not participate in freshness" is FALSE under pinned Snakemake 9.6.2 (default params rerun-trigger; no --rerun-triggers override; R5 independently verified the same on extract_climate_grid — dev/followups.md §R3). Reviewer WAS right on the ancient() snapshot side (case b), which the digest param closes. No `[UNVERIFIED]` on the matrix. | design-v3.md |
| ext1-2 | external-r1 | major | accepted | §3d/§3a/§4b: resolved coherently with ext1-1 (shared `ancient()` lever). Two changes: (1) the guard sentinel is made PER-EXPERIMENT (`experiments/<name>/.project_consistency_ok`), so A and B have distinct sentinel files; (2) the shared `extract_climate_grid` consumes the sentinel `ancient()` (ordering-only, timestamp ignored) while the four non-shared per-experiment roots take it fresh. Verified by the same probe (case b: ancient() content/mtime change → "Nothing to be done"), so B's newer sentinel cannot re-trigger A's cached keyed extraction. Exact A-then-B-then-C DAG behavior specified (A runs; B SKIPS extraction, reuses key K; C re-extracts to new key K') and pinned by gate 4. Ordering/gating preserved (a failing guard never writes the sentinel → extraction blocked); false re-extraction eliminated. | design-v3.md |
| ext1-3 | external-r1 | major | accepted | §6a step 3: reviewer correct — v2's "resolve each pointer against its own toml dir then compare directly" is broken across roots (ref tree and current tree live under DIFFERENT project roots, so `<ref-root>/hydrology_model/staticmaps.nc` ≠ `<cur-root>/hydrology_model/staticmaps.nc` → every relocated run toml would be a `failures` entry, gate could never pass). Fixed comparator (in our `semantic_tree_diff.py`): (1) resolve field lexically against its own toml dir, (2) translate to PROJECT-ROOT-RELATIVE by stripping the side's project root (cancels the different absolute roots; ref_root/cur_root are known inputs), (3) apply the same old→new path map used for file-pair matching, (4) compare mapped project-root-relative targets — equal ⇒ PASS, different ⇒ real `failures` (mis-repoint CAUGHT). **Internal-consistency fix (advisor-caught this revision):** the path map is specified as a DIRECTORY-PREFIX rewrite (`climate_<exp>/ → experiments/<name>/`, etc.), NOT a per-file table (§6a step 2). Only `path_forcing` actually needs the map (its realization dir moved with `exp_dir`); `path_static`/`state.path_input` resolve to unmoved `hydrology_model/` targets and pass with no entry. The forcing inmaps nc is `temp()` (in neither tree), so a per-file map would have no entry for it → untranslated → a `failures` entry on every run toml, re-reddening the gate; the prefix form covers temp targets and closes that gap. Positive (equivalent targets under different roots + moved-but-equivalent forcing temp target via the prefix rule) + negative (mis-repoint to different project-relative target) comparator tests adopted into commit-5. Zero toml allowlist entries; risk-4 masking not re-introduced. | design-v3.md |
| ext1-4 | external-r1 | major | accepted | §2b: centralized `validate_experiment_name()` in `blueearth_cst/shared/snake_utils.py`, called at Snakefile parse immediately after reading `experiment_name` and BEFORE `exp_dir` construction. Conservative slug grammar aligned to naming.md: `^[a-z0-9][a-z0-9_]*$`, nonempty, ≤64 chars (excludes hyphens/dots so the value can never inject a path component/extension). Rejects (clear ValueError naming the input): empty/whitespace; `/ \ : * ? " < > |` + NUL; `.`/`..`/any separator (traversal); absolute forms (leading `/`,`\`, `X:`); Windows-reserved device names (CON/PRN/AUX/NUL/COM1-9/LPT1-9, case- and extension-insensitive); trailing dot/space. Plus a CONTAINMENT assertion: resolved `Path(project_dir,"experiments",name).resolve().parent == Path(project_dir,"experiments").resolve()` (direct child). Parse-time is CORRECT here (unlike the state-dependent guard, §3): it is config validation — a malformed name makes the whole DAG ill-defined, so failing under `--dry-run` is right; `--unlock` needs only a parseable Snakefile (unaffected in the valid case; the invalid case has no recoverable valid workdir). Test matrix (traversal/separators/absolute/reserved/empty/hyphen/dot/length/trailing-space/containment) in `test_validate_experiment_name.py` (gate 9) + a gate-1 `--dry-run` parse-fail on `experiment_name: "../evil"`. Grammar fixed (safety, not cosmetic — §9 row). | design-v3.md |
| ext2-1 | external-r2 | major | accepted | User arbitration 2026-07-23: accepted, fix required (external cap stands). Settled empirically — a new probe (pinned Snakemake 9.6.2, dry-runs + scratch runs) CONFIRMED the reviewer: swapping the per-experiment sentinel input path A→B (same content, keyed output present, `ancient()`) RE-RUNS the shared rule — verbatim reason "set of input files has changed since last execution: extract" ("triggered by provenance information"); `ancient()` suppresses only the mtime trigger, not input-set identity. v3's §3d mechanism REPLACED with the reviewer's shape: the guard rule gains a second, experiment-invariant output — the KEY-LEVEL guard artifact `climate_historical/<key>/.guard_ok` — consumed `ancient()` by `extract_climate_grid` ONLY; the per-experiment sentinel `experiments/<name>/.project_consistency_ok` remains for the four per-experiment roots (fresh) + provenance. Fix-shape probes: with the stable artifact path, B's dry-run does NOT schedule extraction even while B's guard IS scheduled and rewrites the artifact (job stats: guard 1 + per_exp_root 1, total 2 — extract absent; per_exp_root re-triggered by "input files updated by another job", proving the fresh path still couples while the ancient stable path does not), and after B's real run a follow-up dry-run reports "Nothing to be done" (rewrite/mtime does not retro-trigger). Forced consequences specified: guard params must be experiment-invariant across passing configs → `config_path` REMOVED from params (script reads `snakemake.config`); new-key extraction still ordered after/blocked by the current experiment's guard (K′/.guard_ok edge); failure-cleanup caveat (failed guard may remove an existing K/.guard_ok; auto-recovered) recorded. §3a consumption table (which rules consume which artifact), §3b outputs/params, §3d A→B→C behavior, §4b, gate 4 (input-set-invariance assertion + alternation smoke), §9/§10/§11 re-specified. | design-v4.md |
| ext2-2 | external-r2 | major | accepted | User arbitration 2026-07-23: accepted, fix required. Reviewer correct: v3 computed the mandatory wf1 snapshot digest "at parse", so a fresh/partially-recovered project (no wf1 snapshot) would raise during Snakefile evaluation — before DAG build, the designed rule-level `MissingInputException`, or `--unlock` — breaking the documented lock-recovery path in exactly the missing-project-state case. Fixed with the lightest mechanism consistent with v3: `file_digest_or_absent(path) -> str` in `blueearth_cst/shared/snake_utils.py` (peer of `get_config`/`validate_experiment_name`) — returns the SHA-256 hex digest when the file exists, and the literal sentinel string "ABSENT" (uppercase, non-hex, wrong length — cannot collide with a real digest) when missing; it NEVER raises. Used at parse for BOTH wf1 and wf2 snapshot digests (replaces v3's wf2 "empty-sentinel string"). Absence still fails at the guard rule via the `ancient()` input declaration exactly as designed; content changes still flip the digest param (§3c verified triggering unchanged); the ABSENT→present transition also flips it. Reviewer's regression checks adopted as gate 2c: missing-snapshot `--dry-run` parses + reports the rule-level MissingInputException (no parse traceback), `--unlock` succeeds, and the present-branch content trigger is asserted. §3b/§3c/§7/§10-commit-1/§11 updated. | design-v4.md |
| ext2-3 | external-r2 | minor | accepted | User arbitration 2026-07-23: accepted, fix required. The contradiction is real: the §2b test matrix listed `experiment_A` as passing while the fixed grammar `^[a-z0-9][a-z0-9_]*$` rejects uppercase. Fixture corrected to `experiment_a` ⇒ pass; the grammar STANDS (aligned to `dev/conventions/naming.md`, fixed-not-cosmetic per ext1-4/§9). Case policy made explicit in §2b: uppercase is REJECTED with a `ValueError` naming the offending value and the grammar — the validator NEVER silently lowercases; `experiment_A` ⇒ ValueError added to the matrix as an explicit rejection case (exercised via gate 9). | design-v4.md |


---

## Immutable external-review brief (as instantiated)

# External review brief — P3-1 Project/experiment structure, round <n — filled at dispatch>

## Role

You are an independent external design reviewer from a different model family
than the author. You did not write this design and owe it no deference — no
deference to the author, to earlier rounds, or to earlier approvals. Your
value is adversarial pressure: challenge framing, feasibility, and
completeness. Do not copyedit prose.

## Task

Review exactly one document:

- `<absolute path to the latest design-vN.md — filled at dispatch>`

Orientation (neutral): the design reorganizes the output tree of a
Snakemake-orchestrated climate-stress-testing workflow toolbox so one project
directory (a river basin) can hold multiple stress-test experiments —
introducing an `experiments/<name>/` subtree, a shared per-dataset
historical-climate store, a config drift-guard rule, a Wflow run-directory
relocation, and a repoint of the repo's baseline-verification tooling. It is
a decision-record for a behavior-preserving layout milestone: values must
stay identical while paths move.

<round 1: clean-room — the ledger/index block below is OMITTED>
<round ≥ 2 adds, after you form your own view:>
- `<path to ledger.md>` — dispositions of every prior finding
- `<path to internal-review-index.md>` — the internal panel's findings

**Regression duty (round ≥ 2 only):** verify that findings marked resolved
are actually resolved in this version, that no accepted fix introduced a new
defect, and that rejections' rationales hold. Re-raise anything that fails —
your earlier findings may be withdrawn only by you, here.

## Authority boundary

Read-only. Read the files listed above; you may skim files the design
directly cites if needed for context, but do not read broadly through the
repository and do not modify anything.

## Review lenses (in priority order)

1. **Operational feasibility** — would this design work as specified?
   Ambiguous contracts, unimplementable steps, missing inputs, undefined
   behavior.
2. **Failure modes missed** — realistic ways the designed system degrades
   that the design does not cover.
3. **Incentive and process design** — where the design includes loops, gates,
   or criteria: are they gameable, self-defeating, or consensus theater?
4. **Over-engineering** — components whose cost exceeds their value in this
   repo's context; simplifications that lose little.
5. **Gaps** — anything a design of this genre should cover and doesn't.

## Evidence burden

Every `blocking` or `major` finding must state an observable consequence —
what fails, degrades, or costs — not a preference. Cite the design section it
targets. A verdict of `approve` may not coexist with any `blocking` or
`major` finding.

## Output contract (mandatory)

Return ONLY a markdown document with this structure — no preamble:

    ## Verdict
    verdict: approve | revise | reject
    doc_version: <the design-vN.md you reviewed>

    ## Findings
    ### ext<n>-<seq>  [blocking | major | minor]
    - section: <design heading the finding targets>
    - finding: <one-paragraph claim>
    - rationale: <why it matters — observable consequence>
    - suggested_fix: <concrete change, or "none">

Severity calibration: `blocking` = the design as specified would fail,
produce wrong results, or cannot be implemented; `major` = meaningful
degradation, cost, or risk with a clear fix; `minor` = worth noting, author's
discretion. List findings in severity order, blocking first. Aim for the
findings that matter; do not pad. If the design is sound, say so — an empty
findings list with `verdict: approve` is a valid review.

