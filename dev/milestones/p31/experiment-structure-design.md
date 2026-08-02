# P3-1 — Project/experiment structure — design

**Status. ACCEPTED 2026-07-23** via the `p31-experiment-structure`
design-review-loop (G1 framing + G2 approval both granted; converged via
arbitration after the 2-round external cap). Full audit trail — verdict table,
aggregated internal index, both external rounds verbatim, and the 29-row
finding ledger — is in `experiment-structure-design-review-record.md` beside
this file. Implementation is a separate `task-brief` handoff; this is the
accepted plan, not yet built. Provenance of this (v4) content: revised from
`design-v3.md` after external review round 2 (verdict *revise*; ext2-1/-2
major, ext2-3 minor) under the external round cap — the user ruled 2026-07-23
that **ext2-1, ext2-2, ext2-3 are accepted, fix required** and the cap stands;
v4 is confined to exactly those three findings plus the cross-references they
force. (Landed by the run driver with this status header as the only edit —
G2 approved v4 with zero editorial change requests.)

**Scope authority.** `dev/milestones/p31/experiment-structure-intake.md` (the
AUTHORITATIVE scoping record, landed beside this doc from the run dir — "Confirmed scoping decisions" §1–4 and both
2026-07-23 user amendments are fixed anchors, not reopened here), the G1 gate
record (2026-07-23: approach A as elaborated in v1; both scope amendments accepted
— wf3-only store re-key, and baseline-manifest + dev-script repoint incl.
`check_baseline.py` edits and a `semantic_tree_diff` path-map layer), and
`dev/roadmap.md` § "P3-1 — Project/experiment structure". This doc is
self-contained: a reviewer needs only this file plus the cited paths.

**Genre.** Decision-record (a layout/contract change that moves the experiment
dimension from half-built to fully self-describing), mirroring the R6 house
pattern under `dev/r0#/` (Context · Decision · What changes · Behavior-preservation ·
Verification · Commit plan · Alternatives · Consequences · Related).

**Revisions:**
- 2026-07-23: initial draft v1.
- 2026-07-23: v2 — resolves the internal panel (22 findings). Edits marked inline
  as **[v2: <finding-id>]** at the point of change.
- 2026-07-23: v3 — resolves external review round 1 (ext1-1 blocking, ext1-2/-3/-4
  major). Edits marked inline as **[v3: ext1-N]**; prior **[v2:]** markers retained.
  The disposition of every finding is in the run ledger (`ledger.md`).
- 2026-07-23: v4 — ARBITRATION revision under the external round cap (user rulings
  2026-07-23: ext2-1/-2/-3 **accepted, fix required**; cap stands). Edits marked
  inline as **[v4: ext2-N]**, confined to the three arbitrated findings plus the
  cross-references those edits force; prior markers retained. Dispositions in
  `ledger.md`.

**v4 change digest (what a re-reviewer should re-read first):**
- **§3d + §3a/§3b (ext2-1):** the r2 reviewer is **CONFIRMED by a new probe** —
  swapping the per-experiment sentinel *path* (A→B, same content, keyed output
  present, `ancient()`) re-triggers the shared rule via Snakemake's **input-set
  provenance trigger** (verbatim: *"set of input files has changed since last
  execution"*), which `ancient()` does not suppress (it is mtime-only). v3's §3d
  mechanism is REPLACED by the reviewer's shape: the guard rule gains a second,
  experiment-invariant output — a **key-level guard artifact**
  `climate_historical/<key>/.guard_ok` — and **that** is what
  `extract_climate_grid` consumes (`ancient()`); the per-experiment sentinel
  remains for the four per-experiment roots (fresh) and for provenance.
  Probe-verified: with the stable-path artifact, B's dry-run does **not** schedule
  extraction even while B's guard IS scheduled and rewrites the artifact, and a
  post-B rewrite does not retro-trigger. Forced consequence: the guard's `params`
  must be experiment-invariant across passing configs, so `config_path` is
  **removed** from the guard params. §3a consumption table, §3d A→B→C behavior,
  §4b, and gate 4 (incl. a new alternation smoke) re-specified.
- **§3b + §7 gate 2c (ext2-2):** the wf1/wf2 snapshot digests are computed via an
  absence-tolerant helper `file_digest_or_absent()` in
  `blueearth_cst/shared/snake_utils.py` that returns the literal sentinel string
  `"ABSENT"` for a missing file instead of raising — so a fresh project (no wf1
  snapshot) parses, `--dry-run`s, and `--unlock`s cleanly; absence still surfaces
  at the guard rule via the `ancient()` input declaration
  (`MissingInputException`), and snapshot content changes still flip the digest
  param. The reviewer's missing-snapshot `--dry-run`/`--unlock` regression checks
  are adopted as gate 2c.
- **§2b (ext2-3):** the validation test-matrix fixture `experiment_A` contradicted
  the fixed grammar `^[a-z0-9][a-z0-9_]*$`; the fixture is corrected to
  `experiment_a` (the grammar stands — it matches `dev/conventions/naming.md`),
  and the case policy is made explicit: uppercase is **rejected with a message**
  naming the grammar — no silent lowercasing.

**v3 change digest (retained; the ext1-2 bullet is SUPERSEDED by v4 — see the v4
digest and §3d):**
- **§3c (ext1-1, BLOCKING):** the sentinel-invalidation semantics are now defined
  on an **empirically verified** pinned-Snakemake-9.6.2 rerun-trigger matrix (probe
  results verbatim in §3c). Confirmed: (a) a **params-only** change DOES re-trigger
  the guard ("Params have changed since last execution") — so live-config drift is
  covered by threading the guarded live sections through as `params`; (b) an
  `ancient()` **input content** change does **NOT** re-trigger — so the wf1 snapshot
  wrapped in `ancient()` (as v2 specified) would let a snapshot-only change slip. The
  fix: carry a **content digest of the wf1 snapshot (and the wf2 snapshot when
  present)** as a guard `param`, verified to re-trigger on snapshot content change
  and — critically — to be **content-addressed, not mtime-addressed** (reverting the
  snapshot to byte-identical content restores "Nothing to be done"). The reviewer's
  mutate-each-comparand test is adopted as gate 2b (i–l), a named integration/
  `--dry-run` check distinct from the gate-2 comparator unit tests.
- **§4b/§3d (ext1-2):** the guard sentinel is made **per-experiment**
  (`experiments/<name>/.project_consistency_ok`) and the shared `extract_climate_grid`
  rule consumes it with `ancient()` (ordering-only, timestamp ignored) — verified that
  `ancient()` on a content-changed input yields "Nothing to be done", so B's fresh
  sentinel cannot re-trigger A's cached extraction. The exact A-then-B DAG behavior is
  specified and pinned by gate 4.
- **§6a step 3 (ext1-3):** the path-aware toml comparator now translates **both**
  sides to a **project-root-relative** namespace and applies the old→new path map
  before comparing, so equivalent targets under different reference/current roots
  PASS while a mis-repoint FAILS. Positive + negative comparator tests adopted into
  commit-5's test set.
- **§2b (ext1-4):** `experiment_name` gets **centralized slug validation** in a
  shared `snake_utils` helper called at Snakefile parse — conservative grammar
  aligned to `dev/conventions/naming.md`, rejecting separators/`..`/absolute/Windows-
  reserved/empty, and asserting the resolved dir is a direct child of `experiments/`.
  A fail on invalid config at parse is correct even under `--dry-run`; it does not
  break `--unlock` (verified reasoning in §2b). Validation test matrix added.

**v2 change digest (retained — a re-reviewer should still confirm these hold under v3):**
- **§4 (risk-1, blocking):** the CHIRPS orography sidecar has a second, tested
  consumer — `prepare_climate_data_catalog.py:76-84` — hardcoding the old
  `../../climate_historical/raw_data/<src>_orography.nc` path at the old depth. Named
  edit site, with a chirps-path execution/unit gate.
- **§3 (Cluster A: arch-1/repo-1/risk-2):** the guard sentinel is an explicit
  `input:` of the wf3 root rules; the false "additive / existing paths unchanged"
  commit-1 claim is dropped; the wf2 snapshot is a params-checked (not rule-input)
  comparand so the optional projections overlay is not force-required.
- **§5/§5a/§6 (risk-7, RESOLVED):** verified against the vendored hydromt_wflow
  `config.write` — it re-relativizes absolute same-mount paths. Absolute
  `path_static` in ⇒ correct relative on disk. The run-toml path-pointers move
  string value, handled by a path-aware toml comparator (now revised by ext1-3).
- **§4 (risk-3):** "never stale" corrected to "never stale for a fixed catalog";
  catalog immutability an explicit assumption + manual-invalidation escape hatch.
- **§6 (risk-4):** the milestone gate asserts MISSING+EXTRA empty modulo an
  explicit, versioned allowlist; location and review rule named.

---

## 0. Anchors restated (fixed; not reopened)

For a reviewer who has not read the intake, the confirmed decisions this design
builds *under* (from intake § "Confirmed scoping decisions" + amendments):

1. **Experiment scope.** An experiment varies all wf3 stress-test settings
   (perturbation grid, `realizations_num`, `horizontime_climate`, `run_length`,
   `run_historical`, `Tlow`/`Tpeak`, `aggregate_rlz`) **plus** the climate
   window/source (`shared.historical_window`, `shared.clim_historical`).
   Project-level and shared by all experiments: the built model (wf1), the
   projections overlay (wf2), `shared.basin`, catalogs.
2. **Tracking level.** Self-describing tree + provenance only. Each experiment
   owns its config snapshot, logs, benchmarks, plots. **CUT:** machine-readable
   registry, CLI listing/status, cross-experiment comparison outputs.
3. **Approach A.** One full config file per experiment + a wf3 startup **drift
   guard**. Config mechanics unchanged (single `--configfile`,
   `workflow.configfiles[0]` forwarding, `get_config` contract intact);
   `experiment_name` selects the subtree. The guard compares the experiment
   config's `project`, `shared.basin`, `workflows.model_creation`,
   `workflows.climate_projections` sections against the project snapshot.
   Layered project+experiment configs are **rejected** (breaks the
   `configfiles[0]` contract to R).
4. **Constraints.** (a) Baseline handled as a documented, value-identical
   re-record at the milestone gate (paths move, values must not) — R3/R5 style,
   not R6's run-relative-only gate. (b) The current `project_dir` layout is **not**
   an external contract on this fork; a migration note documents old→new for any
   future upstreaming.
5. **Amendments.** `climate_historical/` is a **project-level per-dataset store**
   referenced (not copied) by experiments. `realization_*/` and `stress_test/`
   internal file shapes are kept **structurally as-is** in P3-1.

---

## 1. Context

`project_dir` is one basin project. The experiment dimension is half-built:
`workflows.climate_experiment.experiment_name` namespaces two things and nothing
else. Grep-verified in `Snakefile_climate_experiment`:

- `exp_dir = f"{project_dir}/climate_{experiment}"` (line 62) — the wf3
  realization/results subtree (`stress_test/`, `realization_*/`, `model_results/`,
  `weathergen_config.yml`, `data_catalog_climate_experiment.yml`).
- `f"{basin_dir}/run_climate_{experiment}/"` — the Wflow run tomls + outputs,
  written *inside* the wf1 model artifact directory (`basin_dir = project_dir/
  hydrology_model`, line 61; run-dir paths at lines 227, 245, 247, 248, 260).

Everything else in wf3 is **project-global and collides across experiments**.
Inventory (all in `Snakefile_climate_experiment`, `project_dir`-rooted, no
`experiment` in the path):

| Class | Path (as written) | Rule / line |
| --- | --- | --- |
| config snapshot | `{project_dir}/config/snake_config_climate_experiment.yml` | `copy_config` out, 86 |
| logs | `{project_dir}/logs/3.*.log` | every rule's `log:` |
| benchmarks (parts) | `{project_dir}/benchmarks/_parts/3.*` | every rule's `benchmark:` |
| benchmarks (gather) | `{project_dir}/benchmarks/wf3_benchmarks.md` + `parts_dir` **param** `{project_dir}/benchmarks/_parts` | `gather_benchmarks` out 287, **param 289** |
| historical climate | `{project_dir}/climate_historical/raw_data/extract_historical.nc` | `extract_climate_grid` out, 101; consumed 165 |
| run dirs (tomls+outputs) | `{basin_dir}/run_climate_{experiment}/...` | 227, 245, 247, 248, 260 |
| results | `{exp_dir}/model_results/{Qstats,basin}.csv` | 262–263 (already `experiment`-scoped) |
| plots | (no wf3 plot rule today; class reserved) | — |

**[v2: repo-2, arch-9]** Two rows made precise vs v1: the benchmark row is split
into per-rule `_parts/` writes *and* the `gather_benchmarks` `parts_dir`
**params-string** (line 289) + output (line 287) — a `params:` path, which §7
names as the canonical dry-run-blind surface, so it must move with the per-rule
parts or the gather silently emits an empty report (repo-2). The plots class has
**no wf3 producer today** (confirmed: no rule in `Snakefile_climate_experiment`
writes under `plots/`), so it is *reserved* — pre-partitioned by construction for
a future rule, not a live collision (arch-9).

**[v2: arch-2]** The change is realized by **redefining the single `exp_dir`
binding** (line 62) — see §2 — so the params-level realization paths that derive
from it flow automatically: `prepare_weagen_config` `params.output_path =
f"{exp_dir}/"` (line 132), `prepare_weagen_config_st` `params.output_path =
f"{exp_dir}/realization_.../"` (line 151), and `export_wflow_results`
`params.exp_dir = exp_dir` (line 265). These are `params:` string paths — invisible
to `--dry-run` — so the doc states the redefinition mechanism explicitly rather
than leaving an implementer to edit per-rule expressions and silently miss them.

Two consequences force this milestone:

- **Collision.** A second experiment in the same `project_dir` overwrites the
  config snapshot, logs, benchmarks, and the single `climate_historical/raw_data/
  extract_historical.nc` — even though `model_results/` is already
  experiment-scoped. There is no self-describing record of what settings produced
  a given result.
- **Pollution.** wf3 writes Wflow run artifacts *into* `hydrology_model/`
  (`run_climate_{experiment}/`), so the wf1 "pure built-model" directory
  accumulates per-experiment run debris. This blocks the P3-2 model-swap
  direction, which needs `hydrology_model/` to be a clean, model-neutral input.

Users need multiple stress-test experiments per basin, trackable and
reproducible, **without rebuilding the model**.

### 1a. Two intake assumptions the repo contradicts (recorded; G1 saw these)

Grounding the intake against the tree surfaced two mismatches, both accepted at
G1 as refinements (neither reverses an anchor):

**C1 — `climate_historical/` is already shared between wf1 and wf3, flat, not
per-dataset.** The directory already exists and is written by *both* workflows
into *different flat subdirs*:

- wf1 `add_forcing` writes `{project_dir}/climate_historical/wflow_data/
  inmaps_historical.nc` (`Snakefile_model_creation:185`) — the downscaled
  historical forcing for `run_default`.
- wf3 `extract_climate_grid` writes `{project_dir}/climate_historical/raw_data/
  extract_historical.nc` (`Snakefile_climate_experiment:101`), and
  `extract_historical_climate.py:143` additionally writes
  `{clim_source}_orography.nc` *beside it* in the chirps branch.

So `climate_historical/` holds a wf1 artifact (`wflow_data/`) and a wf3 artifact
(`raw_data/`) today. The per-dataset re-keying (§4) relocates the wf3 `raw_data/`
extraction under a dataset key **without disturbing the wf1
`wflow_data/inmaps_historical.nc` contract** (wf1 is out of scope for behavior
change). Position: keep `wflow_data/inmaps_historical.nc` exactly where it is
(it is wf1's, keyed by the *project* window, not an experiment); re-key only the
wf3 `raw_data/` extraction.

**[v2: risk-6] The C1 pin is load-bearing for a named wf1 consumer.** A grep of
`blueearth_cst/` closes the "confirm no other consumer" gap with a *positive*
result: `setup_time_horizon.py:51` hardcodes
`"../climate_historical/wflow_data/inmaps_historical.nc"` for the `run_default`
toml (`input.path_forcing`). It is **unaffected** because both endpoints stay
project-level (C1 keeps `wflow_data/` exactly put). Recording it here makes the
constraint visible: if a future tidy-up moved `wflow_data/` under the dataset
store, this wf1 default run would break silently. C1's preservation is the
guard against that.

**C2 — the baseline manifest hardcodes `EXPERIMENT_NAME = "experiment"` and the
OLD `climate_experiment/` results path.** `dev/scripts/check_baseline.py:65`
sets `EXPERIMENT_NAME = "experiment"`; `resolve()` (lines 120-125) builds
`exp_dir = f"{project_dir}/climate_{EXPERIMENT_NAME}"`; and the three wf3 TARGETS
(lines 112–114) point at
`{exp_dir}/model_results/{Qstats,basin}.csv` and
`{project_dir}/config/snake_config_climate_experiment.yml`. The new tree (§2)
moves results to `experiments/<name>/model_results/` **and** the config snapshot
to `experiments/<name>/config/`. **This is a re-record surface, not just a value
check** — the manifest *paths* change, so the value-identical re-record (§6) must
repoint `check_baseline.py`'s TARGETS templates and re-resolve (per the accepted
scope amendment).

**[v2: arch-6] The two results TARGETS and the config-snapshot TARGET need
*different* repoints.** Lines 112–113 (Qstats, basin) use the `{exp_dir}`
template and follow the `resolve()`/`EXPERIMENT_NAME` change. Line 114 (the config
snapshot) is templated on **`{project_dir}/config/...`**, not `{exp_dir}/` — under
the new tree the wf3 config snapshot moves to `{exp_dir}/config/`, so line 114
must change its template root from `{project_dir}/config/` to `{exp_dir}/config/`
— a *distinct* edit from the two results rows. §6 step 4 lists both edits.

`dev/scripts/semantic_tree_diff.py` walks the tree structurally and does not
hardcode `climate_experiment`, so it adapts for free (it dispatches by extension
over whatever paths exist) — but it keys on identical relpaths, which is why §6
adds a path-map layer. Its `run_default` special-case
(`semantic_tree_diff.py:328`) is wf1-only and unaffected.

---

## 2. Decision — the target tree

We adopt the intake's confirmed tree shape, with names resolved in §9:

```
<project_dir>/                     # one basin, outside the repo (R6 convention)
  config/                          # PROJECT-LEVEL snapshots (wf1 + wf2 provenance)
    snake_config_model_creation.yml      # wf1 snapshot — GUARD baseline (§3)
    snake_config_climate_projections.yml # wf2 snapshot — GUARD comparand if wf2 ran
  hydrology_model/                 # wf1 built model + run_default — PURE wf1 artifact
  climate_projections/<clim_project>/   # wf2 overlay, project-level, shared
  climate_historical/
    wflow_data/inmaps_historical.nc     # wf1 historical forcing (UNCHANGED — C1)
    <dataset_key>/extract_historical.nc # wf3 per-dataset store (NEW — §4)
    <dataset_key>/<clim>_orography.nc   #   chirps-branch sidecar, co-located
    <dataset_key>/.guard_ok             #   key-level guard artifact (v4 ext2-1, §3b/§3d)
  logs/  benchmarks/  plots/       # wf1 + wf2 scoped ONLY (wf3 moves out)
  experiments/<name>/              # one dir per experiment (name = experiment_name)
    config/snake_config_climate_experiment.yml   # THIS experiment's snapshot
    .project_consistency_ok        # PER-EXPERIMENT guard sentinel (§3, ext1-2)
    stress_test/  realization_*/   # kept AS-IS structurally (anchor 5)
    weathergen_config.yml
    data_catalog_climate_experiment.yml
    model_runs/                    # Wflow tomls + per-rlz/cst outputs
    model_results/                 # Qstats.csv, basin.csv, RT_*.csv
    plots/  logs/  benchmarks/     # incl. benchmarks/_parts/ + wf3_benchmarks.md
```

**Decision statement.** We will move every wf3-owned output that is currently
project-global under a per-experiment root `experiments/<name>/`, move the Wflow
run directory out of `hydrology_model/` into `experiments/<name>/model_runs/`,
convert `climate_historical/` into a project-level per-dataset store keyed by
dataset+window (not copied per experiment), and add a wf3-startup drift guard
that fails loud when an experiment config's project-level sections diverge from
the project snapshot. Config mechanics (single `--configfile`, `configfiles[0]`
forwarding, `get_config`) are unchanged; `experiment_name` selects the subtree.

**[v2: arch-2] The implementation mechanism is exp_dir redefinition.** The primary
edit is `exp_dir = f"{project_dir}/experiments/{experiment}"` at line 62; every
`exp_dir`-derived `output:` and `params:` path (incl. lines 132, 151, 265) moves
with it. The only *manual* edits beyond that are (a) the `project_dir`-rooted wf3
`log:`/`benchmark:`/config-snapshot paths (§2a), (b) the `gather_benchmarks`
`parts_dir` param + output (§2a, repo-2), (c) the run-dir/`basin_dir` paths and
the toml-literal rewrite (§5), (d) the dataset-key extraction path and its two
consumers (§4), (e) the new guard rule + its root-rule guard-artifact/sentinel `input:` wirings
(§3), and (f) **[v3: ext1-4]** the `experiment_name` validation call at parse (§2b).
The change touches `Snakefile_climate_experiment`, `downscale_climate_forcing.py`
(§5), `prepare_climate_data_catalog.py` (§4, chirps sidecar), and
`blueearth_cst/shared/snake_utils.py` (§2b, the validator; **[v4: ext2-2]** also
the `file_digest_or_absent()` helper, §3b). wf1 and wf2 Snakefiles
and scripts are **not** touched except for the shared collision classes below.

### 2a. Shared `logs/`, `benchmarks/`, `config/` — the cross-workflow subtlety

`config/`, `logs/`, and `benchmarks/` are written by **all three** workflows into
the *same* project-level dirs (each `copy_config` writes
`config/snake_config_<workflow>.yml`; benchmark gather writes
`benchmarks/wf{1,2,3}_benchmarks.md`). The collision the intake names is
**wf3-vs-wf3 across experiments**, not wf3-vs-wf1. So:

- **wf3 logs/benchmarks/plots** move to `experiments/<name>/{logs,benchmarks,
  plots}/`. The `3.NN` filename prefix stays (it disambiguates within the
  experiment dir and keeps the naming convention). **[v2: repo-2]** This includes
  BOTH the per-rule `benchmark:` `_parts/3.*` writes AND the `gather_benchmarks`
  rule's `params.parts_dir` (line 289) and `output` (line 287). The gather reads
  its parts from `parts_dir`; if the per-rule parts move but `parts_dir` does not,
  the gather finds no parts and emits an empty/stale `wf3_benchmarks.md` — a
  silent degradation `--dry-run` and every named smoke miss. Both move to
  `experiments/<name>/benchmarks/{_parts/, wf3_benchmarks.md}`, and a gather smoke
  (§7 gate 7) asserts the report is non-empty.
- **wf1/wf2 logs/benchmarks/plots stay** at `{project_dir}/{logs,benchmarks,
  plots}/`. They are project-level (one build, one overlay per project) and do
  not collide across experiments.
- **The wf3 config snapshot** moves from `{project_dir}/config/
  snake_config_climate_experiment.yml` to `experiments/<name>/config/
  snake_config_climate_experiment.yml`. The **project-level** `config/` retains
  the wf1 and wf2 snapshots — and those two become the **drift-guard baseline**
  (§3): the project snapshots the experiment config is compared against.

This split is the crux and the reviewer's first check: the drift guard needs a
*stable project-level snapshot* to compare against, which is exactly the wf1
`config/snake_config_model_creation.yml` (and the wf2
`snake_config_climate_projections.yml`) that stay put.

### 2b. [v3: ext1-4] `experiment_name` validation — centralized, before any path construction

**The gap.** v2 interpolated `experiment_name` directly into filesystem paths
(`exp_dir = f"{project_dir}/experiments/{experiment}"`, and every derived
`output:`/`params:`) with **no validation or containment contract**. A malformed
or adversarial `experiment_name` — path separators, `..`, an absolute form, a
Windows-reserved device name (`CON`, `PRN`, `NUL`, `AUX`, `COM1`…), an empty
value, or a value that normalizes to the same directory as another — could write
outside `experiments/`, collide with another experiment, or fail partway after
creating a partial tree, contradicting the isolation and zero-collision guarantees
(success criterion 1).

**Decision: a centralized `validate_experiment_name()` helper, called once at
Snakefile parse.**

- **Location.** `blueearth_cst/shared/snake_utils.py` (already imported by
  `Snakefile_climate_experiment:9` — no new import surface). The function is a peer
  of the existing `get_config`/`stress_test_grid` helpers.
- **Call site.** `Snakefile_climate_experiment`, immediately after
  `experiment = get_config(my_cfg, "experiment_name", optional=False)` (line 34)
  and **before** `exp_dir` is constructed (line 62). The call is
  `experiment = validate_experiment_name(experiment, project_dir)` — it returns the
  validated name (or raises), so `exp_dir` and all derived paths are built only from
  a vetted value.
- **Grammar (conservative, portable slug — aligned to `dev/conventions/naming.md`).**
  Accept only `^[a-z0-9][a-z0-9_]*$` (lowercase alnum + underscore, must start with
  an alnum), nonempty, with a length cap (**64 chars** — comfortably under the
  Windows path-segment budget while allowing descriptive names). This is a strict
  subset of `naming.md`'s snake_case identifier rule; it deliberately excludes
  hyphens and dots so the value can never introduce a path component or an
  extension. `naming.md`'s three-tier domain-identifier exemptions do not apply —
  `experiment_name` is a path segment, not a domain acronym.
- **Rejections (each raises a clear `ValueError` naming the offending input):**
  - empty / whitespace-only;
  - any of `/ \ : * ? " < > |` or a NUL byte (path separators + Windows-invalid
    characters);
  - `.` or `..` or any value containing a path separator (traversal);
  - an absolute form (leading `/`, `\`, or a `X:` drive prefix);
  - a Windows-reserved device name, compared **case-insensitively** and **including
    any extension** (`CON`, `PRN`, `AUX`, `NUL`, `COM1`–`COM9`, `LPT1`–`LPT9`);
  - a trailing dot or space (Windows strips these, creating a normalization
    collision) — already excluded by the grammar, asserted defensively.
- **Containment assertion (belt to the grammar's braces).** After grammar
  validation, resolve the target dir and assert it is a **direct child** of
  `<project_dir>/experiments`:
  `Path(project_dir, "experiments", name).resolve().parent ==
  Path(project_dir, "experiments").resolve()`. This is a second, independent check:
  even if the grammar were later loosened, a name that escaped the experiments dir
  fails here. (`.resolve()` at parse is safe — it does not require the dir to
  exist.)

**Parse-time is correct here (unlike the guard).** §3 keeps the *drift guard* out
of parse time because a guard failure at parse would abort the DAG build on
`--dry-run`/`--unlock` and break lock recovery. **`experiment_name` validation is
different**: it is *config validation*, not a project-state check. A malformed
`experiment_name` makes the entire DAG ill-defined (every path is built from it),
so failing at parse — including under `--dry-run` — is the **correct** behavior:
`--dry-run` on a config that names an invalid experiment *should* fail loud, and
`--unlock` is unaffected because `--unlock` needs only a *parseable* Snakefile, and
a validated-or-raised `experiment_name` is parseable in the valid case (the invalid
case has no recoverable lock to release — there is no valid workdir for it). This
asymmetry (parse-time OK for config validation, rule-time required for the
state-dependent guard) is deliberate and documented.

**[v4: ext2-3] Case policy — reject, never normalize.** The grammar is
lowercase-only and the validator **never silently lowercases**: an uppercase name
(e.g. `experiment_A`) is rejected with a `ValueError` naming the offending value
and the grammar, exactly like any other grammar violation. (v3's test matrix
listed `experiment_A` as a passing fixture, contradicting the grammar — a fixture
error, corrected below; the grammar itself stands, fixed by ext1-4 and aligned to
`dev/conventions/naming.md`.)

**Test matrix (commit 2, `test_validate_experiment_name.py`):**
**[v4: ext2-3]** `experiment_a` ⇒ pass (returns unchanged); `experiment_A` ⇒
ValueError (uppercase — rejected with a message naming the grammar, **not**
silently lowercased); `""` ⇒ ValueError; `..` ⇒ ValueError;
`a/b` and `a\b` ⇒ ValueError (separators); `/abs` and `C:\x` ⇒ ValueError
(absolute); `con` / `CON` / `nul.txt` ⇒ ValueError (reserved, case- and
extension-insensitive); `Exp-1` ⇒ ValueError (hyphen not in grammar); `exp.1` ⇒
ValueError (dot); a 65-char name ⇒ ValueError (length cap); `exp ` (trailing space)
⇒ ValueError. A `--dry-run` on `Snakefile_climate_experiment` with a scratch config
carrying `experiment_name: "../evil"` fails at parse naming the bad value (gate 1
extension).

---

## 3. Drift guard mechanics

**Purpose.** An experiment config is a *full* config (approach A). Its
project-level sections (`project`, `shared.basin`, `workflows.model_creation`,
`workflows.climate_projections`) must describe the *same* project the built
model/overlay came from — otherwise the experiment silently reuses a
`hydrology_model/` built under different settings. The guard fails loud on
divergence.

**Where it runs.** At **wf3 rule time**, as a new first rule
`check_project_consistency` whose guard outputs (**[v4: ext2-1]** the
per-experiment sentinel *and* the key-level guard artifact, §3a/§3b) are threaded
into the wf3 root rules (see "DAG wiring" below), **not** at Snakefile parse
time. Rationale:

- Parse-time (a bare `assert` in the Snakefile body) runs on *every* invocation
  including `--dry-run` and `--unlock`, and would abort the DAG build before
  Snakemake can even report the lock state — hostile to the documented
  `--unlock` recovery workflow (AGENTS.md). A rule runs only when the DAG
  executes. (Contrast §2b: `experiment_name` validation *is* parse-time because it
  is config validation, not a project-state check — see §2b.)
- A rule gets a `log:`/`benchmark:` and participates in the gather machinery
  uniformly with the other `3.NN` rules, and its failure is a clean rule stop
  naming the diverging key, not a Python traceback at parse.

### 3a. DAG wiring — the guard must be threaded into the root rules

**[v2: arch-1, repo-1 — the panel's core defect]** v1 claimed the guard is a first
rule that "every downstream wf3 rule depends on" *and* (commit 1) that it is
"additive; existing wf3 paths unchanged." **Both cannot hold, and v1 was
self-contradictory.** The wf3 DAG has **five independent root rules** — verified
against `Snakefile_climate_experiment`:

| # | Root rule | line | why it is a root (no upstream wf3 dep) |
| --- | --- | --- | --- |
| 1 | `copy_config` | 78 | input is `config_path` (the live config), not a rule output |
| 2 | `extract_climate_grid` | 91 | input is `ancient(region.geojson)` (a wf1 artifact) |
| 3 | `climate_stress_parameters` | 110 | input is `ancient(config_path)` |
| 4 | `prepare_weagen_config` | 124 | no `input:` at all |
| 5 | `prepare_weagen_config_st` | 144 | no `input:` at all |

Everything else in wf3 descends transitively from these five (extraction →
realization → stress-test → catalog → downscale → run → export). Nothing
consumes `copy_config`'s snapshot except `rule all`. So a sentinel that is merely
*produced* by the guard rule but *consumed by nothing* is an **orphan output**:
Snakemake never schedules it, and the data-producing roots run against a project
whose consistency was never verified.

**Decision: the guard produces TWO artifacts, and each root consumes the one
whose path identity matches its sharing class (ext1-2, revised by ext2-1).** The
guard must be reachable from `rule all` *and* run before any data-producing rule
fires. **[v4: ext2-1]** v3 had the shared root consume the *per-experiment*
sentinel with `ancient()`; a second probe (§3d) confirmed the r2 reviewer:
`ancient()` suppresses only the *mtime* trigger, and swapping the sentinel *path*
(A→B) still re-triggers the shared rule via Snakemake's **input-set provenance
trigger**. The guard rule therefore has two outputs (§3b), consumed as follows —
**exactly these rules consume exactly these artifacts:**

- **The four non-shared per-experiment roots take the PER-EXPERIMENT sentinel
  `experiments/<name>/.project_consistency_ok` FRESH** (ordinary `input:`):
  `copy_config`, `climate_stress_parameters`, `prepare_weagen_config`,
  `prepare_weagen_config_st`. Their outputs live under `experiments/<name>/`, are
  never shared across experiments, and *should* re-run if the guard re-runs.
- **[v4: ext2-1] The shared root `extract_climate_grid` takes the KEY-LEVEL guard
  artifact `climate_historical/<key>/.guard_ok` with `ancient()`** — a path that
  is **invariant across every experiment sharing the key**, so the shared rule's
  input set never changes between A and B and the input-set trigger cannot fire;
  `ancient()` additionally suppresses the mtime/updated-by-another-job triggers
  when a later experiment's guard rewrites the artifact (all probe-verified,
  §3d). A *new* key K′ has no `.guard_ok` yet, so a new-key extraction is still
  ordered after — and blocked by — the current experiment's guard. **No other
  rule consumes `.guard_ok`;** `generate_weather_realization` consumes the keyed
  `extract_historical.nc` `ancient()` (§4b, unchanged).

**Per-experiment sentinel path.** The sentinel is
`experiments/<name>/.project_consistency_ok` — **[v3: ext1-2]** per-experiment, not
project-global. Each experiment gets its own guard pass record; B's sentinel and
A's sentinel are distinct files. **[v4: ext2-1]** The sentinel is consumed **only**
by the four per-experiment roots; the shared extraction consumes the key-level
`.guard_ok` instead, so B's sentinel never appears in the shared rule's input set
at all.

*(Alternative considered and rejected: gate only the model-consuming rules
`downscale_climate_realization` + `run_wflow`. Rejected because §3's
provenance-for-all-artifacts framing requires the extraction, parameter grid, and
weagen configs to also be produced only against a verified project — gating just
the model run would let those artifacts land against an unverified project. Wiring
the roots is the same edit count and closes the gap. — A7, §8.)*

### 3b. Rule shape and the two snapshots

**Rule shape** (spec, not code — implementation is the task-brief handoff):

- Rule `3.00b check_project_consistency` (numbered before `copy_config`).
- **input:** the wf1 project snapshot
  `{project_dir}/config/snake_config_model_creation.yml` (written by wf1
  `copy_config`), marked `ancient(...)` so its **mtime** never forces a wf3
  rebuild (its **content** change is instead tracked by a params digest — §3c).
- **params:** **[v4: ext2-1] every guard param must be EXPERIMENT-INVARIANT
  across passing configs** — load-bearing: the keyed `.guard_ok` output (below) is
  shared across experiments, so an experiment-varying param (e.g. the config
  *path*) would flip the recorded params on the shared output at every A↔B
  alternation, re-run the guard, rewrite A's sentinel, and cascade a spurious
  re-run through the four fresh consumers. Concretely:
  - **a SHA-256 digest of the guarded live-config sections** (a
    canonical-serialization — e.g. sorted-key JSON — of `project` + `shared.basin`
    + `workflows.model_creation` + `workflows.climate_projections`, hashed to a
    string) so live-config drift trips the params rerun-trigger (§3c). A **string
    digest**, not the raw nested section dicts — the string form is exactly what
    the §3c probe verified triggers on; threading raw dicts as params is beyond
    the verified evidence and is deliberately avoided. Two experiments that both
    PASS the guard have guarded sections deep-equal to the same snapshots, hence
    **equal digests — the invariance holds by construction**. **[v4: ext2-1]** The
    live config *path* (`config_path`) is **REMOVED from the guard's params** (it
    varies per experiment); the guard script reads the live guarded sections from
    the parsed `snakemake.config` that Snakemake injects, and the experiment's own
    config snapshot (`copy_config`) remains the provenance record of which config
    file ran. The gate-4 alternation smoke pins this constraint.
  - the four guarded section keys (constant strings);
  - **[v3: ext1-1]** a **content digest of the wf1 snapshot** — **[v4: ext2-2]**
    computed at parse via the absence-tolerant `file_digest_or_absent()` helper
    (below) — this is what makes a snapshot-only content change re-trigger the
    guard despite `ancient()` (verified §3c);
  - **[v2: risk-2 / v3: ext1-1]** the wf2 snapshot path
    `{project_dir}/config/snake_config_climate_projections.yml` **as a params path,
    existence-checked in-script — NOT a rule `input:`** (a project-level constant
    path, experiment-invariant) — **plus its content digest via the same
    `file_digest_or_absent()` helper** (**[v4: ext2-2]** the literal `"ABSENT"`
    when the file is missing, replacing v3's "empty-sentinel string"), so a wf2
    snapshot content change also re-triggers.
- **output:** **[v4: ext2-1] two artifacts, one per sharing class:**
  1. the per-experiment sentinel `experiments/<name>/.project_consistency_ok` (a
     small text file, not `temp()` — it records the pass for provenance; fresh
     `input:` of the four per-experiment roots, §3a);
  2. the **key-level guard artifact** `climate_historical/<key>/.guard_ok`
     (`<key>` resolved from the experiment's `clim_historical` +
     `historical_window`, exactly as §4 keys the store) — an experiment-invariant
     path for every experiment sharing the key, consumed `ancient()` by
     `extract_climate_grid` **only** (§3a, §3d).
  A failing guard writes neither (and Snakemake removes a failed job's outputs),
  so both consumer classes stay blocked on failure.
- **script:** a new `blueearth_cst/experiment/check_project_consistency.py`
  reusing the R6 normalize-then-compare comparator pattern.

**[v2: risk-2] The wf1-input / wf2-params asymmetry — why the projections overlay
is not force-required.** v1's §2a said "wf1 AND wf2 snapshots become the guard
baseline" and §3 guarded `workflows.climate_projections`, but §3's rule `input:`
listed *only* the wf1 snapshot — so the experiment's `climate_projections` section
was validated against **wf1's stale copy**, and an edit-then-rerun-wf2 (not wf1)
would false-fire. The fix keeps `climate_projections` guarded (anchor 3 names it —
dropping it would reopen an anchor) but sources its comparand correctly:

- **wf1 snapshot = a hard rule `input:` (ancient).** Mandatory. If absent,
  Snakemake raises `MissingInputException` naming the file before the script runs
  — wf1 has never built the model, so wf3 cannot proceed. This is the intended
  hard fail.
- **wf2 snapshot = a params path, existence-checked in the script.** The
  projections overlay is **optional** per the CST method (AGENTS.md Background:
  CMIP6 is a plausibility overlay only; wf3 does not consume
  `workflows.climate_projections` as a computational input). Making the wf2
  snapshot a rule `input:` would force wf2 to have run before *any* wf3 run —
  contradicting "overlay optional." Instead the guard compares the experiment's
  `workflows.climate_projections` section against the **wf2 snapshot only when it
  exists**; when the wf2 snapshot is absent (wf2 never ran), the guard **logs
  that the projections section is unchecked and passes** that section (it does not
  fall back to the wf1 copy). The other three sections (`project`, `shared.basin`,
  `workflows.model_creation`) are always compared against the mandatory wf1
  snapshot.

This one-to-one alignment — each guarded section compared against the snapshot of
the workflow that owns it, projections conditionally — is what resolves risk-2's
false-fire-by-edit-order without forcing the optional overlay.

**[v4: ext2-2] Digest computation is absence-tolerant — `file_digest_or_absent()`.**
The r2 reviewer is correct that computing the mandatory wf1 snapshot digest "at
parse" conflicts with the specified rule-time missing-input behavior: hashing a
nonexistent snapshot during Snakefile evaluation would raise **before** Snakemake
can build the DAG, report the designed `MissingInputException`, or honor
`--unlock` — breaking exactly the missing-project-state path (fresh or partially
recovered project) the design claims to handle. The digest params are therefore
computed via a small helper, **`file_digest_or_absent(path) -> str` in
`blueearth_cst/shared/snake_utils.py`** (a peer of `get_config` /
`validate_experiment_name`), whose two branches are:

- **present:** return the SHA-256 hex digest of the file bytes — unchanged
  triggering semantics; any content change flips the param (verified §3c);
- **missing:** return the literal sentinel string `"ABSENT"` — it **never
  raises**. `"ABSENT"` cannot collide with a real digest (uppercase, non-hex,
  wrong length).

It is called at parse in the Snakefile body for **both** the wf1 and wf2 snapshot
digests. Consequences: a fresh project **parses, `--dry-run`s, and `--unlock`s
cleanly**; wf1-snapshot absence surfaces at the **rule level** via the
`ancient()` input declaration (below), exactly as designed; and the
ABSENT→present transition itself flips the digest param, so the first
post-wf1 invocation re-evaluates the guard. Regression checks: gate 2c.

**Missing-snapshot behavior (wf1).** If
`{project_dir}/config/snake_config_model_creation.yml` does not exist, the rule
`input:` is missing and Snakemake raises `MissingInputException` naming the file.
The script also emits a friendlier message (*"No project snapshot at
`config/snake_config_model_creation.yml`; run Snakefile_model_creation first."*) as
a duplicate, not the sole gate.

**Exact guarded key set.** The guard compares, between the live experiment config
and the appropriate snapshot:

- `project` (entire section — `project_dir`, `static_dir`, `data_sources`,
  `data_sources_climate`) — vs wf1 snapshot.
- `shared.basin` (`region`, `resolution`) — vs wf1 snapshot.
- `workflows.model_creation` (entire section) — vs wf1 snapshot.
- `workflows.climate_projections` (entire section) — vs **wf2 snapshot if
  present**, else unchecked+logged.

**NOT guarded** (experiment-variable, anchor 1): `shared.historical_window`,
`shared.clim_historical`, and all of `workflows.climate_experiment`.

**[v2: open-question-4 resolved] Nothing else in `shared` is a hidden guarded
value.** Grep of `Snakefile_climate_experiment` for `shared_cfg[...]`/
`get_config(shared_cfg, ...)` reads exactly three keys: `clim_historical` (line
48), `historical_window` (line 54) — both deliberately experiment-variable and
unguarded — and `basin` (via `shared.basin`, guarded). There is no fourth
`shared.*` key wf3 consumes, so the guarded set is complete for wf3's inputs.

**[v2: repo-5, arch-5] Normalization is DEFENSIVE, symmetric, and not the core
mechanism.** v1 cited reuse of `_normalize_config_paths` +
`COPIED_CONFIG_PATH_MAP` (`semantic_tree_diff.py:238-254`) so "flat compares equal
to binned." But that map is **directional** (pre-R6 flat → post-R6 binned) and, in
`compare_copied_config`, is applied to exactly one side after a reflexivity guard.
For the drift guard **both operands are contemporaneous post-R6 configs**, so the
map matches nothing and is a **no-op** — a plain section-scoped deep-equal already
gives the correct pass/fail. The guard's core mechanism is therefore **section-
scoped deep structural equality** on the guarded sections; normalization is a
*defensive* layer for the one edge case where a hand-migrated flat-path experiment
config is compared against a binned snapshot. If normalization is kept, the guard
applies the OLD→NEW map **symmetrically to BOTH operands** before deep-compare
(so a flat-vs-binned pair converges while two binned configs are unchanged) —
explicitly *unlike* `compare_copied_config`'s directional one-side application.
Gate 2(d) ("flat-vs-binned ⇒ pass") pins this symmetric-defensive behavior.

### 3c. [v3: ext1-1 — BLOCKING] Sentinel-invalidation semantics on the verified trigger matrix

**The finding.** The reviewer argued the persistent `.project_consistency_ok`
sentinel can be **reused after a comparand changes**, because the live config /
wf2 snapshot are only `params` (claimed not to participate in freshness) and the
wf1 snapshot is `ancient()` (mtime ignored). If true, a user could edit a guarded
project section after a first successful run and then execute wf3 with the existing
sentinel — proceeding against a mismatched model, defeating the guard's core
safety contract.

**Empirical verification (this session, throwaway probe against pinned Snakemake
9.6.2, dry-runs + tiny scratch runs only, in the scratchpad — no repo edit, no real
workflow).** I replicated the guard's exact rule shape (an `ancient()` snapshot
input, `params` comparands, a persistent non-`temp()` sentinel output, a downstream
root consuming it) and probed each comparand-change case. **Verbatim reasons
reported by Snakemake:**

| Case | What changed | Snakemake verdict |
| --- | --- | --- |
| **control** | nothing | `Nothing to be done (all requested files are present and up to date).` |
| **(a) params-only** | live comparand `live_A`→`live_B` (same snapshot) | **RE-RUNS** — `reason: Params have changed since last execution: … before: 'live_A' now: 'live_B'` |
| **(b) `ancient()` input content** | snapshot content `v1`→`v2` (same params) | **DOES NOT re-run** — `Nothing to be done` |
| **(c) both** | snapshot content + params both changed | **RE-RUNS** — `reason: Params have changed since last execution: … before: 'live_A' now: 'live_C'` |
| **fix probe: non-ancient input, content change** | snapshot as regular (non-`ancient`) input, content changed | **RE-RUNS** — `reason: Updated input files: snapshot.txt` |
| **fix probe: params content digest, content change** | snapshot NOT an input; SHA-256 of snapshot bytes carried as a `param`; content changed | **RE-RUNS** — `reason: Params have changed since last execution: … before: '74e1d2f7924c' now: '0b6a6f768348'` |
| **fix probe: params digest, content reverted** | digest param, snapshot reverted to byte-identical content (newer mtime) | **DOES NOT re-run** — `Nothing to be done` (content-addressed, mtime-immune) |

**Conclusions.**

1. **The reviewer is HALF right.** The live-config side is **already covered**:
   Snakemake 9.6.2's **default `params` rerun-trigger** re-runs a rule with an
   existing output when a `param` value changes (case (a)) — this repo sets no
   `--rerun-triggers` override, and R5 independently verified the same behavior on
   `extract_climate_grid` (dev/followups.md §R3: an `historical_window` edit
   schedules the rule with reason "Params have changed"). So **threading the guarded
   live-config section values through as `params` (not just the config *path*) makes
   any live-config drift re-trigger the guard.** The reviewer's premise that "params
   do not participate in freshness" is **factually wrong for this pinned Snakemake**
   — stated plainly here and in the ledger.
2. **The reviewer is right on the `ancient()` snapshot side (case (b)).** An
   `ancient()` input's **content** change does **not** re-trigger. So the wf1/wf2
   snapshots, wrapped in `ancient()` (v2's shape), would let a snapshot-only content
   change (e.g. wf1 rebuilt with edited project sections → snapshot overwritten)
   slip past a still-present sentinel. This is the real hole.
3. **The fix (verified).** Carry a **SHA-256 content digest of the wf1 snapshot —
   and of the wf2 snapshot when present — as guard `params`.** The digest probe
   shows a snapshot content change flips the digest and re-triggers via "Params have
   changed", *and* that reverting the snapshot to byte-identical content restores
   "Nothing to be done" — i.e. the invalidation is **content-addressed, not
   mtime-addressed**, so it neither misses a real change nor false-fires on a
   touch/rebuild that produced identical bytes.

**Decision (belt-and-braces, each layer verified):**

- **A SHA-256 digest of the guarded live-config sections → a guard `param` (string).**
  Covers live-config drift (case (a), verified). Not just `config_path` — a
  canonical-serialization digest of the four guarded sections' *values*, so an
  in-place edit flips the digest and trips the params trigger. A **string** digest
  (matching the verified probe form), not raw nested-dict params (unverified for
  param-change detection, avoided). (**[v4: ext2-1]** `config_path` itself is
  dropped from the guard params entirely — it is experiment-varying and would
  defeat the shared keyed guard output; §3b, §3d.)
- **wf1 snapshot → `ancient()` rule `input:` (mandatory-existence) + its SHA-256
  content digest as a `param`** (**[v4: ext2-2]** via the absence-tolerant
  `file_digest_or_absent()`, §3b — so a missing snapshot never breaks parse). The
  `ancient()` input gives the hard missing-snapshot fail (MissingInputException)
  and the ordering edge; the digest `param` supplies the content-change trigger
  the `ancient()` mtime cannot (case (b) → covered by the digest, verified).
- **wf2 snapshot → params path + its SHA-256 digest as a `param`**
  (**[v4: ext2-2]** the `"ABSENT"` literal from `file_digest_or_absent()` when
  the file is missing). Same content-trigger coverage, conditional on existence,
  without forcing the optional overlay.

This makes **every comparand change re-trigger the guard**: live-config edit
(params values), wf1 snapshot change (wf1 digest param), wf2 snapshot change (wf2
digest param) — none relies on `ancient()` mtime.

**Gate additions (the reviewer's mutate-each-comparand test, adopted).** §7 gate 2b
(a named INTEGRATION/`--dry-run` test, distinct from the gate-2 script unit tests)
runs **without deleting the sentinel** between mutations:
- (i) run guard to success; mutate a guarded **live-config** section; re-invoke ⇒
  guard **re-runs and FAILS** on the divergence (not skipped).
- (j) restore live-config; mutate the **wf1 snapshot** content (simulate a wf1
  rebuild under different project sections); re-invoke ⇒ guard **re-runs** (digest
  changed) and **FAILS** if the new snapshot diverges from the experiment config.
- (k) with a wf2 snapshot present, mutate its content ⇒ guard **re-runs** (wf2
  digest changed).
- (l) revert every comparand to its original bytes ⇒ guard **does NOT re-run**
  (content-addressed; no false-fire) — pins the mtime-immunity.

**[v3] Empirically verified against pinned Snakemake 9.6.2 this session — no
`[UNVERIFIED]` marker on this matrix.** The probe used only dry-runs and tiny
scratch Snakefiles in the scratchpad; the guard's real rule is the task-brief's
implementation, gated by 2b (i–l).

### 3d. [v3: ext1-2 / v4: ext2-1] The guard / shared-store interaction — settled by a second probe

**The r1 finding (ext1-2).** Threading each experiment's *fresh* sentinel into the
**shared** `extract_climate_grid` rule conflicts with dataset-store reuse (§4b):
B's newer sentinel could schedule extraction again for the same key, degrading
"extracted once, referenced by N experiments." v3 resolved this with a
per-experiment sentinel consumed `ancient()` by the shared rule.

**The r2 finding (ext2-1).** The r2 reviewer objected that v3 verified the wrong
trigger: the §3c probe changed the **content/mtime** of one `ancient()` input,
whereas A and B hand the shared rule **different input paths** (`sentinel(A)` vs
`sentinel(B)`), and Snakemake's default rerun-triggers include the **input-set
identity** independently of timestamps — `ancient()` suppresses only the mtime
side.

**[v4: ext2-1] Empirical verification (this session, throwaway probe against
pinned Snakemake 9.6.2, dry-runs + tiny scratch runs in the scratchpad — no repo
edit, no real workflow).**

*Probe 1 — v3's exact shape* (per-experiment sentinel consumed `ancient()`,
sentinel path swapped A→B with identical content, shared keyed output present):
the shared rule **RE-RUNS**. Verbatim Snakemake reason:

> `set of input files has changed since last execution:`
> `    extract`
> `Some jobs were triggered by provenance information, see 'reason' section in the`
> `rule displays above.`

**The reviewer is CONFIRMED: v3's §3d mechanism is broken.** The input-set
provenance trigger fires on the path substitution even though the keyed output is
present, the sentinel content is identical, and the input is `ancient()`.
Experiment B would re-extract the shared dataset/window, defeating the
extracted-once contract and gate 4's expected A→B skip.

*Probe 2 — the fix shape* (shared rule depends on a **stable,
experiment-invariant guard-artifact path**; per-experiment sentinel retained for
a per-experiment root; B's guard job **scheduled** — B's sentinel missing — and
rewriting the stable artifact as one of its outputs): the shared rule is **NOT
scheduled**. Verbatim job stats + reasons of B's dry-run:

> `job             count`
> `------------  -------`
> `guard               1`
> `per_exp_root        1`
> `total               2`
> `…`
> `Reasons:`
> `    input files updated by another job:`
> `        per_exp_root`
> `    output files have to be generated:`
> `        guard, per_exp_root`

The per-experiment root (fresh input) **is** re-triggered by the rewritten
sentinel — by design — while the shared rule, whose input set is path-invariant
and whose guard-artifact input is `ancient()`, is untouched even though that
input is an output of the scheduled guard job.

*Probe 3 — after B's REAL run* (stable artifact rewritten on disk, newer mtime),
a follow-up dry-run of the shared output: `Nothing to be done (all requested
files are present and up to date).` — the rewrite does not retro-trigger the
`ancient()` consumer.

**Resolution — a KEY-LEVEL guard artifact makes the shared rule's input set
experiment-invariant (the reviewer's shape, adopted).**

1. **The guard rule gains a second output** (§3b):
   `climate_historical/<key>/.guard_ok`, keyed exactly like the store (§4). For
   every experiment sharing dataset+window key K the path is identical, so the
   shared rule's input set `{ancient(region.geojson), ancient(K/.guard_ok)}`
   never changes across experiments — the input-set trigger cannot fire
   (probe 2) — and `ancient()` covers the mtime/updated-by-another-job side when
   a later experiment's guard rewrites the artifact (probes 2–3).
2. **The per-experiment sentinel remains** for the four per-experiment roots
   (fresh) and as the experiment's ordering/provenance record (§3a) — it no
   longer appears in the shared rule's input set at all.
3. **Per-experiment gating of the shared rule is preserved where it matters.** A
   *new* key K′ (experiment C) has no `K′/.guard_ok`, so C's extraction is
   ordered after — and blocked by — C's **own** guard: a failing guard(C) writes
   neither artifact, and the new-key extraction never runs against an unverified
   config. For an *existing* key K, a divergent experiment's failing guard blocks
   its per-experiment pipeline while the already-verified K extraction is simply
   reused — nothing new is produced against an unverified config.
4. **Forced params constraint** (§3b): because `K/.guard_ok` is a **shared**
   output, the guard's params must be experiment-invariant across passing
   configs — `config_path` is removed from the params. Otherwise A↔B alternation
   would flip the recorded params on the shared output, re-run the guard, rewrite
   A's sentinel, and cascade a spurious re-run through the four fresh consumers.
   The gate-4 alternation smoke pins this.

**Failure-cleanup caveat (recorded; recoverable, not a bug).** If a guard job
that lists an existing `K/.guard_ok` among its outputs FAILS (divergent config),
Snakemake's failed-job output cleanup may remove the pre-existing artifact.
Recovery is automatic and cheap: the next passing invocation's guard re-runs
(missing output), rewrites it, and the keyed extraction still skips — probes 2–3
show neither the scheduled producer nor the rewrite re-triggers the `ancient()`
consumer. No manual intervention needed.

**Exact A-then-B-then-C DAG behavior (pinned by gate 4):**
- **A (first run, key `K` empty):** guard(A) runs → writes sentinel(A) +
  `K/.guard_ok`; `extract_climate_grid` for K has no keyed output yet ⇒ **runs**,
  produces `climate_historical/K/extract_historical.nc`.
- **B (same `clim_historical` + `historical_window` ⇒ same key `K`):** guard(B)
  runs (B's sentinel missing) → writes sentinel(B), rewrites `K/.guard_ok`;
  `extract_climate_grid` for B resolves to the **same** keyed output (present)
  with the **same** input set `{region.geojson, K/.guard_ok}` — no input-set
  trigger (probe 2), no mtime/updated-by-job trigger (`ancient()`, probes 2–3) ⇒
  **extraction SKIPPED** (reused). B's downstream `generate_weather_realization`
  takes the keyed `extract_historical.nc` as `ancient()` input (§4b) and
  proceeds.
- **C (different window ⇒ different key `K′`):** guard(C) runs → writes
  sentinel(C) + `K′/.guard_ok` (a new file); `extract_climate_grid` for C
  resolves to `K′/extract_historical.nc`, which does **not** exist ⇒ **runs** —
  and only after guard(C) succeeds (the `K′/.guard_ok` dependency edge). Genuine
  re-extraction, as intended.
- **A re-invoked after B (alternation):** all of A's outputs exist; the guard's
  params are experiment-invariant across passing configs, so the params recorded
  on the shared `K/.guard_ok` (from B's run) match A's ⇒ **"Nothing to be done"**
  — no params thrash, no cascade (gate-4 alternation smoke).

**Snapshot freshness (wf1 re-run after edits).** The snapshot is a *copy* of the
config wf1 last ran with. If a user edits the project sections and re-runs wf1,
wf1's `copy_config` overwrites the snapshot; the guard's **wf1-digest param** (§3c)
then changes, so the guard re-runs and compares against the *new* snapshot. If the
user edits project sections but does **not** re-run wf1, the snapshot is unchanged
but the **live-config params** change (§3c case (a)), so the guard re-runs and
(correctly) fires on the divergence, forcing the user to either revert the
experiment's project sections or rebuild the model. This is the intended fail-loud
contract — the guard cannot distinguish "I meant to rebuild" from "I forgot to", so
it refuses to run wf3 against a model whose provenance no longer matches. The
snapshot's own freshness vs the built staticmaps is **not** independently checked
(no mtime-vs-staticmaps comparison) — a `reproducible-computing` refinement deferred
to P3-2/P3-3 if needed.

**Guarded-side asymmetry note.** The guard compares *sections* of the experiment
config against the corresponding snapshots; both are full configs. The
`workflows.climate_experiment` section is present in the wf1 snapshot too, but is
deliberately excluded from the compare set (two experiments legitimately differ
there). This is why the guard is a section-scoped deep compare, not a
snapshot==config byte equality.

---

## 4. Historical-climate store: dataset keying vs the window

**The tension.** The wf3 extraction (`extract_climate_grid`,
`extract_historical_climate.py`) depends on **both** the source dataset
(`shared.clim_historical`, e.g. `era5`, `chirps`, `cru_ts`) **and** the window
(`shared.historical_window.{starttime,endtime}`). Both are experiment-variable
(anchor 1). Two experiments may share a dataset but differ in window.

**Decision: key the store by dataset + window.** The store path becomes:

```
{project_dir}/climate_historical/<clim_source>_<start>_<end>/extract_historical.nc
```

where `<start>`/`<end>` are the window dates rendered compactly (e.g.
`era5_20000101_20201231/`). Rationale and the rejected alternatives are in §8.

### 4a. [v2: risk-1 — BLOCKING] The CHIRPS orography sidecar has a second consumer

v1 asserted "no script edit needed for the sidecar" and its `[UNVERIFIED]` grep
only checked the extract producer and the Snakefile (lines 101/165). **That missed
a second, tested consumer.** Verified against the tree:

`prepare_climate_data_catalog.py:76-84` (rule 3.08 `climate_data_catalog`, script
`blueearth_cst/projections/prepare_climate_data_catalog.py`) builds the
`<src>_orography` catalog entry by hardcoding, for `source_like ∈
{chirps, chirps_global}`:

```python
fn_oro = Path(fns[0]).resolve()          # fns[0] is a realization NC
fn_oro = os.path.join(
    os.path.dirname(fn_oro), "..", "..",
    "climate_historical", "raw_data", f"{source_like}_orography.nc")
```

This path **breaks twice** under the new tree:

1. **Target moved.** The sidecar moves from `climate_historical/raw_data/` to the
   dataset-key dir `climate_historical/<clim_source>_<start>_<end>/` (§4, because
   `extract_historical_climate.py:143` derives it from
   `os.path.dirname(fn_out)`, and `fn_out` now lands under the keyed dir). The
   hardcoded `raw_data/` literal points at a non-existent location.
2. **Depth changed.** `fns[0]` is a realization NC. Under the old tree the
   realization dir is `climate_{experiment}/realization_N/` — 2 levels below
   `project_dir`, so `../../climate_historical/...` resolves correctly. Under the
   new tree it is `experiments/<name>/realization_N/` — **3 levels** below
   `project_dir`, so `../..` is now one level short.

For any chirps / chirps_global experiment, rule 3.08 would write a
`data_catalog_climate_experiment.yml` whose `<src>_orography` entry points at a
nonexistent file, and rule 3.09 (`downscale_climate_realization`) fails at runtime
when it resolves that catalog source. **This escapes v1's own verification**:
every gate (3–6) runs the era5 seed config, and the chirps branch is reached only
when `clim_historical` is chirps — an anchor-1 experiment-variable value.

**Fix — named edit site + a non-era5 gate.** `prepare_climate_data_catalog.py:76-84`
becomes a **named edit site** in the store-keying change surface (§2, §10 commit
4). The orography lookup is rewritten to derive the sidecar path from the **same
dataset+window key** the extraction used, at the **correct depth**. The cleanest
mechanism, consistent with the rest of the codebase's absolute-path style, is to
**pass the keyed store dir (or the resolved sidecar path) as a rule `param`** from
`Snakefile_climate_experiment` — the Snakefile already computes the keyed dir for
the extraction output, so it hands the same key to rule 3.08 rather than letting
the script reconstruct it by fragile `../..` walking. The script then reads
`snakemake.params.oro_path` (or `.store_key_dir`) instead of the hardcoded literal.
This removes the depth dependency entirely.

**Non-era5 execution gate (closes the "gates only exercise era5" escape class).**
The gate ladder (§7) gains gate 8: a **chirps-path smoke** — run rule 3.08 on a
chirps (or chirps_global) staged config for one rlz/cst and assert (a) the emitted
`data_catalog_climate_experiment.yml` `<src>_orography.uri` resolves to an existing
file under the keyed dir, and (b) rule 3.09 consumes that catalog without a
missing-source error. Where a full chirps stage is unavailable in CI, a **unit
test on the rewritten lookup** (`test_prepare_climate_data_catalog.py`: given a
realization path under `experiments/<name>/realization_N/` and a store key, the
computed orography path equals the keyed-dir path and exists) is the minimum
mandatory substitute. Either way, the chirps branch is no longer structurally
unexercised.

### 4b. Reuse / staleness semantics

- **Reuse:** if `climate_historical/<key>/extract_historical.nc` already exists
  (a prior experiment extracted the same dataset+window), a new experiment
  referencing the same key **reuses it** — no re-extraction. Snakemake enforces
  this: the extraction rule's output path *is* the keyed path, so a second
  experiment whose rule resolves to the same output finds it present and skips the
  rule. This is the "extracted once, referenced by N experiments" behavior the
  amendment asks for, with zero copy: every experiment's
  `generate_weather_realization` rule takes the keyed path as `ancient(...)` input.
  **[v3: ext1-2 / v4: ext2-1]** The guard/shared-cache interaction that could
  have broken this reuse is resolved in §3d — revised in v4 after a second probe
  confirmed that swapping the per-experiment sentinel *path* re-triggers the
  shared rule via the input-set provenance trigger despite `ancient()`. The
  shared rule now consumes the experiment-invariant **key-level**
  `climate_historical/<key>/.guard_ok` (`ancient()`), so its input set is
  identical for every experiment sharing the key and B cannot mark the shared
  keyed output stale (probes 1–3, §3d). Gate 4 pins the A→B skip.
- **Re-extract:** wf3 re-extracts **only** when the resolved key has no
  `extract_historical.nc` — a genuinely new dataset+window combination. Different
  window ⇒ different key ⇒ new extraction. Different dataset ⇒ different key ⇒
  new extraction.
- **[v2: risk-3] Staleness — corrected claim + explicit assumption.** v1 claimed
  "the key encodes the full identity of the extraction inputs ... a present keyed
  file is *never stale* within a project." **This is overstated.** The extraction
  also depends on the **catalog DATA behind the source** —
  `get_rasterdataset(clim_source, ...)` resolves the source `uri` from the catalog
  (`extract_historical_climate.py:91,159`), and the catalog `uri`/underlying staged
  file is **not in the key**. If a catalog `uri` is repointed to updated source
  data (or the staged file changes) while dataset+window stay the same, the key is
  unchanged, Snakemake finds the keyed file present and skips re-extraction, and
  `ancient()` blocks any rebuild cascade — a **silently reused stale extraction**.
  The drift guard does not catch this either: it compares catalog *paths*, never
  hashes catalog *content*.

  **Corrected contract:** a present keyed file is *never stale **for a fixed
  catalog***. Catalog source data is treated as an **explicit assumption** —
  immutable within a project's lifetime — which holds for CST's rapid,
  first-order basin-assessment target but is an assumption, not a guarantee.
  **Escape hatch (documented invalidation rule):** to force re-extraction after a
  catalog change, **manually delete the affected key dir**
  (`climate_historical/<key>/`); Snakemake then re-runs the extraction on next
  invocation. A content-hash staleness check (hashing the resolved source and
  folding it into the key or a sidecar manifest) is **deferred to
  `reproducible-computing`**, consistent with the §3 snapshot-freshness deferral.
  This deferral is tracked in the §12 open-questions table.

### 4c. [v2: risk-5, arch-4] Sub-day window invariant — fail loud, not collide

The key strips time-of-day (`YYYY-MM-DDTHH:MM:SS` → `YYYYMMDD`). Two windows
differing **only** below the day boundary would render to the same key yet call
`get_rasterdataset(time_range=(starttime, endtime))` with different bounds — so the
second experiment would silently reuse the first's extraction against a different
requested window, and its config snapshot (which records the true window) would
disagree with the on-disk data it consumed. For CST's daily forcing this is latent,
not active, but the reuse mechanism has zero guard against it.

**Decision: state the invariant and enforce it — fail loud.** The store contract
is **window endpoints are day-resolution; time-of-day is dropped by contract.**
`slugify_window()` (§4d) **asserts** `HH:MM:SS == 00:00:00` on both endpoints and
raises a clear error otherwise, so a future sub-daily window fails loud at slug
time instead of colliding onto a shared key. (Chosen over hashing the full ISO
instants because day-resolution is the true domain invariant for CST forcing;
a fail-loud assertion documents and enforces it rather than silently tolerating a
sub-day window that has no meaning here.)

### 4d. Key rendering + chirps sidecar co-location

- **Chirps orography sidecar co-location.** `extract_historical_climate.py:143`
  derives the sidecar path from `os.path.dirname(fn_out)`, so moving `fn_out`
  under the keyed dir carries the sidecar under the *same* key dir automatically —
  **no edit to the *producer***. (The *consumer* — the catalog builder — is the
  risk-1 edit in §4a; the two must not be confused.)
- **`slugify_window()` helper.** The window dates are ISO `YYYY-MM-DDTHH:MM:SS`;
  `:` is illegal in Windows paths. `slugify_window()` in `blueearth_cst/shared/`
  strips time-of-day and separators to `YYYYMMDD` (after the §4c day-resolution
  assertion). Deterministic; the experiment's own config snapshot records the exact
  window, so the dir name is a **cache key, not the source of truth**.
- **[v2: arch-8] The `[UNVERIFIED]` "other consumer" is closed.** The only other
  occurrence of the old path is the **dead standalone `__main__` fallback**
  `extract_historical_climate.py:210-217`, which hardcodes
  `examples/my_project/climate_historical/raw_data/extract_historical.nc`. It is
  **unreached under Snakemake** (the script always runs via the `snakemake` global;
  the `else` branch is a dev convenience never hit in the workflow). So no *live*
  consumer hardcodes the path beyond the risk-1 catalog builder (§4a). The
  `[UNVERIFIED]` marker is removed.

---

## 5. Run-dir move mechanics — exact toml/path fields affected

Moving the Wflow run directory from `{basin_dir}/run_climate_{experiment}/` to
`{project_dir}/experiments/<name>/model_runs/` changes the toml's input paths.
The fields are written in `downscale_climate_forcing.py`'s `setup_config` block
(lines 55–76) and the final `config.write` (lines 114–117).

**[v2: risk-7 — RESOLVED against the vendored hydromt_wflow source, not deferred.]**
v1 flagged `[UNVERIFIED: hydromt_wflow config.write may re-relativize absolute
values]` and deferred it to the gate-3 smoke. **Verified now** by reading
`.pixi/envs/default/lib/site-packages/hydromt_wflow/components/config.py` and
`.../components/utils.py` (read-only inspection of the installed package — no
upstream edit, per Hard Constraints):

- `WflowConfigComponent.write` (config.py:124-164) calls
  `make_config_paths_relative(self.data, rel_path)` at line 163 before
  `write_toml`, where `rel_path = write_path.parent / dir_input-fallback` (line
  162; `dir_input` is unset in our config, so `rel_path` is simply the toml's own
  directory).
- `make_config_paths_relative` (utils.py:49-72) recurses every config value and
  calls `_relpath(val, root)`. `_relpath` (utils.py:28-46): **if the value is an
  absolute path on the same mount as `root`, it is converted to a path relative to
  `root`** (`Path(relpath(value, root))`, line 43); a non-absolute value is
  returned unchanged (line 38).

**Consequence — this resolves the §5a decision definitively.** hydromt_wflow
**re-relativizes** any absolute same-mount path in the toml against the toml's own
directory on write. So passing an **absolute** `input.path_static` produces, on
disk, the **correct relative path from the new run-dir location** — because the
relativization root *is* the new toml dir. Absolute-in ⇒ correct-relative-on-disk,
robust to the depth change, computed by hydromt rather than by us. This is *why*
the current code already works (the `../staticmaps.nc` literal is non-absolute so
`_relpath` leaves it alone; the absolute `fn_out.resolve()` used for forcing is
handled the same way). v1's §5a rationale ("stored verbatim / absolute paths do
not hurt portability because the tomls are not committed") was **wrong on the
mechanism** — the stored value is *relative*, so the toml is in fact portable
within `project_dir`. The decision (pass absolute `path_static`) is unchanged; the
*reason* is corrected.

Enumerated, with the exact current value and effect of the move:

| toml field | current value (downscale_climate_forcing.py) | affected? | resolution |
| --- | --- | --- | --- |
| `input.path_static` | `os.path.join("..","staticmaps.nc")` (line 72) | **YES** — walks up from `run_climate_<exp>/` to `hydrology_model/staticmaps.nc`; new run dir is under `experiments/<name>/model_runs/`, a different depth/branch. | Pass an **absolute** path (`Path(model_root)/"staticmaps.nc"` resolved); hydromt re-relativizes it against the new toml dir on write, yielding the correct relative pointer (verified above). |
| `state.path_input` | `os.path.join("..","instate","instates.nc")` (line 70) | **INERT** — `reinit = true` in the template (`config/templates/wflow_sbm.toml:26`), not overridden by wf3, so Wflow cold-starts and never reads input states (verified: `setup_config` does not set `model.reinit`). Path is set but unused. | Rewrite to absolute (to `hydrology_model/instate/instates.nc`) for future warm-state safety; hydromt re-relativizes on write. Low-risk; unread today. |
| `state.path_output` | `f"outstates_{climate_name}.nc"` (line 71) | NO — relative to `dir_output = "."` (line 69), the toml dir. Moves with the toml. | none |
| `input.path_forcing` | `os.path.relpath(fn_out.resolve(), config_out_root)` (line 73) | **YES (computed, auto-adjusts)** — relpath from the toml dir to the forcing nc. Both move under `experiments/<name>/`; relpath recomputed at write time from the rule's new `output.toml`/`output.nc`, so it re-resolves. | No script-logic change beyond the rule output paths; the value string changes (see §6a step 3 — path-aware toml comparator). |
| `output.csv.path` | `f"output_{climate_name}.csv"` (line 74) | NO — relative to `dir_output = "."`. Moves with the toml. | none |
| `dir_output` | `"."` (line 69) | NO — self-relative. | none |
| `time.*`, `output.csv.*` reducers | unaffected (no paths) | NO | none |

The rule-level toml output path (`Snakefile_climate_experiment:227`) and the
forcing nc output path (line 226) are what relocate the files;
`downscale_climate_forcing.py` reads them from `snakemake.output.toml` /
`snakemake.output.nc`. So the **primary edits are Snakefile path expressions**;
the script edits are confined to the two `../`-relative literals
(`input.path_static`, `state.path_input`), each becoming an absolute path.

### 5a. Absolute vs relative for `input.path_static` — decision (mechanism corrected)

**Use an absolute path** for `input.path_static` (resolved from `model_root =
snakemake.params.model_dir`, line 20 / rule line 229). Reasons:

- **hydromt re-relativizes it on write** (verified §5), so the on-disk value is the
  correct relative pointer from the new toml dir — depth-change-robust, and the
  toml stays portable within `project_dir`.
- The script already resolves absolutes for the forcing write (`fn_out.resolve()`,
  line 113) and the config write (`config_root=...resolve()`, line 116), so an
  absolute static path is consistent with the file's existing style.
- `dir_input` is the alternative (set `dir_input` to the model dir, keep
  `path_static` bare) — rejected because `dir_input` also becomes the
  relativization root for `make_config_paths_relative` (config.py:162), which would
  reroute `path_forcing`/`path_output` resolution and entangle the forcing relpath
  that currently works. Rejected to keep the change surgical (§8 A6).

Set `state.path_input` the same way (absolute to
`hydrology_model/instate/instates.nc`), noting it is inert under `reinit=true`.

**[v2: risk-7] The `[UNVERIFIED]` marker is removed** — the mechanism is verified
against the installed package; the gate-3 smoke (§7) is now a *confirmation*, not
the primary check.

---

## 6. Behavior-preservation stance

**Claim.** Scientific values identical; paths move; and — made explicit in v2 —
**run-toml path *pointers* legitimately change string value** (they point at the
same files from new locations). Every computed number (Qstats, basin indicators,
discharge series, extraction netCDF, change factors) is unchanged; the directory
each lands in changes; and the relative path *strings inside the run tomls* move
with the run dir.

**Why by-construction.** No computational path is edited. The edits are (a)
Snakefile output/log/benchmark path expressions (via `exp_dir` redefinition),
(b) two `../`-relative toml literals in `downscale_climate_forcing.py` made
absolute (hydromt re-relativizes them to point at the *same*
`staticmaps.nc`/`instates.nc`), (c) the new consistency-guard rule (a gate,
produces no scientific output), (d) the dataset-keyed extraction path (same
extraction, different output dir), (e) the chirps orography catalog lookup (§4a,
same file, correct path), (f) **[v3: ext1-4]** `experiment_name` validation (a
gate on the config value — rejects invalid names, does not alter a valid run).
Wflow reads the same staticmaps and forcing; the R weathergen reads the same
extraction netCDF; `export_wflow_results.py` reads the same run CSVs.

### 6a. The milestone gate — R3/R5-style value-identical re-record (anchor 4a)

1. Run wf1 (unchanged) then wf3 on the test config into a fresh `project_dir`,
   producing the **new** tree.
2. Compare against the pre-P3-1 reference tree using
   `dev/scripts/semantic_tree_diff.py` with a **path-map layer** (accepted scope
   amendment). `semantic_tree_diff.py` element-wise-compares `.nc` (`compare_nc`),
   parse-compares `.toml` (`compare_toml`), normalize-compares copied configs
   (`compare_copied_config`), and reuses `check_baseline`'s CSV/PNG/discharge
   comparators. **It keys on identical relpaths** (`diff_trees`, lines 344-368:
   `ref_files & cur_files`; `missing`/`extra` are set differences), so a pure move
   registers as MISSING (in ref) + EXTRA (in cur) — *not* a value diff.

   **[v2: repo-6, arch-3] Path-map scope is narrow.** `semantic_tree_diff.py`
   already **excludes** `logs`, `benchmarks`, `.snakemake` from the walk
   (`EXCLUDED_DIR_NAMES`, line 91; `_is_excluded`, line 307), so the wf3
   logs/benchmarks move (§2a) is **invisible to the diff and needs no path-map
   entry**. The path map covers **only content-bearing relocations**: the results
   CSVs (`Qstats.csv`, `basin.csv`), the wf3 config snapshot, the run-dir tomls,
   the run-dir output CSVs, and the keyed extraction netCDF. (Corollary — repo-2:
   because benchmarks are diff-excluded, the wf3 benchmark move **cannot** be
   validated by the tree diff, which is exactly why §7 adds a separate gather
   smoke.)

   **[v3: ext1-3] The path map is a DIRECTORY-PREFIX rewrite, not a per-file
   lookup.** This distinction is load-bearing for the toml comparator (step 3). The
   map is expressed as a small ordered list of **directory-prefix rules** on
   project-root-relative paths — at minimum:
   `climate_<exp>/ → experiments/<name>/`,
   `run_climate_<exp>/ (under hydrology_model/) → experiments/<name>/model_runs/`,
   `config/snake_config_climate_experiment.yml → experiments/<name>/config/…`, and
   `climate_historical/raw_data/ → climate_historical/<dataset_key>/`. Applying a
   prefix rewrite (rather than an enumerated file-by-file table) means the map
   covers **both** the persisted content-bearing files listed above **and any
   in-toml pointer target under a relocated directory — including `temp()` targets
   that exist in neither tree** (the forcing inmaps nc, `Snakefile:226`,
   `temp()`-deleted after `run_wflow`). A per-file map would miss those temp
   targets; the prefix form does not. This is exactly what step 3 needs for
   `path_forcing` (below).

3. **[v2: risk-4] The gate asserts MISSING+EXTRA EMPTY modulo an explicit
   allowlist.** v1 treated the path map as a mechanical translation but never
   required completeness — so an omitted or mis-mapped moved output would degrade
   to a benign-looking MISSING+EXTRA pair (expected during a restructuring
   milestone) that an operator waves through **while the numbers may in fact have
   changed**. That is precisely how a value regression hides inside a move. The
   gate contract is now:

   > After applying the path map (cur→ref relpath translation) and content-diffing
   > every translated pair, the residual MISSING and EXTRA sets must be **EMPTY**,
   > **modulo an explicitly enumerated, justified allowlist**. A nonempty
   > unexplained MISSING/EXTRA is a **gate FAILURE, not a pass**.

   - **Path map = total over the wf3 content-bearing output set.** Every moved
     content file has exactly one old→new mapping; the task-brief specs the map as
     total over that set (the five classes in step 2).
   - **Allowlist location + review rule.** The allowlist lives **beside the path
     map** in the migration note (`dev/milestones/p31/migration_experiment-structure.md`, §4b
     below) as a small, commented list — each entry stating the file, why it is
     MISSING or EXTRA, and that its absence/newness is intended (e.g. the guard
     sentinel `.project_consistency_ok` — and **[v4: ext2-1]** the key-level
     `climate_historical/<key>/.guard_ok` — are EXTRA-by-design; there is no wf3
     plots producer so no plots file is MISSING). **Review rule:** an allowlist entry may
     be added only with a written justification in the migration note and is
     reviewed at the milestone gate; a diff that requires an *undocumented*
     allowlist entry fails the gate.

   **[v2: advisor / risk-7 downstream — the toml path-pointer diff, and where it
   lands.]** Because hydromt re-relativizes (§5), three run-toml path fields
   **legitimately change string value** old→new: `input.path_static`
   (`../staticmaps.nc` → e.g. `../../../hydrology_model/staticmaps.nc`),
   `input.path_forcing` (`../../climate_<exp>/realization_N/...` →
   `../realization_N/...`), and `state.path_input` likewise. `compare_toml`
   (lines 208-215) does a **raw parsed-dict equality with no path-field
   normalization**. Crucially, once the path map pairs an old toml with its new
   counterpart, both are in the **intersection** `ref_files & cur_files`
   (`diff_trees:356`), so `compare_toml` runs and these three fields land in the
   **`failures` list — NOT in MISSING/EXTRA** (which are file-*presence* set
   differences, `diff_trees:352-353`). **The risk-4 allowlist is defined over
   MISSING/EXTRA and therefore cannot exempt a `failures` entry** — so a bare
   "allowlist the three fields" would leave the gate red on every run toml, and the
   §6 clean-diff thesis would be false.

   **[v3: ext1-3] Decision: a path-aware toml comparator that compares targets in a
   PROJECT-ROOT-RELATIVE namespace, then applies the old→new path map — not a raw
   per-toml-dir resolve.** ext1-3 caught that v2's proposed comparator ("resolve
   each pointer against its own toml directory, then compare the normalized paths
   directly") is **broken across roots**: the reference tree and the current tree
   live under **different project roots** (a fresh `project_dir` for the current
   run vs the pre-P3-1 reference), so resolving `input.path_static` against each
   toml's own dir yields `<ref-root>/hydrology_model/staticmaps.nc` vs
   `<cur-root>/hydrology_model/staticmaps.nc` — **unequal absolute paths**, so
   *every* correctly relocated run toml would be reported as a `failures` entry and
   the clean-diff gate could never pass without an undocumented exemption.

   The corrected comparator, in `semantic_tree_diff.py` (ours to edit; `compare_toml`
   is defined *there*), for each path-valued toml field does:

   1. **Resolve** the field value lexically against **its own toml's directory**
      (`normpath(join(toml_dir, field_val))`) — lexical join, not `.resolve()`, so it
      works after the `temp()` forcing nc is deleted.
   2. **Translate to a project-root-relative path** by stripping the side's project
      root prefix: `relpath(resolved, side_project_root)`. Both sides now live in
      the **same** namespace — the path *relative to that tree's `project_dir`* — so
      the different absolute roots are cancelled. (The two project roots are the
      known inputs to `semantic_tree_diff.py`: `ref_root` and `cur_root`.)
   3. **Apply the same old→new path map** (the directory-prefix rewrite of step 2)
      to the *ref* side's project-root-relative target, so a target that
      legitimately *moved* maps ref→cur before comparison. **Which of the three
      fields actually needs this:** `path_static` and `state.path_input` resolve to
      targets under `hydrology_model/` (staticmaps.nc, instate/instates.nc) — a
      directory that **did not move** — so their project-root-relative targets are
      **identical on both sides with no map entry**, and they PASS at step 4
      directly. **`path_forcing` is the only field that moved:** ref target
      `climate_<exp>/realization_N/inmaps_….nc` vs cur target
      `experiments/<name>/realization_N/inmaps_….nc` — the realization dir moved with
      `exp_dir`. The step-2 **prefix rule** `climate_<exp>/ → experiments/<name>/`
      translates the ref target onto the cur target. Crucially, this forcing nc is
      `temp()` (in neither tree), which is exactly why the map must be a **prefix
      rewrite, not a per-file lookup** (step 2) — a per-file map would have no entry
      for a deleted temp target and would leave `path_forcing` untranslated, emitting
      a `failures` entry on every run toml and re-reddening the gate ext1-3 exists to
      keep green.
   4. **Compare the mapped project-root-relative targets.** Equal ⇒ the pointer move
      is behavior-neutral, **PASS**; different ⇒ a real `failures` entry (a
      **mis-repoint is caught, not hidden**).

   This makes the §6 thesis **true rather than asserted** across the cross-root
   milestone comparison, needs **zero** toml allowlist entries, and — unlike a
   blanket field exemption — does not re-introduce risk-4's masking: a future edit
   that made a pointer resolve to a *different* project-relative target fails the
   gate. The MISSING/EXTRA allowlist stays scoped to genuine presence exemptions
   (the `.project_consistency_ok` sentinel and **[v4: ext2-1]** the key-level
   `.guard_ok` EXTRA-by-design; no wf3 plots producer, so nothing MISSING there). This comparator change lands in §10 commit 5 (which
   already owns `semantic_tree_diff.py`).

   **[v3: ext1-3] Comparator tests (commit 5, `test_semantic_tree_diff.py`):**
   - **positive:** a ref toml with `path_static = "../staticmaps.nc"` under
     `<ref-root>/.../run_climate_experiment/` and a cur toml with
     `path_static = "../../../hydrology_model/staticmaps.nc"` under
     `<cur-root>/experiments/exp/model_runs/` — after project-root-relative
     translation + path map, both resolve to project-relative
     `hydrology_model/staticmaps.nc` ⇒ **PASS**.
   - **positive (moved-but-equivalent forcing — the field that needs the map):** ref
     `path_forcing` pointing at `climate_experiment/realization_1/inmaps_....nc`, cur
     pointing at `experiments/exp/realization_1/inmaps_....nc`; the step-2
     **directory-prefix rule** `climate_experiment/ → experiments/exp/` translates the
     ref project-relative target onto the cur one ⇒ **PASS**. (This target is a
     `temp()` file present in neither tree — the test asserts the prefix-rewrite map,
     not a per-file table, is what makes it pass. `path_static`/`state.path_input`
     need no map entry — their targets under `hydrology_model/` did not move.)
   - **negative (mis-repoint):** cur `path_static` pointing at
     `hydrology_model/staticmaps_WRONG.nc` (a different project-relative target) ⇒
     **FAIL** (`failures` entry naming the field).

4. **Re-record** `dev/baseline/manifest.json`'s wf3 slice at the new paths
   (`check_baseline.py record --workflow climate_experiment`), after the
   `check_baseline.py` repoint. **[v2: arch-6] Two distinct repoints:**
   - `EXPERIMENT_NAME` / `resolve()` (lines 65, 120-125) + the two `{exp_dir}`
     results TARGETS (lines 112-113) move together to `experiments/<name>/`;
   - the config-snapshot TARGET (line 114) additionally changes its template root
     from `{project_dir}/config/...` to `{exp_dir}/config/...` (the wf3 snapshot now
     lives under the experiment dir).

     The record is **value-identical** to the old wf3 slice: the re-record's proof
     obligation is that the new `Qstats.csv`/`basin.csv`/config-snapshot
     fingerprints equal the old ones (captured before the move and diffed).

### 6b. Mixed-provenance caveat (from `baseline-manifest-coverage` memory)

The manifest is mixed-provenance: the wf1 slice is post-const-pars-restoration,
the wf2/wf3 rows are pre-restoration. A fresh wf3 regen *may* trip the byte-exact
`Qstats.csv`/`basin.csv` fingerprints if the sub-tolerance wf1 discharge move
(max|dQ|/mean ~ 1.7e-4) survives rounding into wf3 — the **documented residual**,
not a P3-1 regression. If it fires, follow the ADR 0001 step-7 immaterial-branch
path (`check_baseline.py` docstring): re-run wf3, confirm the movement is
consistent with the recorded wf1 diff, re-record with a note. **P3-1's re-record
subsumes** this pending wf3 re-record — the milestone is the natural place to move
the wf3 rows onto the restored model. Stated in `dev/milestones/p31/baseline_diffs.md`.

**Value-identity is only claimable if wf3 re-runs on the restored wf1 model.** If
the reference tree was produced on the pre-restoration model, the diff shows the
immaterial wf1-propagated move, not zero. Gate success criterion:
`semantic_tree_diff` clean **modulo the path map, the MISSING/EXTRA allowlist
(presence exemptions only), the path-aware toml comparator (which passes the three
re-relativized pointer fields when they resolve to the same project-relative
target — §6a step 3), and the documented immaterial wf1 propagation** (discharge
comparator PASS, not byte-zero) — as R5 handled its slice.

---

## 7. Verification plan — honest about dry-run blindness

`--dry-run` validates the DAG (rule wiring, wildcard resolution, no cycles) and
catches path-expression typos that break `input`/`output` matching. It is **blind
to** (the R6 lesson, carried verbatim):

- **`params:` string paths** — `snakemake.params.model_dir`, the toml literals in
  `downscale_climate_forcing.py`, the `gather_benchmarks` `parts_dir` (repo-2), the
  `exp_dir`-derived `output_path`/`exp_dir` params (arch-2).
- **`shell:`/`Rscript`/Julia command bodies** — the R weathergen and Wflow runs.
- **the toml *content*** written by `setup_config` — `--dry-run` never executes the
  script, so a broken `input.path_static` is invisible.
- **the guard script logic** and the **chirps catalog-lookup rewrite** — both run
  only at execution.

`--dry-run` **is** sensitive, though, to the two things v3 leans on: **(1) the
`params` rerun-trigger** (a changed guard `param` or digest shows up in `--dry-run`
as "Params have changed" — this is exactly how the §3c matrix and gate 2b (i–l) are
checked without a real run), and **(2) parse-time config validation** (a bad
`experiment_name` raises before the DAG builds — gate 1 extension, §2b).

R6 caught (post-Gate-1) three bare sibling imports reachable **only** through the
`script:` runtime path — evidence that green `pytest` + clean `--dry-run` is
insufficient and a **full run** must gate the milestone.

**Gate ladder:**

1. **`pytest tests/test_cli.py`** — dry-runs all three Snakefiles; catches DAG
   breakage from the path edits and the new guard rule's sentinel wirings. Cheapest
   gate; run after every Snakefile edit. **[v3: ext1-4]** Extended: a `--dry-run`
   with a scratch config carrying an invalid `experiment_name` (`../evil`) **fails
   at parse** naming the bad value (parse-time validator, §2b).
2. **Unit test for the guard** — `test_check_project_consistency.py`:
   (a) identical project sections ⇒ pass; (b) mutated `shared.basin.resolution`
   ⇒ fail naming the key; (c) mutated `workflows.model_creation` ⇒ fail;
   (d) flat-vs-binned catalog paths ⇒ pass (**symmetric** normalization, §3);
   (e) missing wf1 snapshot ⇒ fail with the "run wf1 first" message;
   (f) mutated `shared.historical_window` ⇒ **pass** (not guarded);
   (g) **[v2: risk-2]** mutated `workflows.climate_projections` **with** a wf2
   snapshot present ⇒ fail; (h) mutated `workflows.climate_projections` with **no**
   wf2 snapshot ⇒ **pass + logged unchecked**.
   (a–h are pure **script unit tests**: call the comparator on staged
   config/snapshot pairs and assert pass/fail — no Snakemake involved.)
2b. **[v3: ext1-1] Sentinel-invalidation INTEGRATION test** — a DAG/rerun-trigger
   check, **NOT** a `test_check_project_consistency.py` script unit test (a script
   unit test would exercise the comparator and prove nothing about rerun-triggers).
   Named `test_guard_invalidation.py` (or a marked integration case) against a staged
   `project_dir`, using **`--dry-run`** to read Snakemake's scheduling reason; the
   reviewer's mutate-each-comparand test, **run WITHOUT deleting the sentinel between
   mutations**:
   (i) guard passes; mutate a guarded **live-config** section; a `--dry-run` reports
   "Params have changed" for the guard rule ⇒ the guard is **scheduled to re-run**
   (and, on a real run, FAILS on the divergence — not skipped);
   (j) mutate the **wf1 snapshot** content ⇒ `--dry-run` shows the guard scheduled
   (wf1 digest param changed) and it FAILS if divergent;
   (k) with a wf2 snapshot present, mutate its content ⇒ guard scheduled (wf2
   digest param changed);
   (l) revert every comparand to original bytes ⇒ `--dry-run` reports **"Nothing to
   be done"** for the guard (content-addressed; no false-fire).
   This is the specific evidence the ext1-1 blocking finding demands.
2c. **[v4: ext2-2] Fresh-project parse regression — missing wf1 snapshot.** In a
   staged `project_dir` with **no** `config/snake_config_model_creation.yml`:
   (i) `snakemake --dry-run` on `Snakefile_climate_experiment` **parses and
   builds the DAG** — no parse-time traceback from digest computation
   (`file_digest_or_absent()` returns `"ABSENT"`) — and reports the guard's
   missing `ancient()` input via the designed rule-level `MissingInputException`
   naming the snapshot;
   (ii) `snakemake --unlock` **succeeds** (lock recovery works with the snapshot
   absent);
   (iii) with the snapshot present, mutating its content still flips the digest
   param (guard scheduled, "Params have changed") — the absence-tolerant helper
   does not weaken the content trigger (overlaps 2b (j); asserted here
   specifically for the helper's present-branch).
3. **Execution smoke — run-dir move** — force-run rules 3.09 + 3.10 for one
   rlz/cst on a staged region config; confirm Wflow *actually reads* the relocated
   `staticmaps.nc` (via the hydromt-re-relativized pointer) and produces a run CSV.
   **[v2: arch-7]** Add a **purity assertion**: diff `hydrology_model/` contents
   pre/post the wf3 smoke and assert it gained no wf3-written files — settling the
   "pure hydrology_model" claim (§11). `downscale_climate_forcing.py:43` opens
   `WflowSbmModel(mode="r+")` on `model_root=basin_dir` and `mod.close()`s (line
   118); the smoke proves the `r+` session flushes nothing back into
   `hydrology_model/`. If it does, switch the open to `mode="r"` or document the
   residual write as a known non-purity.
4. **Execution smoke — dataset-store reuse + guard/shared-cache interaction** — run
   wf3 for experiment A, then B with the *same* `clim_historical` +
   `historical_window`; **[v3: ext1-2 / v4: ext2-1]** confirm B's guard runs
   (B's sentinel written, `K/.guard_ok` rewritten) **but** B's
   `extract_climate_grid` is **SKIPPED** — and, sharper: a B `--dry-run` must show
   **no** "set of input files has changed" reason for `extract_climate_grid`
   (input-set invariance of the key-level guard artifact is exactly what the v4
   fix buys; §3d probe 1 is the failure mode this asserts against). Then C with a
   *different* window; confirm C **re-extracts** to a new key dir, and only
   **after** guard(C) succeeds. Finally the **[v4: ext2-1] alternation smoke**:
   re-invoke A unchanged ⇒ `--dry-run` reports "Nothing to be done" (no params
   thrash from the shared `K/.guard_ok` output — pins the
   experiment-invariant-params constraint, §3b/§3d item 4). Exercises the §4b
   reuse contract *and* the §3d DAG interaction.
5. **Collision smoke** — two experiments in one `project_dir`; assert zero file
   collisions across the **six live** classes (config snapshot, logs, benchmarks,
   run dirs, results, historical-climate reference). **[v2: arch-9]** The **plots**
   class is *reserved* (no wf3 producer today), so it is pre-partitioned by
   construction, not exercised — "six live verified + one reserved," not
   "seven verified."
6. **Full e2e milestone gate** — full wf1→wf3 run into a fresh `project_dir`;
   `semantic_tree_diff.py` (with the path map + allowlist + the **project-root-
   relative path-aware toml comparator**, §6/ext1-3) clean modulo documented
   residuals; `check_baseline check --workflow climate_experiment` green after
   re-record; `pytest tests/` green.
7. **[v2: repo-2] Benchmark-gather smoke** — after a wf3 run, assert
   `experiments/<name>/benchmarks/wf3_benchmarks.md` is **non-empty and contains
   the expected per-rule rows** (the gather found its parts under the moved
   `parts_dir`). Catches the silent empty-report failure the tree diff cannot see
   (benchmarks are diff-excluded).
8. **[v2: risk-1 — BLOCKING closure] Non-era5 chirps-path gate.** Run rule 3.08 on
   a chirps/chirps_global staged config for one rlz/cst; assert the emitted
   `data_catalog_climate_experiment.yml` `<src>_orography.uri` resolves to an
   existing file under the keyed dir, and rule 3.09 consumes it without a
   missing-source error. Where a full chirps stage is unavailable in CI, the
   **mandatory minimum** is the §4a unit test on the rewritten lookup. This closes
   the "every gate runs era5" escape class that hid risk-1.
9. **[v3: ext1-4] Validation unit test** — `test_validate_experiment_name.py`, the
   §2b matrix (traversal, separators, absolute forms, Windows-reserved names,
   empty, hyphen/dot rejection, length cap, containment assertion). Cheap; runs
   with gate 1.

Gates 3–5, 7, 8 are the ones a real run must cover; **all are mandatory before the
milestone tag**, per the R6 precedent. Gate 2b (i–l) is the specific sentinel-
invalidation integration test the blocking finding demands; gate 8 is the one that would have
caught risk-1; gates 1(ext)/9 cover ext1-4; **[v4]** gate 4 pins ext1-2 **and
ext2-1** (input-set invariance + alternation smoke); gate 2c pins ext2-2
(missing-snapshot `--dry-run`/`--unlock`); the §2b matrix (gate 9) pins ext2-3
(uppercase rejection).

---

## 8. Alternatives considered

**A1 — Layered project + experiment configs (REJECTED — anchor 3).** Split config
into a project base + a per-experiment overlay merged at load. *Why not:* breaks
the `workflow.configfiles[0]` forwarding contract that hands a single config path
to the R weathergen (`Snakefile_climate_experiment:17`, forwarded as
`config_path`/`snake_config` at rule params 130, 149). A merged config has no
single file to forward. *When it would win:* if per-experiment duplication becomes
a real maintenance burden; the intake defers this ("revisit only if duplication
proves painful"). The drift guard is the cheaper substitute — it lets configs stay
full-and-independent while catching project-section divergence.

**A2 — Keep run dirs under `hydrology_model/` (REJECTED tree shape).** *Why not:*
leaves wf1's "pure built-model" dir polluted with per-experiment run debris; blocks
P3-2 (needs a clean model-neutral `hydrology_model/`); keeps the experiment's Wflow
artifacts *outside* `experiments/<name>/`, defeating "reproducible from its own
dir." *When it would win:* if the toml-relative-path rewrite were high-risk — but
§5 shows it is two literals plus a rule path, and hydromt's re-relativization makes
the absolute-path resolution robust. Pollution and P3-2 costs outweigh the rewrite.

**A3 — Dataset store keyed by dataset only, forbid window-divergent reuse
(REJECTED keying).** *Why not:* anchor 1 allows the window to vary per experiment;
forbidding it constrains a confirmed-in-scope capability, and overwriting silently
invalidates the first experiment's extraction. Rejected for dataset+window keying.

**A4 — Dataset store keyed by dataset only, full-period store + downstream subset
(REJECTED keying variant).** *Why not:* the extraction *already* subsets at read
time (`get_rasterdataset(time_range=...)`, lines 91/159); there is no full-period
artifact without changing the extraction to fetch-all-then-store, and the subset
step would land in the R weathergen layer, which anchor 5 keeps as-is. *When it
would win:* a project with dozens of windows on one dataset where storage
dominates — not the first-order target. Rejected (§4).

**A5 — Guard at parse time via a config assertion (REJECTED guard placement).**
*Why not:* fires on `--dry-run` and `--unlock`, breaking lock recovery, and
produces a parse-time traceback instead of a clean rule failure. *When it would
win:* if the guard needed to *prevent DAG construction* — it does not; it blocks
execution. Rejected for a gate rule (§3). **[v3] Note the contrast with `experiment_
name` validation (§2b), which IS parse-time** — because that is *config* validation
(a malformed value makes the whole DAG ill-defined) rather than a *project-state*
check, and failing it under `--dry-run` is correct, not hostile.

**A6 — `dir_input` instead of absolute `path_static` (REJECTED toml mechanic).**
*Why not:* `dir_input` becomes the relativization root for
`make_config_paths_relative` (config.py:162 — verified §5), so setting it would
reroute `path_forcing`/`path_output` resolution and entangle the forcing relpath
that currently resolves correctly. *When it would win:* if *all* inputs lived under
one dir — they do not (static under `hydrology_model/`, forcing under
`experiments/<name>/realization_*/`). Rejected for a targeted absolute
`path_static` (§5a).

**A7 — [v2] Gate only the model-run rules with the guard sentinel (REJECTED guard
wiring).** Thread the sentinel into `downscale_climate_realization` + `run_wflow`
only, not the roots. *Why not:* the extraction, parameter grid, and weagen configs
would then be produced against an unverified project (they descend from the roots,
not from the model-run rules), so a divergent project silently generates those
artifacts before the guard would fire. Same edit count as wiring the roots (§3a);
rejected because it under-gates.

**A8 — [v3: ext1-1] Always-recompute / disposable sentinel via `temp()` or a
`touch`-always rule (CONSIDERED, not needed).** The reviewer offered "make the guard
always-recomputed or disposable." *Why not adopted as the primary mechanism:* the
verified `params`+digest triggering (§3c) already re-runs the guard on **any**
comparand change while correctly **skipping** when nothing changed — cheaper than an
unconditional re-run every invocation, and it preserves the sentinel as a provenance
record (a `temp()` sentinel would be deleted, losing the "this experiment's guard
passed" artifact). The digest approach is *equivalent in safety* (every comparand
change triggers) but strictly better on cost and provenance. Recorded as the
considered alternative; the digest mechanism wins.

**A9 — [v3: ext1-2] Achieve guard-before-extraction ordering WITHOUT a sentinel
input on the shared rule (CONSIDERED).** E.g. a Snakemake `priority`/`localrules`
ordering, or making extraction depend on the guard only through a per-experiment
symlink. *Why not:* `ancient()` on the sentinel input (the chosen fix, §3d) is the
idiomatic Snakemake lever for "depend for ordering, ignore for freshness," is
verified by the probe, and needs no new mechanism. The per-experiment sentinel +
`ancient()` combination is minimal and proven. Rejected in favor of the verified
`ancient()` approach. **[v4: ext2-1]** v3's specific instantiation (the shared
rule consuming the *per-experiment* sentinel `ancient()`) was itself refuted by
the r2 probe — the input-set trigger fires on the path swap (§3d probe 1). The
surviving mechanism is the **key-level guard artifact** of §3d: still an
`ancient()` dependency edge (so this alternative's rejection rationale — no new
ordering mechanism needed — stands), but on an experiment-invariant path.

---

## 9. Naming decisions

Per intake, minor renames are open to this design. Proposals:

| Slot | Proposed | Cosmetic-open? | Rationale |
| --- | --- | --- | --- |
| experiment root | `experiments/<name>/` | cosmetic-open (vs `runs/`) | "experiment" matches `experiment_name` and CST vocabulary; `runs/` collides with Wflow "runs". Keep `experiments/`. |
| Wflow run dir | `model_runs/` | cosmetic-open | Intake's proposal; model-neutral for P3-2. Holds tomls + per-rlz/cst outputs. Keep. |
| results dir | `model_results/` | fixed (already exists) | Unchanged from current `exp_dir/model_results/`. |
| dataset key | `<clim_source>_<start>_<end>/` | cosmetic-open (separator/format) | §4; date form/separator open, dataset+window *content* fixed. |
| guard sentinel | `.project_consistency_ok` | cosmetic-open | Dotfile; per-experiment (§3a). Name open. |
| key-level guard artifact | `climate_historical/<key>/.guard_ok` | cosmetic-open (name only) | **[v4: ext2-1]** §3b/§3d; second guard output; experiment-invariant per key, consumed `ancient()` by `extract_climate_grid` only. The NAME is open; the key-dir placement is load-bearing (input-set invariance) and fixed. |
| experiment_name grammar | `^[a-z0-9][a-z0-9_]*$`, ≤64 | **fixed (ext1-4 safety)** | §2b; the grammar is a safety contract, not cosmetic — loosening it reopens the containment argument. |

None of these names except the `experiment_name` grammar are load-bearing on an
external contract (anchor 4b), so the rest are safe to bikeshed in review. The
grammar is fixed by the ext1-4 containment requirement.

---

## 10. Commit plan (staged, each independently runnable/verifiable)

Per the R6 house pattern (each commit leaves the tree runnable). **[v2: repo-3]**
Prefix `p31:` per the roadmap's milestone-matching prefix pattern (`m*`,
`r01..r06`, `chore:` are registered at `dev/roadmap.md:689-697`; **`p31:` is not
yet registered** — commit 7 registers it, so "per roadmap commit strategy" is not
overstated).

1. **`p31: add project-consistency drift guard (rule + root wiring + digest triggers + tests)`**
   — the `check_project_consistency` rule, its script, gate-2 script unit tests
   (a–h) **and the gate-2b sentinel-invalidation integration test (i–l)**, **the
   root-rule guard wirings** (§3a: per-experiment sentinel fresh on the four
   per-experiment roots; **[v4: ext2-1]** the key-level `K/.guard_ok` second
   output consumed `ancient()` by `extract_climate_grid`), **and the
   params-carried snapshot digests** (wf1 + wf2) plus the guarded-live-section
   digest param (ext1-1) — **[v4: ext2-1]** with `config_path` removed from the
   guard params (experiment-invariance constraint, §3b) and **[v4: ext2-2]** the
   `file_digest_or_absent()` helper in `snake_utils.py` + the gate-2c
   missing-snapshot regression checks. **[v2:
   arch-1, repo-1]** This commit *does* touch existing rule `input:` blocks — the v1
   "additive / existing paths unchanged" claim is **dropped**. Guard's wf1 snapshot
   input is `ancient()` (mtime-neutral); content-change triggering is via the
   digest param. **[v4: ext2-1] Sequencing note:** the store is not yet keyed in
   this commit (that is commit 4), so the guard artifact lands at the
   then-current store dir — `climate_historical/raw_data/.guard_ok` (trivially
   experiment-invariant, the store is un-keyed); commit 4 moves it under the
   keyed dir together with the store. Verify: `pytest tests/test_cli.py` + guard
   unit tests (2 a–h) + the invalidation integration test (2b i–l) + the
   fresh-project parse regression (2c).
2. **`p31: validate experiment_name + move wf3 project-global outputs under experiments/<name>/`**
   — **[v3: ext1-4]** add `validate_experiment_name()` to
   `blueearth_cst/shared/snake_utils.py` and call it at the Snakefile parse site
   (before `exp_dir`), with `test_validate_experiment_name.py` (§2b matrix);
   **then** redefine `exp_dir` (line 62, §2) so all `exp_dir`-derived output/params
   paths (incl. 132, 151, 265) move; edit the `project_dir`-rooted wf3
   log/benchmark/config paths (§2a) **including the `gather_benchmarks` `parts_dir`
   param + output** (repo-2). Verify: `--dry-run` (incl. the invalid-name parse
   fail) + `pytest tests/test_cli.py` + gather smoke (gate 7) + validation tests
   (gate 9).
3. **`p31: move Wflow run dir to experiments/<name>/model_runs + rewrite toml paths`**
   — run-dir relocation (Snakefile 227/245/247/248/260) **and** the
   `downscale_climate_forcing.py` toml-literal rewrite (§5, two literals →
   absolute) in **one commit** (interdependent). Verify: `--dry-run` + run-dir
   execution + purity smoke (gate 3).
4. **`p31: key climate_historical store by dataset+window + fix chirps orography lookup`**
   — the extraction output path + the `generate_weather_realization` consumer input
   to the keyed dir; the `slugify_window` helper (with the §4c day-resolution
   assertion); **[v2: risk-1] and the `prepare_climate_data_catalog.py:76-84`
   orography-lookup rewrite** (derive from the passed key param, correct depth);
   **[v4: ext2-1]** also moves the guard artifact from
   `climate_historical/raw_data/.guard_ok` to the keyed dir
   (`climate_historical/<key>/.guard_ok`) together with the store (see commit 1's
   sequencing note).
   Verify: dataset-store reuse + guard-interaction smoke (gate 4, ext1-2/ext2-1,
   incl. the alternation smoke) **and the chirps-path gate 8**.
5. **`p31: repoint baseline manifest + semantic_tree_diff path map/allowlist/toml-comparator to new tree`**
   — update `check_baseline.py` (`EXPERIMENT_NAME`, `resolve`, the two `{exp_dir}`
   results TARGETS, **and the config-snapshot TARGET template root** — arch-6); add
   the `semantic_tree_diff.py` path-map layer + the MISSING/EXTRA-empty allowlist
   assertion (§6, risk-4) **and the project-root-relative path-aware toml
   comparator** (§6a step 3, ext1-3) with its positive/negative tests. Verify:
   self-diff smoke clean + comparator unit tests.
6. **`p31: re-record wf3 baseline slice (value-identical) + baseline_diffs`** — the
   value-identical re-record after a full wf3 regen; `dev/milestones/p31/baseline_diffs.md`
   documenting the residual (§6b). Verify: milestone e2e gate (gate 6).
7. **`p31: migration note + register p31 prefix + docs`** — **[v2: repo-4]** land
   the old→new output-path map + the risk-4 allowlist at
   **`dev/milestones/p31/migration_experiment-structure.md`** per naming.md §7 (**not** the
   root `MIGRATION.md`, which is R06-scoped and git-ref-anchored to `e33ee45`);
   **[v2: repo-3]** register the `p31:` prefix in `dev/roadmap.md`'s commit-prefix
   list; update any README/CLAUDE reference to the wf3 output layout. Docs-only.

Commits 1–4 are code; 5–6 are the gate; 7 is docs. Each of 1–4 leaves all three
workflows runnable.

---

## 11. Consequences

**Positive.**
- Two experiments coexist in one `project_dir` with zero collisions across the six
  live classes (success criterion 1) — verified by gate 5; plots reserved (arch-9).
- A shared dataset+window extraction is reused, not copied (criterion 2) — gate 4;
  **[v3: ext1-2 / v4: ext2-1]** the guard does not defeat the reuse: the shared
  rule consumes the experiment-invariant key-level `K/.guard_ok` `ancient()` (the
  v3 per-experiment-sentinel shape was probe-refuted on the input-set trigger and
  replaced, §3d).
- Each `experiments/<name>/` dir is self-contained and reproducible from its own
  config snapshot (criterion 3).
- `hydrology_model/` is a pure wf1 artifact again — unblocks P3-2's model-swap
  interface. **[v2: arch-7]** The purity claim is *asserted at gate 3*.
- The drift guard fails loud on a mutated project section and a missing wf1
  snapshot (criterion 4) — gate 2 — **[v3: ext1-1]** and now on **every** comparand
  change (live config, wf1 snapshot, wf2 snapshot) via the verified `params`+digest
  triggering, not merely on a fresh run; the persistent sentinel cannot be reused
  after a comparand changes (gate 2b, i–l).
- **[v3: ext1-4]** A malformed or adversarial `experiment_name` fails loud at parse
  (§2b), so the isolation/zero-collision guarantee is not silently violated.

**Negative.**
- The wf3 baseline manifest slice must be re-recorded (paths change), and
  `check_baseline.py`'s `EXPERIMENT_NAME`/`resolve`/three TARGETS repointed —
  a manifest+dev-script edit, not just a value check (§1a C2, arch-6).
- `semantic_tree_diff.py` needs a path-map layer, a MISSING/EXTRA-empty allowlist
  assertion (§6, risk-4), **and the project-root-relative path-aware toml
  comparator** (§6a step 3, ext1-3) — dev-script additions.
- **[v2: risk-7 corrected]** `input.path_static` is passed as absolute but hydromt
  **re-relativizes** it on write, so the on-disk toml value is *relative* and the
  toml **is portable** within `project_dir`.
- **[v2: advisor/risk-7 downstream / v3: ext1-3]** Three run-toml path *pointer*
  fields change string value (they point at the same files); handled by the
  project-root-relative path-aware toml comparator (§6a step 3) — catches a
  mis-repoint, no allowlist exemption.
- **[v2: risk-3]** Store reuse assumes catalog source data is immutable within a
  project; a repointed catalog `uri` on an existing key is a silent-stale risk with
  a documented manual key-dir-deletion escape hatch (content-hashing deferred).
- A user who edits project sections without re-running wf1 hits a hard guard
  failure (intended, a new failure mode to document).
- **[v3: ext1-1]** The guard rule carries content digests of the wf1/wf2 snapshots
  as params — a small, deterministic parse-time hash cost per wf3 invocation
  (SHA-256 of two small YAML files), negligible. **[v4: ext2-2]** Computed
  absence-tolerantly (`"ABSENT"` for a missing file), so a fresh project still
  parses, `--dry-run`s, and `--unlock`s.
- **[v4: ext2-1]** A failing guard job may remove a pre-existing keyed
  `.guard_ok` (Snakemake failed-job output cleanup); recovery is automatic on the
  next passing invocation, and the keyed extraction still skips (§3d
  failure-cleanup caveat).

**Neutral (must be planned for).**
- The migration note lands at `dev/milestones/p31/migration_experiment-structure.md` (repo-4).
- The mixed-provenance manifest state is resolved for the wf3 rows at this
  milestone (moved onto the restored wf1 model) — a provenance cleanup folded into
  P3-1's re-record.
- wf1/wf2 `logs`/`benchmarks`/`plots`/`config` stay project-level; only wf3's move.
  The asymmetry is deliberate and documented (§2a).
- **[v3: ext1-4]** `experiment_name` is now validated at parse; existing valid
  configs (e.g. the seed `experiment`) pass unchanged — no migration for
  well-named experiments.

---

## 12. Open questions carried to review

| # | Question | Status |
| --- | --- | --- |
| 1 | Path-map/allowlist/comparator *interface* in `semantic_tree_diff.py` (a `--path-map` + `--allowlist` option + the project-root inputs vs a wrapper). In-scope (dev/scripts is ours, accepted amendment); exact interface is the task-brief's call. | open (mechanism, not decision) |
| 2 | Dataset-key rendering format (`era5_20000101_20201231` vs shorter) — cosmetic. | open (cosmetic) |
| 3 | **Content-hash staleness for the store key (risk-3).** Deferred to `reproducible-computing`; tracked here and in a `dev/followups.md` entry the task-brief adds. Manual key-dir deletion is the interim invalidation rule. | **deferred — tracked (this row + dev/followups.md)** |
| 4 | Guard section-set completeness — RESOLVED in §3: grep confirms wf3 consumes only `clim_historical`, `historical_window` (unguarded, experiment-variable) and `shared.basin` (guarded) from `shared`; no fourth key. | closed |
| 5 | **[v3] `experiment_name` grammar strictness (ext1-4).** The `^[a-z0-9][a-z0-9_]*$`/≤64 grammar is the safety floor; whether to allow a wider portable set (e.g. hyphens) is a cosmetic loosening that must re-run the containment argument. Kept strict for P3-1. | open (cosmetic; strict default fixed) |

**No `[UNVERIFIED]` markers remain in v4.** v2 closed v1's three; v3's
rerun-trigger matrix (§3c) stands verified. **[v4: ext2-1]** v3's `ancient()`
shared-cache claim (old §3d) was **REFUTED** by the r2 probe on the input-set
provenance trigger and is replaced by the key-level guard-artifact mechanism,
itself verified by three probes this session (verbatim reasons in §3d).
**[v4: ext2-2/-3]** The absence-tolerant digest helper and the fixture/case-policy
fix introduce no new mechanism claims. No unverified assertion is introduced.

---

## 13. Related

- `dev/roadmap.md` § "P3-1" (goal, cut/deferred, tag `p31-experiments`) and
  § "P3-2" (the model-swap direction this unblocks); commit-prefix list
  (`:689-697`, gains `p31:` in commit 7).
- `dev/milestones/r06/structural-refactor-design.md` — the house rigor pattern and the
  `COPIED_CONFIG_PATH_MAP` / `semantic_tree_diff.py` machinery reused here.
- `dev/milestones/r05/` — the wf3 cleanup this builds on; `historical_window` wiring and
  `stress_test_grid` (`Snakefile_climate_experiment:43`).
- `dev/followups.md` §R3 — R5's independent empirical confirmation that Snakemake
  9.6.2's default `params` rerun-trigger re-runs `extract_climate_grid` on a
  window edit ("Params have changed"); corroborates §3c case (a).
- `dev/scripts/check_baseline.py` (ADR 0001, mixed-provenance manifest) and
  `dev/scripts/semantic_tree_diff.py` — the baseline gates.
- `dev/milestones/p31/migration_experiment-structure.md` — the P3-1 old→new output-path map +
  allowlist (repo-4; naming.md §7).
- `dev/conventions/naming.md` §7 — migration-note placement; and the snake_case
  identifier rule the §2b `experiment_name` grammar subsets.
- `.pixi/envs/default/lib/site-packages/hydromt_wflow/components/config.py` +
  `.../components/utils.py` — the `config.write` re-relativization behavior verified
  for §5/risk-7 (read-only; consumed verbatim, not re-engineered).
- `AGENTS.md` Hard Constraints — CST automation scope: this design consumes
  hydromt/Wflow behavior verbatim and edits only our `Snakefile`/`blueearth_cst/`/
  `dev/scripts/` surfaces.
