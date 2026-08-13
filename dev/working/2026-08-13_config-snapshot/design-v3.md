# Design v3 — What a CST project records about the configuration it ran under

Status: FINAL. Review rounds are closed (cap of 2 reached, arbitration invoked
by the owner); this version resolves every surviving round-2 finding and is the
implementation source. Nobody reviews this before it becomes a task brief.
Repository: `blueearth_cst` (BlueEarth Climate Stress Test).
Author: Claude, 2026-08-13. Supersedes v2.

**Changes in v3 resolve the round-2 findings from two independent external
reviewers (GPT `gpt-5.6-sol`, Fable), both `revise`: one blocking defect found
independently by both (the record writer cannot see toolbox-identity changes),
three shared major clusters (values-used record incomplete, digest semantics,
journal/embedding mechanics), and the production no-git gap. Where the two
reviewers disagreed (`gpt-2`), the driver's premise verification settled it in
GPT's favour and v3 treats the values-used record as previously UNRESOLVED.
Sections 1–4 are unchanged from v2 except where a finding corrected a fact.**

---

## 1. Context you need to read this

`blueearth_cst` is a multi-language (Python + R + Julia) scientific workflow
toolbox stitched together by Snakemake. Three `Snakefile_*` files at the repo
root are the only entry points:

- **WF1 `model_creation`** — builds a distributed Wflow-SBM hydrological model
  from global datasets via hydromt, runs it once on historical forcing.
- **WF2 `climate_projections`** — computes monthly CMIP6 change factors.
- **WF3 `climate_experiment`** — the stress test: a stochastic weather
  generator produces realizations, each perturbed across a temperature ×
  precipitation grid, run through Wflow, reduced to hydrological indicators.

All three parse **one** `--configfile` YAML with a sectioned schema:

```yaml
project:    {project_dir, static_dir, data_sources, data_sources_climate}
shared:     {basin, historical_window, clim_historical}
workflows:
  model_creation:      {...}
  climate_projections: {...}
  climate_experiment:  {...}
```

Outputs land under `project_dir`, which in production lives **outside** the
repository tree. A separate wrapper, `scripts/run_workflows.py`, can invoke the
enabled workflows in order; running `snakemake` directly on a Snakefile is
equally supported and is how both reference fixtures were built. Production
also runs these Snakefiles server-side from a Docker image whose build `ADD`s
the sources **without `.git`** (`Dockerfile:20-23`) — a fact §5.2 and §5.5 must
survive.

The toolbox ships tracked, versioned inputs under `config/`:
`config/catalogs/*.yml` (hydromt data catalogs), `config/defaults/*.yml` (files
a rule reads), `config/templates/*` (scaffolds a user copies).

This design concerns **what each project directory records about the
configuration its runs executed under** — nothing about what the workflows
compute.

---

## 2. What exists today

Rule `X.01 snapshot_config` in each Snakefile (script:
`blueearth_cst/model/copy_config_files.py`) writes **two tiers**.

**Tier A — mutable "current copies"**, overwritten every run:
`<project_dir>/config/runs/snake_config_<workflow>.yml` (WF1, WF2),
`<exp_dir>/config/snake_config_climate_experiment.yml` (WF3), plus verbatim
copies of every referenced catalog, build template and observation file into
`config/{catalogs,templates,observations}/`.

**Tier B — immutable content-addressed bundles**:

```
<project_dir>/config/runs/<workflow>/<12-hex digest>/
    source.yml  effective.yml  referenced-files.json  files/<kind>/<hash>-<name>
```

Directory named by `short_digest(snapshot_bundle_digest(...))` — a SHA-256 over
the effective config, the source config's bytes, and the sha256 of every
referenced input. New directory whenever any of those change; old ones never
removed. WF3's equivalent is rooted at the experiment
(`<exp_dir>/config/runs/climate_experiment/<hex>/`).

Rule X.01's rerun triggers today (`Snakefile_model_creation:420-443`): inputs
are the configfile, the build/waterbodies templates and the two optional
observation files; params are `data_catalogs` (paths), `workflow_name`,
`config_dir`, `effective_config` (the whole parsed config) and
`advanced_settings`. **Toolbox identity appears nowhere in them.**

The current digest (`provenance.py:137-141`) is computed over
`{"schema_version": 1, "project_config": <whole config>, "advanced_settings":
<resolved advanced settings>}` — note it **does** include `advanced_settings`;
v2 claimed otherwise and that claim was false (driver-verified).

**Measured state of the reference fixture (`test_case/test_local`):** 3 + 3 + 1
bundles, `config/runs/` = 1.6 MB; the two flat WF1/WF2 copies byte-identical to
each other; the project's catalog and template copies byte-identical to the
tracked repo originals; bundles stale, recording a `source_config.source` path
that no longer exists in the repo.

---

## 3. Problems identified

- **P1** — Per-workflow naming promises a projection the content does not
  deliver. `snake_config_model_creation.yml` is a verbatim copy of the whole
  configfile. `provenance.py:137` returns `{"project_config": config, ...}` with
  `config` the entire parsed configfile.
- **P2** — The bundle digest is a function of configuration the workflow never
  reads: editing a WF3-only key changes WF1's bundle digest and mints an orphan
  directory. **Bounded**: the snapshot outputs are terminal in the DAG (nothing
  takes them as a rule input), and the WF3 drift guard compares named config
  sections, not digests. Identity imprecision plus accumulation, not a
  correctness or performance defect.
- **P3** — Nothing reads the bundle. One writer, no runtime reader anywhere.
  The only cross-workflow snapshot consumer, WF3's drift guard, reads the flat
  copies (`Snakefile_climate_experiment:376-377`).
- **P4** — Copied inputs duplicate tracked repository files. For templates this
  is stronger: `prepare_build_config.py`, which once merged `region`/`resolution`
  into `setup_basemaps`, is orphaned — `build_wflow_model.py:59` now *rejects*
  `setup_basemaps`, and no Snakefile wires the module. Rule 1.07 reads the
  template verbatim.
- **P5** — Layout inconsistency: WF3's flat copy sits one level shallower than
  WF1/WF2's, outside its own `runs/`.
- **P6** — The toolbox revision is recorded only by `run_workflows.py:375`, into
  `config/runs/invocations/*.json`. A direct `snakemake` invocation records
  nothing, and neither reference fixture has an `invocations/` directory.
- **P7** — The retained records are **current-only in the sense of most recent
  *attempt***, not "the run that produced the outputs on disk". Rule X.01's
  outputs are terminal in the DAG, and WF2 is documented to run under
  `--keep-going` where partial completion is normal. A config edit followed by
  a failed run overwrites the record while the tree still holds outputs from
  the previous config.
- **P8** — `copy_config_files.py:81` copies referenced files using
  `source_path.name` only. Two configured observation files with the same
  basename overwrite one another. The bundle avoided this with hash-prefixed
  archive names (`:144`).
- **P9** — Neither `effective_config_sha256` nor anything else that survives
  tier B's removal covers the **contents of referenced inputs**. A template or
  catalog can change in place with its configured path unchanged.
- **P10 (new, from round 2 — both reviewers, blocking)** — The record writer
  cannot see the toolbox move. Rule X.01's rerun triggers (§2) contain no
  toolbox identity, so the single most common provenance scenario — upgrade the
  toolbox, rerun an existing project — leaves `run_record.yml` stamped with the
  **previous** commit while the science rules regenerate outputs, and no
  journal line marks the invocation. A record that answers "which commit
  produced this" with the wrong commit is worse than the absence it replaces.
- **P11 (new, from round 2)** — Production runs have no git checkout
  (`Dockerfile` carries no `.git`; `run_workflows._git_metadata()` already
  returns `{"commit": None, "dirty": None}` on failure, verified `:424-433`).
  Every mechanism keyed on `git rev-parse` silently degrades exactly in the
  environment whose outputs are quoted in reports.

Where each is resolved: P1 → §7 item 4 (left standing, with reasons);
P2/P3/P4 → §5.1 + §5.5; P5 → §7 item 3; P6/P10 → §5.2; P7 → §5.7 + §5.8;
P8 → §5.5; P9 → §5.4; P11 → §5.2 + §5.5.

---

## 4. Owner requirements (settled — do not re-litigate)

- **R1.** No immutable bundle for WF1/WF2. Only the configuration of the current
  / most recent execution.
- **R2.** WF3 is different: each experiment keeps its own configs, because each
  experiment's setup differs. The experiment directory is the partition.
- **R3.** No build-template copies in the project. "I can always refer to the
  toolbox repository for templates." What is needed instead is **the actual
  values used** for that execution.
- **R4.** General principle: **copy a referenced file into the project only when
  the toolbox repository cannot give it back.**
- **R5 (ruled 2026-08-13; narrowed 2026-08-13 after the P0 probe).** An
  append-only, one-line-per-run journal is **permitted** — R1 forbids the
  content-addressed bundle, not a bounded ledger. **Scope:** the journal records
  invocations **in which at least one job executed or was attempted**, not every
  command typed. The pinned Snakemake fires no lifecycle handler on a "Nothing
  to be done" no-op (`p0-probe-result.md`), and the owner ruled against adding a
  mechanism to reach past it — R5 permitted a ledger, it never mandated
  universal coverage, and params threading already ensures any invocation whose
  configuration, code, environment or referenced inputs moved does execute.
- **R6 (ruled 2026-08-13).** The values-used record is written by the rule
  that consumes the values, into the model's own output directory — not into a
  config snapshot bin.
- **R7 (ruled 2026-08-13).** The toolbox-revision stamp is folded into this
  design rather than deferred.

---

## 5. Proposed solution

### 5.1 Remove tier B entirely

Delete the content-addressed bundle from all three workflows: no
`<workflow>/<digest>/` directory, no `source.yml`, no `referenced-files.json`,
no `files/`. Justification: zero readers (P3); `source.yml` duplicates the
retained flat copy; `files/` is superseded by the R4 policy in §5.5; the digest
is imprecise (P2).

### 5.2 One consolidated `run_record.yml` per workflow — and how it stays fresh

Rule `X.01` writes one file per workflow, atomically (temp file + `os.replace`,
the `run_workflows._write_json_atomic` pattern):

```yaml
schema_version: 2
workflow: model_creation
toolbox:
  commit: <40-hex sha> | null       # null ⇒ identity unavailable (see below)
  commit_source: git | baked | null # HOW the commit was learned
  dirty: true | false | null        # true ⇒ commit does NOT fully identify the
                                    # code; null ⇒ unknowable (no git)
environment:                        # lock-file identity; survives absence of git
  pixi_lock_sha256: <sha> | null
  manifest_toml_sha256: <sha> | null
source_config:
  path: <path as invoked>
  sha256: <sha over the configfile bytes>
effective_config:                   # the consumed-key projection, §5.3
  project: {...}
  shared: {...}
  workflows: {...}
advanced_settings: {...}            # julia_version, min_historical_years, ...
effective_config_sha256: <configuration identity — §5.4>
configuration_inputs_sha256: <configuration + toolbox + environment +
                              referenced-input identity — §5.4>
referenced_inputs:                  # one entry per consumed external file, §5.5
  - role: observations_timeseries
    origin: <path as configured>
    recoverable: false              # the §5.5 predicate's verdict
    archived_path: config/observations/observations_timeseries.csv  # or null
    git_blob: null                  # blob id when recoverable, else null
    sha256: ...
    size_bytes: ...
```

Rationale for retaining a record at all: `advanced_settings`
(`julia_version: 1.11.7`, `min_historical_years: 16`, `julia_threads: 4`) is
recorded nowhere else in the project tree; the source configfile does not
contain it.

**`toolbox.version` is DROPPED** (round-2 finding, driver-verified): the repo
deliberately has no `[project]` table in `pyproject.toml` and no `__version__`
anywhere, so the field had no derivation and every implementer would have
invented one. Code identity is `commit` + `dirty` + the `environment` hashes.
If a real version source ever exists, it enters at the next `schema_version`
bump.

**Toolbox identity resolution** — one shared helper,
`toolbox_identity() -> {"commit", "commit_source", "dirty"}` in
`blueearth_cst/shared/provenance.py`, replacing the private
`run_workflows._git_metadata` (which becomes an import, so there is exactly one
definition). Resolution order:

1. `git rev-parse HEAD` + `git status --porcelain`, cwd = the toolbox repo
   root, 5 s timeout, all failures swallowed (the existing
   `_run_metadata_command` behaviour). Success ⇒ `commit_source: git`, `dirty`
   from porcelain output.
2. Else read one line from `<repo_root>/.toolbox-commit` if present ⇒
   `commit_source: baked`, `dirty: null`. The `Dockerfile` gains
   `ARG TOOLBOX_COMMIT` and writes it to that file at image build (two lines;
   the build command passes `--build-arg TOOLBOX_COMMIT=$(git rev-parse HEAD)`).
   The file is gitignored and absent in a normal checkout.
3. Else `commit: null, commit_source: null, dirty: null`.

`_environment_file_hashes` likewise moves to `shared/provenance.py` (hashes
`pixi.lock` and `Manifest.toml` when present; a missing file yields `null`).

**Why the record stays fresh (P10 — the blocking defect).** Two complementary
mechanisms, each covering a case the other cannot. They are both required:

- **Params threading refreshes the record when the checkout moves.** Each
  Snakefile computes, at parse time,
  `CONFIGURATION_INPUTS_SHA256` (§5.4) — which folds in toolbox identity,
  environment hashes and referenced-input bytes — and threads it through rule
  X.01's `params:` as a **string digest**. Snakemake's default params
  rerun-trigger then re-executes X.01 whenever any component moves: a new
  commit, a dirty flip, a lock-file change, an in-place edit to a referenced
  catalog (closing the P9 gap in the *trigger*, not only in the recorded
  hashes). This is the repo's own probe-verified pattern — the WF3 drift guard
  threads `guarded_sections_digest` through params for exactly this reason
  (`Snakefile_climate_experiment:341-370`). The `effective_config` param
  becomes the §5.3 **projection** rather than the whole config, so a WF3-only
  edit no longer re-fires WF1's X.01 (retiring P2's residual in the retained
  record). Parse-time hashing of referenced files is already the house pattern
  (`CONFIG_SNAPSHOT_DIGEST` does it today, `Snakefile_model_creation:175-180`)
  and costs milliseconds. What params threading **cannot** do: record an
  invocation that executed nothing (X.01 up to date ⇒ no write), or record
  failure — a rule output has no way to say "this run crashed".
- **Lifecycle hooks record every *executed* invocation and its outcome.**
  Journal emission (§5.7) moves **out of rule X.01 entirely** into
  workflow-level `onstart:` / `onsuccess:` / `onerror:` handlers.
  **Amended 2026-08-13 after the P0 probe** (`p0-probe-result.md`) — this
  paragraph previously claimed handlers fire "for every non-dry invocation
  regardless of whether any job ran", and that is **false** on the pinned
  Snakemake 9.6.2: a "Nothing to be done" invocation fires none of the three
  (`workflow.py:1375-1377` returns before `_onstart`, unguarded by any flag).
  Hooks therefore cover the **failed invocation** and any invocation in which at
  least one job executed, and **not** the no-op or the byte-identical re-run.
  R5 is narrowed to match (owner ruling, 2026-08-13, §4): the journal is a
  ledger of executed invocations, not of every command typed. The narrowing is
  affordable because params threading folds toolbox identity, environment
  hashes and referenced-input bytes into the trigger — so any invocation whose
  *inputs moved at all* re-executes X.01 and is therefore recorded. What is
  lost is the invocation where config, code, environment and referenced inputs
  are all identical to the previous run, which carries no information beyond
  "the command was re-typed". What hooks **cannot** do: rewrite
  `run_record.yml` — a hook mutating a rule's declared output behind
  Snakemake's back would corrupt its up-to-date reasoning, which is why the
  record's freshness must come from the params trigger and not from a hook.

Implementation step 0 (**probe, before any code**): confirm on the pinned
Snakemake that (a) the three handlers fire on a normal invocation and on a
"Nothing to be done" no-op, (b) none fires under `--dry-run`.

**Probe result, 2026-08-13** (`p0-probe-result.md`, Snakemake 9.6.2): (b)
holds; (a) holds for the normal invocation and **fails for the no-op**. Two
further legs beyond the four required: a `--forcerun` of an up-to-date target
fires both handlers (confirming the trigger is *job execution*, not
invocation), and a DAG-build failure (`MissingInputException`) fires **nothing**
— not even `onerror`, since it is raised before the handler block is reached.
That last one is a residual of this design, accepted at §7 item 11: a Snakefile
wired to declare an output no script writes fails unrecorded.

### 5.3 Scope by consumed keys — declared, tested, and honest about enforcement

v1 scoped by section ownership; that was round 1's blocking finding, because
WF3 reads `workflows.model_creation` (verified:
`Snakefile_climate_experiment:348` guards
`"project", "shared.basin", "workflows.model_creation",
"workflows.climate_projections"`, and `:480-495` reads `wflow_outvars` out of
`workflows.model_creation`).

**Rule:** each Snakefile declares an explicit **consumed-key projection** — the
config paths that workflow actually reads, including cross-section ones — and
the record and both digests are built from that projection.

| Workflow | Projection |
|---|---|
| `model_creation` | `project`, `shared`, `workflows.model_creation` |
| `climate_projections` | `project`, `shared`, `workflows.climate_projections` |
| `climate_experiment` | **derived** (below): `project`, `shared`, `workflows.model_creation`, `workflows.climate_projections`, `workflows.climate_experiment` |

**WF3's projection is derived, not adjacent** (round-2 major). It is built
programmatically in the Snakefile from the same `guarded_sections` tuple the
drift guard hashes:

```python
PROJECTION_SECTIONS = tuple(sorted(
    {s.split(".")[0] if s == "shared.basin" else s for s in guarded_sections}
    | {"workflows.climate_experiment"}
))
# == ("project", "shared", "workflows.climate_experiment",
#     "workflows.model_creation", "workflows.climate_projections")
```

— widening `shared.basin` to `shared` (the guard narrows to `basin` only
because guard params must be experiment-invariant; WF3 reads other `shared`
keys) and adding its own section. A test asserts the declared projection equals
that union, so a future edit to `guarded_sections` propagates or fails loudly.
v2's "the declaration lives beside the list so the two cannot drift" was
proximity, not enforcement; this is enforcement.

**Mutation test (kept from v2, extended):** per workflow, for every section in
its projection, flipping a leaf key changes both digests; flipping a leaf in a
sibling section outside the projection changes neither; flipping an
`advanced_settings` key changes both (round-2: advanced settings must appear in
the mutation tests, not only in prose).

**What the guarantee is, honestly (round-2, both reviewers).** The mutation
test proves the digest follows the *declaration*; it cannot prove the
declaration matches what the workflow *reads* — the direction round 1's
blocking finding failed in. Three layers, in decreasing strength:

1. **WF3:** the derivation above makes its projection structurally complete
   with respect to the guard tuple, which is itself the maintained list of
   WF3's cross-section reads.
2. **WF1/WF2:** a **static completeness test** scans each Snakefile's source
   for config-access references to *other* workflows' sections (the literal
   patterns `["workflows"]["<other>"]` and `workflows.<other>` inside string
   arguments to `get_config`) and fails if any appears outside the declared
   projection. This is a coarse, textual check — stated as such — but the
   defect class it targets is exactly how gpt-1 manifested: a literal
   cross-section read in a Snakefile. Python `script:` modules cannot widen the
   read set behind its back, because they see only
   `snakemake.input/output/params`, never the raw config.
3. **The residual, stated:** R scripts receive `config_path` and can read any
   key (the repo's own forwarding convention), and a future Python-side read
   routed through a new param is visible to the params trigger but not to the
   projection declaration. For these the projection is a **reviewed
   declaration**: a comment block beside each declaration states "any new
   config read in this workflow updates this tuple", which is the same
   checklist mechanism the guard tuple itself relies on today.

### 5.4 Two digests with explicit, different meanings

Round 1 established that removing tier B trades a *noisy* identity for an
*under-inclusive* one (P9). Round 2 corrected two errors in how v2 drew the
line. Both digests are computed **at parse time** in each Snakefile (the
existing `EFFECTIVE_CONFIG_DIGEST` pattern) and passed to the rules that record
or embed them.

- **`effective_config_sha256`** — over the §5.3 projection **plus the resolved
  `advanced_settings` mapping**. This is *configuration identity*: "the
  settings the workflow was asked to run under". Advanced settings stay inside
  it, deciding the question round 2 raised: they are configuration (constraints
  and defaults that shape the run), today's digest already includes them
  (v2's contrary claim was false — `provenance.py:137-141`, driver-verified),
  and keeping them preserves continuity of meaning for the field WF2 already
  publishes into `summary/provenance.json`. What changed at
  `schema_version: 2` is the *config* term: the projection replaces the whole
  parsed configfile.
- **`configuration_inputs_sha256`** *(renamed from v2's `run_inputs_sha256` —
  round-2 major)* — canonical SHA-256 over: the `effective_config_sha256`
  document, `toolbox.commit`/`commit_source`/`dirty`, the two `environment`
  hashes, and the sha256 of every referenced input from §5.5 (recoverable ones
  contribute their git blob id, copied ones their byte hash; a logical
  identifier with no path contributes its identifier string). It answers "did
  this run see the same **configuration-side** inputs as that one".

**What the second digest deliberately does NOT cover, stated in its contract
and in `config/runs/README.md`:** the contents of scientific datasets addressed
*through* catalogs (a remote or mutable dataset can change under an unchanged
catalog), and generated intermediate inputs. It is configuration-input
identity, never scientific run identity — the rename exists to stop the broader
citation the old name invited. Extending it to fold in resolved-source
provenance was considered and rejected: `hydromt_data.yml` is written by rule
1.07, downstream of X.01, so folding it in creates a cycle; the resolved-source
records (§7 item 2) remain the carrier for that layer.

**Comparability rule, stated in the schema comment and README:** two
`configuration_inputs_sha256` values assert same-code only when both records
have `commit` non-null and `dirty: false`. With `commit: null` (git-less
deployment without a baked commit) or `dirty: true`, equal digests do **not**
imply equal code — the `environment` hashes still pin the dependency set, but
the toolbox source itself is unwitnessed. This is the honest trade of P11: the
`.toolbox-commit` bake (§5.2) is what closes it in the shipped image, and a
deployment that skips the build-arg gets a record that says so (`commit_source:
null`) rather than one that lies.

`schema_version` becomes **2**, so a reader can tell a rescoped digest from a
changed config across the transition.

### 5.5 R4 as a per-file predicate, not a per-bin ruling

`project.data_sources`, `model_build_config` and `waterbodies_config` hold
**arbitrary paths**, so a production project may reference a site-specific
catalog or a custom build config outside the toolbox. A bin-level rule discards
exactly the file R4 exists to protect.

**Predicate**, applied per referenced file at snapshot time:

1. Resolve the referenced path.
2. If `toolbox.commit` is non-null **and** the tracking queries succeed **and**
   the file is inside the toolbox checkout, tracked at that commit, and clean
   (`git ls-files --error-unmatch <path>` plus `git status --porcelain
   <path>`, both from the repo root, failures swallowed) → **do not copy.**
   Record `role`, `origin` (repo-relative), `recoverable: true`, `git_blob`.
3. Otherwise → **copy** into the project and record `role`, `origin`,
   `recoverable: false`, `archived_path`, `sha256`, `size_bytes`.

**Degraded mode is the predicate's else-branch, by construction (P11):** when
`commit` is null or any git query fails or is unevaluable — the Docker case —
step 2 can never be satisfied and **every referenced file is copied**. Stated
trade, accepted deliberately: git-less deployments re-acquire the P4
duplication (a few small YAML copies per project), because in an image that
cannot be interrogated the copies are the only way the project can say what it
ran with. R4's own test — "the toolbox cannot give it back" — is genuinely true
there. The `.toolbox-commit` bake restores the commit but not the tracking
queries (no `.git` in the image), so deployed projects copy regardless;
`commit_source: baked` still identifies the code.

**Collision safety (P8).** Copied files take **role-stable destination names**
(`output_locations.csv`, `observations_timeseries.csv`) rather than
`source_path.name`, and the writer **raises** on an unexpected destination
collision instead of overwriting. Every copy's origin→archive mapping lives in
`referenced_inputs`, which is what `referenced-files.json` used to provide.

Generated inputs that are neither repo files nor external — e.g.
`<exp_dir>/config/catalogs/data_catalog_climate_experiment.yml`, produced at
runtime over generated forcing — are **kept where their producing rule writes
them** and referenced by project-relative path. They are outputs, not config
snapshots. (The migration consequence of this sentence is now explicit — §6.)

### 5.6 The values-used record (R3, R6) — post-normalization, both rules

R3 asks for "the actual values used". The round-2 tie-break (driver-verified)
established that v2 did **not** deliver it: serializing
`read_parameter_steps()` output records the *input template*, while
`build_wflow_model.py:237-268` normalizes per step before calling hydromt — it
pops configured arguments (`hydrography_fn`, `river_geom_fn`, `lulc_fn`,
`lai_fn`), injects P1 dataset objects in their place, and **derives**
`lulc_mapping_fn` at call time (`f"{source_name}_mapping_default"`, off the P1
grid's `lulc_source` attribute — a value that appears in no file on disk). The
decisive case is that function's own 2026-08-13 comment: the template looked
correct throughout while CORINE land cover was read through
`vito_mapping_default` — "Wrong numbers, not a missing setting." A record
serialized from the template cannot catch the defect class the record exists
for.

**Mechanism — single construction point, serialize what the call receives.**
Restructure `_apply_parameter_steps` so each step branch builds its final
keyword mapping `call_kwargs` **once** and uses it **twice**: passed to the
hydromt method, and appended (with dataset substitution, below) to a
`recorded_steps` list. The record therefore cannot drift from the call — there
is no second derivation to get wrong.

Serialization rule for `recorded_steps`, per argument value:

- YAML scalar / list / mapping of scalars → recorded **verbatim,
  post-normalization**. This is what captures the derived
  `lulc_mapping_fn: corine_mapping_default`.
- Injected non-serializable object (an xarray/GeoDataFrame handed in from the
  P1 spatial products) → a **stable source-reference mapping**, never a repr:
  `{injected_from: p1_spatial_catalog, product: <name>}` — e.g.
  `hydrography_fn: {injected_from: p1_spatial_catalog, product: spatial_maps}`,
  `river_geom_fn: {injected_from: p1_spatial_catalog, product: rivers}`,
  `lulc_fn: {injected_from: p1_spatial_catalog, product: spatial_maps,
  variable: land_cover}`. The P1 catalog is itself a declared, provenance-carrying
  artifact, so the reference is resolvable.

After the build succeeds, rule 1.07 writes:

```yaml
# models/hydrology/wflow/hydromt_build_config.yml  (declared output, rule 1.07)
schema_version: 1
modeltype: wflow_sbm
source_template: <path as configured>
source_template_sha256: <sha>
steps:
  - setup_rivers: {hydrography_fn: {injected_from: ...}, river_upa: 32, ...}
  - setup_lulcmaps: {lulc_fn: {...}, lulc_mapping_fn: corine_mapping_default}
  ...
```

An analyst can now distinguish template values from adapter substitutions: the
template is named (and hashed) as the source, the steps are the values hydromt
received.

**The waterbodies twin, named and declared (round-2 minor).** Rule 1.08
`add_reservoirs_lakes_glaciers` consumes `waterbodies_config` and runs each
method individually, where a method may be legitimately **skipped** on
per-basin no-data (`setup_reservoirs_lakes_glaciers.py:22-39` already captures
per-method `ok`/`skipped` outcomes). Its values-used record is
`models/hydrology/wflow/hydromt_update_waterbodies.yml`, a declared output of
rule 1.08, same schema plus a per-step `status: ok|skipped` and `reason` —
because for this rule "the values used" includes *whether the method ran at
all*. The names parallel their `config/defaults/` sources
(`wflow_build_model.yml` → `hydromt_build_config.yml`,
`wflow_update_waterbodies.yml` → `hydromt_update_waterbodies.yml`) and sit
beside hydromt's own `hydromt_data.yml`; with no further review round, both
names are **settled** here (closing v2 open item 5).

Placement is the point (R6): in the model's output directory these are
**values-used records**; in `config/templates/` they would be **input
snapshots**, which R3 forbids.

### 5.7 The run journal (R5) — lifecycle hooks, undeclared side effect

The journal is `<project_dir>/config/runs/journal.jsonl` for all three
workflows (one ledger per project; WF3's experiment partition lives in the
line, not in a second file). It is written **only** by the workflow lifecycle
handlers (§5.2) via one shared helper, `append_journal_line(...)` in
`shared/provenance.py`.

**Declaration semantics (round-2 major — the silent-truncation trap).** The
journal is **never a declared output of any rule**: Snakemake deletes a rule's
declared outputs before executing the job, so a declared `journal.jsonl` would
be truncated to one line on every re-execution — silently, since a one-line
journal is indistinguishable from a young one. Moving emission into hooks
removes the temptation structurally (hooks have no outputs), and a test pins
it: no Snakefile rule declares an output path matching `journal.jsonl`. As an
undeclared side effect it must be **whitelisted in the project-tree inventory**
(`semantic_tree_diff.py` / `tree-check`), which §6 costs explicitly.

**Append mechanics.** `append_journal_line` creates parent directories, opens
with mode `"a"` (`encoding="utf-8"`, `newline="\n"`), writes the full JSON line
plus `\n` in **one** `write()` call, flushes, closes. Readers tolerate a torn
final line: any line that fails `json.loads` is skipped with a warning — the
file's value is the accumulated history, and a torn tail must not poison it.

**Line schema** — two lines per invocation, sharing an `invocation_id`
(`uuid4().hex`, minted at Snakefile parse time in a module-level binding the
three handlers all see):

```json
{"ts":"2026-08-13T12:04:07Z","invocation_id":"9f2c…","workflow":"climate_experiment",
 "experiment":"experiment_2026-08-13","event":"started","commit":"a1b2c3…",
 "commit_source":"git","dirty":false,"effective_config_sha256":"…",
 "configuration_inputs_sha256":"…","source_config_sha256":"…"}
{"ts":"2026-08-13T12:09:42Z","invocation_id":"9f2c…", …, "event":"success"}
```

`event` ∈ `started | success | failed` (`onstart`/`onsuccess`/`onerror`). The
**terminal line is the contract**; the `started` line is best-effort crash
tracing — an invocation killed hard (SIGKILL, power loss) leaves a `started`
line with no terminal partner, which is itself the diagnostic.

**What the journal is a ledger OF** (narrowed 2026-08-13, R5 in §4; probe
evidence in `p0-probe-result.md`): **executed** invocations — those in which at
least one job ran or was attempted. A no-op invocation ("Nothing to be done")
appends nothing, because the pinned Snakemake fires no handler for it. This must
be stated in `config/runs/README.md` (P7): a reader counting journal lines to
answer "how often was this project run" gets executed runs, and an absence of
lines across a period means no work was done in it — not that nobody looked. The
line count is a lower bound on invocations and an exact count of executions. WF3 lines carry
`"experiment"` (round-2 minor): the workflow where per-run identity matters
most must not depend on digest-matching to name its experiment. ~500 bytes per
invocation.

Why the `invocation_id` is **not** embedded in any rule output: a per-invocation
value threaded into a rule would re-run that rule on every invocation, breaking
idempotence — the same reason no path in this repo mints experiment names from
the clock. Correlation between journal lines and tree artifacts rides on
`configuration_inputs_sha256`, which both carry.

**Mandatory test (round-2):** two consecutive executions append — the journal
holds ≥ 2 invocations' lines afterwards, proving accumulation; plus the
torn-line reader test and the not-a-declared-output test above.

### 5.8 Making staleness checkable — sidecar carriers that exist

*(v2 put this inside §5.7 and specified it against artifacts that do not
exist; both round-2 reviews caught it, from opposite ends.)*

The journal identifies invocations; it cannot say which outputs came from
which. Each workflow therefore embeds its identity in a terminal
provenance-bearing artifact, and comparison against `run_record.yml` makes
staleness checkable. Two corrections against v2:

**The embedded digest is `configuration_inputs_sha256`** (round-2 major, GPT).
`effective_config_sha256` was the wrong field: a changed custom template,
catalog byte-edit, or toolbox move alters only the wider digest, so old outputs
would match a fresh record on the narrow one and read as current. The staleness
predicate is: `sidecar.configuration_inputs_sha256 !=
run_record.configuration_inputs_sha256` ⇒ the outputs predate the most recent
recorded configuration/toolbox state; the journal then names the invocations on
either side. (Both fields are still written into the sidecars — the narrow one
answers "same settings?", the wide one "same settings, code and inputs?".)

**The carriers are new declared sidecar outputs; the fingerprinted CSVs are
untouched** (round-2 major, Fable; driver-verified). WF1 has no "evaluation
summary" — its evaluation terminal is `evaluation/performance_metrics.csv`; WF3's
terminal indicator CSV `results/q_indicators.csv` **is** baseline-fingerprinted.
Writing into either would falsify the no-re-record claim. Instead:

- **WF1** — new rule `1.16 write_run_metadata`:
  `input:` `evaluation/performance_metrics.csv`; `params:` both digests;
  `output:` `{basin_dir}/evaluation/run_metadata.json`.
- **WF3** — new rule `3.17 write_run_metadata`:
  `input:` the experiment's indicator tables (the `INDICATOR_TABLES` outputs of
  3.16); `params:` both digests plus `experiment`;
  `output:` `{results_dir}/run_metadata.json`.
- **WF2** — no new file: `summary/provenance.json` already carries
  `effective_config_sha256` (`Snakefile_climate_projections:910`); its rule
  gains a `configuration_inputs_sha256` param and writes both. Verified: that
  file is **not** in `dev/baseline/manifest.json`, so the addition does not
  touch the baseline.

Sidecar content: `{"schema_version": 1, "workflow": …, "experiment": …
(WF3 only), "effective_config_sha256": …, "configuration_inputs_sha256": …}`.

Rerun semantics, and why they give the right answer in the P7 case: the
sidecar's params carry the parse-time digests, so it refreshes whenever
identity moves **and its input chain lets it run**. Config edit + failed run
under `--keep-going`: X.01 (early, cheap) succeeds and refreshes
`run_record.yml`; the terminal rule fails, so the sidecar never runs and keeps
the previous digest → mismatch → the mixed tree is *visible*, which is exactly
what P7 lacked. A docs-only commit refreshes record and sidecars cheaply and
consistently (no false staleness). Honest limit, stated: Snakemake's rerun
model does not see edits to *imported modules*, so a module-only code change
that regenerates nothing can refresh both sides in step and certify outputs the
new code never produced — a limitation shared by every rule in this repo, not
introduced here (§7 item 6).

This also gives the digests a reader and a stated purpose in all three
workflows — closing the round-1 "one writer, no readers in WF1/WF3" finding
without minting a bundle.

### 5.9 Resulting layout

```
<project_dir>/config/
  observations/                              (only when the R4 predicate says copy)
  runs/
    snake_config_model_creation.yml          UNCHANGED PATH
    snake_config_climate_projections.yml     UNCHANGED PATH
    model_creation/run_record.yml
    climate_projections/run_record.yml
    journal.jsonl                            (undeclared side effect, §5.7)
    invocations/*.json
  README.md
<project_dir>/models/hydrology/wflow/
  hydromt_data.yml                           (hydromt's own)
  hydromt_build_config.yml                   NEW — values used, rule 1.07 (§5.6)
  hydromt_update_waterbodies.yml             NEW — values used, rule 1.08 (§5.6)
  evaluation/run_metadata.json               NEW — staleness sidecar (§5.8)
<exp_dir>/config/
  snake_config_climate_experiment.yml        UNCHANGED PATH
  run_record.yml
  catalogs/data_catalog_climate_experiment.yml   (generated — kept, §5.5)
<exp_dir>/results/
  run_metadata.json                          NEW — staleness sidecar (§5.8)
```

**The three flat copies keep their exact paths.** The baseline fingerprints
three of them — the two project-scope copies *and*
`experiments/experiment/config/snake_config_climate_experiment.yml` — all
unchanged in path and content. Every artifact this design adds is a **new
file** outside the manifest, and the one fingerprinted file §5.8 could have
touched (`q_indicators.csv`) is explicitly untouched. **No baseline re-record
is required** — the claim survives because the carriers were re-specified, not
because it was assumed.

P5's layout inconsistency is left standing deliberately: fixing it would move a
baseline-fingerprinted, guard-read contract path for cosmetics.

### 5.10 Naming

**No `_generated` suffix.** `dev/reference/naming.md` §8's file-class row is
titled *"Generated outputs under `project_dir/`"* — everything in that tree is
generated, so the marker partitions nothing; under R4 every remaining copied
file is by construction irreplaceable, so it would be true of the whole set; and
§2 states *"discriminate by the consuming contract, never authorship or whether
the file is checked in vs. generated."* Renaming a data catalog is additionally
a contract-surface rename requiring a migration note.

**When a marker is warranted.** The repo's one precedent is
`wflow_sbm.reference.toml`, whose README states the infix exists *"because the
bare name `wflow_sbm.toml` read as a live input"* — a marker earns its place
when a name would be mistaken for a different **role**, not to record
provenance.

**Term reuse at field level.** `effective_config` remains the name of the
*field*, matching `provenance.py`'s established `effective_config_document` /
`effective_config_digest` / `effective_config_sha256`. The *file* is
`run_record.yml` because it carries more than the effective config. The wider
digest is `configuration_inputs_sha256` — snake_case per naming.md §1, and
deliberately **not** `run_inputs_sha256` (v2's name), whose breadth invited
citation as scientific run identity (§5.4). GPT's suggested
`run_configuration_inputs_sha256` drops its redundant `run_` prefix — the
record is per-run already.

**`config/runs/README.md`** states "everything here is written by the run; edit
the source config instead" — addressing the one genuine trap, that
`snake_config_model_creation.yml` looks editable and silently is not — and
carries the §5.4 digest contracts and comparability rule. Per-bin READMEs are
the house pattern.

---

## 6. What this costs to implement

**Step 0 — probe** the lifecycle-handler behaviour on the pinned Snakemake
(§5.2): normal, no-op, failed, and `--dry-run` invocations. Cheap, and the
journal contract depends on it. **Done 2026-08-13** — it cost the design a
claim: handlers do not fire on a no-op, so R5 was narrowed to executed
invocations rather than a mechanism added (`p0-probe-result.md`, §4 R5,
§7 item 11). The probe paid for itself; the contract it corrected would
otherwise have shipped as an untested assumption inside P4.

- `blueearth_cst/shared/provenance.py` — consumed-key projection document; the
  two digests (§5.4); `toolbox_identity()` and `environment_file_hashes()`
  (moved from `scripts/run_workflows.py`, which now imports them — one
  definition); `append_journal_line()` with the §5.7 append/torn-line
  contract; `schema_version` → 2; **delete `snapshot_bundle_digest`,
  `short_digest`, `SHORT_DIGEST_CHARS`** once §5.1 lands (their only callers
  are the code being removed — verified).
- `blueearth_cst/model/copy_config_files.py` — drop `_write_snapshot_bundle`;
  write `run_record.yml` atomically; implement the §5.5 predicate including
  the null-commit else-branch, role-stable names and collision refusal. (No
  journal code here — hooks own it.)
- `blueearth_cst/model/build_wflow_model.py` + rule 1.07 — single-construction
  `call_kwargs` refactor of `_apply_parameter_steps`; emit
  `hydromt_build_config.yml` as a declared output (§5.6).
- `blueearth_cst/model/setup_reservoirs_lakes_glaciers.py` + rule 1.08 — emit
  `hydromt_update_waterbodies.yml` (values + per-method status) as a declared
  output (§5.6).
- The three Snakefiles — drop `CONFIG_SNAPSHOT_DIR` and the `snapshot_bundle`
  output; declare the consumed-key projection (WF3's derived from
  `guarded_sections`, §5.3); compute both digests at parse time; thread
  `configuration_inputs_sha256` through X.01's params and pass the projection
  (not the whole config) as `effective_config` (§5.2); add
  `onstart`/`onsuccess`/`onerror` handlers calling `append_journal_line`
  (§5.7); add rules `1.16`/`3.17 write_run_metadata` and the WF2
  provenance-param extension (§5.8).
- `Dockerfile` — `ARG TOOLBOX_COMMIT` + write `.toolbox-commit` (§5.2);
  `.gitignore` entry for the file.
- `dev/scripts/semantic_tree_diff.py` — update the project-tree inventory:
  remove bundle/template/catalog snapshot paths; add `run_record.yml` (×3),
  `journal.jsonl` (whitelisted side effect), the two values-used records, the
  two `run_metadata.json` sidecars.
- **Tests:** `tests/test_copy_config_files.py` (predicate incl. degraded mode
  via monkeypatched git absence; collision refusal; atomic record);
  `tests/test_shared_provenance.py` (digest split; identity resolution order;
  journal append/accumulation/torn-line; deleted-helper removals);
  `tests/test_snapshot_config_rules.py` (rewrite — currently asserts the
  bundle at lines 51/67, verified); **new** per-workflow mutation tests incl.
  `advanced_settings` (§5.3); the WF3 projection-derivation equality test and
  the WF1/WF2 static completeness test (§5.3); the journal
  not-a-declared-output test (§5.7); `tests/test_build_wflow_model.py` — the
  record equals the call kwargs, with the derived
  `lulc_mapping_fn` case asserted explicitly (the driver's decisive example);
  the waterbodies record test; sidecar-rule tests;
  `tests/test_run_workflows.py` (helper relocation);
  `tests/test_project_tree_inventory.py` (new inventory).
- **`README.md` ~170–186** documents the bundle and `referenced-files.json` —
  under the repo's "keep configuration references current" rule that is a
  defect once the bundle is gone, not a record. `config/runs/README.md` (§5.10)
  is new.
- **Migration.** Existing trees — including the reference fixture the
  fixture-dependent test layer runs against — hold undeclared orphans the
  moment the inventory changes. Ship a one-shot cleanup in the house pattern of
  the two existing prune tools (**report-only by default, `--delete`
  explicit**) and run it against the fixture **before** the inventory tests are
  rewritten. **Scope, exactly (round-2 minor):** the bundle directories
  `<project_dir>/config/runs/<workflow>/<12-hex>/` and
  `<exp_dir>/config/runs/climate_experiment/<12-hex>/`; and, under
  `<project_dir>/config/templates/` and `<project_dir>/config/catalogs/` only,
  files **byte-identical to a tracked toolbox file** (the R4 predicate applied
  retroactively — anything else is reported, never deleted).
  **`<exp_dir>/config/catalogs/` is never touched**: it holds the generated
  experiment catalog §5.5 keeps, and a pattern-match on `config/catalogs/`
  across the whole tree would delete a kept artifact. `config/observations/`
  copies are kept (outside-repo files — the predicate copies them by design).
- No baseline re-record (§5.9): all three fingerprinted flat copies unchanged;
  the fingerprinted `q_indicators.csv` untouched; WF2's `provenance.json` is
  not fingerprinted (verified); every added artifact is a new file.

---

## 7. Open items and residuals the owner should see

1. **`toolbox.dirty: true` is not recoverable provenance.** The commit does not
   identify the code that ran; §5.4's comparability rule makes the condition
   explicit at comparison time rather than silent, but preserving bytes or a
   patch for a dirty checkout of the workflow code remains unimplemented.
2. **WF3 has no resolved-source record** equivalent to WF1's `hydromt_data.yml`
   or WF2's `provenance.json`. After §5.1 its input provenance is the generated
   experiment catalog plus `run_record.yml`'s `referenced_inputs`.
3. **P5 layout inconsistency** left standing deliberately (§5.9).
4. **P1 left standing deliberately** (round-2: v2 never dispositioned it). The
   flat copies keep whole-config content under per-workflow names because
   their paths are baseline-fingerprinted and drift-guard-read; the
   projection-bearing artifact is now `run_record.yml` beside them, and
   `config/runs/README.md` disarms the editability trap. The name–content
   mismatch itself is accepted.
5. **`prepare_build_config.py` is orphaned dead code** — noted, out of scope.
6. **Code-identity blind spot inherited from Snakemake:** edits to imported
   modules do not trip the rerun model, so record, sidecars and outputs can
   refresh in step around outputs the new code never regenerated (§5.8). Not
   introduced by this design; naming it here so nobody reads the staleness
   check as stronger than it is.
7. **Scientific data identity is excluded** from
   `configuration_inputs_sha256` by contract (§5.4): catalog-addressed remote
   or mutable datasets can change without moving any digest. The resolved-source
   layer (item 2) is where that would live.
8. **R-script config reads** are outside the projection enforcement (§5.3
   layer 3) — a reviewed-declaration residual, same mechanism as the guard
   tuple relies on today.
9. **Residual forced by R4 + production reality (P11):** in git-less
   deployments the predicate must copy everything, re-acquiring bounded P4
   duplication precisely where nobody watches (§5.5). The `.toolbox-commit`
   bake restores commit identity but cannot restore per-file tracking queries.
   If the owner wants deployed projects copy-free, the alternative is baking a
   tracked-file hash manifest into the image — deliberately not designed here.
10. **Hard-kill invocations** (SIGKILL, power loss) leave a `started` journal
    line with no terminal partner (§5.7) — a trace, not a record of outcome.
    Accepted; matches `run_workflows.py`'s own manifest behaviour.
11. **Two invocation classes the journal cannot see** (added 2026-08-13 from the
    P0 probe; ruled accepted by the owner the same day, R5 narrowed to match):
    the **no-op** invocation, where nothing needed doing and Snakemake returns
    before any handler runs; and the **DAG-build failure**, raised earlier still
    (`MissingInputException`), which fires not even `onerror`. The first is
    accepted because params threading makes it informationally empty — an
    invocation whose inputs moved is never a no-op. The second is a real loss:
    a Snakefile wired to declare an output no script writes fails without a
    journal line, which is exactly the P4 wiring error the master brief flags.
    Mitigation is the ordinary one, `pytest tests/test_cli.py`, which dry-runs
    all three Snakefiles and surfaces that class before it reaches a run.
