# External review — Fable, round 2

## Verdict
verdict: revise
doc_version: config-snapshot-design-v2.md

## Findings

### ext2-1  [blocking]
- section: §5.2 (run_record.yml) / §5.7 (journal), interaction with R7
- finding: `run_record.yml` and the journal line are written by rule X.01, whose rerun triggers are the configfile, the referenced inputs, its own script, and its params (`effective_config`, `advanced_settings` — verified at `Snakefile_model_creation:420-443`). `toolbox.commit` and `dirty` change none of these. A code-only commit change (edit science code, commit, rerun) regenerates outputs via the code rerun-trigger on the changed rules while X.01 stays up to date: the record attributes the fresh outputs to the old commit, `run_inputs_sha256` never updates, and the journal gains no line for the invocation that actually produced the tree's contents.
- rationale: The R7 stamp is reliably wrong in the single most common provenance scenario — upgrade the toolbox, rerun an existing project — which is exactly the case P6 exists for (direct `snakemake` invocations recording nothing). A provenance record that answers "which commit produced these outputs" with the previous commit is worse than the absence it replaces, because it is confidently wrong. The §5.7 staleness check cannot catch it either: the embedded field is `effective_config_sha256`, which is unchanged in this scenario, so the mixed tree reads as consistent.
- suggested_fix: Thread `toolbox.commit` and `dirty` (or a digest over the run-identity components) into rule X.01's `params:` so the params rerun-trigger refreshes the record and appends a journal line when the checkout moves. This is the repo's own probe-verified pattern — the WF3 drift guard threads `guarded_sections_digest` through params for precisely this reason (`Snakefile_climate_experiment:~344-370`). Note the side effect is desirable: the journal then records code-identity changes, which the P7 mechanism needs.

### ext2-2  [major]
- section: §5.7 (journal) / §6 ("declare the new outputs")
- finding: The design does not say whether `journal.jsonl` is a declared Snakemake output of rule X.01. §6 instructs the Snakefiles to "declare the new outputs", and Snakemake deletes a rule's declared output files before executing the job — so a declared journal is truncated to a single line on every re-execution of X.01, silently reinstating current-only and voiding the P7 fix while every test still passes (the file exists and holds a valid line). If instead it is an undeclared side effect, it is an artifact `tree-check` and the inventory tests will flag unless whitelisted, and the design's cost section does not name that step. Append atomicity and torn-line handling on crash are also unspecified for a file whose whole value is being append-only.
- rationale: Under the most natural implementation of the design's own §6 instruction, the mechanism that "addresses P7" self-destructs without any observable failure. The failure mode is invisible by construction: a one-line journal looks exactly like a young journal.
- suggested_fix: Specify: the journal is an undeclared side-effect write (open with `O_APPEND`, one `write()` per line; reader tolerates a torn final line), whitelisted in the project-tree inventory; rule X.01's declared outputs remain the flat copy and `run_record.yml`. Add a test asserting the journal accumulates across two X.01 executions.

### ext2-3  [major]
- section: §5.2 (toolbox block) / §5.4 (run_inputs_sha256) / §5.5 (predicate step 2)
- finding: Every new mechanism assumes the toolbox is a git checkout with `git` on PATH. The production deployment path is the opposite: the GUI drives these Snakefiles server-side, and the `Dockerfile` ADDs sources without `.git`. The repo already knows this — `run_workflows._git_metadata()` (verified `scripts/run_workflows.py:424-433`) returns `commit: None, dirty: None` on any failure. The design specifies no degraded mode: §5.2's schema is `dirty: true|false` with no null; §5.5 step 2 ("tracked at `toolbox.commit`") is unevaluable with a null commit, and the fallback direction is unstated; `run_inputs_sha256` folds commit/dirty in, so with both null, two runs of *different* deployed code versions hash equal — the field then answers "did this run see the same inputs" with a false yes. Relatedly, with `dirty: true` two differently-dirty checkouts also hash equal; open item 1 admits the dirty commit does not identify the code but does not warn that the digest asserts equality anyway.
- rationale: The design's provenance guarantees hold on the developer's machine and silently degrade in production — the environment whose outputs are quoted in reports. Additionally, if the null-commit fallback is (correctly) "copy", every shipped catalog and default is re-copied in Docker deployments, resurrecting the P4 duplication in exactly the setting where nobody is watching; that trade should be stated, not discovered.
- suggested_fix: Define the degraded mode explicitly: `commit`/`dirty` nullable in the schema; §5.5 predicate falls back to copy when commit is null or the tracking query fails; `run_inputs_sha256` additionally folds in `toolbox.version` and the environment lock-file hashes (`run_workflows._environment_file_hashes()` already computes pixi.lock/Manifest.toml sha256s) and is documented as comparable only when `dirty` is false and `commit` non-null.

### ext2-4  [major]
- section: §5.7 (complementary mitigation) / §5.8 & §6 ("no baseline re-record")
- finding: The carriers for the embedded `effective_config_sha256` do not exist. WF1 has no "evaluation summary" — its evaluation terminal is `performance_metrics.csv` (verified `Snakefile_model_creation:379`), a CSV; WF3 has no "results metadata" — its terminals are the indicator CSVs, and `q_indicators.csv` is one of the seven baseline-fingerprinted targets (verified `dev/baseline/manifest.json`). If "gain the same field" means modifying those CSVs, the baseline content changes and §6's "No baseline re-record" is false; if it means new sidecar files, the design neither says so nor names them, and the inventory/cost sections do not account for them.
- rationale: The load-bearing half of the P7 closure — the only part that makes staleness *checkable* rather than merely journaled — is specified against artifacts that do not exist, deferring a decision (modify fingerprinted CSVs vs. mint new outputs) that flips a headline claim of the design. An implementer choosing the CSV route discovers the contradiction as a red baseline gate mid-implementation.
- suggested_fix: Specify new declared sidecar outputs (e.g. `evaluation/run_metadata.json` for WF1, `results/run_metadata.json` per experiment for WF3) carrying `effective_config_sha256` plus schema_version; state that the fingerprinted CSVs are untouched, preserving the no-re-record claim; add both to the inventory update in §6.

### ext2-5  [major]
- section: §5.3 (consumed-key projection)
- finding: The mandated mutation test proves the digest is computed from the declared projection — it cannot prove the projection matches what the workflow actually reads, which is the direction the round-1 blocking finding (gpt-1) failed in. A future cross-section read added without a projection update passes every mandated test. The anti-drift claim rests on adjacency ("the declaration lives beside the guarded-sections list … so the two cannot drift"), which is proximity, not enforcement; and it only covers WF3's guarded sections, not WF1/WF2 or unguarded cross-reads.
- rationale: The design presents the test as what makes the projection "a contract rather than an assertion"; as specified it remains an assertion, and the exact defect class that was blocking in round 1 would recur undetected. Regression lens: an accepted fix claiming a resolution strength it does not deliver.
- suggested_fix: Make WF3's projection derived, not adjacent: build it programmatically as `guarded_sections` ∪ `workflows.climate_experiment` from the same tuple the guard hashes, and add a test equating the declared projection with that union. For WF1/WF2, state honestly that the projection is a reviewed declaration, and add a checklist trigger (naming.md or the Snakefile comment) that any new `get_config` read outside the projection must update it.

### ext2-6  [minor]
- section: §3 P1 / §5.9 / §7
- finding: P1 receives no disposition. v2 keeps the flat verbatim whole-config copies under per-workflow names (justified for baseline and guard reasons in §5.8), so the defect P1 describes — a per-workflow name promising a projection the content does not deliver — persists unacknowledged. The README fix addresses editability, not the name-content mismatch. P5 got an explicit "left standing deliberately" entry; P1 did not.
- rationale: A problem list where one item is silently unresolved invites a round-3 reviewer (or the implementer) to rediscover it as a gap; the document should close its own ledger.
- suggested_fix: Add P1 to §7 as deliberately left standing (path pinned by baseline and drift guard; `run_record.yml` is now the projection-bearing artifact), or note it under §5.8.

### ext2-7  [minor]
- section: §5.7 (journal line schema)
- finding: WF3 journal lines carry `workflow: climate_experiment` but no experiment identity. In a project with several experiments, mapping a journal line to an experiment rides entirely on `effective_config_sha256` matching the experiment's `run_record.yml` — indirect, and ambiguous if two experiments ever share a projection.
- rationale: The workflow where per-run identity matters most (R2: each experiment is its own partition) is the one whose journal lines are least self-describing; the fix costs one field.
- suggested_fix: Add `"experiment": <name>` to WF3's journal lines.

### ext2-8  [minor]
- section: §5.6 (values-used record)
- finding: "The same treatment applies to the waterbodies update config consumed by its own rule" is one sentence with no output name or path; two values-used records landing in one model directory need distinct, parallel names, and the cost section (§6) lists only the rule-1.07 emission.
- rationale: Unspecified twin artifacts get improvised names; naming.md makes renames of contract surfaces expensive later.
- suggested_fix: Name it (e.g. `hydromt_update_waterbodies.yml`, beside `hydromt_build_config.yml`) and add the emitting rule to §6.

### ext2-9  [minor]
- section: §6 (migration) / §5.5 (generated inputs)
- finding: The one-shot cleanup sweeps "bundle, template and catalog paths", but §5.5 keeps the generated experiment catalog at `<exp_dir>/config/catalogs/data_catalog_climate_experiment.yml`. A cleanup that pattern-matches `config/catalogs/` across the tree deletes a kept artifact that a re-run must then regenerate — or worse, deletes it in a tree whose forcing has been pruned.
- rationale: The distinction between project-level copied catalogs (remove) and experiment-level generated catalogs (keep) exists only in §5.5 prose; the migration spec should carry it explicitly since the tool's `--delete` mode is the one place the design authorizes deletion.
- suggested_fix: State in §6 that the cleanup's catalog scope is `<project_dir>/config/catalogs/` only, never `<exp_dir>/config/catalogs/`.

## Regression check

- gpt-1: resolved — verified `Snakefile_climate_experiment` guards `project`, `shared.basin`, `workflows.model_creation`, `workflows.climate_projections` and §5.3's WF3 projection covers all cross-reads including `wflow_outvars`; the enforcement gap in the accompanying test is re-raised as ext2-5.
- gpt-2: resolved — §5.6 emits the consumed parameter steps from rule 1.07 into the model directory; verified `config/defaults/wflow_build_model.yml` carries the cited `setup_rivers`/`setup_soilmaps`/`setup_constant_pars` values and `build_wflow_model.py:59-63` rejects `setup_basemaps` as claimed.
- gpt-3: resolved — the §5.5 per-file predicate replaces the bin ruling and protects out-of-repo catalogs; degraded-mode gap re-raised as ext2-3.
- gpt-4: resolved — the two-digest split correctly separates configuration identity from run-input identity; the null-commit/dirty equality hazard of `run_inputs_sha256` is re-raised within ext2-3.
- gpt-5: resolved — verified `copy_config_files.py:81` (basename copy) vs `:144` (hash-prefixed); role-stable names plus raise-on-collision plus the `referenced_inputs` mapping is a sound replacement.
- fbl-1: resolved — the adopted placement (rule 1.07 output in the model directory) satisfies R3/R6 better than a snapshot-bin carrier would; residual naming gap for the waterbodies twin is ext2-8.
- fbl-2: resolved — one-shot cleanup in the house report-only/`--delete` pattern, sequenced before the inventory tests are rewritten; one sweep-scope edge re-raised as ext2-9.
- fbl-3: resolved-with-new-defect — the stamp is folded into `run_record.yml` per R7, but as specified it goes stale on every code-only rerun (ext2-1) and is undefined without a git checkout (ext2-3).
- fbl-4: resolved-with-new-defect — the journal plus embedded digests close the config-edit-then-failed-run case in principle, but the journal's Snakemake mechanics can silently truncate it (ext2-2) and the embedding carriers do not exist (ext2-4); the mechanism's design intent is right, its specification is not yet implementable.
- fbl-5: resolved — the adopted predicate (my path-prefix test extended with tracked-and-clean-at-commit) is strictly stronger than my proposal; I endorse the extension. Fallback when git is absent belongs to ext2-3.
- fbl-6: resolved — `schema_version` bumped to 2, correctly framed as distinguishing a rescoped digest from a changed config.
- fbl-7: resolved — verified `tests/test_snapshot_config_rules.py` asserts the bundle wiring, `README.md:170-186` documents the bundle, and `snapshot_bundle_digest`/`short_digest` call sites all fall inside code §6 removes (the three Snakefile call sites are in §6's removal list, so "only callers are the code being removed" holds).
- fbl-8: resolved — verified `dev/baseline/manifest.json` fingerprints exactly three `snake_config_*` copies including the experiment's; the corrected verification stands, and the no-re-record conclusion holds for the flat copies (the separate `q_indicators.csv` hazard is new in v2's §5.7 and is ext2-4, not a regression of this finding).
- fbl-9: resolved in principle — WF1/WF3 embedding gives the retained digest a reader and a purpose; the delivery gap (nonexistent carrier artifacts) is re-raised as ext2-4 rather than counted against the disposition.
