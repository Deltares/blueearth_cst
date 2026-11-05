# Validation procedures

Last updated: 2026-11-05

Use these procedures when the [validation ladder](dev-validation.md) calls for formal evidence reuse,
Windows test isolation or an all-workflow gate. The ladder chooses checks and states their acceptance
limits; a successful command here never establishes scientific adequacy on its own.

## Pytest temporary directories

### Windows Snakemake cache access

- On Windows, run Snakemake subprocess and in-process DAG checks with normal user permissions outside
  a restricted sandbox, regardless of pytest marker. Pure Python checks can remain sandboxed. This
  is not administrator elevation or authority for production runs or publishing.
- When cache access is uncertain, probe before a broad run:

```powershell
pixi run python dev/scripts/check_snakemake_access.py
```

- The probe tests source-cache directory creation and a file round-trip in a child process. It exits
  0 on success or 2 with a diagnosis; the default timeout is 15 seconds. It runs no workflow or
  permission change. A timeout can also mean slow imports: compare under normal permissions.
- Snakemake's Windows user source cache is separate from pytest scratch. `tempfile.mkdtemp()` can
  retry denied creation despite a writable-looking parent; ACLs may exclude restricted tokens.
  Changing pytest `--basetemp`, `TMP` or `TEMP` does not redirect the cache. Do not patch packages,
  relax ACLs or extend workflow timeouts to hide this failure. Diagnose failures under normal
  permissions before continuing.

### Worktree-local pytest scratch

- `tests/conftest.py` uses `dev/scripts/pytest_temp.py` to place bare pytest and Pixi task fixtures
  in `.tmp/pytest/run-<unique-id>/` per checkout. Xdist workers nest below that root; reusable cache
  is `.tmp/pytest/cache/`. Concurrent default runs do not clear each other's fixtures. This covers
  `tmp_path`, `tmpdir` and `tmp_path_factory`, not application calls to `tempfile`.
- On Windows, inherited workspace ACLs allow the restricted token where Python's `0700` mode would
  deny it; moving an inaccessible directory does not repair its ACL. POSIX retains pytest's private
  permissions. The adapter uses pytest internals: check its fixture test with `-n 2` after upgrades.
  Existing inaccessible roots are left alone.
- Each new default run prunes only this checkout's `.tmp/pytest/run-*` roots untouched for two days
  (`STALE_RUN_SECONDS`), skipping links and undeletable roots. Copy runs needed longer elsewhere.
  Pytest's ordinary three-run retention does not apply to these unique roots.
  Explicit `--basetemp` runs and xdist workers never prune. Do not delete the shared parent during
  testing. `--basetemp <path>` clears that exact directory before use; use distinct paths for
  concurrent runs. Tests needing an outside path must choose it explicitly.

## Recording gates and verifying landing

Use tracked [`verify_landing.py`](../scripts/verify_landing.py) for formal gate reuse. A fresh
landing check can verify itself; ordinary focused checks need no receipt. Keep evidence/logs in
[task scratch](dev-repository-layout.md#task-temporary-material); this example uses standalone
`.tmp/scratchpad/gates/`.

```powershell
pixi run python dev/scripts/verify_landing.py record --evidence .tmp/scratchpad/gates/base.json --gate test-fast --log .tmp/scratchpad/gates/fast.log -- pixi run test-fast
pixi run python dev/scripts/verify_landing.py verify --evidence .tmp/scratchpad/gates/base.json --require-gate test-fast
```

- `record` runs one gate, saves a fresh log and records Git tree, Python executable/version/platform,
  lock identity and named input hashes. Record during the original check; do not rerun just for a
  receipt or import old logs. Capture output in a file, never through `tail`.
- Default candidate is clean/committed. For pre-commit reuse, stage only owned paths and use
  `record --staged`; it refuses unstaged/nonignored untracked changes and rechecks identity. It
  never stages or commits. Schema 2 records pre-test HEAD as `commit` and tested index as `tree`;
  schema 1 remains supported. Commit the identical tree before `verify`. A check editing tracked
  files invalidates its receipt.
- Repeat `--require-gate` for every owed gate, and use `--input` during recording for external
  config/version files that affect interpretation. `test-full` satisfies `test-fast` only for the
  same candidate and environment; execution and numerical gates keep distinct names. Additional
  gates must retain candidate, environment and input identity; use new evidence after a material
  change.
- `verify` runs no gate command. It rejects missing/failed/skipped required gates, altered logs,
  changed named inputs/environment and unclassified tree differences. It measures Python and lock
  files, not every binary, remote service or dataset. Hashing a catalog does not hash its data;
  invalidate evidence if an unrecorded dependency changes. The helper cannot choose coverage or
  assess scientific results.
- Use `--allow-path exact/path.md` only after reviewing that exact intervening Markdown change.
  Test-read documentation needs affected checks; code changes need fresh affected coverage and the
  reviewed-delta procedure below. No wildcard exclusions. Inspect logs for required skips; mark a
  wholly skipped command with `record --skipped` and never count it as passed.

### Reviewed follow-up changes

Use JSON review only for formal reuse across behavior-changing follow-ups. Keep base evidence/logs;
record affected checks separately with the same named inputs and commit a staged candidate before
verify. For a staged base, `base_commit` is the later commit matching the tested index. Review the
*complete cumulative diff*, including transitive readers, deletions and both rename paths; do not
chain unchecked attestations.

The review has exactly these fields, with actual Git identities and changed paths:

```json
{
  "base_commit": "FULL_EARLIER_TESTED_COMMIT",
  "candidate_tree": "CURRENT_GIT_TREE",
  "paths": {
    "blueearth_cst/shared/snake_utils.py": "Wording only; no parsed output or control flow changed."
  },
  "retained_gates": {
    "test-full": "Earlier non-console coverage remains applicable; affected console checks reran.",
    "test-e2e": "Workflow commands, data, calculations and handoffs are unchanged."
  },
  "require_gates": ["console"]
}
```

Every path needs its impact and coverage reason; every retained gate needs a reason its remaining
claims hold. Gates in `require_gates` must pass in the new evidence. The `console` path/gate above
are illustrative. After recording the affected check, verify with `--delta-evidence` and
`--delta-review`, requesting base, retained and new gates through `--require-gate`; see the script
CLI for argument order. Do not combine a delta review with `--allow-path`.

The verifier checks paths, ancestry, tree, environment/inputs and log integrity/results, not impact
reasoning. Selective reuse does not mean full suite or E2E ran on the follow-up. Broad/uncertain
impact needs fresh gates; changed inputs/environment prevent delta reuse. Release/milestone labels
alone do not invalidate evidence. Zero exit means command success, not scientific acceptance or proof
all required tests ran. Retain human numerical-difference and visual assessments with the work.

## The end-to-end runs (`test-e2e-local`, `test-e2e`)

- `pixi run test-e2e-local` executes the same five-workflow rapid case and deliverable assertions
  as live E2E from a sealed, source-only local bundle. Set `BLUEEARTH_CST_E2E_BUNDLE` to its absolute
  directory. It copies four raw CMIP6 basin slices into a fresh project root, points copies of pinned
  HydroMT catalogs at a local source mirror, and regenerates extracts, model, change factors,
  scenarios and simulations. Missing/stale bundles fail; CMIP6 cache misses fail before network
  access. Copied catalogs have no P-drive fallback roots.
- Stage spatial sources through the tracked `dev/scripts/stage_data.py` using
  `dev/scripts/local_e2e_data.yml`; place the four raw CMIP6 slices from `stage_cmip6.py` under
  `<bundle>/cmip6/raw`, preserving bytes, attributes, acquisition digests, units and calendars.
  [Environment maintenance](../scripts/environment-notes.md#local-data-staging-p-drive-bypass)
  explains ordinary staging. Seal and verify the bundle outside Git:

```powershell
pixi run python dev/scripts/stage_data.py --config dev/scripts/local_e2e_data.yml --dst <bundle>/hydromt
pixi run python dev/scripts/local_e2e_bundle.py seal <bundle>
pixi run python dev/scripts/local_e2e_bundle.py verify <bundle>
$env:BLUEEARTH_CST_E2E_BUNDLE = '<absolute-bundle-path>'
pixi run test-e2e-local
```

- The seal checks the 1999-12-31 CHIRPS boundary, full 2000–2016 CHIRPS/ERA5 coverage, pinned
  spatial source families, all four slices and shared region identity. Its inventory records each
  file's SHA-256 plus rapid config/catalog hashes; any added or changed byte needs a new seal. The
  first local run must match its freshly delineated polygon to the slices' digest, not substitute
  an old region.
- Unit tests cannot establish remote availability or every upstream format change. Neither E2E gate
  writes under `test_case/`; both use fresh roots. Live `test-e2e` invokes `scripts/run_workflows.py`
  with rapid config and needs the data mirror, juliaup Julia, `weathergenr` and WF2 network access.
  A missing prerequisite gives a named skip (`-rs`), which supplies no evidence. The main ladder
  chooses local versus live scope and evidence reuse.

## Test-author isolation details

- Repository-workflow CLI tests use `isolated_snakemake` in `tests/conftest.py`. It sets a fresh
  test-owned directory via `SNAKEMAKE_PROFILE`, including child launcher calls. Preserve that
  profile in subprocess environments. Sources and small relative inputs are copied; producing tests
  own outputs and `.snakemake` state. In-process simulation DAG checks set cwd explicitly; parse-only
  helpers declare rules without persistence/IO cache. Synthetic Snakefiles already in `tmp_path`
  retain their own state.
- `test-contract` stays serial: isolated engine state does not prove all output and process resources
  independent. Module-import stubs using `sys.modules.setdefault` can inherit a real module or earlier
  stub depending on collection order. Patch source-module bindings in a fixture and restore per test.
  Do not stub `dask` with `SimpleNamespace`; pandas may inspect `dask.__spec__`. Use the installed
  package or mock the call site. Examples are `tests/test_extract_historical_climate.py` and
  `tests/test_prepare_climate_data_catalog.py`.

## CI inspection commands

Install the tracked pre-push Ruff hook once per clone with `git config core.hooksPath .githooks`.
It does not run the software suite or platform checks. Check `gh`'s selected repository against
`git remote -v`, then specify it:

```powershell
gh run list --repo <owner>/<repo> --branch main
gh run view <run-id> --repo <owner>/<repo>
```

If querying the API by `head_sha`, supply the full 40-character ID; a short ID can silently match
nothing. GitHub does not allow self-approval of a PR. Before enabling auto-merge, require both
platform checks; otherwise it can merge immediately.
