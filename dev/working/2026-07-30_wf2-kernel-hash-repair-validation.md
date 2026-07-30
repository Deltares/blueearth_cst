# Validation record — cache-guard repair + the plot-job deadlock

```
Date:   2026-07-30
Steps:  the r2 review's efficiency repair (§2 step 1) and the blocker it exposed
Commits: kernel_hash repair; plot .load() fix
Gates:  check_baseline OK 15/15; semantic_tree_diff 1 FAIL — CHARACTERIZED, see §4
```

Two per-cause commits, one expensive gate at the boundary, per the revised gate
schedule (`dev/reviews/2026-07-30_wf2-v2-process-review-r2.md` §5). Step 4c's
fixture gates, outstanding since `b9e8556`, are discharged here.

---

## 1. What was repaired, and the falsifier for it

`kernel_hash` excluded string constants **by type**, so five behaviour-changing
edit classes reused cached series silently. Measured before and after with the
same probe:

| Edit class (real shape in the reducer) | Before | After |
|---|---|---|
| `dim="time"` → `"month"` | MISSED | INVALIDATES |
| `ds["pr"]` → `ds["tas"]` | MISSED | INVALIDATES |
| `resample(time="MS")` → `"YS"` | MISSED | INVALIDATES |
| date bound `2014` → `2020` | MISSED | INVALIDATES |
| default arg `273.15` → `0.0` | MISSED | INVALIDATES |
| numeric constant (control) | INVALIDATES | INVALIDATES |

The date-bound row is the same class as the 2014 reference-window defect the
design loop caught — the guard would not have noticed it.

Fix: exclude only the function's own docstring (by identity, not type), hash
`__defaults__`/`__kwdefaults__`, and fold the `pixi.lock` digest in as an
environment fingerprint. All six classes are now pinned as tests that fail
against the prior implementation — the falsification experiment for the guard
itself.

**Accepted cost, stated:** an error-message edit invalidates again. The existing
test that asserted the opposite was split rather than deleted
(`test_kernel_hash_ignores_comments_and_docstrings` keeps the documentation half;
`test_kernel_hash_notices_a_changed_error_message` records the inversion and why),
so a future session cannot "repair" the red by restoring the unsafe filter.

## 2. The blocker was a reproducible deadlock, not a stale handle

The handoff recorded a killed background job holding an OS handle on
`gcm_timeseries.nc`. That was the symptom. The cause:

`plot_climate_proj_timeseries` opened nine netCDFs lazily
(`open_mfdataset`) and let `xr.merge` + `to_netcdf` read them from dask's thread
pool. netCDF4/HDF5 reads take a global lock, so the write **deadlocks on win-64**.

Measured:

| Configuration | Result |
|---|---|
| threaded (default), before fix | **deadlock** — 15 min, 14 threads parked in `Wait/UserRequest`, 6.6 s CPU, no TCP connections, output file created at 12:35:52 and never written |
| `DASK_SCHEDULER=synchronous`, before fix | 42.1 s |
| threaded (default), after `.load()` | **51.5 s** (historical benchmark for this rule: 48.7 s) |

The file was **29 528 bytes at both 09:54 and 12:35** — two independent runs, two
sessions, same stall point and same byte count. The previous session attributed
the held handle to its own kill; the kill was a consequence.

Fix: `.load()` after each `open_mfdataset` in the plot script, making the reads
serial. Value-neutral, and the same idiom `get_stats_climate_proj.py` already
documents for this pathology (its comment records a ~5 h dask round-trip).
Sliced data is a few hundred KB, so eager loading costs nothing.

## 3. Cheap rungs

| Rung | Result |
|---|---|
| `pytest tests/test_series_identity.py` | 46 passed (was 37) |
| `pytest tests/test_cli.py` | 9 passed — all three Snakefiles dry-run |
| `pytest tests/` | **597 passed**, 6 skipped, 1 xfailed (was 583) |
| `snakemake -n` after the repair | 21 jobs — identical to the pre-repair DAG, confirming the repair folded into the already-pending re-derivation instead of adding a second one |

## 4. Expensive rungs

Nine series re-derived from the network, driven one model per call (three calls),
then the run completed.

**`check_baseline` → `OK - 15`.** Carries its standing provenance warning
(manifest recorded on `milestone/r07-layout`, checked from `main`; the fixture is
untracked and shared by every branch).

**`semantic_tree_diff --ref test_case/ref_wf2_pre_valuechange` → 125 compared,
1 failed, 0 missing, 9 extra.** Both differences are accounted for:

- **9 EXTRA** — the step-4b series key generation (`…_r1i1p1f1` suffix). Expected,
  but **see §4a: "expected" is not the whole story.**
- **1 FAIL** — `timeseries/gcm_timeseries.nc`: `dataset attrs {} vs {cst_…}`. The
  print order is `cur vs ref` (`semantic_tree_diff.py:539`), so **current is
  stripped and the reference carries the attrs**. This is step 4c's intended fix:
  `xr.merge` propagates global attrs from the first dataset, so the merged
  nine-series product was claiming one arbitrary series' identity
  (`cst_catalog_entry: cmip6_INM/INM-CM4-8_ssp245_{member}`, one digest, one
  region fingerprint, one source pin). `b9e8556` strips them; the reference tree
  was snapshotted at 4a, before that.

This is the **characterized diff step 4c owed** — its README lists 4c's cause as
"fail-fast + dummy-netCDF removal" — not a regression.

**The reference tree was deliberately NOT refreshed.** It is the baseline for
5a–5e, whose gates are per-cause characterized diffs; re-snapshotting now would
discard exactly what those steps compare against.

## 4a. The monthly series have ZERO gate coverage today

Found by pressing on why the 9 EXTRA entries are "fine". They are not merely
cosmetic — they mean the live artifacts were never compared. Measured:

| Tree | Series files | Names |
|---|---|---|
| `ref_wf2_pre_valuechange` | 9 | pre-4b only (`…_ssp245.nc`) |
| `test_local` | 18 | 9 pre-4b **orphans** (mtime 04:17–04:19, nothing writes them) + 9 **live** (`…_ssp245_r1i1p1f1.nc`, mtime 12:30–12:34, just re-derived) |

So `125 compared, 9 extra` means the gate compared the nine **orphans** — trivially
identical, because no rule has written them since step 4b — and classified the nine
**live** series as EXTRA, which is not a comparison. `check_baseline` does not cover
them either: its WF2 slice is 3 PNGs by size, two summary CSVs, the summary `.nc`
and the config snapshot. **No gate fingerprints a monthly series.**

Tolerable for 4b/4c, whose claims were about other files. **Fatal for 5a/5b/5c**,
whose entire gate *is* a per-cause characterized diff on these series (spherical
weighting, calendar weighting, rounding). Run today, 5a's diff would report the
changed series as EXTRA and the untouched orphans as identical — a green-looking
gate containing none of the change. The handoff sentence "'extra' entries are the
renamed series generations and are fine" is what hid this.

**Required sequence before 5a** (not before item 3, which is value-neutral):

1. Prune the 9 orphans — `dev/scripts/prune_series_cache.py` already finds them and
   defaults to a dry run; deleting is an explicit owner action. Pruning **after** a
   snapshot would bake 18 files into the new reference and preserve the muddle, so
   prune first.
2. *Then* snapshot the fresh reference tree, so ref and cur hold the same 9 live
   names and a 5a diff lands on the files 5a changes.

This is a gate-materialization finding of exactly the class the r2 review's §3.2
names: a gate specified in the plan that cannot yet do the job the plan assigns it.

## 4b. `pixi.lock` is now an attribution hazard, not only a cost

The lock digest feeds the cache key as of `3dc8fa5`. If the lock changes for any
unrelated reason during 5a–5c, every series' `cst_reducer_module_hash` moves and the
per-cause diff becomes unexplainable — destroying the very property those steps are
split to preserve.

**Freeze point, recorded so a change is detectable rather than silently mixed in:**

```
pixi.lock  sha256 dc6f27f0146c5a121239629ddfe4952db9941f2e6172c1480de736c732946288
           640501 bytes, 2026-07-30
```

Treat the lock as frozen until R8 seals. Verify with
`python -c "import hashlib,pathlib; print(hashlib.sha256(pathlib.Path('pixi.lock').read_bytes()).hexdigest())"`.

## 5. Carry-forwards

- The `ssp585` asymmetry the process review flagged as undiagnosed did **not**
  reproduce here: nine series re-derived in three calls without hitting a
  timeout, and the recorded per-series benchmarks are 108–130 s each with no
  scenario outlier. The earlier symptom is more likely the same dask/HDF5
  contention as §2 (three parallel jobs, each lazily reading a remote store) than
  a property of the `ssp585` stores. Re-check before spending a probe on it.
- Any dask-backed multi-file read followed by a write in this repo is a deadlock
  candidate on win-64. `.load()` after slicing is the established fix.
- `check_baseline` passed **trivially** in earlier steps because the workflow had
  not re-executed. It ran against freshly derived outputs here.
