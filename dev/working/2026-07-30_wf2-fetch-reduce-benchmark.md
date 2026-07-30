# Benchmark — fetch vs reduce, and the store-read asymmetry

```
Date:   2026-07-30
Gates:  r8 handoff §4 item 3 — the measurement both critiques made the
        fetch/reduce split conditional on
Probe:  dev/scripts/probe_store_read_timing.py (landed with this note)
Status: split payoff CONFIRMED and larger than estimated; the new anomaly is
        RESOLVED — root cause found and measured (§3.1)
```

Every number below is **measured** on the seed fixture's region
(`INM/INM-CM4-8`, `r1i1p1f1`, `Amon`, buffer 1.0, variables `[precip, temp]`).

---

## 1. The split is justified — and by more than the estimate

Three sources, timed in three phases:

| Phase | 3 sources | What it is |
|---|---|---|
| **open** | ssp585 alone: **1142.1 s** | `get_rasterdataset` — resolves the catalog URI glob `gs://cmip6/.../Amon/{variable}/*/*` against the remote listing, reads store metadata |
| **fetch** | **58.5 s** | `.load()` — the actual data transfer (ssp585: 18.7 s) |
| **reduce** | **0.6 s** | the monthly resample / spatial mean / round arithmetic (ssp585: 0.3 s) |
| raw on disk | **0.20 MB** | the sliced raw netCDF (ssp585: 0.07 MB) |

**A re-reduction from a local raw cache costs ~1 % of a re-derivation** (0.6 s vs
59.1 s of remote work, before counting the open). Nine sources would hold about
**0.6 MB** of raw slices — three orders of magnitude below the "single-digit MB"
the design assumed, so disk is a non-issue.

> **Corrected against the built implementation (2026-07-30, same day).** The 0.6 s
> is the *arithmetic*. A real `reduce_gcm_series` job measures **29–34 s**, because
> interpreter start, `import hydromt` and dask setup dominate — the reduction itself
> is still ~0.6 s of it. So a re-reduction of all 9 series is **~4.6 min**, against
> **~18 min** for a full re-derivation (fetch 28 s + reduce 31 s per series on the
> clean path). A 4x win on the 5a/5b/5c steps, not 100x. Cite this figure, not the
> 1 %: the probe measured a phase, the benchmark below measures a job.

The r2 review estimated "~15 min → seconds" for the remaining value-changing
steps. That is confirmed as an *understatement* of the ratio.

## 2. The design consequence the estimate missed

**The dominant cost is the open, not the transfer.** 1142 s to resolve and open one
source versus 18.7 s to move its data.

So a raw cache that still calls `get_rasterdataset` — to check freshness, to
re-read metadata, to confirm a pin — **saves nothing**. It would remove the 19 s
and keep the 1142 s.

The split must therefore be built so a reduce job with a valid raw slice makes
**zero remote calls**: freshness decided from the local slice's own attributes
against the **D12 store-index pins** already on disk, never by reopening the store.
That is a stronger requirement than "raw content digest folded into the reduce
key" as the r2 review's mitigation list stated it, and it is what makes the split
worth its coherence surface. Stated as an acceptance criterion for the step:

> With every raw slice present and valid, a full re-reduction of all 9 series
> issues no network request. Falsifier: run it with the network unavailable — it
> must succeed, not degrade.

## 3. The `ssp585` asymmetry is refuted as scenario-specific — a new anomaly replaces it

The handoff recorded "three parallel `ssp585` reads exceed a 10-minute tool
timeout" and the r2 review flagged it as undiagnosed. Three measurements now:

| Evidence | Result |
|---|---|
| Today's real snakemake run (`bf1f4a5`) | 9 series re-derived in 3 per-model calls, **none exceeding 600 s** including opens; recorded per-series benchmarks **108–130 s**, no scenario outlier |
| Probe run 1 (full mode, ssp585 last) | ssp585 **open 1142 s** |
| Probe run 2 (open-only, **ssp245 first**) | ssp245 open **> 630 s**, killed before completing |

The slow open happened to `ssp245` when it went first and to `ssp585` when it went
last, so **the scenario is not the variable**. Nor is it purely cold-start.

What is *not* explained: the same call is 5–20× slower in the probe than inside
today's snakemake jobs, with an apparently identical argument set (`bbox`,
`buffer`, `time_range`, `variables` all mirrored from
`get_stats_climate_proj.py:250-258`). Candidates not yet separated: GCS listing
latency varying over the day, the anonymous-credential fallback the probe logs
(`Could not determine bucket type for bucket name cmip6 … falling back to
GCSFileSystem`), and per-process listing caches.

**Hypothesis, with its settling observation:** instrument the open *inside the
rule* — a `log_row` bracketing `get_rasterdataset` — and compare against
`probe_store_read_timing.py --mode open` in the same hour. If the in-rule open is
also minutes, the recorded 108–130 s benchmarks are measuring something else and
the fan-out is hiding the cost; if it stays seconds, the probe path differs and the
probe is the artifact. Until then, **cite neither number as "the" cost of a store
open.**

Note against over-reading run 1: its first two rows were lost to a `Select-Object
-Last 18` in the invocation, not to the probe. The totals survived because they
print last. The probe now flushes per row for exactly this reason.

## 3.1 Root cause: gcsfs's experimental extended filesystem, on by default

The settling observation of §3 was run as a controlled A/B on 2026-07-30 (probe
`--mode open`, `INM/INM-CM4-8` `ssp245` `r1i1p1f1`, one source per process).
Run A went **first** on purpose, so any warming worked *against* the hypothesis:

| Run | `GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT` | open | bucket-type warnings |
|---|---|---|---|
| A | `false` | **57.7 s** (completed) | 0 |
| B | *unset* (gcsfs default) | **> 836 s**, killed before completing | **266 and climbing** |

Read the two numbers carefully: A's is a completed `open_s`; B never printed one.
B's figure is 869 s of wall clock minus ~33 s of interpreter start, hydromt import
and catalog parse (log: start 13:04:01Z, catalog parsed 13:04:12Z, `Reading …`
13:04:33Z). **Ratio ≥ 14×, and a lower bound** — B was killed, not finished.

The mechanism, read from the pinned `gcsfs 2026.4.0` in `.pixi/envs/default`:

* `gcsfs/__init__.py:18` — `os.getenv("GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT", "true")`.
  The experimental `ExtendedGcsFileSystem` is **on by default**, and the switch is
  read at **import** time, so it must be set before anything imports gcsfs —
  including transitively, via `import hydromt`.
* `extended_gcsfs.py:164-195` — every operation calls `_lookup_bucket_type`, a
  `get_storage_layout` control-plane RPC. Under `token="anon"` the storage-control
  client does its own ADC lookup, finds no default credentials, and raises:
  `Could not determine bucket type for bucket name cmip6: Your default credentials
  were not found ... falling back to GCSFileSystem`.
* `extended_gcsfs.py:155-157` — `# Dont cache UNKNOWN type`. The failure is
  therefore **never cached** and repeats on every call, including `_list_objects`
  (line 1443, i.e. the `/*/*` glob resolution) and `_open` (line 214).

Measured throughput of the repetition: **0.31 warnings/s**, sustained over a 45 s
window. That is an aggregate rate, not a per-call cost — fsspec is async and the
open touches two variables, so lookups overlap; do **not** read it as "3.2 s per
lookup". What the measurement supports is the shape of the pathology: volume ×
no-caching. Whether an individual lookup approaches
`STORAGE_CONTROL_RPC_TIMEOUT` (30 s) was not measured.

Why the probe looked slower than the rule: `get_stats_climate_proj.py:15` sets the
variable; `dev/scripts/probe_store_read_timing.py` did not. That is the entire
"5-20× slower in the probe" discrepancy — same code, opposite side of the switch.
GCS latency, time of day and `ssp585` are all exonerated.

**Consequence for the split.** `blueearth_cst/projections/fetch_gcm_raw.py` does
not set the variable either. As WF2's only remote-opening module it would inherit
run B's path, i.e. the split would be built on the slow side of a one-line switch.

**Residual, separate from this one.** Run A's 57.7 s is *not* fast. With the
pathology removed, resolving `{grid_label}/{version}` is still a real cost, and
`config/catalogs/cmip6_store_index.json` already records the pin that would
remove it. Two independent fixes, not one.

## 4. Carry-forwards

- The split is **cleared to build**, with §2's zero-network acceptance criterion
  added to the mitigation list in r2 §4.
- `--mode open` is the cheap re-measurement when asking "is the store slow
  today?" before attributing a slow run to code. It must be run **with**
  `GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT=false`, or it measures §3.1 instead of the
  store.
- The 10-minute-timeout workaround in the handoff §5 was justified by "cold glob
  resolution can take 10–20 minutes for any source". §3.1 shows that figure was an
  artifact of the unset switch; re-derive the budget from run A's 57.7 s, not from
  the 1142 s.
- **§1 and §2 were measured on the unset switch.** Their conclusion survives — a
  re-reduction is still ~100× cheaper than a re-derivation (0.6 s local vs ~77 s
  remote at run A's open) — but every absolute figure in them, 1142 s above all,
  is inflated by §3.1's bug. Restate them against a corrected open, or a later
  reader will find §1 and §3.1 in contradiction.
- Two follow-ups fall out of §3.1, independent of each other and of very different
  size: declaring the switch (one line, ≥14× measured) and replacing the `/*/*`
  glob with the `cmip6_store_index.json` pin (a catalog-shape design question —
  the index pins per (entry, member, variable), the catalog entry is per
  (model, scenario), and `pin_stores` exists because the two disagree; payoff
  unmeasured).
