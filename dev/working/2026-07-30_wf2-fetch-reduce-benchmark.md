# Benchmark — fetch vs reduce, and the store-read asymmetry

```
Date:   2026-07-30
Gates:  r8 handoff §4 item 3 — the measurement both critiques made the
        fetch/reduce split conditional on
Probe:  dev/scripts/probe_store_read_timing.py (landed with this note)
Status: split payoff CONFIRMED and larger than estimated; one new anomaly OPEN
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

## 4. Carry-forwards

- The split is **cleared to build**, with §2's zero-network acceptance criterion
  added to the mitigation list in r2 §4.
- `--mode open` is the cheap re-measurement when asking "is the store slow
  today?" before attributing a slow run to code.
- The 10-minute-timeout workaround in the handoff §5 stays, but its *reason* is
  now "cold glob resolution can take 10–20 minutes for any source", not "ssp585 is
  slow".
