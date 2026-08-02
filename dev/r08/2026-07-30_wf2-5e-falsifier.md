# Falsifier — step 5e (three changes, one migration row)

```
Written: 2026-07-30, BEFORE any 5e code exists
Design:  §5.4 D1 (clip), §5.5 (variable spec), OQ-12 (rename), §8 row 5e
Ref:     test_case/ref_wf2_pre_5e  (post-5d; CLEAN against itself, 126 files)
```

5e bundles three changes that share one property: they all move the **config
contract**, so they share one manifest re-record. They are otherwise independent
and land as **three commits**, per the standing per-cause rule.

| | Change | Seed-visible? |
|---|---|---|
| i | `save_grids` → `save_gridded`, failing loud on the old key (OQ-12) | config only |
| ii | Reference-window clip + per-condition warnings + alignment check (D1) | **no** — `[1990, 2010]` needs no clip |
| iii | Variable spec: `canonical` / `change` / `units` per variable (§5.5) | config shape |

§8: **Output-neutral on the seed**, manifest-unclean via the config target.

## Scope correction the design does not state

D1 names three surfacing sites: stderr, `provenance.json`, `report.md`. **Neither
of the latter two exists yet** — `provenance.json` arrives at 6a, the report at 7.
So 5e implements stderr plus a durable record, and the durable record goes to
`composition.csv`, which has carried `reference_window_nominal` / `_effective`
since 4d. Recorded so 6a knows the fields are already populated rather than
inventing them again.

## K1 — the old key must FAIL LOUD, not be ignored

**Falsified if** a config carrying `save_grids` runs. Silently ignoring it is the
worst outcome: a user who set `save_grids: true` would get `false` behaviour and
no signal. The raise must name the new key.

**Also falsified if** it raises at *run* time rather than DAG build — a config
error should not require scheduling jobs to discover.

## K2 — the new key must behave identically

**Falsified if** `save_gridded: false` and the old `save_grids: false` produce
different DAGs. The rename is a rename.

## K3 — the clip must clip, and say so

Effective reference = `requested ∩ [source start, 2014-12-31]`.

**Falsified if** a requested reference overrunning 2014-12-31 is used unclipped,
or is clipped **silently**. The stderr warning must name requested *and* effective
— "clipped" alone does not let a reader judge the damage.

## K4 — a window entirely after 2014-12-31 must RAISE

The one exception to "never raises". There is nothing to clip to.

**Falsified if** it produces an empty reference and continues, or clips to a
zero-length window.

## K5 — the alignment difference must NOT warn on stderr by default

This is the subtle one, and the seed config is the reason it exists: effective
reference `[1990, 2010]` versus `shared.historical_window` `2000–2020` **differ**,
so a stderr warning here would fire on **100 % of runs** and be filtered out —
"a signal that fires on every run is a signal nobody reads".

**Falsified if** a default run emits an alignment warning to stderr.

**Also falsified if** the difference is not recorded durably at all — silence and
absence are different, and D1 asks for the disclaimer to carry it.

**Promotion case:** stderr *is* warranted when the two windows would have been
equal but for the clip — the user plausibly intended alignment and did not get it.
**Falsified if** that case is also silent.

## K6 — a short effective window must warn

Effective length < 20 years → stderr, per D1. The seed is exactly 20
(`2010 − 1990`), so it must **not** warn. **Falsified if** the seed warns — an
off-by-one on a boundary the fixture sits exactly on.

## K7 — the variable spec must be read, not inferred

`canonical` (`rate` | `state`), `change` (`relative` | `absolute`) and `units`
per variable. Today stage B branches on the literal string `"precip"`.

**Falsified if** a variable named something other than `precip` with
`change: relative` is treated as absolute — i.e. if the name still drives the
arithmetic after the spec exists. That is the whole point of §5.5: "Nothing infers
anything from a name."

## K8 — values must NOT move on the seed

§8 says output-neutral. The seed needs no clip, and the gridded default is
unchanged.

**Falsified if** any `raw/`, `series/`, `timeseries/` or summary **value** moves.
The config snapshot is the only manifest target that may change.

## K9 — the config snapshot MUST move

The inverse, and the reason this step is manifest-unclean by design.

**Falsified if** the config target's sha256 is unchanged — that would mean the
rename and the variable spec never reached the shipped configs, and K1/K7 are
passing against a config nobody uses.

## Order of work

1. Commit i — the rename. Old key raises; update shipped configs.
2. Commit ii — the clip, warnings, alignment. Unit tests over synthetic windows;
   the seed exercises none of the warning paths, which is why they need tests.
3. Commit iii — the variable spec.
4. One gate at the end: K8 by diff, K9 by manifest, then re-record and snapshot
   `ref_wf2_pre_5f`.

---

## Outcome — 2026-07-30, all nine discharged across three commits

| | Result |
|---|---|
| K1 | old key raises at DAG build, naming both keys (`-n` never runs a job, so a nonzero exit there is necessarily parse-time) |
| K2 | new key: identical DAG; only `copy_config` re-ran |
| K3/K4 | 16 unit tests; clip names requested AND effective; entirely-after raises; ending exactly at 2014 does not |
| K5 | seed emits **no** alignment warning; record shows `reference_alignment=differs`; promotion case covered |
| K6 | exactly 20 years silent, 19 warns |
| K7 | 14 unit tests; a `rainfall` variable declared `relative` IS relative; a `precip` declared `absolute` IS absolute |
| K8 | 126 compared, **1 failed** — no value moved |
| K9 | that one file is the config snapshot |
| `check_baseline` | FAILED on the config target only, re-recorded OK 15/15 |

### The mistake worth recording

5e-iii first folded the **whole** variable spec into the digest components, and
the dry-run answered immediately: **9 `fetch_gcm_raw` jobs**. `canonical`, `units`
and `change` cannot change a cached byte — they are read by stage B, which has no
cache — so a `change: relative → absolute` edit would have re-fetched every slice
over the network for arithmetic that touches no stored value.

That is the same over-invalidation the design faulted file-level hashing for at
4c, arrived at from the opposite direction: 4c hashed too much of the *code*, this
hashed too much of the *config*. The digest now carries the source names only —
what was actually fetched — and the semantic fields reach stage B alone.

Worth noting the dry-run caught it in seconds, before any network cost, purely
because K-J7's "which jobs get scheduled" question is asked before every run.

### A second variable family the fixture never shows

Migrating the configs surfaced `snake_config_projections_cmip5_full.yml`, which
declares `temp_min` / `temp_max` — renamed from `tmin`/`tmax` with a −273.15
offset. They needed specs (`state`, degC, absolute) that the seed's two variables
would never have prompted. Under the old list form they were silently treated as
absolute *because they are not named `precip`*, which happened to be right; under
the spec it is stated.
