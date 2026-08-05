Task Brief — R9 gate closure: the run the landing gate is waiting on

Closes the outstanding items in [`landing-gate.md`](landing-gate.md).

**Step 4 is DONE — the shared-store falsifier passed on 2026-08-05**, results in
`landing-gate.md` § *Gate closure*. It is kept below unchanged, because the
assertions and the reasoning behind the discriminator are what a re-run would
need. **Three items remain, and one run covers all three: steps 1, 2 and 3.**

**Owner action, primary checkout.** `AGENTS.md` reserves pipeline runs for the
one checkout: Snakemake keeps "what is up to date" in `.snakemake/` under the
*working directory*, so two checkouts driving one `project_dir` disagree — 12
jobs planned from one, 2 from the other, measured 2026-08-02 — and each holds its
own workdir lock while writing the same outputs.

---

### What each check can actually prove

Read this before running. Two of the four items do **not** mean what the master
brief assumed when it was written, because the tree they were to run against has
moved since.

**The pre-R9 reference is gone.** P2's whole-tree diff used
`test_case/test_local` as its pre-migration reference and got **zero numeric,
zero structure** failures over 161 files. Runs since have overwritten that tree
with post-P3 output. So the pre→post comparison is **already done and cannot be
repeated** — and does not need to be.

What the whole-tree diff can still prove, and what the gate now needs from it, is
different: *does a **P4-inclusive** run reproduce the P3-era values, and does its
path set still map cleanly?* That is a same-era comparison, so it runs with
`--no-path-map`.

**`check_baseline check` is the weaker of the two value checks here, not the
stronger one.** Its manifest keys already carry the R9 tree
(`test_case/test_local/data/climate/projections/cmip6/plots/…`), so P3's
re-record was against the migrated layout — good. But the keys embed
`test_case/test_local`, so `check --project-dir <fresh>` finds nothing and every
target reports missing. It must run against `test_case/test_local`, which the
fresh run does not touch — making it a *"nothing drifted on disk"* sanity check,
not a verdict on the new run. The 161-file semantic diff is the value gate; the
manifest's thin `TARGETS` list is the tripwire.

**Declare P2's F4 before you look at the output.** Two fetches of the same CMIP6
slice produce different global attrs — `variable_id` is `tas` on one side and
`pr` on the other — while every value, coordinate and dimension matches. It made
29 files noisy in P2's diff. Pre-existing, unrelated to R9, and it will recur.
Attr-only failures under `cmip6/raw/` and `cmip6/scalar/` are **not** a P4
regression.

---

### Step 0 — position the checkout

```powershell
cd ~/workspace/blueearth_cst          # the PRIMARY checkout, currently on main @ c9990a5
git status --short                    # must be clean
git checkout milestone/r09-project-tree
git log --oneline -1                  # expect the P5 merge, 4ddcce2
```

Restore `main` when finished. Nothing in this plan commits.

### Step 1 — the run (items 6 and 7)

Into a **fresh** `project_dir`, so the existing tree survives as the comparand
and the mixed-era orphans stay out of the way.

```powershell
$RUN = "$env:TEMP\r09_gate\post_p4"
New-Item -ItemType Directory -Force $RUN
$CFG = "$env:TEMP\r09_gate\seed.yml"
# copy config/workflows/snake_config_model_test.yml with project.project_dir -> $RUN
```

Then the three workflows **in order**, from inside `pixi shell`:

```powershell
snakemake all -c 3 -s Snakefile_model_creation      --configfile $CFG
snakemake all -c 3 -s Snakefile_climate_projections --configfile $CFG --keep-going
snakemake all -c 3 -s Snakefile_climate_experiment  --configfile $CFG
```

`-c 3` deliberately: WF3 fans out over `(rlz, cst)`, and a serial run cannot
exercise the concurrency this milestone's falsifier is about.

**The P4 acceptance check — do this first, before any diff.** Three files must
exist that no previous run produced:

```powershell
Get-ChildItem $RUN\experiments\experiment\config\
#   model_reference.yml     rule 3.01c
#   experiment.yml          rule 3.01e
Get-ChildItem $RUN\experiments\experiment\.model_reference_ok   # rule 3.01d
```

Their **absence** is what proved the last run predated P4. If any is missing the
run did not exercise P4 and the gate is no better off — stop and report rather
than proceeding to the diffs.

**Then prove the guard is load-bearing, which the run alone does not.** A guard
that never fires is indistinguishable from no guard:

```powershell
# perturb the model AFTER the experiment is defined, then re-run WF3
Add-Content $RUN\models\hydrology\wflow\wflow_sbm.toml "`n# drift"
snakemake all -c 3 -s Snakefile_climate_experiment --configfile $CFG
```

Expected: rule 3.01d fails **before** any member simulates, naming the changed
file. That is P4's end-to-end falsifier, outstanding since P4. Restore the TOML
and re-run to green.

### Step 2 — the whole-tree diff (item 7)

Same era on both sides, so no path map:

```powershell
python dev/scripts/semantic_tree_diff.py --no-path-map `
    --ref test_case/test_local --cur $RUN `
    --ref-token test_case/test_local
```

**Read the three residual classes differently:**

| Class | Meaning |
| --- | --- |
| NUMERIC / STRUCTURE failures | a real regression — the gate fails |
| attr-only failures under `cmip6/` | P2's F4, expected, not a finding |
| **MISSING** (in ref, not in the fresh run) | **the orphan list, for free.** `hydrology_model/`, `spatial/`, `climate_historical/`, `climate_projections/`, and inside the experiment `hydrology_runs/`, `indicators/`, `weather_generator/`, `data_catalog_climate_experiment.yml`. A clean run does not produce them; their appearing here is the mixed-era tree being enumerated, which is what P1's F5 said no comparator could do from one tree alone |
| EXTRA (in the fresh run, not in ref) | P4's three new files, and nothing else |

Capture the MISSING list. It is the input to whoever sweeps the fixture tree, and
it is evidence for the F5 finding rather than a chore.

Then re-run P1's falsifier against a **P4-inclusive** path set, which has never
been done:

```powershell
python dev/scripts/snapshot_project_tree.py --configfile $CFG --check-map
```

Expect **zero unmapped**. A P4 file appearing as unmapped means the map has a gap
the declared and observed tiers both predate.

### Step 3 — the concurrency falsifier's missing half (item 8)

It passes today: 12 member logs, 12 correctly attributed, 0 stray `log.txt`.
The half never done is showing it **fails** with `path_log` unset.

```powershell
# temporarily remove the "logging.path_log" entry from
# blueearth_cst/experiment/downscale_climate_forcing.py's setup_config data block
snakemake all -c 3 -s Snakefile_climate_experiment --configfile $CFG --forceall
pixi run pytest tests/test_wflow_log_attribution.py
```

**Expected: RED**, and specifically on *attribution*, not on a file count — the
members should collide into one `log.txt` beside their shared `config/`
directory. A green here means the test cannot detect the condition it exists for,
which is a worse outcome than a red and must be reported as such.

`git checkout -- blueearth_cst/experiment/downscale_climate_forcing.py` after,
and confirm the module is clean before anything else runs.

### Step 4 — the shared-store falsifier (item 9) — DONE 2026-08-05, PASSED

Never run, in any phase. Cheapest of the four and needs no execution — a dry-run
job count answers it.

```powershell
# a second config: same clim_historical + historical_window, different experiment_name
$CFG2 = "$env:TEMP\r09_gate\seed_exp_b.yml"
snakemake all -c 1 -s Snakefile_climate_experiment --configfile $CFG2 --dry-run
```

Two assertions:

1. the shared climate-store rule schedules **zero** jobs for the second
   experiment — the store key resolved to the same directory and the extraction
   is reused;
2. the store rule's input set is **byte-identical** between the two configs.

Assertion 2 is the one that matters. A zero job count alone is also what you get
if the rule is simply not in the second DAG, so the input comparison is what
distinguishes *reuse* from *absence*.

This guards the reason the store key exists: design tree
`data/climate/historical/<key>/` keeps the key as a **cache key**, not as
multi-window support — *"two experiments sharing `clim_historical` +
`historical_window` resolve to the same dir and reuse the extraction"*.

### Step 5 — the tripwire

```powershell
python dev/scripts/check_baseline.py check
```

Against `test_case/test_local`, which nothing above touched. Expect green:
P4 and P5 touch no pinned artifact. **A red here is itself a finding** — it would
mean the tree drifted between P3's re-record and now, by something outside this
milestone's account of itself.

**Do not re-record.** Gate 2 closed at P3 and the master brief says exactly once.
If a re-record ever looks necessary, that is a gate to re-open, not a step to
take.

---

### Acceptance criteria

- `model_reference.yml`, `experiment.yml` and `.model_reference_ok` exist in the
  fresh run — P4's rules executed.
- The drift guard **fired** on a perturbed model, before any member simulated.
- Whole-tree diff: zero NUMERIC and zero STRUCTURE failures; every residual
  classed, with the MISSING set recorded as the orphan inventory.
- `--check-map` zero unmapped on a P4-inclusive path set.
- The log-attribution falsifier shown **RED** with `path_log` unset, and the
  module restored.
- The shared-store falsifier: zero store jobs **and** byte-identical inputs.
- `check_baseline check` green, with no re-record.

### What a failure means

Each of these is a different decision, so do not collapse them into "the gate
failed":

| Failure | Reading |
| --- | --- |
| P4 artifacts absent | the run did not exercise P4 — a configuration problem, retry |
| guard did not fire | P4's central claim is unproven — blocks the merge |
| NUMERIC/STRUCTURE failures | a real regression somewhere in P2–P4 — bisect by phase |
| falsifier green with `path_log` unset | the *test* is defective, not the code |
| shared-store rule absent rather than skipped | a DAG problem, and the falsifier's design assumed the wrong mechanism |
| `check_baseline` red | drift from outside R9's account of itself — investigate before anything else |

### Output

Append the results to [`landing-gate.md`](landing-gate.md) as a *Gate closure*
section — the four items with their evidence, the orphan inventory from step 2,
and any finding above. Do not edit the phase reports; they are records of their
phases.
