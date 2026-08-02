# P3-3 — Performance-passes design: consolidated review record

Durable audit trail of the `p33-performance` design-review-loop run
(2026-07-24). The run directory was pruned at landing; this record preserves
the verdicts, the aggregated internal index, the verbatim internal lens
reviews, both verbatim external rounds, the arbitration record, the final
disposition ledger, and the process observations. Accepted design:
`performance-passes-design.md` (v4, accepted at G2 under arbitration
authority). Scoping authority: `performance-passes-intake.md` (landed
pre-run). Probe evidence: `probes/`.

## Run summary

| Item | Value |
| --- | --- |
| Run | p33-performance (full variant) |
| Author binding | cst-architect |
| Gates | G1 approved 2026-07-24 (with the OQ-2 ruling: batching first, sysimage deferred to commit-2 evidence); G2 approved 2026-07-24 under arbitration authority |
| Versions | v1 (probe-grounded draft) -> v2 (post-panel) -> v3 (post-ext-r1, Fable) -> v4 (stage-6a arbitration fix, Fable; ACCEPTED) |
| External rounds | r1 revise (3 major) on v2; r2 REJECT (1 blocking) on v3 -> round cap -> arbitration |
| Arbitration | ext2-001 ruled ACCEPTED, FIX REQUIRED (2026-07-24); fix probe-verified pre-ruling by the driver; 6a scope-check clean (12 hunks, arbitrated sections only) |
| Ledger closure | 22/22 accepted (18 internal + 3 ext-r1 + 1 ext-r2); none rejected or deferred |
| Dispatches | opus: 5, fable: 3 (r2 revision, 6a revision — both external-re-raise escalations), external GPT: 3 calls (1 transport failure recovered) |
| Verdict table | risk: revise 0/3/3 - architecture: revise 1/3/2 - repo-fit: revise 0/1/5 - ext r1: revise 0/3/0 - ext r2: reject 1/0/0 |

---

## Internal review index (driver aggregation)

# Internal review index — p33-performance (panel on design-v1.md)

Driver aggregation only; every ID/severity/text lives verbatim in the
per-lens files. Nothing deleted or re-graded.

## Verdicts

| Lens | Verdict | blocking | major | minor |
| --- | --- | --- | --- | --- |
| risk (`critical-thinker`) | revise | 0 | 3 (risk-1..3) | 3 (risk-4..6) |
| architecture (`cst-architect`) | revise | 1 (arch-1) | 3 (arch-2..4) | 2 (arch-5..6) |
| repo-fit (`python-engineer`) | revise | 0 | 1 (repo-1) | 5 (repo-2..6) |

Totals: 1 blocking / 7 major / 10 minor. All name `doc_version: design-v1.md`;
consistency rule satisfied.

## Groups

- **A — G1 gate-ruling inversion (BLOCKING arch-1; majors risk-2, repo-1).**
  One root: v1 was drafted before the G1 OQ-2 ruling and still recommends
  sysimage as the default (§6.2/§6.5/§9/OQ-2), while the recorded ruling
  fixes batching-first with sysimage deferred to commit-2 evidence. Rewrite
  §6.5/§9/OQ-2 batching-first; keep the safety analysis as deferral
  rationale, not recommendation. risk-2 adds: because batching now leads,
  the batching path OWNS the C5 degradation (group B) — the design cannot
  lean on "prefer sysimage" to soften it.
- **B — failure isolation is degraded, not preserved (major risk-1).**
  Snakemake removes a failed job's present outputs, so one bad cst deletes
  B−1 completed batch-mates and blocks rule 3.11 for the sweep. State C5 as
  DEGRADED with blast-radius B (or specify + verify a persistence mechanism
  — `--keep-incomplete` interactions verified, not assumed).
- **C — value-identity discipline for the batching lever (majors arch-3,
  risk-3).** Warm-session-vs-cold-process byte identity is an UNTESTED
  assumption (the 84 s warm discount proves execution differs); the P3-1
  evidence covers per-process runs only. And the ADR-0001 immaterial branch
  as written could launder a batching-induced drift. Fix set: reword §8/OQ-3
  (first evidence = commit-2 gate-1); add the discriminating rule — any diff
  correlated with batching (present batched, absent per-process on identical
  inputs) BLOCKS the commit; ADR-0001 triage admissible only after a
  per-process re-run reproduces the diff.
- **D — scaling-model formula wrong (major arch-2).** The boxed
  `wall_batched` does not generate its own table; the correct warm-aware
  form is `ceil(⌈K/B⌉/p) × (F + S_cold + (B−1)·S_warm)` (verified to
  reproduce all four rows). Replace, and re-derive the table from it.
- **E — batched-rule construct under-specified (major arch-4; minor
  repo-3).** The batch_id-wildcard + input-function + computed per-batch
  output-list construct has NO in-repo precedent (export_wflow_results is a
  fixed-output expand aggregator); add a minimal worked sketch (parse-time
  partition dict, output list comprehension, input function form; confirm no
  checkpoint needed) so the task-brief cannot pick an inexpressible shape.
- **F — minors.** arch-5 (K = RLZ_NUM × (ST_NUM + [run_historical]), not
  unconditional +1); arch-6 (state explicitly that benchmark/log
  row-collapse is invisible to both value-identity gates — excluded dirs +
  unmanifested); risk-4 (soften n=1 "firm/decisive" language; repeats if
  cheap); risk-5 (disk ceiling must count BOTH temp classes:
  p×B×(forcing+outstates)); risk-6 (sysimage staleness keyed on full
  Manifest + Julia version, not the Wflow entry — spec note for the deferred
  lever); repo-2 (attribute per-cst CSV identity to gates 1+3, not
  check_baseline); repo-4 (benchmark rows are per-JOB already); repo-5
  (cite the landed p31 evidence + show the --ref/--cur CLI); repo-6
  (`-c N` comes from the invocation, not the profile).

## Panel-verified positives (do not re-litigate)

Rule-graph walk confirms the batched rule keeps 3.11's input contract; HM-7
reduction is filename-keyed (batch order cannot corrupt the gauge identity);
P3-2b validators read exact on-disk paths → byte-compat holds under any DAG
shape; the wall-at-cores metric correction is sound; threads reasoning
consistent; lever-D drop evidence-backed; `Wflow.run(tomlpath)` resolves at
Wflow.jl:250 so the batch driver mechanism is sound; measurement tooling
implementable on this machine.


---

## Internal lens review — risk (verbatim)

```yaml
verdict: revise
doc_version: design-v1.md
findings:
  - id: risk-1
    severity: major
    section: "6.1 Lever A — failure isolation (C5); 6.5 lever ranking; 11 Consequences"
    finding: >
      The batching failure-isolation claim is mislabeled and the C5 decision criterion is degraded,
      not preserved. The §6.1 driver catches a failing cst, logs it, and does NOT rethrow, so the
      batch process exits 0 with that cst's output CSV absent. The rule then trips
      MissingOutputException and is marked FAILED. Snakemake's default behavior on a failed job is to
      REMOVE that job's present output files (absent `--keep-incomplete`) — including the CSVs the
      driver successfully produced for the batch-mates. So the completed batch-mates are deleted and
      the entire batch must re-run. The design's assertion "the isolation is that the *other* csts in
      the batch still complete" (§6.1) and §6.5's "Failure isolation free ... as today" are true only
      at compute level and false at persistence level. Contrast with today: each cst is a separate
      job, so a failing cst's failure deletes only its own output; the siblings persist.
    rationale: >
      Observable consequence: one bad cst in a batch of B causes Snakemake to discard the B−1
      completed sibling CSVs and re-run the whole batch. Downstream, rule 3.11 export_wflow_results
      inputs `expand` over ALL rlz×cst CSVs, so a single failed batch blocks Qstats/basin for the
      WHOLE sweep until that batch re-runs — whereas today only the one failing cst blocks. This is a
      direct violation of decision criterion C5 ("one bad cst must not corrupt/skip others"), which the
      design lists among its own criteria and claims to hold. §11 frames it merely as a "re-run-cost
      regression," understating it: Snakemake actively deletes completed work, it is not a granularity
      preference.
    suggested_fix: >
      State C5 as DEGRADED under batching, not preserved. Quantify: failure blast-radius grows from 1
      cst to B csts (deleted + re-run). Either (a) accept and document the degradation explicitly in
      the ranking (it strengthens the case against batching-first), or (b) specify a mechanism that
      persists sibling outputs across a batch failure — e.g. mark the failing cst's CSV as the only
      missing output while the driver still writes siblings AND pass `--keep-incomplete` in the wf3
      profile, or split the output declaration so Snakemake does not treat sibling CSVs as removable on
      the batch's failure. Note that (b) interacts with `--keep-going` semantics and must be verified,
      not assumed.
  - id: risk-2
    severity: major
    section: "9 Commit plan; 6.2/6.5 lever ranking"
    finding: >
      The commit plan inverts the fixed gate ruling. The panel gate ruling (fixed, 2026-07-24) states
      the PackageCompiler sysimage is DEFERRED to commit-2 evidence and batching proceeds first. But
      §9 commit 2 = "sysimage (if approved) OR batching per OQ-2" — a pick-one-now choice, and §6.2/§6.5
      recommend sysimage as the PREFERRED lever with batching as the reject-fallback. That structure
      does not encode "batching first, sysimage later on evidence"; it encodes "choose the recommended
      sysimage now if the user approves." The design therefore does not deliver the batching-first
      anchor.
    rationale: >
      Observable consequence: if the design is accepted as written and the user approves the
      dependency at OQ-2, commit 2 lands sysimage, contradicting the gate that mandates batching lead
      and sysimage follow on measured commit-2 evidence. This is a fails-to-deliver-anchor defect, not
      a relitigation of the gate: it is the opposite direction (the design under-privileges batching
      relative to what the gate requires). It also couples to risk-1 — because the gate mandates
      batching-first, the batching-first path OWNS the C5 degradation; the design cannot lean on
      "prefer sysimage" to make the C5 cost disappear.
    suggested_fix: >
      Restructure the commit plan so commit 2 is batching (the batching-first path the gate fixes),
      with its before/after wall as the commit-2 evidence, and sysimage becomes a later, separately
      evidence-gated commit. Keep the sysimage specification and its recommendation framing, but
      relabel it as the deferred candidate the gate names, not as the commit-2 default.
  - id: risk-3
    severity: major
    section: "8 Value-identity proof plan (gate 2); 10 Validation plan"
    finding: >
      The value-identity acceptance standard pre-arms an escape hatch that can launder a real
      batching-induced drift as immaterial. Gate 2 requires "byte-identical Qstats.csv/basin.csv," but
      §8 simultaneously invokes the ADR-0001 "immaterial sub-tolerance" branch and check_baseline uses
      a TOLERANCE comparator (per the baseline-manifest-coverage memory), and §8 says a sub-tolerance
      wf1 discharge move "may surface in a re-run Qstats ... handled by its step-7 recovery path, not a
      P3-3 regression." A batching change that induced a small nondeterministic numeric drift could be
      attributed to the known wf1-restoration move rather than to the lever.
    rationale: >
      Observable consequence: a Qstats byte-diff that is actually caused by batching (e.g. a
      process-boundary or ordering effect the design assumes is absent) is indistinguishable, under the
      stated triage, from the pre-existing immaterial wf1 move — and the design's own text supplies the
      "not a P3-3 regression" framing to dismiss it. The strict value-identity anchor (C2, "absolute")
      is then not actually enforced for the one change class P3-3 introduces.
    suggested_fix: >
      Add an explicit rule: any Qstats/basin diff that CORRELATES with the batching change (appears
      only in the batched tree, absent in a per-process re-run of the same inputs) must be treated as a
      lever regression and BLOCK the commit — the ADR-0001-immaterial branch is admissible only when
      the diff is reproduced by a per-process (non-lever) re-run, proving it is the wf1 move and not
      batching. Make the semantic-diff gate (gate 1) run per-process-vs-batched on identical inputs as
      the discriminating test.
  - id: risk-4
    severity: minor
    section: "5.2 Probe 1 / 1c / 1d; 5.5 scaling model"
    finding: >
      The core cost decomposition is single-sample, mixed-condition Windows wall-clock stated at
      certainty language ("firm," "decisive," "ground truth"). Each headline number (135 s fixed, 208 s
      cold sim, 124 s warm sim, 343 s fresh-process) is n=1 with no variance reported; probe 1c was
      measured with the run_logged tee, probe 1d without it (the design itself estimates the tee at
      ~42 s), and idle-single-process vs `-c 3`-concurrent conditions are mixed. Windows wall-clock at
      this granularity carries scheduler/AV/thermal noise that a single sample cannot bound.
    rationale: >
      Observable consequence: the §5.5 regime split (39 % vs 4 %), the §6.5 ranking, and the
      illustrative sweep table all currently rest on these single samples, before commit 1 re-measures.
      The design does hedge this heavily (labels the table "ILLUSTRATIVE," defers to commit 2), which is
      why this is minor rather than major — but the confidence-marking language ("firm/decisive/ground
      truth") overstates n=1 evidence and should be softened so a reader does not treat the
      decomposition as settled before commit 1.
    suggested_fix: >
      Downgrade "firm/decisive/ground truth" to "single-sample point estimate, to be confirmed by
      commit 1." If cheap, take 2–3 repeats of the 343 s fresh-process run and the warm/cold pair and
      report a range; a single anomalous sample would otherwise silently move the 39 % fixed fraction
      that the whole regime argument hinges on.
  - id: risk-5
    severity: minor
    section: "6.1 Lever A — batch partitioning; 11 Consequences (temp disk)"
    finding: >
      The disk-ceiling analysis omits that rule 3.09 (downscale) produces the per-cst forcing NC as a
      `temp()` output feeding rule 3.10, so batching rule 3.10 does not merely hold B forcing NCs — it
      changes WHEN the upstream `temp()` forcing files can be reclaimed. Snakemake deletes a `temp()`
      file once all consuming jobs finish; batching makes one batch job the consumer of B forcing NCs,
      so all B must persist until the batch completes, and with p concurrent batches the peak is
      p×B×forcing (the design states this) — but it does not note that the downscale outputs
      (inmaps_rlz_*.nc) and the run_wflow output states (outstates_*.nc, also `temp()`) are BOTH held,
      roughly doubling the resident-temp footprint versus the design's forcing-only accounting.
    rationale: >
      Observable consequence: peak `temp()` disk under batching is understated if only forcing NCs are
      counted; the outstates NCs (temp, per-cst) are also resident until each batch finishes. On a
      large RLZ_NUM×ST_NUM production sweep this tightens the binding disk ceiling further and pushes B
      smaller than the forcing-only cap implies — reducing batching's achievable win exactly where the
      design already concedes the win is marginal (regime 2).
    suggested_fix: >
      Extend the disk-ceiling formula to include both `temp()` classes held per batch (forcing +
      output states), i.e. peak ≈ p×B×(forcing_size + state_size), and re-derive the B cap against
      that. Confirm the outstates temp reclamation timing under the batched rule.
  - id: risk-6
    severity: minor
    section: "6.2 Lever B — sysimage staleness"
    finding: >
      The sysimage staleness gate is specified as keyed on "the Wflow Manifest entry," but Julia
      recompilation correctness depends on the FULL dependency closure baked into the sysimage
      (Wflow's transitive deps, Julia version 1.11.7, and the project Manifest as a whole), not the
      single Wflow Manifest line. A dep bump that changes a Wflow dependency without changing the Wflow
      entry itself would leave a stale sysimage silently running mismatched compiled code.
    rationale: >
      Observable consequence: a Manifest-hash trigger scoped to only the Wflow entry can miss a
      transitive-dependency or Julia-version change, reintroducing the exact silent-old-code
      correctness risk the gate exists to prevent. This is a real (if lower-probability) staleness hole
      in a lever the design already flags as a correctness risk.
    suggested_fix: >
      Key the rebuild trigger on a hash of the whole project Manifest.toml (plus the pinned Julia
      version), not the single Wflow entry — or on the sysimage's own recorded build-Manifest hash.
      This is deferred anyway per the gate, so it is a spec-refinement note for the eventual sysimage
      commit, not a blocker.

```

---

## Internal lens review — architecture (verbatim)

# Internal review — architecture & internal-consistency lens (P3-3 performance)

```yaml
verdict: revise
doc_version: design-v1.md
findings:
  - id: arch-1
    severity: blocking
    section: "§6.2 / §6.5 / §9 / OQ-2 — lever recommendation vs the G1 gate ruling"
    finding: >
      The doc recommends sysimage as the PREFERRED default and frames batching
      as the approval-free fallback (§6.2 "if approved, sysimage is the preferred
      lever"; §6.5 rank table "Sysimage (B) — the RECOMMENDED default"; §9 commit 2
      "sysimage (if approved) OR batching"; OQ-2 "sysimage (preferred) or batching
      (fallback)"). The recorded, task-fixed G1 ruling is the reverse:
      status.md states "PackageCompiler sysimage DEFERRED to commit-2 evidence —
      batching proceeds first as the no-dependency lever; sysimage needs a fresh
      approval ask only if batching's measured throughput/robustness disappoints",
      and asserts "The revision folds this ruling into §6.5/§9/OQ-2." That fold
      did not happen — design-v1.md is stale against its own approved anchor.
    rationale: >
      The design's headline recommendation directly contradicts the fixed gate
      ruling this run must implement. A downstream reader (model-builder,
      python-engineer) taking §6.5/§9 at face value would build the wrong lever
      first, or wait on a sysimage approval the gate already deferred. This is
      the "design fails to DELIVER the anchors" failure class the task puts in
      scope. It cannot be an approve while the recommended-lever framing inverts
      the G1 decision.
    suggested_fix: >
      Rewrite §6.5/§9/OQ-2 so batching is the built-first commit-2 lever (no-dep),
      and sysimage is a conditional follow-up gated on batching's measured
      throughput/robustness disappointing — the exact status.md ruling. Retain
      the safety-vs-throughput analysis as rationale, but stop labelling sysimage
      the "RECOMMENDED default." Commit 2 in §9 becomes "batching," not "sysimage
      (if approved) OR batching."

  - id: arch-2
    severity: major
    section: "§5.5 — the production scaling model (boxed formula vs illustrative table)"
    finding: >
      The boxed wall_batched formula `ceil(K/(p·B)) × (F + B·S)` does NOT generate
      the illustrative table beneath it. Reproducing the table's numbers
      (934, 1182, 715, 963 for B=2/3/4/6 at K=12, p=3, F=135, S_cold=208,
      S_warm=124) requires a warm-aware per-batch cost:
      `wall = ceil(ceil(K/B)/p) × (F + S_cold + (B−1)·S_warm)` — verified exactly
      (B=2→934, B=3→1182, B=4→715, B=6→963). The boxed formula instead yields
      1102 (cold-S) or 766 (warm-S) for B=2, matching neither. Two defects: (a)
      the boxed cost `F + B·S` applies one sim term to all B runs, but the design's
      own §probe-1d prose one line below says runs 2..N are warm — the formula
      omits the very warm discount the table and §5.5 findings depend on; (b) the
      boxed wave count `ceil(K/(p·B))` is not the correct `ceil(ceil(K/B)/p)`
      (they coincide on these four B values but diverge for non-divisible K/B).
    rationale: >
      The scaling model is a first-class G1/G4 deliverable (§5.5, commit 1 records
      it durably under dev/milestones/p33/ "à la baseline_diffs"). Shipping a headline formula
      that does not reproduce its own worked table is an internal contradiction
      that will mislead any future reader recomputing a production B, and
      undermines the "honest scaling statement" the model exists to provide.
      The table is self-consistent; the stated formula is the wrong one.
    suggested_fix: >
      Replace the boxed batched line with
      `wall_batched ≈ ceil(⌈K/B⌉ / p) × (F + S_cold + (B−1)·S_warm)`
      and note S_cold vs S_warm explicitly in the formula (not only in the prose
      below it). Re-derive the table rows from that formula so model and table
      cite one source of truth.

  - id: arch-3
    severity: major
    section: "§8 determinism handling / OQ-3 — value-identity presupposition for batching"
    finding: >
      §8 asserts "a batched vs per-process Wflow run must produce bit-identical
      CSVs … Julia's numeric result is not process-boundary-dependent, so batching
      is expected bit-identical," and OQ-3 asks only to CONFIRM the existing
      R5/P3-1 per-process re-run determinism evidence. But the repo's recorded
      evidence (dev/milestones/p31/baseline_diffs.md: "0 failed … live confirmation of the
      R5-verified wf3 determinism, seed 123", via the whole-tree semantic diff)
      establishes PER-PROCESS run-to-run exact reproducibility only. Batching's
      new risk is warm-SESSION vs cold-PROCESS byte identity: runs 2..N of a batch
      reuse in-session allocations, JIT-compiled method instances, and GC/global
      state that a fresh process lacks (§probe-1d itself measures an 84 s/run
      warm-cache effect — proof the warm path is materially different execution).
      No existing evidence covers warm-vs-cold byte identity; commit-2 gate-1 is
      its first test. OQ-3 as written points at the OLD evidence, which does not
      discharge the NEW risk.
    rationale: >
      If a warm session perturbs any floating-point reduction order (threaded
      accumulation, allocation-dependent SIMD paths), batched CSVs could differ
      sub-LSB from per-process CSVs — a gate-1 semantic-diff FAIL that is a real
      batching artifact, not baseline nondeterminism. Framing OQ-3 as "confirm the
      per-process evidence" risks mis-attributing such a diff to the ADR-0001
      immaterial branch and waving it through. Since batching is now the
      built-first lever (arch-1), this is the load-bearing value-identity
      assumption of the whole milestone.
    suggested_fix: >
      Reword §8 and OQ-3 to name warm-session-vs-cold-process byte identity as an
      UNTESTED assumption that commit-2 gate-1 (the whole-tree semantic diff at
      tol=0 / exact) is the first evidence for — not something the P3-1 per-process
      re-run already established. Keep the ADR-0001 triage path, but require that a
      gate-1 diff on a run_wflow output be attributed to batching first, cleared
      only after ruling out the warm/cold path.

  - id: arch-4
    severity: major
    section: "§6.1 — batch-membership input-function mechanism deferred"
    finding: >
      The single non-trivial Snakemake mechanism of the built-first lever — the
      batch_id → member-(rlz,cst) partition input function and the multi-named
      per-cst output declaration — is left implementation-deferred to the
      task-brief, with export_wflow_results cited as in-repo precedent. That
      precedent is inexact: rule 3.11 (verified, Snakefile lines 377-395) is a
      2-output AGGREGATOR consuming an `expand()` input and producing exactly two
      fixed outputs (Qstats.csv, basin.csv). The batched rule 3.10 must instead
      declare a batch-VARYING SET of per-cst outputs (output_rlz_<n>_cst_<m>.csv +
      temp() outstates) keyed on a batch_id wildcard whose input is a partition
      FUNCTION — a materially different Snakemake construct (parametrized dynamic
      output set, not a static expand-input aggregator).
    rationale: >
      Under the G1 ruling batching is what actually ships in commit 2, so the
      least-specified mechanism is the one that must work. Deferring it to the
      brief without a worked partition/output shape leaves open whether the rule
      is even expressible without checkpoints (Snakemake cannot expand outputs
      from a value not known at DAG-construction; here K, B, and thus the
      partition ARE parse-time constants, so an input function + Python-generated
      output list is viable — but the design never demonstrates the output-list
      construction, only asserts it "mirrors export_wflow_results," which it does
      not).
    suggested_fix: >
      Add a minimal worked sketch to §6.1: the parse-time partition (a Python
      dict batch_id → list[(rlz,cst)] built from K and B before the rule), the
      output as an explicit Python list comprehension over that batch's members
      (not an expand), and the input function form. Confirm no checkpoint is
      needed (partition is parse-time deterministic). Keep the exact code deferred,
      but pin the CONSTRUCT so the brief cannot pick an inexpressible shape.

  - id: arch-5
    severity: minor
    section: "§5.5 / §6.1 — sweep size K definition vs the fixture"
    finding: >
      §5.5 defines K = RLZ_NUM × ST_NUM(+1). On the fixture, run_wflow fans out
      over cst {1..ST_NUM} only (ST_START = 0 if run_historical else 1; Snakefile
      lines 52-53), and the recorded benchmark has exactly 12 run_wflow rows
      (rlz {1,2} × cst {1..6}) — i.e. K = RLZ_NUM × ST_NUM = 12, NOT +1. The "+1"
      (cst_0) only enters run_wflow when run_historical is true, which the seed
      config is not. The design's own "today ≈ ceil(12/3)×343" arithmetic uses 12,
      consistent with no +1, but the K definition prints "(+1)".
    rationale: >
      A downstream reader sizing a production batch from K = RLZ_NUM × ST_NUM(+1)
      would over-count by RLZ_NUM runs whenever run_historical is false (the
      default), skewing the disk-ceiling and wave-quant batch-size choice.
    suggested_fix: >
      State K = RLZ_NUM × (ST_NUM + [run_historical]) explicitly, or footnote that
      the +1 baseline run applies to run_wflow only when run_historical is set;
      the fixture (and the arithmetic) uses K = 12.

  - id: arch-6
    severity: minor
    section: "§8 gate 2 / §10 — check_baseline mixed-provenance triage is sound; scope note"
    finding: >
      §8 gate-2 correctly invokes check_baseline.py --workflow climate_experiment
      and correctly flags the ADR-0001 mixed-provenance caveat. Verified against
      the tree: check_baseline TARGETS for climate_experiment are exactly
      Qstats.csv, basin.csv, and the wf3 config snapshot (lines 114-116); the
      logs/ and benchmarks/ trees are NOT manifested there and semantic_tree_diff
      excludes them via EXCLUDED_DIR_NAMES (lines 114, 567-573). So the §7.3
      benchmark-row collapse under batching is value-identity-safe — neither gate
      sees it. This is correct but not stated: the design should assert that the
      benchmark/log granularity shift is provably outside BOTH value-identity
      gates, rather than leaving it as an unquantified "documented visibility
      shift."
    rationale: >
      Making the gate-invisibility explicit closes the one place a reviewer might
      suspect the row-collapse threatens value-identity; it does not, and the
      design should say why (excluded dirs + non-manifested), so the point is not
      re-litigated at the value-identity gate.
    suggested_fix: >
      In §7.3 / §11 add one clause: benchmarks/ and logs/ are excluded from
      semantic_tree_diff (EXCLUDED_DIR_NAMES) and absent from check_baseline
      TARGETS, so the per-batch benchmark collapse is invisible to both
      value-identity gates by construction.
```

## Cross-checks that PASSED (recorded so they are not re-litigated)

- **Rule graph walked.** Rule 3.10 `run_wflow` (Snakefile lines 361-374) takes
  `forcing_path` (`inmaps_rlz_*_cst_*.nc`, a `temp()` output of 3.09) + `toml_path`
  (persisted `wflow_sbm_rlz_*_cst_*.toml` from 3.09), emits persisted
  `output_rlz_*_cst_*.csv` + `temp()` `outstates_*`. Downstream 3.11
  `export_wflow_results` consumes the CSVs via `expand()` over
  cst {ST_START..ST_NUM}. A batched rule that (a) declares the same per-cst CSV
  paths and (b) writes each cst's CSV via the same `Wflow.run(toml)` keeps 3.11's
  input contract intact.
- **HM-7 reduction is order-independent under batching.** `export_wflow_results.py`
  derives gauge columns from `csv_fns[0]` (line 60) then, in aggregation mode,
  selects each cst's files by `endswith("cst_"+str(i+1)+".csv")` (line 127) — a
  filename match, not positional. Batch GROUPING of runs therefore cannot corrupt
  the HM-4→HM-5→HM-7 gauge-column identity, provided each per-cst CSV keeps its
  path and `Q_`-prefixed column (validate_hm_gauge_column_identity stays green).
  The relational validator's "first csv" dependency (contract §HM-5) is on column
  IDENTITY, which is per-cst path-scoped, not batch-order-scoped.
- **Per-cst byte identity rides gate 1, not gate 2.** The whole-tree
  semantic_tree_diff at exact tolerance (compare_nc / compare_csv, tol→exact) is
  the byte-identity gate on every persisted per-cst output; check_baseline (gate 2)
  only fingerprints the three manifested wf3 targets. §8 correctly orders both.
- **§5.1 metric correction is sound.** Benchmark TOTAL = 5667.60 s is the summed
  per-rule wall (verified: it is the `s`-column sum with peak-memory), the twelve
  3.10 rows are 387-410 s each, and "today ≈ ceil(12/3)×343 = 1372" is arithmetically
  consistent with the wall-at-cores reframing. Using the 5,668 sum as a baseline
  would indeed overstate every lever by the parallelism factor; the design avoids it.
- **Lever C / threads reasoning consistent.** Benchmark cpu_time ≈ wall at
  mean_load ~95 % on all twelve 3.10 rows confirms the single-core-per-run claim
  (§6.3); keeping `--threads 4` for the production regime is defensible.
- **Lever D drop is evidence-backed** (probe 1b: warm-run sim log-level-independent).

## Verdict rationale

Two blocking/major anchor-and-consistency defects (arch-1 lever/gate inversion;
arch-2 formula-vs-table) each independently bar `approve`. arch-3 (determinism
mis-scoped for the built-first lever) and arch-4 (the shipping lever's one hard
mechanism under-specified) are majors that must land before implementation is
brief-able. The design's measurement backbone, gate ordering, and downstream
contract preservation are otherwise sound and verified against the tree — hence
`revise`, not `reject`.


---

## Internal lens review — repo-fit (verbatim)

# Internal review — Repo-fit & conventions lens (P3-3 performance)

Lens: repo-fit. Verifies that cited file:line references resolve, the proposed
Julia batch-driver / rule restructure fits the repo's script-wrapper and
per-rule `log:`/`benchmark:` conventions, the commit-1 artifacts land in the
right `dev/` locations, the measurement tooling is implementable on this Windows
machine, the value-identity proof steps match the actual tool CLIs, and the
design delivers the fixed anchors (incl. the recorded G1 gate ruling).

Method: read-only spot-checks against rule 3.10 shell body
(`Snakefile_climate_experiment:360-374`), `blueearth_cst/shared/run_logged.py`,
`blueearth_cst/experiment/downscale_climate_forcing.py`,
`profiles/default/config.yaml`, `blueearth_cst/shared/merge_benchmarks.py`,
`tests/test_interchange_contracts.py`, `dev/scripts/check_baseline.py`,
`dev/scripts/semantic_tree_diff.py`, `Wflow.jl` (installed pkg),
`dev/reference/naming.md`, the three Snakefiles, and
`dev/working/design-runs/p33-performance/status.md` (the G1 gate ruling).

**Verified as SUPPORTING the design (no finding needed):** rule 3.10 shell body
matches §5.2/§6.1's production invocation verbatim (`python -u run_logged.py
{log} -- julia +1.11.7 --project=. --threads 4 -e "using Wflow; Wflow.run()"
"{toml}"`); `run_logged.py` tee wrapper is exactly as described; `Wflow.jl:250`
(`run(tomlpath::AbstractString)`) and `:344` (`run()` reading `only(ARGS)`)
resolve exactly, so the batch-driver `for t in ARGS; Wflow.run(t)` mechanism is
sound; `downscale_climate_forcing.py:43` `WflowSbmModel(root=…, mode="r+", …)`
and the per-cst `setup_precip_forcing`/`setup_temp_pet_forcing` (lines 87/92)
resolve, backing §5.3; the proposed `run_wflow_batch.jl` snake_case name fits
`naming.md`; `julia`/`juliaup` are on PATH (WindowsApps shim forwards `+1.11.7`),
so the probe/measurement tooling is implementable here; the P3-2b validators
read outputs by exact on-disk path (`model_runs/output_rlz_{n}_cst_{m}.csv`,
`wflow_sbm_rlz_{n}_cst_{m}.toml`, `model_results/{Qstats,basin}.csv`), so a
batched rule that writes those same paths keeps §8-gate-3 green — the design's
central byte-compatibility anchor holds; `check_baseline.py check --workflow
climate_experiment` is a valid CLI (choices = WORKFLOWS); intake anchors (wf3
throughput only, value-identical, probe-first) are consistent.

```yaml
verdict: revise
doc_version: design-v1.md
findings:
  - id: repo-1
    severity: major
    section: "§6.5 Lever ranking / §6.2 / §9 Commit plan / §12 OQ-2"
    finding: >
      The design recommends SYSIMAGE as the default lever (§6.5 "the RECOMMENDED
      default"; §6.2 "if approved, sysimage is the preferred lever"; §9 commit 2
      = "sysimage (if approved) OR batching"; OQ-2 framed as an open user
      decision), which contradicts the fixed G1 gate ruling recorded in
      status.md: "PackageCompiler sysimage DEFERRED to commit-2 evidence —
      batching proceeds first as the no-dependency lever; sysimage needs a fresh
      approval ask only if batching's measured throughput/robustness
      disappoints." Batching is the committed commit-2 lever; sysimage is
      deferred/conditional. The design-as-written inverts that fixed anchor.
    rationale: >
      A reader (or the implementation task-brief author) taking §6.5/§9/OQ-2 at
      face value would build the sysimage path first or treat the lever choice
      as still-open, directly against the gate ruling the run is bound by. This
      is a "fails to deliver the fixed anchor" finding, explicitly in scope for
      this lens. status.md notes "the revision folds this ruling into
      §6.5/§9/OQ-2" — but design-v1.md as written has not; the document under
      review is stale relative to its own gate.
    suggested_fix: >
      Reframe §6.5 so commit-2 = batching (the committed lever) and sysimage is
      the deferred/conditional option reached only if batching disappoints;
      align §6.2, §9 commit 2, and OQ-2 to the same. Keep the sysimage analysis
      as recorded rationale for the deferral, not as the recommendation.

  - id: repo-2
    severity: minor
    section: "§8 gate 2 / §10 value-identity acceptance criterion"
    finding: >
      §10's acceptance criterion lists "byte-identical Qstats.csv/basin.csv/
      per-cst CSVs" under the check_baseline gate, and §8 gate 2 invokes
      `check_baseline.py check --workflow climate_experiment` for "manifested
      targets unchanged." But the check_baseline manifest fingerprints only
      Qstats.csv, basin.csv, and the config snapshot (TARGETS in
      check_baseline.py:114-116) — the per-cst `output_rlz_*.csv` are NOT
      manifested. Per-cst CSV byte-identity is covered by gate 1
      (semantic_tree_diff full-tree) and gate 3 (P3-2b `validate_hm5`), not by
      check_baseline.
    rationale: >
      As written, §10 conflates two distinct gates and could lead a validator to
      expect check_baseline to assert per-cst CSV identity (it does not), or to
      under-run the semantic diff assuming check_baseline already covered them.
    suggested_fix: >
      Attribute per-cst CSV identity to gate 1 (semantic_tree_diff) + gate 3
      (P3-2b), and restrict the check_baseline gate to its actual manifested
      targets (Qstats/basin/config snapshot).

  - id: repo-3
    severity: minor
    section: "§6.1 rule restructuring (batched-rule precedent)"
    finding: >
      §6.1 cites `export_wflow_results` as the in-repo precedent for the batched
      rule's shape and says the output list uses "the same pattern
      export_wflow_results uses for its expand input." export_wflow_results
      (Snakefile_climate_experiment:377-395) is an `expand`-INPUT aggregator with
      two FIXED, hardcoded outputs (Qstats/basin) and NO wildcard. The batched
      rule's load-bearing mechanism — a `batch_id` wildcard + a Snakemake INPUT
      FUNCTION mapping batch_id → member NCs/TOMLs + a computed per-batch OUTPUT
      list — is not demonstrated by that rule. A grep of all three Snakefiles
      found no `lambda wildcards` / input-function / `unpack()` precedent
      anywhere in the repo.
    rationale: >
      "Mirrors export_wflow_results" oversells the precedent: the aggregator
      covers only the expand-input aspect, not the input-function/wildcard
      mechanism the lever actually needs (and which the design itself marks
      implementation-deferred). An implementer expecting a copyable in-repo
      pattern will find none.
    suggested_fix: >
      State plainly that the input-function/batch-partition mechanism has no
      in-repo precedent (export_wflow_results demonstrates only the
      expand-input, fixed-output aggregator shape), so it is genuinely new
      Snakemake surface for the batching task-brief.

  - id: repo-4
    severity: minor
    section: "§5.1 metric framing (benchmark row granularity)"
    finding: >
      §5.1 describes the wf3_benchmarks TOTAL as a "sum of per-rule wall times
      (one row per rule; fan-out rules aggregate all their jobs)." That is the
      merge_benchmarks LEGEND text, but merge_benchmarks derives the `rule`
      column from each part file's relative path
      (merge_benchmarks.py:66) — so `3.10_run_wflow/rlz_N_cst_M.tsv` yields ONE
      ROW PER JOB (12 rows for run_wflow), not one aggregated row. The TOTAL is
      a sum over per-JOB rows.
    rationale: >
      Harmless to §5.1's core point (TOTAL is still a sum, not wall clock), but
      the "one row per rule" phrasing is internally inconsistent with §6.1/§7.3,
      which correctly state that batching collapses "12 per-cst rows to
      ceil(K/B)." §6.1/§7.3 are the accurate description; §5.1's parenthetical is
      not.
    suggested_fix: >
      Align §5.1's parenthetical with §6.1/§7.3: the current benchmark already
      emits one row per (rlz,cst) job for run_wflow (12 rows), and batching
      collapses those to ceil(K/B).

  - id: repo-5
    severity: minor
    section: "§5 preamble / §8 gate 1 (semantic-diff precedent artifact)"
    finding: >
      §8 gate 1 cites `dev/milestones/p31/_semantic_diff.out` as "the precedent output
      form" for semantic_tree_diff. That path is untracked and empty (created in
      the current working tree, not a committed landed artifact); the actual
      landed dev/milestones/p31 semantic-diff evidence is `_wf3_regen.log` +
      `baseline_diffs.md`. Additionally, §8 gate 1 invokes semantic_tree_diff.py
      without naming its required `--ref <dir> --cur <dir>` args (its actual
      CLI).
    rationale: >
      Citing a non-durable artifact as a precedent is a spurious reference; and a
      validator reading gate 1 verbatim gets no `--ref/--cur` invocation form.
      Neither blocks the gate (the tool is self-documenting and the correct
      precedent exists), hence minor.
    suggested_fix: >
      Point gate 1 at the actual landed precedent (dev/milestones/p31/baseline_diffs.md /
      _wf3_regen.log) and show the real CLI form
      (`semantic_tree_diff.py --ref <ref-tree> --cur <cur-tree>`).

  - id: repo-6
    severity: minor
    section: "§5.1 (profile / cores wording)"
    finding: >
      §5.1 states "wf3 runs under the repo profile at `-c N`." The auto-loaded
      profile (profiles/default/config.yaml) sets only `quiet: reason`; it does
      NOT set a core count. `-c N` comes from the command line
      (run_snake_test.cmd / run_workflows.py default), not the profile.
    rationale: >
      Minor wording drift; the parallelism reasoning is unaffected (the -c 3
      default is real), but the profile is not its source.
    suggested_fix: >
      Attribute `-c N` to the invocation (run_snake_test.cmd / wrapper default),
      not to the workflow profile.
```

## Notes on the fixed anchors (in-scope delivery check)

- **wf3 throughput only, value-identical** — honored. Levers touch only the
  invocation/DAG, not outputs, entry points, or the wrapper contract; §8 proof
  plan is aimed at the right surfaces (semantic diff + manifest + P3-2b + CLI
  dry-runs).
- **Structural latitude (DAG may change, outputs may not)** — honored and, on
  the load-bearing axis, VERIFIED: the P3-2b validators read outputs by exact
  on-disk path, so a batched rule writing the identical per-cst CSV/TOML paths
  keeps them green regardless of DAG shape. This is the design's strongest
  repo-fit claim and it checks out.
- **Probe-set criteria** — honored (intake probe-first anchor consistent).
- **GATE RULING: sysimage DEFERRED, batching first** — NOT honored in the
  document as written (finding repo-1, major). This is the one anchor the design
  fails to deliver.

`approve` is invalid with a major finding; verdict is `revise`.


---

## External review round 1 (verbatim; doc_version design-v2.md)

## Verdict
verdict: revise
doc_version: design-v2.md

## Findings
### ext1-001  [major]
- section: 5.5 The production scaling model
- finding: The batched-wall formula treats every batch as containing `B` runs: `ceil(ceil(K/B)/p) × (F + S_cold + (B−1)·S_warm)`. For the stated general production case, `K` need not divide by `B`; the final batch has fewer runs and its duration differs. Scheduling makes the makespan depend on which batch lands in each worker wave, not simply the number of waves times a full-batch duration.
- rationale: The model is used to select and justify the batch-size knob, yet it can materially mis-rank batch sizes for production sweep sizes with a remainder, undermining the design’s honest-scaling and tuning claims.
- suggested_fix: Define batches with sizes `b_i` (including the remainder), calculate each batch duration as `F + S_cold + (b_i−1)·S_warm`, and state a conservative or simulated scheduling method for estimating makespan across `p` workers.

### ext1-002  [major]
- section: 5.5 The production scaling model
- finding: The scaling model treats effective parallelism `p`, Julia’s `--threads` setting, and the “single-core-per-run” assumption as effectively independent, despite the measured invocation using `--threads 4` and production runs potentially being threadable. The design does not specify the CPU-resource budget or how `-c N`, batch concurrency, and per-session Julia threads will be jointly fixed for baseline, comparison, and deployment.
- rationale: A batched implementation can appear faster or slower solely through oversubscription or changed thread allocation rather than fixed-cost amortization. It also makes the production scaling statement non-actionable: operators cannot derive a safe `B`, `-c`, and `--threads` combination from the model.
- suggested_fix: Add a resource contract that fixes total CPU threads for each measurement and deployment mode, derives concurrent batches from that budget, and benchmarks the selected `-c`/threads combination under the same rule for both pre- and post-change runs.

### ext1-003  [major]
- section: 9. Commit plan
- finding: The conditional sysimage follow-up is triggered when batching’s “measured throughput/robustness disappoints,” but neither term has an acceptance threshold. The design also acknowledges slower batch-size choices, degraded failure isolation, higher disk use, and coarser benchmark rows without defining which measured outcome requires stopping after commit 2 versus requesting PackageCompiler approval.
- rationale: The stated fallback decision cannot be executed consistently. A batching result can be declared successful or disappointing after the fact, leaving the dependency decision and a known reliability regression to subjective interpretation rather than the documented gate.
- suggested_fix: Before commit 2, define explicit go/no-go criteria: minimum wall-clock improvement at the fixed resource budget, disk-ceiling compliance, required failure-injection behavior, and any acceptable re-run blast radius; specify that failure of any criterion produces the fresh sysimage approval ask.

---

## External review round 2 (verbatim; doc_version design-v3.md)

## Verdict
verdict: reject
doc_version: design-v3.md

## Findings
### ext2-001  [blocking]
- section: 6.1 Lever A — batching N Wflow runs per Julia session
- finding: The proposed batched-rule construct is not expressible as specified: it uses callable `output:` entries (`lambda w: [...]`) to generate a batch-dependent list of declared CSV and `temp()` outputs. Snakemake supports input functions, but rule outputs must be statically declared patterns/paths; a module-level lookup does not make a wildcard-dependent variable output set routable either. Thus the unchanged per-cst targets cannot be mapped to one batch job through this construct.
- rationale: Commit 2’s core rule cannot construct its DAG or satisfy `rule all`/rule 3.11 as claimed, so the batching lever is not implementable under the pinned mechanism. This also means the accepted arch-4/repo-3 resolution has not actually resolved the prior under-specification.
- suggested_fix: Replace the construct with a Snakemake-valid DAG design and demonstrate it with a minimal dry-run before sealing: either a statically enumerable set of batch rules/outputs at parse time, or a different valid producer/consumer arrangement that preserves the existing per-cst output paths and downstream targets. Update §6.1, the commit plan, and validation gate to name the valid mechanism rather than deferring its output declaration.

---

## Final disposition ledger (verbatim)

# Disposition ledger — p33-performance

One row per finding ID. Internal panel (18: risk-1..6, arch-1..6, repo-1..6;
Round = internal-panel; design-v1.md → design-v2.md; blocking arch-1
accepted-with-resolution — a rejected blocking would trigger arbitration).
External round 1 (3: ext1-001..003; Round = external-r1; design-v2.md →
design-v3.md).

| ID | Round | Severity | Disposition | Resolution or rationale | Doc version |
| --- | --- | --- | --- | --- | --- |
| arch-1 | internal-panel | blocking | accepted | Group A. Rewrote §5.5/§6.1/§6.2/§6.5/§7.2/§9/§11/OQ-2 batching-first per the recorded G1 gate ruling: batching is the built-first no-dependency commit-2 lever; the sysimage is a deferred, doubly-gated follow-up (batching-disappoints AND fresh-approval). All "sysimage RECOMMENDED default / preferred lever / commit-2 approval fork" framing removed; the safety analysis retained only as the deferral rationale. §6.5 ranking table replaced (fixed build order, not APPROVED/REJECTED fork). | design-v2.md |
| risk-2 | internal-panel | major | accepted | Group A. §9 commit 2 is now batching (not "sysimage if approved OR batching"); sysimage relabeled a conditional follow-up outside the sealed three-commit plan; OQ-2 reworded to state batching-first, sysimage deferred. Coupled to risk-1: the batching-first path owns the C5 degradation. | design-v2.md |
| repo-1 | internal-panel | major | accepted | Group A. Same batching-first reframe of §6.2/§6.5/§9/OQ-2; sysimage analysis kept as recorded rationale for the deferral, not the recommendation. | design-v2.md |
| risk-1 | internal-panel | major | accepted | Group B. C5 stated DEGRADED under batching (blast radius B: Snakemake deletes the failed job's present outputs → B−1 completed batch-mates deleted + whole batch re-runs; rule 3.11 blocked for the sweep). Chose ACCEPT-AND-DOCUMENT (measure-first: an unverified `--keep-incomplete` mechanism would violate verify-don't-assume). The `--keep-incomplete` ↔ `--keep-going` interaction is named as a commit-2 probe candidate with accept-the-degradation as the explicit fallback. §6.1 + §11 + §10 aligned. | design-v2.md |
| arch-3 | internal-panel | major | accepted | Group C. §8 determinism handling + OQ-3 reworded: warm-session-vs-cold-process byte identity is UNTESTED (P3-1 evidence is per-process only; the 84 s warm discount proves execution differs); first evidence = commit-2 gate-1. Added the discriminating rule: any batching-correlated diff (present batched, absent per-process on identical inputs) BLOCKS; ADR-0001 branch admissible only after a per-process re-run reproduces the diff. | design-v2.md |
| risk-3 | internal-panel | major | accepted | Group C. Same §8/§10 cluster edit: the ADR-0001 immaterial branch cannot launder a batching drift; gate 1 is run per-process-vs-batched on identical inputs as the discriminating test; a batching-correlated Qstats/basin/per-cst diff BLOCKS the commit. | design-v2.md |
| arch-2 | internal-panel | major | accepted | Group D. Replaced the boxed `wall_batched ≈ ceil(K/(p·B)) × (F + B·S)` with the warm-aware `wall_batched ≈ ceil(⌈K/B⌉ / p) × (F + S_cold + (B−1)·S_warm)`; verified it reproduces the table exactly (B=2→934, B=3→1182, B=4→715, B=6→963; today→1372; sysimage→840). Table re-derived from this single source of truth; S_cold/S_warm named in the formula. Table values unchanged (already self-consistent). | design-v2.md |
| arch-4 | internal-panel | major | accepted | Group E. Added the worked construct sketch to §6.1: parse-time partition dict `batch_id → [(rlz,cst),...]` from K and B; input function form; output as an explicit Python list comprehension over the batch's members (NOT expand); confirmed no checkpoint needed (partition is parse-time deterministic). Exact signatures deferred; the construct pinned. | design-v2.md |
| repo-3 | internal-panel | minor | accepted | Group E. §6.1 now states plainly there is NO in-repo precedent for the input-function/batch-partition construct (export_wflow_results demonstrates only the fixed-output expand aggregator; a grep of all three Snakefiles finds no lambda/input-function/unpack precedent). | design-v2.md |
| arch-5 | internal-panel | minor | accepted | Group F. §5.5 states K = RLZ_NUM × (ST_NUM + [run_historical]); the cst_0 baseline run enters K only when run_historical is set (Snakefile lines 52-53); the seed fixture has it false → K = 12. | design-v2.md |
| arch-6 | internal-panel | minor | accepted | Group F. §8 (and §6.1/§11) state the benchmark/log row-collapse is invisible to both value-identity gates by construction: benchmarks/ + logs/ excluded from semantic_tree_diff (EXCLUDED_DIR_NAMES) and absent from check_baseline TARGETS. | design-v2.md |
| risk-4 | internal-panel | minor | accepted (partial) | Group F. Softened "firm/decisive/ground truth" to "single-sample point estimate, confirmed by commit 1" (OQ-1, §5.5). NO repeats run — sweep repeats are not cheap read-only; folded the confirmation into commit 1's measurement spec (§9 commit 1) rather than running them now. | design-v2.md |
| risk-5 | internal-panel | minor | accepted | Group F. Disk ceiling re-derived as p × B × (forcing_size + state_size) — both temp() classes (3.09 forcing NCs + 3.10 outstates NCs co-resident per batch), verified against Snakefile line 368; §6.1 + §11 aligned; outstates reclamation timing flagged as a commit-2 verify-not-assume. | design-v2.md |
| risk-6 | internal-panel | minor | accepted | Group F. §6.2 + §11 key the sysimage staleness trigger on a hash of the full Manifest.toml plus the pinned Julia version (not the single Wflow entry — a transitive-dep bump would slip through). Spec note in the deferred-lever section; not active while the lever is deferred. | design-v2.md |
| repo-2 | internal-panel | minor | accepted | Group F. §8 gate 2 + §10 acceptance criterion: per-cst output_rlz_*.csv identity attributed to gate 1 (semantic_tree_diff) + gate 3 (P3-2b validate_hm5), NOT check_baseline; check_baseline restricted to its manifested TARGETS (Qstats/basin/config snapshot, check_baseline.py:114-116). | design-v2.md |
| repo-4 | internal-panel | minor | accepted | Group F. §5.1 parenthetical fixed: run_wflow already emits one row per (rlz,cst) JOB (12 rows, merge_benchmarks.py:66), not one aggregated per-rule row; TOTAL sums over per-job rows. Aligned with §6.1/§7.3. | design-v2.md |
| repo-5 | internal-panel | minor | accepted | Group F. §8 gate 1 now cites the landed precedent dev/milestones/p31/baseline_diffs.md + _wf3_regen.log (the untracked _semantic_diff.out demoted to a working artifact) and shows the real CLI `semantic_tree_diff.py --ref <ref-tree> --cur <cur-tree>`. | design-v2.md |
| repo-6 | internal-panel | minor | accepted | Group F. §5.1 attributes -c N to the invocation (run_snake_test.cmd / run_workflows.py default), noting the auto-loaded profile sets only `quiet: reason`, not a core count. | design-v2.md |
| ext1-001 | external-r1 | major | accepted | §5.5 model rebuilt remainder-aware (supersedes the arch-2 fix): batch sizes b_i = ⌊K/B⌋ full + remainder r = K mod B; per-batch duration D(b_i) = F + S_cold + (b_i−1)·S_warm; makespan = greedy LPT simulation (estimator of record, ~10 lines, landed by commit 1 as dev/scripts/estimate_batch_makespan.py) bracketed by Graham list-scheduling bounds [max(D_max, ΣD/p), ΣD/p + (1−1/p)·D_max]. Verified numerically: for B | K the simulation reduces exactly to the v2 wave formula, fixture table unchanged (934/1182/715/963; today 1372; sysimage 840); for B ∤ K the wave formula overestimates AND mis-ranks (K=13, p=3: wave B=4 1430 vs simulated 1058, so wave prefers B=3 1182 while the simulation prefers B=4). Precision claim added: honest ranking within the Graham bracket, not exact prediction; commit 2 is the evidence of record. §6.1/§11 batch-size selection rewired to the estimator; "wave-quantization" renamed scheduling-quantization. | design-v3.md |
| ext1-002 | external-r1 | major | accepted | New §5.6 CPU resource contract: every measured mode states its (-c N, --threads t, B) triple; nominal cap N × t ≤ C_logical (dev box i5-1335U, 10 physical / 12 logical; today's (3,4) exactly at the nominal cap, effectively ~3 busy cores per the ~95 % single-core benchmark evidence — --threads 4 buys nothing on the fixture); commit-2 before/after MUST use the identical (N, t) as the commit-1 baseline so B is the only moved knob (win/loss attributable to amortization, not oversubscription/reallocation; mismatched-budget comparisons invalid as commit-2 evidence); operator derivation order t → N → B (basin threadability → machine cap → §5.5 simulation under the §6.1 disk ceiling). Verified against the tree: rule 3.10 declares no `threads:` directive (each job counts 1 against -c N); the batched rule keeps that accounting. §6.1/§6.3/§9/§10 bound to the contract. | design-v3.md |
| ext1-003 | external-r1 | major | accepted | §9 commit-2 gate defines "batching disappoints" executably as failing any of: GN-1 measured sweep-wall reduction ≥ 15 % at the fixed §5.6 budget (model-derived: half the most-conservative feasible-B win, B=2 → −32 %, confounds only erode); GN-2 all §8 value-identity gates green incl. the discriminating per-process-vs-batched diff (fail = no-go AND blocks commit 2, C2 absolute); GN-3 measured disk peak within the p × B × (forcing+state) cap + outstates temp reclaimed at batch end; GN-4 failure injection realizes exactly the documented C5 blast radius (B−1 siblings deleted, batch re-runs, 3.11 blocked only until re-run, per-cst driver log lines present incl. the failed cst, clean re-convergence). Decision rule: all-pass → sysimage dormant; any-fail → fresh PackageCompiler approval ask naming the failed criterion (the G1 ruling's trigger). Anchor distinction stated: the intake's no-a-priori-floor governs the MILESTONE gate (user sign-off, floor-free); GN-1 is an internal LEVER-routing threshold only, so a GN-1 failure can coexist with user acceptance of the measured numbers. §6/§6.2/§6.5/§7.2/§10/§11/OQ-2 rewired from "disappoints" to GN-1..GN-4; §10 adds the go/no-go adjudication row (cst-architect evaluates; the dependency ask goes to the user). | design-v3.md |
| ext2-001 | external-r2 | blocking | accepted (arbitration ruling 2026-07-24, fix mandated) | v3 §6.1 pinned callable `output:` entries (`csvs = lambda w: [...]`) — not expressible: probe `probes/snakemake_output_expressibility/Snakefile_lambda` fails on the pinned Snakemake 9.6.2 with `RuleException: Only input files can be specified as functions` (rule outputs must be static; only inputs may be functions). Replaced with the user-mandated, probe-verified construct: parse-time loop-generated anonymous rules (`for _b, _members in batches.items(): rule:` with `name: f"run_wflow_batch_{_b}"`), each declaring a STATIC per-batch output-list comprehension (per-cst CSV + temp() outstates paths unchanged) with members via `params:` — no wildcard-dependent outputs, no input function, no checkpoint; probe `Snakefile_looprules` (same dir) dry-runs clean on the same pinned Snakemake with rule all resolving every per-cst target to its generated batch rule. §6.1 rebuilt (construct + follow-on facts: per-batch rule names, batch-id log/benchmark naming under the retained 3.10_run_wflow part dir, per-cst driver-log preservation unchanged, rule all/3.11 resolution unchanged, no-precedent statement now covers loop-generated rules); §9 commit 2 + §10 name the mechanism and add the ruling-mandated minimal dry-run demonstration (probe pattern re-run against the real rules as part of commit 2's gate). Supersedes the construct half of the accepted arch-4/repo-3 resolution. | design-v4.md |


---

## Process observations (run log, verbatim)

# Process observations — p33-performance

- 2026-07-24 external-r1 attempt 1: codex exec transport failure — Windows
  sandbox runner could not spawn the WindowsApps (Store) pwsh 7.6.3
  (CreateProcessAsUserW error 5, ACL-restricted dir); review never started.
  p32b rounds succeeded earlier the same day with the identical invocation —
  codex's shell choice is nondeterministic. Recovery: fresh re-dispatch with
  explicit prompt guidance to use System32 powershell.exe/cmd.exe, not
  WindowsApps pwsh. Skill-improvement candidate: bake the shell guidance into
  the codex-adapter dispatch template.
- 2026-07-24 ~20:05: two tracked files (dev/scripts/stage_data.py,
  tests/_stage_equiv_harness.py) modified + three stage_data: commits
  (102829b, 4d77eca, 52cf6c5, 18:53-19:47) appeared on local main from a
  CONCURRENT session in this same worktree (CHIRPS+E-OBS staging work).
  Initially suspected a codex sandbox breach — transcript exonerated codex
  (no writes). Driver discipline adopted: explicit pathspecs only, no push
  while foreign mid-stream commits sit on main, run-dir artifacts
  unaffected. git-workflow parallel-agents isolation was not in effect
  because the second session was started externally, not spawned here.
- 2026-07-24 external-r2: verdict reject with a probe-verifiable blocking
  finding (callable outputs). Driver settled the fact mechanically
  (two-Snakefile probe) BEFORE arbitration — evidence-based arbitration
  worked well; candidate skill addition: "when a surviving finding is a
  checkable mechanism claim, the driver probes it before presenting
  arbitration options."

