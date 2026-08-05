ADR 0004 — Order model-root readers on a terminal build sentinel, not on a declared output

Status: proposed
Date: 2026-08-05
Deciders: Ümit Taner
Consulted: —
Supersedes: none
Revisions:
  - 2026-08-05: initial draft; raised from R9 P2 F5's unfixed root cause, and
    from the measurement below showing that F5's own fix is insufficient.

### Context

hydromt's model is a **mutable directory** that successive `setup_*` calls
rewrite in place. Snakemake's model is **immutable dataflow**: a file has one
producing rule, and declaring it as an input orders you after that rule. WF1
sits on the seam, and `models/hydrology/wflow/{staticmaps.nc,wflow_sbm.toml}`
is where the two models disagree.

Both files are **created by rule 1.03** and then **rewritten in place** by rules
1.04 (`mod.write()`/`mod.close()`), 1.05 (same), and 1.08 (`hydromt update
wflow_sbm`). Only 1.03 declares them. So Snakemake attributes both files to
1.03, and a reader that declares `staticmaps.nc` is ordered after 1.03 — not
after the last rule that writes it. The repository already works around this
with a chain of completion sentinels (`.model_built` → `reservoirs_lakes_glaciers.txt`
→ `.outputs_configured`), which is the impedance-matching layer between the two
models and is not itself the problem.

R9 P2 F5 diagnosed the consequence: rule 1.12 `plot_map` reads `staticmaps.nc`
straight off disk, declared only the gauges layer, and at `-c 3` was scheduled
concurrently with a writer. It dies with **no Python traceback** — the pixi env
sets `HDF5_USE_FILE_LOCKING = "FALSE"`, so an unprotected concurrent read aborts
below Python. F5 fixed it by anchoring 1.12 on `.outputs_configured`, rule 1.05's
sentinel, reasoning that it is "the earliest edge that means every writer of
staticmaps.nc is done".

**That reasoning is wrong, and the R9 gate run measures it.** From the fixture
tree and `benchmarks/wf1_benchmarks.md` for that `-c 3` run:

| Artifact / rule | Time |
|---|---|
| `.outputs_configured` (1.05's sentinel — the current anchor) | 22:59:20 |
| 1.12 `plot_map` outputs written (rule took 8.77 s) | 22:59:28 |
| 1.08 `add_forcing` window (rule took 12.95 s, forcing written 22:59:41) | ≈22:59:28 → 22:59:41 |
| **`staticmaps.nc` last written** | **22:59:37** |

`staticmaps.nc`'s final write falls **inside rule 1.08's window**, seventeen
seconds *after* the sentinel that is supposed to mean "all writers are done".
1.12 finished at the moment 1.08 started; the run survived on a margin of about
nine seconds, and nothing in the DAG produced that margin. 1.12 becomes runnable
when 1.05 completes, 1.08 becomes runnable when 1.07 completes, and the two are
free to overlap. A basin large enough to make 1.12 take ~18 s instead of ~9 s
puts its read on top of 1.08's write, and the failure mode is a silent abort.

So the live defect is not that 1.04's write is undeclared. It is that **with
in-place mutation the last writer is the only correct anchor, and the workflow
currently anchors on an intermediate one.** No decision means the race stays
live and the next reader inherits it, because the anchor's name
(`.outputs_configured`) reads as a completion marker for the whole model.

### Decision

We will introduce a **terminal build sentinel**, `.model_final`, declared as a
`touch()` output of the last rule that writes the model root (currently rule
1.08 `add_forcing`), and every rule that reads any model-root artifact will
declare `ancient(.model_final)` as its ordering edge. The existing per-rule
sentinel chain is retained unchanged for ordering *within* the build; the
terminal sentinel is the single edge that means "the model directory is final".
The output declarations on rules 1.03–1.08 are **not** changed. A test asserts
that every rule reading a model-root artifact declares the terminal sentinel, so
the 1.12 defect cannot recur silently.

### Consequences

*Positive*

- The measured race closes: no reader can be scheduled before 1.08's last write,
  because the edge is produced by 1.08 itself rather than by an earlier rule.
- The correctness condition becomes checkable rather than remembered. Today a
  new reader is correct only if its author knows which of five rules writes the
  model root last; after this it is correct if it declares one named file, and a
  test says so when it does not.
- The anchor's name stops lying. `.outputs_configured` means rule 1.05 finished;
  `.model_final` means what readers actually need.

*Negative*

- A fifth sentinel in the model root, for a workflow that already has three. The
  sentinel chain grows before it shrinks, and this ADR does not reduce it.
- The terminal sentinel is only as correct as the claim "1.08 is the last
  writer". If a future rule mutates the model root after 1.08, the sentinel must
  move with it. The enforcing test cannot detect that — it checks that readers
  declare the sentinel, not that the sentinel is attached to the last writer.
  **This is the residual risk and it should be stated in the Snakefile comment.**
- Readers gain an edge they did not have, so `--dry-run` job counts and the
  rulegraph change shape. Any recorded DAG render is stale.

*Neutral*

- `dev/scripts/semantic_tree_diff.py` needs one row: the model-root leaf list at
  lines ~658-659 already enumerates `.model_built` and `.outputs_configured`,
  and `.model_final` joins them. The R9 declared inventory gains one path.
- **No baseline re-record.** Sentinels are not `rule all` targets and are not
  fingerprinted by `check_baseline.py`; no artifact's content changes. This
  corrects the initial sizing of this item, which assumed a re-record.
- Rule numbering is untouched, so `naming.md` §9's stable-identifier rule and
  the `LOG_RULES` labels are unaffected.

### Alternatives considered

**Move the output declaration to the last writer.** Declare `staticmaps.nc` and
`wflow_sbm.toml` as outputs of 1.08 rather than 1.03, and drop the
`ancient(staticmaps.nc)` inputs on 1.04/1.05 (which would otherwise form a
cycle). Readers could then declare the files plainly and be ordered correctly by
Snakemake with no sentinel at all — the honest dataflow answer, and the one the
F5 write-up proposed.

Not chosen because it inverts Snakemake's failure semantics against us:
**Snakemake deletes a failed job's declared outputs** (absent
`--keep-incomplete`), so a transient failure in 1.08 — a network hiccup in
`hydromt update`, a bad forcing catalog entry — would delete the built model and
force a full rebuild from 1.03. Today a failed 1.08 leaves the model intact
because the model is 1.03's output. Trading a race for "a flaky forcing step
destroys an hour of build" is not an improvement. It would be preferred if 1.08
were made atomic (write to a temp model root, promote on success), which is a
larger change and its own decision. *This disqualifier rests on Snakemake's
documented delete-on-failure behaviour and should be confirmed empirically
before the alternative is revisited.*

**Collapse the model mutators into one rule.** Merge 1.04, 1.05 and 1.08's model
updates so a single rule owns the model root and declares it honestly. Rejected
on three counts: it destroys the per-rule log and benchmark granularity R3
deliberately introduced (each rule owns a `logs/_parts/` entry and a benchmark
row); rule numbers are stable identifiers per `naming.md` §9 and appear in
`LOG_RULES`, log paths and prose, so a merge renumbers a contract surface; and it
fights hydromt's staged `setup_*` model, which AGENTS.md's hard constraint
forbids re-engineering. It would be preferred if the three steps were ever found
to be inseparable for correctness rather than merely adjacent.

**Do nothing and document the hazard.** Add a comment naming `.outputs_configured`
as the anchor and move on. Rejected: the measurement above shows that anchor is
already insufficient, so documenting it would propagate the wrong rule. This is
the status quo and it is what the ADR exists to end.

### Related

- `dev/milestones/r09/phase-2-report.md` F5 — the original race, its measurement,
  and the incorrect anchor this ADR corrects.
- ADR 0003 — one shared region artifact; same class of question (which rule owns
  an artifact), and the precedent for deciding it in a record rather than in a
  Snakefile comment.
- `dev/reference/naming.md` §9 — rule numbers are stable identifiers, which
  constrains the merge alternative.
- `AGENTS.md`, *Hard Constraints* — do not re-engineer how hydromt's `setup_*`
  methods work internally.
- `dev/milestones/r09/closing-record.md` — carries this as an open item.
