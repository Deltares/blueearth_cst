Task Brief — R9 P4 commit 4: experiment ID collisions and config immutability

**Supersedes items 5–6 of** [`phase-4-fingerprint-task-brief.md`](phase-4-fingerprint-task-brief.md).
Written 2026-08-04 after P4 commits 1–3 landed and the original items were found
to diverge from the code in four ways ([`phase-4-report.md`](phase-4-report.md),
*Commit 4: why it stopped*). The fingerprint half is complete and merged; this
is the remaining half, re-specified against what the code actually does.

### Ground truth — verified, not assumed

Read this section before the checklist. Four claims in the original brief do not
hold, and each changes what the work is.

| Original brief | Reality |
| --- | --- |
| default is `stress_test_<YYYYMMDD>` | `suggest_experiment_name` returns **`<project-basename>_<YYYYMMDD>`** — `test_local/` yields `test_local_20260804`. The string `stress_test_` appears nowhere in the codebase |
| `experiment.yml` becomes immutable at first run | **Nothing writes `experiment.yml`.** It is named in design tree v10 and by nothing else |
| "reservation must be atomic" | **There is no reservation step.** A name is proposed, written into a config, and the directory appears when a rule first writes into it |
| scope is `blueearth_cst/experiment/**` | The creation path is **`scripts/suggest_experiment_name.py`**, a user-facing runner |

What already exists and must not be re-invented:

- **`snake_utils.validate_experiment_name`** — grammar `^[a-z0-9][a-z0-9_]*$`,
  ≤64 chars, uppercase rejected (never silently lowercased), Windows reserved
  device names rejected, and a containment assertion that the target is a direct
  child of `<project_dir>/experiments`. Called at WF3 parse time before `exp_dir`
  is built. **It performs no collision check** — that is the gap.
- **`snake_utils.suggest_experiment_name`** — slugifies the `project_dir`
  basename, appends the date, and passes the result back through
  `validate_experiment_name`. A *suggestion writer*, deliberately never a
  runtime generator: a name derived at run time would make every invocation
  target a fresh `experiments/<id>/`, so nothing would be up to date and
  `--dry-run` would mislead.
- **`scripts/suggest_experiment_name.py`** — writes the value into a config once
  and **refuses to overwrite an existing `experiment_name`**.

### Four decisions the owner must make first

This brief does not settle them. Each changes the implementation materially.

1. **Which names get a `_v2` suffix?** The original reserved suffixing for "the
   generated default". With the default being `<basename>_<date>`, does
   suffixing apply to any generated name, or should a `stress_test_` default be
   introduced (a behaviour change to `suggest_experiment_name`)?
2. **What is in `experiment.yml`?** Design tree v10 names the file and stops
   there. Minimum viable: the experiment's own `workflows.climate_experiment`
   section. Alternatives: the full resolved config, or a hand-authored input.
   This decides whether it is generated (and by which rule) or authored.
3. **Atomic against what?** Two concurrent creations on one machine, or the
   general case? `os.mkdir` is atomic on both platforms and would suffice for
   the first; anything stronger needs a stated threat model.
4. **May the scope include `scripts/`?** ID allocation lives there. Without it,
   only the immutability half is reachable.

### Goal

Experiment creation cannot silently reuse or overwrite an existing experiment,
and an experiment's configuration stops being editable once it has produced
results.

### Non-goals

- **`config/project.yml`** — out of scope for the whole program.
- No change to `validate_experiment_name`'s grammar. It is correct and is
  depended on at WF3 parse time.
- No change to the fingerprint (P4 commits 1–3) or to rule 3.00b's sentinels.
- No runtime name generation. The suggestion-writer stance is load-bearing.

### Allowed scope

**Permitted** — `blueearth_cst/experiment/**`; `blueearth_cst/shared/snake_utils.py`
(the two name helpers only); `Snakefile_climate_experiment`; tests.

**Approval-gated** — `scripts/suggest_experiment_name.py` (decision 4);
rule 3.00b's declared inputs and sentinel paths.

**Forbidden** — `config/project.yml`; `dev/baseline/**`; the grammar in
`validate_experiment_name`.

### Required changes (checklist)

Conditional on the decisions above; written for the recommended reading.

1. **Collision rejection.** A **user-supplied** `experiment_name` naming an
   existing `experiments/<id>/` is rejected with an error naming the existing
   experiment and what to do. Distinguish *reuse* (resuming the same experiment
   — allocates nothing, must stay legal, or incremental reruns break) from
   *collision* (a new experiment claiming an occupied name).
2. **Suffixing for generated names only.** A generated name that collides gets
   `_v2`, then `_v3`. A user-supplied one never does — silently renaming what a
   human chose is the surprise `validate_experiment_name` already refuses to
   make with case.
3. **Reservation.** Directory creation is the reservation; use an atomic
   primitive (`os.mkdir`, not `exists()`-then-`mkdir`).
4. **`experiment.yml`,** per decision 2, written into
   `experiments/<id>/config/`.
5. **Immutability at first successful run, not at creation.** The trigger must
   be an artifact that exists only after a successful run. Candidates already
   present: the merged `logs/wf3_climate_experiment.log`, or the two
   `results/*.csv`. Editing before the first run must stay legal.

### Commit plan

| # | Subject | Invariant preserved |
|---|---|---|
| 1 | `r09: reject colliding experiment names` | resume still allocates nothing — the incremental-execution constraint |
| 2 | `r09: suffix generated experiment names on collision` | user-supplied names are never silently renamed |
| 3 | `r09: write experiment.yml per experiment` | the file exists before anything guards it |
| 4 | `r09: freeze experiment.yml after the first successful run` | the guard lands only once the file it guards exists |

### Validation

**Named scope:** `tests/test_validate_experiment_name.py`,
`tests/test_suggest_experiment_name.py`, the new lifecycle tests, and
`pixi run test-cli` after each Snakefile edit. This touches no WF1/WF2 code.

**Falsifiers.** Each asserts something must NOT happen, so a test that only
confirms the happy path is half the job:

- **Resume is not a collision** — re-running an existing experiment ID allocates
  nothing and is not rejected. *This is the one that can break the pipeline*: if
  resume is treated as a collision, every incremental rerun fails.
- **A user-supplied duplicate is rejected**, and the message names the existing
  experiment.
- **A generated same-day duplicate becomes `_v2`, then `_v3`** — the third
  collision is the discriminator; an implementation that only handles the second
  passes a `_v2`-only test.
- **Reservation is atomic** — two creations racing for one name produce one
  winner and one clean error, never two experiments believing they own it.
  Demonstrate with concurrent attempts, not by inspecting the code.
- **Immutability has both directions** — `experiment.yml` is writable *before*
  the first successful run and refused *after*. A test that only checks the
  refusal would pass against a file frozen at creation, which is the behaviour
  this explicitly rejects.

### Acceptance criteria

- Every falsifier passes, including resume-is-not-a-collision.
- `validate_experiment_name`'s grammar is unchanged and its callers unaffected.
- Three clean `--dry-run`s; `pixi run test-fast` green.
- The four decisions are recorded in the phase report with their rulings.
- Rollback: if immutability cannot be tied to a *successful run* — as opposed to
  creation — revert commit 4 and report. Freezing at creation is not a weaker
  version of this feature; it is a different and worse one.

### Output requirements

A phase report giving each falsifier its result, the four decisions and how they
were ruled, and — if `scripts/` stayed out of scope — an explicit statement of
which half of the lifecycle rule shipped.

### Task constraints

- Do not make `suggest_experiment_name` a runtime generator.
- Do not silently rename a user-supplied name under any circumstance.
- A collision error must name the existing experiment; "name already exists" is
  not actionable when the user cannot see which project it is in.
