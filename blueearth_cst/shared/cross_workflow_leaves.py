"""The wf1 leaves WF2/WF3 declare and Snakemake will not satisfy on its own.

A CROSS-WORKFLOW LEAF is a file some rule in WF2 or WF3 declares as an input
while no rule in that workflow declares it as an output. Snakemake cannot
produce it, so anything driving WF2/WF3 without a wf1 run behind it — a dry-run
test, the layout scaffolder, the `run_workflows` preflight — has to reckon with
it first.

**Why this is one definition rather than three.** It was three, and they drifted
independently. R9 P4's rule 3.01c `write_model_reference` is the first WF3 rule
ever to declare model FILES as inputs; before it, WF3 reached the model only
through `params` and the DAG could not see the dependency. Two of the three
copies were updated for it and one was not, so `test_guard_invalidation`'s gate
2c(iii) went red in a way that read as a guard defect (R9 P5 F3). The third copy,
`scaffold_project_tree.py`, was never updated at all and had also fallen two
milestones behind on the model root, so WF3 contributed 0 of its 95 declared
outputs while the tool still exited 0.

**A shared list still goes stale — it just does so in one place.** What stops it
is `tests/test_cross_workflow_inputs.py`, which proves against the real DAG that
`LEAVES` is COMPLETE (staging exactly it lets WF2 and WF3 dry-runs resolve) and
MINIMAL (drop any one and a dry-run fails). A rule declaring a new leaf turns
that test red immediately, which is the escape above. Every consumer here
inherits that proof; a hand-kept second copy would inherit nothing.

**Why this lives in `shared/` and not beside its staging helper.** The list was
defined in `dev/scripts/cross_workflow_inputs.py` until 2026-08-17, which was
right while every consumer was a test or a dev tool. `scripts/run_workflows.py`
then needed it for its wf1 preflight, and that is a RUN path — AGENTS.md's
invocation-model split makes `dev/scripts/` "never part of a run", so importing
it there would have made a dev-only tree a runtime dependency of the user-facing
wrapper. The paths moved here; the STAGING machinery (`stage`, `content_for`,
the `EXTRA_*` non-leaves and the minimal file bodies) stayed there, because it
is test-fixture code with no run-path caller. `cross_workflow_inputs` re-exports
these names, so its own consumers were unaffected by the move.

This module is deliberately paths-and-constants only — no I/O, no filesystem
checks — so that both a fixture stager and a preflight can build what they each
need on top of the same list.
"""

from __future__ import annotations

#: WF3 rule 3.01 `check_project_consistency` takes this as a mandatory
#: `ancient()` input; its absence is a rule-level MissingInputException.
LEAF_WF1_SNAPSHOT = "config/runs/snake_config_build_model.yml"

#: WF3 rule 3.01c `write_model_reference` (R9 P4), the first WF3 rule to declare
#: model files as inputs. Both are `ancient()`: the reference must not re-derive
#: because the model was rebuilt, only because the config moved.
LEAF_MODEL_TOML = "models/hydrology/wflow/wflow_sbm.toml"
LEAF_MODEL_READY = "models/hydrology/wflow/.outputs_configured"

#: The full set. Order is stable so failure messages read the same way twice —
#: and it is DAG order, so the first entry is the one Snakemake would have
#: reported had the run been allowed to start.
LEAVES: tuple[str, ...] = (LEAF_WF1_SNAPSHOT, LEAF_MODEL_TOML, LEAF_MODEL_READY)

#: The workflow that produces every leaf above. Named rather than spelled at
#: each call site, so a message can say what to RUN and not only what is absent.
LEAF_PRODUCER = "build_model"
