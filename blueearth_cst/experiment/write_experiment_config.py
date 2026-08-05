"""Record an experiment's own configuration, and freeze it once it has run.

R9 P4 commit 4. ``experiments/<id>/config/experiment.yml`` holds the experiment
id and the resolved ``workflows.climate_experiment`` section — the parameters
that define *this* experiment, beside the ``model_reference.yml`` that records
which model it used.

Generated, not authored. A hand-written file here would be a second source of
truth competing with the ``--configfile``, which is the
``config/project.yml`` direction the R9 master brief puts out of scope. Being
generated also means it always matches what actually ran.

**Immutable at the FIRST SUCCESSFUL RUN, not at creation.** Editing an
experiment's parameters before it has produced anything is ordinary work; doing
so afterwards would silently redefine what the existing results mean. Freezing
at creation would be a different and worse feature — it would forbid the legal
case to make the illegal one easy.

The marker for "has run successfully" is the merged workflow log. It is written
by the last rule in WF3 and is one of ``rule all``'s targets, so it exists only
after a complete run: a run that failed midway never reaches the merge. The
marker is read from the filesystem rather than declared as an input, because
declaring it would invert the DAG — this file is written long before the log.
"""

from __future__ import annotations

from pathlib import Path

import yaml

#: Written by WF3's last rule; present only after a complete successful run.
RUN_MARKER = "logs/wf3_climate_experiment.log"


class ExperimentConfigFrozenError(RuntimeError):
    """The experiment has already run; its configuration is settled."""


def build_experiment_config(experiment: str, experiment_cfg) -> dict:
    """The document: the id plus this experiment's own resolved section."""
    return {
        "experiment_name": experiment,
        "climate_experiment": dict(experiment_cfg or {}),
    }


def has_run_successfully(exp_dir) -> bool:
    """Whether a complete WF3 run has produced this experiment's merged log."""
    return (Path(exp_dir) / RUN_MARKER).is_file()


def check_not_frozen(exp_dir, out_path, document: dict) -> None:
    """Raise if the experiment has run and the configuration has changed.

    An unchanged rewrite is always allowed: Snakemake may re-run this rule for
    reasons that have nothing to do with the config, and failing on a no-op edit
    would make the guard fire on its own bookkeeping.
    """
    out_path = Path(out_path)
    if not out_path.is_file() or not has_run_successfully(exp_dir):
        return
    recorded = yaml.safe_load(out_path.read_text(encoding="utf-8")) or {}
    if recorded == document:
        return
    changed = sorted(
        key for key in set(recorded.get("climate_experiment", {}))
        | set(document.get("climate_experiment", {}))
        if recorded.get("climate_experiment", {}).get(key)
        != document.get("climate_experiment", {}).get(key)
    )
    raise ExperimentConfigFrozenError(
        f"experiment {document.get('experiment_name')!r} has already produced "
        f"results, so its configuration is settled; changing it now would "
        f"silently redefine what those results mean.\n"
        f"  changed: {changed or ['experiment_name']}\n"
        f"  recorded in: {out_path}\n"
        f"Create a NEW experiment for the changed settings."
    )


def write_experiment_config(exp_dir, out_path, experiment: str, experiment_cfg) -> dict:
    """Write the experiment's configuration record, refusing a frozen change."""
    document = build_experiment_config(experiment, experiment_cfg)
    check_not_frozen(exp_dir, out_path, document)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump(document, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return document


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import log_row, tee_to_log

        with tee_to_log(sm.log[0]):
            doc = write_experiment_config(
                exp_dir=sm.params.exp_dir,
                out_path=sm.output.experiment_config,
                experiment=sm.params.experiment,
                experiment_cfg=sm.params.experiment_cfg,
            )
            log_row(
                f"experiment config recorded for {doc['experiment_name']!r} "
                f"({len(doc['climate_experiment'])} setting(s))",
                module="experiment",
            )
    else:
        raise ValueError("This script should be run from a snakemake environment")
