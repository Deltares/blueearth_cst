"""Record an experiment's own configuration, and freeze it once it has run.

R9 P4 commit 4. ``experiments/<id>/config/experiment.yml`` holds the experiment
id and the resolved ``workflows.run_stress_test`` section — the parameters
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

**The marker PATH is passed in, never rebuilt here.** It used to be a module
constant joined onto ``exp_dir``, which the 2026-08-11 move of WF3's run records
to ``{project_dir}/logs/wf3_run_stress_test_<experiment>.log`` invalidated:
the log is no longer under the experiment at all. A marker path this module
composes for itself is a second spelling of a name the Snakefile owns, and when
the two drift the failure is SILENT — ``has_run_successfully`` returns ``False``
forever and the freeze guard stops firing, which is the one thing this module
exists to do. Rule 3.07 hands over the same string rule 3.18 declares as its
``output:``.
"""

from pathlib import Path

import yaml

from blueearth_cst.shared.indicator_tables import retirement_preserves_results


class ExperimentConfigFrozenError(RuntimeError):
    """The experiment has already run; its configuration is settled."""


def build_experiment_config(experiment: str, experiment_cfg) -> dict:
    """The document: the id plus this experiment's own resolved section."""
    return {
        "experiment_name": experiment,
        "run_stress_test": dict(experiment_cfg or {}),
    }


def has_run_successfully(run_marker) -> bool:
    """Whether a complete WF3 run has produced this experiment's merged log.

    ``run_marker`` is the merged log's full path, as rule 3.18 declares it.
    """
    return Path(run_marker).is_file()


def _frozen_differences(recorded: dict, document: dict) -> list:
    """Which settings differ in a way that redefines the recorded results.

    Two events reach this comparison and they are not the same thing:

    * **the user changed a setting** — ``horizontime_climate: 2050 → 2085``. The
      results really are redefined and the freeze must refuse;
    * **the toolbox changed which settings EXIST** — a key retired between
      releases. The user changed nothing, and whether the results are redefined
      depends entirely on what the retirement did.

    Until 2026-08-12 they were indistinguishable: the comparison is a key-union
    diff, so a key present in the record and absent from the config reads as
    changed, and every already-run experiment refused after any such retirement
    (`t2608072234`). That was ruled per-milestone twice — correctly for
    ``aggregate_rlz``, whose removal changed the table's GRAIN, and wrongly for
    ``Tpeak``/``Tlow``, which were value-preserving.

    So the second event is now resolved by DECLARATION rather than inference:
    ``indicator_tables.RETIRED_EXPERIMENT_KEYS`` records, per retired key,
    whether existing results still mean what they did. Only whoever retires the
    key knows that, which is why it cannot be worked out here.

    **An unregistered key still counts as changed.** Forgetting to register a
    retirement must fail loud rather than silently unfreeze every experiment in
    the project — the same default, and the same reasoning, as
    ``refuse_retired_experiment_keys``.
    """
    was = recorded.get("run_stress_test") or {}
    now = document.get("run_stress_test") or {}
    changed = sorted(
        key
        for key in set(was) | set(now)
        if was.get(key) != now.get(key)
        # A key the CONFIG still declares is a live setting; a difference there
        # is the user's edit whatever the registry says about the name.
        and not (key not in now and retirement_preserves_results(key))
    )
    if recorded.get("experiment_name") != document.get("experiment_name"):
        changed.append("experiment_name")
    # Anything outside the two keys `build_experiment_config` emits: an older
    # record shape. Unexplained, so it refuses rather than being dropped.
    changed += sorted(
        key
        for key in set(recorded) | set(document)
        if key not in ("experiment_name", "run_stress_test")
    )
    return changed


def check_not_frozen(run_marker, out_path, document: dict) -> None:
    """Raise if the experiment has run and the configuration has changed.

    An unchanged rewrite is always allowed: Snakemake may re-run this rule for
    reasons that have nothing to do with the config, and failing on a no-op edit
    would make the guard fire on its own bookkeeping.

    A rewrite whose ONLY differences are transparent retirements is likewise
    allowed, and the record then migrates forward — the retired keys drop out of
    ``experiment.yml`` on the next write. See :func:`_frozen_differences`.
    """
    out_path = Path(out_path)
    if not out_path.is_file() or not has_run_successfully(run_marker):
        return
    recorded = yaml.safe_load(out_path.read_text(encoding="utf-8")) or {}
    if recorded == document:
        return
    changed = _frozen_differences(recorded, document)
    if not changed:
        return
    raise ExperimentConfigFrozenError(
        f"experiment {document.get('experiment_name')!r} has already produced "
        f"results, so its configuration is settled; changing it now would "
        f"silently redefine what those results mean.\n"
        f"  changed: {changed}\n"
        f"  recorded in: {out_path}\n"
        f"Create a NEW experiment for the changed settings."
    )


def write_experiment_config(
    run_marker, out_path, experiment: str, experiment_cfg
) -> dict:
    """Write the experiment's configuration record, refusing a frozen change."""
    document = build_experiment_config(experiment, experiment_cfg)
    check_not_frozen(run_marker, out_path, document)
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
            # Read BEFORE the write, so a record migrated forward says which keys
            # it dropped instead of the migration being invisible.
            _prev = Path(sm.output.experiment_config)
            before = (
                (yaml.safe_load(_prev.read_text(encoding="utf-8")) or {}).get(
                    "run_stress_test"
                )
                or {}
                if _prev.is_file()
                else {}
            )
            doc = write_experiment_config(
                run_marker=sm.params.run_marker,
                out_path=sm.output.experiment_config,
                experiment=sm.params.experiment,
                experiment_cfg=sm.params.experiment_cfg,
            )
            dropped = sorted(set(before) - set(doc["run_stress_test"]))
            log_row(
                f"experiment config recorded for {doc['experiment_name']!r} "
                f"({len(doc['run_stress_test'])} setting(s))"
                + (
                    f"; migrated forward, dropped retired key(s): {', '.join(dropped)}"
                    if dropped
                    else ""
                ),
                module="experiment",
            )
    else:
        raise ValueError("This script should be run from a snakemake environment")
