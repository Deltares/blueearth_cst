"""Write the staleness sidecar that ties a workflow's outputs to its record.

The run journal identifies invocations; it cannot say which OUTPUTS came from
which. This sidecar closes that: it sits beside a workflow's terminal products
and carries the same digests ``run_record.yml`` carries, so

    sidecar.configuration_inputs_sha256 != run_record.configuration_inputs_sha256

means the outputs predate the most recently recorded configuration and toolbox
state -- and the journal then names the invocations on either side.

Why the WIDE digest is the staleness test: ``effective_config_sha256`` moves
only when the settings move, so a changed custom template, an in-place catalog
edit, or a toolbox commit would leave stale outputs matching a fresh record and
reading as current. Both fields are written -- the narrow one answers "same
settings?", the wide one "same settings, code and inputs?".

The sidecar is a NEW declared output, never an addition to an existing table:
WF3's ``results/q_indicators.csv`` is baseline-fingerprinted, so writing into it
would falsify the design's no-re-record claim.
"""

import json
from pathlib import Path
from typing import Optional, Union

SCHEMA_VERSION = 1


def build_run_metadata(
    workflow: str,
    effective_config_sha256: str,
    configuration_inputs_sha256: str,
    experiment: Optional[str] = None,
) -> dict:
    """Assemble the sidecar document."""
    document = {
        "schema_version": SCHEMA_VERSION,
        "workflow": workflow,
        "effective_config_sha256": effective_config_sha256,
        "configuration_inputs_sha256": configuration_inputs_sha256,
    }
    if experiment is not None:
        document["experiment"] = experiment
    return document


def write_run_metadata(
    path: Union[str, Path],
    workflow: str,
    effective_config_sha256: str,
    configuration_inputs_sha256: str,
    experiment: Optional[str] = None,
) -> None:
    """Write the sidecar as JSON beside the workflow's terminal outputs."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    document = build_run_metadata(
        workflow,
        effective_config_sha256,
        configuration_inputs_sha256,
        experiment,
    )
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__" and "snakemake" in globals():
    sm = globals()["snakemake"]
    write_run_metadata(
        path=sm.output.run_metadata,
        workflow=sm.params.workflow_name,
        effective_config_sha256=sm.params.effective_config_sha256,
        configuration_inputs_sha256=sm.params.configuration_inputs_sha256,
        experiment=getattr(sm.params, "experiment", None),
    )
