"""Write the staleness sidecar that ties a workflow's outputs to its record.

The run journal identifies invocations; it cannot say which OUTPUTS came from
which. This sidecar closes that: it sits beside a workflow's terminal products
and carries the same digests ``run_record.yml`` carries, so

    sidecar.configuration_inputs_sha256 != run_record.configuration_inputs_sha256

means the outputs predate the most recently recorded configuration and toolbox
state -- and the journal then names the invocations on either side.

The wide digest includes the toolbox revision. It is read from the captured
archive when this rule executes, rather than used as a scheduling parameter:
a new commit or dirty-state change alone must not refresh the provenance of
reused outputs. Settings, referenced-file contents, environment changes and
updated upstream products still refresh this sidecar. A code-only invocation
is recorded in the launcher archive and journal; the sidecar retains its
previous digest until the outputs or their configuration inputs change.

The sidecar is a separate declared output; it never adds metadata columns to
scientific indicator tables or changes their retained numerical values.
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
    from blueearth_cst.shared.workflow_archive_launch import require_capture

    sm = globals()["snakemake"]
    record = require_capture(
        sm.params.workflow_name, sm.params.config_path, dry_run=False
    )
    write_run_metadata(
        path=sm.output.run_metadata,
        workflow=sm.params.workflow_name,
        effective_config_sha256=sm.params.effective_config_sha256,
        configuration_inputs_sha256=record["configuration_projection"][
            "configuration_inputs_sha256"
        ],
        experiment=getattr(sm.params, "experiment", None),
    )
