"""Record which model state an experiment used (R9 P4, design *Model
reproducibility contract*).

There is ONE mutable live Wflow model, and experiments are long-lived. Without a
record, re-running an old experiment after the model is rebuilt silently mixes
new physics or state into old results. This writes
``experiments/<id>/config/model_reference.yml``: the model's relative path, a
pointer-derived digest, and the per-input hashes the digest was built from.

**A path plus a digest — the model is not copied.** Copying would duplicate a
multi-hundred-MB staticmaps per experiment to answer a question a hash answers.

**The per-input hashes are recorded, not just the digest.** A bare digest can
only report that *something* moved; the guard that reads this file has to name
the changed input, or an operator cannot act on it.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from blueearth_cst.shared.model_digest import (
    DIGEST_VERSION,
    MODEL_TOML_NAME,
    model_digest,
    model_digest_entries,
)


def build_model_reference(
    model_dir, project_dir, toml_name: str = MODEL_TOML_NAME
) -> dict:
    """The reference document for one experiment.

    ``model_path`` is stored RELATIVE to ``project_dir`` and in POSIX form, so
    the reference survives the project being moved or read on another platform
    — the same reason the digest itself hashes no absolute path.
    """
    model_dir, project_dir = Path(model_dir), Path(project_dir)
    rel_model = os.path.relpath(model_dir, project_dir).replace("\\", "/")
    return {
        "digest_version": DIGEST_VERSION,
        "model_path": rel_model,
        "model_toml": toml_name,
        "digest": model_digest(model_dir, toml_name),
        # Insertion order is the digest's own sorted order; sort_keys=False on
        # write keeps the file readable in that order rather than alphabetised
        # by chance.
        "inputs": dict(model_digest_entries(model_dir, toml_name)),
    }


def write_model_reference(
    model_dir, project_dir, out_path, toml_name: str = MODEL_TOML_NAME
) -> dict:
    """Build the reference and write it as YAML. Returns the document."""
    doc = build_model_reference(model_dir, project_dir, toml_name)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump(doc, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return doc


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import log_row, tee_to_log

        with tee_to_log(sm.log[0]):
            doc = write_model_reference(
                model_dir=sm.params.model_dir,
                project_dir=sm.params.project_dir,
                out_path=sm.output.model_reference,
            )
            log_row(
                f"model reference: {doc['model_path']} digest "
                f"{doc['digest'][:12]}... over {len(doc['inputs'])} input(s)",
                module="experiment",
            )
    else:
        raise ValueError("This script should be run from a snakemake environment")
