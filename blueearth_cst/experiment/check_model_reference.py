"""Refuse to simulate when the live model has changed since the experiment ran.

The other half of the *Model reproducibility contract*: rule 3.01c records which
model state an experiment used, and this recomputes the fingerprint and fails
loud on a mismatch **before WF3 performs any simulation work**.

**Ordering is the whole guard.** A check that runs after the work is not a
guard, it is a post-mortem: the forcing has been downscaled and the members have
run, against physics the experiment never recorded. So this rule's sentinel is
declared as an input of rule 3.09, the first rule that touches the model.

**Why the reference does not simply follow the model.** Rule 3.01c declares its
model inputs ``ancient()``, so a rebuilt model does NOT re-trigger it. That is
load-bearing rather than incidental: if the reference were rewritten whenever
the model changed, it would always match, the comparison would always pass, and
the guard would be decorative.

The failure names the changed input. A digest mismatch alone tells an operator
that something moved; it does not tell them whether the forcing was replaced or
the parameters were re-derived, and those call for different responses.
"""

from pathlib import Path

import yaml

from blueearth_cst.shared.model_digest import (
    DIGEST_VERSION,
    compare_model_digest,
    model_digest,
)


class ModelDriftError(RuntimeError):
    """The live model no longer matches what this experiment recorded."""


def compare_reference(reference: dict, model_dir) -> list[str]:
    """Differences between a recorded reference and the live model.

    Returns ``[]`` when they agree. A digest-version change is reported rather
    than silently compared: entries hashed under a different scheme are not
    comparable, and treating them as a mismatch would send an operator looking
    for a model change that never happened.
    """
    recorded_version = reference.get("digest_version")
    if recorded_version != DIGEST_VERSION:
        return [
            f"model_reference.yml was written with digest_version "
            f"{recorded_version!r}, but this toolbox computes version "
            f"{DIGEST_VERSION}. The two are not comparable; re-create the "
            f"experiment rather than reading across the change."
        ]
    toml_name = reference.get("model_toml", "wflow_sbm.toml")
    diffs = compare_model_digest(
        model_dir, reference.get("inputs", {}), toml_name
    )
    if not diffs:
        # Belt and braces: the entry list agreeing implies the digest agrees,
        # so a disagreement here would mean the digest function changed without
        # its version. Cheap to check, and it fails in the right direction.
        live = model_digest(model_dir, toml_name)
        if live != reference.get("digest"):
            diffs.append(
                f"per-input hashes agree but the digest does not "
                f"({live} vs {reference.get('digest')}) -- the digest function "
                f"changed without DIGEST_VERSION being raised"
            )
    return diffs


def check_model_reference(reference_path, model_dir, experiment: str = "") -> None:
    """Raise :class:`ModelDriftError` if the live model has drifted.

    Raises rather than returning a report: this runs as a rule, and a rule that
    reports drift and exits zero would let the simulation proceed.
    """
    reference = yaml.safe_load(Path(reference_path).read_text(encoding="utf-8"))
    diffs = compare_reference(reference, model_dir)
    if not diffs:
        return
    label = f" for experiment {experiment!r}" if experiment else ""
    raise ModelDriftError(
        f"the live Wflow model has changed since this experiment recorded "
        f"it{label}, so re-running would mix new model state into old results:\n"
        + "\n".join(f"  - {d}" for d in diffs)
        + f"\n\nRecorded in {reference_path}. Create a NEW experiment to "
        f"simulate against the current model; the recorded one is not "
        f"re-runnable against different physics or state."
    )


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import log_row, tee_to_log

        with tee_to_log(sm.log[0]):
            check_model_reference(
                reference_path=sm.input.model_reference,
                model_dir=sm.params.model_dir,
                experiment=sm.params.experiment,
            )
            log_row(
                "model reference matches the live model; simulation may proceed",
                module="experiment",
            )
    else:
        raise ValueError("This script should be run from a snakemake environment")
