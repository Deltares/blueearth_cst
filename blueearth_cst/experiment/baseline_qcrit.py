"""Resolve authored Qcrit rules from WF1's historical model output only."""

from __future__ import annotations

import tomllib
from pathlib import Path

from blueearth_cst.experiment.wflow_response_reader import (
    NativeRunArtifacts,
    ResponseRequest,
    open_responses,
)
from blueearth_cst.indicators.sets.crop_water.calculations import prepare_baseline_qcrit
from blueearth_cst.shared.provenance import file_sha256


class HistoricalQcritUnavailable(ValueError):
    """WF1 has not produced an input required for historical Qcrit."""


def resolve_thresholds(
    parameters: dict,
    set_path: Path,
    *,
    prepare: bool = True,
    model_root: Path | None = None,
) -> dict:
    """Replace a baseline_quantile rule with sealed per-location thresholds.

    Omitted model roots use the WF4 project model context; legacy explicit paths
    resolve against the authored set file. Only the owned WF1 historical
    output layout is accepted; observations and retained scenario responses
    cannot be selected through this interface. Input digests enter calculation
    parameters and therefore the indicator identity, including on reuse.
    """
    thresholds = parameters.get("thresholds")
    if not isinstance(thresholds, dict) or "baseline_quantile" not in thresholds:
        return parameters
    if set(thresholds) != {"baseline_quantile"}:
        raise ValueError("baseline_quantile cannot be mixed with explicit thresholds")
    rule = thresholds["baseline_quantile"]
    legacy_keys = {"model_root", "source_id"}
    optional = {
        "years",
        "quantile",
        "convention",
        "reference_period",
        "quantile_method",
    }
    if not isinstance(rule, dict) or set(rule) - legacy_keys - optional:
        raise ValueError(
            "baseline_quantile must be a mapping of supported years/quantile settings"
        )
    if "model_root" in rule and (
        not isinstance(rule["model_root"], str) or not rule["model_root"]
    ):
        raise ValueError("baseline_quantile.model_root must be a path string")
    source_id = rule.get("source_id", "wf1-historical-model-discharge")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError(
            "baseline_quantile.source_id must identify historical model discharge"
        )
    from blueearth_cst.indicators.sets.crop_water.calculations import (
        validate_qcrit_options,
    )

    options = {key: rule[key] for key in optional if key in rule}
    if "years" in rule and rule["years"] is None:
        raise ValueError(
            "baseline_quantile.years must be all or a nonempty list of calendar years"
        )
    validate_qcrit_options(source_id=source_id, **options)
    if not prepare:
        # Validate the remaining calculation with its ordinary absent-parameter
        # declaration. This transient declaration must never enter an identity.
        return {**parameters, "thresholds": {}}
    if "model_root" in rule:
        model = (set_path.parent / rule["model_root"]).resolve()
    elif model_root is not None:
        model = Path(model_root).resolve()
    else:
        raise ValueError(
            "baseline_quantile requires WF4 project model context when model_root is omitted"
        )
    csv_path = model / "run_default/output.csv"
    toml_path = model / "wflow_sbm.toml"
    # Parse any existing model configuration even when its CSV is still absent.
    # A pending upstream output must not mask a malformed historical binding.
    toml_digest = None
    if toml_path.exists():
        toml_digest = file_sha256(toml_path)
        with toml_path.open("rb") as stream:
            config = tomllib.load(stream)
        forcing = config.get("input", {}).get("path_forcing", "")
        if not isinstance(forcing, str) or Path(forcing).name != "inmaps_historical.nc":
            raise ValueError(
                "baseline_quantile requires WF1 historical model forcing inmaps_historical.nc"
            )
    for label, path in (
        ("model TOML", toml_path),
        ("historical discharge CSV", csv_path),
    ):
        if not path.exists():
            raise HistoricalQcritUnavailable(
                f"Qcrit requires missing WF1 {label}: {path}. "
                "Run WF1 build_model including its historical simulation before WF4."
            )
    before = [file_sha256(csv_path), toml_digest]
    # Hash the exact native input bytes; recheck after reading to refuse concurrent
    # WF1 writes rather than assign an identity to a mixed snapshot.
    identity = f"{source_id};csv-sha256={before[0]};toml-sha256={before[1]}"
    responses = open_responses(
        identity, NativeRunArtifacts(csv_path, toml_path), ResponseRequest(("q",))
    )
    prepared = prepare_baseline_qcrit(responses, source_id=identity, **options)
    after = [file_sha256(path) for path in (csv_path, toml_path)]
    if before != after:
        raise ValueError("Qcrit historical model inputs changed during preparation")
    return {**parameters, "thresholds": prepared}
