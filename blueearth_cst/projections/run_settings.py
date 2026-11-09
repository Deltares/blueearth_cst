"""Compact WF2 console settings from configured variables and resolved members."""

from collections.abc import Mapping, Sequence

from blueearth_cst.projections.resolution import Combination
from blueearth_cst.projections.variable_spec import VariableSpec
from blueearth_cst.shared.snake_utils import listed


def projection_run_settings(
    models: Sequence[str],
    scenarios: Sequence[str],
    combinations: Sequence[Combination],
    variables: Mapping[str, VariableSpec],
) -> dict[str, str]:
    """Count unique resolved model–member pairs, independent of scenario repeats."""
    short = [model.rsplit("/", 1)[-1] for model in models]
    labels = short if len(set(short)) == len(short) else list(models)
    members = {model: set() for model in models}
    by_scenario = {
        (model, scenario): set() for model in models for scenario in scenarios
    }
    for item in combinations:
        if item.resolved and (item.dataset, item.scenario) in by_scenario:
            members[item.dataset].add(item.member)
            by_scenario[item.dataset, item.scenario].add(item.member)
    counts = [len(members[model]) for model in models]
    total = sum(counts)
    if len(models) <= 3:
        ensemble = (
            "; ".join(
                f"{label}: {count}" for label, count in zip(labels, counts, strict=True)
            )
            + f" ({total} total)"
        )
    else:
        low, high = min(counts), max(counts)
        span = str(low) if low == high else f"{low}–{high}"
        noun = "member" if low == high == 1 else "members"
        ensemble = f"{total} model–member pairs; {span} {noun} per model"
    if any(
        len({frozenset(by_scenario[model, scenario]) for scenario in scenarios}) > 1
        for model in models
    ):
        ensemble += "; varies by scenario"
    settings = {
        "climate models": f"{listed(labels, limit=3)} ({len(models)} total)",
        "ensemble members": ensemble,
        "variables": listed(variables),
    }
    if "pet" in variables and variables["pet"].method:
        settings["PET method"] = {"debruin": "de Bruin", "makkink": "Makkink"}[
            variables["pet"].method
        ]
    if "tdew" in variables and variables["tdew"].method:
        settings["dew-point method"] = "Specific humidity and surface pressure"
    if "wind" in variables and variables["wind"].method:
        settings["wind statistic"] = (
            "Monthly mean wind speed at 10 m"
            if variables["wind"].method == "sfcwind"
            else "Magnitude of monthly mean wind vector at 10 m"
        )
    return settings
