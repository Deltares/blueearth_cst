"""WF0/WF1 producer helpers for named forcing recipes: stores and run names.

Separate from ``named_forcings`` so WF4's identity-bearing readers of the
recipe layout do not import the forcing normalizer, and outside
``model/historical_runs.py``, a declared scientific-code input of every
retained historical run.
"""

from __future__ import annotations

from collections.abc import Mapping


def forcing_store_sources(climate: Mapping[str, object]) -> frozenset[str]:
    """Sources a declared named recipe extracts into the plain source store.

    A recipe with its source's default extraction shares the
    ``<source>_<window>`` folder with that source's comparison candidate. Every
    declarer of that folder must then use the full forcing contract, or WF0 and
    WF1 re-extract it against each other. Legacy projects return an empty set.

    Comparison entries (WF0) receive recipes in their public, unvalidated form;
    those are resolved here. A draft that does not resolve is skipped: WF1
    refuses such a configuration, so it declares no store for it.
    """
    from blueearth_cst.shared.forcing_recipe import (
        normalize_forcing_recipe,
        resolve_climate_forcing,
    )

    sources = set()
    for recipe in (climate.get("forcings") or {}).values():
        if "selected" not in recipe:
            try:
                recipe, _ = resolve_climate_forcing(
                    {
                        "sources": climate.get("sources", []),
                        "forcing": recipe["forcing"],
                    },
                    required=True,
                    notices=False,
                )
            except ValueError:
                continue
        source = recipe["selected"]
        default = normalize_forcing_recipe(source).extraction_fingerprint
        if (
            normalize_forcing_recipe(
                source, recipe.get("forcing")
            ).extraction_fingerprint
            == default
        ):
            sources.add(source)
    return frozenset(sources)


def retained_run_names(
    name: str | None, forcing_ids: tuple[str, ...] | None
) -> dict[str | None, str | None]:
    """Map each executed recipe to its retained run name.

    Legacy layout keeps the authored name. Named layout derives ``<name>_<id>``;
    two or more recipes require a name so the comparison reads retained runs.
    """
    if forcing_ids is None:
        return {None: name}
    if name is None:
        if len(forcing_ids) > 1:
            raise ValueError(
                "build_model.historical_run.name is required when "
                "historical_forcings lists two or more recipes"
            )
        return {forcing_ids[0]: None}
    from blueearth_cst.model.historical_runs import historical_run_name

    return {
        forcing_id: historical_run_name({"name": f"{name}_{forcing_id}"})
        for forcing_id in forcing_ids
    }
