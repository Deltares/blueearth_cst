"""Inject snake-config values into the hydromt build config.

hydromt 1.3 dropped `--opt key=value` and `--region` on the CLI, so
snake-config values that previously came in as flags
(setup_basemaps.region, setup_basemaps.res) are merged into a runtime
copy of the static `steps:`-format build-config template here.

It is also where the R07 B1 agreement check lives: the shared climate store
delineates its basin from `shared.basin.hydrography` / `basin_index`, the Wflow
build takes its hydrography from the template's `setup_basemaps`, and this rule
(1.02, the first build step) raises if the two ever name different datasets.
"""

import ast
from pathlib import Path

import yaml

#: (template key under ``setup_basemaps``, ``shared.basin`` key) pairs the
#: store's delineation and the model build must agree on (R07 B1 / ext1-01).
#: ``setup_rivers.hydrography_fn`` stays an intra-template concern, out of scope.
_BASIN_DATASET_KEYS = (
    ("hydrography_fn", "hydrography"),
    ("basin_index_fn", "basin_index"),
)


def _check_basin_dataset_agreement(
    template_path, basemaps_kwargs, hydrography, basin_index, project_config_path
):
    """Raise when the build template and ``shared.basin`` name different datasets.

    The climate store delineates its region from ``shared.basin.hydrography`` /
    ``basin_index``; the Wflow build takes its hydrography from the template's
    ``setup_basemaps``. R07 keeps them decoupled (the template stays the
    hydromt-conventional home of build datasets) but enforces agreement here, so
    a custom production template with different hydrography fails loud at the
    first build step instead of silently producing a store and a model on
    different basins. Injecting the config values into the generated build
    config is a recorded REJECTED alternative — it would override user template
    edits, and the template carries a second ``hydrography_fn`` under
    ``setup_rivers``.

    A key absent from the template is not a disagreement: only keys the template
    actually declares are compared (the shipped template declares both), and a
    ``None`` config value means the caller supplied nothing to compare.
    """
    supplied = {"hydrography": hydrography, "basin_index": basin_index}
    mismatches = [
        f"{template_path} setup_basemaps.{tmpl_key}="
        f"{basemaps_kwargs[tmpl_key]!r} vs {project_config_path} "
        f"shared.basin.{cfg_key}={supplied[cfg_key]!r}"
        for tmpl_key, cfg_key in _BASIN_DATASET_KEYS
        if supplied[cfg_key] is not None
        and tmpl_key in basemaps_kwargs
        and basemaps_kwargs[tmpl_key] != supplied[cfg_key]
    ]
    if mismatches:
        raise RuntimeError(
            "Build template and project config disagree on the basin datasets: "
            + "; ".join(mismatches)
            + ". The climate store's delineation and the Wflow build must use "
            "the same hydrography; edit one side so both name the same catalog "
            "entries."
        )


def merge_build_config(
    template_path,
    out_path,
    model_resolution,
    model_region,
    hydrography=None,
    basin_index=None,
    project_config_path="the project config",
):
    """Merge region/resolution into the template's setup_basemaps step.

    Parameters
    ----------
    template_path : str | os.PathLike
        The static ``steps:``-format hydromt build-config template.
    out_path : str | os.PathLike
        Where to write the merged runtime config (parents created).
    model_resolution : float | str
        Value for ``setup_basemaps.res`` (coerced to float).
    model_region : dict | str
        Value for ``setup_basemaps.region``. May be a Python-dict-literal
        string (as it arrives from the project config, e.g.
        ``"{'subbasin': [9.666, 0.4476], 'uparea': 100}"``); parsed via
        ``ast.literal_eval`` so it serializes as proper YAML.
    hydrography, basin_index : str, optional
        ``shared.basin.hydrography`` / ``shared.basin.basin_index`` — the
        catalog entry names the climate store delineates its region with. When
        given, they are cross-checked against the template's
        ``setup_basemaps.hydrography_fn`` / ``basin_index_fn`` and a
        disagreement raises. Never injected into the generated build config.
    project_config_path : str | os.PathLike, optional
        Named in the cross-check error so the message points at both files.

    Raises
    ------
    RuntimeError
        If the template has no ``setup_basemaps`` step under ``steps:``, or if
        its basin datasets disagree with ``shared.basin``.
    """
    template_path = Path(template_path)
    out_path = Path(out_path)

    if isinstance(model_region, str):
        model_region = ast.literal_eval(model_region)

    with template_path.open("r") as f:
        cfg = yaml.safe_load(f)

    # steps is a list of single-key dicts: [{"setup_basemaps": {...}}, ...]
    steps = cfg.get("steps", [])
    basemaps_step = next(
        (s for s in steps if isinstance(s, dict) and "setup_basemaps" in s), None
    )
    if basemaps_step is None:
        raise RuntimeError(f"{template_path} has no setup_basemaps step under steps:")
    kwargs = basemaps_step["setup_basemaps"] or {}
    # Cross-check BEFORE writing anything, so a disagreement never leaves a
    # generated config on disk.
    _check_basin_dataset_agreement(
        template_path, kwargs, hydrography, basin_index, project_config_path
    )
    kwargs["region"] = model_region
    kwargs["res"] = float(model_resolution)
    basemaps_step["setup_basemaps"] = kwargs

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import log_row, tee_to_log

        with tee_to_log(sm.log[0]):
            merge_build_config(
                template_path=sm.input.template,
                out_path=sm.output.merged,
                model_resolution=sm.params.model_resolution,
                model_region=sm.params.model_region,
                hydrography=sm.params.hydrography,
                basin_index=sm.params.basin_index,
                project_config_path=sm.params.project_config,
            )
            log_row(
                f"Prepared hydromt build config "
                f"(res={sm.params.model_resolution}, region={sm.params.model_region}) "
                f"-> {sm.output.merged}",
                module="config",
            )
