"""Checks WF1 runs before it uses a registered ``user_managed`` model (t2609281102).

WF1 never rebuilds or rewrites a registered model: its build steps (1.06-1.08)
are not defined for it, and every forcing recipe writes its own keyed TOML. So
the model must already declare the outputs the project asks for, which a model
copied from the toolbox-built ``wflow`` does. A missing declaration is refused
here, naming it, rather than added to the user's files.
"""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from blueearth_cst.model.setup_gauges_and_outputs import WFLOW_VARS
from blueearth_cst.shared.model_registry import require_unchanged
from blueearth_cst.shared.wflow_outputs import code_for


def missing_output_declarations(
    toml_path: str | Path, outvars: list[str], netcdf_grid: dict | None
) -> list[str]:
    """The configured outputs the model TOML does not declare; ``[]`` when complete."""
    with open(toml_path, "rb") as handle:
        output = tomllib.load(handle).get("output") or {}
    columns = {
        (column.get("header"), column.get("map"), column.get("parameter"))
        for column in (output.get("csv") or {}).get("column") or []
    }
    discharge = WFLOW_VARS["river discharge"]
    expected = [("Q", "outlets", discharge)]
    if "gauges_locations" in {column[1] for column in columns}:
        expected.append(("Q", "gauges_locations", discharge))
    expected += [
        (code_for(name), "subcatchment", WFLOW_VARS[name])
        for name in outvars
        if name != "river discharge"
    ]
    missing = [
        f"output.csv column header={header!r} map={mapname!r}"
        for header, mapname, parameter in expected
        if (header, mapname, parameter) not in columns
    ]
    if netcdf_grid is not None and "netcdf_grid" not in output:
        missing.append("output.netcdf_grid (build_model.output requests a grid)")
    if netcdf_grid is None and "netcdf_grid" in output:
        missing.append("no output.netcdf_grid (build_model.output requests none)")
    return missing


def check_registered_model(
    model_root: str | Path, outvars: list[str], netcdf_grid: dict | None
) -> None:
    """Refuse a registered model that changed or lacks the configured outputs."""
    record = require_unchanged(model_root)
    toml = Path(model_root) / record["model_toml"]
    missing = missing_output_declarations(toml, outvars, netcdf_grid)
    if missing:
        raise ValueError(
            f"registered model {record['name']!r} does not match the configured "
            f"outputs ({'; '.join(missing)}). WF1 does not edit a registered model: "
            "declare these in its TOML and register it under a new name, or change "
            "model.outvars / build_model.output."
        )
