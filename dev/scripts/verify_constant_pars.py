"""One-time landing + precedence check for ADR 0001 constant-parameter restoration.

ADR 0001 (t260719a) steps 3a/3b. `setup_constant_pars` does NOT write a map into
`staticmaps.nc`; for each recognized CSDMS name it writes a scalar TOML entry
`[input.static.<name>] value = <v>` in `wflow_sbm.toml`
(hydromt_wflow/wflow_base.py). This script asserts, for a BUILT model, that every
constant the restored config asks for actually landed there with the right value,
and guards the two silent modes the ADR names:

  3a. Landing assertion — each expected `input.static.<name>.value` is present and
      equals the reference value (catches a typo that silently mapped to a
      different recognized name, or a value the plugin never wrote).
  3b. Precedence / inactivity — confirm the name landed as a SCALAR (not as a
      per-cell staticmaps layer string that would shadow it), that no
      `staticmaps.nc` variable collides with the CSDMS name, and report which
      owning modules are active (`snow__flag`, `glacier__flag`) so inert-here
      constants (glacier on a `glacier__flag=false` basin) are read correctly.

It is a diagnostic, NOT a standing test: `test_cli.py` is dry-run only and
`test_model_creation.py` does not build, so this has no home in the suite (ADR
step 3). Run it against a restored build during implementation and paste the
report into dev/decisions/0001-restore-wflow-constant-parameters/baseline_diffs.md.

    python dev/scripts/verify_constant_pars.py \
        --model-dir test_case/test_local/models/hydrology/wflow \
        --config dev/decisions/0001-restore-wflow-constant-parameters/config_restored.yml

Exit 0 iff every expected constant landed as a correct scalar with no shadowing.
Run against the current (baseline) model it will report the 13 not-yet-restored
constants as ABSENT and exit 1 — that is the expected "before" state.
"""
from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

import yaml

# Restored CSDMS names whose owning module can be inactive on a given basin.
# Used only to annotate the report (an inert constant still must land correctly).
GLACIER_PARS = frozenset({
    "glacier_ice__degree_day_coefficient",
    "glacier_firn_accumulation__snowpack_dry_snow_leq_depth_fraction",
    "glacier_ice__melting_temperature_threshold",
})
SNOW_PARS = frozenset({
    "snowpack__degree_day_coefficient",
    "atmosphere_air__snowfall_temperature_threshold",
    "atmosphere_air__snowfall_temperature_interval",
    "snowpack__melting_temperature_threshold",
    "snowpack__liquid_water_holding_capacity",
})

FLOAT_RTOL = 1e-9


def load_expected(config_path: Path) -> dict[str, float]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for step in cfg.get("steps", []):
        if "setup_constant_pars" in step:
            return {str(k): float(v) for k, v in step["setup_constant_pars"].items()}
    raise ValueError(f"{config_path}: no setup_constant_pars block found")


def classify(name: str, expected: float, static: dict) -> tuple[str, str]:
    """Return (status, detail). status in {OK, MISMATCH, SHADOWED, ABSENT}."""
    if name not in static:
        return "ABSENT", "no input.static entry (falls back to Wflow.jl default)"
    entry = static[name]
    if isinstance(entry, str):
        # Per-cell staticmaps layer reference -> shadows any scalar (silent mode 1).
        return "SHADOWED", f"mapped to per-cell layer '{entry}', not a constant scalar"
    if isinstance(entry, dict) and "value" in entry:
        got = float(entry["value"])
        if abs(got - expected) <= FLOAT_RTOL * max(abs(expected), abs(got), 1.0):
            return "OK", f"value = {got}"
        return "MISMATCH", f"value = {got}, expected {expected}"
    return "ABSENT", f"input.static entry present but no scalar 'value' ({entry!r})"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model-dir", type=Path,
                   default=Path("test_case/test_local/models/hydrology/wflow"),
                   help="Built model dir (holds wflow_sbm.toml + staticmaps.nc)")
    p.add_argument("--config", type=Path,
                   default=Path("dev/decisions/0001-restore-wflow-constant-parameters/config_restored.yml"),
                   help="Build config whose setup_constant_pars block is the reference")
    args = p.parse_args()

    toml_path = args.model_dir / "wflow_sbm.toml"
    if not toml_path.exists():
        sys.stderr.write(f"No wflow_sbm.toml under {args.model_dir}\n")
        return 2

    expected = load_expected(args.config)
    toml_data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    static = toml_data.get("input", {}).get("static", {})
    model = toml_data.get("model", {})
    snow_on = bool(model.get("snow__flag", False))
    glacier_on = bool(model.get("glacier__flag", False))

    # staticmaps variable names, for the collision arm of the precedence check.
    static_vars: set[str] = set()
    nc_path = args.model_dir / "staticmaps.nc"
    if nc_path.exists():
        import xarray as xr

        with xr.open_dataset(nc_path) as ds:
            static_vars = set(map(str, ds.variables))

    print(f"# Landing + precedence check ({toml_path})")
    print(f"# snow__flag = {snow_on}   glacier__flag = {glacier_on}")
    print(f"# {len(expected)} expected constant(s) from {args.config}\n")

    ok = 0
    failures: list[str] = []
    for name in expected:
        status, detail = classify(name, expected[name], static)
        notes = []
        if name in static_vars:
            notes.append(f"COLLISION: staticmaps.nc has a variable named '{name}'")
        if name in GLACIER_PARS and not glacier_on:
            notes.append("inert here (glacier__flag=false) — certified by landing, not discharge")
        if name in SNOW_PARS and not snow_on:
            notes.append("inert here (snow__flag=false)")
        note = ("  [" + "; ".join(notes) + "]") if notes else ""
        print(f"  {status:9s} {name} -- {detail}{note}")
        if status == "OK" and not any(n.startswith("COLLISION") for n in notes):
            ok += 1
        else:
            failures.append(name)

    print(f"\n{ok}/{len(expected)} constant(s) landed correctly as scalars with no shadowing.")
    if failures:
        print(f"NOT OK: {len(failures)} -> {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
