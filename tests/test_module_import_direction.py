"""R7-4: the model-free climate path must not import from the model package.

`blueearth_cst/climate_analysis/` exists so a full climate analysis runs from
region + catalog alone, with no wflow setup or run (design P4). An import from
`blueearth_cst/model/` does not necessarily COUPLE anything -- the parity
transform it used to reach for is pure xarray in/out -- but the direction
contradicts the one claim the package is there to make, and a convention that
nothing checks drifts back.

The transform now lives in `shared/`, which is where genuinely engine-neutral
helpers belong: two callers use it from opposite sides, `model/plot_results.py`
at model parity and `climate_analysis/plot_climate_source.py` on the source
grid.
"""

import ast
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "blueearth_cst"
CLIMATE = PKG / "climate_analysis"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module)
    return out


@pytest.mark.parametrize(
    "module", sorted(p.name for p in CLIMATE.glob("*.py") if p.name != "__init__.py")
)
def test_climate_analysis_does_not_import_the_model_package(module):
    offenders = {
        m
        for m in _imported_modules(CLIMATE / module)
        if m.startswith("blueearth_cst.model")
    }
    assert not offenders, (
        f"{module} imports {sorted(offenders)}. climate_analysis/ is the "
        f"model-free path (P4); engine-neutral helpers belong in "
        f"blueearth_cst/shared/, not blueearth_cst/model/."
    )


def test_the_parity_transform_is_reachable_from_shared():
    """Pin the new home, so a well-meaning revert is a failing test."""
    from blueearth_cst.shared.climate_parity import model_parity_climate

    assert callable(model_parity_climate)


def test_the_parity_transform_is_engine_neutral():
    """It earns its place in shared/ only while it imports no model code and
    touches no model object -- the P3-2a C1 criterion its docstring claims."""
    offenders = {
        m
        for m in _imported_modules(PKG / "shared" / "climate_parity.py")
        if m.startswith("blueearth_cst.model")
    }
    assert not offenders, f"climate_parity reaches into the model package: {offenders}"


# --- t2609171028: the console tier must stay OUT of the code fingerprint -----

SHARED = PKG / "shared"


def test_snake_utils_does_not_import_the_console_tier():
    """The invariant the console/snake_utils split rests on, and nothing else checks.

    `experiment/content_identity.repository_code_inventory` fingerprints a
    transitive static-import closure and hashes every file in it. `snake_utils`
    is in all three closures and belongs there. `console_style` is in none, and
    that is the whole point: a banner edit must not invalidate a generated
    scenario collection.

    A single convenience re-export here -- `from ... console_style import
    rule_banner`, to spare a Snakefile one import line -- pulls the console tier
    back into all three closures and silently restores the ~12-minute
    regeneration this split removed. Nothing about the console output would look
    wrong, and no other test would fail.

    The dependency runs one way only: `console_style` imports from
    `snake_utils`, never the reverse.
    """
    offenders = {
        module
        for module in _imported_modules(SHARED / "snake_utils.py")
        if "console_style" in module
    }
    assert not offenders, (
        f"snake_utils must not import the console tier, found: {sorted(offenders)}; "
        "see this test's docstring -- it puts console_style back in the fingerprint"
    )


def test_console_style_is_absent_from_every_fingerprinted_closure():
    """The property itself, measured rather than inferred from the import edge.

    The test above pins the mechanism; this pins the OUTCOME, so a future change
    to how `repository_code_inventory` resolves entry paths cannot quietly drag
    the console tier back in by another route.
    """
    import re

    from blueearth_cst.experiment.content_identity import repository_code_inventory

    root = PKG.parent
    callers = (
        "blueearth_cst/experiment/generation_plan.py",
        "blueearth_cst/experiment/indicator_plan.py",
        "blueearth_cst/experiment/simulation_record.py",
    )
    for src in callers:
        text = (root / src).read_text(encoding="utf-8")
        entries = sorted(
            {p for p in re.findall(r'"(blueearth_cst/[^"]+\.(?:py|R))"', text)} | {src}
        )
        entries = [e for e in entries if (root / e).is_file()]
        closure = {item["path"] for item in repository_code_inventory(root, entries)}
        assert not {p for p in closure if "console_style" in p}, (
            f"{src}: console_style is inside the code fingerprint again"
        )
        # Guard against the assertion above passing because the closure broke.
        assert src in closure and len(closure) > 1, f"{src}: closure looks wrong"
