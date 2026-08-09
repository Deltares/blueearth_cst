"""[R7-22] Every module under `blueearth_cst/` must import without Snakemake.

Snakemake's `script:` directive injects a `snakemake` object into the module's
globals before executing it. A module that reads that object at TOP LEVEL --
`config = snakemake.input.x` outside any function -- therefore works under
Snakemake and raises `NameError` under `import`, which makes it invisible to
unit tests. The repo's answer is the guarded idiom:

    if __name__ == "__main__":
        if "snakemake" in globals():
            sm = globals()["snakemake"]
            ...

`downscale_climate_forcing.py` was the last holdout and carried an `F821`
per-file-ignore in `pyproject.toml` to keep the lint gate green. Converting it
let that entry be deleted; this sweep is what stops a new one being needed.

Ruff's F821 catches the same class from the other side, statically. This test is
the dynamic half: it fails on a module that is unimportable for any reason --
an import-time side effect that needs a run directory, a top-level read of a
config file -- not only on an undefined name.
"""

import importlib
import pkgutil
from pathlib import Path

import pytest

import blueearth_cst

PKG_ROOT = Path(blueearth_cst.__file__).resolve().parent


def _module_names():
    """Every importable module under `blueearth_cst/`, dotted."""
    return sorted(
        name
        for _, name, ispkg in pkgutil.walk_packages(
            blueearth_cst.__path__, prefix="blueearth_cst."
        )
        if not ispkg
    )


def test_the_sweep_actually_finds_modules():
    """A guard on the guard: an empty sweep would pass every assertion below."""
    names = _module_names()
    assert len(names) > 20, f"expected the full script: layer, found {names}"


@pytest.mark.parametrize("name", _module_names())
def test_module_imports_without_a_snakemake_global(name):
    """Importing must not require the injected `snakemake` object."""
    assert "snakemake" not in globals(), "the test process must not carry one"
    importlib.import_module(name)


def test_no_module_reads_snakemake_at_module_scope():
    """The static half: `snakemake.` must not appear outside a function body.

    Catches the shape before it reaches an import error -- and catches it in a
    module whose top-level read happens to be inside a `try:` that swallows the
    NameError, which the dynamic sweep above would let through.
    """
    import ast

    offenders = []
    for path in sorted(PKG_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        # Walk only the module's own top-level statements, descending into the
        # `if __name__ == "__main__":` / `if "snakemake" in globals():` guards
        # is exactly what we do NOT want -- reads there are the correct idiom.
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, ast.If):
                continue  # a guard; the idiom lives here
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and sub.id == "snakemake":
                    offenders.append(f"{path.relative_to(PKG_ROOT)}:{sub.lineno}")
    assert not offenders, (
        "these read the injected `snakemake` object at module scope; move the "
        f"read inside the `if __name__ == '__main__':` guard: {offenders}"
    )
