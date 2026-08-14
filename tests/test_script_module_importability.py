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

import ast
import importlib
import pkgutil
import re
from pathlib import Path

import pytest

import blueearth_cst

PKG_ROOT = Path(blueearth_cst.__file__).resolve().parent
REPO = PKG_ROOT.parent


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


# ---------------------------------------------------------------------------
# The mirror-image rule: a `script:` target must NOT carry a __future__ import
# ---------------------------------------------------------------------------


def _script_targets():
    """Every module a Snakefile names in a `script:` directive."""
    targets = set()
    # `*.smk`, and asserted non-empty: this globbed `Snakefile_*` until the
    # 2026-08-14 rename, after which it matched nothing and this function
    # returned an empty target set -- so every `script:` module went unchecked
    # and the test still passed.
    entry_points = sorted(REPO.glob("*.smk"))
    assert entry_points, f"no workflow entry points (*.smk) under {REPO}"
    for snakefile in entry_points:
        targets |= set(
            re.findall(
                r"blueearth_cst/[a-z_0-9/]*\.py", snakefile.read_text(encoding="utf-8")
            )
        )
    return sorted(targets)


def _future_import_line(path):
    """The line number of a real `from __future__ import ...`, or None.

    Parsed rather than grepped: three modules DISCUSS the rule in a comment
    (`add_climate_forcing.py` is the one that explains it), and a text search
    reports those as violations.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            return node.lineno
    return None


def test_the_script_target_sweep_actually_finds_targets():
    """A guard on the guard, as above: an empty sweep asserts nothing."""
    targets = _script_targets()
    assert len(targets) > 20, f"expected the full script: layer, found {targets}"


@pytest.mark.parametrize("rel", _script_targets())
def test_no_script_target_carries_a_future_import(rel):
    """Snakemake's `script:` preamble displaces it, and it then raises.

    A `__future__` import must be the first statement of a module. Snakemake
    prepends its own preamble to a `script:` module before executing it, so the
    import is no longer first and the rule dies at RUN time with a SyntaxError —
    after the DAG is built, and invisible to every check that merely imports the
    module (where the file is first again and the import is legal).

    The complement of the rule is fine and widely used: 26 modules under
    `blueearth_cst/` carry the import today and none of them is a `script:`
    target. This test is what keeps that split true as modules move between the
    two roles — promoting a library module to a `script:` target is exactly the
    edit that breaks it, and nothing else would report it.
    """
    lineno = _future_import_line(REPO / rel)
    assert lineno is None, (
        f"{rel}:{lineno} is a `script:` target and carries a `__future__` import; "
        "Snakemake's preamble displaces it and the rule fails at run time. "
        "Drop the import (see blueearth_cst/model/add_climate_forcing.py)."
    )


def test_a_library_module_may_still_carry_one():
    """The rule is scoped to `script:` targets, not to the package.

    Stated as a test so a future sweep does not "fix" the 26 legal ones.
    """
    carriers = [
        p.relative_to(REPO).as_posix()
        for p in sorted(PKG_ROOT.rglob("*.py"))
        if _future_import_line(p) is not None
    ]
    assert carriers, "expected library modules to use the import"
    assert not (set(carriers) & set(_script_targets()))
