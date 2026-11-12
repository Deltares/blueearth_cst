"""Declared project layout (`blueearth_cst.shared.layout_paths`, t017 step 1)."""

import ast
import re
from pathlib import Path

import pytest

from blueearth_cst.shared import layout_paths as L

MODULE = Path(L.__file__)
DECLARED = L.declarations()

#: Names that probe each grammar's edges; owners decide which are valid.
CANDIDATES = (
    "a",
    "a1",
    "1a",
    "_a",
    "a_b",
    "a-b",
    "default",
    "showcase_stochastic_01",
    "crop-water",
    "crop_water",
    "A",
    "a" * 32,
    "a" * 33,
    "a" * 64,
    "a" * 65,
)


def test_module_imports_no_repository_code():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert "blueearth_cst" not in imported


def test_digest_length_matches_provenance():
    from blueearth_cst.shared.provenance import SHORT_DIGEST_CHARS

    assert L.SHORT_DIGEST_CHARS == SHORT_DIGEST_CHARS


def _owner_accepts(check, name: str) -> bool:
    try:
        check(name)
    except (ValueError, TypeError):
        return False
    return True


def _owners():
    from blueearth_cst.experiment.indicator_sets import (
        SET_ID_MAX_LENGTH,
        SET_ID_PATTERN,
    )
    from blueearth_cst.model.historical_runs import historical_run_name
    from blueearth_cst.shared.named_forcings import FORCING_ID
    from blueearth_cst.shared.snake_utils import (
        _EXPERIMENT_NAME_MAX_LEN,
        _EXPERIMENT_NAME_RE,
    )

    def set_id(name):
        if len(name) > SET_ID_MAX_LENGTH or not SET_ID_PATTERN.fullmatch(name):
            raise ValueError(name)

    def recipe(name):
        if not FORCING_ID.fullmatch(name):
            raise ValueError(name)

    def experiment(name):
        if len(name) > _EXPERIMENT_NAME_MAX_LEN or not _EXPERIMENT_NAME_RE.match(name):
            raise ValueError(name)

    return {
        "set_id": set_id,
        "recipe": recipe,
        "experiment_name": experiment,
        "historical_run": lambda name: historical_run_name({"name": name}),
    }


@pytest.mark.parametrize(
    "grammar", ["set_id", "recipe", "experiment_name", "historical_run"]
)
def test_grammar_accepts_every_name_its_owner_accepts(grammar):
    check = _owners()[grammar]
    pattern = re.compile(L.GRAMMARS[grammar])
    rejected = [
        name
        for name in CANDIDATES
        if _owner_accepts(check, name) and not pattern.fullmatch(name)
    ]
    assert not rejected


def test_declarations_are_unique_and_typed():
    paths = [item.path for item in DECLARED]
    assert len(paths) == len(set(paths))
    assert {item.kind for item in DECLARED} <= set(L.KINDS)
    assert {item.unit for item in DECLARED} == set(L.UNITS)


def test_unknown_fields_and_kinds_are_refused():
    with pytest.raises(ValueError, match="unknown field"):
        L.Declaration("wf1", "models/{nope}/x", "product")
    with pytest.raises(ValueError, match="unknown kind"):
        L.Declaration("wf1", "models/x", "work")


def test_bound_fields_take_caller_values():
    bound = L.bind(
        DECLARED, clim_project="cmip6", experiments=["b", "a"], model_name="wflow_x"
    )
    patterns = [pattern for _, pattern in bound]
    assert any(p.fullmatch("models/hydrology/wflow_x/staticmaps.nc") for p in patterns)
    assert not any(
        p.fullmatch("models/hydrology/wflow/staticmaps.nc") for p in patterns
    )
    assert any(p.fullmatch("experiments/a/experiment.yml") for p in patterns)
    assert any(p.fullmatch("experiments/b/experiment.yml") for p in patterns)
    assert not any(p.fullmatch("experiments/c/experiment.yml") for p in patterns)
    scoped = sum("{experiment}" in item.path for item in DECLARED)
    assert len(bound) == len(DECLARED) + scoped


def test_a_missing_binding_is_an_error():
    item = next(item for item in DECLARED if "{experiment}" in item.path)
    with pytest.raises(KeyError):
        item.pattern({"clim_project": "cmip6", "model": "wflow"})


@pytest.mark.parametrize(
    ("path", "rel", "matches"),
    [
        ("a/{x|y}.csv", "a/y.csv", True),
        ("a/{x|y}.csv", "a/xy.csv", False),
        ("a/{?b/}c", "a/c", True),
        ("a/{?b/}c", "a/b/c", True),
        ("a/open/", "a/open/x/y", True),
        ("a/open/", "a/open", False),
        ("a/f.nc", "a/fxnc", False),
    ],
)
def test_template_syntax(path, rel, matches):
    pattern = re.compile(L.Declaration("wf1", path, "product").pattern({}))
    assert bool(pattern.fullmatch(rel)) is matches
