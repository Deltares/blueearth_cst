"""Rules 1.08 and 1.09 must not flush the forcing.

The defect this pins, measured on a real re-run: ``WflowBaseModel.write()``
flushes every component, and the forcing component's own empty-test reads
``self.data``, which lazily loads the file in ``r+`` mode. So on any WF1 re-run
where the forcing netCDF already existed, hydromt read it, wrote it back under
an invented name because the target existed with overwriting off, and repointed
``input.path_forcing`` at the duplicate. See ``blueearth_cst/shared/wflow_write``.

Two kinds of test here, and the second is the one that matters over time. The
fakes pin what our replacement writes. The drift guard reads hydromt's own
source and asserts our sequence is exactly its sequence minus the forcing, so a
plugin upgrade that adds or reorders a component write fails loudly instead of
silently dropping an artifact from every WF1 build.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import textwrap

import pytest

from blueearth_cst.shared.wflow_write import (
    FORCING_COMPONENT,
    GEOMS_FOLDER,
    write_model_except_forcing,
)


class _Recorder:
    """Records the component writes a model receives, in order."""

    def __init__(self, calls, name):
        self._calls = calls
        self._name = name

    def write(self, *args, **kwargs):
        self._calls.append(f"{self._name}.write")

    def write_region(self, *args, **kwargs):
        self._calls.append(f"{self._name}.write_region")

    @property
    def data(self):
        self._calls.append(f"{self._name}.data")
        return {}


class _FakeRoot:
    path = "/some/model"


class _FakeModel:
    def __init__(self):
        self.calls = []
        self.root = _FakeRoot()
        for name in ("config", "forcing", "geoms", "states", "staticmaps", "tables"):
            setattr(self, name, _Recorder(self.calls, name))

    def write_data_catalog(self):
        self.calls.append("write_data_catalog")


def test_the_forcing_is_never_written():
    model = _FakeModel()
    write_model_except_forcing(model)
    assert not any(c.startswith("forcing.") for c in model.calls), model.calls


def test_every_other_component_is_written():
    model = _FakeModel()
    write_model_except_forcing(model)
    for component in ("staticmaps", "geoms", "tables", "states", "config"):
        assert f"{component}.write" in model.calls, (component, model.calls)


def test_the_config_is_written_last():
    """hydromt's own reason: other write methods can still set config values."""
    model = _FakeModel()
    write_model_except_forcing(model)
    assert model.calls[-1] == "config.write", model.calls


def test_the_data_catalog_is_written_first():
    model = _FakeModel()
    write_model_except_forcing(model)
    assert model.calls[0] == "write_data_catalog", model.calls


# --- drift guard ------------------------------------------------------------


def _component_writes(source: str, receiver: str) -> list[tuple[str, str]]:
    """Every ``<receiver>.<component>.write*(...)`` call in a body, in order.

    Parsed from the AST, not matched in the text, so a docstring or a comment
    naming the call it deliberately omits is not counted as making it.
    """
    tree = ast.parse(textwrap.dedent(source))
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or not func.attr.startswith("write"):
            continue
        owner = func.value
        if (
            isinstance(owner, ast.Attribute)
            and isinstance(owner.value, ast.Name)
            and owner.value.id == receiver
        ):
            calls.append((owner.attr, func.attr))
    return calls


def test_our_sequence_is_hydromts_sequence_minus_the_forcing():
    """The cost of replicating a vendored sequence, paid here.

    Reads the plugin's own ``write`` and our replacement and compares the
    component writes each performs. A hydromt upgrade that adds, drops or
    reorders one fails HERE, where the alternative is a WF1 build that quietly
    stops writing an artifact and still exits 0.
    """
    wflow_base = pytest.importorskip("hydromt_wflow.wflow_base")

    theirs = _component_writes(
        inspect.getsource(wflow_base.WflowBaseModel.write), receiver="self"
    )
    ours = _component_writes(
        pathlib.Path("blueearth_cst/shared/wflow_write.py").read_text(encoding="utf-8"),
        receiver="model",
    )

    assert theirs, "could not parse hydromt's write sequence; the guard is blind"
    expected = [call for call in theirs if call[0] != FORCING_COMPONENT]
    assert ours == expected, (
        "our write sequence has drifted from hydromt's.\n"
        f"  hydromt (minus forcing): {expected}\n"
        f"  ours:                    {ours}"
    )
    # And the one we drop really is in theirs, or this test proves nothing.
    assert any(call[0] == FORCING_COMPONENT for call in theirs)


def test_the_geoms_folder_matches_hydromts_default():
    wflow_base = pytest.importorskip("hydromt_wflow.wflow_base")
    signature = inspect.signature(wflow_base.WflowBaseModel.write)
    assert signature.parameters["geoms_folder"].default == GEOMS_FOLDER


# --- the call sites ---------------------------------------------------------


@pytest.mark.parametrize(
    "module", ["setup_reservoirs_lakes_glaciers", "setup_gauges_and_outputs"]
)
def test_neither_rule_calls_a_bare_write(module):
    """A blunt guard against the defect's return.

    A reinstated ``mod.write()`` would reintroduce the duplicate silently: a
    successful run is exactly what the defect looks like.
    """
    source = pathlib.Path("blueearth_cst/model", f"{module}.py").read_text(
        encoding="utf-8"
    )
    assert "mod.write()" not in source, f"{module} flushes every component again"
    assert "write_model_except_forcing(mod)" in source
