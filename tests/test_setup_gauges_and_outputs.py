"""Tests for blueearth_cst/model/setup_gauges_and_outputs.py (R3 sections 7.1, 8)."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from blueearth_cst.model.setup_gauges_and_outputs import (
    WFLOW_VARS as _ALL_VARS,
)
from blueearth_cst.model.setup_gauges_and_outputs import (
    update_wflow_gauges_outputs,  # noqa: E402
)


def test_raises_on_unknown_outvar():
    # Validation runs before any model is opened, so a dummy root is fine: an
    # unknown wflow_outvars name must raise loudly, not be silently dropped.
    with pytest.raises(ValueError, match="Unknown wflow_outvars"):
        update_wflow_gauges_outputs(wflow_root="unused", outputs=["bogus var"])


def test_extras_selection_and_csdms_mapping(monkeypatch):
    """Extras exclude river discharge and map to CSDMS names in order.

    Mocks WflowSbmModel (the lazy hydromt_wflow import) with a recorder so the
    call arguments can be inspected without a real model.
    """
    import types

    from blueearth_cst.model.setup_gauges_and_outputs import WFLOW_VARS

    calls = {}

    class _FakeMod:
        def __init__(self, *a, **k):
            self.staticmaps = SimpleNamespace(
                data=xr.Dataset({"outlets": (("y", "x"), np.array([[101.0]]))})
            )

        def setup_config_output_timeseries(self, **k):
            calls.setdefault("timeseries", []).append(k)

        def write(self):
            calls["write"] = True

        def close(self):
            calls["close"] = True

    fake_hw = types.ModuleType("hydromt_wflow")
    fake_hw.WflowSbmModel = _FakeMod
    monkeypatch.setitem(sys.modules, "hydromt_wflow", fake_hw)

    update_wflow_gauges_outputs(
        wflow_root="x", outputs=["river discharge", "snow", "overland flow"]
    )

    # river discharge uses the inherited outlets map, without recreating it.
    assert calls["timeseries"][0]["mapname"] == "outlets"
    assert calls["timeseries"][0]["param"] == [WFLOW_VARS["river discharge"]]
    # extras drop river discharge; header + param stay in order and are mapped
    assert calls["timeseries"][1]["header"] == [
        "snow_basavg",
        "overland flow_basavg",
    ]
    assert calls["timeseries"][1]["param"] == [
        WFLOW_VARS["snow"],
        WFLOW_VARS["overland flow"],
    ]
    assert calls.get("write") and calls.get("close")


def _record_calls(monkeypatch, with_registry=False):
    """Drive update_wflow_gauges_outputs against a recorder, return its calls."""
    import types

    calls = {}
    maps = {"outlets": (("y", "x"), np.array([[101.0]]))}
    if with_registry:
        maps["gauges_locations"] = (("y", "x"), np.array([[1010.0]]))

    class _FakeMod:
        def __init__(self, *a, **k):
            self.staticmaps = SimpleNamespace(data=xr.Dataset(maps))

        def setup_config_output_timeseries(self, **k):
            calls.setdefault("timeseries", []).append(k)

        def write(self):
            calls["write"] = True

        def close(self):
            calls["close"] = True

    fake_hw = types.ModuleType("hydromt_wflow")
    fake_hw.WflowSbmModel = _FakeMod
    monkeypatch.setitem(sys.modules, "hydromt_wflow", fake_hw)
    return calls


@pytest.mark.parametrize("var", sorted(set(_ALL_VARS) - {"river discharge"}))
def test_every_configurable_outvar_reaches_a_declaration(monkeypatch, var):
    """Every WFLOW_VARS name a config may request must actually be emitted.

    Parametrized over the whole table rather than a sample: the previous test
    exercised two of the six, so a variable that mapped to nothing would have
    gone unnoticed. `river discharge` is excluded here only because it takes the
    outlets/gauges path instead of the basin-average one, and has its own test.
    """
    from blueearth_cst.model.setup_gauges_and_outputs import WFLOW_VARS

    calls = _record_calls(monkeypatch)
    update_wflow_gauges_outputs(wflow_root="x", outputs=["river discharge", var])

    basavg = [c for c in calls["timeseries"] if c["mapname"] == "subcatchment"]
    assert len(basavg) == 1, f"{var} produced no basin-average declaration"
    assert basavg[0]["header"] == [f"{var}_basavg"]
    assert basavg[0]["param"] == [WFLOW_VARS[var]]
    assert basavg[0]["reducer"] == ["mean"]


def test_discharge_is_emitted_even_when_the_config_omits_it(monkeypatch):
    """Discharge is the floor: it ships whatever `wflow_outvars` says.

    The outlets declaration is unconditional, so a config listing only `snow`
    still gets Q. Without this, a user could silently produce a run with no
    discharge at all -- the one variable every downstream indicator needs.
    """
    from blueearth_cst.model.setup_gauges_and_outputs import WFLOW_VARS

    calls = _record_calls(monkeypatch)
    update_wflow_gauges_outputs(wflow_root="x", outputs=["snow"])

    outlets = [c for c in calls["timeseries"] if c["mapname"] == "outlets"]
    assert len(outlets) == 1
    assert outlets[0]["header"] == ["Q"]
    assert outlets[0]["param"] == [WFLOW_VARS["river discharge"]]


def test_the_gauge_block_emits_discharge_only(monkeypatch):
    """Regression guard: no unrequested variable rides along at the gauges.

    A `P` column was appended here unconditionally until 2026-08-10, ignoring
    `wflow_outvars` entirely. It was also a POINT sample (this map passes no
    reducer, so wflow's default `only` applies), i.e. the forcing handed back
    rather than anything derived. Basin-average precipitation remains available
    the designed way, via `precipitation` in wflow_outvars -> subcatchment mean.
    """
    from blueearth_cst.model.setup_gauges_and_outputs import WFLOW_VARS

    calls = _record_calls(monkeypatch, with_registry=True)
    update_wflow_gauges_outputs(
        wflow_root="x", outputs=["river discharge"], location_registry=None
    )

    for call in calls["timeseries"]:
        assert call["header"] == ["Q"], (
            f"{call['mapname']} emits {call['header']}; a config asking only for "
            "river discharge must produce discharge and nothing else"
        )
        assert call["param"] == [WFLOW_VARS["river discharge"]]
