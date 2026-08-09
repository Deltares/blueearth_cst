"""Tests for blueearth_cst/model/setup_gauges_and_outputs.py (R3 sections 7.1, 8)."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from blueearth_cst.model.setup_gauges_and_outputs import update_wflow_gauges_outputs  # noqa: E402


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
