"""Compatibility import for the qualified estimator independent module home."""

import sys

from blueearth_cst.indicators import gev_lmoments as _implementation

# Preserve one module object for existing instrumentation and monkeypatch users.
sys.modules[__name__] = _implementation
