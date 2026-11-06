"""Independent daily climate calculations, separate from WF4 response indicators."""

from blueearth_cst.indicators.climate.events import Event, detect_events

__all__ = ["Event", "detect_events"]
