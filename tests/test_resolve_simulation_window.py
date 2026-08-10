"""`workflows.model_creation.simulation_window` — the model RUN period.

Split from `shared.historical_window` on 2026-08-10. The record a project
extracts for analysis and the period it simulates are different questions, and
nothing couples them: rule 1.10 declares no climate-store input, so the forcing
comes from the data catalog rather than from the extraction.

The load-bearing property here is the passthrough. Every config written before
this key existed carries only `historical_window`, so "absent" has to mean
exactly that window — not a default that happens to look like it.
"""

import pytest

from blueearth_cst.shared.snake_utils import resolve_simulation_window


def _window(start, end):
    return {"starttime": f"{start}T00:00:00", "endtime": f"{end}T00:00:00"}


def _shared(start="2000-01-01", end="2020-12-31"):
    return {"historical_window": _window(start, end)}


# --- the default path: what every pre-existing config takes --------------------


def test_absent_key_passes_the_historical_window_through():
    shared = _shared()
    assert resolve_simulation_window(shared, {}) == shared["historical_window"]


def test_passthrough_is_the_same_object_not_a_reconstruction():
    """A rebuilt-but-equal mapping would drift the moment either side gains a
    field. Identity makes 'unchanged behaviour' structural rather than asserted."""
    shared = _shared()
    assert resolve_simulation_window(shared, {}) is shared["historical_window"]


def test_an_explicit_null_is_still_passthrough():
    """A commented-out key that gets uncommented empty is a plausible edit, and
    it should behave as absent rather than crash."""
    shared = _shared()
    got = resolve_simulation_window(shared, {"simulation_window": None})
    assert got == shared["historical_window"]


# --- the override path ---------------------------------------------------------


def test_an_explicit_window_wins_over_the_historical_one():
    sim = _window("2010-01-01", "2015-12-31")
    got = resolve_simulation_window(_shared(), {"simulation_window": sim})
    assert got == sim


def test_the_simulation_window_need_not_sit_inside_the_record():
    """Deliberately unconstrained. The forcing is resolved from the DATA
    CATALOG (setup_precip_forcing), not from the extracted store, so a project
    may analyse one period and simulate another with no overlap at all. A
    subset check here would reject the separation this key exists to express.
    """
    sim = _window("2021-01-01", "2023-12-31")  # entirely after the record
    got = resolve_simulation_window(
        _shared("2000-01-01", "2020-12-31"), {"simulation_window": sim}
    )
    assert got == sim


def test_a_short_simulation_window_is_allowed():
    """MIN_HISTORICAL_YEARS is weathergenr's floor on the RECORD. Applying it
    here would forbid the short run the split exists to make possible."""
    sim = _window("2000-01-01", "2000-06-30")
    got = resolve_simulation_window(_shared(), {"simulation_window": sim})
    assert got == sim


# --- malformed input fails loudly, naming the key ------------------------------


@pytest.mark.parametrize("bad", [[], "2000-01-01", 2000])
def test_a_non_mapping_is_rejected(bad):
    with pytest.raises(ValueError, match="simulation_window must be a mapping"):
        resolve_simulation_window(_shared(), {"simulation_window": bad})


@pytest.mark.parametrize("missing", ["starttime", "endtime"])
def test_a_missing_endpoint_is_rejected(missing):
    sim = _window("2000-01-01", "2010-12-31")
    del sim[missing]
    with pytest.raises(ValueError, match=f"missing '{missing}'"):
        resolve_simulation_window(_shared(), {"simulation_window": sim})


def test_a_non_iso_endpoint_names_the_offending_key():
    sim = {"starttime": "01/01/2000", "endtime": "2010-12-31T00:00:00"}
    with pytest.raises(ValueError) as excinfo:
        resolve_simulation_window(_shared(), {"simulation_window": sim})
    message = str(excinfo.value)
    assert "simulation_window.starttime" in message and "ISO" in message


def test_a_reversed_window_says_so_rather_than_failing_later():
    """Unlike the record, this has no length floor to trip, so a swapped pair
    would otherwise reach wflow as an empty run period."""
    sim = _window("2015-01-01", "2010-01-01")
    with pytest.raises(ValueError, match="check the order"):
        resolve_simulation_window(_shared(), {"simulation_window": sim})


def test_an_equal_pair_is_rejected_too():
    sim = _window("2010-01-01", "2010-01-01")
    with pytest.raises(ValueError, match="check the order"):
        resolve_simulation_window(_shared(), {"simulation_window": sim})
