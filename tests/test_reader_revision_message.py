"""A frozen experiment read by a changed response reader says what to do."""

from __future__ import annotations

from blueearth_cst.experiment.response_inventory import (
    ResponseReaderUnavailable,
    _reader_changed,
)


def test_the_refusal_names_the_experiment_and_the_way_out(tmp_path):
    error = _reader_changed(
        tmp_path / "gabon", {"name": "wflow-csv", "revision": "145f2ca9b295c01b"}
    )
    assert isinstance(error, ResponseReaderUnavailable)
    message = str(error)
    assert "'gabon'" in message
    assert "145f2ca9b295" in message
    assert "new experiment_name" in message
