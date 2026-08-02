"""Tests for observation-to-registry identity validation."""

from __future__ import annotations

import pandas as pd
import pytest

from blueearth_cst.model.observation_validation import (
    validate_observation_station_ids,
)


def _registry(tmp_path):
    path = tmp_path / "location_registry.csv"
    pd.DataFrame(
        {
            "wflow_id": [101, 102, 1_000_001, 103],
            "location_role": [
                "control",
                "observation",
                "observation",
                "automatic_outlet",
            ],
        }
    ).to_csv(path, index=False)
    return path


def test_valid_header_requires_user_locations_but_not_automatic_outlets(tmp_path):
    observations = tmp_path / "observations.csv"
    observations.write_text(
        "time;101;102;1000001\n2000-01-01T00:00:00;1;2;3\n",
        encoding="utf-8",
    )

    registry = validate_observation_station_ids(observations, _registry(tmp_path))

    assert registry["wflow_id"].tolist() == [101, 102, 1_000_001, 103]


def test_header_reports_missing_duplicate_and_unexpected_ids_together(tmp_path):
    observations = tmp_path / "observations.csv"
    observations.write_text(
        "time;101;101;999\n2000-01-01T00:00:00;1;2;3\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        validate_observation_station_ids(observations, _registry(tmp_path))

    message = str(exc_info.value)
    assert "missing=[102, 1000001]" in message
    assert "duplicate=[101]" in message
    assert "unexpected=[999]" in message


def test_comma_separated_header_fails_with_actionable_message(tmp_path):
    observations = tmp_path / "observations.csv"
    observations.write_text("time,101,102\n2000-01-01,1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="semicolon-separated"):
        validate_observation_station_ids(observations, _registry(tmp_path))
