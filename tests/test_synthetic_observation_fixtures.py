"""The developer generators write inputs the observation readers can consume."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from blueearth_cst.climate_analysis.rain_gauges import read_precip, read_stations
from blueearth_cst.model.historical_comparison import read_observations
from blueearth_cst.model.observation_validation import validate_observation_station_ids

SCRIPTS = Path(__file__).resolve().parents[1] / "dev/scripts"


def test_synthetic_generators_round_trip_through_observation_readers(tmp_path):
    registry = tmp_path / "location_registry.csv"
    registry.write_text(
        "wflow_id,location_role,station_name,original_x,original_y\n"
        "1010,automatic_outlet,Outlet,9.662499999827503,0.44583333326604446\n"
        "1020,control,Upstream,9.704167,0.420833\n",
        encoding="utf-8",
    )
    dates = pd.date_range("2000-01-02", "2001-01-01")
    signal = 3 + np.square(1 + np.cos(np.arange(len(dates)) / 12))
    reference = tmp_path / "output_q.csv"
    pd.DataFrame({"time": dates, "1010": signal, "1020": signal / 2}).to_csv(
        reference, index=False
    )

    def run(script, destination, *options):
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / script),
                "--locations",
                str(registry),
                "--start",
                "2000-01-02",
                "--end",
                "2001-01-01",
                "--seed",
                "42",
                "--out",
                str(destination),
                *options,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    discharge_script = "generate_synthetic_discharge.py"
    first = tmp_path / "discharge"
    result = run(discharge_script, first, "--reference", str(reference))
    assert result.returncode == 0, result.stderr
    discharge_csv = first / "synthetic-observed-discharge.csv"
    validate_observation_station_ids(discharge_csv, registry)
    discharge = read_observations(discharge_csv, {"1010", "1020"})
    assert discharge.index.equals(dates)
    assert np.isfinite(discharge.to_numpy()).all() and (discharge > 0).all().all()
    assert (discharge["1010"] / signal).std() > 0.2
    original_bytes = discharge_csv.read_bytes()
    repeat = tmp_path / "repeat"
    result = run(discharge_script, repeat, "--reference", str(reference))
    assert result.returncode == 0, result.stderr
    assert (repeat / discharge_csv.name).read_bytes() == original_bytes
    refused = run(discharge_script, first)
    assert refused.returncode == 1 and "Choose a new output directory" in refused.stderr
    assert discharge_csv.read_bytes() == original_bytes

    rain = tmp_path / "rain"
    result = run("generate_synthetic_rain_gauges.py", rain)
    assert result.returncode == 0, result.stderr
    stations = read_stations(rain / "synthetic-rain-gauge-stations.csv")
    rain_days = read_precip(rain / "synthetic-rain-gauge-precip.csv", "interval_end")
    assert list(rain_days) == list(stations.index) == ["1010", "1020"]
    assert abs(stations.at["1010", "y"] - 0.44583333326604446) < 1e-12
    assert rain_days.index.equals(dates)
    assert np.isfinite(rain_days.to_numpy()).all() and (rain_days >= 0).all().all()
    assert (rain_days.eq(0).any() & rain_days.gt(0).any()).all()
    assert not rain_days["1010"].equals(rain_days["1020"])
    provenance = json.loads(
        (rain / "synthetic-observations.json").read_text(encoding="utf-8")
    )
    assert provenance["synthetic"] and provenance["seed"] == 42
