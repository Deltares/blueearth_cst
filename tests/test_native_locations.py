"""Every WF4 series is keyed by the registry's wflow_id (t2609251515)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from blueearth_cst.shared.native_locations import native_location_labels

COLUMNS = [
    {"header": "Q", "map": "outlets"},
    {"header": "Q", "map": "gauges_locations"},
    {"header": "gwr", "map": "subcatchment"},
]


def _static(tmp_path, gauge_cells, registry=None):
    """Staticmaps with outlet 101 at (4, 0), subcatchments 101 | 102.

    ``registry`` is ``[(wflow_id, subbasin_id, is_primary)]`` written as the
    model's gauge layer; None writes no layer at all.
    """
    shape = (6, 6)
    outlets = np.full(shape, np.nan)
    outlets[4, 0] = 101
    gauges = np.full(shape, np.nan)
    for ident, (row, col) in gauge_cells.items():
        gauges[row, col] = ident
    subcatchment = np.full(shape, 101.0)
    subcatchment[:, 3:] = 102
    path = tmp_path / "staticmaps.nc"
    xr.Dataset(
        {
            name: (("y", "x"), values)
            for name, values in (
                ("outlets", outlets),
                ("gauges_locations", gauges),
                ("subcatchment", subcatchment),
            )
        }
    ).to_netcdf(path)
    if registry is not None:
        geoms = tmp_path / "staticgeoms"
        geoms.mkdir()
        features = [
            {
                "type": "Feature",
                "geometry": None,
                "properties": {
                    "wflow_id": wflow_id,
                    "subbasin_id": subbasin,
                    "is_primary": primary,
                },
            }
            for wflow_id, subbasin, primary in registry
        ]
        (geoms / "gauges_locations.geojson").write_text(
            json.dumps({"type": "FeatureCollection", "features": features})
        )
    return path


REGISTRY = [(1010, 101, True), (1020, 102, True), (1021, 102, False)]


def test_the_gauge_on_the_outlet_cell_wins_and_areas_take_primary_ids(tmp_path):
    static = _static(tmp_path, {1010: (4, 0), 1020: (2, 4), 1021: (1, 5)}, REGISTRY)
    headers = ["Q_101", "Q_1010", "Q_1020", "Q_1021", "gwr_101", "gwr_102"]
    assert native_location_labels(COLUMNS, headers, static) == {
        "Q_1010": "1010",
        "Q_1020": "1020",
        "Q_1021": "1021",
        "gwr_101": "1010",
        "gwr_102": "1020",
    }


def test_an_outlet_whose_primary_is_off_its_cell_yields_to_that_gauge(tmp_path):
    static = _static(tmp_path, {1010: (3, 1)}, [(1010, 101, True)])
    labels = native_location_labels(COLUMNS, ["Q_101", "Q_1010"], static)
    assert labels == {"Q_1010": "1010"}


def test_an_outlet_outside_the_registry_keeps_its_native_id(tmp_path):
    static = _static(tmp_path, {1020: (2, 4)}, [(1020, 102, True)])
    labels = native_location_labels(COLUMNS, ["Q_101", "Q_1020"], static)
    assert labels == {"Q_101": "101", "Q_1020": "1020"}


def test_the_gauge_wins_whatever_the_header_order(tmp_path):
    static = _static(tmp_path, {1010: (4, 0)}, [(1010, 101, True)])
    assert native_location_labels(COLUMNS, ["Q_101", "Q_1010"], static) == {
        "Q_1010": "1010"
    }


def test_without_a_registry_layer_ids_stay_native(tmp_path):
    static = _static(tmp_path, {1010: (4, 0), 1020: (2, 4)})
    headers = ["Q_101", "Q_1010", "Q_1020", "gwr_101", "gwr_102"]
    assert native_location_labels(COLUMNS, headers, static) == {
        "Q_1010": "1010",
        "Q_1020": "1020",
        "gwr_101": "101",
        "gwr_102": "102",
    }


def test_a_primary_absent_from_the_gauge_map_is_refused(tmp_path):
    static = _static(tmp_path, {1020: (2, 4)}, [(1030, 102, True)])
    with pytest.raises(ValueError, match="no such point"):
        native_location_labels(COLUMNS, ["Q_101", "Q_1020", "gwr_102"], static)


def test_open_responses_labels_before_numbering(tmp_path):
    """Every reader of a native CSV sees one series per cell, ordinals contiguous."""
    import pandas as pd

    from blueearth_cst.experiment.wflow_response_reader import (
        NativeRunArtifacts,
        ResponseRequest,
        open_responses,
    )

    static = _static(tmp_path, {1010: (4, 0), 1020: (2, 4)}, REGISTRY[:2])
    toml = tmp_path / "run_01.toml"
    toml.write_text(
        '[time]\ncalendar = "standard"\ntimestepsecs = 86400\n'
        'starttime = "2000-01-01T00:00:00"\nendtime = "2000-01-03T00:00:00"\n'
        f'[input]\npath_static = "{static.name}"\n'
        "[[output.csv.column]]\n"
        'header = "Q"\nmap = "outlets"\n'
        'parameter = "river_water__volume_flow_rate"\n'
        "[[output.csv.column]]\n"
        'header = "Q"\nmap = "gauges_locations"\n'
        'parameter = "river_water__volume_flow_rate"\n'
        "[[output.csv.column]]\n"
        'header = "gwr"\nmap = "subcatchment"\nreducer = "mean"\n'
        'parameter = "soil_water_saturated_zone_top__net_recharge_volume_flux"\n',
        encoding="utf-8",
    )
    csv = tmp_path / "run_01.csv"
    frame = pd.DataFrame(
        {
            "Q_101": [1.0, 2.0],
            "Q_1010": [1.0, 2.0],
            "Q_1020": [3.0, 4.0],
            "gwr_101": [0.1, 0.2],
            "gwr_102": [0.3, 0.4],
        },
        index=pd.to_datetime(["2000-01-02", "2000-01-03"]),
    )
    frame.index.name = "time"
    frame.to_csv(csv)
    series = open_responses(
        "01", NativeRunArtifacts(csv, toml), ResponseRequest(("q", "gwr"))
    )
    assert [(s.variable, s.location_id, s.location_ordinal) for s in series] == [
        ("q", "1010", 0),
        ("q", "1020", 1),
        ("gwr", "1010", 0),
        ("gwr", "1020", 1),
    ]


def test_dropped_names_the_label_each_duplicate_is_kept_under(tmp_path):
    static = _static(tmp_path, {1010: (4, 0)}, [(1010, 101, True)])
    dropped = {}
    native_location_labels(COLUMNS, ["Q_101", "Q_1010"], static, dropped)
    assert dropped == {"Q_101": "1010"}


def test_an_unresolved_area_id_is_a_warning(tmp_path, capsys):
    static = _static(tmp_path, {1010: (4, 0)}, [(1010, 101, True)])
    labels = native_location_labels(COLUMNS, ["Q_1010", "gwr_101", "gwr_102"], static)
    assert labels["gwr_102"] == "102"
    assert "gwr_102" in capsys.readouterr().out
