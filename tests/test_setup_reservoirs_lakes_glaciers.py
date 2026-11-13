"""Tests for blueearth_cst/model/setup_reservoirs_lakes_glaciers.py (R3 sections 7b, 8).

Method dispatch, no-data capture and the structured sentinel run without
hydromt (the heavy import is lazy inside the main function); the missing-source
check uses a real hydromt catalog over temporary files.
"""

import pytest
import yaml

from blueearth_cst.model.setup_reservoirs_lakes_glaciers import (  # noqa: E402
    _missing_waterbody_sources,
    _run_waterbody_methods,
    write_sentinel,
    write_values_used,
)


class _FakeNoData(Exception):
    pass


class _FakeModel:
    """Stand-in wflow model: one method succeeds, one reports no data."""

    def setup_reservoirs_simple_control(self, **kwargs):
        pass

    def setup_glaciers(self, **kwargs):
        raise _FakeNoData("no glaciers in basin")


def test_run_waterbody_methods_captures_ok_and_skipped():
    config = {
        "setup_reservoirs_simple_control": {"min_area": 1.0},
        "setup_glaciers": None,
    }
    results = _run_waterbody_methods(_FakeModel(), config, (_FakeNoData,))
    assert results == [
        {
            "method": "setup_reservoirs_simple_control",
            "status": "ok",
            "reason": "",
            "values_used": {"min_area": 1.0},
        },
        {
            "method": "setup_glaciers",
            "status": "skipped",
            "reason": "no glaciers in basin",
            "values_used": {},
        },
    ]


def test_values_used_record_separates_a_method_that_ran_from_one_skipped(tmp_path):
    """A skipped method leaves no trace in the model.

    Recording values without the outcome would describe reservoirs and lakes
    that were never added -- the record would be confidently wrong about what
    the model contains. Skipping is legitimate here (the basin has no such
    water bodies in the source), which is precisely why it must be stated
    rather than inferred from an absence.
    """
    config = {
        "setup_reservoirs_simple_control": {"min_area": 1.0},
        "setup_glaciers": None,
    }
    results = _run_waterbody_methods(_FakeModel(), config, (_FakeNoData,))
    path = tmp_path / "hydromt_update_waterbodies.yml"

    write_values_used(path, results, "config/defaults/wflow_update_waterbodies.yml")

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    by_method = {step["method"]: step for step in document["steps"]}

    assert document["waterbodies_config"].endswith("wflow_update_waterbodies.yml")
    assert by_method["setup_reservoirs_simple_control"]["status"] == "ok"
    assert by_method["setup_reservoirs_simple_control"]["values_used"] == {
        "min_area": 1.0
    }
    assert by_method["setup_glaciers"]["status"] == "skipped"
    assert by_method["setup_glaciers"]["reason"] == "no glaciers in basin"


def test_missing_source_is_reported_but_an_empty_one_is_not(tmp_path):
    """A missing file and a basin without features look identical downstream.

    hydromt_wflow reads waterbody sources with ``NoDataStrategy.IGNORE`` and
    logs one "no data" line for both, so the distinction has to be drawn
    before the methods run: a present-but-empty source passes, an absent file
    is named. Keywords (``timeseries_fn: jrc``) and non-catalog values are left
    to hydromt.
    """
    pytest.importorskip("pyogrio")
    gpd = pytest.importorskip("geopandas")
    from hydromt import DataCatalog
    from hydromt.error import NoDataException, NoDataStrategy
    from shapely.geometry import Point

    # Present but empty for any basin: holds no features at all.
    gpd.GeoDataFrame({"a": [1]}, geometry=[Point(0, 0)], crs=4326).iloc[:0].to_file(
        tmp_path / "empty.gpkg"
    )
    (tmp_path / "catalog.yml").write_text(
        "empty: {data_type: GeoDataFrame, driver: {name: pyogrio}, uri: empty.gpkg}\n"
        "absent: {data_type: GeoDataFrame, driver: {name: pyogrio}, uri: absent.gpkg}\n",
        encoding="utf-8",
    )
    config = {
        "setup_reservoirs_simple_control": {
            "reservoirs_fn": "empty",
            "timeseries_fn": "jrc",
        },
        "setup_reservoirs_no_control": None,
        "setup_glaciers": {"glaciers_fn": "absent", "min_area": 1.0},
    }

    missing = _missing_waterbody_sources(
        DataCatalog(str(tmp_path / "catalog.yml")),
        config,
        NoDataStrategy.RAISE,
        (NoDataException,),
    )

    assert len(missing) == 1
    assert missing[0].startswith("setup_glaciers: glaciers_fn='absent'")
    assert missing[0].endswith("absent.gpkg")


def test_write_sentinel_is_structured_tsv(tmp_path):
    path = tmp_path / "reservoirs_lakes_glaciers.txt"
    write_sentinel(
        path,
        [
            {"method": "setup_reservoirs_simple_control", "status": "ok", "reason": ""},
            {"method": "setup_glaciers", "status": "skipped", "reason": "no\tglaciers"},
        ],
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "method\tstatus\treason"
    assert lines[1] == "setup_reservoirs_simple_control\tok\t"
    # tabs/newlines in the reason are neutralized so each row stays one line
    assert lines[2] == "setup_glaciers\tskipped\tno glaciers"
    assert len(lines) == 3
