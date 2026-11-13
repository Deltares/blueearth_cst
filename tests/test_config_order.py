"""Dry runs take the execution configuration's key order (t022)."""

import yaml

from blueearth_cst.shared.config_order import sorted_mappings


def test_sorted_mappings_matches_execution_config_order():
    authored = {
        "variables": {"precip": None, "temp": None, "pet": {"method": "makkink"}},
        "horizons": [{"b": 1, "a": 2}],
        "name": "x",
    }
    archived = yaml.safe_load(yaml.safe_dump(authored, sort_keys=True))
    ordered = sorted_mappings(authored)
    assert ordered == authored
    assert repr(ordered) == repr(archived)
