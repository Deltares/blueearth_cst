"""config/advanced_settings.yml — the toolbox-wide constraints and defaults.

Two things worth testing: that the shipped file really is what the constants
resolve to (so editing it changes behavior, which is the whole point), and that
the schema is CLOSED (so a typo fails instead of silently leaving the built-in
value in force).
"""

import pytest
import yaml

from blueearth_cst.projections.get_change_climate_proj import DEFAULT_STATS
from blueearth_cst.shared import snake_utils as su
from blueearth_cst.spatial.config import (
    DEFAULT_GAUGE_SNAP_TOLERANCE_M,
    DEFAULT_MAX_SUBBASINS_PER_BASIN,
)


def _write(tmp_path, payload):
    path = tmp_path / "advanced_settings.yml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


VALID = {
    # `max_flagged_months` joined `constraints:` with `C-65`. This fixture
    # failing on a schema addition is the schema working: it is closed, so the
    # YAML and `_ADVANCED_SETTINGS_SCHEMA` cannot land apart without something
    # saying so.
    "constraints": {"min_historical_years": 16, "max_flagged_months": 3},
    "defaults": {
        "batch_disk_headroom_fraction": 0.25,
        "seed": 123,
        "water_year_start": "Jan",
        "hydrography": "merit_hydro_ihu",
        "basin_index": "merit_hydro_index",
        "max_subbasins": 11,
        "snap_tolerance_m": 10000.0,
        "spell_factor": [1.0] * 12,
        "change_factor_stats": ["mean", "median", "std"],
    },
    "runtime": {"julia_threads": 4, "julia_version": "1.11.7"},
    "batching": {
        "threads": "auto",
        "max_parallel": "auto",
        "small_basin_cells": 5000,
        "small_threads": 1,
        "small_max_parallel": 2,
        "large_threads": 4,
        "large_max_parallel": 1,
        "memory_per_batch_gb": 1.5,
        "memory_per_cell_kb": 20.0,
    },
}


# --- the shipped file ------------------------------------------------------


def test_the_shipped_file_is_where_the_constants_come_from():
    """Not a tautology: it reads the file from disk independently of the
    module-level load, so a constant left hardcoded would show up here."""
    on_disk = yaml.safe_load(su.ADVANCED_SETTINGS_PATH.read_text(encoding="utf-8"))
    assert su.MIN_HISTORICAL_YEARS == on_disk["constraints"]["min_historical_years"]
    assert su.DEFAULT_JULIA_THREADS == on_disk["runtime"]["julia_threads"]
    assert su.DEFAULT_SEED == on_disk["defaults"]["seed"]
    assert su.DEFAULT_WATER_YEAR_START == on_disk["defaults"]["water_year_start"]


def test_the_six_C36_defaults_come_from_the_file_too():
    """`C-36`: each of these backed a config key from a Python literal, so the
    key and its fallback lived in different tiers and a user reading the config
    could not discover what a key defaults to (`parameter-placement.md` M3).

    Three of the six are read from modules OUTSIDE `snake_utils`, which is the
    half a grep of one file would miss: a literal left behind in
    `spatial/config.py` or in the projections reducer would still satisfy every
    other test in this file.
    """
    on_disk = yaml.safe_load(su.ADVANCED_SETTINGS_PATH.read_text(encoding="utf-8"))
    defaults = on_disk["defaults"]
    assert su.DEFAULT_HYDROGRAPHY == defaults["hydrography"]
    assert su.DEFAULT_BASIN_INDEX == defaults["basin_index"]
    assert su.DEFAULT_SPELL_FACTOR == defaults["spell_factor"]
    assert DEFAULT_MAX_SUBBASINS_PER_BASIN == defaults["max_subbasins"]
    assert DEFAULT_GAUGE_SNAP_TOLERANCE_M == defaults["snap_tolerance_m"]
    assert list(DEFAULT_STATS) == defaults["change_factor_stats"]


def test_the_C36_move_changed_no_value():
    """The move is non-breaking by construction, so the shipped values are the
    ones the Python literals held. Written out rather than compared to the file:
    this is the check that would catch a transcription slip in the YAML, which
    reading the YAML back cannot."""
    assert su.DEFAULT_HYDROGRAPHY == "merit_hydro_ihu"
    assert su.DEFAULT_BASIN_INDEX == "merit_hydro_index"
    assert su.DEFAULT_SPELL_FACTOR == [1.0] * 12
    assert DEFAULT_MAX_SUBBASINS_PER_BASIN == 11
    assert DEFAULT_GAUGE_SNAP_TOLERANCE_M == 10_000.0
    assert tuple(DEFAULT_STATS) == ("mean", "median", "std")


def test_the_shipped_file_lives_under_config():
    assert su.ADVANCED_SETTINGS_PATH.is_file()
    assert su.ADVANCED_SETTINGS_PATH.parent.name == "config"


def test_the_shipped_values_are_the_documented_ones():
    """A bare-eyes check on the values other work is anchored to: 16 is
    weathergenr's wavelet minimum, 4 is P3-3's frozen baseline thread count,
    1.11.7 is what Manifest.toml was resolved against."""
    assert su.MIN_HISTORICAL_YEARS == 16
    assert su.DEFAULT_JULIA_THREADS == 4
    assert su.JULIA_VERSION == "1.11.7"
    # 123 is what dev/baseline/manifest.json was recorded with, so changing it
    # here invalidates every baseline comparison — override per project with
    # `seed:` in the generate_scenarios file instead.
    assert su.DEFAULT_SEED == 123
    # Jan is the calendar year, which is what every recorded result used; a
    # non-Jan default would move annual extremes for every existing project.
    assert su.DEFAULT_WATER_YEAR_START == "Jan"


def test_schema_and_file_cover_exactly_the_same_keys():
    """The file and the schema must be edited together; this is what catches a
    setting added to one and not the other."""
    on_disk = yaml.safe_load(su.ADVANCED_SETTINGS_PATH.read_text(encoding="utf-8"))
    assert set(on_disk) == set(su._ADVANCED_SETTINGS_SCHEMA)
    for section, keys in su._ADVANCED_SETTINGS_SCHEMA.items():
        assert set(on_disk[section]) == set(keys), section


# --- the loader's contract -------------------------------------------------


def test_a_valid_file_round_trips(tmp_path):
    assert su.load_advanced_settings(_write(tmp_path, VALID)) == VALID


def test_unknown_section_is_rejected(tmp_path):
    payload = {**VALID, "tuning": {"whatever": 1}}
    with pytest.raises(ValueError, match="unknown section"):
        su.load_advanced_settings(_write(tmp_path, payload))


def test_unknown_key_is_rejected(tmp_path):
    """The typo case the closed schema exists for: 'min_historical_year' would
    otherwise be ignored and the built-in 16 would silently stand."""
    payload = {
        "constraints": {"min_historical_years": 16, "min_historical_year": 8},
        "defaults": {"seed": 123, "water_year_start": "Jan"},
    }
    with pytest.raises(ValueError, match="unknown key"):
        su.load_advanced_settings(_write(tmp_path, payload))


def test_missing_section_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="missing section 'defaults'"):
        su.load_advanced_settings(
            _write(tmp_path, {"constraints": VALID["constraints"]})
        )


def test_missing_key_is_rejected(tmp_path):
    payload = {"constraints": {}, "defaults": VALID["defaults"]}
    with pytest.raises(ValueError, match="missing constraints.min_historical_years"):
        su.load_advanced_settings(_write(tmp_path, payload))


@pytest.mark.parametrize("bad", [0, -1, "16", 16.0, True, None])
def test_non_positive_or_non_integer_values_are_rejected(tmp_path, bad):
    payload = {**VALID, "constraints": {"min_historical_years": bad}}
    with pytest.raises(ValueError, match="constraints.min_historical_years"):
        su.load_advanced_settings(_write(tmp_path, payload))


@pytest.mark.parametrize("bad", ["1.11", "v1.11.7", "1.11.7-rc1", "", 1.11, 111, None])
def test_a_malformed_julia_version_is_rejected(tmp_path, bad):
    """`1.11` is the dangerous one twice over: as a bare YAML scalar it is a
    FLOAT, and even as a string it is a two-part selector juliaup may resolve to
    a patch the manifest was never built against."""
    payload = {**VALID, "runtime": {"julia_threads": 4, "julia_version": bad}}
    with pytest.raises(ValueError, match="runtime.julia_version"):
        su.load_advanced_settings(_write(tmp_path, payload))


def test_an_unquoted_two_part_version_reaches_the_validator_as_a_float(tmp_path):
    """The failure mode the quoting rule exists for, exercised through YAML
    rather than asserted about it."""
    path = tmp_path / "advanced_settings.yml"
    # Every section but `runtime:` comes from the fixture, so a schema addition
    # cannot turn this into a "missing key" test that still passes its `match`.
    body = {k: v for k, v in VALID.items() if k != "runtime"}
    path.write_text(
        yaml.safe_dump(body) + "runtime:\n  julia_threads: 4\n  julia_version: 1.11\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="float"):
        su.load_advanced_settings(path)


def test_a_missing_file_is_an_error_not_a_silent_fallback(tmp_path):
    """A fallback would let a deleted settings file change what the toolbox
    enforces without saying so."""
    with pytest.raises(FileNotFoundError, match="advanced settings file not found"):
        su.load_advanced_settings(tmp_path / "absent.yml")


def test_a_non_mapping_file_is_rejected(tmp_path):
    path = tmp_path / "advanced_settings.yml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a YAML mapping"):
        su.load_advanced_settings(path)


# --- the C-36 validators ---------------------------------------------------


def _with_default(key, value):
    return {**VALID, "defaults": {**VALID["defaults"], key: value}}


@pytest.mark.parametrize("bad", ["", " ", "merit hydro", "merit_hydro_ihu ", 1, None])
def test_a_malformed_catalog_entry_name_is_rejected(tmp_path, bad):
    """Only the SHAPE is checkable here — whether the name resolves is a
    property of the catalog passed with `-d`. What this catches is the slip that
    reaches hydromt as a lookup for a source nobody registered."""
    payload = _with_default("hydrography", bad)
    with pytest.raises(ValueError, match="defaults.hydrography"):
        su.load_advanced_settings(_write(tmp_path, payload))


@pytest.mark.parametrize("bad", [0, -1, "10000", True, None])
def test_a_non_positive_snap_tolerance_is_rejected(tmp_path, bad):
    payload = _with_default("snap_tolerance_m", bad)
    with pytest.raises(ValueError, match="defaults.snap_tolerance_m"):
        su.load_advanced_settings(_write(tmp_path, payload))


def test_an_integer_snap_tolerance_is_accepted_and_widened(tmp_path):
    """`10000` should not have to be written `10000.0` to be a distance."""
    resolved = su.load_advanced_settings(
        _write(tmp_path, _with_default("snap_tolerance_m", 10000))
    )
    assert resolved["defaults"]["snap_tolerance_m"] == 10000.0
    assert isinstance(resolved["defaults"]["snap_tolerance_m"], float)


@pytest.mark.parametrize(
    "bad", [[1.0] * 11, [1.0] * 13, [], "111111111111", [1.0] * 11 + ["x"], None]
)
def test_a_spell_factor_that_is_not_twelve_numbers_is_rejected(tmp_path, bad):
    """The LENGTH is the point: weathergenr indexes these by month, so R would
    recycle an eleven-element list rather than reject it, and the run would
    perturb the wrong months in silence."""
    payload = _with_default("spell_factor", bad)
    with pytest.raises(ValueError, match="defaults.spell_factor"):
        su.load_advanced_settings(_write(tmp_path, payload))


@pytest.mark.parametrize(
    "bad", [[], "mean", ["mean", "mean"], ["q 90"], ["q_00"], ["q_100"], [3], None]
)
def test_a_malformed_statistic_set_is_rejected(tmp_path, bad):
    payload = _with_default("change_factor_stats", bad)
    with pytest.raises(ValueError, match="defaults.change_factor_stats"):
        su.load_advanced_settings(_write(tmp_path, payload))


@pytest.mark.parametrize("ok", [["q_95"], ["mean", "q_5"], ["var", "median"]])
def test_the_statistic_set_is_open_not_an_enumeration_of_todays_eight(tmp_path, ok):
    """`q_95` is computed correctly by parsing the percentile out of the name,
    so a closed list of the four documented quantiles would refuse a statistic
    the code supports."""
    resolved = su.load_advanced_settings(
        _write(tmp_path, _with_default("change_factor_stats", ok))
    )
    assert resolved["defaults"]["change_factor_stats"] == ok


@pytest.mark.parametrize("value", ["auto", 2])
def test_batching_threads_take_auto_or_a_count(tmp_path, value):
    path = tmp_path / "advanced_settings.yml"
    payload = {**VALID, "batching": {**VALID["batching"], "threads": value}}
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    assert su.load_advanced_settings(path)["batching"]["threads"] == value


@pytest.mark.parametrize("value", ["fast", 0])
def test_batching_threads_refuse_anything_else(tmp_path, value):
    path = tmp_path / "advanced_settings.yml"
    payload = {**VALID, "batching": {**VALID["batching"], "threads": value}}
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="batching.threads"):
        su.load_advanced_settings(path)
