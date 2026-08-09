"""config/advanced_settings.yml — the toolbox-wide constraints and defaults.

Two things worth testing: that the shipped file really is what the constants
resolve to (so editing it changes behavior, which is the whole point), and that
the schema is CLOSED (so a typo fails instead of silently leaving the built-in
value in force).
"""

import pytest
import yaml

from blueearth_cst.shared import snake_utils as su


def _write(tmp_path, payload):
    path = tmp_path / "advanced_settings.yml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


VALID = {
    "constraints": {"min_historical_years": 16},
    "defaults": {"julia_threads": 4},
    "runtime": {"julia_version": "1.11.7"},
}


# --- the shipped file ------------------------------------------------------


def test_the_shipped_file_is_where_the_constants_come_from():
    """Not a tautology: it reads the file from disk independently of the
    module-level load, so a constant left hardcoded would show up here."""
    on_disk = yaml.safe_load(su.ADVANCED_SETTINGS_PATH.read_text(encoding="utf-8"))
    assert su.MIN_HISTORICAL_YEARS == on_disk["constraints"]["min_historical_years"]
    assert su.DEFAULT_JULIA_THREADS == on_disk["defaults"]["julia_threads"]


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
        "defaults": {"julia_threads": 4},
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
    payload = {**VALID, "runtime": {"julia_version": bad}}
    with pytest.raises(ValueError, match="runtime.julia_version"):
        su.load_advanced_settings(_write(tmp_path, payload))


def test_an_unquoted_two_part_version_reaches_the_validator_as_a_float(tmp_path):
    """The failure mode the quoting rule exists for, exercised through YAML
    rather than asserted about it."""
    path = tmp_path / "advanced_settings.yml"
    path.write_text(
        "constraints:\n  min_historical_years: 16\n"
        "defaults:\n  julia_threads: 4\n"
        "runtime:\n  julia_version: 1.11\n",
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
