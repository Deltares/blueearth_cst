"""Unit tests for dev/scripts/prune_climate_store.py (R09 phase 1).

The historical climate store is keyed `<clim_source>_<window>` and that key is
retained by R09 (design Finding 3), so changing the source or the window strands
its predecessor with nothing to report it. This script is that report, and its
one hard contract is that it deletes NOTHING without `--delete`.
"""

import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dev", "scripts"))
import prune_climate_store as pcs  # noqa: E402

ACTIVE = "era5_20000101_20201231"


def _config(project_dir, source="era5", start="2000-01-01T00:00:00",
            end="2020-12-31T00:00:00"):
    return {
        "project": {"project_dir": str(project_dir).replace("\\", "/")},
        "shared": {
            "clim_historical": source,
            "historical_window": {"starttime": start, "endtime": end},
        },
    }


def _store(project_dir, key, payload=b"x" * 32):
    d = project_dir / pcs.STORE_ROOT / key
    d.mkdir(parents=True, exist_ok=True)
    (d / "extract_historical.nc").write_bytes(payload)
    return d


def _write_config(tmp_path, cfg):
    p = tmp_path / "cfg.yml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def test_active_store_key_matches_the_workflow_key():
    """Derived the same way `snake_utils.climate_store_rule` derives it."""
    assert pcs.active_store_key(_config("proj")) == ACTIVE
    assert pcs.active_store_key(
        _config("proj", source="chirps", start="1990-01-01T00:00:00",
                end="2010-01-01T00:00:00")
    ) == "chirps_19900101_20100101"


def test_a_changed_window_strands_its_predecessor(tmp_path):
    """The failure mode the script exists for (design Finding 3)."""
    proj = tmp_path / "proj"
    _store(proj, ACTIVE)
    _store(proj, "era5_19900101_20101231")   # the previous window
    orphans = pcs.find_orphans(proj, ACTIVE)
    assert [p.name for p in orphans] == ["era5_19900101_20101231"]


def test_a_changed_source_strands_its_predecessor(tmp_path):
    proj = tmp_path / "proj"
    _store(proj, ACTIVE)
    _store(proj, "chirps_20000101_20201231")
    assert [p.name for p in pcs.find_orphans(proj, ACTIVE)] == \
        ["chirps_20000101_20201231"]


def test_the_active_store_is_never_an_orphan(tmp_path):
    proj = tmp_path / "proj"
    _store(proj, ACTIVE)
    assert pcs.find_orphans(proj, ACTIVE) == []


def test_missing_store_root_is_not_an_error(tmp_path, capsys):
    cfg = _write_config(tmp_path, _config(tmp_path / "proj"))
    assert pcs.main(["--config", str(cfg)]) == 0
    assert "does not exist" in capsys.readouterr().out


def test_dry_run_is_the_default_and_deletes_nothing(tmp_path, capsys):
    """The hard contract: reporting never destroys."""
    proj = tmp_path / "proj"
    _store(proj, ACTIVE)
    stale = _store(proj, "era5_19900101_20101231")
    cfg = _write_config(tmp_path, _config(proj))

    assert pcs.main(["--config", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "ORPHAN   era5_19900101_20101231/" in out
    assert "DRY RUN" in out
    assert stale.is_dir(), "dry run must not delete"
    assert (stale / "extract_historical.nc").exists()


def test_delete_flag_removes_only_the_orphans(tmp_path, capsys):
    proj = tmp_path / "proj"
    active = _store(proj, ACTIVE)
    stale = _store(proj, "era5_19900101_20101231")
    cfg = _write_config(tmp_path, _config(proj))

    assert pcs.main(["--config", str(cfg), "--delete"]) == 0
    assert not stale.exists()
    assert (active / "extract_historical.nc").exists()
    assert "deleted 1 store(s)" in capsys.readouterr().out


def test_an_absent_active_store_is_reported_not_treated_as_an_orphan(
    tmp_path, capsys
):
    """A project whose window just changed has no active store yet; the stale
    one is still the orphan, and the active key is merely noted as pending."""
    proj = tmp_path / "proj"
    stale = _store(proj, "era5_19900101_20101231")
    cfg = _write_config(tmp_path, _config(proj))
    assert pcs.main(["--config", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "the active store is absent" in out
    assert stale.is_dir()


def test_a_sub_day_window_fails_loud(tmp_path):
    """The store key is day-resolution; a sub-day window would collide silently."""
    with pytest.raises(ValueError, match="time-of-day"):
        pcs.active_store_key(_config("proj", start="2000-01-01T06:00:00"))
