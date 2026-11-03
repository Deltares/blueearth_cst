"""Numerical preparation survives display changes and rejects stale/corrupt fields."""

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import hydromt  # noqa: F401 -- registers the raster accessor
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from blueearth_cst.climate_analysis.source_map_fields import SourceMapSpec
from blueearth_cst.shared import map_preparation as mp
from blueearth_cst.shared import map_reference as mr


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    from blueearth_cst.climate_analysis.prepare_source_map_fields import run

    monkeypatch.setattr(mp, "environment_identity", lambda: {"fixture": "qualified"})
    monkeypatch.setattr(mr, "environment_identity", lambda: {"fixture": "qualified"})
    climate = tmp_path / "chirps/extract_historical.nc"
    climate.parent.mkdir()
    raw = xr.Dataset(
        {"precip": (("time", "latitude", "longitude"), np.full((365, 2, 2), 2.0))},
        coords={
            "time": pd.date_range("2001-01-01", "2001-12-31"),
            "latitude": [1.5, 1.0],
            "longitude": [2.0, 2.5],
        },
    )
    raw.precip.attrs["units"] = "mm/day"
    raw.raster.set_crs(4326)
    raw.to_netcdf(climate)
    cells = climate.parent / "basin_cells.csv"
    cells.write_text(
        "latitude,longitude\n1.5,2.0\n1.5,2.5\n1.0,2.0\n1.0,2.5\n", encoding="utf-8"
    )
    spec = SourceMapSpec("chirps", climate, cells, (), ("2001-01-01", "2001-12-31"))
    policy = mp.resolve_map_completeness()
    run(
        SimpleNamespace(
            params=SimpleNamespace(
                source_spec=mp.source_spec_dict(spec), m0=1, completeness=policy
            ),
            output=SimpleNamespace(fields=str(mp.source_fields_path(spec))),
        )
    )
    return spec, policy


@pytest.mark.parametrize(
    ("changed", "scientific"),
    [
        ("shared/map_reference.py", False),
        ("plotting/prepared_maps.py", False),
        ("plotting/export.py", False),
        ("climate_analysis/climate_figures.py", False),
        ("climate_analysis/diagnostic_maps.py", True),
        ("climate_analysis/source_capabilities.py", True),
        ("climate_analysis/plot_climate_source.py", True),
        ("shared/climate_parity.py", True),
        ("shared/yaml_documents.py", True),
    ],
)
def test_source_edits_preserve_only_compatible_preparations(
    prepared, monkeypatch, changed, scientific
):
    spec, policy = prepared
    before = mp.preparation_identity(spec, 1, policy)
    path = mp.source_fields_path(spec)
    saved = path.read_bytes()
    original = mp.file_digest
    changed_path = Path(__file__).resolve().parents[1] / "blueearth_cst" / changed

    def edited_digest(source):
        source = Path(source)
        if source.resolve() == changed_path:
            return hashlib.sha256(source.read_bytes() + b"\n# edit\n").hexdigest()
        return original(source)

    monkeypatch.setattr(mp, "file_digest", edited_digest)
    after = mp.preparation_identity(spec, 1, policy)
    if scientific:
        assert after != before
        with pytest.raises(ValueError, match="stale preparation"):
            mp.read_source_fields(spec, 1, policy)
    else:
        assert after == before
        fields, _ = mp.read_source_fields(spec, 1, policy)
        np.testing.assert_array_equal(fields.precip.values, np.full((2, 2), 730.0))
    assert path.read_bytes() == saved


def test_display_reference_reuses_fields_without_scientific_execution(
    prepared, monkeypatch
):
    from blueearth_cst.climate_analysis import source_map_fields

    spec, policy = prepared

    def refuse(*args, **kwargs):
        raise AssertionError("Display must consume sealed fields, not recompute them")

    monkeypatch.setattr(source_map_fields, "prepare_source_map_fields", refuse)
    target = mp.source_fields_path(spec)
    before = target.read_bytes()
    reference = mr.build_reference([spec], "chirps", 1, policy, {})
    mr.validate_reference(reference, [spec], "chirps", 1, policy, {})
    np.testing.assert_array_equal(
        mp.read_source_fields(spec, 1, policy)[0].precip_valid_years, np.ones((2, 2))
    )
    assert target.read_bytes() == before


def test_changed_input_and_corrupt_payload_are_refused(prepared):
    spec, policy = prepared
    target = mp.source_fields_path(spec)
    with xr.open_dataset(target) as opened:
        corrupted = opened.load()
    corrupted.precip.values[0, 0] += 1
    corrupted.to_netcdf(target)
    with pytest.raises(ValueError, match="payload corruption"):
        mp.read_source_fields(spec, 1, policy)
    with xr.open_dataset(spec.climate_nc) as opened:
        changed = opened.load()
    changed.precip.values[0, 0, 0] += 1
    changed.to_netcdf(spec.climate_nc)
    with pytest.raises(ValueError, match="stale preparation"):
        mp.read_source_fields(spec, 1, policy)


def test_environment_checks_read_current_lock_bytes_with_reused_yaml(
    tmp_path, monkeypatch
):
    import yaml

    from blueearth_cst.shared import yaml_documents

    lock = tmp_path / "pixi.lock"
    data = (
        b"version: 7\nenvironments: {default: {packages: {win-64: [], linux-64: []}}}\n"
    )
    lock.write_bytes(data)
    stamp = lock.stat()
    monkeypatch.setattr(mp, "_repo_root", lambda: tmp_path)
    yaml_documents._parse.cache_clear()
    parsed = []
    safe_load = yaml.safe_load

    def counted(raw):
        parsed.append(raw)
        return safe_load(raw)

    monkeypatch.setattr(yaml, "safe_load", counted)
    try:
        for _ in range(2):
            # Only parsing is reused; environment qualification still runs.
            with pytest.raises(ValueError, match="inconsistent with Pixi lock"):
                mp.environment_identity()
        changed = data.replace(b"version: 7", b"version: 8")
        lock.write_bytes(changed)
        os.utime(lock, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        with pytest.raises(ValueError, match="requires Pixi lock schema 7"):
            mp.environment_identity()
        assert parsed == [data, changed]
    finally:
        yaml_documents._parse.cache_clear()
