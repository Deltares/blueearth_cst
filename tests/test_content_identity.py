"""Exact-byte and invalidation checks for R12 collection-canon/1."""

import hashlib

import pytest

from blueearth_cst.experiment.content_identity import (
    canonical_json_bytes,
    confined_path,
    content_sha256,
    identity_segment,
    read_canonical_json,
)


def test_canonical_bytes_have_unicode_order_and_final_lf():
    expected = '{"a":[true,null,2,1.5],"é":"水"}\n'.encode()
    value = {"é": "水", "a": [True, None, 2, 1.5]}
    assert canonical_json_bytes(value) == expected
    assert content_sha256(value) == hashlib.sha256(expected).hexdigest()
    assert content_sha256({"a": 1, "b": 2}) == content_sha256({"b": 2, "a": 1})
    assert content_sha256([1, 2]) != content_sha256([2, 1])
    assert content_sha256(True) != content_sha256(1)
    assert canonical_json_bytes([1.0, -0.0, 1e-7]) == b"[1.0,-0.0,1e-07]\n"


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}\n',
        b'{"a": 1}\n',
        b'{"a":1}',
        b'{"a":NaN}\n',
        b'{"a":1e999}\n',
        b'{"a":1}\r\n',
    ],
)
def test_persisted_documents_refuse_ambiguous_or_noncanonical_bytes(tmp_path, raw):
    path = tmp_path / "document.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        read_canonical_json(path)
    assert path.read_bytes() == raw


def test_persisted_canonical_document_round_trips(tmp_path):
    path = tmp_path / "document.json"
    value = {"source": "水", "items": [1, None, True]}
    path.write_bytes(canonical_json_bytes(value))
    assert read_canonical_json(path) == value


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        -float("inf"),
        {1: "integer key"},
        (1, 2),
        {"nested": [object()]},
    ],
)
def test_non_json_values_are_refused(value):
    with pytest.raises((ValueError, TypeError)):
        canonical_json_bytes(value)


@pytest.mark.parametrize(
    "relative",
    [
        "",
        "/a",
        "C:/a",
        "C:a",
        "a//b",
        "a/",
        "../a",
        "a/../b",
        "./a",
        "a\\b",
        "a:stream",
    ],
)
def test_noncanonical_or_escaping_paths_are_refused(tmp_path, relative):
    with pytest.raises(ValueError, match="path"):
        confined_path(tmp_path, relative)


def test_paths_resolve_from_collection_not_cwd(tmp_path, monkeypatch):
    root = tmp_path / "collection"
    root.mkdir()
    monkeypatch.chdir(tmp_path.parent)
    assert confined_path(root, "forcing/run_001.nc") == root / "forcing/run_001.nc"


def test_resolved_escape_is_refused(tmp_path):
    root = tmp_path / "collection"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (root / "link").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink unavailable: {exc}")
    with pytest.raises(ValueError, match="outside"):
        confined_path(root, "link/forcing.nc")


@pytest.mark.parametrize(
    "relative",
    ["collection.json.", "collection.json ", "dir./payload.nc", "NUL", "aux.nc"],
)
def test_windows_alias_spellings_are_not_portable_artifact_paths(tmp_path, relative):
    with pytest.raises(ValueError, match="path"):
        confined_path(tmp_path, relative)


def test_code_inventory_follows_package_initializers_and_relative_imports(tmp_path):
    from blueearth_cst.experiment.content_identity import repository_code_inventory

    package = tmp_path / "blueearth_cst"
    child = package / "stage"
    child.mkdir(parents=True)
    (package / "__init__.py").write_text("from . import helper\n")
    (package / "helper.py").write_text("VALUE = 1\n")
    (child / "__init__.py").write_text("from .. import helper\n")
    (child / "entry.py").write_text("from . import inner\n")
    (child / "inner.py").write_text("VALUE = 2\n")
    inventory = repository_code_inventory(tmp_path, ["blueearth_cst/stage/entry.py"])
    assert {entry["path"] for entry in inventory} == {
        "blueearth_cst/__init__.py",
        "blueearth_cst/helper.py",
        "blueearth_cst/stage/__init__.py",
        "blueearth_cst/stage/entry.py",
        "blueearth_cst/stage/inner.py",
    }
    (package / "__init__.py").write_text("from . import helper\nVALUE = 3\n")
    assert inventory != repository_code_inventory(
        tmp_path, ["blueearth_cst/stage/entry.py"]
    )


# --- path segments and their collision policy --------------------------------


def test_a_path_segment_is_the_leading_prefix_and_the_field_stays_complete():
    identity = "a1b2c3d4e5f6" + "0" * 52
    assert identity_segment(identity, "collection_id") == "a1b2c3d4e5f6"
    # The strict field validator is untouched: a display handle is still refused
    # wherever an identity is STORED, which is the property truncation must not
    # cost. Truncation applies to the path segment and nothing else.
    with pytest.raises(ValueError, match="collection_id"):
        identity_segment("a1b2c3d4e5f6", "collection_id")


def test_code_inventory_ignores_checkout_line_endings(tmp_path):
    from blueearth_cst.experiment.content_identity import repository_code_inventory

    package = tmp_path / "blueearth_cst"
    package.mkdir()
    entry = package / "entry.py"
    entry.write_bytes(b"A = 1\nB = 2\n")
    lf = repository_code_inventory(tmp_path, ["blueearth_cst/entry.py"])
    entry.write_bytes(b"A = 1\r\nB = 2\r\n")
    assert repository_code_inventory(tmp_path, ["blueearth_cst/entry.py"]) == lf


def _write_climate(path, *, temp=1.5, units="degC"):
    import numpy as np
    import xarray as xr

    xr.Dataset(
        {"temp": (("time", "y", "x"), np.full((3, 2, 2), temp), {"units": units})},
        coords={"time": [0, 1, 2], "y": [0.0, 1.0], "x": [0.0, 1.0]},
        attrs={"region_bbox": [0.0, 0.0, 1.0, 1.0], "source": "era5"},
    ).to_netcdf(path)


def test_netcdf_content_digest_ignores_container_bytes(tmp_path):
    import time

    from blueearth_cst.experiment.content_identity import netcdf_content_sha256

    first, second = tmp_path / "a.nc", tmp_path / "b.nc"
    _write_climate(first)
    time.sleep(1.1)
    _write_climate(second)
    assert netcdf_content_sha256(first) == netcdf_content_sha256(second)


@pytest.mark.parametrize("change", [{"temp": 1.6}, {"units": "K"}])
def test_netcdf_content_digest_sees_values_and_attributes(tmp_path, change):
    from blueearth_cst.experiment.content_identity import netcdf_content_sha256

    base, changed = tmp_path / "a.nc", tmp_path / "b.nc"
    _write_climate(base)
    _write_climate(changed, **change)
    assert netcdf_content_sha256(base) != netcdf_content_sha256(changed)


def test_source_entry_reads_both_identity_versions():
    from blueearth_cst.experiment.content_identity import (
        NETCDF_CONTENT_SCHEME,
        SOURCE_IDENTITY_V2,
        SOURCE_IDENTITY_V3,
        scientific_source_entry,
    )

    file = {"sha256": "a" * 64, "size_bytes": 10}
    metadata = {"content_scheme": NETCDF_CONTENT_SCHEME, "content_sha256": "b" * 64}
    role = "historical_climate"
    assert scientific_source_entry(role, file, metadata, SOURCE_IDENTITY_V2) == {
        "role": role,
        "sha256": "a" * 64,
        "size_bytes": 10,
    }
    assert scientific_source_entry(role, file, metadata, SOURCE_IDENTITY_V3) == {
        "role": role,
        "content_sha256": "b" * 64,
    }
    with pytest.raises(ValueError, match="lacks"):
        scientific_source_entry(role, file, {}, SOURCE_IDENTITY_V3)
    assert "size_bytes" in scientific_source_entry(
        "basin_cells", file, {}, SOURCE_IDENTITY_V3
    )
