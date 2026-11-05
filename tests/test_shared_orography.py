"""Content-addressed WF4 orography storage remains outside WF3 collections."""

import pytest

from blueearth_cst.experiment.shared_orography import (
    publish_shared_orography,
    relative_hydromt_uri,
    resolve_hydromt_uri,
    resolve_shared_orography,
)


def _describe(path):
    return {"size_bytes": path.stat().st_size}


def test_shared_orography_reuses_verified_bytes_and_refuses_tampering(tmp_path):
    source = tmp_path / "external.nc"
    source.write_bytes(b"elevation")
    project = tmp_path / "project"
    args = {
        "source_name": "era5",
        "basename": "era5_orography_2018.nc",
        "describe": _describe,
    }
    reference, descriptor = publish_shared_orography(source, project, **args)
    target = resolve_shared_orography(project, reference, descriptor, **args)
    assert target.read_bytes() == b"elevation"
    catalog = (
        project
        / "experiments"
        / "test"
        / "hydrology"
        / "wflow"
        / "run_settings"
        / "forcing_elevation_catalog.yml"
    )
    catalog.parent.mkdir(parents=True)
    uri = relative_hydromt_uri(catalog, reference, project)
    assert uri.startswith("../../../../../data/climate/ancillary/era5/")
    assert resolve_hydromt_uri(catalog, uri, reference, project) == target
    (catalog.parent.parent / "wrong.nc").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="differs"):
        resolve_hydromt_uri(catalog, "../wrong.nc", reference, project)
    assert publish_shared_orography(source, project, **args) == (reference, descriptor)
    before = source.read_bytes()
    target.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="target differs"):
        publish_shared_orography(source, project, **args)
    assert source.read_bytes() == before
    with pytest.raises(ValueError, match="artifact reference bytes differ"):
        resolve_shared_orography(project, reference, descriptor, **args)


def test_shared_orography_uses_the_short_handle_and_still_reads_the_long_form(
    tmp_path,
):
    import shutil

    source = tmp_path / "external.nc"
    source.write_bytes(b"elevation")
    project = tmp_path / "project"
    args = {
        "source_name": "era5",
        "basename": "era5_orography_2018.nc",
        "describe": _describe,
    }
    reference, descriptor = publish_shared_orography(source, project, **args)
    target = resolve_shared_orography(project, reference, descriptor, **args)
    assert target.parent.name == reference["sha256"][:12]
    # An experiment recorded before 2026-09-24 points at the 64-character folder.
    legacy = target.parent.parent / reference["sha256"] / target.name
    legacy.parent.mkdir()
    shutil.copyfile(target, legacy)
    legacy_reference = {
        **reference,
        "path": legacy.relative_to(project.resolve()).as_posix(),
    }
    assert resolve_shared_orography(project, legacy_reference, descriptor, **args) == (
        legacy
    )
