"""Portable WF2 bundles: verified export, refusals and legacy products."""

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from blueearth_cst.projections.series_identity import region_fingerprint
from blueearth_cst.shared.product_record import RECORD_DIRECTORY, RECORD_NAME
from blueearth_cst.shared.result_bundle import BundleError, export_bundle
from blueearth_cst.shared.workflow_config_snapshot import read_archive
from scripts import transfer_results
from tests.test_product_record import _run, _writer, make_project

SQUARE = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
# Same bounding box as SQUARE; a different basin.
NOTCHED = [[0, 0], [1, 0], [0.5, 0.5], [1, 1], [0, 1], [0, 0]]


@pytest.fixture
def project(tmp_path):
    return make_project(tmp_path)


def write_region(project_root, ring=SQUARE):
    path = project_root / "data/spatial/geoms/region.geojson"
    path.parent.mkdir(parents=True, exist_ok=True)
    feature = {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": [feature]}),
        encoding="utf-8",
    )
    return path


def product_files(project_root, *, digest_from_archive=True, plot=b"png-1"):
    """A minimal readable WF2 product bound to the captured configuration."""
    region = write_region(project_root)

    def files(env):
        projection = read_archive(
            project_root, "analyze_projections", "config/analyze_projections"
        )["configuration_projection"]
        provenance = {
            "schema_version": "1",
            "region_fingerprint": region_fingerprint(region),
            "effective_config_sha256": projection["effective_config_sha256"]
            if digest_from_archive
            else "0" * 64,
            "configuration_inputs_sha256": projection["configuration_inputs_sha256"],
            "variable_spec": {"precip": {"units": "mm/day", "change": "relative"}},
            "sources": [{"series_key": "s1", "members": "r1i1p1f1"}],
        }
        return {
            "report.md": "# WF2\n",
            "summary/composition.csv": "models\n1\n",
            "summary/provenance.json": json.dumps(provenance),
            "_engine/analysis_record.json": json.dumps(
                {"schema_version": 1, "windows": [], "water_year_start": "Jan"}
            ),
            "plots/full_period/fig.png": plot.decode(),
        }

    return files


def run_product(monkeypatch, config, output, **options):
    product = output / "data/climate/projections/cmip6"
    build = product_files(output, **options)

    def child(env):
        return _writer(product, build(env))(env)

    assert _run(monkeypatch, config, child) == 0
    return product


def read_bundle(path):
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        members = {name: archive.read(name) for name in archive.namelist()}
    return manifest, members


def test_export_carries_product_configuration_and_region(
    project, monkeypatch, tmp_path
):
    config, output = project
    product = run_product(monkeypatch, config, output)
    bundle = tmp_path / "out/ntoum.zip"
    report = export_bundle(output, bundle)
    manifest, members = read_bundle(bundle)
    assert manifest["schema_version"] == "cst-result-bundle/1"
    assert report["bundle_id"] == manifest["bundle_id"]
    assert manifest["attribution"]["mode"] == "production-record"
    assert manifest["attribution"]["unattributed_files"] == 0
    listed = {item["path"]: item for item in manifest["files"]}
    assert set(listed) == set(members) - {"manifest.json"}
    for name, item in listed.items():
        assert hashlib.sha256(members[name]).hexdigest() == item["sha256"]
        assert len(members[name]) == item["size_bytes"]
    assert "product/cmip6/plots/full_period/fig.png" in listed
    assert f"product/cmip6/{RECORD_DIRECTORY}/{RECORD_NAME}" in listed
    (snapshot,) = manifest["configuration"]["snapshots"]
    assert f"{snapshot['path']}/run_record.yml" in listed
    assert manifest["configuration"]["provenance_snapshot"] == snapshot["path"]
    assert (
        members["dependencies/region.geojson"]
        == (output / "data/spatial/geoms/region.geojson").read_bytes()
    )
    assert manifest["compatibility"]["configuration"]["ensemble"] == "cmip6"
    assert manifest["product"]["destination"] == "data/climate/projections/cmip6"
    # No machine runtime state travels.
    assert not any(
        part in name
        for name in members
        for part in (".lock", ".snakemake", "invocations/", "execution-configs/")
    )
    assert product.is_dir()


def test_export_refuses_existing_output_and_incomplete_products(
    project, monkeypatch, tmp_path
):
    config, output = project
    product = run_product(monkeypatch, config, output)
    existing = tmp_path / "taken.zip"
    existing.write_bytes(b"keep")
    with pytest.raises(BundleError, match="already exists"):
        export_bundle(output, existing)
    assert existing.read_bytes() == b"keep"

    failed = _writer(product, {"plots/full_period/fig.png": "half"}, exit_code=1)
    assert _run(monkeypatch, config, failed) == 1
    with pytest.raises(BundleError, match="incomplete"):
        export_bundle(output, tmp_path / "incomplete.zip")
    assert not (tmp_path / "incomplete.zip").exists()


def test_export_verifies_record_against_current_bytes(project, monkeypatch, tmp_path):
    config, output = project
    product = run_product(monkeypatch, config, output)
    (product / "plots/full_period/fig.png").write_text("edited", encoding="utf-8")
    manifest, _ = read_bundle(Path(export_bundle(output, tmp_path / "b.zip")["bundle"]))
    listed = {item["path"]: item for item in manifest["files"]}
    assert listed["product/cmip6/plots/full_period/fig.png"]["producer"] is None
    assert listed["product/cmip6/report.md"]["producer"] is not None
    assert manifest["attribution"]["unattributed_files"] == 1


def test_legacy_product_binds_configuration_by_provenance_digest(
    project, monkeypatch, tmp_path
):
    config, output = project
    product = run_product(monkeypatch, config, output)
    for path in (product / RECORD_DIRECTORY).rglob("*"):
        if path.is_file():
            path.unlink()
    manifest, _ = read_bundle(Path(export_bundle(output, tmp_path / "b.zip")["bundle"]))
    assert manifest["attribution"]["mode"] == "legacy-unattributed"
    product_entries = [i for i in manifest["files"] if i["path"].startswith("product/")]
    assert all(item["producer"] is None for item in product_entries)
    assert len(manifest["configuration"]["snapshots"]) == 1


@pytest.mark.parametrize(
    "damage, message",
    [
        ("digest", "configuration digests"),
        ("region", "region fingerprint"),
        ("required", "required files"),
    ],
)
def test_export_refuses_unprovable_bundles(
    project, monkeypatch, tmp_path, damage, message
):
    config, output = project
    product = run_product(
        monkeypatch, config, output, digest_from_archive=damage != "digest"
    )
    if damage == "region":
        write_region(output, NOTCHED)
    if damage == "required":
        (product / "summary/composition.csv").unlink()
    with pytest.raises(BundleError, match=message):
        export_bundle(output, tmp_path / "b.zip")
    assert not list(tmp_path.glob("*.zip")) and not list(tmp_path.glob(".b.zip*"))


def test_cli_reports_refusals(project, tmp_path, capsys):
    _, output = project
    output.mkdir(parents=True)
    code = transfer_results.main(
        ["export", "--project-dir", str(output), "--output", str(tmp_path / "b.zip")]
    )
    assert code == 2
    assert "Refused" in capsys.readouterr().err
