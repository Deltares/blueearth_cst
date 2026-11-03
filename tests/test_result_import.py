"""Importing WF2 bundles: integrity, compatibility, conflicts and interruption."""

import json
import shutil
import zipfile
from pathlib import Path

import pytest
import yaml

from blueearth_cst.shared import workflow_config_snapshot
from blueearth_cst.shared.result_bundle import (
    BundleError,
    export_bundle,
    import_bundle,
)
from blueearth_cst.shared.workflow_config_snapshot import validate_archive
from scripts import transfer_results
from tests.test_product_record import make_project
from tests.test_result_bundle import NOTCHED, run_product, write_region

PRODUCT = "data/climate/projections/cmip6"


@pytest.fixture
def exported(tmp_path, monkeypatch):
    """A bundle whose source project no longer exists, and an empty destination."""
    (tmp_path / "configs").mkdir()
    config, output = make_project(tmp_path / "configs")
    source = output.parent.parent / "source"
    text = config.read_text(encoding="utf-8").replace(str(output), str(source))
    config.write_text(text, encoding="utf-8")
    product = run_product(monkeypatch, config, source)
    expected = {
        path.relative_to(product).as_posix(): path.read_bytes()
        for path in product.rglob("*")
        if path.is_file()
    }
    bundle = tmp_path / "bundle.zip"
    export_bundle(source, bundle)
    shutil.rmtree(tmp_path / "source")
    destination = tmp_path / "destination"
    destination.mkdir()
    return bundle, destination, expected, config


def tree(root):
    """Destination bytes, excluding the runtime execution lock import holds."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and path.suffix != ".lock"
    }


def rewrite(bundle, output, change):
    """Copy a bundle, letting `change(members, manifest)` alter it in place."""
    with zipfile.ZipFile(bundle) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(members.pop("manifest.json"))
    change(members, manifest)
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for name, data in members.items():
            archive.writestr(name, data)
    return output


def test_round_trip_without_the_source(exported):
    bundle, destination, expected, _ = exported
    report = transfer_results.main(
        ["import", str(bundle), "--project-dir", str(destination)]
    )
    assert report == 0
    product = destination / PRODUCT
    imported = tree(product)
    carried = {name for name in imported if name.startswith("_engine/imported/")}
    assert {name: imported[name] for name in set(imported) - carried} == expected
    (directory,) = (product / "_engine/imported").iterdir()
    record = json.loads((directory / "import_record.json").read_text())
    assert record["region"]["status"] == "unverified"
    assert record["configuration"]["compared_with"] is None
    assert "none" in record["runner_adoption"]
    (snapshot,) = (directory / "configuration").iterdir()
    assert validate_archive(snapshot)["workflow"] == "analyze_projections"
    assert (directory / "dependencies/region.geojson").is_file()
    # Destination configuration archives and spatial foundation stay untouched;
    # only the runtime execution lock appears under config/.
    assert {
        path.relative_to(destination).as_posix()
        for path in (destination / "config").rglob("*")
        if path.is_file()
    } == {"config/_engine/archive-transactions/project-execution.lock"}
    assert not (destination / "data/spatial").exists()
    assert not list((destination / PRODUCT).parent.glob(".*.staged"))

    # The same bundle again is a reported no-op.
    assert import_bundle(bundle, destination)["status"] == "already_imported"
    # An imported product can travel on; its carried snapshot and region bind it.
    again = export_bundle(
        destination, destination.parent / "again.zip", ensemble="cmip6"
    )
    assert again["configuration_snapshots"]


def test_destination_polygon_must_match_not_its_bounding_box(exported):
    bundle, destination, _, _ = exported
    write_region(destination)
    report = import_bundle(bundle, destination)
    assert report["region"]["status"] == "verified"

    other = destination.parent / "other"
    other.mkdir()
    write_region(other, NOTCHED)
    before = tree(other)
    with pytest.raises(BundleError, match="basin polygon differs"):
        import_bundle(bundle, other)
    assert tree(other) == before


def _alter(members, manifest):
    name = next(item for item in members if item.endswith("report.md"))
    members[name] = b"# changed science\n"


def _unlisted(members, manifest):
    members["product/cmip6/extra.csv"] = b"x"


def _missing(members, manifest):
    del members[next(item for item in members if item.endswith("composition.csv"))]


def _unsafe(members, manifest):
    members["../escape.txt"] = b"x"


def test_link_members_are_refused(exported):
    bundle, destination, _, _ = exported
    linked = bundle.with_name("linked.zip")
    shutil.copyfile(bundle, linked)
    with zipfile.ZipFile(linked, "a") as archive:
        info = zipfile.ZipInfo("product/cmip6/link")
        info.external_attr = 0o120777 << 16
        archive.writestr(info, "../../outside")
    with pytest.raises(BundleError, match="not a file"):
        import_bundle(linked, destination)
    assert tree(destination) == {}


def _schema(members, manifest):
    manifest["schema_version"] = "cst-result-bundle/99"


def _identity(members, manifest):
    manifest["files"] = [
        item for item in manifest["files"] if not item["path"].endswith(".png")
    ]
    del members[next(item for item in members if item.endswith(".png"))]


@pytest.mark.parametrize(
    "change, message",
    [
        (_alter, "bytes differ"),
        (_unlisted, "differ from manifest"),
        (_missing, "differ from manifest"),
        (_unsafe, "unsafe"),
        (_schema, "unsupported bundle schema"),
        (_identity, "bundle id"),
    ],
)
def test_damaged_bundles_are_refused_before_any_write(exported, change, message):
    bundle, destination, _, _ = exported
    damaged = rewrite(bundle, bundle.with_name("damaged.zip"), change)
    with pytest.raises(BundleError, match=message):
        import_bundle(damaged, destination)
    assert tree(destination) == {}


def test_existing_destination_product_is_never_replaced(exported):
    bundle, destination, _, _ = exported
    existing = destination / PRODUCT / "report.md"
    existing.parent.mkdir(parents=True)
    existing.write_text("destination result", encoding="utf-8")
    before = tree(destination)
    with pytest.raises(BundleError, match="refuses to replace or merge"):
        import_bundle(bundle, destination)
    assert tree(destination) == before


def test_interrupted_import_publishes_nothing(exported, monkeypatch):
    bundle, destination, _, _ = exported

    def crash(source, target):
        raise OSError("power lost before publication")

    monkeypatch.setattr(workflow_config_snapshot, "_replace", crash)
    with pytest.raises(OSError, match="power lost"):
        import_bundle(bundle, destination)
    parent = destination / PRODUCT
    assert not parent.exists()
    assert not list(parent.parent.glob(".*.staged"))

    # A hard kill can leave staging behind; it is never the published product
    # and the next import removes it under the project lock.
    monkeypatch.undo()
    stale = parent.parent / ".cmip6.import-0000.staged"
    (stale / "summary").mkdir(parents=True)
    report = import_bundle(bundle, destination)
    assert report["removed_interrupted_imports"] == [stale.name]
    assert not stale.exists()


def test_configuration_differences_need_explicit_acceptance(exported, tmp_path):
    bundle, destination, _, config = exported
    project = yaml.safe_load(config.read_text())
    stanza = project["workflows"]["analyze_projections"]
    workflow = yaml.safe_load(Path(stanza["config_path"]).read_text())
    workflow["scenarios"] = ["ssp585"]
    changed = tmp_path / "workflow.yml"
    changed.write_text(yaml.safe_dump(workflow), encoding="utf-8")
    stanza["config_path"] = str(changed)
    destination_config = tmp_path / "destination.yml"
    destination_config.write_text(yaml.safe_dump(project), encoding="utf-8")

    with pytest.raises(BundleError, match="differs in scenarios"):
        import_bundle(bundle, destination, config=destination_config)
    assert tree(destination) == {}
    report = import_bundle(
        bundle, destination, config=destination_config, accept_differences=True
    )
    assert report["configuration"]["accepted"] is True
    assert report["configuration"]["differences"]["scenarios"] == {
        "bundle": ["ssp245"],
        "destination": ["ssp585"],
    }
