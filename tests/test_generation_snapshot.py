"""WF3 consumes a content pin instead of a mutable external source path."""

import hashlib
import os

import pytest

from blueearth_cst.experiment.generation_plan import (
    generation_configuration,
    snapshot_generation_input,
)
from blueearth_cst.shared.config_composition import load_composed_config
from blueearth_cst.shared.workflow_archive_launch import stage_shared_dependency

ROOT = os.path.dirname(os.path.dirname(__file__))


def test_snapshot_is_independent_of_live_source_and_refuses_changed_bytes(tmp_path):
    source = tmp_path / "external" / "extract_historical.nc"
    source.parent.mkdir()
    source.write_bytes(b"first")
    project = tmp_path / "project"
    pinned, digest, size = snapshot_generation_input(project, source)
    assert (digest, size) == (hashlib.sha256(b"first").hexdigest(), 5)
    assert pinned.read_bytes() == b"first"
    stat = source.stat()
    source.write_bytes(b"later")
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert pinned.read_bytes() == b"first"
    assert snapshot_generation_input(project, source)[0] != pinned
    assert pinned.read_bytes() == b"first"
    pinned.chmod(0o666)
    pinned.write_bytes(b"wrong")
    source.write_bytes(b"first")
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="snapshot bytes differ"):
        snapshot_generation_input(project, source.parent / source.name)


def test_wf3_region_uses_the_cross_workflow_catalog_path(tmp_path):
    """WF3 must not replace the shared region rule's catalog path with its pin."""
    config = load_composed_config(
        os.path.join(ROOT, "test_case", "project_config_rapid.yml")
    )
    project = tmp_path / "project"
    catalog = tmp_path / "deltares_data.yml"
    catalog.write_text("meta:\n  roots: [C:/data]\n", encoding="utf-8")
    config["project"]["project_dir"] = str(project)
    config["project"]["catalog"] = str(catalog)

    settings = generation_configuration(config, ROOT)
    expected = stage_shared_dependency(project, "project_catalog_0", catalog)

    assert settings["region"].inputs["catalog"] == str(expected)
    assert settings["catalogs"] == [str(catalog)]
