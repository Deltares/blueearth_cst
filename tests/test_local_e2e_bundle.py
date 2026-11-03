"""The local E2E gate must reject changed source bytes before executing."""

from __future__ import annotations

import pytest

from dev.scripts import local_e2e_bundle


def test_sealed_bundle_rejects_modified_or_added_source(tmp_path, monkeypatch):
    monkeypatch.setattr(local_e2e_bundle, "_check_sources", lambda _bundle: "region")
    monkeypatch.setattr(local_e2e_bundle, "_contracts", lambda: {"config": "digest"})
    source = tmp_path / "hydromt" / "source.nc"
    source.parent.mkdir()
    source.write_bytes(b"source")

    local_e2e_bundle.seal_bundle(tmp_path)
    assert local_e2e_bundle.verify_bundle(tmp_path)["region_fingerprint"] == "region"

    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksums changed"):
        local_e2e_bundle.verify_bundle(tmp_path)

    source.write_bytes(b"source")
    (source.parent / "unexpected.nc").write_bytes(b"new")
    with pytest.raises(ValueError, match="inventory"):
        local_e2e_bundle.verify_bundle(tmp_path)
