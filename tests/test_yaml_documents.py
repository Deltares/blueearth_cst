"""Parsing reuse must preserve current bytes and independent mutable results."""

import pytest
import yaml

from blueearth_cst.shared import yaml_documents


def test_exact_bytes_reuse_parsing_without_sharing_mutations(monkeypatch):
    yaml_documents._parse.cache_clear()
    loads = []
    safe_load = yaml.safe_load

    def counted(data):
        loads.append(data)
        return safe_load(data)

    monkeypatch.setattr(yaml, "safe_load", counted)
    data = b"settings: &settings {years: [2000, 2016]}\nalias: *settings\n"
    first = yaml_documents.parse_yaml_document(data)
    first["settings"]["years"].append(2020)
    second = yaml_documents.parse_yaml_document(data)
    assert second["settings"]["years"] == [2000, 2016]
    assert second["settings"] is second["alias"]
    changed = data.replace(b"2016", b"2017")
    assert yaml_documents.parse_yaml_document(changed)["settings"]["years"] == [
        2000,
        2017,
    ]
    assert loads == [data, changed]
    yaml_documents._parse.cache_clear()


def test_unsafe_yaml_is_still_refused():
    with pytest.raises(yaml.constructor.ConstructorError):
        yaml_documents.parse_yaml_document(b"!!python/object:builtins.object {}")
