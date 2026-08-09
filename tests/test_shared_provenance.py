"""Tests for deterministic shared CST provenance digests."""

from collections import OrderedDict
from pathlib import Path

import pytest

from blueearth_cst.shared.provenance import (
    SHORT_DIGEST_CHARS,
    canonical_data,
    canonical_sha256,
    effective_config_digest,
    file_sha256,
    short_digest,
    snapshot_bundle_digest,
)


def test_short_digest_is_a_prefix_of_the_full_digest() -> None:
    """The naming form must stay findable from the record it stands for."""
    digest = canonical_sha256({"a": 1})

    assert len(short_digest(digest)) == SHORT_DIGEST_CHARS
    assert digest.startswith(short_digest(digest))


def test_short_digest_rejects_a_value_that_is_not_a_digest() -> None:
    """Truncating a non-digest would name an artifact after nothing."""
    with pytest.raises(ValueError):
        short_digest("abc")
    with pytest.raises(TypeError):
        short_digest(None)


def test_canonical_digest_is_mapping_order_independent() -> None:
    """Equivalent mappings hash identically regardless of insertion order."""
    first = {"nested": {"b": 2, "a": 1}, "items": [True, None, 1.5]}
    second = OrderedDict([("items", [True, None, 1.5]), ("nested", {"a": 1, "b": 2})])

    assert canonical_sha256(first) == canonical_sha256(second)


def test_canonical_data_preserves_supported_types() -> None:
    """Type tags prevent distinct supported values from collapsing."""
    values = [None, False, 0, 0.0, "0", Path("data/input.nc"), [], ()]

    documents = [canonical_data(value) for value in values]

    assert len({canonical_sha256(value) for value in values}) == len(values)
    assert documents[-2]["type"] == "list"
    assert documents[-1]["type"] == "tuple"
    assert documents[5] == {"type": "path", "value": "data/input.nc"}


def test_canonical_data_rejects_unsupported_values() -> None:
    """Unsupported objects fail instead of acquiring unstable repr strings."""
    with pytest.raises(TypeError, match="unsupported provenance value"):
        canonical_data({"bad": object()})


def test_file_sha256_hashes_exact_bytes(tmp_path: Path) -> None:
    """File digests are byte-based rather than text-normalized."""
    source = tmp_path / "source.yml"
    source.write_bytes(b"key: value\r\n")

    assert (
        file_sha256(source)
        == "db5735d4d5b4974b003686308b1e5e4564d1e42e27187b13cabfd53b638cdb8f"
    )


def test_effective_config_digest_covers_advanced_settings() -> None:
    """Toolbox-wide settings participate in the scientific config identity."""
    config = {"project": {"project_dir": "out"}}
    first = {"defaults": {"julia_threads": 2}}
    second = {"defaults": {"julia_threads": 4}}

    assert effective_config_digest(config, first) != effective_config_digest(
        config, second
    )


def test_snapshot_bundle_digest_hashes_files_but_not_physical_paths(
    tmp_path: Path,
) -> None:
    """Local references use content identity and logical references use IDs."""
    source = tmp_path / "config.yml"
    source.write_text("project: test\n", encoding="utf-8")
    first_file = tmp_path / "first.yml"
    second_file = tmp_path / "second.yml"
    first_file.write_text("same: bytes\n", encoding="utf-8")
    second_file.write_text("same: bytes\n", encoding="utf-8")
    common = [{"kind": "catalog", "identifier": "era5"}]

    first = snapshot_bundle_digest(
        {},
        {},
        source,
        [*common, {"kind": "template", "identifier": "forcing", "path": first_file}],
    )
    second = snapshot_bundle_digest(
        {},
        {},
        source,
        [*common, {"kind": "template", "identifier": "forcing", "path": second_file}],
    )

    assert first == second


def test_snapshot_bundle_digest_rejects_ambiguous_reference() -> None:
    """Reference descriptors require explicit kind and logical identifier."""
    with pytest.raises(ValueError, match="kind.*identifier"):
        snapshot_bundle_digest({}, {}, __file__, [{"path": __file__}])
