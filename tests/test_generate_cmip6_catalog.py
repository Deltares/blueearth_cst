"""Unit tests for ``dev/scripts/generate_cmip6_catalog.py``'s store-index pass.

WF2 v2.0 migration step 2a (design D12, finding ``ext2-04``). The catalog URI
ends ``/{variable}/*/*``, so grid label and version are globbed away and an
entry name identifies a *logical* source, not the bytes read. ``pin_stores``
walks the two hidden levels and records what the crawl observed.

These tests are offline: ``gcsfs`` is never contacted. A fake filesystem stands
in for the bucket so the resolution logic, the multi-match case, and the
one-crawl coupling between catalog and index are checkable in CI.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "dev" / "scripts" / "generate_cmip6_catalog.py"


@pytest.fixture(scope="module")
def gen():
    """Import the maintenance script by path (``dev/scripts/`` is not a package)."""
    pytest.importorskip("gcsfs", reason="generator imports gcsfs at module scope")
    spec = importlib.util.spec_from_file_location("_generate_cmip6_catalog", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeFS:
    """Minimal ``fs.ls`` over a dict of directory -> children.

    Mirrors gcsfs's contract as ``pin_stores`` uses it: ``ls`` returns full
    paths, and a missing directory raises ``FileNotFoundError``.
    """

    def __init__(self, tree: dict[str, list[str]]):
        self.tree = tree
        self.calls: list[str] = []

    def ls(self, path: str) -> list[str]:
        self.calls.append(path)
        if path not in self.tree:
            raise FileNotFoundError(path)
        return [f"{path}/{child}" for child in self.tree[path]]


def _bucket(**stores: dict[str, list[str]]) -> dict[str, list[str]]:
    """Build a fake bucket tree from ``{variable: {grid: [versions]}}`` maps."""
    base = "cmip6/CMIP6/CMIP/INST/MODEL/historical/r1i1p1f1/Amon"
    tree: dict[str, list[str]] = {}
    for variable, grids in stores.items():
        tree[f"{base}/{variable}"] = sorted(grids)
        for grid, versions in grids.items():
            tree[f"{base}/{variable}/{grid}"] = sorted(versions)
    return tree


INVENTORY = {("CMIP", "INST/MODEL", "historical"): ["r1i1p1f1"]}


def test_pins_the_single_grid_version_pair_behind_the_glob(gen):
    fs = FakeFS(_bucket(pr={"gn": ["v20190815"]}, tas={"gn": ["v20190815"]}))
    payload = gen.pin_stores(fs, INVENTORY)

    entry = "cmip6_INST/MODEL_historical_{member}"
    assert payload["sources"][entry]["r1i1p1f1"] == {
        "pr": ["gn/v20190815"],
        "tas": ["gn/v20190815"],
    }
    assert payload["multi_match_count"] == 0


def test_records_every_match_when_the_glob_is_ambiguous(gen):
    """The inventory's `NCC/NorCPM1` case: one variable, two published versions.

    Recorded rather than silently collapsed — a consumer needing exactly one
    store asserts ``len == 1`` and fails loudly instead of reading whichever
    the glob happened to order first.
    """
    fs = FakeFS(
        _bucket(
            pr={"gn": ["v20190914"]},
            tas={"gn": ["v20190914", "v20200724"]},
        )
    )
    payload = gen.pin_stores(fs, INVENTORY)

    entry = "cmip6_INST/MODEL_historical_{member}"
    assert payload["sources"][entry]["r1i1p1f1"]["tas"] == [
        "gn/v20190914",
        "gn/v20200724",
    ]
    assert payload["multi_match_count"] == 1


def test_multiple_grid_labels_are_both_recorded(gen):
    """Grid label is globbed away too — `gn` and `gr` are different stores."""
    fs = FakeFS(_bucket(pr={"gn": ["v1"], "gr": ["v1"]}, tas={"gn": ["v1"]}))
    payload = gen.pin_stores(fs, INVENTORY)

    entry = "cmip6_INST/MODEL_historical_{member}"
    assert payload["sources"][entry]["r1i1p1f1"]["pr"] == ["gn/v1", "gr/v1"]
    assert payload["multi_match_count"] == 1


def test_absent_store_yields_an_empty_pin_not_an_error(gen):
    """A variable directory that does not exist resolves to no pins.

    ``pin_stores`` runs over the crawl's own inventory, so this should not
    happen for a certified variable — but it must not raise if the bucket
    changes under a long crawl.
    """
    fs = FakeFS(_bucket(pr={"gn": ["v1"]}))  # tas missing entirely
    payload = gen.pin_stores(fs, INVENTORY)

    entry = "cmip6_INST/MODEL_historical_{member}"
    assert payload["sources"][entry]["r1i1p1f1"]["tas"] == []


def test_only_certified_variables_are_pinned(gen):
    """`kin`/`press_msl` are best-effort, never pinned — they were never checked."""
    fs = FakeFS(_bucket(pr={"gn": ["v1"]}, tas={"gn": ["v1"]}, rsds={"gn": ["v1"]}))
    payload = gen.pin_stores(fs, INVENTORY)

    entry = "cmip6_INST/MODEL_historical_{member}"
    assert set(payload["sources"][entry]["r1i1p1f1"]) == set(gen.REQUIRED_VARS)
    assert not any("rsds" in call for call in fs.calls)


def test_entry_key_matches_the_catalog_entry_name(gen):
    """The index joins to the catalog by entry name, so the spelling must match."""
    fs = FakeFS(_bucket(pr={"gn": ["v1"]}, tas={"gn": ["v1"]}))
    payload = gen.pin_stores(fs, INVENTORY)

    catalog = gen.render(INVENTORY, crawled_on="2026-07-29")
    for entry in payload["sources"]:
        assert f"{entry}:" in catalog, (
            f"index key {entry!r} does not appear as a catalog entry; the two "
            "artifacts would not join"
        )


def test_catalog_and_index_are_written_from_one_crawl_date(gen, tmp_path, monkeypatch):
    """``crawled_on`` is stamped once and shared — the equal-date assertion (R14).

    Guards the desync risk the design names: two artifacts written from separate
    crawls could disagree about which members exist, with nothing to detect it.
    """
    import json

    monkeypatch.setattr(
        gen.gcsfs,
        "GCSFileSystem",
        lambda *a, **k: FakeFS(_bucket(pr={"gn": ["v1"]}, tas={"gn": ["v1"]})),
    )
    monkeypatch.setattr(gen, "crawl", lambda fs: INVENTORY)
    out = tmp_path / "cmip6_data.yml"
    index_out = tmp_path / "cmip6_store_index.json"
    monkeypatch.setattr(
        sys, "argv", ["gen", "--out", str(out), "--index-out", str(index_out)]
    )

    gen.main()

    catalog_text = out.read_text(encoding="utf-8")
    index = json.loads(index_out.read_text(encoding="utf-8"))
    catalog_date = next(
        line.split("crawled_on:")[1].strip()
        for line in catalog_text.splitlines()
        if "crawled_on:" in line
    )
    assert index["crawled_on"] == catalog_date
    assert index["table"] == gen.TABLE
    assert index["certified_variables"] == sorted(gen.REQUIRED_VARS)


def test_no_index_flag_leaves_the_index_untouched(gen, tmp_path, monkeypatch):
    """``--no-index`` writes only the catalog, and says the index is now stale."""
    monkeypatch.setattr(
        gen.gcsfs,
        "GCSFileSystem",
        lambda *a, **k: FakeFS(_bucket(pr={"gn": ["v1"]}, tas={"gn": ["v1"]})),
    )
    monkeypatch.setattr(gen, "crawl", lambda fs: INVENTORY)
    out = tmp_path / "cmip6_data.yml"
    index_out = tmp_path / "cmip6_store_index.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["gen", "--out", str(out), "--index-out", str(index_out), "--no-index"],
    )

    gen.main()

    assert out.exists()
    assert not index_out.exists()
