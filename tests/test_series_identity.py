"""Unit tests for ``blueearth_cst/projections/series_identity.py``.

WF2 v2.0 migration step 2b. The series store stops being ``temp()`` and becomes
persistent, so its identity is what stands between a cache and a
silent-wrong-numbers path. These tests pin the properties the design's findings
turn on:

* ``ext2-01`` / **D9** — the polygon is identified by CONTENT, so a rewritten
  geometry changes the digest and an identical geometry does not.
* ``ext2-04`` / **D12** — the physical store pin is part of the identity, and a
  re-publication is detected at read time rather than silently read.
* ``risk-03`` — the reducer version is mechanical, and a stale series fails loud.
* §5.3's exclusions — regenerating the catalog after the store gains a member
  must re-derive ZERO series; a changed shared driver block must re-derive ALL.

All offline: no network, no hydromt.
"""
from __future__ import annotations

import json

import pytest

from blueearth_cst.projections import series_identity as si


# --------------------------------------------------------------------------
# fixtures: a minimal generated-catalog shape, including the merge-key anchor
# --------------------------------------------------------------------------

ANCHOR_ENTRY = "cmip6_AAA/MODEL-1_historical_{member}"
MERGED_ENTRY = "cmip6_NOAA-GFDL/GFDL-ESM4_ssp245_{member}"

CATALOG_YAML = """\
meta:
  version: 2026.07
  crawled_on: 2026-07-29
  entries: 2
cmip6_AAA/MODEL-1_historical_{member}: &cmip6_amon
  data_type: RasterDataset
  uri: gs://cmip6/CMIP6/CMIP/AAA/MODEL-1/historical/{member}/Amon/{variable}/*/*
  driver:
    name: raster_xarray
    options:
      drop_variables:
      - time_bnds
      decode_times: true
  data_adapter:
    unit_add:
      temp: -273.15
    unit_mult:
      precip: 86400
    rename:
      pr: precip
      tas: temp
  metadata:
    crs: 4326
    category: climate
  placeholders:
    member:
    - r1i1p1f1
cmip6_NOAA-GFDL/GFDL-ESM4_ssp245_{member}:
  <<: *cmip6_amon
  uri: gs://cmip6/CMIP6/ScenarioMIP/NOAA-GFDL/GFDL-ESM4/ssp245/{member}/Amon/{variable}/*/*
  placeholders:
    member:
    - r1i1p1f1
    - r2i1p1f1
"""


@pytest.fixture
def catalog(tmp_path):
    path = tmp_path / "cmip6_data.yml"
    path.write_text(CATALOG_YAML, encoding="utf-8")
    return path


@pytest.fixture
def index(tmp_path):
    payload = {
        "generated_by": "dev/scripts/generate_cmip6_catalog.py",
        "crawled_on": "2026-07-29",
        "table": "Amon",
        "certified_variables": ["pr", "tas"],
        "sources": {
            MERGED_ENTRY: {
                "r1i1p1f1": {"pr": ["gr1/v20180701"], "tas": ["gr1/v20180701"]},
            }
        },
    }
    path = tmp_path / "cmip6_store_index.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _components(catalog_path, members=("r1i1p1f1",), **overrides):
    entry = si.load_catalog_entry(catalog_path, MERGED_ENTRY)
    base = dict(
        catalog_entry=MERGED_ENTRY,
        entry=entry,
        members=list(members),
        pins_by_member={
            "r1i1p1f1": {"pr": ["gr1/v20180701"], "tas": ["gr1/v20180701"]}
        },
        buffer_degrees=1.0,
        variable_spec=["precip", "temp"],
        experiment="ssp245",
        reducer_module_hash="deadbeef",
    )
    base.update(overrides)
    return si.digest_components(**base)


# --------------------------------------------------------------------------
# acquisition window and key grammar
# --------------------------------------------------------------------------

def test_acquisition_window_is_fixed_per_experiment_class():
    assert si.acquisition_window("historical") == ("1950-01-01", "2014-12-31")
    for ssp in ("ssp126", "ssp245", "ssp370", "ssp585", "ssp534-over"):
        assert si.acquisition_window(ssp) == ("2015-01-01", "2100-12-31")


def test_series_key_sanitizes_the_vendor_path_segment():
    """`/` in the model name would otherwise become a directory (as today)."""
    key = si.series_key(MERGED_ENTRY, "r1i1p1f1")
    assert key == "cmip6_NOAA-GFDL_GFDL-ESM4_ssp245_r1i1p1f1"
    assert "/" not in key


# --------------------------------------------------------------------------
# catalog parsing: merge keys MUST resolve
# --------------------------------------------------------------------------

def test_merge_keys_resolve_on_a_non_anchor_entry(catalog):
    """§5.3: a parser that ignores `<<` sees no driver block, silently.

    The digest would then miss the component that determines what every read
    means, so this is asserted on the MERGED entry, not the anchor.
    """
    entry = si.load_catalog_entry(catalog, MERGED_ENTRY)
    assert entry["driver"]["name"] == "raster_xarray"
    assert entry["data_adapter"]["unit_mult"]["precip"] == 86400
    assert entry["metadata"]["crs"] == 4326
    # the merged entry keeps its OWN uri, not the anchor's
    assert "ssp245" in entry["uri"]


def test_absent_entry_raises_naming_itself(catalog):
    with pytest.raises(KeyError, match="not found"):
        si.load_catalog_entry(catalog, "cmip6_NOPE_historical_{member}")


# --------------------------------------------------------------------------
# §5.3 exclusions — the three falsifiable cache consequences
# --------------------------------------------------------------------------

def test_adding_a_member_to_placeholders_does_not_change_the_digest(catalog, tmp_path):
    """Consequence 1: regeneration after the store gains a member re-derives ZERO.

    `placeholders` determines which sources EXIST (resolution), not what a given
    series read (identity).
    """
    before = _components(catalog)
    grown = CATALOG_YAML.replace("    - r2i1p1f1\n", "    - r2i1p1f1\n    - r3i1p1f1\n")
    grown_path = tmp_path / "grown.yml"
    grown_path.write_text(grown, encoding="utf-8")
    after = _components(grown_path)
    assert si.series_digest(before, "fp") == si.series_digest(after, "fp")


def test_meta_crawled_on_does_not_change_the_digest(catalog, tmp_path):
    """`crawled_on` changes on every regeneration by construction."""
    before = _components(catalog)
    recrawled = CATALOG_YAML.replace("crawled_on: 2026-07-29", "crawled_on: 2026-08-15")
    path = tmp_path / "recrawled.yml"
    path.write_text(recrawled, encoding="utf-8")
    assert si.series_digest(before, "fp") == si.series_digest(_components(path), "fp")


def test_changing_the_shared_driver_block_changes_the_digest(catalog, tmp_path):
    """Consequence 2: a changed shared block re-derives EVERY series, intentionally.

    Exercised through the MERGED entry, so it also proves the merge key is
    resolved — a parser ignoring `<<` would see no change here.
    """
    before = _components(catalog)
    changed = CATALOG_YAML.replace("      - time_bnds\n", "      - time_bnds\n      - lat_bnds\n")
    path = tmp_path / "changed_driver.yml"
    path.write_text(changed, encoding="utf-8")
    assert si.series_digest(before, "fp") != si.series_digest(_components(path), "fp")


def test_changing_unit_mult_changes_the_digest(catalog, tmp_path):
    """The adapter maps change what a read MEANS, so they are identity."""
    before = _components(catalog)
    changed = CATALOG_YAML.replace("precip: 86400", "precip: 86401")
    path = tmp_path / "changed_adapter.yml"
    path.write_text(changed, encoding="utf-8")
    assert si.series_digest(before, "fp") != si.series_digest(_components(path), "fp")


def test_changing_metadata_crs_changes_the_digest(catalog, tmp_path):
    """ext2-04's second half: metadata affects interpretation, so it is in."""
    before = _components(catalog)
    changed = CATALOG_YAML.replace("crs: 4326", "crs: 4327")
    path = tmp_path / "changed_meta.yml"
    path.write_text(changed, encoding="utf-8")
    assert si.series_digest(before, "fp") != si.series_digest(_components(path), "fp")


def test_repinned_store_changes_only_that_series_digest(catalog):
    """Consequence 3: a re-publication re-derives EXACTLY the affected series."""
    before = _components(catalog)
    after = _components(
        catalog,
        pins_by_member={
            "r1i1p1f1": {"pr": ["gr1/v20990101"], "tas": ["gr1/v20180701"]}
        },
    )
    assert si.series_digest(before, "fp") != si.series_digest(after, "fp")


def test_horizons_are_not_a_digest_component(catalog):
    """G5: `future_horizons` is not an input to stage A at all.

    Asserted structurally — no key in the component mapping mentions horizons —
    because the guarantee is "cannot be passed in", not "happens not to differ".
    """
    components = _components(catalog)
    flat = json.dumps(components, default=str).lower()
    assert "horizon" not in flat
    assert "future_horizons" not in components


# --------------------------------------------------------------------------
# D9 — region identified by content
# --------------------------------------------------------------------------

def _write_region(path, coords):
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "Polygon", "coordinates": [coords]},
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


SQUARE = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]


def test_region_fingerprint_is_stable_across_formatting(tmp_path):
    """Two writes of the SAME geometry must fingerprint identically.

    Otherwise every WF1 rerun would re-derive the archive — the property
    `ancient()` used to buy (design D9, "what happens to the property").
    """
    pytest.importorskip("geopandas")
    a = _write_region(tmp_path / "a.geojson", SQUARE)
    b = tmp_path / "b.geojson"
    # same geometry, different JSON formatting (indentation + key order)
    b.write_text(json.dumps(json.loads(a.read_text(encoding="utf-8")), indent=4), encoding="utf-8")
    assert si.region_fingerprint(a) == si.region_fingerprint(b)


def test_region_fingerprint_changes_when_the_geometry_changes(tmp_path):
    """ext2-01: a rewritten polygon must invalidate, even under an unchanged spec."""
    pytest.importorskip("geopandas")
    a = _write_region(tmp_path / "a.geojson", SQUARE)
    moved = [[x + 0.001, y] for x, y in SQUARE]
    b = _write_region(tmp_path / "b.geojson", moved)
    assert si.region_fingerprint(a) != si.region_fingerprint(b)


def test_digest_depends_on_the_region_fingerprint(catalog):
    components = _components(catalog)
    assert si.series_digest(components, "fp-one") != si.series_digest(components, "fp-two")


# --------------------------------------------------------------------------
# risk-03 — mechanical reducer version
# --------------------------------------------------------------------------

def test_module_hash_changes_with_content_and_ignores_directory(tmp_path):
    one = tmp_path / "d1"
    two = tmp_path / "d2"
    one.mkdir()
    two.mkdir()
    (one / "reducer.py").write_text("x = 1\n", encoding="utf-8")
    (two / "reducer.py").write_text("x = 1\n", encoding="utf-8")
    assert si.module_hash([one / "reducer.py"]) == si.module_hash([two / "reducer.py"])

    (two / "reducer.py").write_text("x = 2\n", encoding="utf-8")
    assert si.module_hash([one / "reducer.py"]) != si.module_hash([two / "reducer.py"])


def test_module_hash_is_order_independent(tmp_path):
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("a\n", encoding="utf-8")
    b.write_text("b\n", encoding="utf-8")
    assert si.module_hash([a, b]) == si.module_hash([b, a])


def test_module_hash_notices_a_rename(tmp_path):
    """Basename is mixed in, so moving logic between files invalidates."""
    a = tmp_path / "a.py"
    a.write_text("shared\n", encoding="utf-8")
    b = tmp_path / "b.py"
    b.write_text("shared\n", encoding="utf-8")
    assert si.module_hash([a]) != si.module_hash([b])


# --------------------------------------------------------------------------
# D12 — read-time pin verification
# --------------------------------------------------------------------------

def test_verify_pins_passes_when_the_store_matches():
    pinned = {"pr": ["gn/v1"], "tas": ["gn/v1"]}
    si.verify_pins(pinned, pinned, MERGED_ENTRY, "r1i1p1f1")  # no raise


def test_verify_pins_raises_on_a_republished_store():
    pinned = {"pr": ["gn/v1"]}
    observed = {"pr": ["gn/v2"]}
    with pytest.raises(RuntimeError, match="pin mismatch"):
        si.verify_pins(observed, pinned, MERGED_ENTRY, "r1i1p1f1")


def test_verify_pins_skips_unpinned_best_effort_variables():
    """Ruling A3: a best-effort variable has no pin to check against."""
    si.verify_pins({"kin": ["gn/v9"]}, {}, MERGED_ENTRY, "r1i1p1f1")  # no raise


def test_load_pins_returns_empty_for_a_missing_index(tmp_path):
    """A project predating the sidecar must degrade, not crash."""
    assert si.load_pins(tmp_path / "absent.json", MERGED_ENTRY, "r1i1p1f1") == {}


def test_load_pins_reads_the_recorded_pin(index):
    pins = si.load_pins(index, MERGED_ENTRY, "r1i1p1f1")
    assert pins == {"pr": ["gr1/v20180701"], "tas": ["gr1/v20180701"]}


def test_load_pins_returns_empty_for_an_unknown_member(index):
    assert si.load_pins(index, MERGED_ENTRY, "r9i9p9f9") == {}


# --------------------------------------------------------------------------
# revalidation and the stage-B backstop
# --------------------------------------------------------------------------

def _write_series(path, digest, version=si.SCHEMA_VERSION):
    xr = pytest.importorskip("xarray")
    ds = xr.Dataset({"precip": ("time", [1.0, 2.0])}, coords={"time": [0, 1]})
    ds.attrs["cst_series_digest"] = digest
    ds.attrs["cst_schema_version"] = version
    ds.to_netcdf(path)
    return path


def test_cache_hit_true_when_every_output_matches(tmp_path):
    a = _write_series(tmp_path / "a.nc", "abc")
    b = _write_series(tmp_path / "b.nc", "abc")
    assert si.cache_hit([a, b], "abc") is True


def test_cache_hit_false_when_one_output_is_missing(tmp_path):
    """A newly-enabled gridded output must force re-derivation (D9 item 3)."""
    a = _write_series(tmp_path / "a.nc", "abc")
    assert si.cache_hit([a, tmp_path / "absent.nc"], "abc") is False


def test_cache_hit_false_on_digest_or_schema_mismatch(tmp_path):
    a = _write_series(tmp_path / "a.nc", "abc")
    assert si.cache_hit([a], "different") is False
    b = _write_series(tmp_path / "b.nc", "abc", version="99")
    assert si.cache_hit([b], "abc") is False


def test_cache_hit_false_on_a_truncated_file(tmp_path):
    """An interrupted run really does leave half-written netCDFs."""
    broken = tmp_path / "broken.nc"
    broken.write_bytes(b"\x89HDF\r\n\x1a\n truncated")
    assert si.cache_hit([broken], "abc") is False


def test_assert_series_identity_passes_on_a_match(tmp_path):
    path = _write_series(tmp_path / "s.nc", "abc")
    si.assert_series_identity(path, "abc", "series X")  # no raise


def test_assert_series_identity_raises_naming_both_digests(tmp_path):
    """risk-03 mechanism 2 / D9 route (b): the in-job backstop."""
    path = _write_series(tmp_path / "s.nc", "on-disk-digest")
    with pytest.raises(RuntimeError) as excinfo:
        si.assert_series_identity(path, "expected-digest", "series X")
    message = str(excinfo.value)
    assert "on-disk-digest" in message
    assert "expected-digest" in message
    assert "series X" in message


def test_assert_series_identity_rejects_an_unknown_schema_version(tmp_path):
    path = _write_series(tmp_path / "s.nc", "abc", version="99")
    with pytest.raises(RuntimeError, match="schema version"):
        si.assert_series_identity(path, "abc", "series X")


def test_member_list_is_part_of_the_identity(catalog):
    """Step 2b loops `members:` inside one job, so the list is identity.

    Two runs whose configs list different members produce different content in
    the same output file, so they must not share a digest.
    """
    one = _components(catalog, members=["r1i1p1f1"])
    two = _components(catalog, members=["r1i1p1f1", "r2i1p1f1"])
    assert si.series_digest(one, "fp") != si.series_digest(two, "fp")


def test_member_order_does_not_change_the_identity(catalog):
    """Config list order is not meaningful; the digest must not depend on it."""
    a = _components(catalog, members=["r1i1p1f1", "r2i1p1f1"])
    b = _components(catalog, members=["r2i1p1f1", "r1i1p1f1"])
    assert si.series_digest(a, "fp") == si.series_digest(b, "fp")


# --------------------------------------------------------------------------
# kernel_hash — invalidation tracks BEHAVIOUR, not file bytes (efficiency fix 2)
# --------------------------------------------------------------------------

def test_kernel_hash_ignores_comments_docstrings_and_error_strings():
    """The step-4c lesson: an error-handling-only edit must not cost 9 remote reads.

    module_hash could not distinguish a reworded message from a changed formula,
    so an error-path edit invalidated every cached series.

    Both versions compile under the SAME function name, because the hash is
    deliberately name-sensitive (see the qualname test below) -- comparing two
    differently-named functions would conflate the two properties.
    """
    q = chr(34) * 3  # triple quote, built to avoid nesting it in this file
    src_before = chr(10).join([
        "def reduce(x):",
        "    " + q + "One docstring." + q,
        "    # a comment",
        "    if x < 0:",
        "        raise ValueError('negative')",
        "    return x * 2",
    ])
    src_after = chr(10).join([
        "def reduce(x):",
        "    " + q + "A completely different and much longer docstring." + q,
        "    # an entirely different comment, and more of it",
        "    if x < 0:",
        "        raise ValueError('the value must not be negative, see the docs')",
        "    return x * 2",
    ])
    ns_before, ns_after = {}, {}
    exec(compile(src_before, "<before>", "exec"), ns_before)
    exec(compile(src_after, "<after>", "exec"), ns_after)

    assert si.kernel_hash([ns_before["reduce"]]) == si.kernel_hash([ns_after["reduce"]])

def test_kernel_hash_changes_when_the_formula_changes():
    def before(x):
        return x * 2

    def after(x):
        return x * 3

    assert si.kernel_hash([before]) != si.kernel_hash([after])


def test_kernel_hash_changes_when_a_numeric_threshold_changes():
    """A changed constant is a behaviour change even if the code shape is identical."""
    def before(x):
        return x > 0.1

    def after(x):
        return x > 0.5

    assert si.kernel_hash([before]) != si.kernel_hash([after])


def test_kernel_hash_changes_when_an_attribute_lookup_changes():
    def before(ds):
        return ds.mean()

    def after(ds):
        return ds.sum()

    assert si.kernel_hash([before]) != si.kernel_hash([after])


def test_kernel_hash_is_order_independent_but_name_sensitive():
    def a(x):
        return x + 1

    def b(x):
        return x + 2

    assert si.kernel_hash([a, b]) == si.kernel_hash([b, a])

    def a_renamed(x):
        return x + 1

    # same body, different qualname -> moving logic between functions invalidates
    assert si.kernel_hash([a]) != si.kernel_hash([a_renamed])
