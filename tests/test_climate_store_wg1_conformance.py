# -*- coding: utf-8 -*-
"""The climate store's WG-1 conformance at the write path.

Board item `t2608161450`. A chirps store failed `validate_wg1` on eight counts
and WF0 drew its figures and exited 0 regardless, so a candidate source could
WIN a forcing comparison and then fail when it was promoted to
`shared.clim_historical` and WF3 read it.

The fix is at the single write path rather than in the branch that failed, and
that is what these tests pin: a hand-assembled store — precipitation-only,
float64, carrying nothing but its bbox, which is exactly what the chirps branch
produced — comes out conforming, and an era5-shaped store that already conforms
is left alone.

The falsifier that cannot live here is the one that needs a real chirps store on
disk: whether `precip` units of `mm` are a wrong LABEL or a wrong MAGNITUDE. The
producer does not touch units for that reason, and the board note keeps that
step.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from blueearth_cst.climate_analysis import extract_historical_climate as ehc
from blueearth_cst.shared import interchange_contracts as ic


def _store(*, float64=False, variables=("precip",), attrs=None):
    """A store shaped like the chirps branch's output when float64=True."""
    dtype = "float64" if float64 else "float32"
    times = pd.date_range("2000-01-01", periods=3, freq="D")
    lat = np.array([0.35, 0.45], dtype=dtype)
    lon = np.array([9.65, 9.75], dtype=dtype)
    data = {
        name: (
            ("time", "latitude", "longitude"),
            np.ones((len(times), len(lat), len(lon)), dtype=dtype),
        )
        for name in variables
    }
    return xr.Dataset(
        data,
        coords={"time": times, "latitude": lat, "longitude": lon},
        attrs=dict(attrs or {}),
    )


class _Entry:
    def __init__(self, metadata):
        self.metadata = metadata


class _Catalog:
    """Minimal stand-in for hydromt's DataCatalog.get_source."""

    def __init__(self, metadata=None, raises=False):
        self._metadata = metadata or {}
        self._raises = raises

    def get_source(self, name):
        if self._raises:
            raise KeyError(name)
        return _Entry(self._metadata)


class TestCoerceStoreDtypes:
    def test_float64_coords_and_variables_become_float32(self):
        """One validate_wg1 row per coordinate and per variable otherwise."""
        out = ehc._coerce_store_dtypes(
            _store(float64=True, variables=("precip", "temp"))
        )
        assert str(out["latitude"].dtype) == "float32"
        assert str(out["longitude"].dtype) == "float32"
        assert str(out["precip"].dtype) == "float32"
        assert str(out["temp"].dtype) == "float32"

    def test_a_conforming_store_is_left_alone(self):
        """Idempotent, which is what lets this sit on the shared write path
        without changing what the era5 branch produces."""
        before = _store()
        after = ehc._coerce_store_dtypes(before.copy(deep=True))
        xr.testing.assert_identical(before, after)

    def test_absent_variables_are_not_invented(self):
        """A precipitation-only source has no temp; creating one is the failure
        mode `skip-outputs-for-missing-variables` exists to prevent."""
        out = ehc._coerce_store_dtypes(_store(float64=True, variables=("precip",)))
        assert list(out.data_vars) == ["precip"]

    def test_values_survive_the_narrowing_cast(self):
        store = _store(float64=True)
        store["precip"].values[:] = 4.25
        out = ehc._coerce_store_dtypes(store)
        assert float(out["precip"].mean()) == pytest.approx(4.25)


class TestStampCatalogMetadata:
    def test_the_two_contract_attrs_are_always_stamped(self):
        """crs and category are constants of WG-1, not of a source: every store
        this toolbox writes is EPSG:4326 meteorological data."""
        out = ehc._stamp_catalog_metadata(_store(), _Catalog(), "chirps")
        assert out.attrs["crs"] == 4326
        assert out.attrs["category"] == "meteo"

    def test_the_catalog_citation_block_is_carried_onto_the_store(self):
        """The eight-attribute loss the chirps branch showed was one cause, not
        two omissions; this is the other six."""
        catalog = _Catalog(
            {
                "paper_doi": "10.1038/sdata.2015.66",
                "source_url": "https://example.invalid/chirps",
                "source_version": "v2.0",
                "source_license": "CC0",
            }
        )
        out = ehc._stamp_catalog_metadata(_store(), catalog, "chirps")
        assert out.attrs["paper_doi"] == "10.1038/sdata.2015.66"
        assert out.attrs["source_version"] == "v2.0"

    def test_existing_attributes_win(self):
        """Fills gaps rather than overwriting an answer it did not compute -- so
        a source that already carried its metadata keeps what it read."""
        store = _store(attrs={"source_version": "read-from-the-file"})
        out = ehc._stamp_catalog_metadata(
            store, _Catalog({"source_version": "v9"}), "s"
        )
        assert out.attrs["source_version"] == "read-from-the-file"

    def test_a_catalog_that_cannot_answer_does_not_fail_the_run(self):
        """A missing citation block degrades the run record; it must not
        degrade the store's conformance, which is what a raise would do."""
        out = ehc._stamp_catalog_metadata(_store(), _Catalog(raises=True), "chirps")
        assert out.attrs["crs"] == 4326

    def test_structured_metadata_is_rendered_rather_than_dropped(self):
        """netCDF attributes are scalars or arrays of them; a nested block must
        still reach a reader instead of vanishing on to_netcdf."""
        out = ehc._stamp_catalog_metadata(
            _store(), _Catalog({"extent": {"bbox": [0, 1, 2, 3]}}), "chirps"
        )
        assert isinstance(out.attrs["extent"], str)
        assert "bbox" in out.attrs["extent"]

    def test_none_values_are_skipped(self):
        out = ehc._stamp_catalog_metadata(_store(), _Catalog({"paper_ref": None}), "s")
        assert "paper_ref" not in out.attrs


class TestEndToEndConformance:
    """The falsifier: the chirps-shaped store must FAIL, then PASS."""

    def _chirps_shaped(self):
        # Precip only, float64, one attribute -- measured from the real store on
        # 2026-08-16, whose entire attribute set was {'region_bbox': [...]}.
        return _store(
            float64=True,
            variables=("precip",),
            attrs={"region_bbox": [9.6, 0.3, 9.8, 0.4]},
        )

    def test_the_unfixed_store_fails_wg1(self):
        """If this ever passes, the fix below is asserting a condition that was
        already true and proves nothing."""
        diffs = ic.validate_wg1(self._chirps_shaped())
        assert diffs, "expected the pre-fix store to be non-conforming"
        joined = " ".join(diffs)
        assert "crs" in joined
        assert "category" in joined
        assert "float32" in joined

    def test_the_fixed_store_clears_the_dtype_and_attribute_rows(self):
        store = self._chirps_shaped()
        store = ehc._stamp_catalog_metadata(store, _Catalog(), "chirps")
        store = ehc._coerce_store_dtypes(store)
        diffs = ic.validate_wg1(store)
        joined = " ".join(diffs)
        assert "coord 'latitude'" not in joined
        assert "coord 'longitude'" not in joined
        assert "global attr 'crs'" not in joined
        assert "global attr 'category'" not in joined

    def test_the_run_level_provenance_attrs_are_untouched(self):
        """region_bbox and its siblings are written after this and must survive
        both calls -- they are the store's extent provenance."""
        store = self._chirps_shaped()
        store = ehc._stamp_catalog_metadata(store, _Catalog(), "chirps")
        store = ehc._coerce_store_dtypes(store)
        assert "region_bbox" in store.attrs
