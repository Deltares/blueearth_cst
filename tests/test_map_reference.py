"""Runtime contracts for historical map policy and retained-field provenance."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from blueearth_cst.shared import map_reference as mr


@pytest.mark.parametrize("fails", [False, True])
def test_orography_planning_hides_catalog_info_and_preserves_warnings(
    capfd, monkeypatch, fails
):
    import logging
    import sys

    import hydromt

    logger = logging.getLogger("hydromt.data_catalog.data_catalog")
    handler = logging.StreamHandler(sys.stderr)
    logger.addHandler(handler)
    original_level = logger.level
    logger.setLevel(logging.INFO)
    original_catalog = hydromt.DataCatalog

    def catalog_with_warning(**kwargs):
        logger.warning("WARNING: catalog planning diagnostic")
        catalog = original_catalog(**kwargs)
        if fails:
            raise ValueError("catalog planning failed")
        return catalog

    monkeypatch.setattr(hydromt, "DataCatalog", catalog_with_warning)
    try:
        catalogs = [Path("tests/data/tests_data_catalog.yml")]
        if fails:
            with pytest.raises(ValueError, match="catalog planning failed"):
                mr._orography_descriptor(catalogs)
        else:
            source = mr._orography_descriptor(catalogs)
            assert source.data_type == "RasterDataset"
        output = capfd.readouterr()
        assert ("Parsing data catalog" in output.out + output.err) is fails
        assert "catalog planning diagnostic" in output.err
        logger.info("catalog logging restored")
        assert "catalog logging restored" in capfd.readouterr().err
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)


@pytest.mark.parametrize("month", [28, -1, True, 1.5])
def test_reference_rejects_unsupported_month_policy(month):
    with pytest.raises(ValueError, match="max_missing_days_month"):
        mr.resolve_map_completeness({"max_missing_days_month": month})


def test_reference_accepts_unbounded_year_policy_and_requires_wf0_agreement():
    policy = mr.resolve_map_completeness(
        {"max_missing_days_month": 27, "max_missing_days_year": 1000}
    )
    mr.require_wf0_completeness(policy, policy)
    with pytest.raises(ValueError, match="must equal climate.map_reference"):
        mr.require_wf0_completeness(mr.resolve_map_completeness(), policy)
    with pytest.raises(ValueError, match="unknown keys"):
        mr.resolve_map_completeness({"extra": 3})


@pytest.mark.parametrize(
    "low,high,step",
    [
        (0.0, 1.6, 0.2),
        (20.0, 21.6, 0.2),
        (20.0, 24.0, 0.5),
        (20.0, np.nextafter(21.6, np.inf), 0.5),
        (20.0, np.nextafter(24.0, np.inf), 1.0),
        (-2.1, np.nextafter(-0.5, np.inf), 0.5),
        (-0.1, np.nextafter(3.9, np.inf), 1.0),
    ],
)
def test_temperature_class_step_is_stable_at_decimal_thresholds(low, high, step):
    scale = mr.scale_record("temp", low, high)
    np.testing.assert_allclose(np.diff(scale["boundaries"]), step, rtol=0, atol=1e-12)


def _fields():
    coords = {"latitude": [1.0], "longitude": [2.0, 2.5]}
    dims = ("latitude", "longitude")
    return xr.Dataset(
        {
            "precip": xr.DataArray(
                [[365.0, np.nan]],
                dims=dims,
                coords=coords,
                attrs={
                    "units": "mm/year",
                    "latitude_resolution": 0.5,
                    "longitude_resolution": 0.5,
                },
            ),
            "precip_valid_years": (dims, np.array([[1, 0]], dtype=np.int32)),
            "precip_unavailable": (dims, np.array([[False, True]])),
        },
        attrs={
            "reporting_year_label": "start",
            "period_start": 2001,
            "period_end": 2001,
        },
    )


def test_native_edges_preserve_quantized_float32_coordinates():
    coordinates = np.array([9.625, 9.675, 9.725, 9.775, 9.825, 9.875], dtype=np.float32)
    original = coordinates.copy()
    half = (coordinates[1] - coordinates[0]) / 2
    expected = np.r_[
        coordinates[0] - half,
        (coordinates[:-1] + coordinates[1:]) / 2,
        coordinates[-1] + half,
    ]
    edges = mr.native_edges(coordinates)
    assert edges.dtype == np.float32
    np.testing.assert_array_equal(edges, expected)
    np.testing.assert_array_equal(coordinates, original)
    # The same uneven values cannot claim float32 quantization after promotion.
    with pytest.raises(ValueError, match="regular monotonic"):
        mr.native_edges(coordinates.astype(np.float64))
    with pytest.raises(ValueError, match="regular monotonic"):
        mr.native_edges(np.array([9.625, 9.675, 9.726], dtype=np.float32))


def test_fields_require_matching_identity_and_actual_payload(tmp_path):
    fields = _fields()
    expected = {"source": "chirps", "input_sha256": "a" * 64}
    with pytest.raises(ValueError, match="regenerate WF0 compute"):
        mr.validate_fields(fields, expected, {}, source="chirps")
    entry = mr.seal_fields(fields, expected)
    path = tmp_path / "fields.nc"
    fields.to_netcdf(path)
    with xr.open_dataset(path) as reopened:
        mr.validate_fields(reopened, expected, entry, source="chirps")
        changed = reopened.load().copy(deep=True)
    changed.precip.values[0, 0] += 1
    with pytest.raises(ValueError, match="payload corruption"):
        mr.validate_fields(changed, expected, entry, source="chirps")
    with pytest.raises(ValueError, match="stale preparation identity"):
        mr.validate_fields(
            fields, {**expected, "input_sha256": "b" * 64}, entry, source="chirps"
        )
    changed = fields.copy(deep=True)
    changed.precip.attrs["longitude_resolution"] = 1.0
    with pytest.raises(ValueError, match="payload corruption"):
        mr.validate_fields(changed, expected, entry, source="chirps")


@pytest.mark.parametrize(
    "name,value",
    [
        ("precip_valid_years", 1.5),
        ("precip_unavailable", 2),
    ],
)
def test_payload_rejects_lossy_domain_coercion(name, value):
    fields = _fields()
    fields[name] = fields[name].astype(float)
    fields[name].values[0, 0] = value
    with pytest.raises(ValueError):
        mr.fields_digest(fields)


def test_payload_normalizes_nan_and_zero_but_preserves_infinity_sign():
    a = _fields()
    b = a.copy(deep=True)
    a.precip.values[0, 0] = 0.0
    b.precip.values[0, 0] = -0.0
    # Different NaN payload bits represent the same missing value.
    b.precip.values[0, 1] = np.array([0x7FF8000000000001], dtype=np.uint64).view(
        np.float64
    )[0]
    assert mr.fields_digest(a) == mr.fields_digest(b)
    a.precip.values[0, 0], b.precip.values[0, 0] = np.inf, -np.inf
    assert mr.fields_digest(a) != mr.fields_digest(b)
