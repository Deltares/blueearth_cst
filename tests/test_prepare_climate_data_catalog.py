"""Unit tests for blueearth_cst/climate_analysis/prepare_climate_data_catalog.py.

The tested function builds a hydromt 1.x data catalog dict for R-generated
realization netCDFs, inheriting driver/metadata from a source_like entry.
Mocks hydromt.DataCatalog to return a controllable fake. The
yaml.safe_dump workaround (bypassing DataCatalog.to_yml because hydromt
1.3 strips driver.options.preprocess) is itself the subject of one of
the xfail regression tests below.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


# Fake hydromt.DataCatalog. Tests configure _CATALOG to control to_dict() output.
class _FakeDataCatalog:
    _CATALOG = {}  # set by test before instantiation

    def __init__(self, data_libs=None):
        self._data_libs = data_libs

    def to_dict(self):
        # Return a deep-ish copy so tests can assert the function doesn't
        # mutate the source catalog.
        return {k: dict(v) for k, v in type(self)._CATALOG.items()}

    def from_dict(self, d):
        return self


# No sys.modules stub: each fixture below monkeypatches pcdc.hydromt.DataCatalog
# per test, which monkeypatch undoes. A module-level stub is process-wide -- a
# `setdefault` that won the race replaced the real hydromt for every later test
# in the worker, and a later `import hydromt` elsewhere then got this fake
# (t2610041227: test_downscale_climate_forcing failed on that, by xdist order).

from blueearth_cst.climate_analysis import (  # noqa: E402
    prepare_climate_data_catalog as pcdc,
)


@pytest.fixture
def era5_like_catalog(monkeypatch):
    """A minimal era5 source dict shaped like hydromt 1.x DataCatalog.to_dict()."""
    _FakeDataCatalog._CATALOG = {
        "era5": {
            "data_type": "RasterDataset",
            "uri": "/some/source/era5.nc",
            "driver": {
                "name": "netcdf",
                "options": {"preprocess": "harmonise_dims", "lock": False},
            },
            "data_adapter": {"unit_mult": {"precip": 1000.0}},
            "root": "/data/root",
            "metadata": {"crs": 4326, "category": "meteo"},
        }
    }
    monkeypatch.setattr(pcdc.hydromt, "DataCatalog", _FakeDataCatalog)
    yield
    _FakeDataCatalog._CATALOG = {}


@pytest.fixture
def chirps_like_catalog(monkeypatch):
    _FakeDataCatalog._CATALOG = {
        "chirps_global": {
            "data_type": "RasterDataset",
            "uri": "/some/source/chirps.nc",
            "driver": {"name": "netcdf"},
            "metadata": {"category": "meteo"},
        }
    }
    monkeypatch.setattr(pcdc.hydromt, "DataCatalog", _FakeDataCatalog)
    yield
    _FakeDataCatalog._CATALOG = {}


def test_single_fn_produces_one_entry_named_after_basename(tmp_path, era5_like_catalog):
    rlz_nc = tmp_path / "rlz_1_st_0.nc"
    rlz_nc.write_bytes(b"")  # the function uses str(fn.resolve()), not contents
    fn_out = tmp_path / "catalog.yml"

    pcdc.prepare_clim_data_catalog(
        fns=[rlz_nc],
        data_libs_like="dummy_catalog.yml",
        source_like="era5",
        fn_out=fn_out,
    )

    written = yaml.safe_load(fn_out.read_text())
    assert list(written.keys()) == ["rlz_1_st_0"]


def test_one_at_a_time_yields_exactly_the_aggregate_entry(tmp_path, era5_like_catalog):
    """The claim the per-member catalog rests on: building one member at a time
    produces byte-identical entries to building the whole sweep at once.

    Rule 3.13 built ONE catalog over every member and rule 3.14 read a single
    entry out of it; since 2026-08-18 rule 3.14 builds its own member's entry.
    That is only a refactor if the entry does not change, and nothing else in
    this module compares the two call shapes -- each pins one of them.
    """
    members = ["rlz_1_st_0.nc", "rlz_1_st_1.nc", "rlz_2_st_1.nc"]
    paths = []
    for name in members:
        path = tmp_path / name
        path.write_bytes(b"")
        paths.append(path)

    aggregate_fn = tmp_path / "aggregate.yml"
    pcdc.prepare_clim_data_catalog(
        fns=paths,
        data_libs_like="dummy_catalog.yml",
        source_like="era5",
        fn_out=aggregate_fn,
    )
    aggregate = yaml.safe_load(aggregate_fn.read_text())

    for path in paths:
        member_fn = tmp_path / f"{path.stem}.yml"
        pcdc.prepare_clim_data_catalog(
            fns=[path],
            data_libs_like="dummy_catalog.yml",
            source_like="era5",
            fn_out=member_fn,
        )
        member = yaml.safe_load(member_fn.read_text())
        assert list(member) == [path.stem]
        assert member[path.stem] == aggregate[path.stem]


def test_multiple_fns_produce_one_entry_per_input(tmp_path, era5_like_catalog):
    fns = []
    for name in ["rlz_1_st_0.nc", "rlz_2_st_0.nc", "rlz_3_st_0.nc"]:
        p = tmp_path / name
        p.write_bytes(b"")
        fns.append(p)
    fn_out = tmp_path / "catalog.yml"

    pcdc.prepare_clim_data_catalog(
        fns=fns,
        data_libs_like="dummy_catalog.yml",
        source_like="era5",
        fn_out=fn_out,
    )

    written = yaml.safe_load(fn_out.read_text())
    assert set(written.keys()) == {"rlz_1_st_0", "rlz_2_st_0", "rlz_3_st_0"}


def test_driver_options_preprocess_is_harmonise_dims(tmp_path, era5_like_catalog):
    """The function MUST set driver.options.preprocess='harmonise_dims' on every
    entry, regardless of what the source_like driver had. This is what the
    yaml.safe_dump workaround protects against hydromt 1.3 silently stripping."""
    rlz_nc = tmp_path / "rlz_1_st_0.nc"
    rlz_nc.write_bytes(b"")
    fn_out = tmp_path / "catalog.yml"

    pcdc.prepare_clim_data_catalog(
        fns=[rlz_nc],
        data_libs_like="dummy_catalog.yml",
        source_like="era5",
        fn_out=fn_out,
    )

    written = yaml.safe_load(fn_out.read_text())
    entry = written["rlz_1_st_0"]
    assert entry["driver"]["name"] == "raster_xarray"
    assert entry["driver"]["options"]["preprocess"] == "harmonise_dims"
    assert entry["driver"]["options"]["lock"] is False


def test_data_adapter_and_root_are_dropped(tmp_path, era5_like_catalog):
    """The R-generated NCs are already in standard units, so unit_mult /
    rename adapters from the source must be discarded. Same for `root`,
    which would point at the wrong filesystem."""
    rlz_nc = tmp_path / "rlz_1_st_0.nc"
    rlz_nc.write_bytes(b"")
    fn_out = tmp_path / "catalog.yml"

    pcdc.prepare_clim_data_catalog(
        fns=[rlz_nc],
        data_libs_like="dummy_catalog.yml",
        source_like="era5",
        fn_out=fn_out,
    )

    written = yaml.safe_load(fn_out.read_text())
    entry = written["rlz_1_st_0"]
    assert "data_adapter" not in entry
    assert "root" not in entry


def test_uri_resolves_to_absolute_path(tmp_path, era5_like_catalog):
    rlz_nc = tmp_path / "rlz_1_st_0.nc"
    rlz_nc.write_bytes(b"")
    fn_out = tmp_path / "catalog.yml"

    pcdc.prepare_clim_data_catalog(
        fns=[rlz_nc],
        data_libs_like="dummy_catalog.yml",
        source_like="era5",
        fn_out=fn_out,
    )

    written = yaml.safe_load(fn_out.read_text())
    assert Path(written["rlz_1_st_0"]["uri"]).is_absolute()


def test_processing_metadata_for_era5_source(tmp_path, era5_like_catalog):
    rlz_nc = tmp_path / "rlz_1_st_0.nc"
    rlz_nc.write_bytes(b"")
    fn_out = tmp_path / "catalog.yml"

    pcdc.prepare_clim_data_catalog(
        fns=[rlz_nc],
        data_libs_like="dummy_catalog.yml",
        source_like="era5",
        fn_out=fn_out,
    )

    written = yaml.safe_load(fn_out.read_text())
    proc = written["rlz_1_st_0"]["metadata"]["processing"]
    assert "era5" in proc
    assert "weathergenr" in proc
    assert "chirps" not in proc  # the era5-branch message


def test_processing_metadata_mentions_chirps_and_era5_for_chirps_source(
    tmp_path, chirps_like_catalog
):
    rlz_nc = tmp_path / "rlz_1_st_0.nc"
    rlz_nc.write_bytes(b"")
    oro_fn = tmp_path / "chirps_global_orography.nc"
    oro_fn.write_bytes(b"")
    fn_out = tmp_path / "catalog.yml"

    pcdc.prepare_clim_data_catalog(
        fns=[rlz_nc],
        data_libs_like="dummy_catalog.yml",
        source_like="chirps_global",
        fn_out=fn_out,
        oro_path=oro_fn,
    )

    written = yaml.safe_load(fn_out.read_text())
    proc = written["rlz_1_st_0"]["metadata"]["processing"]
    # The chirps branch explicitly notes both precip source and era5 fallback.
    assert "chirps_global" in proc
    assert "era5" in proc


def test_chirps_source_adds_orography_entry(tmp_path, chirps_like_catalog):
    """Chirps branch adds an extra '<source>_orography' entry whose uri is the
    caller-passed oro_path (design §4a: the sidecar under the keyed store dir,
    no ../.. reconstruction). The realization dir depth is now irrelevant."""
    # Realization NC under the R07 experiments/<name>/weather_generator/output/
    # layout — the rewritten lookup must NOT walk relative to this file.
    rlz_dir = tmp_path / "experiments" / "expa" / "weather_generator" / "output"
    rlz_dir.mkdir(parents=True)
    rlz_nc = rlz_dir / "rlz_1_st_0.nc"
    rlz_nc.write_bytes(b"")
    # Sidecar under the keyed store dir, passed explicitly.
    store_dir = tmp_path / "climate_historical" / "chirps_global_20000101_20201231"
    store_dir.mkdir(parents=True)
    oro_fn = store_dir / "chirps_global_orography.nc"
    oro_fn.write_bytes(b"")
    fn_out = tmp_path / "catalog.yml"

    pcdc.prepare_clim_data_catalog(
        fns=[rlz_nc],
        data_libs_like="dummy_catalog.yml",
        source_like="chirps_global",
        fn_out=fn_out,
        oro_path=oro_fn,
    )

    written = yaml.safe_load(fn_out.read_text())
    assert "chirps_global_orography" in written
    oro = written["chirps_global_orography"]
    assert oro["data_type"] == "RasterDataset"
    assert oro["driver"]["name"] == "raster_xarray"
    assert oro["metadata"]["crs"] == 4326
    # §4a: the emitted uri resolves to the passed sidecar path, which exists.
    assert Path(oro["uri"]).resolve() == oro_fn.resolve()
    assert Path(oro["uri"]).exists()


def test_chirps_source_requires_oro_path(tmp_path, chirps_like_catalog):
    """Omitting oro_path for a chirps source raises (design §4a — no silent
    fallback to a reconstructed path)."""
    rlz_dir = tmp_path / "experiments" / "expa" / "weather_generator" / "output"
    rlz_dir.mkdir(parents=True)
    rlz_nc = rlz_dir / "rlz_1_st_0.nc"
    rlz_nc.write_bytes(b"")
    fn_out = tmp_path / "catalog.yml"

    with pytest.raises(ValueError, match="oro_path"):
        pcdc.prepare_clim_data_catalog(
            fns=[rlz_nc],
            data_libs_like="dummy_catalog.yml",
            source_like="chirps_global",
            fn_out=fn_out,
        )


def test_era5_source_ignores_oro_path(tmp_path, era5_like_catalog):
    """The era5 branch adds no orography entry and needs no oro_path."""
    rlz_nc = tmp_path / "rlz_1_st_0.nc"
    rlz_nc.write_bytes(b"")
    fn_out = tmp_path / "catalog.yml"

    pcdc.prepare_clim_data_catalog(
        fns=[rlz_nc],
        data_libs_like="dummy_catalog.yml",
        source_like="era5",
        fn_out=fn_out,
    )
    written = yaml.safe_load(fn_out.read_text())
    assert not any(k.endswith("_orography") for k in written)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "hydromt 1.3 silently strips driver.options.preprocess on "
        "DataCatalog().from_dict(...).to_yml(...). "
        "blueearth_cst/climate_analysis/prepare_climate_data_catalog.py works around this with "
        "yaml.safe_dump. When upstream fixes to_yml, this test will pass, "
        "strict=True will fail CI, and the workaround can be removed. "
        "See dev/records/milestones/phase-1/m02b/handoff.md for the upstream reproducer."
    ),
)
def test_hydromt_to_yml_round_trip_preserves_preprocess(tmp_path):
    """Regression test for the M2b workaround.

    Imports the REAL hydromt to test the upstream bug. Marked xfail until
    upstream fixes it. When this passes, remove the yaml.safe_dump
    bypass in blueearth_cst/climate_analysis/prepare_climate_data_catalog.py.
    """
    import hydromt as real_hydromt

    catalog = {
        "test_source": {
            "data_type": "RasterDataset",
            "uri": "/tmp/dummy.nc",
            "driver": {
                "name": "raster_xarray",
                "options": {"preprocess": "harmonise_dims", "lock": False},
            },
        }
    }
    out_yml = tmp_path / "round_trip.yml"
    real_hydromt.DataCatalog().from_dict(catalog).to_yml(out_yml)
    written = yaml.safe_load(out_yml.read_text())

    # The assertion that fails until upstream fix:
    assert written["test_source"]["driver"]["options"]["preprocess"] == "harmonise_dims"


def test_orography_entry_points_at_the_standardised_sidecar(
    tmp_path, chirps_like_catalog
):
    """R07 B1: the emitted sidecar is `orography.nc`, not `<source>_orography.nc`.

    The CATALOG ENTRY key stays `<source>_orography` — downstream consumers
    (plot_results, the parity transform, weathergenr's downscaling) look it up
    by that name. Only the on-disk FILENAME standardises, so producer
    (rule 1.04/3.08's declared `oro_nc`) and consumer (rule 3.14's `oro_path`)
    finally agree. The seed config is era5, so nothing else in this repo would
    catch the pre-R07 mismatch.
    """
    rlz_nc = tmp_path / "rlz_1_st_0.nc"
    rlz_nc.write_bytes(b"")
    store_dir = tmp_path / "climate_historical" / "chirps_global_20000101_20201231"
    store_dir.mkdir(parents=True)
    oro_fn = store_dir / "orography.nc"  # the R07 filename
    oro_fn.write_bytes(b"")
    fn_out = tmp_path / "catalog.yml"

    pcdc.prepare_clim_data_catalog(
        fns=[rlz_nc],
        data_libs_like="dummy_catalog.yml",
        source_like="chirps_global",
        fn_out=fn_out,
        oro_path=oro_fn,
    )

    written = yaml.safe_load(fn_out.read_text())
    entry = written["chirps_global_orography"]
    assert Path(entry["uri"]).name == "orography.nc"
    assert Path(entry["uri"]).resolve() == oro_fn.resolve()
    assert Path(entry["uri"]).exists()


# --- resolved_unit_interpretation: the reviewed-variable guard (ADR 0010) -----
#
# The guard's subject is the arithmetic applied to the seven consumed variables,
# not the size of the catalog entry that carries it. These tests pin both halves:
# surplus description of OTHER variables is accepted, and anything that reaches a
# reviewed variable by a route the review did not cover is refused.

_REVIEWED_ERA5_ADAPTER = {
    "unit_add": {"temp": -273.15, "temp_min": -273.15, "temp_max": -273.15},
    "unit_mult": {"kin": 0.000277778, "kout": 0.000277778, "press_msl": 0.01},
    "rename": {
        "msl": "press_msl",
        "ssrd": "kin",
        "t2m": "temp",
        "tisr": "kout",
        "tmax": "temp_max",
        "tmin": "temp_min",
        "tp": "precip",
    },
}


def _era5_catalog_with(adapter, monkeypatch):
    """Install a fake catalog whose single `era5` entry carries `adapter`."""
    _FakeDataCatalog._CATALOG = {"era5": {"data_adapter": adapter}}
    monkeypatch.setattr(pcdc.hydromt, "DataCatalog", _FakeDataCatalog)
    monkeypatch.setattr(pcdc, "file_sha256", lambda path: "stub")


def _adapter_plus(**extra):
    """The reviewed adapter with per-key additions merged in."""
    merged = {key: dict(value) for key, value in _REVIEWED_ERA5_ADAPTER.items()}
    for key, value in extra.items():
        merged[key].update(value)
    return merged


def test_reviewed_binding_is_accepted(monkeypatch):
    _era5_catalog_with(_REVIEWED_ERA5_ADAPTER, monkeypatch)
    interpretation = pcdc.resolved_unit_interpretation(["cat.yml"], "era5")
    assert interpretation.revision.endswith("/2")


def test_surplus_variables_are_accepted(monkeypatch):
    """hydromt's own `deltares_data` entry, which the pre-ADR-0010 guard refused.

    It states the identical seven conversions and additionally describes
    dewpoint, the two wind components and net shortwave radiation -- none of
    which this toolbox reads.
    """
    _era5_catalog_with(
        _adapter_plus(
            rename={"d2m": "temp_dew", "u10": "wind10_u", "v10": "wind10_v"},
            unit_add={"temp_dew": -273.15},
            unit_mult={"ssr": 0.000277778},
        ),
        monkeypatch,
    )
    assert pcdc.resolved_unit_interpretation(["cat.yml"], "era5") is not None


@pytest.mark.parametrize(
    "adapter, why",
    [
        (
            _adapter_plus(unit_mult={"precip": 1000.0}),
            "scales a reviewed variable the review left unconverted",
        ),
        (
            _adapter_plus(unit_add={"precip": 1.0}),
            "offsets a reviewed variable the review left unconverted",
        ),
        (
            _adapter_plus(rename={"mtpr": "precip"}),
            "renames another native variable onto a reviewed name",
        ),
        (
            _adapter_plus(unit_mult={"press_msl": 1.0}),
            "changes a reviewed conversion's value",
        ),
        (
            {
                **{
                    key: value
                    for key, value in _REVIEWED_ERA5_ADAPTER.items()
                    if key != "unit_add"
                },
                "unit_add": {"temp": -273.15, "temp_min": -273.15},
            },
            "drops a reviewed conversion",
        ),
        ({}, "states no arithmetic at all"),
    ],
)
def test_departures_from_the_reviewed_variables_are_refused(adapter, why, monkeypatch):
    _era5_catalog_with(adapter, monkeypatch)
    with pytest.raises(ValueError, match="UnverifiedForcingUnits"):
        pcdc.resolved_unit_interpretation(["cat.yml"], "era5")


def test_chirps_precipitation_branch_stays_exact(monkeypatch):
    """The CHIRPS assertion is that precipitation carries NO scaling.

    `unit_mult` being empty is the substance of that claim, so this branch keeps
    exact equality where the era5 branch loosened -- ADR 0010 says so explicitly.
    """
    _FakeDataCatalog._CATALOG = {
        "era5": {"data_adapter": _REVIEWED_ERA5_ADAPTER},
        "chirps": {
            "data_adapter": {
                "rename": {"precipitation": "precip"},
                "unit_add": {"time": 86400},
                "unit_mult": {"precip": 2.0},
            }
        },
    }
    monkeypatch.setattr(pcdc.hydromt, "DataCatalog", _FakeDataCatalog)
    monkeypatch.setattr(pcdc, "file_sha256", lambda path: "stub")
    with pytest.raises(ValueError, match="precipitation/time adapter"):
        pcdc.resolved_unit_interpretation(["cat.yml"], "chirps")
