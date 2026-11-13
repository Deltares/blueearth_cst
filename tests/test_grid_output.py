"""Grid request admission, native-file integrity and pair publication recovery."""

import shutil
import tomllib
from pathlib import Path

import netCDF4
import pytest
import tomli_w
from pyproj import CRS

from blueearth_cst.experiment.batch_staging import (
    BatchMember,
    prepare_staging,
    restore_staged,
)
from blueearth_cst.shared.grid_output import (
    configure_grid_output,
    normalize_grid_output,
    validate_grid,
)
from blueearth_cst.shared.provenance import file_sha256

STANDARD = "river_water__volume_flow_rate"


@pytest.mark.parametrize("output", [None, {}, {"netcdf_grid": None}])
def test_omission_and_null_disable(output):
    assert normalize_grid_output(output) is None


@pytest.mark.parametrize(
    "grid",
    [
        True,
        False,
        {},
        {"variables": {}},
        {"variables": {STANDARD: "x"}},
        {"variables": {STANDARD: "Bad"}},
        {"variables": {STANDARD: "q", "snowpack_liquid_water__depth": "q"}},
        {"variables": {STANDARD: "q"}, "compressionlevel": True},
        {"variables": {STANDARD: "q"}, "compressionlevel": 10},
        {"variables": {STANDARD: "q"}, "path": "somewhere.nc"},
    ],
)
def test_invalid_request_rejects(grid):
    with pytest.raises(ValueError):
        normalize_grid_output({"netcdf_grid": grid})


def test_independent_declarations_replace_inheritance():
    request = normalize_grid_output({"netcdf_grid": {"variables": {STANDARD: "q"}}})
    assert request["compressionlevel"] == 1
    assert request == normalize_grid_output({"netcdf_grid": {**request}})
    config = {}
    configure_grid_output(config, request, "output_grid.nc")
    configure_grid_output(config, None, "run_1.nc")
    assert "netcdf_grid" not in config["output"]
    configure_grid_output(config, request, "run_2.nc")
    assert config["output"]["netcdf_grid"]["path"] == "run_2.nc"


def test_grid_disk_estimate_counts_rectangular_cells_layers_and_leap_days(tmp_path):
    from blueearth_cst.experiment.batch_sizing import estimate_grid_payload

    (tmp_path / "wflow_sbm.toml").write_text(
        '[input]\nsubbasin_location__count="basin_mask"\n', encoding="utf-8"
    )
    with netCDF4.Dataset(tmp_path / "staticmaps.nc", "w") as ds:
        for name, count in (("y", 2), ("x", 3), ("layer", 4)):
            ds.createDimension(name, count)
        ds.createVariable("basin_mask", "i4", ("y", "x"))[:] = 1
    assert estimate_grid_payload(tmp_path, 2020, 2020, 3) == 6 * 4 * 366 * 3 * 8


def _pair(tmp_path):
    """Synthetic native-format payload for transaction/corruption tests only."""
    toml = tmp_path / "run_1.toml"
    toml.write_text(
        'dir_output="."\n[input]\npath_static="staticmaps.nc"\n'
        '[time]\ncalendar="standard"\n'
        'starttime="2046-01-01"\nendtime="2046-01-03"\ntimestepsecs=86400\n'
        '[output.netcdf_grid]\npath="run_1.nc"\ncompressionlevel=1\n'
        f'[output.netcdf_grid.variables]\n{STANDARD}="q"\n',
        encoding="utf-8",
    )
    with netCDF4.Dataset(tmp_path / "staticmaps.nc", "w") as ds:
        for name in ("latitude", "longitude"):
            ds.createDimension(name, 2)
            ds.createVariable(name, "f8", (name,))[:] = [1, 2]
        ds.createVariable("spatial_ref", "i4").crs_wkt = CRS.from_epsg(4326).to_wkt()
        ds.createVariable("locations", "f8", ("latitude", "longitude"))[:] = [
            [9, 2],
            [float("nan"), float("nan")],
        ]
    grid = tmp_path / "run_1.nc"
    with netCDF4.Dataset(grid, "w") as ds:
        for name, count in (("time", 2), ("lat", 2), ("lon", 2)):
            ds.createDimension(name, count)
            coord = ds.createVariable(name, "f8", (name,))
            coord[:] = [1, 2]
        ds["time"].units = "days since 2046-01-01"
        ds["time"].calendar = "standard"
        crs = ds.createVariable("spatial_ref", "i4")
        crs.crs_wkt = CRS.from_epsg(4326).to_wkt()
        value = ds.createVariable(
            "q", "f4", ("time", "lat", "lon"), fill_value=float("nan")
        )
        value[:] = 1
        value.units = "m3 s-1"
        value.wflow_standard_name = STANDARD
        value.grid_mapping = "spatial_ref"
    csv = tmp_path / "run_1.csv"
    csv.write_text("time,Q_1\n2046-01-02,1\n2046-01-03,1\n", encoding="utf-8")
    _contract(toml, tmp_path / "staticmaps.nc")
    return BatchMember("run_1", toml, csv), grid


def _contract(toml, static):
    """Independent two-by-two native-model expectation for transaction fixtures."""
    contract = {
        "schema_version": "wflow-grid-contract/1",
        "toml_sha256": file_sha256(toml),
        "static_sha256": file_sha256(static),
        "wflow_version": "fixture",
        "crs_wkt": CRS.from_epsg(4326).to_wkt(),
        "coordinates": {"lat": [1.0, 2.0], "lon": [1.0, 2.0]},
        "supports": {"river": [[1, 1], [2, 2]]},
        "variables": {
            "q": {
                "standard_name": STANDARD,
                "units": "m3 s-1",
                "dimensions": ["time", "lat", "lon"],
                "layers": [],
                "support": "river",
            }
        },
    }
    Path(str(toml) + ".grid.toml").write_text(tomli_w.dumps(contract), encoding="utf-8")


def test_truncated_grid_rejects(tmp_path):
    member, grid = _pair(tmp_path)
    assert validate_grid(grid, member.toml_path)["timesteps"] == 2
    grid.write_bytes(grid.read_bytes()[:100])
    with pytest.raises((OSError, ValueError)):
        validate_grid(grid, member.toml_path)


@pytest.mark.parametrize("corruption", ["longitude", "crs"])
def test_grid_rejects_spatial_metadata_different_from_model(tmp_path, corruption):
    member, grid = _pair(tmp_path)
    with netCDF4.Dataset(grid, "r+") as ds:
        if corruption == "longitude":
            ds["lon"][:] += 10
        else:
            ds["spatial_ref"].crs_wkt = CRS.from_epsg(3857).to_wkt()
    with pytest.raises(ValueError, match="differs from model staticmaps"):
        validate_grid(grid, member.toml_path)


@pytest.mark.parametrize("corruption", ["nan", "masked", "infinity", "units"])
def test_grid_rejects_native_support_or_unit_corruption(tmp_path, corruption):
    member, grid = _pair(tmp_path)
    with netCDF4.Dataset(grid, "r+") as ds:
        if corruption == "units":
            ds["q"].units = "kg"
        else:
            ds["q"][0, 0, 0] = {
                "nan": float("nan"),
                "masked": float("nan"),
                "infinity": float("inf"),
            }[corruption]
    with pytest.raises(ValueError, match="native"):
        validate_grid(grid, member.toml_path)


def test_grid_accepts_outside_support_masks_and_rejects_changed_contract(tmp_path):
    member, grid = _pair(tmp_path)
    with netCDF4.Dataset(grid, "r+") as ds:
        ds["q"][:, 0, 1] = float("nan")
    assert validate_grid(grid, member.toml_path)["timesteps"] == 2
    member.toml_path.write_text(member.toml_path.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="contract differs"):
        validate_grid(grid, member.toml_path)


def test_grid_requires_independent_contract_and_native_dimensions(tmp_path):
    member, grid = _pair(tmp_path)
    contract_path = Path(str(member.toml_path) + ".grid.toml")
    contract = tomllib.loads(contract_path.read_text())
    contract["variables"]["q"]["dimensions"] = ["time", "layer", "lat", "lon"]
    contract["variables"]["q"]["layers"] = [1.0, 2.0]
    contract_path.write_text(tomli_w.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="dimensions differ"):
        validate_grid(grid, member.toml_path)
    contract_path.unlink()
    with pytest.raises(FileNotFoundError):
        validate_grid(grid, member.toml_path)


@pytest.mark.parametrize("corruption", ["nan", "units", "coordinates"])
def test_retry_rejects_malformed_pair_even_with_matching_checksums(
    tmp_path, corruption
):
    member, grid = _pair(tmp_path)
    staging = tmp_path / "staging"
    prepare_staging([member], staging, "simulation")
    member.native_output_path.replace(staging / "run_1.csv")
    grid.replace(staging / "run_1.nc")
    with netCDF4.Dataset(staging / "run_1.nc", "r+") as ds:
        if corruption == "nan":
            ds["q"][0, 0, 0] = float("nan")
        elif corruption == "units":
            ds["q"].units = "kg"
        else:
            ds["lon"][:] += 10
    (staging / "run_1.sha256").write_text(
        f'csv="{file_sha256(staging / "run_1.csv")}"\n'
        f'grid="{file_sha256(staging / "run_1.nc")}"\n',
        encoding="utf-8",
    )
    (staging / "run_1.expect").replace(staging / "run_1.ok")
    with pytest.raises(ValueError):
        restore_staged([member], staging)
    assert prepare_staging([member], staging, "simulation") == [member]


def test_pair_survives_interrupted_publication(tmp_path, monkeypatch):
    member, grid = _pair(tmp_path)
    staging = tmp_path / "staging"
    prepare_staging([member], staging, "simulation")
    staged_csv = staging / "run_1.csv"
    staged_grid = staging / "run_1.nc"
    member.native_output_path.replace(staged_csv)
    grid.replace(staged_grid)
    (staging / "run_1.sha256").write_text(
        f'csv="{file_sha256(staged_csv)}"\ngrid="{file_sha256(staged_grid)}"\n',
        encoding="utf-8",
    )
    (staging / "run_1.expect").replace(staging / "run_1.ok")
    real_copy = shutil.copyfile

    def crash(source, target):
        if Path(target).suffix == ".csv":
            raise OSError("interrupted between pair copies")
        return real_copy(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(shutil, "copyfile", crash)
        with pytest.raises(OSError, match="interrupted"):
            restore_staged([member], staging)
    assert staged_csv.exists() and staged_grid.exists()
    assert not member.native_output_path.exists()
    assert prepare_staging([member], staging, "simulation") == []
    restore_staged([member], staging)
    assert member.native_output_path.read_bytes() == staged_csv.read_bytes()
    assert grid.read_bytes() == staged_grid.read_bytes()
    staged_grid.write_bytes(b"corrupted")
    assert prepare_staging([member], staging, "simulation") == [member]


def test_grid_inventory_binds_compact_responses_and_reads_without_raw_csv(
    tmp_path, monkeypatch
):
    from blueearth_cst.experiment import simulation_record
    from blueearth_cst.experiment.response_inventory import (
        publish_response_inventory_v2,
    )
    from blueearth_cst.experiment.simulation_record import (
        publish_simulation_v2,
        read_simulation_v2,
    )
    from blueearth_cst.shared.response_tables import publish_response_tables
    from tests._v2_experiment import TEMPORAL, build_v2_experiment, native_runs

    scalar = build_v2_experiment(tmp_path / "scalar")
    before = {
        path: path.read_bytes() for path in scalar.root.rglob("*") if path.is_file()
    }
    read_simulation_v2(scalar.root)
    assert before == {path: path.read_bytes() for path in before}
    original = simulation_record.simulation_intent_v2
    grid_request = {"compressionlevel": 1, "variables": {STANDARD: "q"}}

    def with_grid(name, collection, simulator, documents, identity):
        documents = {
            **documents,
            "settings": {**documents["settings"], "netcdf_grid": grid_request},
        }
        return original(name, collection, simulator, documents, identity)

    with monkeypatch.context() as patch:
        patch.setattr(simulation_record, "simulation_intent_v2", with_grid)
        experiment = build_v2_experiment(tmp_path / "grid", publish=False)
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    _, sample = _pair(sample_dir)
    runs = native_runs(experiment.root)
    for run, native in runs.items():
        native.toml_path.write_text(
            'dir_output="."\n'
            + native.toml_path.read_text(encoding="utf-8")
            .replace('header="Q"', 'header="Q"\nmap="locations"')
            .replace('header="gwr"', 'header="gwr"\nmap="locations"')
            + f'\n[input]\npath_static="{(sample_dir / "staticmaps.nc").as_posix()}"\n',
            encoding="utf-8",
        )
        with native.toml_path.open("a", encoding="utf-8") as handle:
            handle.write(
                # R17: the grid lives in simulations/<run>/, two levels above run_settings/.
                f'\n[output.netcdf_grid]\npath="../../simulations/{run}/output_grid.nc"'
                "\ncompressionlevel=1\n"
                f'[output.netcdf_grid.variables]\n{STANDARD}="q"\n'
            )
        grid = experiment.root / f"simulations/{run}/output_grid.nc"
        grid.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sample, grid)
        _contract(native.toml_path, sample_dir / "staticmaps.nc")
    first_grid = experiment.root / f"simulations/{next(iter(runs))}/output_grid.nc"
    with netCDF4.Dataset(first_grid, "r+") as ds:
        ds["q"][0, 0, 0] = float("nan")
    records = {
        run: publish_response_tables(
            native,
            experiment.root / "simulations" / run,
            variables=("gwr",),
            record_path=experiment.root
            / f"_engine/simulations/{run}/response_tables.json",
        )
        for run, native in runs.items()
    }
    tomls = {run: native.toml_path for run, native in runs.items()}
    with pytest.raises(ValueError, match="native support"):
        publish_response_inventory_v2(
            experiment.root, records, TEMPORAL, grid_tomls=tomls
        )
    shutil.copyfile(sample, first_grid)
    inventory = publish_response_inventory_v2(
        experiment.root, records, TEMPORAL, grid_tomls=tomls
    )
    assert inventory["schema_version"] == "response-inventory/5"
    assert len(inventory["grids"]) == 2
    publish_simulation_v2(experiment.root, inventory)
    # Retained reads use the inventory-bound native contract, not local inputs
    # or an engine installation; temporary producer records are unnecessary.
    for native in runs.values():
        native.csv_path.unlink()
        Path(str(native.toml_path) + ".grid.toml").unlink()
    (sample_dir / "staticmaps.nc").unlink()
    read_simulation_v2(experiment.root)
    first_grid.unlink()
    with pytest.raises(ValueError):
        read_simulation_v2(experiment.root)
