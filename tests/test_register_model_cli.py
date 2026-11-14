"""The model registration CLI (t2609281102)."""

import importlib.util
from pathlib import Path

import yaml

_SPEC = importlib.util.spec_from_file_location(
    "register_model",
    Path(__file__).resolve().parents[1] / "scripts" / "register_model.py",
)
register_model = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(register_model)

_TOML = '[input]\npath_static = "staticmaps.nc"\npath_forcing = "historical/default/forcing/f.nc"\n'


def _parent(project):
    root = project / "models" / "hydrology" / "wflow"
    (root / "staticgeoms").mkdir(parents=True)
    (root / "historical" / "default" / "run").mkdir(parents=True)
    (root / "wflow_sbm.toml").write_text(_TOML, encoding="utf-8")
    (root / "staticmaps.nc").write_bytes(b"S")
    (root / "staticgeoms" / "region.geojson").write_text("{}", encoding="utf-8")
    (root / "historical" / "default" / "run" / "output.csv").write_text(
        "q", encoding="utf-8"
    )
    return root


def _run(*argv):
    return register_model.main([str(arg) for arg in argv])


def test_register_from_parent_then_list_and_refuse_again(tmp_path, capsys):
    _parent(tmp_path)
    child = tmp_path / "models" / "hydrology" / "wflow_calibrated"
    child.mkdir(parents=True)
    (child / "staticmaps.nc").write_bytes(b"CALIBRATED")
    assert (
        _run(
            "register",
            "--project-dir",
            tmp_path,
            "--name",
            "wflow_calibrated",
            "--parent",
            "wflow",
            "--copy-from-parent",
            "--notes",
            "manual",
        )
        == 0
    )
    assert (child / "wflow_sbm.toml").is_file() and not (child / "historical").exists()
    assert _run("list", "--project-dir", tmp_path) == 0
    out = capsys.readouterr().out
    assert "wflow  built" in out
    assert "wflow_calibrated  user_managed  parent=wflow  unchanged" in out
    assert (
        _run("register", "--project-dir", tmp_path, "--name", "wflow_calibrated") == 2
    )
    (child / "staticmaps.nc").write_bytes(b"EDITED")
    _run("list", "--project-dir", tmp_path)
    assert "changed since registration" in capsys.readouterr().out


def test_register_rejects_bad_names_and_missing_static_maps(tmp_path):
    assert _run("register", "--project-dir", tmp_path, "--name", "Bad-Name") == 2
    (tmp_path / "models" / "hydrology" / "empty").mkdir(parents=True)
    assert _run("register", "--project-dir", tmp_path, "--name", "empty") == 2


def test_prune_reports_first_and_refuses_used_models(tmp_path, capsys):
    _parent(tmp_path)
    assert _run("prune", "--project-dir", tmp_path, "--name", "wflow") == 0
    assert (tmp_path / "models" / "hydrology" / "wflow").is_dir()
    reference = tmp_path / "experiments" / "exp1" / "_engine" / "model_reference.yml"
    reference.parent.mkdir(parents=True)
    reference.write_text(
        yaml.safe_dump({"model_path": "models/hydrology/wflow"}), encoding="utf-8"
    )
    assert _run("prune", "--project-dir", tmp_path, "--name", "wflow", "--delete") == 1
    assert "exp1" in capsys.readouterr().out
    reference.unlink()
    assert _run("prune", "--project-dir", tmp_path, "--name", "wflow", "--delete") == 0
    assert not (tmp_path / "models" / "hydrology" / "wflow").exists()
