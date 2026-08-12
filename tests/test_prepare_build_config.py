"""Tests for blueearth_cst/model/prepare_build_config.py (R3 section 8). Hermetic (yaml only)."""

from pathlib import Path

import pytest
import yaml

from blueearth_cst.model.prepare_build_config import merge_build_config  # noqa: E402


def _template(path):
    path.write_text(
        yaml.safe_dump(
            {
                "modeltype": "wflow_sbm",
                "steps": [
                    {"setup_basemaps": {"hydrography_fn": "merit_hydro_ihu"}},
                    {"setup_rivers": {"river_upa": 32}},
                ],
            }
        )
    )


def test_merge_injects_region_and_res_preserving_other_keys(tmp_path):
    template = tmp_path / "build.yml"
    _template(template)
    out = tmp_path / "out" / "run.yml"  # parent does not exist yet
    merge_build_config(
        template, out, 0.00833333, "{'subbasin': [9.66, 0.44], 'uparea': 100}"
    )
    cfg = yaml.safe_load(out.read_text())
    basemaps = next(s["setup_basemaps"] for s in cfg["steps"] if "setup_basemaps" in s)
    assert basemaps["region"] == {"subbasin": [9.66, 0.44], "uparea": 100}
    assert basemaps["res"] == 0.00833333
    assert basemaps["hydrography_fn"] == "merit_hydro_ihu"  # preserved
    assert any("setup_rivers" in s for s in cfg["steps"])  # other steps untouched


def test_merge_accepts_dict_region(tmp_path):
    template = tmp_path / "build.yml"
    _template(template)
    out = tmp_path / "run.yml"
    merge_build_config(template, out, "0.05", {"subbasin": [1.0, 2.0]})
    cfg = yaml.safe_load(out.read_text())
    basemaps = next(s["setup_basemaps"] for s in cfg["steps"] if "setup_basemaps" in s)
    assert basemaps["region"] == {"subbasin": [1.0, 2.0]}
    assert basemaps["res"] == 0.05


def test_merge_raises_without_setup_basemaps(tmp_path):
    template = tmp_path / "build.yml"
    template.write_text(yaml.safe_dump({"steps": [{"setup_rivers": {}}]}))
    with pytest.raises(RuntimeError, match="setup_basemaps"):
        merge_build_config(template, tmp_path / "run.yml", 0.01, {"subbasin": [1, 2]})


# --- R07 B1: the build/store hydrography cross-check --------------------------


def _full_template(path):
    """Template declaring BOTH basin dataset keys, as the shipped one does."""
    path.write_text(
        yaml.safe_dump(
            {
                "modeltype": "wflow_sbm",
                "steps": [
                    {
                        "setup_basemaps": {
                            "hydrography_fn": "merit_hydro_ihu",
                            "basin_index_fn": "merit_hydro_index",
                        }
                    },
                    {"setup_rivers": {"hydrography_fn": "merit_hydro_ihu"}},
                ],
            }
        )
    )


def test_agreeing_hydrography_passes_and_is_not_injected(tmp_path):
    """Agreement is enforced, not shared-sourced: the template is never rewritten."""
    template = tmp_path / "build.yml"
    _full_template(template)
    out = tmp_path / "run.yml"
    merge_build_config(
        template,
        out,
        0.00833,
        {"subbasin": [1.0, 2.0]},
        hydrography="merit_hydro_ihu",
        basin_index="merit_hydro_index",
    )
    cfg = yaml.safe_load(out.read_text())
    basemaps = next(s["setup_basemaps"] for s in cfg["steps"] if "setup_basemaps" in s)
    assert basemaps["hydrography_fn"] == "merit_hydro_ihu"
    assert basemaps["basin_index_fn"] == "merit_hydro_index"
    # setup_rivers.hydrography_fn stays an intra-template concern, untouched.
    rivers = next(s["setup_rivers"] for s in cfg["steps"] if "setup_rivers" in s)
    assert rivers["hydrography_fn"] == "merit_hydro_ihu"


@pytest.mark.parametrize(
    "kwargs, needle",
    [
        ({"hydrography": "merit_hydro_1k"}, "hydrography_fn"),
        ({"basin_index": "other_index"}, "basin_index_fn"),
    ],
)
def test_disagreeing_basin_dataset_raises_naming_both_sides(tmp_path, kwargs, needle):
    """A custom template with different hydrography fails loud at the first build step."""
    template = tmp_path / "build.yml"
    _full_template(template)
    out = tmp_path / "run.yml"
    with pytest.raises(RuntimeError) as excinfo:
        merge_build_config(
            template,
            out,
            0.00833,
            {"subbasin": [1.0, 2.0]},
            snake_config_path="my_config.yml",
            **kwargs,
        )
    message = str(excinfo.value)
    assert needle in message
    # Both files and both values are named.
    assert str(template) in message
    assert "my_config.yml" in message
    assert next(iter(kwargs.values())) in message
    assert "shared.basin" in message
    # Nothing was written: the check runs before the merge.
    assert not out.exists()


def test_absent_template_key_is_not_a_disagreement(tmp_path):
    """Only keys the template declares are compared (judgement call, R07 commit 7)."""
    template = tmp_path / "build.yml"
    _template(template)  # declares hydrography_fn only
    out = tmp_path / "run.yml"
    merge_build_config(
        template,
        out,
        0.00833,
        {"subbasin": [1.0, 2.0]},
        hydrography="merit_hydro_ihu",
        basin_index="merit_hydro_index",
    )
    assert out.exists()


def test_shipped_template_excludes_competing_domain_setup():
    """P2 consumes P1 and cannot independently delineate through basemaps."""
    repo = Path(__file__).resolve().parents[1]
    template = yaml.safe_load(
        (repo / "config" / "defaults" / "wflow_build_model.yml").read_text(
            encoding="utf-8"
        )
    )
    assert not any("setup_basemaps" in step for step in template["steps"])
