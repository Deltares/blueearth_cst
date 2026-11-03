"""Shared palette loading, comparison identity and explicit scientific presets."""

import json
import subprocess
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest
import yaml

from blueearth_cst.shared import plot_colors as colors
from blueearth_cst.shared import plot_setup, plot_style


def test_expanded_original_cycle_and_comparison_identity():
    from blueearth_cst.climate_analysis.diagnostic_figures import source_styles
    from blueearth_cst.shared.plot_evaluation import COLOR_OBSERVED, COLOR_SIMULATED

    assert colors.DEFAULT_COLOR_CYCLE == (
        "#0D38E0",
        "#CC593D",
        "#00B389",
        "#D99A19",
        "#8054B3",
        "#BE5084",
        "#169CC4",
        "#66768A",
    )
    styles = source_styles(["CHIRPS", "era5", "other", "another"])
    assert styles["era5"][0] == COLOR_OBSERVED == "#0D38E0"
    assert styles["CHIRPS"][0] == COLOR_SIMULATED == "#CC593D"
    assert styles["other"][0] == "#00B389"
    assert styles["another"][0] == "#D99A19"
    assert source_styles(["a", "b"])["b"][0] == "#CC593D"
    # Additional categories retain every source and gain another marker on wrap.
    many = source_styles([f"source{i}" for i in range(9)])
    assert len(many) == 9 and many["source0"][1] != many["source8"][1]


def test_shared_contexts_apply_cycle_without_changing_process_defaults():
    from blueearth_cst.shared.plot_preset import STYLE

    original = mpl.rcParams["axes.prop_cycle"]
    for settings in (plot_style.rcparams(), STYLE):
        with plt.rc_context(settings):
            fig, ax = plt.subplots()
            lines = [ax.plot([0, 1], [i, i + 1])[0] for i in range(8)]
            assert (
                tuple(line.get_color() for line in lines) == colors.DEFAULT_COLOR_CYCLE
            )
            plt.close(fig)
    assert mpl.rcParams["axes.prop_cycle"] == original


def test_scientific_presets_are_preserved():
    from blueearth_cst.projections.projection_figures import SCENARIO_COLORS

    assert SCENARIO_COLORS == {
        "ssp119": "#00a9cf",
        "ssp126": "#003466",
        "ssp434": "#2274ae",
        "ssp245": "#f69320",
        "ssp534-over": "#92397a",
        "ssp460": "#b0724e",
        "ssp370": "#df0000",
        "ssp370-lowNTCF": "#e61d25",
        "ssp585": "#980002",
    }
    assert colors.PRECIP_ANOMALY_COLORMAP == "BrBG"
    assert colors.HISTORICAL_MAP_COLORS == {
        "temp": {
            "colors": ["#ffffcc", "#ffe895", "#fec45f", "#fd953f", "#fa4a29"],
            "under": "#fffbe6",
            "over": "#a50f15",
        },
        "precip_pet": {
            "colors": [
                "#deebf7",
                "#c6dbef",
                "#9ecae1",
                "#6baed6",
                "#4292c6",
                "#2171b5",
                "#08519c",
            ],
            "under": "#f7fbff",
            "over": "#041f4a",
        },
    }
    for name in (*colors.CLIMATE_COLORMAPS.values(), colors.PRECIP_ANOMALY_COLORMAP):
        assert name in mpl.colormaps


@pytest.mark.parametrize(
    "section,key,value,message",
    [
        ("default_cycle", "original_blue", "blue", "six-digit hex"),
        ("default_cycle", "original_blue", None, "six-digit hex"),
        ("default_cycle", "original_blue", "#cc593d", "distinct"),
        ("datasets", "era5", "absent", "must name"),
        ("evaluation", "observed", ["original_blue"], "must name"),
        ("climate_maps", "precip", "", "colormap name"),
    ],
)
def test_invalid_palette_reports_file_and_setting(
    tmp_path, section, key, value, message
):
    palette = yaml.safe_load(colors.PALETTE_PATH.read_text(encoding="utf-8"))
    palette[section][key] = value
    path = tmp_path / "plot_colors.yml"
    path.write_text(yaml.safe_dump(palette), encoding="utf-8")
    with pytest.raises(ValueError, match=message) as exc:
        colors.load_palette(path)
    assert str(path) in str(exc.value)


def test_yaml_edits_change_renderer_identity_and_resolve_named_colours(
    tmp_path, monkeypatch
):
    path = tmp_path / "plot_colors.yml"
    path.write_bytes(colors.PALETTE_PATH.read_bytes())
    (tmp_path / "colors.yml").write_bytes(path.read_bytes())
    (tmp_path / "style.yml").write_bytes(
        (plot_setup.PLOTTING_DIR / "style.yml").read_bytes()
    )
    monkeypatch.setattr(plot_setup, "PLOTTING_DIR", tmp_path)
    before = plot_setup.setup_digest()
    path.write_text(path.read_text().replace("#0D38E0", "#123456"), encoding="utf-8")
    (tmp_path / "colors.yml").write_bytes(path.read_bytes())
    assert plot_setup.setup_digest() != before
    loaded = colors.load_palette(path)
    assert loaded["datasets"]["era5"] == loaded["evaluation"]["observed"] == "#123456"


def test_style_edits_change_renderer_identity(tmp_path, monkeypatch):
    for name in ("colors.yml", "style.yml"):
        (tmp_path / name).write_bytes((plot_setup.PLOTTING_DIR / name).read_bytes())
    monkeypatch.setattr(plot_setup, "PLOTTING_DIR", tmp_path)
    before = plot_setup.setup_digest()
    style = plot_setup.load_style()
    style["page"]["figure_width_mm"] = 160
    (tmp_path / "style.yml").write_text(yaml.safe_dump(style), encoding="utf-8")
    assert plot_setup.setup_digest() != before
    assert plot_setup.load_style()["page"]["figure_width_mm"] == 160


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("page", "raster_dpi", -1),
        ("typography", "base", True),
        ("charts", "fallback_markers", []),
    ],
)
def test_invalid_style_reports_the_setting(tmp_path, section, key, value):
    style = plot_setup.load_style()
    style[section][key] = value
    path = tmp_path / "style.yml"
    path.write_text(yaml.safe_dump(style), encoding="utf-8")
    with pytest.raises(ValueError, match=f"{section}.{key}"):
        plot_setup.load_style(path)


@pytest.mark.process_isolation
def test_palette_and_workflow_contract_imports_stay_matplotlib_free():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, json; import blueearth_cst.shared.plot_colors; "
            "import blueearth_cst.projections.projection_figures; "
            "print(json.dumps([name for name in ('matplotlib', 'cartopy') if name in sys.modules]))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == []


@pytest.mark.workflow_contract
@pytest.mark.parametrize(
    "snakefile,rules",
    [
        (
            "analyze_climate.smk",
            ["plot_climate_diagnostics_era5", "plot_climate_comparison"],
        ),
        (
            "build_model.smk",
            ["plot_model_evaluation", "plot_basin_map", "plot_model_forcing"],
        ),
        ("analyze_projections.smk", ["plot_climate_projections"]),
    ],
)
def test_palette_edits_reach_renderers_without_changing_science(
    tmp_path, monkeypatch, snakefile, rules
):
    from blueearth_cst.shared.config_composition import load_composed_config
    from tests.conftest import parse_workflow, write_config

    root = Path(__file__).resolve().parents[1]
    config = load_composed_config(root / "test_case/project_config_rapid.yml")
    config["project"]["project_dir"] = (tmp_path / "project").as_posix()
    config["project"]["catalog"] = (
        root / "tests/data/tests_data_catalog.yml"
    ).as_posix()
    config["climate"].update(sources=["era5", "chirps"], selected="era5")
    config_path = write_config(tmp_path, config)
    path = tmp_path / "colors.yml"
    path.write_bytes(colors.PALETTE_PATH.read_bytes())
    (tmp_path / "style.yml").write_bytes(
        (plot_setup.PLOTTING_DIR / "style.yml").read_bytes()
    )
    monkeypatch.setattr(plot_setup, "PLOTTING_DIR", tmp_path)
    before = parse_workflow(snakefile, config_path)
    original_digest = before.globals["EFFECTIVE_CONFIG_DIGEST"]
    old = plot_setup.setup_digest()
    for name in rules:
        assert before.get_rule(name).params.plot_setup_digest == old
    path.write_text(path.read_text().replace("#0D38E0", "#123456"), encoding="utf-8")
    after = parse_workflow(snakefile, config_path)
    assert after.globals["EFFECTIVE_CONFIG_DIGEST"] == original_digest
    for name in rules:
        assert (
            after.get_rule(name).params.plot_setup_digest
            == plot_setup.setup_digest()
            != old
        )
