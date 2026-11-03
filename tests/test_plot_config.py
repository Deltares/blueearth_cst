"""Project annotation validation and its boundary with scientific identity."""

from pathlib import Path

import pytest

from blueearth_cst.experiment.generation_plan import generation_configuration
from blueearth_cst.shared.config_composition import load_composed_config
from blueearth_cst.shared.plot_config import resolve_annotation
from blueearth_cst.shared.provenance import effective_config_digest


@pytest.mark.parametrize("mode", ["none", "caption", "title"])
def test_annotation_modes(mode):
    assert resolve_annotation({"plotting": {"annotation": mode}}) == mode


def test_default_annotation():
    assert resolve_annotation({}) == "none"
    assert resolve_annotation({"plotting": {}}) == "none"


@pytest.mark.parametrize(
    "block",
    [
        None,
        False,
        "title",
        {"annotation": None},
        {"annotation": True},
        {"annotation": "clean"},
        {"annotations": "title"},
    ],
)
def test_invalid_plotting_is_refused(block):
    with pytest.raises(ValueError, match="plotting"):
        resolve_annotation({"plotting": block})


def test_annotations_do_not_change_generation_request_or_scientific_projection():
    root = Path(__file__).resolve().parents[1]
    config = load_composed_config(root / "test_case/project_config_rapid.yml")
    original = generation_configuration(config, root, stage=False)
    projection = ("project", "basin", "climate", "model", "workflows.build_model")
    original_digest = effective_config_digest(config, {}, projection)
    for mode in ("none", "caption", "title"):
        config["plotting"] = {"annotation": mode}
        changed = generation_configuration(config, root, stage=False)
        assert changed["request"] == original["request"]
        assert changed["request_path"] == original["request_path"]
        assert effective_config_digest(config, {}, projection) == original_digest
