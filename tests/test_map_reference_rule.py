"""Shared historical reference declarations must agree across WF0 and WF1."""

from pathlib import Path

import pytest

from blueearth_cst.shared.config_composition import load_composed_config
from tests.conftest import parse_workflow, write_config
from tests.test_climate_store_contract import _COMPARED

pytestmark = pytest.mark.workflow_contract
CONFIG = Path(__file__).parent / "project_config_fixture.yml"


@pytest.mark.parametrize("selected", ["era5", "chirps"])
def test_reference_and_extra_extractions_are_identical_in_both_dags(tmp_path, selected):
    cfg = load_composed_config(CONFIG)
    cfg["project"]["project_dir"] = (tmp_path / "project").as_posix()
    cfg["climate"].update(sources=["era5", "chirps"], selected=selected)
    config = write_config(tmp_path, cfg)
    a = parse_workflow("analyze_climate.smk", config)
    b = parse_workflow("build_model.smk", config)
    ar = a.get_rule("produce_historical_map_reference")
    br = b.get_rule("produce_historical_map_reference")
    for name, signature in _COMPARED.items():
        assert signature(a, ar) == signature(b, br), name
    for source in ("era5", "chirps"):
        ar = a.get_rule(f"prepare_historical_map_fields_{source}")
        br = b.get_rule(f"prepare_historical_map_fields_{source}")
        for name, signature in _COMPARED.items():
            assert signature(a, ar) == signature(b, br), (source, name)
        diagnostic = a.get_rule(f"compute_climate_diagnostics_{source}")
        assert set(ar.output).isdisjoint(diagnostic.output)
        preparation_code = set(ar.input.preparation_code)
        assert not any(
            name in path
            for path in preparation_code
            for name in ("map_reference.py", "prepared_maps.py", "climate_figures.py")
        )
    extra = "chirps" if selected == "era5" else "era5"
    ar = a.get_rule(f"extract_climate_datasets_{extra}")
    br = b.get_rule(f"extract_map_candidates_{extra}")
    for name, signature in _COMPARED.items():
        if name != "name":
            assert signature(a, ar) == signature(b, br), name
    assert br.params.enforce_min_years is False
    if extra == "era5":
        assert "forcing_required" not in dict(br.params.items())


def test_wf0_policy_disagreement_is_rejected_at_dag_parse(tmp_path):
    cfg = load_composed_config(CONFIG)
    cfg["climate"]["map_reference"] = {"max_missing_days_month": 2}
    config = write_config(tmp_path, cfg)
    with pytest.raises(Exception, match="must equal climate.map_reference"):
        parse_workflow("analyze_climate.smk", config)
