"""The project display name shared by the launcher header and DAG render files."""

from blueearth_cst.shared.project_label import CASE_TRIALS_MARKER, project_name


def _trial(root, readme):
    trial = root / "gabon-ntoum" / "showcase"
    trial.mkdir(parents=True)
    (root / "gabon-ntoum" / "README.md").write_text(readme, encoding="utf-8")
    return trial / "project_config.yml", trial / "outputs"


def test_a_case_trial_is_named_after_its_case(tmp_path):
    config, outputs = _trial(tmp_path, f"# gabon-ntoum\n\n{CASE_TRIALS_MARKER}\n")
    assert project_name({}, outputs, config) == "gabon-ntoum"


def test_an_explicit_project_name_wins(tmp_path):
    config, outputs = _trial(tmp_path, CASE_TRIALS_MARKER)
    assert project_name({"project_name": "Ntoum"}, outputs, config) == "Ntoum"


def test_a_config_outside_a_case_keeps_the_folder_basename(tmp_path):
    """No marker, no case: a repository `test_case/` config is not a case."""
    config, outputs = _trial(tmp_path, "# not a case README\n")
    assert project_name({}, outputs, config) == "outputs"
    assert project_name({}, tmp_path / "test_rapid") == "test_rapid"
