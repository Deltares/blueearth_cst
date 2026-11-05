"""The composed-config snapshot WF3 and WF4 write beside their artifacts.

WF0-WF2 write one overwritable record per project. WF3 and WF4 are
many-per-project and content-addressed, so their snapshot lives inside the
collection or experiment directory. These tests pin the two properties that
make that placement safe -- the snapshot is unreferenced, so no identity moves
-- and the three gaps it exists to close.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from blueearth_cst.shared.workflow_config_snapshot import (
    SCHEMA_VERSION,
    SNAPSHOT_NAME,
    archive_lock,
    composed_workflow_section,
    render_snapshot,
    snapshot_bytes,
    snapshot_document,
    write_snapshot,
)
from dev.scripts.semantic_tree_diff import (
    apply_path_map_matched,
    build_project_tree_rules,
)

REPO = Path(__file__).resolve().parents[1]


def test_project_lock_contender_waits_without_reading_owned_byte(tmp_path):
    """A second process waits for ownership instead of failing on Windows read."""
    code = (
        "import sys; from pathlib import Path; "
        "from blueearth_cst.shared.workflow_config_snapshot import archive_lock; "
        "print('waiting', flush=True); "
        "\nwith archive_lock(Path(sys.argv[1]), 'project-execution'):"
        "\n print('owned', flush=True)"
    )
    process = None
    try:
        with archive_lock(tmp_path, "project-execution"):
            process = subprocess.Popen(
                [sys.executable, "-c", code, str(tmp_path)],
                cwd=REPO,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            assert process.stdout.readline().strip() == "waiting"
            with pytest.raises(subprocess.TimeoutExpired):
                process.communicate(timeout=0.5)
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 0, stderr
        assert stdout.strip() == "owned"
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate()


PROJECT = {
    "schema_version": 2,
    "project": {"project_dir": "test_case/test_rapid"},
    "basin": {"resolution": 0.00833},
    "workflows": {
        "generate_scenarios": {
            "enabled": True,
            "config_path": "project_config_rapid_generate_scenarios.yml",
        },
        "simulate_system": {
            "enabled": True,
            "config_path": "project_config_rapid_simulate_system.yml",
        },
    },
}


@pytest.fixture
def configs(tmp_path):
    """A project file and one workflow settings file beside it."""
    source = tmp_path / "project_config_rapid.yml"
    source.write_text(yaml.safe_dump(PROJECT), encoding="utf-8")
    workflow = tmp_path / "project_config_rapid_simulate_system.yml"
    workflow.write_text(
        "experiment_name: experiment_rapid\noperation: simulate-and-indicators\n",
        encoding="utf-8",
    )
    return source, workflow


def test_records_both_source_files_by_path_and_digest(configs):
    """The gap the digest documents left open: which file did this come from?

    `collection.json` records the resolved VALUES and never their origin, so
    "is this still the config that produced it?" had no answer anywhere in the
    scenario or experiment trees.
    """
    source, workflow = configs
    document = snapshot_document("simulate_system", source, workflow, PROJECT)
    assert document["source_config"]["path"] == str(source.resolve())
    assert re.fullmatch(r"[0-9a-f]{64}", document["source_config"]["sha256"])
    assert document["workflow_config"]["path"] == str(workflow.resolve())
    assert re.fullmatch(r"[0-9a-f]{64}", document["workflow_config"]["sha256"])
    assert document["schema_version"] == SCHEMA_VERSION
    assert document["workflow"] == "simulate_system"
    # Both resolved the same way: the project file arrives absolute and the
    # workflow file relative, and a record holding one of each leaves a reader
    # unable to tell which anchor the relative one used.
    assert Path(document["source_config"]["path"]).is_absolute()
    assert Path(document["workflow_config"]["path"]).is_absolute()


def test_digests_track_an_edit_to_either_file(configs):
    """A changed source must change the recorded digest, or the record is inert."""
    source, workflow = configs
    before = snapshot_document("simulate_system", source, workflow, PROJECT)
    workflow.write_text(
        "experiment_name: experiment_rapid\noperation: indicators-only\n",
        encoding="utf-8",
    )
    after = snapshot_document("simulate_system", source, workflow, PROJECT)
    assert before["workflow_config"]["sha256"] != after["workflow_config"]["sha256"]
    assert before["source_config"]["sha256"] == after["source_config"]["sha256"]


def test_absent_workflow_file_is_an_explicit_null(configs):
    """A project may declare no settings file; that must not read as "no record"."""
    source, _ = configs
    document = snapshot_document("generate_scenarios", source, None, PROJECT)
    assert "workflow_config" in document
    assert document["workflow_config"] is None


def test_operation_survives_into_the_snapshot(configs):
    """`operation:` reached no record anywhere before this file existed."""
    source, workflow = configs
    settings = {"experiment_name": "experiment_rapid", "operation": "indicators-only"}
    composed = composed_workflow_section(PROJECT, "simulate_system", settings)
    document = snapshot_document("simulate_system", source, workflow, composed)
    assert (
        document["effective_config"]["workflows"]["simulate_system"]["operation"]
        == "indicators-only"
    )


def test_composed_section_matches_compose_config_shape():
    """`config_path` is a pointer the composed document has already followed.

    The WF0-WF2 snapshots drop it and keep `enabled`. WF4 does not go through
    `compose_config`, so this is the one place that shape can drift.
    """
    settings = {"experiment_name": "experiment_rapid", "operation": "indicators-only"}
    composed = composed_workflow_section(PROJECT, "simulate_system", settings)
    stanzas = composed["workflows"]
    assert set(stanzas) == {"generate_scenarios", "simulate_system"}
    for stanza in stanzas.values():
        assert "config_path" not in stanza
    assert stanzas["simulate_system"]["enabled"] is True
    assert stanzas["simulate_system"]["experiment_name"] == "experiment_rapid"
    # Untouched sections travel verbatim.
    assert composed["basin"] == PROJECT["basin"]
    assert composed["schema_version"] == 2
    # The source mapping is not mutated in place.
    assert "config_path" in PROJECT["workflows"]["simulate_system"]


def test_render_is_byte_stable_and_carries_the_do_not_edit_header(configs):
    """Two runs of one configuration must produce the same bytes."""
    source, workflow = configs
    document = snapshot_document("simulate_system", source, workflow, PROJECT)
    first = render_snapshot(document)
    second = render_snapshot(dict(reversed(list(document.items()))))
    assert first == second
    assert first.startswith("# Written by the run.")
    assert "--configfile" in first.splitlines()[2] + first.splitlines()[3]


def test_rendered_snapshot_round_trips_as_yaml(configs):
    """It has to be readable with the same tool the source config is read with."""
    source, workflow = configs
    loaded = yaml.safe_load(
        snapshot_bytes("simulate_system", source, workflow, PROJECT)
    )
    assert loaded["effective_config"] == PROJECT
    assert loaded["workflow"] == "simulate_system"


def test_write_snapshot_uses_the_one_shared_name(tmp_path, configs):
    source, workflow = configs
    written = write_snapshot(
        tmp_path / "collection", "generate_scenarios", source, workflow, PROJECT
    )
    assert written.name == SNAPSHOT_NAME
    assert written.is_file()


def test_snapshot_is_named_by_no_identity_document():
    """The property that makes the placement safe.

    `simulation_id` and `collection_id` hash digests, never path strings. If
    the snapshot were ever promoted into the simulation documents or into the
    collection intent, every retained indicator set's identity would move -- so the
    name must appear in neither module's document set nor in the id projection.
    """
    for module, pattern in (
        (
            "blueearth_cst/experiment/simulation_record.py",
            r"_V2_DOCUMENTS = frozenset\((.*?)\)",
        ),
        (
            "blueearth_cst/experiment/scenario_collection_v2.py",
            r"_DOCUMENTS = \{(.*?)\}",
        ),
    ):
        text = (REPO / module).read_text(encoding="utf-8")
        body = re.search(pattern, text, re.S)
        assert body, f"{module}: document set not found"
        assert SNAPSHOT_NAME not in body.group(1)
    identity = (REPO / "blueearth_cst/experiment/simulation_record.py").read_text(
        encoding="utf-8"
    )
    projection = re.search(
        r"def _v2_simulation_id\(intent\):(.*?)\n\ndef ", identity, re.S
    )
    assert projection and SNAPSHOT_NAME not in projection.group(1)


def test_both_entry_points_publish_from_their_pinned_plan():
    """Successor publication reads the launcher-pinned plan, never live config."""
    wf3 = (REPO / "generate_scenarios.smk").read_text(encoding="utf-8")
    assert "read_pinned_plan(" in wf3
    assert 'os.environ["CST_GENERATION_PLAN_PATH"]' in wf3
    wf4 = (
        REPO / "blueearth_cst/experiment/rules/simulate_and_indicators.smk"
    ).read_text(encoding="utf-8")
    assert "freeze_simulation_v2(" in wf4
    assert "live_simulation_inputs_v2(" in wf4


def test_simulate_and_indicators_all_targets_the_selected_indicator_set():
    """The default WF4 target must deliver both halves of its operation name."""
    text = (
        REPO / "blueearth_cst/experiment/rules/simulate_and_indicators.smk"
    ).read_text(encoding="utf-8")
    rule_all = re.search(
        r"rule all:(.*?)(?:\nif not _simulation_complete:)", text, re.S
    )
    assert rule_all
    assert "_selected_indicator_outputs" in rule_all.group(1)


def test_completed_simulation_validation_suppresses_successful_dependency_chatter():
    """WF4 parse-time validation must not bypass the styled console."""
    workflow = (
        REPO / "blueearth_cst/experiment/rules/simulate_and_indicators.smk"
    ).read_text(encoding="utf-8")
    # Captured, and its warnings deferred to the run header (`defer_rows`).
    assert "with captured_output(replay=defer_rows):" in workflow


def test_tree_tooling_knows_current_provenance_paths():
    """Captured filenames are open; undeclared nested directories remain stale."""
    rules = build_project_tree_rules("experiment", "era5_20000101_20201231")
    for root in ("scenarios/abcdef123456", "experiments/experiment"):
        for leaf in (
            "run_record.yml",
            "SOURCES.md",
            "project.yml",
            "catalogs/catalog.yml",
            "sources/SOURCES.md",
            "sources/project.yml",
        ):
            path = f"{root}/config/{leaf}"
            assert apply_path_map_matched(path, rules) == (path, True)
        retired = f"{root}/config/retired/{SNAPSHOT_NAME}"
        assert apply_path_map_matched(retired, rules) == (retired, False)
