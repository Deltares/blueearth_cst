"""Reuse frozen responses across audited non-scientific implementation changes."""

from pathlib import Path
from typing import Any

from blueearth_cst.experiment.content_identity import content_sha256
from blueearth_cst.experiment.simulation_record import (
    SimulationFrozenError,
    _v2_simulation_id,
)

# Keep the original complete inventory as provenance. Only this observational
# module is ignored when comparing an existing simulation with current inputs.
_DIAGNOSTIC_CODE = {"blueearth_cst/shared/execution_outputs.py"}

# Audited source-boundary transition from 040a8e61: no numerical producer
# changed. Pin complete inventories on both sides, including the new boundary
# definition. An unknown old revision or any subsequent scientific edit fails
# closed. The old inventory is retained in tests/data for migration coverage.
_BOUNDARY_BEFORE = "69dcf9de9882f029d615873460572f1a1d9f639dee8bc2a7ab51e01514085a2c"
# Re-audited after merging main's reference migration: scientific-source
# comments/docstrings changed, while executable calculations stayed identical.
# Re-audited after 14e423cc: the response reader only gained the native PET
# mapping and a PET-only declaration check; existing variables read unchanged.
_BOUNDARY_PREVIOUS = "fe2e4d92b48cbbca8a29ade215e55951dba5bfbf1417ff29ebde3845212f5cfa"
# Re-audited after f6841501: temperature extends the native mapping and its
# declaration guard only; all seven existing variables read identically.
# Model TOML bytes and response requirements remain independently checked.
# See dev/records/validation/qcrit-temperature-outputs.md for audit evidence.
_BOUNDARY_TEMPERATURE = (
    "12c02734236bbb1d4fa0cf9b4538f815d0c9d196074432f11378b5baed3a1b76"
)
# Re-audited after 094bab3a: provenance.py only gained the display helper
# toolbox_release_tag and its `re` import; no existing function changed and no
# simulation path calls the helper.
_BOUNDARY_AFTER = "96e3512bf017e8a92cd95e574cf8785d21bd32c10f5470aa64a901cc83dd8991"
# Audited inventories, oldest first. Any older-to-newer transition is accepted.
_AUDITED_BOUNDARIES = (
    _BOUNDARY_BEFORE,
    _BOUNDARY_PREVIOUS,
    _BOUNDARY_TEMPERATURE,
    _BOUNDARY_AFTER,
)

# Audited archive-lifecycle migration: input staging and execution settings are
# unchanged. Pin BOTH versions, rather than exempting this launcher wholesale:
# it also owns scientific dependency staging, which must remain fingerprinted.
# Any subsequent edit requires a new audit; it is not accepted by this bridge.
_ARCHIVE_LAUNCH = "blueearth_cst/shared/workflow_archive_launch.py"
_ARCHIVE_BEFORE = "5e6820a0fd81387a5e449d6755c2e3b39af2c1487aa1c6342a41675a36958f14"
_ARCHIVE_PREVIOUS = {
    _ARCHIVE_LAUNCH: "b5ec137d352bbe1cb7594a8fd1de259be4bf3f52a868ff7dbea55ac84fe26ff1",
    "blueearth_cst/shared/configuration_archive.py": (
        "02d8168156949d51a2bbc4cbd06d73c750db74a9134bfbd35f3ad1c1ecbe4e80"
    ),
}
# Re-audited after t2610221058: the launcher only gained WF2 product observation
# and production-record publication around the child; staging and execution
# settings are unchanged. Current simulation inventories exclude this launcher.
_ARCHIVE_PRODUCT = {
    _ARCHIVE_LAUNCH: "1efda5e4ad723b44214b23ba601356e992e40f38abbcd8a7839436b35cec3938",
    "blueearth_cst/shared/configuration_archive.py": (
        "ddee2d3ad42c37f875824915d78474dc94d996939d3df14def93397019724257"
    ),
}
# Schema-3 portability audit: declarations resolve before staging; source bytes
# remain exact and dependency bytes/settings remain independently fingerprinted.
# Schema-2 staging is unchanged. No scientific source inventory changed.
# Covered by test_config_paths (relocation/staging), archive lifecycle, and reuse
# rejection tests. Keep the preceding product-observation bridge valid as well.
_ARCHIVE_PORTABLE = {
    **_ARCHIVE_PRODUCT,
    _ARCHIVE_LAUNCH: "28592d85d8c7c9cb3614b53b7ddf3c018a139de671fa781c3adeee1546498508",
}
# Re-audited after 7697721c: only --unlock admission and its comment changed.
# Unlock runs no producer; scientific staging, inputs and execution are unchanged.
# Preserve the schema-3 target as well as allowing its upgrade to this revision.
_ARCHIVE_AFTER = {
    **_ARCHIVE_PORTABLE,
    _ARCHIVE_LAUNCH: "9895b3f9ecffb8fda76a1db58cfbb079a38e68696c8672180ec821811d7a4915",
}


def _archive_upgrade_paths(
    retained: list[dict[str, str]], current: list[dict[str, str]]
) -> set[str]:
    old = {item["path"]: item["sha256"] for item in retained}
    new = {item["path"]: item["sha256"] for item in current}
    if len(old) != len(retained) or len(new) != len(current):
        return set()
    if {path: old[path] for path in _ARCHIVE_AFTER if path in old} in (
        {_ARCHIVE_LAUNCH: _ARCHIVE_BEFORE},
        _ARCHIVE_PREVIOUS,
        _ARCHIVE_PRODUCT,
        _ARCHIVE_PORTABLE,
    ) and any(
        all(new.get(path) == checksum for path, checksum in target.items())
        for target in (_ARCHIVE_PRODUCT, _ARCHIVE_PORTABLE, _ARCHIVE_AFTER)
    ):
        return set(_ARCHIVE_AFTER)
    return set()


def check_simulation_reuse(retained: dict[str, Any], current: dict[str, Any]) -> None:
    """Accept identical scientific inputs without rewriting the frozen intent.

    Older intents include execution diagnostics through static transitive
    imports. Compare the remaining inventory exactly, then use the existing
    identity calculation to check every other identity component unchanged.
    """
    if current["simulation_id"] == retained["simulation_id"]:
        return
    ignored = _DIAGNOSTIC_CODE | _archive_upgrade_paths(
        retained["documents"]["simulator_adapter_code"],
        current["documents"]["simulator_adapter_code"],
    )
    retained_code = retained["documents"]["simulator_adapter_code"]
    current_code = current["documents"]["simulator_adapter_code"]
    code_matches = [item for item in retained_code if item["path"] not in ignored] == [
        item for item in current_code if item["path"] not in ignored
    ]
    code_matches = code_matches or (
        content_sha256(retained_code),
        content_sha256(current_code),
    ) in {
        (old, new)
        for index, old in enumerate(_AUDITED_BOUNDARIES)
        for new in _AUDITED_BOUNDARIES[index + 1 :]
    }
    comparable = {
        **current,
        "document_digests": {
            **current["document_digests"],
            "simulator_adapter_code": retained["document_digests"][
                "simulator_adapter_code"
            ],
        },
    }
    if code_matches and _v2_simulation_id(comparable) == retained["simulation_id"]:
        return
    raise SimulationFrozenError("simulation inputs changed; use a new experiment name")


def check_completed_simulation_inputs(
    experiment_root,
    *,
    project_root,
    model_root,
    collection_path,
    collection,
    resolution_mode,
    preparation,
    simulation_window,
    julia_command,
    include_unperturbed=True,
    _ready_simulation=None,
) -> None:
    """Recheck a completed simulation without regenerating its native layout.

    Wflow's output layout is derived from the model and installed simulator.
    Retain that frozen layout only if freshly observed model bytes, code,
    environment, preparation, settings and selected collection still match.
    This saves Julia startup/JIT without caching any input observation.
    """
    from blueearth_cst.experiment.content_identity import (
        content_sha256,
        read_canonical_json,
        stage_environment,
    )
    from blueearth_cst.experiment.simulation_code import simulation_code_inventory
    from blueearth_cst.experiment.simulation_record import (
        _assert_model_reference_current,
        read_simulation_v2,
        simulation_intent_v2,
    )
    from blueearth_cst.experiment.simulation_runs import selected_run_ids
    from blueearth_cst.experiment.wf4_ancillary_descriptor import (
        preparation_identity_v2,
    )
    from blueearth_cst.experiment.write_model_reference import build_model_reference
    from blueearth_cst.shared.workflow_config_snapshot import (
        file_reference,
        resolve_file_reference,
    )

    root = Path(experiment_root).resolve()
    project = Path(project_root).resolve()
    model = Path(model_root).resolve()
    ready = read_simulation_v2(root) if _ready_simulation is None else _ready_simulation
    retained = read_canonical_json(
        resolve_file_reference(ready["intent"], {"project_root": project})
    )
    reference = build_model_reference(model, project)
    _assert_model_reference_current(
        retained["documents"]["model_reference"], reference, model
    )
    repo = Path(__file__).resolve().parents[2]
    code = simulation_code_inventory(repo)
    environment = stage_environment(
        ["hydromt-wflow", "xarray", "netCDF4", "pyproj"],
        julia_manifest=repo / "Manifest.toml",
        julia_command=julia_command,
    )
    documents = {
        "model_reference": reference,
        "settings": {"simulation_window": dict(simulation_window)},
        "environment": environment,
        "simulator_adapter_code": code,
        "response_request": retained["documents"]["response_request"],
        "preparation": preparation,
    }
    if not include_unperturbed:
        documents["settings"]["include_unperturbed"] = False
    selection = {
        "resolution_mode": resolution_mode,
        "manifest": file_reference(collection_path, "project_root", project),
        "collection_id": collection["collection_id"],
        "collection_revision": collection["collection_revision"],
        "run_ids": selected_run_ids(
            collection, collection_path, include=include_unperturbed
        ),
    }
    simulator = {
        "name": "wflow",
        "revision": content_sha256(
            {"julia:Wflow": environment["packages"]["julia:Wflow"]}
        ),
    }
    current = simulation_intent_v2(
        root.name,
        selection,
        simulator,
        documents,
        preparation_identity_v2(
            preparation, project_root=project, experiment_root=root
        ),
    )
    check_simulation_reuse(retained, current)
