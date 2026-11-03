"""Reservation-aware WF3 orchestration, outside the generator code inventory."""

import hashlib
from copy import deepcopy
from pathlib import Path

from blueearth_cst.experiment.content_identity import (
    atomic_record,
    canonical_json_bytes,
    confined_path,
    read_canonical_json,
)
from blueearth_cst.experiment.generation_plan import (
    build_candidate_intent,
    build_reuse_plan,
    generation_configuration,
    publish_plan_pointer,
    read_pinned_plan,
    select_generation_plan,
)
from blueearth_cst.experiment.generation_publication import validate_installed_generator
from blueearth_cst.experiment.scenario_collection_v2 import read_collection_v2
from blueearth_cst.shared.workflow_config_snapshot import file_reference


def bind_ready_collection_owned(
    root,
    name,
    marker_path,
    *,
    resolution_mode="project-generation",
    generation_plan=None,
):
    """Atomically bind a ready collection and any previously unbound plan seed."""
    from blueearth_cst.experiment.reservation import (
        bind_reservation_owned,
        read_reservation,
    )
    from blueearth_cst.experiment.simulation_runs import (
        include_unperturbed,
        selected_run_ids,
    )

    root = Path(root).resolve()
    marker_path = confined_path(
        root, Path(marker_path).resolve().relative_to(root).as_posix()
    )
    marker = read_collection_v2(marker_path)
    reservation = read_reservation(root, name)
    resolution = {
        "resolution_mode": resolution_mode,
        "manifest": file_reference(marker_path, "project_root", root),
        "collection_id": marker["collection_id"],
        "collection_revision": marker["collection_revision"],
        "run_ids": selected_run_ids(
            marker,
            marker_path,
            include=include_unperturbed(reservation["request"]["simulation"]),
        ),
    }
    seed = None
    if generation_plan is not None:
        plan_path = confined_path(root, generation_plan["path"])
        plan = read_pinned_plan(plan_path, generation_plan["sha256"], root)
        if plan["collection_id"] != marker["collection_id"] or (
            plan["collection_revision"] is not None
            and plan["collection_revision"] != marker["collection_revision"]
        ):
            raise ValueError("ready collection differs from selected generation plan")
        seed = _plan_seed_binding(plan, generation_plan)
    bind_reservation_owned(root, name, seed=seed, resolution=resolution)
    return marker


def _plan_seed_binding(plan, reference):
    config_document = plan["candidate_intent"]["documents"]["generation_config"]
    return {
        "state": "bound",
        "requested": config_document["seed"]["requested"],
        "resolved": config_document["seed"]["resolved"],
        "seed_resolution": config_document["seed_resolution"],
        "plan": reference,
        "source_identity_sha256": plan["candidate_identity_digests"][
            "source_inventory"
        ],
    }


def select_reserved_plan_owned(config, repository, root, name):
    """Return (settings, plan, path, sha256, ready) before ordinary plan selection."""
    from blueearth_cst.experiment.reservation import (
        bind_reservation_owned,
        compare_request,
        read_reservation,
    )

    root = Path(root).resolve()
    reservation = read_reservation(root, name)
    settings = generation_configuration(config, repository)
    candidate = build_candidate_intent(settings)
    seed = reservation["seed"]
    if seed["state"] == "bound":
        plan_path = confined_path(root, seed["plan"]["path"])
        plan = read_pinned_plan(plan_path, seed["plan"]["sha256"], root)
        relocated = canonical_json_bytes(plan["request"]) != canonical_json_bytes(
            settings["request"]
        )
        current_request = deepcopy(settings["request"])
        if relocated:
            # These are locators, not scientific identities. Candidate digests
            # and seed evidence below must still match the bound creator.
            current_request["template"] = plan["request"]["template"]
            current_request["settings"]["weathergen_config"] = plan["request"][
                "settings"
            ]["weathergen_config"]
            if len(current_request["source"]["catalogs"]) == len(
                plan["request"]["source"]["catalogs"]
            ):
                current_request["source"]["catalogs"] = deepcopy(
                    plan["request"]["source"]["catalogs"]
                )
        compare_request(plan["request"], current_request)
        compare_request(
            plan["candidate_identity_digests"], candidate["identity_digests"]
        )
        compare_request(
            seed["seed_resolution"],
            candidate["documents"]["generation_config"]["seed_resolution"],
        )
        marker_path = confined_path(
            root, plan["outputs"]["record_root"] + "/collection.json"
        )
        if marker_path.exists():
            marker = read_collection_v2(marker_path)
            actual = read_canonical_json(
                marker_path.with_name("collection_intent.json")
            )
            compare_request(plan["intent"], actual)
            if marker["collection_id"] != plan["collection_id"]:
                raise ValueError("ready collection differs from bound generation plan")
            if (
                plan["collection_revision"] is not None
                and marker["collection_revision"] != plan["collection_revision"]
            ):
                raise ValueError(
                    "ready collection revision differs from bound reuse plan"
                )
            validate_installed_generator(plan, root, relocatable=True)
            if reservation["request"]["selection"]["mode"] == "project-generation":
                bind_ready_collection_owned(root, name, marker_path)
            if relocated:
                plan = build_reuse_plan(settings, candidate, marker_path)
                plan_path, plan_sha256 = publish_plan_pointer(settings, plan)
                return settings, plan, plan_path, plan_sha256, True
            publish_plan_pointer(settings, plan)
            return settings, plan, plan_path, seed["plan"]["sha256"], True
        # Only a validated ready collection may be relocated. Uninitialized or
        # partial generation remains tied to the exact original request.
        compare_request(plan["request"], settings["request"])
        record_root = marker_path.parent
        data_root = confined_path(root, plan["outputs"]["data_root"])
        receipts = plan_path.parent / "initializations"
        related = []
        if receipts.exists():
            for receipt_path in receipts.glob("*.json"):
                receipt = read_canonical_json(receipt_path)
                if receipt.get("plan_sha256") == seed["plan"]["sha256"]:
                    related.append(receipt)
        if related or record_root.exists() or data_root.exists():
            if related:
                raise ValueError(
                    "IncompleteWF3: initialized collection is incomplete; partial generation cannot resume. "
                    "Preserve its inputs and use a fresh generation request/project; reservation recovery cannot repair it."
                )
            raise ValueError(
                "PartialWF3Initialization: collection namespace lacks complete initialization evidence; "
                "preserve it for diagnosis. Reservation recovery cannot repair it."
            )
        if plan["decision"] != "create" or reservation["resolution"] is not None:
            raise ValueError(
                "MissingWF3Collection: bound collection is missing; no recreation is authorized"
            )
        publish_plan_pointer(settings, plan)
        return settings, plan, plan_path, seed["plan"]["sha256"], False
    if seed["state"] != "awaiting_sources":
        raise ValueError("reservation does not admit generation")
    plan = select_generation_plan(settings, candidate)
    plan_bytes = canonical_json_bytes(plan)
    sha256 = hashlib.sha256(plan_bytes).hexdigest()
    plan_path = Path(settings["request_path"]).with_name(f"{sha256}.json")
    confined_path(root, plan_path.relative_to(root).as_posix())
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        atomic_record(plan_path, plan)
    except FileExistsError:
        if plan_path.read_bytes() != plan_bytes:
            raise ValueError("generation plan digest collision") from None
    seed = _plan_seed_binding(plan, file_reference(plan_path, "project_root", root))
    bind_reservation_owned(root, name, seed=seed)
    publish_plan_pointer(settings, plan)
    return settings, plan, plan_path, sha256, False
