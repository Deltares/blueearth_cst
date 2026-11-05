"""One owned experiment admission carried across the public workflow runners."""

import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from blueearth_cst.experiment.collection_resolution import (
    GeneratedCollectionUnavailable,
    resolve_explicit_collection_v2,
)
from blueearth_cst.experiment.reservation import (
    CapturedSource,
    admit_experiment_owned,
    compare_request,
    preview_experiment,
    read_reservation,
)
from blueearth_cst.experiment.reservation_generation import bind_ready_collection_owned
from blueearth_cst.experiment.reservation_request import capture_request
from blueearth_cst.experiment.simulation_runner import (
    resolve_collection_with_plan,
    resolve_selected_collection,
    simulation_settings,
)

NAME_ENV = "CST_EXPERIMENT_NAME"
RESERVATION_ENV = "CST_EXPERIMENT_RESERVATION_ID"


def resolve_experiment_name(root: Path, configured_name: str | None) -> str:
    """Consume owned handoff or perform a read-only advisory lookup."""
    name = os.environ.get(NAME_ENV)
    if name:
        reservation_id = os.environ.get(RESERVATION_ENV)
        if (
            reservation_id
            and read_reservation(root, name)["reservation_id"] != reservation_id
        ):
            raise ValueError("owned experiment handoff reservation differs")
        return name
    selected = preview_experiment(root, configured_name)
    if selected.action in {"refuse", "missing"}:
        raise ValueError(selected.reason)
    return selected.name


@dataclass
class ReservationContext:
    """Retain one selected name and producer applicability while ownership is held."""

    root: Path
    config_path: Path
    repository: Path
    name: str
    reservation_id: str | None
    execute_generation: bool
    execute_model: bool
    request: dict | None
    action: str | None = None

    def compare(self, retained: dict, requested: dict) -> None:
        """Compare requests, allowing only already-bound producer/selector changes."""
        if self.reservation_id is None:
            compare_request(retained, requested)
            return
        reservation = read_reservation(self.root, self.name)
        current = deepcopy(requested)
        resolution = reservation["resolution"]
        if resolution is not None:
            current["selection"] = retained["selection"]
            if not self.execute_generation:
                current["generation"] = retained["generation"]
            if not self.execute_model:
                current["simulation"]["model_request"] = retained["simulation"][
                    "model_request"
                ]
            # Diagnose changed science before looking up its as-yet-unproduced
            # collection. Producer-mode exceptions require a bound resolution.
            compare_request(retained, current)
            from blueearth_cst.experiment.simulation_runs import (
                include_unperturbed,
                selected_run_ids,
            )

            try:
                selection, marker = resolve_selected_collection(
                    self.config_path, self.repository
                )
            except GeneratedCollectionUnavailable:
                if (
                    not self.execute_generation
                    or requested["selection"]["mode"] != "project-generation"
                ):
                    raise
                from blueearth_cst.shared.workflow_config_snapshot import (
                    resolve_file_reference,
                )

                # WF3 owns rebuilding its mutable request pointer. Admission can
                # use the immutable bound collection after the science comparison;
                # WF3 still checks its pinned plan before doing any producer work.
                manifest = resolve_file_reference(
                    resolution["manifest"], {"project_root": self.root}
                )
                selection, marker = resolve_explicit_collection_v2(
                    {"manifest_path": str(manifest)}
                )
            compare_request(
                {
                    key: resolution[key]
                    for key in ("collection_id", "collection_revision", "run_ids")
                },
                {
                    "collection_id": marker["collection_id"],
                    "collection_revision": marker["collection_revision"],
                    "run_ids": selected_run_ids(
                        marker,
                        selection["manifest_path"],
                        include=include_unperturbed(requested["simulation"]),
                    ),
                },
            )
        compare_request(retained, current)

    def recheck(self) -> None:
        """Refuse changed effective inputs before each consumer can write."""
        if self.request is None:
            return
        current, _ = capture_request(
            self.config_path,
            self.repository,
            execute_generation=self.execute_generation,
            execute_model=self.execute_model,
        )
        # Within one invocation no producer-mode or selector change is admitted.
        compare_request(self.request, current)
        reference = (
            self.root / "experiments" / self.name / "_engine/model_reference.yml"
        )
        if reference.exists():
            from blueearth_cst.experiment.check_model_reference import (
                check_model_reference,
            )

            check_model_reference(
                reference, self.root / "models/hydrology/wflow", self.name
            )

    def bind_selection(self) -> None:
        """Pin exact validated ready collection and producer-owned run IDs."""
        if self.reservation_id is None or self.request is None:
            return
        self.recheck()
        selection, _, plan_reference = resolve_collection_with_plan(
            self.config_path, self.repository
        )
        reservation = read_reservation(self.root, self.name)
        mode = (reservation["resolution"] or {}).get(
            "resolution_mode", selection["resolution_mode"]
        )
        bind_ready_collection_owned(
            self.root,
            self.name,
            selection["manifest_path"],
            resolution_mode=mode,
            generation_plan=plan_reference
            if reservation["seed"]["state"] == "awaiting_sources"
            else None,
        )

    def environment(self) -> dict[str, str]:
        """Carry the single admission to Snakemake without editing user YAML."""
        result = {NAME_ENV: self.name}
        if self.reservation_id is not None:
            result[RESERVATION_ENV] = self.reservation_id
        return result


def admit_context_owned(
    config_path: Path,
    repository: Path,
    root: Path,
    invocation_id: str,
    *,
    execute_generation: bool = False,
    execute_model: bool = False,
    new_experiment: bool = False,
    note: str | None = None,
    recover: bool = False,
    preview: bool = False,
    announce: bool = True,
) -> ReservationContext:
    """Select before any producer; callers already hold project-execution ownership."""
    _, settings = simulation_settings(config_path)
    indicators = settings["operation"] == "indicators-only"
    selected = preview_experiment(
        root,
        settings.get("experiment_name"),
        new_experiment=new_experiment,
        note=note,
        indicators_only=indicators,
        recover=recover,
    )
    if selected.action in {"refuse", "missing"}:
        raise ValueError(selected.reason)
    request, captures = (
        (None, [])
        if indicators
        else capture_request(
            config_path,
            repository,
            execute_generation=execute_generation,
            execute_model=execute_model,
        )
    )
    if not indicators and not execute_generation:
        # Standalone simulation cannot reserve an unresolved generation request.
        resolve_selected_collection(config_path, repository)
    context = ReservationContext(
        Path(root).resolve(),
        Path(config_path).resolve(),
        Path(repository).resolve(),
        selected.name,
        selected.reservation_id,
        execute_generation,
        execute_model,
        request,
    )
    if not preview:
        selected = admit_experiment_owned(
            root,
            settings.get("experiment_name"),
            request=request,
            sources=[CapturedSource(role, path, data) for role, path, data in captures],
            creator_invocation_id=invocation_id,
            new_experiment=new_experiment,
            note=note,
            indicators_only=indicators,
            recover=recover,
            request_comparator=context.compare,
        )
        context.name = selected.name
        context.reservation_id = selected.reservation_id
    elif selected.action == "continue" and selected.reservation_id and not indicators:
        # Indicators-only never captures a request; admission compares none either.
        context.compare(read_reservation(root, selected.name)["request"], request)
    recipe = (
        context.root / "experiments" / context.name / "_engine/simulation-freeze.json"
    )
    if not preview and not indicators and recipe.exists() and context.reservation_id:
        from blueearth_cst.experiment.reservation_freeze import finish_freeze_owned

        finish_freeze_owned(recipe.parent.parent, recover=recover)
    context.action = selected.action
    if announce:
        print(
            f"Selected experiment: {context.name}; action: {context.action}; project: {context.root}",
            flush=True,
        )
    context.recheck()
    return context
