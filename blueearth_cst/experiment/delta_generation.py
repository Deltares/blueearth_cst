"""Owned runner admission and provenance for independent delta generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from blueearth_cst.experiment.delta_scenarios import generate_collection
from blueearth_cst.shared.config_composition import compose_config
from blueearth_cst.shared.config_paths import resolve_path
from blueearth_cst.shared.provenance import file_sha256
from blueearth_cst.shared.yaml_documents import parse_yaml_document


@dataclass(frozen=True)
class DeltaGeneration:
    """One captured generation operation, independent of WF4 selection."""

    config_path: Path
    settings: dict
    sources: tuple[tuple[Path, bytes], ...]
    inputs: tuple[tuple[Path, str], ...]

    def recheck(self) -> None:
        """Refuse changed authored files or input bytes before publication."""
        for path, data in self.sources:
            if path.read_bytes() != data:
                raise ValueError(f"delta generation configuration changed: {path}")
        for path, digest in self.inputs:
            if file_sha256(path) != digest:
                raise ValueError(f"delta generation input changed: {path}")

    def run(self, root: Path, *, preview: bool) -> Path | None:
        """Validate and publish through the same producer as the standalone CLI.

        A preview of a WF2-sourced request returns None when the WF2 products
        this invocation would create do not exist yet.
        """
        self.recheck()
        if preview and self.settings["schema_version"] == "delta-change/3":
            from blueearth_cst.experiment.delta_wf2 import ensemble_dir

            grids = ensemble_dir(root, self.settings["projections"])
            if not (grids / "analysis/gridded_changes/inventory.json").is_file():
                return None
        manifest = generate_collection(
            self.config_path,
            root,
            settings=self.settings,
            preview=preview,
            expected_input_hashes=[digest for _, digest in self.inputs],
        )
        self.recheck()
        return manifest

    def provenance(self) -> dict:
        """Return the independent operation's effective settings and input evidence."""
        import hashlib

        return {
            "method": "delta_change",
            "settings": self.settings,
            "configuration": [
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "document": data.decode("utf-8"),
                }
                for path, data in self.sources
            ],
            "inputs": [
                {"path": str(path), "sha256": digest} for path, digest in self.inputs
            ],
        }


def capture_delta(
    config_path: Path,
    extra: list[str],
    *,
    project_dir: Path | None = None,
    wf2_ensemble: str | None = None,
) -> DeltaGeneration:
    """Admit supported flags and capture all producer inputs before other stages.

    For a WF2-sourced request (``delta-change/3``), ``wf2_ensemble`` is the
    ensemble WF2 produces in this invocation (None when WF2 is disabled). WF2
    outputs are then captured at publication by the producer's own identity
    checks; with WF2 disabled they must already exist.
    """
    unsupported = set(extra) - {"--dry-run", "--dryrun", "-n"}
    if unsupported:
        raise ValueError(
            f"delta generation does not support forwarded options: {sorted(unsupported)}"
        )
    config_path = config_path.resolve(strict=True)
    project_bytes = config_path.read_bytes()
    project = parse_yaml_document(project_bytes)
    source = Path(
        resolve_path(
            project["workflows"]["generate_scenarios"]["config_path"], config_path
        )
    ).resolve(strict=True)
    source_bytes = source.read_bytes()
    config, _ = compose_config(
        project,
        config_path,
        entry="generate_scenarios",
        source_bytes={source: source_bytes},
    )
    settings = config["workflows"]["generate_scenarios"]["delta_change"]
    paths = [Path(settings["historical_path"])]
    if settings["schema_version"] == "delta-change/3":
        projections = settings["projections"]
        if wf2_ensemble is not None and wf2_ensemble != projections["ensemble"]:
            raise ValueError(
                f"delta projections.ensemble {projections['ensemble']!r} differs from "
                f"the enabled analyze_projections ensemble {wf2_ensemble!r}"
            )
        if wf2_ensemble is None:
            if project_dir is None:
                raise ValueError("WF2-sourced delta generation requires project_dir")
            from blueearth_cst.experiment.delta_wf2 import resolve_members

            # Refuse before any stage runs when the upstream products are absent.
            resolve_members(project_dir, projections)
    else:
        paths.extend(
            Path(member[key])
            for member in settings["members"]
            for key in ("reference_path", "future_path")
        )
    operation = DeltaGeneration(
        config_path,
        settings,
        ((config_path, project_bytes), (source, source_bytes)),
        tuple((path, file_sha256(path)) for path in paths),
    )
    operation.recheck()
    return operation
