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

    def run(self, root: Path, *, preview: bool) -> Path:
        """Validate and publish through the same producer as the standalone CLI."""
        self.recheck()
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


def capture_delta(config_path: Path, extra: list[str]) -> DeltaGeneration:
    """Admit supported flags and capture all producer inputs before other stages."""
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
