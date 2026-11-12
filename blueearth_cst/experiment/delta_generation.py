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
    #: delta-change/3: ``{"path", "source"}`` of the historical record.
    historical: dict | None = None
    #: The derived record is produced by WF0/WF1 later in this invocation.
    historical_pending: bool = False

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
        expected = [digest for _, digest in self.inputs]
        if self.settings["schema_version"] == "delta-change/3":
            from blueearth_cst.experiment.delta_wf2 import ensemble_dir

            grids = ensemble_dir(root, self.settings["projections"])
            ready = (grids / "analysis/gridded_changes/inventory.json").is_file()
            if self.historical_pending:
                path = Path(self.historical["path"])
                if not path.is_file():
                    if preview:
                        return None
                    raise ValueError(
                        f"historical record was not produced by this invocation: {path}"
                    )
                expected = [file_sha256(path)]
            if preview and not ready:
                return None
            if not preview:
                self._snapshot_outside_inputs(root)
        manifest = generate_collection(
            self.config_path,
            root,
            settings=self.settings,
            preview=preview,
            expected_input_hashes=expected,
            historical=self.historical,
            documents=tuple((path.name, data) for path, data in self.sources),
        )
        self.recheck()
        return manifest

    def index(self, root: Path, manifest: Path) -> None:
        """Rebuild ``collections.csv``; a configured name differing from the reused
        set's folder becomes an alias (design §5.3). Call under the project lock."""
        from blueearth_cst.experiment.collection_index import rebuild_collection_index
        from blueearth_cst.experiment.content_identity import read_canonical_json

        name = self.settings.get("name")
        identity = read_canonical_json(manifest)["collection_id"]
        rebuild_collection_index(root, alias=(identity, name) if name else None)

    def _snapshot_outside_inputs(self, root: Path) -> None:
        """Content-addressed copies of admitted inputs outside the project."""
        from blueearth_cst.experiment.generation_plan import snapshot_generation_input

        root = Path(root).resolve()
        for path, digest in self.inputs:
            if not path.resolve().is_relative_to(root):
                _, copied, _ = snapshot_generation_input(root, path)
                if copied != digest:
                    raise ValueError(f"delta input changed during snapshot: {path}")

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


def historical_binding(config: dict, settings: dict) -> dict:
    """The historical record of a delta-change/3 request (design §3.1).

    Default: the case's climate-store extract, derived with the stochastic rule;
    ``historical_override.path`` replaces it with a declared supported source.
    """
    override = settings.get("historical_override", {})
    if "path" in override:
        return {
            "path": Path(override["path"]),
            "source": override["source"],
            "recipe": None,
        }
    from blueearth_cst.shared.forcing_recipe import normalize_forcing_recipe
    from blueearth_cst.shared.wf3_science import (
        DEFAULT_BASIN_INDEX,
        DEFAULT_HYDROGRAPHY,
        climate_store_rule,
    )

    project, climate, basin = config["project"], config["climate"], config["basin"]
    sources = basin.get("sources") or {}
    recipe = normalize_forcing_recipe(climate["selected"], climate.get("forcing"))
    store = climate_store_rule(
        project_dir=project["project_dir"],
        model_region=basin["region"],
        data_sources=project["catalog"],
        hydrography=sources.get("hydrography", DEFAULT_HYDROGRAPHY),
        basin_index=sources.get("basin_index", DEFAULT_BASIN_INDEX),
        clim_source=climate["selected"],
        historical_window=climate["window"],
        forcing=recipe.settings(),
    )
    return {
        "path": Path(store.store_dir) / "extract_historical.nc",
        "source": climate["selected"],
        # The named recipe WF4 must match (None without named forcings).
        "recipe": climate.get("forcing_id", climate.get("default_forcing")),
    }


def capture_delta(
    config_path: Path,
    extra: list[str],
    *,
    project_dir: Path | None = None,
    wf2_ensemble: str | None = None,
    historical_producer: bool = False,
) -> DeltaGeneration:
    """Admit supported flags and capture all producer inputs before other stages.

    For a WF2-sourced request (``delta-change/3``), ``wf2_ensemble`` is the
    ensemble WF2 produces in this invocation (None when WF2 is disabled), and
    ``historical_producer`` says WF0 or WF1 runs first and may create the
    derived historical record. Products a stage creates later are admitted as
    pending and captured at publication; otherwise they must already exist.
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
    historical, pending = None, False
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
            from blueearth_cst.experiment.delta_wf2 import resolve_windows

            # Refuse before any stage runs when the upstream products are absent.
            resolve_windows(project_dir, projections)
        historical = historical_binding(config, settings)
        derived = "path" not in settings.get("historical_override", {})
        if not historical["path"].is_file():
            if not (derived and historical_producer):
                raise ValueError(
                    f"historical record missing: {historical['path']}. "
                    + (
                        "Enable analyze_climate or build_model to extract it."
                        if derived
                        else "Check historical_override.path."
                    )
                )
            pending = True
        paths = [] if pending else [historical["path"]]
    else:
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
        historical,
        pending,
    )
    operation.recheck()
    return operation
