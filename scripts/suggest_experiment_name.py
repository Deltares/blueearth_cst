"""Preview experiment selection without reserving a name or editing configuration."""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blueearth_cst.experiment.reservation import preview_experiment  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Print one advisory selection; creation belongs to the execution runners."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="project configuration YAML")
    parser.add_argument(
        "--name", help="override configured family or exact name for this preview"
    )
    parser.add_argument("--new-experiment", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compatibility option; every invocation is read-only",
    )
    args = parser.parse_args(argv)
    try:
        project = yaml.safe_load(args.config.read_bytes())
        from blueearth_cst.shared.config_paths import resolve_project_paths

        project = resolve_project_paths(project, args.config)
        root = Path(project["project"]["project_dir"]).resolve()
        stanza = project["workflows"]["simulate_system"]
        workflow = yaml.safe_load(
            (args.config.resolve().parent / stanza["config_path"]).read_bytes()
        )
        selected = preview_experiment(
            root,
            args.name if args.name is not None else workflow.get("experiment_name"),
            new_experiment=args.new_experiment,
            indicators_only=workflow.get("operation") == "indicators-only"
            or workflow.get("operations") == ["indicators"],
            wf4_enabled=stanza["enabled"],
        )
        if selected.action in {"refuse", "missing"}:
            raise ValueError(selected.reason)
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(selected.name)
    print(
        "Advisory only; use --new-experiment on the execution runner to create another experiment.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
