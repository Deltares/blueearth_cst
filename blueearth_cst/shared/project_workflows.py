"""Read the project file's workflow switches and workflow file paths.

A project file declares its workflows in one of two equivalent forms:

- ``workflows: {<name>: bool}`` with ``workflow_configs: {<name>: path}``;
- the legacy stanza ``workflows: {<name>: {enabled, config_path}}``.

Every reader of a raw (uncomposed) project mapping goes through
``normalize_project_workflows``, which returns the legacy stanza form. Identity,
reservation and snapshot code therefore see exactly the mapping they saw before
the switch/path split; the legacy form stays accepted because retained
reservations recompose captured project bytes.

Stdlib only: raw readers import this without loading the composer.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

WORKFLOW_CONFIGS_KEY = "workflow_configs"


def _form(project_path: object, workflows: Mapping[str, Any], configs: object) -> str:
    """Return ``"legacy"`` or ``"split"``, refusing a mixture of the two."""
    kinds = set()
    for name, value in workflows.items():
        if isinstance(value, bool):
            kinds.add("split")
        elif isinstance(value, Mapping):
            kinds.add("legacy")
        else:
            raise ValueError(
                f"{project_path}: workflows.{name} must be true or false, "
                f"got {type(value).__name__}."
            )
    if configs is not None:
        kinds.add("split")
    if len(kinds) > 1:
        raise ValueError(
            f"{project_path}: use `workflows: {{<name>: true|false}}` with "
            "`workflow_configs: {<name>: <path>}`; do not mix it with "
            "`workflows.<name>.enabled`/`config_path` stanzas."
        )
    return kinds.pop() if kinds else "split"


def legacy_workflow_stanzas(
    project: Mapping[str, Any], project_path: object = "project config"
) -> dict[str, dict[str, Any]]:
    """Return the project's workflows as ``{name: {enabled?, config_path?}}``.

    A workflow named only in ``workflow_configs`` has no ``enabled`` key, and a
    workflow named only in ``workflows`` has no ``config_path``, exactly as an
    equivalent legacy stanza would omit them.
    """
    project_config = project
    workflows = project_config.get("workflows") or {}
    configs = project_config.get("workflow_configs")
    if not isinstance(workflows, Mapping):
        raise ValueError(f"{project_path}: `workflows:` must be a mapping if present.")
    if configs is not None and not isinstance(configs, Mapping):
        raise ValueError(
            f"{project_path}: `{WORKFLOW_CONFIGS_KEY}:` must map workflow names to "
            "file paths."
        )
    if _form(project_path, workflows, configs) == "legacy":
        return {name: dict(stanza) for name, stanza in workflows.items()}
    configs = configs or {}
    for name, path in configs.items():
        if not isinstance(path, str) or not path:
            raise ValueError(
                f"{project_path}: {WORKFLOW_CONFIGS_KEY}.{name} must be a file path."
            )
    stanzas: dict[str, dict[str, Any]] = {}
    for name in [*workflows, *(name for name in configs if name not in workflows)]:
        stanza: dict[str, Any] = {}
        if name in workflows:
            stanza["enabled"] = workflows[name]
        if name in configs:
            stanza["config_path"] = configs[name]
        stanzas[name] = stanza
    return stanzas


def normalize_project_workflows(
    project: Mapping[str, Any], project_path: object = "project config"
) -> dict[str, Any]:
    """Return a shallow copy of ``project`` with legacy workflow stanzas.

    ``workflow_configs`` is removed; ``workflows`` is present in the result
    only when the input declared either key, so normalizing adds no key.
    """
    normalized = dict(project)
    declared = "workflows" in normalized or WORKFLOW_CONFIGS_KEY in normalized
    stanzas = legacy_workflow_stanzas(normalized, project_path)
    normalized.pop(WORKFLOW_CONFIGS_KEY, None)
    if declared:
        normalized["workflows"] = stanzas
    return normalized


def workflow_config_path(project: Mapping[str, Any], name: str) -> str | None:
    """Return the declared (unresolved) settings-file path for ``name``."""
    return legacy_workflow_stanzas(project).get(name, {}).get("config_path")


def set_workflow_config_path(
    project: MutableMapping[str, Any], name: str, path: str
) -> None:
    """Rewrite one workflow's settings-file path in place, keeping the file's form."""
    workflows = project.get("workflows") or {}
    if _form("project config", workflows, project.get("workflow_configs")) == "legacy":
        workflows[name]["config_path"] = path
    else:
        project.setdefault(WORKFLOW_CONFIGS_KEY, {})[name] = path
