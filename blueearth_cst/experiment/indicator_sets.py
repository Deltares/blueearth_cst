"""Named WF4 indicator sets: operation selection, closed set files and resolution.

A set file selects trusted calculations; it never executes Python. Existing
definitions and authored aliases remain supported. Local calculations resolve
through a selected trusted set, without requiring a legacy equivalent.
Titles, descriptions and authored citations are presentation only.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from blueearth_cst.experiment import indicator_registry as registry

OPERATIONS = ("simulations", "indicators")
DEFAULT_SET = "default"
DEFAULT_PRESET = "current_default"
SET_SCHEMA_VERSION = 1
SET_ID_PATTERN = re.compile(r"[a-z][a-z0-9_-]*")
SET_ID_MAX_LENGTH = 64
_WINDOWS_RESERVED = {"con", "prn", "aux", "nul"} | {
    f"{stem}{n}" for stem in ("com", "lpt") for n in range(1, 10)
}
#: Toolbox-shipped sets, anchored to the installed repository, never the cwd.
BUNDLED_ROOT = Path(__file__).resolve().parents[2] / "config" / "indicator_sets"
SELECTION_KEYS = (
    "operations",
    "indicator_sets",
    "indicator_set_definitions",
    "indicator_selection",
    "indicator_evaluation",
)
SELECTION_MODES = ("include", "exclude")
#: ``strict`` refuses a selected indicator lacking an input; ``partial`` publishes
#: it as unavailable. Evaluation never enters numerical identity.
EVALUATION_MODES = ("strict", "partial")


class IndicatorSetError(ValueError):
    """A named indicator-set selection or set file is invalid."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses duplicate mapping keys."""


def _unique_mapping(loader, node, deep=False):
    keys = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in keys:
            raise IndicatorSetError(
                f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}"
            )
        keys.add(key)
    return loader.construct_mapping(node, deep=deep)


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def load_unique_yaml(text: str):
    """Parse YAML safely, refusing duplicate keys and arbitrary objects."""
    return yaml.load(text, Loader=_UniqueKeyLoader)  # noqa: S506 - SafeLoader subclass


def validate_id(value, label: str) -> str:
    """Return a safe set or authored indicator id, or refuse it."""
    if (
        not isinstance(value, str)
        or len(value) > SET_ID_MAX_LENGTH
        or not SET_ID_PATTERN.fullmatch(value)
        or value in _WINDOWS_RESERVED
    ):
        raise IndicatorSetError(
            f"{label} {value!r} must match [a-z][a-z0-9_-]*, be at most "
            f"{SET_ID_MAX_LENGTH} characters and not be a Windows reserved name"
        )
    return value


def _string_list(value, label: str, *, allow_empty: bool) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise IndicatorSetError(
            f"{label} must be a {'list' if allow_empty else 'nonempty list'}"
        )
    if any(not isinstance(item, str) for item in value):
        raise IndicatorSetError(f"{label} entries must be strings")
    if len(set(value)) != len(value):
        raise IndicatorSetError(f"{label} has duplicate entries")
    return list(value)


@dataclass(frozen=True)
class SetSource:
    """A selected set's resolved YAML source, read once per invocation."""

    set_id: str
    path: Path
    bundled: bool
    text: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Selection:
    """Normalized WF4 operation and indicator-set selection."""

    operations: tuple[str, ...]
    sets: tuple[SetSource, ...]
    legacy: bool
    #: Per-set ``(set_id, mode, indicator keys)``; mode is include or exclude.
    choices: tuple[tuple[str, str, tuple[str, ...]], ...] = ()
    evaluation: str = "strict"

    @property
    def partial(self) -> bool:
        return self.evaluation == "partial"

    @property
    def simulations(self) -> bool:
        return "simulations" in self.operations

    @property
    def indicators(self) -> bool:
        return "indicators" in self.operations


def uses_named_selection(settings: dict) -> bool:
    """Whether a WF4 config uses the named-set interface rather than ``operation``."""
    return any(key in settings for key in SELECTION_KEYS)


def parse_selection(settings: dict, workflow_path) -> Selection:
    """Validate the named-set interface and capture selected set sources once.

    Relative custom ``config_path`` values resolve against the WF4 YAML that
    declares them; bundled names resolve against :data:`BUNDLED_ROOT`. With
    simulations alone, structure is validated but no set file is opened.
    """
    for key in ("operation", "indicators"):
        if key in settings:
            raise IndicatorSetError(
                f"simulate_system.{key} is the legacy interface and cannot be mixed "
                "with operations/indicator_sets; replace operation with "
                "operations: [simulations, indicators] and indicators with an explicit "
                "indicator set"
            )
    operations = settings.get("operations", list(OPERATIONS))
    operations = _string_list(operations, "operations", allow_empty=False)
    unknown = sorted(set(operations) - set(OPERATIONS))
    if unknown:
        raise IndicatorSetError(
            f"operations has unknown entries {unknown}; allowed: {list(OPERATIONS)}"
        )
    indicators = "indicators" in operations
    entries = settings.get("indicator_sets", [DEFAULT_SET])
    if not isinstance(entries, list) or (indicators and not entries):
        raise IndicatorSetError("indicator_sets must be a nonempty list")
    inline = {}
    names = []
    for entry in entries:
        if isinstance(entry, dict):
            document = _flat_document(entry)
            name = document["id"]
            inline[name] = document
        else:
            name = entry
        validate_id(name, "indicator set")
        names.append(name)
    if len(set(names)) != len(names):
        raise IndicatorSetError("indicator_sets has duplicate entries")
    for name in names:
        validate_id(name, "indicator set")
    definitions = settings.get("indicator_set_definitions", {})
    if not isinstance(definitions, dict):
        raise IndicatorSetError("indicator_set_definitions must be a mapping")
    for name, entry in definitions.items():
        validate_id(name, "indicator_set_definitions key")
        if name == DEFAULT_SET:
            raise IndicatorSetError(
                f"custom indicator set {name!r} collides with the default preset"
            )
        if (
            not isinstance(entry, dict)
            or set(entry) != {"config_path"}
            or not isinstance(entry["config_path"], str)
            or not entry["config_path"]
        ):
            raise IndicatorSetError(
                f"indicator_set_definitions.{name} must be exactly {{config_path: <nonempty string>}}"
            )
    if set(inline) & set(definitions):
        raise IndicatorSetError(
            "inline sets cannot also have indicator_set_definitions"
        )
    choices = _parse_choices(settings.get("indicator_selection", {}), names)
    evaluation = settings.get("indicator_evaluation", "strict")
    if evaluation not in EVALUATION_MODES:
        raise IndicatorSetError(
            f"indicator_evaluation must be one of {list(EVALUATION_MODES)}; "
            f"got {evaluation!r}"
        )
    ordered = tuple(item for item in OPERATIONS if item in operations)
    if not indicators:
        return Selection(ordered, (), legacy=False, evaluation=evaluation)
    base = Path(workflow_path).resolve().parent
    sources = []
    for name in names:
        if name in inline:
            document = inline[name]
            # The unmodified default keeps its response-driven preset semantics.
            if document == _flat_document({"set": DEFAULT_SET}):
                path = BUNDLED_ROOT / "default.yml"
                sources.append(
                    SetSource(name, path, True, path.read_text(encoding="utf-8"))
                )
            else:
                sources.append(
                    SetSource(
                        name,
                        Path(workflow_path).resolve(),
                        False,
                        yaml.safe_dump(document),
                    )
                )
            continue
        if name in definitions:
            path = Path(definitions[name]["config_path"])
            path = (path if path.is_absolute() else base / path).resolve()
            bundled = False
        else:
            path = BUNDLED_ROOT / f"{name}.yml"
            bundled = True
            if not path.is_file():
                known = sorted(p.stem for p in BUNDLED_ROOT.glob("*.yml"))
                raise IndicatorSetError(
                    f"unknown indicator set {name!r}; discoverable sets: {known}; "
                    "custom sets require indicator_set_definitions"
                )
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise IndicatorSetError(
                f"indicator set {name!r}: cannot read {path}: {exc}"
            ) from exc
        sources.append(SetSource(name, path, bundled, text))
    return Selection(
        ordered, tuple(sources), legacy=False, choices=choices, evaluation=evaluation
    )


def _flat_document(entry: dict) -> dict:
    """Normalize an inline or standalone registry declaration without reading data."""
    owner = validate_id(entry.get("set"), "registry set")
    name = validate_id(entry.get("id", owner), "indicator set")
    version = entry.get("schema_version", SET_SCHEMA_VERSION)
    if type(version) is not int or version != SET_SCHEMA_VERSION:
        raise IndicatorSetError(f"schema_version must be {SET_SCHEMA_VERSION}")
    metadata = {"set", "id", "title", "description", "schema_version"}
    return {
        "schema_version": version,
        "id": name,
        "title": entry.get("title", name),
        "description": entry.get("description", ""),
        "selection": {
            "registry": {
                "set": owner,
                "overrides": {
                    key: value for key, value in entry.items() if key not in metadata
                },
            }
        },
    }


def _parse_choices(value, names) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """Validate ``indicator_selection``: one include or exclude list per selected set.

    Indicator keys are checked against the resolved set by :func:`apply_choice`.
    """
    if not isinstance(value, dict):
        raise IndicatorSetError("indicator_selection must be a mapping")
    choices = []
    for set_id, entry in value.items():
        validate_id(set_id, "indicator_selection key")
        if set_id not in names:
            raise IndicatorSetError(
                f"indicator_selection.{set_id} names a set absent from "
                f"indicator_sets {list(names)}"
            )
        if (
            not isinstance(entry, dict)
            or len(entry) != 1
            or not (set(entry) <= set(SELECTION_MODES))
        ):
            raise IndicatorSetError(
                f"indicator_selection.{set_id} must be exactly one of "
                "{include: [...]} or {exclude: [...]}"
            )
        mode = "include" if "include" in entry else "exclude"
        keys = entry.get("include", entry.get("exclude"))
        keys = _string_list(
            keys, f"indicator_selection.{set_id}.{mode}", allow_empty=False
        )
        choices.append((set_id, mode, tuple(keys)))
    return tuple(choices)


def apply_choice(resolved: ResolvedSet, choices) -> ResolvedSet:
    """Keep the indicators an include/exclude choice selects, in set order."""
    choice = next((item for item in choices if item[0] == resolved.set_id), None)
    if choice is None:
        return resolved
    _, mode, keys = choice
    known = [item.indicator for item in resolved.indicators]
    unknown = sorted(set(keys) - set(known))
    if unknown:
        raise IndicatorSetError(
            f"indicator_selection.{resolved.set_id}.{mode} names indicators absent "
            f"from the set: {unknown}; available: {known}"
        )
    kept = tuple(
        item
        for item in resolved.indicators
        if (item.indicator in keys) == (mode == "include")
    )
    if not kept:
        raise IndicatorSetError(
            f"indicator_selection.{resolved.set_id} leaves no indicators; "
            "remove the set from indicator_sets instead"
        )
    return replace(resolved, indicators=kept)


def legacy_selection(settings: dict) -> Selection:
    """Map the scalar ``operation`` interface onto operations and the default set."""
    operation = settings.get("operation")
    if operation == "simulate-and-indicators":
        operations = ("simulations", "indicators")
    elif operation in ("indicators-only", None):
        # Absent only in indicators-only projections that never read it.
        operations = ("indicators",)
    else:
        raise IndicatorSetError(f"unsupported legacy operation {operation!r}")
    path = BUNDLED_ROOT / f"{DEFAULT_SET}.yml"
    return Selection(
        operations,
        (SetSource(DEFAULT_SET, path, True, path.read_text(encoding="utf-8")),),
        legacy=True,
    )


@dataclass(frozen=True)
class ResolvedIndicator:
    """One selected indicator: numerical declaration plus presentation."""

    declaration: registry.IndicatorDeclaration
    definition: str
    method: str
    parameters: dict
    title: str
    description: str
    references: tuple[str, ...]
    authored_references: tuple[str, ...]
    equivalent_definition: str
    calculation_set: str | None = None
    pending_baseline: bool = False

    @property
    def indicator(self) -> str:
        return self.declaration.name


@dataclass(frozen=True)
class ResolvedSet:
    """A validated set: identity-relevant declarations and publication metadata."""

    set_id: str
    title: str
    description: str
    source: SetSource
    indicators: tuple[ResolvedIndicator, ...]

    @property
    def declarations(self) -> tuple[registry.IndicatorDeclaration, ...]:
        return tuple(item.declaration for item in self.indicators)


def _registered(definition: str) -> ResolvedIndicator:
    method, parameters, _ = registry.DEFINITIONS[definition]
    title, description = registry.DEFINITION_TEXT[definition]
    return ResolvedIndicator(
        declaration=registry.declaration_for(definition),
        definition=definition,
        method=method,
        parameters=dict(sorted(parameters.items())),
        title=title,
        description=description,
        references=registry.DEFINITION_REFERENCES.get(definition, ()),
        authored_references=(),
        equivalent_definition="",
    )


def _authored(entry: dict, label: str, set_id: str = DEFAULT_SET) -> ResolvedIndicator:
    expected = {"id", "title", "description", "references", "calculation"}
    if set(entry) != expected:
        raise IndicatorSetError(
            f"{label}: an authored entry has exactly {sorted(expected)}; "
            f"got {sorted(entry)}"
        )
    indicator = validate_id(entry["id"], f"{label} id")
    if indicator in registry.DEFINITIONS:
        raise IndicatorSetError(
            f"{label}: authored id {indicator!r} equals a registered definition; "
            "select it with {definition: ...} instead"
        )
    for field in ("title", "description"):
        if not isinstance(entry[field], str):
            raise IndicatorSetError(f"{label}: {field} must be a string")
    references = entry["references"]
    if not isinstance(references, list) or any(
        not isinstance(item, str) or not item for item in references
    ):
        raise IndicatorSetError(
            f"{label}: references must be a list of nonempty strings"
        )
    calculation = entry["calculation"]
    if not isinstance(calculation, dict) or set(calculation) not in (
        {
            "method",
            "parameters",
        },
        {"set", "method", "parameters"},
    ):
        raise IndicatorSetError(
            f"{label}/{indicator}: calculation requires method, parameters; optional set"
        )
    if "set" in calculation:
        from blueearth_cst.indicators import loader

        owner = calculation["set"]
        try:
            declaration = loader.resolve(
                owner,
                calculation["method"],
                calculation["parameters"],
                indicator_name=indicator,
            )
        except ValueError as exc:
            raise IndicatorSetError(f"{set_id}/{indicator}: {exc}") from exc
        return ResolvedIndicator(
            declaration=declaration,
            definition=f"{owner}:{indicator}",
            method=calculation["method"],
            parameters=dict(calculation["parameters"]),
            title=entry["title"],
            description=entry["description"],
            references=(),
            authored_references=tuple(references),
            equivalent_definition="",
            calculation_set=owner,
        )
    try:
        definition, parameters = registry.validate_calculation(
            calculation["method"], calculation["parameters"]
        )
    except ValueError as exc:
        raise IndicatorSetError(f"{set_id}/{indicator} ({label}): {exc}") from exc
    return ResolvedIndicator(
        declaration=replace(registry.declaration_for(definition), name=indicator),
        definition=definition,
        method=calculation["method"],
        parameters=parameters,
        title=entry["title"],
        description=entry["description"],
        references=registry.DEFINITION_REFERENCES.get(definition, ()),
        authored_references=tuple(references),
        equivalent_definition=definition,
    )


def resolve_set(
    source: SetSource,
    default_tokens=None,
    *,
    legacy: bool = False,
    prepare_baseline: bool = True,
    model_root: Path | None = None,
) -> ResolvedSet:
    """Validate one captured set file and resolve its entries.

    ``default_tokens`` are the response variables the ``current_default``
    preset expands; they are required only by a set using the preset. Legacy
    mode keeps the old ``indicators: []`` behaviour of an empty selection.
    """
    label = f"indicator set {source.set_id!r} ({source.path.as_posix()})"
    try:
        document = load_unique_yaml(source.text)
    except yaml.YAMLError as exc:
        raise IndicatorSetError(f"{label}: invalid YAML: {exc}") from exc
    if isinstance(document, dict) and "set" in document:
        document = _flat_document(document)
    expected = {"schema_version", "id", "title", "description", "selection"}
    if not isinstance(document, dict) or set(document) != expected:
        raise IndicatorSetError(
            f"{label}: a set file has exactly {sorted(expected)}"
            + ("" if not isinstance(document, dict) else f"; got {sorted(document)}")
        )
    if (
        document["schema_version"] != SET_SCHEMA_VERSION
        or type(document["schema_version"]) is not int
    ):
        raise IndicatorSetError(f"{label}: schema_version must be {SET_SCHEMA_VERSION}")
    if document["id"] != source.set_id:
        raise IndicatorSetError(
            f"{label}: id {document['id']!r} differs from its selected name"
        )
    if not isinstance(document["title"], str) or not document["title"]:
        raise IndicatorSetError(f"{label}: title must be a nonempty string")
    if not isinstance(document["description"], str):
        raise IndicatorSetError(f"{label}: description must be a string")
    selection = document["selection"]
    if not isinstance(selection, dict) or len(selection) != 1:
        raise IndicatorSetError(
            f"{label}: selection is exactly one of preset, registry or indicators"
        )
    ((kind, value),) = selection.items()
    if kind == "preset":
        if value != DEFAULT_PRESET or not (
            source.bundled and source.set_id == DEFAULT_SET
        ):
            raise IndicatorSetError(
                f"{label}: only the bundled default set may use preset {DEFAULT_PRESET}"
            )
        if default_tokens is None or (not default_tokens and not legacy):
            raise IndicatorSetError(
                f"{label}: the default preset expanded to no response variables"
            )
        indicators = [
            _registered(item.name) for item in registry.declarations(default_tokens)
        ]
    elif kind == "registry":
        from blueearth_cst.indicators import loader

        overrides = {}
        if isinstance(value, dict):
            if set(value) != {"set", "overrides"} or not isinstance(
                value["overrides"], dict
            ):
                raise IndicatorSetError(
                    f"{label}: registry mapping needs set and overrides"
                )
            overrides = value["overrides"]
            value = value["set"]
        try:
            local = loader.declarations(value)
            module = loader.load_set(value)
        except ValueError as exc:
            raise IndicatorSetError(f"{label}: {exc}") from exc
        unknown = set(overrides) - {item.name for item in local}
        if unknown:
            raise IndicatorSetError(
                f"{label}: registry overrides name unknown indicators {sorted(unknown)}"
            )
        pending = set()
        configured = []
        for item in local:
            patch = overrides.get(item.name, {})
            if not isinstance(patch, dict):
                raise IndicatorSetError(
                    f"{label}/{item.name}: parameters must be a mapping"
                )
            enabled = patch.get("enabled", True)
            if type(enabled) is not bool:
                raise IndicatorSetError(
                    f"{label}/{item.name}: enabled must be a boolean"
                )
            parameters = {
                **item.calculation_parameters,
                **{k: v for k, v in patch.items() if k != "enabled"},
            }
            thresholds = parameters.get("thresholds")
            baseline = (
                value == "crop-water"
                and item.name == "summer_low_flow_days"
                and isinstance(thresholds, dict)
                and "baseline_quantile" in thresholds
            )
            if baseline:
                from blueearth_cst.experiment.baseline_qcrit import resolve_thresholds

                parameters = resolve_thresholds(
                    parameters, source.path, prepare=False, model_root=model_root
                )
            # Validate disabled entries too, without preparing their historical data.
            declaration = loader.resolve(
                value, item.calculation_method, parameters, indicator_name=item.name
            )
            if not enabled:
                continue
            if baseline and prepare_baseline:
                parameters = resolve_thresholds(
                    {**parameters, "thresholds": thresholds},
                    source.path,
                    model_root=model_root,
                )
                declaration = loader.resolve(
                    value, item.calculation_method, parameters, indicator_name=item.name
                )
            elif baseline:
                pending.add(item.name)
            configured.append(declaration)
        local = tuple(configured)
        if value == DEFAULT_SET and default_tokens is not None:
            preset_names = {item.name for item in registry.declarations(default_tokens)}
            local = tuple(item for item in local if item.name in preset_names)
        if not local:
            raise IndicatorSetError(
                f"{label}: no enabled indicators; remove the set from indicator_sets instead"
            )
        descriptions = {item.id: item.description for item in module.INDICATORS}
        indicators = [
            ResolvedIndicator(
                declaration=item,
                definition=f"{value}:{item.name}",
                method=item.calculation_method,
                parameters=dict(item.calculation_parameters),
                title=item.name.replace("_", " "),
                description=descriptions[item.name],
                references=(),
                authored_references=(),
                equivalent_definition="",
                calculation_set=value,
                pending_baseline=item.name in pending,
            )
            for item in local
        ]
    elif kind == "indicators":
        if not isinstance(value, list) or not value:
            raise IndicatorSetError(f"{label}: indicators must be a nonempty list")
        indicators = []
        for number, entry in enumerate(value, 1):
            entry_label = f"{label} entry {number}"
            if not isinstance(entry, dict):
                raise IndicatorSetError(f"{entry_label}: must be a mapping")
            if "definition" in entry:
                if set(entry) != {"definition"}:
                    raise IndicatorSetError(
                        f"{entry_label}: a definition selector has no overrides; got {sorted(entry)}"
                    )
                if entry["definition"] not in registry.DEFINITIONS:
                    raise IndicatorSetError(
                        f"{entry_label}: unknown registered definition {entry['definition']!r}"
                    )
                indicators.append(_registered(entry["definition"]))
            else:
                indicators.append(_authored(entry, entry_label, source.set_id))
    else:
        raise IndicatorSetError(f"{label}: unknown selection kind {kind!r}")
    by_name, contracts = {}, {}
    for item in indicators:
        if item.indicator in by_name:
            raise IndicatorSetError(f"{label}: repeats indicator {item.indicator!r}")
        by_name[item.indicator] = item
        if item.definition in contracts:
            raise IndicatorSetError(
                f"{label}: {contracts[item.definition]!r} and {item.indicator!r} share one "
                f"calculation contract ({item.definition})"
            )
        contracts[item.definition] = item.indicator
    return ResolvedSet(
        source.set_id,
        document["title"],
        document["description"],
        source,
        tuple(indicators),
    )


def default_tokens(
    project: dict, *, simulation_path: bool, retained_variables
) -> list[str]:
    """Response variables the new-mode ``current_default`` preset expands.

    ``model.outvars`` when it is a nonempty list; absent, the toolbox defaults on
    a simulation path or the retained response variables on indicator-only.
    Explicit null or an empty list is refused rather than normalized.
    ``retained_variables`` is a callable, read only when consulted.
    """
    from blueearth_cst.shared.indicator_tables import indicator_tables
    from blueearth_cst.shared.snake_utils import DEFAULT_WFLOW_OUTVARS

    model = project.get("model") or {}
    if "outvars" in model:
        outvars = model["outvars"]
        if not isinstance(outvars, list) or not outvars:
            raise IndicatorSetError(
                "the default indicator set needs model.outvars as a nonempty list "
                f"(got {outvars!r}); remove the key to use the defaults"
            )
        return list(indicator_tables(outvars))
    if simulation_path:
        return list(indicator_tables(DEFAULT_WFLOW_OUTVARS))
    return list(retained_variables())


def legacy_tokens(
    project: dict, settings: dict, context: str, retained_variables
) -> list[str]:
    """Legacy ``indicators``/``outvars`` precedence, preserved per observed branch.

    The three pre-named-set branches disagree on null/absent values; each
    expression is kept verbatim rather than normalized (design v3, P3/O5).
    ``context`` is ``simulation_rule``, ``indicators_only`` or ``export``.
    """
    from blueearth_cst.shared.indicator_tables import indicator_tables
    from blueearth_cst.shared.snake_utils import DEFAULT_WFLOW_OUTVARS

    model = project.get("model") or {}
    if context == "simulation_rule":
        return list(
            settings.get(
                "indicators",
                indicator_tables(model.get("outvars", DEFAULT_WFLOW_OUTVARS)),
            )
        )
    tokens = settings.get("indicators")
    if tokens is not None:
        return list(tokens)
    outvars = model.get("outvars")
    if context == "indicators_only":
        if outvars is not None:
            return list(indicator_tables(outvars))
        return list(retained_variables())
    if context == "export":
        if outvars is not None or settings["operation"] == "simulate-and-indicators":
            return list(
                indicator_tables(DEFAULT_WFLOW_OUTVARS if outvars is None else outvars)
            )
        return list(retained_variables())
    raise ValueError(f"unknown legacy token context {context!r}")


#: Environment variable naming the runner's captured selection snapshot, so
#: every Snakemake parse/job process uses the bytes admitted at start.
SNAPSHOT_ENV = "CST_INDICATOR_SELECTION"


def write_selection_snapshot(selection: Selection, path: Path) -> Path:
    """Persist the admitted set sources (paths and exact text) for this invocation."""
    import json

    payload = {
        "schema_version": "indicator-selection/2",
        "operations": list(selection.operations),
        "choices": [
            {"set_id": set_id, "mode": mode, "indicators": list(keys)}
            for set_id, mode, keys in selection.choices
        ],
        "evaluation": selection.evaluation,
        "legacy": selection.legacy,
        "sets": [
            {
                "set_id": item.set_id,
                "path": item.path.as_posix(),
                "bundled": item.bundled,
                "text": item.text,
            }
            for item in selection.sets
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    return path


def read_selection_snapshot(path) -> Selection:
    """Read a snapshot written by :func:`write_selection_snapshot`."""
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "indicator-selection/2":
        raise IndicatorSetError(f"unsupported indicator selection snapshot {path}")
    return Selection(
        tuple(payload["operations"]),
        tuple(
            SetSource(item["set_id"], Path(item["path"]), item["bundled"], item["text"])
            for item in payload["sets"]
        ),
        legacy=payload["legacy"],
        choices=tuple(
            (item["set_id"], item["mode"], tuple(item["indicators"]))
            for item in payload["choices"]
        ),
        evaluation=payload["evaluation"],
    )


def workflow_selection(settings: dict, workflow_path) -> Selection:
    """The invocation's selection: the runner snapshot when present, else live."""
    import os

    snapshot = os.environ.get(SNAPSHOT_ENV)
    if snapshot:
        return read_selection_snapshot(snapshot)
    if uses_named_selection(settings):
        return parse_selection(settings, workflow_path)
    return legacy_selection(settings)


def selected_sets(
    project: dict,
    settings: dict,
    selection: Selection,
    *,
    simulation_path: bool,
    retained_variables,
    legacy_context: str,
    prepare_baseline: bool = True,
) -> tuple[ResolvedSet, ...]:
    """Resolve every selected set, expanding the default preset by mode.

    An ``indicator_selection`` choice then narrows each set before any caller
    checks requirements, so deselected indicators are never requested.

    ``retained_variables`` is a callable; ``legacy_context`` names the legacy
    branch (see :func:`legacy_tokens`) whose precedence applies.
    """
    from blueearth_cst.experiment.baseline_qcrit import HistoricalQcritUnavailable

    resolved = []
    unavailable = []
    for source in selection.sets:
        tokens = None
        try:
            document = load_unique_yaml(source.text)
        except yaml.YAMLError as exc:
            raise IndicatorSetError(
                f"indicator set {source.set_id!r}: invalid YAML: {exc}"
            ) from exc
        authored_selection = (
            document.get("selection") if isinstance(document, dict) else None
        )
        registry_selection = (
            authored_selection.get("registry")
            if isinstance(authored_selection, dict)
            else None
        )
        owner = (
            registry_selection.get("set")
            if isinstance(registry_selection, dict)
            else registry_selection
        )
        if (
            (source.bundled and source.set_id == DEFAULT_SET)
            or owner == DEFAULT_SET
            or (isinstance(document, dict) and document.get("set") == DEFAULT_SET)
        ):
            tokens = (
                legacy_tokens(project, settings, legacy_context, retained_variables)
                if selection.legacy
                else default_tokens(
                    project,
                    simulation_path=simulation_path,
                    retained_variables=retained_variables,
                )
            )
        try:
            item = resolve_set(
                source,
                tokens,
                legacy=selection.legacy,
                prepare_baseline=prepare_baseline,
                model_root=(
                    Path(project["project"]["project_dir"]).resolve()
                    / "models/hydrology/wflow"
                )
                if (project.get("project") or {}).get("project_dir")
                else None,
            )
        except HistoricalQcritUnavailable as exc:
            # Inspect the remaining sets: a pending producer must not hide
            # malformed rules or corrupt existing historical data in another set.
            unavailable.append(exc)
            continue
        resolved.append(apply_choice(item, selection.choices))
    if unavailable:
        raise unavailable[0]
    return tuple(resolved)


def calculation_records(indicators) -> list[dict]:
    """Identity-relevant calculation records, canonically sorted by indicator key.

    Presentation (titles, descriptions, citations), set id, source path and
    selection order are excluded; the indicator key, method, normalized parameters
    and resolved registered calculation are retained by value.
    """
    indicators = tuple(indicators)
    if any(item.pending_baseline for item in indicators):
        raise IndicatorSetError(
            "historical Qcrit preparation is pending; calculation identity is unavailable"
        )
    return sorted(
        (
            {
                "indicator": item.indicator,
                "definition": item.definition,
                "method": item.method,
                "parameters": dict(sorted(item.parameters.items())),
                **({"set": item.calculation_set} if item.calculation_set else {}),
            }
            for item in indicators
        ),
        key=lambda record: record["indicator"],
    )


def default_calculations(tokens) -> list[dict]:
    """Calculation records of the registered default suite for ``tokens``."""
    return calculation_records(
        _registered(item.name) for item in registry.declarations(tokens)
    )


def declarations_for_calculations(records) -> tuple[registry.IndicatorDeclaration, ...]:
    """Re-derive declarations from retained calculation records against live code.

    Each record's method and parameters are re-validated, so a changed
    parameter-to-reducer mapping is observed rather than trusted.
    """
    records = tuple(records)
    result = []
    legacy = None
    revisions = {}
    for record in records:
        if "set" in record:
            from blueearth_cst.indicators import loader

            if set(record) != {
                "set",
                "indicator",
                "definition",
                "method",
                "parameters",
            }:
                raise IndicatorSetError(f"invalid local calculation record {record!r}")
            if record["definition"] != f"{record['set']}:{record['indicator']}":
                raise IndicatorSetError(
                    f"{record['indicator']}: local identity differs"
                )
            if record["set"] not in revisions:
                revisions[record["set"]] = loader.implementation_revision(record["set"])
            result.append(
                loader.resolve(
                    record["set"],
                    record["method"],
                    record["parameters"],
                    indicator_name=record["indicator"],
                    _revision=revisions[record["set"]],
                )
            )
            continue
        if set(record) != {"indicator", "definition", "method", "parameters"}:
            raise IndicatorSetError(f"invalid calculation record {record!r}")
        definition, _ = registry.validate_calculation(
            record["method"], record["parameters"]
        )
        if definition != record["definition"]:
            raise IndicatorSetError(
                f"{record['indicator']}: calculation now resolves to {definition!r}, "
                f"not the recorded {record['definition']!r}"
            )
        if legacy is None:
            legacy = registry.declaration_catalog(
                [item["definition"] for item in records if "set" not in item]
            )
        result.append(replace(legacy[definition], name=record["indicator"]))
    return tuple(result)


def required_variables(sets) -> list[str]:
    """Union of response variables the resolved sets need, in first-use order."""
    seen = {}
    for item in sets:
        for declaration in item.declarations:
            for requirement in declaration.requirements:
                seen.setdefault(requirement.variable, None)
    return list(seen)
