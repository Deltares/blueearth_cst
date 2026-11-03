"""Discover trusted installed sets and orchestrate their local callables."""

import hashlib
import importlib
import inspect
import pkgutil
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from blueearth_cst.indicators import sets
from blueearth_cst.indicators.contract import (
    EvaluationContext,
    IndicatorDeclaration,
    IndicatorResult,
)
from blueearth_cst.indicators.temporal import validate_requirement
from blueearth_cst.indicators.validation import validate_parameters, validate_result


def _set_packages() -> dict[str, str]:
    """Map public set IDs to installed packages, allowing an explicit SET_ID."""
    packages = {}
    for item in pkgutil.iter_modules(sets.__path__):
        if not item.ispkg:
            continue
        name = f"{sets.__name__}.{item.name}"
        package = importlib.import_module(name)
        set_id = getattr(package, "SET_ID", item.name)
        if not isinstance(set_id, str) or not set_id or set_id in packages:
            raise ValueError(
                f"invalid or duplicate installed indicator set ID: {set_id!r}"
            )
        packages[set_id] = name
    return packages


def available_sets() -> tuple[str, ...]:
    """List installed trusted packages; YAML cannot name an arbitrary import path."""
    return tuple(sorted(_set_packages()))


def load(set_id: str):
    """Load one discovered package's set-owned registration module."""
    packages = _set_packages()
    if set_id not in packages:
        raise ValueError(
            f"unknown indicator set {set_id!r}; discoverable sets: {list(available_sets())}"
        )
    return importlib.import_module(f"{packages[set_id]}.registry")


load_set = load


def execution_sources(set_ids=None) -> tuple[Path, ...]:
    """Return execution-only sources; optional checking code has its own identity."""
    root = Path(__file__).parent
    selected = available_sets() if set_ids is None else set_ids
    paths = list(root.glob("*.py"))
    paths += [root / "sets" / "__init__.py"]
    for set_id in selected:
        registry = load(set_id)
        paths += list(Path(registry.__file__).parent.glob("*.py"))
        pending, visited = [registry], set()
        while pending:
            module = pending.pop()
            if module.__name__ in visited:
                continue
            visited.add(module.__name__)
            if module.__file__:
                paths.append(Path(module.__file__))
            pending.extend(
                value
                for value in vars(module).values()
                if inspect.ismodule(value)
                and value.__name__.startswith(f"{sets.__name__}.")
            )
        for indicator in registry.INDICATORS:
            path = inspect.getsourcefile(indicator.calculate)
            if path:
                paths.append(Path(path))
    return tuple(
        sorted(
            {p.resolve() for p in paths if p.name not in {"checks.py", "checking.py"}}
        )
    )


def implementation_revision(set_id: str) -> str:
    """Content identity of the contributing execution sources, without checkers."""
    paths = execution_sources([set_id])
    signature = tuple(
        (str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in paths
    )
    return _source_revision(signature)


@lru_cache(maxsize=32)
def _source_revision(signature) -> str:
    """Reuse source-byte hashing while the actual file identities remain unchanged."""
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for name, _, _ in signature:
        path = Path(name)
        try:
            relative = path.relative_to(root)
        except ValueError:
            relative = Path(*path.parts[-2:])
        digest.update(str(relative).replace("\\", "/").encode())
        digest.update(path.read_text(encoding="utf-8").encode("utf-8"))
    return digest.hexdigest()


def _declaration(
    set_id, indicator, *, name=None, parameters=None, revision=None
) -> IndicatorDeclaration:
    params = dict(indicator.parameters if parameters is None else parameters)
    module = importlib.import_module(indicator.calculate.__module__)
    try:
        params = validate_parameters(indicator.calculate, params, module.PARAMETERS)
        if hasattr(module, "validate"):
            module.validate(indicator.calculate, params)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{set_id}/{name or indicator.id}: {error}") from error
    methods = load(set_id).METHODS
    method = indicator.method or next(
        (key for key, fn in methods.items() if fn is indicator.calculate),
        indicator.calculate.__name__,
    )
    period = params.get("return_period", indicator.return_period)
    return IndicatorDeclaration(
        name=name or indicator.id,
        statistic=indicator.statistic,
        grain=indicator.grain,
        required_responses=indicator.input,
        bundle_by="same_design_point" if indicator.grain == "bundle" else None,
        reference=indicator.reference,
        reference_location=indicator.reference_location,
        return_period=period,
        blocks_per_return_period=indicator.blocks_per_return_period,
        implementation_revision=(
            implementation_revision(set_id) if revision is None else revision
        ),
        value_validity="finite required; reject invalid GEV fit; no clipping"
        if indicator.grain == "bundle"
        else "finite or native-reduction NaN; refuse infinity; no clipping",
        group_reduction=indicator.group_reduction,
        calculation_set=set_id,
        calculation_method=method,
        calculation_parameters=params,
        additional_responses=indicator.additional_inputs,
        output_variable=indicator.output_variable,
    )


def declarations(set_id: str) -> tuple[IndicatorDeclaration, ...]:
    """Resolve all registrations without central calculation inventory edits."""
    registry = load(set_id)
    revision = implementation_revision(set_id)
    return tuple(
        _declaration(set_id, item, revision=revision) for item in registry.INDICATORS
    )


def resolve(
    set_id: str, method: str, parameters, *, indicator_name=None, _revision=None
) -> IndicatorDeclaration:
    """Validate a local callable and return its resolved declaration."""
    identity = f"{set_id}/{indicator_name or method}"
    registry = load(set_id)
    functions = dict(registry.METHODS)
    functions.update(
        {item.calculate.__name__: item.calculate for item in registry.INDICATORS}
    )
    if method not in functions:
        raise ValueError(
            f"{identity}: unknown calculation {method!r}; accepted calculations: {sorted(functions)}"
        )
    function = functions[method]
    module = importlib.import_module(function.__module__)
    try:
        if hasattr(module, "normalize_parameters"):
            parameters = module.normalize_parameters(function, parameters)
        normalized = validate_parameters(function, parameters, module.PARAMETERS)
        if hasattr(module, "validate"):
            module.validate(function, normalized)
        candidates = [
            item for item in registry.INDICATORS if item.calculate is function
        ]
        exact = [item for item in candidates if dict(item.parameters) == normalized]
        named = [item for item in candidates if item.id == indicator_name]
        if named:
            template = named[0]
            # Parameters encoded in a registered name cannot silently change its meaning.
            for key in ("season", "window", "extreme", "return_period", "direction"):
                if (
                    key in template.parameters
                    and normalized.get(key) != template.parameters[key]
                ):
                    raise ValueError(
                        f"{key} must remain {template.parameters[key]!r} for registered indicator {indicator_name}"
                    )
        elif exact:
            template = exact[0]
        elif len(candidates) == 1:
            template = candidates[0]
        elif hasattr(registry, "resolve_indicator"):
            template = registry.resolve_indicator(function, normalized)
        else:
            raise ValueError(
                f"no unambiguous declaration matches parameters {normalized!r}"
            )
        return _declaration(
            set_id,
            template,
            name=indicator_name,
            parameters=normalized,
            revision=_revision,
        )
    except (ValueError, TypeError) as error:
        raise ValueError(f"{identity}: {error}") from error


def local_declaration(indicator: IndicatorDeclaration) -> IndicatorDeclaration:
    """Adapt the legacy registry's vocabulary without making it a dependency."""
    if indicator.calculation_set:
        return indicator
    # Legacy statistics have fixed owners. Discovering new sets here could
    # silently redirect an existing statistic to a newly installed callable.
    for set_id in ("default", "flow_regime"):
        for item in load(set_id).INDICATORS:
            if (item.statistic, item.input.variable) == (
                indicator.statistic,
                indicator.required_responses.variable,
            ):
                candidate = _declaration(
                    set_id, item, revision=indicator.implementation_revision
                )
                return replace(
                    candidate,
                    name=indicator.name,
                    implementation_revision=indicator.implementation_revision,
                    reference=indicator.reference,
                    reference_location=indicator.reference_location,
                    group_reduction=indicator.group_reduction,
                    return_period=indicator.return_period,
                    blocks_per_return_period=indicator.blocks_per_return_period,
                )
    raise ValueError(
        f"legacy/{indicator.name}: no set-local calculation for statistic {indicator.statistic!r}"
    )


def unavailable(indicator: IndicatorDeclaration, locations=None) -> tuple:
    """``(location, reason)`` pairs an indicator cannot evaluate for absent inputs.

    A set declares absence through an optional callable-keyed ``AVAILABILITY``
    beside ``PARAMETERS``. ``locations=None`` asks only for location-independent
    absence; a ``None`` location in the result means every location. Malformed
    parameters are refused at resolution and are never reported here.
    """
    indicator = local_declaration(indicator)
    registry = load(indicator.calculation_set)
    function = registry.METHODS.get(indicator.calculation_method) or next(
        item.calculate
        for item in registry.INDICATORS
        if item.calculate.__name__ == indicator.calculation_method
    )
    module = importlib.import_module(function.__module__)
    check = getattr(module, "AVAILABILITY", {}).get(function)
    if check is None:
        return ()
    return tuple(check(indicator.calculation_parameters, locations))


def registration(indicator: IndicatorDeclaration):
    """Return the set-owned registration supplying units and descriptive text."""
    indicator = local_declaration(indicator)
    entries = load(indicator.calculation_set).INDICATORS
    matches = [item for item in entries if item.id == indicator.name]
    if not matches:
        matches = [
            item
            for item in entries
            if dict(item.parameters) == indicator.calculation_parameters
            and (item.method or item.calculate.__name__)
            in {
                indicator.calculation_method,
                indicator.calculation_method.split("/")[0],
            }
        ]
    if not matches:
        matches = [
            item
            for item in entries
            if (item.method or item.calculate.__name__)
            in {
                indicator.calculation_method,
                indicator.calculation_method.split("/")[0],
            }
        ]
    if not matches:
        raise ValueError(
            f"{indicator.calculation_set}/{indicator.name}: no local registration"
        )
    return matches[0]


def mean_run_results(indicator, results, *, expected_run_ids) -> IndicatorResult:
    """Average already evaluated members with explicit membership and NaN propagation."""
    members = tuple(expected_run_ids)
    if (
        indicator.grain != "run"
        or indicator.group_reduction != "equal-weight-run-reductions-propagate-nan/1"
    ):
        raise ValueError("no reviewed group mean policy")
    if not members or len(set(members)) != len(members) or set(results) != set(members):
        raise ValueError(
            "group membership requires unique evaluated members matching the declaration"
        )
    ordered = [results[run] for run in members]
    for result in ordered:
        validate_result(result)
    if any(set(item.values.index) != set(ordered[0].values.index) for item in ordered):
        raise ValueError("group location sets differ")
    values = pd.Series(
        np.mean(
            np.stack(
                [
                    item.values.reindex(ordered[0].values.index).to_numpy()
                    for item in ordered
                ]
            ),
            axis=0,
        ),
        index=ordered[0].values.index,
    )
    result = IndicatorResult(
        values,
        metadata={"member_run_ids": members},
        statuses={
            str(location): "undefined" if pd.isna(value) else "available"
            for location, value in values.items()
        },
    )
    validate_result(result)
    return result


def evaluate(
    indicator,
    series,
    *,
    anchor,
    reference=None,
    expected_run_ids=None,
    validation=None,
    group=False,
    _prepared=None,
) -> IndicatorResult:
    """Evaluate locations or member groups; unexpected calculation faults propagate."""
    indicator = local_declaration(indicator)
    identity = f"{indicator.calculation_set}/{indicator.name}"
    registry = load(indicator.calculation_set)
    function = registry.METHODS.get(indicator.calculation_method)
    if function is None:
        function = next(
            item.calculate
            for item in registry.INDICATORS
            if item.calculate.__name__ == indicator.calculation_method
        )
    members = tuple(
        sorted({item.run_id for item in series})
        if expected_run_ids is None
        else expected_run_ids
    )
    prepared = {} if _prepared is None else _prepared
    context = EvaluationContext(anchor, reference, members, validation, prepared)
    try:
        offset = pd.tseries.frequencies.to_offset(anchor)
        if not offset.freqstr.startswith(("YE-", "YS-")) or offset.n != 1:
            raise ValueError(
                "anchor: expected a one-year water-year frequency such as YE-DEC or YE-SEP"
            )
        validate_requirement(indicator, series)
        if group and indicator.grain == "run":
            if (
                indicator.group_reduction
                != "equal-weight-run-reductions-propagate-nan/1"
            ):
                raise ValueError("no reviewed group mean policy")
            if (
                not members
                or len(set(members)) != len(members)
                or set(members) != {item.run_id for item in series}
            ):
                raise ValueError(
                    "group membership requires unique evaluated members matching the declaration"
                )
            results = {
                run: evaluate(
                    indicator,
                    [item for item in series if item.run_id == run],
                    anchor=anchor,
                    reference=reference,
                    validation=validation,
                    _prepared=prepared,
                )
                for run in members
            }
            result = mean_run_results(indicator, results, expected_run_ids=members)
        else:
            result = function(indicator, series, context)
        validate_result(result)
    except Exception as error:
        locations = sorted({item.location_id for item in series})
        # Preserve exception type, evidence attributes, cause and traceback.
        error.args = (f"{identity}/location={locations}: {error}", *error.args[1:])
        raise
    statuses = result.statuses or {
        str(location): "undefined" if pd.isna(value) else "available"
        for location, value in result.values.items()
    }
    return replace(result, statuses=statuses)


def check(
    indicator,
    series,
    result,
    *,
    anchor,
    reference=None,
    expected_run_ids=None,
    validation=None,
    group=False,
) -> dict:
    """Delegate numerical checking lazily, outside execution source identity."""
    from blueearth_cst.indicators.checking import check as numerical_check

    return numerical_check(
        indicator,
        series,
        result,
        anchor=anchor,
        reference=reference,
        expected_run_ids=expected_run_ids,
        validation=validation,
        group=group,
    )
