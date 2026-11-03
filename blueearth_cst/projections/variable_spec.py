"""The variable spec: canonical quantity and change semantics (design §5.5, 5e).

``variables`` used to be a bare list — ``[precip, temp]`` — and everything
downstream inferred meaning from the *name*: stage B branched on the literal
string ``"precip"`` to decide that a change should be relative rather than
absolute. That works exactly as long as nobody adds a variable, and fails
silently the moment somebody does, because a new relative variable named anything
else is differenced as though it were a temperature.

The spec states what a name cannot:

```yaml
variables:
  precip: {source: precip, canonical: rate,  units: mm/day, change: relative}
  temp:   {source: temp,   canonical: state, units: degC,   change: absolute}
```

* ``source`` names the **post-rename** variable, because the catalog's
  ``data_adapter.rename`` maps ``pr → precip`` and ``tas → temp`` before the
  reducer sees the data.
* ``canonical`` is what the stored monthly series **is** — a monthly mean *rate*
  in the declared units, or a monthly mean *state*. Converting a source's native
  frequency to the canonical quantity is a property of the **source**, not the
  variable; in v2.0 there is one source frequency (``Amon``) so it is the
  identity.
* ``change`` is read by stage B. **Nothing infers anything from a name.**

Falsifier K7: a variable named something other than ``precip``, declared
``change: relative``, must be treated as relative.
"""

from __future__ import annotations

from typing import NamedTuple

from blueearth_cst.shared.variable_registry import (
    CANONICAL_KINDS,
    CHANGE_KINDS,
    VARIABLES,
)

__all__ = ["CANONICAL_KINDS", "CHANGE_KINDS", "VariableSpec", "parse", "source_names"]


class VariableSpec(NamedTuple):
    """One configured variable and what it means."""

    name: str
    source: str
    canonical: str
    units: str
    change: str
    #: `C-64`. **Appended and must stay last** -- the params boundary flattens
    #: this to a list and rebuilds it positionally.
    min_denominator: float | None = None

    @property
    def long_name(self) -> str:
        return f"{self.name} ({self.canonical}, {self.units})"


def _threshold(name, body):
    """The relative-change threshold for ``name``: config first, registry second.

    `C-64` moved the shipped defaults out of ``dry_month.DEFAULT_MIN_REFERENCE``
    and into the registry, which introduces a precedence question that did not
    exist before -- both a registry value and a declared value can now be
    present. **The config wins.** The other order is the quiet failure: a
    project that deliberately set a threshold would silently get the shipped
    one, and the result is a plausible number rather than an error.

    The registry fallback is what keeps every shipped config working. All four
    declare `precip` in the LONG form and none of them names a threshold, so
    without it `resolve_thresholds` would refuse every existing project on the
    next run.

    An unregistered variable that declares neither gets ``None`` and is refused
    later, by name, in :func:`dry_month.resolve_thresholds` -- which is the
    right place, because only there is it known whether the variable is
    relative and therefore whether a threshold was needed at all.
    """
    # `is not None` rather than `in body`: the SHORT form resolves through
    # `_resolve`, which hands back the registry entry whole -- including
    # `min_denominator: None` for an absolute variable. Testing for the key's
    # presence would read that null as a declared value and refuse `temp:`.
    # It also gives an explicit `min_denominator: null` the only sensible
    # meaning: defer to the registry.
    if body.get("min_denominator") is not None:
        value = body["min_denominator"]
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"variables.{name}.min_denominator must be a number; got {value!r}"
            ) from None
        if value <= 0:
            raise ValueError(
                f"variables.{name}.min_denominator must be > 0; got {value!r}. "
                "It is a near-zero guard on the DENOMINATOR of a relative "
                "change; zero or negative would admit the division it exists "
                "to prevent."
            )
        return value
    entry = VARIABLES.get(name)
    if entry is not None and entry.projections is not None:
        return entry.projections.min_denominator
    return None


def _resolve(name, body):
    """Fill a bare ``precip:`` from the registry (`C-57`), or refuse saying how.

    The SHORT form is a name with no body:

    ```yaml
    variables:
      precip:
      temp:
    ```

    which resolves to exactly what the long form would have said, from
    ``shared/variable_registry.py``. The long form keeps working untouched: it
    is the only way to declare a variable the registry has never heard of, and
    removing it would make the registry a wall rather than a default.

    **Two refusals, because there are two different faults**, and telling a user
    the wrong remedy is worse than telling them nothing:

    * a name the registry does not know -- add it to the registry, or declare it
      in full here;
    * a name the registry knows but does not DIFFERENCE, which today is ``pet``
      -- it has presentation metadata and no projection spec, so "add it to the
      registry" would be false advice about a variable that is already in it.
    """
    if body is not None:
        return body
    entry = VARIABLES.get(name)
    if entry is None:
        raise ValueError(
            f"variables.{name} is declared with no body, which resolves it from "
            f"the variable registry -- and {name!r} is not in it.\n"
            "  Either add it to `blueearth_cst/shared/variable_registry.py`, or "
            "declare it in full here:\n"
            f"    {name}: {{source: ..., canonical: rate|state, units: ..., "
            "change: relative|absolute}"
        )
    if entry.projections is None:
        raise ValueError(
            f"variables.{name} is declared with no body, and {name!r} IS in the "
            "variable registry but carries no projection spec -- it is a "
            "variable this toolbox plots and does not difference.\n"
            "  Declare it in full here if WF2 should difference it:\n"
            f"    {name}: {{source: ..., canonical: rate|state, units: ..., "
            "change: relative|absolute}"
        )
    return dict(entry.projections._asdict())


def parse(variables) -> dict[str, VariableSpec]:
    """Parse the ``variables:`` config block into a validated spec map.

    A bare list — the pre-5e shape — **raises**, naming the new form. Accepting it
    silently would leave the name-based inference in place for exactly the configs
    that had not been migrated, which is the failure mode the spec exists to end;
    and the migration is mechanical enough to state in the error.
    """
    if isinstance(variables, (list, tuple)):
        example = ", ".join(
            f"{v}: {{source: {v}, canonical: ..., units: ..., change: ...}}"
            for v in list(variables)[:2]
        )
        raise ValueError(
            "analyze_projections.variables is a list, which is the pre-5e shape. "
            "It is now a mapping declaring each variable's canonical quantity and "
            f"change semantics (design §5.5), e.g.\n  variables:\n    {example}\n"
            "Refusing rather than assuming: under the list form stage B inferred "
            "`relative` from the literal name 'precip', so any other relative "
            "variable was silently differenced as if it were a temperature."
        )
    if not isinstance(variables, dict) or not variables:
        raise ValueError(
            f"analyze_projections.variables must be a non-empty mapping; got {variables!r}"
        )

    spec: dict[str, VariableSpec] = {}
    for name, body in variables.items():
        body = _resolve(name, body)
        if not isinstance(body, dict):
            raise ValueError(
                f"variables.{name} must be a mapping with source/canonical/units/"
                f"change; got {body!r}"
            )
        missing = {"source", "canonical", "units", "change"} - set(body)
        if missing:
            raise ValueError(
                f"variables.{name} is missing {sorted(missing)}. Every field is "
                "required: an absent `change` would put the name-based guess back."
            )
        if body["canonical"] not in CANONICAL_KINDS:
            raise ValueError(
                f"variables.{name}.canonical must be one of {sorted(CANONICAL_KINDS)}; "
                f"got {body['canonical']!r}"
            )
        if body["change"] not in CHANGE_KINDS:
            raise ValueError(
                f"variables.{name}.change must be one of {sorted(CHANGE_KINDS)}; "
                f"got {body['change']!r}"
            )
        spec[name] = VariableSpec(
            name=name,
            source=str(body["source"]),
            canonical=str(body["canonical"]),
            units=str(body["units"]),
            change=str(body["change"]),
            min_denominator=_threshold(name, body),
        )
    return spec


def source_names(spec: dict[str, VariableSpec]) -> list[str]:
    """Post-rename variable names to request from the catalog, sorted.

    Sorted for the same reason ``intersection`` is: this list reaches the digest
    components, and an unordered one would make the cache key depend on mapping
    iteration order.
    """
    return sorted(v.source for v in spec.values())


def change_kind(spec, variable_name, default="absolute") -> str:
    """How ``variable_name``'s change is computed — from the SPEC, not the name.

    ``default`` covers callers with no spec (the synthetic datasets in the change
    unit tests). Production always passes one.
    """
    if not spec:
        # Pre-5e behaviour, retained only for spec-less callers: infer from the
        # name. Kept narrow deliberately -- it is the thing 5e exists to replace.
        return "relative" if variable_name == "precip" else default
    entry = spec.get(variable_name)
    return entry.change if entry is not None else default


def canonical_kind(spec, variable_name, default="state") -> str:
    """Whether ``variable_name`` is a `rate` or a `state` — from the SPEC.

    This drives the ANNUAL AGGREGATION (a rate integrates over the year, a state
    averages), which is a different decision from :func:`change_kind`'s. Both used
    to hang off the same `var == "precip"` test, so the two could never disagree
    even though they are independent properties.
    """
    if not spec:
        return "rate" if variable_name == "precip" else default
    entry = spec.get(variable_name)
    return entry.canonical if entry is not None else default


def as_digest_component(spec: dict[str, VariableSpec]) -> list:
    """A stable, ordered projection of the spec for the series digest.

    The spec determines the numbers — `change` picks the arithmetic — so it must
    be part of the cache key, and it must serialise deterministically or the key
    moves for no reason.
    """
    return [list(spec[name]) for name in sorted(spec)]
