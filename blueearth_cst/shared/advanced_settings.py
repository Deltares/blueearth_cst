"""The closed ``config/advanced_settings.yml`` schema, its validators and loader.

Kept out of ``wf3_science`` on purpose: that module is in WF3's generation code
inventory, and a section added here for another workflow (WF4's ``batching:``,
2026-09-25) must not re-key every scenario collection. The generation-relevant
VALUES this file resolves -- the default seed, water-year start, spell factors --
reach the generation request and collection identity by value, so excluding the
schema code from the inventory loses no coverage.
"""

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

ADVANCED_SETTINGS_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "advanced_settings.yml"
)

_ADVANCED_SETTINGS_SCHEMA = {
    # `max_flagged_months` joined with `C-65`. A CONSTRAINT rather than a
    # default: D-10.6 classes it a hard limit a project may not relax, so
    # unlike `C-54` there is no overriding key to name (D-10.7).
    "constraints": {
        "min_historical_years": "positive_int",
        "max_flagged_months": "positive_int",
    },
    "defaults": {
        "batch_disk_headroom_fraction": "unit_fraction",
        "seed": "nonnegative_int",
        "water_year_start": "month_abbrev",
        # `C-36`: six defaults that backed a config key from a Python literal,
        # so the key and the value it falls back to lived in different tiers and
        # a reader of the config could not discover either from the other
        # (`parameter-placement.md` M3, owner ruling `Q-E`).
        "hydrography": "catalog_entry_name",
        "basin_index": "catalog_entry_name",
        "max_subbasins": "positive_int",
        "snap_tolerance_m": "positive_float",
        "spell_factor": "monthly_factors",
        "change_factor_stats": "statistic_names",
    },
    # `julia_threads` moved here from `defaults:` with `C-54`, which removed the
    # per-project override. `defaults:` is for values a project could have
    # overridden; nothing overrides this one now.
    "runtime": {"julia_threads": "positive_int", "julia_version": "version_string"},
    # WF4's batched Wflow runs (t2609242342). `threads`/`max_parallel` may be
    # `auto`, which picks a regime by the model's active cell count; the rest
    # are the regimes' values and the memory estimate that caps parallelism.
    "batching": {
        "threads": "auto_or_positive_int",
        "max_parallel": "auto_or_positive_int",
        "small_basin_cells": "positive_int",
        "small_threads": "positive_int",
        "small_max_parallel": "positive_int",
        "large_threads": "positive_int",
        "large_max_parallel": "positive_int",
        "memory_per_batch_gb": "positive_float",
        "memory_per_cell_kb": "positive_float",
    },
}

_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def _positive_int(value, where: str) -> int:
    """A whole number >= 1, rejecting the bool that ``isinstance(x, int)`` admits."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where} must be an integer, got {value!r}")
    if value < 1:
        raise ValueError(f"{where} must be >= 1, got {value}")
    return value


def _nonnegative_int(value, where: str) -> int:
    """A whole number >= 0. Separate from ``_positive_int`` because 0 is a
    legitimate randomization seed, and rejecting it would be an arbitrary hole
    in the accepted range rather than a constraint anything needs."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where} must be an integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{where} must be >= 0, got {value}")
    return value


_MONTH_ABBREVS = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)  # fmt: skip


def _month_abbrev(value, where: str) -> str:
    """A three-letter month name, normalized to ``Jan``-style capitalization.

    Defined up here with the other settings validators rather than beside the
    water-year helpers below: ``_VALIDATORS`` is built at module level, so a
    later definition would be a NameError at import.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"{where} must be a three-letter month name like 'Oct', got "
            f"{value!r} ({type(value).__name__})"
        )
    token = value.strip().capitalize()
    if token not in _MONTH_ABBREVS:
        raise ValueError(
            f"{where} must be one of {', '.join(_MONTH_ABBREVS)}, got {value!r}"
        )
    return token


def _unit_fraction(value, where: str) -> float:
    """A share of a whole, in ``(0, 1]``.

    Rejects 0 (a zero budget would cap every batch at 1 while claiming to have
    computed something) and anything above 1 (no share of free disk can exceed
    the free disk). Accepts an int so ``1`` need not be written ``1.0``, but
    not a bool, which ``isinstance(x, int)`` would otherwise admit.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{where} must be a number between 0 and 1, got {value!r} "
            f"({type(value).__name__})"
        )
    if not 0 < float(value) <= 1:
        raise ValueError(f"{where} must be > 0 and <= 1, got {value}")
    return float(value)


def _version_string(value, where: str) -> str:
    """A quoted three-part version.

    The non-string rejection is load-bearing rather than defensive: unquoted
    ``1.10`` in YAML parses to the FLOAT 1.11, which would silently become the
    selector ``+1.10`` and let juliaup pick whatever patch it likes.
    """
    if not isinstance(value, str):
        raise ValueError(
            f'{where} must be a quoted string like "1.11.7", got {value!r} '
            f"({type(value).__name__}) — an unquoted X.Y is parsed as a number"
        )
    if not _VERSION_RE.match(value):
        raise ValueError(f"{where} must be a three-part version X.Y.Z, got {value!r}")
    return value


def _positive_float(value, where: str) -> float:
    """A number > 0, with no upper bound.

    Separate from ``_unit_fraction`` because a tolerance in physical units is not a share
    of anything: 10000.0 is a legitimate value and 1.0 is not an implicit
    ceiling. Accepts an int so ``10000`` need not be written ``10000.0``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{where} must be a number, got {value!r} ({type(value).__name__})"
        )
    if float(value) <= 0:
        raise ValueError(f"{where} must be > 0, got {value}")
    return float(value)


def _catalog_entry_name(value, where: str) -> str:
    """A hydromt data-catalog entry name.

    Only the SHAPE is checkable here — whether the name resolves is a property
    of the catalog passed with ``-d``, which this file cannot see and must not
    pretend to. What it does catch is the failure that reads as a missing
    dataset: an empty string or a stray space, which reaches hydromt as a
    lookup for a source nobody registered.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"{where} must be a catalog entry name, got {value!r} "
            f"({type(value).__name__})"
        )
    token = value.strip()
    if not token or token != value or any(c.isspace() for c in token):
        raise ValueError(
            f"{where} must be a catalog entry name with no whitespace, got {value!r}"
        )
    return token


def _monthly_factors(value, where: str) -> list[float]:
    """Twelve numbers, one per calendar month.

    The LENGTH is the whole check. These reach weathergenr, which indexes them
    by month, so R would recycle or truncate a ten-element list rather than
    reject it and the run would perturb the wrong months in silence. Same
    predicate ``validate_spell_factor`` holds the per-project override to.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            f"{where} must be a list of 12 monthly coefficients, got "
            f"{value!r} ({type(value).__name__})"
        )
    if len(value) != 12:
        raise ValueError(
            f"{where} must have 12 entries, one per month, got {len(value)}"
        )
    out = []
    for index, item in enumerate(value, start=1):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{where}[{index}] must be a number, got {item!r}")
        out.append(float(item))
    return out


_QUANTILE_STAT_RE = re.compile(r"^q_(\d+)$")


def _statistic_names(value, where: str) -> list[str]:
    """A non-empty set of statistic names WF2 can actually compute.

    Deliberately not a closed enumeration. Each name is either ``q_<percentile>``
    — dispatched by parsing the number out, so the admissible set is every
    percentile rather than the four that happen to be documented — or an xarray
    reduction method looked up with ``getattr`` on the grouped object. Listing
    today's eight would refuse ``q_95``, which the code computes correctly.

    So the check is the shape both branches require: a non-empty, duplicate-free
    sequence of bare identifiers, with a quantile's percentile in range. A
    misspelled reduction still reaches ``getattr`` and fails there, by name, on
    the object that would have computed it — which is a better message than
    anything this function could invent.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            f"{where} must be a list of statistic names, got {value!r} "
            f"({type(value).__name__})"
        )
    if not value:
        raise ValueError(
            f"{where} must name at least one statistic; an empty set writes "
            f"change-factor tables with no values in them"
        )
    out = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, str) or not item.isidentifier():
            raise ValueError(
                f"{where}[{index}] must be a statistic name like 'mean' or "
                f"'q_90', got {item!r}"
            )
        quantile = _QUANTILE_STAT_RE.match(item)
        if quantile and not 1 <= int(quantile.group(1)) <= 99:
            raise ValueError(
                f"{where}[{index}] must be a percentile in 1..99, got {item!r}"
            )
        if item in out:
            raise ValueError(f"{where}[{index}] repeats {item!r}")
        out.append(item)
    # A list, not a tuple: `load_advanced_settings` round-trips the file, and a
    # validator that changed the container type would make the resolved settings
    # unequal to the YAML they came from for no gain.
    return out


def _auto_or_positive_int(value, where: str):
    """``auto`` (defer to the regime), or a whole number >= 1."""
    if value == "auto":
        return value
    return _positive_int(value, where)


_VALIDATORS = {
    "auto_or_positive_int": _auto_or_positive_int,
    "positive_int": _positive_int,
    "nonnegative_int": _nonnegative_int,
    "month_abbrev": _month_abbrev,
    "unit_fraction": _unit_fraction,
    "version_string": _version_string,
    "positive_float": _positive_float,
    "catalog_entry_name": _catalog_entry_name,
    "monthly_factors": _monthly_factors,
    "statistic_names": _statistic_names,
}


def load_advanced_settings(path=None) -> dict:
    """Read and validate ``config/advanced_settings.yml``.

    Returns ``{section: {key: value}}``. Raises ``ValueError`` naming the
    offending section or key on anything the schema does not admit: a missing
    section, a missing key, an unknown section, an unknown key, or a value that
    fails its validator.

    Deliberately has NO built-in fallback. A silent fallback would mean a
    deleted or mistyped settings file changes what the toolbox enforces without
    saying so — exactly the failure mode the closed schema exists to prevent.
    The file is tracked; if it is absent the checkout is broken, and that should
    be said plainly at import.
    """
    settings_path = Path(path) if path is not None else ADVANCED_SETTINGS_PATH
    try:
        raw = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"advanced settings file not found at {settings_path}. It is "
            f"tracked in the repository; a checkout without it cannot state "
            f"what the toolbox enforces"
        ) from None
    if not isinstance(raw, Mapping):
        raise ValueError(f"{settings_path} is not a YAML mapping")

    unknown_sections = sorted(set(raw) - set(_ADVANCED_SETTINGS_SCHEMA))
    if unknown_sections:
        raise ValueError(
            f"{settings_path}: unknown section(s) {unknown_sections}; expected "
            f"{sorted(_ADVANCED_SETTINGS_SCHEMA)}"
        )

    resolved = {}
    for section, keys in _ADVANCED_SETTINGS_SCHEMA.items():
        if section not in raw:
            raise ValueError(f"{settings_path}: missing section {section!r}")
        body = raw[section]
        if not isinstance(body, Mapping):
            raise ValueError(f"{settings_path}: section {section!r} is not a mapping")
        unknown_keys = sorted(set(body) - set(keys))
        if unknown_keys:
            raise ValueError(
                f"{settings_path}: unknown key(s) {unknown_keys} in section "
                f"{section!r}; expected {sorted(keys)}"
            )
        resolved[section] = {}
        for key, validator in keys.items():
            if key not in body:
                raise ValueError(f"{settings_path}: missing {section}.{key}")
            resolved[section][key] = _VALIDATORS[validator](
                body[key], f"{section}.{key}"
            )
    return resolved


ADVANCED_SETTINGS = load_advanced_settings()
