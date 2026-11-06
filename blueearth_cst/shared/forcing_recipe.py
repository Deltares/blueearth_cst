"""Pure normalization and scientific identity of selected forcing recipes.

Catalog references describe intent, not source availability or qualification.
Consumers must validate data and call the workflow guard before side effects.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast


def _canonical_json(document: dict[str, object]) -> bytes:
    """Use sorted compact UTF-8 JSON with a final LF, like collection identities."""
    return (
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True)
class ForcingRecipe:
    """Effective choices returned by :func:`normalize_forcing_recipe`.

    ``press_correction`` and ``pet_method`` are None for supplied PET. The full
    identity describes preparation intent; only the extraction identity belongs
    in climate-store attributes used to derive automatic generator seeds.
    """

    precip_source: str
    temp_pet_source: str
    orography_source: str
    temp_correction: bool
    press_correction: bool | None
    pet_mode: Literal["computed", "direct"]
    pet_method: Literal["debruin", "makkink"] | None
    pet_source: str | None

    @property
    def companion_variables(self) -> tuple[str, ...]:
        """Variables required from the companion, including generator extrema."""
        if self.pet_mode == "direct":
            return ("temp",)
        variables = ("temp", "temp_min", "temp_max", "press_msl", "kin")
        return variables + (("kout",) if self.pet_method == "debruin" else ())

    @property
    def required_variables(self) -> tuple[str, ...]:
        """Complete extracted-store variables, independent of the first workflow."""
        return (
            ("precip",)
            + self.companion_variables
            + (("pet",) if self.pet_mode == "direct" else ())
        )

    def to_dict(self) -> dict[str, object]:
        """Return fresh JSON-compatible full effective recipe provenance."""
        pet: dict[str, object] = {"mode": self.pet_mode}
        if self.pet_mode == "computed":
            pet["method"] = self.pet_method
        else:
            pet["source"] = self.pet_source
        return {
            "schema_version": "forcing-recipe/1",
            "precip_source": self.precip_source,
            "temp_pet_source": self.temp_pet_source,
            "orography_source": self.orography_source,
            "temp_correction": self.temp_correction,
            "press_correction": self.press_correction,
            "pet": pet,
        }

    def extraction_dict(self) -> dict[str, object]:
        """Project only extraction inputs; exclude preparation-only corrections.

        Same-source extraction reads native fields without lapse correction.
        Cross-source extraction regrids the companion and carries its declared
        elevation/correction choices. PET algorithms enter through their variable
        requirements, while direct PET adds its independent source.
        """
        result: dict[str, object] = {
            "schema_version": "forcing-extraction/1",
            "precip_source": self.precip_source,
            "temp_pet_source": self.temp_pet_source,
            "required_variables": list(self.required_variables),
        }
        if self.temp_pet_source != self.precip_source:
            result["orography_source"] = self.orography_source
            result["temp_correction"] = self.temp_correction
        if self.pet_mode == "direct":
            result["pet_source"] = self.pet_source
        return result

    def settings(self) -> dict[str, object]:
        """Return the equivalent closed ``climate.forcing`` configuration."""
        result = self.to_dict()
        result.pop("schema_version")
        result.pop("precip_source")
        if self.pet_mode == "direct":
            result.pop("press_correction")
        return result

    def canonical_bytes(self) -> bytes:
        """Serialize the full effective recipe deterministically."""
        return _canonical_json(self.to_dict())

    def extraction_bytes(self) -> bytes:
        """Serialize only the extraction projection deterministically."""
        return _canonical_json(self.extraction_dict())

    @property
    def fingerprint(self) -> str:
        """Full SHA-256 of effective preparation intent, not dataset contents."""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @property
    def extraction_fingerprint(self) -> str:
        """Full SHA-256 of extraction intent, excluding preparation-only choices."""
        return hashlib.sha256(self.extraction_bytes()).hexdigest()


def _mapping(value: object, allowed: set[str], field: str) -> Mapping[str, object]:
    """Reject non-mappings and misspelled or non-string configuration keys."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    for key in value:
        if not isinstance(key, str) or key not in allowed:
            raise ValueError(f"{field}: unknown key {key!r}")
    return value


def _source(value: object, field: str) -> str:
    """Require a nonempty catalog name, without silently normalizing identity."""
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or any(char in value for char in ("/", "\\", "\n", "\r", "\t"))
        or value in {".", ".."}
    ):
        raise ValueError(f"{field} must be a nonempty catalog entry name, not a path")
    return value


def _boolean(value: object, field: str) -> bool:
    """Reject truthy strings and integers rather than changing scientific intent."""
    if type(value) is not bool:
        raise ValueError(f"{field} must be a boolean")
    return value


def normalize_forcing_recipe(
    precip_source: str, forcing: Mapping[str, object] | None = None
) -> ForcingRecipe:
    """Resolve selected precipitation and optional ``climate.forcing`` settings.

    Raises ValueError for unsupported selected sources, unknown keys, wrong
    types, mode conflicts or an alternate companion without explicit orography.
    This function performs no I/O and does not admit the mapping into composition.
    """
    precip_source = _source(precip_source, "climate.selected")
    if precip_source not in {"era5", "chirps", "chirps_global"}:
        raise ValueError("climate.selected must be era5, chirps or chirps_global")
    fields = _mapping(
        {} if forcing is None else forcing,
        {
            "temp_pet_source",
            "orography_source",
            "temp_correction",
            "press_correction",
            "pet",
        },
        "climate.forcing",
    )
    companion = _source(
        fields.get("temp_pet_source", "era5"), "climate.forcing.temp_pet_source"
    )
    if companion != "era5" and "orography_source" not in fields:
        raise ValueError(
            "climate.forcing.orography_source is required for an alternate temp_pet_source"
        )
    orography = _source(
        fields.get("orography_source", "era5_orography"),
        "climate.forcing.orography_source",
    )
    temp_correction = _boolean(
        fields.get("temp_correction", True), "climate.forcing.temp_correction"
    )
    pet = _mapping(
        fields.get("pet", {}), {"mode", "method", "source"}, "climate.forcing.pet"
    )
    mode = pet.get("mode", "computed")
    if not isinstance(mode, str) or mode not in {"computed", "direct"}:
        raise ValueError("climate.forcing.pet.mode must be computed or direct")
    if mode == "direct":
        if "method" in pet or "press_correction" in fields:
            raise ValueError(
                "direct PET forbids pet.method and explicit press_correction; remove these settings"
            )
        pet_source = _source(pet.get("source"), "climate.forcing.pet.source")
        method = None
        pressure = None
    else:
        if "source" in pet:
            raise ValueError(
                "computed PET forbids pet.source; use pet.mode: direct for supplied PET"
            )
        method = pet.get("method", "debruin")
        if not isinstance(method, str) or method not in {"debruin", "makkink"}:
            raise ValueError("climate.forcing.pet.method must be debruin or makkink")
        pressure = _boolean(
            fields.get("press_correction", True), "climate.forcing.press_correction"
        )
        pet_source = None
    return ForcingRecipe(
        precip_source,
        companion,
        orography,
        temp_correction,
        pressure,
        cast(Literal["computed", "direct"], mode),
        cast(Literal["debruin", "makkink"] | None, method),
        pet_source,
    )


def require_workflow_compatibility(recipe: ForcingRecipe, workflow: str) -> None:
    """Reject unsupported direct-PET use before a caller performs side effects.

    Workflow names are the repository entry-point stems, not aliases or numbers.
    Unknown names fail closed so a misspelling cannot bypass the restriction.
    """
    if workflow not in (
        "analyze_climate",
        "build_model",
        "analyze_projections",
        "generate_scenarios",
        "simulate_system",
    ):
        raise ValueError(f"unknown workflow {workflow!r}")
    if recipe.pet_mode == "direct" and workflow in (
        "generate_scenarios",
        "simulate_system",
    ):
        raise ValueError(
            f"{workflow} does not support direct PET; use computed PET or restrict the recipe to historical forcing"
        )


def read_forcing_recipe(document: Mapping[str, object]) -> ForcingRecipe:
    """Validate archived full provenance without silently filling missing fields."""
    if not isinstance(document, Mapping):
        raise ValueError("effective forcing recipe must be a mapping")
    settings = {
        key: value
        for key, value in document.items()
        if key not in {"schema_version", "precip_source"}
    }
    if (
        isinstance(settings.get("pet"), Mapping)
        and settings["pet"].get("mode") == "direct"
    ):
        if settings.get("press_correction") is None:
            settings.pop("press_correction", None)
    recipe = normalize_forcing_recipe(document.get("precip_source"), settings)
    if recipe.to_dict() != document:
        raise ValueError(
            "effective forcing recipe is incomplete or has an unsupported version"
        )
    return recipe
