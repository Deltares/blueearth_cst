"""Pure scenario enumeration and ancestry checks for the WF3 provider seam.

Rows contain generation meaning; simulators consume only their opaque run ids.
This module performs no file access and does not resolve stochastic seeds.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


class EmptyScenarioSetError(ValueError):
    """A generation request cannot produce an empty scenario set."""


class UnitNamespaceCapacityError(ValueError):
    """The explicitly supplied namespace cannot contain the requested runs."""


@dataclass(frozen=True)
class ScenarioRow:
    """One ordered forcing case and its provider-owned scenario payload."""

    run_id: str
    derived_from: str
    evaluated: bool
    scenario_type: str
    payload: tuple[tuple[str, str], ...]

    def as_record(self) -> dict[str, str]:
        """Return the normative column order with textual cross-language values."""
        return {
            "run_id": self.run_id,
            "derived_from": self.derived_from,
            "evaluated": str(self.evaluated).lower(),
            "scenario_type": self.scenario_type,
            **dict(self.payload),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, str]) -> "ScenarioRow":
        """Restore a row passed through Snakemake's serializable params."""
        if record["evaluated"] not in {"true", "false"}:
            raise ValueError("evaluated must be textual true or false")
        core = {"run_id", "derived_from", "evaluated", "scenario_type"}
        return cls(
            record["run_id"],
            record["derived_from"],
            record["evaluated"] == "true",
            record["scenario_type"],
            tuple((key, value) for key, value in record.items() if key not in core),
        )


def _positive_integer(value: object, field: str) -> int:
    """Refuse coercions that would change a declared count."""
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer, got {value!r}")
    return value


def validate_forest(rows: Sequence[ScenarioRow], *, unit_id_capacity: int) -> None:
    """Validate ordered ids, payload separation and an in-collection forest.

    An empty ancestry graph is valid for an independent fixture provider.
    Stochastic pairing and configured completeness require the stronger checker.
    """
    capacity = _positive_integer(unit_id_capacity, "unit_id_capacity")
    if not rows:
        raise EmptyScenarioSetError("generation requires at least one scenario row")
    if len(rows) > capacity:
        raise UnitNamespaceCapacityError(
            f"capacity={capacity}, known runs={len(rows)}, bundles=not-evaluated; "
            f"minimum replacement unit_id_capacity={len(rows)}"
        )
    width = len(str(capacity))
    expected_ids = [f"run_{i:0{width}d}" for i in range(1, len(rows) + 1)]
    if [row.run_id for row in rows] != expected_ids:
        raise ValueError(
            "run_id must be the ordered, capacity-padded sequence run_1..run_n"
        )
    by_id = {row.run_id: row for row in rows}
    if len({row.scenario_type for row in rows}) != 1:
        raise ValueError("a scenario table must contain exactly one scenario type")
    reserved = {"run_id", "derived_from", "evaluated", "scenario_type"}
    for row in rows:
        names = [name for name, _ in row.payload]
        if len(names) != len(set(names)) or reserved.intersection(names):
            raise ValueError(
                f"run_id={row.run_id}: duplicate or reserved payload field"
            )
        if type(row.evaluated) is not bool or not row.scenario_type:
            raise ValueError(f"run_id={row.run_id}: invalid evaluated/scenario_type")
        if row.derived_from and row.derived_from not in by_id:
            raise ValueError(
                f"run_id={row.run_id}: missing ancestor {row.derived_from!r}"
            )
    completed: set[str] = set()
    for row in rows:
        path: set[str] = set()
        current = row.run_id
        while current and current not in completed:
            if current in path:
                raise ValueError(f"run_id={row.run_id}: cyclic ancestry at {current}")
            path.add(current)
            current = by_id[current].derived_from
        completed.update(path)


def enumerate_stochastic(
    generation_spec: Mapping, *, unit_id_capacity: int
) -> tuple[ScenarioRow, ...]:
    """Enumerate configured draws × design points, with one evaluated root each.

    Args:
        generation_spec: Composed workflow settings; requested seed is not resolved.
        unit_id_capacity: Explicit generation-only capacity, independent of indicators.

    Returns:
        Immutable rows ordered by realization, then root, then design point.
    """
    if "run_historical" in generation_spec:
        raise ValueError("run_historical is retired and cannot select evaluated rows")
    realizations = generation_spec["n_realizations"]
    if type(realizations) is int and realizations == 0:
        raise EmptyScenarioSetError("n_realizations=0 gives an empty scenario set")
    realizations = _positive_integer(realizations, "n_realizations")
    perturbations = generation_spec["climate_perturbations"]
    temp_count = _positive_integer(perturbations["temp"]["n_levels"], "temp.n_levels")
    precip_count = _positive_integer(
        perturbations["precip"]["n_levels"], "precip.n_levels"
    )
    return stochastic_rows(
        realizations, temp_count * precip_count, unit_id_capacity=unit_id_capacity
    )


def stochastic_rows(
    n_realizations: int, st_num: int, *, unit_id_capacity: int
) -> tuple[ScenarioRow, ...]:
    """Mint stochastic rows from already resolved configured axis counts."""
    n_realizations = _positive_integer(n_realizations, "n_realizations")
    st_num = _positive_integer(st_num, "st_num")
    capacity = _positive_integer(unit_id_capacity, "unit_id_capacity")
    count = n_realizations * (st_num + 1)
    if count > capacity:
        raise UnitNamespaceCapacityError(
            f"capacity={capacity}, known runs={count}, bundles=not-evaluated; "
            f"minimum replacement unit_id_capacity={count}"
        )
    width, rlz_width, st_width = (
        len(str(capacity)),
        len(str(n_realizations)),
        len(str(st_num)),
    )
    rows = []
    for rlz in range(1, n_realizations + 1):
        root_id = f"run_{len(rows) + 1:0{width}d}"
        for st in range(st_num + 1):
            rows.append(
                ScenarioRow(
                    run_id=f"run_{len(rows) + 1:0{width}d}",
                    derived_from=root_id if st else "",
                    evaluated=True,
                    scenario_type="stochastic",
                    payload=(
                        ("rlz", f"{rlz:0{rlz_width}d}"),
                        ("st_id", f"{st:0{st_width}d}" if st else ""),
                    ),
                )
            )
    return tuple(rows)


def validate_stochastic(
    rows: Sequence[ScenarioRow],
    *,
    n_realizations: int,
    st_num: int,
    unit_id_capacity: int,
) -> None:
    """Refuse missing cross-product cells or altered same-draw root ancestry."""
    validate_forest(rows, unit_id_capacity=unit_id_capacity)
    expected = stochastic_rows(
        n_realizations, st_num, unit_id_capacity=unit_id_capacity
    )
    if tuple(rows) != expected:
        raise ValueError(
            "stochastic rows must equal the configured cross-product in normative "
            "order; every evaluated design point derives directly from its same-rlz root"
        )
