"""WF4 stochastic row grouping, independent of the WF3 provider inventory."""

from collections.abc import Sequence

from blueearth_cst.experiment.scenario_rows import ScenarioRow, validate_stochastic


def indicator_groups(
    rows: Sequence[ScenarioRow],
    *,
    n_realizations: int,
    st_num: int,
    unit_id_capacity: int,
) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...], dict[str, int]]:
    """Resolve same-design-point bundles and their reference members."""
    validate_stochastic(
        rows,
        n_realizations=n_realizations,
        st_num=st_num,
        unit_id_capacity=unit_id_capacity,
    )
    groups: dict[str, list[str]] = {}
    realizations = {}
    for row in rows:
        payload = dict(row.payload)
        groups.setdefault(payload["st_id"], []).append(row.run_id)
        realizations[row.run_id] = int(payload["rlz"])
    return (
        {key: tuple(members) for key, members in groups.items()},
        tuple(row.run_id for row in rows if not row.derived_from),
        realizations,
    )
