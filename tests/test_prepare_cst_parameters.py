"""Unit tests for prepare_cst_parameters.prep_cst_parameters (R5 §8).

Drives the lookup generation on a synthetic in-memory config written to a
tmp_path YAML, with lookup_fn=None so the function writes
``stress_test_lookup.csv`` into the config's directory. Uses only
pandas/numpy/yaml — the function is already import-clean (guarded), no heavy-dep
stub, no sys.modules pollution risk.
"""

import glob
import math

import numpy as np
import pandas as pd
import pytest
import yaml

from blueearth_cst.experiment.prepare_cst_parameters import (
    LOOKUP_COLUMNS,
    MULTIPLIER_DOMAIN,
    MultiplierDomainError,
    prep_cst_parameters,
    refuse_out_of_domain_multipliers,
)

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _twelve(v):
    return [float(v)] * 12


def _write_cfg(
    tmp_path,
    *,
    temp_step=1,
    precip_step=2,
    var_min=1.0,
    var_max=1.0,
    precip_min=0.7,
    precip_max=1.3,
):
    """Write a synthetic snake config and return its path (str)."""
    cfg = {
        "workflows": {
            "run_stress_test": {
                "stress_test": {
                    "temp": {
                        "step_num": temp_step,
                        "mean": {"min": _twelve(0.0), "max": _twelve(3.0)},
                    },
                    "precip": {
                        "step_num": precip_step,
                        "mean": {
                            "min": _twelve(precip_min),
                            "max": _twelve(precip_max),
                        },
                        "variance": {"min": _twelve(var_min), "max": _twelve(var_max)},
                    },
                }
            }
        }
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


def _read_lookup(tmp_path):
    """Read the written lookup the way WG-2 requires: `st_id` as TEXT."""
    return pd.read_csv(tmp_path / "stress_test_lookup.csv", dtype={"st_id": str})


def _level(v):
    """The grid level a member carries — the module's own quantization rule."""
    return float(str(np.float32(v)))


def test_seed_like_grid_shape_and_endpoints(tmp_path):
    """temp step 1 x precip step 2 -> 6 members x 12 months, correct endpoints."""
    cfg_path = _write_cfg(tmp_path, temp_step=1, precip_step=2)
    prep_cst_parameters(cfg_path)

    df = _read_lookup(tmp_path)
    assert list(df.columns) == list(LOOKUP_COLUMNS)
    assert len(df) == 6 * 12  # (1+1) * (2+1) members, twelve months each
    assert df["st_id"].nunique() == 6

    # temp spans [0, 3] degC; precip spans [-30, +30] PERCENT across the grid.
    assert df["temp_change"].min() == pytest.approx(0.0)
    assert df["temp_change"].max() == pytest.approx(3.0)
    assert df["precip_change"].min() == pytest.approx(-30.0, abs=1e-4)
    assert df["precip_change"].max() == pytest.approx(30.0, abs=1e-4)


def test_every_member_carries_the_twelve_calendar_months(tmp_path):
    """WG-2: the (st_id, month) grid is complete and duplicate-free."""
    cfg_path = _write_cfg(tmp_path, temp_step=2, precip_step=3)  # 12 members
    prep_cst_parameters(cfg_path)

    df = _read_lookup(tmp_path)
    for st_id, member in df.groupby("st_id"):
        assert list(member["month"]) == list(range(1, 13)), st_id
    assert not df.duplicated(subset=["st_id", "month"]).any()
    # Sorted by (st_id, month), which the padding makes lexical == numeric.
    assert df[["st_id", "month"]].equals(
        df.sort_values(["st_id", "month"]).reset_index(drop=True)[["st_id", "month"]]
    )


def test_precip_variance_grid_uses_max_endpoint(tmp_path):
    """The precip_variance grid spans up to variance.max (t260720a, fixed).

    Regression guard for the max-reads-min bug: prepare_cst_parameters once read
    variance['min'] into the max endpoint, collapsing a non-degenerate range
    (min=1.0, max=1.5) to [1.0, 1.0]. With the fix the grid max is variance.max
    — now read as a PERCENT, so 1.5 is +50.
    """
    cfg_path = _write_cfg(
        tmp_path, temp_step=1, precip_step=1, var_min=1.0, var_max=1.5
    )
    prep_cst_parameters(cfg_path)
    assert _read_lookup(tmp_path)["precip_variance_change"].max() == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# R11 P2 — zero-padded member ids (C27), now carried by the lookup's key column
# ---------------------------------------------------------------------------
#
# The tracked test config has ST_NUM = 6, so its width is 1 and NOTHING pads.
# That is correct per C27 -- st_1..st_6 already sort correctly -- but it means
# the fixture and the baseline cannot exercise padding at all. These cases carry
# that load: every padded assertion below uses a grid of ten or more.


def test_ids_are_unpadded_below_ten(tmp_path):
    """C27: width comes from the COUNT, so a small grid is not padded.

    Padding a 6-member grid would move every member token on every existing
    project to fix an ordering problem it does not have.
    """
    cfg_path = _write_cfg(tmp_path, temp_step=1, precip_step=2)  # 2 * 3 = 6
    prep_cst_parameters(cfg_path)

    ids = sorted(set(_read_lookup(tmp_path)["st_id"]))
    assert ids == [str(m) for m in range(1, 7)]


def test_ids_pad_once_the_count_reaches_ten(tmp_path):
    """C27: 12 members -> width 2, and LEXICAL order now matches RUN order.

    The falsifier is the sort: unpadded, `sorted()` yields 1, 10, 11, 12, 2 ...
    which is the whole reason this padding exists. Under the lookup it matters
    more, not less: `st_id` is now the JOIN KEY between this table and the
    indicator tables, so a width disagreement is a silently missing join rather
    than a mis-sorted listing.
    """
    cfg_path = _write_cfg(tmp_path, temp_step=2, precip_step=3)  # 3 * 4 = 12
    prep_cst_parameters(cfg_path)

    ids = sorted(set(_read_lookup(tmp_path)["st_id"]))
    assert ids == [f"{m:02d}" for m in range(1, 13)]
    assert ids == sorted(ids, key=int)  # lexical == numeric, which is the property


def test_st_0_has_no_row(tmp_path):
    """D4: the table is the PARAMETER GRID, and the baseline has no parameters.

    Its absence is load-bearing rather than tidy — it is what makes "not on the
    surface" a structural fact instead of a convention, so a consumer joining
    results to the lookup cannot place `st_0` on the surface by accident. An
    all-zero row would be indistinguishable from an identity member's while
    denoting a differently-processed climate.
    """
    cfg_path = _write_cfg(tmp_path, temp_step=2, precip_step=3)  # 12 members
    prep_cst_parameters(cfg_path)

    ids = set(_read_lookup(tmp_path)["st_id"])
    assert "00" not in ids and "0" not in ids
    assert ids == {f"{m:02d}" for m in range(1, 13)}


def test_units_are_percent_not_multipliers(tmp_path):
    """S1/D3: a 1.3 mean factor is +30.0 here, not 1.3.

    Stated so a future edit cannot quietly switch back. The R side reconstructs
    `1 + p/100`, so a table that carried multipliers would be read as a +30 000%
    perturbation rather than +30%.
    """
    cfg_path = _write_cfg(tmp_path, temp_step=1, precip_step=2)
    prep_cst_parameters(cfg_path)

    df = _read_lookup(tmp_path)
    assert df["precip_change"].abs().max() > 1.0
    # A flat variance vector of 1.0 is ZERO percent, not 1.0 — the conversion is
    # a rule over the percent columns, and omitting it here would hand the
    # generator a variance FACTOR of zero on every shipped config.
    assert (df["precip_variance_change"] == 0.0).all()


def test_a_third_stress_axis_refuses_naming_c28(tmp_path):
    """C28's second obligation: a new dimension REFUSES, never silently drops.

    A third axis that merely went unrecorded would leave a lookup describing a
    different experiment than the one that ran. Merging the artifacts removed
    the SHAPE barrier -- a new parameter is now just a column -- and this is the
    contract barrier that deliberately survives it.
    """
    cfg_path = _write_cfg(tmp_path, temp_step=1, precip_step=2)
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    cfg["workflows"]["run_stress_test"]["stress_test"]["wind"] = {
        "step_num": 1,
        "mean": {"min": _twelve(0.0), "max": _twelve(1.0)},
    }
    open(cfg_path, "w", encoding="utf-8").write(yaml.safe_dump(cfg))

    with pytest.raises(ValueError, match="C28"):
        prep_cst_parameters(cfg_path)


# ---------------------------------------------------------------------------
# V20 — the reconstructed MULTIPLIER, and nothing downstream of it
# ---------------------------------------------------------------------------
#
# The bound is on what the generator receives: `1 + p/100` must land within one
# `float64` ulp of the `float32` grid level. It is NOT a bound on any indicator
# -- between the multiplier and a metric sit a quantile mapping, wet-day
# thresholds, caps and floors, and a distributed hydrological model, none of
# which is Lipschitz in the forcing with a constant anyone has measured.


def _reconstruct(percent):
    """The R side's inverse, spelled as D25 pins it."""
    return 1 + percent / 100


@pytest.mark.parametrize(
    "precip_min, precip_max, precip_step",
    [
        (0.6, 1.4, 3),  # non-round: levels 0.6, 0.866667, 1.133333, 1.4
        (0.5, 1.5, 4),  # anchored at the domain FLOOR
        (0.7, 1.3, 2),  # the shipped grid, which reconstructs exactly
    ],
)
def test_reconstruction_is_within_one_ulp(
    tmp_path, precip_min, precip_max, precip_step
):
    """V20: the round trip is bounded across the admitted domain, not just on
    the shipped configs.

    The shipped grids are in D25's exactly-invertible set, which is precisely
    why the baseline comparison at step 7 cannot observe the conversion residual
    at all — this case is what covers that gap.
    """
    cfg_path = _write_cfg(
        tmp_path,
        temp_step=1,
        precip_step=precip_step,
        precip_min=precip_min,
        precip_max=precip_max,
    )
    prep_cst_parameters(cfg_path)

    levels = {
        _level(v)
        for v in np.linspace(
            _twelve(precip_min), _twelve(precip_max), precip_step + 1, axis=1
        ).ravel()
    }
    checked = 0
    for percent in set(_read_lookup(tmp_path)["precip_change"]):
        got = _reconstruct(percent)
        nearest = min(levels, key=lambda lv: abs(lv - got))
        assert abs(got - nearest) <= math.ulp(nearest), (
            f"{percent} reconstructed to {got!r}, "
            f"{abs(got - nearest) / math.ulp(nearest):.1f} ulps from {nearest!r}"
        )
        checked += 1
    assert checked, "no percent values were reconstructed"


def test_reconstruction_holds_across_the_percent_binade_crossings():
    """V20, widened: dense sweeps where the error is ATTAINED, not slack.

    Random draws inside the domain are not an acceptable substitute. The bound
    fails DOWNWARD only, at the first percent-binade crossing that `ulp(level)`
    no longer keeps up with, so the levels that matter are the ones where
    `|percent|` crosses a power of two — near 0.5, 0.68, 1.32, 1.64 and 2.0.
    Each is swept through consecutive `float32` values in both directions.
    """
    floor, _ = MULTIPLIER_DOMAIN
    for anchor in (0.5, 0.68, 1.32, 1.64, 2.0):
        value = np.float32(anchor)
        for _ in range(64):  # step down through consecutive float32 values
            value = np.nextafter(value, np.float32(0.0), dtype=np.float32)
        for _ in range(128):
            level = float(str(value))
            if level >= floor:
                got = _reconstruct(level * 100 - 100)
                assert abs(got - level) <= math.ulp(level), (
                    f"level {level!r} near anchor {anchor}: reconstructed {got!r}, "
                    f"{abs(got - level) / math.ulp(level):.1f} ulps out"
                )
            value = np.nextafter(value, np.float32(1e9), dtype=np.float32)


def test_the_bound_does_fail_below_the_domain():
    """The domain is a refusal because the claim is FALSE outside it.

    Recorded as an executable fact rather than a remark: without this, a later
    reader could reasonably conclude the floor was conservative and relax it.
    """
    level = _level(0.013596006)
    got = _reconstruct(level * 100 - 100)
    assert abs(got - level) > 10 * math.ulp(level)


# ---------------------------------------------------------------------------
# V23 — the domain guard, refusing before the DAG is built
# ---------------------------------------------------------------------------


def _stress_cfg(**kwargs):
    return {
        "temp": {"mean": {"min": _twelve(0.0), "max": _twelve(3.0)}},
        "precip": {
            "mean": {"min": _twelve(kwargs.get("mean_min", 0.7)), "max": _twelve(1.3)},
            "variance": {
                "min": _twelve(kwargs.get("var_min", 1.0)),
                "max": _twelve(kwargs.get("var_max", 1.0)),
            },
        },
    }


@pytest.mark.parametrize(
    "key, cfg",
    [
        ("precip.mean", _stress_cfg(mean_min=0.4)),
        ("precip.variance", _stress_cfg(var_min=0.2)),
    ],
)
def test_multiplier_domain_refused(key, cfg):
    """V23: BOTH percent-converted keys carry the domain, not just `mean`.

    A domain covering `mean` alone would re-create, one layer up, exactly the
    defect D3 corrected: the conversion is a rule over the percent COLUMNS
    rather than a formula for one of them.
    """
    with pytest.raises(MultiplierDomainError) as excinfo:
        refuse_out_of_domain_multipliers(cfg)
    assert key in str(excinfo.value)
    assert "0.5" in str(excinfo.value)


def test_temp_carries_no_domain():
    """`temp` is additive degC and crosses unconverted, so it cannot be out of
    bound — refusing a negative cooling scenario would be a real regression."""
    cfg = _stress_cfg()
    cfg["temp"]["mean"]["min"] = _twelve(-2.0)
    refuse_out_of_domain_multipliers(cfg)  # must not raise


@pytest.mark.parametrize(
    "config_path",
    sorted(glob.glob(str(REPO_ROOT / "test_case" / "snake_config_*.yml")))
    + [str(REPO_ROOT / "config" / "templates" / "snake_config.template.yml")],
)
def test_shipped_configs_are_inside_the_domain(config_path):
    """V23's other half: the guard must not refuse anything we ship.

    A refusal that fires on the seeds would make every `--dry-run` in the repo
    fail, so this is the case that keeps the guard honest rather than merely
    strict.
    """
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    stress_test_cfg = cfg["workflows"]["run_stress_test"]["stress_test"]
    refuse_out_of_domain_multipliers(stress_test_cfg)
