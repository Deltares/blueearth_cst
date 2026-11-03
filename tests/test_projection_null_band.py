"""Independent OLS and Gaussian/F-distribution null-band falsifiers."""

import numpy as np
import pytest
from scipy.stats import f

from blueearth_cst.projections import projection_analysis as analysis


def _member(reference, future, mean=10.0, sd=1.0):
    return {
        "reference_years": tuple(reference),
        "future_years": tuple(future),
        "reference_mean": mean,
        "reference_sd": sd,
    }


def _independent_ratios(members, statistic, seed=9371, count=2000):
    rng = np.random.default_rng(seed)
    all_ratios = []
    for member in members:
        reference = np.asarray(member["reference_years"])
        future = np.asarray(member["future_years"])
        union = np.union1d(reference, future)
        values = rng.normal(
            member["reference_mean"], member["reference_sd"], size=(count, len(union))
        )
        estimates = []
        for years in (reference, future):
            selected = values[:, np.searchsorted(union, years)]
            design = np.column_stack([np.ones(len(years)), years - years[0]])
            coefficients = np.linalg.lstsq(design, selected.T, rcond=None)[0]
            residual = selected.T - design @ coefficients
            sd = np.sqrt((residual**2).mean(axis=0))
            estimates.append(
                sd / selected.mean(axis=1) if statistic == "cv_detrended" else sd
            )
        all_ratios.append(estimates[1] / estimates[0])
    return np.exp(np.log(all_ratios).mean(axis=0))


@pytest.mark.parametrize("n_ref,n_fut", [(30, 30), (29, 27)])
def test_disjoint_sd_band_matches_population_sd_f_oracle(n_ref, n_fut):
    member = _member(range(1985, 1985 + n_ref), range(2036, 2036 + n_fut))
    band = analysis.null_ratio_band([member])
    probability = [0.05, 0.95]
    correction = ((n_fut - 2) / n_fut) / ((n_ref - 2) / n_ref)
    oracle = np.sqrt(correction * f.ppf(probability, n_fut - 2, n_ref - 2))
    np.testing.assert_allclose(
        [band["null_lo"], band["null_hi"]], oracle, atol=0.01, rtol=0
    )
    if n_ref != n_fut:
        uncorrected = np.sqrt(f.ppf(probability, n_fut - 2, n_ref - 2))
        assert not np.allclose(oracle, uncorrected, atol=1e-4, rtol=0)


@pytest.mark.parametrize(
    "statistic,members",
    [
        ("std_detrended", [_member(range(1985, 2015), range(2008, 2038))]),
        ("cv_detrended", [_member(range(1985, 2015), range(2036, 2066), sd=1.0)]),
        ("cv_detrended", [_member(range(1985, 2015), range(2036, 2066), sd=4.0)]),
        (
            "cv_detrended",
            [
                _member(range(1985, 2015), range(2036, 2066), 10, 1),
                _member(range(1985, 2012), range(2036, 2065), 20, 5),
                _member(range(1985, 2015), range(2008, 2038), 30, 12),
            ],
        ),
        (
            "std_detrended",
            [
                _member(range(1985, 2015), range(2036, 2066), 10, 1),
                _member(range(1985, 2012), range(2036, 2065), 20, 5),
                _member(range(1985, 2015), range(2008, 2038), 30, 12),
            ],
        ),
        (
            "cv_detrended",
            [
                _member(
                    [y for y in range(1985, 2015) if y != 2012],
                    [y for y in range(2008, 2038) if y != 2010],
                    10,
                    4,
                )
            ],
        ),
        (
            "std_detrended",
            [
                _member(
                    [y for y in range(1985, 2015) if y not in (1988, 1994)],
                    [y for y in range(2036, 2066) if y not in (2040, 2053)],
                )
            ],
        ),
    ],
)
def test_band_covers_exact_heterogeneous_masked_statistic(statistic, members):
    band = analysis.null_ratio_band(members, statistic=statistic)
    values = _independent_ratios(members, statistic)
    coverage = ((values >= band["null_lo"]) & (values <= band["null_hi"])).mean()
    assert 0.88 <= coverage <= 0.92
    assert band["n_invalid"] == 0
    assert band["n_valid"] == 20_000


def test_null_cache_preserves_coordinates_and_invalid_cv_draw_counts():
    first = _member(
        [y for y in range(1985, 2015) if y != 2012],
        [y for y in range(2008, 2038) if y != 2010],
    )
    second = _member(
        [y for y in range(1985, 2015) if y != 1988],
        [y for y in range(2008, 2038) if y != 2020],
    )
    a = analysis.null_ratio_band([first])
    b = analysis.null_ratio_band([second])
    assert (a["null_lo"], a["null_hi"]) != (b["null_lo"], b["null_hi"])
    assert a["overlap"] == [5]
    assert b["overlap"] == [7]
    invalid = analysis.null_ratio_band(
        [_member(range(3), range(5, 8), 1, 10)], statistic="cv_detrended"
    )
    assert invalid["n_invalid"] > 0
    assert invalid["n_valid"] + invalid["n_invalid"] == 20_000
    with pytest.raises(ValueError, match="unique"):
        analysis.null_ratio_band([_member([1, 1, 2], [5, 6, 7])])
