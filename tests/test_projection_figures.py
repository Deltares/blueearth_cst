# -*- coding: utf-8 -*-
"""Configured-horizon parsing contracts; suite artists test the folder set."""

import pytest

from blueearth_cst.projections import projection_figures as pf


class TestParseHorizonPeriod:
    def test_accepts_a_list_of_years(self):
        assert pf.parse_horizon_period([2070, 2090]) == (2070, 2090)

    def test_accepts_the_pre_r01_comma_string(self):
        """Pre-R01 configs wrote `"2070, 2090"`; such a project is still valid."""
        assert pf.parse_horizon_period("2070, 2090") == (2070, 2090)

    def test_rejects_a_period_that_is_not_a_pair(self):
        with pytest.raises(ValueError, match="two years"):
            pf.parse_horizon_period([2070, 2080, 2090])

    def test_rejects_non_integer_years(self):
        with pytest.raises(ValueError, match="integers"):
            pf.parse_horizon_period(["twenty-seventy", "2090"])

    def test_rejects_a_reversed_period(self):
        with pytest.raises(ValueError, match="must not exceed"):
            pf.parse_horizon_period([2090, 2070])


class TestSanitizeHorizonName:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("far", "far"),
            ("Far Future", "far_future"),
            ("mid_century", "mid_century"),
            ("2050s!!", "2050s"),
            ("  near  ", "near"),
        ],
    )
    def test_collapses_to_a_portable_slug(self, name, expected):
        assert pf.sanitize_horizon_name(name) == expected

    def test_rejects_a_name_with_nothing_portable_in_it(self):
        """Caught here rather than at the moment a figure cannot be written."""
        with pytest.raises(ValueError, match="no portable characters"):
            pf.sanitize_horizon_name("///")


class TestHorizonDirectory:
    def test_carries_the_years_not_only_the_label(self):
        """A redefined window must not overwrite the old figure in place."""
        assert pf.horizon_directory("far", [2070, 2090]) == "far_2070_2090"
        assert pf.horizon_directory("far", [2060, 2080]) == "far_2060_2080"
