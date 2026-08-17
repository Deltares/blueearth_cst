# -*- coding: utf-8 -*-
"""The WF2 figure-set contract — `blueearth_cst/projections/projection_figures.py`.

This module is the single definition every downstream declaration derives from
(the Snakefile's targets, rule 2.06's `figure_names` promise, rule 2.07's
outputs, the tree inventory, `check_baseline`). So the tests here are about the
SET and its ORDER, not about drawing: if this contract is wrong, every one of
those declarations is wrong in the same way and nothing else would notice.
"""

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
            ("Far Future", "far-future"),
            ("mid_century", "mid-century"),
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
        assert pf.horizon_directory("far", [2070, 2090]) == "far-2070-2090"
        assert pf.horizon_directory("far", [2060, 2080]) == "far-2060-2080"


class TestFigureRelativePaths:
    def test_single_horizon_set_omits_the_combined_cloud(self):
        """With one horizon the combined view would be the faceted one again."""
        assert pf.figure_relative_paths({"far": [2070, 2090]}) == [
            "overview/annual-precipitation.png",
            "overview/annual-temperature.png",
            "overview/change-factor-cloud.png",
            "windows/far-2070-2090/monthly-change-factors.png",
        ]

    def test_multi_horizon_set_adds_the_combined_cloud_and_one_window_each(self):
        assert pf.figure_relative_paths(
            {"near": [2040, 2060], "far": [2070, 2090]}
        ) == [
            "overview/annual-precipitation.png",
            "overview/annual-temperature.png",
            "overview/change-factor-cloud.png",
            "overview/change-factor-cloud-combined.png",
            "windows/near-2040-2060/monthly-change-factors.png",
            "windows/far-2070-2090/monthly-change-factors.png",
        ]

    def test_window_order_follows_the_config_not_the_alphabet(self):
        """`figure_names` is a params: value, so a reordering re-triggers 2.06."""
        far_first = pf.figure_relative_paths(
            {"far": [2070, 2090], "near": [2040, 2060]}
        )
        assert far_first[-2:] == [
            "windows/far-2070-2090/monthly-change-factors.png",
            "windows/near-2040-2060/monthly-change-factors.png",
        ]

    def test_rejects_an_empty_horizon_map(self):
        with pytest.raises(ValueError, match="no horizons configured"):
            pf.figure_relative_paths({})

    def test_rejects_horizons_that_collide_on_disk(self):
        """Distinct names, same slug and years -- one figure would clobber the other."""
        with pytest.raises(ValueError, match="duplicate paths"):
            pf.figure_relative_paths({"far": [2070, 2090], "FAR!": [2070, 2090]})

    def test_paths_are_relative_and_use_forward_slashes(self):
        """They are joined onto `plots/` by the Snakefile and by the producers."""
        for path in pf.figure_relative_paths(
            {"near": [2040, 2060], "far": [2070, 2090]}
        ):
            assert not path.startswith("/")
            assert "\\" not in path


class TestMonthlyFigurePaths:
    def test_maps_each_horizon_name_to_its_own_window_path(self):
        assert pf.monthly_figure_paths({"near": [2040, 2060], "far": [2070, 2090]}) == {
            "near": "windows/near-2040-2060/monthly-change-factors.png",
            "far": "windows/far-2070-2090/monthly-change-factors.png",
        }

    def test_agrees_with_the_window_entries_of_the_full_set(self):
        """The producer and the Snakefile must not disagree about which window
        a figure describes -- both derive from `horizon_directory`."""
        horizons = {"near": [2040, 2060], "far": [2070, 2090]}
        from_set = [
            p for p in pf.figure_relative_paths(horizons) if p.startswith("windows/")
        ]
        assert sorted(pf.monthly_figure_paths(horizons).values()) == sorted(from_set)
