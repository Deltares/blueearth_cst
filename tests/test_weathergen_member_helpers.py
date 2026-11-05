"""The weathergen rules' own decisions, tested without weathergenr.

`blueearth_cst/weathergen/member_helpers.R` holds what generate_weather.R and
impose_climate_change.R decide on OUR side of the weathergenr seam. Inline, each
was reachable only after weathergenr loaded a netCDF, and two shipped broken
past a green suite: the basin-subset ids that failed rule 3.11 on every real
basin (2026-08-13) and the trailing-slash out_dir (2026-09-05). The first case
below replays the 3.11 shape -- cells 7 and 12 of a 20-cell store.

Cheap for the same reason `test_read_member_grid.py` is: the file sources
nothing, so each case is one `Rscript -e`. cwd is pinned to the repo root
because this repo's R scripts are sourced by repo-relative path.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "blueearth_cst" / "weathergen" / "member_helpers.R"

pytestmark = pytest.mark.skipif(
    shutil.which("Rscript") is None,
    reason="Rscript not on PATH (r-base is in the pixi env; run inside it)",
)

#: A 20-cell store on a 4 x 5 lattice, ids stamped 1..20 the way
#: `read_netcdf` stamps them, each cell's data frame tagged with its own id.
STORE = (
    "grid <- expand.grid(x = 10 + 0:4 * 0.25, y = -1 + 0:3 * 0.25); "
    "grid$id <- seq_len(nrow(grid)); "
    "ncdata <- list(grid = grid, "
    "data = lapply(grid$id, function(i) data.frame(cell = i))); "
)


def _r(body):
    result = subprocess.run(
        ["Rscript", "--vanilla", "-e", f'source("{SCRIPT.as_posix()}"); {body}'],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
    )
    return result


def _out(body):
    result = _r(body)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_basin_subset_renumbers_ids_to_index_its_data():
    # Cells 7 and 12 of 20: the exact shape that failed rule 3.11.
    out = _out(
        STORE + "mask <- data.frame(latitude = grid$y[c(7, 12)], "
        "longitude = grid$x[c(7, 12)]); "
        "b <- subset_basin_cells(ncdata, mask); "
        'cat(b$keep, "|", b$grid$id, "|", sapply(b$data, `[[`, "cell"))'
    )
    assert out.split("|") == ["7 12 ", " 1 2 ", " 7 12"]


def test_basin_subset_matches_coordinates_within_rounding():
    # The mask comes from a different producer; float noise below 1e-6 must not
    # drop a cell.
    out = _out(
        STORE + "mask <- data.frame(latitude = grid$y[3] + 1e-9, "
        "longitude = grid$x[3] - 1e-9); "
        "cat(subset_basin_cells(ncdata, mask)$keep)"
    )
    assert out == "3"


def test_basin_subset_without_id_column_is_left_without_one():
    out = _out(
        STORE + "ncdata$grid$id <- NULL; "
        "mask <- data.frame(latitude = grid$y[5], longitude = grid$x[5]); "
        'cat("id" %in% names(subset_basin_cells(ncdata, mask)$grid))'
    )
    assert out == "FALSE"


def test_basin_subset_matching_nothing_stops_naming_the_source():
    result = _r(
        STORE + "mask <- data.frame(latitude = 50, longitude = 50); "
        'subset_basin_cells(ncdata, mask, source_label = "store.nc")'
    )
    assert result.returncode != 0
    assert "matched no cell in store.nc" in result.stderr


@pytest.mark.parametrize(
    "root", ["exp/climate/weathergenr", "exp/climate/weathergenr//"]
)
def test_weathergen_paths_carry_no_double_slash(root):
    out = _out(f'p <- weathergen_paths("{root}"); cat(p$output, p$plots, sep = "\\n")')
    assert out.splitlines() == [
        "exp/climate/weathergenr/output",
        "exp/climate/weathergenr/evaluation/plots",
    ]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("out/rlz_1_st_02.nc", "rlz_1_st_02 rlz_1_st 02"),
        ("out/rlz_1_cst_2.nc", "rlz_1_cst_2 rlz_1_cst 2"),
        ("out/rlz_3_identity.nc", "rlz_3_identity rlz_3 identity"),
    ],
)
def test_output_stem_splits_at_the_last_underscore(path, expected):
    out = _out(f's <- split_output_stem("{path}"); cat(s$stem, s$prefix, s$suffix)')
    assert out == expected


def test_output_stem_without_underscore_stops():
    result = _r('split_output_stem("out/member.nc")')
    assert result.returncode != 0
    assert "carries no underscore" in result.stderr


def test_identity_grid_is_twelve_zero_change_months():
    out = _out(
        "g <- identity_member_grid(); "
        "cat(nrow(g), identical(g$month, 1:12), "
        "all(unlist(g[c('temp_change', 'precip_change', "
        "'precip_variance_change')]) == 0))"
    )
    assert out == "12 TRUE TRUE"


def test_percent_to_factor_keeps_zero_variance_at_identity():
    # A variance change of 0 % must be the identity multiplier, never 0.
    out = _out('cat(percent_to_factor(c(0, -30, 25, -50)), sep = " ")')
    assert out == "1 0.7 1.25 0.5"


def test_perturbation_row_flat_grid_omits_flat_variance():
    out = _out(
        "g <- identity_member_grid(); g$temp_change <- 1.5; "
        "g$precip_change <- -30; cat(format_perturbation(g))"
    )
    assert out == "temp +1.5 degC, precip -30.0%"


def test_perturbation_row_shows_ranges_and_varying_variance():
    out = _out(
        "g <- identity_member_grid(); g$temp_change <- seq(0, 2.2, length.out = 12); "
        "g$precip_variance_change <- 10; cat(format_perturbation(g))"
    )
    assert out == "temp +0.0..+2.2 degC, precip +0.0%, precip var +10.0%"
