"""Rule 0.06: the cross-source comparison table and figures.

The properties, and the first three are ones the rest of the suite cannot reach:

* **the ≥2-carrier filter** — a variable only one candidate carries gets no
  comparison figure, while the TABLE still lists every candidate. That split is
  the whole owner ruling (2026-08-17), and it is easy to collapse into one
  filter by accident;
* **common ground** — the sources are masked to the basin cells and clipped to
  the period they share before anything is derived. Both corrections are
  invisible in a rendered figure (two plausible lines either way), so they are
  asserted numerically here or nowhere;
* **the multi-source DAG direction** — ``tests/snake_config_fixture.yml`` sets
  no ``candidate_sources``, so ``test_cli`` and ``test_log_rules_contract`` both
  parse WF0 in its SINGLE-source shape and neither sees rule 0.06 at all. This
  module parses it with two sources, which is where the conditional rule, its
  declared outputs and its appended ``LOG_RULES`` label have to agree;
* the table's window column is the EXTRACTED span, not the published record;
* the figures render, and are named exactly as the rule declared them.

Hermetic: the stores are written here, so nothing needs a data mirror, a model
or the network.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from blueearth_cst.climate_analysis.climate_figures import CLIMATE_VARS, annual_series
from blueearth_cst.climate_analysis.compare_sources import (
    COMPARABLE_VARS,
    DISPLAY_HEADERS,
    FOOTNOTE_COLUMN,
    MISSING,
    TABLE_STEM,
    basin_cell_mask,
    compare_climate_sources,
    comparison_caveat,
    comparison_figure_names,
    comparison_outputs,
    comparison_variables,
    mutual_window,
    summarize_sources,
)

TESTDIR = Path(__file__).resolve().parent
SNAKEDIR = TESTDIR.parent
CONFIG_FN = TESTDIR / "snake_config_fixture.yml"

_START, _END = "2001-01-01", "2017-12-31"

#: Everything east of this is "outside the basin" in the fixture, and carries a
#: very different rainfall — so a mean taken over the whole extraction cannot
#: coincide with one taken over the basin cells.
_BASIN_EAST_EDGE = 9.30


def _store(
    path: Path,
    *,
    step: float,
    precip_scale: float,
    attrs: dict | None = None,
    start: str = _START,
    end: str = _END,
    buffer_cells: int = 2,
) -> Path:
    """A synthetic ``extract_historical.nc`` on its own grid resolution.

    Two fixture properties earn their keep:

    * the grid spans the basin PLUS ``buffer_cells`` in each direction, counted
      in cells as a real store's buffer is — so the coarse and fine grids cover
      physically different footprints, which is the defect the basin mask fixes;
    * rainfall east of :data:`_BASIN_EAST_EDGE` is three times the basin's, so
      including those cells moves the mean by an amount an assertion can see.

    Temperature is written even for a precipitation-only source, because a
    CHIRPS store really does carry era5's — the comparison must exclude it on
    the SOURCE NAME rather than on its absence.
    """
    time = pd.date_range(start, end, freq="D")
    lats = np.arange(0.0, 0.2 + step / 2, step)
    lons = np.arange(9.0, _BASIN_EAST_EDGE + step / 2, step)
    # The buffer, in cells — the wider the grid, the further it reaches.
    lons = np.append(lons, [lons[-1] + step * (i + 1) for i in range(buffer_cells)])
    lats = np.append(lats, [lats[-1] + step * (i + 1) for i in range(buffer_cells)])

    season = np.sin(2 * np.pi * time.dayofyear.values / 365.25)
    base = precip_scale * (1.6 + season)[:, None, None]
    outside = np.where(lons > _BASIN_EAST_EDGE + step / 2, 3.0, 1.0)[None, None, :]
    precip = base * np.ones((time.size, lats.size, lons.size)) * outside
    temp = 24.0 + 4.0 * season[:, None, None] * np.ones(
        (time.size, lats.size, lons.size)
    )
    ds = xr.Dataset(
        {
            "precip": (("time", "latitude", "longitude"), precip.astype("float32")),
            "temp": (("time", "latitude", "longitude"), temp.astype("float32")),
        },
        coords={"time": time, "latitude": lats, "longitude": lons},
        attrs=attrs or {},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    ds.close()
    return path


def _basin_cells(store: Path) -> Path:
    """The store's ``basin_cells.csv`` — the cells west of the basin edge.

    Written in the shape ``write_basin_cell_mask`` writes, since that is the
    file rule 0.06 consumes.
    """
    with xr.open_dataset(store) as ds:
        lats = [float(v) for v in ds["latitude"].values]
        lons = [float(v) for v in ds["longitude"].values if v <= _BASIN_EAST_EDGE]
    frame = pd.DataFrame(
        [(la, lo) for la in lats for lo in lons], columns=["latitude", "longitude"]
    )
    out = store.parent / "basin_cells.csv"
    frame.to_csv(out, index=False)
    return out


@pytest.fixture
def stores(tmp_path) -> dict:
    """An era5-like store and a precipitation-only chirps-like one."""
    return {
        "era5": _store(
            tmp_path / "era5" / "extract_historical.nc",
            step=0.25,
            precip_scale=3.0,
            attrs={
                "source_version": "ERA5 daily",
                "paper_ref": "Hersbach et al. (2019)",
                "source_url": "https://doi.org/10.24381/cds.bd0915c6",
                "notes": "Resampled by Deltares to daily frequency",
            },
        ),
        "chirps": _store(
            tmp_path / "chirps" / "extract_historical.nc",
            step=0.05,
            precip_scale=2.4,
            attrs={
                "source_version": "v2.0",
                "paper_ref": "Funk et al (2015)",
                "source_url": "https://www.chc.ucsb.edu/data/chirps",
            },
        ),
    }


@pytest.fixture
def cells(stores) -> dict:
    return {name: _basin_cells(Path(path)) for name, path in stores.items()}


# --- what gets compared -------------------------------------------------------


def test_only_variables_two_sources_carry_are_compared():
    """chirps is precipitation-only, so temp has ONE carrier and no figure."""
    assert comparison_variables(["era5", "chirps"]) == ("precip",)


def test_two_full_sources_would_compare_every_stored_variable():
    """The filter is a count, not a hardcoded 'precip only'."""
    assert comparison_variables(["era5", "era5"]) == COMPARABLE_VARS


def test_a_single_source_compares_nothing():
    assert comparison_variables(["era5"]) == ()


def test_pet_is_never_compared():
    """It is derived per source, not stored -- see the module docstring."""
    assert "pet" not in COMPARABLE_VARS
    assert all("pet" not in name for name in comparison_outputs(["era5", "era5"]))


def test_figure_names_follow_the_declared_scheme():
    assert comparison_figure_names(["precip"]) == [
        "comparison_precip_annual.png",
        "comparison_precip_monthly.png",
    ]


def test_unknown_variable_is_refused():
    with pytest.raises(ValueError, match="unknown variables"):
        comparison_figure_names(["runoff"])


# --- common ground: the basin mask -------------------------------------------


def test_mask_selects_exactly_the_declared_cells(stores, cells):
    with xr.open_dataset(stores["era5"]) as ds:
        mask = basin_cell_mask(ds, cells["era5"])
        assert mask is not None
        assert int(mask.values.sum()) == len(pd.read_csv(cells["era5"]))
        # The buffer cells are excluded, so the mask is a strict subset.
        assert int(mask.values.sum()) < ds["precip"].isel(time=0).size


def test_mask_changes_the_value_the_figure_plots(stores, cells):
    """The correction is worth making — the two domains do not agree.

    Through ``annual_series``, the derivation the figures use, so this is the
    number that actually reaches the panel rather than a proxy for it.
    """
    with xr.open_dataset(stores["era5"]) as ds:
        spec = CLIMATE_VARS["precip"]
        whole = float(annual_series(ds["precip"], spec).mean())
        basin = float(
            annual_series(
                ds.where(basin_cell_mask(ds, cells["era5"]))["precip"], spec
            ).mean()
        )
    assert basin < whole
    # The buffer is wetter by construction, so ignoring it is not a rounding
    # difference -- it is tens of percent.
    assert (whole - basin) / whole > 0.1


def test_two_grids_disagree_more_before_masking_than_after(stores, cells):
    """The apples-to-oranges case, stated as a number.

    Both fixtures carry the SAME basin rainfall shape and differ only in scale
    (3.0 vs 2.4) and resolution. Over the buffered extractions the coarse grid
    also reaches further into the wet strip, so the gap between them is
    distorted; over the basin cells it is the scale ratio and nothing else.
    """
    spec = CLIMATE_VARS["precip"]
    whole, basin = {}, {}
    for name, path in stores.items():
        with xr.open_dataset(path) as ds:
            whole[name] = float(annual_series(ds["precip"], spec).mean())
            masked = ds.where(basin_cell_mask(ds, cells[name]))
            basin[name] = float(annual_series(masked["precip"], spec).mean())
    assert basin["era5"] / basin["chirps"] == pytest.approx(3.0 / 2.4, rel=0.02)
    assert abs(whole["era5"] / whole["chirps"] - 3.0 / 2.4) > 0.02


def test_absent_or_unmatched_cells_fall_back_to_the_full_grid(stores, tmp_path):
    """A missing mask costs the correction, never the figure."""
    with xr.open_dataset(stores["era5"]) as ds:
        assert basin_cell_mask(ds, None) is None
        assert basin_cell_mask(ds, tmp_path / "absent.csv") is None
        elsewhere = tmp_path / "elsewhere.csv"
        pd.DataFrame({"latitude": [51.5], "longitude": [4.2]}).to_csv(
            elsewhere, index=False
        )
        assert basin_cell_mask(ds, elsewhere) is None


# --- common ground: the shared period -----------------------------------------


def test_mutual_window_is_the_intersection(tmp_path):
    paths = [
        _store(
            tmp_path / "a" / "extract_historical.nc",
            step=0.25,
            precip_scale=3.0,
            start="2001-01-01",
            end="2017-12-31",
        ),
        _store(
            tmp_path / "b" / "extract_historical.nc",
            step=0.25,
            precip_scale=3.0,
            start="2005-01-01",
            end="2020-12-31",
        ),
    ]
    opened = [xr.open_dataset(p) for p in paths]
    try:
        lower, upper = mutual_window(opened)
    finally:
        for ds in opened:
            ds.close()
    assert lower == pd.Timestamp("2005-01-01")
    assert upper == pd.Timestamp("2017-12-31")


def test_non_overlapping_records_have_no_mutual_window(tmp_path):
    paths = [
        _store(
            tmp_path / "a" / "extract_historical.nc",
            step=0.25,
            precip_scale=3.0,
            start="1990-01-01",
            end="1995-12-31",
        ),
        _store(
            tmp_path / "b" / "extract_historical.nc",
            step=0.25,
            precip_scale=3.0,
            start="2005-01-01",
            end="2010-12-31",
        ),
    ]
    opened = [xr.open_dataset(p) for p in paths]
    try:
        assert mutual_window(opened) is None
    finally:
        for ds in opened:
            ds.close()


def test_the_common_ground_reaches_every_renderer(tmp_path, monkeypatch):
    """The COMPOSITION, which the helper tests above cannot see.

    ``mutual_window`` and ``basin_cell_mask`` can both be correct while the
    plotting path forgets to apply one of them to one source, and the figure
    would still look right — shorter lines read as a shorter record. So the
    renderers are spied on and asked what they were actually handed.
    """
    import blueearth_cst.climate_analysis.compare_sources as module

    stores = {
        "era5": _store(
            tmp_path / "era5" / "extract_historical.nc",
            step=0.25,
            precip_scale=3.0,
            start="2001-01-01",
            end="2017-12-31",
        ),
        "chirps": _store(
            tmp_path / "chirps" / "extract_historical.nc",
            step=0.05,
            precip_scale=2.4,
            start="2005-01-01",
            end="2020-12-31",
        ),
    }
    cells = {name: _basin_cells(Path(path)) for name, path in stores.items()}

    calls = []
    original = dict(module._RENDERERS)

    def spy(kind):
        def wrapped(datasets, var, anchor, caveat):
            calls.append(
                {
                    "caveat": caveat,
                    "spans": {
                        name: (
                            pd.Timestamp(ds["time"].values.min()),
                            pd.Timestamp(ds["time"].values.max()),
                        )
                        for name, ds in datasets.items()
                    },
                    "cells": {
                        name: int(np.isfinite(ds["precip"].isel(time=0)).sum())
                        for name, ds in datasets.items()
                    },
                }
            )
            return original[kind](datasets, var, anchor, caveat)

        return wrapped

    for kind in module.COMPARISON_KINDS:
        monkeypatch.setitem(module._RENDERERS, kind, spy(kind))

    module.plot_comparison_figures(
        stores, tmp_path / "out", ("precip",), basin_cells=cells
    )

    assert len(calls) == len(module.COMPARISON_KINDS)
    for call in calls:
        # Every source clipped to the INTERSECTION, not to its own record.
        for span in call["spans"].values():
            assert span == (pd.Timestamp("2005-01-01"), pd.Timestamp("2017-12-31"))
        # ...and masked, so the buffer cells are NaN rather than averaged in.
        for name, kept in call["cells"].items():
            assert kept == len(pd.read_csv(cells[name]))
        assert "common period 2005-01-01 to 2017-12-31" in call["caveat"]
        assert "Basin cells only" in call["caveat"]


def test_caveat_states_the_domain_and_the_period():
    window = (pd.Timestamp("2005-01-01"), pd.Timestamp("2017-12-31"))
    masked = comparison_caveat(window, masked=True)
    assert "Basin cells only" in masked
    assert "common period 2005-01-01 to 2017-12-31" in masked
    unmasked = comparison_caveat(None, masked=False)
    assert "Full extraction grids" in unmasked
    assert "own extracted period" in unmasked


# --- the summary table --------------------------------------------------------


def test_table_is_five_columns_plus_the_footnote(stores):
    """Compact by ruling: Dataset, time step, window, grid size, reference."""
    table = summarize_sources(stores)
    assert list(table.columns) == [*DISPLAY_HEADERS, FOOTNOTE_COLUMN]
    assert list(DISPLAY_HEADERS) == [
        "source",
        "temporal_resolution",
        "time_window",
        "spatial_resolution",
        "reference",
    ]


def test_table_lists_every_candidate(stores):
    """The >=2 filter is about FIGURES; the table summarises all candidates."""
    assert list(summarize_sources(stores)["source"]) == ["era5", "chirps"]


def test_table_reports_the_extracted_window_as_one_cell(stores):
    table = summarize_sources(stores).set_index("source")
    assert table.loc["era5", "time_window"] == f"{_START} → {_END}"
    assert table.loc["era5", "temporal_resolution"] == "daily"
    # Each source keeps its OWN resolution: the coarser grid is a finding.
    assert table.loc["era5", "spatial_resolution"] == "0.25°"
    assert table.loc["chirps", "spatial_resolution"] == "0.05°"


def test_table_carries_the_catalog_provenance(stores):
    table = summarize_sources(stores).set_index("source")
    assert table.loc["chirps", "reference"] == "Funk et al (2015)"
    # No `notes` on the chirps entry, but the borrowed-fields caveat is added.
    assert "precipitation only" in table.loc["chirps", FOOTNOTE_COLUMN]
    assert "Deltares" in table.loc["era5", FOOTNOTE_COLUMN]


def test_missing_provenance_renders_rather_than_raising(tmp_path):
    """A locally staged catalog entry legitimately carries no reference."""
    bare = {
        "era5": _store(
            tmp_path / "bare" / "extract_historical.nc", step=0.25, precip_scale=3.0
        )
    }
    assert summarize_sources(bare).loc[0, "reference"] == MISSING


def test_provenance_falls_back_to_the_catalog(tmp_path):
    """A store that kept NO metadata still gets its Reference column.

    This is the real chirps case, not a hypothetical: that branch fetches one
    variable and calls ``.to_dataset()``, and the entry's metadata does not
    survive — measured 2026-08-17, the store's only attribute is
    ``region_bbox``. Reading the store alone blanks the provenance columns for
    exactly the precipitation-only sources a comparison exists to judge.
    """
    catalog = tmp_path / "catalog.yml"
    catalog.write_text(
        yaml.safe_dump(
            {
                "meta": {"version": "test"},
                "chirps": {
                    "data_type": "RasterDataset",
                    "uri": "meteo/chirps_{year}.nc",
                    "driver": {"name": "raster_xarray"},
                    "metadata": {
                        "crs": 4326,
                        "paper_ref": "Funk et al (2015)",
                        "source_version": "v2.0",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    bare = {
        "chirps": _store(
            tmp_path / "bare" / "extract_historical.nc", step=0.05, precip_scale=2.4
        )
    }
    assert summarize_sources(bare).loc[0, "reference"] == MISSING
    assert (
        summarize_sources(bare, data_sources=str(catalog)).loc[0, "reference"]
        == "Funk et al (2015)"
    )


def test_an_unreadable_catalog_costs_columns_not_the_table(tmp_path):
    """Provenance is never worth failing the rule over."""
    bare = {
        "era5": _store(
            tmp_path / "bare" / "extract_historical.nc", step=0.25, precip_scale=3.0
        )
    }
    table = summarize_sources(bare, data_sources=str(tmp_path / "absent.yml"))
    assert table.loc[0, "reference"] == MISSING


# --- the rendered outputs -----------------------------------------------------


def test_writes_exactly_what_the_rule_declares(stores, cells, tmp_path):
    out_dir = tmp_path / "comparison"
    written = compare_climate_sources(stores, out_dir, basin_cells=cells)
    assert [path.name for path in written] == comparison_outputs(list(stores))
    assert all(path.is_file() and path.stat().st_size > 0 for path in written)


def test_markdown_keeps_the_grid_compact_and_notes_below(stores, cells, tmp_path):
    out_dir = tmp_path / "comparison"
    compare_climate_sources(stores, out_dir, basin_cells=cells)
    text = (out_dir / f"{TABLE_STEM}.md").read_text(encoding="utf-8")
    header = next(line for line in text.splitlines() if line.startswith("| Dataset"))
    assert header.count("|") == len(DISPLAY_HEADERS) + 1
    for discarded in ("DOI", "Years", "Compared", "Mean annual"):
        assert discarded not in header
    # The free-text column is rendered BELOW the grid, not inside it.
    assert "Remarks" not in header
    assert "- **chirps** — " in text
    # The CSV keeps it as a column.
    csv = pd.read_csv(out_dir / f"{TABLE_STEM}.csv")
    assert FOOTNOTE_COLUMN in csv.columns
    assert len(csv) == len(stores)


# --- the DAG direction the fixture config cannot reach ------------------------


def _parse_workflow(config_path: Path):
    """Parse WF0 in-process; same private accessor as test_log_rules_contract."""
    import snakemake.api as api

    with api.SnakemakeApi() as sa:
        wf_api = sa.workflow(
            resource_settings=api.ResourceSettings(cores=1),
            config_settings=api.ConfigSettings(configfiles=[config_path]),
            storage_settings=api.StorageSettings(),
            workflow_settings=api.WorkflowSettings(),
            snakefile=SNAKEDIR / "analyze_climate.smk",
            workdir=SNAKEDIR,
        )
        workflow = wf_api._workflow
        workflow.include(workflow.main_snakefile, overwrite_default_target=True)
        return workflow


@pytest.fixture
def two_source_config(tmp_path) -> Path:
    cfg = yaml.safe_load(CONFIG_FN.read_text(encoding="utf-8"))
    cfg["workflows"]["analyze_climate"]["candidate_sources"] = ["chirps"]
    path = tmp_path / "snake_config_two_sources.yml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _rule(workflow, name):
    return next((r for r in workflow.rules if r.name == name), None)


def test_rule_is_absent_on_a_single_source_config():
    """With no candidate_sources, WF0 stays exactly what WF1 already draws."""
    assert _rule(_parse_workflow(CONFIG_FN), "compare_climate_sources") is None


def test_rule_declares_the_module_named_outputs_on_two_sources(two_source_config):
    rule = _rule(_parse_workflow(two_source_config), "compare_climate_sources")
    assert rule is not None, "rule 0.06 must be declared for a multi-source run"
    declared = sorted(Path(str(path)).name for path in rule.output)
    assert declared == sorted(comparison_outputs(["era5", "chirps"]))
    # Every output lands in the one comparison directory beside the stores.
    assert all(Path(str(path)).parent.name == "comparison" for path in rule.output)


def test_rule_takes_each_store_and_its_basin_cells(two_source_config):
    """The masking is carried by the DAG, not read behind Snakemake's back."""
    rule = _rule(_parse_workflow(two_source_config), "compare_climate_sources")
    names = [Path(str(path)).name for path in rule.input]
    assert names.count("extract_historical.nc") == 2
    assert names.count("basin_cells.csv") == 2


def test_the_appended_log_label_matches_the_rule(two_source_config):
    """Both halves of the conditional LOG_RULES entry, pinned together.

    ``test_log_rules_contract`` reads the LOG_RULES *literal* and parses on the
    single-source fixture, so neither half is visible to it.
    """
    rule = _rule(_parse_workflow(two_source_config), "compare_climate_sources")
    label = Path(str(rule.log[0])).name[: -len(".log")]
    assert label == "0.06_compare_climate_sources"
    text = (SNAKEDIR / "analyze_climate.smk").read_text(encoding="utf-8")
    assert re.search(
        r"LOG_RULES\.append\(\s*[\"']0\.06_compare_climate_sources[\"']\s*\)", text
    ), "the rule's log label must be appended to LOG_RULES for a multi-source run"
