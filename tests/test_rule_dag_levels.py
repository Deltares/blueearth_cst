"""Unit tests for dev/scripts/rule_dag_levels.py.

Pure unit tests -- no snakemake invocation, no project tree, so they run on a
bare checkout in CI. The DOT fixtures below are verbatim excerpts of what
snakemake 9.6.2 emitted for `Snakefile_climate_projections` on 2026-07-31.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# dev/scripts is not an importable package; load the module by path.
_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "dev" / "scripts" / "rule_dag_levels.py"
)
_spec = importlib.util.spec_from_file_location("rule_dag_levels", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
rdl = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = rdl
_spec.loader.exec_module(rdl)


# --- fixtures: real snakemake 9.6.2 output ------------------------------------

# `--rulegraph dot`, WF2. Ids are snakemake's own; the edge list is verbatim.
WF2_RULEGRAPH = """digraph snakemake_dag {
    graph[bgcolor=white, margin=0];
    0[label = "all", color = "0.00 0.6 0.85", style="rounded"];
    1[label = "derive_change_factors", color = "0.13 0.6 0.85", style="rounded"];
    2[label = "reduce_gcm_series", color = "0.60 0.6 0.85", style="rounded"];
    3[label = "extract_historical_climate", color = "0.20 0.6 0.85", style="rounded"];
    4[label = "fetch_gcm_slice", color = "0.27 0.6 0.85", style="rounded"];
    5[label = "plot_gcm_timeseries", color = "0.47 0.6 0.85", style="rounded"];
    6[label = "snapshot_config", color = "0.07 0.6 0.85", style="rounded"];
    7[label = "gather_raw_logs", color = "0.33 0.6 0.85", style="rounded"];
    8[label = "gather_series_logs", color = "0.40 0.6 0.85", style="rounded"];
    9[label = "gather_benchmarks", color = "0.53 0.6 0.85", style="rounded"];
    8 -> 0
    6 -> 0
    9 -> 0
    1 -> 0
    5 -> 0
    7 -> 0
    2 -> 1
    3 -> 1
    4 -> 2
    3 -> 2
    3 -> 4
    1 -> 5
    2 -> 5
    4 -> 7
    2 -> 8
    1 -> 9
    5 -> 9
}
"""

# `--dag dot`, WF2, abridged: a running fetch job with a wildcard in its label,
# an up-to-date (dashed) job, and a plain one.
WF2_DAG = """digraph snakemake_dag {
    0[label = "all", color = "0.00 0.6 0.85", style="rounded"];
    3[label = "extract_historical_climate", color = "0.20 0.6 0.85", style="rounded,dashed"];
    4[label = "fetch_gcm_slice\\nseries_key: cmip6_INM_INM-CM4-8_historical_r1i1p1f1", color = "0.27 0.6 0.85", style="rounded"];
    5[label = "reduce_gcm_series", color = "0.60 0.6 0.85", style="rounded,dashed"];
    6[label = "fetch_gcm_slice\\nseries_key: cmip6_INM_INM-CM5-0_ssp245_r1i1p1f1", color = "0.27 0.6 0.85", style="rounded"];
    4 -> 0
    3 -> 4
}
"""


# --- parse_dot ----------------------------------------------------------------


def test_parse_dot_reads_every_node_and_edge():
    nodes, edges = rdl.parse_dot(WF2_RULEGRAPH)
    assert len(nodes) == 10
    assert len(edges) == 17
    assert nodes[4].rule == "fetch_gcm_slice"
    # Direction: snakemake draws dependency -> dependent.
    assert (3, 4) in edges, "extract_historical_climate must point at fetch_gcm_slice"


def test_parse_dot_strips_the_wildcard_line_from_a_job_label():
    """A job label is `rule\\nwildcard: value`; only the first line names the rule."""
    nodes, _ = rdl.parse_dot(WF2_DAG)
    assert nodes[4].rule == "fetch_gcm_slice"
    assert nodes[6].rule == "fetch_gcm_slice"


def test_parse_dot_detects_the_dashed_up_to_date_style():
    nodes, _ = rdl.parse_dot(WF2_DAG)
    assert nodes[3].up_to_date is True, "dashed means snakemake will skip it"
    assert nodes[4].up_to_date is False
    assert nodes[0].up_to_date is False


def test_parse_dot_ignores_non_node_lines():
    nodes, edges = rdl.parse_dot("digraph d {\n graph[bgcolor=white];\n}\n")
    assert nodes == {} and edges == []


# --- topological_levels -------------------------------------------------------


def test_levels_take_the_longest_path_not_the_shortest():
    """A node reachable by a short AND a long chain belongs at the long one.

    This is the WF2 shape: plot_climate_proj_timeseries depends on both
    reduce_gcm_series (2) and derive_change_factors (3), so it is 4, not 3.
    """
    nodes = {n: rdl.Node(rule=str(n), up_to_date=False) for n in range(4)}
    # 0 -> 1 -> 2 -> 3 and a shortcut 0 -> 3
    levels = rdl.topological_levels(nodes, [(0, 1), (1, 2), (2, 3), (0, 3)])
    assert levels == {0: 0, 1: 1, 2: 2, 3: 3}


def test_a_cycle_is_reported_rather_than_silently_truncated():
    nodes = {n: rdl.Node(rule=f"r{n}", up_to_date=False) for n in range(3)}
    with pytest.raises(ValueError, match="not acyclic"):
        rdl.topological_levels(nodes, [(0, 1), (1, 2), (2, 0)])


def test_isolated_nodes_are_level_zero():
    nodes = {0: rdl.Node("lonely", False), 1: rdl.Node("also_lonely", False)}
    assert rdl.topological_levels(nodes, []) == {0: 0, 1: 0}


# --- rule_levels: the real WF2 topology ---------------------------------------


def test_wf2_rule_levels_match_the_measured_topology():
    assert rdl.rule_levels(WF2_RULEGRAPH) == {
        "snapshot_config": 0,
        "extract_historical_climate": 0,
        "fetch_gcm_slice": 1,
        "gather_raw_logs": 2,
        "reduce_gcm_series": 2,
        "derive_change_factors": 3,
        "gather_series_logs": 3,
        "plot_gcm_timeseries": 4,
        "gather_benchmarks": 5,
        "all": 6,
    }


def test_wf2_fetch_precedes_reduce_precedes_derive():
    """The ordering claim the helper exists to make, stated independently."""
    levels = rdl.rule_levels(WF2_RULEGRAPH)
    assert levels["fetch_gcm_slice"] < levels["reduce_gcm_series"]
    assert levels["reduce_gcm_series"] < levels["derive_change_factors"]
    assert levels["derive_change_factors"] < levels["plot_gcm_timeseries"]
    assert levels["all"] == max(levels.values())


# --- job_counts ---------------------------------------------------------------


def test_job_counts_split_runnable_from_up_to_date():
    assert rdl.job_counts(WF2_DAG) == {
        "all": (1, 0),
        "extract_historical_climate": (0, 1),
        "fetch_gcm_slice": (2, 0),
        "reduce_gcm_series": (0, 1),
    }


# --- format_table -------------------------------------------------------------


def test_format_table_orders_by_level_then_name():
    levels = rdl.rule_levels(WF2_RULEGRAPH)
    counts = {"fetch_gcm_slice": (9, 0), "reduce_gcm_series": (9, 0)}
    body = [line for line in rdl.format_table(levels, counts) if line.strip()]
    rules = [line.split()[1] for line in body[2:]]
    assert rules[0] in {"snapshot_config", "extract_historical_climate"}
    assert rules.index("fetch_gcm_slice") < rules.index("reduce_gcm_series")
    assert rules[-1] == "all"


def test_format_table_marks_a_rule_missing_from_the_rule_graph():
    """Should be impossible; visible rather than dropped if it ever happens."""
    lines = rdl.format_table({"a": 0}, {"a": (1, 0), "ghost": (3, 0)})
    assert any(line.startswith("    ?") and "ghost" in line for line in lines)


def test_format_table_shows_a_dash_for_zero():
    lines = rdl.format_table({"idle": 0}, {})
    assert any("idle" in line and line.rstrip().endswith("-") for line in lines)
