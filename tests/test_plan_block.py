"""The plan block: the run's rules, one per line, keyed on rule id.

What this pins, and why each case is here rather than left to a real run:
`onstart:` does NOT fire on `--dry-run` (probed 2026-09-05), so
`tests/test_cli.py` -- the cheap check that dry-runs all five entry points --
cannot see this block at all. Everything below is therefore the only automated
cover it has short of executing a workflow.
"""

from __future__ import annotations

import pytest

# The plan block moved to `console_style` with the rest of the Snakemake
# console tier (t2609171028): it is rendered by `_ConsoleHandler`, and
# nothing in a fingerprinted code closure reaches it.
from blueearth_cst.shared import console_style as cs

WF1_STATS = """Job stats:
job                             count
----------------------------  -------
all                                 1
snapshot_config                     1
write_gauge_index                  1
total                               3
"""


@pytest.fixture
def rules(monkeypatch):
    """Install a rule-number registry, as `rule_banner` would at parse time."""

    def install(mapping):
        monkeypatch.setattr(cs, "_RULE_NUMBERS", dict(mapping))

    return install


# --- the table parser -------------------------------------------------------


def test_the_stats_table_parses_to_counts():
    assert cs._run_info_counts(WF1_STATS) == {
        "all": 1,
        "snapshot_config": 1,
        "write_gauge_index": 1,
    }


def test_a_table_that_is_not_one_yields_nothing():
    """Fail open: an unparsable message must fall back, never guess."""
    assert cs._run_info_counts("Nothing resembling a table here.") == {}


# --- grouping ---------------------------------------------------------------


@pytest.mark.parametrize("target", cs._PLAN_EXCLUDED_RULES)
def test_a_target_rule_is_excluded_and_not_unlisted(rules, target):
    """A target does no work, but Snakemake counts it as a job.

    The block's rule count and Snakemake's job count differ by exactly this
    rule, on purpose -- so the exclusion has to be deliberate and pinned. A
    target registers no number, and must not read as an unlisted rule either.
    """
    rules({"a": "1.01"})
    assert [row[0] for row in cs._plan_rows({target: 1})] == ["1.01"]
    head, _ = cs._plan_lines({target: 1, "a": 1})
    assert "unlisted" not in head


def test_rules_sharing_a_number_collapse_to_one_row(rules):
    """WF0 builds one rule per climate source in a Python loop.

    Three rule objects, one number, one row -- with the jobs summed, not the
    first one taken.
    """
    rules(
        {
            "extract_climate_datasets_era5": "0.04",
            "extract_climate_datasets_chirps": "0.04",
            "extract_climate_datasets_eobs": "0.04",
        }
    )
    (row,) = cs._plan_rows(
        {
            "extract_climate_datasets_era5": 1,
            "extract_climate_datasets_chirps": 1,
            "extract_climate_datasets_eobs": 1,
        }
    )
    assert row == ("0.04", "extract_climate_datasets", 3)


def test_a_shared_number_with_no_common_prefix_keeps_a_real_name(rules):
    """Better a real rule name from the Snakefile than an empty truncation."""
    rules({"alpha": "9.01", "beta": "9.01"})
    (row,) = cs._plan_rows({})
    assert row[1] == "alpha"


def test_numbers_sort_lexicographically(rules):
    """`1.14` < `1.14b` < `1.15` as plain strings -- no version parser."""
    rules(
        {
            "run_wflow": "1.14",
            "export_simulation_tables": "1.14b",
            "plot_model_evaluation": "1.15",
            "write_run_metadata": "1.15b",
            "snapshot_config": "1.01",
        }
    )
    assert [row[0] for row in cs._plan_rows({})] == [
        "1.01",
        "1.14",
        "1.14b",
        "1.15",
        "1.15b",
    ]


# --- the head line ----------------------------------------------------------
#
# BARE: no `plan --` prefix and no indent. The opening block joins it to the
# workflow title (`wf1 build_model -- 2 of 19 rules to run, ...`) and the
# standalone block prefixes it; returning it decorated put the prefix in the
# middle of the title line.


def test_head_says_all_to_run_when_nothing_is_up_to_date(rules):
    rules({"a": "1.01", "b": "1.02"})
    head, _ = cs._plan_lines({"a": 1, "b": 1})
    assert head == "2 rules  |  all to run"


def test_head_names_both_halves_on_a_partial_run(rules):
    rules({"a": "1.01", "b": "1.02", "c": "1.03"})
    head, _ = cs._plan_lines({"a": 1})
    assert head == "1 of 3 rules to run  |  2 up to date"


def test_head_carries_the_job_count_only_when_something_fans_out(rules):
    """The line this replaced always carried it; here it is only news."""
    rules({"a": "1.01", "b": "1.02"})
    plain, _ = cs._plan_lines({"a": 1, "b": 1})
    assert "jobs" not in plain

    fanned, _ = cs._plan_lines({"a": 1, "b": 8})
    assert fanned.endswith("  |  9 jobs")


def test_a_single_rule_is_not_pluralized(rules):
    rules({"a": "1.01"})
    head, _ = cs._plan_lines({"a": 1})
    assert head == "1 rule  |  all to run"


# --- the rows ---------------------------------------------------------------


def test_the_gutter_marks_what_will_run(rules):
    """Flush left: the gutter opens at column 0, not two columns in."""
    rules({"a": "1.01", "b": "1.02"})
    _, rows = cs._plan_lines({"b": 1})
    texts = [text for text, _ in rows]
    assert texts[0].startswith("   1.01")
    assert texts[1].startswith(">  1.02")


def test_the_gutter_column_goes_when_no_rule_is_up_to_date(rules):
    """Not reserved and blank -- dropped, or it is indent on every row."""
    rules({"a": "1.01", "b": "1.02"})
    _, rows = cs._plan_lines({"a": 1, "b": 1})
    assert not any(">" in text for text, _ in rows)
    assert all(text.startswith("1.0") for text, _ in rows)


def test_every_running_rule_carries_its_count(rules):
    """Including the ones that are 1, which is what Snakemake's table shows."""
    rules({"perturb": "3.12", "seed": "3.11"})
    _, rows = cs._plan_lines({"perturb": 8, "seed": 1})
    assert [text.split()[-1] for text, _ in rows] == ["1", "8"]


def test_an_up_to_date_rule_has_an_empty_count_cell(rules):
    """Its count is UNKNOWN, not zero: it is absent from Snakemake's table.

    A `-` marked these until 2026-09-17 and was dropped as redundant -- the
    gutter already says which rows run, and it survives a pipe too.
    """
    rules({"a": "1.01", "b": "1.02"})
    _, rows = cs._plan_lines({"b": 4})
    assert rows[0][0] == "   1.01  a"
    assert rows[1][0] == ">  1.02  b  4"


def test_rows_report_whether_they_run(rules):
    """The caller colours whole lines from this flag; `_paint` never fields."""
    rules({"a": "1.01", "b": "1.02"})
    _, rows = cs._plan_lines({"b": 1})
    assert [running for _, running in rows] == [False, True]


def test_pre_dag_steps_are_listed_in_number_order(rules, monkeypatch):
    """Planning done outside Snakemake keeps its place in the numbered spine."""
    rules({"cached": "3.03", "generate": "3.07", "perturb": "3.08"})
    monkeypatch.setattr(cs, "_PRE_DAG_STEPS", {"plan": "3.04", "claim": "3.05"})

    head, rows = cs._plan_lines({"generate": 1, "perturb": 12})

    assert head == (
        "2 of 3 rules to run  |  1 up to date  |  2 completed before DAG execution  |  13 jobs"
    )
    assert rows == [
        ("   3.03  cached", False),
        ("   3.04  plan", False),
        ("   3.05  claim", False),
        (">  3.07  generate   1", True),
        (">  3.08  perturb   12", True),
    ]


def test_pre_dag_steps_alone_render_no_table(rules, monkeypatch):
    """Without a Snakemake rule there is no plan to extend."""
    rules({})
    monkeypatch.setattr(cs, "_PRE_DAG_STEPS", {"plan": "3.04"})
    assert cs._plan_lines({"whatever": 1}) is None


def test_no_row_carries_trailing_whitespace(rules):
    """A workflow without fan-out has no name column to pad."""
    rules({"a": "1.01", "bbbbbbbb": "1.02"})
    _, rows = cs._plan_lines({"a": 1})
    assert all(text == text.rstrip() for text, _ in rows)


def test_the_block_is_ascii(rules):
    """A Windows console is cp1252 and raises on typographic characters."""
    rules({"perturb": "3.12", "seed": "3.11"})
    head, rows = cs._plan_lines({"perturb": 8})
    ("\n".join([head] + [text for text, _ in rows])).encode("cp1252")


def test_an_empty_registry_falls_back(rules):
    """A workflow whose rules never called `rule_banner` gets Snakemake's line."""
    rules({})
    assert cs._plan_lines({"whatever": 1}) is None


def test_every_rule_up_to_date_still_renders(rules):
    """Reachable when only the excluded `all` job remains."""
    rules({"a": "1.01"})
    head, rows = cs._plan_lines({"all": 1})
    assert head == "1 rule  |  all up to date"
    assert not any(">" in text for text, _ in rows)


def test_a_rule_missing_from_the_ledger_is_declared(rules):
    """No silent caps.

    Every rule in all five Snakefiles calls `rule_banner` today, so this
    should never fire -- but a block that listed 18 of 19 rules and said
    nothing would be worse than one that admits the gap.
    """
    rules({"a": "1.01"})
    head, rows = cs._plan_lines({"a": 1, "undeclared_rule": 1})
    assert head.endswith("  |  1 unlisted")
    assert len(rows) == 1


def test_nothing_is_declared_when_the_ledger_is_complete(rules):
    rules({"a": "1.01"})
    head, _ = cs._plan_lines({"all": 1, "a": 1})
    assert "unlisted" not in head


# --- rules behind a checkpoint ---------------------------------------------


def test_a_rule_behind_a_checkpoint_shows_its_planned_count(rules, monkeypatch):
    """WF4's 4.07/4.08 are absent from the opening table because a checkpoint
    has not run yet, not because they are satisfied: they show the jobs they
    plan, as runnable rows, and the head counts them."""
    rules({"publish": "4.06", "plan": "4.07", "derive": "4.08", "done": "4.05"})
    monkeypatch.setattr(cs, "_AFTER_CHECKPOINT_RULES", {"plan": 1, "derive": 1})
    head, rows = cs._plan_lines({"publish": 1})
    assert head == "3 of 4 rules to run  |  1 up to date"
    texts = [text for text, _ in rows]
    assert texts[2].startswith(">") and texts[2].endswith("1")
    assert texts[3].startswith(">") and texts[3].endswith("1")
    assert "checkpoint" not in head + "".join(texts)


def test_a_checkpoint_rule_without_a_planned_count_is_left_empty(rules, monkeypatch):
    rules({"publish": "4.06", "plan": "4.07"})
    monkeypatch.setattr(cs, "_AFTER_CHECKPOINT_RULES", {"plan": None})
    head, rows = cs._plan_lines({"publish": 1})
    assert rows[1][0].rstrip().endswith("plan")
    assert "checkpoint" not in head


def test_a_checkpoint_rule_with_jobs_shows_its_count(rules, monkeypatch):
    """Once the checkpoint has resolved (a re-run), the table lists the jobs."""
    rules({"plan": "4.07"})
    monkeypatch.setattr(cs, "_AFTER_CHECKPOINT_RULES", {"plan": 1})
    head, rows = cs._plan_lines({"plan": 1})
    assert "after checkpoint" not in head
    assert rows[0][0].endswith("1")


def test_a_declared_rule_lists_even_when_its_banner_never_renders(rules, tmp_path):
    """WF3 reusing a collection defines no generation rules, so no `message:`
    renders their banner; declaring the rule is what puts it in the table."""
    rules({"publish": "3.09"})
    registry = cs.RuleRegistry(str(tmp_path / "logs"), str(tmp_path / "bench"))
    registry.logged("3.07", "generate")
    cs.declared_step("3.04", "snapshot")
    _, rows = cs._plan_lines({"publish": 1})
    texts = [text.split() for text, _ in rows]
    assert [t[1] if t[0] == ">" else t[0] for t in texts] == ["3.04", "3.07", "3.09"]
    assert texts[0] == ["3.04", "snapshot"] and texts[1] == ["3.07", "generate"]


def test_defer_rows_queues_a_captured_warning_for_the_run_header(monkeypatch, capsys):
    """A parse-time warning row waits for the header instead of printing above it."""
    monkeypatch.setattr(cs, "_DEFERRED_WARNINGS", [])
    # A header printed by an earlier test on this worker would drain the row.
    monkeypatch.setattr(cs, "_HEADER_PRINTED", False)
    cs.defer_rows(
        "13:27:21 - responses - WARNING - Duplicate discharge columns: Q_1 as Q_2\n"
        "Traceback-free stray line\n"
    )
    assert cs._DEFERRED_WARNINGS == [
        ("13:27:21", "responses", "Duplicate discharge columns: Q_1 as Q_2")
    ]
    assert capsys.readouterr().err == "Traceback-free stray line\n"
