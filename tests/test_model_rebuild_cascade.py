"""R7-1: re-firing build_wflow_model must rebuild the TOML-writing chain.

`hydrology_model/wflow_sbm.toml` is CREATED by rule 1.03 `build_wflow_model` and then
updated IN PLACE by rules 1.04-1.08 (waterbodies, gauges/outputs, runtime,
forcing) -- but only 1.03 declares it. Before this fix, anything that re-fired
1.03 alone left the toml stripped of every section the later rules add, and the
next wflow run died on the missing key. It bit three times during the R7
milestone; each recovery needed --forceall.

The obvious fix -- dropping the `ancient()` on staticmaps in 1.04/1.05 -- is
WRONG, and this module pins why: those rules commit writes back into
staticmaps.nc themselves (`mod.write()` / `mod.close()`), so a plain input edge
would make each re-trigger on its own execution forever.
"""

import re
import subprocess
from pathlib import Path

import pytest

SNAKEDIR = Path(__file__).resolve().parents[1]
SNAKEFILE = SNAKEDIR / "Snakefile_model_creation"
CFG = SNAKEDIR / "config" / "workflows" / "snake_config_model_test.yml"


def _rule_block(name: str) -> str:
    text = SNAKEFILE.read_text(encoding="utf-8")
    m = re.search(rf"^rule {name}:\n(.*?)(?=^rule |\Z)", text, re.S | re.M)
    assert m, f"rule {name} not found"
    return m.group(1)


def test_build_wflow_model_declares_a_completion_sentinel():
    """Only 1.03 may write it -- that is what makes it a safe trigger."""
    out = _rule_block("build_wflow_model")
    assert ".model_built" in out
    assert "touch(" in out, "the sentinel should use Snakemake's touch() output"
    # nothing else writes it
    text = SNAKEFILE.read_text(encoding="utf-8")
    assert text.count("touch(f\"{basin_dir}/.model_built\")") == 1


def test_the_rebuild_edge_is_not_ancient():
    """The whole point: this edge must fire when create_model re-runs."""
    block = _rule_block("add_reservoirs_lakes_glaciers")
    assert "model_built" in block, "1.04 must consume the sentinel"
    m = re.search(r"model_built\s*=\s*(.+)", block)
    assert m and "ancient(" not in m.group(1), (
        "the sentinel edge must NOT be ancient() -- ancient() suppresses "
        "exactly the trigger this fix exists to restore"
    )


def test_staticmaps_edges_stay_ancient():
    """Guard the reason the obvious fix is wrong: 1.04 and 1.05 mutate
    staticmaps.nc via mod.write()/mod.close(), so a plain input edge would
    re-trigger them on their own execution."""
    for rule in ("add_reservoirs_lakes_glaciers", "declare_wflow_outputs"):
        block = _rule_block(rule)
        m = re.search(r"basin_nc\s*=\s*(.+)", block)
        assert m and "ancient(" in m.group(1), (
            f"{rule}'s staticmaps input must stay ancient()"
        )


@pytest.mark.skipif(
    not (SNAKEDIR / "test_case" / "test_local" / "hydrology_model").is_dir(),
    reason="untracked test_case/test_local fixture tree not present",
)
@pytest.mark.workflow_contract
def test_rerunning_build_wflow_model_reschedules_the_whole_toml_chain():
    """THE REGRESSION. If build_wflow_model re-fires, every rule that writes into
    wflow_sbm.toml must come with it. Before the fix the DAG scheduled
    build_wflow_model and stopped, leaving the toml stripped of [output] and
    [input.forcing], and the next wflow run died on the missing key.

    Uses --forcerun rather than touching an input: mtime state is shared
    fixture state that other tests disturb, so a touch-based version passed in
    isolation and reported "Nothing to be done" inside the full suite. The
    property under test is "when build_wflow_model runs, does the chain follow?",
    which --forcerun expresses directly and deterministically.
    """
    res = subprocess.run(
        f"snakemake all -c 1 -s {SNAKEFILE} --configfile {CFG} "
        f"--dry-run --forcerun build_wflow_model",
        shell=True, capture_output=True, text=True, cwd=SNAKEDIR,
    )
    combined = (res.stdout or "") + (res.stderr or "")
    assert res.returncode == 0, combined[-2000:]
    assert "build_wflow_model" in combined, combined[-2000:]
    # `setup_runtime` was listed here until [R10-1] merged it into the forcing
    # rule. The entry outlived the rule: this case is fixture-gated, so no
    # worktree run could see that it now demands a rule the DAG cannot contain.
    for rule in ("add_reservoirs_lakes_glaciers", "declare_wflow_outputs",
                 "add_climate_forcing"):
        assert rule in combined, (
            f"{rule} was NOT rescheduled alongside build_wflow_model -- the toml "
            f"would be left stripped\n{combined[-2000:]}"
        )
