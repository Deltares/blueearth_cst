"""WF3's climate catalog is per member, and nothing fans in over the sweep.

Rule 3.13 ``write_climate_data_catalog`` built ONE hydromt catalog naming every
member, and rule 3.14 read a single entry out of it. The entries differed only
in ``uri``, and the shape cost three things that had nothing to do with what a
catalog is for:

* **a fan-in barrier** — 3.13 declared every member's perturbed NC, so no member
  could be downscaled until every member had been perturbed;
* **peak transient disk** — those NCs are ``temp()``, and Snakemake cannot
  delete one until its last consumer is done, so all N had to coexist;
* **a straggler coupling** — one slow member held back the rest.

Rule 3.14 now writes its own one-entry catalog as a ``temp()`` output beside the
member's TOML. These tests pin that the barrier is gone and cannot come back by
someone re-adding an ``expand`` to 3.14's inputs; the CONTENT of the entry is
pinned in ``tests/test_prepare_climate_data_catalog.py``.

The catalog is still a catalog, not a path handed to hydromt: hydromt_wflow's
``setup_precip_forcing`` / ``setup_temp_pet_forcing`` forward to
``get_rasterdataset`` with no ``source_kwargs``, so a bare path resolves through
the fallback driver and silently loses ``preprocess=harmonise_dims`` and
``crs=4326``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

TESTDIR = Path(__file__).resolve().parent
SNAKEDIR = TESTDIR.parent
CONFIG_FN = TESTDIR / "snake_config_fixture.yml"

SNAKEFILE = "run_stress_test.smk"
RULE_NAME = "downscale_climate_realization"


def _parse_workflow(snakefile: str, config_path):
    """Parse a Snakefile in-process and return its ``Workflow``.

    Same helper, same pinning caveat, as ``test_region_rule.py``: the rules are
    built exactly as a real invocation builds them, and ``wf_api._workflow`` is
    private on the pinned Snakemake.
    """
    import snakemake.api as api

    with api.SnakemakeApi() as sa:
        wf_api = sa.workflow(
            resource_settings=api.ResourceSettings(cores=1),
            config_settings=api.ConfigSettings(configfiles=[Path(config_path)]),
            storage_settings=api.StorageSettings(),
            workflow_settings=api.WorkflowSettings(),
            snakefile=SNAKEDIR / snakefile,
            workdir=SNAKEDIR,
        )
        workflow = wf_api._workflow
        workflow.include(workflow.main_snakefile, overwrite_default_target=True)
        return workflow


@pytest.fixture(scope="module")
def workflow():
    return _parse_workflow(SNAKEFILE, CONFIG_FN)


@pytest.fixture(scope="module")
def downscale(workflow):
    return workflow.get_rule(RULE_NAME)


@pytest.mark.slow
@pytest.mark.workflow_contract
def test_no_rule_builds_a_catalog_over_the_whole_sweep(workflow):
    """The removed rule, by name and by artifact.

    Both, because either could come back on its own: a rule re-added under a new
    name would still fan in, and the old filename re-appearing under some other
    rule would still be a catalog nobody reads per member.
    """
    names = {rule.name for rule in workflow.rules}
    assert "write_climate_data_catalog" not in names

    declared = {str(path) for rule in workflow.rules for path in rule.output}
    assert not [p for p in declared if "data_catalog_run_stress_test" in p]


@pytest.mark.slow
@pytest.mark.workflow_contract
def test_the_member_catalog_sits_beside_the_member_toml(downscale):
    """One stem, three artifacts: the forcing, the run TOML, and the catalog."""
    outputs = [str(path) for path in downscale.output]
    toml = next(p for p in outputs if p.endswith(".toml"))
    catalog = next(p for p in outputs if p.endswith(".yml"))
    assert Path(catalog).parent == Path(toml).parent
    assert Path(catalog).stem == Path(toml).stem


@pytest.mark.slow
@pytest.mark.workflow_contract
def test_the_member_catalog_is_temporary(downscale):
    """It is scaffolding for one hydromt call, not a result of the run.

    Kept, it would put one file per member into a tree that already records the
    same two facts -- which member, and which forcing file -- in the member's
    own TOML and in its name.
    """
    catalog = next(path for path in downscale.output if str(path).endswith(".yml"))
    assert catalog in downscale.temp_output


@pytest.mark.slow
@pytest.mark.workflow_contract
def test_the_downscale_rule_reads_only_its_own_member(downscale):
    """The barrier regression test.

    Every input is either that member's own file (carrying the wildcards), a
    project-scope catalog, or the experiment-scope drift sentinel. An input
    naming a DIFFERENT member -- which is what an `expand` over the grid
    produces -- is the fan-in coming back.
    """
    member_token = "{rlz_num}"
    inputs = [str(path) for path in downscale.input]
    # Non-vacuity: an empty or wildcard-free input list would pass the loop
    # below while saying nothing, which is the failure mode this whole module
    # exists to prevent.
    assert any(member_token in text for text in inputs), inputs
    for path in downscale.input:
        text = str(path)
        if member_token in text:
            continue  # this member's own, whichever member that is
        assert "rlz_" not in Path(text).name, f"input names another member: {text}"
