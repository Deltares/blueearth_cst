"""R07 B1: the three ``extract_climate_datasets`` declarations are ONE rule.

The store producer is declared as ``extract_climate_datasets`` in
``build_model.smk`` (rule 1.04) and ``generate_scenarios.smk`` (rule 3.08), both
from the same ``snake_utils.climate_store_rule`` object.
``analyze_projections.smk`` declares it NOT AT ALL — ADR 0003 removed it, and
``test_wf2_declares_no_store_and_no_extraction`` is what keeps it removed.
``analyze_climate.smk`` is the third declarer but generates one rule per
candidate source rather than copying the shared one; see the block above
``_wf0_rule`` for why, and the two ``test_wf0_*`` cases for the weaker
invariant that replaces byte-identity there.

Nothing in the rule grammar enforces that the two byte-identical declarations
stay identical, and a per-workflow difference re-creates the wf1<->wf3
re-extraction oscillation the design forbids (P2(b), ext1-02/ext2-01). This
module is the enforcement: it parses the workflows in-process and compares the
**full normalized contract** — rule name, script, input set, outputs, params,
**and every content- or execution-affecting directive**.

Two properties, deliberately separate:

* ``test_ruleinfo_field_universe_is_fully_bucketed`` — DENY BY DEFAULT. Every
  field of the pinned Snakemake's ``RuleInfo`` must be classified into exactly
  one of three buckets: compared, allowed-local (``message``/``log``/
  ``benchmark`` only), or structurally-irrelevant-with-a-written-reason. A
  Snakemake upgrade that adds a directive fails HERE, loudly, instead of
  silently widening the hole. This is what makes the enumeration below a check
  on the derivation rather than its only source.
* ``test_declarations_are_identical`` — the comparison itself, over the
  **effective built-rule state**, not the source text.

``message``/``log``/``benchmark`` are the only permitted local differences: none
is content-determining and none participates in a rerun trigger (Snakemake
records the log list but compares code, input, params, mtime and software-env).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from blueearth_cst.shared.config_composition import load_composed_config  # noqa: E402
from tests.conftest import parse_workflow, write_config  # noqa: E402

SNAKEDIR = Path(__file__).resolve().parents[1]
CONFIG_FN = Path(__file__).resolve().parent / "project_config_fixture.yml"
RULE_NAME = "extract_climate_datasets"

#: Sentinel for "the built Rule carries no such attribute on this Snakemake".
#: Compared as a value, so absent-on-both is equal and absent-on-one fails.
_ABSENT = "<<absent>>"


_parse_workflow = parse_workflow


# --- normalizers --------------------------------------------------------------


def _iofile_signature(namedlist):
    """(positional paths, sorted keyword->path) for an input/output namedlist.

    Input functions are rebuilt on every parse, so they compare by name.
    """

    def plain(value):
        return f"<callable {value.__name__}>" if callable(value) else str(value)

    return (
        tuple(plain(item) for item in namedlist),
        tuple(sorted((key, plain(value)) for key, value in namedlist.items())),
    )


def _params_signature(params):
    """Sorted keyword->repr for a params namedlist (values are not all str)."""
    return (
        tuple(repr(item) for item in params),
        tuple(sorted((key, repr(value)) for key, value in params.items())),
    )


def _resources_signature(resources):
    """Sorted resource items; callables (e.g. ``tmpdir``) collapse to a marker."""
    return tuple(
        sorted(
            (key, "<callable>" if callable(value) else repr(value))
            for key, value in dict(resources or {}).items()
        )
    )


def _plain(value):
    return _ABSENT if value is _ABSENT else repr(value)


# --- the three buckets --------------------------------------------------------
# Keys are RuleInfo field names (snakemake.ruleinfo.RuleInfo.__init__), so the
# universe test below can assert the buckets cover it exactly.

#: Compared between the two declarations. Values map a RuleInfo field to the
#: effective state it produces on a built Rule / on the Workflow.
_COMPARED = {
    "name": lambda wf, rule: rule.name,
    "input": lambda wf, rule: _iofile_signature(rule.input),
    "output": lambda wf, rule: _iofile_signature(rule.output),
    "params": lambda wf, rule: _params_signature(rule.params),
    "script": lambda wf, rule: _plain(rule.script),
    "shellcmd": lambda wf, rule: _plain(rule.shellcmd),
    "norun": lambda wf, rule: _plain(rule.norun),
    "docstring": lambda wf, rule: _plain(rule.docstring),
    "conda_env": lambda wf, rule: _plain(rule.conda_env),
    "container_img": lambda wf, rule: _plain(rule.container_img),
    "is_containerized": lambda wf, rule: _plain(rule.is_containerized),
    "env_modules": lambda wf, rule: _plain(rule.env_modules),
    "wildcard_constraints": lambda wf, rule: tuple(
        sorted((k, str(v)) for k, v in dict(rule.wildcard_constraints or {}).items())
    ),
    # `threads:` lands in the rule's `_cores` resource, not on a `threads` attr.
    "threads": lambda wf, rule: _plain(dict(rule.resources or {}).get("_cores")),
    "shadow_depth": lambda wf, rule: _plain(rule.shadow_depth),
    "resources": lambda wf, rule: _resources_signature(rule.resources),
    "priority": lambda wf, rule: _plain(rule.priority),
    "retries": lambda wf, rule: _plain(getattr(rule, "restart_times", _ABSENT)),
    "group": lambda wf, rule: _plain(rule.group),
    "notebook": lambda wf, rule: _plain(rule.notebook),
    "wrapper": lambda wf, rule: _plain(rule.wrapper),
    "template_engine": lambda wf, rule: _plain(rule.template_engine),
    "cwl": lambda wf, rule: _plain(rule.cwl),
    # Workflow-level state, not rule attributes.
    "cache": lambda wf, rule: _plain(wf.cache_rules.get(rule.name)),
    "handover": lambda wf, rule: _plain(getattr(rule, "is_handover", _ABSENT)),
    "default_target": lambda wf, rule: _plain(wf.default_target == rule.name),
    "localrule": lambda wf, rule: _plain(
        rule.name in set(getattr(wf, "_localrules", ()) or ())
    ),
}

#: The ONLY directives permitted to differ between the two declarations.
_ALLOWED_LOCAL = {"message", "log", "benchmark"}

#: Not comparable, each with the reason it cannot carry a cross-DAG difference.
_STRUCTURAL = {
    "func": (
        "the auto-generated rule-body wrapper object (`__<rulename>`); never "
        "equal across two parses, and the executable content is `script:`, "
        "which IS compared"
    ),
    "path_modifier": (
        "module-system internal (listed in `RuleInfo.ref_attributes`); not a "
        "rule-body directive and unset outside `module:`/`use rule`"
    ),
}


def _ruleinfo_fields():
    import snakemake.api  # noqa: F401 -- resolves snakemake's circular imports
    from snakemake.ruleinfo import RuleInfo

    return set(RuleInfo().__dict__)


def test_ruleinfo_field_universe_is_fully_bucketed():
    """Deny by default: every RuleInfo field is classified, none is unclassified.

    Derives the universe from the pinned Snakemake rather than from a hardcoded
    list, so a version that adds a directive fails here instead of quietly
    escaping the equality check below.
    """
    fields = _ruleinfo_fields()
    buckets = set(_COMPARED) | _ALLOWED_LOCAL | set(_STRUCTURAL)

    unclassified = sorted(fields - buckets)
    assert not unclassified, (
        "RuleInfo directives not covered by any bucket: "
        f"{unclassified}. Classify each as compared, allowed-local, or "
        "structural-with-a-reason before the contract test can be trusted."
    )
    stale = sorted(buckets - fields)
    assert not stale, f"buckets name directives this Snakemake does not have: {stale}"
    # The buckets must not overlap: a directive cannot be both compared and
    # permitted to differ.
    assert set(_COMPARED).isdisjoint(_ALLOWED_LOCAL)
    assert set(_COMPARED).isdisjoint(_STRUCTURAL)
    assert _ALLOWED_LOCAL.isdisjoint(_STRUCTURAL)


#: shared.basin values the shipped test config does NOT declare, so the
#: "custom_basin" variant below proves both declarations READ the config rather
#: than both falling back to the same module default (which is all a
#: defaults-only run can show).
# `C-15`: every spatial input lives under `basin.sources:` -- `hydrography`
# and `basin_index` are named datasets like the rest.
_CUSTOM_BASIN = {
    "sources": {"hydrography": "merit_hydro_1k", "basin_index": "my_basin_index"}
}


@pytest.fixture(scope="module")
def config_variants(tmp_path_factory):
    """The shipped test config, plus one declaring both optional basin keys."""
    cfg = load_composed_config(CONFIG_FN)
    cfg["basin"].update(_CUSTOM_BASIN)
    custom = write_config(
        tmp_path_factory.mktemp("cfg"), cfg, stem="project_config_custom_basin"
    )
    return {"defaults": CONFIG_FN, "custom_basin": custom}


@pytest.fixture(scope="module", params=["defaults", "custom_basin"])
def declarations(request, config_variants):
    """The built ``extract_climate_datasets`` rule from both workflows, one config.

    Parametrized over both config variants: a defaults-only comparison passes
    whenever the two Snakefiles happen to share a fallback, which is a weaker
    property than "both read the same config key".
    """
    config_path = config_variants[request.param]
    out = {"_variant": request.param}
    for label, snakefile in (
        ("wf1", "build_model.smk"),
        ("wf3", "generate_scenarios.smk"),
    ):
        workflow = _parse_workflow(snakefile, config_path)
        rule = workflow.get_rule(RULE_NAME)
        out[label] = (workflow, rule)
    # WF2 keeps its workflow object but has no store rule since ADR 0003 — it
    # declared the producer only to obtain the region polygon, and the region is
    # now its own artifact. Kept here so the absence is ASSERTED rather than
    # silently untested.
    out["wf2_workflow"] = _parse_workflow("analyze_projections.smk", config_path)
    return out


@pytest.mark.workflow_contract
def test_optional_basin_keys_are_read_from_the_config_by_both(declarations):
    """Both declarations honour ``basin.sources.hydrography``/``basin_index``.

    Without this, a per-workflow divergence in the *default* (or one side
    forgetting to read the key at all) stays invisible on the shipped config.
    """
    # The config nests these under `basin.sources:` since `C-15`, but the RULE
    # still carries them as flat params -- a params name is internal and need
    # not track the section its value came from. So the expectation is the
    # nested VALUES, flattened.
    expected = (
        _CUSTOM_BASIN["sources"]
        if declarations["_variant"] == "custom_basin"
        else {"hydrography": "merit_hydro_ihu", "basin_index": "merit_hydro_index"}
    )
    for label in ("wf1", "wf3"):
        _workflow, rule = declarations[label]
        for key, value in expected.items():
            assert rule.params[key] == value, (
                f"{label}: params.{key} is {rule.params[key]!r}, expected {value!r}"
            )


@pytest.mark.workflow_contract
def test_rule_exists_in_both_workflows(declarations):
    for label in ("wf1", "wf3"):
        _workflow, rule = declarations[label]
        assert rule is not None, f"{label} has no {RULE_NAME} rule"
        assert rule.name == RULE_NAME


@pytest.mark.workflow_contract
def test_wf2_declares_no_store_and_no_extraction(declarations):
    """ADR 0003: a projections-only run does no climate extraction at all.

    WF2 used to declare the whole store producer to obtain the delineated
    polygon, and never read the gridded extraction it also wrote. The region is
    now its own artifact, so the extraction is gone from this workflow — the
    point of the change, and the thing most likely to be undone by someone
    re-adding the rule "for symmetry".
    """
    rule_names = {rule.name for rule in declarations["wf2_workflow"].rules}
    assert RULE_NAME not in rule_names
    assert "delineate_region" in rule_names


@pytest.mark.workflow_contract
def test_declarations_are_identical(declarations):
    """Every compared directive matches across ALL declarations.

    Compared pairwise against wf1 as the reference: with three declarations a
    single left/right comparison would let two agree while the third drifted.
    """
    import itertools

    labels = ("wf1", "wf3")
    differences = []
    for left_label, right_label in itertools.combinations(labels, 2):
        left_workflow, left_rule = declarations[left_label]
        right_workflow, right_rule = declarations[right_label]
        for field, extract in sorted(_COMPARED.items()):
            left = extract(left_workflow, left_rule)
            right = extract(right_workflow, right_rule)
            if left != right:
                differences.append(
                    f"{field} ({left_label} vs {right_label}):\n"
                    f"    {left_label} = {left}\n    {right_label} = {right}"
                )
    assert not differences, (
        f"{RULE_NAME} differs across the declaring workflows on "
        f"{len(differences)} directive comparison(s). Only message/log/benchmark "
        "may differ; everything else must come from climate_store_rule.\n"
        + "\n".join(differences)
    )


@pytest.mark.workflow_contract
def test_the_single_input_is_the_catalog(declarations):
    """Exactly one input, keyed ``catalog``, and NOT ancient() — ext2-01.

    An asymmetric or absent input set is what the oscillation needs; the catalog
    file is the store's declared freshness boundary.
    """
    for label in ("wf1", "wf3"):
        _workflow, rule = declarations[label]
        assert list(rule.input.keys()) == ["catalog", "region_geojson"], (
            f"{label}: {RULE_NAME} inputs are {list(rule.input.keys())}, "
            "expected exactly ['catalog', 'region_geojson']"
        )
        assert len(rule.input) == 2, f"{label}: extra positional inputs"
        ancient_paths = {str(f) for f in rule.input if getattr(f, "is_ancient", False)}
        assert not ancient_paths, (
            f"{label}: the catalog input must be plain, not ancient() — "
            f"ancient inputs found: {ancient_paths}"
        )


@pytest.mark.workflow_contract
def test_outputs_are_the_store_artifacts(declarations):
    """The era5 seed branch declares the extraction and its basin-cell mask.

    ADR 0003 retired the per-store-key ``store_region.geojson``: the polygon is
    one project artifact, declared here as an INPUT, and the store's extent
    provenance moved into the extraction's own attributes.

    ``basin_cells.csv`` joined on 2026-08-10 and is part of the contract for
    both workflows, not a WF3-local artifact: it says which extracted cells the
    basin touches, which is a property of THIS extraction's grid and derivable
    only where that grid meets the region polygon. Rule 3.11 averages over
    exactly those cells instead of over every cell the bbox+buffer read
    happened to include.
    """
    for label in ("wf1", "wf3"):
        _workflow, rule = declarations[label]
        keys = sorted(rule.output.keys())
        assert keys == ["basin_cells", "climate_nc"], f"{label}: {keys}"
        assert str(rule.output.basin_cells).endswith("/basin_cells.csv"), label
        assert str(rule.output.climate_nc).endswith("/extract_historical.nc"), label
        assert not [str(path) for path in rule.output if "store_region" in str(path)], (
            label
        )


@pytest.mark.workflow_contract
def test_retired_declarations_are_gone(declarations):
    """No wf1-only store, and no rule anywhere writes under ``wf1_raw/``."""
    wf1_workflow, _ = declarations["wf1"]
    rule_names = {rule.name for rule in wf1_workflow.rules}
    assert "extract_climate_datasets_wf1" not in rule_names
    stale = [
        (rule.name, str(path))
        for rule in wf1_workflow.rules
        for path in rule.output
        if "wf1_raw" in str(path)
    ]
    assert not stale, f"rules still writing into the retired wf1_raw store: {stale}"


@pytest.mark.workflow_contract
def test_generation_has_no_model_consistency_guard(declarations):
    """Generation owns its sources and has no WF1 model/snapshot dependency."""
    wf3_workflow, producer = declarations["wf3"]
    assert "check_project_consistency" not in {rule.name for rule in wf3_workflow.rules}

    producer_inputs = {str(path).replace("\\", "/") for path in producer.input}
    assert not any(".guard_ok" in path for path in producer_inputs)
    # The retired edge is the MODEL's region, not any region. R07 B1 made the
    # store model-free by dropping
    # ancient(hydrology_model/staticgeoms/region.geojson); ADR 0003 gives it
    # spatial/geoms/region.geojson instead, which is model-free by
    # construction. Asserting "no path containing region.geojson" would forbid
    # the replacement along with the thing it replaced, so the check now names
    # the coupling it actually guards against.
    assert not any("hydrology_model/" in path for path in producer_inputs)
    assert not any("staticgeoms/" in path for path in producer_inputs)
    assert any(
        path.endswith("/spatial/geoms/region.geojson") for path in producer_inputs
    ), producer_inputs


@pytest.mark.workflow_contract
def test_chirps_branch_declares_and_consumes_one_orography_path(tmp_path):
    """R07 standardises the sidecar on ``orography.nc``, producer and consumer.

    Pre-R07 the two stores spelled it differently (``wf1_raw/orography.nc`` vs
    ``<key>/<clim_source>_orography.nc``), and the consumer's ``oro_path``
    params string pointed at the second spelling. The seed config is era5, so no
    gate in this repo would otherwise exercise the chirps branch at all.

    The consumer is rule 3.14 ``downscale_scenario_series`` since
    2026-08-18: it writes its own member catalog, and building the chirps
    ``<source>_orography`` entry is what needs the sidecar path. It was rule
    3.13 ``write_climate_data_catalog``, which is gone with the aggregate
    catalog — and this test failing on that move is the whole reason it exists,
    since the era5 fixture would not have noticed.
    """

    cfg = load_composed_config(CONFIG_FN)
    # `C-43`: `selected` names a MEMBER of `sources`, and the loader refuses
    # one that is not -- so switching the source means declaring it too.
    cfg["climate"]["selected"] = "chirps_global"
    cfg["climate"]["sources"] = ["chirps_global"]
    cfg_path = write_config(tmp_path, cfg, stem="project_config_chirps")

    workflow = _parse_workflow("generate_scenarios.smk", cfg_path)
    producer = workflow.get_rule(RULE_NAME)

    # Since static WF3 planning (2026-09) no WF3 rule consumes the sidecar: its
    # pinned sources are the store's climate and cells. The consumer is WF4's
    # downscaling, covered by tests/test_member_catalog_rule.py.
    oro_out = str(producer.output.oro_nc)
    assert oro_out.endswith("/orography.nc"), oro_out
    assert "chirps_global_orography" not in oro_out

    # wf1 declares the same sidecar output on the same branch.
    wf1 = _parse_workflow("build_model.smk", cfg_path)
    assert str(wf1.get_rule(RULE_NAME).output.oro_nc) == oro_out


# ---------------------------------------------------------------------------
# WF0 — equivalence under binding, not byte-identity
# ---------------------------------------------------------------------------
#
# `analyze_climate.smk` is the one workflow that does NOT carry a byte-identical
# copy of the store declaration. It GENERATES one concrete rule per candidate
# source because selected CHIRPS forcing and comparison-only CHIRPS have
# different output sets, while a Snakemake rule has a fixed output set.
#
# The invariant that replaces byte-identity: BINDING THE GENERATED RULE TO
# `climate.selected` MUST YIELD THE SHARED CONTRACT. If it does not, WF0
# and WF1 would extract into the same directory from different params -- the
# re-extraction oscillation the shared contract exists to prevent, arriving
# through a rule name instead of through a diverging input set.


def _wf0_rule(workflow, source):
    """The generated store rule for one candidate source."""
    return workflow.get_rule(f"{RULE_NAME}_{source}")


@pytest.mark.workflow_contract
def test_wf0_primary_source_rule_equals_the_shared_contract(tmp_path):
    """WF0's generated rule for the project's own source == WF1's declaration."""

    cfg_path = CONFIG_FN
    wf0 = _parse_workflow("analyze_climate.smk", cfg_path)
    wf1 = _parse_workflow("build_model.smk", cfg_path)

    cfg = load_composed_config(CONFIG_FN)
    primary = cfg["climate"]["selected"]

    generated = _wf0_rule(wf0, primary)
    shared = wf1.get_rule(RULE_NAME)

    # Script, inputs, outputs and params -- everything content- or
    # execution-determining. The rule NAME differs by construction and is the
    # one difference this test allows.
    assert Path(generated.script).name == Path(shared.script).name
    assert sorted(map(str, generated.input)) == sorted(map(str, shared.input))
    assert sorted(map(str, generated.output)) == sorted(map(str, shared.output))
    for key in shared.params.keys():
        assert str(generated.params[key]) == str(shared.params[key]), (
            f"params.{key}: wf0 {generated.params[key]!r} != wf1 {shared.params[key]!r}"
        )


@pytest.mark.workflow_contract
def test_wf0_candidate_source_gets_its_own_store_and_family_outputs(tmp_path):
    """A comparison-only CHIRPS store carries no forcing-only sidecar."""

    cfg = load_composed_config(CONFIG_FN)
    assert cfg["climate"]["selected"] == "era5"
    cfg["climate"]["sources"] = [cfg["climate"]["selected"], "chirps"]
    cfg_path = write_config(tmp_path, cfg, stem="project_config_two_sources")

    wf0 = _parse_workflow("analyze_climate.smk", cfg_path)
    era5 = _wf0_rule(wf0, "era5")
    chirps = _wf0_rule(wf0, "chirps")

    # Distinct stores, both under the ONE historical bin -- a candidate that
    # wins the comparison is then already extracted where WF1 would look.
    era5_nc = str(era5.output.climate_nc)
    chirps_nc = str(chirps.output.climate_nc)
    assert era5_nc != chirps_nc
    assert "/data/climate/historical/era5_" in era5_nc.replace("\\", "/")
    assert "/data/climate/historical/chirps_" in chirps_nc.replace("\\", "/")

    # Neither source needs an orography sidecar for source comparison. A
    # selected CHIRPS store still declares one for wf1/wf3 forcing.
    assert not hasattr(era5.output, "oro_nc") or "oro_nc" not in era5.output.keys()
    assert not hasattr(chirps.output, "oro_nc") or "oro_nc" not in chirps.output.keys()
    assert chirps.params["forcing_required"] is False


@pytest.mark.workflow_contract
def test_wf0_rejects_an_unsupported_candidate_source(tmp_path):
    """An unsupported candidate fails at PARSE time, naming the config key.

    Deferring it to the generated rule would surface the failure under a rule
    name that does not say which config key put the source there.
    """

    cfg = load_composed_config(CONFIG_FN)
    cfg["climate"]["sources"] = [cfg["climate"]["selected"], "eobs"]
    cfg_path = write_config(tmp_path, cfg, stem="project_config_bad_source")

    with pytest.raises(Exception) as exc:
        _parse_workflow("analyze_climate.smk", cfg_path)
    message = str(exc.value)
    assert "eobs" in message
    assert "climate.sources" in message or "sources" in message


# ---------------------------------------------------------------------------
# The MIN_HISTORICAL_YEARS floor splits with the source's ROLE (2026-08-16)
# ---------------------------------------------------------------------------
#
# `climate.window` is a ceiling, not a demand: a source that cannot
# fill it is extracted over what it holds. The floor still binds the source that
# FEEDS the pipeline -- weathergenr's wavelet minimum -- and not a candidate that
# ends at a comparison figure.
#
# The flag rides in `params`, which is a Snakemake rerun trigger, so its ABSENCE
# on the default path is load-bearing: emitting `enforce_min_years: True` would
# re-extract every store already on disk and break the byte-identity above.


@pytest.mark.workflow_contract
def test_the_enforced_default_emits_no_param_at_all():
    """The params dict is a rerun trigger; the default path must not touch it."""
    from blueearth_cst.shared.snake_utils import climate_store_rule

    spec = climate_store_rule(
        project_dir="/tmp/p",
        model_region="{'subbasin': [1.0, 2.0]}",
        clim_source="era5",
        historical_window={"start": 2000, "end": 2020},  # `C-70`: years
        data_sources="catalog.yml",
    )
    assert "enforce_min_years" not in spec.params
    assert "forcing_required" not in spec.params


@pytest.mark.workflow_contract
def test_only_a_relaxed_store_carries_the_flag():
    from blueearth_cst.shared.snake_utils import climate_store_rule

    spec = climate_store_rule(
        project_dir="/tmp/p",
        model_region="{'subbasin': [1.0, 2.0]}",
        clim_source="chirps",
        historical_window={"start": 2000, "end": 2020},  # `C-70`: years
        data_sources="catalog.yml",
        enforce_min_years=False,
    )
    assert spec.params["enforce_min_years"] is False
    # Everything else is unchanged -- the flag is not a second store key.
    assert "/data/climate/historical/chirps_20000101_20201231" in spec.store_dir


@pytest.mark.workflow_contract
def test_wf0_relaxes_the_floor_for_candidates_only(tmp_path):
    """The primary keeps the floor; the extras relax it.

    This is what stops a short candidate store from being promoted silently: WF1
    and WF3 declare the store WITHOUT the flag, so switching
    `climate.selected` onto a candidate changes the params Snakemake
    recorded and re-extracts it under the floor.
    """

    cfg = load_composed_config(CONFIG_FN)
    assert cfg["climate"]["selected"] == "era5"
    cfg["climate"]["sources"] = [cfg["climate"]["selected"], "chirps"]
    cfg_path = write_config(tmp_path, cfg, stem="project_config_floor_split")

    wf0 = _parse_workflow("analyze_climate.smk", cfg_path)
    primary = _wf0_rule(wf0, "era5")
    candidate = _wf0_rule(wf0, "chirps")

    assert "enforce_min_years" not in primary.params.keys()
    assert candidate.params["enforce_min_years"] is False


@pytest.mark.workflow_contract
def test_generation_plan_declares_the_store_before_seed_resolution(tmp_path):
    """Actual source bytes and the historical floor precede the generator config.

    Planning (3.04-3.06) runs in the launcher after the source phase, so the
    source phase's own target must be the extracted store it will pin.
    """
    workflow = _parse_workflow("generate_scenarios.smk", CONFIG_FN)
    rule = workflow.get_rule("all")
    store_nc = str(workflow.get_rule(RULE_NAME).output.climate_nc)
    assert store_nc in list(map(str, rule.expand_input({})[0]))
