"""R07 B1: the three ``extract_historical_climate`` declarations are ONE rule.

The store producer is declared as ``extract_historical_climate`` in
``build_model.smk`` (rule 1.04) and ``run_stress_test.smk`` (rule 3.08), both
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

SNAKEDIR = Path(__file__).resolve().parents[1]
CONFIG_FN = Path(__file__).resolve().parent / "snake_config_fixture.yml"
RULE_NAME = "extract_historical_climate"

#: Sentinel for "the built Rule carries no such attribute on this Snakemake".
#: Compared as a value, so absent-on-both is equal and absent-on-one fails.
_ABSENT = "<<absent>>"


def _parse_workflow(snakefile: str, config_path):
    """Parse a Snakefile in-process and return its ``Workflow``.

    Uses the ``snakemake.api`` entry point so rules are built exactly as a real
    invocation builds them — the comparison then runs against effective rule
    state (post-``RuleInfo``-application), which is what actually determines
    reruns. ``wf_api._workflow`` is private on Snakemake 9.6.2; there is no
    public accessor for the parsed workflow object, so this is pinned to the
    pinned version deliberately.
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


# --- normalizers --------------------------------------------------------------


def _iofile_signature(namedlist):
    """(positional paths, sorted keyword->path) for an input/output namedlist."""
    return (
        tuple(str(item) for item in namedlist),
        tuple(sorted((key, str(value)) for key, value in namedlist.items())),
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
_CUSTOM_BASIN = {"hydrography": "merit_hydro_1k", "basin_index": "my_basin_index"}


@pytest.fixture(scope="module")
def config_variants(tmp_path_factory):
    """The shipped test config, plus one declaring both optional basin keys."""
    import yaml

    cfg = yaml.safe_load(CONFIG_FN.read_text(encoding="utf-8"))
    cfg["shared"]["basin"].update(_CUSTOM_BASIN)
    custom = tmp_path_factory.mktemp("cfg") / "snake_config_custom_basin.yml"
    custom.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return {"defaults": CONFIG_FN, "custom_basin": custom}


@pytest.fixture(scope="module", params=["defaults", "custom_basin"])
def declarations(request, config_variants):
    """The built ``extract_historical_climate`` rule from both workflows, one config.

    Parametrized over both config variants: a defaults-only comparison passes
    whenever the two Snakefiles happen to share a fallback, which is a weaker
    property than "both read the same config key".
    """
    config_path = config_variants[request.param]
    out = {"_variant": request.param}
    for label, snakefile in (
        ("wf1", "build_model.smk"),
        ("wf3", "run_stress_test.smk"),
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


@pytest.mark.slow
@pytest.mark.workflow_contract
def test_optional_basin_keys_are_read_from_the_config_by_both(declarations):
    """Both declarations honour ``shared.basin.hydrography``/``basin_index``.

    Without this, a per-workflow divergence in the *default* (or one side
    forgetting to read the key at all) stays invisible on the shipped config.
    """
    expected = (
        _CUSTOM_BASIN
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
    assert "extract_historical_climate_wf1" not in rule_names
    stale = [
        (rule.name, str(path))
        for rule in wf1_workflow.rules
        for path in rule.output
        if "wf1_raw" in str(path)
    ]
    assert not stale, f"rules still writing into the retired wf1_raw store: {stale}"


@pytest.mark.workflow_contract
def test_guard_keeps_its_receipt_but_loses_its_edge(declarations):
    """Rule 3.00b is untouched; only rule 3.08's DAG edge to ``.guard_ok`` retires."""
    wf3_workflow, producer = declarations["wf3"]
    guard = wf3_workflow.get_rule("check_project_consistency")
    guard_outputs = sorted(guard.output.keys())
    assert guard_outputs == ["guard_ok", "sentinel"], guard_outputs
    assert str(guard.output.guard_ok).endswith("/.guard_ok")

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

    The consumer is rule 3.14 ``downscale_climate_realization`` since
    2026-08-18: it writes its own member catalog, and building the chirps
    ``<source>_orography`` entry is what needs the sidecar path. It was rule
    3.13 ``write_climate_data_catalog``, which is gone with the aggregate
    catalog — and this test failing on that move is the whole reason it exists,
    since the era5 fixture would not have noticed.
    """
    import yaml

    cfg = yaml.safe_load(CONFIG_FN.read_text(encoding="utf-8"))
    cfg["shared"]["clim_historical"] = "chirps_global"
    cfg_path = tmp_path / "snake_config_chirps.yml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    workflow = _parse_workflow("run_stress_test.smk", cfg_path)
    producer = workflow.get_rule(RULE_NAME)
    consumer = workflow.get_rule("downscale_climate_realization")

    oro_out = str(producer.output.oro_nc)
    assert oro_out.endswith("/orography.nc"), oro_out
    assert "chirps_global_orography" not in oro_out
    assert str(consumer.params.oro_path) == oro_out, (
        "the catalog builder's oro_path must resolve to the emitted sidecar, got "
        f"{consumer.params.oro_path!r} vs {oro_out!r}"
    )

    # wf1 declares the same sidecar output on the same branch.
    wf1 = _parse_workflow("build_model.smk", cfg_path)
    assert str(wf1.get_rule(RULE_NAME).output.oro_nc) == oro_out


# ---------------------------------------------------------------------------
# WF0 — equivalence under binding, not byte-identity
# ---------------------------------------------------------------------------
#
# `analyze_climate.smk` is the one workflow that does NOT carry a byte-identical
# copy of the store declaration. It GENERATES one concrete rule per candidate
# source, because `climate_store_rule` returns an `oro_nc` output for chirps and
# none for era5, and a Snakemake rule has a fixed output set -- so a single
# wildcard rule cannot cover both families and a wildcard split would hard-code
# a source taxonomy the factory already knows.
#
# The invariant that replaces byte-identity: BINDING THE GENERATED RULE TO
# `shared.clim_historical` MUST YIELD THE SHARED CONTRACT. If it does not, WF0
# and WF1 would extract into the same directory from different params -- the
# re-extraction oscillation the shared contract exists to prevent, arriving
# through a rule name instead of through a diverging input set.


def _wf0_rule(workflow, source):
    """The generated store rule for one candidate source."""
    return workflow.get_rule(f"{RULE_NAME}_{source}")


@pytest.mark.slow
@pytest.mark.workflow_contract
def test_wf0_primary_source_rule_equals_the_shared_contract(tmp_path):
    """WF0's generated rule for the project's own source == WF1's declaration."""
    import yaml

    cfg_path = CONFIG_FN
    wf0 = _parse_workflow("analyze_climate.smk", cfg_path)
    wf1 = _parse_workflow("build_model.smk", cfg_path)

    cfg = yaml.safe_load(CONFIG_FN.read_text(encoding="utf-8"))
    primary = cfg["shared"]["clim_historical"]

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


@pytest.mark.slow
@pytest.mark.workflow_contract
def test_wf0_candidate_source_gets_its_own_store_and_family_outputs(tmp_path):
    """A second source mints a second store, with ITS family's output set.

    era5 carries no orography sidecar and chirps does, which is precisely why
    these are generated rules rather than one wildcard rule. Asserting both in
    one DAG is what proves the generation reads the spec rather than a taxonomy
    written into the Snakefile.
    """
    import yaml

    cfg = yaml.safe_load(CONFIG_FN.read_text(encoding="utf-8"))
    assert cfg["shared"]["clim_historical"] == "era5"
    cfg["workflows"]["analyze_climate"]["candidate_sources"] = ["chirps"]
    cfg_path = tmp_path / "snake_config_two_sources.yml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

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

    # The family split the output set encodes.
    assert not hasattr(era5.output, "oro_nc") or "oro_nc" not in era5.output.keys()
    assert "oro_nc" in chirps.output.keys()


@pytest.mark.workflow_contract
def test_wf0_rejects_an_unsupported_candidate_source(tmp_path):
    """An unsupported candidate fails at PARSE time, naming the config key.

    Deferring it to the generated rule would surface the failure under a rule
    name that does not say which config key put the source there.
    """
    import yaml

    cfg = yaml.safe_load(CONFIG_FN.read_text(encoding="utf-8"))
    cfg["workflows"]["analyze_climate"]["candidate_sources"] = ["eobs"]
    cfg_path = tmp_path / "snake_config_bad_source.yml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(Exception) as exc:
        _parse_workflow("analyze_climate.smk", cfg_path)
    message = str(exc.value)
    assert "eobs" in message
    assert "candidate_sources" in message


# ---------------------------------------------------------------------------
# The MIN_HISTORICAL_YEARS floor splits with the source's ROLE (2026-08-16)
# ---------------------------------------------------------------------------
#
# `shared.historical_window` is a ceiling, not a demand: a source that cannot
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
        historical_window={
            "starttime": "2000-01-01T00:00:00",
            "endtime": "2020-12-31T00:00:00",
        },
        data_sources="catalog.yml",
    )
    assert "enforce_min_years" not in spec.params


@pytest.mark.workflow_contract
def test_only_a_relaxed_store_carries_the_flag():
    from blueearth_cst.shared.snake_utils import climate_store_rule

    spec = climate_store_rule(
        project_dir="/tmp/p",
        model_region="{'subbasin': [1.0, 2.0]}",
        clim_source="chirps",
        historical_window={
            "starttime": "2000-01-01T00:00:00",
            "endtime": "2020-12-31T00:00:00",
        },
        data_sources="catalog.yml",
        enforce_min_years=False,
    )
    assert spec.params["enforce_min_years"] is False
    # Everything else is unchanged -- the flag is not a second store key.
    assert "/data/climate/historical/chirps_20000101_20201231" in spec.store_dir


@pytest.mark.slow
@pytest.mark.workflow_contract
def test_wf0_relaxes_the_floor_for_candidates_only(tmp_path):
    """The primary keeps the floor; the extras relax it.

    This is what stops a short candidate store from being promoted silently: WF1
    and WF3 declare the store WITHOUT the flag, so switching
    `shared.clim_historical` onto a candidate changes the params Snakemake
    recorded and re-extracts it under the floor.
    """
    import yaml

    cfg = yaml.safe_load(CONFIG_FN.read_text(encoding="utf-8"))
    assert cfg["shared"]["clim_historical"] == "era5"
    cfg["workflows"]["analyze_climate"]["candidate_sources"] = ["chirps"]
    cfg_path = tmp_path / "snake_config_floor_split.yml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    wf0 = _parse_workflow("analyze_climate.smk", cfg_path)
    primary = _wf0_rule(wf0, "era5")
    candidate = _wf0_rule(wf0, "chirps")

    assert "enforce_min_years" not in primary.params.keys()
    assert candidate.params["enforce_min_years"] is False


@pytest.mark.workflow_contract
def test_wf3_config_prep_declares_the_store_so_it_can_check_it(tmp_path):
    """Rule 3.10 guards what rule 3.11 cannot.

    3.11 is a `shell:` running R, so the floor check has to sit in the Python
    rule ahead of it. The edge is `ancient()` for the same reason 3.11's is: a
    re-extraction must not by itself re-run the config prep.
    """
    workflow = _parse_workflow("run_stress_test.smk", CONFIG_FN)
    rule = workflow.get_rule("prepare_weathergen_config")

    store_nc = str(workflow.get_rule(RULE_NAME).output.climate_nc)
    assert str(rule.input.climate_nc) == store_nc
    ancient_paths = {str(f) for f in rule.input if getattr(f, "is_ancient", False)}
    assert store_nc in ancient_paths, (
        f"the store input must be ancient(), got ancient inputs {ancient_paths}"
    )
    # The source name travels too, so the message can say WHICH source fell short.
    assert str(rule.params.clim_source) == str(
        workflow.get_rule(RULE_NAME).params.clim_source
    )
