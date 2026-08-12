# -*- coding: utf-8 -*-
"""Which indicator tables a WF3 experiment emits, derived from ``wflow_outvars``.

CR-2 splits the response-surface results into **one table per output variable**
instead of the two fixed tables (``q_indicators.csv`` + ``basin_indicators.csv``)
that preceded it. The set is therefore config-dependent, and four places need to
agree on it before any of them can run:

- ``Snakefile_climate_experiment`` — ``WF3_TARGETS`` and rule 3.16's ``output:``,
  at DAG-construction time;
- ``blueearth_cst/experiment/export_wflow_results.py`` — what it writes, and the
  ``variable`` half of each composite ``metric``;
- ``blueearth_cst/shared/interchange_contracts.py`` — HM-7's per-table checks;
- ``dev/scripts/check_baseline.py`` and ``semantic_tree_diff.py`` — the target
  list and the path map.

Runtime discovery is not an option for the first of those. The pre-CR-2 writer
found its variables by reading the wflow run CSV's own columns
(``[x for x in sim.columns if "basavg" in x]``), which cannot work when Snakemake
needs the output paths *before* any rule has produced a CSV to inspect.

**The tokens are a third spelling, and that is the accepted cost.** Alongside the
CSDMS names (``river_water__volume_flow_rate``) and the Tier 2 display labels,
these short tokens exist because a composite metric built from snake-cased
semantic names would read ``actual_evapotranspiration_annual_total`` — which
undercuts the readability that motivated composing the name at all. CR-2 places
this mapping in the seam contract for that reason: it is a contract, not an
implementation detail.

The rule for minting future tokens, so they are not chosen ad hoc: **where the
repo already has a canonical short name, use it; only mint where none exists; and
disambiguate against names already in use.** Its four consequences are why
``precip`` is not ``p`` (``naming.md`` §6 tier 2 declares ``precip`` canonical, and
``p`` would be a seventh spelling), why ``aet`` is not ``et`` (``pet`` is already
canonical here and one letter apart in the same file is a misreading waiting to
happen), why ``snow`` is not ``swe`` (the CSDMS name is
``snowpack_liquid_water__depth`` — snowpack *liquid water*, not total water
equivalent, so minting ``swe`` would assert a physical claim upstream does not
make, which ``AGENTS.md`` puts out of scope), and — since 2026-08-11 — why
groundwater recharge is ``gwr`` rather than ``recharge``: ``gwr`` is the code
``wflow_outputs.CODES`` already writes into the TOML and therefore into every
run csv header, so the first clause of the rule applies and ``recharge`` was the
violation. That rename is the one case where the rule *removed* a spelling
instead of choosing between two new ones; see ``OUTPUT_CODES`` below, where the
token and the code now coincide for this variable.

The published table of all of these, one row per variable and one column per
spelling, is ``dev/reference/indicator-glossary.md``. It is DERIVED from the
dicts here and in ``wflow_outputs`` — ``tests/test_indicator_glossary.py`` parses
it and fails when the two disagree — so extend the dicts first and the glossary
second, never the reverse.
"""

from __future__ import annotations

from blueearth_cst.shared.wflow_outputs import CODES as _WFLOW_CODES

#: Semantic name (as it appears in ``workflows.model_creation.wflow_outvars``)
#: → short token used in filenames and in the composite ``metric``.
#:
#: Authoritative source for the semantic names: the ``WFLOW_VARS`` map in
#: ``dev/reference/workflows/model_creation.md``. **Six entries, not five** —
#: ``precipitation`` is one of them, emitted at registry locations with header
#: ``P`` when a ``location_registry`` is configured.
VARIABLE_TOKENS = {
    "river discharge": "q",
    "precipitation": "precip",
    "actual evapotranspiration": "aet",
    "groundwater recharge": "gwr",
    "overland flow": "overland_flow",
    "snow": "snow",
}

#: Indicator token → the CODE that variable's columns carry in a wflow run csv.
#:
#: **A fourth spelling, and unlike the other three it is not ours to choose.** The
#: code is `wflow_outputs.CODES`, which reaches the csv header via the TOML the
#: model build writes (``gwr_101``, ``aet_101``); the token is what this file
#: mints for filenames and metric names. Three of the five still differ —
#: ``precip`` is ``p``, ``snow`` is ``swe``, ``overland_flow`` is ``qof`` — so a
#: reducer that reuses the token as the column prefix reads exactly the two where
#: they coincide (``aet``, ``gwr``) and silently finds nothing for the rest.
#: **Two agreeing tokens is worse than one for a naive fix**, not better: it
#: doubles the chance that whichever variable a hand-written test reaches for is
#: one the broken matcher happens to work on.
#:
#: DERIVED from the two tables rather than hand-written, because a third
#: hand-maintained copy of the variable list is how this drifted in the first
#: place: 8bd51de changed the header scheme and the reducer's own matcher kept
#: looking for the retired ``<label>_basavg`` spelling, which emptied
#: ``aet_indicators.csv`` and ``recharge_indicators.csv`` (as the recharge table
#: was then named — it is ``gwr_indicators.csv`` since 2026-08-11) with the run
#: green. Both
#: tables are keyed by the SEMANTIC LABEL a config writes, which is what lets them
#: be joined here; a label in one and not the other raises at import.
OUTPUT_CODES = {VARIABLE_TOKENS[label]: code for label, code in _WFLOW_CODES.items()}

#: Filename suffix shared by every indicator table. ``q_indicators.csv`` keeps the
#: name it was given in R9, which is why the pattern is token-first.
_TABLE_SUFFIX = "_indicators.csv"

#: The seven columns of every indicator table, in order. Fixed regardless of gauge
#: count — that fixity is the point of the long shape.
#:
#: Lives here rather than in the writer because BOTH the writer and ``validate_hm7``
#: need it, and ``shared/`` may not import from ``experiment/``. Stating it once is
#: also what stops the producer and its validator disagreeing about the header,
#: which is the specific failure this pairing exists to prevent.
#:
#: **Ordered identifier-first, owner ruling 2026-08-11.** The columns now read
#: *what* (``metric``, ``location``), then *which member* (``st_id``, ``rlz_id``),
#: then *where on the surface* (``temp_change``, ``precip_change``), then the
#: number. That groups the two id columns adjacently instead of splitting them
#: around the perturbation axes, which is how a reader scans a long table: pick the
#: series, then the member, then read across. Same seven columns and the same
#: fixity — this is a reorder and one rename, not a shape change.
#:
#: ``realization_id`` became ``rlz_id`` in the same ruling, matching the ``rlz_``
#: member token already carried by the run filenames (``rlz_1_st_0.csv``) and by
#: ``RLZ_NUM``. It also sets the two id columns to the same ``<thing>_id`` width,
#: which is what makes the adjacency above read as a pair.
INDICATOR_COLUMNS = (
    "metric",
    "location",
    "st_id",
    "rlz_id",
    "temp_change",
    "precip_change",
    "value",
)

#: The stress dimensions this shape can express. C28 was ruled "at this stage" —
#: `st_id` ALONGSIDE the perturbation columns rather than replacing them, chosen
#: for plottable-without-a-join, with an explicit revisit when a third dimension
#: arrives. That re-couples the header to the dimension count, which is what CR-2
#: removed on the location axis, so the coupling is made refusable rather than
#: silent: the writer raises when a design table carries an axis outside this set.
DESIGN_AXES = ("temp_change", "precip_change")

#: ``rlz_id`` for a pooled row. A numeric sentinel in a numeric key column is safe
#: ONLY because no metric emits both grains; if that ever changes it must become a
#: string, or ``groupby("rlz_id")`` folds pooled rows in as another realization.
POOLED_REALIZATION = 0

#: The reserved ``location`` for a basin-scalar value (Q11, 2026-08-07): emitted
#: independently rather than derived from per-location values, because whether
#: subcatchments nest or tile decides whether an area-weighted mean is even valid.
#:
#: **Reserved, and currently unemitted.** Since 8bd51de the model's csv columns are
#: per-subcatchment means (``map = "subcatchment"``), so no whole-basin column
#: exists in a run to carry this value — the reducer emits one row per subcatchment
#: id, exactly as it does per gauge. Q11 is why it stays that way rather than being
#: area-weighted into a basin row here: filling this constant needs a whole-basin
#: column declared in the TOML, which is a WF1 change, not a reduction one.
BASIN_LOCATION = "basin"


#: Discharge metrics: internal statistic key → (metric suffix, grain class).
#:
#: The suffix is composed systematically rather than borrowed from conventional
#: hydrological shorthand, because two of ours differ from the established
#: meaning: our ``q95`` is the mean annual 95th percentile (a HIGH flow), while
#: conventional *Q95* is the flow exceeded 95% of the time (a LOW-flow drought
#: index) — opposite ends of the distribution. Hence ``_p95``. Verbose beats
#: wrong.
#:
#: ``mean_annual_`` is not padding: it says "annual statistic, then mean over
#: years", which the pre-R11 names hid.
#:
#: **"Return level", not "return interval" or "return period" — the distinction is
#: real, not pedantry.** The *return period* is the input ``T`` (years); the
#: *return level* is what comes back, a discharge magnitude, which is what this
#: row's ``value`` holds. Naming the metric after the period would name the wrong
#: quantity. "Return interval" is additionally non-standard: the established
#: synonyms are *return period* (engineering, extreme-value theory) and
#: *recurrence interval* (USGS usage) — the two the knowledge base's
#: ``hydrological-indicators`` note uses throughout. The statistic keys below were
#: renamed off the inherited ``returninterval`` spelling for that reason.
#:
#: Return periods, in years, that the two GEV return-level indicators are
#: evaluated at. **Toolbox constants since 2026-08-12, not config values.**
#:
#: They were the ``Tpeak`` / ``Tlow`` keys of ``workflows.climate_experiment``
#: until then, and the owner retired them from the project config: a return
#: period is a property of the indicator set this toolbox defines, and indicator
#: definitions live here rather than in a per-project scaffold. Both keys shipped
#: at these values in every config the repo carried, so retiring them changed no
#: emitted name and no number.
#:
#: A project that genuinely needs a different design standard changes them here,
#: which is a toolbox edit and re-names two indicators — deliberately visible,
#: rather than a per-project knob that silently redefines what a column means.
RETURN_PERIOD_PEAK_YR = 10
RETURN_PERIOD_LOW_YR = 2

#: The two return levels still carry their return period in the NAME, which is
#: the property R11 bought: before it, two runs at different periods produced
#: identical-looking rows meaning different things — a 10-year and a 20-year
#: flood level, indistinguishable once the file left the project folder. The
#: period is no longer a config value (see :data:`RETURN_PERIOD_PEAK_YR`), so
#: the vocabulary is now a fixed enumeration rather than a pattern.
#:
#: Grain classes (CR-2): **A** is emitted per realization, **B** and **C** only
#: pooled (``rlz_id = 0``). See ``METRIC_CLASSES``.
Q_METRIC_SUFFIXES = {
    "mean": ("annual_mean", "A"),
    "max": ("mean_annual_max", "A"),
    "min": ("mean_annual_min", "A"),
    "q95": ("mean_annual_p95", "A"),
    "Q7day_max": ("mean_annual_7day_max", "A"),
    "Q7day_min": ("mean_annual_7day_min", "A"),
    "BaseFlowIndex": ("baseflow_index", "A"),
    "return_level_max": (f"return_level_{RETURN_PERIOD_PEAK_YR}yr_max", "B"),
    "return_level_7day_min": (
        f"return_level_{RETURN_PERIOD_LOW_YR}yr_7day_min",
        "B",
    ),
    "wetmonth_mean": ("wettest_month_mean", "C"),
    "drymonth_mean": ("driest_month_mean", "C"),
}

#: Basin-scalar variables: token → (metric suffix, annual reduction).
#:
#: Ruled 2026-08-08 (Q10 scope + metric naming). Two things worth reading
#: together, because they are why this table is not uniform:
#:
#: 1. **Overland flow reduces with a MEAN, not a sum.** It is a volume flow rate
#:    (m³ s⁻¹), so summing daily values produces a quantity in no unit anyone
#:    wants; the annual mean preserves the native unit. ET, recharge and
#:    precipitation keep ``annual_total`` in mm/yr — a daily sum of a mm Δt⁻¹
#:    flux is a legitimate time-integral, which is precisely why overland flow
#:    was the odd one out. Scoped deliberately to overland flow: reporting ET as
#:    mm/day instead would rescale it by 365 and is not what a water-balance
#:    reader expects.
#: 2. **The suffixes omit the ``mean_`` prefix** the q vocabulary uses
#:    (``snow_annual_max``, not ``snow_mean_annual_max``), by owner ruling
#:    2026-08-08. **Accepted asymmetry, recorded so a reader does not read it as
#:    a bug:** ``q_mean_annual_max`` and ``snow_annual_max`` describe the same
#:    reduction shape — annual statistic, then mean over years — spelled two
#:    ways. The q vocabulary makes the mean-over-years step visible; these do not.
BASIN_METRIC_SUFFIXES = {
    "aet": ("annual_total", "sum"),
    "gwr": ("annual_total", "sum"),
    "precip": ("annual_total", "sum"),
    "snow": ("annual_max", "max"),
    "overland_flow": ("annual_mean", "mean"),
}

#: Grain per class. `0` means pooled over realizations; see the
#: `rlz_id = 0` decision, and note the sentinel is only safe because no
#: metric emits both grains — if that ever changes, it must become a string.
METRIC_CLASSES = {"A": "per-realization", "B": "pooled", "C": "pooled"}


def q_metric_name(statistic: str) -> str:
    """Composite metric name for one discharge statistic.

    Took ``tpeak``/``tlow`` arguments until 2026-08-12, when the return periods
    stopped being config values; the names they produced are unchanged.
    """
    suffix, _ = Q_METRIC_SUFFIXES[statistic]
    return f"q_{suffix}"


def basin_metric_name(token: str) -> str:
    """Composite metric name for one basin-scalar variable."""
    suffix, _ = BASIN_METRIC_SUFFIXES[token]
    return f"{token}_{suffix}"


def output_code(token: str) -> str:
    """The wflow csv column code for one indicator token (``'precip'`` → ``'p'``)."""
    return OUTPUT_CODES[token]


def basin_reduction(token: str) -> str:
    """The annual reduction a basin-scalar variable uses (``sum``/``max``/``mean``)."""
    return BASIN_METRIC_SUFFIXES[token][1]


def metric_grain(token: str, metric: str) -> str | None:
    """``'per-realization'`` / ``'pooled'`` for a metric name, or ``None`` if unknown.

    A plain enumeration since 2026-08-12. It was a PATTERN — the two return-level
    suffixes interpolated ``Tpeak``/``Tlow``, so their names were partly
    config-derived and an enumerating validator would have rejected every project
    whose return periods differed from the fixture's. Retiring those config keys
    (:data:`RETURN_PERIOD_PEAK_YR`) makes the vocabulary closed, so the check can
    say exactly which names are legal instead of which shapes are.

    Returning ``None`` rather than raising lets the caller report *which* metric is
    unrecognised alongside its other findings, instead of dying on the first one.
    """
    if token == "q":
        for suffix, grain_class in Q_METRIC_SUFFIXES.values():
            if metric == f"q_{suffix}":
                return METRIC_CLASSES[grain_class]
        return None
    if token in BASIN_METRIC_SUFFIXES:
        # Basin-scalar metrics are linear in years, so they carry the finest
        # grain available, exactly as class A does.
        return "per-realization" if metric == basin_metric_name(token) else None
    return None


#: Config keys ``workflows.climate_experiment`` no longer has, and what to tell
#: someone whose config still declares one. Keyed by config key; each entry is
#: ``{"why": <what to do about it>, "note": <where the migration is written>}``.
#:
#: Refused rather than ignored, per the ``variable_spec.parse`` precedent. The
#: hazard is specific to this repo's config handling: **workflow configs silently
#: ignore keys nothing reads**, so a retired key does not fail, does not warn, and
#: does not take effect — it leaves a user believing a setting is in force while
#: it does nothing. A refusal that states the migration is strictly better than a
#: setting that lies.
#:
#: **Registering a retirement here is not optional.** It was, in effect, until
#: 2026-08-12: ``Tpeak``/``Tlow`` were removed from every config and from the
#: reader without an entry, so for the length of one commit a project declaring
#: ``Tpeak: 25`` got exactly the silent no-op this registry exists to prevent.
#: Nothing catches the omission — the removal itself makes the key unread, which
#: is indistinguishable from it never having existed. The obligation is on
#: whoever retires the key.
#:
#: The migration note is PER ENTRY rather than one module constant, because
#: retirements come from different milestones and a single pointer would send a
#: reader to the wrong record. It was one constant while only R11 had retired
#: anything.
RETIRED_EXPERIMENT_KEYS = {
    "aggregate_rlz": {
        "why": (
            "In the long table shape 'aggregated' is no longer a SHAPE choice, "
            "which is the only reason this flag existed. Every table now carries "
            "the finest grain available -- metrics linear in years per "
            "realization, the GEV fits and month-selecting metrics pooled -- and "
            "downstream aggregates as it likes. Delete the line; nothing "
            "replaces it."
        ),
        "note": "dev/milestones/r11/migration_indicator-tables.md",
    },
    "Tpeak": {
        "why": (
            "A return period is a property of the indicator set the toolbox "
            "defines, not of a project, so it moved to "
            "indicator_tables.RETURN_PERIOD_PEAK_YR (10 -- the value every "
            "shipped config carried, so no emitted name or number changed). "
            "Delete the line. A different design standard is a toolbox edit, "
            "which re-names the indicator and is meant to be visible."
        ),
        "note": "dev/reviews/2026-08-11_test-suite-bloat-assessment.md",
    },
    "Tlow": {
        "why": (
            "As Tpeak: now indicator_tables.RETURN_PERIOD_LOW_YR (2). Delete the line."
        ),
        "note": "dev/reviews/2026-08-11_test-suite-bloat-assessment.md",
    },
}

#: Kept as a name because the R11 tests and the migration note both cite it. It
#: is now the note for ONE entry rather than for the registry.
MIGRATION_NOTE = RETIRED_EXPERIMENT_KEYS["aggregate_rlz"]["note"]


class RetiredConfigKeyError(ValueError):
    """A project config still declares a key the toolbox has removed."""


def refuse_retired_experiment_keys(experiment_cfg) -> None:
    """Raise if ``workflows.climate_experiment`` still declares a retired key.

    Called at DAG-construction time so the run stops before producing anything,
    rather than after a sweep whose grain silently ignored the setting.
    """
    if not isinstance(experiment_cfg, dict):
        return
    found = sorted(key for key in RETIRED_EXPERIMENT_KEYS if key in experiment_cfg)
    if not found:
        return
    lines = [
        f"workflows.climate_experiment declares {len(found)} retired key(s): "
        f"{', '.join(found)}."
    ]
    for key in found:
        entry = RETIRED_EXPERIMENT_KEYS[key]
        lines.append(f"\n  {key}: {entry['why']}")
        lines.append(f"\n    migration: {entry['note']}")
    raise RetiredConfigKeyError("".join(lines))


class UnknownOutputVariableError(ValueError):
    """``wflow_outvars`` names a variable with no token, so no table can be named.

    Raised rather than skipped. A silently ignored entry would produce a run whose
    results are missing a variable the config asked for, with nothing in the tree
    saying so — and the absence would look identical to "that variable was never
    requested".
    """


def variable_token(outvar: str) -> str:
    """Short token for one ``wflow_outvars`` entry."""
    try:
        return VARIABLE_TOKENS[outvar]
    except KeyError:
        known = ", ".join(sorted(VARIABLE_TOKENS))
        raise UnknownOutputVariableError(
            f"wflow_outvars names {outvar!r}, which has no indicator-table token. "
            f"Known variables: {known}. Add the variable to VARIABLE_TOKENS in "
            f"blueearth_cst/shared/indicator_tables.py AND to the seam contract "
            f"dev/reference/contracts/hydrological-model-seam.md, which is where "
            f"this mapping is published."
        ) from None


def indicator_table_filename(outvar: str) -> str:
    """``'river discharge'`` → ``'q_indicators.csv'``."""
    return f"{variable_token(outvar)}{_TABLE_SUFFIX}"


def indicator_tables(wflow_outvars) -> dict[str, str]:
    """Map each configured output variable's TOKEN to its table filename.

    Keyed by token rather than by semantic name because the token is what every
    consumer downstream carries: it is in the filename, in the composite
    ``metric``, and in the Snakemake output key. Keying by the semantic name would
    make ``"river discharge"`` a dict key with a space in it, which then has to be
    translated at every use site.

    Order follows ``wflow_outvars`` so the derived output set is stable for a
    given config; duplicates collapse, since two identical entries would name one
    table.
    """
    tables: dict[str, str] = {}
    for outvar in wflow_outvars or []:
        token = variable_token(outvar)
        tables[token] = f"{token}{_TABLE_SUFFIX}"
    return tables
