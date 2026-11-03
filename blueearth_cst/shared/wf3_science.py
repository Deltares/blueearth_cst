# Current guidance: dev/reference/toolbox-spatial-foundation.md
"""WF3-safe scientific helpers shared with other workflows.

The generation code inventory includes this whole file. Keep WF4-only code in
``snake_utils``, and settings schema in ``advanced_settings``, so their edits
cannot change a collection identity.
"""

import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from blueearth_cst.shared.advanced_settings import (
    _MONTH_ABBREVS,
    ADVANCED_SETTINGS,
    _month_abbrev,
    _nonnegative_int,
)

DEFAULT_SEED = ADVANCED_SETTINGS["defaults"]["seed"]

_SEED_MODULUS = 2**31


def derive_seed(experiment_name: str) -> int:
    """The seed ``seed: auto`` resolves to, from the experiment name.

    Deterministic on the name alone: the same experiment re-runs with the same
    seed (so nothing downstream re-runs), a different experiment gets a
    different one, and "what seed did experiment X use?" stays answerable
    forever without reading any artifact.

    **``zlib.crc32``, never the builtin ``hash``.** ``hash`` is salted per
    process by ``PYTHONHASHSEED``, so it would return a different seed for the
    same experiment in every interpreter — silently breaking the one property
    this function exists to provide, and doing it in a way that looks like
    reproducible behaviour until two runs are compared.
    """
    if not isinstance(experiment_name, str) or not experiment_name:
        raise ValueError(
            f"cannot derive a seed from experiment_name={experiment_name!r}: "
            "`seed: auto` needs the experiment's name, which WF3 always "
            "resolves before rule 3.09 runs"
        )
    return zlib.crc32(experiment_name.encode("utf-8")) % _SEED_MODULUS


def resolve_seed(value, experiment_name: str) -> int:
    """Resolve the generate_scenarios ``seed:`` to the integer the generator is handed.

    ``None`` (key absent) takes ``defaults.seed``; ``auto`` derives from the
    experiment name; anything else must be a non-negative integer. A string
    that is not ``auto`` is refused rather than coerced — ``seed: "123"`` and
    ``seed: random`` would otherwise both reach weathergenr, one working by
    accident and one as ``NULL``.
    """
    if value is None:
        value = DEFAULT_SEED
    if isinstance(value, str):
        if value.strip().lower() != "auto":
            raise ValueError(
                f"`seed:` in the generate_scenarios config must be an integer or the literal 'auto', got {value!r}"
            )
        return derive_seed(experiment_name)
    return _nonnegative_int(value, "`seed:` in the generate_scenarios config")


DEFAULT_WATER_YEAR_START = ADVANCED_SETTINGS["defaults"]["water_year_start"]


def resolve_water_year_start(value) -> str:
    """Resolve ``shared.water_year_start`` to a canonical ``Jan``-style month."""
    if value is None:
        value = DEFAULT_WATER_YEAR_START
    return _month_abbrev(value, "shared.water_year_start")


def water_year_start_number(month: str) -> int:
    """Calendar month number 1..12 — what weathergenr's ``year_start_month`` takes."""
    return _MONTH_ABBREVS.index(_month_abbrev(month, "water_year_start")) + 1


def historical_window_bounds(historical_window):
    """``(start, end)`` of ``climate.window``, as datetimes.

    **The R14 retype is absorbed here, and only here** (`C-70`). The project
    config now declares ``climate.window: {start, end}`` as INCLUSIVE YEARS.
    This pair of helpers is the one place that already parsed the window, so
    every caller downstream — ``climate_window.py``, ``add_climate_forcing.py``,
    ``extract_historical_climate.py``, ``reference_window.py`` — keeps receiving
    exactly the datetimes it received before, and none of them changed.

    Inclusive means ``{start: 2000, end: 2016}`` spans 2000-01-01 to 2016-12-31.
    That is value-preserving for every config the toolbox ships: their v1 ISO
    endpoints were already whole-year aligned on exactly those two dates.

    Raises ``ValueError`` naming the offending key when an endpoint is missing
    or is not a year — the same fail-loud stance ``slugify_window`` takes on
    the same two values.
    """
    if not isinstance(historical_window, Mapping):
        raise ValueError(
            f"climate.window must be a mapping with start/end years, got "
            f"{historical_window!r}"
        )
    years = []
    for key in ("start", "end"):
        if key not in historical_window:
            raise ValueError(
                f"climate.window is missing {key!r}. It is a pair of INCLUSIVE "
                "YEARS now, not ISO timestamps: `window: {start: 1990, end: 2020}`."
            )
        try:
            years.append(int(str(historical_window[key]).strip()))
        except (TypeError, ValueError):
            raise ValueError(
                f"climate.window.{key} is not a year: {historical_window[key]!r}. "
                "`climate.window` takes INCLUSIVE YEARS, not ISO timestamps."
            ) from None
    start, end = years
    return (datetime(start, 1, 1), datetime(end, 12, 31))


def window_year_pair(window, key):
    """``[start, end]`` CALENDAR years from a ``{start, end}`` mapping.

    R14 retypes several windows from a two-element list to a mapping of
    INCLUSIVE YEARS. ``historical_window_bounds`` absorbs that for
    ``climate.window`` and returns datetimes, because every one of its callers
    wanted datetimes. The windows this helper serves want the YEARS: WF2's
    ``reference_window`` (`C-59`) is clipped against the GCM historical
    experiment as integers, and its result reaches the digest.

    **No water-year offset is applied, deliberately** (`C-74`, D-7.4).
    ``hydrological_year_bounds()`` already trims to complete water years one
    layer down, so routing a calendar window through the water-year path would
    apply the offset twice and move every change factor without saying so.

    ``key`` names the config key in the error, because by the time this raises
    the caller has usually lost track of which of several windows it was.
    """
    if not isinstance(window, Mapping):
        raise ValueError(
            f"{key} must be a mapping with start/end years, got {window!r}"
        )
    years = []
    for bound in ("start", "end"):
        if bound not in window:
            raise ValueError(f"{key} is missing `{bound}`; got {window!r}")
        value = window[bound]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"{key}.{bound} must be a whole year, got {value!r}. R14 retyped "
                "this key from a two-element list to inclusive years."
            )
        years.append(value)
    if years[0] > years[1]:
        raise ValueError(f"{key}.start ({years[0]}) is after end ({years[1]})")
    return years


def slugify_window(start, end) -> str:
    """Render a window ``(start, end)`` to a compact ``YYYYMMDD_YYYYMMDD`` slug.

    Builds the dataset-store key component for the wf3 historical-climate store
    (dev/records/milestones/p31/experiment-structure-design.md §4/§4c/§4d). The store dir is
    ``data/climate/historical/<clim_source>_<start>_<end>/`` where
    ``<start>``/``<end>`` are this function's output. The window endpoints are ISO
    ``YYYY-MM-DDTHH:MM:SS``; ``:`` is illegal in Windows paths, so time-of-day and
    separators are stripped to ``YYYYMMDD``.

    Day-resolution invariant (§4c): the store is keyed at day resolution, so two
    windows differing ONLY below the day boundary would render to the same key
    yet request different bounds — a silent stale-reuse. This helper therefore
    **asserts** ``HH:MM:SS == 00:00:00`` on both endpoints and raises
    ``ValueError`` otherwise, failing loud instead of colliding.

    Parameters
    ----------
    start, end : str
        Window endpoints as ISO ``YYYY-MM-DDTHH:MM:SS`` (or ``YYYY-MM-DD``).

    Returns
    -------
    str
        ``"<YYYYMMDD>_<YYYYMMDD>"``.

    Raises
    ------
    ValueError
        If an endpoint is not parseable at day resolution, or carries a nonzero
        time-of-day component.
    """

    def _day_slug(value, which):
        text = str(value).strip()
        # Split date from an optional time-of-day on the 'T' separator (or a space).
        if "T" in text:
            date_part, time_part = text.split("T", 1)
        elif " " in text:
            date_part, time_part = text.split(" ", 1)
        else:
            date_part, time_part = text, ""
        try:
            dt = datetime.strptime(date_part, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(
                f"climate.window {which} {value!r} is not a YYYY-MM-DD date"
            ) from exc
        if time_part:
            # Accept only an all-zero time-of-day; anything else is sub-day
            # resolution the day-keyed store cannot represent (§4c). Drop any
            # fractional seconds, then check every digit is zero.
            hms = time_part.split(".", 1)[0]
            if hms.replace(":", "").strip("0") != "":
                raise ValueError(
                    f"climate.window {which} {value!r} has a nonzero "
                    "time-of-day; the store key is day-resolution (§4c) — "
                    "sub-day windows are not supported"
                )
        return dt.strftime("%Y%m%d")

    return f"{_day_slug(start, 'starttime')}_{_day_slug(end, 'endtime')}"


DEFAULT_HYDROGRAPHY = ADVANCED_SETTINGS["defaults"]["hydrography"]

DEFAULT_BASIN_INDEX = ADVANCED_SETTINGS["defaults"]["basin_index"]

CLIMATE_STORE_SCRIPT = "blueearth_cst/climate_analysis/extract_historical_climate.py"

REGION_SCRIPT = "blueearth_cst/spatial/delineate_region.py"


@dataclass(frozen=True)
class RegionRule:
    """The producer contract for the one project region artifact.

    Same shape and same purpose as :class:`ClimateStoreRule`: all three
    workflows declare ``delineate_region`` from this object, so the three rule
    bodies cannot drift apart.
    """

    region_geojson: str
    script: str
    inputs: Mapping
    outputs: Mapping
    params: Mapping


def region_geojson_path(project_dir):
    """The project's one delineated-region artifact, given its root.

    R9 P2 commit 2: engine-neutral geometry lives under `data/` (design v10).
    Defined ONCE here and splatted into all three workflows' delineate_region
    rule through :func:`region_rule`, so a move lands in every workflow at the
    same instant -- tests/test_region_rule.py parses all three and fails on any
    difference.

    Split out of `region_rule` so a caller that wants only the PATH need not
    invent a region specification and a catalog to get one:
    `scripts/run_workflows.py` reads the polygon to state the project's bounding
    box in its opening block, long before any rule has been built.
    """
    return f"{project_dir}/data/spatial/geoms/region.geojson"


def region_rule(
    project_dir,
    model_region,
    data_sources,
    hydrography=DEFAULT_HYDROGRAPHY,
    basin_index=DEFAULT_BASIN_INDEX,
) -> RegionRule:
    """Build the one producer contract for ``spatial/geoms/region.geojson``.

    ONE rule definition, declared in all three workflows (``1.01b`` / ``2.03b``
    / ``3.01b``), over the ONE delineation of ``shared.basin.region``. Before
    the shared-region refactor the polygon was derived twice — by rule 1.01 on its way to
    ``basins.geojson``, and per climate-store key as ``store_region.geojson`` —
    and WF2 ran the whole climate-store producer to obtain it.

    The artifact lives in ``spatial/geoms/`` beside ``basins.geojson``,
    ``catchments.geojson`` and ``locations.geojson``: that is where this project
    keeps the vector description of where the model is.

    Model-free by construction, which is the property R07 B1 bought for the
    climate store and this must preserve — the single declared input is the data
    catalog, and the params carry only the region specification and the two
    catalog ENTRY NAMES (not paths) for the delineation.

    Parameters
    ----------
    project_dir : str
        ``project.project_dir``.
    model_region : str | Mapping
        ``shared.basin.region`` — the hydromt region specification (usually a
        Python-dict-literal string). Carried in ``params``, never resolved here.
    data_sources : str
        ``project.data_sources`` — the hydromt catalog path. The single declared
        input.
    hydrography, basin_index : str
        ``shared.basin.hydrography`` / ``shared.basin.basin_index`` — catalog
        entry names for the delineation. The defaults equal the shipped build
        template's ``setup_basemaps`` values; rule 1.01 fails loud if the two
        ever disagree.

    Returns
    -------
    RegionRule
        ``region_geojson``, ``script``, ``inputs``, ``outputs``, ``params``.
    """
    region_geojson = region_geojson_path(project_dir)
    return RegionRule(
        region_geojson=region_geojson,
        script=REGION_SCRIPT,
        inputs={"catalog": data_sources},
        outputs={"region_geojson": region_geojson},
        params={
            "model_region": model_region,
            "hydrography": hydrography,
            "basin_index": basin_index,
        },
    )


@dataclass(frozen=True)
class ClimateStoreRule:
    """The complete producer contract for the shared historical-climate store.

    Attribute-accessible and dict-splattable: a Snakefile writes

    ``input: **SPEC.inputs`` / ``output: **SPEC.outputs`` /
    ``params: **SPEC.params`` / ``script: SPEC.script``

    so every content- or execution-determining field of the two declarations
    comes from one object rather than from two hand-maintained rule bodies.
    """

    store_dir: str
    script: str
    inputs: Mapping
    outputs: Mapping
    params: Mapping


def climate_store_rule(
    project_dir,
    model_region,
    clim_source,
    historical_window: Mapping,
    data_sources,
    hydrography=DEFAULT_HYDROGRAPHY,
    basin_index=DEFAULT_BASIN_INDEX,
    enforce_min_years=True,
    forcing_required=True,
) -> ClimateStoreRule:
    """Build the one producer contract for ``data/climate/historical/<key>/``
    (R07 B1).

    ONE rule definition, declared in ``build_model.smk`` (rule 1.03) and
    ``generate_scenarios.smk`` (rule 3.02) as ``extract_climate_datasets``, and
    generated per candidate source by ``analyze_climate.smk`` (rule 0.03) as
    ``extract_historical_climate_<source>``. All three resolve to the same
    store directory, so whichever workflow runs first extracts and the others
    read what is already there. Over the
    model-independent region specification + data catalog. wf1's `wf1_raw/`
    store and its `staticmaps.nc`-derived bbox are retired: the extent is now a
    pure function of ``shared.basin`` + the catalog, so a climate-only run needs
    no ``models/hydrology/wflow/`` on disk and a region change re-extracts through
    Snakemake's params rerun-trigger (design § B1).

    **The input set is exactly the catalog and shared region in both DAGs.** An
    asymmetric input set re-creates the wf1<->wf3 re-extraction oscillation
    (design P2(b) / ext1-02). The catalog **file** is the source freshness
    boundary (ext2-01), while the region declares the extraction extent; both
    are plain inputs, never ``ancient()``. Data behind an unchanged catalog
    entry is out of scope — edit the entry, or use ``snakemake --forcerun
    extract_climate_datasets`` (in wf0, name the generated source rule).

    Parameters
    ----------
    project_dir : str
        ``project.project_dir``; the store lands under
        ``<project_dir>/data/climate/historical/``.
    model_region : str | Mapping
        ``shared.basin.region`` — the hydromt region specification (usually a
        Python-dict-literal string). Carried in ``params``, never resolved here.
    clim_source : str
        ``shared.clim_historical``. Selects the chirps orography branch.
    historical_window : Mapping
        The ``shared.historical_window`` section, with ``starttime`` and
        ``endtime``. Keyed at day resolution by ``slugify_window``.
    data_sources : str | sequence of str
        ``project.catalog`` — the HydroMT catalog path or ordered paths.
    hydrography, basin_index : str
        ``shared.basin.hydrography`` / ``shared.basin.basin_index`` — catalog
        ENTRY NAMES for the delineation, not paths. Optional config keys; the
        defaults equal the shipped build template's ``setup_basemaps`` values,
        and rule 1.01 fails loud if the two ever disagree.
    enforce_min_years : bool, optional
        Whether a DELIVERED record below ``MIN_HISTORICAL_YEARS`` fails the
        extraction. ``True`` for every store that feeds the pipeline — which is
        every caller except wf0's extra ``candidate_sources``, whose stores end
        at a comparison figure (2026-08-16 owner ruling; see
        ``shared/climate_window.py``).

        ``True`` emits **no param at all**, rather than ``enforce_min_years:
        True``. The params dict is a Snakemake rerun trigger, so adding a key to
        the default path would re-extract every store already on disk and break
        the byte-identity ``tests/test_climate_store_contract.py`` pins across
        the four workflows. Only the relaxed candidates carry the key — and
        because they carry it, promoting one to ``shared.clim_historical``
        changes the params WF1/WF3 declare and re-extracts it under the floor.
    forcing_required : bool, optional
        Whether a precipitation-only source must be enriched into the full
        seven-variable forcing store required by WF1/WF3. ``False`` only for
        wf0's extra comparison candidates. Like ``enforce_min_years``, the
        default is omitted from params so selected-store declarations remain
        unchanged; promotion therefore re-extracts the full forcing store.

    Returns
    -------
    ClimateStoreRule
        ``store_dir``, ``script``, ``inputs``, ``outputs``, ``params``.

    Raises
    ------
    TypeError
        If ``historical_window`` is not a mapping.
    ValueError
        If either window endpoint is missing, or carries a sub-day component
        the day-resolution store key cannot represent (``slugify_window``).
    """
    if not isinstance(historical_window, Mapping):
        raise TypeError(
            "climate_store_rule: historical_window must be the `climate.window` "
            "mapping with 'start'/'end' years, got "
            f"{type(historical_window).__name__}"
        )
    # Through the one parser (R14 `C-70`), so the store key is derived from the
    # same bounds every other reader sees. The KEY IS UNCHANGED by the retype:
    # every shipped config's v1 ISO endpoints were whole-year aligned, so
    # `{start: 2000, end: 2020}` slugs to the same `20000101_20201231` the ISO
    # pair did -- which is what keeps an extracted store from being re-extracted
    # into a new directory on migration.
    _start, _end = historical_window_bounds(historical_window)
    # Rendered back to ISO for `params:`. BYTE-IDENTICAL to the v1 values: the
    # shipped configs' endpoints were whole-year aligned, so `{2000, 2020}`
    # renders the same `2000-01-01T00:00:00` / `2020-12-31T00:00:00` the ISO
    # pair carried -- which is what keeps the params digest, and every rule that
    # threads these two values, unmoved by the retype.
    starttime, endtime = _start.isoformat(), _end.isoformat()

    # Byte-for-byte the key wf3 built inline before R07 (P3-1 §4/§4c/§4d): two
    # experiments sharing clim_historical + historical_window resolve to the
    # same dir and reuse the extraction.
    store_key = f"{clim_source}_{slugify_window(starttime, endtime)}"
    # R9 P2 commit 2: the store moves under `data/climate/`, and the KEY IS
    # RETAINED. `<clim_source>_<window>` is a cache key, not multi-window
    # support (R9 design Finding 3): two experiments sharing a source and a
    # window must still resolve to the same directory and reuse the extraction,
    # so the path stays EXPERIMENT-INVARIANT across the move. That invariant is
    # this commit's, and it is why the key survives the relocation unchanged.
    store_dir = f"{project_dir}/data/climate/historical/{store_key}"

    outputs = {
        "climate_nc": f"{store_dir}/extract_historical.nc",
        # Which extracted cells the basin TOUCHES. Part of the store contract
        # rather than a WF3-local artifact: the store is what both workflows
        # share, and the mask is a property of this extraction's grid, so it is
        # only derivable where the grid and the region polygon meet. Consumers
        # (rule 3.10) average over exactly these cells instead of over every
        # cell the bbox+buffer read happened to include.
        "basin_cells": f"{store_dir}/basin_cells.csv",
    }
    if forcing_required and clim_source in ("chirps", "chirps_global"):
        # Resolved at parse time from clim_historical, so there are no dynamic
        # outputs. The filename is clim_source-INDEPENDENT (R07 standardises the
        # two pre-R07 spellings on `orography.nc`).
        outputs["oro_nc"] = f"{store_dir}/orography.nc"

    params = {
        "model_region": model_region,
        "clim_source": clim_source,
        "starttime": starttime,
        "endtime": endtime,
        "hydrography": hydrography,
        "basin_index": basin_index,
    }
    # Present ONLY when relaxed — see the parameter's docstring for why the
    # default path must emit no key.
    if not enforce_min_years:
        params["enforce_min_years"] = False
    if not forcing_required:
        params["forcing_required"] = False

    return ClimateStoreRule(
        store_dir=store_dir,
        script=CLIMATE_STORE_SCRIPT,
        # Two declared inputs since the shared-region refactor. The catalog is the store's
        # freshness boundary (ext2-01); the region is the extent it cuts to,
        # produced once per project by `delineate_region` rather than
        # re-delineated per store key. `region_rule` owns the path, so the two
        # helpers cannot disagree about where the polygon lives.
        inputs={
            "catalog": data_sources,
            "region_geojson": region_rule(
                project_dir,
                model_region,
                data_sources,
                hydrography=hydrography,
                basin_index=basin_index,
            ).region_geojson,
        },
        outputs=outputs,
        params=params,
    )


_AXIS_SUBKEYS = {
    "temp": frozenset({"n_levels", "trajectory", "mean"}),
    "precip": frozenset({"n_levels", "trajectory", "mean", "variance"}),
}


def _reject_unknown_axis_subkeys(stress_test_cfg: Mapping) -> None:
    """Refuse a sub-key no axis reads, naming it and what the axis accepts."""
    for axis, allowed in _AXIS_SUBKEYS.items():
        axis_cfg = stress_test_cfg.get(axis)
        if not isinstance(axis_cfg, Mapping):
            continue
        unknown = sorted(set(axis_cfg) - allowed)
        if not unknown:
            continue
        detail = ""
        # The two R14 spellings get their destination named rather than just
        # being listed as unsupported: `n_levels` is a RETYPE of `step_num`
        # (+1) and `trajectory` an enum where `transient_change` was a bool,
        # so neither is fixed by copying the old value across.
        if "step_num" in unknown:
            detail += (
                " `step_num` is now `n_levels` and counts LEVELS, not intervals:"
                " `n_levels` = `step_num` + 1 (`C-31`)."
            )
        if "transient_change" in unknown:
            detail += (
                " `transient_change: true` is now `trajectory: transient`"
                " (`C-32`); it is required, with no default."
            )
        if axis == "temp" and "variance" in unknown:
            detail = (
                " Temperature variance is not a supported stress dimension: "
                "only precipitation variance reaches the weather generator. "
                "Remove it rather than expecting it to perturb anything."
            )
        raise ValueError(
            f"workflows.generate_scenarios.climate_perturbations.{axis} carries "
            f"unsupported key(s) {unknown}; it accepts {sorted(allowed)}.{detail}"
        )


def _require_n_levels(axis_cfg, axis_name):
    """Read and validate a required ``n_levels`` from a perturbation axis.

    **`C-31` is a RETYPE, not a rename.** ``step_num`` counted INTERVALS and
    every caller added one for the endpoints; ``n_levels`` is that sum, the
    number of grid levels on the axis, declared directly. So ``step_num: 1``
    and ``n_levels: 2`` describe the same axis, and a config that merely
    renamed the key without adding one would silently drop a level from every
    axis — which is why the old spelling is refused rather than accepted.

    The floor moves with the meaning: zero intervals was legal (a single
    unperturbed level), so the minimum level count is ONE, not zero.

    Strict by contract: a missing axis section or ``n_levels`` raises
    ``KeyError`` (parity with ``prepare_cst_parameters.py``'s direct read); a
    value that is not a positive integer raises ``ValueError``. ``bool`` is
    rejected — ``True``/``False`` are not valid level counts.
    """
    n_levels = axis_cfg[axis_name]["n_levels"]  # KeyError on missing axis/key
    if isinstance(n_levels, bool) or not isinstance(n_levels, int):
        raise ValueError(
            f"climate_perturbations.{axis_name}.n_levels must be a positive "
            f"int, got {n_levels!r}"
        )
    if n_levels < 1:
        raise ValueError(
            f"climate_perturbations.{axis_name}.n_levels must be at least 1 "
            f"(one level = the unperturbed axis), got {n_levels}"
        )
    return n_levels


def stress_test_grid(stress_test_cfg: Mapping) -> tuple[int, int, int]:
    """Return ``(temp_step_count, precip_step_count, st_num)`` for a stress_test cfg.

    Single source of truth for the stress-test grid arithmetic, which was
    previously derived twice (inline in ``run_stress_test.smk`` and in
    ``blueearth_cst/experiment/prepare_cst_parameters.py``). Both call sites now read this helper.

    STRICT: ``temp.n_levels`` and ``precip.n_levels`` are REQUIRED — a missing
    axis section or ``n_levels`` raises ``KeyError``, and a value that is not a
    positive integer raises ``ValueError``. The helper never silently invents a
    grid. Per-axis level count IS ``n_levels`` since `C-31` (it was
    ``step_num + 1``), and ``st_num = temp_step_count * precip_step_count``.

    Parameters
    ----------
    stress_test_cfg : Mapping
        The ``workflows.generate_scenarios.climate_perturbations`` config section,
        with ``temp`` and ``precip`` axis sub-sections each carrying
        ``n_levels``.

    Returns
    -------
    tuple[int, int, int]
        ``(temp_step_count, precip_step_count, st_num)``.

    Raises
    ------
    KeyError
        If the ``temp``/``precip`` axis section or its ``n_levels`` is absent.
    ValueError
        If an ``n_levels`` is not a positive integer.
    """
    _reject_unknown_axis_subkeys(stress_test_cfg)
    # No `+ 1` any more: `C-31` moved that addition into the config, where the
    # author can see it. The RESULT is unchanged for an equivalent config.
    temp_step_count = _require_n_levels(stress_test_cfg, "temp")
    precip_step_count = _require_n_levels(stress_test_cfg, "precip")
    return temp_step_count, precip_step_count, temp_step_count * precip_step_count


DEFAULT_SPELL_FACTOR = list(ADVANCED_SETTINGS["defaults"]["spell_factor"])


def validate_spell_factor(value, where: str) -> list[float]:
    """Validate a monthly spell-length coefficient list from ``stress_test``.

    Twelve numbers, one per calendar month. ``None`` (key absent) yields the
    identity, because "no adjustment" is a defensible default in a way that,
    say, ``transient_change`` is not — there the house rule is to refuse.

    The LENGTH check is the point. weathergenr indexes these by month, so a
    ten-element list would be recycled or truncated by R rather than rejected,
    and the run would silently perturb the wrong months.
    """
    if value is None:
        return list(DEFAULT_SPELL_FACTOR)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            f"{where} must be a list of 12 monthly coefficients, got "
            f"{value!r} ({type(value).__name__})"
        )
    if len(value) != 12:
        raise ValueError(
            f"{where} must have 12 entries, one per month, got {len(value)}"
        )
    out = []
    for index, item in enumerate(value, start=1):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{where}[{index}] must be a number, got {item!r}")
        out.append(float(item))
    return out


def index_width(count: int) -> int:
    """Digits needed to render ``0..count`` so LEXICAL order matches NUMERIC.

    C27: the width is derived from the COUNT, never fixed. A 6-member grid gets
    width 1 (``st_1``) because ``st_1 … st_6`` already sort correctly; a
    12-member grid gets width 2 (``st_01 … st_12``) because ``st_1, st_11,
    st_2`` does not. 100 members gets 3.

    This is what makes an ``ls``, a glob expansion, an IDE tree and the WG-5
    catalog's key order read in run order. It is NOT cosmetic for the design
    table: C28 puts ``st_id`` in the indicator tables, and padding both from
    this one function makes the column and the filename textually identical, so
    a consumer joining a plot to its run needs no integer coercion.

    **The width is stable for a collection's life.** It is a function of
    ``ST_NUM`` / ``RLZ_NUM``, both of which feed the generation request's
    fingerprint — so a grid change that would move the width yields a different
    ``collection_id``, and therefore a different collection directory, rather
    than re-widening an existing one. No existing tree can be renamed underneath
    itself.

    Argued through ``experiment.yml``'s freeze and ``check_not_frozen`` until
    2026-09-17. That mechanism was never wired into R12's rule set and its
    module has been deleted (``t2608290250``); content-addressed identity gives
    the same guarantee structurally, by making a changed grid a NEW artifact
    instead of a refused write.

    Raises
    ------
    ValueError
        If ``count`` is not a positive integer. A zero or negative count has no
        width, and returning 1 for it would paper over a broken grid.
    """
    if isinstance(count, bool) or not isinstance(count, int):
        raise ValueError(f"index_width needs a positive int count, got {count!r}")
    if count < 1:
        raise ValueError(f"index_width needs a positive count, got {count}")
    return len(str(count))
