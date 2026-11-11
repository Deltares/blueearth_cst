"""Compose a project config and its per-workflow files into one mapping.

The project file carries closed ``{enabled, config_path}`` workflow stanzas;
each workflow file holds that workflow's settings. This module validates the
boundary and composes the files for existing consumers.

Current guidance: ``dev/reference/toolbox-architecture.md``, configuration
and execution records. This module owns the exact schema and composition.
Section references and ``D-`` tags below identify historical rationale in
``dev/records/milestones/r13/config-tiers-design.md``, not additional requirements.

**Schema 2 preserves the composition invariant (D-8.1).** ``compose_config``
returns a mapping whose shape is identical to the pre-split ``config`` dict:
``config["workflows"]["generate_scenarios"]`` and ``config["basin"]``
resolve exactly as they did, to exactly the same values.
Because that holds, ``effective_config_digest``, ``guarded_sections_digest``, the
experiment freeze, ``resolve_simulation_window`` and every ``get_config`` call
site are unchanged *by construction* — same inputs, same bytes hashed. Two
deviations are declared and only two: the narrowing (§8.1, §10.7) and the
rebinding requirement (D-8.6). Schema 3 additionally resolves declared file
paths beside their owning YAML, using config_paths.py. Exact source bytes
remain in archives; runtime mappings carry resolved locators.

**The shared seam (§9).** A key read by more than one workflow lives in T1, never
in a T2 file. Convention cannot enforce that, and a full key registry would be a
second source of truth that drifts, so the rule is carried by four closed
parse-time checks — D-9.1 (the T1 stanza is closed), D-9.2 (a T2 file may not
declare a T1-owned name), D-9.3 (the same rejection, widened
to every declared T2 file that resolves), D-9.5 (T1's top level is closed) —
plus the static read scan (D-9.6) that governs the one case parse time cannot
see: a value read reaching *across* the section boundary at runtime.

**Design invariants this module obeys (do not relax without a design change):**

1. **Pure.** ``compose_config`` takes an already-parsed mapping plus a path and
   does no I/O beyond reading the T2 files it is told to read. No global state.
   That is what makes it callable outside Snakemake, which the raw-T1 tool
   consumers depend on (D-12.0, §12.0).
2. **Composition moves keys between files; it never creates or drops one.**
   ``effective_config_document`` digests the config *mapping*, so a key that is
   present rather than absent moves ``effective_config_digest`` even when the
   resolved value is identical. Adding one during composition would therefore
   re-key every already-generated collection in the project: the digest moves,
   so nothing matches what is on disk and the work is regenerated under a new
   identity. Hence D-8.7: an omitted ``config_path`` composes to ``{}`` and adds
   nothing.

   Stated as a refusal through ``_frozen_differences`` until 2026-09-17, when
   that comparator was deleted as unreachable (``t2608290250``). The conclusion
   is unchanged and the cost is, if anything, larger: R12 regenerates rather
   than refusing, so the damage is silent recomputation, not an error.
3. **``enabled`` is merged back in; ``config_path`` is not** (D-10.1). Dropping
   ``enabled`` would refuse every already-run experiment; adding ``config_path``
   would make run identity depend on where a project stores its files.
4. **Fail loud at parse time.** Every refusal is a ``ValueError`` raised while
   the Snakefile body executes, before the DAG is built, so
   ``pytest tests/test_cli.py`` is their gate. Messages name the resolved
   absolute path, the anchor it was resolved against, and the migration doc.

Stdlib + pyyaml only — no new dependency.
"""

from __future__ import annotations

import os
import re
import tokenize
from collections.abc import Iterable, Mapping, Sequence
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import yaml

from blueearth_cst.shared.config_paths import (
    resolve_project_paths,
    resolve_workflow_paths,
)
from blueearth_cst.shared.named_forcings import (
    declares_named,
    historical_forcing_ids,
    legacy_view,
    named_recipes,
    simulation_climate,
)
from blueearth_cst.shared.plot_config import resolve_annotation

if TYPE_CHECKING:
    from blueearth_cst.shared.workflow_config_snapshot import CapturedSource

# ---------------------------------------------------------------------------
# Contract constants
# ---------------------------------------------------------------------------

#: The release that still ships the v1 -> v2 migrator and its guide. Both were
#: retired from the toolbox after it (t2610041227), so every refusal below sends
#: a user to that release rather than to a path this checkout no longer has.
MIGRATION_RELEASE = "v0.3.0"

#: The user-facing migration guide, as it exists in `MIGRATION_RELEASE`.
MIGRATION_DOC = "docs/site/guide/migrating-project-config.qmd"

#: Current guidance for configuring a project and its workflow files.
CONFIG_DOC = "docs/technical-manual/configuration.md"

#: The project config shape this loader accepts (R14 D-11.1). A document with no
#: `schema_version`, or one below 2, is a v1 set and is refused rather than
#: guessed at: the v1 and v2 spellings overlap enough that a tolerant loader
#: would compose a half-migrated document into something that runs and produces
#: different numbers.
SCHEMA_VERSION: int = 3  # Schema 2 keeps its original path bases.

#: The migration TOOL every refusal points a user at, as it exists in
#: `MIGRATION_RELEASE`. Named once so the refusals cannot disagree on it.
MIGRATION_COMMAND = "scripts/migrate_project_config.py"

#: **T1's shape, declared ONCE** — section name -> the leaf names that section
#: owns. R14 D-7.2 dissolves `shared:` into `basin:`, `climate:` and `model:` at
#: T1 top level, and D-10.3 requires `SHARED_SEAM_KEYS` to be RE-DERIVED from
#: the result rather than edited name by name. This mapping is what makes
#: "re-derived" mean something structural: both `T1_TOP_LEVEL` and
#: `SHARED_SEAM_KEYS` fall out of it, so the v2 top level is written down in
#: exactly one place and a section that gains a leaf gains seam coverage for
#: that leaf in the same edit.
#:
#: **Why the leaves and not just the section names.** `SHARED_SEAM_KEYS` is a
#: flat set of NAMES, so nesting moves seam coverage from leaf to group:
#: `historical_window` was a `shared:` leaf and sat in the set by name, but
#: `climate.window` is a leaf of a group. Derived from section names alone the
#: set would carry `climate` and not `window`, and a T2 file could declare a
#: bare `window:` uncaught. `basin` already had that property under R13, which
#: is the reason the design calls for an explicit re-derivation rather than
#: treating it as a tolerated pattern to repeat.
#:
#: **`project:` is deliberately NOT here.** It was never part of `shared:` and
#: never in the seam set; it is rejected in T2 by name, as a section, the way it
#: always was. Including its leaves would be actively wrong: `catalog` is a T1
#: leaf (`C-40`) AND a legitimate key at the top of the WF2 file (`C-39`, the
#: climate catalog, a single-reader value that belongs down-tier). Two tiers,
#: one leaf name, on purpose -- so a seam set covering `project`'s leaves would
#: refuse a shape the design ships.
T1_SHARED_SECTIONS: dict[str, tuple[str, ...]] = {
    "basin": ("region", "resolution", "output_locations", "delineation", "sources"),
    "climate": (
        "sources",
        "selected",
        "window",
        "water_year_start",
        "map_reference",
        "forcing",
        "forcings",
        "default_forcing",
    ),
    "model": ("outvars",),
    "plotting": ("annotation",),
}

#: T1's top level, closed (D-9.5). Closing it is what makes the migration
#: detector complete: a config whose only unmigrated element is a stray
#: top-level section produces no extra key under any `workflows.<name>`, so the
#: stanza check alone would see nothing and the project would run with an
#: undefined precedence between two records of the same section.
T1_TOP_LEVEL: tuple[str, ...] = (
    "schema_version",
    "project",
    *T1_SHARED_SECTIONS,
    "workflows",
)

#: The closed `workflows.<name>` stanza (D-9.1). `enabled` is T1-owned;
#: `config_path` is OPTIONAL (D-8.7) and, when present, must resolve.
#: This closure does three jobs at once: it is half the seam enforcement, it is
#: the migration detector (§15.2), and it is what stops a T1 file quietly
#: growing back into today's monolith.
WORKFLOW_STANZA_KEYS: frozenset[str] = frozenset({"enabled", "config_path"})

#: The four workflows, in the fixed pipeline order `scripts/run_workflows.py`
#: drives. Used to attribute a scanned config access to a workflow (D-9.6);
#: composition itself never restates this list — it reads what T1 declares.
WORKFLOW_NAMES: tuple[str, ...] = (
    "analyze_climate",
    "build_model",
    "analyze_projections",
    "generate_scenarios",
    "simulate_system",
)

GENERATION_KEYS = frozenset(
    {
        "n_realizations",
        "simulation_window",
        "climate_perturbations",
        "weathergen_config",
        "seed",
        "unit_id_capacity",
    }
)
SIMULATION_KEYS = frozenset(
    {
        "experiment_name",
        "scenario_collection",
        "operation",
        "compute",
        "indicators",
        "include_unperturbed",
        "initialization",
        "historical_forcing",
    }
)


def normalize_indicator_settings(settings: dict) -> dict:
    """Accept pre-rename WF4 inputs while exposing only indicator terminology."""
    settings = dict(settings)
    if "metrics" in settings:
        if "indicators" in settings:
            raise ValueError(
                "use only indicators; both old and new selection keys were supplied"
            )
        settings["indicators"] = settings.pop("metrics")
    aliases = {
        "simulate-and-metrics": "simulate-and-indicators",
        "metrics-only": "indicators-only",
    }
    if settings.get("operation") in aliases:
        settings["operation"] = aliases[settings["operation"]]
    return settings


#: Names that belong to T1 whether or not the current T1 declares them —
#: **derived**, never edited (D-10.3). Both the section names and every leaf
#: they own, because the set is flat and a leaf must be covered by its own name.
#:
#: Deriving the rejection set from the T1 document alone has a hole: a T1
#: omitting an optional key would not reject a copy of it planted in a T2 file,
#: which is the exact failure the seam rule is written against. This frozen
#: floor closes it. There is no COUPLED EDIT note any more, and its absence is
#: the point: adding a leaf to `T1_SHARED_SECTIONS` adds it here by
#: construction, so the two cannot drift.
#:
#: Two names that left rather than moved: `seed` is now WF3's alone (`C-51`,
#: a single reader, so the seam rule does not reach it) and `julia_threads`
#: left the project config entirely for `advanced_settings` (`C-54`). Both are
#: refused by name through `RETIRED_KEYS` instead, which says where they went.
SHARED_SEAM_KEYS: frozenset[str] = frozenset(T1_SHARED_SECTIONS) | frozenset(
    leaf for leaves in T1_SHARED_SECTIONS.values() for leaf in leaves
)

#: There is no hoist registry: T2 sections compose only into their workflow.
#: The D-9.6 scan permits no cross-workflow value reads. Shared values belong
#: in T1's basin, climate, or model sections. See the R14 config-shape design.
#:
#: Every v1 spelling this loader can SEE, and where it went (D-10.4 row 2).
#: Dotted path -> the sentence a user needs. `T1.` prefixes a project-file path;
#: `T2.<workflow>.` prefixes one workflow file's own top level; `T2.*.` matches
#: that name at the top of ANY workflow file.
#:
#: This diagnostic table covers only retired paths visible at parse time.
#: Schema version is checked first; these entries identify individual keys
#: left in a config that already declares version 2.
RETIRED_KEYS: dict[str, str] = {
    "T1.project.static_dir": ("deleted (`C-07`) -- nothing read it; remove the key"),
    "T1.project.data_sources": "renamed to `project.catalog` (`C-40`)",
    "T1.project.data_sources_climate": (
        "moved DOWN to `catalog:` at the top of the analyze_projections file "
        "(`C-39`) -- it has one reader, so it does not belong in the project file"
    ),
    "T1.shared": (
        "dissolved into `basin:`, `climate:` and `model:` at the project file's "
        "top level (`C-01`, `C-10`, `C-16`, `C-19`) -- sections by KIND, not a "
        "bag of everything more than one workflow reads"
    ),
    "T1.basin.gauge_points": (
        "renamed to `basin.output_locations` (`C-41`) -- these points are where "
        "output is written, which is not the same thing as a gauge"
    ),
    "T1.basin.automatic_subbasins": (
        "regrouped under `basin.delineation:` (`C-42`, `C-13`, `C-14`)"
    ),
    # The other three basin leaves `C-14` and `C-15` empty. Added 2026-08-27:
    # they were ACCEPTED SILENTLY in a v2 config until then, which is the
    # failure this table exists to prevent -- nothing reads them after the
    # regroup, so a project that migrated everything else kept a setting that
    # had quietly stopped applying. Found by the register-to-mapping
    # cross-check in `tests/test_migration_mapping.py`, not by review.
    "T1.basin.gauge_snap_tolerance_m": (
        "regrouped AND renamed to `basin.delineation.snap_tolerance_m` "
        "(`C-42`) -- the `gauge_` prefix went with `basin.gauge_points`, "
        "which is `basin.output_locations` now (`C-41`)"
    ),
    "T1.basin.river_uparea_km2": (
        "regrouped as `basin.delineation.river_uparea_km2` (`C-14`)"
    ),
    "T1.basin.spatial_sources": (
        "regrouped as `basin.sources:` (`C-15`) -- one section for every"
        " spatial input, so `hydrography` and `basin_index` join it"
    ),
    "T1.basin.hydrography": "regrouped as `basin.sources.hydrography` (`C-15`)",
    "T1.basin.basin_index": "regrouped as `basin.sources.basin_index` (`C-15`)",
    "T1.shared.historical_window": (
        "renamed AND retyped to `climate.window: {start, end}` (`C-70`) -- "
        "inclusive WATER years now, not ISO timestamps"
    ),
    "T1.shared.clim_historical": "renamed to `climate.selected` (`C-44`)",
    "T1.shared.water_year_start": ("regrouped as `climate.water_year_start` (`C-53`)"),
    "T1.shared.wflow_outvars": "regrouped as `model.outvars` (`C-19`)",
    "T1.shared.seed": (
        "moved DOWN to `seed:` at the top of the generate_scenarios file (`C-51`) "
        "-- one workflow runs the weather generator, so one workflow owns it"
    ),
    "T1.shared.julia_threads": (
        "REMOVED from the project config (`C-54`). The project-level override is "
        "gone; the toolbox-wide setting is "
        "`advanced_settings.runtime.julia_threads`"
    ),
    "T2.*.reporting": (
        "deleted (`C-77`) -- the reporting surface is removed from the config "
        "entirely, and with it the hoist mechanism that carried it"
    ),
    "T2.*.run_historical": (
        "deleted (`C-69`) -- `st_0`, the unperturbed baseline, is now ALWAYS "
        "produced. If this was `false`, the run gains two month indicators it "
        "should always have had"
    ),
    "T2.*.stress_test": (
        "renamed to `climate_perturbations:` (`C-68`) -- and inside it "
        "`step_num` becomes `n_levels` (which is step_num + 1, `C-31`) and "
        "`transient_change: true` becomes `trajectory: transient` (`C-32`)"
    ),
    "T2.*.realizations_num": "renamed to `n_realizations` (`C-29`)",
    "T2.*.horizontime_climate": (
        "folded with `run_length` into `simulation_window: {start, end}` "
        "(`C-67`) -- INCLUSIVE years, declared rather than derived from a "
        "centre and a span"
    ),
    "T2.*.run_length": (
        "folded with `horizontime_climate` into `simulation_window: "
        "{start, end}` (`C-67`) -- INCLUSIVE years. Note the old pair spanned "
        "`run_length + 1` calendar years whenever the halves snapped outward, "
        "so copy the window the run actually used, not the length."
    ),
    "T2.*.batch_size": "regrouped as `compute.batch_size` (`C-34`)",
    "T2.*.batch_size_max": ("regrouped as `compute.batch_size_max` (`C-34`)"),
    "T2.*.disk_headroom_gb": ("regrouped as `compute.disk_headroom_gb` (`C-34`)"),
    # `C-33`'s two spell factors are NOT listed. They live INSIDE
    # `stress_test:`, beside `temp:`/`precip:` rather than under them, and
    # the T2 matcher below partitions on the first dot -- so a nested path
    # here can never fire. Their parent `T2.run_stress_test.stress_test` is
    # refused wholesale, which covers them and says where the section went.
    #
    # They WERE listed, at the top level, where no v1 config has ever put
    # them. Dead entries, found when the mapping cross-check disagreed with
    # this table (R14 P4).
    "T2.build_model.model_build_config": (
        "regrouped as `engine.build_config` (`C-22`)"
    ),
    "T2.build_model.waterbodies_config": (
        "regrouped as `engine.waterbodies_config` (`C-22`)"
    ),
    "T2.build_model.observations_timeseries": (
        "replaced by `observations:`, a mapping KEYED BY VARIABLE whose keys "
        "come from `model.outvars` (`C-56`)"
    ),
    "T2.analyze_projections.clim_project": "renamed to `ensemble:` (`C-25`)",
    "T2.analyze_projections.member_selection": (
        "folded into the `members:` group as `members.selection` (`C-63`)"
    ),
    "T2.analyze_projections.member_overrides": (
        "folded into the `members:` group as `members.overrides` (`C-63`)"
    ),
    "T2.analyze_projections.historical_year_range": (
        "renamed and retyped to `reference_window: {start, end}` (`C-59`) -- "
        "CALENDAR years, deliberately, not water years (`C-74`)"
    ),
    "T2.analyze_projections.future_horizons": (
        "renamed and retyped to `future_windows:`, a LIST of "
        "`{start, end, name}` mappings (`C-60`, `C-61`)"
    ),
    "T2.analyze_projections.relative_change": (
        "dissolved (`C-66`): `min_denominator` is per-variable registry "
        "metadata now (`C-64`) and `max_flagged_months` is "
        "`advanced_settings.constraints.max_flagged_months` (`C-65`)"
    ),
    "T2.analyze_climate.candidate_sources": (
        "moved UP and widened to `climate.sources` in the project file (`C-43`) "
        "-- the full candidate set, not the extras beside a privileged one"
    ),
}

#: Declared IDENTITY comparisons (D-9.6 class 2): whole sections read to be
#: COMPARED against the owning workflow's snapshot, never consumed as settings.
#: These are not S4 value-sharing — the drift guard's entire job is to REFUSE a
#: run when another workflow's section changed under an already-built model, so
#: an edit to the WF1 T2 file is caught loudly at rule 3.01. That is the
#: opposite of the silent coupling the seam rule exists to prevent.
#:
#: (file, section) pairs, enumerated APART from the value reads so that
#: retiring the value-read registry could not silently absorb one -- which is
#: exactly what the separation bought when D-9.7 retired it.
#: **One entry since P2, and the move is the point.** The two
#: `run_stress_test.smk` entries were the Snakefile's literal
#: `config.get("workflows", {}).get("build_model")` lookups inside
#: `guarded_sections_digest` -- a hand-written restatement of the guard's list
#: that had already drifted from it (D-9.6). P2 derived them away, so the
#: minimality direction of the inventory test requires retiring them: a declared
#: entry with no live site is as much a failure as an undeclared read.
#:
#: The comparator's own access replaces them, and it is spelled `"*"` because
#: there is no workflow name left to write. `guarded_paths` walks whatever
#: `workflows` mapping the SNAPSHOT carries, so which sections get compared is a
#: property of the document rather than of this file -- which is exactly the
#: guarantee P2 was after, and it means the inventory can no longer name them.
IDENTITY_COMPARISONS: frozenset[tuple[str, str]] = frozenset(
    {
        ("blueearth_cst/experiment/check_project_consistency.py", "*"),
        # The mandatory runner selects its own closed stanza from raw T1.
        ("blueearth_cst/experiment/simulation_runner.py", "*"),
        ("blueearth_cst/experiment/simulation_runner.py", "generate_scenarios"),
        ("blueearth_cst/experiment/generation_publication.py", "generate_scenarios"),
        # Reservation admission compares frozen intent across producer sections;
        # their settings remain owned by the corresponding producer builders.
        ("blueearth_cst/experiment/reservation_request.py", "*"),
        ("blueearth_cst/experiment/reservation_request.py", "build_model"),
        ("blueearth_cst/experiment/reservation_request.py", "generate_scenarios"),
    }
)

#: Declared OWNERLESS reads (D-9.6 class 3): tools and wrappers that own no
#: workflow section and read one anyway. Sanctioned because each composes or
#: reads T1 stanza level for its own purpose rather than consuming another
#: workflow's settings. Section `"*"` means the access reaches the `workflows`
#: mapping without naming a workflow literally (the wrapper's enable-flag loop).
OWNERLESS_SECTION_READS: frozenset[tuple[str, str]] = frozenset(
    {
        ("scripts/run_workflows.py", "*"),
        ("scripts/run_workflows.py", "generate_scenarios"),
        ("scripts/generate_scenarios.py", "generate_scenarios"),
        ("blueearth_cst/shared/generation_config.py", "generate_scenarios"),
        ("blueearth_cst/shared/workflow_archive_launch.py", "*"),
        # WF2 production records and result bundles read WF2's captured section
        # to locate its product and to state or compare its scientific settings;
        # neither consumes them as another workflow's inputs.
        ("blueearth_cst/shared/product_record.py", "*"),
        ("blueearth_cst/shared/result_bundle.py", "*"),
        # The case scaffold checks enable flags and follows config_path entries
        # to copy a template set; it consumes no workflow settings.
        ("scripts/create_case.py", "*"),
        ("scripts/suggest_experiment_name.py", "simulate_system"),
        ("scripts/plot_workflow_dag.py", "simulate_system"),
        # The artifact-scoped config snapshot rebuilds `compose_config`'s shape
        # for WF4, which reads its project file directly rather than through the
        # composer. It walks the whole `workflows` mapping to drop `config_path`
        # from every stanza -- a pointer the composed document has already
        # followed -- and consumes no workflow's SETTINGS but the one it was
        # handed. `"*"` because it names no workflow literally: the caller
        # supplies the name, so which stanza is merged is a property of the call
        # rather than of this file.
        ("blueearth_cst/shared/workflow_config_snapshot.py", "*"),
        # R13's `scripts/split_project_config.py` held a fourth entry here for
        # its already-split detector. It retired with the tool (R14 Gate A), and
        # the entry had to go in the same commit: this enumeration is checked
        # for MINIMALITY as well as completeness, so a declared entry with no
        # live site is as much a failure as an undeclared read.
    }
)


# ---------------------------------------------------------------------------
# Path resolution (D-8.4)
# ---------------------------------------------------------------------------


def _resolve_config_path(raw: object, t1_dir: str, name: str) -> str:
    """Return the absolute path a stanza's ``config_path`` names.

    ``config_path`` is ``expanduser``-ed and, when still relative, resolved
    against **the directory containing T1** — not the CWD, which is how every
    other path key in the same file resolves.

    That asymmetry is deliberate (D-8.4, a reversal of the design's first
    revision). Every other path key references an EXTERNAL input — a data
    catalog, a basin CSV, an output root — whose natural anchor is the
    invocation. ``config_path`` is the only key that references *a fragment of
    the config document itself*: a file the same author wrote at the same time
    and moves as one unit with T1. Anchoring a document's own fragments at the
    document is what makes a project's config set relocatable, which a single
    file got for free. Under CWD resolution the splitter could emit no value
    that both resolves and matches the design's own examples, for the population
    the migration exists to serve — a production ``project_dir`` outside the
    repository tree.
    """
    if not isinstance(raw, (str, os.PathLike)):
        raise ValueError(
            f"workflows.{name}.config_path must be a path string, got "
            f"{type(raw).__name__}. See {CONFIG_DOC}."
        )
    text = os.path.expanduser(os.fspath(raw))
    if not text.strip():
        raise ValueError(
            f"workflows.{name}.config_path is empty. Omit the key entirely to "
            f"give the workflow no settings. See {CONFIG_DOC}."
        )
    if not os.path.isabs(text):
        text = os.path.join(t1_dir, text)
    return os.path.normpath(os.path.abspath(text))


def _declared_form(resolved_abs: str) -> str:
    """Return the form of ``resolved_abs`` a Snakemake rule input can carry.

    CWD-relative when one is computable, absolute otherwise. The values of
    ``workflow_config_paths`` become declared rule inputs (D-10.3), so they must
    be resolvable by Snakemake from the working directory.

    On Windows ``os.path.relpath`` RAISES ``ValueError`` across drives, so the
    fallback is a caught exception rather than an assumption that a relative
    form exists. A project on ``D:`` driven from a checkout on ``C:`` is an
    ordinary case here.
    """
    try:
        return os.path.relpath(resolved_abs, os.getcwd())
    except ValueError:
        return resolved_abs


# ---------------------------------------------------------------------------
# Parse-time refusals (R14 D-10.4). Every message names the FIX, not just the
# fault: a refusal that says what is wrong and not what to do about it costs a
# user the same round trip a missing check would have.
# ---------------------------------------------------------------------------


def _check_schema_version(t1: Mapping[str, Any], t1_path: str | os.PathLike) -> None:
    """Refuse a v1 project config, naming the migration command (D-10.4 row 1).

    Checked FIRST, before any other refusal, and that ordering is the whole
    design of the refusal set. A v1 document trips half a dozen other checks --
    `shared:` at top level, retired keys everywhere, no `climate.selected` --
    and reporting the first of those would tell a user to fix one key in a file
    that needs migrating wholesale. One command answers all of it.

    The project config carried no version key before R14, so absent and `1` are
    the same statement and get the same message (`C-05`).
    """
    raw = t1.get("schema_version")
    if type(raw) is int and raw in (2, SCHEMA_VERSION):
        return
    if type(raw) is int and raw > SCHEMA_VERSION:
        raise ValueError(
            f"{t1_path}: unsupported schema_version: {raw}; supported: 2, 3"
        )
    if raw is None:
        found = "no `schema_version` key at all, which means it was written before R14"
    else:
        found = f"`schema_version: {raw!r}`"
    raise ValueError(
        f"{t1_path}: this project config declares {found}, but this toolbox "
        f"reads `schema_version: {SCHEMA_VERSION}`.\n"
        f"  Migrate it with toolbox {MIGRATION_RELEASE}:  "
        f"python {MIGRATION_COMMAND} {t1_path}\n"
        f"See {MIGRATION_DOC} in {MIGRATION_RELEASE}."
    )


def _retired_hits(
    t1: Mapping[str, Any], bodies: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """Return one `path -> where it went` line per retired key present."""
    hits: list[str] = []
    for path, destination in RETIRED_KEYS.items():
        tier, _, rest = path.partition(".")
        if tier == "T1":
            node: Any = t1
            parts = rest.split(".")
            for part in parts[:-1]:
                node = node.get(part) if isinstance(node, Mapping) else None
            if isinstance(node, Mapping) and parts[-1] in node:
                hits.append(f"  {rest} -- {destination}")
        else:
            owner, _, name = rest.partition(".")
            targets = bodies if owner == "*" else {owner: bodies.get(owner) or {}}
            for workflow, body in targets.items():
                if isinstance(body, Mapping) and name in body:
                    hits.append(f"  {workflow} file: {name} -- {destination}")
    return hits


def _check_retired_keys(
    t1: Mapping[str, Any], bodies: Mapping[str, Mapping[str, Any]]
) -> None:
    """Refuse a retired key, naming its new home or that it is gone (row 2).

    Reached only by a document that already claims `schema_version: 2`, i.e. a
    HALF-migrated one, which is why naming the individual key earns its place
    here — see `RETIRED_KEYS`.

    **Called TWICE, and the first call is what makes half the table reachable.**
    A leftover `shared:` block is both a retired key and a stray top-level name,
    and `_check_t1_top_level` answers it first — with *"the top level is closed
    to [...]"*, which names none of `basin:`, `climate:` or `model:`. That is a
    strictly worse message for the most likely half-migration there is, and it
    silently killed the seven `T1.shared.*` rows: each needs `shared` to exist
    at T1 top level, which is exactly what the closure refuses. So the T1 half
    runs BEFORE the closure and the T2 half after the files are read, which is
    the earliest either can run.
    """
    hits = _retired_hits(t1, bodies)
    if not hits:
        return
    raise ValueError(
        "This config declares `schema_version: "
        f"{SCHEMA_VERSION}` but still carries retired key(s):\n"
        + "\n".join(sorted(hits))
        + f"\nRemove or move these keys as indicated. See {CONFIG_DOC}."
    )


def _check_observations(
    t1: Mapping[str, Any], bodies: Mapping[str, Mapping[str, Any]]
) -> None:
    """Refuse an ``observations:`` key that is not a declared outvar (row 5).

    D-7.3's parse-time invariant: `observations:` is a mapping KEYED BY
    VARIABLE, drawn from `model.outvars` verbatim, so observations can only be
    supplied for a variable the model was asked to output. Without the check the
    typo is silent -- the evaluation simply finds no series to compare against
    and draws an empty panel.
    """
    body = bodies.get("build_model") or {}
    observations = body.get("observations")
    if not isinstance(observations, Mapping):
        return
    outvars = (t1.get("model") or {}).get("outvars") or []
    stray = [key for key in observations if key not in outvars]
    if stray:
        raise ValueError(
            f"the build_model config declares observations for {sorted(stray)!r}, "
            "which `model.outvars` does not ask the model to output.\n"
            f"  `model.outvars` declares: {list(outvars)!r}\n"
            "  Either add the variable to `model.outvars` in the project file, "
            "or drop the observations entry -- an observed series with no "
            f"modelled counterpart has nothing to be compared against. "
            f"See {CONFIG_DOC}."
        )


#: The perturbation axes `trajectory:` is required on. Two, and not derived from
#: the config: an axis a user forgot to declare is the case this refuses.
PERTURBATION_AXES: tuple[str, ...] = ("temp", "precip")


def _check_trajectories(bodies: Mapping[str, Mapping[str, Any]]) -> None:
    """Refuse a perturbation axis with no ``trajectory:`` (row 6).

    `trajectory:` is a REQUIRED enum with no default, deliberately (`C-32`).
    The two values describe physically different experiments -- a step change
    held flat across the run versus one ramped through it -- and there is no
    answer to "which did the user mean" that is right often enough to guess.
    Defaulting either way would silently produce a response surface computed
    under an assumption nobody made.
    """
    body = bodies.get("generate_scenarios") or {}
    perturbations = body.get("climate_perturbations")
    if not isinstance(perturbations, Mapping):
        return
    missing = [
        axis
        for axis in PERTURBATION_AXES
        if isinstance(perturbations.get(axis), Mapping)
        and "trajectory" not in perturbations[axis]
    ]
    if missing:
        raise ValueError(
            "the generate_scenarios config declares "
            f"`climate_perturbations` for {missing!r} with no `trajectory:`.\n"
            "  `trajectory:` is REQUIRED on every axis and has NO default, "
            "deliberately: `constant` holds the perturbation flat across the "
            "run "
            "and `transient` ramps it through, which are different experiments, "
            "and guessing would produce a response surface computed under an "
            f"assumption nobody made. See {CONFIG_DOC}."
        )


# ---------------------------------------------------------------------------
# Seam checks (D-9.1, D-9.2, D-9.3, D-9.5)
# ---------------------------------------------------------------------------


def _check_t1_top_level(t1: Mapping[str, Any]) -> None:
    """Refuse a T1 top-level key outside ``T1_TOP_LEVEL`` (D-9.5).

    Closing the top level is what makes the migration detector complete: a
    config whose only unmigrated element is a stray top-level section produces
    no extra key under any ``workflows.<name>``, so the stanza check alone would
    see nothing and the project would run with an undefined precedence between
    two records of the same section.
    """
    stray = [key for key in t1 if key not in T1_TOP_LEVEL]
    if stray:
        raise ValueError(
            f"T1 project config declares top-level key(s) {sorted(stray)!r}. "
            f"The project file's top level is closed to {list(T1_TOP_LEVEL)!r}: "
            "workflow settings belong in that workflow's own config file. "
            f"See {CONFIG_DOC}."
        )


def _check_stanza_closed(name: str, stanza: Mapping[str, Any]) -> None:
    """Refuse a ``workflows.<name>`` key outside ``{enabled, config_path}`` (D-9.1).

    Validated for **every stanza present in T1**, whichever workflow it names —
    not only for the workflows this entry point loads. Scoping the closure to
    the loaded set would make a half-split config legal: a project that split
    `build_model` and left the other three inline would run WF1 clean.

    Absence of a whole stanza is **not this check's business**. A popped stanza
    is absent, not malformed, which is what keeps the additive-carve regression
    (`tests/test_cli.py`) green.
    """
    stray = [key for key in stanza if key not in WORKFLOW_STANZA_KEYS]
    if stray:
        raise ValueError(
            f"workflows.{name} declares key(s) {sorted(stray)!r}, but a project "
            f"config's workflow stanza is closed to {sorted(WORKFLOW_STANZA_KEYS)!r}. "
            f"Move those settings into a workflow config file named "
            f"`<project_config_stem>_{name}.yml` beside this file and point "
            f"`workflows.{name}.config_path` at it. See {CONFIG_DOC}."
        )


def _t1_declared_leaves(t1: Mapping[str, Any]) -> frozenset[str]:
    """Return every leaf name T1's shared sections actually declare.

    The DYNAMIC half of the rejection set, kept alongside the frozen
    `SHARED_SEAM_KEYS` floor for the reason R13 gave: a section may carry a leaf
    this module has not enumerated, and a copy of it planted in a T2 file is the
    same seam breach as a copy of one that is enumerated.
    """
    leaves: set[str] = set()
    for section in T1_SHARED_SECTIONS:
        body = t1.get(section)
        if isinstance(body, Mapping):
            leaves.update(body)
    return frozenset(leaves)


def _rejected_in_t2(t1: Mapping[str, Any]) -> frozenset[str]:
    """Return the names a T2 file may not declare (D-9.2).

    T1's own top-level names, every leaf T1's shared sections declare, and the
    derived `SHARED_SEAM_KEYS` floor.

    Two terms that used to sit here are gone. `shared` is now refused through
    `T1_TOP_LEVEL` and `RETIRED_KEYS`, which say where its contents went; and
    every hoisted section except this workflow's own retired with the hoist map
    (R14 D-10.1) -- it closed the unguarded direction of a mechanism that no
    longer exists.

    The result no longer varies by workflow, so the parameter went with it. It
    never did vary except through the hoist term, and a signature implying
    otherwise invites a caller to believe there is a per-workflow exemption.
    """
    return frozenset(
        set(T1_TOP_LEVEL) | {"enabled"} | _t1_declared_leaves(t1) | SHARED_SEAM_KEYS
    )


def _check_t2_names(
    name: str, body: Mapping[str, Any], path: str, t1: Mapping[str, Any]
) -> None:
    """Refuse a globally-owned name at the top level of a T2 file (D-9.2/D-9.3).

    Reads **key names only** and merges nothing, so the narrowing holds even
    when this runs over a file outside the composed set.
    """
    rejected = _rejected_in_t2(t1)
    # Local figure selection is separate from shared plotting.annotation.
    if name in {"analyze_climate", "build_model"}:
        rejected -= {"plotting"}
    stray = sorted(key for key in body if key in rejected)
    if not stray:
        return
    owners = {
        leaf: section
        for section, leaves in T1_SHARED_SECTIONS.items()
        for leaf in leaves
    }
    fixes = []
    for key in stray:
        if key in owners and key not in T1_TOP_LEVEL:
            fixes.append(f"{key!r} belongs in the project file's `{owners[key]}:`")
        else:
            fixes.append(f"{key!r} belongs at the project file's top level")
    raise ValueError(
        f"{path}: the {name!r} workflow config declares {stray!r} at its top "
        "level, but those names are owned outside a single workflow — a key "
        "read by more than one workflow lives in the project file, never in a "
        "workflow file. " + "; ".join(fixes) + f". See {CONFIG_DOC}."
    )


# ---------------------------------------------------------------------------
# Composition (D-8.5)
# ---------------------------------------------------------------------------


class _Probe(NamedTuple):
    """One declared T2 file, resolved and read once."""

    name: str
    declared: str  # the raw `config_path` value, for messages
    resolved: str  # absolute
    status: str  # "ok" | "missing" | "unparseable" | "not_mapping"
    body: dict[str, Any]
    detail: str


def _probe_t2(
    name: str,
    raw: object,
    t1_dir: str,
    source_bytes: Mapping[Path, bytes] | None = None,
) -> _Probe:
    """Resolve and read one declared T2 file without deciding whether to fail.

    Separating the read from the verdict is what lets the D-9.3 tolerance clause
    exist: the same file is a hard error inside ``R(entry)`` and a logged skip
    outside it, and both need the parse attempt.
    """
    resolved = _resolve_config_path(raw, t1_dir, name)
    if source_bytes is not None and Path(resolved).resolve() not in source_bytes:
        return _Probe(name, os.fspath(raw), resolved, "not_captured", {}, "")
    if source_bytes is None and not os.path.isfile(resolved):
        return _Probe(name, os.fspath(raw), resolved, "missing", {}, "")
    try:
        data = (
            source_bytes[Path(resolved).resolve()]
            if source_bytes is not None
            else Path(resolved).read_bytes()
        )
        if name == "simulate_system":
            from blueearth_cst.experiment.indicator_sets import load_unique_yaml

            loaded = load_unique_yaml(data)
        else:
            loaded = yaml.safe_load(data)
    except (yaml.YAMLError, ValueError) as exc:
        return _Probe(name, os.fspath(raw), resolved, "unparseable", {}, str(exc))
    if loaded is None:
        # An empty file is a workflow with no settings, same as an omitted key.
        return _Probe(name, os.fspath(raw), resolved, "ok", {}, "")
    if not isinstance(loaded, Mapping):
        return _Probe(
            name,
            os.fspath(raw),
            resolved,
            "not_mapping",
            {},
            type(loaded).__name__,
        )
    loaded = (
        normalize_indicator_settings(loaded)
        if name == "simulate_system"
        else dict(loaded)
    )
    return _Probe(name, os.fspath(raw), resolved, "ok", loaded, "")


def _raise_for_probe(probe: _Probe, t1_dir: str) -> None:
    """Apply §8.4's failure table to a file this entry point must load."""
    if probe.status == "not_captured":
        raise ValueError(
            f"workflows.{probe.name}.config_path was not in the pre-parse "
            f"capture: {probe.resolved}"
        )
    if probe.status == "missing":
        raise ValueError(
            f"workflows.{probe.name}.config_path names a file that does not "
            f"exist: {probe.resolved}\n"
            f"  declared as: {probe.declared!r}\n"
            f"  resolved against: {t1_dir}\n"
            "A relative `config_path` is anchored at the project config file's "
            "own directory, not the working directory. To give this workflow no "
            f"settings at all, delete the `config_path` key. See {CONFIG_DOC}."
        )
    if probe.status == "unparseable":
        raise ValueError(
            f"workflows.{probe.name}.config_path names a file that is not valid "
            f"YAML: {probe.resolved}\n"
            f"  resolved against: {t1_dir}\n"
            f"  parser said: {probe.detail}"
        )
    if probe.status == "not_mapping":
        raise ValueError(
            f"workflows.{probe.name}.config_path names a file that parses to "
            f"{probe.detail}, not a mapping of settings: {probe.resolved}\n"
            f"  resolved against: {t1_dir}\n"
            "A workflow config file's top level is that workflow's own settings "
            f"at column zero. See {CONFIG_DOC}."
        )


def _check_no_duplicate_paths(probes: Sequence[_Probe]) -> None:
    """Refuse one T2 file shared by two stanzas of one T1 (§8.4).

    Two workflows sharing one settings file would silently give each the other's
    keys, and the seam rule would have nothing to test against.

    Compared on ``normcase(abspath(...))`` rather than the raw string: a raw
    comparison is evaded by ``./x.yml`` versus ``x.yml``, a trailing separator,
    or a case-only difference on Windows, and no name-level check can substitute
    because with one file on disk there is only one file to compare. Same
    comparison key, and the same reason, as ``copy_config_files``' destination
    collision check. ``resolve()`` is deliberately not used, matching that
    precedent: it follows symlinks, and two deliberate symlinks to one file are
    a user's choice rather than a typo.
    """
    seen: dict[str, str] = {}
    for probe in probes:
        key = os.path.normcase(probe.resolved)
        if key in seen:
            raise ValueError(
                f"workflows.{seen[key]}.config_path and "
                f"workflows.{probe.name}.config_path resolve to the same file: "
                f"{probe.resolved}\n"
                "One workflow config file per workflow — sharing one would give "
                "each workflow the other's settings with nothing to detect it."
            )
        seen[key] = probe.name


def compose_config(
    t1: Mapping[str, Any],
    t1_path: str | os.PathLike,
    entry: str | None = None,
    declared_sections: Sequence[str] | None = None,
    source_bytes: Mapping[Path, bytes] | None = None,
    *,
    validate_forcing: bool = True,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Merge T1 and its T2 files into a config of today's shape.

    Parameters
    ----------
    t1 : Mapping
        The parsed ``--configfile`` mapping, **after** Snakemake has merged any
        ``--config key=value`` overrides into it. Overrides therefore apply to
        T1 and are validated like any other T1 content (D-8.6b).
    t1_path : str or PathLike
        ``workflow.configfiles[0]``. The resolution anchor for every relative
        ``config_path`` (D-8.4).
    entry : str, optional
        The entry point's own workflow key. ``None`` selects the tool-facing
        scope — see ``load_composed_config``.
    declared_sections : Sequence[str], optional
        This Snakefile's ``CONFIG_PROJECTION`` tuple, passed in rather than
        restated here. ``R(entry)`` is derived from it:
        ``{entry} | {s.split(".")[1] for s in declared_sections if s.startswith("workflows.")}``.
        For generation that yields ``{generate_scenarios}``
        because WF3's ``CONFIG_PROJECTION`` is itself derived from
        ``guarded_sections`` — so the loader is one more consumer of the same
        maintained literal rather than a second copy of it.
    validate_forcing : bool, optional
        False for structural runner admission before a forcing consumer is
        reached. Historical comparison and projection entry points always
        omit model-forcing adjustments and validation; with validation enabled,
        they retain an unambiguous source hint for native shared producers.

    Returns
    -------
    (composed_config, workflow_config_paths)
        ``workflow_config_paths`` maps each **loaded** workflow to its resolved
        path, in the form a Snakemake rule input can carry. It is an explicit
        part of the contract: §10.3 declares those paths as rule inputs, §10.5
        registers them in ``CONFIG_REFERENCES``, and §11 hands them to the
        snapshot writer.

    Raises
    ------
    ValueError
        On any seam violation or unresolvable ``config_path``. Every message
        names the resolved absolute path, the anchor, and the migration doc.

    Notes
    -----
    **The caller must bind the result to the Snakefile-global name** ``config``,
    or update the existing mapping in place (D-8.6a). Any other binding is a
    defect: ``check_project_consistency`` takes its live config from
    ``sm.config`` — Snakemake's ``workflow.config`` — not from the Snakefile's
    local name, so a ``composed = compose_config(...)`` refactor would leave the
    drift guard comparing ``{enabled, config_path}`` against the snapshot's full
    section and fail every WF3 run at rule 3.01, after WF1 and WF2 had already
    run, with a message blaming project drift rather than the binding.
    """
    if not isinstance(t1, Mapping):
        raise ValueError(
            f"{t1_path}: the project config must parse to a mapping, got "
            f"{type(t1).__name__}."
        )
    # FIRST, before every other refusal: a v1 document trips most of the checks
    # below, and reporting one of those would point a user at a single key in a
    # file that needs migrating whole. See `_check_schema_version`.
    _check_schema_version(t1, t1_path)
    # BEFORE the top-level closure: a leftover `shared:` is a stray top-level
    # name AND a retired key, and only the second message says where its
    # contents went. See `_check_retired_keys`.
    _check_retired_keys(t1, {})
    _check_t1_top_level(t1)
    resolve_annotation(t1)

    t1_dir = os.path.dirname(os.path.abspath(os.fspath(t1_path))) or os.getcwd()
    for section in T1_SHARED_SECTIONS:
        if section in t1 and not isinstance(t1[section], Mapping):
            raise ValueError(
                f"{t1_path}: `{section}:` must be a mapping if present, got "
                f"{type(t1[section]).__name__}."
            )

    workflows = t1.get("workflows") or {}
    if not isinstance(workflows, Mapping):
        raise ValueError(f"{t1_path}: `workflows:` must be a mapping if present.")
    if "run_stress_test" in workflows:
        raise ValueError(
            f"{t1_path}: workflows.run_stress_test is retired. Split it into "
            "workflows.generate_scenarios and workflows.simulate_system using "
            f"python {MIGRATION_COMMAND} <this file> from toolbox {MIGRATION_RELEASE}."
        )
    unknown = set(workflows) - set(WORKFLOW_NAMES)
    if unknown:
        raise ValueError(f"{t1_path}: unknown workflow stanza(s): {sorted(unknown)!r}")

    # D-9.1 over EVERY stanza present, then resolve every declared path once.
    probes: list[_Probe] = []
    for name, stanza in workflows.items():
        if not isinstance(stanza, Mapping):
            raise ValueError(
                f"{t1_path}: workflows.{name} must be a mapping carrying "
                f"{sorted(WORKFLOW_STANZA_KEYS)!r}, got {type(stanza).__name__}."
            )
        _check_stanza_closed(name, stanza)
        if "config_path" in stanza:
            probes.append(_probe_t2(name, stanza["config_path"], t1_dir, source_bytes))
    _check_no_duplicate_paths(probes)
    from blueearth_cst.shared.generation_config import normalize_generation

    probes = [
        probe._replace(body=normalize_generation(probe.body, t1["schema_version"]))
        if probe.name == "generate_scenarios" and probe.status == "ok"
        else probe
        for probe in probes
    ]
    from blueearth_cst.shared.workflow_plotting import normalize_workflow_plotting

    probes = [
        probe._replace(body=normalize_workflow_plotting(probe.name, probe.body))
        if probe.status == "ok"
        else probe
        for probe in probes
    ]
    by_name = {probe.name: probe for probe in probes}

    # R(entry). With no entry point, scope is "whatever section a tool indexes":
    # every workflow whose stanza declares a config_path that actually loaded.
    if entry is None:
        loaded_names = {probe.name for probe in probes if probe.status == "ok"}
    else:
        loaded_names = {entry} | {
            section.split(".", 1)[1]
            for section in (declared_sections or ())
            if section.startswith("workflows.")
        }

    # §8.4's failure table applies only inside R(entry). Outside it the D-9.3
    # tolerance clause holds: a workflow's own file must not be able to fail
    # another workflow's parse, which is the inverse of this milestone's goal.
    skipped: list[str] = []
    for probe in probes:
        if probe.status == "ok":
            continue
        if probe.name in loaded_names:
            _raise_for_probe(probe, t1_dir)
        elif probe.status != "not_captured":
            skipped.append(f"{probe.name} ({probe.status}: {probe.resolved})")

    # D-9.2/D-9.3: the same rejection set, over every T2 file that resolved —
    # inside R(entry) or not. Three of the four entry points have a singleton
    # R(entry), so a check confined to it could never see a shared name planted
    # in another workflow's file.
    for probe in probes:
        if probe.status == "ok":
            _check_t2_names(probe.name, probe.body, probe.resolved, t1)

    # The content refusals (D-10.4 rows 2-6), over every T2 body that resolved —
    # the same scope, and the same reason, as the name check above. Ordered
    # retired-first: a key that moved is a better message than the consequence
    # of it having moved, and a half-migrated WF3 file would otherwise report a
    # missing `trajectory:` for an axis still spelled `stress_test:`.
    bodies = {probe.name: probe.body for probe in probes if probe.status == "ok"}
    for owner, body in bodies.items():
        if owner == "build_model":
            from blueearth_cst.model.historical_runs import historical_run_name
            from blueearth_cst.shared.named_forcing_producers import (
                retained_run_names,
            )

            retained_run_names(
                historical_run_name(body.get("historical_run")),
                historical_forcing_ids(t1.get("climate") or {}, body),
            )
        if owner in {"build_model", "simulate_system"}:
            from blueearth_cst.shared.grid_output import normalize_grid_output

            normalize_grid_output(body.get("output"))
        diagnostics = body.get("diagnostics")
        if "captioned_figures" in body or (
            isinstance(diagnostics, Mapping) and "captioned_figures" in diagnostics
        ):
            raise ValueError(
                f"{owner} file: captioned_figures is retired; set "
                "plotting.annotation: caption in the project file instead "
                "(or none for unannotated figures)."
            )
    _check_retired_keys(t1, bodies)  # the T2 half; the T1 half ran above
    _check_observations(t1, bodies)
    _check_trajectories(bodies)
    for owner, forbidden, destination in (
        ("generate_scenarios", SIMULATION_KEYS, "simulate_system"),
        ("simulate_system", GENERATION_KEYS, "generate_scenarios"),
    ):
        misplaced = set(bodies.get(owner) or {}) & forbidden
        if misplaced:
            raise ValueError(
                f"{owner} file: {sorted(misplaced)!r} belong in {destination}."
            )

    composed: dict[str, Any] = {
        key: value
        for key, value in resolve_project_paths(t1, t1_path).items()
        if key != "workflows"
    }
    from blueearth_cst.shared.forcing_recipe import (
        require_workflow_compatibility,
        resolve_climate_forcing,
    )

    delta_generation = (
        entry == "generate_scenarios"
        and bodies.get("generate_scenarios", {}).get("method") == "delta_change"
    )
    required = entry in {"build_model", "generate_scenarios"} and not delta_generation
    retained_only = False
    if entry == "simulate_system":
        simulation = normalize_indicator_settings(bodies.get("simulate_system", {}))
        required = not (
            simulation.get("operation") == "indicators-only"
            or simulation.get("operations") == ["indicators"]
        )
        retained_only = not required
    comparison_only = entry in {"analyze_climate", "analyze_projections"}
    # Named recipes resolve to the legacy single-recipe form for the active one:
    # WF1 and WF3 use the default, WF4 its `historical_forcing` selection.
    declared_climate = composed.get("climate", {})
    requested = (bodies.get("simulate_system") or {}).get("historical_forcing")
    forcing_id = None
    if declares_named(declared_climate):
        composed["climate"], forcing_id = simulation_climate(
            declared_climate, requested if entry == "simulate_system" else None
        )
    elif requested is not None:
        simulation_climate(declared_climate, requested)
    if not validate_forcing or comparison_only:
        # Comparison/projection workflows do not validate model forcing. Startup
        # may also check structural configuration before a consumer is reached.
        candidates = dict(composed.get("climate", {}))
        if "sources" not in candidates and isinstance(candidates.get("selected"), str):
            candidates["sources"] = [candidates["selected"]]
        candidates.pop("selected", None)
        candidates.pop("forcing", None)
        climate, recipe = resolve_climate_forcing(candidates, required=False)
        if comparison_only and validate_forcing:
            # A source hint keeps native shared producers aligned without
            # validating or applying a draft model-forcing recipe.
            declared = composed.get("climate", {})
            forcing = declared.get("forcing", {})
            public = forcing if isinstance(forcing, Mapping) else {}
            selected = public.get("precip_source", declared.get("selected"))
            if "precip_source" in public and "selected" in declared:
                selected = None
            elif "precip_source" not in public and declared.get("selected") is None:
                sources = climate.get("sources", [])
                selected = sources[0] if len(sources) == 1 else None
            if isinstance(selected, str) and selected in climate.get("sources", []):
                climate["selected"] = selected
    elif retained_only or delta_generation:
        climate, recipe = composed.get("climate", {}), None
    else:
        climate, recipe = resolve_climate_forcing(
            composed.get("climate", {}), required=required
        )
    if recipe is not None and entry is not None:
        require_workflow_compatibility(recipe, entry)
    if forcing_id is not None:
        climate = dict(climate)
        climate.update(
            _resolve_named_map(
                declared_climate,
                bodies.get("build_model") if entry == "build_model" else None,
                validate=validate_forcing and not comparison_only,
            )
        )
        climate["forcing_id"] = forcing_id
    if "climate" in composed or required:
        composed["climate"] = climate
    if entry == "analyze_climate" and not climate.get("sources"):
        raise ValueError("climate.sources must contain comparison candidates")
    merged: dict[str, Any] = {}
    workflow_config_paths: dict[str, str] = {}
    for name, stanza in workflows.items():
        section = {key: value for key, value in stanza.items() if key != "config_path"}
        if name in loaded_names:
            probe = by_name.get(name)
            body = dict(probe.body) if probe is not None else {}
            if probe is not None:
                body = resolve_workflow_paths(
                    body, probe.resolved, name, t1["schema_version"]
                )
            section.update(body)
            if probe is not None:
                workflow_config_paths[name] = _declared_form(probe.resolved)
        merged[name] = section
    composed["workflows"] = merged

    if skipped:
        print(
            "[config_composition] skipped unreadable workflow config(s) outside "
            f"this entry point's scope: {', '.join(skipped)}"
        )
    # Configuration is read repeatedly during admission and child startup.
    # The runner presents the forcing summary once, after preparation completes.
    return composed, workflow_config_paths


def _resolve_named_map(
    declared: Mapping[str, Any],
    build_model: Mapping[str, Any] | None,
    *,
    validate: bool,
) -> dict[str, Any]:
    """Normalize every named recipe into the internal form consumers read.

    Each recipe passes the ordinary single-recipe resolver, so source and recipe
    admissibility stay in one place. Two names for one recipe are refused: a
    comparison between them would attribute identical forcing to two labels.
    WF1's requested recipes must also be admissible for building the model.
    """
    from blueearth_cst.shared.forcing_recipe import (
        require_workflow_compatibility,
        resolve_climate_forcing,
    )

    recipes, default = named_recipes(declared)
    resolved: dict[str, dict[str, Any]] = {}
    seen: dict[Any, str] = {}
    requested = (
        historical_forcing_ids(declared, build_model) if build_model is not None else ()
    )
    for name in recipes:
        view = legacy_view(declared, name)
        if not validate:
            resolved[name] = {"forcing": dict(recipes[name])}
            continue
        internal, recipe = resolve_climate_forcing(view, required=True, notices=False)
        if recipe in seen:
            raise ValueError(
                f"climate.forcings.{name} repeats the recipe of {seen[recipe]!r}; "
                "each name must describe a distinct forcing"
            )
        seen[recipe] = name
        if name in requested:
            require_workflow_compatibility(recipe, "build_model")
        resolved[name] = {
            "selected": internal["selected"],
            **({"forcing": internal["forcing"]} if "forcing" in internal else {}),
        }
    return {"forcings": resolved, "default_forcing": default}


def load_composed_config(
    t1_path: str | os.PathLike,
    entry: str | None = None,
    declared_sections: Sequence[str] | None = None,
    *,
    validate_forcing: bool = True,
) -> dict[str, Any]:
    """Parse T1 from disk and return the composed config. Tools pass ``entry=None``.

    The tool-facing entry point (D-12.0). Because ``compose_config`` is a pure
    function of a parsed mapping plus a path, every raw-T1 consumer — the
    ``tree-check`` snapshot tool, the cache pruners, the DAG renderer, the
    conftest fixture — becomes a two-line change: import this and call it
    instead of ``yaml.safe_load``.

    With ``entry=None`` the loaded set is every workflow whose stanza declares a
    ``config_path`` that resolves and parses, which is the right scope for a
    caller that is not an entry point; the D-9.3 tolerance clause then means a
    broken WF3 file does not break a ``tree-check``.
    """
    text = Path(os.fspath(t1_path)).read_text(encoding="utf-8")
    t1 = yaml.safe_load(text) or {}
    composed, _ = compose_config(
        t1, t1_path, entry, declared_sections, validate_forcing=validate_forcing
    )
    return composed


def compose_captured_config(
    project_source: CapturedSource,
    sources: Sequence[CapturedSource],
    entry: str,
    declared_sections: Sequence[str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Parse the exact launcher-captured buffers, never the mutable originals.

    The project source path remains the resolution anchor for config_path.
    A launcher may add overrides to the parsed project mapping before calling
    ``compose_config``; this direct form covers the no-override case.
    """
    captured = {source.original_path: source.data for source in sources}
    if project_source.original_path not in captured:
        raise ValueError("project source is absent from capture")
    project = yaml.safe_load(project_source.data) or {}
    if overrides:
        project.update(overrides)
    return compose_config(
        project,
        project_source.original_path,
        entry,
        declared_sections,
        source_bytes=captured,
    )


def capture_configuration_sources(
    project_path: str | os.PathLike,
    entry: str,
    declared_sections: Sequence[str] = (),
    custom_dependencies: Sequence[tuple[str, str, str | os.PathLike]] = (),
) -> tuple[CapturedSource, ...]:
    """Capture precisely the project, in-scope workflow and custom files.

    The project buffer is parsed once to discover config_path identities. Later
    composition consumes these captured buffers, even if live paths mutate.
    The caller supplies custom dependency roles and already resolved paths;
    no catalog name is guessed to be a file.
    """
    from blueearth_cst.shared.workflow_config_snapshot import capture_sources

    project_source = capture_sources([("project", "project_config", project_path)])[0]
    project = yaml.safe_load(project_source.data) or {}
    workflows = project.get("workflows") or {}
    if not isinstance(workflows, Mapping):
        raise ValueError("project workflows must be a mapping")
    loaded_names = {entry} | {
        section.split(".", 1)[1]
        for section in declared_sections
        if section.startswith("workflows.")
    }
    source_specs = []
    for name in sorted(loaded_names):
        stanza = workflows.get(name) or {}
        if not isinstance(stanza, Mapping):
            raise ValueError(f"workflows.{name} must be a mapping")
        if "config_path" in stanza:
            resolved = _resolve_config_path(
                stanza["config_path"], str(project_source.original_path.parent), name
            )
            source_specs.append(
                (f"workflow_{name}", f"workflow_config_{name}", resolved)
            )
    source_specs.extend(custom_dependencies)
    others = capture_sources(source_specs) if source_specs else ()
    return (project_source, *others)


# ---------------------------------------------------------------------------
# The cross-workflow read scan (D-9.6)
# ---------------------------------------------------------------------------

#: Which workflow each entry point owns.
_SNAKEFILE_OWNER: dict[str, str] = {f"{name}.smk": name for name in WORKFLOW_NAMES}

#: Which workflow each package stage owns. `shared/`, `spatial/` and
#: `weathergen/` own no section: they are called from more than one workflow, so
#: any `workflows.<name>` access there is ownerless by construction.
_STAGE_OWNER: dict[str, str] = {
    "climate_analysis": "analyze_climate",
    "model": "build_model",
    "projections": "analyze_projections",
    "experiment": "simulate_system",
}

_NAMES_ALT = "|".join(WORKFLOW_NAMES)

#: An access that REACHES the `workflows` mapping: `["workflows"]` indexing or a
#: `.get("workflows"` chain.
_WORKFLOWS_ACCESS = re.compile(
    r"""(?: \[ \s* ["']workflows["'] \s* \] | \.get\( \s* ["']workflows["'] )""",
    re.VERBOSE,
)

#: The workflow name that follows such an access, allowing for the `or {}` and
#: default-argument noise real chains carry between the two steps.
_SECTION_AFTER = re.compile(
    rf"""^ [\s\S]{{0,60}}? (?: \[ \s* ["'](?P<a>{_NAMES_ALT})["'] \s* \]
                            | \.get\( \s* ["'](?P<b>{_NAMES_ALT})["'] )""",
    re.VERBOSE,
)

#: A settings key read straight off the section: `section["key"]` / `.get("key"`.
_KEY_AFTER = re.compile(
    r"""^ \s* (?: \[ \s* ["'](?P<a>\w+)["'] \s* \]
                                    | \.get\( \s* ["'](?P<b>\w+)["'] )""",
    re.VERBOSE,
)

#: The house accessor. When the chain is its first argument the key is its
#: second, past the intervening `or {}` / default-argument noise. The trailing
#: `\w*` absorbs the mapping the chain starts from (`get_config(config.get(...`).
_GET_CONFIG_BEFORE = re.compile(r"get_config\(\s*\w*$")
#: Applied ONLY when the chain is `get_config`'s first argument, where the next
#: quoted word is the key by that function's signature. Unanchored because the
#: chain's own `.get(section, {})` default and an `or {}` fallback sit between.
_GET_CONFIG_KEY = re.compile(r"""["'](?P<k>\w+)["']""")
_GET_CONFIG_WINDOW = 120

#: The projection / guard spelling: a string whose ENTIRE content is
#: `workflows.<name>`. Anchoring on both quotes is what separates a section
#: projection from a message: `"workflows.build_model.simulation_window must be
#: a mapping"` and `f"workflows.run_stress_test.{key} must be >= 1"` carry a
#: further dotted segment, so neither matches, while `guarded_sections`'
#: `"workflows.build_model"` does.
_PROJECTION_LITERAL = re.compile(rf"""["']workflows\.(?P<n>{_NAMES_ALT})["']""")


class ScanHit(NamedTuple):
    """One config-section access found on a scanned surface.

    ``reader`` is the workflow the file belongs to, or ``None`` when the file
    owns no section. ``key`` is the settings key read straight off the section,
    or ``None`` when the whole section is taken.
    """

    file: str
    line: int
    reader: str | None
    section: str
    key: str | None


#: Token types that can precede a docstring — i.e. that mean the STRING token
#: starting here opens a statement rather than continuing an expression.
_PROSE_OPENERS = frozenset(
    {tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING}
)


def _blank(lines: list[str], start: tuple[int, int], end: tuple[int, int]) -> None:
    """Overwrite a token's span with spaces, preserving every offset."""
    (row_start, col_start), (row_end, col_end) = start, end
    for row in range(row_start, row_end + 1):
        line = lines[row - 1]
        first = col_start if row == row_start else 0
        last = col_end if row == row_end else len(line.rstrip("\n"))
        lines[row - 1] = line[:first] + " " * (last - first) + line[last:]


def _strip_prose(source: str) -> str:
    """Return ``source`` with comments and docstrings blanked, offsets preserved.

    Tokenizing rather than regexing is what keeps a commented-out example from
    reading as a live access — and docstrings need the identical treatment for
    the identical reason. ``provenance.project_config``'s docstring quotes the
    projection tuple verbatim (``"workflows.build_model"``) to explain what a
    projection is; that is prose about a read, not a read, and a scan that could
    not tell them apart would force a prose sentence into a contract
    enumeration.

    A docstring is identified structurally: a STRING token that opens a
    statement and is immediately followed by a line end, i.e. a bare string
    expression. Such a token is never a read.

    Line and column offsets are preserved so a hit still reports the line a
    human would open. A file that will not tokenize — which no surface here is —
    is returned unchanged rather than dropped, so the scan can never go quiet by
    failing to read something.
    """
    try:
        tokens = list(tokenize.generate_tokens(StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return source
    lines = source.splitlines(keepends=True)
    for token in tokens:
        if token.type == tokenize.COMMENT:
            _blank(lines, token.start, token.end)
    code = [token for token in tokens if token.type != tokenize.COMMENT]
    for index, token in enumerate(code):
        if token.type != tokenize.STRING:
            continue
        previous = code[index - 1].type if index else tokenize.ENCODING
        following = code[index + 1].type if index + 1 < len(code) else tokenize.NEWLINE
        if previous in _PROSE_OPENERS and following in (
            tokenize.NEWLINE,
            tokenize.NL,
            tokenize.ENDMARKER,
        ):
            _blank(lines, token.start, token.end)
    return "".join(lines)


def _owner_of(rel: str) -> str | None:
    """Return the workflow a scanned file owns, or ``None``."""
    if rel in {
        "blueearth_cst/experiment/generation_plan.py",
        "blueearth_cst/experiment/delta_generation.py",
        "blueearth_cst/experiment/prepare_cst_parameters.py",
    }:
        return "generate_scenarios"
    parts = Path(rel).parts
    if len(parts) == 1:
        return _SNAKEFILE_OWNER.get(parts[0])
    if parts[0] == "blueearth_cst" and len(parts) > 2:
        return _STAGE_OWNER.get(parts[1])
    return None


def _scan_text(rel: str, source: str) -> list[ScanHit]:
    """Return every config-section access in one file."""
    owner = _owner_of(rel)
    text = _strip_prose(source)
    hits: list[ScanHit] = []

    def line_of(offset: int) -> int:
        return text.count("\n", 0, offset) + 1

    for match in _WORKFLOWS_ACCESS.finditer(text):
        tail = text[match.end() :]
        section_match = _SECTION_AFTER.match(tail)
        if section_match is None:
            # The access reaches `workflows` without naming a workflow: a
            # stanza-level read, which is what the wrapper's enable-flag loop
            # does. Recorded rather than dropped, so it must still be declared.
            hits.append(ScanHit(rel, line_of(match.start()), owner, "*", None))
            continue
        section = section_match.group("a") or section_match.group("b")
        rest = tail[section_match.end() :]
        key_match = _KEY_AFTER.match(rest)
        if key_match is not None:
            key = key_match.group("a") or key_match.group("b")
        elif _GET_CONFIG_BEFORE.search(
            text[max(0, match.start() - 40) : match.start()]
        ):
            key_match = _GET_CONFIG_KEY.search(rest[:_GET_CONFIG_WINDOW])
            key = key_match.group("k") if key_match else None
        else:
            key = None
        hits.append(ScanHit(rel, line_of(match.start()), owner, section, key))

    for match in _PROJECTION_LITERAL.finditer(text):
        hits.append(ScanHit(rel, line_of(match.start()), owner, match.group("n"), None))
    return hits


def scan_surfaces(repo_root: str | os.PathLike) -> tuple[str, ...]:
    """Return every runtime-consumer surface the D-9.6 scan reads, repo-relative.

    The four ``.smk`` entry points, ``blueearth_cst/`` and ``scripts/`` — the
    same surface the design's inventory swept, because a permanent scan narrower
    than the inventory cannot certify what the inventory claimed.

    ``dev/scripts/`` stays outside the scan on the repository's own
    invocation-model rule: it is never part of a run, and a ``dev/scripts/``
    tool that acquires a run-path caller must move into the shipped package —
    where the scan sees it.

    **This module excludes itself**, and that is the one exclusion. It is the
    loader: it necessarily carries the access forms as patterns and the section
    names as contract data, so scanning it would report the scanner's own
    grammar as findings. Nothing here reads a config at all — every function
    takes the mapping from its caller — so the exclusion gives up no coverage a
    reviewer of this file would not have to provide anyway.
    """
    root = Path(repo_root)
    found = [f"{name}.smk" for name in WORKFLOW_NAMES]
    for package in ("blueearth_cst", "scripts"):
        for path in sorted((root / package).rglob("*.py")):
            if "__pycache__" in path.parts or path.name == Path(__file__).name:
                continue
            found.append(path.relative_to(root).as_posix())
    return tuple(found)


def scan_cross_workflow_reads(repo_root: str | os.PathLike) -> tuple[ScanHit, ...]:
    """Return every CROSS-workflow config access on the scanned surfaces (D-9.6).

    A hit is cross-workflow when the file does not own the section it reads. The
    caller — ``tests/test_config_composition.py`` — asserts that the value
    reads are **empty**, and that the other two hit sets equal
    ``IDENTITY_COMPARISONS`` and ``OWNERLESS_SECTION_READS``. The three were
    enumerated separately so that retiring the value-read registry (D-9.7)
    could not silently absorb an entry into the others.

    Both directions are checked there: **completeness**, so an undeclared read
    turns the test red with a message saying *promote the key to* ``shared:``,
    *do not add an entry*; and **minimality**, so a declared entry with no live
    site is equally a failure and no enumeration can drift stale.

    The completeness claim is relative to the access forms above — an access the
    patterns cannot see (a dynamically constructed key) is out of reach, which
    is one reason the parse-time checks stay rather than being subsumed by this.
    """
    root = Path(repo_root)
    hits: set[ScanHit] = set()
    for rel in scan_surfaces(root):
        path = root / rel
        if not path.is_file():
            continue
        for hit in _scan_text(rel, path.read_text(encoding="utf-8")):
            if hit.reader is not None and hit.section == hit.reader:
                continue  # a workflow reading its own section
            hits.add(hit)
    return tuple(sorted(hits))


def partition_hits(
    hits: Iterable[ScanHit],
) -> tuple[
    frozenset[tuple[str, str, str]],
    frozenset[tuple[str, str]],
    frozenset[tuple[str, str]],
]:
    """Split scan hits into the three declared enumerations' comparable shapes.

    Returns ``(value_reads, identity_comparisons, ownerless_reads)`` — the first
    as ``(reader, owner, key)`` triples, which must be EMPTY, the other two as
    ``(file, section)`` pairs.
    """
    value_reads = {
        (hit.reader, hit.section, hit.key)
        for hit in hits
        if hit.reader is not None and hit.key is not None
    }
    identity = {
        (hit.file, hit.section)
        for hit in hits
        if hit.reader is not None and hit.key is None
    }
    ownerless = {(hit.file, hit.section) for hit in hits if hit.reader is None}
    return frozenset(value_reads), frozenset(identity), frozenset(ownerless)
