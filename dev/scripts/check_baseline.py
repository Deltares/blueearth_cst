"""Record / check fingerprints for the M1 replication baseline.

Walks the formal `rule all` targets across the three Snakefiles and writes
a JSON manifest under dev/baseline/. Fingerprint format follows the
roadmap and the pipeline-regression-testing skill: per-variable summary
stats for netCDF, normalized SHA256 for CSV/YAML, size-only for PNG.

One target is special: the workflow-1 Wflow **discharge** series
(`models/hydrology/wflow/run_default/output.csv`). It is NOT a `rule all` target of
Snakefile_model_creation (whose `rule all` lists only the 3 PNGs + config
snapshot + outlet_index.csv); it is fingerprinted beyond `rule all` for
constant-parameter-preservation coverage (ADR 0001, t260719a). A byte-hash is
wrong for it: raw daily discharge is full float64 and maximally LSB-sensitive,
so an exact hash fails on any solver/env drift and a match cannot be attributed
to a config change. Instead `record` stores a reduced reference series (time
index + Q column) under dev/baseline/discharge_ref/, and `check` runs a
time-index-aligned numeric comparator (ADR step 6): structural checks, then a
per-timestep absolute+relative tolerance. The SAME comparator (`compare_discharge`)
is exposed via the `compare` subcommand for the one-off restored-vs-reference and
reproducibility comparisons (ADR steps 4b/5), so reproducibility, materiality, and
the durable regression check cannot disagree.

The wf3 **indicator tables** are the second such target (type "indicator", R11
Q8). Same reason, arrived at from the other direction: they used to be
byte-hashable only because they were rounded, and P1 dropped the rounding as part
of moving them to a long float32 shape. `record` stores a reference copy of the
table under dev/baseline/indicator_ref/ and `check` aligns the two on their row
keys and applies a per-group absolute+relative tolerance. Grouped, not global,
because one long table stacks metrics in different units and gauges of different
catchment areas -- a single file-wide mean would let the biggest series set the
threshold for the smallest.

**Mixed-provenance baseline (ADR 0001 t260719a, immaterial branch).** Since the
constant-parameter restoration the wf1 slice reflects the RESTORED model, while the
wf2/wf3 rows are the pre-restoration recording (the restored discharge move was
immaterial — 0/7670 timesteps over tolerance — so wf3 was deliberately not re-run).
**RESIDUAL TESTED AND CLOSED 2026-08-05.** This used to warn that a future wf3
regen + `check` MAY fail the byte-exact q_indicators/basin_indicators fingerprints
if the sub-tolerance wf1 move (max|dQ|/mean ~ 1.7e-4) survived their rounding, and
to follow the ADR 0001 step-7 immaterial-branch path if it did. It has now been
run. A wf3 regen from the RESTORED model reproduced both tables byte-identically
below the header: the only fingerprint movement was the R9-followup rename of the
perturbation-axis columns (tavg/prcp -> temp_change/precip_change), proven by
reverting the header line alone and recovering both recorded sha256 exactly, with
both sizes moving by exactly +16 bytes -- the header delta and nothing else. So
the wf1 delta demonstrably does NOT propagate into the wf3 reduction, and the wf3
slice is no longer mixed-provenance in any way that matters: it was re-recorded
from main@03e546c on those same numbers.
Evidence: dev/milestones/r09/migration_indicator-axis-columns.md §5;
dev/decisions/0001-restore-wflow-constant-parameters/baseline_diffs.md.
(That paragraph describes the world before R11: both tables were byte-hashed and
`basin_indicators.csv` still existed. Kept as written because it is the record of
how the residual was closed, not a description of the current gate.)

Usage:
    python dev/scripts/check_baseline.py record
    python dev/scripts/check_baseline.py record --workflow model_creation   # merge one slice
    python dev/scripts/check_baseline.py check
    python dev/scripts/check_baseline.py check --workflow model_creation
    python dev/scripts/check_baseline.py compare --ref A/output.csv --cur B/output.csv
    python dev/scripts/check_baseline.py {record,check} --project-dir test_case/test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from math import floor, log10
from pathlib import Path

# Force netCDF4 to load before xarray's lazy backend triggers it. On
# Windows under pixi, xarray's deferred `import netCDF4` inside its
# backend fails with a DLL load error, but a direct top-level import
# succeeds and primes the loader so subsequent imports work.
import netCDF4  # noqa: F401
import numpy as np
import pandas as pd
import xarray as xr
import yaml

PROJECT_DIR_DEFAULT = "test_case/test_local"
CLIM_PROJECT = "cmip6"
EXPERIMENT_NAME = "experiment"

MANIFEST_PATH_DEFAULT = Path("dev/baseline/manifest.json")
# v2 adds the workflow-1 discharge target (type "discharge"): a stored reference
# series under dev/baseline/discharge_ref/ compared with a tolerance comparator
# rather than a byte hash (ADR 0001 step 6).
MANIFEST_VERSION = 3
SIG_FIGS = 10

REPO_ROOT = Path(__file__).resolve().parents[2]


def git_provenance(repo_root: Path = REPO_ROOT) -> dict | None:
    """Which branch/commit is writing this manifest, and is the tree dirty?

    R7-21. The baseline fixture (`project_dir`) is **untracked**, so it belongs
    to no branch: every branch, worktree and session that runs a workflow writes
    into the same tree. `check` therefore answers "does the tree match the
    manifest" for whichever branch ran LAST, not for the branch you are on -- a
    green check can mean someone else's code is consistent with your manifest.

    Observed, not hypothetical: a `basin_area.png` produced on
    `feat/outputs-figures` sat in the fixture for days and was read as the
    pre-R07 baseline reference, until a byte-size mismatch at the R07 gate
    forced the question (see dev/followups-archive.md R7-3 / R7-21).

    Best-effort by design: a missing `git`, a non-repository checkout or a
    detached HEAD returns None rather than raising. Provenance is an aid to
    attribution, and must never be the reason a baseline command fails.
    """

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(repo_root), *args],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return None
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "branch": branch or "(unknown)",
        "commit": commit,
        # A dirty tree means the recorded artifacts were produced by code that
        # matches no commit -- worth knowing when a later check disagrees.
        "dirty": bool(status),
    }


def format_provenance(prov: dict | None) -> str:
    if not prov:
        return "(unrecorded)"
    return (
        f"{prov.get('branch')}@{str(prov.get('commit'))[:12]}"
        f"{' +dirty' if prov.get('dirty') else ''}"
    )


PNG_TOLERANCE_FRAC = 0.10

# Discharge comparator tolerances (ADR 0001 step 6). ATOL is set per-comparison
# to DISCHARGE_ATOL_FRAC * mean(Q_reference); RTOL is the low-flow tightener.
DISCHARGE_ATOL_FRAC = 1e-3
DISCHARGE_RTOL = 0.01
# Subdir (relative to the manifest dir) holding reduced reference discharge series.
DISCHARGE_REF_SUBDIR = "discharge_ref"

# Indicator-table comparator tolerances (R11 Q8, ruled 2026-08-05). Deliberately
# the SAME numbers as the discharge anchor, under their own names so the two can
# diverge later without one silently dragging the other: 1e-3 of a group's own
# mean magnitude, tightened to 1% relative wherever the reference value is large
# enough for a relative test to mean anything.
#
# Why a tolerance at all: the tables dropped `.round(2)` / `.round(4)` in P1, and
# that rounding had been an ACCIDENTAL drift buffer. Without it a byte-exact
# sha256 fails on any harmless numeric nudge -- a solver rebuild, a BLAS version,
# a float32 reduction reordering -- without indicating a defect. That is the same
# argument that excludes FIGURE_KINDS by default: a gate that cannot distinguish
# noise from a defect stops being read.
#
# These starting values are the repo's established materiality threshold rather
# than a measured property of indicator tables. They are honest as a default and
# unproven as a tuning: the two GEV fits are optimizer output and are the rows
# most likely to move for reasons no physics change explains. If a re-record ever
# shows those rows failing alone, tighten or loosen THEM, not the whole table.
INDICATOR_ATOL_FRAC = 1e-3
INDICATOR_RTOL = 0.01
# Subdir (relative to the manifest dir) holding reference indicator tables.
INDICATOR_REF_SUBDIR = "indicator_ref"
# The value column; every other column is part of the row key. Kept as a literal
# rather than imported from blueearth_cst.shared.indicator_tables on purpose --
# dev/scripts/ is importable from a bare checkout with no package install, and a
# baseline gate that cannot run because the package moved is worse than a
# duplicated string. tests/test_check_baseline_indicator.py pins the two together.
INDICATOR_VALUE_COLUMN = "value"
# Rows are grouped by these columns before a tolerance is derived, so each group's
# ATOL comes from its OWN magnitude. Grouping matters: `metric` alone would let a
# large-catchment gauge set the threshold for a small one, and no grouping at all
# would let a return level in m3/s set it for a volume in mm.
INDICATOR_GROUP_COLUMNS = ("metric", "location")

# Volatile attrs stripped before fingerprinting netCDF files.
VOLATILE_NC_ATTRS = frozenset(
    {
        "history",
        "creation_date",
        "Conventions",
        "software",
        "software_version",
        "production_date",
        "creation_time",
        "date_created",
        "date_modified",
    }
)

# (workflow, kind, path-template). Templates are resolved against project_dir.
# The workflow tag scopes `check --workflow <name>` / `record --workflow <name>`
# (repeatable); it selects a path universe applied symmetrically to the recorded
# and current sides. Mirrors `rule all` across Snakefile_model_creation,
# Snakefile_climate_projections, Snakefile_climate_experiment — plus the one
# beyond-`rule all` discharge target (see module docstring / ADR 0001).
# R07 (dev/milestones/r07/migration_project-layout.md §3a is the authority; this list is
# written FROM that table). 14 live targets: all 14 change manifest key via the
# examples/ -> test_case/ rename, 10 also move within the tree, 3 change
# content. Retargeted here, in the fixture-rename commit, as the SOLE owner of
# this edit -- so "the baseline blackout starts at commit 4" is literally true
# rather than reworded. `check_baseline check` is therefore RED by construction
# from this commit until the commit-14 re-record: the templates below describe
# the post-R07 tree, which commits 7-12 have not yet produced. That is expected,
# not a regression signal. The substitute gates for the window are the per-slice
# semantic_tree_diff runs against the retained pre-R07 reference tree and the
# comparator-based discharge anchor (migration map §7a).
TARGETS: list[tuple[str, str, str]] = [
    # Snakefile_model_creation -- B10 (commit 12) splits the project-level
    # plots/ tree by DEPICTED subject: model inputs, the model, the run.
    (
        "model_creation",
        "png",
        "{project_dir}/models/hydrology/wflow/evaluation/plots/hydro_wflow_1.png",
    ),
    (
        "model_creation",
        "png",
        "{project_dir}/data/spatial/plots/basin_area.png",
    ),
    (
        "model_creation",
        "png",
        "{project_dir}/models/hydrology/wflow/forcing/plots/forcing_precip_map.png",
    ),
    (
        "model_creation",
        "yaml",
        "{project_dir}/config/runs/snake_config_model_creation.yml",
    ),
    # Unmoved within the tree (prefix change only) -- and exception 3(d)
    # requires it to stay that way: if discharge moves at all, stop.
    (
        "model_creation",
        "discharge",
        "{project_dir}/models/hydrology/wflow/run_default/output.csv",
    ),
    # Snakefile_climate_projections -- B3 (commit 9) tiers ONLY the three
    # summary files; the three PNGs deliberately stay put (arch-10).
    # S8-05: a SWAP, not a subtraction. The three wide
    # `annual_change_scalar_stats_summary*` files were retired, and dropping them
    # without replacement would have taken coverage from 15 targets to 12 and left
    # the change factors unfingerprinted. The two tidy tables take their place and
    # carry strictly more -- both values per row, per-row provenance, and the
    # future level the wide form never held.
    (
        "climate_projections",
        "csv",
        "{clim_project_dir}/summary/{clim_project}_change_factors_annual.csv",
    ),
    (
        "climate_projections",
        "csv",
        "{clim_project_dir}/summary/{clim_project}_change_factors_monthly.csv",
    ),
    # S8-07 renamed all three figures.
    (
        "climate_projections",
        "png",
        "{clim_project_dir}/plots/{clim_project}_change_factor_cloud.png",
    ),
    (
        "climate_projections",
        "png",
        "{clim_project_dir}/plots/{clim_project}_precip_annual_absolute.png",
    ),
    (
        "climate_projections",
        "png",
        "{clim_project_dir}/plots/{clim_project}_temp_annual_absolute.png",
    ),
    (
        "climate_projections",
        "yaml",
        "{project_dir}/config/runs/snake_config_climate_projections.yml",
    ),
    # Snakefile_climate_experiment. R9 P3 renames the two tables and moves them
    # from indicators/ to results/. The wf3 config snapshot does NOT join
    # config/runs/: it stays inside the experiment (arch-10), content only.
    # R11 CR-2: ONE table per output variable, so this set follows the SEED
    # config's `workflows.model_creation.wflow_outvars`, which is
    # `["river discharge"]` -> `q_indicators.csv` alone. `basin_indicators.csv`
    # no longer exists; its contents are now per-variable tables, and the seed
    # requests no basin variables.
    #
    # Pinned here rather than read from a config, deliberately: this file already
    # pins EXPERIMENT_NAME and CLIM_PROJECT the same way, because it describes the
    # SEED tree and not an arbitrary project. **Adding a variable to the seed
    # config means adding its row here** -- the two are coupled and nothing
    # enforces it, which is the cost of that choice.
    # R11 Q8: `indicator`, not `csv`. P1 dropped the `.round(2)`/`.round(4)` that
    # had been an accidental drift buffer, so a byte-exact sha256 here now fails
    # on numeric noise that indicates no defect. Compared against a stored
    # reference table with a per-group tolerance instead -- see the indicator
    # block and INDICATOR_ATOL_FRAC.
    ("climate_experiment", "indicator", "{exp_dir}/results/q_indicators.csv"),
    (
        "climate_experiment",
        "yaml",
        "{exp_dir}/config/snake_config_climate_experiment.yml",
    ),
]

WORKFLOWS = ("model_creation", "climate_projections", "climate_experiment")

#: Fingerprint kinds that are FIGURES rather than data. Excluded by default.
#:
#: A figure is a terminal artifact: no rule consumes it, so a change to one
#: cannot propagate into a number anywhere downstream. And `fingerprint_png`
#: compares only `size_bytes`, so ANY cosmetic edit -- a font size, a legend
#: corner, a line weight -- turns the gate red while telling you nothing about
#: whether the model still produces the same results. That cost lands on every
#: figure refactor and buys no signal, which is why the default flipped
#: (2026-08-03).
#:
#: What still covers figures: `rule all` fails if one is not produced, and
#: `tests/test_wf1_plot_outputs.py` pins the declared output set. What is NOT
#: covered is their content -- inspect a changed figure by looking at it.
#:
#: Pass `--include-figures` to either subcommand when the figures themselves are
#: what is under review. Record and check must AGREE: a manifest recorded
#: without figures and checked with them reports them as missing.
FIGURE_KINDS = frozenset({"png"})


def active_targets(
    workflows: set[str] | None = None, include_figures: bool = False
) -> list[tuple[str, str, str]]:
    """TARGETS after the workflow and figure filters.

    The SINGLE place both filters are applied. `compute_manifest`,
    `record_discharge` and the two manifest-prune loops all read from here, so a
    row can never be fingerprinted by one and pruned by another -- which is how
    a merge would silently drop rows it never re-recorded.
    """
    return [
        (workflow, kind, template)
        for workflow, kind, template in TARGETS
        if (workflows is None or workflow in workflows)
        and (include_figures or kind not in FIGURE_KINDS)
    ]


def resolve(template: str, project_dir: str) -> str:
    return template.format(
        project_dir=project_dir,
        clim_project_dir=f"{project_dir}/data/climate/projections/{CLIM_PROJECT}",
        # S8-04/07: artifact names carry the archive as a prefix, so the templates
        # need the bare project name as well as the directory built from it.
        clim_project=CLIM_PROJECT,
        exp_dir=f"{project_dir}/experiments/{EXPERIMENT_NAME}",
    )


def round_sig(x: float | None, n: int = SIG_FIGS) -> float | None:
    if x is None:
        return None
    x = float(x)
    if not np.isfinite(x):
        return None
    if x == 0.0:
        return 0.0
    return round(x, n - 1 - int(floor(log10(abs(x)))))


def fingerprint_nc(path: str) -> dict:
    with xr.open_dataset(path) as ds:
        per_var: dict[str, dict] = {}
        for name in sorted(ds.variables):
            arr = ds[name]
            values = np.asarray(arr.values)
            entry: dict = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
            }
            if np.issubdtype(values.dtype, np.number):
                finite = values[np.isfinite(values)]
                entry["count_non_nan"] = int(finite.size)
                if finite.size > 0:
                    entry["min"] = round_sig(float(finite.min()))
                    entry["max"] = round_sig(float(finite.max()))
                    entry["mean"] = round_sig(float(finite.mean()))
                    entry["std"] = round_sig(float(finite.std()))
                else:
                    entry["min"] = entry["max"] = entry["mean"] = entry["std"] = None
            else:
                entry["count_non_nan"] = int(values.size)
                entry["min"] = entry["max"] = entry["mean"] = entry["std"] = None
            attrs = {
                k: str(v) for k, v in arr.attrs.items() if k not in VOLATILE_NC_ATTRS
            }
            entry["attrs"] = dict(sorted(attrs.items()))
            per_var[name] = entry
    js = json.dumps(per_var, sort_keys=True, ensure_ascii=False)
    return {
        "type": "netcdf",
        "summary_sha256": hashlib.sha256(js.encode("utf-8")).hexdigest(),
        "summary": per_var,
    }


def fingerprint_csv(path: str) -> dict:
    raw = Path(path).read_bytes()
    text = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    norm = b"\n".join(line.rstrip() for line in text.split(b"\n"))
    return {
        "type": "csv",
        "sha256": hashlib.sha256(norm).hexdigest(),
        "size_bytes": len(raw),
    }


def fingerprint_png(path: str) -> dict:
    return {
        "type": "png",
        "exists": True,
        "size_bytes": Path(path).stat().st_size,
    }


def fingerprint_yaml(path: str) -> dict:
    data = yaml.safe_load(Path(path).read_text())
    js = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    return {
        "type": "yaml",
        "sha256": hashlib.sha256(js.encode("utf-8")).hexdigest(),
    }


FINGERPRINTERS = {
    "png": fingerprint_png,
    "nc": fingerprint_nc,
    "csv": fingerprint_csv,
    "yaml": fingerprint_yaml,
}


#: Kinds compared against a STORED REFERENCE with a tolerance comparator rather
#: than a self-contained fingerprint. `compute_manifest` skips them; each has its
#: own record_*/check_* pair. A kind listed here has no entry in FINGERPRINTERS,
#: so forgetting the skip is a KeyError at record time rather than a silent hash.
REFERENCE_KINDS = frozenset({"discharge", "indicator"})


def compute_manifest(
    project_dir: str,
    workflows: set[str] | None = None,
    include_figures: bool = False,
) -> tuple[dict, list[str]]:
    """Fingerprint every self-contained target.

    REFERENCE_KINDS targets are handled by their own record_*/check_* pairs
    (they need a stored reference plus a tolerance comparator, not a hash).
    """
    out: dict[str, dict] = {}
    missing: list[str] = []
    for _workflow, kind, template in active_targets(workflows, include_figures):
        if kind in REFERENCE_KINDS:
            continue
        path = resolve(template, project_dir)
        if not Path(path).exists():
            missing.append(path)
            continue
        out[path] = FINGERPRINTERS[kind](path)
    return out, missing


# ---------------------------------------------------------------------------
# Discharge series: read, compare (ADR 0001 step 6), record/check integration.
# ---------------------------------------------------------------------------


def read_discharge_series(path: str) -> tuple[list[str], np.ndarray, str]:
    """Parse a Wflow `output.csv` (or a stored reduced reference series).

    Returns (time_strings, q_values, column_name). The first column is the time
    index; the discharge column is the sole remaining column, the sole one named
    ``Q``/``Q_*``, or the primary outlet identified by the sibling deterministic
    ``staticgeoms/outlet_index.csv`` crosswalk. Raises on ambiguity.
    """
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"{path}: expected a time column plus a discharge column")
    value_cols = list(df.columns[1:])
    if len(value_cols) == 1:
        col = value_cols[0]
    else:
        q_cols = [c for c in value_cols if c == "Q" or str(c).startswith("Q_")]
        if not q_cols:
            raise ValueError(
                f"{path}: cannot identify the discharge column among {value_cols}"
            )
        if len(q_cols) == 1:
            col = q_cols[0]
        else:
            model_root = Path(path).parent.parent
            outlet_index_path = model_root / "staticgeoms" / "outlet_index.csv"
            if not outlet_index_path.is_file():
                raise ValueError(
                    f"{path}: cannot identify the primary outlet discharge among "
                    f"{q_cols}; missing {outlet_index_path}"
                )
            outlets = pd.read_csv(outlet_index_path)
            required = {"compat_station_name", "subcatchment_id"}
            missing = sorted(required.difference(outlets.columns))
            if missing:
                raise ValueError(
                    f"{outlet_index_path}: missing primary-outlet columns {missing}"
                )
            primary = outlets.loc[outlets["compat_station_name"].eq("wflow_1")]
            if len(primary) != 1:
                raise ValueError(
                    f"{outlet_index_path}: expected exactly one wflow_1 row"
                )
            expected = f"Q_{int(primary.iloc[0]['subcatchment_id'])}"
            if expected not in q_cols:
                raise ValueError(
                    f"{path}: primary outlet {expected} from {outlet_index_path} "
                    f"is absent from discharge columns {q_cols}"
                )
            col = expected
    times = df[df.columns[0]].astype(str).tolist()
    q = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return times, q, str(col)


def compare_discharge(
    ref_times: list[str],
    ref_q: np.ndarray,
    cur_times: list[str],
    cur_q: np.ndarray,
) -> dict:
    """Time-index-aligned numeric discharge comparator (ADR 0001 step 6).

    Structural checks first (any hit ⇒ structural FAIL, never a numeric pass):
    duplicate timestamps in either series, unequal time-index sets, non-finite
    values. Then, per aligned timestep t, with ATOL = DISCHARGE_ATOL_FRAC *
    mean(Q_ref) and RTOL = DISCHARGE_RTOL::

        fail(t) := |dQ(t)| > ATOL  OR  (Q_ref(t) >= ATOL AND |dQ(t)| > RTOL*Q_ref(t))

    The relative clause is skipped where Q_ref(t) < ATOL (division-safe; near-dry
    steps cannot manufacture materiality). Pass iff no structural mismatch and no
    failing timestep.
    """
    ref_q = np.asarray(ref_q, dtype=float)
    cur_q = np.asarray(cur_q, dtype=float)

    structural: list[str] = []
    if len(set(ref_times)) != len(ref_times):
        structural.append("duplicate timestamps in reference series")
    if len(set(cur_times)) != len(cur_times):
        structural.append("duplicate timestamps in current series")
    n_ref_bad = int((~np.isfinite(ref_q)).sum())
    n_cur_bad = int((~np.isfinite(cur_q)).sum())
    if n_ref_bad:
        structural.append(f"{n_ref_bad} non-finite value(s) in reference series")
    if n_cur_bad:
        structural.append(f"{n_cur_bad} non-finite value(s) in current series")
    ref_set, cur_set = set(ref_times), set(cur_times)
    if ref_set != cur_set:
        only_ref = sorted(ref_set - cur_set)
        only_cur = sorted(cur_set - ref_set)
        structural.append(
            f"time-index mismatch: {len(only_ref)} only-ref, {len(only_cur)} only-cur "
            f"(ref-only e.g. {only_ref[:3]}; cur-only e.g. {only_cur[:3]})"
        )

    base = {
        "ok": False,
        "structural": structural,
        "atol": None,
        "rtol": DISCHARGE_RTOL,
        "mean_ref": None,
        "n": len(ref_times),
        "n_fail": None,
        "max_norm_abs": None,
        "max_rel": None,
        "first_fail": None,
        "worst_fail": None,
    }
    if structural:
        return base

    # Index sets equal and both dedup'd ⇒ reorder current onto reference order.
    cur_map = dict(zip(cur_times, cur_q))
    cur_aligned = np.array([cur_map[t] for t in ref_times], dtype=float)

    mean_ref = float(np.mean(ref_q))
    atol = DISCHARGE_ATOL_FRAC * abs(mean_ref)
    rtol = DISCHARGE_RTOL
    dq = np.abs(cur_aligned - ref_q)

    abs_fail = dq > atol
    rel_subset = ref_q >= atol
    rel_fail = np.zeros_like(dq, dtype=bool)
    if atol > 0:
        rel_fail[rel_subset] = dq[rel_subset] > rtol * ref_q[rel_subset]
    fail = abs_fail | rel_fail
    n_fail = int(fail.sum())

    max_norm_abs = float(dq.max() / abs(mean_ref)) if mean_ref != 0 else float(dq.max())
    if rel_subset.any():
        max_rel = float((dq[rel_subset] / ref_q[rel_subset]).max())
    else:
        max_rel = 0.0
    first_fail = ref_times[int(np.argmax(fail))] if n_fail else None
    worst_fail = ref_times[int(np.argmax(dq))] if dq.size else None

    return {
        "ok": n_fail == 0,
        "structural": [],
        "atol": atol,
        "rtol": rtol,
        "mean_ref": mean_ref,
        "n": len(ref_times),
        "n_fail": n_fail,
        "max_norm_abs": max_norm_abs,
        "max_rel": max_rel,
        "first_fail": first_fail,
        "worst_fail": worst_fail,
    }


def _discharge_report_lines(report: dict) -> list[str]:
    if report["structural"]:
        return [f"structural: {s}" for s in report["structural"]]
    atol = report["atol"]
    lines = [
        f"{report['n_fail']}/{report['n']} timestep(s) exceed tolerance "
        f"(ATOL={atol:.4g} = {DISCHARGE_ATOL_FRAC:g}*mean(Q_ref), "
        f"RTOL={report['rtol']:.0%})",
        f"max |dQ|/mean(Q_ref) = {report['max_norm_abs']:.4g}; "
        f"max relative (Q_ref>=ATOL) = {report['max_rel']:.4g}",
    ]
    if report["n_fail"]:
        lines.append(
            f"first offending: {report['first_fail']}; worst |dQ| at: {report['worst_fail']}"
        )
    return lines


def _write_reference_series(
    path: Path, times: list[str], q: np.ndarray, col: str
) -> None:
    """Write a reduced reference series (time,Q) round-trippably (full float repr)."""
    out = [f"time,{col}"]
    out.extend(f"{t},{float(v)!r}" for t, v in zip(times, q))
    path.write_text("\n".join(out) + "\n")


def _discharge_slug(resolved_path: str) -> str:
    return hashlib.sha1(resolved_path.encode("utf-8")).hexdigest()[:16] + ".csv"


def record_discharge(
    project_dir: str, ref_dir: Path, workflows: set[str] | None = None
) -> tuple[dict, list[str]]:
    """Store reduced reference series for every in-scope discharge target and
    return their manifest rows plus any missing target paths."""
    rows: dict[str, dict] = {}
    missing: list[str] = []
    # Discharge is never a FIGURE_KIND, so the figure filter cannot touch it —
    # include_figures=True keeps this loop's universe identical either way.
    for _workflow, kind, template in active_targets(workflows, include_figures=True):
        if kind != "discharge":
            continue
        path = resolve(template, project_dir)
        if not Path(path).exists():
            missing.append(path)
            continue
        times, q, col = read_discharge_series(path)
        rel = f"{DISCHARGE_REF_SUBDIR}/{_discharge_slug(path)}"
        sidecar = ref_dir / rel
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        _write_reference_series(sidecar, times, q, col)
        rows[path] = {
            "type": "discharge",
            "column": col,
            "n_rows": len(times),
            "mean_ref": round_sig(float(np.mean(q))),
            "ref_series": rel,
        }
    return rows, missing


def check_discharge(
    project_dir: str, ref_dir: Path, rec_targets: dict
) -> list[tuple[str, list[str]]]:
    """Compare every recorded discharge target (already scoped by the caller)
    against its stored reference series. Returns (path, diff-lines) failures."""
    failures: list[tuple[str, list[str]]] = []
    for path, rec in rec_targets.items():
        if not isinstance(rec, dict) or rec.get("type") != "discharge":
            continue
        if not Path(path).exists():
            failures.append((path, ["target missing on disk"]))
            continue
        sidecar = ref_dir / rec["ref_series"]
        if not sidecar.exists():
            failures.append((path, [f"stored reference series missing: {sidecar}"]))
            continue
        ref_times, ref_q, _ = read_discharge_series(str(sidecar))
        cur_times, cur_q, _ = read_discharge_series(path)
        report = compare_discharge(ref_times, ref_q, cur_times, cur_q)
        if not report["ok"]:
            failures.append((path, _discharge_report_lines(report)))
    return failures


# ---------------------------------------------------------------------------
# Indicator tables: read, compare (R11 Q8), record/check integration.
#
# Structurally parallel to the discharge block above, and deliberately so: Q8
# ruled these targets onto "check_baseline.py's existing compare_discharge-style
# tolerance comparator". The one real difference is that a discharge target is a
# SINGLE series with one meaningful magnitude, while an indicator table stacks
# many series -- metrics in different units, gauges with different catchment
# areas -- into one long table. So the tolerance is derived per group rather than
# once for the file. Everything else (structural-checks-first, ATOL from the
# reference's own mean, RTOL as a large-value tightener, division-safe skip
# below ATOL) is the same rule.
# ---------------------------------------------------------------------------


def read_indicator_table(path: str) -> pd.DataFrame:
    """Parse an indicator table (or a stored reference copy) as strings + value.

    Every column but ``value`` is read as a STRING and left untouched. That is
    what makes the key alignment exact: ``temp_change`` and ``precip_change``
    are floats on disk, and letting pandas parse them would make two runs that
    wrote ``1.3`` and ``1.3000000000000003`` land in different groups and report
    as a key-set mismatch instead of the sub-tolerance move they are.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if INDICATOR_VALUE_COLUMN not in df.columns:
        raise ValueError(
            f"{path}: no {INDICATOR_VALUE_COLUMN!r} column; columns are "
            f"{list(df.columns)}"
        )
    df[INDICATOR_VALUE_COLUMN] = pd.to_numeric(
        df[INDICATOR_VALUE_COLUMN], errors="coerce"
    )
    return df


def _indicator_key_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != INDICATOR_VALUE_COLUMN]


def compare_indicator_table(ref: pd.DataFrame, cur: pd.DataFrame) -> dict:
    """Key-aligned numeric comparator for a long indicator table (R11 Q8).

    Structural checks first (any hit ⇒ structural FAIL, never a numeric pass):
    column-set or column-order mismatch, duplicate row keys in either table,
    unequal row-key sets, non-finite values. Then, per group g of
    ``INDICATOR_GROUP_COLUMNS`` present in the table, with
    ATOL(g) = INDICATOR_ATOL_FRAC * mean(|value_ref| within g) and
    RTOL = INDICATOR_RTOL::

        fail(r) := |dv(r)| > ATOL(g)
                   OR (|v_ref(r)| >= ATOL(g) AND |dv(r)| > RTOL*|v_ref(r)|)

    Pass iff no structural mismatch and no failing row.
    """
    # The group loop maps index labels back to positions via get_indexer, which
    # is only sound on a clean 0..n-1 index. read_indicator_table always gives
    # one; a caller that hands in a FILTERED frame would otherwise mis-index
    # silently rather than fail. Refuse instead of trusting the caller.
    ref = ref.reset_index(drop=True)
    cur = cur.reset_index(drop=True)

    structural: list[str] = []

    ref_cols, cur_cols = list(ref.columns), list(cur.columns)
    if ref_cols != cur_cols:
        only_ref = [c for c in ref_cols if c not in cur_cols]
        only_cur = [c for c in cur_cols if c not in ref_cols]
        if only_ref or only_cur:
            structural.append(
                f"column mismatch: {len(only_ref)} only-ref {only_ref}, "
                f"{len(only_cur)} only-cur {only_cur}"
            )
        else:
            structural.append(f"column ORDER changed: {cur_cols} vs {ref_cols}")

    base = {
        "ok": False,
        "structural": structural,
        "rtol": INDICATOR_RTOL,
        "n": len(ref),
        "n_groups": None,
        "n_fail": None,
        "max_rel": None,
        "worst": None,
        "failing_groups": [],
    }
    if structural:
        return base

    key_cols = _indicator_key_columns(ref)
    ref_key = ref[key_cols].agg("\x1f".join, axis=1)
    cur_key = cur[key_cols].agg("\x1f".join, axis=1)

    if ref_key.duplicated().any():
        structural.append(
            f"{int(ref_key.duplicated().sum())} duplicate row key(s) in reference table"
        )
    if cur_key.duplicated().any():
        structural.append(
            f"{int(cur_key.duplicated().sum())} duplicate row key(s) in current table"
        )
    n_ref_bad = int((~np.isfinite(ref[INDICATOR_VALUE_COLUMN])).sum())
    n_cur_bad = int((~np.isfinite(cur[INDICATOR_VALUE_COLUMN])).sum())
    if n_ref_bad:
        structural.append(f"{n_ref_bad} non-finite value(s) in reference table")
    if n_cur_bad:
        structural.append(f"{n_cur_bad} non-finite value(s) in current table")

    ref_set, cur_set = set(ref_key), set(cur_key)
    if ref_set != cur_set:
        only_ref = sorted(ref_set - cur_set)
        only_cur = sorted(cur_set - ref_set)

        def _show(keys: list[str]) -> list[str]:
            return [k.replace("\x1f", "|") for k in keys[:3]]

        structural.append(
            f"row-key mismatch: {len(only_ref)} only-ref, {len(only_cur)} only-cur "
            f"(ref-only e.g. {_show(only_ref)}; cur-only e.g. {_show(only_cur)})"
        )

    if structural:
        base["structural"] = structural
        return base

    # Keys equal and both dedup'd ⇒ reorder current onto reference order.
    cur_vals = cur.set_index(cur_key)[INDICATOR_VALUE_COLUMN]
    cur_aligned = cur_vals.reindex(ref_key).to_numpy(dtype=float)
    ref_vals = ref[INDICATOR_VALUE_COLUMN].to_numpy(dtype=float)
    dv = np.abs(cur_aligned - ref_vals)

    group_cols = [c for c in INDICATOR_GROUP_COLUMNS if c in ref.columns]
    if group_cols:
        group_key = ref[group_cols].agg("\x1f".join, axis=1)
    else:
        # No grouping column present ⇒ one group, exactly the discharge rule.
        group_key = pd.Series(["*"] * len(ref), index=ref.index)

    fail = np.zeros(len(ref), dtype=bool)
    rel = np.zeros(len(ref), dtype=float)
    atol_of: dict[str, float] = {}
    for gname, idx in group_key.groupby(group_key).groups.items():
        pos = ref.index.get_indexer(idx)
        atol = INDICATOR_ATOL_FRAC * float(np.mean(np.abs(ref_vals[pos])))
        atol_of[str(gname)] = atol
        g_dv, g_ref = dv[pos], np.abs(ref_vals[pos])
        abs_fail = g_dv > atol
        rel_subset = g_ref >= atol
        rel_fail = np.zeros_like(g_dv, dtype=bool)
        if atol > 0:
            rel_fail[rel_subset] = g_dv[rel_subset] > INDICATOR_RTOL * g_ref[rel_subset]
        fail[pos] = abs_fail | rel_fail
        with np.errstate(divide="ignore", invalid="ignore"):
            rel[pos] = np.where(g_ref > 0, g_dv / np.where(g_ref > 0, g_ref, 1.0), 0.0)

    n_fail = int(fail.sum())
    worst = None
    if len(ref):
        w = int(np.argmax(rel))
        worst = ref_key.iloc[w].replace("\x1f", "|")
    failing_groups = sorted({str(g) for g in group_key[fail].unique()})

    return {
        "ok": n_fail == 0,
        "structural": [],
        "rtol": INDICATOR_RTOL,
        "n": len(ref),
        "n_groups": len(atol_of),
        "n_fail": n_fail,
        "max_rel": float(rel.max()) if len(ref) else 0.0,
        "worst": worst,
        "failing_groups": failing_groups,
    }


def _indicator_report_lines(report: dict) -> list[str]:
    if report["structural"]:
        return [f"structural: {s}" for s in report["structural"]]
    lines = [
        f"{report['n_fail']}/{report['n']} row(s) exceed tolerance across "
        f"{report['n_groups']} group(s) "
        f"(ATOL={INDICATOR_ATOL_FRAC:g}*mean|value_ref| per "
        f"{'+'.join(INDICATOR_GROUP_COLUMNS)}, RTOL={report['rtol']:.0%})",
        f"max relative move = {report['max_rel']:.4g} at {report['worst']}",
    ]
    if report["n_fail"]:
        shown = report["failing_groups"][:5]
        more = len(report["failing_groups"]) - len(shown)
        lines.append(
            f"failing group(s): {[g.replace(chr(31), '|') for g in shown]}"
            + (f" (+{more} more)" if more > 0 else "")
        )
    return lines


def _indicator_slug(resolved_path: str) -> str:
    return hashlib.sha1(resolved_path.encode("utf-8")).hexdigest()[:16] + ".csv"


def record_indicator(
    project_dir: str, ref_dir: Path, workflows: set[str] | None = None
) -> tuple[dict, list[str]]:
    """Store a reference copy of every in-scope indicator table and return their
    manifest rows plus any missing target paths.

    The stored copy is the table VERBATIM. Unlike the discharge sidecar there is
    nothing to reduce -- the table is already the reduction -- and copying bytes
    keeps the reference readable as the artifact it mirrors.
    """
    rows: dict[str, dict] = {}
    missing: list[str] = []
    # Never a FIGURE_KIND, so include_figures=True keeps this universe identical
    # either way -- same reasoning as record_discharge.
    for _workflow, kind, template in active_targets(workflows, include_figures=True):
        if kind != "indicator":
            continue
        path = resolve(template, project_dir)
        if not Path(path).exists():
            missing.append(path)
            continue
        df = read_indicator_table(path)
        rel = f"{INDICATOR_REF_SUBDIR}/{_indicator_slug(path)}"
        sidecar = ref_dir / rel
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_bytes(Path(path).read_bytes())
        group_cols = [c for c in INDICATOR_GROUP_COLUMNS if c in df.columns]
        n_groups = (
            int(df[group_cols].agg("\x1f".join, axis=1).nunique()) if group_cols else 1
        )
        rows[path] = {
            "type": "indicator",
            "columns": list(df.columns),
            "n_rows": int(len(df)),
            "n_groups": n_groups,
            "ref_table": rel,
        }
    return rows, missing


def check_indicator(
    project_dir: str, ref_dir: Path, rec_targets: dict
) -> list[tuple[str, list[str]]]:
    """Compare every recorded indicator target (already scoped by the caller)
    against its stored reference table. Returns (path, diff-lines) failures."""
    failures: list[tuple[str, list[str]]] = []
    for path, rec in rec_targets.items():
        if not isinstance(rec, dict) or rec.get("type") != "indicator":
            continue
        if not Path(path).exists():
            failures.append((path, ["target missing on disk"]))
            continue
        sidecar = ref_dir / rec["ref_table"]
        if not sidecar.exists():
            failures.append((path, [f"stored reference table missing: {sidecar}"]))
            continue
        report = compare_indicator_table(
            read_indicator_table(str(sidecar)), read_indicator_table(path)
        )
        if not report["ok"]:
            failures.append((path, _indicator_report_lines(report)))
    return failures


def diff_png(rec: dict, cur: dict) -> list[str]:
    if not cur.get("exists"):
        return ["missing"]
    rec_size, cur_size = rec["size_bytes"], cur["size_bytes"]
    if rec_size == 0:
        return [] if cur_size == 0 else [f"size {cur_size}, expected 0"]
    rel = abs(cur_size - rec_size) / rec_size
    if rel > PNG_TOLERANCE_FRAC:
        return [
            f"size {cur_size} vs {rec_size} ({rel:.1%} drift > {PNG_TOLERANCE_FRAC:.0%})"
        ]
    return []


NUMERIC_STATS = ("min", "max", "mean", "std")


def _within_tol(r: float | None, c: float | None, tol: float) -> bool:
    if r is None or c is None or tol <= 0:
        return False
    denom = max(abs(r), abs(c), 1e-300)
    return abs(c - r) / denom <= tol


def diff_nc(rec: dict, cur: dict, tolerance: float = 0.0) -> list[str]:
    diffs: list[str] = []
    rec_summary = rec.get("summary", {})
    cur_summary = cur.get("summary", {})
    for var in sorted(set(rec_summary) | set(cur_summary)):
        if var not in rec_summary:
            diffs.append(f"variable {var}: new in current run")
            continue
        if var not in cur_summary:
            diffs.append(f"variable {var}: missing in current run")
            continue
        for stat in (
            "shape",
            "dtype",
            "count_non_nan",
            "min",
            "max",
            "mean",
            "std",
            "attrs",
        ):
            r, c = rec_summary[var].get(stat), cur_summary[var].get(stat)
            if r == c:
                continue
            if stat in NUMERIC_STATS and _within_tol(r, c, tolerance):
                continue
            diffs.append(f"variable {var} {stat}: {c} vs {r}")
    return diffs


def diff_hashed(rec: dict, cur: dict) -> list[str]:
    if rec.get("sha256") != cur.get("sha256"):
        return [f"sha256 {cur.get('sha256')} vs {rec.get('sha256')}"]
    return []


def diff_records(rec: dict, cur: dict, tolerance: float = 0.0) -> list[str]:
    if rec["type"] == "png":
        return diff_png(rec, cur)
    if rec["type"] == "netcdf":
        return diff_nc(rec, cur, tolerance)
    return diff_hashed(rec, cur)


def _want_figures(args: argparse.Namespace) -> bool:
    """Read the flag defensively.

    `cmd_record`/`cmd_check` are called directly from tests with a hand-built
    Namespace that predates this option, exactly as `--workflow` already is.
    """
    return bool(getattr(args, "include_figures", False))


def cmd_record(args: argparse.Namespace) -> int:
    """Record fingerprints. With `--workflow`, record ONLY the selected
    workflow(s) and MERGE into the existing manifest (preserve the other rows);
    without it, record every target and overwrite (the canonical full record)."""
    selected = set(args.workflow) if getattr(args, "workflow", None) else None
    ref_dir = args.manifest.parent

    manifest, missing = compute_manifest(
        args.project_dir, workflows=selected, include_figures=_want_figures(args)
    )
    disch_rows, disch_missing = record_discharge(
        args.project_dir, ref_dir, workflows=selected
    )
    ind_rows, ind_missing = record_indicator(
        args.project_dir, ref_dir, workflows=selected
    )
    missing = missing + disch_missing + ind_missing
    if missing:
        scope = "" if selected is None else f" for workflow(s) {sorted(selected)}"
        sys.stderr.write(
            f"Missing targets{scope} -- refusing to record an incomplete manifest:\n"
        )
        for p in missing:
            sys.stderr.write(f"  - {p}\n")
        return 1

    new_rows = {**manifest, **disch_rows, **ind_rows}
    if selected is not None and args.manifest.exists():
        # Merge: keep every recorded row NOT owned by a selected workflow, then
        # overlay the freshly-recorded selected rows. Never clobber wf2/wf3.
        existing = json.loads(args.manifest.read_text()).get("targets", {})
        # Prune exactly what was re-recorded — same filter, so an excluded
        # figure row is left alone rather than deleted by a merge that never
        # intended to touch it.
        selected_paths = {
            resolve(template, args.project_dir)
            for _workflow, _kind, template in active_targets(
                selected, include_figures=_want_figures(args)
            )
        }
        targets = {p: r for p, r in existing.items() if p not in selected_paths}
        targets.update(new_rows)
        verb = f"merged workflow(s) {sorted(selected)}"
    else:
        targets = new_rows
        verb = "recorded"

    payload = {
        "version": MANIFEST_VERSION,
        "project_dir": args.project_dir,
        # R7-21: who wrote this. The fixture is branch-shared mutable state, so
        # without provenance a later `check` cannot tell "my code drifted" from
        # "another branch last wrote this tree".
        "recorded_by": git_provenance(),
        "targets": targets,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(
        f"{verb}: {len(new_rows)} target(s) -> {args.manifest} ({len(targets)} total)"
    )
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    if not args.manifest.exists():
        sys.stderr.write(f"Manifest not found: {args.manifest}\n")
        sys.stderr.write("Run `check_baseline.py record` first.\n")
        return 2
    recorded = json.loads(args.manifest.read_text())
    rec_targets = recorded["targets"]
    ref_dir = args.manifest.parent

    # R7-21: provenance is advisory and printed BEFORE the verdict, so a
    # disagreement frames whatever follows. It never changes the exit code --
    # the failure mode it guards against is silent MISATTRIBUTION (someone
    # else's branch last wrote the shared fixture), not corruption, and a
    # cross-branch check is a legitimate thing to do deliberately.
    rec_prov = recorded.get("recorded_by")
    cur_prov = git_provenance()
    if rec_prov is None:
        print(
            "note: this manifest predates provenance stamping; cannot tell "
            "which branch produced the recorded artifacts. Re-record to stamp it."
        )
    elif cur_prov is not None:
        if rec_prov.get("commit") != cur_prov.get("commit"):
            same_branch = rec_prov.get("branch") == cur_prov.get("branch")
            print(
                f"WARNING: manifest recorded by {format_provenance(rec_prov)}, "
                f"checking from {format_provenance(cur_prov)}."
            )
            print(
                "         The baseline fixture is untracked and therefore "
                "SHARED BY EVERY BRANCH, so a pass here may mean the tree "
                "matches another branch's code."
                if not same_branch
                else "         Same branch, different commit -- expected if the "
                "recorded run predates your latest commits."
            )
        elif cur_prov.get("dirty") and not rec_prov.get("dirty"):
            print(
                "note: manifest recorded from a clean tree; checking from a "
                "dirty one. Uncommitted changes are not in the recorded code."
            )

    selected = set(args.workflow) if args.workflow else None
    current, missing = compute_manifest(
        args.project_dir, workflows=selected, include_figures=_want_figures(args)
    )
    # Apply the in-scope universe symmetrically: filter the recorded side to the
    # same resolved paths so the missing/diff/orphan/count logic all operate on
    # one target set (design ext2-1). This now covers the FIGURE filter as well
    # as `--workflow`: without it, a manifest recorded before figures were
    # excluded keeps reporting its stale png rows in the "N target(s) match"
    # count while nothing actually compares them.
    in_scope_paths = {
        resolve(template, args.project_dir)
        for _workflow, _kind, template in active_targets(
            selected, include_figures=_want_figures(args)
        )
    }
    rec_targets = {p: rec for p, rec in rec_targets.items() if p in in_scope_paths}

    failures: list[tuple[str, list[str]]] = []
    for p in missing:
        if p in rec_targets:
            failures.append((p, ["target missing on disk"]))
    for path, rec in rec_targets.items():
        if path not in current:
            continue
        diffs = diff_records(rec, current[path], args.tolerance)
        if diffs:
            failures.append((path, diffs))
    # REFERENCE_KINDS: compared via their tolerance comparators, not fingerprints.
    failures.extend(check_discharge(args.project_dir, ref_dir, rec_targets))
    failures.extend(check_indicator(args.project_dir, ref_dir, rec_targets))
    for path in sorted(set(current) - set(rec_targets)):
        failures.append((path, ["target present but not in manifest"]))

    tol_note = f" (tolerance {args.tolerance:g})" if args.tolerance > 0 else ""
    if not failures:
        print(f"OK - {len(rec_targets)} target(s) match manifest{tol_note}.")
        return 0
    print(f"FAIL - {len(failures)} target(s) differ from manifest{tol_note}:")
    for path, diffs in failures:
        print(f"  {path}")
        for d in diffs:
            print(f"    - {d}")
    return 1


def cmd_compare(args: argparse.Namespace) -> int:
    """One-off discharge comparison of two output.csv files (ADR steps 4b/5).
    Exit 0 = pass (reproducible / immaterial); 1 = differ (material / structural)."""
    ref_times, ref_q, _ = read_discharge_series(args.ref)
    cur_times, cur_q, _ = read_discharge_series(args.cur)
    report = compare_discharge(ref_times, ref_q, cur_times, cur_q)
    if report["structural"]:
        print(f"STRUCTURAL MISMATCH: ref={args.ref} cur={args.cur}")
    else:
        verdict = (
            "PASS (immaterial / reproducible)" if report["ok"] else "FAIL (material)"
        )
        print(f"{verdict}: ref={args.ref} cur={args.cur}")
    for line in _discharge_report_lines(report):
        print(f"  - {line}")
    if not report["structural"] and report["ok"]:
        print(
            f"  - 0/{report['n']} timesteps exceed tolerance "
            f"(ATOL={report['atol']:.4g}, RTOL={report['rtol']:.0%}); "
            f"max |dQ|/mean(Q_ref) = {report['max_norm_abs']:.4g}"
        )
    return 0 if report["ok"] else 1


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--project-dir",
        default=PROJECT_DIR_DEFAULT,
        help=f"Project directory (default: {PROJECT_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH_DEFAULT,
        help=f"Manifest path (default: {MANIFEST_PATH_DEFAULT})",
    )
    # Only `record` and `check` take this — `compare` does not read TARGETS.
    parser.add_argument(
        "--include-figures",
        action="store_true",
        help="Include figure targets "
        f"({'/'.join(sorted(FIGURE_KINDS))}). Excluded by default: "
        "a figure is a terminal artifact nothing downstream reads, "
        "and it is fingerprinted by BYTE SIZE, so any cosmetic edit "
        "fails the gate without indicating a defect. Record and "
        "check must pass this flag identically.",
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    record_p = sub.add_parser("record", help="Record fingerprints to manifest")
    _add_common(record_p)
    record_p.add_argument(
        "--workflow",
        action="append",
        choices=list(WORKFLOWS),
        default=None,
        help="Record ONLY the given workflow(s) and MERGE into the "
        "existing manifest (other workflows' rows are preserved). "
        "Repeatable. Omit to record all targets (overwrite).",
    )

    check_p = sub.add_parser("check", help="Check current outputs against manifest")
    _add_common(check_p)
    check_p.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Relative tolerance for netCDF numeric stats "
        "(default 0 = exact). Use 1e-9 for cross-env comparison.",
    )
    check_p.add_argument(
        "--workflow",
        action="append",
        choices=list(WORKFLOWS),
        default=None,
        help="Restrict the check to targets tagged with the given "
        "workflow(s). Repeatable. Applied symmetrically to the "
        "recorded and current sides. Omit to check all targets.",
    )

    compare_p = sub.add_parser(
        "compare", help="Compare two Wflow output.csv discharge series (ADR steps 4b/5)"
    )
    compare_p.add_argument("--ref", required=True, help="Reference output.csv")
    compare_p.add_argument("--cur", required=True, help="Candidate output.csv")

    args = p.parse_args()
    if args.cmd == "record":
        sys.exit(cmd_record(args))
    if args.cmd == "compare":
        sys.exit(cmd_compare(args))
    sys.exit(cmd_check(args))


if __name__ == "__main__":
    main()
