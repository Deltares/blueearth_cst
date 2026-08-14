"""Unit tests for dev/scripts/semantic_tree_diff.py (R06 milestone tooling).

Each comparator: equal inputs pass; a seeded perturbation fails. The NetCDF
comparator gets a dedicated coordinate-PERMUTATION test -- the discriminator
that proves it is element-wise (no realignment), not aggregate-stat based
(design §9 ext2-02).
"""

import os
import sys

import numpy as np
import pytest
import xarray as xr
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dev", "scripts"))
import semantic_tree_diff as std  # noqa: E402

# ---------------------------------------------------------------------------
# .nc element-wise comparator
# ---------------------------------------------------------------------------


def _write_nc(path, data, x=(0, 1, 2), attrs=None):
    ds = xr.Dataset(
        {"var": (("x",), np.asarray(data, dtype=float))},
        coords={"x": list(x)},
        attrs=attrs or {},
    )
    ds.to_netcdf(path)


def test_nc_equal_passes(tmp_path):
    a = tmp_path / "a.nc"
    b = tmp_path / "b.nc"
    _write_nc(a, [1.0, 2.0, 3.0])
    _write_nc(b, [1.0, 2.0, 3.0])
    assert std.compare_nc(str(a), str(b), tol=0.0) == []


def test_nc_value_perturbation_fails(tmp_path):
    a = tmp_path / "a.nc"
    b = tmp_path / "b.nc"
    _write_nc(a, [1.0, 2.0, 3.0])
    _write_nc(b, [1.0, 2.0, 3.5])  # one element perturbed
    diffs = std.compare_nc(str(a), str(b), tol=1e-9)
    assert diffs and any("out of tolerance" in d for d in diffs)


def test_nc_coordinate_permutation_fails(tmp_path):
    """The acid test: same data + coords but permuted coord ORDER must FAIL.

    A permutation preserves every aggregate stat (min/max/mean/std/count) --
    only an element-wise, no-realign comparator catches it."""
    a = tmp_path / "a.nc"
    b = tmp_path / "b.nc"
    # same {x: value} pairs, different stored order
    _write_nc(a, [10.0, 20.0, 30.0], x=(0, 1, 2))
    _write_nc(b, [30.0, 10.0, 20.0], x=(2, 0, 1))
    diffs = std.compare_nc(str(a), str(b), tol=0.0)
    assert diffs, "permuted coordinate order must FAIL (no realignment)"


def test_nc_nan_mask_mismatch_fails(tmp_path):
    a = tmp_path / "a.nc"
    b = tmp_path / "b.nc"
    _write_nc(a, [1.0, np.nan, 3.0])
    _write_nc(b, [1.0, 2.0, 3.0])
    diffs = std.compare_nc(str(a), str(b), tol=1e-9)
    assert any("NaN mask" in d for d in diffs)


def test_nc_nonvolatile_attr_change_fails(tmp_path):
    a = tmp_path / "a.nc"
    b = tmp_path / "b.nc"
    _write_nc(a, [1.0, 2.0, 3.0], attrs={"units": "m3/s"})
    _write_nc(b, [1.0, 2.0, 3.0], attrs={"units": "cfs"})
    diffs = std.compare_nc(str(a), str(b), tol=0.0)
    assert any("attrs" in d for d in diffs)


def test_nc_volatile_attr_change_passes(tmp_path):
    a = tmp_path / "a.nc"
    b = tmp_path / "b.nc"
    _write_nc(a, [1.0, 2.0, 3.0], attrs={"history": "created monday"})
    _write_nc(b, [1.0, 2.0, 3.0], attrs={"history": "created tuesday"})
    assert std.compare_nc(str(a), str(b), tol=0.0) == []


# --- the CMIP6 merge classes' inherited attrs (R9 P2 F4) -------------------
# The exclusion is SCOPED to a path class, so it takes two tests: one that it
# applies where it should, and one that it does not apply anywhere else. The
# second is the load-bearing half -- folding these keys into the global
# VOLATILE_NC_ATTRS would have made the first pass on its own.

_INHERITED = {
    "variable_id": ("tas", "pr"),
    "tracking_id": ("hdl:21.14100/aaa", "hdl:21.14100/bbb"),
    "status": ("2020-03-22;created", "2020-02-06;created"),
}


def _write_inherited_pair(root, subdir):
    """Two identical datasets differing ONLY in the inherited CMIP6 attrs."""
    paths = []
    for side in ("ref", "cur"):
        d = root / side / subdir
        d.mkdir(parents=True, exist_ok=True)
        path = d / "slice.nc"
        idx = 0 if side == "ref" else 1
        _write_nc(
            path,
            [1.0, 2.0, 3.0],
            attrs={k: v[idx] for k, v in _INHERITED.items()},
        )
        paths.append(path)
    return paths


def test_inherited_cmip6_attrs_ignored_in_the_merge_classes(tmp_path):
    """`cmip6/{raw,scalar}/` files merge pr and tas, so these attrs are wrong.

    Whichever member wins `xr.merge`'s attr resolution stamps the file, so two
    independent fetches of the same slice disagree while every value matches.
    """
    for subdir in (
        "data/climate/projections/cmip6/raw",
        "data/climate/projections/cmip6/scalar",
    ):
        ref, cur = _write_inherited_pair(tmp_path / subdir.replace("/", "_"), subdir)
        assert std.compare_nc(str(ref), str(cur), tol=0.0) == [], subdir


def test_inherited_cmip6_attrs_still_compared_everywhere_else(tmp_path):
    """The exclusion is path-scoped, and this is what pins that.

    If these keys were added to the global VOLATILE_NC_ATTRS instead, this test
    fails -- which is the whole reason the scoped form was chosen.
    """
    ref, cur = _write_inherited_pair(tmp_path, "data/spatial")
    diffs = std.compare_nc(str(ref), str(cur), tol=0.0)
    assert any("attrs" in d for d in diffs), "attr difference went unreported"
    reported = " ".join(diffs)
    for key in _INHERITED:
        assert key in reported, f"{key} masked outside the cmip6 merge classes"


# ---------------------------------------------------------------------------
# .toml normalized comparator
# ---------------------------------------------------------------------------


def test_toml_key_order_and_comments_pass(tmp_path):
    a = tmp_path / "a.toml"
    b = tmp_path / "b.toml"
    a.write_text("# comment A\n[s]\nx = 1\ny = 2\n")
    b.write_text("[s]\ny = 2\nx = 1\n# comment B\n")  # reordered + diff comment
    assert std.compare_toml(str(a), str(b)) == []


def test_toml_value_change_fails(tmp_path):
    a = tmp_path / "a.toml"
    b = tmp_path / "b.toml"
    a.write_text("[s]\nx = 1\n")
    b.write_text("[s]\nx = 2\n")
    diffs = std.compare_toml(str(a), str(b))
    assert diffs and "s.x" in diffs[0]


# ---------------------------------------------------------------------------
# copied-config YAML normalize-then-compare
# ---------------------------------------------------------------------------


def _write_yaml(path, doc):
    path.write_text(yaml.safe_dump(doc))


def test_copied_config_mapped_path_normalizes(tmp_path):
    """The documented old->new path rewrite is the ONLY allowed difference."""
    ref = tmp_path / "config" / "ref.yml"
    cur = tmp_path / "config" / "cur.yml"
    ref.parent.mkdir(parents=True)
    _write_yaml(ref, {"project": {"data_sources": "config/deltares_data.yml"}})
    _write_yaml(cur, {"project": {"data_sources": "config/catalogs/deltares_data.yml"}})
    assert std.compare_copied_config(str(ref), str(cur)) == []


def test_copied_config_all_four_keys_normalize(tmp_path):
    """data_sources_climate is in the map (commit-2 as-built inventory)."""
    ref = tmp_path / "ref.yml"
    cur = tmp_path / "cur.yml"
    _write_yaml(
        ref,
        {
            "project": {
                "data_sources": "config/deltares_data.yml",
                "data_sources_climate": "config/cmip6_data.yml",
            },
            "workflows": {
                "build_model": {
                    "model_build_config": "config/wflow_build_model.yml",
                    "waterbodies_config": "config/wflow_update_waterbodies.yml",
                }
            },
        },
    )
    _write_yaml(
        cur,
        {
            "project": {
                "data_sources": "config/catalogs/deltares_data.yml",
                "data_sources_climate": "config/catalogs/cmip6_data.yml",
            },
            "workflows": {
                "build_model": {
                    "model_build_config": "config/defaults/wflow_build_model.yml",
                    "waterbodies_config": "config/defaults/wflow_update_waterbodies.yml",
                }
            },
        },
    )
    assert std.compare_copied_config(str(ref), str(cur)) == []


def test_copied_config_normalizes_the_intermediate_templates_era(tmp_path):
    """These two keys have moved TWICE (pre-R6 flat -> config/templates/ ->
    config/defaults/, 2026-08-11), so a reference tree recorded in the middle
    era must normalize onto the current path just as a pre-R6 one does. Without
    the second hop in COPIED_CONFIG_PATH_MAP this compares as a real drift."""
    ref = tmp_path / "ref.yml"
    cur = tmp_path / "cur.yml"
    _write_yaml(
        ref,
        {
            "workflows": {
                "build_model": {
                    "model_build_config": "config/templates/wflow_build_model.yml",
                    "waterbodies_config": "config/templates/wflow_update_waterbodies.yml",
                }
            }
        },
    )
    _write_yaml(
        cur,
        {
            "workflows": {
                "build_model": {
                    "model_build_config": "config/defaults/wflow_build_model.yml",
                    "waterbodies_config": "config/defaults/wflow_update_waterbodies.yml",
                }
            }
        },
    )
    assert std.compare_copied_config(str(ref), str(cur)) == []


def test_copied_config_reflexive_self_compare_clean(tmp_path):
    """Reflexivity: comparing a pre-R6 (OLD-path) snapshot against itself is
    clean. Pins the bug the self-smoke caught -- the directional normalize must
    not false-fail on identical inputs."""
    x = tmp_path / "x.yml"
    _write_yaml(
        x,
        {
            "project": {
                "data_sources": "config/deltares_data.yml",
                "data_sources_climate": "config/cmip6_data.yml",
            },
        },
    )
    assert std.compare_copied_config(str(x), str(x)) == []


def test_copied_config_nonpath_value_change_fails(tmp_path):
    """A non-path value change is a real FAIL even with a valid path rewrite."""
    ref = tmp_path / "ref.yml"
    cur = tmp_path / "cur.yml"
    _write_yaml(
        ref,
        {
            "project": {"data_sources": "config/deltares_data.yml"},
            "shared": {"clim_historical": "era5"},
        },
    )
    _write_yaml(
        cur,
        {
            "project": {"data_sources": "config/catalogs/deltares_data.yml"},
            "shared": {"clim_historical": "chirps"},
        },
    )  # drift
    diffs = std.compare_copied_config(str(ref), str(cur))
    assert diffs and any("clim_historical" in d for d in diffs)


def test_copied_config_unmapped_path_value_fails(tmp_path):
    """A path value not in the map is left untouched and must fail equality."""
    ref = tmp_path / "ref.yml"
    cur = tmp_path / "cur.yml"
    _write_yaml(ref, {"project": {"data_sources": "config/some_other.yml"}})
    _write_yaml(cur, {"project": {"data_sources": "config/catalogs/some_other.yml"}})
    diffs = std.compare_copied_config(str(ref), str(cur))
    assert diffs


# ---------------------------------------------------------------------------
# walker: self-compare clean; a perturbed file fails; exclusions honored
# ---------------------------------------------------------------------------


def test_diff_trees_self_compare_clean(tmp_path):
    root = tmp_path / "tree"
    (root / "config").mkdir(parents=True)
    _write_yaml(
        root / "config" / "snake.yml", {"project": {"data_sources": "config/x.yml"}}
    )
    (root / "a.toml").write_text("[s]\nx = 1\n")
    _write_nc(root / "d.nc", [1.0, 2.0, 3.0])
    report = std.diff_trees(str(root), str(root), tol=0.0)
    assert report["passed"], std.format_report(report)
    assert report["n_compared"] == 3


def test_diff_trees_detects_perturbation(tmp_path):
    ref = tmp_path / "ref"
    cur = tmp_path / "cur"
    (ref).mkdir()
    (cur).mkdir()
    _write_nc(ref / "d.nc", [1.0, 2.0, 3.0])
    _write_nc(cur / "d.nc", [1.0, 2.0, 9.0])  # perturbed
    report = std.diff_trees(str(ref), str(cur), tol=1e-9)
    assert not report["passed"]
    assert report["failures"]


def test_diff_trees_excludes_logs_and_benchmarks(tmp_path):
    ref = tmp_path / "ref"
    cur = tmp_path / "cur"
    for root in (ref, cur):
        (root / "logs").mkdir(parents=True)
        (root / "benchmarks").mkdir(parents=True)
        (root / ".snakemake").mkdir(parents=True)
    (ref / "logs" / "a.log").write_text("ref timestamp 1")
    (cur / "logs" / "a.log").write_text("cur timestamp 2")  # differs, but excluded
    (ref / "benchmarks" / "b.txt").write_text("10s")
    (cur / "benchmarks" / "b.txt").write_text("99s")
    report = std.diff_trees(str(ref), str(cur), tol=0.0)
    assert report["passed"], std.format_report(report)
    assert report["n_compared"] == 0


def test_diff_trees_reports_missing_and_extra(tmp_path):
    ref = tmp_path / "ref"
    cur = tmp_path / "cur"
    ref.mkdir()
    cur.mkdir()
    (ref / "only_ref.csv").write_text("a,b\n1,2\n")
    (cur / "only_cur.csv").write_text("a,b\n1,2\n")
    report = std.diff_trees(str(ref), str(cur), tol=0.0)
    assert not report["passed"]
    assert report["missing"] and report["extra"]


# ---------------------------------------------------------------------------
# P3-1: path map + allowlist + path-aware toml comparator (design §6a, commit 5)
# ---------------------------------------------------------------------------

P31_MAP = std.build_p31_path_map("experiment", "era5_20000101_20201231")


def _write_run_toml(
    path,
    path_static,
    path_forcing="../realization_1/x.nc",
    path_input="../../../hydrology_model/instate/instates.nc",
):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "[input]\n"
        f'path_static = "{path_static}"\n'
        f'path_forcing = "{path_forcing}"\n'
        "[state]\n"
        f'path_input = "{path_input}"\n'
        'path_output = "outstates.nc"\n'
        "[csv]\n"
        'path = "output.csv"\n'
    )


def test_toml_path_static_relocation_passes(tmp_path):
    """§6a positive: old-depth vs new-depth path_static both resolve to
    project-relative hydrology_model/staticmaps.nc (no map entry needed)."""
    ref_root = tmp_path / "ref"
    cur_root = tmp_path / "cur"
    ref_toml = ref_root / "hydrology_model" / "run_run_stress_test" / "a.toml"
    cur_toml = cur_root / "experiments" / "experiment" / "model_runs" / "a.toml"
    _write_run_toml(
        ref_toml,
        "../staticmaps.nc",
        path_forcing="../../run_stress_test/realization_1/x.nc",
        path_input="../instate/instates.nc",
    )
    _write_run_toml(cur_toml, "../../../hydrology_model/staticmaps.nc")
    diffs = std.compare_toml(
        str(ref_toml),
        str(cur_toml),
        ref_root=str(ref_root),
        cur_root=str(cur_root),
        path_map=P31_MAP,
    )
    assert diffs == [], diffs


def test_toml_path_forcing_prefix_map_passes(tmp_path):
    """§6a positive: path_forcing target moved with exp_dir; the DIRECTORY-PREFIX
    rule run_stress_test/ -> experiments/experiment/ translates the ref
    target onto the cur one. The target is a temp() file existing in NEITHER
    tree -- asserts the prefix-rewrite form of the map, not a per-file table."""
    ref_root = tmp_path / "ref"
    cur_root = tmp_path / "cur"
    ref_toml = ref_root / "hydrology_model" / "run_run_stress_test" / "a.toml"
    cur_toml = cur_root / "experiments" / "experiment" / "model_runs" / "a.toml"
    _write_run_toml(
        ref_toml,
        "../staticmaps.nc",
        path_forcing="../../run_stress_test/realization_1/inmaps_rlz_1_cst_1.nc",
        path_input="../instate/instates.nc",
    )
    _write_run_toml(
        cur_toml,
        "../../../hydrology_model/staticmaps.nc",
        path_forcing="../realization_1/inmaps_rlz_1_cst_1.nc",
    )
    # the forcing target exists in neither tree (temp()-deleted)
    assert not (ref_root / "run_stress_test").exists()
    assert not (cur_root / "experiments" / "experiment" / "realization_1").exists()
    diffs = std.compare_toml(
        str(ref_toml),
        str(cur_toml),
        ref_root=str(ref_root),
        cur_root=str(cur_root),
        path_map=P31_MAP,
    )
    assert diffs == [], diffs


def test_toml_path_static_mis_repoint_fails(tmp_path):
    """§6a negative: cur path_static resolving to a DIFFERENT project-relative
    target fails, naming the field (mis-repoint caught, not hidden)."""
    ref_root = tmp_path / "ref"
    cur_root = tmp_path / "cur"
    ref_toml = ref_root / "hydrology_model" / "run_run_stress_test" / "a.toml"
    cur_toml = cur_root / "experiments" / "experiment" / "model_runs" / "a.toml"
    _write_run_toml(
        ref_toml,
        "../staticmaps.nc",
        path_forcing="../../run_stress_test/realization_1/x.nc",
        path_input="../instate/instates.nc",
    )
    _write_run_toml(cur_toml, "../../../hydrology_model/staticmaps_WRONG.nc")
    diffs = std.compare_toml(
        str(ref_toml),
        str(cur_toml),
        ref_root=str(ref_root),
        cur_root=str(cur_root),
        path_map=P31_MAP,
    )
    assert diffs and any("path_static" in d and "mis-repoint" in d for d in diffs)


def test_diff_trees_path_map_pairs_moved_files(tmp_path):
    """A pure move (same bytes, mapped path) is content-diffed and CLEAN --
    not MISSING+EXTRA."""
    ref = tmp_path / "ref"
    cur = tmp_path / "cur"
    (ref / "run_stress_test" / "model_results").mkdir(parents=True)
    (cur / "experiments" / "experiment" / "model_results").mkdir(parents=True)
    (ref / "run_stress_test" / "model_results" / "Qstats.csv").write_text("a,b\n1,2\n")
    (cur / "experiments" / "experiment" / "model_results" / "Qstats.csv").write_text(
        "a,b\n1,2\n"
    )
    (ref / "climate_historical" / "raw_data").mkdir(parents=True)
    (cur / "climate_historical" / "era5_20000101_20201231").mkdir(parents=True)
    _write_nc(
        ref / "climate_historical" / "raw_data" / "extract_historical.nc",
        [1.0, 2.0, 3.0],
    )
    _write_nc(
        cur / "climate_historical" / "era5_20000101_20201231" / "extract_historical.nc",
        [1.0, 2.0, 3.0],
    )
    report = std.diff_trees(str(ref), str(cur), tol=0.0, path_map=P31_MAP)
    assert report["passed"], std.format_report(report)
    assert report["n_compared"] == 2


def test_diff_trees_path_map_value_diff_still_fails(tmp_path):
    """The map pairs moved files but does NOT mask a value change (risk-4)."""
    ref = tmp_path / "ref"
    cur = tmp_path / "cur"
    (ref / "run_stress_test" / "model_results").mkdir(parents=True)
    (cur / "experiments" / "experiment" / "model_results").mkdir(parents=True)
    (ref / "run_stress_test" / "model_results" / "Qstats.csv").write_text("a,b\n1,2\n")
    (cur / "experiments" / "experiment" / "model_results" / "Qstats.csv").write_text(
        "a,b\n1,999\n"
    )
    report = std.diff_trees(str(ref), str(cur), tol=0.0, path_map=P31_MAP)
    assert not report["passed"]
    assert report["failures"] and not report["missing"] and not report["extra"]


def test_diff_trees_allowlist_gate_contract(tmp_path):
    """Allowlisted EXTRA entries pass (reported as allowed); an unexplained
    EXTRA fails the gate."""
    allow = std.build_p31_allowlist("experiment", "era5_20000101_20201231")
    ref = tmp_path / "ref"
    cur = tmp_path / "cur"
    ref.mkdir()
    (cur / "experiments" / "experiment").mkdir(parents=True)
    (cur / "climate_historical" / "era5_20000101_20201231").mkdir(parents=True)
    (cur / "experiments" / "experiment" / ".project_consistency_ok").write_text("ok")
    (cur / "climate_historical" / "era5_20000101_20201231" / ".guard_ok").write_text(
        "ok"
    )
    report = std.diff_trees(
        str(ref), str(cur), tol=0.0, path_map=P31_MAP, allowlist=allow
    )
    assert report["passed"], std.format_report(report)
    assert len(report["allowed"]) == 2
    # now an unexplained extra appears -> gate FAILURE
    (cur / "experiments" / "experiment" / "unexplained.csv").write_text("a\n1\n")
    report = std.diff_trees(
        str(ref), str(cur), tol=0.0, path_map=P31_MAP, allowlist=allow
    )
    assert not report["passed"]
    assert report["extra"] == ["experiments/experiment/unexplained.csv"]


def test_diff_trees_self_compare_clean_with_p31_map(tmp_path):
    """Self-diff smoke: a NEW-layout tree diffed against itself with the map
    active is clean (old-layout prefixes match nothing; map is a no-op)."""
    root = tmp_path / "tree"
    (root / "experiments" / "experiment" / "model_results").mkdir(parents=True)
    (root / "experiments" / "experiment" / "model_results" / "Qstats.csv").write_text(
        "a,b\n1,2\n"
    )
    _write_run_toml(
        root / "experiments" / "experiment" / "model_runs" / "a.toml",
        "../../../hydrology_model/staticmaps.nc",
    )
    report = std.diff_trees(
        str(root),
        str(root),
        tol=0.0,
        path_map=P31_MAP,
        allowlist=std.build_p31_allowlist("experiment", "era5_20000101_20201231"),
    )
    assert report["passed"], std.format_report(report)


# ---------------------------------------------------------------------------
# P3-1 commit 5b: cross-root YAML path normalization + run-log file exclusion
# (adjudicated milestone-diff classes; dev/milestones/p31/baseline_diffs.md)
# ---------------------------------------------------------------------------


def test_yaml_cross_root_path_leaves_pass(tmp_path):
    """A yml whose string leaves differ ONLY by each tree's own root token +
    the old->new layout move is behavior-neutral under the cross-root
    normalization: the root becomes <PROJECT_ROOT> on both sides and the ref
    remainder goes through the path map. Covers the weathergen-config and
    project_dir-snapshot classes from the milestone diff."""
    ref_root = tmp_path / "ref"
    cur_root = tmp_path / "cur"
    ref_yml = ref_root / "run_stress_test" / "weathergen_config.yml"
    cur_yml = cur_root / "experiments" / "experiment" / "weathergen_config.yml"
    ref_yml.parent.mkdir(parents=True)
    cur_yml.parent.mkdir(parents=True)
    _write_yaml(
        ref_yml,
        {
            "project_dir": ref_root.as_posix(),
            "output": {"path": f"{ref_root.as_posix()}/run_stress_test/realization_1/"},
            "seed": 123,
        },
    )
    _write_yaml(
        cur_yml,
        {
            "project_dir": cur_root.as_posix(),
            "output": {
                "path": f"{cur_root.as_posix()}/experiments/experiment/realization_1/"
            },
            "seed": 123,
        },
    )
    diffs = std.compare_yaml(
        str(ref_yml),
        str(cur_yml),
        cur_yml.relative_to(cur_root),
        ref_root=str(ref_root),
        cur_root=str(cur_root),
        path_map=P31_MAP,
    )
    assert diffs == [], diffs


def test_yaml_cross_root_nonpath_value_still_fails(tmp_path):
    """The normalization is path-leaf-only: a non-path value drift (the seed)
    under the SAME root/layout moves is still a real FAIL."""
    ref_root = tmp_path / "ref"
    cur_root = tmp_path / "cur"
    ref_yml = ref_root / "run_stress_test" / "weathergen_config.yml"
    cur_yml = cur_root / "experiments" / "experiment" / "weathergen_config.yml"
    ref_yml.parent.mkdir(parents=True)
    cur_yml.parent.mkdir(parents=True)
    _write_yaml(
        ref_yml,
        {
            "output": {"path": f"{ref_root.as_posix()}/run_stress_test/realization_1/"},
            "seed": 123,
        },
    )
    _write_yaml(
        cur_yml,
        {
            "output": {
                "path": f"{cur_root.as_posix()}/experiments/experiment/realization_1/"
            },
            "seed": 456,  # drift
        },
    )
    diffs = std.compare_yaml(
        str(ref_yml),
        str(cur_yml),
        cur_yml.relative_to(cur_root),
        ref_root=str(ref_root),
        cur_root=str(cur_root),
        path_map=P31_MAP,
    )
    assert diffs and any("seed" in d for d in diffs)


def test_yaml_backslash_absolute_uri_normalizes(tmp_path):
    """The data-catalog class: absolute backslashed uris under each root
    normalize to the same <PROJECT_ROOT>-relative target."""
    ref_root = tmp_path / "ref"
    cur_root = tmp_path / "cur"
    ref_yml = ref_root / "run_stress_test" / "cat.yml"
    cur_yml = cur_root / "experiments" / "experiment" / "cat.yml"
    ref_yml.parent.mkdir(parents=True)
    cur_yml.parent.mkdir(parents=True)
    ref_abs = str(
        ref_root.resolve() / "run_stress_test" / "realization_1" / "x.nc"
    ).replace("/", "\\")
    cur_abs = str(
        cur_root.resolve() / "experiments" / "experiment" / "realization_1" / "x.nc"
    ).replace("/", "\\")
    _write_yaml(ref_yml, {"rlz": {"uri": ref_abs}})
    _write_yaml(cur_yml, {"rlz": {"uri": cur_abs}})
    diffs = std.compare_yaml(
        str(ref_yml),
        str(cur_yml),
        cur_yml.relative_to(cur_root),
        ref_root=str(ref_root),
        cur_root=str(cur_root),
        path_map=P31_MAP,
    )
    assert diffs == [], diffs


def test_diff_trees_excludes_run_log_files(tmp_path):
    """Run-log FILES outside logs/ dirs (hydromt.log, model_runs/log.txt) are
    excluded from the walk -- same non-content-bearing class as logs/ dirs."""
    ref = tmp_path / "ref"
    cur = tmp_path / "cur"
    for root in (ref, cur):
        (root / "hydrology_model" / "run_default").mkdir(parents=True)
    (ref / "hydrology_model" / "hydromt.log").write_text("ts 1")
    (cur / "hydrology_model" / "hydromt.log").write_text("ts 2")
    (ref / "hydrology_model" / "run_default" / "log.txt").write_text("ts 1")
    (cur / "hydrology_model" / "run_default" / "log.txt").write_text("ts 2")
    report = std.diff_trees(str(ref), str(cur), tol=0.0)
    assert report["passed"], std.format_report(report)
    assert report["n_compared"] == 0


# ---------------------------------------------------------------------------
# R07: the declared many-to-one merge class (design B1 / migration map 2e).
#
# B1 collapses two climate stores into one. `diff_trees` keys the reference
# tree by MAPPED relpath, so a prefix rule expressing that collapse makes two
# reference files land on one key and raises -- the gate aborts before it can
# report. The merge class is the fix, and its contract is that the survivor
# must match EVERY collapsed source: allowlisting one side as MISSING was
# rejected because it lets the gate go green while proving nothing about the
# store that disappeared.
# ---------------------------------------------------------------------------


def _merge_trees(tmp_path, wf1_data, key_data, survivor_data):
    """Two ref stores (wf1_raw + <key>) collapsing to one current store."""
    ref = tmp_path / "ref"
    cur = tmp_path / "cur"
    (ref / "climate_historical" / "wf1_raw").mkdir(parents=True)
    (ref / "climate_historical" / "k1").mkdir(parents=True)
    (cur / "climate_historical" / "k1").mkdir(parents=True)
    _write_nc(
        ref / "climate_historical" / "wf1_raw" / "extract_historical.nc", wf1_data
    )
    _write_nc(ref / "climate_historical" / "k1" / "extract_historical.nc", key_data)
    _write_nc(
        cur / "climate_historical" / "k1" / "extract_historical.nc", survivor_data
    )
    return ref, cur


_MERGE = [
    (
        "climate_historical/k1/extract_historical.nc",
        [
            "climate_historical/wf1_raw/extract_historical.nc",
            "climate_historical/k1/extract_historical.nc",
        ],
    )
]


def test_merge_passes_when_survivor_matches_every_source(tmp_path):
    ref, cur = _merge_trees(tmp_path, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    rep = std.diff_trees(str(ref), str(cur), tol=0.0, merges=_MERGE)
    assert rep["passed"], std.format_report(rep)
    # Both sides are compared, not just the one that happens to share a path.
    assert len(rep["merged"]) == 2
    assert rep["missing"] == [] and rep["extra"] == []


def test_merge_fails_when_only_one_source_matches(tmp_path):
    """The property arch-2 / risk-2 were about: a merge is proven by BOTH
    comparisons. The survivor here is bit-identical to the <key> store but
    differs from wf1_raw -- a one-sided check would call this clean."""
    ref, cur = _merge_trees(tmp_path, [9.0, 9.0, 9.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    rep = std.diff_trees(str(ref), str(cur), tol=0.0, merges=_MERGE)
    assert not rep["passed"], std.format_report(rep)
    labels = [lbl for lbl, _ in rep["failures"]]
    assert any("wf1_raw" in lbl for lbl in labels), labels
    # ...and the side that DID match is still reported OK, so a reader can
    # tell which half of the collapse moved (the 2e asymmetry read).
    assert any("k1/extract_historical.nc" in m for m in rep["merged"])


def test_merge_sources_are_not_reported_missing(tmp_path):
    """Without the merge class, wf1_raw/* reads as MISSING and the survivor as
    EXTRA -- the failure mode the stale migration map papered over."""
    ref, cur = _merge_trees(tmp_path, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    bare = std.diff_trees(str(ref), str(cur), tol=0.0)
    assert any("wf1_raw" in m for m in bare["missing"])
    merged = std.diff_trees(str(ref), str(cur), tol=0.0, merges=_MERGE)
    assert merged["missing"] == [] and merged["extra"] == []


def test_merge_survivor_absent_fails(tmp_path):
    ref, cur = _merge_trees(tmp_path, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    (cur / "climate_historical" / "k1" / "extract_historical.nc").unlink()
    rep = std.diff_trees(str(ref), str(cur), tol=0.0, merges=_MERGE)
    assert not rep["passed"]
    assert any("survivor missing" in r for _, rs in rep["failures"] for r in rs)


def test_merge_declared_source_absent_fails(tmp_path):
    """A merge declared against a source that is not in the reference tree is
    a mis-declaration, not a silent pass."""
    ref, cur = _merge_trees(tmp_path, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    (ref / "climate_historical" / "wf1_raw" / "extract_historical.nc").unlink()
    rep = std.diff_trees(str(ref), str(cur), tol=0.0, merges=_MERGE)
    assert not rep["passed"]
    assert any(
        "declared merge source missing" in r for _, rs in rep["failures"] for r in rs
    )


def test_path_map_collision_still_raises_and_names_the_merge_fix(tmp_path):
    """The guard stays: an UNdeclared many-to-one is still a hard error, and
    the message now names the escape hatch."""
    ref, cur = _merge_trees(tmp_path, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    collide = [("climate_historical/wf1_raw/", "climate_historical/k1/")]
    with pytest.raises(ValueError, match="path map collision"):
        std.diff_trees(str(ref), str(cur), tol=0.0, path_map=collide)
    with pytest.raises(ValueError, match="--merge "):
        std.diff_trees(str(ref), str(cur), tol=0.0, path_map=collide)


# ---------------------------------------------------------------------------
# R07 commit 8: .geojson is compared by GEOMETRY, not by bytes.
#
# `.geojson` fell through to compare_hashed, which is byte-exact. Regenerating
# an identical model re-serializes the vectors with different coordinate
# formatting, so the byte hash reported a difference where the geometry was
# provably the same -- it only ever passed because the reference tree and the
# current tree were the same never-regenerated files.
# ---------------------------------------------------------------------------


def _write_geojson(path, coords, value=1):
    import geopandas as gpd
    from shapely.geometry import Polygon

    gpd.GeoDataFrame(
        {"value": [value]}, geometry=[Polygon(coords)], crs="EPSG:4326"
    ).to_file(path, driver="GeoJSON")


_SQUARE = [(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]


def test_geojson_reserialization_passes(tmp_path):
    """Same shape, different serialization -> PASS (the commit-8 case)."""
    a, b = tmp_path / "a.geojson", tmp_path / "b.geojson"
    _write_geojson(a, _SQUARE)
    # same polygon, different starting vertex + closing repeat: byte-different,
    # topologically equal -- exactly what a model rebuild produces.
    _write_geojson(b, [(1, 1), (1, 0), (0, 0), (0, 1), (1, 1)])
    assert a.read_bytes() != b.read_bytes(), "fixture must differ in bytes"
    assert std.compare_geojson(str(a), str(b)) == []


def test_geojson_real_shape_change_fails_with_magnitude(tmp_path):
    """A genuine geometry change still fails -- and reports how much."""
    a, b = tmp_path / "a.geojson", tmp_path / "b.geojson"
    _write_geojson(a, _SQUARE)
    _write_geojson(b, [(0, 0), (0, 2), (1, 2), (1, 0), (0, 0)])
    reasons = std.compare_geojson(str(a), str(b))
    assert reasons and "geometry differs" in reasons[0]
    assert "symmetric difference area 1" in reasons[0], reasons


def test_geojson_attribute_change_fails(tmp_path):
    """Geometry alone is not the contract -- attributes are compared too."""
    a, b = tmp_path / "a.geojson", tmp_path / "b.geojson"
    _write_geojson(a, _SQUARE, value=1)
    _write_geojson(b, _SQUARE, value=99)
    reasons = std.compare_geojson(str(a), str(b))
    assert any("value" in r for r in reasons), reasons


# ---------------------------------------------------------------------------
# R09 phase 1: the shared applier gains a fall-through signal.
#
# `apply_path_map` returns its input unchanged both when an identity rule fires
# and when nothing matches, which makes "the map covers every artifact"
# inexpressible. `apply_path_map_matched` reports whether a rule actually
# fired; `apply_path_map` is now a projection of it, so the two cannot drift.
# Every pre-existing call site must behave exactly as before.
# ---------------------------------------------------------------------------


def test_apply_path_map_matched_reports_fall_through():
    m = [("a/b.txt", "c/d.txt")]
    assert std.apply_path_map_matched("a/b.txt", m) == ("c/d.txt", True)
    assert std.apply_path_map_matched("x/y.txt", m) == ("x/y.txt", False)


def test_apply_path_map_matched_distinguishes_identity_from_fall_through():
    """The discriminator the R09 falsifier is built on: same string, different
    verdict."""
    ident = [("keep/me.txt", "keep/me.txt")]
    assert std.apply_path_map_matched("keep/me.txt", ident) == ("keep/me.txt", True)
    assert std.apply_path_map_matched("keep/other.txt", ident) == (
        "keep/other.txt",
        False,
    )


def test_apply_path_map_matched_never_reports_a_match_without_rules():
    """An empty or absent map must report EVERY path as unmatched.

    If it did not, a map with no rules would green the falsifier
    unconditionally -- the exact false pass the reporting parameter exists to
    prevent."""
    for empty in (None, []):
        assert std.apply_path_map_matched("anything/at/all.nc", empty) == (
            "anything/at/all.nc",
            False,
        )


def test_apply_path_map_matched_normalizes_backslashes_like_the_original():
    m = [("a/b.txt", "c/d.txt")]
    assert std.apply_path_map_matched("a\\b.txt", m) == ("c/d.txt", True)
    # unmatched paths come back normalized too, as apply_path_map has always
    # returned them
    assert std.apply_path_map_matched("x\\y.txt", m) == ("x/y.txt", False)
    assert std.apply_path_map("x\\y.txt", m) == "x/y.txt"


def test_apply_path_map_is_the_projection_of_the_reporting_sibling():
    """Pins the delegation: no second matching pass that could drift.

    Driven off the post-migration INVENTORY since 2026-08-11, when the R07 map
    this used was retired (`dev/reviews/2026-08-11_test-suite-bloat-assessment.md`
    §6a). Any non-trivial map serves — what the case needs is a mix of rules
    that fire and a path that falls through, which the last entry supplies.
    """
    m = std.build_project_tree_rules("experiment", "era5_20000101_20201231")
    for rel in (
        "models/hydrology/wflow/staticmaps.nc",
        "data/climate/projections/cmip6/series/x.nc",
        "logs/wf1_build_model.log",
        "nothing/matches/this.txt",
    ):
        assert std.apply_path_map(rel, m) == std.apply_path_map_matched(rel, m)[0]


def test_geojson_dispatches_by_suffix(tmp_path):
    """dispatch() routes .geojson to the semantic comparator, not the hash."""
    from pathlib import Path as _P

    a, b = tmp_path / "a.geojson", tmp_path / "b.geojson"
    _write_geojson(a, _SQUARE)
    _write_geojson(b, [(1, 1), (1, 0), (0, 0), (0, 1), (1, 1)])
    assert std.dispatch(_P("staticgeoms/a.geojson"), str(a), str(b), 0.0) == []
