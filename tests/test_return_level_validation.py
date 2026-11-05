"""Shipped GF15 report loader and the closed declaration it backs (D4, D5).

The discriminating checks here are the portability one -- the loader must work
from a relocated copy with the original checkout unreachable -- and the tamper
ones, which must fail rather than silently downgrade to an unassessed status.
"""

import ast
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from blueearth_cst.experiment import return_level_validation as rlv

REPO = Path(__file__).resolve().parents[1]
ASSET = REPO / "blueearth_cst/experiment/data" / rlv.REPORT_RESOURCE


def test_shipped_asset_is_present_and_tracked():
    """A missing asset must fail the suite, never skip it."""
    assert ASSET.is_file(), f"shipped report asset is absent: {ASSET}"
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", ASSET.relative_to(REPO).as_posix()],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert tracked.returncode == 0, "the shipped report asset is not tracked"


def test_loader_reads_the_packaged_bytes():
    data = rlv.load_report_bytes()
    assert data == ASSET.read_bytes()
    assert rlv.report_digest(data) == hashlib.sha256(data).hexdigest()


def test_report_decodes_with_the_accepted_identity():
    report = rlv.decode_report(rlv.load_report_bytes())
    assert report["schema_version"] == "gf15-benchmark-report/1"
    assert report["estimator_id"] == "gf15-lmoments-c/1"
    assert report["policy_id"] == "gf15-accuracy-8x-v1"
    assert report["verdict"]["scientific_gate_conjunction"] is True
    assert report["verdict"]["new_fits"] == 0
    # The original and threefold failures must still be visible in the report.
    counts = report["counts"]["C"]
    assert counts["original"]["all_individual_gates_passed"] is False
    assert counts["accuracy_3x"]["all_individual_gates_passed"] is False
    assert counts["accuracy_8x"]["all_individual_gates_passed"] is True


def test_declaration_is_the_closed_accepted_record():
    declaration = rlv.build_declaration()
    assert set(declaration) == rlv.DECLARATION_FIELDS
    assert declaration["schema_version"] == "return-level-validation/1"
    assert declaration["policy_id"] == "gf15-accuracy-8x-v1"
    assert declaration["screening_ratio"] == 1.0
    assert declaration["screening_floor"] == 10
    assert declaration["screening_policy_status"] == "provisional_operational"
    assert declaration["screening_policy_validated"] is False
    assert declaration["benchmark_status"] == "reviewed_bounded"
    assert declaration["application_scope"] == "operational_screen_only"
    assert declaration["relative_error_near_zero"] == "unvalidated"
    assert declaration["actual_bundle_applicability"] == "unestablished"
    assert declaration["evidence_use"] == "post_results_development_rescore"
    assert declaration["fit_uncertainty"] == "point_estimates_only"
    assert declaration["benchmark_record"] == {
        "path": "return_level_benchmark.json",
        "sha256": rlv.report_digest(rlv.load_report_bytes()),
    }


def test_declared_domain_is_the_accepted_domain():
    domain = rlv.build_declaration()["tested_domain"]
    assert domain == {
        "shapes_c": [-0.2, 0.0, 0.2],
        "counts": [10, 18, 30, 60],
        "probabilities": [0.9, 0.5],
        "ratios": [0.0, 0.05, 0.5, 2.0, 10.0],
        "relative_qualified_ratios": [0.5, 2.0, 10.0],
        "draws_per_base_cell": 1000,
    }


def test_declared_criteria_carry_the_eightfold_limits():
    criteria = rlv.build_declaration()["criteria"]
    assert criteria["policy_id"] == "gf15-accuracy-8x-v1"
    assert criteria["valid_rate_min"] == 0.95
    assert criteria["median_scale_abs_max"] == 0.80
    assert criteria["p90_abs_scale_max"] == {"0.9": 4.00, "0.5": 2.00}
    assert criteria["median_relative_abs_max"] == 0.80
    assert criteria["p90_abs_relative_max"] == {"0.9": 4.00, "0.5": 2.00}
    assert criteria["translation_paired_abs_max"] == 1e-06
    assert criteria["original_criteria_sha256"]


def test_legacy_dispatch_value_is_exact():
    assert rlv.is_legacy(
        {"status": "provisional_operational", "benchmark": "not assessed"}
    )
    assert not rlv.is_legacy({"status": "provisional_operational"})
    assert not rlv.is_legacy(dict(rlv.LEGACY_VALIDATION, extra=1))
    assert not rlv.is_legacy(rlv.build_declaration())


def test_verify_declaration_accepts_its_own_report():
    data = rlv.load_report_bytes()
    report = rlv.verify_declaration(rlv.build_declaration(data), data)
    assert report["policy_id"] == "gf15-accuracy-8x-v1"


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: dict(d, schema_version="return-level-validation/2"), "unknown"),
        (lambda d: {k: v for k, v in d.items() if k != "fit_uncertainty"}, "closed"),
        (lambda d: dict(d, unexpected=True), "closed"),
        (lambda d: dict(d, screening_policy_validated=True), "differs"),
        (lambda d: dict(d, benchmark_status="validated"), "differs"),
        (
            lambda d: dict(d, benchmark_record={"path": "other.json", "sha256": "x"}),
            "return_level_benchmark.json",
        ),
    ],
)
def test_altered_declarations_are_refused(mutate, message):
    data = rlv.load_report_bytes()
    with pytest.raises(rlv.ReturnLevelValidationError, match=message):
        rlv.verify_declaration(mutate(rlv.build_declaration(data)), data)


def test_declaration_is_refused_against_different_report_bytes(tmp_path):
    data = rlv.load_report_bytes()
    declaration = rlv.build_declaration(data)
    altered = json.loads(data.decode("utf-8"))
    altered["criteria"] = dict(altered["criteria"], median_scale_abs_max=9.0)
    with pytest.raises(rlv.ReturnLevelValidationError, match="digest"):
        rlv.verify_declaration(declaration, json.dumps(altered).encode("utf-8"))


def test_tampered_report_criteria_are_caught_by_their_embedded_source():
    """Changing the packaged criteria must not silently redefine the policy."""
    report = json.loads(rlv.load_report_bytes().decode("utf-8"))
    report["criteria"] = dict(report["criteria"], median_scale_abs_max=9.0)
    with pytest.raises(rlv.ReturnLevelValidationError, match="embedded source"):
        rlv.decode_report(json.dumps(report).encode("utf-8"))


def test_tampered_embedded_text_is_caught_by_its_digest():
    report = json.loads(rlv.load_report_bytes().decode("utf-8"))
    report["embedded_sources"][0] = dict(report["embedded_sources"][0], text="tampered")
    with pytest.raises(
        rlv.ReturnLevelValidationError, match="does not match its digest"
    ):
        rlv.decode_report(json.dumps(report).encode("utf-8"))


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda r: dict(r, schema_version="gf15-benchmark-report/2"), "unknown"),
        (lambda r: dict(r, estimator_id="other/1"), "accepted estimator"),
        (lambda r: dict(r, policy_id="gf15-accuracy-3x-v1"), "accepted estimator"),
        (lambda r: {k: v for k, v in r.items() if k != "verdict"}, "missing"),
        (lambda r: dict(r, embedded_sources=[]), "embeds no sources"),
    ],
)
def test_unknown_or_incomplete_reports_are_refused(mutate, message):
    report = json.loads(rlv.load_report_bytes().decode("utf-8"))
    with pytest.raises(rlv.ReturnLevelValidationError, match=message):
        rlv.decode_report(json.dumps(mutate(report)).encode("utf-8"))


def test_malformed_report_bytes_are_refused():
    with pytest.raises(rlv.ReturnLevelValidationError, match="valid UTF-8 JSON"):
        rlv.decode_report(b"{not json")
    with pytest.raises(rlv.ReturnLevelValidationError, match="not a JSON object"):
        rlv.decode_report(b"[]\n")


def _relocated_tree(root: Path) -> Path:
    """Copy the importable package and its asset, and nothing else.

    D5 asks for "a relocated copy containing the required package source and
    asset". The whole package is copied because the loader legitimately imports
    a sibling module; what is deliberately NOT copied is everything outside it --
    `dev/` evidence, `tests/`, session scratch and the repository root.
    """
    package = root / "blueearth_cst"
    shutil.copytree(
        REPO / "blueearth_cst",
        package,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    assert (package / "experiment/data" / rlv.REPORT_RESOURCE).is_file()
    return root


def _run_relocated(root: Path, body: str):
    """Run a probe with ONLY the relocated tree importable.

    `-S` and an emptied PYTHONPATH keep the original checkout, `dev/` evidence
    and session scratch out of the loader's reach, so a hidden dependency on any
    of them surfaces as an import or lookup failure rather than passing here and
    failing in a deployed copy.
    """
    return subprocess.run(
        [sys.executable, "-B", "-S", "-c", body],
        cwd=root,
        capture_output=True,
        text=True,
        env={
            "PATH": "",
            "SYSTEMROOT": "",
            "PYTHONPATH": str(root),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )


def test_loader_works_from_a_relocated_copy(tmp_path):
    root = _relocated_tree(tmp_path / "relocated")
    result = _run_relocated(
        root,
        "import json, site; site.main();"
        "from blueearth_cst.experiment import return_level_validation as r;"
        "d = r.load_report_bytes();"
        "print(json.dumps({'digest': r.report_digest(d),"
        " 'policy': r.build_declaration(d)['policy_id'],"
        " 'file': r.__file__}))",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["digest"] == rlv.report_digest(rlv.load_report_bytes())
    assert payload["policy"] == "gf15-accuracy-8x-v1"
    assert (
        Path(payload["file"]).resolve()
        == (root / "blueearth_cst/experiment/return_level_validation.py").resolve()
    ), "the relocated probe imported a different copy"


def test_relocated_copy_without_the_asset_fails(tmp_path):
    """An absent asset must fail, never degrade to an unassessed declaration."""
    root = _relocated_tree(tmp_path / "relocated")
    (root / "blueearth_cst/experiment/data" / rlv.REPORT_RESOURCE).unlink()
    result = _run_relocated(
        root,
        "import site; site.main();"
        "from blueearth_cst.experiment import return_level_validation as r;"
        "\ntry:\n r.load_report_bytes()\n print('LOADED')\n"
        "except r.ReturnLevelValidationError as e:\n print('REFUSED')",
    )
    assert result.returncode == 0, result.stderr
    assert "REFUSED" in result.stdout


def test_relocated_copy_with_a_corrupt_asset_fails(tmp_path):
    root = _relocated_tree(tmp_path / "relocated")
    (root / "blueearth_cst/experiment/data" / rlv.REPORT_RESOURCE).write_bytes(b"{}\n")
    result = _run_relocated(
        root,
        "import site; site.main();"
        "from blueearth_cst.experiment import return_level_validation as r;"
        "\ntry:\n r.build_declaration()\n print('BUILT')\n"
        "except r.ReturnLevelValidationError as e:\n print('REFUSED')",
    )
    assert result.returncode == 0, result.stderr
    assert "REFUSED" in result.stdout


def _imported_modules(path: Path) -> set[str]:
    """Top-level module names a source file imports, from its syntax tree."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_loader_does_not_require_the_estimator_dependency():
    """D6: reading retained evidence must not need lmoments3 installed.

    Checked against the import graph rather than the file text, so a mention in
    a docstring neither passes nor fails it.
    """
    imported = _imported_modules(
        REPO / "blueearth_cst/experiment/return_level_validation.py"
    )
    assert "lmoments3" not in imported
    assert "scipy" not in imported
    assert "numpy" not in imported


def test_loader_imports_without_the_estimator_dependency(tmp_path):
    """The stronger form: import and declare with lmoments3 unimportable."""
    root = _relocated_tree(tmp_path / "relocated")
    blocker = root / "sitecustomize.py"
    blocker.write_text(
        '''import sys


class _Block:
    """Make lmoments3 unimportable, however it is reached."""

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "lmoments3":
            raise ImportError("lmoments3 is unavailable in this probe")
        return None


sys.meta_path.insert(0, _Block())
''',
        encoding="utf-8",
    )
    result = _run_relocated(
        root,
        "import site; site.main(); import sitecustomize;"
        "from blueearth_cst.experiment import return_level_validation as r;"
        "print(r.build_declaration()['policy_id'])",
    )
    assert result.returncode == 0, result.stderr
    assert "gf15-accuracy-8x-v1" in result.stdout
