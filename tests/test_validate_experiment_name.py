"""Unit tests for blueearth_cst.shared.snake_utils.validate_experiment_name.

The §2b test matrix (dev/milestones/p31/experiment-structure-design.md): the validator is
called once at Snakefile_climate_experiment parse time, before exp_dir is built,
so a malformed or adversarial experiment_name can never introduce a path
component, escape the experiments/ dir, or collide via normalization. Uppercase
is REJECTED (never silently lowercased).
"""

from __future__ import annotations

import pytest

from blueearth_cst.shared.snake_utils import validate_experiment_name  # noqa: E402

PROJECT_DIR = "/tmp/proj"  # value irrelevant; only its "experiments/" child matters


# --- Accepted -----------------------------------------------------------------


def test_valid_lowercase_snake_returns_unchanged():
    assert validate_experiment_name("experiment_a", PROJECT_DIR) == "experiment_a"


@pytest.mark.parametrize("name", ["experiment", "exp1", "a", "0", "exp_1_baseline"])
def test_various_valid_names_pass(name):
    assert validate_experiment_name(name, PROJECT_DIR) == name


# --- Rejected: grammar --------------------------------------------------------


def test_uppercase_rejected_not_lowercased():
    # §2b/ext2-3: uppercase is a grammar violation, NOT silently normalized.
    with pytest.raises(ValueError, match="grammar"):
        validate_experiment_name("experiment_A", PROJECT_DIR)


def test_empty_rejected():
    with pytest.raises(ValueError):
        validate_experiment_name("", PROJECT_DIR)


def test_whitespace_only_rejected():
    with pytest.raises(ValueError):
        validate_experiment_name("   ", PROJECT_DIR)


def test_leading_underscore_rejected():
    # Grammar requires a leading alphanumeric.
    with pytest.raises(ValueError, match="grammar"):
        validate_experiment_name("_exp", PROJECT_DIR)


@pytest.mark.parametrize("name", ["Exp-1", "exp-1"])
def test_hyphen_rejected(name):
    with pytest.raises(ValueError):
        validate_experiment_name(name, PROJECT_DIR)


def test_dot_rejected():
    with pytest.raises(ValueError):
        validate_experiment_name("exp.1", PROJECT_DIR)


# --- Rejected: traversal / separators -----------------------------------------


def test_dotdot_rejected():
    with pytest.raises(ValueError):
        validate_experiment_name("..", PROJECT_DIR)


def test_single_dot_rejected():
    with pytest.raises(ValueError):
        validate_experiment_name(".", PROJECT_DIR)


@pytest.mark.parametrize("name", ["a/b", "a\\b", "../evil", "..\\evil"])
def test_path_separators_rejected(name):
    with pytest.raises(ValueError):
        validate_experiment_name(name, PROJECT_DIR)


# --- Rejected: absolute forms -------------------------------------------------


@pytest.mark.parametrize("name", ["/abs", "\\abs", "C:\\x", "c:/x"])
def test_absolute_forms_rejected(name):
    with pytest.raises(ValueError):
        validate_experiment_name(name, PROJECT_DIR)


# --- Rejected: Windows-reserved device names (case- and extension-insensitive)-


@pytest.mark.parametrize("name", ["con", "CON", "prn", "aux", "nul", "com1", "lpt9"])
def test_windows_reserved_names_rejected(name):
    with pytest.raises(ValueError, match="reserved"):
        validate_experiment_name(name, PROJECT_DIR)


def test_windows_reserved_with_extension_rejected():
    # "nul.txt" -> stem "nul" is reserved (extension-insensitive).
    with pytest.raises(ValueError, match="reserved"):
        validate_experiment_name("nul.txt", PROJECT_DIR)


# --- Rejected: length + trailing space ----------------------------------------


def test_length_cap_rejected():
    with pytest.raises(ValueError, match="limit"):
        validate_experiment_name("a" * 65, PROJECT_DIR)


def test_length_cap_boundary_ok():
    name = "a" * 64
    assert validate_experiment_name(name, PROJECT_DIR) == name


def test_trailing_space_rejected():
    # A trailing space is not in the grammar; Windows would strip it (collision).
    with pytest.raises(ValueError):
        validate_experiment_name("exp ", PROJECT_DIR)


# --- Containment assertion is independent of the grammar ----------------------


def test_containment_direct_child_holds_for_valid_name(tmp_path):
    # A grammar-valid name resolves to a direct child of <project_dir>/experiments.
    name = "myexp"
    assert validate_experiment_name(name, str(tmp_path)) == name
