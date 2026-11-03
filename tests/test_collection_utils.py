"""Tests for lightweight deterministic collection helpers."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from blueearth_cst.shared.collection_utils import intersection


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (["precip", "temp"], ["precip", "temp"]),
        (["temp", "precip"], ["temp", "precip"]),
        (["temp", "precip"], ["precip", "temp"]),
        (["precip", "temp"], ["temp", "precip"]),
    ],
)
def test_intersection_order_is_input_order_independent(left, right):
    assert intersection(left, right) == ["precip", "temp"]


def test_intersection_is_sorted_and_deduplicated():
    result = intersection(["temp", "precip", "temp", "kin"], ["kin", "precip", "temp"])

    assert result == ["kin", "precip", "temp"]
    assert len(result) == len(set(result))


def test_intersection_drops_non_shared_members():
    assert intersection(["precip", "temp", "pet"], ["precip", "temp"]) == [
        "precip",
        "temp",
    ]


@pytest.mark.process_isolation
def test_intersection_stable_across_hash_seeds():
    code = (
        "from blueearth_cst.shared.collection_utils import intersection;"
        "print(intersection(['temp','precip','kin'], ['kin','temp','precip']))"
    )
    outputs = []
    for seed in ("0", "1", "42", "random"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        outputs.append(result.stdout.strip())

    assert outputs == ["['kin', 'precip', 'temp']"] * 4
