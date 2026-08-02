"""Lightweight helpers for deterministic collection operations."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

_T = TypeVar("_T")


def intersection(left: Iterable[_T], right: Iterable[_T]) -> list[_T]:
    """Return shared members once each in deterministic sorted order."""
    return sorted(set(left).intersection(right))
