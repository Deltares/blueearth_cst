"""Reuse safe YAML parsing without caching file reads or mutable documents."""

from copy import deepcopy
from functools import lru_cache
from typing import Any

import yaml


@lru_cache(maxsize=32)
def _parse(data: bytes) -> Any:
    return yaml.safe_load(data)


def parse_yaml_document(data: bytes) -> Any:
    """Parse exact bytes and return an independent document on every call.

    Callers still read and verify current source bytes. Paths, timestamps and
    resolved catalog metadata are deliberately absent from this bounded cache.
    """
    return deepcopy(_parse(data))
