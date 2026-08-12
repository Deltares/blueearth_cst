"""Pointer-derived fingerprint of the live Wflow model's runtime inputs.

Implements the *Model reproducibility contract* in
``dev/milestones/r09/project-tree-design.md``: each experiment records which
model state it used, so a changed live model cannot silently re-run an old
experiment against different physics or state.

**Pointer-derived, not a fixed file list.** The digest covers ``wflow_sbm.toml``
plus every model-root file the TOML *points at*, discovered by walking the
parsed document for path-valued keys rather than by naming them. A fixed triple
of TOML + ``staticmaps.nc`` + ``instates.nc`` was rejected in design: it is
correct only for the TOML shape the toolbox happens to emit today. Any hydromt
``setup_*`` that writes a TOML-referenced side file — lake rating curves,
glacier tables — adds a runtime input. A fixed digest would catch the POINTER
changing, because the TOML is hashed, but not a later in-place edit of the file
pointed at. Discovery closes that class; enumeration only lists its current
members.

**The exclusions are structural, not a blocklist.** ``staticgeoms/``,
``hydromt.log`` and ``hydromt_data.yml`` are excluded because Wflow.jl does not
read them at run time — and they are excluded *by construction*, since nothing
in the TOML points at them. That is deliberate: if some future TOML did point at
one, Wflow would read it, it would be a runtime input, and it SHOULD enter the
digest. A hardcoded blocklist would then be wrong in the one case that matters.

**Resolution is lexical, against the model root.** ``normpath``/``join``, never
``resolve()``, so a pointer to a not-yet-created output does not depend on the
filesystem. ``dir_output`` is deliberately NOT applied: Wflow resolves output
pointers against ``dirname(toml) + dir_output``, so output keys such as
``state.path_output`` and ``output.csv.path`` resolve here to paths that
normally do not exist and are recorded with the absence marker. The consequence
is intended — the fingerprint stays stable across runs of the model, instead of
changing every time the historical run rewrites its own outputs.

**Determinism across platforms** is a hard requirement: entries are sorted by
relative POSIX path, and no absolute path ever enters the hashed material.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

try:  # tomllib is stdlib >=3.11; the pixi env is 3.12
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for <3.11
    import tomli as tomllib  # type: ignore

#: The model's run configuration, and the root of the pointer graph.
MODEL_TOML_NAME = "wflow_sbm.toml"

#: Recorded for a path-valued key whose target does not exist. An explicit
#: marker, never omission: "the optional warm state is absent" and "there is no
#: such key" are different model states and must give different digests.
ABSENT = "<absent>"

#: Digest format identifier. Recorded in `model_reference.yml` so a future
#: change to what is hashed is detectable rather than silently incomparable.
DIGEST_VERSION = 1


def _is_path_key(key: str) -> bool:
    """Whether a TOML key names a path.

    Wflow's convention: ``path``, or a ``path_*`` compound (``path_static``,
    ``path_forcing``, ``path_input``, ``path_output``). Matching the KEY rather
    than sniffing the value is what makes discovery safe — a string value that
    merely looks path-like (a variable name, a CSDMS Standard Name) is not
    dragged in.

    ``dir_*`` keys are excluded: they name directories, not files, and
    ``dir_output`` in particular is a resolution modifier rather than an input.
    """
    return key == "path" or key.startswith("path_")


def _walk_path_values(node, trail=()):
    """Yield ``(dotted_key, value)`` for every path-valued string leaf."""
    if isinstance(node, dict):
        for key, value in node.items():
            here = trail + (str(key),)
            if isinstance(value, str):
                if _is_path_key(str(key)):
                    yield ".".join(here), value
            else:
                yield from _walk_path_values(value, here)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_path_values(item, trail)


def _relative_to_root(raw: str, model_root: Path) -> str:
    """Resolve one pointer LEXICALLY against the model root.

    Returns the target's root-relative POSIX path.

    Raises
    ------
    ValueError
        If the pointer escapes the model root. A pointer that resolves outside
        is an error, never a silently widened digest: the fingerprint claims to
        cover the model, and a file elsewhere on the machine is not part of it.
    """
    value = raw.replace("\\", "/")
    if os.path.isabs(value):
        target = os.path.normpath(value)
        root = os.path.normpath(str(model_root))
    else:
        target = os.path.normpath(os.path.join(str(model_root), value))
        root = os.path.normpath(str(model_root))
    rel = os.path.relpath(target, root).replace("\\", "/")
    if rel == ".." or rel.startswith("../"):
        raise ValueError(
            f"model pointer {raw!r} resolves outside the model root "
            f"({rel!r} relative to {model_root}); the digest covers the model "
            f"only, so an escaping pointer is an error rather than a widened "
            f"digest"
        )
    return rel


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def model_file_set(model_root, toml_name: str = MODEL_TOML_NAME) -> list[str]:
    """The root-relative paths the digest covers, sorted, TOML first.

    Separated from the hashing so a caller — or a phase report — can show WHICH
    files a given model contributes without recomputing content hashes.
    """
    model_root = Path(model_root)
    toml_path = model_root / toml_name
    if not toml_path.is_file():
        raise FileNotFoundError(f"no {toml_name} at model root {model_root}")
    with open(toml_path, "rb") as handle:
        doc = tomllib.load(handle)
    rels = {toml_name}
    for _key, value in _walk_path_values(doc):
        rels.add(_relative_to_root(value, model_root))
    return sorted(rels)


def model_digest(model_root, toml_name: str = MODEL_TOML_NAME) -> str:
    """Deterministic SHA-256 over the model's runtime inputs.

    Parameters
    ----------
    model_root : str | Path
        The live model root (``models/hydrology/wflow``).
    toml_name : str
        The run configuration's filename; the root of the pointer graph.

    Returns
    -------
    str
        Hex digest. Depends only on the sorted root-relative paths and their
        contents, so it is identical on Windows and Linux for the same model.
    """
    entries = model_digest_entries(model_root, toml_name)
    h = hashlib.sha256()
    h.update(f"cst-model-digest-v{DIGEST_VERSION}\n".encode())
    for rel, content in entries:
        # Both the PATH and the content are hashed: renaming a pointed-at file
        # changes the model even when the bytes are unchanged.
        h.update(f"{rel}\n{content}\n".encode())
    return h.hexdigest()


def model_digest_entries(
    model_root, toml_name: str = MODEL_TOML_NAME
) -> list[tuple[str, str]]:
    """The ``(relative_path, content_hash_or_marker)`` pairs the digest hashes.

    Exposed because a mismatch must be able to NAME the changed input; a bare
    digest comparison can only say that something moved.
    """
    model_root = Path(model_root)
    entries: list[tuple[str, str]] = []
    for rel in model_file_set(model_root, toml_name):
        target = model_root / rel
        entries.append((rel, _sha256_file(target) if target.is_file() else ABSENT))
    return entries


def compare_model_digest(
    model_root, expected_entries, toml_name: str = MODEL_TOML_NAME
) -> list[str]:
    """Report how the live model differs from a recorded entry list.

    Returns ``[]`` when they agree. Mirrors the house drift-guard shape (a
    ``list[str]`` report, never an exception) so every difference surfaces at
    once and the caller decides how loudly to fail.
    """
    now = dict(model_digest_entries(model_root, toml_name))
    was = {str(k): str(v) for k, v in dict(expected_entries).items()}
    diffs: list[str] = []
    for rel in sorted(set(was) | set(now)):
        old, new = was.get(rel), now.get(rel)
        if old == new:
            continue
        if old is None:
            diffs.append(f"model input added since the experiment ran: {rel}")
        elif new is None:
            diffs.append(f"model input no longer referenced by the TOML: {rel}")
        elif old == ABSENT:
            diffs.append(f"model input appeared since the experiment ran: {rel}")
        elif new == ABSENT:
            diffs.append(f"model input disappeared since the experiment ran: {rel}")
        else:
            diffs.append(f"model input changed since the experiment ran: {rel}")
    return diffs
