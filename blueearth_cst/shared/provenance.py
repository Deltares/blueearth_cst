"""Deterministic digest helpers for CST configuration provenance."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any

logger = logging.getLogger(__name__)

#: Repository root, three levels up from ``blueearth_cst/shared/provenance.py``.
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Commit baked into a deployed image by the ``Dockerfile``.
#:
#: The image carries no ``.git`` -- sources are ADDed individually -- so a
#: container has no checkout to interrogate. Gitignored, and therefore absent in
#: a normal checkout, which is what keeps :func:`toolbox_identity`'s git branch
#: authoritative wherever a checkout exists.
TOOLBOX_COMMIT_FILE = ".toolbox-commit"

#: Lock files whose bytes identify the resolved dependency set.
#:
#: These survive the absence of git, so they are the identity that still holds
#: in a deployed image whose ``.toolbox-commit`` was never baked.
ENVIRONMENT_FILES = ("pixi.lock", "Manifest.toml")

#: Schema of the effective-config document (:func:`effective_config_document`).
#:
#: 2 -- the document carries a declared consumed-key PROJECTION rather than the
#: whole parsed configfile, so a reader can tell a rescoped digest from a
#: changed config across the transition. A version-1 document digested every
#: key in the file, which made an unrelated workflow's edit move this
#: workflow's digest.
EFFECTIVE_CONFIG_SCHEMA_VERSION = 2

_CANONICAL_JSON_OPTIONS = {
    "allow_nan": False,
    "ensure_ascii": False,
    "separators": (",", ":"),
    "sort_keys": True,
}

#: Characters kept when a digest is used as a NAME rather than as proof.
#:
#: A snapshot bundle's directory and each archived file inside it are named
#: after their digest, and a full SHA-256 makes both unreadable — a 64-hex
#: directory under ``config/runs/<workflow>/`` tells a user nothing except that
#: something hashed it. Twelve hex is 48 bits: within one project, whose
#: bundles number in the tens, a collision is not a practical concern, and the
#: name is short enough to read, compare by eye, and type.
#:
#: The full digest is never lost — it is recorded INSIDE the artifact
#: (``snapshot_bundle_sha256`` and the per-file ``sha256`` in a bundle's
#: ``referenced-files.json``), so the short form is a handle and the long form
#: stays the record. Both start with the same characters, so a short name is a
#: prefix of the digest it stands for.
#:
#: ONE constant, because the bundle directory and the files inside it must not
#: drift apart: a 64-character directory holding 12-character filenames is what
#: this replaced.
SHORT_DIGEST_CHARS = 12


def short_digest(digest: str) -> str:
    """Return the naming form of a hex digest.

    Args:
        digest: Full hex digest, as returned by the ``*_sha256`` helpers here.

    Returns:
        Its first :data:`SHORT_DIGEST_CHARS` characters.

    Raises:
        ValueError: If ``digest`` is too short to truncate, which means the
            caller passed something that is not a digest — silently returning
            it would name an artifact after a value with no collision
            properties at all.
    """
    if not isinstance(digest, str):
        raise TypeError("digest must be a string")
    if len(digest) < SHORT_DIGEST_CHARS:
        raise ValueError(
            f"expected a hex digest of at least {SHORT_DIGEST_CHARS} "
            f"characters, got {digest!r}"
        )
    return digest[:SHORT_DIGEST_CHARS]


def canonical_data(value: Any) -> dict[str, Any]:
    """Return a deterministic, explicitly typed JSON-compatible value.

    Supported values are mappings, lists, tuples, paths, and YAML scalar
    primitives (``None``, booleans, integers, floats, and strings). Unsupported
    objects raise instead of being silently converted with ``str`` or ``repr``.

    Args:
        value: Value to convert to canonical data.

    Returns:
        Type-tagged data composed only of JSON-compatible primitives.

    Raises:
        TypeError: If ``value`` or a nested value has an unsupported type.
        ValueError: If a mapping has duplicate canonical keys.
    """
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        return {"type": "float", "value": _canonical_float(value)}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, PurePath):
        return {"type": "path", "value": value.as_posix()}
    if isinstance(value, Mapping):
        return _canonical_mapping(value)
    if isinstance(value, list):
        return {"type": "list", "items": [canonical_data(item) for item in value]}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [canonical_data(item) for item in value]}
    raise TypeError(
        f"unsupported provenance value of type {type(value).__name__}; "
        "expected a mapping, list, tuple, path, or YAML scalar primitive"
    )


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 of a value's canonical typed JSON representation."""
    payload = json.dumps(canonical_data(value), **_CANONICAL_JSON_OPTIONS).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_config(
    config: Mapping[Any, Any], projection: Sequence[str]
) -> dict[str, Any]:
    """Select a workflow's declared consumed-key paths from a config mapping.

    A projection is the explicit list of dotted config paths a workflow reads,
    including cross-section ones -- ``("project", "shared",
    "workflows.model_creation")`` for WF1. Digesting the projection instead of
    the whole file is what stops an edit to one workflow's section from moving
    another workflow's digest.

    Args:
        config: Parsed configuration mapping.
        projection: Dotted paths to select. No path may be a prefix of another;
            declaring both ``shared`` and ``shared.basin`` is ambiguous about
            which one the digest covers.

    Returns:
        A nested mapping holding only the selected paths, shaped like the
        source config so a reader sees familiar structure.

    Raises:
        TypeError: If ``config`` is not a mapping, or a path is not a
            non-empty string.
        ValueError: If two paths overlap.
        KeyError: If a declared path is absent from ``config``. This is loud on
            purpose: the projection is a DECLARATION by the Snakefile author
            about what the workflow reads, so a missing path is either a typo'd
            declaration or a config missing a section the workflow needs.
            Silently omitting it would produce a digest that fails to move when
            that section changes -- a provenance record that quietly stops
            recording.
    """
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    paths = list(projection)
    for path in paths:
        if not isinstance(path, str) or not path:
            raise TypeError("each projection path must be a non-empty string")
    for path in paths:
        for other in paths:
            if path is not other and other.startswith(f"{path}."):
                raise ValueError(
                    f"projection paths overlap: {path!r} contains {other!r}; "
                    "declare one or the other, not both"
                )

    projected: dict[str, Any] = {}
    for path in paths:
        parts = path.split(".")
        source: Any = config
        walked: list[str] = []
        for part in parts:
            walked.append(part)
            if not isinstance(source, Mapping) or part not in source:
                raise KeyError(
                    f"projection path {path!r} is not present in the config "
                    f"(missing at {'.'.join(walked)!r})"
                )
            source = source[part]
        target = projected
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = source
    return projected


def effective_config_document(
    config: Mapping[Any, Any],
    advanced_settings: Mapping[Any, Any],
    projection: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build the scientific configuration document used for digesting.

    Execution-only options such as cores, dry-run, and verbosity do not belong
    in this document. ``config`` is the configuration mapping actually resolved
    for the workflow, while ``advanced_settings`` is the validated toolbox-wide
    settings mapping.

    Args:
        config: Parsed configuration mapping.
        advanced_settings: Validated toolbox-wide settings mapping. These stay
            INSIDE the document: they are configuration -- constraints and
            defaults that shape the run -- and they are recorded nowhere else
            in a project tree.
        projection: Declared consumed-key paths (see :func:`project_config`),
            or ``None`` to cover the whole config. ``None`` is right for a
            caller with no single workflow scope, such as the
            ``run_workflows.py`` wrapper manifest that spans all three.

            It defaults to ``None`` so that adding this parameter does not
            break the callers that predate it -- the Snakefiles and
            ``copy_config_files.py`` acquire their projections in later phases,
            and each phase has to leave the tree working on its own. The
            default is also the SAFE direction: an omitted projection digests
            more than necessary, so the record is noisy rather than blind. The
            enforcement that each Snakefile actually declares one lives in the
            Snakefile-side tests, not in this signature.
    """
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if not isinstance(advanced_settings, Mapping):
        raise TypeError("advanced_settings must be a mapping")
    scoped = config if projection is None else project_config(config, projection)
    return {
        "schema_version": EFFECTIVE_CONFIG_SCHEMA_VERSION,
        "projection": None if projection is None else sorted(projection),
        "project_config": scoped,
        "advanced_settings": advanced_settings,
    }


def effective_config_digest(
    config: Mapping[Any, Any],
    advanced_settings: Mapping[Any, Any],
    projection: Sequence[str] | None = None,
) -> str:
    """Return the canonical SHA-256 of the effective scientific config.

    This is CONFIGURATION IDENTITY: "the settings the workflow was asked to run
    under". It deliberately says nothing about the code, the environment, or
    the referenced input files -- :func:`configuration_inputs_digest` is the
    wider question.
    """
    return canonical_sha256(
        effective_config_document(config, advanced_settings, projection)
    )


def configuration_inputs_digest(
    effective_config_sha256: str,
    toolbox: Mapping[str, Any],
    environment: Mapping[str, Any],
    referenced_inputs: Iterable[Mapping[str, Any]] = (),
) -> str:
    """Digest configuration identity together with code, environment and inputs.

    Answers "did this run see the same CONFIGURATION-SIDE inputs as that one".
    Threading it through a rule's ``params:`` is what makes the run record
    refresh when the checkout moves, not only when the config is edited.

    **What this deliberately does NOT cover:** the contents of scientific
    datasets addressed *through* a catalog. A remote or mutable dataset can
    change under an unchanged catalog entry and this digest will not move. It
    is configuration-input identity, never scientific run identity -- the name
    says ``configuration_inputs`` rather than ``run_inputs`` for exactly that
    reason. Generated intermediate inputs are likewise excluded.

    **Comparability:** two values assert same-code only when both records have
    a non-null ``commit`` and ``dirty`` false. With ``commit: None`` (a
    git-less deployment with no baked commit) or ``dirty: True``, equal digests
    do not imply equal code -- the environment hashes still pin the dependency
    set, but the toolbox source itself is unwitnessed.

    Args:
        effective_config_sha256: Output of :func:`effective_config_digest`. The
            digest string is folded in rather than the document it covers; the
            two are the same identity, and the string keeps this function
            independent of the document's shape.
        toolbox: Output of :func:`toolbox_identity`.
        environment: Output of :func:`environment_file_hashes`.
        referenced_inputs: Descriptors for external files the run consumes. A
            recoverable file contributes its git blob id, a copied one its byte
            hash, and a logical identifier with no path contributes its
            identifier string -- so the digest tracks CONTENT, whichever way
            the file is retained. Order does not affect the result.
    """
    if not isinstance(effective_config_sha256, str):
        raise TypeError("effective_config_sha256 must be a string")
    identities = sorted(_reference_identity(item) for item in referenced_inputs)
    document = {
        "schema_version": EFFECTIVE_CONFIG_SCHEMA_VERSION,
        "effective_config_sha256": effective_config_sha256,
        "toolbox": {
            key: toolbox.get(key) for key in ("commit", "commit_source", "dirty")
        },
        "environment": {name: environment.get(name) for name in ENVIRONMENT_FILES},
        "referenced_inputs": identities,
    }
    return canonical_sha256(document)


def snapshot_bundle_digest(
    config: Mapping[Any, Any],
    advanced_settings: Mapping[Any, Any],
    source_config_path: str | Path,
    referenced_inputs: Iterable[Mapping[str, Any]],
) -> str:
    """Digest effective config, source bytes, and explicit referenced inputs.

    Each reference descriptor requires string ``kind`` and ``identifier``
    fields. A descriptor with ``path`` identifies a local file: its logical
    identifier and byte SHA-256 are included, but its machine-specific physical
    path is not. A descriptor without ``path`` is a logical identifier such as
    a catalog source name. Descriptor input order does not affect the digest.

    Args:
        config: Effective project configuration mapping.
        advanced_settings: Resolved toolbox-wide settings mapping.
        source_config_path: Source YAML whose exact bytes are hashed.
        referenced_inputs: Explicit file or logical reference descriptors.

    Returns:
        Canonical SHA-256 for the complete snapshot bundle.
    """
    references = [_reference_document(item) for item in referenced_inputs]
    references.sort(key=_canonical_json)
    document = {
        "schema_version": 1,
        # projection=None keeps this legacy digest over the whole config, which
        # is what it has always covered. The bundle it names is removed in P2;
        # rescoping it on the way out would change a digest nobody reads.
        "effective_config": effective_config_document(config, advanced_settings, None),
        "source_config_sha256": file_sha256(source_config_path),
        "referenced_inputs": references,
    }
    return canonical_sha256(document)


def toolbox_identity(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Resolve which revision of the toolbox is running.

    Resolution order, first hit wins:

    1. ``git rev-parse HEAD`` plus ``git status --porcelain`` -- the live
       checkout. ``commit_source: "git"``.
    2. ``.toolbox-commit`` in the repo root -- a commit baked into a deployed
       image, which carries no ``.git``. ``commit_source: "baked"``, and
       ``dirty`` is ``None`` because a baked file cannot know.
    3. Nothing. All three fields ``None``.

    Every failure is swallowed: a provenance helper must never crash a run.
    An unresolvable identity is recorded as ``None`` rather than guessed, so a
    record that cannot witness its own code says so instead of lying.

    Returns:
        ``{"commit": ..., "commit_source": ..., "dirty": ...}``. ``dirty``
        being ``True`` means the commit does NOT fully identify the code that
        ran; ``None`` means it was unknowable.
    """
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    commit_result = _run_metadata_command(["git", "rev-parse", "HEAD"], root)
    if commit_result is not None and commit_result.strip():
        status_result = _run_metadata_command(["git", "status", "--porcelain"], root)
        return {
            "commit": commit_result.strip(),
            "commit_source": "git",
            "dirty": None if status_result is None else bool(status_result.strip()),
        }

    baked = _read_baked_commit(root / TOOLBOX_COMMIT_FILE)
    if baked is not None:
        return {"commit": baked, "commit_source": "baked", "dirty": None}

    return {"commit": None, "commit_source": None, "dirty": None}


def environment_file_hashes(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Hash the lock files that identify the resolved dependency set.

    Returns:
        One entry per :data:`ENVIRONMENT_FILES`, its byte SHA-256 or ``None``
        when the file is absent. The key is always present: an explicit
        ``None`` records "this file was not there", which an omitted key
        cannot distinguish from "nobody looked".
    """
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    hashes: dict[str, Any] = {}
    for name in ENVIRONMENT_FILES:
        path = root / name
        try:
            hashes[name] = file_sha256(path) if path.is_file() else None
        except OSError:
            hashes[name] = None
    return hashes


def append_journal_line(path: str | Path, record: Mapping[str, Any]) -> None:
    """Append one JSON line to a run journal.

    The journal is an append-only ledger of EXECUTED invocations. It must never
    be a declared output of any Snakemake rule: Snakemake deletes a rule's
    declared outputs before executing the job, which would truncate the journal
    to one line on every run -- silently, since a one-line journal is
    indistinguishable from a young one.

    The whole line is written in a single ``write()`` under ``O_APPEND`` (mode
    ``"a"``), which is what keeps concurrent writers from interleaving within a
    line. Failures are swallowed for the same reason as in
    :func:`toolbox_identity`: losing a journal line must never fail a run that
    otherwise succeeded.
    """
    line = json.dumps(dict(record), **_CANONICAL_JSON_OPTIONS)
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(f"{line}\n")
            handle.flush()
    except OSError as error:
        logger.warning("could not append to run journal %s: %s", path, error)


def journal_event(
    *,
    invocation_id: str,
    workflow: str,
    event: str,
    toolbox: Mapping[str, Any],
    effective_config_sha256: str,
    configuration_inputs_sha256: str,
    source_config_sha256: str | None = None,
    experiment: str | None = None,
) -> dict[str, Any]:
    """Build one journal line.

    ``event`` is ``started``, ``success`` or ``failed``. The TERMINAL line is
    the contract; ``started`` is best-effort crash tracing, so an invocation
    killed hard leaves a ``started`` with no partner, which is itself the
    diagnostic.

    WF3 lines carry ``experiment``: the workflow where per-run identity matters
    most should not depend on digest-matching to name its own experiment.
    """
    record = {
        "ts": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "invocation_id": invocation_id,
        "workflow": workflow,
        "event": event,
        "commit": toolbox.get("commit"),
        "commit_source": toolbox.get("commit_source"),
        "dirty": toolbox.get("dirty"),
        "effective_config_sha256": effective_config_sha256,
        "configuration_inputs_sha256": configuration_inputs_sha256,
    }
    if source_config_sha256 is not None:
        record["source_config_sha256"] = source_config_sha256
    if experiment is not None:
        record["experiment"] = experiment
    return record


def read_journal_lines(path: str | Path) -> list[dict[str, Any]]:
    """Read a run journal, tolerating a torn final line.

    An invocation killed mid-write can leave a partial last line. The file's
    value is its accumulated history, so a torn tail must not poison it: any
    line that fails to parse is skipped with a warning rather than raising.

    Returns:
        The parsed lines, in file order. A missing file reads as empty.
    """
    target = Path(path)
    if not target.is_file():
        return []
    records: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                logger.warning(
                    "skipping unparsable journal line %d in %s", number, path
                )
                continue
            if isinstance(record, dict):
                records.append(record)
            else:
                logger.warning(
                    "skipping non-object journal line %d in %s", number, path
                )
    return records


def _read_baked_commit(path: Path) -> str | None:
    """Return the first non-empty line of a baked commit file, if any."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _run_metadata_command(command: list[str], cwd: Path) -> str | None:
    """Run a cheap metadata query without making provenance capture fragile."""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return getattr(result, "stdout", "")


def _reference_identity(reference: Mapping[str, Any]) -> str:
    """Reduce one referenced-input descriptor to its content identity.

    A file is identified by its byte hash, and a logical reference with no
    bytes -- a catalog source name -- by its own identifier.

    **Deviation from design-v3 §5.4, deliberate and flagged for review.** The
    design says recoverable files contribute their *git blob id* and copied
    ones their byte hash. Every descriptor carries ``sha256`` regardless
    (§5.2's schema), so preferring it uniformly costs nothing and buys one
    property the split rule loses: the digest then follows CONTENT alone.
    Under the split rule, re-classifying a file from recoverable to copied
    moves the digest without any byte changing -- a policy change masquerading
    as an input change, in the one digest whose job is to say whether the
    inputs differ. ``git_blob`` is still honoured when no hash is present.
    """
    if not isinstance(reference, Mapping):
        raise TypeError("each referenced input must be a mapping")
    for field in ("sha256", "git_blob"):
        value = reference.get(field)
        if isinstance(value, str) and value:
            return f"{field}:{value}"
    # origin before role: the role is a CLASS ("data_catalog") and several
    # references can share one, so falling back to it first would give two
    # different pathless catalogs the same identity and collapse them in the
    # digest. The origin is what distinguishes them.
    for field in ("origin", "identifier", "role"):
        value = reference.get(field)
        if isinstance(value, str) and value:
            return f"identifier:{value}"
    raise ValueError(
        "a referenced input needs a 'sha256', a 'git_blob', or an identifying "
        f"'origin'/'identifier'/'role'; got {sorted(reference)}"
    )


def referenced_inputs_for_digest(
    items: Iterable[tuple[str, Any]],
) -> list[dict[str, Any]]:
    """Describe a workflow's referenced config inputs at Snakefile PARSE time.

    Hash what is on disk, name what is not -- hydromt's predefined catalogs are
    identifiers with no path. This runs on every invocation before the DAG is
    built, which is what lets a rule's ``params:`` carry the result: an
    in-place edit to a referenced catalog then re-fires the rule that records
    it, closing the gap where only the recorded hash moved and nothing re-ran.

    Costs milliseconds -- these are small YAML and CSV files.

    Args:
        items: ``(role, path_or_identifier)`` pairs. Roles may repeat; the
            origin is what distinguishes two references sharing one.
    """
    entries: list[dict[str, Any]] = []
    for role, identifier in items:
        origin = str(identifier)
        entry: dict[str, Any] = {"role": role, "origin": origin}
        path = Path(origin)
        try:
            if path.is_file():
                entry["sha256"] = file_sha256(path)
        except OSError:
            pass
        entries.append(entry)
    return entries


def _canonical_float(value: float) -> str:
    """Return an exact, platform-stable representation of a float."""
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return value.hex()


def _canonical_mapping(value: Mapping[Any, Any]) -> dict[str, Any]:
    """Canonicalize and sort a mapping by the canonical form of each key."""
    items: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    seen_keys: set[str] = set()
    for key, item in value.items():
        canonical_key = canonical_data(key)
        sort_key = _typed_json(canonical_key)
        if sort_key in seen_keys:
            raise ValueError("mapping contains duplicate canonical keys")
        seen_keys.add(sort_key)
        items.append((sort_key, canonical_key, canonical_data(item)))
    items.sort(key=lambda item: item[0])
    return {
        "type": "mapping",
        "items": [{"key": key, "value": item_value} for _, key, item_value in items],
    }


def _canonical_json(value: Any) -> str:
    """Serialize supported data canonically for stable sorting."""
    return json.dumps(canonical_data(value), **_CANONICAL_JSON_OPTIONS)


def _typed_json(value: dict[str, Any]) -> str:
    """Serialize an already canonicalized value without adding more tags."""
    return json.dumps(value, **_CANONICAL_JSON_OPTIONS)


def _reference_document(reference: Mapping[str, Any]) -> dict[str, str]:
    """Validate and normalize one snapshot reference descriptor."""
    if not isinstance(reference, Mapping):
        raise TypeError("each referenced input must be a mapping")
    allowed = {"kind", "identifier", "path"}
    unknown = set(reference) - allowed
    kind = reference.get("kind")
    identifier = reference.get("identifier")
    if unknown or not isinstance(kind, str) or not isinstance(identifier, str):
        raise ValueError(
            "each referenced input requires string 'kind' and 'identifier' "
            "fields, plus only an optional 'path'"
        )
    document = {"kind": kind, "identifier": identifier}
    if "path" in reference:
        path = reference["path"]
        if not isinstance(path, (str, PurePath)):
            raise TypeError("referenced input 'path' must be a string or path")
        document["sha256"] = file_sha256(path)
    return document
