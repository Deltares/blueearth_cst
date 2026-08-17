"""Report whether a checkout's environment can actually run the workflows.

``pixi install`` is not the whole environment, and the gap is invisible until a
run reaches the layer that is missing. This checks the three things it does
NOT guarantee, each of which has failed on its own:

1. **weathergenr.** It is installed from GitHub via ``remotes``
   (``pixi run install``), not from ``pixi.toml``, so a checkout that has only
   had ``pixi install`` looks healthy and dies at WF3's weather generator with
   ``there is no package called 'weathergenr'``.
2. **The Julia environment.** Same split — ``Pkg.instantiate()`` runs from
   ``pixi run install``. It usually passes on a second checkout because the
   depot under ``~/.julia`` is shared, which is exactly why its absence is
   surprising when it does happen.
3. **Console scripts a distribution CLAIMS but that are not on disk.**
   ``pixi install`` reconciles against its own metadata, so it neither detects
   nor repairs this, and reports success either way. Only code that SHELLS OUT
   to the executable breaks, while everything importing the module passes — a
   suite failing in one narrow layer for no visible reason (watch-item
   ``t2608121104``; the repair is recorded there).

**Worktrees are the case this exists for.** Each carries its own tracked
``pixi.toml`` and therefore builds its own ``.pixi/``, so every session slot needs the
``pixi run install`` layer separately. ``--all`` checks every worktree that
``git worktree list`` knows about, which is the question actually being asked:
not "is this checkout complete" but "are they all".

Read-only: this reports, and never installs. The fix it points at is
``pixi run install`` in the checkout that failed. Exits non-zero when anything
is incomplete, so it can gate a script.

Usage (from the repo root, inside pixi)::

    python dev/scripts/check_env.py            # this checkout
    python dev/scripts/check_env.py --all      # every registered worktree
    python dev/scripts/check_env.py --root <path>

Not part of a run: this inspects a checkout (see AGENTS.md, "Three homes for
executables").
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Julia's Windows libcurl uses Schannel, which does not support the CAPATH that
# conda-forge's openssl activation puts in SSL_CERT_DIR -- the same clearing
# `pixi.toml`'s install-julia task documents. Cleared for the probe only.
_TLS_UNSET = {"SSL_CERT_DIR": "", "SSL_CERT_FILE": ""}


def _env_root(root: Path) -> Path:
    return root / ".pixi" / "envs" / "default"


def _missing_console_scripts(root: Path) -> list[str]:
    """Names each installed distribution claims in ``Scripts/`` but did not write.

    Windows-shaped on purpose: this is where the failure was observed, and the
    POSIX layout puts these in ``bin/`` with no ``.exe``. A checkout whose env
    has neither directory reports nothing rather than every entry.
    """
    env = _env_root(root)
    site = env / "Lib" / "site-packages"
    scripts = env / "Scripts"
    if not site.is_dir() or not scripts.is_dir():
        return []
    missing = []
    for record in site.glob("*.dist-info/RECORD"):
        text = record.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            claimed = line.split(",")[0].replace("\\", "/")
            if "Scripts/" not in claimed:
                continue
            name = claimed.split("Scripts/")[-1]
            # An extension-less POSIX name beside a real `.exe` is a packaging
            # artifact (numba ships both), not a missing executable.
            if (scripts / name).exists() or (scripts / f"{name}.exe").exists():
                continue
            missing.append(f"{record.parent.name} -> {name}")
    return sorted(missing)


def _rscript(root: Path) -> Path | None:
    env = _env_root(root)
    for candidate in (
        env / "lib" / "R" / "bin" / "Rscript.exe",
        env / "Scripts" / "Rscript.exe",
        env / "lib" / "R" / "bin" / "Rscript",
        env / "bin" / "Rscript",
    ):
        if candidate.exists():
            return candidate
    return None


def _weathergenr(root: Path) -> tuple[bool, str]:
    """Whether weathergenr LOADS, which a present directory does not prove."""
    rscript = _rscript(root)
    if rscript is None:
        return False, "no Rscript in this env"
    probe = subprocess.run(
        [
            str(rscript),
            "--vanilla",
            "-e",
            'cat(as.character(packageVersion("weathergenr")))',
        ],
        capture_output=True,
        text=True,
        cwd=str(root),
        env={**os.environ, **_TLS_UNSET},
    )
    version = (probe.stdout or "").strip()
    if probe.returncode != 0 or not version:
        return False, "not installed -- run `pixi run install`"
    return True, version


def _julia(root: Path) -> tuple[bool, str]:
    """Whether this checkout's Julia project resolves and carries Wflow."""
    probe = subprocess.run(
        [
            "julia",
            "+1.11.7",
            f"--project={root}",
            "-e",
            "using Pkg; io=IOBuffer(); Pkg.status(io=io); print(String(take!(io)))",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, **_TLS_UNSET},
    )
    if probe.returncode != 0:
        detail = (probe.stderr or "").strip().splitlines()
        return False, detail[-1][:120] if detail else "julia failed"
    if "Wflow" not in probe.stdout:
        return False, "no Wflow -- run `pixi run install`"
    return True, "Wflow present"


def check(root: Path) -> bool:
    """Print one block for ``root``; return whether everything is present."""
    print(f"== {root}")
    if not _env_root(root).is_dir():
        print("   pixi env         ABSENT -- run `pixi install`")
        return False
    print("   pixi env         present")

    ok = True

    missing = _missing_console_scripts(root)
    if missing:
        ok = False
        print(f"   console scripts  MISSING ({len(missing)}):")
        for entry in missing:
            print(f"       {entry}")
        print("       repair: see dev/tasks/t2608121104-*.md")
    else:
        print("   console scripts  ok")

    found, detail = _weathergenr(root)
    ok = ok and found
    print(f"   weathergenr      {detail if found else 'MISSING -- ' + detail}")

    found, detail = _julia(root)
    ok = ok and found
    print(f"   julia env        {detail if found else 'INCOMPLETE -- ' + detail}")

    return ok


def _worktrees() -> list[Path]:
    """Every checkout ``git worktree list`` knows about, primary included."""
    listing = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    if listing.returncode != 0:
        return [REPO]
    return [
        Path(line.split(" ", 1)[1])
        for line in listing.stdout.splitlines()
        if line.startswith("worktree ")
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--all",
        action="store_true",
        help="check every worktree git knows about, not just this checkout",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO,
        help="checkout to check (default: the repository this script is in)",
    )
    args = parser.parse_args(argv)

    roots = _worktrees() if args.all else [args.root.resolve()]
    results = []
    for index, root in enumerate(roots):
        if index:
            print()
        results.append(check(root))

    if all(results):
        return 0
    print()
    print("Incomplete. In each checkout above: pixi install && pixi run install")
    return 1


if __name__ == "__main__":
    sys.exit(main())
