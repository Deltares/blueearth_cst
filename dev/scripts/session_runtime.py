"""Run a session command through a compatible, owner-maintained Pixi runtime.

Bootstrap with the runtime's Python (stdlib only); actual commands use Pixi
activation, executable mode, and --as-is. No environment is linked or installed.
The runtime owner must finish provisioning before lending it and must not
install or update it while borrowers are running. This is cooperative session
coordination, not a security boundary against arbitrary child code.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tomllib
from pathlib import Path

CONTRACT_FILES = (
    "pixi.toml",
    "pixi.lock",
    "Project.toml",
    "Manifest.toml",
    "dev/scripts/install_weathergenr.R",
    "dev/scripts/pixi_activate.bat",
)


def check_compatibility(source: Path, runtime: Path) -> None:
    """Reject changed dependencies/activation before invoking Pixi at all."""
    for relative in CONTRACT_FILES:
        left, right = source / relative, runtime / relative
        # Git may check out the same contract with LF or CRLF.
        left_text = left.read_text(encoding="utf-8")
        right_text = right.read_text(encoding="utf-8")
        matches = (
            tomllib.loads(left_text) == tomllib.loads(right_text)
            if relative.endswith(".toml")
            else left_text == right_text
        )
        if not matches:
            raise ValueError(
                f"Incompatible runtime: {relative} differs. Use this session's "
                "isolated environment: pixi install --locked; pixi run install."
            )
    prefix = runtime / ".pixi/envs/default"
    if not (prefix / "conda-meta/pixi").is_file():
        raise ValueError("Runtime is not provisioned; its owner must install it.")
    if os.name == "nt":
        # Activation otherwise copies aliases into the borrowed installation.
        for dll in ("ffi.dll", "cairo.dll"):
            if not (prefix / "Library/bin" / dll).is_file():
                raise ValueError(f"Runtime owner must complete activation: {dll}")


def command_environment(source: Path) -> dict[str, str]:
    """Keep imports, cache writes and Julia's active project in the session."""
    env = dict(os.environ)
    scratch = source / ".tmp/session-runtime"
    scratch.mkdir(parents=True, exist_ok=True)
    env.update(
        PYTHONPATH=str(source),
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        JULIA_PROJECT=str(source),
        JULIA_PKG_PRECOMPILE_AUTO="0",
        PIXI_NO_INSTALL="1",
        PIXI_FROZEN="1",
        PIXI_PROJECT_ROOT=str(source),
        PIXI_PROJECT_MANIFEST=str(source / "pixi.toml"),
        R_LIBS_USER=str(scratch / "disabled-r-user-library"),
    )
    env.pop("R_LIBS", None)
    env.pop("R_LIBS_SITE", None)
    for variable, relative in (
        ("MPLCONFIGDIR", "matplotlib"),
        ("XDG_CACHE_HOME", "cache"),
        ("XDG_DATA_HOME", "data"),
        ("NUMBA_CACHE_DIR", "numba"),
        ("PROJ_USER_WRITABLE_DIRECTORY", "proj"),
        ("RUFF_CACHE_DIR", "ruff"),
    ):
        location = scratch / relative
        location.mkdir(exist_ok=True)
        env[variable] = str(location)
    # Julia writes caches to the first depot. Later depots supply installed
    # packages/artifacts read-only by convention. A trailing empty component
    # expands SYSTEM depots, so the normal user depot must be explicit too.
    depot = scratch / "julia"
    depot.mkdir(exist_ok=True)
    installed_depots = env.get(
        "JULIA_DEPOT_PATH", str(Path.home() / ".julia") + os.pathsep
    )
    env["JULIA_DEPOT_PATH"] = str(depot) + os.pathsep + installed_depots
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--activated", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    source, runtime = args.source_root.resolve(), args.runtime_root.resolve()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("supply an executable after -- (Pixi task names are not expanded)")
    try:
        check_compatibility(source, runtime)
        if Path(command[0]).stem.lower() in {
            "pixi",
            "pip",
            "pip3",
            "conda",
            "mamba",
            "uv",
        }:
            raise ValueError("Installation commands require an isolated session.")
        if command[:3] == ["python", "-m", "pip"]:
            raise ValueError("Installation commands require an isolated session.")
        if not args.activated:
            # -x is essential: named tasks switch cwd to the runtime checkout.
            return subprocess.call(
                [
                    "pixi",
                    "run",
                    "--as-is",
                    "--executable",
                    "--manifest-path",
                    str(runtime / "pixi.toml"),
                    "python",
                    "-B",
                    str(Path(__file__).resolve()),
                    "--runtime-root",
                    str(runtime),
                    "--source-root",
                    str(source),
                    "--activated",
                    "--",
                    *command,
                ],
                cwd=source,
            )
        prefix = runtime / ".pixi/envs/default"
        if Path(sys.prefix).resolve() != prefix.resolve():
            raise ValueError(f"Wrong interpreter prefix: {sys.prefix}")
        env = command_environment(source)
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import blueearth_cst; print(blueearth_cst.__file__)",
            ],
            cwd=source,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        imported = Path(probe.stdout.strip()).resolve()
        if not imported.is_relative_to(source / "blueearth_cst"):
            raise ValueError(f"Wrong Python source: {imported}")
        print(
            f"source={source}\nruntime={prefix}\npython_source={imported}", flush=True
        )
        return subprocess.call(command, cwd=source, env=env)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"session-runtime: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
