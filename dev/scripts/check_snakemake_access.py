"""Bounded check of the installed Snakemake source cache under the current token.

The child creates and removes only its own temporary runtime-cache directory.
No workflow, model, dependency installation or cache-permission change is run.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def _probe() -> None:
    """Exercise Snakemake's actual temporary-directory and file-access path."""
    from snakemake.common import get_appdirs
    from snakemake.sourcecache import SourceCache

    cache_path = Path(get_appdirs().user_cache_dir) / "snakemake/source-cache"
    print(cache_path, flush=True)
    cache = SourceCache(cache_path)
    try:
        probe = Path(cache.runtime_cache_path) / "access-probe.txt"
        probe.write_text("cache-access", encoding="utf-8")
        if probe.read_text(encoding="utf-8") != "cache-access":
            raise OSError("source-cache file did not round-trip")
    finally:
        cache.runtime_cache.cleanup()


def main(argv: list[str] | None = None) -> int:
    """Exit zero for usable cache access, or two with a bounded failure diagnosis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout", type=int, default=15, help="child timeout in seconds (default: 15)"
    )
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.probe:
        _probe()
        return 0
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--probe"],
            capture_output=True,
            text=True,
            timeout=args.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(
            f"Snakemake cache probe exceeded {args.timeout}s; its child was stopped.",
            file=sys.stderr,
        )
    else:
        if result.returncode == 0:
            print(f"Snakemake cache access OK: {result.stdout.strip()}")
            return 0
        detail = result.stderr.strip().splitlines()
        print(
            f"Snakemake cache access failed: {detail[-1] if detail else result.returncode}",
            file=sys.stderr,
        )
    print(
        "On Windows, rerun Snakemake checks with normal user permissions outside the restricted agent sandbox. "
        "If this also fails there, inspect the cache error before running workflows.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
