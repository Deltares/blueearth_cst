"""Windows activation resolves Quarto tools from the active environment."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(os.name != "nt", reason="Windows batch activation")
def test_quarto_activation_replaces_stale_paths(tmp_path: Path) -> None:
    prefix = tmp_path / "relocated environment"
    library = prefix / "Library"
    expected = {
        "QUARTO_CONDA_PREFIX": library,
        "QUARTO_SHARE_PATH": library / "share/quarto",
        "QUARTO_DENO_DOM": library / "lib/deno_dom.dll",
        "QUARTO_DENO": library / "bin/deno.exe",
        "QUARTO_PANDOC": library / "bin/pandoc.exe",
        "QUARTO_ESBUILD": library / "bin/esbuild.exe",
        "QUARTO_DART_SASS": library / "bin/sass.exe",
        "QUARTO_TYPST": library / "bin/typst.exe",
    }
    env = os.environ | {name: "C:\\obsolete\\tool" for name in expected}
    env["CONDA_PREFIX"] = str(prefix)
    script = Path(__file__).resolve().parents[1] / "dev/scripts/pixi_activate.bat"
    result = subprocess.run(
        f'cmd.exe /d /c call "{script}" && set QUARTO_',
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    activated = dict(line.split("=", 1) for line in result.stdout.splitlines())
    for name, path in expected.items():
        assert Path(activated[name]) == path
