---
title: pixi install reports success without repairing a missing console script
type: watch-item
area: environment
origin: 2026-08-12 test-suite session
created: 2026-08-12
updated: 2026-08-12
---

> [!note] Overview
> **What** — `pixi install` trusts its own metadata over the filesystem, so it neither detects nor repairs a console script that is missing from `.pixi/envs/default/Scripts/` while the package itself is present and importable.
> **Why** — The failure is narrow and misleading: only code that SHELLS OUT to the executable breaks, while everything importing the module passes, so a suite fails in one layer with no obvious cause. `pixi list` and a clean `pixi install` both report healthy. Cost 50 minutes on 2026-08-12, most of it spent waiting on a wrong diagnosis (a concurrent env rebuild) rather than investigating.
> **Trigger** — A pixi-provided executable is missing while `pixi list` reports its package installed — or pixi gains a verify/repair verb that makes the workaround unnecessary.

## Detail

Observed 2026-08-12 on `fix/improvements`. **Exactly one file** was missing —
`.pixi/envs/default/Scripts/snakemake.exe` — and everything else about the
install looked healthy:

- `python -c "import snakemake"` worked, version 9.6.2;
- `pixi list` reported `snakemake 9.6.2 ... pypi`;
- the package's own `RECORD` **claimed the file**, with a sha256:
  `../../Scripts/snakemake.exe,sha256=va069z...,46080`.

So the install was internally inconsistent, and only `tests/test_cli.py` felt
it — those tests shell out to `snakemake` as a *command*, while the 1,998 tests
that merely import the module all passed. A suite that fails in one narrow layer
for no visible reason is the signature.

**`pixi install` does not repair this.** Run against the damaged env it printed
`✔ The default environment has been installed.` and changed nothing: it
reconciles against its own metadata, which already said the package was there.

### Diagnosing it

Compare what every installed distribution CLAIMS in `Scripts/` against what is on
disk. This found 2 of 33 entries missing, one of them the real defect (the other,
`numba`'s extension-less POSIX name, is a Windows packaging artifact —
`numba.exe` and `numba-script.py` both exist):

```python
import pathlib
env = pathlib.Path(".pixi/envs/default"); scripts = env / "Scripts"
for record in (env / "Lib" / "site-packages").glob("*.dist-info/RECORD"):
    for line in record.read_text(encoding="utf-8", errors="replace").splitlines():
        p = line.split(",")[0].replace("\\", "/")
        if "Scripts/" in p and not (scripts / p.split("Scripts/")[-1]).exists():
            print(record.parent.name, "->", p)
```

### Repairing it

Make pixi see the package as absent, then let pixi reinstall it. This stays
inside pixi's model — no `pip` reaching into the env behind its back, and
`pixi.lock` / `pixi.toml` are untouched, so CI's `locked: true` still holds:

```bash
rm -rf .pixi/envs/default/Lib/site-packages/<package>-<version>.dist-info
pixi install
```

`pixi clean` + `pixi install` would also work and is more obviously correct, at
the cost of a full env rebuild.

### The 50 minutes

Not spent on the repair, which took two commands. Spent on a **wrong diagnosis
held too long**: a stale `pixi.exe` in the process list was read as "a concurrent
session is rebuilding the env", and that story was consistent enough with the
evidence to justify waiting rather than investigating. The tell that should have
broken it earlier: an env being rebuilt does not keep 305 other scripts and 1,998
passing imports intact while losing exactly one executable.
