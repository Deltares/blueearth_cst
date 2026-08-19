---
title: Stop test_stage_data.py stubbing geopandas process-globally
type: todo-item
status: backlog
effort: 1
area: test hygiene
queue:
created: 2026-08-19
updated: 2026-08-19
---

> [!note] Overview
> **What** — Scope test_stage_data.py's sys.modules stubs to that module instead of installing fake geopandas / rasterio / xarray for the whole pytest process.
> **Why** — A subset run of two modules reports a false failure: test_stage_cmip6.py's digest test dies with AttributeError instead of skipping, so a gate result has to be argued with rather than believed.
> **Effort** — Small and local: four lines in one test module. The open question
> is which replacement the stubs want — a `pytest` fixture that installs and
> restores them, or importing the real packages, which are already a hard
> dependency of the suite everywhere else.

## Progress

- [ ] Decide whether the stubs are still needed at all — the suite imports real
      geopandas/rasterio/xarray elsewhere, so the four lines may simply go
- [ ] Scope or remove them, keeping `test_stage_data.py` green on its own
- [ ] Re-run the pair together and confirm the digest test skips rather than errors

## Detail

**The defect.** `tests/test_stage_data.py` opens with four module-level lines:

```python
sys.modules.setdefault("geopandas", types.SimpleNamespace())
sys.modules.setdefault("rasterio", types.SimpleNamespace())
sys.modules.setdefault("rasterio.windows", types.SimpleNamespace())
sys.modules.setdefault("xarray", types.SimpleNamespace(Dataset=object))
```

They run at import, mutate `sys.modules` for the whole pytest **process**, and
are never restored. `setdefault` is what makes it order-dependent: it installs
the fake only when the real package has not been imported yet, so whether a
later module sees geopandas or a `SimpleNamespace` depends entirely on what ran
before it.

`series_identity.py:161` imports geopandas lazily, inside
`region_fingerprint`:

```python
import geopandas as gpd
geom = gpd.read_file(region_path)
```

Against the stub that is `AttributeError: 'types.SimpleNamespace' object has no
attribute 'read_file'`.

**Why it is worth fixing rather than watching.** The failure is invisible in the
two runs anyone actually makes and appears in the one used while iterating:

| Invocation | Result |
|---|---|
| `pytest tests/test_stage_cmip6.py` | 11 passed, 1 **skipped** — the honest answer |
| `pytest tests/test_stage_cmip6.py tests/test_stage_data.py` | 35 passed, 1 **failed** |
| `pixi run test-fast` (whole suite) | 2766 passed — some earlier module imports the real geopandas first, so `setdefault` is a no-op |

So the full gate is green, the single module is green, and the two-module subset
a developer runs while working on staging reports a defect that does not exist.
AGENTS.md's own framing applies: a gate result you have to decide whether to
believe is worse than no gate.

**Not caused by anything recent.** Reproduced on pristine `main` with no branch
work applied, by checking out `main` detached in a session slot and running the
pair. Surfaced 2026-08-19 while verifying the console-formatting resync, which
touched neither module's imports.

## Refs

- `tests/test_stage_data.py:9-12` — the four stub lines.
- `blueearth_cst/projections/series_identity.py:161` — the lazy `gpd.read_file`
  that meets the stub.
- `tests/test_stage_cmip6.py::test_the_tool_reproduces_a_digest_wf2_itself_wrote`
  — the test that reports the false failure.
