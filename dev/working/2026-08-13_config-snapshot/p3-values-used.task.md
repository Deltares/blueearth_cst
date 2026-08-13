# Task Brief — P3: values-used records

### Context

`AGENTS.md`; design `design-v3.md` §5.6; evidence in `driver-verification.md`.

- **This is the phase the design exists for.** R3 asks for "the actual values
  used", and the template is *not* it: `build_wflow_model.py:237-268` pops
  configured arguments and derives others — notably
  `kwargs.setdefault("lulc_mapping_fn", f"{source_name}_mapping_default")`,
  which appears in no file on disk.
- The decisive case is in that function's own comment: until 2026-08-13 the
  mapping source came from the template's `lulc_fn`, so
  `spatial_sources.lulc: corine` ran CORINE through `vito_mapping_default` —
  *"Wrong numbers, not a missing setting."* A record serialising the template
  would not have caught it. This is the acceptance test.

### Goal

Emit, from the rules that consume them, the **post-normalization** values
hydromt was actually handed.

### Non-goals

No Snakefile output declarations here (P4 wires them) beyond what rules 1.07 and
1.08 need to run. No changes to what the build computes.

### Allowed scope

- **Permitted:** `blueearth_cst/model/build_wflow_model.py`,
  `blueearth_cst/model/setup_reservoirs_lakes_glaciers.py`,
  `tests/test_build_wflow_model.py`, and the 1.07/1.08 output blocks.
- **Forbidden:** any behavioural change to the model build itself.

### Required changes (checklist)

1. Refactor `_apply_parameter_steps` so each step's final `call_kwargs` is
   built **once** and used **twice** — for the hydromt call and for the record.
   A second construction path would let the two drift, which is the defect.
2. Record injected P1 datasets as `{injected_from, product}` references, never
   `repr()` of an xarray object.
3. Rule 1.07 emits `models/hydrology/wflow/hydromt_build_config.yml`.
4. Rule 1.08 emits `models/hydrology/wflow/hydromt_update_waterbodies.yml`,
   including per-method ok/skipped status.

### Validation

- Rung 1: `pytest tests/test_build_wflow_model.py`.
- Rung 2 (new behavioural tests):
  - **the decisive one** — the emitted record's `setup_lulcmaps.lulc_mapping_fn`
    equals the **derived** value, not the template's `lulc_fn`;
  - `setup_rivers` record shows the injected P1 hydrography reference, not the
    template's `hydrography_fn`;
  - the waterbodies record distinguishes a method that ran from one skipped.
- Rung 3: `pytest tests/test_cli.py` — rules gained declared outputs.
- Rung 4: `pixi run test-fast`.

**Falsifier for "the record equals what hydromt received":** construct a config
whose template `lulc_fn` and derived `lulc_mapping_fn` disagree; a record
showing the template value disproves the property. Build this test *before* the
refactor.

### Acceptance criteria

Both records emitted as declared outputs; the derived-`lulc_mapping_fn` test
passes; no numerical change to the built model.

### Task constraints

Stay within CST's automation scope (`AGENTS.md`): record what is passed to
hydromt; do not re-engineer how hydromt's `setup_*` methods work.
