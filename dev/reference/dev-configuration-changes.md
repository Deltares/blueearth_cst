# Configuration changes

Last updated: 2026-11-14

Use this procedure when adding, moving, renaming or retiring a configuration key, including
`config/advanced_settings.yml` keys. The
[configuration reference](../../docs/technical-manual/configuration.md) explains composition, paths and
settings to users; the modules named below own exact schemas and defaults.
Read the [naming conventions](dev-naming-conventions.md) before choosing a key name.

## Choose the owning file

- **Project file:** `project`, the shared `basin`, `climate`, `model` and `plotting` sections, and one
  closed `{enabled, config_path}` stanza per workflow. A key read by more than one workflow belongs in
  one of the shared sections, never in a workflow file.
- **Workflow file:** settings read by that workflow alone, reached through its stanza's `config_path`.
- **`config/advanced_settings.yml`:** toolbox-wide `constraints`, `defaults` and `runtime` values read for
  every project by `snake_utils`. It is never a `--configfile` target.
- **`config/defaults/`:** recipes read during execution, so a change there changes runs.
  `config/templates/` holds scaffolds only. The [repository layout](dev-repository-layout.md#config) owns
  placement of these directories.

## Register the key in every closed list

Unknown keys are refused rather than ignored, so a new key fails until each applicable owner admits it:

- **Project-file sections and leaves:** `T1_TOP_LEVEL`, `T1_SHARED_SECTIONS` and `WORKFLOW_STANZA_KEYS` in
  [`config_composition.py`][composition].
- **Keys misplaced in the other workflow's file:** `GENERATION_KEYS` and `SIMULATION_KEYS` in the same
  module.
- **Advanced settings:** `_ADVANCED_SETTINGS_SCHEMA` in [`advanced_settings.py`][advanced], kept equal to
  the shipped YAML by [`test_advanced_settings.py`](../../tests/test_advanced_settings.py).
- **WF4 and WF1 fields captured for a reservation:** the unclassified-field checks for `simulate_system`,
  `compute`, `build_model` and `engine` in `capture_request`
  ([`reservation_request.py`][reservation-request]).
- **weathergenr configuration fields:** `GENERATOR_SEED_FIELDS` in [`generation_plan.py`][generation],
  which also declares whether each field enters the seed.
- **The `delta_change` block:** the authored `delta-change/3` keys in [`delta_scenarios.py`][delta].
- **File paths:** the field lists in `resolve_project_paths` and `resolve_workflow_paths`
  ([`config_paths.py`][paths]).

Composed basin fields are parsed by [`spatial/config.py`](../../blueearth_cst/spatial/config.py). Snakefiles
read values with `get_config(config, key, default, optional)` from `snake_utils.py`: a required key raises
when missing; an optional key returns its default. In schema 3, a path field resolves beside the YAML
that declares it and `toolbox:` names a bundled resource; schema 2 keeps invocation-relative paths. A path
key that is not listed in `config_paths.py` is not resolved.

## Preserve existing identities

Classify each new key as scientific or not. A scientific value must reach the identity of the product it
changes; a non-scientific key, such as a display or compute setting, must stay out of it. In both cases
**omitting the new key must leave existing identities unchanged**: write a default into the resolved
value only where the identity already records it, and add an optional request field only when it differs
from the default, as `capture_request` does for `include_unperturbed`, `initialization` and
`historical_forcing`. Composition never adds or drops keys (invariant 2 in `config_composition.py`).
The [run identity reference](toolbox-run-identity.md) lists what each identity hashes and how to check
whether a module edit changes it.

## Keep the starter and seeds current

Update [`config/templates/example-basin/`](../../config/templates/example-basin) in the same change as the
key, live or as a commented example; `scripts/create_case.py` scaffolds new cases from it
([`test_create_case.py`](../../tests/test_create_case.py)). Update affected seeds under `test_case/` and the
test fixtures `tests/project_config_fixture*.yml`. Do not add a second starter elsewhere.

Run the unread-key sweep:

```powershell
python dev/scripts/sweep_unread_config_keys.py
```

[`sweep_unread_config_keys.py`](../../dev/scripts/sweep_unread_config_keys.py) reports a template key that
no reader reads and a seed key that no template documents. It fails closed: an empty reader or declared
side is reported as a failure, not a clean result
([`test_unread_config_key_sweep.py`](../../tests/test_unread_config_key_sweep.py)).

## Rename or retire a key

A renamed or retired user-facing key needs a migration note with the old-to-new mapping, affected
consumers and checks, following
[naming conventions §7](dev-naming-conventions.md#7-rename-only-with-a-migration-note).
Add the old path to `RETIRED_KEYS` in `config_composition.py` with the sentence that tells a user where it
went; the loader refuses it by name. Update every live reference, including templates, seeds, docs and
comments. Write a user-facing guide only when users must act. Version-1 project files are refused with a
pointer to the `v0.3.0` release, which still ships the migrator.

## Verification

Composition refusals are raised while the Snakefile parses, so `python -m pytest tests/test_cli.py` covers
them. Also run the affected owners' tests:
[`test_config_composition.py`](../../tests/test_config_composition.py),
[`test_v2_config_shape.py`](../../tests/test_v2_config_shape.py),
[`test_config_paths.py`](../../tests/test_config_paths.py),
[`test_advanced_settings.py`](../../tests/test_advanced_settings.py) and
[`test_reservation_request.py`](../../tests/test_reservation_request.py). Dry-run an affected workflow through
its supported launcher, and use the [validation ladder](dev-validation.md) for anything wider.

[composition]: ../../blueearth_cst/shared/config_composition.py
[advanced]: ../../blueearth_cst/shared/advanced_settings.py
[reservation-request]: ../../blueearth_cst/experiment/reservation_request.py
[generation]: ../../blueearth_cst/experiment/generation_plan.py
[delta]: ../../blueearth_cst/experiment/delta_scenarios.py
[paths]: ../../blueearth_cst/shared/config_paths.py
