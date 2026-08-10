# Archived data catalogs

hydromt data catalogs (`-d` targets) whose only consumers were the archived
single-workflow configs under `config/templates/archive/`. Parked here on
2026-08-10, alongside those configs, so the two halves stay together.

| File | Consumers |
| --- | --- |
| `deltares_data_climate_projections.yml` | `../../templates/archive/snake_config_projections_cmip5_full.yml`, `…_isimip3.yml` |
| `deltares_data_climate_projections_linux.yml` | the `_linux` siblings of both |

The archived configs were rewired to these paths at the same commit, so that
pairing still resolves. Nothing else in the repository reads either file.

**The live catalogs stay in the parent directory** and are unaffected:
`deltares_data.yml`, `deltares_data_linux.yml`, `cmip6_data.yml`, and the
generated `cmip6_store_index.json`.

## One thing not to change

`blueearth_cst/experiment/check_project_consistency.py` and
`dev/scripts/semantic_tree_diff.py` both carry a pre-R6 → post-R6 path map that
names these files at their *former* `config/catalogs/` location. That is
deliberately not updated to point here: the map records where the R6 migration
put them, which is a historical fact, and the only case it fires on is a
hand-migrated pre-R6 flat config — which by definition cannot reference an
`archive/` path that did not exist then. Repointing it would misstate the
migration and desynchronize the two copies of the map.
