# R06 — structural refactor — internal rename record

**Reconstructed 2026-07-29 during R07.** `dev/reference/naming.md` §7 requires
a `dev/<milestone>/migration_<topic>.md` rename record for every milestone that
renames a contract surface. R06 renamed four classes of them but wrote its note
as a root-level `MIGRATION.md` instead — the ambiguity R07's §7 amendment
resolves. That file is now `docs/migration-r06.md`; this is the internal record
§7 asks for, and it is the authority for **which classes** were renamed. The
exhaustive per-file tables stay in the moved document rather than being
duplicated here.

**Anchor.** R06 landed at git ref `e33ee45`. Every mapping below is relative to
that boundary.

## Renamed contract surfaces, by §7 class

| §7 class | R06 change | Detail |
| --- | --- | --- |
| Test fixture paths read by `check_baseline.py` / `conftest.py` | Python package root `src/` → `blueearth_cst/` | `docs/migration-r06.md` §1 — the full per-module table, plus the import-prefix cheat-sheet at §"Import-prefix migration cheat-sheet" |
| Checked-in example config keys | `config/` flattened → `config/{workflows,catalogs,templates}/` | `docs/migration-r06.md` §2, including §"Config path values rewritten inside configs" — values *inside* configs changed, not just filenames |
| Scripts invoked by users | runners moved from the repo root → `scripts/` | `docs/migration-r06.md` §3 |
| Test fixture / dev-script path literals | rewritten **in place**, not moved | `docs/migration-r06.md` §4 |

`rule all` output filenames were **not** renamed by R06 — the baseline manifest
contract was preserved, which is why R06 needed no re-record. (R07 is the
milestone that moves those; its record is
`dev/milestones/r07/migration_project-layout.md`.)

## Machinery kept in lockstep

`dev/scripts/semantic_tree_diff.py`'s `COPIED_CONFIG_PATH_MAP` carries the
config-path old→new values so a copied snapshot from before R06 normalizes
rather than failing the tree diff. That table and `docs/migration-r06.md` §2 are
maintained together; R07 extended the same map for its own moves
(`dev/milestones/r07/migration_project-layout.md` §2d).

## Corrections applied since

- **O-10 (R07).** The subpackage `__init__.py` list in
  `docs/migration-r06.md` §1 omitted `blueearth_cst/climate_analysis/__init__.py`.
  Corrected; `blueearth_cst/weathergen/` correctly has no marker, holding R
  sources rather than an importable Python subpackage.
- The `docs/config/` mirror that §2 referenced was deleted in R07 (O-05);
  `config/` is now the single source.
