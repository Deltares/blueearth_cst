# dev/reference/

Rules the code must obey, and the durable descriptions of how it is put
together. Consulted while working; rewritten rarely and deliberately.

This is the **stays-true** tier. What happened lives in `../milestones/`,
`../decisions/`, and `../tasks/`; what is happening lives in `../TODO.md` and
`../working/`; snapshots that decay live in `../reviews/`.

| Path | Holds |
|---|---|
| `naming.md` | Prescriptive style guide for identifiers and files, with `MUST` / `SHOULD` / `MAY` normative force |
| `agent-activation.md` | How roles and skills become available to Claude Code and Codex here, and why the two runtimes differ |
| `branches-and-tags.md` | Inventory of durable refs and what each is for; transient branches excluded |
| `contracts/` | The two substitution seams — hydrological model, weather generator — pinned as machine-checked contracts (P3-2b) |
| `workflows/` | Per-workflow contract docs for wf1 / wf2 / wf3, plus the WF2 v2.0 design and CMIP6 member inventories |

## Two things to know before editing

- **These paths are cited from shipped code.** `workflows/` is referenced from
  module docstrings and `config/workflows/snake_config.template.yml`;
  `contracts/` from `blueearth_cst/shared/interchange_contracts.py` and its
  test; `naming.md` from `AGENTS.md`. Renaming a file here means updating those
  citations in the same commit.
- **`workflows/` is not `config/workflows/` or `.github/workflows/`.** Three
  different directories share the word. This one holds prose contracts; the
  other two hold Snakemake configs and CI definitions.

Everything here moved from the `dev/` root on 2026-08-02 — a path change only,
no file renamed, split, or edited beyond the prefix. `conventions/` was
flattened into this folder (it held two files); `contracts/` and `workflows/`
kept their folders because each is a coherent multi-file unit.
