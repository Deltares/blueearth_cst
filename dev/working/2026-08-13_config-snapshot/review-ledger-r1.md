# Findings ledger — round 1 → v2

Two independent reviewers, same brief, clean-room. Both returned `revise`.
Reviewer IDs are prefixed here (`gpt-`, `fbl-`) because both used the `ext1-N`
sequence. Every original finding is dispositioned. Nothing was re-graded or
deleted; where the two reviewers found the same defect it is noted, and both
readings are preserved.

Driver premise-verification (done before revision, against the repo):
`gpt-1`, `gpt-5`, `gpt-2`/`fbl-1`, `fbl-7`, `fbl-8` were checked in code and all
**confirmed**. Verification notes are inline.

| ID | Sev | Claim (abridged) | Disposition | Where in v2 |
|---|---|---|---|---|
| `gpt-1` | blocking | v1's `project + shared + own section` scope is under-inclusive for WF3, which reads `workflows.model_creation` | **Accepted.** VERIFIED: `Snakefile_climate_experiment:348` guards `"project", "shared.basin", "workflows.model_creation"`; `:480-495` reads `wflow_outvars` from that section. Replaced section-ownership with an explicit consumed-key projection + mandatory mutation tests | §5.3 |
| `gpt-2` | blocking | The values actually used to build the model are not recorded; R3 unmet. Compounded by the unresolved revision stamp | **Accepted** (owner ruled R6, R7). VERIFIED: `config/defaults/wflow_build_model.yml` carries `setup_rivers.river_upa`, `slope_len`, `setup_soilmaps`, `setup_constant_pars`, consumed verbatim by rule 1.07. Adopted Fable's placement over GPT's | §5.6, §5.2 |
| `gpt-3` | major | R4 not operationalized: `data_sources`/`model_build_config`/`waterbodies_config` can name paths outside the toolbox; §6 said "stop copying" flatly | **Accepted.** Same defect as `fbl-5`. Bin-level ruling replaced by a per-file recoverability predicate | §5.5 |
| `gpt-4` | major | The retained digest covers config mappings but not referenced-input contents; removing the bundle converts a noisy identity into an under-inclusive one | **Accepted.** Split into `effective_config_sha256` (configuration identity, explicitly labelled) and `run_inputs_sha256` (adds toolbox identity + consumed-input hashes) | §5.4, P9 |
| `gpt-5` | major | Observation copies use `source_path.name`; two configured files with the same basename overwrite one another. The bundle avoided this with hash-prefixed names | **Accepted.** VERIFIED: `copy_config_files.py:81` vs `:144`. Role-stable destination names + raise on unexpected collision + origin→archive mapping in `referenced_inputs` | §5.5, P8 |
| `fbl-1` | major | R3's "actual values used" has no named carrier; only `hydromt.log`'s `setup_X.param=value` lines, an unprotected debug log, currently holds them | **Accepted.** Same defect as `gpt-2`; **Fable's fix adopted** — rule 1.07 emits the consumed parameter steps into the model's own directory, which is a values-used record rather than a template snapshot bin | §5.6 |
| `fbl-2` | major | Migration unaddressed: once the inventory drops bundle/template/catalog paths, existing trees (incl. the reference fixture) hold undeclared orphans → `tree-check` red, inventory tests fail | **Accepted.** One-shot cleanup added in the house report-only/`--delete` pattern, run against the fixture before the inventory tests are rewritten | §6 |
| `fbl-3` | major | The revision stamp is wrongly deferred — R4 is inoperable without it, and the design deletes the fallback before installing what makes deletion safe | **Accepted** (owner ruled R7). Folded into `run_record.yml` as `toolbox.{commit,dirty,version}` | §5.2, R7 |
| `fbl-4` | major | "Current" means most recent *attempt*, not the run that produced the outputs; a failed run under `--keep-going` overwrites the record while old outputs remain. Tier B could arbitrate; nothing can afterwards | **Accepted** (owner ruled R5). Journal added; plus `effective_config_sha256` embedded in WF1/WF3 terminal outputs, mirroring WF2, so staleness is detectable without keeping history | §5.7, P7 |
| `fbl-5` | major | "Remove tracked-catalog copies" is a bin-level ruling with no decision procedure; a site-specific catalog outside the repo would be silently dropped | **Accepted.** Same defect as `gpt-3`; **Fable's predicate adopted** (path-prefix test against the toolbox checkout), extended with tracked-and-clean-at-commit | §5.5 |
| `fbl-6` | minor | `schema_version` stays 1 while the document's shape and every derived digest change | **Accepted.** Bumped to 2 | §5.4 |
| `fbl-7` | minor | Cost inventory incomplete: `test_snapshot_config_rules.py`, `README.md`, and newly-dead provenance helpers | **Accepted.** VERIFIED: test asserts the bundle at lines 51/67; README ~174–182 documents it; `short_digest`/`snapshot_bundle_digest` have callers only in the removed code. All three added, with deletion of the dead helpers in the same change | §6 |
| `fbl-8` | minor | v1's "the manifest fingerprints only the two flat copies" is wrong — it fingerprints three, including the experiment's | **Accepted.** VERIFIED: three entries. v1's *conclusion* (no re-record) survives, since all three paths and contents are unchanged; the stated verification was wrong and is corrected | §5.8 |
| `fbl-9` | minor | The retained digest has a reader only in WF2; for WF1/WF3 it reproduces the "one writer, no readers" defect used to justify removing the bundle | **Accepted.** Resolved by the same change as `fbl-4`: WF1 and WF3 embed the digest in a terminal output, giving it a reader and a stated purpose | §5.7 |

## Alternatives proposed in round 1

Both reviewers independently proposed consolidating the record and the revision
stamp into **one artifact** rather than two half-records (GPT: `run-record.yml`;
Fable: `run_stamp.yml`). **Adopted** as `run_record.yml` (§5.2) — snake_case per
`naming.md` §8, which GPT's hyphenated spelling would violate.

Fable's second alternative — an append-only journal, explicitly referred to the
owner rather than assumed — was **put to the owner and allowed** (R5), and is
now §5.7.

GPT's fuller proposal (git blob identities for repo-backed inputs; exact content
records for dirty inputs; atomic replace) is **adopted in part**: blob identity
and atomic replace are in §5.2/§5.5; preserving bytes for a dirty *checkout of
the workflow code* is recorded as open item 1 rather than implemented.

## Not adopted

Nothing was rejected. No finding was downgraded.
